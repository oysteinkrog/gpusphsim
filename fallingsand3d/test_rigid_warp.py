"""Regression tests for rigid body force/torque accumulation.

bd-r4-epic-x2j.1: the per-lane accumulation helper (now atomic_accumulate3,
bd-r5-review-n28.182) must not blend lanes of different bodies.
bd-r5-review-n28.182: K_Step2 sums a particle's reaction on its first body
in registers; per-body force and torque must match the per-pair build.

The old code used a full-warp __shfl_down_sync reduction then let only lane 0
atomicAdd. When called from a divergent neighbor-loop branch where different
lanes own different body_ids, this blends forces across bodies (force bleed).

The fix replaces that with a direct per-lane atomicAdd so each lane writes only
its own contribution to its own body.

Test strategy:
  - Compile a tiny harness kernel (no external headers needed) that reproduces
    BOTH the buggy and the fixed accumulation logic.
  - Launch with a single warp (32 threads), assigning the first 16 threads to
    body 0 and the next 16 threads to body 1, each contributing a known
    float3 value.
  - Assert that each body accumulates only its own 16-thread sum.
    Under the old code, lane 0 of the warp would atomicAdd the full 32-thread
    shuffle-reduced sum into body 0, causing both bodies' forces to bleed.
"""

from __future__ import annotations

import sys
import os

# Ensure conftest NVRTC path fixup runs (mirrors how pytest loads it).
sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: F401  -- side-effects only

import cupy as cp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Inline CUDA kernels: buggy (old) and fixed (new)
# ---------------------------------------------------------------------------

# Each thread contributes val = (lane+1, lane+2, lane+3) to its body.
# Threads 0..15  -> body 0
# Threads 16..31 -> body 1
#
# Expected body 0 sum: sum_{lane=0}^{15}  (lane+1) = 136, etc.
# Expected body 1 sum: sum_{lane=16}^{31} (lane+1) = 392, etc.
#
# Under the BUGGY code:
#   - The full-warp shuffle reduces ALL 32 threads together.
#   - Only lane 0 writes, using body_id=0 -> body 0 gets the 32-thread total.
#   - Body 1 gets nothing (its lane 0 is lane 16, which is NOT threadIdx.x==0).
#   This fails both bodies.
#
# Under the FIXED code:
#   - Every lane atomicAdds its own val to its own body.
#   - Body 0 gets exactly the sum of lanes 0..15.
#   - Body 1 gets exactly the sum of lanes 16..31.

_BUGGY_KERNEL_SRC = r"""
extern "C" __global__ void test_buggy_warp_reduce(float* out) {
    // out layout: [body0.x, body0.y, body0.z, pad, body1.x, body1.y, body1.z, pad]
    int lane = threadIdx.x & 31;
    int body_id = (lane < 16) ? 0 : 1;

    float3 val;
    val.x = (float)(lane + 1);
    val.y = (float)(lane + 2);
    val.z = (float)(lane + 3);

    // --- BUGGY: full-warp __shfl_down_sync then lane-0-only atomicAdd ---
    for (int offset = 16; offset > 0; offset >>= 1) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
        val.z += __shfl_down_sync(0xffffffff, val.z, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&out[body_id * 4 + 0], val.x);
        atomicAdd(&out[body_id * 4 + 1], val.y);
        atomicAdd(&out[body_id * 4 + 2], val.z);
    }
}
"""

_FIXED_KERNEL_SRC = r"""
extern "C" __global__ void test_fixed_warp_reduce(float* out) {
    // out layout: [body0.x, body0.y, body0.z, pad, body1.x, body1.y, body1.z, pad]
    int lane = threadIdx.x & 31;
    int body_id = (lane < 16) ? 0 : 1;

    float3 val;
    val.x = (float)(lane + 1);
    val.y = (float)(lane + 2);
    val.z = (float)(lane + 3);

    // --- FIXED: direct per-lane atomicAdd ---
    atomicAdd(&out[body_id * 4 + 0], val.x);
    atomicAdd(&out[body_id * 4 + 1], val.y);
    atomicAdd(&out[body_id * 4 + 2], val.z);
}
"""


def _ptx_fixup() -> None:
    """Make NVRTC emit PTX when the GPU is newer than NVRTC supports."""
    try:
        from cupy.cuda import compiler as _compiler
        from cupy.cuda import device as _device
        gpu_cc = _device.Device().compute_capability
        nvrtc_max = _compiler._get_max_compute_capability()
        if int(gpu_cc) > int(nvrtc_max):
            _compiler._use_ptx = True
            if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
                _compiler._get_arch_for_options_for_nvrtc._cache = {}
            if hasattr(_compiler._get_arch, "_cache"):
                _compiler._get_arch._cache = {}
    except Exception:
        pass


def _compile_module(src: str, name: str) -> cp.RawModule:
    """Compile a CuPy RawModule, applying PTX workaround if needed."""
    _ptx_fixup()
    return cp.RawModule(code=src, options=("--std=c++11",), name_expressions=None)


# ---------------------------------------------------------------------------
# Expected values
# ---------------------------------------------------------------------------

def _expected_sum(lane_start: int, lane_end: int) -> tuple[float, float, float]:
    """Sum of (lane+1, lane+2, lane+3) for lanes in [lane_start, lane_end)."""
    xs = sum(l + 1 for l in range(lane_start, lane_end))
    ys = sum(l + 2 for l in range(lane_start, lane_end))
    zs = sum(l + 3 for l in range(lane_start, lane_end))
    return float(xs), float(ys), float(zs)


# Body 0: lanes 0..15
EXP_B0 = _expected_sum(0, 16)   # (136, 152, 168)
# Body 1: lanes 16..31
EXP_B1 = _expected_sum(16, 32)  # (392, 408, 424)

# Under the buggy code the full 32-thread shuffle-reduced value (all lanes)
# ends up in body 0 from lane 0. Body 1 gets nothing because lane 16 is not
# (threadIdx.x & 31) == 0.
BUGGY_FULL = _expected_sum(0, 32)  # (528, 560, 592)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_buggy_kernel_bleeds_force():
    """The old shuffle-based code must produce incorrect per-body sums.

    This test is expected to PASS (i.e. confirm the bug exists in the old code).
    It asserts that body 0 gets the full 32-thread total (force bleed) and
    body 1 gets nothing -- both wrong.
    """
    mod = _compile_module(_BUGGY_KERNEL_SRC, "buggy")
    kernel = mod.get_function("test_buggy_warp_reduce")

    out = cp.zeros(8, dtype=cp.float32)
    kernel((1,), (32,), (out,))
    cp.cuda.Device().synchronize()
    result = out.get()

    b0 = (result[0], result[1], result[2])
    b1 = (result[4], result[5], result[6])

    # Body 0 should have the FULL 32-thread total (the bug: force bleed from body 1).
    assert b0 == pytest.approx(BUGGY_FULL, rel=1e-5), (
        f"Expected buggy body0={BUGGY_FULL}, got {b0}"
    )
    # Body 1 should be zero (its contribution was silently stolen by body 0 / dropped).
    assert b1 == pytest.approx((0.0, 0.0, 0.0), abs=1e-5), (
        f"Expected buggy body1=(0,0,0), got {b1}"
    )


def test_fixed_kernel_correct_per_body():
    """The fixed per-lane atomicAdd must give correct per-body sums with no bleed."""
    mod = _compile_module(_FIXED_KERNEL_SRC, "fixed")
    kernel = mod.get_function("test_fixed_warp_reduce")

    out = cp.zeros(8, dtype=cp.float32)
    kernel((1,), (32,), (out,))
    cp.cuda.Device().synchronize()
    result = out.get()

    b0 = (result[0], result[1], result[2])
    b1 = (result[4], result[5], result[6])

    assert b0 == pytest.approx(EXP_B0, rel=1e-5), (
        f"Expected fixed body0={EXP_B0}, got {b0}"
    )
    assert b1 == pytest.approx(EXP_B1, rel=1e-5), (
        f"Expected fixed body1={EXP_B1}, got {b1}"
    )


def _compile_shared(src: str) -> cp.RawModule:
    """Compile a harness that includes physics/kernels/sph_shared.cuh."""
    _ptx_fixup()
    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    return cp.RawModule(
        code=src,
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )


def test_atomic_accumulate3_per_lane():
    """atomic_accumulate3 from sph_shared.cuh adds each lane to its own body."""
    src = r"""
#include "sph_shared.cuh"

extern "C" __global__ void test_aa3(float* out) {
    int lane = threadIdx.x & 31;
    int body_id = (lane < 16) ? 0 : 1;
    float3 val = make_float3((float)(lane+1), (float)(lane+2), (float)(lane+3));
    atomic_accumulate3(out, val, body_id);
}
"""
    kernel = _compile_shared(src).get_function("test_aa3")

    out = cp.zeros(8, dtype=cp.float32)
    kernel((1,), (32,), (out,))
    result = out.get()

    b0 = (result[0], result[1], result[2])
    b1 = (result[4], result[5], result[6])
    assert b0 == pytest.approx(EXP_B0, rel=1e-5)
    assert b1 == pytest.approx(EXP_B1, rel=1e-5)


# ---------------------------------------------------------------------------
# K_Step2 per-particle rigid reaction (bd-r5-review-n28.182)
# ---------------------------------------------------------------------------
#
# K_Step2 sums a fluid particle's pressure reaction on the first body it
# touches in registers and writes it once after the loop, with the torque
# x_i x F. Pairs on any other body keep per-pair atomics. The reference
# build (-DSTEP2_RIGID_PER_PAIR_REFERENCE) writes every pair at once, with
# the torque x_j x F. Both are about the world origin (bd-r5-review-n28.195).
#
# Scene: two kinematic 12 cm boxes 4 cm apart in a water block, so the fluid
# in the gap touches both bodies. Each K_Step2 launch in the substep is also
# run with the reference module on the same inputs; the per-body force and
# torque must agree, and the fluid-side output must match within float noise.

HALF = 0.06
GAP = 0.04
H = 0.04


def _two_body_sim():
    from materials import WATER
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    xc = HALF + GAP / 2.0
    world = World(120_000)
    ex = HALF + 0.005
    world.spawn_cube((-0.3, -0.2, -0.2), (0.3, 0.2, 0.2), WATER, 0.02,
                     exclude_box=((-xc - ex, -ex, -ex), (xc + ex, ex, ex)))
    # Fluid in the gap between the two boxes.
    world.spawn_cube((-0.01, -HALF, -HALF), (0.01, HALF, HALF), WATER, 0.02)
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=1.0)
    sim.set_solver_profile(PROFILES["WCSPH"])
    sim._spawn_substep = sim._damping_duration
    rbm = sim.rigid_body_manager
    for x in (-xc, xc):
        rbm.add_rigid_body_with_particles(
            position=(x, 0.0, 0.0), half_extents=(HALF, HALF, HALF),
            mass=4.32, is_kinematic=True)
    # Spin body 1 so its viscous reaction and torque are not trivial.
    rbm._bodies_cpu[1]["ang_vel"][:3] = (0.0, 3.0, 1.0)
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    return world, sim, rbm


def _copy_constants(dst_mod, src_mod) -> None:
    """Copy K_Step2's constant memory from src_mod to dst_mod."""
    import step1
    import step2
    from materials import build_material_array

    sizes = {
        "c_grid": 32,
        "c_sim": step1.build_sim_params(smoothing_length=H).nbytes,
        "c_precalc": step1.build_precalc_params(smoothing_length=H, viscosity=1.0).nbytes,
        "c_materials": build_material_array().nbytes,
        "c_granular": step2.build_granular_params().nbytes,
    }
    for name, size in sizes.items():
        cp.cuda.runtime.memcpy(int(dst_mod.get_global(name)),
                               int(src_mod.get_global(name)), size, 3)


def test_step2_rigid_reaction_matches_per_pair_reference(monkeypatch):
    import kernel_loader
    import step2

    world, sim, rbm = _two_body_sim()
    # The wrapper below reads results back to the host, which a CUDA graph
    # capture does not allow; run every substep directly.
    def direct(n, gather_only):
        sim._run_graph_body(n)
        return None
    monkeypatch.setattr(sim, "_capture_graph", direct)
    sim._sim_step(world._high_water)  # inject the boundary block
    assert rbm.boundary_injected

    # Some fluid particles must sit in the gap, within h of both boxes.
    n = world._high_water
    pos = world.position[:n, :3].get()
    mat = world.packed_info[:n].get() & 0xFF
    from materials import WATER
    gap = (mat == WATER) & (np.abs(pos[:, 0]) < GAP / 2.0) \
        & (np.abs(pos[:, 1]) < HALF) & (np.abs(pos[:, 2]) < HALF)
    assert gap.sum() > 20, gap.sum()

    ref_mod = kernel_loader.load_module("step2.cu", ("-DSTEP2_RIGID_PER_PAIR_REFERENCE",))
    new_mod = step2._get_module()
    real = step2.compute_step2
    got = []

    def both(*args, **kw):
        F, T = kw["d_rigid_forces"], kw["d_rigid_torques"]
        F0, T0 = F.copy(), T.copy()
        # K_Step2 writes nothing for STATIC and sleeping particles, so the
        # reference output starts from the same buffer contents.
        out0 = kw["sph_force_out"].copy()
        real(*args, **kw)
        dF, dT = (F - F0).get(), (T - T0).get()
        out_new = kw["sph_force_out"].copy()
        # Same inputs through the reference build, into separate accumulators.
        _copy_constants(ref_mod, new_mod)
        rF, rT = cp.zeros_like(F), cp.zeros_like(T)
        ref_out = out0
        kw_ref = dict(kw, d_rigid_forces=rF, d_rigid_torques=rT,
                      sph_force_out=ref_out, veleval_out=cp.zeros_like(kw["veleval_out"]))
        monkeypatch.setattr(step2, "_get_module", lambda: ref_mod)
        try:
            real(*args, **kw_ref)
        finally:
            monkeypatch.setattr(step2, "_get_module", lambda: new_mod)
        # The two builds may contract the fluid-side math differently, so
        # compare within float noise rather than bitwise.
        diff = float(cp.abs(out_new - ref_out).max())
        scale = float(cp.abs(ref_out).max())
        got.append((dF, dT, rF.get(), rT.get(), diff <= 1e-5 * scale))

    monkeypatch.setattr(step2, "compute_step2", both)
    for _ in range(5):
        sim._sim_step(world._high_water)
    cp.cuda.Device().synchronize()
    assert len(got) == 5

    for dF, dT, rF, rT, fluid_same in got:
        assert fluid_same, "fluid-side K_Step2 output differs from the reference"
        for b in range(2):
            f_ref, t_ref = rF[b, :3], rT[b, :3]
            assert np.linalg.norm(f_ref) > 1e-3, f_ref
            assert np.linalg.norm(t_ref) > 1e-6, t_ref
            np.testing.assert_allclose(dF[b, :3], f_ref, rtol=0,
                                       atol=1e-4 * np.linalg.norm(f_ref))
            np.testing.assert_allclose(dT[b, :3], t_ref, rtol=0,
                                       atol=1e-4 * np.linalg.norm(t_ref) + 1e-7)
        assert np.all(dF[2:] == 0.0) and np.all(rF[2:] == 0.0)
