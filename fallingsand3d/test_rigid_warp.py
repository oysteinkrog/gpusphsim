"""Regression tests for rigid body force/torque accumulation.

bd-r4-epic-x2j.1: the per-lane accumulation helper (now atomic_accumulate3,
bd-r5-review-n28.182) must not blend lanes of different bodies.
bd-r5-review-n28.182: RigidAccum sums a thread's pairs in registers and
flushes on body change, and must give the same per-body sums.

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


# Harness for RigidAccum (bd-r5-review-n28.182). Thread t walks its row of
# `pairs` (body id, -1 = no rigid neighbour) the way a coupling loop walks its
# neighbours, adding F = vals[t][p] and tau = 2 * vals[t][p] per pair, then
# flushes once after the loop.
_ACCUM_SRC = r"""
#include "sph_shared.cuh"

extern "C" __global__ void test_rigid_accum(
    const int* pairs, const float* vals, int n_threads, int n_pairs,
    float* forces, float* torques
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_threads) return;
    RigidAccum acc = rigid_accum_init();
    for (int p = 0; p < n_pairs; p++) {
        int body = pairs[t * n_pairs + p];
        if (body < 0) continue;
        const float* v = vals + 3 * (t * n_pairs + p);
        float3 F = make_float3(v[0], v[1], v[2]);
        float3 tau = make_float3(2.0f * v[0], 2.0f * v[1], 2.0f * v[2]);
        rigid_accum_add(acc, forces, torques, body, F, tau);
    }
    rigid_accum_flush(acc, forces, torques);
}
"""


def _run_accum(pairs: np.ndarray, vals: np.ndarray, n_bodies: int):
    kernel = _compile_shared(_ACCUM_SRC).get_function("test_rigid_accum")
    n_threads, n_pairs = pairs.shape
    d_pairs = cp.asarray(pairs.astype(np.int32))
    d_vals = cp.asarray(vals.astype(np.float32))
    forces = cp.zeros((n_bodies, 4), dtype=cp.float32)
    torques = cp.zeros((n_bodies, 4), dtype=cp.float32)
    block = 128
    grid = (n_threads + block - 1) // block
    kernel((grid,), (block,), (d_pairs, d_vals, np.int32(n_threads),
                               np.int32(n_pairs), forces, torques))
    return forces.get(), torques.get()


def _naive_sums(pairs: np.ndarray, vals: np.ndarray, n_bodies: int):
    """Per-pair sums in float64, the result the old per-pair atomics gave."""
    F = np.zeros((n_bodies, 4))
    for t in range(pairs.shape[0]):
        for p in range(pairs.shape[1]):
            b = pairs[t, p]
            if b >= 0:
                F[b, :3] += vals[t, p].astype(np.float64)
    return F, 2.0 * F


def test_rigid_accum_two_particles_two_bodies():
    """Flush on switch sends each pair to its own body.

    Particle 0 touches body 0, then body 1, then body 0 again (two switches).
    Particle 1 touches body 1 twice with a non-rigid neighbour between.
    """
    pairs = np.array([[0, 1, 0],
                      [1, -1, 1]])
    vals = np.array([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
                     [[1000.0, 0.0, -1.0], [9e9, 9e9, 9e9], [0.5, 0.25, 0.125]]])
    forces, torques = _run_accum(pairs, vals, 2)

    np.testing.assert_allclose(forces[0, :3], [101.0, 202.0, 303.0], rtol=1e-6)
    np.testing.assert_allclose(forces[1, :3], [1010.5, 20.25, 29.125], rtol=1e-6)
    np.testing.assert_allclose(torques, 2.0 * forces, rtol=1e-6)
    assert np.all(forces[:, 3] == 0.0) and np.all(torques[:, 3] == 0.0)


def test_rigid_accum_matches_naive_many_threads():
    """Random neighbour lists over 8 bodies match the per-pair sum within 1e-5."""
    rng = np.random.default_rng(182)
    n_threads, n_pairs, n_bodies = 4096, 24, 8
    # Mostly one body per particle, like a fluid shell around a body, with
    # some non-rigid neighbours and some particles between two bodies.
    home = rng.integers(0, n_bodies, n_threads)
    pairs = np.repeat(home[:, None], n_pairs, axis=1)
    pairs[rng.random(pairs.shape) < 0.3] = -1
    mixed = rng.random(pairs.shape) < 0.1
    pairs[mixed] = rng.integers(0, n_bodies, int(mixed.sum()))
    vals = rng.uniform(-1.0, 1.0, (n_threads, n_pairs, 3))

    forces, torques = _run_accum(pairs, vals, n_bodies)
    F_ref, T_ref = _naive_sums(pairs, vals.astype(np.float32), n_bodies)

    scale = np.abs(vals).sum() / n_bodies
    np.testing.assert_allclose(forces, F_ref, rtol=0, atol=1e-5 * scale)
    np.testing.assert_allclose(torques, T_ref, rtol=0, atol=2e-5 * scale)
