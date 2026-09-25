"""max_displacement is reduced per warp before the atomicMax (bd-r5-review-n28.178).

K_Integrate, the rigid boundary update, K_PBF_Finalize and K_DFSPH_Finalize
call warp_atomic_max_uint instead of one atomicMax per thread. These tests
check that the helper stores the same maximum as the per-thread atomic, with
early-returning and divergent lanes, and that each solver's max_displacement
still equals the largest drift from the anchor after one substep.
"""

from __future__ import annotations

import os
import re
import struct

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, '_cache'):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, '_get_arch_for_options_for_nvrtc'):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, '_cache'):
                fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402
from materials import WATER  # noqa: E402
from simulation import Simulation  # noqa: E402
from world import World  # noqa: E402

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physics", "kernels")
KERNEL_FILES = ("integrate.cu", "pbf_solver.cu", "dfsph_solver.cu")
_HELPER_RE = re.compile(
    r"__device__ __forceinline__ void warp_atomic_max_uint\(.*?\n}\n", re.S)


def _helper_source(name):
    with open(os.path.join(KERNEL_DIR, name)) as f:
        m = _HELPER_RE.search(f.read())
    assert m, f"warp_atomic_max_uint not found in {name}"
    return m.group(0)


def test_helper_is_the_same_in_all_three_kernels():
    srcs = {name: _helper_source(name) for name in KERNEL_FILES}
    assert len(set(srcs.values())) == 1, srcs


def test_no_per_thread_displacement_atomic_left():
    for name in KERNEL_FILES:
        with open(os.path.join(KERNEL_DIR, name)) as f:
            src = f.read()
        assert "atomicMax(max_displacement_out" not in src, name
        assert "warp_atomic_max_uint(max_displacement_out" in src, name


_TEST_KERNEL = r'''
typedef unsigned int uint;
%s
// mode bit 0: lane returns early; bit 1: lane takes the second branch.
extern "C" __global__ void K_Test(const uint* bits, const uint* mode, int n,
                                  uint* out_warp, uint* out_ref) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mode[i] & 1u) return;
    uint b = bits[i];
    atomicMax(out_ref, b);
    if (mode[i] & 2u) {
        // Data-dependent delay so the two branches reach the helper apart.
        for (uint k = 0; k < (b & 7u); ++k) __nanosleep(32);
        warp_atomic_max_uint(out_warp, b);
    } else {
        warp_atomic_max_uint(out_warp, b);
    }
}
'''


@pytest.mark.parametrize("seed", range(6))
def test_warp_max_equals_per_thread_atomic(seed):
    rng = np.random.default_rng(seed)
    mod = cp.RawModule(code=_TEST_KERNEL % _helper_source("integrate.cu"),
                       options=("--std=c++11",))
    k = mod.get_function("K_Test")
    n = 100_003  # not a multiple of the block size: a partial last warp
    disp_sq = (rng.random(n, dtype=np.float32) * 1e-4).astype(np.float32)
    disp_sq[rng.random(n) < 0.3] = 0.0
    if seed == 1:
        disp_sq[:] = 0.0          # all zero: the word must stay 0
    if seed == 2:
        disp_sq[n // 2] = np.inf  # a blown-up particle must still read as inf
    bits = disp_sq.view(np.uint32)
    mode = rng.integers(0, 4, n, dtype=np.uint32)
    if seed == 3:
        mode[:] = 1               # every lane returns early
        mode[-1] = 0
    d_bits = cp.asarray(bits)
    d_mode = cp.asarray(mode)
    out = cp.zeros(2, dtype=cp.uint32)
    block = 128
    k(((n + block - 1) // block,), (block,),
      (d_bits, d_mode, np.int32(n), out[0:1], out[1:2]))
    got = out.get()  # test-only readback
    expected = int(bits[(mode & 1) == 0].max()) if np.any((mode & 1) == 0) else 0
    assert int(got[1]) == expected
    assert int(got[0]) == expected


MAX_PARTICLES = 20_000
DT = 0.001


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


@pytest.mark.parametrize("profile_name", ["WCSPH", "PBF", "DFSPH"])
def test_solver_max_displacement_is_max_drift(world, profile_name):
    """After one full-sort substep, max_displacement is max |x - anchor|^2."""
    from solver_profiles import PROFILES

    world.packed_info[:] = 0
    world._high_water = 0
    world.foam_count[:] = 0
    sim = Simulation(world, dt=DT, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=1)
    try:
        sim.fixed_dt_value = DT
        sim._upload_dt(DT)
        world.spawn_cube(min_corner=(-0.2, -0.1, -0.2), max_corner=(0.2, 0.1, 0.2),
                         material_id=WATER, spacing=0.02)
        sim.set_solver_profile(PROFILES[profile_name])
        sim.set_gravity(0.0)
        sim._spawn_substep = sim._damping_duration
        sim.notify_scene_changed()
        n = world._high_water
        rng = np.random.default_rng(7)
        world.velocity[:n, :3] = cp.asarray(
            rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32))
        sim._sort_skip_next = False
        sim._sim_step(n)
        cp.cuda.Device().synchronize()
        # Test-only readbacks.
        bits = int(world.max_displacement.get()[0])
        perm = sim._sort_perm[:n]
        pos = world.position[perm, :3]
        anchor = world.sorted_anchor[:n, :3]
        host_max = float(cp.max(cp.sum((pos - anchor) ** 2, axis=1)).get())
        got = struct.unpack('f', struct.pack('I', bits))[0]
        assert host_max > 0.0
        assert got == pytest.approx(host_max, rel=1e-5, abs=0.0), (got, host_max)
    finally:
        sim._sort_skip_next = False
        sim.close()
