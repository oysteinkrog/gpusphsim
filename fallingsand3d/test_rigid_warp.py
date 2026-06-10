"""Regression test for bd-r4-epic-x2j.1: warp_reduce_accumulate divergent-branch fix.

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


def _compile_module(src: str, name: str) -> cp.RawModule:
    """Compile a CuPy RawModule, applying PTX workaround if needed."""
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


def test_sph_shared_compiles_with_fix():
    """sph_shared.cuh must compile cleanly with the fixed warp_reduce_accumulate.

    This test compiles a minimal kernel that includes sph_shared.cuh and calls
    warp_reduce_accumulate, verifying the fixed function signature works end-to-end.
    """
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

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    src = r"""
#include "sph_shared.cuh"

extern "C" __global__ void test_wra_compile_check(float* out) {
    int lane = threadIdx.x & 31;
    int body_id = (lane < 16) ? 0 : 1;
    float3 val = make_float3((float)(lane+1), (float)(lane+2), (float)(lane+3));
    warp_reduce_accumulate(out, val, body_id);
}
"""
    mod = cp.RawModule(
        code=src,
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )
    kernel = mod.get_function("test_wra_compile_check")

    out = cp.zeros(8, dtype=cp.float32)
    kernel((1,), (32,), (out,))
    cp.cuda.Device().synchronize()
    result = out.get()

    b0 = (result[0], result[1], result[2])
    b1 = (result[4], result[5], result[6])

    assert b0 == pytest.approx(EXP_B0, rel=1e-5), (
        f"sph_shared body0 mismatch: expected {EXP_B0}, got {b0}"
    )
    assert b1 == pytest.approx(EXP_B1, rel=1e-5), (
        f"sph_shared body1 mismatch: expected {EXP_B1}, got {b1}"
    )
