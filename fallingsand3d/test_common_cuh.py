"""Test that common.cuh compiles and all structs/enums/macros are usable.

Verifies US-005 acceptance criteria:
  - Compiles via CuPy RawModule
  - Struct sizes match expected values
  - Enum values correct
  - packed_info macros produce correct results
  - Constant memory symbols exist
"""

import os
import sys

import cupy
import numpy as np
import pytest
from cupy.cuda import compiler as _compiler
from cupy.cuda import device as _device


def _ensure_ptx_if_needed():
    """Force PTX mode if GPU arch exceeds NVRTC's max sm target."""
    gpu_cc = _device.Device().compute_capability
    nvrtc_max = _compiler._get_max_compute_capability()
    if int(gpu_cc) > int(nvrtc_max):
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
            _compiler._get_arch_for_options_for_nvrtc._cache = {}
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}


# Test kernel source that exercises all structs, enums, and macros
TEST_KERNEL = r"""
#include "common.cuh"

extern "C" __global__
void test_sizes(int* out) {
    // out[0] = sizeof(MaterialProps)
    // out[1] = sizeof(Interaction)
    // out[2] = sizeof(GridParams)
    // out[3] = sizeof(SimParams)
    // out[4] = sizeof(PrecalcParams)
    out[0] = (int)sizeof(MaterialProps);
    out[1] = (int)sizeof(Interaction);
    out[2] = (int)sizeof(GridParams);
    out[3] = (int)sizeof(SimParams);
    out[4] = (int)sizeof(PrecalcParams);
}

extern "C" __global__
void test_enums(int* out) {
    out[0] = FLUID;
    out[1] = GRANULAR;
    out[2] = GAS;
    out[3] = STATIC;
}

extern "C" __global__
void test_packed_info(uint* in_packed, int* out) {
    uint p = in_packed[0];
    out[0] = (int)GET_MATERIAL_ID(p);
    out[1] = (int)GET_BEHAVIOR(p);
    out[2] = (int)IS_SLEEPING(p);
    out[3] = (int)HAS_SPAWN_FLAG(p);
    out[4] = (int)HAS_JUST_WOKE(p);

    // Test SET/CLEAR macros
    uint sleeping = SET_SLEEPING(p);
    out[5] = (int)IS_SLEEPING(sleeping);
    uint cleared = CLEAR_SLEEPING(sleeping);
    out[6] = (int)IS_SLEEPING(cleared);

    uint spawned = SET_SPAWN_FLAG(p);
    out[7] = (int)HAS_SPAWN_FLAG(spawned);

    // Test MAKE_PACKED
    uint made = MAKE_PACKED(42, 2);  // material 42, GAS
    out[8] = (int)GET_MATERIAL_ID(made);
    out[9] = (int)GET_BEHAVIOR(made);
}

extern "C" __global__
void test_constant_read(float* out) {
    // Read from c_materials[0] to verify constant memory is accessible
    out[0] = c_materials[0].rest_density;
    out[1] = c_materials[1].eos_stiffness;
    // Read from c_interactions[0][1]
    out[2] = c_interactions[0][1].reaction_rate;
    // Read from c_sim
    out[3] = c_sim.smoothing_length;
    // Read from c_grid
    out[4] = c_grid.grid_min.x;
    // Read from c_precalc
    out[5] = c_precalc.poly6_coeff;
}
"""


@pytest.fixture(scope="module")
def compiled_module():
    """Compile the test kernel once per test module."""
    _ensure_ptx_if_needed()
    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    module = cupy.RawModule(
        code=TEST_KERNEL,
        options=("--std=c++11", f"-I{kernel_dir}"),
    )
    return module


def test_compilation(compiled_module):
    """Verify common.cuh compiles via CuPy RawModule."""
    assert compiled_module is not None
    print("[OK] common.cuh compiled via CuPy RawModule")


def test_struct_sizes(compiled_module):
    """Verify struct sizes match expected values."""
    k_sizes = compiled_module.get_function("test_sizes")
    out = cupy.zeros(5, dtype=cupy.int32)
    k_sizes((1,), (1,), (out,))
    cupy.cuda.Device().synchronize()
    sizes = out.get()
    print(f"  MaterialProps: {sizes[0]} bytes (expected 72)")
    print(f"  Interaction:   {sizes[1]} bytes (expected 8)")
    print(f"  GridParams:    {sizes[2]} bytes (expected 32)")
    print(f"  SimParams:     {sizes[3]} bytes")
    print(f"  PrecalcParams: {sizes[4]} bytes")
    # MaterialProps: 18 fields x 4 bytes = 72 bytes (struct was extended from 64)
    assert sizes[0] == 72, f"MaterialProps size {sizes[0]} != 72"
    assert sizes[1] == 8, f"Interaction size {sizes[1]} != 8"
    # GridParams: float3 grid_min(12) + float3 grid_delta(12) + uint table_size(4)
    #             + uint table_mask(4) = 32 bytes (changed from 52: no grid_max/res/num_cells)
    assert sizes[2] == 32, f"GridParams size {sizes[2]} != 32"
    print("  [OK] all struct sizes correct")


def test_enum_values(compiled_module):
    """Verify BehaviorClass enum values are correct."""
    k_enums = compiled_module.get_function("test_enums")
    out = cupy.zeros(4, dtype=cupy.int32)
    k_enums((1,), (1,), (out,))
    cupy.cuda.Device().synchronize()
    enums = out.get()
    assert enums[0] == 0, f"FLUID={enums[0]} != 0"
    assert enums[1] == 1, f"GRANULAR={enums[1]} != 1"
    assert enums[2] == 2, f"GAS={enums[2]} != 2"
    assert enums[3] == 3, f"STATIC={enums[3]} != 3"
    print("  [OK] FLUID=0, GRANULAR=1, GAS=2, STATIC=3")


def test_packed_info_macros(compiled_module):
    """Verify packed_info macros produce correct results."""
    k_packed = compiled_module.get_function("test_packed_info")
    # Create packed value: material 5, behavior GRANULAR(1), not sleeping, no flags
    packed_val = (5 & 0xFF) | ((1 & 0x3) << 8)  # 0x105
    in_packed = cupy.array([packed_val], dtype=cupy.uint32)
    out = cupy.zeros(10, dtype=cupy.int32)
    k_packed((1,), (1,), (in_packed, out))
    cupy.cuda.Device().synchronize()
    results = out.get()
    assert results[0] == 5, f"GET_MATERIAL_ID={results[0]} != 5"
    assert results[1] == 1, f"GET_BEHAVIOR={results[1]} != 1 (GRANULAR)"
    assert results[2] == 0, f"IS_SLEEPING={results[2]} != 0"
    assert results[3] == 0, f"HAS_SPAWN_FLAG={results[3]} != 0"
    assert results[4] == 0, f"HAS_JUST_WOKE={results[4]} != 0"
    assert results[5] == 1, f"SET_SLEEPING then IS_SLEEPING={results[5]} != 1"
    assert results[6] == 0, f"CLEAR_SLEEPING then IS_SLEEPING={results[6]} != 0"
    assert results[7] == 1, f"SET_SPAWN_FLAG then HAS_SPAWN_FLAG={results[7]} != 1"
    assert results[8] == 42, f"MAKE_PACKED material={results[8]} != 42"
    assert results[9] == 2, f"MAKE_PACKED behavior={results[9]} != 2 (GAS)"
    print("  [OK] all macros produce correct results")


def test_constant_memory_symbols(compiled_module):
    """Verify all constant memory symbols are accessible and read as zero by default."""
    k_const = compiled_module.get_function("test_constant_read")
    out = cupy.zeros(6, dtype=cupy.float32)
    # Just verify the kernel runs without error (constant memory is zeroed by default)
    k_const((1,), (1,), (out,))
    cupy.cuda.Device().synchronize()
    const_vals = out.get()
    for i, v in enumerate(const_vals):
        assert v == 0.0, f"Unexpected non-zero constant memory at index {i}: {v}"
    print("  [OK] all constant memory symbols accessible (default zero)")

    # Verify constant memory symbols can be looked up
    for sym_name in ["c_materials", "c_interactions", "c_grid", "c_sim", "c_precalc"]:
        ptr = compiled_module.get_global(sym_name)
        assert int(ptr) != 0, f"get_global('{sym_name}') returned null"
        print(f"  {sym_name}: device ptr = 0x{int(ptr):x}")
    print("  [OK] all symbols resolved")


def main():
    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")

    print("Compiling common.cuh via CuPy RawModule...")
    module = cupy.RawModule(
        code=TEST_KERNEL,
        options=("--std=c++11", f"-I{kernel_dir}"),
    )
    print("  OK - compilation succeeded")

    # --- Test struct sizes ---
    print("\nTesting struct sizes...")
    k_sizes = module.get_function("test_sizes")
    out = cupy.zeros(5, dtype=cupy.int32)
    k_sizes((1,), (1,), (out,))
    cupy.cuda.Device().synchronize()
    sizes = out.get()
    print(f"  MaterialProps: {sizes[0]} bytes (expected 72)")
    print(f"  Interaction:   {sizes[1]} bytes (expected 8)")
    print(f"  GridParams:    {sizes[2]} bytes (expected 32)")
    print(f"  SimParams:     {sizes[3]} bytes")
    print(f"  PrecalcParams: {sizes[4]} bytes")
    assert sizes[0] == 72, f"MaterialProps size {sizes[0]} != 72"
    assert sizes[1] == 8, f"Interaction size {sizes[1]} != 8"
    assert sizes[2] == 32, f"GridParams size {sizes[2]} != 32"
    print("  OK - all sizes correct")

    # --- Test enum values ---
    print("\nTesting BehaviorClass enum values...")
    k_enums = module.get_function("test_enums")
    out = cupy.zeros(4, dtype=cupy.int32)
    k_enums((1,), (1,), (out,))
    cupy.cuda.Device().synchronize()
    enums = out.get()
    assert enums[0] == 0, f"FLUID={enums[0]} != 0"
    assert enums[1] == 1, f"GRANULAR={enums[1]} != 1"
    assert enums[2] == 2, f"GAS={enums[2]} != 2"
    assert enums[3] == 3, f"STATIC={enums[3]} != 3"
    print("  OK - FLUID=0, GRANULAR=1, GAS=2, STATIC=3")

    # --- Test packed_info macros ---
    print("\nTesting packed_info macros...")
    k_packed = module.get_function("test_packed_info")
    packed_val = (5 & 0xFF) | ((1 & 0x3) << 8)  # 0x105
    in_packed = cupy.array([packed_val], dtype=cupy.uint32)
    out = cupy.zeros(10, dtype=cupy.int32)
    k_packed((1,), (1,), (in_packed, out))
    cupy.cuda.Device().synchronize()
    results = out.get()
    assert results[0] == 5, f"GET_MATERIAL_ID={results[0]} != 5"
    assert results[1] == 1, f"GET_BEHAVIOR={results[1]} != 1 (GRANULAR)"
    assert results[2] == 0, f"IS_SLEEPING={results[2]} != 0"
    assert results[3] == 0, f"HAS_SPAWN_FLAG={results[3]} != 0"
    assert results[4] == 0, f"HAS_JUST_WOKE={results[4]} != 0"
    assert results[5] == 1, f"SET_SLEEPING then IS_SLEEPING={results[5]} != 1"
    assert results[6] == 0, f"CLEAR_SLEEPING then IS_SLEEPING={results[6]} != 0"
    assert results[7] == 1, f"SET_SPAWN_FLAG then HAS_SPAWN_FLAG={results[7]} != 1"
    assert results[8] == 42, f"MAKE_PACKED material={results[8]} != 42"
    assert results[9] == 2, f"MAKE_PACKED behavior={results[9]} != 2 (GAS)"
    print("  OK - all macros produce correct results")

    # --- Test constant memory symbols exist ---
    print("\nTesting constant memory symbol access...")
    k_const = module.get_function("test_constant_read")
    out = cupy.zeros(6, dtype=cupy.float32)
    k_const((1,), (1,), (out,))
    cupy.cuda.Device().synchronize()
    const_vals = out.get()
    for i, v in enumerate(const_vals):
        assert v == 0.0, f"Unexpected non-zero constant memory at index {i}: {v}"
    print("  OK - all constant memory symbols accessible (default zero)")

    print("\nVerifying constant memory symbol lookup...")
    for sym_name in ["c_materials", "c_interactions", "c_grid", "c_sim", "c_precalc"]:
        ptr = module.get_global(sym_name)
        assert int(ptr) != 0, f"get_global('{sym_name}') returned null"
        print(f"  {sym_name}: device ptr = 0x{int(ptr):x}")
    print("  OK - all symbols resolved")

    print("\n=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
