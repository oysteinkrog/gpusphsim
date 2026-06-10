"""Test material property table and interaction matrix.

Verifies US-006 acceptance criteria:
  - MaterialDef dataclass fields match MaterialProps C struct
  - MATERIALS dict has 19 entries with correct IDs (0-18)
  - Struct sizes match common.cuh (72 bytes MaterialProps, 8 bytes Interaction)
  - 32x32 interaction matrix with correct values
  - IDs 19-31 are zeroed
  - GPU upload via CuPy RawModule constant memory
  - Test kernel reads c_materials[5].rest_density == 2500.0 (water, game-tuned)
  - Test kernel reads c_interactions[8][10].reaction_rate == 0.3 (acid-metal)
"""

import sys

import numpy as np


def test_host_side():
    """Test host-side data structures without GPU."""
    from materials import (
        ACID, DEAD, DIRT, FIRE, GRAVEL, GUNPOWDER, ICE, LAVA,
        MATERIAL_PROPS_DTYPE, MATERIALS, MAX_MATERIALS, METAL, NUM_DEFINED,
        OIL, SAND, SMOKE, STEAM, STONE, WATER, WOOD,
        WET_SAND, MUD, MAT_RIGID,
        INTERACTION_DTYPE, build_interaction_matrix, build_material_array,
    )

    print("=== Host-side tests ===")

    # --- 19 materials defined (IDs 0-18: original 16 + WET_SAND, MUD, MAT_RIGID) ---
    print("\nChecking material count...")
    assert len(MATERIALS) == 19, f"Expected 19 materials, got {len(MATERIALS)}"
    print(f"  OK - {len(MATERIALS)} materials defined")

    # --- All 19 IDs present ---
    print("\nChecking material IDs 0-18...")
    expected_ids = {
        DEAD: "DEAD", STONE: "STONE", SAND: "SAND", DIRT: "DIRT",
        GRAVEL: "GRAVEL", WATER: "WATER", OIL: "OIL", LAVA: "LAVA",
        ACID: "ACID", WOOD: "WOOD", METAL: "METAL", ICE: "ICE",
        STEAM: "STEAM", SMOKE: "SMOKE", FIRE: "FIRE", GUNPOWDER: "GUNPOWDER",
        WET_SAND: "WET_SAND", MUD: "MUD", MAT_RIGID: "RIGID",
    }
    for mat_id, expected_name in expected_ids.items():
        assert mat_id in MATERIALS, f"Missing material ID {mat_id} ({expected_name})"
        assert MATERIALS[mat_id].name == expected_name, (
            f"Material {mat_id}: expected name '{expected_name}', got '{MATERIALS[mat_id].name}'"
        )
    print("  OK - all 19 material IDs present with correct names")

    # --- Struct sizes ---
    print("\nChecking struct sizes...")
    assert MATERIAL_PROPS_DTYPE.itemsize == 72, (
        f"MaterialProps: {MATERIAL_PROPS_DTYPE.itemsize} != 72"
    )
    assert INTERACTION_DTYPE.itemsize == 8, (
        f"Interaction: {INTERACTION_DTYPE.itemsize} != 8"
    )
    print(f"  MaterialProps: {MATERIAL_PROPS_DTYPE.itemsize} bytes (expected 72)")
    print(f"  Interaction: {INTERACTION_DTYPE.itemsize} bytes (expected 8)")
    print("  OK")

    # --- Water rest_density ---
    # Game-tuned: 2500.0 kg/m^3 (SPH density is ~2.5x physical to keep Tait EOS stable
    # with k=500, viscosity=1.0 at particle_spacing=0.02m)
    print("\nChecking water rest_density...")
    assert MATERIALS[WATER].rest_density == 2500.0, (
        f"Water rest_density: {MATERIALS[WATER].rest_density} != 2500.0"
    )
    print(f"  WATER rest_density = {MATERIALS[WATER].rest_density}")
    print("  OK")

    # --- Build material array ---
    print("\nBuilding material array (32 entries)...")
    arr = build_material_array()
    assert arr.shape == (32,), f"Shape: {arr.shape} != (32,)"
    assert arr.dtype == MATERIAL_PROPS_DTYPE
    assert arr[WATER]["rest_density"] == np.float32(2500.0), (
        f"arr[5].rest_density = {arr[WATER]['rest_density']}"
    )
    print(f"  arr[WATER].rest_density = {arr[WATER]['rest_density']}")
    print("  OK")

    # --- IDs 19-31 are zeroed (IDs 0-18 are defined: original 16 + WET_SAND, MUD, MAT_RIGID) ---
    print("\nChecking IDs 19-31 are zeroed...")
    for i in range(NUM_DEFINED, 32):
        for field in MATERIAL_PROPS_DTYPE.names:
            val = arr[i][field]
            assert val == 0, f"arr[{i}].{field} = {val} (expected 0)"
    print(f"  OK - all reserved slots ({NUM_DEFINED}-31) zeroed")

    # --- Build interaction matrix ---
    print("\nBuilding interaction matrix (32x32)...")
    matrix = build_interaction_matrix()
    assert matrix.shape == (32, 32), f"Shape: {matrix.shape} != (32, 32)"
    assert matrix.dtype == INTERACTION_DTYPE
    print("  OK")

    # --- Acid-metal reaction_rate ---
    print("\nChecking acid-metal interaction...")
    acid_metal_rate = matrix[ACID][METAL]["reaction_rate"]
    assert acid_metal_rate == np.float32(0.3), (
        f"acid-metal reaction_rate: {acid_metal_rate} != 0.3"
    )
    # Symmetric check
    metal_acid_rate = matrix[METAL][ACID]["reaction_rate"]
    assert metal_acid_rate == np.float32(0.3), (
        f"metal-acid reaction_rate (symmetric): {metal_acid_rate} != 0.3"
    )
    print(f"  interactions[ACID][METAL].reaction_rate = {acid_metal_rate}")
    print(f"  interactions[METAL][ACID].reaction_rate = {metal_acid_rate} (symmetric)")
    print("  OK")

    # --- Water-water has no reaction ---
    print("\nChecking water-water interaction (should be zero)...")
    ww_rate = matrix[WATER][WATER]["reaction_rate"]
    ww_heat = matrix[WATER][WATER]["heat_exchange"]
    assert ww_rate == 0.0, f"water-water reaction_rate: {ww_rate} != 0"
    assert ww_heat == 0.0, f"water-water heat_exchange: {ww_heat} != 0"
    print("  OK - water-water has zero reaction and heat exchange")

    # --- Total constant memory size ---
    mat_bytes = MAX_MATERIALS * MATERIAL_PROPS_DTYPE.itemsize
    int_bytes = MAX_MATERIALS * MAX_MATERIALS * INTERACTION_DTYPE.itemsize
    total = mat_bytes + int_bytes
    print(f"\nConstant memory usage:")
    print(f"  Materials: {MAX_MATERIALS} x {MATERIAL_PROPS_DTYPE.itemsize} = {mat_bytes} bytes")
    print(f"  Interactions: {MAX_MATERIALS}x{MAX_MATERIALS} x {INTERACTION_DTYPE.itemsize} = {int_bytes} bytes")
    print(f"  Total: {total} bytes ({total/1024:.1f} KB, limit 64 KB)")
    assert total <= 65536, f"Exceeds 64 KB constant memory limit: {total}"
    print("  OK - within 64 KB limit")

    print("\n=== All host-side tests passed ===")


def test_gpu():
    """Test GPU upload and constant memory readback via test kernel."""
    import cupy
    from cupy.cuda import compiler as _compiler
    from cupy.cuda import device as _device

    from materials import upload_to_gpu, get_module

    print("\n=== GPU tests ===")

    # Force PTX if needed (Blackwell workaround)
    gpu_cc = _device.Device().compute_capability
    nvrtc_max = _compiler._get_max_compute_capability()
    print(f"\nGPU compute capability: {gpu_cc}, NVRTC max: {nvrtc_max}")
    if int(gpu_cc) > int(nvrtc_max):
        print("  Enabling PTX mode for forward compatibility")
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
            _compiler._get_arch_for_options_for_nvrtc._cache = {}
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}

    # --- Compile and upload ---
    print("\nUploading materials to GPU constant memory...")
    upload_to_gpu()
    print("  OK - upload completed")

    # --- Test kernel readback ---
    print("\nRunning test_read_materials kernel...")
    module = get_module()
    k_test = module.get_function("test_read_materials")
    out = cupy.zeros(2, dtype=cupy.float32)
    k_test((1,), (1,), (out,))
    cupy.cuda.Device().synchronize()
    results = out.get()

    print(f"  c_materials[5].rest_density = {results[0]} (expected 2500.0)")
    print(f"  c_interactions[8][10].reaction_rate = {results[1]} (expected 0.3)")

    assert abs(results[0] - 2500.0) < 1e-3, (
        f"Water rest_density readback: {results[0]} != 2500.0"
    )
    assert abs(results[1] - 0.3) < 1e-6, (
        f"Acid-metal reaction_rate readback: {results[1]} != 0.3"
    )
    print("  OK - GPU readback matches expected values")

    # --- Verify constant memory symbols exist ---
    print("\nVerifying constant memory symbol lookup...")
    for sym_name in ["c_materials", "c_interactions"]:
        ptr = module.get_global(sym_name)
        assert int(ptr) != 0, f"get_global('{sym_name}') returned null"
        print(f"  {sym_name}: device ptr = 0x{int(ptr):x}")
    print("  OK - all symbols resolved")

    print("\n=== All GPU tests passed ===")


def main():
    test_host_side()
    test_gpu()
    print("\n=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
