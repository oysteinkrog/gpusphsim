"""Integration test for materials.py.

Uploads MaterialProps[32] and Interaction[32][32] to CUDA constant memory,
then runs a test kernel that reads back specific values to verify correctness.

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.

Acceptance-criteria checks:
  - c_materials[5].rest_density  == 1000.0  (water)
  - c_interactions[8][10].reaction_rate == 0.3  (acid corrodes metal)
"""

from __future__ import annotations

import sys

import cupy  # type: ignore[import-untyped]
import numpy as np

from materials import (
    ACID,
    DEAD,
    DIRT,
    FIRE,
    GRAVEL,
    GUNPOWDER,
    ICE,
    INTERACTION_DTYPE,
    LAVA,
    MATERIAL_PROPS_DTYPE,
    MATERIALS,
    MAX_MATERIALS,
    METAL,
    NUM_DEFINED,
    OIL,
    SAND,
    SMOKE,
    STEAM,
    STONE,
    WATER,
    WOOD,
    build_interaction_matrix,
    build_material_array,
    get_module,
    upload_to_gpu,
)


def _approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) < tol


def test_host_data() -> None:
    """Verify host-side material array and interaction matrix."""
    print("--- Host-side checks ---")

    # Material count
    assert len(MATERIALS) == NUM_DEFINED == 16
    print(f"[OK] {NUM_DEFINED} materials defined")

    # ID constants
    expected_ids = {
        DEAD: 0, STONE: 1, SAND: 2, DIRT: 3, GRAVEL: 4,
        WATER: 5, OIL: 6, LAVA: 7, ACID: 8, WOOD: 9,
        METAL: 10, ICE: 11, STEAM: 12, SMOKE: 13, FIRE: 14,
        GUNPOWDER: 15,
    }
    for mat_id, expected in expected_ids.items():
        assert mat_id == expected, f"ID mismatch: {mat_id} != {expected}"
    print("[OK] All 16 material IDs correct (0-15)")

    # Material array shape and reserved slots
    arr = build_material_array()
    assert arr.shape == (MAX_MATERIALS,)
    assert arr.dtype == MATERIAL_PROPS_DTYPE
    print(f"[OK] MaterialProps array: shape={arr.shape}, itemsize={arr.dtype.itemsize}")

    # Water rest_density
    assert _approx(float(arr[WATER]["rest_density"]), 1000.0), (
        f"Water rest_density={arr[WATER]['rest_density']}, expected 1000.0"
    )
    print(f"[OK] c_materials[{WATER}].rest_density = {arr[WATER]['rest_density']}")

    # Reserved slots (16-31) should be zeroed
    for i in range(NUM_DEFINED, MAX_MATERIALS):
        assert float(arr[i]["rest_density"]) == 0.0, (
            f"Reserved slot {i} has non-zero rest_density"
        )
    print("[OK] IDs 16-31 are zeroed (reserved)")

    # Interaction matrix
    mat = build_interaction_matrix()
    assert mat.shape == (MAX_MATERIALS, MAX_MATERIALS)
    assert mat.dtype == INTERACTION_DTYPE
    print(f"[OK] Interaction matrix: shape={mat.shape}, itemsize={mat.dtype.itemsize}")

    # Acid-metal reaction_rate
    assert _approx(float(mat[ACID][METAL]["reaction_rate"]), 0.3), (
        f"acid-metal reaction_rate={mat[ACID][METAL]['reaction_rate']}, expected 0.3"
    )
    print(f"[OK] c_interactions[{ACID}][{METAL}].reaction_rate = "
          f"{mat[ACID][METAL]['reaction_rate']}")

    # Symmetry check
    assert _approx(
        float(mat[ACID][METAL]["reaction_rate"]),
        float(mat[METAL][ACID]["reaction_rate"]),
    )
    assert _approx(
        float(mat[ACID][METAL]["heat_exchange"]),
        float(mat[METAL][ACID]["heat_exchange"]),
    )
    print("[OK] Interaction matrix is symmetric (spot-checked acid<->metal)")


def test_gpu_upload_and_readback() -> None:
    """Upload to GPU constant memory, run test kernel, verify readback."""
    print("\n--- GPU upload + readback ---")

    # Upload
    upload_to_gpu()
    print("[OK] upload_to_gpu() completed")

    # Run the test kernel from KERNEL_SOURCE
    module = get_module()
    test_kernel = module.get_function("test_read_materials")

    out = cupy.zeros(2, dtype=cupy.float32)
    test_kernel((1,), (1,), (out,))
    cupy.cuda.Device().synchronize()

    result = cupy.asnumpy(out)
    print(f"     out[0] (water rest_density) = {result[0]}")
    print(f"     out[1] (acid-metal reaction_rate) = {result[1]}")

    # Verify: c_materials[5].rest_density == 1000.0 (water)
    assert _approx(float(result[0]), 1000.0), (
        f"GPU readback: water rest_density = {result[0]}, expected 1000.0"
    )
    print(f"[OK] c_materials[5].rest_density = {result[0]} (water)")

    # Verify: c_interactions[8][10].reaction_rate == 0.3 (acid corrodes metal)
    assert _approx(float(result[1]), 0.3, tol=1e-4), (
        f"GPU readback: acid-metal reaction_rate = {result[1]}, expected 0.3"
    )
    print(f"[OK] c_interactions[8][10].reaction_rate = {result[1]} (acid corrodes metal)")


def test_struct_sizes() -> None:
    """Verify struct sizes match C expectations."""
    print("\n--- Struct size checks ---")
    assert MATERIAL_PROPS_DTYPE.itemsize == 64, (
        f"MaterialProps: {MATERIAL_PROPS_DTYPE.itemsize} bytes, expected 64"
    )
    print(f"[OK] sizeof(MaterialProps) = {MATERIAL_PROPS_DTYPE.itemsize} bytes")

    assert INTERACTION_DTYPE.itemsize == 8, (
        f"Interaction: {INTERACTION_DTYPE.itemsize} bytes, expected 8"
    )
    print(f"[OK] sizeof(Interaction) = {INTERACTION_DTYPE.itemsize} bytes")

    # Total constant memory usage
    mat_bytes = MAX_MATERIALS * MATERIAL_PROPS_DTYPE.itemsize
    int_bytes = MAX_MATERIALS * MAX_MATERIALS * INTERACTION_DTYPE.itemsize
    total = mat_bytes + int_bytes
    print(f"[OK] Total constant memory: {mat_bytes} + {int_bytes} = {total} bytes "
          f"({total / 1024:.1f} KB, limit 64 KB)")
    assert total <= 65536, f"Exceeds 64 KB constant memory limit: {total} bytes"


def main() -> None:
    test_struct_sizes()
    test_host_data()
    test_gpu_upload_and_readback()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
