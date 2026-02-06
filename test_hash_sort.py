"""Integration test for hash_sort.py -- K_CalcHash grid hashing kernel.

Acceptance criteria:
  - K_CalcHash computes grid cell: cell = int3((pos - grid_min) * grid_delta),
    clamped to [0, grid_res-1]
  - Hash = cell.z * grid_res.y * grid_res.x + cell.y * grid_res.x + cell.x
  - Grid parameters read from __constant__ GridParams c_grid
  - Grid sizing: grid_min=(-1,-1,-1), grid_max=(1,1,1), cell_size=h=0.04,
    grid_res=50, grid_delta=(25,25,25), num_cells=125000
  - CuPy RawModule compiles hash_sort.cu including common.cuh without errors
  - Test: 100K particles uniformly distributed in unit cube -> valid hashes [0, num_cells-1]
  - Test: particles at/outside grid boundaries -> clamped hashes (no OOB)
  - Kernel runs without errors for 500K particles

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import cupy  # type: ignore[import-untyped]
import numpy as np

from hash_sort import (
    CELL_SIZE,
    GRID_DELTA,
    GRID_MAX,
    GRID_MIN,
    GRID_PARAMS_DTYPE,
    GRID_RES,
    NUM_CELLS,
    build_grid_params,
    calc_hash,
    get_module,
    upload_grid_params,
)


def _host_hash(pos: np.ndarray) -> np.ndarray:
    """Reference CPU implementation of the hash for validation."""
    grid_min = GRID_MIN
    grid_delta = GRID_DELTA
    grid_res = GRID_RES.astype(np.float32)

    # calcGridCell: int3((pos - grid_min) * grid_delta)
    cell = np.floor((pos[:, :3] - grid_min) * grid_delta).astype(np.int32)

    # clamp to [0, grid_res - 1]
    cell = np.clip(cell, 0, GRID_RES - 1)

    # linear hash: z * ry * rx + y * rx + x
    rx = int(grid_res[0])
    ry = int(grid_res[1])
    h = cell[:, 2] * ry * rx + cell[:, 1] * rx + cell[:, 0]
    return h.astype(np.uint32)


def test_grid_constants() -> None:
    """Verify grid sizing matches acceptance criteria."""
    print("--- Grid constant checks ---")

    np.testing.assert_array_almost_equal(GRID_MIN, [-1.0, -1.0, -1.0])
    np.testing.assert_array_almost_equal(GRID_MAX, [1.0, 1.0, 1.0])
    assert abs(float(CELL_SIZE) - 0.04) < 1e-7
    np.testing.assert_array_equal(GRID_RES, [50, 50, 50])
    np.testing.assert_array_almost_equal(GRID_DELTA, [25.0, 25.0, 25.0])
    assert NUM_CELLS == 125_000
    print("[OK] Grid constants: grid_min=(-1,-1,-1) grid_max=(1,1,1) "
          "cell_size=0.04 grid_res=50 grid_delta=25 num_cells=125000")


def test_grid_params_struct() -> None:
    """Verify GridParams numpy dtype layout."""
    print("\n--- GridParams struct checks ---")

    # 5 float3 fields = 5 * 12 = 60 bytes
    assert GRID_PARAMS_DTYPE.itemsize == 60, (
        f"GridParams size mismatch: {GRID_PARAMS_DTYPE.itemsize} != 60"
    )
    print(f"[OK] sizeof(GridParams) = {GRID_PARAMS_DTYPE.itemsize} bytes")

    params = build_grid_params()
    np.testing.assert_array_almost_equal(params[0]["grid_size"], [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(params[0]["grid_min"], [-1.0, -1.0, -1.0])
    np.testing.assert_array_almost_equal(params[0]["grid_max"], [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(params[0]["grid_res"], [50.0, 50.0, 50.0])
    np.testing.assert_array_almost_equal(params[0]["grid_delta"], [25.0, 25.0, 25.0])
    print("[OK] GridParams fields populated correctly")


def test_compilation() -> None:
    """Verify CuPy RawModule compiles hash_sort.cu including common.cuh."""
    print("\n--- Compilation check ---")

    module = get_module()
    assert module is not None
    print("[OK] CuPy RawModule compiled hash_sort.cu (includes common.cuh)")

    # Verify kernel function is accessible
    kernel = module.get_function("K_CalcHash")  # type: ignore[union-attr]
    assert kernel is not None
    print("[OK] K_CalcHash kernel function found")

    # Verify constant memory symbol is accessible
    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]
    assert int(d_ptr) != 0
    print("[OK] c_grid constant memory symbol found")


def test_100k_uniform_unit_cube() -> None:
    """100K particles uniformly distributed in unit cube -> valid hashes."""
    print("\n--- 100K uniform unit-cube test ---")

    upload_grid_params()

    rng = np.random.default_rng(42)
    # Unit cube [0, 1]^3 is entirely inside grid [-1, 1]^3
    pos_np = np.zeros((100_000, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(0.0, 1.0, size=(100_000, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes, indices = calc_hash(pos_gpu)
    cupy.cuda.Device().synchronize()

    h_hashes = cupy.asnumpy(hashes)
    h_indices = cupy.asnumpy(indices)

    # All hashes in [0, NUM_CELLS - 1]
    assert h_hashes.min() >= 0, f"Min hash = {h_hashes.min()}, expected >= 0"
    assert h_hashes.max() < NUM_CELLS, (
        f"Max hash = {h_hashes.max()}, expected < {NUM_CELLS}"
    )
    print(f"[OK] All 100K hashes in [0, {NUM_CELLS - 1}]: "
          f"min={h_hashes.min()}, max={h_hashes.max()}")

    # Indices should be identity permutation
    np.testing.assert_array_equal(h_indices, np.arange(100_000, dtype=np.uint32))
    print("[OK] Indices are identity [0..99999]")

    # Cross-check against CPU reference
    ref = _host_hash(pos_np)
    np.testing.assert_array_equal(h_hashes, ref)
    print("[OK] GPU hashes match CPU reference for all 100K particles")


def test_boundary_clamping() -> None:
    """Particles at or outside grid boundaries produce clamped hashes."""
    print("\n--- Boundary clamping test ---")

    upload_grid_params()

    # Test positions: inside, on boundary, and outside
    test_positions = np.array([
        # Inside (center of grid)
        [0.0, 0.0, 0.0, 0.0],
        # At grid_min exactly
        [-1.0, -1.0, -1.0, 0.0],
        # At grid_max exactly (should clamp to res-1)
        [1.0, 1.0, 1.0, 0.0],
        # Far outside negative
        [-10.0, -10.0, -10.0, 0.0],
        # Far outside positive
        [10.0, 10.0, 10.0, 0.0],
        # Mixed: some axes inside, some outside
        [0.5, -5.0, 3.0, 0.0],
        # Just inside near grid_max
        [0.99, 0.99, 0.99, 0.0],
    ], dtype=np.float32)

    pos_gpu = cupy.asarray(test_positions)
    hashes, indices = calc_hash(pos_gpu)
    cupy.cuda.Device().synchronize()

    h_hashes = cupy.asnumpy(hashes)

    # All hashes must be in valid range
    for i, h in enumerate(h_hashes):
        assert 0 <= h < NUM_CELLS, (
            f"Particle {i} at {test_positions[i, :3]}: hash={h}, "
            f"expected in [0, {NUM_CELLS - 1}]"
        )
    print(f"[OK] All {len(test_positions)} boundary test particles have valid hashes")

    # Verify specific expected values:
    # Particle at grid_min (-1,-1,-1): cell = (0,0,0), hash = 0
    assert h_hashes[1] == 0, f"grid_min hash = {h_hashes[1]}, expected 0"
    print("[OK] Particle at grid_min -> hash = 0")

    # Particle far outside negative: should clamp to (0,0,0), hash = 0
    assert h_hashes[3] == 0, f"far-negative hash = {h_hashes[3]}, expected 0"
    print("[OK] Particle far outside negative -> hash = 0 (clamped)")

    # Particle far outside positive: should clamp to (49,49,49)
    max_hash = 49 * 50 * 50 + 49 * 50 + 49  # = 124999
    assert h_hashes[4] == max_hash, (
        f"far-positive hash = {h_hashes[4]}, expected {max_hash}"
    )
    print(f"[OK] Particle far outside positive -> hash = {max_hash} (clamped to max)")

    # Particle at grid_max (1,1,1): cell unclamped = (50,50,50), clamped to (49,49,49)
    assert h_hashes[2] == max_hash, (
        f"grid_max hash = {h_hashes[2]}, expected {max_hash}"
    )
    print(f"[OK] Particle at grid_max -> hash = {max_hash} (clamped)")

    # Cross-check all against CPU reference
    ref = _host_hash(test_positions)
    np.testing.assert_array_equal(h_hashes, ref)
    print("[OK] Boundary hashes match CPU reference")


def test_500k_no_errors() -> None:
    """Kernel runs without errors for 500K particles."""
    print("\n--- 500K particle stress test ---")

    upload_grid_params()

    rng = np.random.default_rng(123)
    # Distribute across entire grid range [-1, 1]^3
    pos_np = np.zeros((500_000, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-1.0, 1.0, size=(500_000, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes, indices = calc_hash(pos_gpu)
    cupy.cuda.Device().synchronize()

    h_hashes = cupy.asnumpy(hashes)

    # All valid
    assert h_hashes.min() >= 0
    assert h_hashes.max() < NUM_CELLS
    print(f"[OK] 500K particles hashed without errors: "
          f"min={h_hashes.min()}, max={h_hashes.max()}")

    # Spot-check against CPU reference (first 1000)
    ref = _host_hash(pos_np[:1000])
    np.testing.assert_array_equal(h_hashes[:1000], ref)
    print("[OK] First 1000 hashes match CPU reference")


def test_known_position() -> None:
    """Verify hash for a particle with a known grid cell."""
    print("\n--- Known position test ---")

    upload_grid_params()

    # Position (0.02, 0.02, 0.02):
    #   cell = int3((0.02 - (-1)) * 25, (0.02 - (-1)) * 25, (0.02 - (-1)) * 25)
    #        = int3(25.5, 25.5, 25.5) = (25, 25, 25)
    #   hash = 25 * 50 * 50 + 25 * 50 + 25 = 62500 + 1250 + 25 = 63775
    pos = np.array([[0.02, 0.02, 0.02, 0.0]], dtype=np.float32)
    pos_gpu = cupy.asarray(pos)

    hashes, _ = calc_hash(pos_gpu)
    cupy.cuda.Device().synchronize()

    h = int(cupy.asnumpy(hashes)[0])
    expected = 25 * 50 * 50 + 25 * 50 + 25  # 63775
    assert h == expected, f"hash = {h}, expected {expected}"
    print(f"[OK] Position (0.02, 0.02, 0.02) -> cell (25,25,25) -> hash = {h}")


def main() -> None:
    test_grid_constants()
    test_grid_params_struct()
    test_compilation()
    test_known_position()
    test_100k_uniform_unit_cube()
    test_boundary_clamping()
    test_500k_no_errors()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
