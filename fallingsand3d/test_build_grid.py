"""Integration test for build_grid.py -- K_BuildDataStruct kernel.

Acceptance criteria:
  - K_BuildDataStruct writes cell_indexes_start[hash] = first particle index
  - K_BuildDataStruct writes cell_indexes_end[hash] = one past last particle index
  - Empty cells have cell_indexes_start = 0xFFFFFFFF
  - cell_indexes_start and cell_indexes_end allocated as CuPy uint32 arrays of
    size num_cells (125000)
  - cell_indexes_start memset to 0xFFFFFFFF before each frame's kernel launch
  - Test: for a known configuration (8 particles in 2 cells), cell_start/end correct
  - Test: sum of (cell_end - cell_start) across all non-empty cells equals num_particles
  - Kernel runs without errors for 500K particles
  - Block size = 256

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import sys
import os

# Ensure fallingsand3d directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np

from hash_sort import (
    NUM_CELLS,
    calc_hash,
    sort_by_hash,
    upload_grid_params as upload_hash_grid_params,
)
from build_grid import (
    BLOCK_SIZE,
    EMPTY_CELL,
    allocate_cell_tables,
    build_data_struct,
    get_module,
    upload_grid_params,
)


def test_compilation() -> None:
    """Verify CuPy RawModule compiles build_grid.cu."""
    print("--- Compilation check ---")

    module = get_module()
    assert module is not None
    print("[OK] CuPy RawModule compiled build_grid.cu (includes common.cuh)")

    kernel = module.get_function("K_BuildDataStruct")  # type: ignore[union-attr]
    assert kernel is not None
    print("[OK] K_BuildDataStruct kernel function found")

    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]
    assert int(d_ptr) != 0
    print("[OK] c_grid constant memory symbol found")


def test_allocation() -> None:
    """Verify cell table allocation dimensions and dtype."""
    print("\n--- Allocation check ---")

    cell_start, cell_end = allocate_cell_tables()
    assert cell_start.shape == (NUM_CELLS,), f"shape {cell_start.shape}"
    assert cell_end.shape == (NUM_CELLS,), f"shape {cell_end.shape}"
    assert cell_start.dtype == cupy.uint32
    assert cell_end.dtype == cupy.uint32
    print(f"[OK] cell_start/end: shape=({NUM_CELLS},), dtype=uint32")


def test_block_size() -> None:
    """Verify block size is 256."""
    print("\n--- Block size check ---")
    assert BLOCK_SIZE == 256, f"BLOCK_SIZE = {BLOCK_SIZE}, expected 256"
    print("[OK] BLOCK_SIZE = 256")


def test_empty_cells_sentinel() -> None:
    """With one particle, all other cells should have 0xFFFFFFFF sentinel."""
    print("\n--- Empty cells sentinel test ---")

    sorted_hashes = cupy.array([0], dtype=cupy.uint32)
    cell_start, cell_end = build_data_struct(sorted_hashes)
    cupy.cuda.Device().synchronize()

    h_start = cupy.asnumpy(cell_start)
    h_end = cupy.asnumpy(cell_end)

    # Cell 0 should be occupied: start=0, end=1
    assert h_start[0] == 0, f"cell_start[0] = {h_start[0]}, expected 0"
    assert h_end[0] == 1, f"cell_end[0] = {h_end[0]}, expected 1"
    print("[OK] Cell 0: start=0, end=1")

    # All other cells should have 0xFFFFFFFF sentinel
    empty_mask = h_start[1:] == EMPTY_CELL
    assert empty_mask.all(), (
        f"Not all empty cells have sentinel: "
        f"{np.sum(~empty_mask)} cells have non-sentinel values"
    )
    print(f"[OK] All {NUM_CELLS - 1} other cells have start = 0xFFFFFFFF")


def test_known_8_particles_2_cells() -> None:
    """8 particles in 2 cells: verify exact cell_start/end values."""
    print("\n--- Known configuration: 8 particles in 2 cells ---")

    # 5 particles in cell hash=10, 3 particles in cell hash=42.
    # Sorted hashes: [10, 10, 10, 10, 10, 42, 42, 42]
    sorted_hashes = cupy.array(
        [10, 10, 10, 10, 10, 42, 42, 42], dtype=cupy.uint32
    )

    cell_start, cell_end = build_data_struct(sorted_hashes)
    cupy.cuda.Device().synchronize()

    h_start = cupy.asnumpy(cell_start)
    h_end = cupy.asnumpy(cell_end)

    # Cell 10: particles 0..4
    assert h_start[10] == 0, f"cell_start[10] = {h_start[10]}, expected 0"
    assert h_end[10] == 5, f"cell_end[10] = {h_end[10]}, expected 5"
    print("[OK] Cell 10: start=0, end=5 (5 particles)")

    # Cell 42: particles 5..7
    assert h_start[42] == 5, f"cell_start[42] = {h_start[42]}, expected 5"
    assert h_end[42] == 8, f"cell_end[42] = {h_end[42]}, expected 8"
    print("[OK] Cell 42: start=5, end=8 (3 particles)")

    # All other cells empty
    for c in [0, 1, 9, 11, 41, 43, 100, NUM_CELLS - 1]:
        assert h_start[c] == EMPTY_CELL, (
            f"cell_start[{c}] = {h_start[c]}, expected 0xFFFFFFFF"
        )
    print("[OK] All other sampled cells have start = 0xFFFFFFFF")


def test_sum_equals_num_particles() -> None:
    """Sum of (cell_end - cell_start) over non-empty cells equals num_particles."""
    print("\n--- Particle count consistency test ---")

    upload_grid_params()
    upload_hash_grid_params()

    rng = np.random.default_rng(99)
    n = 10_000
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.9, 0.9, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes)

    cell_start, cell_end = build_data_struct(sorted_hashes)
    cupy.cuda.Device().synchronize()

    h_start = cupy.asnumpy(cell_start)
    h_end = cupy.asnumpy(cell_end)

    # Non-empty cells
    non_empty = h_start != EMPTY_CELL
    total = np.sum(h_end[non_empty].astype(np.int64) - h_start[non_empty].astype(np.int64))

    assert total == n, f"Sum of (end - start) = {total}, expected {n}"
    print(f"[OK] Sum of (cell_end - cell_start) across {np.sum(non_empty)} "
          f"non-empty cells = {total} == {n} particles")


def test_cell_boundaries_match_sorted_hashes() -> None:
    """Verify cell_start/end indices point to correct hash values."""
    print("\n--- Cell boundary validation test ---")

    upload_grid_params()
    upload_hash_grid_params()

    rng = np.random.default_rng(77)
    n = 5_000
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.9, 0.9, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes)

    cell_start, cell_end = build_data_struct(sorted_hashes)
    cupy.cuda.Device().synchronize()

    h_start = cupy.asnumpy(cell_start)
    h_end = cupy.asnumpy(cell_end)
    h_sorted = cupy.asnumpy(sorted_hashes)

    # For each non-empty cell, check the sorted hash at start/end indices
    non_empty_cells = np.where(h_start != EMPTY_CELL)[0]
    for cell_hash in non_empty_cells:
        s = h_start[cell_hash]
        e = h_end[cell_hash]
        # All hashes in [s, e) should equal cell_hash
        assert np.all(h_sorted[s:e] == cell_hash), (
            f"Cell {cell_hash}: hashes in [{s},{e}) are not all {cell_hash}"
        )
        # hash before start (if any) should be different
        if s > 0:
            assert h_sorted[s - 1] != cell_hash, (
                f"Cell {cell_hash}: hash at {s-1} is still {cell_hash}"
            )
        # hash after end (if any) should be different
        if e < n:
            assert h_sorted[e] != cell_hash, (
                f"Cell {cell_hash}: hash at {e} is still {cell_hash}"
            )

    print(f"[OK] All {len(non_empty_cells)} non-empty cells have correct boundaries")


def test_memset_between_frames() -> None:
    """Calling build_data_struct twice resets cell_start properly."""
    print("\n--- Memset between frames test ---")

    # Frame 1: particles in cells 10 and 20
    sorted_hashes_1 = cupy.array(
        [10, 10, 10, 20, 20], dtype=cupy.uint32
    )
    cell_start, cell_end = allocate_cell_tables()
    build_data_struct(sorted_hashes_1, cell_start, cell_end)
    cupy.cuda.Device().synchronize()

    h_start_1 = cupy.asnumpy(cell_start)
    assert h_start_1[10] == 0
    assert h_start_1[20] == 3

    # Frame 2: particles only in cell 30 (cells 10, 20 should reset to empty)
    sorted_hashes_2 = cupy.array([30, 30, 30, 30], dtype=cupy.uint32)
    build_data_struct(sorted_hashes_2, cell_start, cell_end)
    cupy.cuda.Device().synchronize()

    h_start_2 = cupy.asnumpy(cell_start)
    assert h_start_2[10] == EMPTY_CELL, (
        f"Cell 10 not reset: {h_start_2[10]}"
    )
    assert h_start_2[20] == EMPTY_CELL, (
        f"Cell 20 not reset: {h_start_2[20]}"
    )
    assert h_start_2[30] == 0
    print("[OK] cell_start properly reset between frames via memset")


def test_500k_no_errors() -> None:
    """Kernel runs without errors for 500K particles."""
    print("\n--- 500K particle stress test ---")

    upload_grid_params()
    upload_hash_grid_params()

    rng = np.random.default_rng(123)
    n = 500_000
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes)

    cell_start, cell_end = build_data_struct(sorted_hashes)
    cupy.cuda.Device().synchronize()

    h_start = cupy.asnumpy(cell_start)
    h_end = cupy.asnumpy(cell_end)

    # Sum check
    non_empty = h_start != EMPTY_CELL
    total = np.sum(h_end[non_empty].astype(np.int64) - h_start[non_empty].astype(np.int64))
    assert total == n, f"Sum = {total}, expected {n}"

    num_non_empty = int(np.sum(non_empty))
    num_empty = NUM_CELLS - num_non_empty
    print(f"[OK] 500K particles: {num_non_empty} non-empty cells, "
          f"{num_empty} empty cells, sum(end-start) = {total}")


def main() -> None:
    test_compilation()
    test_allocation()
    test_block_size()
    test_empty_cells_sentinel()
    test_known_8_particles_2_cells()
    test_sum_equals_num_particles()
    test_cell_boundaries_match_sorted_hashes()
    test_memset_between_frames()
    test_500k_no_errors()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
