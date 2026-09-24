"""Integration test for hash_sort.py -- K_CalcHash grid hashing kernel.

Acceptance criteria:
  - K_CalcHash computes grid cell: cell = int3((pos - grid_min) * grid_delta),
    clamped to valid range via integer wrap + table_mask
  - Spatial hash = (cx*P1 + cy*P2 + cz*P3) & table_mask (uint32 wrap)
  - The 27 cells of every 3x3x3 stencil land in 27 distinct buckets,
    including cells on the world edges (coordinate 0), so no neighbour
    is scanned twice (bd-r5-review-n28.29)
  - Grid parameters read from __constant__ GridParams c_grid
  - Grid sizing: grid_min=(-1,-1,-1), cell_size=h=0.04,
    grid_delta=(25,25,25), table_size=262144
  - CuPy RawModule compiles hash_sort.cu including common.cuh without errors
  - Test: 100K particles uniformly distributed in unit cube -> valid hashes [0, table_size-1]
  - Test: particles at/outside grid boundaries -> valid hashes (no OOB)
  - Kernel runs without errors for 500K particles

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
    CELL_SIZE,
    GRID_PARAMS_DTYPE,
    NUM_CELLS,
    TABLE_SIZE,
    TABLE_MASK,
    build_grid_params,
    calc_hash,
    get_module,
    upload_grid_params,
)

# Backward-compat: the old API exposed GRID_MIN/MAX/RES/DELTA as named constants.
# The new API uses a spatial hash with TABLE_SIZE/TABLE_MASK instead of a dense
# grid_res^3 layout.  Tests below use the new constants directly.
_GRID_MIN = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
_GRID_DELTA = np.array([1.0 / float(CELL_SIZE)] * 3, dtype=np.float32)


def _host_hash(pos: np.ndarray) -> np.ndarray:
    """Reference CPU implementation of the spatial hash for validation."""
    grid_min = _GRID_MIN
    grid_delta = _GRID_DELTA

    # calcGridCell: int3(floor((pos - grid_min) * grid_delta))
    # No clamping -- spatial hash wraps naturally via table_mask
    cell = np.floor((pos[:, :3] - grid_min) * grid_delta).astype(np.int32)

    return _model_hash(cell, TABLE_MASK).astype(np.uint32)


_PRIMES = (73856093, 19349669, 83492791)
_STENCIL = np.array([(dx, dy, dz) for dx in (-1, 0, 1)
                     for dy in (-1, 0, 1) for dz in (-1, 0, 1)], dtype=np.int64)


def _model_hash(cell: np.ndarray, mask: int) -> np.ndarray:
    """CPU copy of spatialHash(): (cx*P1 + cy*P2 + cz*P3) in uint32, then mask."""
    c = np.asarray(cell, dtype=np.int64)
    h = (c[..., 0] * _PRIMES[0] + c[..., 1] * _PRIMES[1]
         + c[..., 2] * _PRIMES[2]) & 0xFFFFFFFF
    return (h & mask).astype(np.int64)


def _xor_hash(cell: np.ndarray, mask: int) -> np.ndarray:
    """The retired XOR combiner, kept to show the defect the test guards."""
    c = np.asarray(cell, dtype=np.int64)
    h = (((c[..., 0] * _PRIMES[0]) & 0xFFFFFFFF)
         ^ ((c[..., 1] * _PRIMES[1]) & 0xFFFFFFFF)
         ^ ((c[..., 2] * _PRIMES[2]) & 0xFFFFFFFF))
    return (h & mask).astype(np.int64)


def _stencil_dup_count(hash_fn, cells: np.ndarray, mask: int) -> np.ndarray:
    """Per cell: 27 minus the number of distinct buckets its stencil hits."""
    hs = hash_fn(cells[:, None, :] + _STENCIL[None, :, :], mask)  # (N, 27)
    hs = np.sort(hs, axis=1)
    return (hs[:, 1:] == hs[:, :-1]).sum(axis=1)


def _stencil_test_cells() -> np.ndarray:
    """Several thousand cells: the default 50^3 world's edges and faces
    (coordinate 0 and 49), a random interior sample, and cells far outside
    the world including negative coordinates."""
    rng = np.random.default_rng(3)
    r = np.arange(50)
    edges = []
    for a in r:
        for b in (0, 49):
            for c in (0, 49):
                edges += [(a, b, c), (b, a, c), (b, c, a)]
    face = [(0, a, b) for a in r[::3] for b in r[::3]]
    face += [(a, 0, b) for a in r[::3] for b in r[::3]]
    face += [(a, b, 0) for a in r[::3] for b in r[::3]]
    interior = rng.integers(1, 49, size=(2000, 3))
    wide = rng.integers(-300, 300, size=(2000, 3))
    return np.concatenate([np.array(edges), np.array(face), interior, wide])


def test_grid_constants() -> None:
    """Verify grid sizing matches acceptance criteria."""
    print("--- Grid constant checks ---")

    assert abs(float(CELL_SIZE) - 0.04) < 1e-7
    assert TABLE_SIZE == 262144
    assert TABLE_MASK == TABLE_SIZE - 1
    assert NUM_CELLS == TABLE_SIZE  # backward-compat alias
    np.testing.assert_array_almost_equal(_GRID_MIN, [-1.0, -1.0, -1.0])
    inv_cs = 1.0 / float(CELL_SIZE)  # 25.0
    np.testing.assert_array_almost_equal(_GRID_DELTA, [inv_cs, inv_cs, inv_cs])
    print(f"[OK] Grid constants: grid_min=(-1,-1,-1) cell_size=0.04 "
          f"grid_delta=25 table_size={TABLE_SIZE} table_mask={TABLE_MASK:#010x}")


def test_grid_params_struct() -> None:
    """Verify GridParams numpy dtype layout matches common.cuh struct."""
    print("\n--- GridParams struct checks ---")

    # struct GridParams { float3 grid_min (12) + float3 grid_delta (12)
    #   + uint table_size (4) + uint table_mask (4) } = 32 bytes
    assert GRID_PARAMS_DTYPE.itemsize == 32, (
        f"GridParams size mismatch: {GRID_PARAMS_DTYPE.itemsize} != 32"
    )
    print(f"[OK] sizeof(GridParams) = {GRID_PARAMS_DTYPE.itemsize} bytes")

    params = build_grid_params()
    np.testing.assert_array_almost_equal(params[0]["grid_min"], [-1.0, -1.0, -1.0])
    np.testing.assert_array_almost_equal(params[0]["grid_delta"], [25.0, 25.0, 25.0])
    assert params[0]["table_size"] == TABLE_SIZE
    assert params[0]["table_mask"] == TABLE_MASK
    print("[OK] GridParams fields populated correctly")


def test_compilation() -> None:
    """Verify CuPy RawModule compiles hash_sort.cu including common.cuh."""
    print("\n--- Compilation check ---")

    module = get_module()
    assert module is not None
    print("[OK] CuPy RawModule compiled hash_sort.cu (includes common.cuh)")

    # Verify kernel function is accessible
    kernel = module.get_function("K_CalcHash")
    assert kernel is not None
    print("[OK] K_CalcHash kernel function found")

    # Verify constant memory symbol is accessible
    d_ptr = module.get_global("c_grid")
    assert int(d_ptr) != 0
    print("[OK] c_grid constant memory symbol found")


def test_known_position() -> None:
    """Verify hash for a particle with a known grid cell."""
    print("\n--- Known position test ---")

    upload_grid_params()

    # Position (0.02, 0.02, 0.02):
    #   cell = int3(floor((0.02 - (-1)) * 25), ...) = int3(floor(25.5), ...) = (25, 25, 25)
    #   hash = (25*73856093 + 25*19349669 + 25*83492791) & TABLE_MASK
    pos = np.array([[0.02, 0.02, 0.02, 0.0]], dtype=np.float32)
    pos_gpu = cupy.asarray(pos)

    hashes = calc_hash(pos_gpu)
    cupy.cuda.Device().synchronize()

    h = int(cupy.asnumpy(hashes)[0])
    ref = int(_host_hash(pos)[0])
    assert h == ref, f"hash = {h}, expected {ref}"
    assert 0 <= h < TABLE_SIZE, f"hash {h} out of range [0, {TABLE_SIZE})"
    print(f"[OK] Position (0.02, 0.02, 0.02) -> cell (25,25,25) -> hash = {h}")


def test_100k_uniform_unit_cube() -> None:
    """100K particles uniformly distributed in unit cube -> valid hashes."""
    print("\n--- 100K uniform unit-cube test ---")

    upload_grid_params()

    rng = np.random.default_rng(42)
    # Unit cube [0, 1]^3 is entirely inside grid [-1, 1]^3
    pos_np = np.zeros((100_000, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(0.0, 1.0, size=(100_000, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes = calc_hash(pos_gpu)
    cupy.cuda.Device().synchronize()

    h_hashes = cupy.asnumpy(hashes)

    # All hashes in [0, TABLE_SIZE - 1]
    assert h_hashes.min() >= 0, f"Min hash = {h_hashes.min()}, expected >= 0"
    assert h_hashes.max() < TABLE_SIZE, (
        f"Max hash = {h_hashes.max()}, expected < {TABLE_SIZE}"
    )
    print(f"[OK] All 100K hashes in [0, {TABLE_SIZE - 1}]: "
          f"min={h_hashes.min()}, max={h_hashes.max()}")

    # Cross-check against CPU reference
    ref = _host_hash(pos_np)
    np.testing.assert_array_equal(h_hashes, ref)
    print("[OK] GPU hashes match CPU reference for all 100K particles")


def test_boundary_clamping() -> None:
    """Particles at or outside grid boundaries produce valid hashes."""
    print("\n--- Boundary (out-of-range positions) test ---")

    upload_grid_params()

    # Test positions: inside, on boundary, and outside
    test_positions = np.array([
        # Inside (center of grid)
        [0.0, 0.0, 0.0, 0.0],
        # At grid_min exactly
        [-1.0, -1.0, -1.0, 0.0],
        # At grid_max exactly
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
    hashes = calc_hash(pos_gpu)
    cupy.cuda.Device().synchronize()

    h_hashes = cupy.asnumpy(hashes)

    # All hashes must be in valid range (spatial hash wraps via table_mask)
    for i, h in enumerate(h_hashes):
        assert 0 <= h < TABLE_SIZE, (
            f"Particle {i} at {test_positions[i, :3]}: hash={h}, "
            f"expected in [0, {TABLE_SIZE - 1}]"
        )
    print(f"[OK] All {len(test_positions)} test particles have valid hashes")

    # Cross-check all against CPU reference
    ref = _host_hash(test_positions)
    np.testing.assert_array_equal(h_hashes, ref)
    print("[OK] Hashes match CPU reference")


def test_500k_no_errors() -> None:
    """Kernel runs without errors for 500K particles."""
    print("\n--- 500K particle stress test ---")

    upload_grid_params()

    rng = np.random.default_rng(123)
    # Distribute across entire grid range [-1, 1]^3
    pos_np = np.zeros((500_000, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-1.0, 1.0, size=(500_000, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes = calc_hash(pos_gpu)
    cupy.cuda.Device().synchronize()

    h_hashes = cupy.asnumpy(hashes)

    # All valid
    assert h_hashes.min() >= 0
    assert h_hashes.max() < TABLE_SIZE
    print(f"[OK] 500K particles hashed without errors: "
          f"min={h_hashes.min()}, max={h_hashes.max()}")

    # Spot-check against CPU reference (first 1000)
    ref = _host_hash(pos_np[:1000])
    np.testing.assert_array_equal(h_hashes[:1000], ref)
    print("[OK] First 1000 hashes match CPU reference")


def test_stencil_buckets_distinct_model() -> None:
    """Every 27-cell stencil hits 27 distinct buckets at all table sizes
    compute_table_size can pick (2^18..2^20), edges at coordinate 0 included."""
    cells = _stencil_test_cells()
    assert len(cells) > 5000
    for bits in (18, 19, 20):
        mask = (1 << bits) - 1
        dups = _stencil_dup_count(_model_hash, cells, mask)
        assert int(dups.max()) == 0, (
            f"2^{bits}: {int((dups > 0).sum())} cells have a stencil bucket clash"
        )
    # The retired XOR combiner fails on the edges through the origin:
    # 6 duplicate offset pairs for (1,0,0), (0,1,0) and (0,0,1).
    edge = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.int64)
    np.testing.assert_array_equal(_stencil_dup_count(_xor_hash, edge, TABLE_MASK), 6)
    np.testing.assert_array_equal(_stencil_dup_count(_model_hash, edge, TABLE_MASK), 0)


def test_stencil_buckets_distinct_gpu() -> None:
    """Device hash of the 27 stencil cell centres of each test cell matches
    the model and gives 27 distinct buckets."""
    upload_grid_params()
    cells = _stencil_test_cells()
    nb = (cells[:, None, :] + _STENCIL[None, :, :]).reshape(-1, 3)
    pos = np.zeros((nb.shape[0], 4), dtype=np.float32)
    pos[:, :3] = (_GRID_MIN.astype(np.float64)
                  + (nb + 0.5) * float(CELL_SIZE)).astype(np.float32)
    hashes = cupy.asnumpy(calc_hash(cupy.asarray(pos))).astype(np.int64)
    np.testing.assert_array_equal(hashes, _model_hash(nb, TABLE_MASK))
    hs = np.sort(hashes.reshape(-1, 27), axis=1)
    clashes = int((hs[:, 1:] == hs[:, :-1]).any(axis=1).sum())
    assert clashes == 0, f"{clashes} cells have a stencil bucket clash on the GPU"


def test_step1_density_on_world_edges_matches_brute_force() -> None:
    """K_Step1 density of a lattice block in the world corner (cells with
    coordinate 0 on two axes) equals a brute-force poly6 sum. Under the XOR
    combiner the edge particles scanned cell (a,1,1) twice (+3 to +12%)."""
    import math

    import step1
    from conftest import counting_sort_scene, default_grid_params
    from materials import WATER, build_interaction_matrix, build_material_array

    h = float(CELL_SIZE)
    spacing = 0.02
    m = 0.008
    k = np.arange(10)
    gx, gy, gz = np.meshgrid(k, k, k, indexing="ij")
    lat = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    n = lat.shape[0]
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, :3] = (-1.0 + 0.5 * spacing + lat * spacing).astype(np.float32)
    pos[:, 3] = 1.0

    step1.upload_sim_params(step1.build_sim_params(
        smoothing_length=h, particle_mass=m, particle_spacing=spacing,
    ))
    step1.upload_precalc_params(step1.build_precalc_params(smoothing_length=h))
    step1.upload_materials(build_material_array())
    step1.upload_interactions(build_interaction_matrix())

    scene = counting_sort_scene(
        pos, velocity=np.zeros((n, 4), dtype=np.float32),
        mass=np.full(n, m, dtype=np.float32),
        packed_info=np.full(n, WATER, dtype=np.uint32),
        grid_params=default_grid_params(),
    )
    density, *_rest = step1.compute_step1(
        scene["position"], scene["velocity"], scene["mass"], None,
        scene["packed_info"], scene["temperature"],
        scene["cell_start"], scene["cell_end"],
        velocity_h=scene["velocity_h"],
    )
    cupy.cuda.Device().synchronize()
    rho = cupy.asnumpy(density).astype(np.float64)

    perm = cupy.asnumpy(scene["sort_perm"]).astype(np.int64)
    sp = pos[perm, :3].astype(np.float64)
    d2 = ((sp[:, None, :] - sp[None, :, :]) ** 2).sum(axis=2)
    w = np.where(d2 < h * h, (h * h - d2) ** 3, 0.0)
    poly6 = 315.0 / (64.0 * math.pi * h ** 9)
    rho_ref = np.maximum(1.0, poly6 * m * w.sum(axis=1))

    cells = np.floor((sp + 1.0) / h).astype(np.int64)
    on_edge = (cells == 0).sum(axis=1) >= 2
    assert on_edge.sum() > 0
    np.testing.assert_allclose(rho[on_edge], rho_ref[on_edge], rtol=2e-3)
    np.testing.assert_allclose(rho, rho_ref, rtol=2e-3)


def main() -> None:
    test_grid_constants()
    test_grid_params_struct()
    test_compilation()
    test_known_position()
    test_100k_uniform_unit_cube()
    test_boundary_clamping()
    test_500k_no_errors()
    test_stencil_buckets_distinct_model()
    test_stencil_buckets_distinct_gpu()
    test_step1_density_on_world_edges_matches_brute_force()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
