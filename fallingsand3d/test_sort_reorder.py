"""Integration tests for US-009: CuPy radix sort and data reorder.

Acceptance criteria:
  - Sort produces sorted_hash and sorted_index arrays (sorted by hash key)
    using cupy.argsort
  - All SoA arrays gathered into sorted-order temporary buffers using sorted_index
  - sorted_index (sorted->original mapping) preserved for Integrate writeback
  - Test: after sort, consecutive particles have same or increasing hash values
  - Test: data integrity preserved -- sum of all masses before sort equals sum after sort
  - Pre-allocated temporary buffers for sorted copies (no per-frame allocations)
  - Sort + reorder completes without errors for 500K particles

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
    upload_grid_params,
)
from fused_reorder import fused_reorder, get_module as get_reorder_module
from world import World


def test_sort_produces_sorted_hashes() -> None:
    """After sort, consecutive particles have same or increasing hash values."""
    print("--- Sort produces sorted hashes ---")

    upload_grid_params()

    rng = np.random.default_rng(42)
    n = 10_000
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.9, 0.9, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes, indices = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes, indices)
    cupy.cuda.Device().synchronize()

    h_sorted = cupy.asnumpy(sorted_hashes)

    # Verify non-decreasing order
    diffs = np.diff(h_sorted.astype(np.int64))
    assert np.all(diffs >= 0), (
        f"Sort failed: {np.sum(diffs < 0)} inversions found"
    )
    print(f"[OK] {n} particles sorted: hashes non-decreasing, "
          f"min={h_sorted[0]}, max={h_sorted[-1]}")

    # All hashes still valid
    assert h_sorted.min() >= 0
    assert h_sorted.max() < NUM_CELLS
    print("[OK] All sorted hashes in valid range [0, NUM_CELLS)")


def test_sorted_index_maps_correctly() -> None:
    """sorted_index[sorted_slot] = original_id, verified by reconstruction."""
    print("\n--- sorted_index maps correctly ---")

    upload_grid_params()

    n = 5_000
    rng = np.random.default_rng(99)
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.8, 0.8, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes, indices = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes, indices)
    cupy.cuda.Device().synchronize()

    h_hashes = cupy.asnumpy(hashes)
    h_sorted_hashes = cupy.asnumpy(sorted_hashes)
    h_sorted_indices = cupy.asnumpy(sorted_indices)

    # Verify: sorted_hashes[i] == hashes[sorted_indices[i]]
    reconstructed = h_hashes[h_sorted_indices]
    np.testing.assert_array_equal(h_sorted_hashes, reconstructed)
    print(f"[OK] sorted_hashes == hashes[sorted_indices] for all {n} particles")

    # Verify sorted_indices is a permutation of [0, n)
    assert sorted(h_sorted_indices.tolist()) == list(range(n)), (
        "sorted_indices is not a valid permutation"
    )
    print(f"[OK] sorted_indices is a valid permutation of [0, {n})")


def test_sort_with_preallocated_buffers() -> None:
    """sort_by_hash works with pre-allocated output buffers."""
    print("\n--- Sort with pre-allocated buffers ---")

    upload_grid_params()

    n = 8_000
    rng = np.random.default_rng(7)
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    hashes, indices = calc_hash(pos_gpu)

    # Pre-allocate
    sorted_hashes_buf = cupy.empty(n, dtype=cupy.uint32)
    sorted_indices_buf = cupy.empty(n, dtype=cupy.uint32)

    sorted_hashes, sorted_indices = sort_by_hash(
        hashes, indices,
        sorted_hashes_out=sorted_hashes_buf,
        sorted_indices_out=sorted_indices_buf,
    )
    cupy.cuda.Device().synchronize()

    h_sorted = cupy.asnumpy(sorted_hashes)
    diffs = np.diff(h_sorted.astype(np.int64))
    assert np.all(diffs >= 0), "Sort with pre-allocated buffers failed"
    print(f"[OK] sort_by_hash with pre-allocated buffers: {n} particles sorted")


def test_reorder_kernel_compilation() -> None:
    """fused_reorder.cu compiles via CuPy RawModule."""
    print("\n--- Reorder kernel compilation ---")

    module = get_reorder_module()
    assert module is not None
    print("[OK] fused_reorder.cu compiled (includes common.cuh)")

    kernel = module.get_function("K_FusedReorder")
    assert kernel is not None
    print("[OK] K_FusedReorder kernel function found")


def test_data_integrity_mass_sum() -> None:
    """Data integrity: sum of all masses before sort equals sum after sort."""
    print("\n--- Data integrity: mass sum preserved ---")

    upload_grid_params()

    n = 20_000
    rng = np.random.default_rng(55)

    # Create unsorted arrays
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.8, 0.8, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    mass_np = rng.uniform(0.001, 0.01, size=n).astype(np.float32)
    mass_gpu = cupy.asarray(mass_np)

    vel_gpu = cupy.asarray(rng.uniform(-1.0, 1.0, (n, 4)).astype(np.float32))
    veleval_gpu = cupy.asarray(rng.uniform(-0.5, 0.5, (n, 4)).astype(np.float32))
    packed_gpu = cupy.asarray(rng.integers(1, 256, size=n, dtype=np.uint32).astype(np.uint32))
    temp_gpu = cupy.asarray(rng.uniform(200, 1500, size=n).astype(np.float32))
    health_gpu = cupy.asarray(rng.uniform(0.0, 1.0, size=n).astype(np.float32))
    lifetime_gpu = cupy.asarray(rng.uniform(0.0, 5.0, size=n).astype(np.float32))
    color_gpu = cupy.asarray(rng.uniform(0.0, 1.0, (n, 4)).astype(np.float32))
    sleep_gpu = cupy.asarray(rng.integers(0, 10, size=n, dtype=np.uint8).astype(np.uint8))
    shear_gpu = cupy.asarray(rng.uniform(0.0, 100.0, size=n).astype(np.float32))

    # Hash and sort
    hashes, indices = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes, indices)

    # Pre-allocate sorted buffers
    s_pos = cupy.zeros((n, 4), dtype=cupy.float32)
    s_vel = cupy.zeros((n, 4), dtype=cupy.float32)
    s_veleval = cupy.zeros((n, 4), dtype=cupy.float32)
    s_mass = cupy.zeros(n, dtype=cupy.float32)
    s_packed = cupy.zeros(n, dtype=cupy.uint32)
    s_temp = cupy.zeros(n, dtype=cupy.float32)
    s_health = cupy.zeros(n, dtype=cupy.float32)
    s_lifetime = cupy.zeros(n, dtype=cupy.float32)
    s_color = cupy.zeros((n, 4), dtype=cupy.float32)
    s_sleep = cupy.zeros(n, dtype=cupy.uint8)
    s_shear = cupy.zeros(n, dtype=cupy.float32)

    # Fused reorder
    fused_reorder(
        n, sorted_indices,
        pos_gpu, vel_gpu, veleval_gpu, mass_gpu, packed_gpu,
        temp_gpu, health_gpu, lifetime_gpu, color_gpu, sleep_gpu, shear_gpu,
        s_pos, s_vel, s_veleval, s_mass, s_packed,
        s_temp, s_health, s_lifetime, s_color, s_sleep, s_shear,
    )
    cupy.cuda.Device().synchronize()

    # Check mass sum
    mass_sum_before = float(cupy.sum(mass_gpu))
    mass_sum_after = float(cupy.sum(s_mass))
    assert abs(mass_sum_before - mass_sum_after) < 1e-3, (
        f"Mass sum mismatch: before={mass_sum_before}, after={mass_sum_after}"
    )
    print(f"[OK] Mass sum preserved: before={mass_sum_before:.6f}, "
          f"after={mass_sum_after:.6f}")

    # Check temperature sum
    temp_sum_before = float(cupy.sum(temp_gpu))
    temp_sum_after = float(cupy.sum(s_temp))
    assert abs(temp_sum_before - temp_sum_after) / temp_sum_before < 1e-5, (
        f"Temp sum mismatch: before={temp_sum_before}, after={temp_sum_after}"
    )
    print(f"[OK] Temperature sum preserved: before={temp_sum_before:.1f}, "
          f"after={temp_sum_after:.1f}")

    # Check health sum
    health_sum_before = float(cupy.sum(health_gpu))
    health_sum_after = float(cupy.sum(s_health))
    assert abs(health_sum_before - health_sum_after) < 1e-3, (
        f"Health sum mismatch: before={health_sum_before}, after={health_sum_after}"
    )
    print("[OK] Health sum preserved")


def test_reorder_position_correctness() -> None:
    """Verify reorder gathers positions correctly using sorted_index."""
    print("\n--- Reorder position correctness ---")

    upload_grid_params()

    n = 1_000
    rng = np.random.default_rng(12)

    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    # Dummy arrays for the rest (we only check position)
    vel_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    veleval_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    mass_gpu = cupy.ones(n, dtype=cupy.float32)
    packed_gpu = cupy.ones(n, dtype=cupy.uint32)
    temp_gpu = cupy.zeros(n, dtype=cupy.float32)
    health_gpu = cupy.ones(n, dtype=cupy.float32)
    lifetime_gpu = cupy.zeros(n, dtype=cupy.float32)
    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    sleep_gpu = cupy.zeros(n, dtype=cupy.uint8)
    shear_gpu = cupy.zeros(n, dtype=cupy.float32)

    hashes, indices = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes, indices)

    s_pos = cupy.zeros((n, 4), dtype=cupy.float32)
    s_vel = cupy.zeros((n, 4), dtype=cupy.float32)
    s_veleval = cupy.zeros((n, 4), dtype=cupy.float32)
    s_mass = cupy.zeros(n, dtype=cupy.float32)
    s_packed = cupy.zeros(n, dtype=cupy.uint32)
    s_temp = cupy.zeros(n, dtype=cupy.float32)
    s_health = cupy.zeros(n, dtype=cupy.float32)
    s_lifetime = cupy.zeros(n, dtype=cupy.float32)
    s_color = cupy.zeros((n, 4), dtype=cupy.float32)
    s_sleep = cupy.zeros(n, dtype=cupy.uint8)
    s_shear = cupy.zeros(n, dtype=cupy.float32)

    fused_reorder(
        n, sorted_indices,
        pos_gpu, vel_gpu, veleval_gpu, mass_gpu, packed_gpu,
        temp_gpu, health_gpu, lifetime_gpu, color_gpu, sleep_gpu, shear_gpu,
        s_pos, s_vel, s_veleval, s_mass, s_packed,
        s_temp, s_health, s_lifetime, s_color, s_sleep, s_shear,
    )
    cupy.cuda.Device().synchronize()

    # Verify: sorted_pos[i] == pos[sorted_index[i]] for all i
    h_sorted_idx = cupy.asnumpy(sorted_indices)
    h_pos = cupy.asnumpy(pos_gpu)
    h_sorted_pos = cupy.asnumpy(s_pos)

    expected_sorted_pos = h_pos[h_sorted_idx]
    np.testing.assert_array_almost_equal(h_sorted_pos, expected_sorted_pos, decimal=6)
    print(f"[OK] Sorted positions match pos[sorted_indices] for all {n} particles")


def test_world_sorted_buffers_preallocated() -> None:
    """World class pre-allocates sorted_* temporary buffers at init."""
    print("\n--- World sorted buffer pre-allocation ---")

    world = World(max_particles=1000)

    # Check all sorted buffers exist and have correct shapes
    assert world.sorted_position.shape == (1000, 4)
    assert world.sorted_velocity.shape == (1000, 4)
    assert world.sorted_veleval.shape == (1000, 4)
    assert world.sorted_sph_force.shape == (1000, 4)
    assert world.sorted_color.shape == (1000, 4)
    assert world.sorted_density.shape == (1000,)
    assert world.sorted_mass.shape == (1000,)
    assert world.sorted_temperature.shape == (1000,)
    assert world.sorted_health.shape == (1000,)
    assert world.sorted_lifetime.shape == (1000,)
    assert world.sorted_shear_rate.shape == (1000,)
    assert world.sorted_packed_info.shape == (1000,)
    assert world.sorted_sleep_counter.shape == (1000,)
    print("[OK] All sorted_* buffers pre-allocated with shape (1000, ...)")

    # Check dtypes
    assert world.sorted_position.dtype == cupy.float32
    assert world.sorted_mass.dtype == cupy.float32
    assert world.sorted_packed_info.dtype == cupy.uint32
    assert world.sorted_sleep_counter.dtype == cupy.uint8
    print("[OK] sorted_* buffer dtypes correct")

    # Check hash/index buffers
    assert world.hashes.shape == (1000,)
    assert world.indices.shape == (1000,)
    assert world.sorted_hashes.shape == (1000,)
    assert world.sorted_indices.shape == (1000,)
    assert world.hashes.dtype == cupy.uint32
    print("[OK] Hash/index buffers pre-allocated")

    # After resize, buffers should be re-allocated
    world.resize(2000)
    assert world.sorted_position.shape == (2000, 4)
    assert world.sorted_mass.shape == (2000,)
    assert world.hashes.shape == (2000,)
    print("[OK] resize() re-allocates sorted buffers to new size")


def test_end_to_end_world_sort_reorder() -> None:
    """Full pipeline: spawn -> hash -> sort -> reorder using World buffers."""
    print("\n--- End-to-end World sort+reorder ---")

    upload_grid_params()

    world = World(max_particles=5000)
    spawned = world.spawn_sphere((0.0, 0.0, 0.0), 0.5, 5, 3000)  # water
    assert spawned > 0
    n = world._high_water
    print(f"  Spawned {n} water particles")

    # Hash
    hashes, indices = calc_hash(world.position[:n])

    # Sort into pre-allocated buffers
    sorted_hashes, sorted_indices = sort_by_hash(
        hashes, indices,
        sorted_hashes_out=world.sorted_hashes[:n],
        sorted_indices_out=world.sorted_indices[:n],
    )

    # Fused reorder into pre-allocated sorted buffers
    fused_reorder(
        n, world.sorted_indices[:n],
        world.position[:n], world.velocity[:n], world.veleval[:n],
        world.mass[:n], world.packed_info[:n],
        world.temperature[:n], world.health[:n], world.lifetime[:n],
        world.color[:n], world.sleep_counter[:n], world.shear_rate[:n],
        world.sorted_position[:n], world.sorted_velocity[:n],
        world.sorted_veleval[:n], world.sorted_mass[:n],
        world.sorted_packed_info[:n], world.sorted_temperature[:n],
        world.sorted_health[:n], world.sorted_lifetime[:n],
        world.sorted_color[:n], world.sorted_sleep_counter[:n],
        world.sorted_shear_rate[:n],
    )
    cupy.cuda.Device().synchronize()

    # Verify sorted hashes are non-decreasing
    h_sorted = cupy.asnumpy(world.sorted_hashes[:n])
    diffs = np.diff(h_sorted.astype(np.int64))
    assert np.all(diffs >= 0), "Sorted hashes not non-decreasing"
    print("[OK] Sorted hashes non-decreasing")

    # Verify mass sum preserved
    mass_before = float(cupy.sum(world.mass[:n]))
    mass_after = float(cupy.sum(world.sorted_mass[:n]))
    assert abs(mass_before - mass_after) < 1e-3
    print(f"[OK] Mass sum preserved: {mass_before:.6f} -> {mass_after:.6f}")

    # Verify sorted_indices is valid permutation
    h_idx = cupy.asnumpy(world.sorted_indices[:n])
    assert len(set(h_idx.tolist())) == n, "sorted_indices has duplicates"
    assert h_idx.min() >= 0 and h_idx.max() < n
    print(f"[OK] sorted_indices is valid permutation of [0, {n})")


def test_500k_sort_reorder_no_errors() -> None:
    """Sort + reorder completes without errors for 500K particles."""
    print("\n--- 500K particle sort+reorder stress test ---")

    upload_grid_params()

    n = 500_000
    rng = np.random.default_rng(77)

    # Create arrays directly (faster than World.spawn for stress test)
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.95, 0.95, size=(n, 3)).astype(np.float32)
    pos_gpu = cupy.asarray(pos_np)

    mass_gpu = cupy.full(n, 0.008, dtype=cupy.float32)
    vel_gpu = cupy.asarray(rng.uniform(-1, 1, (n, 4)).astype(np.float32))
    veleval_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    packed_gpu = cupy.full(n, 0x0005, dtype=cupy.uint32)  # water, FLUID
    temp_gpu = cupy.full(n, 293.0, dtype=cupy.float32)
    health_gpu = cupy.ones(n, dtype=cupy.float32)
    lifetime_gpu = cupy.zeros(n, dtype=cupy.float32)
    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    color_gpu[:, 2] = 1.0  # blue
    color_gpu[:, 3] = 1.0
    sleep_gpu = cupy.zeros(n, dtype=cupy.uint8)
    shear_gpu = cupy.zeros(n, dtype=cupy.float32)

    # Pre-allocate sorted buffers
    s_pos = cupy.zeros((n, 4), dtype=cupy.float32)
    s_vel = cupy.zeros((n, 4), dtype=cupy.float32)
    s_veleval = cupy.zeros((n, 4), dtype=cupy.float32)
    s_mass = cupy.zeros(n, dtype=cupy.float32)
    s_packed = cupy.zeros(n, dtype=cupy.uint32)
    s_temp = cupy.zeros(n, dtype=cupy.float32)
    s_health = cupy.zeros(n, dtype=cupy.float32)
    s_lifetime = cupy.zeros(n, dtype=cupy.float32)
    s_color = cupy.zeros((n, 4), dtype=cupy.float32)
    s_sleep = cupy.zeros(n, dtype=cupy.uint8)
    s_shear = cupy.zeros(n, dtype=cupy.float32)

    # Hash
    hashes, indices = calc_hash(pos_gpu)

    # Sort
    sorted_hashes, sorted_indices = sort_by_hash(hashes, indices)

    # Reorder
    fused_reorder(
        n, sorted_indices,
        pos_gpu, vel_gpu, veleval_gpu, mass_gpu, packed_gpu,
        temp_gpu, health_gpu, lifetime_gpu, color_gpu, sleep_gpu, shear_gpu,
        s_pos, s_vel, s_veleval, s_mass, s_packed,
        s_temp, s_health, s_lifetime, s_color, s_sleep, s_shear,
    )
    cupy.cuda.Device().synchronize()

    # Verify sorted hashes non-decreasing
    h_sorted = cupy.asnumpy(sorted_hashes)
    diffs = np.diff(h_sorted.astype(np.int64))
    assert np.all(diffs >= 0), f"500K sort failed: {np.sum(diffs < 0)} inversions"
    print(f"[OK] 500K hashes sorted: min={h_sorted[0]}, max={h_sorted[-1]}")

    # Verify mass sum preserved
    mass_sum_before = float(cupy.sum(mass_gpu))
    mass_sum_after = float(cupy.sum(s_mass))
    assert abs(mass_sum_before - mass_sum_after) < 0.01, (
        f"Mass sum mismatch: {mass_sum_before} vs {mass_sum_after}"
    )
    print(f"[OK] Mass sum preserved for 500K: {mass_sum_before:.3f} -> {mass_sum_after:.3f}")

    # Verify sorted_indices is valid
    h_idx = cupy.asnumpy(sorted_indices)
    assert h_idx.min() >= 0 and h_idx.max() < n
    print("[OK] sorted_indices range valid [0, 500K)")

    # Spot-check: sorted positions match gather
    h_pos = cupy.asnumpy(pos_gpu)
    h_spos = cupy.asnumpy(s_pos)
    sample = [0, 100, 1000, 10000, 100000, 499999]
    for i in sample:
        orig = h_idx[i]
        np.testing.assert_array_almost_equal(h_spos[i], h_pos[orig], decimal=6)
    print("[OK] Spot-check: sorted positions match gather for sample indices")

    print(f"[OK] 500K sort + reorder completed without errors")


def test_zero_particles() -> None:
    """Sort and reorder handle zero particles gracefully."""
    print("\n--- Zero particles edge case ---")

    upload_grid_params()

    pos_gpu = cupy.zeros((0, 4), dtype=cupy.float32)
    hashes = cupy.zeros(0, dtype=cupy.uint32)
    indices = cupy.zeros(0, dtype=cupy.uint32)

    sorted_hashes, sorted_indices = sort_by_hash(hashes, indices)
    assert sorted_hashes.shape[0] == 0
    assert sorted_indices.shape[0] == 0
    print("[OK] sort_by_hash handles 0 particles")

    # fused_reorder with num_particles=0 should be a no-op
    fused_reorder(
        0, cupy.zeros(0, dtype=cupy.uint32),
        cupy.zeros((0, 4), dtype=cupy.float32),
        cupy.zeros((0, 4), dtype=cupy.float32),
        cupy.zeros((0, 4), dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.uint32),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros((0, 4), dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.uint8),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros((0, 4), dtype=cupy.float32),
        cupy.zeros((0, 4), dtype=cupy.float32),
        cupy.zeros((0, 4), dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.uint32),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.float32),
        cupy.zeros((0, 4), dtype=cupy.float32),
        cupy.zeros(0, dtype=cupy.uint8),
        cupy.zeros(0, dtype=cupy.float32),
    )
    print("[OK] fused_reorder handles 0 particles")


def main() -> None:
    test_sort_produces_sorted_hashes()
    test_sorted_index_maps_correctly()
    test_sort_with_preallocated_buffers()
    test_reorder_kernel_compilation()
    test_data_integrity_mass_sum()
    test_reorder_position_correctness()
    test_world_sorted_buffers_preallocated()
    test_end_to_end_world_sort_reorder()
    test_500k_sort_reorder_no_errors()
    test_zero_particles()
    print("\n=== ALL US-009 CHECKS PASSED ===")


if __name__ == "__main__":
    main()
