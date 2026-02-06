"""Tests for dead particle compaction (US-029).

Tests World.compact() and Simulation compaction scheduling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cupy as cp
import cupy.cuda.compiler as _compiler
import numpy as np

# Blackwell (sm_120) PTX workaround
_compiler._use_ptx = True
for _fn in (_compiler._get_arch, _compiler._get_arch_for_options_for_nvrtc):
    if hasattr(_fn, '_cache'):
        _fn._cache = {}

from world import World, _MAKE_PACKED
from materials import WATER, SAND, LAVA, STONE, FLUID, GRANULAR, STATIC


# ---------------------------------------------------------------------------
# World.compact() tests
# ---------------------------------------------------------------------------

def test_compact_no_dead():
    """Compaction with no dead particles is a no-op."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    assert n > 0
    hw_before = w._high_water
    result = w.compact()
    assert result == hw_before
    assert w._high_water == hw_before
    assert w.num_active == n
    print(f"PASS: compact no dead ({n} particles unchanged)")


def test_compact_all_dead():
    """Compaction when all particles are dead sets _high_water to 0."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    assert n > 0
    # Kill all particles
    w.packed_info[:n] = cp.uint32(0)
    assert w.num_active == 0
    result = w.compact()
    assert result == 0
    assert w._high_water == 0
    print(f"PASS: compact all dead ({n} -> 0)")


def test_compact_empty():
    """Compaction with zero particles is a no-op."""
    w = World(10_000)
    result = w.compact()
    assert result == 0
    assert w._high_water == 0
    print("PASS: compact empty world")


def test_compact_removes_dead():
    """Compaction moves alive particles to front and reduces _high_water."""
    w = World(10_000)
    n = w.spawn_cube((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2), WATER, 0.02)
    assert n > 100
    # Kill particles in center
    killed = w.kill_in_sphere((0.0, 0.0, 0.0), 0.1)
    assert killed > 0
    alive_before = w.num_active
    assert alive_before == n - killed

    result = w.compact()
    assert result == alive_before
    assert w._high_water == alive_before
    assert w.num_active == alive_before
    # No dead particles in the active range
    active_packed = w.packed_info[:w._high_water]
    assert int(cp.sum((active_packed & cp.uint32(0xFF)) == 0)) == 0
    print(f"PASS: compact removes dead ({n} -> {alive_before}, killed {killed})")


def test_compact_preserves_positions():
    """Compaction preserves all position data for alive particles."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    # Record alive positions before compaction
    alive_mask = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    positions_before = w.position[:n][alive_mask].copy()

    # Kill some particles (every other one)
    kill_indices = cp.arange(0, n, 2, dtype=cp.int64)
    w.packed_info[kill_indices] = cp.uint32(0)
    alive_after_kill = w.num_active

    # Record alive positions before compaction
    alive_mask2 = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    positions_before_compact = w.position[:n][alive_mask2].copy()

    w.compact()
    assert w._high_water == alive_after_kill

    # Check positions match
    positions_after = w.position[:w._high_water].get()
    expected = positions_before_compact.get()
    np.testing.assert_allclose(positions_after, expected, rtol=0, atol=0)
    print(f"PASS: compact preserves positions ({alive_after_kill} particles)")


def test_compact_preserves_velocities():
    """Compaction preserves velocity data for alive particles."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    # Set unique velocities
    w.velocity[:n, 0] = cp.arange(n, dtype=cp.float32) * 0.001
    w.velocity[:n, 1] = cp.arange(n, dtype=cp.float32) * -0.001

    # Kill every 3rd particle
    kill_indices = cp.arange(0, n, 3, dtype=cp.int64)
    w.packed_info[kill_indices] = cp.uint32(0)

    alive_mask = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    vel_before = w.velocity[:n][alive_mask].copy()

    w.compact()
    vel_after = w.velocity[:w._high_water]
    np.testing.assert_allclose(vel_after.get(), vel_before.get(), rtol=0, atol=0)
    print(f"PASS: compact preserves velocities ({w._high_water} particles)")


def test_compact_preserves_materials():
    """Compaction preserves packed_info (material type) for alive particles."""
    w = World(10_000)
    n_water = w.spawn_cube((-0.2, -0.2, -0.2), (0.0, 0.0, 0.0), WATER, 0.04)
    n_sand = w.spawn_cube((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), SAND, 0.04)
    n = n_water + n_sand
    assert n > 0

    # Record packed_info of all alive particles
    alive_mask = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    packed_before = w.packed_info[:n][alive_mask].copy()

    # Kill half the water particles
    kill_count = n_water // 2
    w.packed_info[:kill_count] = cp.uint32(0)

    alive_mask2 = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    packed_before_compact = w.packed_info[:n][alive_mask2].copy()

    w.compact()

    packed_after = w.packed_info[:w._high_water]
    np.testing.assert_array_equal(packed_after.get(), packed_before_compact.get())

    # Verify we still have both WATER and SAND particles
    mat_ids = packed_after.get() & 0xFF
    assert WATER in mat_ids
    assert SAND in mat_ids
    print(f"PASS: compact preserves materials (water={np.sum(mat_ids==WATER)}, sand={np.sum(mat_ids==SAND)})")


def test_compact_preserves_temperature():
    """Compaction preserves temperature for alive particles."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    # Set unique temperatures
    w.temperature[:n] = cp.arange(n, dtype=cp.float32) * 0.5 + 293.0

    # Kill every 4th particle
    kill_indices = cp.arange(0, n, 4, dtype=cp.int64)
    w.packed_info[kill_indices] = cp.uint32(0)

    alive_mask = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    temp_before = w.temperature[:n][alive_mask].copy()

    w.compact()
    temp_after = w.temperature[:w._high_water]
    np.testing.assert_allclose(temp_after.get(), temp_before.get(), rtol=0, atol=0)
    print(f"PASS: compact preserves temperature ({w._high_water} particles)")


def test_compact_preserves_mass():
    """Compaction preserves mass for alive particles."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)

    # Kill some
    killed = w.kill_in_sphere((0.0, 0.0, 0.0), 0.05)
    alive_mask = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    mass_before = w.mass[:n][alive_mask].copy()

    w.compact()
    mass_after = w.mass[:w._high_water]
    np.testing.assert_allclose(mass_after.get(), mass_before.get(), rtol=0, atol=0)
    print(f"PASS: compact preserves mass ({w._high_water} particles)")


def test_compact_preserves_health():
    """Compaction preserves health for alive particles."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    # Set varying health
    w.health[:n] = cp.linspace(0.1, 1.0, n, dtype=cp.float32)

    # Kill some
    kill_indices = cp.arange(0, n, 5, dtype=cp.int64)
    w.packed_info[kill_indices] = cp.uint32(0)

    alive_mask = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    health_before = w.health[:n][alive_mask].copy()

    w.compact()
    health_after = w.health[:w._high_water]
    np.testing.assert_allclose(health_after.get(), health_before.get(), rtol=0, atol=0)
    print(f"PASS: compact preserves health ({w._high_water} particles)")


def test_compact_preserves_sleep_counter():
    """Compaction preserves sleep_counter for alive particles."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    # Set varying sleep counters
    w.sleep_counter[:n] = cp.arange(n, dtype=cp.uint8) % 255

    # Kill some
    kill_indices = cp.arange(0, n, 3, dtype=cp.int64)
    w.packed_info[kill_indices] = cp.uint32(0)

    alive_mask = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    sc_before = w.sleep_counter[:n][alive_mask].copy()

    w.compact()
    sc_after = w.sleep_counter[:w._high_water]
    np.testing.assert_array_equal(sc_after.get(), sc_before.get())
    print(f"PASS: compact preserves sleep_counter ({w._high_water} particles)")


def test_compact_preserves_color():
    """Compaction preserves color for alive particles."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)

    # Kill some
    killed = w.kill_in_sphere((0.0, 0.0, 0.0), 0.05)
    alive_mask = (w.packed_info[:n] & cp.uint32(0xFF)) != 0
    color_before = w.color[:n][alive_mask].copy()

    w.compact()
    color_after = w.color[:w._high_water]
    np.testing.assert_allclose(color_after.get(), color_before.get(), rtol=0, atol=0)
    print(f"PASS: compact preserves color ({w._high_water} particles)")


def test_compact_dead_with_nonzero_packed_info():
    """Compaction correctly handles dead particles with packed_info=0x300 (MAKE_PACKED(0, STATIC))."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)

    # Set some particles to DEAD with non-zero packed_info (as Reactions kernel does)
    dead_packed = _MAKE_PACKED(0, STATIC)  # 0x300
    assert dead_packed != 0  # non-zero packed_info but dead (material_id=0)
    kill_indices = cp.arange(0, n, 2, dtype=cp.int64)
    w.packed_info[kill_indices] = cp.uint32(dead_packed)

    alive_before = w.num_active
    w.compact()
    assert w._high_water == alive_before
    assert w.num_active == alive_before
    # All particles in active range are truly alive (material_id != 0)
    mat_ids = w.packed_info[:w._high_water].get() & 0xFF
    assert np.all(mat_ids != 0)
    print(f"PASS: compact handles 0x300 dead particles ({alive_before} alive)")


def test_compact_10k_kill_and_render():
    """After killing 10K particles and compacting, only alive particles remain."""
    w = World(50_000)
    n = w.spawn_sphere((0.0, 0.0, 0.0), 0.5, WATER, 20_000)
    assert n == 20_000

    # Kill 10K particles (inner sphere)
    killed = w.kill_in_sphere((0.0, 0.0, 0.0), 0.32)
    assert killed >= 5_000  # inner sphere should have many particles
    alive = w.num_active

    w.compact()
    assert w._high_water == alive
    assert w.num_active == alive

    # All particles in [0, _high_water) have valid packed_info
    mat_ids = w.packed_info[:w._high_water].get() & 0xFF
    assert np.all(mat_ids != 0), "Dead particles found in active range after compaction"

    # Positions are all within original sphere (radius 0.5)
    pos = w.position[:w._high_water, :3].get()
    dist = np.sqrt(np.sum(pos**2, axis=1))
    assert np.all(dist <= 0.5 + 1e-4), "Particle outside original sphere"

    # And all should be OUTSIDE the kill sphere (radius 0.32)
    assert np.all(dist > 0.32 - 1e-4), "Dead particle position found in active range"
    print(f"PASS: 10K kill + compact (killed {killed}, {alive} remain)")


def test_compact_spawn_after_compact():
    """After compaction, new particles can be spawned into reclaimed slots."""
    w = World(1_000)
    n1 = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.04)
    assert n1 > 0

    # Kill half
    half = n1 // 2
    w.packed_info[:half] = cp.uint32(0)
    alive_after_kill = w.num_active

    w.compact()
    assert w._high_water == alive_after_kill

    # Spawn new particles - should succeed since slots were reclaimed
    available_before = w.max_particles - w._high_water
    n2 = w.spawn_sphere((0.0, 0.5, 0.0), 0.05, SAND, min(100, available_before))
    assert n2 > 0
    assert w.num_active == alive_after_kill + n2
    assert w._high_water == alive_after_kill + n2
    print(f"PASS: spawn after compact ({alive_after_kill} + {n2} = {w.num_active})")


def test_compact_contiguous_after():
    """After compaction, all arrays remain contiguous CuPy arrays."""
    w = World(10_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    w.kill_in_sphere((0.0, 0.0, 0.0), 0.05)
    w.compact()

    for name in ["position", "velocity", "veleval", "sph_force", "density",
                 "mass", "packed_info", "temperature", "health", "lifetime",
                 "shear_rate", "exposure_heat", "exposure_corrode", "color",
                 "sleep_counter"]:
        arr = getattr(w, name)
        assert isinstance(arr, cp.ndarray), f"{name} not CuPy"
        assert arr.flags["C_CONTIGUOUS"], f"{name} not C-contiguous"
    print("PASS: arrays contiguous after compact")


def test_compact_repeated():
    """Multiple compactions work correctly."""
    w = World(10_000)
    n = w.spawn_cube((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2), WATER, 0.02)

    for i in range(5):
        # Kill some particles
        killed = w.kill_in_sphere((0.0, 0.0, 0.0), 0.05 + i * 0.01)
        alive = w.num_active
        w.compact()
        assert w._high_water == alive
        assert w.num_active == alive
        # No dead in active range
        mat_ids = w.packed_info[:w._high_water].get() & 0xFF
        assert np.all(mat_ids != 0), f"Dead particles at iteration {i}"

    print(f"PASS: repeated compaction ({w._high_water} particles remain)")


def test_compact_stress_500k():
    """Stress test: compact 500K particles with scattered dead."""
    w = World(500_000)
    n = w.spawn_sphere((0.0, 0.0, 0.0), 0.8, WATER, 200_000)
    assert n == 200_000

    # Kill random 50% of particles
    kill_mask = cp.random.uniform(0, 1, n) < 0.5
    w.packed_info[:n][kill_mask] = cp.uint32(0)
    alive = w.num_active
    assert alive > 0

    w.compact()
    assert w._high_water == alive
    assert w.num_active == alive

    # Verify no dead in active range
    mat_ids = w.packed_info[:w._high_water].get() & 0xFF
    assert np.all(mat_ids != 0)

    # Verify positions are finite
    pos = w.position[:w._high_water].get()
    assert np.all(np.isfinite(pos))

    print(f"PASS: 500K stress test ({n} -> {alive} after random kill+compact)")


# ---------------------------------------------------------------------------
# num_active fix test (material_id check vs packed_info != 0)
# ---------------------------------------------------------------------------

def test_num_active_with_dead_static():
    """num_active correctly excludes DEAD particles with behavior=STATIC (packed_info=0x300)."""
    w = World(1_000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.04)
    assert n > 0
    alive_all = w.num_active
    assert alive_all == n

    # Mark half as DEAD via reactions-style (material_id=0, behavior=STATIC -> 0x300)
    half = n // 2
    dead_packed = _MAKE_PACKED(0, STATIC)  # 0x300
    w.packed_info[:half] = cp.uint32(dead_packed)

    alive_after = w.num_active
    assert alive_after == n - half, f"Expected {n - half}, got {alive_after}"
    print(f"PASS: num_active handles 0x300 dead ({n} -> {alive_after})")


if __name__ == "__main__":
    test_compact_no_dead()
    test_compact_all_dead()
    test_compact_empty()
    test_compact_removes_dead()
    test_compact_preserves_positions()
    test_compact_preserves_velocities()
    test_compact_preserves_materials()
    test_compact_preserves_temperature()
    test_compact_preserves_mass()
    test_compact_preserves_health()
    test_compact_preserves_sleep_counter()
    test_compact_preserves_color()
    test_compact_dead_with_nonzero_packed_info()
    test_compact_10k_kill_and_render()
    test_compact_spawn_after_compact()
    test_compact_contiguous_after()
    test_compact_repeated()
    test_compact_stress_500k()
    test_num_active_with_dead_static()
    print("\nALL TESTS PASSED")
