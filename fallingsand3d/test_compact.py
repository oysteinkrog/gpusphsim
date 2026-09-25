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
    # Kill particles in center. kill_in_sphere no longer returns a count
    # (bd-mzc.39 removed the GPU sync); measure via num_active delta instead.
    active_pre_kill = w.num_active
    w.kill_in_sphere((0.0, 0.0, 0.0), 0.1)
    alive_before = w.num_active
    killed = active_pre_kill - alive_before
    assert killed > 0

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

    # Kill ~10K particles (inner sphere). kill_in_sphere no longer returns a
    # count (bd-mzc.39); measure via num_active delta instead.
    active_pre_kill = w.num_active
    w.kill_in_sphere((0.0, 0.0, 0.0), 0.32)
    alive = w.num_active
    killed = active_pre_kill - alive
    assert killed >= 5_000  # inner sphere should have many particles

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

    for name in ["position", "velocity", "mass", "packed_info", "temperature",
                 "health", "lifetime", "sleep_counter", "kappa", "kappa_v"]:
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


# ---------------------------------------------------------------------------
# Fused compaction (bd-r5-review-n28.149)
# ---------------------------------------------------------------------------

from world import _COMPACT_FIELDS, DEAD_PACKED  # noqa: E402


def _random_scene(max_particles, n, dead_frac, seed=149):
    """World with n slots of random bits in every compacted field.

    Floats get random 32-bit patterns (NaN payloads included), so the check
    is on raw bytes. dead_frac of the slots get material 0 with random
    upper bits; the rest get material 1..31.
    """
    rng = np.random.default_rng(seed)
    w = World(max_particles)
    for f in _COMPACT_FIELDS:
        arr = getattr(w, f)
        nbytes = arr[:n].nbytes
        raw = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
        arr[:n] = cp.asarray(raw).view(arr.dtype).reshape(arr[:n].shape)
    pk = rng.integers(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)
    pk = (pk & np.uint32(0xFFFFFF00)) | rng.integers(1, 32, size=n).astype(np.uint32)
    dead = rng.random(n) < dead_frac
    pk[dead] &= np.uint32(0xFFFFFF00)
    w.packed_info[:n] = cp.asarray(pk)
    w._high_water = n
    return w, dead


def test_fused_compact_parity_bit_exact():
    """Every field after compact() equals a masked gather, byte for byte (10% dead)."""
    n = 50_000
    w, dead = _random_scene(60_000, n, 0.10)
    alive = ~dead
    expected = {f: cp.asnumpy(getattr(w, f)[:n])[alive] for f in _COMPACT_FIELDS}
    num_alive = int(alive.sum())

    assert w.compact(num_alive=num_alive) == num_alive
    assert w._high_water == num_alive
    for f in _COMPACT_FIELDS:
        got = cp.asnumpy(getattr(w, f)[:num_alive])
        assert got.shape == expected[f].shape, f
        assert got.tobytes() == expected[f].tobytes(), f"{f} differs"
    # Dead tail markers
    assert (cp.asnumpy(w.packed_info[num_alive:n]) == DEAD_PACKED).all()
    assert (cp.asnumpy(w.kappa[num_alive:n]) == 0).all()
    assert (cp.asnumpy(w.kappa_v[num_alive:n]) == 0).all()


def test_fused_compact_counts_when_not_given():
    """compact() with no count reads it from the scan and gives the same result."""
    n = 20_000
    w, dead = _random_scene(20_000, n, 0.25, seed=7)
    w2, _ = _random_scene(20_000, n, 0.25, seed=7)
    num_alive = int((~dead).sum())
    assert w.compact() == num_alive
    assert w2.compact(num_alive=num_alive) == num_alive
    for f in _COMPACT_FIELDS:
        assert (cp.asnumpy(getattr(w, f)[:num_alive]).tobytes()
                == cp.asnumpy(getattr(w2, f)[:num_alive]).tobytes()), f


def test_fused_compact_swaps_buffers():
    """compact() swaps primary and scratch arrays and bumps buffer_generation."""
    w, dead = _random_scene(10_000, 10_000, 0.1)
    before = {f: getattr(w, f).data.ptr for f in _COMPACT_FIELDS}
    before_sorted = {f: getattr(w, "sorted_" + f).data.ptr for f in _COMPACT_FIELDS}
    gen = w.buffer_generation
    w.compact(num_alive=int((~dead).sum()))
    assert w.buffer_generation == gen + 1
    for f in _COMPACT_FIELDS:
        assert getattr(w, f).data.ptr == before_sorted[f], f
        assert getattr(w, "sorted_" + f).data.ptr == before[f], f


_SPIN_SRC = r'''
extern "C" __global__ void spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
'''


def test_fused_compact_no_host_sync_with_count():
    """compact(num_alive=k) with no rigid block does not wait for the GPU.

    A spin kernel keeps the stream busy for about 0.5 s. If compact() read
    anything back, the host call would block until the spin ends.
    """
    import time
    n = 100_000
    w, dead = _random_scene(100_000, n, 0.1)
    num_alive = int((~dead).sum())
    w2, _ = _random_scene(100_000, n, 0.1)
    w2.compact(num_alive=num_alive)  # warm up: compile K_Compact and cumsum
    cp.cuda.Device().synchronize()

    spin = cp.RawKernel(_SPIN_SRC, "spin")
    clock_khz = cp.cuda.runtime.getDeviceProperties(
        cp.cuda.Device().id)["clockRate"]
    spin((1,), (1,), (np.int64(clock_khz * 500),))  # ~500 ms
    t0 = time.perf_counter()
    w.compact(num_alive=num_alive)
    host_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    cp.cuda.Device().synchronize()
    wait_ms = (time.perf_counter() - t1) * 1e3
    assert wait_ms > 100, f"spin kernel too short to detect a sync ({wait_ms:.0f} ms)"
    assert host_ms < 100, f"compact() blocked for {host_ms:.0f} ms"


def test_compact_invalidates_cuda_graph_at_same_n():
    """After compact() swaps buffers, a graph captured at the same n is not replayed."""
    import time
    from simulation import Simulation
    w = World(max_particles=30_000)
    w.spawn_cube(min_corner=(-0.2, -0.9, -0.2), max_corner=(0.2, -0.7, 0.2),
                 material_id=SAND, spacing=0.02)
    sim = Simulation(w, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=True,
                     max_substeps=2)
    try:
        sim.fixed_dt_value = 0.001
        sim._upload_dt(0.001)
        sim._spawn_substep = sim._damping_duration
        sim.compact_interval = 0  # no scheduled compaction in this test
        sim.notify_scene_changed()
        for _ in range(6):
            if sim._last_frame_time is None:
                sim.step_frame()
            sim._last_frame_time = time.perf_counter() - 1.0
            sim.step_frame()
            cp.cuda.Device().synchronize()
        old = [g for g in (sim._graph_full_sort, sim._graph_gather_only) if g is not None]
        assert old, "no CUDA graph captured"
        n = w._high_water
        w.packed_info[:10] = cp.uint32(DEAD_PACKED)
        assert w.compact(num_alive=n - 10) == n - 10
        assert w.add_dead_reserve(10) == 10
        assert w._high_water == n
        sim.notify_scene_changed()
        sim._sim_step(n)
        cp.cuda.Device().synchronize()
        for g in (sim._graph_full_sort, sim._graph_gather_only):
            assert all(g is not o for o in old), "stale CUDA graph replayed"
        assert bool(cp.isfinite(w.position[:n, :3]).all())
    finally:
        sim.close()


def test_fused_compact_timing_1m():
    """compact() at 1M with 10% dead: GPU time near 0.5 ms (CUDA events).

    Timing only: the GPU may be shared, so the best of several runs is used.
    """
    n = 1_000_000
    w, dead = _random_scene(n, n, 0.1)
    num_alive = int((~dead).sum())
    best = float("inf")
    for i in range(5):
        # Restore the uncompacted state for each run.
        if i:
            w, _ = _random_scene(n, n, 0.1)
        cp.cuda.Device().synchronize()
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        w.compact(num_alive=num_alive)
        stop.record()
        stop.synchronize()
        best = min(best, cp.cuda.get_elapsed_time(start, stop))
    print(f"fused compact 1M, 10% dead: best {best:.3f} ms GPU")
    # Acceptance target is 0.5 ms (about the bandwidth floor at 1M). The
    # assert allows 2x for a GPU shared with other work.
    assert best < 1.0, f"compact() took {best:.3f} ms"


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
    test_compact_dead_with_nonzero_packed_info()
    test_compact_10k_kill_and_render()
    test_compact_spawn_after_compact()
    test_compact_contiguous_after()
    test_compact_repeated()
    test_compact_stress_500k()
    test_num_active_with_dead_static()
    print("\nALL TESTS PASSED")
