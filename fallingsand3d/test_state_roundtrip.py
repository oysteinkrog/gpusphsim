"""End-to-end GPU round-trip tests for scene state (bd-r5-review-n28.74).

These run save_scene/load_scene and SnapshotRing capture/restore on a real
World and Simulation, so a state bug (_high_water, DEAD slots, sleep and
spawn bits, warm-start reset, colour, sim_time, rigid body reinjection)
fails here even when the CPU-stub tests in test_save_load.py and
test_snapshot.py pass.

Related end-to-end tests that already live elsewhere (kept there, not
duplicated here):
- spawn after rigid injection keeps body ids, and the compaction variant:
  test_rigid_boundary.py::test_brush_spawn_does_not_move_block and
  ::test_compaction_shifts_block_intact
- resize reallocates solver arrays: test_world.py::test_resize_reallocates_solver_arrays
- kill_in_sphere zeroes mass, health, velocity and colour:
  test_world.py::test_kill_in_sphere
- corrupt save leaves the world untouched:
  test_save_load.py::test_load_corrupt_file_leaves_world_untouched
- preset spawn boxes do not overlap: test_presets.py::test_preset_spawn_boxes_do_not_overlap
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device")
except Exception:  # pragma: no cover - CPU-only box
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from camera import OrbitCamera  # noqa: E402
from materials import MAT_RIGID, SAND, STONE, WATER  # noqa: E402

SLEEP_BIT = 1 << 10   # packed_info bit 10: is_sleeping (common.cuh)
SPAWN_FLAG = 1 << 11  # packed_info bit 11: spawn flag (common.cuh)

# Arrays save_scene writes, in the order of save_load._SCENE_ARRAYS.
SAVED = ("position", "velocity", "packed_info", "temperature", "health",
         "lifetime", "mass", "particle_dye", "density")
# Arrays a SnapshotRing slot holds.
SNAPPED = ("position", "velocity", "packed_info", "temperature", "health",
           "lifetime", "mass", "particle_dye")
# Slot fields stored as float16 (bd-r5-review-n28.113). A restore gives the
# float16 rounding of the captured value, still compared bit for bit.
SNAPPED_FP16 = ("health", "lifetime", "mass")


@pytest.fixture(scope="module", autouse=True)
def _compiled_kernels():
    """Compile the solver and rigid body kernels once per module.

    conftest.py keeps the CuPy kernel cache in memory, so the first
    Simulation in a pytest process compiles every module (about 20 s). Doing
    it here keeps that one-time cost in setup, and each test's own time is
    what the 10 s budget measures.
    """
    from simulation import Simulation
    from world import World

    world = World(20_000)
    world.spawn_cube((-0.2, -0.2, -0.2), (0.2, 0.0, 0.2), WATER, 0.04)
    sim = Simulation(world, dt=0.001, fixed_dt=True)
    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(
        position=(0.0, 0.2, 0.0), half_extents=(0.05, 0.05, 0.05), mass=2.0,
    )
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    sim._sim_step(world._high_water)
    cp.cuda.Device().synchronize()
    sim.close()


def _bits(a: np.ndarray) -> np.ndarray:
    """View as raw 32-bit words so the comparison is bit-exact (NaN, -0.0)."""
    return np.ascontiguousarray(a).view(np.uint32)


def _host_state(world, names) -> dict:
    n = world._high_water
    return {k: getattr(world, k)[:n].get() for k in names}


def _assert_state_equal(world, expected: dict, names, fp16=()) -> None:
    n = world._high_water
    for k in names:
        got = getattr(world, k)[:n].get()
        want = expected[k]
        if k in fp16:
            want = want.astype(np.float16).astype(np.float32)
        assert got.shape == want.shape, k
        assert np.array_equal(_bits(got), _bits(want)), k


def _mixed_world():
    """World + Simulation with every particle kind the state paths handle.

    WATER, a sleeping SAND (GRANULAR) block, one WATER particle with the
    spawn flag, a STONE (STATIC) block, parked DEAD reserve slots and one
    brush-killed slot in the middle of the live range.
    """
    from simulation import Simulation
    from world import World

    world = World(40_000)
    world.spawn_cube((-0.3, -0.3, -0.3), (0.0, 0.0, 0.0), WATER, 0.03)
    sand_lo = world._high_water
    world.spawn_cube((0.1, -0.3, -0.3), (0.3, -0.1, -0.1), SAND, 0.03)
    sand_hi = world._high_water
    world.sleep_counter[sand_lo:sand_hi] = 200
    assert world.spawn_sphere((0.0, 0.3, 0.0), 0.01, WATER, 1) == 1
    spawned = world._high_water - 1
    world.spawn_cube((-0.3, -0.5, 0.1), (0.3, -0.44, 0.3), STONE, 0.03)
    world.kill_in_sphere((-0.15, -0.15, -0.15), 0.02)
    assert world.add_dead_reserve(64) == 64

    sim = Simulation(world, dt=0.001, fixed_dt=True)
    # Set the bits and non-trivial per-particle values on the host: each
    # distinct CuPy expression is a fresh NVRTC compile under the in-memory
    # cache, and host writes keep the test inside its time budget.
    n = world._high_water
    packed = world.packed_info[:n].get()
    packed[sand_lo:sand_hi] |= np.uint32(SLEEP_BIT)
    packed[spawned] |= np.uint32(SPAWN_FLAG)
    world.packed_info[:n] = cp.asarray(packed)
    ramp = np.arange(n, dtype=np.float32)
    world.temperature[:n] = cp.asarray(world.temperature[:n].get() + ramp * np.float32(0.001))
    world.lifetime[:n] = cp.asarray(ramp * np.float32(0.01))
    world.density[:n] = cp.asarray(np.float32(2500.0) + ramp)
    dye = world.particle_dye[:n].get()
    dye[:, 0] = 0.25
    world.particle_dye[:n] = cp.asarray(dye)
    vel = world.velocity[:n].get()
    vel[:, 1] = -0.5
    world.velocity[:n] = cp.asarray(vel)
    return world, sim


def _scramble(world, sim, camera) -> None:
    """Change everything load or restore is supposed to put back."""
    world.spawn_cube((-0.2, 0.2, -0.2), (0.2, 0.4, 0.2), WATER, 0.03)
    n = world._high_water
    for k in ("position", "velocity", "particle_dye"):
        getattr(world, k)[:n] += cp.float32(1.5)
    for k in ("temperature", "health", "lifetime", "mass", "density"):
        getattr(world, k)[:n] = cp.float32(-7.0)
    world.packed_info[:n] = cp.uint32(STONE)
    world.kappa[:n] = 5.0
    world.kappa_v[:n] = 5.0
    world.sleep_counter[:n] = 255
    world.color[:] = 9.0
    world._spawned_material_ids = {99}
    sim.sim_time = 99.0
    camera.target = np.array([9.0, 9.0, 9.0], np.float32)
    camera.distance = 20.0
    camera.azimuth = -45.0
    camera.elevation = 80.0


def _assert_fresh_warm_start(world) -> None:
    """Loaded or restored slots start with no warm-start or sleep state."""
    n = world._high_water
    for k in ("kappa", "kappa_v", "sleep_counter"):
        assert not np.any(getattr(world, k)[:n].get()), k
    assert np.array_equal(_bits(world.veleval[:n].get()), _bits(world.velocity[:n].get()))
    # Tail beyond the restored range holds no live particle.
    assert not np.any(world.packed_info[n:].get())


def _assert_color_matches_material(world) -> None:
    """color.w carries behavior * 0.25, DEAD gets -1 (refresh_color)."""
    n = world._high_water
    packed = world.packed_info[:n].get()
    w = world.color[:n, 3].get()
    mat = packed & 0xFF
    beh = (packed >> 8) & 0x3
    expected = np.where(mat == 0, np.float32(-1.0), beh.astype(np.float32) * np.float32(0.25))
    assert np.array_equal(w, expected)


def test_save_load_roundtrip_live_world(tmp_path, monkeypatch):
    import save_load as sl

    monkeypatch.setattr(sl, "SAVE_DIR", str(tmp_path))
    world, sim = _mixed_world()
    camera = OrbitCamera(target=np.array([0.1, -0.2, 0.3], np.float32),
                         distance=2.5, azimuth=33.0, elevation=12.5)
    sim.sim_time = 1.25

    n = world._high_water
    expected = _host_state(world, SAVED)
    mats = expected["packed_info"] & 0xFF
    # The fixture really holds every kind the test claims to cover.
    assert np.any(mats == 0)
    assert np.any((expected["packed_info"] & SLEEP_BIT) != 0)
    assert int(np.count_nonzero(expected["packed_info"] & SPAWN_FLAG)) == 1
    assert {WATER, SAND, STONE} <= set(mats.tolist())
    expected_mats = {int(m) for m in np.unique(mats) if m != 0}

    sl.save_scene(world, sim, camera, "roundtrip")
    _scramble(world, sim, camera)
    assert sl.load_scene(world, sim, camera, "roundtrip") == n
    cp.cuda.Device().synchronize()

    assert world._high_water == n
    _assert_state_equal(world, expected, SAVED)
    assert world._spawned_material_ids == expected_mats
    assert sim.sim_time == 1.25
    assert np.array_equal(camera.target, np.array([0.1, -0.2, 0.3], np.float32))
    assert camera.distance == 2.5
    assert camera.azimuth == 33.0
    assert camera.elevation == 12.5
    _assert_fresh_warm_start(world)
    _assert_color_matches_material(world)


def test_snapshot_capture_restore_live_world():
    from snapshot import SnapshotRing

    world, sim = _mixed_world()
    camera = OrbitCamera()
    ring = SnapshotRing(max_snapshots=4, max_particles=world.max_particles)

    states = []
    for i in range(3):
        states.append((world._high_water, _host_state(world, SNAPPED),
                       set(world._spawned_material_ids)))
        ring.capture(world, sim=sim)
        # Grow and change the world between captures.
        world.spawn_cube((-0.2, 0.2 + 0.1 * i, -0.2), (0.2, 0.25 + 0.1 * i, 0.2),
                         SAND if i == 0 else WATER, 0.03)
        world.position[:world._high_water, 0] += cp.float32(0.01)
    _scramble(world, sim, camera)
    assert ring.num_snapshots == 3

    # Undo walks back one capture at a time: third, second, first. The last
    # two form the two-step rewind past a capture that grew the world.
    for hw, expected, mats in reversed(states):
        assert ring.restore(world, steps_back=1, sim=sim)
        cp.cuda.Device().synchronize()
        assert world._high_water == hw
        _assert_state_equal(world, expected, SNAPPED, SNAPPED_FP16)
        assert world._spawned_material_ids == mats
        _assert_fresh_warm_start(world)
        _assert_color_matches_material(world)
        _scramble(world, sim, camera)
    assert not ring.restore(world, steps_back=1, sim=sim)

    # restore(steps_back=2) jumps over one capture in a single call.
    world2, sim2 = _mixed_world()
    first = (world2._high_water, _host_state(world2, SNAPPED))
    ring2 = SnapshotRing(max_snapshots=4, max_particles=world2.max_particles)
    ring2.capture(world2, sim=sim2)
    world2.spawn_cube((-0.2, 0.2, -0.2), (0.2, 0.3, 0.2), WATER, 0.03)
    ring2.capture(world2, sim=sim2)
    _scramble(world2, sim2, camera)
    assert ring2.restore(world2, steps_back=2, sim=sim2)
    cp.cuda.Device().synchronize()
    assert world2._high_water == first[0]
    _assert_state_equal(world2, first[1], SNAPPED, SNAPPED_FP16)
    assert ring2.num_snapshots == 0


def _step(sim, k: int) -> None:
    for _ in range(k):
        sim._sim_step(sim.world._high_water)
    cp.cuda.Device().synchronize()


def test_load_scene_with_rigid_bodies_steps(tmp_path, monkeypatch):
    """Archimedes Pool: step, save, load, step. Bodies and block come back once."""
    import save_load as sl
    from presets import PRESETS
    from simulation import Simulation
    from world import World

    monkeypatch.setattr(sl, "SAVE_DIR", str(tmp_path))
    world = World(260_000)
    sim = Simulation(world, dt=0.001, fixed_dt=True)
    # Same sequence main.py runs for a preset load.
    sim.reset_scene_objects()
    PRESETS["Archimedes Pool"](world)
    rbm = sim.rigid_body_manager
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    sim.sdf_manager.upload_if_dirty()
    sim.notify_scene_changed()
    sim.reset_spawn_damping()
    assert rbm.num_bodies == 3
    nb = rbm.num_boundary_particles
    assert nb > 0

    _step(sim, 3)
    assert rbm.boundary_injected
    n_fluid = world._high_water - nb
    saved_bodies = rbm.download_bodies()
    fluid_before = _host_state(world, SAVED)
    keep = (fluid_before["packed_info"] & 0xFF) != MAT_RIGID
    sim_time = sim.sim_time

    camera = OrbitCamera()
    sl.save_scene(world, sim, camera, "pool")
    _scramble(world, sim, camera)
    assert sl.load_scene(world, sim, camera, "pool") == n_fluid
    assert sim.sim_time == sim_time
    assert world._high_water == n_fluid
    for k in SAVED:
        got = getattr(world, k)[:n_fluid].get()
        assert np.array_equal(_bits(got), _bits(fluid_before[k][keep])), k
    loaded = rbm.download_bodies()
    for key in ("position", "rotation", "lin_vel", "ang_vel"):
        np.testing.assert_allclose(loaded[key], saved_bodies[key], atol=1e-6, err_msg=key)

    _step(sim, 3)
    assert rbm.num_bodies == 3
    assert rbm.boundary_injected
    assert rbm.boundary_offset == n_fluid
    assert world._high_water == n_fluid + nb
    n = world._high_water
    mat = world.packed_info[:n].get() & 0xFF
    assert int(np.count_nonzero(mat == MAT_RIGID)) == nb
    assert np.all(mat[n_fluid:] == MAT_RIGID)
    assert np.all(np.isfinite(world.position[:n].get()))
    assert np.all(np.isfinite(world.velocity[:n].get()))
    assert np.all(np.isfinite(rbm.download_bodies()["position"]))
