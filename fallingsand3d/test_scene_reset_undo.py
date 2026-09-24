"""R reset and undo handle rigid bodies (bd-r5-review-n28.39).

The R key must drop the preset's rigid bodies and SDF objects, and undo
snapshots must rewind the rigid body state together with the fluid.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Shared reset helper (no GPU needed)
# ---------------------------------------------------------------------------

class _Spy:
    def __init__(self, log, name):
        self._log = log
        self._name = name

    def __getattr__(self, attr):
        def call(*args, **kwargs):
            self._log.append(f"{self._name}.{attr}")
        return call


def test_reset_scene_objects_clears_sdf_and_rigid_state():
    from simulation import Simulation

    log: list[str] = []

    class _FakeSim:
        sdf_manager = _Spy(log, "sdf")
        rigid_body_manager = _Spy(log, "rbm")

        def get_all_modules(self):
            return []

    Simulation.reset_scene_objects(_FakeSim())
    assert log == ["sdf.clear", "rbm.reset", "rbm.upload"]


def test_all_scene_reset_paths_use_the_shared_helper():
    """R, solver switch and preset load must not drift apart."""
    src = open(os.path.join(os.path.dirname(__file__), "main.py"), encoding="utf-8").read()
    assert src.count("sim.reset_scene_objects()") == 3
    assert "sim.rigid_body_manager.reset()" not in src
    assert "sim.sdf_manager.clear()" not in src


# ---------------------------------------------------------------------------
# GPU tests with a real World and RigidBodyManager
# ---------------------------------------------------------------------------

cp = pytest.importorskip("cupy")

from materials import MAT_RIGID, WATER  # noqa: E402


def _make_sim():
    from simulation import Simulation
    from world import World

    world = World(60_000)
    world.spawn_cube((-0.3, -0.3, -0.3), (0.3, 0.0, 0.3), WATER, 0.03)
    sim = Simulation(world, dt=0.001, fixed_dt=True)
    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(
        position=(0.0, 0.2, 0.0),
        half_extents=(0.05, 0.05, 0.05),
        mass=2.0,
    )
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    return world, sim, rbm


def _step(sim, k: int = 1) -> None:
    for _ in range(k):
        sim._sim_step(sim.world._high_water)
    cp.cuda.Device().synchronize()


def _download(rbm) -> np.ndarray:
    nb = rbm.num_bodies
    host = np.empty(nb, dtype=rbm._bodies_cpu.dtype)
    cp.cuda.Device().synchronize()
    rbm.d_rigid_bodies.data.copy_to_host(host.ctypes.data, host.nbytes)
    return host


def _upload(rbm, host: np.ndarray) -> None:
    rbm.d_rigid_bodies.data.copy_from_host(host.ctypes.data, host.nbytes)
    cp.cuda.Device().synchronize()


def _block_centroid(world, rbm) -> np.ndarray:
    off, nb = rbm.boundary_offset, rbm.num_boundary_particles
    assert off >= 0
    return world.position[off:off + nb, :3].get().mean(axis=0)


def test_r_reset_removes_rigid_bodies():
    """Mirror of the R key handler in main.py."""
    world, sim, rbm = _make_sim()
    _step(sim)
    assert rbm.num_bodies == 1 and rbm.boundary_injected

    world.packed_info[:] = 0
    world._high_water = 0
    world.foam_count.fill(0)
    sim.reset_scene_objects()
    world.spawn_cube((-0.2, 0.3, -0.2), (0.2, 0.7, 0.2), WATER, 0.03)
    n = world._high_water
    sim.notify_scene_changed()
    _step(sim)

    assert rbm.num_bodies == 0
    assert rbm.boundary_offset == -1
    assert world._high_water == n
    mats = world.packed_info[:world._high_water].get() & 0xFF
    assert int(np.count_nonzero(mats == MAT_RIGID)) == 0


def test_undo_restores_rigid_body_pose():
    from snapshot import SnapshotRing

    world, sim, rbm = _make_sim()
    _step(sim, 3)
    ring = SnapshotRing(max_snapshots=4, max_particles=world.max_particles)
    ring.capture(world, sim=sim)
    captured = _download(rbm)
    cap_pos = np.array(captured[0]["position"][:3], dtype=np.float64)

    # Move the body 0.5 m sideways behind the snapshot's back.
    moved = captured.copy()
    moved[0]["position"][0] += 0.5
    _upload(rbm, moved)
    _step(sim)
    assert abs(_download(rbm)[0]["position"][0] - cap_pos[0]) > 0.4

    assert ring.restore(world, sim=sim)
    np.testing.assert_array_equal(rbm._bodies_cpu[0]["position"], captured[0]["position"])
    np.testing.assert_array_equal(_download(rbm)[0]["position"], captured[0]["position"])

    # Same as the Ctrl+Z handler: rebuild the boundary block next substep.
    rbm.mark_for_reinjection()
    sim.notify_scene_changed()
    _step(sim)
    centroid = _block_centroid(world, rbm)
    # One substep at dt=1 ms moves the body far less than 1 cm.
    np.testing.assert_allclose(centroid, cap_pos, atol=0.01)
    mats = world.packed_info[:world._high_water].get() & 0xFF
    assert int(np.count_nonzero(mats == MAT_RIGID)) == rbm.num_boundary_particles


def test_undo_skips_rigid_restore_when_body_count_changed(capsys):
    from snapshot import SnapshotRing

    world, sim, rbm = _make_sim()
    _step(sim)
    ring = SnapshotRing(max_snapshots=2, max_particles=world.max_particles)
    ring.capture(world, sim=sim)

    rbm.reset()
    assert ring.restore(world, sim=sim)
    assert rbm.num_bodies == 0
    assert "rigid bodies were not restored" in capsys.readouterr().out


def test_capture_without_sim_keeps_no_rigid_state():
    from snapshot import SnapshotRing

    world, sim, rbm = _make_sim()
    _step(sim)
    ring = SnapshotRing(max_snapshots=2, max_particles=world.max_particles)
    ring.capture(world)
    before = _download(rbm)
    assert ring.restore(world, sim=sim)
    np.testing.assert_array_equal(_download(rbm), before)
