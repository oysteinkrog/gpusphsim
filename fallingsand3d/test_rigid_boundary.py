"""Rigid boundary block keeps a stable slot range (bd-r5-review-n28.22).

The boundary particles of all rigid bodies live in one contiguous block
[rbm.boundary_offset, +num_boundary_particles) inside world._high_water.
These GPU tests check that brush spawns, compaction, brush kill, solver
switch and reinjection all leave that block with the right packed_info,
body id and mass, and that it exists exactly once.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

from materials import MAT_RIGID, WATER  # noqa: E402


def _make_sim(profile_name: str | None = None, n_bodies: int = 2):
    from simulation import Simulation
    from world import World

    world = World(60_000)
    world.spawn_cube((-0.3, -0.3, -0.3), (0.3, 0.0, 0.3), WATER, 0.03)
    sim = Simulation(world, dt=0.001, fixed_dt=True)
    if profile_name is not None:
        from solver_profiles import PROFILES
        sim.set_solver_profile(PROFILES[profile_name])
    rbm = sim.rigid_body_manager
    for i in range(n_bodies):
        rbm.add_rigid_body_with_particles(
            position=(-0.15 + 0.3 * i, 0.2, 0.0),
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


def _check_block(world, rbm) -> None:
    """Every block slot is MAT_RIGID with the right body id and mass == psi."""
    off, nb = rbm.boundary_offset, rbm.num_boundary_particles
    assert off >= 0
    assert off + nb <= world._high_water
    packed = world.packed_info[off:off + nb].get()
    bd = rbm.boundary_data.get()
    assert np.all((packed & 0xFF) == MAT_RIGID)
    assert np.array_equal((packed >> 13) & 0xFF, bd[:, 4].astype(np.uint32))
    np.testing.assert_allclose(world.mass[off:off + nb].get(), bd[:, 3], rtol=0, atol=0)
    # Exactly one block: no other MAT_RIGID slot in the live range.
    all_mat = world.packed_info[:world._high_water].get() & 0xFF
    assert int(np.count_nonzero(all_mat == MAT_RIGID)) == nb


def test_first_substep_injects_inside_high_water():
    world, sim, rbm = _make_sim()
    n_fluid = world._high_water
    _step(sim)
    assert rbm.boundary_injected
    assert rbm.boundary_offset == n_fluid
    assert world._high_water == n_fluid + rbm.num_boundary_particles
    _check_block(world, rbm)


def test_brush_spawn_does_not_move_block():
    world, sim, rbm = _make_sim()
    _step(sim)
    off = rbm.boundary_offset
    hw = world._high_water
    k = world.spawn_sphere((0.0, 0.4, 0.0), 0.05, WATER, 200)
    assert k > 0
    assert world._high_water == hw + k  # spawn lands after the block
    sim.notify_scene_changed()
    _step(sim, 2)
    assert rbm.boundary_offset == off
    _check_block(world, rbm)


def test_compaction_shifts_block_intact():
    world, sim, rbm = _make_sim()
    _step(sim)
    off = rbm.boundary_offset
    # Kill 50 fluid slots below the block and 30 spawned slots above it.
    world.spawn_sphere((0.0, 0.4, 0.0), 0.05, WATER, 100)
    hw = world._high_water
    world.packed_info[10:60] = cp.uint32(0)
    world.packed_info[hw - 30:hw] = cp.uint32(0)
    world.compact()
    assert rbm.boundary_injected
    assert rbm.boundary_offset == off - 50
    _check_block(world, rbm)
    sim.notify_scene_changed()
    _step(sim, 2)
    _check_block(world, rbm)


def test_brush_kill_skips_rigid_particles():
    world, sim, rbm = _make_sim(n_bodies=1)
    _step(sim)
    # A sphere that covers the whole body and some fluid.
    world.kill_in_sphere((-0.15, 0.2, 0.0), 0.3)
    world.compact()
    assert rbm.boundary_injected
    _check_block(world, rbm)


def test_broken_block_is_rebuilt_once():
    world, sim, rbm = _make_sim()
    _step(sim)
    off = rbm.boundary_offset
    world.packed_info[off + 3] = cp.uint32(0)  # e.g. a reaction killed one slot
    world.compact()
    assert not rbm.boundary_injected
    _step(sim, 2)
    assert rbm.boundary_injected
    _check_block(world, rbm)


def test_set_solver_profile_then_add_body_injects():
    """F290: set_solver_profile used to leave the injection flag stuck."""
    world, sim, rbm = _make_sim(profile_name="WCSPH")
    _step(sim, 2)
    _check_block(world, rbm)


def test_mark_for_reinjection_does_not_duplicate():
    world, sim, rbm = _make_sim()
    _step(sim)
    hw = world._high_water
    rbm.mark_for_reinjection()
    _step(sim)
    assert world._high_water == hw
    _check_block(world, rbm)


def test_second_inject_without_mark_asserts():
    world, sim, rbm = _make_sim(n_bodies=1)
    _step(sim)
    with pytest.raises(AssertionError):
        rbm.inject_boundary_particles(world)


def test_reset_clears_flag_and_offset():
    world, sim, rbm = _make_sim(n_bodies=1)
    _step(sim)
    rbm.reset()
    assert not rbm.boundary_injected
    assert rbm.boundary_offset == -1


def test_old_flag_names_are_gone():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("main.py", "rigid_bodies.py", "save_load.py", "simulation.py", "world.py"):
        with open(os.path.join(here, name), encoding="utf-8") as fh:
            src = fh.read()
        assert "_rigid_boundary_initialized" not in src, name
        assert "_boundary_particles_injected" not in src, name


def test_load_scene_rebuilds_block(tmp_path, monkeypatch):
    """bd-r5-review-n28.35: save with bodies active, load, step: one intact block."""
    import types

    import save_load

    monkeypatch.setattr(save_load, "SAVE_DIR", str(tmp_path))
    world, sim, rbm = _make_sim()
    _step(sim, 2)
    n_fluid = world._high_water - rbm.num_boundary_particles
    camera = types.SimpleNamespace(
        target=np.zeros(3, dtype=np.float32), distance=2.0,
        azimuth=0.0, elevation=0.0,
    )
    save_load.save_scene(world, sim, camera, "rigid_block")
    # Move _high_water before loading so a stale offset would be wrong.
    world.spawn_sphere((0.0, 0.4, 0.0), 0.05, WATER, 300)
    n = save_load.load_scene(world, sim, camera, "rigid_block")
    assert n == n_fluid  # the saved block was dropped, fluid kept
    assert not rbm.boundary_injected
    sim.notify_scene_changed()
    _step(sim)
    assert rbm.boundary_offset == n_fluid
    _check_block(world, rbm)
