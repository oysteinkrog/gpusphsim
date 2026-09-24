"""Rigid boundary slots are reserved in the spawn capacity (bd-r5-review-n28.58).

Fluid spawned before the boundary block is injected must leave room for it.
If the block still does not fit, the manager flags it, the bodies freeze
instead of falling through the fluid, and a later compaction retries.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

from materials import MAT_RIGID, WATER  # noqa: E402

MAX_PARTICLES = 5000


def _make_sim(max_particles: int = MAX_PARTICLES):
    from simulation import Simulation
    from world import World

    world = World(max_particles)
    sim = Simulation(world, dt=0.001, fixed_dt=True)
    return world, sim, sim.rigid_body_manager


def _add_box(sim, rbm) -> None:
    rbm.add_rigid_body_with_particles(
        position=(0.0, 0.3, 0.0),
        half_extents=(0.05, 0.05, 0.05),
        mass=2.0,
    )
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())


def _step(sim, k: int = 1) -> None:
    for _ in range(k):
        sim._sim_step(sim.world._high_water)
    cp.cuda.Device().synchronize()


def _check_block(world, rbm) -> None:
    off, nb = rbm.boundary_offset, rbm.num_boundary_particles
    assert off >= 0
    assert off + nb <= world._high_water
    packed = world.packed_info[off:off + nb].get()
    assert np.all((packed & 0xFF) == MAT_RIGID)


def test_spawn_leaves_room_for_boundary_block():
    world, sim, rbm = _make_sim()
    _add_box(sim, rbm)
    nb = rbm.num_boundary_particles
    assert nb > 0
    assert world.reserved_tail == nb

    n = world.spawn_sphere((0.0, -0.3, 0.0), 0.3, WATER, 10_000)
    assert n == MAX_PARTICLES - nb
    assert world._high_water == MAX_PARTICLES - nb
    assert world.free_capacity() == 0

    _step(sim)
    assert not rbm.boundary_dropped
    assert world._high_water == MAX_PARTICLES
    _check_block(world, rbm)
    # The block now counts inside _high_water: no double reservation.
    assert world.reserved_tail == 0


def test_after_injection_spawn_fills_to_cap_without_evicting_block():
    world, sim, rbm = _make_sim()
    world.spawn_sphere((0.0, -0.3, 0.0), 0.3, WATER, 1000)
    _add_box(sim, rbm)
    _step(sim)
    nb = rbm.num_boundary_particles
    assert world._high_water == 1000 + nb

    world.spawn_sphere((0.0, -0.3, 0.0), 0.3, WATER, 10_000)
    assert world._high_water == MAX_PARTICLES
    _step(sim)
    _check_block(world, rbm)
    assert not rbm.boundary_dropped


def test_over_capacity_flags_and_freezes_body_then_retries():
    world, sim, rbm = _make_sim()
    # Fluid first, as the presets do, so the reservation comes too late.
    world.spawn_sphere((0.0, -0.3, 0.0), 0.3, WATER, 10_000)
    assert world._high_water == MAX_PARTICLES
    _add_box(sim, rbm)
    nb = rbm.num_boundary_particles

    pos0 = rbm.download_bodies()["position"].copy()
    _step(sim, 5)
    assert rbm.boundary_dropped
    assert rbm.boundary_offset < 0
    assert world._high_water == MAX_PARTICLES
    np.testing.assert_array_equal(rbm.download_bodies()["position"], pos0)
    # Fluid may not take the block's slots while it waits.
    assert world.free_capacity() == 0

    # Kill enough fluid, compact on a due frame: the block is retried.
    world.packed_info[:nb + 500] = cp.uint32(0)
    world.dead_reserve = 0  # plain compaction, no parked DEAD top-up
    sim.compact_threshold = 1
    sim._frame_counter = 0
    sim._maybe_compact()
    assert not rbm.boundary_dropped
    _step(sim)
    assert not rbm.boundary_dropped
    _check_block(world, rbm)
    _step(sim, 5)
    assert not np.array_equal(rbm.download_bodies()["position"], pos0)


def test_reset_clears_reservation():
    world, sim, rbm = _make_sim()
    _add_box(sim, rbm)
    assert world.reserved_tail > 0
    sim.reset_scene_objects()
    assert world.reserved_tail == 0
    assert world.free_capacity() == MAX_PARTICLES
