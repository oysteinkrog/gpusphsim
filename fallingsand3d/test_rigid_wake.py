"""A rigid box dropped on a sleeping sand block lands on it (bd-r5-review-n28.87).

Rigid boundary particles are STATIC and sleeping particles return from the
solver before the rigid coupling. Before the fix the box fell through the
sleeping block to the floor with no force either way. Now the moving box
wakes the sand near it, and the sand holds it up as it does when the block
starts awake.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

from materials import SAND  # noqa: E402

HALF = 0.06            # box half extent
SAND_TOP = -0.3        # sand block spans y in [-0.5, -0.3]
FLOOR = -0.5           # world_half_size = 0.5
SLEEPING_BIT = 0x400


def _run(profile_name: str, substeps: int):
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(120_000)
    world.spawn_cube((-0.2, FLOOR, -0.2), (0.2, SAND_TOP, 0.2), SAND, 0.02)
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=0.5)
    sim.set_solver_profile(PROFILES[profile_name])
    sim._spawn_substep = sim._damping_duration
    n = world._high_water
    # The whole block starts asleep, as a settled Sand Castle does.
    world.packed_info[:n] |= cp.uint32(SLEEPING_BIT)
    world.sleep_counter[:n] = 200

    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(
        position=(0.0, -0.1, 0.0), half_extents=(HALF, HALF, HALF), mass=4.32,
    )
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())

    woke_at = None
    for step in range(substeps):
        sim._sim_step(world._high_water)
        if woke_at is None and step % 10 == 9:
            nn = world._high_water
            pi = world.packed_info[:nn].get()
            sand = (pi & 0xFF) == SAND
            if np.any(((pi >> 10) & 1)[sand] == 0):
                woke_at = step + 1
    cp.cuda.Device().synchronize()
    box_y = float(rbm.d_rigid_bodies.get()[0]["position"][1])
    return world, box_y, woke_at


@pytest.mark.parametrize("profile_name", ["WCSPH", "PBF", "DFSPH"])
def test_box_lands_on_sleeping_sand(profile_name):
    world, box_y, woke_at = _run(profile_name, 1000)

    # The box starts 0.14 m above the sand and reaches it after ~0.17 s.
    assert woke_at is not None and woke_at <= 300, (
        f"{profile_name}: no sand woke as the box arrived (woke_at={woke_at})"
    )
    # Before the fix the box ended on the floor (centre at FLOOR + HALF).
    # With support it stays well above that; an awake block holds it at
    # about -0.37 after 1 s.
    assert box_y > FLOOR + HALF + 0.03, (
        f"{profile_name}: box centre at {box_y:.3f} m, fell through the sand"
    )

    # Sand under the box is awake.
    n = world._high_water
    pi = world.packed_info[:n].get()
    pos = world.position[:n, :3].get()
    sand = (pi & 0xFF) == SAND
    under = sand & (np.abs(pos[:, 0]) < HALF) & (np.abs(pos[:, 2]) < HALF) \
        & (pos[:, 1] > box_y - HALF - 0.06)
    assert under.sum() > 0
    awake_frac = float(np.mean(((pi >> 10) & 1)[under] == 0))
    assert awake_frac > 0.5, (
        f"{profile_name}: only {awake_frac:.0%} of the sand under the box is awake"
    )
