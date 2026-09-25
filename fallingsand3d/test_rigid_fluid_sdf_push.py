"""Fluid cannot end up inside a fast rigid body (bd-r5-review-n28.123).

The fluid used to feel a rigid body only through the Akinci force of its one
2 cm boundary layer. A body that moves more than about one spacing per
substep (PBF at dt = 1/120 moves 2.5 cm per substep at 3 m/s) overtakes fluid
particles, which then sit inside the hollow body and stay there.

K_Integrate, K_PBF_Finalize and K_DFSPH_Finalize now run rigid_body_boundary
after the SDF object loop: each fluid particle is tested against every body's
box or sphere SDF at the body's end-of-substep pose and pushed out to
BOUNDARY_MARGIN outside the surface.

Test: a still water column, a kinematic 12 cm box driven straight down into
it at 3 m/s. After every substep, transform the fluid into the box frame and
count the particles strictly inside the half extents. The count must be 0.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

from materials import WATER  # noqa: E402

HALF = 0.06          # box half extent (12 cm box)
SPEED = 3.0          # m/s, downward
SPACING = 0.02
TRAVEL = 0.9         # metres the box moves during the test


def _make_sim(profile_name: str):
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(120_000)
    body_pos = (0.0, 0.1, 0.0)
    # Water column 40 cm wide, from the floor to y = 0. The box starts just
    # above the surface and is driven down through it.
    world.spawn_cube((-0.2, -1.46, -0.2), (0.2, 0.0, 0.2), WATER, SPACING)
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=1.5)
    sim.set_solver_profile(PROFILES[profile_name])
    sim._spawn_substep = sim._damping_duration
    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(
        position=body_pos,
        half_extents=(HALF, HALF, HALF),
        mass=4.32,
        is_kinematic=True,
        velocity=(0.0, -SPEED, 0.0),
    )
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    return world, sim, rbm


def _count_inside(world, rbm) -> int:
    n = world._high_water
    mat = world.packed_info[:n].get() & 0xFF
    pos = world.position[:n, :3].get()[mat == WATER].astype(np.float64)
    body = rbm.d_rigid_bodies.view(cp.float32).reshape(-1, 24)[0].get()
    com = body[0:3].astype(np.float64)
    qx, qy, qz, qw = body[4:8].astype(np.float64)
    # Rotate world offsets by the inverse quaternion (conjugate).
    rel = pos - com
    u = np.array([-qx, -qy, -qz])
    t = 2.0 * np.cross(u, rel)
    local = rel + qw * t + np.cross(u, t)
    inside = np.all(np.abs(local) < HALF, axis=1)
    return int(inside.sum())


@pytest.mark.parametrize("profile_name", ["WCSPH", "PBF", "DFSPH"])
def test_fast_box_leaves_no_fluid_inside(profile_name):
    world, sim, rbm = _make_sim(profile_name)
    dt = sim.dt
    steps = int(round(TRAVEL / (SPEED * dt)))
    worst = 0
    worst_step = -1
    for k in range(steps):
        sim._sim_step(world._high_water)
        cp.cuda.Device().synchronize()
        c = _count_inside(world, rbm)
        if c > worst:
            worst, worst_step = c, k
    y_end = float(rbm.d_rigid_bodies.view(cp.float32).reshape(-1, 24)[0, 1].get())
    assert y_end < -0.6, f"box did not move into the water (y = {y_end:.3f})"
    assert worst == 0, (
        f"{profile_name}: up to {worst} water particles inside the box "
        f"(first seen at substep {worst_step} of {steps}, dt = {dt:.5f})"
    )
