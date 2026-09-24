"""GAS does not pass through rigid bodies (bd-r5-review-n28.101).

GAS used to skip every non-GAS neighbour in step2 before the MAT_RIGID
Akinci branch, and static_particle_boundary skipped MAT_RIGID for every
particle, so smoke drifted straight through a falling box. Now GAS reaches
the Akinci branch in WCSPH step2, and static_particle_boundary pushes GAS
out of MAT_RIGID particles in all three solvers.

Setup: a permanent SMOKE block at rest, no gravity, and a dynamic 12 cm box
that starts just above it moving down at 1.5 m/s. After 0.15 s the box has
swept about 0.2 m into the block. No SMOKE particle may be inside the box.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

from materials import SMOKE  # noqa: E402

HALF = 0.06
SPACING = 0.02
BOX_START = (0.0, 0.08, 0.0)
BOX_VEL = (0.0, -1.5, 0.0)
GAS_MIN = (-0.16, -0.30, -0.16)
GAS_MAX = (0.16, 0.0, 0.16)
# A particle counts as inside when it is this far inside every face.
# Boundary particles sit on the faces, and the push keeps gas 0.5h = 0.02 m
# away from them, so any gas this deep has crossed the surface.
INSIDE_MARGIN = 0.005


def _make_sim(profile_name: str):
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(60_000)
    world.spawn_cube(GAS_MIN, GAS_MAX, SMOKE, SPACING, lifetime=0.0)
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=1.0)
    sim.set_solver_profile(PROFILES[profile_name])
    sim.set_gravity(0.0)
    sim._spawn_substep = sim._damping_duration
    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(
        position=BOX_START,
        half_extents=(HALF, HALF, HALF),
        mass=50.0,
        velocity=BOX_VEL,
    )
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    return world, sim, rbm


def _quat_rotate_inv(q, v):
    """Rotate the (N, 3) rows of v by the inverse of unit quaternion q = (x, y, z, w)."""
    qv = -np.asarray(q[:3], dtype=np.float64)
    w = float(q[3])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _gas_inside_box(world, rbm) -> tuple[int, float]:
    n = world._high_water
    mat = world.packed_info[:n].get() & 0xFF
    pos = world.position[:n, :3].get().astype(np.float64)[mat == SMOKE]
    body = rbm.download_bodies()[0]
    c = body["position"][:3].astype(np.float64)
    local = _quat_rotate_inv(body["rotation"], pos - c)
    inside = np.all(np.abs(local) < HALF - INSIDE_MARGIN, axis=1)
    return int(inside.sum()), float(c[1])


@pytest.mark.parametrize("profile_name", ["WCSPH", "PBF", "DFSPH"])
def test_falling_box_keeps_smoke_out(profile_name):
    world, sim, rbm = _make_sim(profile_name)
    n_gas0 = int(((world.packed_info[: world._high_water].get() & 0xFF) == SMOKE).sum())
    assert n_gas0 > 1000
    steps = 150
    for _ in range(steps):
        sim._sim_step(world._high_water)
    cp.cuda.Device().synchronize()
    assert rbm.boundary_injected
    n_inside, box_y = _gas_inside_box(world, rbm)
    print(
        f"{profile_name}: box y {BOX_START[1]:.3f} -> {box_y:.3f}, "
        f"smoke inside box: {n_inside}"
    )
    # The box must really have entered the smoke block, or the test proves nothing.
    assert box_y + HALF < GAS_MAX[1] - 0.05, f"box only reached y={box_y}"
    assert n_inside == 0, f"{n_inside} smoke particles inside the box"
