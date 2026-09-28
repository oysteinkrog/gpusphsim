"""GAS does not pass through rigid bodies (bd-r5-review-n28.101).

GAS used to skip every non-GAS neighbour in step2 before the MAT_RIGID
Akinci branch, and static_particle_boundary skipped MAT_RIGID for every
particle, so smoke drifted straight through a falling box. Now GAS reaches
the Akinci branch in WCSPH step2, and static_particle_boundary pushes GAS
out of MAT_RIGID particles in all three solvers.

Setup: a permanent SMOKE block at ambient temperature (no buoyancy), no
gravity, and a dynamic 12 cm box that starts just above it moving down at
1 m/s. Over 0.2 s the box sweeps 0.2 m into the block. Checked every 5
substeps: no SMOKE particle inside the box.

All three solvers also run rigid_body_boundary (bd-r5-review-n28.123):
each particle is pushed out of the body's box SDF at its end-of-substep
pose. In PBF that push is what keeps the smoke out, since the box moves
8 mm per substep and the gas density solve does not see the body.
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
BOX_VEL = (0.0, -1.0, 0.0)
SIM_TIME = 0.2
T_AMBIENT = 293.0
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
    n_gas = world.spawn_cube(GAS_MIN, GAS_MAX, SMOKE, SPACING, lifetime=0.0)
    # Ambient temperature: no buoyancy, so only the box moves the smoke.
    world.temperature[:n_gas] = T_AMBIENT
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


@pytest.mark.parametrize(
    "profile_name",
    ["WCSPH", "PBF", "DFSPH"],
)
def test_falling_box_keeps_smoke_out(profile_name):
    world, sim, rbm = _make_sim(profile_name)
    n_gas0 = int(((world.packed_info[: world._high_water].get() & 0xFF) == SMOKE).sum())
    assert n_gas0 > 1000
    steps = int(round(SIM_TIME / sim.dt))
    worst = 0
    for k in range(steps):
        sim._sim_step(world._high_water)
        if k % 5 == 4 or k == steps - 1:
            cp.cuda.Device().synchronize()
            n_inside, box_y = _gas_inside_box(world, rbm)
            worst = max(worst, n_inside)
    assert rbm.boundary_injected
    print(
        f"{profile_name}: dt {sim.dt:.5f}, box y {BOX_START[1]:.3f} -> {box_y:.3f}, "
        f"most smoke inside box at any check: {worst}"
    )
    # The box must really have entered the smoke block, or the test proves nothing.
    assert box_y - HALF < GAS_MAX[1] - 0.1, f"box only reached y={box_y}"
    assert worst == 0, f"up to {worst} smoke particles inside the box"
