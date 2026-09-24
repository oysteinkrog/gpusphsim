"""Regression test for bd-r5-review-n28.36: vorticity sign in all three solvers.

A 7^3 WATER lattice (spacing 0.02, h 0.04) turns as a rigid body,
v = Omega x r with Omega = (0, 0, 1). The curl of that field is 2*Omega, so
interior particles must report vorticity.z close to +2. Before the fix every
solver accumulated minus the SPH curl and reported about -2.

The vorticity is read from world.sorted_vorticity after one substep. It is
written by K_Step1 (WCSPH), K_PBF_ComputeLambda (PBF) and
K_DFSPH_ComputeDensityAlpha (DFSPH).
"""

import numpy as np
import pytest

import cupy as cp
from materials import WATER
from simulation import Simulation
from solver_profiles import PROFILES
from world import World

SPACING = 0.02
OMEGA = 1.0


@pytest.fixture(scope="module")
def world():
    return World(max_particles=4096)


def _rigid_rotation_vorticity(world, profile_name, substeps):
    sim = Simulation(
        world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20
    )
    sim.set_solver_profile(PROFILES[profile_name])
    sim.set_gravity(0.0)
    sim._spawn_substep = sim._damping_duration  # no spawn velocity damping
    sim.sim_time = 0.0
    sim._last_frame_time = None

    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube(
        (-0.07, -0.07, -0.07), (0.07, 0.07, 0.07), WATER, spacing=SPACING
    )
    assert n == 7**3
    pos = world.position[:n, :3]
    center = pos.mean(axis=0)
    rx = pos[:, 0] - center[0]
    ry = pos[:, 1] - center[1]
    world.velocity[:n, 0] = -OMEGA * ry
    world.velocity[:n, 1] = OMEGA * rx
    world.velocity[:n, 2] = 0.0

    # WCSPH step1 reads neighbour density from the previous substep, so run
    # more than one substep before trusting the magnitude.
    for _ in range(substeps):
        sim._sim_step(n)

    vort = cp.asnumpy(world.sorted_vorticity[:n])
    spos = cp.asnumpy(world.sorted_position[:n, :3])
    center_np = cp.asnumpy(center)
    # Interior: the central 3^3 block, whose kernel support (2 spacings) is
    # fully inside the lattice.
    interior = np.all(np.abs(spos - center_np) < 1.05 * SPACING, axis=1)
    assert interior.sum() == 27
    return vort[interior]


@pytest.mark.parametrize(
    "profile_name,substeps",
    [
        ("WCSPH", 3),
        ("PBF", 1),
        ("DFSPH", 1),
    ],
)
def test_rigid_rotation_vorticity_is_plus_two_omega(world, profile_name, substeps):
    vort = _rigid_rotation_vorticity(world, profile_name, substeps)
    wz = vort[:, 2]
    print(
        f"{profile_name}: interior omega_z min {wz.min():.3f} "
        f"mean {wz.mean():.3f} max {wz.max():.3f}; "
        f"|omega_xy| max {np.abs(vort[:, :2]).max():.3e}"
    )
    assert (wz > 0.0).all(), f"{profile_name}: omega_z has the wrong sign: {wz}"
    assert np.abs(wz - 2.0 * OMEGA).max() < 0.6, (
        f"{profile_name}: omega_z {wz.min():.3f}..{wz.max():.3f}, expected ~2"
    )
    assert abs(wz.mean() - 2.0 * OMEGA) < 0.2 * OMEGA, f"mean omega_z {wz.mean():.3f}"
    assert np.abs(vort[:, :2]).max() < 0.1 * OMEGA
    assert np.allclose(
        vort[:, 3], np.linalg.norm(vort[:, :3], axis=1), rtol=1e-3, atol=1e-4
    )
