"""Regression test for bd-r5-review-n28.158: packed (p / rho^2, |omega|) carrier.

K_Step1 writes (p / rho^2, |omega|) per particle into world.sorted_pr2_omega,
and K_Step2 reads it for FLUID neighbours instead of pressure_in and the
vorticity float4. With a NULL carrier K_Step2 takes the old loads. The two
paths compute the same numbers with the same expressions, so the forces and
the XSPH velocity must agree to float rounding (the bead asks for rtol 1e-5).

The scene is a small dam break (water column) with a sand block, a smoke
block and a stone block, so FLUID, GRANULAR, GAS and STATIC neighbours all
occur. Vorticity confinement is on (default epsilon), so the |omega| lane is
used. The comparison runs every 10 substeps over 100 substeps.
"""

import numpy as np
import pytest

import cupy as cp
import step2
from materials import SAND, SMOKE, STONE, WATER
from simulation import Simulation
from solver_profiles import PROFILES
from world import World

SPACING = 0.02
SUBSTEPS = 100
CHECK_EVERY = 10
RTOL = 1e-5


def _run_step2(sim, n, packed):
    w = sim.world
    rbm = sim.rigid_body_manager
    force = cp.zeros((n, 4), dtype=cp.float32)
    veleval = cp.zeros((n, 4), dtype=cp.float32)
    step2.compute_step2(
        w.sorted_position[:n],
        w.sorted_velocity[:n],
        w.sorted_mass[:n],
        w.sorted_packed_info[:n],
        w.sorted_shear_rate[:n],
        sim._cell_start,
        sim._cell_end,
        vorticity_in=w.sorted_vorticity,
        normal_in=w.sorted_normal,
        sph_force_out=force,
        veleval_out=veleval,
        velocity_h=w.sorted_velocity_h,
        pressure_in=w.sorted_pressure,
        d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
        d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
        d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
        pr2_omega_in=w.sorted_pr2_omega if packed else None,
    )
    return cp.asnumpy(force), cp.asnumpy(veleval)


def _assert_close(a, b, what):
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    np.testing.assert_allclose(
        a, b, rtol=RTOL, atol=RTOL * max(scale, 1e-6), err_msg=what
    )


@pytest.mark.gpu
def test_packed_carrier_matches_unpacked_path():
    world = World(max_particles=40_000)
    n_w = world.spawn_cube(
        (-0.9, -0.9, -0.15), (-0.5, -0.1, 0.15), WATER, spacing=SPACING
    )
    n_s = world.spawn_cube((0.2, -0.9, -0.1), (0.4, -0.7, 0.1), SAND, spacing=SPACING)
    n_g = world.spawn_cube(
        (-0.3, -0.5, -0.1), (-0.1, -0.3, 0.1), SMOKE, spacing=SPACING
    )
    n_b = world.spawn_cube(
        (-0.4, -0.9, -0.1), (-0.3, -0.6, 0.1), STONE, spacing=SPACING
    )
    assert min(n_w, n_s, n_g, n_b) > 0

    sim = Simulation(
        world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=True, max_substeps=1
    )
    sim.set_solver_profile(PROFILES["WCSPH"])
    sim._spawn_substep = sim._damping_duration  # no spawn damping ramp

    checks = 0
    for step in range(1, SUBSTEPS + 1):
        n = world._high_water
        sim._sort_skip_next = False  # full sort, so sorted buffers match n
        sim._sim_step(n)
        if step % CHECK_EVERY:
            continue
        # The sorted buffers still hold this substep's step1 outputs.
        f_new, v_new = _run_step2(sim, n, packed=True)
        f_old, v_old = _run_step2(sim, n, packed=False)
        assert np.all(np.isfinite(f_new))
        _assert_close(f_new, f_old, f"sph_force at substep {step}")
        _assert_close(v_new, v_old, f"veleval at substep {step}")
        checks += 1

    assert checks == SUBSTEPS // CHECK_EVERY

    # The carrier holds what the old path divides out: p / rho^2 and |omega|.
    n = world._high_water
    pk = cp.asnumpy(world.sorted_pr2_omega[:n])
    p = cp.asnumpy(world.sorted_pressure[:n])
    rho = cp.asnumpy(world.sorted_position[:n, 3])
    om = cp.asnumpy(world.sorted_vorticity[:n, 3])
    np.testing.assert_allclose(pk[:, 0], p / (rho * rho), rtol=1e-6, atol=0)
    np.testing.assert_array_equal(pk[:, 1], om)
    # Vorticity is non-trivial in a falling column, so the |omega| lane is exercised.
    assert float(np.max(om)) > 1e-3
