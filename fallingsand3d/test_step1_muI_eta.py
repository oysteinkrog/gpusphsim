"""Regression test for bd-r5-review-n28.169: mu(I) eta computed once in K_Step1.

K_Step1 PostCalc writes the GRANULAR mu(I) viscosity eta per particle to
eta_out, and K_Step2 reads it per GRANULAR pair (eta_in) instead of calling
compute_muI_eta for every pair. With eta_in NULL, K_Step2 recomputes eta from
shear_rate the old way. Both paths use the same inputs (density, pressure,
gamma_dot, per-material mu_s and the lithostatic pressure floor), so the
forces must agree to rtol 1e-5 over 100 substeps (the bead's criterion).

The scene is a sand pile falling next to a water column and a stone block,
so GRANULAR-GRANULAR, FLUID-GRANULAR and STATIC pairs all occur.
"""

import numpy as np
import pytest

import cupy as cp
import step1
import step2
from materials import SAND, STONE, WATER, GRANULAR
from simulation import Simulation
from solver_profiles import PROFILES
from world import World

SPACING = 0.02
SUBSTEPS = 100
CHECK_EVERY = 10
RTOL = 1e-5


def _run_step1(sim, n):
    """Re-run K_Step1 on this substep's sorted state with eta_out on."""
    w = sim.world
    f32 = cp.float32
    out = dict(
        density_out=cp.zeros(n, f32),
        shear_rate_out=cp.zeros(n, f32),
        dTdt_out=cp.zeros(n, f32),
        exposure_heat_out=cp.zeros(n, f32),
        exposure_corrode_out=cp.zeros(n, f32),
        vorticity_out=cp.zeros((n, 4), f32),
        normal_out=cp.zeros((n, 4), f32),
        pressure_out=cp.zeros(n, f32),
        pr2_omega_out=cp.zeros((n, 2), f32),
        eta_out=cp.full(n, -1.0, f32),
    )
    step1.compute_step1(
        w.sorted_position[:n],
        w.sorted_velocity[:n],
        w.sorted_mass[:n],
        w.sorted_packed_info[:n],
        w.sorted_temperature[:n],
        sim._cell_start,
        sim._cell_end,
        velocity_h=w.sorted_velocity_h,
        temperature_h=w.sorted_temperature_h,
        **out,
    )
    return out


def _run_step2(sim, n, s1, use_eta):
    w = sim.world
    rbm = sim.rigid_body_manager
    force = cp.zeros((n, 4), dtype=cp.float32)
    veleval = cp.zeros((n, 4), dtype=cp.float32)
    step2.compute_step2(
        w.sorted_position[:n],
        w.sorted_velocity[:n],
        w.sorted_mass[:n],
        w.sorted_packed_info[:n],
        s1["shear_rate_out"],
        sim._cell_start,
        sim._cell_end,
        vorticity_in=s1["vorticity_out"],
        normal_in=s1["normal_out"],
        sph_force_out=force,
        veleval_out=veleval,
        velocity_h=w.sorted_velocity_h,
        pressure_in=s1["pressure_out"],
        d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
        d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
        d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
        pr2_omega_in=s1["pr2_omega_out"],
        eta_in=s1["eta_out"] if use_eta else None,
    )
    return cp.asnumpy(force), cp.asnumpy(veleval)


def _assert_close(a, b, what):
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    np.testing.assert_allclose(
        a, b, rtol=RTOL, atol=RTOL * max(scale, 1e-6), err_msg=what
    )


@pytest.mark.gpu
def test_step1_eta_matches_per_pair_muI():
    world = World(max_particles=40_000)
    n_s = world.spawn_cube((-0.2, -0.9, -0.2), (0.2, -0.3, 0.2), SAND, spacing=SPACING)
    n_w = world.spawn_cube((-0.9, -0.9, -0.1), (-0.5, -0.3, 0.1), WATER, spacing=SPACING)
    n_b = world.spawn_cube((0.4, -0.9, -0.1), (0.5, -0.6, 0.1), STONE, spacing=SPACING)
    assert min(n_s, n_w, n_b) > 0

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
        s1 = _run_step1(sim, n)
        f_new, v_new = _run_step2(sim, n, s1, use_eta=True)
        f_old, v_old = _run_step2(sim, n, s1, use_eta=False)
        assert np.all(np.isfinite(f_new))
        _assert_close(f_new, f_old, f"sph_force at substep {step}")
        _assert_close(v_new, v_old, f"veleval at substep {step}")

        # eta is written for every particle: > 0 for GRANULAR, 0 for the rest.
        eta = cp.asnumpy(s1["eta_out"])
        behavior = (cp.asnumpy(world.sorted_packed_info[:n]) >> 8) & 0x3
        gran = behavior == GRANULAR
        assert gran.sum() > 0
        assert np.all(eta[gran] > 0.0)
        assert np.all(eta[~gran] == 0.0)
        checks += 1

    assert checks == SUBSTEPS // CHECK_EVERY
