"""DFSPH mu(I) friction uses the solver pressure and stored shear rate (bd-r5-review-n28.92).

K_DFSPH_NonPressureForces computes the GRANULAR confining pressure as
p = p_prev * rho, where p_prev is the previous substep's converged p/rho
from the density solver (sorted_kappa), floored at the WCSPH one-layer
lithostatic pressure rho0 * |g| * spacing. The neighbour's eta uses the
neighbour's own stored shear rate and pressure, so for a GRANULAR pair
eta_ij computed by i equals eta_ji computed by j.

Checks on a two-particle SAND pair:
- the pair force is antisymmetric (a_i + a_j = 0 along the pair) when the
  two particles have different shear rates and pressures;
- the friction scales with eta(p_prev * rho) / eta(floor), which pins the
  p/rho convention (p_prev * rho^2 would sit at mu_max);
- p_prev below the floor, and p_prev = None, both give the floor result.
"""

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, '_cache'):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, '_get_arch_for_options_for_nvrtc'):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, '_cache'):
                fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402

import dfsph_solver  # noqa: E402
import step2  # noqa: E402
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import SAND, MATERIALS  # noqa: E402

RHO = 4000.0
DV = 1e-4  # m/s, keeps the viscous acceleration far below ACCEL_MAX


def _eta_host(gamma_dot, p_eff, rho):
    """Host copy of compute_muI_eta (sph_shared.cuh) with the step2 defaults."""
    spacing = float(step2.DEFAULT_PARTICLE_SPACING)
    mu_s = float(step2.DEFAULT_MU_S)
    mu_2 = float(step2.DEFAULT_MU_2)
    i0 = float(step2.DEFAULT_I0)
    mu_max = float(step2.DEFAULT_MU_MAX)
    mu0 = float(step2.DEFAULT_MU0)
    i_num = gamma_dot * spacing / np.sqrt(p_eff / rho)
    mu_i = mu_s + (mu_2 - mu_s) / (1.0 + i0 / max(i_num, 1e-8))
    return min(mu_max, mu0 + mu_i * p_eff / (gamma_dot + 1e-6))


@pytest.fixture(scope="module")
def pair():
    world = World(max_particles=1000)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=False, max_substeps=20)
    world.packed_info[:] = 0
    world._high_water = 0
    # Two SAND particles 0.02 m apart along x.
    n = world.spawn_cube((0.0, 0.3, 0.0), (0.04, 0.32, 0.02), SAND, spacing=0.02)
    assert n == 2
    sim.set_solver_profile(PROFILES["DFSPH"])
    world._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)
    cp.cuda.Device().synchronize()
    return world, sim, n


def _accel_x(pair, shear, p_prev):
    """Viscous x acceleration of each particle, with x velocities of +DV and -DV."""
    w, sim, n = pair
    pos = cp.asnumpy(w.sorted_position[:n])
    order = np.argsort(pos[:, 0])  # order[0] is the left particle
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[order[0], 0] = DV
    vel[order[1], 0] = -DV
    shear_s = np.zeros(n, dtype=np.float32)
    shear_s[order] = shear
    w.sorted_velocity[:n] = cp.asarray(vel)
    w.sorted_density[:n] = cp.float32(RHO)
    w.sorted_shear_rate[:n] = cp.asarray(shear_s)
    pp = None
    if p_prev is not None:
        pp_s = np.zeros(n, dtype=np.float32)
        pp_s[order] = p_prev
        pp = cp.asarray(pp_s)
    out = cp.zeros((n, 4), dtype=cp.float32)
    dfsph_solver.compute_non_pressure_forces(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_density[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], w.sorted_shear_rate[:n],
        w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        out,
        p_prev=pp,
    )
    cp.cuda.Device().synchronize()
    dvx = cp.asnumpy(out[:n, 0]) - vel[:, 0]
    return dvx[order] / sim.fixed_dt_value


def test_granular_pair_force_is_antisymmetric(pair):
    # Different shear rates and pressures give eta_i != eta_j; the pair
    # force must still be equal and opposite.
    a = _accel_x(pair, shear=[10.0, 20.0], p_prev=[0.5, 0.2])
    print(f"a_x = {a}")
    assert a[0] < 0.0 < a[1], "friction must slow the approach"
    assert abs(a[0] + a[1]) < 1e-3 * abs(a[0]), f"pair force not antisymmetric: {a}"


def test_friction_scales_with_solver_pressure(pair):
    w, sim, _ = pair
    g = abs(sim.gravity_y)
    rho0 = float(MATERIALS[SAND].rest_density)
    p_floor = rho0 * g * float(step2.DEFAULT_PARTICLE_SPACING)
    gamma = 10.0
    p_prev = 0.5  # p = 0.5 * 4000 = 2000 Pa, well above the floor
    a_floor = _accel_x(pair, shear=[gamma, gamma], p_prev=None)
    a_press = _accel_x(pair, shear=[gamma, gamma], p_prev=[p_prev, p_prev])
    ratio = a_press[0] / a_floor[0]
    expected = _eta_host(gamma, p_prev * RHO, RHO) / _eta_host(gamma, p_floor, RHO)
    print(f"p_floor={p_floor:.1f} Pa ratio={ratio:.4f} expected={expected:.4f}")
    assert expected > 2.0  # the case really separates floor from solver pressure
    assert abs(ratio - expected) < 0.01 * expected


def test_low_solver_pressure_uses_floor(pair):
    a_none = _accel_x(pair, shear=[10.0, 10.0], p_prev=None)
    a_zero = _accel_x(pair, shear=[10.0, 10.0], p_prev=[0.0, 0.0])
    a_tiny = _accel_x(pair, shear=[10.0, 10.0], p_prev=[1e-4, 1e-4])
    assert a_none[0] < 0.0
    np.testing.assert_allclose(a_zero, a_none, rtol=1e-5)
    np.testing.assert_allclose(a_tiny, a_none, rtol=1e-5)
