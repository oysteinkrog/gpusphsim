"""DFSPH volume carrier and predicted velocity (bd-r5-review-n28.184).

- K_DFSPH_PackVolume writes m / max(rho, eps) into position.w, or the mass
  for a regular STATIC wall.
- K_DFSPH_ComputePressureAccel writes vt = clamp_v(v + dt * clamp_a(a_press)),
  the value the old K_DFSPH_ApplyPressureVelocity gave the velocity, and
  vt = v for STATIC particles.
- K_DFSPH_DensitySolverUpdate predicts density from that vt with the Akinci
  wall volume m_j / rho0_i.

Each check compares against an all-pairs numpy reference on a WATER block
resting on one STONE layer, so fluid-wall pairs are covered.
"""

import math

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
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import MATERIALS, STATIC, STONE, WATER  # noqa: E402

RHO_EPSILON = 0.01


@pytest.fixture(scope="module")
def world():
    return World(max_particles=4000)


@pytest.fixture
def sim(world):
    return Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                      fixed_dt=False, max_substeps=20)


def _scene(w, sim):
    """7x7x7 WATER block on a 9x9 STONE layer; density/alpha twice, then pack."""
    profile = PROFILES["DFSPH"]
    w.packed_info[:] = 0
    w._high_water = 0
    n_stone = w.spawn_cube((-0.09, 0.18, -0.09), (0.09, 0.2, 0.09), STONE, spacing=0.02)
    n = n_stone + w.spawn_cube((-0.07, 0.2, -0.07), (0.07, 0.34, 0.07), WATER, spacing=0.02)
    sim.set_solver_profile(profile)
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)
    for pass_idx in range(2):
        density_in = None if pass_idx == 0 else w.sorted_density[:n].copy()
        dfsph_solver.compute_density_alpha(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_mass[:n], density_in,
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            sim._cell_start, sim._cell_end,
            w.sorted_density, w.sorted_alpha_dfsph,
            w.sorted_shear_rate, w.sorted_dTdt,
            w.sorted_exposure_heat, w.sorted_exposure_corrode,
        )
    dfsph_solver.pack_volume(
        w.sorted_position[:n], w.sorted_density[:n],
        w.sorted_mass[:n], w.sorted_packed_info[:n],
    )
    cp.cuda.Device().synchronize()
    return n, profile


def _host(w, n):
    pos = w.sorted_position[:n].get().astype(np.float64)
    rho = w.sorted_density[:n].get().astype(np.float64)
    m = w.sorted_mass[:n].get().astype(np.float64)
    pi = w.sorted_packed_info[:n].get()
    wall = ((pi >> 8) & 3) == STATIC
    return pos, rho, m, pi, wall


def _pairs(pos, h):
    """r_ij = x_i - x_j and the spiky gradient for every pair within h (i != j)."""
    r = pos[:, None, :3] - pos[None, :, :3]
    rlen = np.linalg.norm(r, axis=2)
    near = (rlen <= h) & (rlen * rlen >= 1e-12)
    coeff = -45.0 / (math.pi * h ** 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(near, coeff * (h - rlen) ** 2 / rlen, 0.0)
    return g[:, :, None] * r


def _clamp_len(v, limit):
    mag = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.where(mag > limit, limit / np.maximum(mag, 1e-30), 1.0)
    return v * scale


def test_pack_volume_carrier(world, sim):
    n, _ = _scene(world, sim)
    pos, rho, m, _, wall = _host(world, n)
    assert 50 < wall.sum() < n
    expect = np.where(wall, m, m / np.maximum(rho, RHO_EPSILON))
    np.testing.assert_allclose(pos[:, 3], expect, rtol=1e-6)


@pytest.mark.parametrize("p_scale,v_scale", [(1.0, 0.5), (1.0e3, 0.5), (1.0, 200.0)],
                         ids=["no-clamp", "accel-clamp", "velocity-clamp"])
def test_pressure_accel_writes_clamped_vt(world, sim, p_scale, v_scale):
    """vt equals the old v + dt * a_press with both clamps, and v for walls."""
    w = world
    n, profile = _scene(w, sim)
    pos, rho, m, _, wall = _host(w, n)
    h = float(sim._h)
    dt = float(profile.dt)
    v_limit = float(sim._sim_params[0]["velocity_limit"])
    rng = np.random.default_rng(7)
    p = np.where(wall, 0.0, rng.uniform(0.0, p_scale, n))
    vel = np.zeros((n, 4))
    vel[:, :3] = rng.uniform(-v_scale, v_scale, (n, 3))
    vel[wall, :3] = 0.0

    vel_d = cp.asarray(vel.astype(np.float32))
    vt_d = cp.zeros((n, 4), dtype=cp.float32)
    dfsph_solver.compute_pressure_accel(
        cp.asarray(p.astype(np.float32)), vel_d, w.sorted_position[:n],
        w.sorted_density[:n], w.sorted_mass[:n], w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end, vt_d,
    )
    vt = vt_d.get().astype(np.float64)

    gw = _pairs(pos, h)
    V = m / np.maximum(rho, RHO_EPSILON)
    p_j = np.where(wall, 0.0, p)
    coef = -V[None, :] * (p[:, None] + p_j[None, :])
    a = np.sum(coef[:, :, None] * gw, axis=1)
    a = _clamp_len(a, h / (dt * dt))
    ref = _clamp_len(vel[:, :3] + dt * a, v_limit)
    ref[wall] = vel[wall, :3]

    fluid = ~wall
    if p_scale > 1.0:
        assert np.any(np.linalg.norm(np.sum(coef[:, :, None] * gw, axis=1)[fluid], axis=1)
                      > h / (dt * dt)), "scene never reaches the h/dt^2 clamp"
    if v_scale > v_limit:
        assert np.any(np.linalg.norm(vel[fluid, :3] + dt * a[fluid], axis=1) > v_limit)
    scale = max(float(np.abs(ref).max()), 1e-6)
    np.testing.assert_allclose(vt[:, :3], ref, rtol=1e-3, atol=2e-4 * scale)
    np.testing.assert_array_equal(vt[wall], vel[wall])


def test_density_update_uses_vt_and_akinci_wall_volume(world, sim):
    w = world
    n, profile = _scene(w, sim)
    pos, rho, m, _, wall = _host(w, n)
    h = float(sim._h)
    dt = float(profile.dt)
    rng = np.random.default_rng(3)
    vt = np.zeros((n, 4))
    vt[:, :3] = rng.uniform(-0.3, 0.3, (n, 3))
    vt[wall, :3] = 0.0
    p_old = np.where(wall, 0.0, rng.uniform(0.0, 50.0, n))

    p_d = cp.asarray(p_old.astype(np.float32))
    dfsph_solver.density_solver_update(
        cp.asarray(vt.astype(np.float32)), w.sorted_position[:n], w.sorted_density[:n],
        w.sorted_alpha_dfsph[:n], w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end, p_d,
    )
    p_new = p_d.get().astype(np.float64)

    gw = _pairs(pos, h)
    rho0 = max(float(MATERIALS[WATER].rest_density), 1.0)
    V = np.where(wall, m / rho0, m / np.maximum(rho, RHO_EPSILON))
    dv = vt[:, None, :3] - vt[None, :, :3]
    drho = np.sum(V[None, :] * np.sum(dv * gw, axis=2), axis=1)
    alpha = w.sorted_alpha_dfsph[:n].get().astype(np.float64)
    res = rho / rho0 * (1.0 + dt * drho) - 1.0
    ref = np.maximum(p_old + profile.dfsph_omega * res * alpha / (dt * dt), 0.0)
    ref[wall] = 0.0

    fluid = ~wall
    touches_wall = np.any((np.linalg.norm(gw, axis=2) > 0) & wall[None, :], axis=1) & fluid
    assert touches_wall.sum() > 20
    scale = float(np.abs(ref).max())
    np.testing.assert_allclose(p_new, ref, rtol=1e-3, atol=1e-4 * scale)
