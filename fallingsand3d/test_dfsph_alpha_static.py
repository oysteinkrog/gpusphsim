"""DFSPH alpha keeps STATIC neighbours out of the squared-gradient term (bd-r5-review-n28.108).

alpha_i = 1 / (|SUM_j V_j gradW_ij|^2 + SUM_{j not STATIC} V_j^2 |gradW_ij|^2)

The second term is neighbour j's own response to kappa_i. A STATIC particle
(STONE, MAT_RIGID) never moves under pressure, so it belongs in the first term
only (Bender and Koschier 2015, SPlisHSPlasH computeDFSPHFactor).

Scene: a WATER block resting on a two-layer STONE floor. The test runs
K_DFSPH_ComputeDensityAlpha with the alpha cap off and recomputes denom on the
host from the same positions, masses and density_in:

- every WATER particle matches the host formula with STATIC excluded;
- WATER particles with no STATIC neighbour match the old formula too
  (interior alpha unchanged);
- WATER particles next to the floor get a larger alpha than the old formula.
"""

import dataclasses

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler

        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}
        fn = getattr(_compiler, "_get_arch_for_options_for_nvrtc", None)
        if fn is not None and hasattr(fn, "_cache"):
            fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402

from materials import FLUID, STATIC, STONE, WATER  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from world import World  # noqa: E402

RHO_EPSILON = 0.01


@pytest.fixture(scope="module")
def world():
    return World(max_particles=20_000)


def _host_denom(pos, mass, rho_in, behavior, idx, h, skip_static_sq, rho0_i=2500.0):
    """denom for particle idx, brute force over all particles."""
    coeff = -45.0 / (np.pi * h**6)
    r = pos[idx] - pos
    r_sq = np.einsum("ij,ij->i", r, r)
    nb = (r_sq <= h * h) & (r_sq > 1e-12) & (behavior != 2)  # GAS = 2 skipped
    nb[idx] = False
    rl = np.sqrt(r_sq[nb])
    gW = (coeff * (h - rl) ** 2 / rl)[:, None] * r[nb]
    rho_j = np.where(rho_in[nb] <= 0.0, 1000.0, rho_in[nb])
    w = mass[nb] / np.maximum(rho_j, RHO_EPSILON)
    # grad_sum gives a STATIC wall the Akinci volume m_j / rho0_i with the
    # fluid's own rest density (bd-r5-review-n28.116).
    rho_grad = np.where(behavior[nb] == STATIC, rho0_i, rho_j)
    w_grad = mass[nb] / np.maximum(rho_grad, RHO_EPSILON)
    grad_sum = (w_grad[:, None] * gW).sum(axis=0)
    sq = w * w * np.einsum("ij,ij->i", gW, gW)
    n_static = int((behavior[nb] == STATIC).sum())
    if skip_static_sq:
        sq = sq[behavior[nb] != STATIC]
    return float(grad_sum @ grad_sum + sq.sum()), n_static


def test_dfsph_alpha_excludes_static_squared_term(world):
    import dfsph_solver

    assert FLUID == 0 and STATIC == 3
    w = world
    w.packed_info[:] = 0
    w._high_water = 0
    s = 0.02
    n_water = w.spawn_cube(
        min_corner=(-0.06, 0.0, -0.06),
        max_corner=(0.06, 0.12, 0.06),
        material_id=WATER,
        spacing=s,
    )
    n_stone = w.spawn_cube(
        min_corner=(-0.12, -0.04, -0.12),
        max_corner=(0.12, 0.0, 0.12),
        material_id=STONE,
        spacing=s,
    )
    n = n_water + n_stone
    assert n_water > 200 and n_stone > 100

    sim = Simulation(
        w, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20
    )
    sim.set_solver_profile(
        dataclasses.replace(PROFILES["DFSPH"], dfsph_alpha_limit=1.0e9)
    )
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    for pass_idx in range(2):
        density_in = None if pass_idx == 0 else w.sorted_density[:n].copy()
        dfsph_solver.compute_density_alpha(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_mass[:n],
            density_in,
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            sim._cell_start,
            sim._cell_end,
            w.sorted_density,
            w.sorted_alpha_dfsph,
            w.sorted_shear_rate,
            w.sorted_dTdt,
            w.sorted_exposure_heat,
            w.sorted_exposure_corrode,
        )
    cp.cuda.Device().synchronize()

    pos = w.sorted_position[:n, :3].get().astype(np.float64)
    mass = w.sorted_mass[:n].get().astype(np.float64)
    rho_in = density_in.get().astype(np.float64)
    pi = w.sorted_packed_info[:n].get()
    behavior = (pi >> 8) & 0x3
    alpha = w.sorted_alpha_dfsph[:n].get().astype(np.float64)
    h = float(sim._h)

    fluid = np.nonzero(behavior == FLUID)[0]
    assert len(fluid) == n_water
    wall_gain, interior_err, wall_err = [], [], []
    for i in fluid:
        d_new, n_static = _host_denom(pos, mass, rho_in, behavior, i, h, True)
        d_old, _ = _host_denom(pos, mass, rho_in, behavior, i, h, False)
        a_new = 1.0 / d_new
        err = abs(alpha[i] - a_new) / a_new
        if n_static == 0:
            interior_err.append(err)
            # Old and new agree exactly without STATIC neighbours.
            assert d_old == d_new
        else:
            wall_err.append(err)
            wall_gain.append(alpha[i] * d_old)  # alpha_gpu / alpha_old

    interior_err = np.array(interior_err)
    wall_err = np.array(wall_err)
    wall_gain = np.array(wall_gain)
    print(f"  interior n={len(interior_err)} max rel err={interior_err.max():.2e}")
    print(
        f"  wall     n={len(wall_err)} max rel err={wall_err.max():.2e} "
        f"alpha_new/alpha_old min={wall_gain.min():.4f} mean={wall_gain.mean():.4f} "
        f"max={wall_gain.max():.4f}"
    )
    assert len(interior_err) > 50 and len(wall_err) > 30
    # Interior alpha unchanged: GPU matches the (shared) host formula.
    assert interior_err.max() < 1e-3
    # Wall alpha matches the STATIC-excluded formula...
    assert wall_err.max() < 1e-3, (
        "GPU alpha does not match the STATIC-excluded denominator"
    )
    # ...and is larger than the old formula that summed STATIC in both terms.
    # Particles that see STATIC only at the kernel edge change by float noise;
    # the 6x6 bottom layer in contact with the floor gains about 2%.
    assert wall_gain.min() > 1.0 - 1e-5
    assert (wall_gain > 1.01).sum() >= 36
