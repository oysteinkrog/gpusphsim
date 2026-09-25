"""DFSPH alpha with zero sorted density uses the neighbour's rest density (bd-r5-review-n28.110).

sorted_density is zero at program start and for unused slots. When density_in
reads zero, K_DFSPH_ComputeDensityAlpha falls back to the neighbour's material
rest density (floored at RHO_EPSILON). It used 1000 kg/m^3, which for WATER
(rest density 2500) made V_j 2.5x too large and alpha about 6.25x too small.

Scene: a WATER block, alpha cap off. Alpha from density_in = 0 must match
alpha from density_in pre-filled with the WATER rest density within 5 percent.
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

from materials import FLUID, MATERIALS, WATER  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from world import World  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return World(max_particles=20_000)


def _run_alpha(w, sim, n, density_in):
    import dfsph_solver

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
    return w.sorted_alpha_dfsph[:n].get().astype(np.float64)


def test_dfsph_alpha_zero_density_uses_rest_density(world):
    w = world
    w.packed_info[:] = 0
    w._high_water = 0
    n = w.spawn_cube(
        min_corner=(-0.08, 0.0, -0.08),
        max_corner=(0.08, 0.16, 0.08),
        material_id=WATER,
        spacing=0.02,
    )
    assert n > 300

    sim = Simulation(
        w, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20
    )
    sim.set_solver_profile(
        dataclasses.replace(PROFILES["DFSPH"], dfsph_alpha_limit=1.0e9)
    )
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    pi = w.sorted_packed_info[:n].get()
    assert np.all(((pi >> 8) & 0x3) == FLUID)

    rho0 = float(MATERIALS[WATER].rest_density)
    assert rho0 == pytest.approx(2500.0)

    alpha_zero = _run_alpha(w, sim, n, cp.zeros(n, dtype=cp.float32))
    alpha_null = _run_alpha(w, sim, n, None)
    alpha_ref = _run_alpha(w, sim, n, cp.full(n, rho0, dtype=cp.float32))

    rel_zero = np.abs(alpha_zero - alpha_ref) / alpha_ref
    rel_null = np.abs(alpha_null - alpha_ref) / alpha_ref
    print(
        f"  n={n} alpha_ref mean={alpha_ref.mean():.3e} "
        f"max rel err zero={rel_zero.max():.2e} null={rel_null.max():.2e}"
    )
    assert rel_zero.max() < 0.05
    assert rel_null.max() < 0.05
