"""bd-r5-review-n28.81: the PBF position correction conserves momentum across
materials with different rest densities.

K_PBF_ComputeDelta gives dx_i = SUM_j m_j (lambda_i/rho0_i + lambda_j/rho0_j)
gradW_ij. The lambda_j term used to divide by rho0_i, so for each WATER/SAND
pair m_i*dx_i + m_j*dx_j was m_i*m_j*(lambda_i+lambda_j)*(1/rho0_i-1/rho0_j)
*gradW instead of 0.
"""

import os

import numpy as np
import pytest

import cupy as cp

import pbf_solver
from world import World
from simulation import Simulation
from solver_profiles import PROFILES
from materials import WATER, SAND

MAX_PARTICLES = 20_000
H = 0.04
FIX_LINE = "if (rho0_j != rho0) lambda_eff_j = lambda_j * (rho0 / rho0_j);"


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


@pytest.fixture
def sim(world):
    return Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                      fixed_dt=False, max_substeps=20)


def _pre_fix_module():
    """The PBF module with the lambda_j rest-density fix taken out."""
    kernel_dir = os.path.join(os.path.dirname(pbf_solver.__file__), "physics", "kernels")
    with open(os.path.join(kernel_dir, "pbf_solver.cu"), encoding="utf-8") as f:
        src = f.read()
    assert src.count(FIX_LINE) == 1, "fix line not found in pbf_solver.cu"
    return cp.RawModule(code=src.replace(FIX_LINE, ""),
                        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"))


def _spawn(world, blocks):
    world.packed_info[:] = 0
    world._high_water = 0
    n = 0
    for lo, hi, mat, spacing in blocks:
        n += world.spawn_cube(min_corner=lo, max_corner=hi, material_id=mat, spacing=spacing)
    return n


def _one_delta(world, sim, blocks, module=None):
    """Spawn, sort, run ComputeLambda + ComputeDelta once on the positions.

    Returns (delta xyz, mass, material id), all sorted order, read back once.
    """
    saved = pbf_solver._module
    try:
        if module is not None:
            pbf_solver._module = module
        sim.set_solver_profile(PROFILES["PBF"])
        n = _spawn(world, blocks)
        world._density_initialized = True
        sim._run_grid_setup(n, force_sort=True)
        pbf_solver.pbf_compute_lambda(
            world.sorted_position[:n], world.sorted_mass[:n],
            world.sorted_packed_info[:n], sim._cell_start, sim._cell_end,
            world.sorted_density, world.sorted_lambda_pbf,
        )
        delta = cp.zeros((n, 4), dtype=cp.float32)
        pbf_solver.pbf_compute_delta(
            world.sorted_position[:n], world.sorted_lambda_pbf[:n],
            world.sorted_mass[:n], world.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end, delta,
        )
        cp.cuda.Device().synchronize()
        d = cp.asnumpy(delta)[:, :3].astype(np.float64)
        m = cp.asnumpy(world.sorted_mass[:n]).astype(np.float64)
        mat = cp.asnumpy(world.sorted_packed_info[:n]) & 0xFF
    finally:
        pbf_solver._module = saved
        sim.set_solver_profile(PROFILES["PBF"])
    return d, m, mat


def _momentum_residual(d, m):
    p = (m[:, None] * d).sum(axis=0)
    scale = np.abs(m[:, None] * d).sum()
    return float(np.linalg.norm(p)), float(scale)


# Compressed WATER block against a compressed SAND block (spacing 0.016 < 0.02
# rest spacing, so both sides are over-dense and every lambda is nonzero).
MIXED = [
    ((-0.096, -0.08, -0.08), (-0.004, 0.08, 0.08), WATER, 0.016),
    ((0.004, -0.08, -0.08), (0.096, 0.08, 0.08), SAND, 0.016),
]
WATER_ONLY = [((-0.096, -0.08, -0.08), (0.096, 0.08, 0.08), WATER, 0.016)]


def test_mixed_block_delta_conserves_momentum(world, sim):
    d, m, mat = _one_delta(world, sim, MIXED)
    assert (mat == WATER).any() and (mat == SAND).any()
    assert np.linalg.norm(d, axis=1).max() < 0.49 * H, "0.5h clamp would break the balance"
    p, scale = _momentum_residual(d, m)
    assert scale > 0.0
    assert p < 1e-4 * scale, f"|sum m dx| = {p:.3e}, sum |m dx| = {scale:.3e}"

    # The pre-fix form leaves a clear net momentum on the same scene.
    d0, m0, _ = _one_delta(world, sim, MIXED, module=_pre_fix_module())
    p0, scale0 = _momentum_residual(d0, m0)
    assert p0 > 1e-2 * scale0, f"pre-fix residual {p0 / scale0:.3e} too small to test"


def test_sand_water_pair_delta_conserves_momentum(world, sim):
    """Two particles 0.8 * spacing apart with zero velocity."""
    gap = 0.8 * 0.02
    pair = [
        ((0.0, 0.0, 0.0), (0.001, 0.001, 0.001), SAND, 0.02),
        ((gap, 0.0, 0.0), (gap + 0.001, 0.001, 0.001), WATER, 0.02),
    ]
    d, m, mat = _one_delta(world, sim, pair)
    assert len(m) == 2 and set(mat.tolist()) == {WATER, SAND}
    p, scale = _momentum_residual(d, m)
    assert scale > 0.0, "pair gave no correction"
    assert p < 1e-9, f"|sum m dx| = {p:.3e} (scale {scale:.3e})"


def test_water_only_delta_bitwise_unchanged(world, sim):
    d_new, _, _ = _one_delta(world, sim, WATER_ONLY)
    d_old, _, _ = _one_delta(world, sim, WATER_ONLY, module=_pre_fix_module())
    assert np.abs(d_new).max() > 0.0
    assert np.array_equal(d_new, d_old), "same-material deltas changed"
