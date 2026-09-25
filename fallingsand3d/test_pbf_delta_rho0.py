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

    Returns (delta xyz, mass, material id) in spawn (unsorted) order, read
    back once. The counting sort orders particles inside a cell by atomics, so
    sorted order differs from run to run.
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
        # K_PBF_ComputeDelta writes the corrected position (bd-r5-review-n28.185);
        # the scenes sit away from the walls, so pred_out - pred_in is the delta.
        pred_out = cp.zeros((n, 4), dtype=cp.float32)
        pbf_solver.pbf_compute_delta(
            world.sorted_position[:n], world.sorted_lambda_pbf[:n],
            world.sorted_mass[:n], world.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end, pred_out,
        )
        cp.cuda.Device().synchronize()
        d = (cp.asnumpy(pred_out)[:, :3].astype(np.float64)
             - cp.asnumpy(world.sorted_position[:n])[:, :3].astype(np.float64))
        m = cp.asnumpy(world.sorted_mass[:n]).astype(np.float64)
        mat = cp.asnumpy(world.sorted_packed_info[:n]) & 0xFF
        perm = cp.asnumpy(sim._sort_perm[:n])
        d[perm], m[perm], mat[perm] = d.copy(), m.copy(), mat.copy()
    finally:
        pbf_solver._module = saved
        sim.set_solver_profile(PROFILES["PBF"])
    return d, m, mat


def _momentum_residual(d, m):
    p = (m[:, None] * d).sum(axis=0)
    scale = np.abs(m[:, None] * d).sum()
    return float(np.linalg.norm(p)), float(scale)


# Compressed WATER block against a compressed SAND block (spacing 0.019 < 0.02
# rest spacing, so both sides are over-dense and every lambda is nonzero,
# while the largest delta stays well under the 0.5h clamp).
MIXED = [
    ((-0.095, -0.076, -0.076), (-0.0005, 0.076, 0.076), WATER, 0.019),
    ((0.0005, -0.076, -0.076), (0.095, 0.076, 0.076), SAND, 0.019),
]
WATER_ONLY = [((-0.095, -0.076, -0.076), (0.095, 0.076, 0.076), WATER, 0.019)]


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
        ((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), SAND, 0.02),
        ((gap - 0.01, -0.01, -0.01), (gap + 0.01, 0.01, 0.01), WATER, 0.02),
    ]
    d, m, mat = _one_delta(world, sim, pair)
    assert len(m) == 2 and set(mat.tolist()) == {WATER, SAND}
    p, scale = _momentum_residual(d, m)
    assert scale > 0.0, "pair gave no correction"
    assert p < 1e-9, f"|sum m dx| = {p:.3e} (scale {scale:.3e})"


def test_water_only_delta_unchanged(world, sim):
    """Same rest density never takes the new branch. The two runs can still
    differ in the last bits because the neighbour summation order inside a
    cell is not fixed, so compare at float precision."""
    d_new, _, _ = _one_delta(world, sim, WATER_ONLY)
    d_old, _, _ = _one_delta(world, sim, WATER_ONLY, module=_pre_fix_module())
    top = np.abs(d_new).max()
    assert top > 0.0
    assert np.abs(d_new - d_old).max() <= 1e-5 * top, "same-material deltas changed"
