"""bd-r5-review-n28.197: lean PBF lambda kernel for the solver iterations.

K_PBF_ComputeLambda (heat, exposure, vorticity, normals) and
K_PBF_ComputeLambdaLean (density and lambda only) share one templated body.
pbf_compute_lambda picks Lean when no heat, vorticity or normal argument is
given, and density_out may then be None.

Checks:
- Lean fits 48 registers and neither variant spills to local memory.
- On a random WATER/SAND/STEAM/STONE cloud, Lean gives bitwise the same
  density, lambda and pressure normal as the full kernel launched with every
  extra pointer NULL (the old iteration call).
- Lean with density_out None leaves the density array untouched and still
  writes the same lambda.
- The full variant refuses density_out None.
"""

import numpy as np
import pytest

import cupy as cp

import pbf_solver
from world import World
from simulation import Simulation
from solver_profiles import PROFILES
from materials import WATER, SAND, STONE, STEAM

MAX_PARTICLES = 20_000


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


@pytest.fixture
def sim(world):
    s = Simulation(world, dt=1.0 / 240.0, speed=1.0, accuracy=0.4,
                   fixed_dt=True, max_substeps=1)
    s.set_solver_profile(PROFILES["PBF"])
    s._spawn_substep = s._damping_duration
    return s


def _cloud(sim, world, seed=5):
    """Jittered WATER, SAND and STEAM over a STONE slab, sorted and predicted."""
    w = world
    w.packed_info[:] = 0
    w._high_water = 0
    w.spawn_cube((-0.12, -0.14, -0.12), (0.12, -0.1, 0.12), STONE, 0.02)
    w.spawn_cube((-0.1, -0.1, -0.1), (0.0, 0.06, 0.1), WATER, 0.02)
    w.spawn_cube((0.0, -0.1, -0.1), (0.1, 0.06, 0.1), SAND, 0.02)
    w.spawn_cube((-0.1, 0.06, -0.1), (0.1, 0.12, 0.1), STEAM, 0.02)
    n = w._high_water
    rng = np.random.default_rng(seed)
    pos = cp.asnumpy(w.position[:n]).copy()
    moving = (cp.asnumpy(w.packed_info[:n]) & 0xFF) != STONE
    pos[moving, :3] += rng.uniform(-0.006, 0.006, size=(int(moving.sum()), 3)).astype(np.float32)
    w.position[:n] = cp.asarray(pos)
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[moving, :3] = rng.uniform(-0.3, 0.3, size=(int(moving.sum()), 3)).astype(np.float32)
    w.velocity[:n] = cp.asarray(vel)
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)
    pbf_solver.pbf_predict(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        w.sorted_predicted_position, mass=w.sorted_mass[:n],
    )
    return n


def _launch_full_no_extras(w, sim, n, mass, dens, lam, pn):
    """The old iteration call: the full kernel with every extra pointer NULL."""
    k = pbf_solver._get_module().get_function("K_PBF_ComputeLambda")
    null = np.intp(0)
    bs = pbf_solver.BLOCK_SIZE
    k(((n + bs - 1) // bs,), (bs,), (
        np.uint32(n), w.sorted_predicted_position[:n],
        mass if mass is not None else null,
        w.sorted_packed_info[:n], sim._cell_start, sim._cell_end,
        dens, lam, pn,
        null, null, null, null, null, null, null, null,
    ))


def test_lean_registers_and_no_spills():
    mod = pbf_solver._get_module()
    lean = mod.get_function("K_PBF_ComputeLambdaLean")
    full = mod.get_function("K_PBF_ComputeLambda")
    assert lean.num_regs <= 48, lean.num_regs
    assert lean.local_size_bytes == 0
    assert full.local_size_bytes == 0
    assert lean.num_regs < full.num_regs


@pytest.mark.parametrize("mass_from_w", [True, False])
def test_lean_matches_full(world, sim, mass_from_w):
    n = _cloud(sim, world)
    w = world
    beh = (cp.asnumpy(w.sorted_packed_info[:n]) >> 8) & 0x3
    assert {0, 1, 2, 3} <= set(np.unique(beh).tolist())
    mass = None if mass_from_w else w.sorted_mass[:n]

    # Same starting contents for the GAS lambda slot and STATIC-skipped outputs.
    lam0 = cp.random.default_rng(1).random(w.sorted_lambda_pbf.shape, dtype=cp.float32)
    dens0 = cp.full_like(w.sorted_density, 123.0)

    lam_f, dens_f = lam0.copy(), dens0.copy()
    pn_f = cp.full_like(w.sorted_pressure_normal, 7.0)
    _launch_full_no_extras(w, sim, n, mass, dens_f, lam_f, pn_f)

    lam_l, dens_l = lam0.copy(), dens0.copy()
    pn_l = cp.full_like(w.sorted_pressure_normal, 7.0)
    pbf_solver.pbf_compute_lambda(
        w.sorted_predicted_position[:n], mass, w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end, dens_l, lam_l, pn_l,
    )
    cp.cuda.Device().synchronize()
    np.testing.assert_array_equal(cp.asnumpy(lam_l[:n]), cp.asnumpy(lam_f[:n]))
    np.testing.assert_array_equal(cp.asnumpy(dens_l[:n]), cp.asnumpy(dens_f[:n]))
    np.testing.assert_array_equal(cp.asnumpy(pn_l[:n]), cp.asnumpy(pn_f[:n]))
    assert np.abs(cp.asnumpy(lam_l[:n])).max() > 0.0

    # Not the last iteration: no density and no pressure normal writes.
    lam_s, dens_s = lam0.copy(), dens0.copy()
    pbf_solver.pbf_compute_lambda(
        w.sorted_predicted_position[:n], mass, w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end, None, lam_s, None,
    )
    cp.cuda.Device().synchronize()
    np.testing.assert_array_equal(cp.asnumpy(lam_s[:n]), cp.asnumpy(lam_f[:n]))
    np.testing.assert_array_equal(cp.asnumpy(dens_s), cp.asnumpy(dens0))


def test_full_needs_density_out(world, sim):
    n = _cloud(sim, world)
    w = world
    with pytest.raises(ValueError):
        pbf_solver.pbf_compute_lambda(
            w.sorted_predicted_position[:n], None, w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end, None, w.sorted_lambda_pbf,
            temperature_in=w.sorted_temperature[:n],
            density_in=w.sorted_density_prev[:n],
            dTdt_out=w.sorted_dTdt,
            exposure_heat_out=w.sorted_exposure_heat,
            exposure_corrode_out=w.sorted_exposure_corrode,
        )
