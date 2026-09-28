"""bd-r5-review-n28.185: PBF predicted positions ping-pong and carry mass in .w.

K_PBF_ComputeDelta applies its correction into the other predicted-position
buffer (K_PBF_ApplyDelta is gone). K_PBF_Predict writes the mass into
predicted.w, the first delta pass rewrites it from sorted_mass, and the later
passes and K_PBF_Finalize read it from .w.

Checks:
- Predict writes predicted.w = mass, STATIC included.
- One fused delta pass equals x* + clamp(dx) from a float64 reference of the
  delta formula (the old ComputeDelta + ApplyDelta pair) on a random cloud.
- Reading mass from .w gives the same lambda and delta as the mass array.
- The full _run_pbf_body at 4 and 5 iterations (even and odd final buffer)
  equals a reference pipeline that reads mass from the array every pass,
  applies each delta back into buffer A, and gives Finalize the mass
  array.
"""

import dataclasses

import numpy as np
import pytest

import cupy as cp

import pbf_solver
from world import World
from simulation import Simulation
from solver_profiles import PROFILES
from materials import WATER, SAND, STONE, STATIC as B_STATIC, GAS as B_GAS

MAX_PARTICLES = 20_000
H = 0.04


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


def _random_cloud(world, n_fluid=600, seed=3):
    """WATER and SAND scattered in a 0.2 m box above a STONE slab."""
    world.packed_info[:] = 0
    world._high_water = 0
    world.spawn_cube((-0.12, -0.14, -0.12), (0.12, -0.1, 0.12), STONE, 0.02)
    n_w = world.spawn_cube((-0.1, -0.1, -0.1), (0.0, 0.1, 0.1), WATER, 0.02)
    n_s = world.spawn_cube((0.0, -0.1, -0.1), (0.1, 0.1, 0.1), SAND, 0.02)
    n = world._high_water
    assert n_w > 0 and n_s > 0
    rng = np.random.default_rng(seed)
    pos = cp.asnumpy(world.position[:n]).copy()
    mat = cp.asnumpy(world.packed_info[:n]) & 0xFF
    moving = mat != STONE
    pos[moving, :3] += rng.uniform(-0.006, 0.006, size=(int(moving.sum()), 3)).astype(np.float32)
    world.position[:n] = cp.asarray(pos)
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[moving, :3] = rng.uniform(-0.3, 0.3, size=(int(moving.sum()), 3)).astype(np.float32)
    world.velocity[:n] = cp.asarray(vel)
    world._density_initialized = True
    return n


def _sort_predict_lambda(sim, world, n):
    w = world
    sim._run_grid_setup(n, force_sort=True)
    pbf_solver.pbf_predict(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        w.sorted_predicted_position, mass=w.sorted_mass[:n],
    )
    pbf_solver.pbf_compute_lambda(
        w.sorted_predicted_position[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], sim._cell_start, sim._cell_end,
        w.sorted_density, w.sorted_lambda_pbf,
    )


def test_predict_writes_mass_in_w(world, sim):
    n = _random_cloud(world)
    _sort_predict_lambda(sim, world, n)
    pred_w = cp.asnumpy(world.sorted_predicted_position[:n, 3])
    mass = cp.asnumpy(world.sorted_mass[:n])
    beh = (cp.asnumpy(world.sorted_packed_info[:n]) >> 8) & 0x3
    assert (beh == B_STATIC).any() and (mass > 0).all()
    np.testing.assert_array_equal(pred_w, mass)


def _reference_delta(pred, lam, mass, pi, rho0_table):
    """float64 dx_i = sum_j m_j (l_i + l_j * rho0_i/rho0_j) / rho0_i gradW,
    lambda_j mirrored to lambda_i for STATIC j, clamped to 0.5h."""
    x = pred[:, :3].astype(np.float64)
    beh = (pi >> 8) & 0x3
    mat = pi & 0xFF
    rho0 = np.maximum(rho0_table[mat], 1.0)
    coeff = -45.0 / (np.pi * H ** 6)
    n = len(x)
    out = np.zeros((n, 3))
    for i in range(n):
        if beh[i] == B_STATIC:
            continue
        r = x[i] - x
        r2 = (r * r).sum(axis=1)
        sel = (r2 <= H * H) & (r2 >= 1e-12)
        sel[i] = False
        sel &= beh != B_GAS
        rl = np.sqrt(r2[sel])
        gw = (coeff * (H - rl) ** 2 / rl)[:, None] * r[sel]
        lj = np.where(beh[sel] == B_STATIC, lam[i], lam[sel] * (rho0[i] / rho0[sel]))
        c = mass[sel] * (lam[i] + lj) / rho0[i]
        d = (c[:, None] * gw).sum(axis=0)
        dl = np.linalg.norm(d)
        if dl > 0.5 * H:
            d *= 0.5 * H / dl
        out[i] = d
    return out


def test_fused_delta_matches_reference(world, sim):
    n = _random_cloud(world)
    _sort_predict_lambda(sim, world, n)
    w = world
    out = cp.zeros((n, 4), dtype=cp.float32)
    pbf_solver.pbf_compute_delta(
        w.sorted_predicted_position[:n], w.sorted_lambda_pbf[:n],
        w.sorted_mass[:n], w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end, out,
    )
    pred = cp.asnumpy(w.sorted_predicted_position[:n])
    lam = cp.asnumpy(w.sorted_lambda_pbf[:n]).astype(np.float64)
    mass = cp.asnumpy(w.sorted_mass[:n]).astype(np.float64)
    pi = cp.asnumpy(w.sorted_packed_info[:n]).astype(np.int64)
    from materials import MATERIALS
    rho0_table = np.zeros(256)
    for mid, m in MATERIALS.items():
        rho0_table[int(mid)] = m.rest_density
    ref = pred[:, :3].astype(np.float64) + _reference_delta(pred, lam, mass, pi, rho0_table)
    got = cp.asnumpy(out)
    d_ref = np.linalg.norm(ref - pred[:, :3], axis=1)
    assert d_ref.max() > 1e-5, "scene gave no correction"
    err = np.abs(got[:, :3] - ref).max()
    assert err < 1e-6 + 1e-3 * d_ref.max(), f"fused delta off by {err:.3e} (max delta {d_ref.max():.3e})"
    # .w is the mass; STATIC keeps its position.
    np.testing.assert_array_equal(got[:, 3], mass.astype(np.float32))
    static = ((pi >> 8) & 0x3) == B_STATIC
    np.testing.assert_array_equal(got[static, :3], pred[static, :3])


def test_mass_from_w_matches_mass_array(world, sim):
    n = _random_cloud(world)
    _sort_predict_lambda(sim, world, n)
    w = world
    dens_a = w.sorted_density[:n].copy()
    lam_a = w.sorted_lambda_pbf[:n].copy()
    pbf_solver.pbf_compute_lambda(
        w.sorted_predicted_position[:n], None,
        w.sorted_packed_info[:n], sim._cell_start, sim._cell_end,
        w.sorted_density, w.sorted_lambda_pbf,
    )
    assert bool(cp.array_equal(w.sorted_density[:n], dens_a))
    assert bool(cp.array_equal(w.sorted_lambda_pbf[:n], lam_a))

    out_a = cp.zeros((n, 4), dtype=cp.float32)
    out_b = cp.zeros((n, 4), dtype=cp.float32)
    for m, out in ((w.sorted_mass[:n], out_a), (None, out_b)):
        pbf_solver.pbf_compute_delta(
            w.sorted_predicted_position[:n], w.sorted_lambda_pbf[:n],
            m, w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end, out,
        )
    assert bool(cp.array_equal(out_a, out_b))


def test_delta_rejects_aliased_output(world, sim):
    n = _random_cloud(world)
    _sort_predict_lambda(sim, world, n)
    w = world
    with pytest.raises(ValueError):
        pbf_solver.pbf_compute_delta(
            w.sorted_predicted_position[:n], w.sorted_lambda_pbf[:n],
            w.sorted_mass[:n], w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end, w.sorted_predicted_position,
        )


def _reference_body(sim, world, n, iters):
    """Old data flow: mass array every pass, delta applied back into A,
    pressure normal every lambda call, Finalize reads the mass array."""
    w = world
    pbf_solver.pbf_predict(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        w.sorted_predicted_position,
    )
    # Previous density in this substep's slot order (bd-r5-review-n28.216).
    cp.copyto(w.sorted_density_prev[:n], w.sorted_position[:n, 3])
    pbf_solver.pbf_compute_lambda(
        w.sorted_predicted_position[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], sim._cell_start, sim._cell_end,
        w.sorted_density, w.sorted_lambda_pbf, w.sorted_pressure_normal,
        temperature_in=w.sorted_temperature[:n],
        density_in=w.sorted_density_prev[:n],
        dTdt_out=w.sorted_dTdt,
        exposure_heat_out=w.sorted_exposure_heat,
        exposure_corrode_out=w.sorted_exposure_corrode,
        velocity_in=w.sorted_velocity[:n],
        vorticity_out=w.sorted_vorticity,
        normal_out=w.sorted_normal,
    )
    tmp = cp.zeros_like(w.sorted_predicted_position)
    for it in range(iters):
        if it > 0:
            pbf_solver.pbf_compute_lambda(
                w.sorted_predicted_position[:n], w.sorted_mass[:n],
                w.sorted_packed_info[:n], sim._cell_start, sim._cell_end,
                w.sorted_density, w.sorted_lambda_pbf, w.sorted_pressure_normal,
            )
        pbf_solver.pbf_compute_delta(
            w.sorted_predicted_position[:n], w.sorted_lambda_pbf[:n],
            w.sorted_mass[:n], w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end, tmp,
        )
        cp.copyto(w.sorted_predicted_position[:n], tmp[:n])
    pbf_solver.pbf_finalize(
        w.sorted_predicted_position[:n], w.sorted_position[:n],
        w.sorted_velocity[:n], w.sorted_density[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        w.sorted_health[:n], w.sorted_dTdt[:n], w.sorted_sleep_counter[:n],
        sim._sort_perm[:n], sim._cell_start, sim._cell_end,
        w.position, w.velocity, w.packed_info,
        w.sleep_counter, w.temperature,
        vorticity_in=w.sorted_vorticity[:n],
        normal_in=w.sorted_normal[:n],
        pressure_normal_in=w.sorted_pressure_normal[:n],
    )


@pytest.mark.parametrize("iters", [4, 5])
def test_body_matches_reference_pipeline(world, sim, monkeypatch, iters):
    sim.set_solver_profile(dataclasses.replace(PROFILES["PBF"], pbf_iterations=iters))
    sim._spawn_substep = sim._damping_duration
    n = _random_cloud(world)
    w = world
    sim._run_grid_setup(n, force_sort=True)
    dens0 = w.sorted_density[:n].copy()

    _reference_body(sim, w, n, iters)
    cp.cuda.Device().synchronize()
    ref_pos = cp.asnumpy(w.position[:n])
    ref_vel = cp.asnumpy(w.velocity[:n])

    # Same sorted state: no re-sort, no reactions, same previous density.
    cp.copyto(w.sorted_density[:n], dens0)
    monkeypatch.setattr(sim, "_run_grid_setup", lambda *a, **k: None)
    monkeypatch.setattr(sim, "_run_reactions_spawn", lambda *a, **k: None)
    sim._run_pbf_body(n)
    cp.cuda.Device().synchronize()
    pos = cp.asnumpy(w.position[:n])
    vel = cp.asnumpy(w.velocity[:n])

    moved = np.linalg.norm(pos[:, :3] - cp.asnumpy(w.sorted_position[:n])[:, :3].mean(0), axis=1)
    assert np.isfinite(pos).all() and moved.max() > 0
    np.testing.assert_allclose(pos[:, :3], ref_pos[:, :3], rtol=0, atol=1e-6)
    np.testing.assert_allclose(vel[:, :3], ref_vel[:, :3], rtol=0, atol=1e-4)
    vmax = np.abs(ref_vel[:, :3]).max()
    assert vmax > 0.05, f"scene too quiet: max |v| {vmax:.3g}"
