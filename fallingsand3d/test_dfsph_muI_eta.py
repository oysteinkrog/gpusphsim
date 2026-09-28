"""Regression test for bd-r5-review-n28.219: DFSPH mu(I) eta computed once.

K_DFSPH_PackVolume writes the GRANULAR mu(I) eta per particle to eta_out
(world.sorted_eta), and K_DFSPH_NonPressureForces reads it per GRANULAR pair
(eta_in) instead of calling compute_muI_eta for every pair. With eta_in NULL
the kernel recomputes eta from shear_rate and p_prev the old way. A carried
eta that is not above 0 also falls back to the recompute.

Checks, on the Sand Castle preset under the DFSPH profile:
- every 30 substeps over 300, NonPressureForces gives the same velocity with
  eta_in as without it (rtol 1e-5);
- eta is > 0 on GRANULAR and 0 elsewhere after a live substep;
- a particle made GRANULAR after PackVolume (eta 0) gets the recompute;
- the 300-substep trajectory with eta_in stays within the run-to-run noise
  of the per-pair path.
"""

import numpy as np
import pytest

import cupy as cp
import dfsph_solver
import presets
from materials import GRANULAR, WATER, WET_SAND
from simulation import Simulation
from solver_profiles import PROFILES
from world import World

SUBSTEPS = 300
CHECK_EVERY = 30
RTOL = 1e-5


def _sand_castle():
    profile = PROFILES["DFSPH"]
    world = World(max_particles=400_000)
    world.packed_info[:] = 0
    world._high_water = 0
    sim = Simulation(world, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    sim.set_solver_profile(profile)
    sim.fixed_dt_value = profile.dt
    presets.load_sand_castle(world)
    sim._spawn_substep = sim._damping_duration  # no spawn damping ramp
    return world, sim


def _eta(sim, n, packed_info=None):
    """Run PackVolume on a copy of this substep's sorted positions, eta on."""
    w = sim.world
    pos = w.sorted_position[:n].copy()
    eta = cp.full(n, -1.0, dtype=cp.float32)
    dfsph_solver.pack_volume(
        pos,
        w.sorted_density[:n],
        w.sorted_mass[:n],
        w.sorted_packed_info[:n] if packed_info is None else packed_info,
        shear_rate=w.sorted_shear_rate[:n],
        p_prev=w.sorted_kappa[:n],
        eta_out=eta,
    )
    return eta


def _non_pressure(sim, n, eta_in):
    """K_DFSPH_NonPressureForces on this substep's sorted state (no rigid bodies)."""
    w = sim.world
    out = cp.zeros((n, 4), dtype=cp.float32)
    dfsph_solver.compute_non_pressure_forces(
        w.sorted_position[:n],
        w.sorted_velocity[:n],
        w.sorted_density[:n],
        w.sorted_mass[:n],
        w.sorted_packed_info[:n],
        w.sorted_shear_rate[:n],
        w.sorted_temperature[:n],
        sim._cell_start,
        sim._cell_end,
        out,
        vorticity_in=w.sorted_vorticity[:n],
        normal_in=w.sorted_normal[:n],
        p_prev=w.sorted_kappa[:n],
        eta_in=eta_in,
    )
    return cp.asnumpy(out)


def _assert_close(a, b, what):
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    np.testing.assert_allclose(
        a, b, rtol=RTOL, atol=RTOL * max(scale, 1e-6), err_msg=what
    )


def _behavior(world, n):
    return (cp.asnumpy(world.sorted_packed_info[:n]) >> 8) & 0x3


@pytest.fixture(scope="module")
def castle():
    return _sand_castle()


@pytest.mark.gpu
def test_eta_matches_per_pair_muI_over_300_substeps(castle):
    world, sim = castle
    n = world._high_water
    checks = 0
    for step in range(1, SUBSTEPS + 1):
        sim._sim_step(n)
        if step % CHECK_EVERY:
            continue
        eta = _eta(sim, n)
        v_new = _non_pressure(sim, n, eta)
        v_old = _non_pressure(sim, n, None)
        assert np.all(np.isfinite(v_new))
        _assert_close(v_new, v_old, f"velocity_out at substep {step}")

        # Written for every particle: > 0 for GRANULAR, 0 for the rest.
        eta_h = cp.asnumpy(eta)
        gran = _behavior(world, n) == GRANULAR
        assert gran.sum() > 0
        assert np.all(eta_h[gran] > 0.0)
        assert np.all(eta_h[~gran] == 0.0)
        checks += 1
    assert checks == SUBSTEPS // CHECK_EVERY


@pytest.mark.gpu
def test_live_dfsph_substep_writes_sorted_eta(castle):
    """simulation.py passes world.sorted_eta to PackVolume and
    NonPressureForces, so a plain _sim_step leaves eta > 0 on GRANULAR and
    0 elsewhere, equal to a fresh PackVolume on the same inputs."""
    world, sim = castle
    n = world._high_water
    world.sorted_eta.fill(-1.0)
    sim._sim_step(n)
    eta = cp.asnumpy(world.sorted_eta[:n])
    gran = _behavior(world, n) == GRANULAR
    assert gran.sum() > 0
    assert np.all(eta[gran] > 0.0)
    assert np.all(eta[~gran] == 0.0)


@pytest.mark.gpu
def test_eta_fallback_for_particle_made_granular_after_pack_volume(castle):
    """PackVolume writes eta = 0 for WATER. If WATER becomes GRANULAR before
    NonPressureForces reads eta, the kernel must recompute eta and match the
    per-pair path."""
    world, sim = castle
    n = world._high_water
    sim._sim_step(n)
    eta = _eta(sim, n)

    pi = world.sorted_packed_info[:n]
    saved = pi.copy()
    is_water = (pi & 0xFF) == WATER
    assert int(cp.count_nonzero(is_water)) > 0
    new_pi = (pi & ~cp.uint32(0x3FF)) | cp.uint32(WET_SAND) | cp.uint32(GRANULAR << 8)
    world.sorted_packed_info[:n] = cp.where(is_water, new_pi, pi)
    try:
        assert np.all(cp.asnumpy(eta[is_water]) == 0.0)
        v_new = _non_pressure(sim, n, eta)
        v_old = _non_pressure(sim, n, None)
    finally:
        world.sorted_packed_info[:n] = saved
    assert np.all(np.isfinite(v_new))
    _assert_close(v_new, v_old, "velocity_out after FLUID -> GRANULAR conversion")


def _run_castle(use_eta, monkeypatch):
    """300 DFSPH substeps of Sand Castle; returns shape stats of the GRANULAR
    particles. The unsorted slot order differs between runs, so compare
    order-free statistics: the centre of mass and the 5..95 percentiles of
    height and horizontal radius."""
    orig = dfsph_solver.compute_non_pressure_forces

    def per_pair(*args, **kwargs):
        kwargs["eta_in"] = None
        return orig(*args, **kwargs)

    with monkeypatch.context() as m:
        if not use_eta:
            m.setattr(dfsph_solver, "compute_non_pressure_forces", per_pair)
        world, sim = _sand_castle()
        n = world._high_water
        for _ in range(SUBSTEPS):
            sim._sim_step(n)
        cp.cuda.Device().synchronize()
    pos = cp.asnumpy(world.position[:n, :3])
    gran = ((cp.asnumpy(world.packed_info[:n]) >> 8) & 0x3) == GRANULAR
    p = pos[gran]
    assert np.all(np.isfinite(p))
    q = np.arange(5, 100, 5)
    r_xz = np.hypot(p[:, 0], p[:, 2])
    stats = np.concatenate(
        [p.mean(axis=0), np.percentile(p[:, 1], q), np.percentile(r_xz, q)]
    )
    return stats, int(gran.sum())


@pytest.mark.gpu
def test_sand_castle_trajectory_matches_per_pair_path(monkeypatch):
    """After 300 substeps the GRANULAR pile shape with eta_in differs from the
    per-pair path by no more than two per-pair runs differ from each other
    (atomic scatter order in the sort makes runs not bit-identical), with a
    0.1 mm floor."""
    s_a, n_a = _run_castle(False, monkeypatch)
    s_b, n_b = _run_castle(False, monkeypatch)
    s_e, n_e = _run_castle(True, monkeypatch)
    assert n_a > 100_000
    # Reactions (SAND + WATER -> WET_SAND) vary slightly run to run.
    assert abs(n_e - n_a) <= 1e-3 * n_a and abs(n_b - n_a) <= 1e-3 * n_a
    noise = np.abs(s_a - s_b).max()
    diff = np.abs(s_e - s_a).max()
    print(
        f"max shape stat delta: per-pair run to run {noise:.3e} m, eta vs per-pair {diff:.3e} m"
    )
    assert diff <= max(2.0 * noise, 1e-4)
