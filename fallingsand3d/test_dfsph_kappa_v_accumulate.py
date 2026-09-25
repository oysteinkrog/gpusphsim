"""DFSPH divergence warm start carries the accumulated kappa_v (bd-r5-review-n28.83).

- K_DFSPH_ComputeKappaV writes the per-iteration increment to kappa_v_out and
  adds it to kappa_v_total at the particle's own index.
- Warm-starting the divergence solve from the accumulated total leaves a
  lower residual than warm-starting from the last increment (the old
  behaviour), on a WATER block under a steady compressing flow.
- The full DFSPH substep keeps sorted_kappa_v >= 0 and finite.
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
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import WATER  # noqa: E402

MAX_PARTICLES = 60_000


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


@pytest.fixture
def sim(world):
    return Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                      fixed_dt=False, max_substeps=20)


def _block(w, sim, profile, half_n=5):
    """Spawn a (2*half_n+1)^3 WATER block at spacing 0.02 and run the density/alpha pass twice."""
    w.packed_info[:] = 0
    w._high_water = 0
    half = 0.02 * half_n + 0.01
    n = w.spawn_cube((-half, 0.3 - half, -half), (half, 0.3 + half, half), WATER, spacing=0.02)
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
            normal_out=w.sorted_normal,
        )
    dfsph_solver.pack_volume(
        w.sorted_position[:n], w.sorted_density[:n],
        w.sorted_mass[:n], w.sorted_packed_info[:n],
    )
    cp.cuda.Device().synchronize()
    return n


def _kappa(w, sim, n, vel, out, total=None):
    dfsph_solver.compute_kappa_v(
        vel, w.sorted_alpha_dfsph[:n],
        w.sorted_packed_info[:n], w.sorted_position[:n],
        sim._cell_start, sim._cell_end, out,
        normal_in=w.sorted_normal[:n], kappa_v_total=total,
    )


def _correct(w, sim, n, vel, kv):
    dfsph_solver.correct_velocity_div(
        vel, w.sorted_density[:n], w.sorted_mass[:n], kv,
        w.sorted_packed_info[:n], w.sorted_position[:n],
        sim._cell_start, sim._cell_end,
    )


def _compressing_velocity(w, n, dt):
    pos = w.sorted_position[:n, :3].get()
    centre = pos.mean(axis=0)
    vel = cp.zeros((n, 4), dtype=cp.float32)
    vel[:, :3] = cp.asarray((-0.1 / (3.0 * dt) * (pos - centre)).astype(np.float32))
    return vel


def test_compute_kappa_v_adds_increment_to_total(world, sim):
    w = world
    profile = PROFILES["DFSPH"]
    n = _block(w, sim, profile)
    vel = _compressing_velocity(w, n, float(profile.dt))

    ref = cp.zeros(n, dtype=cp.float32)
    _kappa(w, sim, n, vel, ref)
    start = cp.asarray(np.linspace(0.0, 3.0, n, dtype=np.float32))
    total = start.copy()
    delta = cp.full(n, -1.0, dtype=cp.float32)
    _kappa(w, sim, n, vel, delta, total)

    ref_h, delta_h, total_h = ref.get(), delta.get(), total.get()
    np.testing.assert_array_equal(delta_h, ref_h)
    np.testing.assert_allclose(total_h, start.get() + ref_h, rtol=1e-6, atol=1e-6)
    assert np.count_nonzero(delta_h) > 0


def test_accumulated_warm_start_lowers_residual(world, sim):
    """Second substep of a steady compressing flow, warm-started two ways."""
    w = world
    profile = PROFILES["DFSPH"]
    n = _block(w, sim, profile)
    dt = float(profile.dt)
    ws = float(profile.dfsph_div_warm_start)
    alpha = w.sorted_alpha_dfsph[:n].get()
    act = alpha > 0.0
    v0 = _compressing_velocity(w, n, dt)
    scratch = cp.zeros(n, dtype=cp.float32)

    def mean_div(vel):
        _kappa(w, sim, n, vel, scratch)
        kv = scratch.get()
        return float(np.mean(kv[act] * dt / (alpha[act] * profile.dfsph_omega)))

    # Substep 1 from a cold start: record the total and the last increment.
    vel = v0.copy()
    total = cp.zeros(n, dtype=cp.float32)
    delta = cp.zeros(n, dtype=cp.float32)
    for _ in range(profile.dfsph_div_iters):
        _kappa(w, sim, n, vel, delta, total)
        _correct(w, sim, n, vel, delta)
    last = delta.copy()

    cold = mean_div(v0)
    results = {}
    for name, carried in (("last increment", last), ("accumulated", total)):
        vel = v0.copy()
        _correct(w, sim, n, vel, carried * ws)
        results[name] = mean_div(vel)
    print(f"  mean positive div_v: cold={cold:.4f}, after warm pass: "
          + ", ".join(f"{k}={v:.4f}" for k, v in results.items()))
    assert results["accumulated"] < results["last increment"] < cold


def test_dfsph_substep_keeps_kappa_v_total_valid(world):
    w = world
    profile = PROFILES["DFSPH"]
    w.packed_info[:] = 0
    w._high_water = 0
    n = w.spawn_cube((-0.2, -0.9, -0.2), (0.2, -0.5, 0.2), WATER, spacing=0.02)
    s = Simulation(w, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    s.set_solver_profile(profile)
    s._spawn_substep = s._damping_duration
    for _ in range(60):
        s._sim_step(n)
    cp.cuda.Device().synchronize()
    kv = w.kappa_v[:n].get()
    v = w.velocity[:n, :3].get()
    assert np.isfinite(kv).all() and np.isfinite(v).all()
    assert np.all(kv >= 0.0)
    assert np.count_nonzero(kv) > 0, "warm start never carried a kappa_v"
