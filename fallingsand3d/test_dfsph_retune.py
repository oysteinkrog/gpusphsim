"""DFSPH alpha cap, unilateral divergence solver and profile defaults (bd-r5-review-n28.37).

- The alpha cap (alpha_limit * h^2) must not bind on interior particles at the
  shipped profiles: interior alpha equals 1/denom.
- K_DFSPH_ComputeKappaV corrects only compressing flow (div_v > 0) and skips
  particles with fewer than 20 neighbours (bd-r5-review-n28.71).
- SolverProfile(solver_type=DFSPH) carries the Stable DFSPH values.
- A settling WATER block at the Stable profile keeps the compression error
  under 1% and does not boil (the bilateral solver with the full diagonal did).
"""

import dataclasses

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

from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES, SolverProfile, SolverType  # noqa: E402
from materials import WATER, MATERIALS  # noqa: E402

MAX_PARTICLES = 60_000
H = 0.04


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


@pytest.fixture
def sim(world):
    return Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                      fixed_dt=False, max_substeps=20)


def _dfsph_fields(p):
    names = [f.name for f in dataclasses.fields(p)
             if f.name.startswith("dfsph_") or f.name == "velocity_limit_factor"]
    return {k: getattr(p, k) for k in names}


def test_dfsph_profile_defaults_match_stable():
    custom = SolverProfile(name="custom", solver_type=SolverType.DFSPH)
    assert _dfsph_fields(custom) == _dfsph_fields(PROFILES["DFSPH"])
    assert custom.velocity_limit_factor == 0.4
    assert SolverProfile(name="w", solver_type=SolverType.WCSPH).velocity_limit_factor == 0.9
    assert SolverProfile(name="d", solver_type=SolverType.DFSPH,
                         velocity_limit_factor=0.7).velocity_limit_factor == 0.7


def _block(w, sim, profile, half_n=5):
    """Spawn a (2*half_n+1)^3 WATER block at spacing 0.02 and run the density/alpha pass twice."""
    import dfsph_solver
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
    cp.cuda.Device().synchronize()
    return n


@pytest.mark.parametrize("profile_name", ["DFSPH", "DFSPH (Fast)"])
def test_dfsph_interior_alpha_not_capped(world, sim, profile_name):
    """Interior alpha must equal 1/denom (the alpha_limit = 1e9 value) at the shipped profile."""
    w = world

    def run(profile):
        n = _block(w, sim, profile)
        pos = w.sorted_position[:n, :3].get()
        order = np.lexsort(np.round(pos / 0.02).astype(np.int64).T)  # sort order may differ per run
        return pos[order], w.sorted_alpha_dfsph[:n].get()[order]

    pos_e, exact = run(dataclasses.replace(PROFILES[profile_name], dfsph_alpha_limit=1.0e9))
    pos, capped = run(PROFILES[profile_name])
    np.testing.assert_allclose(pos, pos_e, atol=1e-5)
    centre = pos.mean(axis=0)
    # Full kernel support: at least h inside the outermost particle layer.
    interior = np.max(np.abs(pos - centre), axis=1) <= 0.1 - H + 1e-4
    assert interior.sum() >= 100, f"only {interior.sum()} interior particles"
    ratio = capped[interior] / exact[interior]
    print(f"  {profile_name}: interior alpha mean={exact[interior].mean():.4e} "
          f"(= {exact[interior].mean() / H**2:.3f} h^2), capped/exact min={ratio.min():.4f}")
    np.testing.assert_allclose(capped[interior], exact[interior], rtol=1e-6)
    # The old dt^2 cap (alpha_limit 10 at dt = 1/300) held interior alpha at ~0.1x.
    old_cap = 10.0 * (1.0 / 300.0) ** 2
    assert exact[interior].min() > 5.0 * old_cap


def test_dfsph_divergence_solver_is_unilateral(world, sim):
    """Expanding flow (div_v < 0) gets kappa_v = 0; compressing flow is reduced each iteration."""
    import dfsph_solver
    w = world
    profile = PROFILES["DFSPH"]
    n = _block(w, sim, profile)
    dt = float(profile.dt)
    pos = w.sorted_position[:n, :3].get()
    centre = pos.mean(axis=0)
    alpha = w.sorted_alpha_dfsph[:n].get()
    k = 0.1 / (3.0 * dt)
    kv = cp.zeros(n, dtype=cp.float32)

    def kappa(vel):
        dfsph_solver.compute_kappa_v(
            vel, w.sorted_density[:n], w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
            w.sorted_packed_info[:n], w.sorted_position[:n],
            sim._cell_start, sim._cell_end, kv,
        )
        return kv.get()

    # Expansion: every particle separates, nothing to correct.
    vel = cp.zeros((n, 4), dtype=cp.float32)
    vel[:, :3] = cp.asarray((k * (pos - centre)).astype(np.float32))
    kv_exp = kappa(vel)
    assert np.all(kv_exp == 0.0), f"expanding flow got kappa_v != 0 at {np.count_nonzero(kv_exp)} particles"

    # Compression: the mean positive divergence must drop monotonically.
    vel[:, :3] = cp.asarray((-k * (pos - centre)).astype(np.float32))
    act = alpha > 0.0
    divs = []
    for _ in range(profile.dfsph_div_iters + 1):
        kv_h = kappa(vel)
        assert np.all(kv_h >= 0.0)
        divs.append(float(np.mean(kv_h[act] * dt / (alpha[act] * profile.dfsph_omega))))
        dfsph_solver.correct_velocity_div(
            vel, w.sorted_density[:n], w.sorted_mass[:n], kv,
            w.sorted_packed_info[:n], w.sorted_position[:n],
            sim._cell_start, sim._cell_end,
        )
    print("  mean positive div_v per iteration:", ", ".join(f"{d:.3f}" for d in divs))
    assert all(b < a for a, b in zip(divs, divs[1:])), f"divergence not decreasing: {divs}"
    assert divs[-1] < 0.5 * divs[0], f"divergence solver too weak: {divs}"


def test_dfsph_stable_profile_settles_without_boiling(world):
    """36K WATER block dropped to the floor, 2 s at the Stable profile."""
    w = world
    rho0 = float(MATERIALS[WATER].rest_density)
    profile = PROFILES["DFSPH"]
    w.packed_info[:] = 0
    w._high_water = 0
    n = w.spawn_cube((-0.3, -0.9, -0.3), (0.3, -0.1, 0.3), WATER, spacing=0.02)
    s = Simulation(w, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    s.set_solver_profile(profile)
    s._spawn_substep = s._damping_duration
    for _ in range(int(round(2.0 / profile.dt))):
        s._sim_step(n)
    cp.cuda.Device().synchronize()
    pi = w.sorted_packed_info[:n].get()
    rho = w.sorted_density[:n].get()
    fluid = (((pi >> 8) & 3) == 0) & ((pi & 0xFF) != 0)
    comp = float(np.mean(np.maximum(rho[fluid] / rho0 - 1.0, 0.0)))
    v = w.velocity[:n, :3].get()
    m = w.mass[:n].get()
    ke = float(0.5 * np.sum(m * np.sum(v * v, axis=1)))
    print(f"  Stable DFSPH at 2 s: compression error={comp * 100:.2f}%  KE={ke:.1f}")
    assert np.isfinite(v).all()
    assert comp < 0.01, f"compression error {comp * 100:.2f}% >= 1%"
    # Boiling free surface (bilateral solver with the full diagonal) gave KE > 1000.
    assert ke < 200.0, f"KE {ke:.1f} at 2 s: the free surface is not settling"


def test_dfsph_divergence_solver_skips_sparse_neighbourhoods(world, sim):
    """Particles with fewer than 20 neighbours get kappa_v = 0 (bd-r5-review-n28.71)."""
    import dfsph_solver
    w = world
    profile = PROFILES["DFSPH"]
    n = _block(w, sim, profile)
    dt = float(profile.dt)
    pos = w.sorted_position[:n, :3].get()
    centre = pos.mean(axis=0)
    nc = w.sorted_normal[:n, 3].get()
    vel = cp.zeros((n, 4), dtype=cp.float32)
    vel[:, :3] = cp.asarray((-0.1 / (3.0 * dt) * (pos - centre)).astype(np.float32))

    def kappa(normal_in):
        kv = cp.zeros(n, dtype=cp.float32)
        dfsph_solver.compute_kappa_v(
            vel, w.sorted_density[:n], w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
            w.sorted_packed_info[:n], w.sorted_position[:n],
            sim._cell_start, sim._cell_end, kv, normal_in=normal_in,
        )
        return kv.get()

    ungated = kappa(None)
    gated = kappa(w.sorted_normal[:n])
    sparse = nc < 20.0
    print(f"  neighbour count min={nc.min():.0f} max={nc.max():.0f}, "
          f"{sparse.sum()} of {n} below 20")
    assert 0 < sparse.sum() < n
    assert np.all(gated >= 0.0)
    assert np.all(gated[sparse] == 0.0)
    assert np.count_nonzero(ungated[sparse]) > 0, "test needs compressing sparse particles"
    np.testing.assert_array_equal(gated[~sparse], ungated[~sparse])
    assert np.count_nonzero(gated[~sparse]) > 0
