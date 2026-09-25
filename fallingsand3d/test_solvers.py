"""Headless solver diagnostic -- runs each solver for a few frames and prints stats."""

import sys
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

import cupy as cp
from world import World
from simulation import Simulation
from solver_profiles import PROFILES, SolverType
from materials import WATER, SAND

MAX_PARTICLES = 100_000


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    """Create a World instance shared across all tests in this module."""
    return World(max_particles=MAX_PARTICLES)


@pytest.fixture
def sim(world):
    """Create a Simulation instance for each test (uses module-scoped world)."""
    return Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                      fixed_dt=False, max_substeps=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spawn_scene(world):
    """Spawn water cube + sand bed."""
    world.packed_info[:] = 0
    world._high_water = 0
    n_water = world.spawn_cube(
        min_corner=(-0.2, 0.3, -0.2),
        max_corner=(0.2, 0.7, 0.2),
        material_id=WATER,
        spacing=0.02,
    )
    n_sand = world.spawn_cube(
        min_corner=(-0.8, -0.5, -0.8),
        max_corner=(0.8, -0.3, 0.8),
        material_id=SAND,
        spacing=0.04,
    )
    return n_water + n_sand


def diagnose(world, label, sorted_arrays=None):
    """Print diagnostic stats."""
    n = world._high_water
    pos = world.position[:n].get()
    vel = world.velocity[:n].get()
    pi = world.packed_info[:n].get()
    active = pi != 0
    n_active = active.sum()
    if n_active == 0:
        print(f"  [{label}] NO ACTIVE PARTICLES!")
        return False
    pos_a = pos[active, :3]
    vel_a = vel[active, :3]
    vel_mag = np.linalg.norm(vel_a, axis=1)
    has_nan = np.isnan(pos_a).any() or np.isnan(vel_a).any()
    has_inf = np.isinf(pos_a).any() or np.isinf(vel_a).any()
    print(f"  [{label}] n={n_active}  pos_y=[{pos_a[:,1].min():.3f},{pos_a[:,1].mean():.3f},{pos_a[:,1].max():.3f}]  "
          f"vel=[{vel_mag.mean():.4f},{vel_mag.max():.4f}]  nan={has_nan}  inf={has_inf}")

    if sorted_arrays:
        for name, arr in sorted_arrays.items():
            a = arr[:n].get() if hasattr(arr, 'get') else arr[:n]
            if a.ndim == 1:
                print(f"    {name}: min={a.min():.6f} mean={a.mean():.6f} max={a.max():.6f} "
                      f"nonzero={np.count_nonzero(a)}/{len(a)}")
            elif a.ndim == 2:
                mag = np.linalg.norm(a[:, :3], axis=1)
                print(f"    {name}: mag min={mag.min():.6f} mean={mag.mean():.6f} max={mag.max():.6f}")
    return not has_nan and not has_inf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pbf_detailed(world, sim):
    """Test PBF with intermediate diagnostics."""
    import pbf_solver
    profile = PROFILES["PBF"]
    print(f"\n{'='*60}")
    print(f"PBF Detailed Test (dt={profile.dt:.6f})")
    print(f"{'='*60}")

    n = spawn_scene(world)
    sim.set_solver_profile(profile)
    sim.sim_time = 0.0
    sim._last_frame_time = None

    # Run 3 substeps with detailed intermediate checks
    for step in range(3):
        print(f"\n  --- Substep {step} ---")

        w = world
        sim._frame_counter_d.fill(sim._frame_counter)
        w._density_initialized = True

        # Grid setup: full counting sort, as in the app
        sim._run_grid_setup(n, force_sort=True)

        # Check sorted positions
        sp = w.sorted_position[:n].get()
        sv = w.sorted_velocity[:n].get()
        print(f"  sorted_pos_y: [{sp[:,1].min():.4f}, {sp[:,1].mean():.4f}, {sp[:,1].max():.4f}]")
        print(f"  sorted_vel_mag: mean={np.linalg.norm(sv[:,:3], axis=1).mean():.4f}")

        # PBF Predict
        pbf_solver.pbf_predict(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_predicted_position,
        )
        cp.cuda.Device().synchronize()
        pred = w.sorted_predicted_position[:n].get()
        pred_mag = np.linalg.norm(pred[:,:3] - sp[:,:3], axis=1)
        print(f"  predicted displacement: mean={pred_mag.mean():.6f} max={pred_mag.max():.6f}")
        print(f"  predicted_pos_y: [{pred[:,1].min():.4f}, {pred[:,1].mean():.4f}, {pred[:,1].max():.4f}]")

        # PBF ComputeLambda (first iteration)
        pbf_solver.pbf_compute_lambda(
            w.sorted_predicted_position[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end,
            w.sorted_density, w.sorted_lambda_pbf,
        )
        cp.cuda.Device().synchronize()

        dens = w.sorted_density[:n].get()
        lam = w.sorted_lambda_pbf[:n].get()
        print(f"  density: min={dens.min():.2f} mean={dens.mean():.2f} max={dens.max():.2f} "
              f"nonzero={np.count_nonzero(dens)}/{n}")
        print(f"  lambda:  min={lam.min():.6f} mean={lam.mean():.6f} max={lam.max():.6f} "
              f"nonzero={np.count_nonzero(lam)}/{n}")

        # Full PBF iteration
        for it in range(profile.pbf_iterations):
            if it > 0:
                pbf_solver.pbf_compute_lambda(
                    w.sorted_predicted_position[:n], w.sorted_mass[:n],
                    w.sorted_packed_info[:n],
                    sim._cell_start, sim._cell_end,
                    w.sorted_density, w.sorted_lambda_pbf,
                )
            pbf_solver.pbf_compute_delta(
                w.sorted_predicted_position[:n], w.sorted_lambda_pbf[:n],
                w.sorted_mass[:n], w.sorted_packed_info[:n],
                sim._cell_start, sim._cell_end,
                w.sorted_delta_position,
            )
            cp.cuda.Device().synchronize()
            delta = w.sorted_delta_position[:n].get()
            delta_mag = np.linalg.norm(delta[:,:3], axis=1)
            print(f"  iter {it}: delta mag min={delta_mag.min():.8f} mean={delta_mag.mean():.8f} max={delta_mag.max():.8f}")

            pbf_solver.pbf_apply_delta(
                w.sorted_predicted_position[:n], w.sorted_delta_position[:n],
                w.sorted_packed_info[:n],
            )

        # Finalize — API matches current pbf_finalize positional signature
        pbf_solver.pbf_finalize(
            w.sorted_predicted_position[:n], w.sorted_position[:n],
            w.sorted_velocity[:n], w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_dTdt[:n], w.sorted_sleep_counter[:n],
            sim._sort_perm[:n], sim._cell_start, sim._cell_end,
            w.position, w.velocity, w.packed_info,
            w.sleep_counter, w.temperature,
        )
        cp.cuda.Device().synchronize()

        # Check output
        out_pos = w.position[:n].get()
        out_vel = w.velocity[:n].get()
        out_vel_mag = np.linalg.norm(out_vel[:,:3], axis=1)
        print(f"  output pos_y: [{out_pos[:,1].min():.4f}, {out_pos[:,1].mean():.4f}, {out_pos[:,1].max():.4f}]")
        print(f"  output vel: mean={out_vel_mag.mean():.4f} max={out_vel_mag.max():.4f}")

        sim._frame_counter += 1


def test_dfsph_detailed(world, sim):
    """Test DFSPH with intermediate diagnostics."""
    import dfsph_solver
    profile = PROFILES["DFSPH"]
    print(f"\n{'='*60}")
    print(f"DFSPH Detailed Test (dt={profile.dt:.6f})")
    print(f"{'='*60}")

    n = spawn_scene(world)
    sim.set_solver_profile(profile)
    sim.sim_time = 0.0
    sim._last_frame_time = None

    for step in range(3):
        print(f"\n  --- Substep {step} ---")

        w = world

        sim._frame_counter_d.fill(sim._frame_counter)
        w._density_initialized = True

        sim._run_grid_setup(n, force_sort=True)

        # Density + alpha
        dfsph_solver.compute_density_alpha(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_mass[:n],
            w.sorted_density if hasattr(w, '_density_initialized') else None,
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            sim._cell_start, sim._cell_end,
            w.sorted_density, w.sorted_alpha_dfsph,
            w.sorted_shear_rate, w.sorted_dTdt,
            w.sorted_exposure_heat, w.sorted_exposure_corrode,
        )
        cp.cuda.Device().synchronize()

        dens = w.sorted_density[:n].get()
        alpha = w.sorted_alpha_dfsph[:n].get()
        print(f"  density: min={dens.min():.2f} mean={dens.mean():.2f} max={dens.max():.2f}")
        print(f"  alpha:   min={alpha.min():.6f} mean={alpha.mean():.6f} max={alpha.max():.6f}")

        # Non-pressure forces
        dfsph_solver.compute_non_pressure_forces(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_shear_rate[:n],
            w.sorted_temperature[:n],
            sim._cell_start, sim._cell_end,
            w.sorted_velocity,
        )
        cp.cuda.Device().synchronize()

        sv = w.sorted_velocity[:n].get()
        sv_mag = np.linalg.norm(sv[:,:3], axis=1)
        print(f"  after non-pressure: vel mean={sv_mag.mean():.4f} max={sv_mag.max():.4f}")

        # Divergence solver
        for it in range(profile.dfsph_div_iters):
            dfsph_solver.compute_kappa_v(
                w.sorted_velocity[:n], w.sorted_density[:n],
                w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
                w.sorted_packed_info[:n], w.sorted_position[:n],
                sim._cell_start, sim._cell_end,
                w.sorted_kappa_v,
            )
            cp.cuda.Device().synchronize()
            kv = w.sorted_kappa_v[:n].get()
            print(f"  div iter {it}: kappa_v min={kv.min():.4f} mean={kv.mean():.4f} max={kv.max():.4f}")

            dfsph_solver.correct_velocity_div(
                w.sorted_velocity, w.sorted_density[:n],
                w.sorted_mass[:n], w.sorted_kappa_v[:n],
                w.sorted_packed_info[:n], w.sorted_position[:n],
                sim._cell_start, sim._cell_end,
            )

        # Density solver: the live Jacobi loop on p/rho^2 (same calls as
        # Simulation._sim_step DFSPH path).  sorted_kappa holds p_rho2 and
        # sorted_predicted_position holds a_press.
        p_rho2 = w.sorted_kappa  # K_DFSPH_Finalize already applied the warm start
        a_press = w.sorted_predicted_position
        dfsph_solver.compute_pressure_accel(
            p_rho2[:n], w.sorted_position[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end,
            a_press,
        )
        for it in range(profile.dfsph_dens_iters):
            dfsph_solver.density_solver_update(
                w.sorted_velocity[:n], a_press[:n],
                w.sorted_position[:n], w.sorted_density[:n],
                w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
                w.sorted_packed_info[:n],
                sim._cell_start, sim._cell_end,
                p_rho2,
            )
            dfsph_solver.compute_pressure_accel(
                p_rho2[:n], w.sorted_position[:n],
                w.sorted_density[:n], w.sorted_mass[:n],
                w.sorted_packed_info[:n],
                sim._cell_start, sim._cell_end,
                a_press,
            )
            cp.cuda.Device().synchronize()
            pr = p_rho2[:n].get()
            ap = a_press[:n].get()
            assert np.isfinite(pr).all(), f"p_rho2 not finite at dens iter {it}"
            assert np.isfinite(ap).all(), f"a_press not finite at dens iter {it}"
            print(f"  dens iter {it}: p_rho2 min={pr.min():.6f} mean={pr.mean():.6f} max={pr.max():.6f}")

        dfsph_solver.apply_pressure_velocity(
            w.sorted_velocity[:n], a_press[:n],
            w.sorted_packed_info[:n],
        )
        cp.cuda.Device().synchronize()

        # Finalize — API drift: sorted_kappa required after sorted_sleep_counter;
        # kappa_out required after temperature_out.
        dfsph_solver.finalize(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_dTdt[:n],
            w.sorted_sleep_counter[:n],
            w.sorted_kappa[:n],  # API drift: sorted_kappa inserted here
            sim._sort_perm[:n], sim._cell_start, sim._cell_end,
            w.position, w.velocity, w.packed_info,
            w.sleep_counter, w.temperature,
            w.kappa,  # API drift: kappa_out appended here
        )
        cp.cuda.Device().synchronize()

        out_vel = w.velocity[:n].get()
        out_vel_mag = np.linalg.norm(out_vel[:,:3], axis=1)
        out_pos = w.position[:n].get()
        print(f"  output pos_y: [{out_pos[:,1].min():.4f}, {out_pos[:,1].mean():.4f}, {out_pos[:,1].max():.4f}]")
        print(f"  output vel: mean={out_vel_mag.mean():.4f} max={out_vel_mag.max():.4f}")

        sim._frame_counter += 1


def _run_stability(world, sim, profile_name, num_substeps=10):
    """Internal helper: run N substeps and return True if stable."""
    profile = PROFILES[profile_name]
    print(f"\n{'='*60}")
    print(f"Stability Test: {profile_name} ({num_substeps} substeps)")
    print(f"{'='*60}")

    n = spawn_scene(world)
    sim.set_solver_profile(profile)
    sim.sim_time = 0.0
    sim._last_frame_time = None

    for step in range(num_substeps):
        sim._sim_step(n)
        ok = diagnose(world, f"step {step}")
        if not ok:
            print(f"  FAILED at step {step}")
            return False
    print(f"  PASSED -- {num_substeps} substeps stable")
    return True


def test_stability_pbf(world, sim):
    """Stability test: PBF solver runs 10 substeps without NaN or Inf."""
    assert _run_stability(world, sim, "PBF", 10), \
        "PBF solver became unstable within 10 substeps"


def test_stability_dfsph(world, sim):
    """Stability test: DFSPH solver runs 10 substeps without NaN or Inf."""
    assert _run_stability(world, sim, "DFSPH", 10), \
        "DFSPH solver became unstable within 10 substeps"


# ---------------------------------------------------------------------------
# Density error bounds (bd-r5-review-n28.128)
# ---------------------------------------------------------------------------
# The stability tests above only catch NaN/Inf. These catch finite but wrong
# PBF/DFSPH physics, such as the 50x PBF delta bug (missing m_j factor).
#
# Measured on fix/r5-review (2026-09-25), 39,600 WATER particles after 100
# substeps, interior = FLUID with sorted_normal.w (neighbour count) >= 25:
#   PBF   (dt 1/120): mean |rho/rho0 - 1| = 0.017 (about 29.7K interior)
#   DFSPH (dt 1/300): mean |rho/rho0 - 1| = 0.007 (about 33.0K interior)
# With the PBF m_j factor dropped from the delta, the column blows apart and
# no interior particle is left (test_pbf_density_bound_catches_missing_mass_factor).
#
# Which buffer holds the density: PBF's sorted_density is the density of the
# last constraint iteration's lambda pass, so it lags the final delta by one
# iteration but is a real post-correction density. DFSPH's sorted_density is
# computed at the start of the substep from the positions the previous
# substep's solver produced, so it measures how well that solve held rho0.

PBF_DENSITY_ERR_BOUND = 0.12
DFSPH_DENSITY_ERR_BOUND = 0.03
_INTERIOR_MIN_NEIGHBOURS = 25   # nc < 25 is the surface threshold in the kernels


def _interior_density_error(world, profile_name, substeps=100):
    """Settle a ~40K WATER column and return the interior mean |rho/rho0 - 1|.

    One readback after the loop. Returns (err, n_interior); err is NaN when
    there are no interior particles or a read density is not finite.
    """
    from materials import FLUID, MATERIALS

    profile = PROFILES[profile_name]
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube(
        min_corner=(-0.3, -0.98, -0.3),
        max_corner=(0.3, -0.1, 0.3),
        material_id=WATER,
        spacing=0.02,
    )
    sim = Simulation(world, dt=profile.dt, speed=1.0,
                     fixed_dt=True, max_substeps=1)
    sim.set_solver_profile(profile)
    for _ in range(substeps):
        sim._sim_step(n)

    rho = world.sorted_density[:n].get()
    nc = world.sorted_normal[:n, 3].get()
    pi = world.sorted_packed_info[:n].get()
    interior = (((pi >> 8) & 0x3) == FLUID) & (nc >= _INTERIOR_MIN_NEIGHBOURS)
    n_interior = int(interior.sum())
    if n_interior == 0 or not np.isfinite(rho[interior]).all():
        return float('nan'), n_interior
    rho0 = MATERIALS[WATER].rest_density
    err = float(np.abs(rho[interior] / rho0 - 1.0).mean())
    print(f"  {profile_name}: interior={n_interior}/{n} "
          f"mean|rho/rho0-1|={err:.4f}")
    return err, n_interior


@pytest.mark.gpu
def test_pbf_density_error_bound(world):
    """PBF at dt = 1/120 must hold interior density within 12% of rho0."""
    err, n_interior = _interior_density_error(world, "PBF")
    assert n_interior > 50, 'scene produced no test candidates'
    assert err < PBF_DENSITY_ERR_BOUND, (
        f"PBF interior density error {err:.4f} >= {PBF_DENSITY_ERR_BOUND}")


@pytest.mark.gpu
def test_dfsph_density_error_bound(world):
    """DFSPH must hold interior density within 3% of rho0."""
    err, n_interior = _interior_density_error(world, "DFSPH")
    assert n_interior > 50, 'scene produced no test candidates'
    assert err < DFSPH_DENSITY_ERR_BOUND, (
        f"DFSPH interior density error {err:.4f} >= {DFSPH_DENSITY_ERR_BOUND}")


@pytest.mark.gpu
def test_pbf_density_bound_catches_missing_mass_factor(world, monkeypatch):
    """Mutation check: the PBF bound must fail when the delta drops m_j.

    Compiles pbf_solver.cu with the old delta coeff (no m_j, so corrections
    are 1/m_j = 50x too large) and checks the density test would catch it.
    """
    import os
    import pbf_solver

    good = "float coeff = m_j * (lambda_i + lambda_eff_j + s_corr) / rho0;"
    bad = "float coeff = (lambda_i + lambda_eff_j + s_corr) / rho0;"
    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(pbf_solver.__file__)),
                              "physics", "kernels")
    with open(os.path.join(kernel_dir, "pbf_solver.cu"), encoding="utf-8") as f:
        source = f.read()
    assert source.count(good) == 1, "PBF delta coeff line changed; update this test"
    pbf_solver._ensure_ptx_if_needed()
    mutated = cp.RawModule(
        code=source.replace(good, bad),
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )
    monkeypatch.setattr(pbf_solver, "_module", mutated)
    # The bug blows the column apart, so it can also fail the candidate count.
    err, n_interior = _interior_density_error(world, "PBF")
    caught = n_interior <= 50 or not (err < PBF_DENSITY_ERR_BOUND)
    assert caught, (
        f"PBF density test missed the missing-m_j delta bug: err={err:.4f}, "
        f"interior={n_interior}")


# ---------------------------------------------------------------------------
# Regression tests for review-2026-06-10 beads
# ---------------------------------------------------------------------------

def test_surface_tension_cohesion_dfsph(world, sim):
    """bd-mzc.21: DFSPH surface tension must cohere (pull inward), not repel.

    Uses the real simulation with surface normals injected into sorted_normal.
    Particle at the surface with outward normal in +x direction must gain velocity
    in +x (the norm direction) — the sign test for cohesion vs repulsion.

    The surface normal norm_i points from sparse toward dense (inward toward the
    fluid body). +gamma * norm_i applies force in the norm direction, which
    for a surface particle at +x boundary points toward the fluid interior (+x
    side has less fluid), pulling the particle inward — cohesion.
    """
    import dfsph_solver
    import step2
    from materials import WATER, FLUID

    profile = PROFILES["DFSPH"]
    n = spawn_scene(world)
    sim.set_solver_profile(profile)

    w = world
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    # Find a FLUID water particle to serve as our test subject
    pi = w.sorted_packed_info[:n].get()
    behaviors = (pi >> 8) & 0x3
    fluid_idx = np.where(behaviors == FLUID)[0]
    assert len(fluid_idx) > 50, 'scene produced no test candidates'
    idx = int(fluid_idx[0])

    # Inject a surface normal for particle idx (pointing +x, nc < 25 = surface)
    normal_arr = cp.zeros((n, 4), dtype=cp.float32)
    normal_arr[idx, 0] = 1.0   # norm_i.x = +1 (outward +x)
    normal_arr[idx, 3] = 5.0   # nc < 25 (surface particle)

    def _run_npf(gamma_val):
        gp = step2.build_granular_params(surface_tension_gamma=gamma_val, xsph_epsilon=0.1)
        dfsph_solver.upload_granular_params(gp)
        vel_out = cp.zeros((n, 4), dtype=cp.float32)
        dfsph_solver.compute_non_pressure_forces(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_shear_rate[:n],
            w.sorted_temperature[:n],
            sim._cell_start, sim._cell_end,
            vel_out,
            normal_in=normal_arr,
        )
        cp.cuda.Device().synchronize()
        return vel_out.get()

    vel_no_st = _run_npf(0.0)
    vel_with_st = _run_npf(1.0)

    # Difference is entirely due to surface tension: delta = dt * gamma * norm_i.x
    delta_x = vel_with_st[idx, 0] - vel_no_st[idx, 0]
    print(f"  DFSPH surface tension delta_x[{idx}]: {delta_x:.6f} "
          f"(vel_no_st={vel_no_st[idx,0]:.6f}, vel_with_st={vel_with_st[idx,0]:.6f})")
    assert delta_x != 0.0, "DFSPH surface tension produced zero velocity change"
    assert delta_x > 0.0, (
        f"DFSPH surface tension sign is wrong: delta_x={delta_x:.6f} should be > 0 "
        f"(+gamma*norm_i = cohesion; -gamma*norm_i = repulsion)"
    )


def test_surface_tension_cohesion_pbf(world, sim):
    """bd-mzc.21: PBF surface tension must cohere (pull inward), not repel.

    Mirrors the DFSPH test using pbf_finalize (which applies surface tension).
    A surface particle with outward normal +x should gain velocity in +x after
    the finalize step with gamma > 0.
    """
    import pbf_solver
    import step2
    from materials import WATER, FLUID

    profile = PROFILES["PBF"]
    n = spawn_scene(world)
    sim.set_solver_profile(profile)

    w = world
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    pi = w.sorted_packed_info[:n].get()
    behaviors = (pi >> 8) & 0x3
    fluid_idx = np.where(behaviors == FLUID)[0]
    assert len(fluid_idx) > 50, 'scene produced no test candidates'
    idx = int(fluid_idx[0])

    # Inject surface normal for particle idx
    normal_arr = cp.zeros((n, 4), dtype=cp.float32)
    normal_arr[idx, 0] = 1.0   # norm_i.x = +1
    normal_arr[idx, 3] = 5.0   # nc < 25 (surface)

    sort_perm = sim._sort_perm[:n]
    orig_idx = int(sort_perm[idx].get())

    def _run_pbf_finalize(gamma_val):
        gp = step2.build_granular_params(surface_tension_gamma=gamma_val)
        pbf_solver.upload_granular_params(gp)
        pos_out = cp.zeros_like(w.position)
        vel_out = cp.zeros_like(w.velocity)
        pi_out = cp.zeros_like(w.packed_info)
        sleep_out = cp.zeros_like(w.sleep_counter)
        temp_out = cp.zeros_like(w.temperature)
        pbf_solver.pbf_finalize(
            w.sorted_position[:n], w.sorted_position[:n],
            w.sorted_velocity[:n], w.sorted_density[:n],
            w.sorted_mass[:n], w.sorted_packed_info[:n],
            w.sorted_temperature[:n], w.sorted_health[:n],
            w.sorted_dTdt[:n], w.sorted_sleep_counter[:n],
            sort_perm, sim._cell_start, sim._cell_end,
            pos_out, vel_out, pi_out,
            sleep_out, temp_out,
            normal_in=normal_arr,
        )
        cp.cuda.Device().synchronize()
        return float(vel_out[orig_idx, 0].get())

    # K_PBF_Finalize: vel_new = (pred_pos - orig_pos)/dt + surface_tension.
    # With pred==orig, vel_new = 0 + dt*gamma*norm_i.x. Run twice to isolate ST.
    vel_no_st = _run_pbf_finalize(0.0)
    vel_with_st = _run_pbf_finalize(1.0)

    delta_x = vel_with_st - vel_no_st
    print(f"  PBF surface tension delta_x[sorted={idx}, orig={orig_idx}]: {delta_x:.6f} "
          f"(no_st={vel_no_st:.6f}, with_st={vel_with_st:.6f})")
    assert delta_x != 0.0, "PBF surface tension produced zero velocity change"
    assert delta_x > 0.0, (
        f"PBF surface tension sign is wrong: delta_x={delta_x:.6f} should be > 0 "
        f"(+gamma*norm_i = cohesion)"
    )


def test_dfsph_density_convergence(world, sim):
    """bd-r5-review-n28.26: the DFSPH Jacobi density solve must feel its own pressure.

    K_DFSPH_DensitySolverUpdate predicts rho*/rho0 = (rho_i/rho0) * (1 + dt*drho),
    where drho = SUM_j V_j (vt_i - vt_j) . gradW is already a relative rate (1/s).
    An extra /rho0 on the drho term (bd-mzc.22) made the A*p feedback rho0 times
    too small (2500x for WATER): the solver ignored both the velocity divergence
    and the pressure it had applied, and p_rho2 grew linearly with iterations.

    Scene: one isolated 9x9x9 WATER block (spacing 0.02), alpha uncapped
    (dfsph_alpha_limit = 1e9) and velocity limit raised so the vt clamps stay off.

    Sub-test 1 (zero velocity): p_rho2 stays small at rest.
    Sub-test 2 (diagonal): a unit p_rho2 on the centre particle alone must lower
      its own predicted residual by about dt^2 / alpha_i, because alpha is the
      inverse diagonal of this residual. The old formula gave about 1/rho0 of that.
    Sub-test 3 (compression, omega = 1): v = -k (x - c), a uniform divergence of
      -3k. One Jacobi iteration must cut the residual by more than 50%.
    Sub-test 4 (convergence, omega = 0.7): after 8 iterations the residual is
      under 2% and the p_rho2 increments have shrunk. omega = 1 with the exact
      diagonal overshoots the short-wavelength modes and the residual grows back
      after about 4 iterations; picking omega and the alpha cap is bead
      bd-r5-review-n28.37.

    The residual is read from the Jacobi increment itself:
    p_new - p_old = omega * residual * alpha / dt^2 wherever p_new > 0. Where the
    clamp holds p_new at 0 the constraint rho <= rho0 is inactive and the
    complementarity residual is 0.
    """
    import dataclasses
    import dfsph_solver
    from materials import MATERIALS

    w = world
    rho0 = float(MATERIALS[WATER].rest_density)

    def _setup(omega):
        w.packed_info[:] = 0
        w._high_water = 0
        half = 0.09
        n = w.spawn_cube(
            min_corner=(-half, 0.3 - half, -half),
            max_corner=(half, 0.3 + half, half),
            material_id=WATER,
            spacing=0.02,
        )
        profile = dataclasses.replace(
            PROFILES["DFSPH"], dfsph_omega=omega, dfsph_alpha_limit=1.0e9,
            velocity_limit_factor=50.0,
        )
        sim.set_solver_profile(profile)
        w._density_initialized = True
        sim._run_grid_setup(n, force_sort=True)
        # Two passes: the second uses the first pass's density for V_j = m_j/rho_j,
        # as the real step does with sorted_density_prev.
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
        cp.cuda.Device().synchronize()
        return n, profile

    def _update(n, vel, accel, p):
        dfsph_solver.density_solver_update(
            vel, accel,
            w.sorted_position[:n], w.sorted_density[:n],
            w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
            w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end,
            p,
        )

    def _accel(n, p, accel):
        dfsph_solver.compute_pressure_accel(
            p, w.sorted_position[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end,
            accel,
        )

    def _compression_run(omega, n_updates=9):
        n, profile = _setup(omega)
        dt = float(profile.dt)
        pos = w.sorted_position[:n, :3].get()
        alpha = w.sorted_alpha_dfsph[:n].get()
        active = alpha > 0.0
        assert active.sum() > 500, f"alpha is zero for most particles ({active.sum()}/{n})"
        centre = pos.mean(axis=0)
        k = 0.1 / (3.0 * dt)  # one-substep compression dt * 3k = 10%
        vel = cp.zeros((n, 4), dtype=cp.float32)
        vel[:, :3] = cp.asarray((-k * (pos - centre)).astype(np.float32))
        p = cp.zeros(n, dtype=cp.float32)
        accel = cp.zeros((n, 4), dtype=cp.float32)
        history = [p.copy()]
        for _ in range(n_updates):
            _update(n, vel, accel, p)
            _accel(n, p, accel)
            history.append(p.copy())
        cp.cuda.Device().synchronize()
        hist = np.stack([hp.get() for hp in history])
        assert np.isfinite(hist).all(), "DFSPH density solver produced NaN/Inf in p_rho2"
        residuals, increments = [], []
        a = alpha[active]
        for kk in range(n_updates):
            p_new = hist[kk + 1][active]
            dp = p_new - hist[kk][active]
            r = np.where(p_new > 0.0, dp * dt * dt / (omega * a), 0.0)
            residuals.append(float(np.mean(np.abs(r))))
            increments.append(float(np.mean(np.abs(dp))))
        print(f"  DFSPH compression omega={omega}: n={n} active={int(active.sum())}")
        for kk, (r, d) in enumerate(zip(residuals, increments)):
            print(f"    before update {kk}: mean residual={r:.5f}  mean|dp|={d:.4e}")
        return residuals, increments

    # --- Sub-test 1: zero velocity: p_rho2 must stay small at rest ---
    n, profile = _setup(1.0)
    dt = float(profile.dt)
    zero4 = cp.zeros((n, 4), dtype=cp.float32)
    p_zero = cp.zeros(n, dtype=cp.float32)
    for _ in range(20):
        _update(n, zero4, zero4, p_zero)
    cp.cuda.Device().synchronize()
    p_zero_h = p_zero.get()
    max_p_zero = float(np.abs(p_zero_h).max())
    print(f"  DFSPH density solver (zero vel): max_p_rho2={max_p_zero:.4e}")
    assert np.isfinite(p_zero_h).all(), "DFSPH density solver (zero vel) produced NaN/Inf"
    assert max_p_zero < 0.5 * rho0, (
        f"DFSPH density solver (zero vel) has spurious pressure: max_p_rho2={max_p_zero:.4e}"
    )

    # --- Sub-test 2: diagonal feedback of a single particle's pressure ---
    pos = w.sorted_position[:n, :3].get()
    alpha = w.sorted_alpha_dfsph[:n].get()
    rho = w.sorted_density[:n].get()
    i = int(np.argmin(np.linalg.norm(pos - pos.mean(axis=0), axis=1)))
    assert alpha[i] > 0.0, "centre particle has alpha = 0"
    p_single = cp.zeros(n, dtype=cp.float32)
    p_single[i] = 1.0
    accel = cp.zeros((n, 4), dtype=cp.float32)
    _accel(n, p_single, accel)
    # Probe with a large offset so the >= 0 clamp never fires; omega = 1 here.
    offset = 1.0e6
    probe = cp.full(n, offset, dtype=cp.float32)
    _update(n, zero4, accel, probe)
    cp.cuda.Device().synchronize()
    r_with_p = (float(probe[i].get()) - offset) * dt * dt / float(alpha[i])
    r_rest = float(rho[i]) / rho0 - 1.0
    ratio = (r_with_p - r_rest) / (dt * dt / float(alpha[i]))
    print(f"  DFSPH diagonal: rho_i/rho0={rho[i] / rho0:.4f} residual change per unit "
          f"p_rho2 = {ratio:.4f} x (dt^2/alpha_i)  (expect about -1; old /rho0 bug gave about -0.02)")
    assert -1.3 < ratio < -0.7, (
        f"a unit p_rho2 on particle {i} changed its predicted residual by {ratio:.4f} "
        f"x dt^2/alpha (expect about -1). Check density_adv = (rho_i/rho0) * (1 + dt*drho)."
    )

    # --- Sub-test 3: one Jacobi iteration at omega = 1 halves the residual ---
    res1, _ = _compression_run(1.0)
    assert res1[0] > 0.02, f"compression setup too weak: initial residual {res1[0]:.4f}"
    assert res1[1] < 0.5 * res1[0], (
        f"one Jacobi iteration only cut the residual from {res1[0]:.4f} to {res1[1]:.4f} "
        f"(need > 50%). Check density_adv = (rho_i/rho0) * (1 + dt*drho)."
    )

    # --- Sub-test 4: 8 iterations at omega = 0.7 converge ---
    res7, inc7 = _compression_run(0.7)
    assert res7[8] < 0.02, (
        f"residual after 8 iterations is {res7[8]:.4f} (need < 0.02). "
        f"The Jacobi density solve is not converging."
    )
    # Increments are omega*residual*alpha/dt^2, so they shrink with the residual.
    # With the old /rho0 bug they stay constant (p_rho2 grows linearly).
    assert inc7[7] < 0.2 * inc7[0] and inc7[8] < 0.2 * inc7[0], (
        f"p_rho2 still growing linearly: increments {inc7[0]:.4e} (update 0), "
        f"{inc7[7]:.4e} (update 7), {inc7[8]:.4e} (update 8)."
    )


def test_pbf_wall_density(world, sim):
    """bd-mzc.25: PBF STATIC boundary contribution must not be inflated 2x.

    The bug: boundary_scale = (behavior_j == STATIC) ? 2.0 : 1.0 doubles the
    density contribution from every STATIC neighbor. The fix removes this
    multiplier, using standard Akinci psi_b = m_j.

    Test approach: spawn a WATER particle at a specific position and a STONE
    (STATIC) particle at a known distance. Compute lambda twice — once with the
    actual kernel (which should use no multiplier after the fix) and verify that
    the density contribution from the STATIC neighbor is consistent with m_j * W(r)
    and NOT 2 * m_j * W(r).

    Implementation: we verify the density of a fluid particle near a static wall
    is the same whether we treat the static neighbor as STATIC or as FLUID
    (by temporarily patching packed_info). If boundary_scale=2 were active, the
    STATIC version would give 2x the FLUID version for that neighbor's contribution.
    """
    import pbf_solver
    from materials import WATER, FLUID, STONE, STATIC

    profile = PROFILES["PBF"]
    world.packed_info[:] = 0
    world._high_water = 0

    # Place 2 particles: one FLUID water (particle 0) and one STONE/STATIC (particle 1)
    # at a distance of h*0.5 apart so they are neighbors.
    h = 0.04
    spacing = h * 0.3   # close neighbor

    n_water = world.spawn_cube(
        min_corner=(-0.01, -0.01, -0.01),
        max_corner=(0.01, 0.01, 0.01),
        material_id=WATER,
        spacing=0.02,
    )
    n_stone = world.spawn_cube(
        min_corner=(spacing - 0.01, -0.01, -0.01),
        max_corner=(spacing + 0.01, 0.01, 0.01),
        material_id=STONE,
        spacing=0.02,
    )
    n = n_water + n_stone

    sim.set_solver_profile(profile)

    w = world
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    # Run compute_lambda on the FLUID+STATIC scene (after fix: no boundary_scale)
    pbf_solver.pbf_compute_lambda(
        w.sorted_position[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end,
        w.sorted_density, w.sorted_lambda_pbf,
    )
    cp.cuda.Device().synchronize()

    pi = w.sorted_packed_info[:n].get()
    rho_with_static = w.sorted_density[:n].get()
    behaviors = (pi >> 8) & 0x3
    fluid_mask = behaviors == FLUID
    static_mask = behaviors == STATIC

    assert fluid_mask.sum() > 0 and static_mask.sum() > 0, \
        'scene produced no test candidates'

    # Now run compute_lambda again but with all particles marked as FLUID
    # (i.e., remove the STATIC flag). Without boundary_scale, this should give
    # the same density for fluid particles as the STATIC version.
    # With boundary_scale=2, the STATIC version would give 2x higher density
    # for the contribution from the static neighbors.
    _MAKE_PACKED = lambda mat, beh: (int(mat) & 0xFF) | ((int(beh) & 0x3) << 8)
    pi_all_fluid = pi.copy()
    pi_all_fluid[static_mask] = _MAKE_PACKED(WATER, FLUID)
    pi_all_fluid_d = cp.array(pi_all_fluid, dtype=cp.uint32)

    # Replace sorted packed_info temporarily
    saved_pi = w.sorted_packed_info[:n].copy()
    w.sorted_packed_info[:n] = pi_all_fluid_d

    pbf_solver.pbf_compute_lambda(
        w.sorted_position[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end,
        w.sorted_density, w.sorted_lambda_pbf,
    )
    cp.cuda.Device().synchronize()
    rho_all_fluid = w.sorted_density[:n].get()

    # Restore
    w.sorted_packed_info[:n] = saved_pi

    rho_static_fluid = rho_with_static[fluid_mask]
    rho_fluid_fluid = rho_all_fluid[fluid_mask]

    # The ratio of static-neighbor density to fluid-neighbor density for the same
    # spatial configuration. With fix (no boundary_scale): ratio should be ~1.0.
    # With bug (boundary_scale=2): ratio should be ~2.0 for STATIC-dominated particles.
    # For particles with NO static neighbors, both runs give identical density.
    # For particles with static neighbors: fix => ratio ≈ 1.0; bug => ratio ≈ 2.0.
    nonzero = rho_fluid_fluid > 100.0
    assert nonzero.sum() > 0, 'scene produced no test candidates'

    ratio = rho_static_fluid[nonzero] / rho_fluid_fluid[nonzero]
    max_ratio = ratio.max()
    mean_ratio = ratio.mean()
    print(f"  PBF boundary density ratio (static/fluid): "
          f"max={max_ratio:.4f}, mean={mean_ratio:.4f} "
          f"(expected ~1.0 with fix, ~2.0 with bug)")
    # After the fix (no boundary_scale), STATIC and FLUID neighbors contribute equally.
    # Allow up to 1.3 (some particles may have partial static coverage), but not 2.0.
    assert max_ratio < 1.3, (
        f"PBF STATIC boundary density inflated: max ratio={max_ratio:.4f} > 1.3. "
        f"Suggests boundary_scale=2.0 is still active. Remove it and use psi_b = m_j."
    )


def test_dfsph_density_solver_velocity_clamp(world, sim):
    """bd-mzc.33: DFSPH density solver must not produce runaway velocities.

    Inject a large pressure acceleration (50x the velocity limit) into the
    density solver update kernel and run 12 iterations. Velocity magnitude must
    stay bounded (below 10x the velocity limit) — the v_total clamp prevents
    the Jacobi iterations from amplifying an initial overshoot into a runaway.
    """
    import dfsph_solver
    import step1

    profile = PROFILES["DFSPH"]
    n = spawn_scene(world)
    sim.set_solver_profile(profile)

    w = world
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    # Compute density + alpha first
    dfsph_solver.compute_density_alpha(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_mass[:n], None,
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        w.sorted_density, w.sorted_alpha_dfsph,
        w.sorted_shear_rate, w.sorted_dTdt,
        w.sorted_exposure_heat, w.sorted_exposure_corrode,
    )
    cp.cuda.Device().synchronize()

    # Velocity limit from sim params
    vel_limit = float(sim._sim_params[0]["velocity_limit"])
    if vel_limit <= 0:
        vel_limit = 10.0

    # Inject large velocity into all FLUID particles (100x vel_limit) to simulate
    # an overshoot scenario. Without the v_total clamp, density_solver_update would
    # compute an extreme drho from this velocity, driving p_rho2 to diverge.
    vel_injected = w.sorted_velocity[:n].copy()
    vel_injected[:, 0] = 100.0 * vel_limit   # huge +x velocity on all particles

    accel_press = cp.zeros((n, 4), dtype=cp.float32)
    p_rho2 = cp.zeros(n, dtype=cp.float32)

    # Run 12 Jacobi density solver iterations (normal inner loop count).
    # With the v_total clamp: vt_i is clamped to VELOCITY_LIMIT before drho,
    # so p_rho2 stays bounded. Without the clamp, p_rho2 would diverge.
    for _ in range(12):
        dfsph_solver.density_solver_update(
            vel_injected, accel_press,
            w.sorted_position[:n], w.sorted_density[:n],
            w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
            w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end,
            p_rho2,
        )
    cp.cuda.Device().synchronize()

    p = p_rho2.get()
    max_p = np.abs(p).max()
    # p_rho2 = pressure / rho0^2.  A physically reasonable value is ~rho0 * c_s^2 / rho0^2
    # ~ c_s^2 / rho0.  For water: c_s~1500 m/s, rho0=1000, so ~2250.
    # We just check it's not 1e10+ (diverged).  The clamp keeps drho ~ vel_limit / h,
    # so p_rho2 saturates within a few iterations.
    threshold_p = 1e8
    print(f"  DFSPH density solver p_rho2: max={max_p:.4e}, threshold={threshold_p:.4e}")
    assert max_p < threshold_p, (
        f"DFSPH density solver p_rho2 diverged: max={max_p:.4e} > {threshold_p:.4e}. "
        f"The v_total clamp in K_DFSPH_DensitySolverUpdate is required."
    )


def test_dfsph_rigid_viscous_force_scaling(world, sim):
    """bd-unl.11 / bd-mzc.23: Rigid viscous reaction force must include /rho_i.

    The bug (bd-unl.1): the two-way reaction force on the rigid body was computed as
        F_on_body = -F_visc * m_i                   # BUGGY: missing /rho_i
    instead of the correct:
        F_on_body = -(F_visc / rho_i) * m_i         # FIXED

    For WATER (rho_i ≈ rho0 = 2500), the buggy code overscales the reaction force
    by a factor of ~2500x.

    This test sets up a deterministic 2-particle configuration:
      - Particle 0: WATER (FLUID, mat_id=5), velocity = (1, 0, 0) m/s
      - Particle 1: RIGID boundary (STATIC, mat_id=18, body_id=0), velocity = (0, 0, 0)
        placed at distance rlen = h * 0.5 = 0.02 m from particle 0

    Expected force on the rigid body (1 neighbor pair, dv = 1 m/s):
      visc_factor = mu_b * m_j * (h - rlen) / rho_i
      mu_b        = base_viscosity * viscosity_lap_coeff = 1.0 * 45/(pi*h^6) ≈ 3.50e9
      visc_factor ≈ 3.50e9 * 0.02 * 0.02 / 2500 ≈ 560
      F_visc      ≈ 560 * 1.0 ≈ 560 N/kg (force density on fluid particle)

      CORRECT: F_on_body = F_visc / rho_i * m_i ≈ 560 / 2500 * 0.02 ≈ 4.5e-3 N
      BUGGY:   F_on_body = F_visc * m_i          ≈ 560       * 0.02 ≈ 11.2 N  (2500x too large)

    Assertion threshold: force magnitude must be < 0.5 N.
      - CORRECT code: ~4.5e-3 N  → PASS (4.5e-3 < 0.5)
      - BUGGY code:   ~11.2 N    → FAIL (11.2 > 0.5)

    The threshold gap to the buggy value is >20x, so small numerical variation
    from real density (not exactly rho0) or multiple neighbors cannot produce
    false failures.
    """
    import dfsph_solver
    from materials import WATER, FLUID, MAT_RIGID, STATIC, MATERIALS
    from rigid_bodies import RigidBodyManager, SDF_BOX

    profile = PROFILES["DFSPH"]
    world.packed_info[:] = 0
    world._high_water = 0

    # --- Scene setup ---
    # h = 0.04 (smoothing length, same as simulation default)
    # Place a 2x2x2 cube of WATER particles at the origin (8 particles).
    # Place one RIGID boundary particle at (h*0.5, 0, 0) = (0.02, 0, 0) from the
    # cluster center, within the smoothing length so the kernel finds it as a neighbor.
    h = 0.04
    spacing = 0.02   # particle spacing = h/2

    # Spawn a small water cluster
    n_water = world.spawn_cube(
        min_corner=(-0.02, -0.02, -0.02),
        max_corner=(0.02, 0.02, 0.02),
        material_id=WATER,
        spacing=spacing,
    )

    # Manually insert one rigid boundary particle at position (h*0.5, 0, 0) from center.
    # packed_info encoding:
    #   bits [7:0]   = MAT_RIGID = 18
    #   bits [9:8]   = STATIC = 3  => (3 << 8) = 768
    #   bits [20:13] = body_id = 0 => (0 << 13) = 0
    rigid_pi = np.uint32(MAT_RIGID | (STATIC << 8) | (0 << 13))
    rho0 = float(MATERIALS[WATER].rest_density)  # 2500.0
    m = float(0.02)  # particle_mass (matches simulation default)

    rigid_pos = np.array([[h * 0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
    rigid_vel = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rigid_mass = np.array([m], dtype=np.float32)

    idx_rigid = n_water  # append after fluid particles
    world.position[idx_rigid]   = cp.asarray(rigid_pos[0])
    world.velocity[idx_rigid]   = cp.asarray(rigid_vel[0])
    world.mass[idx_rigid]       = cp.float32(m)
    world.packed_info[idx_rigid] = cp.uint32(rigid_pi)
    world._high_water = n_water + 1
    n = n_water + 1

    # Give fluid particles a velocity of +1 m/s in x so there is a relative velocity
    # between fluid and the stationary rigid body.
    world.velocity[:n_water, 0] = 1.0

    # --- Grid and solver setup ---
    sim.set_solver_profile(profile)
    w = world

    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    # Compute density so rho_i is available for the force kernel.
    dfsph_solver.compute_density_alpha(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_mass[:n], None,
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        w.sorted_density, w.sorted_alpha_dfsph,
        w.sorted_shear_rate, w.sorted_dTdt,
        w.sorted_exposure_heat, w.sorted_exposure_corrode,
    )
    cp.cuda.Device().synchronize()

    # --- Set up a RigidBodyManager with one body at the rigid particle position ---
    # The kernel reads d_rigid_bodies[body_id].{position, lin_vel, ang_vel}.
    # Body 0 is at rest at (h*0.5, 0, 0); its inv_mass does not matter for this test.
    rbm = RigidBodyManager(max_bodies=8)
    rbm.add_rigid_body(
        position=(h * 0.5, 0.0, 0.0),
        half_extents=(0.02, 0.02, 0.02),
        mass=1.0,
        sdf_type=SDF_BOX,
    )

    # Upload body count to dfsph_solver constant memory and body data to GPU global.
    # rbm.upload() uses cupy.asarray(structured_dtype) which fails on some CuPy builds;
    # use raw memcpy instead (same path as the existing constant-memory uploads).
    module = dfsph_solver.get_module()

    body_bytes = rbm._bodies_cpu[:rbm._num_bodies].view(np.uint8)
    d_ptr_rb = int(rbm.d_rigid_bodies.data.ptr)
    cp.cuda.runtime.memcpy(d_ptr_rb, body_bytes.ctypes.data, body_bytes.nbytes, 1)

    h_count = np.array([rbm._num_bodies], dtype=np.int32)
    try:
        d_count_ptr = module.get_global("c_num_rigid_bodies")
        cp.cuda.runtime.memcpy(int(d_count_ptr), h_count.ctypes.data, 4, 1)
    except Exception:
        pass  # Symbol may be absent if compiled without rigid bodies

    # Zero force/torque accumulators.
    rbm.zero_accumulators()

    # --- Run K_DFSPH_NonPressureForces ---
    vel_out = cp.zeros((n, 4), dtype=cp.float32)
    dfsph_solver.compute_non_pressure_forces(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_density[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], w.sorted_shear_rate[:n],
        w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        vel_out,
        d_rigid_bodies=rbm.d_rigid_bodies,
        d_rigid_forces=rbm.rigid_forces,
        d_rigid_torques=rbm.rigid_torques,
    )
    cp.cuda.Device().synchronize()

    # --- Read accumulated force on body 0 ---
    forces_cpu = rbm.rigid_forces.get()  # shape (8, 4)
    fx, fy, fz = forces_cpu[0, 0], forces_cpu[0, 1], forces_cpu[0, 2]
    force_mag = float(np.sqrt(fx * fx + fy * fy + fz * fz))

    # --- Thresholds ---
    # CORRECT formula: F_on_body ≈ F_visc / rho_i * m_i per contributing neighbor pair.
    # With N_neighbors fluid particles near the rigid particle at rlen ≈ h*0.5:
    # Single-pair magnitude ≈ 4.5e-3 N; with up to ~8 fluid neighbours ≈ 3.6e-2 N.
    # BUGGY formula: same values × rho_i ≈ 2500 → ≈ 11–90 N range.
    #
    # We set the threshold at 0.5 N:
    #   - Correct code (few-neighbour cluster): expected < 0.05 N  → PASS
    #   - Buggy code (missing /rho_i, 2500x):   expected > 10 N   → FAIL
    # The 20x margin from 0.5 to 10 N guards against any numerical variation.
    threshold = 0.5  # N

    print(
        f"  Rigid viscous reaction force: ({fx:.4e}, {fy:.4e}, {fz:.4e}) N, "
        f"magnitude={force_mag:.4e} N  (threshold={threshold} N)  "
        f"[buggy code would give ~{force_mag * rho0:.1f} N]"
    )

    assert force_mag > 0.0, (
        "Rigid body received zero reaction force — no fluid-rigid neighbor pairs found. "
        "Check that the rigid boundary particle is within h of the fluid cluster."
    )
    assert force_mag < threshold, (
        f"Rigid viscous reaction force too large: {force_mag:.4e} N > {threshold} N. "
        f"The formula must be F_on_body = -(F_visc / rho_i) * m_i. "
        f"Without /rho_i the force is ~{rho0:.0f}x too large (rho_i overscale, bd-unl.1)."
    )


def main():
    world = World(max_particles=MAX_PARTICLES)
    n = spawn_scene(world)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20)

    # Quick stability checks (20 substeps each)
    _run_stability(world, sim, "PBF", 20)
    _run_stability(world, sim, "DFSPH", 20)


if __name__ == "__main__":
    main()


def test_wcsph_density_in_carries_previous_density(world, sim):
    """Regression (round-5 bead 0-density-w-carry): WCSPH step1 reads neighbour
    density rho_j from sorted_position.w, which the reorder gathers from the
    unsorted position.w that K_Integrate writes (bd-r5-review-n28.163).
    K_Integrate used to write .w = 1.0 for every awake particle, so from the
    second substep every awake neighbour had rho_j = 1 instead of ~1000-2500,
    inflating heat, vorticity, normals and strain rate by that factor."""
    n = spawn_scene(world)
    sim.set_solver_profile(PROFILES["WCSPH"])
    sim.sim_time = 0.0
    sim._last_frame_time = None
    for _ in range(4):
        sim._sim_step(n)
    # The next substep's step1 reads this .w, reordered, as rho_j.
    prev = cp.asnumpy(world.position[:n, 3])
    info = cp.asnumpy(world.packed_info[:n])
    live = (info & 0xFF) != 0  # not MAT_DEAD
    frac_unit = float(np.mean(np.abs(prev[live] - 1.0) < 1e-3))
    assert frac_unit < 0.01, f"{frac_unit:.1%} of position.w entries are 1.0"
    assert np.median(prev[live]) > 100.0, f"median position.w {np.median(prev[live]):.2f}"
    # The unsorted .w is this substep's step1 density for each particle.
    dens = cp.asnumpy(world.sorted_density[:n])
    perm = cp.asnumpy(sim._sort_perm[:n]).astype(np.int64)
    live_s = live[perm]
    np.testing.assert_allclose(prev[perm][live_s], dens[live_s], rtol=1e-6)


def test_dfsph_module_has_only_live_kernels():
    """bd-r5-review-n28.11: dfsph_solver.cu holds only the 8 live kernels."""
    import os
    import re
    import dfsph_solver
    live = {
        "K_DFSPH_ComputeDensityAlpha", "K_DFSPH_NonPressureForces",
        "K_DFSPH_ComputeKappaV", "K_DFSPH_CorrectVelocityDiv",
        "K_DFSPH_ComputePressureAccel", "K_DFSPH_DensitySolverUpdate",
        "K_DFSPH_ApplyPressureVelocity", "K_DFSPH_Finalize",
    }
    cu_path = os.path.join(os.path.dirname(dfsph_solver.__file__),
                           "physics", "kernels", "dfsph_solver.cu")
    with open(cu_path, encoding="utf-8") as f:
        src = f.read()
    defined = set(re.findall(r"^void\s+(K_DFSPH_\w+)\s*\(", src, re.M))
    assert defined == live
    assert "LEGACY" not in src
    module = dfsph_solver.get_module()
    for name in live:
        module.get_function(name)
    assert dfsph_solver.DFSPH_PARAMS_DTYPE.itemsize == 32
    for gone in ("predict_position", "compute_density_adv", "compute_kappa",
                 "compute_kappa_from_velocity", "correct_velocity_dens"):
        assert not hasattr(dfsph_solver, gone)


# ---------------------------------------------------------------------------
# bd-r5-review-n28.30: K_DFSPH_Finalize sleep gate and FLUID wall friction
# ---------------------------------------------------------------------------

_SLEEP_BIT = 0x400  # SET_SLEEPING in common.cuh


def _dfsph_zero_g(world, sim):
    """DFSPH profile, zero gravity, no spawn damping ramp."""
    sim.set_solver_profile(PROFILES["DFSPH"])
    sim.set_gravity(0.0)
    sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None


def test_dfsph_sleep_is_granular_only(world, sim):
    """A still DFSPH water block never sleeps; a still sand block does.

    Every particle starts with its sleep counter past SLEEP_THRESHOLD, and half
    the water starts with a stale SLEEPING bit. Before the fix, K_DFSPH_Finalize
    counted sleep for every behaviour, so the water froze.
    """
    from materials import GRANULAR
    world.packed_info[:] = 0
    world._high_water = 0
    # The first DFSPH substep builds alpha from the previous substep's
    # sorted_density. Left over from an earlier test it is the wrong scene,
    # and SAND at its own rho0 (bd-r5-review-n28.53) then gets a 0.15 m/s kick.
    world.sorted_density[:] = 0.0
    n_water = world.spawn_cube((-0.6, -0.2, -0.2), (-0.2, 0.2, 0.2), WATER, spacing=0.02)
    # At spacing 0.02 the lattice interior sits at 1.01 rho0, so the density
    # solver pushes the block apart at up to 0.15 m/s and it is not still.
    # Spacing 0.0202 gives 0.98 rho0: no pressure, the block stays at rest
    # (bd-r5-review-n28.100).
    n_sand = world.spawn_cube((0.2, -0.2, -0.2), (0.6, 0.2, 0.2), SAND, spacing=0.0202)
    n = n_water + n_sand
    _dfsph_zero_g(world, sim)
    world.sleep_counter[:n] = 200
    world.velocity[:n] = 0.0
    world.packed_info[0:n_water:2] |= cp.uint32(_SLEEP_BIT)
    try:
        for _ in range(20):
            sim._sim_step(n)
        pi = cp.asnumpy(world.packed_info[:n])
        sc = cp.asnumpy(world.sleep_counter[:n])
    finally:
        sim.set_gravity(-4.0)
    mat = pi & 0xFF
    beh = (pi >> 8) & 0x3
    sleeping = (pi & _SLEEP_BIT) != 0
    water = mat == WATER
    sand = mat == SAND
    assert water.sum() == n_water and sand.sum() == n_sand
    assert not sleeping[water].any(), f"{sleeping[water].sum()} WATER particles asleep"
    assert (sc[water] == 0).all(), f"WATER sleep_counter max {sc[water].max()}"
    assert (beh[sand] == GRANULAR).all()
    assert sleeping[sand].any(), "still SAND block should have sleepers"


def test_dfsph_fluid_has_no_wall_friction(world, sim):
    """A lone WATER particle sliding into the floor keeps its tangential speed;
    a lone SAND particle loses mu_wall * |v_n| of it (wall_friction = 0.5)."""
    world.packed_info[:] = 0
    world._high_water = 0
    y = -1.0 + 0.001  # just above the floor (world_half_size = 1)
    vel = (1.0, -2.0, 0.0)
    # spawn_cube puts one particle at min_corner + spacing/2 per axis.
    n_w = world.spawn_cube((-0.51, y - 0.01, -0.01), (-0.49, y + 0.01, 0.01),
                           WATER, spacing=0.02, velocity=vel)
    n_s = world.spawn_cube((0.49, y - 0.01, -0.01), (0.51, y + 0.01, 0.01),
                           SAND, spacing=0.02, velocity=vel)
    assert n_w == 1 and n_s == 1
    n = 2
    _dfsph_zero_g(world, sim)
    try:
        sim._sim_step(n)
        pi = cp.asnumpy(world.packed_info[:n])
        v = cp.asnumpy(world.velocity[:n])
        p = cp.asnumpy(world.position[:n])
    finally:
        sim.set_gravity(-4.0)
    mat = pi & 0xFF
    w = int(np.nonzero(mat == WATER)[0][0])
    s = int(np.nonzero(mat == SAND)[0][0])
    # Both hit the floor this substep (moved at least |v_y| * dt > 1 mm).
    assert p[w, 1] < y and p[s, 1] < y
    assert v[w, 1] >= 0.0 and v[s, 1] >= 0.0, "normal velocity should reflect"
    assert abs(v[w, 0] - 1.0) < 0.02, f"WATER tangential {v[w, 0]:.4f}, want 1.0"
    assert v[s, 0] < 0.5, f"SAND tangential {v[s, 0]:.4f}, want < 0.5 (Coulomb)"


# ---------------------------------------------------------------------------
# bd-r5-review-n28.34: a brush kill inside DFSPH or PBF water stays finite
# ---------------------------------------------------------------------------

def _kill_center_and_run(world, sim, profile_name, old_encoding=False, substeps=30):
    """Spawn a water block, kill a sphere at its centre, run substeps.

    With old_encoding the killed slots are rewritten to packed_info 0 (DEAD
    with FLUID behavior and mass 0.02), the state the brush left before the
    fix, so only the rest density floor in the kernels protects the solver.
    Returns host copies of position, velocity, p_rho2 (sorted_kappa),
    packed_info and mass.
    """
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2), WATER, spacing=0.02)
    sim.set_solver_profile(PROFILES[profile_name])
    sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None
    sim._sim_step(n)  # settle the grid once before the kill
    before = world.packed_info[:n].copy()
    world.kill_in_sphere((0.0, 0.0, 0.0), 0.06)
    if old_encoding:
        killed = (world.packed_info[:n] & cp.uint32(0xFF)) == 0
        killed &= (before & cp.uint32(0xFF)) != 0
        world.packed_info[:n][killed] = cp.uint32(0)
        world.mass[:n][killed] = cp.float32(0.02)
    killed_idx = cp.nonzero(((before & cp.uint32(0xFF)) != 0)
                            & ((world.packed_info[:n] & cp.uint32(0xFF)) == 0))[0]
    for _ in range(substeps):
        sim._sim_step(n)
    return (cp.asnumpy(world.position[:n]), cp.asnumpy(world.velocity[:n]),
            cp.asnumpy(world.sorted_kappa[:n]), cp.asnumpy(world.packed_info[:n]),
            cp.asnumpy(world.mass[:n]), cp.asnumpy(killed_idx))


@pytest.mark.parametrize("profile_name", ["DFSPH", "PBF"])
def test_kill_in_sphere_stays_finite(world, sim, profile_name):
    """Killed slots are STATIC DEAD with mass 0, and the solver stays finite."""
    from materials import STATIC
    pos, vel, p_rho2, pi, mass, killed = _kill_center_and_run(world, sim, profile_name)
    assert killed.size > 50, f"only {killed.size} particles killed"
    assert (pi[killed] == (STATIC << 8)).all(), "killed slots must be MAKE_PACKED(DEAD, STATIC)"
    assert (mass[killed] == 0.0).all()
    assert np.isfinite(pos).all(), f"{profile_name}: non-finite position"
    assert np.isfinite(vel).all(), f"{profile_name}: non-finite velocity"
    assert np.isfinite(p_rho2).all(), f"{profile_name}: non-finite p_rho2"


@pytest.mark.parametrize("profile_name", ["DFSPH", "PBF"])
def test_old_dead_encoding_stays_finite(world, sim, profile_name):
    """packed_info 0 with mass left in place must not produce inf or NaN
    (rest density of DEAD is 0; the kernels floor it)."""
    pos, vel, p_rho2, _, _, killed = _kill_center_and_run(
        world, sim, profile_name, old_encoding=True)
    assert killed.size > 50
    assert np.isfinite(pos).all(), f"{profile_name}: non-finite position"
    assert np.isfinite(vel).all(), f"{profile_name}: non-finite velocity"
    assert np.isfinite(p_rho2).all(), f"{profile_name}: non-finite p_rho2"


def test_dfsph_density_solver_dead_fluid_slot_is_finite(world, sim):
    """A DEAD slot with FLUID behavior (packed_info 0, what the brush wrote
    before bd-r5-review-n28.34) must not drive p_rho2 or a_press to inf/NaN.

    DEAD has rest_density 0, so K_DFSPH_DensitySolverUpdate divided by zero.
    The kernel now floors rho0 at 1. Under --use_fast_math the unfloored
    division happens to come out as NaN, which fmaxf(new_p, 0) clamps to 0,
    so this check also passes without the floor today. It guards p_rho2 and
    a_press against inf or NaN if the build or the clamp changes.
    """
    import dataclasses
    import dfsph_solver

    w = world
    w.packed_info[:] = 0
    w._high_water = 0
    half = 0.09
    n = w.spawn_cube((-half, 0.3 - half, -half), (half, 0.3 + half, half),
                     WATER, spacing=0.02)
    profile = dataclasses.replace(PROFILES["DFSPH"], dfsph_alpha_limit=1.0e9)
    sim.set_solver_profile(profile)
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    pos = w.sorted_position[:n, :3].get()
    centre = pos.mean(axis=0)
    dist = np.linalg.norm(pos - centre, axis=1)
    dead = cp.asarray(np.nonzero(dist < 0.03)[0])
    assert dead.size > 5
    w.sorted_packed_info[dead] = cp.uint32(0)  # MAT_DEAD, FLUID

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
    assert float(w.sorted_alpha_dfsph[dead].max()) > 0.0, "dead slot has alpha 0"

    dt = float(profile.dt)
    k = 0.1 / (3.0 * dt)
    vel = cp.zeros((n, 4), dtype=cp.float32)
    vel[:, :3] = cp.asarray((-k * (pos - centre)).astype(np.float32))
    p = cp.zeros(n, dtype=cp.float32)
    accel = cp.zeros((n, 4), dtype=cp.float32)
    for _ in range(8):
        dfsph_solver.density_solver_update(
            vel, accel, w.sorted_position[:n], w.sorted_density[:n],
            w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
            w.sorted_packed_info[:n], sim._cell_start, sim._cell_end, p,
        )
        dfsph_solver.compute_pressure_accel(
            p, w.sorted_position[:n], w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], sim._cell_start, sim._cell_end, accel,
        )
    p_h = p.get()
    a_h = accel.get()
    assert np.isfinite(p_h).all(), f"p_rho2 not finite at {np.count_nonzero(~np.isfinite(p_h))} slots"
    assert np.isfinite(a_h).all(), f"a_press not finite at {np.count_nonzero(~np.isfinite(a_h).any(axis=1))} slots"


# ---------------------------------------------------------------------------
# bd-r5-review-n28.60: DEAD slots get color.w < 0 so the shaders cull them
# ---------------------------------------------------------------------------

def _render_alpha(sim, n):
    """color.w of the first n slots as the renderer gets it: one
    K_ComputeColor launch over the unsorted state (bd-r5-review-n28.145)."""
    out = cp.zeros((n, 4), dtype=cp.float32)
    sim.compute_render_color(out, n)
    return cp.asnumpy(out[:, 3])


@pytest.mark.parametrize("profile_name", ["WCSPH", "PBF", "DFSPH"])
def test_dead_slot_color_is_culled(world, sim, profile_name):
    """A slot set to MAKE_PACKED(DEAD, STATIC) gets color.w < 0 after one
    substep, and brush-killed slots get it at once. Live WATER keeps 0.0."""
    from materials import STATIC
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, spacing=0.02)
    sim.set_solver_profile(PROFILES[profile_name])
    sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None

    dead_slot = n // 2
    world.packed_info[dead_slot] = cp.uint32((STATIC & 0x3) << 8)  # DEAD, STATIC
    world.mass[dead_slot] = cp.float32(0.0)
    world.health[dead_slot] = cp.float32(0.0)
    sim._sim_step(n)

    alpha = _render_alpha(sim, n)
    assert alpha[dead_slot] < 0.0, (
        f"{profile_name}: DEAD slot alpha {alpha[dead_slot]} must be < 0")
    live = np.ones(n, dtype=bool)
    live[dead_slot] = False
    assert (alpha[live] >= 0.0).all(), f"{profile_name}: a live slot has alpha < 0"
    assert np.allclose(alpha[live], 0.0), f"{profile_name}: WATER alpha must stay 0.0"

    # Brush kill: culled right away, before any substep, and after one.
    before = world.packed_info[:n].copy()
    world.kill_in_sphere((0.0, 0.0, 0.0), 0.05)
    killed = cp.asnumpy(((before & cp.uint32(0xFF)) != 0)
                        & ((world.packed_info[:n] & cp.uint32(0xFF)) == 0))
    assert killed.sum() > 5, f"only {killed.sum()} particles killed"
    assert (_render_alpha(sim, n)[killed] < 0.0).all(), "kill must set alpha < 0"
    sim._sim_step(n)
    alpha = _render_alpha(sim, n)
    assert (alpha[killed] < 0.0).all(), f"{profile_name}: killed slot not culled after a substep"
    alive = cp.asnumpy((world.packed_info[:n] & cp.uint32(0xFF)) != 0)
    assert (alpha[alive] >= 0.0).all(), f"{profile_name}: a live slot has alpha < 0"


def test_add_dead_reserve_color_is_culled():
    """Parked reserve slots are culled from the moment they are added."""
    w = World(max_particles=4096)
    w.spawn_cube((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05), WATER, spacing=0.02)
    start = w._high_water
    added = w.add_dead_reserve(64)
    assert added == 64
    sim = Simulation(w, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20)
    assert (_render_alpha(sim, w._high_water)[start:start + added] < 0.0).all()


# ---------------------------------------------------------------------------
# bd-r5-review-n28.55: PBF grid is bucketed by predicted positions
# ---------------------------------------------------------------------------

def _stencil_hashes(sim, pos):
    """Hashes of the 27 cells around each position, matching common.cuh."""
    gp = sim._grid_params[0]
    gmin = np.array(gp["grid_min"], dtype=np.float32)
    gdel = np.array(gp["grid_delta"], dtype=np.float32)
    mask = np.uint64(gp["table_mask"])
    cell = np.floor((pos.astype(np.float32) - gmin) * gdel).astype(np.int64)
    offs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
                     for c in (-1, 0, 1)], dtype=np.int64)
    primes = np.array([73856093, 19349669, 83492791], dtype=np.uint64)
    stencil = (cell[:, None, :] + offs[None, :, :]).astype(np.uint64)
    return ((stencil * primes).sum(axis=2) & mask).astype(np.uint32)


def _pbf_sort_and_predict(sim, world, n):
    import pbf_solver
    sim._run_grid_setup(n, force_sort=True)
    pbf_solver.pbf_predict(
        world.sorted_position[:n], world.sorted_velocity[:n],
        world.sorted_packed_info[:n], world.sorted_temperature[:n],
        world.sorted_predicted_position,
    )


def test_pbf_grid_finds_every_predicted_pair(world, sim):
    """Every pair with |x*_i - x*_j| < h is reachable from the 27-cell
    stencil of cell(x*_i) after a full sort, on a block moving at 2 m/s."""
    world.packed_info[:] = 0
    world._high_water = 0
    sim.set_solver_profile(PROFILES["PBF"])
    n = world.spawn_cube(
        min_corner=(-0.15, -0.15, -0.15), max_corner=(0.15, 0.15, 0.15),
        material_id=WATER, spacing=0.02, velocity=(2.0, 0.6, -1.0),
    )
    h = 0.04
    _pbf_sort_and_predict(sim, world, n)

    pred = cp.asnumpy(world.sorted_predicted_position[:n])[:, :3]
    stencil_d = cp.asarray(_stencil_hashes(sim, pred))        # (n, 27)
    shash_d = world.hashes[:n][sim._sort_perm[:n].astype(cp.int64)]
    pred_d = cp.asarray(pred)
    pairs = 0
    missed = 0
    for s in range(0, n, 512):
        e = min(n, s + 512)
        d = pred_d[s:e, None, :] - pred_d[None, :, :]
        close = (d * d).sum(axis=2) < h * h
        close[cp.arange(e - s), cp.arange(s, e)] = False
        reach = (shash_d[None, :, None] == stencil_d[s:e, None, :]).any(axis=2)
        pairs += int(close.sum())
        missed += int((close & ~reach).sum())
    assert pairs > 10 * n, f"scene too sparse: {pairs} pairs for {n} particles"
    assert missed == 0, f"{missed} of {pairs} predicted-position pairs not in the stencil"


def test_pbf_lambda_sees_trailing_neighbour(world, sim):
    """Two particles both move 0.6h in one substep; j trails i by 0.95h after
    predict. Bucketed by x, j would sit 2 cells behind x*_i and be skipped."""
    import pbf_solver
    world.packed_info[:] = 0
    world._high_water = 0
    sim.set_solver_profile(PROFILES["PBF"])
    spawned = world.spawn_cube(
        min_corner=(-0.02, -0.02, -0.02), max_corner=(0.02, 0.02, 0.02),
        material_id=WATER, spacing=0.02,
    )
    assert spawned >= 2
    n = 2
    h = 0.04
    dt = float(sim.dt)
    speed = 0.6 * h / dt
    gp = sim._grid_params[0]
    gmin_x = float(gp["grid_min"][0])
    cs = 1.0 / float(gp["grid_delta"][0])
    # x*_i sits 0.05h past a cell edge, so x*_j = x*_i - 0.95h is one cell
    # behind and x_j = x*_j - 0.6h is two cells behind.
    k = int(np.floor((0.0 - gmin_x) / cs))
    pred_ix = gmin_x + k * cs + 0.05 * h

    def density_i(gap):
        pos = cp.asnumpy(world.position[:n]).copy()
        vel = np.zeros((n, 4), dtype=np.float32)
        vel[:, 0] = speed
        pos[0, :3] = (pred_ix - 0.6 * h, 0.0, 0.0)
        pos[1, :3] = (pred_ix - gap - 0.6 * h, 0.0, 0.0)
        world.position[:n] = cp.asarray(pos)
        world.velocity[:n] = cp.asarray(vel)
        _pbf_sort_and_predict(sim, world, n)
        pbf_solver.pbf_compute_lambda(
            world.sorted_predicted_position[:n], world.sorted_mass[:n],
            world.sorted_packed_info[:n], sim._cell_start, sim._cell_end,
            world.sorted_density, world.sorted_lambda_pbf,
        )
        perm = cp.asnumpy(sim._sort_perm[:n])
        s_i = int(np.nonzero(perm == 0)[0][0])
        pred = cp.asnumpy(world.sorted_predicted_position[:n])
        gap_pred = pred[s_i, 0] - pred[1 - s_i, 0]
        return float(cp.asnumpy(world.sorted_density[:n])[s_i]), gap_pred

    rho_alone, _ = density_i(10.0 * h)
    rho_pair, gap_pred = density_i(0.95 * h)
    assert abs(gap_pred - 0.95 * h) < 1e-4, f"predicted gap {gap_pred / h:.3f}h"
    mass = float(cp.asnumpy(world.sorted_mass[:n])[0])
    w_poly6 = 315.0 / (64.0 * np.pi * h ** 9) * (h * h - (0.95 * h) ** 2) ** 3
    expected = mass * w_poly6
    assert rho_pair - rho_alone > 0.5 * expected, (
        f"lambda_i did not see j: rho gain {rho_pair - rho_alone:.4g}, "
        f"expected about {expected:.4g}"
    )


# ---------------------------------------------------------------------------
# bd-r5-review-n28.49: gunpowder burst and blast wave survive PBF Finalize
# ---------------------------------------------------------------------------

def test_pbf_gunpowder_blast_moves_water(world, sim, monkeypatch):
    """One gunpowder particle inside a water block ignites during a PBF
    substep. K_PBF_Predict has already consumed velocity, so the burst and
    blast must reach x* or Finalize's (x* - x)/dt drops them."""
    from materials import GUNPOWDER
    world.packed_info[:] = 0
    world._high_water = 0
    sim.set_solver_profile(PROFILES["PBF"])
    sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None
    tiny = ((-0.001, -0.001, -0.001), (0.001, 0.001, 0.001))
    # spawn_cube puts lattice points at min + spacing/2, so both cubes put
    # a point at the origin; the water one is excluded.
    n_w = world.spawn_cube((-0.07, -0.07, -0.07), (0.07, 0.07, 0.07), WATER,
                           spacing=0.02, exclude_box=tiny)
    n_g = world.spawn_cube((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), GUNPOWDER,
                           spacing=0.02)
    assert n_g == 1
    n = n_w + n_g
    g_idx = n - 1
    h = 0.04

    # Ignite the gunpowder: Reactions reads sorted_exposure_heat, which the
    # first lambda pass has just written. Force it above the threshold.
    orig = sim._run_reactions_spawn

    def ignite(n_, mark=None, predicted_position=None):
        gp = (world.sorted_packed_info[:n_] & cp.uint32(0xFF)) == cp.uint32(GUNPOWDER)
        world.sorted_exposure_heat[:n_][gp] = cp.float32(200.0)
        if mark is None:
            return orig(n_, predicted_position=predicted_position)
        return orig(n_, mark, predicted_position=predicted_position)

    monkeypatch.setattr(sim, "_run_reactions_spawn", ignite)

    pos0 = cp.asnumpy(world.position[:n])[:, :3]
    sim._sim_step(n)

    pi = cp.asnumpy(world.packed_info[:n])
    vel = cp.asnumpy(world.velocity[:n])[:, :3]
    from materials import FIRE
    assert (pi[g_idx] & 0xFF) == FIRE, "gunpowder did not ignite"

    r = pos0[:n_w] - pos0[g_idx]
    d = np.linalg.norm(r, axis=1)
    near = d < h
    assert near.sum() >= 20, f"only {near.sum()} water neighbours within h"
    radial = (vel[:n_w][near] * (r[near] / d[near, None])).sum(axis=1)
    mean_radial = float(radial.mean())
    fire_speed = float(np.linalg.norm(vel[g_idx]))
    print(f"  PBF blast: mean radial water speed {mean_radial:.3f} m/s, "
          f"fire speed {fire_speed:.3f} m/s")
    assert mean_radial > 1.0, (
        f"PBF lost the blast: mean outward water speed {mean_radial:.3f} m/s")
    # The 25 m/s burst is far above the CFL clamp (0.9 h / dt, 4.3 m/s at
    # dt = 1/120), so a kept burst leaves the FIRE particle at the clamp.
    v_limit = float(sim._sim_params[0]["velocity_limit"])
    assert fire_speed > 0.9 * v_limit, (
        f"PBF lost the gunpowder burst: {fire_speed:.3f} m/s, limit {v_limit:.3f}")


# ---------------------------------------------------------------------------
# bd-r5-review-n28.72: PBF and DFSPH exposure matches the normalised K_Step1
# ---------------------------------------------------------------------------

def test_exposure_matches_across_solvers(world, sim):
    """WCSPH K_Step1, K_PBF_ComputeLambda (do_heat call) and
    K_DFSPH_ComputeDensityAlpha must all output exposure as the support
    fraction sum(rate * W) / sum(W) over non-self, non-DEAD neighbours.

    The centre SAND particle has exactly 10 WATER neighbours; the rest of the
    block mixes SAND, WATER, WOOD (STATIC), FIRE (GAS, cross-phase path) and
    MAT_RIGID, with WATER and FIRE hotter so exposure_heat is non-zero.
    """
    import dfsph_solver
    import pbf_solver
    import step1
    from materials import WOOD, FIRE, MAT_RIGID, MATERIALS, build_interaction_matrix

    w = world
    w.packed_info[:] = 0
    w._high_water = 0
    # Upload constants to both solver modules; the last profile set wins for
    # sim state, but each RawModule keeps its own copy of the constants.
    sim.set_solver_profile(PROFILES["PBF"])
    sim.set_solver_profile(PROFILES["DFSPH"])
    half = 0.05
    n = w.spawn_cube((-half, 0.3 - half, -half), (half, 0.3 + half, half),
                     SAND, spacing=0.02)
    assert n >= 100
    rng = np.random.default_rng(72)
    pos = cp.asnumpy(w.position[:n]).copy()
    pos[:, :3] += rng.uniform(-0.002, 0.002, size=(n, 3)).astype(np.float32)
    w.position[:n] = cp.asarray(pos)
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    h = 0.04
    spos = cp.asnumpy(w.sorted_position[:n])[:, :3].astype(np.float64)
    centre = spos.mean(axis=0)
    c = int(np.argmin(np.linalg.norm(spos - centre, axis=1)))
    d_c = np.linalg.norm(spos - spos[c], axis=1)
    nbr_c = np.nonzero((d_c < h) & (np.arange(n) != c))[0]
    assert nbr_c.size >= 20, f"centre has only {nbr_c.size} neighbours"

    def packed(mat):
        return (mat & 0xFF) | ((MATERIALS[mat].behavior_class & 0x3) << 8)

    mats = np.full(n, SAND, dtype=np.int64)
    far = np.nonzero(d_c >= h)[0]
    choice = rng.choice([SAND, WATER, WOOD, FIRE, MAT_RIGID], size=far.size,
                        p=[0.4, 0.3, 0.1, 0.1, 0.1])
    mats[far] = choice
    water_c = rng.choice(nbr_c, size=10, replace=False)
    mats[water_c] = WATER
    info = np.array([packed(int(m)) for m in mats], dtype=np.uint32)
    w.sorted_packed_info[:n] = cp.asarray(info)
    temp = np.where((mats == WATER) | (mats == FIRE), 330.0, 300.0).astype(np.float32)
    w.sorted_temperature[:n] = cp.asarray(temp)

    def outs():
        return (cp.zeros(n, cp.float32), cp.zeros(n, cp.float32),
                cp.zeros(n, cp.float32))

    # WCSPH
    s1_dTdt, s1_eh, s1_ec = outs()
    # rho_j = rest density, the same fallback PBF and DFSPH use for density_in=None.
    w.sorted_position[:n, 3] = 0.0
    step1.compute_step1(
        w.sorted_position[:n], w.sorted_velocity[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        density_out=cp.zeros(n, cp.float32), shear_rate_out=cp.zeros(n, cp.float32),
        dTdt_out=s1_dTdt, exposure_heat_out=s1_eh, exposure_corrode_out=s1_ec,
    )
    # PBF: predicted positions = current positions, do_heat call
    pbf_dTdt, pbf_eh, pbf_ec = outs()
    pbf_solver.pbf_compute_lambda(
        w.sorted_position[:n], w.sorted_mass[:n], w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end,
        cp.zeros(n, cp.float32), cp.zeros(n, cp.float32),
        temperature_in=w.sorted_temperature[:n],
        dTdt_out=pbf_dTdt, exposure_heat_out=pbf_eh, exposure_corrode_out=pbf_ec,
    )
    # DFSPH
    df_dTdt, df_eh, df_ec = outs()
    dfsph_solver.compute_density_alpha(
        w.sorted_position[:n], w.sorted_velocity[:n], w.sorted_mass[:n], None,
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        cp.zeros(n, cp.float32), cp.zeros(n, cp.float32), cp.zeros(n, cp.float32),
        df_dTdt, df_eh, df_ec,
    )

    ec = {k: cp.asnumpy(v) for k, v in
          (("wcsph", s1_ec), ("pbf", pbf_ec), ("dfsph", df_ec))}
    eh = {k: cp.asnumpy(v) for k, v in
          (("wcsph", s1_eh), ("pbf", pbf_eh), ("dfsph", df_eh))}

    # Centre SAND particle: W-weighted water fraction times the SAND-WATER rate.
    inter = build_interaction_matrix()
    rate_sw = float(inter[SAND][WATER]["reaction_rate"])
    wvar = (h * h - d_c[nbr_c] ** 2) ** 3
    is_w = np.isin(nbr_c, water_c)
    expected_c = rate_sw * wvar[is_w].sum() / wvar.sum()
    for name in ec:
        assert abs(ec[name][c] - expected_c) < 1e-4, (
            f"{name} centre exposure_corrode {ec[name][c]:.6f}, "
            f"expected {expected_c:.6f}")

    max_rate = float(inter["reaction_rate"].max())
    live = mats != MAT_RIGID  # MAT_RIGID slots early-return differently per solver
    for name in ec:
        assert ec[name].max() <= max_rate + 1e-5, (
            f"{name} exposure_corrode {ec[name].max():.4g} > max rate {max_rate}")
    for name in ("pbf", "dfsph"):
        dc = np.abs(ec[name][live] - ec["wcsph"][live]).max()
        assert dc < 1e-4, f"{name} exposure_corrode differs from WCSPH by {dc:.3g}"
        ref_h = np.abs(eh["wcsph"][live]).max()
        assert ref_h > 1.0, f"scene gives no heat exposure ({ref_h:.3g})"
        dh = np.abs(eh[name][live] - eh["wcsph"][live]).max()
        assert dh < 1e-4 * max(1.0, ref_h), (
            f"{name} exposure_heat differs from WCSPH by {dh:.3g} (max {ref_h:.3g})")


# ---------------------------------------------------------------------------
# bd-r5-review-n28.94: GAS/non-GAS heat conduction agrees across solvers
# ---------------------------------------------------------------------------

def test_cross_phase_heat_matches_across_solvers(world, sim):
    """WCSPH K_Step1, K_PBF_ComputeLambda (do_heat call) and
    K_DFSPH_ComputeDensityAlpha give the same dTdt in a block that mixes
    WATER, SAND, STONE (STATIC), FIRE and STEAM (GAS).

    Every non-GAS particle starts at 293 K and every GAS particle is hotter,
    so the only heat source for WATER, SAND and STONE is the cross-phase term.
    Before bd-r5-review-n28.94 all three solvers skipped it and every non-GAS
    dTdt was exactly 0.
    """
    import dfsph_solver
    import pbf_solver
    import step1
    from materials import STONE, FIRE, STEAM, MATERIALS

    w = world
    w.packed_info[:] = 0
    w._high_water = 0
    sim.set_solver_profile(PROFILES["PBF"])
    sim.set_solver_profile(PROFILES["DFSPH"])
    half = 0.05
    n = w.spawn_cube((-half, 0.3 - half, -half), (half, 0.3 + half, half),
                     WATER, spacing=0.02)
    assert n >= 100
    rng = np.random.default_rng(94)
    pos = cp.asnumpy(w.position[:n]).copy()
    pos[:, :3] += rng.uniform(-0.002, 0.002, size=(n, 3)).astype(np.float32)
    w.position[:n] = cp.asarray(pos)
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)

    def packed(mat):
        return (mat & 0xFF) | ((MATERIALS[mat].behavior_class & 0x3) << 8)

    mats = rng.choice([WATER, SAND, STONE, FIRE, STEAM], size=n,
                      p=[0.4, 0.15, 0.1, 0.25, 0.1])
    info = np.array([packed(int(m)) for m in mats], dtype=np.uint32)
    w.sorted_packed_info[:n] = cp.asarray(info)
    is_gas = (mats == FIRE) | (mats == STEAM)
    temp = np.where(mats == FIRE, 1200.0, np.where(mats == STEAM, 400.0, 293.0))
    w.sorted_temperature[:n] = cp.asarray(temp.astype(np.float32))

    def outs():
        return (cp.zeros(n, cp.float32), cp.zeros(n, cp.float32),
                cp.zeros(n, cp.float32))

    s1_dTdt, s1_eh, s1_ec = outs()
    # rho_j = rest density, the same fallback PBF and DFSPH use for density_in=None.
    w.sorted_position[:n, 3] = 0.0
    step1.compute_step1(
        w.sorted_position[:n], w.sorted_velocity[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        density_out=cp.zeros(n, cp.float32), shear_rate_out=cp.zeros(n, cp.float32),
        dTdt_out=s1_dTdt, exposure_heat_out=s1_eh, exposure_corrode_out=s1_ec,
    )
    pbf_dTdt, pbf_eh, pbf_ec = outs()
    pbf_solver.pbf_compute_lambda(
        w.sorted_position[:n], w.sorted_mass[:n], w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end,
        cp.zeros(n, cp.float32), cp.zeros(n, cp.float32),
        temperature_in=w.sorted_temperature[:n],
        dTdt_out=pbf_dTdt, exposure_heat_out=pbf_eh, exposure_corrode_out=pbf_ec,
    )
    df_dTdt, df_eh, df_ec = outs()
    dfsph_solver.compute_density_alpha(
        w.sorted_position[:n], w.sorted_velocity[:n], w.sorted_mass[:n], None,
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        cp.zeros(n, cp.float32), cp.zeros(n, cp.float32), cp.zeros(n, cp.float32),
        df_dTdt, df_eh, df_ec,
    )
    d = {k: cp.asnumpy(v).astype(np.float64) for k, v in
         (("wcsph", s1_dTdt), ("pbf", pbf_dTdt), ("dfsph", df_dTdt))}

    # The cross-phase term must be live: non-GAS particles only gain heat,
    # and most of them gain some.
    solid = ~is_gas
    ref = d["wcsph"]
    print(f"  non-GAS dTdt: max {ref[solid].max():.1f} K/s, "
          f"{np.count_nonzero(ref[solid] > 0)}/{solid.sum()} heated")
    assert ref[solid].min() >= -1e-3, ref[solid].min()
    assert np.count_nonzero(ref[solid] > 1.0) > 0.5 * solid.sum()

    scale = np.abs(ref).max()
    for name in ("pbf", "dfsph"):
        diff = np.abs(d[name] - ref).max()
        print(f"  {name}: max |dTdt - WCSPH| = {diff:.3g} (scale {scale:.3g})")
        assert diff < 1e-4 * scale, (
            f"{name} dTdt differs from WCSPH by {diff:.3g} (max {scale:.3g})")


def _read_pbf_params():
    import pbf_solver
    module = pbf_solver._get_module()
    d_ptr = module.get_global("c_pbf")
    out = np.zeros(1, dtype=pbf_solver.PBF_PARAMS_DTYPE)
    cp.cuda.runtime.memcpy(out.ctypes.data, int(d_ptr), out.nbytes, 2)
    return out[0]


def test_pbf_s_corr_w_dq_zero_when_k_zero():
    """bd-r5-review-n28.174: with s_corr_k = 0, W_dq uploads as 0 so
    K_PBF_ComputeDelta skips the s_corr branch. With k > 0, W_dq is the
    analytic poly6 value at dq*h."""
    import dataclasses
    import math
    import pbf_solver
    h = 0.04
    base = PROFILES["PBF"]

    pbf_solver.upload_pbf_params(dataclasses.replace(base, pbf_s_corr_k=0.0), h=h)
    p = _read_pbf_params()
    assert float(p["s_corr_W_dq"]) == 0.0

    prof = dataclasses.replace(base, pbf_s_corr_k=0.1)
    pbf_solver.upload_pbf_params(prof, h=h)
    p = _read_pbf_params()
    dq = prof.pbf_s_corr_dq * h
    expected = 315.0 / (64.0 * math.pi * h ** 9) * (h * h - dq * dq) ** 3
    assert float(p["s_corr_k"]) == pytest.approx(0.1, rel=1e-6)
    assert float(p["s_corr_W_dq"]) == pytest.approx(expected, rel=1e-6)

    # Leave the default profile in constant memory for later tests.
    pbf_solver.upload_pbf_params(base, h=h)


# ---------------------------------------------------------------------------
# Wake mark written by K_PBF_Finalize / K_DFSPH_Finalize (bd-r5-review-n28.167)
# ---------------------------------------------------------------------------

# Sleeper and mover sit 0.05 m apart (more than h), one cell apart in y, so
# only the wake pass can wake the sleeper. The 0.015 offsets keep them off a
# cell boundary. The far sleeper is well outside the mover's 27 cells.
_WAKE_NEAR = (0.015, 0.02, 0.015)
_WAKE_FAR = (0.515, 0.02, 0.515)
_WAKE_MOVER = (0.015, 0.07, 0.015)


def _spawn_one(world, center, material_id, velocity=(0.0, 0.0, 0.0)):
    lo = tuple(c - 0.001 for c in center)
    hi = tuple(c + 0.001 for c in center)
    n = world.spawn_cube(lo, hi, material_id, spacing=0.002, velocity=velocity)
    assert n == 1


def _grid_cell(p, grid_params):
    """calcGridCell in float32, as the kernels compute it."""
    gmin = np.asarray(grid_params[0]["grid_min"], dtype=np.float32)
    delta = np.asarray(grid_params[0]["grid_delta"], dtype=np.float32)
    return np.floor((np.asarray(p, dtype=np.float32) - gmin) * delta).astype(np.int64)


@pytest.mark.parametrize("solver", ["PBF", "DFSPH"])
@pytest.mark.parametrize("mover_material,wakes", [(WATER, True), ("SMOKE", False)])
def test_finalize_wake_mark_wakes_sleeping_sand(world, sim, monkeypatch,
                                                solver, mover_material, wakes):
    """A fast FLUID particle one cell above a sleeping SAND particle wakes it
    after one substep, with K_MarkWakeCells never launched: the stamp comes
    from the finalize kernel. A fast GAS particle does not wake it (step1 and
    step2 give GAS no forces on non-GAS). A far sleeper stays asleep."""
    import wake
    from materials import SMOKE
    if mover_material == "SMOKE":
        mover_material = SMOKE

    def _no_mark(*args, **kwargs):
        raise AssertionError(f"{solver} launched K_MarkWakeCells")

    monkeypatch.setattr(wake, "mark_wake_cells", _no_mark)

    world.packed_info[:] = 0
    world._high_water = 0
    world.sorted_density[:] = 0.0
    world.sleep_counter[:] = 0
    world.velocity[:] = 0.0
    _spawn_one(world, _WAKE_NEAR, SAND)
    _spawn_one(world, _WAKE_FAR, SAND)
    world.packed_info[:2] |= cp.uint32(_SLEEP_BIT)
    world.sleep_counter[:2] = 200
    _spawn_one(world, _WAKE_MOVER, mover_material, velocity=(0.0, -0.1, 0.0))
    n = 3

    sim.set_solver_profile(PROFILES[solver])
    sim.set_gravity(0.0)
    sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None
    try:
        sim._sim_step(n)
        pi = cp.asnumpy(world.packed_info[:n])
        sc = cp.asnumpy(world.sleep_counter[:n])
        vel = cp.asnumpy(world.velocity[:n])
        pos = cp.asnumpy(world.position[:n])[:, :3]
    finally:
        sim.set_gravity(-4.0)

    gp = sim._grid_params
    near_d = _grid_cell(pos[0], gp) - _grid_cell(pos[2], gp)
    far_d = _grid_cell(pos[1], gp) - _grid_cell(pos[2], gp)
    assert np.abs(near_d).max() == 1, f"near sleeper not in the next cell: {near_d}"
    assert np.abs(far_d).max() > 1, f"far sleeper inside the stencil: {far_d}"
    assert np.dot(vel[2, :3], vel[2, :3]) > 0.02 ** 2, f"mover slowed: {vel[2]}"

    near_asleep = bool(pi[0] & _SLEEP_BIT)
    far_asleep = bool(pi[1] & _SLEEP_BIT)
    print(f"\n  {solver} mover {mover_material}: near asleep {near_asleep} "
          f"(sc {sc[0]}), far asleep {far_asleep} (sc {sc[1]})")
    assert far_asleep and sc[1] == 200, f"{solver}: far sand woke"
    if wakes:
        assert not near_asleep, f"{solver}: finalize stamp did not wake the sand"
        assert sc[0] == 0, f"{solver}: sleep_counter not reset: {sc[0]}"
    else:
        assert near_asleep and sc[0] == 200, f"{solver}: GAS mover woke the sand"
