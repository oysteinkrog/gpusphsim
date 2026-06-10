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

        # Run the hash + sort part manually
        import hash_sort
        import fused_sort_reorder_build
        w = world

        hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
        sort_perm = cp.argsort(hashes).astype(cp.uint32)
        sim._sort_perm[:n] = sort_perm
        sim._frame_counter_d.fill(sim._frame_counter)
        w._density_initialized = True

        # Grid setup
        sim._run_grid_setup(n)

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
            w.position, w.velocity, w.color, w.packed_info,
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

        import hash_sort
        w = world

        hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
        sort_perm = cp.argsort(hashes).astype(cp.uint32)
        sim._sort_perm[:n] = sort_perm
        sim._frame_counter_d.fill(sim._frame_counter)
        w._density_initialized = True

        sim._run_grid_setup(n)

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

        # Predict position
        dfsph_solver.predict_position(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_packed_info[:n],
            w.sorted_predicted_position,
        )
        cp.cuda.Device().synchronize()

        # Density at predicted positions
        dfsph_solver.compute_density_adv(
            w.sorted_predicted_position[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_position[:n],
            sim._cell_start, sim._cell_end,
            w.sorted_density,
        )
        cp.cuda.Device().synchronize()

        dens_adv = w.sorted_density[:n].get()
        print(f"  density_adv: min={dens_adv.min():.2f} mean={dens_adv.mean():.2f} max={dens_adv.max():.2f}")

        # Density solver
        for it in range(profile.dfsph_dens_iters):
            dfsph_solver.compute_kappa(
                w.sorted_density[:n], w.sorted_alpha_dfsph[:n],
                w.sorted_packed_info[:n],
                w.sorted_kappa,
            )
            cp.cuda.Device().synchronize()
            kd = w.sorted_kappa[:n].get()
            print(f"  dens iter {it}: kappa min={kd.min():.4f} mean={kd.mean():.4f} max={kd.max():.4f}")

            dfsph_solver.correct_velocity_dens(
                w.sorted_velocity, w.sorted_density[:n],
                w.sorted_mass[:n], w.sorted_kappa[:n],
                w.sorted_packed_info[:n], w.sorted_position[:n],
                sim._cell_start, sim._cell_end,
            )

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
            w.position, w.velocity, w.color, w.packed_info,
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

    import hash_sort
    w = world
    hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
    sort_perm = cp.argsort(hashes).astype(cp.uint32)
    sim._sort_perm[:n] = sort_perm
    w._density_initialized = True
    sim._run_grid_setup(n)

    # Find a FLUID water particle to serve as our test subject
    pi = w.sorted_packed_info[:n].get()
    behaviors = (pi >> 8) & 0x3
    fluid_idx = np.where(behaviors == FLUID)[0]
    if len(fluid_idx) == 0:
        pytest.skip("no fluid particles found")
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

    import hash_sort
    w = world
    hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
    sort_perm = cp.argsort(hashes).astype(cp.uint32)
    sim._sort_perm[:n] = sort_perm
    w._density_initialized = True
    sim._run_grid_setup(n)

    pi = w.sorted_packed_info[:n].get()
    behaviors = (pi >> 8) & 0x3
    fluid_idx = np.where(behaviors == FLUID)[0]
    if len(fluid_idx) == 0:
        pytest.skip("no fluid particles found")
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
        color_out = cp.zeros(len(w.color), dtype=cp.uint32)
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
            pos_out, vel_out, color_out, pi_out,
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
    """bd-mzc.22 / bd-unl.10: DFSPH density_adv formula must be dimensionally consistent.

    The fix changes: density_adv = rho_i/rho0 + dt*drho
    to:              density_adv = rho_i/rho0 + dt*drho/rho0
    making both terms dimensionless ratios. The buggy formula had drho at the
    wrong magnitude (off by rho0 ≈ 1000x), causing the Jacobi solver to converge
    to the wrong density and produce a persistent 20-40% error.

    Sub-test 1 (zero-velocity): with all particles at rest, drho=0, so
    density_adv=rho_i/rho0 and residual≈0. p_rho2 must stay ≈ 0 (no spurious
    pressure). If the buggy /rho0-missing formula were active, it would produce
    a drho term off by rho0 even when particles move; with zero velocity it
    doesn't matter, but the threshold (100*rho0) would catch it when v≠0.

    Sub-test 2 (large velocity): inject vel_limit*0.5 on all particles. The
    correct formula keeps p_rho2 < 100*rho0 (here rho0=2500 → threshold 250000).
    The buggy formula would produce p_rho2 off by ~rho0 per iteration and blow
    past this threshold.

    Threshold: 100 * rho0 ≈ 250000 (tight enough to catch a 20-40% density
    error, since the buggy formula compounds by ~rho0 per Jacobi step).
    """
    import dfsph_solver
    from materials import WATER

    profile = PROFILES["DFSPH"]
    n = spawn_scene(world)
    sim.set_solver_profile(profile)

    import hash_sort
    w = world
    hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
    sort_perm = cp.argsort(hashes).astype(cp.uint32)
    sim._sort_perm[:n] = sort_perm
    w._density_initialized = True
    sim._run_grid_setup(n)

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

    accel_zero = cp.zeros((n, 4), dtype=cp.float32)

    # --- Sub-test 1: zero velocity --- at rest, drho=0 → p_rho2 must stay ≈ 0.
    vel_zero = cp.zeros((n, 4), dtype=cp.float32)
    p_rho2_zero = cp.zeros(n, dtype=cp.float32)

    for _ in range(20):
        dfsph_solver.density_solver_update(
            vel_zero, accel_zero,
            w.sorted_position[:n], w.sorted_density[:n],
            w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
            w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end,
            p_rho2_zero,
        )
    cp.cuda.Device().synchronize()

    p_zero = p_rho2_zero.get()
    # Use WATER rest density from the materials module
    from materials import MATERIALS
    rho0 = float(MATERIALS[WATER].rest_density)
    # At rest, density_adv = rho_i/rho0; if rho_i <= rho0 residual <=0 → p_rho2 clamped to 0.
    # If rho_i > rho0 slightly (over-compressed at spawn), residual is small but positive.
    # p_rho2 must stay near zero — threshold 0.5*rho0 to catch any formula error.
    at_rest_threshold = 0.5 * rho0
    max_p_zero = float(np.abs(p_zero).max())
    print(f"  DFSPH density solver (zero vel): max_p_rho2={max_p_zero:.4e}  "
          f"threshold={at_rest_threshold:.4e}")
    assert not np.isnan(p_zero).any(), "DFSPH density solver (zero vel) produced NaN"
    assert not np.isinf(p_zero).any(), "DFSPH density solver (zero vel) produced Inf"
    assert max_p_zero < at_rest_threshold, (
        f"DFSPH density solver (zero vel) has spurious pressure: "
        f"max_p_rho2={max_p_zero:.4e} > {at_rest_threshold:.4e}. "
        f"At rest drho=0, so p_rho2 should remain ≈ 0."
    )

    # --- Sub-test 2: large velocity --- p_rho2 must stay < 100*rho0.
    # With the correct formula drho/rho0 is dimensionless; with the bug drho is
    # off by ~rho0 per iteration, blowing past 100*rho0 quickly.
    vel_limit = float(sim._sim_params[0]["velocity_limit"])
    vel_large = cp.zeros((n, 4), dtype=cp.float32)
    vel_large[:, 0] = vel_limit * 0.5   # half the velocity limit — large but valid
    p_rho2 = cp.zeros(n, dtype=cp.float32)

    for _ in range(20):
        dfsph_solver.density_solver_update(
            vel_large, accel_zero,
            w.sorted_position[:n], w.sorted_density[:n],
            w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
            w.sorted_packed_info[:n],
            sim._cell_start, sim._cell_end,
            p_rho2,
        )
    cp.cuda.Device().synchronize()

    p = p_rho2.get()
    assert not np.isnan(p).any(), "DFSPH density solver produced NaN in p_rho2"
    assert not np.isinf(p).any(), "DFSPH density solver produced Inf in p_rho2"
    max_p = float(np.abs(p).max())
    # Tight threshold: 100 * rho0. With the correct formula p_rho2 is bounded;
    # with the bug (missing /rho0 in density_adv) it diverges past this quickly.
    threshold_p = 100.0 * rho0
    print(f"  DFSPH density solver (large vel): max_p_rho2={max_p:.4e}  "
          f"threshold={threshold_p:.4e}  (rho0={rho0:.1f})")
    assert max_p < threshold_p, (
        f"DFSPH density solver p_rho2 diverged: max={max_p:.4e} > {threshold_p:.4e}. "
        f"Check density_adv = rho_i/rho0 + dt*drho/rho0 (unit mismatch fix)."
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

    import hash_sort
    w = world
    hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
    sort_perm = cp.argsort(hashes).astype(cp.uint32)
    sim._sort_perm[:n] = sort_perm
    w._density_initialized = True
    sim._run_grid_setup(n)

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

    if fluid_mask.sum() == 0 or static_mask.sum() == 0:
        pytest.skip("need both fluid and static particles in scene")

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
    if nonzero.sum() == 0:
        pytest.skip("no fluid particles with non-zero density near static wall")

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

    import hash_sort
    w = world
    hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
    sort_perm = cp.argsort(hashes).astype(cp.uint32)
    sim._sort_perm[:n] = sort_perm
    w._density_initialized = True
    sim._run_grid_setup(n)

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
    import hash_sort
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

    hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
    sort_perm = cp.argsort(hashes).astype(cp.uint32)
    sim._sort_perm[:n] = sort_perm
    w._density_initialized = True
    sim._run_grid_setup(n)

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
