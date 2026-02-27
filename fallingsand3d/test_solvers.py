"""Headless solver diagnostic -- runs each solver for a few frames and prints stats."""

import sys
import numpy as np

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

        # Finalize
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

        # Finalize
        dfsph_solver.finalize(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_dTdt[:n],
            w.sorted_sleep_counter[:n],
            sim._sort_perm[:n], sim._cell_start, sim._cell_end,
            w.position, w.velocity, w.color, w.packed_info,
            w.sleep_counter, w.temperature,
        )
        cp.cuda.Device().synchronize()

        out_vel = w.velocity[:n].get()
        out_vel_mag = np.linalg.norm(out_vel[:,:3], axis=1)
        out_pos = w.position[:n].get()
        print(f"  output pos_y: [{out_pos[:,1].min():.4f}, {out_pos[:,1].mean():.4f}, {out_pos[:,1].max():.4f}]")
        print(f"  output vel: mean={out_vel_mag.mean():.4f} max={out_vel_mag.max():.4f}")

        sim._frame_counter += 1


def test_stability(world, sim, profile_name, num_substeps=10):
    """Quick stability check -- run N substeps and print summary."""
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


def main():
    world = World(max_particles=MAX_PARTICLES)
    n = spawn_scene(world)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20)

    # Quick stability checks (20 substeps each)
    test_stability(world, sim, "PBF", 20)
    test_stability(world, sim, "DFSPH", 20)


if __name__ == "__main__":
    main()
