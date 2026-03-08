"""Performance benchmark harness for SPH simulation.

Runs headless (no rendering) with configurable particle counts and
reports detailed per-kernel GPU timing, memory usage, and throughput.

Usage:
    python benchmark.py                    # default: 100K particles, 50 steps
    python benchmark.py --particles 1000000 --steps 100
    python benchmark.py --sweep            # run 100K, 250K, 500K, 1M
    python benchmark.py --profile          # nsight-compatible single run

Outputs:
    - Per-kernel timing breakdown (ms) with min/max/avg
    - Total substep time
    - Throughput: particles/sec, interactions/sec (estimated)
    - Memory usage: per-particle bytes, total VRAM
    - Sort bandwidth and skip rate
"""

from __future__ import annotations

import argparse
import time
import sys
import os
import math

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def create_dam_break(world, num_particles: int, spacing: float = 0.02):
    """Spawn a dam break scene: water block filling left quarter of domain."""
    from materials import WATER

    # Compute cube dimensions to get approximately num_particles
    # For a cubic arrangement: N = (Lx/s) * (Ly/s) * (Lz/s)
    # We want a tall block on the left side
    # Aspect ratio: width=1, height=2, depth=1 (relative)
    # N = (W/s) * (2W/s) * (W/s) = 2*(W/s)^3
    # W = s * (N/2)^(1/3)
    target = num_particles
    w = spacing * (target / 2.0) ** (1.0 / 3.0)
    h = 2.0 * w

    # Center the block on the left side of the domain
    x0, x1 = -0.9, -0.9 + w
    y0, y1 = -0.9, -0.9 + h
    z0, z1 = -w / 2.0, w / 2.0

    actual = world.spawn_cube(
        min_corner=(x0, y0, z0),
        max_corner=(x1, y1, z1),
        material_id=WATER,
        spacing=spacing,
    )
    return actual


class BenchmarkTimer:
    """CUDA event-based timer for kernel stages."""

    def __init__(self):
        import cupy
        self.events = []
        self.labels = []
        self._cupy = cupy

    def mark(self, label: str):
        e = self._cupy.cuda.Event()
        e.record()
        self.events.append(e)
        self.labels.append(label)

    def results_ms(self) -> dict:
        """Synchronize and return {label: ms} for each interval."""
        if len(self.events) < 2:
            return {}
        self.events[-1].synchronize()
        out = {}
        for i in range(1, len(self.events)):
            out[self.labels[i]] = self._cupy.cuda.get_elapsed_time(
                self.events[i - 1], self.events[i]
            )
        return out


def run_benchmark(
    num_particles: int,
    num_steps: int,
    warmup_steps: int = 10,
    solver: str = "WCSPH",
    world_half_size: float = 1.0,
    verbose: bool = True,
    use_neighbor_list: bool = False,
    use_fused_nl: bool = False,
):
    """Run benchmark and return results dict."""
    import cupy
    import numpy as np
    from world import World
    from simulation import Simulation
    from solver_profiles import PROFILES
    import hash_sort
    import step1

    # Estimate max_particles needed (spawn may produce slightly different count)
    max_p = int(num_particles * 1.2) + 10000
    # Ensure world is big enough for 1M particles
    if num_particles >= 500_000:
        world_half_size = max(world_half_size, 2.0)
    if num_particles >= 1_000_000:
        world_half_size = max(world_half_size, 3.0)

    world = World(max_particles=max_p)
    actual = create_dam_break(world, num_particles)

    if verbose:
        mode = "fused-nl" if use_fused_nl else ("neighbor-list" if use_neighbor_list else "grid")
        print(f"  Spawned: {actual:,} particles (target: {num_particles:,}) [{mode} mode]")

    # Create simulation with fixed dt for reproducibility
    sim = Simulation(
        world, dt=0.001, speed=1.0, accuracy=0.4,
        fixed_dt=True, max_substeps=1,
        world_half_size=world_half_size,
    )

    # Set solver profile
    if solver in PROFILES:
        sim.set_solver_profile(PROFILES[solver])
        sim.reset_spawn_damping()
        # Skip spawn damping
        sim._spawn_substep = sim._damping_duration

    n = world._high_water

    # Warmup: compile kernels, fill caches, upload constants
    if verbose:
        print(f"  Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        sim._sim_step(n)
    cupy.cuda.Device().synchronize()

    # Neighbor list setup (after warmup so constants are uploaded)
    nl_indices = None
    nl_count = None
    MAX_NB = 64
    if use_neighbor_list or use_fused_nl:
        nl_indices = cupy.empty(max_p * MAX_NB, dtype=cupy.uint32)
        nl_count = cupy.empty(max_p, dtype=cupy.uint32)

    if use_neighbor_list:
        import neighbor_list as nl
        import step2 as step2_mod
        # Build and upload constant memory to neighbor_list module
        hs = sim._world_half_size
        wmin, wmax = (-hs, -hs, -hs), (hs, hs, hs)
        grid_params, _ = hash_sort.build_grid_params_for_world(
            wmin, wmax, sim._h, num_particles=n,
        )
        vel_limit = 0.9 * sim._h / max(sim.dt, 1e-8)
        sim_params = step1.build_sim_params(
            smoothing_length=sim._h, particle_mass=0.02, particle_spacing=0.02,
            gravity=(0.0, -4.0, 0.0), dt=sim.dt, restitution=0.3,
            wall_friction=0.5, world_min=wmin, world_max=wmax,
            velocity_limit=vel_limit,
        )
        precalc_params = step1.build_precalc_params(smoothing_length=0.04, viscosity=1.0)
        from materials import build_material_array, build_interaction_matrix
        nl.upload_grid_params(grid_params)
        nl.upload_sim_params(sim_params)
        nl.upload_precalc_params(precalc_params)
        nl.upload_materials(build_material_array())
        nl.upload_interactions(build_interaction_matrix())
        nl.upload_granular_params(step2_mod.build_granular_params())
        # Compile NL kernels (warmup)
        nl.build_neighbor_list(
            world.sorted_position[:n], sim._cell_start, sim._cell_end,
            nl_indices, nl_count,
        )
        cupy.cuda.Device().synchronize()
    elif use_fused_nl:
        import neighbor_list as nl
        import step2 as step2_mod
        # Upload constant memory to neighbor_list module for K_Step2_NL
        hs = sim._world_half_size
        wmin, wmax = (-hs, -hs, -hs), (hs, hs, hs)
        grid_params, _ = hash_sort.build_grid_params_for_world(
            wmin, wmax, sim._h, num_particles=n,
        )
        vel_limit = 0.9 * sim._h / max(sim.dt, 1e-8)
        sim_params = step1.build_sim_params(
            smoothing_length=sim._h, particle_mass=0.02, particle_spacing=0.02,
            gravity=(0.0, -4.0, 0.0), dt=sim.dt, restitution=0.3,
            wall_friction=0.5, world_min=wmin, world_max=wmax,
            velocity_limit=vel_limit,
        )
        precalc_params = step1.build_precalc_params(smoothing_length=0.04, viscosity=1.0)
        from materials import build_material_array, build_interaction_matrix
        nl.upload_grid_params(grid_params)
        nl.upload_sim_params(sim_params)
        nl.upload_precalc_params(precalc_params)
        nl.upload_materials(build_material_array())
        nl.upload_interactions(build_interaction_matrix())
        nl.upload_granular_params(step2_mod.build_granular_params())
        # Warmup fused kernel
        from step1 import compute_step1_build_nl
        compute_step1_build_nl(
            world.sorted_position[:n], world.sorted_velocity[:n],
            world.sorted_mass[:n], world.sorted_density,
            world.sorted_packed_info[:n], world.sorted_temperature[:n],
            sim._cell_start, sim._cell_end,
            nl_indices, nl_count, MAX_NB,
        )
        cupy.cuda.Device().synchronize()

    # Timed run: per-kernel event timing
    if verbose:
        print(f"  Running {num_steps} timed steps...")

    all_timings = []

    for step in range(num_steps):
        timer = BenchmarkTimer()
        timer.mark("start")

        # -- Grid setup (counting sort or gather) --
        sim._frame_counter_d.fill(sim._substep_counter)
        world._density_initialized = True

        sim._run_grid_setup(n)
        timer.mark("grid_setup")

        if use_fused_nl:
            # -- Step1 + build NL (fused) --
            from step1 import compute_step1_build_nl
            compute_step1_build_nl(
                world.sorted_position[:n], world.sorted_velocity[:n],
                world.sorted_mass[:n],
                world.sorted_density if hasattr(world, '_density_initialized') else None,
                world.sorted_packed_info[:n], world.sorted_temperature[:n],
                sim._cell_start, sim._cell_end,
                nl_indices, nl_count, MAX_NB,
                density_out=world.sorted_density,
                shear_rate_out=world.sorted_shear_rate,
                dTdt_out=world.sorted_dTdt,
                exposure_heat_out=world.sorted_exposure_heat,
                exposure_corrode_out=world.sorted_exposure_corrode,
                vorticity_out=world.sorted_vorticity,
                normal_out=world.sorted_normal,
                particle_dye_in=world.sorted_particle_dye[:n],
                dye_rate_out=world.sorted_dye_rate,
                velocity_h=world.sorted_velocity_h,
                pressure_out=world.sorted_pressure,
                temperature_h=world.sorted_temperature_h,
                dye_h=world.sorted_dye_h,
            )
            timer.mark("step1+nl")

            # -- Reactions + Spawn --
            sim._run_reactions_spawn(n)
            timer.mark("reactions_spawn")

            # -- Step2 (neighbor-list) --
            rbm = sim.rigid_body_manager
            nl.compute_step2_nl(
                world.sorted_position[:n], world.sorted_velocity[:n],
                world.sorted_mass[:n], world.sorted_packed_info[:n],
                world.sorted_shear_rate[:n],
                nl_indices, nl_count,
                vorticity_in=world.sorted_vorticity,
                normal_in=world.sorted_normal,
                sph_force_out=world.sorted_sph_force,
                veleval_out=world.sorted_veleval,
                velocity_h=world.sorted_velocity_h,
                pressure_in=world.sorted_pressure,
                d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
                d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
                d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            )
            timer.mark("step2")
        elif use_neighbor_list:
            # -- Build neighbor list --
            nl.build_neighbor_list(
                world.sorted_position[:n],
                sim._cell_start, sim._cell_end,
                nl_indices, nl_count,
            )
            timer.mark("build_nl")

            # -- Step1 (neighbor-list) --
            nl.compute_step1_nl(
                world.sorted_position[:n], world.sorted_velocity[:n],
                world.sorted_mass[:n],
                world.sorted_density if hasattr(world, '_density_initialized') else None,
                world.sorted_packed_info[:n], world.sorted_temperature[:n],
                nl_indices, nl_count,
                density_out=world.sorted_density,
                shear_rate_out=world.sorted_shear_rate,
                dTdt_out=world.sorted_dTdt,
                exposure_heat_out=world.sorted_exposure_heat,
                exposure_corrode_out=world.sorted_exposure_corrode,
                vorticity_out=world.sorted_vorticity,
                normal_out=world.sorted_normal,
                particle_dye_in=world.sorted_particle_dye[:n],
                dye_rate_out=world.sorted_dye_rate,
                velocity_h=world.sorted_velocity_h,
                pressure_out=world.sorted_pressure,
                temperature_h=world.sorted_temperature_h,
                dye_h=world.sorted_dye_h,
            )
            timer.mark("step1")

            # -- Reactions + Spawn --
            sim._run_reactions_spawn(n)
            timer.mark("reactions_spawn")

            # -- Step2 (neighbor-list) --
            rbm = sim.rigid_body_manager
            nl.compute_step2_nl(
                world.sorted_position[:n], world.sorted_velocity[:n],
                world.sorted_mass[:n], world.sorted_packed_info[:n],
                world.sorted_shear_rate[:n],
                nl_indices, nl_count,
                vorticity_in=world.sorted_vorticity,
                normal_in=world.sorted_normal,
                sph_force_out=world.sorted_sph_force,
                veleval_out=world.sorted_veleval,
                velocity_h=world.sorted_velocity_h,
                pressure_in=world.sorted_pressure,
                d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
                d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
                d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            )
            timer.mark("step2")
        else:
            # -- Step1 (grid) --
            from step1 import compute_step1
            compute_step1(
                world.sorted_position[:n], world.sorted_velocity[:n],
                world.sorted_mass[:n],
                world.sorted_density if hasattr(world, '_density_initialized') else None,
                world.sorted_packed_info[:n], world.sorted_temperature[:n],
                sim._cell_start, sim._cell_end,
                density_out=world.sorted_density,
                shear_rate_out=world.sorted_shear_rate,
                dTdt_out=world.sorted_dTdt,
                exposure_heat_out=world.sorted_exposure_heat,
                exposure_corrode_out=world.sorted_exposure_corrode,
                vorticity_out=world.sorted_vorticity,
                normal_out=world.sorted_normal,
                particle_dye_in=world.sorted_particle_dye[:n],
                dye_rate_out=world.sorted_dye_rate,
                velocity_h=world.sorted_velocity_h,
                pressure_out=world.sorted_pressure,
                temperature_h=world.sorted_temperature_h,
                dye_h=world.sorted_dye_h,
            )
            timer.mark("step1")

            # -- Reactions + Spawn --
            sim._run_reactions_spawn(n)
            timer.mark("reactions_spawn")

            # -- Step2 (grid) --
            from step2 import compute_step2
            rbm = sim.rigid_body_manager
            compute_step2(
                world.sorted_position[:n], world.sorted_velocity[:n],
                world.sorted_mass[:n], world.sorted_packed_info[:n],
                world.sorted_shear_rate[:n],
                sim._cell_start, sim._cell_end,
                vorticity_in=world.sorted_vorticity,
                normal_in=world.sorted_normal,
                sph_force_out=world.sorted_sph_force,
                veleval_out=world.sorted_veleval,
                velocity_h=world.sorted_velocity_h,
                pressure_in=world.sorted_pressure,
                d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
                d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
                d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            )
            timer.mark("step2")

        # -- Integrate --
        from integrate import integrate as integrate_fn
        integrate_fn(
            world.sorted_position[:n], world.sorted_velocity[:n],
            world.sorted_veleval[:n], world.sorted_sph_force[:n],
            world.sorted_mass[:n], world.sorted_packed_info[:n],
            world.sorted_temperature[:n], world.sorted_health[:n],
            sorted_density=world.sorted_density[:n],
            sorted_shear_rate=world.sorted_shear_rate[:n],
            sorted_dTdt=world.sorted_dTdt[:n],
            sorted_sleep_counter=world.sorted_sleep_counter[:n],
            sorted_dye_rate=world.sorted_dye_rate[:n],
            sorted_particle_dye=world.sorted_particle_dye[:n],
            sorted_vorticity=world.sorted_vorticity[:n],
            sorted_angular_velocity=world.sorted_angular_velocity[:n],
            sort_indexes=sim._sort_perm[:n],
            position_out=world.position, velocity_out=world.velocity,
            color_out=world.color, packed_info_out=world.packed_info,
            sleep_counter_out=world.sleep_counter,
            temperature_out=world.temperature,
            particle_dye_out=world.particle_dye,
            angular_velocity_out=world.angular_velocity,
            max_displacement=world.max_displacement,
        )
        timer.mark("integrate")

        # -- Wake --
        from wake import run_wake_propagation
        run_wake_propagation(
            world.position[:n], world.velocity[:n], world.packed_info[:n],
            world.sleep_counter[:n], sim._cell_wake_flags, num_particles=n,
        )
        timer.mark("wake")

        timings = timer.results_ms()
        all_timings.append(timings)

        sim.sim_time += sim.dt
        sim._substep_counter += 1

        # Force full sort every step for consistent benchmarking
        sim._sort_skip_next = False
        sim._sort_skip_consecutive = 0

    # Aggregate results
    stages = list(all_timings[0].keys())
    mode_str = "fused-nl" if use_fused_nl else ("neighbor-list" if use_neighbor_list else "grid")
    results = {"num_particles": actual, "num_steps": num_steps, "solver": solver,
                "mode": mode_str}

    total_ms = 0.0
    for stage in stages:
        values = [t[stage] for t in all_timings]
        avg = sum(values) / len(values)
        results[f"{stage}_avg_ms"] = avg
        results[f"{stage}_min_ms"] = min(values)
        results[f"{stage}_max_ms"] = max(values)
        total_ms += avg

    results["total_substep_avg_ms"] = total_ms
    results["substeps_per_sec"] = 1000.0 / total_ms if total_ms > 0 else 0
    results["particles_per_sec"] = actual * results["substeps_per_sec"]

    # Estimate neighbor interactions (avg ~50 neighbors per particle)
    avg_neighbors = 50
    results["interactions_per_sec"] = actual * avg_neighbors * results["substeps_per_sec"]

    # Memory usage
    mem_info = cupy.cuda.runtime.memGetInfo()
    results["vram_free_mb"] = mem_info[0] / (1024 * 1024)
    results["vram_total_mb"] = mem_info[1] / (1024 * 1024)
    results["vram_used_mb"] = results["vram_total_mb"] - results["vram_free_mb"]

    # Estimate per-particle memory (total GPU memory / particles is rough)
    # More precise: count world arrays
    per_particle_bytes = estimate_per_particle_bytes(world)
    results["per_particle_bytes"] = per_particle_bytes
    results["particle_data_mb"] = actual * per_particle_bytes / (1024 * 1024)

    return results


def estimate_per_particle_bytes(world) -> int:
    """Count bytes per particle across all allocated arrays."""
    import cupy as cp

    total = 0
    # Count all arrays that scale with max_particles
    for attr_name in dir(world):
        if attr_name.startswith('_'):
            continue
        attr = getattr(world, attr_name)
        if isinstance(attr, cp.ndarray) and len(attr.shape) >= 1:
            if attr.shape[0] == world.max_particles:
                bytes_per = attr.dtype.itemsize
                if len(attr.shape) > 1:
                    bytes_per *= attr.shape[1]
                total += bytes_per

    return total


def print_results(results: dict, compact: bool = False):
    """Pretty-print benchmark results."""
    n = results["num_particles"]
    solver = results["solver"]

    if compact:
        # One-line summary for sweep mode
        total = results["total_substep_avg_ms"]
        pps = results["particles_per_sec"]
        print(f"  {n:>10,} particles | {total:7.2f} ms/step | "
              f"{pps/1e6:7.2f}M particles/s | "
              f"{results['vram_used_mb']:.0f} MB VRAM")
        return

    mode = results.get("mode", "grid")
    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS: {n:,} particles, {solver} solver [{mode}]")
    print(f"{'='*70}")

    # Per-stage breakdown
    stages = [k.replace("_avg_ms", "") for k in results if k.endswith("_avg_ms")
              and k != "total_substep_avg_ms"]

    print(f"\n  {'Stage':<20} {'Avg (ms)':>10} {'Min':>10} {'Max':>10} {'%':>8}")
    print(f"  {'-'*58}")

    total = results["total_substep_avg_ms"]
    for stage in stages:
        avg = results[f"{stage}_avg_ms"]
        mn = results[f"{stage}_min_ms"]
        mx = results[f"{stage}_max_ms"]
        pct = 100.0 * avg / total if total > 0 else 0
        print(f"  {stage:<20} {avg:10.3f} {mn:10.3f} {mx:10.3f} {pct:7.1f}%")

    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<20} {total:10.3f} ms/substep")

    # Throughput
    print(f"\n  Throughput:")
    print(f"    Substeps/sec:     {results['substeps_per_sec']:,.0f}")
    print(f"    Particles/sec:    {results['particles_per_sec']/1e6:,.2f} M")
    print(f"    Interactions/sec: {results['interactions_per_sec']/1e9:,.2f} G (est.)")

    # Memory
    print(f"\n  Memory:")
    print(f"    Per-particle:     {results['per_particle_bytes']} bytes")
    print(f"    Particle data:    {results['particle_data_mb']:.1f} MB")
    print(f"    VRAM used:        {results['vram_used_mb']:.0f} MB / {results['vram_total_mb']:.0f} MB")

    # Target assessment
    target_ms = 16.67  # 60 FPS
    substeps_at_60fps = target_ms / total if total > 0 else 0
    print(f"\n  At 60 FPS ({target_ms:.1f}ms budget):")
    print(f"    Substeps/frame:   {substeps_at_60fps:.1f}")
    print(f"    Speedup needed:   {total / target_ms:.1f}x" if total > target_ms else
          f"    Headroom:         {target_ms / total:.1f}x")

    print(f"{'='*70}\n")


def run_sweep(steps: int = 50, solver: str = "WCSPH"):
    """Run benchmarks at multiple particle counts."""
    counts = [100_000, 250_000, 500_000, 1_000_000]

    print(f"\n{'='*70}")
    print(f"  PERFORMANCE SWEEP ({solver} solver, {steps} steps each)")
    print(f"{'='*70}\n")

    all_results = []
    for count in counts:
        print(f"  Benchmarking {count:,} particles...")
        try:
            results = run_benchmark(
                num_particles=count,
                num_steps=steps,
                warmup_steps=5,
                solver=solver,
                verbose=False,
            )
            print_results(results, compact=True)
            all_results.append(results)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append(None)

    # Scaling analysis
    print(f"\n  Scaling Analysis:")
    print(f"  {'N':>12} {'ms/step':>10} {'ms/N':>12} {'scaling':>10}")
    print(f"  {'-'*46}")
    base_ratio = None
    for r in all_results:
        if r is None:
            continue
        n = r["num_particles"]
        ms = r["total_substep_avg_ms"]
        ratio = ms / n * 1000  # us per particle
        if base_ratio is None:
            base_ratio = ratio
            scale = "1.00x"
        else:
            scale = f"{ratio / base_ratio:.2f}x"
        print(f"  {n:>12,} {ms:>10.2f} {ratio:>10.2f}us {scale:>10}")

    print()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="SPH Performance Benchmark")
    parser.add_argument("--particles", "-n", type=int, default=100_000,
                        help="Number of particles (default: 100000)")
    parser.add_argument("--steps", "-s", type=int, default=50,
                        help="Number of timed substeps (default: 50)")
    parser.add_argument("--warmup", "-w", type=int, default=10,
                        help="Warmup substeps (default: 10)")
    parser.add_argument("--solver", type=str, default="WCSPH",
                        choices=["WCSPH", "PBF", "DFSPH"],
                        help="Solver to benchmark (default: WCSPH)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run scaling sweep (100K to 1M)")
    parser.add_argument("--profile", action="store_true",
                        help="Single profiling run (fewer steps, no warmup)")
    parser.add_argument("--world-size", type=float, default=1.0,
                        help="World half-size (default: 1.0, auto-scaled for large N)")
    parser.add_argument("--neighbor-list", "--nl", action="store_true",
                        help="Use neighbor-list kernels instead of grid-based")
    parser.add_argument("--fused-nl", action="store_true",
                        help="Use fused step1+NL build, step2 reads NL (best of both)")
    parser.add_argument("--compare", action="store_true",
                        help="Run grid, neighbor-list, and fused-nl side by side")
    args = parser.parse_args()

    _apply_ptx_workaround()

    print(f"SPH Performance Benchmark")
    print(f"  GPU: ", end="")

    import cupy
    dev = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(dev.id)
    print(f"{props['name'].decode()} ({props['totalGlobalMem'] // (1024**2)} MB)")
    print(f"  Compute: sm_{props['major']}{props['minor']}")

    if args.compare:
        # A/B comparison: grid vs neighbor-list
        print(f"\n{'='*70}")
        print(f"  A/B COMPARISON: grid vs neighbor-list ({args.particles:,} particles)")
        print(f"{'='*70}\n")

        print("  [A] Grid-based kernels...")
        r_grid = run_benchmark(
            num_particles=args.particles, num_steps=args.steps,
            warmup_steps=args.warmup, solver=args.solver,
            world_half_size=args.world_size, use_neighbor_list=False,
        )
        print_results(r_grid)

        print("  [B] Fused step1+NL build, step2 reads NL...")
        r_fused = run_benchmark(
            num_particles=args.particles, num_steps=args.steps,
            warmup_steps=args.warmup, solver=args.solver,
            world_half_size=args.world_size, use_fused_nl=True,
        )
        print_results(r_fused)

        # Summary comparison
        print(f"\n{'='*70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*70}")
        t_grid = r_grid["total_substep_avg_ms"]
        t_fused = r_fused["total_substep_avg_ms"]
        speedup = t_grid / t_fused if t_fused > 0 else 0
        print(f"  Grid total:           {t_grid:.3f} ms/substep")
        print(f"  Fused-NL total:       {t_fused:.3f} ms/substep")
        print(f"  Speedup:              {speedup:.2f}x")

        # Per-stage comparison
        stages_grid = {k.replace("_avg_ms", ""): v for k, v in r_grid.items() if k.endswith("_avg_ms") and k != "total_substep_avg_ms"}
        stages_fused = {k.replace("_avg_ms", ""): v for k, v in r_fused.items() if k.endswith("_avg_ms") and k != "total_substep_avg_ms"}
        all_stages = list(dict.fromkeys(list(stages_grid.keys()) + list(stages_fused.keys())))
        print(f"\n  {'Stage':<20} {'Grid (ms)':>10} {'Fused (ms)':>10} {'Speedup':>10}")
        print(f"  {'-'*52}")
        for stage in all_stages:
            g = stages_grid.get(stage, 0)
            f = stages_fused.get(stage, 0)
            sp = g / f if f > 0 else float('inf') if g > 0 else 1.0
            sp_str = f"{sp:.2f}x" if sp < 100 else "N/A"
            print(f"  {stage:<20} {g:10.3f} {f:10.3f} {sp_str:>10}")
        print(f"{'='*70}\n")

    elif args.sweep:
        run_sweep(steps=args.steps, solver=args.solver)
    elif args.profile:
        results = run_benchmark(
            num_particles=args.particles,
            num_steps=10,
            warmup_steps=2,
            solver=args.solver,
            world_half_size=args.world_size,
            use_neighbor_list=args.neighbor_list,
            use_fused_nl=args.fused_nl,
        )
        print_results(results)
    else:
        results = run_benchmark(
            num_particles=args.particles,
            num_steps=args.steps,
            warmup_steps=args.warmup,
            solver=args.solver,
            world_half_size=args.world_size,
            use_neighbor_list=args.neighbor_list,
            use_fused_nl=args.fused_nl,
        )
        print_results(results)


if __name__ == "__main__":
    main()
