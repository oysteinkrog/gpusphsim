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
):
    """Run benchmark and return results dict."""
    import cupy
    import numpy as np
    from world import World
    from simulation import Simulation
    from solver_profiles import PROFILES

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
        print(f"  Spawned: {actual:,} particles (target: {num_particles:,})")

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
            velocity_h=world.sorted_velocity_h,
            pressure_out=world.sorted_pressure,
            temperature_h=world.sorted_temperature_h,
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
            sorted_vorticity=world.sorted_vorticity[:n],
            sorted_angular_velocity=world.sorted_angular_velocity[:n],
            sort_indexes=sim._sort_perm[:n],
            position_out=world.position, velocity_out=world.velocity,
            packed_info_out=world.packed_info,
            sleep_counter_out=world.sleep_counter,
            temperature_out=world.temperature,
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

        # Force full sort every step for consistent benchmarking. To time the
        # grid-reuse (gather-only) path as step_frame drives it, run
        # bench_substep.py --grid-reuse.
        sim._sort_skip_next = False
        sim._sort_skip_consecutive = 0

    # Aggregate results
    stages = list(all_timings[0].keys())
    results = {"num_particles": actual, "num_steps": num_steps, "solver": solver}

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

    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS: {n:,} particles, {solver} solver")
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
    args = parser.parse_args()

    _apply_ptx_workaround()

    print(f"SPH Performance Benchmark")
    print(f"  GPU: ", end="")

    import cupy
    dev = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(dev.id)
    print(f"{props['name'].decode()} ({props['totalGlobalMem'] // (1024**2)} MB)")
    print(f"  Compute: sm_{props['major']}{props['minor']}")

    if args.sweep:
        run_sweep(steps=args.steps, solver=args.solver)
    elif args.profile:
        results = run_benchmark(
            num_particles=args.particles,
            num_steps=10,
            warmup_steps=2,
            solver=args.solver,
            world_half_size=args.world_size,
        )
        print_results(results)
    else:
        results = run_benchmark(
            num_particles=args.particles,
            num_steps=args.steps,
            warmup_steps=args.warmup,
            solver=args.solver,
            world_half_size=args.world_size,
        )
        print_results(results)


if __name__ == "__main__":
    main()
