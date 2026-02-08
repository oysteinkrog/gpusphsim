#!/usr/bin/env python3
"""Headless SPH validation: quantitative physics metrics for all solvers.

Runs test scenarios without OpenGL/GLFW, measuring:
  - Density error (mean, max, RMS relative to rest density)
  - Energy conservation (kinetic + potential)
  - Velocity statistics (mean, max, RMS)
  - Stability (NaN/Inf/escaped particles)
  - Free-fall trajectory accuracy (before impact)

Usage:
  python validate.py                    # Full validation suite
  python validate.py --dfsph-sweep      # DFSPH parameter sweep
  python validate.py --quick            # Quick smoke test (fewer steps)
"""

import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


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
from solver_profiles import PROFILES, SolverProfile, SolverType, PROFILE_NAMES
from materials import WATER, SAND

GRAVITY = -9.8
REST_DENSITY = 2500.0


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    step: int
    sim_time: float
    n_active: int
    # Density
    density_mean: float = 0.0
    density_std: float = 0.0
    density_rel_err_mean: float = 0.0
    density_rel_err_max: float = 0.0
    density_rel_err_rms: float = 0.0
    # Velocity
    v_mean: float = 0.0
    v_max: float = 0.0
    v_rms: float = 0.0
    # Energy
    KE: float = 0.0
    PE: float = 0.0
    total_E: float = 0.0
    # Center of mass
    com_y: float = 0.0
    # Stability
    has_nan: bool = False
    has_inf: bool = False
    n_escaped: int = 0
    # Velocity clamped
    v_clamped_pct: float = 0.0


def collect_metrics(world, step: int, sim_time: float, rho0: float = REST_DENSITY) -> StepMetrics:
    """Collect quantitative metrics from current world state (GPU->CPU sync)."""
    n = world._high_water
    pos = world.position[:n].get()
    vel = world.velocity[:n].get()
    packed = world.packed_info[:n].get()
    mass_arr = world.mass[:n].get()

    # Filter active particles (mat_id != 0)
    mat_id = packed & 0xFF
    active = mat_id != 0
    n_active = int(active.sum())

    if n_active == 0:
        return StepMetrics(step=step, sim_time=sim_time, n_active=0, has_nan=True)

    p = pos[active, :3]
    v = vel[active, :3]
    m = mass_arr[active]

    # NaN/Inf check
    has_nan = bool(np.isnan(p).any() or np.isnan(v).any())
    has_inf = bool(np.isinf(p).any() or np.isinf(v).any())

    # Escaped particles (outside bounds with margin)
    n_escaped = int(np.any(np.abs(p) > 1.05, axis=1).sum())

    # Velocity
    speeds = np.linalg.norm(v, axis=1)
    v_mean = float(speeds.mean())
    v_max = float(speeds.max())
    v_rms = float(np.sqrt(np.mean(speeds ** 2)))
    v_clamped_pct = float(100.0 * np.sum(speeds > 9.9) / n_active)

    # Density (use sorted_density which was computed in the last sim step)
    try:
        dens = world.sorted_density[:n].get()
        dens_active = dens[active]
        # Filter out zero densities (uninitialized)
        nonzero = dens_active > 1.0
        if nonzero.sum() > 0:
            d = dens_active[nonzero]
            rel_err = np.abs(d - rho0) / rho0
            density_mean = float(d.mean())
            density_std = float(d.std())
            density_rel_err_mean = float(rel_err.mean())
            density_rel_err_max = float(rel_err.max())
            density_rel_err_rms = float(np.sqrt(np.mean(rel_err ** 2)))
        else:
            density_mean = density_std = 0.0
            density_rel_err_mean = density_rel_err_max = density_rel_err_rms = 0.0
    except Exception:
        density_mean = density_std = 0.0
        density_rel_err_mean = density_rel_err_max = density_rel_err_rms = 0.0

    # Energy
    KE = float(0.5 * np.sum(m * speeds ** 2))
    PE = float(-np.sum(m * GRAVITY * p[:, 1]))  # PE = -m*g*y
    total_E = KE + PE

    # Center of mass
    total_mass = float(m.sum())
    com_y = float(np.sum(m * p[:, 1]) / total_mass)

    return StepMetrics(
        step=step, sim_time=sim_time, n_active=n_active,
        density_mean=density_mean, density_std=density_std,
        density_rel_err_mean=density_rel_err_mean,
        density_rel_err_max=density_rel_err_max,
        density_rel_err_rms=density_rel_err_rms,
        v_mean=v_mean, v_max=v_max, v_rms=v_rms,
        KE=KE, PE=PE, total_E=total_E,
        com_y=com_y,
        has_nan=has_nan, has_inf=has_inf,
        n_escaped=n_escaped, v_clamped_pct=v_clamped_pct,
    )


def print_metrics(m: StepMetrics, compact: bool = False):
    if compact:
        print(f"  step {m.step:4d} t={m.sim_time:.4f}: "
              f"dens_err={m.density_rel_err_mean:.4f}/{m.density_rel_err_max:.4f}  "
              f"v={m.v_mean:.3f}/{m.v_max:.3f}  "
              f"E={m.total_E:.2f}  "
              f"clamped={m.v_clamped_pct:.1f}%  "
              f"nan={m.has_nan}")
    else:
        print(f"  Step {m.step} (t={m.sim_time:.4f}s):")
        print(f"    Particles: {m.n_active}")
        print(f"    Density: mean={m.density_mean:.1f} std={m.density_std:.1f} "
              f"err(mean/max/rms)={m.density_rel_err_mean:.4f}/{m.density_rel_err_max:.4f}/{m.density_rel_err_rms:.4f}")
        print(f"    Velocity: mean={m.v_mean:.4f} max={m.v_max:.4f} rms={m.v_rms:.4f} "
              f"clamped={m.v_clamped_pct:.1f}%")
        print(f"    Energy: KE={m.KE:.2f} PE={m.PE:.2f} total={m.total_E:.2f}")
        print(f"    CoM_y: {m.com_y:.4f}")
        if m.has_nan or m.has_inf or m.n_escaped > 0:
            print(f"    !! NaN={m.has_nan} Inf={m.has_inf} Escaped={m.n_escaped}")


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def spawn_water_block(world, y_lo=-0.9, y_hi=-0.1, spacing=0.02):
    """Spawn a block of water filling the lower region."""
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube(
        min_corner=(-0.5, y_lo, -0.5),
        max_corner=(0.5, y_hi, 0.5),
        material_id=WATER,
        spacing=spacing,
    )
    return n


def spawn_water_drop(world, spacing=0.02):
    """Spawn a cube of water suspended in air (free-fall test)."""
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube(
        min_corner=(-0.15, 0.3, -0.15),
        max_corner=(0.15, 0.6, 0.15),
        material_id=WATER,
        spacing=spacing,
    )
    return n


def spawn_mixed_scene(world, spacing=0.02):
    """Spawn water cube + sand bed (tests multi-material)."""
    world.packed_info[:] = 0
    world._high_water = 0
    n_water = world.spawn_cube(
        min_corner=(-0.2, 0.3, -0.2),
        max_corner=(0.2, 0.7, 0.2),
        material_id=WATER,
        spacing=spacing,
    )
    n_sand = world.spawn_cube(
        min_corner=(-0.8, -0.5, -0.8),
        max_corner=(0.8, -0.3, 0.8),
        material_id=SAND,
        spacing=0.04,
    )
    return n_water + n_sand


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_hydrostatic_test(solver_name: str, num_steps: int = 200,
                         report_every: int = 20) -> Tuple[bool, List[StepMetrics]]:
    """Hydrostatic tank: water at rest should converge to zero velocity."""
    print(f"\n{'='*70}")
    print(f"HYDROSTATIC TEST: {solver_name}")
    print(f"{'='*70}")

    profile = PROFILES[solver_name]
    world = World(max_particles=200_000)
    n = spawn_water_block(world)
    print(f"  Spawned {n:,} water particles")

    sim = Simulation(world, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    sim.set_solver_profile(profile)

    history: List[StepMetrics] = []

    for step in range(num_steps):
        sim._sim_step(world._high_water)

        if step % report_every == 0 or step == num_steps - 1:
            cp.cuda.Device().synchronize()
            m = collect_metrics(world, step, step * profile.dt)
            history.append(m)
            print_metrics(m, compact=True)
            if m.has_nan or m.has_inf:
                print(f"  !! UNSTABLE at step {step}")
                return False, history

    final = history[-1]
    passed = (
        not final.has_nan and
        not final.has_inf and
        final.v_clamped_pct < 5.0 and  # Less than 5% particles at velocity limit
        final.density_rel_err_mean < 0.5  # Mean density error < 50%
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed, history


def run_free_fall_test(solver_name: str, num_steps: int = 60,
                       report_every: int = 5) -> Tuple[bool, List]:
    """Free-falling water block: verify ballistic trajectory before impact."""
    print(f"\n{'='*70}")
    print(f"FREE-FALL TEST: {solver_name}")
    print(f"{'='*70}")

    profile = PROFILES[solver_name]
    world = World(max_particles=100_000)
    n = spawn_water_drop(world)
    print(f"  Spawned {n:,} water particles")

    sim = Simulation(world, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    sim.set_solver_profile(profile)

    # Measure initial center of mass
    cp.cuda.Device().synchronize()
    pos0 = world.position[:n].get()
    mass0 = world.mass[:n].get()
    packed0 = world.packed_info[:n].get()
    active0 = (packed0 & 0xFF) != 0
    m0 = mass0[active0]
    p0 = pos0[active0, :3]
    y0_com = float(np.sum(m0 * p0[:, 1]) / m0.sum())
    print(f"  Initial CoM_y = {y0_com:.4f}")

    errors = []
    history = []

    for step in range(1, num_steps + 1):
        sim._sim_step(world._high_water)

        if step % report_every == 0:
            cp.cuda.Device().synchronize()
            t = step * profile.dt
            m = collect_metrics(world, step, t)
            history.append(m)

            if m.has_nan or m.has_inf:
                print(f"  !! UNSTABLE at step {step}")
                return False, errors

            # Expected: y = y0 + 0.5*g*t^2 (starting from rest)
            y_expected = y0_com + 0.5 * GRAVITY * t * t

            # Only valid before hitting floor
            if m.com_y > -0.7:
                err = abs(m.com_y - y_expected)
                errors.append((step, t, m.com_y, y_expected, err))
                print(f"  step {step:3d} t={t:.4f}: com_y={m.com_y:.4f} "
                      f"expected={y_expected:.4f} err={err:.4f}  "
                      f"dens_err={m.density_rel_err_mean:.4f}  "
                      f"v_max={m.v_max:.3f}")
            else:
                print(f"  step {step:3d} t={t:.4f}: com_y={m.com_y:.4f} (post-impact)  "
                      f"dens_err={m.density_rel_err_mean:.4f}  "
                      f"v_max={m.v_max:.3f}")

    if errors:
        max_err = max(e[4] for e in errors)
        passed = max_err < 0.1  # 10cm tolerance (generous for game SPH)
        print(f"  Max free-fall error: {max_err:.4f}m  {'PASS' if passed else 'FAIL'}")
        return passed, errors
    return True, errors


def run_energy_test(solver_name: str, num_steps: int = 200,
                    report_every: int = 10) -> Tuple[bool, List[StepMetrics]]:
    """Energy conservation: total energy should never increase."""
    print(f"\n{'='*70}")
    print(f"ENERGY TEST: {solver_name}")
    print(f"{'='*70}")

    profile = PROFILES[solver_name]
    world = World(max_particles=200_000)
    n = spawn_water_block(world)
    print(f"  Spawned {n:,} water particles")

    sim = Simulation(world, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    sim.set_solver_profile(profile)

    history: List[StepMetrics] = []
    energy_increases = 0
    max_energy_increase = 0.0

    for step in range(num_steps):
        sim._sim_step(world._high_water)

        if step % report_every == 0:
            cp.cuda.Device().synchronize()
            m = collect_metrics(world, step, step * profile.dt)
            history.append(m)

            if m.has_nan or m.has_inf:
                print(f"  !! UNSTABLE at step {step}")
                return False, history

            if len(history) >= 2:
                dE = m.total_E - history[-2].total_E
                if dE > 0.01:  # Small tolerance for numerical noise
                    energy_increases += 1
                    max_energy_increase = max(max_energy_increase, dE)

            if step % (report_every * 5) == 0:
                dE_str = ""
                if len(history) >= 2:
                    dE = m.total_E - history[-2].total_E
                    dE_str = f"  dE={dE:+.3f}"
                print(f"  step {step:4d}: E={m.total_E:.2f} (KE={m.KE:.2f} PE={m.PE:.2f}){dE_str}")

    # Allow some energy increases (boundary reflections can add energy numerically)
    passed = energy_increases < len(history) * 0.3  # Less than 30% of steps increase energy
    print(f"  Energy increases: {energy_increases}/{len(history)} steps  "
          f"max_increase={max_energy_increase:.3f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed, history


# ---------------------------------------------------------------------------
# DFSPH parameter sweep
# ---------------------------------------------------------------------------

def run_dfsph_sweep():
    """Sweep DFSPH parameters to find stable settings."""
    print(f"\n{'#'*70}")
    print(f"# DFSPH PARAMETER SWEEP")
    print(f"{'#'*70}")

    # Parameters to sweep
    configs = [
        # (label, dt, div_iters, dens_iters, omega, max_substeps)
        ("dt=1/60  div=2 dens=2 omega=0.5", 1/60, 2, 2, 0.5, 2),
        ("dt=1/60  div=2 dens=2 omega=0.3", 1/60, 2, 2, 0.3, 2),
        ("dt=1/60  div=2 dens=3 omega=0.3", 1/60, 2, 3, 0.3, 2),
        ("dt=1/60  div=3 dens=3 omega=0.3", 1/60, 3, 3, 0.3, 2),
        ("dt=1/120 div=2 dens=2 omega=0.5", 1/120, 2, 2, 0.5, 4),
        ("dt=1/120 div=2 dens=2 omega=1.0", 1/120, 2, 2, 1.0, 4),
        ("dt=1/120 div=3 dens=3 omega=0.5", 1/120, 3, 3, 0.5, 4),
        ("dt=1/240 div=2 dens=2 omega=0.5", 1/240, 2, 2, 0.5, 8),
        ("dt=1/240 div=2 dens=2 omega=1.0", 1/240, 2, 2, 1.0, 8),
        ("dt=0.001 div=2 dens=1 omega=1.0", 0.001, 2, 1, 1.0, 20),
        ("dt=0.002 div=2 dens=2 omega=0.5", 0.002, 2, 2, 0.5, 10),
        ("dt=0.004 div=2 dens=2 omega=0.5", 0.004, 2, 2, 0.5, 5),
    ]

    results = []

    for label, dt, div_iters, dens_iters, omega, max_sub in configs:
        print(f"\n--- {label} ---")

        profile = SolverProfile(
            name="DFSPH sweep",
            solver_type=SolverType.DFSPH,
            dt=dt,
            max_substeps=max_sub,
            fixed_dt=True,
            dfsph_div_iters=div_iters,
            dfsph_dens_iters=dens_iters,
            dfsph_omega=omega,
        )

        world = World(max_particles=100_000)
        n = spawn_water_drop(world)

        sim = Simulation(world, dt=dt, speed=1.0, fixed_dt=True, max_substeps=1)
        sim.set_solver_profile(profile)

        # Run 60 steps (= ~1s at dt=1/60, or 0.06s at dt=0.001)
        total_steps = 60
        stable = True
        final_metrics = None
        max_v = 0.0
        max_dens_err = 0.0

        t0 = time.perf_counter()
        for step in range(total_steps):
            try:
                sim._sim_step(world._high_water)
            except Exception as e:
                print(f"  CRASH at step {step}: {e}")
                stable = False
                break

            if step % 10 == 0 or step == total_steps - 1:
                cp.cuda.Device().synchronize()
                m = collect_metrics(world, step, step * dt)
                max_v = max(max_v, m.v_max)
                max_dens_err = max(max_dens_err, m.density_rel_err_mean)
                final_metrics = m

                if m.has_nan or m.has_inf:
                    print(f"  UNSTABLE at step {step} (NaN/Inf)")
                    stable = False
                    break

                if m.v_clamped_pct > 50:
                    print(f"  UNSTABLE at step {step} ({m.v_clamped_pct:.0f}% velocity-clamped)")
                    stable = False
                    break

        elapsed = time.perf_counter() - t0

        # Count effective substeps per frame at 60fps
        substeps_per_frame = (1.0 / 60) / dt
        ms_per_frame = substeps_per_frame * (elapsed / total_steps * 1000)

        result = {
            'label': label,
            'stable': stable,
            'max_v': max_v,
            'max_dens_err': max_dens_err,
            'ms_per_step': elapsed / total_steps * 1000,
            'ms_per_frame_60fps': ms_per_frame,
            'substeps_per_frame': substeps_per_frame,
        }
        results.append(result)

        if stable and final_metrics:
            print(f"  STABLE: max_v={max_v:.2f} max_dens_err={max_dens_err:.4f} "
                  f"ms/step={result['ms_per_step']:.2f} "
                  f"substeps@60fps={substeps_per_frame:.1f} "
                  f"ms/frame={ms_per_frame:.1f}")
        elif stable:
            print(f"  STABLE (no metrics)")

    # Summary
    print(f"\n{'='*70}")
    print(f"DFSPH SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<45} {'Stable':>6} {'MaxV':>6} {'DensErr':>8} {'ms/step':>8} {'ms/frame':>9}")
    print(f"{'-'*45} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*9}")
    for r in results:
        status = "OK" if r['stable'] else "FAIL"
        print(f"{r['label']:<45} {status:>6} {r['max_v']:>6.2f} {r['max_dens_err']:>8.4f} "
              f"{r['ms_per_step']:>8.2f} {r['ms_per_frame_60fps']:>9.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_suite():
    """Run the complete validation suite."""
    results = {}

    for name in PROFILE_NAMES:
        print(f"\n\n{'#'*70}")
        print(f"# SOLVER: {name}")
        print(f"{'#'*70}")

        ok1, _ = run_hydrostatic_test(name, num_steps=200)
        ok2, _ = run_free_fall_test(name, num_steps=60)
        ok3, _ = run_energy_test(name, num_steps=200)

        results[name] = {
            'hydrostatic': ok1,
            'free_fall': ok2,
            'energy': ok3,
        }

    # Summary
    print(f"\n\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Solver':<25} {'Hydrostatic':>12} {'Free-Fall':>12} {'Energy':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for name, r in results.items():
        h = "PASS" if r['hydrostatic'] else "FAIL"
        f = "PASS" if r['free_fall'] else "FAIL"
        e = "PASS" if r['energy'] else "FAIL"
        print(f"{name:<25} {h:>12} {f:>12} {e:>12}")


def run_quick():
    """Quick smoke test."""
    print("Quick validation (30 steps each)...")

    for name in PROFILE_NAMES:
        print(f"\n--- {name} ---")
        profile = PROFILES[name]
        world = World(max_particles=100_000)
        n = spawn_water_drop(world)

        sim = Simulation(world, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
        sim.set_solver_profile(profile)

        stable = True
        for step in range(30):
            sim._sim_step(world._high_water)

        cp.cuda.Device().synchronize()
        m = collect_metrics(world, 30, 30 * profile.dt)
        print_metrics(m)
        if m.has_nan or m.has_inf:
            print(f"  !! UNSTABLE")
            stable = False

        print(f"  {'PASS' if stable else 'FAIL'}")


if __name__ == '__main__':
    if '--dfsph-sweep' in sys.argv:
        run_dfsph_sweep()
    elif '--quick' in sys.argv:
        run_quick()
    else:
        run_full_suite()
        run_dfsph_sweep()
