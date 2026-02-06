#!/usr/bin/env python3
"""Profile each pipeline stage to find the actual bottlenecks."""

import time

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
import numpy as np
from world import World
from simulation import Simulation
from materials import WATER
import hash_sort
import fused_sort_reorder_build
import step1
import step2
import integrate
import reactions
import spawn
import wake

# Setup
world = World(max_particles=600_000)
n = world.spawn_cube(
    min_corner=(-0.79, -0.79, -0.79),
    max_corner=(0.79, 0.79, 0.79),
    material_id=WATER,
    spacing=0.02,
)
print(f"Particles: {n:,}")

sim = Simulation(world, dt=0.005, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20)

# Warm up
for _ in range(3):
    sim._sim_step(world._high_water)
cp.cuda.Device().synchronize()

# Profile 50 steps, timing each stage
NUM = 50
w = world
n = world._high_water

timings = {
    'hash': 0, 'argsort': 0, 'fused_srb': 0,
    'step1': 0, 'reactions': 0, 'spawn': 0, 'step2': 0,
    'integrate': 0, 'wake': 0,
}

for _ in range(NUM):
    cp.cuda.Device().synchronize()

    t0 = time.perf_counter()
    hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    timings['hash'] += t1 - t0

    sort_perm = cp.argsort(hashes).astype(cp.uint32)
    cp.cuda.Device().synchronize()
    t2 = time.perf_counter()
    timings['argsort'] += t2 - t1

    sim._cell_start.data.memset(0xFF, sim._cell_start.nbytes)
    sim._cell_end.data.memset(0x00, sim._cell_end.nbytes)
    sorted_indices = sort_perm
    fused_sort_reorder_build.fused_sort_reorder_build(
        n, sort_perm, hashes,
        w.sorted_hashes, sim._cell_start, sim._cell_end,
        w.position, w.velocity,
        w.mass, w.packed_info, w.temperature,
        w.health, w.lifetime,
        w.sleep_counter,
        w.sorted_position, w.sorted_velocity,
        w.sorted_mass, w.sorted_packed_info, w.sorted_temperature,
        w.sorted_health, w.sorted_lifetime,
        w.sorted_sleep_counter,
    )
    cp.cuda.Device().synchronize()
    t3 = time.perf_counter()
    timings['fused_srb'] += t3 - t2

    step1.compute_step1(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_mass[:n], w.sorted_density,
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        sim._cell_start, sim._cell_end,
        density_out=w.sorted_density,
        shear_rate_out=w.sorted_shear_rate,
        dTdt_out=w.sorted_dTdt,
        exposure_heat_out=w.sorted_exposure_heat,
        exposure_corrode_out=w.sorted_exposure_corrode,
    )
    cp.cuda.Device().synchronize()
    t4 = time.perf_counter()
    timings['step1'] += t4 - t3

    spawn.reset_freelist(sim._dead_count)
    sim._frame_counter_d.fill(sim._frame_counter)
    reactions.compute_reactions(
        w.sorted_packed_info[:n], w.sorted_temperature[:n],
        w.sorted_health[:n], w.sorted_lifetime[:n],
        w.sorted_velocity[:n], w.sorted_exposure_heat[:n],
        w.sorted_exposure_corrode[:n],
        frame_d=sim._frame_counter_d,
        dead_indices=sim._dead_indices, dead_count=sim._dead_count,
    )
    cp.cuda.Device().synchronize()
    t5 = time.perf_counter()
    timings['reactions'] += t5 - t4

    spawn.compute_spawn(
        w.sorted_packed_info[:n], w.sorted_position[:n],
        w.sorted_velocity[:n], w.sorted_veleval[:n],
        w.sorted_mass[:n], w.sorted_temperature[:n],
        w.sorted_health[:n], w.sorted_lifetime[:n],
        w.sorted_color[:n], w.sorted_sleep_counter[:n],
        w.sorted_density[:n], w.sorted_shear_rate[:n],
        sim._dead_indices, sim._dead_count,
    )
    cp.cuda.Device().synchronize()
    t6 = time.perf_counter()
    timings['spawn'] += t6 - t5

    step2.compute_step2(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_density[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end,
        sph_force_out=w.sorted_sph_force,
        veleval_out=w.sorted_veleval,
    )
    cp.cuda.Device().synchronize()
    t7 = time.perf_counter()
    timings['step2'] += t7 - t6

    integrate.integrate(
        w.sorted_position[:n], w.sorted_velocity[:n],
        w.sorted_veleval[:n], w.sorted_sph_force[:n],
        w.sorted_mass[:n], w.sorted_packed_info[:n],
        w.sorted_temperature[:n], w.sorted_health[:n],
        sorted_density=w.sorted_density[:n],
        sorted_shear_rate=w.sorted_shear_rate[:n],
        sorted_dTdt=w.sorted_dTdt[:n],
        sorted_sleep_counter=w.sorted_sleep_counter[:n],
        sort_indexes=sorted_indices[:n],
        position_out=w.position, velocity_out=w.velocity,
        color_out=w.color, packed_info_out=w.packed_info,
        sleep_counter_out=w.sleep_counter,
        temperature_out=w.temperature,
    )
    cp.cuda.Device().synchronize()
    t8 = time.perf_counter()
    timings['integrate'] += t8 - t7

    wake.run_wake_propagation(
        w.position[:n], w.packed_info[:n], w.sleep_counter[:n],
        sim._cell_wake_flags, num_particles=n,
    )
    cp.cuda.Device().synchronize()
    t9 = time.perf_counter()
    timings['wake'] += t9 - t8

    sim.sim_time += sim.dt
    sim._frame_counter += 1

total = sum(timings.values())
print(f"\n{'Stage':<15} {'Total (ms)':>10} {'Per Step (ms)':>13} {'%':>6}")
print("-" * 48)
for name, t in sorted(timings.items(), key=lambda x: -x[1]):
    ms_total = t * 1000
    ms_step = ms_total / NUM
    pct = t / total * 100
    print(f"{name:<15} {ms_total:>10.1f} {ms_step:>13.2f} {pct:>5.1f}%")
print("-" * 48)
print(f"{'TOTAL':<15} {total*1000:>10.1f} {total*1000/NUM:>13.2f} {'100.0':>5}%")
print(f"\nEffective rate: {NUM/total:.1f} steps/s")
