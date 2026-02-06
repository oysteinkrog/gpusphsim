#!/usr/bin/env python3
"""Headless SPH benchmark: ~500K water particles, no rendering."""

import time


def _apply_ptx_workaround():
    """Force CuPy to emit PTX instead of cubin for forward-compat with newer GPUs."""
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
from materials import WATER

NUM_STEPS = 200

# 80 particles per axis at spacing 0.02 => 512,000 particles
# Cube from -0.79 to +0.79 fits inside world bounds (-1..1)
world = World(max_particles=600_000)

print("Spawning water cube...")
n = world.spawn_cube(
    min_corner=(-0.79, -0.79, -0.79),
    max_corner=(0.79, 0.79, 0.79),
    material_id=WATER,
    spacing=0.02,
)
print(f"Spawned {n:,} water particles")

print("Compiling kernels & uploading constants...")
sim = Simulation(
    world,
    dt=0.005,
    speed=1.0,
    accuracy=0.4,
    fixed_dt=False,
    max_substeps=20,
)

# Warm-up: 2 steps to fill caches / JIT
print("Warm-up (2 steps)...")
for _ in range(2):
    sim._sim_step(world._high_water)
cp.cuda.Device().synchronize()

print(f"\nBenchmarking {NUM_STEPS} substeps on {world._high_water:,} particles...")
cp.cuda.Device().synchronize()
t0 = time.perf_counter()

for i in range(NUM_STEPS):
    sim._sim_step(world._high_water)
    if (i + 1) % 50 == 0:
        cp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - t0
        rate = (i + 1) / elapsed
        print(f"  Step {i+1:4d}: {rate:.1f} steps/s  ({elapsed:.2f}s elapsed)")

cp.cuda.Device().synchronize()
total = time.perf_counter() - t0

print(f"\n--- Results ---")
print(f"Particles:  {world._high_water:,}")
print(f"Steps:      {NUM_STEPS}")
print(f"Total time: {total:.2f}s")
print(f"Rate:       {NUM_STEPS / total:.1f} steps/s")
print(f"Per step:   {total / NUM_STEPS * 1000:.1f} ms")
print(f"dt:         {sim.dt}")

# At 60fps with speed=1.0, we need ~(1/60)/dt substeps per frame
fps_target = 60
substeps_per_frame = (1.0 / fps_target) / sim.dt
ms_per_frame = substeps_per_frame * (total / NUM_STEPS * 1000)
print(f"\nAt {fps_target} FPS (speed=1.0):")
print(f"  Substeps/frame: {substeps_per_frame:.1f}")
print(f"  Sim time/frame: {ms_per_frame:.1f} ms  (budget: {1000/fps_target:.1f} ms)")
