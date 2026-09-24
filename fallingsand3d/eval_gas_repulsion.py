"""Measure GAS self-repulsion under WCSPH, DFSPH and PBF (bd-r5-review-n28.91).

Spawns a permanent gas block, runs 2 s of simulated time under each
solver and reads the gas state once at the end:

  - density mean and 95th percentile (sorted_density, gas only)
  - horizontal plume radius (RMS distance from the gas centre in x, z)
  - median nearest-neighbour distance and the fraction of particles with a
    neighbour closer than 0.25 * spacing (collapse into strands or points)

Run with Windows Python from this directory:
    python eval_gas_repulsion.py          # hot STEAM, buoyancy presses it to the ceiling
    python eval_gas_repulsion.py --free   # ambient SMOKE, zero gravity, free expansion
"""

from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy as cp
import numpy as np

from materials import SMOKE, STEAM
from simulation import Simulation
from solver_profiles import PROFILES
from world import World

SPACING = 0.02
SIM_TIME = 2.0


def _profile(name: str):
    if name == "WCSPH":
        # Fixed dt so the run length does not depend on the CFL step.
        return dataclasses.replace(PROFILES["WCSPH"], dt=0.0005, fixed_dt=True)
    return PROFILES[name]


def measure(name: str, sim_time: float = SIM_TIME, free: bool = False) -> dict:
    """free=False: hot STEAM, default gravity; buoyancy presses it to the ceiling.
    free=True: SMOKE at ambient temperature with zero gravity, so only the gas
    pressure, viscosity and drag act (no buoyancy, no wall contact)."""
    mat = SMOKE if free else STEAM
    world = World(max_particles=4096)
    n = world.spawn_cube((-0.1, 0.3, -0.1), (0.1, 0.5, 0.1), mat,
                         spacing=SPACING, lifetime=0.0)
    if free:
        world.temperature[:n] = 293.0
    profile = _profile(name)
    sim = Simulation(world, dt=profile.dt, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=40)
    sim.set_solver_profile(profile)
    if free:
        sim.set_gravity(0.0)
    # No spawn damping: it runs for 30 substeps, so it would cover 0.25 s
    # of PBF but 15 ms of WCSPH and skew the comparison.
    sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None
    steps = max(1, int(round(sim_time / sim.fixed_dt_value)))
    n = world._high_water
    for _ in range(steps):
        sim._sim_step(n)
    cp.cuda.Device().synchronize()

    info = cp.asnumpy(world.sorted_packed_info[:n])
    rho = cp.asnumpy(world.sorted_density[:n])
    pos = cp.asnumpy(world.sorted_position[:n])[:, :3]
    gas = (info & 0xFF) == mat
    rho_g = rho[gas]
    p = pos[gas]
    c = p.mean(axis=0)
    r_xz = np.sqrt(((p[:, 0] - c[0]) ** 2 + (p[:, 2] - c[2]) ** 2).mean())
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return {
        "solver": name,
        "steps": steps,
        "count": int(gas.sum()),
        "rho_mean": float(rho_g.mean()),
        "rho_p95": float(np.percentile(rho_g, 95)),
        "r_xz": float(r_xz),
        "height": float(p[:, 1].max() - p[:, 1].min()),
        "nn_median": float(np.median(nn)),
        "close_frac": float((nn < 0.25 * SPACING).mean()),
        "nan": bool(np.isnan(p).any()),
    }


def main() -> None:
    args = sys.argv[1:]
    free = "--free" in args
    if free:
        args.remove("--free")
    names = args or ["WCSPH", "DFSPH", "PBF"]
    print("scene:", "SMOKE, ambient T, zero gravity" if free else "hot STEAM, default gravity")
    print(f"{'solver':8s} {'steps':>5s} {'count':>5s} {'rho_mean':>9s} {'rho_p95':>9s} "
          f"{'r_xz':>7s} {'height':>7s} {'nn_med':>7s} {'close':>6s}")
    for name in names:
        m = measure(name, free=free)
        print(f"{m['solver']:8s} {m['steps']:5d} {m['count']:5d} {m['rho_mean']:9.1f} "
              f"{m['rho_p95']:9.1f} {m['r_xz']:7.4f} {m['height']:7.4f} "
              f"{m['nn_median']:7.4f} {m['close_frac']:6.3f}"
              + ("  NaN!" if m["nan"] else ""))


if __name__ == "__main__":
    main()
