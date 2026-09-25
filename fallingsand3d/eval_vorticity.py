"""eval_vorticity.py -- print-only probes for the WCSPH vorticity-confinement term.

These probes used to live in test_odd_viscosity.py as pytest cases. Two of
them were xfail(strict=False), so they could never pass or fail. They are a
measurement, not a regression guard, so they now print numbers and assert
nothing. The regression guards are:

  * test_step1.py::test_rigid_rotation_curl_is_two_omega (curl sign and size)
  * test_vorticity_confinement.py (Taylor-Green energy with and without
    confinement)

Probes, all on a gravity-free WATER cube (side BLOCK, spacing 0.02) run with
the WCSPH profile and a chosen vorticity_epsilon:

  1. Net L_z drift of a Taylor-Green field (zero net linear and angular
     momentum at t = 0) after FRAMES * SUBSTEPS_PER_FRAME substeps, for each
     eps in EPS_LIST. The "Scan-Order Hall Fluid" theory in
     ODD_VISCOSITY_THEORY.md predicts a drift that grows with eps.
  2. Transverse RMS v_x induced by a pure shear v_y(x), with eps 0 and 0.10.
  3. Kinetic energy ratio KE(t) / KE(0) of the Taylor-Green field for each eps.

Recorded results are in ../PROBLEMS.md (G3) and ODD_VISCOSITY_THEORY.md.

Run:  python eval_vorticity.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cupy
import numpy as np

import step2
from materials import WATER
from simulation import Simulation
from solver_profiles import PROFILES
from world import DEFAULT_SPACING, World

BLOCK = 0.4  # side length of the fluid cube (m)
FRAMES = 30
SUBSTEPS_PER_FRAME = 10
EPS_LIST = (0.0, 0.05, 0.10)
CENTER = 0.5


def build_sim(vorticity_epsilon: float, block: float = BLOCK,
              max_particles: int = 200_000):
    """Gravity-free WATER cube with the WCSPH profile and a chosen eps."""
    world = World(max_particles=max_particles)
    lo = CENTER - block / 2
    hi = CENTER + block / 2
    n = world.spawn_cube((lo, lo, lo), (hi, hi, hi), WATER, spacing=DEFAULT_SPACING)
    sim = Simulation(world, world_half_size=1.0)
    sim.set_solver_profile(PROFILES["WCSPH"])
    sim.set_gravity(0.0)
    sim._spawn_substep = sim._damping_duration  # no spawn velocity damping
    params = sim._wcsph_granular_params()
    params[0]["vorticity_epsilon"] = np.float32(vorticity_epsilon)
    step2.upload_granular_params(params)
    sim._invalidate_graphs()
    return world, sim, n


def seed_taylor_green(world: World, n: int, block: float = BLOCK,
                      amp: float = 0.3) -> None:
    """u = A sin(kx) cos(ky), v = -A cos(kx) sin(ky): divergence free,
    zero net linear and angular momentum."""
    pos = cupy.asnumpy(world.position[:n, :3])
    x = pos[:, 0] - CENTER
    y = pos[:, 1] - CENTER
    kk = 2.0 * np.pi / block
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = amp * np.sin(kk * x) * np.cos(kk * y)
    vel[:, 1] = -amp * np.cos(kk * x) * np.sin(kk * y)
    world.velocity[:n] = cupy.asarray(vel)


def advance(sim: Simulation, world: World, substeps: int) -> None:
    """Run substeps directly; step_frame() is paced by wall-clock time."""
    for _ in range(substeps):
        sim._sim_step(world._high_water)


def net_lz(world: World, n: int) -> float:
    """Net z angular momentum about the block centre."""
    pos = cupy.asnumpy(world.position[:n, :3]).astype(np.float64)
    vel = cupy.asnumpy(world.velocity[:n, :3]).astype(np.float64)
    mass = cupy.asnumpy(world.mass[:n]).astype(np.float64)
    x = pos[:, 0] - CENTER
    y = pos[:, 1] - CENTER
    return float(np.sum(mass * (x * vel[:, 1] - y * vel[:, 0])))


def kinetic_energy(world: World, n: int) -> float:
    vel = cupy.asnumpy(world.velocity[:n, :3]).astype(np.float64)
    mass = cupy.asnumpy(world.mass[:n]).astype(np.float64)
    return float(0.5 * np.sum(mass * np.sum(vel * vel, axis=1)))


def taylor_green_run(vorticity_epsilon: float, substeps: int,
                     block: float = BLOCK, max_particles: int = 200_000):
    """Return (L_z drift, KE(t)/KE(0), particle count)."""
    world, sim, n = build_sim(vorticity_epsilon, block, max_particles)
    seed_taylor_green(world, n, block)
    l0 = net_lz(world, n)
    e0 = kinetic_energy(world, n)
    advance(sim, world, substeps)
    n = world._high_water
    return net_lz(world, n) - l0, kinetic_energy(world, n) / e0, n


def shear_response(vorticity_epsilon: float, substeps: int = 5 * SUBSTEPS_PER_FRAME) -> float:
    """Transverse RMS v_x induced by a pure shear v_y = 0.3 sin(k x)."""
    world, sim, n = build_sim(vorticity_epsilon)
    pos = cupy.asnumpy(world.position[:n, :3])
    kk = 2.0 * np.pi / BLOCK
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 1] = 0.3 * np.sin(kk * (pos[:, 0] - CENTER))
    world.velocity[:n] = cupy.asarray(vel)
    advance(sim, world, substeps)
    n = world._high_water
    v = cupy.asnumpy(world.velocity[:n, :3])
    return float(np.sqrt(np.mean(v[:, 0] ** 2)))


def main() -> None:
    substeps = FRAMES * SUBSTEPS_PER_FRAME
    print(f"== Taylor-Green, block {BLOCK} m, {substeps} substeps ==")
    for eps in EPS_LIST:
        drift, ke_ratio, n = taylor_green_run(eps, substeps)
        print(f"  eps={eps:<5.3f} n={n}  L_z drift {drift:+.3e}  KE/KE0 {ke_ratio:.5f}")

    print("== transverse shear coupling, 50 substeps ==")
    r0 = shear_response(0.0)
    r1 = shear_response(0.10)
    print(f"  RMS v_x: eps=0 {r0:.4e}  eps=0.10 {r1:.4e}  ratio {r1 / r0:.4f}")


if __name__ == "__main__":
    main()
