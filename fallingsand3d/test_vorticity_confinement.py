"""Regression test for bd-r5-review-n28.139: vorticity confinement feeds a vortex.

A gravity-free 0.4 m WATER cube (8000 particles, spacing 0.02) starts with a
Taylor-Green field u = A sin(kx) cos(ky), v = -A cos(kx) sin(ky), A = 0.3 m/s.
After 100 WCSPH substeps the kinetic energy with vorticity_epsilon = 0.05 must
be higher than with 0: confinement adds energy along the vortex, it must not
damp it. A sign error in the curl (bd-r5-review-n28.36) or in the confinement
force flips this.

Measured 2026-09-25 (RTX 5070 Ti), deterministic across repeats:
KE/KE0 = 0.967605 at eps 0 and 0.970470 at eps 0.05, a gap of 2.87e-3.
The test requires a gap above 1.0e-3.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

import step2
from materials import WATER
from simulation import Simulation
from solver_profiles import PROFILES
from world import DEFAULT_SPACING, World

BLOCK = 0.4
CENTER = 0.5
AMP = 0.3
SUBSTEPS = 100
MIN_KE_GAP = 1.0e-3  # measured 2.87e-3, see module docstring


def _taylor_green_ke_ratio(vorticity_epsilon: float) -> float:
    world = World(max_particles=16384)
    lo, hi = CENTER - BLOCK / 2, CENTER + BLOCK / 2
    n = world.spawn_cube((lo, lo, lo), (hi, hi, hi), WATER, spacing=DEFAULT_SPACING)
    assert n == 8000
    sim = Simulation(world, world_half_size=1.0)
    sim.set_solver_profile(PROFILES["WCSPH"])
    sim.set_gravity(0.0)
    sim._spawn_substep = sim._damping_duration  # no spawn velocity damping
    params = sim._wcsph_granular_params()
    params[0]["vorticity_epsilon"] = np.float32(vorticity_epsilon)
    step2.upload_granular_params(params)
    sim._invalidate_graphs()

    pos = cupy.asnumpy(world.position[:n, :3])
    kk = 2.0 * np.pi / BLOCK
    x, y = pos[:, 0] - CENTER, pos[:, 1] - CENTER
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = AMP * np.sin(kk * x) * np.cos(kk * y)
    vel[:, 1] = -AMP * np.cos(kk * x) * np.sin(kk * y)
    world.velocity[:n] = cupy.asarray(vel)

    mass = cupy.asnumpy(world.mass[:n]).astype(np.float64)

    def ke() -> float:
        v = cupy.asnumpy(world.velocity[:n, :3]).astype(np.float64)
        return float(0.5 * np.sum(mass * np.sum(v * v, axis=1)))

    e0 = ke()
    for _ in range(SUBSTEPS):
        sim._sim_step(world._high_water)
    assert world._high_water == n
    return ke() / e0


def test_confinement_feeds_taylor_green_vortex() -> None:
    r0 = _taylor_green_ke_ratio(0.0)
    r1 = _taylor_green_ke_ratio(0.05)
    print(f"  KE/KE0 after {SUBSTEPS} substeps: eps=0 {r0:.6f}  eps=0.05 {r1:.6f}  "
          f"gap {r1 - r0:.3e} (min {MIN_KE_GAP:.1e})")
    assert 0.5 < r0 < 1.0, f"viscosity should damp the vortex a little, got {r0:.4f}"
    assert r1 - r0 > MIN_KE_GAP, (
        f"confinement did not feed the vortex: KE/KE0 {r0:.6f} -> {r1:.6f}")
