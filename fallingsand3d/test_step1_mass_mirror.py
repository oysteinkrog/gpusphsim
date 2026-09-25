"""K_Step1 reads the neighbour mass from the FP16 velocity mirror's .w lane.

bd-r5-review-n28.161: the counting sort stores mass / MASS_H_UNIT in
sorted_velocity_h.w, and K_Step1 takes m_j from there instead of a separate
4 B load from the mass array. These tests compare the mirror path against
the float path (velocity_h = None) on a Dam Break water block:

  - PARTICLE_MASS (0.02) water: the mirror holds exactly 1.0, so density
    matches the float path bit for bit.
  - Mixed spawn masses (SAND 0.032, OIL 0.017, steam child 0.0067 and
    random masses): density within 0.1 % (the FP16 bound is 2^-11).
  - The kernel really reads .w: zeroing only the .w lanes drops every
    density to the floor.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np

from conftest import counting_sort_scene, default_grid_params
from step1 import compute_step1
from test_step1 import PACKED_WATER, SPACING, _upload_all_params

PARTICLE_MASS = 0.02


def _dam_break_block(k: int = 16, jitter: float = 0.002, seed: int = 1):
    """k x 2k x k water block at 0.02 m spacing with small position jitter."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(k), np.arange(2 * k), np.arange(k),
                             indexing="ij"), axis=-1).reshape(-1, 3)
    pos = np.zeros((g.shape[0], 4), dtype=np.float32)
    pos[:, :3] = (g * SPACING - 0.2 + rng.uniform(-jitter, jitter, g.shape)).astype(np.float32)
    pos[:, 3] = 1.0
    vel = np.zeros_like(pos)
    vel[:, :3] = rng.uniform(-0.5, 0.5, (g.shape[0], 3)).astype(np.float32)
    return pos, vel


def _densities(pos, vel, mass, zero_mass_lane=False):
    """Step1 density with the FP16 mirror and with the float path (sorted order)."""
    _upload_all_params()
    n = pos.shape[0]
    scene = counting_sort_scene(
        pos, velocity=vel, mass=mass,
        packed_info=np.full(n, PACKED_WATER, dtype=np.uint32),
        temperature=np.full(n, 293.0, dtype=np.float32),
        grid_params=default_grid_params(),
    )
    vh = scene["velocity_h"]
    if zero_mass_lane:
        # half4 = 2 x uint32; .w is the high 16 bits of the second word.
        vh[:, 1] &= cupy.uint32(0x0000FFFF)
    args = (scene["position"], scene["velocity"], scene["mass"], None,
            scene["packed_info"], scene["temperature"],
            scene["cell_start"], scene["cell_end"])
    rho_h = compute_step1(*args, velocity_h=vh)[0].copy()
    rho_f = compute_step1(*args, velocity_h=None)[0].copy()
    return cupy.asnumpy(rho_h), cupy.asnumpy(rho_f)


def test_uniform_particle_mass_matches_float_path_exactly() -> None:
    pos, vel = _dam_break_block()
    mass = np.full(pos.shape[0], PARTICLE_MASS, dtype=np.float32)
    rho_h, rho_f = _densities(pos, vel, mass)
    assert rho_f.max() > 100.0, "scene has no neighbours"
    np.testing.assert_array_equal(rho_h, rho_f)


def test_mixed_masses_within_tenth_percent() -> None:
    pos, vel = _dam_break_block(seed=2)
    n = pos.shape[0]
    rng = np.random.default_rng(3)
    choices = np.array([0.02, 0.032, 0.017, 0.0067, 0.052, 0.008], dtype=np.float32)
    mass = choices[rng.integers(0, len(choices), n)]
    mass[::7] = rng.uniform(0.005, 0.06, mass[::7].shape).astype(np.float32)
    rho_h, rho_f = _densities(pos, vel, mass)
    rel = np.abs(rho_h - rho_f) / rho_f
    print(f"max rel density diff {rel.max():.3e}, mean {rel.mean():.3e}")
    assert rel.max() < 1e-3


def test_step1_reads_mass_from_mirror() -> None:
    pos, vel = _dam_break_block(k=8)
    mass = np.full(pos.shape[0], PARTICLE_MASS, dtype=np.float32)
    rho_h, rho_f = _densities(pos, vel, mass, zero_mass_lane=True)
    assert rho_f.min() > 100.0
    # Every m_j is 0, so the density sum is 0 and the output is the floor.
    assert rho_h.max() <= 1.0
