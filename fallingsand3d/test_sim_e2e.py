"""End-to-end tests that drive a real Simulation through _sim_step.

Most solver tests call kernels directly with hand-built inputs, so a bug in
how Simulation wires the kernels together (buffer writeback, sort reorder,
finalize state) can pass them all. Tests here build World and Simulation,
step the scene with sim._sim_step (never step_frame, so there is no frame
timing or substep count logic in the way), and read the state back once at
the end.

Add new Simulation-level regression tests to this module and build them on
make_sim. Keep every case under 20K particles and the module under 60 s.
"""

import math

import numpy as np
import pytest

import cupy as cp
from world import World
from simulation import Simulation
from solver_profiles import PROFILES
from materials import WATER

pytestmark = pytest.mark.gpu

_SLEEP_BIT = 0x400  # SET_SLEEPING in physics/kernels/common.cuh

_worlds = {}


def _get_world(n_max):
    """One World per capacity for the whole module (allocation is slow)."""
    w = _worlds.get(n_max)
    if w is None:
        w = World(max_particles=n_max)
        _worlds[n_max] = w
    return w


class SimCase:
    """A World plus Simulation with a scene spawned and a solver profile set."""

    def __init__(self, world, sim, n):
        self.world = world
        self.sim = sim
        self.n = n

    def run(self, substeps):
        """Run substeps with no host readback inside the loop."""
        sim = self.sim
        world = self.world
        for _ in range(substeps):
            sim._sim_step(world._high_water)

    def substeps_for(self, seconds):
        return int(math.ceil(seconds / self.sim.dt))


def make_sim(n_max, solver, spawn_fn, spawn_damping=False):
    """Build a Simulation on a clean World and spawn a scene into it.

    solver: a PROFILES key ("WCSPH", "PBF", "DFSPH").
    spawn_fn(world) spawns the scene and returns the particle count.
    spawn_damping=False skips the 30-substep spawn damping ramp, so the
    first substeps run the plain solver.
    """
    world = _get_world(n_max)
    world.packed_info[:] = 0
    world._high_water = 0
    # The first DFSPH substep builds alpha from the previous substep's
    # sorted_density; clear what an earlier case left there.
    world.sorted_density[:] = 0.0
    world.sleep_counter[:] = 0
    n = spawn_fn(world)
    assert 0 < n <= n_max
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=False, max_substeps=20)
    sim.set_solver_profile(PROFILES[solver])
    if not spawn_damping:
        sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None
    return SimCase(world, sim, n)


# ---------------------------------------------------------------------------
# DFSPH still-water pool never sleeps (bd-r5-review-n28.130, guards n28.30)
# ---------------------------------------------------------------------------

_POOL_V0 = 0.1  # m/s sideways, so the pool has KE to lose


def _spawn_pool(world):
    # Floor is at y = -1 (world_half_size = 1). 30 x 10 x 30 = 9000 WATER.
    return world.spawn_cube((-0.3, -1.0, -0.3), (0.3, -0.8, 0.3), WATER,
                            spacing=0.02, velocity=(_POOL_V0, 0.0, 0.0))


def test_dfsph_pool_never_sleeps():
    """A DFSPH water pool run for 2 s of sim time has no sleeping FLUID and
    has not frozen (KE ratio above 0).

    Before bd-r5-review-n28.30, K_DFSPH_Finalize counted sleep for every
    behaviour, and a resting pool went fully asleep (19200/19200) in 2 s.
    The KE ratio catches a pool frozen some other way: eval_solvers.py only
    checks that KE decays, which a frozen pool passes with ratio 0.
    """
    case = make_sim(20_000, "DFSPH", _spawn_pool)
    n = case.n
    case.run(case.substeps_for(2.0))

    pi = cp.asnumpy(case.world.packed_info[:n])
    vel = cp.asnumpy(case.world.velocity[:n])

    mat = pi & 0xFF
    beh = (pi >> 8) & 0x3
    assert (mat == WATER).all(), "pool lost or changed particles"
    fluid = beh == 0  # FLUID behavior class
    assert fluid.all()
    asleep = (pi & _SLEEP_BIT) != 0
    assert not asleep[fluid].any(), f"{asleep[fluid].sum()}/{n} FLUID particles asleep"

    assert np.isfinite(vel[:, :3]).all()
    # Uniform WATER mass, so the KE ratio is the ratio of summed |v|^2.
    ke0 = n * _POOL_V0 ** 2
    ke = float(np.sum(vel[:, :3].astype(np.float64) ** 2))
    ratio = ke / ke0
    print(f"\n  DFSPH pool: n={n}, substeps={case.substeps_for(2.0)}, KE ratio={ratio:.4g}")
    assert ratio > 0.0, "pool is frozen (KE exactly 0)"
    assert ratio < 10.0, f"pool gained energy (KE ratio {ratio:.3g})"
