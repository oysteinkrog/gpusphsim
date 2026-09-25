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
from materials import FLUID, WATER

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


def make_sim(n_max, solver, spawn_fn, spawn_damping=False, world_half_size=1.0):
    """Build a Simulation on a clean World and spawn a scene into it.

    solver: a PROFILES key ("WCSPH", "PBF", "DFSPH").
    spawn_fn(world) spawns the scene and returns the particle count.
    spawn_damping=False skips the 30-substep spawn damping ramp, so the
    first substeps run the plain solver.
    world_half_size sets the box walls at +-world_half_size on every axis.
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
                     fixed_dt=False, max_substeps=20,
                     world_half_size=world_half_size)
    sim.set_solver_profile(PROFILES[solver])
    if not spawn_damping:
        sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None
    return SimCase(world, sim, n)


# ---------------------------------------------------------------------------
# DFSPH still-water pool never sleeps (bd-r5-review-n28.130, guards n28.30)
# ---------------------------------------------------------------------------

# Slow uniform drift, below V_SLEEP (0.005 m/s in sph_shared.cuh). Water this
# slow is what a FLUID sleep counter puts to sleep, and sleeping zeroes
# velocity, so a frozen block reads KE exactly 0.
_POOL_V0 = 0.002


def _spawn_still_block(world):
    # 30 x 10 x 30 = 9000 WATER. At spacing 0.0202 the interior sits at 0.98
    # rho0 (bd-r5-review-n28.100), so the density solver does not push it
    # apart at the start.
    return world.spawn_cube((-0.3, -0.1, -0.3), (0.306, 0.102, 0.306), WATER,
                            spacing=0.0202, velocity=(_POOL_V0, 0.0, 0.0))


def _pool_state(world, n):
    """Host copies of packed_info and xyz velocity (float64)."""
    pi = cp.asnumpy(world.packed_info[:n])
    vel = cp.asnumpy(world.velocity[:n])[:, :3].astype(np.float64)
    return pi, vel


def test_dfsph_still_water_never_sleeps():
    """A still DFSPH water block, run for 2 s of sim time with zero gravity,
    has no sleeping FLUID and has not frozen (KE ratio above 0).

    Before bd-r5-review-n28.30, K_DFSPH_Finalize counted sleep for every
    behaviour, and a resting pool went fully asleep. The block is checked at
    three times: 0.05 s (15 substeps, past SLEEP_THRESHOLD = 10), 0.5 s and
    2 s. The KE ratio also catches a block frozen some other way:
    eval_solvers.py only checks that KE decays, which a frozen pool passes
    with ratio 0.

    Zero gravity keeps the block still without a container. A water pool
    under gravity is not used: DFSPH keeps it moving at about 0.15 m/s rms,
    so no particle stays below V_SLEEP long enough to test the sleep gate.
    """
    case = make_sim(20_000, "DFSPH", _spawn_still_block)
    n = case.n
    total = case.substeps_for(2.0)
    checkpoints = [15, case.substeps_for(0.5), total]
    states = []
    case.sim.set_gravity(0.0)
    try:
        done = 0
        for c in checkpoints:
            case.run(c - done)
            done = c
            states.append(_pool_state(case.world, n))
    finally:
        case.sim.set_gravity(-4.0)

    ke0 = n * _POOL_V0 ** 2  # uniform WATER mass: KE ratio = ratio of sum |v|^2
    for c, (pi, vel) in zip(checkpoints, states):
        t = c * case.sim.dt
        mat = pi & 0xFF
        beh = (pi >> 8) & 0x3
        assert (mat == WATER).all(), f"t={t:.2f} s: block lost or changed particles"
        fluid = beh == FLUID
        assert fluid.all()
        asleep = (pi & _SLEEP_BIT) != 0
        assert not asleep[fluid].any(), \
            f"t={t:.2f} s: {asleep[fluid].sum()}/{n} FLUID particles asleep"
        assert np.isfinite(vel).all()
        ratio = float(np.sum(vel * vel)) / ke0
        vmax = float(np.sqrt((vel * vel).sum(axis=1)).max())
        print(f"\n  DFSPH still block t={t:.2f} s: n={n}, KE ratio={ratio:.4g}, vmax={vmax:.4f}",
              end="")
        assert ratio > 0.0, f"t={t:.2f} s: block is frozen (KE ratio exactly 0)"
        # Blow-up guard only. The block does not stay still: its corners
        # spread at up to about 1.6 m/s by 2 s (reported with
        # bd-r5-review-n28.130).
        assert vmax < 3.0, f"t={t:.2f} s: block blew up (vmax {vmax:.3f} m/s)"
