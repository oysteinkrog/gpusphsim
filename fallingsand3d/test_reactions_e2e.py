"""End-to-end reaction test through Simulation._sim_step (bd-r5-review-n28.126).

The other reaction tests call compute_reactions on sorted arrays, or copy the
sorted state back by hand. This one drives the real substep pipeline, so it
fails if a solver's finalize kernel stops carrying health, lifetime or mass
back to the unsorted arrays (the round-4 carry-back bug).
"""

import numpy as np
import pytest

from world import World
from simulation import Simulation
from solver_profiles import PROFILES
from materials import DEAD, FIRE, SMOKE, STEAM, WATER, ACID, METAL, GAS

pytestmark = pytest.mark.gpu

MAX_PARTICLES = 30_000  # room for the 15 000 parked DEAD reserve
SIM_SECONDS = 1.0
FIRE_LIFETIME = 0.5
# Lifetime a gas gets when it appears: FIRE-to-SMOKE in reactions.cu,
# SPAWN_LIFETIME for boiled steam in spawn.cu.
SPAWN_LIFETIME = {SMOKE: 3.0, STEAM: 12.0}


def _spawn_scene(w):
    # Groups sit in three far corners. Splashed ACID at 293 K cools STEAM
    # below its 360 K condense point, and condensation keeps mass only on
    # average, so the mass check needs them apart.
    n_fire = w.spawn_sphere((-0.6, -0.6, -0.6), 0.08, FIRE, 200,
                            lifetime=FIRE_LIFETIME)
    w0 = w._high_water
    n_water = w.spawn_cube((0.6, -1.0, -0.7), (0.7, -0.84, -0.6), WATER, spacing=0.02)
    w.temperature[w0:w0 + n_water] = 400.0  # WATER boils above 373 K
    m0 = w._high_water
    n_metal = w.spawn_cube((-0.7, -1.0, 0.6), (-0.6, -0.92, 0.7), METAL, spacing=0.02)
    metal = slice(m0, m0 + n_metal)
    n_acid = w.spawn_cube((-0.7, -0.92, 0.6), (-0.6, -0.84, 0.7), ACID, spacing=0.02)
    assert (n_fire, n_water, n_metal, n_acid) == (200, 200, 100, 100)
    return metal


@pytest.mark.parametrize("solver", ["WCSPH", "PBF", "DFSPH"])
def test_fire_steam_acid_through_sim_step(solver):
    w = World(max_particles=MAX_PARTICLES)
    sim = Simulation(w)
    sim.set_solver_profile(PROFILES[solver])
    metal = _spawn_scene(w)
    sim.notify_scene_changed()
    sim._maybe_compact()  # adds the parked DEAD reserve K_SpawnGas claims
    mass0 = float(w.mass[:w._high_water].sum())

    for _ in range(int(round(SIM_SECONDS / sim.dt))):
        sim._sim_step(w._high_water)

    n = w._high_water
    mat = (w.packed_info[:n].get() & 0xFF).astype(np.int32)
    beh = ((w.packed_info[:n].get() >> 8) & 0x3).astype(np.int32)
    mass = w.mass[:n].get()
    health = w.health[:n].get()
    lifetime = w.lifetime[:n].get()

    assert np.count_nonzero(mat == FIRE) == 0, "FIRE did not burn out"
    assert np.count_nonzero(mat == STEAM) > 0, "no WATER boiled into STEAM"
    assert np.all(mass[mat == DEAD] == 0.0), "a DEAD slot kept its mass"
    alive_mass = float(mass[mat != DEAD].astype(np.float64).sum())
    counts = {int(m): int((mat == m).sum()) for m in np.unique(mat)}
    assert alive_mass == pytest.approx(mass0, rel=1e-3), counts
    metal_mat = mat[metal]
    assert np.all(metal_mat == METAL)
    assert health[metal].mean() < 1.0, "ACID did not corrode METAL"

    # Negative control: with any lifetime_out pointer reverted to NULL the
    # unsorted lifetime keeps its spawn value (or 0 for a steam child).
    for m, spawn_lt in SPAWN_LIFETIME.items():
        lt = lifetime[(mat == m) & (beh == GAS)]
        assert lt.size > 0, f"no surviving material {m}"
        assert np.all(lt > 0.0) and np.all(lt < spawn_lt), (m, lt.min(), lt.max())
