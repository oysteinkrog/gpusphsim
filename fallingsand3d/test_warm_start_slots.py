"""DFSPH warm-start state never leaks into a reused slot (bd-r5-review-n28.121).

kappa (p/rho^2) and kappa_v are carried from one substep to the next. A slot
that compaction frees, or that the rigid boundary block is written into, must
not hand the previous occupant's values to the new particle. The DFSPH
neighbour loops also treat a STATIC neighbour's kappa_v and p_rho2 as zero, so
a wall never pushes with a stale pressure.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

from materials import MAT_RIGID, STONE, WATER  # noqa: E402

SPACING = 0.02


def _dfsph_sim(world):
    from simulation import Simulation
    from solver_profiles import PROFILES

    sim = Simulation(world, dt=0.001, fixed_dt=True)
    sim.set_solver_profile(PROFILES["DFSPH"])
    # No spawn damping: it scales the first substeps' velocities.
    sim._spawn_substep = sim._damping_duration
    return sim


def _step(sim, k: int = 1) -> None:
    for _ in range(k):
        sim._sim_step(sim.world._high_water)
    cp.cuda.Device().synchronize()


def test_compact_zeroes_warm_start_tail():
    from world import World

    w = World(5000)
    n = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, SPACING)
    assert n > 200
    w.kappa[:] = 17.0
    w.kappa_v[:] = 3.0
    w.packed_info[:200] = cp.uint32(0)  # kill 200
    alive = w.compact()
    assert alive == n - 200
    assert np.all(w.kappa[alive:n].get() == 0.0)
    assert np.all(w.kappa_v[alive:n].get() == 0.0)
    # Live particles keep their warm start.
    assert np.all(w.kappa[:alive].get() == 17.0)
    assert np.all(w.kappa_v[:alive].get() == 3.0)


def test_inject_boundary_zeroes_warm_start():
    from simulation import Simulation
    from world import World

    w = World(8000)
    w.spawn_cube((-0.3, -0.3, -0.3), (-0.1, -0.1, -0.1), WATER, SPACING)
    sim = Simulation(w, dt=0.001, fixed_dt=True)
    # Stale values everywhere, as left by earlier occupants of the slots.
    w.kappa[:] = 17.0
    w.kappa_v[:] = 3.0
    w.angular_velocity[:] = 1.0
    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(
        position=(0.3, 0.3, 0.3), half_extents=(0.05, 0.05, 0.05), mass=2.0
    )
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    rbm.inject_boundary_particles(w)
    off, nb = rbm.boundary_offset, rbm.num_boundary_particles
    assert off >= 0 and nb > 0
    assert np.all((w.packed_info[off : off + nb].get() & 0xFF) == MAT_RIGID)
    assert np.all(w.kappa[off : off + nb].get() == 0.0)
    assert np.all(w.kappa_v[off : off + nb].get() == 0.0)
    assert np.all(w.angular_velocity[off : off + nb].get() == 0.0)


def _pool_on_floor(stale: bool):
    from world import World

    w = World(20_000)
    w.spawn_cube((-0.2, -0.6, -0.2), (0.2, -0.56, 0.2), STONE, SPACING)
    n_floor = w._high_water
    w.spawn_cube((-0.15, -0.56, -0.15), (0.15, -0.46, 0.15), WATER, SPACING)
    if stale:
        # A STATIC slot holding a fluid-sized warm start.
        w.kappa[:n_floor] = 17.0
        w.kappa_v[:n_floor] = 3.0
    sim = _dfsph_sim(w)
    _step(sim, 1)
    return w.velocity[n_floor : w._high_water, :3].get()


def test_static_neighbour_warm_start_is_ignored():
    v_clean = _pool_on_floor(stale=False)
    v_stale = _pool_on_floor(stale=True)
    diff = np.abs(v_stale - v_clean).max()
    # Measured: 2e-6 m/s with the STATIC guard, 0.037 m/s without it (the
    # bottom water layer sits 0.02 m above the floor, inside h = 0.04).
    assert diff < 1e-3, f"stale STATIC warm start moved fluid by {diff:.4g} m/s"


def _first_substep_speed(compacted: bool) -> float:
    from world import World

    w = World(20_000)
    if compacted:
        w.spawn_cube((-0.8, -0.8, -0.8), (-0.5, -0.6, -0.5), WATER, SPACING)
        sim = _dfsph_sim(w)
        _step(sim, 20)
        w.packed_info[:200] = cp.uint32(0)  # kill 200
        w.compact()
    start = w._high_water
    n = w.spawn_cube((0.3, 0.3, 0.3), (0.45, 0.45, 0.45), WATER, SPACING)
    assert n > 0
    if compacted:
        sim.notify_scene_changed()
    else:
        sim = _dfsph_sim(w)
    _step(sim, 1)
    v = w.velocity[start : start + n, :3].get()
    return float(np.linalg.norm(v, axis=1).mean())


def test_spawn_after_compaction_matches_fresh_scene():
    fresh = _first_substep_speed(compacted=False)
    reused = _first_substep_speed(compacted=True)
    assert fresh > 0.0
    assert abs(reused - fresh) <= 0.1 * fresh, (
        f"first-substep |v| after compaction {reused:.4g} vs fresh {fresh:.4g}"
    )
