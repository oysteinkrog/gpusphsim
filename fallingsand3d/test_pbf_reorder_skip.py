"""PBF reorder skips the buffers PBF never reads (bd-r5-review-n28.45).

PBF recomputes lambda from scratch every iteration, so there is no lambda
warm start to carry through the sort. PBF also never reads the DFSPH
warm-start state (kappa, kappa_v) or the FP16 mirrors (velocity and
temperature), so _run_grid_setup passes None for those and the reorder kernels skip
both the gather and the store.

bd-r5-review-n28.165 extends the gating: WCSPH skips kappa and kappa_v, and
DFSPH skips the FP16 mirrors, since only WCSPH kernels read them.
"""

import cupy as cp
import numpy as np
import pytest

from materials import WATER
from simulation import Simulation
from solver_profiles import PROFILES
from world import World

SENTINEL = np.float32(-12345.0)


@pytest.fixture(scope="module")
def world():
    return World(max_particles=20_000)


def _run(world, profile_name, full_sort_substeps=1):
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, spacing=0.02)
    sim = Simulation(world, dt=1.0 / 120.0, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=20)
    sim.set_solver_profile(PROFILES[profile_name])
    sim.notify_scene_changed()
    sim._spawn_substep = sim._damping_duration
    # Distinct unsorted values so a reorder would overwrite the sentinels.
    world.kappa[:n] = cp.float32(1.0)
    world.kappa_v[:n] = cp.float32(2.0)
    world.sorted_kappa[:] = SENTINEL
    world.sorted_kappa_v[:] = SENTINEL
    world.sorted_velocity_h[:] = cp.uint32(0xDEADBEEF)
    world.sorted_temperature_h[:] = cp.uint16(0xBEEF)
    for _ in range(full_sort_substeps):
        sim._run_grid_setup(n, force_sort=True)
    # And once through the grid-reuse gather path.
    sim._sort_skip_next = True
    sim._run_grid_setup(n)
    cp.cuda.Device().synchronize()
    return n


def test_world_has_no_lambda_warm_start(world):
    assert not hasattr(world, "lambda_pbf"), "lambda_pbf warm start was removed"
    assert hasattr(world, "sorted_lambda_pbf"), "PBF still needs the per-iteration scratch"


def test_pbf_reorder_skips_unread_buffers(world):
    n = _run(world, "PBF")
    assert (cp.asnumpy(world.sorted_kappa[:n]) == SENTINEL).all()
    assert (cp.asnumpy(world.sorted_kappa_v[:n]) == SENTINEL).all()
    assert (cp.asnumpy(world.sorted_velocity_h[:n]) == 0xDEADBEEF).all()
    assert (cp.asnumpy(world.sorted_temperature_h[:n]) == 0xBEEF).all()


def test_dfsph_reorder_writes_warm_start_only(world):
    n = _run(world, "DFSPH")
    assert (cp.asnumpy(world.sorted_kappa[:n]) == 1.0).all()
    assert (cp.asnumpy(world.sorted_kappa_v[:n]) == 2.0).all()
    # No DFSPH kernel reads the FP16 mirrors (bd-r5-review-n28.165).
    assert (cp.asnumpy(world.sorted_velocity_h[:n]) == 0xDEADBEEF).all()
    assert (cp.asnumpy(world.sorted_temperature_h[:n]) == 0xBEEF).all()


def test_wcsph_reorder_writes_mirrors_only(world):
    n = _run(world, "WCSPH")
    # WCSPH never reads the DFSPH warm start (bd-r5-review-n28.165).
    assert (cp.asnumpy(world.sorted_kappa[:n]) == SENTINEL).all()
    assert (cp.asnumpy(world.sorted_kappa_v[:n]) == SENTINEL).all()
    assert not (cp.asnumpy(world.sorted_velocity_h[:n]) == 0xDEADBEEF).any()
    assert not (cp.asnumpy(world.sorted_temperature_h[:n]) == 0xBEEF).any()


def test_switch_through_wcsph_keeps_kappa_warm_start(world):
    """DFSPH -> WCSPH -> DFSPH: WCSPH skips kappa in the reorder and nothing
    in its pipeline writes the unsorted kappa, so the DFSPH warm start comes
    back unchanged on the next DFSPH reorder."""
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube((-0.1, 0.0, -0.1), (0.1, 0.2, 0.1), WATER, spacing=0.02)
    sim = Simulation(world, dt=1.0 / 120.0, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=20)
    sim.set_solver_profile(PROFILES["DFSPH"])
    sim.notify_scene_changed()
    sim._spawn_substep = sim._damping_duration
    for _ in range(5):
        sim._sim_step(n)
    # Distinct per-particle values stand in for the warm start, so the check
    # does not depend on how far the scene has settled.
    rng = np.random.default_rng(5)
    kappa_before = rng.uniform(0.1, 1.0, n).astype(np.float32)
    kappa_v_before = rng.uniform(0.1, 1.0, n).astype(np.float32)
    world.kappa[:n] = cp.asarray(kappa_before)
    world.kappa_v[:n] = cp.asarray(kappa_v_before)

    sim.set_solver_profile(PROFILES["WCSPH"])
    sim._spawn_substep = sim._damping_duration
    for _ in range(5):
        sim._sim_step(n)
    np.testing.assert_array_equal(cp.asnumpy(world.kappa[:n]), kappa_before)
    np.testing.assert_array_equal(cp.asnumpy(world.kappa_v[:n]), kappa_v_before)

    sim.set_solver_profile(PROFILES["DFSPH"])
    world.sorted_kappa[:] = SENTINEL
    sim._run_grid_setup(n, force_sort=True)
    perm = cp.asnumpy(sim._sort_perm[:n]).astype(np.int64)
    np.testing.assert_array_equal(cp.asnumpy(world.sorted_kappa[:n]),
                                  kappa_before[perm])


def test_pbf_steps_stay_finite(world):
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube((-0.1, 0.0, -0.1), (0.1, 0.2, 0.1), WATER, spacing=0.02)
    sim = Simulation(world, dt=1.0 / 120.0, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=20)
    sim.set_solver_profile(PROFILES["PBF"])
    sim.notify_scene_changed()
    sim._spawn_substep = sim._damping_duration
    for _ in range(20):
        sim._sim_step(n)
    pos = cp.asnumpy(world.position[:n])
    vel = cp.asnumpy(world.velocity[:n])
    assert np.isfinite(pos).all() and np.isfinite(vel).all()
    assert np.abs(vel[:, :3]).max() < 10.0
