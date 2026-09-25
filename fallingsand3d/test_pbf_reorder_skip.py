"""PBF reorder skips the buffers PBF never reads (bd-r5-review-n28.45).

PBF recomputes lambda from scratch every iteration, so there is no lambda
warm start to carry through the sort. PBF also never reads the DFSPH
warm-start state (kappa, kappa_v) or the FP16 mirrors (velocity, temperature,
dye), so _run_grid_setup passes None for those and the reorder kernels skip
both the gather and the store.
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
    world.sorted_dye_h[:] = cp.uint32(0xDEADBEEF)
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
    assert (cp.asnumpy(world.sorted_dye_h[:n]) == 0xDEADBEEF).all()
    assert (cp.asnumpy(world.sorted_temperature_h[:n]) == 0xBEEF).all()


def test_dfsph_reorder_still_writes_buffers(world):
    n = _run(world, "DFSPH")
    assert (cp.asnumpy(world.sorted_kappa[:n]) == 1.0).all()
    assert (cp.asnumpy(world.sorted_kappa_v[:n]) == 2.0).all()
    assert not (cp.asnumpy(world.sorted_velocity_h[:n]) == 0xDEADBEEF).any()


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
