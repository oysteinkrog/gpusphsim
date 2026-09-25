"""bd-r5-review-n28.70: world-box restitution and wall friction under PBF.

K_PBF_Predict and K_PBF_ComputeDelta clamp the predicted position to the wall
margin, so K_PBF_Finalize sees a particle sitting exactly on the margin. The
box collision in Finalize must still treat that as contact (pos <= margin with
velocity into the wall), or PBF walls end up perfectly inelastic and
frictionless.

Each test places one particle 15 mm above the floor margin moving down at
2 m/s. One PBF substep (dt = 1/120) would carry it about 17 mm, so Predict
clamps it to the margin and Finalize sees an impact velocity of
v_n = -0.015 / dt = -1.8 m/s.
"""

import numpy as np
import pytest

import cupy as cp
from world import World
from simulation import Simulation
from solver_profiles import PROFILES
from materials import WATER, SAND

GAP = 0.015  # m above the floor margin
BOUNDARY_MARGIN = 1e-4


@pytest.fixture(scope="module")
def world():
    return World(max_particles=1024)


@pytest.fixture
def sim(world):
    s = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                   fixed_dt=False, max_substeps=20)
    s.set_solver_profile(PROFILES["PBF"])
    # Spawn damping would scale the velocity on the first substeps.
    s._spawn_substep = s._damping_duration
    return s


def _drop_one(world, sim, material_id, vel, gap=GAP):
    """Spawn one particle `gap` above the floor margin with velocity vel."""
    world.packed_info[:] = 0
    world._high_water = 0
    floor = -sim.world_half_size + BOUNDARY_MARGIN
    y = floor + gap
    n = world.spawn_cube(
        min_corner=(-0.01, y - 0.01, -0.01),
        max_corner=(0.01, y + 0.01, 0.01),
        material_id=material_id,
        spacing=0.02,
        velocity=vel,
    )
    assert n == 1
    world.position[0, 1] = y
    sim.notify_scene_changed()
    sim._spawn_substep = sim._damping_duration
    sim._sim_step(1)
    cp.cuda.Device().synchronize()
    pos = world.position[0].get()
    v = world.velocity[0].get()
    return floor, pos, v, sim.dt


def test_pbf_water_bounces_off_floor(world, sim):
    floor, pos, v, dt = _drop_one(world, sim, WATER, (0.0, -2.0, 0.0))
    v_n = -GAP / dt  # impact velocity seen by Finalize
    print(f"  WATER: y={pos[1]:.6f} floor={floor:.6f} v_y={v[1]:.4f} "
          f"(expected ~{-0.3 * v_n:.4f})")
    assert abs(pos[1] - floor) < 1e-5, "particle should sit on the floor margin"
    assert v[1] > 0.3, f"PBF floor gave v_y={v[1]:.4f}, expected a bounce > 0.3 m/s"
    assert v[1] == pytest.approx(-0.3 * v_n, rel=0.05)


def test_pbf_sand_wall_friction(world, sim):
    floor, pos, v, dt = _drop_one(world, sim, SAND, (1.0, -2.0, 0.0))
    v_n = GAP / dt  # |impact normal velocity|
    ts_before = 1.0
    ts_after = float(np.hypot(v[0], v[2]))
    print(f"  SAND: v_x={v[0]:.4f} v_y={v[1]:.4f} tangential {ts_before:.3f} -> "
          f"{ts_after:.4f} (|vn|={v_n:.3f})")
    assert abs(pos[1] - floor) < 1e-5, "particle should sit on the floor margin"
    # Coulomb wall friction: tangential speed drops by min(mu * |vn|, ts).
    expected_drop = min(0.5 * v_n, ts_before)
    assert ts_before - ts_after >= 0.9 * expected_drop, (
        f"tangential speed dropped {ts_before - ts_after:.4f}, "
        f"expected >= {expected_drop:.4f} from wall_friction")


def test_pbf_resting_particle_stays_at_rest(world, sim):
    """A particle resting on the floor margin gets no bounce from the <= contact."""
    floor, pos, v, dt = _drop_one(world, sim, WATER, (0.0, 0.0, 0.0), gap=0.0)
    print(f"  resting: y={pos[1]:.6f} v={v[:3]}")
    assert abs(pos[1] - floor) < 1e-5
    assert abs(v[1]) < 1e-4, f"resting particle got v_y={v[1]:.6f}"
