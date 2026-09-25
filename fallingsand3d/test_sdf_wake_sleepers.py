"""Kinematic SDF objects and world resizes wake sleeping particles (bd-r5-review-n28.119).

K_Integrate's sleeping branch used to return before the SDF object loop and
the world box clamp. A kinematic box moving through a settled sand pile then
passed through it: no push-out, and nothing woke the sand. The sleeping
branch now wakes a particle that is inside an SDF object or outside the
world box, so the normal path pushes it out this substep.

Scene: a 0.4 x 0.2 x 0.4 m sand block (spacing 0.02) that starts fully
asleep, as a settled Sand Castle does. A 0.1 m kinematic box oscillates along
x through the block's left edge, the Wrecking Ball / Water Mill case. WCSPH,
dt = 1 ms, 60 substeps.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

from materials import SAND  # noqa: E402

SLEEPING_BIT = 0x400
FLOOR = -0.5                # world_half_size = 0.5
SAND_TOP = -0.3
BOX_CENTER = (-0.2, -0.4, 0.0)
BOX_HALF = (0.05, 0.05, 0.05)
AMPLITUDE = 0.1             # m, box sweeps from x=-0.2 to x=-0.1 in 60 substeps
FREQUENCY = 4.0             # Hz, peak speed 2.5 m/s
STEPS = 60


def _sdf_box(p, center):
    """Signed distance to the axis-aligned box, same formula as sdf_box()."""
    q = np.abs(p - np.asarray(center)) - np.asarray(BOX_HALF)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


def _sleeping_block(world_half_size=0.5):
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(40_000)
    world.spawn_cube((-0.2, FLOOR, -0.2), (0.2, SAND_TOP, 0.2), SAND, 0.02)
    sim = Simulation(world, dt=0.001, fixed_dt=True,
                     world_half_size=world_half_size)
    sim.set_solver_profile(PROFILES["WCSPH"])
    sim._spawn_substep = sim._damping_duration
    n = world._high_water
    world.packed_info[:n] |= cp.uint32(SLEEPING_BIT)
    world.sleep_counter[:n] = 200
    world.velocity[:n] = 0.0
    return world, sim


@pytest.fixture(scope="module")
def box_sweep():
    import rigid_bodies

    world, sim = _sleeping_block()
    try:
        sdf = sim.sdf_manager
        obj = sdf.add_sdf_object(
            rigid_bodies.SDF_BOX, position=BOX_CENTER, size=BOX_HALF,
            restitution=0.0, friction=0.3)
        sdf.add_kinematic_motion(
            obj, "oscillate_x",
            {"amplitude": AMPLITUDE, "frequency": FREQUENCY})
        sdf.force_upload()

        n = world._high_water
        pos0 = world.position[:n, :3].get()
        inside0 = int((_sdf_box(pos0, BOX_CENTER) < 0.0).sum())

        for _ in range(STEPS):
            sim._sim_step(world._high_water)
        cp.cuda.Device().synchronize()

        n = world._high_water
        pos = world.position[:n, :3].get()
        pi = world.packed_info[:n].get()
        center = sdf.get_sdf_objects()[obj]["position"]
        # Box travel per substep at the last step: the kernel pushes out
        # against the box position of that substep, which then moves on.
        vx = sdf.get_sdf_objects()[obj]["velocity"][0]
        yield {
            "pos0": pos0, "pos": pos, "pi": pi, "center": center,
            "inside0": inside0, "travel": abs(vx) * sim.dt,
        }
    finally:
        # SDF objects live in module constant memory shared by later tests.
        rigid_bodies.upload_sdf_objects(None, 0)


def test_scene_starts_with_sleepers_inside_box(box_sweep):
    assert box_sweep["inside0"] > 0, "setup: no sand starts inside the box"


def test_no_sleeping_particle_inside_box(box_sweep):
    d = _sdf_box(box_sweep["pos"], box_sweep["center"])
    sleeping = ((box_sweep["pi"] >> 10) & 1) == 1
    bad = int(((d < 0.0) & sleeping).sum())
    print(f"  sleeping particles inside the box after {STEPS} substeps: {bad}")
    assert bad == 0, f"{bad} sleeping particles remain inside the kinematic box"


def test_no_particle_left_inside_box_volume(box_sweep):
    d = _sdf_box(box_sweep["pos"], box_sweep["center"])
    # Allow the box's own travel in the last substep plus float slack.
    tol = box_sweep["travel"] + 1e-4
    worst = float(d.min())
    print(f"  deepest particle: {-worst * 1000:.2f} mm inside, "
          f"tolerance {tol * 1000:.2f} mm")
    assert worst >= -tol, (
        f"a particle is {-worst * 1000:.2f} mm inside the box volume")


def test_box_displaces_sand(box_sweep):
    moved = np.linalg.norm(box_sweep["pos"] - box_sweep["pos0"], axis=1)
    # Every particle that started inside the box has been pushed out.
    started_inside = _sdf_box(box_sweep["pos0"], BOX_CENTER) < 0.0
    assert moved[started_inside].min() > 0.0
    assert int((moved > 0.01).sum()) >= box_sweep["inside0"]


def test_world_shrink_wakes_and_clamps_sleepers():
    """A world resize onto a sleeping pile pulls every particle back inside."""
    world, sim = _sleeping_block(world_half_size=0.5)
    new_half = 0.15   # block spans x, z in [-0.2, 0.2]
    sim.set_world_size(new_half)
    n = world._high_water
    pos0 = world.position[:n, :3].get()
    assert (np.abs(pos0[:, 0]) > new_half).any(), "setup: nothing outside"
    sim._sim_step(n)
    cp.cuda.Device().synchronize()
    pos = world.position[:n, :3].get()
    outside = int((np.abs(pos) > new_half + 1e-6).any(axis=1).sum())
    print(f"  particles outside the shrunk box after one substep: {outside}")
    assert outside == 0
