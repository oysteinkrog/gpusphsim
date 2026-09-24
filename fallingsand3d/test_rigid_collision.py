"""Rigid body contact with SDF objects and world walls (bd-r5-review-n28.98).

Drives K_IntegrateRigidBodies + K_RigidBodyCollisions directly with no
fluid. A body must rest with its real surface on the contact:
  - box on an SDF plane: bottom face on the plane (no bounding-sphere gap)
  - box rotated 45 degrees about z: lowest corner on the plane
  - sphere on an SDF plane: surface on the plane
  - box on the world floor / pushed into a world wall: surface stays
    BOUNDARY_MARGIN inside the wall (no fixed 0.05 m COM margin)
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

cp = pytest.importorskip("cupy")

import integrate  # noqa: E402
from rigid_bodies import (  # noqa: E402
    MAX_RIGID_BODIES,
    SDF_BOX,
    SDF_OBJECT_DTYPE,
    SDF_PLANE,
    SDF_SPHERE,
    RigidBodyManager,
    make_sdf_object,
    upload_sdf_objects,
)
from step1 import build_sim_params  # noqa: E402

DT = 1.0e-3
STEPS = 500
TOL = 5.0e-3
BOUNDARY_MARGIN = 1.0e-4
PLANE_Y = -0.5
WORLD_MIN = (-1.0, -1.0, -1.0)
WORLD_MAX = (1.0, 1.0, 1.0)


def _setup(sdf_objects: list) -> None:
    params = build_sim_params(dt=DT, world_min=WORLD_MIN, world_max=WORLD_MAX)
    integrate.upload_sim_params(params)
    if sdf_objects:
        arr = np.zeros(len(sdf_objects), dtype=SDF_OBJECT_DTYPE)
        for i, obj in enumerate(sdf_objects):
            arr[i] = obj
        upload_sdf_objects(arr, len(sdf_objects))
    else:
        upload_sdf_objects(None, 0)


def _plane() -> np.ndarray:
    return make_sdf_object(
        SDF_PLANE,
        position=(0.0, PLANE_Y, 0.0),
        size=(0.0, 1.0, 0.0),
        restitution=0.1,
        friction=0.3,
    )


def _run(
    position,
    half_extents,
    sdf_type=SDF_BOX,
    rotation=None,
    velocity=(0.0, 0.0, 0.0),
    gravity=(0.0, -9.81, 0.0),
    trace=None,
):
    mgr = RigidBodyManager()
    mgr.add_rigid_body(
        position=position,
        half_extents=half_extents,
        mass=1.0,
        sdf_type=sdf_type,
        restitution=0.1,
        friction=0.3,
        velocity=velocity,
    )
    bodies = mgr._bodies_cpu.copy()
    if rotation is not None:
        bodies[0]["rotation"][:] = rotation
    d_bodies = cp.asarray(bodies.view(np.float32))
    d_forces = cp.zeros((MAX_RIGID_BODIES, 4), dtype=cp.float32)
    d_torques = cp.zeros((MAX_RIGID_BODIES, 4), dtype=cp.float32)
    for _ in range(STEPS):
        integrate.integrate_rigid_bodies(d_bodies, d_forces, d_torques, 1, DT, gravity)
        integrate.rigid_body_collisions(d_bodies, 1)
        if trace is not None:
            trace.append(d_bodies[0].copy())  # body 0 position.x
    cp.cuda.Device().synchronize()
    return cp.asnumpy(d_bodies).view(bodies.dtype)[0]


def test_box_rests_face_down_on_sdf_plane():
    _setup([_plane()])
    body = _run((0.0, -0.3, 0.0), (0.06, 0.06, 0.06))
    bottom = float(body["position"][1]) - 0.06
    assert abs(bottom - PLANE_Y) < TOL, f"box bottom {bottom:.4f} vs plane {PLANE_Y}"


def test_rotated_box_corner_rests_on_sdf_plane():
    _setup([_plane()])
    half = math.radians(45.0) / 2.0
    q = (0.0, 0.0, math.sin(half), math.cos(half))
    body = _run((0.0, -0.3, 0.0), (0.06, 0.06, 0.06), rotation=q)
    lowest = float(body["position"][1]) - 0.06 * math.sqrt(2.0)
    assert abs(lowest - PLANE_Y) < TOL, f"lowest corner {lowest:.4f} vs plane {PLANE_Y}"


def test_sphere_rests_on_sdf_plane():
    _setup([_plane()])
    r = 0.08
    body = _run((0.0, -0.3, 0.0), (r, r, r), sdf_type=SDF_SPHERE)
    bottom = float(body["position"][1]) - r
    assert abs(bottom - PLANE_Y) < TOL, f"sphere bottom {bottom:.4f} vs plane {PLANE_Y}"


def test_box_rests_on_world_floor():
    _setup([])
    hy = 0.1
    body = _run((0.0, -0.7, 0.0), (0.1, hy, 0.1))
    bottom = float(body["position"][1]) - hy
    target = WORLD_MIN[1] + BOUNDARY_MARGIN
    assert abs(bottom - target) < 1.0e-3, f"box bottom {bottom:.5f} vs floor {target}"


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_long_box_stays_inside_world_wall(sign):
    _setup([])
    he = (0.15, 0.04, 0.06)  # Submarine hull
    trace = []
    _run(
        (0.5 * sign, 0.0, 0.0),
        he,
        velocity=(3.0 * sign, 0.0, 0.0),
        gravity=(0.0, 0.0, 0.0),
        trace=trace,
    )
    com_x = cp.asnumpy(cp.stack(trace))
    faces = com_x + sign * he[0]
    wall = WORLD_MAX[0] if sign > 0 else WORLD_MIN[0]
    reach = float(np.max(sign * faces))  # farthest the face got toward the wall
    # The face reaches the wall and never goes more than TOL past it.
    assert reach <= sign * wall + TOL, f"face reached {sign * reach:.4f}, wall {wall}"
    assert reach >= sign * wall - TOL, (
        f"face only reached {sign * reach:.4f}, wall {wall}"
    )
