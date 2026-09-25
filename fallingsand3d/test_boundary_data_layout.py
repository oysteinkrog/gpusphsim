"""K_UpdateBoundaryParticles reads boundary_data correctly (bd-r5-review-n28.183).

The kernel loads each boundary record's (x_local, y_local, z_local, psi) as
one float4 and the body id as one float. These GPU tests check its output
against a float64 NumPy reference for two rotating, moving bodies, and that
the dead normal columns are no longer uploaded.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")


def _quat_rotate(q, v):
    """quat_rotate from common.cuh in float64: q * v * q^-1, q = (x, y, z, w)."""
    u = q[:3]
    s = q[3]
    return (2.0 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v
            + 2.0 * s * np.cross(u, v))


def _make_manager():
    from rigid_bodies import SDF_SPHERE, RigidBodyManager

    rbm = RigidBodyManager()
    rbm.add_rigid_body_with_particles(
        position=(0.1, 0.3, -0.2), half_extents=(0.06, 0.04, 0.05), mass=2.0)
    rbm.add_rigid_body_with_particles(
        position=(-0.2, 0.5, 0.1), half_extents=(0.05, 0.05, 0.05), mass=1.0,
        sdf_type=SDF_SPHERE)
    rbm.finalize_boundary_data()
    # Give both bodies a rotation and a linear and angular velocity, so every
    # term of the transform and of v_body + omega x r is exercised.
    rots = [(0.1, 0.3, -0.2), (-0.4, 0.2, 0.5)]
    for k, axis in enumerate(rots):
        a = np.asarray(axis, dtype=np.float64)
        ang = np.linalg.norm(a)
        q = np.concatenate([np.sin(ang / 2) * a / ang, [np.cos(ang / 2)]])
        rbm._bodies_cpu[k]["rotation"] = q.astype(np.float32)
        rbm._bodies_cpu[k]["lin_vel"][:3] = (0.3 * (k + 1), -0.5, 0.2)
        rbm._bodies_cpu[k]["ang_vel"][:3] = (1.5, -2.0 * (k + 1), 0.7)
    import integrate
    rbm.upload([integrate.get_module()])
    return rbm


def _run_kernel(rbm, offset=7):
    import integrate

    nb = rbm.num_boundary_particles
    m = offset + nb + 5
    pos = cp.zeros((m, 4), dtype=cp.float32)
    vel = cp.zeros((m, 4), dtype=cp.float32)
    mass = cp.zeros(m, dtype=cp.float32)
    integrate.update_boundary_particles(
        rbm.boundary_data, rbm.d_rigid_bodies, pos, vel, mass,
        offset=offset, num_boundary=nb)
    cp.cuda.Device().synchronize()
    return pos.get(), vel.get(), mass.get()


def test_boundary_positions_velocities_and_psi_match_reference():
    rbm = _make_manager()
    offset = 7
    pos, vel, mass = _run_kernel(rbm, offset)
    nb = rbm.num_boundary_particles
    bd = rbm.boundary_data.get().astype(np.float64)
    bodies = rbm._bodies_cpu

    ref_pos = np.zeros((nb, 3))
    ref_vel = np.zeros((nb, 3))
    for i in range(nb):
        b = bodies[int(round(bd[i, 4]))]
        r = _quat_rotate(b["rotation"].astype(np.float64), bd[i, :3])
        ref_pos[i] = b["position"][:3] + r
        ref_vel[i] = b["lin_vel"][:3] + np.cross(b["ang_vel"][:3].astype(np.float64), r)

    blk = slice(offset, offset + nb)
    np.testing.assert_allclose(pos[blk, :3], ref_pos, rtol=0, atol=2e-6)
    np.testing.assert_allclose(vel[blk, :3], ref_vel, rtol=0, atol=2e-6)
    assert np.all(pos[blk, 3] == 2500.0)
    assert np.all(vel[blk, 3] == 0.0)
    # psi goes to mass unchanged.
    np.testing.assert_array_equal(mass[blk], bd[:, 3].astype(np.float32))
    # Slots outside the block stay untouched.
    assert not pos[:offset].any() and not pos[offset + nb:].any()
    assert not mass[:offset].any() and not mass[offset + nb:].any()


def test_boundary_data_has_no_normals():
    """Columns 5-7 used to hold surface normals that no kernel read."""
    rbm = _make_manager()
    bd = rbm.boundary_data.get()
    assert bd.shape == (rbm.num_boundary_particles, 8)
    assert bd.dtype == np.float32
    assert np.all(bd[:, 5:8] == 0.0)
    ids = bd[:, 4]
    assert set(np.unique(ids).tolist()) == {0.0, 1.0}
    # Records are grouped by body, in body order.
    assert np.all(np.diff(ids) >= 0.0)
