"""Torque-free rigid body rotation (bd-r5-review-n28.114).

Drives K_IntegrateRigidBodies directly with no fluid and no gravity.
A Submarine-shaped box (half extents 0.15, 0.04, 0.06) has inertia
Ixx < Izz < Iyy, so z is the intermediate axis.

  - Spun mostly about x: |L| stays at |L0| * damping^N within 0.1 %.
  - Spun about z with a small perturbation: the body-frame omega_z
    reverses sign (intermediate-axis flip).
  - Spun about y (major axis) with a small perturbation: no flip.
  - Cube with a constant torque: omega follows (omega + Iinv*tau*dt)*damp.

The trace is gathered on the GPU and read back once at the end.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

cp = pytest.importorskip("cupy")

import integrate  # noqa: E402
from rigid_bodies import MAX_RIGID_BODIES, SDF_BOX, RigidBodyManager  # noqa: E402
from step1 import build_sim_params  # noqa: E402

DT = 1.0e-3
ANG_DAMP = 1.0 - 0.05 * DT  # hardcoded in K_IntegrateRigidBodies
SUB_HE = (0.15, 0.04, 0.06)
NO_GRAVITY = (0.0, 0.0, 0.0)


def _setup() -> None:
    params = build_sim_params(
        dt=DT, world_min=(-5.0, -5.0, -5.0), world_max=(5.0, 5.0, 5.0)
    )
    integrate.upload_sim_params(params)


def _make_body(half_extents, omega):
    mgr = RigidBodyManager()
    mgr.add_rigid_body(
        position=(0.0, 0.0, 0.0),
        half_extents=half_extents,
        mass=1.0,
        sdf_type=SDF_BOX,
        restitution=0.1,
        friction=0.3,
    )
    bodies = mgr._bodies_cpu.copy()
    bodies[0]["ang_vel"][:3] = omega
    return bodies


def _rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _run(half_extents, omega, steps, torque=None):
    """Return (body dtype, trace of [rotation(4), ang_vel(4)] per step)."""
    bodies = _make_body(half_extents, omega)
    d_bodies = cp.asarray(bodies.view(np.float32))
    d_forces = cp.zeros((MAX_RIGID_BODIES, 4), dtype=cp.float32)
    d_torques = cp.zeros((MAX_RIGID_BODIES, 4), dtype=cp.float32)
    trace = cp.zeros((steps + 1, 8), dtype=cp.float32)
    view = d_bodies.reshape(MAX_RIGID_BODIES, -1)
    rot_off = bodies.dtype.fields["rotation"][1] // 4
    ang_off = bodies.dtype.fields["ang_vel"][1] // 4
    trace[0, :4] = view[0, rot_off : rot_off + 4]
    trace[0, 4:] = view[0, ang_off : ang_off + 4]
    if torque is not None:
        tau = cp.asarray(np.array(torque, dtype=np.float32))
    for i in range(steps):
        if torque is not None:
            d_torques[0, :3] = tau
        integrate.integrate_rigid_bodies(
            d_bodies, d_forces, d_torques, 1, DT, NO_GRAVITY
        )
        trace[i + 1, :4] = view[0, rot_off : rot_off + 4]
        trace[i + 1, 4:] = view[0, ang_off : ang_off + 4]
    tr = cp.asnumpy(trace)
    return bodies, tr


def _body_inertia(bodies):
    return 1.0 / bodies[0]["inertia_inv"][:3].astype(np.float64)


def _world_L(I_b, q, omega):
    R = _rot(q)
    return R @ (I_b * (R.T @ omega[:3]))


def test_submarine_angular_momentum_conserved():
    _setup()
    steps = 1000
    omega0 = (5.0, 0.3, 0.2)  # mostly x, off-axis so the gyro term is non-zero
    bodies, tr = _run(SUB_HE, omega0, steps)
    I_b = _body_inertia(bodies)
    L0 = _world_L(I_b, tr[0, :4], tr[0, 4:].astype(np.float64))
    L1 = _world_L(I_b, tr[-1, :4], tr[-1, 4:].astype(np.float64))
    expected = np.linalg.norm(L0) * ANG_DAMP**steps
    rel = abs(np.linalg.norm(L1) - expected) / expected
    assert rel < 1.0e-3, f"|L| drift {rel:.2e} (|L0|={np.linalg.norm(L0):.5f})"
    # Direction of L is conserved too (no torque).
    cos = float(np.dot(L0, L1) / (np.linalg.norm(L0) * np.linalg.norm(L1)))
    assert cos > 1.0 - 1.0e-4, f"L direction changed, cos={cos:.6f}"


def _body_omega(tr):
    return np.array([_rot(r[:4]).T @ r[4:7] for r in tr.astype(np.float64)])


def test_submarine_intermediate_axis_flips():
    _setup()
    # z is the intermediate axis: Ixx < Izz < Iyy for these half extents.
    _, tr = _run(SUB_HE, (0.01, 0.0, 5.0), 5000)
    wb = _body_omega(tr)
    assert wb[0, 2] > 4.9
    assert wb[:, 2].min() < -2.5, f"no flip, min body omega_z {wb[:, 2].min():.3f}"


def test_submarine_major_axis_stable():
    _setup()
    _, tr = _run(SUB_HE, (0.01, 5.0, 0.0), 5000)
    wb = _body_omega(tr)
    floor = 0.95 * 5.0 * ANG_DAMP**5000  # damping alone gives 5 * 0.778
    assert wb[:, 1].min() > floor, f"major axis spin unstable, min {wb[:, 1].min():.3f}"


def test_cube_matches_isotropic_update():
    _setup()
    steps = 200
    omega0 = np.array([1.0, -2.0, 3.0])
    tau = np.array([0.002, 0.001, -0.003])
    bodies, tr = _run((0.06, 0.06, 0.06), tuple(omega0), steps, torque=tuple(tau))
    iinv = bodies[0]["inertia_inv"][:3].astype(np.float64)
    w = omega0.copy()
    for _ in range(steps):
        w = (w + iinv * tau * DT) * ANG_DAMP
    got = tr[-1, 4:7].astype(np.float64)
    assert np.allclose(got, w, rtol=1.0e-3, atol=1.0e-4), f"got {got}, expected {w}"
