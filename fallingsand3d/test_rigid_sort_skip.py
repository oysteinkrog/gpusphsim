"""Rigid body motion reaches the grid-reuse tracker (bd-r5-review-n28.65).

Rigid boundary particles are STATIC, so K_Integrate and the PBF/DFSPH
finalize kernels never record their motion. K_UpdateBoundaryParticles keeps
a per-particle anchor (position at the last full sort) and feeds the drift
since then into max_displacement. A moving body must force a full sort; a
resting body must not stop sort skipping.
"""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, '_cache'):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, '_get_arch_for_options_for_nvrtc'):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, '_cache'):
                fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402
from materials import WATER  # noqa: E402
from simulation import Simulation  # noqa: E402
from world import World  # noqa: E402

MAX_PARTICLES = 20_000
DT = 0.001
SUBSTEPS = 10


def _make_sim(body_velocity):
    world = World(max_particles=MAX_PARTICLES)
    sim = Simulation(world, dt=DT, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=SUBSTEPS)
    sim.fixed_dt_value = DT
    sim._upload_dt(DT)
    sim._spawn_substep = sim._damping_duration
    sim.set_gravity(0.0)
    # Calm water block, well away from the body so they do not interact.
    world.spawn_cube(min_corner=(-0.6, -0.1, -0.1), max_corner=(-0.4, 0.1, 0.1),
                     material_id=WATER, spacing=0.02)
    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(
        position=(0.3, 0.0, 0.0),
        half_extents=(0.05, 0.05, 0.05),
        mass=2.0,
        velocity=body_velocity,
    )
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    sim.notify_scene_changed()
    return world, sim, rbm


def _disp_sq(world):
    # Test-only readback (host sync is fine here).
    bits = int(world.max_displacement.get()[0])
    return struct.unpack('f', struct.pack('I', bits))[0]


def _frame(sim):
    if sim._last_frame_time is None:
        sim.step_frame()  # first call only sets the clock
    sim._last_frame_time = time.perf_counter() - 1.0
    assert sim.step_frame() == SUBSTEPS
    cp.cuda.Device().synchronize()


def test_boundary_drift_accumulates_over_gather_substeps():
    """A body at 1 m/s drifts about k mm after k substeps since the sort."""
    world, sim, rbm = _make_sim((1.0, 0.0, 0.0))
    try:
        sim._sim_step(world._high_water)   # injects the block, full sort
        cp.cuda.Device().synchronize()
        assert rbm.boundary_anchor_valid
        off, nb = rbm.boundary_offset, rbm.num_boundary_particles
        np.testing.assert_array_equal(
            rbm.boundary_anchor[:, :3].get(),
            world.position[off:off + nb, :3].get())
        n = world._high_water
        sim._sort_skip_next = True
        for _ in range(9):
            sim._sim_step(n)               # gather only
        cp.cuda.Device().synchronize()
        assert sim._sort_skip_next         # nothing forced a sort
        drift = _disp_sq(world)
        # The anchor was written on the first substep, so the body has moved
        # 9 substeps since then. Before the fix this stayed near zero.
        assert drift >= 0.8 * (9 * DT * 1.0) ** 2, drift
    finally:
        sim._sort_skip_next = False
        sim.close()


def test_missing_anchor_forces_full_sort():
    """A freshly built block has no anchor, so the substep must sort."""
    world, sim, rbm = _make_sim((0.0, 0.0, 0.0))
    try:
        sim._sim_step(world._high_water)
        n = world._high_water
        rbm.boundary_anchor_valid = False
        sim._sort_skip_next = True
        skipped = sim._sort_skipped_count
        sim._sim_step(n)
        cp.cuda.Device().synchronize()
        assert sim._sort_skipped_count == skipped
        assert rbm.boundary_anchor_valid
    finally:
        sim._sort_skip_next = False
        sim.close()


def test_moving_body_forces_full_sort_every_frame():
    """Body at 2 m/s through calm water: 20 mm per frame, no gather frames."""
    world, sim, rbm = _make_sim((2.0, 0.0, 0.0))
    try:
        for _ in range(3):
            _frame(sim)                     # warm up the async readback
        skipped = sim._sort_skipped_count
        for _ in range(8):
            _frame(sim)
            assert not sim._sort_skip_next
        assert sim._sort_skipped_count == skipped
        vel = rbm.download_bodies()["lin_vel"][0, 0]
        assert vel == pytest.approx(2.0, rel=0.05)   # no fluid contact
    finally:
        sim.close()


def test_resting_body_keeps_sort_skipping():
    """A body at rest adds no drift, so calm water still skips sorts."""
    world, sim, rbm = _make_sim((0.0, 0.0, 0.0))
    try:
        for _ in range(3):
            _frame(sim)
        skipped = sim._sort_skipped_count
        for _ in range(10):
            _frame(sim)
        assert sim._sort_skipped_count > skipped
    finally:
        sim.close()
