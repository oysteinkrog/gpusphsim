"""Velocity CFL for adaptive dt and v_max substep split (bd-r5-review-n28.143).

The host estimates v_max from the drift readback it already latches each
frame. WCSPH adaptive dt shrinks with it; PBF/DFSPH split fixed_dt into more
substeps so the velocity clamp rises with the speed.
"""

import math
import struct

import numpy as np
import pytest

try:
    import cupy.cuda.compiler as _compiler
    _compiler._use_ptx = True
except Exception:
    pass

import cupy as cp

from materials import GAS, MATERIALS, STATIC, WATER
from simulation import (
    CFL_NUMBER, DT_GROWTH_PER_FRAME, FIXED_SUBDIV_MAX, Simulation,
)
from solver_profiles import PROFILES
from world import World

MAX_PARTICLES = 100_000


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


def _make_sim(world, profile="WCSPH", clock=None):
    world.packed_info[:] = 0
    world._high_water = 0
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=False, max_substeps=20, clock=clock)
    sim.set_solver_profile(PROFILES[profile])
    # Accuracy 0.4 puts the cap at dt_max (1 ms), so 10 m/s binds.
    sim.accuracy = 0.4
    return sim


def _bits(x):
    return struct.unpack('I', struct.pack('f', x))[0]


def test_fast_v_max_gives_smaller_dt(world):
    sim = _make_sim(world)
    try:
        dt_slow = sim._compute_adaptive_dt(1, v_max=0.1)
        dt_fast = sim._compute_adaptive_dt(1, v_max=10.0)
        assert dt_fast < dt_slow
        assert dt_fast == pytest.approx(
            CFL_NUMBER * sim._h / (sim._c_sound + 10.0))
        # A calm scene sits at the velocity-free cap.
        assert dt_slow == pytest.approx(sim._adaptive_dt_cap())
    finally:
        sim.close()


def test_c_sound_excludes_gas(world):
    sim = _make_sim(world)
    try:
        expected = max(
            math.sqrt(m.eos_stiffness * m.eos_gamma / m.rest_density)
            for m in MATERIALS.values()
            if m.behavior_class not in (STATIC, GAS)
            and m.rest_density > 0 and m.eos_stiffness > 0 and m.eos_gamma > 0)
        assert sim._c_sound == pytest.approx(expected)
    finally:
        sim.close()


def test_dt_shrinks_at_once_and_grows_at_most_10_percent(world):
    sim = _make_sim(world)
    try:
        cap = sim._adaptive_dt_cap()
        sim.dt = 0.5 * cap
        sim._v_max_lagged = 0.0
        assert sim._next_adaptive_dt(1) == pytest.approx(
            0.5 * cap * DT_GROWTH_PER_FRAME)
        sim.dt = cap
        sim._v_max_lagged = 40.0
        target = sim._compute_adaptive_dt(1)
        assert target < 0.5 * cap
        assert sim._next_adaptive_dt(1) == pytest.approx(target)
    finally:
        sim.close()


def test_latch_v_max_from_drift(world):
    sim = _make_sim(world)
    try:
        # 1 cm drift over 2 ms since the last full sort: 5 m/s.
        sim._last_max_disp_uint = _bits(1e-4)
        sim._last_disp_sim_time = 0.002
        sim._latch_v_max()
        assert sim._v_max_lagged == pytest.approx(5.0, rel=1e-5)
        # Unknown or teleport sentinel: keep the previous estimate.
        sim._last_max_disp_uint = Simulation._MAX_DISP_UNKNOWN
        sim._latch_v_max()
        assert sim._v_max_lagged == pytest.approx(5.0, rel=1e-5)
        sim._last_max_disp_uint = 0x7F800000
        sim._latch_v_max()
        assert sim._v_max_lagged == pytest.approx(5.0, rel=1e-5)
        # No drift at all: at rest.
        sim._last_max_disp_uint = 0
        sim._latch_v_max()
        assert sim._v_max_lagged == 0.0
    finally:
        sim.close()


def test_fixed_subdiv_grows_at_once_and_falls_by_one(world):
    sim = _make_sim(world, "PBF")
    try:
        vlf = sim._profile.velocity_limit_factor
        v_cap = vlf * sim._h / sim.fixed_dt_value  # 4.32 m/s at 1/120
        sim._v_max_lagged = 0.1
        assert sim._next_fixed_subdiv() == 1
        sim._v_max_lagged = 1.4 * v_cap          # 2.8 margins: n = 3
        assert sim._next_fixed_subdiv() == 3
        sim._fixed_subdiv = 3
        sim._v_max_lagged = 0.0
        assert sim._next_fixed_subdiv() == 2
        sim._v_max_lagged = 1e6
        assert sim._next_fixed_subdiv() == FIXED_SUBDIV_MAX
    finally:
        sim.close()


def test_pbf_drop_exceeds_fixed_dt_velocity_clamp(world):
    """A PBF water block in free fall passes the 1/120 s clamp (4.32 m/s)."""
    t = [0.0]
    sim = _make_sim(world, "PBF", clock=lambda: t[0])
    try:
        world.spawn_cube(min_corner=(-0.2, 0.3, -0.2),
                         max_corner=(0.2, 0.5, 0.2),
                         material_id=WATER, spacing=0.02)
        sim.notify_scene_changed()
        sim._spawn_substep = sim._damping_duration
        v_cap = (sim._profile.velocity_limit_factor * sim._h
                 / sim.fixed_dt_value)
        sim.step_frame()  # first call only sets the clock
        peak = 0.0
        max_subdiv = 1
        for _ in range(60):
            t[0] += 1.0 / 60.0
            sim.step_frame()
            max_subdiv = max(max_subdiv, sim._fixed_subdiv)
            # Test-only readback (host sync is fine here).
            v = world.velocity[:world._high_water, :3]
            peak = max(peak, float(cp.sqrt((v * v).sum(axis=1)).max()))
        assert max_subdiv > 1
        assert peak > 1.1 * v_cap
        assert np.isfinite(peak)
    finally:
        cp.cuda.Device().synchronize()
        sim.close()
