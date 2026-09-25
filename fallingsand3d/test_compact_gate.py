"""Compaction sync gate (bd-r5-review-n28.135).

_maybe_compact must not call World.num_active (a reduction plus a blocking
device-to-host read) on an interval frame when the lagged freelist size
from the async readback shows that no compaction and no reserve refill is
due. When enough particles die, it must still count them and compact.
"""

from __future__ import annotations

import time

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
from materials import SAND, STATIC, WATER  # noqa: E402
from simulation import Simulation  # noqa: E402
from world import World  # noqa: E402

MAX_PARTICLES = 30_000
SUBSTEPS = 2
DT = 0.001
DEAD_PACKED = STATIC << 8  # material DEAD (0), behavior STATIC


@pytest.fixture
def counted(monkeypatch):
    """Count World.num_active calls (test code reads it only via _raw)."""
    calls = {"n": 0}
    orig = World.num_active

    def _counted(self):
        calls["n"] += 1
        return orig.fget(self)

    monkeypatch.setattr(World, "num_active", property(_counted))
    calls["raw"] = orig.fget
    return calls


def _make_sim(material, lo, hi):
    w = World(max_particles=MAX_PARTICLES)
    w.spawn_cube(min_corner=lo, max_corner=hi, material_id=material, spacing=0.02)
    s = Simulation(w, dt=DT, speed=1.0, accuracy=0.4, fixed_dt=True,
                   max_substeps=SUBSTEPS)
    s.fixed_dt_value = DT
    s._upload_dt(DT)
    s._spawn_substep = s._damping_duration
    s.notify_scene_changed()
    return w, s


def _frame(sim):
    """One step_frame with exactly SUBSTEPS substeps."""
    if sim._last_frame_time is None:
        sim.step_frame()  # first call only sets the clock
    sim._last_frame_time = time.perf_counter() - 1.0
    assert sim.step_frame() == SUBSTEPS
    # Test-only sync, standing in for the render step that keeps the host
    # within a frame of the GPU, so each copy lands before the next latch.
    cp.cuda.Device().synchronize()


def test_no_deaths_no_sync_without_reserve(counted):
    """Sand only (no reserve): 200 frames, 3 interval frames, 0 num_active calls."""
    w, sim = _make_sim(SAND, (-0.2, -0.9, -0.2), (0.2, -0.7, 0.2))
    try:
        assert not w.needs_dead_reserve
        for _ in range(200):
            _frame(sim)
        assert sim._frame_counter == 200
        assert counted["n"] == 0
        assert sim._num_active_syncs == 0
        assert sim._last_dead_count == 0
    finally:
        sim.close()


def test_no_deaths_no_sync_with_reserve(counted):
    """Water (parked DEAD reserve): after the startup refill, 0 calls in 200 frames."""
    w, sim = _make_sim(WATER, (-0.2, -0.9, -0.2), (0.2, -0.7, 0.2))
    try:
        assert w.needs_dead_reserve
        # Frame 1 parks the reserve (one exact count after the scene change);
        # frame 2 re-checks it once after the notify that parking triggers.
        for _ in range(2):
            _frame(sim)
        startup = counted["n"]
        assert startup <= 2
        for _ in range(200):
            _frame(sim)
        assert counted["n"] == startup
        assert sim._last_dead_count == w.dead_reserve
    finally:
        sim.close()


def test_few_deaths_no_sync(counted):
    """Deaths below compact_threshold: interval frames still skip the count."""
    w, sim = _make_sim(SAND, (-0.2, -0.9, -0.2), (0.2, -0.7, 0.2))
    try:
        _frame(sim)
        n0 = w._high_water
        k = sim.compact_threshold // 2
        w.packed_info[:k] = DEAD_PACKED
        w.mass[:k] = 0.0
        for _ in range(sim.compact_interval + 1):
            _frame(sim)
        assert sim._last_dead_count == k
        assert counted["n"] == 0
        assert w._high_water == n0
    finally:
        sim.close()


def test_many_deaths_sync_once_and_compact(counted):
    """200 dead particles: one exact count on the interval frame, then compaction."""
    w, sim = _make_sim(SAND, (-0.2, -0.9, -0.2), (0.2, -0.7, 0.2))
    try:
        _frame(sim)
        n0 = w._high_water
        k = 200
        assert k >= sim.compact_threshold
        # Kill outside the substep loop, like the brush, without a notify.
        w.packed_info[:k] = DEAD_PACKED
        w.mass[:k] = 0.0
        while sim._frame_counter < sim.compact_interval:
            _frame(sim)
        assert counted["n"] == 1
        assert w._high_water == n0 - k
        assert counted["raw"](w) == n0 - k
        # The compaction's scene change makes the lagged value unknown; the
        # next copy shows no deaths, so the following interval skips.
        while sim._frame_counter < 2 * sim.compact_interval:
            _frame(sim)
        assert counted["n"] == 1
        assert sim._last_dead_count == 0
    finally:
        sim.close()
