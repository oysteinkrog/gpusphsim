"""VBO copy gate (bd-r5-review-n28.150).

main.py skips the per-frame VBO copy when no substep ran unless
Simulation.scene_dirty is set. notify_scene_changed() and set_world_size()
set it, and copy_to_vbos() clears it. The first tests use a fake sim; the
last one runs a real paused Simulation on the GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

simulation = pytest.importorskip("simulation")
Simulation = simulation.Simulation


class _FakeGLBuffer:
    def __init__(self, n):
        self.host = np.zeros((n, 4), dtype=np.float32)
        self.nbytes = self.host.nbytes

    def device_pointer_as_cupy_array(self, shape, dtype):
        return self.host


def _fake_sim(high_water=10, capacity=32):
    world = SimpleNamespace(
        _high_water=high_water,
        position=np.ones((capacity, 4), dtype=np.float32),
        velocity=np.full((capacity, 4), 2.0, dtype=np.float32),
    )

    def compute_render_color(out, n=None):
        out[:n] = 3.0

    return SimpleNamespace(
        world=world, scene_dirty=True, _readback_epoch=0,
        _MAX_DISP_UNKNOWN=Simulation._MAX_DISP_UNKNOWN, compute_render_color=compute_render_color,
    )


def test_copy_to_vbos_clears_scene_dirty():
    sim = _fake_sim()
    pos, col, vel = _FakeGLBuffer(32), _FakeGLBuffer(32), _FakeGLBuffer(32)
    Simulation.copy_to_vbos(sim, pos, col, vel)
    assert sim.scene_dirty is False
    assert np.all(pos.host[:10] == 1.0)
    assert np.all(col.host[:10] == 3.0)
    assert np.all(vel.host[:10] == 2.0)


def test_copy_to_vbos_clears_scene_dirty_when_empty():
    sim = _fake_sim(high_water=0)
    Simulation.copy_to_vbos(sim, _FakeGLBuffer(4), _FakeGLBuffer(4))
    assert sim.scene_dirty is False


def test_notify_scene_changed_sets_scene_dirty():
    sim = _fake_sim()
    sim.scene_dirty = False
    Simulation.notify_scene_changed(sim)
    assert sim.scene_dirty is True


def test_paused_frames_stay_clean_until_brush_spawn():
    """Paused frames leave scene_dirty False; a spawn sets it for one copy."""
    import cupy as cp
    import time
    from materials import SAND, WATER
    from world import World

    w = World(max_particles=20_000)
    w.spawn_cube(min_corner=(-0.1, -0.9, -0.1), max_corner=(0.1, -0.8, 0.1),
                 material_id=SAND, spacing=0.02)
    sim = Simulation(w, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=True,
                     max_substeps=2)
    try:
        assert sim.scene_dirty  # a new sim has never filled the VBOs
        n0 = w._high_water
        pos, col = cp.zeros((20_000, 4), cp.float32), cp.zeros((20_000, 4), cp.float32)

        class _Buf:
            def __init__(self, arr):
                self.arr = arr
                self.nbytes = arr.nbytes

            def device_pointer_as_cupy_array(self, shape, dtype):
                return self.arr

        sim.copy_to_vbos(_Buf(pos), _Buf(col))
        assert not sim.scene_dirty

        sim.toggle_pause()
        sim.step_frame()  # sets the clock
        for _ in range(5):
            sim._last_frame_time = time.perf_counter() - 1.0
            assert sim.step_frame() == 0
            assert not sim.scene_dirty  # main.py skips the copy

        # Brush spawn while paused: main.py calls notify_scene_changed.
        added = w.spawn_sphere(center=(0.0, 0.0, 0.0), radius=0.05,
                               material_id=WATER, count=50)
        assert added > 0
        sim.notify_scene_changed()
        assert sim.scene_dirty
        sim.copy_to_vbos(_Buf(pos), _Buf(col))
        assert not sim.scene_dirty
        cp.cuda.Device().synchronize()
        n1 = w._high_water
        assert n1 == n0 + added
        np.testing.assert_array_equal(cp.asnumpy(pos[:n1]),
                                      cp.asnumpy(w.position[:n1]))
    finally:
        sim.close()
