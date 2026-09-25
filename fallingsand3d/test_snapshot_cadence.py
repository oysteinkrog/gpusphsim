"""Snapshot ring cadence and restore wiring (bd-r5-review-n28.85).

- A paused sim (unchanged sim_time) never adds a snapshot.
- Snapshots are spaced in simulated time, so 10x speed gives the same spacing.
- capture() with the last captured sim_time is a no-op.
- after_restore() (shared by Ctrl+Z and the timeline Restore) keeps the ring,
  so a second Restore still reaches older slots.
"""

import contextlib
import types

import cupy as cp

from snapshot import SnapshotRing, after_restore

N = 64


class _World:
    def __init__(self):
        self.position = cp.zeros((N, 4), dtype=cp.float32)
        self.velocity = cp.zeros((N, 4), dtype=cp.float32)
        self.veleval = cp.zeros((N, 4), dtype=cp.float32)
        self.packed_info = cp.full(N, 5, dtype=cp.uint32)
        self.temperature = cp.zeros(N, dtype=cp.float32)
        self.health = cp.zeros(N, dtype=cp.float32)
        self.lifetime = cp.zeros(N, dtype=cp.float32)
        self.mass = cp.zeros(N, dtype=cp.float32)
        self.foam_count = cp.ones(1, dtype=cp.uint32)
        self._high_water = N
        self._spawned_material_ids = set()

    def refresh_color(self, n):
        pass

    def mark(self, v):
        self.position[:, 0] = v


class _Sim:
    def __init__(self):
        self.sim_time = 0.0
        self.calls = []
        self._last_foam_count = 7
        self.rigid_body_manager = types.SimpleNamespace(
            mark_for_reinjection=lambda: self.calls.append("reinject"))

    def _invalidate_graphs(self):
        self.calls.append("graphs")

    def notify_scene_changed(self):
        self.calls.append("scene")

    def reset_spawn_damping(self):
        self.calls.append("damping")

    def copy_to_vbos(self, *bufs):
        self.calls.append("vbo")


class _Renderer:
    def __init__(self):
        self.num_active = 0
        self.cuda_pos = contextlib.nullcontext("pos")
        self.cuda_col = contextlib.nullcontext("col")
        self.cuda_vel = contextlib.nullcontext("vel")


def _run(ring, world, t0, dt, frames):
    """Tick ``frames`` times with sim_time advancing by ``dt`` per frame."""
    t = t0
    for _ in range(frames):
        t += dt
        ring.tick(world, sim_time=t)
    return t


def test_paused_ticks_do_not_capture():
    ring = SnapshotRing(max_snapshots=12, max_particles=N)
    world = _World()
    t = _run(ring, world, 0.0, 1.0 / 60.0, 900)  # 15 s of sim time
    count = ring.num_snapshots
    widx = ring._write_idx
    assert count == 2  # 5 s interval, first tick sets the baseline
    for _ in range(5000):  # paused: sim_time stays put
        ring.tick(world, sim_time=t)
    assert ring.num_snapshots == count
    assert ring._write_idx == widx


def test_spacing_is_in_sim_time_at_any_speed():
    slow = SnapshotRing(max_snapshots=12, max_particles=N)
    fast = SnapshotRing(max_snapshots=12, max_particles=N)
    world = _World()
    # 1x for 1800 frames and 10x for 180 frames both cover 30 s of sim time.
    _run(slow, world, 0.0, 1.0 / 60.0, 1800)
    _run(fast, world, 0.0, 10.0 / 60.0, 180)
    assert slow.num_snapshots == fast.num_snapshots == 5


def test_tick_reads_sim_time_from_sim():
    ring = SnapshotRing(max_snapshots=12, max_particles=N)
    world = _World()
    sim = type("S", (), {"sim_time": 0.0})()
    for _ in range(4000):  # sim never advances
        ring.tick(world, sim=sim)
    assert ring.num_snapshots == 0


def test_capture_with_same_sim_time_is_noop():
    ring = SnapshotRing(max_snapshots=12, max_particles=N)
    world = _World()
    ring.capture(world, sim_time=3.0)
    ring.capture(world, sim_time=3.0)
    assert ring.num_snapshots == 1
    ring.capture(world, sim_time=3.5)
    assert ring.num_snapshots == 2


def test_frame_fallback_without_time():
    ring = SnapshotRing(max_snapshots=12, max_particles=N)
    world = _World()
    for _ in range(600):
        ring.tick(world)
    assert ring.num_snapshots == 2


def test_restore_then_after_restore_keeps_history():
    ring = SnapshotRing(max_snapshots=12, max_particles=N)
    world = _World()
    sim = _Sim()
    renderer = _Renderer()
    for i in range(4):
        world.mark(float(i + 1))
        ring.capture(world, sim_time=float(i))
    world.mark(99.0)

    assert ring.restore(world, steps_back=1)
    after_restore(world, sim, renderer)
    assert ring.num_snapshots == 3
    assert float(world.position[0, 0]) == 4.0
    assert renderer.num_active == N
    assert int(world.foam_count[0]) == 0 and sim._last_foam_count == 0
    assert sim.calls == ["reinject", "graphs", "scene", "damping", "vbo"]

    # A second Restore still reaches older slots.
    assert ring.restore(world, steps_back=2)
    after_restore(world, sim, renderer)
    assert ring.num_snapshots == 1
    assert float(world.position[0, 0]) == 2.0
