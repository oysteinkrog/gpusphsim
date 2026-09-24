"""bd-r5-review-n28.78: load_scene notifies the simulation once it replaces
the particles, and a bad header fails before any world array changes.

A corrupt optional block (rigid bodies, camera) must not leave a half-loaded
world that skips the full sort. It loads the particles, records a warning,
and the next substep does a full counting sort.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

try:
    import cupy as cp
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - CPU-only box
    _HAS_GPU = False

needs_gpu = pytest.mark.skipif(not _HAS_GPU, reason="needs CuPy and a CUDA GPU")


def _camera():
    return types.SimpleNamespace(target=np.zeros(3, dtype=np.float32),
                                 distance=2.0, azimuth=0.0, elevation=0.3)


def _read(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _write(path: str, arrays: dict, header: dict) -> None:
    arrays = dict(arrays)
    arrays["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)


class _RecordingRBM:
    def __init__(self) -> None:
        self.max_bodies = 8
        self.resets = 0
        self.reinjections = 0

    def reset(self) -> None:
        self.resets += 1

    def upload(self, modules) -> None:
        pass

    def mark_for_reinjection(self) -> None:
        self.reinjections += 1


class _RecordingSim:
    """Just enough of Simulation for load_scene; counts the notify calls."""

    def __init__(self) -> None:
        self._profile = types.SimpleNamespace(name="WCSPH")
        self.world_half_size = 1.0
        self.gravity_y = -9.81
        self.sim_time = 0.0
        self._last_frame_time = None
        self.rigid_body_manager = _RecordingRBM()
        self.notify_calls = 0
        self.invalidate_calls = 0

    def set_world_size(self, v) -> None:
        self.world_half_size = v

    def set_gravity(self, g) -> None:
        self.gravity_y = g

    def reset_spawn_damping(self) -> None:
        pass

    def get_all_modules(self) -> list:
        return []

    def notify_scene_changed(self) -> None:
        self.notify_calls += 1

    def _invalidate_graphs(self) -> None:
        self.invalidate_calls += 1


def _corrupt_rigid(header: dict) -> dict:
    # One v4 body whose "build" block lost its keys: the rebuild raises KeyError.
    header = dict(header)
    header["rigid_bodies"] = {"count": 1, "bodies": [{"position": [0, 0, 0], "build": {}}]}
    return header


@needs_gpu
def test_corrupt_rigid_block_loads_particles_and_notifies(tmp_path, monkeypatch) -> None:
    import save_load as sl
    from materials import WATER
    from world import World

    monkeypatch.setattr(sl, "SAVE_DIR", str(tmp_path))
    world = World(10_000)
    n = world.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    sim = _RecordingSim()
    sl.save_scene(world, sim, _camera(), "good")
    arrays = _read(str(tmp_path / "good.fs3d"))
    header = json.loads(arrays.pop("header").tobytes())
    header["camera"] = {"target": [0, 0]}  # also a broken camera
    _write(str(tmp_path / "bad_rigid.fs3d"), arrays, _corrupt_rigid(header))

    # Put the world in a different state so the load has to change it.
    world.foam_count.fill(5)
    world.spawn_sphere((0.0, 0.4, 0.0), 0.05, WATER, 50)
    assert world._high_water != n

    cam = _camera()
    cam.distance = 9.0
    report: dict = {}
    loaded = sl.load_scene(world, sim, cam, "bad_rigid", report)

    assert loaded == n
    assert world._high_water == n
    assert report["world_modified"] is True
    assert any("Rigid bodies" in w for w in report["warnings"])
    assert any("Camera" in w for w in report["warnings"])
    assert sim.notify_calls == 1
    assert sim.invalidate_calls == 1
    assert int(world.foam_count.get().sum()) == 0
    assert sim.rigid_body_manager.resets == 1  # half-built manager cleared
    assert cam.distance == 9.0  # broken camera left alone


def test_bad_header_value_fails_before_world_write(tmp_path, monkeypatch) -> None:
    """A bad world_half_size raises before any world field changes (CPU only)."""
    import save_load as sl
    from materials import WATER

    monkeypatch.setattr(sl, "SAVE_DIR", str(tmp_path))
    n = 8
    arrays = {
        "position": np.zeros((n, 4), np.float32),
        "velocity": np.zeros((n, 4), np.float32),
        "packed_info": np.full(n, WATER, np.uint32),
        "temperature": np.zeros(n, np.float32),
        "health": np.ones(n, np.float32),
        "lifetime": np.zeros(n, np.float32),
        "mass": np.ones(n, np.float32),
        "particle_dye": np.zeros((n, 4), np.float32),
        "density": np.ones(n, np.float32),
    }
    for bad in ("big", -1.0, float("nan"), None):
        header = {"version": 4, "particle_count": n, "world_half_size": bad,
                  "sdf": {}, "rigid_bodies": {}}
        _write(str(tmp_path / "bad.fs3d"), arrays, header)
        world = types.SimpleNamespace(
            position=np.full((16, 4), 7.0, np.float32),
            packed_info=np.full(16, 5, np.uint32),
            _high_water=16,
        )
        sim = _RecordingSim()
        report: dict = {}
        with pytest.raises(ValueError, match="world_half_size"):
            sl.load_scene(world, sim, _camera(), "bad.fs3d", report)
        assert report["world_modified"] is False
        assert world._high_water == 16
        assert np.all(world.packed_info == 5)
        assert np.all(world.position == 7.0)
        assert sim.notify_calls == 0


@needs_gpu
def test_corrupt_rigid_block_next_substep_full_sorts(tmp_path, monkeypatch) -> None:
    """With a real Simulation the next substep after the load does a full sort."""
    import save_load as sl
    from materials import WATER
    from simulation import Simulation
    from world import World

    monkeypatch.setattr(sl, "SAVE_DIR", str(tmp_path))
    world = World(60_000)
    world.spawn_cube((-0.3, -0.3, -0.3), (0.3, 0.0, 0.3), WATER, 0.03)
    sim = Simulation(world, dt=0.001, fixed_dt=True)
    for _ in range(2):
        sim._sim_step(world._high_water)
    n = world._high_water
    sl.save_scene(world, sim, _camera(), "pool")
    arrays = _read(str(tmp_path / "pool.fs3d"))
    header = json.loads(arrays.pop("header").tobytes())
    _write(str(tmp_path / "pool_bad.fs3d"), arrays, _corrupt_rigid(header))

    report: dict = {}
    loaded = sl.load_scene(world, sim, _camera(), "pool_bad", report)
    assert loaded == n
    assert any("Rigid bodies" in w for w in report["warnings"])
    assert sim._last_full_sort_n == -1
    assert sim._sort_skip_next is False
    assert sim._graph_full_sort is None and sim._graph_gather_only is None

    sim._sim_step(world._high_water)
    cp.cuda.Device().synchronize()
    assert sim._last_full_sort_n == n
    assert sim.rigid_body_manager.num_bodies == 0
