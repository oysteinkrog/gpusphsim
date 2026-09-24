"""A gather-only CUDA graph never captures a full sort (bd-r5-review-n28.84).

A full-sort graph replay skips the host bookkeeping in _run_grid_setup. After
a same-n notify_scene_changed (kill brush) the host value _last_full_sort_n
stayed -1, so the next gather-only capture took the forced full-sort branch
and every later "gather" replay ran a full counting sort. The trigger needs
frames with exactly one substep, so this test drives _sim_step by hand.
"""

from __future__ import annotations

import struct

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
import counting_sort  # noqa: E402
import simulation as simulation_mod  # noqa: E402
from materials import WATER  # noqa: E402
from simulation import Simulation  # noqa: E402
from world import World  # noqa: E402

DT = 0.001
SENTINEL = 1.0  # squared metres; far above any drift in a few substeps


def _float_bits(x: float) -> int:
    return struct.unpack('I', struct.pack('f', x))[0]


def _disp_sq(world) -> float:
    # Test-only readback (host sync is fine here).
    bits = int(world.max_displacement.get()[0])
    return struct.unpack('f', struct.pack('I', bits))[0]


@pytest.fixture()
def sim_and_world():
    world = World(max_particles=20_000)
    sim = Simulation(world, dt=DT, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=1)
    sim.fixed_dt_value = DT
    sim._upload_dt(DT)
    sim._spawn_substep = sim._damping_duration
    world.spawn_cube(min_corner=(-0.2, -0.1, -0.1), max_corner=(0.2, 0.1, 0.1),
                     material_id=WATER, spacing=0.02)
    sim.notify_scene_changed()
    try:
        yield sim, world
    finally:
        sim._sort_skip_next = False
        sim.close()


def _step(sim, n, skip):
    sim._sort_skip_next = skip
    sim._sim_step(n)


def test_gather_graph_does_not_bake_in_full_sort(sim_and_world, monkeypatch):
    sim, world = sim_and_world
    n = world._high_water
    assert n > 0
    assert sim.rigid_body_manager.num_bodies == 0  # graphs enabled

    # One-substep frames: full direct run, full capture, gather direct run.
    _step(sim, n, False)
    _step(sim, n, False)
    assert sim._graph_full_sort is not None
    _step(sim, n, True)
    assert sim._graph_gather_only is None
    assert sim._graph_gather_only_n == n

    # Kill brush: same n, scene changed.
    world.kill_in_sphere((0.15, 0.0, 0.0), 0.03)
    sim.notify_scene_changed()
    assert world._high_water == n

    # Full-sort graph replay must record the sort for n.
    _step(sim, n, False)
    assert sim._last_full_sort_n == n

    # The gather capture must not run a full counting sort.
    calls = []
    real_full = counting_sort.counting_sort_full

    def counted(*args, **kwargs):
        calls.append(1)
        return real_full(*args, **kwargs)

    monkeypatch.setattr(simulation_mod.counting_sort, 'counting_sort_full', counted)
    _step(sim, n, True)
    assert sim._graph_gather_only is not None
    assert calls == [], "gather-only capture ran a full counting sort"

    # Five gather replays: a full sort would memset max_displacement to 0.
    world.max_displacement.fill(_float_bits(SENTINEL))
    skipped_before = sim._sort_skipped_count
    for _ in range(5):
        _step(sim, n, True)
    cp.cuda.Device().synchronize()
    assert calls == []
    assert sim._sort_skipped_count == skipped_before  # replays skip host code
    assert _disp_sq(world) >= SENTINEL, "gather replay ran a full sort"


def test_gather_capture_refused_when_last_sort_stale(sim_and_world):
    """Guard: with a stale _last_full_sort_n the gather slot runs directly."""
    sim, world = sim_and_world
    n = world._high_water
    _step(sim, n, False)
    _step(sim, n, True)           # gather slot primed (graph_n = n)
    sim._last_full_sort_n = -1    # stale, as before the replay fix
    _step(sim, n, True)
    assert sim._graph_gather_only is None
    assert sim._last_full_sort_n == n  # the direct run did a full sort
    _step(sim, n, True)           # now a real gather capture
    assert sim._graph_gather_only is not None
    assert sim._last_setup_was_full_sort is False
