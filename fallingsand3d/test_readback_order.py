"""Async readback order for the grid-reuse decision (bd-r5-review-n28.46).

The sort-skip decision at the end of frame k must use the max_displacement
copied at the end of frame k-1, latch it only once the copy has landed,
and the next frame's kernels must wait for that copy on the device.
"""

from __future__ import annotations

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
import foam  # noqa: E402
import spawn  # noqa: E402
from materials import (  # noqa: E402
    DEAD, FLUID, STATIC, STONE, WATER, build_material_array,
)
from simulation import Simulation  # noqa: E402
from step1 import build_sim_params  # noqa: E402
from world import World  # noqa: E402

MAX_PARTICLES = 20_000
SUBSTEPS = 4
DT = 0.001


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


def _make_sim(world):
    world.packed_info[:] = 0
    world._high_water = 0
    world.foam_count[:] = 0
    s = Simulation(world, dt=DT, speed=1.0, accuracy=0.4,
                   fixed_dt=True, max_substeps=SUBSTEPS)
    s.fixed_dt_value = DT
    s._upload_dt(DT)
    # No spawn damping ramp: it would bleed off the test velocities.
    s._spawn_substep = s._damping_duration
    return s


def _frame(sim):
    """Run one step_frame with exactly SUBSTEPS substeps."""
    if sim._last_frame_time is None:
        sim.step_frame()  # first call only sets the clock
    sim._last_frame_time = time.perf_counter() - 1.0
    steps = sim.step_frame()
    assert steps == SUBSTEPS
    # Test-only sync. The app's render step (VBO map, buffer swap) keeps the
    # host within a frame of the GPU; without it the host can run frames
    # ahead, the readback is not done at the next latch, and the decision
    # rightly refuses to skip.
    cp.cuda.Device().synchronize()
    return steps


def _actual_disp_uint(world):
    # Test-only readback (host sync is fine here).
    return int(world.max_displacement.get()[0])


def test_startup_sentinel_and_first_frames_sort(world):
    """Before any readback the displacement is unknown and never skips."""
    sim = _make_sim(world)
    try:
        assert sim._last_max_disp_uint == Simulation._MAX_DISP_UNKNOWN
        world.spawn_cube(min_corner=(-0.1, -0.9, -0.1), max_corner=(0.1, -0.8, 0.1),
                         material_id=STONE, spacing=0.02)
        sim.notify_scene_changed()
        _frame(sim)
        # Frame 1: the pinned data predates the scene, so nothing is latched.
        assert sim._last_max_disp_uint == Simulation._MAX_DISP_UNKNOWN
        assert sim._sort_skip_next is False
    finally:
        sim.close()


def test_fast_block_second_frame_is_full_sort(world):
    """A block spawned with a large velocity must not get a gather frame.

    K_Integrate records the largest per-substep displacement, so the block
    moves 2 cm per 1 ms substep (0.5h) against the 0.25h threshold.
    """
    sim = _make_sim(world)
    try:
        world.spawn_cube(min_corner=(-0.1, 0.0, -0.1), max_corner=(0.1, 0.1, 0.1),
                         material_id=WATER, spacing=0.02, velocity=(20.0, 0.0, 0.0))
        sim.notify_scene_changed()
        decisions = []
        for _ in range(3):
            _frame(sim)
            decisions.append(sim._sort_skip_next)
        # Decision after frame 1 sets up frame 2; after frame 2 sets up frame 3.
        assert decisions[0] is False
        assert decisions[1] is False
        assert decisions[2] is False
    finally:
        sim.close()


def test_decision_lag_is_one_frame(world):
    """The latched value at the end of frame k equals the actual at end of k-1."""
    sim = _make_sim(world)
    try:
        world.spawn_cube(min_corner=(-0.2, -0.2, -0.2), max_corner=(0.2, 0.2, 0.2),
                         material_id=WATER, spacing=0.02)
        sim.notify_scene_changed()
        actual = []
        latched = []
        for _ in range(12):
            _frame(sim)
            latched.append(sim._last_max_disp_uint)
            actual.append(_actual_disp_uint(world))
        # Frame 1 drops the pre-scene pinned data. Frame 1 also parks the DEAD
        # reserve, which calls notify_scene_changed() again, so frame 2 drops
        # the frame-1 copy too. From frame 3 on the lag is exactly one frame.
        assert latched[0] == Simulation._MAX_DISP_UNKNOWN
        assert latched[1] == Simulation._MAX_DISP_UNKNOWN
        for k in range(2, len(actual)):
            assert latched[k] == actual[k - 1], (k, latched, actual)
        # Every latch found the copy done.
        assert sim._readback_not_ready == 0
        assert sim._readback_latched == len(actual) - 2
    finally:
        sim.close()


def test_resting_scene_skips_and_push_is_seen(world):
    """At rest the sort is skipped; a pushed particle is never read as 0."""
    sim = _make_sim(world)
    try:
        world.spawn_cube(min_corner=(-0.2, -0.9, -0.2), max_corner=(0.2, -0.8, 0.2),
                         material_id=STONE, spacing=0.02)
        n = world.spawn_cube(min_corner=(0.0, 0.0, 0.0), max_corner=(0.03, 0.03, 0.03),
                             material_id=WATER, spacing=0.02)
        assert n >= 1
        water_idx = world._high_water - 1
        sim.notify_scene_changed()
        # No gravity, so the small water blob stays near rest.
        sim.set_gravity(0.0)
        skipped = False
        for _ in range(6):
            _frame(sim)
            skipped = skipped or sim._sort_skip_next
        assert skipped, "a resting scene should get grid-reuse frames"

        # Push one water particle hard: 20 m/s is 2 cm (0.5h) per 1 ms
        # substep, over the 0.25h per-substep threshold.
        world.velocity[water_idx, 0] = 20.0
        _frame(sim)
        pushed_actual = _actual_disp_uint(world)
        assert pushed_actual > 0
        _frame(sim)
        assert sim._last_max_disp_uint == pushed_actual
        assert sim._last_max_disp_uint != 0
        assert sim._sort_skip_next is False
    finally:
        sim.set_gravity(-4.0)
        sim.close()


def test_not_ready_copy_keeps_values_and_sorts(world):
    """A copy that has not landed is not latched and does not allow a skip."""
    sim = _make_sim(world)
    try:
        class _PendingEvent:
            done = False

            def record(self, stream=None):
                pass

        sim._readback_stale = False
        sim._last_max_disp_uint = 0  # would skip if trusted
        sim._last_foam_count = 7
        sim._readback_done = _PendingEvent()
        sim._update_sort_skip_decision()
        assert sim._sort_skip_next is False
        assert sim._last_max_disp_uint == 0
        assert sim._last_foam_count == 7
        assert sim._readback_not_ready == 1
    finally:
        sim._readback_done = cp.cuda.Event()
        sim.close()


def test_wait_event_issued_before_first_substep(world, monkeypatch):
    """step_frame makes the compute stream wait on _readback_done first."""
    sim = _make_sim(world)
    try:
        world.spawn_cube(min_corner=(-0.1, -0.9, -0.1), max_corner=(0.1, -0.8, 0.1),
                         material_id=STONE, spacing=0.02)
        sim.notify_scene_changed()
        calls = []
        real_stream = cp.cuda.get_current_stream

        class _Spy:
            def __init__(self, s):
                self._s = s

            def wait_event(self, ev):
                calls.append(("wait", ev))
                return self._s.wait_event(ev)

            def __getattr__(self, name):
                return getattr(self._s, name)

        monkeypatch.setattr(cp.cuda, "get_current_stream",
                            lambda *a, **k: _Spy(real_stream(*a, **k)))
        real_step = sim._sim_step

        def _step(n):
            calls.append(("step", None))
            return real_step(n)

        sim._sim_step = _step
        _frame(sim)
        monkeypatch.undo()
        first_step = calls.index(("step", None))
        waits = [i for i, c in enumerate(calls[:first_step])
                 if c[0] == "wait" and c[1] is sim._readback_done]
        assert waits, calls
    finally:
        sim.close()


def test_notify_scene_changed_drops_in_flight_copy(world):
    sim = _make_sim(world)
    try:
        sim._last_max_disp_uint = 0
        sim._readback_stale = False
        sim.notify_scene_changed()
        assert sim._last_max_disp_uint == Simulation._MAX_DISP_UNKNOWN
        assert sim._readback_stale is True
        sim._update_sort_skip_decision()
        assert sim._sort_skip_next is False
        assert sim._last_max_disp_uint == Simulation._MAX_DISP_UNKNOWN
    finally:
        sim.close()


def _make_packed(mat_id, behavior):
    return (mat_id & 0xFF) | ((behavior & 0x3) << 8)


def test_spawn_gas_forces_sort():
    """K_SpawnGas writes the +inf bits into max_displacement when it spawns."""
    params = build_sim_params(
        smoothing_length=0.04, particle_mass=0.008, particle_spacing=0.02,
        gravity=(0.0, -9.8, 0.0), dt=DT, restitution=0.3, wall_friction=0.5,
        world_min=(-1.0, -1.0, -1.0), world_max=(1.0, 1.0, 1.0),
    )
    spawn.upload_sim_params(params)
    spawn.upload_materials(build_material_array())
    n = 10
    f4 = lambda: cp.zeros((n, 4), dtype=cp.float32)  # noqa: E731
    packed = cp.zeros(n, dtype=cp.uint32)
    packed[0] = _make_packed(WATER, FLUID) | 0x800  # SPAWN flag
    for idx in (7, 8, 9):
        packed[idx] = _make_packed(DEAD, STATIC)
    mass = cp.zeros(n, dtype=cp.float32)
    mass[0] = 0.008
    dead_indices, dead_count = spawn.allocate_freelist(n)
    dead_indices[:3] = cp.asarray([7, 8, 9], dtype=cp.uint32)
    dead_count[0] = 3
    max_disp = cp.zeros(1, dtype=cp.uint32)
    max_disp[0] = np.float32(1e-6).view(np.uint32)

    def run(md):
        spawn.compute_spawn(
            packed, f4(), f4(), f4(), mass,
            cp.full(n, 400.0, dtype=cp.float32), cp.ones(n, dtype=cp.float32),
            cp.zeros(n, dtype=cp.float32), f4(), cp.zeros(n, dtype=cp.uint8),
            cp.full(n, 1000.0, dtype=cp.float32), cp.zeros(n, dtype=cp.float32),
            dead_indices, dead_count, md,
        )

    run(max_disp)
    assert int(max_disp.get()[0]) == 0x7F800000
    assert int(dead_count.get()[0]) == 0

    # No spawn (flag cleared, freelist empty): tracker untouched. NULL is safe.
    max_disp[0] = 5
    run(max_disp)
    run(None)
    assert int(max_disp.get()[0]) == 5


def test_foam_physics_grid_headroom(monkeypatch):
    """The foam physics grid covers 2x the deferred count plus the pad."""
    launched = {}

    class _K:
        def __call__(self, grid, block, args):
            launched["threads"] = grid[0] * block[0]

    class _M:
        def get_function(self, name):
            return _K()

    monkeypatch.setattr(foam, "_get_module", lambda: _M())
    foam.foam_physics(None, None, None, max_foam=100_000, last_foam_count=5000)
    assert launched["threads"] >= 2 * 5000 + foam._MIN_FOAM_GRID_SIZE
    foam.foam_physics(None, None, None, max_foam=3000, last_foam_count=5000)
    assert launched["threads"] < 3000 + foam._BLOCK
