"""Grid reuse (gather-only substeps) is exercised outside the app (bd-r5-review-n28.127).

Only step_frame sets _sort_skip_next, and every other test drives _sim_step
with a full sort. These tests cover the gather path directly:

1. K_GatherReorder writes sorted_X[i] == X[sort_perm[i]] for every field it
   gathers, and the FP16 mirrors round-trip against astype(float16). This
   runs on the direct path, the gather-only graph capture and its replay.
2. Density from a gather substep matches a full-sort substep per particle.
3. The CRIT-5 guard: a gather after the particle count changed runs a full
   sort instead.
4. The step_frame decision skips while drift is zero and forces a full sort
   when _sort_skip_consecutive reaches the cap. A fake clock drives frames.
5. A moving block with reuse on keeps its density and neighbour counts close
   to a run that sorts every substep.

All readbacks happen after the GPU work, in test code only.
"""

from __future__ import annotations

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

pytestmark = pytest.mark.gpu

MAX_PARTICLES = 40_000
DT = 0.001

# (unsorted, sorted) buffer names that K_GatherReorder copies for WCSPH.
# lambda_pbf is not in the list: it is carried only by the full sort.
GATHERED_FIELDS = [
    ("position", "sorted_position"),
    ("velocity", "sorted_velocity"),
    ("mass", "sorted_mass"),
    ("packed_info", "sorted_packed_info"),
    ("temperature", "sorted_temperature"),
    ("health", "sorted_health"),
    ("lifetime", "sorted_lifetime"),
    ("sleep_counter", "sorted_sleep_counter"),
    ("kappa", "sorted_kappa"),
    ("kappa_v", "sorted_kappa_v"),
]


class FakeClock:
    """Wall clock that advances by a fixed step on every read."""

    def __init__(self, step: float) -> None:
        self.t = 100.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


@pytest.fixture(scope="module")
def world():
    w = World(max_particles=MAX_PARTICLES)
    # No parked DEAD reserve: it changes n after the first frame.
    w.dead_reserve = 0
    return w


def _make_sim(world, max_substeps=1, clock=None, gravity=None):
    world.packed_info[:] = 0
    world._high_water = 0
    world.foam_count[:] = 0
    s = Simulation(world, dt=DT, speed=1.0, accuracy=0.4,
                   fixed_dt=True, max_substeps=max_substeps, clock=clock)
    s.fixed_dt_value = DT
    s._upload_dt(DT)
    s._spawn_substep = s._damping_duration
    if gravity is not None:
        s.set_gravity(gravity)
    return s


def _spawn_block(world, velocity=(0.0, 0.0, 0.0)):
    """About 30K WATER particles at 0.02 m spacing."""
    world.spawn_cube(min_corner=(-0.3, -0.3, -0.3), max_corner=(0.3, 0.3, 0.3),
                     material_id=WATER, spacing=0.02, velocity=velocity)
    return world._high_water


def _snapshot(world, n):
    """Device copies of the unsorted fields (no host sync)."""
    return {u: getattr(world, u)[:n].copy() for u, _ in GATHERED_FIELDS}


def _temperature_mirror(temp):
    """Expected raw uint16 bits of the temperature mirror.

    The mirror is either IEEE half, or fixed point T * TEMP_MIRROR_SCALE
    rounded to uint16 when common.cuh defines that scale.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).parent / "physics" / "kernels" / "common.cuh").read_text()
    m = re.search(r"#define\s+TEMP_MIRROR_SCALE\s+([0-9.]+)f?", src)
    if m is None:
        return temp.astype(np.float16).view(np.uint16)
    q = np.clip(temp * np.float32(float(m.group(1))), 0.0, 65535.0)
    return np.rint(q).astype(np.uint16)


def _check_gather(world, sim, n, snap, skip=()):
    perm = sim._sort_perm[:n].get().astype(np.int64)
    assert np.array_equal(np.sort(perm), np.arange(n)), "sort_perm is not a permutation"
    for u, s in GATHERED_FIELDS:
        if s in skip:
            continue
        expected = snap[u].get()[perm]
        got = getattr(world, s)[:n].get()
        assert np.array_equal(got, expected), f"{s} != {u}[sort_perm]"
    vel = snap["velocity"].get()[perm]
    vel_h = world.sorted_velocity_h[:n].get().view(np.float16).reshape(n, 4)
    assert np.array_equal(vel_h.view(np.uint16),
                          vel.astype(np.float16).view(np.uint16)), "sorted_velocity_h"
    temp = snap["temperature"].get()[perm]
    temp_h = world.sorted_temperature_h[:n].get().view(np.uint16)
    assert np.array_equal(temp_h, _temperature_mirror(temp)), "sorted_temperature_h"


def _gather_step(world, sim, n, run_substep):
    """Snapshot, run one substep, then check the gathered sorted buffers.

    run_substep=False calls _run_grid_setup alone. Later kernels in the
    substep may write sorted buffers in place, so the per-field check is
    made on the grid setup; the substep itself runs afterwards.
    """
    snap = _snapshot(world, n)
    sim._sort_skip_next = True
    if run_substep:
        sim._sim_step(n)
    else:
        sim._run_grid_setup(n)
    cp.cuda.Device().synchronize()
    return snap


def test_gather_matches_sort_perm_on_every_path(world):
    """Direct gather, gather-graph capture and replay all gather correctly."""
    sim = _make_sim(world)
    try:
        n = _spawn_block(world, velocity=(0.2, 0.0, 0.0))
        assert n > 25_000
        sim.notify_scene_changed()
        sim._sort_skip_next = False
        sim._sim_step(n)
        assert sim._last_full_sort_n == n

        # The gather-only grid setup, checked field by field.
        snap = _gather_step(world, sim, n, run_substep=False)
        _check_gather(world, sim, n, snap)

        # Substep 2: gather run directly (the gather graph is sized for n
        # only from here on). _sort_skipped_count counts non-replay gathers.
        before = sim._sort_skipped_count
        sim._sort_skip_next = True
        sim._sim_step(n)
        assert sim._sort_skipped_count == before + 1
        assert sim._graph_gather_only is None
        assert sim._graph_gather_only_n == n
        # Substep 3: gather graph captured and launched.
        sim._sim_step(n)
        assert sim._graph_gather_only is not None
        assert sim._sort_skipped_count == before + 2
        # Substep 4: gather graph replay. The gathered buffers are checked
        # with the grid setup that follows it, which reads what the replay's
        # integrate wrote.
        sim._sim_step(n)
        assert sim._substeps_since_sort == 4
        cp.cuda.Device().synchronize()
        snap = _gather_step(world, sim, n, run_substep=False)
        _check_gather(world, sim, n, snap)
        assert sim._last_full_sort_n == n
    finally:
        sim._sort_skip_next = False
        sim.close()


def test_gather_substeps_match_sort_perm(world):
    """Direct, captured and replayed gather substeps all gather correctly.

    Later kernels in the substep write sorted_position in place, so it is
    left out here; the grid-setup check above covers it.
    """
    sim = _make_sim(world)
    try:
        n = _spawn_block(world, velocity=(0.2, 0.0, 0.0))
        sim.notify_scene_changed()
        sim._sort_skip_next = False
        sim._sim_step(n)
        for path in ("direct", "capture", "replay"):
            snap = _gather_step(world, sim, n, run_substep=True)
            assert (sim._graph_gather_only is not None) == (path != "direct"), path
            _check_gather(world, sim, n, snap, skip=("sorted_position",))
        assert sim._substeps_since_sort == 4
    finally:
        sim._sort_skip_next = False
        sim.close()


def _density_by_particle(world, sim, n):
    perm = sim._sort_perm[:n]
    rho = cp.empty(n, dtype=cp.float32)
    rho[perm] = world.sorted_density[:n]
    return rho.get()


def _two_substeps(world, gather):
    sim = _make_sim(world)
    try:
        n = _spawn_block(world, velocity=(0.2, 0.0, 0.0))
        sim.notify_scene_changed()
        sim._sort_skip_next = False
        sim._sim_step(n)
        sim._sort_skip_next = gather
        before = sim._sort_skipped_count
        sim._sim_step(n)
        cp.cuda.Device().synchronize()
        assert sim._sort_skipped_count == before + (1 if gather else 0)
        return _density_by_particle(world, sim, n)
    finally:
        sim._sort_skip_next = False
        sim.close()


def test_gather_density_matches_full_sort_per_particle(world):
    """Density per original particle id agrees within 1e-4 relative."""
    rho_full = _two_substeps(world, gather=False)
    rho_gather = _two_substeps(world, gather=True)
    assert rho_full.shape == rho_gather.shape
    assert np.all(rho_full > 0)
    rel = np.abs(rho_gather - rho_full) / rho_full
    assert float(rel.max()) < 1e-4, float(rel.max())


def test_n_change_forces_full_sort(world):
    """Spawning after a gather step makes the next skip a full sort (CRIT-5)."""
    sim = _make_sim(world)
    try:
        n = _spawn_block(world)
        sim.notify_scene_changed()
        sim._sort_skip_next = False
        sim._sim_step(n)
        sim._sort_skip_next = True
        sim._sim_step(n)
        skipped = sim._sort_skipped_count
        # About 1000 more particles, clear of the block. No
        # notify_scene_changed: the n guard alone must catch this.
        world.spawn_cube(min_corner=(0.5, 0.5, 0.5), max_corner=(0.7, 0.7, 0.7),
                         material_id=WATER, spacing=0.02)
        n2 = world._high_water
        assert 700 <= n2 - n <= 1400, n2 - n
        vel_before = world.velocity[:n2].copy()
        assert sim._sort_skip_next
        sim._sim_step(n2)
        cp.cuda.Device().synchronize()
        assert sim._sort_skipped_count == skipped
        assert sim._last_full_sort_n == n2
        assert sim._last_setup_was_full_sort
        perm = sim._sort_perm[:n2].get().astype(np.int64)
        assert np.array_equal(np.sort(perm), np.arange(n2))
        # sorted_position is written in place later in the substep, so the
        # reorder is checked on velocity.
        assert np.array_equal(world.sorted_velocity[:n2].get(),
                              vel_before.get()[perm])
    finally:
        sim._sort_skip_next = False
        sim.close()


def test_step_frame_decision_and_consecutive_cap(world):
    """Zero drift skips, and the cap of 4 frames forces a full sort."""
    substeps = 4
    clock = FakeClock(step=(substeps + 1) * DT)  # max_substeps caps at 4
    sim = _make_sim(world, max_substeps=substeps, clock=clock, gravity=0.0)
    try:
        # 0.1 m spacing is 2.5h: no neighbours, no force, zero drift.
        world.spawn_cube(min_corner=(-0.3, -0.3, -0.3), max_corner=(0.3, 0.3, 0.3),
                         material_id=WATER, spacing=0.1)
        sim.notify_scene_changed()
        n = world._high_water
        assert sim.step_frame() == 0      # first call only sets the clock
        cap = sim._MAX_SORT_SKIP_FRAMES
        history = []
        for _ in range(2 * cap + 4):
            skip_in = sim._sort_skip_next
            assert sim.step_frame() == substeps
            assert world._high_water == n
            # Test-only sync, standing in for the app's render step, so the
            # readback copy of this frame lands before the next decision.
            cp.cuda.Device().synchronize()
            history.append((skip_in, sim._substeps_since_sort,
                            sim._sort_skip_consecutive, sim._sort_skip_next))
        # A frame that started with the skip flag clear sorted on every
        # substep; one that started with it set ran no full sort.
        prev_since = 0
        for skip_in, since, _, _ in history:
            if skip_in:
                assert since == prev_since + substeps, history
            else:
                assert since == 1, history
            prev_since = since
        # Frame 1 decides on the unknown readback, so it never skips.
        assert history[0][3] is False, history
        # From the second decision on the one-frame-old readback shows zero
        # drift: skip for `cap` frames, then one forced full sort, repeat.
        decisions = [h[3] for h in history[1:]]
        expected = ([True] * cap + [False]) * 3
        assert decisions == expected[:len(decisions)], history
        consec = [h[2] for h in history[1:cap + 2]]
        assert consec == list(range(1, cap + 1)) + [0], history
    finally:
        sim._sort_skip_next = False
        sim.close()


def _run_moving_block(world, reuse, frames=5, substeps=17):
    """1 m/s block, gravity 0; per-substep density and neighbour count."""
    clock = FakeClock(step=(substeps + 1) * DT)
    sim = _make_sim(world, max_substeps=substeps, clock=clock, gravity=0.0)
    rho_all, nb_all = [], []
    try:
        world.spawn_cube(min_corner=(-0.6, -0.1, -0.1), max_corner=(-0.3, 0.1, 0.1),
                         material_id=WATER, spacing=0.02, velocity=(1.0, 0.0, 0.0))
        sim.notify_scene_changed()
        if not reuse:
            sim._MAX_SORT_SKIP_FRAMES = 0   # force_sort on every substep
        real_step = sim._sim_step

        def _step(n_, **kw):
            real_step(n_, **kw)
            # Debug readback, test only: map back to original ids.
            perm = sim._sort_perm[:n_]
            rho = cp.empty(n_, dtype=cp.float32)
            rho[perm] = world.sorted_density[:n_]
            nb = cp.empty(n_, dtype=cp.float32)
            nb[perm] = world.sorted_normal[:n_, 3]
            rho_all.append(rho.get())
            nb_all.append(nb.get())

        sim._sim_step = _step
        sim.step_frame()
        for _ in range(frames):
            assert sim.step_frame() == substeps
            cp.cuda.Device().synchronize()
        return np.array(rho_all), np.array(nb_all), sim._sort_skipped_count
    finally:
        sim.close()


def test_moving_block_reuse_matches_full_sort(world):
    """5 frames x 17 substeps at 1 m/s: density within 5%, neighbours within 10%."""
    rho_r, nb_r, skipped_r = _run_moving_block(world, reuse=True)
    rho_f, nb_f, skipped_f = _run_moving_block(world, reuse=False)
    assert skipped_f == 0
    assert rho_r.shape == rho_f.shape == (5 * 17, rho_r.shape[1])
    for k in range(rho_f.shape[0]):
        for name, r, f, tol in (("density", rho_r[k], rho_f[k], 0.05),
                                ("neighbours", nb_r[k], nb_f[k], 0.10)):
            med_r, med_f = float(np.median(r)), float(np.median(f))
            p5_r, p5_f = float(np.percentile(r, 5)), float(np.percentile(f, 5))
            assert abs(med_r - med_f) / med_f < tol, (name, k, med_r, med_f)
            assert abs(p5_r - p5_f) / p5_f < tol, (name, k, p5_r, p5_f)
