"""Grid-reuse drift is measured from the last full sort (bd-r5-review-n28.54).

K_ScatterReorder writes sorted_anchor on each full sort. The finalize
kernels compare the new position against it, so max_displacement is the
drift since the last full sort, not the largest single-substep move. The
host scales that drift to the end of the next frame before it allows a
gather-only frame.
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
H = 0.04


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


def _make_sim(world, substeps):
    world.packed_info[:] = 0
    world._high_water = 0
    world.foam_count[:] = 0
    s = Simulation(world, dt=DT, speed=1.0, accuracy=0.4,
                   fixed_dt=True, max_substeps=substeps)
    s.fixed_dt_value = DT
    s._upload_dt(DT)
    # No spawn damping ramp: it would bleed off the test velocities.
    s._spawn_substep = s._damping_duration
    s.set_gravity(0.0)
    return s


def _disp_sq(world):
    # Test-only readback (host sync is fine here).
    bits = int(world.max_displacement.get()[0])
    return struct.unpack('f', struct.pack('I', bits))[0]


def _frame(sim, substeps):
    if sim._last_frame_time is None:
        sim.step_frame()  # first call only sets the clock
    sim._last_frame_time = time.perf_counter() - 1.0
    steps = sim.step_frame()
    assert steps == substeps
    # Test-only sync, standing in for the app's render step.
    cp.cuda.Device().synchronize()


def test_max_displacement_accumulates_over_gather_substeps(world):
    """20 substeps at 1 m/s after one full sort read about (20 mm)^2."""
    sim = _make_sim(world, 1)
    try:
        world.spawn_cube(min_corner=(-0.5, -0.1, -0.1), max_corner=(-0.3, 0.1, 0.1),
                         material_id=WATER, spacing=0.02, velocity=(1.0, 0.0, 0.0))
        sim.notify_scene_changed()
        n = world._high_water
        sim._sort_skip_next = False
        sim._sim_step(n)          # full sort, writes the anchor
        cp.cuda.Device().synchronize()
        one_step = _disp_sq(world)
        assert one_step == pytest.approx((DT * 1.0) ** 2, rel=0.2)
        sim._sort_skip_next = True
        for _ in range(19):
            sim._sim_step(n)      # gather only, anchor untouched
        cp.cuda.Device().synchronize()
        assert sim._substeps_since_sort == 20
        drift = _disp_sq(world)
        # Before the fix this stayed at about (1 mm)^2.
        assert drift >= 0.9 * (20 * DT * 1.0) ** 2, drift
    finally:
        sim._sort_skip_next = False
        sim.close()


def test_anchor_matches_sorted_position_after_full_sort(world):
    """The anchor is the sort-time position and gather leaves it alone."""
    sim = _make_sim(world, 1)
    try:
        world.spawn_cube(min_corner=(-0.2, -0.2, -0.2), max_corner=(0.2, 0.0, 0.2),
                         material_id=WATER, spacing=0.02, velocity=(0.5, 0.0, 0.0))
        sim.notify_scene_changed()
        n = world._high_water
        sim._sort_skip_next = False
        sim._run_grid_setup(n)
        cp.cuda.Device().synchronize()
        anchor = world.sorted_anchor[:n, :3].get()
        assert np.array_equal(anchor, world.sorted_position[:n, :3].get())
        world.position[:n, 0] += 0.01
        sim._sort_skip_next = True
        sim._run_grid_setup(n)
        cp.cuda.Device().synchronize()
        assert np.array_equal(world.sorted_anchor[:n, :3].get(), anchor)
        assert not np.array_equal(world.sorted_position[:n, :3].get(), anchor)
    finally:
        sim._sort_skip_next = False
        sim.close()


def test_resting_particles_read_zero_after_full_sort(world):
    """Isolated particles at rest have exactly zero drift, sorted or not."""
    sim = _make_sim(world, 1)
    try:
        # 0.1 m spacing is 2.5h: no neighbours, so no force and no motion.
        world.spawn_cube(min_corner=(-0.3, -0.3, -0.3), max_corner=(0.3, 0.3, 0.3),
                         material_id=WATER, spacing=0.1)
        sim.notify_scene_changed()
        n = world._high_water
        sim._sort_skip_next = False
        sim._sim_step(n)
        cp.cuda.Device().synchronize()
        assert int(world.max_displacement.get()[0]) == 0
        sim._sort_skip_next = True
        for _ in range(10):
            sim._sim_step(n)
        cp.cuda.Device().synchronize()
        assert int(world.max_displacement.get()[0]) == 0
    finally:
        sim._sort_skip_next = False
        sim.close()


def _run_translating_block(world, substeps, frames, reuse):
    """Run a 2 m/s WCSPH block; return per-substep densities and skip count."""
    sim = _make_sim(world, substeps)
    densities = []
    try:
        world.spawn_cube(min_corner=(-0.6, -0.1, -0.1), max_corner=(-0.3, 0.1, 0.1),
                         material_id=WATER, spacing=0.02, velocity=(2.0, 0.0, 0.0))
        sim.notify_scene_changed()
        n = world._high_water
        if not reuse:
            sim._MAX_SORT_SKIP_FRAMES = 0
        real_step = sim._sim_step

        def _step(n_):
            real_step(n_)
            # Debug readback: density per original particle index.
            perm = sim._sort_perm[:n_]
            rho = cp.empty(n_, dtype=cp.float32)
            rho[perm] = world.sorted_density[:n_]
            # The DEAD reserve parked after frame 1 raises n; keep the water
            # block, which holds the first n original slots.
            densities.append(rho[:n].get())

        sim._sim_step = _step
        for _ in range(frames):
            _frame(sim, substeps)
        return np.array(densities), sim._sort_skipped_count, n
    finally:
        sim.close()


def test_fast_block_density_matches_full_sort(world):
    """A 2 m/s block keeps its density with reuse on (5 frames x 17 substeps).

    Before the fix the reuse run skipped whole frames, drifted over 3 cells
    on a stale grid and lost its neighbours (density fell to the floor).
    """
    rho_reuse, skipped, n = _run_translating_block(world, 17, 5, reuse=True)
    rho_full, skipped_full, _ = _run_translating_block(world, 17, 5, reuse=False)
    assert skipped_full == 0
    assert rho_reuse.shape == rho_full.shape
    # Per substep, the bulk density with reuse matches the full-sort run.
    for k in range(rho_full.shape[0]):
        med_r = float(np.median(rho_reuse[k]))
        med_f = float(np.median(rho_full[k]))
        p5_r = float(np.percentile(rho_reuse[k], 5))
        p5_f = float(np.percentile(rho_full[k], 5))
        assert abs(med_r - med_f) / med_f < 0.05, (k, med_r, med_f)
        assert abs(p5_r - p5_f) / p5_f < 0.05, (k, p5_r, p5_f)
    # At 2 mm per substep the predicted drift is over 0.25h, so no substep
    # may reuse the grid.
    assert skipped == 0, skipped
