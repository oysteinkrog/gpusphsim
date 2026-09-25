"""Per-substep constant uploads do not block the host (bd-r5-review-n28.168).

Kinematic SDF motion rewrites the SDF table every substep, and the spawn
damping ramp rewrites c_sim.velocity_damping for 30 substeps after every
scene change. Both used blocking pageable cudaMemcpy, which drains the GPU
queue before each copy, and the ramp also invalidated the CUDA graphs every
substep. They now go through rigid_bodies.PinnedUploadRing (cudaMemcpyAsync
from pinned memory on the current stream) and the graphs stay valid.

Scene: a 0.3 x 0.2 x 0.3 m water block (spacing 0.02) next to a kinematic
cylinder spinning about y at 2 rad/s, the Water Mill case.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

import rigid_bodies  # noqa: E402
import simulation as simulation_mod  # noqa: E402
from materials import WATER  # noqa: E402

SPIN = 2.0  # rad/s


def _mill(solver: str):
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(20_000)
    world.spawn_cube((-0.35, -0.5, -0.15), (-0.05, -0.3, 0.15), WATER, 0.02)
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=0.5)
    sim.set_solver_profile(PROFILES[solver])
    sdf = sim.sdf_manager
    obj = sdf.add_sdf_object(
        rigid_bodies.SDF_CYLINDER, position=(0.05, -0.4, 0.0),
        size=(0.1, 0.1, 0.1), restitution=0.3, friction=0.5)
    sdf.add_kinematic_motion(obj, "rotate_y", {"speed": SPIN})
    sdf.force_upload()
    sim.notify_scene_changed()
    sim.reset_spawn_damping()
    cp.cuda.Device().synchronize()
    return world, sim


def _cleanup_sdf():
    rigid_bodies.upload_sdf_objects(None, 0)
    cp.cuda.Device().synchronize()


def _read_sdf_table(module, count):
    d_count = module.get_global("c_num_sdf_objects")
    d_objs = module.get_global("c_sdf_objects")
    n = np.zeros(1, dtype=np.int32)
    cp.cuda.runtime.memcpy(n.ctypes.data, int(d_count), 4, 2)
    objs = np.zeros(count, dtype=rigid_bodies.SDF_OBJECT_DTYPE)
    cp.cuda.runtime.memcpy(objs.ctypes.data, int(d_objs), objs.nbytes, 2)
    return int(n[0]), objs


def test_no_blocking_memcpy_in_sim_step(monkeypatch):
    world, sim = _mill("WCSPH")
    try:
        # Warm up so one-time module compiles and symbol lookups are done.
        for _ in range(3):
            sim._sim_step(world._high_water)
        cp.cuda.Device().synchronize()

        calls = []
        real = cp.cuda.runtime.memcpy

        def counting(*a, **k):
            calls.append(a[3] if len(a) > 3 else None)
            return real(*a, **k)

        monkeypatch.setattr(cp.cuda.runtime, "memcpy", counting)
        # 20 ramp substeps (damping changes every substep) then 20 steady.
        for _ in range(40):
            sim._sim_step(world._high_water)
        cp.cuda.Device().synchronize()
        h2d = [c for c in calls if c == 1]
        assert h2d == [], f"{len(h2d)} blocking H2D cudaMemcpy in 40 substeps"
    finally:
        _cleanup_sdf()


def test_sdf_table_matches_manager_after_substeps():
    import integrate

    world, sim = _mill("WCSPH")
    try:
        for _ in range(10):
            sim._sim_step(world._high_water)
        cp.cuda.Device().synchronize()
        expected = sim.sdf_manager._build_array()[:1]
        count, got = _read_sdf_table(integrate.get_module(), 1)
        assert count == 1
        for field in rigid_bodies.SDF_OBJECT_DTYPE.names:
            np.testing.assert_array_equal(got[field], expected[field], field)
        # The pose is for the last substep's sim_time (one-substep lag).
        t_last = sim.sim_time - sim.dt
        half = 0.5 * SPIN * t_last
        np.testing.assert_allclose(
            got["quat"][0], [0.0, np.sin(half), 0.0, np.cos(half)],
            rtol=0, atol=1e-6)
    finally:
        _cleanup_sdf()


def test_ramp_keeps_graphs_and_damping_applies(monkeypatch):
    world, sim = _mill("PBF")
    try:
        captures = []
        real_capture = sim._capture_graph

        def counting_capture(n, gather_only):
            captures.append(gather_only)
            return real_capture(n, gather_only)

        monkeypatch.setattr(sim, "_capture_graph", counting_capture)
        invalidations = []
        real_inv = sim._invalidate_graphs

        def counting_inv():
            invalidations.append(sim._spawn_substep)
            real_inv()

        monkeypatch.setattr(sim, "_invalidate_graphs", counting_inv)

        damping_seen = []
        for _ in range(40):
            sim._sim_step(world._high_water)
            damping_seen.append(float(sim._sim_params[0]["velocity_damping"]))
        cp.cuda.Device().synchronize()

        # The ramp ran: damping falls from 0.8 to 0 over 30 substeps.
        assert damping_seen[0] == pytest.approx(0.8)
        assert damping_seen[29] > 0.0 and damping_seen[30] == 0.0
        # No invalidation during the ramp, and each graph path captured at
        # most once in 40 substeps.
        assert invalidations == []
        assert captures.count(False) <= 1 and captures.count(True) <= 1
        assert sim._graph_full_sort is not None or sim._graph_gather_only is not None
    finally:
        _cleanup_sdf()


def _run_positions(monkeypatch, old_path: bool):
    """40 substeps on PBF (ramp + motion). old_path restores sync copies and
    per-ramp-substep graph invalidation."""
    with monkeypatch.context() as m:
        if old_path:
            def sync_upload(dst_ptr, arr):
                a = np.ascontiguousarray(arr)
                cp.cuda.runtime.memcpy(int(dst_ptr), a.ctypes.data, a.nbytes, 1)

            m.setattr(rigid_bodies, "pinned_upload", sync_upload)
            m.setattr(simulation_mod, "pinned_upload", sync_upload)
        world, sim = _mill("PBF")
        try:
            if old_path:
                real_ramp = sim._apply_damping_ramp

                def ramp_and_invalidate():
                    active = sim._spawn_substep < sim._damping_duration
                    real_ramp()
                    if active:
                        sim._invalidate_graphs()

                m.setattr(sim, "_apply_damping_ramp", ramp_and_invalidate)
            for _ in range(40):
                sim._sim_step(world._high_water)
            cp.cuda.Device().synchronize()
            n = world._high_water
            return world.position[:n, :3].get().copy()
        finally:
            _cleanup_sdf()


def test_positions_match_old_upload_path(monkeypatch):
    new = _run_positions(monkeypatch, old_path=False)
    old = _run_positions(monkeypatch, old_path=True)
    assert new.shape == old.shape
    err = np.abs(new - old).max()
    assert err < 1e-4, f"max position difference {err:.3e} m"
