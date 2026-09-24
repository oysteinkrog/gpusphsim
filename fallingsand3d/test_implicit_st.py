"""Implicit surface tension keeps its own scratch buffer (bd-r5-review-n28.56).

run_implicit_st used to ping-pong through sorted_veleval. Step2 writes
the XSPH advection velocity there and K_Integrate advects FLUID with it,
so turning IST on silently dropped XSPH. IST now uses
World.sorted_ist_scratch.

Checks:
- sorted_veleval is unchanged by run_implicit_st (odd and even counts).
- With IST on and sigma = 0, two colliding water blocks match IST off
  over 20 substeps, within the run-to-run noise of float atomics.
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
import implicit_st  # noqa: E402
from materials import WATER  # noqa: E402
from simulation import Simulation  # noqa: E402
from world import World  # noqa: E402

MAX_PARTICLES = 20_000
DT = 0.001
SUBSTEPS = 20


def _make_sim(ist_enabled, sigma):
    world = World(max_particles=MAX_PARTICLES)
    sim = Simulation(world, dt=DT, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=1)
    sim.fixed_dt_value = DT
    sim._upload_dt(DT)
    sim._spawn_substep = sim._damping_duration
    sim.ist_enabled = ist_enabled
    sim.ist_sigma = sigma
    sim.update_ist_params()
    # Two water blocks hitting each other at 2 m/s closing speed, so the
    # XSPH velocity differs from the particle velocity at the contact.
    world.spawn_cube(min_corner=(-0.21, -0.1, -0.1), max_corner=(-0.01, 0.1, 0.1),
                     material_id=WATER, spacing=0.02, velocity=(1.0, 0.0, 0.0))
    world.spawn_cube(min_corner=(0.01, -0.1, -0.1), max_corner=(0.21, 0.1, 0.1),
                     material_id=WATER, spacing=0.02, velocity=(-1.0, 0.0, 0.0))
    sim.notify_scene_changed()
    sim._invalidate_graphs()
    return sim, world


def _run(sim, world):
    n = world._high_water
    for _ in range(SUBSTEPS):
        sim._sim_step(n)
    cp.cuda.Device().synchronize()
    # Test-only readback.
    return cp.asnumpy(world.position[:n]), cp.asnumpy(world.velocity[:n])


def test_ist_sigma_zero_matches_ist_off():
    """IST with sigma = 0 must not change trajectories (XSPH kept)."""
    sim_off, w_off = _make_sim(False, 0.5)
    pos_off, vel_off = _run(sim_off, w_off)
    sim_on, w_on = _make_sim(True, 0.0)
    pos_on, vel_on = _run(sim_on, w_on)
    implicit_st.upload_ist_params()  # restore defaults for other tests
    assert pos_off.shape == pos_on.shape
    # Two IST-off runs already differ by about 1e-8 m and 1e-7 m/s (float
    # atomics in the pipeline), so bit-for-bit is not reachable. Borrowing
    # veleval as scratch gave 1.8e-3 m and 4e-2 m/s here.
    assert np.abs(pos_off[:, :3] - pos_on[:, :3]).max() < 1e-6
    assert np.abs(vel_off[:, :3] - vel_on[:, :3]).max() < 1e-5


@pytest.mark.parametrize("iterations", [4, 5])
def test_run_implicit_st_leaves_veleval_unchanged(iterations):
    """sorted_veleval holds the step2 XSPH output after the IST pass."""
    sim, world = _make_sim(True, 0.5)
    sim.ist_iterations = iterations
    sim.update_ist_params()
    n = world._high_water
    for _ in range(3):
        sim._sim_step(n)

    # Rerun the body up to step2 by hand: sort state and step1 outputs are
    # current after _sim_step, so running IST on the sorted buffers here
    # matches what the substep did.
    world.ensure_ist_scratch(sim._IST_MAX_PARTICLES)
    rng = np.random.default_rng(0)
    world.sorted_veleval[:n] = cp.asarray(
        rng.standard_normal((n, 4)).astype(np.float32))
    before = world.sorted_veleval[:n].copy()
    vel_before = world.sorted_velocity[:n].copy()
    result = implicit_st.run_implicit_st(
        world.sorted_velocity[:n],
        world.sorted_ist_scratch[:n],
        world.sorted_position[:n],
        world.sorted_density[:n],
        world.sorted_mass[:n],
        world.sorted_packed_info[:n],
        world.sorted_normal[:n],
        sim._cell_start,
        sim._cell_end,
        num_iterations=iterations,
    )
    cp.cuda.Device().synchronize()
    implicit_st.upload_ist_params()
    assert result.data.ptr != world.sorted_veleval[:n].data.ptr
    assert bool(cp.array_equal(world.sorted_veleval[:n], before))
    # The pass did something: some surface velocity changed.
    assert not bool(cp.array_equal(result, vel_before))


def test_substep_leaves_veleval_as_step2_output():
    """With IST on, integrate still sees XSPH veleval, not the IST result."""
    sim, world = _make_sim(True, 0.5)
    n = world._high_water
    for _ in range(3):
        sim._sim_step(n)
    cp.cuda.Device().synchronize()
    implicit_st.upload_ist_params()
    # Before the fix veleval equalled the IST velocity (odd count of 5).
    assert not bool(cp.array_equal(world.sorted_veleval[:n, :3],
                                   world.sorted_velocity[:n, :3]))
