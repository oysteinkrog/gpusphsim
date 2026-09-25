"""Regression tests: the Timing panel path must run the same physics (bd-r5-review-n28.2).

_sim_step_timed used to be a hand copy of the WCSPH pipeline. It skipped the
STATIC particle boundary in integrate (no cell_start), blast waves, implicit
surface tension and kinematic SDF motion, so ticking the Timing checkbox
changed the scene.  These tests run the same scene through _sim_step and
_sim_step_timed and compare the results.
"""

import numpy as np


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

import simulation as simulation_mod  # noqa: E402
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from materials import WATER, STONE  # noqa: E402
from rigid_bodies import SDF_BOX  # noqa: E402

MAX_PARTICLES = 30_000

# Stage labels the timed path must report for a plain WCSPH scene.
WCSPH_STAGES = {"sort", "step1", "reactions", "spawn", "step2", "integrate", "wake", "foam"}


def _build_scene():
    """Water falling onto a STONE slab (STATIC particles) past a moving SDF box.

    Water sits one spacing above the slab, inside the 0.8 h STATIC guard, so
    the static_particle_boundary pass in integrate acts from the first substep.
    """
    world = World(max_particles=MAX_PARTICLES)
    sim = Simulation(world, dt=0.001, speed=1.0, fixed_dt=True, max_substeps=20)
    world.spawn_cube(
        min_corner=(-0.3, -0.12, -0.3), max_corner=(0.3, -0.06, 0.3),
        material_id=STONE, spacing=0.02,
    )
    world.spawn_cube(
        min_corner=(-0.2, -0.04, -0.2), max_corner=(0.2, 0.16, 0.2),
        material_id=WATER, spacing=0.02, velocity=(0.0, -1.0, 0.0),
    )
    obj = sim.sdf_manager.add_sdf_object(
        SDF_BOX, position=(0.0, 0.08, 0.0), size=(0.05, 0.05, 0.05),
    )
    sim.sdf_manager.add_kinematic_motion(
        obj, "oscillate_x", {"amplitude": 0.05, "frequency": 2.0},
    )
    sim.sdf_manager.upload_if_dirty()
    # Skip the spawn damping ramp so both runs see identical constants.
    sim._spawn_substep = sim._damping_duration
    return world, sim, obj


def _run(timed: bool, substeps: int):
    world, sim, obj = _build_scene()
    n = world._high_water
    step = sim._sim_step_timed if timed else sim._sim_step
    for _ in range(substeps):
        step(n)
    cp.cuda.Device().synchronize()
    pos = cp.asnumpy(world.position[:n, :3])
    sdf_pos = list(sim.sdf_manager.get_sdf_objects()[obj]["position"])
    timings = dict(sim._last_timings)
    del sim, world
    cp.get_default_memory_pool().free_all_blocks()
    return pos, sdf_pos, timings


def test_timed_path_single_substep_matches_untimed():
    """One substep from the same start state: positions agree to 1e-6."""
    pos_a, sdf_a, _ = _run(timed=False, substeps=1)
    pos_b, sdf_b, _ = _run(timed=True, substeps=1)
    assert np.allclose(sdf_a, sdf_b), f"SDF pose differs: {sdf_a} vs {sdf_b}"
    diff = float(np.max(np.abs(pos_a - pos_b)))
    assert diff <= 1e-6, f"max position difference after 1 substep: {diff:.3e}"


def test_timed_path_many_substeps_matches_untimed():
    """20 substeps: kinematic SDF keeps moving and positions agree to 1e-4.

    Not bitwise: counting-sort intra-cell order comes from atomics, so
    neighbour sums can be added in a different order.
    """
    pos_a, sdf_a, _ = _run(timed=False, substeps=20)
    pos_b, sdf_b, timings = _run(timed=True, substeps=20)
    assert abs(sdf_b[0]) > 1e-3, f"kinematic SDF did not move in timed path: {sdf_b}"
    assert np.allclose(sdf_a, sdf_b, atol=1e-7), f"SDF pose differs: {sdf_a} vs {sdf_b}"
    diff = float(np.max(np.abs(pos_a - pos_b)))
    assert diff <= 1e-4, f"max position difference after 20 substeps: {diff:.3e}"
    missing = WCSPH_STAGES - set(timings)
    assert not missing, f"timing panel stages missing: {sorted(missing)}"


def test_timed_path_integrate_gets_wake_table_and_no_cell_tables(monkeypatch):
    """The timed path's integrate call gets the wake table and frame counter.

    WCSPH passes no cell_start/cell_end on purpose, so the STATIC push-out
    in K_Integrate stays off (bd-r5-review-n28.166).
    """
    world, sim, _ = _build_scene()
    n = world._high_water
    seen = []
    real = simulation_mod.integrate.integrate

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(simulation_mod.integrate, "integrate", spy)
    sim._sim_step_timed(n)
    assert len(seen) == 1
    kw = seen[0]
    assert kw.get("cell_start") is None and kw.get("cell_end") is None
    assert kw.get("cell_wake_flags") is sim._cell_wake_flags
    assert kw.get("frame_counter") is sim._frame_counter_d
