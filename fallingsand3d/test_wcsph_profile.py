"""WCSPH profile upload and accuracy-to-dt binding (bd-r5-review-n28.25).

Checks that step2's c_granular carries the active WCSPH profile's
xsph_epsilon, force_scale and mu0 after init and after a profile switch, and
that the accuracy knob changes the adaptive dt.
"""

import dataclasses

import numpy as np
import pytest

try:
    import cupy.cuda.compiler as _compiler
    _compiler._use_ptx = True
except Exception:
    pass

import cupy as cp

import step1
import step2
from materials import WATER
from simulation import DT_MAX, Simulation
from solver_profiles import PROFILES
from world import World

MAX_PARTICLES = 20_000


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


@pytest.fixture
def sim(world):
    world.packed_info[:] = 0
    world._high_water = 0
    return Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                      fixed_dt=False, max_substeps=20)


def _read_const(module, name, dtype):
    """Test-only readback of a __constant__ struct (host sync is fine here)."""
    buf = np.zeros(1, dtype=dtype)
    ptr = module.get_global(name)
    cp.cuda.runtime.memcpy(buf.ctypes.data, int(ptr), buf.nbytes, 2)
    return buf[0]


def _read_granular():
    return _read_const(step2.get_module(), "c_granular",
                       step2.GRANULAR_PARAMS_DTYPE)


def _assert_granular_matches(profile):
    g = _read_granular()
    assert g["xsph_epsilon"] == pytest.approx(profile.xsph_epsilon)
    assert g["force_scale"] == pytest.approx(profile.force_scale)
    assert g["mu0"] == pytest.approx(profile.mu0)


def test_init_uploads_active_profile(sim):
    _assert_granular_matches(sim.solver_profile)


@pytest.mark.parametrize("name", ["WCSPH", "WCSPH (Fast)"])
def test_profile_switch_uploads_granular(sim, name):
    # Go through a non-WCSPH profile first so the switch back must re-upload.
    sim.set_solver_profile(PROFILES["DFSPH"])
    sim.set_solver_profile(PROFILES[name])
    _assert_granular_matches(PROFILES[name])


def test_custom_profile_values_reach_step2(sim):
    custom = dataclasses.replace(
        PROFILES["WCSPH"], name="custom",
        xsph_epsilon=0.3, force_scale=0.015, mu0=2.0,
    )
    sim.set_solver_profile(custom)
    _assert_granular_matches(custom)
    sim.set_solver_profile(PROFILES["WCSPH"])
    _assert_granular_matches(PROFILES["WCSPH"])


@pytest.mark.parametrize("accuracy,expected_dt", [
    (0.1, 0.25e-3), (0.2, 0.5e-3), (0.4, 1.0e-3), (1.0, DT_MAX),
])
def test_accuracy_sets_adaptive_dt(sim, accuracy, expected_dt):
    sim.accuracy = accuracy
    assert sim._compute_adaptive_dt(1) == pytest.approx(expected_dt)


def test_adjust_accuracy_changes_dt(sim):
    sim.accuracy = 0.4
    dt_hi = sim._compute_adaptive_dt(1)
    sim.adjust_accuracy(-0.3)
    dt_lo = sim._compute_adaptive_dt(1)
    assert dt_lo < dt_hi


def _run_profile(world, sim, name, sim_seconds=0.05):
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube(min_corner=(-0.1, -0.3, -0.1),
                         max_corner=(0.1, 0.0, 0.1),
                         material_id=WATER, spacing=0.02)
    sim.set_solver_profile(PROFILES[name])
    # Same dt path step_frame uses in adaptive mode.
    sim._upload_dt(sim._compute_adaptive_dt(n))
    c_sim = _read_const(step2.get_module(), "c_sim", step1.SIM_PARAMS_DTYPE)
    assert float(c_sim["dt"]) == pytest.approx(sim.dt)
    for _ in range(int(round(sim_seconds / sim.dt))):
        sim._sim_step(n)
    cp.cuda.Device().synchronize()
    return sim.dt, world.position[:n].get().copy()


def test_wcsph_and_fast_differ(world, sim):
    dt_a, pos_a = _run_profile(world, sim, "WCSPH")
    dt_b, pos_b = _run_profile(world, sim, "WCSPH (Fast)")
    assert dt_a == pytest.approx(0.5e-3)
    assert dt_b == pytest.approx(1.0e-3)
    assert np.isfinite(pos_a).all() and np.isfinite(pos_b).all()
    assert pos_a.shape == pos_b.shape
    assert np.abs(pos_a - pos_b).max() > 1e-6
