"""Foam module tests: c_sim stays in sync with the rest of the simulation."""

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
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402

MAX_PARTICLES = 20_000
DEFAULT_GRAVITY = -4.0


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


@pytest.fixture
def sim(world):
    s = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                   fixed_dt=False, max_substeps=20)
    yield s
    s.set_gravity(DEFAULT_GRAVITY)


def _foam_c_sim(sim) -> np.ndarray:
    """Read the foam module's c_sim constant back to the host."""
    dtype = sim._sim_params.dtype
    d_ptr = foam._get_module().get_global("c_sim")
    out = np.zeros(1, dtype=dtype)
    cp.cuda.runtime.memcpy(out.ctypes.data, int(d_ptr), out.nbytes, 2)
    return out


def test_set_gravity_reaches_foam_module(sim):
    """set_gravity must update the foam module's c_sim.gravity."""
    sim.set_gravity(-20.0)
    c_sim = _foam_c_sim(sim)
    assert c_sim[0]["gravity"][1] == np.float32(-20.0)


def _seed_spray(world, sim, k: int) -> float:
    """Put k resting SPRAY particles in the middle of the world."""
    wmin = np.array(sim._sim_params[0]["world_min"], dtype=np.float32)
    wmax = np.array(sim._sim_params[0]["world_max"], dtype=np.float32)
    mid = 0.5 * (wmin + wmax)
    pos = np.zeros((k, 4), dtype=np.float32)
    pos[:, :3] = mid
    pos[:, 3] = 0.0  # SPRAY
    vel = np.zeros((k, 4), dtype=np.float32)
    vel[:, 3] = 100.0  # lifetime, long enough for the test
    world.foam_position[:k] = cp.asarray(pos)
    world.foam_velocity[:k] = cp.asarray(vel)
    world.foam_count[:] = k
    return float(mid[1])


@pytest.mark.parametrize("gy,expect", [(0.0, "still"), (9.81, "rises")])
def test_spray_follows_new_gravity(world, sim, gy, expect):
    """After set_gravity, spray velocity follows the new gravity, not the old -4."""
    k = 256
    y0 = _seed_spray(world, sim, k)
    sim.set_gravity(gy)
    for _ in range(50):
        foam.foam_physics(world.foam_position, world.foam_velocity,
                          world.foam_count, world._max_foam, last_foam_count=k)
    vy = cp.asnumpy(world.foam_velocity[:k, 1])
    py = cp.asnumpy(world.foam_position[:k, 1])
    world.foam_count[:] = 0
    if expect == "still":
        # Drag only: no -g*dt trend (old gravity would give about -0.2 m/s).
        assert np.abs(vy).max() < 1e-6, f"spray vy {vy.min():.4f} under zero gravity"
        assert np.abs(py - y0).max() < 1e-6
    else:
        assert vy.min() > 0.1, f"spray vy {vy.min():.4f}, want upward"
        assert py.min() > y0
