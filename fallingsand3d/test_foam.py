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


# ---------------------------------------------------------------------------
# Emission cadence (bd-r5-review-n28.122)
# ---------------------------------------------------------------------------

WATER_INFO = 5  # material WATER, behavior FLUID (bits 9:8 = 0)


def test_generate_rate_scales_with_dt_emit(world):
    """10K identical surface particles with phi = 4: expected k_rate*(phi-thr)*dt each."""
    k_rate, threshold, dt_emit = 5.0, 2.0, 0.01
    n = 10_000
    params = foam.build_foam_params(max_foam=world._max_foam, k_rate=k_rate,
                                    threshold=threshold)
    foam.upload_foam_params(params)
    # phi = 0.5*s + 0.2*0.5*s^2 = 4 (no wave crest: zero normal).
    s = (-5.0 + np.sqrt(25.0 + 160.0)) / 2.0
    phi = 0.5 * s + 0.1 * s * s
    assert abs(phi - 4.0) < 1e-6
    pos = cp.zeros((n, 4), dtype=cp.float32)
    vel = cp.zeros((n, 4), dtype=cp.float32)
    vel[:, 0] = s
    nrm = cp.zeros((n, 4), dtype=cp.float32)
    nrm[:, 3] = 10.0  # neighbor count: surface particle
    info = cp.full(n, WATER_INFO, dtype=cp.uint32)
    expect = n * k_rate * (phi - threshold) * dt_emit  # 1000
    counts = []
    for seed in (7, 1234, 99999):
        world.foam_count[:] = 0
        foam.foam_generate(pos, vel, nrm, info, world.foam_position,
                           world.foam_velocity, world.foam_count, n, seed, dt_emit)
        counts.append(int(world.foam_count.get()[0]))
    world.foam_count[:] = 0
    for c in counts:
        assert abs(c - expect) < 0.15 * expect, f"emitted {counts}, want about {expect:.0f}"
    # Doubling dt_emit doubles the expected count.
    foam.foam_generate(pos, vel, nrm, info, world.foam_position,
                       world.foam_velocity, world.foam_count, n, 7, 2 * dt_emit)
    c2 = int(world.foam_count.get()[0])
    world.foam_count[:] = 0
    assert abs(c2 - 2 * expect) < 0.15 * 2 * expect, f"emitted {c2} at 2*dt_emit"


def test_step_frame_generates_once_per_frame_and_compacts_every_8(world, monkeypatch):
    """16 step_frame calls: 16 generate launches and exactly 2 compactions."""
    import time
    world.foam_count[:] = 0
    world.spawn_cube(min_corner=(-0.2, -0.2, -0.2), max_corner=(0.2, 0.2, 0.2),
                     material_id=5, spacing=0.02, velocity=(0.0, -3.0, 0.0))
    world.foam_enabled = True
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=20)
    calls = {"gen": [], "compact": 0}
    real_gen, real_compact = foam.foam_generate, foam.foam_compact

    def _gen(*a, **k):
        calls["gen"].append(a[-1])  # dt_emit
        return real_gen(*a, **k)

    def _compact(*a, **k):
        calls["compact"] += 1
        return real_compact(*a, **k)

    monkeypatch.setattr(foam, "foam_generate", _gen)
    monkeypatch.setattr(foam, "foam_compact", _compact)
    try:
        sim.step_frame()  # first call only sets the clock
        steps = []
        for _ in range(16):
            sim._last_frame_time = time.perf_counter() - 1.0 / 60.0
            steps.append(sim.step_frame())
        cp.cuda.Device().synchronize()
        assert all(s > 1 for s in steps), steps
        assert len(calls["gen"]) == 16, calls["gen"]
        for s_, dt_emit in zip(steps, calls["gen"]):
            assert abs(dt_emit - s_ * sim.dt) < 1e-9
        assert calls["compact"] == 2
    finally:
        monkeypatch.undo()
        world.foam_enabled = False
        world.foam_count[:] = 0
        sim.close()
