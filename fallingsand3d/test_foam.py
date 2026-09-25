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
                          world.foam_count, world._max_foam)
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


# ---------------------------------------------------------------------------
# Dead foam is marked in position.w and culled (bd-r5-review-n28.146)
# ---------------------------------------------------------------------------

def _seed_foam(world, pos: np.ndarray, vel: np.ndarray) -> None:
    k = pos.shape[0]
    world.foam_position[:k] = cp.asarray(pos)
    world.foam_velocity[:k] = cp.asarray(vel)
    world.foam_count[:] = k


def _mid(sim) -> np.ndarray:
    wmin = np.array(sim._sim_params[0]["world_min"], dtype=np.float32)
    wmax = np.array(sim._sim_params[0]["world_max"], dtype=np.float32)
    return 0.5 * (wmin + wmax)


def test_expired_foam_marked_dead_in_position(world, sim):
    """A particle whose lifetime runs out gets position.w = -1 on that substep."""
    pos = np.zeros((2, 4), dtype=np.float32)
    pos[:, :3] = _mid(sim)
    pos[:, 3] = [1.0, 0.0]  # FOAM, SPRAY
    vel = np.zeros((2, 4), dtype=np.float32)
    vel[:, 3] = [0.5 * sim.dt, 100.0]  # first dies this substep
    _seed_foam(world, pos, vel)
    foam.foam_physics(world.foam_position, world.foam_velocity,
                      world.foam_count, world._max_foam)
    p = cp.asnumpy(world.foam_position[:2])
    v = cp.asnumpy(world.foam_velocity[:2])
    world.foam_count[:] = 0
    assert p[0, 3] < 0 and v[0, 3] < 0
    assert p[1, 3] == 0.0 and v[1, 3] > 0


def test_floor_kill_marks_dead_in_position(world, sim):
    """A particle pushed below the floor gets position.w = -1 on that substep."""
    wmin = np.array(sim._sim_params[0]["world_min"], dtype=np.float32)
    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0, :3] = _mid(sim)
    pos[0, 1] = wmin[1] - 0.01
    pos[0, 3] = 0.0  # SPRAY
    vel = np.zeros((1, 4), dtype=np.float32)
    vel[0, 1] = -1.0
    vel[0, 3] = 100.0
    _seed_foam(world, pos, vel)
    foam.foam_physics(world.foam_position, world.foam_velocity,
                      world.foam_count, world._max_foam)
    p = cp.asnumpy(world.foam_position[0])
    v = cp.asnumpy(world.foam_velocity[0])
    world.foam_count[:] = 0
    assert p[3] < 0 and v[3] < 0


def test_compaction_clears_stale_tail(world, sim):
    """After compaction the active buffer has w < 0 from the alive count up."""
    k = 20
    pos = np.zeros((k, 4), dtype=np.float32)
    pos[:, :3] = _mid(sim)
    pos[:, 3] = 1.0
    vel = np.zeros((k, 4), dtype=np.float32)
    vel[:, 3] = np.where(np.arange(k) % 2 == 0, 100.0, -1.0)  # 10 alive
    _seed_foam(world, pos, vel)
    # Stale positions from an earlier cycle in the output buffer.
    world.foam_position_b[:] = cp.float32(2.0)
    sim._frame_counter = 8
    enabled = world.foam_enabled
    world.foam_enabled = True
    try:
        sim._maybe_compact_foam()
    finally:
        world.foam_enabled = enabled
    count = int(world.foam_count[0])
    p = cp.asnumpy(world.foam_position[:])
    world.foam_count[:] = 0
    assert count == 10
    assert np.all(p[:10, 3] == 1.0)
    assert np.all(p[10:, 3] < 0), "stale tail still drawable after compaction"


# ---------------------------------------------------------------------------
# Generate load order, pool-full skip, compaction (bd-r5-review-n28.179)
# ---------------------------------------------------------------------------

def test_compaction_keeps_exactly_the_alive_set(world):
    """50K alive plus 50K dead in: exactly the 50K alive (pos, vel) rows out."""
    k = 100_000
    assert world._max_foam >= k
    rng = np.random.default_rng(179)
    pos = rng.standard_normal((k, 4)).astype(np.float32)
    pos[:, 3] = rng.integers(0, 3, k).astype(np.float32)
    vel = rng.standard_normal((k, 4)).astype(np.float32)
    alive = rng.permutation(k) < k // 2
    vel[:, 3] = np.where(alive, rng.uniform(0.1, 2.0, k), -1.0).astype(np.float32)
    _seed_foam(world, pos, vel)
    world.foam_alive_count[:] = 12345  # the wrapper must reset it
    foam.foam_compact(world.foam_position, world.foam_velocity,
                      world.foam_position_b, world.foam_velocity_b,
                      world.foam_alive_count, world.foam_count, world._max_foam)
    n_out = int(world.foam_alive_count.get()[0])
    out = np.concatenate([cp.asnumpy(world.foam_position_b[:n_out]),
                          cp.asnumpy(world.foam_velocity_b[:n_out])], axis=1)
    world.foam_count[:] = 0
    assert n_out == k // 2
    want = np.concatenate([pos[alive], vel[alive]], axis=1)
    order = lambda a: a[np.lexsort(a.T[::-1])]  # noqa: E731
    np.testing.assert_array_equal(order(out), order(want))


def _surface_inputs(n: int, s: float, nbr: float):
    pos = cp.zeros((n, 4), dtype=cp.float32)
    pos[:, 0] = cp.arange(n, dtype=cp.float32) * 1e-3
    vel = cp.zeros((n, 4), dtype=cp.float32)
    vel[:, 0] = s
    nrm = cp.zeros((n, 4), dtype=cp.float32)
    nrm[:, 3] = nbr
    info = cp.full(n, WATER_INFO, dtype=cp.uint32)
    return pos, vel, nrm, info


def test_generate_half_velocity_matches_float(world):
    """The half4 velocity path emits the same particles as the float4 path."""
    n = 10_000
    foam.upload_foam_params(foam.build_foam_params(max_foam=world._max_foam))
    s = 3.0  # exactly representable in half
    pos, vel, nrm, info = _surface_inputs(n, s, 10.0)
    vel_h = vel.astype(cp.float16).view(cp.uint32).reshape(n, 2)
    runs = []
    for vh in (None, vel_h):
        world.foam_count[:] = 0
        foam.foam_generate(pos, vel, nrm, info, world.foam_position,
                           world.foam_velocity, world.foam_count, n, 42, 0.01,
                           sorted_velocity_h=vh)
        c = int(world.foam_count.get()[0])
        rows = np.concatenate([cp.asnumpy(world.foam_position[:c]),
                               cp.asnumpy(world.foam_velocity[:c])], axis=1)
        runs.append(rows[np.lexsort(rows.T[::-1])])
    world.foam_count[:] = 0
    assert runs[0].shape[0] > 0
    np.testing.assert_array_equal(runs[0], runs[1])


def test_generate_skips_interior_and_respects_full_pool(world):
    """Interior particles (neighbor count > 25) never emit; a full pool stays full."""
    n = 10_000
    foam.upload_foam_params(foam.build_foam_params(max_foam=world._max_foam))
    pos, vel, nrm, info = _surface_inputs(n, 3.0, 30.0)
    world.foam_count[:] = 0
    foam.foam_generate(pos, vel, nrm, info, world.foam_position,
                       world.foam_velocity, world.foam_count, n, 42, 1.0)
    assert int(world.foam_count.get()[0]) == 0
    nrm[:, 3] = 10.0
    world.foam_count[:] = world._max_foam
    foam.foam_generate(pos, vel, nrm, info, world.foam_position,
                       world.foam_velocity, world.foam_count, n, 42, 1.0)
    c = int(world.foam_count.get()[0])
    world.foam_count[:] = 0
    assert c == world._max_foam
