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


def _seed(v: int) -> cp.ndarray:
    """RNG seed as the (1,) uint32 device array K_FoamGenerate reads."""
    return cp.asarray([v], dtype=cp.uint32)


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
                           world.foam_velocity, world.foam_count, n, _seed(seed), dt_emit)
        counts.append(int(world.foam_count.get()[0]))
    world.foam_count[:] = 0
    for c in counts:
        assert abs(c - expect) < 0.15 * expect, f"emitted {counts}, want about {expect:.0f}"
    # Doubling dt_emit doubles the expected count.
    foam.foam_generate(pos, vel, nrm, info, world.foam_position,
                       world.foam_velocity, world.foam_count, n, _seed(7), 2 * dt_emit)
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
                           world.foam_velocity, world.foam_count, n, _seed(42), 0.01,
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
                       world.foam_velocity, world.foam_count, n, _seed(42), 1.0)
    assert int(world.foam_count.get()[0]) == 0
    nrm[:, 3] = 10.0
    world.foam_count[:] = world._max_foam
    foam.foam_generate(pos, vel, nrm, info, world.foam_position,
                       world.foam_velocity, world.foam_count, n, _seed(42), 1.0)
    c = int(world.foam_count.get()[0])
    world.foam_count[:] = 0
    assert c == world._max_foam


# ---------------------------------------------------------------------------
# Device RNG counter and foam physics in the graph (bd-r5-review-n28.180)
# ---------------------------------------------------------------------------

def test_substep_counter_advances_once_per_substep(world):
    """5 substeps (direct run, capture, replays) advance _frame_counter_d by 5."""
    world.foam_count[:] = 0
    world.spawn_cube(min_corner=(-0.1, -0.1, -0.1), max_corner=(0.1, 0.1, 0.1),
                     material_id=5, spacing=0.02)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=20)
    sim._spawn_substep = sim._damping_duration  # no ramp: graphs get used
    n = world.num_active
    enabled = world.foam_enabled
    world.foam_enabled = True
    try:
        for _ in range(3):  # warm up until graphs are captured
            sim._sim_step(n)
        start = int(sim._frame_counter_d.get()[0])
        seeds = []
        for _ in range(5):
            sim._sim_step(n)
            seeds.append(int(sim._frame_counter_d.get()[0]))
        assert sim._graph_full_sort is not None or sim._graph_gather_only is not None
        assert seeds == [start + k for k in range(1, 6)], (start, seeds)
    finally:
        world.foam_enabled = enabled
        world.foam_count[:] = 0
        sim.close()


def test_foam_physics_runs_on_graph_replay(world):
    """Foam moves on every replayed substep, and survives a compaction."""
    world.foam_count[:] = 0
    world.spawn_cube(min_corner=(-0.1, -0.1, -0.1), max_corner=(0.1, 0.1, 0.1),
                     material_id=5, spacing=0.02)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=20)
    sim._spawn_substep = sim._damping_duration
    n = world.num_active
    enabled = world.foam_enabled
    world.foam_enabled = True
    try:
        for _ in range(3):
            sim._sim_step(n, last_substep=False)
        # One SPRAY particle mid-air with a long life.
        k = 1
        pos = np.zeros((k, 4), dtype=np.float32)
        pos[:, :3] = _mid(sim)
        pos[:, 1] += 0.3
        vel = np.zeros((k, 4), dtype=np.float32)
        vel[:, 3] = 100.0
        _seed_foam(world, pos, vel)
        ys = []
        for _ in range(4):
            sim._sim_step(n, last_substep=False)
            ys.append(float(world.foam_position[0, 1].get()))
        assert all(b < a for a, b in zip(ys, ys[1:])), ys
        # Compaction copies back into the same arrays the graph captured.
        ptr = world.foam_position.data.ptr
        sim._frame_counter = 8
        sim._maybe_compact_foam()
        assert world.foam_position.data.ptr == ptr
        assert int(world.foam_count.get()[0]) == 1
        y0 = float(world.foam_position[0, 1].get())
        sim._sim_step(n, last_substep=False)
        assert float(world.foam_position[0, 1].get()) < y0
    finally:
        world.foam_enabled = enabled
        world.foam_count[:] = 0
        sim.close()


def test_generate_seed_differs_between_substeps(world):
    """Two counter values give different emission sets for the same input."""
    n = 10_000
    foam.upload_foam_params(foam.build_foam_params(max_foam=world._max_foam))
    pos, vel, nrm, info = _surface_inputs(n, 3.0, 10.0)
    runs = []
    for v in (11, 12):
        world.foam_count[:] = 0
        foam.foam_generate(pos, vel, nrm, info, world.foam_position,
                           world.foam_velocity, world.foam_count, n, _seed(v), 0.01)
        c = int(world.foam_count.get()[0])
        runs.append(cp.asnumpy(world.foam_velocity[:c, 3]))
    world.foam_count[:] = 0
    assert runs[0].size > 0 and runs[1].size > 0
    assert not (runs[0].size == runs[1].size
                and np.array_equal(np.sort(runs[0]), np.sort(runs[1])))


# ---------------------------------------------------------------------------
# Foam collides with SDF objects (bd-r5-review-n28.194)
# ---------------------------------------------------------------------------

def test_spray_lands_on_sdf_box(world, sim):
    """A SPRAY particle falling onto an SDF box ends on its top, not inside."""
    import rigid_bodies as rb

    mid = _mid(sim)
    wmin = np.array(sim._sim_params[0]["world_min"], dtype=np.float32)
    half = np.array([0.1, 0.05, 0.1], dtype=np.float32)
    center = mid.copy()
    center[1] = wmin[1] + 0.2  # box well above the floor kill plane
    box = np.atleast_1d(rb.make_sdf_object(rb.SDF_BOX, position=tuple(center.tolist()),
                             size=tuple(half.tolist())))
    top = float(center[1] + half[1])

    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0, :3] = center
    pos[0, 1] = top + 0.1
    pos[0, 3] = 0.0  # SPRAY
    vel = np.zeros((1, 4), dtype=np.float32)
    vel[0, 1] = -3.0
    vel[0, 3] = 100.0
    rb.upload_sdf_objects(box, 1)
    try:
        _seed_foam(world, pos, vel)
        for _ in range(200):
            foam.foam_physics(world.foam_position, world.foam_velocity,
                              world.foam_count, world._max_foam)
        p = cp.asnumpy(world.foam_position[0])
        v = cp.asnumpy(world.foam_velocity[0])
    finally:
        rb.upload_sdf_objects(None, 0)
        world.foam_count[:] = 0

    local = np.abs(p[:3] - center) - half
    sdf = float(np.linalg.norm(np.maximum(local, 0.0))
                + min(float(local.max()), 0.0))
    assert v[3] > 0, "particle must still be alive"
    assert sdf >= 0.0
    assert p[1] >= top
    assert abs(p[1] - top) < 1e-2  # resting on the top face
    assert p[3] == 1.0  # SPRAY became FOAM on contact


def test_sdf_replay_reaches_late_foam_module(world, sim):
    """A foam module compiled after the SDF upload gets the objects too."""
    import rigid_bodies as rb

    box = np.atleast_1d(rb.make_sdf_object(rb.SDF_BOX, position=(0.0, 0.0, 0.0)))
    rb.upload_sdf_objects(box, 1)
    try:
        saved = foam._module
        foam._module = None
        try:
            mod = foam._get_module()
            n = np.zeros(1, dtype=np.int32)
            cp.cuda.runtime.memcpy(n.ctypes.data,
                                   int(mod.get_global("c_num_sdf_objects")),
                                   4, 2)
        finally:
            foam._module = saved
    finally:
        rb.upload_sdf_objects(None, 0)
    assert n[0] == 1


# ---------------------------------------------------------------------------
# In-fluid test from the SPH hash grid (bd-r5-review-n28.193)
# ---------------------------------------------------------------------------
def _host_cell_hash(sim, p: np.ndarray) -> int:
    """calcGridCell + spatialHash from common.cuh, on the host."""
    g = sim._grid_params[0]
    gmin = np.asarray(g["grid_min"], dtype=np.float32)
    delta = np.asarray(g["grid_delta"], dtype=np.float32)
    c = np.floor((p.astype(np.float32) - gmin) * delta).astype(np.int64)
    primes = (73856093, 19349669, 83492791)
    h = sum((int(c[k]) & 0xFFFFFFFF) * primes[k] for k in range(3))
    return int(h & 0xFFFFFFFF) & int(g["table_mask"])


def _cell_tables(sim, occupied_hashes) -> tuple:
    """Separate cell_start/cell_end arrays: one particle in each listed hash."""
    size = int(sim._grid_params[0]["table_size"])
    start = np.zeros(size, dtype=np.uint32)
    end = np.zeros(size, dtype=np.uint32)
    for h in occupied_hashes:
        end[h] = 1
    return cp.asarray(start), cp.asarray(end)


def _step_one(world, sim, pos, vel, occupied):
    _seed_foam(world, pos, vel)
    cs, ce = _cell_tables(sim, occupied)
    foam.foam_physics(world.foam_position, world.foam_velocity,
                      world.foam_count, world._max_foam, cs, ce)
    p = cp.asnumpy(world.foam_position[:pos.shape[0]])
    v = cp.asnumpy(world.foam_velocity[:pos.shape[0]])
    world.foam_count[:] = 0
    return p, v


def test_bubble_in_empty_cell_becomes_foam(world, sim):
    """A BUBBLE in an empty cell becomes FOAM with vy = 0 after one step."""
    mid = _mid(sim)
    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0, :3] = mid
    pos[0, 3] = 2.0  # BUBBLE
    vel = np.zeros((1, 4), dtype=np.float32)
    vel[0, 1] = 0.8
    vel[0, 3] = 100.0
    p, v = _step_one(world, sim, pos, vel, occupied=[])
    assert p[0, 3] == 1.0, f"type {p[0, 3]}, want FOAM"
    # vy is zeroed, then one substep of weak gravity (0.1 g dt).
    g = float(sim._sim_params[0]["gravity"][1])
    assert abs(v[0, 1] - 0.1 * g * sim.dt) < 1e-5


def test_bubble_in_fluid_cell_stays_bubble(world, sim):
    """A BUBBLE whose cell holds a particle keeps rising as a BUBBLE."""
    mid = _mid(sim)
    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0, :3] = mid
    pos[0, 3] = 2.0
    vel = np.zeros((1, 4), dtype=np.float32)
    vel[0, 3] = 100.0
    p, v = _step_one(world, sim, pos, vel, occupied=[_host_cell_hash(sim, mid)])
    assert p[0, 3] == 2.0
    assert v[0, 1] > 0.0


def test_foam_in_fluid_cell_rises(world, sim):
    """FOAM in a cell with fluid gets positive ay; in an empty cell it falls."""
    mid = _mid(sim)
    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0, :3] = mid
    pos[0, 3] = 1.0  # FOAM
    vel = np.zeros((1, 4), dtype=np.float32)
    vel[0, 3] = 100.0
    _, v_in = _step_one(world, sim, pos, vel,
                        occupied=[_host_cell_hash(sim, mid)])
    _, v_out = _step_one(world, sim, pos, vel, occupied=[])
    assert v_in[0, 1] > 0.0, f"in-fluid FOAM vy {v_in[0, 1]}, want > 0"
    assert v_out[0, 1] < 0.0


def test_foam_rises_to_top_of_fluid_layer(world, sim):
    """FOAM released at the bottom of a 3-cell fluid column ends near its top."""
    mid = _mid(sim)
    cell = 1.0 / float(sim._grid_params[0]["grid_delta"][0])
    gmin = np.asarray(sim._grid_params[0]["grid_min"], dtype=np.float32)
    # Snap to a cell centre so the column cells are exact.
    base = gmin + (np.floor((mid - gmin) / cell) + 0.5) * cell
    column = [base + np.array([0, k * cell, 0], dtype=np.float32) for k in range(3)]
    occupied = [_host_cell_hash(sim, c) for c in column]
    top = float(column[-1][1] + 0.5 * cell)

    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0, :3] = column[0]
    pos[0, 3] = 1.0
    vel = np.zeros((1, 4), dtype=np.float32)
    vel[0, 3] = 100.0
    _seed_foam(world, pos, vel)
    cs, ce = _cell_tables(sim, occupied)
    for _ in range(int(2.0 / sim.dt)):
        foam.foam_physics(world.foam_position, world.foam_velocity,
                          world.foam_count, world._max_foam, cs, ce)
    p = cp.asnumpy(world.foam_position[0])
    world.foam_count[:] = 0
    assert p[3] == 1.0
    assert abs(p[1] - top) < 0.5 * cell, f"y {p[1]:.4f}, top {top:.4f}"
