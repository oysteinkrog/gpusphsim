"""In-place K_Integrate writeback on grid-reuse substeps (bd-r5-review-n28.196).

On a grid-reuse WCSPH substep that is not the frame's last, K_Integrate
writes to the sorted arrays at slot i and the next substep skips
K_GatherReorder. The frame's last substep scatters to the unsorted arrays.
These tests check:

1. Kernel level: in place gives the same state as a scatter, keeps the
   state-dirty bit, and rewrites the FP16 mirrors; the scatter after an
   in-place run stores the dirty health and every sleeper's zero velocity.
2. Simulation level: a frame of four reuse substeps with in-place
   writeback leaves the same unsorted state (to float rounding) as four
   gather substeps, on the
   direct path and on the captured graphs, and launches one gather instead
   of four.
3. The plan is off for PBF and on the frame's last substep, and step_frame
   never returns with the unsorted arrays stale.

All readbacks happen after the GPU work, in test code only.
"""

from __future__ import annotations

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler

        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, "_get_arch_for_options_for_nvrtc"):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, "_cache"):
                fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402

pytestmark = pytest.mark.gpu

DT = 0.001
STATE_DIRTY_BIT = 0x200000
SLEEPING_BIT = 0x400

UNSORTED_FIELDS = [
    "position",
    "velocity",
    "mass",
    "packed_info",
    "temperature",
    "health",
    "lifetime",
    "sleep_counter",
]


# ---------------------------------------------------------------------------
# 1. Kernel level
# ---------------------------------------------------------------------------


def _kernel_inputs():
    """Four particles: awake WATER (dirty), sleeping SAND, STONE, awake SAND."""
    from test_integrate import MAKE_PACKED, MASS, setup_params
    from materials import FLUID, GRANULAR, STATIC, WATER, SAND, STONE

    setup_params()
    n = 4
    pos = np.zeros((n, 4), np.float32)
    pos[:, 0] = [0.0, 0.2, 0.4, -0.2]
    pos[:, 3] = 1000.0
    vel = np.zeros((n, 4), np.float32)
    vel[0, :3] = (0.1, -0.2, 0.3)
    vel[3, :3] = (0.5, 0.0, 0.0)
    pi = np.array(
        [
            MAKE_PACKED(WATER, FLUID) | STATE_DIRTY_BIT,
            MAKE_PACKED(SAND, GRANULAR) | SLEEPING_BIT,
            MAKE_PACKED(STONE, STATIC),
            MAKE_PACKED(SAND, GRANULAR),
        ],
        np.uint32,
    )
    force = np.zeros((n, 4), np.float32)
    force[0, :3] = (1.0, 2.0, 3.0)
    return {
        "sorted_position": cp.asarray(pos),
        "sorted_velocity": cp.asarray(vel),
        "sorted_veleval": cp.asarray(vel),
        "sorted_sph_force": cp.asarray(force),
        "sorted_mass": cp.full(n, MASS, cp.float32),
        "sorted_packed_info": cp.asarray(pi),
        "sorted_temperature": cp.asarray(np.array([300, 310, 320, 330], np.float32)),
        "sorted_health": cp.asarray(np.array([0.5, 1, 1, 1], np.float32)),
        "sorted_lifetime": cp.zeros(n, cp.float32),
        "sorted_density": cp.full(n, 1000.0, cp.float32),
        "sorted_shear_rate": cp.zeros(n, cp.float32),
        "sorted_dTdt": cp.full(n, 5.0, cp.float32),
        "sorted_sleep_counter": cp.asarray(np.array([0, 200, 0, 0], np.uint8)),
    }


def _copy(d):
    return {k: v.copy() for k, v in d.items()}


def test_in_place_matches_scatter_and_keeps_dirty_bit():
    """Mode 1 leaves the sorted arrays equal to a mode-0 scatter through the
    identity permutation, except that the dirty bit stays set."""
    from integrate import integrate

    ref_in = _kernel_inputs()
    n = 4
    perm = cp.arange(n, dtype=cp.uint32)
    outs = {
        "position_out": cp.zeros((n, 4), cp.float32),
        "velocity_out": cp.zeros((n, 4), cp.float32),
        "packed_info_out": cp.zeros(n, cp.uint32),
        "sleep_counter_out": cp.zeros(n, cp.uint8),
        "temperature_out": cp.zeros(n, cp.float32),
        "health_out": cp.zeros(n, cp.float32),
    }
    integrate(**_copy(ref_in), sort_indexes=perm, **outs)

    ip = _copy(ref_in)
    vel_h = cp.zeros((n, 2), cp.uint32)
    temp_h = cp.zeros(n, cp.uint16)
    integrate(**ip, in_place=True, velocity_h_out=vel_h, temperature_h_out=temp_h)

    np.testing.assert_array_equal(
        ip["sorted_position"].get(), outs["position_out"].get()
    )
    np.testing.assert_array_equal(
        ip["sorted_velocity"].get(), outs["velocity_out"].get()
    )
    np.testing.assert_array_equal(
        ip["sorted_temperature"].get(), outs["temperature_out"].get()
    )
    np.testing.assert_array_equal(
        ip["sorted_sleep_counter"].get(), outs["sleep_counter_out"].get()
    )
    pi_ip = ip["sorted_packed_info"].get()
    pi_sc = outs["packed_info_out"].get()
    assert pi_ip[0] & STATE_DIRTY_BIT, "in place must keep the dirty bit"
    assert not pi_sc[0] & STATE_DIRTY_BIT, "the scatter clears the dirty bit"
    np.testing.assert_array_equal(pi_ip & ~np.uint32(STATE_DIRTY_BIT), pi_sc)
    # health stays in sorted space; the scatter carried it out
    assert ip["sorted_health"].get()[0] == 0.5
    assert outs["health_out"].get()[0] == 0.5

    # FP16 mirrors match what K_GatherReorder would build from the result
    v = ip["sorted_velocity"].get()
    m = ref_in["sorted_mass"].get()
    want = np.concatenate([v[:, :3], (m / np.float32(0.02))[:, None]], axis=1)
    got = vel_h.get().view(np.float16).reshape(n, 4).astype(np.float32)
    np.testing.assert_allclose(
        got, want.astype(np.float16).astype(np.float32), rtol=0, atol=0
    )
    assert (temp_h.get() != 0).all()


def test_scatter_after_in_place_stores_sleeper_zero_velocity():
    """Mode 2 stores a sleeper's zero velocity even when the sorted input is
    already zero; mode 0 relies on the unsorted copy being zero already."""
    from integrate import integrate

    n = 4
    perm = cp.arange(n, dtype=cp.uint32)

    def run(after):
        d = _kernel_inputs()
        vel_out = cp.full((n, 4), 7.0, cp.float32)  # stale unsorted velocity
        health_out = cp.ones(n, cp.float32)
        integrate(
            **d,
            sort_indexes=perm,
            position_out=cp.zeros((n, 4), cp.float32),
            velocity_out=vel_out,
            packed_info_out=cp.zeros(n, cp.uint32),
            sleep_counter_out=cp.zeros(n, cp.uint8),
            temperature_out=cp.zeros(n, cp.float32),
            health_out=health_out,
            after_in_place=after,
        )
        return vel_out.get(), health_out.get()

    vel0, health0 = run(False)
    vel2, health2 = run(True)
    assert (vel0[1] == 7.0).all(), "mode 0 skips the store for a zero input"
    assert (vel2[1] == 0.0).all(), "mode 2 must store the sleeper's zero"
    assert health0[0] == 0.5 and health2[0] == 0.5


def test_in_place_rejects_static_pushout():
    from integrate import integrate

    d = _kernel_inputs()
    cs = cp.zeros(8, cp.uint32)
    with pytest.raises(ValueError):
        integrate(**d, in_place=True, cell_start=cs, cell_end=cs)


# ---------------------------------------------------------------------------
# 2. Simulation level
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    from materials import WATER, SAND
    from simulation import Simulation
    from world import World

    w = World(max_particles=40_000)
    w.dead_reserve = 0
    w.spawn_cube(
        min_corner=(-0.5, -0.9, -0.2),
        max_corner=(-0.1, -0.5, 0.2),
        material_id=WATER,
        spacing=0.02,
    )
    w.spawn_cube(
        min_corner=(0.1, -0.9, -0.2),
        max_corner=(0.4, -0.7, 0.2),
        material_id=SAND,
        spacing=0.02,
    )
    s = Simulation(w, dt=DT, speed=1.0, accuracy=0.4, fixed_dt=True, max_substeps=4)
    s.fixed_dt_value = DT
    s._upload_dt(DT)
    s._spawn_substep = s._damping_duration
    n = w._high_water
    for _ in range(40):
        s._sim_step(n)
    cp.cuda.Device().synchronize()
    return s


def _device_state(sim):
    """Every device array the substep reads or writes."""
    arrs = {}
    for owner, name in ((sim.world, "w"), (sim, "s")):
        for k, v in vars(owner).items():
            if isinstance(v, cp.ndarray):
                arrs[(name, k)] = v
    return arrs


def _save(sim):
    return {k: v.copy() for k, v in _device_state(sim).items()}


def _restore(sim, saved, host):
    for k, v in _device_state(sim).items():
        if k in saved and saved[k].shape == v.shape:
            cp.copyto(v, saved[k])
    for k, v in host.items():
        setattr(sim, k, v)


def _host_state(sim):
    keys = [
        "sim_time",
        "_substep_counter",
        "_substeps_since_sort",
        "_sim_time_since_sort",
        "_frame_substeps",
        "_sort_skipped_count",
    ]
    return {k: getattr(sim, k) for k in keys}


def _run_frame(sim, n, in_place, timed):
    fn = sim._sim_step_timed if timed else sim._sim_step
    for i in range(4):
        last = i == 3
        fn(n, last_substep=last, in_place_ok=in_place and not last)
    cp.cuda.Device().synchronize()
    return {f: getattr(sim.world, f)[:n].get() for f in UNSORTED_FIELDS}


@pytest.mark.parametrize("timed", [True, False], ids=["direct", "graph"])
def test_frame_in_place_matches_gather(sim, timed, monkeypatch):
    import counting_sort

    n = sim.world._high_water
    # Full sort now, then reuse the grid for the next frame.
    sim._sort_skip_next = False
    sim._sim_step(n)
    sim._sort_skip_next = True
    cp.cuda.Device().synchronize()
    saved, host = _save(sim), _host_state(sim)

    calls = []
    real = counting_sort.gather_reorder
    monkeypatch.setattr(
        counting_sort,
        "gather_reorder",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )

    # The graph path needs a second frame: the first call per variant runs
    # directly (n change), the second captures, the third replays.
    reps = 1 if timed else 3
    for _ in range(reps):
        _restore(sim, saved, host)
        calls.clear()
        ref = _run_frame(sim, n, in_place=False, timed=timed)
        ref_calls = len(calls)
    for _ in range(reps):
        _restore(sim, saved, host)
        calls.clear()
        got = _run_frame(sim, n, in_place=True, timed=timed)
        got_calls = len(calls)
    assert not sim._sorted_is_current

    # Two gather runs already differ in the last bits of some velocities
    # (float reductions in step2 are not bit-reproducible), so floats get a
    # tolerance; flags, counters and reaction state must match exactly.
    for f in UNSORTED_FIELDS:
        if got[f].dtype.kind == "f":
            np.testing.assert_allclose(got[f], ref[f], rtol=1e-5, atol=1e-5, err_msg=f)
        else:
            np.testing.assert_array_equal(got[f], ref[f], err_msg=f)
    if timed:
        assert ref_calls == 4
        assert got_calls == 1, "only the frame's first substep gathers"
    else:
        assert sim._graph_reuse_variants, "the in-place variants were captured"
    sim._sort_skip_next = False


def test_plan_off_for_last_substep_and_pbf(sim):
    from solver_profiles import PROFILES

    n = sim.world._high_water
    sim._sort_skip_next = False
    sim._sim_step(n)
    sim._sort_skip_next = True
    sim._plan_substep_writeback(n, in_place_ok=False)
    assert not sim._integrate_in_place
    sim._plan_substep_writeback(n, in_place_ok=True)
    assert sim._integrate_in_place
    # A full sort next: never in place.
    sim._sort_skip_next = False
    sim._plan_substep_writeback(n, in_place_ok=True)
    assert not sim._integrate_in_place
    # Stale unsorted arrays before a full sort is a bug, not a silent path.
    sim._sorted_is_current = True
    with pytest.raises(RuntimeError):
        sim._plan_substep_writeback(n, in_place_ok=True)
    sim._sorted_is_current = False
    prof = sim._profile
    try:
        sim._profile = PROFILES["PBF"]
        sim._sort_skip_next = True
        sim._plan_substep_writeback(n, in_place_ok=True)
        assert not sim._integrate_in_place
    finally:
        sim._profile = prof
        sim._sort_skip_next = False
        sim._integrate_in_place = False
