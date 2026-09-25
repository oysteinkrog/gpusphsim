"""Tests for world.py -- World particle manager."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cupy as cp
import cupy.cuda.compiler as _compiler
import numpy as np
import pytest

# Blackwell (sm_120) PTX workaround
_compiler._use_ptx = True
for _fn in (_compiler._get_arch, _compiler._get_arch_for_options_for_nvrtc):
    if hasattr(_fn, '_cache'):
        _fn._cache = {}
from world import World, DEFAULT_SPACING, T_AMBIENT, _MAKE_PACKED, PARTICLE_MASS, _DEFAULT_TEMPS, spawn_mass
from materials import (
    MATERIALS, WATER, SAND, LAVA, FIRE, STEAM, SMOKE, ICE, STONE, DIRT, MUD,
    FLUID, GRANULAR, GAS, STATIC,
)


def test_constructor():
    w = World()
    assert w.max_particles == 500_000
    assert w.num_active == 0
    print("PASS: constructor default")

    w2 = World(100_000)
    assert w2.max_particles == 100_000
    assert w2.num_active == 0
    print("PASS: constructor custom")


def _render_color(world, n):
    """Render color of the first n slots, as the renderer gets it: one
    K_ComputeColor launch over the unsorted state (bd-r5-review-n28.145)."""
    from simulation import Simulation

    out = cp.zeros((n, 4), dtype=cp.float32)
    Simulation(world).compute_render_color(out, n)
    return out.get()


def test_array_shapes_dtypes():
    w = World(1000)
    n = 1000
    # float4 arrays
    for name in ["position", "velocity"]:
        arr = getattr(w, name)
        assert arr.shape == (n, 4), f"{name} shape: {arr.shape}"
        assert arr.dtype == cp.float32, f"{name} dtype: {arr.dtype}"
    # float arrays
    for name in ["mass", "temperature", "health", "lifetime"]:
        arr = getattr(w, name)
        assert arr.shape == (n,), f"{name} shape: {arr.shape}"
        assert arr.dtype == cp.float32, f"{name} dtype: {arr.dtype}"
    # uint32
    assert w.packed_info.shape == (n,)
    assert w.packed_info.dtype == cp.uint32
    # uint8
    assert w.sleep_counter.shape == (n,)
    assert w.sleep_counter.dtype == cp.uint8
    # All are CuPy arrays
    for name in ["position", "velocity", "mass", "packed_info", "temperature",
                 "health", "lifetime", "sleep_counter"]:
        assert isinstance(getattr(w, name), cp.ndarray), f"{name} not CuPy"
    print("PASS: array shapes, dtypes, and GPU residency")


@pytest.mark.parametrize("mat_id,expected", [
    (WATER, 0.02), (SAND, 0.032), (LAVA, 0.052), (MUD, 0.036),
    (STONE, 0.02), (ICE, 0.02), (STEAM, 0.02), (FIRE, 0.02),
])
def test_spawn_mass_per_material(mat_id, expected):
    """FLUID and GRANULAR spawn at 0.02 * rho0 / 2500; STATIC and GAS keep 0.02 (bd-r5-review-n28.53)."""
    assert spawn_mass(mat_id) == pytest.approx(expected, rel=1e-6)
    w = World(10_000)
    n = w.spawn_cube((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05), mat_id, spacing=0.02)
    assert n > 0
    m = w.mass[:n].get()
    np.testing.assert_allclose(m, expected, rtol=1e-6)


def test_spawn_cube_water():
    w = World(500_000)
    n = w.spawn_cube((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), WATER, spacing=0.02)
    assert n > 0
    assert w.num_active == n
    # WATER (rho0 2500) spawns with the reference mass PARTICLE_MASS = 0.02
    mass_val = float(w.mass[0])
    assert abs(mass_val - PARTICLE_MASS) < 1e-6, f"mass={mass_val}, expected PARTICLE_MASS={PARTICLE_MASS}"
    assert spawn_mass(WATER) == PARTICLE_MASS
    # packed_info = MAKE_PACKED(WATER, FLUID)
    pi_val = int(w.packed_info[0])
    expected = _MAKE_PACKED(WATER, FLUID)
    assert pi_val == expected, f"packed_info={pi_val} expected={expected}"
    # temperature = 293K (ambient)
    assert abs(float(w.temperature[0]) - T_AMBIENT) < 0.1
    # health = 1.0
    assert abs(float(w.health[0]) - 1.0) < 1e-6
    # Render color (K_ComputeColor, bd-r5-review-n28.145): color.w =
    # behavior_class * 0.25 (FLUID = 0) for the SSFR shaders, rgb in [0, 1].
    col = _render_color(w, n)
    assert np.all(col[:, 3] == FLUID * 0.25)
    assert np.all((col[:, :3] >= 0.0) & (col[:, :3] <= 1.0))
    print(f"PASS: spawn_cube water ({n} particles)")


def test_spawn_cube_max_particles():
    """Spawning up to max_particles water particles in a cube completes without error."""
    w = World(500_000)
    n = w.spawn_cube((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), WATER, spacing=0.02)
    assert n <= 500_000
    assert w.num_active == n
    print(f"PASS: spawn_cube max particles ({n} spawned, max 500K)")


def test_spawn_sphere():
    w = World(100_000)
    n = w.spawn_sphere((0.0, 0.5, 0.0), 0.3, LAVA, 5000)
    assert n == 5000
    assert w.num_active == 5000
    # All positions within sphere
    pos = w.position[:5000, :3]
    center = cp.array([0.0, 0.5, 0.0], dtype=cp.float32)
    diff = pos - center
    dist = cp.sqrt(cp.sum(diff * diff, axis=1))
    assert float(cp.max(dist)) <= 0.3 + 1e-5
    # Lava temperature = 1500K
    assert abs(float(w.temperature[0]) - 1500.0) < 0.1
    # packed_info = MAKE_PACKED(LAVA, FLUID) - lava is FLUID
    pi_val = int(w.packed_info[0])
    expected = _MAKE_PACKED(LAVA, FLUID)
    assert pi_val == expected, f"packed_info={pi_val} expected={expected}"
    print("PASS: spawn_sphere lava")


def test_spawn_fire_temp():
    w = World(10_000)
    n = w.spawn_sphere((0.0, 0.0, 0.0), 0.1, FIRE, 100)
    assert n == 100
    assert abs(float(w.temperature[0]) - 1200.0) < 0.1
    print("PASS: fire temperature 1200K")


def test_spawn_steam_temp():
    w = World(10_000)
    n = w.spawn_sphere((0.0, 0.0, 0.0), 0.1, STEAM, 100)
    assert n == 100
    assert abs(float(w.temperature[0]) - 373.0) < 0.1
    print("PASS: steam temperature 373K")


def test_kill_in_sphere():
    w = World(100_000)
    w.spawn_sphere((0.0, 0.0, 0.0), 0.5, WATER, 10_000)
    before = w.num_active
    assert before == 10_000
    # count=True returns the number killed; the UI brush uses it so main.py
    # sees a nonzero delta (bd-r5-review-n28.155).
    killed = w.kill_in_sphere((0.0, 0.0, 0.0), 0.25, count=True)
    # Killed slots are MAKE_PACKED(DEAD, STATIC) with zero mass, health and
    # velocity, and render with color (0, 0, 0, -1) so the vertex shaders
    # cull them (bd-r5-review-n28.34, bd-r5-review-n28.60).
    n = w._high_water
    dead_mask = (w.packed_info[:n] & 0xFF) == 0
    dead_count = int(cp.sum(dead_mask))
    assert dead_count > 0, "Expected some particles to be killed inside sphere"
    assert killed == dead_count, f"kill_in_sphere returned {killed}, {dead_count} slots are DEAD"
    # A second kill in the same sphere finds nothing live and returns 0.
    assert w.kill_in_sphere((0.0, 0.0, 0.0), 0.25, count=True) == 0
    assert bool(cp.all(w.packed_info[:n][dead_mask] == 0x300))
    assert bool(cp.all(w.mass[:n][dead_mask] == 0.0))
    assert bool(cp.all(w.health[:n][dead_mask] == 0.0))
    assert bool(cp.all(w.velocity[:n][dead_mask] == 0.0))
    col = _render_color(w, n)
    dead_h = dead_mask.get()
    assert np.all(col[dead_h, :3] == 0.0)
    assert np.all(col[dead_h, 3] == -1.0)
    assert np.all(col[~dead_h, 3] == 0.0)
    # Particles outside the sphere are untouched.
    r2 = cp.sum(w.position[:n, :3] ** 2, axis=1)
    assert bool(cp.all(dead_mask == (r2 <= 0.25 * 0.25)))
    assert bool(cp.all(w.mass[:n][~dead_mask] > 0.0))
    print(f"PASS: kill_in_sphere ({dead_count} killed)")


def test_num_active():
    w = World(10_000)
    assert w.num_active == 0
    n1 = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), SAND, 0.02)
    assert w.num_active == n1
    n2 = w.spawn_sphere((0.5, 0.5, 0.5), 0.1, WATER, 500)
    assert w.num_active == n1 + n2
    print(f"PASS: num_active tracking ({n1} + {n2} = {n1 + n2})")


def test_resize():
    w = World(10_000)
    w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, 0.02)
    assert w.num_active > 0
    w.resize(200_000)
    assert w.max_particles == 200_000
    assert w.num_active == 0
    assert w.position.shape == (200_000, 4)
    assert w.packed_info.shape == (200_000,)
    print("PASS: resize")


def test_resize_reallocates_solver_arrays():
    """resize() must re-create lazy PBF/DFSPH arrays at the new size (grow and shrink)."""
    names = ('predicted_position', 'sorted_predicted_position',
             'sorted_delta_position', 'sorted_pressure_normal',
             'sorted_alpha_dfsph')
    for old_n, new_n in ((1000, 2000), (2000, 1000)):
        w = World(old_n)
        w.allocate_pbf_arrays()
        w.allocate_dfsph_arrays()
        for name in names:
            assert getattr(w, name).shape[0] == old_n, name
        w.resize(new_n)
        w.allocate_pbf_arrays()
        w.allocate_dfsph_arrays()
        for name in names:
            assert getattr(w, name).shape[0] == new_n, (name, old_n, new_n)

    # Solver arrays are only created by the solver that needs them.
    w = World(1000)
    w.allocate_dfsph_arrays()
    w.resize(3000)
    assert not hasattr(w, 'sorted_alpha_dfsph')
    w.allocate_pbf_arrays()
    for name in names[:4]:
        assert getattr(w, name).shape[0] == 3000, name

    # A size mismatch reallocates even without resize().
    w = World(1000)
    w.allocate_pbf_arrays()
    w.max_particles = 1500
    w.allocate_pbf_arrays()
    assert w.predicted_position.shape[0] == 1500
    print("PASS: resize reallocates solver arrays")


def test_contiguous_arrays():
    """All arrays are contiguous CuPy arrays on GPU."""
    w = World(1000)
    for name in ["position", "velocity", "mass", "packed_info", "temperature",
                 "health", "lifetime", "sleep_counter"]:
        arr = getattr(w, name)
        assert isinstance(arr, cp.ndarray)
        # Check contiguous (C-order by default)
        assert arr.flags["C_CONTIGUOUS"], f"{name} not C-contiguous"
    print("PASS: all arrays contiguous CuPy on GPU")


def test_ice_spawn_temp():
    """bd-mzc.18 regression: ICE spawns below melt point (253 K), not at ambient 293 K."""
    # _DEFAULT_TEMPS must have ICE entry at or below 273 K
    assert ICE in _DEFAULT_TEMPS, "_DEFAULT_TEMPS missing ICE entry (mat id 11)"
    ice_default = _DEFAULT_TEMPS[ICE]
    assert ice_default <= 273.0, (
        f"ICE default temp {ice_default} K >= melt point 273 K; ICE would melt on frame 1"
    )
    # Verify spawn actually uses it
    w = World(10_000)
    n = w.spawn_sphere((0.0, 0.0, 0.0), 0.1, ICE, 100)
    assert n == 100
    temps = w.temperature[:100].get()
    for i in range(100):
        assert temps[i] <= 273.0, (
            f"Particle {i}: spawn temp {temps[i]} K >= melt point 273 K"
        )
    print(f"PASS: test_ice_spawn_temp (default={ice_default} K)")


def test_num_active_counts_dead_particles():
    """bd-mzc.38 regression: num_active must accurately reflect killed particles.

    The first attempt at bd-mzc.38 replaced the count_nonzero with a host-side
    counter that was never maintained, so num_active always returned
    _high_water and broke compaction accounting (test_compact regressions).
    num_active must report the true non-DEAD count. The per-frame no-sync
    guarantee is provided by _maybe_compact's interval gate (it is the only
    per-frame caller and runs once every compact_interval frames), not by
    making this query non-syncing.
    """
    w = World(10_000)
    w.spawn_sphere((0.0, 0.0, 0.0), 0.1, WATER, 500)
    n0 = w.num_active
    assert n0 == 500, f"expected 500 active after spawn, got {n0}"

    # Kill roughly half via a large brush sphere, then re-query.
    w.kill_in_sphere((0.0, 0.0, 0.0), 0.05)
    n1 = w.num_active
    assert 0 <= n1 < n0, f"num_active ({n1}) must drop after kill (was {n0})"
    print("PASS: test_num_active_counts_dead_particles")


def test_kill_in_sphere_no_gpu_sync():
    """bd-mzc.39 regression: kill_in_sphere must not call int(cp.sum(...))."""
    import unittest.mock as mock
    w = World(10_000)
    w.spawn_sphere((0.0, 0.0, 0.0), 0.5, WATER, 1_000)

    original_sum = cp.sum
    int_sum_calls = []

    # Track if int() is called on a CuPy result from cp.sum
    original_int = __builtins__.__class__.__mro__  # just to have a ref

    # We patch kill_in_sphere indirectly: track cp.sum calls that return arrays
    sum_calls = []
    def mock_sum(*a, **kw):
        result = original_sum(*a, **kw)
        sum_calls.append(result)
        return result

    with mock.patch("world.cp.sum", side_effect=mock_sum):
        # kill_in_sphere uses cp.sum inside; after fix it should not call int() on it
        # We check via monkeypatching builtins.int and verifying no call on CuPy array
        original_builtin_int = __builtins__["int"] if isinstance(__builtins__, dict) else int
        cupy_int_calls = []

        import builtins
        orig_int = builtins.int
        def spy_int(x, *a, **kw):
            if isinstance(x, cp.ndarray):
                cupy_int_calls.append(x)
            return orig_int(x, *a, **kw)

        builtins.int = spy_int
        try:
            # Default count=False: no readback, returns 0.
            assert w.kill_in_sphere((0.0, 0.0, 0.0), 0.25) == 0
        finally:
            builtins.int = orig_int

    assert len(cupy_int_calls) == 0, (
        f"kill_in_sphere called int() on a CuPy array {len(cupy_int_calls)} time(s) (GPU sync); bd-mzc.39 not fixed"
    )
    print("PASS: test_kill_in_sphere_no_gpu_sync")


def test_spawn_sphere_no_loop_sync():
    """bd-mzc.40 regression: spawn_sphere must not call len() on a CuPy array in its hot path."""
    import builtins
    w = World(10_000)

    orig_len = builtins.len
    cupy_len_calls = []
    def spy_len(x):
        if isinstance(x, cp.ndarray):
            cupy_len_calls.append(type(x).__name__)
        return orig_len(x)

    builtins.len = spy_len
    try:
        n = w.spawn_sphere((0.0, 0.0, 0.0), 0.3, WATER, 500)
    finally:
        builtins.len = orig_len

    assert n == 500, f"spawn_sphere returned {n}, expected 500"
    assert len(cupy_len_calls) == 0, (
        f"spawn_sphere called len() on CuPy array {len(cupy_len_calls)} time(s) (GPU sync); bd-mzc.40 not fixed"
    )
    print("PASS: test_spawn_sphere_no_loop_sync")


# ---------------------------------------------------------------------------
# bd-r5-review-n28.96: shared spawn slot init
# ---------------------------------------------------------------------------

def _dirty_world(n=20_000):
    """World whose per-particle arrays hold 1.0, as if a previous scene used them."""
    w = World(n)
    for arr in (w.position, w.velocity, w.kappa, w.kappa_v, w.lifetime,
                w.sorted_lambda_pbf):
        arr[:] = 1.0
    return w


_SPAWNERS = {
    "sphere": lambda w, m, **kw: w.spawn_sphere((0.0, 0.0, 0.0), 0.2, m, 300, **kw),
    "cube": lambda w, m, **kw: w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), m, **kw),
    "ramp": lambda w, m, **kw: w.spawn_ramp((-0.2, 0.2, 0.0), (0.2, 0.0, 0.0), 0.1, 0.04, m, **kw),
    "bowl": lambda w, m, **kw: w.spawn_bowl((0.0, 0.0, 0.0), 0.15, 0.04, m, **kw),
    "cylinder_shell": lambda w, m, **kw: w.spawn_cylinder_shell((0.0, 0.0, 0.0), 0.1, 0.1, 0.04, m, **kw),
}


@pytest.mark.parametrize("shape", sorted(_SPAWNERS))
@pytest.mark.parametrize("material_id", [WATER, STONE, FIRE])
def test_spawn_clears_stale_slot_state(shape, material_id):
    """Every spawn_* zeroes warm-start state and packs rest density in position.w."""
    w = _dirty_world()
    n = _SPAWNERS[shape](w, material_id)
    assert n > 0 and w._high_water == n
    sl = slice(0, n)
    mat = MATERIALS[material_id]
    for name in ("kappa", "kappa_v"):
        vals = getattr(w, name)[sl].get()
        assert np.all(vals == 0.0), f"{shape}: {name} kept stale values"
    assert np.all(w.velocity[sl, 3].get() == 0.0)
    np.testing.assert_array_equal(w.position[sl, 3].get(), np.float32(mat.rest_density))
    expected_lt = 1.5 if material_id == FIRE else 0.0
    np.testing.assert_allclose(w.lifetime[sl].get(), expected_lt)
    # Slots past the spawn are untouched
    assert float(w.kappa[n].get()) == 1.0


def test_spawn_gas_default_lifetime():
    """GAS spawns get the per-material lifetime, other materials 0, explicit values win."""
    for mat_id, expected in ((FIRE, 1.5), (SMOKE, 3.0), (STEAM, 5.0), (SAND, 0.0), (WATER, 0.0)):
        w = World(10_000)
        n = w.spawn_sphere((0.0, 0.0, 0.0), 0.2, mat_id, 200)
        np.testing.assert_allclose(w.lifetime[:n].get(), expected,
                                   err_msg=f"material {mat_id}")
    w = World(10_000)
    n = w.spawn_sphere((0.0, 0.0, 0.0), 0.2, FIRE, 200, lifetime=0.0)
    assert np.all(w.lifetime[:n].get() == 0.0), "explicit lifetime=0 not honoured"
    n2 = w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), FIRE, lifetime=0.7)
    np.testing.assert_allclose(w.lifetime[n:n + n2].get(), 0.7)


def test_spawned_fire_turns_to_smoke_then_dies():
    """Brush-spawned FIRE becomes SMOKE by 1.5 s and DEAD by 4.5 s of K_Reactions time."""
    from reactions import compute_reactions, upload_sim_params, upload_materials
    from step1 import build_sim_params
    from materials import build_material_array

    dt = 0.01
    upload_sim_params(build_sim_params(
        smoothing_length=0.04, particle_mass=PARTICLE_MASS, particle_spacing=DEFAULT_SPACING,
        gravity=(0.0, 0.0, 0.0), dt=dt, restitution=0.3, wall_friction=0.5,
        world_min=(-1.0, -1.0, -1.0), world_max=(1.0, 1.0, 1.0)))
    upload_materials(build_material_array())

    w = World(10_000)
    n = w.spawn_sphere((0.0, 0.0, 0.0), 0.2, FIRE, 200)
    sl = slice(0, n)
    args = dict(
        sorted_packed_info=w.packed_info[sl], sorted_temperature=w.temperature[sl],
        sorted_health=w.health[sl], sorted_lifetime=w.lifetime[sl],
        sorted_velocity=w.velocity[sl], sorted_exposure_heat=w.sorted_exposure_heat[sl],
        sorted_exposure_corrode=w.sorted_exposure_corrode[sl], sorted_mass=w.mass[sl],
    )

    def mats():
        return w.packed_info[sl].get() & 0xFF

    for _ in range(152):  # 1.52 s
        compute_reactions(**args, frame=0)
    assert np.all(mats() == SMOKE), f"FIRE did not turn to SMOKE: {np.unique(mats())}"
    for _ in range(300):  # 3.0 s more, 4.52 s total
        compute_reactions(**args, frame=0)
    assert np.all(mats() == 0), f"SMOKE did not die: {np.unique(mats())}"


def test_clear_world_zeroes_warm_start():
    """presets._clear_world wipes kappa, kappa_v and the PBF lambda scratch."""
    from presets import _clear_world
    w = _dirty_world(5_000)
    w.spawn_sphere((0.0, 0.0, 0.0), 0.2, WATER, 100)
    w.kappa[:] = 1.0
    _clear_world(w)
    assert w._high_water == 0
    for name in ("kappa", "kappa_v", "sorted_lambda_pbf"):
        assert not bool(cp.any(getattr(w, name) != 0.0)), f"{name} not zeroed"


def test_run_spawner_respects_budget():
    """The periodic spawner stops once _high_water passes the budget."""
    from presets import run_spawner
    w = World(20_000)
    spawner = {"material_id": WATER, "min_corner": (-0.1, 0.0, -0.1),
               "max_corner": (0.1, 0.1, 0.1), "spacing": 0.02,
               "interval_frames": 1, "budget": 3_000}
    batch = run_spawner(w, spawner)
    assert batch > 0
    for _ in range(50):
        run_spawner(w, spawner)
    assert w._high_water <= spawner["budget"] + batch
    before = w._high_water
    assert run_spawner(w, spawner) == 0
    assert w._high_water == before


@pytest.mark.parametrize("loader", ["load_fire_and_ice", "load_water_mill"])
def test_source_presets_set_spawner_budget(loader):
    """Fire and Ice and Water Mill give their spawner a budget below max_particles."""
    from unittest.mock import MagicMock
    import presets
    w = World(500_000)
    w.sdf_manager = MagicMock()  # Water Mill adds a kinematic SDF cylinder
    _, spawner = getattr(presets, loader)(w)
    assert spawner is not None
    assert 0 < spawner["budget"] < w.max_particles



# ---------------------------------------------------------------------------
# bd-r5-review-n28.152: per-particle buffers that no kernel reads are gone
# ---------------------------------------------------------------------------

_REMOVED_BUFFERS = ("veleval", "sph_force", "density", "shear_rate",
                    "exposure_heat", "exposure_corrode", "sorted_hashes",
                    "sorted_color", "delta_position", "alpha_dfsph")

# Bytes per particle of every max_particles-sized World array after
# World(n), allocate_pbf_arrays() and allocate_dfsph_arrays(). Unsorted
# state 65 B + sorted scratch 183 B + lazy solver arrays 52 B
# (predicted_position and its sorted twin, sorted_delta_position,
# sorted_pressure_normal, sorted_alpha_dfsph). Before n28.152 the total
# was 368 B: the removed buffers were 68 B.
_BYTES_PER_PARTICLE_MAX = 65 + 183 + 52


def _per_particle_bytes(w):
    n = w.max_particles
    total = 0
    for v in vars(w).values():
        if isinstance(v, cp.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            total += v.nbytes
    return total / n


def test_removed_buffers_are_gone():
    w = World(1000)
    w.allocate_pbf_arrays()
    w.allocate_dfsph_arrays()
    for name in _REMOVED_BUFFERS:
        assert not hasattr(w, name), f"World still has {name}"
    # The unsorted predicted_position stays: PBF hashes the sort by x*.
    assert w.predicted_position.shape == (1000, 4)


def test_world_memory_per_particle():
    """Device memory of World(n) matches the arrays it holds, and the
    per-particle footprint is at most the post-n28.152 sum."""
    n = 200_000
    pool = cp.get_default_memory_pool()
    cp.cuda.Device().synchronize()
    before = pool.used_bytes()
    w = World(n)
    w.allocate_pbf_arrays()
    w.allocate_dfsph_arrays()
    cp.cuda.Device().synchronize()
    used = pool.used_bytes() - before

    per_particle = _per_particle_bytes(w)
    assert per_particle <= _BYTES_PER_PARTICLE_MAX, per_particle
    attr_bytes = sum(v.nbytes for v in vars(w).values() if isinstance(v, cp.ndarray))
    # The pool rounds each allocation up to 512 B; nothing else is hidden.
    n_arrays = sum(1 for v in vars(w).values() if isinstance(v, cp.ndarray))
    assert attr_bytes <= used <= attr_bytes + 512 * n_arrays, (used, attr_bytes)
    print(f"PASS: {per_particle:.0f} B/particle, {used / 1e6:.1f} MB at {n}")

if __name__ == "__main__":
    test_constructor()
    test_array_shapes_dtypes()
    test_spawn_cube_water()
    test_spawn_cube_max_particles()
    test_spawn_sphere()
    test_spawn_fire_temp()
    test_spawn_steam_temp()
    test_kill_in_sphere()
    test_num_active()
    test_resize()
    test_contiguous_arrays()
    test_ice_spawn_temp()
    test_num_active_no_gpu_sync()
    test_kill_in_sphere_no_gpu_sync()
    test_spawn_sphere_no_loop_sync()
    print("\nALL TESTS PASSED")
