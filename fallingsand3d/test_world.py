"""Tests for world.py -- World particle manager."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cupy as cp
import cupy.cuda.compiler as _compiler
import numpy as np

# Blackwell (sm_120) PTX workaround
_compiler._use_ptx = True
for _fn in (_compiler._get_arch, _compiler._get_arch_for_options_for_nvrtc):
    if hasattr(_fn, '_cache'):
        _fn._cache = {}
from world import World, DEFAULT_SPACING, T_AMBIENT, _MAKE_PACKED, PARTICLE_MASS, _DEFAULT_TEMPS
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


def test_array_shapes_dtypes():
    w = World(1000)
    n = 1000
    # float4 arrays
    for name in ["position", "velocity", "veleval", "sph_force", "color"]:
        arr = getattr(w, name)
        assert arr.shape == (n, 4), f"{name} shape: {arr.shape}"
        assert arr.dtype == cp.float32, f"{name} dtype: {arr.dtype}"
    # float arrays
    for name in ["density", "mass", "temperature", "health", "lifetime",
                 "shear_rate", "exposure_heat", "exposure_corrode"]:
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
    for name in ["position", "velocity", "veleval", "sph_force", "density",
                 "mass", "packed_info", "temperature", "health", "lifetime",
                 "shear_rate", "exposure_heat", "exposure_corrode", "color",
                 "sleep_counter"]:
        assert isinstance(getattr(w, name), cp.ndarray), f"{name} not CuPy"
    print("PASS: array shapes, dtypes, and GPU residency")


def test_spawn_cube_water():
    w = World(500_000)
    n = w.spawn_cube((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), WATER, spacing=0.02)
    assert n > 0
    assert w.num_active == n
    # mass = PARTICLE_MASS = 0.02 (game-tuned constant, not rho0*dx^3)
    mass_val = float(w.mass[0])
    assert abs(mass_val - PARTICLE_MASS) < 1e-6, f"mass={mass_val}, expected PARTICLE_MASS={PARTICLE_MASS}"
    # packed_info = MAKE_PACKED(WATER, FLUID)
    pi_val = int(w.packed_info[0])
    expected = _MAKE_PACKED(WATER, FLUID)
    assert pi_val == expected, f"packed_info={pi_val} expected={expected}"
    # temperature = 293K (ambient)
    assert abs(float(w.temperature[0]) - T_AMBIENT) < 0.1
    # health = 1.0
    assert abs(float(w.health[0]) - 1.0) < 1e-6
    # color = water color
    mat = MATERIALS[WATER]
    assert abs(float(w.color[0, 0]) - mat.color_r) < 1e-6
    assert abs(float(w.color[0, 1]) - mat.color_g) < 1e-6
    assert abs(float(w.color[0, 2]) - mat.color_b) < 1e-6
    assert abs(float(w.color[0, 3]) - 1.0) < 1e-6
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
    killed = w.kill_in_sphere((0.0, 0.0, 0.0), 0.25)
    assert killed > 0
    assert w.num_active < before
    # Killed particles have packed_info=0
    dead_mask = w.packed_info[:before] == 0
    assert int(cp.sum(dead_mask)) >= killed
    print(f"PASS: kill_in_sphere ({killed} killed, {w.num_active} remain)")


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


def test_contiguous_arrays():
    """All arrays are contiguous CuPy arrays on GPU."""
    w = World(1000)
    for name in ["position", "velocity", "veleval", "sph_force", "density",
                 "mass", "packed_info", "temperature", "health", "lifetime",
                 "shear_rate", "exposure_heat", "exposure_corrode", "color",
                 "sleep_counter"]:
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


def test_num_active_no_gpu_sync():
    """bd-mzc.38 regression: num_active property must not call int() on a CuPy array."""
    import unittest.mock as mock
    w = World(10_000)
    w.spawn_sphere((0.0, 0.0, 0.0), 0.1, WATER, 500)

    # Patch cp.count_nonzero to detect if num_active calls it
    original_cnz = cp.count_nonzero
    calls = []
    def mock_cnz(*a, **kw):
        calls.append(1)
        return original_cnz(*a, **kw)

    with mock.patch("world.cp.count_nonzero", side_effect=mock_cnz):
        # If num_active still uses count_nonzero the call list will be non-empty
        # After bd-mzc.38 the property should NOT call count_nonzero
        _ = w.num_active

    assert len(calls) == 0, (
        "num_active called cp.count_nonzero (GPU sync detected); bd-mzc.38 not fixed"
    )
    print("PASS: test_num_active_no_gpu_sync")


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
            w.kill_in_sphere((0.0, 0.0, 0.0), 0.25)
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
