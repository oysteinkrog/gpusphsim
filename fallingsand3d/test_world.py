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
from world import World, DEFAULT_SPACING, T_AMBIENT, _MAKE_PACKED
from materials import (
    MATERIALS, WATER, SAND, LAVA, FIRE, STEAM, SMOKE, ICE, STONE,
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
    # mass = rho0 * spacing^3 = 1000 * 8e-6 = 0.008
    mass_val = float(w.mass[0])
    assert abs(mass_val - 0.008) < 1e-6, f"mass={mass_val}"
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
    print("\nALL TESTS PASSED")
