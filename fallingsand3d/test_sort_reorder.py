"""Tests for the counting sort and data reorder (counting_sort.counting_sort_full).

The app builds its neighbour grid with counting_sort_full. These tests check
that it orders particles by grid hash, that every sorted array is the
unsorted array gathered through sort_perm, and that it copes with 0 and
500K particles.

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import sys
import os

# Ensure fallingsand3d directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np

from conftest import counting_sort_scene
from grid_params import NUM_CELLS
from world import World


def _random_positions(n: int, half: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, :3] = rng.uniform(-half, half, size=(n, 3)).astype(np.float32)
    return pos


def _check_order(scene: dict, n: int) -> np.ndarray:
    """Assert sorted hashes are non-decreasing and sort_perm is a permutation.

    Returns sort_perm as int64 numpy.
    """
    perm = cupy.asnumpy(scene["sort_perm"]).astype(np.int64)
    hashes = cupy.asnumpy(scene["hashes"]).astype(np.int64)
    sorted_hashes = hashes[perm]
    assert np.all(np.diff(sorted_hashes) >= 0), (
        f"{int(np.sum(np.diff(sorted_hashes) < 0))} hash inversions"
    )
    assert sorted_hashes.min() >= 0 and sorted_hashes.max() < NUM_CELLS
    assert np.array_equal(np.sort(perm), np.arange(n)), "sort_perm is not a permutation"
    return perm


def test_sort_orders_by_hash() -> None:
    """After the sort, hashes are non-decreasing and sort_perm is a permutation."""
    n = 10_000
    scene = counting_sort_scene(_random_positions(n, 0.9, 42), upload_kernels=False)
    cupy.cuda.Device().synchronize()
    _check_order(scene, n)


def test_reorder_gathers_every_array() -> None:
    """Each sorted array equals the unsorted array gathered through sort_perm."""
    n = 5_000
    rng = np.random.default_rng(99)
    pos = _random_positions(n, 0.8, 99)
    vel = rng.uniform(-1, 1, size=(n, 4)).astype(np.float32)
    mass = rng.uniform(0.001, 0.1, size=n).astype(np.float32)
    pinfo = rng.integers(1, 16, size=n).astype(np.uint32)
    temp = rng.uniform(250.0, 1500.0, size=n).astype(np.float32)
    health = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    life = rng.uniform(0.0, 5.0, size=n).astype(np.float32)
    sleep = rng.integers(0, 255, size=n).astype(np.uint8)

    scene = counting_sort_scene(
        pos, velocity=vel, mass=mass, packed_info=pinfo, temperature=temp,
        health=health, lifetime=life, sleep_counter=sleep, upload_kernels=False,
    )
    cupy.cuda.Device().synchronize()
    perm = _check_order(scene, n)

    for name, src in (
        ("position", pos), ("velocity", vel), ("mass", mass),
        ("packed_info", pinfo), ("temperature", temp), ("health", health),
        ("lifetime", life), ("sleep_counter", sleep),
    ):
        np.testing.assert_array_equal(cupy.asnumpy(scene[name]), src[perm], err_msg=name)

    assert abs(float(mass.sum()) - float(cupy.asnumpy(scene["mass"]).sum())) < 1e-3


def test_world_sorted_buffers_preallocated() -> None:
    """World class pre-allocates sorted_* temporary buffers at init."""
    print("\n--- World sorted buffer pre-allocation ---")

    world = World(max_particles=1000)

    # Check all sorted buffers exist and have correct shapes
    assert world.sorted_position.shape == (1000, 4)
    assert world.sorted_velocity.shape == (1000, 4)
    assert world.sorted_veleval.shape == (1000, 4)
    assert world.sorted_sph_force.shape == (1000, 4)
    assert world.sorted_density.shape == (1000,)
    assert world.sorted_mass.shape == (1000,)
    assert world.sorted_temperature.shape == (1000,)
    assert world.sorted_health.shape == (1000,)
    assert world.sorted_lifetime.shape == (1000,)
    assert world.sorted_shear_rate.shape == (1000,)
    assert world.sorted_packed_info.shape == (1000,)
    assert world.sorted_sleep_counter.shape == (1000,)
    print("[OK] All sorted_* buffers pre-allocated with shape (1000, ...)")

    # Check dtypes
    assert world.sorted_position.dtype == cupy.float32
    assert world.sorted_mass.dtype == cupy.float32
    assert world.sorted_packed_info.dtype == cupy.uint32
    assert world.sorted_sleep_counter.dtype == cupy.uint8
    print("[OK] sorted_* buffer dtypes correct")

    # Check the hash buffer
    assert world.hashes.shape == (1000,)
    # The counting sort writes no sorted hash copy (bd-r5-review-n28.152).
    assert not hasattr(world, "sorted_hashes")
    assert world.hashes.dtype == cupy.uint32
    print("[OK] Hash buffers pre-allocated")

    # After resize, buffers should be re-allocated
    world.resize(2000)
    assert world.sorted_position.shape == (2000, 4)
    assert world.sorted_mass.shape == (2000,)
    assert world.hashes.shape == (2000,)
    print("[OK] resize() re-allocates sorted buffers to new size")


def test_end_to_end_world_sort_reorder() -> None:
    """Spawn into a World, sort its arrays and check the sorted copies."""
    world = World(max_particles=5000)
    spawned = world.spawn_sphere((0.0, 0.0, 0.0), 0.5, 5, 3000)  # water
    assert spawned > 0
    n = world._high_water

    scene = counting_sort_scene(
        world.position[:n], velocity=world.velocity[:n], mass=world.mass[:n],
        packed_info=world.packed_info[:n], temperature=world.temperature[:n],
        health=world.health[:n], lifetime=world.lifetime[:n],
        sleep_counter=world.sleep_counter[:n], upload_kernels=False,
    )
    cupy.cuda.Device().synchronize()
    perm = _check_order(scene, n)

    np.testing.assert_array_equal(
        cupy.asnumpy(scene["position"]), cupy.asnumpy(world.position[:n])[perm]
    )
    mass_before = float(cupy.asnumpy(world.mass[:n]).sum())
    mass_after = float(cupy.asnumpy(scene["mass"]).sum())
    assert abs(mass_before - mass_after) < 1e-3


def test_500k_sort_reorder_no_errors() -> None:
    """Sort and reorder complete without errors for 500K particles."""
    n = 500_000
    rng = np.random.default_rng(77)
    pos = _random_positions(n, 0.95, 77)
    vel = rng.uniform(-1, 1, (n, 4)).astype(np.float32)

    scene = counting_sort_scene(pos, velocity=vel, upload_kernels=False)
    cupy.cuda.Device().synchronize()
    perm = _check_order(scene, n)

    h_spos = cupy.asnumpy(scene["position"])
    h_svel = cupy.asnumpy(scene["velocity"])
    np.testing.assert_array_equal(h_spos, pos[perm])
    np.testing.assert_array_equal(h_svel, vel[perm])
    assert abs(float(cupy.asnumpy(scene["mass"]).sum()) - n * 0.008) < 0.05


def test_zero_particles() -> None:
    """The sort is a no-op for zero particles."""
    scene = counting_sort_scene(np.zeros((0, 4), dtype=np.float32), upload_kernels=False)
    cupy.cuda.Device().synchronize()
    assert scene["sort_perm"].shape[0] == 0
    assert scene["position"].shape[0] == 0


def test_reorder_wrappers_accept_null_fp16_outputs() -> None:
    """bd-r5-review-n28.155: sorted_velocity_h and sorted_temperature_h
    may be None. The wrappers pass a NULL pointer and
    K_ScatterReorder / K_GatherReorder skip those stores, so neither the
    full sort nor the gather reorder faults, and the float32 outputs are
    still the unsorted arrays gathered through sort_perm."""
    import counting_sort
    import grid_params

    rng = np.random.default_rng(3)
    n = 1500
    pos_h = np.zeros((n, 4), dtype=np.float32)
    pos_h[:, :3] = rng.uniform(-0.3, 0.3, size=(n, 3))
    pos_h[:, 3] = 1.0
    vel_h = rng.uniform(-1.0, 1.0, size=(n, 4)).astype(np.float32)

    grid_params = grid_params.build_grid_params()
    num_cells = int(grid_params[0]["table_size"])
    counting_sort.upload_grid_params(grid_params)
    scratch = counting_sort.allocate_scratch(num_cells, n)

    def unsorted() -> dict:
        return dict(
            position=cupy.asarray(pos_h), velocity=cupy.asarray(vel_h),
            mass=cupy.full(n, 0.02, dtype=cupy.float32),
            packed_info=cupy.full(n, 5, dtype=cupy.uint32),
            temperature=cupy.full(n, 293.0, dtype=cupy.float32),
            health=cupy.ones(n, dtype=cupy.float32),
            lifetime=cupy.zeros(n, dtype=cupy.float32),
            sleep_counter=cupy.zeros(n, dtype=cupy.uint8),
            kappa=cupy.zeros(n, dtype=cupy.float32),
            kappa_v=cupy.zeros(n, dtype=cupy.float32),
        )

    def sorted_out() -> dict:
        return dict(
            sorted_position=cupy.empty((n, 4), dtype=cupy.float32),
            sorted_velocity=cupy.empty((n, 4), dtype=cupy.float32),
            sorted_mass=cupy.empty(n, dtype=cupy.float32),
            sorted_packed_info=cupy.empty(n, dtype=cupy.uint32),
            sorted_temperature=cupy.empty(n, dtype=cupy.float32),
            sorted_health=cupy.empty(n, dtype=cupy.float32),
            sorted_lifetime=cupy.empty(n, dtype=cupy.float32),
            sorted_sleep_counter=cupy.empty(n, dtype=cupy.uint8),
            sorted_kappa=None,
            sorted_kappa_v=None,
            sorted_velocity_h=None,
            sorted_temperature_h=None,
        )

    src = unsorted()
    out = sorted_out()
    sort_perm = cupy.zeros(n, dtype=cupy.uint32)
    counting_sort.counting_sort_full(
        num_particles=n, num_cells=num_cells,
        histogram=scratch["histogram"], write_offset=scratch["write_offset"],
        cell_start=scratch["cell_start"], cell_end=scratch["cell_end"],
        sort_perm=sort_perm, positions=src["position"],
        hashes=cupy.zeros(n, dtype=cupy.uint32),
        **src, **out,
    )
    cupy.cuda.Device().synchronize()
    perm = cupy.asnumpy(sort_perm).astype(np.int64)
    assert np.array_equal(np.sort(perm), np.arange(n))
    np.testing.assert_array_equal(cupy.asnumpy(out["sorted_velocity"]), vel_h[perm])

    # Grid-reuse path: re-gather with the same sort_perm and changed velocity.
    vel_h = vel_h * 2.0
    src = unsorted()
    out2 = sorted_out()
    counting_sort.gather_reorder(
        num_particles=n, sort_perm=sort_perm, **src, **out2,
    )
    cupy.cuda.Device().synchronize()
    np.testing.assert_array_equal(cupy.asnumpy(out2["sorted_velocity"]), vel_h[perm])
    np.testing.assert_array_equal(cupy.asnumpy(out2["sorted_position"]), pos_h[perm])
    print("[OK] reorder wrappers accept None FP16 outputs")


def test_legacy_sort_modules_removed() -> None:
    """bd-r5-review-n28.153: the argsort pipeline (K_CalcHash, argsort,
    K_FusedReorder, K_BuildDataStruct, K_FusedSortReorderBuild) is gone and no
    tracked module imports it. counting_sort is the only sort path."""
    import ast
    import pathlib

    base = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    legacy = ("hash_sort", "build_grid", "fused_reorder", "fused_sort_reorder_build")
    for name in legacy:
        assert not (base / f"{name}.py").exists(), f"{name}.py still exists"
        assert not (base / "physics" / "kernels" / f"{name}.cu").exists(), (
            f"{name}.cu still exists"
        )

    # Only tracked modules count: agents leave untracked scratch scripts here.
    import subprocess
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--", "*.py"], cwd=base,
            capture_output=True, text=True, check=True,
        ).stdout.split()
        sources = [base / f for f in tracked if "/" not in f]
    except (OSError, subprocess.CalledProcessError):
        sources = list(base.glob("*.py"))

    found = []
    for src_path in sources:
        if not src_path.exists():
            continue
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            found += [f"{src_path.name}: {n}" for n in names if n in legacy]
    assert not found, f"legacy sort modules imported: {found}"

    # K_ScatterReorder takes no cell_end_out: K_BuildCellEnd writes cell_end.
    cu = (base / "physics" / "kernels" / "counting_sort.cu").read_text(encoding="utf-8")
    scatter = cu[cu.index("void K_ScatterReorder("):cu.index(") {", cu.index("void K_ScatterReorder("))]
    assert "cell_end" not in scatter


def main() -> None:
    test_sort_orders_by_hash()
    test_reorder_gathers_every_array()
    test_world_sorted_buffers_preallocated()
    test_end_to_end_world_sort_reorder()
    test_500k_sort_reorder_no_errors()
    test_zero_particles()
    test_reorder_wrappers_accept_null_fp16_outputs()
    test_legacy_sort_modules_removed()
    print("\n=== ALL SORT/REORDER CHECKS PASSED ===")


if __name__ == "__main__":
    main()
