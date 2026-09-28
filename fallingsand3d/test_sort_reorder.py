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
import pytest

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
    K_GatherReorder skips those stores, so neither the
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
            # No sorted kappa output, so no input either (bd-r5-review-n28.165).
            kappa=None,
            kappa_v=None,
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
        histogram=scratch["histogram"], block_sums=scratch["block_sums"],
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

    # A sorted output without its input is rejected on the host
    # (the kernel would read a NULL pointer).
    import pytest
    bad = sorted_out()
    bad["sorted_kappa"] = cupy.empty(n, dtype=cupy.float32)
    with pytest.raises(ValueError):
        counting_sort.gather_reorder(
            num_particles=n, sort_perm=sort_perm, **unsorted(), **bad,
        )


def test_reorder_skips_each_null_output_alone() -> None:
    """bd-r5-review-n28.165: each optional output (kappa, kappa_v,
    velocity_h, temperature_h) can be None on its own. The None one is not
    written, and every array still given is the unsorted
    array gathered through sort_perm, on the full sort and the gather path."""
    import counting_sort
    import grid_params

    rng = np.random.default_rng(11)
    n = 2000
    pos_h = np.zeros((n, 4), dtype=np.float32)
    pos_h[:, :3] = rng.uniform(-0.3, 0.3, size=(n, 3))
    vel_h = rng.uniform(-1.0, 1.0, size=(n, 4)).astype(np.float32)
    kappa_h = rng.uniform(0.0, 5.0, size=n).astype(np.float32)
    kappa_v_h = rng.uniform(0.0, 5.0, size=n).astype(np.float32)
    temp_h = rng.uniform(250.0, 900.0, size=n).astype(np.float32)

    gp = grid_params.build_grid_params()
    num_cells = int(gp[0]["table_size"])
    counting_sort.upload_grid_params(gp)
    scratch = counting_sort.allocate_scratch(num_cells, n)

    optional = ("kappa", "kappa_v", "velocity_h", "temperature_h")
    for skip in optional:
        src = dict(
            position=cupy.asarray(pos_h), velocity=cupy.asarray(vel_h),
            mass=cupy.full(n, 0.02, dtype=cupy.float32),
            packed_info=cupy.full(n, 5, dtype=cupy.uint32),
            temperature=cupy.asarray(temp_h),
            health=cupy.ones(n, dtype=cupy.float32),
            lifetime=cupy.zeros(n, dtype=cupy.float32),
            sleep_counter=cupy.zeros(n, dtype=cupy.uint8),
            kappa=None if skip == "kappa" else cupy.asarray(kappa_h),
            kappa_v=None if skip == "kappa_v" else cupy.asarray(kappa_v_h),
        )
        opt_out = {
            "kappa": cupy.full(n, -1.0, dtype=cupy.float32),
            "kappa_v": cupy.full(n, -1.0, dtype=cupy.float32),
            "velocity_h": cupy.full((n, 2), 0xDEADBEEF, dtype=cupy.uint32),
            "temperature_h": cupy.full(n, 0xBEEF, dtype=cupy.uint16),
        }
        out = dict(
            sorted_position=cupy.empty((n, 4), dtype=cupy.float32),
            sorted_velocity=cupy.empty((n, 4), dtype=cupy.float32),
            sorted_mass=cupy.empty(n, dtype=cupy.float32),
            sorted_packed_info=cupy.empty(n, dtype=cupy.uint32),
            sorted_temperature=cupy.empty(n, dtype=cupy.float32),
            sorted_health=cupy.empty(n, dtype=cupy.float32),
            sorted_lifetime=cupy.empty(n, dtype=cupy.float32),
            sorted_sleep_counter=cupy.empty(n, dtype=cupy.uint8),
        )
        for name in optional:
            out[f"sorted_{name}"] = None if name == skip else opt_out[name]

        sort_perm = cupy.zeros(n, dtype=cupy.uint32)
        counting_sort.counting_sort_full(
            num_particles=n, num_cells=num_cells,
            histogram=scratch["histogram"], block_sums=scratch["block_sums"],
            cell_start=scratch["cell_start"], cell_end=scratch["cell_end"],
            sort_perm=sort_perm, positions=src["position"],
            hashes=cupy.zeros(n, dtype=cupy.uint32), **src, **out,
        )
        cupy.cuda.Device().synchronize()
        perm = cupy.asnumpy(sort_perm).astype(np.int64)

        def check(tag: str) -> None:
            np.testing.assert_array_equal(
                cupy.asnumpy(out["sorted_velocity"]), vel_h[perm], err_msg=tag)
            if skip != "kappa":
                np.testing.assert_array_equal(
                    cupy.asnumpy(opt_out["kappa"]), kappa_h[perm], err_msg=tag)
            if skip != "kappa_v":
                np.testing.assert_array_equal(
                    cupy.asnumpy(opt_out["kappa_v"]), kappa_v_h[perm], err_msg=tag)
            if skip != "velocity_h":
                assert not (cupy.asnumpy(opt_out["velocity_h"]) == 0xDEADBEEF).any(), tag
            if skip != "temperature_h":
                assert not (cupy.asnumpy(opt_out["temperature_h"]) == 0xBEEF).any(), tag
            # The skipped buffer is untouched.
            untouched = cupy.asnumpy(opt_out[skip])
            first = untouched.reshape(-1)[0]
            assert (untouched == first).all(), f"{tag}: {skip} was written"

        check(f"full sort, {skip}=None")

        # Gather path with new values: kept outputs follow, skipped stays put.
        vel_h = vel_h * 2.0
        kappa_h = kappa_h + 1.0
        kappa_v_h = kappa_v_h + 1.0
        src["velocity"] = cupy.asarray(vel_h)
        if src["kappa"] is not None:
            src["kappa"] = cupy.asarray(kappa_h)
        if src["kappa_v"] is not None:
            src["kappa_v"] = cupy.asarray(kappa_v_h)
        counting_sort.gather_reorder(
            num_particles=n, sort_perm=sort_perm, **src, **out,
        )
        cupy.cuda.Device().synchronize()
        check(f"gather, {skip}=None")


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

    # No sort kernel takes cell_start: the scan writes cell_end only, and
    # cell_start is a view of the same offsets table (bd-r5-review-n28.160).
    cu = (base / "physics" / "kernels" / "counting_sort.cu").read_text(encoding="utf-8")
    for name in ("K_HashHistogram", "K_ScanReduce", "K_ScanDownsweep",
                 "K_ScatterPerm", "K_GatherReorder"):
        head = cu.index(f"void {name}(")
        assert "cell_start" not in cu[head:cu.index(") {", head)], name


# ---------------------------------------------------------------------------
# Deterministic within-cell order (bd-r5-review-n28.200)
# ---------------------------------------------------------------------------

def _sort_perm_only(pos: "cupy.ndarray", scratch: dict, perm_scratch) -> np.ndarray:
    """Run counting_sort_full on pos and return (sort_perm, hashes) as numpy."""
    import counting_sort
    import grid_params as grid_params_mod

    gp = grid_params_mod.build_grid_params()
    counting_sort.upload_grid_params(gp)
    num_cells = int(gp[0]["table_size"])
    n = int(pos.shape[0])
    f4 = lambda: cupy.zeros((n, 4), dtype=cupy.float32)
    f1 = lambda: cupy.zeros(n, dtype=cupy.float32)
    perm = cupy.zeros(n, dtype=cupy.uint32)
    hashes = cupy.zeros(n, dtype=cupy.uint32)
    counting_sort.counting_sort_full(
        num_particles=n, num_cells=num_cells,
        histogram=scratch["histogram"], block_sums=scratch["block_sums"],
        cell_start=scratch["cell_start"], cell_end=scratch["cell_end"],
        sort_perm=perm, positions=pos, hashes=hashes,
        position=pos, velocity=f4(), mass=f1(),
        packed_info=cupy.zeros(n, dtype=cupy.uint32), temperature=f1(),
        health=f1(), lifetime=f1(), sleep_counter=cupy.zeros(n, dtype=cupy.uint8),
        kappa=None, kappa_v=None,
        sorted_position=f4(), sorted_velocity=f4(), sorted_mass=f1(),
        sorted_packed_info=cupy.zeros(n, dtype=cupy.uint32),
        sorted_temperature=f1(), sorted_health=f1(), sorted_lifetime=f1(),
        sorted_sleep_counter=cupy.zeros(n, dtype=cupy.uint8),
        sorted_kappa=None, sorted_kappa_v=None,
        perm_scratch=perm_scratch,
    )
    return cupy.asnumpy(perm).astype(np.int64), cupy.asnumpy(hashes).astype(np.int64)


def test_deterministic_sort_perm_is_stable() -> None:
    """With perm_scratch, sort_perm is the stable sort of the hashes.

    Dense input (about 30 particles per cell) so the atomic order inside a
    cell would differ between runs. Two sorts of the same input must give the
    same sort_perm, equal to a stable argsort of the hashes.
    """
    import counting_sort
    import grid_params as grid_params_mod

    n = 60_000
    pos = cupy.asarray(_random_positions(n, 0.25, 7))
    num_cells = int(grid_params_mod.build_grid_params()[0]["table_size"])
    scratch = counting_sort.allocate_scratch(num_cells, n)
    perm_scratch = cupy.zeros(n, dtype=cupy.uint32)

    perm_a, hashes = _sort_perm_only(pos, scratch, perm_scratch)
    perm_b, _ = _sort_perm_only(pos, scratch, perm_scratch)
    assert np.array_equal(np.sort(perm_a), np.arange(n))
    assert np.array_equal(perm_a, perm_b), "deterministic sort_perm changed between runs"
    assert np.array_equal(perm_a, np.argsort(hashes, kind="stable")), (
        "deterministic sort_perm is not the stable sort of the hashes"
    )

    # The default mode still gives a valid bucket order.
    perm_c, _ = _sort_perm_only(pos, scratch, None)
    assert np.array_equal(np.sort(perm_c), np.arange(n))
    assert np.all(np.diff(hashes[perm_c]) >= 0)


def test_deterministic_sort_rejects_aliased_scratch() -> None:
    """perm_scratch may not be sort_perm itself (K_RankInCell reads and writes)."""
    import counting_sort
    import grid_params as grid_params_mod

    n = 8
    num_cells = int(grid_params_mod.build_grid_params()[0]["table_size"])
    scratch = counting_sort.allocate_scratch(num_cells, n)
    pos = cupy.zeros((n, 4), dtype=cupy.float32)
    perm = cupy.zeros(n, dtype=cupy.uint32)
    z4 = cupy.zeros((n, 4), dtype=cupy.float32)
    z1 = cupy.zeros(n, dtype=cupy.float32)
    zu = cupy.zeros(n, dtype=cupy.uint32)
    zb = cupy.zeros(n, dtype=cupy.uint8)
    try:
        counting_sort.counting_sort_full(
            num_particles=n, num_cells=num_cells,
            histogram=scratch["histogram"], block_sums=scratch["block_sums"],
            cell_start=scratch["cell_start"], cell_end=scratch["cell_end"],
            sort_perm=perm, positions=pos, hashes=zu.copy(),
            position=pos, velocity=z4, mass=z1, packed_info=zu, temperature=z1,
            health=z1, lifetime=z1, sleep_counter=zb, kappa=None, kappa_v=None,
            sorted_position=z4.copy(), sorted_velocity=z4.copy(),
            sorted_mass=z1.copy(), sorted_packed_info=zu.copy(),
            sorted_temperature=z1.copy(), sorted_health=z1.copy(),
            sorted_lifetime=z1.copy(), sorted_sleep_counter=zb.copy(),
            sorted_kappa=None, sorted_kappa_v=None,
            perm_scratch=perm,
        )
    except ValueError:
        return
    raise AssertionError("aliased perm_scratch was accepted")


def _dam_break_positions(deterministic: bool, substeps: int) -> np.ndarray:
    from presets import PRESETS
    from simulation import Simulation

    world = World(max_particles=400_000)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=20)
    try:
        sim.deterministic_sort = deterministic
        PRESETS["Dam Break"](world)
        sim.notify_scene_changed()
        n = world._high_water
        for _ in range(substeps):
            sim._sim_step(n)
        cupy.cuda.Device().synchronize()
        return cupy.asnumpy(world.position[:n]).copy()
    finally:
        sim.close()


@pytest.mark.slow
def test_deterministic_sort_dam_break_bit_identical() -> None:
    """Two 200-substep Dam Break runs with deterministic_sort on end with
    bit-identical positions (unsorted order, which is the spawn order).

    Needs the K_Step1 density pack outside the kernel (bd-r5-review-n28.215):
    an in-place position.w write let neighbours read either density."""
    a = _dam_break_positions(True, 200)
    b = _dam_break_positions(True, 200)
    assert a.shape == b.shape
    diff = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    assert diff == 0, f"{diff} position words differ between runs"


def main() -> None:
    test_sort_orders_by_hash()
    test_reorder_gathers_every_array()
    test_world_sorted_buffers_preallocated()
    test_end_to_end_world_sort_reorder()
    test_500k_sort_reorder_no_errors()
    test_zero_particles()
    test_reorder_wrappers_accept_null_fp16_outputs()
    test_reorder_skips_each_null_output_alone()
    test_legacy_sort_modules_removed()
    print("\n=== ALL SORT/REORDER CHECKS PASSED ===")


if __name__ == "__main__":
    main()
