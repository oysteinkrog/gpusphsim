"""Direct unit test of counting_sort.counting_sort_full (bd-r5-review-n28.6).

The app builds its neighbour grid only with counting_sort_full
(Simulation._run_grid_setup). These checks pin the live cell-table layout:

  - sorted hashes are non-decreasing and sort_perm is a permutation
  - every sorted array equals the unsorted array gathered through sort_perm
  - the FP16 velocity mirror decodes to the sorted velocity
  - cell_end[c] - cell_start[c] == histogram[c] for occupied cells
  - cell_end[c] == 0 for empty cells (the live empty-cell marker; the retired
    build_grid path used a 0xFFFFFFFF cell_start sentinel instead)
  - two grid cells that collide in the 2^18 hash table share one bucket
  - a 27-cell grid walk finds every neighbour within h that an O(N^2)
    brute force finds, on ~2K particles
  - K_Step1 density on the sorted scene matches a brute-force poly6 sum

Requires cupy and an NVIDIA GPU.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np

from conftest import counting_sort_scene, default_grid_params

H = 0.04
MASK = (1 << 18) - 1
GRID_MIN = -1.0
PACKED_WATER = 5  # material 5 (WATER), behaviour FLUID (0)


def _cell(p: np.ndarray) -> np.ndarray:
    return np.floor((p - GRID_MIN) * np.float32(1.0 / H)).astype(np.int64)


def _hash(c: np.ndarray) -> np.ndarray:
    """CPU copy of spatialHash() in common.cuh (uint32 wrap, then mask)."""
    c = c.astype(np.int64)
    h = (c[..., 0] * 73856093 + c[..., 1] * 19349669
         + c[..., 2] * 83492791) & 0xFFFFFFFF
    return (h & MASK).astype(np.int64)


def _find_colliding_cells() -> tuple:
    """Return two far-apart cells in the default world [0, 50)^3 with the
    same table hash. The additive hash is linear, so two cells collide only
    when their offset d has d.P = 0 mod 2^18; the shortest such offset is
    (18, 47, 45), so the search needs the whole world."""
    r = np.arange(0, 50)
    cx, cy, cz = np.meshgrid(r, r, r, indexing="ij")
    cells = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1)
    hashes = _hash(cells)
    order = np.argsort(hashes, kind="stable")
    hs = hashes[order]
    for k in np.nonzero(hs[1:] == hs[:-1])[0]:
        a = cells[order[k]]
        b = cells[order[k + 1]]
        if np.abs(a - b).max() > 3:
            return a, b
    raise AssertionError("no colliding cell pair found in the test range")


def _make_scene_inputs(seed: int = 7) -> tuple:
    rng = np.random.default_rng(seed)
    n_rand = 2000
    pos = np.zeros((n_rand, 4), dtype=np.float32)
    pos[:, :3] = rng.uniform(-0.2, 0.2, size=(n_rand, 3))

    cell_a, cell_b = _find_colliding_cells()
    clusters = []
    for cell, count in ((cell_a, 40), (cell_b, 25)):
        lo = GRID_MIN + cell * H
        c = np.zeros((count, 4), dtype=np.float32)
        c[:, :3] = lo + rng.uniform(0.05 * H, 0.95 * H, size=(count, 3))
        clusters.append(c)
    pos = np.concatenate([pos] + clusters, axis=0)
    pos[:, 3] = 1.0
    n = pos.shape[0]
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, :3] = rng.uniform(-2.0, 2.0, size=(n, 3))
    return pos, vel, cell_a, cell_b, (40, 25)


def test_counting_sort_full_cell_tables() -> None:
    pos, vel, cell_a, cell_b, (na, nb) = _make_scene_inputs()
    n = pos.shape[0]
    mass = np.linspace(0.005, 0.01, n).astype(np.float32)
    scene = counting_sort_scene(
        pos, velocity=vel, mass=mass, grid_params=default_grid_params(),
    )
    cupy.cuda.Device().synchronize()

    num_cells = scene["num_cells"]
    assert num_cells == MASK + 1
    perm = cupy.asnumpy(scene["sort_perm"]).astype(np.int64)
    sorted_hashes = cupy.asnumpy(scene["sorted_hashes"]).astype(np.int64)
    hashes = cupy.asnumpy(scene["hashes"]).astype(np.int64)
    hist = cupy.asnumpy(scene["histogram"]).astype(np.int64)
    cs = cupy.asnumpy(scene["cell_start"]).astype(np.int64)
    ce = cupy.asnumpy(scene["cell_end"]).astype(np.int64)

    # Device hash equals the CPU spatialHash copy.
    np.testing.assert_array_equal(hashes, _hash(_cell(pos[:, :3])))

    # sort_perm is a permutation; sorted hashes non-decreasing.
    assert np.array_equal(np.sort(perm), np.arange(n)), "sort_perm is not a permutation"
    assert np.all(np.diff(sorted_hashes) >= 0), "sorted hashes decrease"
    np.testing.assert_array_equal(sorted_hashes, hashes[perm])

    # Sorted arrays are the unsorted arrays gathered through sort_perm.
    np.testing.assert_array_equal(cupy.asnumpy(scene["position"]), pos[perm])
    np.testing.assert_array_equal(cupy.asnumpy(scene["velocity"]), vel[perm])
    np.testing.assert_array_equal(cupy.asnumpy(scene["mass"]), mass[perm])

    # FP16 velocity mirror (read by step1/step2 neighbour loops).
    vh = cupy.asnumpy(scene["velocity_h"]).view(np.float16).reshape(n, 4)
    np.testing.assert_allclose(vh.astype(np.float32), vel[perm], rtol=1e-3, atol=1e-3)

    # Histogram and cell tables.
    assert hist.sum() == n
    np.testing.assert_array_equal(hist, np.bincount(hashes, minlength=num_cells))
    occupied = hist > 0
    np.testing.assert_array_equal(ce[occupied] - cs[occupied], hist[occupied])
    assert np.all(ce[~occupied] == 0), "empty cells must have cell_end == 0"
    excl = np.concatenate([[0], np.cumsum(hist)[:-1]])
    np.testing.assert_array_equal(cs[occupied], excl[occupied])
    for c in np.nonzero(occupied)[0][:50]:
        assert np.all(sorted_hashes[cs[c]:ce[c]] == c)

    # Two cells that collide in the 2^18 table share one bucket.
    h_a = int(_hash(cell_a[None])[0])
    assert h_a == int(_hash(cell_b[None])[0])
    in_a = np.all(_cell(pos[:, :3]) == cell_a, axis=1).sum()
    in_b = np.all(_cell(pos[:, :3]) == cell_b, axis=1).sum()
    assert in_a >= na and in_b >= nb
    assert hist[h_a] == in_a + in_b, (
        f"collision bucket holds {hist[h_a]}, expected {in_a}+{in_b}"
    )

    # Grid walk vs O(N^2) brute force: no neighbour within h is missed.
    sp = pos[perm, :3].astype(np.float64)
    d2 = ((sp[:, None, :] - sp[None, :, :]) ** 2).sum(axis=2)
    brute = d2 < H * H
    cells_sorted = _cell(pos[perm, :3])
    offs = np.array([(dx, dy, dz) for dx in (-1, 0, 1)
                     for dy in (-1, 0, 1) for dz in (-1, 0, 1)])
    missed = 0
    for i in range(n):
        found = np.zeros(n, dtype=bool)
        for hsh in np.unique(_hash(cells_sorted[i] + offs)):
            start, end = cs[hsh], ce[hsh]
            if end > start:
                found[start:end] = True
        missed += int(np.count_nonzero(brute[i] & ~found))
    assert missed == 0, f"grid walk missed {missed} neighbour pairs within h"


def test_step1_density_on_counting_sort_matches_brute_force() -> None:
    import step1
    from materials import build_interaction_matrix, build_material_array

    pos, vel, _a, _b, _counts = _make_scene_inputs(seed=11)
    n = pos.shape[0]
    mass = np.full(n, 0.008, dtype=np.float32)

    step1.upload_sim_params(step1.build_sim_params(
        smoothing_length=H, particle_mass=0.008, particle_spacing=0.02,
    ))
    step1.upload_precalc_params(step1.build_precalc_params(smoothing_length=H))
    step1.upload_materials(build_material_array())
    step1.upload_interactions(build_interaction_matrix())

    scene = counting_sort_scene(
        pos, velocity=vel, mass=mass,
        packed_info=np.full(n, PACKED_WATER, dtype=np.uint32),
        grid_params=default_grid_params(),
    )
    density, *_rest = step1.compute_step1(
        scene["position"], scene["velocity"], scene["mass"], None,
        scene["packed_info"], scene["temperature"],
        scene["cell_start"], scene["cell_end"],
        velocity_h=scene["velocity_h"],
    )
    cupy.cuda.Device().synchronize()
    rho = cupy.asnumpy(density).astype(np.float64)

    perm = cupy.asnumpy(scene["sort_perm"]).astype(np.int64)
    sp = pos[perm, :3].astype(np.float64)
    d2 = ((sp[:, None, :] - sp[None, :, :]) ** 2).sum(axis=2)
    w = np.where(d2 < H * H, (H * H - d2) ** 3, 0.0)
    poly6 = 315.0 / (64.0 * math.pi * H ** 9)
    rho_ref = np.maximum(1.0, poly6 * (w * mass[perm][None, :]).sum(axis=1))

    np.testing.assert_allclose(rho, rho_ref, rtol=2e-3)
