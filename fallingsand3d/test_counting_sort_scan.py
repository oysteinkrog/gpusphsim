"""Owned counting-sort scan and allocation-free full sort (bd-r5-review-n28.189).

  - inclusive_scan matches cupy.cumsum on random histograms of 2^18, 2^20
    and a size that is not a multiple of the scan tile
  - a full sort puts every particle exactly once in [cell_start, cell_end)
    of its own hash, empty cells have start == end, and the histogram is
    back at zero afterwards, so a second sort on the same scratch gives the
    same tables
  - counting_sort_full makes no memory-pool allocation, also inside a CUDA
    graph capture, and Simulation._capture_graph leaves the pool's used
    bytes unchanged

Requires cupy and an NVIDIA GPU.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np
import pytest
from cupy.cuda import memory_hook

import counting_sort
from conftest import counting_sort_scene, default_grid_params


class _MallocCounter(memory_hook.MemoryHook):
    name = "n28_189_malloc_counter"

    def __init__(self) -> None:
        self.sizes: list = []

    def malloc_preprocess(self, **kw) -> None:
        self.sizes.append(kw.get("size"))

    def alloc_preprocess(self, **kw) -> None:
        self.sizes.append(kw.get("mem_size"))


@pytest.mark.parametrize("num_cells", [1 << 18, 1 << 20, (1 << 18) + 37, 5])
def test_inclusive_scan_matches_cumsum(num_cells: int) -> None:
    rng = np.random.default_rng(num_cells)
    # Mostly empty with some dense cells, like a particle table.
    hist = np.where(rng.random(num_cells) < 0.1,
                    rng.integers(1, 60, num_cells), 0).astype(np.uint32)
    d_hist = cupy.asarray(hist)
    block_sums = counting_sort.allocate_block_sums(num_cells)
    # Unaligned output, as for cell_end = offsets[1:].
    offsets = cupy.full(num_cells + 1, 0xDEADBEEF, dtype=cupy.uint32)
    offsets[0] = 0
    counting_sort.inclusive_scan(num_cells, d_hist, block_sums, offsets[1:])
    got = cupy.asnumpy(offsets)
    ref = cupy.asnumpy(cupy.cumsum(d_hist, dtype=cupy.uint32))
    assert got[0] == 0
    np.testing.assert_array_equal(got[1:], ref)


def test_full_sort_round_trip_and_histogram_drain() -> None:
    rng = np.random.default_rng(189)
    n = 150_000
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, :3] = rng.uniform(-0.6, 0.6, size=(n, 3))
    # Dense clump so some cells hold many particles.
    pos[:5000, :3] = rng.uniform(0.0, 0.03, size=(5000, 3))
    scene = counting_sort_scene(pos, grid_params=default_grid_params(),
                                upload_kernels=False)
    scratch = scene["scratch"]
    num_cells = scene["num_cells"]

    cs = cupy.asnumpy(scene["cell_start"]).astype(np.int64)
    ce = cupy.asnumpy(scene["cell_end"]).astype(np.int64)
    hashes = cupy.asnumpy(scene["hashes"]).astype(np.int64)
    perm = cupy.asnumpy(scene["sort_perm"]).astype(np.int64)

    # Scratch histogram is drained back to zero by K_ScatterPerm.
    assert int(cupy.count_nonzero(scratch["histogram"])) == 0

    counts = np.bincount(hashes, minlength=num_cells)
    np.testing.assert_array_equal(ce - cs, counts)
    np.testing.assert_array_equal(ce, np.cumsum(counts))
    assert np.all(cs[counts == 0] == ce[counts == 0])

    # Every particle exactly once, inside its own hash's range.
    assert np.array_equal(np.sort(perm), np.arange(n))
    slot_of = np.empty(n, dtype=np.int64)
    slot_of[perm] = np.arange(n)
    h = hashes
    assert np.all(slot_of >= cs[h]) and np.all(slot_of < ce[h])
    np.testing.assert_array_equal(cupy.asnumpy(scene["position"]), pos[perm])

    # Second sort on the same scratch: same tables, histogram zero again.
    scene2 = counting_sort_scene(pos, grid_params=default_grid_params(),
                                 upload_kernels=False, scratch=scratch)
    np.testing.assert_array_equal(cupy.asnumpy(scene2["cell_start"]), cs)
    np.testing.assert_array_equal(cupy.asnumpy(scene2["cell_end"]), ce)
    assert int(cupy.count_nonzero(scratch["histogram"])) == 0


def _sort_args(n: int, num_cells: int) -> dict:
    rng = np.random.default_rng(5)
    pos = cupy.asarray(np.concatenate(
        [rng.uniform(-0.5, 0.5, (n, 3)), np.ones((n, 1))], axis=1).astype(np.float32))
    scratch = counting_sort.allocate_scratch(num_cells, n)
    f4 = lambda: cupy.zeros((n, 4), dtype=cupy.float32)  # noqa: E731
    f1 = lambda: cupy.zeros(n, dtype=cupy.float32)  # noqa: E731
    return dict(
        num_particles=n, num_cells=num_cells,
        histogram=scratch["histogram"], block_sums=scratch["block_sums"],
        cell_start=scratch["cell_start"], cell_end=scratch["cell_end"],
        sort_perm=cupy.zeros(n, dtype=cupy.uint32),
        positions=pos, hashes=cupy.zeros(n, dtype=cupy.uint32),
        position=pos, velocity=f4(), mass=f1(),
        packed_info=cupy.full(n, 5, dtype=cupy.uint32), temperature=f1(),
        health=f1(), lifetime=f1(), sleep_counter=cupy.zeros(n, dtype=cupy.uint8),
        kappa=f1(), kappa_v=f1(),
        sorted_position=f4(), sorted_velocity=f4(), sorted_mass=f1(),
        sorted_packed_info=cupy.zeros(n, dtype=cupy.uint32),
        sorted_temperature=f1(), sorted_health=f1(), sorted_lifetime=f1(),
        sorted_sleep_counter=cupy.zeros(n, dtype=cupy.uint8),
        sorted_kappa=f1(), sorted_kappa_v=f1(),
        sorted_velocity_h=cupy.zeros((n, 2), dtype=cupy.uint32),
        sorted_temperature_h=cupy.zeros(n, dtype=cupy.uint16),
        sorted_anchor=f4(),
    )


def test_full_sort_makes_no_pool_allocation_in_capture() -> None:
    counting_sort.upload_grid_params(default_grid_params())
    args = _sort_args(20_000, 1 << 18)
    counting_sort.counting_sort_full(**args)  # compile and warm up
    cupy.cuda.Device().synchronize()

    direct = _MallocCounter()
    with direct:
        counting_sort.counting_sort_full(**args)
    assert direct.sizes == [], f"direct sort allocated {direct.sizes}"

    s = cupy.cuda.Stream(non_blocking=True)
    captured = _MallocCounter()
    with s:
        s.begin_capture()
        with captured:
            counting_sort.counting_sort_full(**args)
        graph = s.end_capture()
    assert captured.sizes == [], f"captured sort allocated {captured.sizes}"

    graph.launch(stream=s)
    s.synchronize()
    assert int(cupy.count_nonzero(args["histogram"])) == 0
    ce = cupy.asnumpy(args["cell_end"])
    assert ce[-1] == 20_000


def test_capture_graph_keeps_pool_used_bytes() -> None:
    from materials import WATER
    from simulation import Simulation
    from world import World

    world = World(max_particles=20_000)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=1)
    try:
        sim.fixed_dt_value = 0.001
        sim._upload_dt(0.001)
        sim._spawn_substep = sim._damping_duration
        world.spawn_cube(min_corner=(-0.2, -0.1, -0.1),
                         max_corner=(0.2, 0.1, 0.1),
                         material_id=WATER, spacing=0.02)
        sim.notify_scene_changed()
        n = world._high_water
        sim._sort_skip_next = False
        sim._sim_step(n)  # direct run: compile and allocate lazily-made buffers
        cupy.cuda.Device().synchronize()

        pool = cupy.get_default_memory_pool()
        used_before = pool.used_bytes()
        graph = sim._capture_graph(n, gather_only=False)
        cupy.cuda.Device().synchronize()
        assert graph is not None
        assert pool.used_bytes() == used_before
    finally:
        sim._sort_skip_next = False
        sim.close()
