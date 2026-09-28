"""bd-r5-review-n28.216: PBF reads the previous density in the current slot order.

K_PBF_ComputeLambda reads each neighbour's previous-substep density from
density_in (sorted_density_prev) for the heat, exposure and other m_j / rho_j
weights. That snapshot must be in this substep's slot order. It used to be a
copy of sorted_density, which is still in the previous substep's slot order,
so after a re-sort every moved slot read another particle's density. The
Morton hash moves more slots per re-sort than the old prime hash, so results
depended on the hash layout.

The hash is compiled into the kernels, so this test checks the slot order
directly instead of comparing two hash builds: for every substep after the
first, density_in[i] must equal the density the previous substep computed for
the particle now in slot i. Every sort is a full sort, with
deterministic_sort on, so the permutation changes between substeps. The scene
has a STATIC floor, a falling water block and a sand pile, so the STATIC,
FLUID, GRANULAR and (when some sand falls asleep) sleeping writebacks in
K_PBF_Finalize are all covered.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np
import pytest

SUBSTEPS = 120
SLEEP_BIT = 1 << 10


def _scene(world) -> None:
    from materials import SAND, STONE, WATER
    from presets import _clear_world

    _clear_world(world)
    world.spawn_cube(
        min_corner=(-0.6, -0.9, -0.6),
        max_corner=(0.6, -0.84, 0.6),
        material_id=STONE,
        spacing=0.02,
    )
    world.spawn_cube(
        min_corner=(-0.5, -0.8, -0.3),
        max_corner=(-0.1, -0.5, 0.3),
        material_id=WATER,
        spacing=0.02,
        velocity=(1.0, 0.0, 0.0),
    )
    world.spawn_cube(
        min_corner=(0.1, -0.83, -0.2),
        max_corner=(0.4, -0.7, 0.2),
        material_id=SAND,
        spacing=0.02,
    )


@pytest.mark.gpu
def test_pbf_density_in_follows_the_resort() -> None:
    import pbf_solver
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(max_particles=200_000)
    sim = Simulation(
        world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=True, max_substeps=20
    )
    captured: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    real_lambda = pbf_solver.pbf_compute_lambda

    def spy(*args, **kwargs):
        dens_in = kwargs.get("density_in")
        if dens_in is not None:
            # First lambda call of the substep: sorted_density still holds
            # the previous substep's result, in the previous slot order.
            m = dens_in.shape[0]
            captured.append(
                (
                    cupy.asnumpy(dens_in).copy(),
                    cupy.asnumpy(world.sorted_density[:m]).copy(),
                    cupy.asnumpy(sim._sort_perm[:m]).copy(),
                    cupy.asnumpy(world.sorted_packed_info[:m]).copy(),
                )
            )
        return real_lambda(*args, **kwargs)

    pbf_solver.pbf_compute_lambda = spy
    # Run every substep directly (no CUDA graph), so the spy sees each call.
    sim._graph_slot = lambda is_gather: (None, -1)
    try:
        sim.deterministic_sort = True
        sim.set_solver_profile(PROFILES["PBF"])
        _scene(world)
        sim.notify_scene_changed()
        n = world._high_water
        for _ in range(SUBSTEPS):
            sim._sort_skip_next = False  # full sort: the permutation changes
            sim._sim_step(n)
        cupy.cuda.Device().synchronize()
    finally:
        pbf_solver.pbf_compute_lambda = real_lambda
        sim.close()

    assert len(captured) == SUBSTEPS
    moved_total = 0
    checked = {"STATIC": 0, "FLUID": 0, "GRANULAR": 0, "sleeping": 0}
    worst = (0, None)
    for k in range(1, SUBSTEPS):
        dens_in, prev_dens, perm, pi = captured[k]
        prev_perm = captured[k - 1][2]
        # prev_dens was sampled at the start of substep k: the density
        # substep k-1 computed, in substep k-1's slot order (prev_perm).
        unsorted = np.empty_like(prev_dens)
        unsorted[prev_perm] = prev_dens
        expected = unsorted[perm]
        moved_total += int(np.count_nonzero(perm != prev_perm))
        bad = np.flatnonzero(expected.view(np.uint32) != dens_in.view(np.uint32))
        if bad.size > worst[0]:
            worst = (bad.size, k)
        beh = (pi >> 8) & 0x3
        checked["STATIC"] += int(np.count_nonzero(beh == 3))
        checked["FLUID"] += int(np.count_nonzero(beh == 0))
        checked["GRANULAR"] += int(np.count_nonzero(beh == 1))
        checked["sleeping"] += int(np.count_nonzero(pi & SLEEP_BIT))
    print(
        f"\n  PBF density_in check: n={n}, slots moved by re-sorts={moved_total}, "
        f"slots checked by kind={checked}"
    )
    assert moved_total > 0, "no re-sort moved a slot; the check proves nothing"
    assert checked["STATIC"] > 0 and checked["FLUID"] > 0 and checked["GRANULAR"] > 0
    assert worst[0] == 0, (
        f"substep {worst[1]}: {worst[0]}/{n} slots read another particle's "
        f"previous density"
    )
