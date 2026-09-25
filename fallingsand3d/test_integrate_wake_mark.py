"""K_Integrate writes the own-cell wake stamp (bd-r5-review-n28.166).

WCSPH no longer launches K_MarkWakeCells: K_Integrate stamps the own grid
cell of every just-woke or fast awake particle from registers, and
run_wake_propagation runs with mark=False. These tests check that the
stamps K_Integrate writes match what K_MarkWakeCells writes for the same
final state, that the sleeper probe then wakes the right particles, and
that Simulation uploads c_grid to the integrate module (it did not before,
so every particle hashed to cell 0).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest

cupy = pytest.importorskip("cupy")
import numpy as np  # noqa: E402

from grid_params import NUM_CELLS, build_grid_params  # noqa: E402
import integrate as integrate_mod  # noqa: E402
import wake  # noqa: E402
from materials import GAS, GRANULAR, SAND, STATIC, STEAM, STONE  # noqa: E402
from test_integrate import (  # noqa: E402
    MAKE_PACKED,
    SET_SLEEPING,
    make_simple_particles,
    setup_params,
    unsorted_copy,
)

SLEEPING_BIT = 0x400
JUST_WOKE_BIT = 0x1000


def _upload_grids() -> None:
    gp = build_grid_params()
    integrate_mod.upload_grid_params(gp)
    wake.upload_grid_params(gp)


def _particles(rows):
    """rows: list of (x, y, z, vy, packed_info). Reversed sort order, so the
    kernel's orig_idx scatter is exercised."""
    n = len(rows)
    pos = np.zeros((n, 4), dtype=np.float32)
    vel = np.zeros((n, 4), dtype=np.float32)
    for k, (x, y, z, vy, _) in enumerate(rows):
        pos[k] = (x, y, z, 1000.0)
        vel[k, 1] = vy
    d = make_simple_particles(n, SAND, GRANULAR, pos=pos, vel=vel)
    d["sorted_packed_info"] = cupy.asarray(
        np.array([r[4] for r in rows], dtype=np.uint32))
    d["sort_indexes"] = cupy.asarray(np.arange(n - 1, -1, -1).astype(np.uint32))
    d["sorted_sleep_counter"] = cupy.asarray(
        np.array([200 if (r[4] & SLEEPING_BIT) else 0 for r in rows],
                 dtype=np.uint8))
    d["health_out"] = unsorted_copy(d["sorted_health"], d["sort_indexes"])
    return d


def _run_integrate(d, flags, frame):
    return integrate_mod.integrate(**d, cell_wake_flags=flags,
                                   frame_counter=frame)


# One particle per category, each in its own far-apart cell.
ROWS = [
    # x,    y,    z,   vy,   packed_info
    (-0.4, 0.0, 0.0, -1.0, MAKE_PACKED(SAND, GRANULAR)),                 # fast awake: marks
    (-0.2, 0.0, 0.0, 0.0, SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR))),    # still asleep: no mark
    (0.0, 0.0, 0.0, -1.0, MAKE_PACKED(STEAM, GAS)),                      # fast GAS: no mark
    (0.2, 0.0, 0.0, -1.0, MAKE_PACKED(STONE, STATIC)),                   # STATIC: no mark
    (0.4, 0.0, 0.0, 0.0, MAKE_PACKED(SAND, GRANULAR)),                   # slow awake: no mark
    (0.0, 0.3, 0.0, -0.5, SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR))),    # wakes now: marks
]


def test_integrate_stamps_match_mark_wake_cells():
    """The table K_Integrate writes equals the one K_MarkWakeCells writes
    from K_Integrate's unsorted outputs."""
    setup_params()
    _upload_grids()
    frame = cupy.array([41], dtype=cupy.uint32)
    d = _particles(ROWS)

    fused = cupy.zeros(NUM_CELLS, dtype=cupy.uint32)
    pos, vel, pi, _sc, _t = _run_integrate(d, fused, frame)

    ref = cupy.zeros(NUM_CELLS, dtype=cupy.uint32)
    wake.mark_wake_cells(pos, vel, pi, ref, len(ROWS), frame_counter=frame)

    fused_h = fused.get()
    ref_h = ref.get()
    assert np.array_equal(fused_h, ref_h)
    # Fast awake sand and the particle that woke this substep; nothing else.
    assert int(np.count_nonzero(fused_h)) == 2
    assert set(np.unique(fused_h[fused_h != 0]).tolist()) == {42}
    # The waking particle carries JUST_WOKE into the unsorted packed_info.
    pi_h = pi.get()
    woke = int(pi_h[d["sort_indexes"].get()[5]])
    assert woke & JUST_WOKE_BIT and not (woke & SLEEPING_BIT)


def test_null_table_writes_no_stamp():
    """Without cell_wake_flags K_Integrate marks nothing (the reference
    K_MarkWakeCells path)."""
    setup_params()
    _upload_grids()
    d = _particles(ROWS)
    integrate_mod.integrate(**d)  # no flags: must not fault
    cupy.cuda.Device().synchronize()


def test_integrate_mark_then_probe_wakes_neighbour():
    """WCSPH path: stamps from K_Integrate, then run_wake_propagation with
    mark=False. A sleeper one cell from a fast particle wakes; one three
    cells away stays asleep."""
    setup_params()
    _upload_grids()
    frame = cupy.array([7], dtype=cupy.uint32)
    sleeping = SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR))
    rows = [
        (0.0, 0.0, 0.0, -1.0, MAKE_PACKED(SAND, GRANULAR)),   # fast mover
        (0.04, 0.0, 0.0, 0.0, sleeping),                      # one cell away
        (0.14, 0.0, 0.0, 0.0, sleeping),                      # three cells away
    ]
    d = _particles(rows)
    # Stale stamps from other substeps must not count.
    flags = cupy.full(NUM_CELLS, 3, dtype=cupy.uint32)
    pos, vel, pi, sc, _t = _run_integrate(d, flags, frame)
    wake.run_wake_propagation(pos, vel, pi, sc, flags, num_particles=3,
                              frame_counter=frame, mark=False)
    order = d["sort_indexes"].get()
    pi_h = pi.get()
    sc_h = sc.get()
    near, far = int(order[1]), int(order[2])
    assert not (int(pi_h[near]) & SLEEPING_BIT), "neighbour of a mover must wake"
    assert int(sc_h[near]) == 0
    assert int(pi_h[far]) & SLEEPING_BIT, "sleeper three cells away must stay asleep"
    assert not (int(pi_h[order[0]]) & JUST_WOKE_BIT)


def test_simulation_uploads_integrate_grid():
    """Simulation gives the integrate module the same c_grid as the sort."""
    from simulation import Simulation
    from world import World
    import counting_sort

    world = World(4_000)
    world.spawn_cube((-0.1, -0.5, -0.1), (0.1, -0.4, 0.1), SAND, 0.02)
    Simulation(world, dt=0.001, fixed_dt=True, world_half_size=0.5)

    def read(mod):
        buf = np.zeros(8, dtype=np.uint32)
        cupy.cuda.runtime.memcpy(buf.ctypes.data, int(mod.get_global("c_grid")),
                                 buf.nbytes, 2)
        return buf

    a = read(integrate_mod.get_module())
    b = read(counting_sort._get_module())
    assert a[7] != 0, "integrate c_grid table_mask is zero"
    assert np.array_equal(a, b)


def test_wcsph_sleeping_block_wakes_under_mover():
    """End to end under WCSPH: a fast sand grain dropped onto a sleeping
    block wakes the block within a few substeps."""
    from simulation import Simulation
    from world import World

    world = World(20_000)
    world.spawn_cube((-0.1, -0.5, -0.1), (0.1, -0.4, 0.1), SAND, 0.02)
    n_block = world._high_water
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=0.5)
    sim._spawn_substep = sim._damping_duration
    world.packed_info[:n_block] |= cupy.uint32(SLEEPING_BIT)
    world.sleep_counter[:n_block] = 200
    # One grain just above the block, moving down fast.
    world.spawn_cube((-0.01, -0.37, -0.01), (0.01, -0.35, 0.01), SAND, 0.02)
    n = world._high_water
    assert n > n_block
    world.velocity[n_block:n, 1] = -2.0

    for _ in range(10):
        sim._sim_step(world._high_water)
    cupy.cuda.Device().synchronize()
    pi = world.packed_info[:n_block].get()
    awake = int(np.count_nonzero((pi & SLEEPING_BIT) == 0))
    assert awake > 0, "no block particle woke under a fast grain"
