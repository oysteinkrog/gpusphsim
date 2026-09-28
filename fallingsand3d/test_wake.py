"""Integration test for wake.py -- cell-flag wake propagation kernels.

Acceptance criteria:
  - cell_wake_flags stamp table (num_cells uint32), zero at allocation and
    never cleared per substep when a frame counter is passed
  - Phase 1 kernel (K_MarkWakeCells): each particle with HAS_JUST_WOKE (or an
    awake mover above V_WAKE) writes the stamp into its own cell
  - Phase 2 kernel (K_WakeSleepers): a sleeping particle wakes if any of its
    27 cells holds the stamp: CLEAR_SLEEPING in packed_info, reset
    sleep_counter to 0
  - JUST_WOKE flag cleared after Phase 1 runs
  - Test: drop a fast-moving particle onto sleeping sand pile -- sand near
    impact wakes up
  - Test: particles far from impact stay sleeping
  - Both kernels run without errors for 500K particles

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np

from grid_params import (
    NUM_CELLS,
    build_grid_params,
)
from wake import (
    BLOCK_SIZE,
    allocate_cell_wake_flags,
    get_module,
    mark_wake_cells,
    mark_wake_cells_rigid,
    run_wake_propagation,
    upload_grid_params as upload_wake_grid_params,
    wake_sleepers_and_clear_just_woke,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# packed_info helpers (mirrors common.cuh macros)
def MAKE_PACKED(mat_id: int, behavior: int) -> int:
    return (mat_id & 0xFF) | ((behavior & 0x3) << 8)

def SET_SLEEPING(p: int) -> int:
    return p | 0x400

def IS_SLEEPING(p: int) -> int:
    return (p >> 10) & 1

def SET_JUST_WOKE(p: int) -> int:
    return p | 0x1000

def HAS_JUST_WOKE(p: int) -> int:
    return (p >> 12) & 1

def CLEAR_JUST_WOKE(p: int) -> int:
    return p & ~0x1000

def CLEAR_SLEEPING(p: int) -> int:
    return p & ~0x400

FLUID = 0
GRANULAR = 1
GAS = 2
SAND = 2
WATER = 5
SMOKE = 13

# ---------------------------------------------------------------------------
# Helper: upload grid params for wake module
# ---------------------------------------------------------------------------

_params_uploaded = False

def setup_params():
    """Upload GridParams to wake module's constant memory."""
    global _params_uploaded
    if not _params_uploaded:
        grid_params = build_grid_params()
        upload_wake_grid_params(grid_params)
        _params_uploaded = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compilation() -> None:
    """Verify CuPy RawModule compiles wake.cu."""
    print("--- Compilation check ---")
    module = get_module()
    assert module is not None
    # Phase 1 kernel and fused Phase 2+3 kernel (API refactored from 3 separate kernels)
    k1 = module.get_function("K_MarkWakeCells")
    k2 = module.get_function("K_WakeSleepersAndClearJustWoke")
    assert k1 is not None
    assert k2 is not None
    print("  Compilation OK: K_MarkWakeCells, K_WakeSleepersAndClearJustWoke found")


def test_block_size() -> None:
    """Verify block size = 256."""
    print("--- Block size check ---")
    assert BLOCK_SIZE == 256
    print(f"  BLOCK_SIZE = {BLOCK_SIZE}")


def test_cell_wake_flags_allocation() -> None:
    """Verify cell_wake_flags array allocation and clearing."""
    print("--- Cell wake flags allocation ---")
    flags = allocate_cell_wake_flags(NUM_CELLS)
    assert flags.shape == (NUM_CELLS,)
    assert flags.dtype == cupy.uint32
    assert int(cupy.sum(flags)) == 0
    print(f"  Allocated {flags.shape[0]} uint32 flags, all zero")


def test_mark_wake_cells_single_particle() -> None:
    """A single just-woke particle marks only its own cell."""
    print("--- Mark wake cells: single particle ---")
    setup_params()

    # Place one particle at the center of the grid (0, 0, 0)
    # Grid: min=(-1,-1,-1), max=(1,1,1), res=50, delta=25
    # Cell for (0,0,0): int((0 - (-1)) * 25) = int(25) = 25 per axis
    # Hash = 25*50*50 + 25*50 + 25 = 62500 + 1250 + 25 = 63775
    pos = cupy.zeros((1, 4), dtype=cupy.float32)
    pos[0] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)
    vel = cupy.zeros((1, 4), dtype=cupy.float32)  # zero velocity

    packed = MAKE_PACKED(SAND, GRANULAR)
    packed = SET_JUST_WOKE(packed)
    pi = cupy.array([np.uint32(packed)], dtype=cupy.uint32)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    mark_wake_cells(pos, vel, pi, flags, num_particles=1)

    # Own cell only (bd-r5-review-n28.141); sleepers probe their 27 cells.
    flagged_count = int(cupy.count_nonzero(flags))
    print(f"  Flagged cells: {flagged_count}")
    assert flagged_count == 1, f"Expected 1 flagged cell, got {flagged_count}"

    # The center cell (25,25,25), hashed like spatialHash() in common.cuh
    from grid_params import spatial_hash_np
    center_hash = int(spatial_hash_np((25, 25, 25), NUM_CELLS - 1))
    assert int(flags[center_hash]) != 0
    print(f"  Center cell {center_hash} flagged correctly")


def test_mark_wake_cells_corner_particle() -> None:
    """A just-woke particle at the grid corner also marks one cell."""
    print("--- Mark wake cells: corner particle ---")
    setup_params()

    # Place particle at grid min corner (-1, -1, -1) -> cell (0, 0, 0)
    pos = cupy.zeros((1, 4), dtype=cupy.float32)
    pos[0] = cupy.array([-0.99, -0.99, -0.99, 1.0], dtype=cupy.float32)
    vel = cupy.zeros((1, 4), dtype=cupy.float32)  # zero velocity

    packed = MAKE_PACKED(SAND, GRANULAR)
    packed = SET_JUST_WOKE(packed)
    pi = cupy.array([np.uint32(packed)], dtype=cupy.uint32)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    mark_wake_cells(pos, vel, pi, flags, num_particles=1)

    # The spatial hash has no bounds, so the corner cell is an ordinary entry.
    flagged_count = int(cupy.count_nonzero(flags))
    print(f"  Flagged cells (corner): {flagged_count}")
    assert flagged_count == 1, f"Expected 1 flagged cell at corner, got {flagged_count}"


def test_mark_wake_cells_no_just_woke() -> None:
    """Particles without JUST_WOKE flag should not mark any cells when velocity is zero."""
    print("--- Mark wake cells: no just-woke ---")
    setup_params()

    pos = cupy.zeros((10, 4), dtype=cupy.float32)
    pos[:, 3] = 1.0
    vel = cupy.zeros((10, 4), dtype=cupy.float32)  # zero velocity (below V_WAKE threshold)

    # Normal particles (not sleeping, not just_woke)
    packed = MAKE_PACKED(SAND, GRANULAR)
    pi = cupy.full(10, np.uint32(packed), dtype=cupy.uint32)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    mark_wake_cells(pos, vel, pi, flags, num_particles=10)

    flagged_count = int(cupy.count_nonzero(flags))
    assert flagged_count == 0, f"Expected 0 flagged cells, got {flagged_count}"
    print("  No cells flagged (correct)")


def test_wake_sleepers_in_flagged_cell() -> None:
    """Sleeping particle in a flagged cell should wake up."""
    print("--- Wake sleepers: flagged cell ---")
    setup_params()

    # Place a sleeping particle at origin
    pos = cupy.zeros((1, 4), dtype=cupy.float32)
    pos[0] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)

    packed = MAKE_PACKED(SAND, GRANULAR)
    packed = SET_SLEEPING(packed)
    pi = cupy.array([np.uint32(packed)], dtype=cupy.uint32)
    sc = cupy.array([50], dtype=cupy.uint8)

    # Flag the cell at origin: cell (25,25,25), hashed like spatialHash()
    from grid_params import spatial_hash_np
    flags = allocate_cell_wake_flags(NUM_CELLS)
    center_hash = int(spatial_hash_np((25, 25, 25), NUM_CELLS - 1))
    flags[center_hash] = cupy.uint32(1)

    wake_sleepers_and_clear_just_woke(pos, pi, sc, flags, num_particles=1)

    pi_host = int(pi[0])
    sc_host = int(sc[0])
    print(f"  packed_info after wake: 0x{pi_host:08X}, sleep_counter: {sc_host}")
    assert IS_SLEEPING(pi_host) == 0, "Particle should no longer be sleeping"
    assert sc_host == 0, "Sleep counter should be reset to 0"
    print("  Sleeping particle woke up correctly")


def test_wake_sleepers_unflagged_cell() -> None:
    """Sleeping particle in an unflagged cell should stay sleeping."""
    print("--- Wake sleepers: unflagged cell ---")
    setup_params()

    pos = cupy.zeros((1, 4), dtype=cupy.float32)
    pos[0] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)

    packed = MAKE_PACKED(SAND, GRANULAR)
    packed = SET_SLEEPING(packed)
    pi = cupy.array([np.uint32(packed)], dtype=cupy.uint32)
    sc = cupy.array([50], dtype=cupy.uint8)

    # All flags are zero
    flags = allocate_cell_wake_flags(NUM_CELLS)

    wake_sleepers_and_clear_just_woke(pos, pi, sc, flags, num_particles=1)

    pi_host = int(pi[0])
    sc_host = int(sc[0])
    assert IS_SLEEPING(pi_host) == 1, "Particle should still be sleeping"
    assert sc_host == 50, "Sleep counter should be unchanged"
    print("  Sleeping particle stayed sleeping (correct)")


def test_clear_just_woke() -> None:
    """K_WakeSleepersAndClearJustWoke clears JUST_WOKE from all particles."""
    print("--- Clear JUST_WOKE flag ---")
    setup_params()

    packed_base = MAKE_PACKED(SAND, GRANULAR)
    packed_woke = SET_JUST_WOKE(packed_base)
    packed_sleeping = SET_SLEEPING(packed_base)

    pos = cupy.zeros((3, 4), dtype=cupy.float32)
    pos[:, 3] = 1.0
    pi = cupy.array([
        np.uint32(packed_woke),     # has JUST_WOKE
        np.uint32(packed_base),     # normal
        np.uint32(packed_sleeping), # sleeping, no JUST_WOKE
    ], dtype=cupy.uint32)
    sc = cupy.zeros(3, dtype=cupy.uint8)

    # Use all-zero flags so no sleepers are woken (only JUST_WOKE clearing is tested)
    flags = allocate_cell_wake_flags(NUM_CELLS)
    wake_sleepers_and_clear_just_woke(pos, pi, sc, flags, num_particles=3)

    pi_host = pi.get()
    assert HAS_JUST_WOKE(int(pi_host[0])) == 0, "JUST_WOKE should be cleared"
    assert int(pi_host[1]) == packed_base, "Normal particle unchanged"
    assert int(pi_host[2]) == packed_sleeping, "Sleeping particle unchanged"
    print("  JUST_WOKE flags cleared correctly")


def test_full_pipeline_impact_wake() -> None:
    """Drop a fast-moving particle onto sleeping sand: nearby sand wakes up."""
    print("--- Full pipeline: impact wake ---")
    setup_params()

    # Create 100 sleeping sand particles in a flat bed at y=0, spread in x/z
    n_sleeping = 100
    n_total = n_sleeping + 1  # +1 for the impacting particle

    pos = cupy.zeros((n_total, 4), dtype=cupy.float32)
    vel = cupy.zeros((n_total, 4), dtype=cupy.float32)  # zero velocity
    pi = cupy.zeros(n_total, dtype=cupy.uint32)
    sc = cupy.zeros(n_total, dtype=cupy.uint8)

    # Sleeping sand bed: spread across x in [-0.1, 0.1], y=0, z=0
    for idx in range(n_sleeping):
        x = -0.1 + 0.2 * idx / max(n_sleeping - 1, 1)
        pos[idx] = cupy.array([x, 0.0, 0.0, 1.0], dtype=cupy.float32)
        packed = SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR))
        pi[idx] = cupy.uint32(packed)
        sc[idx] = cupy.uint8(50)

    # Impacting particle: just woke at (0, 0, 0) — simulates a fast-moving
    # particle that just triggered the wake condition in Integrate
    packed_woke = SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR))
    pos[n_sleeping] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[n_sleeping] = cupy.uint32(packed_woke)
    sc[n_sleeping] = cupy.uint8(0)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n_total)

    pi_host = pi.get()
    sc_host = sc.get()

    # Count how many sleeping particles woke up
    woke_count = 0
    still_sleeping = 0
    for idx in range(n_sleeping):
        if IS_SLEEPING(int(pi_host[idx])) == 0:
            woke_count += 1
            assert int(sc_host[idx]) == 0, f"Woke particle {idx} should have counter=0"
        else:
            still_sleeping += 1

    print(f"  Woke up: {woke_count}, still sleeping: {still_sleeping}")
    assert woke_count > 0, "At least some nearby particles should wake up"

    # The impacting particle should have JUST_WOKE cleared
    assert HAS_JUST_WOKE(int(pi_host[n_sleeping])) == 0, \
        "Impacting particle JUST_WOKE should be cleared"
    print("  Impacting particle JUST_WOKE cleared")


def test_far_particles_stay_sleeping() -> None:
    """Particles far from impact should stay sleeping."""
    print("--- Far particles stay sleeping ---")
    setup_params()

    # One just-woke particle at origin, one sleeping particle far away at (0.9, 0.9, 0.9)
    # Grid cell size = 0.04, so 3x3x3 reach is ~0.12 in each direction
    # 0.9 is ~22 cells away from origin (0.9 / 0.04 = 22.5 cells)
    n = 2
    pos = cupy.zeros((n, 4), dtype=cupy.float32)
    vel = cupy.zeros((n, 4), dtype=cupy.float32)  # zero velocity
    pi = cupy.zeros(n, dtype=cupy.uint32)
    sc = cupy.zeros(n, dtype=cupy.uint8)

    # Just-woke particle at origin
    pos[0] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[0] = cupy.uint32(SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR)))
    sc[0] = cupy.uint8(0)

    # Sleeping particle far away
    pos[1] = cupy.array([0.9, 0.9, 0.9, 1.0], dtype=cupy.float32)
    pi[1] = cupy.uint32(SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)))
    sc[1] = cupy.uint8(100)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n)

    pi_host = pi.get()
    sc_host = sc.get()

    assert IS_SLEEPING(int(pi_host[1])) == 1, "Far particle should still be sleeping"
    assert int(sc_host[1]) == 100, "Far particle sleep counter should be unchanged"
    print("  Far particle stayed sleeping (correct)")


def test_stress_500k() -> None:
    """Both kernels run without errors for 500K particles."""
    print("--- 500K stress test ---")
    setup_params()

    n = 500_000
    pos = cupy.random.uniform(-0.9, 0.9, (n, 4)).astype(cupy.float32)
    pos[:, 3] = 1.0
    vel = cupy.zeros((n, 4), dtype=cupy.float32)  # zero velocity

    # 90% sleeping, 5% just_woke, 5% normal
    packed_sleeping = SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR))
    packed_woke = SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR))
    packed_normal = MAKE_PACKED(SAND, GRANULAR)

    pi = cupy.full(n, np.uint32(packed_sleeping), dtype=cupy.uint32)
    pi[:25_000] = cupy.uint32(packed_woke)
    pi[25_000:50_000] = cupy.uint32(packed_normal)

    sc = cupy.full(n, 50, dtype=cupy.uint8)
    sc[:50_000] = cupy.uint8(0)

    flags = allocate_cell_wake_flags(NUM_CELLS)

    # Run the full pipeline
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n)

    # Basic sanity: no JUST_WOKE flags remain
    pi_host = pi.get()
    just_woke_remaining = np.sum((pi_host >> 12) & 1)
    assert just_woke_remaining == 0, f"{just_woke_remaining} particles still have JUST_WOKE"

    # Some sleeping particles near woke particles should have woken up
    sleeping_now = np.sum((pi_host >> 10) & 1)
    print(f"  500K test: sleeping remaining = {sleeping_now} (of original {n - 50_000})")
    assert sleeping_now < (n - 50_000), "At least some sleeping particles should have woken"

    # No CUDA errors -- synchronize to catch async errors
    cupy.cuda.Device().synchronize()
    print("  500K stress test passed")


def test_multiple_woke_particles() -> None:
    """Multiple just-woke particles at different locations produce correct flags."""
    print("--- Multiple woke particles ---")
    setup_params()

    # Two just-woke particles at different locations, one sleeping between them
    n = 3
    pos = cupy.zeros((n, 4), dtype=cupy.float32)
    vel = cupy.zeros((n, 4), dtype=cupy.float32)  # zero velocity
    pi = cupy.zeros(n, dtype=cupy.uint32)
    sc = cupy.zeros(n, dtype=cupy.uint8)

    # Woke particle at (-0.5, 0, 0)
    pos[0] = cupy.array([-0.5, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[0] = cupy.uint32(SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR)))

    # Woke particle at (0.5, 0, 0)
    pos[1] = cupy.array([0.5, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[1] = cupy.uint32(SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR)))

    # Sleeping particle at origin (0, 0, 0) -- far from both woke particles
    # (-0.5 to 0.0 = 12.5 cells away, well beyond 3x3x3 reach)
    pos[2] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[2] = cupy.uint32(SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)))
    sc[2] = cupy.uint8(100)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n)

    pi_host = pi.get()
    sc_host = sc.get()

    # Origin particle should still be sleeping (12+ cells away from both woke particles)
    assert IS_SLEEPING(int(pi_host[2])) == 1, \
        "Particle at origin should stay sleeping (too far from woke particles)"
    assert int(sc_host[2]) == 100

    # Both woke particles should have JUST_WOKE cleared
    assert HAS_JUST_WOKE(int(pi_host[0])) == 0
    assert HAS_JUST_WOKE(int(pi_host[1])) == 0
    print("  Multiple woke particles test passed")


def test_adjacent_sleeping_wakes() -> None:
    """A sleeping particle in an adjacent cell to a just-woke particle should wake."""
    print("--- Adjacent sleeping particle wakes ---")
    setup_params()

    # cell_size = 0.04. Place woke particle at (0.0, 0.0, 0.0) -> cell (25, 25, 25)
    # Place sleeping particle one cell away at (0.04, 0.0, 0.0) -> cell (26, 25, 25)
    n = 2
    pos = cupy.zeros((n, 4), dtype=cupy.float32)
    vel = cupy.zeros((n, 4), dtype=cupy.float32)  # zero velocity
    pi = cupy.zeros(n, dtype=cupy.uint32)
    sc = cupy.zeros(n, dtype=cupy.uint8)

    pos[0] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[0] = cupy.uint32(SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR)))

    pos[1] = cupy.array([0.04, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[1] = cupy.uint32(SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)))
    sc[1] = cupy.uint8(200)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n)

    pi_host = pi.get()
    sc_host = sc.get()

    assert IS_SLEEPING(int(pi_host[1])) == 0, \
        "Adjacent sleeping particle should wake up"
    assert int(sc_host[1]) == 0, "Sleep counter should be reset"
    print("  Adjacent sleeping particle woke up correctly")


def test_diagonal_neighbor_wakes() -> None:
    """A sleeping particle in a diagonal neighbor cell should also wake."""
    print("--- Diagonal neighbor wakes ---")
    setup_params()

    # Woke at (0.0, 0.0, 0.0) -> cell (25, 25, 25)
    # Sleeping at (0.04, 0.04, 0.04) -> cell (26, 26, 26) -- diagonal neighbor
    n = 2
    pos = cupy.zeros((n, 4), dtype=cupy.float32)
    vel = cupy.zeros((n, 4), dtype=cupy.float32)  # zero velocity
    pi = cupy.zeros(n, dtype=cupy.uint32)
    sc = cupy.zeros(n, dtype=cupy.uint8)

    pos[0] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[0] = cupy.uint32(SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR)))

    pos[1] = cupy.array([0.04, 0.04, 0.04, 1.0], dtype=cupy.float32)
    pi[1] = cupy.uint32(SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)))
    sc[1] = cupy.uint8(255)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n)

    pi_host = pi.get()
    sc_host = sc.get()

    assert IS_SLEEPING(int(pi_host[1])) == 0, \
        "Diagonal sleeping particle should wake up"
    assert int(sc_host[1]) == 0
    print("  Diagonal neighbor woke up correctly")


def test_two_cells_away_stays_sleeping() -> None:
    """A sleeping particle 2 cells away should NOT wake up."""
    print("--- Two cells away stays sleeping ---")
    setup_params()

    # Woke at (0.0, 0.0, 0.0) -> cell (25, 25, 25)
    # Sleeping at (0.08, 0.0, 0.0) -> cell (27, 25, 25) -- 2 cells away
    n = 2
    pos = cupy.zeros((n, 4), dtype=cupy.float32)
    vel = cupy.zeros((n, 4), dtype=cupy.float32)  # zero velocity
    pi = cupy.zeros(n, dtype=cupy.uint32)
    sc = cupy.zeros(n, dtype=cupy.uint8)

    pos[0] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[0] = cupy.uint32(SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR)))

    pos[1] = cupy.array([0.09, 0.0, 0.0, 1.0], dtype=cupy.float32)  # >2 cells away
    pi[1] = cupy.uint32(SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)))
    sc[1] = cupy.uint8(100)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n)

    pi_host = pi.get()
    sc_host = sc.get()

    assert IS_SLEEPING(int(pi_host[1])) == 1, \
        "Particle 2+ cells away should still be sleeping"
    assert int(sc_host[1]) == 100
    print("  Two cells away stayed sleeping (correct)")


def test_non_sleeping_unaffected() -> None:
    """Non-sleeping, non-just-woke particles should be unaffected."""
    print("--- Non-sleeping particles unaffected ---")
    setup_params()

    n = 3
    pos = cupy.zeros((n, 4), dtype=cupy.float32)
    pos[:, 3] = 1.0
    vel = cupy.zeros((n, 4), dtype=cupy.float32)  # zero velocity

    packed_normal = MAKE_PACKED(SAND, GRANULAR)
    packed_woke = SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR))

    pi = cupy.array([
        np.uint32(packed_woke),    # will have JUST_WOKE cleared
        np.uint32(packed_normal),  # should be completely unchanged
        np.uint32(packed_normal),  # should be completely unchanged
    ], dtype=cupy.uint32)
    sc = cupy.zeros(n, dtype=cupy.uint8)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n)

    pi_host = pi.get()
    # Normal particles should be unchanged
    assert int(pi_host[1]) == packed_normal
    assert int(pi_host[2]) == packed_normal
    # Woke particle should have JUST_WOKE cleared but otherwise same
    expected_0 = CLEAR_JUST_WOKE(packed_woke)
    assert int(pi_host[0]) == expected_0
    print("  Non-sleeping particles unaffected (correct)")


def _moving_particle_above_sleeping_sand(mat_id: int, behavior: int) -> tuple[int, int]:
    """Run wake propagation with a particle moving up at 2 m/s one cell above
    a sleeping SAND particle. Returns (packed_info, sleep_counter) of the sand."""
    setup_params()
    n = 2
    pos = cupy.zeros((n, 4), dtype=cupy.float32)
    vel = cupy.zeros((n, 4), dtype=cupy.float32)
    pi = cupy.zeros(n, dtype=cupy.uint32)
    sc = cupy.zeros(n, dtype=cupy.uint8)

    # cell_size = 0.04: mover at (0, 0.04, 0) is one cell above the sand.
    pos[0] = cupy.array([0.0, 0.04, 0.0, 1.0], dtype=cupy.float32)
    vel[0] = cupy.array([0.0, 2.0, 0.0, 0.0], dtype=cupy.float32)
    pi[0] = cupy.uint32(MAKE_PACKED(mat_id, behavior))

    pos[1] = cupy.array([0.0, 0.0, 0.0, 1.0], dtype=cupy.float32)
    pi[1] = cupy.uint32(SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)))
    sc[1] = cupy.uint8(200)

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=n)
    return int(pi.get()[1]), int(sc.get()[1])


def test_gas_does_not_wake() -> None:
    """A fast GAS particle gets no forces across phases, so it must not wake
    a sleeping GRANULAR particle next to it."""
    print("--- GAS does not wake sleeping sand ---")
    sand_pi, sand_sc = _moving_particle_above_sleeping_sand(SMOKE, GAS)
    assert IS_SLEEPING(sand_pi) == 1, "SMOKE moving at 2 m/s woke sleeping SAND"
    assert sand_sc == 200
    print("  Sleeping sand stayed asleep under moving smoke (correct)")


def test_fluid_wakes_sleeping_sand() -> None:
    """Control for test_gas_does_not_wake: a FLUID particle in the same place
    and at the same speed wakes the sand."""
    print("--- FLUID wakes sleeping sand ---")
    sand_pi, sand_sc = _moving_particle_above_sleeping_sand(WATER, FLUID)
    assert IS_SLEEPING(sand_pi) == 0, "WATER moving at 2 m/s should wake SAND"
    assert sand_sc == 0
    print("  Sleeping sand woke under moving water (correct)")


def _rigid_body_over_sleeping_sand(lin_vel, ang_vel=(0.0, 0.0, 0.0),
                                   body_pos=(0.0, 0.05, 0.0)) -> tuple[int, int]:
    """Run wake propagation with one sleeping SAND particle at the origin and
    one 0.1 m box whose bounding cube (grown by one cell) covers the origin.
    Returns (packed_info, sleep_counter) of the sand (bd-r5-review-n28.87)."""
    from rigid_bodies import RIGID_BODY_DTYPE

    setup_params()
    pos = cupy.zeros((1, 4), dtype=cupy.float32)
    pos[0, 3] = 1.0
    vel = cupy.zeros((1, 4), dtype=cupy.float32)
    pi = cupy.array([SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR))], dtype=cupy.uint32)
    sc = cupy.array([200], dtype=cupy.uint8)

    body = np.zeros(1, dtype=RIGID_BODY_DTYPE)
    body["position"][0] = (*body_pos, 1.0)
    body["rotation"][0] = (0.0, 0.0, 0.0, 1.0)
    body["lin_vel"][0] = (*lin_vel, 0.5)
    body["ang_vel"][0] = (*ang_vel, 0.3)
    body["half_extents"][0] = (0.05, 0.05, 0.05, 0.0)
    d_bodies = cupy.asarray(body.view(np.float32))  # raw 96-byte struct

    flags = allocate_cell_wake_flags(NUM_CELLS)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=1,
                         rigid_bodies=d_bodies, num_bodies=1)
    return int(pi.get()[0]), int(sc.get()[0])


def test_moving_rigid_body_wakes_sleeping_sand() -> None:
    """A falling rigid body wakes the sleeping sand under it. Its boundary
    particles are STATIC, so without K_MarkWakeCellsRigid the sand stays
    asleep and the body falls through the pile."""
    sand_pi, sand_sc = _rigid_body_over_sleeping_sand((0.0, -1.0, 0.0))
    assert IS_SLEEPING(sand_pi) == 0, "falling rigid body did not wake sand"
    assert sand_sc == 0


def test_spinning_rigid_body_wakes_sleeping_sand() -> None:
    """A body spinning in place (|omega| * r > V_WAKE) also wakes sand."""
    sand_pi, _ = _rigid_body_over_sleeping_sand((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert IS_SLEEPING(sand_pi) == 0, "spinning rigid body did not wake sand"


def test_resting_rigid_body_does_not_wake() -> None:
    """A body at rest marks no cells, so the sand stays asleep."""
    sand_pi, sand_sc = _rigid_body_over_sleeping_sand((0.0, 0.0, 0.0))
    assert IS_SLEEPING(sand_pi) == 1, "resting rigid body woke sand"
    assert sand_sc == 200
    # Below V_WAKE (0.02 m/s) also counts as resting.
    sand_pi, _ = _rigid_body_over_sleeping_sand((0.0, -0.01, 0.0))
    assert IS_SLEEPING(sand_pi) == 1, "body below V_WAKE woke sand"


def test_far_rigid_body_does_not_wake() -> None:
    """A moving body far from the sand leaves it asleep."""
    sand_pi, _ = _rigid_body_over_sleeping_sand((0.0, -1.0, 0.0),
                                                body_pos=(0.0, 0.5, 0.0))
    assert IS_SLEEPING(sand_pi) == 1, "far rigid body woke sand"


def test_mark_wake_cells_rigid_zero_bodies() -> None:
    """num_bodies = 0 launches nothing and marks nothing."""
    setup_params()
    flags = allocate_cell_wake_flags(NUM_CELLS)
    mark_wake_cells_rigid(cupy.zeros(1, dtype=cupy.float32), 0, flags)
    assert int(cupy.count_nonzero(flags)) == 0


# ---------------------------------------------------------------------------
# Generation stamp (bd-r5-review-n28.141)
# ---------------------------------------------------------------------------


def _pair(woke_pos, sleep_pos, woke_packed=None):
    """Arrays for one marker particle and one sleeping SAND particle."""
    pos = cupy.zeros((2, 4), dtype=cupy.float32)
    vel = cupy.zeros((2, 4), dtype=cupy.float32)
    pos[0] = cupy.array([*woke_pos, 1.0], dtype=cupy.float32)
    pos[1] = cupy.array([*sleep_pos, 1.0], dtype=cupy.float32)
    if woke_packed is None:
        woke_packed = SET_JUST_WOKE(MAKE_PACKED(SAND, GRANULAR))
    pi = cupy.array([woke_packed, SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR))],
                    dtype=cupy.uint32)
    sc = cupy.array([0, 100], dtype=cupy.uint8)
    return pos, vel, pi, sc


def test_stamp_no_memset() -> None:
    """Two substeps with no memset between them: a mark from substep 1 does
    not wake a sleeper in substep 2, because substep 2 uses a new stamp."""
    setup_params()
    flags = allocate_cell_wake_flags(NUM_CELLS)
    frame = cupy.array([41], dtype=cupy.uint32)

    # Substep 1: a just-woke marker at the origin, sleeper far away.
    pos, vel, pi, sc = _pair((0.0, 0.0, 0.0), (0.9, 0.9, 0.9))
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=2,
                         frame_counter=frame)
    assert IS_SLEEPING(int(pi.get()[1])) == 1
    assert int(cupy.count_nonzero(flags)) == 1, "marker stamps its own cell"

    # Substep 2: the marker is at rest now (no mark); a sleeper sits one
    # cell from the old mark. The old mark is still in the table.
    rest = MAKE_PACKED(SAND, GRANULAR)
    pos, vel, pi, sc = _pair((0.0, 0.0, 0.0), (0.04, 0.0, 0.0), woke_packed=rest)
    frame.fill(42)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=2,
                         frame_counter=frame)
    assert int(cupy.count_nonzero(flags)) == 1, "no memset between substeps"
    assert IS_SLEEPING(int(pi.get()[1])) == 1, "stale mark woke a sleeper"
    assert int(sc.get()[1]) == 100

    # Control: with the substep-1 stamp the same sleeper would wake, so the
    # test above really exercises a stale mark.
    frame.fill(41)
    wake_sleepers_and_clear_just_woke(pos, pi, sc, flags, num_particles=2,
                                      frame_counter=frame)
    assert IS_SLEEPING(int(pi.get()[1])) == 0


def _cascade_with_stamp(sleep_pos) -> int:
    """Run one stamped substep on a table full of stale stamps; return the
    sleeper's packed_info."""
    setup_params()
    flags = cupy.full(NUM_CELLS, 7, dtype=cupy.uint32)  # stamp of substep 6
    frame = cupy.array([100], dtype=cupy.uint32)
    pos, vel, pi, sc = _pair((0.0, 0.0, 0.0), sleep_pos)
    run_wake_propagation(pos, vel, pi, sc, flags, num_particles=2,
                         frame_counter=frame)
    return int(pi.get()[1])


def test_stamp_cascade_matches_flag_semantics() -> None:
    """With the stamp and a dirty table, the cascade keeps the old reach:
    own, face and diagonal cells wake; two cells away stays asleep."""
    assert IS_SLEEPING(_cascade_with_stamp((0.0, 0.0, 0.0))) == 0
    assert IS_SLEEPING(_cascade_with_stamp((0.04, 0.0, 0.0))) == 0
    assert IS_SLEEPING(_cascade_with_stamp((-0.04, 0.04, 0.04))) == 0
    assert IS_SLEEPING(_cascade_with_stamp((0.09, 0.0, 0.0))) == 1
    assert IS_SLEEPING(_cascade_with_stamp((0.0, -0.09, 0.0))) == 1


def test_rigid_stamp_reach() -> None:
    """A moving body stamps its cube; a sleeper one cell past the cube wakes,
    one two cells past stays asleep."""
    from rigid_bodies import RIGID_BODY_DTYPE

    setup_params()
    body = np.zeros(1, dtype=RIGID_BODY_DTYPE)
    body["position"][0] = (0.0, 0.0, 0.0, 1.0)
    body["rotation"][0] = (0.0, 0.0, 0.0, 1.0)
    body["lin_vel"][0] = (0.0, -1.0, 0.0, 0.5)
    body["ang_vel"][0] = (0.0, 0.0, 0.0, 0.3)
    body["half_extents"][0] = (0.02, 0.02, 0.02, 0.0)  # r = 0.0346
    d_bodies = cupy.asarray(body.view(np.float32))

    def sleeper_after(x: float) -> int:
        flags = cupy.full(NUM_CELLS, 3, dtype=cupy.uint32)
        frame = cupy.array([9], dtype=cupy.uint32)
        pos = cupy.array([[x, 0.0, 0.0, 1.0]], dtype=cupy.float32)
        vel = cupy.zeros((1, 4), dtype=cupy.float32)
        pi = cupy.array([SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR))], dtype=cupy.uint32)
        sc = cupy.array([100], dtype=cupy.uint8)
        run_wake_propagation(pos, vel, pi, sc, flags, num_particles=1,
                             rigid_bodies=d_bodies, num_bodies=1,
                             frame_counter=frame)
        return int(pi.get()[0])

    # Cube [-0.0346, 0.0346] covers cells 24..25 in x (cell 0.04, origin at
    # cell 25). Cell 26 (x=0.05) is adjacent, cell 27 (x=0.09) is not.
    assert IS_SLEEPING(sleeper_after(0.05)) == 0
    assert IS_SLEEPING(sleeper_after(0.09)) == 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("test_wake.py -- cell-flag wake propagation kernels")
    print("=" * 60)

    test_compilation()
    test_block_size()
    test_cell_wake_flags_allocation()
    test_mark_wake_cells_single_particle()
    test_mark_wake_cells_corner_particle()
    test_mark_wake_cells_no_just_woke()
    test_wake_sleepers_in_flagged_cell()
    test_wake_sleepers_unflagged_cell()
    test_clear_just_woke()
    test_full_pipeline_impact_wake()
    test_far_particles_stay_sleeping()
    test_stress_500k()
    test_multiple_woke_particles()
    test_adjacent_sleeping_wakes()
    test_diagonal_neighbor_wakes()
    test_two_cells_away_stays_sleeping()
    test_non_sleeping_unaffected()
    test_gas_does_not_wake()
    test_fluid_wakes_sleeping_sand()
    test_moving_rigid_body_wakes_sleeping_sand()
    test_spinning_rigid_body_wakes_sleeping_sand()
    test_resting_rigid_body_does_not_wake()
    test_far_rigid_body_does_not_wake()
    test_mark_wake_cells_rigid_zero_bodies()
    test_stamp_no_memset()
    test_stamp_cascade_matches_flag_semantics()
    test_rigid_stamp_reach()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
