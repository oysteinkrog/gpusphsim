"""Spatial hash grid parameters shared by every CUDA module.

Kernel-free: this module only builds the numpy ``GridParams`` struct.
Each kernel module uploads it to its own ``__constant__ GridParams c_grid``
(counting_sort, step1, step2, solvers, neighbor_list, ...).

Uses a spatial hash with a fixed-size table (power of 2) instead of dense
arrays sized grid_res^3, so world size does not drive memory use.

Grid params:
- grid_min: world-space minimum corner
- grid_delta: 1/cell_size per axis (= 1/h)
- table_size: hash table size (power of 2)
- table_mask: table_size - 1
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

CELL_SIZE = np.float32(0.04)

# Default hash table size for small particle counts.
# Scaled up via compute_table_size() for large N.
TABLE_SIZE = 262144
TABLE_MASK = TABLE_SIZE - 1

# Backward-compat alias for old test files
NUM_CELLS = TABLE_SIZE


def compute_table_size(num_particles: int) -> int:
    """Compute hash table size for given particle count.

    Sized so the *occupied cell count* has good load factor.
    At spacing=0.02 / cell_size=0.04: ~8 particles per cell,
    so num_cells ~= N/8.  We want table ~= 4x num_cells = N/2.
    Minimum: 262144 (2^18).  Maximum: 1048576 (2^20) to limit
    memset/histogram overhead.
    """
    target = max(num_particles // 2, 262144)
    target = min(target, 1048576)  # cap at 1M entries
    # Round up to next power of 2
    target -= 1
    target |= target >> 1
    target |= target >> 2
    target |= target >> 4
    target |= target >> 8
    target |= target >> 16
    return target + 1


# ---------------------------------------------------------------------------
# Numpy dtype matching ``struct GridParams`` in common.cuh
#
# struct GridParams {
#     float3 grid_min;    // 12 bytes
#     float3 grid_delta;  // 12 bytes
#     uint   table_size;  //  4 bytes
#     uint   table_mask;  //  4 bytes
# };                      // Total: 32 bytes
#
# CUDA float3 is 12 bytes with no trailing pad. align=False required.
# ---------------------------------------------------------------------------

GRID_PARAMS_DTYPE = np.dtype(
    [
        ("grid_min", np.float32, (3,)),
        ("grid_delta", np.float32, (3,)),
        ("table_size", np.uint32),
        ("table_mask", np.uint32),
    ],
    align=False,
)

assert GRID_PARAMS_DTYPE.itemsize == 32, (
    f"GridParams size mismatch: {GRID_PARAMS_DTYPE.itemsize} != 32"
)


def build_grid_params() -> np.ndarray:
    """Build a single GridParams struct using the default grid constants."""
    gmin = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    inv_cs = 1.0 / float(CELL_SIZE)
    grid_delta = np.array([inv_cs, inv_cs, inv_cs], dtype=np.float32)
    params = np.zeros(1, dtype=GRID_PARAMS_DTYPE)
    params[0]["grid_min"] = gmin
    params[0]["grid_delta"] = grid_delta
    params[0]["table_size"] = np.uint32(TABLE_SIZE)
    params[0]["table_mask"] = np.uint32(TABLE_MASK)
    return params


def build_grid_params_for_world(
    grid_min,
    grid_max,
    cell_size: float = 0.04,
    num_particles: int = 0,
) -> Tuple[np.ndarray, int]:
    """Build a GridParams struct from arbitrary world bounds and cell size.

    Parameters
    ----------
    grid_max
        Unused by the spatial hash (kept for call-site compatibility).
    num_particles : int
        Active particle count. When > 0, hash table is auto-sized to
        maintain ~50% load factor. When 0, uses default TABLE_SIZE.

    Returns
    -------
    params : np.ndarray
        A single GridParams structured array element.
    table_size : int
        Hash table size (scaled with particle count).
    """
    gmin = np.array(grid_min, dtype=np.float32)
    inv_cs = 1.0 / cell_size
    grid_delta = np.array([inv_cs, inv_cs, inv_cs], dtype=np.float32)

    ts = compute_table_size(num_particles) if num_particles > 0 else TABLE_SIZE
    tm = ts - 1

    params = np.zeros(1, dtype=GRID_PARAMS_DTYPE)
    params[0]["grid_min"] = gmin
    params[0]["grid_delta"] = grid_delta
    params[0]["table_size"] = np.uint32(ts)
    params[0]["table_mask"] = np.uint32(tm)
    return params, ts
