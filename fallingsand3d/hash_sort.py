"""Grid hashing kernel for spatial partitioning of SPH particles.

Compiles physics/kernels/hash_sort.cu via CuPy RawModule and provides
functions to upload GridParams to ``__constant__ GridParams c_grid`` and
launch the ``K_CalcHash`` kernel.

Uses a spatial hash with fixed-size table (power of 2) instead of
dense arrays sized grid_res^3.  Enables arbitrarily large worlds.

Grid params:
- grid_min: world-space minimum corner
- grid_delta: 1/cell_size per axis (= 1/h)
- table_size: hash table size (power of 2)
- table_mask: table_size - 1
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cupy
import numpy as np

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

CELL_SIZE = np.float32(0.04)

# Default hash table size for small particle counts.
# Dynamically scaled up via compute_table_size() for large N.
TABLE_SIZE = 262144
TABLE_MASK = TABLE_SIZE - 1

# Backward-compat alias for old test files
NUM_CELLS = TABLE_SIZE


def compute_table_size(num_particles: int) -> int:
    """Compute hash table size for given particle count.

    Sized so the *occupied cell count* has good load factor.
    At spacing=0.02 / cell_size=0.04: ~8 particles per cell,
    so num_cells ≈ N/8.  We want table ≈ 4x num_cells = N/2.
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
    num_particles : int
        Active particle count. When > 0, hash table is auto-sized to
        maintain ~50% load factor. When 0, uses default TABLE_SIZE.

    Returns
    -------
    params : np.ndarray
        A single GridParams structured array element.
    table_size : int
        Hash table size (dynamically scaled based on particle count).
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


# ---------------------------------------------------------------------------
# CuPy RawModule compilation
# ---------------------------------------------------------------------------

_module: Optional[object] = None


def _ensure_ptx_if_needed() -> None:
    """Force PTX compilation when GPU arch exceeds NVRTC's max sm target."""
    from cupy.cuda import compiler as _compiler
    from cupy.cuda import device as _device

    gpu_cc = _device.Device().compute_capability
    nvrtc_max = _compiler._get_max_compute_capability()
    if int(gpu_cc) > int(nvrtc_max):
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
            _compiler._get_arch_for_options_for_nvrtc._cache = {}
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}


def _get_module() -> "object":
    """Compile (or return cached) CuPy RawModule from hash_sort.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "hash_sort.cu")
    with open(cu_path) as f:
        source = f.read()

    _module = cupy.RawModule(
        code=source,
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )
    return _module


def get_module() -> "object":
    """Return the compiled CuPy RawModule (public accessor)."""
    return _get_module()


# ---------------------------------------------------------------------------
# Constant memory upload
# ---------------------------------------------------------------------------


def upload_grid_params(params: Optional[np.ndarray] = None) -> None:
    """Upload GridParams to ``__constant__ GridParams c_grid``.

    Parameters
    ----------
    params : numpy structured array, optional
        A single GridParams element.  If *None*, uses ``build_grid_params()``.
    """
    if params is None:
        params = build_grid_params()

    module = _get_module()
    d_ptr = module.get_global("c_grid")

    cupy.cuda.runtime.memcpy(
        int(d_ptr),
        params.ctypes.data,
        params.nbytes,
        1,  # cudaMemcpyHostToDevice
    )


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def calc_hash(
    positions: cupy.ndarray,
    hashes_out: Optional[cupy.ndarray] = None,
) -> cupy.ndarray:
    """Launch K_CalcHash and return hashes array.

    Parameters
    ----------
    positions : cupy.ndarray
        (N, 4) float32 array of particle positions (x, y, z, w).
    hashes_out : cupy.ndarray, optional
        Pre-allocated (N,) uint32 output buffer. If None, allocates new.

    Returns
    -------
    hashes : cupy.ndarray, shape (N,), dtype uint32
    """
    n = positions.shape[0]
    if hashes_out is not None:
        hashes = hashes_out[:n]
    else:
        hashes = cupy.empty(n, dtype=cupy.uint32)

    module = _get_module()
    kernel = module.get_function("K_CalcHash")

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(grid, block, (np.uint32(n), positions, hashes))

    return hashes


# ---------------------------------------------------------------------------
# Sort by hash
# ---------------------------------------------------------------------------


def sort_by_hash(
    hashes: cupy.ndarray,
    sorted_hashes_out: Optional[cupy.ndarray] = None,
    sorted_indices_out: Optional[cupy.ndarray] = None,
) -> Tuple[cupy.ndarray, cupy.ndarray]:
    """Sort particles by grid hash using CuPy argsort (Thrust radix sort).

    Parameters
    ----------
    hashes : cupy.ndarray, shape (N,), dtype uint32
        Per-particle grid hash values from ``calc_hash()``.
    sorted_hashes_out : cupy.ndarray, optional
        Pre-allocated output buffer for sorted hashes.  If *None*, allocates.
    sorted_indices_out : cupy.ndarray, optional
        Pre-allocated output buffer for sorted indices.  If *None*,
        allocates.

    Returns
    -------
    sorted_hashes : cupy.ndarray, shape (N,), dtype uint32
        Hash values in non-decreasing order.
    sorted_indices : cupy.ndarray, shape (N,), dtype uint32
        sort_perm itself: sorted_indices[i] = original particle index for
        sorted slot i. (Formerly indices[sort_perm], but indices was always
        identity, so sort_perm == sorted_indices.)
    """
    n = hashes.shape[0]

    # cupy.argsort uses Thrust radix sort internally for integer dtypes;
    # cast to uint32 since CUDA kernels expect uint* pointers
    sort_perm = cupy.argsort(hashes).astype(cupy.uint32)

    # Gather sorted hashes
    if sorted_hashes_out is not None:
        sorted_hashes_out[:n] = hashes[sort_perm]
        sorted_hashes = sorted_hashes_out[:n]
    else:
        sorted_hashes = hashes[sort_perm]

    # sort_perm IS sorted_indices (indices was always identity 0..N-1)
    if sorted_indices_out is not None:
        sorted_indices_out[:n] = sort_perm
        sorted_indices = sorted_indices_out[:n]
    else:
        sorted_indices = sort_perm

    return sorted_hashes, sorted_indices
