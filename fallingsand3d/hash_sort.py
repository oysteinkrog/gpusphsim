"""Grid hashing kernel for spatial partitioning of SPH particles.

Compiles physics/kernels/hash_sort.cu via CuPy RawModule and provides
functions to upload GridParams to ``__constant__ GridParams c_grid`` and
launch the ``K_CalcHash`` kernel.

Grid sizing (from acceptance criteria)
--------------------------------------
- grid_min = (-1, -1, -1), grid_max = (1, 1, 1)
- cell_size = h = 0.04
- grid_res  = 50 per axis
- grid_delta = (25, 25, 25)   (= grid_res / grid_size = 50/2)
- num_cells  = 125000          (= 50^3)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cupy
import numpy as np

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

GRID_MIN = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
GRID_MAX = np.array([1.0, 1.0, 1.0], dtype=np.float32)
CELL_SIZE = np.float32(0.04)
GRID_RES = np.array([50, 50, 50], dtype=np.int32)
GRID_DELTA = np.array([25.0, 25.0, 25.0], dtype=np.float32)  # grid_res / grid_size
NUM_CELLS = 125_000  # 50 * 50 * 50

# ---------------------------------------------------------------------------
# Numpy dtype matching ``struct GridParams`` in common.cuh
#
# struct GridParams {
#     float3 grid_min;    // 12 bytes
#     float3 grid_max;    // 12 bytes
#     int3   grid_res;    // 12 bytes
#     float3 grid_delta;  // 12 bytes
#     int    num_cells;   //  4 bytes
# };                      // Total: 52 bytes
#
# CUDA float3/int3 are 12 bytes with no trailing pad. align=False required.
# ---------------------------------------------------------------------------

GRID_PARAMS_DTYPE = np.dtype(
    [
        ("grid_min", np.float32, (3,)),
        ("grid_max", np.float32, (3,)),
        ("grid_res", np.int32, (3,)),
        ("grid_delta", np.float32, (3,)),
        ("num_cells", np.int32),
    ],
    align=False,
)

assert GRID_PARAMS_DTYPE.itemsize == 52, (
    f"GridParams size mismatch: {GRID_PARAMS_DTYPE.itemsize} != 52"
)


def build_grid_params() -> np.ndarray:
    """Build a single GridParams struct using the default grid constants."""
    params = np.zeros(1, dtype=GRID_PARAMS_DTYPE)
    params[0]["grid_min"] = GRID_MIN
    params[0]["grid_max"] = GRID_MAX
    params[0]["grid_res"] = GRID_RES
    params[0]["grid_delta"] = GRID_DELTA
    params[0]["num_cells"] = NUM_CELLS
    return params


def build_grid_params_for_world(
    grid_min,
    grid_max,
    cell_size: float = 0.04,
) -> Tuple[np.ndarray, int]:
    """Build a GridParams struct from arbitrary world bounds and cell size.

    Returns
    -------
    params : np.ndarray
        A single GridParams structured array element.
    num_cells : int
        Total number of grid cells.
    """
    gmin = np.array(grid_min, dtype=np.float32)
    gmax = np.array(grid_max, dtype=np.float32)
    grid_size = gmax - gmin
    grid_res = np.ceil(grid_size / cell_size).astype(np.int32)
    grid_delta = grid_res.astype(np.float32) / grid_size
    num_cells = int(grid_res[0]) * int(grid_res[1]) * int(grid_res[2])

    params = np.zeros(1, dtype=GRID_PARAMS_DTYPE)
    params[0]["grid_min"] = gmin
    params[0]["grid_max"] = gmax
    params[0]["grid_res"] = grid_res
    params[0]["grid_delta"] = grid_delta
    params[0]["num_cells"] = num_cells
    return params, num_cells


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
