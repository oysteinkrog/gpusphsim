"""BuildDataStruct kernel for building cell start/end index tables.

Compiles physics/kernels/build_grid.cu via CuPy RawModule and provides
a function to launch K_BuildDataStruct which detects cell boundaries in
the sorted hash array and populates cell_indexes_start / cell_indexes_end.

Grid sizing (matches hash_sort.py)
-----------------------------------
- num_cells = 125000 (50^3)
- cell_indexes_start: uint32[num_cells], memset to 0xFFFFFFFF before each frame
- cell_indexes_end:   uint32[num_cells]
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cupy
import numpy as np

from hash_sort import NUM_CELLS, build_grid_params

# Sentinel value for empty cells (matches 0xFFFFFFFF memset).
EMPTY_CELL = np.uint32(0xFFFFFFFF)

BLOCK_SIZE = 256

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
    """Compile (or return cached) CuPy RawModule from build_grid.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "build_grid.cu")
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
    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]

    cupy.cuda.runtime.memcpy(
        int(d_ptr),
        params.ctypes.data,
        params.nbytes,
        1,  # cudaMemcpyHostToDevice
    )


# ---------------------------------------------------------------------------
# Array allocation
# ---------------------------------------------------------------------------


def allocate_cell_tables() -> Tuple[cupy.ndarray, cupy.ndarray]:
    """Allocate cell_indexes_start and cell_indexes_end as CuPy uint32 arrays.

    Returns
    -------
    cell_start : cupy.ndarray, shape (NUM_CELLS,), dtype uint32
    cell_end   : cupy.ndarray, shape (NUM_CELLS,), dtype uint32
    """
    cell_start = cupy.empty(NUM_CELLS, dtype=cupy.uint32)
    cell_end = cupy.empty(NUM_CELLS, dtype=cupy.uint32)
    return cell_start, cell_end


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------


def build_data_struct(
    sorted_hashes: cupy.ndarray,
    cell_start: Optional[cupy.ndarray] = None,
    cell_end: Optional[cupy.ndarray] = None,
) -> Tuple[cupy.ndarray, cupy.ndarray]:
    """Launch K_BuildDataStruct to build cell start/end index tables.

    Parameters
    ----------
    sorted_hashes : cupy.ndarray
        (N,) uint32 array of particle hashes, sorted in ascending order.
    cell_start : cupy.ndarray, optional
        Pre-allocated (NUM_CELLS,) uint32 array.  If None, allocates new.
    cell_end : cupy.ndarray, optional
        Pre-allocated (NUM_CELLS,) uint32 array.  If None, allocates new.

    Returns
    -------
    cell_start : cupy.ndarray, shape (NUM_CELLS,), dtype uint32
        cell_start[hash] = first particle index in that cell, or 0xFFFFFFFF
        if cell is empty.
    cell_end : cupy.ndarray, shape (NUM_CELLS,), dtype uint32
        cell_end[hash] = one past last particle index in that cell.
    """
    n = sorted_hashes.shape[0]

    if cell_start is None or cell_end is None:
        cell_start, cell_end = allocate_cell_tables()

    # Memset cell_start to 0xFFFFFFFF (empty sentinel).
    cell_start.data.memset(0xFF, cell_start.nbytes)

    # cell_end can stay uninitialized -- only written cells matter, and they
    # are always paired with a cell_start write.  Zero it for safety.
    cell_end.data.memset(0x00, cell_end.nbytes)

    module = _get_module()
    kernel = module.get_function("K_BuildDataStruct")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(grid, block, (np.uint32(n), sorted_hashes, cell_start, cell_end))

    return cell_start, cell_end
