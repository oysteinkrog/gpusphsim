"""Wake propagation kernels: two-phase cell-flag wake system.

Compiles physics/kernels/wake.cu via CuPy RawModule and provides functions
to launch the wake propagation pipeline:

  Phase 1 (K_MarkWakeCells):  Just-woke particles mark 3x3x3 neighbor cells.
  Phase 2 (K_WakeSleepers):   Sleeping particles in flagged cells wake up.
  Phase 3 (K_ClearJustWoke):  Clear JUST_WOKE flag from all particles.

Uses shared constant memory from common.cuh:
  c_grid -- GridParams (for position -> grid cell mapping)
"""

from __future__ import annotations

import os
from typing import Optional

import cupy
import numpy as np

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
    """Compile (or return cached) CuPy RawModule from wake.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "wake.cu")
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


def upload_grid_params(params: np.ndarray) -> None:
    """Upload GridParams to ``__constant__ GridParams c_grid``."""
    module = _get_module()
    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


# ---------------------------------------------------------------------------
# Array allocation
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def allocate_cell_wake_flags(num_cells: int = 125_000) -> cupy.ndarray:
    """Allocate the cell_wake_flags array (uint32, num_cells).

    Returns a CuPy uint32 array of size num_cells, initialized to 0.
    """
    return cupy.zeros(num_cells, dtype=cupy.uint32)


# ---------------------------------------------------------------------------
# Kernel launches
# ---------------------------------------------------------------------------


def mark_wake_cells(
    position: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_wake_flags: cupy.ndarray,
    num_particles: Optional[int] = None,
) -> None:
    """Launch K_MarkWakeCells (Phase 1).

    For each particle with HAS_JUST_WOKE flag, atomicOr(1) on its own cell
    and 26 neighboring cells.

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Unsorted particle positions.
    packed_info : cupy.ndarray, (N,) uint32
        Unsorted packed_info (read-only for this kernel).
    cell_wake_flags : cupy.ndarray, (num_cells,) uint32
        Cell flags array. Must be cleared to 0 before calling this.
    num_particles : int, optional
        Number of active particles. Defaults to position.shape[0].
    """
    n = num_particles if num_particles is not None else position.shape[0]
    if n == 0:
        return

    module = _get_module()
    kernel = module.get_function("K_MarkWakeCells")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(grid, block, (np.uint32(n), position, packed_info, cell_wake_flags))


def wake_sleepers_and_clear_just_woke(
    position: cupy.ndarray,
    packed_info: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    cell_wake_flags: cupy.ndarray,
    num_particles: Optional[int] = None,
) -> None:
    """Launch K_WakeSleepersAndClearJustWoke (Phase 2+3 fused).

    For each sleeping particle, if cell_wake_flags[my_cell] != 0,
    clear SLEEPING flag and reset sleep_counter to 0.
    Also clears the JUST_WOKE flag from all particles.

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Unsorted particle positions.
    packed_info : cupy.ndarray, (N,) uint32
        Unsorted packed_info (read+write).
    sleep_counter : cupy.ndarray, (N,) uint8
        Unsorted sleep counter (write: reset to 0 on wake).
    cell_wake_flags : cupy.ndarray, (num_cells,) uint32
        Cell flags from Phase 1.
    num_particles : int, optional
        Number of active particles. Defaults to position.shape[0].
    """
    n = num_particles if num_particles is not None else position.shape[0]
    if n == 0:
        return

    module = _get_module()
    kernel = module.get_function("K_WakeSleepersAndClearJustWoke")

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid, block,
        (np.uint32(n), position, packed_info, sleep_counter, cell_wake_flags),
    )


def run_wake_propagation(
    position: cupy.ndarray,
    packed_info: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    cell_wake_flags: cupy.ndarray,
    num_particles: Optional[int] = None,
) -> None:
    """Run the 2-phase wake propagation pipeline.

    1. Clear cell_wake_flags to 0
    2. K_MarkWakeCells (Phase 1)
    3. K_WakeSleepersAndClearJustWoke (Phase 2+3 fused)

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Unsorted particle positions.
    packed_info : cupy.ndarray, (N,) uint32
        Unsorted packed_info (read+write).
    sleep_counter : cupy.ndarray, (N,) uint8
        Unsorted sleep counter (write on wake).
    cell_wake_flags : cupy.ndarray, (num_cells,) uint32
        Pre-allocated cell flags array.
    num_particles : int, optional
        Number of active particles.
    """
    n = num_particles if num_particles is not None else position.shape[0]
    if n == 0:
        return

    # Clear cell flags to 0 (async memset: graph-capture safe)
    cell_wake_flags.data.memset_async(0x00, cell_wake_flags.nbytes)

    # Phase 1: mark cells near just-woke particles
    mark_wake_cells(position, packed_info, cell_wake_flags, n)

    # Phase 2+3 fused: wake sleepers + clear JUST_WOKE flag
    wake_sleepers_and_clear_just_woke(
        position, packed_info, sleep_counter, cell_wake_flags, n
    )
