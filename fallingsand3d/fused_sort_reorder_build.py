"""Fused sort-reorder-build kernel -- combines reorder + build_grid in one pass.

Compiles physics/kernels/fused_sort_reorder_build.cu via CuPy RawModule.
Replaces the separate K_FusedReorder + K_BuildDataStruct pipeline with a
single kernel launch that reads sort_perm once per thread, gathers 8 particle
arrays, writes sorted hashes, and detects cell boundaries.

Pre-conditions:
  - cell_start memset to 0xFFFFFFFF before launch
  - cell_end memset to 0x00 before launch
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

BLOCK_SIZE = 256


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
    """Compile (or return cached) CuPy RawModule from fused_sort_reorder_build.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "fused_sort_reorder_build.cu")
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
    d_ptr = module.get_global("c_grid")
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------


def fused_sort_reorder_build(
    num_particles: int,
    sort_perm: cupy.ndarray,
    hashes: cupy.ndarray,
    sorted_hashes: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    # Unsorted source arrays (8 arrays)
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    temperature: cupy.ndarray,
    health: cupy.ndarray,
    lifetime: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    # Sorted destination arrays (8 arrays)
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_lifetime: cupy.ndarray,
    sorted_sleep_counter: cupy.ndarray,
) -> None:
    """Launch K_FusedSortReorderBuild.

    Combines sort-gather, reorder, and build_grid into a single kernel.
    cell_start must be memset to 0xFF and cell_end to 0x00 before calling.

    Parameters
    ----------
    num_particles : int
        Number of active particles.
    sort_perm : cupy.ndarray, (N,) uint32
        Argsort result (permutation that sorts by hash).
    hashes : cupy.ndarray, (N,) uint32
        Unsorted per-particle hashes from K_CalcHash.
    sorted_hashes : cupy.ndarray, (N,) uint32
        Output: sorted hashes.
    cell_start, cell_end : cupy.ndarray, (num_cells,) uint32
        Output: cell boundary tables.
    position .. sleep_counter : cupy.ndarray
        Unsorted source arrays.
    sorted_position .. sorted_sleep_counter : cupy.ndarray
        Sorted destination arrays.
    """
    if num_particles == 0:
        return

    module = _get_module()
    kernel = module.get_function("K_FusedSortReorderBuild")

    grid = ((num_particles + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(num_particles),
            sort_perm,
            hashes,
            sorted_hashes,
            cell_start,
            cell_end,
            # Unsorted inputs (8 arrays)
            position,
            velocity,
            mass,
            packed_info,
            temperature,
            health,
            lifetime,
            sleep_counter,
            # Sorted outputs (8 arrays)
            sorted_position,
            sorted_velocity,
            sorted_mass,
            sorted_packed_info,
            sorted_temperature,
            sorted_health,
            sorted_lifetime,
            sorted_sleep_counter,
        ),
    )
