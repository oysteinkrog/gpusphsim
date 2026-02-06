"""Fused reorder kernel -- gathers ALL SoA particle arrays in one pass.

Compiles physics/kernels/fused_reorder.cu via CuPy RawModule and provides
a single ``fused_reorder()`` call that gathers all unsorted arrays into
sorted-order temporary buffers using the sorted_index permutation.

This is far more bandwidth-efficient than N separate CuPy fancy-indexing
calls because sorted_index is read from global memory only once per thread.
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
    """Compile (or return cached) CuPy RawModule from fused_reorder.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "fused_reorder.cu")
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
# Kernel launch
# ---------------------------------------------------------------------------


def fused_reorder(
    num_particles: int,
    sorted_index: cupy.ndarray,
    # Unsorted source arrays
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    veleval: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    temperature: cupy.ndarray,
    health: cupy.ndarray,
    lifetime: cupy.ndarray,
    color: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    shear_rate: cupy.ndarray,
    # Sorted destination arrays
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_veleval: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_lifetime: cupy.ndarray,
    sorted_color: cupy.ndarray,
    sorted_sleep_counter: cupy.ndarray,
    sorted_shear_rate: cupy.ndarray,
) -> None:
    """Launch K_FusedReorder to gather all SoA arrays into sorted order.

    Parameters
    ----------
    num_particles : int
        Number of active particles to process.
    sorted_index : cupy.ndarray, shape (N,), dtype uint32
        Permutation array: sorted_index[sorted_slot] = original_id.
    position .. shear_rate : cupy.ndarray
        Unsorted source arrays (indexed by original particle ID).
    sorted_position .. sorted_shear_rate : cupy.ndarray
        Pre-allocated sorted destination arrays (indexed by sorted slot).
    """
    if num_particles == 0:
        return

    module = _get_module()
    kernel = module.get_function("K_FusedReorder")

    grid = ((num_particles + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(num_particles),
            sorted_index,
            # Unsorted inputs
            position,
            velocity,
            veleval,
            mass,
            packed_info,
            temperature,
            health,
            lifetime,
            color,
            sleep_counter,
            shear_rate,
            # Sorted outputs
            sorted_position,
            sorted_velocity,
            sorted_veleval,
            sorted_mass,
            sorted_packed_info,
            sorted_temperature,
            sorted_health,
            sorted_lifetime,
            sorted_color,
            sorted_sleep_counter,
            sorted_shear_rate,
        ),
    )
