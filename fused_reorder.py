"""Fused reorder kernel: gather particle data from unsorted to sorted order.

Compiles physics/kernels/fused_reorder.cu via CuPy RawModule and provides
a function to launch K_FusedReorder which reorders particle arrays using
the sort_indexes permutation from radix sort.

After hash + argsort: sort_indexes[sorted_slot] = original_particle_id.
This kernel gathers: sorted[slot] = unsorted[sort_indexes[slot]].

Reorders: position, velocity, veleval, behavior_class, flags.
"""

from __future__ import annotations

import os
from typing import Optional

import cupy  # type: ignore[import-untyped]
import numpy as np

# ---------------------------------------------------------------------------
# CuPy RawModule compilation
# ---------------------------------------------------------------------------

_module: Optional[object] = None


def _ensure_ptx_if_needed() -> None:
    """Force PTX compilation when GPU arch exceeds NVRTC's max sm target."""
    from cupy.cuda import compiler as _compiler  # type: ignore[import-untyped]
    from cupy.cuda import device as _device  # type: ignore[import-untyped]

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

BLOCK_SIZE = 256


def fused_reorder(
    sort_indexes: cupy.ndarray,
    position_in: cupy.ndarray,
    velocity_in: cupy.ndarray,
    veleval_in: cupy.ndarray,
    behavior_class_in: cupy.ndarray,
    flags_in: cupy.ndarray,
    position_out: cupy.ndarray,
    velocity_out: cupy.ndarray,
    veleval_out: cupy.ndarray,
    behavior_class_out: cupy.ndarray,
    flags_out: cupy.ndarray,
) -> None:
    """Launch K_FusedReorder to gather particle data into sorted order.

    Parameters
    ----------
    sort_indexes : (N,) uint32 -- sort_indexes[sorted] = original particle ID
    position_in : (N, 4) float32 -- unsorted positions
    velocity_in : (N, 4) float32 -- unsorted velocities
    veleval_in : (N, 4) float32 -- unsorted evaluation velocities
    behavior_class_in : (N,) int32 -- unsorted behavior classes
    flags_in : (N,) uint32 -- unsorted flags
    position_out : (N, 4) float32 -- sorted positions (output)
    velocity_out : (N, 4) float32 -- sorted velocities (output)
    veleval_out : (N, 4) float32 -- sorted evaluation velocities (output)
    behavior_class_out : (N,) int32 -- sorted behavior classes (output)
    flags_out : (N,) uint32 -- sorted flags (output)
    """
    n = sort_indexes.shape[0]

    module = _get_module()
    kernel = module.get_function("K_FusedReorder")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            sort_indexes,
            position_in,
            velocity_in,
            veleval_in,
            behavior_class_in,
            flags_in,
            position_out,
            velocity_out,
            veleval_out,
            behavior_class_out,
            flags_out,
        ),
    )
