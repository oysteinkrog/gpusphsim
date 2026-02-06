"""Step1 kernel: SPH density summation using Poly6 kernel.

Compiles physics/kernels/step1.cu via CuPy RawModule and provides
a function to launch K_Step1 which computes per-particle density
from the sorted particle positions using neighbor iteration.

Density formula
---------------
density_i = mass * kernel_poly6_coeff * SUM_j( (h^2 - |r_ij|^2)^3 )
clamped to max(1.0, ...).

Self-interaction is included (particle i contributes to its own density).
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
    """Compile (or return cached) CuPy RawModule from step1.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "step1.cu")
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


def upload_grid_params(grid_params: np.ndarray) -> None:
    """Upload GridParams to ``__constant__ GridParams c_grid``."""
    module = _get_module()
    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), grid_params.ctypes.data, grid_params.nbytes, 1
    )


def upload_fluid_params(params: np.ndarray) -> None:
    """Upload FluidParams to ``__constant__ FluidParams_Step1 c_fluid``."""
    module = _get_module()
    d_ptr = module.get_global("c_fluid")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


def upload_precalc_params(params: np.ndarray) -> None:
    """Upload PrecalcParams to ``__constant__ PrecalcParams_Step1 c_precalc``."""
    module = _get_module()
    d_ptr = module.get_global("c_precalc")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

BLOCK_SIZE = 128


def compute_step1(
    position: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
) -> cupy.ndarray:
    """Launch K_Step1 and return density array.

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Sorted particle positions.
    cell_start : cupy.ndarray, (num_cells,) uint32
        Grid cell start indices (0xFFFFFFFF for empty).
    cell_end : cupy.ndarray, (num_cells,) uint32
        Grid cell end indices.

    Returns
    -------
    density : cupy.ndarray, (N,) float32
        Per-particle density (clamped >= 1.0).
    """
    n = position.shape[0]
    density = cupy.empty(n, dtype=cupy.float32)

    module = _get_module()
    kernel = module.get_function("K_Step1")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            position,
            cell_start,
            cell_end,
            density,
        ),
    )

    return density
