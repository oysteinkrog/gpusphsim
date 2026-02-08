"""Implicit surface tension via iterative velocity smoothing.

Quality mode for < 100K FLUID particles. Uses Jacobi iterations to smooth
surface particle velocities, producing cohesive fluid surfaces.

Compiles physics/kernels/implicit_st.cu via CuPy RawModule.

Constant memory:
  c_grid, c_sim, c_precalc, c_materials -- shared with other kernels
  c_ist -- ISTParams (sigma, surface_threshold, num_iterations)
"""

from __future__ import annotations

import os
from typing import Optional

import cupy
import numpy as np

# ---------------------------------------------------------------------------
# ISTParams dtype matching struct in implicit_st.cu
# ---------------------------------------------------------------------------

IST_PARAMS_DTYPE = np.dtype(
    [
        ("sigma", np.float32),
        ("surface_threshold", np.float32),
        ("num_iterations", np.int32),
        ("padding", np.float32),
    ],
    align=False,
)

assert IST_PARAMS_DTYPE.itemsize == 16

# ---------------------------------------------------------------------------
# Module compilation
# ---------------------------------------------------------------------------

_module: Optional[object] = None


def _ensure_ptx_if_needed() -> None:
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
    global _module
    if _module is not None:
        return _module
    _ensure_ptx_if_needed()
    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "implicit_st.cu")
    with open(cu_path, encoding="utf-8") as f:
        source = f.read()
    _module = cupy.RawModule(
        code=source,
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )
    return _module


# ---------------------------------------------------------------------------
# Constant memory uploads
# ---------------------------------------------------------------------------

def upload_grid_params(grid_params: np.ndarray) -> None:
    module = _get_module()
    d_ptr = module.get_global("c_grid")
    cupy.cuda.runtime.memcpy(int(d_ptr), grid_params.ctypes.data, grid_params.nbytes, 1)


def upload_sim_params(params: np.ndarray) -> None:
    module = _get_module()
    d_ptr = module.get_global("c_sim")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


def upload_precalc_params(params: np.ndarray) -> None:
    module = _get_module()
    d_ptr = module.get_global("c_precalc")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


def upload_materials(materials_data: np.ndarray) -> None:
    module = _get_module()
    d_ptr = module.get_global("c_materials")
    cupy.cuda.runtime.memcpy(int(d_ptr), materials_data.ctypes.data, materials_data.nbytes, 1)


def upload_ist_params(
    sigma: float = 0.5,
    surface_threshold: float = 25.0,
    num_iterations: int = 5,
) -> None:
    """Upload ISTParams to constant memory."""
    params = np.zeros(1, dtype=IST_PARAMS_DTYPE)
    params[0]["sigma"] = sigma
    params[0]["surface_threshold"] = surface_threshold
    params[0]["num_iterations"] = num_iterations
    params[0]["padding"] = 0.0
    module = _get_module()
    d_ptr = module.get_global("c_ist")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def run_implicit_st(
    velocity: cupy.ndarray,
    velocity_scratch: cupy.ndarray,
    position: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    normal: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    num_iterations: int = 5,
) -> cupy.ndarray:
    """Run implicit surface tension iterations.

    Uses ping-pong between velocity and velocity_scratch.
    Returns the array containing the final result (may be either buffer).

    Parameters
    ----------
    velocity : cupy.ndarray, (N, 4)
        Current sorted velocity (modified in-place if even iterations).
    velocity_scratch : cupy.ndarray, (N, 4)
        Scratch buffer for ping-pong.
    position, density, mass, packed_info, normal : sorted particle data
    cell_start, cell_end : grid arrays
    num_iterations : int
        Number of Jacobi iterations.

    Returns
    -------
    cupy.ndarray
        The array containing final velocities (velocity or velocity_scratch).
    """
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_IST_Iterate")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    src = velocity
    dst = velocity_scratch

    for _ in range(num_iterations):
        kernel(grid, block, (
            np.uint32(n), src, dst,
            position, density, mass, packed_info, normal,
            cell_start, cell_end,
        ))
        # Swap for next iteration
        src, dst = dst, src

    # After the loop, result is in src (last write was to the previous dst,
    # which is now src after the swap)
    return src
