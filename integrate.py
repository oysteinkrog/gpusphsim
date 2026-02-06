"""Integrate kernel: leapfrog integration with wall boundaries and coloring.

Compiles physics/kernels/integrate.cu via CuPy RawModule and provides
a function to launch K_Integrate which performs:
  1. Leapfrog velocity/position update
  2. Wall boundary penalty forces
  3. Velocity-based particle coloring
  4. Writeback to unsorted arrays via sort_indexes permutation

Ported from SPHSimLib/K_SimpleSPH_Integrate.inl.
"""

from __future__ import annotations

import os
from typing import Optional

import cupy  # type: ignore[import-untyped]
import numpy as np

# ---------------------------------------------------------------------------
# IntegrateParams dtype (matches C struct in integrate.cu)
# ---------------------------------------------------------------------------

INTEGRATE_PARAMS_DTYPE = np.dtype(
    [
        ("delta_time", np.float32),
        ("gravity", np.float32),
        ("boundary_stiffness", np.float32),
        ("boundary_dampening", np.float32),
        ("boundary_distance", np.float32),
        ("velocity_limit", np.float32),
    ],
    align=True,
)


def build_integrate_params(
    delta_time: float = 0.001,
    gravity: float = -9.8,
    boundary_stiffness: float = 20000.0,
    boundary_dampening: float = 256.0,
    boundary_distance: float = 0.05,
    velocity_limit: float = 200.0,
) -> np.ndarray:
    """Build IntegrateParams struct as a numpy structured array."""
    params = np.zeros(1, dtype=INTEGRATE_PARAMS_DTYPE)
    params[0]["delta_time"] = delta_time
    params[0]["gravity"] = gravity
    params[0]["boundary_stiffness"] = boundary_stiffness
    params[0]["boundary_dampening"] = boundary_dampening
    params[0]["boundary_distance"] = boundary_distance
    params[0]["velocity_limit"] = velocity_limit
    return params


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
    """Compile (or return cached) CuPy RawModule from integrate.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "integrate.cu")
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


def upload_integrate_params(params: Optional[np.ndarray] = None) -> None:
    """Upload IntegrateParams to ``__constant__ IntegrateParams c_integrate``."""
    if params is None:
        params = build_integrate_params()
    module = _get_module()
    d_ptr = module.get_global("c_integrate")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def integrate(
    position_sorted: cupy.ndarray,
    velocity_sorted: cupy.ndarray,
    veleval_sorted: cupy.ndarray,
    sph_force_sorted: cupy.ndarray,
    density_sorted: cupy.ndarray,
    behavior_class_sorted: cupy.ndarray,
    sort_indexes: cupy.ndarray,
    position_out: cupy.ndarray,
    velocity_out: cupy.ndarray,
    veleval_out: cupy.ndarray,
    color_out: cupy.ndarray,
) -> None:
    """Launch K_Integrate.

    Reads from sorted arrays, integrates, and writes back to unsorted arrays
    using sort_indexes permutation.

    Parameters
    ----------
    position_sorted : (N, 4) float32 -- sorted positions
    velocity_sorted : (N, 4) float32 -- sorted half-step velocities
    veleval_sorted : (N, 4) float32 -- sorted evaluation velocities (XSPH-corrected)
    sph_force_sorted : (N, 4) float32 -- sorted SPH forces from Step2
    density_sorted : (N,) float32 -- sorted densities from Step1
    behavior_class_sorted : (N,) int32 -- sorted behavior class per particle
    sort_indexes : (N,) uint32 -- sort_indexes[sorted] = original index
    position_out : (N, 4) float32 -- unsorted position output
    velocity_out : (N, 4) float32 -- unsorted velocity output
    veleval_out : (N, 4) float32 -- unsorted veleval output
    color_out : (N, 4) float32 -- unsorted color output
    """
    n = position_sorted.shape[0]

    module = _get_module()
    kernel = module.get_function("K_Integrate")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            position_sorted,
            velocity_sorted,
            veleval_sorted,
            sph_force_sorted,
            density_sorted,
            behavior_class_sorted,
            sort_indexes,
            position_out,
            velocity_out,
            veleval_out,
            color_out,
        ),
    )
