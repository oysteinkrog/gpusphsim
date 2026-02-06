"""Integrate kernel: symplectic Euler with SDF boundaries and color update.

Compiles physics/kernels/integrate.cu via CuPy RawModule and provides
a function to launch K_Integrate which performs:
  - Symplectic Euler velocity/position integration
  - Impulse-style SDF box boundary collisions (restitution + Coulomb friction)
  - GAS buoyancy and drag
  - Velocity magnitude clamping
  - Particle color computation from material, temperature, and health
  - Writeback to UNSORTED arrays via sort_indexes permutation

Uses shared constant memory from common.cuh:
  c_sim       -- SimParams (gravity, dt, restitution, wall_friction, world bounds)
  c_materials -- MaterialProps[32] (for color lookup)
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


def upload_sim_params(params: np.ndarray) -> None:
    """Upload SimParams to ``__constant__ SimParams c_sim``."""
    module = _get_module()
    d_ptr = module.get_global("c_sim")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


def upload_materials(materials_data: np.ndarray) -> None:
    """Upload MaterialProps[32] to ``__constant__ MaterialProps c_materials[32]``."""
    module = _get_module()
    d_ptr = module.get_global("c_materials")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), materials_data.ctypes.data, materials_data.nbytes, 1
    )


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def integrate(
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_veleval: cupy.ndarray,
    sorted_sph_force: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sort_indexes: cupy.ndarray,
    position_out: "Optional[cupy.ndarray]" = None,
    velocity_out: "Optional[cupy.ndarray]" = None,
    color_out: "Optional[cupy.ndarray]" = None,
) -> tuple:
    """Launch K_Integrate and return (position_out, velocity_out, color_out).

    All sorted_* inputs are in sorted (grid) order.
    Outputs are written to UNSORTED arrays via sort_indexes[i] mapping.

    Parameters
    ----------
    sorted_position : cupy.ndarray, (N, 4) float32
    sorted_velocity : cupy.ndarray, (N, 4) float32
    sorted_veleval : cupy.ndarray, (N, 4) float32
        XSPH-corrected veleval for FLUID; original velocity for others.
    sorted_sph_force : cupy.ndarray, (N, 4) float32
    sorted_mass : cupy.ndarray, (N,) float32
    sorted_packed_info : cupy.ndarray, (N,) uint32
    sorted_temperature : cupy.ndarray, (N,) float32
    sorted_health : cupy.ndarray, (N,) float32
    sort_indexes : cupy.ndarray, (N,) uint32
        sort_indexes[sorted_i] = original unsorted index.
    position_out : cupy.ndarray, optional
        Pre-allocated (M, 4) float32 unsorted output.
    velocity_out : cupy.ndarray, optional
        Pre-allocated (M, 4) float32 unsorted output.
    color_out : cupy.ndarray, optional
        Pre-allocated (M, 4) float32 unsorted output.

    Returns
    -------
    position_out, velocity_out, color_out : cupy.ndarray
    """
    n = sorted_position.shape[0]
    if n == 0:
        return position_out, velocity_out, color_out

    # Allocate outputs if not provided
    if position_out is None:
        max_idx = int(sort_indexes.max()) + 1 if n > 0 else n
        position_out = cupy.zeros((max_idx, 4), dtype=cupy.float32)
    if velocity_out is None:
        max_idx = int(sort_indexes.max()) + 1 if n > 0 else n
        velocity_out = cupy.zeros((max_idx, 4), dtype=cupy.float32)
    if color_out is None:
        max_idx = int(sort_indexes.max()) + 1 if n > 0 else n
        color_out = cupy.zeros((max_idx, 4), dtype=cupy.float32)

    module = _get_module()
    kernel = module.get_function("K_Integrate")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            sorted_position,
            sorted_velocity,
            sorted_veleval,
            sorted_sph_force,
            sorted_mass,
            sorted_packed_info,
            sorted_temperature,
            sorted_health,
            sort_indexes,
            position_out,
            velocity_out,
            color_out,
        ),
    )

    return position_out, velocity_out, color_out
