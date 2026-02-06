"""Reactions kernel: per-particle phase transitions, combustion, and corrosion.

Compiles physics/kernels/reactions.cu via CuPy RawModule and provides
a function to launch K_Reactions which performs:
  - ICE -> WATER (temp > 273K)
  - LAVA -> STONE (temp < 900K)
  - WATER -> SPAWN_GAS flag (temp > 373K)
  - STEAM -> WATER (temp < 373K)
  - WOOD/OIL/GUNPOWDER -> FIRE (exposure_heat threshold)
  - Corrosion: health -= exposure_corrode * dt
  - GAS lifetime decay -> DEAD when expired

Operates on SORTED arrays, runs after Step1 (which computes exposure)
and before Step2 (which computes forces).

Uses shared constant memory from common.cuh:
  c_sim       -- SimParams (dt)
  c_materials -- MaterialProps[32]
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
    """Compile (or return cached) CuPy RawModule from reactions.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "reactions.cu")
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


def compute_reactions(
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_lifetime: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_exposure_heat: cupy.ndarray,
    sorted_exposure_corrode: cupy.ndarray,
    frame: int = 0,
) -> None:
    """Launch K_Reactions kernel.

    Modifies sorted arrays in-place: packed_info, temperature, health,
    lifetime, velocity.

    Parameters
    ----------
    sorted_packed_info : cupy.ndarray, (N,) uint32
        Packed info (material_id + behavior + flags). Modified in-place.
    sorted_temperature : cupy.ndarray, (N,) float32
        Particle temperature. Modified in-place.
    sorted_health : cupy.ndarray, (N,) float32
        Particle health [0,1]. Modified in-place.
    sorted_lifetime : cupy.ndarray, (N,) float32
        Remaining lifetime in seconds. Modified in-place.
    sorted_velocity : cupy.ndarray, (N, 4) float32
        Particle velocity. Modified in-place (gunpowder explosion).
    sorted_exposure_heat : cupy.ndarray, (N,) float32
        Heat exposure from Step1 (read-only).
    sorted_exposure_corrode : cupy.ndarray, (N,) float32
        Corrosion exposure from Step1 (read-only).
    frame : int
        Frame counter for RNG seed (gunpowder explosion direction).
    """
    n = sorted_packed_info.shape[0]
    if n == 0:
        return

    module = _get_module()
    kernel = module.get_function("K_Reactions")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            np.uint32(frame),
            sorted_packed_info,
            sorted_temperature,
            sorted_health,
            sorted_lifetime,
            sorted_velocity,
            sorted_exposure_heat,
            sorted_exposure_corrode,
        ),
    )
