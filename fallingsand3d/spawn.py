"""Spawn/kill kernel: GPU freelist-based particle spawning for phase transitions.

Compiles physics/kernels/spawn.cu via CuPy RawModule and provides
a function to launch K_SpawnGas which:
  - Scans sorted particles for HAS_SPAWN_FLAG (set by Reactions for boiling water)
  - For each flagged particle, claims N slots from the GPU freelist via atomicSub
  - Writes N steam particles to the claimed slots (mass-conserving)
  - Marks the source water particle as DEAD and adds it to the freelist
  - Clears the SPAWN_GAS flag regardless of success

Operates on SORTED arrays, runs after Reactions and before Step2.

Uses shared constant memory from common.cuh:
  c_sim       -- SimParams
  c_materials -- MaterialProps[32] (for steam color/rest_density)
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
    """Compile (or return cached) CuPy RawModule from spawn.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "spawn.cu")
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
# Freelist allocation
# ---------------------------------------------------------------------------

SPAWN_N = 3  # must match SPAWN_N in spawn.cu


def allocate_freelist(max_particles: int) -> tuple:
    """Allocate GPU freelist arrays.

    Returns
    -------
    dead_indices : cupy.ndarray, (max_particles,) uint32
        Array of dead particle indices (stack).
    dead_count : cupy.ndarray, (1,) uint32
        Atomic counter for number of entries in dead_indices.
    """
    dead_indices = cupy.zeros(max_particles, dtype=cupy.uint32)
    dead_count = cupy.zeros(1, dtype=cupy.uint32)
    return dead_indices, dead_count


def reset_freelist(dead_count: cupy.ndarray) -> None:
    """Reset the freelist counter to 0 (async memset: graph-capture safe)."""
    dead_count.data.memset_async(0x00, dead_count.nbytes)


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def _ptr(arr: cupy.ndarray | None):
    """Array, or a NULL pointer for None."""
    return arr if arr is not None else np.intp(0)


def compute_spawn(
    sorted_packed_info: cupy.ndarray,
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_veleval: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_lifetime: cupy.ndarray,
    sorted_color: cupy.ndarray,
    sorted_sleep_counter: cupy.ndarray,
    sorted_density: cupy.ndarray,
    sorted_shear_rate: cupy.ndarray,
    dead_indices: cupy.ndarray,
    dead_count: cupy.ndarray,
    max_displacement: cupy.ndarray | None = None,
    *,
    predicted_position: cupy.ndarray | None = None,
    velocity_h: cupy.ndarray | None = None,
    pressure: cupy.ndarray | None = None,
    particle_dye: cupy.ndarray | None = None,
    angular_velocity: cupy.ndarray | None = None,
    kappa: cupy.ndarray | None = None,
    kappa_v: cupy.ndarray | None = None,
    lambda_pbf: cupy.ndarray | None = None,
) -> None:
    """Launch K_SpawnGas kernel.

    Processes all particles with HAS_SPAWN_FLAG set, spawning steam particles
    at freelist slots and marking source particles as DEAD.

    All sorted arrays are modified in-place.

    Parameters
    ----------
    sorted_packed_info : cupy.ndarray, (N,) uint32
    sorted_position : cupy.ndarray, (N, 4) float32
    sorted_velocity : cupy.ndarray, (N, 4) float32
    sorted_veleval : cupy.ndarray, (N, 4) float32
    sorted_mass : cupy.ndarray, (N,) float32
    sorted_temperature : cupy.ndarray, (N,) float32
    sorted_health : cupy.ndarray, (N,) float32
    sorted_lifetime : cupy.ndarray, (N,) float32
    sorted_color : cupy.ndarray, (N, 4) float32
    sorted_sleep_counter : cupy.ndarray, (N,) uint8
    sorted_density : cupy.ndarray, (N,) float32
    sorted_shear_rate : cupy.ndarray, (N,) float32
    dead_indices : cupy.ndarray, (max_particles,) uint32
        Freelist of dead particle indices.
    dead_count : cupy.ndarray, (1,) uint32
        Atomic counter for freelist size.
    max_displacement : cupy.ndarray, (1,) uint32, optional
        Grid reuse tracker. When given, a successful spawn writes the bits of
        +inf into it, so the next grid-reuse decision forces a full sort.
        None passes NULL and the kernel skips the write.
    predicted_position : cupy.ndarray, (N, 4) float32, optional
        PBF sorted_predicted_position. Each child gets pos + dt * vel so
        K_PBF_Finalize does not move it to the dead slot's old position.
        Pass it for PBF only: DFSPH reuses this buffer as a_press scratch.
    velocity_h : cupy.ndarray, (N, 2) uint32, optional
        FP16 velocity mirror that WCSPH step2 reads for neighbours.
    pressure : cupy.ndarray, (N,) float32, optional
        WCSPH pre-computed pressure; children get 0 (gas at rest density).
    particle_dye : cupy.ndarray, (N, 4) float32, optional
    kappa, kappa_v, lambda_pbf : cupy.ndarray, (N,) float32, optional
        Per-slot state the child must not inherit from the dead occupant.
        Each is zeroed for the claimed slots.

    angular_velocity : ignored. The micropolar angular velocity had no
        reader and was removed (bd-r5-review-n28.140); the keyword stays so
        callers that still pass it work.

    Any optional array left as None passes NULL. Keep the pointer set fixed
    per solver so a captured CUDA graph stays valid.
    """
    n = sorted_packed_info.shape[0]
    if n == 0:
        return

    module = _get_module()
    kernel = module.get_function("K_SpawnGas")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            sorted_packed_info,
            sorted_position,
            sorted_velocity,
            sorted_veleval,
            sorted_mass,
            sorted_temperature,
            sorted_health,
            sorted_lifetime,
            sorted_color,
            sorted_sleep_counter,
            sorted_density,
            sorted_shear_rate,
            dead_indices,
            dead_count,
            _ptr(max_displacement),
            _ptr(predicted_position),
            _ptr(velocity_h),
            _ptr(pressure),
            _ptr(particle_dye),
            _ptr(kappa),
            _ptr(kappa_v),
            _ptr(lambda_pbf),
        ),
    )
