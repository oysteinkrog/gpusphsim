"""Foam/Spray/Bubble secondary particle system.

Compiles physics/kernels/foam.cu via CuPy RawModule and provides
functions to launch foam generation, physics, and compaction kernels.

Foam particles are separate from SPH particles -- they have no neighbor
loops and use simple ballistic/advection physics. Generated from FLUID
particles based on velocity, surface proximity, and kinetic energy.
"""

from __future__ import annotations

import os
from typing import Optional

import cupy
import numpy as np


# ---------------------------------------------------------------------------
# FoamParams dtype -- matches struct FoamParams in foam.cu
# ---------------------------------------------------------------------------

FOAM_PARAMS_DTYPE = np.dtype(
    [
        ("k_ta", np.float32),
        ("k_wc", np.float32),
        ("k_ke", np.float32),
        ("threshold", np.float32),
        ("spray_lifetime", np.float32),
        ("foam_lifetime", np.float32),
        ("bubble_lifetime", np.float32),
        ("drag_coeff", np.float32),
        ("buoyancy", np.float32),
        ("diffusion", np.float32),
        ("spawn_jitter", np.float32),
        ("max_foam", np.int32),
        ("dt", np.float32),
        ("_pad0", np.float32),
    ],
    align=False,
)


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

def build_foam_params(
    k_ta: float = 0.5,
    k_wc: float = 1.0,
    k_ke: float = 0.2,
    threshold: float = 2.0,
    spray_lifetime: float = 1.0,
    foam_lifetime: float = 2.0,
    bubble_lifetime: float = 1.5,
    drag_coeff: float = 2.0,
    buoyancy: float = 15.0,
    diffusion: float = 0.5,
    spawn_jitter: float = 0.01,
    max_foam: int = 200_000,
    dt: float = 0.001,
) -> np.ndarray:
    """Build FoamParams numpy struct for constant memory upload."""
    params = np.zeros(1, dtype=FOAM_PARAMS_DTYPE)
    params[0]["k_ta"] = np.float32(k_ta)
    params[0]["k_wc"] = np.float32(k_wc)
    params[0]["k_ke"] = np.float32(k_ke)
    params[0]["threshold"] = np.float32(threshold)
    params[0]["spray_lifetime"] = np.float32(spray_lifetime)
    params[0]["foam_lifetime"] = np.float32(foam_lifetime)
    params[0]["bubble_lifetime"] = np.float32(bubble_lifetime)
    params[0]["drag_coeff"] = np.float32(drag_coeff)
    params[0]["buoyancy"] = np.float32(buoyancy)
    params[0]["diffusion"] = np.float32(diffusion)
    params[0]["spawn_jitter"] = np.float32(spawn_jitter)
    params[0]["max_foam"] = np.int32(max_foam)
    params[0]["dt"] = np.float32(dt)
    return params


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
    """Compile (or return cached) CuPy RawModule from foam.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "foam.cu")
    with open(cu_path) as f:
        source = f.read()

    _module = cupy.RawModule(
        code=source,
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )
    return _module


# ---------------------------------------------------------------------------
# Constant memory upload
# ---------------------------------------------------------------------------

def upload_foam_params(params: np.ndarray) -> None:
    """Upload FoamParams to ``__constant__ FoamParams c_foam``."""
    module = _get_module()
    d_ptr = module.get_global("c_foam")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


def upload_sim_params(params: np.ndarray) -> None:
    """Upload SimParams to ``__constant__ SimParams c_sim``."""
    module = _get_module()
    d_ptr = module.get_global("c_sim")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


# ---------------------------------------------------------------------------
# Kernel launch wrappers
# ---------------------------------------------------------------------------

_BLOCK = 256


def foam_generate(
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_normal: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    foam_position: cupy.ndarray,
    foam_velocity: cupy.ndarray,
    foam_count: cupy.ndarray,
    num_particles: int,
    frame_seed: int,
) -> None:
    """Launch K_FoamGenerate: emit secondary particles from FLUID particles."""
    module = _get_module()
    kernel = module.get_function("K_FoamGenerate")
    grid = ((num_particles + _BLOCK - 1) // _BLOCK,)
    kernel(
        grid,
        (_BLOCK,),
        (
            sorted_position,
            sorted_velocity,
            sorted_normal,
            sorted_packed_info,
            foam_position,
            foam_velocity,
            foam_count,
            np.int32(num_particles),
            np.uint32(frame_seed),
        ),
    )


# Minimum grid launch size for K_FoamPhysics to avoid pathologically tiny grids.
_MIN_FOAM_GRID_SIZE = 1024


def foam_physics(
    foam_position: cupy.ndarray,
    foam_velocity: cupy.ndarray,
    foam_count: cupy.ndarray,
    max_foam: int,
    last_foam_count: int = 0,
) -> None:
    """Launch K_FoamPhysics: simple physics for each foam particle.

    Grid is bounded by max(last_foam_count, _MIN_FOAM_GRID_SIZE) rather than
    always max_foam (bd-mzc.44).  last_foam_count is the 1-frame-deferred CPU
    readback available in Simulation._last_foam_count -- zero on the first frame,
    which triggers the MIN_FOAM_GRID_SIZE fallback.  max_foam is kept as a cap.

    No GPU->CPU sync required: the kernel reads the actual count from the
    foam_count device pointer and early-exits for out-of-bounds threads.

    Parameters
    ----------
    foam_position, foam_velocity : cupy.ndarray
        Foam particle SoA arrays (shape (max_foam, 4) float32).
    foam_count : cupy.ndarray
        Device scalar (shape (1,) uint32) with the live foam particle count.
    max_foam : int
        Hard upper bound on foam capacity (used as grid size when
        last_foam_count is 0 or exceeds it, and for safety clamping).
    last_foam_count : int
        1-frame-deferred foam count from async CPU readback.  0 means unknown
        (first frame); in that case the grid falls back to _MIN_FOAM_GRID_SIZE.
    """
    module = _get_module()
    kernel = module.get_function("K_FoamPhysics")
    # Use deferred count to bound the grid; pad up to MIN to avoid tiny launches.
    effective = max(int(last_foam_count), _MIN_FOAM_GRID_SIZE)
    # Never exceed the actual capacity.
    effective = min(effective, max_foam)
    grid = ((effective + _BLOCK - 1) // _BLOCK,)
    kernel(
        grid,
        (_BLOCK,),
        (
            foam_position,
            foam_velocity,
            foam_count,
        ),
    )


def foam_compact(
    foam_position_in: cupy.ndarray,
    foam_velocity_in: cupy.ndarray,
    foam_position_out: cupy.ndarray,
    foam_velocity_out: cupy.ndarray,
    alive_count: cupy.ndarray,
    foam_count: cupy.ndarray,
    max_foam: int,
) -> None:
    """Launch K_FoamCompact: remove dead foam particles.

    After this, alive_count[0] has the new count.
    Caller should swap in/out buffers and copy alive_count to foam_count.
    """
    # Reset alive counter
    alive_count.fill(0)
    module = _get_module()
    kernel = module.get_function("K_FoamCompact")
    grid = ((max_foam + _BLOCK - 1) // _BLOCK,)
    kernel(
        grid,
        (_BLOCK,),
        (
            foam_position_in,
            foam_velocity_in,
            foam_position_out,
            foam_velocity_out,
            alive_count,
            foam_count,
        ),
    )
