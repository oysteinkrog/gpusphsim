"""Foam/Spray/Bubble secondary particle system.

Compiles physics/kernels/foam.cu via CuPy RawModule and provides
functions to launch foam generation, physics, and compaction kernels.

Foam particles are separate from SPH particles -- they have no neighbor
loops and use simple ballistic/advection physics. Generated from FLUID
particles based on velocity, surface proximity, and kinetic energy.

Constant memory: c_foam, c_sim and c_grid are uploaded here (c_grid lets
K_FoamPhysics find its own cell in the SPH hash grid). c_sdf_objects and
c_num_sdf_objects (used by K_FoamPhysics for SDF object collision) are
uploaded by rigid_bodies.upload_sdf_objects, with the same data as the
integrate and solver modules.
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
        ("k_rate", np.float32),
        ("foam_buoyancy", np.float32),
        ("foam_water_drag", np.float32),
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
    k_rate: float = 5.0,
    foam_buoyancy: float = 1.5,
    foam_water_drag: float = 10.0,
) -> np.ndarray:
    """Build FoamParams numpy struct for constant memory upload.

    k_rate is the emission rate per unit of (phi - threshold), in 1/s. The
    default 5.0 makes a surface particle with phi = 4 emit about 10 foam
    particles per second.

    foam_buoyancy (m/s^2) and foam_water_drag (1/s) act on FOAM inside the
    fluid. The defaults give a rise speed of 0.15 m/s, slow enough that
    foam leaving the top cell overshoots by about 1 cm under 0.1 g.
    """
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
    params[0]["k_rate"] = np.float32(k_rate)
    params[0]["foam_buoyancy"] = np.float32(foam_buoyancy)
    params[0]["foam_water_drag"] = np.float32(foam_water_drag)
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
    # K_FoamPhysics reads c_sdf_objects / c_num_sdf_objects. They are sent
    # by rigid_bodies.upload_sdf_objects (SDFManager.upload_if_dirty), which
    # includes this module once it exists; replay the last upload in case
    # the objects were placed before this compile.
    import rigid_bodies
    rigid_bodies.replay_sdf_upload(_module)
    return _module


# ---------------------------------------------------------------------------
# Constant memory upload
# ---------------------------------------------------------------------------

def upload_foam_params(params: np.ndarray) -> None:
    """Upload FoamParams to ``__constant__ FoamParams c_foam``."""
    module = _get_module()
    d_ptr = module.get_global("c_foam")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


def upload_grid_params(params: np.ndarray) -> None:
    """Upload GridParams to ``__constant__ GridParams c_grid``."""
    module = _get_module()
    d_ptr = module.get_global("c_grid")
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
_NULL = cupy.ndarray(0, dtype=cupy.uint32)  # NULL pointer for an optional argument


def foam_generate(
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_normal: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    foam_position: cupy.ndarray,
    foam_velocity: cupy.ndarray,
    foam_count: cupy.ndarray,
    num_particles: int,
    frame_ptr: cupy.ndarray,
    dt_emit: float,
    sorted_velocity_h: Optional[cupy.ndarray] = None,
) -> None:
    """Launch K_FoamGenerate: emit secondary particles from FLUID particles.

    dt_emit is the sim time this launch covers. Each qualifying particle
    emits with probability min(1, k_rate * (phi - threshold) * dt_emit).
    It is a kernel argument, not a c_foam field, so a per-frame value needs
    no constant-memory upload.

    frame_ptr is a device uint32 array whose element 0 is the RNG seed
    (Simulation passes its per-substep counter, advanced on the device by
    increment_counter). The seed is not a host scalar, so the launch works
    the same inside a CUDA graph.

    sorted_velocity_h is the half4 velocity mirror ((N, 2) uint32). When
    given, the kernel reads velocity from it (8 B per particle instead of
    16 B). Pass None when the mirror is not kept current (PBF skips it);
    the kernel then reads sorted_velocity.
    """
    module = _get_module()
    kernel = module.get_function("K_FoamGenerate")
    grid = ((num_particles + _BLOCK - 1) // _BLOCK,)
    kernel(
        grid,
        (_BLOCK,),
        (
            sorted_position,
            sorted_velocity,
            sorted_velocity_h if sorted_velocity_h is not None else _NULL,
            sorted_normal,
            sorted_packed_info,
            foam_position,
            foam_velocity,
            foam_count,
            np.int32(num_particles),
            frame_ptr,
            np.float32(dt_emit),
        ),
    )


def foam_physics(
    foam_position: cupy.ndarray,
    foam_velocity: cupy.ndarray,
    foam_count: cupy.ndarray,
    max_foam: int,
    cell_start: Optional[cupy.ndarray] = None,
    cell_end: Optional[cupy.ndarray] = None,
) -> None:
    """Launch K_FoamPhysics: simple physics for each foam particle.

    cell_start and cell_end are the SPH hash grid tables built this substep
    (Simulation passes its _cell_start and _cell_end). A foam particle
    whose own cell is non-empty counts as inside fluid: FOAM there rises
    back to the surface and a BUBBLE outside fluid turns into FOAM. Pass
    None for both to run without fluid information (BUBBLE always rises,
    FOAM always falls). The foam module needs c_grid for this
    (upload_grid_params).

    The grid covers the whole pool (max_foam threads). The kernel reads the
    live count from the foam_count device pointer and threads past it exit at
    once, so the full launch costs a few microseconds. A grid sized from the
    deferred host count missed foam made in the frames since that readback
    (bd-r5-review-n28.146).

    No GPU->CPU sync required.

    Parameters
    ----------
    foam_position, foam_velocity : cupy.ndarray
        Foam particle SoA arrays (shape (max_foam, 4) float32).
    foam_count : cupy.ndarray
        Device scalar (shape (1,) uint32) with the live foam particle count.
    max_foam : int
        Pool size, used as the grid size.
    """
    module = _get_module()
    kernel = module.get_function("K_FoamPhysics")
    grid = ((max_foam + _BLOCK - 1) // _BLOCK,)
    kernel(
        grid,
        (_BLOCK,),
        (
            foam_position,
            foam_velocity,
            foam_count,
            cell_start if cell_start is not None else _NULL,
            cell_end if cell_end is not None else _NULL,
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

    After this, alive_count[0] has the new count and every slot of
    foam_position_out from alive_count[0] up is marked dead (w = -1), so the
    renderer never draws stale positions from an earlier cycle.
    Caller should swap in/out buffers and copy alive_count to foam_count.
    """
    # Reset alive counter (a 4-byte memset, not an elementwise kernel)
    alive_count.data.memset_async(0, 4)
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
    clear = module.get_function("K_FoamClearTail")
    clear(grid, (_BLOCK,), (foam_position_out, alive_count, np.int32(max_foam)))


def increment_counter(counter: cupy.ndarray) -> None:
    """Launch K_IncrementCounter: counter[0] += 1 on the device (1 thread).

    counter is a (1,) uint32 device array. The launch takes no host value,
    so it can be a node of a captured CUDA graph.
    """
    module = _get_module()
    kernel = module.get_function("K_IncrementCounter")
    kernel((1,), (1,), (counter,))
