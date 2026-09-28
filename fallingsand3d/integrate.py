"""Integrate kernel: symplectic Euler with SDF boundaries, and render color.

Compiles physics/kernels/integrate.cu via CuPy RawModule and provides
a function to launch K_Integrate which performs:
  - Symplectic Euler velocity/position integration
  - Impulse-style SDF box boundary collisions (restitution + Coulomb friction)
  - GAS buoyancy and drag
  - Velocity magnitude clamping
  - Writeback to UNSORTED arrays via sort_indexes permutation, or in place
    to the sorted arrays on a grid-reuse substep (in_place=True)

and K_ComputeColor, which computes the render color once per frame from the
unsorted arrays (compute_color below).

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


def upload_grid_params(params: np.ndarray) -> None:
    """Upload GridParams to ``__constant__ GridParams c_grid``.

    K_Integrate maps positions to grid cells for the STATIC push-out and the
    own-cell wake mark (bd-r5-review-n28.166). Without this upload c_grid is
    all zero and every particle maps to hash 0.
    """
    module = _get_module()
    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


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
    sorted_lifetime: "Optional[cupy.ndarray]" = None,
    sorted_density: "Optional[cupy.ndarray]" = None,
    sorted_shear_rate: "Optional[cupy.ndarray]" = None,
    sorted_dTdt: "Optional[cupy.ndarray]" = None,
    sorted_sleep_counter: "Optional[cupy.ndarray]" = None,
    sorted_vorticity: "Optional[cupy.ndarray]" = None,
    sorted_angular_velocity: "Optional[cupy.ndarray]" = None,
    sort_indexes: "Optional[cupy.ndarray]" = None,
    position_out: "Optional[cupy.ndarray]" = None,
    velocity_out: "Optional[cupy.ndarray]" = None,
    packed_info_out: "Optional[cupy.ndarray]" = None,
    sleep_counter_out: "Optional[cupy.ndarray]" = None,
    temperature_out: "Optional[cupy.ndarray]" = None,
    angular_velocity_out: "Optional[cupy.ndarray]" = None,
    health_out: "Optional[cupy.ndarray]" = None,
    lifetime_out: "Optional[cupy.ndarray]" = None,
    mass_out: "Optional[cupy.ndarray]" = None,
    max_displacement: "Optional[cupy.ndarray]" = None,
    displacement_anchor: "Optional[cupy.ndarray]" = None,
    cell_start: "Optional[cupy.ndarray]" = None,
    cell_end: "Optional[cupy.ndarray]" = None,
    cell_wake_flags: "Optional[cupy.ndarray]" = None,
    frame_counter: "Optional[cupy.ndarray]" = None,
    in_place: bool = False,
    after_in_place: bool = False,
    velocity_h_out: "Optional[cupy.ndarray]" = None,
    temperature_h_out: "Optional[cupy.ndarray]" = None,
) -> tuple:
    """Launch K_Integrate and return outputs tuple.

    All sorted_* inputs are in sorted (grid) order.
    Outputs are written to UNSORTED arrays via sort_indexes[i] mapping.

    Parameters
    ----------
    sorted_position : cupy.ndarray, (N, 4) float32
    sorted_velocity : cupy.ndarray, (N, 4) float32
    sorted_veleval : cupy.ndarray, (N, 4) float32
        XSPH-corrected veleval. Read for awake FLUID only; other rows may
        hold stale data (step2 writes FLUID rows only).
    sorted_sph_force : cupy.ndarray, (N, 4) float32
        Read for awake particles only. A particle that wakes in this kernel
        uses zero force, since step2 skipped it.
    sorted_mass : cupy.ndarray, (N,) float32
    sorted_packed_info : cupy.ndarray, (N,) uint32
    sorted_temperature : cupy.ndarray, (N,) float32
    sorted_health : cupy.ndarray, (N,) float32
    sorted_density : cupy.ndarray, optional, (N,) float32
        Density from Step1. Used for GRANULAR anti-creep check.
        If None, a zeros array is used (anti-creep won't trigger).
    sorted_shear_rate : cupy.ndarray, optional, (N,) float32
        Shear rate from Step1. Used for GRANULAR anti-creep and sleep check.
        If None, a zeros array is used.
    sorted_dTdt : cupy.ndarray, optional, (N,) float32
        Temperature rate of change from Step1 heat diffusion.
        If None, a zeros array is used (no heat diffusion).
    sorted_sleep_counter : cupy.ndarray, optional, (N,) uint8
        Sleep counter from previous frame. If None, zeros are used.
    sort_indexes : cupy.ndarray, (N,) uint32
        sort_indexes[sorted_i] = original unsorted index.
    position_out : cupy.ndarray, optional
        Pre-allocated (N, 4) float32 unsorted output.
    velocity_out : cupy.ndarray, optional
        Pre-allocated (N, 4) float32 unsorted output.
    packed_info_out : cupy.ndarray, optional
        Pre-allocated (N,) uint32 unsorted output for updated packed_info.
    sleep_counter_out : cupy.ndarray, optional
        Pre-allocated (N,) uint8 unsorted output for updated sleep counter.
    temperature_out : cupy.ndarray, optional
        Pre-allocated (N,) float32 unsorted output for updated temperature.
    cell_wake_flags : cupy.ndarray, optional, (table_size,) uint32
        Wake stamp table. When given, each just-woke or fast awake particle
        (not STATIC or GAS, |v| > V_WAKE) stores the stamp in its own grid
        cell, the same test as K_MarkWakeCells, so the caller runs
        run_wake_propagation with mark=False (bd-r5-review-n28.166).
    frame_counter : cupy.ndarray, optional, (1,) uint32
        Per-substep device counter; the stamp is its value + 1. None gives
        stamp 1.

    in_place : bool
        Write every output to the sorted input arrays at slot i instead of
        through sort_indexes (bd-r5-review-n28.196). The *_out arguments
        are ignored and the sorted arrays are returned. Use it only when the
        next substep reuses the grid, so it reads the sorted arrays with no
        gather. health, lifetime and mass stay in sorted space with their
        state-dirty bit set. The STATIC push-out may run: it reads only
        STATIC slots, which K_Integrate rewrites with the same position,
        and behavior and material bits, which it never changes.
    after_in_place : bool
        The unsorted outputs are older than the sorted inputs because the
        previous substeps ran in place. Makes the scatter store every
        sleeper's zero velocity. Ignored with in_place.
    velocity_h_out, temperature_h_out : cupy.ndarray, optional
        With in_place, the sorted FP16 velocity (N, 2) uint32 and fixed-point
        temperature (N,) uint16 mirrors to rewrite, since no gather rebuilds
        them. None skips that mirror.

    sorted_vorticity, sorted_angular_velocity, angular_velocity_out
        Ignored. The micropolar angular velocity had no reader and was
        removed (bd-r5-review-n28.140). The parameters stay only so callers
        that still pass them keep working.

    Returns
    -------
    (position_out, velocity_out, packed_info_out,
     sleep_counter_out, temperature_out)

    Render color is not an output: compute_color() makes it once per frame.
    """
    n = sorted_position.shape[0]
    if n == 0:
        return (position_out, velocity_out, packed_info_out,
                sleep_counter_out, temperature_out)

    # Default density/shear_rate/dTdt/sleep_counter to zeros if not provided
    if sorted_density is None:
        sorted_density = cupy.zeros(n, dtype=cupy.float32)
    if sorted_shear_rate is None:
        sorted_shear_rate = cupy.zeros(n, dtype=cupy.float32)
    if sorted_dTdt is None:
        sorted_dTdt = cupy.zeros(n, dtype=cupy.float32)
    if sorted_sleep_counter is None:
        sorted_sleep_counter = cupy.zeros(n, dtype=cupy.uint8)
    if sorted_lifetime is None:
        sorted_lifetime = cupy.zeros(n, dtype=cupy.float32)

    if in_place:
        position_out = sorted_position
        velocity_out = sorted_velocity
        packed_info_out = sorted_packed_info
        sleep_counter_out = sorted_sleep_counter
        temperature_out = sorted_temperature
        health_out = lifetime_out = mass_out = None
        write_mode = 1
    else:
        write_mode = 2 if after_in_place else 0
        velocity_h_out = temperature_h_out = None
    sort_idx_ptr = sort_indexes if sort_indexes is not None else np.intp(0)
    if sort_indexes is None and not in_place:
        raise ValueError("sort_indexes is required unless in_place")

    # Allocate outputs if not provided.
    # sort_indexes is a permutation of [0, n), so max index = n-1 and output
    # size = n.  Using n directly avoids 8 GPU->CPU syncs (bd-mzc.41).
    max_idx = n
    if position_out is None:
        position_out = cupy.zeros((max_idx, 4), dtype=cupy.float32)
    if velocity_out is None:
        velocity_out = cupy.zeros((max_idx, 4), dtype=cupy.float32)
    if packed_info_out is None:
        packed_info_out = cupy.zeros(max_idx, dtype=cupy.uint32)
    if sleep_counter_out is None:
        sleep_counter_out = cupy.zeros(max_idx, dtype=cupy.uint8)
    if temperature_out is None:
        temperature_out = cupy.zeros(max_idx, dtype=cupy.float32)

    module = _get_module()
    kernel = module.get_function("K_Integrate")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    # Use null pointer (0) when max_displacement not provided
    max_disp_ptr = max_displacement if max_displacement is not None else np.intp(0)
    # Sorted position at the last full sort; NULL measures per-substep motion.
    anchor_ptr = displacement_anchor if displacement_anchor is not None else np.intp(0)
    # Reaction-state carry-back outputs: null pointer when not provided
    # (kernel guards each write with `if (ptr)`).
    health_out_ptr = health_out if health_out is not None else np.intp(0)
    lifetime_out_ptr = lifetime_out if lifetime_out is not None else np.intp(0)
    mass_out_ptr = mass_out if mass_out is not None else np.intp(0)
    # Use null pointer when cell_start/cell_end not provided (disables STATIC repulsion)
    cell_start_ptr = cell_start if cell_start is not None else np.intp(0)
    cell_end_ptr = cell_end if cell_end is not None else np.intp(0)
    # Own-cell wake stamp written from registers (bd-r5-review-n28.166).
    # NULL table: no mark, so the caller must run K_MarkWakeCells instead.
    wake_ptr = cell_wake_flags if cell_wake_flags is not None else np.intp(0)
    frame_ptr = frame_counter if frame_counter is not None else np.intp(0)

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
            sorted_lifetime,
            sorted_density,
            sorted_shear_rate,
            sorted_dTdt,
            sorted_sleep_counter,
            sort_idx_ptr,
            cell_start_ptr,
            cell_end_ptr,
            position_out,
            velocity_out,
            packed_info_out,
            sleep_counter_out,
            temperature_out,
            health_out_ptr,
            lifetime_out_ptr,
            mass_out_ptr,
            max_disp_ptr,
            anchor_ptr,
            wake_ptr,
            frame_ptr,
            np.uint32(write_mode),
            velocity_h_out if velocity_h_out is not None else np.intp(0),
            temperature_h_out if temperature_h_out is not None else np.intp(0),
        ),
    )

    return (position_out, velocity_out, packed_info_out,
            sleep_counter_out, temperature_out)


# ---------------------------------------------------------------------------
# Render color, once per frame (bd-r5-review-n28.145)
# ---------------------------------------------------------------------------

def compute_color(
    n: int,
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    packed_info: cupy.ndarray,
    temperature: cupy.ndarray,
    health: cupy.ndarray,
    color_out: cupy.ndarray,
) -> None:
    """Launch K_ComputeColor over the first ``n`` UNSORTED slots.

    Writes the render color of slot i to ``color_out[i]``: material base
    color with heat glow and health fade, plus the FLUID depth, density and
    speed tint, and ``w`` = behavior * 0.25 (DEAD slots get -1). position.w
    must hold the density, which every finalize kernel writes there.

    ``color_out`` is any (>= n, 4) float32 device array, normally the mapped
    color VBO from Simulation.copy_to_vbos. Uses the integrate module's
    c_sim (world bounds) and c_materials. No host sync.
    """
    if n <= 0:
        return
    kernel = _get_module().get_function("K_ComputeColor")  # type: ignore[union-attr]
    kernel(
        ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,),
        (BLOCK_SIZE,),
        (
            np.uint32(n),
            position,
            velocity,
            packed_info,
            temperature,
            health,
            color_out,
        ),
    )


# ---------------------------------------------------------------------------
# Rigid body integration (US-017)
# ---------------------------------------------------------------------------

def integrate_rigid_bodies(
    d_rigid_bodies: cupy.ndarray,
    d_rigid_forces: cupy.ndarray,
    d_rigid_torques: cupy.ndarray,
    num_bodies: int,
    dt: float,
    gravity: tuple = (0.0, -9.81, 0.0),
) -> None:
    """Launch K_IntegrateRigidBodies: one thread per body.

    Integrates linear/angular velocity from accumulated forces/torques,
    updates quaternion, zeroes accumulators. All in global memory.
    No GPU-CPU sync needed.

    Parameters
    ----------
    d_rigid_bodies : cupy.ndarray
        GPU array of RigidBody structs (global memory, read-write).
    d_rigid_forces : cupy.ndarray, (MAX_RIGID_BODIES, 4) float32
        Accumulated forces from fluid coupling.
    d_rigid_torques : cupy.ndarray, (MAX_RIGID_BODIES, 4) float32
        Accumulated torques from fluid coupling.
    num_bodies : int
        Number of active rigid bodies.
    dt : float
        Simulation timestep.
    gravity : tuple
        Gravity vector (x, y, z).
    """
    if num_bodies <= 0:
        return

    module = _get_module()
    kernel = module.get_function("K_IntegrateRigidBodies")

    grav = np.array([gravity[0], gravity[1], gravity[2]], dtype=np.float32)

    # Single block, one thread per body (max 8 bodies)
    kernel(
        (1,), (num_bodies,),
        (
            d_rigid_bodies,
            d_rigid_forces,
            d_rigid_torques,
            np.int32(num_bodies),
            np.float32(dt),
            grav[0], grav[1], grav[2],
        ),
    )


# ---------------------------------------------------------------------------
# Rigid body collisions (US-020)
# ---------------------------------------------------------------------------

def rigid_body_collisions(
    d_rigid_bodies: cupy.ndarray,
    num_bodies: int,
) -> None:
    """Launch K_RigidBodyCollisions: push-apart for body-SDF and body-body.

    Runs after K_IntegrateRigidBodies, before K_UpdateBoundaryParticles.
    One thread per body, checks bounding sphere vs SDF objects and other bodies.
    """
    if num_bodies <= 0:
        return

    module = _get_module()
    kernel = module.get_function("K_RigidBodyCollisions")

    kernel(
        (1,), (num_bodies,),
        (d_rigid_bodies, np.int32(num_bodies)),
    )


# ---------------------------------------------------------------------------
# Boundary particle state sync (US-018)
# ---------------------------------------------------------------------------

def update_boundary_particles(
    boundary_data: cupy.ndarray,
    boundary_body_id: cupy.ndarray,
    d_rigid_bodies: cupy.ndarray,
    position_out: cupy.ndarray,
    velocity_out: cupy.ndarray,
    mass_out: cupy.ndarray,
    offset: int,
    num_boundary: int,
    anchor: "Optional[cupy.ndarray]" = None,
    max_displacement: "Optional[cupy.ndarray]" = None,
    reset_anchor: bool = False,
) -> None:
    """Launch K_UpdateBoundaryParticles: transform local -> world positions.

    Runs after K_IntegrateRigidBodies, before hash/sort for next substep.
    Sets boundary particle positions and velocities from rigid body state.

    Parameters
    ----------
    boundary_data : cupy.ndarray, (N, 4) float32
        Combined boundary records (x_local, y_local, z_local, psi). The kernel
        reads each row as one float4, so it must be C-contiguous.
    boundary_body_id : cupy.ndarray, (N,) int32
        Parent rigid body index of each boundary particle.
    d_rigid_bodies : cupy.ndarray
        GPU array of RigidBody structs.
    position_out : cupy.ndarray, (M, 4) float32
        Main unsorted position array (written at offset..offset+N).
    velocity_out : cupy.ndarray, (M, 4) float32
        Main unsorted velocity array.
    mass_out : cupy.ndarray, (M,) float32
        Main unsorted mass array.
    offset : int
        Start index in main arrays for boundary particles.
    num_boundary : int
        Number of boundary particles.
    anchor : cupy.ndarray, (N, 4) float32, optional
        Each boundary particle's position at the last full sort. With
        max_displacement it lets grid reuse see rigid body motion
        (bd-r5-review-n28.65). None turns tracking off.
    max_displacement : cupy.ndarray, (1,) uint32, optional
        Grid-reuse drift tracker (world.max_displacement). The kernel
        atomicMaxes the drift from the anchor, extrapolated one substep
        ahead, into it.
    reset_anchor : bool
        True when this substep runs a full sort: the kernel stores the new
        positions as the anchor first. The caller must clear
        max_displacement before this launch.
    """
    if num_boundary <= 0:
        return
    track = anchor is not None and max_displacement is not None
    anchor_ptr = anchor if track else np.intp(0)
    max_disp_ptr = max_displacement if track else np.intp(0)

    module = _get_module()
    kernel = module.get_function("K_UpdateBoundaryParticles")

    block = 256
    grid = (num_boundary + block - 1) // block

    kernel(
        (grid,), (block,),
        (
            boundary_data,
            boundary_body_id,
            d_rigid_bodies,
            position_out,
            velocity_out,
            mass_out,
            np.int32(offset),
            np.int32(num_boundary),
            anchor_ptr,
            max_disp_ptr,
            np.int32(1 if reset_anchor else 0),
        ),
    )
