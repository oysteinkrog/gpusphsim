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
    sorted_density: "Optional[cupy.ndarray]" = None,
    sorted_shear_rate: "Optional[cupy.ndarray]" = None,
    sorted_dTdt: "Optional[cupy.ndarray]" = None,
    sorted_sleep_counter: "Optional[cupy.ndarray]" = None,
    sorted_dye_rate: "Optional[cupy.ndarray]" = None,
    sorted_particle_dye: "Optional[cupy.ndarray]" = None,
    sorted_vorticity: "Optional[cupy.ndarray]" = None,
    sorted_angular_velocity: "Optional[cupy.ndarray]" = None,
    sort_indexes: "Optional[cupy.ndarray]" = None,
    position_out: "Optional[cupy.ndarray]" = None,
    velocity_out: "Optional[cupy.ndarray]" = None,
    color_out: "Optional[cupy.ndarray]" = None,
    packed_info_out: "Optional[cupy.ndarray]" = None,
    sleep_counter_out: "Optional[cupy.ndarray]" = None,
    temperature_out: "Optional[cupy.ndarray]" = None,
    particle_dye_out: "Optional[cupy.ndarray]" = None,
    angular_velocity_out: "Optional[cupy.ndarray]" = None,
    max_displacement: "Optional[cupy.ndarray]" = None,
    cell_start: "Optional[cupy.ndarray]" = None,
    cell_end: "Optional[cupy.ndarray]" = None,
) -> tuple:
    """Launch K_Integrate and return outputs tuple.

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
        Pre-allocated (M, 4) float32 unsorted output.
    velocity_out : cupy.ndarray, optional
        Pre-allocated (M, 4) float32 unsorted output.
    color_out : cupy.ndarray, optional
        Pre-allocated (M, 4) float32 unsorted output.
    packed_info_out : cupy.ndarray, optional
        Pre-allocated (M,) uint32 unsorted output for updated packed_info.
    sleep_counter_out : cupy.ndarray, optional
        Pre-allocated (M,) uint8 unsorted output for updated sleep counter.
    temperature_out : cupy.ndarray, optional
        Pre-allocated (M,) float32 unsorted output for updated temperature.

    Returns
    -------
    (position_out, velocity_out, color_out, packed_info_out,
     sleep_counter_out, temperature_out)
    """
    n = sorted_position.shape[0]
    if n == 0:
        return (position_out, velocity_out, color_out, packed_info_out,
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
    if sorted_dye_rate is None:
        sorted_dye_rate = cupy.zeros((n, 4), dtype=cupy.float32)
    if sorted_particle_dye is None:
        sorted_particle_dye = cupy.zeros((n, 4), dtype=cupy.float32)
    if sorted_vorticity is None:
        sorted_vorticity = cupy.zeros((n, 4), dtype=cupy.float32)
    if sorted_angular_velocity is None:
        sorted_angular_velocity = cupy.zeros((n, 4), dtype=cupy.float32)

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
    if packed_info_out is None:
        max_idx = int(sort_indexes.max()) + 1 if n > 0 else n
        packed_info_out = cupy.zeros(max_idx, dtype=cupy.uint32)
    if sleep_counter_out is None:
        max_idx = int(sort_indexes.max()) + 1 if n > 0 else n
        sleep_counter_out = cupy.zeros(max_idx, dtype=cupy.uint8)
    if temperature_out is None:
        max_idx = int(sort_indexes.max()) + 1 if n > 0 else n
        temperature_out = cupy.zeros(max_idx, dtype=cupy.float32)
    if particle_dye_out is None:
        max_idx = int(sort_indexes.max()) + 1 if n > 0 else n
        particle_dye_out = cupy.zeros((max_idx, 4), dtype=cupy.float32)
    if angular_velocity_out is None:
        max_idx = int(sort_indexes.max()) + 1 if n > 0 else n
        angular_velocity_out = cupy.zeros((max_idx, 4), dtype=cupy.float32)

    module = _get_module()
    kernel = module.get_function("K_Integrate")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    # Use null pointer (0) when max_displacement not provided
    max_disp_ptr = max_displacement if max_displacement is not None else np.intp(0)
    # Use null pointer when cell_start/cell_end not provided (disables STATIC repulsion)
    cell_start_ptr = cell_start if cell_start is not None else np.intp(0)
    cell_end_ptr = cell_end if cell_end is not None else np.intp(0)

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
            sorted_density,
            sorted_shear_rate,
            sorted_dTdt,
            sorted_sleep_counter,
            sorted_dye_rate,
            sorted_particle_dye,
            sorted_vorticity,
            sorted_angular_velocity,
            sort_indexes,
            cell_start_ptr,
            cell_end_ptr,
            position_out,
            velocity_out,
            color_out,
            packed_info_out,
            sleep_counter_out,
            temperature_out,
            particle_dye_out,
            angular_velocity_out,
            max_disp_ptr,
        ),
    )

    return (position_out, velocity_out, color_out, packed_info_out,
            sleep_counter_out, temperature_out, particle_dye_out,
            angular_velocity_out)


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
    d_rigid_bodies: cupy.ndarray,
    position_out: cupy.ndarray,
    velocity_out: cupy.ndarray,
    mass_out: cupy.ndarray,
    offset: int,
    num_boundary: int,
) -> None:
    """Launch K_UpdateBoundaryParticles: transform local -> world positions.

    Runs after K_IntegrateRigidBodies, before hash/sort for next substep.
    Sets boundary particle positions and velocities from rigid body state.

    Parameters
    ----------
    boundary_data : cupy.ndarray, (N, 8) float32
        Combined boundary data: (x_local, y_local, z_local, psi, body_id, nx, ny, nz).
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
    """
    if num_boundary <= 0:
        return

    module = _get_module()
    kernel = module.get_function("K_UpdateBoundaryParticles")

    block = 256
    grid = (num_boundary + block - 1) // block

    kernel(
        (grid,), (block,),
        (
            boundary_data,
            d_rigid_bodies,
            position_out,
            velocity_out,
            mass_out,
            np.int32(offset),
            np.int32(num_boundary),
        ),
    )
