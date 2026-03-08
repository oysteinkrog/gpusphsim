"""Neighbor list kernel wrappers: build compact lists + NL variants of Step1/Step2.

Compiles physics/kernels/neighbor_list.cu via CuPy RawModule and provides
functions to build neighbor lists and run step1/step2 using pre-built lists
instead of 27-cell grid scanning.

Kernels:
  K_BuildNeighborList -- build compact per-particle neighbor indices
  K_Step1_NL          -- density + strain-rate + heat (neighbor-list variant)
  K_Step2_NL          -- pressure + viscosity + XSPH (neighbor-list variant)

Memory layout (CSR-like, fixed max per particle):
  neighbor_indices[N * MAX_NB]: packed neighbor indices (uint32)
  neighbor_count[N]:            valid entry count per particle (uint32)
"""

from __future__ import annotations

import os
from typing import Optional

import cupy
import numpy as np

MAX_NB = 64
BLOCK_SIZE = 256

# ---------------------------------------------------------------------------
# CuPy RawModule compilation
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
    cu_path = os.path.join(kernel_dir, "neighbor_list.cu")
    with open(cu_path) as f:
        source = f.read()

    _module = cupy.RawModule(
        code=source,
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )
    return _module


def get_module() -> "object":
    return _get_module()


# ---------------------------------------------------------------------------
# Constant memory uploads (must mirror all symbols used by neighbor_list.cu)
# ---------------------------------------------------------------------------


def upload_grid_params(params: np.ndarray) -> None:
    module = _get_module()
    d_ptr = module.get_global("c_grid")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


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


def upload_interactions(interactions_data: np.ndarray) -> None:
    module = _get_module()
    d_ptr = module.get_global("c_interactions")
    cupy.cuda.runtime.memcpy(int(d_ptr), interactions_data.ctypes.data, interactions_data.nbytes, 1)


def upload_granular_params(params: np.ndarray) -> None:
    module = _get_module()
    d_ptr = module.get_global("c_granular")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


# ---------------------------------------------------------------------------
# Buffer allocation
# ---------------------------------------------------------------------------


def allocate_neighbor_buffers(
    max_particles: int, max_nb: int = MAX_NB
) -> tuple:
    """Allocate neighbor_indices and neighbor_count arrays.

    Returns (neighbor_indices, neighbor_count).
    """
    neighbor_indices = cupy.empty(max_particles * max_nb, dtype=cupy.uint32)
    neighbor_count = cupy.empty(max_particles, dtype=cupy.uint32)
    return neighbor_indices, neighbor_count


# ---------------------------------------------------------------------------
# K_BuildNeighborList
# ---------------------------------------------------------------------------


def build_neighbor_list(
    position: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    neighbor_indices: cupy.ndarray,
    neighbor_count: cupy.ndarray,
    max_nb: int = MAX_NB,
) -> None:
    """Build compact neighbor lists from spatial hash grid."""
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_BuildNeighborList")

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(grid, block, (
        np.uint32(n),
        position,
        cell_start,
        cell_end,
        neighbor_indices,
        neighbor_count,
        np.uint32(max_nb),
    ))


# ---------------------------------------------------------------------------
# K_Step1_NL
# ---------------------------------------------------------------------------


def compute_step1_nl(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    mass: cupy.ndarray,
    density_in: Optional[cupy.ndarray],
    packed_info: cupy.ndarray,
    temperature_in: cupy.ndarray,
    neighbor_indices: cupy.ndarray,
    neighbor_count: cupy.ndarray,
    density_out: Optional[cupy.ndarray] = None,
    shear_rate_out: Optional[cupy.ndarray] = None,
    dTdt_out: Optional[cupy.ndarray] = None,
    exposure_heat_out: Optional[cupy.ndarray] = None,
    exposure_corrode_out: Optional[cupy.ndarray] = None,
    vorticity_out: Optional[cupy.ndarray] = None,
    normal_out: Optional[cupy.ndarray] = None,
    particle_dye_in: Optional[cupy.ndarray] = None,
    dye_rate_out: Optional[cupy.ndarray] = None,
    velocity_h: Optional[cupy.ndarray] = None,
    pressure_out: Optional[cupy.ndarray] = None,
    temperature_h: Optional[cupy.ndarray] = None,
    dye_h: Optional[cupy.ndarray] = None,
    max_nb: int = MAX_NB,
) -> tuple:
    """Launch K_Step1_NL (neighbor-list variant of Step1)."""
    n = position.shape[0]
    if density_out is None:
        density_out = cupy.empty(n, dtype=cupy.float32)
    if shear_rate_out is None:
        shear_rate_out = cupy.empty(n, dtype=cupy.float32)
    if dTdt_out is None:
        dTdt_out = cupy.empty(n, dtype=cupy.float32)
    if exposure_heat_out is None:
        exposure_heat_out = cupy.empty(n, dtype=cupy.float32)
    if exposure_corrode_out is None:
        exposure_corrode_out = cupy.empty(n, dtype=cupy.float32)
    if vorticity_out is None:
        vorticity_out = cupy.empty((n, 4), dtype=cupy.float32)
    if normal_out is None:
        normal_out = cupy.empty((n, 4), dtype=cupy.float32)
    if particle_dye_in is None:
        particle_dye_in = cupy.zeros((n, 4), dtype=cupy.float32)
    if dye_rate_out is None:
        dye_rate_out = cupy.empty((n, 4), dtype=cupy.float32)
    if pressure_out is None:
        pressure_out = cupy.empty(n, dtype=cupy.float32)

    density_in_ptr = density_in if density_in is not None else cupy.ndarray(0, dtype=cupy.float32)
    _null = cupy.ndarray(0, dtype=cupy.uint32)
    velocity_h_ptr = velocity_h if velocity_h is not None else _null
    temperature_h_ptr = temperature_h if temperature_h is not None else _null
    dye_h_ptr = dye_h if dye_h is not None else _null

    module = _get_module()
    kernel = module.get_function("K_Step1_NL")

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(grid, block, (
        np.uint32(n),
        position,
        velocity,
        mass,
        density_in_ptr,
        packed_info,
        temperature_in,
        density_out,
        shear_rate_out,
        dTdt_out,
        exposure_heat_out,
        exposure_corrode_out,
        vorticity_out,
        normal_out,
        particle_dye_in,
        dye_rate_out,
        velocity_h_ptr,
        pressure_out,
        temperature_h_ptr,
        dye_h_ptr,
        neighbor_indices,
        neighbor_count,
        np.uint32(max_nb),
    ))

    return density_out, shear_rate_out, dTdt_out, exposure_heat_out, exposure_corrode_out, pressure_out


# ---------------------------------------------------------------------------
# K_Step2_NL
# ---------------------------------------------------------------------------


def compute_step2_nl(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    shear_rate: cupy.ndarray,
    neighbor_indices: cupy.ndarray,
    neighbor_count: cupy.ndarray,
    vorticity_in: Optional[cupy.ndarray] = None,
    normal_in: Optional[cupy.ndarray] = None,
    sph_force_out: Optional[cupy.ndarray] = None,
    veleval_out: Optional[cupy.ndarray] = None,
    velocity_h: Optional[cupy.ndarray] = None,
    pressure_in: Optional[cupy.ndarray] = None,
    d_rigid_bodies: Optional[cupy.ndarray] = None,
    d_rigid_forces: Optional[cupy.ndarray] = None,
    d_rigid_torques: Optional[cupy.ndarray] = None,
    max_nb: int = MAX_NB,
) -> tuple:
    """Launch K_Step2_NL (neighbor-list variant of Step2)."""
    n = position.shape[0]
    if vorticity_in is None:
        vorticity_in = cupy.zeros((n, 4), dtype=cupy.float32)
    if normal_in is None:
        normal_in = cupy.zeros((n, 4), dtype=cupy.float32)
    if sph_force_out is None:
        sph_force_out = cupy.zeros((n, 4), dtype=cupy.float32)
    if veleval_out is None:
        veleval_out = cupy.zeros((n, 4), dtype=cupy.float32)
    if pressure_in is None:
        pressure_in = cupy.zeros(n, dtype=cupy.float32)

    velocity_h_ptr = velocity_h if velocity_h is not None else cupy.ndarray(0, dtype=cupy.uint32)
    _null = cupy.ndarray(0, dtype=cupy.float32)
    rb_ptr = d_rigid_bodies if d_rigid_bodies is not None else _null
    rf_ptr = d_rigid_forces if d_rigid_forces is not None else _null
    rt_ptr = d_rigid_torques if d_rigid_torques is not None else _null

    module = _get_module()
    kernel = module.get_function("K_Step2_NL")

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(grid, block, (
        np.uint32(n),
        position,
        velocity,
        mass,
        packed_info,
        shear_rate,
        vorticity_in,
        normal_in,
        pressure_in,
        sph_force_out,
        veleval_out,
        velocity_h_ptr,
        rb_ptr,
        rf_ptr,
        rt_ptr,
        neighbor_indices,
        neighbor_count,
        np.uint32(max_nb),
    ))

    return sph_force_out, veleval_out
