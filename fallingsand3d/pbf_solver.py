"""PBF solver kernel wrapper -- Position Based Fluids (Macklin & Muller 2013).

Compiles physics/kernels/pbf_solver.cu via CuPy RawModule and provides
launch functions for the 5 PBF kernels.

Constant memory (each RawModule has its own address space):
  c_grid, c_sim, c_precalc, c_materials -- shared with other kernels
  c_pbf       -- PBF-specific parameters
  c_granular  -- mu(I) parameters (for GRANULAR friction in Finalize)
"""

from __future__ import annotations

import os
import math
from typing import Optional

import cupy
import numpy as np

# ---------------------------------------------------------------------------
# PBFParams dtype matching struct in pbf_solver.cu
# ---------------------------------------------------------------------------

PBF_PARAMS_DTYPE = np.dtype(
    [
        ("num_iterations", np.int32),
        ("relaxation", np.float32),
        ("s_corr_k", np.float32),
        ("s_corr_n", np.int32),
        ("s_corr_dq_sq", np.float32),
        ("s_corr_W_dq", np.float32),
        ("xsph_c", np.float32),
        ("neg_c_scale", np.float32),
    ],
    align=False,
)

assert PBF_PARAMS_DTYPE.itemsize == 32

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
    cu_path = os.path.join(kernel_dir, "pbf_solver.cu")
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


def upload_interactions(interactions_data: np.ndarray) -> None:
    """Upload Interaction[32][32] to ``__constant__ Interaction c_interactions[32][32]``."""
    module = _get_module()
    d_ptr = module.get_global("c_interactions")
    cupy.cuda.runtime.memcpy(int(d_ptr), interactions_data.ctypes.data, interactions_data.nbytes, 1)


def upload_granular_params(params: np.ndarray) -> None:
    module = _get_module()
    d_ptr = module.get_global("c_granular")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


def upload_pbf_params(profile) -> None:
    """Upload PBFParams to constant memory from a SolverProfile."""
    h = 0.04  # smoothing length
    dq = profile.pbf_s_corr_dq * h
    dq_sq = dq * dq
    h_sq = h * h
    # Precompute W_poly6(dq*h)
    poly6_coeff = 315.0 / (64.0 * math.pi * h ** 9)
    diff = h_sq - dq_sq
    W_dq = poly6_coeff * diff ** 3 if diff > 0 else 0.0

    params = np.zeros(1, dtype=PBF_PARAMS_DTYPE)
    params[0]["num_iterations"] = profile.pbf_iterations
    params[0]["relaxation"] = profile.pbf_relaxation
    params[0]["s_corr_k"] = profile.pbf_s_corr_k
    params[0]["s_corr_n"] = profile.pbf_s_corr_n
    params[0]["s_corr_dq_sq"] = dq_sq
    params[0]["s_corr_W_dq"] = W_dq
    params[0]["xsph_c"] = profile.pbf_xsph_c
    params[0]["neg_c_scale"] = profile.pbf_neg_c_scale

    module = _get_module()
    d_ptr = module.get_global("c_pbf")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


# ---------------------------------------------------------------------------
# Kernel launches
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def pbf_predict(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    packed_info: cupy.ndarray,
    temperature: cupy.ndarray,
    predicted_out: cupy.ndarray,
) -> None:
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_PBF_Predict")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), position, velocity, packed_info, temperature, predicted_out,
    ))


def pbf_compute_lambda(
    predicted_pos: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    density_out: cupy.ndarray,
    lambda_out: cupy.ndarray,
    pressure_normal_out: cupy.ndarray = None,
    # Optional heat diffusion + exposure (pass on first call only)
    temperature_in: "cupy.ndarray | None" = None,
    density_in: "cupy.ndarray | None" = None,
    dTdt_out: "cupy.ndarray | None" = None,
    exposure_heat_out: "cupy.ndarray | None" = None,
    exposure_corrode_out: "cupy.ndarray | None" = None,
    particle_dye_in: "cupy.ndarray | None" = None,
    dye_rate_out: "cupy.ndarray | None" = None,
    velocity_in: "cupy.ndarray | None" = None,
    vorticity_out: "cupy.ndarray | None" = None,
    normal_out: "cupy.ndarray | None" = None,
) -> None:
    n = predicted_pos.shape[0]
    if pressure_normal_out is None:
        pressure_normal_out = cupy.zeros((n, 4), dtype=cupy.float32)
    # NULL pointers for optional params
    _null = np.intp(0)
    t_in = temperature_in if temperature_in is not None else _null
    d_in = density_in if density_in is not None else _null
    dt_out = dTdt_out if dTdt_out is not None else _null
    eh_out = exposure_heat_out if exposure_heat_out is not None else _null
    ec_out = exposure_corrode_out if exposure_corrode_out is not None else _null
    dye_in = particle_dye_in if particle_dye_in is not None else _null
    dye_out = dye_rate_out if dye_rate_out is not None else _null
    vel_in = velocity_in if velocity_in is not None else _null
    vort_out = vorticity_out if vorticity_out is not None else _null
    norm_out = normal_out if normal_out is not None else _null
    module = _get_module()
    kernel = module.get_function("K_PBF_ComputeLambda")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), predicted_pos, mass, packed_info,
        cell_start, cell_end, density_out, lambda_out,
        pressure_normal_out,
        t_in, d_in, dt_out, eh_out, ec_out,
        dye_in, dye_out,
        vel_in, vort_out, norm_out,
    ))


def pbf_compute_delta(
    predicted_pos: cupy.ndarray,
    lambda_pbf: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    delta_out: cupy.ndarray,
) -> None:
    n = predicted_pos.shape[0]
    module = _get_module()
    kernel = module.get_function("K_PBF_ComputeDelta")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), predicted_pos, lambda_pbf, mass, packed_info,
        cell_start, cell_end, delta_out,
    ))


def pbf_apply_delta(
    predicted_pos: cupy.ndarray,
    delta_pos: cupy.ndarray,
    packed_info: cupy.ndarray,
) -> None:
    n = predicted_pos.shape[0]
    module = _get_module()
    kernel = module.get_function("K_PBF_ApplyDelta")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), predicted_pos, delta_pos, packed_info,
    ))


def pbf_finalize(
    predicted_pos: cupy.ndarray,
    original_pos: cupy.ndarray,
    original_vel: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    temperature: cupy.ndarray,
    health: cupy.ndarray,
    dTdt: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    sort_indexes: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    position_out: cupy.ndarray,
    velocity_out: cupy.ndarray,
    color_out: cupy.ndarray,
    packed_info_out: cupy.ndarray,
    sleep_counter_out: cupy.ndarray,
    temperature_out: cupy.ndarray,
    sorted_particle_dye: "Optional[cupy.ndarray]" = None,
    sorted_dye_rate: "Optional[cupy.ndarray]" = None,
    particle_dye_out: "Optional[cupy.ndarray]" = None,
    sorted_angular_velocity: "Optional[cupy.ndarray]" = None,
    angular_velocity_out: "Optional[cupy.ndarray]" = None,
    vorticity_in: "cupy.ndarray | None" = None,
    normal_in: "cupy.ndarray | None" = None,
    pressure_normal_in: "cupy.ndarray | None" = None,
) -> None:
    n = predicted_pos.shape[0]
    _null = np.intp(0)
    if sorted_particle_dye is None:
        sorted_particle_dye = cupy.zeros((n, 4), dtype=cupy.float32)
    if particle_dye_out is None:
        particle_dye_out = cupy.zeros((n, 4), dtype=cupy.float32)
    s_dye_rate = sorted_dye_rate if sorted_dye_rate is not None else _null
    if sorted_angular_velocity is None:
        sorted_angular_velocity = cupy.zeros((n, 4), dtype=cupy.float32)
    if angular_velocity_out is None:
        angular_velocity_out = cupy.zeros((n, 4), dtype=cupy.float32)
    vort_in = vorticity_in if vorticity_in is not None else _null
    norm_in = normal_in if normal_in is not None else _null
    pn_in = pressure_normal_in if pressure_normal_in is not None else _null
    module = _get_module()
    kernel = module.get_function("K_PBF_Finalize")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), predicted_pos, original_pos, original_vel,
        density, mass, packed_info, temperature, health, dTdt,
        sleep_counter, sort_indexes, cell_start, cell_end,
        position_out, velocity_out, color_out, packed_info_out,
        sleep_counter_out, temperature_out,
        sorted_particle_dye, s_dye_rate, particle_dye_out,
        sorted_angular_velocity, angular_velocity_out,
        vort_in, norm_in, pn_in,
    ))
