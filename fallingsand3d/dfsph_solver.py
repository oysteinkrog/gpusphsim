"""DFSPH solver kernel wrapper -- Divergence-Free SPH (Bender & Koschier, SCA 2015).

Compiles physics/kernels/dfsph_solver.cu via CuPy RawModule and provides
launch functions for the 9 DFSPH kernels.

Constant memory (each RawModule has its own address space):
  c_grid, c_sim, c_precalc, c_materials -- shared with other kernels
  c_dfsph -- DFSPH-specific parameters
"""

from __future__ import annotations

import os
from typing import Optional

import cupy
import numpy as np

# ---------------------------------------------------------------------------
# DFSPHParams dtype matching struct in dfsph_solver.cu
# ---------------------------------------------------------------------------

DFSPH_PARAMS_DTYPE = np.dtype(
    [
        ("div_iters", np.int32),
        ("dens_iters", np.int32),
        ("warm_start", np.float32),
        ("omega", np.float32),
        ("alpha_limit", np.float32),
        ("_pad0", np.float32),
        ("_pad1", np.float32),
        ("_pad2", np.float32),
    ],
    align=False,
)

assert DFSPH_PARAMS_DTYPE.itemsize == 32

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
    cu_path = os.path.join(kernel_dir, "dfsph_solver.cu")
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


def upload_granular_params(params: np.ndarray) -> None:
    """Upload GranularParams to ``__constant__ GranularParams c_granular``."""
    module = _get_module()
    d_ptr = module.get_global("c_granular")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


def upload_interactions(interactions_data: np.ndarray) -> None:
    """Upload Interaction[32][32] to ``__constant__ Interaction c_interactions[32][32]``."""
    module = _get_module()
    d_ptr = module.get_global("c_interactions")
    cupy.cuda.runtime.memcpy(int(d_ptr), interactions_data.ctypes.data, interactions_data.nbytes, 1)


def upload_dfsph_params(profile) -> None:
    """Upload DFSPHParams to constant memory from a SolverProfile."""
    params = np.zeros(1, dtype=DFSPH_PARAMS_DTYPE)
    params[0]["div_iters"] = profile.dfsph_div_iters
    params[0]["dens_iters"] = profile.dfsph_dens_iters
    params[0]["warm_start"] = profile.dfsph_warm_start
    params[0]["omega"] = profile.dfsph_omega
    params[0]["alpha_limit"] = profile.dfsph_alpha_limit

    module = _get_module()
    d_ptr = module.get_global("c_dfsph")
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


# ---------------------------------------------------------------------------
# Kernel launches
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def compute_density_alpha(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    mass: cupy.ndarray,
    density_in: cupy.ndarray | None,
    packed_info: cupy.ndarray,
    temperature: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    density_out: cupy.ndarray,
    alpha_out: cupy.ndarray,
    shear_rate_out: cupy.ndarray,
    dTdt_out: cupy.ndarray,
    exposure_heat_out: cupy.ndarray,
    exposure_corrode_out: cupy.ndarray,
    particle_dye_in: "cupy.ndarray | None" = None,
    dye_rate_out: "cupy.ndarray | None" = None,
    vorticity_out: "cupy.ndarray | None" = None,
    normal_out: "cupy.ndarray | None" = None,
) -> None:
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ComputeDensityAlpha")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    # density_in can be None on first frame -- pass empty array (null pointer)
    _null = np.intp(0)
    din = density_in if density_in is not None else cupy.ndarray(0, dtype=cupy.float32)
    dye_in = particle_dye_in if particle_dye_in is not None else _null
    dye_out = dye_rate_out if dye_rate_out is not None else _null
    vort_out = vorticity_out if vorticity_out is not None else _null
    norm_out = normal_out if normal_out is not None else _null
    kernel(grid, block, (
        np.uint32(n), position, velocity, mass, din,
        packed_info, temperature, cell_start, cell_end,
        density_out, alpha_out, shear_rate_out,
        dTdt_out, exposure_heat_out, exposure_corrode_out,
        dye_in, dye_out, vort_out, norm_out,
    ))


def compute_non_pressure_forces(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    shear_rate: cupy.ndarray,
    temperature: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    velocity_out: cupy.ndarray,
    vorticity_in: "cupy.ndarray | None" = None,
    normal_in: "cupy.ndarray | None" = None,
) -> None:
    n = position.shape[0]
    _null = np.intp(0)
    vort_in = vorticity_in if vorticity_in is not None else _null
    norm_in = normal_in if normal_in is not None else _null
    module = _get_module()
    kernel = module.get_function("K_DFSPH_NonPressureForces")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), position, velocity, density, mass,
        packed_info, shear_rate, temperature,
        cell_start, cell_end, velocity_out,
        vort_in, norm_in,
    ))


def compute_kappa_v(
    velocity: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    alpha: cupy.ndarray,
    packed_info: cupy.ndarray,
    position: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    kappa_v_out: cupy.ndarray,
) -> None:
    n = velocity.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ComputeKappaV")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), velocity, density, mass, alpha,
        packed_info, position, cell_start, cell_end,
        kappa_v_out,
    ))


def correct_velocity_div(
    velocity: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    kappa_v: cupy.ndarray,
    packed_info: cupy.ndarray,
    position: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
) -> None:
    n = velocity.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_CorrectVelocityDiv")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), velocity, density, mass, kappa_v,
        packed_info, position, cell_start, cell_end,
    ))


def predict_position(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    packed_info: cupy.ndarray,
    predicted_out: cupy.ndarray,
) -> None:
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_PredictPosition")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), position, velocity, packed_info, predicted_out,
    ))


def compute_density_adv(
    predicted_pos: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    original_pos: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    density_out: cupy.ndarray,
) -> None:
    n = predicted_pos.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ComputeDensityAdv")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), predicted_pos, mass, packed_info,
        original_pos, cell_start, cell_end, density_out,
    ))


def compute_kappa(
    density: cupy.ndarray,
    alpha: cupy.ndarray,
    packed_info: cupy.ndarray,
    kappa_out: cupy.ndarray,
) -> None:
    n = density.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ComputeKappa")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), density, alpha, packed_info, kappa_out,
    ))


def compute_kappa_from_velocity(
    velocity: cupy.ndarray,
    position: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    alpha: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    kappa_out: cupy.ndarray,
) -> None:
    """Compute density kappa using velocity-based density prediction.

    Instead of computing poly6 density at predicted positions (stale grid),
    predicts density from the velocity divergence: rho_adv = rho + dt * drho/dt.
    """
    n = velocity.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ComputeKappaFromVelocity")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), velocity, position, density, mass, alpha,
        packed_info, cell_start, cell_end, kappa_out,
    ))


def correct_velocity_dens(
    velocity: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    kappa: cupy.ndarray,
    packed_info: cupy.ndarray,
    position: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
) -> None:
    n = velocity.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_CorrectVelocityDens")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), velocity, density, mass, kappa,
        packed_info, position, cell_start, cell_end,
    ))


def compute_pressure_accel(
    p_rho2: cupy.ndarray,
    position: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    accel_out: cupy.ndarray,
) -> None:
    """Compute pressure acceleration from p/rho^2 (Jacobi density solver step 1)."""
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ComputePressureAccel")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), p_rho2, position, density, mass,
        packed_info, cell_start, cell_end, accel_out,
    ))


def density_solver_update(
    velocity: cupy.ndarray,
    accel_press: cupy.ndarray,
    position: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    alpha: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    p_rho2: cupy.ndarray,
) -> None:
    """Jacobi update: predict density with pressure, update p_rho2 (step 2)."""
    n = velocity.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_DensitySolverUpdate")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), velocity, accel_press, position, density, mass,
        alpha, packed_info, cell_start, cell_end, p_rho2,
    ))


def apply_pressure_velocity(
    velocity: cupy.ndarray,
    accel_press: cupy.ndarray,
    packed_info: cupy.ndarray,
) -> None:
    """Apply converged pressure acceleration to velocity (step 3)."""
    n = velocity.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ApplyPressureVelocity")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), velocity, accel_press, packed_info,
    ))


def finalize(
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_density: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_dTdt: cupy.ndarray,
    sorted_sleep_counter: cupy.ndarray,
    sorted_kappa: cupy.ndarray,
    sort_indexes: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    position_out: cupy.ndarray,
    velocity_out: cupy.ndarray,
    color_out: cupy.ndarray,
    packed_info_out: cupy.ndarray,
    sleep_counter_out: cupy.ndarray,
    temperature_out: cupy.ndarray,
    kappa_out: cupy.ndarray,
    sorted_particle_dye: "Optional[cupy.ndarray]" = None,
    sorted_dye_rate: "Optional[cupy.ndarray]" = None,
    particle_dye_out: "Optional[cupy.ndarray]" = None,
    sorted_angular_velocity: "Optional[cupy.ndarray]" = None,
    angular_velocity_out: "Optional[cupy.ndarray]" = None,
    vorticity_in: "cupy.ndarray | None" = None,
) -> None:
    n = sorted_position.shape[0]
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
    module = _get_module()
    kernel = module.get_function("K_DFSPH_Finalize")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), sorted_position, sorted_velocity,
        sorted_density, sorted_mass, sorted_packed_info,
        sorted_temperature, sorted_health, sorted_dTdt,
        sorted_sleep_counter, sorted_kappa, sort_indexes, cell_start, cell_end,
        position_out, velocity_out, color_out, packed_info_out,
        sleep_counter_out, temperature_out, kappa_out,
        sorted_particle_dye, s_dye_rate, particle_dye_out,
        sorted_angular_velocity, angular_velocity_out,
        vort_in,
    ))
