"""Step1 kernel: SPH density summation + strain-rate tensor + heat diffusion
+ exposure accumulation.

Compiles physics/kernels/step1.cu via CuPy RawModule and provides
a function to launch K_Step1 which computes per-particle density
from the sorted particle positions and masses using neighbor iteration.

Additionally computes heat diffusion dTdt via SPH Laplacian of temperature:
  dTdt = kappa_i * viscosity_lap_coeff * sum_j (m_j/rho_j) * (T_j - T_i) * (h - |r|)

Exposure accumulation via c_interactions table lookup (no branching):
  exposure_corrode_i += c_interactions[mat_i][mat_j].reaction_rate * W_poly6(r_sq, h_sq)
  exposure_heat_i += c_interactions[mat_i][mat_j].heat_exchange * max(T_j - T_i, 0) * W_poly6

For GRANULAR particles, computes the symmetric strain-rate tensor D from
SPH velocity gradient (spiky kernel) and writes the second invariant
gamma_dot = sqrt(2 * D:D) to the shear_rate output array.
Non-GRANULAR particles get shear_rate = 0.

Density formula
---------------
density_sum += m_j * (h^2 - |r_ij|^2)^3   for all neighbors j within h
density_i = max(1.0, poly6_coeff * density_sum)

Self-interaction is included (particle i contributes to its own density).
Per-particle mass m_j supports multi-material and mass splitting.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import cupy
import numpy as np

# ---------------------------------------------------------------------------
# Numpy dtypes matching constant memory structs in common.cuh
# ---------------------------------------------------------------------------

# SimParams -- matches struct SimParams in common.cuh
# struct SimParams {
#     float  smoothing_length;     //  4 bytes
#     float  smoothing_length_sq;  //  4 bytes
#     float  particle_mass;        //  4 bytes
#     float  particle_spacing;     //  4 bytes
#     float3 gravity;              // 12 bytes
#     float  dt;                   //  4 bytes
#     float  restitution;          //  4 bytes
#     float  wall_friction;        //  4 bytes
#     float3 world_min;            // 12 bytes
#     float3 world_max;            // 12 bytes
# };                               // Total: 64 bytes
SIM_PARAMS_DTYPE = np.dtype(
    [
        ("smoothing_length", np.float32),
        ("smoothing_length_sq", np.float32),
        ("particle_mass", np.float32),
        ("particle_spacing", np.float32),
        ("gravity", np.float32, (3,)),
        ("dt", np.float32),
        ("restitution", np.float32),
        ("wall_friction", np.float32),
        ("world_min", np.float32, (3,)),
        ("world_max", np.float32, (3,)),
    ],
    align=False,
)

# PrecalcParams -- matches struct PrecalcParams in common.cuh
# struct PrecalcParams {
#     float poly6_coeff;           //  4 bytes
#     float spiky_grad_coeff;      //  4 bytes
#     float viscosity_lap_coeff;   //  4 bytes
#     float pressure_precalc;      //  4 bytes
#     float viscosity_precalc;     //  4 bytes
# };                               // Total: 20 bytes
PRECALC_PARAMS_DTYPE = np.dtype(
    [
        ("poly6_coeff", np.float32),
        ("spiky_grad_coeff", np.float32),
        ("viscosity_lap_coeff", np.float32),
        ("pressure_precalc", np.float32),
        ("viscosity_precalc", np.float32),
    ],
    align=False,
)


def build_sim_params(
    smoothing_length: float = 0.04,
    particle_mass: float = 0.008,
    particle_spacing: float = 0.02,
    gravity: tuple = (0.0, -9.8, 0.0),
    dt: float = 0.001,
    restitution: float = 0.3,
    wall_friction: float = 0.5,
    world_min: tuple = (-1.0, -1.0, -1.0),
    world_max: tuple = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Build a single SimParams struct as a numpy structured array."""
    params = np.zeros(1, dtype=SIM_PARAMS_DTYPE)
    params[0]["smoothing_length"] = smoothing_length
    params[0]["smoothing_length_sq"] = smoothing_length * smoothing_length
    params[0]["particle_mass"] = particle_mass
    params[0]["particle_spacing"] = particle_spacing
    params[0]["gravity"] = gravity
    params[0]["dt"] = dt
    params[0]["restitution"] = restitution
    params[0]["wall_friction"] = wall_friction
    params[0]["world_min"] = world_min
    params[0]["world_max"] = world_max
    return params


def build_precalc_params(
    smoothing_length: float = 0.04,
    viscosity: float = 0.001,
) -> np.ndarray:
    """Build PrecalcParams from smoothing length.

    Parameters
    ----------
    smoothing_length : float
        SPH smoothing length h.
    viscosity : float
        Base viscosity mu (Pa*s).
    """
    h = smoothing_length
    params = np.zeros(1, dtype=PRECALC_PARAMS_DTYPE)
    params[0]["poly6_coeff"] = np.float32(315.0 / (64.0 * math.pi * h**9))
    params[0]["spiky_grad_coeff"] = np.float32(-45.0 / (math.pi * h**6))
    params[0]["viscosity_lap_coeff"] = np.float32(45.0 / (math.pi * h**6))
    params[0]["pressure_precalc"] = np.float32(45.0 / (math.pi * h**6))
    params[0]["viscosity_precalc"] = np.float32(viscosity * 45.0 / (math.pi * h**6))
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
    """Compile (or return cached) CuPy RawModule from step1.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "step1.cu")
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
    """Upload GridParams to ``__constant__ GridParams c_grid``."""
    module = _get_module()
    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


def upload_sim_params(params: Optional[np.ndarray] = None) -> None:
    """Upload SimParams to ``__constant__ SimParams c_sim``."""
    if params is None:
        params = build_sim_params()
    module = _get_module()
    d_ptr = module.get_global("c_sim")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


def upload_precalc_params(params: Optional[np.ndarray] = None) -> None:
    """Upload PrecalcParams to ``__constant__ PrecalcParams c_precalc``."""
    if params is None:
        params = build_precalc_params()
    module = _get_module()
    d_ptr = module.get_global("c_precalc")  # type: ignore[union-attr]
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


def upload_interactions(interactions_data: np.ndarray) -> None:
    """Upload Interaction[32][32] to ``__constant__ Interaction c_interactions[32][32]``."""
    module = _get_module()
    d_ptr = module.get_global("c_interactions")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), interactions_data.ctypes.data, interactions_data.nbytes, 1
    )


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def compute_step1(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    mass: cupy.ndarray,
    density_in: Optional[cupy.ndarray],
    packed_info: cupy.ndarray,
    temperature_in: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
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
) -> tuple:
    """Launch K_Step1 and return (density, shear_rate, dTdt, exposure_heat, exposure_corrode).

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Sorted particle positions.
    velocity : cupy.ndarray, (N, 4) float32
        Sorted particle velocities (used for strain-rate computation).
    mass : cupy.ndarray, (N,) float32
        Sorted per-particle masses.
    density_in : cupy.ndarray or None, (N,) float32
        Density from previous step for m_j/rho_j weighting in strain-rate
        and heat diffusion. Pass None on first frame (kernel uses rho0=1000 fallback).
    packed_info : cupy.ndarray, (N,) uint32
        Sorted packed_info for behavior class and material ID.
    temperature_in : cupy.ndarray, (N,) float32
        Sorted temperature per particle (for heat diffusion).
    cell_start : cupy.ndarray, (num_cells,) uint32
        Grid cell start indices (0xFFFFFFFF for empty).
    cell_end : cupy.ndarray, (num_cells,) uint32
        Grid cell end indices.
    density_out : cupy.ndarray, optional
        Pre-allocated (N,) float32 output buffer. If None, allocates new.
    shear_rate_out : cupy.ndarray, optional
        Pre-allocated (N,) float32 output buffer. If None, allocates new.
    dTdt_out : cupy.ndarray, optional
        Pre-allocated (N,) float32 output buffer. If None, allocates new.
    exposure_heat_out : cupy.ndarray, optional
        Pre-allocated (N,) float32 output buffer. If None, allocates new.
    exposure_corrode_out : cupy.ndarray, optional
        Pre-allocated (N,) float32 output buffer. If None, allocates new.

    Returns
    -------
    (density, shear_rate, dTdt, exposure_heat, exposure_corrode) : tuple of cupy.ndarray
        density: Per-particle density (clamped >= 1.0).
        shear_rate: Per-particle gamma_dot (0 for non-GRANULAR).
        dTdt: Per-particle temperature rate of change from heat diffusion.
        exposure_heat: Per-particle heat exposure from interaction table.
        exposure_corrode: Per-particle corrosion exposure from interaction table.
    """
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

    # density_in can be None (first frame) -- pass null pointer
    density_in_ptr = density_in if density_in is not None else cupy.ndarray(0, dtype=cupy.float32)
    # velocity_h can be None -- pass null pointer (kernel falls back to float4 reads)
    velocity_h_ptr = velocity_h if velocity_h is not None else cupy.ndarray(0, dtype=cupy.uint32)

    module = _get_module()
    kernel = module.get_function("K_Step1")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            position,
            velocity,
            mass,
            density_in_ptr,
            packed_info,
            temperature_in,
            cell_start,
            cell_end,
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
        ),
    )

    return density_out, shear_rate_out, dTdt_out, exposure_heat_out, exposure_corrode_out


PACK_BLOCK_SIZE = 256


def pack_density(
    position: cupy.ndarray,
    density: cupy.ndarray,
    n: int,
) -> None:
    """Launch K_PackDensity to write density into position.w.

    Must be called between Step1 and Step2 so that Step2 can read
    density from position.w instead of a separate array.
    """
    module = _get_module()
    kernel = module.get_function("K_PackDensity")
    grid = ((n + PACK_BLOCK_SIZE - 1) // PACK_BLOCK_SIZE,)
    block = (PACK_BLOCK_SIZE,)
    kernel(grid, block, (position, density, np.uint32(n)))
