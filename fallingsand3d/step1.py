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
from typing import Optional

import cupy
import numpy as np

from kernel_loader import load_module

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
#     float  velocity_damping;     //  4 bytes
#     float  velocity_limit;       //  4 bytes
#     int    thermal_enabled;      //  4 bytes
# };                               // Total: 76 bytes
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
        ("velocity_damping", np.float32),
        ("velocity_limit", np.float32),
        ("thermal_enabled", np.int32),
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
    gravity: tuple = (0.0, -4.0, 0.0),
    dt: float = 0.001,
    restitution: float = 0.3,
    wall_friction: float = 0.5,
    world_min: tuple = (-1.0, -1.0, -1.0),
    world_max: tuple = (1.0, 1.0, 1.0),
    velocity_damping: float = 0.0,
    velocity_limit: float = 10.0,
    thermal_enabled: bool = True,
) -> np.ndarray:
    """Build a single SimParams struct as a numpy structured array.

    thermal_enabled False tells K_Step1 to skip heat diffusion and exposure
    and write 0 for dTdt and both exposures. Only use it for a scene where
    every temperature is T_AMBIENT and no present pair of materials has a
    non-zero c_interactions entry (Simulation sets it from that rule).
    """
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
    params[0]["velocity_damping"] = velocity_damping
    params[0]["velocity_limit"] = velocity_limit
    params[0]["thermal_enabled"] = 1 if thermal_enabled else 0
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

def _get_module() -> "object":
    """Compile (or return cached) CuPy RawModule from step1.cu."""
    return load_module("step1.cu")


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


def upload_granular_params(params: np.ndarray) -> None:
    """Upload GranularParams to ``__constant__ GranularParams c_granular``.

    K_Step1 reads it for the mu(I) eta it writes to eta_out. step2's
    upload_granular_params calls this, so both modules hold the same values.
    """
    module = _get_module()
    d_ptr = module.get_global("c_granular")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


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
    velocity_h: Optional[cupy.ndarray] = None,
    pressure_out: Optional[cupy.ndarray] = None,
    temperature_h: Optional[cupy.ndarray] = None,
    pr2_omega_out: Optional[cupy.ndarray] = None,
    eta_out: Optional[cupy.ndarray] = None,
    frame_counter: Optional[cupy.ndarray] = None,
    force_refresh: bool = True,
) -> tuple:
    """Launch K_Step1 and return (density, shear_rate, dTdt, exposure_heat, exposure_corrode).

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Sorted particle positions. .w is read as each neighbour's density
        rho_j for the m_j/rho_j weights in heat diffusion (the previous
        substep's density; 0 or less falls back to the rest density), and
        K_Step1 overwrites .w with this call's density.
    velocity : cupy.ndarray, (N, 4) float32
        Sorted particle velocities (used for strain-rate computation).
    mass : cupy.ndarray, (N,) float32
        Sorted per-particle masses.
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
    pr2_omega_out : cupy.ndarray, optional
        (N, 2) float32 output of (p / rho^2, |omega|), the packed neighbour
        carrier K_Step2 reads (pass the same array as its pr2_omega_in).
        If None, K_Step1 skips the write.
    eta_out : cupy.ndarray, optional
        (N,) float32 output of the mu(I) viscosity eta per GRANULAR particle
        (0 for the rest), which K_Step2 reads as eta_in instead of
        recomputing it per pair. Needs c_granular in this module, which
        step2.upload_granular_params uploads. If None, K_Step1 skips it.
    frame_counter : cupy.ndarray, optional
        (1,) uint32 per-substep device counter (Simulation._frame_counter_d).
        With it, sleeping and non-rigid STATIC particles run the neighbour loop
        only when (counter + index // 32) % 8 == 0 and otherwise write density,
        pressure and pr2_omega from the carried density and leave the other
        outputs as they are (bd-r5-review-n28.159). The output buffers must
        then be the ones the last K_Step1 wrote, in the same sorted order.
        If None, every particle computes every call.
    force_refresh : bool
        True makes every inactive particle compute this call. Pass True on
        any substep whose sort changed the sorted order (a full sort), since
        the carried outputs then belong to other particles.

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
    if pressure_out is None:
        pressure_out = cupy.empty(n, dtype=cupy.float32)

    # velocity_h can be None -- pass null pointer (kernel falls back to float4 reads)
    _null = cupy.ndarray(0, dtype=cupy.uint32)
    velocity_h_ptr = velocity_h if velocity_h is not None else _null
    temperature_h_ptr = temperature_h if temperature_h is not None else _null
    pr2_omega_ptr = pr2_omega_out if pr2_omega_out is not None else _null
    eta_ptr = eta_out if eta_out is not None else _null

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
            velocity_h_ptr,
            pressure_out,
            temperature_h_ptr,
            pr2_omega_ptr,
            eta_ptr,
            frame_counter if frame_counter is not None else _null,
            np.uint32(1 if force_refresh else 0),
        ),
    )

    return density_out, shear_rate_out, dTdt_out, exposure_heat_out, exposure_corrode_out, pressure_out
