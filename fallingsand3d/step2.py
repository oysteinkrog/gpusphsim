"""Step2 kernel: pressure, viscosity, and XSPH force computation.

Compiles physics/kernels/step2.cu via CuPy RawModule and provides
functions to upload GranularParams to constant memory and launch
the K_Step2 kernel.

Uses shared constant memory from common.cuh:
  c_grid     -- GridParams (uploaded by hash_sort.py)
  c_sim      -- SimParams (uploaded by step1.py)
  c_precalc  -- PrecalcParams (uploaded by step1.py)
  c_materials -- MaterialProps[32] (uploaded by materials.py)

Additional constant memory local to this module:
  c_granular -- GranularParams (mu(I) rheology + xsph_epsilon)

Tait EOS pressure
-----------------
- FLUID/GRANULAR: p_raw = k * (pow(rho/rho0, 7) - 1)
  - FLUID:    p = max(p_raw, -0.5*k)
  - GRANULAR: p = max(p_raw, 0)
- GAS: p = k_gas * max(rho - rho0, 0) with gamma=1

Force accumulation
------------------
- Pressure force: pressure_precalc * m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky_variable
- Viscosity force: viscosity_precalc * m_j * (v_j - v_i) / rho_j * lap_visc_variable
- XSPH (FLUID only): epsilon * sum(m_j / rho_avg * (v_j - v_i) * W_poly6)
"""

from __future__ import annotations

import os
from typing import Optional

import cupy
import numpy as np

# ---------------------------------------------------------------------------
# Default physical parameters
# ---------------------------------------------------------------------------

DEFAULT_MU_S = np.float32(0.55)   # tan(~29°), higher for sandcastle-like repose
DEFAULT_MU_2 = np.float32(0.85)   # dynamic friction (flowing sand)
DEFAULT_I0 = np.float32(0.3)
DEFAULT_MU_MAX = np.float32(500.0)  # CFL-safe: nu=mu/rho=500/4000=0.125, dt_visc=0.125*h^2/nu=0.016s >> dt=0.001
DEFAULT_PARTICLE_SPACING = np.float32(0.02)
DEFAULT_MU0 = np.float32(1.0)
DEFAULT_XSPH_EPSILON = np.float32(0.8)
DEFAULT_FORCE_SCALE = np.float32(0.02)
DEFAULT_VORTICITY_EPSILON = np.float32(0.05)
DEFAULT_SURFACE_TENSION_GAMMA = np.float32(1.0)
DEFAULT_TAN_PHI_F = np.float32(0.781)  # tan(38°) for Drucker-Prager friction
DEFAULT_COHESION = np.float32(0.002)   # small cohesion for DP stability

# ---------------------------------------------------------------------------
# Numpy dtype matching GranularParams struct in step2.cu
#
# struct GranularParams {
#     float mu_s;                    //  4 bytes
#     float mu_2;                    //  4 bytes
#     float I0;                      //  4 bytes
#     float mu_max;                  //  4 bytes
#     float particle_spacing;        //  4 bytes
#     float mu0;                     //  4 bytes
#     float xsph_epsilon;            //  4 bytes
#     float force_scale;             //  4 bytes
#     float vorticity_epsilon;       //  4 bytes
#     float surface_tension_gamma;   //  4 bytes
#     float tan_phi_f;               //  4 bytes
#     float cohesion;                //  4 bytes
# };                                 // Total: 48 bytes
# ---------------------------------------------------------------------------

GRANULAR_PARAMS_DTYPE = np.dtype(
    [
        ("mu_s", np.float32),
        ("mu_2", np.float32),
        ("I0", np.float32),
        ("mu_max", np.float32),
        ("particle_spacing", np.float32),
        ("mu0", np.float32),
        ("xsph_epsilon", np.float32),
        ("force_scale", np.float32),
        ("vorticity_epsilon", np.float32),
        ("surface_tension_gamma", np.float32),
        ("tan_phi_f", np.float32),
        ("cohesion", np.float32),
    ],
    align=False,
)

assert GRANULAR_PARAMS_DTYPE.itemsize == 48, (
    f"GranularParams size mismatch: {GRANULAR_PARAMS_DTYPE.itemsize} != 48"
)


def build_granular_params(
    mu_s: float = DEFAULT_MU_S,
    mu_2: float = DEFAULT_MU_2,
    I0: float = DEFAULT_I0,
    mu_max: float = DEFAULT_MU_MAX,
    particle_spacing: float = DEFAULT_PARTICLE_SPACING,
    mu0: float = DEFAULT_MU0,
    xsph_epsilon: float = DEFAULT_XSPH_EPSILON,
    force_scale: float = DEFAULT_FORCE_SCALE,
    vorticity_epsilon: float = DEFAULT_VORTICITY_EPSILON,
    surface_tension_gamma: float = DEFAULT_SURFACE_TENSION_GAMMA,
    tan_phi_f: float = DEFAULT_TAN_PHI_F,
    cohesion: float = DEFAULT_COHESION,
) -> np.ndarray:
    """Build a GranularParams struct as a numpy structured array."""
    params = np.zeros(1, dtype=GRANULAR_PARAMS_DTYPE)
    params[0]["mu_s"] = mu_s
    params[0]["mu_2"] = mu_2
    params[0]["I0"] = I0
    params[0]["mu_max"] = mu_max
    params[0]["particle_spacing"] = particle_spacing
    params[0]["mu0"] = mu0
    params[0]["xsph_epsilon"] = xsph_epsilon
    params[0]["force_scale"] = force_scale
    params[0]["vorticity_epsilon"] = vorticity_epsilon
    params[0]["surface_tension_gamma"] = surface_tension_gamma
    params[0]["tan_phi_f"] = tan_phi_f
    params[0]["cohesion"] = cohesion
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
    """Compile (or return cached) CuPy RawModule from step2.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "step2.cu")
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


def upload_grid_params(grid_params: np.ndarray) -> None:
    """Upload GridParams to ``__constant__ GridParams c_grid``."""
    module = _get_module()
    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), grid_params.ctypes.data, grid_params.nbytes, 1
    )


def upload_sim_params(params: np.ndarray) -> None:
    """Upload SimParams to ``__constant__ SimParams c_sim``."""
    module = _get_module()
    d_ptr = module.get_global("c_sim")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


def upload_precalc_params(params: np.ndarray) -> None:
    """Upload PrecalcParams to ``__constant__ PrecalcParams c_precalc``."""
    module = _get_module()
    d_ptr = module.get_global("c_precalc")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


def upload_granular_params(params: Optional[np.ndarray] = None) -> None:
    """Upload GranularParams to ``__constant__ GranularParams c_granular``."""
    if params is None:
        params = build_granular_params()
    module = _get_module()
    d_ptr = module.get_global("c_granular")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


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


def compute_pressure(
    density: cupy.ndarray,
    packed_info: cupy.ndarray,
    pressure_out: cupy.ndarray,
) -> None:
    """Launch K_ComputePressure to pre-compute per-particle pressure (PERF-007).

    Must be called after Step1 (density available) and before Step2.
    WCSPH only -- PBF/DFSPH have their own constraint forces.
    """
    n = density.shape[0]
    module = _get_module()
    kernel = module.get_function("K_ComputePressure")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (np.uint32(n), density, packed_info, pressure_out))


def compute_step2(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    shear_rate: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    vorticity_in: "Optional[cupy.ndarray]" = None,
    normal_in: "Optional[cupy.ndarray]" = None,
    sph_force_out: "Optional[cupy.ndarray]" = None,
    veleval_out: "Optional[cupy.ndarray]" = None,
    velocity_h: "Optional[cupy.ndarray]" = None,
    pressure_in: "Optional[cupy.ndarray]" = None,
    d_rigid_bodies: "Optional[cupy.ndarray]" = None,
    d_rigid_forces: "Optional[cupy.ndarray]" = None,
    d_rigid_torques: "Optional[cupy.ndarray]" = None,
) -> tuple:
    """Launch K_Step2 and return (sph_force, veleval_out).

    Density is read from position.w (packed by K_PackDensity after Step1).

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Sorted particle positions (density packed in .w by K_PackDensity).
    velocity : cupy.ndarray, (N, 4) float32
        Sorted evaluation velocities.
    mass : cupy.ndarray, (N,) float32
        Sorted per-particle mass.
    packed_info : cupy.ndarray, (N,) uint32
        Sorted packed_info (material_id + behavior_class + flags).
    shear_rate : cupy.ndarray, (N,) float32
        Sorted shear rate / gamma_dot (from Step1).
    cell_start : cupy.ndarray, (num_cells,) uint32
        Grid cell start indices (0xFFFFFFFF for empty).
    cell_end : cupy.ndarray, (num_cells,) uint32
        Grid cell end indices.
    sph_force_out : cupy.ndarray, optional
        Pre-allocated (N, 4) float32 output. If None, allocates new.
    veleval_out : cupy.ndarray, optional
        Pre-allocated (N, 4) float32 output. If None, allocates new.

    Returns
    -------
    sph_force : cupy.ndarray, (N, 4) float32
        Accumulated SPH force per particle.
    veleval_out : cupy.ndarray, (N, 4) float32
        XSPH-corrected evaluation velocity (FLUID only; others unchanged).
    """
    n = position.shape[0]
    if vorticity_in is None:
        vorticity_in = cupy.zeros((n, 4), dtype=cupy.float32)
    if normal_in is None:
        normal_in = cupy.zeros((n, 4), dtype=cupy.float32)
    if sph_force_out is None:
        sph_force_out = cupy.zeros((n, 4), dtype=cupy.float32)
    if veleval_out is None:
        veleval_out = cupy.zeros((n, 4), dtype=cupy.float32)

    module = _get_module()
    kernel = module.get_function("K_Step2")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    # velocity_h can be None -- pass null pointer (kernel falls back to float4 reads)
    velocity_h_ptr = velocity_h if velocity_h is not None else cupy.ndarray(0, dtype=cupy.uint32)

    # Pressure array (PERF-007): must be pre-computed via compute_pressure()
    if pressure_in is None:
        pressure_in = cupy.zeros(n, dtype=cupy.float32)

    # Rigid body pointers: pass null if no bodies
    _null = cupy.ndarray(0, dtype=cupy.float32)
    rb_ptr = d_rigid_bodies if d_rigid_bodies is not None else _null
    rf_ptr = d_rigid_forces if d_rigid_forces is not None else _null
    rt_ptr = d_rigid_torques if d_rigid_torques is not None else _null

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            position,
            velocity,
            mass,
            packed_info,
            shear_rate,
            cell_start,
            cell_end,
            vorticity_in,
            normal_in,
            pressure_in,
            sph_force_out,
            veleval_out,
            velocity_h_ptr,
            rb_ptr,
            rf_ptr,
            rt_ptr,
        ),
    )

    return sph_force_out, veleval_out
