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

DEFAULT_MU_S = np.float32(0.36)
DEFAULT_MU_2 = np.float32(0.70)
DEFAULT_I0 = np.float32(0.3)
DEFAULT_MU_MAX = np.float32(10000.0)
DEFAULT_PARTICLE_SPACING = np.float32(0.02)
DEFAULT_MU0 = np.float32(0.1)
DEFAULT_XSPH_EPSILON = np.float32(0.5)
DEFAULT_FORCE_SCALE = np.float32(0.02)

# ---------------------------------------------------------------------------
# Numpy dtype matching GranularParams struct in step2.cu
#
# struct GranularParams {
#     float mu_s;               //  4 bytes
#     float mu_2;               //  4 bytes
#     float I0;                 //  4 bytes
#     float mu_max;             //  4 bytes
#     float particle_spacing;   //  4 bytes
#     float mu0;                //  4 bytes
#     float xsph_epsilon;       //  4 bytes
#     float force_scale;        //  4 bytes
# };                            // Total: 32 bytes
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
    ],
    align=False,
)

assert GRANULAR_PARAMS_DTYPE.itemsize == 32, (
    f"GranularParams size mismatch: {GRANULAR_PARAMS_DTYPE.itemsize} != 32"
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


def compute_step2(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    sph_force_out: "Optional[cupy.ndarray]" = None,
    veleval_out: "Optional[cupy.ndarray]" = None,
) -> tuple:
    """Launch K_Step2 and return (sph_force, veleval_out).

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Sorted particle positions.
    velocity : cupy.ndarray, (N, 4) float32
        Sorted evaluation velocities.
    density : cupy.ndarray, (N,) float32
        Sorted particle densities (from Step1).
    mass : cupy.ndarray, (N,) float32
        Sorted per-particle mass.
    packed_info : cupy.ndarray, (N,) uint32
        Sorted packed_info (material_id + behavior_class + flags).
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
    if sph_force_out is None:
        sph_force_out = cupy.zeros((n, 4), dtype=cupy.float32)
    if veleval_out is None:
        veleval_out = cupy.zeros((n, 4), dtype=cupy.float32)

    module = _get_module()
    kernel = module.get_function("K_Step2")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid,
        block,
        (
            np.uint32(n),
            position,
            velocity,
            density,
            mass,
            packed_info,
            cell_start,
            cell_end,
            sph_force_out,
            veleval_out,
        ),
    )

    return sph_force_out, veleval_out
