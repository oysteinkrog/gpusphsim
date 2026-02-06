"""Step2 kernel: pressure, viscosity, and XSPH force computation.

Compiles physics/kernels/step2.cu via CuPy RawModule and provides
functions to upload FluidParams/PrecalcParams to constant memory and
launch the K_Step2 kernel.

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

Behavior classes
----------------
- FLUID=1, GRANULAR=2, STATIC=3, GAS=4
- STATIC and SLEEPING particles are skipped (zero force, early return)
"""

from __future__ import annotations

import math
import os
from typing import Optional

import cupy  # type: ignore[import-untyped]
import numpy as np

# ---------------------------------------------------------------------------
# Behavior class constants (must match step2.cu #defines)
# ---------------------------------------------------------------------------

BEHAVIOR_FLUID = 1
BEHAVIOR_GRANULAR = 2
BEHAVIOR_STATIC = 3
BEHAVIOR_GAS = 4

FLAG_IS_SLEEPING = 1 << 0

# ---------------------------------------------------------------------------
# Default physical parameters
# ---------------------------------------------------------------------------

DEFAULT_SMOOTHING_LENGTH = np.float32(0.04)
DEFAULT_PARTICLE_MASS = np.float32(0.02)
DEFAULT_REST_DENSITY = np.float32(1000.0)
DEFAULT_GAS_STIFFNESS = np.float32(3.0)  # k for Tait EOS
DEFAULT_GAS_STIFFNESS_GAS = np.float32(1.5)  # k_gas for GAS type
DEFAULT_GAMMA = np.float32(7.0)
DEFAULT_VISCOSITY = np.float32(3.5)
DEFAULT_XSPH_EPSILON = np.float32(0.5)

# ---------------------------------------------------------------------------
# Numpy dtypes matching CUDA structs in step2.cu
# ---------------------------------------------------------------------------

FLUID_PARAMS_DTYPE = np.dtype(
    [
        ("smoothing_length", np.float32),
        ("particle_mass", np.float32),
        ("rest_density", np.float32),
        ("gas_stiffness", np.float32),
        ("gas_stiffness_gas", np.float32),
        ("gamma", np.float32),
        ("viscosity", np.float32),
        ("xsph_epsilon", np.float32),
    ],
    align=True,
)

PRECALC_PARAMS_DTYPE = np.dtype(
    [
        ("smoothing_length_pow2", np.float32),
        ("pressure_precalc", np.float32),
        ("viscosity_precalc", np.float32),
        ("kernel_poly6_coeff", np.float32),
    ],
    align=True,
)


def build_fluid_params(
    smoothing_length: float = DEFAULT_SMOOTHING_LENGTH,
    particle_mass: float = DEFAULT_PARTICLE_MASS,
    rest_density: float = DEFAULT_REST_DENSITY,
    gas_stiffness: float = DEFAULT_GAS_STIFFNESS,
    gas_stiffness_gas: float = DEFAULT_GAS_STIFFNESS_GAS,
    gamma: float = DEFAULT_GAMMA,
    viscosity: float = DEFAULT_VISCOSITY,
    xsph_epsilon: float = DEFAULT_XSPH_EPSILON,
) -> np.ndarray:
    """Build a FluidParams struct as a numpy structured array."""
    params = np.zeros(1, dtype=FLUID_PARAMS_DTYPE)
    params[0]["smoothing_length"] = smoothing_length
    params[0]["particle_mass"] = particle_mass
    params[0]["rest_density"] = rest_density
    params[0]["gas_stiffness"] = gas_stiffness
    params[0]["gas_stiffness_gas"] = gas_stiffness_gas
    params[0]["gamma"] = gamma
    params[0]["viscosity"] = viscosity
    params[0]["xsph_epsilon"] = xsph_epsilon
    return params


def build_precalc_params(
    smoothing_length: float = DEFAULT_SMOOTHING_LENGTH,
    viscosity: float = DEFAULT_VISCOSITY,
) -> np.ndarray:
    """Build a PrecalcParams struct with precomputed kernel coefficients.

    pressure_precalc  = +45 / (pi * h^6)  (POSITIVE; absorbs double negative)
    viscosity_precalc = viscosity * 45 / (pi * h^6)
    kernel_poly6_coeff = 315 / (64 * pi * h^9)
    """
    h = float(smoothing_length)
    mu = float(viscosity)
    lap_const = 45.0 / (math.pi * h**6)
    params = np.zeros(1, dtype=PRECALC_PARAMS_DTYPE)
    params[0]["smoothing_length_pow2"] = h * h
    params[0]["pressure_precalc"] = lap_const
    params[0]["viscosity_precalc"] = mu * lap_const
    params[0]["kernel_poly6_coeff"] = 315.0 / (64.0 * math.pi * h**9)
    return params


# ---------------------------------------------------------------------------
# CuPy RawModule compilation
# ---------------------------------------------------------------------------

_module: Optional[object] = None


def _ensure_ptx_if_needed() -> None:
    """Force PTX compilation when GPU arch exceeds NVRTC's max sm target."""
    from cupy.cuda import compiler as _compiler  # type: ignore[import-untyped]
    from cupy.cuda import device as _device  # type: ignore[import-untyped]

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


def upload_fluid_params(params: Optional[np.ndarray] = None) -> None:
    """Upload FluidParams to ``__constant__ FluidParams c_fluid``."""
    if params is None:
        params = build_fluid_params()
    module = _get_module()
    d_ptr = module.get_global("c_fluid")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


def upload_precalc_params(params: Optional[np.ndarray] = None) -> None:
    """Upload PrecalcParams to ``__constant__ PrecalcParams c_precalc``."""
    if params is None:
        params = build_precalc_params()
    module = _get_module()
    d_ptr = module.get_global("c_precalc")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(int(d_ptr), params.ctypes.data, params.nbytes, 1)


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

BLOCK_SIZE = 128


def compute_step2(
    position: cupy.ndarray,
    veleval: cupy.ndarray,
    density: cupy.ndarray,
    behavior_class: cupy.ndarray,
    flags: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
) -> tuple:
    """Launch K_Step2 and return (sph_force, veleval_out).

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Sorted particle positions.
    veleval : cupy.ndarray, (N, 4) float32
        Sorted evaluation velocities.
    density : cupy.ndarray, (N,) float32
        Sorted particle densities (from Step1).
    behavior_class : cupy.ndarray, (N,) int32
        Behavior class per particle (1=FLUID, 2=GRANULAR, 3=STATIC, 4=GAS).
    flags : cupy.ndarray, (N,) uint32
        Particle flags (bit 0 = IS_SLEEPING).
    cell_start : cupy.ndarray, (num_cells,) uint32
        Grid cell start indices (0xFFFFFFFF for empty).
    cell_end : cupy.ndarray, (num_cells,) uint32
        Grid cell end indices.

    Returns
    -------
    sph_force : cupy.ndarray, (N, 4) float32
        Accumulated SPH force per particle.
    veleval_out : cupy.ndarray, (N, 4) float32
        XSPH-corrected evaluation velocity (FLUID only; others unchanged).
    """
    n = position.shape[0]
    sph_force = cupy.zeros((n, 4), dtype=cupy.float32)
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
            veleval,
            density,
            behavior_class,
            flags,
            cell_start,
            cell_end,
            sph_force,
            veleval_out,
        ),
    )

    return sph_force, veleval_out
