"""Solver profile definitions for multi-solver SPH architecture.

Defines SolverType enum, SolverProfile dataclass, and preset PROFILES dict.
Each profile bundles solver type + timestep + solver-specific parameters.
"""

from enum import IntEnum
from dataclasses import dataclass


class SolverType(IntEnum):
    WCSPH = 0
    PBF = 1
    DFSPH = 2


@dataclass
class SolverProfile:
    name: str
    solver_type: SolverType
    # Timestep
    dt: float = 0.001
    max_substeps: int = 40
    fixed_dt: bool = False
    accuracy: float = 0.4          # CFL number (WCSPH only)
    # WCSPH params
    force_scale: float = 0.02
    xsph_epsilon: float = 0.8
    mu0: float = 1.0
    # PBF params
    pbf_iterations: int = 4
    pbf_relaxation: float = 0.01   # epsilon for denominator stability
    pbf_s_corr_k: float = 0.0     # artificial pressure coefficient (disabled -- too large for small h)
    pbf_s_corr_n: int = 4         # artificial pressure exponent
    pbf_s_corr_dq: float = 0.3    # artificial pressure reference distance (fraction of h)
    pbf_xsph_c: float = 0.01      # XSPH viscosity factor for PBF
    pbf_friction_ratio: float = 0.25  # position-space tan(phi_f) for PBF granular DP yield
    pbf_friction_cohesion: float = 0.0  # position-space cohesion for PBF granular (meters)
    # DFSPH params
    dfsph_div_iters: int = 2       # divergence solver iterations
    dfsph_dens_iters: int = 8      # density solver iterations (Jacobi on pressure)
    dfsph_warm_start: float = 0.5  # warm start factor for kappa (0.5 optimal, >0.8 unstable)
    dfsph_omega: float = 1.0       # relaxation for Jacobi density update
    # Micropolar params
    micropolar_nu_t: float = 0.1   # micropolar coupling viscosity (angular vel relaxation rate)


PROFILES = {
    "WCSPH (Default)": SolverProfile(
        name="WCSPH (Default)",
        solver_type=SolverType.WCSPH,
        dt=0.001, max_substeps=40, accuracy=0.4,
    ),
    "PBF (Position Based)": SolverProfile(
        name="PBF (Position Based)",
        solver_type=SolverType.PBF,
        dt=1/120, max_substeps=4, fixed_dt=True,
        pbf_iterations=4, pbf_xsph_c=0.05,
    ),
    "DFSPH (Div-Free)": SolverProfile(
        name="DFSPH (Div-Free)",
        solver_type=SolverType.DFSPH,
        dt=1/300, max_substeps=10, fixed_dt=True,
        dfsph_div_iters=2, dfsph_dens_iters=8, dfsph_omega=1.0,
        dfsph_warm_start=0.5,
    ),
}

# Ordered list of profile names for UI combo box
PROFILE_NAMES = list(PROFILES.keys())
