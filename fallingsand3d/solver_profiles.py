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
    pbf_friction_ratio: float = 0.50  # velocity-space tan(phi_f) for PBF granular DP yield
    pbf_friction_cohesion: float = 0.0  # position-space cohesion for PBF granular (meters)
    pbf_neg_c_scale: float = 0.05  # scale factor for negative constraint (surface cohesion)
    # DFSPH params
    dfsph_div_iters: int = 2       # divergence solver iterations
    dfsph_dens_iters: int = 8      # density solver iterations (Jacobi on pressure)
    dfsph_warm_start: float = 0.5  # warm start factor for kappa (0.5 optimal, >0.8 unstable)
    dfsph_omega: float = 1.0       # relaxation for Jacobi density update
    dfsph_alpha_limit: float = 1.0  # max alpha = alpha_limit * dt^2 (higher = faster convergence)
    # Micropolar params
    micropolar_nu_t: float = 0.1   # micropolar coupling viscosity (angular vel relaxation rate)


PROFILES = {
    # --- Stable profiles (defaults) -- prioritize quality over speed ---
    "WCSPH": SolverProfile(
        name="WCSPH",
        solver_type=SolverType.WCSPH,
        dt=0.001, max_substeps=40, accuracy=0.2,
        force_scale=0.02, xsph_epsilon=0.8,
    ),
    "PBF": SolverProfile(
        name="PBF",
        solver_type=SolverType.PBF,
        dt=1/120, max_substeps=4, fixed_dt=True,
        pbf_iterations=6, pbf_xsph_c=0.05,
    ),
    "DFSPH": SolverProfile(
        name="DFSPH",
        solver_type=SolverType.DFSPH,
        dt=1/300, max_substeps=10, fixed_dt=True,
        dfsph_div_iters=3, dfsph_dens_iters=12, dfsph_omega=1.0,
        dfsph_warm_start=0.5, dfsph_alpha_limit=10.0,
    ),
    # --- Fast profiles -- fewer iterations, looser CFL ---
    "WCSPH (Fast)": SolverProfile(
        name="WCSPH (Fast)",
        solver_type=SolverType.WCSPH,
        dt=0.001, max_substeps=40, accuracy=0.4,
    ),
    "PBF (Fast)": SolverProfile(
        name="PBF (Fast)",
        solver_type=SolverType.PBF,
        dt=1/120, max_substeps=4, fixed_dt=True,
        pbf_iterations=4, pbf_xsph_c=0.05,
    ),
    "DFSPH (Fast)": SolverProfile(
        name="DFSPH (Fast)",
        solver_type=SolverType.DFSPH,
        dt=1/300, max_substeps=10, fixed_dt=True,
        dfsph_div_iters=2, dfsph_dens_iters=8, dfsph_omega=1.0,
        dfsph_warm_start=0.5, dfsph_alpha_limit=5.0,
    ),
}

# Ordered list of profile names for UI combo box (all profiles, stable first)
PROFILE_NAMES = list(PROFILES.keys())

# Safe/stable profile names for validation scripts
SAFE_PROFILE_NAMES = [k for k in PROFILES if "(Fast)" not in k]
