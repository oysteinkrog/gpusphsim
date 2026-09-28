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
    accuracy: float = 0.4          # adaptive dt = 1 ms * accuracy / 0.4, capped at 1 ms (WCSPH only)
    # WCSPH params, uploaded to step2's c_granular at init and on profile switch
    force_scale: float = 0.02
    xsph_epsilon: float = 0.8      # matches step2.DEFAULT_XSPH_EPSILON; see PHYSICS.md 3.4
    mu0: float = 1.0               # GRANULAR base viscosity in c_granular
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
    # DFSPH params. Defaults are the Stable "DFSPH" profile values
    # (bd-r5-review-n28.37), so a custom SolverProfile(solver_type=DFSPH) runs
    # the tuned configuration. dt is shared with WCSPH and is not changed here.
    dfsph_div_iters: int = 3       # divergence solver iterations (2 sweeps each)
    dfsph_dens_iters: int = 4      # density solver iterations (Jacobi on p/rho^2, 2 sweeps each)
    dfsph_warm_start: float = 0.5  # warm start factor for p/rho^2 (0.5 optimal, >0.8 unstable)
    dfsph_div_warm_start: float = 0.5  # warm start factor for kappa_v (divergence solver)
    dfsph_omega: float = 0.7       # Jacobi relaxation, shared by the divergence and density solvers
    dfsph_alpha_limit: float = 1.0  # alpha cap in units of h^2; interior alpha ~ 0.67 h^2, so the cap
                                    # only binds on particles with few neighbours
    # Velocity limit: v_limit = factor * h / dt (CFL-based velocity clamp).
    # None picks the solver default in __post_init__: 0.4 for DFSPH, 0.9 otherwise.
    velocity_limit_factor: float | None = None

    def __post_init__(self) -> None:
        if self.velocity_limit_factor is None:
            self.velocity_limit_factor = 0.4 if self.solver_type == SolverType.DFSPH else 0.9


PROFILES = {
    # --- Stable profiles (defaults) -- prioritize quality over speed ---
    "WCSPH": SolverProfile(
        name="WCSPH",
        solver_type=SolverType.WCSPH,
        dt=0.001, max_substeps=40, accuracy=0.2,  # adaptive dt 0.5 ms
        force_scale=0.02, xsph_epsilon=0.8,
    ),
    "PBF": SolverProfile(
        name="PBF",
        solver_type=SolverType.PBF,
        dt=1/120, max_substeps=4, fixed_dt=True,
        pbf_iterations=6, pbf_xsph_c=0.05,
    ),
    "DFSPH": SolverProfile(
        # 3 div + 4 dens iterations at dt = 1/300: 18 neighbour sweeps per substep,
        # 5400 per simulated second (was 34 and 10200 with 3 + 12 iterations).
        name="DFSPH",
        solver_type=SolverType.DFSPH,
        dt=1/300, max_substeps=10, fixed_dt=True,
        dfsph_div_iters=3, dfsph_dens_iters=4, dfsph_omega=0.7,
        dfsph_warm_start=0.5, dfsph_alpha_limit=1.0,
        velocity_limit_factor=0.4,
    ),
    # --- Fast profiles -- fewer iterations, looser CFL ---
    "WCSPH (Fast)": SolverProfile(
        name="WCSPH (Fast)",
        solver_type=SolverType.WCSPH,
        dt=0.001, max_substeps=40, accuracy=0.4,  # adaptive dt 1 ms
        xsph_epsilon=0.8,
    ),
    "PBF (Fast)": SolverProfile(
        name="PBF (Fast)",
        solver_type=SolverType.PBF,
        dt=1/120, max_substeps=4, fixed_dt=True,
        pbf_iterations=4, pbf_xsph_c=0.05,
    ),
    "DFSPH (Fast)": SolverProfile(
        # 2 div + 3 dens iterations at dt = 1/200: 14 sweeps per substep,
        # 2800 per simulated second. Compression error about 1% instead of 0.35%.
        name="DFSPH (Fast)",
        solver_type=SolverType.DFSPH,
        dt=1/200, max_substeps=7, fixed_dt=True,
        dfsph_div_iters=2, dfsph_dens_iters=3, dfsph_omega=0.7,
        dfsph_warm_start=0.5, dfsph_alpha_limit=1.0,
        velocity_limit_factor=0.4,
    ),
}

# Ordered list of profile names for UI combo box (all profiles, stable first)
PROFILE_NAMES = list(PROFILES.keys())

# Safe/stable profile names for validation scripts
SAFE_PROFILE_NAMES = [k for k in PROFILES if "(Fast)" not in k]
