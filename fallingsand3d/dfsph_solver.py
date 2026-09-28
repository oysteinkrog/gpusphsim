"""DFSPH solver kernel wrapper -- Divergence-Free SPH (Bender & Koschier, SCA 2015).

Compiles physics/kernels/dfsph_solver.cu via CuPy RawModule and provides
launch functions for the 9 DFSPH kernels: ComputeDensityAlpha, PackVolume,
NonPressureForces, ComputeKappaV, CorrectVelocityDiv, ComputePressureAccel,
CorrectDivAndPressureAccel (the two fused), DensitySolverUpdate and Finalize.

The solver loops read each neighbour's volume from position.w (the volume
carrier), so pack_volume must run after compute_density_alpha (and after
reactions and spawn) and before compute_kappa_v, correct_velocity_div,
compute_pressure_accel and density_solver_update.

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
        ("_pad_iters0", np.float32),
        ("_pad_iters1", np.float32),
        ("warm_start", np.float32),
        ("omega", np.float32),
        ("alpha_limit", np.float32),
        ("kappa_carry", np.float32),
        ("kappa_v_carry", np.float32),
        ("kappa_uncarry", np.float32),
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


def get_module() -> "object":
    """Return the compiled CuPy RawModule (public accessor)."""
    return _get_module()


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
    params[0]["warm_start"] = profile.dfsph_warm_start
    params[0]["omega"] = profile.dfsph_omega
    params[0]["alpha_limit"] = profile.dfsph_alpha_limit
    # K_DFSPH_Finalize scales the warm start as it writes it back
    # (bd-r5-review-n28.173). kappa keeps a nonzero carry so mu(I) can still
    # recover the unscaled p/rho; with warm_start <= 0 the host zeroes p_rho2
    # before the density solver instead.
    ws = float(profile.dfsph_warm_start)
    params[0]["kappa_carry"] = ws if ws > 0.0 else 1.0
    params[0]["kappa_uncarry"] = 1.0 / ws if ws > 0.0 else 1.0
    dws = float(profile.dfsph_div_warm_start)
    params[0]["kappa_v_carry"] = dws if dws > 0.0 else 0.0

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
    vort_out = vorticity_out if vorticity_out is not None else _null
    norm_out = normal_out if normal_out is not None else _null
    kernel(grid, block, (
        np.uint32(n), position, velocity, mass, din,
        packed_info, temperature, cell_start, cell_end,
        density_out, alpha_out, shear_rate_out,
        dTdt_out, exposure_heat_out, exposure_corrode_out,
        vort_out, norm_out,
    ))


def pack_volume(
    position: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
) -> None:
    """Write the volume carrier into position[:, 3] (bd-r5-review-n28.184).

    A regular STATIC wall gets its mass, every other particle
    m / max(rho, RHO_EPSILON). The four solver loops read V_j from there
    instead of loading mass and density per neighbour.
    """
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_PackVolume")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (np.uint32(n), position, density, mass, packed_info))


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
    d_rigid_bodies: "cupy.ndarray | None" = None,
    d_rigid_forces: "cupy.ndarray | None" = None,
    d_rigid_torques: "cupy.ndarray | None" = None,
    p_prev: "cupy.ndarray | None" = None,
) -> None:
    """Viscosity, mu(I) friction and gravity into velocity_out.

    p_prev is the previous substep's converged density-solver value p/rho as
    K_DFSPH_Finalize stored it, already times the warm-start carry
    (w.sorted_kappa). The kernel divides the carry back out, so GRANULAR
    mu(I) friction uses p = p/rho * rho as its confining pressure, floored at
    the one-layer lithostatic pressure. None means floor only.

    velocity and velocity_out must not alias. The DFSPH substep passes
    sorted_velocity in and sorted_veleval out, and every later DFSPH kernel
    reads sorted_veleval as the velocity (bd-r5-review-n28.173).
    """
    n = position.shape[0]
    _null = np.intp(0)
    _null_f = cupy.ndarray(0, dtype=cupy.float32)
    pp_ptr = p_prev if p_prev is not None else _null
    vort_in = vorticity_in if vorticity_in is not None else _null
    norm_in = normal_in if normal_in is not None else _null
    rb_ptr = d_rigid_bodies if d_rigid_bodies is not None else _null_f
    rf_ptr = d_rigid_forces if d_rigid_forces is not None else _null_f
    rt_ptr = d_rigid_torques if d_rigid_torques is not None else _null_f
    module = _get_module()
    kernel = module.get_function("K_DFSPH_NonPressureForces")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), position, velocity, density, mass,
        packed_info, shear_rate, temperature,
        cell_start, cell_end, velocity_out,
        vort_in, norm_in,
        rb_ptr, rf_ptr, rt_ptr,
        pp_ptr,
    ))


def _rigid_ptrs(d_rigid_bodies, d_rigid_forces, d_rigid_torques):
    """Kernel arguments for the rigid buffers; NULL when a buffer is None."""
    null = np.intp(0)
    return (
        d_rigid_bodies if d_rigid_bodies is not None else null,
        d_rigid_forces if d_rigid_forces is not None else null,
        d_rigid_torques if d_rigid_torques is not None else null,
    )


def compute_kappa_v(
    velocity: cupy.ndarray,
    alpha: cupy.ndarray,
    packed_info: cupy.ndarray,
    position: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    kappa_v_out: cupy.ndarray,
    normal_in: "cupy.ndarray | None" = None,
    kappa_v_total: "cupy.ndarray | None" = None,
) -> None:
    """kappa_v_out gets this iteration's increment (what correct_velocity_div applies).
    kappa_v_total, when given, is the running sum across iterations and gets the
    increment added in place; it is the warm start that finalize writes back.

    normal_in is sorted_normal from compute_density_alpha; its .w neighbour count
    gates the solve (fewer than 20 neighbours gives kappa_v = 0). None disables the gate.

    Neighbour volumes come from position[:, 3]; run pack_volume first."""
    n = velocity.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ComputeKappaV")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    norm_in = normal_in if normal_in is not None else np.intp(0)
    kv_total = kappa_v_total if kappa_v_total is not None else np.intp(0)
    kernel(grid, block, (
        np.uint32(n), velocity, alpha,
        packed_info, position, cell_start, cell_end,
        kappa_v_out, norm_in, kv_total,
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
    d_rigid_bodies: "cupy.ndarray | None" = None,
    d_rigid_forces: "cupy.ndarray | None" = None,
    d_rigid_torques: "cupy.ndarray | None" = None,
) -> None:
    """Apply the divergence correction. With rigid bodies, also adds the
    reaction of each fluid-rigid pair to d_rigid_forces / d_rigid_torques.

    Neighbour volumes come from position[:, 3] (run pack_volume first);
    density is read for the own particle and wall neighbours, mass for fluid
    neighbours (symmetric pair form) and the rigid reaction."""
    n = velocity.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_CorrectVelocityDiv")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    rb_ptr, rf_ptr, rt_ptr = _rigid_ptrs(d_rigid_bodies, d_rigid_forces, d_rigid_torques)
    kernel(grid, block, (
        np.uint32(n), velocity, density, mass, kappa_v,
        packed_info, position, cell_start, cell_end,
        rb_ptr, rf_ptr, rt_ptr,
    ))


def compute_pressure_accel(
    p_rho2: cupy.ndarray,
    velocity: cupy.ndarray,
    position: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    vt_out: cupy.ndarray,
    d_rigid_bodies: "cupy.ndarray | None" = None,
    d_rigid_forces: "cupy.ndarray | None" = None,
    d_rigid_torques: "cupy.ndarray | None" = None,
) -> None:
    """Pressure acceleration from p/rho, written as the predicted velocity.

    vt_out gets clamp_v(v + dt * clamp_a(a_press)): |a_press| capped at
    h/dt^2, |vt| at the velocity limit. STATIC, GAS and sleeping particles
    get vt = v. density_solver_update reads vt for i and its neighbours.

    vt_out may be velocity itself (each thread touches only its own index).
    The last call of a substep does that: it applies the pressure to the
    velocity, and only that call gets the rigid buffers, so the kernel adds
    the reaction of each fluid-rigid pair to d_rigid_forces / d_rigid_torques
    once. Neighbour volumes come from position[:, 3]; run pack_volume first.
    """
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_ComputePressureAccel")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    rb_ptr, rf_ptr, rt_ptr = _rigid_ptrs(d_rigid_bodies, d_rigid_forces, d_rigid_torques)
    kernel(grid, block, (
        np.uint32(n), p_rho2, velocity, position, density, mass,
        packed_info, cell_start, cell_end, vt_out,
        rb_ptr, rf_ptr, rt_ptr,
    ))


def correct_div_and_pressure_accel(
    velocity: cupy.ndarray,
    density: cupy.ndarray,
    mass: cupy.ndarray,
    kappa_v: cupy.ndarray,
    p_rho2: cupy.ndarray,
    packed_info: cupy.ndarray,
    position: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    vt_out: cupy.ndarray,
    d_rigid_bodies: "cupy.ndarray | None" = None,
    d_rigid_forces: "cupy.ndarray | None" = None,
    d_rigid_torques: "cupy.ndarray | None" = None,
    pressure_reaction: bool = False,
) -> None:
    """correct_velocity_div(kappa_v) then compute_pressure_accel(p_rho2) in one
    neighbour sweep (bd-r5-review-n28.192).

    velocity gets the divergence-corrected v', vt_out the predicted velocity
    from v', exactly as the two calls in sequence. vt_out may be velocity
    itself (no density iterations follow). With rigid bodies the divergence
    reaction is always added; the pressure reaction only when
    pressure_reaction is True (the last pressure call of the substep).
    Neighbour volumes come from position[:, 3]; run pack_volume first.
    """
    n = position.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_CorrectDivAndPressureAccel")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    rb_ptr, rf_ptr, rt_ptr = _rigid_ptrs(d_rigid_bodies, d_rigid_forces, d_rigid_torques)
    kernel(grid, block, (
        np.uint32(n), velocity, density, mass, kappa_v, p_rho2,
        packed_info, position, cell_start, cell_end, vt_out,
        rb_ptr, rf_ptr, rt_ptr, np.int32(1 if pressure_reaction else 0),
    ))


def density_solver_update(
    vt: cupy.ndarray,
    position: cupy.ndarray,
    density: cupy.ndarray,
    alpha: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    p_rho2: cupy.ndarray,
) -> None:
    """Jacobi update: predict density from vt, update p_rho2.

    vt is the predicted velocity from compute_pressure_accel (already
    clamped). Neighbour volumes come from position[:, 3]; run pack_volume
    first. density is read for the particle's own density only.
    """
    n = vt.shape[0]
    module = _get_module()
    kernel = module.get_function("K_DFSPH_DensitySolverUpdate")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), vt, position, density,
        alpha, packed_info, cell_start, cell_end, p_rho2,
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
    packed_info_out: cupy.ndarray,
    sleep_counter_out: cupy.ndarray,
    temperature_out: cupy.ndarray,
    kappa_out: cupy.ndarray,
    sorted_angular_velocity: "Optional[cupy.ndarray]" = None,
    angular_velocity_out: "Optional[cupy.ndarray]" = None,
    vorticity_in: "cupy.ndarray | None" = None,
    sorted_kappa_v: "cupy.ndarray | None" = None,
    kappa_v_out: "cupy.ndarray | None" = None,
    d_rigid_bodies: "cupy.ndarray | None" = None,
    d_rigid_forces: "cupy.ndarray | None" = None,
    d_rigid_torques: "cupy.ndarray | None" = None,
    sorted_lifetime: "cupy.ndarray | None" = None,
    health_out: "cupy.ndarray | None" = None,
    lifetime_out: "cupy.ndarray | None" = None,
    mass_out: "cupy.ndarray | None" = None,
    max_displacement: "cupy.ndarray | None" = None,
    displacement_anchor: "cupy.ndarray | None" = None,
    cell_wake_flags: "cupy.ndarray | None" = None,
    frame_counter: "cupy.ndarray | None" = None,
) -> None:
    """Run K_DFSPH_Finalize.

    sorted_angular_velocity, angular_velocity_out and vorticity_in are
    ignored: the micropolar angular velocity had no reader and was removed
    (bd-r5-review-n28.140). They stay so callers that still pass them work.
    """
    n = sorted_position.shape[0]
    _null = np.intp(0)
    _null_f = cupy.ndarray(0, dtype=cupy.float32)
    skv_ptr = sorted_kappa_v if sorted_kappa_v is not None else _null
    kv_out_ptr = kappa_v_out if kappa_v_out is not None else np.intp(0)
    rb_ptr = d_rigid_bodies if d_rigid_bodies is not None else _null_f
    rf_ptr = d_rigid_forces if d_rigid_forces is not None else _null_f
    rt_ptr = d_rigid_torques if d_rigid_torques is not None else _null_f
    # Reaction-state carry-back (bd-r4fix-uup.9): kernel guards reads/writes on null
    s_lifetime_ptr = sorted_lifetime if sorted_lifetime is not None else _null
    health_out_ptr = health_out if health_out is not None else np.intp(0)
    lifetime_out_ptr = lifetime_out if lifetime_out is not None else np.intp(0)
    mass_out_ptr = mass_out if mass_out is not None else np.intp(0)
    max_disp_ptr = max_displacement if max_displacement is not None else np.intp(0)
    # Sorted position at the last full sort; NULL measures per-substep motion.
    anchor_ptr = displacement_anchor if displacement_anchor is not None else np.intp(0)
    # Own-cell wake stamp written from registers (bd-r5-review-n28.167).
    # NULL table: no mark, so the caller must run K_MarkWakeCells instead.
    wake_ptr = cell_wake_flags if cell_wake_flags is not None else np.intp(0)
    frame_ptr = frame_counter if frame_counter is not None else np.intp(0)
    module = _get_module()
    kernel = module.get_function("K_DFSPH_Finalize")
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)
    kernel(grid, block, (
        np.uint32(n), sorted_position, sorted_velocity,
        sorted_density, sorted_mass, sorted_packed_info,
        sorted_temperature, sorted_health, sorted_dTdt,
        sorted_sleep_counter, sorted_kappa, sort_indexes, cell_start, cell_end,
        position_out, velocity_out, packed_info_out,
        sleep_counter_out, temperature_out, kappa_out,
        skv_ptr, kv_out_ptr,
        rb_ptr, rf_ptr, rt_ptr,
        s_lifetime_ptr, health_out_ptr, lifetime_out_ptr, mass_out_ptr,
        max_disp_ptr,
        anchor_ptr,
        wake_ptr, frame_ptr,
    ))
