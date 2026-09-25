"""Integration test for step1.py -- K_Step1 density + strain-rate + exposure kernel.

Acceptance criteria (density, US-011):
  - K_Step1 iterates 3x3x3 neighbor cells using cell_start/cell_end
  - density_sum += m_j * (h_sq - r_sq)^3 with PER-PARTICLE mass
  - Final density = max(1.0, poly6_coeff * density_sum)
  - Smoothing length uses squared distance (r_sq < h_sq)
  - Neighbor position loaded via __ldg() (read-only cache)
  - Self-contribution included (j==i NOT skipped)
  - Test: uniform field at rest density -> rho ~= rho0 (within 10%)
  - Test: single isolated particle -> rho = mass * poly6_coeff * h^6
  - Block size = 128, runs without errors for 500K particles

Acceptance criteria (strain-rate, US-015):
  - Step1 accumulates 6 symmetric D components for GRANULAR only
  - D tensor uses spiky gradient (NOT poly6 gradient), m_j/rho_j weighting
  - PostCalc: gamma_dot = sqrt(2 * (Dxx^2 + Dyy^2 + Dzz^2 + 2*(Dxy^2 + Dxz^2 + Dyz^2)))
  - gamma_dot written to sorted shear_rate array
  - Non-GRANULAR particles get shear_rate = 0
  - Test: stationary sand -> gamma_dot ~ 0
  - Test: sand in shear flow -> gamma_dot > 0

Acceptance criteria (exposure, US-021, normalised in bd-r5-review-n28.48):
  - exposure_corrode_i = sum_j c_interactions[mat_i][mat_j].reaction_rate * W_ij / sum_j W_ij
  - exposure_heat_i = sum_j heat_exchange_ij * max(T_j-T_i, 0) * W_ij / sum_j W_ij
  - sum_j W_ij covers every non-self, non-DEAD neighbour in range
  - No if/else branching on material type (pure table lookup via c_interactions)
  - Test: acid near metal -> metal's exposure_corrode > 0
  - Test: fire near wood -> wood's exposure_heat > 0
  - Test: water near water -> exposure_corrode == 0

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np
import pytest

from grid_params import (
    NUM_CELLS,
)
from conftest import counting_sort_scene, default_grid_params
from step1 import (
    BLOCK_SIZE,
    build_precalc_params,
    build_sim_params,
    compute_step1,
    get_module,
    upload_grid_params as upload_step1_grid_params,
    upload_interactions as upload_step1_interactions,
    upload_materials as upload_step1_materials,
    upload_precalc_params,
    upload_sim_params,
)
from materials import build_interaction_matrix, build_material_array

# Simulation constants matching the acceptance criteria
H = 0.04                # smoothing length
H_SQ = H * H
SPACING = 0.02          # particle spacing
RHO0 = 1000.0           # rest density of water
MASS = RHO0 * SPACING**3  # 0.008 kg
POLY6_COEFF = 315.0 / (64.0 * math.pi * H**9)

# packed_info constants
# MAKE_PACKED(mat, beh) = (mat & 0xFF) | ((beh & 0x3) << 8)
FLUID = 0
GRANULAR = 1
PACKED_WATER = 5 | (FLUID << 8)     # material 5, FLUID
PACKED_SAND  = 2 | (GRANULAR << 8)  # material 2, GRANULAR


def _mirror_mass(m: float) -> float:
    """Mass as K_Step1 reads it for a neighbour: FP16(m / MASS_H_UNIT) * MASS_H_UNIT.

    The counting sort packs mass into the FP16 velocity mirror's .w lane
    (bd-r5-review-n28.161), so exact density checks use this rounded value.
    """
    unit = np.float32(0.02)
    return float(np.float32(np.float16(np.float32(m) / unit)) * unit)


def _upload_all_params() -> None:
    """Upload grid, sim, precalc, and materials params to all kernel modules."""
    gp = default_grid_params()
    upload_step1_grid_params(gp)

    sp = build_sim_params(
        smoothing_length=H,
        particle_mass=MASS,
        particle_spacing=SPACING,
    )
    upload_sim_params(sp)

    pp = build_precalc_params(smoothing_length=H)
    upload_precalc_params(pp)

    materials_data = build_material_array()
    upload_step1_materials(materials_data)

    interactions_data = build_interaction_matrix()
    upload_step1_interactions(interactions_data)


def _set_rho_prev(sorted_pos: cupy.ndarray, rho_sorted) -> None:
    """Put the neighbour density K_Step1 reads (position.w) into sorted_pos.

    rho_sorted is a sorted-order array or scalar; None writes 0, which the
    kernel replaces with the neighbour's rest density (bd-r5-review-n28.163).
    K_Step1 rewrites .w during the call, so in a scene larger than one warp a
    neighbour can read the new density instead; see PHYSICS.md 20.1.
    """
    sorted_pos[:, 3] = 0.0 if rho_sorted is None else rho_sorted


def _run_full_pipeline(
    positions_np: np.ndarray,
    mass_np: np.ndarray,
    velocity_np: np.ndarray | None = None,
    packed_info_np: np.ndarray | None = None,
    density_in_np: np.ndarray | None = None,
    temperature_np: np.ndarray | None = None,
) -> tuple:
    """Run counting sort -> step1 and return results.

    Parameters
    ----------
    positions_np : (N, 4) float32
    mass_np : (N,) float32
    velocity_np : (N, 4) float32, optional (default: zeros)
    packed_info_np : (N,) uint32, optional (default: PACKED_WATER)
    density_in_np : (N,) float32, optional
        Previous-substep density, written into position.w for K_Step1 to read
        as rho_j. Default None writes 0, so the kernel uses the rest density.
    temperature_np : (N,) float32, optional (default: 293.0 ambient)

    Returns
    -------
    density_cpu : (N,) float32 -- density in sorted order
    shear_rate_cpu : (N,) float32 -- gamma_dot in sorted order
    dTdt_cpu : (N,) float32 -- temperature rate of change in sorted order
    exposure_heat_cpu : (N,) float32 -- heat exposure in sorted order
    exposure_corrode_cpu : (N,) float32 -- corrosion exposure in sorted order
    sorted_positions_cpu : (N, 4) float32
    sorted_mass_cpu : (N,) float32
    """
    n = positions_np.shape[0]

    if velocity_np is None:
        velocity_np = np.zeros((n, 4), dtype=np.float32)
    if packed_info_np is None:
        packed_info_np = np.full(n, PACKED_WATER, dtype=np.uint32)
    if temperature_np is None:
        temperature_np = np.full(n, 293.0, dtype=np.float32)

    # Same grid build as the app: counting_sort_full (cell_start == cell_end marks an
    # empty cell), with the FP16 velocity mirror the neighbour loop reads.
    scene = counting_sort_scene(
        positions_np,
        velocity=velocity_np,
        mass=mass_np,
        packed_info=packed_info_np,
        temperature=temperature_np,
        grid_params=default_grid_params(),
    )
    sorted_pos = scene["position"]
    sorted_mass = scene["mass"]
    sorted_vel = scene["velocity"]
    sorted_pi = scene["packed_info"]
    sorted_temp = scene["temperature"]
    cell_start = scene["cell_start"]
    cell_end = scene["cell_end"]
    sorted_density_in = None
    if density_in_np is not None:
        perm = scene["sort_perm"].astype(cupy.int64)
        sorted_density_in = cupy.asarray(density_in_np, dtype=cupy.float32)[perm]
    _set_rho_prev(sorted_pos, sorted_density_in)

    # compute_step1 returns 6 values; the 6th (pressure_out) is not used by pipeline
    density, shear_rate, dTdt, exposure_heat, exposure_corrode, _pressure = compute_step1(
        sorted_pos, sorted_vel, sorted_mass,
        sorted_pi,
        sorted_temp,
        cell_start, cell_end,
        velocity_h=scene["velocity_h"],
    )
    cupy.cuda.Device().synchronize()

    return (
        cupy.asnumpy(density),
        cupy.asnumpy(shear_rate),
        cupy.asnumpy(dTdt),
        cupy.asnumpy(exposure_heat),
        cupy.asnumpy(exposure_corrode),
        cupy.asnumpy(sorted_pos),
        cupy.asnumpy(sorted_mass),
    )


# ===================================================================
# Density tests (unchanged acceptance criteria from US-011)
# ===================================================================

def test_compilation() -> None:
    """Verify CuPy RawModule compiles step1.cu."""
    print("--- Compilation check ---")

    module = get_module()
    assert module is not None
    print("[OK] CuPy RawModule compiled step1.cu (includes common.cuh)")

    kernel = module.get_function("K_Step1")  # type: ignore[union-attr]
    assert kernel is not None
    print("[OK] K_Step1 kernel function found")

    # Verify all constant memory symbols accessible
    for sym in ["c_grid", "c_sim", "c_precalc", "c_interactions"]:
        d_ptr = module.get_global(sym)  # type: ignore[union-attr]
        assert int(d_ptr) != 0, f"{sym} symbol not found"
    print("[OK] c_grid, c_sim, c_precalc, c_interactions constant memory symbols found")


def test_block_size() -> None:
    """Verify block size is 256."""
    print("\n--- Block size check ---")
    assert BLOCK_SIZE == 256, f"BLOCK_SIZE = {BLOCK_SIZE}, expected 256"
    print("[OK] BLOCK_SIZE = 256")


def test_struct_sizes() -> None:
    """Verify SimParams and PrecalcParams dtypes match CUDA struct sizes."""
    print("\n--- Struct size check ---")

    sp = build_sim_params()
    pp = build_precalc_params()

    # SimParams: 72 bytes (smoothing_length, smoothing_length_sq, particle_mass,
    # particle_spacing, gravity(3), dt, restitution, wall_friction, world_min(3),
    # world_max(3), velocity_damping, velocity_limit = 18 floats)
    assert sp.nbytes == 72, f"SimParams: {sp.nbytes} != 72 bytes"
    print(f"[OK] SimParams: {sp.nbytes} bytes")

    # PrecalcParams: 20 bytes
    assert pp.nbytes == 20, f"PrecalcParams: {pp.nbytes} != 20 bytes"
    print(f"[OK] PrecalcParams: {pp.nbytes} bytes")


def test_precalc_coefficients() -> None:
    """Verify poly6_coeff matches expected value for h=0.04."""
    print("\n--- Precalc coefficient check ---")

    pp = build_precalc_params(smoothing_length=H)
    poly6 = pp[0]["poly6_coeff"]

    expected = np.float32(315.0 / (64.0 * math.pi * H**9))
    rel_err = abs(float(poly6) - float(expected)) / abs(float(expected))

    print(f"  poly6_coeff = {poly6:.6e} (expected {expected:.6e})")
    assert rel_err < 1e-5, f"poly6_coeff relative error {rel_err:.2e} > 1e-5"
    print("[OK] poly6_coeff matches 315/(64*pi*h^9)")


def test_single_isolated_particle() -> None:
    """Single isolated particle: rho = mass * poly6_coeff * h^6.

    Self-contribution: r_sq = 0, so (h^2 - 0)^3 = h^6.
    density = poly6_coeff * mass * h^6.
    """
    print("\n--- Single isolated particle test ---")
    _upload_all_params()

    # Place particle at center of grid
    positions = np.zeros((1, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]

    mass_arr = np.array([MASS], dtype=np.float32)

    density, shear_rate, _, _, _, _, _ = _run_full_pipeline(positions, mass_arr)

    # poly6_coeff * mass * h^6, with the mass K_Step1 reads from the FP16 mirror
    expected = POLY6_COEFF * _mirror_mass(MASS) * H_SQ**3
    actual = density[0]

    rel_err = abs(actual - expected) / abs(expected)
    print(f"  Expected density: {expected:.6f}")
    print(f"  Actual density:   {actual:.6f}")
    print(f"  Relative error:   {rel_err:.6e}")

    assert rel_err < 1e-4, (
        f"Single particle density mismatch: {actual:.6f} vs {expected:.6f} "
        f"(rel err {rel_err:.6e})"
    )

    # Non-GRANULAR (default WATER/FLUID) -> shear_rate should be 0
    assert shear_rate[0] == 0.0, f"FLUID particle shear_rate = {shear_rate[0]}, expected 0"

    print("[OK] Single isolated particle density matches mass * poly6_coeff * h^6")
    print("[OK] FLUID particle shear_rate = 0")


def test_two_particles_within_h() -> None:
    """Two particles within h should have higher density than one alone."""
    print("\n--- Two particles within smoothing length test ---")
    _upload_all_params()

    sep = H * 0.5  # half the smoothing length
    positions = np.zeros((2, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]
    positions[1] = [sep, 0.0, 0.0, 1.0]

    mass_arr = np.full(2, MASS, dtype=np.float32)

    density, _, _, _, _, _, _ = _run_full_pipeline(positions, mass_arr)

    # Self-contribution only
    single_rho = POLY6_COEFF * MASS * H_SQ**3

    # Both particles should have density > single particle
    assert density[0] > single_rho * 0.99, (
        f"Particle 0 density {density[0]:.4f} not > single {single_rho:.4f}"
    )
    assert density[1] > single_rho * 0.99, (
        f"Particle 1 density {density[1]:.4f} not > single {single_rho:.4f}"
    )

    # By symmetry, densities should be approximately equal
    rel_diff = abs(density[0] - density[1]) / max(density[0], density[1])
    assert rel_diff < 1e-4, (
        f"Asymmetric densities: {density[0]:.6f} vs {density[1]:.6f}"
    )

    print(f"  Single particle density: {single_rho:.6f}")
    print(f"  Particle 0 density:      {density[0]:.6f}")
    print(f"  Particle 1 density:      {density[1]:.6f}")
    print("[OK] Two particles within h produce symmetric density > single particle")


def test_two_particles_beyond_h() -> None:
    """Two particles separated by more than h should not see each other."""
    print("\n--- Two particles beyond smoothing length test ---")
    _upload_all_params()

    sep = H * 1.5  # beyond smoothing length
    positions = np.zeros((2, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]
    positions[1] = [sep, 0.0, 0.0, 1.0]

    mass_arr = np.full(2, MASS, dtype=np.float32)

    density, _, _, _, _, _, _ = _run_full_pipeline(positions, mass_arr)

    single_rho = POLY6_COEFF * _mirror_mass(MASS) * H_SQ**3

    # Both should have same density as single particle (only self-contribution)
    for i in range(2):
        rel_err = abs(density[i] - single_rho) / single_rho
        assert rel_err < 1e-4, (
            f"Particle {i} density {density[i]:.6f} != single {single_rho:.6f} "
            f"(rel err {rel_err:.2e})"
        )

    print(f"  Particle 0 density: {density[0]:.6f}")
    print(f"  Particle 1 density: {density[1]:.6f}")
    print(f"  Expected (self-only): {single_rho:.6f}")
    print("[OK] Particles beyond h have self-contribution-only density")


def test_per_particle_mass() -> None:
    """Two particles with different masses produce different densities."""
    print("\n--- Per-particle mass test ---")
    _upload_all_params()

    sep = H * 0.3  # within smoothing length
    positions = np.zeros((2, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]
    positions[1] = [sep, 0.0, 0.0, 1.0]

    # Particle 0 has 2x mass
    mass_arr = np.array([MASS * 2.0, MASS], dtype=np.float32)

    density, _, _, _, _, _, sorted_mass = _run_full_pipeline(positions, mass_arr)

    # Particle with larger mass should generally produce higher self-density
    # But both particles contribute to each other, and mass affects contribution
    # The key thing: densities differ due to different masses
    assert density[0] != density[1], (
        f"Expected different densities but got {density[0]} == {density[1]}"
    )

    print(f"  Mass[0]={sorted_mass[0]:.6f}, density[0]={density[0]:.6f}")
    print(f"  Mass[1]={sorted_mass[1]:.6f}, density[1]={density[1]:.6f}")
    print("[OK] Per-particle mass produces different densities")


def test_density_clamp_minimum() -> None:
    """Density is clamped to minimum 1.0."""
    print("\n--- Density clamp test ---")
    _upload_all_params()

    # Use extremely small mass so poly6_coeff * mass * h^6 < 1.0
    tiny_mass = 1e-20  # absurdly small
    positions = np.zeros((1, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]

    mass_arr = np.array([tiny_mass], dtype=np.float32)

    density, _, _, _, _, _, _ = _run_full_pipeline(positions, mass_arr)

    assert density[0] >= 1.0, (
        f"Density {density[0]} < 1.0 (clamp failed)"
    )
    print(f"  Density with tiny mass: {density[0]:.6f}")
    print("[OK] Density clamped to >= 1.0")


def test_uniform_field_rest_density() -> None:
    """Uniform particle field at rest density produces rho ~= rho0 (within 10%).

    Sets up a regular grid of particles with spacing = 0.02 in a small cube,
    with mass = rho0 * spacing^3. Interior particles should have density
    close to rho0.
    """
    print("\n--- Uniform field rest density test ---")
    _upload_all_params()

    # Create a regular grid of particles
    spacing = SPACING
    # Use a region large enough for interior particles to have full neighborhood
    extent = 0.2  # -0.1 to +0.1 in each axis
    coords = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n = xx.size

    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, 0] = xx.ravel()
    positions[:, 1] = yy.ravel()
    positions[:, 2] = zz.ravel()
    positions[:, 3] = 1.0

    mass_arr = np.full(n, MASS, dtype=np.float32)

    density, _, _, _, _, sorted_pos, _ = _run_full_pipeline(positions, mass_arr)

    # Filter to interior particles (at least 2h from edge to avoid boundary effects)
    margin = 2.0 * H  # 0.08
    interior = (
        (sorted_pos[:, 0] > -extent + margin)
        & (sorted_pos[:, 0] < extent - margin)
        & (sorted_pos[:, 1] > -extent + margin)
        & (sorted_pos[:, 1] < extent - margin)
        & (sorted_pos[:, 2] > -extent + margin)
        & (sorted_pos[:, 2] < extent - margin)
    )
    num_interior = int(np.sum(interior))

    if num_interior == 0:
        print(f"  WARNING: no interior particles (n={n}, margin={margin})")
        print("  Skipping rest density check, using overall stats instead")
        mean_rho = float(np.mean(density))
        print(f"  Mean density (all): {mean_rho:.2f}")
        return

    interior_density = density[interior]
    mean_rho = float(np.mean(interior_density))
    min_rho = float(np.min(interior_density))
    max_rho = float(np.max(interior_density))

    rel_err = abs(mean_rho - RHO0) / RHO0

    print(f"  Total particles: {n}")
    print(f"  Interior particles: {num_interior}")
    print(f"  Mean interior density: {mean_rho:.2f} (expected ~{RHO0:.0f})")
    print(f"  Min/Max interior density: {min_rho:.2f} / {max_rho:.2f}")
    print(f"  Relative error: {rel_err:.4f}")

    assert rel_err < 0.10, (
        f"Mean interior density {mean_rho:.2f} differs from rho0={RHO0:.0f} "
        f"by {rel_err*100:.1f}% (> 10%)"
    )
    print("[OK] Uniform field interior density within 10% of rho0")


def test_self_contribution_included() -> None:
    """Verify self-contribution is included by checking r_sq=0 case."""
    print("\n--- Self-contribution included test ---")
    _upload_all_params()

    # Single particle should have nonzero density (from self-contribution)
    positions = np.zeros((1, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]
    mass_arr = np.array([MASS], dtype=np.float32)

    density, _, _, _, _, _, _ = _run_full_pipeline(positions, mass_arr)

    # Without self-contribution, density would be 0 -> clamped to 1.0
    # With self-contribution, density = poly6_coeff * mass * h^6 which is >> 1.0
    expected_self = POLY6_COEFF * MASS * H_SQ**3
    assert density[0] > 1.0, (
        f"Density {density[0]} == 1.0 means self-contribution missing"
    )
    assert density[0] > expected_self * 0.99, (
        f"Density {density[0]:.4f} too low vs expected self-only {expected_self:.4f}"
    )

    print(f"  Density: {density[0]:.6f} (expected self-only: {expected_self:.6f})")
    print("[OK] Self-contribution is included in density computation")


# ===================================================================
# Strain-rate tensor tests (US-015)
# ===================================================================

def test_stationary_sand_zero_gamma_dot() -> None:
    """Stationary GRANULAR particles produce gamma_dot approximately 0.

    All particles at rest (velocity=0), so the velocity gradient tensor D
    should be zero everywhere, and gamma_dot = sqrt(2*D:D) = 0.
    """
    print("\n--- Stationary sand zero gamma_dot test ---")
    _upload_all_params()

    # Regular grid of GRANULAR particles, all at rest
    spacing = SPACING
    extent = 0.1
    coords = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n = xx.size

    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, 0] = xx.ravel()
    positions[:, 1] = yy.ravel()
    positions[:, 2] = zz.ravel()
    positions[:, 3] = 1.0

    velocities = np.zeros((n, 4), dtype=np.float32)  # all at rest
    mass_arr = np.full(n, MASS, dtype=np.float32)
    packed_info = np.full(n, PACKED_SAND, dtype=np.uint32)
    # Provide density_in for m_j/rho_j weighting
    density_in = np.full(n, RHO0, dtype=np.float32)

    density, shear_rate, _, _, _, sorted_pos, _ = _run_full_pipeline(
        positions, mass_arr, velocities, packed_info, density_in,
    )

    # All shear_rates should be approximately 0 (within numerical tolerance)
    max_gamma_dot = float(np.max(np.abs(shear_rate)))
    mean_gamma_dot = float(np.mean(np.abs(shear_rate)))

    print(f"  {n} GRANULAR particles at rest")
    print(f"  Max |gamma_dot|:  {max_gamma_dot:.6e}")
    print(f"  Mean |gamma_dot|: {mean_gamma_dot:.6e}")

    assert max_gamma_dot < 1e-3, (
        f"Stationary particles have gamma_dot = {max_gamma_dot:.6e}, expected ~0"
    )
    print("[OK] Stationary sand particles produce gamma_dot ~ 0")


def test_shear_flow_nonzero_gamma_dot() -> None:
    """Sand particles in shear flow produce gamma_dot > 0.

    Sets up a simple shear flow: v_x = shear_rate * y (linear shear in x-y plane).
    The analytical strain rate D_xy = 0.5 * dv_x/dy = 0.5 * shear_rate_applied.
    gamma_dot = sqrt(2 * 2 * D_xy^2) = sqrt(4 * 0.25 * SR^2) = |SR|.
    """
    print("\n--- Shear flow nonzero gamma_dot test ---")
    _upload_all_params()

    # Regular grid of GRANULAR particles with linear shear velocity
    spacing = SPACING
    extent = 0.1
    coords = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n = xx.size

    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, 0] = xx.ravel()
    positions[:, 1] = yy.ravel()
    positions[:, 2] = zz.ravel()
    positions[:, 3] = 1.0

    # Shear flow: v_x = SR * y, v_y = 0, v_z = 0
    applied_shear_rate = 10.0  # s^-1
    velocities = np.zeros((n, 4), dtype=np.float32)
    velocities[:, 0] = applied_shear_rate * positions[:, 1]  # v_x = SR * y

    mass_arr = np.full(n, MASS, dtype=np.float32)
    packed_info = np.full(n, PACKED_SAND, dtype=np.uint32)
    density_in = np.full(n, RHO0, dtype=np.float32)

    density, shear_rate, _, _, _, sorted_pos, _ = _run_full_pipeline(
        positions, mass_arr, velocities, packed_info, density_in,
    )

    # Interior particles should have gamma_dot > 0
    margin = 2.0 * H
    interior = (
        (sorted_pos[:, 0] > -extent + margin)
        & (sorted_pos[:, 0] < extent - margin)
        & (sorted_pos[:, 1] > -extent + margin)
        & (sorted_pos[:, 1] < extent - margin)
        & (sorted_pos[:, 2] > -extent + margin)
        & (sorted_pos[:, 2] < extent - margin)
    )
    num_interior = int(np.sum(interior))

    if num_interior == 0:
        print(f"  WARNING: no interior particles (n={n}, margin={margin})")
        print("  Using all particles for shear rate check")
        interior_shear = shear_rate
    else:
        interior_shear = shear_rate[interior]

    mean_gamma_dot = float(np.mean(interior_shear))
    max_gamma_dot = float(np.max(interior_shear))
    min_gamma_dot = float(np.min(interior_shear))

    print(f"  {n} GRANULAR particles in shear flow (SR={applied_shear_rate})")
    print(f"  Interior particles: {num_interior}")
    print(f"  Mean gamma_dot: {mean_gamma_dot:.4f}")
    print(f"  Range: [{min_gamma_dot:.4f}, {max_gamma_dot:.4f}]")
    print(f"  Expected (analytical): ~{applied_shear_rate:.1f}")

    # gamma_dot should be significantly > 0
    assert mean_gamma_dot > 1.0, (
        f"Mean gamma_dot = {mean_gamma_dot:.4f} too low for SR={applied_shear_rate}"
    )

    # gamma_dot should be in the right ballpark (within factor of 5 of analytical)
    # SPH approximation won't be exact due to kernel support, discretization, boundary effects
    assert mean_gamma_dot < applied_shear_rate * 5.0, (
        f"Mean gamma_dot = {mean_gamma_dot:.4f} way too high for SR={applied_shear_rate}"
    )

    print("[OK] Sand particles in shear flow produce gamma_dot > 0")


def test_non_granular_zero_shear_rate() -> None:
    """Non-GRANULAR particles get shear_rate = 0 even with velocity gradients."""
    print("\n--- Non-GRANULAR zero shear_rate test ---")
    _upload_all_params()

    n = 100
    rng = np.random.default_rng(123)

    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, :3] = rng.uniform(-0.1, 0.1, size=(n, 3)).astype(np.float32)
    positions[:, 3] = 1.0

    # Give particles random velocities (nonzero velocity gradient)
    velocities = np.zeros((n, 4), dtype=np.float32)
    velocities[:, :3] = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)

    mass_arr = np.full(n, MASS, dtype=np.float32)
    packed_info = np.full(n, PACKED_WATER, dtype=np.uint32)  # FLUID, not GRANULAR

    density, shear_rate, _, _, _, _, _ = _run_full_pipeline(
        positions, mass_arr, velocities, packed_info,
    )

    # All shear_rates should be exactly 0 for FLUID particles
    assert np.all(shear_rate == 0.0), (
        f"FLUID particles have nonzero shear_rate: max={np.max(np.abs(shear_rate)):.6e}"
    )

    print(f"  {n} FLUID particles with random velocities")
    print(f"  All shear_rate = 0 (confirmed)")
    print("[OK] Non-GRANULAR particles get shear_rate = 0")


def test_500k_no_errors() -> None:
    """Kernel runs without errors for 500K particles (mixed FLUID and GRANULAR)."""
    print("\n--- 500K particle stress test ---")
    _upload_all_params()

    rng = np.random.default_rng(42)
    n = 500_000

    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, :3] = rng.uniform(-0.9, 0.9, size=(n, 3)).astype(np.float32)
    positions[:, 3] = 1.0

    velocities = np.zeros((n, 4), dtype=np.float32)
    velocities[:, :3] = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)

    mass_arr = np.full(n, MASS, dtype=np.float32)

    # Mix of FLUID and GRANULAR particles (50/50)
    packed_info = np.full(n, PACKED_WATER, dtype=np.uint32)
    packed_info[n // 2:] = PACKED_SAND

    density_in = np.full(n, RHO0, dtype=np.float32)

    # Random temperatures for stress test
    temperature = rng.uniform(200.0, 1500.0, size=n).astype(np.float32)

    density, shear_rate, dTdt, exposure_heat, exposure_corrode, _, _ = _run_full_pipeline(
        positions, mass_arr, velocities, packed_info, density_in, temperature,
    )

    # Basic sanity checks
    assert density.shape == (n,), f"density shape {density.shape} != ({n},)"
    assert shear_rate.shape == (n,), f"shear_rate shape {shear_rate.shape} != ({n},)"
    assert dTdt.shape == (n,), f"dTdt shape {dTdt.shape} != ({n},)"
    assert exposure_heat.shape == (n,), f"exposure_heat shape {exposure_heat.shape} != ({n},)"
    assert exposure_corrode.shape == (n,), f"exposure_corrode shape {exposure_corrode.shape} != ({n},)"
    assert np.all(density >= 1.0), "Some densities < 1.0 (clamp failed)"
    assert np.all(np.isfinite(density)), "NaN or Inf in densities"
    assert np.all(np.isfinite(shear_rate)), "NaN or Inf in shear_rates"
    assert np.all(shear_rate >= 0.0), "Negative shear_rate values found"
    assert np.all(np.isfinite(dTdt)), "NaN or Inf in dTdt"
    assert np.all(np.isfinite(exposure_heat)), "NaN or Inf in exposure_heat"
    assert np.all(np.isfinite(exposure_corrode)), "NaN or Inf in exposure_corrode"
    assert np.all(exposure_heat >= 0.0), "Negative exposure_heat values found"
    assert np.all(exposure_corrode >= 0.0), "Negative exposure_corrode values found"

    mean_rho = float(np.mean(density))
    min_rho = float(np.min(density))
    max_rho = float(np.max(density))
    mean_sr = float(np.mean(shear_rate))
    max_sr = float(np.max(shear_rate))
    mean_dTdt = float(np.mean(dTdt))
    max_dTdt = float(np.max(np.abs(dTdt)))
    mean_exp_h = float(np.mean(exposure_heat))
    mean_exp_c = float(np.mean(exposure_corrode))

    print(f"  500K particles processed (250K FLUID + 250K GRANULAR)")
    print(f"  Density range: [{min_rho:.2f}, {max_rho:.2f}], mean: {mean_rho:.2f}")
    print(f"  Shear rate: mean={mean_sr:.4f}, max={max_sr:.4f}")
    print(f"  dTdt: mean={mean_dTdt:.4f}, max |dTdt|={max_dTdt:.4f}")
    print(f"  Exposure heat: mean={mean_exp_h:.6e}, corrode: mean={mean_exp_c:.6e}")
    print("[OK] 500K particles: no CUDA errors, all values finite")


# ===================================================================
# Heat diffusion tests (US-020)
# ===================================================================

# Material IDs from materials.py
MAT_WATER = 5   # thermal_conductivity = 0.6
MAT_METAL = 10  # thermal_conductivity = 50.0

def test_hot_particle_among_cold() -> None:
    """Hot particle (1000K) among cold particles (293K):
    dTdt is positive for cold neighbors, negative for hot particle.
    """
    print("\n--- Hot particle among cold: dTdt sign test ---")
    _upload_all_params()

    # Place one hot particle at center, surrounded by cold ones
    # Use a small cluster: 1 hot + neighbors in a regular grid
    spacing = SPACING
    extent = 0.06  # small region so all particles are neighbors
    coords = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n = xx.size

    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, 0] = xx.ravel()
    positions[:, 1] = yy.ravel()
    positions[:, 2] = zz.ravel()
    positions[:, 3] = 1.0

    mass_arr = np.full(n, MASS, dtype=np.float32)

    # All METAL (high thermal conductivity = 50.0)
    packed_metal = MAT_METAL | (FLUID << 8)
    packed_info = np.full(n, packed_metal, dtype=np.uint32)

    # All cold (293K) except the center particle (1000K)
    temperature = np.full(n, 293.0, dtype=np.float32)
    # Find the particle closest to origin
    dists = positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2
    center_idx = np.argmin(dists)
    temperature[center_idx] = 1000.0

    density_in = np.full(n, RHO0, dtype=np.float32)

    density, shear_rate, dTdt, _, _, sorted_pos, _ = _run_full_pipeline(
        positions, mass_arr, None, packed_info, density_in, temperature,
    )

    # Find the hot particle in sorted order (closest to origin)
    sorted_dists = sorted_pos[:, 0]**2 + sorted_pos[:, 1]**2 + sorted_pos[:, 2]**2
    hot_sorted_idx = np.argmin(sorted_dists)

    hot_dTdt = dTdt[hot_sorted_idx]
    cold_dTdt = np.delete(dTdt, hot_sorted_idx)

    print(f"  {n} METAL particles, center=1000K, rest=293K")
    print(f"  Hot particle dTdt: {hot_dTdt:.4f} (should be negative)")
    print(f"  Cold particles dTdt mean: {np.mean(cold_dTdt):.4f} (should be positive)")
    print(f"  Cold particles dTdt range: [{np.min(cold_dTdt):.4f}, {np.max(cold_dTdt):.4f}]")

    # Hot particle loses heat -> dTdt < 0
    assert hot_dTdt < 0.0, f"Hot particle dTdt = {hot_dTdt:.4f}, expected negative"
    # Cold neighbors near hot particle gain heat -> at least some dTdt > 0
    assert np.any(cold_dTdt > 0.0), "No cold particle has positive dTdt"

    print("[OK] Hot particle dTdt negative, cold neighbors dTdt positive")


def test_uniform_temperature_zero_dTdt() -> None:
    """All particles at same temperature -> dTdt should be ~0."""
    print("\n--- Uniform temperature zero dTdt test ---")
    _upload_all_params()

    spacing = SPACING
    extent = 0.08
    coords = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n = xx.size

    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, 0] = xx.ravel()
    positions[:, 1] = yy.ravel()
    positions[:, 2] = zz.ravel()
    positions[:, 3] = 1.0

    mass_arr = np.full(n, MASS, dtype=np.float32)

    packed_water = MAT_WATER | (FLUID << 8)
    packed_info = np.full(n, packed_water, dtype=np.uint32)

    # All at 500K uniform
    temperature = np.full(n, 500.0, dtype=np.float32)
    density_in = np.full(n, RHO0, dtype=np.float32)

    _, _, dTdt, _, _, _, _ = _run_full_pipeline(
        positions, mass_arr, None, packed_info, density_in, temperature,
    )

    max_abs_dTdt = float(np.max(np.abs(dTdt)))
    print(f"  {n} WATER particles at uniform 500K")
    print(f"  Max |dTdt|: {max_abs_dTdt:.6e}")

    assert max_abs_dTdt < 1e-3, (
        f"Uniform temperature dTdt = {max_abs_dTdt:.6e}, expected ~0"
    )
    print("[OK] Uniform temperature produces dTdt ~ 0")


def test_heat_conductivity_material_dependence() -> None:
    """Metal (kappa=50) conducts heat much faster than water (kappa=0.6)."""
    print("\n--- Heat conductivity material dependence test ---")
    _upload_all_params()

    # Two clusters: one metal, one water, same geometry
    spacing = SPACING
    extent = 0.04
    coords = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n_each = xx.size
    n = n_each * 2

    # Metal cluster near origin
    pos_metal = np.zeros((n_each, 4), dtype=np.float32)
    pos_metal[:, 0] = xx.ravel()
    pos_metal[:, 1] = yy.ravel()
    pos_metal[:, 2] = zz.ravel()
    pos_metal[:, 3] = 1.0

    # Water cluster offset far away (no interaction)
    pos_water = np.zeros((n_each, 4), dtype=np.float32)
    pos_water[:, 0] = xx.ravel() + 0.5
    pos_water[:, 1] = yy.ravel()
    pos_water[:, 2] = zz.ravel()
    pos_water[:, 3] = 1.0

    positions = np.vstack([pos_metal, pos_water])
    mass_arr = np.full(n, MASS, dtype=np.float32)

    packed_metal = MAT_METAL | (FLUID << 8)
    packed_water = MAT_WATER | (FLUID << 8)
    packed_info = np.empty(n, dtype=np.uint32)
    packed_info[:n_each] = packed_metal
    packed_info[n_each:] = packed_water

    # Center particle of each cluster is hot, rest cold
    temperature = np.full(n, 293.0, dtype=np.float32)
    # Metal center
    metal_dists = pos_metal[:, 0]**2 + pos_metal[:, 1]**2 + pos_metal[:, 2]**2
    metal_center = np.argmin(metal_dists)
    temperature[metal_center] = 1000.0
    # Water center
    water_dists = (pos_water[:, 0] - 0.5)**2 + pos_water[:, 1]**2 + pos_water[:, 2]**2
    water_center = n_each + np.argmin(water_dists)
    temperature[water_center] = 1000.0

    density_in = np.full(n, RHO0, dtype=np.float32)

    _, _, dTdt, _, _, sorted_pos, _ = _run_full_pipeline(
        positions, mass_arr, None, packed_info, density_in, temperature,
    )

    # Find metal and water hot particles in sorted order
    sorted_metal_dists = sorted_pos[:, 0]**2 + sorted_pos[:, 1]**2 + sorted_pos[:, 2]**2
    sorted_water_dists = (sorted_pos[:, 0] - 0.5)**2 + sorted_pos[:, 1]**2 + sorted_pos[:, 2]**2

    metal_hot_sorted = np.argmin(sorted_metal_dists)
    water_hot_sorted = np.argmin(sorted_water_dists)

    metal_hot_dTdt = abs(dTdt[metal_hot_sorted])
    water_hot_dTdt = abs(dTdt[water_hot_sorted])

    ratio = metal_hot_dTdt / max(water_hot_dTdt, 1e-12)

    print(f"  Metal (kappa=50) hot particle |dTdt|: {metal_hot_dTdt:.4f}")
    print(f"  Water (kappa=0.6) hot particle |dTdt|: {water_hot_dTdt:.4f}")
    print(f"  Ratio: {ratio:.1f}x")

    # Metal should conduct at least 10x faster than water (50/0.6 = 83x theoretically)
    assert metal_hot_dTdt > water_hot_dTdt * 5.0, (
        f"Metal dTdt {metal_hot_dTdt:.4f} not significantly larger than water {water_hot_dTdt:.4f}"
    )
    print("[OK] Metal conducts heat much faster than water")


MAT_LAVA = 7    # thermal_conductivity = 500, heat_capacity = 1000


def _to_original_order(sorted_vals: np.ndarray, sorted_pos: np.ndarray,
                       positions: np.ndarray) -> np.ndarray:
    """Map a sorted-order array back to input order by matching positions."""
    key = lambda p: (round(float(p[0]) / 1e-4), round(float(p[1]) / 1e-4), round(float(p[2]) / 1e-4))
    lookup = {key(p): k for k, p in enumerate(sorted_pos)}
    idx = np.array([lookup[key(p)] for p in positions], dtype=np.int64)
    return sorted_vals[idx]


def _two_material_slab_energy_drift(mat_hot: int, mat_cold: int, steps: int) -> tuple[float, float]:
    """Hot slab of mat_hot (x < 0, 1000 K) against cold mat_cold (x >= 0, 293 K).

    Geometry is fixed, density_in is the self-consistent step1 density, and
    T is advanced with explicit Euler from dTdt. Returns
    (rate imbalance, relative drift of sum m_i cp_i T_i after `steps`).
    The rate imbalance is |sum m_i cp_i dTdt_i| / sum |m_i cp_i dTdt_i| at step 0.
    """
    _upload_all_params()
    cp_table = build_material_array()["heat_capacity"].astype(np.float64)

    coords = np.arange(-0.07, 0.07 + SPACING * 0.5, SPACING, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n = xx.size
    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, 0] = xx.ravel() + 0.5 * SPACING  # no particle on x = 0
    positions[:, 1] = yy.ravel()
    positions[:, 2] = zz.ravel()
    positions[:, 3] = 1.0
    hot = positions[:, 0] < 0.0

    mass_arr = np.full(n, MASS, dtype=np.float32)
    mat = np.where(hot, mat_hot, mat_cold).astype(np.uint32)
    packed_info = (mat | (FLUID << 8)).astype(np.uint32)
    temperature = np.where(hot, 1000.0, 293.0).astype(np.float32)
    mcp = mass_arr.astype(np.float64) * cp_table[mat]

    # Self-consistent density: PostCalc divides by density_out, the neighbour
    # weight uses density_in. Energy balances pairwise only when they match.
    density_s, _, _, _, _, sorted_pos, _ = _run_full_pipeline(
        positions, mass_arr, None, packed_info, None, temperature,
    )
    density_in = _to_original_order(density_s, sorted_pos, positions)

    dt = 1.0e-4
    e0 = float(np.sum(mcp * temperature))
    imbalance = 0.0
    for step in range(steps):
        _, _, dTdt_s, _, _, sorted_pos, _ = _run_full_pipeline(
            positions, mass_arr, None, packed_info, density_in, temperature,
        )
        dTdt = _to_original_order(dTdt_s, sorted_pos, positions).astype(np.float64)
        if step == 0:
            rate = mcp * dTdt
            imbalance = abs(float(np.sum(rate))) / max(float(np.sum(np.abs(rate))), 1e-30)
        temperature = (temperature + dt * dTdt).astype(np.float32)
    e1 = float(np.sum(mcp * temperature.astype(np.float64)))
    return imbalance, abs(e1 - e0) / e0


def test_heat_energy_conserved_single_material() -> None:
    """WATER/WATER slab: pairwise heat exchange conserves sum m cp T."""
    imbalance, drift = _two_material_slab_energy_drift(MAT_WATER, MAT_WATER, steps=100)
    print(f"  WATER/WATER rate imbalance {imbalance:.2e}, energy drift {drift:.2e}")
    assert imbalance < 1e-4, f"WATER/WATER heat rate imbalance {imbalance:.2e}"
    assert drift < 1e-5, f"WATER/WATER energy drift {drift:.2e}"


def test_heat_energy_conserved_lava_water() -> None:
    """LAVA/WATER slab: harmonic-mean k_ij keeps sum m cp T constant.

    With kappa_i applied per particle, lava (500) loses 10x the energy water
    (50) gains, so the rate imbalance is about 0.8 (bd-r5-review-n28.67).
    """
    imbalance, drift = _two_material_slab_energy_drift(MAT_LAVA, MAT_WATER, steps=100)
    print(f"  LAVA/WATER rate imbalance {imbalance:.2e}, energy drift {drift:.2e}")
    assert imbalance < 1e-4, f"LAVA/WATER heat rate imbalance {imbalance:.2e}"
    assert drift < 1e-5, f"LAVA/WATER energy drift {drift:.2e}"


# ===================================================================
# Exposure accumulation tests (US-021)
# ===================================================================

# Material IDs from materials.py
MAT_ACID = 8
MAT_FIRE = 14
MAT_WOOD = 9
MAT_SAND = 2

# Behavior classes
STATIC = 3

# packed_info helpers
PACKED_ACID = MAT_ACID | (FLUID << 8)
PACKED_METAL = MAT_METAL | (STATIC << 8)
PACKED_WOOD = MAT_WOOD | (STATIC << 8)
PACKED_FIRE = 14 | (2 << 8)  # GAS = 2


def test_acid_metal_exposure_corrode() -> None:
    """Acid particle near metal particle -> metal's exposure_corrode > 0.

    c_interactions[METAL][ACID].reaction_rate = 0.3 (from materials.py).
    """
    print("\n--- Acid-metal exposure_corrode test ---")
    _upload_all_params()

    sep = H * 0.3  # well within smoothing length
    positions = np.zeros((2, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]    # metal
    positions[1] = [sep, 0.0, 0.0, 1.0]     # acid

    mass_arr = np.full(2, MASS, dtype=np.float32)

    # Metal at mat_id=10, Acid at mat_id=8
    packed_info = np.array([PACKED_METAL, PACKED_ACID], dtype=np.uint32)
    temperature = np.full(2, 293.0, dtype=np.float32)

    _, _, _, exposure_heat, exposure_corrode, _, _ = _run_full_pipeline(
        positions, mass_arr, None, packed_info, None, temperature,
    )

    print(f"  Metal exposure_corrode: {exposure_corrode[0]:.6e}")
    print(f"  Metal exposure_heat:    {exposure_heat[0]:.6e}")
    print(f"  Acid exposure_corrode:  {exposure_corrode[1]:.6e}")

    # Metal should have nonzero exposure_corrode from acid neighbor
    # c_interactions[10][8].reaction_rate = 0.3
    assert exposure_corrode[0] > 0.0, (
        f"Metal exposure_corrode = {exposure_corrode[0]}, expected > 0"
    )
    # Acid also sees metal: c_interactions[8][10].reaction_rate = 0.3 (symmetric)
    assert exposure_corrode[1] > 0.0, (
        f"Acid exposure_corrode = {exposure_corrode[1]}, expected > 0"
    )

    print("[OK] Acid near metal -> metal's exposure_corrode > 0")


def test_fire_wood_exposure_heat() -> None:
    """Fire particle near wood particle -> wood's exposure_heat > 0.

    Fire is at 1200K, wood at 293K. c_interactions[WOOD][FIRE].heat_exchange = 2.0.
    exposure_heat uses max(T_j - T_i, 0) so wood (cold) gains heat from fire (hot).
    """
    print("\n--- Fire-wood exposure_heat test ---")
    _upload_all_params()

    sep = H * 0.3  # well within smoothing length
    positions = np.zeros((2, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]    # wood
    positions[1] = [sep, 0.0, 0.0, 1.0]     # fire

    mass_arr = np.full(2, MASS, dtype=np.float32)

    packed_info = np.array([PACKED_WOOD, PACKED_FIRE], dtype=np.uint32)
    temperature = np.array([293.0, 1200.0], dtype=np.float32)  # wood cold, fire hot

    _, _, _, exposure_heat, exposure_corrode, _, _ = _run_full_pipeline(
        positions, mass_arr, None, packed_info, None, temperature,
    )

    print(f"  Wood exposure_heat:    {exposure_heat[0]:.6e}")
    print(f"  Wood exposure_corrode: {exposure_corrode[0]:.6e}")
    print(f"  Fire exposure_heat:    {exposure_heat[1]:.6e}")

    # Wood should have nonzero exposure_heat from fire (T_fire > T_wood)
    assert exposure_heat[0] > 0.0, (
        f"Wood exposure_heat = {exposure_heat[0]}, expected > 0 (fire is hotter)"
    )
    # Fire sees max(T_wood - T_fire, 0) = max(293 - 1200, 0) = 0 -> exposure_heat_fire = 0
    assert exposure_heat[1] == 0.0, (
        f"Fire exposure_heat = {exposure_heat[1]}, expected 0 (wood is colder)"
    )

    print("[OK] Fire near wood -> wood's exposure_heat > 0, fire's = 0")


def test_water_water_no_exposure() -> None:
    """Water near water -> exposure_corrode == 0 (reaction_rate is 0 in table).

    c_interactions[WATER][WATER] is all zeros (no self-interaction defined).
    """
    print("\n--- Water-water zero exposure test ---")
    _upload_all_params()

    sep = H * 0.3  # well within smoothing length
    positions = np.zeros((2, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]
    positions[1] = [sep, 0.0, 0.0, 1.0]

    mass_arr = np.full(2, MASS, dtype=np.float32)
    packed_info = np.full(2, PACKED_WATER, dtype=np.uint32)
    temperature = np.full(2, 293.0, dtype=np.float32)

    _, _, _, exposure_heat, exposure_corrode, _, _ = _run_full_pipeline(
        positions, mass_arr, None, packed_info, None, temperature,
    )

    print(f"  Water[0] exposure_corrode: {exposure_corrode[0]:.6e}")
    print(f"  Water[1] exposure_corrode: {exposure_corrode[1]:.6e}")
    print(f"  Water[0] exposure_heat:    {exposure_heat[0]:.6e}")

    # Water-water interaction has reaction_rate = 0
    assert exposure_corrode[0] == 0.0, (
        f"Water exposure_corrode = {exposure_corrode[0]}, expected 0"
    )
    assert exposure_corrode[1] == 0.0, (
        f"Water exposure_corrode = {exposure_corrode[1]}, expected 0"
    )
    # Same temperature -> max(T_j - T_i, 0) = 0 -> exposure_heat = 0
    assert exposure_heat[0] == 0.0, (
        f"Water exposure_heat = {exposure_heat[0]}, expected 0 (same temp)"
    )

    print("[OK] Water near water -> exposure_corrode == 0, exposure_heat == 0")


# ===================================================================
# Exposure normalisation (bd-r5-review-n28.48)
# ===================================================================

MAT_WATER_ID = 5
PACKED_SAND_G = MAT_SAND | (GRANULAR << 8)


def _reaction_define(name: str) -> float:
    """Read a #define threshold from reactions.cu so the test follows the kernel."""
    import re
    path = os.path.join(os.path.dirname(__file__), "physics", "kernels", "reactions.cu")
    with open(path, encoding="utf-8") as f:
        m = re.search(r"#define\s+" + name + r"\s+([0-9.]+)f", f.read())
    assert m, f"{name} not found in reactions.cu"
    return float(m.group(1))


def _exposure_step1(positions, packed_info, temperature):
    """Sort the scene, run K_Step1 and return a one-item list
    [(exposure_heat, exposure_corrode, all exposure_corrode)] for original
    particle 0."""
    _upload_all_params()
    n = positions.shape[0]
    mass = np.full(n, MASS, dtype=np.float32)
    scene = counting_sort_scene(
        positions, velocity=np.zeros((n, 4), np.float32), mass=mass,
        packed_info=packed_info, temperature=temperature,
        grid_params=default_grid_params(),
    )
    k0 = int(np.where(cupy.asnumpy(scene["sort_perm"]) == 0)[0][0])
    _set_rho_prev(scene["position"], None)
    args = (scene["position"], scene["velocity"], scene["mass"],
            scene["packed_info"], scene["temperature"],
            scene["cell_start"], scene["cell_end"])
    out = []
    _, _, _, eh, ec, _ = compute_step1(*args, velocity_h=scene["velocity_h"])
    out.append((float(eh[k0].get()), float(ec[k0].get()), cupy.asnumpy(ec)))
    cupy.cuda.Device().synchronize()
    return out


def _w_var(offsets):
    r_sq = np.sum(np.asarray(offsets, np.float64) ** 2, axis=1)
    return np.where(r_sq <= H_SQ, (H_SQ - r_sq) ** 3, 0.0)


def test_exposure_corrode_is_support_fraction() -> None:
    """SAND with 10 WATER and 20 SAND neighbours inside h: exposure_corrode is
    0.4 * (water share of sum W) within 5%, in K_Step1,
    and no particle exceeds the largest reaction_rate."""
    rng = np.random.default_rng(48)
    offsets = []
    while len(offsets) < 30:
        v = rng.uniform(-0.95 * H, 0.95 * H, 3)
        if 0.3 * H < np.linalg.norm(v) < 0.95 * H:
            offsets.append(v)
    offsets = np.array(offsets)
    n = 31
    pos = np.zeros((n, 4), np.float32)
    pos[:, 3] = 1.0
    pos[1:, :3] = offsets
    packed = np.full(n, PACKED_SAND_G, np.uint32)
    packed[1:11] = PACKED_WATER
    temp = np.full(n, 293.0, np.float32)

    w = _w_var(offsets)
    expected = 0.4 * w[:10].sum() / w.sum()
    max_rate = float(build_interaction_matrix()["reaction_rate"].max())
    for name, (_, exp_c, all_c) in zip(("K_Step1",),
                                       _exposure_step1(pos, packed, temp)):
        print(f"  {name}: exposure_corrode {exp_c:.5f}, expected {expected:.5f}")
        assert abs(exp_c - expected) <= 0.05 * expected, (name, exp_c, expected)
        assert np.all(all_c <= max_rate + 1e-6), (name, all_c.max())
        assert np.all(all_c >= 0.0)
    # Half-water sand sits at the SAND -> WET_SAND threshold, not far past it.
    assert expected < 0.4


def test_exposure_heat_one_fire_neighbour_near_wood_threshold() -> None:
    """WOOD on a 0.5h lattice with one FIRE neighbour at r = h/2: exposure_heat
    equals heat_exchange * dT * W(h/2) / sum W within 5%, and is within an order
    of magnitude of WOOD_IGNITE_EXPOSURE. A FIRE neighbour at 0.85h stays below it."""
    grid = np.arange(-2, 3) * (0.5 * H)
    offsets = np.array([(x, y, z) for x in grid for y in grid for z in grid
                        if 0 < x * x + y * y + z * z < 0.99 * H_SQ])
    assert len(offsets) == 26
    fire_k = int(np.argmin(np.linalg.norm(offsets - [0.5 * H, 0, 0], axis=1)))
    threshold = _reaction_define("WOOD_IGNITE_EXPOSURE")
    heat_exchange = float(build_interaction_matrix()[MAT_WOOD][MAT_FIRE]["heat_exchange"])

    def run(offs, fire_idx):
        n = len(offs) + 1
        pos = np.zeros((n, 4), np.float32)
        pos[:, 3] = 1.0
        pos[1:, :3] = offs
        packed = np.full(n, PACKED_WOOD, np.uint32)
        packed[1 + fire_idx] = PACKED_FIRE
        temp = np.full(n, 293.0, np.float32)
        temp[1 + fire_idx] = 1200.0
        w = _w_var(offs)
        expected = heat_exchange * (1200.0 - 293.0) * w[fire_idx] / w.sum()
        return expected, _exposure_step1(pos, packed, temp)

    expected, results = run(offsets, fire_k)
    for name, (exp_h, _, _) in zip(("K_Step1",), results):
        print(f"  {name}: exposure_heat {exp_h:.1f} K, expected {expected:.1f}, "
              f"threshold {threshold}")
        assert abs(exp_h - expected) <= 0.05 * expected, (name, exp_h, expected)
        assert threshold / 10.0 <= exp_h <= threshold * 10.0
        assert exp_h > threshold

    # Move the FIRE particle out to 0.85h: it now holds about 0.5% of the support.
    far = offsets.copy()
    far[fire_k] = [0.85 * H, 0.0, 0.0]
    expected, results = run(far, fire_k)
    for name, (exp_h, _, _) in zip(("K_Step1",), results):
        assert abs(exp_h - expected) <= 0.05 * expected, (name, exp_h, expected)
        assert exp_h < threshold, (name, exp_h)


# ===================================================================
# Rigid neighbours in the surface normal (bd-r5-review-n28.68)
# ===================================================================

def _wall_contact_normals(wall: str, layers: int):
    """Water block (10 x 6 x 10 at spacing 0.02) on a wall of `layers` planes.

    `wall` is "stone" (STONE, mass = water mass), "rigid" (MAT_RIGID, mass =
    Akinci psi_b from rigid_bodies.compute_boundary_volumes) or "massless"
    (MAT_RIGID with mass 0, which the kernels skip like DEAD slots, so it
    reproduces a wall the normal does not see). Returns the mean normal_out
    (n_x, n_y, n_z, neighbor_count) of the interior contact layer and of the
    interior free-surface top layer, in original order.
    """
    from materials import MAT_RIGID, STATIC, STONE, WATER
    from rigid_bodies import compute_boundary_volumes

    s, rho = 0.02, 2500.0
    m_water = rho * s**3
    water = np.array([(i * s, k * s, j * s) for i in range(10) for k in range(6)
                      for j in range(10)], dtype=np.float32)
    wall_pos = np.array([(i * s, -(l + 1) * s, j * s) for l in range(layers)
                         for i in range(-4, 14) for j in range(-4, 14)], dtype=np.float32)
    if wall == "stone":
        wall_mass = np.full(len(wall_pos), m_water, dtype=np.float32)
        wall_pi = STONE | (STATIC << 8)
    elif wall == "rigid":
        wall_mass = cupy.asnumpy(compute_boundary_volumes(wall_pos, H, rho))
        wall_pi = MAT_RIGID | (STATIC << 8)
    else:
        wall_mass = np.zeros(len(wall_pos), dtype=np.float32)
        wall_pi = MAT_RIGID | (STATIC << 8)

    nw = len(water)
    n = nw + len(wall_pos)
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:nw, :3] = water
    pos[nw:, :3] = wall_pos
    mass = np.concatenate([np.full(nw, m_water, dtype=np.float32), wall_mass])
    packed = np.concatenate([np.full(nw, WATER | (FLUID << 8), dtype=np.uint32),
                             np.full(len(wall_pos), wall_pi, dtype=np.uint32)])

    _upload_all_params()
    scene = counting_sort_scene(
        pos, velocity=np.zeros((n, 4), np.float32), mass=mass, packed_info=packed,
        temperature=np.full(n, 293.0, np.float32), grid_params=default_grid_params(),
    )
    _set_rho_prev(scene["position"], rho)
    normal = cupy.zeros((n, 4), dtype=cupy.float32)
    args = (scene["position"], scene["velocity"], scene["mass"],
            scene["packed_info"], scene["temperature"],
            scene["cell_start"], scene["cell_end"])
    compute_step1(*args, normal_out=normal, velocity_h=scene["velocity_h"])
    cupy.cuda.Device().synchronize()

    out = np.zeros((n, 4), dtype=np.float32)
    out[cupy.asnumpy(scene["sort_perm"]).astype(np.int64)] = cupy.asnumpy(normal)
    grid = out[:nw].reshape(10, 6, 10, 4)
    contact = grid[3:7, 0, 3:7].reshape(-1, 4).mean(axis=0)
    top = grid[3:7, 5, 3:7].reshape(-1, 4).mean(axis=0)
    return contact, top


def test_rigid_wall_contact_layer_matches_stone() -> None:
    """Water on a rigid box face must see its wall like water on STONE.

    Before the fix the rigid wall was invisible to the normal: the contact
    layer had about 21 neighbours (surface, nc < 25) and a normal as large
    as a free surface's, pointing away from the body (anti-wetting). With
    MAT_RIGID counted, the neighbour count matches STONE within 10%, the
    layer is interior, and the normal no longer points away from the wall.
    The rigid normal is not zero like STONE's: psi_b of a one-plane shell is
    about 1.6 V, so the wall side outweighs the fluid side and n points into
    the body at about 0.6 of a free-surface normal. Every consumer gates on
    nc < 25, so an interior particle gets no force from it.
    """
    stone, top = _wall_contact_normals("stone", 3)
    n_free = abs(float(top[1]))
    assert n_free > 0.0 and top[1] < 0.0, top  # normal points into the bulk

    hidden, _ = _wall_contact_normals("massless", 1)
    assert hidden[3] < 25.0, hidden
    assert hidden[1] > 0.9 * n_free, hidden  # the old anti-wetting normal

    for layers in (1, 3):
        rigid, _ = _wall_contact_normals("rigid", layers)
        assert abs(rigid[3] - stone[3]) <= 0.10 * stone[3], (layers, rigid, stone)
        assert rigid[3] >= 25.0, (layers, rigid)
        assert rigid[1] <= 0.10 * n_free, (layers, rigid, n_free)
        assert abs(rigid[1]) < n_free, (layers, rigid, n_free)
        # Lateral components cancel like STONE's.
        assert abs(rigid[0]) <= 0.10 * n_free and abs(rigid[2]) <= 0.10 * n_free, rigid


# ---------------------------------------------------------------------------
# bd-r5-review-n28.69: FP16 pairing rule for pair velocity differences
# ---------------------------------------------------------------------------

def _translating_block_step1(packed: int) -> tuple:
    """Step1 on a 10^3 lattice moving at (1.3, -0.7, 2.1).

    Returns (vorticity, shear_rate) as numpy arrays.
    """

    k = 10
    idx = np.arange(k, dtype=np.float32) * SPACING
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    n = xx.size
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0], pos[:, 1], pos[:, 2] = xx.ravel(), yy.ravel(), zz.ravel()
    pos[:, 3] = 1.0
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, :3] = (1.3, -0.7, 2.1)

    _upload_all_params()
    upload_sim_params(build_sim_params(
        smoothing_length=H, particle_mass=MASS, particle_spacing=SPACING,
    ))
    scene = counting_sort_scene(
        pos, velocity=vel, mass=np.full(n, MASS, dtype=np.float32),
        packed_info=np.full(n, packed, dtype=np.uint32),
        grid_params=default_grid_params(),
    )

    _set_rho_prev(scene["position"], RHO0)
    vort = cupy.zeros((n, 4), dtype=cupy.float32)
    shear = cupy.zeros(n, dtype=cupy.float32)
    args = (scene["position"], scene["velocity"], scene["mass"],
            scene["packed_info"], scene["temperature"],
            scene["cell_start"], scene["cell_end"])
    kw = dict(shear_rate_out=shear, vorticity_out=vort, velocity_h=scene["velocity_h"])
    compute_step1(*args, **kw)
    cupy.cuda.Device().synchronize()
    out = (cupy.asnumpy(vort), cupy.asnumpy(shear))
    _upload_all_params()
    return out


def test_translating_block_zero_vorticity_strain() -> None:
    """A uniformly translating block has exactly zero vorticity and strain
    rate. Neighbour velocity comes from the FP16 mirror; the particle's own
    value must come from the same mirror, or every pair difference is
    fp16(v) - v instead of zero."""
    vort, _ = _translating_block_step1(PACKED_WATER)
    _, shear = _translating_block_step1(PACKED_SAND)
    max_vort = np.abs(vort[:, :3]).max()
    max_shear = np.abs(shear).max()
    print(f"  max |curl v| = {max_vort:.3e}, max gamma_dot = {max_shear:.3e}")
    assert max_vort == 0.0, f"spurious vorticity {max_vort}"
    assert max_shear == 0.0, f"spurious strain rate {max_shear}"


# ---------------------------------------------------------------------------
# bd-r5-review-n28.139: rigid rotation gives curl v = +2 Omega
# ---------------------------------------------------------------------------

def test_rigid_rotation_curl_is_two_omega() -> None:
    """A 10^3 WATER lattice turning as a rigid body, v = Omega x r with
    Omega = (0, 0, 1.5) rad/s. One step1 call must report vorticity.z within
    10% of +2 * Omega for interior particles (full kernel support), and near
    zero x and y components."""

    omega = 1.5
    k = 10
    idx = np.arange(k, dtype=np.float32) * SPACING
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    n = xx.size
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0], pos[:, 1], pos[:, 2] = xx.ravel(), yy.ravel(), zz.ravel()
    pos[:, 3] = 1.0
    c = 0.5 * (k - 1) * SPACING
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = -omega * (pos[:, 1] - c)
    vel[:, 1] = omega * (pos[:, 0] - c)

    _upload_all_params()
    scene = counting_sort_scene(
        pos, velocity=vel, mass=np.full(n, MASS, dtype=np.float32),
        packed_info=np.full(n, PACKED_WATER, dtype=np.uint32),
        grid_params=default_grid_params(),
    )
    _set_rho_prev(scene["position"], RHO0)
    vort = cupy.zeros((n, 4), dtype=cupy.float32)
    args = (scene["position"], scene["velocity"], scene["mass"],
            scene["packed_info"], scene["temperature"],
            scene["cell_start"], scene["cell_end"])
    kw = dict(vorticity_out=vort, velocity_h=scene["velocity_h"])
    compute_step1(*args, **kw)
    cupy.cuda.Device().synchronize()

    v = cupy.asnumpy(vort)
    spos = cupy.asnumpy(scene["position"][:, :3])
    # Interior: at least h = 2 spacings from every face, so the kernel
    # support is fully inside the lattice.
    lo, hi = 2 * SPACING - 1e-4, (k - 3) * SPACING + 1e-4
    interior = np.all((spos >= lo) & (spos <= hi), axis=1)
    assert interior.sum() == (k - 4) ** 3
    wz = v[interior, 2]
    wxy = np.abs(v[interior, :2]).max()
    print(f"  interior omega_z min {wz.min():.4f} "
          f"mean {wz.mean():.4f} max {wz.max():.4f} (expect {2 * omega:.4f}); "
          f"max |omega_xy| {wxy:.3e}")
    assert np.abs(wz - 2.0 * omega).max() < 0.10 * 2.0 * omega, (
        f"omega_z {wz.min():.4f}..{wz.max():.4f}, expected {2 * omega} +- 10%")
    assert wxy < 0.02 * omega, f"spurious in-plane vorticity {wxy:.3e}"


def main() -> None:
    test_compilation()
    test_block_size()
    test_struct_sizes()
    test_precalc_coefficients()
    test_single_isolated_particle()
    test_two_particles_within_h()
    test_two_particles_beyond_h()
    test_per_particle_mass()
    test_density_clamp_minimum()
    test_uniform_field_rest_density()
    test_self_contribution_included()
    test_stationary_sand_zero_gamma_dot()
    test_shear_flow_nonzero_gamma_dot()
    test_non_granular_zero_shear_rate()
    test_hot_particle_among_cold()
    test_uniform_temperature_zero_dTdt()
    test_heat_conductivity_material_dependence()
    test_heat_energy_conserved_single_material()
    test_heat_energy_conserved_lava_water()
    test_acid_metal_exposure_corrode()
    test_fire_wood_exposure_heat()
    test_water_water_no_exposure()
    test_exposure_corrode_is_support_fraction()
    test_exposure_heat_one_fire_neighbour_near_wood_threshold()
    test_rigid_wall_contact_layer_matches_stone()
    test_500k_no_errors()
    print("\n=== ALL CHECKS PASSED ===")


# ---------------------------------------------------------------------------
# bd-r5-review-n28.77: rigid boundary psi_b uses the fluid's own rest density
# ---------------------------------------------------------------------------

def _fluid_rigid_pair_density(fluid_mat: int) -> tuple[float, float, float]:
    """One FLUID particle of `fluid_mat` with one MAT_RIGID neighbour at 0.5 h.

    The rigid particle carries mass RIGID_PSI_RHO0 * V_b, as
    rigid_bodies.inject_boundary_particles stores it. Returns
    (fluid density, fluid self term, W_ij) with W_ij = poly6 at the pair
    distance, all in kernel units (kg/m^3 and 1/m^3).
    """
    from materials import MAT_RIGID, MATERIALS, STATIC
    from rigid_bodies import RIGID_PSI_RHO0

    V_b = 1.5e-5  # arbitrary boundary volume (m^3)
    m_fluid = MATERIALS[fluid_mat].rest_density * SPACING**3
    sep = 0.5 * H
    pos = np.zeros((2, 4), dtype=np.float32)
    pos[1, 0] = sep
    mass = np.array([m_fluid, RIGID_PSI_RHO0 * V_b], dtype=np.float32)
    packed = np.array([fluid_mat | (FLUID << 8), MAT_RIGID | (STATIC << 8)], dtype=np.uint32)

    _upload_all_params()
    scene = counting_sort_scene(
        pos, velocity=np.zeros((2, 4), np.float32), mass=mass, packed_info=packed,
        temperature=np.full(2, 293.0, np.float32), grid_params=default_grid_params(),
    )
    _set_rho_prev(scene["position"], None)
    args = (scene["position"], scene["velocity"], scene["mass"],
            scene["packed_info"], scene["temperature"],
            scene["cell_start"], scene["cell_end"])
    density = compute_step1(*args, velocity_h=scene["velocity_h"])[0]
    cupy.cuda.Device().synchronize()

    perm = cupy.asnumpy(scene["sort_perm"]).astype(np.int64)
    rho = np.zeros(2, dtype=np.float64)
    rho[perm] = cupy.asnumpy(density)
    self_term = POLY6_COEFF * _mirror_mass(m_fluid) * H_SQ**3
    w_ij = POLY6_COEFF * (H_SQ - sep * sep) ** 3
    return float(rho[0]), self_term, w_ij * V_b


@pytest.mark.parametrize("fluid_name,rho0", [
    ("OIL", 2125.0), ("WATER", 2500.0), ("MUD", 4500.0), ("LAVA", 6500.0),
])
def test_rigid_neighbour_density_uses_fluid_rest_density(fluid_name: str, rho0: float) -> None:
    """A rigid neighbour adds rho0_i * V_b * W_ij, with the fluid's own rho0.

    Before the fix it added RIGID_PSI_RHO0 * V_b * W_ij = 2500 * V_b * W_ij
    for every fluid, so OIL (2125) read the wall 1.18x too dense, MUD (4500)
    0.56x and LAVA (6500) 0.38x. WATER must be unchanged.
    """
    import materials
    from materials import MATERIALS
    fluid_mat = getattr(materials, fluid_name)
    assert MATERIALS[fluid_mat].rest_density == rho0
    rho, self_term, vw = _fluid_rigid_pair_density(fluid_mat)
    got = rho - self_term
    want = rho0 * vw
    assert abs(got - want) <= 1e-4 * want, (fluid_mat, got, want, 2500.0 * vw)



# ---------------------------------------------------------------------------
# bd-r5-review-n28.94: GAS conducts heat to non-GAS neighbours, capped per pair
# ---------------------------------------------------------------------------

def _water_fire_pair_dTdt(dt: float) -> tuple[float, float]:
    """WATER at 293 K with one FIRE neighbour at 1200 K at r = h/2.

    Both carry the spawn mass 0.02 and their previous-substep density
    (WATER 2500, FIRE 1000). Returns (dTdt_water, dTdt_fire) from K_Step1
    with SimParams.dt = dt.
    """
    _upload_all_params()
    upload_sim_params(build_sim_params(smoothing_length=H, particle_mass=MASS,
                                       particle_spacing=SPACING, dt=dt))
    pos = np.zeros((2, 4), np.float32)
    pos[:, 3] = 1.0
    pos[1, 0] = 0.5 * H
    packed = np.array([PACKED_WATER, PACKED_FIRE], np.uint32)
    temp = np.array([293.0, 1200.0], np.float32)
    mass = np.full(2, 0.02, np.float32)
    scene = counting_sort_scene(
        pos, velocity=np.zeros((2, 4), np.float32), mass=mass,
        packed_info=packed, temperature=temp, grid_params=default_grid_params(),
    )
    perm = cupy.asnumpy(scene["sort_perm"]).astype(np.int64)
    _set_rho_prev(scene["position"], cupy.asarray(np.array([2500.0, 1000.0], np.float32)[perm]))
    args = (scene["position"], scene["velocity"], scene["mass"],
            scene["packed_info"], scene["temperature"],
            scene["cell_start"], scene["cell_end"])
    dTdt = compute_step1(*args, velocity_h=scene["velocity_h"])[2]
    out = np.zeros(2, np.float64)
    out[perm] = cupy.asnumpy(dTdt)
    # Put the default dt back for the tests that follow.
    upload_sim_params(build_sim_params(smoothing_length=H, particle_mass=MASS,
                                       particle_spacing=SPACING))
    return float(out[0]), float(out[1])


def test_fire_heats_water_across_phase() -> None:
    """One FIRE neighbour heats WATER and cools itself, within 0.5 dT per substep.

    Before bd-r5-review-n28.94 the GAS/non-GAS branch skipped heat, so both
    rates were exactly 0 and fire never warmed water or ice.
    """
    dt = 0.001
    gap = 1200.0 - 293.0
    d_w, d_f = _water_fire_pair_dTdt(dt)
    print(f"  WATER dTdt {d_w:.1f} K/s, FIRE dTdt {d_f:.1f} K/s")
    assert d_w > 0.0, d_w
    assert d_f < 0.0, d_f
    assert d_w * dt < 0.5 * gap
    assert -d_f * dt < 0.5 * gap
    # A fire layer must be able to bring water to a boil in a few seconds:
    # one neighbour at h/2 already gives well over 16 K/s (80 K in 5 s).
    assert d_w > 16.0, d_w


def test_cross_phase_heat_capped_at_half_gap() -> None:
    """With a 10 s step the uncapped pair would overshoot; the cap holds each
    side to exactly 0.5 * dT per step (the isolated pair's density is its
    own self term, which is the bound the cap uses)."""
    dt = 10.0
    gap = 1200.0 - 293.0
    d_w, d_f = _water_fire_pair_dTdt(dt)
    print(f"  WATER dT/step {d_w * dt:.1f} K, "
          f"FIRE dT/step {d_f * dt:.1f} K (cap {0.5 * gap:.1f} K)")
    assert d_w * dt == pytest.approx(0.5 * gap, rel=1e-3)
    assert -d_f * dt == pytest.approx(0.5 * gap, rel=1e-3)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# bd-r5-review-n28.132: neighbour temperature mirror is uint16 fixed point,
# 0.125 K per step, and T_i in pair differences reads the same mirror.
# ---------------------------------------------------------------------------

T_MIRROR_STEP = 0.125  # K per uint16 step in the temperature mirror
MAT_STONE = 1


def _step1_with_temp_mirror(pos, packed_info, temperature, density_in=None):
    """Counting sort (which writes the temperature mirror) then step1 reading it.

    Returns (dTdt, exposure_heat, density, sort_perm) in sorted order.
    """

    n = pos.shape[0]
    scene = counting_sort_scene(
        pos, mass=np.full(n, MASS, dtype=np.float32),
        packed_info=packed_info, temperature=temperature,
        grid_params=default_grid_params(),
    )
    assert scene["temperature_h"].dtype == cupy.uint16
    perm = scene["sort_perm"].astype(cupy.int64)
    _set_rho_prev(scene["position"],
                  None if density_in is None else cupy.asarray(density_in, dtype=cupy.float32)[perm])
    dTdt = cupy.zeros(n, dtype=cupy.float32)
    heat = cupy.zeros(n, dtype=cupy.float32)
    rho = cupy.zeros(n, dtype=cupy.float32)
    args = (scene["position"], scene["velocity"], scene["mass"],
            scene["packed_info"], scene["temperature"],
            scene["cell_start"], scene["cell_end"])
    kw = dict(density_out=rho, dTdt_out=dTdt, exposure_heat_out=heat,
              velocity_h=scene["velocity_h"], temperature_h=scene["temperature_h"])
    compute_step1(*args, **kw)
    return dTdt, heat, rho, scene["sort_perm"]


def _walled_block(offset_x: float, k: int = 8):
    """k^3 lattice whose outer layer is STATIC STONE; returns (pos, is_wall)."""
    idx = np.arange(k, dtype=np.float32) * SPACING
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    pos = np.zeros((xx.size, 4), dtype=np.float32)
    pos[:, 0], pos[:, 1], pos[:, 2] = xx.ravel() + offset_x, yy.ravel(), zz.ravel()
    pos[:, 3] = 1.0
    edge = lambda a: (a == 0) | (a == k - 1)
    ii, jj, kk = np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing="ij")
    wall = (edge(ii) | edge(jj) | edge(kk)).ravel()
    return pos, wall


def test_uniform_off_grid_temperature_zero_dTdt() -> None:
    """WATER at 293.2 K and LAVA at 1499.7 K (neither on the FP16 grid) in
    STATIC STONE walls at the same temperature: every dTdt and exposure_heat
    is exactly zero. Neighbour T comes from the fixed-point mirror, so the
    particle's own T must come from it too."""
    _upload_all_params()
    pos_w, wall_w = _walled_block(0.0)
    pos_l, wall_l = _walled_block(1.0)
    pos = np.vstack([pos_w, pos_l])
    fluid = np.concatenate([
        np.where(wall_w, MAT_STONE | (STATIC << 8), MAT_WATER | (FLUID << 8)),
        np.where(wall_l, MAT_STONE | (STATIC << 8), MAT_LAVA | (FLUID << 8)),
    ]).astype(np.uint32)
    temperature = np.concatenate([
        np.full(pos_w.shape[0], 293.2), np.full(pos_l.shape[0], 1499.7),
    ]).astype(np.float32)
    dTdt, heat, _, _ = _step1_with_temp_mirror(pos, fluid, temperature)
    cupy.cuda.Device().synchronize()
    max_dTdt = float(cupy.abs(dTdt).max())
    max_heat = float(cupy.abs(heat).max())
    print(f"  max |dTdt| = {max_dTdt:.3e}, max exposure_heat = {max_heat:.3e}")
    assert max_dTdt == 0.0, f"uniform off-grid T gives dTdt {max_dTdt}"
    assert max_heat == 0.0, f"uniform off-grid T gives exposure_heat {max_heat}"


def test_temp_mirror_is_fixed_point_eighth_kelvin() -> None:
    """The counting sort writes round(T * 8) as uint16 (0.125 K steps, T_MAX 5000 fits)."""
    temps = np.array([0.0, 293.0, 293.2, 1499.7, 2047.9, 4999.94, 5000.0], dtype=np.float32)
    n = temps.size
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0] = np.arange(n, dtype=np.float32) * 1.0
    pos[:, 3] = 1.0
    scene = counting_sort_scene(pos, temperature=temps, grid_params=default_grid_params())
    perm = cupy.asnumpy(scene["sort_perm"]).astype(np.int64)
    mirror = cupy.asnumpy(scene["temperature_h"])
    expected = np.rint(temps.astype(np.float64) * 8.0).astype(np.uint16)
    np.testing.assert_array_equal(mirror, expected[perm])
    decoded = mirror.astype(np.float64) * T_MIRROR_STEP
    assert np.all(np.abs(decoded - temps[perm]) <= 0.0625 + 1e-4)


def test_heat_energy_conserved_with_temp_mirror() -> None:
    """LAVA (1499.7 K) against WATER (293.2 K) through the temperature mirror:
    sum m cp T stays constant to 1e-6 relative over 1000 substeps. Conduction
    is the only heat term in step1 (cooling lives in integrate)."""
    _upload_all_params()
    cp_table = build_material_array()["heat_capacity"].astype(np.float64)
    coords = np.arange(-0.07, 0.07 + SPACING * 0.5, SPACING, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n = xx.size
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0] = xx.ravel() + 0.5 * SPACING
    pos[:, 1], pos[:, 2] = yy.ravel(), zz.ravel()
    pos[:, 3] = 1.0
    hot = pos[:, 0] < 0.0
    mat = np.where(hot, MAT_LAVA, MAT_WATER).astype(np.uint32)
    packed = (mat | (FLUID << 8)).astype(np.uint32)
    temperature = cupy.asarray(np.where(hot, 1499.7, 293.2).astype(np.float32))
    mcp = cupy.asarray(MASS * cp_table[mat])

    # Self-consistent density (see _two_material_slab_energy_drift).
    _, _, rho_s, perm = _step1_with_temp_mirror(pos, packed, cupy.asnumpy(temperature))
    density_in = cupy.empty_like(rho_s)
    density_in[perm.astype(cupy.int64)] = rho_s

    dt = 1.0e-4
    e0 = float(cupy.sum(mcp * temperature.astype(cupy.float64)))
    for _ in range(1000):
        dTdt_s, _, _, perm = _step1_with_temp_mirror(pos, packed, temperature, density_in)
        dTdt = cupy.empty_like(dTdt_s)
        dTdt[perm.astype(cupy.int64)] = dTdt_s
        temperature = (temperature + cupy.float32(dt) * dTdt).astype(cupy.float32)
    e1 = float(cupy.sum(mcp * temperature.astype(cupy.float64)))
    drift = abs(e1 - e0) / e0
    t_min, t_max = float(temperature.min()), float(temperature.max())
    print(f"  LAVA/WATER via mirror: energy drift {drift:.2e} after 1000 substeps, "
          f"T range [{t_min:.2f}, {t_max:.2f}]")
    # Interior particles stay put (flat mirror field); the interface must move.
    assert t_max < 1499.7 - 0.05, "no conduction happened"
    assert drift < 1e-6, f"energy drift {drift:.2e} through the temperature mirror"


def test_heat_weight_reads_neighbour_density_from_position_w() -> None:
    """bd-r5-review-n28.163: K_Step1 takes rho_j for the m_j / rho_j heat
    weight from the neighbour's position.w. Halving every .w doubles dTdt,
    and .w <= 0 falls back to the rest density."""
    n = 2
    positions = np.zeros((n, 4), dtype=np.float32)
    positions[1, 0] = 0.5 * H
    mass_arr = np.full(n, MASS, dtype=np.float32)
    packed_info = np.full(n, PACKED_WATER, dtype=np.uint32)
    temperature = np.array([293.0, 353.0], dtype=np.float32)

    def dTdt_hot_to_cold(rho: float) -> float:
        _, _, dTdt, _, _, sorted_pos, _ = _run_full_pipeline(
            positions, mass_arr, None, packed_info,
            np.full(n, rho, dtype=np.float32), temperature,
        )
        cold = int(np.argmin(np.abs(sorted_pos[:, 0])))
        return float(dTdt[cold])

    d_rest = dTdt_hot_to_cold(RHO0)
    d_half = dTdt_hot_to_cold(0.5 * RHO0)
    d_zero = dTdt_hot_to_cold(0.0)
    assert d_rest > 0.0
    np.testing.assert_allclose(d_half, 2.0 * d_rest, rtol=1e-5)
    # .w = 0 uses the WATER table rest density, which differs from RHO0.
    from materials import MATERIALS
    rho_table = float(MATERIALS[PACKED_WATER & 0xFF].rest_density)
    np.testing.assert_allclose(d_zero, d_rest * RHO0 / rho_table, rtol=1e-5)
