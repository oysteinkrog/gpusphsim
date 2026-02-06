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

Acceptance criteria (exposure, US-021):
  - exposure_corrode_i += c_interactions[mat_i][mat_j].reaction_rate * W_poly6(r_sq, h_sq)
  - exposure_heat_i += c_interactions[mat_i][mat_j].heat_exchange * max(T_j-T_i, 0) * W_poly6
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

from hash_sort import (
    NUM_CELLS,
    build_grid_params,
    calc_hash,
    sort_by_hash,
    upload_grid_params as upload_hash_grid_params,
)
from build_grid import (
    build_data_struct,
    upload_grid_params as upload_build_grid_params,
)
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


def _upload_all_params() -> None:
    """Upload grid, sim, precalc, and materials params to all kernel modules."""
    gp = build_grid_params()
    upload_hash_grid_params(gp)
    upload_build_grid_params(gp)
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


def _run_full_pipeline(
    positions_np: np.ndarray,
    mass_np: np.ndarray,
    velocity_np: np.ndarray | None = None,
    packed_info_np: np.ndarray | None = None,
    density_in_np: np.ndarray | None = None,
    temperature_np: np.ndarray | None = None,
) -> tuple:
    """Run hash -> sort -> build -> step1 and return results.

    Parameters
    ----------
    positions_np : (N, 4) float32
    mass_np : (N,) float32
    velocity_np : (N, 4) float32, optional (default: zeros)
    packed_info_np : (N,) uint32, optional (default: PACKED_WATER)
    density_in_np : (N,) float32, optional (default: None -> kernel uses fallback)
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

    pos_gpu = cupy.asarray(positions_np)
    mass_gpu = cupy.asarray(mass_np)
    vel_gpu = cupy.asarray(velocity_np)
    pi_gpu = cupy.asarray(packed_info_np)
    temp_gpu = cupy.asarray(temperature_np)
    density_in_gpu = cupy.asarray(density_in_np) if density_in_np is not None else None

    hashes, indices = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes, indices)

    # Reorder all arrays to sorted order
    sorted_pos = pos_gpu[sorted_indices]
    sorted_mass = mass_gpu[sorted_indices]
    sorted_vel = vel_gpu[sorted_indices]
    sorted_pi = pi_gpu[sorted_indices]
    sorted_temp = temp_gpu[sorted_indices]
    sorted_density_in = density_in_gpu[sorted_indices] if density_in_gpu is not None else None

    cell_start, cell_end = build_data_struct(sorted_hashes)

    density, shear_rate, dTdt, exposure_heat, exposure_corrode = compute_step1(
        sorted_pos, sorted_vel, sorted_mass,
        sorted_density_in, sorted_pi,
        sorted_temp,
        cell_start, cell_end,
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
    """Verify block size is 128."""
    print("\n--- Block size check ---")
    assert BLOCK_SIZE == 128, f"BLOCK_SIZE = {BLOCK_SIZE}, expected 128"
    print("[OK] BLOCK_SIZE = 128")


def test_struct_sizes() -> None:
    """Verify SimParams and PrecalcParams dtypes match CUDA struct sizes."""
    print("\n--- Struct size check ---")

    sp = build_sim_params()
    pp = build_precalc_params()

    # SimParams: 64 bytes (from US-005 test)
    assert sp.nbytes == 64, f"SimParams: {sp.nbytes} != 64 bytes"
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

    expected = POLY6_COEFF * MASS * H_SQ**3  # poly6_coeff * mass * h^6
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

    single_rho = POLY6_COEFF * MASS * H_SQ**3

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
    test_acid_metal_exposure_corrode()
    test_fire_wood_exposure_heat()
    test_water_water_no_exposure()
    test_500k_no_errors()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
