"""Integration test for step1.py -- K_Step1 SPH density summation kernel.

Acceptance criteria:
  - K_Step1 iterates 3x3x3 neighbor cells using cell_start/cell_end
  - density_sum += m_j * (h_sq - r_sq)^3 with PER-PARTICLE mass
  - Final density = max(1.0, poly6_coeff * density_sum)
  - Smoothing length uses squared distance (r_sq < h_sq)
  - Neighbor position loaded via __ldg() (read-only cache)
  - Self-contribution included (j==i NOT skipped)
  - Test: uniform field at rest density -> rho ~= rho0 (within 10%)
  - Test: single isolated particle -> rho = mass * poly6_coeff * h^6
  - Block size = 128, runs without errors for 500K particles

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
    upload_precalc_params,
    upload_sim_params,
)

# Simulation constants matching the acceptance criteria
H = 0.04                # smoothing length
H_SQ = H * H
SPACING = 0.02          # particle spacing
RHO0 = 1000.0           # rest density of water
MASS = RHO0 * SPACING**3  # 0.008 kg
POLY6_COEFF = 315.0 / (64.0 * math.pi * H**9)


def _upload_all_params() -> None:
    """Upload grid, sim, and precalc params to all kernel modules."""
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


def _run_full_pipeline(
    positions_np: np.ndarray,
    mass_np: np.ndarray,
) -> tuple:
    """Run hash -> sort -> build -> step1 and return (density, sorted_mass).

    Parameters
    ----------
    positions_np : (N, 4) float32
    mass_np : (N,) float32

    Returns
    -------
    density_cpu : (N,) float32 -- density in sorted order
    sorted_positions_cpu : (N, 4) float32
    sorted_mass_cpu : (N,) float32
    """
    pos_gpu = cupy.asarray(positions_np)
    mass_gpu = cupy.asarray(mass_np)

    hashes, indices = calc_hash(pos_gpu)
    sorted_hashes, sorted_indices = sort_by_hash(hashes, indices)

    # Reorder position and mass to sorted order
    sorted_pos = pos_gpu[sorted_indices]
    sorted_mass = mass_gpu[sorted_indices]

    cell_start, cell_end = build_data_struct(sorted_hashes)

    density = compute_step1(sorted_pos, sorted_mass, cell_start, cell_end)
    cupy.cuda.Device().synchronize()

    return (
        cupy.asnumpy(density),
        cupy.asnumpy(sorted_pos),
        cupy.asnumpy(sorted_mass),
    )


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
    for sym in ["c_grid", "c_sim", "c_precalc"]:
        d_ptr = module.get_global(sym)  # type: ignore[union-attr]
        assert int(d_ptr) != 0, f"{sym} symbol not found"
    print("[OK] c_grid, c_sim, c_precalc constant memory symbols found")


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

    density, _, _ = _run_full_pipeline(positions, mass_arr)

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
    print("[OK] Single isolated particle density matches mass * poly6_coeff * h^6")


def test_two_particles_within_h() -> None:
    """Two particles within h should have higher density than one alone."""
    print("\n--- Two particles within smoothing length test ---")
    _upload_all_params()

    sep = H * 0.5  # half the smoothing length
    positions = np.zeros((2, 4), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0, 1.0]
    positions[1] = [sep, 0.0, 0.0, 1.0]

    mass_arr = np.full(2, MASS, dtype=np.float32)

    density, _, _ = _run_full_pipeline(positions, mass_arr)

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

    density, _, _ = _run_full_pipeline(positions, mass_arr)

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

    density, _, sorted_mass = _run_full_pipeline(positions, mass_arr)

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

    density, _, _ = _run_full_pipeline(positions, mass_arr)

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

    density, sorted_pos, _ = _run_full_pipeline(positions, mass_arr)

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

    density, _, _ = _run_full_pipeline(positions, mass_arr)

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


def test_500k_no_errors() -> None:
    """Kernel runs without errors for 500K particles."""
    print("\n--- 500K particle stress test ---")
    _upload_all_params()

    rng = np.random.default_rng(42)
    n = 500_000

    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, :3] = rng.uniform(-0.9, 0.9, size=(n, 3)).astype(np.float32)
    positions[:, 3] = 1.0

    mass_arr = np.full(n, MASS, dtype=np.float32)

    density, _, _ = _run_full_pipeline(positions, mass_arr)

    # Basic sanity checks
    assert density.shape == (n,), f"shape {density.shape} != ({n},)"
    assert np.all(density >= 1.0), "Some densities < 1.0 (clamp failed)"
    assert np.all(np.isfinite(density)), "NaN or Inf in densities"

    mean_rho = float(np.mean(density))
    min_rho = float(np.min(density))
    max_rho = float(np.max(density))

    print(f"  500K particles processed")
    print(f"  Density range: [{min_rho:.2f}, {max_rho:.2f}]")
    print(f"  Mean density: {mean_rho:.2f}")
    print("[OK] 500K particles: no CUDA errors, all densities finite and >= 1.0")


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
    test_500k_no_errors()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
