"""Integration test for step2.py -- K_Step2 pressure, viscosity, XSPH kernel.

Acceptance criteria:
  - Tait EOS: p_raw = k * (pow(rho/rho0, gamma) - 1); GRANULAR clamps >= 0,
    FLUID clamps >= -0.5*k, GAS uses linear k * max(rho - rho0, 0)
  - Pressure force: pressure_precalc * m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky_variable
  - Viscosity force: viscosity_precalc * m_j * (v_j - v_i) / rho_j * lap_visc_variable
  - XSPH (FLUID only): epsilon * sum(m_j / rho_avg * (v_j - v_i) * W_poly6)
  - STATIC (behavior_class == 3) -> zero force (early return)
  - SLEEPING (IS_SLEEPING flag) -> zero force (early return)
  - sph_force as float4 to sorted buffer; veleval_xsph to sorted veleval buffer
  - Two fluid particles at rest density -> ~zero net force
  - Compressed fluid particles -> repulsive forces (pushed apart)
  - Block size = 128, runs for 500K particles

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest
import cupy
import numpy as np

from hash_sort import (
    NUM_CELLS,
    TABLE_SIZE,
)
from conftest import counting_sort_scene, default_grid_params, unsort
from step1 import (
    build_precalc_params,
    build_sim_params,
    upload_grid_params as upload_step1_grid_params,
    upload_precalc_params as upload_step1_precalc,
    upload_sim_params,
)
from step2 import (
    BLOCK_SIZE,
    GRANULAR_PARAMS_DTYPE,
    build_granular_params,
    compute_pressure,
    compute_step2,
    get_module,
    upload_granular_params,
    upload_grid_params as upload_step2_grid_params,
    upload_materials,
    upload_precalc_params as upload_step2_precalc,
    upload_sim_params as upload_step2_sim,
)
from materials import (
    FLUID,
    GRANULAR,
    GAS,
    STATIC,
    MATERIALS,
    MATERIAL_PROPS_DTYPE,
    MAX_MATERIALS,
    build_material_array,
)

# ---------------------------------------------------------------------------
# Constants matching acceptance criteria
# ---------------------------------------------------------------------------

H = 0.04
H_SQ = H * H
SPACING = 0.02
RHO0 = 1000.0
MASS = RHO0 * SPACING**3  # 0.008 kg
VISCOSITY = 3.5
POLY6_COEFF = 315.0 / (64.0 * math.pi * H**9)
PRESSURE_PRECALC = 45.0 / (math.pi * H**6)
VISCOSITY_PRECALC = VISCOSITY * 45.0 / (math.pi * H**6)

# packed_info helpers (mirror common.cuh macros)
def MAKE_PACKED(mat_id: int, behavior: int) -> int:
    return (mat_id & 0xFF) | ((behavior & 0x3) << 8)

def SET_SLEEPING(p: int) -> int:
    return p | 0x400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upload_all_params() -> None:
    """Upload grid, sim, precalc, materials, and granular params to step2 module."""
    gp = default_grid_params()
    upload_step2_grid_params(gp)

    sp = build_sim_params(
        smoothing_length=H,
        particle_mass=MASS,
        particle_spacing=SPACING,
    )
    upload_step2_sim(sp)

    pp = build_precalc_params(smoothing_length=H, viscosity=VISCOSITY)
    upload_step2_precalc(pp)

    # Upload materials
    mat_array = build_material_array()
    upload_materials(mat_array)

    # Upload granular params with base viscosity matching VISCOSITY
    upload_granular_params(build_granular_params(mu0=VISCOSITY))


def _make_two_particle_setup(
    pos_a: list,
    pos_b: list,
    vel_a: list | None = None,
    vel_b: list | None = None,
    density_a: float = 1000.0,
    density_b: float = 1000.0,
    mass_a: float = MASS,
    mass_b: float = MASS,
    mat_id_a: int = 5,   # WATER
    mat_id_b: int = 5,   # WATER
    bclass_a: int = FLUID,
    bclass_b: int = FLUID,
    sleeping_a: bool = False,
    sleeping_b: bool = False,
) -> dict:
    """Create a minimal 2-particle setup with grid cell data.

    Returns dict with all GPU arrays needed for compute_step2.
    Density is packed into position.w as required by the current Step2 API.
    """
    if vel_a is None:
        vel_a = [0.0, 0.0, 0.0]
    if vel_b is None:
        vel_b = [0.0, 0.0, 0.0]

    # Pack density into position.w (Step2 reads rho from pos.w after K_PackDensity)
    position = np.array(
        [pos_a + [density_a], pos_b + [density_b]], dtype=np.float32
    )
    velocity = np.array(
        [vel_a + [0.0], vel_b + [0.0]], dtype=np.float32
    )
    mass = np.array([mass_a, mass_b], dtype=np.float32)

    pi_a = MAKE_PACKED(mat_id_a, bclass_a)
    pi_b = MAKE_PACKED(mat_id_b, bclass_b)
    if sleeping_a:
        pi_a = SET_SLEEPING(pi_a)
    if sleeping_b:
        pi_b = SET_SLEEPING(pi_b)
    packed_info = np.array([pi_a, pi_b], dtype=np.uint32)

    # Shear rate (from Step1); use zeros for setup tests
    shear_rate = np.zeros(2, dtype=np.float32)

    # Same grid build as the app: counting_sort_full. Within one cell the
    # sorted order is not deterministic, so sort_perm is returned and
    # _step2_original_order maps results back to particle a/b order.
    scene = counting_sort_scene(
        position, velocity=velocity, mass=mass, packed_info=packed_info,
        grid_params=default_grid_params(),
    )
    perm = scene["sort_perm"].astype(cupy.int64)
    position_gpu = scene["position"]
    packed_info_gpu = scene["packed_info"]

    # Pre-compute pressure via K_ComputePressure (PERF-007: Step2 reads from pressure_in)
    # Density is packed in position.w; extract as a contiguous array for K_ComputePressure
    density_gpu = cupy.ascontiguousarray(position_gpu[:, 3])
    pressure_gpu = cupy.zeros(len(position), dtype=cupy.float32)
    compute_pressure(density_gpu, packed_info_gpu, pressure_gpu)

    return {
        "position": position_gpu,
        "velocity": scene["velocity"],
        "mass": scene["mass"],
        "packed_info": packed_info_gpu,
        "shear_rate": cupy.asarray(shear_rate)[perm],
        "cell_start": scene["cell_start"],
        "cell_end": scene["cell_end"],
        "pressure_in": pressure_gpu,
        "velocity_h": scene["velocity_h"],
        "sort_perm": scene["sort_perm"],
    }


def _step2_original_order(data: dict) -> tuple:
    """Run compute_step2 on a setup dict; return (force, veleval) in a/b order."""
    args = dict(data)
    perm = args.pop("sort_perm")
    sph_force, veleval_out = compute_step2(**args)
    return unsort(sph_force, perm), unsort(veleval_out, perm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compilation() -> None:
    """Verify CuPy RawModule compiles step2.cu."""
    print("--- Compilation check ---")

    module = get_module()
    assert module is not None
    print("[OK] CuPy RawModule compiled step2.cu (includes common.cuh)")

    kernel = module.get_function("K_Step2")  # type: ignore[union-attr]
    assert kernel is not None
    print("[OK] K_Step2 kernel function found")

    # Verify constant memory symbols
    for sym in ("c_grid", "c_sim", "c_precalc", "c_materials", "c_granular"):
        d_ptr = module.get_global(sym)  # type: ignore[union-attr]
        assert int(d_ptr) != 0, f"Symbol {sym} not found"
    print("[OK] All constant memory symbols found (c_grid, c_sim, c_precalc, c_materials, c_granular)")


def test_struct_sizes() -> None:
    """Verify struct dtype sizes match CUDA."""
    print("\n--- Struct size checks ---")

    # GranularParams grew from 32 to 48 bytes with added fields:
    # vorticity_epsilon, surface_tension_gamma, tan_phi_f, cohesion
    assert GRANULAR_PARAMS_DTYPE.itemsize == 48, (
        f"GranularParams size: {GRANULAR_PARAMS_DTYPE.itemsize} != 48"
    )
    print(f"[OK] sizeof(GranularParams) = {GRANULAR_PARAMS_DTYPE.itemsize}")

    assert BLOCK_SIZE == 256
    print(f"[OK] Block size = {BLOCK_SIZE}")


def test_precalc_values() -> None:
    """Verify precalculated kernel coefficients."""
    print("\n--- Precalc coefficient checks ---")

    pp = build_precalc_params(smoothing_length=H, viscosity=VISCOSITY)

    # pressure_precalc = +45 / (pi * h^6)
    actual_press = float(pp[0]["pressure_precalc"])
    assert abs(actual_press - PRESSURE_PRECALC) / PRESSURE_PRECALC < 1e-5
    assert actual_press > 0, "pressure_precalc must be POSITIVE"
    print(f"[OK] pressure_precalc = {actual_press:.6e} (positive)")

    # viscosity_precalc = mu * 45 / (pi * h^6)
    actual_visc = float(pp[0]["viscosity_precalc"])
    assert abs(actual_visc - VISCOSITY_PRECALC) / VISCOSITY_PRECALC < 1e-5
    print(f"[OK] viscosity_precalc = {actual_visc:.6e} (includes mu={VISCOSITY})")

    # poly6 coefficient = 315 / (64 * pi * h^9)
    actual_poly6 = float(pp[0]["poly6_coeff"])
    assert abs(actual_poly6 - POLY6_COEFF) / POLY6_COEFF < 1e-5
    print(f"[OK] poly6_coeff = {actual_poly6:.6e}")


def test_rest_density_zero_force() -> None:
    """Two fluid particles at rest density produce ~zero net force."""
    print("\n--- Rest density -> ~zero force test ---")

    _upload_all_params()

    sep = H * 0.5
    pos_a = [0.0, 0.0, 0.0]
    pos_b = [sep, 0.0, 0.0]

    data = _make_two_particle_setup(
        pos_a=pos_a,
        pos_b=pos_b,
        density_a=RHO0,
        density_b=RHO0,
    )

    sph_force, veleval_out = _step2_original_order(data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)
    f_mag_0 = np.linalg.norm(forces[0, :3])
    f_mag_1 = np.linalg.norm(forces[1, :3])

    print(f"  Particle 0 force magnitude: {f_mag_0:.6e}")
    print(f"  Particle 1 force magnitude: {f_mag_1:.6e}")

    assert f_mag_0 < 1e-3, f"Expected ~zero force, got {f_mag_0}"
    assert f_mag_1 < 1e-3, f"Expected ~zero force, got {f_mag_1}"
    print("[OK] Two fluid particles at rest density -> approximately zero force")


def test_compressed_repulsive() -> None:
    """Compressed fluid particles produce repulsive forces (pushed apart)."""
    print("\n--- Compressed -> repulsive force test ---")

    _upload_all_params()

    sep = H * 0.3
    pos_a = [0.0, 0.0, 0.0]
    pos_b = [sep, 0.0, 0.0]
    # WATER material (id=5) has rest_density=2500; use 10% above to get positive Tait pressure
    WATER_RHO0 = 2500.0
    compressed_density = WATER_RHO0 * 1.1

    data = _make_two_particle_setup(
        pos_a=pos_a,
        pos_b=pos_b,
        density_a=compressed_density,
        density_b=compressed_density,
    )

    sph_force, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)

    fx_0 = forces[0, 0]
    fx_1 = forces[1, 0]

    print(f"  Particle 0 force: ({forces[0, 0]:.4e}, {forces[0, 1]:.4e}, {forces[0, 2]:.4e})")
    print(f"  Particle 1 force: ({forces[1, 0]:.4e}, {forces[1, 1]:.4e}, {forces[1, 2]:.4e})")

    # Particle 0 at origin, particle 1 at +x
    # For particle 0: r = pos_0 - pos_1 = (-sep, 0, 0) -> force in -x
    # For particle 1: r = pos_1 - pos_0 = (+sep, 0, 0) -> force in +x
    assert fx_0 < 0, f"Particle 0 should be pushed in -x, got fx={fx_0}"
    assert fx_1 > 0, f"Particle 1 should be pushed in +x, got fx={fx_1}"

    # Newton's 3rd law: forces approximately equal and opposite
    assert abs(fx_0 + fx_1) < abs(fx_0) * 0.01, (
        f"Forces not equal/opposite: {fx_0} + {fx_1} = {fx_0 + fx_1}"
    )
    print("[OK] Compressed particles -> repulsive forces (Newton's 3rd law satisfied)")


def test_static_skipped() -> None:
    """STATIC particles (behavior_class == 3) produce zero force."""
    print("\n--- STATIC skip test ---")

    _upload_all_params()

    sep = H * 0.3
    # WATER (id=5) rest_density=2500; use 20% above so Tait EOS gives positive pressure
    WATER_RHO0 = 2500.0
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        density_a=WATER_RHO0 * 1.2,   # STONE (STATIC): pressure doesn't matter (skipped)
        density_b=WATER_RHO0 * 1.2,   # WATER (FLUID): needs density > 2500 for +pressure
        mat_id_a=1,  # STONE
        bclass_a=STATIC,
        mat_id_b=5,  # WATER
        bclass_b=FLUID,
    )

    sph_force, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)

    f_mag_static = np.linalg.norm(forces[0, :3])
    assert f_mag_static == 0.0, f"STATIC particle force = {f_mag_static}, expected 0"
    print("[OK] STATIC particle -> zero force (early return)")

    f_mag_fluid = np.linalg.norm(forces[1, :3])
    assert f_mag_fluid > 0, f"FLUID particle force = {f_mag_fluid}, expected > 0"
    print(f"[OK] FLUID neighbor still gets force: magnitude = {f_mag_fluid:.4e}")


def test_sleeping_skipped() -> None:
    """SLEEPING particles produce zero force."""
    print("\n--- SLEEPING skip test ---")

    _upload_all_params()

    sep = H * 0.3
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        density_a=RHO0 * 1.2,
        density_b=RHO0 * 1.2,
        sleeping_a=True,  # particle 0 is sleeping
    )

    sph_force, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)

    f_mag_sleeping = np.linalg.norm(forces[0, :3])
    assert f_mag_sleeping == 0.0, (
        f"SLEEPING particle force = {f_mag_sleeping}, expected 0"
    )
    print("[OK] SLEEPING particle -> zero force (early return)")


def test_xsph_fluid_only() -> None:
    """XSPH correction applied to FLUID particles only."""
    print("\n--- XSPH (FLUID only) test ---")

    _upload_all_params()

    sep = H * 0.3
    vel_a = [1.0, 0.0, 0.0]
    vel_b = [-1.0, 0.0, 0.0]

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=RHO0,
        density_b=RHO0,
    )

    _, veleval_out = _step2_original_order(data)
    cupy.cuda.Device().synchronize()

    vels = cupy.asnumpy(veleval_out)

    vx_0 = vels[0, 0]
    vx_1 = vels[1, 0]

    print(f"  Original vel: particle 0 = ({vel_a[0]}, 0, 0), particle 1 = ({vel_b[0]}, 0, 0)")
    print(f"  XSPH vel:     particle 0 = ({vx_0:.4f}, {vels[0, 1]:.4f}, {vels[0, 2]:.4f})")
    print(f"  XSPH vel:     particle 1 = ({vx_1:.4f}, {vels[1, 1]:.4f}, {vels[1, 2]:.4f})")

    # Particle 0: v_i=1.0, neighbor v_j=-1.0, XSPH pulls vx toward -1 -> vx < 1.0
    assert vx_0 < 1.0, f"XSPH should reduce vx for particle 0: {vx_0}"
    # Particle 1: v_i=-1.0, neighbor v_j=1.0, XSPH pulls vx toward +1 -> vx > -1.0
    assert vx_1 > -1.0, f"XSPH should increase vx for particle 1: {vx_1}"
    print("[OK] XSPH correction shifts velocities toward neighbors (FLUID)")


def test_xsph_not_for_granular() -> None:
    """XSPH correction is NOT applied to GRANULAR particles."""
    print("\n--- XSPH not for GRANULAR test ---")

    _upload_all_params()

    sep = H * 0.3
    vel_a = [1.0, 0.0, 0.0]
    vel_b = [-1.0, 0.0, 0.0]

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=RHO0,
        density_b=RHO0,
        mat_id_a=2,  # SAND
        mat_id_b=2,  # SAND
        bclass_a=GRANULAR,
        bclass_b=GRANULAR,
    )

    _, veleval_out = _step2_original_order(data)
    cupy.cuda.Device().synchronize()

    vels = cupy.asnumpy(veleval_out)

    # GRANULAR should keep original velocity (no XSPH)
    assert abs(vels[0, 0] - vel_a[0]) < 1e-6, (
        f"GRANULAR should keep v_x={vel_a[0]}, got {vels[0, 0]}"
    )
    assert abs(vels[1, 0] - vel_b[0]) < 1e-6, (
        f"GRANULAR should keep v_x={vel_b[0]}, got {vels[1, 0]}"
    )
    print("[OK] GRANULAR particles -> veleval unchanged (no XSPH)")


@pytest.mark.parametrize("sleeping", [True, False], ids=["sleeping", "awake"])
def test_xsph_skips_granular_neighbour(sleeping: bool) -> None:
    """FLUID XSPH ignores GRANULAR neighbours (bd-r5-review-n28.104).

    Water sliding over a sand bed must advect at its own velocity: the XSPH
    sum over a single GRANULAR neighbour is 0, so veleval == v exactly.
    """
    _upload_all_params()

    vel_a = [1.0, 0.0, 0.0]
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[H * 0.3, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=[0.0, 0.0, 0.0],
        density_a=2500.0,
        density_b=2500.0,
        mat_id_a=5,  # WATER
        bclass_a=FLUID,
        mat_id_b=2,  # SAND
        bclass_b=GRANULAR,
        sleeping_b=sleeping,
    )

    _, veleval_out = _step2_original_order(data)
    vels = cupy.asnumpy(veleval_out)

    assert np.array_equal(vels[0, :3], np.array(vel_a, dtype=np.float32)), (
        f"FLUID veleval moved off v by GRANULAR neighbour: {vels[0, :3]}"
    )


def test_viscosity_force() -> None:
    """Viscosity force opposes relative motion between neighbors."""
    print("\n--- Viscosity force test ---")

    _upload_all_params()
    # Override with high viscosity for this test
    pp = build_precalc_params(smoothing_length=H, viscosity=10.0)
    upload_step2_precalc(pp)

    sep = H * 0.3
    vel_a = [2.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=RHO0,
        density_b=RHO0,
    )

    sph_force, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)
    fx_0 = forces[0, 0]
    fx_1 = forces[1, 0]

    print(f"  Particle 0 force: ({forces[0, 0]:.4e}, {forces[0, 1]:.4e}, {forces[0, 2]:.4e})")
    print(f"  Particle 1 force: ({forces[1, 0]:.4e}, {forces[1, 1]:.4e}, {forces[1, 2]:.4e})")

    # At rest density, pressure ~ 0, force dominated by viscosity
    # (v_j - v_i).x = -2 -> fx should be negative for particle 0
    assert fx_0 < 0, f"Viscosity should slow particle 0 (fx < 0), got {fx_0}"
    assert fx_1 > 0, f"Viscosity should accelerate particle 1 (fx > 0), got {fx_1}"
    print("[OK] Viscosity force opposes relative motion")

    # Restore default precalc
    pp = build_precalc_params(smoothing_length=H, viscosity=VISCOSITY)
    upload_step2_precalc(pp)


def test_granular_pressure_clamp() -> None:
    """GRANULAR pressure clamps to 0 (no tensile forces)."""
    print("\n--- GRANULAR pressure clamp test ---")

    _upload_all_params()
    # Override with zero viscosity for this test
    pp = build_precalc_params(smoothing_length=H, viscosity=0.0)
    upload_step2_precalc(pp)
    upload_granular_params(build_granular_params(mu0=0.0))

    sep = H * 0.3
    # Density below rest (sand rho0=1600) -> Tait EOS gives negative p_raw
    low_density = 1600.0 * 0.9

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        density_a=low_density,
        density_b=low_density,
        mat_id_a=2,  # SAND
        mat_id_b=2,  # SAND
        bclass_a=GRANULAR,
        bclass_b=GRANULAR,
    )

    sph_force, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)
    f_mag_0 = np.linalg.norm(forces[0, :3])
    f_mag_1 = np.linalg.norm(forces[1, :3])

    print(f"  GRANULAR at low density ({low_density}):")
    print(f"  Particle 0 force magnitude: {f_mag_0:.6e}")
    print(f"  Particle 1 force magnitude: {f_mag_1:.6e}")

    assert f_mag_0 < 1e-3, f"Expected ~zero force for GRANULAR at low density, got {f_mag_0}"
    assert f_mag_1 < 1e-3, f"Expected ~zero force for GRANULAR at low density, got {f_mag_1}"
    print("[OK] GRANULAR at low density -> pressure clamped to 0, ~zero force")

    # Restore defaults
    _upload_all_params()


def test_granular_muI_viscosity() -> None:
    """GRANULAR particles with relative motion get mu(I) enhanced viscosity force."""
    print("\n--- mu(I) viscosity for GRANULAR test ---")

    _upload_all_params()

    sep = H * 0.3
    compressed_density = 1600.0 * 1.05  # Sand rest density * 1.05
    vel_a = [2.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]

    data_granular = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=compressed_density,
        density_b=compressed_density,
        mat_id_a=2,  # SAND
        mat_id_b=2,
        bclass_a=GRANULAR,
        bclass_b=GRANULAR,
    )

    sph_force_gran, _ = _step2_original_order(data_granular)
    cupy.cuda.Device().synchronize()
    forces_gran = cupy.asnumpy(sph_force_gran)

    # Compare with FLUID particles in similar configuration
    data_fluid = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=compressed_density,
        density_b=compressed_density,
        mat_id_a=5,  # WATER
        mat_id_b=5,
        bclass_a=FLUID,
        bclass_b=FLUID,
    )

    sph_force_fluid, _ = _step2_original_order(data_fluid)
    cupy.cuda.Device().synchronize()
    forces_fluid = cupy.asnumpy(sph_force_fluid)

    f_gran_mag = np.linalg.norm(forces_gran[0, :3])
    f_fluid_mag = np.linalg.norm(forces_fluid[0, :3])

    print(f"  GRANULAR force magnitude: {f_gran_mag:.4e}")
    print(f"  FLUID force magnitude:    {f_fluid_mag:.4e}")

    # mu(I) viscosity should produce a different force than constant viscosity
    assert f_gran_mag != f_fluid_mag, (
        f"GRANULAR and FLUID should have different forces: "
        f"gran={f_gran_mag:.4e}, fluid={f_fluid_mag:.4e}"
    )
    assert f_gran_mag > 0, f"Expected non-zero GRANULAR force, got {f_gran_mag}"
    print("[OK] GRANULAR mu(I) produces different viscosity than constant mu0")


def test_granular_muI_harmonic_mean() -> None:
    """Harmonic mean eta_ij is used for GRANULAR-GRANULAR viscosity."""
    print("\n--- mu(I) harmonic mean test ---")

    _upload_all_params()

    sep = H * 0.3
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=[3.0, 0.0, 0.0],
        vel_b=[-1.0, 0.0, 0.0],
        density_a=1600.0 * 1.1,
        density_b=1600.0 * 1.01,
        mat_id_a=2,  # SAND
        mat_id_b=2,
        bclass_a=GRANULAR,
        bclass_b=GRANULAR,
    )

    sph_force, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()
    forces = cupy.asnumpy(sph_force)

    f_mag_0 = np.linalg.norm(forces[0, :3])
    f_mag_1 = np.linalg.norm(forces[1, :3])
    print(f"  Particle 0 force magnitude: {f_mag_0:.4e}")
    print(f"  Particle 1 force magnitude: {f_mag_1:.4e}")

    assert f_mag_0 > 0, f"Expected non-zero force on particle 0, got {f_mag_0}"
    assert f_mag_1 > 0, f"Expected non-zero force on particle 1, got {f_mag_1}"

    assert np.isfinite(forces[0, :3]).all(), "NaN/Inf in particle 0 force"
    assert np.isfinite(forces[1, :3]).all(), "NaN/Inf in particle 1 force"
    print("[OK] Harmonic mean eta_ij produces finite, non-zero forces")


def test_fluid_unchanged_by_muI() -> None:
    """FLUID particles still use constant mu0 viscosity (unchanged by mu(I))."""
    print("\n--- FLUID unchanged by mu(I) test ---")

    _upload_all_params()

    sep = H * 0.3
    vel_a = [2.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=RHO0,
        density_b=RHO0,
    )

    sph_force1, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()
    forces1 = cupy.asnumpy(sph_force1)

    # Run again with very different granular params
    upload_granular_params(build_granular_params(mu_s=0.9, mu_2=0.99, mu_max=99999.0))

    data2 = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=RHO0,
        density_b=RHO0,
    )

    sph_force2, _ = _step2_original_order(data2)
    cupy.cuda.Device().synchronize()
    forces2 = cupy.asnumpy(sph_force2)

    diff = np.abs(forces1 - forces2).max()
    print(f"  Max force difference between runs: {diff:.6e}")
    assert diff < 1e-6, f"FLUID forces changed with granular params: diff={diff}"
    print("[OK] FLUID viscosity is unchanged by mu(I) granular parameters")

    # Restore defaults
    _upload_all_params()


def test_granular_no_nan() -> None:
    """No NaN velocities in GRANULAR forces (mu_max clamp prevents divergence)."""
    print("\n--- GRANULAR no-NaN test ---")

    _upload_all_params()

    sep = H * 0.5
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=[0.0, 0.0, 0.0],
        vel_b=[1e-10, 0.0, 0.0],  # near-zero relative motion
        density_a=1600.0 * 1.1,
        density_b=1600.0 * 1.1,
        mat_id_a=2,  # SAND
        mat_id_b=2,
        bclass_a=GRANULAR,
        bclass_b=GRANULAR,
    )

    sph_force, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()
    forces = cupy.asnumpy(sph_force)

    assert np.isfinite(forces).all(), f"NaN/Inf in granular forces: {forces}"
    print(f"  Force magnitudes: [{np.linalg.norm(forces[0, :3]):.4e}, {np.linalg.norm(forces[1, :3]):.4e}]")
    print("[OK] No NaN/Inf for near-zero shear rate (mu_max clamp works)")


def test_500k_no_errors() -> None:
    """Kernel runs without errors for 500K particles."""
    print("\n--- 500K particle stress test ---")

    _upload_all_params()

    n = 500_000
    rng = np.random.default_rng(42)

    # Random positions in [-0.5, 0.5]^3
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)

    vel_np = np.zeros((n, 4), dtype=np.float32)
    vel_np[:, :3] = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)

    density_np = rng.uniform(900.0, 1100.0, size=n).astype(np.float32)
    mass_np = np.full(n, MASS, dtype=np.float32)

    # Mix of FLUID and GRANULAR particles
    packed_info_np = np.zeros(n, dtype=np.uint32)
    packed_info_np[:n // 2] = MAKE_PACKED(5, FLUID)     # Water
    packed_info_np[n // 2:] = MAKE_PACKED(2, GRANULAR)   # Sand

    shear_rate_np = np.zeros(n, dtype=np.float32)

    # Pack density into position.w (Step2 reads rho from pos.w)
    pos_np[:, 3] = density_np

    # Same grid build as the app: counting_sort_full + FP16 velocity mirror
    scene = counting_sort_scene(
        pos_np, velocity=vel_np, mass=mass_np, packed_info=packed_info_np,
        grid_params=default_grid_params(),
    )
    pos_gpu = scene["position"]
    pi_gpu = scene["packed_info"]
    cell_start_gpu = scene["cell_start"]
    cell_end_gpu = scene["cell_end"]

    # Pre-compute pressure via K_ComputePressure before calling Step2
    # Density is packed in pos.w; extract as a contiguous array for K_ComputePressure
    density_gpu = cupy.ascontiguousarray(pos_gpu[:, 3])
    pressure_gpu = cupy.zeros(n, dtype=cupy.float32)
    compute_pressure(density_gpu, pi_gpu, pressure_gpu)

    sph_force, veleval_out = compute_step2(
        pos_gpu,
        scene["velocity"],
        scene["mass"],
        pi_gpu,
        cupy.asarray(shear_rate_np),
        cell_start_gpu,
        cell_end_gpu,
        pressure_in=pressure_gpu,
        velocity_h=scene["velocity_h"],
    )
    cupy.cuda.Device().synchronize()

    forces_np = cupy.asnumpy(sph_force)
    assert not np.any(np.isnan(forces_np)), "NaN in sph_force"
    assert not np.any(np.isinf(forces_np)), "Inf in sph_force"

    vels_np = cupy.asnumpy(veleval_out)
    assert not np.any(np.isnan(vels_np)), "NaN in veleval_out"
    assert not np.any(np.isinf(vels_np)), "Inf in veleval_out"

    print("[OK] 500K particles processed without errors")
    print(f"     Force range: [{forces_np[:, :3].min():.4e}, {forces_np[:, :3].max():.4e}]")
    print(f"     Veleval range: [{vels_np[:, :3].min():.4e}, {vels_np[:, :3].max():.4e}]")


# ---------------------------------------------------------------------------
# bd-r5-review-n28.24: GRANULAR boundary viscous drag must carry 1/rho_b
# ---------------------------------------------------------------------------

ACCEL_MAX_GRANULAR = 200.0  # integrate.cu clamp for GRANULAR acceleration


def _granular_drag_x(mat_id_b: int, bclass_b: int, density_b: float,
                     rigid: bool = False) -> tuple:
    """One SAND particle moving at +1 m/s in x next to one still neighbour.

    The neighbour sits along +y, so pressure acts only in y and the x force
    on the sand particle is pure viscous drag. Returns (fx_sand, body_fx).
    """
    from materials import SAND
    _upload_all_params()
    rho0_sand = float(MATERIALS[SAND].rest_density)
    mass = float(MASS)
    # A rigid particle stores mass = RIGID_PSI_RHO0 * V_b and the kernel
    # weights it by rho0_i / RIGID_PSI_RHO0 (bd-r5-review-n28.77). Give it the
    # mass that makes its psi_b for SAND equal MASS, the same weight as the
    # pair reference neighbour.
    mass_b = mass
    if rigid:
        from rigid_bodies import RIGID_PSI_RHO0
        mass_b = mass * RIGID_PSI_RHO0 / rho0_sand
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[0.0, H * 0.5, 0.0],
        vel_a=[1.0, 0.0, 0.0],
        vel_b=[0.0, 0.0, 0.0],
        density_a=rho0_sand,
        density_b=density_b,
        mass_a=mass,
        mass_b=mass_b,
        mat_id_a=SAND,
        mat_id_b=mat_id_b,
        bclass_a=GRANULAR,
        bclass_b=bclass_b,
    )
    body_fx = 0.0
    if rigid:
        from rigid_bodies import RIGID_BODY_DTYPE
        data["d_rigid_bodies"] = cupy.zeros(1, dtype=RIGID_BODY_DTYPE)
        data["d_rigid_forces"] = cupy.zeros(4, dtype=cupy.float32)
        data["d_rigid_torques"] = cupy.zeros(4, dtype=cupy.float32)
    forces, _ = _step2_original_order(data)
    cupy.cuda.Device().synchronize()
    fx = float(cupy.asnumpy(forces)[0, 0])
    if rigid:
        body_fx = float(cupy.asnumpy(data["d_rigid_forces"])[0])
    return fx, body_fx


def _granular_pair_reference_x() -> float:
    """Same geometry, but the neighbour is a still FLUID particle at the
    sand rest density, rescaled to the GRANULAR viscosity mu0.

    The FLUID-GRANULAR pair uses the harmonic mean of mu0 and the FLUID
    dynamic viscosity base_viscosity (bd-r5-review-n28.41, n28.79), so the
    pair drag is multiplied by mu0 / mu_pair to get the mu0 drag.
    """
    from materials import SAND, WATER
    rho0_sand = float(MATERIALS[SAND].rest_density)
    fx, _ = _granular_drag_x(WATER, FLUID, rho0_sand)
    mu_f = float(MATERIALS[WATER].base_viscosity)
    mu_g = VISCOSITY
    mu_pair = 2.0 * mu_f * mu_g / (mu_f + mu_g)
    return fx * mu_g / mu_pair


def test_granular_static_drag_matches_pair() -> None:
    """GRANULAR-STATIC drag equals the GRANULAR pair drag within 10%.

    Before the fix the boundary term divided by rho_i only, so the drag was
    about rho0_sand (~4000x) larger than the pair form with the same mu0.
    """
    from materials import STONE, SAND
    ref = _granular_pair_reference_x()
    rho0_stone = float(MATERIALS[STONE].rest_density)
    fx, _ = _granular_drag_x(STONE, STATIC, rho0_stone)
    print(f"  pair ref fx={ref:.4e}  static fx={fx:.4e}  ratio={fx / ref:.3f}")
    assert ref < 0.0, f"Pair drag should oppose +x motion, got {ref}"
    assert fx < 0.0, f"STATIC drag should oppose +x motion, got {fx}"
    assert abs(fx / ref - 1.0) < 0.10, (
        f"GRANULAR-STATIC drag {fx:.4e} differs from pair drag {ref:.4e} "
        f"by more than 10% (ratio {fx / ref:.3f}); rho0_sand={MATERIALS[SAND].rest_density}"
    )


def test_granular_rigid_drag_matches_pair() -> None:
    """GRANULAR-MAT_RIGID drag equals the GRANULAR pair drag within 10%,
    stays under ACCEL_MAX_GRANULAR, and the body reaction is -m_i * a_i."""
    from materials import MAT_RIGID
    ref = _granular_pair_reference_x()
    fx, body_fx = _granular_drag_x(MAT_RIGID, STATIC, 0.0, rigid=True)
    print(f"  pair ref fx={ref:.4e}  rigid fx={fx:.4e}  body fx={body_fx:.4e}")
    assert fx < 0.0, f"Rigid drag should oppose +x motion, got {fx}"
    assert abs(fx) < ACCEL_MAX_GRANULAR, (
        f"Rigid drag accel {fx:.4e} exceeds ACCEL_MAX_GRANULAR={ACCEL_MAX_GRANULAR}"
    )
    assert abs(fx / ref - 1.0) < 0.10, (
        f"GRANULAR-rigid drag {fx:.4e} differs from pair drag {ref:.4e} "
        f"by more than 10% (ratio {fx / ref:.3f})"
    )
    expected_body = -float(MASS) * fx
    assert abs(body_fx - expected_body) <= 1e-3 * abs(expected_body) + 1e-12, (
        f"Body reaction {body_fx:.4e} != -m_i * a_i = {expected_body:.4e}"
    )


# ---------------------------------------------------------------------------
# bd-r5-review-n28.69: FP16 pairing rule for pair velocity differences
# ---------------------------------------------------------------------------

# Off the FP16 grid in every component: fp16(v) - v is about 1e-4 to 1e-3.
TRANSLATE_VEL = (1.3, -0.7, 2.1)


def _translating_block_step2(mat_id: int) -> tuple:
    """Run step2 on a 10^3 FLUID lattice moving at TRANSLATE_VEL, and at rest.

    Both runs use the same sorted scene and pressure, so any difference in
    the force comes from the velocity terms alone. Returns
    (force_moving, force_rest, veleval_moving), all in sorted order.
    """
    _upload_all_params()
    k = 10
    idx = np.arange(k, dtype=np.float32) * SPACING
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    n = xx.size
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0], pos[:, 1], pos[:, 2] = xx.ravel(), yy.ravel(), zz.ravel()
    pos[:, 3] = RHO0
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, :3] = TRANSLATE_VEL
    scene = counting_sort_scene(
        pos, velocity=vel, mass=np.full(n, MASS, dtype=np.float32),
        packed_info=np.full(n, MAKE_PACKED(mat_id, FLUID), dtype=np.uint32),
        grid_params=default_grid_params(),
    )
    density = cupy.ascontiguousarray(scene["position"][:, 3])
    pressure = cupy.zeros(n, dtype=cupy.float32)
    compute_pressure(density, scene["packed_info"], pressure)

    def run():
        f, ve = compute_step2(
            scene["position"], scene["velocity"], scene["mass"],
            scene["packed_info"], cupy.zeros(n, dtype=cupy.float32),
            scene["cell_start"], scene["cell_end"],
            pressure_in=pressure, velocity_h=scene["velocity_h"],
        )
        return cupy.asnumpy(f), cupy.asnumpy(ve)

    f_moving, ve_moving = run()
    scene["velocity"][:] = 0.0
    scene["velocity_h"][:] = 0
    f_rest, _ = run()
    return f_moving, f_rest, ve_moving


@pytest.mark.parametrize("mat_name", ["WATER", "LAVA"])
def test_translating_block_no_fp16_viscous_bias(mat_name: str) -> None:
    """A uniformly translating block gets exactly zero viscous force and XSPH.

    Neighbour velocities come from the FP16 mirror. Before the fix particle
    i's own velocity was FP32, so every pair difference was fp16(v) - v and
    summed to a body force (about 4.5 m/s^2 for LAVA at 0.5 m/s).
    """
    import materials
    mat_id = getattr(materials, mat_name)
    f_moving, f_rest, ve_moving = _translating_block_step2(mat_id)
    v = np.array(TRANSLATE_VEL, dtype=np.float32)
    dv_force = np.abs(f_moving[:, :3] - f_rest[:, :3]).max()
    dv_xsph = np.abs(ve_moving[:, :3] - v).max()
    print(f"  {mat_name}: max |force(v) - force(0)| = {dv_force:.3e}, "
          f"max |veleval - v| = {dv_xsph:.3e}")
    assert dv_force == 0.0, f"translation changed the force by {dv_force}"
    assert dv_xsph == 0.0, f"XSPH moved veleval off v by {dv_xsph}"


# ---------------------------------------------------------------------------
# Per-material static friction (bd-r5-review-n28.89)
# ---------------------------------------------------------------------------

SAND_ID, DIRT_ID = 2, 3


def _shear_force_y(mu_a: float, mu_b: float) -> float:
    """|f_y| on particle a of a GRANULAR pair sliding in y.

    Slot 3 (DIRT) is overwritten with SAND's row so the two materials differ
    only in friction_coeff. mu_max is lifted so the mu(I) cap does not hide
    the friction difference.
    """
    _upload_all_params()
    mats = build_material_array()
    mats[DIRT_ID] = mats[SAND_ID]
    mats[SAND_ID]["friction_coeff"] = mu_a
    mats[DIRT_ID]["friction_coeff"] = mu_b
    upload_materials(mats)
    upload_granular_params(build_granular_params(mu0=VISCOSITY, mu_max=1e9))
    rho = float(MATERIALS[SAND_ID].rest_density)  # p = 0, so p_eff = p_floor
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[H * 0.3, 0.0, 0.0],
        vel_a=[0.0, 0.5, 0.0],
        vel_b=[0.0, 0.0, 0.0],
        density_a=rho,
        density_b=rho,
        mass_a=rho * SPACING**3,
        mass_b=rho * SPACING**3,
        mat_id_a=SAND_ID,
        mat_id_b=DIRT_ID,
        bclass_a=GRANULAR,
        bclass_b=GRANULAR,
    )
    force, _ = _step2_original_order(data)
    f = cupy.asnumpy(force)
    _upload_all_params()
    assert np.isfinite(f).all()
    return abs(float(f[0, 1]))


def test_granular_friction_coeff_sets_mu_s() -> None:
    """Higher friction_coeff gives a larger mu(I) shear force; a mixed pair
    lies between the two uniform pairs (harmonic mean of eta)."""
    low = _shear_force_y(0.3, 0.3)
    high = _shear_force_y(0.9, 0.9)
    mixed = _shear_force_y(0.3, 0.9)
    print(f"  |f_y|: mu 0.3 {low:.4e}, mu 0.9 {high:.4e}, mixed {mixed:.4e}")
    assert high > 1.5 * low, f"friction_coeff has no effect: low={low}, high={high}"
    assert low < mixed < high, f"mixed pair {mixed} not between {low} and {high}"


def test_granular_friction_coeff_zero_falls_back_to_global_mu_s() -> None:
    """friction_coeff = 0 uses c_granular.mu_s."""
    mu_s = float(build_granular_params()[0]["mu_s"])
    fallback = _shear_force_y(0.0, 0.0)
    explicit = _shear_force_y(mu_s, mu_s)
    np.testing.assert_allclose(fallback, explicit, rtol=1e-5)


# ---------------------------------------------------------------------------
# bd-r5-review-n28.118: regression pins for the round-5 review findings
# ---------------------------------------------------------------------------

APP_MASS = 0.02  # app particle mass: WATER rho0 (2500) * spacing^3


def _water_next_to_static_force(mat_id_b: int, use_nl: bool) -> np.ndarray:
    """Force on a moving, compressed WATER particle with one STATIC neighbour
    of mass 0.02 at 0.3h. Returns the WATER force (x, y, z)."""
    from materials import WATER
    rho0_w = float(MATERIALS[WATER].rest_density)
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[0.3 * H, 0.0, 0.0],
        vel_a=[0.4, -0.3, 0.2],
        density_a=1.2 * rho0_w,
        density_b=1.2 * rho0_w,
        mass_a=APP_MASS,
        mass_b=APP_MASS,
        mat_id_a=WATER,
        mat_id_b=mat_id_b,
        bclass_a=FLUID,
        bclass_b=STATIC,
    )
    perm = data.pop("sort_perm")
    if use_nl:
        import neighbor_list as nl
        from test_cross_force_scale import _upload_nl
        _upload_nl()
        n = data["position"].shape[0]
        nb_idx, nb_cnt = nl.allocate_neighbor_buffers(n)
        nl.build_neighbor_list(
            data["position"], data["cell_start"], data["cell_end"], nb_idx, nb_cnt
        )
        force, _ = nl.compute_step2_nl(
            data["position"], data["velocity"], data["mass"], data["packed_info"],
            data["shear_rate"], nb_idx, nb_cnt,
            velocity_h=data["velocity_h"], pressure_in=data["pressure_in"],
        )
    else:
        force, _ = compute_step2(**data)
    return cupy.asnumpy(unsort(force, perm))[0, :3]


@pytest.mark.parametrize("use_nl", [False, True], ids=["K_Step2", "K_Step2_NL"])
def test_stone_neighbour_control_pushes_fluid(use_nl: bool) -> None:
    """Control for the DEAD test below: a STONE neighbour at the same place
    pushes the WATER particle away (-x) and drags it."""
    from materials import STONE
    _upload_all_params()
    f = _water_next_to_static_force(STONE, use_nl)
    print(f"  WATER force next to STONE: {f}")
    assert f[0] < 0.0, f"STONE control did not push WATER away: {f}"


@pytest.mark.xfail(
    strict=True,
    reason="bd-r5-review-n28.23 fix step 2: step2 has no MAT_DEAD skip, so a "
           "DEAD/STATIC slot that still has mass acts as an Akinci wall",
)
@pytest.mark.parametrize("use_nl", [False, True], ids=["K_Step2", "K_Step2_NL"])
def test_fluid_next_to_dead_static_slot_gets_zero_force(use_nl: bool) -> None:
    """A MAT_DEAD/STATIC neighbour with a stale mass of 0.02 gives WATER no
    pressure push and no drag. An isolated WATER particle gets zero force."""
    from materials import DEAD
    _upload_all_params()
    f = _water_next_to_static_force(DEAD, use_nl)
    print(f"  WATER force next to DEAD slot: {f}")
    assert np.all(f == 0.0), f"DEAD slot exerted a force on WATER: {f}"


def _confinement_lattice_force(eps_v: float) -> tuple:
    """7^3 WATER lattice at rest and rest density, with a prescribed vorticity
    omega = (0, 0, 5 + 100 x). Returns (force, sorted position) in sorted
    order. Pressure is 0 and velocity is 0, so only confinement can differ
    between two runs with different vorticity_epsilon."""
    from materials import WATER
    rho0_w = float(MATERIALS[WATER].rest_density)
    k = 7
    idx = (np.arange(k, dtype=np.float32) - 3) * SPACING
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    n = xx.size
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0], pos[:, 1], pos[:, 2] = xx.ravel(), yy.ravel(), zz.ravel()
    pos[:, 3] = rho0_w
    scene = counting_sort_scene(
        pos, velocity=np.zeros((n, 4), dtype=np.float32),
        mass=np.full(n, APP_MASS, dtype=np.float32),
        packed_info=np.full(n, MAKE_PACKED(WATER, FLUID), dtype=np.uint32),
        grid_params=default_grid_params(),
    )
    pressure = cupy.zeros(n, dtype=cupy.float32)
    sorted_x = scene["position"][:, 0]
    vort = cupy.zeros((n, 4), dtype=cupy.float32)
    vort[:, 2] = 5.0 + 100.0 * sorted_x
    vort[:, 3] = cupy.abs(vort[:, 2])
    upload_granular_params(build_granular_params(
        mu0=VISCOSITY, vorticity_epsilon=eps_v, surface_tension_gamma=0.0))
    f, _ = compute_step2(
        scene["position"], scene["velocity"], scene["mass"],
        scene["packed_info"], cupy.zeros(n, dtype=cupy.float32),
        scene["cell_start"], scene["cell_end"],
        vorticity_in=vort, pressure_in=pressure, velocity_h=scene["velocity_h"],
    )
    return cupy.asnumpy(f), cupy.asnumpy(scene["position"]), cupy.asnumpy(vort)


def test_vorticity_confinement_sign() -> None:
    """Confinement force is eps * fs * (N x omega), with N = grad|omega| / |grad|omega||
    pointing toward higher |omega|.

    |omega| grows along +x and omega is along +z, so N = +x and the force on
    the centre particle is -eps * fs * omega_z * y. That is the rotation
    direction of a vortex whose core is at +x, so confinement spins it up.
    A reversed N (the old sign bug) gives +y and spins the vortex down.
    """
    _upload_all_params()
    eps_v = 0.05
    f_on, pos, vort = _confinement_lattice_force(eps_v)
    f_off, _, _ = _confinement_lattice_force(0.0)
    upload_granular_params(build_granular_params(mu0=VISCOSITY))
    centre = int(np.argmin(np.linalg.norm(pos[:, :3], axis=1)))
    fs = float(build_granular_params()[0]["force_scale"])
    diff = f_on[centre, :3] - f_off[centre, :3]
    expected_y = -eps_v * fs * float(vort[centre, 2])
    print(f"  confinement force {diff}, expected y {expected_y:.4e}")
    assert expected_y < 0.0
    assert abs(diff[1] - expected_y) <= 1e-3 * abs(expected_y), (
        f"confinement f_y {diff[1]:.4e}, expected {expected_y:.4e}")
    assert abs(diff[0]) <= 1e-3 * abs(expected_y) and abs(diff[2]) <= 1e-3 * abs(expected_y), (
        f"confinement force has off-axis parts: {diff}")


def main() -> None:
    test_compilation()
    test_struct_sizes()
    test_precalc_values()
    test_rest_density_zero_force()
    test_compressed_repulsive()
    test_static_skipped()
    test_sleeping_skipped()
    test_xsph_fluid_only()
    test_xsph_not_for_granular()
    test_xsph_skips_granular_neighbour(True)
    test_xsph_skips_granular_neighbour(False)
    test_viscosity_force()
    test_granular_pressure_clamp()
    test_granular_muI_viscosity()
    test_granular_muI_harmonic_mean()
    test_fluid_unchanged_by_muI()
    test_granular_no_nan()
    test_500k_no_errors()
    test_granular_static_drag_matches_pair()
    test_granular_rigid_drag_matches_pair()
    test_granular_friction_coeff_sets_mu_s()
    test_granular_friction_coeff_zero_falls_back_to_global_mu_s()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
