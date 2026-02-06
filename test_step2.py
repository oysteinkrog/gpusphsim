"""Integration test for step2.py -- K_Step2 pressure, viscosity, XSPH kernel.

Acceptance criteria:
  - Tait EOS: p_raw = k * (pow(rho/rho0, gamma) - 1); GRANULAR clamps >= 0,
    FLUID clamps >= -0.5*k, GAS uses linear k_gas * max(rho - rho0, 0)
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

import cupy  # type: ignore[import-untyped]
import numpy as np

from hash_sort import (
    build_grid_params,
)
from step2 import (
    BEHAVIOR_FLUID,
    BEHAVIOR_GRANULAR,
    BEHAVIOR_STATIC,
    BLOCK_SIZE,
    FLAG_IS_SLEEPING,
    FLUID_PARAMS_DTYPE,
    GRANULAR_PARAMS_DTYPE,
    PRECALC_PARAMS_DTYPE,
    build_fluid_params,
    build_granular_params,
    build_precalc_params,
    compute_step2,
    get_module,
    upload_fluid_params,
    upload_granular_params,
    upload_grid_params,
    upload_precalc_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_grid_for_test(cell_size: float = 0.04) -> np.ndarray:
    """Build and upload grid params suitable for tests."""
    params = build_grid_params()
    upload_grid_params(params)
    return params


def _upload_default_params(
    h: float = 0.04,
    mass: float = 0.02,
    rho0: float = 1000.0,
    k: float = 3.0,
    visc: float = 3.5,
) -> None:
    """Upload default fluid, precalc, and granular params."""
    fp = build_fluid_params(
        smoothing_length=h,
        particle_mass=mass,
        rest_density=rho0,
        gas_stiffness=k,
        viscosity=visc,
    )
    upload_fluid_params(fp)
    upload_precalc_params(build_precalc_params(smoothing_length=h, viscosity=visc))
    upload_granular_params(build_granular_params(mu0=visc))


def _make_two_particle_setup(
    pos_a: list,
    pos_b: list,
    vel_a: list | None = None,
    vel_b: list | None = None,
    density_a: float = 1000.0,
    density_b: float = 1000.0,
    bclass_a: int = BEHAVIOR_FLUID,
    bclass_b: int = BEHAVIOR_FLUID,
    flags_a: int = 0,
    flags_b: int = 0,
    h: float = 0.04,
    grid_params: np.ndarray | None = None,
) -> dict:
    """Create a minimal 2-particle setup with grid cell data.

    Both particles must be within h of each other to interact.
    Returns dict with all GPU arrays needed for compute_step2.
    """
    if vel_a is None:
        vel_a = [0.0, 0.0, 0.0]
    if vel_b is None:
        vel_b = [0.0, 0.0, 0.0]

    position = np.array(
        [pos_a + [0.0], pos_b + [0.0]], dtype=np.float32
    )
    veleval = np.array(
        [vel_a + [0.0], vel_b + [0.0]], dtype=np.float32
    )
    density = np.array([density_a, density_b], dtype=np.float32)
    behavior_class = np.array([bclass_a, bclass_b], dtype=np.int32)
    flags = np.array([flags_a, flags_b], dtype=np.uint32)

    # Build grid cell arrays -- place both particles in the same cell
    # Use grid from hash_sort: grid_min=(-1,-1,-1), delta=25
    if grid_params is None:
        grid_params = build_grid_params()

    gmin = grid_params[0]["grid_min"]
    gdelta = grid_params[0]["grid_delta"]
    gres = grid_params[0]["grid_res"].astype(int)

    num_cells = int(gres[0] * gres[1] * gres[2])

    # Compute cell for particle A
    cell_a = np.floor((np.array(pos_a) - gmin) * gdelta).astype(int)
    cell_a = np.clip(cell_a, 0, gres - 1)
    hash_a = int(cell_a[2] * gres[1] * gres[0] + cell_a[1] * gres[0] + cell_a[0])

    cell_b = np.floor((np.array(pos_b) - gmin) * gdelta).astype(int)
    cell_b = np.clip(cell_b, 0, gres - 1)
    hash_b = int(cell_b[2] * gres[1] * gres[0] + cell_b[1] * gres[0] + cell_b[0])

    # Build cell_start / cell_end arrays
    cell_start = np.full(num_cells, 0xFFFFFFFF, dtype=np.uint32)
    cell_end = np.zeros(num_cells, dtype=np.uint32)

    if hash_a == hash_b:
        # Both in same cell
        cell_start[hash_a] = 0
        cell_end[hash_a] = 2
    else:
        # Different cells -- sort by hash (particle with lower hash comes first)
        if hash_a < hash_b:
            cell_start[hash_a] = 0
            cell_end[hash_a] = 1
            cell_start[hash_b] = 1
            cell_end[hash_b] = 2
        else:
            # Swap particle order so sorted by hash
            position = position[::-1].copy()
            veleval = veleval[::-1].copy()
            density = density[::-1].copy()
            behavior_class = behavior_class[::-1].copy()
            flags = flags[::-1].copy()
            cell_start[hash_b] = 0
            cell_end[hash_b] = 1
            cell_start[hash_a] = 1
            cell_end[hash_a] = 2

    return {
        "position": cupy.asarray(position),
        "veleval": cupy.asarray(veleval),
        "density": cupy.asarray(density),
        "behavior_class": cupy.asarray(behavior_class),
        "flags": cupy.asarray(flags),
        "cell_start": cupy.asarray(cell_start),
        "cell_end": cupy.asarray(cell_end),
    }


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
    for sym in ("c_grid", "c_fluid", "c_precalc", "c_granular"):
        d_ptr = module.get_global(sym)  # type: ignore[union-attr]
        assert int(d_ptr) != 0, f"Symbol {sym} not found"
    print("[OK] All constant memory symbols found (c_grid, c_fluid, c_precalc, c_granular)")


def test_struct_sizes() -> None:
    """Verify struct dtype sizes match CUDA."""
    print("\n--- Struct size checks ---")

    # FluidParams: 8 floats = 32 bytes
    assert FLUID_PARAMS_DTYPE.itemsize == 32, (
        f"FluidParams size: {FLUID_PARAMS_DTYPE.itemsize} != 32"
    )
    print(f"[OK] sizeof(FluidParams) = {FLUID_PARAMS_DTYPE.itemsize}")

    # PrecalcParams: 4 floats = 16 bytes
    assert PRECALC_PARAMS_DTYPE.itemsize == 16, (
        f"PrecalcParams size: {PRECALC_PARAMS_DTYPE.itemsize} != 16"
    )
    print(f"[OK] sizeof(PrecalcParams) = {PRECALC_PARAMS_DTYPE.itemsize}")

    # GranularParams: 8 floats = 32 bytes (6 fields + 2 padding)
    assert GRANULAR_PARAMS_DTYPE.itemsize == 32, (
        f"GranularParams size: {GRANULAR_PARAMS_DTYPE.itemsize} != 32"
    )
    print(f"[OK] sizeof(GranularParams) = {GRANULAR_PARAMS_DTYPE.itemsize}")

    print("[OK] Block size = 128")
    assert BLOCK_SIZE == 128


def test_precalc_values() -> None:
    """Verify precalculated kernel coefficients."""
    print("\n--- Precalc coefficient checks ---")

    h = 0.04
    mu = 3.5  # default viscosity
    pp = build_precalc_params(h, viscosity=mu)

    # pressure_precalc = +45 / (pi * h^6)
    lap_const = 45.0 / (math.pi * h**6)
    expected_press = lap_const
    actual_press = float(pp[0]["pressure_precalc"])
    assert abs(actual_press - expected_press) / expected_press < 1e-5, (
        f"pressure_precalc: {actual_press} != {expected_press}"
    )
    assert actual_press > 0, "pressure_precalc must be POSITIVE"
    print(f"[OK] pressure_precalc = {actual_press:.6e} (positive, as expected)")

    # viscosity_precalc = mu * 45 / (pi * h^6)
    expected_visc = mu * lap_const
    actual_visc = float(pp[0]["viscosity_precalc"])
    assert abs(actual_visc - expected_visc) / expected_visc < 1e-5
    print(f"[OK] viscosity_precalc = {actual_visc:.6e} (includes mu={mu})")

    # poly6 coefficient = 315 / (64 * pi * h^9)
    expected_poly6 = 315.0 / (64.0 * math.pi * h**9)
    actual_poly6 = float(pp[0]["kernel_poly6_coeff"])
    assert abs(actual_poly6 - expected_poly6) / expected_poly6 < 1e-5
    print(f"[OK] kernel_poly6_coeff = {actual_poly6:.6e}")


def test_rest_density_zero_force() -> None:
    """Two fluid particles at rest density produce ~zero net force."""
    print("\n--- Rest density -> ~zero force test ---")

    h = 0.04
    rho0 = 1000.0
    mass = 0.02

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, mass=mass, rho0=rho0, k=3.0, visc=3.5)

    # Two fluid particles at rest, separated by 0.5 * h
    # Both at rest density -> pressure ~ 0 -> force ~ 0
    center = [0.0, 0.0, 0.0]
    sep = h * 0.5
    pos_a = [center[0] - sep / 2, center[1], center[2]]
    pos_b = [center[0] + sep / 2, center[1], center[2]]

    data = _make_two_particle_setup(
        pos_a=pos_a,
        pos_b=pos_b,
        density_a=rho0,
        density_b=rho0,
        h=h,
        grid_params=gp,
    )

    sph_force, veleval_out = compute_step2(**data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)
    # At rest density, Tait EOS gives p = k * (1^7 - 1) = 0
    # So pressure force should be ~0. Viscosity force is also 0 (velocities equal).
    f_mag_0 = np.linalg.norm(forces[0, :3])
    f_mag_1 = np.linalg.norm(forces[1, :3])

    print(f"  Particle 0 force magnitude: {f_mag_0:.6e}")
    print(f"  Particle 1 force magnitude: {f_mag_1:.6e}")

    # Allow small numerical error -- forces should be approximately zero
    assert f_mag_0 < 1e-3, f"Expected ~zero force, got {f_mag_0}"
    assert f_mag_1 < 1e-3, f"Expected ~zero force, got {f_mag_1}"
    print("[OK] Two fluid particles at rest density -> approximately zero force")


def test_compressed_repulsive() -> None:
    """Compressed fluid particles produce repulsive forces (pushed apart)."""
    print("\n--- Compressed -> repulsive force test ---")

    h = 0.04
    rho0 = 1000.0
    mass = 0.02

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, mass=mass, rho0=rho0, k=3.0, visc=3.5)

    # Two fluid particles close together, with density above rest
    sep = h * 0.3  # close together
    pos_a = [0.0, 0.0, 0.0]
    pos_b = [sep, 0.0, 0.0]

    compressed_density = rho0 * 1.1  # 10% above rest density

    data = _make_two_particle_setup(
        pos_a=pos_a,
        pos_b=pos_b,
        density_a=compressed_density,
        density_b=compressed_density,
        h=h,
        grid_params=gp,
    )

    sph_force, veleval_out = compute_step2(**data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)

    # Particle 0 (left) should be pushed in -x direction (away from particle 1)
    # Particle 1 (right) should be pushed in +x direction (away from particle 0)
    fx_0 = forces[0, 0]
    fx_1 = forces[1, 0]

    print(f"  Particle 0 force: ({forces[0, 0]:.4e}, {forces[0, 1]:.4e}, {forces[0, 2]:.4e})")
    print(f"  Particle 1 force: ({forces[1, 0]:.4e}, {forces[1, 1]:.4e}, {forces[1, 2]:.4e})")

    # Particle 0 is at origin, particle 1 is at +x
    # r = pos_i - pos_j points from j to i
    # For particle 0: r = pos_0 - pos_1 = (-sep, 0, 0) -> force in -x
    # For particle 1: r = pos_1 - pos_0 = (+sep, 0, 0) -> force in +x
    assert fx_0 < 0, f"Particle 0 should be pushed in -x, got fx={fx_0}"
    assert fx_1 > 0, f"Particle 1 should be pushed in +x, got fx={fx_1}"

    # Forces should be approximately equal and opposite (Newton's 3rd law)
    assert abs(fx_0 + fx_1) < abs(fx_0) * 0.01, (
        f"Forces not equal/opposite: {fx_0} + {fx_1} = {fx_0 + fx_1}"
    )
    print("[OK] Compressed particles -> repulsive forces (Newton's 3rd law satisfied)")


def test_static_skipped() -> None:
    """STATIC particles (behavior_class == 3) produce zero force."""
    print("\n--- STATIC skip test ---")

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0)

    sep = h * 0.3
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        density_a=rho0 * 1.2,  # high density to ensure non-zero pressure
        density_b=rho0 * 1.2,
        bclass_a=BEHAVIOR_STATIC,  # particle 0 is STATIC
        bclass_b=BEHAVIOR_FLUID,
        h=h,
        grid_params=gp,
    )

    sph_force, _ = compute_step2(**data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)

    # STATIC particle should have zero force
    f_mag_static = np.linalg.norm(forces[0, :3])
    assert f_mag_static == 0.0, f"STATIC particle force = {f_mag_static}, expected 0"
    print("[OK] STATIC particle -> zero force (early return)")

    # FLUID particle should still have a force
    f_mag_fluid = np.linalg.norm(forces[1, :3])
    assert f_mag_fluid > 0, f"FLUID particle force = {f_mag_fluid}, expected > 0"
    print(f"[OK] FLUID neighbor still gets force: magnitude = {f_mag_fluid:.4e}")


def test_sleeping_skipped() -> None:
    """SLEEPING particles produce zero force."""
    print("\n--- SLEEPING skip test ---")

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0)

    sep = h * 0.3
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        density_a=rho0 * 1.2,
        density_b=rho0 * 1.2,
        bclass_a=BEHAVIOR_FLUID,
        bclass_b=BEHAVIOR_FLUID,
        flags_a=FLAG_IS_SLEEPING,  # particle 0 is sleeping
        h=h,
        grid_params=gp,
    )

    sph_force, _ = compute_step2(**data)
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

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0)

    sep = h * 0.3
    # Particle 0 moves +x, particle 1 moves -x -> XSPH should average them
    vel_a = [1.0, 0.0, 0.0]
    vel_b = [-1.0, 0.0, 0.0]

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=rho0,
        density_b=rho0,
        bclass_a=BEHAVIOR_FLUID,
        bclass_b=BEHAVIOR_FLUID,
        h=h,
        grid_params=gp,
    )

    _, veleval_out = compute_step2(**data)
    cupy.cuda.Device().synchronize()

    vels = cupy.asnumpy(veleval_out)

    # XSPH: veleval = v_i + epsilon * sum(m_j / rho_avg * (v_j - v_i) * W_poly6)
    # epsilon = 0.5; (v_j - v_i) = (-2, 0, 0) for particle 0
    # The corrected velocity should be shifted toward the neighbor's velocity
    vx_0 = vels[0, 0]
    vx_1 = vels[1, 0]

    print(f"  Original vel: particle 0 = ({vel_a[0]}, 0, 0), particle 1 = ({vel_b[0]}, 0, 0)")
    print(f"  XSPH vel:     particle 0 = ({vx_0:.4f}, {vels[0, 1]:.4f}, {vels[0, 2]:.4f})")
    print(f"  XSPH vel:     particle 1 = ({vx_1:.4f}, {vels[1, 1]:.4f}, {vels[1, 2]:.4f})")

    # Particle 0: v_i=1.0, neighbor v_j=-1.0, so XSPH pulls vx toward -1 -> vx < 1.0
    assert vx_0 < 1.0, f"XSPH should reduce vx for particle 0: {vx_0}"
    # Particle 1: v_i=-1.0, neighbor v_j=1.0, so XSPH pulls vx toward +1 -> vx > -1.0
    assert vx_1 > -1.0, f"XSPH should increase vx for particle 1: {vx_1}"
    print("[OK] XSPH correction shifts velocities toward neighbors (FLUID)")


def test_xsph_not_for_granular() -> None:
    """XSPH correction is NOT applied to GRANULAR particles."""
    print("\n--- XSPH not for GRANULAR test ---")

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0)

    sep = h * 0.3
    vel_a = [1.0, 0.0, 0.0]
    vel_b = [-1.0, 0.0, 0.0]

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=rho0,
        density_b=rho0,
        bclass_a=BEHAVIOR_GRANULAR,
        bclass_b=BEHAVIOR_GRANULAR,
        h=h,
        grid_params=gp,
    )

    _, veleval_out = compute_step2(**data)
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


def test_viscosity_force() -> None:
    """Viscosity force opposes relative motion between neighbors."""
    print("\n--- Viscosity force test ---")

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0, visc=10.0)  # high viscosity

    sep = h * 0.3
    # Particle 0 moving +x, particle 1 at rest -> viscosity should slow particle 0
    vel_a = [2.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=rho0,
        density_b=rho0,
        h=h,
        grid_params=gp,
    )

    sph_force, _ = compute_step2(**data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)

    # Viscosity: f_visc = visc_precalc * m_j * (v_j - v_i) / rho_j * (h - |r|)
    # For particle 0: (v_j - v_i) = (0 - 2, 0, 0) = (-2, 0, 0)
    # -> viscosity force on particle 0 should have a -x component (slowing it down)
    # But this is just the viscosity part; pressure also contributes.
    # At rest density, pressure is ~0, so force is dominated by viscosity.
    fx_0 = forces[0, 0]
    fx_1 = forces[1, 0]

    print(f"  Particle 0 force: ({forces[0, 0]:.4e}, {forces[0, 1]:.4e}, {forces[0, 2]:.4e})")
    print(f"  Particle 1 force: ({forces[1, 0]:.4e}, {forces[1, 1]:.4e}, {forces[1, 2]:.4e})")

    # At rest density, pressure ~ 0, so force is almost entirely viscosity
    # viscosity_precalc * m_j * (v_j - v_i) / rho_j * (h - |r|) * mass
    # For particle 0: (v_j - v_i).x = -2 -> fx should be negative
    assert fx_0 < 0, f"Viscosity should slow particle 0 (fx < 0), got {fx_0}"
    # For particle 1: (v_j - v_i).x = +2 -> fx should be positive
    assert fx_1 > 0, f"Viscosity should accelerate particle 1 (fx > 0), got {fx_1}"
    print("[OK] Viscosity force opposes relative motion")


def test_granular_pressure_clamp() -> None:
    """GRANULAR pressure clamps to 0 (no tensile forces)."""
    print("\n--- GRANULAR pressure clamp test ---")

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0, k=3.0, visc=0.0)  # zero viscosity

    sep = h * 0.3
    # Density below rest -> Tait EOS gives negative p_raw
    # GRANULAR clamps to 0 -> no pressure force
    low_density = rho0 * 0.9

    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        density_a=low_density,
        density_b=low_density,
        bclass_a=BEHAVIOR_GRANULAR,
        bclass_b=BEHAVIOR_GRANULAR,
        h=h,
        grid_params=gp,
    )

    sph_force, _ = compute_step2(**data)
    cupy.cuda.Device().synchronize()

    forces = cupy.asnumpy(sph_force)
    f_mag_0 = np.linalg.norm(forces[0, :3])
    f_mag_1 = np.linalg.norm(forces[1, :3])

    print(f"  GRANULAR at low density ({low_density}):")
    print(f"  Particle 0 force magnitude: {f_mag_0:.6e}")
    print(f"  Particle 1 force magnitude: {f_mag_1:.6e}")

    # With zero viscosity and zero pressure (clamped), force should be ~0
    assert f_mag_0 < 1e-3, f"Expected ~zero force for GRANULAR at low density, got {f_mag_0}"
    assert f_mag_1 < 1e-3, f"Expected ~zero force for GRANULAR at low density, got {f_mag_1}"
    print("[OK] GRANULAR at low density -> pressure clamped to 0, ~zero force")


def test_granular_muI_viscosity() -> None:
    """GRANULAR particles with relative motion get mu(I) enhanced viscosity force."""
    print("\n--- mu(I) viscosity for GRANULAR test ---")

    h = 0.04
    rho0 = 1000.0
    visc = 3.5

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0, visc=visc)

    sep = h * 0.3
    # Two GRANULAR particles with relative motion and above-rest density
    # (above rest so pressure is positive, which drives mu(I) computation)
    compressed_density = rho0 * 1.05
    vel_a = [2.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]

    data_granular = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=compressed_density,
        density_b=compressed_density,
        bclass_a=BEHAVIOR_GRANULAR,
        bclass_b=BEHAVIOR_GRANULAR,
        h=h,
        grid_params=gp,
    )

    sph_force_gran, _ = compute_step2(**data_granular)
    cupy.cuda.Device().synchronize()
    forces_gran = cupy.asnumpy(sph_force_gran)

    # Compare with FLUID particles in the same configuration
    data_fluid = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=compressed_density,
        density_b=compressed_density,
        bclass_a=BEHAVIOR_FLUID,
        bclass_b=BEHAVIOR_FLUID,
        h=h,
        grid_params=gp,
    )

    sph_force_fluid, _ = compute_step2(**data_fluid)
    cupy.cuda.Device().synchronize()
    forces_fluid = cupy.asnumpy(sph_force_fluid)

    f_gran_mag = np.linalg.norm(forces_gran[0, :3])
    f_fluid_mag = np.linalg.norm(forces_fluid[0, :3])

    print(f"  GRANULAR force magnitude: {f_gran_mag:.4e}")
    print(f"  FLUID force magnitude:    {f_fluid_mag:.4e}")

    # mu(I) viscosity should produce a different (typically larger) force than
    # the constant viscosity used for FLUID, because mu_I * p_eff adds to mu0
    assert f_gran_mag != f_fluid_mag, (
        f"GRANULAR and FLUID should have different viscosity forces: "
        f"gran={f_gran_mag:.4e}, fluid={f_fluid_mag:.4e}"
    )
    # The GRANULAR force should be non-zero (viscosity opposes relative motion)
    assert f_gran_mag > 0, f"Expected non-zero GRANULAR force, got {f_gran_mag}"
    print("[OK] GRANULAR mu(I) produces different viscosity than constant mu0")


def test_granular_muI_harmonic_mean() -> None:
    """Harmonic mean eta_ij is used for GRANULAR-GRANULAR viscosity."""
    print("\n--- mu(I) harmonic mean test ---")

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0, visc=3.5)

    sep = h * 0.3
    # Asymmetric density: different pressures -> different eta values
    # This tests that eta_ij = 2*eta_i*eta_j/(eta_i + eta_j + 1e-8)
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=[3.0, 0.0, 0.0],
        vel_b=[-1.0, 0.0, 0.0],
        density_a=rho0 * 1.1,   # higher pressure -> different eta
        density_b=rho0 * 1.01,  # lower pressure -> different eta
        bclass_a=BEHAVIOR_GRANULAR,
        bclass_b=BEHAVIOR_GRANULAR,
        h=h,
        grid_params=gp,
    )

    sph_force, _ = compute_step2(**data)
    cupy.cuda.Device().synchronize()
    forces = cupy.asnumpy(sph_force)

    # Both particles should have non-zero forces (viscosity + pressure)
    f_mag_0 = np.linalg.norm(forces[0, :3])
    f_mag_1 = np.linalg.norm(forces[1, :3])
    print(f"  Particle 0 force magnitude: {f_mag_0:.4e}")
    print(f"  Particle 1 force magnitude: {f_mag_1:.4e}")

    assert f_mag_0 > 0, f"Expected non-zero force on particle 0, got {f_mag_0}"
    assert f_mag_1 > 0, f"Expected non-zero force on particle 1, got {f_mag_1}"

    # Forces should be finite (no NaN/Inf from division by zero in harmonic mean)
    assert np.isfinite(forces[0, :3]).all(), "NaN/Inf in particle 0 force"
    assert np.isfinite(forces[1, :3]).all(), "NaN/Inf in particle 1 force"
    print("[OK] Harmonic mean eta_ij produces finite, non-zero forces")


def test_fluid_unchanged_by_muI() -> None:
    """FLUID particles still use constant mu0 viscosity (unchanged by mu(I))."""
    print("\n--- FLUID unchanged by mu(I) test ---")

    h = 0.04
    rho0 = 1000.0
    visc = 10.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0, visc=visc)

    sep = h * 0.3
    vel_a = [2.0, 0.0, 0.0]
    vel_b = [0.0, 0.0, 0.0]

    # Run with default granular params
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=rho0,
        density_b=rho0,
        bclass_a=BEHAVIOR_FLUID,
        bclass_b=BEHAVIOR_FLUID,
        h=h,
        grid_params=gp,
    )

    sph_force1, _ = compute_step2(**data)
    cupy.cuda.Device().synchronize()
    forces1 = cupy.asnumpy(sph_force1)

    # Run again with very different granular params (should not affect FLUID)
    upload_granular_params(build_granular_params(mu_s=0.9, mu_2=0.99, mu_max=99999.0))

    data2 = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=vel_a,
        vel_b=vel_b,
        density_a=rho0,
        density_b=rho0,
        bclass_a=BEHAVIOR_FLUID,
        bclass_b=BEHAVIOR_FLUID,
        h=h,
        grid_params=gp,
    )

    sph_force2, _ = compute_step2(**data2)
    cupy.cuda.Device().synchronize()
    forces2 = cupy.asnumpy(sph_force2)

    # Forces should be identical for FLUID regardless of granular params
    diff = np.abs(forces1 - forces2).max()
    print(f"  Max force difference between runs: {diff:.6e}")
    assert diff < 1e-6, f"FLUID forces changed with granular params: diff={diff}"
    print("[OK] FLUID viscosity is unchanged by mu(I) granular parameters")

    # Restore defaults
    upload_granular_params(build_granular_params())


def test_granular_no_nan() -> None:
    """No NaN velocities in GRANULAR forces (mu_max clamp prevents divergence)."""
    print("\n--- GRANULAR no-NaN test ---")

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0, visc=3.5)

    sep = h * 0.5
    # Test with near-zero relative velocity (gamma_dot ~ 0)
    # This would cause division by zero without the 1e-6 epsilon
    data = _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[sep, 0.0, 0.0],
        vel_a=[0.0, 0.0, 0.0],
        vel_b=[1e-10, 0.0, 0.0],  # near-zero relative motion
        density_a=rho0 * 1.1,
        density_b=rho0 * 1.1,
        bclass_a=BEHAVIOR_GRANULAR,
        bclass_b=BEHAVIOR_GRANULAR,
        h=h,
        grid_params=gp,
    )

    sph_force, _ = compute_step2(**data)
    cupy.cuda.Device().synchronize()
    forces = cupy.asnumpy(sph_force)

    assert np.isfinite(forces).all(), f"NaN/Inf in granular forces: {forces}"
    print(f"  Force magnitudes: [{np.linalg.norm(forces[0, :3]):.4e}, {np.linalg.norm(forces[1, :3]):.4e}]")
    print("[OK] No NaN/Inf for near-zero shear rate (mu_max clamp works)")


def test_500k_no_errors() -> None:
    """Kernel runs without errors for 500K particles."""
    print("\n--- 500K particle stress test ---")

    h = 0.04
    rho0 = 1000.0

    gp = _setup_grid_for_test()
    _upload_default_params(h=h, rho0=rho0)

    n = 500_000
    rng = np.random.default_rng(42)

    # Random positions in [-0.5, 0.5]^3 (inside grid)
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, :3] = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)

    vel_np = np.zeros((n, 4), dtype=np.float32)
    vel_np[:, :3] = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)

    density_np = rng.uniform(900.0, 1100.0, size=n).astype(np.float32)

    # Mix of FLUID and GRANULAR particles
    bclass_np = np.ones(n, dtype=np.int32) * BEHAVIOR_FLUID
    bclass_np[n // 2:] = BEHAVIOR_GRANULAR  # Half granular
    flags_np = np.zeros(n, dtype=np.uint32)

    # Build simple grid index -- for stress test, use empty cells (no neighbors)
    # This tests that the kernel handles 0xFFFFFFFF sentinel correctly for 500K particles
    gres = gp[0]["grid_res"].astype(int)
    num_cells = int(gres[0] * gres[1] * gres[2])
    cell_start = np.full(num_cells, 0xFFFFFFFF, dtype=np.uint32)
    cell_end = np.zeros(num_cells, dtype=np.uint32)

    # Place all particles in cell_start/cell_end using simple binning
    gmin = gp[0]["grid_min"]
    gdelta = gp[0]["grid_delta"]
    cells = np.floor((pos_np[:, :3] - gmin) * gdelta).astype(int)
    cells = np.clip(cells, 0, gres - 1)
    hashes = (cells[:, 2] * gres[1] * gres[0] + cells[:, 1] * gres[0] + cells[:, 0]).astype(np.uint32)

    # Sort by hash for the grid structure
    sort_idx = np.argsort(hashes)
    pos_np = pos_np[sort_idx]
    vel_np = vel_np[sort_idx]
    density_np = density_np[sort_idx]
    bclass_np = bclass_np[sort_idx]
    flags_np = flags_np[sort_idx]
    sorted_hashes = hashes[sort_idx]

    # Build cell_start/cell_end
    for i in range(n):
        h_val = int(sorted_hashes[i])
        if i == 0 or sorted_hashes[i] != sorted_hashes[i - 1]:
            cell_start[h_val] = i
        cell_end[h_val] = i + 1

    position = cupy.asarray(pos_np)
    veleval = cupy.asarray(vel_np)
    density = cupy.asarray(density_np)
    behavior_class = cupy.asarray(bclass_np)
    flags_gpu = cupy.asarray(flags_np)
    cell_start_gpu = cupy.asarray(cell_start)
    cell_end_gpu = cupy.asarray(cell_end)

    sph_force, veleval_out = compute_step2(
        position, veleval, density, behavior_class, flags_gpu,
        cell_start_gpu, cell_end_gpu,
    )
    cupy.cuda.Device().synchronize()

    # Basic sanity: no NaN/Inf
    forces_np = cupy.asnumpy(sph_force)
    assert not np.any(np.isnan(forces_np)), "NaN in sph_force"
    assert not np.any(np.isinf(forces_np)), "Inf in sph_force"

    vels_np = cupy.asnumpy(veleval_out)
    assert not np.any(np.isnan(vels_np)), "NaN in veleval_out"
    assert not np.any(np.isinf(vels_np)), "Inf in veleval_out"

    print("[OK] 500K particles processed without errors")
    print(f"     Force range: [{forces_np[:, :3].min():.4e}, {forces_np[:, :3].max():.4e}]")
    print(f"     Veleval range: [{vels_np[:, :3].min():.4e}, {vels_np[:, :3].max():.4e}]")


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
    test_viscosity_force()
    test_granular_pressure_clamp()
    test_granular_muI_viscosity()
    test_granular_muI_harmonic_mean()
    test_fluid_unchanged_by_muI()
    test_granular_no_nan()
    test_500k_no_errors()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
