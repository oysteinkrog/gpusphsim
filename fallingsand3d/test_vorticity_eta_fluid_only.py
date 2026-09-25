"""Vorticity eta must sum over FLUID neighbours only (bd-r5-review-n28.66).

The vorticity-confinement direction N = eta / |eta| with
eta = sum_j (m_j / rho_j) (|omega_j| - |omega_i|) grad W_ij.
K_Step2 and K_PBF_Finalize used to add every in-range neighbour to eta.
A MAT_RIGID neighbour (position.w = 0, mass psi) got weight psi / 0.01,
about 5e4 times a water neighbour, and a GAS neighbour (rho about 0.4)
about 2500 times. Both have omega = 0, so N pointed at the body or gas
and the confinement force became a tangential push along the boundary.

Checks:
  - K_Step2: the confinement force on a FLUID particle is the same with or
    without an extra MAT_RIGID, GAS or SAND neighbour.
  - K_PBF_Finalize: the same check on the velocity kick dt*eps*(N x omega).
  - K_Step1: a GAS particle next to a MAT_RIGID particle gets no heat or
    corrosion exposure from it, even with a non-zero interaction entry,
    and the MAT_RIGID early-out packs 2500 into position.w.

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np
import pytest

import pbf_solver
import step1 as s1
from conftest import counting_sort_scene, default_grid_params, unsort
from materials import (
    FIRE,
    FLUID,
    GAS,
    GRANULAR,
    MAT_RIGID,
    SAND,
    SMOKE,
    STATIC,
    WATER,
    build_interaction_matrix,
    build_material_array,
)
from solver_profiles import PROFILES
from step2 import (
    build_granular_params,
    compute_pressure,
    compute_step2,
    upload_granular_params,
    upload_grid_params,
    upload_materials,
    upload_precalc_params,
    upload_sim_params,
)

H = 0.04
SPACING = 0.02
MASS = 0.02
PSI = 0.008  # typical Akinci boundary mass psi_b
EPS = 0.05
DT = 1.0 / 120.0


def _packed(mat: int, beh: int) -> int:
    return (mat & 0xFF) | ((beh & 0x3) << 8)


# Third neighbour variants: (label, packed_info, rho packed in .w, mass, |omega|)
THIRD = {
    "rigid": (_packed(MAT_RIGID, STATIC), 0.0, PSI, 0.0),
    "gas": (_packed(SMOKE, GAS), 0.4, MASS, 0.0),
    "sand": (_packed(SAND, GRANULAR), 1600.0, MASS, 0.0),
}


def _scene(third: str | None) -> tuple[dict, np.ndarray]:
    """FLUID i at the origin, FLUID j at +x with twice the vorticity,
    and an optional third neighbour at +y.

    With only FLUID neighbours N = +x, so eps*(N x omega_i) points along -y.
    A heavy third neighbour at +y with omega = 0 would turn N toward -y and
    the force toward x.
    """
    pos = [[0.0, 0.0, 0.0, 1000.0], [SPACING, 0.0, 0.0, 1000.0]]
    packed = [_packed(WATER, FLUID), _packed(WATER, FLUID)]
    mass = [MASS, MASS]
    vort = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]
    if third is not None:
        pi3, rho3, m3, w3 = THIRD[third]
        pos.append([0.0, SPACING, 0.0, rho3])
        packed.append(pi3)
        mass.append(m3)
        vort.append([0.0, 0.0, w3, w3])
    n = len(pos)
    scene = counting_sort_scene(
        np.array(pos, dtype=np.float32),
        velocity=np.zeros((n, 4), dtype=np.float32),
        mass=np.array(mass, dtype=np.float32),
        packed_info=np.array(packed, dtype=np.uint32),
        grid_params=default_grid_params(),
    )
    perm = scene["sort_perm"].astype(cupy.int64)
    vort_sorted = cupy.asarray(np.array(vort, dtype=np.float32))[perm]
    return scene, vort_sorted


# ---------------------------------------------------------------------------
# K_Step2
# ---------------------------------------------------------------------------


def _upload_step2(eps: float) -> None:
    upload_grid_params(default_grid_params())
    upload_sim_params(
        s1.build_sim_params(
            smoothing_length=H, particle_mass=MASS, particle_spacing=SPACING
        )
    )
    upload_precalc_params(s1.build_precalc_params(smoothing_length=H, viscosity=1.0))
    upload_materials(build_material_array())
    upload_granular_params(build_granular_params(mu0=1.0, vorticity_epsilon=eps))


def _step2_conf_force(third: str | None) -> np.ndarray:
    """Confinement force on particle 0: force(eps) - force(eps = 0)."""
    out = []
    for eps in (EPS, 0.0):
        _upload_step2(eps)
        scene, vort = _scene(third)
        n = scene["position"].shape[0]
        density = cupy.ascontiguousarray(scene["position"][:, 3])
        pressure = cupy.zeros(n, dtype=cupy.float32)
        compute_pressure(density, scene["packed_info"], pressure)
        force, _ = compute_step2(
            scene["position"],
            scene["velocity"],
            scene["mass"],
            scene["packed_info"],
            cupy.zeros(n, dtype=cupy.float32),
            scene["cell_start"],
            scene["cell_end"],
            vorticity_in=vort,
            velocity_h=scene["velocity_h"],
            pressure_in=pressure,
        )
        out.append(cupy.asnumpy(unsort(force, scene["sort_perm"]))[0, :3])
    return out[0] - out[1]


@pytest.mark.parametrize("third", sorted(THIRD))
def test_step2_eta_ignores_non_fluid_neighbour(third: str) -> None:
    ref = _step2_conf_force(None)
    got = _step2_conf_force(third)
    print(f"  K_Step2 {third}: f_conf fluid-only={ref}  with {third}={got}")
    assert np.linalg.norm(ref) > 0.0, "no confinement force in the reference scene"
    assert ref[1] < 0.0 and abs(ref[0]) < 1e-3 * abs(ref[1]), (
        f"reference force should point along -y, got {ref}"
    )
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6 * np.linalg.norm(ref))


# ---------------------------------------------------------------------------
# K_PBF_Finalize
# ---------------------------------------------------------------------------


def _upload_pbf(eps: float) -> None:
    pbf_solver.upload_grid_params(default_grid_params())
    pbf_solver.upload_sim_params(
        s1.build_sim_params(
            smoothing_length=H,
            particle_mass=MASS,
            particle_spacing=SPACING,
            gravity=(0.0, 0.0, 0.0),
            dt=DT,
        )
    )
    pbf_solver.upload_precalc_params(
        s1.build_precalc_params(smoothing_length=H, viscosity=1.0)
    )
    pbf_solver.upload_materials(build_material_array())
    pbf_solver.upload_interactions(build_interaction_matrix())
    pbf_solver.upload_granular_params(
        build_granular_params(mu0=1.0, vorticity_epsilon=eps, surface_tension_gamma=0.0)
    )
    pbf_solver.upload_pbf_params(PROFILES["PBF"], h=H)


def _pbf_conf_kick(third: str | None) -> np.ndarray:
    """Velocity kick on particle 0 from confinement: v(eps) - v(eps = 0)."""
    out = []
    for eps in (EPS, 0.0):
        _upload_pbf(eps)
        scene, vort = _scene(third)
        n = scene["position"].shape[0]
        pos = scene["position"]
        density = cupy.ascontiguousarray(pos[:, 3])
        f32 = lambda: cupy.zeros(n, dtype=cupy.float32)  # noqa: E731
        vel_out = cupy.zeros((n, 4), dtype=cupy.float32)
        pbf_solver.pbf_finalize(
            pos.copy(),
            pos.copy(),
            scene["velocity"],
            density,
            scene["mass"],
            scene["packed_info"],
            cupy.full(n, 293.0, dtype=cupy.float32),
            cupy.ones(n, dtype=cupy.float32),
            f32(),
            cupy.zeros(n, dtype=cupy.uint32),
            scene["sort_perm"],
            scene["cell_start"],
            scene["cell_end"],
            cupy.zeros((n, 4), dtype=cupy.float32),
            vel_out,
            cupy.zeros(n, dtype=cupy.uint32),
            cupy.zeros(n, dtype=cupy.uint32),
            f32(),
            vorticity_in=vort,
        )
        out.append(cupy.asnumpy(vel_out)[0, :3])
    return out[0] - out[1]


@pytest.mark.parametrize("third", sorted(THIRD))
def test_pbf_eta_ignores_non_fluid_neighbour(third: str) -> None:
    ref = _pbf_conf_kick(None)
    got = _pbf_conf_kick(third)
    print(f"  K_PBF_Finalize {third}: dv fluid-only={ref}  with {third}={got}")
    expected = DT * EPS  # |N x omega_i| = 1
    assert ref[1] == pytest.approx(-expected, rel=1e-3), (
        f"reference kick should be -dt*eps along y, got {ref}"
    )
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4 * expected)


# ---------------------------------------------------------------------------
# K_Step1: gas exposure from a MAT_RIGID neighbour, rigid position.w
# ---------------------------------------------------------------------------


@pytest.fixture
def step1_interactions():
    """Upload step1 params with FIRE reacting to MAT_RIGID and WATER; restore after."""
    s1.upload_grid_params(default_grid_params())
    s1.upload_sim_params(
        s1.build_sim_params(
            smoothing_length=H, particle_mass=MASS, particle_spacing=SPACING
        )
    )
    s1.upload_precalc_params(s1.build_precalc_params(smoothing_length=H))
    s1.upload_materials(build_material_array())
    table = build_interaction_matrix()
    for a, b in ((FIRE, MAT_RIGID), (FIRE, WATER)):
        table[a][b]["reaction_rate"] = 1.0
        table[a][b]["heat_exchange"] = 1.0
    s1.upload_interactions(table)
    yield
    s1.upload_interactions(build_interaction_matrix())


def _run_step1_fire_next_to(mat_j: int, beh_j: int, m_j: float) -> tuple:
    pos = np.array([[0.0, 0.0, 0.0, 0.0], [SPACING, 0.0, 0.0, 0.0]], dtype=np.float32)
    packed = np.array([_packed(FIRE, GAS), _packed(mat_j, beh_j)], dtype=np.uint32)
    temp = np.array([293.0, 1000.0], dtype=np.float32)
    scene = counting_sort_scene(
        pos,
        velocity=np.zeros((2, 4), dtype=np.float32),
        mass=np.array([MASS, m_j], dtype=np.float32),
        packed_info=packed,
        temperature=temp,
        grid_params=default_grid_params(),
    )
    _, _, _, exp_heat, exp_corr, _ = s1.compute_step1(
        scene["position"],
        scene["velocity"],
        scene["mass"],
        None,
        scene["packed_info"],
        scene["temperature"],
        scene["cell_start"],
        scene["cell_end"],
        velocity_h=scene["velocity_h"],
    )
    perm = scene["sort_perm"]
    heat = cupy.asnumpy(unsort(exp_heat, perm))
    corr = cupy.asnumpy(unsort(exp_corr, perm))
    pos_w = cupy.asnumpy(unsort(scene["position"][:, 3], perm))
    return heat, corr, pos_w


def test_step1_gas_exposure_skips_rigid(step1_interactions) -> None:
    heat, corr, pos_w = _run_step1_fire_next_to(MAT_RIGID, STATIC, PSI)
    print(
        f"  FIRE next to MAT_RIGID: exposure_heat={heat[0]} corrode={corr[0]} rigid .w={pos_w[1]}"
    )
    assert heat[0] == 0.0, f"gas got heat exposure {heat[0]} from a rigid body"
    assert corr[0] == 0.0, f"gas got corrosion exposure {corr[0]} from a rigid body"
    assert pos_w[1] == 2500.0, f"MAT_RIGID early-out left position.w = {pos_w[1]}"


def test_step1_gas_exposure_control(step1_interactions) -> None:
    """Same table entry with a WATER neighbour: exposure is non-zero."""
    heat, corr, _ = _run_step1_fire_next_to(WATER, FLUID, MASS)
    print(f"  FIRE next to WATER: exposure_heat={heat[0]} corrode={corr[0]}")
    assert heat[0] > 0.0
    assert corr[0] > 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
