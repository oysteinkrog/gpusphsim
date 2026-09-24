"""WCSPH FLUID viscosity uses the DFSPH dynamic form (bd-r5-review-n28.79).

K_Step2 used to compute the FLUID pair term as
mu_ij * lap * m_j * lap_v / rho_j, scaled by force_scale. That gives
F_ij / F_ji = rho_i / rho_j, so every free surface (rho down to about
0.6 rho0) created momentum, and WATER had nu = 0.02 m^2/s against
mu / rho0 = 4e-4 m^2/s in DFSPH.

The pair term is now mu_ij * lap * m_j * lap_v / (rho_i * rho_j), with no
force_scale, the same number and formula DFSPH uses.

Checks (the viscous part is the force with velocities minus the force
with zero velocities, so pressure cancels):
  - WATER pair with rho_i != rho_j and different masses:
    m_a a_a + m_b a_b = 0 for K_Step2 and K_Step2_NL.
  - The pair accel matches the analytic dynamic form value.
  - A WATER block with a free-surface density profile and random
    velocities: total viscous momentum is at float round-off.

dt is set to 1e-7 s so the implicit self-term scale 1 / (1 + dt * c_tot)
(bd-r5-review-n28.52) is 1 and the pair formula is tested on its own.

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

import neighbor_list as nl
from conftest import counting_sort_scene, default_grid_params, unsort
from materials import FLUID, MATERIALS, OIL, WATER, build_material_array
from step1 import build_precalc_params, build_sim_params
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
from test_step2 import MAKE_PACKED, _make_two_particle_setup

H = 0.04
SPACING = 0.02
MASS = 0.02
DT_TINY = 1e-7


def _params():
    return (
        default_grid_params(),
        build_sim_params(
            smoothing_length=H,
            particle_mass=MASS,
            particle_spacing=SPACING,
            gravity=(0.0, 0.0, 0.0),
            dt=DT_TINY,
        ),
        build_precalc_params(smoothing_length=H, viscosity=1.0),
        build_material_array(),
        build_granular_params(mu0=1.0),
    )


def _upload_step2() -> None:
    g, s, p, m, gr = _params()
    upload_grid_params(g)
    upload_sim_params(s)
    upload_precalc_params(p)
    upload_materials(m)
    upload_granular_params(gr)


def _upload_nl() -> None:
    g, s, p, m, gr = _params()
    nl.upload_grid_params(g)
    nl.upload_sim_params(s)
    nl.upload_precalc_params(p)
    nl.upload_materials(m)
    nl.upload_granular_params(gr)


def _run(data: dict, use_nl: bool) -> np.ndarray:
    data = dict(data)
    perm = data.pop("sort_perm")
    if use_nl:
        n = data["position"].shape[0]
        nb_idx, nb_cnt = nl.allocate_neighbor_buffers(n)
        nl.build_neighbor_list(
            data["position"], data["cell_start"], data["cell_end"], nb_idx, nb_cnt
        )
        force, _ = nl.compute_step2_nl(
            data["position"],
            data["velocity"],
            data["mass"],
            data["packed_info"],
            data["shear_rate"],
            nb_idx,
            nb_cnt,
            velocity_h=data["velocity_h"],
            pressure_in=data["pressure_in"],
        )
    else:
        force, _ = compute_step2(**data)
    return cupy.asnumpy(unsort(force, perm))[:, :3].astype(np.float64)


# Velocities are multiples of 1/64 so the half4 neighbour copy is exact.
VEL_A = [0.25, -0.125, 0.0625]
VEL_B = [-0.5, 0.375, 0.1875]
RHO_A = 0.6 * float(MATERIALS[WATER].rest_density)  # free-surface particle
RHO_B = 1.05 * float(MATERIALS[WATER].rest_density)  # interior particle
MASS_A = MASS
MASS_B = 0.75 * MASS
POS_B = [0.011, 0.017, -0.006]


def _pair(mat_b: int, moving: bool) -> dict:
    return _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=POS_B,
        vel_a=VEL_A if moving else None,
        vel_b=VEL_B if moving else None,
        density_a=RHO_A,
        density_b=RHO_B,
        mass_a=MASS_A,
        mass_b=MASS_B,
        mat_id_a=WATER,
        mat_id_b=mat_b,
        bclass_a=FLUID,
        bclass_b=FLUID,
    )


@pytest.mark.parametrize("use_nl", [False, True], ids=["K_Step2", "K_Step2_NL"])
@pytest.mark.parametrize("mat_b", [WATER, OIL], ids=["water-water", "water-oil"])
def test_pair_viscous_momentum_and_value(use_nl: bool, mat_b: int) -> None:
    _upload_step2()
    if use_nl:
        _upload_nl()
    a_visc = _run(_pair(mat_b, True), use_nl) - _run(_pair(mat_b, False), use_nl)

    p = np.array([MASS_A, MASS_B])[:, None] * a_visc
    total = p.sum(axis=0)
    scale = np.abs(p).max()
    print(f"  m*a_visc a={p[0]}  b={p[1]}  sum={total}")
    assert scale > 0.0, "no viscous force between the pair"
    assert np.abs(total).max() <= 1e-4 * scale, (
        f"viscous pair momentum not conserved: sum={total}, largest term {scale:.4e} "
        f"(old / rho_j form gave a ratio rho_a/rho_b = {RHO_A / RHO_B:.2f})"
    )

    # Analytic dynamic form: a_a = mu_ab * lapc * m_b * (h - r) / (rho_a rho_b) * (v_b - v_a)
    mu_a = float(MATERIALS[WATER].base_viscosity)
    mu_b = float(MATERIALS[mat_b].base_viscosity)
    mu_ab = 2.0 * mu_a * mu_b / (mu_a + mu_b)
    r = float(np.linalg.norm(POS_B))
    lapc = 45.0 / (math.pi * H**6)
    coeff = mu_ab * lapc * MASS_B * (H - r) / (RHO_A * RHO_B)
    expected_a = coeff * (np.array(VEL_B) - np.array(VEL_A))
    print(f"  a_visc a={a_visc[0]}  expected={expected_a}")
    np.testing.assert_allclose(a_visc[0], expected_a, rtol=2e-3, atol=1e-6)


def _free_surface_block(moving: bool, seed: int = 7) -> dict:
    """7^3 WATER lattice; density falls from rho0 inside to 0.6 rho0 at the faces."""
    n_side = 7
    idx = np.arange(n_side) - n_side // 2
    xs, ys, zs = np.meshgrid(idx, idx, idx, indexing="ij")
    grid = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    pos = grid.astype(np.float32) * SPACING
    depth = (n_side // 2) - np.abs(grid).max(axis=1)  # 0 on the faces
    rho0 = float(MATERIALS[WATER].rest_density)
    rho = rho0 * np.minimum(0.6 + 0.2 * depth, 1.0)
    n = pos.shape[0]
    rng = np.random.default_rng(seed)
    vel = np.zeros((n, 4), dtype=np.float32)
    if moving:
        vel[:, :3] = np.round(rng.uniform(-1.0, 1.0, (n, 3)) * 64.0) / 64.0
    mass = (MASS * rng.uniform(0.8, 1.2, n)).astype(np.float32)
    position = np.concatenate([pos, rho[:, None]], axis=1).astype(np.float32)
    scene = counting_sort_scene(
        position,
        velocity=vel,
        mass=mass,
        packed_info=np.full(n, MAKE_PACKED(WATER, FLUID), dtype=np.uint32),
        grid_params=default_grid_params(),
    )
    density = cupy.ascontiguousarray(scene["position"][:, 3])
    pressure = cupy.zeros(n, dtype=cupy.float32)
    compute_pressure(density, scene["packed_info"], pressure)
    return {
        "position": scene["position"],
        "velocity": scene["velocity"],
        "mass": scene["mass"],
        "packed_info": scene["packed_info"],
        "shear_rate": cupy.zeros(n, dtype=cupy.float32),
        "cell_start": scene["cell_start"],
        "cell_end": scene["cell_end"],
        "pressure_in": pressure,
        "velocity_h": scene["velocity_h"],
        "sort_perm": scene["sort_perm"],
        "_mass_host": mass,
    }


@pytest.mark.parametrize("use_nl", [False, True], ids=["K_Step2", "K_Step2_NL"])
def test_free_surface_block_viscous_momentum(use_nl: bool) -> None:
    _upload_step2()
    if use_nl:
        _upload_nl()
    moving = _free_surface_block(True)
    still = _free_surface_block(False)
    m = moving.pop("_mass_host").astype(np.float64)
    still.pop("_mass_host")
    a_visc = _run(moving, use_nl) - _run(still, use_nl)
    p = m[:, None] * a_visc
    total = p.sum(axis=0)
    gross = np.abs(p).sum(axis=0)
    rel = np.abs(total) / gross
    print(f"  total m*a_visc={total}  sum|m*a|={gross}  rel={rel}")
    assert np.all(gross > 0.0)
    assert np.all(rel < 1e-4), f"viscous momentum drift {rel} is not at float round-off"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
