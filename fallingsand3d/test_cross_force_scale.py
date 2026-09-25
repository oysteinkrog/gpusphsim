"""FLUID-GRANULAR pair forces must obey Newton's third law (bd-r5-review-n28.41).

K_Step2 used to scale the whole force sum of particle i by its own
force_scale (1.0 GRANULAR, 0.02 FLUID). A water-sand pair then gave sand
50x the pressure force water got, so every interface created momentum and
submerged sand was thrown out at the ACCEL_MAX_GRANULAR clamp.

Checks:
  - WATER-SAND pair, both compressed and moving: m_a*a_a + m_b*a_b ~ 0
    for K_Step2.
  - One SAND particle inside a WATER block with a vertical pressure
    gradient gets the same upward push as the water around it (neutral
    buoyancy for equal particle mass), not 50x.

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np
import pytest

from conftest import counting_sort_scene, default_grid_params, unsort
from materials import FLUID, GRANULAR, MATERIALS, SAND, WATER, build_material_array
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
MASS = 0.02  # app particle mass: WATER rho0 * spacing^3
G = 9.81
ACCEL_MAX_GRANULAR = 200.0


def _upload(module_uploads) -> None:
    grid, sim, precalc, mats, gran = module_uploads
    grid(default_grid_params())
    sim(
        build_sim_params(
            smoothing_length=H, particle_mass=MASS, particle_spacing=SPACING
        )
    )
    precalc(build_precalc_params(smoothing_length=H, viscosity=1.0))
    mats(build_material_array())
    gran(build_granular_params(mu0=1.0))


def _upload_step2() -> None:
    _upload(
        (
            upload_grid_params,
            upload_sim_params,
            upload_precalc_params,
            upload_materials,
            upload_granular_params,
        )
    )


def _mixed_pair() -> dict:
    """Compressed WATER and SAND particle with different masses and velocities."""
    rho0_w = float(MATERIALS[WATER].rest_density)
    rho0_s = float(MATERIALS[SAND].rest_density)
    return _make_two_particle_setup(
        pos_a=[0.0, 0.0, 0.0],
        pos_b=[0.011, 0.017, -0.006],
        vel_a=[0.3, -0.2, 0.1],
        vel_b=[-0.5, 0.4, 0.2],
        density_a=1.3 * rho0_w,
        density_b=1.2 * rho0_s,
        mass_a=MASS,
        mass_b=0.75 * MASS,
        mat_id_a=WATER,
        mat_id_b=SAND,
        bclass_a=FLUID,
        bclass_b=GRANULAR,
    )


def _check_momentum(acc: np.ndarray, masses: np.ndarray, label: str) -> None:
    p = masses[:, None] * acc[:, :3]
    total = p.sum(axis=0)
    scale = np.abs(p).max()
    print(f"  {label}: m*a water={p[0]}  sand={p[1]}  sum={total}")
    assert scale > 0.0, f"{label}: no force between the pair"
    assert np.all(np.isfinite(acc)), f"{label}: non-finite force"
    assert np.abs(total).max() <= 1e-3 * scale, (
        f"{label}: pair momentum not conserved, sum={total}, largest term {scale:.4e}"
    )


def test_mixed_pair_momentum_step2() -> None:
    _upload_step2()
    data = _mixed_pair()
    perm = data.pop("sort_perm")
    force, _ = compute_step2(**data)
    acc = cupy.asnumpy(unsort(force, perm))
    _check_momentum(acc, np.array([MASS, 0.75 * MASS]), "K_Step2")


def test_submerged_sand_not_ejected() -> None:
    """SAND in a WATER block with a vertical pressure gradient.

    Densities are set so the WATER pressure grows linearly with depth.
    The sand particle has the same mass as a water particle, so it should
    feel about the same upward pressure acceleration as its water
    neighbours (neutral buoyancy) and stay far below ACCEL_MAX_GRANULAR.
    Before the fix it got 50x the water value.
    """
    _upload_step2()
    k = float(MATERIALS[WATER].eos_stiffness)
    gam = float(MATERIALS[WATER].eos_gamma)
    rho0 = float(MATERIALS[WATER].rest_density)

    n_side = 9
    idx = np.arange(n_side) - n_side // 2
    xs, ys, zs = np.meshgrid(idx, idx, idx, indexing="ij")
    pos = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32) * SPACING
    y_top = pos[:, 1].max()
    # Pressure grows 0.5*k per 0.1 m of depth, starting at 0.1*k at the top.
    p = k * (0.1 + 5.0 * (y_top - pos[:, 1]))
    rho = rho0 * (p / k + 1.0) ** (1.0 / gam)

    n = pos.shape[0]
    centre = int(np.argmin(np.abs(pos).sum(axis=1)))
    packed = np.full(n, MAKE_PACKED(WATER, FLUID), dtype=np.uint32)
    packed[centre] = MAKE_PACKED(SAND, GRANULAR)
    rho[centre] = rho0  # below SAND rho0 (4000), so p_sand = 0 as in the app

    position = np.concatenate([pos, rho[:, None].astype(np.float32)], axis=1)
    velocity = np.zeros((n, 4), dtype=np.float32)
    mass = np.full(n, MASS, dtype=np.float32)
    scene = counting_sort_scene(
        position,
        velocity=velocity,
        mass=mass,
        packed_info=packed,
        grid_params=default_grid_params(),
    )
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
        velocity_h=scene["velocity_h"],
        pressure_in=pressure,
    )
    acc = cupy.asnumpy(unsort(force, scene["sort_perm"]))

    # Water particles one spacing above/below the sand see the same gradient.
    above = int(np.argmin(np.abs(pos - np.array([0, SPACING, 0])).sum(axis=1)))
    below = int(np.argmin(np.abs(pos - np.array([0, -SPACING, 0])).sum(axis=1)))
    a_water = 0.5 * (acc[above, 1] + acc[below, 1])
    a_sand = acc[centre, 1]
    print(
        f"  water a_y={a_water:.3f}  sand a_y={a_sand:.3f}  ratio={a_sand / a_water:.3f}"
    )
    assert np.all(np.isfinite(acc))
    assert a_water > 0.0, f"water should be pushed up by the gradient, got {a_water}"
    assert 0.0 < a_sand <= 1.2 * a_water, (
        f"sand pressure accel {a_sand:.3f} should match the water value {a_water:.3f}, "
        f"not exceed it (was 50x before bd-r5-review-n28.41)"
    )
    assert abs(a_sand) < 0.25 * ACCEL_MAX_GRANULAR


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
