"""Implicit viscous self-term in K_Step2 (bd-r5-review-n28.52).

The viscous accel of particle i is fs_i * sum_j c_j (v_j - v_i). With
explicit Euler, dt * fs_i * c_tot above 2 makes a velocity mode that
alternates between neighbours grow every substep. At dt = 1 ms that
number was 18.75 for LAVA and 9.4 for MUD with the old force_scale
viscosity form (before bd-r5-review-n28.79), and is about 1.9 for
resting SAND at mu_max. Step2 now scales the viscous sum by 1 / (1 + dt * fs_i * c_tot),
which is backward Euler on the self-term and stable for any value.

Check: a lattice block at uniform density with an alternating velocity
field (v_x = +A / -A on neighbouring sites). One substep v' = v + dt * a
must shrink every interior velocity (|v'| < |v|). For the smaller
coefficients (WATER, OIL, SAND at mu_max) the sign must also stay the
same, so there is no sign flip between substeps. Before the fix LAVA
gave v'/v of about -19.

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
from materials import FLUID, GRANULAR, LAVA, MUD, OIL, SAND, WATER, build_material_array
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
from test_step2 import MAKE_PACKED

H = 0.04
SPACING = 0.02
MASS = 0.02  # app particle mass for every material
RHO = MASS / SPACING**3  # lattice SPH density, 2500
DT = 0.001
AMP = 0.1
N_SIDE = 9
EDGE = 2  # interior = at least this many sites from a face

CASES = [
    # (material, behaviour, max |v'/v|, sign must hold)
    (LAVA, FLUID, 0.95, False),
    (MUD, FLUID, 0.95, False),
    # OIL and WATER use the dynamic form mu / (rho_i rho_j) since
    # bd-r5-review-n28.79 (nu = mu / rho0, 50x below the old force_scale
    # form), so one substep only removes about 2% and 0.4% of the mode.
    (OIL, FLUID, 0.99, True),
    (WATER, FLUID, 0.999, True),
    (SAND, GRANULAR, 0.95, True),
]


def _upload(uploads) -> None:
    grid, sim, precalc, mats, gran = uploads
    grid(default_grid_params())
    sim(
        build_sim_params(
            dt=DT, smoothing_length=H, particle_mass=MASS, particle_spacing=SPACING
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


def _lattice(mode: str):
    idx = np.arange(N_SIDE) - N_SIDE // 2
    xs, ys, zs = np.meshgrid(idx, idx, idx, indexing="ij")
    ijk = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    pos = ijk.astype(np.float32) * SPACING
    if mode == "checker":
        sign = np.where(ijk.sum(axis=1) % 2 == 0, 1.0, -1.0)
    else:  # alternating planes along x, velocity along y
        sign = np.where(ijk[:, 0] % 2 == 0, 1.0, -1.0)
    interior = np.all(np.abs(ijk) <= N_SIDE // 2 - EDGE, axis=1)
    return pos, sign.astype(np.float32), interior


def _run(mat, behav, mode: str):
    pos, sign, interior = _lattice(mode)
    n = pos.shape[0]
    axis = 0 if mode == "checker" else 1
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, axis] = AMP * sign
    position = np.concatenate([pos, np.full((n, 1), RHO, np.float32)], axis=1)
    packed = np.full(n, MAKE_PACKED(mat, behav), dtype=np.uint32)
    scene = counting_sort_scene(
        position,
        velocity=vel,
        mass=np.full(n, MASS, np.float32),
        packed_info=packed,
        grid_params=default_grid_params(),
    )
    density = cupy.ascontiguousarray(scene["position"][:, 3])
    pressure = cupy.zeros(n, dtype=cupy.float32)
    compute_pressure(density, scene["packed_info"], pressure)
    shear = cupy.zeros(n, dtype=cupy.float32)  # resting: eta = mu_max for SAND
    force, _ = compute_step2(
        scene["position"],
        scene["velocity"],
        scene["mass"],
        scene["packed_info"],
        shear,
        scene["cell_start"],
        scene["cell_end"],
        velocity_h=scene["velocity_h"],
        pressure_in=pressure,
    )
    acc = cupy.asnumpy(unsort(force, scene["sort_perm"]))
    v0 = vel[interior, axis]
    v1 = v0 + DT * acc[interior, axis]
    return v1 / v0


@pytest.mark.parametrize("mode", ["checker", "planes"])
@pytest.mark.parametrize(
    "mat,behav,max_ratio,keep_sign", CASES, ids=[str(c[0]) for c in CASES]
)
def test_alternating_mode_decays(mat, behav, max_ratio, keep_sign, mode):
    _upload_step2()
    ratio = _run(mat, behav, mode)
    print(f"  mat={mat} {mode} v'/v min={ratio.min():+.4f} max={ratio.max():+.4f}")
    assert np.all(np.isfinite(ratio))
    assert np.abs(ratio).max() < max_ratio, (
        f"alternating mode must decay in one substep, |v'/v| max={np.abs(ratio).max():.3f}"
    )
    if keep_sign:
        assert ratio.min() > 0.0, (
            f"velocity flips sign relative to neighbours, v'/v min={ratio.min():.3f}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
