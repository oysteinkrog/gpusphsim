"""K_Step1 inactive-particle skip (bd-r5-review-n28.159).

Sleeping and non-rigid STATIC particles run the neighbour loop only when
(substep + index // 32) % 8 == 0 or when force_refresh is set. On other calls they
write density from position.w, pressure and pr2_omega from it, and leave every
other output buffer untouched. MAT_DEAD slots write fixed outputs.

Checks:
  - skip call: inactive outputs come from position.w and the carried buffers,
    active particles match the always-compute call bitwise
  - refresh call and force_refresh: inactive outputs match the full compute
  - DEAD slot: zero heat, exposure, shear, vorticity, normal; density 1
  - heat into a sleeping pile over 1 s matches the always-on path within 5%

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np

from conftest import counting_sort_scene, default_grid_params
from materials import (
    DEAD,
    LAVA,
    SAND,
    STONE,
    build_interaction_matrix,
    build_material_array,
)
from step1 import (
    build_precalc_params,
    build_sim_params,
    compute_step1,
    upload_grid_params,
    upload_interactions,
    upload_materials,
    upload_precalc_params,
    upload_sim_params,
)

H = 0.04
SPACING = 0.02
FLUID, GRANULAR, STATIC = 0, 1, 3
SLEEP_BIT = 0x400
PERIOD = 8


def _packed(mat: int, beh: int, sleeping: bool = False) -> int:
    return (mat & 0xFF) | ((beh & 0x3) << 8) | (SLEEP_BIT if sleeping else 0)


def _upload(dt: float) -> None:
    upload_grid_params(default_grid_params())
    upload_sim_params(
        build_sim_params(smoothing_length=H, particle_spacing=SPACING, dt=dt)
    )
    upload_precalc_params(build_precalc_params(smoothing_length=H))
    upload_materials(build_material_array())
    upload_interactions(build_interaction_matrix())


def _scene(hot_lava: bool = True):
    """Stone slab, lava layer, sleeping sand block, one DEAD slot."""
    pts, packed, temp = [], [], []
    for ix in range(6):
        for iz in range(6):
            x, z = ix * SPACING, iz * SPACING
            pts.append((x, 0.0, z))
            packed.append(_packed(STONE, STATIC))
            temp.append(293.0)
            pts.append((x, SPACING, z))
            packed.append(_packed(LAVA, FLUID))
            temp.append(1500.0 if hot_lava else 293.0)
            for iy in range(3):
                pts.append((x, (2 + iy) * SPACING, z))
                packed.append(_packed(SAND, GRANULAR, sleeping=True))
                temp.append(293.0)
    pts.append((0.05, 0.03, 0.05))
    packed.append(_packed(DEAD, STATIC))
    temp.append(293.0)
    n = len(pts)
    pos = np.zeros((n, 4), np.float32)
    pos[:, :3] = np.asarray(pts, np.float32)
    mass = np.full(n, 0.008, np.float32)
    mass[-1] = 0.0
    return pos, mass, np.asarray(packed, np.uint32), np.asarray(temp, np.float32)


def _sorted(pos, mass, packed, temp):
    sc = counting_sort_scene(
        pos,
        mass=mass,
        packed_info=packed,
        temperature=temp,
        grid_params=default_grid_params(),
    )
    return sc


class _Bufs:
    def __init__(self, n: int) -> None:
        f = cupy.float32
        self.density = cupy.zeros(n, f)
        self.shear = cupy.zeros(n, f)
        self.dTdt = cupy.zeros(n, f)
        self.eh = cupy.zeros(n, f)
        self.ec = cupy.zeros(n, f)
        self.vort = cupy.zeros((n, 4), f)
        self.normal = cupy.zeros((n, 4), f)
        self.pressure = cupy.zeros(n, f)
        self.pr2 = cupy.zeros((n, 2), f)

    def fill(self, v: float) -> None:
        for a in (
            self.density,
            self.shear,
            self.dTdt,
            self.eh,
            self.ec,
            self.vort,
            self.normal,
            self.pressure,
            self.pr2,
        ):
            a.fill(v)

    def snap(self) -> dict:
        return {
            k: cupy.asnumpy(getattr(self, k))
            for k in (
                "density",
                "shear",
                "dTdt",
                "eh",
                "ec",
                "vort",
                "normal",
                "pressure",
                "pr2",
            )
        }


def _run(
    sc,
    pos_sorted,
    bufs: _Bufs,
    temp_sorted=None,
    frame=None,
    force=True,
):
    compute_step1(
        pos_sorted,
        sc["velocity"],
        sc["mass"],
        sc["packed_info"],
        sc["temperature"] if temp_sorted is None else temp_sorted,
        sc["cell_start"],
        sc["cell_end"],
        density_out=bufs.density,
        shear_rate_out=bufs.shear,
        dTdt_out=bufs.dTdt,
        exposure_heat_out=bufs.eh,
        exposure_corrode_out=bufs.ec,
        vorticity_out=bufs.vort,
        normal_out=bufs.normal,
        velocity_h=sc["velocity_h"],
        pressure_out=bufs.pressure,
        pr2_omega_out=bufs.pr2,
        frame_counter=frame,
        force_refresh=force,
    )
    cupy.cuda.Device().synchronize()


def _classes(sc):
    pi = cupy.asnumpy(sc["packed_info"])
    mat = pi & 0xFF
    beh = (pi >> 8) & 0x3
    dead = mat == DEAD
    inactive = (((pi & SLEEP_BIT) != 0) | (beh == STATIC)) & ~dead
    return inactive, dead


def test_skip_and_refresh_outputs() -> None:
    _upload(dt=0.001)
    pos, mass, packed, temp = _scene()
    sc = _sorted(pos, mass, packed, temp)
    n = pos.shape[0]
    inactive, dead = _classes(sc)
    active = ~inactive & ~dead
    assert inactive.sum() > 100 and dead.sum() == 1 and active.sum() > 10

    # Reference: every particle computes (frame_counter None). It also packs
    # the density into position.w, as K_Step1 does every substep.
    pos_ref = sc["position"].copy()
    ref_b = _Bufs(n)
    _run(sc, pos_ref, ref_b)
    ref = ref_b.snap()

    frame_val = 3
    frame = cupy.asarray([frame_val], dtype=cupy.uint32)
    refresh = ((frame_val + np.arange(n) // 32) % PERIOD) == 0
    skip = inactive & ~refresh

    # Skip call: carried buffers hold a sentinel so an untouched output shows.
    b = _Bufs(n)
    b.fill(777.0)
    pos_run = pos_ref.copy()  # .w = reference density
    _run(sc, pos_run, b, frame=frame, force=False)
    out = b.snap()

    # Active particles and inactive particles on their refresh substep: same
    # as the full compute. Uniform input temperature per material is not
    # uniform here (hot lava), so inactive refresh dTdt differs by the
    # backward-Euler divisor; compare it separately.
    full = ~skip
    for k in ("density", "shear", "eh", "ec", "vort", "normal", "pressure", "pr2"):
        np.testing.assert_array_equal(out[k][full], ref[k][full], err_msg=k)
    np.testing.assert_array_equal(
        out["dTdt"][active | dead], ref["dTdt"][active | dead]
    )
    ir = inactive & refresh
    assert ir.sum() > 0
    # Backward Euler: same sign, never larger in magnitude.
    d_ref, d_out = ref["dTdt"][ir], out["dTdt"][ir]
    assert np.all(np.abs(d_out) <= np.abs(d_ref) * (1 + 1e-6) + 1e-9)
    assert np.all(np.sign(d_out) == np.sign(d_ref))

    # Skipped: density from .w, pressure and pr2 match the full compute
    # bitwise (same density in, same formula), the rest untouched.
    assert skip.sum() > 50
    np.testing.assert_array_equal(
        out["density"][skip], cupy.asnumpy(pos_ref[:, 3])[skip]
    )
    np.testing.assert_array_equal(out["density"][skip], ref["density"][skip])
    np.testing.assert_array_equal(out["pressure"][skip], ref["pressure"][skip])
    np.testing.assert_array_equal(out["pr2"][skip][:, 0], ref["pr2"][skip][:, 0])
    np.testing.assert_array_equal(out["pr2"][skip][:, 1], 0.0)
    for k in ("shear", "dTdt", "eh", "ec", "vort", "normal"):
        assert np.all(out[k][skip] == 777.0), k

    # DEAD slot: fixed outputs.
    assert out["density"][dead][0] == 1.0
    for k in ("shear", "dTdt", "eh", "ec", "vort", "normal"):
        assert np.all(out[k][dead] == 0.0), k

    # force_refresh: every inactive particle computes.
    b2 = _Bufs(n)
    b2.fill(777.0)
    _run(sc, pos_ref.copy(), b2, frame=frame, force=True)
    out2 = b2.snap()
    for k in ("density", "shear", "eh", "ec", "vort", "normal", "pressure", "pr2"):
        np.testing.assert_array_equal(out2[k], ref[k], err_msg=k)


def _heat_run(sc, n, dt, steps, use_skip):
    pos_run = sc["position"].copy()
    temp = sc["temperature"].copy()
    b = _Bufs(n)
    frame = cupy.zeros(1, dtype=cupy.uint32)
    for s in range(steps):
        frame.fill(s + 1)
        _run(
            sc,
            pos_run,
            b,
            temp_sorted=temp,
            frame=frame if use_skip else None,
            force=(s == 0),
        )
        temp += b.dTdt * np.float32(dt)
    return cupy.asnumpy(temp)


def test_heat_into_sleeping_pile_matches_always_on() -> None:
    dt = 1.0 / 240.0
    _upload(dt=dt)
    pos, mass, packed, temp = _scene(hot_lava=True)
    sc = _sorted(pos, mass, packed, temp)
    n = pos.shape[0]
    pi = cupy.asnumpy(sc["packed_info"])
    sand = (pi & 0xFF) == SAND
    steps = 240  # 1 s
    t0 = cupy.asnumpy(sc["temperature"])
    t_on = _heat_run(sc, n, dt, steps, use_skip=False)
    t_skip = _heat_run(sc, n, dt, steps, use_skip=True)
    gain_on = float(np.sum(t_on[sand] - t0[sand]))
    gain_skip = float(np.sum(t_skip[sand] - t0[sand]))
    print(f"sand heat gain over 1 s: always-on {gain_on:.2f} K, skip {gain_skip:.2f} K")
    assert gain_on > 1.0
    assert abs(gain_skip - gain_on) <= 0.05 * abs(gain_on)
    assert np.all(np.isfinite(t_skip))
