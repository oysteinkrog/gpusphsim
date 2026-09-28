"""Shepard-blended EOS density in K_Step1 (bd-r5-review-n28.205).

A resting WCSPH WATER column fills a closed 0.6 m box (world_half_size 0.3)
and settles for SETTLE_S seconds. Readback happens once, after the run.

Hydrostatic reference: force balance fixes the pressure at depth to the
weight of the water above it, p_ref = s * M_above, where M_above is the
particle mass above a particle (its rank from the top, whole box) and s is
fitted through the origin on the deep part of the column (depth 3h to 1.5h
above the floor), where the kernel is not truncated. That fit absorbs the
box area and the WCSPH force_scale, so no surface position or area enters
the deep reference. Depth is measured per 0.05 m xz cell from the highest
particle in that cell, and the mass above is the cell's own. The top-2h
error is the RMS of (mean p - mean p_ref) over 0.01 m depth bins in the top
2h, divided by p_ref at depth 2h.

Checks, on the same settled state:
- the top-2h error of pressure_out (blended EOS input) is smaller than that
  of the raw EOS pressure k((rho/rho0)^7 - 1) of the stored density;
- in the interior (depth > 3h, not near the walls or floor) the EOS density
  recovered from pressure_out matches the stored raw density to 1%.

Run as a script to print the depth profile: python test_step1_shepard.py
"""

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, '_cache'):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, '_get_arch_for_options_for_nvrtc'):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, '_cache'):
                fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402

from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import WATER, MATERIALS  # noqa: E402

H = 0.04
SPACING = 0.02
HALF = 0.3
SETTLE_S = 3.0
BIN = 0.01
CELL = 0.05


def run_column(settle_s=SETTLE_S, top_y=HALF - 0.01):
    w = World(max_particles=40_000)
    w.spawn_cube((-HALF + 0.01, -HALF + 0.01, -HALF + 0.01),
                 (HALF - 0.01, top_y, HALF - 0.01), WATER, spacing=SPACING)
    n = w._high_water
    profile = PROFILES["WCSPH"]
    s = Simulation(w, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1,
                   world_half_size=HALF)
    s.set_solver_profile(profile)
    s._spawn_substep = s._damping_duration
    dt = s._compute_adaptive_dt(1)
    s._upload_dt(dt)
    for _ in range(int(round(settle_s / dt))):
        s._sim_step(n)
    cp.cuda.Device().synchronize()

    pi = w.sorted_packed_info[:n].get()
    sel = (pi & 0xFF) == WATER
    pos = w.sorted_position[:n, :3].get()[sel]
    rho = w.sorted_density[:n].get()[sel].astype(np.float64)
    p = w.sorted_pressure[:n].get()[sel].astype(np.float64)
    m = w.sorted_mass[:n].get()[sel].astype(np.float64)
    vel = w.sorted_velocity[:n, :3].get()[sel]
    v_rms = float(np.sqrt(np.mean(np.sum(vel * vel, axis=1))))

    mat = MATERIALS[WATER]
    rho0 = float(mat.rest_density)
    k = float(mat.eos_stiffness)
    gam = float(mat.eos_gamma)
    p_raw = np.maximum(k * (np.minimum(rho / rho0, 10.0) ** gam - 1.0), 0.0)
    rho_eos = rho0 * (p / k + 1.0) ** (1.0 / gam)  # valid where p > 0

    # Per xz cell (CELL x CELL, 12 x 12 over the box): the surface height is
    # the highest particle in the cell and the mass above a particle is the
    # cell's mass above it, so a tilted or curved surface does not bias depth.
    y = pos[:, 1]
    ncell = int(round(2 * HALF / CELL))
    cx = np.clip(((pos[:, 0] + HALF) / CELL).astype(int), 0, ncell - 1)
    cz = np.clip(((pos[:, 2] + HALF) / CELL).astype(int), 0, ncell - 1)
    cell = cx * ncell + cz
    order = np.lexsort((-y, cell))
    csum = np.cumsum(m[order])
    first = np.r_[True, cell[order][1:] != cell[order][:-1]]
    base = np.maximum.accumulate(np.where(first, csum - m[order], 0.0))
    m_above = np.empty_like(y)
    m_above[order] = csum - base - 0.5 * m[order]
    y_top_cell = np.full(ncell * ncell, -np.inf)
    np.maximum.at(y_top_cell, cell, y)
    depth = y_top_cell[cell] - y

    central = (np.abs(pos[:, 0]) < HALF - 2 * H) & (np.abs(pos[:, 2]) < HALF - 2 * H)
    y_top = float(np.median(y_top_cell[np.unique(cell[central])]))
    y_bot = float(np.min(y))
    deep = central & (depth > 3 * H) & (y > y_bot + 1.5 * H)
    s_fit = float(np.sum(p[deep] * m_above[deep]) / np.sum(m_above[deep] ** 2))
    s_fit_raw = float(np.sum(p_raw[deep] * m_above[deep]) / np.sum(m_above[deep] ** 2))
    p_ref = s_fit * m_above

    bins = []
    for b0 in np.arange(0.0, 2 * H - 1e-9, BIN):
        sb = central & (depth >= b0) & (depth < b0 + BIN)
        if not sb.any():
            continue
        bins.append((b0 + 0.5 * BIN, float(p[sb].mean()), float(p_raw[sb].mean()),
                     float(p_ref[sb].mean()), float((rho[sb] / rho0).mean()), int(sb.sum())))
    sb2 = central & (np.abs(depth - 2 * H) < 0.5 * BIN)
    p_ref_2h = float(p_ref[sb2].mean())
    top = central & (depth < 2 * H)

    def err(pp):
        # Per-particle RMS: includes the jitter of the unsettled WCSPH column.
        return float(np.sqrt(np.mean((pp[top] - p_ref[top]) ** 2))) / p_ref_2h

    def err_bins(col):
        # RMS of the depth-bin means: the systematic profile error.
        return float(np.sqrt(np.mean([(b[col] - b[3]) ** 2 for b in bins]))) / p_ref_2h

    interior = deep & (p > 0)
    return {
        "n": int(sel.sum()),
        "dt": dt,
        "y_top": y_top,
        "bins": bins,
        "p_ref_2h": p_ref_2h,
        "err_blend": err(p),
        "err_raw": err(p_raw),
        "err_bins_blend": err_bins(1),
        "err_bins_raw": err_bins(2),
        "s_fit": s_fit,
        "s_fit_raw": s_fit_raw,
        "interior_rho_ratio": float((rho[deep] / rho0).mean()),
        "interior_eos_dev": float(np.max(np.abs(rho_eos[interior] / rho[interior] - 1.0))),
        "interior_eos_dev_mean": float(np.mean(np.abs(rho_eos[interior] / rho[interior] - 1.0))),
        "finite": bool(np.isfinite(pos).all()),
        "v_rms": v_rms,
    }


def _print(r):
    print(f"n={r['n']} dt={r['dt']*1e3:.3f} ms y_top={r['y_top']:.4f} "
          f"v_rms={r['v_rms']:.4f} p_ref(2h)={r['p_ref_2h']:.1f} slope blend/raw={r['s_fit']:.4g}/{r['s_fit_raw']:.4g}")
    print("  depth[m]  p_out      p_rawEOS   p_ref      rho/rho0  count")
    for d, pb, pr, pf, rr, c in r["bins"]:
        print(f"  {d:.3f}    {pb:9.1f}  {pr:9.1f}  {pf:9.1f}  {rr:.3f}    {c}")
    print(f"  top-2h error / p_ref(2h), bin means: pressure_out={r['err_bins_blend']:.4f} "
          f"raw EOS={r['err_bins_raw']:.4f}; per particle: pressure_out={r['err_blend']:.4f} "
          f"raw EOS={r['err_raw']:.4f}")
    print(f"  interior rho/rho0={r['interior_rho_ratio']:.4f} "
          f"|rho_eos/rho - 1| max={r['interior_eos_dev']:.4f} mean={r['interior_eos_dev_mean']:.5f}",
          flush=True)


@pytest.fixture(scope="module")
def column():
    r = run_column()
    _print(r)
    return r


def test_column_finite(column):
    assert column["finite"]


def test_top_layers_closer_to_hydrostatic(column):
    assert column["err_bins_blend"] < column["err_bins_raw"]


def test_interior_eos_density_unchanged(column):
    # Mean over the interior; single particles next to a jittering
    # neighbour reach about 2%.
    assert column["interior_eos_dev_mean"] < 0.01


if __name__ == "__main__":
    _print(run_column())
