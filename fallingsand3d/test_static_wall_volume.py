"""STATIC walls use one volume convention in WCSPH, PBF and DFSPH (bd-r5-review-n28.116).

K_Step1 wrote the SPH-summed density for a STATIC wall particle (about
1200-2500 for STONE) while PBF and DFSPH wrote rest_density (STONE 6500,
METAL 19500, WOOD 1500). Every m_j / rho_j sum then saw the same wall 2.6x to
5x bigger under WCSPH, and DFSPH wall stiffness depended on the wall material.

Now all three solvers store rest_density for STATIC walls and divide the wall
dTdt by it. DFSPH alpha, div_v, drho and XSPH give a STATIC neighbour the
Akinci volume m_j / rho0_i with the fluid's own rest density.

Tests:
- K_Step1 stores rest_density for STONE wall particles.
- A hot two-layer STONE wall heats a WATER block at the same rate under all
  three solvers (zero gravity, reactions off, equal fixed dt).
- DFSPH WATER dropped on a STONE floor and on a WOOD floor settles at the same
  height: the wall material no longer changes the wall stiffness.
"""

import dataclasses

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler

        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}
        fn = getattr(_compiler, "_get_arch_for_options_for_nvrtc", None)
        if fn is not None and hasattr(fn, "_cache"):
            fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402

from materials import MATERIALS, STONE, WATER, WOOD  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from world import T_AMBIENT, World  # noqa: E402

MAX_PARTICLES = 20_000
SPACING = 0.02
DT = 1.0 / 1000.0
T_WALL = 1000.0
SHORT = 5
LONG = 50


def _make_sim(w, profile_name, dt, monkeypatch, gravity=0.0):
    profile = dataclasses.replace(
        PROFILES[profile_name], dt=dt, fixed_dt=True, max_substeps=1
    )
    s = Simulation(w, dt=dt, speed=1.0, fixed_dt=True, max_substeps=1)
    s.set_solver_profile(profile)
    s.set_gravity(gravity)
    s._spawn_substep = s._damping_duration
    monkeypatch.setattr(s, "_run_reactions_spawn", lambda *a, **k: None)
    return s


def _substep(s, n):
    n = s._substep_prologue(n)
    s._count_substep_sort(n)
    s._run_graph_body(n)
    s._substep_epilogue(n)
    return n


def _wall_heat_scene():
    w = World(max_particles=MAX_PARTICLES)
    w.packed_info[:] = 0
    w._high_water = 0
    # 10 x 10 x 10 WATER at x < 0, a two-layer 10 x 10 STONE wall at x > 0.
    n = w.spawn_cube((-0.2, -0.1, -0.1), (0.0, 0.1, 0.1), WATER, spacing=SPACING)
    n_water = n
    n += w.spawn_cube(
        (0.0, -0.1, -0.1), (2 * SPACING, 0.1, 0.1), STONE, spacing=SPACING
    )
    assert n - n_water == 200
    return w, n, n_water


def water_heating(profile_name, monkeypatch):
    """Mean WATER temperature rise (K) after SHORT and LONG substeps."""
    w, n, n_water = _wall_heat_scene()
    s = _make_sim(w, profile_name, DT, monkeypatch)

    def water_rise(n):
        is_water = (w.packed_info[:n] & 0xFF) == WATER
        count = cp.maximum(cp.count_nonzero(is_water), 1)
        return cp.sum(cp.where(is_water, w.temperature[:n] - T_AMBIENT, 0.0)) / count

    # One warm-up substep so PBF and DFSPH read a real previous-substep
    # density, then set the wall hot and the water to ambient.
    n = _substep(s, n)
    mat = w.packed_info[:n] & 0xFF
    w.temperature[:n] = cp.where(mat == STONE, T_WALL, T_AMBIENT).astype(cp.float32)

    rises = {}
    for k in range(1, LONG + 1):
        n = _substep(s, n)
        if k in (SHORT, LONG):
            rises[k] = water_rise(n)  # stays on the GPU until the end

    # Stored STATIC density after the last substep (sorted order).
    smat = w.sorted_packed_info[:n] & 0xFF
    wall_rho = w.sorted_density[:n][smat == STONE]
    is_water = (w.packed_info[:n] & 0xFF) == WATER
    return {
        "short": float(rises[SHORT]),
        "long": float(rises[LONG]),
        "water_n": int(cp.count_nonzero(is_water)),
        "water_spawned": n_water,
        "wall_rho_min": float(wall_rho.min()),
        "wall_rho_max": float(wall_rho.max()),
        "finite": bool(cp.isfinite(w.temperature[:n]).all()),
    }


def _rel(a, b):
    return abs(a - b) / b


def test_hot_stone_wall_heats_water_equally_across_solvers(monkeypatch):
    rho_stone = MATERIALS[STONE].rest_density
    results = {}
    for name in ("WCSPH", "PBF", "DFSPH"):
        with monkeypatch.context() as m:
            results[name] = water_heating(name, m)
        r = results[name]
        print(
            f"  {name}: water n={r['water_n']}/{r['water_spawned']} "
            f"dT({SHORT})={r['short']:.4f} K dT({LONG})={r['long']:.4f} K "
            f"wall rho {r['wall_rho_min']:.1f}..{r['wall_rho_max']:.1f}"
        )
    for name, r in results.items():
        assert r["finite"], name
        assert r["water_n"] == r["water_spawned"], f"{name}: water changed material"
        # Every solver stores rest_density for the STATIC wall.
        assert r["wall_rho_min"] == pytest.approx(rho_stone), name
        assert r["wall_rho_max"] == pytest.approx(rho_stone), name

    wc = results["WCSPH"]
    assert wc["short"] > 0.5, f"WCSPH water barely heated: {wc['short']:.4f} K"
    for a, b in (("PBF", "WCSPH"), ("DFSPH", "WCSPH"), ("DFSPH", "PBF")):
        assert _rel(results[a]["short"], results[b]["short"]) < 0.05, (
            f"dT({SHORT}): {a} {results[a]['short']:.4f} K vs "
            f"{b} {results[b]['short']:.4f} K"
        )
        assert _rel(results[a]["long"], results[b]["long"]) < 0.10, (
            f"dT({LONG}): {a} {results[a]['long']:.4f} K vs "
            f"{b} {results[b]['long']:.4f} K"
        )


def dfsph_floor_rest(floor_mat, monkeypatch, steps=400):
    """Drop a WATER block on a floor; return the WATER heights above the floor top."""
    w = World(max_particles=MAX_PARTICLES)
    w.packed_info[:] = 0
    w._high_water = 0
    floor_top = 0.0
    n = w.spawn_cube(
        (-0.16, -3 * SPACING, -0.16),
        (0.16, floor_top, 0.16),
        floor_mat,
        spacing=SPACING,
    )
    n_floor = n
    n += w.spawn_cube((-0.06, 0.02, -0.06), (0.06, 0.14, 0.06), WATER, spacing=SPACING)
    assert n - n_floor > 200
    dt = PROFILES["DFSPH"].dt
    s = _make_sim(w, "DFSPH", dt, monkeypatch, gravity=-9.81)
    for _ in range(steps):
        n = _substep(s, n)
    pos = w.position[:n].get()
    mat = w.packed_info[:n].get() & 0xFF
    floor_y = pos[mat == floor_mat, 1].max()
    water_y = pos[mat == WATER, 1]
    return {
        "floor_y": float(floor_y),
        "gap_min": float(water_y.min() - floor_y),
        "gap_p5": float(np.percentile(water_y, 5) - floor_y),
        "water_mean": float(water_y.mean() - floor_y),
        "finite": bool(np.isfinite(pos[:, :3]).all()),
    }


def test_dfsph_floor_stiffness_does_not_depend_on_wall_material(monkeypatch):
    res = {}
    for mat in (STONE, WOOD):
        with monkeypatch.context() as m:
            res[mat] = dfsph_floor_rest(mat, m)
        r = res[mat]
        print(
            f"  {MATERIALS[mat].name}: gap min={r['gap_min'] * 1e3:.3f} mm "
            f"p5={r['gap_p5'] * 1e3:.3f} mm mean height={r['water_mean'] * 1e3:.3f} mm"
        )
    st, wd = res[STONE], res[WOOD]
    assert st["finite"] and wd["finite"]
    # Water must rest on the floor, not fall through it.
    assert st["gap_min"] > -0.5 * SPACING and wd["gap_min"] > -0.5 * SPACING
    # Same penetration and resting height on both floors (within 0.2 mm).
    assert abs(st["gap_min"] - wd["gap_min"]) < 2e-4
    assert abs(st["gap_p5"] - wd["gap_p5"]) < 2e-4
    assert abs(st["water_mean"] - wd["water_mean"]) < 2e-4


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-s"])
