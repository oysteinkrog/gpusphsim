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
- DFSPH alpha of WATER on a STONE, WOOD or METAL floor is the same: the wall
  material no longer changes the DFSPH boundary gradient terms.
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

from materials import MATERIALS, METAL, STONE, WATER, WOOD  # noqa: E402
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
            f"dT({SHORT})={r['short']:.6f} K dT({LONG})={r['long']:.6f} K "
            f"wall rho {r['wall_rho_min']:.1f}..{r['wall_rho_max']:.1f}"
        )
    for name, r in results.items():
        assert r["finite"], name
        assert r["water_n"] == r["water_spawned"], f"{name}: water changed material"
        # Every solver stores rest_density for the STATIC wall.
        assert r["wall_rho_min"] == pytest.approx(rho_stone), name
        assert r["wall_rho_max"] == pytest.approx(rho_stone), name

    wc = results["WCSPH"]
    assert wc["short"] > 1e-3, f"WCSPH water barely heated: {wc['short']:.6f} K"
    # Heat rate: all three solvers within 10%.
    for a, b in (("PBF", "WCSPH"), ("DFSPH", "WCSPH"), ("DFSPH", "PBF")):
        assert _rel(results[a]["short"], results[b]["short"]) < 0.10, (
            f"dT({SHORT}): {a} {results[a]['short']:.6f} K vs "
            f"{b} {results[b]['short']:.6f} K"
        )
    # Longer run: DFSPH tracks WCSPH. PBF drifts low over time for a reason
    # outside this bead (see test_static_heat_cross_solver.py), so it is
    # printed but not compared.
    assert _rel(results["DFSPH"]["long"], wc["long"]) < 0.10, (
        f"dT({LONG}): DFSPH {results['DFSPH']['long']:.6f} K vs "
        f"WCSPH {wc['long']:.6f} K"
    )


def dfsph_floor_alpha(floor_mat):
    """DFSPH alpha of every WATER particle resting on a floor of floor_mat."""
    import dfsph_solver

    w = World(max_particles=MAX_PARTICLES)
    w.packed_info[:] = 0
    w._high_water = 0
    n_floor = w.spawn_cube(
        (-0.12, -0.04, -0.12), (0.12, 0.0, 0.12), floor_mat, spacing=SPACING
    )
    n_water = w.spawn_cube(
        (-0.06, 0.0, -0.06), (0.06, 0.12, 0.06), WATER, spacing=SPACING
    )
    n = n_floor + n_water
    sim = Simulation(w, dt=0.001, speed=1.0, fixed_dt=True, max_substeps=1)
    sim.set_solver_profile(PROFILES["DFSPH"])
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)
    # Two passes: the second reads the stored density (rest_density for the
    # floor) as density_in, which is what every later substep sees.
    for pass_idx in range(2):
        density_in = None if pass_idx == 0 else w.sorted_density[:n].copy()
        dfsph_solver.compute_density_alpha(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_mass[:n],
            density_in,
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            sim._cell_start,
            sim._cell_end,
            w.sorted_density,
            w.sorted_alpha_dfsph,
            w.sorted_shear_rate,
            w.sorted_dTdt,
            w.sorted_exposure_heat,
            w.sorted_exposure_corrode,
        )
    pos = w.sorted_position[:n, :3].get()
    mat = w.sorted_packed_info[:n].get() & 0xFF
    alpha = w.sorted_alpha_dfsph[:n].get()
    water = mat == WATER
    # Sort by position so the two scenes line up particle by particle.
    key = np.lexsort(np.round(pos[water] / SPACING).astype(np.int64).T)
    return pos[water][key], alpha[water][key]


def test_dfsph_wall_alpha_does_not_depend_on_wall_material():
    """Water next to STONE, WOOD and METAL floors gets the same DFSPH alpha.

    Before the fix the floor volume was m_j / rest_density_j, so the
    boundary gradient term was 2.6x too weak against STONE, 7.8x against
    METAL and 1.7x too strong against WOOD, relative to a WATER neighbour.
    """
    res = {mat: dfsph_floor_alpha(mat) for mat in (STONE, WOOD, METAL)}
    pos0, a0 = res[STONE]
    bottom = pos0[:, 1] < 0.02  # bottom WATER layer touches the floor
    assert bottom.sum() >= 36
    for mat in (WOOD, METAL):
        pos, a = res[mat]
        assert np.array_equal(pos, pos0)
        rel = np.abs(a - a0) / a0
        print(
            f"  {MATERIALS[mat].name} vs STONE: bottom alpha {a[bottom].mean():.4e} "
            f"vs {a0[bottom].mean():.4e}, max rel diff {rel.max():.2e}"
        )
        assert rel.max() < 1e-5, MATERIALS[mat].name


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-s"])
