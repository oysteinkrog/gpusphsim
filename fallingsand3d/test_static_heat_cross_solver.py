"""A STONE wall next to LAVA heats at the same rate under all three solvers.

All three solvers store rest_density for a STATIC wall particle and divide
its heat sum by that same value (bd-r5-review-n28.116). Neighbours weight the
wall by m / density, so the heat pair term stays conservative. Before
bd-r5-review-n28.103 DFSPH divided by rest_density while WCSPH and PBF used
the SPH-summed density, so the wall heated about 2.5x slower under DFSPH.

Scene: a one-layer STONE wall touching a LAVA block, zero gravity, spawn
damping off, reactions off, equal fixed dt (1 ms) for every solver. After one
warm-up substep the temperatures are reset, then the test compares the mean
wall temperature rise:

- after 5 substeps, DFSPH within 10% of WCSPH, PBF within 12% of both
  (the heat rate itself);
- after 100 substeps, DFSPH within 10% of WCSPH.

Measured (K): 5 substeps WCSPH 17.7, PBF 15.9, DFSPH 17.3; 100 substeps
WCSPH 293.5, PBF 221.5, DFSPH 286.9.

PBF sits about 10% below WCSPH and drifts further low over time. That is a
PBF effect outside this test's scope, so the 100-substep check is DFSPH vs
WCSPH.
"""

import dataclasses

import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler

        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, "_get_arch_for_options_for_nvrtc"):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, "_cache"):
                fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402

from world import World, T_AMBIENT  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import LAVA, STONE  # noqa: E402

MAX_PARTICLES = 20_000
DT = 1.0 / 1000.0
SHORT = 5
LONG = 100
SPACING = 0.02
T_LAVA = 1500.0


def wall_heating(profile_name, monkeypatch):
    """Return the mean STONE temperature rise (K) after SHORT and LONG substeps."""
    w = World(max_particles=MAX_PARTICLES)
    w.packed_info[:] = 0
    w._high_water = 0
    # 10 x 10 x 10 LAVA at x < 0, one 10 x 10 STONE layer at x = 0.01.
    n = w.spawn_cube((-0.2, -0.1, -0.1), (0.0, 0.1, 0.1), LAVA, spacing=SPACING)
    n_lava = n
    n += w.spawn_cube((0.0, -0.1, -0.1), (SPACING, 0.1, 0.1), STONE, spacing=SPACING)
    n_wall = n - n_lava
    assert n_wall == 100

    profile = dataclasses.replace(
        PROFILES[profile_name], dt=DT, fixed_dt=True, max_substeps=1
    )
    s = Simulation(w, dt=DT, speed=1.0, fixed_dt=True, max_substeps=1)
    s.set_solver_profile(profile)
    s.set_gravity(0.0)
    s._spawn_substep = s._damping_duration
    monkeypatch.setattr(s, "_run_reactions_spawn", lambda *a, **k: None)

    def substep(n):
        n = s._substep_prologue(n)
        s._count_substep_sort(n)
        s._run_graph_body(n)
        s._substep_epilogue(n)
        return n

    def wall_rise(n):
        is_wall = (w.packed_info[:n] & 0xFF) == STONE
        count = cp.maximum(cp.count_nonzero(is_wall), 1)
        return cp.sum(cp.where(is_wall, w.temperature[:n] - T_AMBIENT, 0.0)) / count

    # One warm-up substep, then reset the temperatures. PBF and DFSPH read
    # the neighbour density from the previous substep, so their first heat
    # rate at the wall is about 6.5x the later one; the warm-up keeps that
    # out of the comparison.
    n = substep(n)
    mat = w.packed_info[:n] & 0xFF
    w.temperature[:n] = cp.where(mat == LAVA, T_LAVA, T_AMBIENT).astype(cp.float32)

    rises = {}
    for k in range(1, LONG + 1):
        n = substep(n)
        if k in (SHORT, LONG):
            rises[k] = wall_rise(n)  # stays on the GPU until the end

    is_wall = (w.packed_info[:n] & 0xFF) == STONE
    return {
        "short": float(rises[SHORT]),
        "long": float(rises[LONG]),
        "wall_n": int(cp.count_nonzero(is_wall)),
        "wall_spawned": n_wall,
        "finite": bool(cp.isfinite(w.temperature[:n]).all()),
    }


def _rel(a, b):
    return abs(a - b) / b


def test_stone_wall_heats_equally_across_solvers(monkeypatch):
    results = {}
    for name in ("WCSPH", "PBF", "DFSPH"):
        with monkeypatch.context() as m:
            results[name] = wall_heating(name, m)
        r = results[name]
        print(
            f"  {name}: wall n={r['wall_n']}/{r['wall_spawned']} "
            f"dT({SHORT})={r['short']:.3f} K dT({LONG})={r['long']:.3f} K"
        )
    for name, r in results.items():
        assert r["finite"], name
        assert r["wall_n"] == r["wall_spawned"], (
            f"{name}: wall particles changed material"
        )

    wc, pbf, df = results["WCSPH"], results["PBF"], results["DFSPH"]
    # The wall must actually heat, or the comparison proves nothing.
    assert wc["short"] > 5.0, f"WCSPH wall barely heated: {wc['short']:.3f} K"

    # Heat rate: DFSPH vs WCSPH within 10%, PBF against either within 12%.
    assert _rel(df["short"], wc["short"]) < 0.10, (
        f"DFSPH {df['short']:.3f} K vs WCSPH {wc['short']:.3f} K"
    )
    assert _rel(df["short"], pbf["short"]) < 0.12, (
        f"DFSPH {df['short']:.3f} K vs PBF {pbf['short']:.3f} K"
    )
    assert _rel(pbf["short"], wc["short"]) < 0.12, (
        f"PBF {pbf['short']:.3f} K vs WCSPH {wc['short']:.3f} K"
    )
    # Longer run: DFSPH still tracks WCSPH.
    assert _rel(df["long"], wc["long"]) < 0.10, (
        f"DFSPH {df['long']:.3f} K vs WCSPH {wc['long']:.3f} K after {LONG} substeps"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-s"])
