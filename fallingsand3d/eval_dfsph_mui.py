"""Sand Castle pile spread and mu(I) eta by depth, WCSPH vs DFSPH.

Eval for bd-r5-review-n28.92. Loads the Sand Castle preset, runs a fixed
simulated time per solver, then reads back once:
  - sand pile base radius: 95th percentile of horizontal distance of the
    sand particles in the bottom 0.1 m of the pile
  - sand pile height: 99th percentile of sand y minus the floor
  - mean mu(I) eta per depth band (GRANULAR particles), from the solver's
    confining pressure and stored shear rate, same formula as the kernels

Usage: python eval_dfsph_mui.py [sim_seconds] [solver ...] [--old-eta]
  --old-eta  report DFSPH eta with the pre-fix inputs (Tait EOS, 1 Pa floor)
  EVAL_WCSPH_DT=<s>  fixed WCSPH dt (default: the profile dt)
"""

import os
import sys
import time

import numpy as np


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
from solver_profiles import PROFILES, SolverType  # noqa: E402
from materials import SAND, MATERIALS  # noqa: E402
import presets  # noqa: E402
import step2  # noqa: E402

FLOOR_Y = -0.9


def _eta(gamma_dot, p_eff, rho):
    """Host copy of compute_muI_eta (sph_shared.cuh) with step2 defaults."""
    spacing = float(step2.DEFAULT_PARTICLE_SPACING)
    mu_s = float(step2.DEFAULT_MU_S)
    mu_2 = float(step2.DEFAULT_MU_2)
    i0 = float(step2.DEFAULT_I0)
    mu_max = float(step2.DEFAULT_MU_MAX)
    mu0 = float(step2.DEFAULT_MU0)
    i_num = gamma_dot * spacing / np.sqrt(p_eff / rho)
    mu_i = mu_s + (mu_2 - mu_s) / (1.0 + i0 / np.maximum(i_num, 1e-8))
    return np.minimum(mu_max, mu0 + mu_i * p_eff / (gamma_dot + 1e-6))


def _shape(pos, info):
    """(base radius, height) of the sand pile."""
    sand = (info & 0xFF) == int(SAND)
    ps = pos[sand, :3]
    r_xz = np.hypot(ps[:, 0], ps[:, 2])
    y = ps[:, 1]
    y_bot = np.percentile(y, 1.0)
    base = y < y_bot + 0.1
    return float(np.percentile(r_xz[base], 95.0)), float(np.percentile(y, 99.0) - y_bot)


def run(solver: str, sim_seconds: float, old_eta: bool = False):
    profile = PROFILES[solver]
    world = World(max_particles=400_000)
    world.packed_info[:] = 0
    world._high_water = 0
    dt = profile.dt
    if profile.solver_type == SolverType.WCSPH and "EVAL_WCSPH_DT" in os.environ:
        dt = float(os.environ["EVAL_WCSPH_DT"])
    sim = Simulation(world, dt=dt, speed=1.0, fixed_dt=True, max_substeps=1)
    sim.set_solver_profile(profile)
    sim.fixed_dt_value = dt
    presets.load_sand_castle(world)
    n = world._high_water
    r0, h0 = _shape(cp.asnumpy(world.position[:n]), cp.asnumpy(world.packed_info[:n]))
    print(f"{solver}: dt={dt:.5f} initial base_r={r0:.3f} m height={h0:.3f} m")
    steps = max(1, int(round(sim_seconds / sim.fixed_dt_value)))
    t0 = time.perf_counter()
    for _ in range(steps):
        sim._sim_step(n)
    cp.cuda.Device().synchronize()
    wall = time.perf_counter() - t0

    # One debug readback (sorted buffers are from the last substep).
    pos = cp.asnumpy(world.sorted_position[:n])
    info = cp.asnumpy(world.sorted_packed_info[:n])
    rho = cp.asnumpy(world.sorted_density[:n])
    shear = cp.asnumpy(world.sorted_shear_rate[:n])
    kappa = cp.asnumpy(world.sorted_kappa[:n])
    g = abs(sim.gravity_y)

    sand = (info & 0xFF) == int(SAND)
    if not np.all(np.isfinite(pos[sand, :3])):
        return {"solver": solver, "error": "NaN/Inf"}
    ps = pos[sand, :3]
    r_xz = np.hypot(ps[:, 0], ps[:, 2])
    y = ps[:, 1]
    y_bot = np.percentile(y, 1.0)
    base = y < y_bot + 0.1
    base_r = float(np.percentile(r_xz[base], 95.0))
    height = float(np.percentile(y, 99.0) - y_bot)

    rho0 = float(MATERIALS[SAND].rest_density)
    p_floor = rho0 * g * float(step2.DEFAULT_PARTICLE_SPACING)
    rs = np.maximum(rho[sand], 1.0)
    if profile.solver_type == SolverType.DFSPH and not old_eta:
        p_solver = kappa[sand] * rs
    else:
        # WCSPH: Tait with the material's k and gamma, as step2 does.
        m = MATERIALS[SAND]
        p_solver = np.maximum(m.eos_stiffness * (np.power(np.minimum(rs / rho0, 10.0), m.eos_gamma) - 1.0), 0.0)
    if old_eta and profile.solver_type == SolverType.DFSPH:
        p_floor = 1.0  # pre-fix DFSPH floor (Tait with 1 Pa)
    p_eff = np.maximum(p_solver, p_floor)
    eta = _eta(shear[sand], p_eff, rs)
    top = np.percentile(y, 99.0)
    bands = []
    edges = np.linspace(y_bot, top, 6)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (y >= lo) & (y < hi)
        if sel.sum() == 0:
            continue
        depth = float(top - 0.5 * (lo + hi))
        bands.append((depth, int(sel.sum()), float(p_eff[sel].mean()),
                      float(eta[sel].mean()),
                      float(np.mean(eta[sel] >= float(step2.DEFAULT_MU_MAX) - 1e-3))))
    return {"solver": solver, "steps": steps, "wall_s": wall, "base_r": base_r,
            "height": height, "g": g, "bands": bands,
            "dens_err": float(np.mean(np.abs(rs / rho0 - 1.0)))}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    old_eta = "--old-eta" in sys.argv
    sim_seconds = float(args[0]) if args else 5.0
    solvers = args[1:] or ["WCSPH", "DFSPH"]
    results = []
    for s in solvers:
        r = run(s, sim_seconds, old_eta)
        results.append(r)
        if "error" in r:
            print(f"{s}: {r['error']}")
            continue
        print(f"{s}: t={sim_seconds}s steps={r['steps']} wall={r['wall_s']:.1f}s "
              f"base_r={r['base_r']:.3f} m height={r['height']:.3f} m "
              f"sand |rho/rho0-1|={r['dens_err']:.3f} g={r['g']:.2f}")
        print("  depth_m   count   p_eff_Pa   mean_eta   frac_at_mu_max")
        for depth, cnt, p, e, fmax in r["bands"]:
            print(f"  {depth:7.3f} {cnt:7d} {p:10.1f} {e:10.2f} {fmax:10.3f}")
    ok = [r for r in results if "error" not in r]
    ref = next((r for r in ok if r["solver"].startswith("WCSPH")), None)
    if ref is not None:
        for r in ok:
            if r is ref:
                continue
            d = (r["base_r"] - ref["base_r"]) / ref["base_r"] * 100.0
            print(f"{r['solver']} base radius vs {ref['solver']}: {d:+.1f}%")


if __name__ == "__main__":
    main()
