"""bd-r5-review-n28.83 measurement: DFSPH divergence warm start, total vs last increment.

36K WATER block dropped to the floor, Stable DFSPH profile, 2 s. For each
config logs, every 0.25 s: v_max, KE, compression error, and the mean
positive div_v left after the warm-start pass (read from the first
compute_kappa_v call of that substep). 'old' mode emulates the previous
code: the stored warm start is the last increment only.
"""
import dataclasses
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

import dfsph_solver  # noqa: E402
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import WATER, MATERIALS  # noqa: E402

_orig = dfsph_solver.compute_kappa_v
state = {"old": False, "capture": False, "first": True, "div": None, "profile": None}


def patched(velocity, density, mass, alpha, packed_info, position, cell_start, cell_end,
            kappa_v_out, normal_in=None, kappa_v_total=None):
    _orig(velocity, density, mass, alpha, packed_info, position, cell_start, cell_end,
          kappa_v_out, normal_in=normal_in, kappa_v_total=kappa_v_total)
    n = velocity.shape[0]
    if state["capture"] and state["first"]:
        p = state["profile"]
        a = alpha[:n].get()
        kv = kappa_v_out[:n].get()
        act = a > 0
        state["div"] = float(np.mean(kv[act] * p.dt / (a[act] * p.dfsph_omega)))
    state["first"] = False
    if state["old"] and kappa_v_total is not None:
        kappa_v_total[:n] = kappa_v_out[:n]


dfsph_solver.compute_kappa_v = patched

rho0 = float(MATERIALS[WATER].rest_density)
configs = [("old", 0.5), ("new", 0.0), ("new", 0.25), ("new", 0.35), ("new", 0.5)]
if len(sys.argv) > 1:
    configs = [c for c in configs if f"{c[0]}{c[1]}" in sys.argv[1:]]
w = World(max_particles=60_000)
for mode, ws in configs:
    profile = dataclasses.replace(PROFILES["DFSPH"], dfsph_div_warm_start=ws)
    state.update(old=(mode == "old"), profile=profile)
    w.packed_info[:] = 0
    w._high_water = 0
    n = w.spawn_cube((-0.3, -0.9, -0.3), (0.3, -0.1, 0.3), WATER, spacing=0.02)
    s = Simulation(w, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    s.set_solver_profile(profile)
    s._spawn_substep = s._damping_duration
    steps = int(round(2.0 / profile.dt))
    every = int(round(0.25 / profile.dt))
    print(f"\n== {mode} warm={ws}  (div_iters={profile.dfsph_div_iters}, n={n})")
    vmax_hist = []
    divs = []
    t0 = time.perf_counter()
    for step in range(steps):
        log = (step + 1) % every == 0
        state.update(capture=log, first=True)
        if log:  # run this substep directly (not a graph replay) so the hook reads back
            s._graph_full_sort_n = -1
            s._graph_gather_only_n = -1
        s._sim_step(n)
        if log:
            cp.cuda.Device().synchronize()
            v = w.velocity[:n, :3].get()
            m = w.mass[:n].get()
            rho = w.sorted_density[:n].get()
            pi = w.sorted_packed_info[:n].get()
            fluid = (((pi >> 8) & 3) == 0) & ((pi & 0xFF) != 0)
            comp = float(np.mean(np.maximum(rho[fluid] / rho0 - 1.0, 0.0)))
            vmax = float(np.sqrt((v * v).sum(1)).max())
            ke = float(0.5 * np.sum(m * (v * v).sum(1)))
            vmax_hist.append(vmax)
            divs.append(state["div"])
            print(f"  t={(step + 1) * profile.dt:.2f}  v_max={vmax:.3f}  KE={ke:.2f}  "
                  f"comp={comp * 100:.2f}%  div_after_warm={state['div']:.4f}  "
                  f"nan={not np.isfinite(v).all()}")
    wall = time.perf_counter() - t0
    late = vmax_hist[len(vmax_hist) // 2:]
    print(f"  RESULT {mode} warm={ws}: mean div_after_warm(t>=1s)={np.mean(divs[len(divs)//2:]):.4f}  "
          f"v_max late mean={np.mean(late):.3f} spread={max(late) - min(late):.3f}  wall={wall:.1f}s")
