"""Per-stage substep benchmark for all three solvers, with JSON output.

Builds a fixed dam-break scene, warms up, then times substeps through
Simulation._sim_step_timed, which records a CUDA event after each stage of
the active solver body.  Writes per-stage median and p90 (ms) to
bench_results/<git-sha>-<solver>-<n>.json, together with the cupy and driver
versions and a checksum of the final positions, so a perf change that also
changes physics shows up as a different checksum.

Usage:
    python bench_substep.py --solver WCSPH --n 1000000
    python bench_substep.py --solver PBF --n 250000 --grid-reuse
    python bench_substep.py --compare a.json b.json

By default every substep does a full counting sort, so runs are comparable.
--grid-reuse lets the sort-skip logic run the way step_frame drives it: the
decision is made once per frame of --substeps-per-frame substeps from the
async max_displacement readback.  Gather substeps are reported as 'gather'
instead of 'sort'.

Timing runs outside CUDA graph capture (events cannot be captured), so the
numbers are kernel time for the direct-launch path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, "bench_results")
NOISE_BAND = 0.02  # 2 %: stage deltas inside this band count as noise


def _apply_ptx_workaround():
    """Force CuPy to emit PTX instead of cubin for forward-compat with newer GPUs."""
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


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        if not sha:
            return "nogit"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "."],
            cwd=HERE, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def world_half_size_for(n: int) -> float:
    """World half-size that fits an n-particle dam break (same as benchmark.py)."""
    if n >= 1_000_000:
        return 3.0
    if n >= 500_000:
        return 2.0
    return 1.0


def build_dam_break(world, num_particles: int, spacing: float = 0.02) -> int:
    """Water block (1 x 2 x 1 aspect) in the left corner of the domain.

    spawn_cube fills a regular lattice, so the scene is identical run to run.
    """
    from materials import WATER

    w = spacing * (num_particles / 2.0) ** (1.0 / 3.0)
    h = 2.0 * w
    x0, x1 = -0.9, -0.9 + w
    y0, y1 = -0.9, -0.9 + h
    z0, z1 = -w / 2.0, w / 2.0
    return world.spawn_cube(
        min_corner=(x0, y0, z0), max_corner=(x1, y1, z1),
        material_id=WATER, spacing=spacing,
    )


def _percentile(sorted_vals, q: float) -> float:
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def run_bench(
    solver: str = "WCSPH",
    n: int = 1_000_000,
    warmup: int = 50,
    steps: int = 300,
    grid_reuse: bool = False,
    substeps_per_frame: int = 4,
    out_dir: str | None = DEFAULT_OUT_DIR,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run the benchmark and return the result dict (also written as JSON)."""
    _apply_ptx_workaround()
    import cupy
    import numpy as np
    from world import World
    from simulation import Simulation
    from solver_profiles import PROFILES

    if solver not in PROFILES:
        raise ValueError(f"unknown solver profile {solver!r}; one of {sorted(PROFILES)}")
    np.random.seed(seed)
    cupy.random.seed(seed)

    world = World(max_particles=int(n * 1.2) + 10_000)
    actual = build_dam_break(world, n)
    sim = Simulation(
        world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=True,
        max_substeps=1, world_half_size=world_half_size_for(n),
    )
    sim.set_solver_profile(PROFILES[solver])
    sim.reset_spawn_damping()
    sim._spawn_substep = sim._damping_duration  # no damping ramp: constant kernels
    num = world._high_water
    if verbose:
        print(f"  {solver}: {actual:,} particles (target {n:,}), "
              f"warmup {warmup}, timed {steps}, grid reuse {'on' if grid_reuse else 'off'}")

    substep_in_frame = 0

    def end_substep():
        nonlocal substep_in_frame
        substep_in_frame += 1
        if substep_in_frame < substeps_per_frame:
            if not grid_reuse:
                sim._sort_skip_next = False
            return
        substep_in_frame = 0
        sim._frame_counter += 1
        if grid_reuse:
            sim._update_sort_skip_decision()
        else:
            sim._sort_skip_next = False
            sim._sort_skip_consecutive = 0

    if not grid_reuse:
        sim._sort_skip_next = False
    for _ in range(warmup):
        sim._sim_step_timed(num)
        end_substep()
    cupy.cuda.Device().synchronize()

    recorded = []  # (labels, events, was_gather) per timed substep
    t0 = time.perf_counter()
    for _ in range(steps):
        gather = bool(sim._sort_skip_next) and num == sim._last_full_sort_n
        sim._sim_step_timed(num)
        recorded.append((list(sim._pending_timing_labels),
                         list(sim._pending_timing_events), gather))
        end_substep()
    cupy.cuda.Device().synchronize()
    wall_s = time.perf_counter() - t0

    per_stage: dict[str, list[float]] = {}
    totals = []
    stage_order: list[str] = []
    for labels, events, gather in recorded:
        for i in range(1, len(events)):
            label = labels[i]
            if label == "sort" and gather:
                label = "gather"
            ms = cupy.cuda.get_elapsed_time(events[i - 1], events[i])
            per_stage.setdefault(label, []).append(ms)
            if label not in stage_order:
                stage_order.append(label)
        totals.append(cupy.cuda.get_elapsed_time(events[0], events[-1]))

    stages = {}
    for label in stage_order:
        vals = sorted(per_stage[label])
        stages[label] = {
            "median_ms": _percentile(vals, 0.5),
            "p90_ms": _percentile(vals, 0.9),
            "mean_ms": sum(vals) / len(vals),
            "count": len(vals),
        }
    totals_sorted = sorted(totals)

    # Checksum: sum of final positions (float64) so a physics change is visible.
    pos = cupy.asnumpy(world.position[:num, :3]).astype(np.float64)
    checksum = float(pos.sum())
    finite = bool(np.isfinite(pos).all())

    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
    sha = _git_sha()
    result = {
        "git_sha": sha,
        "solver": solver,
        "n_target": n,
        "n": int(num),
        "warmup": warmup,
        "steps": steps,
        "grid_reuse": grid_reuse,
        "substeps_per_frame": substeps_per_frame,
        "dt": float(sim.dt),
        "seed": seed,
        "stages": stages,
        "total": {
            "median_ms": _percentile(totals_sorted, 0.5),
            "p90_ms": _percentile(totals_sorted, 0.9),
            "mean_ms": sum(totals) / len(totals) if totals else 0.0,
        },
        "sort_skip_rate": (stages.get("gather", {}).get("count", 0) / steps) if steps else 0.0,
        "wall_s": wall_s,
        "checksum_position_sum": checksum,
        "positions_finite": finite,
        "gpu": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "cupy_version": cupy.__version__,
        "cuda_runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cupy.cuda.runtime.driverGetVersion(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        suffix = "-reuse" if grid_reuse else ""
        path = os.path.join(out_dir, f"{sha}-{solver}-{n}{suffix}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        result["json_path"] = path
    if verbose:
        print_result(result)
    return result


def print_result(r: dict) -> None:
    print(f"\n  {r['solver']} at {r['n']:,} particles, sha {r['git_sha']}, "
          f"{r['steps']} substeps, grid reuse {'on' if r['grid_reuse'] else 'off'}")
    print(f"  {'stage':<16}{'median ms':>11}{'p90 ms':>10}{'count':>7}")
    for label, s in r["stages"].items():
        print(f"  {label:<16}{s['median_ms']:>11.3f}{s['p90_ms']:>10.3f}{s['count']:>7}")
    t = r["total"]
    print(f"  {'total':<16}{t['median_ms']:>11.3f}{t['p90_ms']:>10.3f}")
    if r["grid_reuse"]:
        print(f"  sort skip rate: {r['sort_skip_rate']:.0%}")
    print(f"  checksum (sum of positions): {r['checksum_position_sum']:.6f}")
    if "json_path" in r:
        print(f"  wrote {r['json_path']}")


def compare(a: dict, b: dict, noise: float = NOISE_BAND) -> list[tuple[str, float, float, float, bool]]:
    """Per-stage median deltas from a to b.

    Returns (stage, a_ms, b_ms, relative_delta, within_noise) per stage,
    followed by a 'total' row.  A stage present in only one run counts as
    outside the band.
    """
    rows = []
    labels = list(a["stages"]) + [k for k in b["stages"] if k not in a["stages"]]
    for label in labels:
        am = a["stages"].get(label, {}).get("median_ms")
        bm = b["stages"].get(label, {}).get("median_ms")
        if am is None or bm is None:
            rows.append((label, am or 0.0, bm or 0.0, float("inf"), False))
            continue
        rel = (bm - am) / am if am > 0 else 0.0
        rows.append((label, am, bm, rel, abs(rel) <= noise))
    at, bt = a["total"]["median_ms"], b["total"]["median_ms"]
    rel = (bt - at) / at if at > 0 else 0.0
    rows.append(("total", at, bt, rel, abs(rel) <= noise))
    return rows


def print_compare(a: dict, b: dict, noise: float = NOISE_BAND) -> None:
    print(f"  A: {a['git_sha']} {a['solver']} n={a['n']:,}")
    print(f"  B: {b['git_sha']} {b['solver']} n={b['n']:,}")
    if a["solver"] != b["solver"] or a["n"] != b["n"]:
        print("  warning: runs differ in solver or particle count")
    print(f"  {'stage':<16}{'A ms':>10}{'B ms':>10}{'delta':>9}  noise band {noise:.0%}")
    for label, am, bm, rel, ok in compare(a, b, noise):
        d = "   n/a" if rel == float("inf") else f"{rel:+8.1%}"
        print(f"  {label:<16}{am:>10.3f}{bm:>10.3f}{d:>9}  {'within' if ok else 'OUTSIDE'}")
    if a.get("checksum_position_sum") != b.get("checksum_position_sum"):
        print(f"  checksum differs: {a.get('checksum_position_sum')} vs "
              f"{b.get('checksum_position_sum')} (physics changed or run settings differ)")
    else:
        print("  checksum identical")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Per-stage SPH substep benchmark (JSON output)")
    p.add_argument("--solver", default="WCSPH",
                   help="solver profile name from solver_profiles.PROFILES (default WCSPH)")
    p.add_argument("--n", type=int, default=1_000_000, help="particle count (default 1000000)")
    p.add_argument("--warmup", type=int, default=50, help="warmup substeps (default 50)")
    p.add_argument("--steps", type=int, default=300, help="timed substeps (default 300)")
    p.add_argument("--grid-reuse", action="store_true",
                   help="let the sort-skip logic run instead of a full sort every substep")
    p.add_argument("--substeps-per-frame", type=int, default=4,
                   help="substeps per frame for the grid-reuse decision (default 4)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="JSON output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"),
                   help="print per-stage deltas between two result files")
    p.add_argument("--noise", type=float, default=NOISE_BAND,
                   help="noise band for --compare as a fraction (default 0.02)")
    args = p.parse_args(argv)

    if args.compare:
        with open(args.compare[0]) as f:
            a = json.load(f)
        with open(args.compare[1]) as f:
            b = json.load(f)
        print_compare(a, b, args.noise)
        return 0

    run_bench(
        solver=args.solver, n=args.n, warmup=args.warmup, steps=args.steps,
        grid_reuse=args.grid_reuse, substeps_per_frame=args.substeps_per_frame,
        out_dir=args.out_dir, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
