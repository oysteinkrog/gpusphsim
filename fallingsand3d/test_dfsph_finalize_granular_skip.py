"""bd-r5-review-n28.188: DFSPH Finalize skips the XSPH work for GRANULAR.

GRANULAR gets no XSPH in K_DFSPH_Finalize, so its neighbour loop only
gathers the STATIC push. The kernel now skips non-STATIC neighbours before
the position load, skips the mass, velocity and density loads, and skips the
loop when the push takes the separate scan.

The test runs sand, water and smoke falling onto a STONE slab under DFSPH
and replays every Finalize call with the kernel as it was before this bead
(rebuilt from the current source by undoing the three text changes). All
outputs must match exactly for every behaviour.

Run this file directly to time both kernels on a large sand block:
    python test_dfsph_finalize_granular_skip.py [n_sand]
"""

import inspect
import os
import sys

import cupy as cp
import numpy as np
import pytest

import dfsph_solver
from materials import SAND, SMOKE, STONE, WATER, GRANULAR, FLUID, GAS
from simulation import Simulation
from solver_profiles import PROFILES
from test_static_push_fold import KERNEL_DIR, _copy_constants
from world import World

STEPS = 40
OUT_NAMES = ("position_out", "velocity_out", "packed_info_out",
             "sleep_counter_out", "temperature_out", "kappa_out",
             "kappa_v_out", "health_out", "lifetime_out", "mass_out")

# Text changes made by this bead; each is undone to rebuild the old kernel.
_UNDO = (
    ("    if (xsph_self || (behavior == GRANULAR && spush_in_loop)) {\n",
     "    if (behavior == FLUID || behavior == GRANULAR) {\n"),
    ("                        if (!xsph_self && bj != STATIC) continue;\n", ""),
    ("                        if (!xsph_self) continue;\n", ""),
)


def _old_kernel():
    with open(os.path.join(KERNEL_DIR, "dfsph_solver.cu"), encoding="utf-8") as f:
        src = f.read()
    for new, old in _UNDO:
        assert src.count(new) == 1, f"bead change not found: {new.strip()}"
        src = src.replace(new, old)
    return cp.RawModule(code=src, options=("--std=c++11", "--use_fast_math",
                                            f"-I{KERNEL_DIR}"))


def _scene(world, sim, sand_only_n=0):
    base = -sim.world_half_size
    top = base + 0.08
    n = world.spawn_cube((-0.3, base, -0.2), (0.3, top, 0.2), STONE, spacing=0.02)
    if sand_only_n:
        side = int(round(sand_only_n ** (1.0 / 3.0)))
        ext = side * 0.02
        n += world.spawn_cube((-ext / 2, top + 0.005, -ext / 2),
                              (ext / 2, top + 0.005 + ext, ext / 2),
                              SAND, spacing=0.02)
        return n
    n += world.spawn_cube((-0.26, top + 0.005, -0.1), (-0.02, top + 0.2, 0.1),
                          WATER, spacing=0.02, velocity=(0.0, -1.5, 0.0))
    n += world.spawn_cube((0.02, top + 0.005, -0.1), (0.26, top + 0.14, 0.1),
                          SAND, spacing=0.02, velocity=(0.0, -1.5, 0.0))
    n += world.spawn_cube((-0.1, top + 0.005, 0.12), (0.1, top + 0.05, 0.18),
                          SMOKE, spacing=0.02, velocity=(0.0, -1.0, 0.0), lifetime=0)
    return n


def _run(monkeypatch, steps=STEPS, sand_only_n=0, max_particles=16384, timing=False):
    """Run the scene. Return per-call (new outputs, old outputs, behaviour),
    and the summed kernel times (new_ms, old_ms) when timing is set."""
    world = World(max_particles=max_particles)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=False, max_substeps=20)
    sim.set_solver_profile(PROFILES["DFSPH"])
    monkeypatch.setattr(
        sim, "_capture_graph",
        lambda n, gather_only=False: (sim._run_graph_body(n), None)[1])
    n = _scene(world, sim, sand_only_n)
    sim.notify_scene_changed()
    sim._spawn_substep = sim._damping_duration

    real_mod = dfsph_solver._get_module()
    old_mod = _old_kernel()
    orig = dfsph_solver.finalize
    sig = inspect.signature(orig)
    records = []
    times = [0.0, 0.0]

    def launch(module, args):
        _copy_constants(real_mod, module)
        dfsph_solver._module = module
        try:
            if not timing:
                orig(**args)
                return 0.0
            orig(**args)  # warm-up
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for _ in range(5):
                orig(**args)
            e1.record()
            e1.synchronize()
            return cp.cuda.get_elapsed_time(e0, e1) / 5.0
        finally:
            dfsph_solver._module = real_mod

    def replay(module, bound):
        args = dict(bound.arguments)
        for k in OUT_NAMES:
            if args.get(k) is not None:
                args[k] = args[k].copy()
        args["max_displacement"] = None
        args["cell_wake_flags"] = None
        t = launch(module, args)
        return tuple(args[k] for k in OUT_NAMES if args.get(k) is not None), t

    def spy(*a, **k):
        orig(*a, **k)
        bound = sig.bind(*a, **k)
        bound.apply_defaults()
        new, t_new = replay(real_mod, bound)
        old, t_old = replay(old_mod, bound)
        times[0] += t_new
        times[1] += t_old
        beh = (bound.arguments["packed_info_out"] >> 8) & 0x3
        if not timing:
            records.append((new, old, beh))

    monkeypatch.setattr(dfsph_solver, "finalize", spy)
    for _ in range(steps):
        sim._sim_step(n)
    cp.cuda.Device().synchronize()
    monkeypatch.undo()

    out = []
    for new, old, beh in records:
        out.append(([cp.asnumpy(x[:n]) for x in new],
                    [cp.asnumpy(x[:n]) for x in old],
                    cp.asnumpy(beh[:n])))
    return out, n, times


@pytest.fixture(scope="module")
def records():
    mp = pytest.MonkeyPatch()
    try:
        return _run(mp)[0]
    finally:
        mp.undo()


@pytest.mark.parametrize("behaviour,name", [(GRANULAR, "GRANULAR"), (FLUID, "FLUID"),
                                            (GAS, "GAS")])
def test_finalize_matches_old_kernel(records, behaviour, name):
    assert len(records) == STEPS
    seen = 0
    for new, old, beh in records:
        m = beh == behaviour
        seen += int(m.sum())
        for a, b in zip(new, old):
            assert np.array_equal(a[m], b[m]), f"{name}: output differs from the old kernel"
    assert seen > 0, f"no {name} particles in the scene"


def test_granular_moves(records):
    """Sand falls and lands, so the replay covers moving GRANULAR particles."""
    first_new, _, beh = records[0]
    last_new, _, _ = records[-1]
    m = beh == GRANULAR
    dy = np.abs(last_new[0][m, 1] - first_new[0][m, 1]).max()
    assert dy > 1e-3


if __name__ == "__main__":
    n_sand = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    mp = pytest.MonkeyPatch()
    try:
        _, n, (t_new, t_old) = _run(mp, steps=10, sand_only_n=n_sand,
                                    max_particles=n_sand + 20000, timing=True)
    finally:
        mp.undo()
    print(f"particles={n}  K_DFSPH_Finalize summed over 10 substeps: "
          f"old={t_old:.3f} ms  new={t_new:.3f} ms  "
          f"per substep old={t_old / 10:.3f} new={t_new / 10:.3f} "
          f"({100.0 * (t_new - t_old) / t_old:+.1f}%)")
