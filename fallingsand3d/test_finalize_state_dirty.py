"""bd-r5-review-n28.218: PBF and DFSPH Finalize scatter only dirty reaction state.

K_PBF_Finalize and K_DFSPH_Finalize scatter health, lifetime and mass back to
the unsorted arrays only for particles K_Reactions or K_SpawnGas marked with
the state-dirty bit (packed_info bit 21, physics/kernels/state_dirty.cuh), and
clear the bit. This test rebuilds each solver module with the old
unconditional scatter and checks that 300 substeps of Acid Rain and Volcano
end with bitwise equal health, lifetime and mass (and positions) under both
kernels.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np
import pytest

from world import World

STATE_DIRTY_BIT = 0x200000
SUBSTEPS = 300

GATE = "if (HAS_STATE_DIRTY(pi)) {\n        pi = CLEAR_STATE_DIRTY(pi);\n"


def _old_module(solver_mod, cu_name: str):
    """The solver module with the old unconditional Finalize scatter."""
    kernel_dir = os.path.join(
        os.path.dirname(os.path.abspath(solver_mod.__file__)), "physics", "kernels"
    )
    with open(os.path.join(kernel_dir, cu_name), encoding="utf-8") as f:
        source = f.read()
    assert source.count(GATE) == 1, (
        f"{cu_name}: Finalize dirty gate changed; update this test"
    )
    solver_mod._ensure_ptx_if_needed()
    return cupy.RawModule(
        code=source.replace(GATE, "{\n"),
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )


def _run(preset: str, solver: str, module) -> dict:
    import dfsph_solver
    import pbf_solver
    from presets import PRESETS
    from simulation import Simulation
    from solver_profiles import PROFILES

    solver_mod = pbf_solver if solver == "PBF" else dfsph_solver
    saved = solver_mod._module
    solver_mod._module = module if module is not None else None
    world = World(max_particles=400_000)
    sim = Simulation(
        world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=True, max_substeps=20
    )
    try:
        sim.deterministic_sort = True
        sim.set_solver_profile(PROFILES[solver])
        PRESETS[preset](world)
        sim.notify_scene_changed()
        n = world._high_water
        health0 = cupy.asnumpy(world.health[:n]).copy()
        lifetime0 = cupy.asnumpy(world.lifetime[:n]).copy()
        for _ in range(SUBSTEPS):
            sim._sim_step(n)
        cupy.cuda.Device().synchronize()
        return {
            "n": n,
            "health0": health0,
            "lifetime0": lifetime0,
            "health": cupy.asnumpy(world.health[:n]).copy(),
            "lifetime": cupy.asnumpy(world.lifetime[:n]).copy(),
            "mass": cupy.asnumpy(world.mass[:n]).copy(),
            "position": cupy.asnumpy(world.position[:n]).copy(),
            "packed_info": cupy.asnumpy(world.packed_info[:n]).copy(),
        }
    finally:
        sim.close()
        solver_mod._module = saved


def _diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("solver", ["PBF", "DFSPH"])
@pytest.mark.parametrize("preset", ["Acid Rain", "Volcano"])
def test_finalize_dirty_scatter_matches_unconditional(preset: str, solver: str) -> None:
    import dfsph_solver
    import pbf_solver

    solver_mod = pbf_solver if solver == "PBF" else dfsph_solver
    cu_name = "pbf_solver.cu" if solver == "PBF" else "dfsph_solver.cu"
    new = _run(preset, solver, None)
    old = _run(preset, solver, _old_module(solver_mod, cu_name))

    assert new["n"] == old["n"]
    # The reaction state must actually change, or the check proves nothing.
    changed = _diff(new["health"], new["health0"]) + _diff(
        new["lifetime"], new["lifetime0"]
    )
    assert changed > 0, f"{preset}/{solver}: no reaction changed health or lifetime"

    for key in ("health", "lifetime", "mass", "position"):
        d = _diff(new[key], old[key])
        assert d == 0, f"{preset}/{solver}: {d} {key} words differ from the old kernel"
    mask = np.uint32(~STATE_DIRTY_BIT & 0xFFFFFFFF)
    d = int(
        np.count_nonzero((new["packed_info"] & mask) != (old["packed_info"] & mask))
    )
    assert d == 0, f"{preset}/{solver}: {d} packed_info words differ outside bit 21"
    # Finalize writes every packed_info back with the bit cleared.
    left = int(np.count_nonzero(new["packed_info"] & STATE_DIRTY_BIT))
    assert left == 0, f"{preset}/{solver}: {left} unsorted particles keep the dirty bit"
