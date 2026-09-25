"""Regression test for per-stage timing of PBF and DFSPH (bd-r5-review-n28.120).

_sim_step_timed used to fall back to the untimed _sim_step for any solver
other than WCSPH, so the timing panel had no PBF or DFSPH stages.  This calls
_sim_step_timed directly under each non-WCSPH profile and checks that the
stage dict carries that solver's labels with positive durations.
"""

import pytest

import bench_substep
from simulation import PBF_STAGES, DFSPH_STAGES, EPILOGUE_STAGES

EXPECTED = {
    "PBF": {"sort", "predict", "lambda0", "reactions", "spawn", "iterations",
            "finalize", "wake"},
    "DFSPH": {"sort", "density_alpha", "reactions", "spawn", "non_pressure",
              "divergence", "density_solve", "finalize", "wake"},
}
ALLOWED = {"PBF": set(PBF_STAGES), "DFSPH": set(DFSPH_STAGES)}


def _make_sim(solver: str, n: int = 20_000):
    bench_substep._apply_ptx_workaround()
    from world import World
    from simulation import Simulation
    from solver_profiles import PROFILES

    world = World(max_particles=int(n * 1.2) + 10_000)
    bench_substep.build_dam_break(world, n)
    sim = Simulation(
        world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=True,
        max_substeps=1, world_half_size=bench_substep.world_half_size_for(n),
    )
    sim.set_solver_profile(PROFILES[solver])
    sim.reset_spawn_damping()
    return sim, world._high_water


@pytest.mark.parametrize("solver", ["PBF", "DFSPH"])
def test_sim_step_timed_reports_solver_stages(solver):
    sim, num = _make_sim(solver)
    # Timings are read one substep late, so run a few substeps.
    for _ in range(4):
        sim._sort_skip_next = False
        sim._sim_step_timed(num)

    timings = sim._last_timings
    labels = set(timings)
    assert EXPECTED[solver] <= labels, f"missing {EXPECTED[solver] - labels}"
    assert labels <= ALLOWED[solver] | set(EPILOGUE_STAGES)
    assert "solver" not in labels
    for label in EXPECTED[solver]:
        assert timings[label] > 0.0, f"{label} = {timings[label]}"
    # The EMA used by the timing panel is filled for the same labels.
    assert EXPECTED[solver] <= set(sim._timing_ema)
