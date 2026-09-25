"""Regression test for the Kernel Timings panel stage list (bd-r5-review-n28.151).

The panel used a fixed list with "hash" and "reorder", which no stage
records (rows stuck at 0.00), and left out "foam", so the total was short.
After a solver switch it kept the old solver's timings as if they were live.

Checks, per solver: every label _sim_step_timed records (except "start") is
a panel row, every panel row was recorded, and the rows sum to the recorded
total.  Then checks that set_solver_profile clears all timing state.
"""

import pytest

import ui
from solver_profiles import PROFILES
from test_sim_step_timed import _make_sim


@pytest.mark.parametrize("solver", ["WCSPH", "PBF", "DFSPH"])
def test_panel_rows_match_recorded_stages(solver):
    sim, num = _make_sim(solver)
    # Timings are read one substep late, so run a few substeps.
    for _ in range(4):
        sim._sort_skip_next = False
        sim._sim_step_timed(num)

    recorded = set(sim._last_timings)
    assert recorded, "no timings recorded"
    assert "start" not in recorded
    assert "foam" in recorded
    # Every recorded label belongs to the active solver's stage list.
    assert recorded <= set(sim.timing_stages()), recorded - set(sim.timing_stages())

    rows = ui.timing_rows(sim)
    assert set(rows) == set(sim._timing_ema)
    assert set(rows) == recorded
    assert len(rows) == len(set(rows))
    for stale in ("hash", "reorder"):
        assert stale not in rows
    # Rows keep the order the stages run in.
    order = sim.timing_stages()
    assert rows == sorted(rows, key=order.index)
    total = sum(sim._timing_ema[name] for name in rows)
    assert total == pytest.approx(sum(sim._timing_ema.values()))


def test_set_solver_profile_clears_timings():
    sim, num = _make_sim("WCSPH")
    for _ in range(3):
        sim._sort_skip_next = False
        sim._sim_step_timed(num)
    assert sim._timing_ema and sim._last_timings
    assert sim._pending_timing_events

    sim.set_solver_profile(PROFILES["PBF"])
    assert sim._timing_ema == {}
    assert sim._last_timings == {}
    assert sim._pending_timing_events == []
    assert sim._pending_timing_labels == []
    assert ui.timing_rows(sim) == []

    # The first PBF timings carry only PBF and epilogue labels.
    for _ in range(3):
        sim._sort_skip_next = False
        sim._sim_step_timed(num)
    assert set(sim._timing_ema) <= set(sim.timing_stages())
    assert "step1" not in sim._timing_ema
