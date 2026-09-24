"""Smoke test for bench_substep.py (bd-r5-review-n28.8).

Runs a few substeps at 20K for each solver and checks that the JSON is
written and that each solver reports its own stage names, not a single
'solver' stage or the WCSPH stages.
"""

import json

import pytest

import bench_substep
from simulation import WCSPH_STAGES, PBF_STAGES, DFSPH_STAGES, EPILOGUE_STAGES

EXPECTED = {
    "WCSPH": {"sort", "step1", "reactions", "spawn", "step2", "integrate", "wake"},
    "PBF": {"sort", "predict", "lambda0", "reactions", "spawn", "iterations",
            "finalize", "wake"},
    "DFSPH": {"sort", "density_alpha", "reactions", "spawn", "non_pressure",
              "divergence", "density_solve", "finalize", "wake"},
}
ALLOWED = {
    "WCSPH": set(WCSPH_STAGES),
    "PBF": set(PBF_STAGES),
    "DFSPH": set(DFSPH_STAGES),
}


@pytest.mark.parametrize("solver", ["WCSPH", "PBF", "DFSPH"])
def test_bench_writes_json_with_solver_stages(solver, tmp_path):
    r = bench_substep.run_bench(
        solver=solver, n=20_000, warmup=2, steps=5,
        out_dir=str(tmp_path), verbose=False,
    )
    files = list(tmp_path.glob(f"*-{solver}-20000.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())

    stages = set(data["stages"])
    assert EXPECTED[solver] <= stages
    assert stages <= ALLOWED[solver] | set(EPILOGUE_STAGES)
    assert "solver" not in stages
    for s in data["stages"].values():
        assert s["count"] == 5
        assert s["median_ms"] >= 0.0
        assert s["p90_ms"] >= s["median_ms"]
    assert data["total"]["median_ms"] > 0.0
    assert data["positions_finite"]
    assert data["solver"] == solver and data["n"] == r["n"] > 15_000
    for key in ("cupy_version", "cuda_driver_version", "checksum_position_sum", "git_sha"):
        assert key in data


def test_bench_grid_reuse_reports_gather(tmp_path):
    """With --grid-reuse the settled lattice skips sorts; those show as 'gather'."""
    r = bench_substep.run_bench(
        solver="WCSPH", n=20_000, warmup=8, steps=16, grid_reuse=True,
        substeps_per_frame=2, out_dir=str(tmp_path), verbose=False,
    )
    counts = {k: v["count"] for k, v in r["stages"].items()}
    assert counts.get("sort", 0) + counts.get("gather", 0) == 16
    assert counts.get("gather", 0) > 0
    assert 0.0 < r["sort_skip_rate"] < 1.0


def test_compare_same_run_within_noise():
    a = {"stages": {"sort": {"median_ms": 1.0}, "step1": {"median_ms": 2.0}},
         "total": {"median_ms": 3.0}}
    b = {"stages": {"sort": {"median_ms": 1.01}, "step1": {"median_ms": 2.1}},
         "total": {"median_ms": 3.11}}
    rows = {row[0]: row for row in bench_substep.compare(a, b)}
    assert rows["sort"][4] is True       # +1 %
    assert rows["step1"][4] is False     # +5 %
    assert rows["total"][4] is False
    c = {"stages": {"sort": {"median_ms": 1.0}}, "total": {"median_ms": 1.0}}
    rows = {row[0]: row for row in bench_substep.compare(a, c)}
    assert rows["step1"][4] is False     # stage missing from one run
