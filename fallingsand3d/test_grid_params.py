"""Tests for the kernel-free grid_params module (bd-r5-review-n28.17)."""

import numpy as np
import pytest

import grid_params


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 262144),
        (20_000, 262144),
        (524_288, 262144),
        (600_000, 524288),
        (1_000_000, 524288),
        (1_100_000, 1048576),
        (10_000_000, 1048576),
    ],
)
def test_compute_table_size_thresholds(n, expected):
    assert grid_params.compute_table_size(n) == expected


def test_dtype_is_32_bytes():
    assert grid_params.GRID_PARAMS_DTYPE.itemsize == 32
    assert grid_params.build_grid_params().nbytes == 32


# Expected values are the output of the old hash_sort.build_grid_params_for_world
# (same formula, recorded as literal numbers).
@pytest.mark.parametrize(
    "half, h, n, table_size",
    [
        (1.0, 0.04, 0, 262144),
        (2.5, 0.04, 600_000, 524288),
        (10.0, 0.05, 20_000, 262144),
    ],
)
def test_build_grid_params_for_world_matches_old_output(half, h, n, table_size):
    params, ts = grid_params.build_grid_params_for_world(
        (-half, -half, -half), (half, half, half), h, num_particles=n,
    )
    assert ts == table_size
    assert params.dtype == grid_params.GRID_PARAMS_DTYPE
    assert params.shape == (1,)
    p = params[0]
    np.testing.assert_array_equal(p["grid_min"], np.full(3, -half, np.float32))
    np.testing.assert_array_equal(
        p["grid_delta"], np.full(3, np.float32(1.0 / h), np.float32)
    )
    assert int(p["table_size"]) == table_size
    assert int(p["table_mask"]) == table_size - 1


def test_hash_sort_reexports_grid_params():
    hash_sort = pytest.importorskip("hash_sort")
    assert hash_sort.build_grid_params_for_world is grid_params.build_grid_params_for_world
    assert hash_sort.build_grid_params is grid_params.build_grid_params
    assert hash_sort.compute_table_size is grid_params.compute_table_size
    assert hash_sort.GRID_PARAMS_DTYPE is grid_params.GRID_PARAMS_DTYPE
    assert hash_sort.TABLE_SIZE == grid_params.TABLE_SIZE


def test_simulation_does_not_import_legacy_sort_modules():
    import pathlib

    src = (pathlib.Path(__file__).parent / "simulation.py").read_text(encoding="utf-8")
    assert "hash_sort" not in src
    assert "fused_sort_reorder_build" not in src
