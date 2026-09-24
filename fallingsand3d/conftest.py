"""Pytest configuration: ensure CuPy's NVRTC can locate its builtins DLL.

CuPy 13.x bundles ``nvrtc64_120_0.dll`` but loads ``nvrtc-builtins64_<ver>.dll``
lazily on the *second* on-the-fly compile. If the matching CUDA toolkit ``bin``
directory is not on the DLL search path, that load fails with::

    nvrtc: error: failed to open nvrtc-builtins64_128.dll

The interactive app works because Windows happens to resolve the DLL from the
process PATH, but pytest subprocesses do not always inherit a PATH that points
at the right toolkit version. We add every installed CUDA ``bin`` directory that
contains an ``nvrtc-builtins`` DLL to the DLL search path at import time.
"""
import glob
import os

# CuPy's on-disk kernel cache is keyed on the .cu source text only, not on the
# headers it #includes (common.cuh, sph_shared.cuh). After a header change, a
# module whose .cu file is unchanged would load a stale binary. Keep the cache
# in memory for test runs so every pytest process compiles from current headers.
os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")


def _add_cuda_nvrtc_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    roots = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        os.environ.get("CUDA_PATH", ""),
    ]
    dirs = []
    seen = set()
    for root in roots:
        if not root:
            continue
        for binary in glob.glob(os.path.join(root, "**", "nvrtc-builtins*.dll"), recursive=True):
            d = os.path.dirname(binary)
            if d in seen:
                continue
            seen.add(d)
            dirs.append(d)
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    # cupy's NVRTC is loaded via plain LoadLibrary, which consults the process
    # PATH at load time (add_dll_directory is not honored by that loader), so
    # prepend the toolkit bin dirs to PATH as well.
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_nvrtc_dirs()


# ---------------------------------------------------------------------------
# Stress marker and CuPy pool cleanup (bd-r5-review-n28.7)
# ---------------------------------------------------------------------------
#
# 500K-particle tests are marked ``stress`` and deselected by pytest.ini
# (``addopts = -m "not stress"``). Run them on their own with
# ``pytest -m stress``. Tests in files this bead does not own are marked here
# by node id; tests that carry ``@pytest.mark.stress`` themselves need no entry.

import sys

import pytest

STRESS_TESTS = frozenset({
    "test_build_grid.py::test_500k_no_errors",
    "test_compact.py::test_compact_stress_500k",
    "test_hash_sort.py::test_500k_no_errors",
    "test_reactions.py::test_500k_stress",
    "test_sort_reorder.py::test_500k_sort_reorder_no_errors",
    "test_spawn.py::test_500k_stress",
    "test_step1.py::test_500k_no_errors",
    "test_step2.py::test_500k_no_errors",
    "test_wake.py::test_stress_500k",
    "test_world.py::test_spawn_cube_max_particles",
})


def pytest_collection_modifyitems(config, items):
    for item in items:
        key = f"{item.path.name}::{getattr(item, 'originalname', item.name)}"
        if key in STRESS_TESTS:
            item.add_marker(pytest.mark.stress)


@pytest.fixture(autouse=True)
def _free_cupy_pools():
    """Return cached CuPy device and pinned memory after every test.

    Without this, blocks from a 500K-particle test stay in the pools for the
    rest of the session, so peak memory depends on test order.
    """
    yield
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# Counting-sort scene helper (bd-r5-review-n28.6)
# ---------------------------------------------------------------------------
#
# The app builds its neighbour grid with counting_sort.counting_sort_full
# (Simulation._run_grid_setup). Kernel unit tests use this helper so they run
# on the same cell-table layout as the live path: cell_start comes from the
# exclusive prefix sum of the histogram and an empty cell has cell_end == 0.
# The retired hash_sort/build_grid path used a 0xFFFFFFFF cell_start sentinel
# instead, which the app never produces.


def counting_sort_scene(
    position,
    velocity=None,
    mass=None,
    packed_info=None,
    temperature=None,
    health=None,
    lifetime=None,
    sleep_counter=None,
    grid_params=None,
    upload_kernels=True,
    scratch=None,
):
    """Sort a particle set with counting_sort_full and return the sorted views.

    Inputs may be numpy or cupy arrays in unsorted (original) order. Missing
    arrays get neutral defaults (zero velocity, mass 0.008, WATER/FLUID,
    293 K, full health).

    ``grid_params`` defaults to ``hash_sort.build_grid_params()``. The same
    struct is uploaded to counting_sort and, when ``upload_kernels`` is true,
    to step1 and step2, so table_size/table_mask match across the hash and the
    neighbour walk (CRIT-4). ``scratch`` may be a dict from
    ``counting_sort.allocate_scratch`` to reuse across calls.

    Returns a dict with the sorted arrays under their unsorted names
    (``position``, ``velocity``, ``mass``, ``packed_info``, ``temperature``,
    ``health``, ``lifetime``, ``sleep_counter``), the FP16 mirrors
    (``velocity_h``, ``temperature_h``, ``dye_h``), the cell tables
    (``cell_start``, ``cell_end``), ``sort_perm`` (sorted index -> original
    index, uint32), ``hashes`` (unsorted), ``sorted_hashes``, ``histogram``
    and ``num_cells``.
    """
    import cupy
    import counting_sort
    import hash_sort

    pos = cupy.ascontiguousarray(cupy.asarray(position, dtype=cupy.float32))
    n = int(pos.shape[0])

    def _arr(a, shape, dtype, fill):
        if a is None:
            return cupy.full(shape, fill, dtype=dtype)
        return cupy.ascontiguousarray(cupy.asarray(a, dtype=dtype))

    vel = _arr(velocity, (n, 4), cupy.float32, 0.0)
    m = _arr(mass, (n,), cupy.float32, 0.008)
    pi = _arr(packed_info, (n,), cupy.uint32, 5)  # WATER, FLUID
    temp = _arr(temperature, (n,), cupy.float32, 293.0)
    hp = _arr(health, (n,), cupy.float32, 1.0)
    lt = _arr(lifetime, (n,), cupy.float32, 0.0)
    sc = _arr(sleep_counter, (n,), cupy.uint8, 0)
    kappa = cupy.zeros(n, dtype=cupy.float32)
    kappa_v = cupy.zeros(n, dtype=cupy.float32)
    lambda_pbf = cupy.zeros(n, dtype=cupy.float32)
    dye = cupy.zeros((n, 4), dtype=cupy.float32)
    omega = cupy.zeros((n, 4), dtype=cupy.float32)

    if grid_params is None:
        grid_params = hash_sort.build_grid_params()
    num_cells = int(grid_params[0]["table_size"])
    counting_sort.upload_grid_params(grid_params)
    if upload_kernels:
        import step1
        import step2
        step1.upload_grid_params(grid_params)
        step2.upload_grid_params(grid_params)

    if scratch is None or scratch["histogram"].shape[0] != num_cells:
        scratch = counting_sort.allocate_scratch(num_cells, n)

    out = {
        "position": cupy.empty_like(pos),
        "velocity": cupy.empty_like(vel),
        "mass": cupy.empty_like(m),
        "packed_info": cupy.empty_like(pi),
        "temperature": cupy.empty_like(temp),
        "health": cupy.empty_like(hp),
        "lifetime": cupy.empty_like(lt),
        "sleep_counter": cupy.empty_like(sc),
        "kappa": cupy.empty_like(kappa),
        "particle_dye": cupy.empty_like(dye),
        "angular_velocity": cupy.empty_like(omega),
        "kappa_v": cupy.empty_like(kappa_v),
        "lambda_pbf": cupy.empty_like(lambda_pbf),
        "velocity_h": cupy.zeros((n, 2), dtype=cupy.uint32),
        "temperature_h": cupy.zeros(n, dtype=cupy.float16),
        "dye_h": cupy.zeros((n, 2), dtype=cupy.uint32),
        "hashes": cupy.zeros(n, dtype=cupy.uint32),
        "sorted_hashes": cupy.zeros(n, dtype=cupy.uint32),
        "sort_perm": cupy.zeros(n, dtype=cupy.uint32),
        "cell_start": scratch["cell_start"],
        "cell_end": scratch["cell_end"],
        "histogram": scratch["histogram"],
        "num_cells": num_cells,
    }

    counting_sort.counting_sort_full(
        num_particles=n,
        num_cells=num_cells,
        histogram=scratch["histogram"],
        write_offset=scratch["write_offset"],
        cell_start=scratch["cell_start"],
        cell_end=scratch["cell_end"],
        sort_perm=out["sort_perm"],
        positions=pos,
        hashes=out["hashes"],
        sorted_hashes=out["sorted_hashes"],
        position=pos,
        velocity=vel,
        mass=m,
        packed_info=pi,
        temperature=temp,
        health=hp,
        lifetime=lt,
        sleep_counter=sc,
        kappa=kappa,
        particle_dye=dye,
        angular_velocity=omega,
        kappa_v=kappa_v,
        lambda_pbf=lambda_pbf,
        sorted_position=out["position"],
        sorted_velocity=out["velocity"],
        sorted_mass=out["mass"],
        sorted_packed_info=out["packed_info"],
        sorted_temperature=out["temperature"],
        sorted_health=out["health"],
        sorted_lifetime=out["lifetime"],
        sorted_sleep_counter=out["sleep_counter"],
        sorted_kappa=out["kappa"],
        sorted_particle_dye=out["particle_dye"],
        sorted_angular_velocity=out["angular_velocity"],
        sorted_kappa_v=out["kappa_v"],
        sorted_lambda_pbf=out["lambda_pbf"],
        sorted_velocity_h=out["velocity_h"],
        sorted_temperature_h=out["temperature_h"],
        sorted_dye_h=out["dye_h"],
    )
    out["scratch"] = scratch
    return out


def unsort(sorted_array, sort_perm):
    """Map an array in sorted order back to original particle order."""
    import cupy
    arr = cupy.asarray(sorted_array)
    out = cupy.empty_like(arr)
    out[cupy.asarray(sort_perm).astype(cupy.int64)] = arr
    return out


@pytest.fixture
def sorted_scene():
    """Fixture form of :func:`counting_sort_scene` (returns the builder)."""
    return counting_sort_scene


def default_grid_params():
    """GridParams the kernel tests share: grid_min -1, h=0.04 cells, 2^18 table."""
    import hash_sort
    return hash_sort.build_grid_params()
