"""Counting sort for spatial hash grid -- replaces cupy.argsort (Thrust).

Full sort, five kernel launches (all CUDA-graph-capturable, bd-r5-review-n28.189):
  1. K_HashHistogram -- per-particle grid cell hash, counted per cell (atomicAdd)
  2. K_ScanReduce    -- one total per SCAN_TILE histogram slots
  3. K_ScanDownsweep -- inclusive prefix sum of the histogram -> cell_end
  4. K_ScatterPerm   -- claim each particle's sorted slot, write sort_perm and
                        drain the histogram back to zero
  5. K_GatherReorder -- gather every particle array through sort_perm

The scan is our own: cupy.cumsum takes a CUB workspace from the memory pool
on every call, and inside a CUDA graph capture the instantiated graph would
keep a raw pointer to a block the pool has already taken back. No step
allocates from the pool and no step needs a memset.

Cell table layout: one uint32 offsets array of num_cells + 1 entries with
offsets[0] == 0 and offsets[c + 1] = sum(histogram[0..c]). Cell c holds the
sorted slots offsets[c] .. offsets[c + 1] - 1, and an empty cell has
start == end. cell_start is the view offsets[:-1] and cell_end the view
offsets[1:], so kernels that take (cell_start, cell_end) read two adjacent
words of the same array. See allocate_cell_tables.

Replaces: cupy.argsort + K_FusedSortReorderBuild.
"""

from __future__ import annotations

import os
from typing import Optional

import cupy
import numpy as np

# ---------------------------------------------------------------------------
# CuPy RawModule compilation
# ---------------------------------------------------------------------------

_module: Optional[object] = None

# Zero-length stand-in for a skipped (None) kernel argument; the kernel sees
# a NULL pointer. Made once so a sort never calls the pool allocator, even
# for a size-0 request (bd-r5-review-n28.189).
_null: Optional[cupy.ndarray] = None


def _null_arg() -> cupy.ndarray:
    global _null
    if _null is None:
        _null = _null_arg()
    return _null

BLOCK_SIZE = 256

# Must match SCAN_THREADS * SCAN_ITEMS and SCAN_THREADS in counting_sort.cu.
SCAN_TILE = 1024
SCAN_THREADS = 256


def scan_block_count(num_cells: int) -> int:
    """Number of scan tiles (and block_sums entries) for a table of num_cells."""
    return (num_cells + SCAN_TILE - 1) // SCAN_TILE


def _ensure_ptx_if_needed() -> None:
    """Force PTX compilation when GPU arch exceeds NVRTC's max sm target."""
    from cupy.cuda import compiler as _compiler
    from cupy.cuda import device as _device

    gpu_cc = _device.Device().compute_capability
    nvrtc_max = _compiler._get_max_compute_capability()
    if int(gpu_cc) > int(nvrtc_max):
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
            _compiler._get_arch_for_options_for_nvrtc._cache = {}
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}


def _get_module() -> "object":
    """Compile (or return cached) CuPy RawModule from counting_sort.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "counting_sort.cu")
    with open(cu_path) as f:
        source = f.read()

    _module = cupy.RawModule(
        code=source,
        options=("--std=c++11", "--use_fast_math", f"-I{kernel_dir}"),
    )
    return _module


def get_module() -> "object":
    """Return the compiled CuPy RawModule (public accessor)."""
    return _get_module()


# ---------------------------------------------------------------------------
# Constant memory upload
# ---------------------------------------------------------------------------


def upload_grid_params(params: np.ndarray) -> None:
    """Upload GridParams to ``__constant__ GridParams c_grid``."""
    module = _get_module()
    d_ptr = module.get_global("c_grid")
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


# ---------------------------------------------------------------------------
# Scratch buffer allocation
# ---------------------------------------------------------------------------


def allocate_cell_tables(num_cells: int):
    """Allocate the cell offsets table and its two views.

    Returns (offsets, cell_start, cell_end). offsets has num_cells + 1 uint32
    entries and starts zeroed, so offsets[0] stays 0 for its whole life
    (counting_sort_full writes only offsets[1:]). cell_start = offsets[:-1] and
    cell_end = offsets[1:] share its memory, so cell_end[c] == cell_start[c + 1]
    and an empty cell has cell_start[c] == cell_end[c].
    """
    offsets = cupy.zeros(num_cells + 1, dtype=cupy.uint32)
    return offsets, offsets[:-1], offsets[1:]


def allocate_block_sums(num_cells: int) -> cupy.ndarray:
    """Scan scratch for counting_sort_full: one uint32 per SCAN_TILE cells."""
    return cupy.zeros(scan_block_count(num_cells), dtype=cupy.uint32)


def allocate_scratch(num_cells: int, max_particles: int) -> dict:
    """Allocate scratch buffers for counting sort.

    Returns a dict with:
      - histogram: (num_cells,) uint32  -- particle count per cell. Zero between
        sorts: counting_sort_full drains it back to zero.
      - block_sums: (scan_block_count(num_cells),) uint32  -- scan scratch
      - cell_offsets: (num_cells + 1,) uint32  -- see allocate_cell_tables
      - cell_start: view cell_offsets[:-1]  -- first sorted slot per cell
      - cell_end: view cell_offsets[1:]  -- one past the last sorted slot per cell
    """
    offsets, cell_start, cell_end = allocate_cell_tables(num_cells)
    return {
        "histogram": cupy.zeros(num_cells, dtype=cupy.uint32),
        "block_sums": allocate_block_sums(num_cells),
        "cell_offsets": offsets,
        "cell_start": cell_start,
        "cell_end": cell_end,
    }


# ---------------------------------------------------------------------------
# Full counting sort pipeline
# ---------------------------------------------------------------------------


def _check_optional_pairs(kappa, sorted_kappa, kappa_v, sorted_kappa_v) -> None:
    """The reorder kernels read kappa / kappa_v only when the sorted output is
    given (NULL output = skip the gather and the store). A sorted output
    without its input would read a NULL pointer, so reject it on the host.
    Host-only check on Python objects: no device sync."""
    if sorted_kappa is not None and kappa is None:
        raise ValueError("sorted_kappa given without kappa")
    if sorted_kappa_v is not None and kappa_v is None:
        raise ValueError("sorted_kappa_v given without kappa_v")


def inclusive_scan(
    num_cells: int,
    histogram: cupy.ndarray,
    block_sums: cupy.ndarray,
    out: cupy.ndarray,
) -> None:
    """out[i] = sum(histogram[0..i]) for i < num_cells, in two launches.

    uint32 in and out (sums wrap at 2^32). block_sums is scratch from
    allocate_block_sums(num_cells). out may be an unaligned view such as
    cell_end. Allocates nothing, so it is safe inside a CUDA graph capture.
    """
    module = _get_module()
    grid_s = (scan_block_count(num_cells),)
    block_s = (SCAN_THREADS,)
    cells_u32 = np.uint32(num_cells)
    module.get_function("K_ScanReduce")(grid_s, block_s, (
        cells_u32, histogram, block_sums,
    ))
    module.get_function("K_ScanDownsweep")(grid_s, block_s, (
        cells_u32, histogram, block_sums, out,
    ))


def counting_sort_full(
    num_particles: int,
    num_cells: int,
    # Pre-allocated scratch
    histogram: cupy.ndarray,
    block_sums: cupy.ndarray,
    cell_start: cupy.ndarray,
    cell_end: cupy.ndarray,
    sort_perm: cupy.ndarray,
    # Hash input/output
    positions: cupy.ndarray,
    hashes: cupy.ndarray,
    # Unsorted particle arrays (read)
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    temperature: cupy.ndarray,
    health: cupy.ndarray,
    lifetime: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    kappa: "Optional[cupy.ndarray]",
    kappa_v: "Optional[cupy.ndarray]",
    # Sorted particle arrays (write)
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_lifetime: cupy.ndarray,
    sorted_sleep_counter: cupy.ndarray,
    sorted_kappa: "Optional[cupy.ndarray]",
    sorted_kappa_v: "Optional[cupy.ndarray]",
    # FP16 velocity output (OPT-4.3)
    sorted_velocity_h: "Optional[cupy.ndarray]" = None,
    # Fixed-point temperature output (uint16, T*8) (PERF-011)
    sorted_temperature_h: "Optional[cupy.ndarray]" = None,
    # Grid-reuse drift anchor (bd-r5-review-n28.54)
    sorted_anchor: "Optional[cupy.ndarray]" = None,
    # Ignored: the micropolar angular velocity was removed (n28.140)
    angular_velocity: "Optional[cupy.ndarray]" = None,
    sorted_angular_velocity: "Optional[cupy.ndarray]" = None,
) -> None:
    """Run the full counting sort (hash + histogram + prefix sum + scatter + gather).

    cell_start and cell_end must be the two views of one offsets table from
    allocate_cell_tables (cell_end starts one uint32 after cell_start). The
    prefix sum writes cell_end only; cell_start[0] is the table's fixed zero.

    histogram must be all zero on entry (it is at allocation) and is all zero
    again on return, so it holds no counts after the sort; use
    cell_end - cell_start for them. block_sums is scan scratch from
    allocate_block_sums(num_cells).

    sorted_anchor, when given, receives a copy of sorted_position. Grid-reuse
    gathers never write it, so it keeps the positions of the last full sort
    and the finalize kernels measure drift since that sort against it.

    sorted_kappa, sorted_kappa_v, sorted_velocity_h and sorted_temperature_h
    may be None: the kernel then skips that gather and store. kappa and
    kappa_v may be None when their sorted output is None.

    All operations are CUDA-graph-capturable.
    """
    if num_particles == 0:
        return

    # Host-side pointer check only (no device sync).
    if (cell_end.data.ptr != cell_start.data.ptr + 4
            or cell_start.shape[0] != num_cells
            or cell_end.shape[0] != num_cells):
        raise ValueError(
            "cell_start/cell_end must be the views offsets[:-1] and offsets[1:] "
            "of one table from counting_sort.allocate_cell_tables"
        )
    num_scan_blocks = scan_block_count(num_cells)
    if histogram.shape[0] != num_cells or block_sums.shape[0] < num_scan_blocks:
        raise ValueError("histogram / block_sums do not match num_cells")

    _check_optional_pairs(kappa, sorted_kappa, kappa_v, sorted_kappa_v)

    module = _get_module()

    grid_p = ((num_particles + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block_p = (BLOCK_SIZE,)
    n_u32 = np.uint32(num_particles)

    # 1. Hash and count. histogram is zero here (see the docstring).
    module.get_function("K_HashHistogram")(grid_p, block_p, (
        n_u32, positions, hashes, histogram,
    ))

    # 2 + 3. Inclusive prefix sum of histogram into offsets[1:] (= cell_end).
    # offsets[0] is fixed at 0, so offsets[:-1] (= cell_start) is then the
    # exclusive sum: cell_start[c] = sum(histogram[0..c-1]).
    inclusive_scan(num_cells, histogram, block_sums, cell_end)

    # 4. Sorted slot per particle -> sort_perm; histogram ends at zero.
    module.get_function("K_ScatterPerm")(grid_p, block_p, (
        n_u32, hashes, cell_end, histogram, sort_perm,
    ))

    # 5. Gather the particle data through sort_perm, drift anchor included.
    gather_reorder(
        num_particles, sort_perm,
        position, velocity, mass, packed_info, temperature, health,
        lifetime, sleep_counter, kappa, kappa_v,
        sorted_position, sorted_velocity, sorted_mass, sorted_packed_info,
        sorted_temperature, sorted_health, sorted_lifetime,
        sorted_sleep_counter, sorted_kappa, sorted_kappa_v,
        sorted_velocity_h=sorted_velocity_h,
        sorted_temperature_h=sorted_temperature_h,
        sorted_anchor=sorted_anchor,
    )


def gather_reorder(
    num_particles: int,
    sort_perm: cupy.ndarray,
    # Unsorted particle arrays (read -- updated by K_Integrate)
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    mass: cupy.ndarray,
    packed_info: cupy.ndarray,
    temperature: cupy.ndarray,
    health: cupy.ndarray,
    lifetime: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    kappa: "Optional[cupy.ndarray]",
    kappa_v: "Optional[cupy.ndarray]",
    # Sorted particle arrays (write)
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_lifetime: cupy.ndarray,
    sorted_sleep_counter: cupy.ndarray,
    sorted_kappa: "Optional[cupy.ndarray]",
    sorted_kappa_v: "Optional[cupy.ndarray]",
    # FP16 velocity output (OPT-4.3)
    sorted_velocity_h: "Optional[cupy.ndarray]" = None,
    # Fixed-point temperature output (uint16, T*8) (PERF-011)
    sorted_temperature_h: "Optional[cupy.ndarray]" = None,
    # Drift anchor: counting_sort_full only (bd-r5-review-n28.54)
    sorted_anchor: "Optional[cupy.ndarray]" = None,
    # Ignored: the micropolar angular velocity was removed (n28.140)
    angular_velocity: "Optional[cupy.ndarray]" = None,
    sorted_angular_velocity: "Optional[cupy.ndarray]" = None,
) -> None:
    """Gather unsorted data into sorted order through sort_perm.

    The last stage of counting_sort_full, and on its own the grid-reuse path:
    when particles moved less than 0.25*h since the last sort,
    cell_start/cell_end are still valid and only the particle data needs
    re-gathering. Grid-reuse callers leave sorted_anchor as None so the drift
    anchor from the last full sort stays untouched; counting_sort_full passes
    it. None outputs and inputs work as in counting_sort_full.
    """
    if num_particles == 0:
        return

    module = _get_module()

    grid = ((num_particles + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    _check_optional_pairs(kappa, sorted_kappa, kappa_v, sorted_kappa_v)
    _null = cupy.ndarray(0, dtype=cupy.uint32)

    k_gather = module.get_function("K_GatherReorder")
    k_gather(grid, block, (
        np.uint32(num_particles),
        sort_perm,
        # Unsorted inputs
        position,
        velocity,
        mass,
        packed_info,
        temperature,
        health,
        lifetime,
        sleep_counter,
        kappa if sorted_kappa is not None else _null,
        kappa_v if sorted_kappa_v is not None else _null,
        # Sorted outputs
        sorted_position,
        sorted_velocity,
        sorted_mass,
        sorted_packed_info,
        sorted_temperature,
        sorted_health,
        sorted_lifetime,
        sorted_sleep_counter,
        sorted_kappa if sorted_kappa is not None else _null,
        sorted_kappa_v if sorted_kappa_v is not None else _null,
        # FP16 velocity output (OPT-4.3)
        sorted_velocity_h if sorted_velocity_h is not None else _null,
        # Fixed-point temperature (PERF-011)
        sorted_temperature_h if sorted_temperature_h is not None else _null,
        # Drift anchor (NULL = skip)
        sorted_anchor if sorted_anchor is not None else _null,
    ))
