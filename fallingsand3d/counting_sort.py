"""Counting sort for spatial hash grid -- replaces cupy.argsort (Thrust).

3-phase counting sort pipeline (all CUDA-graph-capturable):
  1. K_CalcHashCS    -- compute per-particle grid cell hash
  2. K_Histogram     -- count particles per cell (atomicAdd)
  3. cupy.cumsum     -- prefix sum on histogram -> cell offsets table
  4. K_ScatterReorder -- scatter particles to sorted order + build sort_perm

Cell table layout: one uint32 offsets array of num_cells + 1 entries with
offsets[0] == 0 and offsets[c + 1] = sum(histogram[0..c]). Cell c holds the
sorted slots offsets[c] .. offsets[c + 1] - 1, and an empty cell has
start == end. cell_start is the view offsets[:-1] and cell_end the view
offsets[1:], so kernels that take (cell_start, cell_end) read two adjacent
words of the same array. See allocate_cell_tables.

Replaces: cupy.argsort + K_FusedSortReorderBuild.
Advantage: entire pipeline is graph-capturable (no Thrust sync).
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

BLOCK_SIZE = 256


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


def allocate_scratch(num_cells: int, max_particles: int) -> dict:
    """Allocate scratch buffers for counting sort.

    Returns a dict with:
      - histogram: (num_cells,) uint32  -- particle count per cell
      - write_offset: (num_cells,) uint32  -- scatter write position per cell
      - cell_offsets: (num_cells + 1,) uint32  -- see allocate_cell_tables
      - cell_start: view cell_offsets[:-1]  -- first sorted slot per cell
      - cell_end: view cell_offsets[1:]  -- one past the last sorted slot per cell
    """
    offsets, cell_start, cell_end = allocate_cell_tables(num_cells)
    return {
        "histogram": cupy.zeros(num_cells, dtype=cupy.uint32),
        "write_offset": cupy.zeros(num_cells, dtype=cupy.uint32),
        "cell_offsets": offsets,
        "cell_start": cell_start,
        "cell_end": cell_end,
    }


# ---------------------------------------------------------------------------
# Full counting sort pipeline
# ---------------------------------------------------------------------------


def counting_sort_full(
    num_particles: int,
    num_cells: int,
    # Pre-allocated scratch
    histogram: cupy.ndarray,
    write_offset: cupy.ndarray,
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
    kappa: cupy.ndarray,
    kappa_v: cupy.ndarray,
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
    """Run the full counting sort pipeline (hash + histogram + prefix sum + scatter).

    cell_start and cell_end must be the two views of one offsets table from
    allocate_cell_tables (cell_end starts one uint32 after cell_start). The
    prefix sum writes cell_end only; cell_start[0] is the table's fixed zero.

    sorted_anchor, when given, receives a copy of sorted_position. gather_reorder
    never writes it, so it keeps the positions of the last full sort and the
    finalize kernels measure drift since that sort against it.

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

    module = _get_module()

    grid_p = ((num_particles + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block_p = (BLOCK_SIZE,)

    # --- Phase 1: Hash ---
    k_hash = module.get_function("K_CalcHashCS")
    k_hash(grid_p, block_p, (
        np.uint32(num_particles),
        positions,
        hashes,
    ))

    # --- Phase 2: Histogram ---
    # Zero histogram (graph-capturable memset)
    histogram.data.memset_async(0x00, histogram.nbytes)

    k_histogram = module.get_function("K_Histogram")
    k_histogram(grid_p, block_p, (
        np.uint32(num_particles),
        hashes,
        histogram,
    ))

    # --- Phase 3: Prefix sum -> cell offsets ---
    # The inclusive sum goes into offsets[1:] (= cell_end). Because offsets[0]
    # is fixed at 0, offsets[:-1] (= cell_start) is then the exclusive sum:
    # cell_start[c] = sum(histogram[0..c-1]), cell_end[c] = cell_start[c+1].
    # cupy.cumsum is graph-capturable (uses CUB internally)
    cupy.cumsum(histogram, out=cell_end)

    # --- Phase 4: Scatter + Reorder ---
    # Zero write_offset (graph-capturable memset)
    write_offset.data.memset_async(0x00, write_offset.nbytes)

    _null = cupy.ndarray(0, dtype=cupy.uint32)

    k_scatter = module.get_function("K_ScatterReorder")
    k_scatter(grid_p, block_p, (
        np.uint32(num_particles),
        hashes,
        cell_start,
        write_offset,
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
        kappa,
        kappa_v,
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
    kappa: cupy.ndarray,
    kappa_v: cupy.ndarray,
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
    # Ignored: the micropolar angular velocity was removed (n28.140)
    angular_velocity: "Optional[cupy.ndarray]" = None,
    sorted_angular_velocity: "Optional[cupy.ndarray]" = None,
) -> None:
    """Re-scatter unsorted data to sorted order using existing sort_perm.

    Used for grid reuse: when particles moved less than 0.25*h since last sort,
    cell_start/cell_end are still valid. Only the particle data needs re-gathering.
    Cheaper than full counting sort (no hash, no histogram, no prefix sum, no atomics).
    It leaves the drift anchor from the last full sort untouched.
    """
    if num_particles == 0:
        return

    module = _get_module()

    grid = ((num_particles + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

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
        kappa,
        kappa_v,
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
    ))
