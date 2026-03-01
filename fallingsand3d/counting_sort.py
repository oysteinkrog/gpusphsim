"""Counting sort for spatial hash grid -- replaces cupy.argsort (Thrust).

3-phase counting sort pipeline (all CUDA-graph-capturable):
  1. K_CalcHashCS    -- compute per-particle grid cell hash
  2. K_Histogram     -- count particles per cell (atomicAdd)
  3. cupy.cumsum     -- exclusive prefix sum on histogram -> cell_start
  4. K_ScatterReorder -- scatter particles to sorted order + build sort_perm
  5. K_BuildCellEnd  -- cell_end = cell_start + count

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


def allocate_scratch(num_cells: int, max_particles: int) -> dict:
    """Allocate scratch buffers for counting sort.

    Returns a dict with:
      - histogram: (num_cells,) uint32  -- particle count per cell
      - write_offset: (num_cells,) uint32  -- scatter write position per cell
      - cell_start: (num_cells,) uint32  -- prefix sum result = first particle index per cell
      - cell_end: (num_cells,) uint32  -- one past last particle index per cell
    """
    return {
        "histogram": cupy.zeros(num_cells, dtype=cupy.uint32),
        "write_offset": cupy.zeros(num_cells, dtype=cupy.uint32),
        "cell_start": cupy.empty(num_cells, dtype=cupy.uint32),
        "cell_end": cupy.zeros(num_cells, dtype=cupy.uint32),
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
    sorted_hashes: cupy.ndarray,
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
    particle_dye: cupy.ndarray,
    angular_velocity: cupy.ndarray,
    kappa_v: cupy.ndarray,
    lambda_pbf: cupy.ndarray,
    # Sorted particle arrays (write)
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_lifetime: cupy.ndarray,
    sorted_sleep_counter: cupy.ndarray,
    sorted_kappa: cupy.ndarray,
    sorted_particle_dye: cupy.ndarray,
    sorted_angular_velocity: cupy.ndarray,
    sorted_kappa_v: cupy.ndarray,
    sorted_lambda_pbf: cupy.ndarray,
    # FP16 velocity output (OPT-4.3)
    sorted_velocity_h: "Optional[cupy.ndarray]" = None,
    # FP16 temperature + dye outputs (PERF-011)
    sorted_temperature_h: "Optional[cupy.ndarray]" = None,
    sorted_dye_h: "Optional[cupy.ndarray]" = None,
) -> None:
    """Run the full counting sort pipeline (hash + histogram + prefix_sum + scatter + cell_end).

    All operations are CUDA-graph-capturable.
    """
    if num_particles == 0:
        return

    module = _get_module()

    grid_p = ((num_particles + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block_p = (BLOCK_SIZE,)
    grid_c = ((num_cells + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block_c = (BLOCK_SIZE,)

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

    # --- Phase 3: Exclusive prefix sum -> cell_start ---
    # cell_start[c] = sum(histogram[0..c-1])
    # cupy.cumsum is graph-capturable (uses CUB internally)
    cupy.cumsum(histogram, out=cell_start)
    # cumsum gives inclusive sum; shift to exclusive: cell_start[c] = cumsum[c] - histogram[c]
    # Equivalent to: cell_start = [0, cumsum[0], cumsum[1], ..., cumsum[N-2]]
    # But we can't do a shift in-place easily. Instead:
    # exclusive_sum[i] = inclusive_sum[i] - histogram[i]
    cell_start -= histogram

    # --- Phase 4: Scatter + Reorder ---
    # Zero write_offset (graph-capturable memset)
    write_offset.data.memset_async(0x00, write_offset.nbytes)
    # Zero cell_end
    cell_end.data.memset_async(0x00, cell_end.nbytes)

    _null = cupy.ndarray(0, dtype=cupy.uint32)

    k_scatter = module.get_function("K_ScatterReorder")
    k_scatter(grid_p, block_p, (
        np.uint32(num_particles),
        hashes,
        cell_start,
        write_offset,
        cell_end,         # unused by scatter kernel but in signature
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
        particle_dye,
        angular_velocity,
        kappa_v,
        lambda_pbf,
        # Sorted outputs
        sorted_hashes,
        sorted_position,
        sorted_velocity,
        sorted_mass,
        sorted_packed_info,
        sorted_temperature,
        sorted_health,
        sorted_lifetime,
        sorted_sleep_counter,
        sorted_kappa,
        sorted_particle_dye,
        sorted_angular_velocity,
        sorted_kappa_v,
        sorted_lambda_pbf,
        # FP16 velocity output (OPT-4.3)
        sorted_velocity_h if sorted_velocity_h is not None else _null,
        # FP16 temperature + dye (PERF-011)
        sorted_temperature_h if sorted_temperature_h is not None else _null,
        sorted_dye_h if sorted_dye_h is not None else _null,
    ))

    # --- Phase 5: Build cell_end ---
    k_cellend = module.get_function("K_BuildCellEnd")
    k_cellend(grid_c, block_c, (
        np.uint32(num_cells),
        cell_start,
        write_offset,
        cell_end,
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
    particle_dye: cupy.ndarray,
    angular_velocity: cupy.ndarray,
    kappa_v: cupy.ndarray,
    lambda_pbf: cupy.ndarray,
    # Sorted particle arrays (write)
    sorted_position: cupy.ndarray,
    sorted_velocity: cupy.ndarray,
    sorted_mass: cupy.ndarray,
    sorted_packed_info: cupy.ndarray,
    sorted_temperature: cupy.ndarray,
    sorted_health: cupy.ndarray,
    sorted_lifetime: cupy.ndarray,
    sorted_sleep_counter: cupy.ndarray,
    sorted_kappa: cupy.ndarray,
    sorted_particle_dye: cupy.ndarray,
    sorted_angular_velocity: cupy.ndarray,
    sorted_kappa_v: cupy.ndarray,
    sorted_lambda_pbf: cupy.ndarray,
    # FP16 velocity output (OPT-4.3)
    sorted_velocity_h: "Optional[cupy.ndarray]" = None,
    # FP16 temperature + dye outputs (PERF-011)
    sorted_temperature_h: "Optional[cupy.ndarray]" = None,
    sorted_dye_h: "Optional[cupy.ndarray]" = None,
) -> None:
    """Re-scatter unsorted data to sorted order using existing sort_perm.

    Used for grid reuse: when particles moved less than 0.25*h since last sort,
    cell_start/cell_end are still valid. Only the particle data needs re-gathering.
    Cheaper than full counting sort (no hash, no histogram, no prefix sum, no atomics).
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
        particle_dye,
        angular_velocity,
        kappa_v,
        lambda_pbf,
        # Sorted outputs
        sorted_position,
        sorted_velocity,
        sorted_mass,
        sorted_packed_info,
        sorted_temperature,
        sorted_health,
        sorted_lifetime,
        sorted_sleep_counter,
        sorted_kappa,
        sorted_particle_dye,
        sorted_angular_velocity,
        sorted_kappa_v,
        sorted_lambda_pbf,
        # FP16 velocity output (OPT-4.3)
        sorted_velocity_h if sorted_velocity_h is not None else _null,
        # FP16 temperature + dye (PERF-011)
        sorted_temperature_h if sorted_temperature_h is not None else _null,
        sorted_dye_h if sorted_dye_h is not None else _null,
    ))
