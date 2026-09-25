"""Wake propagation kernels: two-phase cell-flag wake system.

Compiles physics/kernels/wake.cu via CuPy RawModule and provides functions
to launch the wake propagation pipeline:

  Phase 1 (K_MarkWakeCells):  Just-woke and fast awake particles stamp their
                              own grid cell.
  Phase 1b (K_MarkWakeCellsRigid): Moving rigid bodies stamp the cells
                              their bounding cube covers.
  Phase 2+3 (K_WakeSleepersAndClearJustWoke): A sleeping particle wakes when
                              any of its 27 cells holds this substep's stamp;
                              JUST_WOKE is cleared on every particle.

Cells hold a generation stamp, frame_counter[0] + 1, instead of a 0/1 flag,
so the table is never cleared per substep (bd-r5-review-n28.141). Pass the
per-substep device counter as ``frame_counter``. Without it the stamp is 1
and run_wake_propagation clears the table first.

Uses shared constant memory from common.cuh:
  c_grid -- GridParams (for position -> grid cell mapping)
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
    """Compile (or return cached) CuPy RawModule from wake.cu."""
    global _module
    if _module is not None:
        return _module

    _ensure_ptx_if_needed()

    kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
    cu_path = os.path.join(kernel_dir, "wake.cu")
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
    d_ptr = module.get_global("c_grid")  # type: ignore[union-attr]
    cupy.cuda.runtime.memcpy(
        int(d_ptr), params.ctypes.data, params.nbytes, 1
    )


# ---------------------------------------------------------------------------
# Array allocation
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256


def allocate_cell_wake_flags(num_cells: int = 125_000) -> cupy.ndarray:
    """Allocate the cell_wake_flags stamp table (uint32, num_cells).

    Returns a CuPy uint32 array of size num_cells, initialized to 0. A
    stamp is never 0, so a fresh table marks nothing.
    """
    return cupy.zeros(num_cells, dtype=cupy.uint32)


# ---------------------------------------------------------------------------
# Kernel launches
# ---------------------------------------------------------------------------

_NULL = np.intp(0)


def _frame_arg(frame_counter: Optional[cupy.ndarray]):
    """Kernel argument for frame_ptr: the device counter, or NULL (stamp 1)."""
    return frame_counter if frame_counter is not None else _NULL



def mark_wake_cells(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    packed_info: cupy.ndarray,
    cell_wake_flags: cupy.ndarray,
    num_particles: Optional[int] = None,
    frame_counter: Optional[cupy.ndarray] = None,
) -> None:
    """Launch K_MarkWakeCells (Phase 1).

    Writes the stamp into the own grid cell of:
    1. Particles with HAS_JUST_WOKE flag (wake cascade).
    2. Active non-sleeping particles moving above V_WAKE threshold
       (so flowing water wakes sleeping sand in adjacent cells).

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Unsorted particle positions.
    velocity : cupy.ndarray, (N, 4) float32
        Unsorted particle velocities.
    packed_info : cupy.ndarray, (N,) uint32
        Unsorted packed_info (read-only for this kernel).
    cell_wake_flags : cupy.ndarray, (num_cells,) uint32
        Cell stamp table.
    num_particles : int, optional
        Number of active particles. Defaults to position.shape[0].
    frame_counter : cupy.ndarray, (1,) uint32, optional
        Per-substep device counter; the stamp is its value + 1. None gives
        stamp 1, so the caller must clear the table first.
    """
    n = num_particles if num_particles is not None else position.shape[0]
    if n == 0:
        return

    module = _get_module()
    kernel = module.get_function("K_MarkWakeCells")  # type: ignore[union-attr]

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(grid, block, (np.uint32(n), position, velocity, packed_info,
                         cell_wake_flags, _frame_arg(frame_counter)))


RIGID_BLOCK_SIZE = 128


def mark_wake_cells_rigid(
    rigid_bodies: cupy.ndarray,
    num_bodies: int,
    cell_wake_flags: cupy.ndarray,
    frame_counter: Optional[cupy.ndarray] = None,
) -> None:
    """Launch K_MarkWakeCellsRigid (Phase 1b), one block per body.

    A body whose fastest point can move faster than V_WAKE stamps every cell
    of its bounding cube. Sleepers probe their 27 cells, so sleeping sand
    up to one cell past a moving body wakes and feels the rigid coupling (bd-r5-review-n28.87).
    Rigid boundary particles are STATIC, so K_MarkWakeCells never marks
    for them. A body at rest marks nothing.

    Parameters
    ----------
    rigid_bodies : cupy.ndarray, RIGID_BODY_DTYPE
        Device body state (RigidBodyManager.d_rigid_bodies).
    num_bodies : int
        Number of live bodies (host-side count, no sync).
    cell_wake_flags : cupy.ndarray, (num_cells,) uint32
        Cell stamp table.
    frame_counter : cupy.ndarray, (1,) uint32, optional
        Per-substep device counter (see mark_wake_cells).
    """
    if num_bodies <= 0:
        return
    module = _get_module()
    kernel = module.get_function("K_MarkWakeCellsRigid")  # type: ignore[union-attr]
    kernel(
        (int(num_bodies),), (RIGID_BLOCK_SIZE,),
        (rigid_bodies, np.int32(num_bodies), cell_wake_flags,
         _frame_arg(frame_counter)),
    )


def wake_sleepers_and_clear_just_woke(
    position: cupy.ndarray,
    packed_info: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    cell_wake_flags: cupy.ndarray,
    num_particles: Optional[int] = None,
    frame_counter: Optional[cupy.ndarray] = None,
) -> None:
    """Launch K_WakeSleepersAndClearJustWoke (Phase 2+3 fused).

    For each sleeping particle, if any of its 27 cells holds the stamp,
    clear SLEEPING flag and reset sleep_counter to 0.
    Also clears the JUST_WOKE flag from all particles.

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Unsorted particle positions.
    packed_info : cupy.ndarray, (N,) uint32
        Unsorted packed_info (read+write).
    sleep_counter : cupy.ndarray, (N,) uint8
        Unsorted sleep counter (write: reset to 0 on wake).
    cell_wake_flags : cupy.ndarray, (num_cells,) uint32
        Cell stamps from Phase 1.
    num_particles : int, optional
        Number of active particles. Defaults to position.shape[0].
    frame_counter : cupy.ndarray, (1,) uint32, optional
        Per-substep device counter (see mark_wake_cells).
    """
    n = num_particles if num_particles is not None else position.shape[0]
    if n == 0:
        return

    module = _get_module()
    kernel = module.get_function("K_WakeSleepersAndClearJustWoke")

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    block = (BLOCK_SIZE,)

    kernel(
        grid, block,
        (np.uint32(n), position, packed_info, sleep_counter, cell_wake_flags,
         _frame_arg(frame_counter)),
    )


def run_wake_propagation(
    position: cupy.ndarray,
    velocity: cupy.ndarray,
    packed_info: cupy.ndarray,
    sleep_counter: cupy.ndarray,
    cell_wake_flags: cupy.ndarray,
    num_particles: Optional[int] = None,
    rigid_bodies: Optional[cupy.ndarray] = None,
    num_bodies: int = 0,
    frame_counter: Optional[cupy.ndarray] = None,
) -> None:
    """Run the 2-phase wake propagation pipeline.

    1. K_MarkWakeCells (Phase 1)
    2. K_MarkWakeCellsRigid (Phase 1b), when rigid_bodies is given and
       num_bodies > 0
    3. K_WakeSleepersAndClearJustWoke (Phase 2+3 fused)

    With frame_counter the table is not cleared: each substep writes a new
    stamp, so older marks stop matching. Without it (stamp 1) the table is
    cleared first.

    Parameters
    ----------
    position : cupy.ndarray, (N, 4) float32
        Unsorted particle positions.
    velocity : cupy.ndarray, (N, 4) float32
        Unsorted particle velocities.
    packed_info : cupy.ndarray, (N,) uint32
        Unsorted packed_info (read+write).
    sleep_counter : cupy.ndarray, (N,) uint8
        Unsorted sleep counter (write on wake).
    cell_wake_flags : cupy.ndarray, (num_cells,) uint32
        Pre-allocated cell stamp table.
    num_particles : int, optional
        Number of active particles.
    rigid_bodies : cupy.ndarray, optional
        Device rigid body state. Moving bodies wake sleepers near them.
    num_bodies : int
        Number of live rigid bodies (host-side count).
    frame_counter : cupy.ndarray, (1,) uint32, optional
        Per-substep device counter; must change every substep.
    """
    n = num_particles if num_particles is not None else position.shape[0]
    if n == 0:
        return

    if frame_counter is None:
        # Fixed stamp 1: clear old marks (async memset, graph-capture safe)
        cell_wake_flags.data.memset_async(0x00, cell_wake_flags.nbytes)

    # Phase 1: stamp the own cell of just-woke and fast-moving particles
    mark_wake_cells(position, velocity, packed_info, cell_wake_flags, n,
                    frame_counter=frame_counter)

    # Phase 1b: moving rigid bodies stamp the cells they cover
    if rigid_bodies is not None and num_bodies > 0:
        mark_wake_cells_rigid(rigid_bodies, num_bodies, cell_wake_flags,
                              frame_counter=frame_counter)

    # Phase 2+3 fused: wake sleepers + clear JUST_WOKE flag
    wake_sleepers_and_clear_just_woke(
        position, packed_info, sleep_counter, cell_wake_flags, n,
        frame_counter=frame_counter,
    )
