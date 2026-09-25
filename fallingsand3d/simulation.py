"""SPH simulation orchestrator.

Wires up the full SPH pipeline: Hash -> Sort -> Reorder -> Build -> Step1 ->
Step2 -> Integrate.  Decouples simulation from rendering via a speed parameter,
accuracy parameter (CFL number), and substep budget.

Each render frame:
  1. Pick dt on the host from the max velocity latched by the previous
     frame's async readback (no GPU sync). Adaptive mode (WCSPH): velocity
     CFL, accuracy cap and profile dt_max. Fixed mode (PBF/DFSPH): split
     fixed_dt into more substeps when the fastest particle nears the
     velocity clamp.
  2. Compute sim_steps = clamp(round(speed * wall_dt / sim_dt), 0, max_substeps)
  3. Run sim_steps simulation substeps
  4. Copy UNSORTED pos to the mapped VBO and compute color straight into the
     color VBO (rendering happens once per frame)
"""

from __future__ import annotations

import collections
import math
import time
from typing import Callable, Optional

import numpy as np

from world import World, T_AMBIENT, _DEFAULT_TEMPS
from materials import (
    WATER, SAND, MATERIALS, GRANULAR, GAS, STATIC,
    build_material_array, build_interaction_matrix, upload_to_gpu,
)
from solver_profiles import SolverType, SolverProfile, PROFILES

import cupy

import grid_params as grid_params_mod
import counting_sort
import step1
import step2
import integrate
import reactions
import spawn
import wake
import foam
import implicit_st
from rigid_bodies import pinned_upload

# Adaptive timestep limits
DT_MIN = 1e-5
# Adaptive dt cap used when the solver profile sets no dt_max.
DT_MAX = 0.001
# Accuracy value that maps to dt_max; lower accuracy scales dt down linearly.
ACCURACY_REF = 0.4
# CFL number of the velocity CFL dt <= CFL_NUMBER * h / (c_sound + v_max).
# At v_max = 0 it is the acoustic bound 0.25 * h / c_sound.
CFL_NUMBER = 0.25
# Adaptive dt may grow at most this factor per frame; it shrinks at once.
DT_GROWTH_PER_FRAME = 1.10
# _upload_dt does one constant upload per module, so skip changes under 5%
# unless dt reaches its velocity-free cap.
DT_UPLOAD_REL = 0.05
# Fixed-dt solvers (PBF/DFSPH): split fixed_dt into more substeps once the
# lagged v_max passes this fraction of the velocity clamp at fixed_dt.
FIXED_SUBDIV_MARGIN = 0.5
FIXED_SUBDIV_MAX = 8

# Thermal gate host rule (bd-r5-review-n28.172). Materials that spawn at a
# temperature other than T_AMBIENT, and [a][b] pairs with any non-zero
# interaction entry. A scene with neither has exactly zero heat diffusion and
# exposure, so K_Step1 can skip them.
_NON_AMBIENT_MATERIALS = frozenset(
    m for m, t in _DEFAULT_TEMPS.items() if t != T_AMBIENT)
_INTERACTION_TABLE = build_interaction_matrix()
_PAIR_INTERACTS = ((_INTERACTION_TABLE["reaction_rate"] != 0.0)
                   | (_INTERACTION_TABLE["heat_exchange"] != 0.0))


def _no_mark(label: str) -> None:
    """Default stage callback: does nothing (used by the untimed and graph paths)."""


StageMark = Callable[[str], None]

# Stage labels each solver body passes to mark(), in order.  'blast', 'ist'
# and 'rigid' only appear when that feature is active in the scene.
WCSPH_STAGES = ("sort", "step1", "reactions", "blast", "spawn", "step2",
                "ist", "integrate", "wake")
PBF_STAGES = ("sort", "predict", "lambda0", "reactions", "blast", "spawn",
              "iterations", "finalize", "wake")
DFSPH_STAGES = ("sort", "density_alpha", "reactions", "blast", "spawn",
                "non_pressure", "divergence", "density_solve", "finalize", "wake")
# 'foam_physics' is the tail of the graph body (only with foam on); 'rigid'
# and 'foam' (generation) run in the epilogue, outside the graph.
EPILOGUE_STAGES = ("foam_physics", "rigid", "foam")


class Simulation:
    """Orchestrates the full SPH pipeline with sim/render decoupling.

    Parameters
    ----------
    world : World
        Particle state manager.
    dt : float
        Initial simulation timestep (seconds). Used as fixed_dt_value when
        fixed_dt is ON.
    speed : float
        Time scale: 1.0 = realtime, <1 = slow-mo, >1 = fast-forward.
    accuracy : float
        Timestep scale for adaptive dt (0.1-1.0, default 0.4).
        The cap is dt_max * accuracy / 0.4 (dt_max from the solver profile,
        DT_MAX = 1 ms if it sets none), so 0.4 gives dt_max and 0.2 half of
        it. Values above 0.4 are capped at dt_max. The velocity CFL can
        lower dt below the cap. Ignored when fixed_dt is True.
    fixed_dt : bool
        If True, use fixed timestep (no CFL computation). Default False.
    max_substeps : int
        Maximum substeps per render frame to prevent spiral-of-death.
    clock : callable, optional
        Wall clock for step_frame, in seconds. Defaults to time.perf_counter.
    """

    # _last_max_disp_uint value meaning "no trustworthy readback yet". Any
    # uint at or above the +inf float bits (0x7F800000) never allows a skip.
    _MAX_DISP_UNKNOWN = 0xFFFFFFFF

    def __init__(
        self,
        world: World,
        dt: float = 0.001,
        speed: float = 1.0,
        accuracy: float = 0.4,
        fixed_dt: bool = False,
        max_substeps: int = 40,
        compact_interval: int = 60,
        compact_threshold: int = 100,
        world_half_size: float = 1.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.world = world
        # Wall clock read by step_frame (seconds). Tests pass a fake clock to
        # drive step_frame at a fixed frame time.
        self._clock: Callable[[], float] = clock if clock is not None else time.perf_counter
        self.dt = dt
        self.speed = speed
        self.accuracy = accuracy
        self.fixed_dt = fixed_dt
        self.fixed_dt_value = dt
        self.max_substeps = max_substeps
        self._profile = PROFILES["WCSPH"]
        self._paused = False
        self._prev_speed = speed
        self._last_frame_time: Optional[float] = None
        self._last_substeps = 0
        self._time_accumulator = 0.0  # fractional time carried between frames
        self.sim_time = 0.0
        self._frame_counter = 0
        # First frame checks the parked DEAD reserve (see _maybe_compact).
        self._dead_reserve_check_pending = True
        # Host count of substeps run. The device RNG counter _frame_counter_d
        # is advanced by K_IncrementCounter at the start of each substep body.
        self._substep_counter = 0
        self.compact_interval = compact_interval
        self.compact_threshold = compact_threshold
        # --- CUDA graph capture state ---
        self._sort_perm = cupy.empty(world.max_particles, dtype=cupy.uint32)
        self._frame_counter_d = cupy.zeros(1, dtype=cupy.uint32)
        # world.foam_enabled the graphs were captured with. The body launches
        # K_FoamPhysics only with foam on, so a toggle drops the graphs.
        self._graph_foam_enabled = False
        # Rigid body state the graphs were captured with (see _rigid_graph_key).
        self._graph_rigid_key = (0, False)
        # Two-graph system: separate graphs for full-sort and gather-only paths
        # so sort-skip decisions don't invalidate the graph.
        self._graph_full_sort = None      # cupy.cuda.Graph for full-sort path
        self._graph_gather_only = None    # cupy.cuda.Graph for gather-only path
        self._graph_full_sort_n = 0       # n when full-sort graph was captured
        self._graph_gather_only_n = 0     # n when gather-only graph was captured
        # World.buffer_generation the graphs were captured with. compact()
        # swaps array references, which changes the pointers a graph baked in.
        self._graph_buffer_gen = getattr(world, "buffer_generation", 0)

        # --- Async readback (1-frame-deferred, avoids GPU→CPU sync stalls) ---
        import ctypes as _ctypes
        self._readback_stream = cupy.cuda.Stream(non_blocking=True)
        self._readback_event = cupy.cuda.Event()  # for cross-stream sync
        # Each frame's pair of D2H copies gets an event recorded on
        # _readback_stream after them, queued with the scene epoch it was
        # launched in. The host latches the pinned buffers once a copy from
        # the current epoch is done (non-blocking cudaEventQuery). Copies on
        # one stream land in order, so the pinned value is then that copy or
        # a newer one. The compute stream waits on the newest event at the
        # top of step_frame, so the next full-sort memset of
        # max_displacement and the foam kernels cannot run before or during
        # the copy. An explicit wait is used rather than a blocking readback
        # stream, so the ordering is visible here and does not depend on
        # legacy-stream rules.
        self._readback_pending: collections.deque = collections.deque()
        self._readback_free_events: list = []
        # Newest copy's event. A never-recorded event counts as done.
        self._readback_done = cupy.cuda.Event(disable_timing=True)
        # Bumped by notify_scene_changed(): copies launched before a change
        # describe the old particle set and are not latched.
        self._readback_epoch = 0
        self._readback_latched = 0      # latches of the previous frame's copy
        self._readback_not_ready = 0    # latches where that copy had not landed
        # Pinned host buffers for async D2H copy
        self._pinned_displacement = cupy.cuda.alloc_pinned_memory(4)
        self._pinned_foam_count = cupy.cuda.alloc_pinned_memory(4)
        self._pinned_dead_count = cupy.cuda.alloc_pinned_memory(4)
        # Zero-init pinned buffers
        _ctypes.memset(self._pinned_displacement.ptr, 0, 4)
        _ctypes.memset(self._pinned_foam_count.ptr, 0, 4)
        _ctypes.memset(self._pinned_dead_count.ptr, 0, 4)
        # Previous frame values (used for decisions while async copy is in-flight).
        # The displacement starts at the "unknown" sentinel, which never skips.
        self._last_max_disp_uint = self._MAX_DISP_UNKNOWN
        self._last_foam_count = 0
        # True when the particle state changed outside the substep loop and
        # the GL VBOs have not been refilled since. Set by
        # notify_scene_changed() and set_world_size(); cleared by
        # copy_to_vbos(). main.py skips the VBO copy on frames with no
        # substeps unless this is set (bd-r5-review-n28.150).
        self.scene_dirty = True
        # Max particle speed (m/s) estimated from the latched drift: drift
        # since the last full sort / sim time since that sort. 1-2 frames
        # old. Kept when the latch is unknown (scene change, +inf sentinel).
        self._v_max_lagged = 0.0
        # Substeps fixed_dt is split into (PBF/DFSPH velocity subdivision).
        self._fixed_subdiv = 1
        # Freelist size (_dead_count) at the end of the last substep of the
        # previous frame: the DEAD count in [0, _high_water), since
        # K_Reactions pushes every DEAD slot and K_SpawnGas pops the ones it
        # claims. None until a copy from the current scene epoch lands.
        # _maybe_compact uses it to skip the num_active sync when no
        # compaction or reserve refill can be due (bd-r5-review-n28.135).
        self._last_dead_count: Optional[int] = None
        self._num_active_syncs = 0  # num_active calls made by _maybe_compact
        # Sim time since the last foam generation launch (host float).
        self._foam_emit_dt = 0.0

        # --- Counting sort scratch buffers (allocated in _upload_constants) ---
        self._cs_histogram = None
        self._cs_block_sums = None

        # --- World size ---
        self._world_half_size = world_half_size
        self._table_size = 0  # set by _upload_constants

        # --- Kernel timing mode ---
        self.timing_enabled = False
        self._last_timings: dict = {}      # stage_name -> ms (raw last frame)
        self._timing_ema: dict = {}        # stage_name -> ms (exponential moving avg)
        self._timing_ema_alpha = 0.1       # EMA smoothing factor
        # Deferred timing: hold last substep's events+labels so we can read
        # them one step later without a per-substep synchronize() stall.
        self._pending_timing_events: list = []   # cupy.cuda.Event objects
        self._pending_timing_labels: list = []   # parallel label strings

        # Precompute acoustic speed from materials table
        self._c_sound = self._compute_c_sound()

        # Smoothing length (must match _upload_constants)
        self._h = 0.04

        # --- Grid reuse state ---
        # When the predicted drift since the last full sort stays under the
        # threshold, skip the full counting sort and use gather_reorder
        # (reuse cell_start/cell_end). max_displacement holds the largest
        # |x - x_at_last_full_sort|^2 (drift anchor, bd-r5-review-n28.54).
        # Safety: force sort after MAX_SORT_SKIP_FRAMES consecutive skipped frames.
        self._sort_skip_next = False
        # Threshold: (0.25 * h)^2 on the drift since the last full sort,
        # predicted to the end of the next frame.
        self._sort_skip_threshold_sq = (0.25 * 0.04) ** 2  # 0.0001
        # Host-side substep counts for the drift prediction. Counted on the
        # host from the skip flag, so graph replays need no device state.
        self._substeps_since_sort = 0   # substeps since the last full sort
        self._frame_substeps = 0        # substeps since the last decision
        self._last_disp_substeps = 0    # substeps since sort when the latched copy was taken
        self._sim_time_since_sort = 0.0  # sim seconds since the last full sort
        self._last_disp_sim_time = 0.0   # the same, when the latched copy was taken
        # PBF/DFSPH: tighter threshold (0.15*h) and lower max skip count
        self._sort_skip_threshold_sq_tight = (0.15 * 0.04) ** 2  # 0.000036
        self._sort_skipped_count = 0  # stats for UI
        # n of the last full sort (-1: none yet, or invalidated by a scene change)
        self._last_full_sort_n = -1
        # True when the last _run_grid_setup took the full-sort branch
        self._last_setup_was_full_sort = True
        self._sort_skip_consecutive = 0  # consecutive frames with sort skipped
        # True when _update_rigid_boundary already cleared max_displacement
        # for this substep's full sort, so _run_grid_setup must not clear it
        # again and wipe the boundary drift (bd-r5-review-n28.65).
        self._disp_cleared_for_sort = False
        self._MAX_SORT_SKIP_FRAMES = 4  # force sort after this many frames (WCSPH)
        self._MAX_SORT_SKIP_FRAMES_TIGHT = 2  # force sort after this many frames (PBF/DFSPH)

        # Grid cell tables — allocated in _upload_constants based on world size
        self._cell_start = None
        self._cell_end = None
        self._cell_offsets = None

        # Cell wake flags — allocated in _upload_constants based on world size
        self._cell_wake_flags = None

        # Freelist for spawn/kill system (pre-allocated, reset each step)
        self._dead_indices, self._dead_count = spawn.allocate_freelist(
            world.max_particles
        )

        # --- Implicit surface tension ---
        self.ist_enabled = False
        self.ist_sigma = 0.5
        # Even by default: the result then lands back in sorted_velocity and
        # the odd-count copy-back after run_implicit_st is skipped.
        self.ist_iterations = 4
        self.ist_surface_threshold = 25.0
        self._IST_MAX_PARTICLES = 100_000  # disable above this count

        # Cached sim_params for dt updates (set during _upload_constants)
        self._sim_params: Optional[np.ndarray] = None
        # (module, symbol) -> (module, device address) for _upload_async.
        self._const_ptrs: dict = {}

        # Spawn velocity damping: smooth PBF/DFSPH initial transient
        self._spawn_substep = 0
        self._damping_duration = 30  # substeps

        # SDF object manager (Phase A rigid body collision)
        from rigid_bodies import SDFManager, RigidBodyManager
        self.sdf_manager = SDFManager()
        world.sdf_manager = self.sdf_manager

        # Rigid body manager (Akinci two-way coupling)
        self.rigid_body_manager = RigidBodyManager()
        world.rigid_body_manager = self.rigid_body_manager
        # The manager keeps world.reserved_tail in step with its boundary
        # block, so spawns leave room for it (bd-r5-review-n28.58).
        self.rigid_body_manager.world = world
        world.reserved_tail = 0

        # Thermal gate (bd-r5-review-n28.172). World adds to this set on every
        # spawn. A load or undo replaces it with a set rebuilt from the
        # restored particles, whose temperatures are unknown, so the gate may
        # only turn off while the set is still this object.
        self._thermal_trusted_mats = getattr(world, "_spawned_material_ids", None)
        self._thermal_key: Optional[tuple] = None

        # Compile all kernel modules and upload constant memory
        self._upload_constants()

        # The SDF constant memory lives in the compiled modules, which outlive
        # any one Simulation. Upload the new, empty SDF list now so a previous
        # Simulation's objects cannot act on this scene before the first
        # upload_if_dirty(). Runs once at construction, not per frame.
        self.sdf_manager.force_upload()

    def _compute_c_sound(self) -> float:
        """Compute acoustic speed: max over active materials of sqrt(k * gamma / rho0)."""
        c_sound = 0.0
        for mat in MATERIALS.values():
            # GAS is left out: its soft EOS must not set the liquid CFL.
            if (mat.behavior_class in (STATIC, GAS)
                    or mat.rest_density <= 0.0):
                continue
            k = mat.eos_stiffness
            gamma = mat.eos_gamma
            rho0 = mat.rest_density
            if k > 0.0 and gamma > 0.0 and rho0 > 0.0:
                c = math.sqrt(k * gamma / rho0)
                c_sound = max(c_sound, c)
        return max(c_sound, 0.01)  # floor to avoid zero

    def _dt_max(self) -> float:
        """Adaptive dt cap: the profile's dt_max, or DT_MAX if it sets none."""
        dt_max = getattr(self._profile, 'dt_max', None)
        return float(dt_max) if dt_max else DT_MAX

    def _adaptive_dt_cap(self) -> float:
        """Velocity-free part of the adaptive dt: dt_max * accuracy / ACCURACY_REF."""
        dt_max = self._dt_max()
        dt_accuracy = dt_max * self.accuracy / ACCURACY_REF
        return max(DT_MIN, min(dt_max, dt_accuracy))

    def _compute_adaptive_dt(self, n: int, v_max: Optional[float] = None) -> float:
        """Target adaptive dt (CPU-only, no GPU sync).

          dt_accuracy = dt_max * accuracy / ACCURACY_REF
          dt_cfl      = CFL_NUMBER * h / (c_sound + v_max)
          dt = min(dt_accuracy, dt_cfl) clamped to [DT_MIN, dt_max]

        v_max defaults to the lagged estimate from the async readback
        (_v_max_lagged). At v_max = 0, dt_cfl is the acoustic bound
        0.25 * h / c_sound, about 3.4 ms with c_sound = 2.96 m/s (SAND;
        GAS is excluded). With dt_max = 1 ms the velocity CFL binds above
        about 7 m/s: 10 m/s gives 0.77 ms, 20 m/s gives 0.43 ms.

        There is no viscous CFL. Step2 treats the viscous self-term
        implicitly (bd-r5-review-n28.52), which is stable at any dt.

        step_frame rate-limits this target (_next_adaptive_dt).
        """
        if v_max is None:
            v_max = self._v_max_lagged
        dt_cfl = CFL_NUMBER * self._h / (self._c_sound + max(v_max, 0.0))
        return max(DT_MIN, min(self._adaptive_dt_cap(), dt_cfl))

    def _next_adaptive_dt(self, n: int) -> float:
        """Rate-limited adaptive dt: shrinks at once, grows at most 10% per frame."""
        target = self._compute_adaptive_dt(n)
        if target < self.dt:
            return target
        return min(target, self.dt * DT_GROWTH_PER_FRAME)

    def _next_fixed_subdiv(self) -> int:
        """Substeps to split fixed_dt into for PBF/DFSPH (fixed-dt solvers).

        The velocity clamp at fixed_dt is v_cap = factor * h / fixed_dt
        (4.32 m/s for PBF at 1/120). Once the lagged v_max passes
        FIXED_SUBDIV_MARGIN * v_cap, dt = fixed_dt / n raises the clamp to
        n * v_cap. n grows at once and falls by one per frame.
        """
        vlf = getattr(self._profile, 'velocity_limit_factor', 0.9)
        v_cap = vlf * self._h / max(self.fixed_dt_value, 1e-8)
        target = math.ceil(self._v_max_lagged / (FIXED_SUBDIV_MARGIN * v_cap))
        target = max(1, min(FIXED_SUBDIV_MAX, target))
        if target >= self._fixed_subdiv:
            return target
        return self._fixed_subdiv - 1

    def _upload_dt(self, new_dt: float) -> None:
        """Update dt in c_sim constant memory across all modules that read it."""
        if self._sim_params is None:
            return
        self._sim_params[0]["dt"] = np.float32(new_dt)
        # Recalculate CFL velocity limit for new dt
        vlf = getattr(self._profile, 'velocity_limit_factor', 0.9)
        vel_limit = vlf * self._h / max(new_dt, 1e-8)
        self._sim_params[0]["velocity_limit"] = np.float32(vel_limit)
        self.dt = new_dt
        # WCSPH modules, the active solver module and IST read c_sim.dt.
        # The foam module's c_sim is not rewritten here, only c_foam.dt.
        self._upload_sim_params_async(include_foam=False)
        if self._foam_params is not None:
            self._foam_params[0]["dt"] = np.float32(new_dt)
            self._upload_async(foam._get_module(), "c_foam", self._foam_params)

    def _upload_async(self, module: object, symbol: str,
                      params: np.ndarray) -> None:
        """Async stream-ordered write of params to a module's constant symbol.

        The symbol address is cached per module object. See
        rigid_bodies.PinnedUploadRing for why this does not block the host.
        """
        key = (id(module), symbol)
        entry = self._const_ptrs.get(key)
        if entry is None or entry[0] is not module:
            entry = (module, int(module.get_global(symbol)))  # type: ignore[union-attr]
            self._const_ptrs[key] = entry
        pinned_upload(entry[1], params)

    def _upload_sim_params_async(self, include_foam: bool = True) -> None:
        """Write _sim_params to c_sim in every module that reads it, async.

        Used for the per-substep damping ramp, dt changes and gravity
        changes (bd-r5-review-n28.168): no host-device sync, and captured
        CUDA graphs stay valid because kernels read constant memory when
        they run, not when they are captured.
        """
        mods = [step1._get_module(), step2._get_module(),
                integrate._get_module(), reactions._get_module(),
                spawn._get_module()]
        if self._profile.solver_type == SolverType.PBF:
            import pbf_solver
            mods.append(pbf_solver._get_module())
        elif self._profile.solver_type == SolverType.DFSPH:
            import dfsph_solver
            mods.append(dfsph_solver._get_module())
        mods.append(implicit_st._get_module())
        if include_foam:
            # Foam kernels read c_sim.gravity for spray and foam motion.
            mods.append(foam._get_module())
        for m in mods:
            self._upload_async(m, "c_sim", self._sim_params)

    def _upload_sim_params_all(self) -> None:
        """Re-upload sim_params to all modules (without changing self.dt)."""
        self._upload_sim_params_async(include_foam=True)

    @property
    def gravity_y(self) -> float:
        """Current gravity Y component."""
        if self._sim_params is not None:
            return float(self._sim_params[0]["gravity"][1])
        return -4.0

    def set_gravity(self, gy: float) -> None:
        """Change gravity Y and re-upload to all CUDA constant memory modules."""
        if self._sim_params is None:
            return
        self._sim_params[0]["gravity"][1] = np.float32(gy)
        # Graphs stay valid: kernels read c_sim.gravity at execution time.
        self._upload_sim_params_all()

    @property
    def world_half_size(self) -> float:
        return self._world_half_size

    def set_world_size(self, half_size: float) -> None:
        """Change the world half-extent and re-upload all constants."""
        self._world_half_size = half_size
        self._upload_constants()
        # Re-upload grid/sim constants to the active non-WCSPH solver module too;
        # _upload_constants only touches the WCSPH modules, so PBF/DFSPH would
        # otherwise keep stale grid_min/cell_size after a resize (CRIT-4).
        if self._profile.solver_type == SolverType.PBF:
            import pbf_solver
            self._upload_solver_constants(pbf_solver)
        elif self._profile.solver_type == SolverType.DFSPH:
            import dfsph_solver
            self._upload_solver_constants(dfsph_solver)
        # Invalidate CUDA graphs (grid changed) and reset sort skip
        self._invalidate_graphs()
        self._sort_skip_next = False
        self._sort_skip_consecutive = 0
        # The render color's depth tint uses the world bounds.
        self.scene_dirty = True

    def _upload_constants(self) -> None:
        """Compile all .cu kernels and upload constant memory to each module."""
        hs = self._world_half_size
        wmin = (-hs, -hs, -hs)
        wmax = (hs, hs, hs)

        # Build shared param structs -- auto-size hash table for particle count
        grid_params, table_size = grid_params_mod.build_grid_params_for_world(
            wmin, wmax, self._h,
            num_particles=self.world._high_water,
        )

        # Cache the authoritative grid params so solver modules (PBF/DFSPH) get
        # the SAME table_size/table_mask as counting_sort uses (CRIT-4).
        self._grid_params = grid_params

        # Reallocate cell arrays if table size changed
        if table_size != self._table_size:
            self._table_size = table_size
            # One offsets table of table_size + 1; cell_start and cell_end are
            # its views [:-1] and [1:] (empty cell: start == end).
            (self._cell_offsets, self._cell_start,
             self._cell_end) = counting_sort.allocate_cell_tables(table_size)
            # Wake stamp table: never cleared per substep. Stamps come from
            # the monotonic device counter _frame_counter_d, so old marks never match.
            self._cell_wake_flags = wake.allocate_cell_wake_flags(table_size)
            # Counting sort scratch buffers
            # (histogram is zero between sorts: counting_sort_full drains it)
            self._cs_histogram = cupy.zeros(table_size, dtype=cupy.uint32)
            self._cs_block_sums = counting_sort.allocate_block_sums(table_size)

        # CFL velocity limit: v_max = factor * h / dt
        vlf = getattr(self._profile, 'velocity_limit_factor', 0.9)
        vel_limit = vlf * self._h / max(self.dt, 1e-8)
        # Preserve the user's current gravity across re-uploads (e.g. world
        # resize); only fall back to the default on the very first upload.
        # Previously this hardcoded (0,-4,0) and silently reset user gravity (HIGH-3).
        prev = getattr(self, '_sim_params', None)
        if prev is not None:
            g = prev[0]["gravity"]
            gravity = (float(g[0]), float(g[1]), float(g[2]))
        else:
            gravity = (0.0, -4.0, 0.0)
        sim_params = step1.build_sim_params(
            smoothing_length=self._h,
            particle_mass=0.02,
            particle_spacing=0.02,
            gravity=gravity,
            dt=self.dt,
            restitution=0.3,
            wall_friction=0.5,
            world_min=wmin,
            world_max=wmax,
            velocity_limit=vel_limit,
            thermal_enabled=self._thermal_rule(),
        )
        self._sim_params = sim_params
        precalc_params = step1.build_precalc_params(
            smoothing_length=0.04,
            viscosity=1.0,
        )
        granular_params = self._wcsph_granular_params()
        materials_data = build_material_array()
        interactions_data = build_interaction_matrix()

        # --- counting_sort module: c_grid ---
        counting_sort.upload_grid_params(grid_params)

        # --- step1 module: c_grid, c_sim, c_precalc, c_materials, c_interactions ---
        step1.upload_grid_params(grid_params)
        step1.upload_sim_params(sim_params)
        step1.upload_precalc_params(precalc_params)
        step1.upload_materials(materials_data)
        step1.upload_interactions(interactions_data)

        # --- step2 module: c_grid, c_sim, c_precalc, c_materials, c_granular ---
        step2.upload_grid_params(grid_params)
        step2.upload_sim_params(sim_params)
        step2.upload_precalc_params(precalc_params)
        step2.upload_materials(materials_data)
        step2.upload_granular_params(granular_params)

        # --- integrate module: c_grid, c_sim, c_materials ---
        # c_grid feeds the STATIC push-out and the own-cell wake mark
        # (bd-r5-review-n28.166); it was never uploaded before.
        integrate.upload_grid_params(grid_params)
        integrate.upload_sim_params(sim_params)
        integrate.upload_materials(materials_data)

        # --- reactions module: c_grid, c_sim, c_materials ---
        reactions.upload_grid_params(grid_params)
        reactions.upload_sim_params(sim_params)
        reactions.upload_materials(materials_data)

        # --- spawn module: c_sim, c_materials ---
        spawn.upload_sim_params(sim_params)
        spawn.upload_materials(materials_data)

        # --- wake module: c_grid ---
        wake.upload_grid_params(grid_params)

        # --- materials module (its own internal module): c_materials, c_interactions ---
        upload_to_gpu()

        # --- foam module: c_foam, c_sim ---
        foam_params = foam.build_foam_params(
            max_foam=self.world._max_foam,
            dt=self.dt,
        )
        self._foam_params = foam_params
        foam.upload_foam_params(foam_params)
        foam.upload_sim_params(sim_params)

        # --- implicit_st module: c_grid, c_sim, c_precalc, c_materials, c_ist ---
        implicit_st.upload_grid_params(grid_params)
        implicit_st.upload_sim_params(sim_params)
        implicit_st.upload_precalc_params(precalc_params)
        implicit_st.upload_materials(materials_data)
        implicit_st.upload_ist_params(
            sigma=self.ist_sigma,
            surface_threshold=self.ist_surface_threshold,
            num_iterations=self.ist_iterations,
        )

    def update_ist_params(self) -> None:
        """Re-upload IST constant memory after UI parameter change."""
        implicit_st.upload_ist_params(
            sigma=self.ist_sigma,
            surface_threshold=self.ist_surface_threshold,
            num_iterations=self.ist_iterations,
        )

    @property
    def solver_profile(self) -> SolverProfile:
        return self._profile

    def set_solver_profile(self, profile: SolverProfile) -> None:
        """Switch to a new solver profile.

        Updates timestep, substep budget, and solver-specific parameters.
        Invalidates CUDA graph. Allocates solver-specific arrays if needed.
        Uploads ALL shared constant memory to the solver module.
        """
        self._profile = profile
        self.dt = profile.dt
        self.fixed_dt_value = profile.dt
        self.fixed_dt = profile.fixed_dt
        self.max_substeps = profile.max_substeps
        self.accuracy = profile.accuracy
        self._fixed_subdiv = 1

        # Upload new dt to all WCSPH modules
        self._upload_dt(profile.dt)

        # Allocate solver-specific arrays
        if profile.solver_type == SolverType.PBF:
            self.world.allocate_pbf_arrays()
        elif profile.solver_type == SolverType.DFSPH:
            self.world.allocate_dfsph_arrays()

        # Upload ALL shared constants to solver module (each RawModule has own address space)
        if profile.solver_type == SolverType.PBF:
            import pbf_solver
            self._upload_solver_constants(pbf_solver)
            pbf_solver.upload_pbf_params(profile)
            # PBF uses position-space friction (much lower ratio than force-space)
            # Lower vorticity/surface tension: fixed dt (no CFL) can cause energy feedback
            granular_params = step2.build_granular_params(
                tan_phi_f=profile.pbf_friction_ratio,
                cohesion=profile.pbf_friction_cohesion,
                vorticity_epsilon=0.001,
                surface_tension_gamma=0.0,
            )
            pbf_solver.upload_granular_params(granular_params)
        elif profile.solver_type == SolverType.DFSPH:
            import dfsph_solver
            self._upload_solver_constants(dfsph_solver)
            dfsph_solver.upload_dfsph_params(profile)
            # DFSPH uses lower XSPH than WCSPH (implicit solver already smooths)
            dfsph_solver.upload_granular_params(step2.build_granular_params(
                xsph_epsilon=0.1,
            ))
        else:
            # WCSPH: step2 reads xsph_epsilon, force_scale and mu0 from
            # c_granular, so re-upload them from the new profile.
            step2.upload_granular_params(self._wcsph_granular_params())

        # Drop the old solver's stage timings so the panel never shows them
        # as live numbers for the new solver.
        self._clear_timings()

        # Invalidate CUDA graphs and reset grid reuse
        self._invalidate_graphs()
        self._sort_skip_next = False
        self._sort_skip_consecutive = 0
        # Reset spawn damping for new solver
        self._spawn_substep = 0

    def _clear_timings(self) -> None:
        """Forget all per-stage timings, including the unread events."""
        self._last_timings = {}
        self._timing_ema = {}
        self._pending_timing_events = []
        self._pending_timing_labels = []

    def timing_stages(self) -> tuple:
        """Stage labels _sim_step_timed records for the active solver, in order.

        The solver body stages followed by the epilogue stages.  Optional
        stages ('blast', 'ist', 'rigid') only get a timing when active.
        """
        solver = self._profile.solver_type
        if solver == SolverType.PBF:
            body = PBF_STAGES
        elif solver == SolverType.DFSPH:
            body = DFSPH_STAGES
        else:
            body = WCSPH_STAGES
        return body + EPILOGUE_STAGES

    def _wcsph_granular_params(self):
        """GranularParams for step2 built from the active WCSPH profile."""
        profile = self._profile
        return step2.build_granular_params(
            xsph_epsilon=profile.xsph_epsilon,
            force_scale=profile.force_scale,
            mu0=profile.mu0,
        )

    def _upload_solver_constants(self, solver_module) -> None:
        """Upload all shared constants (grid, sim, precalc, materials, interactions) to a solver module."""
        # Reuse the exact grid params counting_sort/step1/step2 use, so the
        # solver kernels hash into the same table (matching table_size/mask and
        # grid_min/cell_size). Rebuilding here without num_particles previously
        # left the solver on the default 262144-entry table (CRIT-4).
        grid_params = getattr(self, '_grid_params', None)
        if grid_params is None:
            hs = self._world_half_size
            wmin = (-hs, -hs, -hs)
            wmax = (hs, hs, hs)
            grid_params, _ = grid_params_mod.build_grid_params_for_world(
                wmin, wmax, self._h, num_particles=self.world._high_water,
            )
        precalc_params = step1.build_precalc_params(smoothing_length=0.04, viscosity=1.0)
        materials_data = build_material_array()
        interactions_data = build_interaction_matrix()

        solver_module.upload_grid_params(grid_params)
        solver_module.upload_sim_params(self._sim_params)
        solver_module.upload_precalc_params(precalc_params)
        solver_module.upload_materials(materials_data)
        solver_module.upload_interactions(interactions_data)

        # Re-upload SDF objects to new solver module's constant memory
        if hasattr(self, 'sdf_manager'):
            self.sdf_manager.force_upload()

    def _update_rigid_boundary(self, n: int) -> int:
        """Keep the rigid boundary block in the unsorted arrays up to date.

        The block lives at [rbm.boundary_offset, +num_boundary_particles)
        inside world._high_water. When rbm.boundary_injected is False (new
        preset, scene load, undo, R, or a block broken by compaction) the block
        is written once by inject_boundary_particles(), which syncs. Every
        substep then runs K_UpdateBoundaryParticles at the stored offset, so
        positions, velocities and psi come from the current GPU body state.

        The kernel also feeds the boundary drift since the last full sort into
        world.max_displacement, so a moving body forces a full sort the same
        way moving fluid does (bd-r5-review-n28.65). On a substep that runs a
        full sort it writes the anchor first. The body moves after the solver,
        so the kernel extrapolates one substep to match the fluid, which
        records its drift after it moves. On a full-sort substep the tracker
        is cleared here, before the kernel, and _run_grid_setup skips its own
        clear. Until the anchor has been written for the current block this
        substep is made a full sort.

        Returns world._high_water (the new n when injection grew the range).
        """
        rbm = self.rigid_body_manager
        w = self.world
        if not rbm.boundary_injected:
            rbm.inject_boundary_particles(w)
        if rbm.boundary_offset >= 0:
            full_sort = self._next_setup_is_full_sort(w._high_water)
            if not full_sort and not rbm.boundary_anchor_valid:
                # No anchor for this block yet: drift is unknown, so sort.
                self._sort_skip_next = False
                self._sort_skip_consecutive = 0
                full_sort = True
            if full_sort:
                w.max_displacement.data.memset_async(0x00, w.max_displacement.nbytes)
                self._disp_cleared_for_sort = True
            integrate.update_boundary_particles(
                rbm.boundary_data,
                rbm.d_rigid_bodies,
                w.position,
                w.velocity,
                w.mass,
                offset=rbm.boundary_offset,
                num_boundary=rbm.num_boundary_particles,
                anchor=rbm.boundary_anchor,
                max_displacement=w.max_displacement,
                reset_anchor=full_sort,
            )
            if full_sort:
                rbm.boundary_anchor_valid = True
        return w._high_water

    def _next_setup_is_full_sort(self, n: int) -> bool:
        """True when the next _run_grid_setup(n) will run a full sort.

        Mirrors its branch: a full sort unless the skip flag is set and n
        still matches the last full sort. Host-only, no device sync.
        """
        return (not self._sort_skip_next
                or n != getattr(self, '_last_full_sort_n', -1))

    def reset_scene_objects(self) -> None:
        """Remove all SDF objects and rigid bodies (bd-r5-review-n28.39).

        Shared by every path that replaces the scene with a new one: the R
        reset, the solver switch and the preset load. Using one helper keeps
        the three paths from drifting apart. reset() also clears the boundary
        injection state, so no stale boundary block is written into slots the
        caller has just zeroed. The zero body count is uploaded to every
        compiled module so the kernels stop reading the old bodies.

        Runs on scene events only; it uploads to constant memory and is not
        meant for the per-substep path.
        """
        self.sdf_manager.clear()
        self.rigid_body_manager.reset()
        self.rigid_body_manager.upload(self.get_all_modules())

    def notify_scene_changed(self) -> None:
        """Signal that the particle set was mutated outside the substep loop
        (brush/gas spawn, R reset, preset load, Ctrl+Z undo, timeline restore).

        Forces a full counting sort on the next step and drops any pending
        grid-reuse skip, so the reuse path never re-gathers through a stale
        sort permutation (wrong binning when n is unchanged, or out-of-bounds
        reads when n grew). n-change alone is caught by the guard in
        _run_grid_setup; this also covers same-n mutations. (CRIT-5)
        """
        self._sort_skip_next = False
        self._sort_skip_consecutive = 0
        self._last_full_sort_n = -1
        # Copies in flight describe the old particle set: drop them and treat
        # the displacement as unknown until a copy from after this change lands.
        self._last_max_disp_uint = self._MAX_DISP_UNKNOWN
        self._last_dead_count = None
        self._readback_epoch += 1
        # Check the parked DEAD reserve on the next frame (bd-r5-review-n28.27).
        self._dead_reserve_check_pending = True
        # The VBOs show the old particle set (bd-r5-review-n28.150).
        self.scene_dirty = True

    def _run_grid_setup(self, n: int, force_sort: bool = False) -> None:
        """Common grid setup: full counting sort or gather-only (grid reuse).

        When _sort_skip_next is True and force_sort is False, uses gather_reorder
        to re-scatter unsorted data using the previous sort_perm.  Cell_start/cell_end
        are reused from the previous frame.  Saves hash, histogram, prefix_sum, and
        cell_end computation.
        """
        w = self.world

        # Safety guard: the grid-reuse gather reads self._sort_perm[:n], which is
        # only valid for the n that produced it. If the particle count changed
        # since the last full sort (brush spawn, gas spawn, preset load, reset,
        # undo, timeline restore), perm entries in [last_n:n] are uninitialised
        # -> garbage indices -> OOB device reads. Force a full sort on any n
        # change so no mutation path can leave a stale/undersized perm (CRIT-5).
        if n != getattr(self, '_last_full_sort_n', -1):
            force_sort = True

        # The reorder only fills the sorted buffers the active solver reads:
        # the warm-start state (kappa, kappa_v) for DFSPH only, and the FP16
        # mirrors (velocity, temperature) for WCSPH only, since only step1,
        # step2 and spawn read them. A skipped buffer keeps its unsorted
        # values (only K_DFSPH_Finalize writes kappa back), so a switch back
        # to DFSPH gathers the same warm start. The pointer set depends only
        # on the solver, and a solver switch invalidates the CUDA graphs, so
        # a captured graph always sees one fixed set (bd-r5-review-n28.45,
        # bd-r5-review-n28.165).
        solver = self._profile.solver_type
        is_pbf = solver == SolverType.PBF
        is_dfsph = solver == SolverType.DFSPH
        is_wcsph = not (is_pbf or is_dfsph)
        sorted_kappa = w.sorted_kappa if is_dfsph else None
        sorted_kappa_v = w.sorted_kappa_v if is_dfsph else None
        sorted_velocity_h = w.sorted_velocity_h if is_wcsph else None
        sorted_temperature_h = w.sorted_temperature_h if is_wcsph else None

        # Host-side record of the branch taken, read by _capture_graph to check
        # that a gather-only graph never bakes in a full sort.
        self._last_setup_was_full_sort = not (self._sort_skip_next and not force_sort)

        if self._sort_skip_next and not force_sort:
            # Grid reuse: just re-gather unsorted -> sorted using old sort_perm
            self._sort_skipped_count += 1
            counting_sort.gather_reorder(
                num_particles=n,
                sort_perm=self._sort_perm[:n],
                position=w.position,
                velocity=w.velocity,
                mass=w.mass,
                packed_info=w.packed_info,
                temperature=w.temperature,
                health=w.health,
                lifetime=w.lifetime,
                sleep_counter=w.sleep_counter,
                kappa=w.kappa if is_dfsph else None,
                kappa_v=w.kappa_v if is_dfsph else None,
                sorted_position=w.sorted_position,
                sorted_velocity=w.sorted_velocity,
                sorted_mass=w.sorted_mass,
                sorted_packed_info=w.sorted_packed_info,
                sorted_temperature=w.sorted_temperature,
                sorted_health=w.sorted_health,
                sorted_lifetime=w.sorted_lifetime,
                sorted_sleep_counter=w.sorted_sleep_counter,
                sorted_kappa=sorted_kappa,
                sorted_kappa_v=sorted_kappa_v,
                sorted_velocity_h=sorted_velocity_h,
                sorted_temperature_h=sorted_temperature_h,
            )
            return

        # Full sort: reset max_displacement tracker (fresh baseline for grid reuse).
        # With rigid bodies _update_rigid_boundary has already cleared it and
        # recorded the boundary drift for this substep. A captured graph bakes
        # in the branch; _sim_step recaptures when _rigid_graph_key changes.
        if self._disp_cleared_for_sort:
            self._disp_cleared_for_sort = False
        else:
            w.max_displacement.data.memset_async(0x00, w.max_displacement.nbytes)
        # Record the count this full sort was built for; grid reuse is only
        # valid while n stays equal to this (see guard above).
        self._last_full_sort_n = n

        # PBF queries the grid with predicted positions x*, so bucket the cells
        # by x* too. K_PBF_Predict runs here on the unsorted arrays into the
        # otherwise unused w.predicted_position; the per-particle result is the
        # same one _run_pbf_body gets after the sort, so the cell of every x*
        # matches the bucket it was sorted into (bd-r5-review-n28.55).
        hash_positions = w.position[:n]
        if is_pbf:
            import pbf_solver
            pbf_solver.pbf_predict(
                w.position[:n], w.velocity[:n],
                w.packed_info[:n], w.temperature[:n],
                w.predicted_position,
            )
            hash_positions = w.predicted_position[:n]

        counting_sort.counting_sort_full(
            num_particles=n,
            num_cells=self._table_size,
            # Scratch buffers
            histogram=self._cs_histogram,
            block_sums=self._cs_block_sums,
            cell_start=self._cell_start,
            cell_end=self._cell_end,
            sort_perm=self._sort_perm[:n],
            # Hash I/O
            positions=hash_positions,
            hashes=w.hashes,
            # Unsorted particle arrays
            position=w.position,
            velocity=w.velocity,
            mass=w.mass,
            packed_info=w.packed_info,
            temperature=w.temperature,
            health=w.health,
            lifetime=w.lifetime,
            sleep_counter=w.sleep_counter,
            kappa=w.kappa if is_dfsph else None,
            kappa_v=w.kappa_v if is_dfsph else None,
            # Sorted particle arrays
            sorted_position=w.sorted_position,
            sorted_velocity=w.sorted_velocity,
            sorted_mass=w.sorted_mass,
            sorted_packed_info=w.sorted_packed_info,
            sorted_temperature=w.sorted_temperature,
            sorted_health=w.sorted_health,
            sorted_lifetime=w.sorted_lifetime,
            sorted_sleep_counter=w.sorted_sleep_counter,
            sorted_kappa=sorted_kappa,
            sorted_kappa_v=sorted_kappa_v,
            sorted_velocity_h=sorted_velocity_h,
            sorted_temperature_h=sorted_temperature_h,
            sorted_anchor=w.sorted_anchor,
        )

    def _run_reactions_spawn(
        self,
        n: int,
        mark: StageMark = _no_mark,
        predicted_position: cupy.ndarray | None = None,
    ) -> None:
        """Common reactions + spawn pass (used by all solvers after density).

        mark(label) is called after each stage ('reactions', 'blast', 'spawn');
        only the timed path passes a real callback.

        predicted_position is PBF's sorted_predicted_position: K_PBF_Predict has
        already run, so K_SpawnGas writes each steam child's prediction there,
        and the gunpowder burst and blast wave add dt * dv to it so Finalize's
        (x* - x)/dt keeps their kicks.
        WCSPH and DFSPH pass None (DFSPH reuses that buffer as vt scratch).
        """
        w = self.world
        # Reset freelist counter (graph-safe memset)
        spawn.reset_freelist(self._dead_count)

        # Reactions (always included — zero-cost early-exit for non-reactive)
        reactions.compute_reactions(
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            w.sorted_health[:n],
            w.sorted_lifetime[:n],
            w.sorted_velocity[:n],
            w.sorted_exposure_heat[:n],
            w.sorted_exposure_corrode[:n],
            frame_d=self._frame_counter_d,
            dead_indices=self._dead_indices,
            dead_count=self._dead_count,
            sorted_mass=w.sorted_mass[:n],
            predicted_position=(predicted_position[:n]
                                if predicted_position is not None else None),
        )
        mark("reactions")

        # Blast wave: radial impulse from freshly-exploded gunpowder
        from materials import GUNPOWDER
        if GUNPOWDER in w._spawned_material_ids:
            # Blast impulse on rigid boundary particles goes to the body
            # force/torque accumulators (zeroed in the substep prologue).
            rbm = self.rigid_body_manager
            has_bodies = rbm.num_bodies > 0
            reactions.compute_blast_wave(
                w.sorted_position[:n],
                w.sorted_velocity[:n],
                w.sorted_packed_info[:n],
                w.sorted_lifetime[:n],
                self._cell_start,
                self._cell_end,
                smoothing_length=self._h,
                predicted_position=(predicted_position[:n]
                                    if predicted_position is not None else None),
                sorted_mass=w.sorted_mass[:n],
                d_rigid_bodies=rbm.d_rigid_bodies if has_bodies else None,
                d_rigid_forces=rbm.rigid_forces if has_bodies else None,
                d_rigid_torques=rbm.rigid_torques if has_bodies else None,
            )
            mark("blast")

        # Spawn (always included — zero-cost when no SPAWN_GAS flags set)
        solver = self._profile.solver_type
        is_pbf = solver == SolverType.PBF
        is_dfsph = solver == SolverType.DFSPH
        is_wcsph = solver not in (SolverType.PBF, SolverType.DFSPH)
        spawn.compute_spawn(
            w.sorted_packed_info[:n],
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_veleval[:n],
            w.sorted_mass[:n],
            w.sorted_temperature[:n],
            w.sorted_health[:n],
            w.sorted_lifetime[:n],
            w.sorted_sleep_counter[:n],
            w.sorted_density[:n],
            w.sorted_shear_rate[:n],
            self._dead_indices,
            self._dead_count,
            max_displacement=w.max_displacement,
            # Per-slot state the child must not inherit from the dead slot. The
            # pointer set depends only on the solver, so a captured CUDA graph
            # always sees one fixed set (a solver switch recaptures it).
            predicted_position=predicted_position,
            velocity_h=w.sorted_velocity_h if is_wcsph else None,
            pressure=w.sorted_pressure if is_wcsph else None,
            angular_velocity=w.sorted_angular_velocity,
            kappa=w.sorted_kappa if is_dfsph else None,
            kappa_v=w.sorted_kappa_v if is_dfsph else None,
            lambda_pbf=w.sorted_lambda_pbf if is_pbf else None,
        )
        mark("spawn")

    def _run_graph_body(self, n: int, mark: StageMark = _no_mark) -> None:
        """Execute the graph-capturable portion of the pipeline.

        Dispatches to the correct solver body based on the active profile.
        mark is a per-stage callback for the timed path.  It must stay the
        no-op default during graph capture (event records cannot be captured).
        Each solver body reports its own stage labels (see PBF_STAGES,
        DFSPH_STAGES and WCSPH_STAGES).

        The first node advances the device RNG counter by one, so each
        substep gets a new seed on replay too. The last is foam physics
        (with foam on), whose pool buffers are never swapped
        (see _maybe_compact_foam), so the captured pointers stay valid.
        """
        foam.increment_counter(self._frame_counter_d)
        if self._profile.solver_type == SolverType.PBF:
            self._run_pbf_body(n, mark)
        elif self._profile.solver_type == SolverType.DFSPH:
            self._run_dfsph_body(n, mark)
        else:
            self._run_wcsph_body(n, mark)
        w = self.world
        if w.foam_enabled:
            # Physics for existing foam, before this frame's generation, so
            # new particles get physics next substep. The kernel reads the
            # live count from foam_count on the device.
            foam.foam_physics(w.foam_position, w.foam_velocity, w.foam_count,
                              w._max_foam)
            mark("foam_physics")

    def _run_wcsph_body(self, n: int, mark: StageMark = _no_mark) -> None:
        """WCSPH pipeline: Grid -> Step1 -> Reactions -> Spawn -> Step2 -> Integrate -> Wake.

        mark(label) is called after each stage; see _run_graph_body.
        """
        w = self.world

        # Grid setup (may skip sort if particles barely moved).
        # max_displacement is reset only on full sort (in _run_grid_setup),
        # so it accumulates across substeps since the last full sort.
        self._run_grid_setup(n)
        mark("sort")

        # Step1: density + strain-rate + heat diffusion + exposure + vorticity + normal.
        # K_Step1 reads each neighbour's previous-substep density from
        # sorted_position.w (K_Integrate writes it, the reorder gathers it), so
        # WCSPH needs no density_in snapshot (bd-r5-review-n28.163).
        step1.compute_step1(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            self._cell_start,
            self._cell_end,
            density_out=w.sorted_density,
            shear_rate_out=w.sorted_shear_rate,
            dTdt_out=w.sorted_dTdt,
            exposure_heat_out=w.sorted_exposure_heat,
            exposure_corrode_out=w.sorted_exposure_corrode,
            vorticity_out=w.sorted_vorticity,
            normal_out=w.sorted_normal,
            velocity_h=w.sorted_velocity_h,
            pressure_out=w.sorted_pressure,
            temperature_h=w.sorted_temperature_h,
            pr2_omega_out=w.sorted_pr2_omega,
            # Inactive refresh cadence (bd-r5-review-n28.159). A full sort
            # reorders the sorted buffers, so every inactive particle must
            # compute that substep. Each CUDA graph bakes in its own branch.
            frame_counter=self._frame_counter_d,
            force_refresh=self._last_setup_was_full_sort,
        )
        mark("step1")

        # Pack density into position.w now done inside K_Step1 (PERF-004)
        # Pressure pre-computed inside K_Step1 (PERF-007)

        # Reactions + Spawn
        self._run_reactions_spawn(n, mark)

        # Step2: pressure + viscosity + XSPH + vorticity confinement + surface tension
        # (density read from position.w, packed by K_PackDensity above)
        rbm = self.rigid_body_manager
        step2.compute_step2(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            w.sorted_shear_rate[:n],
            self._cell_start,
            self._cell_end,
            vorticity_in=w.sorted_vorticity,
            normal_in=w.sorted_normal,
            sph_force_out=w.sorted_sph_force,
            veleval_out=w.sorted_veleval,
            velocity_h=w.sorted_velocity_h,
            pressure_in=w.sorted_pressure,
            d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
            d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
            d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            pr2_omega_in=w.sorted_pr2_omega,
        )
        mark("step2")

        # Implicit surface tension (WCSPH only, quality mode for < 100K particles)
        if self.ist_enabled and n < self._IST_MAX_PARTICLES:
            # No-op after the substep prologue; covers direct body calls.
            w.ensure_ist_scratch(self._IST_MAX_PARTICLES)
            result = implicit_st.run_implicit_st(
                w.sorted_velocity[:n],
                # Own scratch: sorted_veleval holds the step2 XSPH velocity
                # that integrate reads, so IST must not overwrite it.
                w.sorted_ist_scratch[:n],
                w.sorted_position[:n],
                w.sorted_density[:n],
                w.sorted_mass[:n],
                w.sorted_packed_info[:n],
                w.sorted_normal[:n],
                self._cell_start,
                self._cell_end,
                num_iterations=self.ist_iterations,
            )
            # If result ended up in the scratch (odd iteration count), copy back to velocity
            if result.data.ptr != w.sorted_velocity[:n].data.ptr:
                w.sorted_velocity[:n] = result
            mark("ist")

        # Integrate (pass max_displacement for grid reuse tracking)
        integrate.integrate(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_veleval[:n],
            w.sorted_sph_force[:n],
            w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            w.sorted_health[:n],
            sorted_lifetime=w.sorted_lifetime[:n],
            sorted_density=w.sorted_density[:n],
            sorted_shear_rate=w.sorted_shear_rate[:n],
            sorted_dTdt=w.sorted_dTdt[:n],
            sorted_sleep_counter=w.sorted_sleep_counter[:n],
            sorted_vorticity=w.sorted_vorticity[:n],
            sorted_angular_velocity=w.sorted_angular_velocity[:n],
            sort_indexes=self._sort_perm[:n],
            position_out=w.position,
            velocity_out=w.velocity,
            packed_info_out=w.packed_info,
            sleep_counter_out=w.sleep_counter,
            temperature_out=w.temperature,
            angular_velocity_out=w.angular_velocity,
            health_out=w.health,
            lifetime_out=w.lifetime,
            mass_out=w.mass,
            max_displacement=w.max_displacement,
            displacement_anchor=w.sorted_anchor,
            # No cell_start/cell_end: the STATIC push-out stays off in WCSPH.
            # It never ran before c_grid reached this module (it scanned hash
            # bucket 0 only), and turning it on costs about 0.16 ms per
            # substep at 1M. Step2's Akinci STATIC term handles walls.
            # Enabling it is a separate physics decision (bd-r5-review-n28.166).
            cell_wake_flags=self._cell_wake_flags,
            frame_counter=self._frame_counter_d,
        )
        mark("integrate")

        # Wake propagation: integrate wrote the own-cell stamps, so only the
        # rigid marks and the sleeper probe run (per-substep stamp, no memset)
        wake.run_wake_propagation(
            w.position[:n],
            w.velocity[:n],
            w.packed_info[:n],
            w.sleep_counter[:n],
            self._cell_wake_flags,
            num_particles=n,
            frame_counter=self._frame_counter_d,
            mark=False,  # K_Integrate wrote the own-cell stamps
            **self._wake_rigid_kwargs(),
        )
        mark("wake")

    def _wake_rigid_kwargs(self) -> dict:
        """Rigid bodies for run_wake_propagation (bd-r5-review-n28.87).

        Moving bodies wake the sleeping particles near them, so a body
        dropped on a sleeping pile gets the pile's support. Host-side count,
        no device sync.
        """
        rbm = self.rigid_body_manager
        if rbm.num_bodies <= 0:
            return {}
        return {"rigid_bodies": rbm.d_rigid_bodies, "num_bodies": rbm.num_bodies}

    def _rigid_kwargs(self) -> dict:
        """Rigid-body buffers for solver kernels that add the fluid-body reaction.

        Empty when there are no bodies, so the kernels get NULL pointers.
        """
        rbm = self.rigid_body_manager
        if rbm.num_bodies <= 0:
            return {}
        return {
            "d_rigid_bodies": rbm.d_rigid_bodies,
            "d_rigid_forces": rbm.rigid_forces,
            "d_rigid_torques": rbm.rigid_torques,
        }

    def _run_pbf_body(self, n: int, mark: StageMark = _no_mark) -> None:
        """PBF pipeline: Grid -> Predict -> [Lambda -> Delta]xN -> Finalize.

        Predicted positions ping-pong between sorted_predicted_position (A)
        and sorted_delta_position (B): iteration k reads one and K_PBF_ComputeDelta
        writes the corrected position into the other, so after N iterations
        the result is in A for even N and B for odd N. N is fixed per graph
        capture. .w of a predicted position is the particle mass
        (bd-r5-review-n28.185).

        mark(label) is called after each stage (PBF_STAGES); see _run_graph_body.
        """
        import pbf_solver
        w = self.world
        num_iters = int(self._profile.pbf_iterations)
        if num_iters < 1:
            raise ValueError(f"pbf_iterations must be >= 1, got {num_iters}")
        pred_bufs = (w.sorted_predicted_position, w.sorted_delta_position)

        # Grid setup
        self._run_grid_setup(n)
        mark("sort")

        # 1. Predict positions (gravity integration)
        pbf_solver.pbf_predict(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_predicted_position,
            mass=w.sorted_mass[:n],
        )
        mark("predict")

        # Snapshot previous-substep density into a dedicated read buffer so density_in
        # is not aliased with the density_out target (sorted_density); the kernel
        # value-guards zero entries to 1000 (bd-r4fix-uup.14).
        cupy.copyto(w.sorted_density_prev[:n], w.sorted_density[:n])

        # 2. Initial density + lambda + heat diffusion + exposure (first call only).
        # Mass comes from predicted.w (K_PBF_Predict wrote it). The pressure
        # normal is written only by the last lambda call, the one Finalize reads.
        pbf_solver.pbf_compute_lambda(
            w.sorted_predicted_position[:n], None,
            w.sorted_packed_info[:n],
            self._cell_start, self._cell_end,
            w.sorted_density, w.sorted_lambda_pbf,
            w.sorted_pressure_normal if num_iters == 1 else None,
            temperature_in=w.sorted_temperature[:n],
            density_in=w.sorted_density_prev[:n],
            dTdt_out=w.sorted_dTdt,
            exposure_heat_out=w.sorted_exposure_heat,
            exposure_corrode_out=w.sorted_exposure_corrode,
            velocity_in=w.sorted_velocity[:n],
            vorticity_out=w.sorted_vorticity,
            normal_out=w.sorted_normal,
        )
        mark("lambda0")

        # 3. Reactions + Spawn (spawn writes the steam children's predictions)
        self._run_reactions_spawn(
            n, mark, predicted_position=w.sorted_predicted_position,
        )

        # 4. Solver iterations (fixed count for CUDA graph).
        # Each iteration's delta moves the fluid, so each adds its rigid reaction.
        # Iteration 0 reads mass from sorted_mass: reactions and spawn may have
        # changed it since K_PBF_Predict, and that pass writes it into .w for
        # the later passes.
        rbm = self.rigid_body_manager
        rigid_kw = self._rigid_kwargs()
        for i in range(num_iters):
            src = pred_bufs[i % 2]
            dst = pred_bufs[(i + 1) % 2]
            last = i == num_iters - 1
            if i > 0:
                pbf_solver.pbf_compute_lambda(
                    src[:n], None,
                    w.sorted_packed_info[:n],
                    self._cell_start, self._cell_end,
                    w.sorted_density, w.sorted_lambda_pbf,
                    w.sorted_pressure_normal if last else None,
                )

            pbf_solver.pbf_compute_delta(
                src[:n], w.sorted_lambda_pbf[:n],
                w.sorted_mass[:n] if i == 0 else None,
                w.sorted_packed_info[:n],
                self._cell_start, self._cell_end,
                dst,
                **rigid_kw,
            )
        mark("iterations")
        pred_final = pred_bufs[num_iters % 2]

        # 5. Finalize: velocity update, XSPH, friction, sleep, writeback
        pbf_solver.pbf_finalize(
            pred_final[:n], w.sorted_position[:n],
            w.sorted_velocity[:n], w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_dTdt[:n], w.sorted_sleep_counter[:n],
            self._sort_perm[:n], self._cell_start, self._cell_end,
            w.position, w.velocity, w.packed_info,
            w.sleep_counter, w.temperature,
            sorted_angular_velocity=w.sorted_angular_velocity[:n],
            angular_velocity_out=w.angular_velocity,
            vorticity_in=w.sorted_vorticity[:n],
            normal_in=w.sorted_normal[:n],
            pressure_normal_in=w.sorted_pressure_normal[:n],
            d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
            d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
            d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            sorted_lifetime=w.sorted_lifetime[:n],
            health_out=w.health, lifetime_out=w.lifetime, mass_out=w.mass,
            max_displacement=w.max_displacement,
            displacement_anchor=w.sorted_anchor,
            cell_wake_flags=self._cell_wake_flags,
            frame_counter=self._frame_counter_d,
            mass_in_pred_w=True,
        )
        mark("finalize")

        # Wake propagation: without this, sleeping GRANULAR piles under PBF never
        # wake from a disturbance (bd-r4fix-uup.15).  Operates on the unsorted
        # arrays the finalize just scattered back.
        wake.run_wake_propagation(
            w.position[:n],
            w.velocity[:n],
            w.packed_info[:n],
            w.sleep_counter[:n],
            self._cell_wake_flags,
            num_particles=n,
            frame_counter=self._frame_counter_d,
            mark=False,  # finalize wrote the own-cell stamps
            **self._wake_rigid_kwargs(),
        )
        mark("wake")

    def _ensure_kappa_v_delta(self) -> None:
        """Allocate the DFSPH per-iteration kappa_v increment buffer.

        Called outside the graph body so a capture never allocates. Host-only
        shape check, no device sync (bd-r5-review-n28.83).
        """
        n = self.world.max_particles
        buf = getattr(self, '_kappa_v_delta', None)
        if buf is None or buf.shape[0] != n:
            self._kappa_v_delta = cupy.zeros(n, dtype=cupy.float32)

    def _run_dfsph_body(self, n: int, mark: StageMark = _no_mark) -> None:
        """DFSPH pipeline: Grid -> DensityAlpha -> NonPressure -> DivSolve -> PredictPos -> DensAdv -> DensSolve -> Finalize.

        mark(label) is called after each stage (DFSPH_STAGES); see _run_graph_body.
        """
        import dfsph_solver
        w = self.world

        # Grid setup
        self._run_grid_setup(n)
        mark("sort")

        # 1. Density + alpha precompute (also computes shear_rate for GRANULAR)
        # Snapshot previous-substep density into a dedicated read buffer so density_in
        # is not the same buffer as the density_out target (sorted_density).  The old
        # in-place aliasing was a __restrict__ read/write race; the kernel value-guards
        # any zero entry (frame 1 / fresh spawns) to 1000 (bd-r4fix-uup.14).
        cupy.copyto(w.sorted_density_prev[:n], w.sorted_density[:n])
        dfsph_solver.compute_density_alpha(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_mass[:n],
            w.sorted_density_prev[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            self._cell_start, self._cell_end,
            w.sorted_density, w.sorted_alpha_dfsph,
            w.sorted_shear_rate, w.sorted_dTdt,
            w.sorted_exposure_heat, w.sorted_exposure_corrode,
            vorticity_out=w.sorted_vorticity,
            normal_out=w.sorted_normal,
        )
        mark("density_alpha")

        # 2. Reactions + Spawn
        self._run_reactions_spawn(n, mark)

        # Volume carrier into sorted_position.w for the four solver loops
        # (bd-r5-review-n28.184). After reactions and spawn, which change mass
        # and material. The next reorder overwrites .w, so it lives only
        # inside this substep.
        dfsph_solver.pack_volume(
            w.sorted_position[:n], w.sorted_density[:n],
            w.sorted_mass[:n], w.sorted_packed_info[:n],
        )

        # 3. Non-pressure forces (viscosity, gravity, mu(I), vorticity confinement, surface tension)
        rbm = self.rigid_body_manager
        vel = w.sorted_veleval[:n]
        dfsph_solver.compute_non_pressure_forces(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_shear_rate[:n],
            w.sorted_temperature[:n],
            self._cell_start, self._cell_end,
            # Write to veleval, not in-place: velocity is a __restrict__
            # neighbour input here, so aliasing it with the output is UB and a
            # read/write race (MED-3). K_DFSPH_NonPressureForces writes every
            # particle in [0,n), so from here on sorted_veleval is the DFSPH
            # velocity: every later kernel in this substep, and foam
            # generation, reads it. The routing is fixed (no per-substep
            # buffer swap) because CUDA graphs bake the pointers
            # (bd-r5-review-n28.173).
            vel,
            vorticity_in=w.sorted_vorticity[:n],
            normal_in=w.sorted_normal[:n],
            d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
            d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
            d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            # Last substep's converged p/rho, stored by K_DFSPH_Finalize times
            # the warm-start carry; the kernel divides the carry back out.
            # mu(I) friction reads it as its confining pressure
            # (bd-r5-review-n28.92).
            p_prev=w.sorted_kappa[:n],
        )
        mark("non_pressure")

        # 4. Divergence solver (warm-started kappa_v -- PERF-008)
        # Every correction pass (warm start included) is applied to the fluid,
        # so each one adds its rigid reaction.
        rigid_kw = self._rigid_kwargs()
        # sorted_kappa_v is the running total: the scaled warm start from the
        # previous substep plus every iteration's increment. Finalize writes it
        # back times the div warm start, so it arrives here already scaled
        # (bd-r5-review-n28.173). Each iteration applies only its own
        # increment, held in _kappa_v_delta (bd-r5-review-n28.83).
        self._ensure_kappa_v_delta()  # no-op here: _sim_step allocated it
        kv_total = w.sorted_kappa_v[:n]
        kv_delta = self._kappa_v_delta[:n]
        if self._profile.dfsph_div_warm_start > 0.0:
            dfsph_solver.correct_velocity_div(
                vel, w.sorted_density[:n],
                w.sorted_mass[:n], kv_total,
                w.sorted_packed_info[:n], w.sorted_position[:n],
                self._cell_start, self._cell_end,
                **rigid_kw,
            )
        else:
            # Finalize already wrote 0 with a zero carry. This clears values
            # carried from before a switch to a zero warm start.
            kv_total.fill(0)
        for _ in range(self._profile.dfsph_div_iters):
            dfsph_solver.compute_kappa_v(
                vel, w.sorted_alpha_dfsph[:n],
                w.sorted_packed_info[:n], w.sorted_position[:n],
                self._cell_start, self._cell_end,
                kv_delta,
                # .w neighbour count gates the solve below 20 neighbours
                # (bd-r5-review-n28.71).
                normal_in=w.sorted_normal[:n],
                kappa_v_total=kv_total,
            )
            dfsph_solver.correct_velocity_div(
                vel, w.sorted_density[:n],
                w.sorted_mass[:n], kv_delta,
                w.sorted_packed_info[:n], w.sorted_position[:n],
                self._cell_start, self._cell_end,
                **rigid_kw,
            )

        mark("divergence")

        # kappa_v writeback now done in K_DFSPH_Finalize (PERF-008)

        # 5. Density solver (SPlisHSPlasH-style Jacobi iteration on pressure)
        # Key insight: iterate on PRESSURE variable p/rho^2, not velocity.
        # Each iteration:
        #   a) DensitySolverUpdate: predict density from vt = v + dt*a_press,
        #      compute residual, update p_rho2 via Jacobi step
        #   b) ComputePressureAccel: a_press from the updated p_rho2, written
        #      as the clamped predicted velocity vt (bd-r5-review-n28.184)
        # The last ComputePressureAccel writes vt into vel itself, which
        # applies the converged pressure; there is no separate apply pass.
        # Buffer reuse: sorted_kappa -> p_rho2 (warm-started from prev frame),
        #               sorted_predicted_position -> vt
        p_rho2 = w.sorted_kappa  # already contains warm-started values from reorder
        vt = w.sorted_predicted_position
        # Warm start: K_DFSPH_Finalize already scaled the previous pressure by
        # dfsph_warm_start (0.5 = carry 50%) as it wrote it back
        # (bd-r5-review-n28.173). With a zero warm start it keeps the
        # unscaled value for mu(I), so start the solver from zero here.
        if self._profile.dfsph_warm_start <= 0.0:
            p_rho2[:n].fill(0)
        # Only the last a_press is applied to the fluid, so only the last
        # compute_pressure_accel call adds the rigid reaction.
        dens_iters = self._profile.dfsph_dens_iters
        # Initial vt from the warm-started p_rho2
        dfsph_solver.compute_pressure_accel(
            p_rho2[:n], vel, w.sorted_position[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            self._cell_start, self._cell_end,
            vel if dens_iters == 0 else vt,
            **(rigid_kw if dens_iters == 0 else {}),
        )
        for it in range(dens_iters):
            last = it == dens_iters - 1
            dfsph_solver.density_solver_update(
                vt[:n], w.sorted_position[:n], w.sorted_density[:n],
                w.sorted_alpha_dfsph[:n], w.sorted_packed_info[:n],
                self._cell_start, self._cell_end,
                p_rho2,
            )
            dfsph_solver.compute_pressure_accel(
                p_rho2[:n], vel, w.sorted_position[:n],
                w.sorted_density[:n], w.sorted_mass[:n],
                w.sorted_packed_info[:n],
                self._cell_start, self._cell_end,
                vel if last else vt,
                **(rigid_kw if last else {}),
            )
        mark("density_solve")

        # 8. Finalize + writeback (includes sleep logic via velocity magnitude)
        # Pass sorted_kappa (= p_rho2) through to unsorted kappa for warm-start next frame
        dfsph_solver.finalize(
            w.sorted_position[:n], vel,
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_dTdt[:n],
            w.sorted_sleep_counter[:n],
            p_rho2[:n],  # sorted_kappa with current pressure values
            self._sort_perm[:n], self._cell_start, self._cell_end,
            w.position, w.velocity, w.packed_info,
            w.sleep_counter, w.temperature,
            w.kappa,  # unsorted kappa out for warm-start
            sorted_angular_velocity=w.sorted_angular_velocity[:n],
            angular_velocity_out=w.angular_velocity,
            vorticity_in=w.sorted_vorticity[:n],
            sorted_kappa_v=w.sorted_kappa_v[:n],
            kappa_v_out=w.kappa_v,
            d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
            d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
            d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            sorted_lifetime=w.sorted_lifetime[:n],
            health_out=w.health, lifetime_out=w.lifetime, mass_out=w.mass,
            max_displacement=w.max_displacement,
            displacement_anchor=w.sorted_anchor,
            cell_wake_flags=self._cell_wake_flags,
            frame_counter=self._frame_counter_d,
        )
        mark("finalize")

        # Wake propagation: sleeping GRANULAR piles under DFSPH otherwise never
        # wake from a disturbance (bd-r4fix-uup.15).  Unsorted arrays from finalize.
        wake.run_wake_propagation(
            w.position[:n],
            w.velocity[:n],
            w.packed_info[:n],
            w.sleep_counter[:n],
            self._cell_wake_flags,
            num_particles=n,
            frame_counter=self._frame_counter_d,
            mark=False,  # finalize wrote the own-cell stamps
            **self._wake_rigid_kwargs(),
        )
        mark("wake")

    def _apply_damping_ramp(self) -> None:
        """Apply spawn velocity-damping ramp (shared by _sim_step and _sim_step_timed).

        Increments _spawn_substep and updates velocity_damping in c_sim with
        an async stream-ordered copy.  Must be called once per substep before
        any kernel launches that read c_sim.velocity_damping.  CUDA graphs
        stay valid during the ramp: their kernels read c_sim when they run,
        and the copy is queued ahead of the graph launch on the same stream.
        """
        if self._spawn_substep < self._damping_duration:
            t = self._spawn_substep / self._damping_duration
            damping = 0.8 * (1.0 - t)  # linear ramp 0.8 -> 0.0
            self._spawn_substep += 1
        else:
            damping = 0.0
        # Upload damping to c_sim.velocity_damping only when it actually changed.
        # This guards against the post-ramp steady state (damping == 0) triggering
        # repeated redundant uploads each substep.  During the active ramp, damping
        # decrements each substep so the upload fires once per substep — necessary
        # because constant memory must match what the GPU kernels will use.
        if self._sim_params is not None:
            old_damping = float(self._sim_params[0]["velocity_damping"])
            if abs(damping - old_damping) > 1e-8:
                self._sim_params[0]["velocity_damping"] = np.float32(damping)
                self._upload_sim_params_all()  # only uploads when value changed

    def _thermal_rule(self) -> bool:
        """Return the value of c_sim.thermal_enabled for the current scene.

        False only when at least one material was spawned, every one of them
        starts at T_AMBIENT, and no pair of them (a material with itself
        included) has a non-zero reaction_rate or heat_exchange. Then every temperature stays
        at T_AMBIENT, the heat and exposure sums in K_Step1 are exactly 0, and
        no reaction can create a new material. Host-only, no GPU sync.
        """
        mats = getattr(self.world, "_spawned_material_ids", None)
        if mats is None or mats is not self._thermal_trusted_mats:
            return True
        present = [m for m in mats if 0 < m < _PAIR_INTERACTS.shape[0]]
        # An empty world stays on: a caller may spawn and launch K_Step1
        # directly before the next substep re-evaluates the rule.
        if not present or any(m in _NON_AMBIENT_MATERIALS for m in present):
            return True
        return bool(_PAIR_INTERACTS[np.ix_(present, present)].any())

    def _update_thermal_gate(self) -> None:
        """Re-evaluate the thermal gate when the spawned material set changed.

        Called once per substep before any kernel launch. The set only grows
        or is replaced, so its identity and size are enough to detect a
        change. On a change of the flag, c_sim is re-uploaded with the same
        async copy as the damping ramp, so captured graphs stay valid.
        """
        mats = getattr(self.world, "_spawned_material_ids", None)
        key = (id(mats), len(mats) if mats is not None else 0)
        if key == self._thermal_key or self._sim_params is None:
            return
        self._thermal_key = key
        on = 1 if self._thermal_rule() else 0
        if int(self._sim_params[0]["thermal_enabled"]) != on:
            self._sim_params[0]["thermal_enabled"] = on
            self._upload_sim_params_all()

    def _invalidate_graphs(self) -> None:
        """Invalidate both CUDA graphs (call when constants/solver/grid changes)."""
        self._graph_full_sort = None
        self._graph_gather_only = None
        self._graph_full_sort_n = 0
        self._graph_gather_only_n = 0

    def _capture_graph(self, n: int, gather_only: bool) -> cupy.cuda.Graph | None:
        """Capture the graph-body into a CUDA graph for replay.

        Creates a non-default capture stream, records all ops, and returns
        the resulting Graph object for repeated launch().

        A gather-only graph is only valid when the last full sort was for n.
        If it was not (for example after a same-n notify_scene_changed), the
        body would take the forced full-sort branch and every later replay of
        the "gather" graph would run a full counting sort. In that case the
        body runs directly, nothing is captured, and None is returned; the
        next gather substep captures instead (bd-r5-review-n28.84).
        """
        if gather_only and self._last_full_sort_n != n:
            self._run_graph_body(n)
            return None
        s = cupy.cuda.Stream(non_blocking=True)
        with s:
            s.begin_capture()
            self._run_graph_body(n)
            graph = s.end_capture()
        if gather_only:
            assert not self._last_setup_was_full_sort, (
                "gather-only graph captured a full sort")
            self._graph_gather_only = graph
            self._graph_gather_only_n = n
        else:
            self._graph_full_sort = graph
            self._graph_full_sort_n = n
        return graph

    def _sim_step(self, n: int, last_substep: bool = True) -> None:
        """Run one simulation substep on n particles.

        last_substep: step_frame passes True only for the frame's last
        substep, so foam generation runs once per frame. Direct callers keep
        the default and generate every substep.

        Pipeline (fully graph-capturable):
          1. Counting sort (or gather-only if sort skipped)
          2. Step1/Step2/Integrate/Wake (solver-specific)

        Two-graph strategy: separate graphs for full-sort and gather-only paths
        so sort-skip decisions don't invalidate graphs. When n changes, fall back
        to direct kernel launches (no graph capture overhead).
        """
        n = self._substep_prologue(n)
        self._count_substep_sort(n)
        # compact() swapped the world's arrays: captured graphs point at the
        # old buffers, even when n came back to the same value.
        gen = getattr(self.world, "buffer_generation", 0)
        if self._graph_buffer_gen != gen:
            self._invalidate_graphs()
            self._graph_buffer_gen = gen
        foam_on = bool(self.world.foam_enabled)
        if self._graph_foam_enabled != foam_on:
            self._invalidate_graphs()
            self._graph_foam_enabled = foam_on
        # Rigid bodies stay in the graph (bd-r5-review-n28.182): the prologue
        # zeroes the accumulators and moves the boundary block, and the
        # epilogue integrates the bodies, both outside the graph. The body
        # buffers are allocated once, so only the state the body bakes in
        # needs a new capture when it changes.
        rigid_key = self._rigid_graph_key()
        if self._graph_rigid_key != rigid_key:
            self._invalidate_graphs()
            self._graph_rigid_key = rigid_key

        # Select graph based on sort-skip state
        is_gather = self._sort_skip_next
        if is_gather:
            graph = self._graph_gather_only
            graph_n = self._graph_gather_only_n
        else:
            graph = self._graph_full_sort
            graph_n = self._graph_full_sort_n

        if graph_n != n:
            # n changed — invalidate this path's graph, run directly
            if is_gather:
                self._graph_gather_only = None
                self._graph_gather_only_n = n
            else:
                self._graph_full_sort = None
                self._graph_full_sort_n = n
            self._run_graph_body(n)
        elif graph is None:
            # n stable but no graph yet — capture then launch
            g = self._capture_graph(n, gather_only=is_gather)
            if g is not None:
                g.launch()
        else:
            # n stable, graph exists — fast replay
            graph.launch()
            # The replay skips the host code in _run_grid_setup that consumes
            # this flag; the captured branch already matches it.
            self._disp_cleared_for_sort = False
            if not is_gather:
                # The full-sort graph sorts for n, but the host-side
                # bookkeeping in _run_grid_setup does not run on replay.
                # Record it here so a same-n notify_scene_changed (which sets
                # -1) cannot leave the value stale (bd-r5-review-n28.84).
                self._last_full_sort_n = n

        self._substep_epilogue(n, last_substep=last_substep)

    def _rigid_graph_key(self) -> tuple[int, bool]:
        """Rigid body state that a captured graph body bakes in.

        The body count selects NULL or the body buffers in every solver and is
        a PBF/DFSPH kernel argument. The flag says whether _update_rigid_boundary
        clears max_displacement on a full sort, in which case the captured
        _run_grid_setup skips its own memset. Call after _substep_prologue.
        """
        rbm = self.rigid_body_manager
        nb = rbm.num_bodies
        clears = (nb > 0 and rbm.num_boundary_particles > 0
                  and rbm.boundary_offset >= 0)
        return (nb, clears)

    def _count_substep_sort(self, n: int) -> None:
        """Count this substep for the grid-reuse drift prediction.

        Mirrors the branch _run_grid_setup takes: a full sort unless the skip
        flag is set and n still matches the last full sort. Host-only, so it
        also holds for graph replays.
        """
        full_sort = self._next_setup_is_full_sort(n)
        self._substeps_since_sort = 1 if full_sort else self._substeps_since_sort + 1
        self._sim_time_since_sort = (self.dt if full_sort
                                     else self._sim_time_since_sort + self.dt)
        self._frame_substeps += 1

    def _substep_prologue(self, n: int) -> int:
        """Per-substep host-side setup shared by _sim_step and _sim_step_timed.

        Applies the spawn damping ramp, moves
        rigid-body boundary particles (may change n), zeroes the rigid force
        accumulators and advances kinematic SDF objects.  Returns the new n.
        """
        # --- Spawn velocity damping ramp ---
        self._apply_damping_ramp()
        self._update_thermal_gate()

        # IST scratch is allocated here, outside the graph body, so a graph
        # capture never allocates (bd-r5-review-n28.56).
        if self.ist_enabled:
            self.world.ensure_ist_scratch(self._IST_MAX_PARTICLES)
        if self._profile.solver_type == SolverType.DFSPH:
            self._ensure_kappa_v_delta()

        # --- Rigid body: update boundary particles in unsorted arrays ---
        rbm = self.rigid_body_manager
        if rbm.num_bodies > 0 and rbm.num_boundary_particles > 0:
            n = self._update_rigid_boundary(n)

        # Zero rigid body force/torque accumulators before solver
        if rbm.num_bodies > 0:
            rbm.rigid_forces.fill(0)
            rbm.rigid_torques.fill(0)

        # --- Update kinematic SDF objects ---
        if self.sdf_manager._motions:
            self.sdf_manager.update_kinematics(self.sim_time, self.dt)
            self.sdf_manager.upload_if_dirty()
        return n

    def _substep_epilogue(self, n: int, mark: StageMark = _no_mark,
                          last_substep: bool = True) -> None:
        """Per-substep work after the solver body (outside any CUDA graph).

        Rigid-body integration and collisions, foam, then advance sim time.
        """
        # Rigid body integration (outside graph -- forces vary per substep)
        rbm = self.rigid_body_manager
        # boundary_dropped: the block did not fit, so the fluid cannot push
        # the bodies. Freeze them instead of letting gravity drop them through
        # the fluid (bd-r5-review-n28.58). Host-side flag, no device sync.
        if rbm.num_bodies > 0 and not rbm.boundary_dropped:
            integrate.integrate_rigid_bodies(
                rbm.d_rigid_bodies,
                rbm.rigid_forces,
                rbm.rigid_torques,
                rbm.num_bodies,
                self.dt,
                gravity=tuple(float(x) for x in self._sim_params[0]["gravity"]),
            )
            # US-020: Rigid body collision push-apart (body-vs-SDF and body-vs-body)
            integrate.rigid_body_collisions(rbm.d_rigid_bodies, rbm.num_bodies)
            mark("rigid")

        # Foam generation, once per frame (outside the graph: dt_emit varies
        # with the frame's substep count). Foam physics is in the graph body.
        self._foam_emit_dt += self.dt
        self._run_foam_step(n, generate=last_substep)
        mark("foam")

        self.sim_time += self.dt
        self._substep_counter += 1

    def _run_foam_step(self, n: int, generate: bool = True) -> None:
        """Run foam generation when asked (outside CUDA graph).

        Foam physics runs every substep at the end of the graph body
        (_run_graph_body). Generation runs when generate is True (once per
        frame from step_frame) and covers the sim time since the last
        generation, so the emission rate does not depend on the substep
        count. Compaction is in _maybe_compact_foam, once per frame.

        No GPU->CPU sync: the kernel reads foam_count and the RNG seed
        (_frame_counter_d) from device memory.
        """
        w = self.world
        if not w.foam_enabled:
            self._foam_emit_dt = 0.0
            return
        if not generate:
            return

        # Generate new foam from FLUID particles. DFSPH keeps its substep
        # velocity in sorted_veleval (bd-r5-review-n28.173).
        foam_vel = (w.sorted_veleval if self._profile.solver_type == SolverType.DFSPH
                    else w.sorted_velocity)
        foam.foam_generate(
            w.sorted_position[:n],
            foam_vel[:n],
            w.sorted_normal[:n],
            w.sorted_packed_info[:n],
            w.foam_position,
            w.foam_velocity,
            w.foam_count,
            n,
            # Per-substep RNG seed on the device; _frame_counter repeats
            # across a frame's substeps (bd-r4fix-uup.15).
            self._frame_counter_d,
            self._foam_emit_dt,
        )
        self._foam_emit_dt = 0.0

    def _maybe_compact_foam(self) -> None:
        """Compact dead foam once per frame when due (outside CUDA graph).

        Runs every 8th frame, and every frame while the deferred foam count is
        above 80% of the pool. step_frame calls it once after the substep loop.
        No GPU->CPU sync: the new count is a device-to-device copy.

        The result is copied back into foam_position/foam_velocity instead
        of swapping the buffers: the captured graph bodies launch
        K_FoamPhysics on those arrays, and a swap would leave the graphs
        pointing at the scratch pair. The copy is 2 x max_foam x 16 B
        (6.4 MB at 200K) once per 8 frames.
        """
        w = self.world
        if not w.foam_enabled:
            return
        max_foam = w._max_foam
        if not (self._frame_counter % 8 == 0
                or self._last_foam_count > 0.8 * max_foam):
            return
        foam.foam_compact(
            w.foam_position, w.foam_velocity,
            w.foam_position_b, w.foam_velocity_b,
            w.foam_alive_count,
            w.foam_count,
            max_foam,
        )
        # Copy back (device-to-device). The whole pool, so the dead tail
        # K_FoamClearTail marked comes along.
        cupy.copyto(w.foam_position, w.foam_position_b)
        cupy.copyto(w.foam_velocity, w.foam_velocity_b)
        # Update count from compaction result (device-to-device copy, no sync)
        w.foam_count[:] = w.foam_alive_count

    def _sim_step_timed(self, n: int, last_substep: bool = True) -> None:
        """Run one substep with CUDA event timing (bypasses graph capture).

        Runs the same prologue, solver body and epilogue as _sim_step, with a
        mark callback that records a CUDA event after each stage.  Only the
        timed path passes a real mark; graph capture always uses the no-op.
        """
        events = []
        labels = []

        def mark(label):
            e = cupy.cuda.Event()
            e.record()
            events.append(e)
            labels.append(label)

        mark("start")
        n = self._substep_prologue(n)
        self._count_substep_sort(n)
        self._run_graph_body(n, mark)
        self._substep_epilogue(n, mark, last_substep=last_substep)
        self._read_deferred_timings()
        # Store this substep's events for reading next substep
        self._pending_timing_events = events
        self._pending_timing_labels = labels

    def _read_deferred_timings(self) -> None:
        """Turn the previous timed substep's events into stage timings (ms)."""
        # --- Deferred timing read (1-substep latency, no per-substep sync stall) ---
        # Read timings from the PREVIOUS substep's events; by the time we get here
        # those GPU kernels are almost certainly already complete (a full substep
        # of GPU work has elapsed since they were enqueued), so this is effectively
        # a non-blocking query rather than a stall.  Displayed values lag by one
        # substep, which is imperceptible to the user.
        prev_events = self._pending_timing_events
        prev_labels = self._pending_timing_labels
        if prev_events and len(prev_events) > 1:
            try:
                # Synchronise only on the PREVIOUS substep's last event.
                # This event was recorded a full substep ago, so the wait is
                # typically 0 ms (GPU already past it).
                prev_events[-1].synchronize()
                raw = {}
                for i in range(1, len(prev_events)):
                    raw[prev_labels[i]] = cupy.cuda.get_elapsed_time(
                        prev_events[i - 1], prev_events[i]
                    )
                self._last_timings = raw
                # Update EMA
                alpha = self._timing_ema_alpha
                for k, v in raw.items():
                    if k in self._timing_ema:
                        self._timing_ema[k] = alpha * v + (1 - alpha) * self._timing_ema[k]
                    else:
                        self._timing_ema[k] = v
            except Exception:
                pass  # timing is advisory — never crash the sim loop

    def _latch_v_max(self) -> None:
        """Update _v_max_lagged from the just-latched drift (host only).

        v_max = sqrt(max drift^2 since the last full sort) / sim time since
        that sort. While the scene moves, every substep sorts, so this is
        the largest one-substep displacement / dt. While sorts are skipped
        it is the fastest average speed over the skip window. The +inf
        sentinel (K_SpawnGas teleport, unknown) keeps the previous value.
        """
        import struct
        bits = self._last_max_disp_uint
        if bits >= 0x7F800000:
            return
        t = self._last_disp_sim_time
        if bits == 0 or t <= 0.0:
            self._v_max_lagged = 0.0
            return
        disp_sq = struct.unpack('f', struct.pack('I', bits))[0]
        self._v_max_lagged = math.sqrt(max(disp_sq, 0.0)) / t

    def _update_sort_skip_decision(self) -> None:
        """Decide whether the next substeps may skip the full sort (grid reuse).

        Called once per rendered frame that ran substeps, after the substeps.
        Order (bd-r5-review-n28.46):
          1. Latch the pinned values if a copy launched since the last
             notify_scene_changed() has landed (non-blocking event queries,
             no host sync). When the host keeps pace with the GPU, the
             landed copy is the one from the end of the previous frame, so
             the decision at the end of frame k uses the displacement at the
             end of frame k-1. When the host runs ahead, it is the newest
             copy that has landed. If none has, the previous value is kept.
          2. Decide from the latched displacement.
          3. Launch this frame's D2H copies and queue an event after them.
        bench_substep.py --grid-reuse calls this too.
        """
        import struct
        import ctypes as _ctypes

        # --- 1. Latch the newest landed copy from the current scene ---
        landed_epoch = None
        landed_substeps = 0
        landed_sim_time = 0.0
        while self._readback_pending and self._readback_pending[0][0].done:
            ev, landed_epoch, landed_substeps, landed_sim_time = (
                self._readback_pending.popleft())
            self._readback_free_events.append(ev)
        if landed_epoch is not None:
            if self._readback_pending:
                self._readback_not_ready += 1
            else:
                self._readback_latched += 1
            self._last_foam_count = _ctypes.c_uint32.from_address(
                self._pinned_foam_count.ptr).value
            if landed_epoch == self._readback_epoch:
                self._last_max_disp_uint = _ctypes.c_uint32.from_address(
                    self._pinned_displacement.ptr).value
                self._last_disp_substeps = landed_substeps
                self._last_disp_sim_time = landed_sim_time
                self._last_dead_count = _ctypes.c_uint32.from_address(
                    self._pinned_dead_count.ptr).value
                self._latch_v_max()
        elif self._readback_pending:
            self._readback_not_ready += 1

        # --- 2. Grid reuse decision for the next frame ---
        # max_displacement holds the float bits of the largest squared drift
        # |x - x_anchor|^2 since the last full sort (atomicMax in the finalize
        # kernels against sorted_anchor, bd-r5-review-n28.54).
        # Positive float bits order like uints; anything at or above the +inf
        # bits (K_SpawnGas teleport, or the unknown sentinel) never skips.
        max_disp_sq_uint = self._last_max_disp_uint
        if max_disp_sq_uint >= 0x7F800000:
            max_disp_sq = float('inf')
        elif max_disp_sq_uint > 0:
            max_disp_sq = struct.unpack('f', struct.pack('I', max_disp_sq_uint))[0]
        else:
            max_disp_sq = 0.0
        if self._profile.solver_type == SolverType.WCSPH:
            threshold = self._sort_skip_threshold_sq
            max_consec = self._MAX_SORT_SKIP_FRAMES
        else:
            threshold = self._sort_skip_threshold_sq_tight
            max_consec = self._MAX_SORT_SKIP_FRAMES_TIGHT
        # The latched drift is up to two frames old: it was copied at the end
        # of the previous frame, and the decision covers the next frame. Turn
        # it into a drift rate per substep (drift / substeps since the sort
        # when it was copied) and predict the drift at the end of the next
        # frame: substeps since the last sort now, plus one more frame of the
        # size just run. A settled scene keeps a near-zero rate and skips; a
        # moving front sorts every frame.
        k_copy = max(self._last_disp_substeps, 1)
        k_next = max(self._substeps_since_sort + self._frame_substeps, 1)
        predicted_sq = max_disp_sq * (k_next / k_copy) ** 2
        self._frame_substeps = 0
        can_skip = (predicted_sq < threshold
                    and self._sort_skip_consecutive < max_consec)
        if can_skip:
            self._sort_skip_next = True
            self._sort_skip_consecutive += 1
        else:
            self._sort_skip_next = False
            self._sort_skip_consecutive = 0

        # --- 3. Launch this frame's copies for the next frame's latch ---
        # Record on the compute stream so the copies start after this frame's
        # kernels, then queue an event after the copies.
        self._readback_event.record()
        self._readback_stream.wait_event(self._readback_event)
        cupy.cuda.runtime.memcpyAsync(
            self._pinned_displacement.ptr,
            self.world.max_displacement.data.ptr,
            4, cupy.cuda.runtime.memcpyDeviceToHost,
            self._readback_stream.ptr,
        )
        cupy.cuda.runtime.memcpyAsync(
            self._pinned_foam_count.ptr,
            self.world.foam_count.data.ptr,
            4, cupy.cuda.runtime.memcpyDeviceToHost,
            self._readback_stream.ptr,
        )
        cupy.cuda.runtime.memcpyAsync(
            self._pinned_dead_count.ptr,
            self._dead_count.data.ptr,
            4, cupy.cuda.runtime.memcpyDeviceToHost,
            self._readback_stream.ptr,
        )
        ev = (self._readback_free_events.pop() if self._readback_free_events
              else cupy.cuda.Event(disable_timing=True))
        ev.record(self._readback_stream)
        self._readback_pending.append(
            (ev, self._readback_epoch, self._substeps_since_sort,
             self._sim_time_since_sort))
        self._readback_done = ev

    def step_frame(self) -> int:
        """Advance simulation for one render frame. Returns number of substeps run."""
        now = self._clock()
        if self._last_frame_time is None:
            self._last_frame_time = now
            self._time_accumulator = 0.0
            return 0

        wall_dt = now - self._last_frame_time
        self._last_frame_time = now

        if self._paused or self.speed <= 0.0:
            self._last_substeps = 0
            return 0

        n = self.world._high_water
        if n == 0:
            self._last_substeps = 0
            return 0

        # Pick this frame's dt from the lagged v_max (host only, no sync).
        if not self.fixed_dt:
            new_dt = self._next_adaptive_dt(n)
            delta = abs(new_dt - self.dt)
            if (delta > DT_UPLOAD_REL * self.dt
                    or (delta > 1e-12 and new_dt == self._adaptive_dt_cap())):
                self._upload_dt(new_dt)
        elif self._profile.solver_type != SolverType.WCSPH:
            subdiv = self._next_fixed_subdiv()
            if subdiv != self._fixed_subdiv:
                self._fixed_subdiv = subdiv
                self._upload_dt(self.fixed_dt_value / subdiv)

        # Accumulate sim time; fire substeps when accumulator >= dt.
        # This correctly handles high FPS with large fixed_dt (e.g. PBF/DFSPH at dt=1/60).
        self._time_accumulator += self.speed * wall_dt
        sim_steps = max(0, min(
            self.max_substeps,
            int(self._time_accumulator / self.dt),
        ))
        self._time_accumulator -= sim_steps * self.dt
        # Clamp accumulator to prevent runaway after long stalls
        self._time_accumulator = min(self._time_accumulator, self.dt * 2.0)

        step_fn = self._sim_step_timed if self.timing_enabled else self._sim_step
        if sim_steps > 0:
            # Device-side wait (no host sync): this frame's kernels, including
            # the full-sort memset of max_displacement and the foam kernels
            # that write foam_count, start only after the previous frame's
            # readback copies have finished (bd-r5-review-n28.46).
            cupy.cuda.get_current_stream().wait_event(self._readback_done)
        for i in range(sim_steps):
            step_fn(n, last_substep=(i == sim_steps - 1))

        # Foam compaction once per frame, before the readback copies foam_count
        if sim_steps > 0:
            self._maybe_compact_foam()

        # Increment frame counter once per render frame (I4 fix)
        if sim_steps > 0:
            self._frame_counter += 1

        if sim_steps > 0:
            self._update_sort_skip_decision()
        # else: sim_steps==0 — keep previous _sort_skip_next

        # Periodic compaction: remove dead particles from the active range
        if sim_steps > 0:
            n = self._maybe_compact()

        self._last_substeps = sim_steps
        return sim_steps

    def _maybe_compact(self) -> int:
        """Run compaction if due, and keep the parked DEAD reserve topped up.

        Runs every compact_interval frames, and on the first frame after
        notify_scene_changed(), so a freshly loaded scene gets its reserve
        before any water can boil. While a boilable material is present
        (World.needs_dead_reserve), compaction only runs once there are more
        than dead_reserve + compact_threshold dead slots, and afterwards the
        reserve is refilled with parked DEAD particles so K_SpawnGas always
        has slots to claim (bd-r5-review-n28.27).

        Returns the current _high_water (possibly changed).
        """
        w = self.world
        n = w._high_water
        pending = self._dead_reserve_check_pending
        self._dead_reserve_check_pending = False
        if n == 0:
            return n

        interval_due = (self.compact_interval > 0
                        and self._frame_counter % self.compact_interval == 0)
        if not (interval_due or pending):
            return n

        reserve = w.dead_reserve if w.needs_dead_reserve else 0
        if not interval_due and reserve == 0:
            return n

        # The lagged freelist size is the DEAD count one frame ago (host int
        # from the async readback, no sync). If it shows no compaction and no
        # reserve refill is due, skip the exact count. Deaths since that copy
        # (brush kills, the last frame's substeps) show up in the next copy.
        # After a scene change it is None, and the exact count below runs.
        lagged_dead = self._last_dead_count
        if lagged_dead is not None:
            if reserve > 0:
                exact_needed = (lagged_dead > reserve + self.compact_threshold
                                or lagged_dead < reserve)
            else:
                exact_needed = lagged_dead >= self.compact_threshold
            if not exact_needed:
                return n

        # Count dead particles = _high_water - num_active. This is the only
        # per-frame caller of num_active. It runs on an interval frame only
        # when the lagged count says compaction or a refill is due, or once
        # after a scene change.
        self._num_active_syncs += 1
        num_alive = w.num_active
        num_dead = n - num_alive
        changed = False

        if reserve > 0:
            # Keep the reserve: compacting it away would empty the freelist.
            compact_due = num_dead > reserve + self.compact_threshold
        else:
            compact_due = num_dead >= self.compact_threshold
        if interval_due and compact_due:
            # Reuse the count we already paid for so compact() does not sync again.
            w.compact(num_alive=num_alive)
            num_dead = 0
            changed = True
            # A rigid boundary block that did not fit may fit now. Retry the
            # injection on the next substep; the bodies stay frozen until it
            # lands (bd-r5-review-n28.58). Host-side ints only.
            rbm = self.rigid_body_manager
            if (rbm.boundary_dropped
                    and w._high_water + rbm.num_boundary_particles <= w.max_particles):
                rbm.mark_for_reinjection()

        if num_dead < reserve:
            if w.add_dead_reserve(reserve - num_dead, spacing=self._h,
                                  grid_min=(-self._world_half_size,) * 3) > 0:
                changed = True

        if changed:
            # n changed: force a full sort next time
            self.notify_scene_changed()
            self._dead_reserve_check_pending = False
        return w._high_water

    def compute_render_color(self, out, n: int | None = None) -> None:
        """Write the render color of the first ``n`` slots into ``out``.

        ``out`` is a (>= n, 4) float32 device array: the mapped color VBO in
        copy_to_vbos, or a plain CuPy array in tests and tools. One
        K_ComputeColor launch over the unsorted arrays, no host sync.
        ``n`` defaults to the high-water mark, which covers the rigid
        boundary tail.
        """
        w = self.world
        if n is None:
            n = w._high_water
        integrate.compute_color(
            n, w.position, w.velocity, w.packed_info,
            w.temperature, w.health, out,
        )

    def copy_to_vbos(self, cuda_pos, cuda_col, cuda_vel=None) -> None:
        """Fill the mapped GL VBOs from the UNSORTED arrays.

        Position and velocity are copied; color is computed into its VBO by
        compute_render_color.

        Parameters
        ----------
        cuda_pos : CudaGLBuffer
            Position VBO wrapper (must be context-managed externally).
        cuda_col : CudaGLBuffer
            Color VBO wrapper (must be context-managed externally).
        cuda_vel : CudaGLBuffer, optional
            Velocity VBO wrapper for motion blur.

        Clears scene_dirty: the VBOs now match the current state.
        """
        self.scene_dirty = False
        n = self.world._high_water
        if n == 0:
            return

        # Position: world pos (float4) -> VBO pos (float4)
        pos_arr = cuda_pos.device_pointer_as_cupy_array(
            (cuda_pos.nbytes // 16, 4), np.float32,
        )
        pos_arr[:n] = self.world.position[:n]

        # Color: computed straight into the VBO once per frame from the
        # unsorted state (bd-r5-review-n28.145). No solver stores a color.
        col_arr = cuda_col.device_pointer_as_cupy_array(
            (cuda_col.nbytes // 16, 4), np.float32,
        )
        self.compute_render_color(col_arr, n)

        # Velocity: world velocity (float4) -> VBO vel (float4) for motion blur
        if cuda_vel is not None:
            vel_arr = cuda_vel.device_pointer_as_cupy_array(
                (cuda_vel.nbytes // 16, 4), np.float32,
            )
            vel_arr[:n] = self.world.velocity[:n]

    def copy_foam_to_vbo(self, cuda_foam_pos) -> int:
        """Copy foam positions to a mapped GL VBO.

        Returns the number of active foam particles. Uses 1-frame-deferred
        async readback (no GPU->CPU sync stall).
        """
        w = self.world
        if not w.foam_enabled:
            return 0

        foam_n = self._last_foam_count  # from async readback (1 frame old)
        if foam_n <= 0:
            return 0

        foam_n = min(foam_n, w._max_foam)
        foam_arr = cuda_foam_pos.device_pointer_as_cupy_array(
            (cuda_foam_pos.nbytes // 16, 4), np.float32,
        )
        foam_arr[:foam_n] = w.foam_position[:foam_n]
        return foam_n

    @property
    def paused(self) -> bool:
        return self._paused

    def toggle_pause(self) -> None:
        self._paused = not self._paused

    def adjust_speed(self, delta: float) -> None:
        """Adjust speed by delta, clamping to [0.1, 10.0]."""
        self.speed = max(0.1, min(10.0, self.speed + delta))

    def adjust_accuracy(self, delta: float) -> None:
        """Adjust accuracy by delta, clamping to [0.1, 1.0].

        Takes effect on the next frame through _compute_adaptive_dt
        (adaptive mode only).
        """
        self.accuracy = max(0.1, min(1.0, self.accuracy + delta))

    def reset_spawn_damping(self) -> None:
        """Reset spawn damping counter (call after preset load or world reset)."""
        self._spawn_substep = 0

    def toggle_fixed_dt(self) -> None:
        """Toggle between adaptive and fixed timestep modes."""
        self.fixed_dt = not self.fixed_dt
        if self.fixed_dt:
            # Switch to fixed mode: use fixed_dt_value
            self._fixed_subdiv = 1
            self._upload_dt(self.fixed_dt_value)

    @property
    def last_substeps(self) -> int:
        return self._last_substeps

    def close(self) -> None:
        """Release CUDA resources held by this Simulation instance.

        Must be called before discarding a Simulation object (e.g. on
        max-particles change) to prevent stream/event/pinned-buffer/graph
        leaks and to avoid an in-flight D2H race on the readback stream.

        Safe to call more than once (idempotent after first call).
        """
        # 1. Drain any in-flight async device-to-host copies on the readback stream.
        #    Without this, the pinned host buffers we are about to release could still
        #    be written by the GPU after Python frees them.
        try:
            if self._readback_stream is not None:
                self._readback_stream.synchronize()
        except Exception:
            pass

        # 2. Free CUDA graph captures so their device-side allocations are returned.
        try:
            self._graph_full_sort = None
            self._graph_gather_only = None
        except Exception:
            pass

        # 3. Release the readback event and stream (CuPy handles the CUDA-level
        #    free when the Python objects are garbage-collected, but nulling here
        #    makes the intent explicit and helps GC run promptly).
        try:
            self._readback_event = None
            self._readback_done = None
            self._readback_pending = None
            self._readback_free_events = None
            self._readback_stream = None
        except Exception:
            pass

        # 4. Release pinned host memory.  CuPy's MemoryPointer.__del__ calls
        #    cudaFreeHost, but we trigger it now to avoid keeping pinned pages
        #    around until the next GC cycle.
        try:
            self._pinned_displacement = None
            self._pinned_foam_count = None
            self._pinned_dead_count = None
        except Exception:
            pass

    def get_all_modules(self) -> list:
        """Return all compiled CuPy RawModule objects known to this Simulation.

        Used by RigidBodyManager.upload() to write c_num_rigid_bodies into
        each module's constant memory via get_global().
        """
        mods = [
            step1.get_module(),
            step2.get_module(),
            integrate.get_module(),
            reactions.get_module(),
            spawn.get_module(),
            wake.get_module(),
            counting_sort.get_module(),
        ]
        # Optional solver modules (may not be compiled yet)
        try:
            import pbf_solver
            mods.append(pbf_solver.get_module())
        except Exception:
            pass
        try:
            import dfsph_solver
            mods.append(dfsph_solver.get_module())
        except Exception:
            pass
        try:
            mods.append(implicit_st.get_module())
        except Exception:
            pass
        try:
            mods.append(foam.get_module())
        except Exception:
            pass
        return [m for m in mods if m is not None]
