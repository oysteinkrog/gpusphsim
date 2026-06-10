"""SPH simulation orchestrator.

Wires up the full SPH pipeline: Hash -> Sort -> Reorder -> Build -> Step1 ->
Step2 -> Integrate.  Decouples simulation from rendering via a speed parameter,
accuracy parameter (CFL number), and substep budget.

Each render frame:
  1. If adaptive mode: compute dt from CFL constraints (acoustic, viscous — CPU-only)
  2. Compute sim_steps = clamp(round(speed * wall_dt / sim_dt), 0, max_substeps)
  3. Run sim_steps simulation substeps
  4. Copy UNSORTED pos/color to mapped VBOs (rendering happens once per frame)
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from world import World
from materials import (
    WATER, SAND, MATERIALS, GRANULAR, GAS, STATIC,
    build_material_array, build_interaction_matrix, upload_to_gpu,
)
from solver_profiles import SolverType, SolverProfile, PROFILES

import cupy

import hash_sort
import fused_sort_reorder_build
import counting_sort
import step1
import step2
import integrate
import reactions
import spawn
import wake
import foam
import implicit_st

# Adaptive timestep limits
DT_MIN = 1e-5
DT_MAX = 0.001


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
        CFL number for adaptive timestep (0.1-1.0, default 0.4).
        Lower = smaller dt = more stable. Higher = larger dt = faster but riskier.
    fixed_dt : bool
        If True, use fixed timestep (no CFL computation). Default False.
    max_substeps : int
        Maximum substeps per render frame to prevent spiral-of-death.
    """

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
    ) -> None:
        self.world = world
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
        self._substep_counter = 0  # unique per-substep counter for GPU RNG seeding
        self.compact_interval = compact_interval
        self.compact_threshold = compact_threshold
        # --- CUDA graph capture state ---
        self._sort_perm = cupy.empty(world.max_particles, dtype=cupy.uint32)
        self._frame_counter_d = cupy.zeros(1, dtype=cupy.uint32)
        # Two-graph system: separate graphs for full-sort and gather-only paths
        # so sort-skip decisions don't invalidate the graph.
        self._graph_full_sort = None      # cupy.cuda.Graph for full-sort path
        self._graph_gather_only = None    # cupy.cuda.Graph for gather-only path
        self._graph_full_sort_n = 0       # n when full-sort graph was captured
        self._graph_gather_only_n = 0     # n when gather-only graph was captured

        # --- Async readback (1-frame-deferred, avoids GPU→CPU sync stalls) ---
        import ctypes as _ctypes
        self._readback_stream = cupy.cuda.Stream(non_blocking=True)
        self._readback_event = cupy.cuda.Event()  # for cross-stream sync
        # Pinned host buffers for async D2H copy
        self._pinned_displacement = cupy.cuda.alloc_pinned_memory(4)
        self._pinned_foam_count = cupy.cuda.alloc_pinned_memory(4)
        # Zero-init pinned buffers
        _ctypes.memset(self._pinned_displacement.ptr, 0, 4)
        _ctypes.memset(self._pinned_foam_count.ptr, 0, 4)
        # Previous frame values (used for decisions while async copy is in-flight)
        self._last_max_disp_uint = 0
        self._last_foam_count = 0

        # --- Counting sort scratch buffers (allocated in _upload_constants) ---
        self._cs_histogram = None
        self._cs_write_offset = None

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
        # Precompute max viscosity and min density for viscous CFL
        self._max_eta, self._rho_min = self._compute_viscosity_bounds()

        # Smoothing length (must match _upload_constants)
        self._h = 0.04

        # --- Grid reuse state ---
        # When max displacement since last sort < threshold, skip full counting
        # sort and use gather_reorder (reuse cell_start/cell_end).
        # Safety: force sort after MAX_SORT_SKIP_FRAMES consecutive skipped frames.
        self._sort_skip_next = False
        # Threshold: (0.25 * h)^2 -- max displacement since last sort
        self._sort_skip_threshold_sq = (0.25 * 0.04) ** 2  # 0.0001
        # PBF/DFSPH: tighter threshold (0.15*h) and lower max skip count
        self._sort_skip_threshold_sq_tight = (0.15 * 0.04) ** 2  # 0.000036
        self._sort_skipped_count = 0  # stats for UI
        self._sort_skip_consecutive = 0  # consecutive frames with sort skipped
        self._MAX_SORT_SKIP_FRAMES = 4  # force sort after this many frames (WCSPH)
        self._MAX_SORT_SKIP_FRAMES_TIGHT = 2  # force sort after this many frames (PBF/DFSPH)

        # Grid cell tables — allocated in _upload_constants based on world size
        self._cell_start = None
        self._cell_end = None

        # Cell wake flags — allocated in _upload_constants based on world size
        self._cell_wake_flags = None

        # Freelist for spawn/kill system (pre-allocated, reset each step)
        self._dead_indices, self._dead_count = spawn.allocate_freelist(
            world.max_particles
        )

        # --- Implicit surface tension ---
        self.ist_enabled = False
        self.ist_sigma = 0.5
        self.ist_iterations = 5
        self.ist_surface_threshold = 25.0
        self._IST_MAX_PARTICLES = 100_000  # disable above this count

        # Cached sim_params for dt updates (set during _upload_constants)
        self._sim_params: Optional[np.ndarray] = None

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

        # Compile all kernel modules and upload constant memory
        self._upload_constants()

    def _compute_c_sound(self) -> float:
        """Compute acoustic speed: max over active materials of sqrt(k * gamma / rho0)."""
        c_sound = 0.0
        for mat in MATERIALS.values():
            if mat.behavior_class == STATIC or mat.rest_density <= 0.0:
                continue
            k = mat.eos_stiffness
            gamma = mat.eos_gamma
            rho0 = mat.rest_density
            if k > 0.0 and gamma > 0.0 and rho0 > 0.0:
                c = math.sqrt(k * gamma / rho0)
                c_sound = max(c_sound, c)
        return max(c_sound, 0.01)  # floor to avoid zero

    def _compute_viscosity_bounds(self) -> tuple:
        """Compute max viscosity and min density for viscous CFL.

        Uses base_viscosity from the material table (not mu_max, which is
        a theoretical clamp rarely reached and would cripple the timestep).
        rho_min excludes GAS materials (they use low constant viscosity,
        not mu(I), so their low density doesn't tighten the viscous CFL).
        """
        max_eta = 0.0
        rho_min = 1e30
        for mat in MATERIALS.values():
            if mat.behavior_class == STATIC or mat.rest_density <= 0.0:
                continue
            if mat.behavior_class != GAS and mat.rest_density < rho_min:
                rho_min = mat.rest_density
            if mat.base_viscosity > max_eta:
                max_eta = mat.base_viscosity
        rho_min = max(rho_min, 0.1)  # floor
        return max_eta, rho_min

    def _compute_adaptive_dt(self, n: int) -> float:
        """Compute adaptive dt from CFL constraints (CPU-only, no GPU sync).

        Two constraints (precomputed from material properties):
          dt_acoustic = 0.25 * h / c_sound
          dt_viscous  = accuracy * rho_min * h^2 / max(max_eta, 1e-6)
          dt = min(dt_acoustic, dt_viscous) clamped to [DT_MIN, DT_MAX]

        The advection CFL (velocity-based) is omitted because velocity_limit
        and accel_max prevent particles from exceeding the speed of sound,
        and the acoustic CFL (dt~0.05) is always well above DT_MAX (0.005).
        """
        h = self._h

        # Acoustic CFL (fixed 0.25 coefficient, not scaled by accuracy)
        dt_acoustic = 0.25 * h / self._c_sound

        # Viscous CFL (uses precomputed max_eta and rho_min)
        dt_viscous = self.accuracy * self._rho_min * h * h / max(self._max_eta, 1e-6)

        dt = min(dt_acoustic, dt_viscous)
        return max(DT_MIN, min(DT_MAX, dt))

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
        # Upload to all WCSPH modules that read c_sim.dt
        step1.upload_sim_params(self._sim_params)
        step2.upload_sim_params(self._sim_params)
        integrate.upload_sim_params(self._sim_params)
        reactions.upload_sim_params(self._sim_params)
        spawn.upload_sim_params(self._sim_params)
        # Also upload to active solver module
        if self._profile.solver_type == SolverType.PBF:
            import pbf_solver
            pbf_solver.upload_sim_params(self._sim_params)
        elif self._profile.solver_type == SolverType.DFSPH:
            import dfsph_solver
            dfsph_solver.upload_sim_params(self._sim_params)
        # IST module also reads c_sim
        implicit_st.upload_sim_params(self._sim_params)
        # Update foam dt
        if self._foam_params is not None:
            self._foam_params[0]["dt"] = np.float32(new_dt)
            foam.upload_foam_params(self._foam_params)

    def _upload_sim_params_all(self) -> None:
        """Re-upload sim_params to all modules (without changing self.dt)."""
        step1.upload_sim_params(self._sim_params)
        step2.upload_sim_params(self._sim_params)
        integrate.upload_sim_params(self._sim_params)
        reactions.upload_sim_params(self._sim_params)
        spawn.upload_sim_params(self._sim_params)
        if self._profile.solver_type == SolverType.PBF:
            import pbf_solver
            pbf_solver.upload_sim_params(self._sim_params)
        elif self._profile.solver_type == SolverType.DFSPH:
            import dfsph_solver
            dfsph_solver.upload_sim_params(self._sim_params)
        implicit_st.upload_sim_params(self._sim_params)

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
        self._upload_sim_params_all()
        self._invalidate_graphs()

    @property
    def world_half_size(self) -> float:
        return self._world_half_size

    def set_world_size(self, half_size: float) -> None:
        """Change the world half-extent and re-upload all constants."""
        self._world_half_size = half_size
        self._upload_constants()
        # Invalidate CUDA graphs (grid changed) and reset sort skip
        self._invalidate_graphs()
        self._sort_skip_next = False
        self._sort_skip_consecutive = 0

    def _upload_constants(self) -> None:
        """Compile all .cu kernels and upload constant memory to each module."""
        hs = self._world_half_size
        wmin = (-hs, -hs, -hs)
        wmax = (hs, hs, hs)

        # Build shared param structs -- auto-size hash table for particle count
        grid_params, table_size = hash_sort.build_grid_params_for_world(
            wmin, wmax, self._h,
            num_particles=self.world._high_water,
        )

        # Reallocate cell arrays if table size changed
        if table_size != self._table_size:
            self._table_size = table_size
            self._cell_start = cupy.empty(table_size, dtype=cupy.uint32)
            self._cell_end = cupy.empty(table_size, dtype=cupy.uint32)
            self._cell_wake_flags = wake.allocate_cell_wake_flags(table_size)
            # Counting sort scratch buffers
            self._cs_histogram = cupy.zeros(table_size, dtype=cupy.uint32)
            self._cs_write_offset = cupy.zeros(table_size, dtype=cupy.uint32)

        # CFL velocity limit: v_max = factor * h / dt
        vlf = getattr(self._profile, 'velocity_limit_factor', 0.9)
        vel_limit = vlf * self._h / max(self.dt, 1e-8)
        sim_params = step1.build_sim_params(
            smoothing_length=self._h,
            particle_mass=0.02,
            particle_spacing=0.02,
            gravity=(0.0, -4.0, 0.0),
            dt=self.dt,
            restitution=0.3,
            wall_friction=0.5,
            world_min=wmin,
            world_max=wmax,
            velocity_limit=vel_limit,
        )
        self._sim_params = sim_params
        precalc_params = step1.build_precalc_params(
            smoothing_length=0.04,
            viscosity=1.0,
        )
        granular_params = step2.build_granular_params()
        materials_data = build_material_array()
        interactions_data = build_interaction_matrix()

        # --- hash_sort module: c_grid ---
        hash_sort.upload_grid_params(grid_params)

        # --- fused_sort_reorder_build module: c_grid ---
        fused_sort_reorder_build.upload_grid_params(grid_params)

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

        # --- integrate module: c_sim, c_materials ---
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

        # Invalidate CUDA graphs and reset grid reuse
        self._invalidate_graphs()
        self._sort_skip_next = False
        self._sort_skip_consecutive = 0
        # Reset spawn damping for new solver
        self._spawn_substep = 0
        # Reset rigid boundary injection flag
        self._rigid_boundary_initialized = False

    def _upload_solver_constants(self, solver_module) -> None:
        """Upload all shared constants (grid, sim, precalc, materials, interactions) to a solver module."""
        hs = self._world_half_size
        wmin = (-hs, -hs, -hs)
        wmax = (hs, hs, hs)
        grid_params, _ = hash_sort.build_grid_params_for_world(wmin, wmax, self._h)
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

    def _update_rigid_boundary(self, n_fluid: int) -> int:
        """Update rigid body boundary particles in unsorted arrays.

        On first call (before any integration), uses inject_boundary_particles
        to set up initial positions. On subsequent calls, uses
        K_UpdateBoundaryParticles GPU kernel for fast position/velocity update.

        Returns n_total = n_fluid + n_boundary.
        """
        rbm = self.rigid_body_manager
        w = self.world
        n_boundary = rbm.num_boundary_particles
        n_total = n_fluid + n_boundary

        if n_total > w.max_particles:
            return n_fluid  # not enough room

        if not hasattr(self, '_rigid_boundary_initialized'):
            # First substep: inject via Python (sets up packed_info, color, etc.)
            rbm.inject_boundary_particles(w)
            self._rigid_boundary_initialized = True
        else:
            # Subsequent substeps: GPU kernel updates positions + velocities
            integrate.update_boundary_particles(
                rbm.boundary_data,
                rbm.d_rigid_bodies,
                w.position,
                w.velocity,
                w.mass,
                offset=n_fluid,
                num_boundary=n_boundary,
            )

        return n_total

    def _ensure_table_size(self, n: int) -> None:
        """Resize hash table if particle count outgrew it (load factor > 1)."""
        needed = hash_sort.compute_table_size(n)
        if needed != self._table_size:
            self._table_size = needed
            self._cell_start = cupy.empty(needed, dtype=cupy.uint32)
            self._cell_end = cupy.empty(needed, dtype=cupy.uint32)
            self._cell_wake_flags = wake.allocate_cell_wake_flags(needed)
            self._cs_histogram = cupy.zeros(needed, dtype=cupy.uint32)
            self._cs_write_offset = cupy.zeros(needed, dtype=cupy.uint32)
            # Re-upload c_grid to all modules with new table_size
            hs = self._world_half_size
            wmin, wmax = (-hs, -hs, -hs), (hs, hs, hs)
            grid_params, _ = hash_sort.build_grid_params_for_world(
                wmin, wmax, self._h, num_particles=n,
            )
            for mod in [hash_sort, fused_sort_reorder_build, counting_sort,
                        step1, step2, wake, implicit_st]:
                mod.upload_grid_params(grid_params)
            # Invalidate CUDA graphs (grid params changed)
            self._invalidate_graphs()

    def _run_grid_setup(self, n: int, force_sort: bool = False) -> None:
        """Common grid setup: full counting sort or gather-only (grid reuse).

        When _sort_skip_next is True and force_sort is False, uses gather_reorder
        to re-scatter unsorted data using the previous sort_perm.  Cell_start/cell_end
        are reused from the previous frame.  Saves hash, histogram, prefix_sum, and
        cell_end computation.
        """
        w = self.world

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
                kappa=w.kappa,
                particle_dye=w.particle_dye,
                angular_velocity=w.angular_velocity,
                kappa_v=w.kappa_v,
                lambda_pbf=w.lambda_pbf,
                sorted_position=w.sorted_position,
                sorted_velocity=w.sorted_velocity,
                sorted_mass=w.sorted_mass,
                sorted_packed_info=w.sorted_packed_info,
                sorted_temperature=w.sorted_temperature,
                sorted_health=w.sorted_health,
                sorted_lifetime=w.sorted_lifetime,
                sorted_sleep_counter=w.sorted_sleep_counter,
                sorted_kappa=w.sorted_kappa,
                sorted_particle_dye=w.sorted_particle_dye,
                sorted_angular_velocity=w.sorted_angular_velocity,
                sorted_kappa_v=w.sorted_kappa_v,
                sorted_lambda_pbf=w.sorted_lambda_pbf,
                sorted_velocity_h=w.sorted_velocity_h,
                sorted_temperature_h=w.sorted_temperature_h,
                sorted_dye_h=w.sorted_dye_h,
            )
            return

        # Full sort: reset max_displacement tracker (fresh baseline for grid reuse)
        w.max_displacement.data.memset_async(0x00, w.max_displacement.nbytes)

        counting_sort.counting_sort_full(
            num_particles=n,
            num_cells=self._table_size,
            # Scratch buffers
            histogram=self._cs_histogram,
            write_offset=self._cs_write_offset,
            cell_start=self._cell_start,
            cell_end=self._cell_end,
            sort_perm=self._sort_perm[:n],
            # Hash I/O
            positions=w.position[:n],
            hashes=w.hashes,
            sorted_hashes=w.sorted_hashes,
            # Unsorted particle arrays
            position=w.position,
            velocity=w.velocity,
            mass=w.mass,
            packed_info=w.packed_info,
            temperature=w.temperature,
            health=w.health,
            lifetime=w.lifetime,
            sleep_counter=w.sleep_counter,
            kappa=w.kappa,
            particle_dye=w.particle_dye,
            angular_velocity=w.angular_velocity,
            kappa_v=w.kappa_v,
            lambda_pbf=w.lambda_pbf,
            # Sorted particle arrays
            sorted_position=w.sorted_position,
            sorted_velocity=w.sorted_velocity,
            sorted_mass=w.sorted_mass,
            sorted_packed_info=w.sorted_packed_info,
            sorted_temperature=w.sorted_temperature,
            sorted_health=w.sorted_health,
            sorted_lifetime=w.sorted_lifetime,
            sorted_sleep_counter=w.sorted_sleep_counter,
            sorted_kappa=w.sorted_kappa,
            sorted_particle_dye=w.sorted_particle_dye,
            sorted_angular_velocity=w.sorted_angular_velocity,
            sorted_kappa_v=w.sorted_kappa_v,
            sorted_lambda_pbf=w.sorted_lambda_pbf,
            sorted_velocity_h=w.sorted_velocity_h,
            sorted_temperature_h=w.sorted_temperature_h,
            sorted_dye_h=w.sorted_dye_h,
        )

    def _run_reactions_spawn(self, n: int) -> None:
        """Common reactions + spawn pass (used by all solvers after density)."""
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
        )

        # Blast wave: radial impulse from freshly-exploded gunpowder
        from materials import GUNPOWDER
        if GUNPOWDER in w._spawned_material_ids:
            reactions.compute_blast_wave(
                w.sorted_position[:n],
                w.sorted_velocity[:n],
                w.sorted_packed_info[:n],
                w.sorted_lifetime[:n],
                self._cell_start,
                self._cell_end,
                smoothing_length=self._h,
            )

        # Spawn (always included — zero-cost when no SPAWN_GAS flags set)
        spawn.compute_spawn(
            w.sorted_packed_info[:n],
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_veleval[:n],
            w.sorted_mass[:n],
            w.sorted_temperature[:n],
            w.sorted_health[:n],
            w.sorted_lifetime[:n],
            w.sorted_color[:n],
            w.sorted_sleep_counter[:n],
            w.sorted_density[:n],
            w.sorted_shear_rate[:n],
            self._dead_indices,
            self._dead_count,
        )

    def _run_graph_body(self, n: int) -> None:
        """Execute the graph-capturable portion of the pipeline.

        Dispatches to the correct solver body based on the active profile.
        """
        if self._profile.solver_type == SolverType.PBF:
            self._run_pbf_body(n)
        elif self._profile.solver_type == SolverType.DFSPH:
            self._run_dfsph_body(n)
        else:
            self._run_wcsph_body(n)

    def _run_wcsph_body(self, n: int) -> None:
        """WCSPH pipeline: Grid -> Step1 -> Reactions -> Spawn -> Step2 -> Integrate -> Wake."""
        w = self.world

        # Grid setup (may skip sort if particles barely moved).
        # max_displacement is reset only on full sort (in _run_grid_setup),
        # so it accumulates across substeps since the last full sort.
        self._run_grid_setup(n)

        # Step1: density + strain-rate + heat diffusion + exposure + vorticity + normal + dye
        step1.compute_step1(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_mass[:n],
            w.sorted_density if hasattr(w, '_density_initialized') else None,
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
            particle_dye_in=w.sorted_particle_dye[:n],
            dye_rate_out=w.sorted_dye_rate,
            velocity_h=w.sorted_velocity_h,
            pressure_out=w.sorted_pressure,
            temperature_h=w.sorted_temperature_h,
            dye_h=w.sorted_dye_h,
        )

        # Pack density into position.w now done inside K_Step1 (PERF-004)
        # Pressure pre-computed inside K_Step1 (PERF-007)

        # Reactions + Spawn
        self._run_reactions_spawn(n)

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
        )

        # Implicit surface tension (WCSPH only, quality mode for < 100K particles)
        if self.ist_enabled and n < self._IST_MAX_PARTICLES:
            result = implicit_st.run_implicit_st(
                w.sorted_velocity[:n],
                w.sorted_veleval[:n],  # reuse veleval as scratch (overwritten by integrate anyway)
                w.sorted_position[:n],
                w.sorted_density[:n],
                w.sorted_mass[:n],
                w.sorted_packed_info[:n],
                w.sorted_normal[:n],
                self._cell_start,
                self._cell_end,
                num_iterations=self.ist_iterations,
            )
            # If result ended up in veleval (odd iteration count), copy back to velocity
            if result.data.ptr != w.sorted_velocity[:n].data.ptr:
                w.sorted_velocity[:n] = result

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
            sorted_density=w.sorted_density[:n],
            sorted_shear_rate=w.sorted_shear_rate[:n],
            sorted_dTdt=w.sorted_dTdt[:n],
            sorted_sleep_counter=w.sorted_sleep_counter[:n],
            sorted_dye_rate=w.sorted_dye_rate[:n],
            sorted_particle_dye=w.sorted_particle_dye[:n],
            sorted_vorticity=w.sorted_vorticity[:n],
            sorted_angular_velocity=w.sorted_angular_velocity[:n],
            sort_indexes=self._sort_perm[:n],
            position_out=w.position,
            velocity_out=w.velocity,
            color_out=w.color,
            packed_info_out=w.packed_info,
            sleep_counter_out=w.sleep_counter,
            temperature_out=w.temperature,
            particle_dye_out=w.particle_dye,
            angular_velocity_out=w.angular_velocity,
            max_displacement=w.max_displacement,
            cell_start=self._cell_start,
            cell_end=self._cell_end,
        )

        # Wake propagation (memset + 2 kernels)
        wake.run_wake_propagation(
            w.position[:n],
            w.velocity[:n],
            w.packed_info[:n],
            w.sleep_counter[:n],
            self._cell_wake_flags,
            num_particles=n,
        )

    def _run_pbf_body(self, n: int) -> None:
        """PBF pipeline: Grid -> Predict -> [Lambda -> Delta -> Apply]xN -> Finalize."""
        import pbf_solver
        w = self.world

        # Grid setup
        self._run_grid_setup(n)

        # 1. Predict positions (gravity integration)
        pbf_solver.pbf_predict(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_predicted_position,
        )

        # 2. Initial density + lambda + heat diffusion + exposure (first call only)
        pbf_solver.pbf_compute_lambda(
            w.sorted_predicted_position[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            self._cell_start, self._cell_end,
            w.sorted_density, w.sorted_lambda_pbf,
            w.sorted_pressure_normal,
            temperature_in=w.sorted_temperature[:n],
            density_in=w.sorted_density if hasattr(w, '_density_initialized') else None,
            dTdt_out=w.sorted_dTdt,
            exposure_heat_out=w.sorted_exposure_heat,
            exposure_corrode_out=w.sorted_exposure_corrode,
            particle_dye_in=w.sorted_particle_dye[:n],
            dye_rate_out=w.sorted_dye_rate,
            velocity_in=w.sorted_velocity[:n],
            vorticity_out=w.sorted_vorticity,
            normal_out=w.sorted_normal,
        )

        # 3. Reactions + Spawn
        self._run_reactions_spawn(n)

        # 4. Solver iterations (fixed count for CUDA graph)
        for i in range(self._profile.pbf_iterations):
            if i > 0:
                pbf_solver.pbf_compute_lambda(
                    w.sorted_predicted_position[:n], w.sorted_mass[:n],
                    w.sorted_packed_info[:n],
                    self._cell_start, self._cell_end,
                    w.sorted_density, w.sorted_lambda_pbf,
                    w.sorted_pressure_normal,
                )

            pbf_solver.pbf_compute_delta(
                w.sorted_predicted_position[:n], w.sorted_lambda_pbf[:n],
                w.sorted_mass[:n], w.sorted_packed_info[:n],
                self._cell_start, self._cell_end,
                w.sorted_delta_position,
            )

            pbf_solver.pbf_apply_delta(
                w.sorted_predicted_position[:n], w.sorted_delta_position[:n],
                w.sorted_packed_info[:n],
            )

        # 5. Finalize: velocity update, XSPH, friction, color, sleep, writeback
        rbm = self.rigid_body_manager
        pbf_solver.pbf_finalize(
            w.sorted_predicted_position[:n], w.sorted_position[:n],
            w.sorted_velocity[:n], w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_dTdt[:n], w.sorted_sleep_counter[:n],
            self._sort_perm[:n], self._cell_start, self._cell_end,
            w.position, w.velocity, w.color, w.packed_info,
            w.sleep_counter, w.temperature,
            sorted_particle_dye=w.sorted_particle_dye[:n],
            sorted_dye_rate=w.sorted_dye_rate[:n],
            particle_dye_out=w.particle_dye,
            sorted_angular_velocity=w.sorted_angular_velocity[:n],
            angular_velocity_out=w.angular_velocity,
            vorticity_in=w.sorted_vorticity[:n],
            normal_in=w.sorted_normal[:n],
            pressure_normal_in=w.sorted_pressure_normal[:n],
            sorted_lambda_pbf=w.sorted_lambda_pbf[:n],
            lambda_pbf_out=w.lambda_pbf,
            d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
            d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
            d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            max_displacement=w.max_displacement,
        )

    def _run_dfsph_body(self, n: int) -> None:
        """DFSPH pipeline: Grid -> DensityAlpha -> NonPressure -> DivSolve -> PredictPos -> DensAdv -> DensSolve -> Finalize."""
        import dfsph_solver
        w = self.world

        # Grid setup
        self._run_grid_setup(n)

        # 1. Density + alpha precompute (also computes shear_rate for GRANULAR)
        # density_in aliasing note: sorted_density is both density_in and density_out.
        # It's not reordered during sort, but the in-place race means threads may read
        # either the previous or current frame's density for neighbors — both are
        # acceptable for the volume weighting used in alpha/heat/dye computation.
        # Using None (rho_j=1000 fallback) is worse because it underestimates surface
        # particle volumes, making alpha too large and increasing boundary oscillation.
        dfsph_solver.compute_density_alpha(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_mass[:n],
            w.sorted_density if hasattr(w, '_density_initialized') else None,
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            self._cell_start, self._cell_end,
            w.sorted_density, w.sorted_alpha_dfsph,
            w.sorted_shear_rate, w.sorted_dTdt,
            w.sorted_exposure_heat, w.sorted_exposure_corrode,
            particle_dye_in=w.sorted_particle_dye[:n],
            dye_rate_out=w.sorted_dye_rate,
            vorticity_out=w.sorted_vorticity,
            normal_out=w.sorted_normal,
        )

        # 2. Reactions + Spawn
        self._run_reactions_spawn(n)

        # 3. Non-pressure forces (viscosity, gravity, mu(I), vorticity confinement, surface tension)
        rbm = self.rigid_body_manager
        dfsph_solver.compute_non_pressure_forces(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_shear_rate[:n],
            w.sorted_temperature[:n],
            self._cell_start, self._cell_end,
            w.sorted_velocity,  # velocity updated in-place
            vorticity_in=w.sorted_vorticity[:n],
            normal_in=w.sorted_normal[:n],
            d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
            d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
            d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
        )

        # 4. Divergence solver (warm-started kappa_v -- PERF-008)
        # Apply warm-started kappa_v from previous substep (scaled during sort reorder writeback)
        if self._profile.dfsph_div_warm_start > 0.0:
            w.sorted_kappa_v[:n] *= self._profile.dfsph_div_warm_start
            dfsph_solver.correct_velocity_div(
                w.sorted_velocity, w.sorted_density[:n],
                w.sorted_mass[:n], w.sorted_kappa_v[:n],
                w.sorted_packed_info[:n], w.sorted_position[:n],
                self._cell_start, self._cell_end,
            )
        for _ in range(self._profile.dfsph_div_iters):
            dfsph_solver.compute_kappa_v(
                w.sorted_velocity[:n], w.sorted_density[:n],
                w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
                w.sorted_packed_info[:n], w.sorted_position[:n],
                self._cell_start, self._cell_end,
                w.sorted_kappa_v,
            )
            dfsph_solver.correct_velocity_div(
                w.sorted_velocity, w.sorted_density[:n],
                w.sorted_mass[:n], w.sorted_kappa_v[:n],
                w.sorted_packed_info[:n], w.sorted_position[:n],
                self._cell_start, self._cell_end,
            )

        # kappa_v writeback now done in K_DFSPH_Finalize (PERF-008)

        # 5. Density solver (SPlisHSPlasH-style Jacobi iteration on pressure)
        # Key insight: iterate on PRESSURE variable p/rho^2, not velocity.
        # Each iteration:
        #   a) DensitySolverUpdate: predict density using v + dt*a_press,
        #      compute residual, update p_rho2 via Jacobi step
        #   b) ComputePressureAccel: compute a_press from updated p_rho2
        # After convergence: apply final a_press to velocity once.
        # Buffer reuse: sorted_kappa -> p_rho2 (warm-started from prev frame),
        #               sorted_predicted_position -> a_press
        p_rho2 = w.sorted_kappa  # already contains warm-started values from reorder
        a_press = w.sorted_predicted_position
        # Warm-start: scale previous pressure by decay factor (0.5 = carry 50%)
        p_rho2[:n] *= self._profile.dfsph_warm_start
        # Compute initial a_press from warm-started p_rho2
        dfsph_solver.compute_pressure_accel(
            p_rho2[:n], w.sorted_position[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            self._cell_start, self._cell_end,
            a_press,
        )
        for _ in range(self._profile.dfsph_dens_iters):
            dfsph_solver.density_solver_update(
                w.sorted_velocity[:n], a_press[:n],
                w.sorted_position[:n], w.sorted_density[:n],
                w.sorted_mass[:n], w.sorted_alpha_dfsph[:n],
                w.sorted_packed_info[:n],
                self._cell_start, self._cell_end,
                p_rho2,
            )
            dfsph_solver.compute_pressure_accel(
                p_rho2[:n], w.sorted_position[:n],
                w.sorted_density[:n], w.sorted_mass[:n],
                w.sorted_packed_info[:n],
                self._cell_start, self._cell_end,
                a_press,
            )
        dfsph_solver.apply_pressure_velocity(
            w.sorted_velocity[:n], a_press[:n],
            w.sorted_packed_info[:n],
        )

        # 8. Finalize + writeback (includes sleep logic via velocity magnitude)
        # Pass sorted_kappa (= p_rho2) through to unsorted kappa for warm-start next frame
        dfsph_solver.finalize(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_density[:n], w.sorted_mass[:n],
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_dTdt[:n],
            w.sorted_sleep_counter[:n],
            p_rho2[:n],  # sorted_kappa with current pressure values
            self._sort_perm[:n], self._cell_start, self._cell_end,
            w.position, w.velocity, w.color, w.packed_info,
            w.sleep_counter, w.temperature,
            w.kappa,  # unsorted kappa out for warm-start
            sorted_particle_dye=w.sorted_particle_dye[:n],
            sorted_dye_rate=w.sorted_dye_rate[:n],
            particle_dye_out=w.particle_dye,
            sorted_angular_velocity=w.sorted_angular_velocity[:n],
            angular_velocity_out=w.angular_velocity,
            vorticity_in=w.sorted_vorticity[:n],
            sorted_kappa_v=w.sorted_kappa_v[:n],
            kappa_v_out=w.kappa_v,
            d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
            d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
            d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
            max_displacement=w.max_displacement,
        )

    def _invalidate_graphs(self) -> None:
        """Invalidate both CUDA graphs (call when constants/solver/grid changes)."""
        self._graph_full_sort = None
        self._graph_gather_only = None
        self._graph_full_sort_n = 0
        self._graph_gather_only_n = 0

    def _capture_graph(self, n: int, gather_only: bool) -> cupy.cuda.Graph:
        """Capture the graph-body into a CUDA graph for replay.

        Creates a non-default capture stream, records all ops, and returns
        the resulting Graph object for repeated launch().
        """
        s = cupy.cuda.Stream(non_blocking=True)
        with s:
            s.begin_capture()
            self._run_graph_body(n)
            graph = s.end_capture()
        if gather_only:
            self._graph_gather_only = graph
            self._graph_gather_only_n = n
        else:
            self._graph_full_sort = graph
            self._graph_full_sort_n = n
        return graph

    def _sim_step(self, n: int) -> None:
        """Run one simulation substep on n particles.

        Pipeline (fully graph-capturable):
          1. Counting sort (or gather-only if sort skipped)
          2. Step1/Step2/Integrate/Wake (solver-specific)

        Two-graph strategy: separate graphs for full-sort and gather-only paths
        so sort-skip decisions don't invalidate graphs. When n changes, fall back
        to direct kernel launches (no graph capture overhead).
        """
        w = self.world

        # Update device substep counter for RNG seeding (unique per substep)
        self._frame_counter_d.fill(self._substep_counter)

        # Mark density as initialized (needed for step1 prev_density path)
        w._density_initialized = True

        # --- Spawn velocity damping ramp ---
        if self._spawn_substep < self._damping_duration:
            t = self._spawn_substep / self._damping_duration
            damping = 0.8 * (1.0 - t)  # linear ramp 0.8 -> 0.0
            self._spawn_substep += 1
            # Invalidate CUDA graphs during damping (constant changes each step)
            self._invalidate_graphs()
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

        # --- Rigid body: update boundary particles in unsorted arrays ---
        rbm = self.rigid_body_manager
        if rbm.num_bodies > 0 and rbm.num_boundary_particles > 0:
            n = self._update_rigid_boundary(n)

        # Zero rigid body force/torque accumulators before solver
        graphs_disabled = False
        if rbm.num_bodies > 0:
            rbm.rigid_forces.fill(0)
            rbm.rigid_torques.fill(0)
            # Disable CUDA graph when rigid bodies are active
            graphs_disabled = True

        # --- Update kinematic SDF objects ---
        if self.sdf_manager._motions:
            self.sdf_manager.update_kinematics(self.sim_time, self.dt)
            self.sdf_manager.upload_if_dirty()

        # Select graph based on sort-skip state
        is_gather = self._sort_skip_next
        if is_gather:
            graph = self._graph_gather_only
            graph_n = self._graph_gather_only_n
        else:
            graph = self._graph_full_sort
            graph_n = self._graph_full_sort_n

        if graphs_disabled:
            # Rigid bodies active — always run directly, no graph
            self._run_graph_body(n)
        elif graph_n != n:
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
            g.launch()
        else:
            # n stable, graph exists — fast replay
            graph.launch()

        # Rigid body integration (outside graph -- forces vary per substep)
        rbm = self.rigid_body_manager
        if rbm.num_bodies > 0:
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

        # Foam generation + physics (outside graph -- uses atomicAdd with variable count)
        self._run_foam_step(n)

        self.sim_time += self.dt
        self._substep_counter += 1

    def _run_foam_step(self, n: int) -> None:
        """Run foam generate + physics + compaction (outside CUDA graph).

        No GPU->CPU sync: kernels read foam_count from device memory,
        grid sizes use max_foam upper bound.
        """
        w = self.world
        if not w.foam_enabled:
            return

        max_foam = w._max_foam

        # 1. Physics for existing foam (before generate, so new particles
        #    get physics next step). Kernel reads count from device pointer.
        foam.foam_physics(w.foam_position, w.foam_velocity, w.foam_count, max_foam)

        # 2. Generate new foam from FLUID particles
        foam.foam_generate(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_normal[:n],
            w.sorted_packed_info[:n],
            w.foam_position,
            w.foam_velocity,
            w.foam_count,
            n,
            self._frame_counter,
        )

        # 3. Compact dead particles every 8th frame (amortize cost)
        if self._frame_counter % 8 == 0:
            foam.foam_compact(
                w.foam_position, w.foam_velocity,
                w.foam_position_b, w.foam_velocity_b,
                w.foam_alive_count,
                w.foam_count,
                max_foam,
            )
            # Swap buffers
            w.foam_position, w.foam_position_b = w.foam_position_b, w.foam_position
            w.foam_velocity, w.foam_velocity_b = w.foam_velocity_b, w.foam_velocity
            # Update count from compaction result (device-to-device copy, no sync)
            w.foam_count[:] = w.foam_alive_count

    def _sim_step_timed(self, n: int) -> None:
        """Run one substep with CUDA event timing (bypasses graph capture).

        Note: Detailed per-kernel timing only available for WCSPH.
        PBF/DFSPH fall through to a single timed graph body call.
        """
        # For non-WCSPH solvers, run un-timed (timing panel shows WCSPH stages)
        if self._profile.solver_type != SolverType.WCSPH:
            self._sim_step(n)
            return

        w = self.world
        events = []
        labels = []

        def mark(label):
            e = cupy.cuda.Event()
            e.record()
            events.append(e)
            labels.append(label)

        mark("start")

        self._frame_counter_d.fill(self._substep_counter)
        w._density_initialized = True

        # Update rigid body boundary particles
        rbm = self.rigid_body_manager
        if rbm.num_bodies > 0 and rbm.num_boundary_particles > 0:
            n = self._update_rigid_boundary(n)

        # 1-4. Grid setup (counting sort or gather-only if skipping)
        # max_displacement reset happens inside _run_grid_setup on full sort
        self._run_grid_setup(n)
        mark("sort")

        # 5. Step1
        step1.compute_step1(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_mass[:n],
            w.sorted_density if hasattr(w, '_density_initialized') else None,
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            self._cell_start, self._cell_end,
            density_out=w.sorted_density,
            shear_rate_out=w.sorted_shear_rate,
            dTdt_out=w.sorted_dTdt,
            exposure_heat_out=w.sorted_exposure_heat,
            exposure_corrode_out=w.sorted_exposure_corrode,
            vorticity_out=w.sorted_vorticity,
            normal_out=w.sorted_normal,
            particle_dye_in=w.sorted_particle_dye[:n],
            dye_rate_out=w.sorted_dye_rate,
            velocity_h=w.sorted_velocity_h,
            pressure_out=w.sorted_pressure,
            temperature_h=w.sorted_temperature_h,
            dye_h=w.sorted_dye_h,
        )
        mark("step1")

        # 6. Reactions
        spawn.reset_freelist(self._dead_count)
        reactions.compute_reactions(
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_lifetime[:n],
            w.sorted_velocity[:n], w.sorted_exposure_heat[:n],
            w.sorted_exposure_corrode[:n],
            frame_d=self._frame_counter_d,
            dead_indices=self._dead_indices, dead_count=self._dead_count,
        )
        mark("reactions")

        # 7. Spawn
        spawn.compute_spawn(
            w.sorted_packed_info[:n], w.sorted_position[:n],
            w.sorted_velocity[:n], w.sorted_veleval[:n],
            w.sorted_mass[:n], w.sorted_temperature[:n],
            w.sorted_health[:n], w.sorted_lifetime[:n],
            w.sorted_color[:n], w.sorted_sleep_counter[:n],
            w.sorted_density[:n], w.sorted_shear_rate[:n],
            self._dead_indices, self._dead_count,
        )
        mark("spawn")

        # 8. Step2 (density packed into position.w by K_Step1 -- PERF-004, pressure from Step1 -- PERF-007)
        rbm = self.rigid_body_manager
        step2.compute_step2(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            w.sorted_shear_rate[:n],
            self._cell_start, self._cell_end,
            vorticity_in=w.sorted_vorticity,
            normal_in=w.sorted_normal,
            sph_force_out=w.sorted_sph_force,
            veleval_out=w.sorted_veleval,
            velocity_h=w.sorted_velocity_h,
            pressure_in=w.sorted_pressure,
            d_rigid_bodies=rbm.d_rigid_bodies if rbm.num_bodies > 0 else None,
            d_rigid_forces=rbm.rigid_forces if rbm.num_bodies > 0 else None,
            d_rigid_torques=rbm.rigid_torques if rbm.num_bodies > 0 else None,
        )
        mark("step2")

        # 9. Integrate (with max_displacement tracking for grid reuse)
        integrate.integrate(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_veleval[:n], w.sorted_sph_force[:n],
            w.sorted_mass[:n], w.sorted_packed_info[:n],
            w.sorted_temperature[:n], w.sorted_health[:n],
            sorted_density=w.sorted_density[:n],
            sorted_shear_rate=w.sorted_shear_rate[:n],
            sorted_dTdt=w.sorted_dTdt[:n],
            sorted_sleep_counter=w.sorted_sleep_counter[:n],
            sorted_dye_rate=w.sorted_dye_rate[:n],
            sorted_particle_dye=w.sorted_particle_dye[:n],
            sorted_vorticity=w.sorted_vorticity[:n],
            sorted_angular_velocity=w.sorted_angular_velocity[:n],
            sort_indexes=self._sort_perm[:n],
            position_out=w.position, velocity_out=w.velocity,
            color_out=w.color, packed_info_out=w.packed_info,
            sleep_counter_out=w.sleep_counter,
            temperature_out=w.temperature,
            particle_dye_out=w.particle_dye,
            angular_velocity_out=w.angular_velocity,
            max_displacement=w.max_displacement,
        )
        mark("integrate")

        # 10. Wake
        wake.run_wake_propagation(
            w.position[:n], w.velocity[:n], w.packed_info[:n],
            w.sleep_counter[:n], self._cell_wake_flags, num_particles=n,
        )
        mark("wake")

        # 10b. Rigid body integration + collision
        if rbm.num_bodies > 0:
            integrate.integrate_rigid_bodies(
                rbm.d_rigid_bodies,
                rbm.rigid_forces,
                rbm.rigid_torques,
                rbm.num_bodies,
                self.dt,
                gravity=tuple(float(x) for x in self._sim_params[0]["gravity"]),
            )
            integrate.rigid_body_collisions(rbm.d_rigid_bodies, rbm.num_bodies)

        # 11. Foam (secondary particles)
        self._run_foam_step(n)
        mark("foam")

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

        # Store this substep's events for reading next substep
        self._pending_timing_events = events
        self._pending_timing_labels = labels

        self.sim_time += self.dt
        self._substep_counter += 1

    def step_frame(self) -> int:
        """Advance simulation for one render frame. Returns number of substeps run."""
        now = time.perf_counter()
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

        # Compute adaptive dt if not using fixed timestep
        if not self.fixed_dt:
            new_dt = self._compute_adaptive_dt(n)
            if abs(new_dt - self.dt) > 1e-8:
                self._upload_dt(new_dt)

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
        for _ in range(sim_steps):
            step_fn(n)

        # Increment frame counter once per render frame (I4 fix)
        if sim_steps > 0:
            self._frame_counter += 1

        # --- Grid reuse decision for next frame (uses 1-frame-old async readback) ---
        # max_displacement accumulates since the last full sort (atomicMax across all
        # substeps). We use last frame's async readback value — safe because the
        # 0.25h threshold is conservative and 1-frame lag is negligible.
        if sim_steps > 0:
            import struct
            import ctypes as _ctypes
            # Read last frame's async readback result (no sync needed)
            max_disp_sq_uint = self._last_max_disp_uint
            if max_disp_sq_uint > 0:
                max_disp_sq = struct.unpack('f', struct.pack('I', max_disp_sq_uint))[0]
            else:
                max_disp_sq = 0.0
            if self._profile.solver_type == SolverType.WCSPH:
                threshold = self._sort_skip_threshold_sq
                max_consec = self._MAX_SORT_SKIP_FRAMES
            else:
                threshold = self._sort_skip_threshold_sq_tight
                max_consec = self._MAX_SORT_SKIP_FRAMES_TIGHT
            can_skip = (max_disp_sq < threshold
                        and self._sort_skip_consecutive < max_consec)
            if can_skip:
                self._sort_skip_next = True
                self._sort_skip_consecutive += 1
            else:
                self._sort_skip_next = False
                self._sort_skip_consecutive = 0

            # Latch the pinned values from the PREVIOUS frame's async copy
            # (by now that copy has completed — a full frame of GPU work elapsed)
            self._last_max_disp_uint = _ctypes.c_uint32.from_address(
                self._pinned_displacement.ptr).value
            self._last_foam_count = _ctypes.c_uint32.from_address(
                self._pinned_foam_count.ptr).value

            # Record event on default stream so readback stream waits for kernels
            self._readback_event.record()
            self._readback_stream.wait_event(self._readback_event)
            # Launch async D2H copy of THIS frame's data for NEXT frame's decision
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
        # else: sim_steps==0 — keep previous _sort_skip_next

        # Periodic compaction: remove dead particles from the active range
        if sim_steps > 0:
            n = self._maybe_compact()

        self._last_substeps = sim_steps
        return sim_steps

    def _maybe_compact(self) -> int:
        """Run compaction if due and enough dead particles exist.

        Returns the current _high_water (possibly reduced by compaction).
        """
        n = self.world._high_water
        if n == 0 or self.compact_interval <= 0:
            return n

        if self._frame_counter % self.compact_interval != 0:
            return n

        # Count dead particles = _high_water - num_active. This is the only
        # per-frame caller of num_active, and it runs only once every
        # compact_interval frames (gated above), so the ~1 Hz sync is fine.
        num_alive = self.world.num_active
        num_dead = n - num_alive
        if num_dead < self.compact_threshold:
            return n

        # Reuse the count we already paid for so compact() does not sync again.
        self.world.compact(num_alive=num_alive)
        # After compaction, n changed — force full sort next time
        self._sort_skip_next = False
        self._sort_skip_consecutive = 0
        return self.world._high_water

    def copy_to_vbos(self, cuda_pos, cuda_col, cuda_vel=None) -> None:
        """Copy UNSORTED pos/color/velocity arrays into mapped GL VBOs.

        Parameters
        ----------
        cuda_pos : CudaGLBuffer
            Position VBO wrapper (must be context-managed externally).
        cuda_col : CudaGLBuffer
            Color VBO wrapper (must be context-managed externally).
        cuda_vel : CudaGLBuffer, optional
            Velocity VBO wrapper for motion blur.
        """
        n = self.world._high_water
        if n == 0:
            return

        # Position: world pos (float4) -> VBO pos (float4)
        pos_arr = cuda_pos.device_pointer_as_cupy_array(
            (cuda_pos.nbytes // 16, 4), np.float32,
        )
        pos_arr[:n] = self.world.position[:n]

        # Color: world color (float4) -> VBO color (float4)
        col_arr = cuda_col.device_pointer_as_cupy_array(
            (cuda_col.nbytes // 16, 4), np.float32,
        )
        col_arr[:n] = self.world.color[:n]

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
        """Adjust accuracy (CFL number) by delta, clamping to [0.1, 1.0]."""
        self.accuracy = max(0.1, min(1.0, self.accuracy + delta))

    def reset_spawn_damping(self) -> None:
        """Reset spawn damping counter (call after preset load or world reset)."""
        self._spawn_substep = 0

    def toggle_fixed_dt(self) -> None:
        """Toggle between adaptive and fixed timestep modes."""
        self.fixed_dt = not self.fixed_dt
        if self.fixed_dt:
            # Switch to fixed mode: use fixed_dt_value
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
            self._readback_stream = None
        except Exception:
            pass

        # 4. Release pinned host memory.  CuPy's MemoryPointer.__del__ calls
        #    cudaFreeHost, but we trigger it now to avoid keeping pinned pages
        #    around until the next GC cycle.
        try:
            self._pinned_displacement = None
            self._pinned_foam_count = None
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
            hash_sort.get_module(),
            fused_sort_reorder_build.get_module(),
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
