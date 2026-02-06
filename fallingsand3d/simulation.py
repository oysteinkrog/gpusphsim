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

import cupy

import hash_sort
import fused_sort_reorder_build
import step1
import step2
import integrate
import reactions
import spawn
import wake

# Adaptive timestep limits
DT_MIN = 1e-5
DT_MAX = 0.005


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
        max_substeps: int = 20,
        compact_interval: int = 60,
        compact_threshold: int = 100,
    ) -> None:
        self.world = world
        self.dt = dt
        self.speed = speed
        self.accuracy = accuracy
        self.fixed_dt = fixed_dt
        self.fixed_dt_value = dt
        self.max_substeps = max_substeps
        self._paused = False
        self._prev_speed = speed
        self._last_frame_time: Optional[float] = None
        self._last_substeps = 0
        self.sim_time = 0.0
        self._frame_counter = 0
        self.compact_interval = compact_interval
        self.compact_threshold = compact_threshold
        # --- CUDA graph capture state ---
        self._sort_perm = cupy.empty(world.max_particles, dtype=cupy.uint32)
        self._frame_counter_d = cupy.zeros(1, dtype=cupy.uint32)
        self._cuda_graph = None   # cupy.cuda.Graph or None
        self._graph_n = 0         # exact n used when graph was captured

        # Precompute acoustic speed from materials table
        self._c_sound = self._compute_c_sound()
        # Precompute max viscosity and min density for viscous CFL
        self._max_eta, self._rho_min = self._compute_viscosity_bounds()

        # Smoothing length (must match _upload_constants)
        self._h = 0.04

        # Grid cell tables (pre-allocated, reused every frame)
        from hash_sort import NUM_CELLS
        self._cell_start = cupy.empty(NUM_CELLS, dtype=cupy.uint32)
        self._cell_end = cupy.empty(NUM_CELLS, dtype=cupy.uint32)

        # Cell wake flags for wake propagation (pre-allocated, cleared each step)
        self._cell_wake_flags = wake.allocate_cell_wake_flags()

        # Freelist for spawn/kill system (pre-allocated, reset each step)
        self._dead_indices, self._dead_count = spawn.allocate_freelist(
            world.max_particles
        )

        # Cached sim_params for dt updates (set during _upload_constants)
        self._sim_params: Optional[np.ndarray] = None

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
        self.dt = new_dt
        # Upload to all modules that read c_sim.dt
        step1.upload_sim_params(self._sim_params)
        step2.upload_sim_params(self._sim_params)
        integrate.upload_sim_params(self._sim_params)
        reactions.upload_sim_params(self._sim_params)
        spawn.upload_sim_params(self._sim_params)

    def _upload_constants(self) -> None:
        """Compile all .cu kernels and upload constant memory to each module."""
        # Build shared param structs
        grid_params = hash_sort.build_grid_params()
        sim_params = step1.build_sim_params(
            smoothing_length=self._h,
            particle_mass=0.02,
            particle_spacing=0.02,
            gravity=(0.0, -9.8, 0.0),
            dt=self.dt,
            restitution=0.3,
            wall_friction=0.5,
            world_min=(-1.0, -1.0, -1.0),
            world_max=(1.0, 1.0, 1.0),
        )
        self._sim_params = sim_params
        precalc_params = step1.build_precalc_params(
            smoothing_length=0.04,
            viscosity=0.1,
        )
        granular_params = step2.build_granular_params()
        materials_data = build_material_array()
        interactions_data = build_interaction_matrix()

        # --- hash_sort module: c_grid ---
        hash_sort.upload_grid_params(grid_params)

        # --- fused_sort_reorder_build module: c_grid ---
        fused_sort_reorder_build.upload_grid_params(grid_params)

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

        # --- reactions module: c_sim, c_materials ---
        reactions.upload_sim_params(sim_params)
        reactions.upload_materials(materials_data)

        # --- spawn module: c_sim, c_materials ---
        spawn.upload_sim_params(sim_params)
        spawn.upload_materials(materials_data)

        # --- wake module: c_grid ---
        wake.upload_grid_params(grid_params)

        # --- materials module (its own internal module): c_materials, c_interactions ---
        upload_to_gpu()

    def _run_graph_body(self, n: int) -> None:
        """Execute the graph-capturable portion of the pipeline (ops 3-14).

        This method is called both during graph capture (on capture stream)
        and can be called directly for debugging. All operations use
        pre-allocated arrays with stable pointers.
        """
        w = self.world
        sort_perm = self._sort_perm[:n]

        # 3. Memset cell tables before fused kernel (async for graph capture)
        self._cell_start.data.memset_async(0xFF, self._cell_start.nbytes)
        self._cell_end.data.memset_async(0x00, self._cell_end.nbytes)

        # 4. Fused sort-reorder-build
        fused_sort_reorder_build.fused_sort_reorder_build(
            n, sort_perm, w.hashes,
            w.sorted_hashes, self._cell_start, self._cell_end,
            w.position, w.velocity,
            w.mass, w.packed_info, w.temperature,
            w.health, w.lifetime,
            w.sleep_counter,
            w.sorted_position, w.sorted_velocity,
            w.sorted_mass, w.sorted_packed_info, w.sorted_temperature,
            w.sorted_health, w.sorted_lifetime,
            w.sorted_sleep_counter,
        )

        # 5. Step1: density summation + strain-rate tensor + heat diffusion + exposure
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
        )

        # 5b. Reset freelist counter (graph-safe memset)
        spawn.reset_freelist(self._dead_count)

        # 5c. Reactions (always included — zero-cost early-exit for non-reactive)
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

        # 5d. Spawn (always included — zero-cost when no SPAWN_GAS flags set)
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

        # 6. Step2: pressure + viscosity + XSPH forces
        step2.compute_step2(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_density[:n],
            w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            self._cell_start,
            self._cell_end,
            sph_force_out=w.sorted_sph_force,
            veleval_out=w.sorted_veleval,
        )

        # 7. Integrate
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
            sort_indexes=self._sort_perm[:n],
            position_out=w.position,
            velocity_out=w.velocity,
            color_out=w.color,
            packed_info_out=w.packed_info,
            sleep_counter_out=w.sleep_counter,
            temperature_out=w.temperature,
        )

        # 8. Wake propagation (memset + 2 kernels)
        wake.run_wake_propagation(
            w.position[:n],
            w.packed_info[:n],
            w.sleep_counter[:n],
            self._cell_wake_flags,
            num_particles=n,
        )

    def _capture_graph(self, n: int) -> None:
        """Capture the graph-body into a CUDA graph for replay.

        Creates a non-default capture stream, records all ops 3-14, and
        stores the resulting Graph object for repeated launch().
        """
        s = cupy.cuda.Stream(non_blocking=True)
        with s:
            s.begin_capture()
            self._run_graph_body(n)
            self._cuda_graph = s.end_capture()
        self._graph_n = n

    def _sim_step(self, n: int) -> None:
        """Run one simulation substep on n particles.

        Pipeline:
          1. K_CalcHash           — normal launch
          2. cupy.argsort + copy  — normal launch (Thrust can't be captured)
          3-14. CUDA graph launch — single dispatch for all remaining ops

        Graph strategy: when n changes (brush painting, compaction), fall back
        to direct kernel launches — no graph capture overhead. When n is stable
        across substeps, capture a graph and replay it for subsequent calls.
        """
        w = self.world

        # 1. Hash particle positions (normal launch)
        hashes = hash_sort.calc_hash(w.position[:n], hashes_out=w.hashes)

        # 2. Argsort (normal — Thrust uses internal cudaStreamSynchronize)
        sort_perm = cupy.argsort(hashes).astype(cupy.uint32)

        # Copy into pre-allocated buffer (stable pointer for graph)
        self._sort_perm[:n] = sort_perm

        # Update device frame counter (value changes, pointer stays stable)
        self._frame_counter_d.fill(self._frame_counter)

        # Mark density as initialized (needed for step1 prev_density path)
        w._density_initialized = True

        # 3-14. Graph capture/replay
        if self._graph_n != n:
            # n changed — invalidate graph, run directly (no capture overhead)
            self._cuda_graph = None
            self._graph_n = n
            self._run_graph_body(n)
        elif self._cuda_graph is None:
            # n stable but no graph yet — capture then launch
            # (capture records but does NOT execute)
            self._capture_graph(n)
            self._cuda_graph.launch()
        else:
            # n stable, graph exists — fast replay
            self._cuda_graph.launch()

        self.sim_time += self.dt
        self._frame_counter += 1

    def step_frame(self) -> int:
        """Advance simulation for one render frame. Returns number of substeps run."""
        now = time.perf_counter()
        if self._last_frame_time is None:
            self._last_frame_time = now
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

        # Compute number of substeps: sim_steps = clamp(round(speed * wall_dt / sim_dt), 0, max)
        sim_steps = max(0, min(
            self.max_substeps,
            round(self.speed * wall_dt / self.dt),
        ))

        for _ in range(sim_steps):
            self._sim_step(n)

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

        # Count dead particles = _high_water - num_active
        num_alive = self.world.num_active
        num_dead = n - num_alive
        if num_dead < self.compact_threshold:
            return n

        self.world.compact()
        return self.world._high_water

    def copy_to_vbos(self, cuda_pos, cuda_col) -> None:
        """Copy UNSORTED pos/color arrays into mapped GL VBOs.

        Parameters
        ----------
        cuda_pos : CudaGLBuffer
            Position VBO wrapper (must be context-managed externally).
        cuda_col : CudaGLBuffer
            Color VBO wrapper (must be context-managed externally).
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

    def toggle_fixed_dt(self) -> None:
        """Toggle between adaptive and fixed timestep modes."""
        self.fixed_dt = not self.fixed_dt
        if self.fixed_dt:
            # Switch to fixed mode: use fixed_dt_value
            self._upload_dt(self.fixed_dt_value)

    @property
    def last_substeps(self) -> int:
        return self._last_substeps
