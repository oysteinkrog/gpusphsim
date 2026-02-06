"""SPH simulation orchestrator.

Wires up the full SPH pipeline: Hash -> Sort -> Reorder -> Build -> Step1 ->
Step2 -> Integrate.  Decouples simulation from rendering via a speed parameter,
accuracy parameter (CFL number), and substep budget.

Each render frame:
  1. If adaptive mode: compute dt from CFL constraints (advection, acoustic, viscous)
  2. Compute sim_steps = clamp(round(speed * wall_dt / sim_dt), 0, max_substeps)
  3. Run sim_steps simulation substeps
  4. Copy UNSORTED pos/color to mapped VBOs (rendering happens once per frame)
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cupy as cp
import numpy as np

from world import World
from materials import (
    WATER, SAND, MATERIALS, GRANULAR, GAS, STATIC,
    build_material_array, build_interaction_matrix, upload_to_gpu,
)

import hash_sort
import fused_reorder
import build_grid
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

        # Precompute acoustic speed from materials table
        self._c_sound = self._compute_c_sound()
        # Precompute max viscosity and min density for viscous CFL
        self._max_eta, self._rho_min = self._compute_viscosity_bounds()

        # Smoothing length (must match _upload_constants)
        self._h = 0.04

        # Grid cell tables (pre-allocated, reused every frame)
        self._cell_start, self._cell_end = build_grid.allocate_cell_tables()

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

        For GRANULAR materials, uses mu_max (10000) as worst-case eta.
        For FLUID/GAS, uses base_viscosity from the material table.
        rho_min excludes GAS materials (they use low constant viscosity,
        not mu(I), so their low density doesn't tighten the viscous CFL).
        """
        max_eta = 0.0
        rho_min = 1e30
        has_granular = False
        for mat in MATERIALS.values():
            if mat.behavior_class == STATIC or mat.rest_density <= 0.0:
                continue
            # rho_min for viscous CFL: exclude GAS (never runs mu(I))
            if mat.behavior_class != GAS and mat.rest_density < rho_min:
                rho_min = mat.rest_density
            if mat.behavior_class == GRANULAR:
                has_granular = True
            if mat.base_viscosity > max_eta:
                max_eta = mat.base_viscosity
        if has_granular:
            max_eta = max(max_eta, float(step2.DEFAULT_MU_MAX))
        rho_min = max(rho_min, 0.1)  # floor
        return max_eta, rho_min

    def _compute_adaptive_dt(self, n: int) -> float:
        """Compute adaptive dt from CFL constraints.

        Three constraints:
          dt_advection = accuracy * h / max(max_v, 1e-6)
          dt_acoustic  = 0.25 * h / c_sound
          dt_viscous   = accuracy * rho_min * h^2 / max(max_eta, 1e-6)
          dt = min(dt_adv, dt_acou, dt_visc) clamped to [DT_MIN, DT_MAX]
        """
        h = self._h

        # Max velocity magnitude via CuPy reduction
        vel = self.world.velocity[:n]  # (n, 4) float32
        vel_sq = vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2
        max_v = float(cp.sqrt(cp.max(vel_sq)))

        # Advection CFL
        dt_advection = self.accuracy * h / max(max_v, 1e-6)

        # Acoustic CFL (fixed 0.25 coefficient, not scaled by accuracy)
        dt_acoustic = 0.25 * h / self._c_sound

        # Viscous CFL (uses precomputed max_eta and rho_min)
        dt_viscous = self.accuracy * self._rho_min * h * h / max(self._max_eta, 1e-6)

        dt = min(dt_advection, dt_acoustic, dt_viscous)
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
            particle_mass=0.008,
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
            viscosity=1.0,
        )
        granular_params = step2.build_granular_params()
        materials_data = build_material_array()
        interactions_data = build_interaction_matrix()

        # --- hash_sort module: c_grid ---
        hash_sort.upload_grid_params(grid_params)

        # --- fused_reorder module: no constant memory needed ---
        fused_reorder.get_module()  # just compile

        # --- build_grid module: c_grid ---
        build_grid.upload_grid_params(grid_params)

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

    def _sim_step(self, n: int) -> None:
        """Run one simulation substep on n particles."""
        w = self.world

        # 1. Hash particle positions
        hashes, indices = hash_sort.calc_hash(w.position[:n])

        # 2. Sort by hash (argsort -> gather sorted hashes and indices)
        sorted_hashes, sorted_indices = hash_sort.sort_by_hash(
            hashes, indices,
            sorted_hashes_out=w.sorted_hashes,
            sorted_indices_out=w.sorted_indices,
        )

        # 3. Fused reorder: gather all unsorted arrays into sorted order
        fused_reorder.fused_reorder(
            n, sorted_indices,
            w.position, w.velocity, w.veleval,
            w.mass, w.packed_info, w.temperature,
            w.health, w.lifetime, w.color,
            w.sleep_counter, w.shear_rate,
            w.sorted_position, w.sorted_velocity, w.sorted_veleval,
            w.sorted_mass, w.sorted_packed_info, w.sorted_temperature,
            w.sorted_health, w.sorted_lifetime, w.sorted_color,
            w.sorted_sleep_counter, w.sorted_shear_rate,
        )

        # 4. Build grid data structure (cell_start / cell_end tables)
        build_grid.build_data_struct(
            sorted_hashes[:n], self._cell_start, self._cell_end
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
        w._density_initialized = True

        # 5b. Reset freelist counter for this step
        spawn.reset_freelist(self._dead_count)

        # 5c. Reactions: phase transitions, combustion, corrosion, gas lifetime
        #     Runs on sorted arrays, modifies packed_info/temperature/health/
        #     lifetime/velocity in-place before Step2 sees them.
        #     Also populates the freelist when particles die.
        reactions.compute_reactions(
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            w.sorted_health[:n],
            w.sorted_lifetime[:n],
            w.sorted_velocity[:n],
            w.sorted_exposure_heat[:n],
            w.sorted_exposure_corrode[:n],
            frame=self._frame_counter,
            dead_indices=self._dead_indices,
            dead_count=self._dead_count,
        )

        # 5d. Spawn: consume from freelist to spawn steam from boiling water
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

        # 7. Integrate: symplectic Euler + SDF boundaries + color + temperature update
        #    Writes back to UNSORTED arrays via sort_indexes
        #    Also updates packed_info (sleep flag), sleep_counter, and temperature
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
            sort_indexes=sorted_indices[:n],
            position_out=w.position,
            velocity_out=w.velocity,
            color_out=w.color,
            packed_info_out=w.packed_info,
            sleep_counter_out=w.sleep_counter,
            temperature_out=w.temperature,
        )

        # 8. Wake propagation: mark cells near just-woke particles,
        #    wake sleeping neighbors, clear JUST_WOKE flags
        wake.run_wake_propagation(
            w.position[:n],
            w.packed_info[:n],
            w.sleep_counter[:n],
            self._cell_wake_flags,
            num_particles=n,
        )

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

        self._last_substeps = sim_steps
        return sim_steps

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
