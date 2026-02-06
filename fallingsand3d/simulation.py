"""SPH simulation orchestrator.

Wires up the full SPH pipeline: Hash -> Sort -> Reorder -> Build -> Step1 ->
Step2 -> Integrate.  Decouples simulation from rendering via a speed parameter
and substep budget.

Each render frame:
  1. Compute sim_steps = clamp(round(speed * wall_dt / sim_dt), 0, max_substeps)
  2. Run sim_steps simulation substeps
  3. Copy UNSORTED pos/color to mapped VBOs (rendering happens once per frame)
"""

from __future__ import annotations

import time
from typing import Optional

import cupy as cp
import numpy as np

from world import World
from materials import (
    WATER, SAND, build_material_array, build_interaction_matrix, upload_to_gpu,
)

import hash_sort
import fused_reorder
import build_grid
import step1
import step2
import integrate


class Simulation:
    """Orchestrates the full SPH pipeline with sim/render decoupling.

    Parameters
    ----------
    world : World
        Particle state manager.
    dt : float
        Fixed simulation timestep (seconds).
    speed : float
        Time scale: 1.0 = realtime, <1 = slow-mo, >1 = fast-forward.
    max_substeps : int
        Maximum substeps per render frame to prevent spiral-of-death.
    """

    def __init__(
        self,
        world: World,
        dt: float = 0.001,
        speed: float = 1.0,
        max_substeps: int = 20,
    ) -> None:
        self.world = world
        self.dt = dt
        self.speed = speed
        self.max_substeps = max_substeps
        self._paused = False
        self._prev_speed = speed
        self._last_frame_time: Optional[float] = None
        self._last_substeps = 0
        self.sim_time = 0.0

        # Grid cell tables (pre-allocated, reused every frame)
        self._cell_start, self._cell_end = build_grid.allocate_cell_tables()

        # Compile all kernel modules and upload constant memory
        self._upload_constants()

    def _upload_constants(self) -> None:
        """Compile all .cu kernels and upload constant memory to each module."""
        # Build shared param structs
        grid_params = hash_sort.build_grid_params()
        sim_params = step1.build_sim_params(
            smoothing_length=0.04,
            particle_mass=0.008,
            particle_spacing=0.02,
            gravity=(0.0, -9.8, 0.0),
            dt=self.dt,
            restitution=0.3,
            wall_friction=0.5,
            world_min=(-1.0, -1.0, -1.0),
            world_max=(1.0, 1.0, 1.0),
        )
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

        # --- step1 module: c_grid, c_sim, c_precalc ---
        step1.upload_grid_params(grid_params)
        step1.upload_sim_params(sim_params)
        step1.upload_precalc_params(precalc_params)

        # --- step2 module: c_grid, c_sim, c_precalc, c_materials, c_granular ---
        step2.upload_grid_params(grid_params)
        step2.upload_sim_params(sim_params)
        step2.upload_precalc_params(precalc_params)
        step2.upload_materials(materials_data)
        step2.upload_granular_params(granular_params)

        # --- integrate module: c_sim, c_materials ---
        integrate.upload_sim_params(sim_params)
        integrate.upload_materials(materials_data)

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

        # 5. Step1: density summation
        step1.compute_step1(
            w.sorted_position[:n],
            w.sorted_mass[:n],
            self._cell_start,
            self._cell_end,
            density_out=w.sorted_density,
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

        # 7. Integrate: symplectic Euler + SDF boundaries + color update
        #    Writes back to UNSORTED arrays via sort_indexes
        integrate.integrate(
            w.sorted_position[:n],
            w.sorted_velocity[:n],
            w.sorted_veleval[:n],
            w.sorted_sph_force[:n],
            w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            w.sorted_health[:n],
            sorted_indices[:n],
            position_out=w.position,
            velocity_out=w.velocity,
            color_out=w.color,
        )

        self.sim_time += self.dt

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

    @property
    def last_substeps(self) -> int:
        return self._last_substeps
