"""SPH Simulation orchestrator -- full pipeline per step.

Pipeline per simulation step (canonical ordering):
  1. Hash        -- K_CalcHash: position -> cell hash + index
  2. Argsort     -- CuPy argsort on hashes (carries sort_indexes)
  3. FusedReorder-- K_FusedReorder: gather unsorted -> sorted arrays
  4. Build       -- K_BuildDataStruct: cell start/end from sorted hashes
  5. Step1       -- K_Step1: density summation (Poly6)
  6. Step2       -- K_Step2: pressure + viscosity + XSPH forces
  7. Integrate   -- K_Integrate: leapfrog, boundaries, coloring, writeback

Sim/render decoupling:
  Each render frame computes sim_steps = clamp(round(speed * wall_dt / sim_dt), 0, max_substeps)
  then runs that many simulation steps before rendering.
"""

from __future__ import annotations

from typing import Optional

import cupy  # type: ignore[import-untyped]
import numpy as np

import build_grid
import fused_reorder
import hash_sort
import integrate
import materials
import step1
import step2
from hash_sort import NUM_CELLS, build_grid_params
from integrate import build_integrate_params
from step2 import (
    BEHAVIOR_FLUID,
    BEHAVIOR_GRANULAR,
    build_fluid_params,
    build_precalc_params,
)

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

DEFAULT_DT = 0.001  # Fixed timestep (adaptive comes in US-025)
DEFAULT_SPEED = 1.0  # Realtime
MAX_SUBSTEPS = 20  # Spiral-of-death cap


class SPHSimulation:
    """Full SPH simulation state and pipeline.

    Parameters
    ----------
    num_particles : int
        Total number of particles.
    dt : float
        Fixed simulation timestep.
    speed : float
        Simulation speed multiplier (1.0=realtime, 0.2=slow-mo, 5.0=fast-forward).
    """

    def __init__(
        self,
        num_particles: int,
        dt: float = DEFAULT_DT,
        speed: float = DEFAULT_SPEED,
    ) -> None:
        self.num_particles = num_particles
        self.dt = dt
        self.speed = speed
        self.paused = False
        self.sim_time = 0.0

        # --- Allocate unsorted particle arrays (world space, original order) ---
        self.position = cupy.zeros((num_particles, 4), dtype=cupy.float32)
        self.velocity = cupy.zeros((num_particles, 4), dtype=cupy.float32)
        self.veleval = cupy.zeros((num_particles, 4), dtype=cupy.float32)
        self.color = cupy.ones((num_particles, 4), dtype=cupy.float32)
        self.behavior_class = cupy.ones(num_particles, dtype=cupy.int32)  # default FLUID
        self.flags = cupy.zeros(num_particles, dtype=cupy.uint32)

        # --- Allocate sorted particle arrays (grid order) ---
        self.position_sorted = cupy.zeros((num_particles, 4), dtype=cupy.float32)
        self.velocity_sorted = cupy.zeros((num_particles, 4), dtype=cupy.float32)
        self.veleval_sorted = cupy.zeros((num_particles, 4), dtype=cupy.float32)
        self.behavior_class_sorted = cupy.ones(num_particles, dtype=cupy.int32)
        self.flags_sorted = cupy.zeros(num_particles, dtype=cupy.uint32)
        self.sph_force_sorted = cupy.zeros((num_particles, 4), dtype=cupy.float32)
        self.density_sorted = cupy.zeros(num_particles, dtype=cupy.float32)

        # --- Allocate grid arrays ---
        self.cell_start = cupy.empty(NUM_CELLS, dtype=cupy.uint32)
        self.cell_end = cupy.empty(NUM_CELLS, dtype=cupy.uint32)

        # --- Upload constant memory to all kernel modules ---
        self._upload_constants()

    def _upload_constants(self) -> None:
        """Upload GridParams, FluidParams, PrecalcParams, IntegrateParams
        to all kernel modules' constant memory."""
        grid_params = build_grid_params()
        fluid_params = build_fluid_params()
        precalc_params = build_precalc_params()
        integrate_params = build_integrate_params(delta_time=self.dt)

        # Hash kernel: GridParams
        hash_sort.upload_grid_params(grid_params)

        # Build kernel: GridParams
        build_grid.upload_grid_params(grid_params)

        # Step1 kernel: GridParams + FluidParams + PrecalcParams
        step1.upload_grid_params(grid_params)
        step1.upload_fluid_params(fluid_params)
        step1.upload_precalc_params(precalc_params)

        # Step2 kernel: GridParams + FluidParams + PrecalcParams
        step2.upload_grid_params(grid_params)
        step2.upload_fluid_params(fluid_params)
        step2.upload_precalc_params(precalc_params)

        # Integrate kernel: GridParams + IntegrateParams
        integrate.upload_grid_params(grid_params)
        integrate.upload_integrate_params(integrate_params)

        # Materials kernel: MaterialProps[32] + Interaction[32][32]
        materials.upload_to_gpu()

    def _simulation_step(self) -> None:
        """Execute one full simulation step (the canonical pipeline)."""
        n = self.num_particles

        # 1. Hash -- compute cell hashes from unsorted positions
        hashes, indices = hash_sort.calc_hash(self.position)

        # 2. Argsort -- sort by hash, get permutation
        sort_order = cupy.argsort(hashes).astype(cupy.uint32)
        sorted_hashes = hashes[sort_order]
        sort_indexes = indices[sort_order]  # sort_indexes[sorted] = original

        # 3. Fused reorder -- gather unsorted -> sorted arrays
        fused_reorder.fused_reorder(
            sort_indexes,
            self.position,
            self.velocity,
            self.veleval,
            self.behavior_class,
            self.flags,
            self.position_sorted,
            self.velocity_sorted,
            self.veleval_sorted,
            self.behavior_class_sorted,
            self.flags_sorted,
        )

        # 4. Build -- cell start/end from sorted hashes
        build_grid.build_data_struct(
            sorted_hashes,
            self.cell_start,
            self.cell_end,
        )

        # 5. Step1 -- density summation
        self.density_sorted = step1.compute_step1(
            self.position_sorted,
            self.cell_start,
            self.cell_end,
        )

        # 6. Step2 -- pressure + viscosity + XSPH forces
        self.sph_force_sorted, veleval_corrected = step2.compute_step2(
            self.position_sorted,
            self.veleval_sorted,
            self.density_sorted,
            self.behavior_class_sorted,
            self.flags_sorted,
            self.cell_start,
            self.cell_end,
        )
        self.veleval_sorted = veleval_corrected

        # 7. Integrate -- leapfrog, boundaries, coloring, writeback to unsorted
        integrate.integrate(
            self.position_sorted,
            self.velocity_sorted,
            self.veleval_sorted,
            self.sph_force_sorted,
            self.density_sorted,
            self.behavior_class_sorted,
            sort_indexes,
            self.position,
            self.velocity,
            self.veleval,
            self.color,
        )

        self.sim_time += self.dt

    def step_frame(self, wall_dt: float) -> int:
        """Run simulation substeps for one render frame.

        Parameters
        ----------
        wall_dt : float
            Wall-clock time since last frame (seconds).

        Returns
        -------
        int
            Number of substeps executed.
        """
        if self.paused:
            return 0

        # Compute number of substeps: sim_time_budget = wall_dt * speed
        sim_budget = wall_dt * self.speed
        num_steps = int(round(sim_budget / self.dt))
        num_steps = max(0, min(num_steps, MAX_SUBSTEPS))

        for _ in range(num_steps):
            self._simulation_step()

        return num_steps

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused

    def adjust_speed(self, delta: float) -> None:
        """Adjust simulation speed by delta, clamped to [0.1, 10.0]."""
        self.speed = max(0.1, min(self.speed + delta, 10.0))


# ---------------------------------------------------------------------------
# Initial scene setup
# ---------------------------------------------------------------------------


def create_particle_block(
    center: tuple[float, float, float],
    half_extent: tuple[float, float, float],
    count: int,
    behavior: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create a block of particles uniformly distributed in a box.

    Parameters
    ----------
    center : (x, y, z)
    half_extent : (hx, hy, hz) -- half-size of the box
    count : int -- number of particles
    behavior : int -- BEHAVIOR_FLUID, BEHAVIOR_GRANULAR, etc.

    Returns
    -------
    positions : (count, 4) float32 -- xyz + w=0
    behavior_classes : (count,) int32
    count : int -- actual count
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pos = np.zeros((count, 4), dtype=np.float32)
    pos[:, 0] = rng.uniform(
        center[0] - half_extent[0], center[0] + half_extent[0], count
    ).astype(np.float32)
    pos[:, 1] = rng.uniform(
        center[1] - half_extent[1], center[1] + half_extent[1], count
    ).astype(np.float32)
    pos[:, 2] = rng.uniform(
        center[2] - half_extent[2], center[2] + half_extent[2], count
    ).astype(np.float32)
    # w = 0

    bclass = np.full(count, behavior, dtype=np.int32)
    return pos, bclass, count


def setup_initial_scene(sim: SPHSimulation) -> None:
    """Set up the initial scene: 10K water cube + 10K sand bed.

    Water: cube at (0, 0.5, 0), size 0.4 (half-extent 0.2)
    Sand: flat bed at y=-0.5 to y=-0.3, x/z spanning -0.8 to 0.8
    """
    rng = np.random.default_rng(42)

    # Water particles: cube at (0, 0.5, 0), half-extent 0.2
    water_n = sim.num_particles // 2
    water_pos, water_bclass, _ = create_particle_block(
        center=(0.0, 0.5, 0.0),
        half_extent=(0.2, 0.2, 0.2),
        count=water_n,
        behavior=BEHAVIOR_FLUID,
        rng=rng,
    )

    # Sand particles: flat bed y in [-0.5, -0.3], x/z in [-0.8, 0.8]
    sand_n = sim.num_particles - water_n
    sand_pos, sand_bclass, _ = create_particle_block(
        center=(0.0, -0.4, 0.0),
        half_extent=(0.8, 0.1, 0.8),
        count=sand_n,
        behavior=BEHAVIOR_GRANULAR,
        rng=rng,
    )

    # Combine and upload
    all_pos = np.concatenate([water_pos, sand_pos], axis=0)
    all_bclass = np.concatenate([water_bclass, sand_bclass], axis=0)

    sim.position[:] = cupy.asarray(all_pos)
    sim.behavior_class[:] = cupy.asarray(all_bclass)

    # Initial velocity and veleval are zero (already initialized)
    # Set initial colors: water = blue, sand = brown
    colors = np.ones((sim.num_particles, 4), dtype=np.float32)
    colors[:water_n, :3] = [0.2, 0.4, 1.0]  # blue
    colors[water_n:, :3] = [0.76, 0.60, 0.42]  # sand/brown
    sim.color[:] = cupy.asarray(colors)
