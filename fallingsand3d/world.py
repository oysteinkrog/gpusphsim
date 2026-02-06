"""World state management for Falling Sand 3D.

Manages CuPy arrays for all per-particle SoA data. Supports configurable
max_particles at startup, spawning particles in sphere/cube shapes, and
tracking num_active via counter.

All arrays are contiguous CuPy float32/uint32/uint8 arrays on GPU.
DEAD particles have packed_info=0 (material_id=0, behavior=STATIC).
"""

from __future__ import annotations

from typing import Tuple

import cupy as cp
import numpy as np

from materials import MATERIALS, FLUID, GRANULAR, GAS, STATIC

# packed_info bitfield macros (mirrors common.cuh)
_MAKE_PACKED = lambda mat, beh: (int(mat) & 0xFF) | ((int(beh) & 0x3) << 8)

# Default particle spacing (h=0.04 -> spacing=h/2=0.02)
DEFAULT_SPACING = 0.02

# Default temperatures for hot materials (Kelvin)
_DEFAULT_TEMPS = {
    7: 1500.0,   # LAVA
    14: 1200.0,  # FIRE
    12: 373.0,   # STEAM
    13: 500.0,   # SMOKE
}

# Ambient temperature
T_AMBIENT = 293.0


class World:
    """Manages all per-particle SoA arrays on GPU.

    Parameters
    ----------
    max_particles : int
        Maximum number of particles to allocate for (default 500000).
    """

    def __init__(self, max_particles: int = 500_000) -> None:
        self.max_particles: int = max_particles
        self._high_water: int = 0  # one past last allocated slot
        self._allocate()

    def _allocate(self) -> None:
        """Allocate all SoA arrays as CuPy arrays of max_particles size."""
        n = self.max_particles
        # float4 arrays (stored as (n, 4) float32)
        self.position = cp.zeros((n, 4), dtype=cp.float32)
        self.velocity = cp.zeros((n, 4), dtype=cp.float32)
        self.veleval = cp.zeros((n, 4), dtype=cp.float32)
        self.sph_force = cp.zeros((n, 4), dtype=cp.float32)
        self.color = cp.zeros((n, 4), dtype=cp.float32)
        # float arrays
        self.density = cp.zeros(n, dtype=cp.float32)
        self.mass = cp.zeros(n, dtype=cp.float32)
        self.temperature = cp.zeros(n, dtype=cp.float32)
        self.health = cp.zeros(n, dtype=cp.float32)
        self.lifetime = cp.zeros(n, dtype=cp.float32)
        self.shear_rate = cp.zeros(n, dtype=cp.float32)
        self.exposure_heat = cp.zeros(n, dtype=cp.float32)
        self.exposure_corrode = cp.zeros(n, dtype=cp.float32)
        # uint32 arrays
        self.packed_info = cp.zeros(n, dtype=cp.uint32)
        # uint8 arrays
        self.sleep_counter = cp.zeros(n, dtype=cp.uint8)

        # --- Sorted temporary buffers (pre-allocated, reused every frame) ---
        # These are the double-buffer targets: after sort+reorder, physics
        # kernels (Step1/Step2) read from these.  Integrate writes back to
        # the unsorted arrays above via sort_indexes.
        self.sorted_position = cp.zeros((n, 4), dtype=cp.float32)
        self.sorted_velocity = cp.zeros((n, 4), dtype=cp.float32)
        self.sorted_veleval = cp.zeros((n, 4), dtype=cp.float32)
        self.sorted_sph_force = cp.zeros((n, 4), dtype=cp.float32)
        self.sorted_color = cp.zeros((n, 4), dtype=cp.float32)
        self.sorted_density = cp.zeros(n, dtype=cp.float32)
        self.sorted_mass = cp.zeros(n, dtype=cp.float32)
        self.sorted_temperature = cp.zeros(n, dtype=cp.float32)
        self.sorted_health = cp.zeros(n, dtype=cp.float32)
        self.sorted_lifetime = cp.zeros(n, dtype=cp.float32)
        self.sorted_shear_rate = cp.zeros(n, dtype=cp.float32)
        self.sorted_dTdt = cp.zeros(n, dtype=cp.float32)
        self.sorted_packed_info = cp.zeros(n, dtype=cp.uint32)
        self.sorted_sleep_counter = cp.zeros(n, dtype=cp.uint8)
        # Sort index arrays (hash + original index, plus sorted versions)
        self.hashes = cp.zeros(n, dtype=cp.uint32)
        self.indices = cp.zeros(n, dtype=cp.uint32)
        self.sorted_hashes = cp.zeros(n, dtype=cp.uint32)
        self.sorted_indices = cp.zeros(n, dtype=cp.uint32)

    def resize(self, new_max: int) -> None:
        """Reallocate all arrays for a new max_particles. Kills all particles."""
        self.max_particles = new_max
        self._high_water = 0
        self._allocate()

    @property
    def num_active(self) -> int:
        """Count of non-DEAD particles (packed_info != 0)."""
        if self._high_water == 0:
            return 0
        return int(cp.count_nonzero(self.packed_info[:self._high_water]))

    def spawn_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float,
        material_id: int,
        count: int,
    ) -> int:
        """Spawn particles with random positions inside a sphere.

        Returns the number of particles actually spawned (may be less than
        count if max_particles is reached).
        """
        if material_id not in MATERIALS:
            raise ValueError(f"Unknown material_id: {material_id}")

        available = self.max_particles - self._high_water
        actual = min(count, available)
        if actual <= 0:
            return 0

        mat = MATERIALS[material_id]
        start = self._high_water

        # Generate random positions in unit sphere using rejection sampling on GPU
        # Over-generate by ~2x to account for rejection, then take first `actual`
        needed = actual
        positions_list = []
        remaining = needed
        while remaining > 0:
            batch = int(remaining * 2.0) + 128
            pts = cp.random.uniform(-1.0, 1.0, (batch, 3)).astype(cp.float32)
            r_sq = cp.sum(pts * pts, axis=1)
            inside = pts[r_sq <= 1.0]
            take = min(len(inside), remaining)
            if take > 0:
                positions_list.append(inside[:take])
                remaining -= take

        sphere_pts = cp.concatenate(positions_list, axis=0)[:actual]
        # Scale to desired radius and translate to center
        cx, cy, cz = center
        sphere_pts[:, 0] = sphere_pts[:, 0] * radius + cx
        sphere_pts[:, 1] = sphere_pts[:, 1] * radius + cy
        sphere_pts[:, 2] = sphere_pts[:, 2] * radius + cz

        sl = slice(start, start + actual)
        self.position[sl, 0] = sphere_pts[:, 0]
        self.position[sl, 1] = sphere_pts[:, 1]
        self.position[sl, 2] = sphere_pts[:, 2]
        self.position[sl, 3] = 0.0

        self.velocity[sl] = 0.0
        self.veleval[sl] = 0.0
        self.sph_force[sl] = 0.0

        # mass = rho0 * spacing^3
        particle_mass = mat.rest_density * (DEFAULT_SPACING ** 3)
        self.mass[sl] = particle_mass

        packed = _MAKE_PACKED(material_id, mat.behavior_class)
        self.packed_info[sl] = cp.uint32(packed)

        temp = _DEFAULT_TEMPS.get(material_id, T_AMBIENT)
        self.temperature[sl] = temp
        self.health[sl] = 1.0
        self.lifetime[sl] = 0.0
        self.shear_rate[sl] = 0.0
        self.exposure_heat[sl] = 0.0
        self.exposure_corrode[sl] = 0.0

        self.color[sl, 0] = mat.color_r
        self.color[sl, 1] = mat.color_g
        self.color[sl, 2] = mat.color_b
        self.color[sl, 3] = 1.0

        self.density[sl] = mat.rest_density
        self.sleep_counter[sl] = 0

        self._high_water += actual
        return actual

    def spawn_cube(
        self,
        min_corner: Tuple[float, float, float],
        max_corner: Tuple[float, float, float],
        material_id: int,
        spacing: float = DEFAULT_SPACING,
    ) -> int:
        """Spawn particles on a regular grid within a cube.

        Returns the number of particles actually spawned.
        """
        if material_id not in MATERIALS:
            raise ValueError(f"Unknown material_id: {material_id}")

        # Build grid positions on CPU (small overhead, predictable count)
        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner
        xs = np.arange(x0 + spacing * 0.5, x1, spacing, dtype=np.float32)
        ys = np.arange(y0 + spacing * 0.5, y1, spacing, dtype=np.float32)
        zs = np.arange(z0 + spacing * 0.5, z1, spacing, dtype=np.float32)

        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            return 0

        total = len(xs) * len(ys) * len(zs)
        available = self.max_particles - self._high_water
        actual = min(total, available)
        if actual <= 0:
            return 0

        # Generate meshgrid positions
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        positions = np.column_stack([
            gx.ravel()[:actual],
            gy.ravel()[:actual],
            gz.ravel()[:actual],
        ]).astype(np.float32)

        actual = len(positions)
        if actual == 0:
            return 0

        mat = MATERIALS[material_id]
        start = self._high_water
        sl = slice(start, start + actual)

        d_positions = cp.asarray(positions)
        self.position[sl, 0] = d_positions[:, 0]
        self.position[sl, 1] = d_positions[:, 1]
        self.position[sl, 2] = d_positions[:, 2]
        self.position[sl, 3] = 0.0

        self.velocity[sl] = 0.0
        self.veleval[sl] = 0.0
        self.sph_force[sl] = 0.0

        particle_mass = mat.rest_density * (spacing ** 3)
        self.mass[sl] = particle_mass

        packed = _MAKE_PACKED(material_id, mat.behavior_class)
        self.packed_info[sl] = cp.uint32(packed)

        temp = _DEFAULT_TEMPS.get(material_id, T_AMBIENT)
        self.temperature[sl] = temp
        self.health[sl] = 1.0
        self.lifetime[sl] = 0.0
        self.shear_rate[sl] = 0.0
        self.exposure_heat[sl] = 0.0
        self.exposure_corrode[sl] = 0.0

        self.color[sl, 0] = mat.color_r
        self.color[sl, 1] = mat.color_g
        self.color[sl, 2] = mat.color_b
        self.color[sl, 3] = 1.0

        self.density[sl] = mat.rest_density
        self.sleep_counter[sl] = 0

        self._high_water += actual
        return actual

    def kill_in_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float,
    ) -> int:
        """Mark particles within a sphere as DEAD (packed_info=0).

        Returns the number of particles killed.
        """
        if self._high_water == 0:
            return 0

        n = self._high_water
        cx, cy, cz = center
        dx = self.position[:n, 0] - cx
        dy = self.position[:n, 1] - cy
        dz = self.position[:n, 2] - cz
        dist_sq = dx * dx + dy * dy + dz * dz

        mask = dist_sq <= (radius * radius)
        # Only kill non-DEAD particles
        mask = mask & (self.packed_info[:n] != 0)
        killed = int(cp.sum(mask))

        if killed > 0:
            self.packed_info[:n][mask] = cp.uint32(0)

        return killed
