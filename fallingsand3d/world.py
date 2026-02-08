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

# Global particle mass -- matches parent project convention.
# Using mass = 0.02 (instead of rho0*dx^3 = 0.008) produces SPH densities
# ~2.5x rest_density, which keeps the Tait EOS in a stable high-pressure
# regime where k=3.0 and viscosity=3.5 are properly tuned.
PARTICLE_MASS = 0.02

# Default temperatures for hot materials (Kelvin)
_DEFAULT_TEMPS = {
    7: 1500.0,   # LAVA
    14: 1200.0,  # FIRE
    12: 373.0,   # STEAM
    13: 500.0,   # SMOKE
}

# Ambient temperature
T_AMBIENT = 293.0

# Default max foam particles
MAX_FOAM_PARTICLES = 200_000


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
        self._spawned_material_ids: set = set()  # CPU-side tracking for conditional skip
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
        self.sorted_exposure_heat = cp.zeros(n, dtype=cp.float32)
        self.sorted_exposure_corrode = cp.zeros(n, dtype=cp.float32)
        self.sorted_packed_info = cp.zeros(n, dtype=cp.uint32)
        self.sorted_sleep_counter = cp.zeros(n, dtype=cp.uint8)
        # Warm-start pressure (DFSPH uses this; always allocated for reorder kernel)
        self.kappa = cp.zeros(n, dtype=cp.float32)
        self.sorted_kappa = cp.zeros(n, dtype=cp.float32)
        # Vorticity (float4: omega_x, omega_y, omega_z, |omega|) -- computed in step1
        self.sorted_vorticity = cp.zeros((n, 4), dtype=cp.float32)
        # Surface normal (float4: n_x, n_y, n_z, neighbor_count_as_float) -- computed in step1
        self.sorted_normal = cp.zeros((n, 4), dtype=cp.float32)
        # Particle dye color (float4: r, g, b, unused) -- persistent, scattered through sort
        self.particle_dye = cp.zeros((n, 4), dtype=cp.float32)
        self.sorted_particle_dye = cp.zeros((n, 4), dtype=cp.float32)
        # Dye diffusion rate (float4: dr, dg, db, unused) -- computed in step1, applied in integrate
        self.sorted_dye_rate = cp.zeros((n, 4), dtype=cp.float32)
        # Micropolar: angular velocity (float4: omega_x, omega_y, omega_z, unused)
        # Persistent (like velocity), scattered through sort pipeline
        self.angular_velocity = cp.zeros((n, 4), dtype=cp.float32)
        self.sorted_angular_velocity = cp.zeros((n, 4), dtype=cp.float32)
        # FP16 velocity copy for neighbor loop bandwidth reduction (OPT-4.3)
        # Written during sort (counting_sort), read by step1/step2 neighbor loops.
        # 8 bytes per particle (half4) vs 16 bytes (float4) = 50% bandwidth savings on velocity reads.
        self.sorted_velocity_h = cp.zeros((n, 2), dtype=cp.uint32)  # half4 = 8 bytes = 2 x uint32
        # Sort index arrays (hash + sorted versions)
        self.hashes = cp.zeros(n, dtype=cp.uint32)
        self.sorted_hashes = cp.zeros(n, dtype=cp.uint32)

        # --- Foam (secondary particle) pool ---
        self._max_foam = MAX_FOAM_PARTICLES
        self.foam_position = cp.zeros((self._max_foam, 4), dtype=cp.float32)
        self.foam_velocity = cp.zeros((self._max_foam, 4), dtype=cp.float32)
        # Double buffer for compaction
        self.foam_position_b = cp.zeros((self._max_foam, 4), dtype=cp.float32)
        self.foam_velocity_b = cp.zeros((self._max_foam, 4), dtype=cp.float32)
        # Active count (device atomic counter)
        self.foam_count = cp.zeros(1, dtype=cp.uint32)
        self.foam_alive_count = cp.zeros(1, dtype=cp.uint32)  # scratch for compact
        self.foam_enabled = False

        # --- Grid reuse: max displacement tracking ---
        self.max_displacement = cp.zeros(1, dtype=cp.uint32)

    def allocate_pbf_arrays(self) -> None:
        """Lazily allocate PBF-specific arrays."""
        n = self.max_particles
        # Shared with DFSPH -- allocate only if missing
        if not hasattr(self, 'predicted_position'):
            self.predicted_position = cp.zeros((n, 4), dtype=cp.float32)
            self.sorted_predicted_position = cp.zeros((n, 4), dtype=cp.float32)
        # PBF-specific arrays -- check independently
        if not hasattr(self, 'lambda_pbf'):
            self.lambda_pbf = cp.zeros(n, dtype=cp.float32)
            self.sorted_lambda_pbf = cp.zeros(n, dtype=cp.float32)
            self.delta_position = cp.zeros((n, 4), dtype=cp.float32)
            self.sorted_delta_position = cp.zeros((n, 4), dtype=cp.float32)
        # Pressure normal for Drucker-Prager friction (only used within PBF iterations)
        if not hasattr(self, 'sorted_pressure_normal'):
            self.sorted_pressure_normal = cp.zeros((n, 4), dtype=cp.float32)

    def allocate_dfsph_arrays(self) -> None:
        """Lazily allocate DFSPH-specific arrays."""
        n = self.max_particles
        # DFSPH also needs predicted_position
        if not hasattr(self, 'predicted_position'):
            self.predicted_position = cp.zeros((n, 4), dtype=cp.float32)
            self.sorted_predicted_position = cp.zeros((n, 4), dtype=cp.float32)
        if not hasattr(self, 'alpha_dfsph'):
            self.alpha_dfsph = cp.zeros(n, dtype=cp.float32)
            self.sorted_alpha_dfsph = cp.zeros(n, dtype=cp.float32)
            self.kappa_v = cp.zeros(n, dtype=cp.float32)
            self.sorted_kappa_v = cp.zeros(n, dtype=cp.float32)

    def resize(self, new_max: int) -> None:
        """Reallocate all arrays for a new max_particles. Kills all particles."""
        self.max_particles = new_max
        self._high_water = 0
        self._spawned_material_ids = set()
        self._allocate()

    @property
    def num_active(self) -> int:
        """Count of non-DEAD particles (material_id != 0)."""
        if self._high_water == 0:
            return 0
        return int(cp.count_nonzero(self.packed_info[:self._high_water] & cp.uint32(0xFF)))

    def compact(self) -> int:
        """Compact particle arrays by moving alive particles to the front.

        Uses CuPy stream compaction (fancy indexing gather) to move all
        non-DEAD particles to contiguous slots [0, num_alive). Dead
        particles are effectively discarded (their slots become available
        for future spawning).

        Returns
        -------
        int
            Number of alive particles after compaction (_high_water).
        """
        n = self._high_water
        if n == 0:
            return 0

        # Find alive particle indices: material_id != 0
        alive_mask = (self.packed_info[:n] & cp.uint32(0xFF)) != cp.uint32(0)
        alive_idx = cp.flatnonzero(alive_mask)
        num_alive = len(alive_idx)

        if num_alive == n:
            # No dead particles, nothing to do
            return n

        if num_alive == 0:
            # All dead
            self._high_water = 0
            return 0

        # Gather alive particles into temporary buffers, then copy back.
        # We use the sorted_* buffers as scratch space since they're
        # ephemeral and not needed between frames.

        # float4 arrays
        self.sorted_position[:num_alive] = self.position[:n][alive_idx]
        self.sorted_velocity[:num_alive] = self.velocity[:n][alive_idx]
        self.sorted_veleval[:num_alive] = self.veleval[:n][alive_idx]
        self.sorted_sph_force[:num_alive] = self.sph_force[:n][alive_idx]
        self.sorted_color[:num_alive] = self.color[:n][alive_idx]
        # float arrays
        self.sorted_density[:num_alive] = self.density[:n][alive_idx]
        self.sorted_mass[:num_alive] = self.mass[:n][alive_idx]
        self.sorted_temperature[:num_alive] = self.temperature[:n][alive_idx]
        self.sorted_health[:num_alive] = self.health[:n][alive_idx]
        self.sorted_lifetime[:num_alive] = self.lifetime[:n][alive_idx]
        self.sorted_shear_rate[:num_alive] = self.shear_rate[:n][alive_idx]
        self.sorted_exposure_heat[:num_alive] = self.exposure_heat[:n][alive_idx]
        self.sorted_exposure_corrode[:num_alive] = self.exposure_corrode[:n][alive_idx]
        # uint32 arrays
        self.sorted_packed_info[:num_alive] = self.packed_info[:n][alive_idx]
        # uint8 arrays
        self.sorted_sleep_counter[:num_alive] = self.sleep_counter[:n][alive_idx]
        # Persistent per-particle state (kappa, dye, angular_velocity)
        self.sorted_kappa[:num_alive] = self.kappa[:n][alive_idx]
        self.sorted_particle_dye[:num_alive] = self.particle_dye[:n][alive_idx]
        self.sorted_angular_velocity[:num_alive] = self.angular_velocity[:n][alive_idx]

        # Copy back from scratch to primary arrays
        self.position[:num_alive] = self.sorted_position[:num_alive]
        self.velocity[:num_alive] = self.sorted_velocity[:num_alive]
        self.veleval[:num_alive] = self.sorted_veleval[:num_alive]
        self.sph_force[:num_alive] = self.sorted_sph_force[:num_alive]
        self.color[:num_alive] = self.sorted_color[:num_alive]
        self.density[:num_alive] = self.sorted_density[:num_alive]
        self.mass[:num_alive] = self.sorted_mass[:num_alive]
        self.temperature[:num_alive] = self.sorted_temperature[:num_alive]
        self.health[:num_alive] = self.sorted_health[:num_alive]
        self.lifetime[:num_alive] = self.sorted_lifetime[:num_alive]
        self.shear_rate[:num_alive] = self.sorted_shear_rate[:num_alive]
        self.exposure_heat[:num_alive] = self.sorted_exposure_heat[:num_alive]
        self.exposure_corrode[:num_alive] = self.sorted_exposure_corrode[:num_alive]
        self.packed_info[:num_alive] = self.sorted_packed_info[:num_alive]
        self.sleep_counter[:num_alive] = self.sorted_sleep_counter[:num_alive]
        self.kappa[:num_alive] = self.sorted_kappa[:num_alive]
        self.particle_dye[:num_alive] = self.sorted_particle_dye[:num_alive]
        self.angular_velocity[:num_alive] = self.sorted_angular_velocity[:num_alive]

        # Zero out the dead tail to prevent stale data
        if num_alive < n:
            self.packed_info[num_alive:n] = cp.uint32(0)

        self._high_water = num_alive
        return num_alive

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

        self.mass[sl] = PARTICLE_MASS

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

        # Initialize particle dye from material color
        self.particle_dye[sl, 0] = mat.color_r
        self.particle_dye[sl, 1] = mat.color_g
        self.particle_dye[sl, 2] = mat.color_b
        self.particle_dye[sl, 3] = 0.0

        self.density[sl] = mat.rest_density
        self.sleep_counter[sl] = 0

        self._high_water += actual
        self._spawned_material_ids.add(material_id)
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

        self.mass[sl] = PARTICLE_MASS

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

        # Initialize particle dye from material color
        self.particle_dye[sl, 0] = mat.color_r
        self.particle_dye[sl, 1] = mat.color_g
        self.particle_dye[sl, 2] = mat.color_b
        self.particle_dye[sl, 3] = 0.0

        self.density[sl] = mat.rest_density
        self.sleep_counter[sl] = 0

        self._high_water += actual
        self._spawned_material_ids.add(material_id)
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
