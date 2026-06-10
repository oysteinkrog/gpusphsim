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
    11: 253.0,   # ICE -- well below melt point of 273 K; prevents frame-1 thaw
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
        # Host-side dead count — updated by Simulation via async D2H readback.
        # num_active uses this instead of count_nonzero to avoid GPU sync.
        self._dead_count_host: int = 0
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
        # Warm-start divergence (DFSPH) and lambda (PBF) -- PERF-008
        self.kappa_v = cp.zeros(n, dtype=cp.float32)
        self.sorted_kappa_v = cp.zeros(n, dtype=cp.float32)
        self.lambda_pbf = cp.zeros(n, dtype=cp.float32)
        self.sorted_lambda_pbf = cp.zeros(n, dtype=cp.float32)
        # Pre-computed pressure (WCSPH PERF-007: eliminates compute_pressure per neighbor)
        self.sorted_pressure = cp.zeros(n, dtype=cp.float32)
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
        # FP16 temperature + dye copies for neighbor loop bandwidth (PERF-011)
        self.sorted_temperature_h = cp.zeros(n, dtype=cp.float16)  # half = 2 bytes per particle
        self.sorted_dye_h = cp.zeros((n, 2), dtype=cp.uint32)  # half4 = 8 bytes = 2 x uint32
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
        # PBF-specific arrays -- lambda_pbf now in _allocate (PERF-008)
        if not hasattr(self, 'delta_position'):
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
        # kappa_v now in _allocate (PERF-008); alpha still lazy
        if not hasattr(self, 'alpha_dfsph'):
            self.alpha_dfsph = cp.zeros(n, dtype=cp.float32)
            self.sorted_alpha_dfsph = cp.zeros(n, dtype=cp.float32)

    def resize(self, new_max: int) -> None:
        """Reallocate all arrays for a new max_particles. Kills all particles."""
        self.max_particles = new_max
        self._high_water = 0
        self._spawned_material_ids = set()
        self._dead_count_host = 0
        self._allocate()

    @property
    def num_active(self) -> int:
        """Count of non-DEAD particles (material_id != 0).

        Uses _high_water minus the host-side dead count that is updated
        asynchronously by Simulation each frame via pinned-memory D2H copy.
        No GPU->CPU synchronization occurs on this path.
        """
        return max(0, self._high_water - self._dead_count_host)

    def compact(self, num_alive: int = -1) -> int:
        """Compact particle arrays by moving alive particles to the front.

        Uses CuPy stream compaction (fancy indexing gather) to move all
        non-DEAD particles to contiguous slots [0, num_alive). Dead
        particles are effectively discarded (their slots become available
        for future spawning).

        Parameters
        ----------
        num_alive : int, optional
            Pre-computed count of alive particles (default -1 = compute here).
            Pass this from the async dead_count readback to avoid a GPU sync.
            If -1, the count is derived from alive_idx.shape[0] after flatnonzero.

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
        # Use caller-supplied count if available to avoid a GPU sync;
        # fall back to alive_idx.shape[0] (same sync cost as len()) only when
        # no count was provided.
        if num_alive < 0:
            num_alive = int(alive_idx.shape[0])

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
        # Persistent per-particle state (kappa, dye, angular_velocity, kappa_v, lambda_pbf)
        self.sorted_kappa[:num_alive] = self.kappa[:n][alive_idx]
        self.sorted_particle_dye[:num_alive] = self.particle_dye[:n][alive_idx]
        self.sorted_angular_velocity[:num_alive] = self.angular_velocity[:n][alive_idx]
        self.sorted_kappa_v[:num_alive] = self.kappa_v[:n][alive_idx]
        self.sorted_lambda_pbf[:num_alive] = self.lambda_pbf[:n][alive_idx]

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
        self.kappa_v[:num_alive] = self.sorted_kappa_v[:num_alive]
        self.lambda_pbf[:num_alive] = self.sorted_lambda_pbf[:num_alive]

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
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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

        # Generate exactly `actual` random positions uniformly inside a unit sphere.
        # Uses the Muller/Marsaglia method: sample direction from Gaussian (normalize),
        # then scale radius by r = uniform^(1/3).  Produces exactly `actual` points
        # in one GPU operation with no rejection loop and no per-iteration GPU sync.
        gauss = cp.random.standard_normal((actual, 3), dtype=cp.float32)
        norms = cp.linalg.norm(gauss, axis=1, keepdims=True)
        direction = gauss / (norms + 1e-10)  # unit sphere surface points
        r_scale = cp.random.uniform(0.0, 1.0, (actual, 1), dtype=cp.float32) ** (1.0 / 3.0)
        sphere_pts = direction * r_scale  # uniform inside unit sphere
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

        vx, vy, vz = velocity
        self.velocity[sl, 0] = vx
        self.velocity[sl, 1] = vy
        self.velocity[sl, 2] = vz
        self.velocity[sl, 3] = 0.0
        self.veleval[sl, 0] = vx
        self.veleval[sl, 1] = vy
        self.veleval[sl, 2] = vz
        self.veleval[sl, 3] = 0.0
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
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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

        vx, vy, vz = velocity
        self.velocity[sl, 0] = vx
        self.velocity[sl, 1] = vy
        self.velocity[sl, 2] = vz
        self.velocity[sl, 3] = 0.0
        self.veleval[sl, 0] = vx
        self.veleval[sl, 1] = vy
        self.veleval[sl, 2] = vz
        self.veleval[sl, 3] = 0.0
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

    def spawn_ramp(
        self,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        width: float,
        thickness: float,
        material_id: int,
        spacing: float = DEFAULT_SPACING,
    ) -> int:
        """Spawn a STATIC particle ramp between two 3D points.

        The ramp surface goes from start_pos to end_pos, with the given width
        perpendicular to the ramp direction (in XZ plane) and thickness below.
        """
        sx, sy, sz = start_pos
        ex, ey, ez = end_pos
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        length = (dx*dx + dy*dy + dz*dz) ** 0.5
        if length < 1e-6:
            return 0

        # Build ramp in local space then transform
        n_along = max(1, int(length / spacing))
        n_width = max(1, int(width / spacing))
        n_thick = max(1, int(thickness / spacing))

        # Direction vectors
        forward = np.array([dx, dy, dz], dtype=np.float32) / length
        # Perpendicular in XZ plane
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-6:
            right = np.array([1, 0, 0], dtype=np.float32)
        else:
            right = right / right_len
        down = np.cross(right, forward)

        positions = []
        for ia in range(n_along):
            t = (ia + 0.5) / n_along
            cx = sx + t * dx
            cy = sy + t * dy
            cz = sz + t * dz
            for iw in range(n_width):
                w_off = (iw + 0.5 - n_width / 2.0) * spacing
                for it in range(n_thick):
                    t_off = -(it + 0.5) * spacing  # below surface
                    px = cx + w_off * right[0] + t_off * down[0]
                    py = cy + w_off * right[1] + t_off * down[1]
                    pz = cz + w_off * right[2] + t_off * down[2]
                    positions.append([px, py, pz])

        if not positions:
            return 0

        pos_arr = np.array(positions, dtype=np.float32)
        available = self.max_particles - self._high_water
        actual = min(len(pos_arr), available)
        if actual <= 0:
            return 0

        mat = MATERIALS[material_id]
        start = self._high_water
        sl = slice(start, start + actual)
        d_positions = cp.asarray(pos_arr[:actual])
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
        self.particle_dye[sl, 0] = mat.color_r
        self.particle_dye[sl, 1] = mat.color_g
        self.particle_dye[sl, 2] = mat.color_b
        self.particle_dye[sl, 3] = 0.0
        self.density[sl] = mat.rest_density
        self.sleep_counter[sl] = 0
        self._high_water += actual
        self._spawned_material_ids.add(material_id)
        return actual

    def spawn_bowl(
        self,
        center: Tuple[float, float, float],
        radius: float,
        thickness: float,
        material_id: int,
        spacing: float = DEFAULT_SPACING,
    ) -> int:
        """Spawn a STATIC particle hemisphere bowl (bottom half of sphere shell)."""
        cx, cy, cz = center
        r_inner = radius - thickness
        positions = []
        n = int((2 * radius) / spacing) + 1
        for ix in range(n):
            px = cx - radius + (ix + 0.5) * spacing
            for iy in range(n):
                py = cy - radius + (iy + 0.5) * spacing
                if py > cy:  # bottom half only
                    continue
                for iz in range(n):
                    pz = cz - radius + (iz + 0.5) * spacing
                    dist_sq = (px-cx)**2 + (py-cy)**2 + (pz-cz)**2
                    if r_inner**2 <= dist_sq <= radius**2:
                        positions.append([px, py, pz])

        if not positions:
            return 0

        pos_arr = np.array(positions, dtype=np.float32)
        available = self.max_particles - self._high_water
        actual = min(len(pos_arr), available)
        if actual <= 0:
            return 0

        mat = MATERIALS[material_id]
        start = self._high_water
        sl = slice(start, start + actual)
        d_positions = cp.asarray(pos_arr[:actual])
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
        self.particle_dye[sl, 0] = mat.color_r
        self.particle_dye[sl, 1] = mat.color_g
        self.particle_dye[sl, 2] = mat.color_b
        self.particle_dye[sl, 3] = 0.0
        self.density[sl] = mat.rest_density
        self.sleep_counter[sl] = 0
        self._high_water += actual
        self._spawned_material_ids.add(material_id)
        return actual

    def spawn_wall_with_gap(
        self,
        wall_min: Tuple[float, float, float],
        wall_max: Tuple[float, float, float],
        gap_y_range: Tuple[float, float],
        material_id: int,
        spacing: float = DEFAULT_SPACING,
    ) -> int:
        """Wall with a rectangular gap for fluid to flow through."""
        gap_lo, gap_hi = gap_y_range
        # Bottom section
        n1 = self.spawn_cube(
            min_corner=wall_min,
            max_corner=(wall_max[0], gap_lo, wall_max[2]),
            material_id=material_id,
            spacing=spacing,
        )
        # Top section
        n2 = self.spawn_cube(
            min_corner=(wall_min[0], gap_hi, wall_min[2]),
            max_corner=wall_max,
            material_id=material_id,
            spacing=spacing,
        )
        return n1 + n2

    def spawn_cylinder_shell(
        self,
        center: Tuple[float, float, float],
        radius: float,
        height: float,
        thickness: float,
        material_id: int,
        spacing: float = DEFAULT_SPACING,
    ) -> int:
        """Hollow cylinder of STATIC particles."""
        cx, cy, cz = center
        r_inner = radius - thickness
        half_h = height / 2.0
        positions = []
        n_xz = int((2 * radius) / spacing) + 1
        n_y = int(height / spacing) + 1
        for ix in range(n_xz):
            px = cx - radius + (ix + 0.5) * spacing
            for iz in range(n_xz):
                pz = cz - radius + (iz + 0.5) * spacing
                dist_sq = (px-cx)**2 + (pz-cz)**2
                if r_inner**2 <= dist_sq <= radius**2:
                    for iy in range(n_y):
                        py = cy - half_h + (iy + 0.5) * spacing
                        positions.append([px, py, pz])

        if not positions:
            return 0

        pos_arr = np.array(positions, dtype=np.float32)
        available = self.max_particles - self._high_water
        actual = min(len(pos_arr), available)
        if actual <= 0:
            return 0

        mat = MATERIALS[material_id]
        start = self._high_water
        sl = slice(start, start + actual)
        d_positions = cp.asarray(pos_arr[:actual])
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

        # Apply mask; return 0 (count unused by any caller -- removing the
        # int(cp.sum(mask)) sync that stalled the GPU pipeline on every brush kill).
        self.packed_info[:n][mask] = cp.uint32(0)

        return 0
