"""World state management for Falling Sand 3D.

Manages CuPy arrays for all per-particle SoA data. Supports configurable
max_particles at startup, spawning particles in sphere/cube shapes, and
tracking num_active via counter.

All arrays are contiguous CuPy float32/uint32/uint8 arrays on GPU.
DEAD particles have material_id 0. Kernels and kill_in_sphere write
DEAD_PACKED (material DEAD, behavior STATIC, 0x300) with mass 0. The alive
test is always material_id != 0, so a bare packed_info of 0 also counts as dead.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import cupy as cp
import numpy as np

from materials import MATERIALS, MAX_MATERIALS, FLUID, GRANULAR, GAS, STATIC, MAT_RIGID, WATER, ICE

# packed_info bitfield macros (mirrors common.cuh)
_MAKE_PACKED = lambda mat, beh: (int(mat) & 0xFF) | ((int(beh) & 0x3) << 8)

# Encoding of a dead slot: material DEAD (0), behavior STATIC. Matches
# MAKE_PACKED(MAT_DEAD, STATIC) in spawn.cu and reactions.cu.
DEAD_PACKED = _MAKE_PACKED(0, STATIC)

# (MAX_MATERIALS, 3) float32 table of base material colors, indexed by
# material id. Built once on first use by World.refresh_color.
_rgb_table = None


def _material_rgb_table():
    """Return the cached GPU lookup of base material colors (r, g, b)."""
    global _rgb_table
    if _rgb_table is None:
        host = np.zeros((MAX_MATERIALS, 3), dtype=np.float32)
        for mat_id, mat in MATERIALS.items():
            host[mat_id] = (mat.color_r, mat.color_g, mat.color_b)
        _rgb_table = cp.asarray(host)
    return _rgb_table

# Kill every live, non-rigid particle inside a sphere (bd-r5-review-n28.34).
# One launch and no host sync. A killed slot becomes DEAD_PACKED with zero
# mass, health and velocity, so no solver sees it as fluid, it adds nothing
# to neighbour density and it cannot move. Its color is (0, 0, 0, -1): the
# negative w makes every particle vertex shader cull it.
_KILL_SPHERE_SRC = r"""
extern "C" __global__
void K_KillSphere(const float4* __restrict__ position,
                  unsigned int* __restrict__ packed_info,
                  float* __restrict__ mass,
                  float* __restrict__ health,
                  float4* __restrict__ velocity,
                  float4* __restrict__ color,
                  float cx, float cy, float cz, float r2,
                  unsigned int mat_rigid, unsigned int dead_packed, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int mat = packed_info[i] & 0xFFu;
    if (mat == 0u || mat == mat_rigid) return;
    float4 p = position[i];
    float dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
    if (dx * dx + dy * dy + dz * dz > r2) return;
    packed_info[i] = dead_packed;
    mass[i] = 0.0f;
    health[i] = 0.0f;
    velocity[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // color.w = -1 marks the slot DEAD for the vertex shaders, which cull it
    // (matches DEAD_COLOR_ALPHA in sph_shared.cuh).
    color[i] = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
}
"""
_kill_sphere_kernel = None


def _get_kill_sphere_kernel():
    global _kill_sphere_kernel
    if _kill_sphere_kernel is None:
        _kill_sphere_kernel = cp.RawKernel(_KILL_SPHERE_SRC, 'K_KillSphere')
    return _kill_sphere_kernel

# Default particle spacing (h=0.04 -> spacing=h/2=0.02)
DEFAULT_SPACING = 0.02

# Reference particle mass. At spacing 0.02 the kernel-sum density of a
# lattice of 0.02 kg particles is about 2525, so a material with
# rest_density MASS_REF_DENSITY = 2500 (WATER, ACID) rests at rho0 with it.
PARTICLE_MASS = 0.02
MASS_REF_DENSITY = 2500.0


def spawn_mass(material_id: int) -> float:
    """Spawn mass for one particle of material_id at the default spacing.

    FLUID and GRANULAR materials get PARTICLE_MASS * rest_density / 2500,
    so every one of them spawns at its own rho0 (SAND 0.032, LAVA 0.052,
    OIL 0.017) instead of at 2500 (bd-r5-review-n28.53).  STATIC keeps
    PARTICLE_MASS: STONE 6500 or METAL 19500 would make wall-adjacent fluid
    read far above its rho0.  GAS keeps PARTICLE_MASS (its units are a
    separate issue, bd-r5-review-n28.80).  K_Reactions applies the same
    rule when a particle changes material.
    """
    mat = MATERIALS[material_id]
    if mat.behavior_class in (FLUID, GRANULAR):
        return PARTICLE_MASS * mat.rest_density / MASS_REF_DENSITY
    return PARTICLE_MASS

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

# --- Parked DEAD reserve for the spawn freelist (bd-r5-review-n28.27) ---
# K_SpawnGas can only turn boiling water into steam by claiming DEAD slots,
# and the freelist is rebuilt each substep from DEAD slots only. A scene where
# nothing has died yet has no DEAD slots, so water never boils. World keeps a
# reserve of placeholder DEAD particles (mass 0) while a boilable material is
# present. They are parked on a lattice far below the world box, beyond the
# camera far plane (100), one per grid cell, so no hash bucket gets more than
# a few of them.
DEAD_RESERVE_DEFAULT = 15_000
BOILABLE_MATERIAL_IDS = frozenset({WATER, ICE})  # ICE melts to WATER
PARK_ORIGIN = (0.0, -1000.0, 0.0)
PARK_SPACING_DEFAULT = 0.04  # default smoothing length h


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
        # Set by every spawn_cube call: True when the particle budget cut the
        # lattice short. spawn_truncated_count counts such calls over the
        # world's lifetime, so a caller can check a whole batch of spawns.
        self.last_spawn_truncated: bool = False
        self.spawn_truncated_count: int = 0
        # Parked DEAD slots kept for the spawn freelist (see DEAD_RESERVE_DEFAULT).
        # 0 turns the reserve off.
        self.dead_reserve: int = DEAD_RESERVE_DEFAULT
        # Slots kept free for a rigid boundary block that is not in the world
        # yet. RigidBodyManager sets it: num_boundary_particles from
        # finalize_boundary_data() until the block is injected, then 0 (the
        # block is counted inside _high_water from then on). Every spawn path
        # takes it off the free capacity, so fluid cannot fill the slots the
        # block needs (bd-r5-review-n28.58).
        self.reserved_tail: int = 0
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
        # Previous-substep density snapshot, read as density_in for strain-rate
        # m_j/rho_j weighting.  Kept separate from sorted_density (the density_out
        # target) to break the density_in==density_out __restrict__ aliasing race
        # (bd-r4fix-uup.14).  Populated per-substep in simulation.py.
        self.sorted_density_prev = cp.zeros(n, dtype=cp.float32)
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
        # Warm-start divergence (DFSPH) -- PERF-008
        self.kappa_v = cp.zeros(n, dtype=cp.float32)
        self.sorted_kappa_v = cp.zeros(n, dtype=cp.float32)
        # PBF lambda scratch: written by K_PBF_ComputeLambda and read by
        # K_PBF_ComputeDelta within one iteration. Never reordered or scattered
        # back; PBF recomputes lambda from scratch every iteration.
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
        # Sorted position at the last full sort (bd-r5-review-n28.54). Written
        # by K_ScatterReorder only; the finalize kernels measure drift against it.
        self.sorted_anchor = cp.zeros((n, 4), dtype=cp.float32)

    # Solver arrays created on demand by allocate_pbf_arrays /
    # allocate_dfsph_arrays. resize() drops them so they are re-created at
    # the new max_particles.
    _LAZY_SOLVER_ARRAYS = (
        'predicted_position', 'sorted_predicted_position',
        'delta_position', 'sorted_delta_position',
        'sorted_pressure_normal',
        'alpha_dfsph', 'sorted_alpha_dfsph',
        'sorted_ist_scratch',
    )

    def _needs_alloc(self, *names: str) -> bool:
        """True if any named array is missing or not sized to max_particles."""
        n = self.max_particles
        for name in names:
            arr = getattr(self, name, None)
            if arr is None or arr.shape[0] != n:
                return True
        return False

    def allocate_pbf_arrays(self) -> None:
        """Lazily allocate PBF-specific arrays."""
        n = self.max_particles
        # Shared with DFSPH -- allocate only if missing or wrong size
        if self._needs_alloc('predicted_position', 'sorted_predicted_position'):
            self.predicted_position = cp.zeros((n, 4), dtype=cp.float32)
            self.sorted_predicted_position = cp.zeros((n, 4), dtype=cp.float32)
        # PBF-specific arrays (the lambda scratch is allocated in _allocate)
        if self._needs_alloc('delta_position', 'sorted_delta_position'):
            self.delta_position = cp.zeros((n, 4), dtype=cp.float32)
            self.sorted_delta_position = cp.zeros((n, 4), dtype=cp.float32)
        # Pressure normal for Drucker-Prager friction (only used within PBF iterations)
        if self._needs_alloc('sorted_pressure_normal'):
            self.sorted_pressure_normal = cp.zeros((n, 4), dtype=cp.float32)

    def allocate_dfsph_arrays(self) -> None:
        """Lazily allocate DFSPH-specific arrays."""
        n = self.max_particles
        # DFSPH also needs predicted_position
        if self._needs_alloc('predicted_position', 'sorted_predicted_position'):
            self.predicted_position = cp.zeros((n, 4), dtype=cp.float32)
            self.sorted_predicted_position = cp.zeros((n, 4), dtype=cp.float32)
        # kappa_v now in _allocate (PERF-008); alpha still lazy
        if self._needs_alloc('alpha_dfsph', 'sorted_alpha_dfsph'):
            self.alpha_dfsph = cp.zeros(n, dtype=cp.float32)
            self.sorted_alpha_dfsph = cp.zeros(n, dtype=cp.float32)

    def ensure_ist_scratch(self, size: int) -> None:
        """Lazily allocate the implicit surface tension ping-pong buffer.

        IST needs its own float4 scratch so it does not overwrite
        sorted_veleval, which holds the step2 XSPH advection velocity that
        K_Integrate reads (bd-r5-review-n28.56). Sized to
        min(size, max_particles); host-only check, no device sync.
        """
        n = min(int(size), self.max_particles)
        arr = getattr(self, 'sorted_ist_scratch', None)
        if arr is None or arr.shape[0] != n:
            self.sorted_ist_scratch = cp.zeros((n, 4), dtype=cp.float32)

    def resize(self, new_max: int) -> None:
        """Reallocate all arrays for a new max_particles. Kills all particles.

        Lazily allocated solver arrays are dropped; the next
        allocate_pbf_arrays / allocate_dfsph_arrays call re-creates them
        at the new size.
        """
        self.max_particles = new_max
        self._high_water = 0
        self.reserved_tail = 0
        self._spawned_material_ids = set()
        for name in self._LAZY_SOLVER_ARRAYS:
            self.__dict__.pop(name, None)
        self._allocate()

    def free_capacity(self) -> int:
        """Slots a spawn may still use: max_particles minus _high_water and
        minus reserved_tail (the rigid boundary block waiting to be injected).
        Host-only arithmetic, no device sync."""
        return max(0, self.max_particles - self.reserved_tail - self._high_water)

    @property
    def num_active(self) -> int:
        """Count of non-DEAD particles (material_id != 0) in [0, _high_water).

        Performs one GPU reduction + D2H sync, so it must NOT be called on the
        per-frame / per-substep hot path. Its only per-frame caller is
        ``Simulation._maybe_compact``, which is gated to run once every
        ``compact_interval`` frames (default 60) — a negligible ~1 Hz sync.
        Also used by UI and tests.
        """
        n = self._high_water
        if n == 0:
            return 0
        return int(cp.count_nonzero(self.packed_info[:n] & cp.uint32(0xFF)))

    def refresh_color(self, n: int | None = None) -> None:
        """Rebuild ``color[:n]`` from the material table and ``packed_info``.

        Writes the base material color (r, g, b) and ``behavior * 0.25`` in w,
        the same encoding as ``compute_color`` in sph_shared.cuh. DEAD slots
        get (0, 0, 0, -1) so the shaders cull them. Temperature glow is left
        out; the next substep adds it. One gather on the GPU, no host sync.

        Call after anything that rewrites ``packed_info`` outside a substep
        (load_scene, snapshot restore), so a paused scene shows the right
        colors (bd-r5-review-n28.63).
        """
        if n is None:
            n = self._high_water
        if n <= 0:
            return
        pi = self.packed_info[:n]
        mat = (pi & cp.uint32(0xFF)).astype(cp.int32)
        rgb = _material_rgb_table()
        self.color[:n, :3] = rgb[mat]
        behavior = ((pi >> cp.uint32(8)) & cp.uint32(0x3)).astype(cp.float32)
        self.color[:n, 3] = cp.where(mat == 0, cp.float32(-1.0),
                                     behavior * cp.float32(0.25))

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

        self._shift_rigid_boundary_block(alive_mask)

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
        # Persistent per-particle state (kappa, dye, angular_velocity, kappa_v)
        self.sorted_kappa[:num_alive] = self.kappa[:n][alive_idx]
        self.sorted_particle_dye[:num_alive] = self.particle_dye[:n][alive_idx]
        self.sorted_angular_velocity[:num_alive] = self.angular_velocity[:n][alive_idx]
        self.sorted_kappa_v[:num_alive] = self.kappa_v[:n][alive_idx]

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

        # Zero out the dead tail to prevent stale data
        if num_alive < n:
            self.packed_info[num_alive:n] = cp.uint32(DEAD_PACKED)

        self._high_water = num_alive
        return num_alive

    def _shift_rigid_boundary_block(self, alive_mask) -> None:
        """Move the rigid boundary offset to where compaction puts the block.

        alive_idx is ascending, so the block stays contiguous and its new start
        is the number of alive slots below the old start. If any block slot is
        dead, the block is broken: ask the manager to rebuild it on the next
        substep instead. compact() already syncs, so the one readback here
        adds no new sync point. (bd-r5-review-n28.22)
        """
        rbm = getattr(self, 'rigid_body_manager', None)
        if rbm is None or not rbm.boundary_injected or rbm.boundary_offset < 0:
            return
        off = rbm.boundary_offset
        nb = rbm.num_boundary_particles
        counts = cp.stack([
            cp.count_nonzero(alive_mask[:off]),
            cp.count_nonzero(alive_mask[off:off + nb]),
        ]).get()
        if int(counts[1]) == nb:
            rbm.boundary_offset = int(counts[0])
        else:
            rbm.mark_for_reinjection()

    @property
    def needs_dead_reserve(self) -> bool:
        """True when a boilable material was spawned and the reserve is on.

        CPU-side only (reads _spawned_material_ids), so it costs no GPU sync.
        """
        return (self.dead_reserve > 0
                and not BOILABLE_MATERIAL_IDS.isdisjoint(self._spawned_material_ids))

    def park_positions(self, slots, spacing: float = PARK_SPACING_DEFAULT,
                       grid_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)):
        """Parking positions (float4, w=0) for DEAD reserve slot indices.

        Each slot index maps to its own site on a cube lattice of side
        ceil(cbrt(max_particles)) at the given spacing, placed under
        PARK_ORIGIN. Distinct slots never share a site, so with spacing = h
        every parked particle sits in its own grid cell. The lattice is
        snapped to the grid that starts at grid_min, and each site is at a
        cell centre, so float rounding cannot push two into one cell.
        """
        slots = cp.asarray(slots, dtype=cp.int64)
        side = max(1, int(np.ceil(self.max_particles ** (1.0 / 3.0))))
        while side ** 3 < self.max_particles:
            side += 1
        ix = slots % side
        iy = slots // side % side
        iz = slots // (side * side)
        # Lattice corner: centred on PARK_ORIGIN in x and z, lattice top at
        # PARK_ORIGIN in y, snapped to a cell corner of the grid.
        corner = (PARK_ORIGIN[0] - side * 0.5 * spacing,
                  PARK_ORIGIN[1] - side * spacing,
                  PARK_ORIGIN[2] - side * 0.5 * spacing)
        base = [g + round((c - g) / spacing) * spacing
                for c, g in zip(corner, grid_min)]
        pos = cp.zeros((slots.shape[0], 4), dtype=cp.float32)
        pos[:, 0] = base[0] + (ix + 0.5) * spacing
        pos[:, 1] = base[1] + (iy + 0.5) * spacing
        pos[:, 2] = base[2] + (iz + 0.5) * spacing
        return pos

    def add_dead_reserve(self, count: int, spacing: float = PARK_SPACING_DEFAULT,
                         grid_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)) -> int:
        """Append up to `count` parked DEAD particles at [_high_water, ...).

        They are DEAD with behavior STATIC, mass 0 and zero state, so
        every kernel early-returns on them and K_Reactions offers them to the
        spawn freelist. Does not change _spawned_material_ids or num_active.
        Returns the number added (capped by max_particles). The caller must
        call Simulation.notify_scene_changed() so the sort is rebuilt.
        """
        count = min(int(count), self.free_capacity())
        if count <= 0:
            return 0
        start = self._high_water
        sl = slice(start, start + count)
        self.position[sl] = self.park_positions(
            cp.arange(start, start + count), spacing, grid_min)
        for arr in (self.velocity, self.veleval, self.sph_force, self.color,
                    self.particle_dye, self.angular_velocity):
            arr[sl] = 0.0
        self.temperature[sl] = T_AMBIENT
        for arr in (self.density, self.mass, self.health, self.lifetime, self.shear_rate, self.exposure_heat,
                    self.exposure_corrode, self.kappa, self.kappa_v):
            arr[sl] = 0.0
        # DEAD with behavior STATIC. Plain 0 would read as behavior FLUID
        # (FLUID = 0), and K_Integrate would drop the parked slot under gravity.
        self.packed_info[sl] = cp.uint32(_MAKE_PACKED(0, STATIC))
        # color.w = -1 marks the slot DEAD, so the vertex shaders cull it
        # (DEAD_COLOR_ALPHA in sph_shared.cuh).
        self.color[sl, 3] = -1.0
        self.sleep_counter[sl] = 0
        self._high_water += count
        return count

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

        available = self.free_capacity()
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

        self.mass[sl] = spawn_mass(material_id)

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
        self.color[sl, 3] = mat.behavior_class * 0.25

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
        exclude_box: Optional[Tuple[Tuple[float, float, float],
                                    Tuple[float, float, float]]] = None,
        inside_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> int:
        """Spawn particles on a regular grid within a cube.

        exclude_box: optional (min_corner, max_corner). Lattice points inside
            it (bounds included) are dropped, so another material can fill
            that box without two lattices sharing volume.
        inside_fn: optional filter. Called with the (N, 3) float32 CPU
            positions, it returns a boolean mask of the points to keep.

        The lattice is ordered with y slowest, so when the particle budget
        runs out the top layers are dropped and the footprint stays whole.
        `last_spawn_truncated` says whether that happened.

        Returns the number of particles actually spawned.
        """
        if material_id not in MATERIALS:
            raise ValueError(f"Unknown material_id: {material_id}")
        self.last_spawn_truncated = False

        # Build grid positions on CPU (small overhead, predictable count)
        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner
        xs = np.arange(x0 + spacing * 0.5, x1, spacing, dtype=np.float32)
        ys = np.arange(y0 + spacing * 0.5, y1, spacing, dtype=np.float32)
        zs = np.arange(z0 + spacing * 0.5, z1, spacing, dtype=np.float32)

        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            return 0

        available = self.free_capacity()
        if available <= 0:
            self.last_spawn_truncated = True
            self.spawn_truncated_count += 1
            return 0

        # y is the slowest axis, so truncation removes whole top layers.
        gy, gx, gz = np.meshgrid(ys, xs, zs, indexing='ij')
        positions = np.column_stack([
            gx.ravel(), gy.ravel(), gz.ravel(),
        ]).astype(np.float32)

        keep = np.ones(len(positions), dtype=bool)
        if exclude_box is not None:
            lo = np.asarray(exclude_box[0], dtype=np.float32)
            hi = np.asarray(exclude_box[1], dtype=np.float32)
            keep &= ~np.all((positions >= lo) & (positions <= hi), axis=1)
        if inside_fn is not None:
            keep &= np.asarray(inside_fn(positions), dtype=bool)
        positions = positions[keep]

        if len(positions) > available:
            positions = positions[:available]
            self.last_spawn_truncated = True
            self.spawn_truncated_count += 1

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

        self.mass[sl] = spawn_mass(material_id)

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
        self.color[sl, 3] = mat.behavior_class * 0.25

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
        available = self.free_capacity()
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
        self.mass[sl] = spawn_mass(material_id)
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
        self.color[sl, 3] = mat.behavior_class * 0.25
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
        available = self.free_capacity()
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
        self.mass[sl] = spawn_mass(material_id)
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
        self.color[sl, 3] = mat.behavior_class * 0.25
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
        available = self.free_capacity()
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
        self.mass[sl] = spawn_mass(material_id)
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
        self.color[sl, 3] = mat.behavior_class * 0.25
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
        """Mark live particles within a sphere as DEAD.

        Killed slots get DEAD_PACKED (material DEAD, behavior STATIC), zero
        mass, health and velocity, and color (0, 0, 0, -1), the same state
        K_SpawnGas and K_Reactions leave behind once a solver writes their
        color. The vertex shaders cull any slot with color.w < 0. Rigid boundary particles are never killed:
        the rigid body manager owns that block (bd-r5-review-n28.22).

        Runs as one kernel with no GPU to host sync. Returns 0 because the
        count would need a sync and no caller uses it.
        """
        if self._high_water == 0:
            return 0

        n = self._high_water
        cx, cy, cz = center
        threads = 256
        blocks = (n + threads - 1) // threads
        _get_kill_sphere_kernel()(
            (blocks,), (threads,),
            (self.position, self.packed_info, self.mass, self.health,
             self.velocity, self.color,
             np.float32(cx), np.float32(cy), np.float32(cz),
             np.float32(radius * radius),
             np.uint32(MAT_RIGID), np.uint32(DEAD_PACKED), np.int32(n)),
        )
        return 0
