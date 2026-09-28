/*
 * reactions.cu -- K_Reactions kernel: per-particle state machine for
 * phase transitions, combustion, corrosion, and gas lifetime.
 *
 * No neighbor loop -- each thread processes one particle independently.
 * Reads exposure accumulators from Step1, checks temperature and exposure
 * thresholds, and modifies material type, behavior class, temperature,
 * health, lifetime, and velocity as needed.
 *
 * Runs on SORTED arrays, between Step1 (which computes exposure) and
 * Step2 (which computes forces). Modifications to sorted_packed_info
 * flow through Step2 and Integrate naturally.
 *
 * Every write to health, lifetime or mass also sets the state-dirty bit
 * (state_dirty.cuh) in packed_info, so K_Integrate scatters those three
 * arrays back to unsorted for this particle (bd-r5-review-n28.162).
 *
 * Constant memory used:
 *   c_sim       -- SimParams (dt)
 *   c_materials -- MaterialProps[32] (for looking up current material props)
 */

#include "common.cuh"
#include "state_dirty.cuh"

/* Forward declare grid helpers for blast wave kernel */

/* ======================================================================
 * Material IDs (must match materials.py)
 * ====================================================================== */

#define MAT_STONE      1
#define MAT_SAND       2
#define MAT_DIRT       3
#define MAT_GRAVEL     4
#define MAT_WATER      5
#define MAT_OIL        6
#define MAT_LAVA       7
#define MAT_ACID       8
#define MAT_WOOD       9
#define MAT_METAL     10
#define MAT_ICE       11
#define MAT_STEAM     12
#define MAT_SMOKE     13
#define MAT_FIRE      14
#define MAT_GUNPOWDER 15
#define MAT_WET_SAND  16
#define MAT_MUD       17

/* Material-set bitmasks for material IDs below 32 (K_Reactions). */
#define MAT_BIT(m) (1u << (m))
/* Acid-corrosion whitelist: the only materials whose health acid lowers. */
#define CORRODIBLE_MATS (MAT_BIT(MAT_STONE) | MAT_BIT(MAT_METAL) | MAT_BIT(MAT_WOOD) \
                       | MAT_BIT(MAT_GRAVEL) | MAT_BIT(MAT_OIL) | MAT_BIT(MAT_GUNPOWDER))

/* ======================================================================
 * Reaction thresholds
 * ====================================================================== */

/* Phase-change temperatures come from the material table (bd-r5-review-n28.89):
 *   ICE -> WATER above      c_materials[MAT_ICE].temp_melt           (273 K)
 *   WATER -> ICE below      c_materials[MAT_WATER].temp_melt - 10 K  (263 K)
 *   WATER boils above       c_materials[MAT_WATER].temp_boil         (373 K)
 *   STEAM -> WATER below    c_materials[MAT_WATER].temp_boil - 13 K  (360 K)
 * The condense point is keyed to WATER's boil point: STEAM has no temp_boil
 * of its own (0 in the table).  The two hysteresis gaps are fixed here. */
#define WATER_FREEZE_HYSTERESIS  10.0f   // freeze this far below the ICE melt point
/* ICE is STATIC, so only slow water freezes; a particle in a moving stream
 * would otherwise stop dead in mid-air.  Same speed as V_WAKE in
 * sph_shared.cuh (this file includes only common.cuh). */
#define WATER_FREEZE_MAX_SPEED 0.02f
// LAVA solidify uses c_materials[MAT_LAVA].temp_melt at runtime
#define STEAM_CONDENSE_HYSTERESIS 13.0f  // condense this far below the WATER boil point (see SPAWN_LIFETIME in spawn.cu)

/* Exposure from Step1 is normalised by the neighbour kernel support weight
 * (bd-r5-review-n28.48):
 *   exposure_heat    = sum_j heat_exchange_ij * max(T_j - T_i, 0) * W_ij / sum_j W_ij   [K]
 *   exposure_corrode = sum_j reaction_rate_ij * W_ij / sum_j W_ij                       [1/s]
 * On a 0.5h lattice one nearest neighbour holds about 10% of the support, so
 * one FIRE neighbour (dT ~ 900 K) gives WOOD 25 * 900 * 0.10 ~ 2300 K, and a
 * FIRE neighbour at 0.85h (0.5% of the support) gives about 110 K. */
#define WOOD_IGNITE_EXPOSURE  500.0f   // WOOD -> FIRE above this exposure_heat (K)
#define OIL_IGNITE_EXPOSURE   300.0f   // OIL  -> FIRE above this exposure_heat (K)
#define GUNPOWDER_EXPOSURE    100.0f   // GUNPOWDER -> FIRE above this exposure_heat (K)

#define FIRE_TEMPERATURE     1200.0f   // temperature set on combustion
#define WOOD_FIRE_LIFETIME      1.0f   // seconds
#define OIL_FIRE_LIFETIME       1.5f   // seconds
#define GUNPOWDER_FIRE_LIFETIME 0.3f   // seconds

#define EXPLOSION_SPEED       25.0f    // velocity magnitude for gunpowder burst

/* FIRE -> SMOKE when wet. FIRE's exposure_corrode comes only from WATER
 * (0.8 * water support fraction; materials.py zeroes the FIRE row for WOOD,
 * OIL, GUNPOWDER and ICE), so 0.3 means more than 37.5% of the support is
 * water. Fire sitting on wood with no water keeps burning. */
#define FIRE_EXTINGUISH_EXPOSURE 0.3f
#define EXTINGUISHED_SMOKE_LIFETIME 1.0f   // seconds
#define EXTINGUISHED_SMOKE_TEMP   373.0f   // K

/* Sand wetting/drying thresholds on exposure_corrode = rate * water fraction.
 * With SAND/WATER 0.4, WET_SAND/WATER 0.6, MUD/WATER 0.3, DIRT/WATER 0.5:
 *   SAND -> WET_SAND above 50% water, WET_SAND -> MUD above 83%,
 *   MUD -> WET_SAND below 33%, WET_SAND -> SAND below 3.3%,
 *   DIRT -> MUD above 40% water. */
#define SAND_WET_THRESHOLD      0.2f   // SAND -> WET_SAND
#define WETSAND_MUD_THRESHOLD   0.5f   // WET_SAND -> MUD
#define WETSAND_DRY_THRESHOLD   0.02f  // WET_SAND -> SAND (drying)
#define MUD_DRY_THRESHOLD       0.1f   // MUD -> WET_SAND (drying)
#define DRYING_TEMP_ACCEL       0.01f  // temperature acceleration for drying

/* ======================================================================
 * Spawn-mass rule on material change (bd-r5-review-n28.53)
 *
 * world.spawn_mass gives FLUID and GRANULAR particles
 * PARTICLE_MASS * rest_density / MASS_REF_DENSITY, so each rests at its own
 * rho0 at spacing 0.02; STATIC and GAS keep PARTICLE_MASS.  A particle that
 * changes material has its mass rescaled by the ratio of the two scale
 * factors, so it keeps resting at the new material's rho0:
 *   SAND -> WET_SAND x1.125, DIRT -> MUD x1.2, STONE -> LAVA x2.6,
 *   LAVA -> GRAVEL x(4250/6500), OIL -> FIRE back to PARTICLE_MASS.
 * Mass is not conserved across these transitions; the rest volume is.
 * MASS_REF_DENSITY must match world.MASS_REF_DENSITY.
 * ====================================================================== */

#define MASS_REF_DENSITY 2500.0f

__device__ inline float spawn_mass_scale(uint mat_id) {
    int b = c_materials[mat_id].behavior_class;
    if (b == FLUID || b == GRANULAR)
        return c_materials[mat_id].rest_density / MASS_REF_DENSITY;
    return 1.0f;
}

/* Rescale mass[i] for a change from old_mat to new_mat (NULL mass: no-op). */
__device__ inline void rescale_mass(float* mass, uint i, uint old_mat, uint new_mat) {
    if (mass != 0)
        mass[i] *= spawn_mass_scale(new_mat) / spawn_mass_scale(old_mat);
}

/* ======================================================================
 * Wang hash RNG for explosion direction
 * ====================================================================== */

__device__ inline uint wang_hash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

__device__ inline float hash_to_float(uint h) {
    // Map hash to [-1, 1]
    return (float)(h & 0xFFFFu) / 32767.5f - 1.0f;
}

/* ======================================================================
 * K_Reactions kernel
 * ====================================================================== */

extern "C" __global__
void K_Reactions(
    uint            numParticles,
    const uint*     frame_ptr,       // device pointer for RNG seed (graph-safe)
    // --- Sorted arrays (read+write in-place) ---
    uint*           __restrict__ packed_info,       // material_id + behavior + flags
    float*          __restrict__ temperature,       // particle temperature
    float*          __restrict__ health,            // particle health [0,1]
    float*          __restrict__ lifetime,          // remaining lifetime (seconds)
    float4*         __restrict__ velocity,          // particle velocity (for explosion)
    // --- Sorted inputs (read-only, from Step1) ---
    const float*    __restrict__ exposure_heat,     // heat exposure accumulator
    const float*    __restrict__ exposure_corrode,  // corrosion exposure accumulator
    // --- Freelist output (for spawn system) ---
    uint*           __restrict__ dead_indices,      // array of dead particle indices
    uint*           __restrict__ dead_count,         // atomic counter
    // --- Sorted mass (NULL skips every mass write) ---
    // Dead slots must have mass 0, or they act as invisible boundary
    // particles in every m_j-weighted neighbour sum (bd-r5-review-n28.23).
    // A material change rescales mass by the spawn-mass rule (rescale_mass).
    float*          __restrict__ mass,
    // --- PBF predicted positions (NULL for WCSPH / DFSPH) ---
    // K_PBF_Predict has already consumed velocity, and K_PBF_Finalize
    // rebuilds velocity as (x* - x)/dt, so a velocity kick must also move
    // x* by dt * dv or PBF loses it (bd-r5-review-n28.49).
    float4*         __restrict__ predicted_position
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_range = (i < numParticles);

    uint pi = in_range ? packed_info[i] : 0u;
    uint mat_id = GET_MATERIAL_ID(pi);
    bool is_dead = in_range && (mat_id == MAT_DEAD);

    // DEAD particles: offer the slot to the spawn freelist, then skip.
    // The freelist holds sorted indices, which go stale after a re-sort, so it
    // is reset every substep and must be rebuilt from ALL dead slots here --
    // not only from particles that die this substep.  Otherwise slots that are
    // not claimed the substep they die are lost for good and boiling water
    // almost never finds a slot to spawn steam into (bd-r4fix-uup.15).
    // Capacity: dead_indices holds max_particles >= numParticles entries.
    //
    // The push is warp-aggregated: one atomicAdd per warp instead of one
    // same-address atomic per dead slot (bd-r5-review-n28.142). Every lane
    // reaches the ballot (out-of-range lanes vote false), so it is convergent.
    // The pointer test is uniform across the launch.
    if (dead_indices != 0 && dead_count != 0) {
        uint dead_mask = __ballot_sync(0xffffffffu, is_dead);
        if (dead_mask != 0u) {
            uint lane = threadIdx.x & 31u;
            uint leader = (uint)__ffs(dead_mask) - 1u;
            uint base = 0u;
            if (lane == leader) base = atomicAdd(dead_count, (uint)__popc(dead_mask));
            base = __shfl_sync(0xffffffffu, base, leader);
            if (is_dead) {
                uint lanemask_lt = (1u << lane) - 1u;
                dead_indices[base + (uint)__popc(dead_mask & lanemask_lt)] = i;
            }
        }
    }
    if (!in_range) return;
    if (is_dead) {
        // Defensive: every dead slot is massless.  Store (and mark the slot
        // for K_Integrate's carry-back) only when the mass is not already 0.
        if (mass != 0 && mass[i] != 0.0f) {
            mass[i] = 0.0f;
            packed_info[i] = SET_STATE_DIRTY(pi);
        }
        return;
    }

    // Load only the per-particle fields this material tests
    // (bd-r5-review-n28.148).  nvcc does not sink these loads into the
    // material branches on its own, so loading all five up front cost
    // 20 B per live particle; WATER now reads 4 B, SAND 8 B.  health is
    // read only on the corrosion and acid paths, lifetime only for GAS.
    // A skipped load leaves 0, which no test below can see: every test on
    // a field is gated on a material that loads it.
    const bool mat_in_table = mat_id < 32u;
    const uint mat_bit = mat_in_table ? (1u << mat_id) : 0u;
    const uint TEMP_MATS = MAT_BIT(MAT_ICE) | MAT_BIT(MAT_STONE) | MAT_BIT(MAT_SAND)
                         | MAT_BIT(MAT_LAVA) | MAT_BIT(MAT_WATER) | MAT_BIT(MAT_STEAM)
                         | MAT_BIT(MAT_WET_SAND) | MAT_BIT(MAT_MUD);
    const uint HEAT_MATS = MAT_BIT(MAT_WOOD) | MAT_BIT(MAT_OIL) | MAT_BIT(MAT_GUNPOWDER);
    const uint CORRODE_MATS = MAT_BIT(MAT_SAND) | MAT_BIT(MAT_DIRT)
                            | MAT_BIT(MAT_WET_SAND) | MAT_BIT(MAT_MUD)
                            | CORRODIBLE_MATS | MAT_BIT(MAT_ACID) | MAT_BIT(MAT_FIRE);
    float temp = (mat_bit & TEMP_MATS) ? temperature[i] : 0.0f;
    float exp_heat = (mat_bit & HEAT_MATS) ? exposure_heat[i] : 0.0f;
    float exp_corrode = (mat_bit & CORRODE_MATS) ? exposure_corrode[i] : 0.0f;
    float hlth = 0.0f;
    float dt = c_sim.dt;

    // ---- Phase transitions based on temperature ----
    // Melt and solidify set temperature to the threshold: the excess above
    // (or deficit below) it is taken as consumed latent heat.  The particle
    // must then gain or lose heat again before any reverse transition
    // (bd-r5-review-n28.64).  No per-particle latent-energy store.

    // ICE (mat=11, temp > 273K) -> WATER (in-place, small density ratio)
    const float ice_melt_temp = c_materials[MAT_ICE].temp_melt;
    if (mat_id == MAT_ICE && temp > ice_melt_temp) {
        pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_WATER, FLUID));
        packed_info[i] = pi;
        rescale_mass(mass, i, MAT_ICE, MAT_WATER);
        temperature[i] = ice_melt_temp;
        return;  // done with this particle
    }

    // STONE (mat=1, temp > temp_melt=1500K) -> LAVA (melting)
    if (mat_id == MAT_STONE && temp > c_materials[MAT_STONE].temp_melt) {
        pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_LAVA, FLUID));
        packed_info[i] = pi;
        rescale_mass(mass, i, MAT_STONE, MAT_LAVA);
        temperature[i] = c_materials[MAT_STONE].temp_melt;
        return;
    }

    // SAND (mat=2, temp > temp_melt=1700K) -> LAVA (melting)
    if (mat_id == MAT_SAND && temp > c_materials[MAT_SAND].temp_melt) {
        pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_LAVA, FLUID));
        packed_info[i] = pi;
        rescale_mass(mass, i, MAT_SAND, MAT_LAVA);
        temperature[i] = c_materials[MAT_SAND].temp_melt;
        return;
    }

    // LAVA (mat=7, temp < temp_melt) -> GRAVEL (GRANULAR, in-place).
    // Not STATIC STONE: a STATIC particle never moves, so lava that cooled in
    // flight or in a water pool would hang in mid-air.  GRAVEL falls until it
    // is supported and then the sleep system pins it (bd-r5-review-n28.112).
    if (mat_id == MAT_LAVA && temp < c_materials[MAT_LAVA].temp_melt) {
        pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_GRAVEL, GRANULAR));
        packed_info[i] = pi;
        rescale_mass(mass, i, MAT_LAVA, MAT_GRAVEL);
        temperature[i] = c_materials[MAT_LAVA].temp_melt;
        return;
    }

    // WATER (mat=5, temp > 373K) -> set SPAWN_GAS flag (large density ratio, US-023)
    const float water_boil_temp = c_materials[MAT_WATER].temp_boil;
    if (mat_id == MAT_WATER && temp > water_boil_temp) {
        // K_SpawnGas leaves the flag set while it waits for freelist slots,
        // so only write when the flag changes (bd-r5-review-n28.142).
        if (!HAS_SPAWN_FLAG(pi)) packed_info[i] = SET_SPAWN_FLAG(pi);
        return;
    }
    // Water that cooled below boiling while waiting for slots must not
    // spawn steam later: drop the stale flag.
    if (mat_id == MAT_WATER && HAS_SPAWN_FLAG(pi)) {
        pi = CLEAR_SPAWN_FLAG(pi);
        packed_info[i] = pi;
    }

    // WATER (mat=5, temp < 263K, sleeping or |v| < 0.02 m/s) -> ICE (STATIC)
    // Temperature is set to the freeze point: the deficit below it is taken
    // as released latent heat, as for the other melt/solidify transitions.
    const float water_freeze_temp = c_materials[MAT_WATER].temp_melt - WATER_FREEZE_HYSTERESIS;
    if (mat_id == MAT_WATER && temp < water_freeze_temp) {
        float4 v = velocity[i];
        float v_sq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (IS_SLEEPING(pi) ||
                v_sq < WATER_FREEZE_MAX_SPEED * WATER_FREEZE_MAX_SPEED) {
            pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_ICE, STATIC));
            packed_info[i] = pi;
            rescale_mass(mass, i, MAT_WATER, MAT_ICE);
            temperature[i] = water_freeze_temp;
            velocity[i] = make_float4(0.0f, 0.0f, 0.0f, v.w);
            return;
        }
    }

    // STEAM (mat=12, temp < 360K) -> WATER, conserving mass in expectation
    // (bd-r5-review-n28.95).  Boiled steam carries 1/SPAWN_N of the source
    // water mass.  Converting it in place would give water at 1/3 of rho0,
    // which has zero Tait pressure and clumps.  Instead the particle survives
    // as WATER with full water spawn mass with probability
    // p = m_steam / m_water (1/3 for boiled steam, 1 for brush steam with
    // mass PARTICLE_MASS), and otherwise dies and returns its slot to the
    // freelist.  The expected water mass equals the steam mass.
    if (mat_id == MAT_STEAM && temp < water_boil_temp - STEAM_CONDENSE_HYSTERESIS) {
        if (mass != 0) {
            float water_mass = c_sim.particle_mass * spawn_mass_scale(MAT_WATER);
            float p_keep = mass[i] / water_mass;
            uint frame = *frame_ptr;
            uint h = wang_hash(wang_hash(i ^ 0x5bd1e995u) + frame * 0x9E3779B9u);
            float u = (float)(h >> 8) * (1.0f / 16777216.0f);  // [0, 1)
            if (u >= p_keep) {
                packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_DEAD, STATIC));
                lifetime[i] = 0.0f;
                health[i] = 0.0f;
                mass[i] = 0.0f;
                if (dead_indices != 0 && dead_count != 0) {
                    uint idx = atomicAdd(dead_count, 1u);
                    dead_indices[idx] = i;
                }
                return;
            }
            mass[i] = water_mass;
        }
        pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_WATER, FLUID));
        packed_info[i] = pi;
        temperature[i] = temp;
        return;
    }

    // ---- Combustion transitions based on exposure_heat ----

    // WOOD (mat=9, exposure_heat > WOOD_IGNITE_EXPOSURE) -> FIRE
    if (mat_id == MAT_WOOD && exp_heat > WOOD_IGNITE_EXPOSURE) {
        pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_FIRE, GAS));
        packed_info[i] = pi;
        rescale_mass(mass, i, MAT_WOOD, MAT_FIRE);
        temperature[i] = FIRE_TEMPERATURE;
        lifetime[i] = WOOD_FIRE_LIFETIME;
        return;
    }

    // OIL (mat=6, exposure_heat > OIL_IGNITE_EXPOSURE) -> FIRE
    if (mat_id == MAT_OIL && exp_heat > OIL_IGNITE_EXPOSURE) {
        pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_FIRE, GAS));
        packed_info[i] = pi;
        rescale_mass(mass, i, MAT_OIL, MAT_FIRE);
        temperature[i] = FIRE_TEMPERATURE;
        lifetime[i] = OIL_FIRE_LIFETIME;
        return;
    }

    // GUNPOWDER (mat=15, exposure_heat > GUNPOWDER_EXPOSURE) -> FIRE + explosion
    if (mat_id == MAT_GUNPOWDER && exp_heat > GUNPOWDER_EXPOSURE) {
        pi = SET_STATE_DIRTY(MAKE_PACKED(MAT_FIRE, GAS));
        packed_info[i] = pi;
        rescale_mass(mass, i, MAT_GUNPOWDER, MAT_FIRE);
        temperature[i] = FIRE_TEMPERATURE;
        lifetime[i] = GUNPOWDER_FIRE_LIFETIME;

        // Random outward burst using wang_hash RNG
        uint frame = *frame_ptr;
        uint h1 = wang_hash(i + frame * 0x9E3779B9u);
        uint h2 = wang_hash(h1);
        uint h3 = wang_hash(h2);
        float dx = hash_to_float(h1);
        float dy = hash_to_float(h2);
        float dz = hash_to_float(h3);
        // Normalize and scale
        float len = sqrtf(dx * dx + dy * dy + dz * dz);
        if (len > 1e-6f) {
            float inv_len = EXPLOSION_SPEED / len;
            float3 dv = make_float3(dx * inv_len, dy * inv_len, dz * inv_len);
            float4 v = velocity[i];
            v.x += dv.x;
            v.y += dv.y;
            v.z += dv.z;
            velocity[i] = v;
            if (predicted_position != NULL) {
                float dt = c_sim.dt;
                float4 p = predicted_position[i];
                p.x += dt * dv.x;
                p.y += dt * dv.y;
                p.z += dt * dv.z;
                predicted_position[i] = p;
            }
        }
        return;
    }

    // ---- DIRT wetting: DIRT + WATER exposure -> MUD (not corrosion) ----
    // This check must come BEFORE the corrosion whitelist so DIRT transitions
    // to MUD rather than being corroded (crumbled) by water exposure.
    // WATER has a non-zero corrosion reaction_rate with DIRT in the interaction
    // matrix, so exp_corrode accumulates when DIRT neighbours WATER.
    if (mat_id == MAT_DIRT && exp_corrode > SAND_WET_THRESHOLD) {
        packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_MUD, FLUID));
        rescale_mass(mass, i, MAT_DIRT, MAT_MUD);
        return;
    }

    // ---- Sand wetting/drying transitions ----
    if (mat_id == MAT_SAND && exp_corrode > SAND_WET_THRESHOLD) {
        packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_WET_SAND, GRANULAR));
        rescale_mass(mass, i, MAT_SAND, MAT_WET_SAND);
        return;
    }
    if (mat_id == MAT_WET_SAND && exp_corrode > WETSAND_MUD_THRESHOLD) {
        packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_MUD, FLUID));
        rescale_mass(mass, i, MAT_WET_SAND, MAT_MUD);
        return;
    }
    if (mat_id == MAT_MUD && exp_corrode < MUD_DRY_THRESHOLD
            * (1.0f + DRYING_TEMP_ACCEL * fmaxf(temp - 293.0f, 0.0f))) {
        packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_WET_SAND, GRANULAR));
        rescale_mass(mass, i, MAT_MUD, MAT_WET_SAND);
        return;
    }
    if (mat_id == MAT_WET_SAND && exp_corrode < WETSAND_DRY_THRESHOLD
            * (1.0f + DRYING_TEMP_ACCEL * fmaxf(temp - 293.0f, 0.0f))) {
        packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_SAND, GRANULAR));
        rescale_mass(mass, i, MAT_WET_SAND, MAT_SAND);
        return;
    }

    // ---- Corrosion: health -= exposure_corrode * dt ----
    // exposure_corrode is rate * acid support fraction, so a particle fully
    // surrounded by acid dies after 1/rate seconds (STONE 6.7 s, METAL 3.3 s).
    // Whitelist: only these materials can be corroded by acid.
    // MAT_DIRT is excluded: DIRT+WATER is handled above as DIRT->MUD wetting,
    // so DIRT should never reach this corrosion path via water exposure.
    // MAT_ICE is excluded: ICE melts by temperature only.
    // exposure_corrode sums every pair with a non-zero reaction_rate, so the
    // interaction table keeps rate 0 for heat-only pairs such as LAVA/METAL;
    // otherwise lava would eat metal here (bd-r5-review-n28.38).
    // TODO: a separate acid exposure channel would remove the dependence on
    // the table having rate 0 for every non-acid pair on the whitelist.
    if (exp_corrode > 0.0f && (mat_bit & CORRODIBLE_MATS)) {
        hlth = health[i];
        float damage = exp_corrode * dt;
        hlth -= damage;
        if (hlth <= 0.0f) {
            // Fully corroded: the particle is removed. It does not become a
            // FIRE spark: heat exposure has no temperature threshold, so a
            // 400 K spark ignited neighbouring wood, oil and gunpowder.
            packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_DEAD, STATIC));
            health[i] = 0.0f;
            lifetime[i] = 0.0f;
            if (mass != 0) mass[i] = 0.0f;
            if (dead_indices != 0 && dead_count != 0) {
                uint idx = atomicAdd(dead_count, 1u);
                dead_indices[idx] = i;
            }
            return;
        }
        health[i] = hlth;
        pi = SET_STATE_DIRTY(pi);
        packed_info[i] = pi;
    }

    // ---- Acid consumption: acid loses health when corroding ----
    if (mat_id == MAT_ACID && exp_corrode > 0.0f) {
        hlth = health[i];
        hlth -= exp_corrode * dt * 0.5f;
        if (hlth <= 0.0f) {
            packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_DEAD, STATIC));
            health[i] = 0.0f;
            if (mass != 0) mass[i] = 0.0f;
            if (dead_indices != 0 && dead_count != 0) {
                uint idx = atomicAdd(dead_count, 1u);
                dead_indices[idx] = i;
            }
            return;
        }
        health[i] = hlth;
        pi = SET_STATE_DIRTY(pi);
        packed_info[i] = pi;
    }

    // ---- Fire extinguished by water: FIRE -> SMOKE (bd-r5-review-n28.73) ----
    if (mat_id == MAT_FIRE && exp_corrode > FIRE_EXTINGUISH_EXPOSURE) {
        packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_SMOKE, GAS));
        rescale_mass(mass, i, MAT_FIRE, MAT_SMOKE);
        lifetime[i] = EXTINGUISHED_SMOKE_LIFETIME;
        temperature[i] = EXTINGUISHED_SMOKE_TEMP;
        return;
    }

    // ---- GAS lifetime decay ----
    int behavior = GET_BEHAVIOR(pi);
    if (behavior != GAS) return;
    float lt = lifetime[i];
    if (lt > 0.0f) {
        lt -= dt;
        if (lt <= 0.0f) {
            if (mat_id == MAT_FIRE) {
                // Fire expires -> becomes SMOKE with new lifetime
                packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_SMOKE, GAS));
                rescale_mass(mass, i, MAT_FIRE, MAT_SMOKE);
                lifetime[i] = 3.0f;
                temperature[i] = fminf(temperature[i], 500.0f);  // cool down
            } else {
                // Other gas expired -- add to freelist
                packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_DEAD, STATIC));
                lifetime[i] = 0.0f;
                health[i] = 0.0f;
                if (mass != 0) mass[i] = 0.0f;
                if (dead_indices != 0 && dead_count != 0) {
                    uint idx = atomicAdd(dead_count, 1u);
                    dead_indices[idx] = i;
                }
            }
            return;
        }
        lifetime[i] = lt;
        packed_info[i] = SET_STATE_DIRTY(pi);
    }
}

/* ======================================================================
 * K_BlastWave -- radial impulse from freshly-exploded gunpowder particles.
 *
 * Scans for FIRE particles with lifetime == GUNPOWDER_FIRE_LIFETIME (just
 * converted this frame). For each blast source, neighbor loop applies
 * radial impulse to all non-STATIC particles within smoothing_length h.
 * Uses atomicAdd on velocity for thread safety. Under PBF it also adds
 * dt * dv to the predicted position of every target.
 *
 * STATIC targets get no velocity write: integrate zeroes STATIC velocity
 * and K_UpdateBoundaryParticles rewrites rigid boundary velocity from the
 * body state, so the kick would vanish. A MAT_RIGID boundary particle
 * instead adds the force F = m_j * dv / dt and its torque about the world
 * origin, x_j x F, to its body's accumulators. K_IntegrateRigidBodies reads
 * them this substep and subtracts COM x F to get the torque about the body
 * centre (bd-r5-review-n28.195). Blasts are rare, so plain atomicAdd is used.
 * ====================================================================== */

#define BLAST_SPEED 15.0f

extern "C" __global__
void K_BlastWave(
    uint            numParticles,
    const float4*   __restrict__ sorted_position,
    float4*         __restrict__ velocity,          // read+write (atomicAdd)
    const uint*     __restrict__ sorted_packed_info,
    const float*    __restrict__ sorted_lifetime,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float           h,                              // smoothing length
    float4*         __restrict__ predicted_position, // PBF x* (atomicAdd dt*dv); NULL otherwise
    const float*    __restrict__ sorted_mass,       // particle mass (read for MAT_RIGID targets)
    const RigidBody* __restrict__ d_rigid_bodies,   // body state (NULL if no bodies); only gates the rigid reaction
    float*          __restrict__ d_rigid_forces,    // force accumulator [MAX_RIGID_BODIES * 4] (NULL if no bodies)
    float*          __restrict__ d_rigid_torques    // torque about the world origin [MAX_RIGID_BODIES * 4] (NULL if no bodies)
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = sorted_packed_info[i];
    uint mat_id = GET_MATERIAL_ID(pi);

    // Only process freshly-exploded gunpowder (FIRE with lifetime == GUNPOWDER_FIRE_LIFETIME)
    if (mat_id != MAT_FIRE) return;
    float lt = sorted_lifetime[i];
    // Accept lifetime within half a substep of GUNPOWDER_FIRE_LIFETIME so the
    // blast fires on exactly one substep: the conversion substep leaves
    // lifetime == GUNPOWDER_FIRE_LIFETIME (Reactions returns before decay),
    // and every later substep has decayed it by >= dt, outside this window.
    // (A fixed +-0.01 window spanned ~10 substeps at dt=0.001.)
    float tol = 0.5f * c_sim.dt;
    if (lt < GUNPOWDER_FIRE_LIFETIME - tol || lt > GUNPOWDER_FIRE_LIFETIME + tol) return;

    float4 pos_i4 = sorted_position[i];
    float3 pos_i = make_float3(pos_i4.x, pos_i4.y, pos_i4.z);
    float h_sq = h * h;

    // Neighbor loop (27-cell stencil)
    int3 cell_i = calcGridCell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint hash = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash];
                uint end = cell_end[hash];
                for (uint j = start; j < end; j++) {
                    if (j == i) continue;
                    float4 pos_j4 = sorted_position[j];
                    float3 diff = make_float3(
                        pos_j4.x - pos_i.x,
                        pos_j4.y - pos_i.y,
                        pos_j4.z - pos_i.z
                    );
                    float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                    if (dist_sq >= h_sq || dist_sq < 1e-10f) continue;

                    float dist = sqrtf(dist_sq);
                    float inv_dist = 1.0f / dist;
                    float falloff = 1.0f - dist / h;  // linear falloff
                    float impulse = BLAST_SPEED * falloff;

                    // Radial direction from blast center to neighbor
                    float3 dv = make_float3(diff.x * inv_dist * impulse,
                                            diff.y * inv_dist * impulse,
                                            diff.z * inv_dist * impulse);

                    uint pi_j = sorted_packed_info[j];
                    if (GET_BEHAVIOR(pi_j) == STATIC) {
                        // No velocity write on STATIC. A rigid boundary
                        // particle passes the impulse to its body as a force.
                        if (GET_MATERIAL_ID(pi_j) == MAT_RIGID
                            && d_rigid_forces != NULL && d_rigid_torques != NULL
                            && d_rigid_bodies != NULL && sorted_mass != NULL) {
                            int body_id = GET_BODY_ID(pi_j);
                            if (body_id >= MAX_RIGID_BODIES) continue;
                            float s = sorted_mass[j] / c_sim.dt;
                            float3 F = make_float3(s * dv.x, s * dv.y, s * dv.z);
                            float3 tau = make_float3(
                                pos_j4.y * F.z - pos_j4.z * F.y,
                                pos_j4.z * F.x - pos_j4.x * F.z,
                                pos_j4.x * F.y - pos_j4.y * F.x);
                            atomicAdd(&d_rigid_forces[body_id * 4 + 0], F.x);
                            atomicAdd(&d_rigid_forces[body_id * 4 + 1], F.y);
                            atomicAdd(&d_rigid_forces[body_id * 4 + 2], F.z);
                            atomicAdd(&d_rigid_torques[body_id * 4 + 0], tau.x);
                            atomicAdd(&d_rigid_torques[body_id * 4 + 1], tau.y);
                            atomicAdd(&d_rigid_torques[body_id * 4 + 2], tau.z);
                        }
                        continue;
                    }

                    atomicAdd(&velocity[j].x, dv.x);
                    atomicAdd(&velocity[j].y, dv.y);
                    atomicAdd(&velocity[j].z, dv.z);
                    // PBF: move x* too, so Finalize's (x* - x)/dt keeps the
                    // kick. STATIC targets were skipped above, so x* = x there.
                    if (predicted_position != NULL) {
                        float dt = c_sim.dt;
                        atomicAdd(&predicted_position[j].x, dt * dv.x);
                        atomicAdd(&predicted_position[j].y, dt * dv.y);
                        atomicAdd(&predicted_position[j].z, dt * dv.z);
                    }
                }
            }
        }
    }
}
