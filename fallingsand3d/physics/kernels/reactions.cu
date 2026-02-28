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
 * Constant memory used:
 *   c_sim       -- SimParams (dt)
 *   c_materials -- MaterialProps[32] (for looking up current material props)
 */

#include "common.cuh"

/* ======================================================================
 * Material IDs (must match materials.py)
 * ====================================================================== */

#define MAT_DEAD       0
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

/* ======================================================================
 * Reaction thresholds
 * ====================================================================== */

#define ICE_MELT_TEMP         273.0f   // ICE -> WATER above this
// LAVA solidify uses c_materials[MAT_LAVA].temp_melt at runtime
#define WATER_BOIL_TEMP       373.0f   // WATER -> SPAWN_GAS flag above this
#define STEAM_CONDENSE_TEMP   360.0f   // STEAM -> WATER below this (13K hysteresis below boil)

#define WOOD_IGNITE_EXPOSURE  0.5f     // WOOD -> FIRE above this exposure_heat
#define OIL_IGNITE_EXPOSURE   0.3f     // OIL  -> FIRE above this exposure_heat
#define GUNPOWDER_EXPOSURE    0.1f     // GUNPOWDER -> FIRE above this exposure_heat

#define FIRE_TEMPERATURE     1200.0f   // temperature set on combustion
#define WOOD_FIRE_LIFETIME      1.0f   // seconds
#define OIL_FIRE_LIFETIME       1.5f   // seconds
#define GUNPOWDER_FIRE_LIFETIME 0.3f   // seconds

#define EXPLOSION_SPEED       25.0f    // velocity magnitude for gunpowder burst

/* Sand wetting/drying thresholds */
#define SAND_WET_THRESHOLD      0.2f   // SAND -> WET_SAND
#define WETSAND_MUD_THRESHOLD   0.5f   // WET_SAND -> MUD
#define WETSAND_DRY_THRESHOLD   0.02f  // WET_SAND -> SAND (drying)
#define MUD_DRY_THRESHOLD       0.1f   // MUD -> WET_SAND (drying)
#define DRYING_TEMP_ACCEL       0.01f  // temperature acceleration for drying

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
    uint*           __restrict__ dead_count          // atomic counter
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    uint mat_id = GET_MATERIAL_ID(pi);

    // Skip DEAD particles
    if (mat_id == MAT_DEAD) return;

    float temp = temperature[i];
    float hlth = health[i];
    float lt = lifetime[i];
    float exp_heat = exposure_heat[i];
    float exp_corrode = exposure_corrode[i];
    float dt = c_sim.dt;

    // ---- Phase transitions based on temperature ----

    // ICE (mat=11, temp > 273K) -> WATER (in-place, small density ratio)
    if (mat_id == MAT_ICE && temp > ICE_MELT_TEMP) {
        pi = MAKE_PACKED(MAT_WATER, FLUID);
        packed_info[i] = pi;
        temperature[i] = temp;
        return;  // done with this particle
    }

    // STONE (mat=1, temp > temp_melt=1500K) -> LAVA (melting)
    if (mat_id == MAT_STONE && temp > c_materials[MAT_STONE].temp_melt) {
        pi = MAKE_PACKED(MAT_LAVA, FLUID);
        packed_info[i] = pi;
        temperature[i] = temp;
        return;
    }

    // SAND (mat=2, temp > temp_melt=1700K) -> LAVA (melting)
    if (mat_id == MAT_SAND && temp > c_materials[MAT_SAND].temp_melt) {
        pi = MAKE_PACKED(MAT_LAVA, FLUID);
        packed_info[i] = pi;
        temperature[i] = temp;
        return;
    }

    // LAVA (mat=7, temp < temp_melt) -> STONE (in-place)
    if (mat_id == MAT_LAVA && temp < c_materials[MAT_LAVA].temp_melt) {
        pi = MAKE_PACKED(MAT_STONE, STATIC);
        packed_info[i] = pi;
        temperature[i] = temp;
        return;
    }

    // WATER (mat=5, temp > 373K) -> set SPAWN_GAS flag (large density ratio, US-023)
    if (mat_id == MAT_WATER && temp > WATER_BOIL_TEMP) {
        pi = SET_SPAWN_FLAG(pi);
        packed_info[i] = pi;
        return;
    }

    // STEAM (mat=12, temp < 373K) -> WATER (in-place)
    if (mat_id == MAT_STEAM && temp < STEAM_CONDENSE_TEMP) {
        pi = MAKE_PACKED(MAT_WATER, FLUID);
        packed_info[i] = pi;
        temperature[i] = temp;
        return;
    }

    // ---- Combustion transitions based on exposure_heat ----

    // WOOD (mat=9, exposure_heat > 0.5) -> FIRE
    if (mat_id == MAT_WOOD && exp_heat > WOOD_IGNITE_EXPOSURE) {
        pi = MAKE_PACKED(MAT_FIRE, GAS);
        packed_info[i] = pi;
        temperature[i] = FIRE_TEMPERATURE;
        lifetime[i] = WOOD_FIRE_LIFETIME;
        return;
    }

    // OIL (mat=6, exposure_heat > 0.3) -> FIRE
    if (mat_id == MAT_OIL && exp_heat > OIL_IGNITE_EXPOSURE) {
        pi = MAKE_PACKED(MAT_FIRE, GAS);
        packed_info[i] = pi;
        temperature[i] = FIRE_TEMPERATURE;
        lifetime[i] = OIL_FIRE_LIFETIME;
        return;
    }

    // GUNPOWDER (mat=15, exposure_heat > 0.1) -> FIRE + explosion
    if (mat_id == MAT_GUNPOWDER && exp_heat > GUNPOWDER_EXPOSURE) {
        pi = MAKE_PACKED(MAT_FIRE, GAS);
        packed_info[i] = pi;
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
            float4 v = velocity[i];
            v.x += dx * inv_len;
            v.y += dy * inv_len;
            v.z += dz * inv_len;
            velocity[i] = v;
        }
        return;
    }

    // ---- Sand wetting/drying transitions ----
    if (mat_id == MAT_SAND && exp_corrode > SAND_WET_THRESHOLD) {
        packed_info[i] = MAKE_PACKED(MAT_WET_SAND, GRANULAR);
        return;
    }
    if (mat_id == MAT_WET_SAND && exp_corrode > WETSAND_MUD_THRESHOLD) {
        packed_info[i] = MAKE_PACKED(MAT_MUD, FLUID);
        return;
    }
    if (mat_id == MAT_MUD && exp_corrode < MUD_DRY_THRESHOLD
            * (1.0f + DRYING_TEMP_ACCEL * fmaxf(temp - 293.0f, 0.0f))) {
        packed_info[i] = MAKE_PACKED(MAT_WET_SAND, GRANULAR);
        return;
    }
    if (mat_id == MAT_WET_SAND && exp_corrode < WETSAND_DRY_THRESHOLD
            * (1.0f + DRYING_TEMP_ACCEL * fmaxf(temp - 293.0f, 0.0f))) {
        packed_info[i] = MAKE_PACKED(MAT_SAND, GRANULAR);
        return;
    }

    // ---- Corrosion: health -= exposure_corrode * dt ----
    // Whitelist: only these materials can be corroded by acid
    if (exp_corrode > 0.0f && (mat_id == MAT_STONE || mat_id == MAT_METAL ||
        mat_id == MAT_WOOD || mat_id == MAT_DIRT || mat_id == MAT_GRAVEL ||
        mat_id == MAT_ICE || mat_id == MAT_OIL || mat_id == MAT_GUNPOWDER)) {
        float damage = exp_corrode * dt;
        hlth -= damage;
        if (hlth <= 0.0f) {
            // Particle dies from corrosion -- add to freelist
            packed_info[i] = MAKE_PACKED(MAT_DEAD, STATIC);
            health[i] = 0.0f;
            if (dead_indices != 0 && dead_count != 0) {
                uint idx = atomicAdd(dead_count, 1u);
                dead_indices[idx] = i;
            }
            return;
        }
        health[i] = hlth;
    }

    // ---- Acid consumption: acid loses health when corroding ----
    if (mat_id == MAT_ACID && exp_corrode > 0.0f) {
        hlth -= exp_corrode * dt * 0.5f;
        if (hlth <= 0.0f) {
            packed_info[i] = MAKE_PACKED(MAT_DEAD, STATIC);
            health[i] = 0.0f;
            if (dead_indices != 0 && dead_count != 0) {
                uint idx = atomicAdd(dead_count, 1u);
                dead_indices[idx] = i;
            }
            return;
        }
        health[i] = hlth;
    }

    // ---- GAS lifetime decay ----
    int behavior = GET_BEHAVIOR(pi);
    if (behavior == GAS && lt > 0.0f) {
        lt -= dt;
        if (lt <= 0.0f) {
            if (mat_id == MAT_FIRE) {
                // Fire expires -> becomes SMOKE with new lifetime
                packed_info[i] = MAKE_PACKED(MAT_SMOKE, GAS);
                lifetime[i] = 3.0f;
                temperature[i] = fminf(temp, 500.0f);  // cool down
            } else {
                // Other gas expired -- add to freelist
                packed_info[i] = MAKE_PACKED(MAT_DEAD, STATIC);
                lifetime[i] = 0.0f;
                health[i] = 0.0f;
                if (dead_indices != 0 && dead_count != 0) {
                    uint idx = atomicAdd(dead_count, 1u);
                    dead_indices[idx] = i;
                }
            }
            return;
        }
        lifetime[i] = lt;
    }
}
