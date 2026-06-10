/*
 * spawn.cu -- K_SpawnGas kernel: spawn/kill system for large phase transitions.
 *
 * When a water particle has HAS_SPAWN_FLAG set (by Reactions kernel when
 * temp > 373K), this kernel kills the water particle and spawns N steam
 * particles in its place using a GPU freelist of dead particle indices.
 *
 * Freelist design:
 *   dead_indices[max_particles]  -- uint32 array of dead particle indices
 *   dead_count                   -- uint32 atomic counter (number of entries)
 *
 * The Reactions kernel populates the freelist when particles die (corrosion
 * death, gas lifetime expiry). This kernel consumes from it via atomicSub.
 *
 * Runs on SORTED arrays after Reactions, before Step2. The spawned particles
 * are written to sorted arrays at the claimed freelist slots.
 *
 * Constant memory used:
 *   c_sim       -- SimParams (for dt, particle_mass)
 *   c_materials -- MaterialProps[32] (for steam properties)
 */

#include "common.cuh"

/* ======================================================================
 * Material IDs (must match materials.py)
 * ====================================================================== */

#define MAT_DEAD       0
#define MAT_WATER      5
#define MAT_STEAM     12

/* ======================================================================
 * Spawn configuration
 * ====================================================================== */

#define SPAWN_N          3        // number of steam particles per water particle
#define SPAWN_TEMP     373.0f     // steam temperature (K)
#define SPAWN_LIFETIME   5.0f     // steam lifetime (seconds)
#define SPAWN_UPKICK     2.0f     // upward velocity kick (m/s)

/* ======================================================================
 * K_SpawnGas kernel
 *
 * For each particle with HAS_SPAWN_FLAG set:
 *   1. atomicSub(&dead_count, N) to claim N freelist slots
 *   2. If result >= 0, read dead_indices to get target slots
 *   3. Write spawned steam particles to those slots
 *   4. Mark source water particle as DEAD and add it to freelist
 *   5. Clear SPAWN_GAS flag
 *
 * If freelist is exhausted (atomicSub result < 0), restore count and skip.
 * ====================================================================== */

extern "C" __global__
void K_SpawnGas(
    uint            numParticles,
    // --- Sorted arrays (read+write in-place) ---
    uint*           __restrict__ packed_info,
    float4*         __restrict__ position,
    float4*         __restrict__ velocity,
    float4*         __restrict__ veleval,
    float*          __restrict__ mass,
    float*          __restrict__ temperature,
    float*          __restrict__ health,
    float*          __restrict__ lifetime,
    float4*         __restrict__ color,
    unsigned char*  __restrict__ sleep_counter,
    float*          __restrict__ density,
    float*          __restrict__ shear_rate,
    // --- Freelist (shared with Reactions) ---
    uint*           __restrict__ dead_indices,   // array of dead particle indices
    uint*           __restrict__ dead_count       // atomic counter
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];

    // Only process particles with SPAWN_GAS flag set
    if (!HAS_SPAWN_FLAG(pi)) return;

    // Clear the flag regardless of whether spawn succeeds
    pi = CLEAR_SPAWN_FLAG(pi);

    // Claim N slots from the freelist atomically using a CAS loop.
    // Pre-check that dead_count >= SPAWN_N, then atomicCAS to decrement by N.
    // This prevents the uint32 underflow (wrap to 0xFFFFFFFF) that occurred
    // when multiple threads raced on atomicSub with dead_count near zero.
    uint claimed_slots[SPAWN_N];

    // Atomically claim N slots: CAS loop decrements dead_count by SPAWN_N only
    // if the current value is >= SPAWN_N.
    uint cur = *dead_count;  // relaxed peek (speculative)
    bool claimed = false;
    for (int retry = 0; retry < 32; retry++) {
        if (cur < (uint)SPAWN_N) {
            // Not enough slots -- do not attempt decrement
            break;
        }
        uint prev = atomicCAS(dead_count, cur, cur - (uint)SPAWN_N);
        if (prev == cur) {
            // CAS succeeded: we own slots [cur-N, cur-1]
            for (int k = 0; k < SPAWN_N; k++) {
                claimed_slots[k] = dead_indices[cur - 1 - k];
            }
            claimed = true;
            break;
        }
        // Another thread changed dead_count; reload and retry
        cur = prev;
    }

    if (!claimed) {
        packed_info[i] = pi;
        return;
    }

    // All N slots claimed successfully
    // Read the source particle's properties
    float4 src_pos = position[i];
    float4 src_vel = velocity[i];
    float  src_mass = mass[i];

    // Compute per-spawned-particle mass (conservation of mass)
    float child_mass = src_mass / (float)SPAWN_N;

    // Steam material packed info
    uint steam_pi = MAKE_PACKED(MAT_STEAM, GAS);

    // Steam color from c_materials
    float cr = c_materials[MAT_STEAM].color_r;
    float cg = c_materials[MAT_STEAM].color_g;
    float cb = c_materials[MAT_STEAM].color_b;

    // Write spawned particles to the claimed freelist slots
    for (int k = 0; k < SPAWN_N; k++) {
        uint slot = claimed_slots[k];

        // Position: same as source (they'll spread via SPH forces)
        position[slot] = src_pos;

        // Velocity: source velocity + upward kick
        float4 child_vel;
        child_vel.x = src_vel.x;
        child_vel.y = src_vel.y + SPAWN_UPKICK;
        child_vel.z = src_vel.z;
        child_vel.w = 0.0f;
        velocity[slot] = child_vel;
        veleval[slot] = child_vel;

        // Scalar properties
        mass[slot] = child_mass;
        temperature[slot] = SPAWN_TEMP;
        health[slot] = 1.0f;
        lifetime[slot] = SPAWN_LIFETIME;
        density[slot] = c_materials[MAT_STEAM].rest_density;
        shear_rate[slot] = 0.0f;
        sleep_counter[slot] = 0;

        // Color
        float4 child_color;
        child_color.x = cr;
        child_color.y = cg;
        child_color.z = cb;
        child_color.w = 1.0f;
        color[slot] = child_color;

        // Material info
        packed_info[slot] = steam_pi;
    }

    // Mark source water particle as DEAD.
    // Use STATIC (not FLUID) so dead particles are excluded from the SPH
    // fluid neighbor loop and cannot pollute density sums until compaction.
    // (NOT added back to freelist here -- would race with concurrent reads
    //  from dead_indices. Compaction (US-029) reclaims dead particles.)
    packed_info[i] = MAKE_PACKED(MAT_DEAD, STATIC);
    health[i] = 0.0f;
    mass[i] = 0.0f;
}
