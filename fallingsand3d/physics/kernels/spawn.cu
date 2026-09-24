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
#define SPAWN_OFFSET_FRAC 0.3f    // child offset radius / particle_spacing
#define SPAWN_VEL_SPREAD 0.1f     // outward child velocity spread (m/s)
#define SPAWN_PI_2_3     2.09439510f  // 2*pi/3
#define SPAWN_BOUNDARY_MARGIN 1e-4f   // matches BOUNDARY_MARGIN in sph_shared.cuh

/* Integer hash for the per-spawn rotation of the child ring. */
__device__ inline uint spawn_hash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return seed;
}

/* Axis-aligned clamp to the world box (same rule as clamp_boundary). */
__device__ inline float3 spawn_clamp(float3 p) {
    p.x = fmaxf(c_sim.world_min.x + SPAWN_BOUNDARY_MARGIN, fminf(p.x, c_sim.world_max.x - SPAWN_BOUNDARY_MARGIN));
    p.y = fmaxf(c_sim.world_min.y + SPAWN_BOUNDARY_MARGIN, fminf(p.y, c_sim.world_max.y - SPAWN_BOUNDARY_MARGIN));
    p.z = fmaxf(c_sim.world_min.z + SPAWN_BOUNDARY_MARGIN, fminf(p.z, c_sim.world_max.z - SPAWN_BOUNDARY_MARGIN));
    return p;
}

/* ======================================================================
 * K_SpawnGas kernel
 *
 * For each particle with HAS_SPAWN_FLAG set:
 *   1. Single atomicSub(&dead_count, N) to claim N freelist slots
 *      (all-or-nothing; no partial claims)
 *   2. If old value >= N, read dead_indices at the N claimed positions
 *   3. Write spawned steam particles to those slots. The children sit on a
 *      horizontal ring of radius SPAWN_OFFSET_FRAC * particle_spacing around
 *      the source, 120 degrees apart, with a hashed ring rotation. Each gets
 *      src_vel + (0, SPAWN_UPKICK, 0) plus SPAWN_VEL_SPREAD along its ring
 *      direction (the spreads sum to zero, so momentum is conserved).
 *      Every per-slot field the solvers read is reset: position.w (steam rest
 *      density), and when the pointers are non-NULL the PBF predicted
 *      position, the FP16 velocity mirror, pressure, dye, angular velocity
 *      and the kappa / kappa_v / lambda warm-start fields.
 *   4. Mark source water particle as DEAD
 *   5. Clear SPAWN_GAS flag
 *   6. Set max_displacement to +inf bits so the next frame does a full sort
 *
 * If the freelist has fewer than N entries, restore the subtracted N
 * and skip (the water particle stays liquid; retried next substep).
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
    uint*           __restrict__ dead_count,      // atomic counter
    // --- Grid reuse tracker (may be NULL) ---
    uint*           __restrict__ max_displacement, // float bits of max |dx|^2 since last sort
    // --- Optional per-slot state (each may be NULL) ---
    float4*         __restrict__ predicted,        // PBF sorted_predicted_position (PBF only)
    uint2*          __restrict__ velocity_h,       // FP16 velocity mirror read by step2
    float*          __restrict__ pressure,         // WCSPH pre-computed pressure read by step2
    float4*         __restrict__ particle_dye,
    float4*         __restrict__ angular_velocity,
    float*          __restrict__ kappa,            // DFSPH density warm start
    float*          __restrict__ kappa_v,          // DFSPH divergence warm start
    float*          __restrict__ lambda_pbf        // PBF lambda
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];

    // Only process particles with SPAWN_GAS flag set
    if (!HAS_SPAWN_FLAG(pi)) return;

    // Clear the flag regardless of whether spawn succeeds
    pi = CLEAR_SPAWN_FLAG(pi);

    // Claim all SPAWN_N slots with a single all-or-nothing atomicSub.
    // If the returned old value is >= SPAWN_N, this thread exclusively owns
    // stack positions [old-SPAWN_N, old-1] and reads its slot indices from
    // them. If old < SPAWN_N there were not enough entries: restore exactly
    // what we subtracted and bail.
    //
    // Why all-or-nothing: a previous version popped slots one at a time and,
    // on partial success, "restored" the stack with a count-only
    // atomicAdd(dead_count, n_claimed). That raised the count over positions
    // whose entries had already been consumed by other racing spawners (or
    // over positions that never held valid entries), so two spawners could
    // be handed the same dead slot. A failed single-sub claim never consumes
    // entries, so the count always stays <= the number of valid (unclaimed)
    // entries below it.
    //
    // dead_count may transiently wrap negative (as int) while several
    // threads race a nearly-empty freelist; each failing thread adds back
    // exactly SPAWN_N, so the modular arithmetic always rebalances.
    uint claimed_slots[SPAWN_N];
    int old_val = (int)atomicSub(dead_count, (uint)SPAWN_N);
    if (old_val < SPAWN_N) {
        // Not enough free slots -- restore and skip. The flag was cleared,
        // but Reactions re-sets it next substep while the water stays hot.
        atomicAdd(dead_count, (uint)SPAWN_N);
        packed_info[i] = pi;
        return;
    }
    for (int k = 0; k < SPAWN_N; k++) {
        claimed_slots[k] = dead_indices[old_val - SPAWN_N + k];
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

    // Ring rotation: hashed from the source index and position so repeated
    // spawns do not all line up. The three directions sum to zero.
    uint seed = spawn_hash(i ^ __float_as_uint(src_pos.x) ^ (__float_as_uint(src_pos.z) * 0x9E3779B9u));
    float phi0 = (float)(seed & 0xFFFFu) * (SPAWN_PI_2_3 / 65536.0f);
    float ring_r = SPAWN_OFFSET_FRAC * c_sim.particle_spacing;
    float steam_rho0 = c_materials[MAT_STEAM].rest_density;
    float dt = c_sim.dt;

    // Write spawned particles to the claimed freelist slots
    for (int k = 0; k < SPAWN_N; k++) {
        uint slot = claimed_slots[k];

        float s_k, c_k;
        __sincosf(phi0 + (float)k * SPAWN_PI_2_3, &s_k, &c_k);

        // Position: on the ring around the source. position.w carries the
        // density WCSPH step2 reads as rho_i, so give it the steam rest
        // density (1200 in kernel-sum units) instead of the source water's
        // 2500. The next Step1 replaces it with the measured density.
        float3 cpos = spawn_clamp(make_float3(src_pos.x + ring_r * c_k,
                                              src_pos.y,
                                              src_pos.z + ring_r * s_k));
        position[slot] = make_float4(cpos.x, cpos.y, cpos.z, steam_rho0);

        // Velocity: source velocity + upward kick + outward spread
        float4 child_vel;
        child_vel.x = src_vel.x + SPAWN_VEL_SPREAD * c_k;
        child_vel.y = src_vel.y + SPAWN_UPKICK;
        child_vel.z = src_vel.z + SPAWN_VEL_SPREAD * s_k;
        child_vel.w = 0.0f;
        velocity[slot] = child_vel;
        veleval[slot] = child_vel;

        // PBF: K_PBF_Predict already ran and left the dead slot's old
        // position here. Finalize derives velocity from pred - pos, so
        // write the child's own prediction.
        if (predicted != NULL) {
            float3 pred = spawn_clamp(make_float3(cpos.x + dt * child_vel.x,
                                                  cpos.y + dt * child_vel.y,
                                                  cpos.z + dt * child_vel.z));
            predicted[slot] = make_float4(pred.x, pred.y, pred.z, 1.0f);
        }
        // Step2 reads neighbour velocity from the FP16 mirror.
        if (velocity_h != NULL) store_half4(&velocity_h[slot], child_vel);
        // Gas EOS p = k * max(rho - rho0, 0) is 0 at rest density.
        if (pressure != NULL) pressure[slot] = 0.0f;
        // Do not inherit the dead occupant's dye, spin or warm starts.
        if (particle_dye != NULL) particle_dye[slot] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (angular_velocity != NULL) angular_velocity[slot] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (kappa != NULL) kappa[slot] = 0.0f;
        if (kappa_v != NULL) kappa_v[slot] = 0.0f;
        if (lambda_pbf != NULL) lambda_pbf[slot] = 0.0f;

        // Scalar properties
        mass[slot] = child_mass;
        temperature[slot] = SPAWN_TEMP;
        health[slot] = 1.0f;
        lifetime[slot] = SPAWN_LIFETIME;
        density[slot] = steam_rho0;
        shear_rate[slot] = 0.0f;
        sleep_counter[slot] = 0;

        // Color
        float4 child_color;
        child_color.x = cr;
        child_color.y = cg;
        child_color.z = cb;
        child_color.w = (float)GAS * 0.25f;  // behavior code, as behavior_to_alpha
        color[slot] = child_color;

        // Material info
        packed_info[slot] = steam_pi;
    }

    // The spawned slots jump from wherever the dead particle was parked to
    // the source position, and Integrate does not see that jump. Write the
    // bits of +inf into the displacement tracker so the next grid-reuse
    // decision forces a full sort (bd-r5-review-n28.46). Positive float bits
    // order like uints, so +inf beats any finite displacement.
    if (max_displacement != NULL) {
        atomicMax(max_displacement, 0x7F800000u);
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
