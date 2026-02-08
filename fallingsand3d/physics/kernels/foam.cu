/*
 * foam.cu -- Secondary particle system: spray, foam, bubbles.
 *
 * Two kernels:
 *   K_FoamGenerate: per FLUID particle, emit secondary particles based on
 *                   trapped air, wave crest, and kinetic energy criteria.
 *   K_FoamPhysics:  per foam particle, simple physics (no neighbor loops).
 *   K_FoamCompact:  remove dead foam particles (lifetime <= 0).
 *
 * Foam types (stored in foam_position.w as float):
 *   0.0 = SPRAY  (ballistic: gravity + air drag)
 *   1.0 = FOAM   (surface: follow fluid surface, diffuse, fade)
 *   2.0 = BUBBLE (buoyant: rise toward surface, dissolve)
 */

#include "common.cuh"

/* Foam generation parameters (uploaded via constant memory) */
struct FoamParams {
    float k_ta;           // trapped air coefficient
    float k_wc;           // wave crest coefficient
    float k_ke;           // kinetic energy coefficient
    float threshold;      // generation threshold (phi > threshold -> emit)
    float spray_lifetime; // seconds
    float foam_lifetime;  // seconds
    float bubble_lifetime;// seconds
    float drag_coeff;     // air drag for spray
    float buoyancy;       // buoyancy force for bubbles
    float diffusion;      // lateral diffusion for foam
    float spawn_jitter;   // random offset for spawned position
    int   max_foam;       // max foam particles (pool size)
    float dt;             // simulation timestep
    float _pad0;          // padding to 56 bytes
};

__constant__ FoamParams c_foam;


/* ======================================================================
 * Random number generation (simple hash-based)
 * ====================================================================== */

__device__ inline float hash_rand(uint seed, uint salt) {
    uint h = seed ^ (salt * 2654435761u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return float(h & 0xFFFFFF) / float(0xFFFFFF);  // [0, 1]
}


/* ======================================================================
 * K_FoamGenerate -- per FLUID particle, emit secondary particles
 *
 * Inputs: sorted_position, sorted_velocity, sorted_normal, sorted_packed_info
 * Output: foam_position, foam_velocity (appended via atomicAdd on foam_count)
 * ====================================================================== */

extern "C" __global__
void K_FoamGenerate(
    const float4* __restrict__ sorted_position,
    const float4* __restrict__ sorted_velocity,
    const float4* __restrict__ sorted_normal,
    const uint*   __restrict__ sorted_packed_info,
    float4*       foam_position,
    float4*       foam_velocity,
    uint*         foam_count,     // atomic counter
    int           num_particles,
    uint          frame_seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    uint info = sorted_packed_info[i];
    int behavior = GET_BEHAVIOR(info);
    if (behavior != FLUID) return;

    float4 pos_i = sorted_position[i];
    float4 vel_i = sorted_velocity[i];
    float4 nrm_i = sorted_normal[i];

    float vx = vel_i.x, vy = vel_i.y, vz = vel_i.z;
    float speed_sq = vx*vx + vy*vy + vz*vz;
    float speed = sqrtf(speed_sq);

    float nx = nrm_i.x, ny = nrm_i.y, nz = nrm_i.z;
    float n_len = sqrtf(nx*nx + ny*ny + nz*nz);
    float neighbor_count = nrm_i.w;

    // Trapped air: high relative velocity (approximated by speed in splash zones)
    float trapped_air = speed;

    // Wave crest: velocity pointing away from surface normal (= away from fluid bulk)
    float wave_crest = 0.0f;
    if (n_len > 1e-6f) {
        float inv_nlen = 1.0f / n_len;
        // dot(v, n_hat): positive = moving toward more fluid, negative = moving away
        float v_dot_n = (vx*nx + vy*ny + vz*nz) * inv_nlen;
        wave_crest = fmaxf(0.0f, -v_dot_n);  // negative dot = moving away from bulk
    }

    // Kinetic energy
    float kinetic = 0.5f * speed_sq;

    // Surface check: only generate near surface (low neighbor count)
    // Interior particles (high neighbor count) should not generate foam
    if (neighbor_count > 25.0f) return;

    float phi = c_foam.k_ta * trapped_air
              + c_foam.k_wc * wave_crest
              + c_foam.k_ke * kinetic;

    if (phi <= c_foam.threshold) return;

    // Claim a slot in the foam pool
    uint slot = atomicAdd(foam_count, 1u);
    if (slot >= (uint)c_foam.max_foam) {
        // Pool full, revert (won't be perfect under contention but good enough)
        atomicSub(foam_count, 1u);
        return;
    }

    // Determine foam type based on velocity direction
    // Upward velocity = spray, near-surface & slow = foam, downward = bubble
    float r = hash_rand((uint)i, frame_seed);
    float foam_type;
    float lifetime;

    if (vy > 0.5f && speed > 1.0f) {
        // Fast upward = SPRAY
        foam_type = 0.0f;
        lifetime = c_foam.spray_lifetime * (0.5f + r);
    } else if (neighbor_count < 15.0f) {
        // Surface with low neighbors = FOAM
        foam_type = 1.0f;
        lifetime = c_foam.foam_lifetime * (0.5f + r);
    } else {
        // Interior near surface = BUBBLE
        foam_type = 2.0f;
        lifetime = c_foam.bubble_lifetime * (0.5f + r);
    }

    // Spawn position: slightly offset from parent in velocity direction
    float jitter_x = (hash_rand((uint)i, frame_seed + 1u) - 0.5f) * c_foam.spawn_jitter;
    float jitter_y = (hash_rand((uint)i, frame_seed + 2u) - 0.5f) * c_foam.spawn_jitter;
    float jitter_z = (hash_rand((uint)i, frame_seed + 3u) - 0.5f) * c_foam.spawn_jitter;

    foam_position[slot] = make_float4(
        pos_i.x + jitter_x,
        pos_i.y + jitter_y,
        pos_i.z + jitter_z,
        foam_type
    );

    // Spawn velocity: inherit parent velocity with some upward bias for spray
    float vel_scale = (foam_type == 0.0f) ? 1.2f : 0.5f;
    foam_velocity[slot] = make_float4(
        vel_i.x * vel_scale,
        vel_i.y * vel_scale + (foam_type == 0.0f ? 1.0f : 0.0f),
        vel_i.z * vel_scale,
        lifetime
    );
}


/* ======================================================================
 * K_FoamPhysics -- per foam particle, simple physics (no SPH)
 *
 * SPRAY:  gravity + air drag (ballistic trajectory)
 * FOAM:   slow drift, lateral diffusion, fade
 * BUBBLE: buoyancy upward, dissolve after lifetime
 * ====================================================================== */

extern "C" __global__
void K_FoamPhysics(
    float4* foam_position,
    float4* foam_velocity,
    const uint* __restrict__ foam_count  // device pointer to current count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num_foam = (int)foam_count[0];
    if (i >= num_foam) return;

    float4 pos = foam_position[i];
    float4 vel = foam_velocity[i];

    float foam_type = pos.w;
    float lifetime = vel.w;

    // Decrement lifetime
    lifetime -= c_foam.dt;
    if (lifetime <= 0.0f) {
        // Mark as dead (negative lifetime)
        foam_velocity[i].w = -1.0f;
        return;
    }

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    if (foam_type < 0.5f) {
        // SPRAY: gravity + air drag
        ax = -vel.x * c_foam.drag_coeff;
        ay = c_sim.gravity.y - vel.y * c_foam.drag_coeff;
        az = -vel.z * c_foam.drag_coeff;
    } else if (foam_type < 1.5f) {
        // FOAM: gentle gravity + horizontal diffusion + damping
        ax = -vel.x * 2.0f;  // strong horizontal damping
        ay = c_sim.gravity.y * 0.1f;  // weak gravity (sits on surface)
        az = -vel.z * 2.0f;
    } else {
        // BUBBLE: buoyancy + gentle drag
        ax = -vel.x * c_foam.drag_coeff;
        ay = c_foam.buoyancy - vel.y * c_foam.drag_coeff * 0.5f;
        az = -vel.z * c_foam.drag_coeff;
    }

    // Symplectic Euler integration
    float dt = c_foam.dt;
    vel.x += ax * dt;
    vel.y += ay * dt;
    vel.z += az * dt;

    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    // World boundary clamp
    float3 wmin = c_sim.world_min;
    float3 wmax = c_sim.world_max;
    if (pos.x < wmin.x) { pos.x = wmin.x; vel.x = 0.0f; }
    if (pos.x > wmax.x) { pos.x = wmax.x; vel.x = 0.0f; }
    if (pos.y < wmin.y) { pos.y = wmin.y; vel.y = 0.0f; lifetime = -1.0f; }  // kill at floor
    if (pos.y > wmax.y) { pos.y = wmax.y; vel.y = 0.0f; }
    if (pos.z < wmin.z) { pos.z = wmin.z; vel.z = 0.0f; }
    if (pos.z > wmax.z) { pos.z = wmax.z; vel.z = 0.0f; }

    // Write back (preserve foam_type in pos.w, lifetime in vel.w)
    foam_position[i] = make_float4(pos.x, pos.y, pos.z, foam_type);
    foam_velocity[i] = make_float4(vel.x, vel.y, vel.z, lifetime);
}


/* ======================================================================
 * K_FoamCompact -- stream compaction: remove dead foam particles
 *
 * Two-pass approach:
 *   Pass 1 (this kernel): compute alive prefix + scatter alive to output
 *   Using simple atomic approach for simplicity (foam count is small)
 * ====================================================================== */

extern "C" __global__
void K_FoamCompact(
    const float4* __restrict__ foam_position_in,
    const float4* __restrict__ foam_velocity_in,
    float4*       foam_position_out,
    float4*       foam_velocity_out,
    uint*         alive_count,       // output: new count (atomic)
    const uint*   __restrict__ foam_count  // device pointer to current count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num_foam = (int)foam_count[0];
    if (i >= num_foam) return;

    float lifetime = foam_velocity_in[i].w;
    if (lifetime <= 0.0f) return;  // dead, skip

    uint slot = atomicAdd(alive_count, 1u);
    foam_position_out[slot] = foam_position_in[i];
    foam_velocity_out[slot] = foam_velocity_in[i];
}
