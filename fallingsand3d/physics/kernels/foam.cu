/*
 * foam.cu -- Secondary particle system: spray, foam, bubbles.
 *
 * Kernels:
 *   K_FoamGenerate: per FLUID particle, emit secondary particles based on
 *                   trapped air, wave crest, and kinetic energy criteria.
 *   K_FoamPhysics:  per foam particle, simple physics (no neighbor loops).
 *   K_FoamCompact:  remove dead foam particles (lifetime <= 0).
 *   K_FoamClearTail: mark every slot past the compacted count as dead.
 *   K_IncrementCounter: add 1 to the per-substep RNG counter on the device.
 *
 * Foam types (stored in foam_position.w as float):
 *   0.0 = SPRAY  (ballistic: gravity + air drag)
 *   1.0 = FOAM   (surface: follow fluid surface, diffuse, fade)
 *   2.0 = BUBBLE (buoyant: rise toward surface, dissolve)
 *  -1.0 = DEAD   (foam.vert culls it; set on death and on the stale tail)
 */

#include "sph_shared.cuh"  // common.cuh + eval_sdf / sdf_normal

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
    float k_rate;         // emission rate per unit (phi - threshold), 1/s
    float foam_buoyancy;  // upward accel for FOAM inside fluid, m/s^2
    float foam_water_drag;// vertical drag for FOAM inside fluid, 1/s
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
 * Inputs: sorted_position, sorted_velocity (or its half4 mirror),
 *         sorted_normal, sorted_packed_info
 * Output: foam_position, foam_velocity (appended via atomicAdd on foam_count)
 *
 * The RNG seed is read from frame_ptr[0], the per-substep counter that
 * K_IncrementCounter advances on the device (the same one K_Reactions and
 * the wake kernels read), so the launch has no per-substep host scalar.
 *
 * Load order: packed_info, then the normal, whose .w (neighbor count)
 * rejects interior particles (85 to 90% of FLUID) before position and
 * velocity are read. nvcc does not sink those loads past the branch on its
 * own (checked in SASS, bd-r5-review-n28.179). When sorted_velocity_h is
 * not NULL, velocity comes from that 8-byte half4 mirror instead of the
 * 16-byte float4 array.
 * ====================================================================== */

extern "C" __global__
void K_FoamGenerate(
    const float4* __restrict__ sorted_position,
    const float4* __restrict__ sorted_velocity,
    const uint2*  __restrict__ sorted_velocity_h,  // half4 mirror, or NULL
    const float4* __restrict__ sorted_normal,
    const uint*   __restrict__ sorted_packed_info,
    float4*       foam_position,
    float4*       foam_velocity,
    uint*         foam_count,     // atomic counter
    int           num_particles,
    const uint*   __restrict__ frame_ptr,  // per-substep RNG counter
    float         dt_emit         // sim time covered by this launch (s)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    uint info = sorted_packed_info[i];
    int behavior = GET_BEHAVIOR(info);
    if (behavior != FLUID) return;

    // Surface check first: only generate near surface (low neighbor count).
    // Interior particles (high neighbor count) never generate foam.
    float4 nrm_i = sorted_normal[i];
    float neighbor_count = nrm_i.w;
    if (neighbor_count > 25.0f) return;

    float4 vel_i = sorted_velocity_h ? load_half4(&sorted_velocity_h[i])
                                     : sorted_velocity[i];

    float vx = vel_i.x, vy = vel_i.y, vz = vel_i.z;
    float speed_sq = vx*vx + vy*vy + vz*vz;
    float speed = sqrtf(speed_sq);

    float nx = nrm_i.x, ny = nrm_i.y, nz = nrm_i.z;
    float n_len = sqrtf(nx*nx + ny*ny + nz*nz);

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

    float phi = c_foam.k_ta * trapped_air
              + c_foam.k_wc * wave_crest
              + c_foam.k_ke * kinetic;

    if (phi <= c_foam.threshold) return;

    // Emit with probability p = k_rate * (phi - threshold) * dt_emit, so the
    // expected rate is k_rate * (phi - threshold) per second whatever the
    // substep count (Ihmsen et al. 2012 use n = dt * phi).
    float p_emit = fminf(1.0f, c_foam.k_rate * (phi - c_foam.threshold) * dt_emit);
    uint frame_seed = frame_ptr[0];
    if (hash_rand((uint)i, frame_seed + 4u) >= p_emit) return;

    // Pool full: skip the atomicAdd/atomicSub pair. The volatile read sees
    // the count as it stands; a stale value only lets a thread reach the
    // atomic below, which still guards the pool bound.
    if (*((volatile uint*)foam_count) >= (uint)c_foam.max_foam) return;

    // Claim a slot in the foam pool
    uint slot = atomicAdd(foam_count, 1u);
    if (slot >= (uint)c_foam.max_foam) {
        // Pool full, revert (won't be perfect under contention but good enough)
        atomicSub(foam_count, 1u);
        return;
    }

    float4 pos_i = sorted_position[i];

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
 * FOAM:   slow drift, lateral diffusion, fade. Inside fluid it gets
 *         c_foam.foam_buoyancy up and c_foam.foam_water_drag on vy, so it
 *         climbs back to the free surface; outside it falls at 0.1 g.
 * BUBBLE: buoyancy upward, dissolve after lifetime. A BUBBLE outside fluid
 *         becomes FOAM with vy = 0, so it stays at the surface instead of
 *         rising to the ceiling.
 *
 * Inside fluid (bd-r5-review-n28.193) means the particle's own hash cell
 * in the SPH grid built this substep holds at least one particle:
 * cell_start[h] < cell_end[h] (no sentinel, see common.cuh). Two 4 B
 * reads, no neighbour loop. Any SPH particle counts, and a hash collision
 * with a far occupied cell reads as inside; both only keep a particle in
 * its in-fluid state a little longer. With cell_start == NULL the kernel
 * has no fluid information: BUBBLE always rises and FOAM always falls.
 * ====================================================================== */

extern "C" __global__
void K_FoamPhysics(
    float4* foam_position,
    float4* foam_velocity,
    const uint* __restrict__ foam_count,  // device pointer to current count
    const uint* __restrict__ cell_start,  // SPH hash grid, or NULL
    const uint* __restrict__ cell_end
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
        // Mark as dead: negative lifetime for compaction, negative type so
        // foam.vert stops drawing it this frame.
        foam_velocity[i].w = -1.0f;
        foam_position[i].w = -1.0f;
        return;
    }

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // FOAM and BUBBLE need to know whether they are inside the fluid.
    bool in_fluid = false;
    bool have_grid = (cell_start != NULL);
    if (have_grid && foam_type > 0.5f) {
        uint h = spatialHash(calcGridCell(make_float3(pos.x, pos.y, pos.z)));
        in_fluid = cell_start[h] < cell_end[h];
    }
    if (have_grid && foam_type > 1.5f && !in_fluid) {
        // BUBBLE left the fluid: it is now surface FOAM at rest vertically.
        foam_type = 1.0f;
        vel.y = 0.0f;
    }

    if (foam_type < 0.5f) {
        // SPRAY: gravity + air drag
        ax = -vel.x * c_foam.drag_coeff;
        ay = c_sim.gravity.y - vel.y * c_foam.drag_coeff;
        az = -vel.z * c_foam.drag_coeff;
    } else if (foam_type < 1.5f) {
        // FOAM: gentle gravity + horizontal diffusion + damping
        ax = -vel.x * 2.0f;  // strong horizontal damping
        ay = in_fluid
           ? c_foam.foam_buoyancy - vel.y * c_foam.foam_water_drag  // climb to surface
           : c_sim.gravity.y * 0.1f;  // weak gravity (sits on surface)
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

    // SDF object collision (bd-r5-review-n28.194). A particle inside or
    // within BOUNDARY_MARGIN of an object moves to BOUNDARY_MARGIN outside
    // along the analytic normal, and the part of its velocity (relative to
    // the object) that points into the surface is removed. SPRAY that hits
    // an object becomes FOAM, so it rests on the surface instead of
    // bouncing. At most MAX_SDF_OBJECTS analytic evaluations, no memory
    // traffic; the loop is empty when no object is placed.
    if (lifetime > 0.0f) {
        for (int s = 0; s < c_num_sdf_objects; s++) {
            SDFObject obj = c_sdf_objects[s];
            float3 p = make_float3(pos.x, pos.y, pos.z);
            float dist = eval_sdf(p, obj);
            if (dist >= BOUNDARY_MARGIN) continue;
            float3 n = sdf_normal(p, obj);
            float push = BOUNDARY_MARGIN - dist;
            pos.x += push * n.x;
            pos.y += push * n.y;
            pos.z += push * n.z;

            // Object surface velocity at the contact: linear + omega x r.
            float3 ov = make_float3(obj.velocity.x, obj.velocity.y, obj.velocity.z);
            float w = obj.velocity.w;
            if (fabsf(w) > 1e-8f) {
                float3 om = make_float3(obj.angular_axis.x * w,
                                        obj.angular_axis.y * w,
                                        obj.angular_axis.z * w);
                float3 r = make_float3(pos.x - obj.pos_and_type.x,
                                       pos.y - obj.pos_and_type.y,
                                       pos.z - obj.pos_and_type.z);
                ov.x += om.y * r.z - om.z * r.y;
                ov.y += om.z * r.x - om.x * r.z;
                ov.z += om.x * r.y - om.y * r.x;
            }
            float vn = (vel.x - ov.x) * n.x + (vel.y - ov.y) * n.y + (vel.z - ov.z) * n.z;
            if (vn < 0.0f) {
                vel.x -= vn * n.x;
                vel.y -= vn * n.y;
                vel.z -= vn * n.z;
            }
            if (foam_type < 0.5f) foam_type = 1.0f;  // SPRAY -> FOAM
        }
    }

    // Write back (foam_type in pos.w, lifetime in vel.w). A floor kill
    // writes -1 in both, like the lifetime death above.
    if (lifetime <= 0.0f) foam_type = -1.0f;
    foam_position[i] = make_float4(pos.x, pos.y, pos.z, foam_type);
    foam_velocity[i] = make_float4(vel.x, vel.y, vel.z, lifetime);
}


/* ======================================================================
 * K_FoamCompact -- stream compaction: remove dead foam particles
 *
 * Two-pass approach:
 *   Pass 1 (this kernel): compute alive prefix + scatter alive to output
 *   Using simple atomic approach for simplicity (foam count is small)
 *
 * The output order is not the input order. ptxas already turns the
 * atomicAdd(alive_count, 1) into one atomic per warp (VOTEU.ANY + POPC +
 * SHFL in the SASS), so hand-written warp aggregation gains nothing
 * (checked for bd-r5-review-n28.179).
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


/* ======================================================================
 * K_FoamClearTail -- after compaction, mark slots [alive_count, max_foam)
 * of the new buffer as dead (pos.w = -1).
 *
 * K_FoamCompact writes only [0, alive_count). The rest of the buffer still
 * holds positions from an earlier cycle, and the draw count is a deferred
 * readback that can be larger than alive_count, so without this those old
 * sprites flash for a frame. foam.vert culls w < 0.
 * ====================================================================== */

extern "C" __global__
void K_FoamClearTail(
    float4*       foam_position,
    const uint*   __restrict__ alive_count,
    int           max_foam
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_foam || i < (int)alive_count[0]) return;
    foam_position[i].w = -1.0f;
}


/* ======================================================================
 * K_IncrementCounter -- one thread: counter[0] += 1.
 *
 * Simulation launches it as the first node of the substep body, so every
 * substep (graph replay or direct launch) advances the RNG counter by
 * exactly one without a host-side fill.
 * ====================================================================== */

extern "C" __global__
void K_IncrementCounter(uint* counter) {
    if (blockIdx.x == 0 && threadIdx.x == 0) counter[0] += 1u;
}
