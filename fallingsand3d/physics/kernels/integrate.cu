/*
 * integrate.cu -- K_Integrate kernel: symplectic Euler integration,
 * impulse-style SDF box boundaries, GAS buoyancy/drag, velocity clamping,
 * color computation, and writeback to UNSORTED arrays via sort_indexes.
 *
 * Per-particle computation:
 *   1. Compute acceleration: accel = sph_force/mass + gravity + gas_buoyancy
 *   2. Symplectic Euler: vel_new = vel + dt * accel
 *   3. GAS drag: vel_new *= (1 - c_drag * dt)
 *   4. Velocity clamp: |vel| <= velocity_limit
 *   5. Position update: pos_new = pos + dt * (vel_new + xsph) for FLUID,
 *      pos_new = pos + dt * vel_new for others
 *   6. Impulse SDF boundary: 6 planes of box, project out, reflect normal vel
 *      with restitution, apply Coulomb friction to tangential vel
 *   7. Compute color from material base color, temperature tint, health fade
 *   8. Write pos, vel, color to UNSORTED arrays via sort_indexes[i]
 *
 * Skips STATIC particles (behavior_class == 3): early return, position unchanged.
 *
 * Constant memory used:
 *   c_sim       -- SimParams from common.cuh (gravity, dt, restitution, etc.)
 *   c_materials -- MaterialProps[32] from common.cuh (for color lookup)
 */

#include "common.cuh"

/* ======================================================================
 * Constants
 * ====================================================================== */

#define GAS_BUOYANCY_BETA  0.01f
#define GAS_AMBIENT_TEMP   293.0f
#define GAS_BUOYANCY_G     9.81f
#define GAS_DRAG_COEFF     2.0f
#define VELOCITY_LIMIT     50.0f
#define VELOCITY_LIMIT_SQ  (VELOCITY_LIMIT * VELOCITY_LIMIT)

/* ======================================================================
 * Impulse-style SDF boundary collision for axis-aligned box.
 *
 * For each of 6 planes: if pos penetrates wall, project out and
 * apply impulse-style velocity correction:
 *   - Normal velocity reflected with restitution coefficient
 *   - Tangential velocity reduced by Coulomb friction
 * ====================================================================== */

__device__ inline void sdf_box_boundary(
    float3& pos,
    float3& vel,
    float3 world_min,
    float3 world_max,
    float restitution,
    float mu_wall
) {
    // Process each axis: min then max boundary
    // X-axis
    if (pos.x < world_min.x) {
        pos.x = world_min.x;
        if (vel.x < 0.0f) {
            float vn = vel.x;                        // normal component (negative)
            float vt_y = vel.y, vt_z = vel.z;        // tangential components
            vel.x = -restitution * vn;                // reflect + restitution
            float tang_speed = sqrtf(vt_y * vt_y + vt_z * vt_z);
            if (tang_speed > 1e-8f) {
                float friction_impulse = mu_wall * fabsf(vn);
                float reduction = fminf(friction_impulse / tang_speed, 1.0f);
                vel.y *= (1.0f - reduction);
                vel.z *= (1.0f - reduction);
            }
        }
    }
    if (pos.x > world_max.x) {
        pos.x = world_max.x;
        if (vel.x > 0.0f) {
            float vn = vel.x;
            float vt_y = vel.y, vt_z = vel.z;
            vel.x = -restitution * vn;
            float tang_speed = sqrtf(vt_y * vt_y + vt_z * vt_z);
            if (tang_speed > 1e-8f) {
                float friction_impulse = mu_wall * fabsf(vn);
                float reduction = fminf(friction_impulse / tang_speed, 1.0f);
                vel.y *= (1.0f - reduction);
                vel.z *= (1.0f - reduction);
            }
        }
    }

    // Y-axis
    if (pos.y < world_min.y) {
        pos.y = world_min.y;
        if (vel.y < 0.0f) {
            float vn = vel.y;
            float vt_x = vel.x, vt_z = vel.z;
            vel.y = -restitution * vn;
            float tang_speed = sqrtf(vt_x * vt_x + vt_z * vt_z);
            if (tang_speed > 1e-8f) {
                float friction_impulse = mu_wall * fabsf(vn);
                float reduction = fminf(friction_impulse / tang_speed, 1.0f);
                vel.x *= (1.0f - reduction);
                vel.z *= (1.0f - reduction);
            }
        }
    }
    if (pos.y > world_max.y) {
        pos.y = world_max.y;
        if (vel.y > 0.0f) {
            float vn = vel.y;
            float vt_x = vel.x, vt_z = vel.z;
            vel.y = -restitution * vn;
            float tang_speed = sqrtf(vt_x * vt_x + vt_z * vt_z);
            if (tang_speed > 1e-8f) {
                float friction_impulse = mu_wall * fabsf(vn);
                float reduction = fminf(friction_impulse / tang_speed, 1.0f);
                vel.x *= (1.0f - reduction);
                vel.z *= (1.0f - reduction);
            }
        }
    }

    // Z-axis
    if (pos.z < world_min.z) {
        pos.z = world_min.z;
        if (vel.z < 0.0f) {
            float vn = vel.z;
            float vt_x = vel.x, vt_y = vel.y;
            vel.z = -restitution * vn;
            float tang_speed = sqrtf(vt_x * vt_x + vt_y * vt_y);
            if (tang_speed > 1e-8f) {
                float friction_impulse = mu_wall * fabsf(vn);
                float reduction = fminf(friction_impulse / tang_speed, 1.0f);
                vel.x *= (1.0f - reduction);
                vel.y *= (1.0f - reduction);
            }
        }
    }
    if (pos.z > world_max.z) {
        pos.z = world_max.z;
        if (vel.z > 0.0f) {
            float vn = vel.z;
            float vt_x = vel.x, vt_y = vel.y;
            vel.z = -restitution * vn;
            float tang_speed = sqrtf(vt_x * vt_x + vt_y * vt_y);
            if (tang_speed > 1e-8f) {
                float friction_impulse = mu_wall * fabsf(vn);
                float reduction = fminf(friction_impulse / tang_speed, 1.0f);
                vel.x *= (1.0f - reduction);
                vel.y *= (1.0f - reduction);
            }
        }
    }
}

/* ======================================================================
 * Particle color computation
 *
 * base_color = c_materials[mat_id].(color_r, color_g, color_b)
 * Hot tint (T > 293K): blend toward red proportional to temperature
 * Health fade: multiply by health [0,1]
 * ====================================================================== */

__device__ inline float4 compute_color(uint mat_id, float temperature, float health) {
    float r = c_materials[mat_id].color_r;
    float g = c_materials[mat_id].color_g;
    float b = c_materials[mat_id].color_b;

    // Hot tint: blend toward red/orange for high temperatures
    if (temperature > 293.0f) {
        float t_excess = fminf((temperature - 293.0f) / 1000.0f, 1.0f);
        r = r + (1.0f - r) * t_excess;
        g = g * (1.0f - 0.5f * t_excess);
        b = b * (1.0f - 0.8f * t_excess);
    }

    // Health fade: multiply all channels by health
    float h = fmaxf(fminf(health, 1.0f), 0.0f);
    r *= h;
    g *= h;
    b *= h;

    return make_float4(r, g, b, 1.0f);
}

/* ======================================================================
 * K_Integrate kernel
 * ====================================================================== */

extern "C" __global__
void K_Integrate(
    uint            numParticles,
    // --- Sorted inputs (read) ---
    const float4*   __restrict__ sorted_position,      // sorted positions
    const float4*   __restrict__ sorted_velocity,      // sorted velocities
    const float4*   __restrict__ sorted_veleval,       // sorted veleval (XSPH-corrected for FLUID)
    const float4*   __restrict__ sorted_sph_force,     // sorted SPH force from Step2
    const float*    __restrict__ sorted_mass,           // sorted per-particle mass
    const uint*     __restrict__ sorted_packed_info,    // sorted packed_info
    const float*    __restrict__ sorted_temperature,    // sorted temperature
    const float*    __restrict__ sorted_health,         // sorted health
    const uint*     __restrict__ sort_indexes,          // sort_indexes[sorted_i] = original_i
    // --- Unsorted outputs (write via sort_indexes) ---
    float4*         __restrict__ position_out,          // unsorted position
    float4*         __restrict__ velocity_out,          // unsorted velocity
    float4*         __restrict__ color_out              // unsorted color
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    // --- Read packed_info ---
    uint pi = sorted_packed_info[i];
    int behavior = GET_BEHAVIOR(pi);
    uint mat_id = GET_MATERIAL_ID(pi);

    // Unsorted index for writeback
    uint orig_idx = sort_indexes[i];

    // --- Skip STATIC particles ---
    if (behavior == STATIC) {
        // Write position unchanged, zero velocity, compute color
        float4 pos4 = sorted_position[i];
        position_out[orig_idx] = pos4;
        velocity_out[orig_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float temp = sorted_temperature[i];
        float hlth = sorted_health[i];
        color_out[orig_idx] = compute_color(mat_id, temp, hlth);
        return;
    }

    // --- Read particle data ---
    float4 pos4 = sorted_position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);

    float4 vel4 = sorted_velocity[i];
    float3 vel = make_float3(vel4.x, vel4.y, vel4.z);

    float4 veleval4 = sorted_veleval[i];
    float3 veleval_xsph = make_float3(veleval4.x, veleval4.y, veleval4.z);

    float4 force4 = sorted_sph_force[i];
    float3 sph_force = make_float3(force4.x, force4.y, force4.z);

    float mass_i = sorted_mass[i];
    float temp = sorted_temperature[i];
    float hlth = sorted_health[i];

    float dt = c_sim.dt;

    // --- Compute acceleration ---
    float inv_mass = 1.0f / fmaxf(mass_i, 1e-12f);
    float3 accel = make_float3(
        sph_force.x * inv_mass + c_sim.gravity.x,
        sph_force.y * inv_mass + c_sim.gravity.y,
        sph_force.z * inv_mass + c_sim.gravity.z
    );

    // GAS buoyancy: f_buoy = beta * (T - 293) * (0, 9.81, 0)
    if (behavior == GAS) {
        float buoyancy = GAS_BUOYANCY_BETA * (temp - GAS_AMBIENT_TEMP) * GAS_BUOYANCY_G;
        accel.y += buoyancy;
    }

    // --- Symplectic Euler: velocity update ---
    float3 vel_new = make_float3(
        vel.x + dt * accel.x,
        vel.y + dt * accel.y,
        vel.z + dt * accel.z
    );

    // GAS drag: vel *= (1 - c_drag * dt)
    if (behavior == GAS) {
        float drag_factor = 1.0f - GAS_DRAG_COEFF * dt;
        drag_factor = fmaxf(drag_factor, 0.0f);
        vel_new.x *= drag_factor;
        vel_new.y *= drag_factor;
        vel_new.z *= drag_factor;
    }

    // --- Velocity magnitude clamp ---
    float vel_sq = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
    if (vel_sq > VELOCITY_LIMIT_SQ) {
        float scale = VELOCITY_LIMIT / sqrtf(vel_sq);
        vel_new.x *= scale;
        vel_new.y *= scale;
        vel_new.z *= scale;
    }

    // --- Position update ---
    float3 pos_new;
    if (behavior == FLUID) {
        // FLUID: use XSPH-corrected velocity for position update
        pos_new = make_float3(
            pos.x + dt * (vel_new.x + veleval_xsph.x - vel.x),
            pos.y + dt * (vel_new.y + veleval_xsph.y - vel.y),
            pos.z + dt * (vel_new.z + veleval_xsph.z - vel.z)
        );
    } else {
        pos_new = make_float3(
            pos.x + dt * vel_new.x,
            pos.y + dt * vel_new.y,
            pos.z + dt * vel_new.z
        );
    }

    // --- Impulse-style SDF boundary ---
    sdf_box_boundary(
        pos_new, vel_new,
        c_sim.world_min, c_sim.world_max,
        c_sim.restitution, c_sim.wall_friction
    );

    // --- Compute color ---
    float4 color = compute_color(mat_id, temp, hlth);

    // --- Write to UNSORTED arrays ---
    position_out[orig_idx] = make_float4(pos_new.x, pos_new.y, pos_new.z, 1.0f);
    velocity_out[orig_idx] = make_float4(vel_new.x, vel_new.y, vel_new.z, 0.0f);
    color_out[orig_idx] = color;
}
