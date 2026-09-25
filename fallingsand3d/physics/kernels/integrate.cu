/*
 * integrate.cu -- K_Integrate kernel: symplectic Euler integration,
 * impulse-style SDF box boundaries, GAS buoyancy/drag, velocity clamping,
 * color computation, and writeback to UNSORTED arrays via sort_indexes.
 *
 * Per-particle computation:
 *   1. Compute acceleration: accel = sph_force + gravity + gas_buoyancy
 *   2. Symplectic Euler: vel_new = vel + dt * accel
 *   3. GAS drag: vel_new *= (1 - c_drag * dt)
 *   4. Velocity clamp: |vel| <= velocity_limit
 *   5. Position update: pos_new = pos + dt * (vel_new + eps*xsph_sum) for FLUID,
 *      where eps*xsph_sum = veleval - vel from step2;
 *      pos_new = pos + dt * vel_new for others
 *   6. Impulse SDF boundary: 6 planes of box, project out, reflect normal vel
 *      with restitution, apply Coulomb friction to tangential vel
 *   7. Temperature integration: T += dTdt*dt - cool_rate*(T-T_ambient)*dt, clamp [0,5000]
 *   8. Compute color from material base color, temperature tint, health fade
 *   9. Write pos, vel, color, temperature to UNSORTED arrays via sort_indexes[i]
 *
 * Skips STATIC particles (behavior_class == 3): early return, position unchanged.
 *
 * Constant memory used:
 *   c_sim       -- SimParams from common.cuh (gravity, dt, restitution, etc.)
 *   c_materials -- MaterialProps[32] from common.cuh (for color lookup)
 */

#include "sph_shared.cuh"

/* ======================================================================
 * Constants (integrate-specific, not in sph_shared.cuh)
 * ====================================================================== */

#define ACCEL_MAX_FLUID    200.0f
#define ACCEL_MAX_GRANULAR 200.0f

/* Micropolar SPH coupling parameter */
#define MICROPOLAR_NU_T    0.1f

/* Anti-creep thresholds for GRANULAR particles */
#define GRANULAR_V_THRESHOLD     0.01f
#define GRANULAR_V_THRESHOLD_SQ  (GRANULAR_V_THRESHOLD * GRANULAR_V_THRESHOLD)
#define GRANULAR_GAMMA_MIN       0.05f
#define GRANULAR_RHO_FACTOR      0.95f
#define GRANULAR_ACCEL_REST      5.0f   // equilibrium check: |accel| must be < this for anti-creep/sleep
#define GRANULAR_ACCEL_REST_SQ   (GRANULAR_ACCEL_REST * GRANULAR_ACCEL_REST)

/* Sleep system: integrate-specific addition */
#define GAMMA_SLEEP      0.01f

/* Rigid body array for the fluid push-out (bd-r5-review-n28.123).
 * RigidBodyManager.upload() writes the d_rigid_bodies address here next to
 * c_num_rigid_bodies. A constant keeps K_Integrate's argument list stable,
 * and the CUDA graph reads it at launch time, not at capture time. */
__constant__ const RigidBody* c_rigid_bodies;

/* ======================================================================
 * K_Integrate kernel
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_Integrate(
    uint            numParticles,
    // --- Sorted inputs (read) ---
    const float4*   __restrict__ sorted_position,      // sorted positions
    const float4*   __restrict__ sorted_velocity,      // sorted velocities
    const float4*   __restrict__ sorted_veleval,       // sorted veleval (vel + eps*xsph_sum for FLUID)
    const float4*   __restrict__ sorted_sph_force,     // sorted SPH force from Step2
    const float*    __restrict__ sorted_mass,           // sorted per-particle mass
    const uint*     __restrict__ sorted_packed_info,    // sorted packed_info
    const float*    __restrict__ sorted_temperature,    // sorted temperature
    const float*    __restrict__ sorted_health,         // sorted health
    const float*    __restrict__ sorted_lifetime,       // sorted remaining lifetime (seconds)
    const float*    __restrict__ sorted_density,        // sorted density (from Step1)
    const float*    __restrict__ sorted_shear_rate,     // sorted shear rate (from Step1)
    const float*    __restrict__ sorted_dTdt,           // sorted dTdt (heat diffusion from Step1)
    const unsigned char* __restrict__ sorted_sleep_counter, // sorted sleep counter (uint8)
    const float4*   __restrict__ sorted_dye_rate,       // sorted dye diffusion rate from Step1
    const float4*   __restrict__ sorted_particle_dye,   // sorted current particle dye color
    const float4*   __restrict__ sorted_vorticity,      // sorted vorticity (curl_v) from Step1
    const float4*   __restrict__ sorted_angular_velocity, // sorted micropolar angular velocity
    const uint*     __restrict__ sort_indexes,          // sort_indexes[sorted_i] = original_i
    // --- Grid arrays for STATIC boundary repulsion ---
    const uint*     __restrict__ cell_start,            // grid cell start indices
    const uint*     __restrict__ cell_end,              // grid cell end indices
    // --- Unsorted outputs (write via sort_indexes) ---
    float4*         __restrict__ position_out,          // unsorted position
    float4*         __restrict__ velocity_out,          // unsorted velocity
    float4*         __restrict__ color_out,             // unsorted color
    uint*           __restrict__ packed_info_out,       // unsorted packed_info (sleep flag updates)
    unsigned char*  __restrict__ sleep_counter_out,     // unsorted sleep counter
    float*          __restrict__ temperature_out,       // unsorted temperature (updated)
    float4*         __restrict__ particle_dye_out,       // unsorted particle dye (updated)
    float4*         __restrict__ angular_velocity_out,   // unsorted angular velocity (micropolar update)
    float*          __restrict__ health_out,             // unsorted health (reaction state carry-back)
    float*          __restrict__ lifetime_out,           // unsorted lifetime (reaction state carry-back)
    float*          __restrict__ mass_out,               // unsorted mass (spawn/reaction state carry-back)
    uint*           __restrict__ max_displacement_out,   // [1] atomicMax of displacement^2 (float-as-uint)
    const float4*   __restrict__ displacement_anchor  // [N] sorted position at the last full sort, or NULL
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    // --- Read packed_info ---
    uint pi = sorted_packed_info[i];
    int behavior = GET_BEHAVIOR(pi);
    uint mat_id = GET_MATERIAL_ID(pi);

    // Unsorted index for writeback
    uint orig_idx = sort_indexes[i];

    // --- Carry reaction/spawn state (health, lifetime, mass) back to unsorted.
    // These are mutated by K_Reactions / K_SpawnGas in sorted space but never by
    // K_Integrate, and every substep re-gathers them from the unsorted arrays.
    // Without this scatter, cumulative reaction progress (fire lifetime decay,
    // acid health loss, gas-spawn mass transfer) is wiped each substep. (CRIT-3)
    // Done unconditionally here so it covers the STATIC / sleeping early-returns. ---
    if (health_out)   health_out[orig_idx]   = sorted_health[i];
    if (lifetime_out) lifetime_out[orig_idx] = sorted_lifetime[i];
    if (mass_out)     mass_out[orig_idx]     = sorted_mass[i];

    // --- Skip STATIC particles ---
    if (behavior == STATIC) {
        // Write position unchanged, zero velocity, compute color.
        // MAT_RIGID boundary particles keep the body surface velocity that
        // K_UpdateBoundaryParticles wrote, so the unsorted array matches the
        // body for later readers such as wake (bd-r5-review-n28.87).
        float4 pos4 = sorted_position[i];
        position_out[orig_idx] = pos4;
        velocity_out[orig_idx] = (mat_id == MAT_RIGID)
            ? sorted_velocity[i]
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float temp = sorted_temperature[i];
        // STATIC particles still conduct heat
        float dTdt = sorted_dTdt[i];
        temp += dTdt * c_sim.dt;
        temp -= COOL_RATE * (temp - T_AMBIENT) * c_sim.dt;
        temp = fmaxf(T_MIN, fminf(temp, T_MAX));
        float hlth = sorted_health[i];
        color_out[orig_idx] = compute_color(mat_id, temp, hlth, STATIC);
        packed_info_out[orig_idx] = pi;
        sleep_counter_out[orig_idx] = sorted_sleep_counter[i];
        temperature_out[orig_idx] = temp;
        particle_dye_out[orig_idx] = sorted_particle_dye[i];
        angular_velocity_out[orig_idx] = sorted_angular_velocity[i];
        return;
    }

    // --- Read sleep counter ---
    unsigned char sc = sorted_sleep_counter[i];
    bool was_sleeping = IS_SLEEPING(pi) != 0;

    // --- Read particle data ---
    float4 pos4 = sorted_position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);

    float4 vel4 = sorted_velocity[i];
    float3 vel = make_float3(vel4.x, vel4.y, vel4.z);

    float temp = sorted_temperature[i];
    float hlth = sorted_health[i];

    // --- Sleeping particle: check wake condition with hysteresis ---
    if (was_sleeping) {
        float vel_sq_wake = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
        // Geometry wake test (bd-r5-review-n28.119): a kinematic SDF object
        // moving into a sleeping pile, or a world resize that leaves the
        // particle outside the box, must wake it so the push-out and the
        // velocity impulse below apply this substep. The test is strict
        // penetration (dist < 0, pos outside the unshrunk box), not the
        // BOUNDARY_MARGIN band: a particle resting on a surface sits at
        // dist ~= BOUNDARY_MARGIN after its last push-out, and waking it on
        // rounding noise would stop sand on static objects from sleeping.
        // Costs nothing when c_num_sdf_objects == 0.
        bool geom_wake =
            pos.x < c_sim.world_min.x || pos.x > c_sim.world_max.x ||
            pos.y < c_sim.world_min.y || pos.y > c_sim.world_max.y ||
            pos.z < c_sim.world_min.z || pos.z > c_sim.world_max.z;
        for (int s = 0; s < c_num_sdf_objects && !geom_wake; s++) {
            if (eval_sdf(pos, c_sdf_objects[s]) < 0.0f) geom_wake = true;
        }
        // Same test for rigid bodies at their end-of-substep pose.
        if (!geom_wake && c_num_rigid_bodies > 0 && c_rigid_bodies != 0) {
            geom_wake = rigid_body_penetrates(pos, c_rigid_bodies, c_num_rigid_bodies, c_sim.dt);
        }
        if (vel_sq_wake <= V_WAKE_SQ && !geom_wake) {
            // Still sleeping: write position/velocity unchanged, keep flag and saturate counter
            // Sleeping particles still conduct heat
            float dTdt_sleep = sorted_dTdt[i];
            temp += dTdt_sleep * c_sim.dt;
            temp -= COOL_RATE * (temp - T_AMBIENT) * c_sim.dt;
            temp = fmaxf(T_MIN, fminf(temp, T_MAX));
            position_out[orig_idx] = pos4;
            velocity_out[orig_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            color_out[orig_idx] = compute_color(mat_id, temp, hlth, behavior);
            packed_info_out[orig_idx] = pi;  // keep SLEEPING flag set
            sleep_counter_out[orig_idx] = (sc < 255) ? sc : (unsigned char)255;
            temperature_out[orig_idx] = temp;
            particle_dye_out[orig_idx] = sorted_particle_dye[i];
            angular_velocity_out[orig_idx] = sorted_angular_velocity[i];
            return;
        }
        // Wake up: clear sleeping flag, set just_woke flag, reset counter
        pi = CLEAR_SLEEPING(pi);
        pi = SET_JUST_WOKE(pi);
        was_sleeping = false;
        sc = 0;
    }

    float4 veleval4 = sorted_veleval[i];
    float3 veleval_xsph = make_float3(veleval4.x, veleval4.y, veleval4.z);

    float4 force4 = sorted_sph_force[i];
    float3 sph_force = make_float3(force4.x, force4.y, force4.z);

    float mass_i = sorted_mass[i];

    float dt = c_sim.dt;

    // --- Compute acceleration ---
    // step2.cu already outputs acceleration (force/mass), so use directly.
    float3 accel = make_float3(
        sph_force.x + c_sim.gravity.x,
        sph_force.y + c_sim.gravity.y,
        sph_force.z + c_sim.gravity.z
    );

    // GAS buoyancy (ideal gas, live gravity vector): -g * (T/T_amb + extra)
    if (behavior == GAS) {
        float3 b = gas_buoyancy_accel(temp, mat_id);
        accel.x += b.x; accel.y += b.y; accel.z += b.z;
    }

    // FLUID thermal convection: Boussinesq buoyancy
    // rho_eff = rho_0 * (1 - beta*(T-T0)), lighter when hot -> moves against g
    if (behavior == FLUID) {
        float beta = c_materials[mat_id].thermal_expansion;
        if (beta > 0.0f) {
            float3 b = boussinesq_accel(temp, beta);
            accel.x += b.x; accel.y += b.y; accel.z += b.z;
        }
    }

    // --- Acceleration clamp (safety net against numerical blowups) ---
    // GRANULAR uses higher limit because force_scale=1.0 gives full-strength SPH forces
    float accel_limit = (behavior == GRANULAR) ? ACCEL_MAX_GRANULAR : ACCEL_MAX_FLUID;
    float accel_limit_sq = accel_limit * accel_limit;
    float accel_sq = accel.x * accel.x + accel.y * accel.y + accel.z * accel.z;
    if (accel_sq > accel_limit_sq) {
        float scale = accel_limit / sqrtf(accel_sq);
        accel.x *= scale;
        accel.y *= scale;
        accel.z *= scale;
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

    // --- Spawn velocity damping (ramps to 0 after first ~30 substeps) ---
    if (c_sim.velocity_damping > 0.0f) {
        float damp = 1.0f - c_sim.velocity_damping;
        vel_new.x *= damp;
        vel_new.y *= damp;
        vel_new.z *= damp;
    }

    // --- GRANULAR anti-creep: zero velocity if nearly at rest AND in equilibrium ---
    // Equilibrium check: only zero velocity when net SPH force (excluding gravity)
    // is small. We compare |sph_force|^2 rather than |accel|^2 because accel
    // includes gravity (~9.8 m/s^2), which always exceeds GRANULAR_ACCEL_REST=5
    // for unsupported particles — making the old check never fire.
    // sph_force is the net SPH acceleration; it approaches 0 when pressure and
    // viscosity forces balance (particle fully supported by its neighbors).
    if (behavior == GRANULAR) {
        float vel_sq_ac = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
        if (vel_sq_ac < GRANULAR_V_THRESHOLD_SQ) {
            float rho_i = sorted_density[i];
            float rho0_i = c_materials[mat_id].rest_density;
            float sr_i = sorted_shear_rate[i];
            float sph_sq = sph_force.x * sph_force.x + sph_force.y * sph_force.y + sph_force.z * sph_force.z;
            if (rho_i > GRANULAR_RHO_FACTOR * rho0_i && sr_i < GRANULAR_GAMMA_MIN
                && sph_sq < GRANULAR_ACCEL_REST_SQ) {
                vel_new.x = 0.0f;
                vel_new.y = 0.0f;
                vel_new.z = 0.0f;
            }
        }
    }

    // --- Position update (symplectic Euler) ---
    // All behaviors advect with vel_new. FLUID adds the XSPH correction on top:
    // step2 writes veleval = vel + eps * xsph_sum, so (veleval - vel) is exactly
    // the XSPH delta. Using vel_new (not veleval) means this substep's forces,
    // clamps, damping and drag reach the position in the same substep.
    float3 advect_vel = vel_new;
    if (behavior == FLUID) {
        advect_vel.x += veleval_xsph.x - vel.x;
        advect_vel.y += veleval_xsph.y - vel.y;
        advect_vel.z += veleval_xsph.z - vel.z;
    }
    float3 pos_new = make_float3(
        pos.x + dt * advect_vel.x,
        pos.y + dt * advect_vel.y,
        pos.z + dt * advect_vel.z
    );

    // --- STATIC particle boundary repulsion ---
    if (cell_start != 0) {
        static_particle_boundary(
            pos_new, vel_new,
            cell_start, cell_end,
            sorted_packed_info, sorted_position,
            i, c_sim.restitution
        );
    }

    // --- Impulse-style SDF boundary ---
    // FLUID: zero wall friction to prevent sticking to domain walls
    float friction = (behavior == FLUID) ? 0.0f : c_sim.wall_friction;
    sdf_box_boundary(
        pos_new, vel_new,
        c_sim.world_min, c_sim.world_max,
        c_sim.restitution, friction
    );

    // --- SDF object collision ---
    // Loop over all placed SDF objects (0 when none defined — no cost)
    for (int s = 0; s < c_num_sdf_objects; s++) {
        SDFObject obj = c_sdf_objects[s];
        float dist = eval_sdf(pos_new, obj);
        if (dist < BOUNDARY_MARGIN) {
            // Push out along analytical normal
            float3 n = sdf_normal(pos_new, obj);
            float push = BOUNDARY_MARGIN - dist;
            pos_new.x += push * n.x;
            pos_new.y += push * n.y;
            pos_new.z += push * n.z;

            // Compute object velocity at contact point (linear + angular)
            float3 obj_vel = make_float3(obj.velocity.x, obj.velocity.y, obj.velocity.z);
            float ang_speed = obj.velocity.w;
            if (fabsf(ang_speed) > 1e-8f) {
                float3 axis = make_float3(obj.angular_axis.x, obj.angular_axis.y, obj.angular_axis.z);
                float3 center = make_float3(obj.pos_and_type.x, obj.pos_and_type.y, obj.pos_and_type.z);
                float3 r = make_float3(pos_new.x - center.x, pos_new.y - center.y, pos_new.z - center.z);
                // omega = axis * ang_speed
                float3 omega = make_float3(axis.x * ang_speed, axis.y * ang_speed, axis.z * ang_speed);
                // v_rot = cross(omega, r)
                obj_vel.x += omega.y * r.z - omega.z * r.y;
                obj_vel.y += omega.z * r.x - omega.x * r.z;
                obj_vel.z += omega.x * r.y - omega.y * r.x;
            }

            // Relative velocity
            float3 v_rel = make_float3(vel_new.x - obj_vel.x, vel_new.y - obj_vel.y, vel_new.z - obj_vel.z);
            float v_dot_n = v_rel.x * n.x + v_rel.y * n.y + v_rel.z * n.z;

            // Only apply if moving into surface
            if (v_dot_n < 0.0f) {
                float obj_restitution = obj.size_and_r.w;
                float obj_friction = obj.angular_axis.w;

                // Normal impulse: reflect with restitution
                vel_new.x -= (1.0f + obj_restitution) * v_dot_n * n.x;
                vel_new.y -= (1.0f + obj_restitution) * v_dot_n * n.y;
                vel_new.z -= (1.0f + obj_restitution) * v_dot_n * n.z;

                // Coulomb friction on tangential component
                float3 v_tan = make_float3(
                    v_rel.x - v_dot_n * n.x,
                    v_rel.y - v_dot_n * n.y,
                    v_rel.z - v_dot_n * n.z
                );
                float v_tan_len = sqrtf(v_tan.x * v_tan.x + v_tan.y * v_tan.y + v_tan.z * v_tan.z);
                if (v_tan_len > 1e-8f) {
                    float red = fminf(obj_friction * fabsf(v_dot_n) / v_tan_len, 1.0f);
                    vel_new.x -= red * v_tan.x;
                    vel_new.y -= red * v_tan.y;
                    vel_new.z -= red * v_tan.z;
                }
            }
        }
    }

    // --- Rigid body collision (bd-r5-review-n28.123) ---
    // Push out of each body's box or sphere SDF at its end-of-substep pose,
    // so a fast body cannot overtake fluid and trap it inside.
    if (c_num_rigid_bodies > 0 && c_rigid_bodies != 0) {
        rigid_body_boundary(pos_new, vel_new, c_rigid_bodies, c_num_rigid_bodies, dt);
    }

    // --- Sleep counter update ---
    // Only GRANULAR particles can accumulate the sleep counter.
    // FLUID and GAS particles must remain awake to respond to pressure
    // gradients; if they sleep, the whole fluid pool freezes (no forces are
    // computed for sleeping particles, so they can never be woken by
    // pressure differentials, causing permanent deadlock).
    // GRANULAR requires: low velocity, low shear rate, AND small SPH force
    // (|sph_force|^2 < GRANULAR_ACCEL_REST_SQ, excluding gravity).
    float vel_sq_sleep = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
    float sr_sleep = sorted_shear_rate[i];

    if (behavior == GRANULAR) {
        if (vel_sq_sleep < V_SLEEP_SQ && sr_sleep < GAMMA_SLEEP) {
            // Use |sph_force|^2, not |accel|^2 (accel includes gravity which always
            // exceeds GRANULAR_ACCEL_REST for unsupported particles).
            float sph_sq = sph_force.x * sph_force.x + sph_force.y * sph_force.y + sph_force.z * sph_force.z;
            if (sph_sq < GRANULAR_ACCEL_REST_SQ) {
                if (sc < 255) sc++;
            } else {
                sc = 0;
            }
        } else {
            sc = 0;
        }
    } else {
        // FLUID/GAS: never accumulate sleep counter (reset if it was set externally)
        sc = 0;
    }

    // Set SLEEPING flag if counter reached threshold
    if (sc >= SLEEP_THRESHOLD) {
        pi = SET_SLEEPING(pi);
    }

    // --- Temperature integration ---
    float dTdt = sorted_dTdt[i];
    temp += dTdt * dt;
    temp -= COOL_RATE * (temp - T_AMBIENT) * dt;
    temp = fmaxf(T_MIN, fminf(temp, T_MAX));

    // --- Compute color ---
    // color.w encodes behavior class for SSFR shader filtering
    float4 color;
    if (behavior == FLUID) {
        float rho_i = sorted_density[i];
        color = compute_fluid_color(mat_id, temp, hlth, pos_new.y, vel_sq_sleep, rho_i);
    } else {
        color = compute_color(mat_id, temp, hlth, behavior);
    }

    // --- Dye update: dye += dye_rate * dt ---
    float4 dye = sorted_particle_dye[i];
    float4 drate = sorted_dye_rate[i];
    float dt_dye = c_sim.dt;
    dye.x = fmaxf(0.0f, fminf(1.0f, dye.x + drate.x * dt_dye));
    dye.y = fmaxf(0.0f, fminf(1.0f, dye.y + drate.y * dt_dye));
    dye.z = fmaxf(0.0f, fminf(1.0f, dye.z + drate.z * dt_dye));

    // --- Micropolar angular velocity update ---
    // Relaxation: omega_new = omega + dt * nu_t * (0.5 * curl_v - omega)
    // curl_v (vorticity) computed in step1, stored in sorted_vorticity.xyz
    // Only for FLUID particles (GRANULAR/GAS don't benefit from micropolar)
    float4 ang_vel = sorted_angular_velocity[i];
    if (behavior == FLUID) {
        float4 vort = sorted_vorticity[i];
        float nu_t = MICROPOLAR_NU_T;
        ang_vel.x += dt * nu_t * (0.5f * vort.x - ang_vel.x);
        ang_vel.y += dt * nu_t * (0.5f * vort.y - ang_vel.y);
        ang_vel.z += dt * nu_t * (0.5f * vort.z - ang_vel.z);
    }

    // Track drift since the last full sort for grid reuse (bd-r5-review-n28.54).
    // The anchor is the sorted position written by K_ScatterReorder at the last
    // full sort; the permutation is frozen between full sorts, so slot i is the
    // same particle. atomicMax over all substeps since that sort then holds the
    // largest drift any particle has made from the cell it was binned in.
    // NULL anchor: fall back to this substep's start position.
    // Positive IEEE754 floats order like their uint bits, so atomicMax on
    // __float_as_uint gives the max float value.
    if (max_displacement_out) {
        float3 ref = pos;
        if (displacement_anchor) {
            const float4 a = __ldg(&displacement_anchor[i]);
            ref = make_float3(a.x, a.y, a.z);
        }
        float dx_d = pos_new.x - ref.x;
        float dy_d = pos_new.y - ref.y;
        float dz_d = pos_new.z - ref.z;
        float disp_sq = dx_d * dx_d + dy_d * dy_d + dz_d * dz_d;
        atomicMax(max_displacement_out, __float_as_uint(disp_sq));
    }

    // --- Write to UNSORTED arrays ---
    // .w carries this substep's density to the next substep: the reorder gathers it
    // into sorted_position.w, which simulation.py copies into sorted_density_prev as
    // step1's density_in.  Writing 1.0 here made every awake neighbour read rho_j = 1.
    position_out[orig_idx] = make_float4(pos_new.x, pos_new.y, pos_new.z, sorted_density[i]);
    velocity_out[orig_idx] = make_float4(vel_new.x, vel_new.y, vel_new.z, 0.0f);
    color_out[orig_idx] = color;
    packed_info_out[orig_idx] = pi;
    sleep_counter_out[orig_idx] = sc;
    temperature_out[orig_idx] = temp;
    particle_dye_out[orig_idx] = dye;
    angular_velocity_out[orig_idx] = ang_vel;
}

/* ======================================================================
 * rigid_support_distance -- how far a body reaches from its COM along n.
 *
 * Box:    s(n) = sum_k he_k * |dot(n, R e_k)|  (exact support of an OBB)
 * Sphere: s(n) = radius (half_extents.x)
 * n must be unit length. Used by the world-wall clamp and the body-vs-SDF
 * push-out so a body rests with its real surface on the contact, not a
 * bounding sphere or a fixed margin. (bd-r5-review-n28.98)
 * ====================================================================== */
__device__ inline float rigid_support_distance(const RigidBody& body, float3 n) {
    int shape = __float_as_int(body.half_extents.w);
    if (shape == SDF_SPHERE) return body.half_extents.x;
    float4 q = body.rotation;
    float3 ax = quat_rotate(q, make_float3(1.0f, 0.0f, 0.0f));
    float3 ay = quat_rotate(q, make_float3(0.0f, 1.0f, 0.0f));
    float3 az = quat_rotate(q, make_float3(0.0f, 0.0f, 1.0f));
    return body.half_extents.x * fabsf(n.x * ax.x + n.y * ax.y + n.z * ax.z)
         + body.half_extents.y * fabsf(n.x * ay.x + n.y * ay.y + n.z * ay.z)
         + body.half_extents.z * fabsf(n.x * az.x + n.y * az.y + n.z * az.z);
}

/* ======================================================================
 * K_IntegrateRigidBodies -- US-017: Rigid body integration kernel.
 *
 * One thread per rigid body. Reads accumulated forces/torques, integrates
 * linear and angular velocity via symplectic Euler, updates quaternion.
 *
 * CRITICAL: Uses world-frame inertia tensor. Body-frame diagonal
 * I_body_inv is rotated into world frame each step:
 *   I_world_inv = R * diag(I_body_inv) * R^T
 *
 * Zeroes force/torque accumulators after reading.
 *
 * Kinematic bodies skip forces, damping and clamps but still move by
 * their stored lin_vel / ang_vel.
 * ====================================================================== */

extern "C" __global__
void K_IntegrateRigidBodies(
    RigidBody*  __restrict__ d_rigid_bodies,
    float*      __restrict__ d_rigid_forces,   // [MAX_RIGID_BODIES * 4]
    float*      __restrict__ d_rigid_torques,  // [MAX_RIGID_BODIES * 4]
    int         num_bodies,
    float       dt,
    float       gravity_x,
    float       gravity_y,
    float       gravity_z
) {
    float3 gravity = make_float3(gravity_x, gravity_y, gravity_z);
    int bid = threadIdx.x;
    if (bid >= num_bodies) return;

    RigidBody body = d_rigid_bodies[bid];

    int is_kinematic = __float_as_int(body.inertia_inv.w);
    float inv_mass = body.position.w;

    // Kinematic (or infinite-mass) bodies: no forces, torque, damping,
    // clamps or wall bounce, but the pose still follows the prescribed
    // lin_vel / ang_vel. The boundary particles and the fluid coupling use
    // lin_vel as the surface velocity, so the body must actually move at
    // that velocity. The force/torque accumulators are left as they are so
    // the fluid reaction on the body stays readable; the substep prologue
    // clears them. (bd-r5-review-n28.105)
    if (is_kinematic || inv_mass <= 0.0f) {
        float3 pos_k = make_float3(body.position.x, body.position.y, body.position.z);
        float4 q_k = body.rotation;
        rigid_integrate_pose(pos_k, q_k,
            make_float3(body.lin_vel.x, body.lin_vel.y, body.lin_vel.z),
            make_float3(body.ang_vel.x, body.ang_vel.y, body.ang_vel.z), dt);
        d_rigid_bodies[bid].position = make_float4(pos_k.x, pos_k.y, pos_k.z, inv_mass);
        d_rigid_bodies[bid].rotation = q_k;
        return;
    }

    float mass = 1.0f / inv_mass;

    // --- Read accumulated force and torque ---
    float3 F = make_float3(
        d_rigid_forces[bid * 4 + 0],
        d_rigid_forces[bid * 4 + 1],
        d_rigid_forces[bid * 4 + 2]
    );
    float3 tau = make_float3(
        d_rigid_torques[bid * 4 + 0],
        d_rigid_torques[bid * 4 + 1],
        d_rigid_torques[bid * 4 + 2]
    );

    // --- Zero accumulators ---
    d_rigid_forces[bid * 4 + 0] = 0.0f;
    d_rigid_forces[bid * 4 + 1] = 0.0f;
    d_rigid_forces[bid * 4 + 2] = 0.0f;
    d_rigid_forces[bid * 4 + 3] = 0.0f;
    d_rigid_torques[bid * 4 + 0] = 0.0f;
    d_rigid_torques[bid * 4 + 1] = 0.0f;
    d_rigid_torques[bid * 4 + 2] = 0.0f;
    d_rigid_torques[bid * 4 + 3] = 0.0f;

    // --- Force clamp: |F| < 1000 * mass ---
    float F_max = 1000.0f * mass;
    float F_sq = F.x * F.x + F.y * F.y + F.z * F.z;
    if (F_sq > F_max * F_max) {
        float s = F_max / sqrtf(F_sq);
        F.x *= s; F.y *= s; F.z *= s;
    }

    // --- Linear integration: v += (F * inv_mass + gravity) * dt ---
    float3 lin_vel = make_float3(body.lin_vel.x, body.lin_vel.y, body.lin_vel.z);
    lin_vel.x += (F.x * inv_mass + gravity.x) * dt;
    lin_vel.y += (F.y * inv_mass + gravity.y) * dt;
    lin_vel.z += (F.z * inv_mass + gravity.z) * dt;

    // Linear velocity damping
    float lin_damp = 1.0f - 0.01f * dt;
    lin_vel.x *= lin_damp;
    lin_vel.y *= lin_damp;
    lin_vel.z *= lin_damp;

    // --- Angular integration in angular momentum form ---
    // Euler's equation I*domega/dt + omega x (I*omega) = tau is integrated
    // through the world-frame angular momentum L = I_world * omega:
    //   L      = R * diag(I_b) * (R^T * omega)      (old rotation)
    //   L     += tau * dt; L *= ang_damp
    //   omega  = R * diag(Iinv) * (R^T * L)         (old rotation, drives the pose)
    //   pose update with omega
    //   omega' = R' * diag(Iinv) * (R'^T * L)       (new rotation, stored)
    // With no torque L is conserved exactly, and the gyroscopic term
    // -I_world_inv * (omega x I_world * omega) comes out of re-deriving
    // omega from L at the new rotation. A non-cubic body therefore shows
    // the intermediate-axis flip. For a cube I_world is isotropic and this
    // reduces to omega += Iinv * tau * dt. (bd-r5-review-n28.114)
    float4 q = body.rotation;
    float3 Iinv = make_float3(body.inertia_inv.x, body.inertia_inv.y, body.inertia_inv.z);
    float3 Ib = make_float3(                                 // guard zero Iinv
        Iinv.x > 0.0f ? 1.0f / Iinv.x : 0.0f,
        Iinv.y > 0.0f ? 1.0f / Iinv.y : 0.0f,
        Iinv.z > 0.0f ? 1.0f / Iinv.z : 0.0f
    );

    float3 ang_vel = make_float3(body.ang_vel.x, body.ang_vel.y, body.ang_vel.z);
    float3 w_b = quat_rotate_inv(q, ang_vel);                // R^T * omega
    float3 L = quat_rotate(q, make_float3(Ib.x * w_b.x, Ib.y * w_b.y, Ib.z * w_b.z));

    // Angular velocity damping (applied to L, same scalar factor)
    float ang_damp = 1.0f - 0.05f * dt;
    L.x = (L.x + tau.x * dt) * ang_damp;
    L.y = (L.y + tau.y * dt) * ang_damp;
    L.z = (L.z + tau.z * dt) * ang_damp;

    float3 L_b = quat_rotate_inv(q, L);
    ang_vel = quat_rotate(q, make_float3(Iinv.x * L_b.x, Iinv.y * L_b.y, Iinv.z * L_b.z));

    // Angular velocity clamp: |omega| < 20 rad/s (scales L with it)
    float omega_sq = ang_vel.x * ang_vel.x + ang_vel.y * ang_vel.y + ang_vel.z * ang_vel.z;
    if (omega_sq > 400.0f) {  // 20^2
        float s = 20.0f / sqrtf(omega_sq);
        ang_vel.x *= s; ang_vel.y *= s; ang_vel.z *= s;
        L.x *= s; L.y *= s; L.z *= s;
    }

    // --- Pose update: pos += v * dt, q += 0.5 * dt * omega_q * q ---
    float3 pos = make_float3(body.position.x, body.position.y, body.position.z);
    rigid_integrate_pose(pos, q, lin_vel, ang_vel, dt);

    // Re-derive omega from the conserved L at the new rotation.
    L_b = quat_rotate_inv(q, L);
    ang_vel = quat_rotate(q, make_float3(Iinv.x * L_b.x, Iinv.y * L_b.y, Iinv.z * L_b.z));

    // --- World boundary clamp for rigid body COM ---
    float3 wmin = c_sim.world_min;
    float3 wmax = c_sim.world_max;
    // Per-axis margin is the body's support distance along that axis (with
    // the updated rotation) plus BOUNDARY_MARGIN, so the body surface, not
    // the COM, stops at the wall. (bd-r5-review-n28.98)
    body.rotation = q;
    float mx = rigid_support_distance(body, make_float3(1.0f, 0.0f, 0.0f)) + BOUNDARY_MARGIN;
    float my = rigid_support_distance(body, make_float3(0.0f, 1.0f, 0.0f)) + BOUNDARY_MARGIN;
    float mz = rigid_support_distance(body, make_float3(0.0f, 0.0f, 1.0f)) + BOUNDARY_MARGIN;
    if (pos.x < wmin.x + mx) { pos.x = wmin.x + mx; if (lin_vel.x < 0) lin_vel.x *= -0.3f; }
    if (pos.x > wmax.x - mx) { pos.x = wmax.x - mx; if (lin_vel.x > 0) lin_vel.x *= -0.3f; }
    if (pos.y < wmin.y + my) { pos.y = wmin.y + my; if (lin_vel.y < 0) lin_vel.y *= -0.3f; }
    if (pos.y > wmax.y - my) { pos.y = wmax.y - my; if (lin_vel.y > 0) lin_vel.y *= -0.3f; }
    if (pos.z < wmin.z + mz) { pos.z = wmin.z + mz; if (lin_vel.z < 0) lin_vel.z *= -0.3f; }
    if (pos.z > wmax.z - mz) { pos.z = wmax.z - mz; if (lin_vel.z > 0) lin_vel.z *= -0.3f; }

    // --- Write back ---
    body.position = make_float4(pos.x, pos.y, pos.z, inv_mass);
    body.rotation = q;
    body.lin_vel = make_float4(lin_vel.x, lin_vel.y, lin_vel.z, body.lin_vel.w);  // preserve restitution
    body.ang_vel = make_float4(ang_vel.x, ang_vel.y, ang_vel.z, body.ang_vel.w);  // preserve friction
    d_rigid_bodies[bid] = body;
}

/* ======================================================================
 * K_RigidBodyCollisions -- US-020: Simple push-apart collisions.
 *
 * One thread per dynamic body. Checks against:
 *   1. All SDF objects (c_sdf_objects) -- body-vs-static/kinematic
 *   2. All other dynamic bodies (bounding sphere) -- body-vs-body
 *
 * Body vs SDF: contact depth = rigid_support_distance(body, n) - sdf(COM),
 * with n the SDF normal at the COM. Body vs body still uses bounding
 * spheres, radius = length(half_extents).
 * Runs after K_IntegrateRigidBodies, before K_UpdateBoundaryParticles.
 * ====================================================================== */

extern "C" __global__
void K_RigidBodyCollisions(
    RigidBody*  __restrict__ d_rigid_bodies,
    int         num_bodies
) {
    int bid = threadIdx.x;
    if (bid >= num_bodies) return;

    RigidBody body = d_rigid_bodies[bid];

    // Skip kinematic bodies
    int is_kinematic = __float_as_int(body.inertia_inv.w);
    if (is_kinematic) return;

    float inv_mass = body.position.w;
    if (inv_mass <= 0.0f) return;

    float3 pos = make_float3(body.position.x, body.position.y, body.position.z);
    float3 vel = make_float3(body.lin_vel.x, body.lin_vel.y, body.lin_vel.z);
    float restitution = body.lin_vel.w;

    // Bounding sphere radius from half_extents
    float3 he = make_float3(body.half_extents.x, body.half_extents.y, body.half_extents.z);
    float b_radius = sqrtf(he.x * he.x + he.y * he.y + he.z * he.z);

    // --- 1. Body vs SDF objects ---
    for (int s = 0; s < c_num_sdf_objects; s++) {
        SDFObject obj = c_sdf_objects[s];
        float dist = eval_sdf(pos, obj);
        // Cheap reject: the body cannot touch if the COM is farther than
        // its bounding radius.
        if (dist >= b_radius) continue;

        // Contact depth uses the body's support distance along the SDF
        // normal, so a box rests face-down on a plane with no gap.
        float3 normal = sdf_normal(pos, obj);
        float reach = rigid_support_distance(body, normal);
        if (dist < reach) {
            float penetration = reach - dist;

            // Push out
            pos.x += normal.x * penetration;
            pos.y += normal.y * penetration;
            pos.z += normal.z * penetration;

            // Velocity reflection along normal
            float vn = vel.x * normal.x + vel.y * normal.y + vel.z * normal.z;
            if (vn < 0.0f) {
                float sdf_restitution = obj.size_and_r.w;
                float e = fminf(restitution, sdf_restitution);
                vel.x -= (1.0f + e) * vn * normal.x;
                vel.y -= (1.0f + e) * vn * normal.y;
                vel.z -= (1.0f + e) * vn * normal.z;

                // Tangential friction
                float mu = obj.angular_axis.w;  // friction stored in angular_axis.w
                float3 vt = make_float3(
                    vel.x - vn * normal.x,
                    vel.y - vn * normal.y,
                    vel.z - vn * normal.z
                );
                float vt_len = sqrtf(vt.x * vt.x + vt.y * vt.y + vt.z * vt.z);
                if (vt_len > 1e-6f) {
                    float friction_impulse = fminf(mu * fabsf(vn) * (1.0f + e), vt_len);
                    float scale = 1.0f - friction_impulse / vt_len;
                    vel.x = vn * normal.x + vt.x * scale;
                    vel.y = vn * normal.y + vt.y * scale;
                    vel.z = vn * normal.z + vt.z * scale;
                }
            }
        }
    }

    // --- 2. Body vs other dynamic bodies (bounding sphere) ---
    for (int other = 0; other < num_bodies; other++) {
        if (other == bid) continue;

        RigidBody ob = d_rigid_bodies[other];
        int ob_kinematic = __float_as_int(ob.inertia_inv.w);
        float ob_inv_mass = ob.position.w;

        float3 ob_pos = make_float3(ob.position.x, ob.position.y, ob.position.z);
        float3 ob_he = make_float3(ob.half_extents.x, ob.half_extents.y, ob.half_extents.z);
        float ob_radius = sqrtf(ob_he.x * ob_he.x + ob_he.y * ob_he.y + ob_he.z * ob_he.z);

        float3 sep = make_float3(pos.x - ob_pos.x, pos.y - ob_pos.y, pos.z - ob_pos.z);
        float dist_sq = sep.x * sep.x + sep.y * sep.y + sep.z * sep.z;
        float min_dist = b_radius + ob_radius;

        if (dist_sq < min_dist * min_dist && dist_sq > 1e-10f) {
            float dist = sqrtf(dist_sq);
            float3 normal = make_float3(sep.x / dist, sep.y / dist, sep.z / dist);
            float penetration = min_dist - dist;

            // Push apart weighted by inverse mass
            float total_inv = inv_mass + (ob_kinematic ? 0.0f : fmaxf(ob_inv_mass, 0.0f));
            if (total_inv > 1e-10f) {
                float my_share = inv_mass / total_inv;
                pos.x += normal.x * penetration * my_share;
                pos.y += normal.y * penetration * my_share;
                pos.z += normal.z * penetration * my_share;
            }

            // Velocity reflection along contact normal
            float3 ob_vel = make_float3(ob.lin_vel.x, ob.lin_vel.y, ob.lin_vel.z);
            float3 rel_vel = make_float3(vel.x - ob_vel.x, vel.y - ob_vel.y, vel.z - ob_vel.z);
            float rel_vn = rel_vel.x * normal.x + rel_vel.y * normal.y + rel_vel.z * normal.z;

            if (rel_vn < 0.0f) {
                // Symmetric body-body collision: each body (thread) independently
                // applies the impulse. Newton's 3rd law is preserved because the
                // two threads apply equal and opposite deltas (opposite normals).
                // No halving is needed: with both threads running, the full
                // standard impulse formula gives the correct energy-conserving result.
                // The body-SDF path uses the same formula since only the particle
                // receives the impulse; the SDF wall is immovable.
                float e = fminf(restitution, ob.lin_vel.w);
                float total_inv_coll = inv_mass + (ob_kinematic ? 0.0f : fmaxf(ob_inv_mass, 0.0f));
                if (total_inv_coll > 1e-10f) {
                    float impulse = -(1.0f + e) * rel_vn * (inv_mass / total_inv_coll);
                    vel.x += impulse * normal.x;
                    vel.y += impulse * normal.y;
                    vel.z += impulse * normal.z;
                }
            }
        }
    }

    // --- Write back ---
    body.position = make_float4(pos.x, pos.y, pos.z, inv_mass);
    body.lin_vel = make_float4(vel.x, vel.y, vel.z, restitution);
    d_rigid_bodies[bid] = body;
}

/* ======================================================================
 * K_UpdateBoundaryParticles -- US-018: State sync kernel.
 *
 * Transforms each boundary particle's local-space position by its
 * parent rigid body's current position + rotation quaternion.
 * Sets velocity to v_body + cross(omega, r_rotated) for correct
 * viscous coupling with fluid.
 *
 * Runs AFTER K_IntegrateRigidBodies, BEFORE hash/sort for next substep.
 *
 * Grid-reuse drift (bd-r5-review-n28.65): boundary particles are STATIC, so
 * K_Integrate and the PBF/DFSPH finalize kernels never record their motion.
 * This kernel records it instead. anchor[i] holds boundary particle i's
 * position at the last full sort. On a substep that runs a full sort the host
 * passes reset_anchor = 1 (and has already cleared max_displacement): the
 * kernel stores the new position as the anchor. The fluid records its drift
 * after it moves, but the body moves after the solver, so the kernel
 * extrapolates one substep, x_end = x + (x - x_old), and atomicMaxes
 * |x_end - anchor|^2 into max_displacement on every substep. A moving body
 * then forces a full sort just like moving fluid. NULL anchor or NULL
 * max_displacement: no tracking.
 * ====================================================================== */

extern "C" __global__
void K_UpdateBoundaryParticles(
    const float*    __restrict__ boundary_data,  // (N, 8): x_local, y_local, z_local, psi, body_id, nx, ny, nz
    const RigidBody* __restrict__ d_rigid_bodies,
    float4*         __restrict__ position_out,    // main unsorted position array
    float4*         __restrict__ velocity_out,    // main unsorted velocity array
    float*          __restrict__ mass_out,         // main unsorted mass array
    int             offset,                        // start index in main arrays
    int             N,                             // number of boundary particles
    float4*         __restrict__ anchor,           // [N] position at the last full sort, or NULL
    uint*           __restrict__ max_displacement_out, // [1] atomicMax of drift^2 (float-as-uint), or NULL
    int             reset_anchor                   // 1: this substep runs a full sort
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Read boundary particle data (8 floats per particle)
    int base = i * 8;
    float3 r_local = make_float3(boundary_data[base + 0], boundary_data[base + 1], boundary_data[base + 2]);
    float psi = boundary_data[base + 3];
    int bid = (int)(boundary_data[base + 4] + 0.5f);

    // Read body state
    RigidBody body = d_rigid_bodies[bid];
    float3 com = make_float3(body.position.x, body.position.y, body.position.z);
    float4 q = body.rotation;

    // Rotate local position to world frame
    float3 r_world = quat_rotate(q, r_local);
    float3 pos = make_float3(com.x + r_world.x, com.y + r_world.y, com.z + r_world.z);

    // Compute boundary velocity: v_body + cross(omega, r_world)
    float3 v_body = make_float3(body.lin_vel.x, body.lin_vel.y, body.lin_vel.z);
    float3 omega = make_float3(body.ang_vel.x, body.ang_vel.y, body.ang_vel.z);
    float3 vel = make_float3(
        v_body.x + (omega.y * r_world.z - omega.z * r_world.y),
        v_body.y + (omega.z * r_world.x - omega.x * r_world.z),
        v_body.z + (omega.x * r_world.y - omega.y * r_world.x)
    );

    // Write to main arrays
    int idx = offset + i;
    const float4 old_pos = position_out[idx];  // last substep's boundary position
    // .w = 2500: the same placeholder density K_Step1 writes for MAT_RIGID.
    position_out[idx] = make_float4(pos.x, pos.y, pos.z, 2500.0f);
    velocity_out[idx] = make_float4(vel.x, vel.y, vel.z, 0.0f);
    mass_out[idx] = psi;  // boundary volume as mass

    // Grid-reuse drift since the last full sort (see header).
    if (anchor && max_displacement_out) {
        float4 a;
        if (reset_anchor) {
            a = make_float4(pos.x, pos.y, pos.z, 0.0f);
            anchor[i] = a;
        } else {
            a = anchor[i];
        }
        // Position at the end of this substep, assuming the last step repeats.
        const float dx = 2.0f * pos.x - old_pos.x - a.x;
        const float dy = 2.0f * pos.y - old_pos.y - a.y;
        const float dz = 2.0f * pos.z - old_pos.z - a.z;
        const float disp_sq = dx * dx + dy * dy + dz * dz;
        atomicMax(max_displacement_out, __float_as_uint(disp_sq));
    }
}
