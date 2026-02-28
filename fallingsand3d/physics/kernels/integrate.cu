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
 *   5. Position update: pos_new = pos + dt * (vel_new + xsph) for FLUID,
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

/* ======================================================================
 * K_Integrate kernel
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
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
    uint*           __restrict__ max_displacement_out    // [1] atomicMax of displacement^2 (float-as-uint)
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
        if (vel_sq_wake <= V_WAKE_SQ) {
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

    // GAS buoyancy: f_buoy = beta * (T - 293) * (0, 9.81, 0)
    if (behavior == GAS) {
        float buoyancy = GAS_BUOYANCY_BETA * (temp - GAS_AMBIENT_TEMP) * GAS_BUOYANCY_G;
        accel.y += buoyancy;
    }

    // FLUID thermal convection: Boussinesq buoyancy
    // rho_eff = rho_0 * (1 - beta*(T-T0)), lighter when hot -> rises
    // a_buoy = +beta * (T - T_ambient) * g  (positive = upward for hot fluid)
    if (behavior == FLUID) {
        float beta = c_materials[mat_id].thermal_expansion;
        if (beta > 0.0f) {
            accel.y += beta * (temp - T_AMBIENT) * 9.81f;
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
    // Equilibrium check: only zero velocity when net acceleration is small.
    // Without this, gravity (9.8 m/s^2 * dt=0.001 = 0.0098 m/s per frame) is always
    // below the threshold (0.01), trapping unsupported particles in mid-air.
    if (behavior == GRANULAR) {
        float vel_sq_ac = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
        if (vel_sq_ac < GRANULAR_V_THRESHOLD_SQ) {
            float rho_i = sorted_density[i];
            float rho0_i = c_materials[mat_id].rest_density;
            float sr_i = sorted_shear_rate[i];
            float accel_eq = accel.x * accel.x + accel.y * accel.y + accel.z * accel.z;
            if (rho_i > GRANULAR_RHO_FACTOR * rho0_i && sr_i < GRANULAR_GAMMA_MIN
                && accel_eq < GRANULAR_ACCEL_REST_SQ) {
                vel_new.x = 0.0f;
                vel_new.y = 0.0f;
                vel_new.z = 0.0f;
            }
        }
    }

    // --- Position update ---
    // FLUID: use XSPH-corrected velocity for smoother advection (Game SPH).
    // Others: use actual velocity.
    float3 advect_vel;
    if (behavior == FLUID) {
        advect_vel = veleval_xsph;
    } else {
        advect_vel = vel_new;
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

    // --- Sleep counter update ---
    // Check if particle should start sleeping (low velocity AND low shear rate)
    // GRANULAR also requires equilibrium (no large unbalanced forces)
    float vel_sq_sleep = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
    float sr_sleep = sorted_shear_rate[i];

    if (vel_sq_sleep < V_SLEEP_SQ && sr_sleep < GAMMA_SLEEP) {
        bool can_sleep = true;
        if (behavior == GRANULAR) {
            float a_eq = accel.x * accel.x + accel.y * accel.y + accel.z * accel.z;
            can_sleep = (a_eq < GRANULAR_ACCEL_REST_SQ);
        }
        if (can_sleep) {
            if (sc < 255) sc++;
        } else {
            sc = 0;
        }
    } else {
        // Conditions not met: reset counter
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

    // --- Track max displacement for grid reuse (Phase 9.2) ---
    // displacement^2 = |pos_new - pos_old|^2
    // Uses float-as-uint trick: positive IEEE754 floats have monotonic bit order,
    // so atomicMax on __float_as_uint gives the max float value.
    if (max_displacement_out) {
        float dx_d = pos_new.x - pos.x;
        float dy_d = pos_new.y - pos.y;
        float dz_d = pos_new.z - pos.z;
        float disp_sq = dx_d * dx_d + dy_d * dy_d + dz_d * dz_d;
        atomicMax(max_displacement_out, __float_as_uint(disp_sq));
    }

    // --- Write to UNSORTED arrays ---
    position_out[orig_idx] = make_float4(pos_new.x, pos_new.y, pos_new.z, 1.0f);
    velocity_out[orig_idx] = make_float4(vel_new.x, vel_new.y, vel_new.z, 0.0f);
    color_out[orig_idx] = color;
    packed_info_out[orig_idx] = pi;
    sleep_counter_out[orig_idx] = sc;
    temperature_out[orig_idx] = temp;
    particle_dye_out[orig_idx] = dye;
    angular_velocity_out[orig_idx] = ang_vel;
}
