/*
 * dfsph_solver.cu -- Divergence-Free SPH (Bender & Koschier, SCA 2015)
 *
 * Kernels:
 *   K_DFSPH_ComputeDensityAlpha -- Density + alpha precompute + shear_rate + heat
 *   K_DFSPH_NonPressureForces   -- Viscosity + gravity -> velocity update
 *   K_DFSPH_ComputeKappaV       -- Divergence correction factor
 *   K_DFSPH_CorrectVelocityDiv  -- Apply divergence correction
 *   K_DFSPH_PredictPosition     -- Position prediction
 *   K_DFSPH_ComputeDensityAdv   -- Recompute density at predicted positions
 *   K_DFSPH_ComputeKappa        -- Density correction factor
 *   K_DFSPH_CorrectVelocityDens -- Apply density correction
 *   K_DFSPH_Finalize            -- Final position, boundary, color, writeback
 *
 * Constant memory:
 *   c_grid, c_sim, c_precalc, c_materials -- from common.cuh
 *   c_dfsph -- local DFSPH parameters
 */

#include "common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ======================================================================
 * DFSPH Parameters
 * ====================================================================== */

struct DFSPHParams {
    int   div_iters;
    int   dens_iters;
    float warm_start;
    float omega;       // under-relaxation factor for corrections (0.3-1.0)
};

__constant__ DFSPHParams c_dfsph;

/* ======================================================================
 * SPH kernel functions
 * ====================================================================== */

__device__ inline float W_poly6(float rlen_sq, float h_sq) {
    float diff = h_sq - rlen_sq;
    return c_precalc.poly6_coeff * diff * diff * diff;
}

__device__ inline float3 grad_spiky(float3 r, float rlen, float h) {
    float h_rlen = h - rlen;
    float coeff = c_precalc.spiky_grad_coeff * h_rlen * h_rlen / rlen;
    return make_float3(coeff * r.x, coeff * r.y, coeff * r.z);
}

__device__ inline float lap_visc_variable(float rlen, float h) {
    return h - rlen;
}

/* ======================================================================
 * Helpers
 * ====================================================================== */

// Uses calcGridCell() and spatialHash() from common.cuh.
// get_cell() is an alias for calcGridCell().
__device__ inline int3 get_cell(float3 pos) { return calcGridCell(pos); }

__device__ inline void clamp_boundary(float3& pos) {
    pos.x = fmaxf(c_sim.world_min.x, fminf(pos.x, c_sim.world_max.x));
    pos.y = fmaxf(c_sim.world_min.y, fminf(pos.y, c_sim.world_max.y));
    pos.z = fmaxf(c_sim.world_min.z, fminf(pos.z, c_sim.world_max.z));
}

__device__ inline void sdf_box_boundary(
    float3& pos, float3& vel,
    float3 wmin, float3 wmax,
    float restitution, float mu_wall
) {
    #define AXIS_BOUNDARY(P, V, LO, HI, VT1, VT2) \
        if (P < LO) { P = LO; if (V < 0.0f) { float vn=V; V=-restitution*vn; float ts=sqrtf(VT1*VT1+VT2*VT2); if(ts>1e-8f){float r=fminf(mu_wall*fabsf(vn)/ts,1.0f);VT1*=(1.0f-r);VT2*=(1.0f-r);}}} \
        if (P > HI) { P = HI; if (V > 0.0f) { float vn=V; V=-restitution*vn; float ts=sqrtf(VT1*VT1+VT2*VT2); if(ts>1e-8f){float r=fminf(mu_wall*fabsf(vn)/ts,1.0f);VT1*=(1.0f-r);VT2*=(1.0f-r);}}}
    AXIS_BOUNDARY(pos.x, vel.x, wmin.x, wmax.x, vel.y, vel.z)
    AXIS_BOUNDARY(pos.y, vel.y, wmin.y, wmax.y, vel.x, vel.z)
    AXIS_BOUNDARY(pos.z, vel.z, wmin.z, wmax.z, vel.x, vel.y)
    #undef AXIS_BOUNDARY
}

/* Encode behavior class into color.w for SSFR material filtering.
 * FLUID=0.0, GRANULAR=0.25, GAS=0.5, STATIC=0.75 */
__device__ inline float behavior_to_alpha(int behavior) {
    return behavior * 0.25f;
}

__device__ inline float4 compute_color(uint mat_id, float temperature, float health, int behavior) {
    float r = c_materials[mat_id].color_r;
    float g = c_materials[mat_id].color_g;
    float b = c_materials[mat_id].color_b;
    if (temperature > 293.0f) {
        float t_excess = fminf((temperature - 293.0f) / 1000.0f, 1.0f);
        r = r + (1.0f - r) * t_excess;
        g = g * (1.0f - 0.5f * t_excess);
        b = b * (1.0f - 0.8f * t_excess);
    }
    float h = fmaxf(fminf(health, 1.0f), 0.0f);
    return make_float4(r * h, g * h, b * h, behavior_to_alpha(behavior));
}

__device__ inline float4 compute_fluid_color(
    uint mat_id, float temperature, float health,
    float pos_y, float vel_sq, float density
) {
    float rho0 = c_materials[mat_id].rest_density;
    float base_r = c_materials[mat_id].color_r;
    float base_g = c_materials[mat_id].color_g;
    float base_b = c_materials[mat_id].color_b;
    float y_range = c_sim.world_max.y - c_sim.world_min.y;
    float depth_t = (pos_y - c_sim.world_min.y) / fmaxf(y_range, 0.01f);
    depth_t = fmaxf(0.0f, fminf(depth_t, 1.0f));
    float r = base_r * (0.45f + 0.70f * depth_t);
    float g = base_g * (0.50f + 0.65f * depth_t);
    float b = base_b * (0.65f + 0.40f * depth_t);
    float ratio = density / fmaxf(rho0, 1.0f);
    float compress = fmaxf(ratio - 1.0f, 0.0f);
    float darken = 1.0f / (1.0f + 0.5f * compress);
    r *= darken; g *= darken; b *= darken;
    float speed = sqrtf(vel_sq);
    float foam_t = fminf(speed / 3.0f, 1.0f);
    foam_t = foam_t * foam_t;
    r = r + (1.0f - r) * foam_t * 0.7f;
    g = g + (1.0f - g) * foam_t * 0.7f;
    b = b + (1.0f - b) * foam_t * 0.7f;
    if (temperature > 293.0f) {
        float t_excess = fminf((temperature - 293.0f) / 1000.0f, 1.0f);
        r = r + (1.0f - r) * t_excess;
        g = g * (1.0f - 0.5f * t_excess);
        b = b * (1.0f - 0.8f * t_excess);
    }
    float hh = fmaxf(fminf(health, 1.0f), 0.0f);
    r = fminf(r * hh, 1.0f);
    g = fminf(g * hh, 1.0f);
    b = fminf(b * hh, 1.0f);
    return make_float4(r, g, b, 0.0f);  // FLUID always 0.0
}

/* Constants */
#define V_SLEEP          0.005f
#define V_SLEEP_SQ       (V_SLEEP * V_SLEEP)
#define V_WAKE           0.02f
#define V_WAKE_SQ        (V_WAKE * V_WAKE)
#define SLEEP_THRESHOLD  10
#define T_AMBIENT        293.0f
#define COOL_RATE        0.1f
#define T_MIN            0.0f
#define T_MAX            5000.0f
#define GAS_BUOYANCY_BETA  0.01f
#define GAS_AMBIENT_TEMP   293.0f
#define GAS_BUOYANCY_G     9.81f
#define GAS_DRAG_COEFF     2.0f
#define VELOCITY_LIMIT     10.0f
#define VELOCITY_LIMIT_SQ  (VELOCITY_LIMIT * VELOCITY_LIMIT)
#define ACCEL_MAX          30.0f
#define ACCEL_MAX_SQ       (ACCEL_MAX * ACCEL_MAX)

/* ======================================================================
 * K_DFSPH_ComputeDensityAlpha
 * Density + alpha factor + shear_rate + heat diffusion + exposure
 * (Modified Step1 that also computes the DFSPH alpha diagonal factor)
 *
 * alpha_i = 1 / (|SUM_j (m_j/rho_j) gradW_ij|^2
 *              + SUM_j (m_j/rho_j)^2 |gradW_ij|^2 + eps)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ComputeDensityAlpha(
    uint            numParticles,
    const float4*   __restrict__ position,
    const float4*   __restrict__ velocity,
    const float*    __restrict__ mass,
    const float*    __restrict__ density_in,      // previous density (NULL first frame)
    const uint*     __restrict__ packed_info,
    const float*    __restrict__ temperature_in,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ density_out,
    float*          __restrict__ alpha_out,
    float*          __restrict__ shear_rate_out,
    float*          __restrict__ dTdt_out,
    float*          __restrict__ exposure_heat_out,
    float*          __restrict__ exposure_corrode_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float4 pos4_i = position[i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;

    uint pi_i = __ldg(&packed_info[i]);
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);
    bool is_granular = (behavior_i == GRANULAR);

    float3 vel_i;
    if (is_granular) {
        float4 v4 = velocity[i];
        vel_i = make_float3(v4.x, v4.y, v4.z);
    }

    float T_i = __ldg(&temperature_in[i]);
    float kappa_i = c_materials[mat_id_i].thermal_conductivity;

    float sum_density = 0.0f;
    float sum_dTdt = 0.0f;
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;

    // Strain-rate tensor (GRANULAR only)
    float Dxx = 0.0f, Dyy = 0.0f, Dzz = 0.0f;
    float Dxy = 0.0f, Dxz = 0.0f, Dyz = 0.0f;

    // Alpha accumulators
    float3 grad_sum = make_float3(0.0f, 0.0f, 0.0f);  // SUM_j (m_j/rho_j) gradW_ij
    float grad_norm_sum = 0.0f;                          // SUM_j (m_j/rho_j)^2 |gradW_ij|^2

    int3 cell_i = get_cell(pos_i);

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    float4 pos4_j = __ldg(&position[j]);
                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq) continue;

                    float m_j = __ldg(&mass[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    int behavior_j = GET_BEHAVIOR(pi_j);

                    // Skip STATIC neighbors for density (keep self)
                    if (behavior_j == STATIC && j != i) continue;

                    // Density (Poly6)
                    float diff = h_sq - r_sq;
                    sum_density += m_j * diff * diff * diff;

                    // Skip self for non-density quantities
                    if (j != i && r_sq > 1e-12f) {
                        float rlen = sqrtf(r_sq);
                        float rho_j = (density_in != 0) ? __ldg(&density_in[j]) : 1000.0f;

                        // Heat diffusion
                        float T_j = __ldg(&temperature_in[j]);
                        float lap_var = h - rlen;
                        sum_dTdt += m_j / fmaxf(rho_j, 1.0f) * (T_j - T_i) * lap_var;

                        // Exposure
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        float w_poly6_var = diff * diff * diff;
                        sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                        sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;

                        // Alpha factor (using Spiky gradient)
                        float3 gW = grad_spiky(r, rlen, h);
                        float w_j = m_j / fmaxf(rho_j, 1.0f);
                        grad_sum.x += w_j * gW.x;
                        grad_sum.y += w_j * gW.y;
                        grad_sum.z += w_j * gW.z;
                        grad_norm_sum += w_j * w_j * (gW.x * gW.x + gW.y * gW.y + gW.z * gW.z);

                        // Strain-rate tensor (GRANULAR only)
                        if (is_granular) {
                            float grad_scalar = c_precalc.spiky_grad_coeff * (h - rlen) * (h - rlen) / rlen;
                            float gWx = grad_scalar * r.x;
                            float gWy = grad_scalar * r.y;
                            float gWz = grad_scalar * r.z;
                            float4 vel4_j = __ldg(&velocity[j]);
                            float dvx = vel_i.x - vel4_j.x;
                            float dvy = vel_i.y - vel4_j.y;
                            float dvz = vel_i.z - vel4_j.z;
                            float weight = m_j / fmaxf(rho_j, 1.0f);
                            Dxx += weight * dvx * gWx;
                            Dyy += weight * dvy * gWy;
                            Dzz += weight * dvz * gWz;
                            Dxy += 0.5f * weight * (dvx * gWy + dvy * gWx);
                            Dxz += 0.5f * weight * (dvx * gWz + dvz * gWx);
                            Dyz += 0.5f * weight * (dvy * gWz + dvz * gWy);
                        }
                    }
                }
            }
        }
    }

    // Density
    float rho = c_precalc.poly6_coeff * sum_density;
    rho = fmaxf(rho, 1.0f);
    density_out[i] = rho;

    // Alpha = 1 / denom (inverse diagonal of pressure Poisson system)
    // Cap alpha to prevent surface particles (few neighbors) from getting
    // extreme correction values. Interior alpha ~ 1e-7, cap at 1e-5.
    float denom = grad_sum.x * grad_sum.x + grad_sum.y * grad_sum.y + grad_sum.z * grad_sum.z
                + grad_norm_sum;
    float alpha_threshold = 1e-6f;
    float alpha_max = 1e-5f;  // ~100x interior value -- safe for surface particles
    if (is_granular) {
        float d = denom * 0.7f;
        alpha_out[i] = (d > alpha_threshold) ? fminf(1.0f / d, alpha_max) : 0.0f;
    } else {
        // DEBUG: also write uncapped alpha (1/denom) into the .w component of alpha_out?
        // Actually, let's write denom to shear_rate for non-granular debug
        alpha_out[i] = (denom > alpha_threshold) ? fminf(1.0f / denom, alpha_max) : 0.0f;
    }
    // Heat diffusion
    dTdt_out[i] = kappa_i * c_precalc.viscosity_lap_coeff * sum_dTdt;

    // Exposure
    exposure_heat_out[i] = c_precalc.poly6_coeff * sum_exposure_heat;
    exposure_corrode_out[i] = c_precalc.poly6_coeff * sum_exposure_corrode;

    // Shear rate
    if (is_granular) {
        float D_sq = Dxx * Dxx + Dyy * Dyy + Dzz * Dzz
                   + 2.0f * (Dxy * Dxy + Dxz * Dxz + Dyz * Dyz);
        shear_rate_out[i] = sqrtf(fmaxf(2.0f * D_sq, 0.0f));
    } else {
        shear_rate_out[i] = 0.0f;
    }
}

/* ======================================================================
 * K_DFSPH_NonPressureForces -- Viscosity + gravity -> velocity update
 * v*_i = v_i + dt * (gravity + viscosity)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_NonPressureForces(
    uint            numParticles,
    const float4*   __restrict__ position,
    const float4*   __restrict__ velocity_in,
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const float*    __restrict__ shear_rate,
    const float*    __restrict__ temperature,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float4*         __restrict__ velocity_out     // updated velocity
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);

    if (behavior_i == STATIC || IS_SLEEPING(pi_i)) {
        velocity_out[i] = velocity_in[i];
        return;
    }

    float4 pos4_i = position[i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float4 vel4_i = velocity_in[i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);

    float rho_i = density[i];
    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;

    float3 f_visc = make_float3(0.0f, 0.0f, 0.0f);
    bool is_granular_i = (behavior_i == GRANULAR);

    // Compute viscosity via neighbor loop
    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP: issue all loads before distance check (OPT-4.2)
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float4 vel4_j = __ldg(&velocity_in[j]);
                    float rho_j = __ldg(&density[j]);
                    float m_j = __ldg(&mass[j]);

                    if (GET_BEHAVIOR(pi_j) == STATIC) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);
                    float3 vel_j = make_float3(vel4_j.x, vel4_j.y, vel4_j.z);

                    float lap_v = lap_visc_variable(rlen, h);
                    float visc_factor = m_j * lap_v / fmaxf(rho_j, 1.0f);
                    f_visc.x += (vel_j.x - vel_i.x) * visc_factor;
                    f_visc.y += (vel_j.y - vel_i.y) * visc_factor;
                    f_visc.z += (vel_j.z - vel_i.z) * visc_factor;
                }
            }
        }
    }

    // Apply viscosity coefficient (divided by rho_i -- standard SPH viscosity)
    // a_visc = (mu / rho_i) * lap_coeff * SUM_j m_j/rho_j * (v_j - v_i) * (h - |r|)
    // viscosity_precalc = mu0 * 45/(pi*h^6), so we divide by rho_i
    float visc_coeff = c_precalc.viscosity_precalc / fmaxf(rho_i, 1.0f);
    float3 accel = make_float3(
        visc_coeff * f_visc.x + c_sim.gravity.x,
        visc_coeff * f_visc.y + c_sim.gravity.y,
        visc_coeff * f_visc.z + c_sim.gravity.z
    );

    // GAS buoyancy + drag
    if (behavior_i == GAS) {
        float temp = __ldg(&temperature[i]);
        accel.y += GAS_BUOYANCY_BETA * (temp - GAS_AMBIENT_TEMP) * GAS_BUOYANCY_G;
    }

    // FLUID thermal convection: Boussinesq buoyancy
    // rho_eff = rho_0 * (1 - beta*(T-T0)), lighter when hot -> rises
    if (behavior_i == FLUID) {
        float temp_i = __ldg(&temperature[i]);
        float beta = c_materials[GET_MATERIAL_ID(pi_i)].thermal_expansion;
        if (beta > 0.0f) {
            accel.y += beta * (temp_i - T_AMBIENT) * 9.81f;
        }
    }

    // Accel clamp
    float accel_sq = accel.x * accel.x + accel.y * accel.y + accel.z * accel.z;
    if (accel_sq > ACCEL_MAX_SQ) {
        float scale = ACCEL_MAX / sqrtf(accel_sq);
        accel.x *= scale;
        accel.y *= scale;
        accel.z *= scale;
    }

    // v* = v + dt * accel
    float3 vel_new = make_float3(
        vel_i.x + dt * accel.x,
        vel_i.y + dt * accel.y,
        vel_i.z + dt * accel.z
    );

    // GAS drag
    if (behavior_i == GAS) {
        float drag = fmaxf(1.0f - GAS_DRAG_COEFF * dt, 0.0f);
        vel_new.x *= drag;
        vel_new.y *= drag;
        vel_new.z *= drag;
    }

    // Velocity clamp
    float v_sq = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
    if (v_sq > VELOCITY_LIMIT_SQ) {
        float scale = VELOCITY_LIMIT / sqrtf(v_sq);
        vel_new.x *= scale;
        vel_new.y *= scale;
        vel_new.z *= scale;
    }

    velocity_out[i] = make_float4(vel_new.x, vel_new.y, vel_new.z, 0.0f);
}

/* ======================================================================
 * K_DFSPH_ComputeKappaV -- Divergence correction factor
 * div_v = SUM_j m_j/rho_j * (v*_i - v*_j) . gradW_ij
 * kappa_v_i = (1/dt) * div_v / max(alpha_i, eps)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ComputeKappaV(
    uint            numParticles,
    const float4*   __restrict__ velocity,
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const float*    __restrict__ alpha,
    const uint*     __restrict__ packed_info,
    const float4*   __restrict__ position,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ kappa_v_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);

    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) {
        kappa_v_out[i] = 0.0f;
        return;
    }

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float4 vel4_i = velocity[i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;

    float div_v = 0.0f;

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP (OPT-4.2)
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float4 vel4_j = __ldg(&velocity[j]);
                    float rho_j = __ldg(&density[j]);
                    float m_j = __ldg(&mass[j]);

                    if (GET_BEHAVIOR(pi_j) == STATIC) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 gW = grad_spiky(r, rlen, h);
                    float dv_dot_gW = (vel_i.x - vel4_j.x) * gW.x
                                    + (vel_i.y - vel4_j.y) * gW.y
                                    + (vel_i.z - vel4_j.z) * gW.z;
                    div_v += (m_j / fmaxf(rho_j, 1.0f)) * dv_dot_gW;
                }
            }
        }
    }

    float alpha_i = alpha[i];
    // DFSPH: kv = (1/dt) * div_v * alpha
    // div_v uses (v_i - v_j) . gradW which gives -drho/dt/rho (note the sign).
    // For diverging particles: div_v < 0, so kv < 0 -> correction pushes inward.
    // omega: under-relaxation to prevent overcorrection
    kappa_v_out[i] = div_v * alpha_i * c_dfsph.omega / dt;
}

/* ======================================================================
 * K_DFSPH_CorrectVelocityDiv -- Apply divergence correction
 * dv_i = -(dt/rho_i) SUM_j m_j * (kv_i/rho_i + kv_j/rho_j) * gradW_ij
 * v*_i += dv_i
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_CorrectVelocityDiv(
    uint            numParticles,
    float4*         __restrict__ velocity,        // in-out
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const float*    __restrict__ kappa_v,
    const uint*     __restrict__ packed_info,
    const float4*   __restrict__ position,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) return;

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;
    float rho_i = density[i];
    float kv_i = kappa_v[i];

    float3 dv = make_float3(0.0f, 0.0f, 0.0f);

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP (OPT-4.2)
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float rho_j = __ldg(&density[j]);
                    float m_j = __ldg(&mass[j]);
                    float kv_j = __ldg(&kappa_v[j]);

                    if (GET_BEHAVIOR(pi_j) == STATIC) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 gW = grad_spiky(r, rlen, h);
                    float V_j = m_j / fmaxf(rho_j, 1.0f);
                    float coeff = V_j * (kv_i + kv_j);
                    dv.x += coeff * gW.x;
                    dv.y += coeff * gW.y;
                    dv.z += coeff * gW.z;
                }
            }
        }
    }

    // Apply velocity correction: delta_v = -dt * dv
    // Clamp |delta_v| to h/dt (one cell per step) to prevent grid-skipping
    float3 delta_v = make_float3(dt * dv.x, dt * dv.y, dt * dv.z);
    float max_delta = c_sim.smoothing_length / dt;
    float delta_sq = delta_v.x * delta_v.x + delta_v.y * delta_v.y + delta_v.z * delta_v.z;
    if (delta_sq > max_delta * max_delta) {
        float scale = max_delta / sqrtf(delta_sq);
        delta_v.x *= scale;
        delta_v.y *= scale;
        delta_v.z *= scale;
    }

    float4 vel4 = velocity[i];
    vel4.x -= delta_v.x;
    vel4.y -= delta_v.y;
    vel4.z -= delta_v.z;
    velocity[i] = vel4;
}

/* ======================================================================
 * K_DFSPH_PredictPosition -- x*_i = x_i + dt * v*_i
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_PredictPosition(
    uint            numParticles,
    const float4*   __restrict__ position,
    const float4*   __restrict__ velocity,
    const uint*     __restrict__ packed_info,
    float4*         __restrict__ predicted_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    if (GET_BEHAVIOR(pi) == STATIC || IS_SLEEPING(pi)) {
        predicted_out[i] = position[i];
        return;
    }

    float4 pos4 = position[i];
    float4 vel4 = velocity[i];
    float dt = c_sim.dt;

    float3 pred = make_float3(
        pos4.x + dt * vel4.x,
        pos4.y + dt * vel4.y,
        pos4.z + dt * vel4.z
    );
    clamp_boundary(pred);
    predicted_out[i] = make_float4(pred.x, pred.y, pred.z, 1.0f);
}

/* ======================================================================
 * K_DFSPH_ComputeDensityAdv -- Recompute density at predicted positions
 * Uses original sorted neighbor structure but predicted positions
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ComputeDensityAdv(
    uint            numParticles,
    const float4*   __restrict__ predicted_pos,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const float4*   __restrict__ original_pos,    // for grid cell lookup
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ density_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    if (GET_BEHAVIOR(pi_i) == STATIC) {
        density_out[i] = c_materials[GET_MATERIAL_ID(pi_i)].rest_density;
        return;
    }

    // Use ORIGINAL position for grid cell lookup (neighbor structure not re-sorted)
    float4 orig4_i = original_pos[i];
    float3 orig_i = make_float3(orig4_i.x, orig4_i.y, orig4_i.z);

    // Use PREDICTED position for distance computation
    float4 pred4_i = predicted_pos[i];
    float3 pred_i = make_float3(pred4_i.x, pred4_i.y, pred4_i.z);

    float h_sq = c_sim.smoothing_length_sq;
    float sum_density = 0.0f;

    int3 cell_i = get_cell(orig_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    uint pi_j = __ldg(&packed_info[j]);
                    if (GET_BEHAVIOR(pi_j) == STATIC && j != i) continue;

                    float4 pred4_j = __ldg(&predicted_pos[j]);
                    float3 r = make_float3(
                        pred_i.x - pred4_j.x,
                        pred_i.y - pred4_j.y,
                        pred_i.z - pred4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq) continue;

                    float m_j = __ldg(&mass[j]);
                    float diff = h_sq - r_sq;
                    sum_density += m_j * diff * diff * diff;
                }
            }
        }
    }

    float rho = c_precalc.poly6_coeff * sum_density;
    density_out[i] = fmaxf(rho, 1.0f);
}

/* ======================================================================
 * K_DFSPH_ComputeKappa -- Density correction factor (old, from poly6 density)
 * Kept for reference; replaced by K_DFSPH_ComputeKappaFromVelocity below.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ComputeKappa(
    uint            numParticles,
    const float*    __restrict__ density,
    const float*    __restrict__ alpha,
    const uint*     __restrict__ packed_info,
    float*          __restrict__ kappa_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    int behavior = GET_BEHAVIOR(pi);

    if (behavior == STATIC || behavior == GAS || IS_SLEEPING(pi)) {
        kappa_out[i] = 0.0f;
        return;
    }

    float rho0 = c_materials[GET_MATERIAL_ID(pi)].rest_density;
    float rho_star = density[i];
    float alpha_i = alpha[i];
    float dt = c_sim.dt;

    float rho_err = rho_star / rho0 - 1.0f;  // bidirectional
    kappa_out[i] = rho_err * alpha_i * c_dfsph.omega / (dt * dt);
}

/* ======================================================================
 * K_DFSPH_ComputeKappaFromVelocity -- Density prediction + kappa
 *
 * SPlisHSPlasH-style density prediction: instead of computing poly6 density
 * at predicted positions (which requires a stale grid), predict the density
 * from the current velocity field:
 *   rho_adv = rho + dt * Sigma_j m_j (v_i - v_j) . grad_W_ij
 * This uses the current grid and current positions, avoiding stale-grid artifacts.
 * The kappa is then: kappa = max((rho_adv/rho0 - 1), 0) * alpha * omega / dt^2
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ComputeKappaFromVelocity(
    uint            numParticles,
    const float4*   __restrict__ velocity,
    const float4*   __restrict__ position,
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const float*    __restrict__ alpha,
    const uint*     __restrict__ packed_info,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ kappa_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);

    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) {
        kappa_out[i] = 0.0f;
        return;
    }

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float4 vel4_i = velocity[i];

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;
    float rho_i = density[i];

    // Compute drho/dt = Sum_j m_j (v_i - v_j) . grad_W_ij
    // This is the SPH-consistent density rate of change.
    float drho_dt = 0.0f;

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP (OPT-4.2)
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float m_j = __ldg(&mass[j]);
                    float4 vel4_j = __ldg(&velocity[j]);

                    if (GET_BEHAVIOR(pi_j) == STATIC) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 gW = grad_spiky(r, rlen, h);
                    float dv_dot_gW = (vel4_i.x - vel4_j.x) * gW.x
                                    + (vel4_i.y - vel4_j.y) * gW.y
                                    + (vel4_i.z - vel4_j.z) * gW.z;
                    drho_dt += m_j * dv_dot_gW;
                }
            }
        }
    }

    // Density prediction: rho_adv = rho + dt * drho/dt
    float rho0 = c_materials[GET_MATERIAL_ID(pi_i)].rest_density;
    float rho_adv = rho_i + dt * drho_dt;

    // Normalized density error -- allow BOTH compression AND expansion correction.
    // Without expansion correction, overcorrection at large dt creates a one-way pump
    // that continuously expands the fluid. Bidirectional correction is self-stabilizing.
    float rho_err = rho_adv / rho0 - 1.0f;

    float alpha_i = alpha[i];
    kappa_out[i] = rho_err * alpha_i * c_dfsph.omega / (dt * dt);
}

/* ======================================================================
 * K_DFSPH_CorrectVelocityDens -- Apply density correction
 * Same structure as CorrectVelocityDiv but uses kappa instead of kappa_v
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_CorrectVelocityDens(
    uint            numParticles,
    float4*         __restrict__ velocity,        // in-out
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const float*    __restrict__ kappa,
    const uint*     __restrict__ packed_info,
    const float4*   __restrict__ position,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) return;

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;
    float rho_i = density[i];
    float k_i = kappa[i];

    float3 dv = make_float3(0.0f, 0.0f, 0.0f);

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP (OPT-4.2)
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float rho_j = __ldg(&density[j]);
                    float m_j = __ldg(&mass[j]);
                    float k_j = __ldg(&kappa[j]);

                    if (GET_BEHAVIOR(pi_j) == STATIC) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 gW = grad_spiky(r, rlen, h);
                    float V_j = m_j / fmaxf(rho_j, 1.0f);
                    float coeff = V_j * (k_i + k_j);
                    dv.x += coeff * gW.x;
                    dv.y += coeff * gW.y;
                    dv.z += coeff * gW.z;
                }
            }
        }
    }

    // Apply velocity correction: delta_v = -dt * dv
    // Clamp |delta_v| to h/dt (one cell per step) to prevent grid-skipping
    float3 delta_v = make_float3(dt * dv.x, dt * dv.y, dt * dv.z);
    float max_delta = c_sim.smoothing_length / dt;
    float delta_sq = delta_v.x * delta_v.x + delta_v.y * delta_v.y + delta_v.z * delta_v.z;
    if (delta_sq > max_delta * max_delta) {
        float scale = max_delta / sqrtf(delta_sq);
        delta_v.x *= scale;
        delta_v.y *= scale;
        delta_v.z *= scale;
    }

    float4 vel4 = velocity[i];
    vel4.x -= delta_v.x;
    vel4.y -= delta_v.y;
    vel4.z -= delta_v.z;
    velocity[i] = vel4;
}

/* ======================================================================
 * K_DFSPH_ComputePressureAccel -- Pressure acceleration from p/rho^2
 *
 * Jacobi density solver kernel 1 of 3:
 * a_press_i = -SUM_j V_j * (p_rho2_i + p_rho2_j) * grad_W_ij
 * where V_j = m_j / rho_j (particle volume)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ComputePressureAccel(
    uint            numParticles,
    const float*    __restrict__ p_rho2,
    const float4*   __restrict__ position,
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float4*         __restrict__ accel_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) {
        accel_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float p_i = p_rho2[i];

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;

    float3 a = make_float3(0.0f, 0.0f, 0.0f);

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP: issue all loads before distance check (OPT-4.2)
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float rho_j = __ldg(&density[j]);
                    float m_j = __ldg(&mass[j]);
                    float p_j = __ldg(&p_rho2[j]);

                    if (GET_BEHAVIOR(pi_j) == STATIC) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 gW = grad_spiky(r, rlen, h);
                    float V_j = m_j / fmaxf(rho_j, 1.0f);
                    float coeff = -V_j * (p_i + p_j);
                    a.x += coeff * gW.x;
                    a.y += coeff * gW.y;
                    a.z += coeff * gW.z;
                }
            }
        }
    }

    accel_out[i] = make_float4(a.x, a.y, a.z, 0.0f);
}

/* ======================================================================
 * K_DFSPH_DensitySolverUpdate -- Jacobi update for density pressure
 *
 * Jacobi density solver kernel 2 of 3:
 * 1) Compute predicted density using v_total = v + dt * a_press:
 *    drho = SUM_j (m_j/rho_j) * (v_total_i - v_total_j) . grad_W_ij
 *    density_adv = rho_i / rho0 + dt * drho
 * 2) Jacobi update pressure:
 *    residual = density_adv - 1.0
 *    p_rho2 = max(p_rho2 + omega * residual * alpha / dt^2, 0)
 *
 * The A*p feedback comes from including a_press in the density prediction,
 * which accounts for the effect of current pressure on neighbors.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_DensitySolverUpdate(
    uint            numParticles,
    const float4*   __restrict__ velocity,
    const float4*   __restrict__ accel_press,
    const float4*   __restrict__ position,
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const float*    __restrict__ alpha,
    const uint*     __restrict__ packed_info,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ p_rho2          // in/out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) {
        p_rho2[i] = 0.0f;
        return;
    }

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float4 vel4_i = velocity[i];
    float4 ap4_i = accel_press[i];

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;
    float rho_i = density[i];
    float rho0 = c_materials[GET_MATERIAL_ID(pi_i)].rest_density;

    // Total velocity = v + dt * a_press (includes pressure effect)
    float3 vt_i = make_float3(
        vel4_i.x + dt * ap4_i.x,
        vel4_i.y + dt * ap4_i.y,
        vel4_i.z + dt * ap4_i.z
    );

    // Compute density rate-of-change from total velocity
    float drho = 0.0f;

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP: issue all loads before distance check (OPT-4.2)
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float m_j = __ldg(&mass[j]);
                    float rho_j = __ldg(&density[j]);
                    float4 vel4_j = __ldg(&velocity[j]);
                    float4 ap4_j = __ldg(&accel_press[j]);

                    if (GET_BEHAVIOR(pi_j) == STATIC) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 vt_j = make_float3(
                        vel4_j.x + dt * ap4_j.x,
                        vel4_j.y + dt * ap4_j.y,
                        vel4_j.z + dt * ap4_j.z
                    );

                    float3 gW = grad_spiky(r, rlen, h);
                    float dv_dot_gW = (vt_i.x - vt_j.x) * gW.x
                                    + (vt_i.y - vt_j.y) * gW.y
                                    + (vt_i.z - vt_j.z) * gW.z;
                    drho += (m_j / fmaxf(rho_j, 1.0f)) * dv_dot_gW;
                }
            }
        }
    }

    // Predicted density ratio (normalized to rest density)
    float density_adv = rho_i / rho0 + dt * drho;

    // Residual: remaining density error (positive = compressed)
    float residual = density_adv - 1.0f;

    // Jacobi update: accumulate pressure correction
    // p_rho2 += omega * residual * alpha / dt^2
    // Clamp total pressure to non-negative (no tensile/attractive pressure)
    float alpha_i = alpha[i];
    float old_p = p_rho2[i];
    float new_p = old_p + c_dfsph.omega * residual * alpha_i / (dt * dt);
    p_rho2[i] = fmaxf(new_p, 0.0f);
}

/* ======================================================================
 * K_DFSPH_ApplyPressureVelocity -- Apply converged pressure to velocity
 *
 * Jacobi density solver kernel 3 of 3:
 * v_i += dt * a_press_i
 * Called once after Jacobi iteration converges.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ApplyPressureVelocity(
    uint            numParticles,
    float4*         __restrict__ velocity,
    const float4*   __restrict__ accel_press,
    const uint*     __restrict__ packed_info
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    if (GET_BEHAVIOR(pi) == STATIC || GET_BEHAVIOR(pi) == GAS || IS_SLEEPING(pi))
        return;

    float dt = c_sim.dt;
    float4 vel = velocity[i];
    float4 ap = accel_press[i];

    vel.x += dt * ap.x;
    vel.y += dt * ap.y;
    vel.z += dt * ap.z;

    // Velocity clamp
    float v_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    if (v_sq > VELOCITY_LIMIT_SQ) {
        float scale = VELOCITY_LIMIT / sqrtf(v_sq);
        vel.x *= scale;
        vel.y *= scale;
        vel.z *= scale;
    }

    velocity[i] = vel;
}

/* ======================================================================
 * K_DFSPH_Finalize -- Final position, boundary, color, sleep, writeback
 * x_final = x + dt * v*_final
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_Finalize(
    uint            numParticles,
    const float4*   __restrict__ sorted_position,
    const float4*   __restrict__ sorted_velocity,
    const float*    __restrict__ sorted_density,
    const float*    __restrict__ sorted_mass,
    const uint*     __restrict__ sorted_packed_info,
    const float*    __restrict__ sorted_temperature,
    const float*    __restrict__ sorted_health,
    const float*    __restrict__ sorted_dTdt,
    const unsigned char* __restrict__ sorted_sleep_counter,
    const float*    __restrict__ sorted_kappa,
    const uint*     __restrict__ sort_indexes,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    // Unsorted outputs
    float4*         __restrict__ position_out,
    float4*         __restrict__ velocity_out,
    float4*         __restrict__ color_out,
    uint*           __restrict__ packed_info_out,
    unsigned char*  __restrict__ sleep_counter_out,
    float*          __restrict__ temperature_out,
    float*          __restrict__ kappa_out,
    const float4*   __restrict__ sorted_particle_dye,
    float4*         __restrict__ particle_dye_out,
    const float4*   __restrict__ sorted_angular_velocity,
    float4*         __restrict__ angular_velocity_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = sorted_packed_info[i];
    int behavior = GET_BEHAVIOR(pi);
    uint mat_id = GET_MATERIAL_ID(pi);
    uint orig_idx = sort_indexes[i];

    float temp = sorted_temperature[i];
    float hlth = sorted_health[i];

    // STATIC: write through
    if (behavior == STATIC) {
        float4 pos4 = sorted_position[i];
        float dTdt = sorted_dTdt[i];
        temp += dTdt * c_sim.dt;
        temp -= COOL_RATE * (temp - T_AMBIENT) * c_sim.dt;
        temp = fmaxf(T_MIN, fminf(temp, T_MAX));
        position_out[orig_idx] = pos4;
        velocity_out[orig_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        color_out[orig_idx] = compute_color(mat_id, temp, hlth, STATIC);
        packed_info_out[orig_idx] = pi;
        sleep_counter_out[orig_idx] = sorted_sleep_counter[i];
        temperature_out[orig_idx] = temp;
        kappa_out[orig_idx] = 0.0f;
        particle_dye_out[orig_idx] = sorted_particle_dye[i];
        angular_velocity_out[orig_idx] = sorted_angular_velocity[i];
        return;
    }

    unsigned char sc = sorted_sleep_counter[i];
    bool was_sleeping = IS_SLEEPING(pi) != 0;

    float4 pos4 = sorted_position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
    float4 vel4 = sorted_velocity[i];
    float3 vel = make_float3(vel4.x, vel4.y, vel4.z);
    float dt = c_sim.dt;

    // Sleeping check
    if (was_sleeping) {
        float vel_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
        if (vel_sq <= V_WAKE_SQ) {
            float dTdt = sorted_dTdt[i];
            temp += dTdt * dt;
            temp -= COOL_RATE * (temp - T_AMBIENT) * dt;
            temp = fmaxf(T_MIN, fminf(temp, T_MAX));
            position_out[orig_idx] = pos4;
            velocity_out[orig_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            color_out[orig_idx] = compute_color(mat_id, temp, hlth, behavior);
            packed_info_out[orig_idx] = pi;
            sleep_counter_out[orig_idx] = (sc < 255) ? sc : (unsigned char)255;
            temperature_out[orig_idx] = temp;
            kappa_out[orig_idx] = 0.0f;
            particle_dye_out[orig_idx] = sorted_particle_dye[i];
            angular_velocity_out[orig_idx] = sorted_angular_velocity[i];
            return;
        }
        pi = CLEAR_SLEEPING(pi);
        pi = SET_JUST_WOKE(pi);
        sc = 0;
    }

    // Final position: x_final = x + dt * v*
    float3 pos_new = make_float3(
        pos.x + dt * vel.x,
        pos.y + dt * vel.y,
        pos.z + dt * vel.z
    );

    // Boundary
    float friction = (behavior == FLUID) ? 0.0f : c_sim.wall_friction;
    sdf_box_boundary(pos_new, vel, c_sim.world_min, c_sim.world_max,
                     c_sim.restitution, friction);

    // Velocity clamp
    float vel_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    if (vel_sq > VELOCITY_LIMIT_SQ) {
        float scale = VELOCITY_LIMIT / sqrtf(vel_sq);
        vel.x *= scale;
        vel.y *= scale;
        vel.z *= scale;
        vel_sq = VELOCITY_LIMIT_SQ;
    }

    // GRANULAR anti-creep: zero velocity when nearly at rest and well-packed
    // (matches WCSPH integrate.cu behavior -- prevents slow spreading)
    if (behavior == GRANULAR && vel_sq < 0.01f * 0.01f) {
        float rho_i = sorted_density[i];
        float rho0_i = c_materials[mat_id].rest_density;
        if (rho_i > 0.95f * rho0_i) {
            vel.x = 0.0f;
            vel.y = 0.0f;
            vel.z = 0.0f;
            vel_sq = 0.0f;
        }
    }

    // Sleep (velocity-based)
    if (vel_sq < V_SLEEP_SQ) {
        if (sc < 255) sc++;
    } else {
        sc = 0;
    }
    if (sc >= SLEEP_THRESHOLD) {
        pi = SET_SLEEPING(pi);
    }

    // Temperature
    float dTdt = sorted_dTdt[i];
    temp += dTdt * dt;
    temp -= COOL_RATE * (temp - T_AMBIENT) * dt;
    temp = fmaxf(T_MIN, fminf(temp, T_MAX));

    // Color (color.w encodes behavior class for SSFR shader filtering)
    float4 color;
    if (behavior == FLUID) {
        color = compute_fluid_color(mat_id, temp, hlth, pos_new.y, vel_sq, sorted_density[i]);
    } else {
        color = compute_color(mat_id, temp, hlth, behavior);
    }

    // Writeback
    position_out[orig_idx] = make_float4(pos_new.x, pos_new.y, pos_new.z, 1.0f);
    velocity_out[orig_idx] = make_float4(vel.x, vel.y, vel.z, 0.0f);
    color_out[orig_idx] = color;
    packed_info_out[orig_idx] = pi;
    sleep_counter_out[orig_idx] = sc;
    temperature_out[orig_idx] = temp;
    kappa_out[orig_idx] = sorted_kappa[i];
    particle_dye_out[orig_idx] = sorted_particle_dye[i];
    angular_velocity_out[orig_idx] = sorted_angular_velocity[i];  // passthrough (no vorticity in DFSPH)
}
