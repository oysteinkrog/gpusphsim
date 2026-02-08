/*
 * pbf_solver.cu -- Position Based Fluids (Macklin & Muller, SIGGRAPH 2013)
 *
 * Kernels:
 *   K_PBF_Predict       -- Apply gravity, predict position
 *   K_PBF_ComputeLambda -- Density + Lagrange multiplier
 *   K_PBF_ComputeDelta  -- Position correction + artificial pressure
 *   K_PBF_ApplyDelta    -- Apply correction + boundary clamp + friction
 *   K_PBF_Finalize      -- Velocity update, XSPH, color, sleep, writeback
 *
 * All particles participate in density constraints (FLUID + GRANULAR + GAS).
 * GRANULAR gets friction position-corrections in ApplyDelta.
 * GAS gets linear drag instead of PBF constraints.
 * STATIC particles are skipped but contribute to neighbor density sums.
 *
 * Constant memory:
 *   c_grid, c_sim, c_precalc, c_materials -- from common.cuh
 *   c_pbf -- local PBF parameters
 *   c_granular -- mu(I) parameters (for GRANULAR friction)
 */

#include "common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ======================================================================
 * PBF Parameters
 * ====================================================================== */

struct PBFParams {
    int   num_iterations;     // fixed iter count for graph capture
    float relaxation;         // epsilon for denominator stability
    float s_corr_k;           // artificial pressure coefficient
    int   s_corr_n;           // artificial pressure exponent
    float s_corr_dq_sq;       // (dq * h)^2 for s_corr reference
    float s_corr_W_dq;        // W_poly6(dq*h) precomputed
    float xsph_c;             // XSPH smoothing coefficient
    float padding;            // pad to 32 bytes
};

__constant__ PBFParams c_pbf;

/* Reuse GranularParams for friction clamp on GRANULAR particles */
struct GranularParams {
    float mu_s;
    float mu_2;
    float I0;
    float mu_max;
    float particle_spacing;
    float mu0;
    float xsph_epsilon;
    float force_scale;
    float vorticity_epsilon;
    float surface_tension_gamma;
    float tan_phi_f;          // tan(friction_angle) for Drucker-Prager (default tan(32°)=0.625)
    float cohesion;           // small cohesion for DP stability (default 0.001)
};

__constant__ GranularParams c_granular;

/* ======================================================================
 * SPH kernel functions (same as step2.cu)
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

/* ======================================================================
 * Boundary clamp helper
 * ====================================================================== */

__device__ inline void clamp_boundary(float3& pos) {
    pos.x = fmaxf(c_sim.world_min.x, fminf(pos.x, c_sim.world_max.x));
    pos.y = fmaxf(c_sim.world_min.y, fminf(pos.y, c_sim.world_max.y));
    pos.z = fmaxf(c_sim.world_min.z, fminf(pos.z, c_sim.world_max.z));
}

/* ======================================================================
 * Impulse-style SDF boundary for finalize (matches integrate.cu)
 * ====================================================================== */

__device__ inline void sdf_box_boundary(
    float3& pos, float3& vel,
    float3 wmin, float3 wmax,
    float restitution, float mu_wall
) {
    // X-axis
    if (pos.x < wmin.x) {
        pos.x = wmin.x;
        if (vel.x < 0.0f) {
            float vn = vel.x;
            vel.x = -restitution * vn;
            float ts = sqrtf(vel.y * vel.y + vel.z * vel.z);
            if (ts > 1e-8f) {
                float red = fminf(mu_wall * fabsf(vn) / ts, 1.0f);
                vel.y *= (1.0f - red);
                vel.z *= (1.0f - red);
            }
        }
    }
    if (pos.x > wmax.x) {
        pos.x = wmax.x;
        if (vel.x > 0.0f) {
            float vn = vel.x;
            vel.x = -restitution * vn;
            float ts = sqrtf(vel.y * vel.y + vel.z * vel.z);
            if (ts > 1e-8f) {
                float red = fminf(mu_wall * fabsf(vn) / ts, 1.0f);
                vel.y *= (1.0f - red);
                vel.z *= (1.0f - red);
            }
        }
    }
    // Y-axis
    if (pos.y < wmin.y) {
        pos.y = wmin.y;
        if (vel.y < 0.0f) {
            float vn = vel.y;
            vel.y = -restitution * vn;
            float ts = sqrtf(vel.x * vel.x + vel.z * vel.z);
            if (ts > 1e-8f) {
                float red = fminf(mu_wall * fabsf(vn) / ts, 1.0f);
                vel.x *= (1.0f - red);
                vel.z *= (1.0f - red);
            }
        }
    }
    if (pos.y > wmax.y) {
        pos.y = wmax.y;
        if (vel.y > 0.0f) {
            float vn = vel.y;
            vel.y = -restitution * vn;
            float ts = sqrtf(vel.x * vel.x + vel.z * vel.z);
            if (ts > 1e-8f) {
                float red = fminf(mu_wall * fabsf(vn) / ts, 1.0f);
                vel.x *= (1.0f - red);
                vel.z *= (1.0f - red);
            }
        }
    }
    // Z-axis
    if (pos.z < wmin.z) {
        pos.z = wmin.z;
        if (vel.z < 0.0f) {
            float vn = vel.z;
            vel.z = -restitution * vn;
            float ts = sqrtf(vel.x * vel.x + vel.y * vel.y);
            if (ts > 1e-8f) {
                float red = fminf(mu_wall * fabsf(vn) / ts, 1.0f);
                vel.x *= (1.0f - red);
                vel.y *= (1.0f - red);
            }
        }
    }
    if (pos.z > wmax.z) {
        pos.z = wmax.z;
        if (vel.z > 0.0f) {
            float vn = vel.z;
            vel.z = -restitution * vn;
            float ts = sqrtf(vel.x * vel.x + vel.y * vel.y);
            if (ts > 1e-8f) {
                float red = fminf(mu_wall * fabsf(vn) / ts, 1.0f);
                vel.x *= (1.0f - red);
                vel.y *= (1.0f - red);
            }
        }
    }
}

/* ======================================================================
 * Fluid color (matches integrate.cu)
 * ====================================================================== */

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
    float h = fmaxf(fminf(health, 1.0f), 0.0f);
    r = fminf(r * h, 1.0f);
    g = fminf(g * h, 1.0f);
    b = fminf(b * h, 1.0f);
    return make_float4(r, g, b, 0.0f);  // FLUID always 0.0
}

/* ======================================================================
 * Sleep/wake constants (match integrate.cu)
 * ====================================================================== */

#define V_SLEEP          0.005f
#define V_SLEEP_SQ       (V_SLEEP * V_SLEEP)
#define V_WAKE           0.02f
#define V_WAKE_SQ        (V_WAKE * V_WAKE)
#define SLEEP_THRESHOLD  10

/* Temperature integration constants */
#define T_AMBIENT        293.0f
#define COOL_RATE        0.1f
#define T_MIN            0.0f
#define T_MAX            5000.0f

/* GAS constants */
#define GAS_BUOYANCY_BETA  0.01f
#define GAS_AMBIENT_TEMP   293.0f
#define GAS_BUOYANCY_G     9.81f
#define GAS_DRAG_COEFF     2.0f

#define VELOCITY_LIMIT     10.0f
#define VELOCITY_LIMIT_SQ  (VELOCITY_LIMIT * VELOCITY_LIMIT)

// Uses calcGridCell() and spatialHash() from common.cuh.
// get_cell() is an alias for calcGridCell().
__device__ inline int3 get_cell(float3 pos) { return calcGridCell(pos); }

/* ======================================================================
 * K_PBF_Predict -- Apply gravity, predict position
 *   x*_i = x_i + dt * (v_i + dt * g)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_PBF_Predict(
    uint            numParticles,
    const float4*   __restrict__ position,
    const float4*   __restrict__ velocity,
    const uint*     __restrict__ packed_info,
    float4*         __restrict__ predicted_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    int behavior = GET_BEHAVIOR(pi);

    float4 pos4 = position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);

    if (behavior == STATIC || IS_SLEEPING(pi)) {
        predicted_out[i] = pos4;
        return;
    }

    float4 vel4 = velocity[i];
    float3 vel = make_float3(vel4.x, vel4.y, vel4.z);
    float dt = c_sim.dt;

    // v* = v + dt * g
    float3 v_star = make_float3(
        vel.x + dt * c_sim.gravity.x,
        vel.y + dt * c_sim.gravity.y,
        vel.z + dt * c_sim.gravity.z
    );

    // GAS: add buoyancy + drag
    if (behavior == GAS) {
        float temp = 293.0f;  // approx (temperature not passed to predict)
        v_star.y += GAS_BUOYANCY_BETA * (temp - GAS_AMBIENT_TEMP) * GAS_BUOYANCY_G * dt;
        float drag = 1.0f - GAS_DRAG_COEFF * dt;
        drag = fmaxf(drag, 0.0f);
        v_star.x *= drag;
        v_star.y *= drag;
        v_star.z *= drag;
    }

    // x* = x + dt * v*
    float3 pred = make_float3(
        pos.x + dt * v_star.x,
        pos.y + dt * v_star.y,
        pos.z + dt * v_star.z
    );

    clamp_boundary(pred);
    predicted_out[i] = make_float4(pred.x, pred.y, pred.z, 1.0f);
}

/* ======================================================================
 * K_PBF_ComputeLambda -- Density + Lagrange multiplier
 *   rho_i = SUM_j m_j W(x*_ij)
 *   C_i = rho_i / rho0 - 1
 *   lambda_i = -C_i / (SUM_j |grad_pj C_i|^2 + epsilon)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_PBF_ComputeLambda(
    uint            numParticles,
    const float4*   __restrict__ predicted_pos,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ density_out,
    float*          __restrict__ lambda_out,
    float4*         __restrict__ pressure_normal_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);

    if (behavior_i == STATIC) {
        density_out[i] = c_materials[mat_id_i].rest_density;
        lambda_out[i] = 0.0f;
        pressure_normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float4 pos4_i = predicted_pos[i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float rho0 = c_materials[mat_id_i].rest_density;

    // Density accumulator (Poly6)
    float sum_density = 0.0f;
    // Gradient sum for constraint denominator (Spiky)
    float3 grad_ci = make_float3(0.0f, 0.0f, 0.0f);  // grad_pi C_i
    float sum_grad_sq = 0.0f;  // SUM_j |grad_pj C_i|^2 for j != i

    int3 cell_i = get_cell(pos_i);

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    // Speculative ILP: issue all loads before distance check (OPT-4.2)
                    float4 pos4_j = __ldg(&predicted_pos[j]);
                    float m_j = __ldg(&mass[j]);
                    uint pi_j = __ldg(&packed_info[j]);

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq) continue;

                    // Skip STATIC neighbors for density
                    if (GET_BEHAVIOR(pi_j) == STATIC && j != i) continue;

                    // Density (Poly6, self-included)
                    float diff = h_sq - r_sq;
                    sum_density += m_j * diff * diff * diff;

                    // Gradient (Spiky, skip self)
                    if (j != i && r_sq > 1e-12f) {
                        float rlen = sqrtf(r_sq);
                        float3 gW = grad_spiky(r, rlen, h);
                        // grad_pj C_i = (1/rho0) * m_j * grad_W(x*_i - x*_j)
                        // But for denominator: |grad_pj C_i|^2 = (m_j/rho0)^2 * |gradW|^2
                        float scale = m_j / rho0;
                        float gx = scale * gW.x;
                        float gy = scale * gW.y;
                        float gz = scale * gW.z;
                        sum_grad_sq += gx * gx + gy * gy + gz * gz;
                        // grad_pi C_i += (1/rho0) * m_j * gradW (accumulate)
                        grad_ci.x += gx;
                        grad_ci.y += gy;
                        grad_ci.z += gz;
                    }
                }
            }
        }
    }

    float rho = c_precalc.poly6_coeff * sum_density;
    rho = fmaxf(rho, 1.0f);
    density_out[i] = rho;

    // Store normalized density gradient as pressure normal for GRANULAR friction
    // grad_ci points in the direction of maximum density increase (into the pile)
    if (behavior_i == GRANULAR) {
        float grad_len = sqrtf(grad_ci.x*grad_ci.x + grad_ci.y*grad_ci.y + grad_ci.z*grad_ci.z);
        if (grad_len > 1e-8f) {
            float inv = 1.0f / grad_len;
            pressure_normal_out[i] = make_float4(grad_ci.x*inv, grad_ci.y*inv, grad_ci.z*inv, grad_len);
        } else {
            // Fallback to gravity direction when no gradient available
            pressure_normal_out[i] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
        }
    } else {
        pressure_normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Constraint: C_i = rho_i / rho0 - 1
    // Over-dense (C>0): full repulsive correction (incompressibility)
    // Under-dense (C<0): weak attractive correction (5%) for surface cohesion
    float C_raw = rho / rho0 - 1.0f;
    float C_i = (C_raw >= 0.0f) ? C_raw : C_raw * 0.05f;

    if (fabsf(C_i) < 1e-8f) {
        lambda_out[i] = 0.0f;
        return;
    }

    // Denominator: |grad_pi C_i|^2 + SUM_j |grad_pj C_i|^2 + epsilon
    float denom = grad_ci.x * grad_ci.x + grad_ci.y * grad_ci.y + grad_ci.z * grad_ci.z
                + sum_grad_sq + c_pbf.relaxation;

    lambda_out[i] = -C_i / denom;
}

/* ======================================================================
 * K_PBF_ComputeDelta -- Position correction + artificial pressure
 *   s_corr = -k * (W(r) / W(dq*h))^n
 *   dx_i = (1/rho0) SUM_j (lambda_i + lambda_j + s_corr) * gradW_spiky
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_PBF_ComputeDelta(
    uint            numParticles,
    const float4*   __restrict__ predicted_pos,
    const float*    __restrict__ lambda_pbf,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float4*         __restrict__ delta_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);

    if (behavior_i == STATIC || IS_SLEEPING(pi_i)) {
        delta_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float4 pos4_i = predicted_pos[i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float rho0 = c_materials[mat_id_i].rest_density;
    float lambda_i = lambda_pbf[i];

    float3 delta = make_float3(0.0f, 0.0f, 0.0f);

    int3 cell_i = get_cell(pos_i);

    float W_dq = c_pbf.s_corr_W_dq;

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
                    float4 pos4_j = __ldg(&predicted_pos[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float lambda_j = __ldg(&lambda_pbf[j]);
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

                    // Artificial pressure (tensile instability fix)
                    float W_ij = W_poly6(r_sq, h_sq);
                    float s_corr = 0.0f;
                    if (W_dq > 1e-12f) {
                        float ratio = W_ij / W_dq;
                        // ratio^n (integer power)
                        float rn = ratio * ratio;  // n=4: ratio^2
                        rn = rn * rn;              // ratio^4
                        s_corr = -c_pbf.s_corr_k * rn;
                    }

                    float3 gW = grad_spiky(r, rlen, h);
                    // Mass-weighted: dp = (m_j/rho0) * (l_i+l_j+s) * gradW
                    float coeff = m_j * (lambda_i + lambda_j + s_corr) / rho0;
                    delta.x += coeff * gW.x;
                    delta.y += coeff * gW.y;
                    delta.z += coeff * gW.z;
                }
            }
        }
    }

    // Clamp delta magnitude to fraction of smoothing length to prevent
    // grid invalidation during iterative solve (grid isn't re-sorted).
    // With correct mass-weighted corrections, typical delta is ~0.2mm
    // so this clamp is a safety net, not routinely hit.
    float max_delta = 0.5f * h;
    float delta_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    if (delta_sq > max_delta * max_delta) {
        float scale = max_delta / sqrtf(delta_sq);
        delta.x *= scale;
        delta.y *= scale;
        delta.z *= scale;
    }

    delta_out[i] = make_float4(delta.x, delta.y, delta.z, 0.0f);
}

/* ======================================================================
 * K_PBF_ApplyDelta -- Apply correction + boundary + GRANULAR friction
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_PBF_ApplyDelta(
    uint            numParticles,
    float4*         __restrict__ predicted_pos,   // in-out
    const float4*   __restrict__ delta_pos,
    const uint*     __restrict__ packed_info,
    const float4*   __restrict__ pressure_normal
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    if (GET_BEHAVIOR(pi) == STATIC || IS_SLEEPING(pi)) return;

    float4 pos4 = predicted_pos[i];
    float4 d4 = delta_pos[i];

    float3 pos = make_float3(pos4.x + d4.x, pos4.y + d4.y, pos4.z + d4.z);

    // GRANULAR: position-space Drucker-Prager friction using pressure normal.
    // Force-space tan(phi_f)=0.78 allows 78% tangential -- far too permissive
    // for position corrections which directly displace particles each iteration.
    // Use a much tighter ratio (0.15) + static friction dead zone to prevent
    // PBF incompressibility corrections from spreading granular material.
    if (GET_BEHAVIOR(pi) == GRANULAR) {
        float4 pn4 = __ldg(&pressure_normal[i]);
        float3 n = make_float3(pn4.x, pn4.y, pn4.z);
        float n_len_sq = n.x*n.x + n.y*n.y + n.z*n.z;

        if (n_len_sq > 0.5f) {  // valid normal (len ~1)
            // Decompose delta into normal and tangential components
            float d_dot_n = d4.x*n.x + d4.y*n.y + d4.z*n.z;
            float3 delta_n = make_float3(d_dot_n*n.x, d_dot_n*n.y, d_dot_n*n.z);
            float3 delta_t = make_float3(d4.x - delta_n.x, d4.y - delta_n.y, d4.z - delta_n.z);

            float tang_sq = delta_t.x*delta_t.x + delta_t.y*delta_t.y + delta_t.z*delta_t.z;
            float norm_mag = fabsf(d_dot_n);

            // Position-space friction ratio (~8.5° effective angle)
            // Much stricter than force-space tan(38°)=0.78 because PBF corrections
            // directly move particles -- accumulated tangential drift causes water-like
            // spreading at tan(38°).
            #define PBF_POS_FRICTION  0.15f
            #define PBF_STATIC_THRESH 0.00005f  // 0.05mm: below this normal, zero tangential

            float max_tang;
            if (norm_mag < PBF_STATIC_THRESH) {
                // Static friction: near-equilibrium, suppress all tangential
                max_tang = 0.0f;
            } else {
                // Dynamic friction: allow limited tangential
                max_tang = PBF_POS_FRICTION * norm_mag;
            }

            if (tang_sq > max_tang * max_tang) {
                if (max_tang > 0.0f && tang_sq > 1e-12f) {
                    float scale = max_tang / sqrtf(tang_sq);
                    delta_t.x *= scale;
                    delta_t.y *= scale;
                    delta_t.z *= scale;
                } else {
                    delta_t.x = 0.0f;
                    delta_t.y = 0.0f;
                    delta_t.z = 0.0f;
                }
            }

            // Recombine: pos = original + clamped_normal + clamped_tangential
            pos.x = pos4.x + delta_n.x + delta_t.x;
            pos.y = pos4.y + delta_n.y + delta_t.y;
            pos.z = pos4.z + delta_n.z + delta_t.z;
        }
        // else: fallback to unclamped delta (already applied above)
    }

    clamp_boundary(pos);
    predicted_pos[i] = make_float4(pos.x, pos.y, pos.z, 1.0f);
}

/* ======================================================================
 * K_PBF_Finalize -- Velocity update, XSPH, color, sleep, writeback
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_PBF_Finalize(
    uint            numParticles,
    const float4*   __restrict__ predicted_pos,
    const float4*   __restrict__ original_pos,
    const float4*   __restrict__ original_vel,
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const float*    __restrict__ temperature_in,
    const float*    __restrict__ health_in,
    const float*    __restrict__ dTdt_in,
    const unsigned char* __restrict__ sleep_counter_in,
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
    const float4*   __restrict__ sorted_particle_dye,
    float4*         __restrict__ particle_dye_out,
    const float4*   __restrict__ sorted_angular_velocity,
    float4*         __restrict__ angular_velocity_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    int behavior = GET_BEHAVIOR(pi);
    uint mat_id = GET_MATERIAL_ID(pi);
    uint orig_idx = sort_indexes[i];

    float temp = temperature_in[i];
    float hlth = health_in[i];

    // STATIC: just write through
    if (behavior == STATIC) {
        float4 pos4 = original_pos[i];
        float dTdt = dTdt_in[i];
        temp += dTdt * c_sim.dt;
        temp -= COOL_RATE * (temp - T_AMBIENT) * c_sim.dt;
        temp = fmaxf(T_MIN, fminf(temp, T_MAX));
        position_out[orig_idx] = pos4;
        velocity_out[orig_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        color_out[orig_idx] = compute_color(mat_id, temp, hlth, STATIC);
        packed_info_out[orig_idx] = pi;
        sleep_counter_out[orig_idx] = sleep_counter_in[i];
        temperature_out[orig_idx] = temp;
        particle_dye_out[orig_idx] = sorted_particle_dye[i];
        angular_velocity_out[orig_idx] = sorted_angular_velocity[i];
        return;
    }

    unsigned char sc = sleep_counter_in[i];
    bool was_sleeping = IS_SLEEPING(pi) != 0;

    float4 pred4 = predicted_pos[i];
    float3 pred = make_float3(pred4.x, pred4.y, pred4.z);

    float4 orig4 = original_pos[i];
    float3 orig = make_float3(orig4.x, orig4.y, orig4.z);

    float dt = c_sim.dt;
    float inv_dt = 1.0f / fmaxf(dt, 1e-8f);

    // Sleeping: check wake condition
    if (was_sleeping) {
        float3 dp = make_float3(pred.x - orig.x, pred.y - orig.y, pred.z - orig.z);
        float dp_sq = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z;
        float vel_sq = dp_sq * inv_dt * inv_dt;
        if (vel_sq <= V_WAKE_SQ) {
            // Still sleeping
            float dTdt = dTdt_in[i];
            temp += dTdt * dt;
            temp -= COOL_RATE * (temp - T_AMBIENT) * dt;
            temp = fmaxf(T_MIN, fminf(temp, T_MAX));
            position_out[orig_idx] = orig4;
            velocity_out[orig_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            color_out[orig_idx] = compute_color(mat_id, temp, hlth, behavior);
            packed_info_out[orig_idx] = pi;
            sleep_counter_out[orig_idx] = (sc < 255) ? sc : (unsigned char)255;
            temperature_out[orig_idx] = temp;
            particle_dye_out[orig_idx] = sorted_particle_dye[i];
            angular_velocity_out[orig_idx] = sorted_angular_velocity[i];
            return;
        }
        pi = CLEAR_SLEEPING(pi);
        pi = SET_JUST_WOKE(pi);
        sc = 0;
    }

    // Standard PBF velocity: v = (x* - x_old) / dt
    // With correct mass-weighted delta, no artificial damping needed.
    float3 vel_new = make_float3(
        (pred.x - orig.x) * inv_dt,
        (pred.y - orig.y) * inv_dt,
        (pred.z - orig.z) * inv_dt
    );

    // FLUID thermal convection: Boussinesq buoyancy
    // rho_eff = rho_0 * (1 - beta*(T-T0)), lighter when hot -> rises
    if (behavior == FLUID) {
        float beta = c_materials[mat_id].thermal_expansion;
        if (beta > 0.0f) {
            vel_new.y += beta * (temp - T_AMBIENT) * 9.81f * dt;
        }
    }

    // XSPH viscosity: v += c * SUM_j (m_j/rho_j)(v_j - v_i) W(r)
    if (behavior == FLUID || behavior == GRANULAR) {
        float h = c_sim.smoothing_length;
        float h_sq = c_sim.smoothing_length_sq;
        float rho_i = density[i];
        float3 xsph = make_float3(0.0f, 0.0f, 0.0f);

        int3 cell_i = get_cell(pred);
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {

                    uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                    uint start = cell_start[hash_c];
                    if (start == 0xFFFFFFFFu) continue;
                    uint end_idx = cell_end[hash_c];

                    for (uint j = start; j < end_idx; j++) {
                        if (j == i) continue;

                        uint pi_j = __ldg(&packed_info[j]);
                        if (GET_BEHAVIOR(pi_j) == STATIC) continue;

                        float4 pred4_j = __ldg(&predicted_pos[j]);
                        float3 r = make_float3(
                            pred.x - pred4_j.x,
                            pred.y - pred4_j.y,
                            pred.z - pred4_j.z
                        );
                        float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                        if (r_sq > h_sq) continue;

                        float4 orig4_j = __ldg(&original_pos[j]);
                        float3 vel_j = make_float3(
                            (pred4_j.x - orig4_j.x) * inv_dt,
                            (pred4_j.y - orig4_j.y) * inv_dt,
                            (pred4_j.z - orig4_j.z) * inv_dt
                        );
                        float m_j = __ldg(&mass[j]);
                        float rho_j = __ldg(&density[j]);
                        float w = W_poly6(r_sq, h_sq);
                        float rho_avg = 0.5f * (rho_i + rho_j);
                        float factor = (m_j / rho_avg) * w;
                        xsph.x += (vel_j.x - vel_new.x) * factor;
                        xsph.y += (vel_j.y - vel_new.y) * factor;
                        xsph.z += (vel_j.z - vel_new.z) * factor;
                    }
                }
            }
        }

        float c_xsph = c_pbf.xsph_c;
        // GRANULAR: higher artificial viscosity
        if (behavior == GRANULAR) c_xsph *= 10.0f;
        vel_new.x += c_xsph * xsph.x;
        vel_new.y += c_xsph * xsph.y;
        vel_new.z += c_xsph * xsph.z;
    }

    // GRANULAR: friction velocity damping when in dense packing.
    // Only apply when the particle has neighbors (rho > 0.7*rho0), so free-falling
    // sand isn't artificially slowed. Damps residual spreading from PBF corrections.
    if (behavior == GRANULAR) {
        float rho_i_damp = density[i];
        float rho0_damp = c_materials[mat_id].rest_density;
        if (rho_i_damp > 0.7f * rho0_damp) {
            float granular_damp = 1.0f - 10.0f * dt;
            granular_damp = fmaxf(granular_damp, 0.3f);
            vel_new.x *= granular_damp;
            vel_new.y *= granular_damp;
            vel_new.z *= granular_damp;
        }
    }

    // Boundary collision on final position
    float3 final_pos = pred;
    float friction = (behavior == FLUID) ? 0.0f : c_sim.wall_friction;
    sdf_box_boundary(final_pos, vel_new, c_sim.world_min, c_sim.world_max,
                     c_sim.restitution, friction);

    // Velocity clamp
    float vel_sq = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
    if (vel_sq > VELOCITY_LIMIT_SQ) {
        float scale = VELOCITY_LIMIT / sqrtf(vel_sq);
        vel_new.x *= scale;
        vel_new.y *= scale;
        vel_new.z *= scale;
        vel_sq = VELOCITY_LIMIT_SQ;
    }

    // GRANULAR anti-creep: zero velocity when nearly at rest and well-packed.
    // Higher threshold than WCSPH (0.05 vs 0.01) because PBF position corrections
    // generate artificial spreading velocity that must be caught here.
    if (behavior == GRANULAR) {
        vel_sq = vel_new.x*vel_new.x + vel_new.y*vel_new.y + vel_new.z*vel_new.z;
        if (vel_sq < 0.05f * 0.05f) {
            float rho_i = density[i];
            float rho0_i = c_materials[mat_id].rest_density;
            if (rho_i > 0.90f * rho0_i) {
                vel_new.x = 0.0f;
                vel_new.y = 0.0f;
                vel_new.z = 0.0f;
                vel_sq = 0.0f;
            }
        }
    }

    // Sleep: velocity-based (replaces shear_rate for PBF)
    if (vel_sq < V_SLEEP_SQ) {
        if (sc < 255) sc++;
    } else {
        sc = 0;
    }
    if (sc >= SLEEP_THRESHOLD) {
        pi = SET_SLEEPING(pi);
    }

    // Temperature integration
    float dTdt = dTdt_in[i];
    temp += dTdt * dt;
    temp -= COOL_RATE * (temp - T_AMBIENT) * dt;
    temp = fmaxf(T_MIN, fminf(temp, T_MAX));

    // Color (color.w encodes behavior class for SSFR shader filtering)
    float4 color;
    if (behavior == FLUID) {
        color = compute_fluid_color(mat_id, temp, hlth, final_pos.y, vel_sq, density[i]);
    } else {
        color = compute_color(mat_id, temp, hlth, behavior);
    }

    // Writeback (unsorted)
    position_out[orig_idx] = make_float4(final_pos.x, final_pos.y, final_pos.z, 1.0f);
    velocity_out[orig_idx] = make_float4(vel_new.x, vel_new.y, vel_new.z, 0.0f);
    color_out[orig_idx] = color;
    packed_info_out[orig_idx] = pi;
    sleep_counter_out[orig_idx] = sc;
    temperature_out[orig_idx] = temp;
    particle_dye_out[orig_idx] = sorted_particle_dye[i];
    angular_velocity_out[orig_idx] = sorted_angular_velocity[i];  // passthrough (no vorticity in PBF)
}
