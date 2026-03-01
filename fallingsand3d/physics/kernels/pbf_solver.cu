/*
 * pbf_solver.cu -- Position Based Fluids (Macklin & Muller, SIGGRAPH 2013)
 *
 * Kernels:
 *   K_PBF_Predict       -- Apply gravity, predict position
 *   K_PBF_ComputeLambda -- Density + Lagrange multiplier
 *   K_PBF_ComputeDelta  -- Position correction + artificial pressure
 *   K_PBF_ApplyDelta    -- Apply correction + boundary clamp
 *   K_PBF_Finalize      -- Velocity update, XSPH, friction, color, sleep, writeback
 *
 * All particles participate in density constraints (FLUID + GRANULAR + GAS).
 * GRANULAR gets velocity-space Drucker-Prager friction in Finalize (iteration-independent).
 * GAS gets linear drag instead of PBF constraints.
 * STATIC particles are skipped but contribute to neighbor density sums.
 *
 * Constant memory:
 *   c_grid, c_sim, c_precalc, c_materials -- from common.cuh
 *   c_pbf -- local PBF parameters
 *   c_granular -- mu(I) parameters (for GRANULAR friction)
 */

#include "sph_shared.cuh"

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
    float neg_c_scale;        // scale factor for negative C (surface cohesion, default 0.05)
};

__constant__ PBFParams c_pbf;

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
    const float*    __restrict__ temperature_in,
    float4*         __restrict__ predicted_out
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    int behavior = GET_BEHAVIOR(pi);

    float4 pos4 = position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);

    // STATIC skips prediction entirely. Sleeping particles still predict
    // (gravity integration) so PBF corrections can wake them via density changes.
    if (behavior == STATIC) {
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
        float temp = __ldg(&temperature_in[i]);
        v_star.y += GAS_BUOYANCY_BETA * (temp - GAS_AMBIENT_TEMP) * GAS_BUOYANCY_G * dt;
        float drag = 1.0f - GAS_DRAG_COEFF * dt;
        drag = fmaxf(drag, 0.0f);
        v_star.x *= drag;
        v_star.y *= drag;
        v_star.z *= drag;
    }

    // FLUID thermal convection: Boussinesq buoyancy in prediction
    if (behavior == FLUID) {
        uint mat_id = GET_MATERIAL_ID(pi);
        float beta = c_materials[mat_id].thermal_expansion;
        if (beta > 0.0f) {
            float temp = __ldg(&temperature_in[i]);
            v_star.y += beta * (temp - T_AMBIENT) * 9.81f * dt;
        }
    }

    // x* = x + dt * v*
    float3 pred = make_float3(
        pos.x + dt * v_star.x,
        pos.y + dt * v_star.y,
        pos.z + dt * v_star.z
    );

    clamp_boundary(pred);

    // SDF object collision: clamp predicted position outside SDF surfaces
    for (int s = 0; s < c_num_sdf_objects; s++) {
        SDFObject obj = c_sdf_objects[s];
        float dist = eval_sdf(pred, obj);
        if (dist < BOUNDARY_MARGIN) {
            float3 n = sdf_normal(pred, obj);
            float push = BOUNDARY_MARGIN - dist;
            pred.x += push * n.x;
            pred.y += push * n.y;
            pred.z += push * n.z;
        }
    }

    predicted_out[i] = make_float4(pred.x, pred.y, pred.z, 1.0f);
}

/* ======================================================================
 * K_PBF_ComputeLambda -- Density + Lagrange multiplier
 *   rho_i = SUM_j m_j W(x*_ij)
 *   C_i = rho_i / rho0 - 1
 *   lambda_i = -C_i / (SUM_j |grad_pj C_i|^2 + epsilon)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 3)
void K_PBF_ComputeLambda(
    uint            numParticles,
    const float4*   __restrict__ predicted_pos,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ density_out,
    float*          __restrict__ lambda_out,
    float4*         __restrict__ pressure_normal_out,
    // Optional heat diffusion + exposure + dye (NULL to skip, used only on first call)
    const float*    __restrict__ temperature_in,
    const float*    __restrict__ density_in,
    float*          __restrict__ dTdt_out,
    float*          __restrict__ exposure_heat_out,
    float*          __restrict__ exposure_corrode_out,
    const float4*   __restrict__ particle_dye_in,
    float4*         __restrict__ dye_rate_out,
    const float4*   __restrict__ velocity_in,           // for vorticity (NULL to skip)
    float4*         __restrict__ vorticity_out,          // (curl_v.x, .y, .z, |curl_v|), NULL to skip
    float4*         __restrict__ normal_out              // (n_x, n_y, n_z, neighbor_count), NULL to skip
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);
    bool is_gas_i = (behavior_i == GAS);

    bool do_heat = (dTdt_out != 0);
    bool do_vort = (vorticity_out != 0) && (behavior_i == FLUID);

    // STATIC/GAS early return: skip density/lambda/gradient when no heat/exposure needed.
    // When do_heat is true (first call), STATIC particles still need exposure from
    // cross-phase neighbors (e.g., fire-to-wood ignition), so they fall through to the
    // neighbor loop but get lambda=0 forced after (see below).
    // GAS skips PBF constraints entirely (compressible phase).
    if ((behavior_i == STATIC || behavior_i == GAS) && !do_heat) {
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

    // Heat diffusion + exposure + dye accumulators (only when do_heat)
    float sum_dTdt = 0.0f;
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;
    float3 dye_rate = make_float3(0.0f, 0.0f, 0.0f);
    float T_i = 0.0f;
    float kappa_i = 0.0f;
    float4 dye_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool dye_on = (c_sim.dye_enabled != 0);
    if (do_heat) {
        T_i = __ldg(&temperature_in[i]);
        kappa_i = c_materials[mat_id_i].thermal_conductivity;
        if (dye_on) dye_i = __ldg(&particle_dye_in[i]);
    }

    // Vorticity accumulator (FLUID only)
    float3 omega = make_float3(0.0f, 0.0f, 0.0f);
    float3 vel_i_vort = make_float3(0.0f, 0.0f, 0.0f);
    if (do_vort) {
        float4 v4 = __ldg(&velocity_in[i]);
        vel_i_vort = make_float3(v4.x, v4.y, v4.z);
    }

    // Surface normal accumulator (FLUID only)
    bool do_normal = (behavior_i == FLUID && normal_out != 0);
    float3 normal = make_float3(0.0f, 0.0f, 0.0f);
    float neighbor_count = 0.0f;

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

                    int behavior_j = GET_BEHAVIOR(pi_j);

                    // GAS/non-GAS phase separation: skip density/gradient,
                    // keep only exposure (fire-to-wood ignition still works)
                    bool is_gas_j = (behavior_j == GAS);
                    if (is_gas_i != is_gas_j && j != i) {
                        if (do_heat) {
                            float diff_xp = h_sq - r_sq;
                            float T_j = __ldg(&temperature_in[j]);
                            uint mat_id_j = GET_MATERIAL_ID(pi_j);
                            float w_poly6_var = diff_xp * diff_xp * diff_xp;
                            sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                            sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
                        }
                        continue;
                    }

                    // Density (Poly6, self-included).
                    // STATIC boundary boost: multiply contribution by 2x so solid
                    // boundaries create stronger density inflation, preventing fluid
                    // from leaking through coarse STATIC particle layers.
                    float diff = h_sq - r_sq;
                    float boundary_scale = (behavior_j == STATIC && j != i) ? 2.0f : 1.0f;
                    sum_density += boundary_scale * m_j * diff * diff * diff;

                    // Gradient + heat + exposure (skip self)
                    if (j != i && r_sq > 1e-12f) {
                        float rlen = sqrtf(r_sq);
                        float3 gW = grad_spiky(r, rlen, h);
                        // grad_pj C_i = (1/rho0) * m_j * grad_W(x*_i - x*_j)
                        // But for denominator: |grad_pj C_i|^2 = (m_j/rho0)^2 * |gradW|^2
                        // Apply same boundary_scale to gradients for consistency
                        float scale = boundary_scale * m_j / rho0;
                        float gx = scale * gW.x;
                        float gy = scale * gW.y;
                        float gz = scale * gW.z;
                        sum_grad_sq += gx * gx + gy * gy + gz * gz;
                        // grad_pi C_i += (1/rho0) * m_j * gradW (accumulate)
                        grad_ci.x += gx;
                        grad_ci.y += gy;
                        grad_ci.z += gz;

                        // Heat diffusion + exposure (skip MAT_RIGID neighbors)
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        if (do_heat && mat_id_j != MAT_RIGID) {
                            float rho_j = (density_in != 0) ? __ldg(&density_in[j]) : 1000.0f;
                            float T_j = __ldg(&temperature_in[j]);

                            float lap_var = h - rlen;
                            float heat_boost = fmaxf(1.0f, c_interactions[mat_id_i][mat_id_j].heat_exchange);
                            sum_dTdt += m_j / fmaxf(rho_j, 1.0f) * (T_j - T_i) * lap_var * heat_boost;

                            float w_poly6_var = diff * diff * diff;
                            sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                            sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;

                            if (dye_on && behavior_j != STATIC) {
                                float vol_j = m_j / fmaxf(rho_j, 1.0f);
                                float dye_factor = 0.01f * vol_j * c_precalc.viscosity_lap_coeff * lap_var;
                                float4 dye_j = __ldg(&particle_dye_in[j]);
                                dye_rate.x += dye_factor * (dye_j.x - dye_i.x);
                                dye_rate.y += dye_factor * (dye_j.y - dye_i.y);
                                dye_rate.z += dye_factor * (dye_j.z - dye_i.z);
                            }
                        }

                        // Vorticity + surface normal (FLUID only, skip MAT_RIGID)
                        if ((do_vort || do_normal) && mat_id_j != MAT_RIGID) {
                            float rho_j_v = (density_in != 0) ? __ldg(&density_in[j]) : 1000.0f;
                            float vol_jv = m_j / fmaxf(rho_j_v, 1.0f);
                            if (do_vort) {
                                float4 vj4 = __ldg(&velocity_in[j]);
                                float dvx = vel_i_vort.x - vj4.x;
                                float dvy = vel_i_vort.y - vj4.y;
                                float dvz = vel_i_vort.z - vj4.z;
                                // dv = v_i - v_j, (v_j-v_i)xgW = -dvxgW = gWxdv
                                omega.x += vol_jv * (gW.y * dvz - gW.z * dvy);
                                omega.y += vol_jv * (gW.z * dvx - gW.x * dvz);
                                omega.z += vol_jv * (gW.x * dvy - gW.y * dvx);
                            }
                            if (do_normal) {
                                normal.x += vol_jv * gW.x;
                                normal.y += vol_jv * gW.y;
                                normal.z += vol_jv * gW.z;
                                neighbor_count += 1.0f;
                            }
                        }
                    }
                }
            }
        }
    }

    float rho = c_precalc.poly6_coeff * sum_density;
    rho = fmaxf(rho, 1.0f);
    density_out[i] = rho;

    // Heat diffusion + exposure + dye output
    if (do_heat) {
        float cp_i = c_materials[mat_id_i].heat_capacity;
        dTdt_out[i] = kappa_i * c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(rho * cp_i, 1.0f);
        exposure_heat_out[i] = c_precalc.poly6_coeff * sum_exposure_heat;
        exposure_corrode_out[i] = c_precalc.poly6_coeff * sum_exposure_corrode;
        if (dye_on) dye_rate_out[i] = make_float4(dye_rate.x, dye_rate.y, dye_rate.z, 0.0f);
    }

    // Vorticity output
    if (do_vort) {
        float omega_mag = sqrtf(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z);
        vorticity_out[i] = make_float4(omega.x, omega.y, omega.z, omega_mag);
    } else if (vorticity_out != 0) {
        vorticity_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Surface normal output (FLUID: n + neighbor_count, others: zero)
    if (normal_out != 0) {
        if (do_normal) {
            normal_out[i] = make_float4(normal.x, normal.y, normal.z, neighbor_count);
        } else {
            normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

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

    // STATIC and GAS: always lambda=0 (Akinci one-way boundary / compressible phase).
    // They may reach here on the do_heat path to accumulate exposure/heat, but must not
    // participate in PBF constraint solving.
    if (behavior_i == STATIC || behavior_i == GAS) {
        density_out[i] = (behavior_i == STATIC) ? c_materials[mat_id_i].rest_density : rho;
        lambda_out[i] = 0.0f;
        pressure_normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Constraint: C_i = rho_i / rho0 - 1
    // Over-dense (C>0): full repulsive correction (incompressibility)
    // Under-dense (C<0): weak attractive correction (5%) for surface cohesion
    float C_raw = rho / rho0 - 1.0f;
    float C_i = (C_raw >= 0.0f) ? C_raw : C_raw * c_pbf.neg_c_scale;

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
    bool is_gas_i = (behavior_i == GAS);

    // GAS skips PBF constraints (compressible phase, not position-corrected).
    // STATIC and sleeping particles also skip.
    // GAS skips PBF constraints (compressible phase). STATIC skips too.
    // Sleeping particles still participate so density corrections can wake them.
    if (behavior_i == STATIC || behavior_i == GAS) {
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

                    int bj_delta = GET_BEHAVIOR(pi_j);

                    // GAS/non-GAS phase separation
                    if (is_gas_i != (bj_delta == GAS)) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    // Artificial pressure (tensile instability fix)
                    // NOTE: Exponent is hardcoded n=4 for performance (avoids powf).
                    // c_pbf.s_corr_n exists but is not used here. At h=0.04, s_corr
                    // is typically disabled (k=0) because Spiky gradients are already
                    // large enough for stability. See PHYSICS.md section 10.6.
                    float W_ij = W_poly6(r_sq, h_sq);
                    float s_corr = 0.0f;
                    if (W_dq > 1e-12f) {
                        float ratio = W_ij / W_dq;
                        float rn = ratio * ratio;  // ratio^2
                        rn = rn * rn;              // ratio^4 (hardcoded n=4)
                        s_corr = -c_pbf.s_corr_k * rn;
                    }

                    float3 gW = grad_spiky(r, rlen, h);

                    // STATIC boundary: stronger one-way repulsion.
                    // Mirror lambda_i (Akinci-style) so boundary pushes fluid out
                    // with strength proportional to the fluid's own compression.
                    float lambda_eff_j = lambda_j;
                    if (bj_delta == STATIC) {
                        lambda_eff_j = lambda_i;  // mirror fluid's own lambda
                    }

                    // Mass-weighted: dp = (m_j/rho0) * (l_i+l_j+s) * gradW
                    float coeff = m_j * (lambda_i + lambda_eff_j + s_corr) / rho0;
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
 * K_PBF_ApplyDelta -- Apply correction + boundary clamp
 * (GRANULAR friction moved to K_PBF_Finalize for iteration-independence)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_PBF_ApplyDelta(
    uint            numParticles,
    float4*         __restrict__ predicted_pos,   // in-out
    const float4*   __restrict__ delta_pos,
    const uint*     __restrict__ packed_info
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    if (GET_BEHAVIOR(pi) == STATIC) return;

    float4 pos4 = predicted_pos[i];
    float4 d4 = delta_pos[i];

    float3 pos = make_float3(pos4.x + d4.x, pos4.y + d4.y, pos4.z + d4.z);

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
    const float4*   __restrict__ sorted_dye_rate,
    float4*         __restrict__ particle_dye_out,
    const float4*   __restrict__ sorted_angular_velocity,
    float4*         __restrict__ angular_velocity_out,
    const float4*   __restrict__ vorticity_in,     // (omega_x,y,z, |omega|) from ComputeLambda, or NULL
    const float4*   __restrict__ normal_in,         // (n_x,n_y,n_z, neighbor_count) from ComputeLambda, or NULL
    const float4*   __restrict__ pressure_normal_in, // density gradient normal for GRANULAR friction, or NULL
    const float*    __restrict__ sorted_lambda_pbf,  // converged lambda for warm-start writeback
    float*          __restrict__ lambda_pbf_out,     // unsorted lambda output (warm-start next frame)
    const RigidBody* __restrict__ d_rigid_bodies,    // rigid body state (NULL if no bodies)
    float*          __restrict__ d_rigid_forces,     // force accumulator (NULL if no bodies)
    float*          __restrict__ d_rigid_torques,    // torque accumulator (NULL if no bodies)
    uint*           __restrict__ max_displacement_out // [1] atomicMax of displacement^2 (float-as-uint), or NULL
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
        if (lambda_pbf_out) lambda_pbf_out[orig_idx] = 0.0f;
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
            if (lambda_pbf_out) lambda_pbf_out[orig_idx] = sorted_lambda_pbf[i];
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

    // NOTE: Thermal convection (Boussinesq buoyancy) is applied ONLY in K_PBF_Predict,
    // not here. PBF derives velocity from corrected positions: v = (x* - x_old)/dt.
    // Adding buoyancy here would double-count it since it's already in predicted positions.

    // GRANULAR: Velocity-space Drucker-Prager friction (applied once, iteration-independent).
    // H3 fix: Apply friction to velocity CORRECTION only (vel_new - v_predicted),
    // making friction frame-independent (operates on constraint corrections only).
    // Uses pressure_normal (density gradient) from ComputeLambda for normal/tangential decomposition.
    // Drucker-Prager criterion: |v_t_corr| <= tan(phi_f) * |v_n_corr| + cohesion / dt
    if (behavior == GRANULAR && pressure_normal_in != 0) {
        // v_predicted = v_old + dt * gravity (from K_PBF_Predict)
        float4 v_old4 = original_vel[i];
        float3 v_pred = make_float3(
            v_old4.x + dt * c_sim.gravity.x,
            v_old4.y + dt * c_sim.gravity.y,
            v_old4.z + dt * c_sim.gravity.z
        );
        // v_corr = vel_new - v_predicted (PBF constraint correction only)
        float3 v_corr = make_float3(
            vel_new.x - v_pred.x,
            vel_new.y - v_pred.y,
            vel_new.z - v_pred.z
        );

        float4 pn4 = __ldg(&pressure_normal_in[i]);
        float3 n = make_float3(pn4.x, pn4.y, pn4.z);
        float n_len_sq = n.x*n.x + n.y*n.y + n.z*n.z;

        if (n_len_sq > 0.5f) {  // valid unit normal
            // Decompose correction velocity along pressure normal
            float vc_dot_n = v_corr.x*n.x + v_corr.y*n.y + v_corr.z*n.z;

            if (vc_dot_n > 0.0f) {
                float3 vc_n = make_float3(vc_dot_n*n.x, vc_dot_n*n.y, vc_dot_n*n.z);
                float3 vc_t = make_float3(v_corr.x - vc_n.x, v_corr.y - vc_n.y, v_corr.z - vc_n.z);

                float tang_sq = vc_t.x*vc_t.x + vc_t.y*vc_t.y + vc_t.z*vc_t.z;

                // Static friction dead zone: below minimum normal velocity, zero tangential
                float max_tang = (vc_dot_n < 5e-4f) ? 0.0f
                               : c_granular.tan_phi_f * vc_dot_n + c_granular.cohesion * inv_dt;

                if (tang_sq > max_tang * max_tang) {
                    if (max_tang > 0.0f && tang_sq > 1e-12f) {
                        float scale = max_tang / sqrtf(tang_sq);
                        vc_t.x *= scale;
                        vc_t.y *= scale;
                        vc_t.z *= scale;
                    } else {
                        vc_t.x = 0.0f;
                        vc_t.y = 0.0f;
                        vc_t.z = 0.0f;
                    }
                    // Reconstruct vel_new = v_predicted + corrected v_corr
                    v_corr.x = vc_n.x + vc_t.x;
                    v_corr.y = vc_n.y + vc_t.y;
                    v_corr.z = vc_n.z + vc_t.z;
                    vel_new.x = v_pred.x + v_corr.x;
                    vel_new.y = v_pred.y + v_corr.y;
                    vel_new.z = v_pred.z + v_corr.z;
                }
            }
        }
    }

    // XSPH viscosity + vorticity eta (folded -- PERF-002)
    // Vorticity eta setup
    bool do_vort_eta = (behavior == FLUID && vorticity_in != 0 && c_granular.vorticity_epsilon > 0.0f);
    float4 vort_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float omega_mag_i = 0.0f;
    float3 eta_vort = make_float3(0.0f, 0.0f, 0.0f);
    if (do_vort_eta) {
        vort_i = __ldg(&vorticity_in[i]);
        omega_mag_i = vort_i.w;
        do_vort_eta = (omega_mag_i > 1e-6f);
    }

    if (behavior == FLUID || behavior == GRANULAR || do_vort_eta) {
        float h = c_sim.smoothing_length;
        float h_sq = c_sim.smoothing_length_sq;
        float rho_i = density[i];
        float3 xsph = make_float3(0.0f, 0.0f, 0.0f);
        bool do_xsph = (behavior == FLUID || behavior == GRANULAR);

        // Akinci pressure mirroring for rigid body force accumulation
        float p_i = compute_pressure(rho_i, behavior, mat_id);
        float rho0_i = c_materials[mat_id].rest_density;

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

                        float4 pred4_j = __ldg(&predicted_pos[j]);
                        float3 r = make_float3(
                            pred.x - pred4_j.x,
                            pred.y - pred4_j.y,
                            pred.z - pred4_j.z
                        );
                        float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                        if (r_sq > h_sq || r_sq < 1e-12f) continue;

                        // Vorticity eta (all neighbors -- PERF-002)
                        if (do_vort_eta) {
                            float rlv = sqrtf(r_sq);
                            float h_rl = h - rlv;
                            float inv_rl = 1.0f / rlv;
                            float gs = c_precalc.spiky_grad_coeff * h_rl * h_rl * inv_rl;
                            float omega_j = __ldg(&vorticity_in[j]).w;
                            float mj_v = __ldg(&mass[j]);
                            float rj_v = __ldg(&density[j]);
                            float wt = mj_v / fmaxf(rj_v, 1.0f);
                            eta_vort.x += wt * omega_j * gs * r.x;
                            eta_vort.y += wt * omega_j * gs * r.y;
                            eta_vort.z += wt * omega_j * gs * r.z;
                        }

                        if (!do_xsph) continue;

                        uint pi_j = __ldg(&packed_info[j]);
                        int bj = GET_BEHAVIOR(pi_j);
                        if (bj == GAS) continue;  // no XSPH across phases

                        float m_j = __ldg(&mass[j]);

                        // Akinci pressure force on rigid body (two-way coupling)
                        if (bj == STATIC && d_rigid_bodies != 0) {
                            uint mat_id_j = GET_MATERIAL_ID(pi_j);
                            if (mat_id_j == MAT_RIGID) {
                                float rlen = sqrtf(r_sq);
                                float3 gW = grad_spiky(r, rlen, h);
                                float psi_b = m_j;
                                float press_akinci = (p_i / (rho_i * rho_i)) + (p_i / (rho0_i * rho0_i));
                                float m_i_val = c_sim.particle_mass;
                                float3 F_on_fluid = make_float3(
                                    m_i_val * psi_b * press_akinci * gW.x,
                                    m_i_val * psi_b * press_akinci * gW.y,
                                    m_i_val * psi_b * press_akinci * gW.z
                                );
                                int body_id = GET_BODY_ID(pi_j);
                                float4 rb_pos = __ldg(&d_rigid_bodies[body_id].position);
                                float3 r_b = make_float3(
                                    pred4_j.x - rb_pos.x, pred4_j.y - rb_pos.y, pred4_j.z - rb_pos.z
                                );
                                float3 F_on_body = make_float3(-F_on_fluid.x, -F_on_fluid.y, -F_on_fluid.z);
                                float3 tau = make_float3(
                                    r_b.y * F_on_body.z - r_b.z * F_on_body.y,
                                    r_b.z * F_on_body.x - r_b.x * F_on_body.z,
                                    r_b.x * F_on_body.y - r_b.y * F_on_body.x
                                );
                                warp_reduce_accumulate(d_rigid_forces, F_on_body, body_id);
                                warp_reduce_accumulate(d_rigid_torques, tau, body_id);
                                continue;  // skip XSPH for boundary particles
                            }
                        }

                        float4 orig4_j = __ldg(&original_pos[j]);
                        float3 vel_j = make_float3(
                            (pred4_j.x - orig4_j.x) * inv_dt,
                            (pred4_j.y - orig4_j.y) * inv_dt,
                            (pred4_j.z - orig4_j.z) * inv_dt
                        );
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

        if (do_xsph) {
            float c_xsph = c_pbf.xsph_c;
            c_xsph = fminf(c_xsph * c_materials[mat_id].base_viscosity, 0.5f);
            if (behavior == GRANULAR) c_xsph = fminf(c_xsph * 3.0f, 0.15f);
            vel_new.x += c_xsph * xsph.x;
            vel_new.y += c_xsph * xsph.y;
            vel_new.z += c_xsph * xsph.z;
        }
    }

    // Vorticity confinement force application (FLUID only, eta from main loop -- PERF-002)
    if (do_vort_eta) {
        float eta_mag = sqrtf(eta_vort.x*eta_vort.x + eta_vort.y*eta_vort.y + eta_vort.z*eta_vort.z);
        if (eta_mag > 1e-6f) {
            float inv_eta = 1.0f / eta_mag;
            float3 N = make_float3(eta_vort.x*inv_eta, eta_vort.y*inv_eta, eta_vort.z*inv_eta);
            float eps_v = c_granular.vorticity_epsilon;
            vel_new.x += dt * eps_v * (N.y * vort_i.z - N.z * vort_i.y);
            vel_new.y += dt * eps_v * (N.z * vort_i.x - N.x * vort_i.z);
            vel_new.z += dt * eps_v * (N.x * vort_i.y - N.y * vort_i.x);
        }
    }

    // Akinci surface tension (FLUID only, surface particles): vel += dt * (-gamma * n)
    if (behavior == FLUID && normal_in != 0 && c_granular.surface_tension_gamma > 0.0f) {
        float4 norm_i = __ldg(&normal_in[i]);
        float nc_i = norm_i.w;  // neighbor count
        if (nc_i < 25.0f) {
            float gamma = c_granular.surface_tension_gamma;
            float n_mag = sqrtf(norm_i.x*norm_i.x + norm_i.y*norm_i.y + norm_i.z*norm_i.z);
            if (n_mag > 0.01f) {
                vel_new.x += dt * (-gamma * norm_i.x);
                vel_new.y += dt * (-gamma * norm_i.y);
                vel_new.z += dt * (-gamma * norm_i.z);
            }
        }
    }

    // GRANULAR: friction velocity damping when in dense packing.
    // Only apply when the particle has neighbors (rho > 0.7*rho0), so free-falling
    // sand isn't artificially slowed. Damps residual spreading from PBF corrections.
    if (behavior == GRANULAR) {
        float rho_i_damp = density[i];
        float rho0_damp = c_materials[mat_id].rest_density;
        if (rho_i_damp > 0.7f * rho0_damp) {
            float granular_damp = 1.0f - 2.0f * dt;
            granular_damp = fmaxf(granular_damp, 0.8f);
            vel_new.x *= granular_damp;
            vel_new.y *= granular_damp;
            vel_new.z *= granular_damp;
        }
    }

    // STATIC particle boundary repulsion (shared function in sph_shared.cuh)
    float3 final_pos = pred;
    static_particle_boundary(
        final_pos, vel_new,
        cell_start, cell_end,
        packed_info, predicted_pos,
        i, c_sim.restitution
    );

    // World box boundary collision
    float friction = (behavior == FLUID) ? 0.0f : c_sim.wall_friction;
    sdf_box_boundary(final_pos, vel_new, c_sim.world_min, c_sim.world_max,
                     c_sim.restitution, friction);

    // SDF object collision with velocity reflection
    for (int s = 0; s < c_num_sdf_objects; s++) {
        SDFObject obj = c_sdf_objects[s];
        float sdf_dist = eval_sdf(final_pos, obj);
        if (sdf_dist < BOUNDARY_MARGIN) {
            float3 n = sdf_normal(final_pos, obj);
            float push = BOUNDARY_MARGIN - sdf_dist;
            final_pos.x += push * n.x;
            final_pos.y += push * n.y;
            final_pos.z += push * n.z;

            // Object velocity at contact point
            float3 obj_vel = make_float3(obj.velocity.x, obj.velocity.y, obj.velocity.z);
            float ang_speed = obj.velocity.w;
            if (fabsf(ang_speed) > 1e-8f) {
                float3 axis = make_float3(obj.angular_axis.x, obj.angular_axis.y, obj.angular_axis.z);
                float3 center = make_float3(obj.pos_and_type.x, obj.pos_and_type.y, obj.pos_and_type.z);
                float3 r = make_float3(final_pos.x - center.x, final_pos.y - center.y, final_pos.z - center.z);
                float3 omega = make_float3(axis.x * ang_speed, axis.y * ang_speed, axis.z * ang_speed);
                obj_vel.x += omega.y * r.z - omega.z * r.y;
                obj_vel.y += omega.z * r.x - omega.x * r.z;
                obj_vel.z += omega.x * r.y - omega.y * r.x;
            }

            float3 v_rel = make_float3(vel_new.x - obj_vel.x, vel_new.y - obj_vel.y, vel_new.z - obj_vel.z);
            float v_dot_n = v_rel.x * n.x + v_rel.y * n.y + v_rel.z * n.z;
            if (v_dot_n < 0.0f) {
                float obj_restitution = obj.size_and_r.w;
                float obj_friction = obj.angular_axis.w;
                vel_new.x -= (1.0f + obj_restitution) * v_dot_n * n.x;
                vel_new.y -= (1.0f + obj_restitution) * v_dot_n * n.y;
                vel_new.z -= (1.0f + obj_restitution) * v_dot_n * n.z;
                float3 v_tan = make_float3(v_rel.x - v_dot_n*n.x, v_rel.y - v_dot_n*n.y, v_rel.z - v_dot_n*n.z);
                float v_tan_len = sqrtf(v_tan.x*v_tan.x + v_tan.y*v_tan.y + v_tan.z*v_tan.z);
                if (v_tan_len > 1e-8f) {
                    float red = fminf(obj_friction * fabsf(v_dot_n) / v_tan_len, 1.0f);
                    vel_new.x -= red * v_tan.x;
                    vel_new.y -= red * v_tan.y;
                    vel_new.z -= red * v_tan.z;
                }
            }
        }
    }

    // Velocity clamp
    float vel_sq = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
    if (vel_sq > VELOCITY_LIMIT_SQ) {
        float scale = VELOCITY_LIMIT / sqrtf(vel_sq);
        vel_new.x *= scale;
        vel_new.y *= scale;
        vel_new.z *= scale;
        vel_sq = VELOCITY_LIMIT_SQ;
    }

    // Spawn velocity damping (ramps to 0 after first ~30 substeps)
    if (c_sim.velocity_damping > 0.0f) {
        float damp = 1.0f - c_sim.velocity_damping;
        vel_new.x *= damp;
        vel_new.y *= damp;
        vel_new.z *= damp;
        vel_sq = vel_new.x * vel_new.x + vel_new.y * vel_new.y + vel_new.z * vel_new.z;
    }

    // GRANULAR anti-creep: zero velocity when nearly at rest and well-packed.
    // Higher threshold than WCSPH (0.05 vs 0.01) because PBF position corrections
    // generate artificial spreading velocity that must be caught here.
    if (behavior == GRANULAR) {
        vel_sq = vel_new.x*vel_new.x + vel_new.y*vel_new.y + vel_new.z*vel_new.z;
        if (vel_sq < 0.01f * 0.01f) {
            float rho_i = density[i];
            float rho0_i = c_materials[mat_id].rest_density;
            if (rho_i > 0.98f * rho0_i) {
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

    // Dye update: dye += dye_rate * dt
    float4 dye = sorted_particle_dye[i];
    if (sorted_dye_rate != 0) {
        float4 drate = sorted_dye_rate[i];
        dye.x = fmaxf(0.0f, fminf(1.0f, dye.x + drate.x * dt));
        dye.y = fmaxf(0.0f, fminf(1.0f, dye.y + drate.y * dt));
        dye.z = fmaxf(0.0f, fminf(1.0f, dye.z + drate.z * dt));
    }

    // Track max displacement for grid reuse (Phase 9.2 / PERF-009)
    // displacement^2 = |final_pos - orig_pos|^2
    if (max_displacement_out) {
        float dx_d = final_pos.x - orig.x;
        float dy_d = final_pos.y - orig.y;
        float dz_d = final_pos.z - orig.z;
        float disp_sq = dx_d * dx_d + dy_d * dy_d + dz_d * dz_d;
        atomicMax(max_displacement_out, __float_as_uint(disp_sq));
    }

    // Writeback (unsorted)
    position_out[orig_idx] = make_float4(final_pos.x, final_pos.y, final_pos.z, 1.0f);
    velocity_out[orig_idx] = make_float4(vel_new.x, vel_new.y, vel_new.z, 0.0f);
    color_out[orig_idx] = color;
    packed_info_out[orig_idx] = pi;
    sleep_counter_out[orig_idx] = sc;
    temperature_out[orig_idx] = temp;
    if (lambda_pbf_out) lambda_pbf_out[orig_idx] = sorted_lambda_pbf[i];
    particle_dye_out[orig_idx] = dye;
    // Micropolar angular velocity: relax toward 0.5 * curl_v (FLUID only)
    float4 ang_vel = sorted_angular_velocity[i];
    if (behavior == FLUID && vorticity_in != 0) {
        float4 vort = __ldg(&vorticity_in[i]);
        const float nu_t = 0.1f;
        ang_vel.x += dt * nu_t * (0.5f * vort.x - ang_vel.x);
        ang_vel.y += dt * nu_t * (0.5f * vort.y - ang_vel.y);
        ang_vel.z += dt * nu_t * (0.5f * vort.z - ang_vel.z);
    }
    angular_velocity_out[orig_idx] = ang_vel;
}
