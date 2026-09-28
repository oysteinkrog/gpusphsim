/*
 * dfsph_solver.cu -- Divergence-Free SPH (Bender & Koschier, SCA 2015)
 *
 * Kernels:
 *   K_DFSPH_ComputeDensityAlpha -- Density + alpha precompute + shear_rate + heat
 *   K_DFSPH_PackVolume          -- Volume carrier into position.w for the solver loops
 *   K_DFSPH_NonPressureForces   -- Viscosity + gravity -> velocity update
 *   K_DFSPH_ComputeKappaV       -- Divergence correction factor
 *   K_DFSPH_CorrectVelocityDiv  -- Apply divergence correction
 *   K_DFSPH_ComputePressureAccel  -- a_press from p/rho^2, writes vt = v + dt * a_press
 *   K_DFSPH_DensitySolverUpdate   -- Predict density from vt, Jacobi update of p/rho^2
 *   K_DFSPH_Finalize              -- Final position, boundary, writeback
 *
 * The divergence and density iteration loops run on the host
 * (simulation.py); the iteration counts are not in constant memory.
 *
 * Constant memory:
 *   c_grid, c_sim, c_precalc, c_materials -- from common.cuh
 *   c_dfsph -- local DFSPH parameters
 */

#include "sph_shared.cuh"

/* Volume divisor for neighbour j in the DFSPH solver terms (alpha, div_v,
 * drho, XSPH). A regular STATIC wall particle takes the Akinci boundary
 * volume with the fluid's own rest density, V_j = m_j / rho0_i, so a STONE,
 * METAL or WOOD wall pushes the same (bd-r5-review-n28.116). Its stored
 * density is its rest_density, which would make V_j depend on the wall
 * material. MAT_RIGID keeps density RIGID_PSI_RHO0, so m_j / rho_j = V_b. */
__device__ __forceinline__ bool dfsph_is_wall(uint pi) {
    return GET_BEHAVIOR(pi) == STATIC && GET_MATERIAL_ID(pi) != MAT_RIGID;
}

__device__ __forceinline__ float dfsph_vol_rho(float rho_j, uint pi_j, float rho0_i) {
    return dfsph_is_wall(pi_j) ? rho0_i : rho_j;
}

/* Volume carrier in sorted_position.w (bd-r5-review-n28.184).
 * K_DFSPH_PackVolume writes it once per substep, after reactions and spawn,
 * so the four solver loops (ComputeKappaV, CorrectVelocityDiv,
 * ComputePressureAccel, DensitySolverUpdate) read V_j from the float4 they
 * already load for the position, instead of loading mass and density.
 *   regular STATIC wall: .w = m_j (the loops divide by the right density)
 *   everything else:     .w = m_j / max(rho_j, RHO_EPSILON)
 * A wall stores its mass because the two loop families use different
 * divisors for it: rho0_i (Akinci, dfsph_vol_rho) and its own density. */
__device__ __forceinline__ float dfsph_volume_carrier(float m, float rho, uint pi) {
    return dfsph_is_wall(pi) ? m : m / fmaxf(rho, RHO_EPSILON);
}

/* Akinci volume m_j / max(dfsph_vol_rho(rho_j, pi_j, rho0_i), eps) from the
 * carrier: the divergence and density-prediction sums. */
__device__ __forceinline__ float dfsph_vol_akinci(float w_j, uint pi_j, float rho0_i) {
    return dfsph_is_wall(pi_j) ? w_j / fmaxf(rho0_i, RHO_EPSILON) : w_j;
}

/* Plain volume m_j / max(rho_j, eps) from the carrier: the pressure and
 * divergence-correction sums. Only a wall neighbour loads its density. */
__device__ __forceinline__ float dfsph_vol_plain(float w_j, uint pi_j,
                                                 const float* __restrict__ density, uint j) {
    return dfsph_is_wall(pi_j) ? w_j / fmaxf(__ldg(&density[j]), RHO_EPSILON) : w_j;
}

/* Add F_on_body and its torque about the body centre to the accumulators of
 * the body that owns boundary particle j. */
__device__ inline void dfsph_rigid_reaction(
    const RigidBody* __restrict__ d_rigid_bodies,
    float* __restrict__ d_rigid_forces,
    float* __restrict__ d_rigid_torques,
    uint pi_j, float4 pos4_j, float3 F_on_body
) {
    int body_id = GET_BODY_ID(pi_j);
    float4 rb_pos = __ldg(&d_rigid_bodies[body_id].position);
    float3 r_b = make_float3(pos4_j.x - rb_pos.x, pos4_j.y - rb_pos.y, pos4_j.z - rb_pos.z);
    float3 tau = make_float3(
        r_b.y * F_on_body.z - r_b.z * F_on_body.y,
        r_b.z * F_on_body.x - r_b.x * F_on_body.z,
        r_b.x * F_on_body.y - r_b.y * F_on_body.x
    );
    atomic_accumulate3(d_rigid_forces, F_on_body, body_id);
    atomic_accumulate3(d_rigid_torques, tau, body_id);
}

/* ======================================================================
 * DFSPH Parameters
 * ====================================================================== */

struct DFSPHParams {
    float _pad_iters[2];  // unused (iteration counts live on the host)
    float warm_start;
    float omega;          // under-relaxation factor for corrections (0.3-1.0)
    float alpha_limit;    // alpha cap in units of h^2 (interior alpha ~ 0.67 h^2); safety bound only
    // Warm-start scaling folded into the K_DFSPH_Finalize writeback
    // (bd-r5-review-n28.173). Finalize writes kappa * kappa_carry and
    // kappa_v * kappa_v_carry, so the next substep reads them already scaled.
    float kappa_carry;    // warm_start, or 1 when warm_start <= 0 (the host zeroes p_rho2 then)
    float kappa_v_carry;  // div warm start, or 0 when it is <= 0
    float kappa_uncarry;  // 1 / kappa_carry: recovers the unscaled p/rho for mu(I)
};

__constant__ DFSPHParams c_dfsph;

/* DFSPH-specific constants */
#define ACCEL_MAX          30.0f
#define ACCEL_MAX_SQ       (ACCEL_MAX * ACCEL_MAX)

/* ======================================================================
 * K_DFSPH_ComputeDensityAlpha
 * Density + alpha factor + shear_rate + heat diffusion + exposure
 * (Modified Step1 that also computes the DFSPH alpha diagonal factor)
 *
 * alpha_i = 1 / (|SUM_j (m_j/rho_j) gradW_ij|^2
 *              + SUM_{j not STATIC} (m_j/rho_j)^2 |gradW_ij|^2 + eps)
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 3)
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
    float*          __restrict__ exposure_corrode_out,
    float4*         __restrict__ vorticity_out,      // (curl_v.x, .y, .z, |curl_v|), NULL to skip
    float4*         __restrict__ normal_out          // (n_x, n_y, n_z, neighbor_count), NULL to skip
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
    bool is_gas_i = (behavior_i == GAS);

    bool is_fluid = (behavior_i == FLUID);

    // Rigid neighbour density scale: psi_b = rho0_i * V_b (bd-r5-review-n28.77)
    float rigid_dens_scale = __fdiv_rn(c_materials[mat_id_i].rest_density, RIGID_PSI_RHO0);
    // Akinci wall volume divisor (dfsph_vol_rho); floor at 1 for DEAD
    float rho0_vol_i = fmaxf(c_materials[mat_id_i].rest_density, 1.0f);

    // MAT_RIGID boundary particles: early-return, no density/alpha needed
    if (mat_id_i == MAT_RIGID) {
        density_out[i] = RIGID_PSI_RHO0;
        alpha_out[i] = 0.0f;
        shear_rate_out[i] = 0.0f;
        dTdt_out[i] = 0.0f;
        exposure_heat_out[i] = 0.0f;
        exposure_corrode_out[i] = 0.0f;
        if (vorticity_out) vorticity_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (normal_out) normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    bool need_vel = is_granular || (is_fluid && vorticity_out != 0);
    float3 vel_i = make_float3(0.0f, 0.0f, 0.0f);
    if (need_vel) {
        float4 v4 = velocity[i];
        vel_i = make_float3(v4.x, v4.y, v4.z);
    }

    float T_i = __ldg(&temperature_in[i]);
    float kappa_i = c_materials[mat_id_i].thermal_conductivity;
    // Cross-phase heat cap (bd-r5-review-n28.94). The self term bounds rho_i from
    // below, so capping a cross-phase pair coefficient at a_x_max keeps
    // viscosity_lap_coeff * a / (rho_i * cp_i) * dt <= 0.5 for that pair.
    float rho_i_lb = fmaxf(c_precalc.poly6_coeff * __ldg(&mass[i]) * h_sq * h_sq * h_sq,
                           is_gas_i ? RHO_EPSILON : 1.0f);
    float a_x_max = 0.5f * rho_i_lb * c_materials[mat_id_i].heat_capacity
                  / fmaxf(c_precalc.viscosity_lap_coeff * c_sim.dt, 1e-30f);

    float sum_density = 0.0f;
    float sum_dTdt = 0.0f;
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;
    // Kernel support weight of all non-self, non-DEAD neighbours (GAS and
    // MAT_RIGID included). Exposure sums are divided by it, as in K_Step1.
    float sum_W = 0.0f;

    // Strain-rate tensor (GRANULAR only)
    float Dxx = 0.0f, Dyy = 0.0f, Dzz = 0.0f;
    float Dxy = 0.0f, Dxz = 0.0f, Dyz = 0.0f;

    // Vorticity accumulator (FLUID only)
    float3 omega = make_float3(0.0f, 0.0f, 0.0f);

    // Surface normal accumulator (FLUID only): n = grad(color field)
    float3 normal = make_float3(0.0f, 0.0f, 0.0f);
    float neighbor_count = 0.0f;

    // Alpha accumulators
    float3 grad_sum = make_float3(0.0f, 0.0f, 0.0f);  // SUM_j (m_j/rho_j) gradW_ij
    float grad_norm_sum = 0.0f;                          // SUM_{j not STATIC} (m_j/rho_j)^2 |gradW_ij|^2

    int3 cell_i = get_cell(pos_i);

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
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

                    // GAS/non-GAS phase separation: skip density/heat/forces,
                    // keep only exposure (so fire-to-wood ignition still works)
                    bool is_gas_j = (behavior_j == GAS);
                    if (is_gas_i != is_gas_j && j != i) {
                        float diff_xp = h_sq - r_sq;
                        float T_j = __ldg(&temperature_in[j]);
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        float w_poly6_var = diff_xp * diff_xp * diff_xp;
                        if (mat_id_j != MAT_DEAD) sum_W += w_poly6_var;
                        sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                        sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
                        // A rigid wall never exchanges heat (same rule as the same-phase branch)
                        if (mat_id_j != MAT_RIGID) {
                            // Cross-phase heat conduction (bd-r5-review-n28.94): the same symmetric
                            // pair term as the same-phase branch, with the coefficient capped at a_x_max
                            // so one neighbour moves T_i by at most half the gap per substep.
                            float rho_x = (density_in != 0) ? __ldg(&density_in[j]) : 0.0f;
                            // Unset or zero density_in (spawn, unused slot): use the neighbour's rest density (bd-r5-review-n28.110)
                            if (rho_x <= 0.0f) rho_x = fmaxf(c_materials[GET_MATERIAL_ID(pi_j)].rest_density, RHO_EPSILON);
                            float kappa_x = c_materials[mat_id_j].thermal_conductivity;
                            float k_x = 2.0f * kappa_i * kappa_x / fmaxf(kappa_i + kappa_x, 1e-6f);
                            float boost_x = fmaxf(1.0f, c_interactions[mat_id_i][mat_id_j].heat_exchange);
                            float a_x = k_x * m_j / fmaxf(rho_x, RHO_EPSILON) * (h - sqrtf(r_sq)) * boost_x;
                            sum_dTdt += fminf(a_x, a_x_max) * (T_j - T_i);
                        }
                        continue;
                    }

                    // Density (Poly6). MAT_RIGID mass is RIGID_PSI_RHO0 * V_b;
                    // Akinci weights it with this particle's rest density
                    // (bd-r5-review-n28.77). Volume terms below keep m_j / rho_j = V_b.
                    float diff = h_sq - r_sq;
                    float m_dens = (GET_MATERIAL_ID(pi_j) == MAT_RIGID) ? m_j * rigid_dens_scale : m_j;
                    sum_density += m_dens * diff * diff * diff;

                    // Exposure support weight: every non-self, non-DEAD neighbour,
                    // MAT_RIGID included, so a wall dilutes exposure (as in K_Step1).
                    if (j != i && GET_MATERIAL_ID(pi_j) != MAT_DEAD) sum_W += diff * diff * diff;

                    // Skip self for non-density quantities
                    if (j != i && r_sq > 1e-12f) {
                        float rlen = sqrtf(r_sq);
                        float rho_j = (density_in != 0) ? __ldg(&density_in[j]) : 0.0f;
                        // Unset or zero density_in (spawn, unused slot): use the neighbour's rest density (bd-r5-review-n28.110)
                        if (rho_j <= 0.0f) rho_j = fmaxf(c_materials[GET_MATERIAL_ID(pi_j)].rest_density, RHO_EPSILON);

                        // Skip MAT_RIGID for heat/exposure/vorticity/strain
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        bool is_rigid_j = (mat_id_j == MAT_RIGID);

                        if (!is_rigid_j) {
                            // Heat diffusion (with cross-material boost)
                            float T_j = __ldg(&temperature_in[j]);
                            float lap_var = h - rlen;
                            float heat_boost = fmaxf(1.0f, c_interactions[mat_id_i][mat_id_j].heat_exchange);
                            // Pairwise harmonic-mean conductivity: the pair term is symmetric,
                            // so heat lost by i equals heat gained by j (bd-r5-review-n28.67).
                            float kappa_j = c_materials[mat_id_j].thermal_conductivity;
                            float k_ij = 2.0f * kappa_i * kappa_j / fmaxf(kappa_i + kappa_j, 1e-6f);
                            sum_dTdt += k_ij * m_j / fmaxf(rho_j, RHO_EPSILON) * (T_j - T_i) * lap_var * heat_boost;

                            // Exposure
                            float w_poly6_var = diff * diff * diff;
                            sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                            sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
                        }

                        // Alpha factor (using Spiky gradient). STATIC neighbours
                        // (STONE, MAT_RIGID, ...) enter only grad_sum: the squared
                        // term is neighbour j's own response to kappa_i, and a
                        // boundary particle never moves under pressure
                        // (SPlisHSPlasH computeDFSPHFactor, bd-r5-review-n28.108).
                        float3 gW = grad_spiky(r, rlen, h);
                        float w_j = m_j / fmaxf(rho_j, RHO_EPSILON);
                        float w_a = m_j / fmaxf(dfsph_vol_rho(rho_j, pi_j, rho0_vol_i), RHO_EPSILON);
                        grad_sum.x += w_a * gW.x;
                        grad_sum.y += w_a * gW.y;
                        grad_sum.z += w_a * gW.z;
                        if (behavior_j != STATIC)
                            grad_norm_sum += w_j * w_j * (gW.x * gW.x + gW.y * gW.y + gW.z * gW.z);

                        // Surface normal and neighbour count include MAT_RIGID
                        // (bd-r5-review-n28.68); vorticity and strain-rate skip it.
                        if (is_rigid_j) {
                            if (m_j > 0.0f) {
                                if (is_fluid && normal_out != 0) {
                                    // The mass slot holds RIGID_PSI_RHO0 * V_b (bd-r5-review-n28.77)
                                    float vol_b = m_j / RIGID_PSI_RHO0;
                                    normal.x += vol_b * gW.x;
                                    normal.y += vol_b * gW.y;
                                    normal.z += vol_b * gW.z;
                                }
                                neighbor_count += 1.0f;
                            }
                        } else {
                            if (is_fluid && normal_out != 0) {
                                normal.x += w_j * gW.x;
                                normal.y += w_j * gW.y;
                                normal.z += w_j * gW.z;
                            }
                            if (m_j > 0.0f) neighbor_count += 1.0f;  // DEAD slots have mass 0

                            if (need_vel) {
                                float4 vel4_j = __ldg(&velocity[j]);
                                float dvx = vel_i.x - vel4_j.x;
                                float dvy = vel_i.y - vel4_j.y;
                                float dvz = vel_i.z - vel4_j.z;

                                if (is_granular) {
                                    float grad_scalar = c_precalc.spiky_grad_coeff * (h - rlen) * (h - rlen) / rlen;
                                    float gWx = grad_scalar * r.x;
                                    float gWy = grad_scalar * r.y;
                                    float gWz = grad_scalar * r.z;
                                    float weight = m_j / fmaxf(rho_j, RHO_EPSILON);
                                    Dxx += weight * dvx * gWx;
                                    Dyy += weight * dvy * gWy;
                                    Dzz += weight * dvz * gWz;
                                    Dxy += 0.5f * weight * (dvx * gWy + dvy * gWx);
                                    Dxz += 0.5f * weight * (dvx * gWz + dvz * gWx);
                                    Dyz += 0.5f * weight * (dvy * gWz + dvz * gWy);
                                }

                                if (is_fluid) {
                                    float vol_j = m_j / fmaxf(rho_j, RHO_EPSILON);
                                    // dv = v_i - v_j, so gradW x (v_j - v_i) = dv x gradW
                                    omega.x += vol_j * (dvy * gW.z - dvz * gW.y);
                                    omega.y += vol_j * (dvz * gW.x - dvx * gW.z);
                                    omega.z += vol_j * (dvx * gW.y - dvy * gW.x);
                                }
                            }
                        } // !is_rigid_j
                    }
                }
            }
        }
    }

    // Density (behavior-aware floor: GAS rho0 ~ 0.2-0.6)
    float rho = c_precalc.poly6_coeff * sum_density;
    rho = fmaxf(rho, is_gas_i ? RHO_EPSILON : 1.0f);

    // H1 fix: STATIC particles use rest_density to prevent gaps at boundaries
    if (behavior_i == STATIC) {
        // Floor at 1: DEAD has rest_density 0, and V_j = m_j / rho_j must stay finite.
        density_out[i] = fmaxf(c_materials[mat_id_i].rest_density, 1.0f);
        alpha_out[i] = 0.0f;
        shear_rate_out[i] = 0.0f;
        // Still write heat/exposure outputs (computed above). Divide by the
        // stored density (rest_density), as K_Step1 and K_PBF_ComputeLambda
        // do: neighbours weight this wall by m / density_out, and the same
        // value in the divisor keeps the heat pair term conservative
        // (bd-r5-review-n28.116).
        float cp_i_s = c_materials[mat_id_i].heat_capacity;
        dTdt_out[i] = c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(density_out[i] * cp_i_s, RHO_EPSILON);
        float inv_sum_W_s = (sum_W > 0.0f) ? 1.0f / sum_W : 0.0f;
        exposure_heat_out[i] = sum_exposure_heat * inv_sum_W_s;
        exposure_corrode_out[i] = sum_exposure_corrode * inv_sum_W_s;
        if (vorticity_out != 0) vorticity_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (normal_out != 0) normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, neighbor_count);
        return;
    }

    density_out[i] = rho;

    // Alpha = 1 / denom (inverse diagonal of pressure Poisson system).
    // The cap is a loose safety bound for particles with very few neighbours
    // (small denom, huge alpha). It must never bind in the interior, where
    // alpha ~ 1.07e-3 = 0.67 h^2 (denom ~ 930 at h = 0.04, spacing 0.02).
    // alpha_cap = alpha_limit * h^2, independent of dt: the Jacobi displacement
    // per iteration is residual * alpha * |gradW|, which does not depend on dt.
    // Default alpha_limit = 1.0 puts the cap at about 1.5x the interior alpha.
    // A cap far above that (3+) lets few-neighbour surface particles take
    // full-size Jacobi steps and the free surface boils (bd-r5-review-n28.37).
    float denom = grad_sum.x * grad_sum.x + grad_sum.y * grad_sum.y + grad_sum.z * grad_sum.z
                + grad_norm_sum;
    float alpha_threshold = 1e-6f;
    float alpha_cap = c_dfsph.alpha_limit * c_sim.smoothing_length_sq;
    float alpha_raw = (denom > alpha_threshold) ? fminf(1.0f / denom, alpha_cap) : 0.0f;
    if (is_granular) {
        // Compliance factor 0.7: reduce alpha to allow some compression in granular piles.
        // Must multiply alpha (not denom) so that alpha gets SMALLER (more compliant).
        alpha_out[i] = alpha_raw * 0.7f;
    } else {
        alpha_out[i] = alpha_raw;
    }
    // Heat diffusion: dTdt = viscosity_lap_coeff * sum_dTdt / (rho * cp); k_ij is inside sum_dTdt
    float cp_i = c_materials[mat_id_i].heat_capacity;
    dTdt_out[i] = c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(rho * cp_i, RHO_EPSILON);

    // Exposure: support fraction, sum(rate*W) / sum(W) (same as K_Step1)
    float inv_sum_W = (sum_W > 0.0f) ? 1.0f / sum_W : 0.0f;
    exposure_heat_out[i] = sum_exposure_heat * inv_sum_W;
    exposure_corrode_out[i] = sum_exposure_corrode * inv_sum_W;

    // Shear rate
    if (is_granular) {
        float D_sq = Dxx * Dxx + Dyy * Dyy + Dzz * Dzz
                   + 2.0f * (Dxy * Dxy + Dxz * Dxz + Dyz * Dyz);
        shear_rate_out[i] = sqrtf(fmaxf(2.0f * D_sq, 0.0f));
    } else {
        shear_rate_out[i] = 0.0f;
    }

    // Vorticity output
    if (vorticity_out != 0) {
        float omega_mag = sqrtf(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z);
        vorticity_out[i] = make_float4(omega.x, omega.y, omega.z, omega_mag);
    }

    // Surface normal output (FLUID: n + neighbor_count, others: zero)
    if (normal_out != 0) {
        if (is_fluid) {
            normal_out[i] = make_float4(normal.x, normal.y, normal.z, neighbor_count);
        } else {
            normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, neighbor_count);
        }
    }
}

/* ======================================================================
 * K_DFSPH_PackVolume -- write the volume carrier into position.w
 * (see dfsph_volume_carrier). Runs after reactions and spawn, which change
 * mass and material, and before the first solver loop.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256)
void K_DFSPH_PackVolume(
    uint            numParticles,
    float4*         __restrict__ position,
    const float*    __restrict__ density,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    position[i].w = dfsph_volume_carrier(mass[i], density[i], packed_info[i]);
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
    float4*         __restrict__ velocity_out,    // updated velocity
    const float4*   __restrict__ vorticity_in,    // (omega_x,y,z, |omega|) from DensityAlpha, or NULL
    const float4*   __restrict__ normal_in,        // (n_x,n_y,n_z, neighbor_count) from DensityAlpha, or NULL
    const RigidBody* __restrict__ d_rigid_bodies,  // rigid body state (NULL if no bodies)
    float*          __restrict__ d_rigid_forces,   // force accumulator (NULL if no bodies)
    float*          __restrict__ d_rigid_torques,  // torque accumulator (NULL if no bodies)
    const float*    __restrict__ p_prev            // previous substep's p/rho times kappa_carry (sorted_kappa), or NULL
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
    bool is_gas_i = (behavior_i == GAS);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);

    // mu(I) rheology for GRANULAR particles (bd-r5-review-n28.92).
    // The confining pressure is the density solver's own pressure from the
    // previous substep, not Tait EOS: the solver drives rho to rho0, so Tait
    // would measure solver error instead of load. The stored p_rho2 is p/rho
    // (ComputePressureAccel uses m_j * (p_i/rho_i + p_j/rho_j)), so p = p_rho2 * rho.
    // The floor is the one-layer lithostatic pressure used by WCSPH step2; it
    // also covers the first substep and woken particles, where p_prev = 0.
    float eta_i = 0.0f;
    if (is_granular_i) {
        float gamma_dot_i = __ldg(&shear_rate[i]);
        float p_floor_i = c_materials[mat_id_i].rest_density
                        * fabsf(c_sim.gravity.y) * c_granular.particle_spacing;
        float p_solver_i = (p_prev != 0)
                         ? __ldg(&p_prev[i]) * c_dfsph.kappa_uncarry * rho_i : 0.0f;
        float p_eff_i = fmaxf(p_solver_i, p_floor_i);
        eta_i = compute_muI_eta(gamma_dot_i, p_eff_i, rho_i);
    }

    // GAS skips the DFSPH pressure solvers, so it gets the WCSPH gas EOS
    // pressure inside its own phase here (bd-r5-review-n28.91). The GAS EOS
    // is linear, p = k * max(rho - rho0, 0), no powf.
    float p_term_i = 0.0f;
    float3 f_gas_press = make_float3(0.0f, 0.0f, 0.0f);
    if (is_gas_i) {
        float rho_i_safe = fmaxf(rho_i, RHO_EPSILON);
        p_term_i = compute_pressure(rho_i, GAS, mat_id_i) / (rho_i_safe * rho_i_safe);
    }

    // Vorticity eta accumulator (FLUID only, folded into viscosity loop -- PERF-002)
    bool do_vort_eta = (behavior_i == FLUID && vorticity_in != 0 && c_granular.vorticity_epsilon > 0.0f);
    float4 vort_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float omega_mag_i = 0.0f;
    float3 eta_vort = make_float3(0.0f, 0.0f, 0.0f);
    if (do_vort_eta) {
        vort_i = __ldg(&vorticity_in[i]);
        omega_mag_i = vort_i.w;
        do_vort_eta = (omega_mag_i > 1e-6f);
    }

    // Akinci 2013 curvature surface tension (FLUID only, bd-r5-review-n28.125):
    // the same pair term as WCSPH step2, summed in the viscosity loop below.
    bool st_on = (behavior_i == FLUID && normal_in != 0 && c_granular.surface_tension_gamma > 0.0f);
    float3 f_st_pair = make_float3(0.0f, 0.0f, 0.0f);
    float3 n_i_st = make_float3(0.0f, 0.0f, 0.0f);
    float rho0_i_st = c_materials[mat_id_i].rest_density;
    float m_i_st = __ldg(&mass[i]);
    if (st_on) {
        float4 n4_i = __ldg(&normal_in[i]);
        n_i_st = make_float3(n4_i.x, n4_i.y, n4_i.z);
    }

    // Compute viscosity + vorticity eta via neighbor loop
    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP: issue all loads before distance check (OPT-4.2)
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float4 vel4_j = __ldg(&velocity_in[j]);
                    float rho_j = __ldg(&density[j]);
                    float m_j = __ldg(&mass[j]);

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    // Vorticity eta accumulation (all neighbors -- PERF-002)
                    // Use difference form (omega_j - omega_mag_i) to match step2/neighbor_list.
                    if (do_vort_eta) {
                        float h_rl = h - rlen;
                        float inv_rl = 1.0f / rlen;
                        float gs = c_precalc.spiky_grad_coeff * h_rl * h_rl * inv_rl;
                        float omega_j = __ldg(&vorticity_in[j]).w;
                        float wt = m_j / fmaxf(rho_j, RHO_EPSILON);
                        float omega_diff = omega_j - omega_mag_i;
                        eta_vort.x += wt * omega_diff * gs * r.x;
                        eta_vort.y += wt * omega_diff * gs * r.y;
                        eta_vort.z += wt * omega_diff * gs * r.z;
                    }

                    int behavior_j = GET_BEHAVIOR(pi_j);

                    // STATIC boundary: MAT_RIGID viscous coupling, others get drag-to-zero
                    if (behavior_j == STATIC) {
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        float lap_v = lap_visc_variable(rlen, h);
                        float mu_b = is_granular_i ? c_precalc.viscosity_precalc
                                   : c_materials[mat_id_i].base_viscosity * c_precalc.viscosity_lap_coeff;

                        if (mat_id_j == MAT_RIGID && d_rigid_bodies != 0) {
                            // MAT_RIGID: viscous coupling with body velocity
                            int body_id = GET_BODY_ID(pi_j);
                            float4 rb_pos = __ldg(&d_rigid_bodies[body_id].position);
                            float4 rb_linvel = __ldg(&d_rigid_bodies[body_id].lin_vel);
                            float4 rb_angvel = __ldg(&d_rigid_bodies[body_id].ang_vel);
                            float3 r_b = make_float3(
                                pos4_j.x - rb_pos.x, pos4_j.y - rb_pos.y, pos4_j.z - rb_pos.z
                            );
                            float3 vel_boundary = make_float3(
                                rb_linvel.x + (rb_angvel.y * r_b.z - rb_angvel.z * r_b.y),
                                rb_linvel.y + (rb_angvel.z * r_b.x - rb_angvel.x * r_b.z),
                                rb_linvel.z + (rb_angvel.x * r_b.y - rb_angvel.y * r_b.x)
                            );
                            // psi_b = rho0_i * V_b; the mass slot holds
                            // RIGID_PSI_RHO0 * V_b (bd-r5-review-n28.77).
                            float psi_b = m_j * (__fdiv_rn(c_materials[mat_id_i].rest_density, RIGID_PSI_RHO0));
                            float visc_factor = mu_b * psi_b * lap_v / fmaxf(rho_i, RHO_EPSILON);
                            float3 F_visc = make_float3(
                                (vel_boundary.x - vel_i.x) * visc_factor,
                                (vel_boundary.y - vel_i.y) * visc_factor,
                                (vel_boundary.z - vel_i.z) * visc_factor
                            );
                            f_visc.x += F_visc.x;
                            f_visc.y += F_visc.y;
                            f_visc.z += F_visc.z;
                            // Two-way: accumulate reaction on rigid body.
                            // F_visc is a force density [N/kg = m/s^2]; the acceleration on
                            // particle i is a_visc = F_visc / rho_i (see a_visc block below).
                            // Newton's 3rd law: reaction force on body = -a_visc * m_i
                            //   = -(F_visc / rho_i) * m_i.
                            // Missing the /rho_i would overscale by ~rho_i (~1000-2500x for water).
                            float m_i = __ldg(&mass[i]);
                            float inv_rho = 1.0f / fmaxf(rho_i, RHO_EPSILON);
                            float3 F_on_body = make_float3(
                                -(F_visc.x * inv_rho) * m_i,
                                -(F_visc.y * inv_rho) * m_i,
                                -(F_visc.z * inv_rho) * m_i
                            );
                            float3 tau = make_float3(
                                r_b.y * F_on_body.z - r_b.z * F_on_body.y,
                                r_b.z * F_on_body.x - r_b.x * F_on_body.z,
                                r_b.x * F_on_body.y - r_b.y * F_on_body.x
                            );
                            atomic_accumulate3(d_rigid_forces, F_on_body, body_id);
                            atomic_accumulate3(d_rigid_torques, tau, body_id);
                        } else {
                            // Regular STATIC: friction drag toward zero
                            float visc_factor = mu_b * m_j * lap_v / fmaxf(rho_i, RHO_EPSILON);
                            f_visc.x += (-vel_i.x) * visc_factor;
                            f_visc.y += (-vel_i.y) * visc_factor;
                            f_visc.z += (-vel_i.z) * visc_factor;
                        }
                        continue;
                    }

                    // GAS/non-GAS phase separation
                    if (is_gas_i != (behavior_j == GAS)) continue;

                    float3 vel_j = make_float3(vel4_j.x, vel4_j.y, vel4_j.z);

                    float lap_v = lap_visc_variable(rlen, h);

                    if (is_granular_i && behavior_j == GRANULAR) {
                        // GRANULAR-GRANULAR: mu(I) with harmonic mean eta.
                        // eta_j uses j's own stored shear rate and pressure,
                        // exactly as j computes eta_i for itself, so
                        // eta_ij == eta_ji and the pair force is antisymmetric.
                        float gamma_dot_j = __ldg(&shear_rate[j]);
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        float p_floor_j = c_materials[mat_id_j].rest_density
                                        * fabsf(c_sim.gravity.y) * c_granular.particle_spacing;
                        float p_solver_j = (p_prev != 0)
                                         ? __ldg(&p_prev[j]) * c_dfsph.kappa_uncarry * rho_j : 0.0f;
                        float p_eff_j = fmaxf(p_solver_j, p_floor_j);
                        float eta_j = compute_muI_eta(gamma_dot_j, p_eff_j, rho_j);
                        float eta_ij = 2.0f * eta_i * eta_j / (eta_i + eta_j + 1e-8f);

                        // Full viscosity with coefficient baked in
                        float visc_factor = eta_ij * c_precalc.viscosity_lap_coeff * m_j * lap_v / fmaxf(rho_j, RHO_EPSILON);
                        f_visc.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_visc.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_visc.z += (vel_j.z - vel_i.z) * visc_factor;
                    } else if (is_granular_i) {
                        // GRANULAR-nonGRANULAR: base viscosity with full coefficient
                        float visc_factor = c_precalc.viscosity_precalc * m_j * lap_v / fmaxf(rho_j, RHO_EPSILON);
                        f_visc.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_visc.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_visc.z += (vel_j.z - vel_i.z) * visc_factor;
                    } else {
                        // FLUID/GAS: per-material viscosity via harmonic mean
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        if (is_gas_i) {
                            // GAS-GAS pressure: sum m_j (p_i/rho_i^2 + p_j/rho_j^2) grad_i W.
                            // grad_spiky points toward j, so subtracting it pushes i away.
                            float rho_j_safe = fmaxf(rho_j, RHO_EPSILON);
                            float p_term_j = compute_pressure(rho_j, GAS, mat_id_j) / (rho_j_safe * rho_j_safe);
                            float3 gW = grad_spiky(r, rlen, h);
                            float c_p = m_j * (p_term_i + p_term_j);
                            f_gas_press.x -= c_p * gW.x;
                            f_gas_press.y -= c_p * gW.y;
                            f_gas_press.z -= c_p * gW.z;
                        }
                        float mu_i = c_materials[mat_id_i].base_viscosity;
                        float mu_j = c_materials[mat_id_j].base_viscosity;
                        float mu_ij = 2.0f * mu_i * mu_j / (mu_i + mu_j + 1e-8f);
                        float visc_factor = mu_ij * c_precalc.viscosity_lap_coeff * m_j * lap_v / fmaxf(rho_j, RHO_EPSILON);
                        f_visc.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_visc.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_visc.z += (vel_j.z - vel_i.z) * visc_factor;
                    }

                    // Akinci curvature pair term, FLUID-FLUID only, as in step2.cu:
                    //   a_i += gamma K_ij h m_ij / m_i (n_i - n_j).
                    // Antisymmetric, so the pair adds no net momentum.
                    if (st_on && behavior_j == FLUID) {
                        float4 n4_j = __ldg(&normal_in[j]);
                        float rho0_j = c_materials[GET_MATERIAL_ID(pi_j)].rest_density;
                        float K_ij = (rho0_i_st + rho0_j) / fmaxf(rho_i + rho_j, RHO_EPSILON);
                        float c_curv = K_ij * h * 2.0f * m_j / fmaxf(m_i_st + m_j, 1e-12f);
                        f_st_pair.x += c_curv * (n_i_st.x - n4_j.x);
                        f_st_pair.y += c_curv * (n_i_st.y - n4_j.y);
                        f_st_pair.z += c_curv * (n_i_st.z - n4_j.z);
                    }
                }
            }
        }
    }

    // Apply viscosity: GRANULAR already has full coefficients, FLUID/GAS need scaling
    float3 a_visc;
    if (is_granular_i) {
        // GRANULAR: viscosity already baked in per-pair, just divide by rho_i
        float inv_rho = 1.0f / fmaxf(rho_i, RHO_EPSILON);
        a_visc = make_float3(f_visc.x * inv_rho, f_visc.y * inv_rho, f_visc.z * inv_rho);
    } else {
        // FLUID/GAS: per-material viscosity already baked in per-pair, divide by rho_i
        float inv_rho = 1.0f / fmaxf(rho_i, RHO_EPSILON);
        a_visc = make_float3(f_visc.x * inv_rho, f_visc.y * inv_rho, f_visc.z * inv_rho);
    }
    float3 accel = make_float3(
        a_visc.x + c_sim.gravity.x,
        a_visc.y + c_sim.gravity.y,
        a_visc.z + c_sim.gravity.z
    );

    // GAS pressure + buoyancy + drag. The pressure term carries force_scale
    // (0.02) so it has the same size as the WCSPH step2 gas pressure.
    if (behavior_i == GAS) {
        float fs = c_granular.force_scale;
        accel.x += fs * f_gas_press.x;
        accel.y += fs * f_gas_press.y;
        accel.z += fs * f_gas_press.z;
        float temp = __ldg(&temperature[i]);
        float3 b = gas_buoyancy_accel(temp, GET_MATERIAL_ID(pi_i));
        accel.x += b.x; accel.y += b.y; accel.z += b.z;
    }

    // FLUID thermal convection: Boussinesq buoyancy
    // rho_eff = rho_0 * (1 - beta*(T-T0)), lighter when hot -> rises
    if (behavior_i == FLUID) {
        float temp_i = __ldg(&temperature[i]);
        float beta = c_materials[GET_MATERIAL_ID(pi_i)].thermal_expansion;
        if (beta > 0.0f) {
            float3 b = boussinesq_accel(temp_i, beta);
            accel.x += b.x; accel.y += b.y; accel.z += b.z;
        }
    }

    // Vorticity confinement force (FLUID only, eta from viscosity loop -- PERF-002)
    if (do_vort_eta) {
        float eta_mag = sqrtf(eta_vort.x*eta_vort.x + eta_vort.y*eta_vort.y + eta_vort.z*eta_vort.z);
        if (eta_mag > 1e-6f) {
            float inv_eta = 1.0f / eta_mag;
            float3 N = make_float3(eta_vort.x*inv_eta, eta_vort.y*inv_eta, eta_vort.z*inv_eta);
            float eps_v = c_granular.vorticity_epsilon;
            accel.x += eps_v * (N.y * vort_i.z - N.z * vort_i.y);
            accel.y += eps_v * (N.z * vort_i.x - N.x * vort_i.z);
            accel.z += eps_v * (N.x * vort_i.y - N.y * vort_i.x);
        }
    }

    // Akinci curvature surface tension (FLUID only, pair sum from the loop
    // above). It carries force_scale, so the same gamma gives the same accel
    // as WCSPH step2, whose final fs_i multiply scales it.
    if (st_on) {
        float g_fs = c_granular.surface_tension_gamma * c_granular.force_scale;
        accel.x += g_fs * f_st_pair.x;
        accel.y += g_fs * f_st_pair.y;
        accel.z += g_fs * f_st_pair.z;
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
 * div_v = SUM_j V_j * (v*_i - v*_j) . gradW_ij = +(1/rho) Drho/Dt,
 *   V_j the Akinci volume from the position.w carrier (dfsph_vol_akinci),
 *   positive under compression (grad_spiky points toward j)
 * dkappa_v_i = max(div_v, 0) * alpha_i * omega / dt
 *   (multiply by alpha = 1/denom, never divide)
 *
 * kappa_v_out gets this iteration's increment, which CorrectVelocityDiv
 * applies. kappa_v_total (optional) is the running sum over the warm-start
 * value and every iteration's increment; Finalize writes it back as next
 * substep's warm start, as in SPlisHSPlasH (bd-r5-review-n28.83).
 * Each thread touches only its own index, so there is no race.
 * ====================================================================== */

#define DFSPH_DIV_MIN_NEIGHBORS 20.0f

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ComputeKappaV(
    uint            numParticles,
    const float4*   __restrict__ velocity,
    const float*    __restrict__ alpha,
    const uint*     __restrict__ packed_info,
    const float4*   __restrict__ position,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ kappa_v_out,    // this iteration's increment
    const float4*   __restrict__ normal_in,    // (n_x,n_y,n_z, neighbor_count) from DensityAlpha, or NULL
    float*          __restrict__ kappa_v_total // running sum for the warm start (in-out), or NULL
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);

    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) {
        kappa_v_out[i] = 0.0f;
        if (kappa_v_total != 0) kappa_v_total[i] = 0.0f;
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
    // Akinci wall volume divisor (dfsph_vol_rho); floor at 1 for DEAD
    float rho0_vol_i = fmaxf(c_materials[GET_MATERIAL_ID(pi_i)].rest_density, 1.0f);

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP (OPT-4.2). pos4_j.w is the volume carrier.
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float4 vel4_j = __ldg(&velocity[j]);

                    int bj = GET_BEHAVIOR(pi_j);
                    if (bj == GAS) continue;

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
                    div_v += dfsph_vol_akinci(pos4_j.w, pi_j, rho0_vol_i) * dv_dot_gW;
                }
            }
        }
    }

    float alpha_i = alpha[i];
    // DFSPH: kv = (1/dt) * div_v * alpha
    // div_v = SUM V_j (v_i - v_j) . gradW is the relative density rate
    // (1/rho) Drho/Dt, the same sign as drho in K_DFSPH_DensitySolverUpdate:
    // div_v > 0 means the particle is being compressed.
    // omega: under-relaxation to prevent overcorrection.
    // Unilateral: only compressing flow is corrected. Pulling separating
    // particles back together is artificial cohesion at free surfaces, and the
    // bilateral Jacobi diverges once alpha is not capped far below 1/denom
    // (bd-r5-review-n28.37). Same as SPlisHSPlasH densityAdv = max(densityAdv, 0).
    div_v = fmaxf(div_v, 0.0f);
    // Neighbour-count gate: with fewer than DFSPH_DIV_MIN_NEIGHBORS neighbours
    // (rigid included, bd-r5-review-n28.68) the kernel is truncated and the
    // divergence estimate is biased, so skip the solve (SPlisHSPlasH: numNeighbors < 20
    // in 3D). NULL normal_in disables the gate (bd-r5-review-n28.71).
    if (normal_in != 0 && __ldg(&normal_in[i]).w < DFSPH_DIV_MIN_NEIGHBORS) div_v = 0.0f;
    float dkv = div_v * alpha_i * c_dfsph.omega / dt;
    kappa_v_out[i] = dkv;
    if (kappa_v_total != 0) kappa_v_total[i] += dkv;
}

/* ======================================================================
 * K_DFSPH_CorrectVelocityDiv -- Apply divergence correction
 * v*_i -= dt * SUM_j c_ij * gradW_ij  (kv >= 0)
 *   fluid j:  c_ij = m_j * (kv_i/rho_i + kv_j/rho_j)
 *   STATIC j: c_ij = V_j * kv_i, V_j = m_j / rho_j (dfsph_vol_plain)
 * The fluid pair term is symmetric in i and j, so the impulses on i and j
 * cancel and momentum is conserved across density jumps (bd-r5-review-n28.100).
 * The STATIC term keeps the plain volume: its reaction goes to the rigid body.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_CorrectVelocityDiv(
    uint            numParticles,
    float4*         __restrict__ velocity,        // in-out
    const float*    __restrict__ density,         // own density and wall neighbours
    const float*    __restrict__ mass,            // neighbour mass (fluid pairs) and own mass (rigid reaction)
    const float*    __restrict__ kappa_v,
    const uint*     __restrict__ packed_info,
    const float4*   __restrict__ position,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    const RigidBody* __restrict__ d_rigid_bodies,  // rigid body state (NULL if no bodies)
    float*          __restrict__ d_rigid_forces,   // force accumulator (NULL if no bodies)
    float*          __restrict__ d_rigid_torques   // torque accumulator (NULL if no bodies)
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) return;
    // Rigid reaction: the pair changes v_i by -dt * V_j * kv_i * gW, so the
    // body gets the impulse +m_i * dt * V_j * kv_i * gW, a force
    // m_i * V_j * kv_i * gW over the substep (kv_j = 0 for STATIC).
    float m_i_rigid = (d_rigid_bodies != 0) ? mass[i] : 0.0f;

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;
    float kv_i = kappa_v[i];
    // kv_i / rho_i is the same for every fluid neighbour, so divide once.
    float kv_rho_i = kv_i / fmaxf(density[i], RHO_EPSILON);

    float3 dv = make_float3(0.0f, 0.0f, 0.0f);

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP (OPT-4.2). pos4_j.w is the volume carrier.
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float kv_j = __ldg(&kappa_v[j]);

                    int bj = GET_BEHAVIOR(pi_j);
                    if (bj == GAS) continue;
                    // STATIC never carries a divergence factor, whatever the
                    // buffer holds (warm start of a fresh slot).
                    if (bj == STATIC) kv_j = 0.0f;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 gW = grad_spiky(r, rlen, h);
                    // Fluid: m_j * kv_i/rho_i + V_j * kv_j, V_j * kv_j = m_j * kv_j/rho_j.
                    // STATIC: V_j * kv_i (kv_j = 0). mass[j] loads only in range.
                    float coeff = (bj == STATIC)
                        ? dfsph_vol_plain(pos4_j.w, pi_j, density, j) * kv_i
                        : __ldg(&mass[j]) * kv_rho_i + pos4_j.w * kv_j;
                    dv.x += coeff * gW.x;
                    dv.y += coeff * gW.y;
                    dv.z += coeff * gW.z;

                    if (bj == STATIC && d_rigid_bodies != 0
                        && GET_MATERIAL_ID(pi_j) == MAT_RIGID) {
                        float fs = m_i_rigid * coeff;
                        dfsph_rigid_reaction(d_rigid_bodies, d_rigid_forces, d_rigid_torques,
                                             pi_j, pos4_j,
                                             make_float3(fs * gW.x, fs * gW.y, fs * gW.z));
                    }
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
 * K_DFSPH_ComputePressureAccel -- Pressure acceleration and predicted velocity
 *
 * Jacobi density solver kernel 1 of 2:
 * a_press_i = -SUM_j c_ij * grad_W_ij
 *   fluid j:  c_ij = m_j * (p_rho2_i/rho_i + p_rho2_j/rho_j)
 *   STATIC j: c_ij = V_j * p_rho2_i, V_j = m_j / rho_j (dfsph_vol_plain)
 * p_rho2 holds p/rho, so p_rho2/rho = p/rho^2. The fluid pair term is
 * symmetric in i and j, so pair forces cancel (bd-r5-review-n28.100).
 *
 * Writes the predicted velocity, not a_press (bd-r5-review-n28.184):
 *   vt_i = clamp_v(v_i + dt * clamp_a(a_press_i))
 * clamp_a limits |a_press| to h/dt^2 (velocity change of at most h/dt per
 * substep), clamp_v limits |vt| to VELOCITY_LIMIT. DensitySolverUpdate reads
 * vt for i and every neighbour, so the clamps run once per particle instead
 * of once per pair. The last call of a substep writes vt into the velocity
 * buffer itself, which applies the pressure (there is no separate apply
 * pass). velocity and vt_out may be the same buffer: each thread reads and
 * writes only its own index, so they are not __restrict__.
 * STATIC, GAS and sleeping particles write vt = v unchanged.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_DFSPH_ComputePressureAccel(
    uint            numParticles,
    const float*    __restrict__ p_rho2,
    const float4*   velocity,                      // own index only; may alias vt_out
    const float4*   __restrict__ position,
    const float*    __restrict__ density,          // own density and wall neighbours
    const float*    __restrict__ mass,             // neighbour mass (fluid pairs) and own mass (rigid reaction)
    const uint*     __restrict__ packed_info,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float4*         vt_out,                        // predicted velocity; may alias velocity
    const RigidBody* __restrict__ d_rigid_bodies,  // non-NULL only on the final call (see below)
    float*          __restrict__ d_rigid_forces,
    float*          __restrict__ d_rigid_torques
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float4 vel4_i = velocity[i];
    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    if (behavior_i == STATIC || behavior_i == GAS || IS_SLEEPING(pi_i)) {
        vt_out[i] = vel4_i;
        return;
    }

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float p_i = p_rho2[i];
    // p_i / rho_i is the same for every fluid neighbour, so divide once.
    float p_rho_i = p_i / fmaxf(density[i], RHO_EPSILON);
    // Rigid reaction: only the last a_press is applied (the final call writes
    // the velocity), so the caller passes the rigid buffers on the final call
    // only. The pair adds -V_j * p_i * gW to a_i (p_j = 0 for STATIC), so the
    // body gets +m_i * V_j * p_i * gW.
    float m_i_rigid = (d_rigid_bodies != 0) ? mass[i] : 0.0f;

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;

    float3 a = make_float3(0.0f, 0.0f, 0.0f);

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP: issue all loads before distance check (OPT-4.2).
                    // pos4_j.w is the volume carrier.
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float p_j = __ldg(&p_rho2[j]);

                    int bj = GET_BEHAVIOR(pi_j);
                    if (bj == GAS) continue;
                    // STATIC never carries pressure, whatever the buffer
                    // holds (warm start of a fresh slot).
                    if (bj == STATIC) p_j = 0.0f;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 gW = grad_spiky(r, rlen, h);
                    // Fluid: m_j * p_i/rho_i + V_j * p_j, V_j * p_j = m_j * p_j/rho_j.
                    // STATIC: V_j * p_i (p_j = 0). mass[j] loads only in range.
                    float coeff = (bj == STATIC)
                        ? -dfsph_vol_plain(pos4_j.w, pi_j, density, j) * p_i
                        : -(__ldg(&mass[j]) * p_rho_i + pos4_j.w * p_j);
                    a.x += coeff * gW.x;
                    a.y += coeff * gW.y;
                    a.z += coeff * gW.z;

                    if (bj == STATIC && d_rigid_bodies != 0
                        && GET_MATERIAL_ID(pi_j) == MAT_RIGID) {
                        float fs = -m_i_rigid * coeff;
                        dfsph_rigid_reaction(d_rigid_bodies, d_rigid_forces, d_rigid_torques,
                                             pi_j, pos4_j,
                                             make_float3(fs * gW.x, fs * gW.y, fs * gW.z));
                    }
                }
            }
        }
    }

    // Clamp pressure acceleration: limit velocity change to h/dt per substep.
    // Prevents single-particle oscillation from Jacobi overcorrection.
    float a_sq = a.x * a.x + a.y * a.y + a.z * a.z;
    float a_press_max = c_sim.smoothing_length / (dt * dt);  // h/dt^2
    if (a_sq > a_press_max * a_press_max) {
        float scale = a_press_max / sqrtf(a_sq);
        a.x *= scale;
        a.y *= scale;
        a.z *= scale;
    }

    float3 vt = make_float3(
        vel4_i.x + dt * a.x,
        vel4_i.y + dt * a.y,
        vel4_i.z + dt * a.z
    );
    // Velocity clamp. In the density prediction it also stops a large a_press
    // overshoot in one Jacobi iteration from producing a huge drho that drives
    // the next iteration further out (runaway oscillation).
    float vt_sq = vt.x * vt.x + vt.y * vt.y + vt.z * vt.z;
    if (vt_sq > VELOCITY_LIMIT_SQ) {
        float scale = VELOCITY_LIMIT / sqrtf(vt_sq);
        vt.x *= scale;
        vt.y *= scale;
        vt.z *= scale;
    }

    vt_out[i] = make_float4(vt.x, vt.y, vt.z, vel4_i.w);
}

/* ======================================================================
 * K_DFSPH_DensitySolverUpdate -- Jacobi update for density pressure
 *
 * Jacobi density solver kernel 2 of 2:
 * 1) Compute predicted density from the predicted velocity vt (written by
 *    K_DFSPH_ComputePressureAccel, already clamped):
 *    drho = SUM_j V_j * (vt_i - vt_j) . grad_W_ij, V_j the Akinci volume
 *    density_adv = (rho_i / rho0) * (1 + dt * drho)
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
    const float4*   __restrict__ vt,             // predicted velocity v + dt * a_press, clamped
    const float4*   __restrict__ position,
    const float*    __restrict__ density,        // own density only
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
    float4 vt_i = vt[i];

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float dt = c_sim.dt;
    float rho_i = density[i];
    // Floor at 1: DEAD has rest_density 0, and rho_i / rho0 would be inf.
    float rho0 = fmaxf(c_materials[GET_MATERIAL_ID(pi_i)].rest_density, 1.0f);

    // Compute density rate-of-change from total velocity
    float drho = 0.0f;

    int3 cell_i = get_cell(pos_i);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    // Speculative ILP: issue all loads before distance check (OPT-4.2).
                    // pos4_j.w is the volume carrier.
                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float4 vt_j = __ldg(&vt[j]);

                    int bj = GET_BEHAVIOR(pi_j);
                    if (bj == GAS) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    float3 gW = grad_spiky(r, rlen, h);
                    float dv_dot_gW = (vt_i.x - vt_j.x) * gW.x
                                    + (vt_i.y - vt_j.y) * gW.y
                                    + (vt_i.z - vt_j.z) * gW.z;
                    drho += dfsph_vol_akinci(pos4_j.w, pi_j, rho0) * dv_dot_gW;
                }
            }
        }
    }

    // Predicted density ratio (normalized to rest density).
    // drho is volume weighted (SUM V_j (vt_i - vt_j) . gradW, units 1/s), so it
    // is already the relative density rate (1/rho) Drho/Dt and dt * drho is
    // dimensionless. rho*/rho0 = (rho_i/rho0) * (1 + dt * drho). Do not divide
    // drho by rho0 again: that shrinks the A*p feedback by rho0 (2500 for WATER)
    // and the Jacobi iteration stops converging. alpha (K_DFSPH_ComputeDensityAlpha)
    // is the inverse diagonal of this residual.
    float density_adv = (rho_i / rho0) * (1.0f + dt * drho);
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
 * K_DFSPH_Finalize -- Final position, boundary, sleep, writeback
 * x_final = x + dt * v*_final
 *
 * Render color is not stored here: K_ComputeColor (integrate.cu) computes
 * it once per frame from the unsorted arrays (bd-r5-review-n28.145).
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
    uint*           __restrict__ packed_info_out,
    unsigned char*  __restrict__ sleep_counter_out,
    float*          __restrict__ temperature_out,
    float*          __restrict__ kappa_out,
    const float*    __restrict__ sorted_kappa_v,        // divergence kappa for warm-start writeback
    float*          __restrict__ kappa_v_out,           // unsorted kappa_v output (warm-start next frame)
    const RigidBody* __restrict__ d_rigid_bodies,       // rigid body state (NULL if no bodies); gates the XSPH skip and the body push-out
    float*          __restrict__ d_rigid_forces,        // unused: reaction is accumulated in the solver kernels
    float*          __restrict__ d_rigid_torques,       // unused: see d_rigid_forces
    const float*    __restrict__ sorted_lifetime,       // reaction-mutated lifetime (sorted), or NULL
    float*          __restrict__ health_out,            // unsorted health writeback, or NULL
    float*          __restrict__ lifetime_out,          // unsorted lifetime writeback, or NULL
    float*          __restrict__ mass_out,              // unsorted mass writeback, or NULL
    uint*           __restrict__ max_displacement_out,   // [1] atomicMax of displacement^2 (float-as-uint), or NULL
    const float4*   __restrict__ displacement_anchor, // [N] sorted position at the last full sort, or NULL
    uint*           __restrict__ cell_wake_flags,     // wake stamp table, or NULL (bd-r5-review-n28.167)
    const uint*     __restrict__ frame_ptr            // per-substep counter for the stamp, or NULL
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = sorted_packed_info[i];
    int behavior = GET_BEHAVIOR(pi);
    uint mat_id = GET_MATERIAL_ID(pi);
    uint orig_idx = sort_indexes[i];

    // Reaction-state carry-back (bd-r4fix-uup.9): K_Reactions/K_SpawnGas mutate
    // health/lifetime/mass in SORTED space every substep; scatter them back to
    // the unsorted arrays or the next substep re-gathers stale values (wiping
    // fire lifetime decay, acid health loss, gas-spawn mass transfer). None of
    // these are modified below, so an early unconditional scatter is correct
    // and also covers the STATIC/sleeping early-return paths.
    if (health_out)                      health_out[orig_idx]   = sorted_health[i];
    if (lifetime_out && sorted_lifetime) lifetime_out[orig_idx] = __ldg(&sorted_lifetime[i]);
    if (mass_out)                        mass_out[orig_idx]     = sorted_mass[i];

    float temp = sorted_temperature[i];

    // STATIC: write through
    if (behavior == STATIC) {
        float4 pos4 = sorted_position[i];
        float dTdt = sorted_dTdt[i];
        temp += dTdt * c_sim.dt;
        temp -= COOL_RATE * (temp - T_AMBIENT) * c_sim.dt;
        temp = fmaxf(T_MIN, fminf(temp, T_MAX));
        position_out[orig_idx] = pos4;
        velocity_out[orig_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        packed_info_out[orig_idx] = pi;
        sleep_counter_out[orig_idx] = sorted_sleep_counter[i];
        temperature_out[orig_idx] = temp;
        kappa_out[orig_idx] = 0.0f;
        if (kappa_v_out) kappa_v_out[orig_idx] = 0.0f;
        return;
    }

    unsigned char sc = sorted_sleep_counter[i];
    bool was_sleeping = IS_SLEEPING(pi) != 0;

    float4 pos4 = sorted_position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
    float4 vel4 = sorted_velocity[i];
    float3 vel = make_float3(vel4.x, vel4.y, vel4.z);
    float dt = c_sim.dt;

    // Sleeping check. Only GRANULAR may stay asleep: a FLUID or GAS particle
    // carrying a stale SLEEPING bit (older build, loaded snapshot) wakes here.
    if (was_sleeping) {
        float vel_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
        if (behavior == GRANULAR && vel_sq <= V_WAKE_SQ) {
            float dTdt = sorted_dTdt[i];
            temp += dTdt * dt;
            temp -= COOL_RATE * (temp - T_AMBIENT) * dt;
            temp = fmaxf(T_MIN, fminf(temp, T_MAX));
            position_out[orig_idx] = pos4;
            velocity_out[orig_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            packed_info_out[orig_idx] = pi;
            sleep_counter_out[orig_idx] = (sc < 255) ? sc : (unsigned char)255;
            temperature_out[orig_idx] = temp;
            kappa_out[orig_idx] = 0.0f;
            if (kappa_v_out) kappa_v_out[orig_idx] = 0.0f;
                return;
        }
        pi = CLEAR_SLEEPING(pi);
        pi = SET_JUST_WOKE(pi);
        sc = 0;
    }

    // STATIC push, handled in the XSPH loop below so Finalize usually scans
    // the neighbourhood once (bd-r5-review-n28.164). The push is measured at
    // pos_new = pos + dt * vel_advect, which needs the XSPH sum, so the loop
    // tests wall particles against pos_adv = pos + dt * vel instead.
    //   GRANULAR: no XSPH, so pos_adv == pos_new and the loop accumulates
    //     the exact push.
    //   FLUID: the loop only flags a wall particle within band + margin of
    //     pos_adv. With no flag and |pos_new - pos_adv| < margin, no wall
    //     particle is within the band of pos_new, so there is no push.
    //     Otherwise the particle runs the separate scan at pos_new.
    // The 27 cells around pos hold every wall particle within band + margin
    // of pos_adv only while dt*|vel| < h - band - margin (cell size is h).
    // Faster particles and GAS (no XSPH loop) take the separate scan, so
    // the result always equals static_particle_boundary(pos_new).
    StaticPush spush = static_push_init();
    bool spush_in_loop = false;
    bool fluid_near_wall = false;
    float sp_band = static_push_band();
    float sp_band_sq = sp_band * sp_band;
    float sp_margin = 0.1f * c_sim.smoothing_length;
    float sp_reach_sq = (sp_band + sp_margin) * (sp_band + sp_margin);
    float3 pos_adv = make_float3(pos.x + dt * vel.x, pos.y + dt * vel.y, pos.z + dt * vel.z);

    // XSPH velocity smoothing. GRANULAR gets no XSPH, so for GRANULAR the
    // loop only gathers the STATIC push: it reads packed_info for every
    // neighbour and position for STATIC ones, and skips the loop when the
    // push falls back to the separate scan (bd-r5-review-n28.188).
    float3 vel_advect = vel;
    bool xsph_self = (behavior == FLUID);
    if (behavior == FLUID || behavior == GRANULAR) {
        float h = c_sim.smoothing_length;
        float step_sq = dt * dt * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        float step_max = h - sp_band - sp_margin;
        spush_in_loop = (step_sq < step_max * step_max);
    }
    if (xsph_self || (behavior == GRANULAR && spush_in_loop)) {
        float h_sq = c_sim.smoothing_length_sq;
        float rho_i = sorted_density[i];
        // Akinci wall volume divisor (dfsph_vol_rho); floor at 1 for DEAD
        float rho0_vol_i = fmaxf(c_materials[mat_id].rest_density, 1.0f);
        float3 xsph = make_float3(0.0f, 0.0f, 0.0f);

        int3 cell_i = get_cell(pos);
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                    uint start = cell_start[hash_c];
                    uint end_idx = cell_end[hash_c];

                    for (uint j = start; j < end_idx; j++) {
                        if (j == i) continue;
                        uint pi_j = __ldg(&sorted_packed_info[j]);
                        int bj = GET_BEHAVIOR(pi_j);
                        if (bj == GAS) continue;
                        if (!xsph_self && bj != STATIC) continue;

                        float4 p4j = __ldg(&sorted_position[j]);

                        // STATIC push at pos_adv, before the h cutoff on pos.
                        if (spush_in_loop && bj == STATIC && static_push_source(pi_j, false)) {
                            float3 ra = make_float3(pos_adv.x - p4j.x, pos_adv.y - p4j.y, pos_adv.z - p4j.z);
                            float ra_sq = ra.x * ra.x + ra.y * ra.y + ra.z * ra.z;
                            if (behavior == GRANULAR) {
                                static_push_add(spush, ra, ra_sq, sp_band, sp_band_sq);
                            } else if (ra_sq < sp_reach_sq) {
                                fluid_near_wall = true;
                            }
                        }
                        if (!xsph_self) continue;

                        float3 r = make_float3(pos.x - p4j.x, pos.y - p4j.y, pos.z - p4j.z);
                        float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                        if (r_sq > h_sq || r_sq < 1e-12f) continue;

                        float m_j = __ldg(&sorted_mass[j]);

                        // Rigid boundary particles get no XSPH. The fluid-body
                        // reaction is accumulated in K_DFSPH_CorrectVelocityDiv and
                        // K_DFSPH_ComputePressureAccel, where the solver pushes the fluid.
                        if (bj == STATIC && d_rigid_bodies != 0
                            && GET_MATERIAL_ID(pi_j) == MAT_RIGID) {
                            continue;
                        }

                        float4 vj4 = __ldg(&sorted_velocity[j]);
                        float rho_j = dfsph_vol_rho(__ldg(&sorted_density[j]), pi_j, rho0_vol_i);

                        float w = W_poly6(r_sq, h_sq);
                        float rho_avg = 0.5f * (rho_i + rho_j);
                        float factor = (m_j / rho_avg) * w;
                        xsph.x += (vj4.x - vel.x) * factor;
                        xsph.y += (vj4.y - vel.y) * factor;
                        xsph.z += (vj4.z - vel.z) * factor;
                    }
                }
            }
        }
        // Apply XSPH to FLUID only (not GRANULAR), matching step2.cu (bd-mzc.34).
        // GRANULAR uses mu(I) rheology for velocity smoothing; XSPH is redundant there.
        if (xsph_self) {
            float c_xsph = c_granular.xsph_epsilon;
            vel_advect.x += c_xsph * xsph.x;
            vel_advect.y += c_xsph * xsph.y;
            vel_advect.z += c_xsph * xsph.z;
        }
    }

    // Final position: x_final = x + dt * v_advect (XSPH-corrected)
    float3 pos_new = make_float3(
        pos.x + dt * vel_advect.x,
        pos.y + dt * vel_advect.y,
        pos.z + dt * vel_advect.z
    );

    // STATIC particle boundary repulsion: the push gathered in the XSPH
    // loop, or the shared 27-cell scan at pos_new for GAS, fast particles
    // and FLUID near a wall (see the note above the XSPH loop).
    if (spush_in_loop && behavior == FLUID) {
        float3 d = make_float3(pos_new.x - pos_adv.x, pos_new.y - pos_adv.y, pos_new.z - pos_adv.z);
        float d_sq = d.x * d.x + d.y * d.y + d.z * d.z;
        if (fluid_near_wall || d_sq >= sp_margin * sp_margin) spush_in_loop = false;
    }
    if (spush_in_loop) {
        static_push_apply(spush, pos_new, vel, c_sim.restitution);
    } else {
        static_particle_boundary(
            pos_new, vel,
            cell_start, cell_end,
            sorted_packed_info, sorted_position,
            i, c_sim.restitution
        );
    }

    // Box boundary. FLUID gets zero wall friction, matching WCSPH and PBF.
    float friction = (behavior == FLUID) ? 0.0f : c_sim.wall_friction;
    sdf_box_boundary(pos_new, vel, c_sim.world_min, c_sim.world_max,
                     c_sim.restitution, friction);

    // SDF object collision with velocity reflection
    for (int s = 0; s < c_num_sdf_objects; s++) {
        SDFObject obj = c_sdf_objects[s];
        float sdf_dist = eval_sdf(pos_new, obj);
        if (sdf_dist < BOUNDARY_MARGIN) {
            float3 n = sdf_normal(pos_new, obj);
            float push = BOUNDARY_MARGIN - sdf_dist;
            pos_new.x += push * n.x;
            pos_new.y += push * n.y;
            pos_new.z += push * n.z;

            float3 obj_vel = make_float3(obj.velocity.x, obj.velocity.y, obj.velocity.z);
            float ang_speed = obj.velocity.w;
            if (fabsf(ang_speed) > 1e-8f) {
                float3 axis = make_float3(obj.angular_axis.x, obj.angular_axis.y, obj.angular_axis.z);
                float3 center = make_float3(obj.pos_and_type.x, obj.pos_and_type.y, obj.pos_and_type.z);
                float3 r = make_float3(pos_new.x - center.x, pos_new.y - center.y, pos_new.z - center.z);
                float3 omega = make_float3(axis.x * ang_speed, axis.y * ang_speed, axis.z * ang_speed);
                obj_vel.x += omega.y * r.z - omega.z * r.y;
                obj_vel.y += omega.z * r.x - omega.x * r.z;
                obj_vel.z += omega.x * r.y - omega.y * r.x;
            }

            float3 v_rel = make_float3(vel.x - obj_vel.x, vel.y - obj_vel.y, vel.z - obj_vel.z);
            float v_dot_n = v_rel.x * n.x + v_rel.y * n.y + v_rel.z * n.z;
            if (v_dot_n < 0.0f) {
                float obj_restitution = obj.size_and_r.w;
                float obj_friction = obj.angular_axis.w;
                vel.x -= (1.0f + obj_restitution) * v_dot_n * n.x;
                vel.y -= (1.0f + obj_restitution) * v_dot_n * n.y;
                vel.z -= (1.0f + obj_restitution) * v_dot_n * n.z;
                float3 v_tan = make_float3(v_rel.x - v_dot_n*n.x, v_rel.y - v_dot_n*n.y, v_rel.z - v_dot_n*n.z);
                float v_tan_len = sqrtf(v_tan.x*v_tan.x + v_tan.y*v_tan.y + v_tan.z*v_tan.z);
                if (v_tan_len > 1e-8f) {
                    float red = fminf(obj_friction * fabsf(v_dot_n) / v_tan_len, 1.0f);
                    vel.x -= red * v_tan.x;
                    vel.y -= red * v_tan.y;
                    vel.z -= red * v_tan.z;
                }
            }
        }
    }

    // Rigid body collision (bd-r5-review-n28.123): push out of each body's
    // box or sphere SDF at its end-of-substep pose, so a fast body cannot
    // overtake fluid and trap it inside. Same helper as K_Integrate.
    if (d_rigid_bodies != 0 && c_num_rigid_bodies > 0) {
        rigid_body_boundary(pos_new, vel, d_rigid_bodies, c_num_rigid_bodies, dt);
    }

    // Velocity clamp
    float vel_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    if (vel_sq > VELOCITY_LIMIT_SQ) {
        float scale = VELOCITY_LIMIT / sqrtf(vel_sq);
        vel.x *= scale;
        vel.y *= scale;
        vel.z *= scale;
        vel_sq = VELOCITY_LIMIT_SQ;
    }

    // Spawn velocity damping (ramps to 0 after first ~30 substeps)
    if (c_sim.velocity_damping > 0.0f) {
        float damp = 1.0f - c_sim.velocity_damping;
        vel.x *= damp;
        vel.y *= damp;
        vel.z *= damp;
        vel_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
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

    // Sleep: velocity-based, GRANULAR only, matching WCSPH (integrate.cu) and
    // PBF. Sleeping particles get no pressure, so a calm FLUID pool or slow GAS
    // would freeze with nothing left to wake it.
    if (behavior == GRANULAR) {
        if (vel_sq < V_SLEEP_SQ) {
            if (sc < 255) sc++;
        } else {
            sc = 0;
        }
        if (sc >= SLEEP_THRESHOLD) {
            pi = SET_SLEEPING(pi);
        }
    } else {
        sc = 0;
    }

    // Temperature
    float dTdt = sorted_dTdt[i];
    temp += dTdt * dt;
    temp -= COOL_RATE * (temp - T_AMBIENT) * dt;
    temp = fmaxf(T_MIN, fminf(temp, T_MAX));

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

    // Own-cell wake mark from registers, so wake.py skips K_MarkWakeCells
    // (bd-r5-review-n28.167). vel_sq is the final |v|^2 here.
    wake_mark_own_cell(cell_wake_flags, frame_ptr, pi, behavior, vel_sq, pos_new);

    // Writeback. .w carries this substep's density for K_ComputeColor,
    // which runs once per frame on the unsorted arrays
    // (bd-r5-review-n28.145). The sorted position.w holds the solver's
    // volume carrier (K_DFSPH_PackVolume), not the density.
    position_out[orig_idx] = make_float4(pos_new.x, pos_new.y, pos_new.z, sorted_density[i]);
    velocity_out[orig_idx] = make_float4(vel.x, vel.y, vel.z, 0.0f);
    packed_info_out[orig_idx] = pi;
    sleep_counter_out[orig_idx] = sc;
    temperature_out[orig_idx] = temp;
    // Warm start for the next substep, scaled here instead of in two
    // separate host-side multiply passes (bd-r5-review-n28.173).
    kappa_out[orig_idx] = sorted_kappa[i] * c_dfsph.kappa_carry;
    if (kappa_v_out) kappa_v_out[orig_idx] = sorted_kappa_v[i] * c_dfsph.kappa_v_carry;
}
