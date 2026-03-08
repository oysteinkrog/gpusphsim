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

#include "sph_shared.cuh"

/* ======================================================================
 * DFSPH Parameters
 * ====================================================================== */

struct DFSPHParams {
    int   div_iters;
    int   dens_iters;
    float warm_start;
    float omega;        // under-relaxation factor for corrections (0.3-1.0)
    float alpha_limit;  // max alpha as fraction of dt^2 (default 1.0, higher = faster convergence)
    float _pad[3];      // pad to 32 bytes
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
 *              + SUM_j (m_j/rho_j)^2 |gradW_ij|^2 + eps)
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
    const float4*   __restrict__ particle_dye_in,
    float4*         __restrict__ dye_rate_out,
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

    // MAT_RIGID boundary particles: early-return, no density/alpha needed
    if (mat_id_i == MAT_RIGID) {
        density_out[i] = 2500.0f;
        alpha_out[i] = 0.0f;
        shear_rate_out[i] = 0.0f;
        dTdt_out[i] = 0.0f;
        exposure_heat_out[i] = 0.0f;
        exposure_corrode_out[i] = 0.0f;
        if (dye_rate_out) dye_rate_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

    float sum_density = 0.0f;
    float sum_dTdt = 0.0f;
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;
    float3 dye_rate = make_float3(0.0f, 0.0f, 0.0f);
    bool dye_on = (c_sim.dye_enabled != 0) && (dye_rate_out != 0);
    float4 dye_i = dye_on ? __ldg(&particle_dye_in[i]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

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

                    // GAS/non-GAS phase separation: skip density/heat/forces,
                    // keep only exposure (so fire-to-wood ignition still works)
                    bool is_gas_j = (behavior_j == GAS);
                    if (is_gas_i != is_gas_j && j != i) {
                        float diff_xp = h_sq - r_sq;
                        float T_j = __ldg(&temperature_in[j]);
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        float w_poly6_var = diff_xp * diff_xp * diff_xp;
                        sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                        sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
                        continue;
                    }

                    // Density (Poly6)
                    float diff = h_sq - r_sq;
                    sum_density += m_j * diff * diff * diff;

                    // Skip self for non-density quantities
                    if (j != i && r_sq > 1e-12f) {
                        float rlen = sqrtf(r_sq);
                        float rho_j = (density_in != 0) ? __ldg(&density_in[j]) : 1000.0f;

                        // Skip MAT_RIGID for heat/exposure/dye/vorticity/strain
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        bool is_rigid_j = (mat_id_j == MAT_RIGID);

                        if (!is_rigid_j) {
                            // Heat diffusion (with cross-material boost)
                            float T_j = __ldg(&temperature_in[j]);
                            float lap_var = h - rlen;
                            float heat_boost = fmaxf(1.0f, c_interactions[mat_id_i][mat_id_j].heat_exchange);
                            sum_dTdt += m_j / fmaxf(rho_j, RHO_EPSILON) * (T_j - T_i) * lap_var * heat_boost;

                            // Exposure
                            float w_poly6_var = diff * diff * diff;
                            sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                            sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;

                            if (dye_on && behavior_j != STATIC) {
                                float vol_j = m_j / fmaxf(rho_j, RHO_EPSILON);
                                float lap_var2 = h - rlen;
                                float dye_factor = 0.01f * vol_j * c_precalc.viscosity_lap_coeff * lap_var2;
                                float4 dye_j = __ldg(&particle_dye_in[j]);
                                dye_rate.x += dye_factor * (dye_j.x - dye_i.x);
                                dye_rate.y += dye_factor * (dye_j.y - dye_i.y);
                                dye_rate.z += dye_factor * (dye_j.z - dye_i.z);
                            }
                        }

                        // Alpha factor (using Spiky gradient)
                        float3 gW = grad_spiky(r, rlen, h);
                        float w_j = m_j / fmaxf(rho_j, RHO_EPSILON);
                        grad_sum.x += w_j * gW.x;
                        grad_sum.y += w_j * gW.y;
                        grad_sum.z += w_j * gW.z;
                        grad_norm_sum += w_j * w_j * (gW.x * gW.x + gW.y * gW.y + gW.z * gW.z);

                        // Surface normal, vorticity, strain-rate: skip MAT_RIGID neighbors
                        if (!is_rigid_j) {
                            if (is_fluid && normal_out != 0) {
                                normal.x += w_j * gW.x;
                                normal.y += w_j * gW.y;
                                normal.z += w_j * gW.z;
                            }
                            neighbor_count += 1.0f;

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
                                    omega.x += vol_j * (gW.y * dvz - gW.z * dvy);
                                    omega.y += vol_j * (gW.z * dvx - gW.x * dvz);
                                    omega.z += vol_j * (gW.x * dvy - gW.y * dvx);
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
        density_out[i] = c_materials[mat_id_i].rest_density;
        alpha_out[i] = 0.0f;
        shear_rate_out[i] = 0.0f;
        // Still write heat/exposure outputs (computed above)
        float cp_i_s = c_materials[mat_id_i].heat_capacity;
        dTdt_out[i] = kappa_i * c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(c_materials[mat_id_i].rest_density * cp_i_s, 1.0f);
        exposure_heat_out[i] = c_precalc.poly6_coeff * sum_exposure_heat;
        exposure_corrode_out[i] = c_precalc.poly6_coeff * sum_exposure_corrode;
        if (dye_rate_out != 0) {
            dye_rate_out[i] = make_float4(dye_rate.x, dye_rate.y, dye_rate.z, 0.0f);
        }
        if (vorticity_out != 0) vorticity_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (normal_out != 0) normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, neighbor_count);
        return;
    }

    density_out[i] = rho;

    // Alpha = 1 / denom (inverse diagonal of pressure Poisson system)
    // Cap alpha to prevent surface particles (few neighbors) from getting
    // extreme correction values. Interior alpha ~ 1.1e-3 (denom ~ 915 at h=0.04, dx=0.02).
    // alpha_cap = alpha_limit * dt^2 controls max correction per iteration:
    //   alpha_limit=1.0 -> alpha_cap ~ 1.1e-5 at dt=1/300 (backward-compatible, conservative)
    //   alpha_limit=5.0 -> alpha_cap ~ 5.6e-5 (faster convergence, needs lower omega)
    float denom = grad_sum.x * grad_sum.x + grad_sum.y * grad_sum.y + grad_sum.z * grad_sum.z
                + grad_norm_sum;
    float alpha_threshold = 1e-6f;
    float dt = c_sim.dt;
    float alpha_cap = c_dfsph.alpha_limit * dt * dt;
    if (is_granular) {
        // Compliance factor 0.7: reduce alpha to allow some compression in granular piles.
        // Must multiply alpha (not denom) so that alpha gets SMALLER (more compliant).
        float alpha_raw = (denom > alpha_threshold) ? fminf(1.0f / denom, alpha_cap) : 0.0f;
        alpha_out[i] = alpha_raw * 0.7f;
    } else {
        alpha_out[i] = (denom > alpha_threshold) ? fminf(1.0f / denom, alpha_cap) : 0.0f;
    }
    // Heat diffusion: dTdt = kappa / (rho * cp) * viscosity_lap_coeff * sum_dTdt
    float cp_i = c_materials[mat_id_i].heat_capacity;
    dTdt_out[i] = kappa_i * c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(rho * cp_i, RHO_EPSILON);

    // Exposure
    exposure_heat_out[i] = c_precalc.poly6_coeff * sum_exposure_heat;
    exposure_corrode_out[i] = c_precalc.poly6_coeff * sum_exposure_corrode;

    // Dye diffusion output
    if (dye_rate_out != 0) {
        dye_rate_out[i] = make_float4(dye_rate.x, dye_rate.y, dye_rate.z, 0.0f);
    }

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
    float*          __restrict__ d_rigid_torques   // torque accumulator (NULL if no bodies)
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

    // mu(I) rheology for GRANULAR particles
    float eta_i = 0.0f;
    if (is_granular_i) {
        float gamma_dot_i = __ldg(&shear_rate[i]);
        float p_eff_i = fmaxf(compute_pressure(rho_i, behavior_i, mat_id_i), 1.0f);
        eta_i = compute_muI_eta(gamma_dot_i, p_eff_i, rho_i);
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

    // Compute viscosity + vorticity eta via neighbor loop
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

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq > h_sq || r_sq < 1e-12f) continue;

                    float rlen = sqrtf(r_sq);

                    // Vorticity eta accumulation (all neighbors -- PERF-002)
                    if (do_vort_eta) {
                        float h_rl = h - rlen;
                        float inv_rl = 1.0f / rlen;
                        float gs = c_precalc.spiky_grad_coeff * h_rl * h_rl * inv_rl;
                        float omega_j = __ldg(&vorticity_in[j]).w;
                        float wt = m_j / fmaxf(rho_j, RHO_EPSILON);
                        eta_vort.x += wt * omega_j * gs * r.x;
                        eta_vort.y += wt * omega_j * gs * r.y;
                        eta_vort.z += wt * omega_j * gs * r.z;
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
                            float visc_factor = mu_b * m_j * lap_v / fmaxf(rho_i, RHO_EPSILON);
                            float3 F_visc = make_float3(
                                (vel_boundary.x - vel_i.x) * visc_factor,
                                (vel_boundary.y - vel_i.y) * visc_factor,
                                (vel_boundary.z - vel_i.z) * visc_factor
                            );
                            f_visc.x += F_visc.x;
                            f_visc.y += F_visc.y;
                            f_visc.z += F_visc.z;
                            // Two-way: accumulate reaction on rigid body
                            float inv_rho = 1.0f / fmaxf(rho_i, RHO_EPSILON);
                            float3 F_on_body = make_float3(
                                -(F_visc.x * inv_rho) * m_j,
                                -(F_visc.y * inv_rho) * m_j,
                                -(F_visc.z * inv_rho) * m_j
                            );
                            float3 tau = make_float3(
                                r_b.y * F_on_body.z - r_b.z * F_on_body.y,
                                r_b.z * F_on_body.x - r_b.x * F_on_body.z,
                                r_b.x * F_on_body.y - r_b.y * F_on_body.x
                            );
                            warp_reduce_accumulate(d_rigid_forces, F_on_body, body_id);
                            warp_reduce_accumulate(d_rigid_torques, tau, body_id);
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
                        // GRANULAR-GRANULAR: mu(I) with harmonic mean eta
                        float dvx = vel_i.x - vel_j.x;
                        float dvy = vel_i.y - vel_j.y;
                        float dvz = vel_i.z - vel_j.z;
                        float gamma_dot_j = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz)
                                          / fmaxf(rlen, 1e-8f);
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        float p_eff_j = fmaxf(compute_pressure(rho_j, behavior_j, mat_id_j), 1.0f);
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
                        float mu_i = c_materials[mat_id_i].base_viscosity;
                        float mu_j = c_materials[mat_id_j].base_viscosity;
                        float mu_ij = 2.0f * mu_i * mu_j / (mu_i + mu_j + 1e-8f);
                        float visc_factor = mu_ij * c_precalc.viscosity_lap_coeff * m_j * lap_v / fmaxf(rho_j, RHO_EPSILON);
                        f_visc.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_visc.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_visc.z += (vel_j.z - vel_i.z) * visc_factor;
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

    // Akinci surface tension (FLUID only, surface particles)
    if (behavior_i == FLUID && normal_in != 0 && c_granular.surface_tension_gamma > 0.0f) {
        float4 norm_i = __ldg(&normal_in[i]);
        float nc_i = norm_i.w;  // neighbor count
        if (nc_i < 25.0f) {
            float gamma = c_granular.surface_tension_gamma;
            float n_mag = sqrtf(norm_i.x*norm_i.x + norm_i.y*norm_i.y + norm_i.z*norm_i.z);
            if (n_mag > 0.01f) {
                accel.x += -gamma * norm_i.x;
                accel.y += -gamma * norm_i.y;
                accel.z += -gamma * norm_i.z;
            }
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
                    div_v += (m_j / fmaxf(rho_j, RHO_EPSILON)) * dv_dot_gW;
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
                    float V_j = m_j / fmaxf(rho_j, RHO_EPSILON);
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

    // SDF object collision: clamp predicted position outside SDF surfaces
    for (int s = 0; s < c_num_sdf_objects; s++) {
        SDFObject obj = c_sdf_objects[s];
        float dist = eval_sdf(pred, obj);
        if (dist < BOUNDARY_MARGIN) {
            float3 n = sdf_normal(pred, obj);
            float push_d = BOUNDARY_MARGIN - dist;
            pred.x += push_d * n.x;
            pred.y += push_d * n.y;
            pred.z += push_d * n.z;
        }
    }

    predicted_out[i] = make_float4(pred.x, pred.y, pred.z, 1.0f);
}

/* ======================================================================
 * K_DFSPH_ComputeDensityAdv -- LEGACY / UNUSED
 *
 * Recomputes density at predicted positions using the ORIGINAL grid (stale
 * neighborhood). Superseded by the velocity-based Jacobi density solver
 * (ComputePressureAccel + DensitySolverUpdate + ApplyPressureVelocity) which
 * avoids stale-grid artifacts. Kept for reference only -- not called from
 * dfsph_solver.py.
 *
 * WARNING: Uses original cell lookup for predicted positions, which can miss
 * neighbors when particles cross cell boundaries during a substep.
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
    int behavior_i = GET_BEHAVIOR(pi_i);
    bool is_gas_i = (behavior_i == GAS);
    if (behavior_i == STATIC) {
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
                    int bj = GET_BEHAVIOR(pi_j);
                    if (is_gas_i != (bj == GAS) && j != i) continue;

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
    density_out[i] = fmaxf(rho, is_gas_i ? RHO_EPSILON : 1.0f);
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
 * K_DFSPH_ComputeKappaFromVelocity -- LEGACY / UNUSED
 *
 * Density prediction + kappa using drho/dt = SUM m_j (v_i - v_j) dot gradW_ij.
 * Superseded by the Jacobi p/rho^2 solver (ComputePressureAccel +
 * DensitySolverUpdate + ApplyPressureVelocity). Kept for reference only --
 * not called from dfsph_solver.py.
 *
 * NOTE: Uses m_j (not m_j/rho_j volume-weighted) unlike the active Jacobi
 * solver, so magnitudes differ by ~rho. Do not mix with Jacobi pipeline.
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
 * K_DFSPH_CorrectVelocityDens -- LEGACY / UNUSED
 *
 * Applies density correction using kappa from ComputeKappaFromVelocity.
 * Superseded by the Jacobi p/rho^2 solver. Kept for reference only --
 * not called from dfsph_solver.py.
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
                    float V_j = m_j / fmaxf(rho_j, RHO_EPSILON);
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
                    float V_j = m_j / fmaxf(rho_j, RHO_EPSILON);
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

                    float3 vt_j = make_float3(
                        vel4_j.x + dt * ap4_j.x,
                        vel4_j.y + dt * ap4_j.y,
                        vel4_j.z + dt * ap4_j.z
                    );

                    float3 gW = grad_spiky(r, rlen, h);
                    float dv_dot_gW = (vt_i.x - vt_j.x) * gW.x
                                    + (vt_i.y - vt_j.y) * gW.y
                                    + (vt_i.z - vt_j.z) * gW.z;
                    drho += (m_j / fmaxf(rho_j, RHO_EPSILON)) * dv_dot_gW;
                }
            }
        }
    }

    // Predicted density ratio (normalized to rest density)
    // Uses rho/rho0 - 1 formulation (matches SPlisHSPlasH reference)
    float density_adv = rho_i / rho0 + dt * drho;
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

    // Clamp pressure acceleration: limit velocity change to h/dt per substep
    // Prevents single-particle oscillation from Jacobi overcorrection
    float ap_sq = ap.x * ap.x + ap.y * ap.y + ap.z * ap.z;
    float a_press_max = c_sim.smoothing_length / (dt * dt);  // h/dt^2
    if (ap_sq > a_press_max * a_press_max) {
        float scale = a_press_max / sqrtf(ap_sq);
        ap.x *= scale;
        ap.y *= scale;
        ap.z *= scale;
    }

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
    const float4*   __restrict__ sorted_dye_rate,
    float4*         __restrict__ particle_dye_out,
    const float4*   __restrict__ sorted_angular_velocity,
    float4*         __restrict__ angular_velocity_out,
    const float4*   __restrict__ vorticity_in,          // (omega_x,y,z, |omega|) for micropolar, or NULL
    const float*    __restrict__ sorted_kappa_v,        // divergence kappa for warm-start writeback
    float*          __restrict__ kappa_v_out,           // unsorted kappa_v output (warm-start next frame)
    const RigidBody* __restrict__ d_rigid_bodies,       // rigid body state (NULL if no bodies)
    float*          __restrict__ d_rigid_forces,        // force accumulator (NULL if no bodies)
    float*          __restrict__ d_rigid_torques,       // torque accumulator (NULL if no bodies)
    uint*           __restrict__ max_displacement_out   // [1] atomicMax of displacement^2 (float-as-uint), or NULL
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
        if (kappa_v_out) kappa_v_out[orig_idx] = 0.0f;
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
            if (kappa_v_out) kappa_v_out[orig_idx] = 0.0f;
            particle_dye_out[orig_idx] = sorted_particle_dye[i];
            angular_velocity_out[orig_idx] = sorted_angular_velocity[i];
            return;
        }
        pi = CLEAR_SLEEPING(pi);
        pi = SET_JUST_WOKE(pi);
        sc = 0;
    }

    // XSPH velocity smoothing + rigid body Akinci pressure force accumulation
    float3 vel_advect = vel;
    if (behavior == FLUID || behavior == GRANULAR) {
        float h = c_sim.smoothing_length;
        float h_sq = c_sim.smoothing_length_sq;
        float rho_i = sorted_density[i];
        float3 xsph = make_float3(0.0f, 0.0f, 0.0f);

        // Akinci pressure mirroring: compute p_i for rigid body force accumulation
        float p_i = compute_pressure(rho_i, behavior, mat_id);
        float rho0_i = c_materials[mat_id].rest_density;

        int3 cell_i = get_cell(pos);
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                    uint start = cell_start[hash_c];
                    if (start == 0xFFFFFFFFu) continue;
                    uint end_idx = cell_end[hash_c];

                    for (uint j = start; j < end_idx; j++) {
                        if (j == i) continue;
                        uint pi_j = __ldg(&sorted_packed_info[j]);
                        int bj = GET_BEHAVIOR(pi_j);
                        if (bj == GAS) continue;

                        float4 p4j = __ldg(&sorted_position[j]);
                        float3 r = make_float3(pos.x - p4j.x, pos.y - p4j.y, pos.z - p4j.z);
                        float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                        if (r_sq > h_sq || r_sq < 1e-12f) continue;

                        float m_j = __ldg(&sorted_mass[j]);

                        // Akinci pressure force on rigid body (two-way coupling)
                        if (bj == STATIC && d_rigid_bodies != 0) {
                            uint mat_id_j = GET_MATERIAL_ID(pi_j);
                            if (mat_id_j == MAT_RIGID) {
                                float rlen = sqrtf(r_sq);
                                float3 gW = grad_spiky(r, rlen, h);
                                float psi_b = m_j;
                                float press_akinci = (p_i / (rho_i * rho_i)) + (p_i / (rho0_i * rho0_i));
                                // Force on fluid from boundary (acceleration * mass)
                                float m_i = c_sim.particle_mass;
                                float3 F_on_fluid = make_float3(
                                    m_i * psi_b * press_akinci * gW.x,
                                    m_i * psi_b * press_akinci * gW.y,
                                    m_i * psi_b * press_akinci * gW.z
                                );
                                // Newton's 3rd law: reaction on body
                                int body_id = GET_BODY_ID(pi_j);
                                float4 rb_pos = __ldg(&d_rigid_bodies[body_id].position);
                                float3 r_b = make_float3(
                                    p4j.x - rb_pos.x, p4j.y - rb_pos.y, p4j.z - rb_pos.z
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

                        float4 vj4 = __ldg(&sorted_velocity[j]);
                        float rho_j = __ldg(&sorted_density[j]);

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
        float c_xsph = c_granular.xsph_epsilon;
        if (behavior == GRANULAR) c_xsph *= 10.0f;
        vel_advect.x += c_xsph * xsph.x;
        vel_advect.y += c_xsph * xsph.y;
        vel_advect.z += c_xsph * xsph.z;
    }

    // Final position: x_final = x + dt * v_advect (XSPH-corrected)
    float3 pos_new = make_float3(
        pos.x + dt * vel_advect.x,
        pos.y + dt * vel_advect.y,
        pos.z + dt * vel_advect.z
    );

    // STATIC particle boundary repulsion
    static_particle_boundary(
        pos_new, vel,
        cell_start, cell_end,
        sorted_packed_info, sorted_position,
        i, c_sim.restitution
    );

    // Box boundary
    sdf_box_boundary(pos_new, vel, c_sim.world_min, c_sim.world_max,
                     c_sim.restitution, c_sim.wall_friction);

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

    // Dye update: dye += dye_rate * dt
    float4 dye = sorted_particle_dye[i];
    if (sorted_dye_rate != 0) {
        float4 drate = sorted_dye_rate[i];
        dye.x = fmaxf(0.0f, fminf(1.0f, dye.x + drate.x * dt));
        dye.y = fmaxf(0.0f, fminf(1.0f, dye.y + drate.y * dt));
        dye.z = fmaxf(0.0f, fminf(1.0f, dye.z + drate.z * dt));
    }

    // Track max displacement for grid reuse (Phase 9.2 / PERF-009)
    // displacement^2 = |pos_new - pos_old|^2
    if (max_displacement_out) {
        float dx_d = pos_new.x - pos.x;
        float dy_d = pos_new.y - pos.y;
        float dz_d = pos_new.z - pos.z;
        float disp_sq = dx_d * dx_d + dy_d * dy_d + dz_d * dz_d;
        atomicMax(max_displacement_out, __float_as_uint(disp_sq));
    }

    // Writeback
    position_out[orig_idx] = make_float4(pos_new.x, pos_new.y, pos_new.z, 1.0f);
    velocity_out[orig_idx] = make_float4(vel.x, vel.y, vel.z, 0.0f);
    color_out[orig_idx] = color;
    packed_info_out[orig_idx] = pi;
    sleep_counter_out[orig_idx] = sc;
    temperature_out[orig_idx] = temp;
    kappa_out[orig_idx] = sorted_kappa[i];
    if (kappa_v_out) kappa_v_out[orig_idx] = sorted_kappa_v[i];
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
