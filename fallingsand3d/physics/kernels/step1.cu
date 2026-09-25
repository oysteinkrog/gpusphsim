/*
 * step1.cu -- K_Step1 SPH density summation + strain-rate tensor + heat diffusion
 *             + exposure accumulation kernel.
 *
 * Per-particle computation:
 *   density_sum += m_j * (h^2 - |r_ij|^2)^3   for all neighbors j within h
 *   density_i = max(1.0, poly6_coeff * density_sum)
 *
 * Heat diffusion (all particles):
 *   dTdt_i = viscosity_lap_coeff / (rho_i cp_i)
 *            * sum_j k_ij (m_j/rho_j) * (T_j - T_i) * (h - |r_ij|) * heat_boost_ij
 *   where k_ij = 2 kappa_i kappa_j / (kappa_i + kappa_j) (harmonic mean of
 *   c_materials[].thermal_conductivity), so each pair exchanges heat symmetrically.
 *   GAS/non-GAS pairs use the same term with the pair coefficient capped so
 *   one neighbour moves T_i by at most half the gap per substep
 *   (bd-r5-review-n28.94); MAT_RIGID neighbours never exchange heat.
 *
 * Exposure accumulation (all particles, via c_interactions table lookup),
 * normalised by the kernel support weight of the neighbourhood:
 *   exposure_corrode_i = sum_j rate_ij * W_ij / sum_j W_ij
 *   exposure_heat_i    = sum_j heat_exchange_ij * max(T_j - T_i, 0) * W_ij / sum_j W_ij
 *   sum_j W_ij runs over every non-self, non-DEAD neighbour in range,
 *   including cross-phase GAS and MAT_RIGID neighbours.
 *   No if/else branching on material type -- pure table lookup.
 *
 * For GRANULAR particles only, also computes the symmetric strain-rate tensor D
 * using the SPH velocity gradient with spiky gradient weighting:
 *   D_ab = 0.5 * sum_j (m_j/rho_j) * (dv_a * gradW_b + dv_b * gradW_a)
 * Then: gamma_dot = sqrt(2 * D:D) = sqrt(2 * (Dxx^2 + Dyy^2 + Dzz^2 + 2*(Dxy^2 + Dxz^2 + Dyz^2)))
 *
 * Operates on SORTED particle arrays (after hash + sort + reorder).
 * Uses 27-cell neighbor iteration with grid cell_start/cell_end tables.
 * Self-interaction IS included for density (j==i NOT skipped).
 * Self-interaction is skipped for strain-rate, heat diffusion, and exposure (j==i skipped).
 * Per-particle mass m_j supports multi-material and mass splitting.
 *
 * Ported from SPHSimLib/K_SimpleSPH_Step1.inl + K_SPH_Kernels_poly6.inl.
 * Neighbor iteration ported from SPHSimLib/K_UniformGrid_Utils.inl.
 *
 * Constant memory (c_grid, c_sim, c_precalc, c_materials, c_interactions) declared in common.cuh.
 */

#include "sph_shared.cuh"

extern "C" __global__ __launch_bounds__(256, 4)
void K_Step1(
    uint            numParticles,
    float4*         __restrict__ position,      // sorted positions (writes density to .w -- PERF-004)
    const float4*   __restrict__ velocity,      // sorted velocities (for strain-rate)
    const float*    __restrict__ mass,           // sorted per-particle mass
    const float*    __restrict__ density_in,     // sorted density from previous step (for strain-rate m_j/rho_j weighting; NULL on first frame)
    const uint*     __restrict__ packed_info,    // sorted packed_info (for behavior class check)
    const float*    __restrict__ temperature_in, // sorted temperature (for heat diffusion)
    const uint*     __restrict__ cell_start,     // grid cell start indices
    const uint*     __restrict__ cell_end,       // grid cell end indices
    float*          __restrict__ density_out,    // output: density per particle
    float*          __restrict__ shear_rate_out, // output: gamma_dot per particle (0 for non-GRANULAR)
    float*          __restrict__ dTdt_out,       // output: temperature rate of change (heat diffusion)
    float*          __restrict__ exposure_heat_out,    // output: heat exposure from interactions
    float*          __restrict__ exposure_corrode_out, // output: corrosion exposure from interactions
    float4*         __restrict__ vorticity_out,        // output: (omega_x, omega_y, omega_z, |omega|) -- FLUID only
    float4*         __restrict__ normal_out,           // output: (n_x, n_y, n_z, neighbor_count) -- FLUID only
    const float4*   __restrict__ particle_dye_in,      // input: particle dye color (r, g, b, unused)
    float4*         __restrict__ dye_rate_out,         // output: dye diffusion rate (dr, dg, db, unused)
    const void*     __restrict__ velocity_h,           // FP16 velocity for neighbor reads (OPT-4.3), may be NULL
    float*          __restrict__ pressure_out,         // output: pre-computed pressure per particle (PERF-007)
    const void*     __restrict__ temperature_h,        // FP16 temperature for neighbor reads (PERF-011), may be NULL
    const void*     __restrict__ dye_h                 // FP16 dye for neighbor reads (PERF-011), may be NULL
) {
    uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_i >= numParticles) return;

    // Const alias for __ldg neighbor reads (position is non-const for .w write -- PERF-004)
    const float4* __restrict__ position_ro = position;

    // Read position of particle i
    float4 pos4_i = position[index_i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h    = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;

    // Check if this particle is GRANULAR
    uint pi_i = __ldg(&packed_info[index_i]);
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);
    bool is_granular = (behavior_i == GRANULAR);
    bool is_gas_i = (behavior_i == GAS);

    bool is_fluid = (behavior_i == FLUID);
    bool is_rigid_i = (mat_id_i == MAT_RIGID);
    // Rigid neighbour density scale: psi_b = rho0_i * V_b (bd-r5-review-n28.77)
    float rigid_dens_scale = __fdiv_rn(c_materials[mat_id_i].rest_density, RIGID_PSI_RHO0);

    // Boundary particles don't need density computation -- early out
    if (is_rigid_i) {
        density_out[index_i] = RIGID_PSI_RHO0;  // rest density placeholder (V_j = m_j / rho_j = V_b)
        pressure_out[index_i] = 0.0f;
        shear_rate_out[index_i] = 0.0f;
        dTdt_out[index_i] = 0.0f;
        exposure_heat_out[index_i] = 0.0f;
        exposure_corrode_out[index_i] = 0.0f;
        vorticity_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        normal_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        dye_rate_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        position[index_i].w = RIGID_PSI_RHO0;  // pack the placeholder density like other particles
        return;
    }

    // Read velocity for strain-rate (GRANULAR) and vorticity (FLUID).
    // FP16 pairing rule (bd-r5-review-n28.69): vel_i and dye_i are used only
    // in pair differences against neighbours read from the FP16 mirrors, so
    // they come from the same mirrors. Co-moving particles then give an exact
    // zero difference instead of fp16(v) - v.
    float4 vel4_i = velocity_h ? load_half4((const uint2*)velocity_h + index_i)
                               : velocity[index_i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);

    // Read dye color for diffusion (skip when dye disabled -- PERF-005)
    bool dye_on = (c_sim.dye_enabled != 0);
    float4 dye_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (dye_on) dye_i = dye_h ? load_half4((const uint2*)dye_h + index_i)
                              : __ldg(&particle_dye_in[index_i]);

    // Read temperature for heat diffusion
    float T_i = __ldg(&temperature_in[index_i]);
    float kappa_i = c_materials[mat_id_i].thermal_conductivity;
    // Cross-phase heat cap (bd-r5-review-n28.94). The self term bounds rho_i from
    // below, so capping a cross-phase pair coefficient at a_x_max keeps
    // viscosity_lap_coeff * a / (rho_i * cp_i) * dt <= 0.5 for that pair.
    float rho_i_lb = fmaxf(c_precalc.poly6_coeff * __ldg(&mass[index_i]) * h_sq * h_sq * h_sq,
                           is_gas_i ? RHO_EPSILON : 1.0f);
    float a_x_max = 0.5f * rho_i_lb * c_materials[mat_id_i].heat_capacity
                  / fmaxf(c_precalc.viscosity_lap_coeff * c_sim.dt, 1e-30f);

    // Density accumulator (variable part of Poly6 kernel)
    float sum_density = 0.0f;

    // Heat diffusion accumulator: dTdt
    float sum_dTdt = 0.0f;

    // Exposure accumulators (from c_interactions table lookup)
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;
    // Kernel support weight of all non-self, non-DEAD neighbours: the
    // exposure sums are divided by it, so exposure is a support fraction.
    float sum_W = 0.0f;

    // Strain-rate tensor accumulators (6 symmetric components)
    // D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
    float Dxx = 0.0f, Dyy = 0.0f, Dzz = 0.0f;
    float Dxy = 0.0f, Dxz = 0.0f, Dyz = 0.0f;

    // Vorticity accumulator (FLUID only): omega = curl(v)
    float3 omega = make_float3(0.0f, 0.0f, 0.0f);

    // Surface normal accumulator (FLUID only): n = grad(color field)
    float3 normal = make_float3(0.0f, 0.0f, 0.0f);
    float neighbor_count = 0.0f;

    // Dye diffusion rate accumulator
    float3 dye_rate = make_float3(0.0f, 0.0f, 0.0f);

    // Grid cell of particle i
    int3 cell_i = calcGridCell(make_float3(pos_i.x, pos_i.y, pos_i.z));

    // Iterate 27 neighbor cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash];

                // Empty cell sentinel
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash];

                for (uint index_j = start; index_j < end_idx; index_j++) {
                    // --- Speculative ILP: issue ALL loads before distance check ---
                    float4 pos4_j = __ldg(&position_ro[index_j]);
                    float m_j = __ldg(&mass[index_j]);
                    uint pi_j = __ldg(&packed_info[index_j]);
                    float T_j = temperature_h ? load_half1((const __half*)temperature_h + index_j)
                                              : __ldg(&temperature_in[index_j]);

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;

                    if (r_sq <= h_sq) {
                        float diff = h_sq - r_sq;

                        int behavior_j = GET_BEHAVIOR(pi_j);

                        // GAS/non-GAS phase separation: skip density/heat/forces,
                        // keep only exposure (so fire-to-wood ignition still works)
                        bool is_gas_j = (behavior_j == GAS);
                        if (is_gas_i != is_gas_j && index_j != index_i) {
                            uint mat_id_j = GET_MATERIAL_ID(pi_j);
                            float w_poly6_var = diff * diff * diff;
                            if (mat_id_j != MAT_DEAD) sum_W += w_poly6_var;
                            // A rigid wall dilutes gas exposure but never reacts with it
                            // (same rule as the same-phase branch, bd-r5-review-n28.66).
                            if (mat_id_j != MAT_RIGID) {
                                sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                                sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
                                // Cross-phase heat conduction (bd-r5-review-n28.94): the same symmetric
                                // pair term as the same-phase branch, with the coefficient capped at a_x_max
                                // so one neighbour moves T_i by at most half the gap per substep.
                                float rho_x = (density_in != 0) ? __ldg(&density_in[index_j]) : 0.0f;
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

                        // --- Density: self-interaction included ---
                        // MAT_RIGID mass is RIGID_PSI_RHO0 * V_b; Akinci weights it
                        // with this particle's rest density (bd-r5-review-n28.77).
                        float m_dens = (GET_MATERIAL_ID(pi_j) == MAT_RIGID) ? m_j * rigid_dens_scale : m_j;
                        sum_density += m_dens * diff * diff * diff;

                        // --- All non-self computations: heat, exposure, vorticity, strain-rate ---
                        if (index_j != index_i) {
                            uint mat_id_j = GET_MATERIAL_ID(pi_j);

                            // Single sqrtf per neighbor (PERF-003)
                            float rlen = sqrtf(r_sq);

                            // Exposure support weight counts MAT_RIGID too, so a wall
                            // dilutes exposure like any other neighbour.
                            if (mat_id_j != MAT_DEAD) sum_W += diff * diff * diff;

                            // Skip MAT_RIGID neighbors for heat/exposure/vorticity/strain
                            // (density contribution already included via mass=psi_b; the else
                            // branch below still adds them to the normal and neighbour count)
                            if (mat_id_j != MAT_RIGID) {
                                // --- Heat diffusion ---
                                float rho_j = (density_in != 0) ? __ldg(&density_in[index_j]) : 0.0f;
                                // Unset or zero density_in (spawn, unused slot): use the neighbour's rest density (bd-r5-review-n28.110)
                                if (rho_j <= 0.0f) rho_j = fmaxf(c_materials[GET_MATERIAL_ID(pi_j)].rest_density, RHO_EPSILON);
                                float lap_var = h - rlen;
                                float heat_boost = fmaxf(1.0f, c_interactions[mat_id_i][mat_id_j].heat_exchange);
                                // Pairwise harmonic-mean conductivity: the pair term is symmetric,
                                // so heat lost by i equals heat gained by j (bd-r5-review-n28.67).
                                float kappa_j = c_materials[mat_id_j].thermal_conductivity;
                                float k_ij = 2.0f * kappa_i * kappa_j / fmaxf(kappa_i + kappa_j, 1e-6f);
                                sum_dTdt += k_ij * m_j / fmaxf(rho_j, RHO_EPSILON) * (T_j - T_i) * lap_var * heat_boost;

                                // --- Exposure accumulation ---
                                float w_poly6_var = diff * diff * diff;
                                sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                                sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;

                                // --- Gradient-based computations ---
                                if (r_sq > 1e-12f) {
                                    float inv_rlen = 1.0f / rlen;

                                    float h_rlen = h - rlen;
                                    float grad_scalar = c_precalc.spiky_grad_coeff * h_rlen * h_rlen * inv_rlen;
                                    float gWx = grad_scalar * r.x;
                                    float gWy = grad_scalar * r.y;
                                    float gWz = grad_scalar * r.z;

                                    float vol_j = m_j / fmaxf(rho_j, RHO_EPSILON);

                                    if (is_fluid) {
                                        float4 vel4_j_v = velocity_h ? load_half4((const uint2*)velocity_h + index_j)
                                                                     : __ldg(&velocity[index_j]);
                                        float dvx_v = vel4_j_v.x - vel_i.x;
                                        float dvy_v = vel4_j_v.y - vel_i.y;
                                        float dvz_v = vel4_j_v.z - vel_i.z;
                                        // curl_i = sum_j V_j gradW x (v_j - v_i)
                                        omega.x += vol_j * (gWy * dvz_v - gWz * dvy_v);
                                        omega.y += vol_j * (gWz * dvx_v - gWx * dvz_v);
                                        omega.z += vol_j * (gWx * dvy_v - gWy * dvx_v);

                                        normal.x += vol_j * gWx;
                                        normal.y += vol_j * gWy;
                                        normal.z += vol_j * gWz;
                                    }

                                    if (m_j > 0.0f) neighbor_count += 1.0f;  // DEAD slots have mass 0

                                    if (dye_on && behavior_j != STATIC) {
                                        float dye_factor = 0.01f * vol_j * c_precalc.viscosity_lap_coeff * lap_var;
                                        float4 dye_j = dye_h ? load_half4((const uint2*)dye_h + index_j)
                                                       : __ldg(&particle_dye_in[index_j]);
                                        dye_rate.x += dye_factor * (dye_j.x - dye_i.x);
                                        dye_rate.y += dye_factor * (dye_j.y - dye_i.y);
                                        dye_rate.z += dye_factor * (dye_j.z - dye_i.z);
                                    }

                                    if (is_granular) {
                                        float4 vel4_j = velocity_h ? load_half4((const uint2*)velocity_h + index_j)
                                                                   : __ldg(&velocity[index_j]);
                                        float dvx = vel_i.x - vel4_j.x;
                                        float dvy = vel_i.y - vel4_j.y;
                                        float dvz = vel_i.z - vel4_j.z;
                                        float weight = m_j / fmaxf(rho_j, RHO_EPSILON);
                                        Dxx += weight * dvx * gWx;
                                        Dyy += weight * dvy * gWy;
                                        Dzz += weight * dvz * gWz;
                                        Dxy += 0.5f * weight * (dvx * gWy + dvy * gWx);
                                        Dxz += 0.5f * weight * (dvx * gWz + dvz * gWx);
                                        Dyz += 0.5f * weight * (dvy * gWz + dvz * gWy);
                                    }
                                }
                            } else if (r_sq > 1e-12f && m_j > 0.0f) {
                                // MAT_RIGID neighbour: count it and add it to the colour-field
                                // normal like a STONE wall, so fluid at a rigid face is not
                                // classed as surface (bd-r5-review-n28.68). The mass slot holds
                                // RIGID_PSI_RHO0 * V_b (bd-r5-review-n28.77).
                                float inv_rlen = 1.0f / rlen;
                                float h_rlen = h - rlen;
                                float grad_scalar = c_precalc.spiky_grad_coeff * h_rlen * h_rlen * inv_rlen;
                                if (is_fluid) {
                                    float vol_b = m_j / RIGID_PSI_RHO0;
                                    normal.x += vol_b * grad_scalar * r.x;
                                    normal.y += vol_b * grad_scalar * r.y;
                                    normal.z += vol_b * grad_scalar * r.z;
                                }
                                neighbor_count += 1.0f;
                            } // mat_id_j != MAT_RIGID
                        }
                    }
                }
            }
        }
    }

    // --- PostCalc: density ---
    // Safety floor only: a live particle's own Poly6 self term (163-490 for
    // GAS, whose rho0 is 1000-1200 in kernel-sum units, bd-r5-review-n28.80)
    // is always above either floor.
    float density = c_precalc.poly6_coeff * sum_density;
    float rho_floor = is_gas_i ? RHO_EPSILON : 1.0f;
    // STATIC walls store rest_density, as PBF and DFSPH do, so every solver
    // gives a wall neighbour the same V_j = m_j / rho_j in heat, normal and
    // dye sums (bd-r5-review-n28.116). Floor at 1: DEAD has rest_density 0.
    // The STATIC dTdt divisor below reads the same value, so the heat pair
    // term stays conservative. Step2 mirrors p_i, so wall forces do not change.
    density_out[index_i] = (behavior_i == STATIC)
        ? fmaxf(c_materials[mat_id_i].rest_density, 1.0f)
        : fmaxf(density, rho_floor);

    // --- PostCalc: pressure (PERF-007) ---
    pressure_out[index_i] = compute_pressure(density_out[index_i], behavior_i, mat_id_i);

    // --- PostCalc: heat diffusion dTdt ---
    // dTdt = viscosity_lap_coeff * sum_dTdt / (rho_i * cp_i); k_ij is inside sum_dTdt
    float cp_i = c_materials[mat_id_i].heat_capacity;
    float rho_i_heat = density_out[index_i];
    dTdt_out[index_i] = c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(rho_i_heat * cp_i, RHO_EPSILON);

    // --- PostCalc: exposure accumulation ---
    // Normalise by the neighbour support weight (poly6_coeff cancels):
    // exposure_corrode is in [0, max reaction_rate] (1/s), exposure_heat is
    // heat_exchange * K weighted by the fraction of the support that is hotter.
    float inv_sum_W = (sum_W > 0.0f) ? 1.0f / sum_W : 0.0f;
    exposure_heat_out[index_i] = sum_exposure_heat * inv_sum_W;
    exposure_corrode_out[index_i] = sum_exposure_corrode * inv_sum_W;

    // --- PostCalc: vorticity ---
    float omega_mag = sqrtf(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z);
    vorticity_out[index_i] = make_float4(omega.x, omega.y, omega.z, omega_mag);

    // --- PostCalc: surface normal + neighbor count ---
    normal_out[index_i] = make_float4(normal.x, normal.y, normal.z, neighbor_count);

    // --- PostCalc: dye diffusion rate (skip when disabled -- PERF-005) ---
    if (dye_on) {
        dye_rate_out[index_i] = make_float4(dye_rate.x, dye_rate.y, dye_rate.z, 0.0f);
    }

    // --- PostCalc: strain-rate magnitude (gamma_dot) ---
    if (is_granular) {
        // gamma_dot = sqrt(2 * D:D)
        //           = sqrt(2 * (Dxx^2 + Dyy^2 + Dzz^2 + 2*(Dxy^2 + Dxz^2 + Dyz^2)))
        float D_sq = Dxx * Dxx + Dyy * Dyy + Dzz * Dzz
                   + 2.0f * (Dxy * Dxy + Dxz * Dxz + Dyz * Dyz);
        float gamma_dot = sqrtf(fmaxf(2.0f * D_sq, 0.0f));
        shear_rate_out[index_i] = gamma_dot;
    } else {
        shear_rate_out[index_i] = 0.0f;
    }

    // Pack density into position.w for Step2 neighbor reads (PERF-004)
    position[index_i].w = density_out[index_i];
}


/* ======================================================================
 * K_PackDensity -- Pack density into sorted_position.w (kept for compatibility)
 *
 * Runs between Step1 and Step2. After this, Step2 can read
 * density_j = position[j].w instead of a separate density array load.
 * Eliminates one global memory load per neighbor interaction.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256)
void K_PackDensity(
    float4*       __restrict__ position,    // sorted positions (write .w)
    const float*  __restrict__ density,     // density computed by Step1
    uint          numParticles
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    position[i].w = density[i];
}


/* ======================================================================
 * K_Step1_BuildNL -- Step1 fused with neighbor list build.
 *
 * Identical to K_Step1 but also writes compact neighbor indices
 * as it scans the 27-cell grid. The neighbor list is then used
 * by K_Step2_NL, avoiding a second grid scan.
 *
 * Extra parameters vs K_Step1:
 *   neighbor_indices_out[N * max_nb] -- output neighbor indices
 *   neighbor_count_out[N]            -- output neighbor count per particle
 *   max_nb                           -- max neighbors per particle
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_Step1_BuildNL(
    uint            numParticles,
    float4*         __restrict__ position,
    const float4*   __restrict__ velocity,
    const float*    __restrict__ mass,
    const float*    __restrict__ density_in,
    const uint*     __restrict__ packed_info,
    const float*    __restrict__ temperature_in,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float*          __restrict__ density_out,
    float*          __restrict__ shear_rate_out,
    float*          __restrict__ dTdt_out,
    float*          __restrict__ exposure_heat_out,
    float*          __restrict__ exposure_corrode_out,
    float4*         __restrict__ vorticity_out,
    float4*         __restrict__ normal_out,
    const float4*   __restrict__ particle_dye_in,
    float4*         __restrict__ dye_rate_out,
    const void*     __restrict__ velocity_h,
    float*          __restrict__ pressure_out,
    const void*     __restrict__ temperature_h,
    const void*     __restrict__ dye_h,
    // Neighbor list outputs
    uint*           __restrict__ neighbor_indices_out,
    uint*           __restrict__ neighbor_count_out,
    uint            max_nb
) {
    uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_i >= numParticles) return;

    const float4* __restrict__ position_ro = position;

    float4 pos4_i = position[index_i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h    = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;

    uint pi_i = __ldg(&packed_info[index_i]);
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);
    bool is_granular = (behavior_i == GRANULAR);
    bool is_gas_i = (behavior_i == GAS);
    bool is_fluid = (behavior_i == FLUID);
    bool is_rigid_i = (mat_id_i == MAT_RIGID);
    // Rigid neighbour density scale: psi_b = rho0_i * V_b (bd-r5-review-n28.77)
    float rigid_dens_scale = __fdiv_rn(c_materials[mat_id_i].rest_density, RIGID_PSI_RHO0);

    // Neighbor list write position
    uint nl_base = index_i * max_nb;
    uint nl_count = 0;

    if (is_rigid_i) {
        density_out[index_i] = RIGID_PSI_RHO0;
        pressure_out[index_i] = 0.0f;
        shear_rate_out[index_i] = 0.0f;
        dTdt_out[index_i] = 0.0f;
        exposure_heat_out[index_i] = 0.0f;
        exposure_corrode_out[index_i] = 0.0f;
        vorticity_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        normal_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        dye_rate_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        position[index_i].w = RIGID_PSI_RHO0;  // pack the placeholder density like other particles
        neighbor_count_out[index_i] = 0;
        return;
    }

    // FP16 pairing rule (bd-r5-review-n28.69): same as K_Step1 above.
    float4 vel4_i = velocity_h ? load_half4((const uint2*)velocity_h + index_i)
                               : velocity[index_i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);

    bool dye_on = (c_sim.dye_enabled != 0);
    float4 dye_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (dye_on) dye_i = dye_h ? load_half4((const uint2*)dye_h + index_i)
                              : __ldg(&particle_dye_in[index_i]);

    float T_i = __ldg(&temperature_in[index_i]);
    float kappa_i = c_materials[mat_id_i].thermal_conductivity;
    // Cross-phase heat cap (bd-r5-review-n28.94). The self term bounds rho_i from
    // below, so capping a cross-phase pair coefficient at a_x_max keeps
    // viscosity_lap_coeff * a / (rho_i * cp_i) * dt <= 0.5 for that pair.
    float rho_i_lb = fmaxf(c_precalc.poly6_coeff * __ldg(&mass[index_i]) * h_sq * h_sq * h_sq,
                           is_gas_i ? RHO_EPSILON : 1.0f);
    float a_x_max = 0.5f * rho_i_lb * c_materials[mat_id_i].heat_capacity
                  / fmaxf(c_precalc.viscosity_lap_coeff * c_sim.dt, 1e-30f);

    float sum_density = 0.0f;
    float sum_dTdt = 0.0f;
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;
    // Kernel support weight of all non-self, non-DEAD neighbours: the
    // exposure sums are divided by it, so exposure is a support fraction.
    float sum_W = 0.0f;
    float Dxx = 0.0f, Dyy = 0.0f, Dzz = 0.0f;
    float Dxy = 0.0f, Dxz = 0.0f, Dyz = 0.0f;
    float3 omega = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal = make_float3(0.0f, 0.0f, 0.0f);
    float neighbor_count = 0.0f;
    float3 dye_rate = make_float3(0.0f, 0.0f, 0.0f);

    int3 cell_i = calcGridCell(make_float3(pos_i.x, pos_i.y, pos_i.z));

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash];

                for (uint index_j = start; index_j < end_idx; index_j++) {
                    float4 pos4_j = __ldg(&position_ro[index_j]);
                    float m_j = __ldg(&mass[index_j]);
                    uint pi_j = __ldg(&packed_info[index_j]);
                    float T_j = temperature_h ? load_half1((const __half*)temperature_h + index_j)
                                              : __ldg(&temperature_in[index_j]);

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;

                    if (r_sq <= h_sq) {
                        float diff = h_sq - r_sq;
                        int behavior_j = GET_BEHAVIOR(pi_j);
                        bool is_gas_j = (behavior_j == GAS);

                        if (is_gas_i != is_gas_j && index_j != index_i) {
                            uint mat_id_j = GET_MATERIAL_ID(pi_j);
                            float w_poly6_var = diff * diff * diff;
                            if (mat_id_j != MAT_DEAD) sum_W += w_poly6_var;
                            // A rigid wall dilutes gas exposure but never reacts with it
                            // (same rule as the same-phase branch, bd-r5-review-n28.66).
                            if (mat_id_j != MAT_RIGID) {
                                sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                                sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
                                // Cross-phase heat conduction (bd-r5-review-n28.94): the same symmetric
                                // pair term as the same-phase branch, with the coefficient capped at a_x_max
                                // so one neighbour moves T_i by at most half the gap per substep.
                                float rho_x = (density_in != 0) ? __ldg(&density_in[index_j]) : 0.0f;
                                // Unset or zero density_in (spawn, unused slot): use the neighbour's rest density (bd-r5-review-n28.110)
                                if (rho_x <= 0.0f) rho_x = fmaxf(c_materials[GET_MATERIAL_ID(pi_j)].rest_density, RHO_EPSILON);
                                float kappa_x = c_materials[mat_id_j].thermal_conductivity;
                                float k_x = 2.0f * kappa_i * kappa_x / fmaxf(kappa_i + kappa_x, 1e-6f);
                                float boost_x = fmaxf(1.0f, c_interactions[mat_id_i][mat_id_j].heat_exchange);
                                float a_x = k_x * m_j / fmaxf(rho_x, RHO_EPSILON) * (h - sqrtf(r_sq)) * boost_x;
                                sum_dTdt += fminf(a_x, a_x_max) * (T_j - T_i);
                            }
                            // Still record as neighbor for step2
                            if (nl_count < max_nb && index_j != index_i) {
                                neighbor_indices_out[nl_base + nl_count] = index_j;
                                nl_count++;
                            }
                            continue;
                        }

                        // MAT_RIGID mass is RIGID_PSI_RHO0 * V_b; Akinci weights it
                        // with this particle's rest density (bd-r5-review-n28.77).
                        float m_dens = (GET_MATERIAL_ID(pi_j) == MAT_RIGID) ? m_j * rigid_dens_scale : m_j;
                        sum_density += m_dens * diff * diff * diff;

                        if (index_j != index_i) {
                            // Record neighbor for step2
                            if (nl_count < max_nb) {
                                neighbor_indices_out[nl_base + nl_count] = index_j;
                                nl_count++;
                            }

                            uint mat_id_j = GET_MATERIAL_ID(pi_j);
                            float rlen = sqrtf(r_sq);

                            if (mat_id_j != MAT_DEAD) sum_W += diff * diff * diff;

                            if (mat_id_j != MAT_RIGID) {
                                float rho_j = (density_in != 0) ? __ldg(&density_in[index_j]) : 0.0f;
                                // Unset or zero density_in (spawn, unused slot): use the neighbour's rest density (bd-r5-review-n28.110)
                                if (rho_j <= 0.0f) rho_j = fmaxf(c_materials[GET_MATERIAL_ID(pi_j)].rest_density, RHO_EPSILON);
                                float lap_var = h - rlen;
                                float heat_boost = fmaxf(1.0f, c_interactions[mat_id_i][mat_id_j].heat_exchange);
                                // Pairwise harmonic-mean conductivity: the pair term is symmetric,
                                // so heat lost by i equals heat gained by j (bd-r5-review-n28.67).
                                float kappa_j = c_materials[mat_id_j].thermal_conductivity;
                                float k_ij = 2.0f * kappa_i * kappa_j / fmaxf(kappa_i + kappa_j, 1e-6f);
                                sum_dTdt += k_ij * m_j / fmaxf(rho_j, RHO_EPSILON) * (T_j - T_i) * lap_var * heat_boost;

                                float w_poly6_var = diff * diff * diff;
                                sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                                sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;

                                if (r_sq > 1e-12f) {
                                    float inv_rlen = 1.0f / rlen;
                                    float h_rlen = h - rlen;
                                    float grad_scalar = c_precalc.spiky_grad_coeff * h_rlen * h_rlen * inv_rlen;
                                    float gWx = grad_scalar * r.x;
                                    float gWy = grad_scalar * r.y;
                                    float gWz = grad_scalar * r.z;
                                    float vol_j = m_j / fmaxf(rho_j, RHO_EPSILON);

                                    if (is_fluid) {
                                        float4 vel4_j_v = velocity_h ? load_half4((const uint2*)velocity_h + index_j)
                                                                     : __ldg(&velocity[index_j]);
                                        float dvx_v = vel4_j_v.x - vel_i.x;
                                        float dvy_v = vel4_j_v.y - vel_i.y;
                                        float dvz_v = vel4_j_v.z - vel_i.z;
                                        // curl_i = sum_j V_j gradW x (v_j - v_i)
                                        omega.x += vol_j * (gWy * dvz_v - gWz * dvy_v);
                                        omega.y += vol_j * (gWz * dvx_v - gWx * dvz_v);
                                        omega.z += vol_j * (gWx * dvy_v - gWy * dvx_v);

                                        normal.x += vol_j * gWx;
                                        normal.y += vol_j * gWy;
                                        normal.z += vol_j * gWz;
                                    }

                                    if (m_j > 0.0f) neighbor_count += 1.0f;  // DEAD slots have mass 0

                                    if (dye_on && behavior_j != STATIC) {
                                        float dye_factor = 0.01f * vol_j * c_precalc.viscosity_lap_coeff * lap_var;
                                        float4 dye_j = dye_h ? load_half4((const uint2*)dye_h + index_j)
                                                       : __ldg(&particle_dye_in[index_j]);
                                        dye_rate.x += dye_factor * (dye_j.x - dye_i.x);
                                        dye_rate.y += dye_factor * (dye_j.y - dye_i.y);
                                        dye_rate.z += dye_factor * (dye_j.z - dye_i.z);
                                    }

                                    if (is_granular) {
                                        float4 vel4_j = velocity_h ? load_half4((const uint2*)velocity_h + index_j)
                                                                   : __ldg(&velocity[index_j]);
                                        float dvx = vel_i.x - vel4_j.x;
                                        float dvy = vel_i.y - vel4_j.y;
                                        float dvz = vel_i.z - vel4_j.z;
                                        float weight = m_j / fmaxf(rho_j, RHO_EPSILON);
                                        Dxx += weight * dvx * gWx;
                                        Dyy += weight * dvy * gWy;
                                        Dzz += weight * dvz * gWz;
                                        Dxy += 0.5f * weight * (dvx * gWy + dvy * gWx);
                                        Dxz += 0.5f * weight * (dvx * gWz + dvz * gWx);
                                        Dyz += 0.5f * weight * (dvy * gWz + dvz * gWy);
                                    }
                                }
                            } else if (r_sq > 1e-12f && m_j > 0.0f) {
                                // MAT_RIGID neighbour: count it and add it to the colour-field
                                // normal like a STONE wall, so fluid at a rigid face is not
                                // classed as surface (bd-r5-review-n28.68). The mass slot holds
                                // RIGID_PSI_RHO0 * V_b (bd-r5-review-n28.77).
                                float inv_rlen = 1.0f / rlen;
                                float h_rlen = h - rlen;
                                float grad_scalar = c_precalc.spiky_grad_coeff * h_rlen * h_rlen * inv_rlen;
                                if (is_fluid) {
                                    float vol_b = m_j / RIGID_PSI_RHO0;
                                    normal.x += vol_b * grad_scalar * r.x;
                                    normal.y += vol_b * grad_scalar * r.y;
                                    normal.z += vol_b * grad_scalar * r.z;
                                }
                                neighbor_count += 1.0f;
                            } // mat_id_j != MAT_RIGID
                        }
                    }
                }
            }
        }
    }

    // Write neighbor count
    neighbor_count_out[index_i] = nl_count;

    // PostCalc (identical to K_Step1)
    float density = c_precalc.poly6_coeff * sum_density;
    float rho_floor = is_gas_i ? RHO_EPSILON : 1.0f;
    // STATIC walls store rest_density (bd-r5-review-n28.116, see K_Step1)
    density_out[index_i] = (behavior_i == STATIC)
        ? fmaxf(c_materials[mat_id_i].rest_density, 1.0f)
        : fmaxf(density, rho_floor);
    pressure_out[index_i] = compute_pressure(density_out[index_i], behavior_i, mat_id_i);

    float cp_i = c_materials[mat_id_i].heat_capacity;
    float rho_i_heat = density_out[index_i];
    dTdt_out[index_i] = c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(rho_i_heat * cp_i, RHO_EPSILON);

    float inv_sum_W = (sum_W > 0.0f) ? 1.0f / sum_W : 0.0f;
    exposure_heat_out[index_i] = sum_exposure_heat * inv_sum_W;
    exposure_corrode_out[index_i] = sum_exposure_corrode * inv_sum_W;

    float omega_mag = sqrtf(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z);
    vorticity_out[index_i] = make_float4(omega.x, omega.y, omega.z, omega_mag);
    normal_out[index_i] = make_float4(normal.x, normal.y, normal.z, neighbor_count);

    if (dye_on) {
        dye_rate_out[index_i] = make_float4(dye_rate.x, dye_rate.y, dye_rate.z, 0.0f);
    }

    if (is_granular) {
        float D_sq = Dxx * Dxx + Dyy * Dyy + Dzz * Dzz
                   + 2.0f * (Dxy * Dxy + Dxz * Dxz + Dyz * Dyz);
        shear_rate_out[index_i] = sqrtf(fmaxf(2.0f * D_sq, 0.0f));
    } else {
        shear_rate_out[index_i] = 0.0f;
    }

    position[index_i].w = density_out[index_i];
}
