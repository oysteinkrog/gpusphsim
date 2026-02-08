/*
 * step1.cu -- K_Step1 SPH density summation + strain-rate tensor + heat diffusion
 *             + exposure accumulation kernel.
 *
 * Per-particle computation:
 *   density_sum += m_j * (h^2 - |r_ij|^2)^3   for all neighbors j within h
 *   density_i = max(1.0, poly6_coeff * density_sum)
 *
 * Heat diffusion (all particles):
 *   dTdt += kappa_i * viscosity_lap_coeff * sum_j (m_j/rho_j) * (T_j - T_i) * (h - |r_ij|)
 *   where kappa_i = c_materials[mat_id].thermal_conductivity
 *
 * Exposure accumulation (all particles, via c_interactions table lookup):
 *   exposure_corrode_i += c_interactions[mat_i][mat_j].reaction_rate * W_poly6(r_sq, h_sq)
 *   exposure_heat_i += c_interactions[mat_i][mat_j].heat_exchange * max(T_j - T_i, 0) * W_poly6(r_sq, h_sq)
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

#include "common.cuh"

extern "C" __global__ __launch_bounds__(256, 4)
void K_Step1(
    uint            numParticles,
    const float4*   __restrict__ position,      // sorted positions
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
    float4*         __restrict__ dye_rate_out          // output: dye diffusion rate (dr, dg, db, unused)
) {
    uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_i >= numParticles) return;

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

    // Read velocity for strain-rate (GRANULAR) and vorticity (FLUID)
    float4 vel4_i = velocity[index_i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);

    // Read dye color for diffusion
    float4 dye_i = __ldg(&particle_dye_in[index_i]);

    // Read temperature for heat diffusion
    float T_i = __ldg(&temperature_in[index_i]);
    float kappa_i = c_materials[mat_id_i].thermal_conductivity;

    // Density accumulator (variable part of Poly6 kernel)
    float sum_density = 0.0f;

    // Heat diffusion accumulator: dTdt
    float sum_dTdt = 0.0f;

    // Exposure accumulators (from c_interactions table lookup)
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;

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
                    float4 pos4_j = __ldg(&position[index_j]);
                    float m_j = __ldg(&mass[index_j]);
                    uint pi_j = __ldg(&packed_info[index_j]);
                    float T_j = __ldg(&temperature_in[index_j]);

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;

                    if (r_sq <= h_sq) {
                        float diff = h_sq - r_sq;

                        // Skip STATIC neighbors for density (keep self i==j)
                        int behavior_j = GET_BEHAVIOR(pi_j);
                        if (behavior_j == STATIC && index_j != index_i) {
                            continue;
                        }

                        // GAS↔non-GAS phase separation: skip density/heat/forces,
                        // keep only exposure (so fire→wood ignition still works)
                        bool is_gas_j = (behavior_j == GAS);
                        if (is_gas_i != is_gas_j && index_j != index_i) {
                            uint mat_id_j = GET_MATERIAL_ID(pi_j);
                            float w_poly6_var = diff * diff * diff;
                            sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                            sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
                            continue;
                        }

                        // --- Density: self-interaction included ---
                        sum_density += m_j * diff * diff * diff;

                        // --- Heat diffusion + exposure: skip self ---
                        if (index_j != index_i) {
                            float rlen = sqrtf(r_sq);
                            // T_j already loaded speculatively before distance check (OPT-4.2)
                            float rho_j = (density_in != 0) ? __ldg(&density_in[index_j]) : 1000.0f;
                            // SPH Laplacian of temperature using viscosity Laplacian kernel:
                            // lap_W_visc(r) = (h - |r|)  (variable part, coeff is viscosity_lap_coeff)
                            // dTdt += kappa * (m_j / rho_j) * (T_j - T_i) * viscosity_lap_coeff * (h - |r|)
                            float lap_var = h - rlen;
                            sum_dTdt += m_j / fmaxf(rho_j, 1.0f) * (T_j - T_i) * lap_var;

                            // --- Exposure accumulation: pure table lookup, no branching ---
                            // W_poly6 variable part: diff^3 = (h^2 - r^2)^3
                            // Full W_poly6 = poly6_coeff * diff^3, but we use just diff^3
                            // as the weighting (unnormalized) -- the absolute magnitude
                            // is tuned via reaction_rate/heat_exchange in the table.
                            uint mat_id_j = GET_MATERIAL_ID(pi_j);
                            float w_poly6_var = diff * diff * diff;  // (h^2 - r^2)^3

                            sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
                            sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
                        }

                        // --- Vorticity + Normal + Dye: skip self, need gradient ---
                        if (index_j != index_i && r_sq > 1e-12f) {
                            float rlen_v = sqrtf(r_sq);
                            float h_rlen_v = h - rlen_v;
                            float inv_rlen_v = 1.0f / rlen_v;

                            // Spiky gradient: gradW = spiky_grad_coeff * (h-r)^2 * r/|r|
                            float grad_scalar_v = c_precalc.spiky_grad_coeff * h_rlen_v * h_rlen_v * inv_rlen_v;
                            float gWx_v = grad_scalar_v * r.x;
                            float gWy_v = grad_scalar_v * r.y;
                            float gWz_v = grad_scalar_v * r.z;

                            float rho_j_v = (density_in != 0) ? __ldg(&density_in[index_j]) : 1000.0f;
                            float vol_j = m_j / fmaxf(rho_j_v, 1.0f);  // V_j = m_j / rho_j

                            if (is_fluid) {
                                // Vorticity: omega += V_j * (v_j - v_i) x gradW
                                float4 vel4_j_v = __ldg(&velocity[index_j]);
                                float dvx_v = vel4_j_v.x - vel_i.x;
                                float dvy_v = vel4_j_v.y - vel_i.y;
                                float dvz_v = vel4_j_v.z - vel_i.z;
                                omega.x += vol_j * (dvy_v * gWz_v - dvz_v * gWy_v);
                                omega.y += vol_j * (dvz_v * gWx_v - dvx_v * gWz_v);
                                omega.z += vol_j * (dvx_v * gWy_v - dvy_v * gWx_v);

                                // Surface normal: n += V_j * gradW
                                normal.x += vol_j * gWx_v;
                                normal.y += vol_j * gWy_v;
                                normal.z += vol_j * gWz_v;
                            }

                            // Neighbor count (all behaviors)
                            neighbor_count += 1.0f;

                            // Dye diffusion: dC/dt += D * V_j * (C_j - C_i) * lap_W
                            // Use viscosity Laplacian kernel: lap_W_var = (h - |r|)
                            float lap_var_d = h - rlen_v;
                            float dye_factor = 0.01f * vol_j * c_precalc.viscosity_lap_coeff * lap_var_d;
                            float4 dye_j = __ldg(&particle_dye_in[index_j]);
                            dye_rate.x += dye_factor * (dye_j.x - dye_i.x);
                            dye_rate.y += dye_factor * (dye_j.y - dye_i.y);
                            dye_rate.z += dye_factor * (dye_j.z - dye_i.z);
                        }

                        // --- Strain-rate: skip self, GRANULAR only ---
                        if (is_granular && index_j != index_i && r_sq > 1e-12f) {
                            float rlen = sqrtf(r_sq);

                            // Spiky gradient: gradW = spiky_grad_coeff * (h - r)^2 * (r/|r|)
                            // spiky_grad_coeff is negative (-45/(pi*h^6))
                            // We need the actual gradient vector pointing from j to i
                            float h_r = h - rlen;
                            float inv_rlen = 1.0f / rlen;
                            // gradW = spiky_grad_coeff * (h-r)^2 * r_hat
                            // where r_hat = r/|r| points from j to i
                            float grad_scalar = c_precalc.spiky_grad_coeff * h_r * h_r * inv_rlen;
                            float gWx = grad_scalar * r.x;
                            float gWy = grad_scalar * r.y;
                            float gWz = grad_scalar * r.z;

                            // dv = v_i - v_j
                            float4 vel4_j = __ldg(&velocity[index_j]);
                            float dvx = vel_i.x - vel4_j.x;
                            float dvy = vel_i.y - vel4_j.y;
                            float dvz = vel_i.z - vel4_j.z;

                            // m_j / rho_j weighting
                            float rho_j = (density_in != 0) ? __ldg(&density_in[index_j]) : 1000.0f;
                            float weight = m_j / fmaxf(rho_j, 1.0f);

                            // Accumulate velocity gradient tensor L_ab = sum (m_j/rho_j) * dv_a * gradW_b
                            // D_ab = 0.5 * (L_ab + L_ba)
                            // Since we compute D directly:
                            //   D_xx += weight * dv_x * gW_x
                            //   D_yy += weight * dv_y * gW_y
                            //   D_zz += weight * dv_z * gW_z
                            //   D_xy += 0.5 * weight * (dv_x * gW_y + dv_y * gW_x)
                            //   D_xz += 0.5 * weight * (dv_x * gW_z + dv_z * gW_x)
                            //   D_yz += 0.5 * weight * (dv_y * gW_z + dv_z * gW_y)
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

    // --- PostCalc: density ---
    float density = c_precalc.poly6_coeff * sum_density;
    density_out[index_i] = fmaxf(density, 1.0f);

    // --- PostCalc: heat diffusion dTdt ---
    // dTdt = kappa_i * viscosity_lap_coeff * sum_dTdt
    dTdt_out[index_i] = kappa_i * c_precalc.viscosity_lap_coeff * sum_dTdt;

    // --- PostCalc: exposure accumulation ---
    // Apply poly6_coeff to get properly normalized exposure values
    exposure_heat_out[index_i] = c_precalc.poly6_coeff * sum_exposure_heat;
    exposure_corrode_out[index_i] = c_precalc.poly6_coeff * sum_exposure_corrode;

    // --- PostCalc: vorticity ---
    float omega_mag = sqrtf(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z);
    vorticity_out[index_i] = make_float4(omega.x, omega.y, omega.z, omega_mag);

    // --- PostCalc: surface normal + neighbor count ---
    normal_out[index_i] = make_float4(normal.x, normal.y, normal.z, neighbor_count);

    // --- PostCalc: dye diffusion rate ---
    dye_rate_out[index_i] = make_float4(dye_rate.x, dye_rate.y, dye_rate.z, 0.0f);

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
}


/* ======================================================================
 * K_PackDensity -- Pack density into sorted_position.w
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
