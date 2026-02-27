/*
 * step2.cu -- K_Step2 pressure, viscosity, and XSPH force kernel.
 *
 * Per-particle computation:
 *   1. Tait EOS pressure from density and per-material EOS parameters
 *   2. For GRANULAR: read gamma_dot from shear_rate_in (computed by Step1)
 *   3. Neighbor loop: accumulate pressure force (spiky gradient),
 *      viscosity force (viscosity Laplacian), and XSPH correction (FLUID only)
 *   4. Write sph_force (float4) and veleval (float4, XSPH-corrected)
 *
 * mu(I) rheology (GRANULAR only):
 *   - gamma_dot read from shear_rate_in (Step1's tensor-based computation)
 *   - I = gamma_dot * spacing / sqrt(p_eff / rho_i)
 *   - mu_I = mu_s + (mu_2 - mu_s) / (1 + I0 / max(I, 1e-8))
 *   - eta_i = min(mu_max, mu0 + mu_I * p_eff / (gamma_dot + 1e-6))
 *   - Harmonic mean eta_ij for GRANULAR-GRANULAR pairs
 *
 * Skips STATIC (behavior_class == 3) and SLEEPING particles (early return).
 *
 * Constant memory used:
 *   c_grid, c_sim, c_precalc -- from common.cuh (shared with other kernels)
 *   c_materials              -- from common.cuh (per-material EOS properties)
 *   c_granular               -- local to this module (mu(I) parameters)
 *
 * Ported from SPHSimLib/K_SimpleSPH_Step2.inl with Tait EOS and multi-material
 * behavior classes per acceptance criteria.
 */

#include "sph_shared.cuh"

/* ======================================================================
 * K_Step2 kernel
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_Step2(
    uint            numParticles,
    const float4*   __restrict__ position,      // sorted positions (density packed in .w by K_PackDensity)
    const float4*   __restrict__ velocity,      // sorted evaluation velocity
    const float*    __restrict__ mass,           // sorted per-particle mass
    const uint*     __restrict__ packed_info,    // sorted packed_info (material + behavior + flags)
    const float*    __restrict__ shear_rate_in,  // sorted shear rate (gamma_dot from Step1)
    const uint*     __restrict__ cell_start,     // grid cell start indices
    const uint*     __restrict__ cell_end,       // grid cell end indices
    const float4*   __restrict__ vorticity_in,   // (omega_x, omega_y, omega_z, |omega|) from Step1
    const float4*   __restrict__ normal_in,      // (n_x, n_y, n_z, neighbor_count) from Step1
    float4*         __restrict__ sph_force_out,  // output: accumulated SPH force
    float4*         __restrict__ veleval_out,    // output: XSPH-corrected veleval
    const void*     __restrict__ velocity_h      // FP16 velocity for neighbor reads (OPT-4.3), may be NULL
) {
    uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_i >= numParticles) return;

    // --- Read packed_info; extract behavior class and flags ---
    uint pi_i = packed_info[index_i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);

    // Skip STATIC particles (early return, zero force)
    if (behavior_i == STATIC) {
        sph_force_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        veleval_out[index_i] = velocity[index_i];
        return;
    }

    // Skip SLEEPING particles (early return, zero force)
    if (IS_SLEEPING(pi_i)) {
        sph_force_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        veleval_out[index_i] = velocity[index_i];
        return;
    }

    // --- PreCalc: read particle i data ---
    float4 pos4_i = position[index_i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float4 vel4_i = velocity[index_i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);

    float rho_i = pos4_i.w;  // density packed into position.w by K_PackDensity (OPT-4.1)
    float p_i = compute_pressure(rho_i, behavior_i, mat_id_i);

    // Accumulators
    float3 f_pressure  = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_viscosity = make_float3(0.0f, 0.0f, 0.0f);
    float3 xsph_sum    = make_float3(0.0f, 0.0f, 0.0f);

    float h    = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;

    bool is_granular_i = (behavior_i == GRANULAR);
    bool is_fluid_i    = (behavior_i == FLUID);
    bool is_gas_i      = (behavior_i == GAS);

    // --- Grid cell of particle i ---
    int3 cell_i = calcGridCell(make_float3(pos_i.x, pos_i.y, pos_i.z));

    // ---------------------------------------------------------------
    // For GRANULAR particles: read gamma_dot from Step1 (tensor-based)
    // and compute mu(I) effective viscosity
    // ---------------------------------------------------------------
    float gamma_dot_i = 0.0f;
    float eta_i = 0.0f;

    if (is_granular_i) {
        gamma_dot_i = __ldg(&shear_rate_in[index_i]);

        // mu(I) rheology computation for particle i
        float p_eff_i = fmaxf(p_i, 1.0f);
        eta_i = compute_muI_eta(gamma_dot_i, p_eff_i, rho_i);
    }

    // ---------------------------------------------------------------
    // Main neighbor loop: pressure, viscosity, XSPH
    // ---------------------------------------------------------------
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint index_j = start; index_j < end_idx; index_j++) {
                    // Skip self-interaction for forces
                    if (index_j == index_i) continue;

                    // --- Speculative ILP: issue ALL loads before distance check ---
                    // All __ldg loads go through texture cache (~200 cycle latency).
                    // By issuing them in parallel before the branch, we hide latency.
                    // ~30% of loads are wasted (out-of-range), but latency savings
                    // on in-range neighbors more than compensate.
                    float4 pos4_j = __ldg(&position[index_j]);
                    uint pi_j = __ldg(&packed_info[index_j]);
                    // OPT-4.3: read from FP16 buffer (8B vs 16B per neighbor)
                    float4 vel4_j = velocity_h ? load_half4((const uint2*)velocity_h + index_j)
                                               : __ldg(&velocity[index_j]);
                    float m_j = __ldg(&mass[index_j]);

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float rlen_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (rlen_sq > h_sq || rlen_sq < 1e-12f) continue;

                    int behavior_j = GET_BEHAVIOR(pi_j);

                    // GAS/non-GAS phase separation: no pressure/viscosity across phases
                    if (is_gas_i != (behavior_j == GAS)) continue;

                    float rlen = sqrtf(rlen_sq);

                    // density_j packed into position.w by K_PackDensity (OPT-4.1)
                    float3 vel_j = make_float3(vel4_j.x, vel4_j.y, vel4_j.z);
                    float rho_j = pos4_j.w;

                    // ---- STATIC boundary: Akinci pressure mirroring + friction ----
                    if (behavior_j == STATIC) {
                        float3 grad_s = grad_spiky_variable(r, rlen, h);
                        // Pressure mirroring: boundary pressure = fluid pressure
                        float press_sym_b = 2.0f * (p_i / (rho_i * rho_i));
                        f_pressure.x += m_j * press_sym_b * grad_s.x;
                        f_pressure.y += m_j * press_sym_b * grad_s.y;
                        f_pressure.z += m_j * press_sym_b * grad_s.z;
                        // Boundary friction: vel_j=0, drag toward zero
                        float lap_v_b = lap_visc_variable(rlen, h);
                        float mu_b = is_granular_i ? c_precalc.viscosity_precalc
                                   : c_materials[mat_id_i].base_viscosity * c_precalc.viscosity_lap_coeff;
                        float visc_factor_b = mu_b * m_j * lap_v_b / fmaxf(rho_i, 1.0f);
                        f_viscosity.x += (-vel_i.x) * visc_factor_b;
                        f_viscosity.y += (-vel_i.y) * visc_factor_b;
                        f_viscosity.z += (-vel_i.z) * visc_factor_b;
                        continue;  // skip XSPH from STATIC
                    }

                    // Neighbor pressure (per-material EOS)
                    uint mat_id_j = GET_MATERIAL_ID(pi_j);
                    float p_j = compute_pressure(rho_j, behavior_j, mat_id_j);

                    // ---- Pressure force ----
                    // f_press += pressure_precalc * m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky_variable
                    float press_sym = (p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j));
                    float3 grad_s = grad_spiky_variable(r, rlen, h);
                    f_pressure.x += m_j * press_sym * grad_s.x;
                    f_pressure.y += m_j * press_sym * grad_s.y;
                    f_pressure.z += m_j * press_sym * grad_s.z;

                    // ---- Viscosity force ----
                    float lap_v = lap_visc_variable(rlen, h);

                    if (is_granular_i && behavior_j == GRANULAR) {
                        // mu(I) viscosity for GRANULAR-GRANULAR pairs
                        float dvx = vel_i.x - vel_j.x;
                        float dvy = vel_i.y - vel_j.y;
                        float dvz = vel_i.z - vel_j.z;
                        float gamma_dot_j = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz)
                                          / fmaxf(rlen, 1e-8f);

                        float p_eff_j = fmaxf(p_j, 1.0f);
                        float eta_j = compute_muI_eta(gamma_dot_j, p_eff_j, rho_j);

                        // Harmonic mean viscosity
                        float eta_ij = 2.0f * eta_i * eta_j / (eta_i + eta_j + 1e-8f);

                        // Full viscosity force with coefficient baked in
                        float visc_lap_const = c_precalc.viscosity_lap_coeff;  // = 45/(pi*h^6)
                        float visc_factor = eta_ij * visc_lap_const * m_j * lap_v / rho_j;
                        f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
                    } else if (is_granular_i) {
                        // GRANULAR-nonGRANULAR pair: use constant mu0 with
                        // full coefficient baked in (same scale as granular path)
                        float visc_factor = c_precalc.viscosity_precalc * m_j * lap_v / rho_j;
                        f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
                    } else {
                        // FLUID/GAS: per-material viscosity via harmonic mean
                        float mu_i = c_materials[mat_id_i].base_viscosity;
                        float mu_j = c_materials[mat_id_j].base_viscosity;
                        float mu_ij = 2.0f * mu_i * mu_j / (mu_i + mu_j + 1e-8f);
                        float visc_factor = mu_ij * c_precalc.viscosity_lap_coeff * m_j * lap_v / rho_j;
                        f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
                    }

                    // ---- XSPH correction (FLUID only) ----
                    if (is_fluid_i) {
                        float w = W_poly6(rlen_sq, h_sq);
                        float rho_avg = 0.5f * (rho_i + rho_j);
                        float xsph_factor = (m_j / rho_avg) * w;
                        xsph_sum.x += (vel_j.x - vel_i.x) * xsph_factor;
                        xsph_sum.y += (vel_j.y - vel_i.y) * xsph_factor;
                        xsph_sum.z += (vel_j.z - vel_i.z) * xsph_factor;
                    }
                }  // end neighbor particle loop
            }
        }
    }  // end 27-cell loop

    // --- Vorticity confinement (FLUID only) ---
    // Read vorticity of particle i computed in step1
    float3 f_vorticity_conf = make_float3(0.0f, 0.0f, 0.0f);
    if (is_fluid_i && c_granular.vorticity_epsilon > 0.0f) {
        float4 vort_i = __ldg(&vorticity_in[index_i]);
        float omega_mag_i = vort_i.w;

        if (omega_mag_i > 1e-6f) {
            // Compute eta = grad(|omega|) via neighbors
            float3 eta = make_float3(0.0f, 0.0f, 0.0f);
            for (int dz2 = -1; dz2 <= 1; dz2++) {
                for (int dy2 = -1; dy2 <= 1; dy2++) {
                    for (int dx2 = -1; dx2 <= 1; dx2++) {
                        uint hash_v = spatialHash(cell_i.x + dx2, cell_i.y + dy2, cell_i.z + dz2);
                        uint start_v = cell_start[hash_v];
                        if (start_v == 0xFFFFFFFFu) continue;
                        uint end_v = cell_end[hash_v];
                        for (uint jv = start_v; jv < end_v; jv++) {
                            if (jv == index_i) continue;
                            float4 pj_v = __ldg(&position[jv]);
                            float3 rv = make_float3(pos_i.x - pj_v.x, pos_i.y - pj_v.y, pos_i.z - pj_v.z);
                            float r2v = rv.x*rv.x + rv.y*rv.y + rv.z*rv.z;
                            if (r2v > h_sq || r2v < 1e-12f) continue;
                            float rlv = sqrtf(r2v);
                            float h_rl = h - rlv;
                            float inv_rl = 1.0f / rlv;
                            float gs = c_precalc.spiky_grad_coeff * h_rl * h_rl * inv_rl;
                            float omega_j = __ldg(&vorticity_in[jv]).w;
                            float mj_v = __ldg(&mass[jv]);
                            float rj_v = pj_v.w;  // density from position.w (OPT-4.1)
                            float wt = mj_v / fmaxf(rj_v, 1.0f);
                            eta.x += wt * omega_j * gs * rv.x;
                            eta.y += wt * omega_j * gs * rv.y;
                            eta.z += wt * omega_j * gs * rv.z;
                        }
                    }
                }
            }
            // N = eta / |eta|
            float eta_mag = sqrtf(eta.x*eta.x + eta.y*eta.y + eta.z*eta.z);
            if (eta_mag > 1e-6f) {
                float inv_eta = 1.0f / eta_mag;
                float3 N = make_float3(eta.x*inv_eta, eta.y*inv_eta, eta.z*inv_eta);
                // f_conf = epsilon * (N x omega_i)
                float eps_v = c_granular.vorticity_epsilon;
                f_vorticity_conf.x = eps_v * (N.y * vort_i.z - N.z * vort_i.y);
                f_vorticity_conf.y = eps_v * (N.z * vort_i.x - N.x * vort_i.z);
                f_vorticity_conf.z = eps_v * (N.x * vort_i.y - N.y * vort_i.x);
            }
        }
    }

    // --- Akinci surface tension (FLUID only, surface particles) ---
    float3 f_surface_tension = make_float3(0.0f, 0.0f, 0.0f);
    if (is_fluid_i && c_granular.surface_tension_gamma > 0.0f) {
        float4 norm_i = __ldg(&normal_in[index_i]);
        float nc_i = norm_i.w;  // neighbor count

        // Surface particles have fewer neighbors than interior particles
        if (nc_i < 25.0f) {
            float gamma = c_granular.surface_tension_gamma;
            // Curvature force: -gamma * n_i (pulls surface inward)
            float n_mag = sqrtf(norm_i.x*norm_i.x + norm_i.y*norm_i.y + norm_i.z*norm_i.z);
            if (n_mag > 0.01f) {
                f_surface_tension.x = -gamma * norm_i.x;
                f_surface_tension.y = -gamma * norm_i.y;
                f_surface_tension.z = -gamma * norm_i.z;
            }
        }
    }

    // --- PostCalc: apply precalc coefficients and write output ---
    float3 total_force;
    if (is_granular_i) {
        // GRANULAR: viscosity already has full coefficients baked in per-pair
        total_force.x = c_precalc.pressure_precalc * f_pressure.x + f_viscosity.x;
        total_force.y = c_precalc.pressure_precalc * f_pressure.y + f_viscosity.y;
        total_force.z = c_precalc.pressure_precalc * f_pressure.z + f_viscosity.z;
    } else {
        // FLUID/GAS: per-material viscosity already baked in per-pair
        total_force.x = c_precalc.pressure_precalc * f_pressure.x + f_viscosity.x;
        total_force.y = c_precalc.pressure_precalc * f_pressure.y + f_viscosity.y;
        total_force.z = c_precalc.pressure_precalc * f_pressure.z + f_viscosity.z;
    }

    // Add vorticity confinement + surface tension (FLUID only, already zero for non-FLUID)
    total_force.x += f_vorticity_conf.x + f_surface_tension.x;
    total_force.y += f_vorticity_conf.y + f_surface_tension.y;
    total_force.z += f_vorticity_conf.z + f_surface_tension.z;

    // GRANULAR uses force_scale=1.0 (full-strength SPH forces).
    // With force_scale=0.02 and rho0=2500, SPH pressure is ~50x too weak vs gravity,
    // causing 200%+ compression instead of 2-5%.  FLUID keeps 0.02 (game-tuned).
    float fs = is_granular_i ? 1.0f : c_granular.force_scale;
    sph_force_out[index_i] = make_float4(total_force.x * fs, total_force.y * fs, total_force.z * fs, 0.0f);

    // XSPH-corrected veleval (FLUID only; others keep original velocity)
    if (is_fluid_i) {
        float eps = c_granular.xsph_epsilon;
        veleval_out[index_i] = make_float4(
            vel_i.x + eps * xsph_sum.x,
            vel_i.y + eps * xsph_sum.y,
            vel_i.z + eps * xsph_sum.z,
            0.0f
        );
    } else {
        veleval_out[index_i] = make_float4(vel_i.x, vel_i.y, vel_i.z, 0.0f);
    }
}
