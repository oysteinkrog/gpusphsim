/*
 * step2.cu -- K_Step2 pressure, viscosity, and XSPH force kernel.
 *
 * Per-particle computation:
 *   1. Tait EOS pressure from density and per-material EOS parameters
 *   2. For GRANULAR: first pass to compute gamma_dot (strain rate) for mu(I)
 *   3. Neighbor loop: accumulate pressure force (spiky gradient),
 *      viscosity force (viscosity Laplacian), and XSPH correction (FLUID only)
 *   4. Write sph_force (float4) and veleval (float4, XSPH-corrected)
 *
 * mu(I) rheology (GRANULAR only):
 *   - gamma_dot computed from SPH velocity gradient in first neighbor pass
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

#include "common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ======================================================================
 * GranularParams -- mu(I) rheology parameters.
 * Local to this module; uploaded from Python before kernel launch.
 * ====================================================================== */

struct GranularParams {
    float mu_s;               // static friction coefficient (0.36)
    float mu_2;               // dynamic friction coefficient (0.70)
    float I0;                 // inertial number reference (0.3)
    float mu_max;             // viscosity clamp (10000 Pa*s)
    float particle_spacing;   // particle spacing d (0.02)
    float mu0;                // base viscosity for GRANULAR
    float xsph_epsilon;       // XSPH blending factor (0.5)
    float _pad0;              // padding to 32 bytes
};

__constant__ GranularParams c_granular;

/* ======================================================================
 * Tait EOS pressure computation
 *
 * Reads per-material rest_density, eos_stiffness, eos_gamma from c_materials.
 *
 *   FLUID:    p_raw = k * (pow(rho/rho0, 7) - 1);  p = max(p_raw, -0.5*k)
 *   GRANULAR: p_raw = k * (pow(rho/rho0, 7) - 1);  p = max(p_raw, 0)
 *   GAS:      p = k_gas * max(rho - rho0, 0)        (gamma=1, linear)
 * ====================================================================== */

__device__ inline float compute_pressure(float rho_i, int behavior, uint mat_id) {
    float rho0 = c_materials[mat_id].rest_density;
    float k    = c_materials[mat_id].eos_stiffness;

    if (behavior == GAS) {
        // GAS: linear EOS with gamma=1
        return k * fmaxf(rho_i - rho0, 0.0f);
    }

    // FLUID / GRANULAR: Tait EOS with gamma=7
    float ratio = rho_i / fmaxf(rho0, 1e-6f);
    float p_raw = k * (powf(ratio, 7.0f) - 1.0f);

    if (behavior == GRANULAR) {
        return fmaxf(p_raw, 0.0f);
    }
    // FLUID: allow small tensile pressure
    return fmaxf(p_raw, -0.5f * k);
}

/* ======================================================================
 * SPH kernel variable parts
 * (constant coefficients applied in PostCalc via c_precalc)
 * ====================================================================== */

/**
 * Spiky gradient variable part: (r/|r|) * (h - |r|)^2
 * Points AWAY from neighbor (r = pos_i - pos_j).
 * No coefficient -- that's in pressure_precalc.
 */
__device__ inline float3 grad_spiky_variable(float3 r, float rlen, float h) {
    float h_rlen = h - rlen;
    float inv_rlen = 1.0f / rlen;
    return make_float3(
        r.x * inv_rlen * (h_rlen * h_rlen),
        r.y * inv_rlen * (h_rlen * h_rlen),
        r.z * inv_rlen * (h_rlen * h_rlen)
    );
}

/**
 * Viscosity Laplacian variable part: (h - |r|)
 */
__device__ inline float lap_visc_variable(float rlen, float h) {
    return h - rlen;
}

/**
 * Poly6 kernel (for XSPH): W_poly6 = poly6_coeff * (h^2 - |r|^2)^3
 * Returns the FULL kernel value.
 */
__device__ inline float W_poly6(float rlen_sq, float h_sq) {
    float diff = h_sq - rlen_sq;
    return c_precalc.poly6_coeff * diff * diff * diff;
}

/**
 * Compute mu(I) effective viscosity for a particle.
 */
__device__ inline float compute_muI_eta(float gamma_dot, float p_eff, float rho) {
    float spacing = c_granular.particle_spacing;
    float I_number = gamma_dot * spacing / sqrtf(p_eff / rho);
    float mu_I = c_granular.mu_s
               + (c_granular.mu_2 - c_granular.mu_s)
                 / (1.0f + c_granular.I0 / fmaxf(I_number, 1e-8f));
    return fminf(c_granular.mu_max,
                 c_granular.mu0 + mu_I * p_eff / (gamma_dot + 1e-6f));
}

/* ======================================================================
 * K_Step2 kernel
 * ====================================================================== */

extern "C" __global__
void K_Step2(
    uint            numParticles,
    const float4*   __restrict__ position,      // sorted positions
    const float4*   __restrict__ velocity,      // sorted evaluation velocity
    const float*    __restrict__ density,        // sorted density (from Step1)
    const float*    __restrict__ mass,           // sorted per-particle mass
    const uint*     __restrict__ packed_info,    // sorted packed_info (material + behavior + flags)
    const uint*     __restrict__ cell_start,     // grid cell start indices
    const uint*     __restrict__ cell_end,       // grid cell end indices
    float4*         __restrict__ sph_force_out,  // output: accumulated SPH force
    float4*         __restrict__ veleval_out     // output: XSPH-corrected veleval
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

    float rho_i = density[index_i];
    float p_i = compute_pressure(rho_i, behavior_i, mat_id_i);

    // Accumulators
    float3 f_pressure  = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_viscosity = make_float3(0.0f, 0.0f, 0.0f);
    float3 xsph_sum    = make_float3(0.0f, 0.0f, 0.0f);

    float h    = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;

    bool is_granular_i = (behavior_i == GRANULAR);
    bool is_fluid_i    = (behavior_i == FLUID);

    // --- Grid cell of particle i ---
    int3 cell_i = make_int3(
        (int)((pos_i.x - c_grid.grid_min.x) * c_grid.grid_delta.x),
        (int)((pos_i.y - c_grid.grid_min.y) * c_grid.grid_delta.y),
        (int)((pos_i.z - c_grid.grid_min.z) * c_grid.grid_delta.z)
    );
    cell_i.x = max(0, min(cell_i.x, c_grid.grid_res.x - 1));
    cell_i.y = max(0, min(cell_i.y, c_grid.grid_res.y - 1));
    cell_i.z = max(0, min(cell_i.z, c_grid.grid_res.z - 1));

    int rx = c_grid.grid_res.x;
    int ry = c_grid.grid_res.y;
    int rz = c_grid.grid_res.z;

    // ---------------------------------------------------------------
    // For GRANULAR particles: first pass to compute gamma_dot_i
    // (strain rate magnitude from SPH velocity gradient)
    // ---------------------------------------------------------------
    float gamma_dot_i = 0.0f;
    float eta_i = 0.0f;

    if (is_granular_i) {
        float gamma_dot_sq_sum = 0.0f;

        for (int dz = -1; dz <= 1; dz++) {
            int cz = cell_i.z + dz;
            if (cz < 0 || cz >= rz) continue;
            for (int dy = -1; dy <= 1; dy++) {
                int cy = cell_i.y + dy;
                if (cy < 0 || cy >= ry) continue;
                for (int dx = -1; dx <= 1; dx++) {
                    int cx = cell_i.x + dx;
                    if (cx < 0 || cx >= rx) continue;

                    uint hash_c = (uint)(cz * ry * rx + cy * rx + cx);
                    uint start = cell_start[hash_c];
                    if (start == 0xFFFFFFFFu) continue;
                    uint end_idx = cell_end[hash_c];

                    for (uint index_j = start; index_j < end_idx; index_j++) {
                        if (index_j == index_i) continue;

                        float4 pos4_j = __ldg(&position[index_j]);
                        float3 r = make_float3(
                            pos_i.x - pos4_j.x,
                            pos_i.y - pos4_j.y,
                            pos_i.z - pos4_j.z
                        );
                        float rlen_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                        if (rlen_sq > h_sq || rlen_sq < 1e-12f) continue;

                        float rlen = sqrtf(rlen_sq);
                        float4 vel4_j = __ldg(&velocity[index_j]);
                        float dvx = vel_i.x - vel4_j.x;
                        float dvy = vel_i.y - vel4_j.y;
                        float dvz = vel_i.z - vel4_j.z;
                        float dv_sq = dvx * dvx + dvy * dvy + dvz * dvz;

                        float rho_j = __ldg(&density[index_j]);
                        float m_j = __ldg(&mass[index_j]);
                        // Spiky gradient magnitude for strain rate: (h - r)^2 / r
                        float h_r = h - rlen;
                        float grad_mag = h_r * h_r / rlen;
                        gamma_dot_sq_sum += (m_j / rho_j) * dv_sq * grad_mag;
                    }
                }
            }
        }

        // Apply SPH gradient normalization: 45/(pi*h^6)
        float lap_const = c_precalc.pressure_precalc;  // = 45/(pi*h^6)
        gamma_dot_i = sqrtf(fmaxf(lap_const * gamma_dot_sq_sum, 0.0f));

        // mu(I) rheology computation for particle i
        float p_eff_i = fmaxf(p_i, 1.0f);
        eta_i = compute_muI_eta(gamma_dot_i, p_eff_i, rho_i);
    }

    // ---------------------------------------------------------------
    // Main neighbor loop: pressure, viscosity, XSPH
    // ---------------------------------------------------------------
    for (int dz = -1; dz <= 1; dz++) {
        int cz = cell_i.z + dz;
        if (cz < 0 || cz >= rz) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int cy = cell_i.y + dy;
            if (cy < 0 || cy >= ry) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int cx = cell_i.x + dx;
                if (cx < 0 || cx >= rx) continue;

                uint hash_c = (uint)(cz * ry * rx + cy * rx + cx);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint index_j = start; index_j < end_idx; index_j++) {
                    // Skip self-interaction for forces
                    if (index_j == index_i) continue;

                    float4 pos4_j = __ldg(&position[index_j]);
                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float rlen_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (rlen_sq > h_sq || rlen_sq < 1e-12f) continue;

                    float rlen = sqrtf(rlen_sq);

                    // Read neighbor data
                    float4 vel4_j = __ldg(&velocity[index_j]);
                    float3 vel_j = make_float3(vel4_j.x, vel4_j.y, vel4_j.z);
                    float rho_j = __ldg(&density[index_j]);
                    float m_j = __ldg(&mass[index_j]);

                    // Neighbor pressure (per-material EOS)
                    uint pi_j = __ldg(&packed_info[index_j]);
                    int behavior_j = GET_BEHAVIOR(pi_j);
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
                        // FLUID/GAS: raw accumulation, viscosity_precalc applied in PostCalc
                        float visc_factor = m_j * lap_v / rho_j;
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

    // --- PostCalc: apply precalc coefficients and write output ---
    float3 total_force;
    if (is_granular_i) {
        // GRANULAR: viscosity already has full coefficients baked in per-pair
        total_force.x = c_precalc.pressure_precalc * f_pressure.x + f_viscosity.x;
        total_force.y = c_precalc.pressure_precalc * f_pressure.y + f_viscosity.y;
        total_force.z = c_precalc.pressure_precalc * f_pressure.z + f_viscosity.z;
    } else {
        // FLUID/GAS: viscosity_precalc = mu0 * 45/(pi*h^6) applied uniformly
        total_force.x = c_precalc.pressure_precalc  * f_pressure.x
                      + c_precalc.viscosity_precalc  * f_viscosity.x;
        total_force.y = c_precalc.pressure_precalc  * f_pressure.y
                      + c_precalc.viscosity_precalc  * f_viscosity.y;
        total_force.z = c_precalc.pressure_precalc  * f_pressure.z
                      + c_precalc.viscosity_precalc  * f_viscosity.z;
    }

    sph_force_out[index_i] = make_float4(total_force.x, total_force.y, total_force.z, 0.0f);

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
