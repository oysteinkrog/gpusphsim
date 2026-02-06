/**
 * K_Step2 -- Pressure, viscosity, and XSPH force kernel.
 *
 * Per-particle computation:
 *   1. Tait EOS pressure from density
 *   2. For GRANULAR: first pass to compute gamma_dot (strain rate) for mu(I) rheology
 *   3. Neighbor loop: accumulate pressure force (spiky gradient),
 *      viscosity force (viscosity Laplacian), and XSPH correction (FLUID only)
 *   4. Write sph_force (float4) and veleval (float4, XSPH-corrected) to sorted buffers
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
 * Ported from SPHSimLib/K_SimpleSPH_Step2.inl with Tait EOS and multi-material
 * behavior classes per acceptance criteria.
 */

#include "common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
// Constant memory -- uploaded from Python before kernel launch
// ---------------------------------------------------------------------------
__constant__ GridParams c_grid;

// SPH fluid/precalc parameters
struct FluidParams {
    float smoothing_length;     // h
    float particle_mass;        // m (constant for all particles)
    float rest_density;         // rho_0
    float gas_stiffness;        // k  (Tait EOS stiffness)
    float gas_stiffness_gas;    // k_gas (for GAS type, gamma=1)
    float gamma;                // 7 for FLUID/GRANULAR, 1 for GAS
    float viscosity;            // mu
    float xsph_epsilon;         // XSPH blending factor (0.5)
};

struct PrecalcParams {
    float smoothing_length_pow2;    // h^2
    float pressure_precalc;         // +45/(pi*h^6) -- POSITIVE
    float viscosity_precalc;        // mu * 45/(pi*h^6)
    float kernel_poly6_coeff;       // 315/(64*pi*h^9) -- for XSPH W_poly6
};

// mu(I) granular rheology parameters
struct GranularParams {
    float mu_s;               // static friction coefficient (0.36)
    float mu_2;               // dynamic friction coefficient (0.70)
    float I0;                 // inertial number reference (0.3)
    float mu_max;             // viscosity clamp (10000 Pa·s)
    float particle_spacing;   // particle spacing d (0.02)
    float mu0;                // base viscosity for GRANULAR (same as c_fluid.viscosity)
    float _pad0;              // padding to 32 bytes
    float _pad1;
};

__constant__ FluidParams    c_fluid;
__constant__ PrecalcParams  c_precalc;
__constant__ GranularParams c_granular;

// ---------------------------------------------------------------------------
// Behavior classes and flags
// ---------------------------------------------------------------------------
#define BEHAVIOR_FLUID    1
#define BEHAVIOR_GRANULAR 2
#define BEHAVIOR_STATIC   3
#define BEHAVIOR_GAS      4

#define FLAG_IS_SLEEPING  (1u << 0)

// ---------------------------------------------------------------------------
// Grid helper functions (shared with hash_sort.cu via common.cuh)
// ---------------------------------------------------------------------------

__device__ inline int3 calcGridCell_step2(float3 p) {
    return make_int3(
        (int)((p.x - c_grid.grid_min.x) * c_grid.grid_delta.x),
        (int)((p.y - c_grid.grid_min.y) * c_grid.grid_delta.y),
        (int)((p.z - c_grid.grid_min.z) * c_grid.grid_delta.z)
    );
}

__device__ inline int3 clampCell_step2(int3 cell) {
    int rx = (int)c_grid.grid_res.x;
    int ry = (int)c_grid.grid_res.y;
    int rz = (int)c_grid.grid_res.z;
    cell.x = max(0, min(cell.x, rx - 1));
    cell.y = max(0, min(cell.y, ry - 1));
    cell.z = max(0, min(cell.z, rz - 1));
    return cell;
}

__device__ inline uint calcGridHash_step2(int3 cell) {
    int rx = (int)c_grid.grid_res.x;
    int ry = (int)c_grid.grid_res.y;
    return (uint)(cell.z * ry * rx + cell.y * rx + cell.x);
}

// ---------------------------------------------------------------------------
// Tait EOS pressure computation
// ---------------------------------------------------------------------------

/**
 * Compute Tait EOS pressure:
 *   p_raw = k * (pow(rho_i / rho0, gamma) - 1)
 *
 * Per behavior class:
 *   FLUID:    p = max(p_raw, -0.5*k)
 *   GRANULAR: p = max(p_raw, 0)
 *   GAS:      p = k_gas * max(rho - rho0, 0)  (gamma=1, linear)
 */
__device__ inline float compute_pressure(float rho_i, int behavior_class) {
    float rho0 = c_fluid.rest_density;
    float k    = c_fluid.gas_stiffness;

    if (behavior_class == BEHAVIOR_GAS) {
        // GAS: linear EOS with gamma=1
        float k_gas = c_fluid.gas_stiffness_gas;
        return k_gas * fmaxf(rho_i - rho0, 0.0f);
    }

    // FLUID / GRANULAR: Tait EOS with gamma=7
    float ratio = rho_i / rho0;
    float p_raw = k * (powf(ratio, 7.0f) - 1.0f);

    if (behavior_class == BEHAVIOR_GRANULAR) {
        return fmaxf(p_raw, 0.0f);
    }
    // FLUID: allow small tensile pressure
    return fmaxf(p_raw, -0.5f * k);
}

// ---------------------------------------------------------------------------
// SPH kernel variable parts (constant coefficients applied in PostCalc)
// ---------------------------------------------------------------------------

/**
 * Spiky gradient variable part: (r/|r|) * (h - |r|)^2
 * Points AWAY from neighbor (r = pos_i - pos_j, so direction is i <- j outward).
 * No coefficient included -- that's in pressure_precalc.
 */
__device__ inline float3 grad_spiky_variable(float3 r, float rlen) {
    float h = c_fluid.smoothing_length;
    float h_rlen = h - rlen;
    // r * (1/rlen) * (h - rlen)^2
    return make_float3(
        r.x * (1.0f / rlen) * (h_rlen * h_rlen),
        r.y * (1.0f / rlen) * (h_rlen * h_rlen),
        r.z * (1.0f / rlen) * (h_rlen * h_rlen)
    );
}

/**
 * Viscosity Laplacian variable part: (h - |r|)
 * Scalar value, multiplied by velocity difference externally.
 */
__device__ inline float lap_visc_variable(float rlen) {
    return c_fluid.smoothing_length - rlen;
}

/**
 * Poly6 kernel (for XSPH): W_poly6 = coeff * (h^2 - |r|^2)^3
 * Returns the FULL kernel value (coeff already in c_precalc.kernel_poly6_coeff).
 */
__device__ inline float W_poly6(float rlen_sq) {
    float hsq = c_precalc.smoothing_length_pow2;
    float diff = hsq - rlen_sq;
    return c_precalc.kernel_poly6_coeff * diff * diff * diff;
}

/**
 * Compute mu(I) effective viscosity for a particle.
 *
 * gamma_dot : local shear rate
 * p_eff     : effective pressure (clamped >= 1)
 * rho       : particle density
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

// ---------------------------------------------------------------------------
// K_Step2 kernel
// ---------------------------------------------------------------------------

extern "C" __global__
void K_Step2(
    uint            numParticles,
    const float4*   __restrict__ position,      // sorted positions
    const float4*   __restrict__ veleval,        // sorted evaluation velocity
    const float*    __restrict__ density,        // sorted density (from Step1)
    const int*      __restrict__ behavior_class, // sorted behavior class per particle
    const uint*     __restrict__ flags,          // sorted particle flags
    const uint*     __restrict__ cell_start,     // grid cell start indices
    const uint*     __restrict__ cell_end,       // grid cell end indices
    float4*         __restrict__ sph_force_out,  // output: accumulated SPH force
    float4*         __restrict__ veleval_out     // output: XSPH-corrected veleval
) {
    uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_i >= numParticles) return;

    // --- Read behavior class and flags; skip STATIC and SLEEPING ---
    int bclass_i = behavior_class[index_i];

    if (bclass_i == BEHAVIOR_STATIC) {
        sph_force_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    uint flags_i = flags[index_i];
    if (flags_i & FLAG_IS_SLEEPING) {
        sph_force_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // --- PreCalc: read particle i data, compute pressure ---
    float4 pos4_i = position[index_i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float4 vel4_i = veleval[index_i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);

    float rho_i = density[index_i];
    float p_i = compute_pressure(rho_i, bclass_i);

    // Accumulators
    float3 f_pressure  = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_viscosity = make_float3(0.0f, 0.0f, 0.0f);
    float3 xsph_sum    = make_float3(0.0f, 0.0f, 0.0f);

    float h_sq = c_precalc.smoothing_length_pow2;
    float m_j  = c_fluid.particle_mass;

    bool is_granular_i = (bclass_i == BEHAVIOR_GRANULAR);

    // --- Grid cell of particle i ---
    int3 cell_i = calcGridCell_step2(pos_i);
    cell_i = clampCell_step2(cell_i);

    int rx = (int)c_grid.grid_res.x;
    int ry = (int)c_grid.grid_res.y;
    int rz = (int)c_grid.grid_res.z;

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

                    uint hash = (uint)(cz * ry * rx + cy * rx + cx);
                    uint start = cell_start[hash];
                    if (start == 0xFFFFFFFFu) continue;
                    uint end = cell_end[hash];

                    for (uint index_j = start; index_j < end; index_j++) {
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
                        float4 vel4_j = __ldg(&veleval[index_j]);
                        float dvx = vel_i.x - vel4_j.x;
                        float dvy = vel_i.y - vel4_j.y;
                        float dvz = vel_i.z - vel4_j.z;
                        float dv_sq = dvx * dvx + dvy * dvy + dvz * dvz;

                        float rho_j = __ldg(&density[index_j]);
                        // Spiky gradient magnitude for strain rate: (h - r)^2 / r
                        float h_r = c_fluid.smoothing_length - rlen;
                        float grad_mag = h_r * h_r / rlen;
                        gamma_dot_sq_sum += (m_j / rho_j) * dv_sq * grad_mag;
                    }
                }
            }
        }

        // Apply SPH gradient normalization: 45/(pi*h^6)
        float lap_const = c_precalc.pressure_precalc;  // = 45/(pi*h^6)
        gamma_dot_i = sqrtf(fmaxf(lap_const * gamma_dot_sq_sum, 0.0f));

        // --- mu(I) rheology computation for particle i ---
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

                uint hash = (uint)(cz * ry * rx + cy * rx + cx);
                uint start = cell_start[hash];

                // Empty cell sentinel
                if (start == 0xFFFFFFFFu) continue;
                uint end = cell_end[hash];

                for (uint index_j = start; index_j < end; index_j++) {
                    // Skip self
                    if (index_j == index_i) continue;

                    // Read neighbor position
                    float4 pos4_j = __ldg(&position[index_j]);
                    float3 pos_j = make_float3(pos4_j.x, pos4_j.y, pos4_j.z);

                    // r = pos_i - pos_j (points away from neighbor)
                    float3 r = make_float3(
                        pos_i.x - pos_j.x,
                        pos_i.y - pos_j.y,
                        pos_i.z - pos_j.z
                    );
                    float rlen_sq = r.x * r.x + r.y * r.y + r.z * r.z;

                    // Distance check: within smoothing radius?
                    if (rlen_sq > h_sq || rlen_sq < 1e-12f) continue;

                    float rlen = sqrtf(rlen_sq);

                    // Read neighbor data
                    float4 vel4_j = __ldg(&veleval[index_j]);
                    float3 vel_j = make_float3(vel4_j.x, vel4_j.y, vel4_j.z);
                    float rho_j = __ldg(&density[index_j]);

                    // Neighbor pressure
                    int bclass_j = __ldg(&behavior_class[index_j]);
                    float p_j = compute_pressure(rho_j, bclass_j);

                    // ---- Pressure force (viscoplastic symmetrization) ----
                    // f_press += pressure_precalc * m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky_variable(r)
                    float press_sym = (p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j));
                    float3 grad_s = grad_spiky_variable(r, rlen);
                    f_pressure.x += m_j * press_sym * grad_s.x;
                    f_pressure.y += m_j * press_sym * grad_s.y;
                    f_pressure.z += m_j * press_sym * grad_s.z;

                    // ---- Viscosity force ----
                    float lap_v = lap_visc_variable(rlen);

                    if (is_granular_i && bclass_j == BEHAVIOR_GRANULAR) {
                        // mu(I) viscosity for GRANULAR-GRANULAR pairs:
                        // Estimate eta_j from pair-wise strain rate
                        float dvx = vel_i.x - vel_j.x;
                        float dvy = vel_i.y - vel_j.y;
                        float dvz = vel_i.z - vel_j.z;
                        float gamma_dot_j = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz)
                                          / fmaxf(rlen, 1e-8f);

                        float p_eff_j = fmaxf(p_j, 1.0f);
                        float eta_j = compute_muI_eta(gamma_dot_j, p_eff_j, rho_j);

                        // Harmonic mean viscosity
                        float eta_ij = 2.0f * eta_i * eta_j / (eta_i + eta_j + 1e-8f);

                        // Full viscosity force: eta_ij * lap_const * m_j * (v_j - v_i) / rho_j * (h - |r|)
                        float visc_lap_const = c_precalc.pressure_precalc;  // = 45/(pi*h^6)
                        float visc_factor = eta_ij * visc_lap_const * m_j * lap_v / rho_j;
                        f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
                    } else if (is_granular_i) {
                        // GRANULAR-nonGRANULAR pair: use constant mu0 but with
                        // full coefficient baked in (same scale as granular path)
                        float visc_factor = c_precalc.viscosity_precalc * m_j * lap_v / rho_j;
                        f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
                    } else {
                        // FLUID/GAS particle i: constant mu0 viscosity
                        // (viscosity_precalc applied in PostCalc)
                        float visc_factor = m_j * lap_v / rho_j;
                        f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
                    }

                    // ---- XSPH correction (FLUID only) ----
                    if (bclass_i == BEHAVIOR_FLUID) {
                        float w = W_poly6(rlen_sq);
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
        // GRANULAR: all viscosity contributions already have full coefficients
        // baked in per-pair (eta_ij * lap_const for granular, viscosity_precalc for mixed)
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

    // sph_force = total_force * particle_mass (mass is constant, moved outside loop)
    sph_force_out[index_i] = make_float4(
        total_force.x * m_j,
        total_force.y * m_j,
        total_force.z * m_j,
        0.0f
    );

    // XSPH-corrected veleval (FLUID only; others keep original veleval)
    if (bclass_i == BEHAVIOR_FLUID) {
        float eps = c_fluid.xsph_epsilon;
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
