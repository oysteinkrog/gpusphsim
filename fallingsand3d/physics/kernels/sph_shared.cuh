/*
 * sph_shared.cuh -- Shared device functions and constants for all SPH solvers.
 *
 * Includes common.cuh and adds:
 *   - Shared simulation constants (sleep, temperature, GAS, velocity)
 *   - GranularParams struct + constant memory
 *   - SPH kernel functions (W_poly6, grad_spiky, etc.)
 *   - EOS pressure computation
 *   - mu(I) effective viscosity
 *   - Grid/boundary helpers (get_cell, clamp_boundary, sdf_box_boundary)
 *   - Color computation (compute_color, compute_fluid_color)
 *
 * All .cu solver files should include this header instead of common.cuh.
 */

#ifndef FALLINGSAND3D_SPH_SHARED_CUH
#define FALLINGSAND3D_SPH_SHARED_CUH

#include "common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ======================================================================
 * Shared simulation constants
 * ====================================================================== */

/* Sleep/wake system with hysteresis */
#define V_SLEEP          0.005f
#define V_SLEEP_SQ       (V_SLEEP * V_SLEEP)
#define V_WAKE           0.02f
#define V_WAKE_SQ        (V_WAKE * V_WAKE)
#define SLEEP_THRESHOLD  10

/* Temperature integration */
#define T_AMBIENT        293.0f
#define COOL_RATE        0.02f
#define T_MIN            0.0f
#define T_MAX            5000.0f

/* GAS physics */
#define GAS_BUOYANCY_BETA  0.01f
#define GAS_AMBIENT_TEMP   293.0f
#define GAS_BUOYANCY_G     9.81f
#define GAS_DRAG_COEFF     2.0f

/* Velocity limits */
#define VELOCITY_LIMIT     10.0f
#define VELOCITY_LIMIT_SQ  (VELOCITY_LIMIT * VELOCITY_LIMIT)

/* ======================================================================
 * GranularParams -- mu(I) rheology and shared solver parameters.
 *
 * Declared in constant memory. Each CuPy RawModule has its own copy;
 * uploaded from Python before kernel launch in solvers that use it.
 * Files that don't upload granular params get zero-initialized memory.
 * ====================================================================== */

struct GranularParams {
    float mu_s;                  // static friction coefficient (0.36)
    float mu_2;                  // dynamic friction coefficient (0.70)
    float I0;                    // inertial number reference (0.3)
    float mu_max;                // viscosity clamp (Pa*s)
    float particle_spacing;      // rest spacing d
    float mu0;                   // base viscosity
    float xsph_epsilon;          // XSPH blending factor
    float force_scale;           // SPH force output scaling
    float vorticity_epsilon;     // vorticity confinement strength
    float surface_tension_gamma; // Akinci surface tension coefficient
    float tan_phi_f;             // tan(friction_angle) for Drucker-Prager
    float cohesion;              // cohesion for DP stability
};

__constant__ GranularParams c_granular;

/* ======================================================================
 * SPH kernel functions
 * ====================================================================== */

/* Poly6 kernel: W(r) = poly6_coeff * (h^2 - |r|^2)^3
 * Clamp diff >= 0 for FP safety: callers guard r <= h, but FP rounding
 * can produce rlen_sq slightly > h_sq, which would make diff^3 negative. */
__device__ inline float W_poly6(float rlen_sq, float h_sq) {
    float diff = fmaxf(h_sq - rlen_sq, 0.0f);
    return c_precalc.poly6_coeff * diff * diff * diff;
}

/* Spiky gradient WITH coefficient baked in.
 * Points away from neighbor (r = pos_i - pos_j).
 * grad_W = spiky_grad_coeff * (h-|r|)^2 / |r| * r */
__device__ inline float3 grad_spiky(float3 r, float rlen, float h) {
    float h_rlen = h - rlen;
    float coeff = c_precalc.spiky_grad_coeff * h_rlen * h_rlen / rlen;
    return make_float3(coeff * r.x, coeff * r.y, coeff * r.z);
}

/* Spiky gradient variable part ONLY (no coefficient).
 * (r/|r|) * (h - |r|)^2
 * Used with pressure_precalc applied later in PostCalc. */
__device__ inline float3 grad_spiky_variable(float3 r, float rlen, float h) {
    float h_rlen = h - rlen;
    float inv_rlen = 1.0f / rlen;
    return make_float3(
        r.x * inv_rlen * (h_rlen * h_rlen),
        r.y * inv_rlen * (h_rlen * h_rlen),
        r.z * inv_rlen * (h_rlen * h_rlen)
    );
}

/* Viscosity Laplacian variable part: (h - |r|) */
__device__ inline float lap_visc_variable(float rlen, float h) {
    return h - rlen;
}

/* ======================================================================
 * EOS pressure computation (per-material gamma)
 *
 *   gamma==1:  Linear EOS:  p = k * max(rho/rho0 - 1, 0)
 *   gamma!=1:  Tait EOS:    p = k * (pow(rho/rho0, gamma) - 1)
 *   GAS:       p = k * max(rho - rho0, 0)
 *   All: p clamped >= 0
 * ====================================================================== */

__device__ inline float compute_pressure(float rho_i, int behavior, uint mat_id) {
    float rho0  = c_materials[mat_id].rest_density;
    float k     = c_materials[mat_id].eos_stiffness;
    float gamma = c_materials[mat_id].eos_gamma;

    if (behavior == GAS) {
        return k * fmaxf(rho_i - rho0, 0.0f);
    }

    // Cap ratio to prevent powf overflow at extreme densities (NaN/Inf propagation).
    // ratio=10 with gamma=7 gives ~1e7, well within float range but still huge pressure
    // that the velocity limiter will clamp anyway.
    float ratio = fminf(rho_i / fmaxf(rho0, 1e-6f), 10.0f);

    float p_raw;
    if (gamma == 1.0f) {
        p_raw = k * fmaxf(ratio - 1.0f, 0.0f);
    } else {
        p_raw = k * (powf(ratio, gamma) - 1.0f);
    }

    return fmaxf(p_raw, 0.0f);
}

/* ======================================================================
 * mu(I) effective viscosity for GRANULAR particles.
 * Requires c_granular to be uploaded by the host.
 * ====================================================================== */

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
 * Grid / boundary helpers
 * ====================================================================== */

/* Alias for calcGridCell (convenience used by PBF/DFSPH) */
__device__ inline int3 get_cell(float3 pos) { return calcGridCell(pos); }

/* Simple axis-aligned clamp to world bounds */
__device__ inline void clamp_boundary(float3& pos) {
    pos.x = fmaxf(c_sim.world_min.x, fminf(pos.x, c_sim.world_max.x));
    pos.y = fmaxf(c_sim.world_min.y, fminf(pos.y, c_sim.world_max.y));
    pos.z = fmaxf(c_sim.world_min.z, fminf(pos.z, c_sim.world_max.z));
}

/* ======================================================================
 * Impulse-style SDF boundary collision for axis-aligned box.
 *
 * For each of 6 planes: if pos penetrates wall, project out and
 * apply impulse-style velocity correction:
 *   - Normal velocity reflected with restitution coefficient
 *   - Tangential velocity reduced by Coulomb friction
 * ====================================================================== */

__device__ inline void sdf_box_boundary(
    float3& pos, float3& vel,
    float3 world_min, float3 world_max,
    float restitution, float mu_wall
) {
    // X-axis
    if (pos.x < world_min.x) {
        pos.x = world_min.x;
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
    if (pos.x > world_max.x) {
        pos.x = world_max.x;
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
    if (pos.y < world_min.y) {
        pos.y = world_min.y;
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
    if (pos.y > world_max.y) {
        pos.y = world_max.y;
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
    if (pos.z < world_min.z) {
        pos.z = world_min.z;
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
    if (pos.z > world_max.z) {
        pos.z = world_max.z;
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
 * Particle color computation
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

/**
 * FLUID-specific color: depth gradient + velocity foam + density variation.
 *
 *   depth_t   = normalized Y position [0=bottom, 1=top of domain]
 *   foam      = velocity magnitude mapped to white highlight
 *   density_t = compression darkening
 */
__device__ inline float4 compute_fluid_color(
    uint mat_id, float temperature, float health,
    float pos_y, float vel_sq, float density
) {
    float rho0 = c_materials[mat_id].rest_density;
    float base_r = c_materials[mat_id].color_r;
    float base_g = c_materials[mat_id].color_g;
    float base_b = c_materials[mat_id].color_b;

    // Depth gradient: 0 at bottom, 1 at top
    float y_range = c_sim.world_max.y - c_sim.world_min.y;
    float depth_t = (pos_y - c_sim.world_min.y) / fmaxf(y_range, 0.01f);
    depth_t = fmaxf(0.0f, fminf(depth_t, 1.0f));
    float r = base_r * (0.45f + 0.70f * depth_t);
    float g = base_g * (0.50f + 0.65f * depth_t);
    float b = base_b * (0.65f + 0.40f * depth_t);

    // Density darkening
    float ratio = density / fmaxf(rho0, 1.0f);
    float compress = fmaxf(ratio - 1.0f, 0.0f);
    float darken = 1.0f / (1.0f + 0.5f * compress);
    r *= darken; g *= darken; b *= darken;

    // Velocity foam
    float speed = sqrtf(vel_sq);
    float foam_t = fminf(speed / 3.0f, 1.0f);
    foam_t = foam_t * foam_t;
    r = r + (1.0f - r) * foam_t * 0.7f;
    g = g + (1.0f - g) * foam_t * 0.7f;
    b = b + (1.0f - b) * foam_t * 0.7f;

    // Hot tint
    if (temperature > 293.0f) {
        float t_excess = fminf((temperature - 293.0f) / 1000.0f, 1.0f);
        r = r + (1.0f - r) * t_excess;
        g = g * (1.0f - 0.5f * t_excess);
        b = b * (1.0f - 0.8f * t_excess);
    }

    // Health fade
    float h = fmaxf(fminf(health, 1.0f), 0.0f);
    r = fminf(r * h, 1.0f);
    g = fminf(g * h, 1.0f);
    b = fminf(b * h, 1.0f);

    return make_float4(r, g, b, 0.0f);  // FLUID always 0.0
}

#endif /* FALLINGSAND3D_SPH_SHARED_CUH */
