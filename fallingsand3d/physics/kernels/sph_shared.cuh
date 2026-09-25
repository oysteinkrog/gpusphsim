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

/* Wake stamp for this substep: *frame_ptr + 1 (never 0, the value of a fresh
 * table), or 1 when frame_ptr is NULL (the caller then clears the table).
 * See wake.cu for the generation-stamp scheme. */
__device__ __forceinline__ uint wake_stamp(const uint* frame_ptr) {
    return frame_ptr ? (*frame_ptr + 1u) : 1u;
}

/* Own-cell wake mark written by a solver's final kernel from registers
 * (bd-r5-review-n28.167). Same predicate as K_MarkWakeCells: the particle
 * just woke, or it is awake, not STATIC or GAS, and moves faster than
 * V_WAKE. pi is the packed_info being written back. NULL flags: no mark. */
__device__ __forceinline__ void wake_mark_own_cell(
    uint* cell_wake_flags, const uint* frame_ptr,
    uint pi, int behavior, float vel_sq, float3 pos)
{
    if (cell_wake_flags == 0) return;
    bool mark = HAS_JUST_WOKE(pi)
             || (!IS_SLEEPING(pi) && behavior != STATIC && behavior != GAS
                 && vel_sq > V_WAKE_SQ);
    if (mark) cell_wake_flags[spatialHash(calcGridCell(pos))] = wake_stamp(frame_ptr);
}

/* Temperature integration */
#define T_AMBIENT        293.0f
#define COOL_RATE        0.02f
#define T_MIN            0.0f
#define T_MAX            5000.0f

/* GAS physics */
#define GAS_AMBIENT_TEMP   293.0f
#define GAS_DRAG_COEFF     2.0f

/* Buoyancy from the live gravity vector c_sim.gravity (all three solvers).
 *
 * GAS (ideal gas in ambient air, Archimedes): the displaced air pushes back
 * with -g * rho_air/rho_gas = -g * (T/T_amb), plus the per-material
 * MaterialProps.buoyancy_extra. Added on top of +g, the net is
 *   a = -g * ((T - T_amb)/T_amb + buoyancy_extra)
 * so gas at T_amb is weightless, hot gas moves against gravity, and the
 * direction follows the gravity slider (including a flipped sign).
 *
 * FLUID (Boussinesq): a = -g * beta * (T - T_amb). */
__device__ __forceinline__ float3 gas_buoyancy_accel(float temp, uint mat_id)
{
    float k = temp * (1.0f / GAS_AMBIENT_TEMP) + c_materials[mat_id].buoyancy_extra;
    return make_float3(-k * c_sim.gravity.x, -k * c_sim.gravity.y, -k * c_sim.gravity.z);
}

__device__ __forceinline__ float3 boussinesq_accel(float temp, float beta)
{
    float k = beta * (temp - T_AMBIENT);
    return make_float3(-k * c_sim.gravity.x, -k * c_sim.gravity.y, -k * c_sim.gravity.z);
}

/* Velocity limits -- now uses c_sim.velocity_limit (CFL-derived per solver) */
#define VELOCITY_LIMIT     (c_sim.velocity_limit)
#define VELOCITY_LIMIT_SQ  (c_sim.velocity_limit * c_sim.velocity_limit)

/* Boundary margin: prevent particles from sitting exactly at wall positions */
#define BOUNDARY_MARGIN    1e-4f

/* RHO_EPSILON now defined in common.cuh (available to all kernels) */

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
 * Per-lane accumulation for rigid body force/torque.
 *
 * Each lane atomically adds its own contribution directly. The function
 * is called inside divergent neighbor-loop branches where lanes process
 * different neighbors and may belong to different body_ids, so a
 * full-warp __shfl_down_sync reduction is incorrect (it would blend
 * values across bodies and include inactive lanes).
 * ====================================================================== */

__device__ inline void warp_reduce_accumulate(float* target, float3 val, int body_id) {
    atomicAdd(&target[body_id * 4 + 0], val.x);
    atomicAdd(&target[body_id * 4 + 1], val.y);
    atomicAdd(&target[body_id * 4 + 2], val.z);
}

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
 * With r = pos_i - pos_j and spiky_grad_coeff < 0 this is the true grad_i W,
 * which points toward the neighbor.
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
    } else if (gamma == 7.0f) {
        float r2 = ratio * ratio;
        float r4 = r2 * r2;
        p_raw = k * (r4 * r2 * ratio - 1.0f);
    } else {
        p_raw = k * (powf(ratio, gamma) - 1.0f);
    }

    return fmaxf(p_raw, 0.0f);
}

/* ======================================================================
 * mu(I) effective viscosity for GRANULAR particles.
 * Requires c_granular to be uploaded by the host.
 * ====================================================================== */

/* Strain-rate floor (1/s) for the mu(I) viscosity denominator. Below it
 * eta = mu0 + mu(I) * p_eff / MUI_GAMMA_DOT_FLOOR, so eta stays finite and
 * scales with pressure instead of growing as 1 / gamma_dot. With the current
 * p_floor (rho0 * g * spacing = 490 Pa for SAND) and mu_max = 500 this value
 * is still above mu_max, so resting sand stays at the cap
 * (bd-r5-review-n28.107). */
#define MUI_GAMMA_DOT_FLOOR 0.1f

/* mu(I) with a caller-supplied static friction mu_s. mu_2 keeps the global
 * gap: mu_2 = mu_s + (c_granular.mu_2 - c_granular.mu_s)
 * (bd-r5-review-n28.89). */
__device__ inline float compute_muI_eta(float gamma_dot, float p_eff, float rho,
                                        float mu_s) {
    float spacing = c_granular.particle_spacing;
    float I_number = gamma_dot * spacing / sqrtf(p_eff / rho);
    float mu_I = mu_s
               + (c_granular.mu_2 - c_granular.mu_s)
                 / (1.0f + c_granular.I0 / fmaxf(I_number, 1e-8f));
    return fminf(c_granular.mu_max,
                 c_granular.mu0
                 + mu_I * p_eff / fmaxf(gamma_dot, MUI_GAMMA_DOT_FLOOR));
}

/* mu(I) with the global c_granular.mu_s. */
__device__ inline float compute_muI_eta(float gamma_dot, float p_eff, float rho) {
    return compute_muI_eta(gamma_dot, p_eff, rho, c_granular.mu_s);
}

/* Per-material static friction: MaterialProps.friction_coeff, or the global
 * c_granular.mu_s when the table value is 0 (bd-r5-review-n28.89). */
__device__ inline float granular_mu_s(uint mat_id) {
    float mu = c_materials[mat_id].friction_coeff;
    return (mu > 0.0f) ? mu : c_granular.mu_s;
}

/* ======================================================================
 * Grid / boundary helpers
 * ====================================================================== */

/* Alias for calcGridCell (convenience used by PBF/DFSPH) */
__device__ inline int3 get_cell(float3 pos) { return calcGridCell(pos); }

/* Simple axis-aligned clamp to world bounds */
__device__ inline void clamp_boundary(float3& pos) {
    pos.x = fmaxf(c_sim.world_min.x + BOUNDARY_MARGIN, fminf(pos.x, c_sim.world_max.x - BOUNDARY_MARGIN));
    pos.y = fmaxf(c_sim.world_min.y + BOUNDARY_MARGIN, fminf(pos.y, c_sim.world_max.y - BOUNDARY_MARGIN));
    pos.z = fmaxf(c_sim.world_min.z + BOUNDARY_MARGIN, fminf(pos.z, c_sim.world_max.z - BOUNDARY_MARGIN));
}

/* ======================================================================
 * Quaternion rotation helpers.
 *
 * Convention: q = (x, y, z, w) where w is the scalar part.
 * quat_rotate:     rotate v by q
 * quat_rotate_inv: rotate v by q^{-1} (conjugate, same as inverse for unit q)
 * ====================================================================== */

__device__ inline float3 quat_rotate(float4 q, float3 v) {
    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;
    float d = u.x * v.x + u.y * v.y + u.z * v.z;          // dot(u, v)
    float uu = u.x * u.x + u.y * u.y + u.z * u.z;         // dot(u, u)
    float3 crs = make_float3(                                // cross(u, v)
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
    float a = s * s - uu;
    return make_float3(
        2.0f * d * u.x + a * v.x + 2.0f * s * crs.x,
        2.0f * d * u.y + a * v.y + 2.0f * s * crs.y,
        2.0f * d * u.z + a * v.z + 2.0f * s * crs.z
    );
}

__device__ inline float3 quat_rotate_inv(float4 q, float3 v) {
    float4 qc = make_float4(-q.x, -q.y, -q.z, q.w);
    return quat_rotate(qc, v);
}

/* ======================================================================
 * Analytical SDF primitives.
 *
 * Each function returns signed distance (negative = inside).
 * Normal functions return the outward unit normal at the query point.
 *
 * Box:      arbitrary orientation via quaternion
 * Sphere:   position + radius
 * Cylinder: position + quaternion, local Y axis, capped ends
 * Plane:    normal stored in size.xyz, position = point on plane
 * ====================================================================== */

/* --- Box SDF (exact exterior + interior signed distance) --- */
__device__ inline float sdf_box(float3 local_pos, float3 half_ext) {
    float3 d = make_float3(
        fabsf(local_pos.x) - half_ext.x,
        fabsf(local_pos.y) - half_ext.y,
        fabsf(local_pos.z) - half_ext.z
    );
    // Exterior: length(max(d, 0))
    float3 d_clamped = make_float3(fmaxf(d.x, 0.0f), fmaxf(d.y, 0.0f), fmaxf(d.z, 0.0f));
    float exterior = sqrtf(d_clamped.x * d_clamped.x + d_clamped.y * d_clamped.y + d_clamped.z * d_clamped.z);
    // Interior: min(max(d.x, d.y, d.z), 0)
    float interior = fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0f);
    return exterior + interior;
}

__device__ inline float3 sdf_box_normal(float3 local_pos, float3 half_ext) {
    float3 d = make_float3(
        fabsf(local_pos.x) - half_ext.x,
        fabsf(local_pos.y) - half_ext.y,
        fabsf(local_pos.z) - half_ext.z
    );
    float3 sign = make_float3(
        local_pos.x >= 0.0f ? 1.0f : -1.0f,
        local_pos.y >= 0.0f ? 1.0f : -1.0f,
        local_pos.z >= 0.0f ? 1.0f : -1.0f
    );

    // Outside: gradient is the clamped vector normalized
    if (d.x > 0.0f || d.y > 0.0f || d.z > 0.0f) {
        float3 g = make_float3(
            fmaxf(d.x, 0.0f) * sign.x,
            fmaxf(d.y, 0.0f) * sign.y,
            fmaxf(d.z, 0.0f) * sign.z
        );
        float len = sqrtf(g.x * g.x + g.y * g.y + g.z * g.z);
        if (len > 1e-8f) {
            float inv = 1.0f / len;
            return make_float3(g.x * inv, g.y * inv, g.z * inv);
        }
    }
    // Inside: normal points along axis with smallest penetration
    if (d.x >= d.y && d.x >= d.z) return make_float3(sign.x, 0.0f, 0.0f);
    if (d.y >= d.x && d.y >= d.z) return make_float3(0.0f, sign.y, 0.0f);
    return make_float3(0.0f, 0.0f, sign.z);
}

/* --- Sphere SDF --- */
__device__ inline float sdf_sphere(float3 rel_pos, float radius) {
    float len = sqrtf(rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z);
    return len - radius;
}

__device__ inline float3 sdf_sphere_normal(float3 rel_pos) {
    float len = sqrtf(rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z);
    if (len > 1e-8f) {
        float inv = 1.0f / len;
        return make_float3(rel_pos.x * inv, rel_pos.y * inv, rel_pos.z * inv);
    }
    return make_float3(0.0f, 1.0f, 0.0f);  // degenerate: point at center
}

/* --- Cylinder SDF (local Y axis, capped) --- */
__device__ inline float sdf_cylinder(float3 local_pos, float radius, float half_h) {
    float radial = sqrtf(local_pos.x * local_pos.x + local_pos.z * local_pos.z) - radius;
    float axial = fabsf(local_pos.y) - half_h;
    // Same pattern as box SDF but in 2D (radial, axial)
    float ext_r = fmaxf(radial, 0.0f);
    float ext_a = fmaxf(axial, 0.0f);
    float exterior = sqrtf(ext_r * ext_r + ext_a * ext_a);
    float interior = fminf(fmaxf(radial, axial), 0.0f);
    return exterior + interior;
}

__device__ inline float3 sdf_cylinder_normal(float3 local_pos, float radius, float half_h) {
    float radial_dist = sqrtf(local_pos.x * local_pos.x + local_pos.z * local_pos.z);
    float radial = radial_dist - radius;
    float axial = fabsf(local_pos.y) - half_h;
    float y_sign = local_pos.y >= 0.0f ? 1.0f : -1.0f;

    if (radial > 0.0f || axial > 0.0f) {
        // Outside: gradient direction
        float gr = fmaxf(radial, 0.0f);
        float ga = fmaxf(axial, 0.0f);
        if (gr > 1e-8f && radial_dist > 1e-8f) {
            float inv_rd = 1.0f / radial_dist;
            float3 n = make_float3(gr * local_pos.x * inv_rd, ga * y_sign, gr * local_pos.z * inv_rd);
            float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
            if (len > 1e-8f) { float inv = 1.0f / len; return make_float3(n.x*inv, n.y*inv, n.z*inv); }
        }
        if (ga > 0.0f) return make_float3(0.0f, y_sign, 0.0f);
    }
    // Inside: use axis with smallest penetration
    if (radial >= axial) {
        if (radial_dist > 1e-8f) {
            float inv = 1.0f / radial_dist;
            return make_float3(local_pos.x * inv, 0.0f, local_pos.z * inv);
        }
        return make_float3(1.0f, 0.0f, 0.0f);
    }
    return make_float3(0.0f, y_sign, 0.0f);
}

/* --- Plane SDF --- */
__device__ inline float sdf_plane(float3 pos, float3 normal, float3 point) {
    return normal.x * (pos.x - point.x) + normal.y * (pos.y - point.y) + normal.z * (pos.z - point.z);
}

/* --- Unified eval_sdf: dispatch by type --- */
__device__ inline float eval_sdf(float3 pos, const SDFObject& obj) {
    int type = __float_as_int(obj.pos_and_type.w);
    float3 center = make_float3(obj.pos_and_type.x, obj.pos_and_type.y, obj.pos_and_type.z);

    if (type == SDF_PLANE) {
        float3 normal = make_float3(obj.size_and_r.x, obj.size_and_r.y, obj.size_and_r.z);
        return sdf_plane(pos, normal, center);
    }

    float3 rel = make_float3(pos.x - center.x, pos.y - center.y, pos.z - center.z);

    if (type == SDF_SPHERE) {
        return sdf_sphere(rel, obj.size_and_r.x);
    }

    // Transform to local space for oriented primitives
    float3 local_pos = quat_rotate_inv(obj.quat, rel);

    if (type == SDF_BOX) {
        float3 half_ext = make_float3(obj.size_and_r.x, obj.size_and_r.y, obj.size_and_r.z);
        return sdf_box(local_pos, half_ext);
    }

    if (type == SDF_CYLINDER) {
        return sdf_cylinder(local_pos, obj.size_and_r.x, obj.size_and_r.y);
    }

    return 1e10f;  // unknown type: no collision
}

/* --- Unified sdf_normal: analytical outward normal --- */
__device__ inline float3 sdf_normal(float3 pos, const SDFObject& obj) {
    int type = __float_as_int(obj.pos_and_type.w);
    float3 center = make_float3(obj.pos_and_type.x, obj.pos_and_type.y, obj.pos_and_type.z);

    if (type == SDF_PLANE) {
        return make_float3(obj.size_and_r.x, obj.size_and_r.y, obj.size_and_r.z);
    }

    float3 rel = make_float3(pos.x - center.x, pos.y - center.y, pos.z - center.z);

    if (type == SDF_SPHERE) {
        return sdf_sphere_normal(rel);
    }

    float3 local_pos = quat_rotate_inv(obj.quat, rel);
    float3 local_normal;

    if (type == SDF_BOX) {
        float3 half_ext = make_float3(obj.size_and_r.x, obj.size_and_r.y, obj.size_and_r.z);
        local_normal = sdf_box_normal(local_pos, half_ext);
    } else if (type == SDF_CYLINDER) {
        local_normal = sdf_cylinder_normal(local_pos, obj.size_and_r.x, obj.size_and_r.y);
    } else {
        return make_float3(0.0f, 1.0f, 0.0f);
    }

    // Rotate normal back to world space
    return quat_rotate(obj.quat, local_normal);
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
    // I5 fix: use BOUNDARY_MARGIN to prevent particles sitting exactly at walls
    float3 wmin = make_float3(world_min.x + BOUNDARY_MARGIN, world_min.y + BOUNDARY_MARGIN, world_min.z + BOUNDARY_MARGIN);
    float3 wmax = make_float3(world_max.x - BOUNDARY_MARGIN, world_max.y - BOUNDARY_MARGIN, world_max.z - BOUNDARY_MARGIN);

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
 * rigid_integrate_pose -- advance a rigid body pose by one step.
 *
 * pos += v * dt; q += 0.5 * dt * (omega, 0) * q, then renormalize q.
 * Shared by dynamic bodies (after the velocity update), kinematic bodies
 * (prescribed velocity) and rigid_body_boundary (end-of-substep pose).
 * (bd-r5-review-n28.105)
 * ====================================================================== */
__device__ __forceinline__
void rigid_integrate_pose(float3& pos, float4& q, float3 lin_vel, float3 ang_vel, float dt) {
    pos.x += lin_vel.x * dt;
    pos.y += lin_vel.y * dt;
    pos.z += lin_vel.z * dt;

    // quat_mult(omega_q, q) = (s1*v2 + s2*v1 + cross(v1,v2), s1*s2 - dot(v1,v2))
    // where omega_q has s1=0, v1=omega; q has s2=q.w, v2=(q.x,q.y,q.z)
    float3 qv = make_float3(q.x, q.y, q.z);
    float qs = q.w;
    float3 cr = make_float3(
        ang_vel.y * qv.z - ang_vel.z * qv.y,
        ang_vel.z * qv.x - ang_vel.x * qv.z,
        ang_vel.x * qv.y - ang_vel.y * qv.x
    );
    float dot_ov = ang_vel.x * qv.x + ang_vel.y * qv.y + ang_vel.z * qv.z;
    float4 dq = make_float4(
        qs * ang_vel.x + cr.x,
        qs * ang_vel.y + cr.y,
        qs * ang_vel.z + cr.z,
        -dot_ov
    );
    q.x += 0.5f * dt * dq.x;
    q.y += 0.5f * dt * dq.y;
    q.z += 0.5f * dt * dq.z;
    q.w += 0.5f * dt * dq.w;

    float q_len = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float q_inv = 1.0f / fmaxf(q_len, 1e-8f);
    q.x *= q_inv; q.y *= q_inv; q.z *= q_inv; q.w *= q_inv;
}

/* ======================================================================
 * Positional fluid collision against rigid bodies (bd-r5-review-n28.123).
 *
 * The Akinci force from the body's one-spacing boundary layer cannot stop
 * a body that moves more than about a spacing per substep (PBF at
 * dt = 1/120 and 3 m/s moves 2.5 cm), so the body overtook fluid and
 * trapped it inside. This is the same push-out and impulse as the SDF
 * object loop, run on each body's box or sphere SDF.
 *
 * The SDF is evaluated at the body's end-of-substep pose: the stored pose
 * advanced by one dt with the stored lin_vel / ang_vel. The fluid kernels
 * run before K_IntegrateRigidBodies, so testing the current pose would
 * leave every particle in front of a moving face up to v*dt inside once
 * the body moves. For a kinematic body this is exactly the pose
 * K_IntegrateRigidBodies writes; for a dynamic body it misses only this
 * substep's velocity change.
 *
 * rigid_body_sdf returns the signed distance to body b at that pose and
 * the outward world normal, contact centre com and surface velocity terms.
 * A bounding-sphere test skips far bodies before the rotation.
 * ====================================================================== */
__device__ inline bool rigid_body_sdf(
    float3 pos, const RigidBody* __restrict__ bodies, int b, float dt,
    float& dist, float3& n, float3& com, float3& v_b, float3& w_b,
    float& restitution, float& friction
) {
    float4 p4 = __ldg(&bodies[b].position);
    float4 q  = __ldg(&bodies[b].rotation);
    float4 lv = __ldg(&bodies[b].lin_vel);
    float4 av = __ldg(&bodies[b].ang_vel);
    float4 he = __ldg(&bodies[b].half_extents);
    com = make_float3(p4.x, p4.y, p4.z);
    v_b = make_float3(lv.x, lv.y, lv.z);
    w_b = make_float3(av.x, av.y, av.z);
    restitution = lv.w;
    friction = av.w;
    rigid_integrate_pose(com, q, v_b, w_b, dt);

    float3 rel = make_float3(pos.x - com.x, pos.y - com.y, pos.z - com.z);
    float r_sq = rel.x * rel.x + rel.y * rel.y + rel.z * rel.z;
    int shape = __float_as_int(he.w);
    float r_bound = (shape == SDF_SPHERE)
        ? he.x : sqrtf(he.x * he.x + he.y * he.y + he.z * he.z);
    r_bound += BOUNDARY_MARGIN;
    if (r_sq >= r_bound * r_bound) return false;

    if (shape == SDF_SPHERE) {
        dist = sdf_sphere(rel, he.x);
        n = sdf_sphere_normal(rel);
    } else {
        float3 half_ext = make_float3(he.x, he.y, he.z);
        float3 local = quat_rotate_inv(q, rel);
        dist = sdf_box(local, half_ext);
        n = quat_rotate(q, sdf_box_normal(local, half_ext));
    }
    return true;
}

/* True when pos is strictly inside any body at its end-of-substep pose.
 * Used by the sleeping-particle geometry wake test. */
__device__ inline bool rigid_body_penetrates(
    float3 pos, const RigidBody* __restrict__ bodies, int num_bodies, float dt
) {
    for (int b = 0; b < num_bodies; b++) {
        float dist, rest, fric;
        float3 n, com, v_b, w_b;
        if (rigid_body_sdf(pos, bodies, b, dt, dist, n, com, v_b, w_b, rest, fric)
            && dist < 0.0f) return true;
    }
    return false;
}

/* Push pos out of every body it is within BOUNDARY_MARGIN of, to
 * BOUNDARY_MARGIN outside the surface along the SDF normal. When the
 * particle moves into the surface relative to the surface velocity
 * v_body + omega x r, the normal part of that relative velocity is
 * reflected with the body's restitution and the tangential part is cut by
 * Coulomb friction with the body's friction, as in the SDF object loop.
 * The body gets no reaction here; the Akinci pressure coupling carries the
 * fluid force on it. */
__device__ inline void rigid_body_boundary(
    float3& pos, float3& vel,
    const RigidBody* __restrict__ bodies, int num_bodies, float dt
) {
    for (int b = 0; b < num_bodies; b++) {
        float dist, rest, fric;
        float3 n, com, v_b, w_b;
        if (!rigid_body_sdf(pos, bodies, b, dt, dist, n, com, v_b, w_b, rest, fric))
            continue;
        if (dist >= BOUNDARY_MARGIN) continue;

        float push = BOUNDARY_MARGIN - dist;
        pos.x += push * n.x;
        pos.y += push * n.y;
        pos.z += push * n.z;

        float3 r = make_float3(pos.x - com.x, pos.y - com.y, pos.z - com.z);
        float3 v_s = make_float3(
            v_b.x + w_b.y * r.z - w_b.z * r.y,
            v_b.y + w_b.z * r.x - w_b.x * r.z,
            v_b.z + w_b.x * r.y - w_b.y * r.x
        );
        float3 v_rel = make_float3(vel.x - v_s.x, vel.y - v_s.y, vel.z - v_s.z);
        float v_dot_n = v_rel.x * n.x + v_rel.y * n.y + v_rel.z * n.z;
        if (v_dot_n < 0.0f) {
            vel.x -= (1.0f + rest) * v_dot_n * n.x;
            vel.y -= (1.0f + rest) * v_dot_n * n.y;
            vel.z -= (1.0f + rest) * v_dot_n * n.z;
            float3 v_tan = make_float3(
                v_rel.x - v_dot_n * n.x,
                v_rel.y - v_dot_n * n.y,
                v_rel.z - v_dot_n * n.z
            );
            float v_tan_len = sqrtf(v_tan.x * v_tan.x + v_tan.y * v_tan.y + v_tan.z * v_tan.z);
            if (v_tan_len > 1e-8f) {
                float red = fminf(fric * fabsf(v_dot_n) / v_tan_len, 1.0f);
                vel.x -= red * v_tan.x;
                vel.y -= red * v_tan.y;
                vel.z -= red * v_tan.z;
            }
        }
    }
}

/* ======================================================================
 * STATIC particle boundary repulsion.
 *
 * After position integration, scan for nearby STATIC particles and push
 * the fluid/granular particle away to prevent leaking through solid
 * boundaries.
 *
 * The band is min_dist = 0.5h, one rest spacing (dx = 0.02 m at
 * h = 0.04 m).  That is the Akinci equilibrium distance, so a particle
 * resting one spacing from the wall is not touched; Akinci density and
 * pressure in step2 hold it there.
 *
 * The push is one correction, not a sum over walls: direction
 * n = normalise(sum_j deficit_j * r_hat_j), length max_j deficit_j.
 * So a particle over a flat floor moves by at most its deepest
 * penetration, however many floor particles are in the band.
 * Velocity is reflected with restitution along n only, and only when
 * the particle moves into the wall.
 *
 * Requires: cell_start, cell_end, packed_info, position arrays in sorted
 * order (same grid as the current step).
 *
 * K_PBF_Finalize and K_DFSPH_Finalize use the same StaticPush helpers
 * inside their XSPH neighbour loop and call this function only for GAS
 * (no XSPH loop) and, in DFSPH, for fast particles and FLUID near a wall.
 * ====================================================================== */

/* Accumulator for the STATIC push. static_particle_boundary uses it, and
 * the PBF and DFSPH Finalize neighbour loops feed it directly so they do
 * not need a second 27-cell scan (bd-r5-review-n28.164). */
struct StaticPush {
    float3 dir_sum;
    float  max_deficit;
};

__device__ inline StaticPush static_push_init() {
    StaticPush a;
    a.dir_sum = make_float3(0.0f, 0.0f, 0.0f);
    a.max_deficit = 0.0f;
    return a;
}

/* Band radius of the push: one rest spacing, 0.5h. */
__device__ inline float static_push_band() {
    return 0.5f * c_sim.smoothing_length;
}

/* True when neighbour pi_j is a wall the push acts on: STATIC, not a free
 * DEAD slot, and not MAT_RIGID unless push_from_rigid (GAS only). */
__device__ inline bool static_push_source(uint pi_j, bool push_from_rigid) {
    if (GET_BEHAVIOR(pi_j) != STATIC) return false;
    uint m = GET_MATERIAL_ID(pi_j);
    if (m == MAT_RIGID && !push_from_rigid) return false;  // Akinci handled by pressure coupling
    if (m == MAT_DEAD) return false;                        // free slot, not a wall
    return true;
}

/* Add one wall neighbour at offset r = x_i - x_j (r_sq = |r|^2). */
__device__ inline void static_push_add(StaticPush& a, float3 r, float r_sq,
                                       float min_dist, float min_dist_sq) {
    if (r_sq < min_dist_sq && r_sq > 1e-12f) {
        float rlen = sqrtf(r_sq);
        float deficit = min_dist - rlen;
        float w = deficit / rlen;
        a.dir_sum.x += w * r.x;
        a.dir_sum.y += w * r.y;
        a.dir_sum.z += w * r.z;
        a.max_deficit = fmaxf(a.max_deficit, deficit);
    }
}

/* Move pos by max_deficit along the mean normal and reflect vel along it. */
__device__ inline void static_push_apply(const StaticPush& a, float3& pos, float3& vel,
                                         float restitution) {
    if (a.max_deficit > 0.0f) {
        float dir_len = sqrtf(a.dir_sum.x*a.dir_sum.x + a.dir_sum.y*a.dir_sum.y + a.dir_sum.z*a.dir_sum.z);
        if (dir_len > 1e-8f) {
            float3 n = make_float3(a.dir_sum.x/dir_len, a.dir_sum.y/dir_len, a.dir_sum.z/dir_len);
            pos.x += a.max_deficit * n.x;
            pos.y += a.max_deficit * n.y;
            pos.z += a.max_deficit * n.z;
            float v_dot_n = vel.x*n.x + vel.y*n.y + vel.z*n.z;
            if (v_dot_n < 0.0f) {
                vel.x -= (1.0f + restitution) * v_dot_n * n.x;
                vel.y -= (1.0f + restitution) * v_dot_n * n.y;
                vel.z -= (1.0f + restitution) * v_dot_n * n.z;
            }
        }
    }
}

__device__ inline void static_particle_boundary(
    float3& pos, float3& vel,
    const uint* __restrict__ cell_start,
    const uint* __restrict__ cell_end,
    const uint* __restrict__ packed_info,
    const float4* __restrict__ position,  // sorted positions of all particles
    uint self_idx,                        // sorted index of current particle
    float restitution
) {
    float min_dist = static_push_band();
    float min_dist_sq = min_dist * min_dist;

    // GAS gets the push from MAT_RIGID body particles too: its density and
    // pressure ignore the body, so the Akinci term alone cannot stop gas
    // entering a moving body (bd-r5-review-n28.101). Other behaviours
    // leave MAT_RIGID to the solver's pressure coupling.
    bool push_from_rigid = (GET_BEHAVIOR(__ldg(&packed_info[self_idx])) == GAS);

    int3 cell = calcGridCell(pos);
    StaticPush acc = static_push_init();

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint hash = spatialHash(cell.x+dx, cell.y+dy, cell.z+dz);
                uint start = cell_start[hash];
                if (start == 0xFFFFFFFFu) continue;
                uint end = cell_end[hash];
                for (uint j = start; j < end; j++) {
                    if (j == self_idx) continue;
                    uint pi_j = __ldg(&packed_info[j]);
                    if (!static_push_source(pi_j, push_from_rigid)) continue;
                    float4 pos4_j = __ldg(&position[j]);
                    float3 r = make_float3(
                        pos.x - pos4_j.x,
                        pos.y - pos4_j.y,
                        pos.z - pos4_j.z
                    );
                    float r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
                    static_push_add(acc, r, r_sq, min_dist, min_dist_sq);
                }
            }
        }
    }

    static_push_apply(acc, pos, vel, restitution);
}

/* ======================================================================
 * Particle color computation
 * ====================================================================== */

/* Encode behavior class into color.w for SSFR material filtering.
 * FLUID=0.0, GRANULAR=0.25, GAS=0.5, STATIC=0.75. DEAD slots get -1.0. */
__device__ inline float behavior_to_alpha(int behavior) {
    return behavior * 0.25f;
}

/* color.w value for a DEAD slot. Every particle vertex shader drops a vertex
 * with color.w < 0, so a dead slot is invisible from the next substep on,
 * without waiting for compaction. position.w is not free for this: step1
 * stores density there. */
#define DEAD_COLOR_ALPHA (-1.0f)

__device__ inline float4 compute_color(uint mat_id, float temperature, float health, int behavior) {
    if (mat_id == MAT_DEAD) return make_float4(0.0f, 0.0f, 0.0f, DEAD_COLOR_ALPHA);
    float r = c_materials[mat_id].color_r;
    float g = c_materials[mat_id].color_g;
    float b = c_materials[mat_id].color_b;

    // Material-specific heat glow approaching melt point
    float temp_melt = c_materials[mat_id].temp_melt;
    if (temp_melt > 501.0f && temperature > 500.0f) {
        float heat_frac = fminf((temperature - 500.0f) / (temp_melt - 500.0f), 1.0f);
        heat_frac = fmaxf(heat_frac, 0.0f);
        // STONE, METAL, SAND, GRAVEL: glow orange-red
        r = r + (1.0f - r) * heat_frac * 0.8f;
        g = g * (1.0f - heat_frac * 0.4f) + 0.3f * heat_frac;  // slight orange
        b = b * (1.0f - heat_frac * 0.9f);
    }
    // ICE: glow blue-white as temperature approaches 273K
    else if (mat_id == 11 && temperature > 250.0f) {  // MAT_ICE=11
        float ice_frac = fminf((temperature - 250.0f) / 23.0f, 1.0f);  // 250->273K
        r = r + (1.0f - r) * ice_frac * 0.3f;
        g = g + (1.0f - g) * ice_frac * 0.1f;
        b = fminf(b + ice_frac * 0.1f, 1.0f);
    }
    // Generic hot tint for other materials
    else if (temperature > 293.0f) {
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
    if (mat_id == MAT_DEAD) return make_float4(0.0f, 0.0f, 0.0f, DEAD_COLOR_ALPHA);
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
