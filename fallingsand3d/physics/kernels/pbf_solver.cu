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
 * GAS gets no density constraint. It gets buoyancy and linear drag in Predict
 * and the WCSPH gas EOS pressure inside its own phase as a position delta in
 * ComputeDelta (lambda slot holds p/rho^2 for GAS).
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

/* One wall plane of the PBF world box. `pn` is the position component along
 * the wall normal, `vn` the velocity component, `vt1`/`vt2` the tangential
 * components. `sign` is -1 for a min wall (inside is pn > wall) and +1 for a
 * max wall. Contact is `pn <= wall` (min) or `pn >= wall` (max) with velocity
 * into the wall. */
__device__ inline void pbf_box_wall(
    float& pn, float& vn, float& vt1, float& vt2,
    float wall, float sign, float restitution, float mu_wall
) {
    if (sign * (pn - wall) < 0.0f) return;   // strictly inside
    pn = wall;
    if (sign * vn <= 0.0f) return;           // not moving into the wall
    float v_in = vn;
    vn = -restitution * v_in;
    float ts = sqrtf(vt1 * vt1 + vt2 * vt2);
    if (ts > 1e-8f) {
        float red = fminf(mu_wall * fabsf(v_in) / ts, 1.0f);
        vt1 *= (1.0f - red);
        vt2 *= (1.0f - red);
    }
}

/* World box collision for K_PBF_Finalize. Same impulse as sdf_box_boundary
 * (restitution on the normal velocity, Coulomb wall friction on the
 * tangential velocity), but contact includes a particle sitting exactly on
 * the margin. K_PBF_Predict and K_PBF_ApplyDelta already clamp the predicted
 * position to world_min + BOUNDARY_MARGIN, so with sdf_box_boundary's strict
 * `<` test Finalize never saw the impact (bd-r5-review-n28.70). A resting
 * particle has zero normal velocity, so the `<=` contact leaves it unchanged. */
__device__ inline void pbf_box_boundary(
    float3& pos, float3& vel,
    float3 world_min, float3 world_max,
    float restitution, float mu_wall
) {
    float3 wmin = make_float3(world_min.x + BOUNDARY_MARGIN, world_min.y + BOUNDARY_MARGIN, world_min.z + BOUNDARY_MARGIN);
    float3 wmax = make_float3(world_max.x - BOUNDARY_MARGIN, world_max.y - BOUNDARY_MARGIN, world_max.z - BOUNDARY_MARGIN);
    pbf_box_wall(pos.x, vel.x, vel.y, vel.z, wmin.x, -1.0f, restitution, mu_wall);
    pbf_box_wall(pos.x, vel.x, vel.y, vel.z, wmax.x, +1.0f, restitution, mu_wall);
    pbf_box_wall(pos.y, vel.y, vel.x, vel.z, wmin.y, -1.0f, restitution, mu_wall);
    pbf_box_wall(pos.y, vel.y, vel.x, vel.z, wmax.y, +1.0f, restitution, mu_wall);
    pbf_box_wall(pos.z, vel.z, vel.x, vel.y, wmin.z, -1.0f, restitution, mu_wall);
    pbf_box_wall(pos.z, vel.z, vel.x, vel.y, wmax.z, +1.0f, restitution, mu_wall);
}

/* Add F_on_body and its torque about the body centre to the accumulators of
 * the body that owns boundary particle j. */
__device__ inline void pbf_rigid_reaction(
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
    warp_reduce_accumulate(d_rigid_forces, F_on_body, body_id);
    warp_reduce_accumulate(d_rigid_torques, tau, body_id);
}

/* Move a predicted position to BOUNDARY_MARGIN outside every SDF object it
 * lies inside or within the margin of. Used by K_PBF_Predict and by every
 * K_PBF_ApplyDelta iteration. Position only: K_PBF_Finalize applies the
 * restitution and friction impulse. */
__device__ __forceinline__ void push_out_of_sdf(float3& p) {
    for (int s = 0; s < c_num_sdf_objects; s++) {
        SDFObject obj = c_sdf_objects[s];
        float dist = eval_sdf(p, obj);
        if (dist < BOUNDARY_MARGIN) {
            float3 n = sdf_normal(p, obj);
            float push = BOUNDARY_MARGIN - dist;
            p.x += push * n.x;
            p.y += push * n.y;
            p.z += push * n.z;
        }
    }
}

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
        float3 b = gas_buoyancy_accel(temp, GET_MATERIAL_ID(pi));
        v_star.x += b.x * dt;
        v_star.y += b.y * dt;
        v_star.z += b.z * dt;
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
            float3 b = boussinesq_accel(temp, beta);
            v_star.x += b.x * dt;
            v_star.y += b.y * dt;
            v_star.z += b.z * dt;
        }
    }

    // x* = x + dt * v*
    float3 pred = make_float3(
        pos.x + dt * v_star.x,
        pos.y + dt * v_star.y,
        pos.z + dt * v_star.z
    );

    clamp_boundary(pred);
    push_out_of_sdf(pred);

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
    // Optional heat diffusion + exposure (NULL to skip, used only on first call)
    const float*    __restrict__ temperature_in,
    const float*    __restrict__ density_in,
    float*          __restrict__ dTdt_out,
    float*          __restrict__ exposure_heat_out,
    float*          __restrict__ exposure_corrode_out,
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
        // STATIC floor at 1: DEAD has rest_density 0. GAS keeps the density the
        // first (do_heat) call computed this substep; heat reads it as rho_j
        // next substep, so overwriting it with rest_density would be wrong.
        // GAS also keeps its lambda slot, which holds the p/rho^2 term the first
        // call wrote for the gas pressure delta (bd-r5-review-n28.91).
        if (behavior_i == STATIC) {
            density_out[i] = fmaxf(c_materials[mat_id_i].rest_density, 1.0f);
            lambda_out[i] = 0.0f;
        }
        pressure_normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float4 pos4_i = predicted_pos[i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    // Floor at 1: DEAD has rest_density 0, and C = rho / rho0 - 1 would be inf.
    float rho0 = fmaxf(c_materials[mat_id_i].rest_density, 1.0f);
    // Rigid neighbour density scale: psi_b = rho0_i * V_b (bd-r5-review-n28.77)
    float rigid_dens_scale = __fdiv_rn(c_materials[mat_id_i].rest_density, RIGID_PSI_RHO0);

    // Density accumulator (Poly6)
    float sum_density = 0.0f;
    // Gradient sum for constraint denominator (Spiky)
    float3 grad_ci = make_float3(0.0f, 0.0f, 0.0f);  // grad_pi C_i
    float sum_grad_sq = 0.0f;  // SUM_j |grad_pj C_i|^2 for j != i

    // Heat diffusion + exposure accumulators (only when do_heat)
    float sum_dTdt = 0.0f;
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;
    // Kernel support weight of all non-self, non-DEAD neighbours (GAS and
    // MAT_RIGID included). Exposure sums are divided by it, as in K_Step1.
    float sum_W = 0.0f;
    float T_i = 0.0f;
    float kappa_i = 0.0f;
    if (do_heat) {
        T_i = __ldg(&temperature_in[i]);
        kappa_i = c_materials[mat_id_i].thermal_conductivity;
    }
    // Cross-phase heat cap (bd-r5-review-n28.94). The self term bounds rho_i from
    // below, so capping a cross-phase pair coefficient at a_x_max keeps
    // viscosity_lap_coeff * a / (rho_i * cp_i) * dt <= 0.5 for that pair.
    float rho_i_lb = fmaxf(c_precalc.poly6_coeff * __ldg(&mass[i]) * h_sq * h_sq * h_sq,
                           is_gas_i ? RHO_EPSILON : 1.0f);
    float a_x_max = 0.5f * rho_i_lb * c_materials[mat_id_i].heat_capacity
                  / fmaxf(c_precalc.viscosity_lap_coeff * c_sim.dt, 1e-30f);

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
                        }
                        continue;
                    }

                    // Density (Poly6, self-included).
                    // Standard Akinci boundary: psi_b = rho0_i * V_b. MAT_RIGID mass
                    // is RIGID_PSI_RHO0 * V_b, so it is scaled by rho0_i / RIGID_PSI_RHO0
                    // (bd-r5-review-n28.77). No other multiplier: the former 2x
                    // boundary_scale inflated near-wall density to ~2*rho0, causing
                    // particles to bounce off walls (unphysical repulsion).
                    float diff = h_sq - r_sq;
                    float m_dens = (GET_MATERIAL_ID(pi_j) == MAT_RIGID) ? m_j * rigid_dens_scale : m_j;
                    sum_density += m_dens * diff * diff * diff;

                    // Exposure support weight: every non-self, non-DEAD neighbour,
                    // MAT_RIGID included, so a wall dilutes exposure (as in K_Step1).
                    if (do_heat && j != i && GET_MATERIAL_ID(pi_j) != MAT_DEAD) sum_W += diff * diff * diff;

                    // Gradient + heat + exposure (skip self)
                    if (j != i && r_sq > 1e-12f) {
                        float rlen = sqrtf(r_sq);
                        float3 gW = grad_spiky(r, rlen, h);
                        // grad_pj C_i = (1/rho0) * m_j * grad_W(x*_i - x*_j)
                        // But for denominator: |grad_pj C_i|^2 = (m_j/rho0)^2 * |gradW|^2
                        float scale = m_j / rho0;
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
                            float rho_j = (density_in != 0) ? __ldg(&density_in[j]) : 0.0f;
                            // Unset or zero density_in (spawn, unused slot): use the neighbour's rest density (bd-r5-review-n28.110)
                            if (rho_j <= 0.0f) rho_j = fmaxf(c_materials[GET_MATERIAL_ID(pi_j)].rest_density, RHO_EPSILON);
                            float T_j = __ldg(&temperature_in[j]);

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
                        }

                        // Vorticity + surface normal (FLUID only); MAT_RIGID adds to the normal only
                        if ((do_vort || do_normal) && mat_id_j != MAT_RIGID) {
                            float rho_j_v = (density_in != 0) ? __ldg(&density_in[j]) : 0.0f;
                            // Unset or zero density_in (spawn, unused slot): use the neighbour's rest density (bd-r5-review-n28.110)
                            if (rho_j_v <= 0.0f) rho_j_v = fmaxf(c_materials[GET_MATERIAL_ID(pi_j)].rest_density, RHO_EPSILON);
                            float vol_jv = m_j / fmaxf(rho_j_v, 1.0f);
                            if (do_vort) {
                                float4 vj4 = __ldg(&velocity_in[j]);
                                float dvx = vel_i_vort.x - vj4.x;
                                float dvy = vel_i_vort.y - vj4.y;
                                float dvz = vel_i_vort.z - vj4.z;
                                // curl_i = sum_j V_j gradW x (v_j - v_i); dv = v_i - v_j, so this is dv x gradW
                                omega.x += vol_jv * (dvy * gW.z - dvz * gW.y);
                                omega.y += vol_jv * (dvz * gW.x - dvx * gW.z);
                                omega.z += vol_jv * (dvx * gW.y - dvy * gW.x);
                            }
                            if (do_normal) {
                                normal.x += vol_jv * gW.x;
                                normal.y += vol_jv * gW.y;
                                normal.z += vol_jv * gW.z;
                                if (m_j > 0.0f) neighbor_count += 1.0f;  // DEAD slots have mass 0
                            }
                        } else if (do_normal && m_j > 0.0f) {
                            // MAT_RIGID neighbour: count it and add it to the colour-field
                            // normal like a STONE wall, so fluid at a rigid face is not
                            // classed as surface (bd-r5-review-n28.68). The mass slot holds
                            // RIGID_PSI_RHO0 * V_b (bd-r5-review-n28.77).
                            float vol_b = m_j / RIGID_PSI_RHO0;
                            normal.x += vol_b * gW.x;
                            normal.y += vol_b * gW.y;
                            normal.z += vol_b * gW.z;
                            neighbor_count += 1.0f;
                        }
                    }
                }
            }
        }
    }

    float rho = c_precalc.poly6_coeff * sum_density;
    rho = fmaxf(rho, is_gas_i ? RHO_EPSILON : 1.0f);
    density_out[i] = rho;

    // Heat diffusion + exposure output
    if (do_heat) {
        float cp_i = c_materials[mat_id_i].heat_capacity;
        // STATIC divides by its stored density (rest_density, written below):
        // neighbours weight the wall by m / density_out, and the same value
        // here keeps the heat pair term conservative (bd-r5-review-n28.116).
        float rho_heat = (behavior_i == STATIC) ? fmaxf(c_materials[mat_id_i].rest_density, 1.0f) : rho;
        dTdt_out[i] = c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(rho_heat * cp_i, RHO_EPSILON);
        // Exposure: support fraction, sum(rate*W) / sum(W) (same as K_Step1)
        float inv_sum_W = (sum_W > 0.0f) ? 1.0f / sum_W : 0.0f;
        exposure_heat_out[i] = sum_exposure_heat * inv_sum_W;
        exposure_corrode_out[i] = sum_exposure_corrode * inv_sum_W;
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
            // No usable gradient: emit zero normal so the n_len_sq > 0.5f
            // validity check in Finalize fails and friction is skipped.
            // (A fabricated up-normal here would apply friction against a
            // fictitious contact for free-falling/isolated particles.)
            pressure_normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    } else {
        pressure_normal_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // STATIC and GAS: no density constraint (Akinci one-way boundary / compressible
    // phase). They may reach here on the do_heat path to accumulate exposure/heat.
    // STATIC gets lambda = 0. GAS gets the EOS term p_i / rho_i^2 in its lambda
    // slot instead; K_PBF_ComputeDelta turns it into the WCSPH gas pressure
    // between GAS pairs (bd-r5-review-n28.91). Later iterations keep it.
    if (behavior_i == STATIC || behavior_i == GAS) {
        if (behavior_i == STATIC) {
            density_out[i] = fmaxf(c_materials[mat_id_i].rest_density, 1.0f);
            lambda_out[i] = 0.0f;
        } else {
            density_out[i] = rho;
            lambda_out[i] = compute_pressure(rho, GAS, mat_id_i) / (rho * rho);
        }
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
 *   dx_i = SUM_j m_j (lambda_i/rho0_i + lambda_j/rho0_j + s_corr/rho0_i) * gradW_spiky
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_PBF_ComputeDelta(
    uint            numParticles,
    const float4*   __restrict__ predicted_pos,
    const float*    __restrict__ lambda_in,     // this iteration's lambda from K_PBF_ComputeLambda (sorted)
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    float4*         __restrict__ delta_out,
    const RigidBody* __restrict__ d_rigid_bodies,  // rigid body state (NULL if no bodies)
    float*          __restrict__ d_rigid_forces,   // force accumulator (NULL if no bodies)
    float*          __restrict__ d_rigid_torques   // torque accumulator (NULL if no bodies)
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi_i = packed_info[i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);
    bool is_gas_i = (behavior_i == GAS);
    // Reaction on a rigid body: the pair's position correction dx on i is an
    // impulse m_i * dx / dt on the fluid over the substep, i.e. a force
    // m_i * dx / dt^2. The body gets the negative of it.
    float rigid_scale = 0.0f;
    if (d_rigid_bodies != 0) {
        float dt_r = fmaxf(c_sim.dt, 1e-8f);
        rigid_scale = mass[i] / (dt_r * dt_r);
    }

    // STATIC skips. Sleeping particles still participate so density
    // corrections can wake them. GAS has no density constraint; it takes the
    // gas EOS pressure from GAS neighbours below (bd-r5-review-n28.91).
    if (behavior_i == STATIC) {
        delta_out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    // GAS: the lambda slot holds p/rho^2. The WCSPH acceleration
    // a_i = -fs * sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad_i W is spread over
    // the iterations as a position delta dt^2 * a_i / num_iterations, so
    // Finalize's (x* - x) / dt adds dt * a_i to the velocity.
    float gas_scale = 0.0f;
    if (is_gas_i) {
        float dt_g = c_sim.dt;
        gas_scale = c_granular.force_scale * dt_g * dt_g
                  / (float)max(c_pbf.num_iterations, 1);
    }

    float4 pos4_i = predicted_pos[i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    // Floor at 1: DEAD has rest_density 0, and C = rho / rho0 - 1 would be inf.
    float rho0 = fmaxf(c_materials[mat_id_i].rest_density, 1.0f);
    float lambda_i = lambda_in[i];

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
                    float lambda_j = __ldg(&lambda_in[j]);
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

                    if (is_gas_i) {
                        // GAS-GAS pair (phase separation above leaves only GAS j).
                        // grad_spiky points toward j, so -coeff * gW pushes i away.
                        float3 gW_g = grad_spiky(r, rlen, h);
                        float coeff_g = -gas_scale * m_j * (lambda_i + lambda_j);
                        delta.x += coeff_g * gW_g.x;
                        delta.y += coeff_g * gW_g.y;
                        delta.z += coeff_g * gW_g.z;
                        continue;
                    }

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
                    // lambda_j belongs to C_j = rho_j / rho0_j - 1, so it
                    // scales by 1/rho0_j, not 1/rho0_i. Written as
                    // lambda_j * (rho0_i / rho0_j) inside the / rho0_i form.
                    // The mirror uses rho0_i for both terms. Same rest density
                    // keeps the ratio at exactly 1, so same-material pairs
                    // give bitwise the same delta as before.
                    float lambda_eff_j = lambda_j;
                    if (bj_delta == STATIC) {
                        lambda_eff_j = lambda_i;  // mirror fluid's own lambda
                    } else {
                        float rho0_j = fmaxf(c_materials[GET_MATERIAL_ID(pi_j)].rest_density, 1.0f);
                        if (rho0_j != rho0) lambda_eff_j = lambda_j * (rho0 / rho0_j);
                    }

                    // Mass-weighted: dp = m_j * (l_i/rho0_i + l_j/rho0_j + s/rho0_i) * gradW.
                    // m_i*dp_i + m_j*dp_j = 0 per pair (s_corr aside), for any masses.
                    float coeff = m_j * (lambda_i + lambda_eff_j + s_corr) / rho0;
                    delta.x += coeff * gW.x;
                    delta.y += coeff * gW.y;
                    delta.z += coeff * gW.z;

                    // Two-way rigid coupling: the body receives exactly minus
                    // what this pair gives the fluid (before the 0.5h clamp below).
                    if (bj_delta == STATIC && d_rigid_bodies != 0
                        && GET_MATERIAL_ID(pi_j) == MAT_RIGID) {
                        float fs = -rigid_scale * coeff;
                        pbf_rigid_reaction(d_rigid_bodies, d_rigid_forces, d_rigid_torques,
                                           pi_j, pos4_j,
                                           make_float3(fs * gW.x, fs * gW.y, fs * gW.z));
                    }
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
    // A delta can push the particle back into an SDF collider, so project
    // it out again every iteration, as K_PBF_Predict does. The loop bound is
    // 0 when the scene has no SDF objects.
    push_out_of_sdf(pos);
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
    const float4*   __restrict__ vorticity_in,     // (omega_x,y,z, |omega|) from ComputeLambda, or NULL
    const float4*   __restrict__ normal_in,         // (n_x,n_y,n_z, neighbor_count) from ComputeLambda, or NULL
    const float4*   __restrict__ pressure_normal_in, // density gradient normal for GRANULAR friction, or NULL
    const RigidBody* __restrict__ d_rigid_bodies,    // rigid body state (NULL if no bodies)
    float*          __restrict__ d_rigid_forces,     // XSPH drag reaction on the body (NULL if no bodies)
    float*          __restrict__ d_rigid_torques,    // torque of that reaction about the body centre
    const float*    __restrict__ sorted_lifetime,    // reaction-mutated lifetime (sorted), or NULL
    float*          __restrict__ health_out,         // unsorted health writeback, or NULL
    float*          __restrict__ lifetime_out,       // unsorted lifetime writeback, or NULL
    float*          __restrict__ mass_out,           // unsorted mass writeback, or NULL
    uint*           __restrict__ max_displacement_out, // [1] atomicMax of displacement^2 (float-as-uint), or NULL
    const float4*   __restrict__ displacement_anchor  // [N] sorted position at the last full sort, or NULL
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    int behavior = GET_BEHAVIOR(pi);
    uint mat_id = GET_MATERIAL_ID(pi);
    uint orig_idx = sort_indexes[i];

    // Reaction-state carry-back (bd-r4fix-uup.9): K_Reactions/K_SpawnGas mutate
    // health/lifetime/mass in SORTED space every substep; scatter them back to
    // the unsorted arrays or the next substep re-gathers stale values (wiping
    // fire lifetime decay, acid health loss, gas-spawn mass transfer). None of
    // these are modified below, so an early unconditional scatter is correct
    // and also covers the STATIC/sleeping early-return paths.
    if (health_out)                      health_out[orig_idx]   = health_in[i];
    if (lifetime_out && sorted_lifetime) lifetime_out[orig_idx] = __ldg(&sorted_lifetime[i]);
    if (mass_out)                        mass_out[orig_idx]     = mass[i];

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
        particle_dye_out[orig_idx] = sorted_particle_dye[i];
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
            particle_dye_out[orig_idx] = sorted_particle_dye[i];
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
            // Decompose correction velocity along pressure normal.
            // n points INTO the pile (direction of increasing density); the PBF
            // constraint pushes over-dense particles OUT of the pile, so the
            // load-bearing normal correction has vc_dot_n < 0. Its magnitude is
            // the Coulomb normal load that bounds the tangential correction.
            float vc_dot_n = v_corr.x*n.x + v_corr.y*n.y + v_corr.z*n.z;

            if (vc_dot_n < 0.0f) {
                float vn_mag = -vc_dot_n;  // normal correction magnitude (> 0)
                float3 vc_n = make_float3(vc_dot_n*n.x, vc_dot_n*n.y, vc_dot_n*n.z);
                float3 vc_t = make_float3(v_corr.x - vc_n.x, v_corr.y - vc_n.y, v_corr.z - vc_n.z);

                float tang_sq = vc_t.x*vc_t.x + vc_t.y*vc_t.y + vc_t.z*vc_t.z;

                // Friction slope: the material's friction_coeff, or the
                // profile's tan_phi_f when the table value is 0
                // (bd-r5-review-n28.89).  Cohesion stays global: it is a
                // per-step position allowance in metres, and the table's
                // cohesion column is not a length.
                float fc = c_materials[mat_id].friction_coeff;
                float tan_phi = (fc > 0.0f) ? fc : c_granular.tan_phi_f;
                // Static friction dead zone: below minimum normal velocity, zero tangential
                float max_tang = (vn_mag < 5e-4f) ? 0.0f
                               : tan_phi * vn_mag + c_granular.cohesion * inv_dt;

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

        float c_xsph = fminf(c_pbf.xsph_c * c_materials[mat_id].base_viscosity, 0.5f);
        if (behavior == GRANULAR) c_xsph = fminf(c_xsph * 3.0f, 0.15f);
        // Rigid drag reaction: the XSPH change dv on i is a force m_i*dv/dt on
        // the fluid; the body gets minus it (bd-r5-review-n28.97).
        float drag_scale = (d_rigid_bodies != 0) ? -mass[i] * c_xsph * inv_dt : 0.0f;
        // MAT_RIGID mass is RIGID_PSI_RHO0 * V_b. Scaled to rho0_i as in the
        // density sum, so m_j / rho_i is the Akinci volume weight V_b*rho0_i/rho_i.
        float rigid_vol_scale = c_materials[mat_id].rest_density / RIGID_PSI_RHO0;

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

                        uint pi_j = __ldg(&packed_info[j]);
                        int bj = GET_BEHAVIOR(pi_j);

                        // Vorticity eta (FLUID neighbours only -- PERF-002).
                        // Use difference form (omega_j - omega_mag_i) to match step2.
                        // STATIC (incl. MAT_RIGID), GAS and GRANULAR neighbours have
                        // omega = 0 and would bias N toward the boundary
                        // (bd-r5-review-n28.66).
                        if (do_vort_eta && bj == FLUID) {
                            float rlv = sqrtf(r_sq);
                            float h_rl = h - rlv;
                            float inv_rl = 1.0f / rlv;
                            float gs = c_precalc.spiky_grad_coeff * h_rl * h_rl * inv_rl;
                            float omega_j = __ldg(&vorticity_in[j]).w;
                            float mj_v = __ldg(&mass[j]);
                            float rj_v = __ldg(&density[j]);
                            float wt = mj_v / fmaxf(rj_v, 1.0f);
                            float omega_diff = omega_j - omega_mag_i;
                            eta_vort.x += wt * omega_diff * gs * r.x;
                            eta_vort.y += wt * omega_diff * gs * r.y;
                            eta_vort.z += wt * omega_diff * gs * r.z;
                        }

                        if (!do_xsph) continue;

                        if (bj == GAS) continue;  // no XSPH across phases

                        float m_j = __ldg(&mass[j]);

                        // Rigid boundary particle: XSPH toward the body surface
                        // velocity v_body + omega x r (K_UpdateBoundaryParticles
                        // stores it in the sorted velocity). This is the PBF
                        // fluid-body drag; the body gets the opposite momentum.
                        if (bj == STATIC && d_rigid_bodies != 0
                            && GET_MATERIAL_ID(pi_j) == MAT_RIGID) {
                            float4 vb = __ldg(&original_vel[j]);
                            float w = W_poly6(r_sq, h_sq);
                            float factor = (m_j * rigid_vol_scale / fmaxf(rho_i, 1.0f)) * w;
                            float3 dvf = make_float3((vb.x - vel_new.x) * factor,
                                                     (vb.y - vel_new.y) * factor,
                                                     (vb.z - vel_new.z) * factor);
                            xsph.x += dvf.x;
                            xsph.y += dvf.y;
                            xsph.z += dvf.z;
                            pbf_rigid_reaction(d_rigid_bodies, d_rigid_forces, d_rigid_torques,
                                               pi_j, pred4_j,
                                               make_float3(drag_scale * dvf.x,
                                                           drag_scale * dvf.y,
                                                           drag_scale * dvf.z));
                            continue;
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

    // Akinci surface tension (FLUID only, surface particles): vel += dt * (+gamma * n)
    // +gamma * norm_i coheres fluid (inward force toward interior, §10.16)
    if (behavior == FLUID && normal_in != 0 && c_granular.surface_tension_gamma > 0.0f) {
        float4 norm_i = __ldg(&normal_in[i]);
        float nc_i = norm_i.w;  // neighbor count
        if (nc_i < 25.0f) {
            float gamma = c_granular.surface_tension_gamma;
            float n_mag = sqrtf(norm_i.x*norm_i.x + norm_i.y*norm_i.y + norm_i.z*norm_i.z);
            if (n_mag > 0.01f) {
                vel_new.x += dt * (+gamma * norm_i.x);
                vel_new.y += dt * (+gamma * norm_i.y);
                vel_new.z += dt * (+gamma * norm_i.z);
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

    // World box boundary collision: restitution + wall friction. Uses the
    // PBF variant because pred is already clamped to the wall margin.
    float friction = (behavior == FLUID) ? 0.0f : c_sim.wall_friction;
    pbf_box_boundary(final_pos, vel_new, c_sim.world_min, c_sim.world_max,
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

    // Rigid body collision (bd-r5-review-n28.123): push out of each body's
    // box or sphere SDF at its end-of-substep pose, so a fast body cannot
    // overtake fluid and trap it inside. Same helper as K_Integrate.
    if (d_rigid_bodies != 0 && c_num_rigid_bodies > 0) {
        rigid_body_boundary(final_pos, vel_new, d_rigid_bodies, c_num_rigid_bodies, dt);
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
    // Same 0.01 m/s velocity threshold as WCSPH (integrate.cu
    // GRANULAR_V_THRESHOLD); catches artificial spreading velocity generated
    // by PBF position corrections.
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

    // Sleep: velocity-based (replaces shear_rate for PBF).
    // GRANULAR only, matching WCSPH (integrate.cu): FLUID and GAS must never
    // accumulate the sleep counter — sleeping particles get no forces, so a
    // slow fluid pool (or gas hovering at terminal velocity) would freeze
    // permanently with no pressure differential left to wake it.
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
        // FLUID/GAS: never accumulate sleep counter (reset if set externally)
        sc = 0;
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

    // Track drift since the last full sort for grid reuse (bd-r5-review-n28.54).
    // The anchor is the sorted position written by K_ScatterReorder at the last
    // full sort; the permutation is frozen between full sorts, so slot i is the
    // same particle. atomicMax over all substeps since that sort then holds the
    // largest drift any particle has made from the cell it was binned in.
    // NULL anchor: fall back to this substep's start position.
    // Positive IEEE754 floats order like their uint bits, so atomicMax on
    // __float_as_uint gives the max float value.
    if (max_displacement_out) {
        float3 ref = orig;
        if (displacement_anchor) {
            const float4 a = __ldg(&displacement_anchor[i]);
            ref = make_float3(a.x, a.y, a.z);
        }
        float dx_d = final_pos.x - ref.x;
        float dy_d = final_pos.y - ref.y;
        float dz_d = final_pos.z - ref.z;
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
    particle_dye_out[orig_idx] = dye;
}
