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
 * K_ComputePressure -- pre-compute per-particle pressure from density.
 *
 * Runs once after K_Step1 so that K_Step2 can read pressure from an array
 * instead of calling compute_pressure() ~50 times per particle (once per
 * neighbor). WCSPH only.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256)
void K_ComputePressure(
    uint            numParticles,
    const float*    __restrict__ density,       // sorted density from Step1
    const uint*     __restrict__ packed_info,   // sorted packed_info
    float*          __restrict__ pressure_out   // output: per-particle pressure
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float rho_i = __ldg(&density[i]);
    uint pi_i = __ldg(&packed_info[i]);
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);

    pressure_out[i] = compute_pressure(rho_i, behavior_i, mat_id_i);
}

/* ======================================================================
 * boundary_visc_factor -- viscous drag coefficient against a STATIC
 * boundary particle (regular STATIC or MAT_RIGID).
 *
 * GRANULAR i uses the same dynamic-viscosity form as its pair code,
 * mu0 * m_b * lap / (rho_i * rho_b), with rho_b = rest density of i's own
 * material (the Akinci mirror: the boundary stands in for fluid of i's
 * kind). Dropping the 1/rho_b made sand-on-wall drag rho0 (~4000x) too
 * strong (bd-r5-review-n28.24).
 *
 * FLUID/GAS i use the same dynamic form with mu = base_viscosity, as in
 * their pair code (bd-r5-review-n28.79). The result is divided by
 * force_scale so the final fs_i multiply leaves the plain accel
 * mu * m_b * lap / (rho_i * rho_b), as in DFSPH.
 * ====================================================================== */
__device__ __forceinline__ float boundary_visc_factor(
    bool is_granular_i, uint mat_id_i, float m_b, float lap_v_b, float rho_i)
{
    float rho_i_safe = fmaxf(rho_i, RHO_EPSILON);
    float rho_b = fmaxf(c_materials[mat_id_i].rest_density, RHO_EPSILON);
    if (is_granular_i) {
        return c_precalc.viscosity_precalc * m_b * lap_v_b / (rho_i_safe * rho_b);
    }
    float mu_b = c_materials[mat_id_i].base_viscosity * c_precalc.viscosity_lap_coeff;
    return mu_b * m_b * lap_v_b / (rho_i_safe * rho_b * fmaxf(c_granular.force_scale, 1e-6f));
}

/* ======================================================================
 * FLUID-GRANULAR pair scaling (bd-r5-review-n28.41)
 *
 * The final output multiplies particle i's sums by fs_i (1.0 GRANULAR,
 * c_granular.force_scale FLUID/GAS). A pair between two behaviours must
 * use one scale on both sides, or F_ij != -F_ji and every water-sand
 * interface creates momentum. Cross pairs therefore use
 *   pressure:  fs_cross = min(fs_FLUID, fs_GRANULAR) = force_scale
 *   viscosity: dynamic form mu_pair * m_j * lap / (rho_i * rho_j),
 *              mu_pair = harmonic mean of the FLUID dynamic viscosity
 *              (base_viscosity * lap_coeff, the same number the FLUID
 *              pair uses, bd-r5-review-n28.79) and the GRANULAR value
 *              (viscosity_precalc = mu * lap_coeff).
 * Both are symmetric in i and j. The terms are divided by fs_i here so
 * the final fs_i multiply leaves exactly the pair value.
 * ====================================================================== */
__device__ __forceinline__ float cross_pair_visc(
    uint mat_fluid, float m_j, float lap_v, float rho_i, float rho_j)
{
    float mu_f = c_materials[mat_fluid].base_viscosity * c_precalc.viscosity_lap_coeff;
    float mu_g = c_precalc.viscosity_precalc;
    float mu_pair = 2.0f * mu_f * mu_g / (mu_f + mu_g + 1e-8f);
    return mu_pair * m_j * lap_v
         / (fmaxf(rho_i, RHO_EPSILON) * fmaxf(rho_j, RHO_EPSILON));
}

/* ======================================================================
 * K_Step2 kernel
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 3)
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
    const float*    __restrict__ pressure_in,    // pre-computed per-particle pressure (PERF-007)
    float4*         __restrict__ sph_force_out,  // output: accumulated SPH force
    float4*         __restrict__ veleval_out,    // output: XSPH-corrected veleval
    const void*     __restrict__ velocity_h,     // FP16 velocity for neighbor reads (OPT-4.3), may be NULL
    const RigidBody* __restrict__ d_rigid_bodies, // rigid body state array (NULL if no bodies)
    float*          __restrict__ d_rigid_forces,  // force accumulator [MAX_RIGID_BODIES * 4] (NULL if no bodies)
    float*          __restrict__ d_rigid_torques  // torque accumulator [MAX_RIGID_BODIES * 4] (NULL if no bodies)
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
    // FP16 pairing rule (bd-r5-review-n28.69): neighbour velocities come from
    // the FP16 mirror, so pair differences (v_j - v_i) must take v_i from the
    // same mirror. Otherwise co-moving particles see fp16(v) - v != 0, a
    // lattice-snapping viscous force. The FP32 vel_i stays for veleval output
    // and for boundary terms, whose other side is FP32.
    float3 vel_i_pair = vel_i;
    if (velocity_h) {
        float4 vh_i = load_half4((const uint2*)velocity_h + index_i);
        vel_i_pair = make_float3(vh_i.x, vh_i.y, vh_i.z);
    }

    float rho_i = pos4_i.w;  // density packed into position.w by K_PackDensity (OPT-4.1)
    float p_i = __ldg(&pressure_in[index_i]);  // pre-computed (PERF-007)

    // Accumulators
    float3 f_pressure  = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_viscosity = make_float3(0.0f, 0.0f, 0.0f);
    float3 xsph_sum    = make_float3(0.0f, 0.0f, 0.0f);

    // Implicit viscous self-term (bd-r5-review-n28.52): c_tot sums every
    // viscous coefficient that multiplies -vel_i. See the PostCalc note.
    float c_tot = 0.0f;
    // Viscous terms left out of the implicit scale (rigid bodies other than
    // the first one this particle touches) and the deferred viscous reaction
    // on that first body, emitted once the scale is known.
    float3 f_visc_explicit = make_float3(0.0f, 0.0f, 0.0f);
    int    rb_pend = -1;
    float3 rb_fv   = make_float3(0.0f, 0.0f, 0.0f);
    float3 rb_tv   = make_float3(0.0f, 0.0f, 0.0f);

    float h    = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;

    bool is_granular_i = (behavior_i == GRANULAR);
    bool is_fluid_i    = (behavior_i == FLUID);
    bool is_gas_i      = (behavior_i == GAS);

    // Per-particle output scale and the FLUID-GRANULAR pair scale
    // (bd-r5-review-n28.41, see cross_pair_visc above).
    float fs_i = is_granular_i ? 1.0f : c_granular.force_scale;
    float inv_fs_i = 1.0f / fmaxf(fs_i, 1e-6f);
    float cross_p_rescale = fminf(c_granular.force_scale, 1.0f) * inv_fs_i;
    // Own mass, used for the rigid-body reaction force (loop-invariant).
    float m_i = __ldg(&mass[index_i]);

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

        // mu(I) rheology computation for particle i.
        // Pressure floor: rho0 * |g| * spacing gives a lithostatic-scale minimum
        // so that densely packed sand at rho < rho0 still gets meaningful friction.
        float rho0_i = c_materials[mat_id_i].rest_density;
        float p_floor = rho0_i * fabsf(c_sim.gravity.y) * c_granular.particle_spacing;
        float p_eff_i = fmaxf(p_i, p_floor);
        eta_i = compute_muI_eta(gamma_dot_i, p_eff_i, rho_i);
    }

    // Vorticity eta accumulator (FLUID only, folded into main loop -- PERF-002)
    bool do_vort_eta = false;
    float omega_mag_i = 0.0f;
    float4 vort_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 eta_vort = make_float3(0.0f, 0.0f, 0.0f);
    if (is_fluid_i && c_granular.vorticity_epsilon > 0.0f) {
        vort_i = __ldg(&vorticity_in[index_i]);
        omega_mag_i = vort_i.w;
        do_vort_eta = (omega_mag_i > 1e-6f);
    }

    // ---------------------------------------------------------------
    // Main neighbor loop: pressure, viscosity, XSPH, vorticity eta
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

                    float rlen = sqrtf(rlen_sq);

                    int behavior_j = GET_BEHAVIOR(pi_j);

                    // Vorticity eta accumulation (FLUID i, FLUID neighbours only -- PERF-002).
                    // STATIC (incl. MAT_RIGID), GAS and GRANULAR neighbours carry
                    // omega = 0 and a rho_j that is not a fluid density, so they
                    // would point N at the body/phase boundary instead of along
                    // grad|omega| (bd-r5-review-n28.66).
                    if (do_vort_eta && behavior_j == FLUID) {
                        float h_rl = h - rlen;
                        float inv_rl = 1.0f / rlen;
                        float gs = c_precalc.spiky_grad_coeff * h_rl * h_rl * inv_rl;
                        float omega_j = __ldg(&vorticity_in[index_j]).w;
                        float rho_j_e = pos4_j.w;
                        float wt = m_j / fmaxf(rho_j_e, RHO_EPSILON);
                        float omega_diff = omega_j - omega_mag_i;
                        eta_vort.x += wt * omega_diff * gs * r.x;
                        eta_vort.y += wt * omega_diff * gs * r.y;
                        eta_vort.z += wt * omega_diff * gs * r.z;
                    }

                    // GAS/non-GAS phase separation: no pressure/viscosity across
                    // phases, except GAS against a MAT_RIGID body particle, which
                    // takes the Akinci branch below so gas is pushed around and
                    // dragged by moving bodies (bd-r5-review-n28.101). GAS keeps
                    // skipping regular STATIC; static_particle_boundary handles it.
                    if (is_gas_i != (behavior_j == GAS)
                        && !(is_gas_i && GET_MATERIAL_ID(pi_j) == MAT_RIGID)) continue;

                    // density_j packed into position.w by K_PackDensity (OPT-4.1)
                    float3 vel_j = make_float3(vel4_j.x, vel4_j.y, vel4_j.z);
                    float rho_j = pos4_j.w;

                    // ---- STATIC boundary: Akinci pressure mirroring + friction ----
                    if (behavior_j == STATIC) {
                        uint mat_id_j = GET_MATERIAL_ID(pi_j);
                        float3 grad_s = grad_spiky_variable(r, rlen, h);

                        if (mat_id_j == MAT_RIGID && d_rigid_bodies != 0) {
                            // --- MAT_RIGID: Akinci pressure mirroring with two-way coupling ---
                            // psi_b = rho0_i * V_b with the fluid's own rest density.
                            // The mass slot holds RIGID_PSI_RHO0 * V_b (bd-r5-review-n28.77).
                            float rho0 = c_materials[mat_id_i].rest_density;
                            float psi_b = m_j * __fdiv_rn(rho0, RIGID_PSI_RHO0);
                            // Same pressure mirror as a regular STATIC wall,
                            // 2 p_i / rho_i^2 (bd-r5-review-n28.106).
                            float press_akinci = 2.0f * (p_i / (rho_i * rho_i));
                            float3 F_on_fluid = make_float3(
                                psi_b * press_akinci * grad_s.x,
                                psi_b * press_akinci * grad_s.y,
                                psi_b * press_akinci * grad_s.z
                            );
                            f_pressure.x += F_on_fluid.x;
                            f_pressure.y += F_on_fluid.y;
                            f_pressure.z += F_on_fluid.z;

                            // Viscous coupling: boundary velocity from rigid body
                            int body_id = GET_BODY_ID(pi_j);
                            float4 rb_pos = __ldg(&d_rigid_bodies[body_id].position);
                            float4 rb_linvel = __ldg(&d_rigid_bodies[body_id].lin_vel);
                            float4 rb_angvel = __ldg(&d_rigid_bodies[body_id].ang_vel);
                            float3 r_b = make_float3(
                                pos4_j.x - rb_pos.x,
                                pos4_j.y - rb_pos.y,
                                pos4_j.z - rb_pos.z
                            );
                            // v_boundary = v_body + cross(omega, r_b)
                            float3 vel_boundary = make_float3(
                                rb_linvel.x + (rb_angvel.y * r_b.z - rb_angvel.z * r_b.y),
                                rb_linvel.y + (rb_angvel.z * r_b.x - rb_angvel.x * r_b.z),
                                rb_linvel.z + (rb_angvel.x * r_b.y - rb_angvel.y * r_b.x)
                            );
                            float lap_v_b = lap_visc_variable(rlen, h);
                            float visc_factor_b = boundary_visc_factor(
                                is_granular_i, mat_id_i, psi_b, lap_v_b, rho_i);
                            float3 F_visc = make_float3(
                                (vel_boundary.x - vel_i.x) * visc_factor_b,
                                (vel_boundary.y - vel_i.y) * visc_factor_b,
                                (vel_boundary.z - vel_i.z) * visc_factor_b
                            );
                            // Viscous reaction: the first body goes into the
                            // implicit sum and its reaction is emitted after the
                            // loop, scaled by the same factor as the fluid side.
                            // A second body stays explicit with an immediate
                            // reaction, so action = reaction in both cases.
                            float3 F_visc_now = make_float3(0.0f, 0.0f, 0.0f);
                            if (rb_pend < 0 || rb_pend == body_id) {
                                rb_pend = body_id;
                                f_viscosity.x += F_visc.x;
                                f_viscosity.y += F_visc.y;
                                f_viscosity.z += F_visc.z;
                                c_tot += visc_factor_b;
                                rb_fv.x += F_visc.x;
                                rb_fv.y += F_visc.y;
                                rb_fv.z += F_visc.z;
                                rb_tv.x += r_b.y * F_visc.z - r_b.z * F_visc.y;
                                rb_tv.y += r_b.z * F_visc.x - r_b.x * F_visc.z;
                                rb_tv.z += r_b.x * F_visc.y - r_b.y * F_visc.x;
                            } else {
                                f_visc_explicit.x += F_visc.x;
                                f_visc_explicit.y += F_visc.y;
                                f_visc_explicit.z += F_visc.z;
                                F_visc_now = F_visc;
                            }

                            // Two-way coupling: accumulate reaction FORCE on rigid body.
                            // step2 outputs acceleration (force/mass) for fluid particles,
                            // but K_IntegrateRigidBodies expects actual force (F = m * a).
                            // Must multiply by fluid particle mass to convert accel -> force.
                            float pp = c_precalc.pressure_precalc;
                            float fs = is_granular_i ? 1.0f : c_granular.force_scale;
                            float3 a_on_fluid = make_float3(
                                (pp * F_on_fluid.x + F_visc_now.x) * fs,
                                (pp * F_on_fluid.y + F_visc_now.y) * fs,
                                (pp * F_on_fluid.z + F_visc_now.z) * fs
                            );
                            // Newton's 3rd law: reaction force on body = -m_i * a_on_fluid
                            float3 F_on_body = make_float3(-m_i * a_on_fluid.x, -m_i * a_on_fluid.y, -m_i * a_on_fluid.z);
                            // Torque = cross(r_b, F_on_body) where r_b = boundary_pos - COM
                            float3 tau = make_float3(
                                r_b.y * F_on_body.z - r_b.z * F_on_body.y,
                                r_b.z * F_on_body.x - r_b.x * F_on_body.z,
                                r_b.x * F_on_body.y - r_b.y * F_on_body.x
                            );
                            warp_reduce_accumulate(d_rigid_forces, F_on_body, body_id);
                            warp_reduce_accumulate(d_rigid_torques, tau, body_id);
                        } else {
                            // --- Regular STATIC boundary: Akinci pressure mirroring + friction ---
                            float press_sym_b = 2.0f * (p_i / (rho_i * rho_i));
                            f_pressure.x += m_j * press_sym_b * grad_s.x;
                            f_pressure.y += m_j * press_sym_b * grad_s.y;
                            f_pressure.z += m_j * press_sym_b * grad_s.z;
                            // Boundary friction: vel_j=0, drag toward zero
                            float lap_v_b = lap_visc_variable(rlen, h);
                            float visc_factor_b = boundary_visc_factor(
                                is_granular_i, mat_id_i, m_j, lap_v_b, rho_i);
                            f_viscosity.x += (-vel_i.x) * visc_factor_b;
                            f_viscosity.y += (-vel_i.y) * visc_factor_b;
                            f_viscosity.z += (-vel_i.z) * visc_factor_b;
                            c_tot += visc_factor_b;
                        }
                        continue;  // skip XSPH from STATIC
                    }

                    // Neighbor pressure from pre-computed array (PERF-007)
                    uint mat_id_j = GET_MATERIAL_ID(pi_j);
                    float p_j = __ldg(&pressure_in[index_j]);

                    // FLUID-GRANULAR pair (GAS and STATIC were handled above)
                    bool cross_ij = (is_granular_i != (behavior_j == GRANULAR));

                    // ---- Pressure force ----
                    // f_press += pressure_precalc * m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky_variable
                    // Cross pairs are rescaled so fs_i * rescale = fs_cross on both sides.
                    float press_sym = (p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j));
                    if (cross_ij) press_sym *= cross_p_rescale;
                    float3 grad_s = grad_spiky_variable(r, rlen, h);
                    f_pressure.x += m_j * press_sym * grad_s.x;
                    f_pressure.y += m_j * press_sym * grad_s.y;
                    f_pressure.z += m_j * press_sym * grad_s.z;

                    // ---- Viscosity force ----
                    float lap_v = lap_visc_variable(rlen, h);

                    if (is_granular_i && behavior_j == GRANULAR) {
                        // mu(I) viscosity for GRANULAR-GRANULAR pairs
                        // Use tensor-based shear_rate from Step1 (less noisy than |dv|/r)
                        float gamma_dot_j = __ldg(&shear_rate_in[index_j]);

                        float rho0_j = c_materials[mat_id_j].rest_density;
                        float p_floor_j = rho0_j * fabsf(c_sim.gravity.y) * c_granular.particle_spacing;
                        float p_eff_j = fmaxf(p_j, p_floor_j);
                        float eta_j = compute_muI_eta(gamma_dot_j, p_eff_j, rho_j);

                        // Harmonic mean viscosity
                        float eta_ij = 2.0f * eta_i * eta_j / (eta_i + eta_j + 1e-8f);

                        // Full viscosity force with coefficient baked in
                        // Includes 1/rho_i for dimensional correctness (G2 fix)
                        float visc_lap_const = c_precalc.viscosity_lap_coeff;  // = 45/(pi*h^6)
                        float visc_factor = eta_ij * visc_lap_const * m_j * lap_v / (rho_j * rho_i);
                        f_viscosity.x += (vel_j.x - vel_i_pair.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i_pair.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i_pair.z) * visc_factor;
                        c_tot += visc_factor;
                    } else if (cross_ij) {
                        // FLUID-GRANULAR pair: one symmetric dynamic-form term on
                        // both sides (bd-r5-review-n28.41), undone by the fs_i multiply.
                        uint mat_fluid = is_granular_i ? mat_id_j : mat_id_i;
                        float visc_factor = cross_pair_visc(mat_fluid, m_j, lap_v, rho_i, rho_j) * inv_fs_i;
                        f_viscosity.x += (vel_j.x - vel_i_pair.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i_pair.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i_pair.z) * visc_factor;
                        c_tot += visc_factor;
                    } else {
                        // FLUID/GAS: per-material dynamic viscosity (harmonic mean),
                        // accel = mu_ij * lap * m_j * lap_v / (rho_i * rho_j) * (v_j - v_i).
                        // Same number and pair formula as DFSPH, symmetric in i and j
                        // so the pair conserves momentum when rho_i != rho_j
                        // (bd-r5-review-n28.79). inv_fs_i undoes the final fs_i multiply,
                        // so force_scale no longer sets the FLUID viscosity.
                        float mu_i = c_materials[mat_id_i].base_viscosity;
                        float mu_j = c_materials[mat_id_j].base_viscosity;
                        float mu_ij = 2.0f * mu_i * mu_j / (mu_i + mu_j + 1e-8f);
                        float visc_factor = mu_ij * c_precalc.viscosity_lap_coeff * m_j * lap_v
                                          / (fmaxf(rho_i, RHO_EPSILON) * fmaxf(rho_j, RHO_EPSILON))
                                          * inv_fs_i;
                        f_viscosity.x += (vel_j.x - vel_i_pair.x) * visc_factor;
                        f_viscosity.y += (vel_j.y - vel_i_pair.y) * visc_factor;
                        f_viscosity.z += (vel_j.z - vel_i_pair.z) * visc_factor;
                        c_tot += visc_factor;
                    }

                    // ---- XSPH correction (FLUID only) ----
                    // GRANULAR position is advected with vel_new (integrate.cu:265-268),
                    // not veleval_xsph, so XSPH for GRANULAR would be computed then
                    // discarded — wasted inner-loop work.  Skip it here (bd-mzc.34).
                    // Only FLUID neighbours count: a GRANULAR neighbour (often
                    // sleeping at v = 0) would drag water's advection velocity
                    // toward 0 while its velocity stays, so water crawled over
                    // sand beds (bd-r5-review-n28.104).
                    if (is_fluid_i && behavior_j == FLUID) {
                        float w = W_poly6(rlen_sq, h_sq);
                        float rho_avg = 0.5f * (rho_i + rho_j);
                        float xsph_factor = (m_j / rho_avg) * w;
                        xsph_sum.x += (vel_j.x - vel_i_pair.x) * xsph_factor;
                        xsph_sum.y += (vel_j.y - vel_i_pair.y) * xsph_factor;
                        xsph_sum.z += (vel_j.z - vel_i_pair.z) * xsph_factor;
                    }
                }  // end neighbor particle loop
            }
        }
    }  // end 27-cell loop

    // --- Vorticity confinement (FLUID only, eta accumulated in main loop -- PERF-002) ---
    float3 f_vorticity_conf = make_float3(0.0f, 0.0f, 0.0f);
    if (do_vort_eta) {
        float eta_mag = sqrtf(eta_vort.x*eta_vort.x + eta_vort.y*eta_vort.y + eta_vort.z*eta_vort.z);
        if (eta_mag > 1e-6f) {
            float inv_eta = 1.0f / eta_mag;
            float3 N = make_float3(eta_vort.x*inv_eta, eta_vort.y*inv_eta, eta_vort.z*inv_eta);
            float eps_v = c_granular.vorticity_epsilon;
            f_vorticity_conf.x = eps_v * (N.y * vort_i.z - N.z * vort_i.y);
            f_vorticity_conf.y = eps_v * (N.z * vort_i.x - N.x * vort_i.z);
            f_vorticity_conf.z = eps_v * (N.x * vort_i.y - N.y * vort_i.x);
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
            // Surface tension: +gamma * n_i (pulls surface inward).
            // Normal n_i points INTO the bulk (toward neighbors) because
            // spiky_grad_coeff is negative and r = pos_i - pos_j points away
            // from neighbors. So +gamma * n pushes toward bulk = cohesion.
            float n_mag = sqrtf(norm_i.x*norm_i.x + norm_i.y*norm_i.y + norm_i.z*norm_i.z);
            if (n_mag > 0.01f) {
                f_surface_tension.x = gamma * norm_i.x;
                f_surface_tension.y = gamma * norm_i.y;
                f_surface_tension.z = gamma * norm_i.z;
            }
        }
    }

    // --- Implicit viscous self-term (bd-r5-review-n28.52) ---
    // The viscous accel is fs_i * sum_j c_j (v_j - v_i). Explicit Euler on it
    // needs dt * fs_i * c_tot < 2, which LAVA (18.75 at dt = 1 ms), MUD and
    // resting sand at mu_max break. Treating v_i implicitly,
    //   v' = (v + dt * fs_i * sum c_j v_j) / (1 + dt * fs_i * c_tot),
    // is the same as scaling the viscous sum by s = 1 / (1 + x), x = dt*fs_i*c_tot.
    // Stable for any x; the checkerboard mode decays by (1 - x) / (1 + x).
    float visc_s = 1.0f / (1.0f + c_sim.dt * fs_i * c_tot);
    f_viscosity.x = f_viscosity.x * visc_s + f_visc_explicit.x;
    f_viscosity.y = f_viscosity.y * visc_s + f_visc_explicit.y;
    f_viscosity.z = f_viscosity.z * visc_s + f_visc_explicit.z;
    if (rb_pend >= 0) {
        // Deferred viscous reaction on the first rigid body, same scale.
        float k_rb = -m_i * fs_i * visc_s;
        warp_reduce_accumulate(d_rigid_forces,
            make_float3(k_rb * rb_fv.x, k_rb * rb_fv.y, k_rb * rb_fv.z), rb_pend);
        warp_reduce_accumulate(d_rigid_torques,
            make_float3(k_rb * rb_tv.x, k_rb * rb_tv.y, k_rb * rb_tv.z), rb_pend);
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
    // FLUID-GRANULAR pair terms were pre-divided by fs_i in the loop.
    sph_force_out[index_i] = make_float4(total_force.x * fs_i, total_force.y * fs_i, total_force.z * fs_i, 0.0f);

    // XSPH-corrected veleval (FLUID only; GRANULAR/GAS/STATIC keep original velocity).
    // GRANULAR position is advected with vel_new in integrate.cu, not veleval_xsph,
    // so XSPH correction for GRANULAR would be discarded.  bd-mzc.34.
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
