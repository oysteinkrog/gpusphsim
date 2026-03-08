/*
 * neighbor_list.cu -- Build compact neighbor lists from spatial hash grid.
 *
 * K_BuildNeighborList: For each particle, iterates 27 neighbor cells and
 * stores actual neighbor indices in a compact per-particle list.
 * Subsequent kernels (step1, step2) iterate this list instead of
 * searching the grid, avoiding hash collision overhead and empty cell checks.
 *
 * Memory layout (CSR-like, fixed max per particle):
 *   neighbor_indices[N * MAX_NB]: packed neighbor indices
 *   neighbor_count[N]:            number of valid entries per particle
 *
 * Constant memory: c_grid, c_sim (from common.cuh)
 */

#include "common.cuh"

/* Max neighbors per particle.  At h=0.04, spacing=0.02: avg ~50, max ~80.
 * 64 covers >99% of cases with some headroom for hash collision overlap. */
#define DEFAULT_MAX_NB 64

extern "C" __global__ __launch_bounds__(256, 4)
void K_BuildNeighborList(
    uint            numParticles,
    const float4*   __restrict__ position,
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end,
    uint*           __restrict__ neighbor_indices,   // [N * max_nb]
    uint*           __restrict__ neighbor_count_out, // [N]
    uint            max_nb                           // max neighbors per particle
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float4 pos4_i = __ldg(&position[i]);
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float h_sq = c_sim.smoothing_length_sq;

    int3 cell_i = calcGridCell(pos_i);
    uint count = 0;
    uint base = i * max_nb;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint hash = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    float4 pos4_j = __ldg(&position[j]);
                    float rx = pos_i.x - pos4_j.x;
                    float ry = pos_i.y - pos4_j.y;
                    float rz = pos_i.z - pos4_j.z;
                    float r_sq = rx * rx + ry * ry + rz * rz;

                    if (r_sq <= h_sq && count < max_nb) {
                        neighbor_indices[base + count] = j;
                        count++;
                    }
                }
            }
        }
    }
    neighbor_count_out[i] = count;
}

/* ======================================================================
 * K_Step1_NL -- Neighbor-list variant of K_Step1.
 *
 * Identical physics to K_Step1 but iterates a pre-built neighbor list
 * instead of scanning 27 grid cells.  Eliminates:
 *   - spatialHash() calls (27 per particle)
 *   - cell_start/cell_end indirection
 *   - Distance checks on hash-collision false positives
 * ====================================================================== */

#include "sph_shared.cuh"

extern "C" __global__ __launch_bounds__(256, 4)
void K_Step1_NL(
    uint            numParticles,
    float4*         __restrict__ position,
    const float4*   __restrict__ velocity,
    const float*    __restrict__ mass,
    const float*    __restrict__ density_in,
    const uint*     __restrict__ packed_info,
    const float*    __restrict__ temperature_in,
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
    // Neighbor list inputs
    const uint*     __restrict__ neighbor_indices,  // [N * max_nb]
    const uint*     __restrict__ neighbor_count,    // [N]
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

    if (is_rigid_i) {
        density_out[index_i] = 2500.0f;
        pressure_out[index_i] = 0.0f;
        shear_rate_out[index_i] = 0.0f;
        dTdt_out[index_i] = 0.0f;
        exposure_heat_out[index_i] = 0.0f;
        exposure_corrode_out[index_i] = 0.0f;
        vorticity_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        normal_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        dye_rate_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float4 vel4_i = velocity[index_i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);

    bool dye_on = (c_sim.dye_enabled != 0);
    float4 dye_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (dye_on) dye_i = __ldg(&particle_dye_in[index_i]);

    float T_i = __ldg(&temperature_in[index_i]);
    float kappa_i = c_materials[mat_id_i].thermal_conductivity;

    // Accumulators
    float sum_density = 0.0f;
    float sum_dTdt = 0.0f;
    float sum_exposure_heat = 0.0f;
    float sum_exposure_corrode = 0.0f;
    float Dxx = 0.0f, Dyy = 0.0f, Dzz = 0.0f;
    float Dxy = 0.0f, Dxz = 0.0f, Dyz = 0.0f;
    float3 omega = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal = make_float3(0.0f, 0.0f, 0.0f);
    float neighbor_count_f = 0.0f;
    float3 dye_rate = make_float3(0.0f, 0.0f, 0.0f);

    // Self-interaction for density
    float m_self = __ldg(&mass[index_i]);
    float self_diff = h_sq;  // r_sq = 0 for self
    sum_density += m_self * self_diff * self_diff * self_diff;

    // Iterate compact neighbor list
    uint nb_count = neighbor_count[index_i];
    uint nb_base = index_i * max_nb;

    for (uint k = 0; k < nb_count; k++) {
        uint index_j = neighbor_indices[nb_base + k];

        // Speculative ILP: issue all loads before computation
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

        // Neighbors are pre-validated: r_sq <= h_sq guaranteed
        // But recheck for safety (particle may have moved since list was built)
        if (r_sq > h_sq) continue;

        float diff = h_sq - r_sq;
        int behavior_j = GET_BEHAVIOR(pi_j);
        bool is_gas_j = (behavior_j == GAS);

        if (is_gas_i != is_gas_j) {
            uint mat_id_j = GET_MATERIAL_ID(pi_j);
            float w_poly6_var = diff * diff * diff;
            sum_exposure_corrode += c_interactions[mat_id_i][mat_id_j].reaction_rate * w_poly6_var;
            sum_exposure_heat += c_interactions[mat_id_i][mat_id_j].heat_exchange * fmaxf(T_j - T_i, 0.0f) * w_poly6_var;
            continue;
        }

        // Density (no self-skip needed — self not in neighbor list)
        sum_density += m_j * diff * diff * diff;

        // Non-self computations
        uint mat_id_j = GET_MATERIAL_ID(pi_j);
        float rlen = sqrtf(r_sq);

        if (mat_id_j != MAT_RIGID) {
            float rho_j = (density_in != 0) ? __ldg(&density_in[index_j]) : 1000.0f;
            float lap_var = h - rlen;
            float heat_boost = fmaxf(1.0f, c_interactions[mat_id_i][mat_id_j].heat_exchange);
            sum_dTdt += m_j / fmaxf(rho_j, RHO_EPSILON) * (T_j - T_i) * lap_var * heat_boost;

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
                    omega.x += vol_j * (dvy_v * gWz - dvz_v * gWy);
                    omega.y += vol_j * (dvz_v * gWx - dvx_v * gWz);
                    omega.z += vol_j * (dvx_v * gWy - dvy_v * gWx);

                    normal.x += vol_j * gWx;
                    normal.y += vol_j * gWy;
                    normal.z += vol_j * gWz;
                }

                neighbor_count_f += 1.0f;

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
        }
    }

    // PostCalc (identical to K_Step1)
    float density = c_precalc.poly6_coeff * sum_density;
    float rho_floor = is_gas_i ? RHO_EPSILON : 1.0f;
    density_out[index_i] = fmaxf(density, rho_floor);
    pressure_out[index_i] = compute_pressure(density_out[index_i], behavior_i, mat_id_i);

    float cp_i = c_materials[mat_id_i].heat_capacity;
    float rho_i_heat = density_out[index_i];
    dTdt_out[index_i] = kappa_i * c_precalc.viscosity_lap_coeff * sum_dTdt / fmaxf(rho_i_heat * cp_i, RHO_EPSILON);

    exposure_heat_out[index_i] = c_precalc.poly6_coeff * sum_exposure_heat;
    exposure_corrode_out[index_i] = c_precalc.poly6_coeff * sum_exposure_corrode;

    float omega_mag = sqrtf(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z);
    vorticity_out[index_i] = make_float4(omega.x, omega.y, omega.z, omega_mag);
    normal_out[index_i] = make_float4(normal.x, normal.y, normal.z, neighbor_count_f);

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


/* ======================================================================
 * K_Step2_NL -- Neighbor-list variant of K_Step2.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 3)
void K_Step2_NL(
    uint            numParticles,
    const float4*   __restrict__ position,
    const float4*   __restrict__ velocity,
    const float*    __restrict__ mass,
    const uint*     __restrict__ packed_info,
    const float*    __restrict__ shear_rate_in,
    const float4*   __restrict__ vorticity_in,
    const float4*   __restrict__ normal_in,
    const float*    __restrict__ pressure_in,
    float4*         __restrict__ sph_force_out,
    float4*         __restrict__ veleval_out,
    const void*     __restrict__ velocity_h,
    const RigidBody* __restrict__ d_rigid_bodies,
    float*          __restrict__ d_rigid_forces,
    float*          __restrict__ d_rigid_torques,
    // Neighbor list inputs
    const uint*     __restrict__ neighbor_indices,
    const uint*     __restrict__ neighbor_count_in,
    uint            max_nb
) {
    uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_i >= numParticles) return;

    uint pi_i = packed_info[index_i];
    int behavior_i = GET_BEHAVIOR(pi_i);
    uint mat_id_i = GET_MATERIAL_ID(pi_i);

    if (behavior_i == STATIC) {
        sph_force_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        veleval_out[index_i] = velocity[index_i];
        return;
    }
    if (IS_SLEEPING(pi_i)) {
        sph_force_out[index_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        veleval_out[index_i] = velocity[index_i];
        return;
    }

    float4 pos4_i = position[index_i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float4 vel4_i = velocity[index_i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);
    float rho_i = pos4_i.w;
    float p_i = __ldg(&pressure_in[index_i]);

    float3 f_pressure  = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_viscosity = make_float3(0.0f, 0.0f, 0.0f);
    float3 xsph_sum    = make_float3(0.0f, 0.0f, 0.0f);

    float h    = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;

    bool is_granular_i = (behavior_i == GRANULAR);
    bool is_fluid_i    = (behavior_i == FLUID);
    bool is_gas_i      = (behavior_i == GAS);

    float gamma_dot_i = 0.0f;
    float eta_i = 0.0f;
    if (is_granular_i) {
        gamma_dot_i = __ldg(&shear_rate_in[index_i]);
        float rho0_i = c_materials[mat_id_i].rest_density;
        float p_floor = rho0_i * fabsf(c_sim.gravity.y) * c_granular.particle_spacing;
        float p_eff_i = fmaxf(p_i, p_floor);
        eta_i = compute_muI_eta(gamma_dot_i, p_eff_i, rho_i);
    }

    bool do_vort_eta = false;
    float omega_mag_i = 0.0f;
    float4 vort_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 eta_vort = make_float3(0.0f, 0.0f, 0.0f);
    if (is_fluid_i && c_granular.vorticity_epsilon > 0.0f) {
        vort_i = __ldg(&vorticity_in[index_i]);
        omega_mag_i = vort_i.w;
        do_vort_eta = (omega_mag_i > 1e-6f);
    }

    // Iterate compact neighbor list
    uint nb_count = neighbor_count_in[index_i];
    uint nb_base = index_i * max_nb;

    for (uint k = 0; k < nb_count; k++) {
        uint index_j = neighbor_indices[nb_base + k];

        float4 pos4_j = __ldg(&position[index_j]);
        uint pi_j = __ldg(&packed_info[index_j]);
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

        // Vorticity eta (FLUID only)
        if (do_vort_eta) {
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

        int behavior_j = GET_BEHAVIOR(pi_j);
        if (is_gas_i != (behavior_j == GAS)) continue;

        float3 vel_j = make_float3(vel4_j.x, vel4_j.y, vel4_j.z);
        float rho_j = pos4_j.w;

        // STATIC boundary
        if (behavior_j == STATIC) {
            uint mat_id_j = GET_MATERIAL_ID(pi_j);
            float3 grad_s = grad_spiky_variable(r, rlen, h);

            if (mat_id_j == MAT_RIGID && d_rigid_bodies != 0) {
                float psi_b = m_j;
                float rho0 = c_materials[mat_id_i].rest_density;
                float press_akinci = (p_i / (rho_i * rho_i)) + (p_i / (rho0 * rho0));
                float3 F_on_fluid = make_float3(
                    psi_b * press_akinci * grad_s.x,
                    psi_b * press_akinci * grad_s.y,
                    psi_b * press_akinci * grad_s.z
                );
                f_pressure.x += F_on_fluid.x;
                f_pressure.y += F_on_fluid.y;
                f_pressure.z += F_on_fluid.z;

                int body_id = GET_BODY_ID(pi_j);
                float4 rb_pos = __ldg(&d_rigid_bodies[body_id].position);
                float4 rb_linvel = __ldg(&d_rigid_bodies[body_id].lin_vel);
                float4 rb_angvel = __ldg(&d_rigid_bodies[body_id].ang_vel);
                float3 r_b = make_float3(pos4_j.x - rb_pos.x, pos4_j.y - rb_pos.y, pos4_j.z - rb_pos.z);
                float3 vel_boundary = make_float3(
                    rb_linvel.x + (rb_angvel.y * r_b.z - rb_angvel.z * r_b.y),
                    rb_linvel.y + (rb_angvel.z * r_b.x - rb_angvel.x * r_b.z),
                    rb_linvel.z + (rb_angvel.x * r_b.y - rb_angvel.y * r_b.x)
                );
                float lap_v_b = lap_visc_variable(rlen, h);
                float mu_b = is_granular_i ? c_precalc.viscosity_precalc
                           : c_materials[mat_id_i].base_viscosity * c_precalc.viscosity_lap_coeff;
                float visc_factor_b = mu_b * psi_b * lap_v_b / fmaxf(rho_i, RHO_EPSILON);
                float3 F_visc = make_float3(
                    (vel_boundary.x - vel_i.x) * visc_factor_b,
                    (vel_boundary.y - vel_i.y) * visc_factor_b,
                    (vel_boundary.z - vel_i.z) * visc_factor_b
                );
                f_viscosity.x += F_visc.x;
                f_viscosity.y += F_visc.y;
                f_viscosity.z += F_visc.z;

                float pp = c_precalc.pressure_precalc;
                float fs = is_granular_i ? 1.0f : c_granular.force_scale;
                float m_i = __ldg(&mass[index_i]);
                float3 a_on_fluid = make_float3(
                    (pp * F_on_fluid.x + F_visc.x) * fs,
                    (pp * F_on_fluid.y + F_visc.y) * fs,
                    (pp * F_on_fluid.z + F_visc.z) * fs
                );
                float3 F_on_body = make_float3(-m_i * a_on_fluid.x, -m_i * a_on_fluid.y, -m_i * a_on_fluid.z);
                float3 tau = make_float3(
                    r_b.y * F_on_body.z - r_b.z * F_on_body.y,
                    r_b.z * F_on_body.x - r_b.x * F_on_body.z,
                    r_b.x * F_on_body.y - r_b.y * F_on_body.x
                );
                warp_reduce_accumulate(d_rigid_forces, F_on_body, body_id);
                warp_reduce_accumulate(d_rigid_torques, tau, body_id);
            } else {
                float press_sym_b = 2.0f * (p_i / (rho_i * rho_i));
                f_pressure.x += m_j * press_sym_b * grad_s.x;
                f_pressure.y += m_j * press_sym_b * grad_s.y;
                f_pressure.z += m_j * press_sym_b * grad_s.z;
                float lap_v_b = lap_visc_variable(rlen, h);
                float mu_b = is_granular_i ? c_precalc.viscosity_precalc
                           : c_materials[mat_id_i].base_viscosity * c_precalc.viscosity_lap_coeff;
                float visc_factor_b = mu_b * m_j * lap_v_b / fmaxf(rho_i, RHO_EPSILON);
                f_viscosity.x += (-vel_i.x) * visc_factor_b;
                f_viscosity.y += (-vel_i.y) * visc_factor_b;
                f_viscosity.z += (-vel_i.z) * visc_factor_b;
            }
            continue;
        }

        // Pressure force
        uint mat_id_j = GET_MATERIAL_ID(pi_j);
        float p_j = __ldg(&pressure_in[index_j]);
        float press_sym = (p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j));
        float3 grad_s = grad_spiky_variable(r, rlen, h);
        f_pressure.x += m_j * press_sym * grad_s.x;
        f_pressure.y += m_j * press_sym * grad_s.y;
        f_pressure.z += m_j * press_sym * grad_s.z;

        // Viscosity
        float lap_v = lap_visc_variable(rlen, h);
        if (is_granular_i && behavior_j == GRANULAR) {
            float gamma_dot_j = __ldg(&shear_rate_in[index_j]);
            float rho0_j = c_materials[mat_id_j].rest_density;
            float p_floor_j = rho0_j * fabsf(c_sim.gravity.y) * c_granular.particle_spacing;
            float p_eff_j = fmaxf(p_j, p_floor_j);
            float eta_j = compute_muI_eta(gamma_dot_j, p_eff_j, rho_j);
            float eta_ij = 2.0f * eta_i * eta_j / (eta_i + eta_j + 1e-8f);
            float visc_lap_const = c_precalc.viscosity_lap_coeff;
            float visc_factor = eta_ij * visc_lap_const * m_j * lap_v / (rho_j * rho_i);
            f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
            f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
            f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
        } else if (is_granular_i) {
            float visc_factor = c_precalc.viscosity_precalc * m_j * lap_v / (rho_j * rho_i);
            f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
            f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
            f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
        } else {
            float mu_i = c_materials[mat_id_i].base_viscosity;
            float mu_j = c_materials[mat_id_j].base_viscosity;
            float mu_ij = 2.0f * mu_i * mu_j / (mu_i + mu_j + 1e-8f);
            float visc_factor = mu_ij * c_precalc.viscosity_lap_coeff * m_j * lap_v / rho_j;
            f_viscosity.x += (vel_j.x - vel_i.x) * visc_factor;
            f_viscosity.y += (vel_j.y - vel_i.y) * visc_factor;
            f_viscosity.z += (vel_j.z - vel_i.z) * visc_factor;
        }

        // XSPH
        if (is_fluid_i || is_granular_i) {
            float w = W_poly6(rlen_sq, h_sq);
            float rho_avg = 0.5f * (rho_i + rho_j);
            float xsph_factor = (m_j / rho_avg) * w;
            xsph_sum.x += (vel_j.x - vel_i.x) * xsph_factor;
            xsph_sum.y += (vel_j.y - vel_i.y) * xsph_factor;
            xsph_sum.z += (vel_j.z - vel_i.z) * xsph_factor;
        }
    }

    // Vorticity confinement (identical to K_Step2)
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

    float3 f_surface_tension = make_float3(0.0f, 0.0f, 0.0f);
    if (is_fluid_i && c_granular.surface_tension_gamma > 0.0f) {
        float4 norm_i = __ldg(&normal_in[index_i]);
        float nc_i = norm_i.w;
        if (nc_i < 25.0f) {
            float gamma = c_granular.surface_tension_gamma;
            float n_mag = sqrtf(norm_i.x*norm_i.x + norm_i.y*norm_i.y + norm_i.z*norm_i.z);
            if (n_mag > 0.01f) {
                f_surface_tension.x = gamma * norm_i.x;
                f_surface_tension.y = gamma * norm_i.y;
                f_surface_tension.z = gamma * norm_i.z;
            }
        }
    }

    float3 total_force;
    total_force.x = c_precalc.pressure_precalc * f_pressure.x + f_viscosity.x + f_vorticity_conf.x + f_surface_tension.x;
    total_force.y = c_precalc.pressure_precalc * f_pressure.y + f_viscosity.y + f_vorticity_conf.y + f_surface_tension.y;
    total_force.z = c_precalc.pressure_precalc * f_pressure.z + f_viscosity.z + f_vorticity_conf.z + f_surface_tension.z;

    float fs = is_granular_i ? 1.0f : c_granular.force_scale;
    sph_force_out[index_i] = make_float4(total_force.x * fs, total_force.y * fs, total_force.z * fs, 0.0f);

    if (is_fluid_i || is_granular_i) {
        float eps = c_granular.xsph_epsilon;
        if (is_granular_i) eps = fminf(eps * 3.0f, 0.5f);
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
