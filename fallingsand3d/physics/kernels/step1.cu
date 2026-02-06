/*
 * step1.cu -- K_Step1 SPH density summation + strain-rate tensor kernel.
 *
 * Per-particle computation:
 *   density_sum += m_j * (h^2 - |r_ij|^2)^3   for all neighbors j within h
 *   density_i = max(1.0, poly6_coeff * density_sum)
 *
 * For GRANULAR particles only, also computes the symmetric strain-rate tensor D
 * using the SPH velocity gradient with spiky gradient weighting:
 *   D_ab = 0.5 * sum_j (m_j/rho_j) * (dv_a * gradW_b + dv_b * gradW_a)
 * Then: gamma_dot = sqrt(2 * D:D) = sqrt(2 * (Dxx^2 + Dyy^2 + Dzz^2 + 2*(Dxy^2 + Dxz^2 + Dyz^2)))
 *
 * Operates on SORTED particle arrays (after hash + sort + reorder).
 * Uses 27-cell neighbor iteration with grid cell_start/cell_end tables.
 * Self-interaction IS included for density (j==i NOT skipped).
 * Self-interaction is skipped for strain-rate (j==i skipped).
 * Per-particle mass m_j supports multi-material and mass splitting.
 *
 * Ported from SPHSimLib/K_SimpleSPH_Step1.inl + K_SPH_Kernels_poly6.inl.
 * Neighbor iteration ported from SPHSimLib/K_UniformGrid_Utils.inl.
 *
 * Constant memory (c_grid, c_sim, c_precalc) declared in common.cuh.
 */

#include "common.cuh"

extern "C" __global__
void K_Step1(
    uint            numParticles,
    const float4*   __restrict__ position,      // sorted positions
    const float4*   __restrict__ velocity,      // sorted velocities (for strain-rate)
    const float*    __restrict__ mass,           // sorted per-particle mass
    const float*    __restrict__ density_in,     // sorted density from previous step (for strain-rate m_j/rho_j weighting; NULL on first frame)
    const uint*     __restrict__ packed_info,    // sorted packed_info (for behavior class check)
    const uint*     __restrict__ cell_start,     // grid cell start indices
    const uint*     __restrict__ cell_end,       // grid cell end indices
    float*          __restrict__ density_out,    // output: density per particle
    float*          __restrict__ shear_rate_out  // output: gamma_dot per particle (0 for non-GRANULAR)
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
    bool is_granular = (behavior_i == GRANULAR);

    // Read velocity for strain-rate (only used if GRANULAR, but cheap to read)
    float3 vel_i;
    if (is_granular) {
        float4 vel4_i = velocity[index_i];
        vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);
    }

    // Density accumulator (variable part of Poly6 kernel)
    float sum_density = 0.0f;

    // Strain-rate tensor accumulators (6 symmetric components)
    // D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
    float Dxx = 0.0f, Dyy = 0.0f, Dzz = 0.0f;
    float Dxy = 0.0f, Dxz = 0.0f, Dyz = 0.0f;

    // Grid cell of particle i
    int3 cell_i = make_int3(
        (int)((pos_i.x - c_grid.grid_min.x) * c_grid.grid_delta.x),
        (int)((pos_i.y - c_grid.grid_min.y) * c_grid.grid_delta.y),
        (int)((pos_i.z - c_grid.grid_min.z) * c_grid.grid_delta.z)
    );
    // Clamp to valid range
    cell_i.x = max(0, min(cell_i.x, c_grid.grid_res.x - 1));
    cell_i.y = max(0, min(cell_i.y, c_grid.grid_res.y - 1));
    cell_i.z = max(0, min(cell_i.z, c_grid.grid_res.z - 1));

    int rx = c_grid.grid_res.x;
    int ry = c_grid.grid_res.y;
    int rz = c_grid.grid_res.z;

    // Iterate 27 neighbor cells
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
                uint end_idx = cell_end[hash];

                for (uint index_j = start; index_j < end_idx; index_j++) {
                    float4 pos4_j = __ldg(&position[index_j]);
                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;

                    if (r_sq <= h_sq) {
                        // --- Density: self-interaction included ---
                        float diff = h_sq - r_sq;
                        float m_j = __ldg(&mass[index_j]);
                        sum_density += m_j * diff * diff * diff;

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
