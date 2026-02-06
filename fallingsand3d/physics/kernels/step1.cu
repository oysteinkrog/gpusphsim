/*
 * step1.cu -- K_Step1 SPH density summation kernel (Poly6).
 *
 * Per-particle computation:
 *   density_sum += m_j * (h^2 - |r_ij|^2)^3   for all neighbors j within h
 *   density_i = max(1.0, poly6_coeff * density_sum)
 *
 * Operates on SORTED particle arrays (after hash + sort + reorder).
 * Uses 27-cell neighbor iteration with grid cell_start/cell_end tables.
 * Self-interaction IS included (particle contributes to its own density).
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
    const float*    __restrict__ mass,           // sorted per-particle mass
    const uint*     __restrict__ cell_start,     // grid cell start indices
    const uint*     __restrict__ cell_end,       // grid cell end indices
    float*          __restrict__ density_out     // output: density per particle
) {
    uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_i >= numParticles) return;

    // Read position of particle i
    float4 pos4_i = position[index_i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h_sq = c_sim.smoothing_length_sq;

    // Density accumulator (variable part of Poly6 kernel)
    float sum_density = 0.0f;

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
                    // Self-interaction included for density (j==i NOT skipped)

                    float4 pos4_j = __ldg(&position[index_j]);
                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;

                    if (r_sq <= h_sq) {
                        // Poly6 variable: (h^2 - |r|^2)^3
                        float diff = h_sq - r_sq;
                        float m_j = __ldg(&mass[index_j]);
                        sum_density += m_j * diff * diff * diff;
                    }
                }
            }
        }
    }

    // density = poly6_coeff * density_sum, clamped to >= 1.0
    float density = c_precalc.poly6_coeff * sum_density;
    density_out[index_i] = fmaxf(density, 1.0f);
}
