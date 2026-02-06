/**
 * K_Step1 -- SPH density summation kernel (Poly6).
 *
 * Per-particle computation:
 *   density_i = mass * kernel_poly6_coeff * SUM_j( (h^2 - |r_ij|^2)^3 )
 *   clamped to max(1.0, ...)
 *
 * Operates on SORTED particle arrays (after hash + sort + reorder).
 * Uses 27-cell neighbor iteration with grid cell_start/cell_end tables.
 *
 * Ported from SPHSimLib/K_SimpleSPH_Step1.inl.
 */

#include "common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
// Constant memory -- uploaded from Python before kernel launch
// ---------------------------------------------------------------------------
__constant__ GridParams c_grid;

struct FluidParams_Step1 {
    float smoothing_length;     // h
    float particle_mass;        // m
    float rest_density;         // rho_0 (unused in Step1 but kept for struct compat)
    float gas_stiffness;        // unused in Step1
    float gas_stiffness_gas;    // unused in Step1
    float gamma;                // unused in Step1
    float viscosity;            // unused in Step1
    float xsph_epsilon;         // unused in Step1
};

struct PrecalcParams_Step1 {
    float smoothing_length_pow2;    // h^2
    float pressure_precalc;         // unused in Step1
    float viscosity_precalc;        // unused in Step1
    float kernel_poly6_coeff;       // 315/(64*pi*h^9)
};

__constant__ FluidParams_Step1   c_fluid;
__constant__ PrecalcParams_Step1 c_precalc;

// ---------------------------------------------------------------------------
// Grid helper functions (suffixed to avoid symbol conflicts across modules)
// ---------------------------------------------------------------------------

__device__ inline int3 calcGridCell_step1(float3 p) {
    return make_int3(
        (int)((p.x - c_grid.grid_min.x) * c_grid.grid_delta.x),
        (int)((p.y - c_grid.grid_min.y) * c_grid.grid_delta.y),
        (int)((p.z - c_grid.grid_min.z) * c_grid.grid_delta.z)
    );
}

__device__ inline int3 clampCell_step1(int3 cell) {
    int rx = (int)c_grid.grid_res.x;
    int ry = (int)c_grid.grid_res.y;
    int rz = (int)c_grid.grid_res.z;
    cell.x = max(0, min(cell.x, rx - 1));
    cell.y = max(0, min(cell.y, ry - 1));
    cell.z = max(0, min(cell.z, rz - 1));
    return cell;
}

// ---------------------------------------------------------------------------
// K_Step1 kernel -- density summation using Poly6
// ---------------------------------------------------------------------------

extern "C" __global__
void K_Step1(
    uint            numParticles,
    const float4*   __restrict__ position,      // sorted positions
    const uint*     __restrict__ cell_start,     // grid cell start indices
    const uint*     __restrict__ cell_end,       // grid cell end indices
    float*          __restrict__ density_out     // output: density per particle
) {
    uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_i >= numParticles) return;

    // Read position of particle i
    float4 pos4_i = position[index_i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);

    float h_sq = c_precalc.smoothing_length_pow2;

    // Density accumulator (variable part of Poly6 kernel)
    float sum_density = 0.0f;

    // Grid cell of particle i
    int3 cell_i = calcGridCell_step1(pos_i);
    cell_i = clampCell_step1(cell_i);

    int rx = (int)c_grid.grid_res.x;
    int ry = (int)c_grid.grid_res.y;
    int rz = (int)c_grid.grid_res.z;

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
                uint end = cell_end[hash];

                for (uint index_j = start; index_j < end; index_j++) {
                    // Include self-interaction for density (particle contributes
                    // to its own density with rlen_sq = 0 -> (h^2)^3)

                    float4 pos4_j = __ldg(&position[index_j]);
                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float rlen_sq = r.x * r.x + r.y * r.y + r.z * r.z;

                    if (rlen_sq <= h_sq) {
                        // Wpoly6 variable part: (h^2 - |r|^2)^3
                        float diff = h_sq - rlen_sq;
                        sum_density += diff * diff * diff;
                    }
                }
            }
        }
    }

    // density = mass * poly6_coeff * sum_density, clamped to >= 1.0
    float density = c_fluid.particle_mass
                  * c_precalc.kernel_poly6_coeff
                  * sum_density;
    density_out[index_i] = fmaxf(density, 1.0f);
}
