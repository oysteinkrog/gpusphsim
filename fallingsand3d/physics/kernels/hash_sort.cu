/*
 * hash_sort.cu -- K_CalcHash kernel for spatial grid hashing.
 *
 * Ported from SPHSimLib/K_UniformGrid_Utils.inl calcGridCell + calcGridHash
 * (linear, NOT Morton). Each thread computes the grid cell for one particle
 * and writes a hash key + original index.
 *
 * GridParams read from __constant__ c_grid declared in common.cuh.
 */

#include "common.cuh"

// Compute grid cell coordinates from a world-space position.
// Ported from SPHSimLib/K_UniformGrid_Utils.inl::calcGridCell.
__device__ inline int3 calcGridCell(float3 p) {
    return make_int3(
        (int)((p.x - c_grid.grid_min.x) * c_grid.grid_delta.x),
        (int)((p.y - c_grid.grid_min.y) * c_grid.grid_delta.y),
        (int)((p.z - c_grid.grid_min.z) * c_grid.grid_delta.z)
    );
}

// Clamp grid cell to valid range [0, grid_res-1].
__device__ inline int3 clampCell(int3 cell) {
    cell.x = max(0, min(cell.x, c_grid.grid_res.x - 1));
    cell.y = max(0, min(cell.y, c_grid.grid_res.y - 1));
    cell.z = max(0, min(cell.z, c_grid.grid_res.z - 1));
    return cell;
}

// Linear hash: z * res_y * res_x + y * res_x + x
// Ported from SPHSimLib/K_UniformGrid_Utils.inl::calcGridHash (non-Morton).
__device__ inline uint calcGridHash(int3 cell) {
    return (uint)(cell.z * c_grid.grid_res.y * c_grid.grid_res.x
                + cell.y * c_grid.grid_res.x
                + cell.x);
}

// K_CalcHash: each thread computes the grid hash for one particle.
// Positions are float4 (x,y,z,w) where w is unused.
extern "C" __global__
void K_CalcHash(uint numParticles,
                const float4* __restrict__ positions,
                uint* __restrict__ hashes,
                uint* __restrict__ indices) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float4 p4 = positions[idx];
    float3 p = make_float3(p4.x, p4.y, p4.z);

    int3 cell = calcGridCell(p);
    cell = clampCell(cell);
    uint hash = calcGridHash(cell);

    hashes[idx] = hash;
    indices[idx] = idx;
}
