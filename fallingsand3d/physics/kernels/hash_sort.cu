/*
 * hash_sort.cu -- K_CalcHash kernel for spatial grid hashing.
 *
 * Each thread computes the spatial hash for one particle's grid cell.
 * Uses calcGridCell() and spatialHash() from common.cuh.
 *
 * GridParams read from __constant__ c_grid declared in common.cuh.
 */

#include "common.cuh"

// K_CalcHash: each thread computes the grid hash for one particle.
// Positions are float4 (x,y,z,w) where w is unused.
extern "C" __global__
void K_CalcHash(uint numParticles,
                const float4* __restrict__ positions,
                uint* __restrict__ hashes) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float4 p4 = positions[idx];
    float3 p = make_float3(p4.x, p4.y, p4.z);

    int3 cell = calcGridCell(p);
    uint hash = spatialHash(cell);

    hashes[idx] = hash;
}
