/**
 * K_FusedReorder -- Reorder particle data from unsorted to sorted order.
 *
 * After hash + argsort, sort_indexes[sorted_slot] = original_particle_id.
 * This kernel gathers unsorted data into sorted arrays using that permutation.
 *
 * Reorders: position, velocity, veleval, behavior_class, flags.
 *
 * Unlike the C++ K_Grid_UpdateSorted, this does NOT build cell start/end
 * tables (that's already done by K_BuildDataStruct). It also does NOT
 * apply scale_to_simulation (Python port operates in world space).
 */

#include "common.cuh"

extern "C" __global__
void K_FusedReorder(
    uint            numParticles,
    const uint*     __restrict__ sort_indexes,     // sort_indexes[sorted] = original
    // Unsorted inputs
    const float4*   __restrict__ position_in,
    const float4*   __restrict__ velocity_in,
    const float4*   __restrict__ veleval_in,
    const int*      __restrict__ behavior_class_in,
    const uint*     __restrict__ flags_in,
    // Sorted outputs
    float4*         __restrict__ position_out,
    float4*         __restrict__ velocity_out,
    float4*         __restrict__ veleval_out,
    int*            __restrict__ behavior_class_out,
    uint*           __restrict__ flags_out
) {
    uint sorted_slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_slot >= numParticles) return;

    uint original_id = sort_indexes[sorted_slot];

    position_out[sorted_slot]       = position_in[original_id];
    velocity_out[sorted_slot]       = velocity_in[original_id];
    veleval_out[sorted_slot]        = veleval_in[original_id];
    behavior_class_out[sorted_slot] = behavior_class_in[original_id];
    flags_out[sorted_slot]          = flags_in[original_id];
}
