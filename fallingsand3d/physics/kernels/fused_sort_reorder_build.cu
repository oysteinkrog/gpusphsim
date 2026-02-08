/*
 * fused_sort_reorder_build.cu -- Fused kernel combining reorder + build_grid.
 *
 * After argsort produces sort_perm, this single kernel:
 *   1. Reads sort_perm[idx] to get the original particle index
 *   2. Reads hashes[perm] and writes sorted_hashes[idx]
 *   3. Detects cell boundaries and writes cell_start/cell_end
 *   4. Gathers 8 particle arrays from unsorted[perm] -> sorted[idx]
 *
 * This replaces three separate operations:
 *   - CuPy fancy-index gather for sorted_hashes
 *   - K_FusedReorder kernel
 *   - K_BuildDataStruct kernel
 *
 * Pre-conditions:
 *   - sort_perm is a valid permutation of [0..N-1] sorted by hash
 *   - cell_indexes_start is memset to 0xFFFFFFFF before launch
 *   - cell_indexes_end is memset to 0x00 before launch
 *
 * Cell boundary detection reads unsorted hashes via hashes[sort_perm[idx-1]]
 * (read-only array, no write-after-read hazard).
 *
 * GridParams read from __constant__ c_grid declared in common.cuh.
 */

#include "common.cuh"

extern "C" __global__
void K_FusedSortReorderBuild(
    const uint     numParticles,
    const uint*   __restrict__ sort_perm,           // argsort result
    const uint*   __restrict__ hashes,              // unsorted hashes
    uint*         __restrict__ sorted_hashes,       // output: sorted hashes
    uint*         __restrict__ cell_indexes_start,  // output: cell start table
    uint*         __restrict__ cell_indexes_end,    // output: cell end table
    // Unsorted inputs (read) -- 8 arrays
    const float4* __restrict__ position_in,
    const float4* __restrict__ velocity_in,
    const float*  __restrict__ mass_in,
    const uint*   __restrict__ packed_info_in,
    const float*  __restrict__ temperature_in,
    const float*  __restrict__ health_in,
    const float*  __restrict__ lifetime_in,
    const unsigned char* __restrict__ sleep_counter_in,
    const float*  __restrict__ kappa_in,
    // Sorted outputs (write) -- 9 arrays
    float4*       __restrict__ position_out,
    float4*       __restrict__ velocity_out,
    float*        __restrict__ mass_out,
    uint*         __restrict__ packed_info_out,
    float*        __restrict__ temperature_out,
    float*        __restrict__ health_out,
    float*        __restrict__ lifetime_out,
    unsigned char* __restrict__ sleep_counter_out,
    float*        __restrict__ kappa_out
)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // 1. Read permutation to get original particle index
    const uint perm = sort_perm[idx];

    // 2. Read unsorted hash and write sorted hash
    const uint hash = __ldg(&hashes[perm]);
    sorted_hashes[idx] = hash;

    // 3. Cell boundary detection (same logic as K_BuildDataStruct)
    //    Read neighbor's hash from unsorted array via sort_perm (no RAW hazard)
    if (idx == 0) {
        // First particle always starts a cell
        cell_indexes_start[hash] = 0;
    } else {
        const uint prev_hash = __ldg(&hashes[sort_perm[idx - 1]]);
        if (hash != prev_hash) {
            cell_indexes_start[hash] = idx;
            cell_indexes_end[prev_hash] = idx;
        }
    }
    // Last particle closes its cell
    if (idx == numParticles - 1) {
        cell_indexes_end[hash] = idx + 1;
    }

    // 4. Gather 9 particle arrays from unsorted[perm] -> sorted[idx]
    position_out[idx]      = __ldg(&position_in[perm]);
    velocity_out[idx]      = __ldg(&velocity_in[perm]);
    mass_out[idx]          = __ldg(&mass_in[perm]);
    packed_info_out[idx]   = __ldg(&packed_info_in[perm]);
    temperature_out[idx]   = __ldg(&temperature_in[perm]);
    health_out[idx]        = __ldg(&health_in[perm]);
    lifetime_out[idx]      = __ldg(&lifetime_in[perm]);
    sleep_counter_out[idx] = __ldg(&sleep_counter_in[perm]);
    kappa_out[idx]         = __ldg(&kappa_in[perm]);
}
