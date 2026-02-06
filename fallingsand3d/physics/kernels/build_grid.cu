/*
 * build_grid.cu -- K_BuildDataStruct kernel for cell start/end tables.
 *
 * Detects cell boundaries in the sorted hash array and writes
 * cell_indexes_start / cell_indexes_end tables.
 *
 * Ported from SPHSimLib/K_UniformGrid_Update.inl::K_Grid_UpdateSorted.
 *
 * Pre-conditions:
 *   - sorted_hashes is sorted in ascending order
 *   - cell_indexes_start is memset to 0xFFFFFFFF before launch
 *
 * GridParams read from __constant__ c_grid declared in common.cuh.
 */

#include "common.cuh"

extern "C" __global__
void K_BuildDataStruct(uint numParticles,
                       const uint* __restrict__ sorted_hashes,
                       uint* __restrict__ cell_indexes_start,
                       uint* __restrict__ cell_indexes_end) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    uint hash = sorted_hashes[idx];

    // Detect start of a new cell: either first particle or hash changed.
    if (idx == 0 || hash != sorted_hashes[idx - 1]) {
        cell_indexes_start[hash] = idx;

        // The previous particle's cell ends here.
        if (idx > 0) {
            cell_indexes_end[sorted_hashes[idx - 1]] = idx;
        }
    }

    // Last particle: close out its cell.
    if (idx == numParticles - 1) {
        cell_indexes_end[hash] = idx + 1;
    }
}
