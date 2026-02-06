#include "common.cuh"

// Grid parameters in constant memory -- uploaded from Python before launch.
__constant__ GridParams c_grid;

// K_BuildDataStruct: detect cell boundaries in sorted hash array and write
// cell_indexes_start / cell_indexes_end tables.
//
// Ported from SPHSimLib/K_UniformGrid_Update.inl::K_Grid_UpdateSorted.
//
// Pre-condition: sorted_hashes is sorted in ascending order.
// Pre-condition: cell_indexes_start is memset to 0xFFFFFFFF before launch.
//
// For each particle i:
//   - If i==0 or sorted_hashes[i] != sorted_hashes[i-1]:
//       cell_indexes_start[sorted_hashes[i]] = i
//       if (i > 0) cell_indexes_end[sorted_hashes[i-1]] = i
//   - If i == numParticles - 1:
//       cell_indexes_end[sorted_hashes[i]] = i + 1

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
