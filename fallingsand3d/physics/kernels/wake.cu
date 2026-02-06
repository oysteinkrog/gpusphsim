/*
 * wake.cu -- Two-phase wake propagation kernels.
 *
 * Phase 1 (K_MarkWakeCells): Particles with HAS_JUST_WOKE flag set
 * mark their own grid cell and 26 neighboring cells via atomicOr(1).
 *
 * Phase 2 (K_WakeSleepers): Sleeping particles in flagged cells wake up
 * (CLEAR_SLEEPING, reset sleep_counter to 0).
 *
 * Phase 3 (K_ClearJustWoke): Clear the JUST_WOKE flag from all particles.
 *
 * All kernels operate on UNSORTED arrays (after Integrate writeback).
 *
 * Constant memory used:
 *   c_grid -- GridParams from common.cuh (for position -> cell mapping)
 */

#include "common.cuh"

/* ======================================================================
 * Grid cell computation (inlined, same as hash_sort.cu)
 * ====================================================================== */

__device__ inline int calcGridHash_wake(int3 cell) {
    return cell.z * c_grid.grid_res.y * c_grid.grid_res.x
         + cell.y * c_grid.grid_res.x
         + cell.x;
}

__device__ inline int3 calcGridCell_wake(float3 pos) {
    int3 cell;
    cell.x = (int)((pos.x - c_grid.grid_min.x) * c_grid.grid_delta.x);
    cell.y = (int)((pos.y - c_grid.grid_min.y) * c_grid.grid_delta.y);
    cell.z = (int)((pos.z - c_grid.grid_min.z) * c_grid.grid_delta.z);
    // Clamp to valid range
    cell.x = max(0, min(cell.x, c_grid.grid_res.x - 1));
    cell.y = max(0, min(cell.y, c_grid.grid_res.y - 1));
    cell.z = max(0, min(cell.z, c_grid.grid_res.z - 1));
    return cell;
}

/* ======================================================================
 * K_MarkWakeCells -- Phase 1: mark 3x3x3 neighbor cells for just-woke
 *                   particles via atomicOr.
 * ====================================================================== */

extern "C" __global__
void K_MarkWakeCells(
    uint        numParticles,
    const float4* __restrict__ position,       // unsorted positions
    const uint*   __restrict__ packed_info,     // unsorted packed_info
    uint*         __restrict__ cell_wake_flags  // num_cells uint32 array
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];

    // Only process particles that just woke up this frame
    if (!HAS_JUST_WOKE(pi)) return;

    // Compute this particle's grid cell
    float4 pos4 = position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
    int3 cell = calcGridCell_wake(pos);

    // Mark own cell and 26 neighbors (3x3x3 block)
    for (int dz = -1; dz <= 1; dz++) {
        int nz = cell.z + dz;
        if (nz < 0 || nz >= c_grid.grid_res.z) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cell.y + dy;
            if (ny < 0 || ny >= c_grid.grid_res.y) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cell.x + dx;
                if (nx < 0 || nx >= c_grid.grid_res.x) continue;

                int hash = nz * c_grid.grid_res.y * c_grid.grid_res.x
                         + ny * c_grid.grid_res.x
                         + nx;
                atomicOr(&cell_wake_flags[hash], 1u);
            }
        }
    }
}

/* ======================================================================
 * K_WakeSleepersAndClearJustWoke -- Phase 2+3 fused:
 *   - Wake sleeping particles in flagged cells
 *   - Clear JUST_WOKE flag from all particles
 * Both iterate all particles and modify packed_info; no data dependency.
 * ====================================================================== */

extern "C" __global__
void K_WakeSleepersAndClearJustWoke(
    uint          numParticles,
    const float4* __restrict__ position,       // unsorted positions
    uint*         __restrict__ packed_info,     // unsorted packed_info (read+write)
    unsigned char* __restrict__ sleep_counter,  // unsorted sleep counter (write)
    const uint*   __restrict__ cell_wake_flags  // num_cells uint32 array
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    uint new_pi = pi;

    // Wake sleeping particles in flagged cells
    if (IS_SLEEPING(pi)) {
        float4 pos4 = position[i];
        float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
        int3 cell = calcGridCell_wake(pos);
        int hash = calcGridHash_wake(cell);

        if (cell_wake_flags[hash] != 0u) {
            new_pi = CLEAR_SLEEPING(new_pi);
            sleep_counter[i] = 0;
        }
    }

    // Clear JUST_WOKE flag
    if (HAS_JUST_WOKE(new_pi)) {
        new_pi = CLEAR_JUST_WOKE(new_pi);
    }

    // Only write back if modified
    if (new_pi != pi) {
        packed_info[i] = new_pi;
    }
}
