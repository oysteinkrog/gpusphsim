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

// Uses calcGridCell() and spatialHash() from common.cuh.

/* ======================================================================
 * K_MarkWakeCells -- Phase 1: mark 3x3x3 neighbor cells for just-woke
 *                   particles via atomicOr.
 * ====================================================================== */

extern "C" __global__
void K_MarkWakeCells(
    uint        numParticles,
    const float4* __restrict__ position,       // unsorted positions
    const float4* __restrict__ velocity,       // unsorted velocities
    const uint*   __restrict__ packed_info,     // unsorted packed_info
    uint*         __restrict__ cell_wake_flags  // num_cells uint32 array
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];

    // Mark cells for two categories of particles:
    // 1. Particles that just woke up this frame (wake cascade)
    // 2. Active non-sleeping, non-STATIC particles moving fast enough
    //    (so flowing water wakes sleeping sand in adjacent cells)
    bool should_mark = HAS_JUST_WOKE(pi);
    if (!should_mark && !IS_SLEEPING(pi) && GET_BEHAVIOR(pi) != STATIC) {
        float4 v = velocity[i];
        float v_sq = v.x*v.x + v.y*v.y + v.z*v.z;
        should_mark = (v_sq > 0.02f * 0.02f);  // V_WAKE_SQ
    }
    if (!should_mark) return;

    // Compute this particle's grid cell
    float4 pos4 = position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
    int3 cell = calcGridCell(pos);

    // Mark own cell and 26 neighbors (3x3x3 block)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint hash = spatialHash(cell.x + dx, cell.y + dy, cell.z + dz);
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
        int3 cell = calcGridCell(pos);
        uint hash = spatialHash(cell);

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
