/*
 * counting_sort.cu -- Counting sort for spatial hash grid.
 *
 * Replaces cupy.argsort (Thrust radix sort) with a 3-phase counting sort:
 *   Phase 1: K_CalcHash     -- compute per-particle grid cell hash (reused from hash_sort.cu)
 *   Phase 2: K_Histogram    -- count particles per cell via atomicAdd
 *   Phase 3: prefix sum     -- cupy.cumsum on histogram (CUB, graph-capturable)
 *   Phase 4: K_ScatterReorder -- scatter particles to sorted order + build cell_start/cell_end
 *
 * All phases are CUDA-graph-capturable (no Thrust synchronization).
 * K_ScatterReorder combines scatter, gather, and cell boundary detection in one kernel.
 *
 * Pre-conditions:
 *   - histogram[] must be zeroed before K_Histogram
 *   - write_offset[] must be zeroed before K_ScatterReorder
 *   - cell_start[] is filled by the prefix-sum (cupy.cumsum) of histogram[] and
 *     does NOT need a 0xFFFFFFFF sentinel -- that sentinel was used by the old
 *     hash_sort K_BuildDataStruct path and does not apply here.
 *
 * GridParams read from __constant__ c_grid declared in common.cuh.
 */

#include "common.cuh"

// Uses calcGridCell() and spatialHash() from common.cuh.

/* ======================================================================
 * K_CalcHashCS -- compute per-particle hash (same as hash_sort.cu K_CalcHash)
 * Duplicated here so entire pipeline is in one RawModule.
 * ====================================================================== */
extern "C" __global__
void K_CalcHashCS(
    const uint     numParticles,
    const float4*  __restrict__ positions,
    uint*          __restrict__ hashes
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float4 p4 = positions[idx];
    float3 p = make_float3(p4.x, p4.y, p4.z);

    int3 cell = calcGridCell(p);
    uint hash = spatialHash(cell);

    hashes[idx] = hash;
}

/* ======================================================================
 * K_Histogram -- count particles per cell.
 *
 * Each thread atomically increments histogram[hash[i]].
 * histogram must be zeroed before launch.
 * ====================================================================== */
extern "C" __global__
void K_Histogram(
    const uint    numParticles,
    const uint*   __restrict__ hashes,
    uint*         __restrict__ histogram     // [num_cells], zeroed
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    const uint hash = hashes[idx];
    atomicAdd(&histogram[hash], 1);
}

/* ======================================================================
 * K_ScatterReorder -- scatter particles to sorted positions + build cell tables.
 *
 * cell_start[cell] = prefix_sum[cell]  (first particle offset for cell)
 * write_offset[cell] starts at 0 and incremented via atomicAdd per particle.
 * sorted_pos = cell_start[hash] + atomicAdd(&write_offset[hash], 1)
 *
 * Also writes cell_end = cell_start + final_count (via write_offset at end).
 * Since multiple threads write to the same cell concurrently, within-cell
 * ordering is non-deterministic (acceptable for SPH).
 *
 * Pre-conditions:
 *   - cell_start[] already set from prefix_sum (cell_start[c] = sum(histogram[0..c-1]))
 *   - write_offset[] zeroed
 * ====================================================================== */
extern "C" __global__
void K_ScatterReorder(
    const uint     numParticles,
    const uint*   __restrict__ hashes,
    const uint*   __restrict__ cell_start,    // [num_cells] from prefix sum
    uint*         __restrict__ write_offset,   // [num_cells] zeroed, scratch
    uint*         __restrict__ cell_end_out,   // [num_cells] output
    uint*         __restrict__ sort_perm_out,  // [N] output: maps sorted idx -> original idx
    // Unsorted inputs (read)
    const float4* __restrict__ position_in,
    const float4* __restrict__ velocity_in,
    const float*  __restrict__ mass_in,
    const uint*   __restrict__ packed_info_in,
    const float*  __restrict__ temperature_in,
    const float*  __restrict__ health_in,
    const float*  __restrict__ lifetime_in,
    const unsigned char* __restrict__ sleep_counter_in,
    const float*  __restrict__ kappa_in,
    const float4* __restrict__ particle_dye_in,
    const float4* __restrict__ angular_velocity_in,
    const float*  __restrict__ kappa_v_in,          // DFSPH div warm-start (PERF-008)
    // Sorted outputs (write)
    uint*         __restrict__ sorted_hashes_out,
    float4*       __restrict__ position_out,
    float4*       __restrict__ velocity_out,
    float*        __restrict__ mass_out,
    uint*         __restrict__ packed_info_out,
    float*        __restrict__ temperature_out,
    float*        __restrict__ health_out,
    float*        __restrict__ lifetime_out,
    unsigned char* __restrict__ sleep_counter_out,
    float*        __restrict__ kappa_out,           // NULL = skip (DFSPH only)
    float4*       __restrict__ particle_dye_out,
    float4*       __restrict__ angular_velocity_out,
    float*        __restrict__ kappa_v_out,         // DFSPH div warm-start (PERF-008), NULL = skip
    // FP16 velocity output (OPT-4.3: half bandwidth in neighbor loops)
    void*         __restrict__ velocity_h_out,      // half4, 8 bytes per particle (NULL = skip)
    // Fixed-point temperature + FP16 dye outputs (PERF-011)
    void*         __restrict__ temperature_h_out,   // uint16 T*8 (0.125 K), 2 bytes per particle (NULL = skip)
    void*         __restrict__ dye_h_out,           // half4, 8 bytes per particle (NULL = skip)
    // Grid-reuse drift anchor: position at this full sort, in sorted order.
    // K_GatherReorder never writes it, so between full sorts anchor_out[i]
    // stays the sort-time position of the particle in sorted slot i.
    float4*       __restrict__ anchor_out           // (NULL = skip)
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    const uint hash = hashes[idx];

    // Claim a slot in the sorted output for this cell
    const uint local_offset = atomicAdd(&write_offset[hash], 1);
    const uint sorted_idx = cell_start[hash] + local_offset;

    // Write sort permutation (sorted -> original mapping)
    sort_perm_out[sorted_idx] = idx;

    // Write sorted hash
    sorted_hashes_out[sorted_idx] = hash;

    // Gather and scatter all particle arrays
    const float4 pos4 = __ldg(&position_in[idx]);
    position_out[sorted_idx]      = pos4;
    if (anchor_out) anchor_out[sorted_idx] = pos4;
    float4 vel = __ldg(&velocity_in[idx]);
    velocity_out[sorted_idx]      = vel;
    mass_out[sorted_idx]          = __ldg(&mass_in[idx]);
    packed_info_out[sorted_idx]   = __ldg(&packed_info_in[idx]);
    float temp = __ldg(&temperature_in[idx]);
    temperature_out[sorted_idx]   = temp;
    health_out[sorted_idx]        = __ldg(&health_in[idx]);
    lifetime_out[sorted_idx]      = __ldg(&lifetime_in[idx]);
    sleep_counter_out[sorted_idx] = __ldg(&sleep_counter_in[idx]);
    // DFSPH-only warm-start state: NULL when the active solver never reads it
    // (PBF). Skipping it saves the scattered gather as well as the store.
    if (kappa_out) kappa_out[sorted_idx] = __ldg(&kappa_in[idx]);
    float4 dye = __ldg(&particle_dye_in[idx]);
    particle_dye_out[sorted_idx]  = dye;
    angular_velocity_out[sorted_idx] = __ldg(&angular_velocity_in[idx]);
    if (kappa_v_out) kappa_v_out[sorted_idx] = __ldg(&kappa_v_in[idx]);

    // Also write FP16 velocity copy for neighbor loop bandwidth reduction (OPT-4.3).
    // NULL when the active solver has no FP16 neighbor reads (PBF).
    if (velocity_h_out) store_half4((uint2*)velocity_h_out + sorted_idx, vel);

    // Fixed-point temperature (0.125 K) + FP16 dye copies for neighbor loop bandwidth (PERF-011)
    if (temperature_h_out) store_t16((unsigned short*)temperature_h_out + sorted_idx, temp);
    if (dye_h_out) store_half4((uint2*)dye_h_out + sorted_idx, dye);
}

/* ======================================================================
 * K_GatherReorder -- re-scatter unsorted data to sorted order using
 * existing sort_perm (from previous frame's counting sort).
 *
 * Used for "grid reuse" when particles moved less than 0.25*h since last sort.
 * Cell_start/cell_end are reused from the previous frame.
 * It must not write the drift anchor: the anchor holds the positions of the
 * last full sort, and the finalize kernels measure drift against it.
 * This is cheaper than full counting sort: no hash, no histogram, no prefix sum,
 * no atomics. Just a simple indexed gather with coalesced writes.
 * ====================================================================== */
extern "C" __global__
void K_GatherReorder(
    const uint     numParticles,
    const uint*   __restrict__ sort_perm,         // sorted_idx -> orig_idx (from previous sort)
    // Unsorted inputs (updated by K_Integrate)
    const float4* __restrict__ position_in,
    const float4* __restrict__ velocity_in,
    const float*  __restrict__ mass_in,
    const uint*   __restrict__ packed_info_in,
    const float*  __restrict__ temperature_in,
    const float*  __restrict__ health_in,
    const float*  __restrict__ lifetime_in,
    const unsigned char* __restrict__ sleep_counter_in,
    const float*  __restrict__ kappa_in,
    const float4* __restrict__ particle_dye_in,
    const float4* __restrict__ angular_velocity_in,
    const float*  __restrict__ kappa_v_in,           // DFSPH div warm-start (PERF-008)
    // Sorted outputs (write)
    float4*       __restrict__ position_out,
    float4*       __restrict__ velocity_out,
    float*        __restrict__ mass_out,
    uint*         __restrict__ packed_info_out,
    float*        __restrict__ temperature_out,
    float*        __restrict__ health_out,
    float*        __restrict__ lifetime_out,
    unsigned char* __restrict__ sleep_counter_out,
    float*        __restrict__ kappa_out,           // NULL = skip (DFSPH only)
    float4*       __restrict__ particle_dye_out,
    float4*       __restrict__ angular_velocity_out,
    float*        __restrict__ kappa_v_out,          // DFSPH div warm-start (PERF-008), NULL = skip
    // FP16 velocity output (OPT-4.3)
    void*         __restrict__ velocity_h_out,      // NULL = skip
    // Fixed-point temperature + FP16 dye outputs (PERF-011)
    void*         __restrict__ temperature_h_out,   // uint16 T*8 (0.125 K), 2 bytes per particle (NULL = skip)
    void*         __restrict__ dye_h_out            // half4, 8 bytes per particle (NULL = skip)
) {
    const uint sorted_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_idx >= numParticles) return;

    const uint orig_idx = sort_perm[sorted_idx];

    // Simple indexed gather: coalesced writes, scattered reads through __ldg
    position_out[sorted_idx]      = __ldg(&position_in[orig_idx]);
    float4 vel = __ldg(&velocity_in[orig_idx]);
    velocity_out[sorted_idx]      = vel;
    mass_out[sorted_idx]          = __ldg(&mass_in[orig_idx]);
    packed_info_out[sorted_idx]   = __ldg(&packed_info_in[orig_idx]);
    float temp = __ldg(&temperature_in[orig_idx]);
    temperature_out[sorted_idx]   = temp;
    health_out[sorted_idx]        = __ldg(&health_in[orig_idx]);
    lifetime_out[sorted_idx]      = __ldg(&lifetime_in[orig_idx]);
    sleep_counter_out[sorted_idx] = __ldg(&sleep_counter_in[orig_idx]);
    // DFSPH-only warm-start state: NULL when the active solver never reads it (PBF)
    if (kappa_out) kappa_out[sorted_idx] = __ldg(&kappa_in[orig_idx]);
    float4 dye = __ldg(&particle_dye_in[orig_idx]);
    particle_dye_out[sorted_idx]  = dye;
    angular_velocity_out[sorted_idx] = __ldg(&angular_velocity_in[orig_idx]);
    if (kappa_v_out) kappa_v_out[sorted_idx] = __ldg(&kappa_v_in[orig_idx]);

    // FP16 velocity copy (OPT-4.3); NULL = skip (PBF)
    if (velocity_h_out) store_half4((uint2*)velocity_h_out + sorted_idx, vel);

    // Fixed-point temperature (0.125 K) + FP16 dye copies (PERF-011)
    if (temperature_h_out) store_t16((unsigned short*)temperature_h_out + sorted_idx, temp);
    if (dye_h_out) store_half4((uint2*)dye_h_out + sorted_idx, dye);
}

/* ======================================================================
 * K_BuildCellEnd -- compute cell_end from cell_start + write_offset (= count).
 *
 * cell_end[c] = cell_start[c] + write_offset[c]
 * Only writes for cells with particles (write_offset > 0).
 * cell_end for empty cells stays at 0 (from memset).
 * ====================================================================== */
extern "C" __global__
void K_BuildCellEnd(
    const uint    numCells,
    const uint*   __restrict__ cell_start,
    const uint*   __restrict__ write_offset,   // = count per cell after scatter
    uint*         __restrict__ cell_end_out
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCells) return;

    const uint count = write_offset[idx];
    if (count > 0) {
        cell_end_out[idx] = cell_start[idx] + count;
    }
    // else: cell_end stays at 0 from memset
}
