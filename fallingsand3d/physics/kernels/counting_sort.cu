/*
 * counting_sort.cu -- Counting sort for spatial hash grid.
 *
 * Full sort, five launches, no pool allocation and no memset
 * (bd-r5-review-n28.189):
 *   1. K_HashHistogram -- per-particle cell hash, stored and counted (atomicAdd)
 *   2. K_ScanReduce    -- one sum per SCAN_TILE histogram slots -> block_sums[]
 *   3. K_ScanDownsweep -- inclusive prefix sum of histogram[] -> cell_end[]
 *   4. K_ScatterPerm   -- claims each particle's sorted slot, writes sort_perm
 *                         and drains histogram[] back to zero
 *   5. K_GatherReorder -- gathers every particle array through sort_perm
 *
 * Deterministic mode (tests and bisects, bd-r5-review-n28.200): K_ScatterPerm
 * writes a scratch permutation and K_RankInCell orders each cell by original
 * index into sort_perm before the gather. Six launches, still no allocation.
 *
 * cell_start / cell_end are the views offsets[:-1] / offsets[1:] of the cell
 * offsets table described next to GridParams in common.cuh. The scan writes
 * the inclusive sum into cell_end only; offsets[0] stays 0, so cell_start is
 * the exclusive sum. An empty cell has cell_start == cell_end.
 *
 * Pre-conditions:
 *   - histogram[] is all zero on entry to K_HashHistogram. It starts zeroed
 *     at allocation and K_ScatterPerm returns it to zero on every full sort,
 *     so no memset is needed.
 *   - block_sums[] holds ceil(num_cells / SCAN_TILE) entries (scratch, any
 *     content).
 *
 * GridParams read from __constant__ c_grid declared in common.cuh.
 */

#include "common.cuh"

// Uses calcGridCell() and spatialHash() from common.cuh.

// Scan kernels: SCAN_THREADS threads, SCAN_ITEMS slots each.
#define SCAN_THREADS 256
#define SCAN_ITEMS   4
#define SCAN_TILE    (SCAN_THREADS * SCAN_ITEMS)
#define SCAN_WARPS   (SCAN_THREADS / 32)

/* ======================================================================
 * K_HashHistogram -- per-particle hash (calcGridCell + spatialHash), stored
 * in hashes[] and counted in histogram[] (must be all zero on entry).
 * ====================================================================== */
extern "C" __global__
void K_HashHistogram(
    const uint     numParticles,
    const float4*  __restrict__ positions,
    uint*          __restrict__ hashes,
    uint*          __restrict__ histogram     // [num_cells], zero on entry
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    const float4 p4 = positions[idx];
    const uint hash = spatialHash(calcGridCell(make_float3(p4.x, p4.y, p4.z)));
    hashes[idx] = hash;
    atomicAdd(&histogram[hash], 1u);
}

/* Block-wide sum of one uint per thread (SCAN_THREADS threads). */
__device__ __forceinline__ uint blockSum(uint v, uint* warp_tmp) {
    const uint lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) warp_tmp[warp] = v;
    __syncthreads();
    uint t = 0;
    #pragma unroll
    for (int w = 0; w < SCAN_WARPS; ++w) t += warp_tmp[w];
    __syncthreads();
    return t;
}

/* ======================================================================
 * K_ScanReduce -- block_sums[b] = sum of histogram[b*SCAN_TILE .. +SCAN_TILE).
 * Grid: ceil(numCells / SCAN_TILE) blocks of SCAN_THREADS.
 * ====================================================================== */
extern "C" __global__ __launch_bounds__(SCAN_THREADS)
void K_ScanReduce(
    const uint   numCells,
    const uint*  __restrict__ histogram,
    uint*        __restrict__ block_sums
) {
    __shared__ uint warp_tmp[SCAN_WARPS];
    const uint base = blockIdx.x * SCAN_TILE;
    uint v = 0;
    #pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) {
        const uint i = base + k * SCAN_THREADS + threadIdx.x;
        if (i < numCells) v += histogram[i];
    }
    v = blockSum(v, warp_tmp);
    if (threadIdx.x == 0) block_sums[blockIdx.x] = v;
}

/* ======================================================================
 * K_ScanDownsweep -- cell_end[i] = sum(histogram[0..i]) (inclusive).
 *
 * Each block adds up block_sums[0 .. blockIdx.x) for its tile offset, then
 * scans its SCAN_TILE slots in shared memory. Loads and stores go through
 * shared memory so both are coalesced. cell_end is offsets + 1, so it is
 * not 16-byte aligned and is written one uint at a time.
 * ====================================================================== */
extern "C" __global__ __launch_bounds__(SCAN_THREADS)
void K_ScanDownsweep(
    const uint   numCells,
    const uint*  __restrict__ histogram,
    const uint*  __restrict__ block_sums,
    uint*        __restrict__ cell_end        // [numCells] = offsets + 1
) {
    __shared__ uint tile[SCAN_TILE];
    __shared__ uint warp_tmp[SCAN_WARPS];
    const uint tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint base = blockIdx.x * SCAN_TILE;

    // Offset of this tile: sum of the earlier tiles' totals.
    uint pre = 0;
    for (uint j = tid; j < blockIdx.x; j += SCAN_THREADS) pre += block_sums[j];
    pre = blockSum(pre, warp_tmp);

    #pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) {
        const uint i = base + k * SCAN_THREADS + tid;
        tile[k * SCAN_THREADS + tid] = (i < numCells) ? histogram[i] : 0u;
    }
    __syncthreads();

    // Thread tid owns tile[tid*SCAN_ITEMS .. +SCAN_ITEMS).
    uint v[SCAN_ITEMS];
    uint run = 0;
    #pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) { run += tile[tid * SCAN_ITEMS + k]; v[k] = run; }

    // Exclusive scan of the per-thread totals across the block.
    uint incl = run;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint n = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= (uint)o) incl += n;
    }
    if (lane == 31) warp_tmp[warp] = incl;
    __syncthreads();
    uint warp_pre = 0;
    #pragma unroll
    for (int w = 0; w < SCAN_WARPS; ++w) warp_pre += (w < (int)warp) ? warp_tmp[w] : 0u;
    const uint thread_pre = pre + warp_pre + incl - run;

    #pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) tile[tid * SCAN_ITEMS + k] = thread_pre + v[k];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) {
        const uint i = base + k * SCAN_THREADS + tid;
        if (i < numCells) cell_end[i] = tile[k * SCAN_THREADS + tid];
    }
}

/* ======================================================================
 * K_ScatterPerm -- claim each particle's sorted slot and write sort_perm.
 *
 * k = atomicSub(&histogram[hash], 1) runs from count down to 1 within a
 * cell, so sorted_idx = cell_end[hash] - k covers cell_start .. cell_end - 1
 * exactly once. Within-cell order is non-deterministic (acceptable for SPH);
 * K_RankInCell makes it deterministic for tests.
 * After the kernel every histogram slot is back at zero, which is the
 * pre-condition of the next K_HashHistogram. Nothing reads histogram[]
 * after this kernel.
 * ====================================================================== */
extern "C" __global__
void K_ScatterPerm(
    const uint     numParticles,
    const uint*   __restrict__ hashes,
    const uint*   __restrict__ cell_end,       // [num_cells] inclusive prefix sum
    uint*         __restrict__ histogram,      // [num_cells] counts, drained to 0
    uint*         __restrict__ sort_perm_out   // [N] sorted idx -> original idx
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    const uint hash = hashes[idx];
    const uint k = atomicSub(&histogram[hash], 1u);
    sort_perm_out[cell_end[hash] - k] = idx;
}

/* ======================================================================
 * K_RankInCell -- deterministic within-cell order (test / bisect mode,
 * bd-r5-review-n28.200).
 *
 * K_ScatterPerm leaves each cell's slots in atomic order, which changes run
 * to run. This kernel reorders every cell by original particle index:
 * sorted slot s holds v = perm_in[s] in cell hash = hashes[v]; its rank is
 * the number of indices in that cell smaller than v, so
 * perm_out[cell_start + rank] = v is a stable counting sort. Indices are
 * unique, so the ranks within a cell are a permutation of 0 .. count - 1.
 *
 * Cost is one pass over the cell per particle (sum of count^2), so it is
 * meant for tests only; production leaves it off. perm_in and perm_out must
 * not alias.
 * ====================================================================== */
extern "C" __global__
void K_RankInCell(
    const uint     numParticles,
    const uint*   __restrict__ hashes,
    const uint*   __restrict__ cell_start,     // [num_cells] = offsets
    const uint*   __restrict__ cell_end,       // [num_cells] = offsets + 1
    const uint*   __restrict__ perm_in,        // [N] atomic order from K_ScatterPerm
    uint*         __restrict__ perm_out        // [N] ordered by index in each cell
) {
    const uint s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= numParticles) return;

    const uint v = perm_in[s];
    const uint hash = hashes[v];
    const uint start = cell_start[hash];
    const uint end = cell_end[hash];
    uint rank = 0;
    for (uint j = start; j < end; ++j) rank += (perm_in[j] < v) ? 1u : 0u;
    perm_out[start + rank] = v;
}

/* ======================================================================
 * K_GatherReorder -- re-scatter unsorted data to sorted order using
 * existing sort_perm (from previous frame's counting sort).
 *
 * Runs as the last stage of every full sort (after K_ScatterPerm) and alone
 * for "grid reuse" when particles moved less than 0.25*h since the last sort
 * (cell_start/cell_end are then reused from that sort).
 * anchor_out is given on a full sort only. A grid-reuse gather must pass
 * NULL: the anchor holds the positions of the last full sort, and the
 * finalize kernels measure drift against it.
 * Indexed gather: coalesced writes, scattered reads.
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
    float*        __restrict__ kappa_v_out,          // DFSPH div warm-start (PERF-008), NULL = skip
    // FP16 velocity output (OPT-4.3)
    void*         __restrict__ velocity_h_out,      // NULL = skip
    // Fixed-point temperature output (PERF-011)
    void*         __restrict__ temperature_h_out,   // uint16 T*8 (0.125 K), 2 bytes per particle (NULL = skip)
    // Grid-reuse drift anchor: sort-time position in sorted order
    float4*       __restrict__ anchor_out           // full sort only; NULL = skip
) {
    const uint sorted_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_idx >= numParticles) return;

    const uint orig_idx = sort_perm[sorted_idx];

    // Simple indexed gather: coalesced writes, scattered reads through __ldg
    const float4 pos4 = __ldg(&position_in[orig_idx]);
    position_out[sorted_idx]      = pos4;
    if (anchor_out) anchor_out[sorted_idx] = pos4;
    float4 vel = __ldg(&velocity_in[orig_idx]);
    velocity_out[sorted_idx]      = vel;
    const float m = __ldg(&mass_in[orig_idx]);
    mass_out[sorted_idx]          = m;
    packed_info_out[sorted_idx]   = __ldg(&packed_info_in[orig_idx]);
    float temp = __ldg(&temperature_in[orig_idx]);
    temperature_out[sorted_idx]   = temp;
    health_out[sorted_idx]        = __ldg(&health_in[orig_idx]);
    lifetime_out[sorted_idx]      = __ldg(&lifetime_in[orig_idx]);
    sleep_counter_out[sorted_idx] = __ldg(&sleep_counter_in[orig_idx]);
    // DFSPH-only warm-start state: NULL when the active solver never reads it (PBF)
    if (kappa_out) kappa_out[sorted_idx] = __ldg(&kappa_in[orig_idx]);
    if (kappa_v_out) kappa_v_out[sorted_idx] = __ldg(&kappa_v_in[orig_idx]);

    // FP16 velocity copy (OPT-4.3); NULL = skip (PBF).
    // .w carries mass / MASS_H_UNIT (bd-r5-review-n28.161).
    if (velocity_h_out)
        store_half4((uint2*)velocity_h_out + sorted_idx,
                    make_float4(vel.x, vel.y, vel.z, __fdiv_rn(m, MASS_H_UNIT)));

    // Fixed-point temperature (0.125 K) copy (PERF-011)
    if (temperature_h_out) store_t16((unsigned short*)temperature_h_out + sorted_idx, temp);
}
