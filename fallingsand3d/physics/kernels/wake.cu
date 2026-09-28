/*
 * wake.cu -- Two-phase wake propagation kernels.
 *
 * Phase 1 (K_MarkWakeCells): just-woke and fast awake particles write
 * this substep's stamp into their OWN grid cell (one plain store). The
 * solvers do this in K_Integrate, K_PBF_Finalize and K_DFSPH_Finalize via
 * wake_mark_own_cell() and skip this kernel (bd-r5-review-n28.166, n28.167).
 * It stays as the reference path for direct callers and the unit tests.
 *
 * Phase 1b (K_MarkWakeCellsRigid): moving rigid bodies stamp every cell
 * their bounding cube covers.
 *
 * Phase 2+3 (K_WakeSleepersAndClearJustWoke): a sleeping particle wakes
 * (CLEAR_SLEEPING, sleep_counter = 0) when any of its 27 cells (own cell
 * and 26 neighbours) holds this substep's stamp. JUST_WOKE is cleared on
 * every particle.
 *
 * Marking one cell and probing 27 is the same predicate as marking 27 and
 * probing one, because the 3x3x3 neighbourhood is symmetric. It replaces
 * 27 scattered atomics per marker with one store (bd-r5-review-n28.141).
 *
 * Generation stamp: the table is never cleared per substep. The stamp is
 * wake_stamp(frame_ptr) = *frame_ptr + 1 (never 0, the value of a fresh
 * table). frame_ptr is the per-substep counter Simulation fills before each
 * substep, so a mark left from an earlier substep holds an older stamp and
 * no longer matches. A NULL frame_ptr gives stamp 1, which needs the caller
 * to clear the table first (wake.py does that).
 *
 * The kernels run on whichever arrays the solver's last kernel wrote: the
 * unsorted arrays after a scatter, or the sorted buffers after an in-place
 * WCSPH integrate. They are order-free, but the sleeper probe's cost is not.
 * At 1M settled sand it takes 48 us in sorted (cell) order and 237 us in
 * unsorted spawn order, because the 27 flag reads per thread scatter across
 * L2 (bd-r5-review-n28.209, follow-up n28.217). The probe stays out of
 * K_ScatterReorder and K_GatherReorder: they would have to read-modify-write
 * the unsorted packed_info and check the previous substep's stamp.
 *
 * Constant memory used:
 *   c_grid -- GridParams from common.cuh (for position -> cell mapping)
 */

#include "sph_shared.cuh"   // common.cuh + V_WAKE_SQ

/* ======================================================================
 * Grid cell computation (inlined, same as counting_sort.cu K_CalcHashCS)
 * ====================================================================== */

// Uses calcGridCell() and spatialHash() from common.cuh.

// wake_stamp() lives in sph_shared.cuh, shared with the solver finalize kernels.

/* ======================================================================
 * K_MarkWakeCells -- Phase 1: stamp the own cell of each just-woke or
 *                   fast awake particle.
 * ====================================================================== */

extern "C" __global__
void K_MarkWakeCells(
    uint        numParticles,
    const float4* __restrict__ position,       // unsorted positions
    const float4* __restrict__ velocity,       // unsorted velocities
    const uint*   __restrict__ packed_info,     // unsorted packed_info
    uint*         __restrict__ cell_wake_flags, // table_size uint32 stamps
    const uint*   __restrict__ frame_ptr        // per-substep counter (or NULL)
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];

    // Mark cells for two categories of particles:
    // 1. Particles that just woke up this frame (wake cascade)
    // 2. Active non-sleeping particles moving fast enough (so flowing water
    //    wakes sleeping sand in adjacent cells). STATIC never moves. GAS is
    //    excluded because step1/step2 skip all forces between GAS and
    //    non-GAS, so a woken pile under a smoke plume would get nothing,
    //    re-sleep, and be woken again every few substeps.
    bool should_mark = HAS_JUST_WOKE(pi);
    int beh = GET_BEHAVIOR(pi);
    if (!should_mark && !IS_SLEEPING(pi) && beh != STATIC && beh != GAS) {
        float4 v = velocity[i];
        float v_sq = v.x*v.x + v.y*v.y + v.z*v.z;
        should_mark = (v_sq > V_WAKE_SQ);
    }
    if (!should_mark) return;

    // Compute this particle's grid cell
    float4 pos4 = position[i];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
    int3 cell = calcGridCell(pos);

    // Own cell only; sleepers probe their 27 cells. Racing stores all
    // write the same stamp, so no atomic is needed.
    cell_wake_flags[spatialHash(cell)] = wake_stamp(frame_ptr);
}

/* ======================================================================
 * K_MarkWakeCellsRigid -- Phase 1b: moving rigid bodies mark every cell
 *                        they can touch (bd-r5-review-n28.87).
 *
 * Rigid boundary particles are STATIC, so K_MarkWakeCells skips them, and
 * a sleeping particle returns from step2 before the rigid coupling. Without
 * this kernel a body dropped on a sleeping pile gets no force and falls
 * through it. One block per body. A body marks when the fastest point on
 * it can exceed V_WAKE: |v| + |omega| * r > V_WAKE, where r = |half_extents|
 * bounds the body (box of any rotation, or sphere). It then stamps every
 * cell of the axis-aligned cube [c - r, c + r]. Sleepers probe their 27
 * cells, so a sleeper up to one cell (~h) past the cube wakes. A body at
 * rest marks nothing. Same stamp as K_MarkWakeCells.
 * ====================================================================== */

extern "C" __global__
void K_MarkWakeCellsRigid(
    const RigidBody* __restrict__ bodies,        // [num_bodies] device body state
    int                            num_bodies,
    uint*            __restrict__ cell_wake_flags,
    const uint*      __restrict__ frame_ptr      // per-substep counter (or NULL)
) {
    int b = blockIdx.x;
    if (b >= num_bodies) return;

    RigidBody body = bodies[b];
    float3 he = make_float3(body.half_extents.x, body.half_extents.y, body.half_extents.z);
    float r = sqrtf(he.x*he.x + he.y*he.y + he.z*he.z);
    float3 v = make_float3(body.lin_vel.x, body.lin_vel.y, body.lin_vel.z);
    float3 w = make_float3(body.ang_vel.x, body.ang_vel.y, body.ang_vel.z);
    float speed = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z)
                + sqrtf(w.x*w.x + w.y*w.y + w.z*w.z) * r;
    if (!(speed > V_WAKE)) return;

    // The sleeper's 27-cell probe adds the one-cell (~h) margin.
    float pad = r;
    uint stamp = wake_stamp(frame_ptr);
    float3 c = make_float3(body.position.x, body.position.y, body.position.z);
    int3 lo = calcGridCell(make_float3(c.x - pad, c.y - pad, c.z - pad));
    int3 hi = calcGridCell(make_float3(c.x + pad, c.y + pad, c.z + pad));
    int nx = hi.x - lo.x + 1;
    int ny = hi.y - lo.y + 1;
    int nz = hi.z - lo.z + 1;
    int total = nx * ny * nz;

    for (int k = threadIdx.x; k < total; k += blockDim.x) {
        int ix = k % nx;
        int iy = (k / nx) % ny;
        int iz = k / (nx * ny);
        cell_wake_flags[spatialHash(lo.x + ix, lo.y + iy, lo.z + iz)] = stamp;
    }
}

/* ======================================================================
 * K_WakeSleepersAndClearJustWoke -- Phase 2+3 fused:
 *   - Wake sleeping particles with a stamped cell among their 27 cells
 *   - Clear JUST_WOKE flag from all particles
 * Both iterate all particles and modify packed_info; no data dependency.
 * ====================================================================== */

extern "C" __global__
void K_WakeSleepersAndClearJustWoke(
    uint          numParticles,
    const float4* __restrict__ position,       // unsorted positions
    uint*         __restrict__ packed_info,     // unsorted packed_info (read+write)
    unsigned char* __restrict__ sleep_counter,  // unsorted sleep counter (write)
    const uint*   __restrict__ cell_wake_flags, // table_size uint32 stamps
    const uint*   __restrict__ frame_ptr        // per-substep counter (or NULL)
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    uint new_pi = pi;

    // Wake a sleeper when any of its 27 cells carries this substep's stamp
    if (IS_SLEEPING(pi)) {
        float4 pos4 = position[i];
        float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
        int3 cell = calcGridCell(pos);
        uint stamp = wake_stamp(frame_ptr);
        bool hit = false;
        for (int dz = -1; dz <= 1 && !hit; dz++) {
            for (int dy = -1; dy <= 1 && !hit; dy++) {
                for (int dx = -1; dx <= 1 && !hit; dx++) {
                    uint hash = spatialHash(cell.x + dx, cell.y + dy, cell.z + dz);
                    hit = (__ldg(&cell_wake_flags[hash]) == stamp);
                }
            }
        }
        if (hit) {
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
