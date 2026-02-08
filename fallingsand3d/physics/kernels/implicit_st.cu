/*
 * implicit_st.cu -- Implicit surface tension via iterative velocity smoothing.
 *
 * Quality mode for < 100K particles. Applies Jacobi iterations of surface
 * tension forces to FLUID surface particles, producing smoother, more
 * cohesive surfaces than the explicit Akinci approach in step2.cu.
 *
 * Algorithm:
 *   For each Jacobi iteration:
 *     For each FLUID particle i with neighbor_count < surface_threshold:
 *       delta_v_i = sigma * SUM_j (m_j/rho_j) * (v_j - v_i) * W(r_ij) * w_surface
 *       v_new_i = v_old_i + delta_v_i
 *     Non-surface and non-FLUID particles: v_new_i = v_old_i (pass through)
 *
 * The surface weight w_surface increases with neighbor deficiency:
 *   w_surface = 1.0 - neighbor_count / surface_threshold
 * So the most exposed particles (fewest neighbors) get the strongest smoothing.
 *
 * Uses ping-pong buffers: reads from vel_src, writes to vel_dst.
 * Caller swaps pointers between iterations.
 *
 * Constant memory: c_grid, c_sim, c_precalc, c_materials (from common.cuh)
 */

#include "common.cuh"

struct ISTParams {
    float sigma;              // surface tension strength
    float surface_threshold;  // neighbor count below which particle is "surface"
    int   num_iterations;     // Jacobi iterations (5-20)
    float padding;
};

__constant__ ISTParams c_ist;

/* SPH kernel (Poly6) for weighting */
__device__ inline float W_poly6_ist(float r_sq, float h_sq) {
    float diff = h_sq - r_sq;
    return c_precalc.poly6_coeff * diff * diff * diff;
}

/* ======================================================================
 * K_IST_Iterate -- One Jacobi iteration of implicit surface tension.
 *
 * For FLUID surface particles: smooth velocity toward neighbor average.
 * Non-surface particles and non-FLUID: pass through unchanged.
 *
 * Uses the surface normal buffer's .w component (neighbor_count) from step1.
 * ====================================================================== */

extern "C" __global__ __launch_bounds__(256, 4)
void K_IST_Iterate(
    uint            numParticles,
    const float4*   __restrict__ vel_src,       // velocity to read from
    float4*         __restrict__ vel_dst,       // velocity to write to
    const float4*   __restrict__ position,      // sorted positions
    const float*    __restrict__ density,       // sorted density
    const float*    __restrict__ mass,          // sorted mass
    const uint*     __restrict__ packed_info,   // sorted packed_info
    const float4*   __restrict__ normal,        // sorted normal (.w = neighbor_count)
    const uint*     __restrict__ cell_start,
    const uint*     __restrict__ cell_end
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint pi = packed_info[i];
    int behavior = GET_BEHAVIOR(pi);

    // Only FLUID particles participate in surface tension
    if (behavior != FLUID || IS_SLEEPING(pi)) {
        vel_dst[i] = vel_src[i];
        return;
    }

    // Check if surface particle (fewer neighbors than threshold)
    float nc = __ldg(&normal[i]).w;  // neighbor count from step1
    float threshold = c_ist.surface_threshold;

    if (nc >= threshold) {
        // Interior particle: no surface tension correction
        vel_dst[i] = vel_src[i];
        return;
    }

    // Surface weight: stronger for more exposed particles
    float w_surface = 1.0f - nc / threshold;
    w_surface = fmaxf(w_surface, 0.0f);

    float4 pos4_i = position[i];
    float3 pos_i = make_float3(pos4_i.x, pos4_i.y, pos4_i.z);
    float4 vel4_i = vel_src[i];
    float3 vel_i = make_float3(vel4_i.x, vel4_i.y, vel4_i.z);
    float rho_i = density[i];

    float h = c_sim.smoothing_length;
    float h_sq = c_sim.smoothing_length_sq;
    float sigma = c_ist.sigma;

    // Accumulate velocity correction from neighbors
    float3 dv = make_float3(0.0f, 0.0f, 0.0f);

    int3 cell_i = calcGridCell(pos_i);

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
                if (start == 0xFFFFFFFFu) continue;
                uint end_idx = cell_end[hash_c];

                for (uint j = start; j < end_idx; j++) {
                    if (j == i) continue;

                    float4 pos4_j = __ldg(&position[j]);
                    uint pi_j = __ldg(&packed_info[j]);
                    float m_j = __ldg(&mass[j]);

                    // Skip non-FLUID neighbors
                    if (GET_BEHAVIOR(pi_j) != FLUID) continue;

                    float3 r = make_float3(
                        pos_i.x - pos4_j.x,
                        pos_i.y - pos4_j.y,
                        pos_i.z - pos4_j.z
                    );
                    float r_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                    if (r_sq >= h_sq) continue;

                    float4 vel4_j = __ldg(&vel_src[j]);
                    float rho_j = __ldg(&density[j]);

                    float w = W_poly6_ist(r_sq, h_sq);
                    float factor = (m_j / fmaxf(rho_j, 1.0f)) * w;

                    dv.x += (vel4_j.x - vel_i.x) * factor;
                    dv.y += (vel4_j.y - vel_i.y) * factor;
                    dv.z += (vel4_j.z - vel_i.z) * factor;
                }
            }
        }
    }

    // Apply correction: v_new = v_old + sigma * w_surface * dv
    float scale = sigma * w_surface;
    vel_dst[i] = make_float4(
        vel_i.x + scale * dv.x,
        vel_i.y + scale * dv.y,
        vel_i.z + scale * dv.z,
        0.0f
    );
}
