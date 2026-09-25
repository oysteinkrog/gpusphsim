/*
 * implicit_st.cu -- Surface velocity smoothing (XSPH-style, surface only).
 *
 * The file and struct keep the "IST" (implicit surface tension) name for
 * history, but this is not surface tension: there is no cohesion or
 * curvature term, so it cannot pull a surface inward or form droplets. It
 * only damps relative velocity between FLUID surface particles. The
 * curvature surface tension force is the explicit Akinci term in step2.cu.
 *
 * Quality mode for < 100K particles, WCSPH only.
 *
 * Algorithm (one Jacobi iteration per launch):
 *   For each FLUID particle i with neighbor_count < surface_threshold:
 *     dv_i = SUM_j (m_j/rho_j) * W(r_ij) * (v_j - v_i)
 *     S_i  = SUM_j (m_j/rho_j) * W(r_ij)
 *     a_i  = min(sigma * w_surface, 1 / S_i)
 *     v_new_i = v_old_i + a_i * dv_i
 *   Non-surface and non-FLUID particles: no write (see below)
 *
 * The clamp a_i * S_i <= 1 makes v_new_i a convex combination of v_i and
 * its neighbours' velocities, so the largest speed can never grow from one
 * iteration to the next, whatever sigma and the iteration count are.
 *
 * The surface weight w_surface increases with neighbor deficiency:
 *   w_surface = 1.0 - neighbor_count / surface_threshold
 * So the most exposed particles (fewest neighbors) get the strongest smoothing.
 *
 * Uses ping-pong buffers: reads from vel_src, writes to vel_dst.
 * Caller swaps pointers between iterations. The caller copies the input
 * velocity into the scratch buffer once before the first iteration, so
 * both buffers start equal. The kernel then writes only surface FLUID
 * particles: interior and non-FLUID entries hold the input value in both
 * buffers for the whole pass, and the surface set does not change between
 * iterations (neighbour count and packed_info are fixed during the pass).
 *
 * Constant memory: c_grid, c_sim, c_precalc, c_materials (from common.cuh)
 */

#include "common.cuh"

struct ISTParams {
    float sigma;              // smoothing strength (clamped per particle, see K_IST_Iterate)
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
 * K_IST_Iterate -- One Jacobi iteration of surface velocity smoothing.
 *
 * For FLUID surface particles: smooth velocity toward neighbor average,
 * with the weight clamped so sigma * w_surface * S_i <= 1.
 * Non-surface particles and non-FLUID: return without writing (vel_dst
 * already holds the same value, see the header).
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

    // Only FLUID particles are smoothed
    if (behavior != FLUID || IS_SLEEPING(pi)) return;

    // Check if surface particle (fewer neighbors than threshold)
    float nc = __ldg(&normal[i]).w;  // neighbor count from step1
    float threshold = c_ist.surface_threshold;

    if (nc >= threshold) return;  // interior particle: no smoothing

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

    // Accumulate velocity correction and total weight S from neighbors
    float3 dv = make_float3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    int3 cell_i = calcGridCell(pos_i);

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                uint hash_c = spatialHash(cell_i.x + dx, cell_i.y + dy, cell_i.z + dz);
                uint start = cell_start[hash_c];
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
                    float factor = (m_j / fmaxf(rho_j, RHO_EPSILON)) * w;

                    dv.x += (vel4_j.x - vel_i.x) * factor;
                    dv.y += (vel4_j.y - vel_i.y) * factor;
                    dv.z += (vel4_j.z - vel_i.z) * factor;
                    weight_sum += factor;
                }
            }
        }
    }

    // Apply correction: v_new = v_old + scale * dv, with scale * S <= 1 so
    // v_new stays a convex combination of v_i and the neighbour velocities
    // (Jacobi stability). S can exceed 1 when a surface particle's own
    // density is below its neighbours'.
    float scale = sigma * w_surface;
    if (scale * weight_sum > 1.0f) {
        scale = 1.0f / weight_sum;
    }
    vel_dst[i] = make_float4(
        vel_i.x + scale * dv.x,
        vel_i.y + scale * dv.y,
        vel_i.z + scale * dv.z,
        0.0f
    );
}
