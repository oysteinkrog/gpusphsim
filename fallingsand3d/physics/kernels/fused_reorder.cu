/*
 * fused_reorder.cu -- Single-pass gather kernel for ALL SoA particle arrays.
 *
 * After argsort produces sorted_index (sorted->original mapping), this kernel
 * reads sorted_index once per thread and gathers all SoA arrays from unsorted
 * to sorted order in a single pass. This is far more bandwidth-efficient than
 * N separate CuPy fancy-indexing calls since sorted_index is read only once.
 *
 * Included from common.cuh for constant memory declarations (unused here but
 * keeps the include pattern consistent).
 */

#include "common.cuh"

extern "C"
__global__ void K_FusedReorder(
    const uint    num_particles,
    const uint*  __restrict__ sorted_index,    // sorted->original mapping
    // Unsorted inputs (read)
    const float4* __restrict__ position_in,
    const float4* __restrict__ velocity_in,
    const float4* __restrict__ veleval_in,
    const float*  __restrict__ mass_in,
    const uint*   __restrict__ packed_info_in,
    const float*  __restrict__ temperature_in,
    const float*  __restrict__ health_in,
    const float*  __restrict__ lifetime_in,
    const float4* __restrict__ color_in,
    const unsigned char* __restrict__ sleep_counter_in,
    const float*  __restrict__ shear_rate_in,
    // Sorted outputs (write)
    float4*       __restrict__ position_out,
    float4*       __restrict__ velocity_out,
    float4*       __restrict__ veleval_out,
    float*        __restrict__ mass_out,
    uint*         __restrict__ packed_info_out,
    float*        __restrict__ temperature_out,
    float*        __restrict__ health_out,
    float*        __restrict__ lifetime_out,
    float4*       __restrict__ color_out,
    unsigned char* __restrict__ sleep_counter_out,
    float*        __restrict__ shear_rate_out
)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // One read of sorted_index per thread
    const uint src = sorted_index[idx];

    // Gather all arrays from unsorted[src] -> sorted[idx]
    position_out[idx]      = __ldg(&position_in[src]);
    velocity_out[idx]      = __ldg(&velocity_in[src]);
    veleval_out[idx]       = __ldg(&veleval_in[src]);
    mass_out[idx]          = __ldg(&mass_in[src]);
    packed_info_out[idx]   = __ldg(&packed_info_in[src]);
    temperature_out[idx]   = __ldg(&temperature_in[src]);
    health_out[idx]        = __ldg(&health_in[src]);
    lifetime_out[idx]      = __ldg(&lifetime_in[src]);
    color_out[idx]         = __ldg(&color_in[src]);
    sleep_counter_out[idx] = __ldg(&sleep_counter_in[src]);
    shear_rate_out[idx]    = __ldg(&shear_rate_in[src]);
}
