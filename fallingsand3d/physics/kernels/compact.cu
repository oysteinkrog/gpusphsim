/*
 * compact.cu -- stream compaction of the unsorted particle arrays.
 *
 * World.compact() moves every alive particle (material_id != 0) of
 * [0, n) to [0, num_alive), keeping their order. The host side builds
 * scan[] = inclusive prefix sum of the alive mask (cupy.cumsum), then one
 * K_Compact launch writes each alive particle's fields to slot scan[i] - 1
 * of the sorted_* scratch arrays. The caller then swaps the Python
 * references, so the scratch arrays become the primary arrays. Each field
 * is read once and written once, and there is no host readback.
 *
 * Fields are copied as raw values, so the result is bit-exact.
 * Standalone: uses no constant memory and no common.cuh.
 */

extern "C" __global__
void K_Compact(
    const unsigned int n,
    const unsigned int* __restrict__ scan,        // inclusive prefix sum of alive mask
    const unsigned int* __restrict__ packed_info,
    const float4* __restrict__ position,
    const float4* __restrict__ velocity,
    const float4* __restrict__ veleval,
    const float4* __restrict__ sph_force,
    const float*  __restrict__ density,
    const float*  __restrict__ mass,
    const float*  __restrict__ temperature,
    const float*  __restrict__ health,
    const float*  __restrict__ lifetime,
    const float*  __restrict__ shear_rate,
    const float*  __restrict__ exposure_heat,
    const float*  __restrict__ exposure_corrode,
    const unsigned char* __restrict__ sleep_counter,
    const float*  __restrict__ kappa,
    const float*  __restrict__ kappa_v,
    unsigned int* __restrict__ out_packed_info,
    float4* __restrict__ out_position,
    float4* __restrict__ out_velocity,
    float4* __restrict__ out_veleval,
    float4* __restrict__ out_sph_force,
    float*  __restrict__ out_density,
    float*  __restrict__ out_mass,
    float*  __restrict__ out_temperature,
    float*  __restrict__ out_health,
    float*  __restrict__ out_lifetime,
    float*  __restrict__ out_shear_rate,
    float*  __restrict__ out_exposure_heat,
    float*  __restrict__ out_exposure_corrode,
    unsigned char* __restrict__ out_sleep_counter,
    float*  __restrict__ out_kappa,
    float*  __restrict__ out_kappa_v
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int p = packed_info[i];
    if ((p & 0xFFu) == 0u) return;   // DEAD: dropped

    const unsigned int d = scan[i] - 1u;

    out_packed_info[d]      = p;
    out_position[d]         = position[i];
    out_velocity[d]         = velocity[i];
    out_veleval[d]          = veleval[i];
    out_sph_force[d]        = sph_force[i];
    out_density[d]          = density[i];
    out_mass[d]             = mass[i];
    out_temperature[d]      = temperature[i];
    out_health[d]           = health[i];
    out_lifetime[d]         = lifetime[i];
    out_shear_rate[d]       = shear_rate[i];
    out_exposure_heat[d]    = exposure_heat[i];
    out_exposure_corrode[d] = exposure_corrode[i];
    out_sleep_counter[d]    = sleep_counter[i];
    out_kappa[d]            = kappa[i];
    out_kappa_v[d]          = kappa_v[i];
}
