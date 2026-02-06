#ifndef PHYSICS_COMMON_CUH
#define PHYSICS_COMMON_CUH

// CUDA 12 compatibility: define uint if not already defined
#ifndef uint
typedef unsigned int uint;
#endif

// Grid parameters uploaded to __constant__ memory before kernel launch.
// Layout mirrors SPHSimLib/UniformGrid.cuh::GridParams.
struct GridParams {
    float3 grid_size;   // grid_max - grid_min
    float3 grid_min;    // world-space minimum corner
    float3 grid_max;    // world-space maximum corner
    float3 grid_res;    // number of cells per axis (as float for calcGridHash)
    float3 grid_delta;  // 1 / cell_size  (= grid_res / grid_size)
};

#endif // PHYSICS_COMMON_CUH
