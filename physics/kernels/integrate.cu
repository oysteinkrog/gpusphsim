/**
 * K_Integrate -- Leapfrog integration with wall boundaries and coloring.
 *
 * Per-particle:
 *   1. Read sorted position, velocity, veleval, sph_force, density
 *   2. Add gravity as external force
 *   3. Apply wall boundary penalty forces (6 walls from grid_min/max)
 *   4. Clamp force magnitude (velocity limit)
 *   5. Leapfrog integration: vnext = vel + force*dt, vel_eval = (vel+vnext)/2
 *   6. Position update: pos += vnext * dt
 *   7. Write back to UNSORTED arrays using sort_indexes permutation
 *   8. Compute velocity-based color
 *
 * Ported from SPHSimLib/K_SimpleSPH_Integrate.inl.
 */

#include "common.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
// Constant memory
// ---------------------------------------------------------------------------
__constant__ GridParams c_grid;

struct IntegrateParams {
    float delta_time;           // dt
    float gravity;              // -9.8 (negative = downward)
    float boundary_stiffness;   // wall penalty spring constant
    float boundary_dampening;   // wall penalty damping
    float boundary_distance;    // distance from wall where force activates
    float velocity_limit;       // max force magnitude
};

__constant__ IntegrateParams c_integrate;

// ---------------------------------------------------------------------------
// Wall boundary forces (ported from K_Boundaries_Walls.inl)
//
// No scale_to_simulation in the Python port -- everything in world space.
// ---------------------------------------------------------------------------

__device__ inline float3 calculateWallForce(
    float3 pos, float3 vel,
    float3 grid_min, float3 grid_max,
    float boundary_dist, float stiffness, float dampening
) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float diff;
    float EPSILON = 0.00001f;

    // Y walls
    diff = boundary_dist - (pos.y - grid_min.y);
    if (diff > EPSILON)
        force.y += stiffness * diff - dampening * vel.y;

    diff = boundary_dist - (grid_max.y - pos.y);
    if (diff > EPSILON)
        force.y -= stiffness * diff + dampening * vel.y;

    // Z walls
    diff = boundary_dist - (pos.z - grid_min.z);
    if (diff > EPSILON)
        force.z += stiffness * diff - dampening * vel.z;

    diff = boundary_dist - (grid_max.z - pos.z);
    if (diff > EPSILON)
        force.z -= stiffness * diff + dampening * vel.z;

    // X walls
    diff = boundary_dist - (pos.x - grid_min.x);
    if (diff > EPSILON)
        force.x += stiffness * diff - dampening * vel.x;

    diff = boundary_dist - (grid_max.x - pos.x);
    if (diff > EPSILON)
        force.x -= stiffness * diff + dampening * vel.x;

    return force;
}

// ---------------------------------------------------------------------------
// Velocity-based coloring (HSV blue-to-red)
// ---------------------------------------------------------------------------

__device__ inline float3 HSVtoRGB(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float3 rgb;

    if (h < 60.0f)       rgb = make_float3(c, x, 0);
    else if (h < 120.0f) rgb = make_float3(x, c, 0);
    else if (h < 180.0f) rgb = make_float3(0, c, x);
    else if (h < 240.0f) rgb = make_float3(0, x, c);
    else if (h < 300.0f) rgb = make_float3(x, 0, c);
    else                  rgb = make_float3(c, 0, x);

    return make_float3(rgb.x + m, rgb.y + m, rgb.z + m);
}

__device__ inline float3 velocityColor(float3 vel) {
    float speed = fabsf(vel.x) + fabsf(vel.y) + fabsf(vel.z);
    float scalar = fminf(speed / 5.0f, 1.0f);  // normalize speed
    // Map 0..1 to hue 240..0 (blue to red)
    float h = (1.0f - scalar) * 240.0f;
    return HSVtoRGB(h, 0.8f, 1.0f);
}

// ---------------------------------------------------------------------------
// K_Integrate kernel
// ---------------------------------------------------------------------------

extern "C" __global__
void K_Integrate(
    uint            numParticles,
    // Sorted arrays (read)
    const float4*   __restrict__ position_sorted,
    const float4*   __restrict__ velocity_sorted,
    const float4*   __restrict__ veleval_sorted,
    const float4*   __restrict__ sph_force_sorted,
    const float*    __restrict__ density_sorted,
    const int*      __restrict__ behavior_class_sorted,
    const uint*     __restrict__ sort_indexes,      // sort_indexes[sorted] = original
    // Unsorted arrays (write)
    float4*         __restrict__ position_out,
    float4*         __restrict__ velocity_out,
    float4*         __restrict__ veleval_out,
    float4*         __restrict__ color_out
) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    // --- Read from sorted arrays ---
    float4 pos4 = position_sorted[index];
    float3 pos = make_float3(pos4.x, pos4.y, pos4.z);

    float4 vel4 = velocity_sorted[index];
    float3 vel = make_float3(vel4.x, vel4.y, vel4.z);

    float4 veleval4 = veleval_sorted[index];
    float3 vel_eval = make_float3(veleval4.x, veleval4.y, veleval4.z);

    float4 sphf4 = sph_force_sorted[index];
    float3 sph_force = make_float3(sphf4.x, sphf4.y, sphf4.z);

    float dt = c_integrate.delta_time;

    // --- External forces ---
    float3 external_force = make_float3(0.0f, 0.0f, 0.0f);
    external_force.y += c_integrate.gravity;  // gravity (negative value)

    // Wall boundary forces
    external_force = make_float3(
        external_force.x + 0.0f,
        external_force.y + 0.0f,
        external_force.z + 0.0f
    );
    float3 wall_force = calculateWallForce(
        pos, vel_eval,
        c_grid.grid_min, c_grid.grid_max,
        c_integrate.boundary_distance,
        c_integrate.boundary_stiffness,
        c_integrate.boundary_dampening
    );
    external_force.x += wall_force.x;
    external_force.y += wall_force.y;
    external_force.z += wall_force.z;

    float3 force = make_float3(
        sph_force.x + external_force.x,
        sph_force.y + external_force.y,
        sph_force.z + external_force.z
    );

    // --- Velocity limit (clamp force magnitude) ---
    float vlimit = c_integrate.velocity_limit;
    float vlimit_sq = vlimit * vlimit;
    float speed_sq = force.x * force.x + force.y * force.y + force.z * force.z;
    if (speed_sq > vlimit_sq) {
        float scale = vlimit / sqrtf(speed_sq);
        force.x *= scale;
        force.y *= scale;
        force.z *= scale;
    }

    // --- Leapfrog integration ---
    // v(t+1/2) = v(t-1/2) + a(t) * dt
    float3 vnext = make_float3(
        vel.x + force.x * dt,
        vel.y + force.y * dt,
        vel.z + force.z * dt
    );
    // vel_eval = (v(t-1/2) + v(t+1/2)) / 2
    vel_eval = make_float3(
        (vel.x + vnext.x) * 0.5f,
        (vel.y + vnext.y) * 0.5f,
        (vel.z + vnext.z) * 0.5f
    );
    vel = vnext;

    // --- Position update ---
    pos.x += vnext.x * dt;
    pos.y += vnext.y * dt;
    pos.z += vnext.z * dt;

    // --- Clamp position to grid bounds (safety) ---
    float margin = 0.001f;
    pos.x = fmaxf(c_grid.grid_min.x + margin, fminf(pos.x, c_grid.grid_max.x - margin));
    pos.y = fmaxf(c_grid.grid_min.y + margin, fminf(pos.y, c_grid.grid_max.y - margin));
    pos.z = fmaxf(c_grid.grid_min.z + margin, fminf(pos.z, c_grid.grid_max.z - margin));

    // --- Write back to UNSORTED arrays ---
    uint originalIndex = sort_indexes[index];

    position_out[originalIndex] = make_float4(pos.x, pos.y, pos.z, pos4.w);
    velocity_out[originalIndex] = make_float4(vel.x, vel.y, vel.z, 0.0f);
    veleval_out[originalIndex]  = make_float4(vel_eval.x, vel_eval.y, vel_eval.z, 0.0f);

    // --- Velocity-based coloring ---
    float3 color = velocityColor(vnext);
    // Encode behavior_class in w channel for rendering hints
    int bclass = behavior_class_sorted[index];
    // Sand/granular gets a brown tint
    if (bclass == 2) {  // GRANULAR
        color = make_float3(
            0.6f + 0.4f * color.x,
            0.4f + 0.3f * color.y,
            0.2f + 0.2f * color.z
        );
    }
    color_out[originalIndex] = make_float4(color.x, color.y, color.z, 1.0f);
}
