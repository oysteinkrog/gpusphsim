/*
 * common.cuh -- Single source of truth for all GPU data layouts.
 *
 * All .cu kernel files include this header. It defines:
 *   - BehaviorClass enum
 *   - MaterialProps / Interaction structs + constant memory arrays
 *   - GridParams struct + constant memory
 *   - SimParams / PrecalcParams structs + constant memory
 *   - packed_info bitfield macros
 *   - Array type conventions (documented in comments)
 */

#ifndef FALLINGSAND3D_COMMON_CUH
#define FALLINGSAND3D_COMMON_CUH

typedef unsigned int uint;

/* ======================================================================
 * Behavior classes
 * ====================================================================== */

enum BehaviorClass {
    FLUID   = 0,
    GRANULAR = 1,
    GAS     = 2,
    STATIC  = 3
};

/* Minimum density for denominator guards (safe for GAS with rho0 ~ 0.2-0.6) */
#ifndef RHO_EPSILON
#define RHO_EPSILON 0.01f
#endif

/* ======================================================================
 * MaterialProps -- per-material constants uploaded to constant memory.
 * 18 fields x 4 bytes = 72 bytes (includes 1 pad field).
 * ====================================================================== */

struct MaterialProps {
    float rest_density;          // kg/m^3
    float eos_stiffness;         // Tait EOS k
    float eos_gamma;             // Tait EOS exponent (7 for liquid, 1 for gas)
    float base_viscosity;        // viscosity coefficient (WCSPH: kinematic-like with force_scale; DFSPH: dynamic mu/rho_i)
    float friction_coeff;        // mu_s for granular friction
    float cohesion;              // cohesion coefficient
    float buoyancy_extra;        // extra buoyancy factor for gas
    float thermal_conductivity;  // W/(m*K)
    float heat_capacity;         // J/(kg*K)
    float temp_melt;             // Kelvin
    float temp_boil;             // Kelvin
    float temp_ignite;           // Kelvin (0 = non-flammable)
    int   behavior_class;        // BehaviorClass enum value
    float color_r;               // default material color R
    float color_g;               // default material color G
    float color_b;               // default material color B
    float thermal_expansion;     // Boussinesq thermal expansion coefficient (1/K)
    float _pad0;                 // padding to 72 bytes (18 fields x 4)
};
// Static assert: sizeof(MaterialProps) must be 72 bytes

/* ======================================================================
 * Interaction -- per-pair (i,j) reaction parameters.
 * 2 fields x 4 bytes = 8 bytes.
 * ====================================================================== */

struct Interaction {
    float reaction_rate;   // reaction speed (0 = no reaction)
    float heat_exchange;   // heat transfer coefficient between pair
};

/* ======================================================================
 * Constant memory: materials and interactions
 * Total: 32*72 + 32*32*8 = 2304 + 8192 = 10496 bytes (well under 64 KB)
 * ====================================================================== */

__constant__ MaterialProps c_materials[32];
__constant__ Interaction   c_interactions[32][32];

/* ======================================================================
 * GridParams -- spatial hash grid parameters.
 *
 * Uses a spatial hash with fixed-size table (power of 2) instead of
 * dense arrays sized grid_res^3.  This enables arbitrarily large worlds
 * without memory blowup.  Hash collisions cause extra (harmless) distance
 * checks in neighbor loops but never miss neighbors.
 *
 * 40 bytes total (10 fields x 4 bytes).
 * ====================================================================== */

struct GridParams {
    float3 grid_min;    // world-space minimum corner (for pos -> cell)
    float3 grid_delta;  // 1 / cell_size per axis (~1/h)
    uint   table_size;  // hash table size (power of 2, e.g. 262144)
    uint   table_mask;  // table_size - 1 (for & masking)
};

__constant__ GridParams c_grid;

/* ======================================================================
 * Spatial hash helpers -- shared by ALL kernel files.
 *
 * Position -> cell:   int3 cell = calcGridCell(pos)
 * Cell -> hash:       uint h = spatialHash(cell)
 * Neighbor cell hash: uint h = spatialHash(make_int3(cx+dx, cy+dy, cz+dz))
 *
 * No bounds checking needed -- any integer cell coords produce a valid
 * hash in [0, table_size).  Particles are already boundary-clamped in
 * the integrate kernel.
 * ====================================================================== */

__device__ inline int3 calcGridCell(float3 p) {
    return make_int3(
        (int)floorf((p.x - c_grid.grid_min.x) * c_grid.grid_delta.x),
        (int)floorf((p.y - c_grid.grid_min.y) * c_grid.grid_delta.y),
        (int)floorf((p.z - c_grid.grid_min.z) * c_grid.grid_delta.z)
    );
}

__device__ inline uint spatialHash(int3 cell) {
    // Large primes for spatial hashing (widely used in GPU SPH literature)
    return ((uint)(cell.x * 73856093)
          ^ (uint)(cell.y * 19349669)
          ^ (uint)(cell.z * 83492791)) & c_grid.table_mask;
}

__device__ inline uint spatialHash(int cx, int cy, int cz) {
    return ((uint)(cx * 73856093)
          ^ (uint)(cy * 19349669)
          ^ (uint)(cz * 83492791)) & c_grid.table_mask;
}

/* ======================================================================
 * SimParams -- simulation-wide parameters.
 *
 * Uploaded once per frame (or when sim config changes).
 * ====================================================================== */

struct SimParams {
    float  smoothing_length;     // h
    float  smoothing_length_sq;  // h^2
    float  particle_mass;        // uniform mass per particle
    float  particle_spacing;     // rest spacing between particles
    float3 gravity;              // (0, -9.8, 0) typically
    float  dt;                   // simulation timestep
    float  restitution;          // wall bounce coefficient
    float  wall_friction;        // wall friction coefficient
    float3 world_min;            // simulation domain min
    float3 world_max;            // simulation domain max
    float  velocity_damping;     // spawn stabilization: 0.0=none, 0.8=heavy
    float  velocity_limit;       // CFL-derived max velocity (replaces VELOCITY_LIMIT define)
    int    dye_enabled;          // 1 when any particle has dye, 0 to skip dye diffusion (PERF-005)
};

__constant__ SimParams c_sim;

/* ======================================================================
 * PrecalcParams -- kernel coefficients precomputed from smoothing_length.
 *
 * These depend on h and are recomputed when h changes. Stored separately
 * from SimParams to keep the intent clear.
 *
 * Sign convention: pressure_precalc is POSITIVE (+45/(pi*h^6)) because
 * it absorbs the double negative from the SPH momentum equation.
 * ====================================================================== */

struct PrecalcParams {
    float poly6_coeff;           // 315 / (64 * pi * h^9)
    float spiky_grad_coeff;      // -45 / (pi * h^6)  (negative)
    float viscosity_lap_coeff;   // 45 / (pi * h^6)
    float pressure_precalc;      // +45 / (pi * h^6)   (positive, see note)
    float viscosity_precalc;     // mu * 45 / (pi * h^6)
};

__constant__ PrecalcParams c_precalc;

/* ======================================================================
 * packed_info bitfield layout (uint32)
 *
 *   Bits [7:0]   material_id    (256 materials max, 32 used)
 *   Bits [9:8]   behavior_class (FLUID=0, GRANULAR=1, GAS=2, STATIC=3)
 *   Bit  [10]    is_sleeping    (1 = asleep, skip force computation)
 *   Bit  [11]    spawn_flag     (1 = just spawned, needs initialization)
 *   Bit  [12]    just_woke      (1 = woke this frame, needs velocity reset)
 *   Bits [20:13] body_id        (rigid body index, 0-255; only valid when MAT_RIGID)
 *   Bits [31:21] reserved
 * ====================================================================== */

#define GET_MATERIAL_ID(p)  ((p) & 0xFF)
#define GET_BEHAVIOR(p)     (((p) >> 8) & 0x3)
#define IS_SLEEPING(p)      (((p) >> 10) & 1)
#define SET_SLEEPING(p)     ((p) | 0x400)
#define CLEAR_SLEEPING(p)   ((p) & ~0x400)
#define HAS_SPAWN_FLAG(p)   (((p) >> 11) & 1)
#define SET_SPAWN_FLAG(p)   ((p) | 0x800)
#define CLEAR_SPAWN_FLAG(p) ((p) & ~0x800)
#define HAS_JUST_WOKE(p)    (((p) >> 12) & 1)
#define SET_JUST_WOKE(p)    ((p) | 0x1000)
#define CLEAR_JUST_WOKE(p)  ((p) & ~0x1000)
#define MAKE_PACKED(mat, beh) (((mat) & 0xFF) | (((beh) & 0x3) << 8))
#define GET_BODY_ID(p)        (((p) >> 13) & 0xFF)
#define SET_BODY_ID(p, bid)   (((p) & ~(0xFF << 13)) | (((bid) & 0xFF) << 13))

/* MAT_RIGID material ID -- Akinci boundary particles (must match materials.py) */
#define MAT_RIGID 18

/* ======================================================================
 * Array type conventions (documented per acceptance criteria)
 *
 * float4 arrays (w component unused, set to 0):
 *   position, velocity, veleval, sph_force, color
 *
 * half4 arrays (FP16 copies for bandwidth reduction in neighbor loops):
 *   sorted_velocity_h -- populated during sort, read in step1/step2
 *
 * float arrays:
 *   density, mass, temperature, health, lifetime,
 *   shear_rate, exposure_heat, exposure_corrode
 *
 * uint32 arrays:
 *   packed_info
 *
 * uint8 arrays:
 *   sleep_counter
 * ====================================================================== */

/* ======================================================================
 * RigidBody -- dynamic rigid body for Akinci two-way coupling.
 *
 * 6 float4s = 96 bytes per body, no padding.
 * Stored in GLOBAL memory (not __constant__) because kernels write to it
 * during rigid body integration. Reads via __ldg() for L1 cache.
 *
 * Max 8 rigid bodies.
 * ====================================================================== */

struct RigidBody {
    float4 position;      // xyz=COM, w=inv_mass
    float4 rotation;      // quaternion (x,y,z,w) — w is scalar part
    float4 lin_vel;       // xyz=velocity, w=restitution
    float4 ang_vel;       // xyz=angular velocity, w=friction
    float4 half_extents;  // xyz=size, w=__int_as_float(sdf_type) (0=box, 1=sphere)
    float4 inertia_inv;   // xyz=1/I_diagonal, w=__int_as_float(is_kinematic) (0 or 1)
};

#define MAX_RIGID_BODIES 8

__constant__ int c_num_rigid_bodies;

/* ======================================================================
 * SDFObject -- analytical SDF primitive for collision detection.
 *
 * 5 float4s = 80 bytes per object, no padding.
 * Type is encoded in pos_and_type.w via __int_as_float/__float_as_int
 * to avoid int/float4 alignment issues.
 *
 * Type values: 0=box, 1=sphere, 2=cylinder, 3=plane
 * ====================================================================== */

struct SDFObject {
    float4 pos_and_type;   // xyz=center, w=__int_as_float(type)
    float4 size_and_r;     // xyz=half_extents/radius/height, w=restitution
    float4 quat;           // orientation quaternion (x,y,z,w) — w is scalar part
    float4 velocity;       // xyz=linear velocity, w=angular_speed (for kinematic)
    float4 angular_axis;   // xyz=rotation axis (for kinematic), w=friction
};

#define SDF_BOX      0
#define SDF_SPHERE   1
#define SDF_CYLINDER 2
#define SDF_PLANE    3

#define MAX_SDF_OBJECTS 16

__constant__ SDFObject c_sdf_objects[MAX_SDF_OBJECTS];
__constant__ int c_num_sdf_objects;

/* ======================================================================
 * FP16 helpers -- load/store half4 as float4 via __half2.
 *
 * Half4 is stored as 8 bytes (4 x FP16). Loaded as uint2 (one 64-bit load)
 * and converted to float4 via __half2float. Computation stays FP32.
 * ====================================================================== */

#include <cuda_fp16.h>

__device__ inline float4 load_half4(const void* ptr) {
    uint2 raw = __ldg((const uint2*)ptr);
    __half2 xy = *reinterpret_cast<const __half2*>(&raw.x);
    __half2 zw = *reinterpret_cast<const __half2*>(&raw.y);
    return make_float4(
        __low2float(xy), __high2float(xy),
        __low2float(zw), __high2float(zw)
    );
}

__device__ inline void store_half4(void* ptr, float4 v) {
    __half2 xy = __floats2half2_rn(v.x, v.y);
    __half2 zw = __floats2half2_rn(v.z, v.w);
    uint2 raw;
    raw.x = *reinterpret_cast<const uint*>(&xy);
    raw.y = *reinterpret_cast<const uint*>(&zw);
    *reinterpret_cast<uint2*>(ptr) = raw;
}

/* --- FP16 scalar: load/store half as float (2 bytes) --- */

__device__ inline float load_half1(const void* ptr) {
    __half h = __ldg((const __half*)ptr);
    return __half2float(h);
}

__device__ inline void store_half1(void* ptr, float v) {
    *(__half*)ptr = __float2half(v);
}

#endif /* FALLINGSAND3D_COMMON_CUH */
