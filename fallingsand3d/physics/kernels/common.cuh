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

/* ======================================================================
 * MaterialProps -- per-material constants uploaded to constant memory.
 * 18 fields x 4 bytes = 72 bytes (includes 1 pad field).
 * ====================================================================== */

struct MaterialProps {
    float rest_density;          // kg/m^3
    float eos_stiffness;         // Tait EOS k
    float eos_gamma;             // Tait EOS exponent (7 for liquid, 1 for gas)
    float base_viscosity;        // Pa*s
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
 * 40 bytes total (10 fields × 4 bytes).
 * ====================================================================== */

struct GridParams {
    float3 grid_min;    // world-space minimum corner (for pos → cell)
    float3 grid_delta;  // 1 / cell_size per axis (≈ 1/h)
    uint   table_size;  // hash table size (power of 2, e.g. 262144)
    uint   table_mask;  // table_size - 1 (for & masking)
};

__constant__ GridParams c_grid;

/* ======================================================================
 * Spatial hash helpers -- shared by ALL kernel files.
 *
 * Position → cell:   int3 cell = calcGridCell(pos)
 * Cell → hash:       uint h = spatialHash(cell)
 * Neighbor cell hash: uint h = spatialHash(make_int3(cx+dx, cy+dy, cz+dz))
 *
 * No bounds checking needed — any integer cell coords produce a valid
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
 *   Bits [31:13] reserved
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

/* ======================================================================
 * Array type conventions (documented per acceptance criteria)
 *
 * float4 arrays (w component unused, set to 0):
 *   position, velocity, veleval, sph_force, color
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

#endif /* FALLINGSAND3D_COMMON_CUH */
