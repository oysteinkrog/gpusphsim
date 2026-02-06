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
 * 16 fields x 4 bytes = 64 bytes (no extra padding needed).
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
};
// Static assert: sizeof(MaterialProps) must be 64 bytes

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
 * Total: 32*64 + 32*32*8 = 2048 + 8192 = 10240 bytes (well under 64 KB)
 * ====================================================================== */

__constant__ MaterialProps c_materials[32];
__constant__ Interaction   c_interactions[32][32];

/* ======================================================================
 * GridParams -- uniform grid parameters for spatial hashing.
 *
 * grid_res uses int3 (not float3 as in the root-level prototype) for
 * correct integer arithmetic in hash computation.
 * ====================================================================== */

struct GridParams {
    float3 grid_min;    // world-space minimum corner
    float3 grid_max;    // world-space maximum corner
    int3   grid_res;    // number of cells per axis
    float3 grid_delta;  // 1 / cell_size = grid_res / grid_size
    int    num_cells;   // grid_res.x * grid_res.y * grid_res.z
};

__constant__ GridParams c_grid;

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
