"""Material property table and interaction matrix for multi-material SPH.

Defines 16 materials (IDs 0-15) as Python dataclasses, a 32x32 interaction
matrix, and an upload_to_gpu() function that copies both tables to CUDA
constant memory via CuPy RawModule symbol lookup.

Constant-memory symbols
-----------------------
- ``c_materials``   : MaterialProps[32]
- ``c_interactions`` : Interaction[32][32]

The C struct layout is defined in KERNEL_SOURCE below and must stay in sync
with any CUDA kernels that read these constants.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CUDA kernel source -- struct definitions + constant memory declarations
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
// ---- struct definitions (must match numpy dtypes in materials.py) --------

struct MaterialProps {
    int   id;               // material ID (0-31)
    float rest_density;     // kg/m^3
    float viscosity;        // Pa*s
    float gas_stiffness;    // pressure response
    float surface_tension;  // surface tension coefficient
    float thermal_conductivity;  // W/(m*K)
    float specific_heat;    // J/(kg*K)
    float melting_point;    // Kelvin
    float boiling_point;    // Kelvin
    float ignition_point;   // Kelvin (0 = non-flammable)
    float friction_static;  // static friction coefficient
    float friction_kinetic; // kinetic friction coefficient
    float restitution;      // coefficient of restitution (bounciness)
    int   is_solid;         // 1 = solid, 0 = fluid/gas
    int   is_flammable;     // 1 = can burn
    int   _pad;             // padding to 16-int boundary (64 bytes)
};

struct Interaction {
    float reaction_rate;    // reaction speed (0 = no reaction)
    float heat_exchange;    // heat transfer coefficient between pair
};

__constant__ MaterialProps c_materials[32];
__constant__ Interaction   c_interactions[32][32];

// Trivial test kernel: read a material field and an interaction field.
extern "C" __global__
void test_read_materials(float* out) {
    // out[0] = c_materials[5].rest_density   (water -> 1000.0)
    // out[1] = c_interactions[8][10].reaction_rate  (acid-metal -> 0.3)
    out[0] = c_materials[5].rest_density;
    out[1] = c_interactions[8][10].reaction_rate;
}
"""

# ---------------------------------------------------------------------------
# Numpy structured dtypes matching the C structs exactly
# ---------------------------------------------------------------------------

MATERIAL_PROPS_DTYPE = np.dtype(
    [
        ("id", np.int32),
        ("rest_density", np.float32),
        ("viscosity", np.float32),
        ("gas_stiffness", np.float32),
        ("surface_tension", np.float32),
        ("thermal_conductivity", np.float32),
        ("specific_heat", np.float32),
        ("melting_point", np.float32),
        ("boiling_point", np.float32),
        ("ignition_point", np.float32),
        ("friction_static", np.float32),
        ("friction_kinetic", np.float32),
        ("restitution", np.float32),
        ("is_solid", np.int32),
        ("is_flammable", np.int32),
        ("_pad", np.int32),
    ],
    align=True,
)

INTERACTION_DTYPE = np.dtype(
    [
        ("reaction_rate", np.float32),
        ("heat_exchange", np.float32),
    ],
    align=True,
)

# Compile-time size checks (must match sizeof(MaterialProps) and sizeof(Interaction))
assert MATERIAL_PROPS_DTYPE.itemsize == 64, (
    f"MaterialProps size mismatch: {MATERIAL_PROPS_DTYPE.itemsize} != 64"
)
assert INTERACTION_DTYPE.itemsize == 8, (
    f"Interaction size mismatch: {INTERACTION_DTYPE.itemsize} != 8"
)

MAX_MATERIALS = 32  # slots in constant memory (IDs 0-31)
NUM_DEFINED = 16  # IDs 0-15 are defined; 16-31 reserved (zeroed)

# ---------------------------------------------------------------------------
# MaterialDef dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MaterialDef:
    """Python-side definition of a single material.

    Field names and semantics match the ``MaterialProps`` C struct.
    """

    id: int
    name: str  # human-readable label (not uploaded to GPU)
    rest_density: float  # kg/m^3
    viscosity: float  # Pa*s
    gas_stiffness: float
    surface_tension: float
    thermal_conductivity: float  # W/(m*K)
    specific_heat: float  # J/(kg*K)
    melting_point: float  # Kelvin
    boiling_point: float  # Kelvin
    ignition_point: float  # Kelvin; 0 = non-flammable
    friction_static: float
    friction_kinetic: float
    restitution: float  # bounciness 0..1
    is_solid: bool
    is_flammable: bool

    def to_numpy(self) -> np.void:
        """Pack into a single numpy structured-array element."""
        return np.array(
            (
                self.id,
                self.rest_density,
                self.viscosity,
                self.gas_stiffness,
                self.surface_tension,
                self.thermal_conductivity,
                self.specific_heat,
                self.melting_point,
                self.boiling_point,
                self.ignition_point,
                self.friction_static,
                self.friction_kinetic,
                self.restitution,
                int(self.is_solid),
                int(self.is_flammable),
                0,  # _pad
            ),
            dtype=MATERIAL_PROPS_DTYPE,
        )


# ---------------------------------------------------------------------------
# Material ID constants
# ---------------------------------------------------------------------------

DEAD = 0
STONE = 1
SAND = 2
DIRT = 3
GRAVEL = 4
WATER = 5
OIL = 6
LAVA = 7
ACID = 8
WOOD = 9
METAL = 10
ICE = 11
STEAM = 12
SMOKE = 13
FIRE = 14
GUNPOWDER = 15

# ---------------------------------------------------------------------------
# Material definitions (16 entries, IDs 0-15)
#
# Property values are chosen for a falling-sand / multi-material SPH game:
#   - rest_density in kg/m^3 (approximate real-world values)
#   - viscosity in Pa*s (scaled for game feel)
#   - gas_stiffness controls pressure response
#   - temperatures in Kelvin
# ---------------------------------------------------------------------------

MATERIALS: Dict[int, MaterialDef] = {
    DEAD: MaterialDef(
        id=DEAD, name="DEAD",
        rest_density=0.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=0.0,
        specific_heat=0.0, melting_point=0.0, boiling_point=0.0,
        ignition_point=0.0, friction_static=0.0, friction_kinetic=0.0,
        restitution=0.0, is_solid=False, is_flammable=False,
    ),
    STONE: MaterialDef(
        id=STONE, name="STONE",
        rest_density=2600.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=2.0,
        specific_heat=800.0, melting_point=1500.0, boiling_point=3000.0,
        ignition_point=0.0, friction_static=0.7, friction_kinetic=0.5,
        restitution=0.2, is_solid=True, is_flammable=False,
    ),
    SAND: MaterialDef(
        id=SAND, name="SAND",
        rest_density=1600.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=0.3,
        specific_heat=830.0, melting_point=1700.0, boiling_point=2500.0,
        ignition_point=0.0, friction_static=0.6, friction_kinetic=0.4,
        restitution=0.1, is_solid=True, is_flammable=False,
    ),
    DIRT: MaterialDef(
        id=DIRT, name="DIRT",
        rest_density=1500.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=0.25,
        specific_heat=900.0, melting_point=1400.0, boiling_point=2200.0,
        ignition_point=0.0, friction_static=0.5, friction_kinetic=0.35,
        restitution=0.05, is_solid=True, is_flammable=False,
    ),
    GRAVEL: MaterialDef(
        id=GRAVEL, name="GRAVEL",
        rest_density=1800.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=0.5,
        specific_heat=840.0, melting_point=1500.0, boiling_point=2800.0,
        ignition_point=0.0, friction_static=0.65, friction_kinetic=0.45,
        restitution=0.15, is_solid=True, is_flammable=False,
    ),
    WATER: MaterialDef(
        id=WATER, name="WATER",
        rest_density=1000.0, viscosity=1.0, gas_stiffness=1.5,
        surface_tension=0.072, thermal_conductivity=0.6,
        specific_heat=4186.0, melting_point=273.0, boiling_point=373.0,
        ignition_point=0.0, friction_static=0.0, friction_kinetic=0.0,
        restitution=0.0, is_solid=False, is_flammable=False,
    ),
    OIL: MaterialDef(
        id=OIL, name="OIL",
        rest_density=800.0, viscosity=5.0, gas_stiffness=1.2,
        surface_tension=0.03, thermal_conductivity=0.15,
        specific_heat=2000.0, melting_point=250.0, boiling_point=570.0,
        ignition_point=480.0, friction_static=0.0, friction_kinetic=0.0,
        restitution=0.0, is_solid=False, is_flammable=True,
    ),
    LAVA: MaterialDef(
        id=LAVA, name="LAVA",
        rest_density=2500.0, viscosity=100.0, gas_stiffness=1.0,
        surface_tension=0.4, thermal_conductivity=1.5,
        specific_heat=1000.0, melting_point=1000.0, boiling_point=2500.0,
        ignition_point=0.0, friction_static=0.0, friction_kinetic=0.0,
        restitution=0.0, is_solid=False, is_flammable=False,
    ),
    ACID: MaterialDef(
        id=ACID, name="ACID",
        rest_density=1200.0, viscosity=1.5, gas_stiffness=1.4,
        surface_tension=0.05, thermal_conductivity=0.5,
        specific_heat=2500.0, melting_point=250.0, boiling_point=380.0,
        ignition_point=0.0, friction_static=0.0, friction_kinetic=0.0,
        restitution=0.0, is_solid=False, is_flammable=False,
    ),
    WOOD: MaterialDef(
        id=WOOD, name="WOOD",
        rest_density=600.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=0.15,
        specific_heat=1700.0, melting_point=0.0, boiling_point=0.0,
        ignition_point=570.0, friction_static=0.5, friction_kinetic=0.3,
        restitution=0.3, is_solid=True, is_flammable=True,
    ),
    METAL: MaterialDef(
        id=METAL, name="METAL",
        rest_density=7800.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=50.0,
        specific_heat=450.0, melting_point=1800.0, boiling_point=3300.0,
        ignition_point=0.0, friction_static=0.6, friction_kinetic=0.4,
        restitution=0.4, is_solid=True, is_flammable=False,
    ),
    ICE: MaterialDef(
        id=ICE, name="ICE",
        rest_density=917.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=2.2,
        specific_heat=2090.0, melting_point=273.0, boiling_point=373.0,
        ignition_point=0.0, friction_static=0.1, friction_kinetic=0.03,
        restitution=0.3, is_solid=True, is_flammable=False,
    ),
    STEAM: MaterialDef(
        id=STEAM, name="STEAM",
        rest_density=0.6, viscosity=0.01, gas_stiffness=3.0,
        surface_tension=0.0, thermal_conductivity=0.025,
        specific_heat=2010.0, melting_point=0.0, boiling_point=0.0,
        ignition_point=0.0, friction_static=0.0, friction_kinetic=0.0,
        restitution=0.0, is_solid=False, is_flammable=False,
    ),
    SMOKE: MaterialDef(
        id=SMOKE, name="SMOKE",
        rest_density=0.3, viscosity=0.01, gas_stiffness=2.5,
        surface_tension=0.0, thermal_conductivity=0.02,
        specific_heat=1000.0, melting_point=0.0, boiling_point=0.0,
        ignition_point=0.0, friction_static=0.0, friction_kinetic=0.0,
        restitution=0.0, is_solid=False, is_flammable=False,
    ),
    FIRE: MaterialDef(
        id=FIRE, name="FIRE",
        rest_density=0.2, viscosity=0.01, gas_stiffness=4.0,
        surface_tension=0.0, thermal_conductivity=0.05,
        specific_heat=1000.0, melting_point=0.0, boiling_point=0.0,
        ignition_point=0.0, friction_static=0.0, friction_kinetic=0.0,
        restitution=0.0, is_solid=False, is_flammable=False,
    ),
    GUNPOWDER: MaterialDef(
        id=GUNPOWDER, name="GUNPOWDER",
        rest_density=1700.0, viscosity=0.0, gas_stiffness=0.0,
        surface_tension=0.0, thermal_conductivity=0.2,
        specific_heat=800.0, melting_point=0.0, boiling_point=0.0,
        ignition_point=480.0, friction_static=0.5, friction_kinetic=0.35,
        restitution=0.1, is_solid=True, is_flammable=True,
    ),
}

assert len(MATERIALS) == NUM_DEFINED

# ---------------------------------------------------------------------------
# 32x32 Interaction matrix
#
# Each (i, j) pair stores reaction_rate and heat_exchange.
# Only non-trivial pairs are listed; everything else defaults to zero.
#
# Key interactions:
#   water + lava   -> steam (high heat exchange)
#   acid  + metal  -> reaction_rate 0.3
#   acid  + stone  -> slow corrosion
#   acid  + wood   -> slow corrosion
#   fire  + wood   -> burning
#   fire  + oil    -> fast burning
#   fire  + gunpowder -> explosion
#   lava  + ice    -> steam + stone
#   lava  + wood   -> burning
#   lava  + water  -> steam + stone
#   water + ice    -> melting
# ---------------------------------------------------------------------------

# (mat_a, mat_b): (reaction_rate, heat_exchange)
# Symmetric: both (a,b) and (b,a) are set.
_INTERACTION_PAIRS: List[Tuple[int, int, float, float]] = [
    # water interactions
    (WATER, LAVA,   0.5,  5.0),    # water + lava -> steam, cools lava
    (WATER, ICE,    0.1,  2.0),    # water near ice -> freezing/melting
    (WATER, FIRE,   0.8,  3.0),    # water extinguishes fire
    (WATER, ACID,   0.05, 0.5),    # water dilutes acid slowly
    # lava interactions
    (LAVA, ICE,     0.6,  6.0),    # lava melts ice rapidly
    (LAVA, WOOD,    0.7,  4.0),    # lava ignites wood
    (LAVA, METAL,   0.05, 3.0),    # lava heats metal
    (LAVA, STONE,   0.0,  2.0),    # lava heats stone (no reaction, just heat)
    (LAVA, OIL,     0.8,  5.0),    # lava ignites oil
    (LAVA, GUNPOWDER, 0.9, 5.0),  # lava detonates gunpowder
    # acid interactions
    (ACID, STONE,   0.15, 0.3),    # acid slowly corrodes stone
    (ACID, METAL,   0.3,  0.5),    # acid corrodes metal
    (ACID, WOOD,    0.2,  0.2),    # acid corrodes wood
    (ACID, DIRT,    0.1,  0.1),    # acid dissolves dirt
    (ACID, ICE,     0.2,  1.0),    # acid melts ice
    # fire interactions
    (FIRE, WOOD,    0.6,  2.0),    # fire burns wood
    (FIRE, OIL,     0.8,  3.0),    # fire burns oil quickly
    (FIRE, GUNPOWDER, 0.9, 4.0),  # fire detonates gunpowder
    (FIRE, ICE,     0.3,  2.5),    # fire melts ice
    (FIRE, SMOKE,   0.0,  0.5),    # fire produces smoke (heat only)
    # oil interactions
    (OIL, WATER,    0.0,  0.3),    # oil floats on water (no reaction, low heat)
    # ice + steam
    (ICE, STEAM,    0.2,  1.5),    # steam melts ice
    # metal + fire
    (FIRE, METAL,   0.0,  2.0),    # fire heats metal (no reaction, heat only)
    # stone + fire
    (FIRE, STONE,   0.0,  0.5),    # fire heats stone slowly
]


def build_interaction_matrix() -> np.ndarray:
    """Build the 32x32 interaction matrix as a numpy structured array.

    Returns a (32, 32) array of Interaction structs.  Interactions are
    symmetric: setting (a, b) also sets (b, a).
    """
    matrix = np.zeros((MAX_MATERIALS, MAX_MATERIALS), dtype=INTERACTION_DTYPE)
    for a, b, rate, heat in _INTERACTION_PAIRS:
        matrix[a][b]["reaction_rate"] = rate
        matrix[a][b]["heat_exchange"] = heat
        matrix[b][a]["reaction_rate"] = rate
        matrix[b][a]["heat_exchange"] = heat
    return matrix


def build_material_array() -> np.ndarray:
    """Build the MaterialProps[32] array as a numpy structured array.

    IDs 0-15 come from MATERIALS; IDs 16-31 are zeroed (reserved).
    """
    arr = np.zeros(MAX_MATERIALS, dtype=MATERIAL_PROPS_DTYPE)
    for mat_id, mat_def in MATERIALS.items():
        arr[mat_id] = mat_def.to_numpy()
    return arr


# ---------------------------------------------------------------------------
# GPU upload via CuPy RawModule
# ---------------------------------------------------------------------------

# Cached CuPy RawModule (compiled on first use)
_module: Optional[object] = None


def _ensure_ptx_if_needed() -> None:
    """Force PTX compilation when the GPU arch exceeds NVRTC's max sm target.

    CuPy's NVRTC backend generates sm_NN cubin by default, but if the GPU
    compute capability is newer than what the bundled NVRTC supports (e.g.
    sm_120 Blackwell with CUDA 12.5 NVRTC that maxes at sm_90), the cubin
    won't load.  Forcing PTX (compute_NN) produces forward-compatible code
    that the driver JIT-compiles at load time.
    """
    from cupy.cuda import compiler as _compiler  # type: ignore[import-untyped]
    from cupy.cuda import device as _device  # type: ignore[import-untyped]

    gpu_cc = _device.Device().compute_capability
    nvrtc_max = _compiler._get_max_compute_capability()
    if int(gpu_cc) > int(nvrtc_max):
        _compiler._use_ptx = True
        # Clear memoized arch-option cache so the new flag takes effect
        if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
            _compiler._get_arch_for_options_for_nvrtc._cache = {}
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}


def _get_module() -> "object":
    """Compile (or return cached) CuPy RawModule from KERNEL_SOURCE."""
    global _module
    if _module is None:
        import cupy  # type: ignore[import-untyped]

        _ensure_ptx_if_needed()
        _module = cupy.RawModule(code=KERNEL_SOURCE, options=("--std=c++11",))
    return _module


def upload_to_gpu() -> None:
    """Copy MaterialProps[32] and Interaction[32][32] to CUDA constant memory.

    Uses ``module.get_global('c_materials')`` / ``module.get_global('c_interactions')``
    to obtain device pointers, then ``cupy.cuda.runtime.memcpy`` with
    ``cudaMemcpyHostToDevice`` to upload the numpy arrays.
    """
    import cupy  # type: ignore[import-untyped]

    cudaMemcpyHostToDevice = 1  # enum value from cuda.h

    module = _get_module()

    # Build host arrays
    h_materials = build_material_array()
    h_interactions = build_interaction_matrix()

    # Get device pointers to constant memory symbols
    d_materials_ptr = module.get_global("c_materials")  # type: ignore[union-attr]
    d_interactions_ptr = module.get_global("c_interactions")  # type: ignore[union-attr]

    # Upload MaterialProps[32]
    cupy.cuda.runtime.memcpy(
        int(d_materials_ptr),
        h_materials.ctypes.data,
        h_materials.nbytes,
        cudaMemcpyHostToDevice,
    )

    # Upload Interaction[32][32]
    cupy.cuda.runtime.memcpy(
        int(d_interactions_ptr),
        h_interactions.ctypes.data,
        h_interactions.nbytes,
        cudaMemcpyHostToDevice,
    )


def get_module() -> "object":
    """Return the compiled CuPy RawModule (public accessor for tests)."""
    return _get_module()
