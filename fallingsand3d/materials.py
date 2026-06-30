"""Material property table and interaction matrix for multi-material SPH.

Defines 16 materials (IDs 0-15) as Python dataclasses, a 32x32 interaction
matrix, and an upload_to_gpu() function that copies both tables to CUDA
constant memory via CuPy RawModule symbol lookup.

Struct layout matches fallingsand3d/physics/kernels/common.cuh exactly.

Constant-memory symbols
-----------------------
- ``c_materials``   : MaterialProps[32]
- ``c_interactions`` : Interaction[32][32]
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Numpy structured dtypes matching the C structs in common.cuh
# ---------------------------------------------------------------------------

MATERIAL_PROPS_DTYPE = np.dtype(
    [
        ("rest_density", np.float32),
        ("eos_stiffness", np.float32),
        ("eos_gamma", np.float32),
        ("base_viscosity", np.float32),
        ("friction_coeff", np.float32),
        ("cohesion", np.float32),
        ("buoyancy_extra", np.float32),
        ("thermal_conductivity", np.float32),
        ("heat_capacity", np.float32),
        ("temp_melt", np.float32),
        ("temp_boil", np.float32),
        ("temp_ignite", np.float32),
        ("behavior_class", np.int32),
        ("color_r", np.float32),
        ("color_g", np.float32),
        ("color_b", np.float32),
        ("thermal_expansion", np.float32),
        ("_pad0", np.float32),
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
assert MATERIAL_PROPS_DTYPE.itemsize == 72, (
    f"MaterialProps size mismatch: {MATERIAL_PROPS_DTYPE.itemsize} != 72"
)
assert INTERACTION_DTYPE.itemsize == 8, (
    f"Interaction size mismatch: {INTERACTION_DTYPE.itemsize} != 8"
)

MAX_MATERIALS = 32  # slots in constant memory (IDs 0-31)
NUM_DEFINED = 19  # IDs 0-18 are defined; 19-31 reserved (zeroed)

# ---------------------------------------------------------------------------
# BehaviorClass enum (mirrors common.cuh)
# ---------------------------------------------------------------------------

FLUID = 0
GRANULAR = 1
GAS = 2
STATIC = 3

# ---------------------------------------------------------------------------
# MaterialDef dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MaterialDef:
    """Python-side definition of a single material.

    Field names and semantics match the ``MaterialProps`` C struct in common.cuh.
    """

    id: int
    name: str  # human-readable label (not uploaded to GPU)
    rest_density: float  # kg/m^3
    eos_stiffness: float  # Tait EOS k
    eos_gamma: float  # Tait EOS exponent (7 for liquid, 1 for gas)
    base_viscosity: float  # Pa*s
    friction_coeff: float  # mu_s for granular friction
    cohesion: float  # cohesion coefficient
    buoyancy_extra: float  # extra buoyancy factor for gas
    thermal_conductivity: float  # W/(m*K)
    heat_capacity: float  # J/(kg*K)
    temp_melt: float  # Kelvin
    temp_boil: float  # Kelvin
    temp_ignite: float  # Kelvin (0 = non-flammable)
    behavior_class: int  # BehaviorClass enum value
    color_r: float  # default material color R [0,1]
    color_g: float  # default material color G [0,1]
    color_b: float  # default material color B [0,1]
    thermal_expansion: float = 0.0  # Boussinesq thermal expansion coefficient (1/K)

    def to_numpy(self) -> np.void:
        """Pack into a single numpy structured-array element."""
        return np.array(
            (
                self.rest_density,
                self.eos_stiffness,
                self.eos_gamma,
                self.base_viscosity,
                self.friction_coeff,
                self.cohesion,
                self.buoyancy_extra,
                self.thermal_conductivity,
                self.heat_capacity,
                self.temp_melt,
                self.temp_boil,
                self.temp_ignite,
                self.behavior_class,
                self.color_r,
                self.color_g,
                self.color_b,
                self.thermal_expansion,
                0.0,  # _pad0
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
WET_SAND = 16
MUD = 17
MAT_RIGID = 18

# ---------------------------------------------------------------------------
# Material definitions (19 entries, IDs 0-18)
#
# Property values chosen for a falling-sand / multi-material SPH game:
#   - rest_density in kg/m^3 (approximate real-world)
#   - eos_stiffness: Tait EOS pressure coefficient (game-tuned)
#   - eos_gamma: 7 for FLUID/GRANULAR, 1 for GAS, 0 for STATIC/DEAD
#   - base_viscosity in Pa*s (scaled for game feel)
#   - temperatures in Kelvin
# ---------------------------------------------------------------------------

MATERIALS: Dict[int, MaterialDef] = {
    DEAD: MaterialDef(
        id=DEAD, name="DEAD",
        rest_density=0.0, eos_stiffness=0.0, eos_gamma=0.0,
        base_viscosity=0.0, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=0.0, heat_capacity=0.0,
        temp_melt=0.0, temp_boil=0.0, temp_ignite=0.0,
        behavior_class=STATIC,
        color_r=0.0, color_g=0.0, color_b=0.0,
    ),
    STONE: MaterialDef(
        id=STONE, name="STONE",
        rest_density=6500.0, eos_stiffness=1500.0, eos_gamma=7.0,
        base_viscosity=0.0, friction_coeff=0.7, cohesion=0.5,
        buoyancy_extra=0.0, thermal_conductivity=200.0, heat_capacity=800.0,
        temp_melt=1500.0, temp_boil=3000.0, temp_ignite=0.0,
        behavior_class=STATIC,
        color_r=0.5, color_g=0.5, color_b=0.5,
    ),
    SAND: MaterialDef(
        id=SAND, name="SAND",
        rest_density=4000.0, eos_stiffness=5000.0, eos_gamma=7.0,
        base_viscosity=0.5, friction_coeff=0.6, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=30.0, heat_capacity=830.0,
        temp_melt=1700.0, temp_boil=2500.0, temp_ignite=0.0,
        behavior_class=GRANULAR,
        color_r=0.76, color_g=0.70, color_b=0.50,
    ),
    DIRT: MaterialDef(
        id=DIRT, name="DIRT",
        rest_density=3750.0, eos_stiffness=3000.0, eos_gamma=7.0,
        base_viscosity=0.5, friction_coeff=0.5, cohesion=0.1,
        buoyancy_extra=0.0, thermal_conductivity=30.0, heat_capacity=900.0,
        temp_melt=1400.0, temp_boil=2200.0, temp_ignite=0.0,
        behavior_class=GRANULAR,
        color_r=0.55, color_g=0.35, color_b=0.17,
    ),
    GRAVEL: MaterialDef(
        id=GRAVEL, name="GRAVEL",
        rest_density=4250.0, eos_stiffness=4000.0, eos_gamma=7.0,
        base_viscosity=0.3, friction_coeff=0.65, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=30.0, heat_capacity=840.0,
        temp_melt=1500.0, temp_boil=2800.0, temp_ignite=0.0,
        behavior_class=GRANULAR,
        color_r=0.45, color_g=0.42, color_b=0.40,
    ),
    WATER: MaterialDef(
        id=WATER, name="WATER",
        rest_density=2500.0, eos_stiffness=500.0, eos_gamma=7.0,
        base_viscosity=1.0, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=50.0, heat_capacity=4186.0,
        temp_melt=273.0, temp_boil=373.0, temp_ignite=0.0,
        behavior_class=FLUID,
        color_r=0.2, color_g=0.5, color_b=0.9,
        thermal_expansion=0.0003,  # Boussinesq beta for water (~3e-4 1/K)
    ),
    OIL: MaterialDef(
        id=OIL, name="OIL",
        rest_density=2125.0, eos_stiffness=400.0, eos_gamma=7.0,
        base_viscosity=5.0, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=20.0, heat_capacity=2000.0,
        temp_melt=250.0, temp_boil=570.0, temp_ignite=480.0,
        behavior_class=FLUID,
        color_r=0.15, color_g=0.10, color_b=0.05,
        thermal_expansion=0.0007,  # oil expands more than water
    ),
    LAVA: MaterialDef(
        id=LAVA, name="LAVA",
        rest_density=6500.0, eos_stiffness=800.0, eos_gamma=7.0,
        base_viscosity=100.0, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=500.0, heat_capacity=1000.0,
        temp_melt=1000.0, temp_boil=2500.0, temp_ignite=0.0,
        behavior_class=FLUID,
        color_r=1.0, color_g=0.3, color_b=0.0,
        thermal_expansion=0.0001,  # lava has low expansion
    ),
    ACID: MaterialDef(
        id=ACID, name="ACID",
        rest_density=2500.0, eos_stiffness=450.0, eos_gamma=7.0,
        base_viscosity=2.0, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=50.0, heat_capacity=2500.0,
        temp_melt=250.0, temp_boil=380.0, temp_ignite=0.0,
        behavior_class=FLUID,
        color_r=0.4, color_g=1.0, color_b=0.1,
        thermal_expansion=0.0003,  # similar to water
    ),
    WOOD: MaterialDef(
        id=WOOD, name="WOOD",
        rest_density=1500.0, eos_stiffness=300.0, eos_gamma=7.0,
        base_viscosity=0.0, friction_coeff=0.5, cohesion=0.3,
        buoyancy_extra=0.0, thermal_conductivity=5.0, heat_capacity=1700.0,
        temp_melt=0.0, temp_boil=0.0, temp_ignite=570.0,
        behavior_class=STATIC,
        color_r=0.55, color_g=0.35, color_b=0.12,
    ),
    METAL: MaterialDef(
        id=METAL, name="METAL",
        rest_density=19500.0, eos_stiffness=2000.0, eos_gamma=7.0,
        base_viscosity=0.0, friction_coeff=0.6, cohesion=1.0,
        buoyancy_extra=0.0, thermal_conductivity=2000.0, heat_capacity=450.0,
        temp_melt=1800.0, temp_boil=3300.0, temp_ignite=0.0,
        behavior_class=STATIC,
        color_r=0.7, color_g=0.7, color_b=0.75,
    ),
    ICE: MaterialDef(
        id=ICE, name="ICE",
        rest_density=2300.0, eos_stiffness=500.0, eos_gamma=7.0,
        base_viscosity=0.0, friction_coeff=0.1, cohesion=0.2,
        buoyancy_extra=0.0, thermal_conductivity=100.0, heat_capacity=2090.0,
        temp_melt=273.0, temp_boil=373.0, temp_ignite=0.0,
        behavior_class=STATIC,
        color_r=0.7, color_g=0.9, color_b=1.0,
    ),
    STEAM: MaterialDef(
        id=STEAM, name="STEAM",
        rest_density=0.6, eos_stiffness=3.0, eos_gamma=1.0,
        base_viscosity=0.01, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.01, thermal_conductivity=10.0, heat_capacity=2010.0,
        temp_melt=0.0, temp_boil=0.0, temp_ignite=0.0,
        behavior_class=GAS,
        color_r=0.85, color_g=0.85, color_b=0.90,
    ),
    SMOKE: MaterialDef(
        id=SMOKE, name="SMOKE",
        rest_density=0.3, eos_stiffness=2.5, eos_gamma=1.0,
        base_viscosity=0.01, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.005, thermal_conductivity=5.0, heat_capacity=1000.0,
        temp_melt=0.0, temp_boil=0.0, temp_ignite=0.0,
        behavior_class=GAS,
        color_r=0.3, color_g=0.3, color_b=0.3,
    ),
    FIRE: MaterialDef(
        id=FIRE, name="FIRE",
        rest_density=0.2, eos_stiffness=4.0, eos_gamma=1.0,
        base_viscosity=0.01, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.02, thermal_conductivity=500.0, heat_capacity=1000.0,
        temp_melt=0.0, temp_boil=0.0, temp_ignite=0.0,
        behavior_class=GAS,
        color_r=1.0, color_g=0.6, color_b=0.1,
    ),
    GUNPOWDER: MaterialDef(
        id=GUNPOWDER, name="GUNPOWDER",
        rest_density=4500.0, eos_stiffness=3000.0, eos_gamma=7.0,
        base_viscosity=0.0, friction_coeff=0.5, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=30.0, heat_capacity=800.0,
        temp_melt=0.0, temp_boil=0.0, temp_ignite=480.0,
        behavior_class=GRANULAR,
        color_r=0.2, color_g=0.2, color_b=0.2,
    ),
    WET_SAND: MaterialDef(
        id=WET_SAND, name="WET_SAND",
        rest_density=4500.0, eos_stiffness=5000.0, eos_gamma=7.0,
        base_viscosity=2.0, friction_coeff=0.75, cohesion=0.15,
        buoyancy_extra=0.0, thermal_conductivity=35.0, heat_capacity=1000.0,
        temp_melt=1700.0, temp_boil=2500.0, temp_ignite=0.0,
        behavior_class=GRANULAR,
        color_r=0.55, color_g=0.48, color_b=0.30,
    ),
    MUD: MaterialDef(
        id=MUD, name="MUD",
        rest_density=4500.0, eos_stiffness=300.0, eos_gamma=7.0,
        base_viscosity=50.0, friction_coeff=0.0, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=40.0, heat_capacity=1200.0,
        temp_melt=1400.0, temp_boil=2200.0, temp_ignite=0.0,
        behavior_class=FLUID,
        color_r=0.35, color_g=0.22, color_b=0.10,
    ),
    MAT_RIGID: MaterialDef(
        id=MAT_RIGID, name="RIGID",
        rest_density=2500.0, eos_stiffness=0.0, eos_gamma=0.0,
        base_viscosity=0.0, friction_coeff=0.5, cohesion=0.0,
        buoyancy_extra=0.0, thermal_conductivity=200.0, heat_capacity=800.0,
        temp_melt=0.0, temp_boil=0.0, temp_ignite=0.0,
        behavior_class=STATIC,
        color_r=0.6, color_g=0.6, color_b=0.65,
    ),
}

assert len(MATERIALS) == NUM_DEFINED

# ---------------------------------------------------------------------------
# 32x32 Interaction matrix
#
# Each (i, j) pair stores reaction_rate and heat_exchange.
# Only non-trivial pairs are listed; everything else defaults to zero.
# Symmetric: both (a,b) and (b,a) are set.
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
#   water + ice    -> melting
# ---------------------------------------------------------------------------

# (mat_a, mat_b): (reaction_rate, heat_exchange)
_INTERACTION_PAIRS: List[Tuple[int, int, float, float]] = [
    # water interactions
    (WATER, LAVA, 0.5, 50.0),        # water + lava -> steam, cools lava (high heat xfer)
    (WATER, ICE, 0.1, 20.0),         # water near ice -> freezing/melting
    (WATER, FIRE, 0.8, 30.0),        # water extinguishes fire
    (WATER, ACID, 0.05, 5.0),        # water dilutes acid slowly
    # lava interactions
    (LAVA, ICE, 0.6, 60.0),          # lava melts ice rapidly
    (LAVA, WOOD, 0.7, 40.0),         # lava ignites wood
    (LAVA, METAL, 0.05, 30.0),       # lava heats metal
    (LAVA, STONE, 0.0, 20.0),        # lava heats stone (no reaction, just heat)
    (LAVA, OIL, 0.8, 50.0),          # lava ignites oil
    (LAVA, GUNPOWDER, 0.9, 50.0),    # lava detonates gunpowder
    # acid interactions
    (ACID, STONE, 0.15, 3.0),        # acid slowly corrodes stone
    (ACID, METAL, 0.3, 5.0),         # acid corrodes metal
    (ACID, WOOD, 0.2, 2.0),          # acid corrodes wood
    (ACID, DIRT, 0.0, 1.0),          # acid does NOT accumulate corrode-exposure on dirt;
                                       # DIRT->MUD wetting fires on exp_corrode (reactions.cu:230)
                                       # so a non-zero rate would route acid through the WATER
                                       # wetting path unintentionally (bd-unl.19)
    (ACID, ICE, 0.2, 10.0),          # acid melts ice
    # fire interactions
    (FIRE, WOOD, 0.9, 25.0),         # fire burns wood (faster spread)
    (FIRE, OIL, 1.0, 35.0),          # fire burns oil quickly (instant)
    (FIRE, GUNPOWDER, 0.9, 60.0),    # fire detonates gunpowder (higher heat)
    (FIRE, ICE, 0.3, 25.0),          # fire melts ice
    (FIRE, SMOKE, 0.0, 5.0),         # fire produces smoke (heat only)
    # oil interactions
    (OIL, WATER, 0.0, 3.0),          # oil floats on water (no reaction, low heat)
    # ice + steam
    (ICE, STEAM, 0.2, 15.0),         # steam melts ice
    # metal + fire
    (FIRE, METAL, 0.0, 20.0),        # fire heats metal (no reaction, heat only)
    # stone + fire
    (FIRE, STONE, 0.0, 5.0),         # fire heats stone slowly
    # sand wetting interactions
    (SAND, WATER, 0.4, 5.0),         # water wets sand
    (WET_SAND, WATER, 0.6, 5.0),     # water saturates wet sand -> mud
    (WET_SAND, FIRE, 0.0, 25.0),     # fire dries wet sand
    (WET_SAND, LAVA, 0.0, 40.0),     # lava dries wet sand
    (MUD, FIRE, 0.0, 25.0),          # fire dries mud
    (MUD, LAVA, 0.0, 40.0),          # lava dries mud
    (MUD, WATER, 0.3, 3.0),          # mud stays wet near water
    (MUD, ACID, 0.05, 3.0),          # acid corrodes mud
    # dirt + water wetting
    (DIRT, WATER, 0.5, 5.0),         # water wets dirt -> mud
    # lava heats granular materials
    (LAVA, SAND, 0.0, 25.0),         # lava heats sand (can melt)
    (LAVA, DIRT, 0.0, 20.0),         # lava heats dirt
    (LAVA, GRAVEL, 0.0, 20.0),       # lava heats gravel
]


def build_interaction_matrix() -> np.ndarray:
    """Build the 32x32 interaction matrix as a numpy structured array.

    Returns a (32, 32) array of Interaction structs. Interactions are
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

    IDs 0-18 come from MATERIALS; IDs 19-31 are zeroed (reserved).
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
    """Force PTX compilation when the GPU arch exceeds NVRTC's max sm target."""
    from cupy.cuda import compiler as _compiler
    from cupy.cuda import device as _device

    gpu_cc = _device.Device().compute_capability
    nvrtc_max = _compiler._get_max_compute_capability()
    if int(gpu_cc) > int(nvrtc_max):
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
            _compiler._get_arch_for_options_for_nvrtc._cache = {}
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}


def _get_module() -> "object":
    """Compile (or return cached) CuPy RawModule with common.cuh included."""
    global _module
    if _module is None:
        import cupy

        _ensure_ptx_if_needed()
        kernel_dir = os.path.join(os.path.dirname(__file__), "physics", "kernels")
        # Minimal CUDA source that includes common.cuh for its constant memory
        # declarations, plus a test kernel for verification.
        source = r"""
#include "common.cuh"

extern "C" __global__
void test_read_materials(float* out) {
    // out[0] = c_materials[5].rest_density   (water -> 1000.0)
    // out[1] = c_interactions[8][10].reaction_rate  (acid-metal -> 0.3)
    out[0] = c_materials[5].rest_density;
    out[1] = c_interactions[8][10].reaction_rate;
}
"""
        _module = cupy.RawModule(
            code=source,
            options=("--std=c++11", f"-I{kernel_dir}"),
        )
    return _module


def upload_to_gpu(module: Optional[object] = None) -> None:
    """Copy MaterialProps[32] and Interaction[32][32] to CUDA constant memory.

    Parameters
    ----------
    module : CuPy RawModule, optional
        If provided, upload to this module's constant memory symbols.
        If None, uses the internal module from _get_module().
    """
    import cupy

    cudaMemcpyHostToDevice = 1

    if module is None:
        module = _get_module()

    h_materials = build_material_array()
    h_interactions = build_interaction_matrix()

    d_materials_ptr = module.get_global("c_materials")
    d_interactions_ptr = module.get_global("c_interactions")

    cupy.cuda.runtime.memcpy(
        int(d_materials_ptr),
        h_materials.ctypes.data,
        h_materials.nbytes,
        cudaMemcpyHostToDevice,
    )

    cupy.cuda.runtime.memcpy(
        int(d_interactions_ptr),
        h_interactions.ctypes.data,
        h_interactions.nbytes,
        cudaMemcpyHostToDevice,
    )


def get_module() -> "object":
    """Return the compiled CuPy RawModule (public accessor for tests)."""
    return _get_module()
