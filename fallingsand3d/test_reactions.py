"""Integration tests for reactions.py -- K_Reactions kernel.

Acceptance criteria:
  - ICE (mat=11, temp > 273K) -> WATER (mat=5, FLUID)
  - WATER (mat=5, temp < 263K, sleeping or |v| < 0.02 m/s) -> ICE (mat=11, STATIC)
  - LAVA (mat=7, temp < 1000K) -> GRAVEL (mat=4, GRANULAR)
  - WATER (mat=5, temp > 373K) -> set SPAWN_GAS flag
  - WOOD (mat=9, exposure_heat > 500 K) -> FIRE (mat=14, GAS, T=1200K, lifetime=1.0s)
  - OIL (mat=6, exposure_heat > 300 K) -> FIRE (mat=14, GAS, T=1200K, lifetime=1.5s)
  - GUNPOWDER (mat=15, exposure_heat > 100 K) -> FIRE + explosion velocity
  - Corrosion: health -= exposure_corrode * dt; health <= 0 -> DEAD

Exposure scale (bd-r5-review-n28.48): Step1 divides the exposure sums by the
neighbour kernel support weight. exposure_corrode = reaction_rate * reacting
support fraction, in 1/s, at most the largest rate (1.0). exposure_heat =
heat_exchange * max(T_j - T_i, 0) weighted the same way, in K. Tests that feed
exposure directly use values on that scale.
  - GAS lifetime: lifetime -= dt; lifetime <= 0 -> DEAD
  - STEAM (mat=12, temp < 373K) -> WATER (mat=5, FLUID)
  - Exposure accumulators reset to 0 after use (inherent: sorted ephemeral arrays)
  - Block size = 256, kernel runs without errors

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cupy
import numpy as np
import pytest

from step1 import (
    SIM_PARAMS_DTYPE,
    build_sim_params,
)
from materials import (
    FLUID, GRANULAR, GAS, STATIC,
    DEAD, STONE, SAND, DIRT, GRAVEL, WATER, OIL, LAVA, WOOD, METAL, ICE, STEAM, FIRE, GUNPOWDER,
    MUD, WET_SAND,
    build_material_array,
)
from reactions import (
    BLOCK_SIZE,
    get_module,
    compute_reactions,
    upload_sim_params,
    upload_materials,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H = 0.04
SPACING = 0.02
DT = 0.001

# packed_info helpers (mirror common.cuh)
def MAKE_PACKED(mat_id: int, behavior: int) -> int:
    return (mat_id & 0xFF) | ((behavior & 0x3) << 8)

# packed_info bit 21 (physics/kernels/state_dirty.cuh): K_Reactions sets it on
# every particle whose health, lifetime or mass it wrote (bd-r5-review-n28.162).
STATE_DIRTY = 0x200000

def GET_MATERIAL_ID(p: int) -> int:
    return p & 0xFF

def GET_BEHAVIOR(p: int) -> int:
    return (p >> 8) & 0x3

def HAS_SPAWN_FLAG(p: int) -> int:
    return (p >> 11) & 1

# ---------------------------------------------------------------------------
# Helper: upload params for all tests
# ---------------------------------------------------------------------------

def setup_params(dt=DT):
    """Upload SimParams and materials to reactions module's constant memory."""
    sim_params = build_sim_params(
        smoothing_length=H,
        particle_mass=0.008,
        particle_spacing=SPACING,
        gravity=(0.0, -9.8, 0.0),
        dt=dt,
        restitution=0.3,
        wall_friction=0.5,
        world_min=(-1.0, -1.0, -1.0),
        world_max=(1.0, 1.0, 1.0),
    )
    upload_sim_params(sim_params)
    materials = build_material_array()
    upload_materials(materials)


def make_particles(n, mat_id, behavior, temp=293.0, health=1.0, lifetime=0.0,
                   vel=None, exposure_heat=0.0, exposure_corrode=0.0):
    """Create sorted particle arrays for testing reactions kernel.

    exposure_heat (K) and exposure_corrode (1/s) are the normalised Step1
    outputs described in the module docstring.
    """
    packed_info = np.full(n, MAKE_PACKED(mat_id, behavior), dtype=np.uint32)
    temperature = np.full(n, temp, dtype=np.float32)
    hlth = np.full(n, health, dtype=np.float32)
    lt = np.full(n, lifetime, dtype=np.float32)

    if vel is None:
        velocity = np.zeros((n, 4), dtype=np.float32)
    else:
        velocity = np.array(vel, dtype=np.float32).reshape(n, 4)

    exp_heat = np.full(n, exposure_heat, dtype=np.float32)
    exp_corrode = np.full(n, exposure_corrode, dtype=np.float32)

    return {
        "sorted_packed_info": cupy.asarray(packed_info),
        "sorted_temperature": cupy.asarray(temperature),
        "sorted_health": cupy.asarray(hlth),
        "sorted_lifetime": cupy.asarray(lt),
        "sorted_velocity": cupy.asarray(velocity),
        "sorted_exposure_heat": cupy.asarray(exp_heat),
        "sorted_exposure_corrode": cupy.asarray(exp_corrode),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compilation():
    """Kernel compiles without errors."""
    module = get_module()
    kernel = module.get_function("K_Reactions")
    assert kernel is not None
    print("PASS: test_compilation")


def test_block_size():
    """Block size is 256."""
    assert BLOCK_SIZE == 256
    print("PASS: test_block_size")


def test_ice_melts_to_water():
    """ICE at temp > 273K transitions to WATER (FLUID)."""
    setup_params()
    d = make_particles(100, ICE, STATIC, temp=280.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == WATER, f"Particle {i}: expected WATER, got {GET_MATERIAL_ID(int(pi[i]))}"
        assert GET_BEHAVIOR(int(pi[i])) == FLUID, f"Particle {i}: expected FLUID"
    print("PASS: test_ice_melts_to_water")


def test_ice_stays_frozen():
    """ICE at temp < 273K stays as ICE."""
    setup_params()
    d = make_particles(100, ICE, STATIC, temp=200.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == ICE, f"Particle {i}: should stay ICE"
    print("PASS: test_ice_stays_frozen")


def test_water_freezes_to_ice():
    """WATER at rest below 263 K becomes STATIC ICE at 263 K. WATER's
    rest_density is MASS_REF_DENSITY, so the spawn-mass scale is 1 on both
    sides and the mass is unchanged."""
    setup_params()
    n = 100
    d = make_particles(n, WATER, FLUID, temp=262.0)
    mass = cupy.full(n, 0.008, dtype=cupy.float32)
    compute_reactions(**d, sorted_mass=mass)

    pi = d["sorted_packed_info"].get()
    temp = d["sorted_temperature"].get()
    m = mass.get()
    for i in range(n):
        assert GET_MATERIAL_ID(int(pi[i])) == ICE, f"Particle {i}: expected ICE"
        assert GET_BEHAVIOR(int(pi[i])) == STATIC, f"Particle {i}: expected STATIC"
    np.testing.assert_allclose(temp, 263.0)
    np.testing.assert_allclose(m, 0.008, rtol=1e-6)


def test_water_hysteresis_band():
    """WATER at 270 K is inside the 263-273 K band and stays WATER."""
    setup_params()
    d = make_particles(100, WATER, FLUID, temp=270.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == WATER, f"Particle {i}: should stay WATER"
        assert GET_BEHAVIOR(int(pi[i])) == FLUID


def test_moving_water_does_not_freeze():
    """Cold WATER moving faster than V_WAKE (0.02 m/s) stays WATER, so a
    stream does not turn into STATIC ice in mid-air. The same particle
    freezes when its sleep flag is set."""
    setup_params()
    n = 2
    d = make_particles(n, WATER, FLUID, temp=250.0,
                       vel=[[0.0, -1.0, 0.0, 0.0]] * n)
    pi = d["sorted_packed_info"].get()
    pi[1] |= 0x400  # SET_SLEEPING
    d["sorted_packed_info"] = cupy.asarray(pi)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    v = d["sorted_velocity"].get()
    assert GET_MATERIAL_ID(int(pi[0])) == WATER, "moving cold water froze"
    assert GET_MATERIAL_ID(int(pi[1])) == ICE, "sleeping cold water did not freeze"
    assert np.all(v[1, :3] == 0.0), "frozen particle kept its velocity"


def test_lava_solidifies_to_stone():
    """LAVA below its temp_melt (1000 K) becomes GRAVEL (GRANULAR), not STATIC
    STONE, so cooled lava still falls until supported (bd-r5-review-n28.112)."""
    setup_params()
    d = make_particles(100, LAVA, FLUID, temp=800.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == GRAVEL, f"Particle {i}: expected GRAVEL"
        assert GET_BEHAVIOR(int(pi[i])) == GRANULAR, f"Particle {i}: expected GRANULAR"
    print("PASS: test_lava_solidifies_to_stone")


def test_cooled_lava_falls_to_floor_and_sleeps():
    """A cooled lava particle 0.5 m above the floor falls, lands and sleeps.

    Full pipeline (step1, reactions, step2, integrate). The particle starts
    as LAVA at 990 K, below temp_melt, so the first substep turns it into
    GRAVEL. It must reach the floor (y = -1.0) and sleep within 1 s of
    landing (bd-r5-review-n28.112). Before the fix it became STATIC STONE
    and stayed at y = -0.5.
    """
    from test_integration_15mat import (
        setup_all_modules, run_full_pipeline_step, MATERIALS as _MATS,
    )
    from counting_sort import allocate_scratch
    from grid_params import NUM_CELLS, build_grid_params

    setup_all_modules()
    grid_params = build_grid_params()
    scratch = allocate_scratch(NUM_CELLS, 0)

    n = 1
    pos = np.array([[0.0, -0.5, 0.0, 1.0]], dtype=np.float32)
    w = {
        "position": cupy.asarray(pos),
        "velocity": cupy.zeros((n, 4), dtype=cupy.float32),
        "veleval": cupy.zeros((n, 4), dtype=cupy.float32),
        "mass": cupy.full(n, _MATS[LAVA].rest_density * SPACING**3, dtype=cupy.float32),
        "packed_info": cupy.full(n, MAKE_PACKED(LAVA, FLUID), dtype=cupy.uint32),
        "temperature": cupy.full(n, 990.0, dtype=cupy.float32),
        "health": cupy.full(n, 1.0, dtype=cupy.float32),
        "lifetime": cupy.zeros(n, dtype=cupy.float32),
        "sleep_counter": cupy.zeros(n, dtype=cupy.uint8),
        "shear_rate": cupy.zeros(n, dtype=cupy.float32),
        "color": cupy.zeros((n, 4), dtype=cupy.float32),
    }

    # Free fall of 0.5 m takes about 0.32 s at g = 9.8. Stop at the first
    # substep the particle is within 2 spacings of the floor.
    landed_step = None
    for step in range(1000):
        w = run_full_pipeline_step(w, grid_params, scratch, frame=step)
        if float(w["position"][0, 1].get()) < -1.0 + 2 * SPACING:
            landed_step = step
            break
    pi = int(w["packed_info"].get()[0])
    assert GET_MATERIAL_ID(pi) == GRAVEL, f"expected GRAVEL, got material {GET_MATERIAL_ID(pi)}"
    assert GET_BEHAVIOR(pi) == GRANULAR
    assert landed_step is not None, (
        f"cooled lava did not reach the floor, y = {float(w['position'][0, 1].get()):.3f}")

    # Sleep within 1 s (1000 substeps at dt = 1 ms) after landing.
    slept = False
    for step in range(1000):
        w = run_full_pipeline_step(w, grid_params, scratch,
                                   frame=landed_step + 1 + step)
        if (int(w["packed_info"].get()[0]) >> 10) & 1:
            slept = True
            break
    y = float(w["position"][0, 1].get())
    assert slept, f"cooled lava on the floor did not sleep within 1 s (y = {y:.4f})"
    assert y < -1.0 + 2 * SPACING, f"sleeping particle left the floor, y = {y:.4f}"


def test_lava_stays_liquid():
    """LAVA at temp > 900K stays as LAVA."""
    setup_params()
    d = make_particles(100, LAVA, FLUID, temp=1500.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == LAVA, f"Particle {i}: should stay LAVA"
    print("PASS: test_lava_stays_liquid")


def test_water_boils_sets_spawn_flag():
    """WATER at temp > 373K gets SPAWN_GAS flag set."""
    setup_params()
    d = make_particles(100, WATER, FLUID, temp=400.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == WATER, f"Particle {i}: should still be WATER"
        assert HAS_SPAWN_FLAG(int(pi[i])) == 1, f"Particle {i}: SPAWN_GAS flag not set"
    print("PASS: test_water_boils_sets_spawn_flag")


def test_water_cool_no_spawn():
    """WATER at temp < 373K does NOT get SPAWN_GAS flag."""
    setup_params()
    d = make_particles(100, WATER, FLUID, temp=300.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert HAS_SPAWN_FLAG(int(pi[i])) == 0, f"Particle {i}: should not have SPAWN_GAS"
    print("PASS: test_water_cool_no_spawn")


def test_steam_condenses_to_water():
    """STEAM at temp < 373K transitions to WATER (FLUID)."""
    setup_params()
    d = make_particles(100, STEAM, GAS, temp=350.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == WATER, f"Particle {i}: expected WATER"
        assert GET_BEHAVIOR(int(pi[i])) == FLUID, f"Particle {i}: expected FLUID"
    print("PASS: test_steam_condenses_to_water")


def test_steam_stays_gas():
    """STEAM at temp > 373K stays as STEAM."""
    setup_params()
    d = make_particles(100, STEAM, GAS, temp=400.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == STEAM, f"Particle {i}: should stay STEAM"
    print("PASS: test_steam_stays_gas")


def test_wood_ignites():
    """WOOD with exposure_heat > 500 K transitions to FIRE (GAS, T=1200K, lifetime=1.0s)."""
    setup_params()
    d = make_particles(100, WOOD, STATIC, exposure_heat=600.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    temp = d["sorted_temperature"].get()
    lt = d["sorted_lifetime"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == FIRE, f"Particle {i}: expected FIRE"
        assert GET_BEHAVIOR(int(pi[i])) == GAS, f"Particle {i}: expected GAS"
        assert abs(temp[i] - 1200.0) < 1.0, f"Particle {i}: temp={temp[i]}, expected 1200"
        assert abs(lt[i] - 1.0) < 0.01, f"Particle {i}: lifetime={lt[i]}, expected 1.0"
    print("PASS: test_wood_ignites")


def test_wood_no_ignite():
    """WOOD with exposure_heat < 500 K stays as WOOD."""
    setup_params()
    d = make_particles(100, WOOD, STATIC, exposure_heat=300.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == WOOD, f"Particle {i}: should stay WOOD"
    print("PASS: test_wood_no_ignite")


def test_oil_ignites():
    """OIL with exposure_heat > 300 K transitions to FIRE (GAS, T=1200K, lifetime=1.5s)."""
    setup_params()
    d = make_particles(100, OIL, FLUID, exposure_heat=400.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    temp = d["sorted_temperature"].get()
    lt = d["sorted_lifetime"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == FIRE, f"Particle {i}: expected FIRE"
        assert GET_BEHAVIOR(int(pi[i])) == GAS, f"Particle {i}: expected GAS"
        assert abs(temp[i] - 1200.0) < 1.0, f"Particle {i}: temp={temp[i]}, expected 1200"
        assert abs(lt[i] - 1.5) < 0.01, f"Particle {i}: lifetime={lt[i]}, expected 1.5"
    print("PASS: test_oil_ignites")


def test_gunpowder_explodes():
    """GUNPOWDER with exposure_heat > 100 K transitions to FIRE + random velocity burst."""
    setup_params()
    d = make_particles(100, GUNPOWDER, GRANULAR, exposure_heat=200.0)
    compute_reactions(**d, frame=42)

    pi = d["sorted_packed_info"].get()
    temp = d["sorted_temperature"].get()
    lt = d["sorted_lifetime"].get()
    vel = d["sorted_velocity"].get()

    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == FIRE, f"Particle {i}: expected FIRE"
        assert GET_BEHAVIOR(int(pi[i])) == GAS, f"Particle {i}: expected GAS"
        assert abs(temp[i] - 1200.0) < 1.0
        assert abs(lt[i] - 0.3) < 0.01

    # Check velocity burst: at least some particles have non-zero velocity
    vel_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
    assert np.all(vel_mag > 1.0), "All particles should have explosion velocity"
    # Velocities should vary (random directions)
    assert np.std(vel[:, 0]) > 0.1, "X velocities should vary due to random RNG"
    print("PASS: test_gunpowder_explodes")


def test_corrosion_reduces_health():
    """Corrosion reduces health by exposure_corrode * dt."""
    setup_params(dt=0.01)
    d = make_particles(100, METAL, STATIC, health=1.0, exposure_corrode=5.0)
    compute_reactions(**d)

    hlth = d["sorted_health"].get()
    # health should be 1.0 - 5.0 * 0.01 = 0.95
    for i in range(100):
        assert abs(hlth[i] - 0.95) < 0.01, f"Particle {i}: health={hlth[i]}, expected ~0.95"
    # Material should still be METAL
    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == METAL
    print("PASS: test_corrosion_reduces_health")


def test_corrosion_kills_particle():
    """Corrosion brings health to 0 -- the particle becomes DEAD (no FIRE spark)."""
    setup_params(dt=0.1)
    d = make_particles(100, METAL, STATIC, health=0.05, exposure_corrode=1.0)
    compute_reactions(**d)

    hlth = d["sorted_health"].get()
    pi = d["sorted_packed_info"].get()
    # health = 0.05 - 1.0 * 0.1 = -0.05 -> DEAD|STATIC, health 0
    for i in range(100):
        assert int(pi[i]) == (MAKE_PACKED(DEAD, STATIC) | STATE_DIRTY), \
            f"Particle {i}: expected DEAD, got mat_id={GET_MATERIAL_ID(int(pi[i]))}"
        assert hlth[i] == 0.0, f"Particle {i}: health={hlth[i]}, expected 0"
    print("PASS: test_corrosion_kills_particle")


def test_gas_lifetime_decay():
    """GAS particles with lifetime > 0 have it decremented by dt."""
    setup_params(dt=0.01)
    d = make_particles(100, FIRE, GAS, lifetime=1.0)
    compute_reactions(**d)

    lt = d["sorted_lifetime"].get()
    # lifetime should be 1.0 - 0.01 = 0.99
    for i in range(100):
        assert abs(lt[i] - 0.99) < 0.001, f"Particle {i}: lifetime={lt[i]}, expected ~0.99"
    # Should still be FIRE
    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == FIRE
    print("PASS: test_gas_lifetime_decay")


def test_gas_lifetime_expires():
    """FIRE GAS particle with expired lifetime transitions to SMOKE (not DEAD)."""
    setup_params(dt=0.1)
    d = make_particles(100, FIRE, GAS, lifetime=0.05)
    compute_reactions(**d)

    lt = d["sorted_lifetime"].get()
    pi = d["sorted_packed_info"].get()
    # FIRE lifetime expiry: kernel converts FIRE -> SMOKE with new lifetime=3.0s
    # (other GAS types would go to DEAD, but FIRE specifically becomes smoke)
    from materials import SMOKE
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == SMOKE, \
            f"Particle {i}: expected SMOKE after FIRE expires, got mat_id={GET_MATERIAL_ID(int(pi[i]))}"
        assert GET_BEHAVIOR(int(pi[i])) == GAS, f"Particle {i}: expected GAS"
        assert abs(lt[i] - 3.0) < 0.1, f"Particle {i}: lifetime={lt[i]}, expected 3.0s (smoke)"
    print("PASS: test_gas_lifetime_expires")


def test_dead_particles_unchanged():
    """DEAD particles are skipped entirely."""
    setup_params()
    d = make_particles(100, DEAD, STATIC, exposure_heat=1.0, exposure_corrode=1.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    hlth = d["sorted_health"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == DEAD, f"Particle {i}: should stay DEAD"
        assert hlth[i] == 1.0, f"Particle {i}: health should be unchanged"
    print("PASS: test_dead_particles_unchanged")


def test_ice_lava_together():
    """Ice and lava together: ice melts to water, lava solidifies to gravel."""
    setup_params()
    n = 200
    packed_info = np.zeros(n, dtype=np.uint32)
    temperature = np.zeros(n, dtype=np.float32)
    health = np.ones(n, dtype=np.float32)
    lifetime = np.zeros(n, dtype=np.float32)
    velocity = np.zeros((n, 4), dtype=np.float32)
    exp_heat = np.zeros(n, dtype=np.float32)
    exp_corrode = np.zeros(n, dtype=np.float32)

    # First 100: ICE at 300K (above 273K -> should melt)
    for i in range(100):
        packed_info[i] = MAKE_PACKED(ICE, STATIC)
        temperature[i] = 300.0

    # Next 100: LAVA at 800K (below 900K -> should solidify)
    for i in range(100, 200):
        packed_info[i] = MAKE_PACKED(LAVA, FLUID)
        temperature[i] = 800.0

    d = {
        "sorted_packed_info": cupy.asarray(packed_info),
        "sorted_temperature": cupy.asarray(temperature),
        "sorted_health": cupy.asarray(health),
        "sorted_lifetime": cupy.asarray(lifetime),
        "sorted_velocity": cupy.asarray(velocity),
        "sorted_exposure_heat": cupy.asarray(exp_heat),
        "sorted_exposure_corrode": cupy.asarray(exp_corrode),
    }
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    # First 100: ICE -> WATER
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == WATER, f"ICE particle {i}: expected WATER"
        assert GET_BEHAVIOR(int(pi[i])) == FLUID

    # Next 100: LAVA -> GRAVEL
    for i in range(100, 200):
        assert GET_MATERIAL_ID(int(pi[i])) == GRAVEL, f"LAVA particle {i}: expected GRAVEL"
        assert GET_BEHAVIOR(int(pi[i])) == GRANULAR

    print("PASS: test_ice_lava_together")


def test_acid_metal_corrosion():
    """Acid near metal: after enough corrosion steps the metal particle is DEAD."""
    setup_params(dt=0.001)
    d = make_particles(100, METAL, STATIC, health=1.0, exposure_corrode=10.0)

    # At dt=0.001 and exp_corrode=10.0, damage=0.01/step -> health=0 near step 100
    for step in range(110):
        # Re-apply exposure each step (simulating continuous acid contact)
        d["sorted_exposure_corrode"] = cupy.full(100, 10.0, dtype=cupy.float32)
        compute_reactions(**d, frame=step)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert int(pi[i]) == (MAKE_PACKED(DEAD, STATIC) | STATE_DIRTY), \
            f"Particle {i}: expected DEAD after corrosion, got mat_id={GET_MATERIAL_ID(int(pi[i]))}"
    print("PASS: test_acid_metal_corrosion")


def test_500k_stress():
    """500K particles run without errors or NaN."""
    setup_params()
    n = 500_000

    # Mix of materials
    packed_info = np.zeros(n, dtype=np.uint32)
    temperature = np.full(n, 293.0, dtype=np.float32)
    health = np.ones(n, dtype=np.float32)
    lifetime = np.zeros(n, dtype=np.float32)
    velocity = np.zeros((n, 4), dtype=np.float32)
    exp_heat = np.zeros(n, dtype=np.float32)
    exp_corrode = np.zeros(n, dtype=np.float32)

    chunk = n // 5
    # Water at room temp
    packed_info[:chunk] = MAKE_PACKED(WATER, FLUID)
    # Sand
    packed_info[chunk:2*chunk] = MAKE_PACKED(SAND, GRANULAR)
    # ICE at 280K (should melt)
    packed_info[2*chunk:3*chunk] = MAKE_PACKED(ICE, STATIC)
    temperature[2*chunk:3*chunk] = 280.0
    # FIRE with lifetime
    packed_info[3*chunk:4*chunk] = MAKE_PACKED(FIRE, GAS)
    lifetime[3*chunk:4*chunk] = 1.0
    # METAL with some corrosion
    packed_info[4*chunk:] = MAKE_PACKED(METAL, STATIC)
    exp_corrode[4*chunk:] = 0.5

    d = {
        "sorted_packed_info": cupy.asarray(packed_info),
        "sorted_temperature": cupy.asarray(temperature),
        "sorted_health": cupy.asarray(health),
        "sorted_lifetime": cupy.asarray(lifetime),
        "sorted_velocity": cupy.asarray(velocity),
        "sorted_exposure_heat": cupy.asarray(exp_heat),
        "sorted_exposure_corrode": cupy.asarray(exp_corrode),
    }

    compute_reactions(**d, frame=0)
    cupy.cuda.Device().synchronize()

    # Check no NaN in temperature
    temp_out = d["sorted_temperature"].get()
    assert not np.any(np.isnan(temp_out)), "NaN in temperature"

    # Check no NaN in health
    hlth_out = d["sorted_health"].get()
    assert not np.any(np.isnan(hlth_out)), "NaN in health"

    # Check ICE melted
    pi = d["sorted_packed_info"].get()
    for i in range(2*chunk, 3*chunk):
        assert GET_MATERIAL_ID(int(pi[i])) == WATER, f"ICE particle {i}: expected WATER"

    # Check FIRE lifetime decremented
    lt = d["sorted_lifetime"].get()
    for i in range(3*chunk, 4*chunk):
        assert abs(lt[i] - 0.999) < 0.01, f"FIRE particle {i}: lifetime={lt[i]}"

    print("PASS: test_500k_stress")


def test_dirt_water_becomes_mud():
    """bd-mzc.27 regression: DIRT with WATER corrode exposure transitions to MUD,
    not to a corrosion (crumble) state.

    The DIRT wetting check fires before the corrosion whitelist, so DIRT with
    exp_corrode > SAND_WET_THRESHOLD (0.2) becomes MAKE_PACKED(MAT_MUD, FLUID).
    """
    setup_params()
    # DIRT particles with enough water-corrode exposure to trigger the wetting transition
    d = make_particles(100, DIRT, GRANULAR, exposure_corrode=0.5)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    hlth = d["sorted_health"].get()
    for i in range(100):
        mat = GET_MATERIAL_ID(int(pi[i]))
        assert mat == MUD, (
            f"Particle {i}: DIRT+WATER should become MUD (mat={MUD}), "
            f"got mat_id={mat}. Old code produced corrosion spark."
        )
        assert GET_BEHAVIOR(int(pi[i])) == FLUID, (
            f"Particle {i}: MUD should be FLUID, got behavior={GET_BEHAVIOR(int(pi[i]))}"
        )
        # Health should be unmodified (wetting path does not damage health)
        assert abs(hlth[i] - 1.0) < 1e-6, (
            f"Particle {i}: health={hlth[i]}, expected 1.0 after wetting (not corrosion)"
        )
    print("PASS: test_dirt_water_becomes_mud")


def test_dirt_no_corrode_below_threshold():
    """DIRT with exp_corrode below the wetting threshold stays as DIRT."""
    setup_params()
    d = make_particles(100, DIRT, GRANULAR, exposure_corrode=0.1)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        mat = GET_MATERIAL_ID(int(pi[i]))
        assert mat == DIRT, f"Particle {i}: DIRT should stay DIRT below threshold, got mat_id={mat}"
    print("PASS: test_dirt_no_corrode_below_threshold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# bd-r5-review-n28.23: dead slots are massless and not walls
# ---------------------------------------------------------------------------

def test_death_zeroes_mass():
    """Consumed acid, expired gas and existing DEAD slots all end with mass 0.
    Live particles keep their mass."""
    from materials import ACID, SMOKE
    setup_params(dt=0.001)
    n = 4
    d = make_particles(n, WATER, FLUID)
    pi = np.array([
        MAKE_PACKED(ACID, FLUID),     # consumed this call
        MAKE_PACKED(SMOKE, GAS),      # expires this call
        MAKE_PACKED(DEAD, FLUID),     # already dead (brush-kill encoding)
        MAKE_PACKED(WATER, FLUID),    # stays alive
    ], dtype=np.uint32)
    d["sorted_packed_info"] = cupy.asarray(pi)
    d["sorted_health"] = cupy.asarray(np.array([0.001, 1.0, 0.0, 1.0], np.float32))
    d["sorted_lifetime"] = cupy.asarray(np.array([0.0, 0.0005, 0.0, 0.0], np.float32))
    d["sorted_exposure_corrode"] = cupy.asarray(np.array([10.0, 0.0, 0.0, 0.0], np.float32))
    mass = cupy.full(n, 0.02, dtype=cupy.float32)
    compute_reactions(**d, sorted_mass=mass)
    out_pi = d["sorted_packed_info"].get()
    out_m = mass.get()
    for i in (0, 1):
        assert int(out_pi[i]) == (MAKE_PACKED(DEAD, STATIC) | STATE_DIRTY), f"slot {i}: packed {int(out_pi[i]):#x}"
    for i in (0, 1, 2):
        assert out_m[i] == 0.0, f"slot {i}: mass {out_m[i]} after death"
    assert out_m[3] == np.float32(0.02), "live WATER mass changed"
    # The dead slot's nonzero mass was cleared, so K_Integrate must carry it
    # back; the untouched WATER needs no carry-back.
    assert int(out_pi[2]) & STATE_DIRTY, "cleared dead-slot mass not marked dirty"
    assert int(out_pi[3]) == MAKE_PACKED(WATER, FLUID), f"packed {int(out_pi[3]):#x}"


def test_death_without_mass_buffer_still_works():
    """sorted_mass is optional: the kernel skips the write on a null pointer."""
    from materials import SMOKE
    setup_params(dt=0.001)
    d = make_particles(8, SMOKE, GAS, lifetime=0.0005)
    compute_reactions(**d)
    pi = d["sorted_packed_info"].get()
    assert all(int(p) == (MAKE_PACKED(DEAD, STATIC) | STATE_DIRTY) for p in pi)


def _boundary_push(neighbor_packed: int) -> tuple:
    """Integrate one FLUID particle 0.3h from one STATIC neighbour with mass 0.02.
    Return the fluid particle's (position, velocity) after one K_Integrate."""
    import integrate as integ
    from grid_params import build_grid_params, CELL_SIZE, TABLE_MASK

    sim_params = build_sim_params(
        smoothing_length=H, particle_mass=0.008, particle_spacing=SPACING,
        gravity=(0.0, 0.0, 0.0), dt=DT, restitution=0.3, wall_friction=0.5,
        world_min=(-1.0, -1.0, -1.0), world_max=(1.0, 1.0, 1.0),
    )
    integ.upload_sim_params(sim_params)
    integ.upload_materials(build_material_array())
    gp = build_grid_params()
    mod = integ.get_module()
    ptr = mod.get_global("c_grid")
    ptr.copy_from_host(gp.ctypes.data, gp.nbytes)

    # Both particles in the same grid cell; the fluid moves toward the neighbour.
    pos = np.array([[0.005, 0.005, 0.005, 1.0],
                    [0.005 + 0.3 * H, 0.005, 0.005, 1.0]], dtype=np.float32)
    vel = np.array([[0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    cell = np.floor((pos[0, :3] + 1.0) / float(CELL_SIZE)).astype(np.int64)
    assert np.array_equal(cell, np.floor((pos[1, :3] + 1.0) / float(CELL_SIZE)).astype(np.int64))
    hsh = int((int(cell[0]) * 73856093 + int(cell[1]) * 19349669
               + int(cell[2]) * 83492791) & 0xFFFFFFFF) & TABLE_MASK
    table = int(gp[0]["table_size"])
    cell_start = np.full(table, 0xFFFFFFFF, dtype=np.uint32)
    cell_end = np.zeros(table, dtype=np.uint32)
    cell_start[hsh] = 0
    cell_end[hsh] = 2

    packed = np.array([MAKE_PACKED(WATER, FLUID), neighbor_packed], dtype=np.uint32)
    pos_out, vel_out, *_ = integ.integrate(
        sorted_position=cupy.asarray(pos),
        sorted_velocity=cupy.asarray(vel),
        sorted_veleval=cupy.asarray(vel),
        sorted_sph_force=cupy.zeros((2, 4), dtype=cupy.float32),
        sorted_mass=cupy.asarray(np.array([0.008, 0.02], np.float32)),
        sorted_packed_info=cupy.asarray(packed),
        sorted_temperature=cupy.full(2, 293.0, dtype=cupy.float32),
        sorted_health=cupy.ones(2, dtype=cupy.float32),
        sort_indexes=cupy.arange(2, dtype=cupy.uint32),
        cell_start=cupy.asarray(cell_start),
        cell_end=cupy.asarray(cell_end),
    )
    return pos_out.get()[0, :3], vel_out.get()[0, :3], pos[0, :3], vel[0, :3]


def test_static_boundary_skips_dead_slot():
    """A MAT_DEAD/STATIC neighbour with mass 0.02 at 0.3h gives zero push and
    no velocity reflection. A STONE neighbour at the same place does push
    (control, proves the test reaches static_particle_boundary). The push band
    is 0.5h (bd-r5-review-n28.43), so the neighbour sits inside it."""
    p_dead, v_dead, p0, v0 = _boundary_push(MAKE_PACKED(DEAD, STATIC))
    free_pos = p0 + DT * v0
    assert np.allclose(p_dead, free_pos, atol=1e-6), f"dead slot pushed fluid: {p_dead} vs {free_pos}"
    assert np.allclose(v_dead, v0, atol=1e-6), f"dead slot changed velocity: {v_dead}"

    p_stone, v_stone, _, _ = _boundary_push(MAKE_PACKED(STONE, STATIC))
    assert p_stone[0] < free_pos[0] - 1e-3, f"STONE control did not push: {p_stone}"
    assert v_stone[0] < 0.0, f"STONE control did not reflect velocity: {v_stone}"


def test_expired_gas_mass_stays_zero_in_simulation():
    """End to end: gas that expires inside Simulation keeps mass 0 in the
    unsorted buffer over 10 substeps (sort, reuse and scatter-back paths)."""
    from materials import SMOKE, MATERIALS
    from world import World
    from simulation import Simulation

    world = World(max_particles=20_000)
    world.packed_info[:] = 0
    world._high_water = 0
    n_water = world.spawn_cube((-0.1, 0.3, -0.1), (0.1, 0.4, 0.1), WATER, spacing=0.02)
    n_smoke = world.spawn_cube((-0.05, 0.42, -0.05), (0.05, 0.46, 0.05), SMOKE, spacing=0.02)
    n = n_water + n_smoke
    assert n_smoke > 0
    world.lifetime[n_water:n] = 0.0005
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=False, max_substeps=20)
    sim.sim_time = 0.0
    sim._last_frame_time = None
    for _ in range(10):
        sim._sim_step(n)
    cupy.cuda.Device().synchronize()
    pi = world.packed_info[:n].get()
    m = world.mass[:n].get()
    dead = (pi & 0xFF) == DEAD
    assert dead[n_water:].all(), "expired SMOKE slots not all DEAD"
    assert not dead[:n_water].any(), "water died unexpectedly"
    assert np.all(m[dead] == 0.0), f"dead slot mass {m[dead].max()}"
    assert np.all(m[~dead] > 0.0), "live particle lost its mass"


# ---------------------------------------------------------------------------
# bd-r5-review-n28.38: corrosion is acid damage only, and a corroded
# particle is removed instead of becoming a 400 K FIRE spark.
# ---------------------------------------------------------------------------

# Materials that burn (ignite on exposure_heat) and may take fire/lava damage.
_COMBUSTIBLES = (WOOD, OIL, GUNPOWDER)


def test_ice_ignores_corrode_exposure():
    """(a) ICE below 273 K keeps full health even with a large corrode exposure.
    ICE only melts by temperature; it is not on the corrosion whitelist."""
    setup_params(dt=0.001)
    d = make_particles(64, ICE, STATIC, temp=253.0, exposure_corrode=1e4)
    for step in range(10):
        compute_reactions(**d, frame=step)
    pi = d["sorted_packed_info"].get()
    hlth = d["sorted_health"].get()
    assert all(GET_MATERIAL_ID(int(p)) == ICE for p in pi), "ICE changed material"
    assert np.all(hlth == 1.0), f"ICE lost health: min={hlth.min()}"


def test_corroded_stone_becomes_dead_not_fire():
    """(b) A fully corroded STONE particle becomes DEAD/STATIC with mass 0 and
    joins the spawn freelist. It never becomes FIRE."""
    from materials import STONE
    setup_params(dt=0.001)
    n = 32
    d = make_particles(n, STONE, STATIC, health=0.0005, exposure_corrode=1.0)
    mass = cupy.full(n, 0.02, dtype=cupy.float32)
    dead_indices = cupy.zeros(n, dtype=cupy.uint32)
    dead_count = cupy.zeros(1, dtype=cupy.uint32)
    compute_reactions(**d, sorted_mass=mass, dead_indices=dead_indices,
                      dead_count=dead_count)
    pi = d["sorted_packed_info"].get()
    for p in pi:
        assert int(p) == (MAKE_PACKED(DEAD, STATIC) | STATE_DIRTY), f"packed {int(p):#x}, expected DEAD|STATIC"
    assert np.all(mass.get() == 0.0), "corroded slot kept its mass"
    assert np.all(d["sorted_health"].get() == 0.0)
    assert int(dead_count.get()[0]) == n
    assert sorted(dead_indices.get().tolist()) == list(range(n))


def test_non_zero_reaction_rate_pairs_are_acid_water_mud_or_combustible():
    """(c) Every pair with a non-zero reaction_rate has ACID, WATER, MUD or a
    combustible on one side. Heat-only pairs (ICE, LAVA/METAL, WATER/ACID)
    must have rate 0 so they never feed the corrosion branch."""
    from materials import ACID, MUD, MAX_MATERIALS, build_interaction_matrix
    m = build_interaction_matrix()
    allowed = {ACID, WATER, MUD, *_COMBUSTIBLES}
    for a in range(MAX_MATERIALS):
        for b in range(MAX_MATERIALS):
            if m[a][b]["reaction_rate"] != 0.0:
                assert a in allowed or b in allowed, \
                    f"pair ({a}, {b}) has reaction_rate {m[a][b]['reaction_rate']}"
    for a, b in ((WATER, ICE), (LAVA, ICE), (FIRE, ICE), (ICE, STEAM),
                 (LAVA, METAL), (WATER, ACID)):
        assert m[a][b]["reaction_rate"] == 0.0, f"({a}, {b}) should be heat-only"
        assert m[a][b]["heat_exchange"] > 0.0, f"({a}, {b}) lost its heat exchange"


def _step1_exposure(packed, temps, masses, offsets):
    """Run the grid build + K_Step1 for particles at H*offsets (x axis) and
    return (exposure_heat, exposure_corrode) for particle 0, found by its
    unique mass in the sorted output."""
    from test_step1 import _run_full_pipeline, _upload_all_params
    _upload_all_params()
    n = len(packed)
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 3] = 1.0
    for k, (dx, dy, dz) in enumerate(offsets):
        pos[k, :3] = (0.005 + dx * H, 0.005 + dy * H, 0.005 + dz * H)
    mass = np.asarray(masses, dtype=np.float32)
    _, _, _, exp_h, exp_c, _, s_mass = _run_full_pipeline(
        pos, mass, None, np.asarray(packed, np.uint32), None,
        np.asarray(temps, np.float32))
    k0 = int(np.argmin(np.abs(s_mass - mass[0])))
    assert abs(s_mass[k0] - mass[0]) < 1e-9
    return float(exp_h[k0]), float(exp_c[k0])


_RING = [(0.0, 0.0, 0.0), (0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0), (0, -0.5, 0),
         (0, 0, 0.5), (0, 0, -0.5)]


def _ring_masses(centre):
    return [centre] + [0.008] * (len(_RING) - 1)


def test_step1_ice_in_water_has_no_corrode_exposure():
    """ICE at 253 K surrounded by 293 K WATER: step1 gives zero corrode
    exposure and positive heat exposure, and ten reaction substeps leave it ICE."""
    packed = [MAKE_PACKED(ICE, STATIC)] + [MAKE_PACKED(WATER, FLUID)] * 6
    temps = [253.0] + [293.0] * 6
    exp_h, exp_c = _step1_exposure(packed, temps, _ring_masses(0.00734), _RING)
    assert exp_c == 0.0, f"ICE exposure_corrode {exp_c}"
    assert exp_h > 0.0, "ICE should still receive heat exposure from warm water"
    setup_params(dt=0.001)
    d = make_particles(1, ICE, STATIC, temp=253.0, exposure_heat=exp_h,
                       exposure_corrode=exp_c)
    for step in range(10):
        compute_reactions(**d, frame=step)
    assert GET_MATERIAL_ID(int(d["sorted_packed_info"].get()[0])) == ICE
    assert float(d["sorted_health"].get()[0]) == 1.0


def test_step1_metal_by_lava_and_acid_in_water_have_no_corrode_exposure():
    """METAL next to LAVA and ACID inside WATER get zero corrode exposure.
    ACID next to STONE still does (positive control)."""
    from materials import ACID
    packed = [MAKE_PACKED(METAL, STATIC)] + [MAKE_PACKED(LAVA, FLUID)] * 6
    _, exp_c = _step1_exposure(packed, [293.0] + [1400.0] * 6,
                               _ring_masses(0.0624), _RING)
    assert exp_c == 0.0, f"METAL next to LAVA exposure_corrode {exp_c}"

    packed = [MAKE_PACKED(ACID, FLUID)] + [MAKE_PACKED(WATER, FLUID)] * 6
    _, exp_c = _step1_exposure(packed, [293.0] * 7, _ring_masses(0.0096), _RING)
    assert exp_c == 0.0, f"ACID in WATER exposure_corrode {exp_c}"

    packed = [MAKE_PACKED(STONE, STATIC)] + [MAKE_PACKED(ACID, FLUID)] * 6
    _, exp_c = _step1_exposure(packed, [293.0] * 7, _ring_masses(0.0208), _RING)
    assert exp_c > 0.0, "ACID must still corrode STONE"


def test_wood_next_to_dead_slot_gets_no_heat_exposure():
    """A dead slot left by corrosion (even if its temperature is 400 K)
    gives a WOOD neighbour no heat exposure, so it cannot ignite it."""
    packed = [MAKE_PACKED(WOOD, STATIC)] + [MAKE_PACKED(DEAD, STATIC)] * 6
    exp_h, _ = _step1_exposure(packed, [293.0] + [400.0] * 6,
                               [0.0048] + [0.0] * 6, _RING)
    assert exp_h == 0.0, f"WOOD exposure_heat {exp_h} from dead slots"


def test_stone_in_acid_survives_about_one_over_rate():
    """bd-r5-review-n28.48: STONE fully surrounded by ACID gets exposure_corrode
    equal to the ACID/STONE rate (0.15/s) from step1, so it survives about
    1/0.15 = 6.7 s instead of dying in one substep."""
    from materials import ACID
    packed = [MAKE_PACKED(STONE, STATIC)] + [MAKE_PACKED(ACID, FLUID)] * 6
    _, exp_c = _step1_exposure(packed, [293.0] * 7, _ring_masses(0.0208), _RING)
    assert abs(exp_c - 0.15) < 1e-4, f"STONE exposure_corrode {exp_c}, expected 0.15"

    dt = 0.01
    setup_params(dt=dt)
    d = make_particles(1, STONE, STATIC, exposure_corrode=exp_c)
    for step in range(640):  # 6.4 s
        compute_reactions(**d, frame=step)
    assert GET_MATERIAL_ID(int(d["sorted_packed_info"].get()[0])) == STONE, \
        "STONE died before 6.4 s in full acid contact"
    for step in range(640, 700):  # to 7.0 s
        compute_reactions(**d, frame=step)
    assert GET_MATERIAL_ID(int(d["sorted_packed_info"].get()[0])) == DEAD, \
        "STONE still alive after 7.0 s in full acid contact"


def test_sand_wetting_is_graded_by_water_fraction():
    """bd-r5-review-n28.48: with normalised exposure, SAND needs more than half
    of its kernel support in WATER to wet, and WET_SAND needs more than 83% to
    turn to MUD. One WATER neighbour among six no longer wets sand."""
    def exposure(centre, n_water):
        packed = ([MAKE_PACKED(centre, GRANULAR)]
                  + [MAKE_PACKED(WATER, FLUID)] * n_water
                  + [MAKE_PACKED(SAND, GRANULAR)] * (6 - n_water))
        _, exp_c = _step1_exposure(packed, [293.0] * 7, _ring_masses(0.0111), _RING)
        return exp_c

    def after(mat, exp_c):
        setup_params(dt=0.001)
        d = make_particles(1, mat, GRANULAR, exposure_corrode=exp_c)
        compute_reactions(**d)
        return GET_MATERIAL_ID(int(d["sorted_packed_info"].get()[0]))

    e1 = exposure(SAND, 1)
    assert abs(e1 - 0.4 / 6) < 1e-4, e1
    assert after(SAND, e1) == SAND
    e4 = exposure(SAND, 4)
    assert abs(e4 - 0.4 * 4 / 6) < 1e-4, e4
    assert after(SAND, e4) == WET_SAND

    w4 = exposure(WET_SAND, 4)
    assert abs(w4 - 0.6 * 4 / 6) < 1e-4, w4
    assert after(WET_SAND, w4) == WET_SAND, "WET_SAND at 67% water must not turn to MUD"
    w6 = exposure(WET_SAND, 6)
    assert abs(w6 - 0.6) < 1e-4, w6
    assert after(WET_SAND, w6) == MUD


# ---------------------------------------------------------------------------
# Latent heat: a phase change clamps temperature to the threshold
# (bd-r5-review-n28.64)
# ---------------------------------------------------------------------------

def _temp_melt(mat_id: int) -> float:
    return float(build_material_array()[mat_id]["temp_melt"])


def test_ice_melt_clamps_temperature():
    """ICE at 280 K becomes WATER at exactly 273 K (the excess is latent heat)."""
    setup_params()
    d = make_particles(1, ICE, STATIC, temp=280.0)
    compute_reactions(**d)

    pi = int(d["sorted_packed_info"].get()[0])
    temp = float(d["sorted_temperature"].get()[0])
    assert GET_MATERIAL_ID(pi) == WATER
    assert temp == 273.0, f"expected 273.0 K after melting, got {temp}"


def test_lava_solidify_clamps_temperature():
    """LAVA 10 K below temp_melt becomes GRAVEL at exactly temp_melt."""
    setup_params()
    t_melt = _temp_melt(LAVA)
    d = make_particles(1, LAVA, FLUID, temp=t_melt - 10.0)
    compute_reactions(**d)

    pi = int(d["sorted_packed_info"].get()[0])
    temp = float(d["sorted_temperature"].get()[0])
    assert GET_MATERIAL_ID(pi) == GRAVEL
    assert temp == np.float32(t_melt), f"expected {t_melt} K after solidifying, got {temp}"


def test_stone_and_sand_melt_clamp_temperature():
    """STONE and SAND above their temp_melt become LAVA at exactly temp_melt."""
    for mat_id, behavior in ((STONE, STATIC), (SAND, GRANULAR)):
        setup_params()
        t_melt = _temp_melt(mat_id)
        d = make_particles(1, mat_id, behavior, temp=t_melt + 50.0)
        compute_reactions(**d)

        pi = int(d["sorted_packed_info"].get()[0])
        temp = float(d["sorted_temperature"].get()[0])
        assert GET_MATERIAL_ID(pi) == LAVA, f"mat {mat_id}: expected LAVA"
        assert temp == np.float32(t_melt), (
            f"mat {mat_id}: expected {t_melt} K after melting, got {temp}")


# ---------------------------------------------------------------------------
# Phase-change temperatures come from the material table (bd-r5-review-n28.89)
# ---------------------------------------------------------------------------

def _setup_with(overrides):
    """setup_params, then re-upload materials with (mat, field) -> value set."""
    setup_params()
    mats = build_material_array()
    for (mat_id, field), value in overrides.items():
        mats[mat_id][field] = value
    upload_materials(mats)


def _run_one(mat_id, behavior, temp):
    d = make_particles(1, mat_id, behavior, temp=temp)
    compute_reactions(**d)
    return int(d["sorted_packed_info"].get()[0]), float(d["sorted_temperature"].get()[0])


def test_temp_boil_override_moves_boiling_point():
    """WATER boils above c_materials[WATER].temp_boil, not a fixed 373 K."""
    try:
        _setup_with({(WATER, "temp_boil"): 450.0})
        pi, _ = _run_one(WATER, FLUID, 400.0)
        assert HAS_SPAWN_FLAG(pi) == 0, "400 K water boiled with temp_boil = 450 K"
        pi, _ = _run_one(WATER, FLUID, 455.0)
        assert HAS_SPAWN_FLAG(pi) == 1, "455 K water did not boil with temp_boil = 450 K"
        # Condensation follows the boil point: 450 - 13 = 437 K.
        pi, _ = _run_one(STEAM, GAS, 440.0)
        assert GET_MATERIAL_ID(pi) == STEAM
        pi, _ = _run_one(STEAM, GAS, 430.0)
        assert GET_MATERIAL_ID(pi) == WATER
    finally:
        setup_params()


def test_temp_melt_override_moves_ice_and_freeze_points():
    """ICE melts above ICE.temp_melt; WATER freezes 10 K below WATER.temp_melt."""
    try:
        _setup_with({(ICE, "temp_melt"): 300.0, (WATER, "temp_melt"): 300.0})
        pi, _ = _run_one(ICE, STATIC, 290.0)
        assert GET_MATERIAL_ID(pi) == ICE, "290 K ice melted with temp_melt = 300 K"
        pi, temp = _run_one(ICE, STATIC, 305.0)
        assert GET_MATERIAL_ID(pi) == WATER and temp == 300.0
        pi, temp = _run_one(WATER, FLUID, 285.0)
        assert GET_MATERIAL_ID(pi) == ICE and temp == 290.0
    finally:
        setup_params()


def test_default_phase_thresholds_unchanged():
    """The table reproduces the old fixed values: 273, 263, 373 and 360 K."""
    setup_params()
    assert GET_MATERIAL_ID(_run_one(ICE, STATIC, 272.9)[0]) == ICE
    assert GET_MATERIAL_ID(_run_one(ICE, STATIC, 273.1)[0]) == WATER
    assert GET_MATERIAL_ID(_run_one(WATER, FLUID, 263.1)[0]) == WATER
    assert GET_MATERIAL_ID(_run_one(WATER, FLUID, 262.9)[0]) == ICE
    assert HAS_SPAWN_FLAG(_run_one(WATER, FLUID, 372.9)[0]) == 0
    assert HAS_SPAWN_FLAG(_run_one(WATER, FLUID, 373.1)[0]) == 1
    assert GET_MATERIAL_ID(_run_one(STEAM, GAS, 360.1)[0]) == STEAM
    assert GET_MATERIAL_ID(_run_one(STEAM, GAS, 359.9)[0]) == WATER


if __name__ == "__main__":
    tests = [
        test_compilation,
        test_block_size,
        test_ice_melts_to_water,
        test_ice_stays_frozen,
        test_lava_solidifies_to_stone,
        test_cooled_lava_falls_to_floor_and_sleeps,
        test_lava_stays_liquid,
        test_water_boils_sets_spawn_flag,
        test_water_cool_no_spawn,
        test_steam_condenses_to_water,
        test_steam_stays_gas,
        test_wood_ignites,
        test_wood_no_ignite,
        test_oil_ignites,
        test_gunpowder_explodes,
        test_corrosion_reduces_health,
        test_corrosion_kills_particle,
        test_gas_lifetime_decay,
        test_gas_lifetime_expires,
        test_dead_particles_unchanged,
        test_ice_lava_together,
        test_acid_metal_corrosion,
        test_500k_stress,
        test_dirt_water_becomes_mud,
        test_dirt_no_corrode_below_threshold,
        test_death_zeroes_mass,
        test_death_without_mass_buffer_still_works,
        test_static_boundary_skips_dead_slot,
        test_expired_gas_mass_stays_zero_in_simulation,
        test_ice_ignores_corrode_exposure,
        test_corroded_stone_becomes_dead_not_fire,
        test_non_zero_reaction_rate_pairs_are_acid_water_mud_or_combustible,
        test_step1_ice_in_water_has_no_corrode_exposure,
        test_step1_metal_by_lava_and_acid_in_water_have_no_corrode_exposure,
        test_wood_next_to_dead_slot_gets_no_heat_exposure,
        test_stone_in_acid_survives_about_one_over_rate,
        test_sand_wetting_is_graded_by_water_fraction,
        test_ice_melt_clamps_temperature,
        test_lava_solidify_clamps_temperature,
        test_stone_and_sand_melt_clamp_temperature,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)


@pytest.mark.parametrize("old_mat,old_beh,new_mat,kw", [
    (SAND, GRANULAR, WET_SAND, {"exposure_corrode": 0.3}),
    (WET_SAND, GRANULAR, SAND, {"exposure_corrode": 0.01}),
    (DIRT, GRANULAR, MUD, {"exposure_corrode": 0.3}),
    (STONE, STATIC, LAVA, {"temp": 1600.0}),
    (LAVA, FLUID, GRAVEL, {"temp": 900.0}),
    (ICE, STATIC, WATER, {"temp": 280.0}),
    (OIL, FLUID, FIRE, {"exposure_heat": 400.0}),
    (GUNPOWDER, GRANULAR, FIRE, {"exposure_heat": 200.0}),
])
def test_material_change_rescales_mass(old_mat, old_beh, new_mat, kw):
    """A material change moves mass from the old spawn mass to the new one (bd-r5-review-n28.53)."""
    from world import spawn_mass
    setup_params(dt=0.001)
    d = make_particles(1, old_mat, old_beh, **kw)
    mass = cupy.full(1, spawn_mass(old_mat), dtype=cupy.float32)
    compute_reactions(**d, sorted_mass=mass)
    assert GET_MATERIAL_ID(int(d["sorted_packed_info"].get()[0])) == new_mat
    np.testing.assert_allclose(mass.get()[0], spawn_mass(new_mat), rtol=1e-5)


# ---------------------------------------------------------------------------
# bd-r5-review-n28.73: water puts out fire
# ---------------------------------------------------------------------------

def test_fire_wet_becomes_smoke():
    """FIRE with exposure_corrode above FIRE_EXTINGUISH_EXPOSURE (0.3) becomes
    SMOKE in one substep, well within 0.1 s, with lifetime 1 s and 373 K."""
    from materials import SMOKE
    setup_params(dt=0.001)
    d = make_particles(16, FIRE, GAS, temp=1200.0, lifetime=1.0,
                       exposure_corrode=0.8)
    mass = cupy.full(16, 0.008, dtype=cupy.float32)
    compute_reactions(**d, sorted_mass=mass)
    pi = d["sorted_packed_info"].get()
    for p in pi:
        assert int(p) == (MAKE_PACKED(SMOKE, GAS) | STATE_DIRTY), f"packed {int(p):#x}, expected SMOKE|GAS"
    np.testing.assert_allclose(d["sorted_lifetime"].get(), 1.0)
    np.testing.assert_allclose(d["sorted_temperature"].get(), 373.0)
    np.testing.assert_allclose(mass.get(), 0.008)


def test_fire_dry_keeps_burning():
    """FIRE with exposure_corrode 0 (or below 0.3) and lifetime 1.0 stays FIRE
    for its full lifetime and only then becomes SMOKE."""
    from materials import SMOKE
    dt = 0.01
    setup_params(dt=dt)
    for exp_c in (0.0, 0.25):
        d = make_particles(4, FIRE, GAS, temp=1200.0, lifetime=1.0,
                           exposure_corrode=exp_c)
        for step in range(95):  # 0.95 s
            compute_reactions(**d, frame=step)
        pi = d["sorted_packed_info"].get()
        assert all(GET_MATERIAL_ID(int(p)) == FIRE for p in pi), \
            f"FIRE went out early at exposure_corrode={exp_c}"
        for step in range(95, 105):
            compute_reactions(**d, frame=step)
        pi = d["sorted_packed_info"].get()
        assert all(GET_MATERIAL_ID(int(p)) == SMOKE for p in pi)


def test_fire_row_rate_is_one_way():
    """The FIRE row has reaction_rate 0 for WOOD, OIL, GUNPOWDER and ICE, so
    those neighbours do not wet fire. The reverse direction keeps the pair's
    rate, heat exchange stays symmetric, and WATER still feeds FIRE."""
    from materials import build_interaction_matrix
    m = build_interaction_matrix()
    for src, reverse_rate in ((WOOD, 0.9), (OIL, 1.0), (GUNPOWDER, 0.9), (ICE, 0.0)):
        assert m[FIRE][src]["reaction_rate"] == 0.0, f"FIRE row rate for {src}"
        assert m[src][FIRE]["reaction_rate"] == pytest.approx(reverse_rate)
        assert m[FIRE][src]["heat_exchange"] == m[src][FIRE]["heat_exchange"] > 0.0
    assert m[FIRE][WATER]["reaction_rate"] == pytest.approx(0.8)
    nonzero = [b for b in range(len(m)) if m[FIRE][b]["reaction_rate"] != 0.0]
    assert nonzero == [WATER], f"FIRE row has non-zero rate from {nonzero}"


def test_step1_fire_in_water_goes_out_fire_on_wood_does_not():
    """Through K_Step1: FIRE ringed by WATER gets exposure_corrode 0.8 and
    becomes SMOKE; FIRE ringed by WOOD gets 0 and stays FIRE."""
    from materials import SMOKE
    packed = [MAKE_PACKED(FIRE, GAS)] + [MAKE_PACKED(WATER, FLUID)] * 6
    _, exp_c = _step1_exposure(packed, [1200.0] + [293.0] * 6,
                               _ring_masses(0.0081), _RING)
    assert exp_c == pytest.approx(0.8, abs=1e-4), f"FIRE in WATER exposure_corrode {exp_c}"
    setup_params(dt=0.001)
    d = make_particles(1, FIRE, GAS, temp=1200.0, lifetime=1.0, exposure_corrode=exp_c)
    compute_reactions(**d)
    assert GET_MATERIAL_ID(int(d["sorted_packed_info"].get()[0])) == SMOKE

    packed = [MAKE_PACKED(FIRE, GAS)] + [MAKE_PACKED(WOOD, STATIC)] * 6
    _, exp_c = _step1_exposure(packed, [1200.0] + [293.0] * 6,
                               _ring_masses(0.0081), _RING)
    assert exp_c == 0.0, f"FIRE on WOOD exposure_corrode {exp_c}"


# ---------------------------------------------------------------------------
# bd-r5-review-n28.95: boiled steam condenses and conserves mass
# ---------------------------------------------------------------------------

def _setup_params_mass(particle_mass):
    sim_params = build_sim_params(
        smoothing_length=H, particle_mass=particle_mass, particle_spacing=SPACING,
        gravity=(0.0, -9.8, 0.0), dt=DT, restitution=0.3, wall_friction=0.5,
        world_min=(-1.0, -1.0, -1.0), world_max=(1.0, 1.0, 1.0),
    )
    upload_sim_params(sim_params)
    upload_materials(build_material_array())


def test_condensed_boiled_steam_keeps_water_mass_in_expectation():
    """3000 boiled STEAM children (mass m/3) condense: about 1/3 survive as
    WATER with mass m, the rest are DEAD with mass 0 and on the freelist."""
    from world import PARTICLE_MASS, spawn_mass
    _setup_params_mass(PARTICLE_MASS)
    n = 3000
    d = make_particles(n, STEAM, GAS, temp=350.0, lifetime=5.0)
    mass = cupy.full(n, PARTICLE_MASS / 3.0, dtype=cupy.float32)
    dead_indices = cupy.zeros(n, dtype=cupy.uint32)
    dead_count = cupy.zeros(1, dtype=cupy.uint32)
    compute_reactions(**d, sorted_mass=mass, dead_indices=dead_indices,
                      dead_count=dead_count, frame=7)
    pi = d["sorted_packed_info"].get()
    m = mass.get()
    mats = pi & 0xFF
    water = mats == WATER
    dead = pi == (MAKE_PACKED(DEAD, STATIC) | STATE_DIRTY)
    assert np.all(water | dead), "every condensed STEAM must be WATER or DEAD"
    np.testing.assert_allclose(m[water], spawn_mass(WATER), rtol=1e-6)
    assert np.all(((pi[water] >> 8) & 0x3) == FLUID)
    assert np.all(m[dead] == 0.0)
    frac = water.sum() / n
    assert abs(frac - 1.0 / 3.0) < 0.02, f"survivor fraction {frac:.3f}"
    cnt = int(dead_count.get()[0])
    assert cnt == int(dead.sum())
    assert set(dead_indices.get()[:cnt].tolist()) == set(np.nonzero(dead)[0].tolist())
    # Mass is conserved to within the sampling noise.
    total_in = n * PARTICLE_MASS / 3.0
    assert abs(m.sum() - total_in) / total_in < 0.06


def test_condensed_full_mass_steam_always_survives():
    """Brush steam carries PARTICLE_MASS, so every particle condenses to WATER."""
    from world import PARTICLE_MASS
    _setup_params_mass(PARTICLE_MASS)
    n = 500
    d = make_particles(n, STEAM, GAS, temp=350.0, lifetime=5.0)
    mass = cupy.full(n, PARTICLE_MASS, dtype=cupy.float32)
    compute_reactions(**d, sorted_mass=mass, frame=3)
    pi = d["sorted_packed_info"].get()
    assert np.all((pi & 0xFF) == WATER)
    np.testing.assert_allclose(mass.get(), PARTICLE_MASS, rtol=1e-6)


def _cu_define(path, name):
    import re
    src = open(os.path.join(os.path.dirname(__file__), "physics", "kernels", path)).read()
    return float(re.search(rf"#define\s+{name}\s+([0-9.]+)f", src).group(1))


def test_boiled_steam_cools_to_condense_before_expiry():
    """Still-air steam cools only by ambient relaxation; it must reach the
    condensation temperature before its spawn lifetime ends."""
    t_spawn = _cu_define("spawn.cu", "SPAWN_TEMP")
    life = _cu_define("spawn.cu", "SPAWN_LIFETIME")
    t_cond = (float(build_material_array()[WATER]["temp_boil"])
              - _cu_define("reactions.cu", "STEAM_CONDENSE_HYSTERESIS"))
    t_amb = _cu_define("sph_shared.cuh", "T_AMBIENT")
    k = _cu_define("sph_shared.cuh", "COOL_RATE")
    t_hit = -np.log((t_cond - t_amb) / (t_spawn - t_amb)) / k
    assert t_hit < life - 1.0, f"steam reaches {t_cond} K at {t_hit:.1f} s, lifetime {life} s"


# ---------------------------------------------------------------------------
# bd-r5-review-n28.134: the blast wave pushes rigid bodies, not STATIC walls
# ---------------------------------------------------------------------------

def _hash_grid(pos, table_size, cell=H, gmin=-1.0):
    """Sort particles by spatial hash (common.cuh spatialHash) and build
    cell_start / cell_end for a table of table_size buckets."""
    c = np.floor((pos - gmin) / cell).astype(np.int64)
    hsh = ((c[:, 0].astype(np.uint32) * np.uint32(73856093)
            + c[:, 1].astype(np.uint32) * np.uint32(19349669)
            + c[:, 2].astype(np.uint32) * np.uint32(83492791))
           & np.uint32(table_size - 1))
    order = np.argsort(hsh, kind="stable")
    hs = hsh[order]
    start = np.full(table_size, 0xFFFFFFFF, dtype=np.uint32)
    end = np.zeros(table_size, dtype=np.uint32)
    for k, b in enumerate(hs):
        if start[b] == 0xFFFFFFFF:
            start[b] = k
        end[b] = k + 1
    return order, start, end


def test_blast_wave_pushes_rigid_body_not_static_wall():
    """Rigid boundary particles 0.5 h from a fresh blast add a force away
    from the blast to their body; a STONE wall particle keeps zero velocity."""
    from grid_params import GRID_PARAMS_DTYPE
    from materials import MAT_RIGID
    from reactions import compute_blast_wave, upload_grid_params
    from rigid_bodies import RIGID_BODY_DTYPE

    setup_params()
    table_size = 1024
    gp = np.zeros(1, dtype=GRID_PARAMS_DTYPE)
    gp[0]["grid_min"] = (-1.0, -1.0, -1.0)
    gp[0]["grid_delta"] = (1.0 / H, 1.0 / H, 1.0 / H)
    gp[0]["table_size"] = table_size
    gp[0]["table_mask"] = table_size - 1
    upload_grid_params(gp)

    blast = np.array([0.011, 0.013, 0.017], dtype=np.float32)
    body_id = 1
    rigid_pk = MAKE_PACKED(MAT_RIGID, STATIC) | (body_id << 13)
    # (offset from blast, packed_info, mass)
    parts = [
        ((0.0, 0.0, 0.0), MAKE_PACKED(FIRE, GAS), 0.008),       # blast source
        ((0.5 * H, 0.0, 0.0), rigid_pk, 0.02),                  # rigid, +x
        ((0.5 * H, 0.004, 0.0), rigid_pk, 0.02),
        ((0.5 * H, -0.004, 0.0), rigid_pk, 0.02),
        ((-0.5 * H, 0.0, 0.0), MAKE_PACKED(STONE, STATIC), 0.02),  # wall, -x
        ((0.0, 0.5 * H, 0.0), MAKE_PACKED(WATER, FLUID), 0.008),   # fluid, +y
    ]
    pos = np.array([blast + np.array(p[0], dtype=np.float32) for p in parts],
                   dtype=np.float32)
    order, start, end = _hash_grid(pos, table_size)
    n = len(parts)
    pos4 = np.zeros((n, 4), dtype=np.float32)
    pos4[:, :3] = pos[order]
    packed = np.array([parts[k][1] for k in order], dtype=np.uint32)
    mass = np.array([parts[k][2] for k in order], dtype=np.float32)
    life = np.zeros(n, dtype=np.float32)
    life[packed & 0xFF == FIRE] = _cu_define("reactions.cu", "GUNPOWDER_FIRE_LIFETIME")
    inv = np.argsort(order)  # original index -> sorted index

    bodies = np.zeros(8, dtype=RIGID_BODY_DTYPE)
    com = blast + np.array([0.5 * H + 0.1, 0.0, 0.0], dtype=np.float32)
    bodies_raw = bodies.view(np.float32).reshape(8, 24)
    bodies_raw[body_id, 0:3] = com
    d_bodies = cupy.asarray(bodies_raw)
    forces = cupy.zeros(8 * 4, dtype=cupy.float32)
    torques = cupy.zeros(8 * 4, dtype=cupy.float32)
    vel = cupy.zeros((n, 4), dtype=cupy.float32)

    compute_blast_wave(
        cupy.asarray(pos4), vel, cupy.asarray(packed), cupy.asarray(life),
        cupy.asarray(start), cupy.asarray(end), smoothing_length=H,
        sorted_mass=cupy.asarray(mass), d_rigid_bodies=d_bodies,
        d_rigid_forces=forces, d_rigid_torques=torques,
    )
    f = forces.get().reshape(8, 4)
    tq = torques.get().reshape(8, 4)
    v = vel.get()
    print(f"\n  blast force on body {body_id}: {f[body_id, :3]}, torque {tq[body_id, :3]}")

    dt = DT
    # Expected: 3 particles * m * 15 m/s * (1 - ~0.5) / dt, about 450 N along +x.
    assert f[body_id, 0] > 100.0, f"no blast force on the body: {f[body_id, :3]}"
    assert abs(f[body_id, 1]) < 0.05 * f[body_id, 0]
    assert abs(f[body_id, 2]) < 1e-3 * f[body_id, 0]
    assert np.all(f[np.arange(8) != body_id] == 0.0)
    # The kernel accumulates torque about the world origin; K_IntegrateRigidBodies
    # subtracts COM x F (bd-r5-review-n28.195). Symmetric about the COM axis:
    # the torque about the COM cancels to ~0 about z.
    tq_com = tq[body_id, :3].astype(np.float64) - np.cross(
        com.astype(np.float64), f[body_id, :3].astype(np.float64))
    assert abs(tq_com[2]) < 1e-3 * f[body_id, 0] * 0.1, tq_com
    # Upper bound: m * 15 / dt per particle.
    assert f[body_id, 0] < 3 * 0.02 * 15.0 / dt

    # No velocity write on any STATIC target (STONE wall and rigid particles).
    static_rows = ((packed >> 8) & 0x3) == STATIC
    assert np.all(v[static_rows] == 0.0), v[static_rows]
    # The fluid neighbour is still kicked outward (+y).
    assert v[inv[5], 1] > 1.0
