"""Integration tests for reactions.py -- K_Reactions kernel.

Acceptance criteria:
  - ICE (mat=11, temp > 273K) -> WATER (mat=5, FLUID)
  - LAVA (mat=7, temp < 900K) -> STONE (mat=1, STATIC)
  - WATER (mat=5, temp > 373K) -> set SPAWN_GAS flag
  - WOOD (mat=9, exposure_heat > 0.5) -> FIRE (mat=14, GAS, T=1200K, lifetime=1.0s)
  - OIL (mat=6, exposure_heat > 0.3) -> FIRE (mat=14, GAS, T=1200K, lifetime=1.5s)
  - GUNPOWDER (mat=15, exposure_heat > 0.1) -> FIRE + explosion velocity
  - Corrosion: health -= exposure_corrode * dt; health <= 0 -> DEAD
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

from step1 import (
    SIM_PARAMS_DTYPE,
    build_sim_params,
)
from materials import (
    FLUID, GRANULAR, GAS, STATIC,
    DEAD, STONE, SAND, WATER, OIL, LAVA, WOOD, METAL, ICE, STEAM, FIRE, GUNPOWDER,
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
    """Create sorted particle arrays for testing reactions kernel."""
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


def test_lava_solidifies_to_stone():
    """LAVA at temp < 900K transitions to STONE (STATIC)."""
    setup_params()
    d = make_particles(100, LAVA, FLUID, temp=800.0)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == STONE, f"Particle {i}: expected STONE"
        assert GET_BEHAVIOR(int(pi[i])) == STATIC, f"Particle {i}: expected STATIC"
    print("PASS: test_lava_solidifies_to_stone")


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
    """WOOD with exposure_heat > 0.5 transitions to FIRE (GAS, T=1200K, lifetime=1.0s)."""
    setup_params()
    d = make_particles(100, WOOD, STATIC, exposure_heat=0.6)
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
    """WOOD with exposure_heat < 0.5 stays as WOOD."""
    setup_params()
    d = make_particles(100, WOOD, STATIC, exposure_heat=0.3)
    compute_reactions(**d)

    pi = d["sorted_packed_info"].get()
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == WOOD, f"Particle {i}: should stay WOOD"
    print("PASS: test_wood_no_ignite")


def test_oil_ignites():
    """OIL with exposure_heat > 0.3 transitions to FIRE (GAS, T=1200K, lifetime=1.5s)."""
    setup_params()
    d = make_particles(100, OIL, FLUID, exposure_heat=0.4)
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
    """GUNPOWDER with exposure_heat > 0.1 transitions to FIRE + random velocity burst."""
    setup_params()
    d = make_particles(100, GUNPOWDER, GRANULAR, exposure_heat=0.2)
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
    """Corrosion brings health to 0 -- particle becomes a brief FIRE spark (corrosion flash)."""
    setup_params(dt=0.1)
    d = make_particles(100, METAL, STATIC, health=0.05, exposure_corrode=1.0)
    compute_reactions(**d)

    hlth = d["sorted_health"].get()
    pi = d["sorted_packed_info"].get()
    temp = d["sorted_temperature"].get()
    lt = d["sorted_lifetime"].get()
    # health = 0.05 - 1.0 * 0.1 = -0.05 -> corrosion flash: brief FIRE spark
    # Kernel sets: packed_info=FIRE|GAS, health=1.0, temp=400K, lifetime=0.08s
    from materials import FIRE, GAS
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == FIRE, \
            f"Particle {i}: expected FIRE spark, got mat_id={GET_MATERIAL_ID(int(pi[i]))}"
        assert GET_BEHAVIOR(int(pi[i])) == GAS, f"Particle {i}: expected GAS"
        assert abs(hlth[i] - 1.0) < 1e-6, f"Particle {i}: health={hlth[i]}, expected 1.0"
        assert abs(temp[i] - 400.0) < 1.0, f"Particle {i}: temp={temp[i]}, expected 400K"
        assert abs(lt[i] - 0.08) < 0.01, f"Particle {i}: lifetime={lt[i]}, expected 0.08s"
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
    """Ice and lava together: ice melts to water, lava solidifies to stone."""
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

    # Next 100: LAVA -> STONE
    for i in range(100, 200):
        assert GET_MATERIAL_ID(int(pi[i])) == STONE, f"LAVA particle {i}: expected STONE"
        assert GET_BEHAVIOR(int(pi[i])) == STATIC

    print("PASS: test_ice_lava_together")


def test_acid_metal_corrosion():
    """Acid near metal: after enough corrosion steps the particle becomes a FIRE spark."""
    setup_params(dt=0.001)
    d = make_particles(100, METAL, STATIC, health=1.0, exposure_corrode=10.0)

    # Run 110 steps of reactions (100 steps to kill METAL, 10 more as FIRE spark)
    # At dt=0.001 and exp_corrode=10.0, damage=0.01/step -> health=0 at step 100
    for step in range(110):
        # Re-apply exposure each step (simulating continuous acid contact)
        d["sorted_exposure_corrode"] = cupy.full(100, 10.0, dtype=cupy.float32)
        compute_reactions(**d, frame=step)

    pi = d["sorted_packed_info"].get()
    lt = d["sorted_lifetime"].get()
    # After step 100: METAL -> FIRE spark (health=1.0, temp=400K, lifetime=0.08s)
    # Steps 101-110: FIRE spark loses lifetime: 0.08 - 10*0.001 = 0.07s -- still FIRE
    from materials import FIRE, GAS
    for i in range(100):
        assert GET_MATERIAL_ID(int(pi[i])) == FIRE, \
            f"Particle {i}: expected FIRE spark after corrosion, got mat_id={GET_MATERIAL_ID(int(pi[i]))}"
        assert GET_BEHAVIOR(int(pi[i])) == GAS, f"Particle {i}: expected GAS"
        assert lt[i] > 0.0, f"Particle {i}: lifetime={lt[i]}, spark should still be alive"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_compilation,
        test_block_size,
        test_ice_melts_to_water,
        test_ice_stays_frozen,
        test_lava_solidifies_to_stone,
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
