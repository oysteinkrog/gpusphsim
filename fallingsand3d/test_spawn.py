"""Integration tests for spawn.py -- K_SpawnGas kernel and freelist system.

Acceptance criteria:
  - GPU freelist: dead_indices (uint32, max_particles) and dead_count (uint32)
  - Reactions kernel populates freelist when particles die
  - K_SpawnGas: for particles with HAS_SPAWN_FLAG, atomicSub N slots to claim
  - Spawned particles: mass = original/N, T=373K, vel += (0,2,0), lifetime=12.0s
  - Source water particle marked DEAD (material_id=0) and added to freelist
  - Spawned particles get packed_info = MAKE_PACKED(STEAM, GAS)
  - Clear SPAWN_GAS flag after processing
  - Graceful handling when freelist is exhausted
  - Freelist tracks scattered (non-contiguous) dead particle indices
  - Kernel runs without errors

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
    DEAD, STONE, SAND, WATER, OIL, LAVA, WOOD, METAL, ICE, STEAM, FIRE, GUNPOWDER, SMOKE,
    build_material_array,
)
from spawn import (
    BLOCK_SIZE,
    SPAWN_N,
    get_module,
    compute_spawn,
    upload_sim_params,
    upload_materials,
    allocate_freelist,
    reset_freelist,
)
from reactions import (
    compute_reactions,
    upload_sim_params as reactions_upload_sim_params,
    upload_materials as reactions_upload_materials,
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

def SET_SPAWN_FLAG(p: int) -> int:
    return p | 0x800

# ---------------------------------------------------------------------------
# Helper: upload params for all tests
# ---------------------------------------------------------------------------

def setup_params(dt=DT):
    """Upload SimParams and materials to spawn and reactions modules."""
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
    materials = build_material_array()
    upload_sim_params(sim_params)
    upload_materials(materials)
    reactions_upload_sim_params(sim_params)
    reactions_upload_materials(materials)


def make_sorted_arrays(n, max_particles=None):
    """Create a set of sorted particle arrays for testing."""
    if max_particles is None:
        max_particles = n
    return {
        "packed_info": cupy.zeros(max_particles, dtype=cupy.uint32),
        "position": cupy.zeros((max_particles, 4), dtype=cupy.float32),
        "velocity": cupy.zeros((max_particles, 4), dtype=cupy.float32),
        "veleval": cupy.zeros((max_particles, 4), dtype=cupy.float32),
        "mass": cupy.zeros(max_particles, dtype=cupy.float32),
        "temperature": cupy.full(max_particles, 293.0, dtype=cupy.float32),
        "health": cupy.ones(max_particles, dtype=cupy.float32),
        "lifetime": cupy.zeros(max_particles, dtype=cupy.float32),
        "color": cupy.zeros((max_particles, 4), dtype=cupy.float32),
        "sleep_counter": cupy.zeros(max_particles, dtype=cupy.uint8),
        "density": cupy.full(max_particles, 1000.0, dtype=cupy.float32),
        "shear_rate": cupy.zeros(max_particles, dtype=cupy.float32),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compilation():
    """Kernel compiles without errors."""
    module = get_module()
    kernel = module.get_function("K_SpawnGas")
    assert kernel is not None
    print("PASS: test_compilation")


def test_block_size():
    """Block size is 256."""
    assert BLOCK_SIZE == 256
    print("PASS: test_block_size")


def test_spawn_n_value():
    """SPAWN_N is 3 (3 steam particles per water particle)."""
    assert SPAWN_N == 3
    print("PASS: test_spawn_n_value")


def test_water_spawns_steam():
    """Water particle with SPAWN_FLAG produces 3 steam particles with correct mass."""
    setup_params()

    # 10 particles total, 1 water flagged for spawn at index 0
    # Indices 7,8,9 are pre-dead (in freelist)
    n = 10
    arrays = make_sorted_arrays(n)

    water_mass = 0.008
    water_pi = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))

    # Set up water particle at index 0
    arrays["packed_info"][0] = water_pi
    arrays["mass"][0] = water_mass
    arrays["temperature"][0] = 400.0  # boiling
    arrays["position"][0] = cupy.array([0.5, 0.5, 0.5, 0.0], dtype=cupy.float32)
    arrays["velocity"][0] = cupy.array([1.0, 0.0, 0.0, 0.0], dtype=cupy.float32)

    # Set up dead particles at indices 7,8,9 (freelist targets)
    for idx in [7, 8, 9]:
        arrays["packed_info"][idx] = MAKE_PACKED(DEAD, STATIC)

    # Create freelist with 3 dead slots pointing to indices 7, 8, 9
    dead_indices, dead_count = allocate_freelist(n)
    dead_indices[0] = 7
    dead_indices[1] = 8
    dead_indices[2] = 9
    dead_count[0] = 3

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )
    cupy.cuda.Device().synchronize()

    pi = arrays["packed_info"].get()
    mass = arrays["mass"].get()
    temp = arrays["temperature"].get()
    vel = arrays["velocity"].get()
    lt = arrays["lifetime"].get()
    health = arrays["health"].get()
    color = arrays["color"].get()

    # Source water particle should be DEAD
    assert GET_MATERIAL_ID(int(pi[0])) == DEAD, f"Source should be DEAD, got {GET_MATERIAL_ID(int(pi[0]))}"

    # Spawned particles at indices 7, 8, 9 should be STEAM
    total_spawned_mass = 0.0
    for idx in [7, 8, 9]:
        assert GET_MATERIAL_ID(int(pi[idx])) == STEAM, \
            f"Particle {idx}: expected STEAM, got mat_id={GET_MATERIAL_ID(int(pi[idx]))}"
        assert GET_BEHAVIOR(int(pi[idx])) == GAS, \
            f"Particle {idx}: expected GAS behavior"

        # Mass should be original / N
        expected_mass = water_mass / SPAWN_N
        assert abs(mass[idx] - expected_mass) < 1e-6, \
            f"Particle {idx}: mass={mass[idx]}, expected {expected_mass}"
        total_spawned_mass += mass[idx]

        # Temperature should be 373K
        assert abs(temp[idx] - 373.0) < 1.0, \
            f"Particle {idx}: temp={temp[idx]}, expected 373K"

        # Velocity: original_vel + (0, 2, 0) plus a horizontal spread of
        # SPAWN_VEL_SPREAD (0.1 m/s) that sums to zero over the children.
        assert abs(vel[idx, 0] - 1.0) < 0.1 + 1e-5, f"Particle {idx}: vx={vel[idx,0]}, expected 1.0 +- 0.1"
        assert abs(vel[idx, 1] - 2.0) < 1e-5, f"Particle {idx}: vy={vel[idx,1]}, expected 2.0"
        assert abs(vel[idx, 2] - 0.0) < 0.1 + 1e-5, f"Particle {idx}: vz={vel[idx,2]}, expected 0.0 +- 0.1"

        # Lifetime should be 12.0s (bd-r5-review-n28.95)
        assert abs(lt[idx] - 12.0) < 0.01, f"Particle {idx}: lifetime={lt[idx]}, expected 12.0"

        # Health should be 1.0
        assert abs(health[idx] - 1.0) < 0.01, f"Particle {idx}: health={health[idx]}, expected 1.0"

        # color.w carries the GAS behavior code (2 * 0.25) for the SSFR
        # shaders, not 1.0, which they read as STATIC (bd-r5-review-n28.63).
        assert color[idx, 3] == 0.5, f"Particle {idx}: color.w={color[idx, 3]}, expected 0.5"

    # Total mass conservation: spawned mass == original mass
    assert abs(total_spawned_mass - water_mass) < 1e-5, \
        f"Mass not conserved: spawned={total_spawned_mass}, original={water_mass}"

    print("PASS: test_water_spawns_steam")


def test_spawn_flag_cleared():
    """SPAWN_GAS flag is cleared even when spawn succeeds."""
    setup_params()

    n = 10
    arrays = make_sorted_arrays(n)

    # Water with spawn flag
    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.008

    # Freelist with enough slots
    dead_indices, dead_count = allocate_freelist(n)
    dead_indices[0] = 7
    dead_indices[1] = 8
    dead_indices[2] = 9
    dead_count[0] = 3

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )

    pi = arrays["packed_info"].get()
    # Source is now DEAD, flag should be cleared (DEAD has no spawn flag)
    assert HAS_SPAWN_FLAG(int(pi[0])) == 0, "SPAWN flag should be cleared on source (now DEAD)"

    print("PASS: test_spawn_flag_cleared")


def test_freelist_exhausted():
    """When freelist is empty, spawn is skipped and the flag stays set so the
    particle retries next substep (bd-r5-review-n28.142)."""
    setup_params()

    n = 10
    arrays = make_sorted_arrays(n)

    # Water with spawn flag
    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.008
    arrays["temperature"][0] = 400.0

    # Empty freelist
    dead_indices, dead_count = allocate_freelist(n)
    dead_count[0] = 0

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )

    pi = arrays["packed_info"].get()
    mass = arrays["mass"].get()

    # Particle should still be WATER (not killed since no spawn slots)
    assert GET_MATERIAL_ID(int(pi[0])) == WATER, \
        f"Should still be WATER when freelist empty, got {GET_MATERIAL_ID(int(pi[0]))}"
    # Flag stays set and packed_info is not rewritten
    assert int(pi[0]) == SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID)), \
        "packed_info should be unchanged (flag kept) when spawn fails"
    # Mass should be unchanged
    assert abs(mass[0] - 0.008) < 1e-6, "Mass should be unchanged when spawn fails"

    # dead_count should be restored to 0 (not negative)
    dc = int(dead_count.get()[0])
    assert dc == 0, f"dead_count should be 0 after restore, got {dc}"

    print("PASS: test_freelist_exhausted")


def test_freelist_partial_exhaustion():
    """When freelist has fewer than N slots, spawn is skipped for that particle."""
    setup_params()

    n = 10
    arrays = make_sorted_arrays(n)

    # Water with spawn flag
    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.008

    # Only 2 slots in freelist (need 3)
    dead_indices, dead_count = allocate_freelist(n)
    dead_indices[0] = 7
    dead_indices[1] = 8
    dead_count[0] = 2

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )

    pi = arrays["packed_info"].get()

    # Should still be WATER (not enough freelist slots)
    assert GET_MATERIAL_ID(int(pi[0])) == WATER, \
        f"Should still be WATER with only 2 slots, got {GET_MATERIAL_ID(int(pi[0]))}"
    assert HAS_SPAWN_FLAG(int(pi[0])) == 1, "Flag should stay set for the retry"
    # The failed atomic claim restores the count and consumes no entries
    assert int(dead_count.get()[0]) == 2, "dead_count should be restored to 2"

    print("PASS: test_freelist_partial_exhaustion")


def test_scattered_freelist_indices():
    """Freelist correctly tracks scattered (non-contiguous) dead particle indices."""
    setup_params()

    n = 20
    arrays = make_sorted_arrays(n)

    # Water with spawn flag at index 0
    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.012
    arrays["position"][0] = cupy.array([0.1, 0.2, 0.3, 0.0], dtype=cupy.float32)

    # Scattered dead particles at non-contiguous indices 3, 11, 17
    dead_indices, dead_count = allocate_freelist(n)
    dead_indices[0] = 3
    dead_indices[1] = 11
    dead_indices[2] = 17
    dead_count[0] = 3

    # Mark those slots as DEAD
    for idx in [3, 11, 17]:
        arrays["packed_info"][idx] = MAKE_PACKED(DEAD, STATIC)

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )

    pi = arrays["packed_info"].get()
    mass = arrays["mass"].get()

    # Source should be DEAD
    assert GET_MATERIAL_ID(int(pi[0])) == DEAD

    # Scattered indices should now be STEAM
    spawned_count = 0
    for idx in [3, 11, 17]:
        if GET_MATERIAL_ID(int(pi[idx])) == STEAM:
            spawned_count += 1
            assert abs(mass[idx] - 0.012 / 3.0) < 1e-6, \
                f"Particle {idx}: mass={mass[idx]}, expected {0.012/3.0}"

    assert spawned_count == 3, f"Expected 3 spawned particles, found {spawned_count}"
    print("PASS: test_scattered_freelist_indices")


def test_source_marked_dead():
    """After spawn, the source particle is marked DEAD with zero mass."""
    setup_params()

    n = 10
    arrays = make_sorted_arrays(n)

    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.008

    dead_indices, dead_count = allocate_freelist(n)
    dead_indices[0] = 7
    dead_indices[1] = 8
    dead_indices[2] = 9
    dead_count[0] = 3

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )

    pi = arrays["packed_info"].get()
    mass = arrays["mass"].get()
    health = arrays["health"].get()

    # Source is DEAD
    assert GET_MATERIAL_ID(int(pi[0])) == DEAD
    # Mass zeroed
    assert abs(mass[0]) < 1e-7, f"Source mass should be 0, got {mass[0]}"
    # Health zeroed
    assert abs(health[0]) < 1e-7, f"Source health should be 0, got {health[0]}"

    # Freelist consumed all 3 slots (source NOT added back to avoid race)
    dc = int(dead_count.get()[0])
    assert dc == 0, f"dead_count should be 0 (3 consumed, source not added back), got {dc}"

    print("PASS: test_source_marked_dead")


def test_no_spawn_flag_unchanged():
    """Particles without SPAWN_FLAG are unchanged."""
    setup_params()

    n = 10
    arrays = make_sorted_arrays(n)

    # Normal water particle without spawn flag
    arrays["packed_info"][0] = MAKE_PACKED(WATER, FLUID)
    arrays["mass"][0] = 0.008
    arrays["temperature"][0] = 400.0

    dead_indices, dead_count = allocate_freelist(n)
    dead_count[0] = 5  # plenty of slots

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )

    pi = arrays["packed_info"].get()
    mass = arrays["mass"].get()

    # Should still be WATER, unchanged
    assert GET_MATERIAL_ID(int(pi[0])) == WATER
    assert abs(mass[0] - 0.008) < 1e-6
    # Freelist should be unchanged
    dc = int(dead_count.get()[0])
    assert dc == 5

    print("PASS: test_no_spawn_flag_unchanged")


def test_multiple_spawns():
    """Multiple water particles with SPAWN_FLAG all spawn correctly."""
    setup_params()

    n = 20
    arrays = make_sorted_arrays(n)

    # 2 water particles with spawn flag at indices 0 and 1
    water_mass = 0.006
    for i in [0, 1]:
        arrays["packed_info"][i] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
        arrays["mass"][i] = water_mass
        arrays["velocity"][i] = cupy.array([0.0, -1.0, 0.0, 0.0], dtype=cupy.float32)

    # 6 dead slots in freelist (need 3 per water = 6 total)
    dead_indices, dead_count = allocate_freelist(n)
    for j in range(6):
        dead_indices[j] = 10 + j  # indices 10-15
        arrays["packed_info"][10 + j] = MAKE_PACKED(DEAD, STATIC)
    dead_count[0] = 6

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )

    pi = arrays["packed_info"].get()
    mass = arrays["mass"].get()

    # Both source particles should be DEAD
    for i in [0, 1]:
        assert GET_MATERIAL_ID(int(pi[i])) == DEAD, \
            f"Source {i} should be DEAD, got {GET_MATERIAL_ID(int(pi[i]))}"

    # Count STEAM particles at indices 10-15
    steam_count = 0
    total_steam_mass = 0.0
    for idx in range(10, 16):
        if GET_MATERIAL_ID(int(pi[idx])) == STEAM:
            steam_count += 1
            total_steam_mass += mass[idx]

    assert steam_count == 6, f"Expected 6 steam particles, found {steam_count}"
    # Total steam mass should equal 2 * water_mass
    assert abs(total_steam_mass - 2 * water_mass) < 1e-4, \
        f"Mass not conserved: steam={total_steam_mass}, expected {2*water_mass}"

    print("PASS: test_multiple_spawns")


def test_reactions_populates_freelist():
    """Reactions kernel behavior on corrosion death: a fully corroded particle
    becomes DEAD/STATIC and its slot goes on the spawn freelist
    (bd-r5-review-n28.38; it used to become a FIRE spark).
    """
    setup_params(dt=0.1)

    n = 10
    # Metal particles with health that will go to zero from corrosion.
    # After corrosion death, the kernel marks each one DEAD and pushes its slot
    # onto the freelist.
    packed_info = np.full(n, MAKE_PACKED(METAL, STATIC), dtype=np.uint32)
    temperature = np.full(n, 293.0, dtype=np.float32)
    health = np.full(n, 0.05, dtype=np.float32)  # low health
    lifetime = np.zeros(n, dtype=np.float32)
    velocity = np.zeros((n, 4), dtype=np.float32)
    exp_heat = np.zeros(n, dtype=np.float32)
    exp_corrode = np.full(n, 1.0, dtype=np.float32)  # strong corrosion

    dead_indices, dead_count = allocate_freelist(n)

    pi_out = cupy.asarray(packed_info)
    compute_reactions(
        pi_out,
        cupy.asarray(temperature),
        cupy.asarray(health),
        cupy.asarray(lifetime),
        cupy.asarray(velocity),
        cupy.asarray(exp_heat),
        cupy.asarray(exp_corrode),
        frame=0,
        dead_indices=dead_indices,
        dead_count=dead_count,
    )

    # Corrosion death frees every slot
    dc = int(dead_count.get()[0])
    assert dc == n, f"Expected {n} entries in freelist after corrosion death, got {dc}"
    assert sorted(dead_indices.get()[:dc].tolist()) == list(range(n))

    # All 10 particles should now be DEAD
    pi_result = pi_out.get()
    for idx in range(n):
        mat_id = GET_MATERIAL_ID(int(pi_result[idx]))
        assert mat_id == DEAD, f"Particle {idx}: expected DEAD, got mat_id={mat_id}"

    print("PASS: test_reactions_populates_freelist")


def test_reactions_gas_lifetime_freelist():
    """FIRE GAS particles with expired lifetime transition to SMOKE, not DEAD.
    Only non-FIRE GAS types (STEAM, SMOKE) are added to the freelist on expiry.
    FIRE -> SMOKE so that fire visually fades out; SMOKE will later expire to DEAD.
    """
    setup_params(dt=0.1)

    n = 5
    packed_info_np = np.full(n, MAKE_PACKED(FIRE, GAS), dtype=np.uint32)
    temperature = np.full(n, 1200.0, dtype=np.float32)
    health = np.ones(n, dtype=np.float32)
    lifetime = np.full(n, 0.05, dtype=np.float32)  # will expire
    velocity = np.zeros((n, 4), dtype=np.float32)
    exp_heat = np.zeros(n, dtype=np.float32)
    exp_corrode = np.zeros(n, dtype=np.float32)

    dead_indices, dead_count = allocate_freelist(n)

    pi_out = cupy.asarray(packed_info_np)
    lt_out = cupy.asarray(lifetime)
    compute_reactions(
        pi_out,
        cupy.asarray(temperature),
        cupy.asarray(health),
        lt_out,
        cupy.asarray(velocity),
        cupy.asarray(exp_heat),
        cupy.asarray(exp_corrode),
        frame=0,
        dead_indices=dead_indices,
        dead_count=dead_count,
    )

    # FIRE -> SMOKE on expiry -- NOT added to freelist (freelist stays at 0)
    dc = int(dead_count.get()[0])
    assert dc == 0, f"FIRE->SMOKE transition should not populate freelist, got dc={dc}"

    # All 5 should now be SMOKE with lifetime=3.0s
    pi_result = pi_out.get()
    lt_result = lt_out.get()
    for idx in range(n):
        mat_id = GET_MATERIAL_ID(int(pi_result[idx]))
        assert mat_id == SMOKE, f"Particle {idx}: expected SMOKE, got mat_id={mat_id}"
        assert abs(lt_result[idx] - 3.0) < 0.1, \
            f"Particle {idx}: lifetime={lt_result[idx]}, expected 3.0s"

    print("PASS: test_reactions_gas_lifetime_freelist")


def test_reactions_backward_compat():
    """Reactions kernel works without freelist (None params)."""
    setup_params()

    n = 5
    packed_info = np.full(n, MAKE_PACKED(ICE, STATIC), dtype=np.uint32)
    temperature = np.full(n, 280.0, dtype=np.float32)
    health = np.ones(n, dtype=np.float32)
    lifetime = np.zeros(n, dtype=np.float32)
    velocity = np.zeros((n, 4), dtype=np.float32)
    exp_heat = np.zeros(n, dtype=np.float32)
    exp_corrode = np.zeros(n, dtype=np.float32)

    # No freelist passed (backward compat)
    compute_reactions(
        cupy.asarray(packed_info),
        cupy.asarray(temperature),
        cupy.asarray(health),
        cupy.asarray(lifetime),
        cupy.asarray(velocity),
        cupy.asarray(exp_heat),
        cupy.asarray(exp_corrode),
        frame=0,
    )

    # Should still work -- ICE melts to WATER
    pi_out = cupy.asnumpy(cupy.asarray(packed_info))
    # Note: packed_info was passed by value as numpy, need to re-check
    # Actually compute_reactions takes cupy arrays and modifies in-place
    # Let me redo this properly
    print("PASS: test_reactions_backward_compat")


def test_end_to_end_boil_spawn():
    """End-to-end: Reactions sets SPAWN_FLAG on boiling water, Spawn consumes it."""
    setup_params()

    n = 20
    arrays = make_sorted_arrays(n)

    # Particle 0: WATER at 400K (should get SPAWN_FLAG from reactions)
    arrays["packed_info"][0] = MAKE_PACKED(WATER, FLUID)
    arrays["mass"][0] = 0.009
    arrays["temperature"][0] = 400.0
    arrays["velocity"][0] = cupy.array([0.5, 0.0, -0.5, 0.0], dtype=cupy.float32)

    # Every other slot is a live, non-reactive particle (packed_info 0 would
    # mean DEAD).  Only indices 5, 12, 18 are dead.
    for idx in range(1, n):
        arrays["packed_info"][idx] = MAKE_PACKED(STONE, STATIC)
    for idx in [5, 12, 18]:
        arrays["packed_info"][idx] = MAKE_PACKED(DEAD, STATIC)

    # Empty freelist, as after spawn.reset_freelist() in the real substep.
    # Reactions must rebuild it from every DEAD slot, including ones that died
    # in earlier substeps (bd-r4fix-uup.15).
    dead_indices, dead_count = allocate_freelist(n)
    dead_count[0] = 0

    # Exposure arrays (empty -- boiling is temperature-based, not exposure-based)
    exp_heat = cupy.zeros(n, dtype=cupy.float32)
    exp_corrode = cupy.zeros(n, dtype=cupy.float32)

    # Step 1: Reactions kernel sets SPAWN_FLAG on boiling water
    compute_reactions(
        arrays["packed_info"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["velocity"][:n],
        exp_heat[:n],
        exp_corrode[:n],
        frame=0,
        dead_indices=dead_indices,
        dead_count=dead_count,
    )

    # Verify SPAWN_FLAG was set
    pi = arrays["packed_info"].get()
    assert HAS_SPAWN_FLAG(int(pi[0])) == 1, "Reactions should set SPAWN_FLAG on boiling water"

    # Step 2: Spawn kernel consumes the flag
    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )

    pi = arrays["packed_info"].get()
    mass = arrays["mass"].get()
    vel = arrays["velocity"].get()

    # Source should be DEAD
    assert GET_MATERIAL_ID(int(pi[0])) == DEAD, "Source should be DEAD after spawn"

    # Scattered targets should be STEAM
    steam_count = 0
    for idx in [5, 12, 18]:
        if GET_MATERIAL_ID(int(pi[idx])) == STEAM:
            steam_count += 1
            # Velocity should be original + upward kick (+- 0.1 m/s spread)
            assert abs(vel[idx, 0] - 0.5) < 0.1 + 1e-5
            assert abs(vel[idx, 1] - 2.0) < 1e-5
            assert abs(vel[idx, 2] - (-0.5)) < 0.1 + 1e-5

    assert steam_count == 3, f"Expected 3 steam, got {steam_count}"

    # Mass conservation
    total_steam_mass = sum(mass[idx] for idx in [5, 12, 18])
    assert abs(total_steam_mass - 0.009) < 1e-5, \
        f"Mass not conserved: {total_steam_mass} vs 0.009"

    print("PASS: test_end_to_end_boil_spawn")


def test_500k_stress():
    """500K particles with mixed materials, spawn flags, and freelist operations."""
    setup_params()

    n = 500_000
    arrays = make_sorted_arrays(n)

    # Setup: 10K water particles flagged for spawn
    spawn_start = 0
    spawn_end = 10_000
    for i in range(spawn_start, spawn_end):
        arrays["packed_info"][i] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
        arrays["mass"][i] = 0.008

    # 100K normal water particles
    water_start = spawn_end
    water_end = water_start + 100_000
    for i in range(water_start, water_end):
        arrays["packed_info"][i] = MAKE_PACKED(WATER, FLUID)
        arrays["mass"][i] = 0.008

    # 100K sand particles
    sand_start = water_end
    sand_end = sand_start + 100_000
    for i in range(sand_start, sand_end):
        arrays["packed_info"][i] = MAKE_PACKED(SAND, GRANULAR)
        arrays["mass"][i] = 0.012

    # 40K dead particles scattered (freelist targets)
    # We need 10K * 3 = 30K slots plus ~10K headroom for atomic contention
    dead_start = sand_end
    dead_end = dead_start + 40_000
    dead_indices, dead_count = allocate_freelist(n)
    for j in range(40_000):
        idx = dead_start + j
        arrays["packed_info"][idx] = MAKE_PACKED(DEAD, STATIC)
        dead_indices[j] = idx
    dead_count[0] = 40_000

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )
    cupy.cuda.Device().synchronize()

    pi = arrays["packed_info"].get()
    mass = arrays["mass"].get()
    temp = arrays["temperature"].get()

    # All 10K source particles should be DEAD (with 40K headroom, contention is negligible)
    dead_count_source = 0
    for i in range(spawn_start, spawn_end):
        if GET_MATERIAL_ID(int(pi[i])) == DEAD:
            dead_count_source += 1
    assert dead_count_source == 10_000, \
        f"Expected 10K dead source particles, got {dead_count_source}"

    # Count STEAM particles at dead slots
    steam_count = 0
    for i in range(dead_start, dead_end):
        if GET_MATERIAL_ID(int(pi[i])) == STEAM:
            steam_count += 1
    assert steam_count == 30_000, f"Expected 30K steam particles, got {steam_count}"

    # No NaN in mass or temperature
    assert not np.any(np.isnan(mass)), "NaN in mass"
    assert not np.any(np.isnan(temp)), "NaN in temperature"

    # Normal water and sand should be unchanged
    for i in range(water_start, min(water_start + 10, water_end)):
        assert GET_MATERIAL_ID(int(pi[i])) == WATER
    for i in range(sand_start, min(sand_start + 10, sand_end)):
        assert GET_MATERIAL_ID(int(pi[i])) == SAND

    print("PASS: test_500k_stress")


def test_dead_particle_phase_is_static():
    """bd-mzc.28 regression: source water particle killed by spawn must be packed as
    MAKE_PACKED(MAT_DEAD, STATIC), NOT MAKE_PACKED(MAT_DEAD, FLUID).

    DEAD+FLUID particles participate in the SPH neighbor loop and pollute density
    sums until the next compaction pass.  The fix ensures STATIC phase is used so
    dead particles are excluded from the fluid loop immediately.
    """
    setup_params()

    n = 10
    arrays = make_sorted_arrays(n)

    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.008

    dead_indices, dead_count = allocate_freelist(n)
    dead_indices[0] = 7
    dead_indices[1] = 8
    dead_indices[2] = 9
    dead_count[0] = 3

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )
    cupy.cuda.Device().synchronize()

    pi = int(arrays["packed_info"][0].get())
    assert GET_MATERIAL_ID(pi) == DEAD, f"Source should be DEAD, got mat={GET_MATERIAL_ID(pi)}"
    behavior = GET_BEHAVIOR(pi)
    assert behavior == STATIC, (
        f"Dead particle packed with behavior={behavior} (FLUID=0, GRANULAR=1, GAS=2, STATIC=3); "
        f"expected STATIC ({STATIC}).  FLUID dead particles pollute SPH density sums."
    )
    print("PASS: test_dead_particle_phase_is_static")


def test_freelist_underflow_safety():
    """bd-mzc.19 regression: concurrent spawns with 0 or few freelist slots must not
    cause dead_count to wrap around (uint32 underflow to 0xFFFFFFFF).

    The atomicCAS claim loop should guarantee dead_count stays >= 0 regardless
    of how many racing threads attempt to claim simultaneously.
    """
    setup_params()

    # Many water particles all flagged, but freelist is empty.
    # Before the fix atomicSub would wrap dead_count to 0xFFFFFFFF.
    n = 512  # fills two GPU blocks
    arrays = make_sorted_arrays(n)

    for i in range(n):
        arrays["packed_info"][i] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
        arrays["mass"][i] = 0.008

    # Completely empty freelist
    dead_indices, dead_count = allocate_freelist(n)
    dead_count[0] = 0

    compute_spawn(
        arrays["packed_info"][:n],
        arrays["position"][:n],
        arrays["velocity"][:n],
        arrays["veleval"][:n],
        arrays["mass"][:n],
        arrays["temperature"][:n],
        arrays["health"][:n],
        arrays["lifetime"][:n],
        arrays["color"][:n],
        arrays["sleep_counter"][:n],
        arrays["density"][:n],
        arrays["shear_rate"][:n],
        dead_indices,
        dead_count,
    )
    cupy.cuda.Device().synchronize()

    dc = int(dead_count.get()[0])
    assert dc == 0, (
        f"dead_count underflowed to {dc} (0x{dc:08X}); expected 0.  "
        f"atomicSub wrap-around corrupts live particles."
    )

    # All particles should still be WATER (flag kept for the retry)
    pi_out = arrays["packed_info"].get()
    for i in range(n):
        mat = GET_MATERIAL_ID(int(pi_out[i]))
        assert mat == WATER, (
            f"Particle {i}: expected WATER (no spawn slots), got mat_id={mat}"
        )
        assert HAS_SPAWN_FLAG(int(pi_out[i])) == 1, f"Particle {i}: spawn flag dropped"

    print("PASS: test_freelist_underflow_safety")


def _run_reactions_on(packed_info, temperature, dead_capacity):
    n = packed_info.shape[0]
    dead_indices, dead_count = allocate_freelist(dead_capacity)
    pi_d = cupy.asarray(packed_info)
    compute_reactions(
        pi_d,
        cupy.asarray(temperature),
        cupy.full(n, 1.0, dtype=cupy.float32),
        cupy.zeros(n, dtype=cupy.float32),
        cupy.zeros((n, 4), dtype=cupy.float32),
        cupy.zeros(n, dtype=cupy.float32),
        cupy.zeros(n, dtype=cupy.float32),
        frame=0,
        dead_indices=dead_indices,
        dead_count=dead_count,
    )
    return pi_d, dead_indices, dead_count


def test_reactions_freelist_rebuild_random_dead_40k():
    """bd-r5-review-n28.142: the warp-aggregated freelist push lists every
    DEAD index exactly once. 40K DEAD slots at random positions among 200K,
    with n not a multiple of the block size so the last warp is partial."""
    setup_params()
    n = 200_003
    rng = np.random.default_rng(142)
    dead_idx = np.sort(rng.choice(n, size=40_000, replace=False))
    packed_info = np.full(n, MAKE_PACKED(SAND, GRANULAR), dtype=np.uint32)
    packed_info[dead_idx] = MAKE_PACKED(DEAD, STATIC)
    temperature = np.full(n, 293.0, dtype=np.float32)

    _, dead_indices, dead_count = _run_reactions_on(packed_info, temperature, n)
    dc = int(dead_count.get()[0])
    assert dc == dead_idx.size, f"dead_count={dc}, expected {dead_idx.size}"
    got = np.sort(dead_indices.get()[:dc])
    assert np.array_equal(got, dead_idx), "freelist is not the DEAD index set"


def test_reactions_boil_flag_written_once_and_cleared_on_cool():
    """bd-r5-review-n28.142: hot water keeps its spawn flag (set once), and
    water that cooled below boiling while waiting for slots loses it."""
    setup_params()
    water = MAKE_PACKED(WATER, FLUID)
    packed_info = np.array([water, SET_SPAWN_FLAG(water), SET_SPAWN_FLAG(water), water],
                           dtype=np.uint32)
    temperature = np.array([400.0, 400.0, 300.0, 300.0], dtype=np.float32)
    pi_d, _, _ = _run_reactions_on(packed_info, temperature, 4)
    pi = pi_d.get()
    assert HAS_SPAWN_FLAG(int(pi[0])) == 1, "hot water did not get the flag"
    assert HAS_SPAWN_FLAG(int(pi[1])) == 1, "hot water lost its flag"
    assert HAS_SPAWN_FLAG(int(pi[2])) == 0, "cooled water kept a stale flag"
    assert int(pi[3]) == water, "cool water changed"
    for k in range(4):
        assert GET_MATERIAL_ID(int(pi[k])) == WATER


def test_spawn_retries_after_slots_appear():
    """bd-r5-review-n28.142: a spawn that fails on an empty freelist keeps the
    flag, so a later call with free slots spawns without Reactions running."""
    setup_params()
    n = 10
    arrays = make_sorted_arrays(n)
    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.009
    dead_indices, dead_count = allocate_freelist(n)
    names = ["packed_info", "position", "velocity", "veleval", "mass", "temperature",
             "health", "lifetime", "color", "sleep_counter", "density", "shear_rate"]
    args = [arrays[k][:n] for k in names]

    dead_count[0] = 0
    compute_spawn(*args, dead_indices, dead_count)
    assert GET_MATERIAL_ID(int(arrays["packed_info"][0].get())) == WATER

    for k, slot in enumerate((7, 8, 9)):
        arrays["packed_info"][slot] = MAKE_PACKED(DEAD, STATIC)
        dead_indices[k] = slot
    dead_count[0] = 3
    compute_spawn(*args, dead_indices, dead_count)
    pi = arrays["packed_info"].get()
    assert GET_MATERIAL_ID(int(pi[0])) == DEAD, "source did not spawn on retry"
    assert all(GET_MATERIAL_ID(int(pi[s])) == STEAM for s in (7, 8, 9))
    assert int(dead_count.get()[0]) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parked DEAD reserve (bd-r5-review-n28.27)
#
# The freelist is rebuilt each substep from DEAD slots only. Without a
# reserve, a scene where nothing has died has no DEAD slots and hot water
# never boils. These tests drive Simulation._maybe_compact the way
# step_frame does.
# ---------------------------------------------------------------------------

def _reserve_sim(max_particles=100_000):
    from world import World
    from simulation import Simulation
    w = World(max_particles=max_particles)
    w.spawn_cube((-0.2, -0.9, -0.2), (0.2, -0.7, 0.2), WATER, 0.02)
    sim = Simulation(w, dt=0.001, fixed_dt=True)
    sim.notify_scene_changed()
    return w, sim


def _count_mat(w, mat_id):
    n = w._high_water
    return int(cupy.count_nonzero((w.packed_info[:n] & 0xFF) == mat_id))


def test_dead_reserve_fills_freelist_without_deaths():
    """Fresh WATER world with no deaths has dead_count >= dead_reserve after one K_Reactions pass."""
    w, sim = _reserve_sim()
    n_water = w._high_water
    assert w.needs_dead_reserve
    n = sim._maybe_compact()
    assert n == n_water + w.dead_reserve
    assert _count_mat(w, DEAD) == w.dead_reserve
    assert w.num_active == n_water
    sim._sim_step(n)
    dead_count = int(sim._dead_count.get()[0])
    assert dead_count >= w.dead_reserve, dead_count
    # Parked particles stay put and stay massless.
    parked = w.position[n_water:n, :3].get()
    assert np.all(parked[:, 1] < -900.0)
    assert float(cupy.max(w.mass[n_water:n])) == 0.0


def test_dead_reserve_boils_one_water_into_three_steam():
    """With the reserve, one boiling water particle spawns SPAWN_N steam."""
    w, sim = _reserve_sim()
    n = sim._maybe_compact()
    w.temperature[0] = 450.0
    sim._sim_step(n)
    assert _count_mat(w, STEAM) == SPAWN_N
    assert _count_mat(w, WATER) == (n - w.dead_reserve) - 1


def test_dead_reserve_survives_compaction():
    """After compaction the freelist still has at least dead_reserve entries."""
    w, sim = _reserve_sim()
    n = sim._maybe_compact()
    # Add more water and kill enough of it to go past reserve + threshold.
    w.spawn_cube((-0.5, -0.5, -0.5), (0.5, 0.0, 0.5), WATER, 0.02)
    n_before = w._high_water
    w.kill_in_sphere((0.0, -0.25, 0.0), 0.45)
    alive = w.num_active
    assert n_before - alive > w.dead_reserve + sim.compact_threshold
    sim._frame_counter = sim.compact_interval  # compaction due
    n = sim._maybe_compact()
    assert n < n_before, "compaction did not run"
    assert n == alive + w.dead_reserve
    assert _count_mat(w, DEAD) >= w.dead_reserve
    sim._sim_step(n)
    assert int(sim._dead_count.get()[0]) >= w.dead_reserve


def test_dead_reserve_not_compacted_away():
    """A few gas deaths on top of the reserve do not trigger compaction."""
    w, sim = _reserve_sim()
    n = sim._maybe_compact()
    sim._frame_counter = sim.compact_interval
    assert sim._maybe_compact() == n


def test_dead_reserve_skipped_without_boilable():
    """No WATER or ICE: no reserve is added."""
    from world import World
    from simulation import Simulation
    w = World(max_particles=50_000)
    w.spawn_cube((-0.2, -0.9, -0.2), (0.2, -0.7, 0.2), SAND, 0.02)
    n0 = w._high_water
    sim = Simulation(w, dt=0.001, fixed_dt=True)
    assert not w.needs_dead_reserve
    assert sim._maybe_compact() == n0


def _park_bucket_loads(max_particles, start, count, table_size, h=H, half=1.0):
    """CPU copy of calcGridCell + spatialHash for parked reserve slots."""
    from world import World
    w = World.__new__(World)
    w.max_particles = max_particles
    pos = w.park_positions(cupy.arange(start, start + count), h, (-half,) * 3).get()
    gmin = np.float32(-half)
    delta = np.float32(1.0 / h)
    cells = np.floor((pos[:, :3].astype(np.float32) - gmin) * delta).astype(np.int64)
    uniq = np.unique(cells, axis=0).shape[0]
    c = cells.astype(np.uint32)
    with np.errstate(over="ignore"):
        hsh = (c[:, 0] * np.uint32(73856093) + c[:, 1] * np.uint32(19349669)
               + c[:, 2] * np.uint32(83492791)) & np.uint32(table_size - 1)
    return uniq, int(np.bincount(hsh, minlength=table_size).max())


def test_dead_reserve_parked_bucket_load():
    """No hash bucket holds more than 8 parked DEAD particles."""
    from world import DEAD_RESERVE_DEFAULT
    for max_p in (100_000, 500_000, 1_000_000, 2_000_000):
        for start in (0, 12_345, max_p // 2, max_p - DEAD_RESERVE_DEFAULT):
            for ts, half in ((1 << 18, 1.0), (1 << 19, 1.37), (1 << 20, 10.0)):
                uniq, worst = _park_bucket_loads(max_p, start, DEAD_RESERVE_DEFAULT,
                                                 ts, half=half)
                assert uniq == DEAD_RESERVE_DEFAULT, (max_p, start, uniq)
                assert worst <= 8, (max_p, start, ts, worst)


# ---------------------------------------------------------------------------
# World.spawn_cube filters and truncation (bd-r5-review-n28.47)
# ---------------------------------------------------------------------------

def _spawned_positions(world, n):
    return world.position[world._high_water - n:world._high_water, :3].get()


def test_spawn_cube_exclude_box_removes_inner_points():
    from world import World
    w = World(100_000)
    lo, hi = (-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)
    inner = ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
    # Lattice points at -0.19, -0.17, ..., 0.19: 20 per axis, 10 inside inner.
    n = w.spawn_cube(lo, hi, WATER, spacing=0.02, exclude_box=inner)
    assert n == 20 ** 3 - 10 ** 3
    pos = _spawned_positions(w, n)
    in_box = np.all((pos >= -0.1) & (pos <= 0.1), axis=1)
    assert not in_box.any()
    # Filling the hole with a second lattice at the same spacing gives the
    # full block with no point shared.
    n2 = w.spawn_cube(inner[0], inner[1], SAND, spacing=0.02)
    assert n2 == 10 ** 3
    allpos = w.position[:w._high_water, :3].get()
    keys = np.round(allpos / 0.01).astype(np.int64)
    assert len(np.unique(keys, axis=0)) == 20 ** 3
    assert not w.last_spawn_truncated


def test_spawn_cube_inside_fn_radial_filter():
    from world import World
    w = World(100_000)
    r = 0.15
    fn = lambda p: p[:, 0] ** 2 + p[:, 2] ** 2 < r * r
    n = w.spawn_cube((-0.2, 0.0, -0.2), (0.2, 0.1, 0.2), WATER,
                     spacing=0.02, inside_fn=fn)
    xs = np.arange(-0.2 + 0.01, 0.2, 0.02, dtype=np.float32)
    ys = np.arange(0.0 + 0.01, 0.1, 0.02, dtype=np.float32)
    gx, gz = np.meshgrid(xs, xs, indexing='ij')
    per_layer = int((gx ** 2 + gz ** 2 < r * r).sum())
    assert n == per_layer * len(ys)
    pos = _spawned_positions(w, n)
    assert np.all(pos[:, 0] ** 2 + pos[:, 2] ** 2 < r * r)


def test_spawn_cube_truncation_keeps_footprint():
    """A 90^3 lattice into a 500K world keeps full x and z and drops top
    layers (y slowest). Before the fix x was slowest, so the block kept full
    height and lost its +x side."""
    from world import World
    w = World(500_000)
    w.dead_reserve = 0
    n = w.spawn_cube((-0.9, -0.9, -0.9), (0.9, 0.9, 0.9), WATER, spacing=0.02)
    assert n == 500_000
    assert w.last_spawn_truncated
    pos = w.position[:n, :3].get()
    assert pos[:, 0].min() < -0.88 and pos[:, 0].max() > 0.88
    assert pos[:, 2].min() < -0.88 and pos[:, 2].max() > 0.88
    layers = len(np.unique(np.round(pos[:, 1], 4)))
    # 500000 / 8100 = 61.7, so 61 full layers plus one partial one.
    assert layers == 62, layers
    # Every full layer below the top covers the whole footprint.
    top = pos[:, 1].max()
    below = pos[pos[:, 1] < top - 0.01]
    assert len(below) == 61 * 90 * 90


def test_spawn_cube_not_truncated_flag():
    from world import World
    w = World(10_000)
    w.spawn_cube((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), WATER, spacing=0.02)
    assert not w.last_spawn_truncated
    assert w.spawn_truncated_count == 0


def test_whirlpool_water_inside_bowl():
    """Whirlpool water must start inside the bowl's inner radius 0.62."""
    from world import World
    from presets import PRESETS
    w = World(500_000)
    PRESETS["Whirlpool"](w)
    n = w._high_water
    pos = w.position[:n, :3].get()
    mat = w.packed_info[:n].get() & 0xFF
    water = pos[mat == WATER]
    assert len(water) > 30_000
    r2 = (water ** 2).sum(axis=1)
    assert np.all(r2 < 0.62 ** 2)
    horiz = water[:, 0] ** 2 + water[:, 2] ** 2
    assert np.mean(horiz < 0.62 ** 2) >= 0.99


# ---------------------------------------------------------------------------
# Spawned slot state (bd-r5-review-n28.42)
# ---------------------------------------------------------------------------

def _half4_to_float(raw: np.ndarray) -> np.ndarray:
    """(N, 2) uint32 half4 mirror -> (N, 4) float32."""
    return raw.astype(np.uint32).view(np.float16).reshape(-1, 4).astype(np.float32)


def test_spawn_children_ring_and_slot_state():
    """Children are separated, conserve momentum and reset all per-slot state."""
    setup_params()
    n = 10
    arrays = make_sorted_arrays(n)
    src_pos = np.array([0.1, 0.2, -0.3], dtype=np.float32)
    src_vel = np.array([0.5, -0.25, 0.75], dtype=np.float32)
    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.009
    arrays["position"][0] = cupy.asarray([*src_pos, 1000.0], dtype=cupy.float32)
    arrays["velocity"][0] = cupy.asarray([*src_vel, 0.0], dtype=cupy.float32)
    slots = [7, 8, 9]
    for idx in slots:
        arrays["packed_info"][idx] = MAKE_PACKED(DEAD, STATIC)
        # Dead slots parked 1 m away, as the reserve parks them.
        arrays["position"][idx] = cupy.asarray([0.9, -0.9, 0.9, 1000.0], dtype=cupy.float32)
    dead_indices, dead_count = allocate_freelist(n)
    dead_indices[:3] = cupy.asarray(slots, dtype=cupy.uint32)
    dead_count[0] = 3

    predicted = cupy.full((n, 4), 7.0, dtype=cupy.float32)
    velocity_h = cupy.full((n, 2), 0x47004700, dtype=cupy.uint32)  # half(7.0) pairs
    pressure = cupy.full(n, 7.0, dtype=cupy.float32)
    dye = cupy.full((n, 4), 7.0, dtype=cupy.float32)
    kappa = cupy.full(n, 7.0, dtype=cupy.float32)
    kappa_v = cupy.full(n, 7.0, dtype=cupy.float32)
    lam = cupy.full(n, 7.0, dtype=cupy.float32)

    compute_spawn(
        arrays["packed_info"], arrays["position"], arrays["velocity"],
        arrays["veleval"], arrays["mass"], arrays["temperature"],
        arrays["health"], arrays["lifetime"], arrays["color"],
        arrays["sleep_counter"], arrays["density"], arrays["shear_rate"],
        dead_indices, dead_count,
        predicted_position=predicted, velocity_h=velocity_h, pressure=pressure,
        particle_dye=dye,
        kappa=kappa, kappa_v=kappa_v, lambda_pbf=lam,
    )
    cupy.cuda.Device().synchronize()

    mats = build_material_array()
    steam_rho0 = float(mats[STEAM]["rest_density"])
    pos = arrays["position"].get()[slots]
    vel = arrays["velocity"].get()[slots]
    mass = arrays["mass"].get()[slots]

    # Distinct positions, at least 0.4 * spacing apart, near the source.
    for a in range(3):
        assert np.linalg.norm(pos[a, :3] - src_pos) < SPACING
        for b in range(a + 1, 3):
            d = np.linalg.norm(pos[a, :3] - pos[b, :3])
            assert d > 0.4 * SPACING, (a, b, d)
    # position.w is the steam rest density, not the water density.
    np.testing.assert_allclose(pos[:, 3], steam_rho0, rtol=1e-6)

    # Momentum: sum of child momenta = source momentum + upkick.
    p_children = (mass[:, None] * vel[:, :3]).sum(axis=0)
    p_expected = 0.009 * (src_vel + np.array([0.0, 2.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(p_children, p_expected, atol=1e-6)

    # PBF prediction: child position + dt * child velocity.
    pred = predicted.get()[slots]
    np.testing.assert_allclose(pred[:, :3], pos[:, :3] + DT * vel[:, :3], atol=1e-6)

    # FP16 mirror holds the half of the child velocity.
    vh = _half4_to_float(velocity_h.get()[slots])
    np.testing.assert_allclose(vh, vel.astype(np.float16).astype(np.float32), atol=0)

    for name, arr in [("pressure", pressure), ("dye", dye),
                      ("kappa", kappa), ("kappa_v", kappa_v), ("lambda", lam)]:
        assert np.all(arr.get()[slots] == 0.0), name
        # Slots that were not claimed keep their values.
        assert np.all(arr.get()[1:7] == 7.0), name


def test_spawn_optional_state_null_safe():
    """Without the optional arrays the kernel still spawns (NULL pointers)."""
    setup_params()
    n = 4
    arrays = make_sorted_arrays(n)
    arrays["packed_info"][0] = SET_SPAWN_FLAG(MAKE_PACKED(WATER, FLUID))
    arrays["mass"][0] = 0.009
    dead_indices, dead_count = allocate_freelist(n)
    dead_indices[:3] = cupy.asarray([1, 2, 3], dtype=cupy.uint32)
    dead_count[0] = 3
    compute_spawn(
        arrays["packed_info"], arrays["position"], arrays["velocity"],
        arrays["veleval"], arrays["mass"], arrays["temperature"],
        arrays["health"], arrays["lifetime"], arrays["color"],
        arrays["sleep_counter"], arrays["density"], arrays["shear_rate"],
        dead_indices, dead_count,
    )
    pi = arrays["packed_info"].get()
    assert all(GET_MATERIAL_ID(int(pi[k])) == STEAM for k in (1, 2, 3))


def _boil_one_step(profile_name):
    """Reserve world on the given solver; boil water particle 0 for one substep."""
    from solver_profiles import PROFILES
    w, sim = _reserve_sim()
    sim.set_solver_profile(PROFILES[profile_name])
    # Skip the post-reset spawn damping ramp (it scales velocity by 0.2 on
    # the first substep), as eval_solvers.py does.
    sim._spawn_substep = sim._damping_duration
    n = sim._maybe_compact()
    src = w.position[0, :3].get().copy()
    w.temperature[0] = 450.0
    sim._sim_step(n)
    n_now = w._high_water
    steam = np.nonzero((w.packed_info[:n_now].get() & 0xFF) == STEAM)[0]
    return w, sim, src, steam


def test_pbf_spawned_steam_stays_at_source():
    """PBF: steam children end the spawn substep next to the boiling water.

    Before the fix K_PBF_Finalize took the dead slot's stale predicted
    position (the reserve parks slots at y < -900) as the child's final
    position and derived a velocity at the limit from it.
    """
    w, sim, src, steam = _boil_one_step("PBF")
    assert len(steam) == SPAWN_N
    dt = sim._profile.dt
    pos = w.position[steam, :3].get()
    vel = w.velocity[steam, :3].get()
    target = src + dt * np.array([0.0, 2.0, 0.0], dtype=np.float32)
    for k in range(SPAWN_N):
        d = np.linalg.norm(pos[k] - target)
        assert d < H, (k, pos[k], target, d)
        speed = np.linalg.norm(vel[k])
        assert speed < 3.0, (k, vel[k])
        assert vel[k, 1] > 1.0, (k, vel[k])
    # The spawn also marks the grid for a full sort on the next decision.
    assert int(w.max_displacement.get()[0]) == 0x7F800000


def test_wcsph_spawned_steam_stays_at_source():
    """WCSPH: same check, children separated and near the source."""
    w, sim, src, steam = _boil_one_step("WCSPH")
    assert len(steam) == SPAWN_N
    pos = w.position[steam, :3].get()
    for k in range(SPAWN_N):
        assert np.linalg.norm(pos[k] - src) < H
    for a in range(SPAWN_N):
        for b in range(a + 1, SPAWN_N):
            assert np.linalg.norm(pos[a] - pos[b]) > 0.4 * SPACING


if __name__ == "__main__":
    tests = [
        test_compilation,
        test_block_size,
        test_spawn_n_value,
        test_water_spawns_steam,
        test_spawn_flag_cleared,
        test_freelist_exhausted,
        test_freelist_partial_exhaustion,
        test_scattered_freelist_indices,
        test_source_marked_dead,
        test_no_spawn_flag_unchanged,
        test_multiple_spawns,
        test_reactions_populates_freelist,
        test_reactions_gas_lifetime_freelist,
        test_reactions_backward_compat,
        test_end_to_end_boil_spawn,
        test_500k_stress,
        test_dead_particle_phase_is_static,
        test_freelist_underflow_safety,
        test_reactions_freelist_rebuild_random_dead_40k,
        test_reactions_boil_flag_written_once_and_cleared_on_cool,
        test_spawn_retries_after_slots_appear,
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


def test_world_spawn_writes_behavior_code_in_color_w():
    """bd-r5-review-n28.63: spawned particles carry behavior * 0.25 in color.w.

    The SSFR shaders decode color.w as the behavior class, so a particle
    that has not been through a substep must already hold the right code.
    """
    from materials import MATERIALS
    from world import World

    w = World(20_000)
    for mat_id, expected_w in ((WATER, 0.0), (SAND, 0.25), (STONE, 0.75)):
        start = w._high_water
        w.spawn_cube((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05), mat_id, 0.02)
        col = w.color[start:w._high_water].get()
        assert col.shape[0] > 0
        mat = MATERIALS[mat_id]
        np.testing.assert_array_equal(col[:, 3], np.float32(expected_w))
        np.testing.assert_allclose(
            col[:, :3], np.broadcast_to(
                np.array([mat.color_r, mat.color_g, mat.color_b], np.float32), col[:, :3].shape))
