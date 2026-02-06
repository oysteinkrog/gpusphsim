"""Integration test for integrate.py -- K_Integrate kernel.

Acceptance criteria:
  - Symplectic Euler: vel_new = vel + dt * accel; pos_new uses XSPH for FLUID
  - Impulse SDF boundary: 6 planes of box (-1,-1,-1) to (1,1,1), restitution=0.3,
    Coulomb friction mu_wall=0.5
  - STATIC particles skipped (position unchanged)
  - GAS buoyancy: beta=0.01 * (T-293) * (0, 9.81, 0)
  - GAS drag: vel *= (1 - 2.0 * dt)
  - Velocity clamp at 50.0
  - Color from c_materials, tinted red for T>293K, faded by health
  - Writeback to UNSORTED arrays via sort_indexes
  - Bounce test: particle dropped from height bounces and loses energy
  - 10K water pool test: no NaN after 1000 steps
  - Block size = 256, 500K stress test

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import math
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
    FLUID,
    GRANULAR,
    GAS,
    STATIC,
    WATER,
    SAND,
    STEAM,
    STONE,
    build_material_array,
)
from integrate import (
    BLOCK_SIZE,
    get_module,
    integrate,
    upload_materials,
    upload_sim_params,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H = 0.04
SPACING = 0.02
RHO0 = 1000.0
MASS = RHO0 * SPACING**3  # 0.008 kg
DT = 0.001

# packed_info helpers
def MAKE_PACKED(mat_id: int, behavior: int) -> int:
    return (mat_id & 0xFF) | ((behavior & 0x3) << 8)

def SET_SLEEPING(p: int) -> int:
    return p | 0x400

# ---------------------------------------------------------------------------
# Helper: upload params for all tests
# ---------------------------------------------------------------------------

def setup_params(dt=DT, restitution=0.3, wall_friction=0.5,
                 gravity=(0.0, -9.8, 0.0)):
    """Upload SimParams and materials to integrate module's constant memory."""
    sim_params = build_sim_params(
        smoothing_length=H,
        particle_mass=MASS,
        particle_spacing=SPACING,
        gravity=gravity,
        dt=dt,
        restitution=restitution,
        wall_friction=wall_friction,
        world_min=(-1.0, -1.0, -1.0),
        world_max=(1.0, 1.0, 1.0),
    )
    upload_sim_params(sim_params)
    materials = build_material_array()
    upload_materials(materials)


def make_simple_particles(n, mat_id, behavior, pos=None, vel=None,
                          temp=293.0, health=1.0):
    """Create sorted particle arrays for n particles. sort_indexes = identity."""
    if pos is None:
        pos = np.zeros((n, 4), dtype=np.float32)
        pos[:, 3] = 1.0
    else:
        pos = np.array(pos, dtype=np.float32).reshape(n, 4)

    if vel is None:
        vel = np.zeros((n, 4), dtype=np.float32)
    else:
        vel = np.array(vel, dtype=np.float32).reshape(n, 4)

    veleval = vel.copy()  # same as vel unless XSPH
    sph_force = np.zeros((n, 4), dtype=np.float32)
    mass = np.full(n, MASS, dtype=np.float32)
    packed_info = np.full(n, MAKE_PACKED(mat_id, behavior), dtype=np.uint32)
    temperature = np.full(n, temp, dtype=np.float32)
    hlth = np.full(n, health, dtype=np.float32)
    sort_idx = np.arange(n, dtype=np.uint32)

    return {
        "sorted_position": cupy.asarray(pos),
        "sorted_velocity": cupy.asarray(vel),
        "sorted_veleval": cupy.asarray(veleval),
        "sorted_sph_force": cupy.asarray(sph_force),
        "sorted_mass": cupy.asarray(mass),
        "sorted_packed_info": cupy.asarray(packed_info),
        "sorted_temperature": cupy.asarray(temperature),
        "sorted_health": cupy.asarray(hlth),
        "sort_indexes": cupy.asarray(sort_idx),
    }


# ===========================================================================
# Tests
# ===========================================================================


def test_compilation():
    """integrate.cu compiles without errors via CuPy RawModule."""
    module = get_module()
    kernel = module.get_function("K_Integrate")
    assert kernel is not None
    print("PASS: test_compilation")


def test_block_size():
    """Block size is 256 per acceptance criteria."""
    assert BLOCK_SIZE == 256
    print("PASS: test_block_size")


def test_static_skip():
    """STATIC particles are skipped -- position and velocity unchanged."""
    setup_params()
    n = 4
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0] = [0.0, 0.1, 0.2, 0.3]
    pos[:, 1] = [0.5, 0.5, 0.5, 0.5]
    pos[:, 3] = 1.0
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 1] = [-1.0, -1.0, -1.0, -1.0]  # downward velocity

    d = make_simple_particles(n, STONE, STATIC, pos=pos, vel=vel)

    pos_out, vel_out, color_out, _, _, _ = integrate(**d)

    pos_h = pos_out.get()
    vel_h = vel_out.get()

    # STATIC: position should be unchanged, velocity should be zero
    for i in range(n):
        assert abs(pos_h[i, 0] - pos[i, 0]) < 1e-6, f"STATIC pos.x changed for particle {i}"
        assert abs(pos_h[i, 1] - pos[i, 1]) < 1e-6, f"STATIC pos.y changed for particle {i}"
        assert abs(vel_h[i, 0]) < 1e-6, f"STATIC vel.x non-zero for particle {i}"
        assert abs(vel_h[i, 1]) < 1e-6, f"STATIC vel.y non-zero for particle {i}"
        assert abs(vel_h[i, 2]) < 1e-6, f"STATIC vel.z non-zero for particle {i}"

    print("PASS: test_static_skip")


def test_gravity_freefall():
    """A particle in free fall (no SPH force) accelerates downward correctly."""
    setup_params(gravity=(0.0, -9.8, 0.0))
    n = 1
    pos = np.array([[0.0, 0.5, 0.0, 1.0]], dtype=np.float32)
    vel = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    pos_out, vel_out, _, _, _, _ = integrate(**d)

    vel_h = vel_out.get()
    pos_h = pos_out.get()

    # After 1 step: vel_y = 0 + dt * (-9.8) = -0.0098
    expected_vy = -9.8 * DT
    assert abs(vel_h[0, 1] - expected_vy) < 1e-5, (
        f"Expected vel_y={expected_vy}, got {vel_h[0, 1]}"
    )

    # Position: pos_y = 0.5 + dt * vel_new_y = 0.5 + 0.001 * (-0.0098) = 0.4999902
    expected_py = 0.5 + DT * expected_vy
    assert abs(pos_h[0, 1] - expected_py) < 1e-5, (
        f"Expected pos_y={expected_py}, got {pos_h[0, 1]}"
    )

    print("PASS: test_gravity_freefall")


def test_bounce_floor():
    """Particle dropped from height bounces off floor boundary and loses energy."""
    setup_params(restitution=0.3, wall_friction=0.5, gravity=(0.0, -9.8, 0.0))

    n = 1
    # Place particle at bottom boundary with downward velocity
    # Position just above floor, velocity pushing through it
    pos = np.array([[0.0, -0.99, 0.0, 1.0]], dtype=np.float32)
    vel = np.array([[0.0, -5.0, 0.0, 0.0]], dtype=np.float32)

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    pos_out, vel_out, _, _, _, _ = integrate(**d)

    vel_h = vel_out.get()
    pos_h = pos_out.get()

    # After integration: vel_new_y = -5.0 + 0.001*(-9.8) = -5.0098
    # pos_new_y = -0.99 + 0.001 * (-5.0098) = -0.9950098  -> penetrates floor at -1.0?
    # No, -0.9950098 > -1.0 so no penetration yet.
    # Let's use a bigger timestep or position closer to boundary.
    # Better test: place particle so that after integration it penetrates.

    # Re-test with position that definitely penetrates
    pos2 = np.array([[0.0, -0.995, 0.0, 1.0]], dtype=np.float32)
    vel2 = np.array([[0.0, -10.0, 0.0, 0.0]], dtype=np.float32)
    d2 = make_simple_particles(n, WATER, FLUID, pos=pos2, vel=vel2)
    pos_out2, vel_out2, _, _, _, _ = integrate(**d2)

    vel_h2 = vel_out2.get()
    pos_h2 = pos_out2.get()

    # vel_new_y = -10.0 + 0.001*(-9.8) = -10.0098
    # pos_new_y = -0.995 + 0.001*(-10.0098) = -1.0050098 -> penetrates!
    # After boundary: pos_y = -1.0, vel_y = 0.3 * 10.0098 = ~3.003

    assert pos_h2[0, 1] >= -1.0 - 1e-5, (
        f"Particle below floor: pos_y={pos_h2[0, 1]}"
    )
    # Velocity should be positive (bounced up) and less than original magnitude
    assert vel_h2[0, 1] > 0.0, (
        f"Expected positive bounce vel_y, got {vel_h2[0, 1]}"
    )
    assert abs(vel_h2[0, 1]) < 10.1, (
        f"Bounce vel_y too large (no energy loss): {vel_h2[0, 1]}"
    )
    # With restitution=0.3, bounced vel_y should be about 3.0
    assert abs(vel_h2[0, 1] - 0.3 * 10.0098) < 0.1, (
        f"Expected bounce vel_y ~{0.3*10.0098}, got {vel_h2[0, 1]}"
    )

    print("PASS: test_bounce_floor")


def test_bounce_multi_step():
    """Particle dropped from height loses energy over multiple bounces."""
    setup_params(restitution=0.3, wall_friction=0.5, gravity=(0.0, -9.8, 0.0))

    n = 1
    pos_np = np.array([[0.0, 0.5, 0.0, 1.0]], dtype=np.float32)
    vel_np = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    pos_gpu = cupy.asarray(pos_np)
    vel_gpu = cupy.asarray(vel_np)

    pi_gpu = cupy.full(n, MAKE_PACKED(WATER, FLUID), dtype=cupy.uint32)
    sc_gpu = cupy.zeros(n, dtype=cupy.uint8)

    # Run 2000 steps
    for step in range(2000):
        d = {
            "sorted_position": pos_gpu,
            "sorted_velocity": vel_gpu,
            "sorted_veleval": vel_gpu.copy(),
            "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
            "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
            "sorted_packed_info": pi_gpu,
            "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
            "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
            "sorted_sleep_counter": sc_gpu,
            "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
            "position_out": pos_gpu,
            "velocity_out": vel_gpu,
            "color_out": cupy.zeros((n, 4), dtype=cupy.float32),
            "packed_info_out": pi_gpu,
            "sleep_counter_out": sc_gpu,
        }
        integrate(**d)

    pos_h = pos_gpu.get()
    vel_h = vel_gpu.get()

    # After many bounces, particle should have settled near bottom
    assert pos_h[0, 1] >= -1.0 - 1e-5, f"Below floor: {pos_h[0, 1]}"
    # Velocity should be small (energy dissipated through bounces)
    speed = math.sqrt(vel_h[0, 0]**2 + vel_h[0, 1]**2 + vel_h[0, 2]**2)
    assert speed < 2.0, f"Too much residual speed after bouncing: {speed}"
    # No NaN
    assert not np.any(np.isnan(pos_h)), "NaN in position"
    assert not np.any(np.isnan(vel_h)), "NaN in velocity"

    print("PASS: test_bounce_multi_step")


def test_gas_buoyancy():
    """GAS particles with T > 293K get upward buoyancy force."""
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 1
    temp = 500.0  # hot gas
    pos = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    vel = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    d = make_simple_particles(n, STEAM, GAS, pos=pos, vel=vel, temp=temp)
    _, vel_out, _, _, _, _ = integrate(**d)

    vel_h = vel_out.get()

    # Expected: accel_y = -9.8 + 0.01 * (500 - 293) * 9.81 = -9.8 + 20.306 = +10.506
    # vel_new_y = 0 + 0.001 * 10.506 = +0.010506
    # Then drag: vel_new_y *= (1 - 2.0 * 0.001) = 0.998 -> ~0.01049
    expected_buoyancy = 0.01 * (500.0 - 293.0) * 9.81
    expected_accel_y = -9.8 + expected_buoyancy
    expected_vel_y = DT * expected_accel_y * (1.0 - 2.0 * DT)

    assert vel_h[0, 1] > 0, (
        f"Expected positive vel_y for hot gas, got {vel_h[0, 1]}"
    )
    assert abs(vel_h[0, 1] - expected_vel_y) < 1e-4, (
        f"Expected vel_y ~{expected_vel_y}, got {vel_h[0, 1]}"
    )

    print("PASS: test_gas_buoyancy")


def test_gas_drag():
    """GAS drag reduces velocity by (1 - c_drag * dt) factor."""
    setup_params(gravity=(0.0, 0.0, 0.0))  # no gravity for clean test

    n = 1
    vel = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pos = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    d = make_simple_particles(n, STEAM, GAS, pos=pos, vel=vel, temp=293.0)
    _, vel_out, _, _, _, _ = integrate(**d)

    vel_h = vel_out.get()

    # No SPH force, no gravity, no buoyancy (T=293K):
    # vel_new = 10.0, then drag: vel *= (1 - 2.0*0.001) = 0.998
    expected_vx = 10.0 * (1.0 - 2.0 * DT)
    assert abs(vel_h[0, 0] - expected_vx) < 1e-4, (
        f"Expected vx={expected_vx}, got {vel_h[0, 0]}"
    )

    print("PASS: test_gas_drag")


def test_velocity_clamp():
    """Velocity magnitude is clamped to 50.0."""
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 1
    # Start with velocity near the limit, add moderate force to push over 50
    # accel = force/mass = 20/0.008 = 2500 (below accel_max 5000)
    # vel_new = 48 + dt*2500 = 48 + 2.5 = 50.5 -> clamped to 50
    pos = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    vel = np.array([[48.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    sph_force = np.array([[20.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    d["sorted_sph_force"] = cupy.asarray(sph_force)
    _, vel_out, _, _, _, _ = integrate(**d)

    vel_h = vel_out.get()
    speed = math.sqrt(vel_h[0, 0]**2 + vel_h[0, 1]**2 + vel_h[0, 2]**2)

    assert speed <= 50.0 + 0.1, (
        f"Velocity not clamped: speed={speed}"
    )
    assert speed > 49.0, (
        f"Velocity too low after clamp: speed={speed}"
    )

    print("PASS: test_velocity_clamp")


def test_color_computation():
    """Color is computed from material base color, temperature, and health."""
    setup_params()

    # Test 1: room temp, full health -> base material color
    n = 1
    d = make_simple_particles(n, WATER, FLUID, temp=293.0, health=1.0)
    d["sorted_position"] = cupy.array([[0.0, 0.0, 0.0, 1.0]], dtype=cupy.float32)
    _, _, color_out, _, _, _ = integrate(**d)
    color_h = color_out.get()

    # Water color: (0.2, 0.5, 0.9)
    assert abs(color_h[0, 0] - 0.2) < 0.02, f"Water R={color_h[0, 0]}, expected 0.2"
    assert abs(color_h[0, 1] - 0.5) < 0.02, f"Water G={color_h[0, 1]}, expected 0.5"
    assert abs(color_h[0, 2] - 0.9) < 0.02, f"Water B={color_h[0, 2]}, expected 0.9"
    assert abs(color_h[0, 3] - 1.0) < 0.01, f"Alpha={color_h[0, 3]}, expected 1.0"

    # Test 2: hot particle -> red tint
    d2 = make_simple_particles(n, WATER, FLUID, temp=1293.0, health=1.0)
    d2["sorted_position"] = cupy.array([[0.0, 0.0, 0.0, 1.0]], dtype=cupy.float32)
    _, _, color_out2, _, _, _ = integrate(**d2)
    color_h2 = color_out2.get()

    # Should be more red than base water
    assert color_h2[0, 0] > color_h[0, 0], "Hot particle not redder"
    assert color_h2[0, 2] < color_h[0, 2], "Hot particle blue not reduced"

    # Test 3: low health -> faded color
    d3 = make_simple_particles(n, WATER, FLUID, temp=293.0, health=0.5)
    d3["sorted_position"] = cupy.array([[0.0, 0.0, 0.0, 1.0]], dtype=cupy.float32)
    _, _, color_out3, _, _, _ = integrate(**d3)
    color_h3 = color_out3.get()

    # Should be ~half brightness
    assert abs(color_h3[0, 0] - 0.1) < 0.02, f"Health fade R={color_h3[0, 0]}"
    assert abs(color_h3[0, 1] - 0.25) < 0.02, f"Health fade G={color_h3[0, 1]}"

    print("PASS: test_color_computation")


def test_sort_indexes_writeback():
    """Results are written to UNSORTED arrays using sort_indexes mapping."""
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 4
    # Sorted order: 0,1,2,3 maps to unsorted: 3,1,0,2
    sort_idx = np.array([3, 1, 0, 2], dtype=np.uint32)

    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0] = [0.0, 0.1, 0.2, 0.3]
    pos[:, 1] = 0.5
    pos[:, 3] = 1.0
    vel = np.zeros((n, 4), dtype=np.float32)

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    d["sort_indexes"] = cupy.asarray(sort_idx)

    out_pos = cupy.zeros((n, 4), dtype=cupy.float32)
    out_vel = cupy.zeros((n, 4), dtype=cupy.float32)
    out_col = cupy.zeros((n, 4), dtype=cupy.float32)

    out_pi = cupy.zeros(n, dtype=cupy.uint32)
    out_sc = cupy.zeros(n, dtype=cupy.uint8)
    integrate(**d, position_out=out_pos, velocity_out=out_vel, color_out=out_col,
              packed_info_out=out_pi, sleep_counter_out=out_sc)

    pos_h = out_pos.get()

    # Sorted particle 0 (x=0.0) -> unsorted slot 3
    assert abs(pos_h[3, 0] - 0.0) < 1e-4, f"Sort writeback failed: slot 3 x={pos_h[3, 0]}"
    # Sorted particle 1 (x=0.1) -> unsorted slot 1
    assert abs(pos_h[1, 0] - 0.1) < 1e-4, f"Sort writeback failed: slot 1 x={pos_h[1, 0]}"
    # Sorted particle 2 (x=0.2) -> unsorted slot 0
    assert abs(pos_h[0, 0] - 0.2) < 1e-4, f"Sort writeback failed: slot 0 x={pos_h[0, 0]}"
    # Sorted particle 3 (x=0.3) -> unsorted slot 2
    assert abs(pos_h[2, 0] - 0.3) < 1e-4, f"Sort writeback failed: slot 2 x={pos_h[2, 0]}"

    print("PASS: test_sort_indexes_writeback")


def test_boundary_all_walls():
    """Particles are contained by all 6 walls of the box."""
    setup_params(gravity=(0.0, 0.0, 0.0), restitution=0.3, wall_friction=0.5)

    n = 6
    # Each particle aimed at a different wall
    pos = np.array([
        [-0.999, 0.0, 0.0, 1.0],  # -> -X wall
        [ 0.999, 0.0, 0.0, 1.0],  # -> +X wall
        [0.0, -0.999, 0.0, 1.0],  # -> -Y wall
        [0.0,  0.999, 0.0, 1.0],  # -> +Y wall
        [0.0, 0.0, -0.999, 1.0],  # -> -Z wall
        [0.0, 0.0,  0.999, 1.0],  # -> +Z wall
    ], dtype=np.float32)

    vel = np.array([
        [-20.0, 0.0, 0.0, 0.0],
        [ 20.0, 0.0, 0.0, 0.0],
        [0.0, -20.0, 0.0, 0.0],
        [0.0,  20.0, 0.0, 0.0],
        [0.0, 0.0, -20.0, 0.0],
        [0.0, 0.0,  20.0, 0.0],
    ], dtype=np.float32)

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    pos_out, vel_out, _, _, _, _ = integrate(**d)

    pos_h = pos_out.get()
    vel_h = vel_out.get()

    for i in range(n):
        assert pos_h[i, 0] >= -1.0 - 1e-5, f"Particle {i} escaped -X: {pos_h[i, 0]}"
        assert pos_h[i, 0] <=  1.0 + 1e-5, f"Particle {i} escaped +X: {pos_h[i, 0]}"
        assert pos_h[i, 1] >= -1.0 - 1e-5, f"Particle {i} escaped -Y: {pos_h[i, 1]}"
        assert pos_h[i, 1] <=  1.0 + 1e-5, f"Particle {i} escaped +Y: {pos_h[i, 1]}"
        assert pos_h[i, 2] >= -1.0 - 1e-5, f"Particle {i} escaped -Z: {pos_h[i, 2]}"
        assert pos_h[i, 2] <=  1.0 + 1e-5, f"Particle {i} escaped +Z: {pos_h[i, 2]}"

    # Verify velocity reflection happened (direction reversed)
    assert vel_h[0, 0] > 0, f"Particle 0 should have bounced +X, got vx={vel_h[0, 0]}"
    assert vel_h[1, 0] < 0, f"Particle 1 should have bounced -X, got vx={vel_h[1, 0]}"
    assert vel_h[2, 1] > 0, f"Particle 2 should have bounced +Y, got vy={vel_h[2, 1]}"
    assert vel_h[3, 1] < 0, f"Particle 3 should have bounced -Y, got vy={vel_h[3, 1]}"
    assert vel_h[4, 2] > 0, f"Particle 4 should have bounced +Z, got vz={vel_h[4, 2]}"
    assert vel_h[5, 2] < 0, f"Particle 5 should have bounced -Z, got vz={vel_h[5, 2]}"

    print("PASS: test_boundary_all_walls")


def test_coulomb_friction():
    """Wall collision applies Coulomb friction to tangential velocity."""
    setup_params(gravity=(0.0, 0.0, 0.0), restitution=0.3, wall_friction=0.5)

    n = 1
    # Particle hitting floor with both normal and tangential velocity
    pos = np.array([[0.5, -0.999, 0.0, 1.0]], dtype=np.float32)
    vel = np.array([[5.0, -10.0, 0.0, 0.0]], dtype=np.float32)

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    _, vel_out, _, _, _, _ = integrate(**d)

    vel_h = vel_out.get()

    # Normal: vy reflected with restitution -> ~0.3 * 10.0 = 3.0 upward
    assert vel_h[0, 1] > 0, f"Expected upward bounce, got vy={vel_h[0, 1]}"

    # Tangential: vx should be reduced by friction
    # friction_impulse = mu_wall * |vn| = 0.5 * 10.0 = 5.0
    # reduction = min(5.0 / |vt|, 1.0) = min(5.0/5.0, 1.0) = 1.0
    # vx = 5.0 * (1 - 1.0) = 0.0
    assert abs(vel_h[0, 0]) < 0.5, (
        f"Expected vx ~0 after friction, got {vel_h[0, 0]}"
    )

    print("PASS: test_coulomb_friction")


def test_xsph_position_update():
    """FLUID particles use XSPH-corrected velocity for position advection."""
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 1
    pos = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    vel = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # veleval has XSPH correction: vel + eps*xsph_sum
    veleval = np.array([[1.5, 0.0, 0.0, 0.0]], dtype=np.float32)  # xsph added 0.5

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    d["sorted_veleval"] = cupy.asarray(veleval)
    pos_out, vel_out, _, _, _, _ = integrate(**d)

    pos_h = pos_out.get()
    vel_h = vel_out.get()

    # vel_new = 1.0 (no force)
    # xsph_correction = veleval - vel = 0.5
    # pos_new = 0 + dt * (vel_new + xsph_correction) = 0 + 0.001 * 1.5 = 0.0015
    expected_px = DT * (1.0 + 0.5)
    assert abs(pos_h[0, 0] - expected_px) < 1e-5, (
        f"Expected pos_x={expected_px}, got {pos_h[0, 0]}"
    )

    # Non-FLUID should NOT use XSPH
    d2 = make_simple_particles(n, SAND, GRANULAR, pos=pos, vel=vel)
    d2["sorted_veleval"] = cupy.asarray(veleval)
    pos_out2, _, _, _, _, _ = integrate(**d2)
    pos_h2 = pos_out2.get()

    expected_px2 = DT * 1.0  # no XSPH
    assert abs(pos_h2[0, 0] - expected_px2) < 1e-5, (
        f"GRANULAR should not use XSPH: expected {expected_px2}, got {pos_h2[0, 0]}"
    )

    print("PASS: test_xsph_position_update")


def test_water_pool_no_nan():
    """10K water particles in box settle without NaN after 1000 steps.

    This is a simplified test without the full neighbor pipeline --
    we just check that integration alone doesn't produce NaN with
    typical SPH-like forces.
    """
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 10000
    rng = np.random.RandomState(42)

    # Particles in a cube near the top, falling
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0] = rng.uniform(-0.5, 0.5, n)
    pos[:, 1] = rng.uniform(0.0, 0.8, n)
    pos[:, 2] = rng.uniform(-0.5, 0.5, n)
    pos[:, 3] = 1.0

    vel = np.zeros((n, 4), dtype=np.float32)

    pos_gpu = cupy.asarray(pos)
    vel_gpu = cupy.asarray(vel)
    veleval_gpu = vel_gpu.copy()
    force_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    mass_gpu = cupy.full(n, MASS, dtype=cupy.float32)
    pi_gpu = cupy.full(n, MAKE_PACKED(WATER, FLUID), dtype=cupy.uint32)
    temp_gpu = cupy.full(n, 293.0, dtype=cupy.float32)
    health_gpu = cupy.full(n, 1.0, dtype=cupy.float32)
    sort_idx = cupy.arange(n, dtype=cupy.uint32)
    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    sc_gpu = cupy.zeros(n, dtype=cupy.uint8)

    for step in range(1000):
        # Apply small random SPH-like forces for realism
        if step % 10 == 0:
            force_gpu[:, :3] = cupy.random.uniform(-0.1, 0.1, (n, 3), dtype=cupy.float32)

        integrate(
            sorted_position=pos_gpu,
            sorted_velocity=vel_gpu,
            sorted_veleval=veleval_gpu,
            sorted_sph_force=force_gpu,
            sorted_mass=mass_gpu,
            sorted_packed_info=pi_gpu,
            sorted_temperature=temp_gpu,
            sorted_health=health_gpu,
            sorted_sleep_counter=sc_gpu,
            sort_indexes=sort_idx,
            position_out=pos_gpu,
            velocity_out=vel_gpu,
            color_out=color_gpu,
            packed_info_out=pi_gpu,
            sleep_counter_out=sc_gpu,
        )
        veleval_gpu[:] = vel_gpu

    pos_h = pos_gpu.get()
    vel_h = vel_gpu.get()

    assert not np.any(np.isnan(pos_h)), "NaN found in positions after 1000 steps"
    assert not np.any(np.isnan(vel_h)), "NaN found in velocities after 1000 steps"

    # All particles should be within boundaries
    assert np.all(pos_h[:, 0] >= -1.0 - 1e-4), "Particles escaped -X"
    assert np.all(pos_h[:, 0] <=  1.0 + 1e-4), "Particles escaped +X"
    assert np.all(pos_h[:, 1] >= -1.0 - 1e-4), "Particles escaped -Y"
    assert np.all(pos_h[:, 1] <=  1.0 + 1e-4), "Particles escaped +Y"
    assert np.all(pos_h[:, 2] >= -1.0 - 1e-4), "Particles escaped -Z"
    assert np.all(pos_h[:, 2] <=  1.0 + 1e-4), "Particles escaped +Z"

    # Most particles should have settled near the bottom
    mean_y = np.mean(pos_h[:, 1])
    assert mean_y < 0.0, f"Mean Y should be negative (settled), got {mean_y}"

    print("PASS: test_water_pool_no_nan")


def test_granular_anti_creep_settled():
    """GRANULAR particles at rest with high density and low shear rate get zeroed velocity.

    Acceptance: sand pile that has settled shows zero velocity for interior particles.
    """
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 100
    # Particles sitting at the bottom with tiny residual velocity
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0] = np.linspace(-0.5, 0.5, n)
    pos[:, 1] = -0.9  # near bottom
    pos[:, 3] = 1.0

    # Small residual velocity (below threshold 0.01)
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = 0.005  # magnitude < 0.01 threshold
    vel[:, 1] = 0.003

    d = make_simple_particles(n, SAND, GRANULAR, pos=pos, vel=vel)

    # Density well above 0.95 * rho0 (sand rho0 = 1600)
    d["sorted_density"] = cupy.full(n, 1700.0, dtype=cupy.float32)
    # Shear rate below gamma_min (0.05)
    d["sorted_shear_rate"] = cupy.full(n, 0.01, dtype=cupy.float32)

    _, vel_out, _, _, _, _ = integrate(**d)
    vel_h = vel_out.get()

    # All velocities should be exactly zero (anti-creep triggered)
    for i in range(n):
        speed = math.sqrt(vel_h[i, 0]**2 + vel_h[i, 1]**2 + vel_h[i, 2]**2)
        assert speed < 1e-6, (
            f"Particle {i} should be zeroed by anti-creep, got speed={speed}"
        )

    print("PASS: test_granular_anti_creep_settled")


def test_granular_anti_creep_flowing():
    """GRANULAR particles with velocity above threshold still flow correctly.

    Acceptance: pouring new sand on top of settled pile still flows correctly.
    """
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 10
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 1] = 0.5  # above ground
    pos[:, 3] = 1.0

    # Velocity well above threshold (0.01)
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 1] = -2.0  # falling fast

    d = make_simple_particles(n, SAND, GRANULAR, pos=pos, vel=vel)
    d["sorted_density"] = cupy.full(n, 1700.0, dtype=cupy.float32)
    d["sorted_shear_rate"] = cupy.full(n, 0.01, dtype=cupy.float32)

    _, vel_out, _, _, _, _ = integrate(**d)
    vel_h = vel_out.get()

    # Velocity should NOT be zeroed (above threshold)
    for i in range(n):
        speed = math.sqrt(vel_h[i, 0]**2 + vel_h[i, 1]**2 + vel_h[i, 2]**2)
        assert speed > 0.5, (
            f"Particle {i} should still be flowing, got speed={speed}"
        )

    print("PASS: test_granular_anti_creep_flowing")


def test_granular_anti_creep_low_density():
    """GRANULAR particles with density below 0.95*rho0 are NOT clamped.

    If density is low (particle expanding / not under compression),
    the anti-creep should not trigger even with slow velocity.
    """
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 10
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 1] = 0.0
    pos[:, 3] = 1.0

    # Small velocity (below threshold)
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = 0.005

    d = make_simple_particles(n, SAND, GRANULAR, pos=pos, vel=vel)
    # Density BELOW 0.95 * 1600 = 1520
    d["sorted_density"] = cupy.full(n, 1000.0, dtype=cupy.float32)
    d["sorted_shear_rate"] = cupy.full(n, 0.01, dtype=cupy.float32)

    _, vel_out, _, _, _, _ = integrate(**d)
    vel_h = vel_out.get()

    # Velocity should NOT be zeroed (density too low for anti-creep)
    for i in range(n):
        # After integration with gravity: vel_y = 0.005_y_component + dt * (-9.8) != 0
        speed = math.sqrt(vel_h[i, 0]**2 + vel_h[i, 1]**2 + vel_h[i, 2]**2)
        assert speed > 1e-6, (
            f"Particle {i} should not be zeroed (low density), got speed={speed}"
        )

    print("PASS: test_granular_anti_creep_low_density")


def test_granular_anti_creep_high_shear():
    """GRANULAR particles with high shear rate are NOT clamped.

    If shear rate is above gamma_min, anti-creep should not trigger.
    """
    setup_params(gravity=(0.0, 0.0, 0.0))  # no gravity for clean test

    n = 10
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 3] = 1.0

    # Small velocity (below threshold)
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = 0.005

    d = make_simple_particles(n, SAND, GRANULAR, pos=pos, vel=vel)
    d["sorted_density"] = cupy.full(n, 1700.0, dtype=cupy.float32)
    # Shear rate ABOVE gamma_min (0.05)
    d["sorted_shear_rate"] = cupy.full(n, 1.0, dtype=cupy.float32)

    _, vel_out, _, _, _, _ = integrate(**d)
    vel_h = vel_out.get()

    # Velocity should NOT be zeroed (shear rate too high)
    for i in range(n):
        assert abs(vel_h[i, 0] - 0.005) < 1e-4, (
            f"Particle {i} vel.x changed unexpectedly: {vel_h[i, 0]}"
        )

    print("PASS: test_granular_anti_creep_high_shear")


def test_granular_no_jitter_5000_steps():
    """Settled sand pile shows no jitter over 5000 steps.

    Acceptance: no visible vibration or jitter in settled sand pile over 5000 steps.
    """
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 100
    pos_np = np.zeros((n, 4), dtype=np.float32)
    pos_np[:, 0] = np.linspace(-0.5, 0.5, n)
    pos_np[:, 1] = -0.95  # near floor
    pos_np[:, 3] = 1.0

    vel_np = np.zeros((n, 4), dtype=np.float32)
    vel_np[:, 1] = 0.002  # tiny residual

    pos_gpu = cupy.asarray(pos_np)
    vel_gpu = cupy.asarray(vel_np)
    # High density (settled/compressed)
    density_gpu = cupy.full(n, 1700.0, dtype=cupy.float32)
    # Low shear rate (at rest)
    shear_rate_gpu = cupy.full(n, 0.01, dtype=cupy.float32)

    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    pi_gpu = cupy.full(n, MAKE_PACKED(SAND, GRANULAR), dtype=cupy.uint32)
    sc_gpu = cupy.zeros(n, dtype=cupy.uint8)

    # Record initial position after first step
    for step in range(5000):
        d = {
            "sorted_position": pos_gpu,
            "sorted_velocity": vel_gpu,
            "sorted_veleval": vel_gpu.copy(),
            "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
            "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
            "sorted_packed_info": pi_gpu,
            "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
            "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
            "sorted_density": density_gpu,
            "sorted_shear_rate": shear_rate_gpu,
            "sorted_sleep_counter": sc_gpu,
            "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
            "position_out": pos_gpu,
            "velocity_out": vel_gpu,
            "color_out": color_gpu,
            "packed_info_out": pi_gpu,
            "sleep_counter_out": sc_gpu,
        }
        integrate(**d)

    vel_h = vel_gpu.get()
    pos_h = pos_gpu.get()

    # All velocities should be zero after settling
    max_speed = 0.0
    for i in range(n):
        speed = math.sqrt(vel_h[i, 0]**2 + vel_h[i, 1]**2 + vel_h[i, 2]**2)
        max_speed = max(max_speed, speed)

    assert max_speed < 1e-6, (
        f"Jitter detected after 5000 steps: max_speed={max_speed}"
    )

    # No NaN
    assert not np.any(np.isnan(pos_h)), "NaN in positions after 5000 steps"
    assert not np.any(np.isnan(vel_h)), "NaN in velocities after 5000 steps"

    # All particles should be within boundaries
    assert np.all(pos_h[:, 1] >= -1.0 - 1e-4), "Particles escaped floor"

    print("PASS: test_granular_no_jitter_5000_steps")


def test_fluid_unaffected_by_anti_creep():
    """FLUID particles with low velocity are NOT affected by anti-creep."""
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 5
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 3] = 1.0

    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = 0.005  # below granular threshold

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    # Even with high density and low shear rate, FLUID should not be affected
    d["sorted_density"] = cupy.full(n, 1200.0, dtype=cupy.float32)
    d["sorted_shear_rate"] = cupy.full(n, 0.01, dtype=cupy.float32)

    _, vel_out, _, _, _, _ = integrate(**d)
    vel_h = vel_out.get()

    # FLUID velocity should be preserved (not zeroed)
    for i in range(n):
        assert abs(vel_h[i, 0] - 0.005) < 1e-4, (
            f"FLUID particle {i} vel.x incorrectly zeroed: {vel_h[i, 0]}"
        )

    print("PASS: test_fluid_unaffected_by_anti_creep")


# ===========================================================================
# Sleep system tests (US-018)
# ===========================================================================

IS_SLEEPING = lambda p: (p >> 10) & 1


def test_sleep_counter_increments():
    """Sleep counter increments each frame when velocity < v_sleep AND shear_rate < gamma_sleep.

    After 10 frames of being still, SLEEPING flag should be set.
    """
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 10
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 3] = 1.0

    # Very low velocity (below v_sleep=0.005)
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = 0.001

    pos_gpu = cupy.asarray(pos)
    vel_gpu = cupy.asarray(vel)
    pi_gpu = cupy.full(n, MAKE_PACKED(SAND, GRANULAR), dtype=cupy.uint32)
    sc_gpu = cupy.zeros(n, dtype=cupy.uint8)
    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)

    # Run 15 steps -- counter should reach threshold (10) and set SLEEPING
    for step in range(15):
        d = {
            "sorted_position": pos_gpu,
            "sorted_velocity": vel_gpu,
            "sorted_veleval": vel_gpu.copy(),
            "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
            "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
            "sorted_packed_info": pi_gpu,
            "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
            "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
            "sorted_density": cupy.full(n, 1700.0, dtype=cupy.float32),
            "sorted_shear_rate": cupy.full(n, 0.005, dtype=cupy.float32),
            "sorted_sleep_counter": sc_gpu,
            "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
            "position_out": pos_gpu,
            "velocity_out": vel_gpu,
            "color_out": color_gpu,
            "packed_info_out": pi_gpu,
            "sleep_counter_out": sc_gpu,
        }
        integrate(**d)

    sc_h = sc_gpu.get()
    pi_h = pi_gpu.get()

    # After 15 steps of being still, counter should be >= 10
    for i in range(n):
        assert sc_h[i] >= 10, f"Sleep counter too low: {sc_h[i]} for particle {i}"
        assert IS_SLEEPING(int(pi_h[i])) == 1, (
            f"Particle {i} should be sleeping: packed_info=0x{pi_h[i]:08x}"
        )

    print("PASS: test_sleep_counter_increments")


def test_sleep_counter_resets():
    """Sleep counter resets to 0 when velocity > v_sleep or shear_rate > gamma_sleep."""
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 5
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 3] = 1.0

    # Velocity ABOVE v_sleep (0.005) -- should not accumulate sleep counter
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = 0.01  # above v_sleep

    pos_gpu = cupy.asarray(pos)
    vel_gpu = cupy.asarray(vel)
    pi_gpu = cupy.full(n, MAKE_PACKED(SAND, GRANULAR), dtype=cupy.uint32)
    sc_gpu = cupy.full(n, np.uint8(5), dtype=cupy.uint8)  # start with counter=5
    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)

    d = {
        "sorted_position": pos_gpu,
        "sorted_velocity": vel_gpu,
        "sorted_veleval": vel_gpu.copy(),
        "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
        "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
        "sorted_packed_info": pi_gpu,
        "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
        "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
        "sorted_density": cupy.full(n, 1700.0, dtype=cupy.float32),
        "sorted_shear_rate": cupy.full(n, 0.005, dtype=cupy.float32),
        "sorted_sleep_counter": sc_gpu,
        "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
        "position_out": pos_gpu,
        "velocity_out": vel_gpu,
        "color_out": color_gpu,
        "packed_info_out": pi_gpu,
        "sleep_counter_out": sc_gpu,
    }
    integrate(**d)

    sc_h = sc_gpu.get()
    for i in range(n):
        assert sc_h[i] == 0, (
            f"Sleep counter should have reset to 0 (velocity above v_sleep), got {sc_h[i]}"
        )

    print("PASS: test_sleep_counter_resets")


def test_sleep_hysteresis_wake():
    """Sleeping particles wake only when velocity > v_wake (0.02), not at v_sleep (0.005).

    Hysteresis: v_wake > v_sleep prevents oscillation between sleeping and awake states.
    """
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 2

    # Particle 0: sleeping, velocity between v_sleep and v_wake -> stays asleep
    # Particle 1: sleeping, velocity above v_wake -> wakes up
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 3] = 1.0

    vel = np.zeros((n, 4), dtype=np.float32)
    vel[0, 0] = 0.01   # between v_sleep (0.005) and v_wake (0.02)
    vel[1, 0] = 0.03   # above v_wake (0.02)

    packed = np.array([
        SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)),
        SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)),
    ], dtype=np.uint32)

    sc = np.array([15, 15], dtype=np.uint8)

    d = {
        "sorted_position": cupy.asarray(pos),
        "sorted_velocity": cupy.asarray(vel),
        "sorted_veleval": cupy.asarray(vel),
        "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
        "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
        "sorted_packed_info": cupy.asarray(packed),
        "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
        "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
        "sorted_density": cupy.full(n, 1700.0, dtype=cupy.float32),
        "sorted_shear_rate": cupy.full(n, 0.005, dtype=cupy.float32),
        "sorted_sleep_counter": cupy.asarray(sc),
        "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
    }

    _, vel_out, _, pi_out, sc_out, _ = integrate(**d)
    pi_h = pi_out.get()
    sc_h = sc_out.get()
    vel_h = vel_out.get()

    # Particle 0: still sleeping (velocity 0.01 <= v_wake 0.02)
    assert IS_SLEEPING(int(pi_h[0])) == 1, (
        f"Particle 0 should stay sleeping: pi=0x{pi_h[0]:08x}"
    )
    # Sleeping particle has zero velocity output
    assert abs(vel_h[0, 0]) < 1e-6, f"Sleeping particle vel should be 0, got {vel_h[0, 0]}"

    # Particle 1: woke up (velocity 0.03 > v_wake 0.02)
    assert IS_SLEEPING(int(pi_h[1])) == 0, (
        f"Particle 1 should be awake: pi=0x{pi_h[1]:08x}"
    )
    assert sc_h[1] == 0, f"Woken particle counter should be 0, got {sc_h[1]}"

    print("PASS: test_sleep_hysteresis_wake")


def test_sleeping_skip_force_integration():
    """Sleeping particles skip force integration -- position unchanged.

    Even with SPH forces and gravity, a sleeping particle's position should not move.
    """
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 2
    # Particle 0: sleeping (should not move)
    # Particle 1: awake (should move normally)
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[0] = [0.0, 0.0, 0.0, 1.0]
    pos[1] = [0.5, 0.5, 0.0, 1.0]

    vel = np.zeros((n, 4), dtype=np.float32)
    vel[0, 0] = 0.001   # very slow (below v_wake)
    vel[1, 1] = -1.0     # falling

    packed = np.array([
        SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)),
        MAKE_PACKED(WATER, FLUID),
    ], dtype=np.uint32)

    sph_force = np.zeros((n, 4), dtype=np.float32)
    sph_force[0] = [100.0, 100.0, 100.0, 0.0]  # large force on sleeping particle
    sph_force[1] = [0.0, 0.0, 0.0, 0.0]

    d = {
        "sorted_position": cupy.asarray(pos),
        "sorted_velocity": cupy.asarray(vel),
        "sorted_veleval": cupy.asarray(vel),
        "sorted_sph_force": cupy.asarray(sph_force),
        "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
        "sorted_packed_info": cupy.asarray(packed),
        "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
        "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
        "sorted_density": cupy.full(n, 1700.0, dtype=cupy.float32),
        "sorted_shear_rate": cupy.full(n, 0.001, dtype=cupy.float32),
        "sorted_sleep_counter": cupy.array([15, 0], dtype=cupy.uint8),
        "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
    }

    pos_out, vel_out, _, pi_out, _, _ = integrate(**d)
    pos_h = pos_out.get()
    vel_h = vel_out.get()

    # Sleeping particle: position should be unchanged
    assert abs(pos_h[0, 0] - 0.0) < 1e-6, f"Sleeping particle moved X: {pos_h[0, 0]}"
    assert abs(pos_h[0, 1] - 0.0) < 1e-6, f"Sleeping particle moved Y: {pos_h[0, 1]}"
    # Sleeping particle: velocity should be zero
    assert abs(vel_h[0, 0]) < 1e-6, f"Sleeping particle has velocity: {vel_h[0, 0]}"
    assert abs(vel_h[0, 1]) < 1e-6, f"Sleeping particle has velocity: {vel_h[0, 1]}"

    # Awake particle: should have moved due to gravity
    assert pos_h[1, 1] < 0.5, f"Awake particle should have fallen, pos_y={pos_h[1, 1]}"

    print("PASS: test_sleeping_skip_force_integration")


def test_sleep_counter_saturates():
    """Sleep counter saturates at 255 (uint8 max)."""
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 1
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 3] = 1.0
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = 0.001  # below v_sleep

    pos_gpu = cupy.asarray(pos)
    vel_gpu = cupy.asarray(vel)
    pi_gpu = cupy.full(n, MAKE_PACKED(SAND, GRANULAR), dtype=cupy.uint32)
    sc_gpu = cupy.full(n, np.uint8(254), dtype=cupy.uint8)
    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)

    # Run 5 steps starting from counter=254
    for _ in range(5):
        d = {
            "sorted_position": pos_gpu,
            "sorted_velocity": vel_gpu,
            "sorted_veleval": vel_gpu.copy(),
            "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
            "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
            "sorted_packed_info": pi_gpu,
            "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
            "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
            "sorted_density": cupy.full(n, 1700.0, dtype=cupy.float32),
            "sorted_shear_rate": cupy.full(n, 0.005, dtype=cupy.float32),
            "sorted_sleep_counter": sc_gpu,
            "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
            "position_out": pos_gpu,
            "velocity_out": vel_gpu,
            "color_out": color_gpu,
            "packed_info_out": pi_gpu,
            "sleep_counter_out": sc_gpu,
        }
        integrate(**d)

    sc_h = sc_gpu.get()
    assert sc_h[0] == 255, f"Sleep counter should saturate at 255, got {sc_h[0]}"

    print("PASS: test_sleep_counter_saturates")


def test_sleeping_particles_density_contribution():
    """Sleeping particles still participate in hash/sort/density (verified by packed_info being preserved).

    This test verifies that the sleeping flag doesn't prevent the particle from being
    reordered and having its data read by other kernels. The sleep logic only skips
    Step2 and force integration, not Step1 density.
    """
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 4
    # Mix of sleeping and awake particles
    packed = np.array([
        SET_SLEEPING(MAKE_PACKED(SAND, GRANULAR)),  # sleeping
        MAKE_PACKED(SAND, GRANULAR),                 # awake
        SET_SLEEPING(MAKE_PACKED(WATER, FLUID)),    # sleeping fluid
        MAKE_PACKED(WATER, FLUID),                   # awake fluid
    ], dtype=np.uint32)

    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0] = [0.0, 0.1, 0.2, 0.3]
    pos[:, 3] = 1.0
    vel = np.zeros((n, 4), dtype=np.float32)  # all zero

    d = {
        "sorted_position": cupy.asarray(pos),
        "sorted_velocity": cupy.asarray(vel),
        "sorted_veleval": cupy.asarray(vel),
        "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
        "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
        "sorted_packed_info": cupy.asarray(packed),
        "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
        "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
        "sorted_density": cupy.full(n, 1700.0, dtype=cupy.float32),
        "sorted_shear_rate": cupy.full(n, 0.001, dtype=cupy.float32),
        "sorted_sleep_counter": cupy.array([15, 0, 15, 0], dtype=cupy.uint8),
        "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
    }

    pos_out, _, _, pi_out, sc_out, _ = integrate(**d)
    pi_h = pi_out.get()
    pos_h = pos_out.get()

    # Sleeping particles should have SLEEPING flag preserved (and position unchanged)
    assert IS_SLEEPING(int(pi_h[0])) == 1, "Particle 0 should still be sleeping"
    assert IS_SLEEPING(int(pi_h[2])) == 1, "Particle 2 should still be sleeping"
    assert abs(pos_h[0, 0] - 0.0) < 1e-6, "Sleeping particle 0 should not move"
    assert abs(pos_h[2, 0] - 0.2) < 1e-6, "Sleeping particle 2 should not move"

    # Awake particles should have SLEEPING flag clear
    assert IS_SLEEPING(int(pi_h[1])) == 0, "Particle 1 should be awake"
    assert IS_SLEEPING(int(pi_h[3])) == 0, "Particle 3 should be awake"

    # Material IDs should be preserved
    assert (pi_h[0] & 0xFF) == SAND, "Sleeping particle 0 lost material ID"
    assert (pi_h[1] & 0xFF) == SAND, "Awake particle 1 lost material ID"
    assert (pi_h[2] & 0xFF) == WATER, "Sleeping particle 2 lost material ID"
    assert (pi_h[3] & 0xFF) == WATER, "Awake particle 3 lost material ID"

    print("PASS: test_sleeping_particles_density_contribution")


def test_sleep_wake_cycle():
    """Full sleep/wake cycle: particles sleep after N frames, wake when disturbed.

    Simulates settling (sleep), then external disturbance (wake).
    """
    setup_params(gravity=(0.0, 0.0, 0.0))

    n = 5
    pos_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    pos_gpu[:, 3] = 1.0

    # Start with very slow velocity
    vel_gpu = cupy.zeros((n, 4), dtype=cupy.float32)
    vel_gpu[:, 0] = 0.001  # below v_sleep

    pi_gpu = cupy.full(n, MAKE_PACKED(SAND, GRANULAR), dtype=cupy.uint32)
    sc_gpu = cupy.zeros(n, dtype=cupy.uint8)
    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)

    # Phase 1: let particles settle and fall asleep (15 steps)
    for _ in range(15):
        d = {
            "sorted_position": pos_gpu,
            "sorted_velocity": vel_gpu,
            "sorted_veleval": vel_gpu.copy(),
            "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
            "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
            "sorted_packed_info": pi_gpu,
            "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
            "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
            "sorted_density": cupy.full(n, 1700.0, dtype=cupy.float32),
            "sorted_shear_rate": cupy.full(n, 0.005, dtype=cupy.float32),
            "sorted_sleep_counter": sc_gpu,
            "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
            "position_out": pos_gpu,
            "velocity_out": vel_gpu,
            "color_out": color_gpu,
            "packed_info_out": pi_gpu,
            "sleep_counter_out": sc_gpu,
        }
        integrate(**d)

    # Verify all sleeping
    pi_h = pi_gpu.get()
    for i in range(n):
        assert IS_SLEEPING(int(pi_h[i])) == 1, f"Particle {i} should be sleeping after settling"

    # Phase 2: give particles high velocity to wake them (simulate impact)
    vel_gpu[:, 0] = 0.05  # above v_wake (0.02)

    for _ in range(1):
        d = {
            "sorted_position": pos_gpu,
            "sorted_velocity": vel_gpu,
            "sorted_veleval": vel_gpu.copy(),
            "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
            "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
            "sorted_packed_info": pi_gpu,
            "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
            "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
            "sorted_density": cupy.full(n, 1700.0, dtype=cupy.float32),
            "sorted_shear_rate": cupy.full(n, 0.005, dtype=cupy.float32),
            "sorted_sleep_counter": sc_gpu,
            "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
            "position_out": pos_gpu,
            "velocity_out": vel_gpu,
            "color_out": color_gpu,
            "packed_info_out": pi_gpu,
            "sleep_counter_out": sc_gpu,
        }
        integrate(**d)

    # Verify all woken up
    pi_h = pi_gpu.get()
    sc_h = sc_gpu.get()
    for i in range(n):
        assert IS_SLEEPING(int(pi_h[i])) == 0, (
            f"Particle {i} should be awake after disturbance"
        )
        assert sc_h[i] == 0, f"Particle {i} counter should be 0 after waking, got {sc_h[i]}"

    print("PASS: test_sleep_wake_cycle")


# ===========================================================================
# Acceleration clamping tests (US-028)
# ===========================================================================


def test_accel_clamp_overlapping_particles():
    """Overlapping particles with huge SPH forces separate without flying to infinity.

    Acceptance: accel_max = 5000 clamps extreme accelerations from numerical blowups.
    Particles should remain within boundaries after multiple steps.
    """
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 10
    # All particles at the SAME position (worst-case overlap)
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0] = 0.0
    pos[:, 1] = 0.0
    pos[:, 2] = 0.0
    pos[:, 3] = 1.0

    vel = np.zeros((n, 4), dtype=np.float32)

    # Huge SPH forces simulating numerical blowup from overlapping particles
    # Without clamping, accel = 1e8 / 0.008 = 1.25e10 m/s^2 -> instant explosion
    sph_force = np.zeros((n, 4), dtype=np.float32)
    sph_force[:, 0] = np.linspace(-1e8, 1e8, n)
    sph_force[:, 1] = np.linspace(1e8, -1e8, n)

    pos_gpu = cupy.asarray(pos)
    vel_gpu = cupy.asarray(vel)
    force_gpu = cupy.asarray(sph_force)
    pi_gpu = cupy.full(n, MAKE_PACKED(WATER, FLUID), dtype=cupy.uint32)
    sc_gpu = cupy.zeros(n, dtype=cupy.uint8)
    color_gpu = cupy.zeros((n, 4), dtype=cupy.float32)

    # Run 100 steps with extreme forces
    for step in range(100):
        d = {
            "sorted_position": pos_gpu,
            "sorted_velocity": vel_gpu,
            "sorted_veleval": vel_gpu.copy(),
            "sorted_sph_force": force_gpu,
            "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
            "sorted_packed_info": pi_gpu,
            "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
            "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
            "sorted_sleep_counter": sc_gpu,
            "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
            "position_out": pos_gpu,
            "velocity_out": vel_gpu,
            "color_out": color_gpu,
            "packed_info_out": pi_gpu,
            "sleep_counter_out": sc_gpu,
        }
        integrate(**d)

    pos_h = pos_gpu.get()
    vel_h = vel_gpu.get()

    # No NaN
    assert not np.any(np.isnan(pos_h)), "NaN in positions after overlapping particle test"
    assert not np.any(np.isnan(vel_h)), "NaN in velocities after overlapping particle test"

    # All particles should be within boundaries (not flying to infinity)
    assert np.all(pos_h[:, 0] >= -1.0 - 1e-4), "Particles escaped -X in blowup test"
    assert np.all(pos_h[:, 0] <=  1.0 + 1e-4), "Particles escaped +X in blowup test"
    assert np.all(pos_h[:, 1] >= -1.0 - 1e-4), "Particles escaped -Y in blowup test"
    assert np.all(pos_h[:, 1] <=  1.0 + 1e-4), "Particles escaped +Y in blowup test"
    assert np.all(pos_h[:, 2] >= -1.0 - 1e-4), "Particles escaped -Z in blowup test"
    assert np.all(pos_h[:, 2] <=  1.0 + 1e-4), "Particles escaped +Z in blowup test"

    # Velocity should be clamped (not infinite)
    speeds = np.sqrt(np.sum(vel_h[:, :3]**2, axis=1))
    assert np.all(speeds <= 50.0 + 0.1), (
        f"Velocity exceeds limit after blowup: max_speed={np.max(speeds)}"
    )

    print("PASS: test_accel_clamp_overlapping_particles")


def test_accel_clamp_normal_unaffected():
    """Normal simulation forces do NOT trigger acceleration clamping.

    With gravity=-9.8 and typical SPH forces, accel << 5000 so clamping
    should not alter the results.
    """
    setup_params(gravity=(0.0, -9.8, 0.0))

    n = 1
    pos = np.array([[0.0, 0.5, 0.0, 1.0]], dtype=np.float32)
    vel = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # Typical SPH repulsive force (not extreme)
    sph_force = np.array([[10.0, 50.0, -5.0, 0.0]], dtype=np.float32)

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    d["sorted_sph_force"] = cupy.asarray(sph_force)
    _, vel_out, _, _, _, _ = integrate(**d)

    vel_h = vel_out.get()

    # accel = force/mass + gravity
    # = (10/0.008, 50/0.008, -5/0.008) + (0, -9.8, 0)
    # = (1250, 6250-9.8, -625) = (1250, 6240.2, -625)
    # |accel| = sqrt(1250^2 + 6240.2^2 + 625^2) = sqrt(1562500 + 38940096 + 390625) = ~6395
    # This exceeds 5000 so clamping WILL activate for this force.
    # Let's use smaller forces that won't trigger.

    # Use a force where accel < 5000
    sph_force2 = np.array([[1.0, 5.0, -0.5, 0.0]], dtype=np.float32)
    d2 = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    d2["sorted_sph_force"] = cupy.asarray(sph_force2)
    _, vel_out2, _, _, _, _ = integrate(**d2)

    vel_h2 = vel_out2.get()

    # accel = (1/0.008, 5/0.008 - 9.8, -0.5/0.008) = (125, 615.2, -62.5)
    # |accel| = sqrt(125^2 + 615.2^2 + 62.5^2) = sqrt(15625 + 378471 + 3906) = ~630.9
    # Well below 5000, so clamping should NOT activate.
    inv_mass = 1.0 / MASS
    expected_vx = DT * (1.0 * inv_mass)
    expected_vy = DT * (5.0 * inv_mass - 9.8)
    expected_vz = DT * (-0.5 * inv_mass)

    assert abs(vel_h2[0, 0] - expected_vx) < 1e-4, (
        f"Accel clamp altered normal vx: expected {expected_vx}, got {vel_h2[0, 0]}"
    )
    assert abs(vel_h2[0, 1] - expected_vy) < 1e-4, (
        f"Accel clamp altered normal vy: expected {expected_vy}, got {vel_h2[0, 1]}"
    )
    assert abs(vel_h2[0, 2] - expected_vz) < 1e-4, (
        f"Accel clamp altered normal vz: expected {expected_vz}, got {vel_h2[0, 2]}"
    )

    print("PASS: test_accel_clamp_normal_unaffected")


def test_accel_clamp_exact_threshold():
    """Acceleration exactly at the clamping threshold is correctly handled.

    Verify that acceleration just above 5000 gets clamped to exactly 5000 magnitude.
    """
    setup_params(gravity=(0.0, 0.0, 0.0))  # no gravity for clean test

    n = 1
    pos = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    vel = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # Force that produces accel = 10000 m/s^2 in Y direction
    # accel = force / mass -> force = accel * mass = 10000 * 0.008 = 80
    target_accel = 10000.0
    force_y = target_accel * MASS
    sph_force = np.array([[0.0, force_y, 0.0, 0.0]], dtype=np.float32)

    d = make_simple_particles(n, WATER, FLUID, pos=pos, vel=vel)
    d["sorted_sph_force"] = cupy.asarray(sph_force)
    _, vel_out, _, _, _, _ = integrate(**d)

    vel_h = vel_out.get()

    # Without clamping: vel_y = dt * 10000 = 10.0
    # With clamping: accel clamped to 5000, vel_y = dt * 5000 = 5.0
    expected_vy = DT * 5000.0
    assert abs(vel_h[0, 1] - expected_vy) < 1e-3, (
        f"Expected clamped vel_y={expected_vy}, got {vel_h[0, 1]}"
    )

    # Direction should be preserved (only Y component)
    assert abs(vel_h[0, 0]) < 1e-6, f"Unexpected vx={vel_h[0, 0]}"
    assert abs(vel_h[0, 2]) < 1e-6, f"Unexpected vz={vel_h[0, 2]}"

    print("PASS: test_accel_clamp_exact_threshold")


def test_500k_stress():
    """500K particles run through integrate without errors."""
    setup_params()

    n = 500_000
    rng = np.random.RandomState(123)

    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, :3] = rng.uniform(-0.9, 0.9, (n, 3))
    pos[:, 3] = 1.0
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, :3] = rng.uniform(-1.0, 1.0, (n, 3))

    # Mix of behaviors
    packed = np.zeros(n, dtype=np.uint32)
    packed[:200000] = MAKE_PACKED(WATER, FLUID)
    packed[200000:400000] = MAKE_PACKED(SAND, GRANULAR)
    packed[400000:480000] = MAKE_PACKED(STEAM, GAS)
    packed[480000:] = MAKE_PACKED(STONE, STATIC)

    # Provide density and shear_rate for the mixed test
    density = np.zeros(n, dtype=np.float32)
    density[:200000] = 1000.0  # water rho0
    density[200000:400000] = 1600.0  # sand rho0
    density[400000:480000] = 0.6  # steam rho0
    density[480000:] = 2500.0  # stone rho0

    shear_rate = np.zeros(n, dtype=np.float32)
    shear_rate[:] = 0.0  # all at rest

    d = {
        "sorted_position": cupy.asarray(pos),
        "sorted_velocity": cupy.asarray(vel),
        "sorted_veleval": cupy.asarray(vel),
        "sorted_sph_force": cupy.zeros((n, 4), dtype=cupy.float32),
        "sorted_mass": cupy.full(n, MASS, dtype=cupy.float32),
        "sorted_packed_info": cupy.asarray(packed),
        "sorted_temperature": cupy.full(n, 293.0, dtype=cupy.float32),
        "sorted_health": cupy.full(n, 1.0, dtype=cupy.float32),
        "sorted_density": cupy.asarray(density),
        "sorted_shear_rate": cupy.asarray(shear_rate),
        "sort_indexes": cupy.arange(n, dtype=cupy.uint32),
    }

    pos_out, vel_out, color_out, _, _, _ = integrate(**d)

    # Synchronize to catch any CUDA errors
    cupy.cuda.Device().synchronize()

    pos_h = pos_out.get()
    vel_h = vel_out.get()
    color_h = color_out.get()

    assert not np.any(np.isnan(pos_h)), "NaN in positions (500K)"
    assert not np.any(np.isnan(vel_h)), "NaN in velocities (500K)"
    assert not np.any(np.isnan(color_h)), "NaN in colors (500K)"

    # All particles should be within boundaries
    assert np.all(pos_h[:, 0] >= -1.0 - 1e-4), "500K: particles escaped -X"
    assert np.all(pos_h[:, 0] <=  1.0 + 1e-4), "500K: particles escaped +X"
    assert np.all(pos_h[:, 1] >= -1.0 - 1e-4), "500K: particles escaped -Y"
    assert np.all(pos_h[:, 1] <=  1.0 + 1e-4), "500K: particles escaped +Y"

    # STATIC particles should keep their original positions
    static_pos_orig = pos[480000:, :3]
    static_pos_new = pos_h[480000:, :3]
    assert np.allclose(static_pos_orig, static_pos_new, atol=1e-5), (
        "STATIC particle positions changed!"
    )

    print("PASS: test_500k_stress")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    tests = [
        test_compilation,
        test_block_size,
        test_static_skip,
        test_gravity_freefall,
        test_bounce_floor,
        test_bounce_multi_step,
        test_gas_buoyancy,
        test_gas_drag,
        test_velocity_clamp,
        test_color_computation,
        test_sort_indexes_writeback,
        test_boundary_all_walls,
        test_coulomb_friction,
        test_xsph_position_update,
        test_water_pool_no_nan,
        test_granular_anti_creep_settled,
        test_granular_anti_creep_flowing,
        test_granular_anti_creep_low_density,
        test_granular_anti_creep_high_shear,
        test_granular_no_jitter_5000_steps,
        test_fluid_unaffected_by_anti_creep,
        # Sleep system tests (US-018)
        test_sleep_counter_increments,
        test_sleep_counter_resets,
        test_sleep_hysteresis_wake,
        test_sleeping_skip_force_integration,
        test_sleep_counter_saturates,
        test_sleeping_particles_density_contribution,
        test_sleep_wake_cycle,
        # Acceleration clamping tests (US-028)
        test_accel_clamp_overlapping_particles,
        test_accel_clamp_normal_unaffected,
        test_accel_clamp_exact_threshold,
        test_500k_stress,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    print("All tests passed!")
