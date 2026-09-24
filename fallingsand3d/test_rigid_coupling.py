"""PBF and DFSPH rigid-body reaction matches the push the fluid gets (bd-r5-review-n28.50).

The body force is accumulated in the solver kernels that move the fluid
(K_PBF_ComputeDelta, K_DFSPH_CorrectVelocityDiv, K_DFSPH_ComputePressureAccel),
so the fluid and the body exchange equal and opposite impulses.

(a) Momentum: zero gravity, a water block at rest with a kinematic box pressed
    into one face. Over the substeps, the fluid momentum gained equals minus
    the impulse accumulated on the box.
(b) Buoyancy: a kinematic 12 cm box fully submerged in a water pool gets an
    upward force close to rho0 * V_eff * g, where V_eff includes the half
    spacing gap between the fluid and the boundary particles.

(c) PBF drag (bd-r5-review-n28.97): K_PBF_Finalize adds XSPH toward the body
    surface velocity v_body + omega x r and gives the body the opposite force.
    A kinematic body does not move here (K_IntegrateRigidBodies skips it), so
    a kinematic body with a velocity acts as a moving surface in place, like a
    conveyor. That isolates the drag from the pressure push.

The kinematic body is not integrated, so rigid_forces still holds the last
substep's accumulated force when _sim_step returns.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

from materials import MAT_RIGID, WATER  # noqa: E402

SOLVERS = ["PBF", "DFSPH"]
HALF = 0.06           # box half extent (12 cm box)
RHO0 = 2500.0         # WATER rest density in this sim
G = 4.0               # default |gravity|
SPACING = 0.02        # fluid lattice spacing


def _make_sim(profile_name: str, fluid_min, fluid_max, body_pos, gravity: float,
              n_max: int = 120_000, world_half_size: float = 1.0,
              half_extents=(HALF, HALF, HALF), is_kinematic: bool = True,
              velocity=(0.0, 0.0, 0.0), ang_vel=None, mass: float = 4.32,
              sdf_type=None):
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(n_max)
    ex = [h + 0.005 for h in half_extents]
    world.spawn_cube(
        fluid_min, fluid_max, WATER, SPACING,
        exclude_box=(
            (body_pos[0] - ex[0], body_pos[1] - ex[1], body_pos[2] - ex[2]),
            (body_pos[0] + ex[0], body_pos[1] + ex[1], body_pos[2] + ex[2]),
        ),
    )
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=world_half_size)
    sim.set_solver_profile(PROFILES[profile_name])
    sim.set_gravity(-gravity)
    # No spawn damping: it scales velocities and would break the balance.
    sim._spawn_substep = sim._damping_duration
    rbm = sim.rigid_body_manager
    kw = {} if sdf_type is None else {"sdf_type": sdf_type}
    rbm.add_rigid_body_with_particles(
        position=body_pos,
        half_extents=half_extents,
        mass=mass,
        is_kinematic=is_kinematic,
        velocity=velocity,
        **kw,
    )
    if ang_vel is not None:
        rbm._bodies_cpu[0]["ang_vel"][:3] = ang_vel
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    return world, sim, rbm


def _fluid_momentum(world) -> np.ndarray:
    n = world._high_water
    mat = world.packed_info[:n].get() & 0xFF
    fluid = mat == WATER
    v = world.velocity[:n, :3].get()[fluid].astype(np.float64)
    m = world.mass[:n].get()[fluid].astype(np.float64)
    return (m[:, None] * v).sum(axis=0)


def _body_force(rbm) -> np.ndarray:
    return rbm.rigid_forces.get().reshape(-1, 4)[0, :3].astype(np.float64)


def _step(sim) -> None:
    sim._sim_step(sim.world._high_water)
    cp.cuda.Device().synchronize()


def _quiet_surface_terms(sim, profile_name: str) -> None:
    """Turn off surface tension and vorticity confinement.

    Both act on single particles without a partner, so they change the fluid
    momentum on their own and would hide the fluid-body balance.
    """
    import step2
    if profile_name == "PBF":
        import pbf_solver
        prof = sim._profile
        pbf_solver.upload_granular_params(step2.build_granular_params(
            tan_phi_f=prof.pbf_friction_ratio, cohesion=prof.pbf_friction_cohesion,
            vorticity_epsilon=0.0, surface_tension_gamma=0.0,
        ))
    else:
        import dfsph_solver
        dfsph_solver.upload_granular_params(step2.build_granular_params(
            xsph_epsilon=0.1, vorticity_epsilon=0.0, surface_tension_gamma=0.0,
        ))
    sim._invalidate_graphs()


@pytest.mark.parametrize("profile_name", SOLVERS)
def test_fluid_body_momentum_balance(profile_name):
    # Water block in mid-air (no walls), box pushed 3 cm into its +x face.
    world, sim, rbm = _make_sim(
        profile_name,
        fluid_min=(-0.2, -0.2, -0.2), fluid_max=(0.2, 0.2, 0.2),
        body_pos=(0.2 + HALF - 0.03, 0.0, 0.0),
        gravity=0.0,
    )
    _quiet_surface_terms(sim, profile_name)
    _step(sim)  # inject the boundary block
    assert rbm.boundary_injected

    p0 = _fluid_momentum(world)
    impulse = np.zeros(3)
    for _ in range(20):
        _step(sim)
        impulse += _body_force(rbm) * sim.dt
    dp_fluid = _fluid_momentum(world) - p0

    print(f"{profile_name}: dp_fluid = {dp_fluid}, body impulse = {impulse}")
    # The exchange must be real (sign depends on whether the fluid next to
    # the box is over- or under-dense), mostly along x.
    assert abs(dp_fluid[0]) > 1e-3, dp_fluid
    # Action = -reaction, within 5 percent of the push.
    err = np.linalg.norm(dp_fluid + impulse)
    assert err <= 0.05 * np.linalg.norm(dp_fluid), (dp_fluid, impulse)


@pytest.mark.parametrize("profile_name", SOLVERS)
def test_submerged_box_gets_buoyancy(profile_name):
    # Pool fills a 0.8 m domain wall to wall (so it does not spread), 0.5 m
    # deep; box centre 0.25 m below the surface, 0.19 m of water under it.
    world, sim, rbm = _make_sim(
        profile_name,
        fluid_min=(-0.4, -0.4, -0.4), fluid_max=(0.4, 0.1, 0.4),
        body_pos=(0.0, -0.15, 0.0),
        gravity=G,
        world_half_size=0.4,
    )
    # Settle for 1 s of sim time.
    for _ in range(int(round(1.0 / sim.dt))):
        sim._sim_step(world._high_water)
    # Average the last 50 substeps: a single PBF or DFSPH substep is noisy.
    fy = []
    for _ in range(50):
        _step(sim)
        fy.append(_body_force(rbm)[1])
    fy = float(np.mean(fy))
    # Fluid particle centres stay about one spacing from the boundary particles,
    # so the displaced volume reaches half a spacing past each face:
    # (0.12 + 0.02)^3 = 2.744e-3 m^3, 27.44 N. Measured on this setup: PBF about
    # 29.9 N, DFSPH about 27.3 N. Before the fix both were under 1 N.
    expected = RHO0 * (2 * HALF + SPACING) ** 3 * G
    print(f"{profile_name}: F_y = {fy:.2f} N, rho0*V_eff*g = {expected:.2f} N")
    assert fy > 0.0
    assert abs(fy - expected) <= 0.25 * expected, fy


def test_rigid_particles_are_mat_rigid():
    """Guard for the helpers above: the injected block is MAT_RIGID."""
    world, sim, rbm = _make_sim(
        "PBF", (-0.2, -0.2, -0.2), (0.2, 0.2, 0.2), (0.0, 0.5, 0.0), gravity=0.0,
    )
    _step(sim)
    off, nb = rbm.boundary_offset, rbm.num_boundary_particles
    assert nb > 0
    assert np.all((world.packed_info[off:off + nb].get() & 0xFF) == MAT_RIGID)


# ---------------------------------------------------------------------------
# (c) PBF fluid-rigid drag (bd-r5-review-n28.97)
# ---------------------------------------------------------------------------

V_BODY = 0.5  # m/s, surface velocity of the kinematic body
PLATE_MIN_VX = 0.1  # m/s after 60 substeps; measured 0.126, 0 without the drag


def _fluid_state(world):
    n = world._high_water
    mat = world.packed_info[:n].get() & 0xFF
    fluid = mat == WATER
    return world.position[:n, :3].get()[fluid], world.velocity[:n, :3].get()[fluid]


def test_pbf_moving_surface_drags_side_layer():
    """Fluid within h of the faces parallel to the body velocity moves with it.

    Kinematic box in a pool, surface velocity +0.5 m/s in x. Measured after
    100 substeps: 0.041 m/s with the drag, -0.001 m/s without it.
    """
    body = (0.0, -0.15, 0.0)
    world, sim, rbm = _make_sim(
        "PBF", fluid_min=(-0.4, -0.4, -0.4), fluid_max=(0.4, 0.1, 0.4),
        body_pos=body, gravity=G, world_half_size=0.4, velocity=(V_BODY, 0.0, 0.0),
        n_max=200_000,
    )
    for _ in range(100):
        sim._sim_step(world._high_water)
    pos, vel = _fluid_state(world)
    h = 0.04
    ry = np.abs(pos[:, 1] - body[1])
    rz = np.abs(pos[:, 2] - body[2])
    in_x = np.abs(pos[:, 0] - body[0]) < HALF
    layer = in_x & (((ry > HALF) & (ry < HALF + h) & (rz < HALF))
                    | ((rz > HALF) & (rz < HALF + h) & (ry < HALF)))
    assert layer.sum() > 50
    vx = float(vel[layer, 0].mean())
    print(f"side layer mean vx = {vx:.4f} m/s ({layer.sum()} particles)")
    assert vx > 0.04 * V_BODY, vx


def test_pbf_water_on_moving_plate_is_dragged():
    """Thin water layer resting on a kinematic plate moving at 0.5 m/s in x."""
    world, sim, rbm = _make_sim(
        "PBF", fluid_min=(-0.2, -0.26, -0.2), fluid_max=(0.2, -0.20, 0.2),
        body_pos=(0.0, -0.30, 0.0), gravity=G, world_half_size=0.4,
        half_extents=(0.3, 0.02, 0.3), velocity=(V_BODY, 0.0, 0.0),
    )
    for _ in range(60):
        sim._sim_step(world._high_water)
    _, vel = _fluid_state(world)
    vx = float(vel[:, 0].mean())
    print(f"layer mean vx = {vx:.4f} m/s")
    assert vx > PLATE_MIN_VX, vx


def test_pbf_spinning_sphere_loses_spin_to_fluid():
    """A free sphere spinning at 5 rad/s in still water slows down.

    A sphere is used because a spinning box also loses spin through the
    pressure push on its corners, which would pass without the drag. Measured
    after 200 substeps: 0.64 rad/s with the drag, 4.33 rad/s without it, and
    the body damping alone predicts 4.60 rad/s at this dt.
    """
    from rigid_bodies import SDF_SPHERE
    r = HALF
    world, sim, rbm = _make_sim(
        "PBF", fluid_min=(-0.4, -0.4, -0.4), fluid_max=(0.4, 0.4, 0.4),
        body_pos=(0.0, -0.15, 0.0), gravity=0.0, world_half_size=0.4,
        half_extents=(r, r, r), is_kinematic=False, sdf_type=SDF_SPHERE,
        mass=RHO0 * 4.0 / 3.0 * np.pi * r ** 3, ang_vel=(0.0, 5.0, 0.0),
        n_max=200_000,
    )
    for _ in range(200):
        sim._sim_step(world._high_water)
    w_y = float(rbm.download_bodies()[0]["ang_vel"][1])
    damp_only = 5.0 * (1.0 - 0.05 * sim.dt) ** 200
    print(f"omega_y = {w_y:.3f} rad/s, damping only = {damp_only:.3f}")
    assert 0.0 < w_y < 0.6 * damp_only, w_y


def test_pbf_drag_reaction_balances_fluid_momentum():
    """The drag reaction on the body is minus the momentum the fluid gains.

    Zero gravity, a water block with a kinematic box in its middle whose
    surface moves at 0.5 m/s in x. The fluid gains +x momentum and the
    body impulse is -x.
    """
    world, sim, rbm = _make_sim(
        "PBF", fluid_min=(-0.2, -0.2, -0.2), fluid_max=(0.2, 0.2, 0.2),
        body_pos=(0.0, 0.0, 0.0), gravity=0.0, velocity=(V_BODY, 0.0, 0.0),
    )
    _quiet_surface_terms(sim, "PBF")
    _step(sim)
    p0 = _fluid_momentum(world)
    impulse = np.zeros(3)
    for _ in range(20):
        _step(sim)
        impulse += _body_force(rbm) * sim.dt
    dp_fluid = _fluid_momentum(world) - p0
    print(f"dp_fluid = {dp_fluid}, body impulse = {impulse}")
    assert dp_fluid[0] > 1e-3, dp_fluid
    assert impulse[0] < 0.0, impulse
    err = np.linalg.norm(dp_fluid + impulse)
    assert err <= 0.05 * np.linalg.norm(dp_fluid), (dp_fluid, impulse)
