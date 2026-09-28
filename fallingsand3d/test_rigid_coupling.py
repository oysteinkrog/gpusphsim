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

(d) Round-4 regressions and WCSPH (bd-r5-review-n28.124): the pressure push on
    a plate points away from the fluid in every solver; the WCSPH reaction is
    -m_i * a_i (doubling the fluid mass doubles it); a kinematic box in a
    WCSPH pool gets rho * V_eff * g with near-zero torque; a free box floats
    or sinks by its mass and keeps its MAT_RIGID boundary slots.
(e) Setup checks from the old _test_rigid*.py smoke scripts.

The rotated-inertia check for K_IntegrateRigidBodies is in test_integrate.py.

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
              sdf_type=None, spacing: float = SPACING):
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(n_max)
    ex = [h + 0.005 for h in half_extents]
    world.spawn_cube(
        fluid_min, fluid_max, WATER, spacing,
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


def _pin_body(rbm, pos) -> None:
    """Put body 0 back at pos after a step, like a conveyor belt.

    A kinematic body moves with its lin_vel (bd-r5-review-n28.105). The drag
    tests want a surface that moves while the body stays put, so they reset
    the position after every step and keep lin_vel as the surface velocity.
    """
    flat = rbm.d_rigid_bodies.view(cp.float32).reshape(-1, 24)
    flat[0, 0:3] = cp.asarray(pos, dtype=cp.float32)


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


# DFSPH needs the symmetric fluid pair term m_j (k_i/rho_i + k_j/rho_j)
# (bd-r5-review-n28.100). The old V_j (k_i + k_j) form did not conserve
# momentum at the free surface of this mid-air block: dp_fluid was -0.233
# against a body impulse of 0.120.
_MOMENTUM_SOLVERS = ["PBF", "DFSPH"]


@pytest.mark.parametrize("profile_name", _MOMENTUM_SOLVERS)
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
        _pin_body(rbm, body)
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
        _pin_body(rbm, (0.0, -0.30, 0.0))
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
    _pin_body(rbm, (0.0, 0.0, 0.0))
    p0 = _fluid_momentum(world)
    impulse = np.zeros(3)
    for _ in range(20):
        _step(sim)
        _pin_body(rbm, (0.0, 0.0, 0.0))
        impulse += _body_force(rbm) * sim.dt
    dp_fluid = _fluid_momentum(world) - p0
    print(f"dp_fluid = {dp_fluid}, body impulse = {impulse}")
    assert dp_fluid[0] > 1e-3, dp_fluid
    assert impulse[0] < 0.0, impulse
    err = np.linalg.norm(dp_fluid + impulse)
    assert err <= 0.05 * np.linalg.norm(dp_fluid), (dp_fluid, impulse)


# ---------------------------------------------------------------------------
# (d) Pressure sign, m_i scaling, WCSPH buoyancy, floating and sinking
#     (bd-r5-review-n28.124)
# ---------------------------------------------------------------------------

# WCSPH water settles at about 2.1 rho0 (PHYSICS.md 3.2). A pool spawned at
# the 0.02 lattice collapses to half its height and sloshes for seconds, so the
# WCSPH scenes spawn at 0.02 / 2.1^(1/3) = 0.0155 with the same particle mass.
# The pool then starts close to its settled density.
WCSPH_SPACING = 0.0155
PLATE = (0.0, -0.30, 0.0)
PLATE_HE = (0.2, 0.02, 0.2)


def _plate_under_block(profile_name: str):
    """Pre-compressed water block resting on a kinematic plate, no gravity.

    The block starts at about 2.1 rho0, so every solver sees p > 0 in the
    first substeps and pushes on the plate.
    """
    return _make_sim(
        profile_name,
        fluid_min=(-0.15, -0.275, -0.15), fluid_max=(0.15, -0.17, 0.15),
        body_pos=PLATE, gravity=0.0, world_half_size=0.4,
        half_extents=PLATE_HE, spacing=WCSPH_SPACING,
    )


@pytest.mark.parametrize("profile_name", ["WCSPH"] + SOLVERS)
def test_pressure_pushes_body_away_from_fluid(profile_name):
    """Round-4 sign fix: the fluid above the plate pushes it down.

    With the Akinci pressure gradient sign flipped, the plate is pulled up
    toward the fluid and F_y > 0.
    """
    world, sim, rbm = _plate_under_block(profile_name)
    _step(sim)  # inject the boundary block
    F = np.zeros(3)
    for _ in range(3):
        _step(sim)
        F += _body_force(rbm)
    n = world._high_water
    mat = world.packed_info[:n].get() & 0xFF
    centroid = world.position[:n, :3].get()[mat == WATER].mean(axis=0)
    toward_fluid = centroid - np.array(PLATE)
    print(f"{profile_name}: F = {F}, fluid centroid - plate = {toward_fluid}")
    assert F[1] < -1.0, F
    assert float(np.dot(F, toward_fluid)) < 0.0, F
    # The block is centred on the plate, so the push is along y.
    assert np.hypot(F[0], F[2]) < 0.05 * abs(F[1]), F


def test_wcsph_body_reaction_scales_with_fluid_mass(monkeypatch):
    """Round-4 m_i fix: the reaction on the body is -m_i * a_i.

    K_Step2 is launched twice on the same sorted state, the second time with
    every fluid mass doubled. Density and pressure come from K_Step1 and stay
    the same, so a_i on the fluid next to the plate is unchanged and the body
    force must double. Without the m_i factor the ratio is 1.
    """
    import step2

    world, sim, rbm = _plate_under_block("WCSPH")
    _step(sim)  # inject the boundary block
    real = step2.compute_step2
    captured = {}

    def twice(pos, vel, mass, packed, *args, **kwargs):
        rbm.rigid_forces.fill(0)
        real(pos, vel, mass, packed, *args, **kwargs)
        captured["F1"] = rbm.rigid_forces[0, :3].copy()
        fluid = (packed & 0xFF) == WATER
        heavy = cp.where(fluid, mass * 2.0, mass).astype(cp.float32)
        rbm.rigid_forces.fill(0)
        real(pos, vel, heavy, packed, *args, **kwargs)
        captured["F2"] = rbm.rigid_forces[0, :3].copy()
        # Leave the substep with the real masses.
        rbm.rigid_forces.fill(0)
        real(pos, vel, mass, packed, *args, **kwargs)

    monkeypatch.setattr(step2, "compute_step2", twice)
    _step(sim)
    monkeypatch.undo()
    F1 = captured["F1"].get().astype(np.float64)
    F2 = captured["F2"].get().astype(np.float64)
    print(f"F(m) = {F1}, F(2m) = {F2}")
    assert F1[1] < -1.0, F1
    assert np.linalg.norm(F2 - 2.0 * F1) <= 0.05 * np.linalg.norm(2.0 * F1), (F1, F2)


def test_wcsph_submerged_box_gets_buoyancy():
    """Kinematic 12 cm box in a settled WCSPH pool: F_y = rho * V_eff * g.

    rho is the measured water density next to the box (about 2.1 rho0 in
    WCSPH, PHYSICS.md 3.2), not rho0, because WCSPH hydrostatic pressure grows
    with the density it actually settles at. Measured: F_y about 63 N against
    rho * V_eff * g about 57 N. The mean torque stays near zero.
    """
    body = (0.0, -0.25, 0.0)
    world, sim, rbm = _make_sim(
        "WCSPH", fluid_min=(-0.4, -0.4, -0.4), fluid_max=(0.4, -0.1, 0.4),
        body_pos=body, gravity=G, world_half_size=0.4, spacing=WCSPH_SPACING,
    )
    n_settle, n_avg = 200, 1000
    for _ in range(n_settle):
        sim._sim_step(world._high_water)
    # One substep of WCSPH is noisy (pressure waves), so average on the GPU.
    f_sum = cp.zeros(3, dtype=cp.float64)
    t_sum = cp.zeros(3, dtype=cp.float64)
    for _ in range(n_avg):
        sim._sim_step(world._high_water)
        f_sum += rbm.rigid_forces[0, :3]
        t_sum += rbm.rigid_torques[0, :3]
    F = (f_sum / n_avg).get()
    tau = (t_sum / n_avg).get()

    n = world._high_water
    mat = world.sorted_packed_info[:n].get() & 0xFF
    pos = world.sorted_position[:n, :3].get()
    rho = world.sorted_density[:n].get()
    side = ((mat == WATER) & (np.abs(pos[:, 1] - body[1]) < HALF)
            & (np.abs(pos[:, 0]) > HALF + 0.02) & (np.abs(pos[:, 0]) < HALF + 0.15))
    rho_local = float(np.median(rho[side]))
    expected = rho_local * (2 * HALF + SPACING) ** 3 * G
    print(f"WCSPH: F = {F}, tau = {tau}, rho_local = {rho_local:.0f}, "
          f"rho * V_eff * g = {expected:.2f} N")
    assert F[1] > 0.0, F
    assert abs(F[1] - expected) <= 0.3 * expected, (F, expected)
    assert np.linalg.norm(tau) < 0.05 * np.linalg.norm(F) * HALF, (tau, F)


# Neutral mass of the 12 cm box: PBF about 7 kg (27-30 N at g = 4), WCSPH
# about 15 kg (the over-compressed water above gives about 60 N).
_FLOAT_SINK = [
    ("PBF", 2.0, +1),
    ("PBF", 12.0, -1),
    ("WCSPH", 2.0, +1),
    ("WCSPH", 30.0, -1),
]


@pytest.mark.parametrize("profile_name,mass,direction", _FLOAT_SINK)
def test_free_box_floats_or_sinks(profile_name, mass, direction):
    """A free submerged box released from rest rises when lighter than the
    water it displaces and sinks when heavier. Its boundary slots keep
    MAT_RIGID with body id 0 the whole time.
    """
    wcsph = profile_name == "WCSPH"
    body = (0.0, -0.25, 0.0) if wcsph else (0.0, -0.15, 0.0)
    world, sim, rbm = _make_sim(
        profile_name, fluid_min=(-0.4, -0.4, -0.4),
        fluid_max=(0.4, -0.1 if wcsph else 0.1, 0.4),
        body_pos=body, gravity=G, world_half_size=0.4,
        spacing=WCSPH_SPACING if wcsph else SPACING,
        is_kinematic=False, mass=mass,
    )
    for _ in range(600):
        sim._sim_step(world._high_water)
    y = float(rbm.download_bodies()[0]["position"][1])
    print(f"{profile_name} mass {mass}: y {body[1]:.3f} -> {y:.3f}")
    assert direction * (y - body[1]) > 0.05, y

    off, nb = rbm.boundary_offset, rbm.num_boundary_particles
    assert nb > 0
    pi = world.packed_info[off:off + nb].get()
    assert np.all((pi & 0xFF) == MAT_RIGID)
    assert np.all((pi >> 13) == 0)


# ---------------------------------------------------------------------------
# (e) Setup checks that used to live in the _test_rigid*.py smoke scripts
# ---------------------------------------------------------------------------

def test_rigid_setup_basics():
    import integrate
    from materials import MATERIALS, NUM_DEFINED, STATIC
    from rigid_bodies import RIGID_BODY_DTYPE, SDF_SPHERE, RigidBodyManager

    assert RIGID_BODY_DTYPE.itemsize == 96
    assert MAT_RIGID == 18
    assert len(MATERIALS) == NUM_DEFINED
    assert MATERIALS[MAT_RIGID].behavior_class == STATIC

    mgr = RigidBodyManager()
    mgr.add_rigid_body((0, 0.5, 0), (0.1, 0.2, 0.3), 2.0)
    mgr.add_rigid_body((1, 0.5, 0), (0.2, 0.2, 0.2), 2.0, sdf_type=SDF_SPHERE)
    mgr.add_rigid_body((2, 0.5, 0), (0.1, 0.1, 0.1), 2.0, is_kinematic=True)
    box, sphere, kin = mgr._bodies_cpu[:3]
    assert box["position"][3] == pytest.approx(0.5)
    np.testing.assert_allclose(
        box["inertia_inv"][:3],
        [3.0 / (2.0 * (0.04 + 0.09)), 3.0 / (2.0 * (0.01 + 0.09)), 3.0 / (2.0 * (0.01 + 0.04))],
        rtol=1e-6)
    np.testing.assert_allclose(sphere["inertia_inv"][:3], 1.0 / (0.4 * 2.0 * 0.04), rtol=1e-6)
    assert kin["position"][3] == 0.0
    assert np.all(kin["inertia_inv"][:3] == 0.0)

    for name in ("integrate_rigid_bodies", "update_boundary_particles", "rigid_body_collisions"):
        assert hasattr(integrate, name), name
    module = integrate._get_module()
    for kernel in ("K_IntegrateRigidBodies", "K_RigidBodyCollisions"):
        module.get_function(kernel)
