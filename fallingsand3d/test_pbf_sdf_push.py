"""bd-r5-review-n28.82: PBF keeps predicted positions outside SDF objects.

K_PBF_Predict pushes predicted positions out of SDF colliders, but the delta
passes in K_PBF_ApplyDelta used to clamp only to the world box. Water resting
on an SDF box then had predicted positions up to 4 mm inside the box before
K_PBF_Finalize. ApplyDelta now runs the same SDF push every iteration.

The scene is a 10 x 12 x 10 water column (spacing 0.02) resting on an SDF box,
run for 60 PBF substeps (dt = 1/120). The same column on the world floor is
the reference for the contact-layer density.

Graph capture is replaced with direct launches so the Python spies below see
every kernel call.
"""

import numpy as np
import pytest

import cupy as cp
import pbf_solver
import rigid_bodies
from world import World
from simulation import Simulation
from solver_profiles import PROFILES
from materials import MATERIALS, WATER

BOX_CENTER = (0.0, -0.7, 0.0)
BOX_HALF = (0.3, 0.1, 0.3)
SPACING = 0.02
STEPS = 60
SETTLE = 20  # substeps skipped before measuring density (initial drop)
BOUNDARY_MARGIN = 1e-4


def _sdf_box(p):
    """Signed distance to the axis-aligned test box, same formula as sdf_box()."""
    c = cp.asarray(BOX_CENTER, dtype=cp.float32)
    h = cp.asarray(BOX_HALF, dtype=cp.float32)
    q = cp.abs(p - c) - h
    outside = cp.linalg.norm(cp.maximum(q, 0.0), axis=1)
    inside = cp.minimum(cp.max(q, axis=1), 0.0)
    return outside + inside


def _run(on_box, monkeypatch):
    world = World(max_particles=4096)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=False, max_substeps=20)
    sim.set_solver_profile(PROFILES["PBF"])
    monkeypatch.setattr(
        sim, "_capture_graph",
        lambda n, gather_only=False: (sim._run_graph_body(n), None)[1])
    if on_box:
        sim.sdf_manager.add_sdf_object(
            rigid_bodies.SDF_BOX, position=BOX_CENTER, size=BOX_HALF,
            restitution=0.0, friction=0.3)
        base = BOX_CENTER[1] + BOX_HALF[1]
    else:
        base = -sim.world_half_size
    sim.sdf_manager.force_upload()

    n = world.spawn_cube(
        min_corner=(-0.1, base + 0.01, -0.1),
        max_corner=(0.1 - 1e-4, base + 0.25, 0.1 - 1e-4),
        material_id=WATER, spacing=SPACING)
    assert n == 1200
    sim.notify_scene_changed()
    sim._spawn_substep = sim._damping_duration

    rho0 = float(MATERIALS[WATER].rest_density)
    min_sdf = []   # per ApplyDelta iteration
    rms = []       # per substep, interior bottom layer, (rho / rho0 - 1) RMS
    orig_apply = pbf_solver.pbf_apply_delta
    orig_final = pbf_solver.pbf_finalize

    def spy_apply(pred, *a, **k):
        orig_apply(pred, *a, **k)
        if on_box:
            min_sdf.append(float(_sdf_box(pred[:n, :3]).min().get()))

    def spy_final(pred, orig_pos, orig_vel, density, *a, **k):
        p = pred[:n, :3]
        layer = ((p[:, 1] < base + 1.5 * SPACING)
                 & (cp.abs(p[:, 0]) < 0.06) & (cp.abs(p[:, 2]) < 0.06))
        d = density[:n][layer]
        assert d.size > 0
        rms.append(float(cp.sqrt(cp.mean((d / rho0 - 1.0) ** 2)).get()))
        return orig_final(pred, orig_pos, orig_vel, density, *a, **k)

    monkeypatch.setattr(pbf_solver, "pbf_apply_delta", spy_apply)
    monkeypatch.setattr(pbf_solver, "pbf_finalize", spy_final)
    for _ in range(STEPS):
        sim._sim_step(n)
    cp.cuda.Device().synchronize()
    monkeypatch.undo()
    return np.array(min_sdf), np.array(rms)


@pytest.fixture(scope="module")
def runs():
    mp = pytest.MonkeyPatch()
    try:
        floor = _run(False, mp)
        box = _run(True, mp)
    finally:
        mp.undo()
        # SDF objects live in module constant memory shared by later tests.
        rigid_bodies.upload_sdf_objects(None, 0)
    return floor, box


def test_no_iteration_leaves_fluid_inside_sdf_box(runs):
    _, (min_sdf, _) = runs
    iters = PROFILES["PBF"].pbf_iterations
    assert len(min_sdf) == STEPS * iters
    worst = float(min_sdf.min())
    print(f"  min SDF distance over {len(min_sdf)} iterations: {worst:.6f} m")
    assert worst >= -BOUNDARY_MARGIN, (
        f"a PBF iteration left water {-worst * 1000:.2f} mm inside the SDF box")


def test_contact_layer_density_on_sdf_box_matches_floor(runs):
    (_, rms_floor), (_, rms_box) = runs
    f = rms_floor[SETTLE:]
    b = rms_box[SETTLE:]
    print(f"  bottom-layer density RMS after settle: floor mean={f.mean():.4f} "
          f"max={f.max():.4f}, box mean={b.mean():.4f} max={b.max():.4f}")
    assert b.max() < 0.10, f"box contact layer density RMS reached {b.max():.3f}"
    # Oscillation on the box is no larger than on the floor. 1.5x allows for
    # the box having no boundary particles under it, unlike the floor.
    assert b.std() <= 1.5 * f.std() + 1e-3, (
        f"box density RMS std {b.std():.4f} vs floor {f.std():.4f}")
