"""Pairwise Akinci surface tension (bd-r5-review-n28.125).

Surface tension used to be a one-body push, gamma * n_i, on any FLUID
particle with fewer than 25 neighbours. It had no reaction, so it added
net momentum, and it did not depend on curvature. WCSPH step2 now sums the
Akinci 2013 pair term (cohesion plus curvature) over FLUID neighbours, and
DFSPH NonPressureForces sums its curvature half.

Checks, all in zero gravity on a 0.4 x 0.16 x 0.16 m water block
(20 x 8 x 8 particles, aspect ratio 2.5):
- The surface tension accel alone sums to zero momentum:
  |sum m_i a_i| is below 1e-5 of sum m_i |a_i| (WCSPH and DFSPH).
- The block rounds up under surface tension: aspect ratio after 1 s is
  below 1.5 (WCSPH) and below 1.6 (DFSPH), and it stays near 2.5 with
  gamma = 0 in WCSPH.
- A water block at rest on a STONE slab is not pushed off it: the fluid
  centre of mass moves less than half a spacing away from the slab
  over 0.5 s.
"""

from __future__ import annotations

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, '_cache'):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, '_get_arch_for_options_for_nvrtc'):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, '_cache'):
                fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402
import dfsph_solver  # noqa: E402
import step2  # noqa: E402
from materials import STONE, WATER  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from world import World  # noqa: E402

SPACING = 0.02
BLOCK = ((-0.2, -0.08, -0.08), (0.2, 0.08, 0.08))


def _granular(profile, gamma):
    if profile.name == "DFSPH":
        # Same call as Simulation.set_solver_profile for DFSPH.
        return step2.build_granular_params(xsph_epsilon=0.1, surface_tension_gamma=gamma)
    return step2.build_granular_params(
        xsph_epsilon=profile.xsph_epsilon, force_scale=profile.force_scale,
        mu0=profile.mu0, surface_tension_gamma=gamma)


def _module(profile):
    return dfsph_solver if profile.name == "DFSPH" else step2


def _make(name, gamma, gravity=0.0, floor=False):
    world = World(max_particles=20_000)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=1)
    profile = PROFILES[name]
    sim.set_solver_profile(profile)
    sim.set_gravity(gravity)
    if floor:
        world.spawn_cube(min_corner=(-0.3, -0.16, -0.16), max_corner=(0.3, -0.10, 0.16),
                         material_id=STONE, spacing=SPACING)
        world.spawn_cube(min_corner=(-0.1, -0.08, -0.1), max_corner=(0.1, 0.06, 0.1),
                         material_id=WATER, spacing=SPACING)
    else:
        world.spawn_cube(min_corner=BLOCK[0], max_corner=BLOCK[1],
                         material_id=WATER, spacing=SPACING)
    sim.notify_scene_changed()
    _module(profile).upload_granular_params(_granular(profile, gamma))
    sim._invalidate_graphs()
    sim._spawn_substep = sim._damping_duration  # no spawn damping
    return sim, world, profile


def _restore(profile):
    # Put the default gamma back for later tests in the same process.
    _module(profile).upload_granular_params(_granular(profile, float(step2.DEFAULT_SURFACE_TENSION_GAMMA)))


def _aspect_ratio(pos):
    d = pos - pos.mean(0)
    ev = np.linalg.eigvalsh(d.T @ d / len(d))
    return float(np.sqrt(ev[-1] / ev[0]))


def _st_accel(sim, world, profile, gamma):
    """Accel from one force pass with this gamma, for the current sorted state."""
    w = world
    n = w._high_water
    mod = _module(profile)
    mod.upload_granular_params(_granular(profile, gamma))
    if profile.name == "DFSPH":
        out = cp.zeros((n, 4), cp.float32)
        dfsph_solver.compute_non_pressure_forces(
            w.sorted_position[:n], w.sorted_velocity[:n], w.sorted_density[:n],
            w.sorted_mass[:n], w.sorted_packed_info[:n], w.sorted_shear_rate[:n],
            w.sorted_temperature[:n], sim._cell_start, sim._cell_end, out,
            vorticity_in=w.sorted_vorticity[:n], normal_in=w.sorted_normal[:n])
        cp.cuda.Device().synchronize()
        return (cp.asnumpy(out[:, :3]) - cp.asnumpy(w.sorted_velocity[:n, :3])) / profile.dt
    force = cp.zeros((n, 4), cp.float32)
    veleval = cp.zeros((n, 4), cp.float32)
    step2.compute_step2(
        w.sorted_position[:n], w.sorted_velocity[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], w.sorted_shear_rate[:n], sim._cell_start, sim._cell_end,
        vorticity_in=w.sorted_vorticity, normal_in=w.sorted_normal, sph_force_out=force,
        veleval_out=veleval, velocity_h=w.sorted_velocity_h, pressure_in=w.sorted_pressure)
    cp.cuda.Device().synchronize()
    return cp.asnumpy(force[:, :3])


@pytest.mark.parametrize("name", ["WCSPH", "DFSPH"])
def test_surface_tension_adds_no_momentum(name):
    sim, world, profile = _make(name, 0.0)
    n = world._high_water
    for _ in range(3):
        sim._sim_step(n)
    a = _st_accel(sim, world, profile, 1.0) - _st_accel(sim, world, profile, 0.0)
    _restore(profile)
    m = cp.asnumpy(world.sorted_mass[:n])
    net = np.abs((m[:, None] * a).sum(0)).max()
    total = (m * np.linalg.norm(a, axis=1)).sum()
    assert total > 0.0, "surface tension produced no accel"
    assert net < 1e-5 * total, f"net {net:.3e} vs total {total:.3e}"


@pytest.mark.parametrize("name,limit", [("WCSPH", 1.5), ("DFSPH", 1.6)])
def test_block_rounds_up(name, limit):
    sim, world, profile = _make(name, 1.0)
    n = world._high_water
    ar0 = _aspect_ratio(cp.asnumpy(world.position[:n, :3]))
    for _ in range(int(round(1.0 / profile.dt))):
        sim._sim_step(n)
    ar1 = _aspect_ratio(cp.asnumpy(world.position[:n, :3]))
    _restore(profile)
    assert ar0 > 2.4
    assert ar1 < limit, f"{name}: aspect ratio {ar0:.2f} -> {ar1:.2f} after 1 s"


def test_block_keeps_shape_without_surface_tension():
    sim, world, profile = _make("WCSPH", 0.0)
    n = world._high_water
    for _ in range(1000):
        sim._sim_step(n)
    ar1 = _aspect_ratio(cp.asnumpy(world.position[:n, :3]))
    _restore(profile)
    assert ar1 > 2.3, f"aspect ratio {ar1:.2f} without surface tension"


@pytest.mark.parametrize("name", ["WCSPH", "DFSPH"])
def test_water_on_stone_stays_on_it(name):
    sim, world, profile = _make(name, 1.0, floor=True)
    n = world._high_water
    pi = cp.asnumpy(world.packed_info[:n])
    water = (pi & 0xFF) == WATER
    y0 = cp.asnumpy(world.position[:n, 1])[water].mean()
    for _ in range(int(round(0.5 / profile.dt))):
        sim._sim_step(n)
    y1 = cp.asnumpy(world.position[:n, 1])[water].mean()
    _restore(profile)
    assert y1 - y0 < 0.5 * SPACING, f"{name}: water COM rose {y1 - y0:.4f} m"
