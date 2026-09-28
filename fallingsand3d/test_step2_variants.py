"""Regression test for bd-r5-review-n28.191: K_Step2 template variants.

step2.cu builds one K_Step2 body templated on HAS_RIGID and HAS_VORT into
four kernels. step2.py picks K_Step2_NoRigid when there are no rigid bodies
(the rigid branch made K_Step2 spill 48 B of local memory), and the NoVort
kernels only when the uploaded vorticity_epsilon is 0.

The checks:
- register and local-memory counts: the no-rigid kernels do not spill, and
  the kernel without both fits the 64 registers of its (256, 4) bound;
- kernel choice from the host-known switches;
- on the same sorted state, every variant that applies gives the same
  sph_force and veleval as K_Step2 (the bead asks for rtol 1e-6; the
  variants only drop dead branches, so the outputs are bit-identical).

The scene mixes FLUID, GRANULAR, GAS and STATIC particles, so every pair
branch runs.
"""

import numpy as np
import pytest

import cupy as cp
import step2
from materials import SAND, SMOKE, STONE, WATER
from simulation import Simulation
from solver_profiles import PROFILES
from world import World

SPACING = 0.02
SUBSTEPS = 40
RTOL = 1e-6

ALL_VARIANTS = ("K_Step2", "K_Step2_NoRigid", "K_Step2_NoVort", "K_Step2_NoRigidNoVort")


def test_kernel_name_choice():
    assert step2.step2_kernel_name(True, True) == "K_Step2"
    assert step2.step2_kernel_name(False, True) == "K_Step2_NoRigid"
    assert step2.step2_kernel_name(True, False) == "K_Step2_NoVort"
    assert step2.step2_kernel_name(False, False) == "K_Step2_NoRigidNoVort"


@pytest.mark.gpu
def test_register_and_spill_counts():
    mod = step2.get_module()
    counts = {}
    for name in ALL_VARIANTS:
        k = mod.get_function(name)
        counts[name] = (k.num_regs, k.local_size_bytes)
    print("K_Step2 variants (regs, local B):", counts)
    assert counts["K_Step2_NoRigid"][1] == 0, counts
    assert counts["K_Step2_NoRigidNoVort"][1] == 0, counts
    assert counts["K_Step2_NoRigidNoVort"][0] <= 64, counts


def _build_sim():
    world = World(max_particles=40_000)
    counts = (
        world.spawn_cube((-0.9, -0.9, -0.15), (-0.5, -0.1, 0.15), WATER, spacing=SPACING),
        world.spawn_cube((0.2, -0.9, -0.1), (0.4, -0.7, 0.1), SAND, spacing=SPACING),
        world.spawn_cube((-0.3, -0.5, -0.1), (-0.1, -0.3, 0.1), SMOKE, spacing=SPACING),
        world.spawn_cube((-0.4, -0.9, -0.1), (-0.3, -0.6, 0.1), STONE, spacing=SPACING),
    )
    assert min(counts) > 0
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=True, max_substeps=1)
    sim.set_solver_profile(PROFILES["WCSPH"])
    sim._spawn_substep = sim._damping_duration  # no spawn damping ramp
    for _ in range(SUBSTEPS):
        sim._sort_skip_next = False  # full sort, so sorted buffers match n
        sim._sim_step(world._high_water)
    return sim


def _run_variant(sim, name, monkeypatch):
    monkeypatch.setattr(step2, "step2_kernel_name", lambda has_rigid, vort: name)
    w = sim.world
    n = w._high_water
    force = cp.zeros((n, 4), dtype=cp.float32)
    veleval = cp.zeros((n, 4), dtype=cp.float32)
    step2.compute_step2(
        w.sorted_position[:n],
        w.sorted_velocity[:n],
        w.sorted_mass[:n],
        w.sorted_packed_info[:n],
        w.sorted_shear_rate[:n],
        sim._cell_start,
        sim._cell_end,
        vorticity_in=w.sorted_vorticity,
        normal_in=w.sorted_normal,
        sph_force_out=force,
        veleval_out=veleval,
        velocity_h=w.sorted_velocity_h,
        pressure_in=w.sorted_pressure,
        pr2_omega_in=w.sorted_pr2_omega,
        eta_in=w.sorted_eta,
    )
    return cp.asnumpy(force), cp.asnumpy(veleval)


def _assert_same(a, b, what):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=0.0, err_msg=what)


@pytest.mark.gpu
def test_variants_match_full_kernel(monkeypatch):
    sim = _build_sim()
    # Default WCSPH profile: vorticity on, no rigid bodies -> live path is NoRigid.
    assert step2._vorticity_on
    assert sim.rigid_body_manager.num_bodies == 0
    f_ref, v_ref = _run_variant(sim, "K_Step2", monkeypatch)
    assert np.all(np.isfinite(f_ref)) and np.any(f_ref != 0.0)
    f, v = _run_variant(sim, "K_Step2_NoRigid", monkeypatch)
    _assert_same(f, f_ref, "sph_force NoRigid, vorticity on")
    _assert_same(v, v_ref, "veleval NoRigid, vorticity on")

    # vorticity_epsilon = 0: all four variants apply and must agree.
    base = sim._wcsph_granular_params()
    off = base.copy()
    off[0]["vorticity_epsilon"] = 0.0
    try:
        step2.upload_granular_params(off)
        assert not step2._vorticity_on
        f_ref, v_ref = _run_variant(sim, "K_Step2", monkeypatch)
        for name in ALL_VARIANTS[1:]:
            f, v = _run_variant(sim, name, monkeypatch)
            _assert_same(f, f_ref, f"sph_force {name}, vorticity off")
            _assert_same(v, v_ref, f"veleval {name}, vorticity off")
    finally:
        step2.upload_granular_params(base)
    assert step2._vorticity_on
