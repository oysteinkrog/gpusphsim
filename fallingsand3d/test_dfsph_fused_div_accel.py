"""Fused last divergence pass and first pressure sweep (bd-r5-review-n28.192).

K_DFSPH_CorrectDivAndPressureAccel must give the same velocity, predicted
velocity vt and rigid reaction as K_DFSPH_CorrectVelocityDiv followed by
K_DFSPH_ComputePressureAccel on the same inputs.
"""

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
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import STATIC, STONE, WATER  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return World(max_particles=4000)


@pytest.fixture
def sim(world):
    return Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                      fixed_dt=False, max_substeps=20)


def _scene(w, sim):
    """7x7x7 WATER block on a 9x9 STONE layer, density/alpha, then pack."""
    w.packed_info[:] = 0
    w._high_water = 0
    n_stone = w.spawn_cube((-0.09, 0.18, -0.09), (0.09, 0.2, 0.09), STONE, spacing=0.02)
    n = n_stone + w.spawn_cube((-0.07, 0.2, -0.07), (0.07, 0.34, 0.07), WATER, spacing=0.02)
    sim.set_solver_profile(PROFILES["DFSPH"])
    w._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)
    for pass_idx in range(2):
        density_in = None if pass_idx == 0 else w.sorted_density[:n].copy()
        dfsph_solver.compute_density_alpha(
            w.sorted_position[:n], w.sorted_velocity[:n],
            w.sorted_mass[:n], density_in,
            w.sorted_packed_info[:n], w.sorted_temperature[:n],
            sim._cell_start, sim._cell_end,
            w.sorted_density, w.sorted_alpha_dfsph,
            w.sorted_shear_rate, w.sorted_dTdt,
            w.sorted_exposure_heat, w.sorted_exposure_corrode,
        )
    dfsph_solver.pack_volume(
        w.sorted_position[:n], w.sorted_density[:n],
        w.sorted_mass[:n], w.sorted_packed_info[:n],
    )
    cp.cuda.Device().synchronize()
    return n


def _inputs(n, wall, seed, kv_scale, p_scale, v_scale):
    rng = np.random.default_rng(seed)
    kv = rng.uniform(0.0, kv_scale, n).astype(np.float32)
    p = rng.uniform(0.0, p_scale, n).astype(np.float32)
    # Walls keep junk in both buffers: the kernels must zero them.
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, :3] = rng.uniform(-v_scale, v_scale, (n, 3))
    vel[:, 3] = rng.uniform(0.0, 1.0, n)
    vel[wall, :3] = 0.0
    return cp.asarray(kv), cp.asarray(p), cp.asarray(vel)


def _two_kernels(w, sim, n, kv, p, vel, alias, rigid=None, press_reaction=True):
    rigid = rigid or {}
    v = vel.copy()
    dfsph_solver.correct_velocity_div(
        v, w.sorted_density[:n], w.sorted_mass[:n], kv,
        w.sorted_packed_info[:n], w.sorted_position[:n],
        sim._cell_start, sim._cell_end, **rigid,
    )
    vt = v if alias else cp.zeros_like(v)
    dfsph_solver.compute_pressure_accel(
        p, v, w.sorted_position[:n], w.sorted_density[:n], w.sorted_mass[:n],
        w.sorted_packed_info[:n], sim._cell_start, sim._cell_end, vt,
        **(rigid if press_reaction else {}),
    )
    return v, vt


def _fused(w, sim, n, kv, p, vel, alias, rigid=None, press_reaction=True):
    rigid = rigid or {}
    v = vel.copy()
    vt = v if alias else cp.zeros_like(v)
    dfsph_solver.correct_div_and_pressure_accel(
        v, w.sorted_density[:n], w.sorted_mass[:n], kv, p,
        w.sorted_packed_info[:n], w.sorted_position[:n],
        sim._cell_start, sim._cell_end, vt,
        **rigid, pressure_reaction=press_reaction,
    )
    return v, vt


@pytest.mark.parametrize("alias", [False, True], ids=["vt-buffer", "vt-is-vel"])
@pytest.mark.parametrize("kv_scale,p_scale,v_scale",
                         [(1.0, 1.0, 0.5), (1.0e4, 1.0e3, 0.5), (1.0, 1.0, 200.0)],
                         ids=["no-clamp", "delta-and-accel-clamp", "velocity-clamp"])
def test_fused_matches_two_kernels(world, sim, alias, kv_scale, p_scale, v_scale):
    w = world
    n = _scene(w, sim)
    wall = ((w.sorted_packed_info[:n].get() >> 8) & 3) == STATIC
    assert 50 < wall.sum() < n
    kv, p, vel = _inputs(n, wall, 11, kv_scale, p_scale, v_scale)

    v_ref, vt_ref = _two_kernels(w, sim, n, kv, p, vel, alias)
    v_new, vt_new = _fused(w, sim, n, kv, p, vel, alias)
    v_ref, vt_ref, v_new, vt_new = (a.get() for a in (v_ref, vt_ref, v_new, vt_new))

    # The corrections must be real, or the check proves nothing.
    assert np.abs(v_ref[~wall, :3] - vel.get()[~wall, :3]).max() > 1e-4
    assert np.abs(vt_ref[~wall, :3] - v_ref[~wall, :3]).max() > 1e-4 or alias
    # Same operations in the same order: equal up to FMA contraction noise.
    scale = max(float(np.abs(vt_ref).max()), 1e-6)
    np.testing.assert_allclose(v_new, v_ref, rtol=1e-5, atol=1e-6 * scale)
    np.testing.assert_allclose(vt_new, vt_ref, rtol=1e-5, atol=1e-6 * scale)
    # Walls: velocity untouched, vt = v.
    np.testing.assert_array_equal(v_new[wall], vel.get()[wall])
    np.testing.assert_array_equal(vt_new[wall], vel.get()[wall])


def test_fused_kernel_register_budget():
    fn = dfsph_solver._get_module().get_function("K_DFSPH_CorrectDivAndPressureAccel")
    # __launch_bounds__(256, 4) allows at most 64 registers per thread.
    assert fn.num_regs <= 64, fn.num_regs
    print(f"K_DFSPH_CorrectDivAndPressureAccel: {fn.num_regs} registers")


@pytest.mark.parametrize("press_reaction", [True, False], ids=["with-press", "div-only"])
def test_fused_rigid_reaction_matches(press_reaction):
    """Body force and torque from the fused sweep equal the two launches."""
    from test_rigid_coupling import HALF, _make_sim, _step

    world, rsim, rbm = _make_sim(
        "DFSPH",
        fluid_min=(-0.2, -0.2, -0.2), fluid_max=(0.2, 0.2, 0.2),
        body_pos=(0.2 + HALF - 0.03, 0.0, 0.0),
        gravity=0.0, n_max=40_000,
    )
    _step(rsim)  # inject the boundary block
    _step(rsim)  # sorted buffers now hold fluid and body particles
    assert rbm.boundary_injected
    n = world._high_water
    pi = world.sorted_packed_info[:n].get()
    wall = ((pi >> 8) & 3) == STATIC
    assert wall.sum() > 0
    kv, p, vel = _inputs(n, wall, 5, 50.0, 50.0, 0.2)
    rigid = {"d_rigid_bodies": rbm.d_rigid_bodies,
             "d_rigid_forces": rbm.rigid_forces,
             "d_rigid_torques": rbm.rigid_torques}

    def run(fn):
        rbm.rigid_forces.fill(0)
        rbm.rigid_torques.fill(0)
        v, vt = fn(world, rsim, n, kv, p, vel, False, rigid, press_reaction)
        cp.cuda.Device().synchronize()
        return (v.get(), vt.get(), rbm.rigid_forces.get().copy(),
                rbm.rigid_torques.get().copy())

    v_ref, vt_ref, f_ref, t_ref = run(_two_kernels)
    v_new, vt_new, f_new, t_new = run(_fused)

    assert np.abs(f_ref).max() > 0.0
    np.testing.assert_allclose(v_new, v_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(vt_new, vt_ref, rtol=1e-5, atol=1e-6)
    # Atomics sum in a different order: float reorder noise only.
    f_scale = float(np.abs(f_ref).max())
    t_scale = max(float(np.abs(t_ref).max()), 1e-12)
    np.testing.assert_allclose(f_new, f_ref, rtol=1e-4, atol=1e-5 * f_scale)
    np.testing.assert_allclose(t_new, t_ref, rtol=1e-4, atol=1e-5 * t_scale)
