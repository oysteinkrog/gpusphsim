"""DFSPH per-substep neighbour list (bd-r5-review-n28.198).

The _NL kernel variants read the neighbour list built by
K_DFSPH_BuildNeighborList instead of walking the hash grid. They see the
same neighbours in the same order, so every result must equal the hash
path, including particles whose neighbour count exceeds the list width
(they fall back to the hash walk).
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
    return World(max_particles=12000)


@pytest.fixture
def sim(world):
    return Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                      fixed_dt=False, max_substeps=20)


def _scene(w, sim, spacing, seed):
    """WATER block on a STONE layer with jittered positions, then density,
    alpha and the volume carrier. A small spacing packs more than
    NL_MAX neighbours into h, which exercises the hash fallback."""
    w.packed_info[:] = 0
    w._high_water = 0
    n_stone = w.spawn_cube((-0.09, 0.18, -0.09), (0.09, 0.2, 0.09), STONE, spacing=0.02)
    n = n_stone + w.spawn_cube((-0.07, 0.2, -0.07), (0.07, 0.34, 0.07), WATER,
                               spacing=spacing)
    rng = np.random.default_rng(seed)
    pos = w.position[:n].get()
    fluid = ((w.packed_info[:n].get() >> 8) & 3) != STATIC
    pos[fluid, :3] += rng.uniform(-0.2, 0.2, (int(fluid.sum()), 3)) * spacing
    w.position[:n] = cp.asarray(pos)
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


def _build(w, sim, n):
    nl = dfsph_solver.alloc_neighbor_list(w.max_particles)
    dfsph_solver.build_neighbor_list(
        w.sorted_position[:n], w.sorted_packed_info[:n],
        sim._cell_start, sim._cell_end, nl,
    )
    return nl


def _inputs(n, wall, seed):
    rng = np.random.default_rng(seed)
    kv = cp.asarray(rng.uniform(0.0, 1.0, n).astype(np.float32))
    p = cp.asarray(rng.uniform(0.0, 1.0, n).astype(np.float32))
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, :3] = rng.uniform(-0.5, 0.5, (n, 3))
    vel[wall, :3] = 0.0
    return kv, p, cp.asarray(vel)


def _all_kernels(w, sim, n, kv, p, vel, nlkw):
    """Run each list-capable kernel once on fixed inputs; return outputs."""
    out = {}
    pos, pi = w.sorted_position[:n], w.sorted_packed_info[:n]
    dens, mass = w.sorted_density[:n], w.sorted_mass[:n]
    cs, ce = sim._cell_start, sim._cell_end

    kv_out = cp.zeros(n, dtype=cp.float32)
    kv_tot = cp.zeros(n, dtype=cp.float32)
    dfsph_solver.compute_kappa_v(vel, w.sorted_alpha_dfsph[:n], pi, pos, cs, ce,
                                 kv_out, kappa_v_total=kv_tot, **nlkw)
    out["kappa_v"] = kv_out

    v = vel.copy()
    dfsph_solver.correct_velocity_div(v, dens, mass, kv, pi, pos, cs, ce, **nlkw)
    out["div_vel"] = v

    vt = cp.zeros_like(vel)
    dfsph_solver.compute_pressure_accel(p, vel, pos, dens, mass, pi, cs, ce, vt, **nlkw)
    out["vt"] = vt

    v2 = vel.copy()
    vt2 = cp.zeros_like(vel)
    dfsph_solver.correct_div_and_pressure_accel(v2, dens, mass, kv, p, pi, pos,
                                                cs, ce, vt2, **nlkw)
    out["fused_vel"] = v2
    out["fused_vt"] = vt2

    p2 = p.copy()
    dfsph_solver.density_solver_update(vt, pos, dens, w.sorted_alpha_dfsph[:n],
                                       pi, cs, ce, p2, **nlkw)
    out["p_rho2"] = p2
    cp.cuda.Device().synchronize()
    return {k: a.get() for k, a in out.items()}


@pytest.mark.parametrize("spacing,expect_overflow",
                         [(0.02, False), (0.009, True)],
                         ids=["normal", "overflow-fallback"])
def test_list_path_equals_hash_path(world, sim, spacing, expect_overflow):
    w = world
    n = _scene(w, sim, spacing, seed=3)
    pi = w.sorted_packed_info[:n].get()
    wall = ((pi >> 8) & 3) == STATIC
    assert 50 < wall.sum() < n
    nl = _build(w, sim, n)
    counts = nl[1][:n].get()
    if expect_overflow:
        assert (counts > dfsph_solver.NL_MAX).any()
    else:
        assert counts.max() <= dfsph_solver.NL_MAX
    assert (counts[wall] == 0).all()
    assert counts[~wall].min() > 0

    kv, p, vel = _inputs(n, wall, seed=7)
    ref = _all_kernels(w, sim, n, kv, p, vel, {})
    new = _all_kernels(w, sim, n, kv, p, vel, {"neighbor_list": nl})
    for key in ref:
        # The fields must carry real values, or equality proves nothing.
        assert np.abs(ref[key]).max() > 0.0, key
        # Same neighbours, same order, same arithmetic: bit-identical.
        np.testing.assert_array_equal(new[key], ref[key], err_msg=key)


def test_list_matches_brute_force(world, sim):
    """Every list entry is a non-GAS neighbour within h, in hash order, and
    the count equals a brute-force count."""
    w = world
    n = _scene(w, sim, 0.02, seed=5)
    nl = _build(w, sim, n)
    lst, cnt, stride = (nl[0].get(), nl[1][:n].get(), nl[2])
    pos = w.sorted_position[:n].get()[:, :3]
    pi = w.sorted_packed_info[:n].get()
    wall = ((pi >> 8) & 3) == STATIC
    h_sq = float(np.float32(sim._h) * np.float32(sim._h))
    rng = np.random.default_rng(1)
    for i in rng.choice(np.flatnonzero(~wall), 200, replace=False):
        d = pos - pos[i]
        r_sq = (d * d).sum(axis=1)
        mask = (r_sq <= h_sq) & (r_sq >= 1e-12)
        mask[i] = False
        # Boundary-of-h float noise would flip a pair; the jitter makes it rare.
        assert abs(int(cnt[i]) - int(mask.sum())) <= 1, i
        ids = lst[np.arange(min(cnt[i], dfsph_solver.NL_MAX)) * stride + i]
        assert len(set(ids.tolist())) == len(ids)
        assert (r_sq[ids] <= h_sq * (1 + 1e-5)).all()


def test_substep_same_with_and_without_list():
    """Three full DFSPH substeps from the same fresh state give the same
    positions and velocities with the list on and off.

    The per-kernel test above checks bit identity. Whole substeps are not
    bit-reproducible run to run under pytest even with the list off (about
    2e-4 m in position and 7e-3 m/s in velocity after three substeps), so
    this check allows that spread and only proves the list is wired in."""
    results = []
    for enabled in (False, True):
        w = World(max_particles=12000)
        n_stone = w.spawn_cube((-0.12, 0.0, -0.12), (0.12, 0.02, 0.12), STONE, spacing=0.02)
        n = n_stone + w.spawn_cube((-0.08, 0.04, -0.08), (0.08, 0.2, 0.08), WATER,
                                   spacing=0.02)
        s = Simulation(w, dt=0.001, speed=1.0, accuracy=0.4,
                       fixed_dt=False, max_substeps=20)
        s.set_solver_profile(PROFILES["DFSPH"])
        s.dfsph_neighbor_list_enabled = enabled
        for _ in range(3):
            s._sim_step(n)
        cp.cuda.Device().synchronize()
        assert (s._dfsph_nl is not None) == enabled
        results.append((w.position[:n].get().copy(), w.velocity[:n].get().copy()))
    (p0, v0), (p1, v1) = results
    assert np.abs(v0[:, :3]).max() > 0.0
    np.testing.assert_allclose(p1[:, :3], p0[:, :3], rtol=0, atol=1e-3)
    np.testing.assert_allclose(v1[:, :3], v0[:, :3], rtol=0, atol=3e-2)


def test_nl_kernel_register_budget():
    mod = dfsph_solver._get_module()
    for name in ("K_DFSPH_ComputeKappaV_NL", "K_DFSPH_CorrectVelocityDiv_NL",
                 "K_DFSPH_ComputePressureAccel_NL",
                 "K_DFSPH_CorrectDivAndPressureAccel_NL",
                 "K_DFSPH_DensitySolverUpdate_NL"):
        fn = mod.get_function(name)
        # __launch_bounds__(256, 4) allows at most 64 registers per thread.
        assert fn.num_regs <= 64, (name, fn.num_regs)
        print(f"{name}: {fn.num_regs} registers")
