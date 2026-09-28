"""DFSPH warm start scaled in K_DFSPH_Finalize, velocity kept in sorted_veleval (bd-r5-review-n28.173).

- Finalize writes kappa times dfsph_warm_start and kappa_v times
  dfsph_div_warm_start, so the host no longer runs two multiply passes.
- With a zero warm start, kappa is written unscaled (mu(I) still reads it)
  and kappa_v is written as 0.
- After K_DFSPH_NonPressureForces the substep velocity lives in
  sorted_veleval: Finalize reads it (unsorted velocity matches it) and foam
  generation is handed sorted_veleval, not the stale sorted_velocity.
"""

import dataclasses

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

import simulation as simulation_mod  # noqa: E402
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import WATER  # noqa: E402

MAX_PARTICLES = 20_000


@pytest.fixture(scope="module")
def world():
    return World(max_particles=MAX_PARTICLES)


def _run(w, profile, substeps=20):
    w.packed_info[:] = 0
    w._high_water = 0
    n = w.spawn_cube((-0.12, -0.9, -0.12), (0.12, -0.6, 0.12), WATER, spacing=0.02)
    s = Simulation(w, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    s.set_solver_profile(profile)
    s._spawn_substep = s._damping_duration  # no spawn damping
    for _ in range(substeps):
        s._sim_step(n)
    cp.cuda.Device().synchronize()
    return s, n


def _unsorted_at_sorted(arr, perm):
    """arr[perm[i]] for each sorted slot i."""
    return arr[perm]


# One substep for the kappa checks: the lattice cube starts just above rho0
# (2524 vs 2500), so the first substep has kappa > 0 on 1300 of 2160
# particles. The free cube then expands and kappa is 0 by substep 5. The
# one-body DFSPH surface tension used to hold it compressed; the pair form
# (bd-r5-review-n28.125) does not.
KAPPA_SUBSTEPS = 1


def test_finalize_writes_scaled_warm_start(world):
    w = world
    profile = PROFILES["DFSPH"]
    s, n = _run(w, profile, substeps=KAPPA_SUBSTEPS)
    perm = s._sort_perm[:n].get()
    kappa = _unsorted_at_sorted(w.kappa[:n].get(), perm)
    kappa_v = _unsorted_at_sorted(w.kappa_v[:n].get(), perm)
    sk = w.sorted_kappa[:n].get()
    skv = w.sorted_kappa_v[:n].get()
    assert np.count_nonzero(sk) > 0 and np.count_nonzero(skv) > 0
    np.testing.assert_allclose(kappa, sk * np.float32(profile.dfsph_warm_start),
                               rtol=1e-6, atol=0)
    np.testing.assert_allclose(kappa_v, skv * np.float32(profile.dfsph_div_warm_start),
                               rtol=1e-6, atol=0)


def test_zero_warm_start_carries_unscaled_kappa_and_zero_kappa_v(world):
    w = world
    profile = dataclasses.replace(PROFILES["DFSPH"], dfsph_warm_start=0.0,
                                  dfsph_div_warm_start=0.0)
    s, n = _run(w, profile, substeps=KAPPA_SUBSTEPS)
    perm = s._sort_perm[:n].get()
    kappa = _unsorted_at_sorted(w.kappa[:n].get(), perm)
    kappa_v = w.kappa_v[:n].get()
    sk = w.sorted_kappa[:n].get()
    v = w.velocity[:n, :3].get()
    assert np.isfinite(v).all()
    assert np.count_nonzero(sk) > 0
    np.testing.assert_array_equal(kappa, sk)
    assert np.all(kappa_v == 0.0)


def test_velocity_routed_through_veleval(world, monkeypatch):
    w = world
    s, n = _run(w, PROFILES["DFSPH"], substeps=10)
    perm = s._sort_perm[:n].get()
    v_out = _unsorted_at_sorted(w.velocity[:n, :3].get(), perm)
    veleval = w.sorted_veleval[:n, :3].get()
    stale = w.sorted_velocity[:n, :3].get()
    # Finalize writes the veleval velocity back (no clamp fires in a calm block).
    np.testing.assert_allclose(v_out, veleval, rtol=0, atol=1e-6)
    # sorted_velocity is the pre-forces input, so gravity alone separates them.
    assert np.max(np.abs(stale - veleval)) > 1e-4

    # Foam generation reads the DFSPH velocity.
    seen = {}

    def fake_generate(pos, vel, *args, **kwargs):
        seen["ptr"] = vel.data.ptr

    monkeypatch.setattr(simulation_mod.foam, "foam_generate", fake_generate)
    monkeypatch.setattr(simulation_mod.foam, "foam_physics", lambda *a, **k: None)
    monkeypatch.setattr(w, "foam_enabled", True)
    s._run_foam_step(n, generate=True)
    assert seen["ptr"] == w.sorted_veleval.data.ptr
