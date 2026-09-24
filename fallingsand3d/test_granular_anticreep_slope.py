"""GRANULAR anti-creep must not freeze grains that are gaining speed (bd-r5-review-n28.93).

K_DFSPH_Finalize and K_PBF_Finalize zero a packed GRANULAR particle's
velocity when it is under 1 cm/s. Before the fix they did this with no
check on whether the grain was speeding up, so a packed grain restarting
from rest with acceleration below 0.01 m/s / dt (3 m/s^2 at DFSPH
dt = 1/300, 1.2 m/s^2 at PBF dt = 1/120) was re-zeroed every substep and
never moved. At the default |g| = 4, a grain on a 45 degree slope has a
tangential acceleration of 2.83 m/s^2.

The zeroing now also requires |v| <= the previous substep's final |v|,
read from the unsorted velocity array before Finalize overwrites it.

These tests call each Finalize kernel directly on one isolated SAND
particle with its density forced to rho0, so the anti-creep density gate
passes and nothing else changes the velocity:
- a grain speeding up from rest (prev 0, now 0.005 m/s) keeps its velocity;
- repeated substeps at a = 2.83 m/s^2 build velocity past 1 cm/s;
- a grain slowing down (prev 0.008, now 0.005 m/s) is still zeroed;
- a loose grain (rho = 0.5 rho0) is never zeroed.
"""

import numpy as np
import pytest


def _apply_ptx_workaround():
    try:
        import cupy.cuda.compiler as _compiler

        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, "_get_arch_for_options_for_nvrtc"):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, "_cache"):
                fn._cache = {}
    except Exception:
        pass


_apply_ptx_workaround()

import cupy as cp  # noqa: E402

import dfsph_solver  # noqa: E402
import pbf_solver  # noqa: E402
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import SAND, MATERIALS  # noqa: E402

RHO0 = float(MATERIALS[SAND].rest_density)
A_SLOPE = 4.0 * np.sqrt(0.5)  # tangential gravity on a 45 degree slope, |g| = 4


@pytest.fixture(scope="module", params=["DFSPH", "PBF"])
def single(request):
    solver = request.param
    world = World(max_particles=100)
    sim = Simulation(
        world, dt=PROFILES[solver].dt, speed=1.0, fixed_dt=True, max_substeps=1
    )
    world.packed_info[:] = 0
    world._high_water = 0
    n = world.spawn_cube((0.0, 0.0, 0.0), (0.02, 0.02, 0.02), SAND, spacing=0.02)
    assert n == 1
    sim.set_solver_profile(PROFILES[solver])
    sim._sim_params[0]["velocity_damping"] = np.float32(0.0)
    sim._upload_sim_params_all()
    world._density_initialized = True
    sim._run_grid_setup(n, force_sort=True)
    cp.cuda.Device().synchronize()
    yield solver, world, sim, n
    sim.close()


def _finalize(single, v_now, v_prev, rho):
    """Run one Finalize with |v| = v_now along x after the solve and v_prev as
    the previous substep's final velocity. Returns the final x velocity."""
    solver, w, sim, n = single
    dt = sim.dt
    w.velocity[:n] = cp.asarray([[v_prev, 0.0, 0.0, 0.0]], dtype=cp.float32)
    w.sorted_density[:n] = cp.float32(rho)
    w.sorted_sleep_counter[:n] = 0
    vel = cp.asarray([[v_now, 0.0, 0.0, 0.0]], dtype=cp.float32)
    common = dict(
        sorted_particle_dye=w.sorted_particle_dye[:n],
        particle_dye_out=w.particle_dye,
        sorted_angular_velocity=w.sorted_angular_velocity[:n],
        angular_velocity_out=w.angular_velocity,
    )
    if solver == "DFSPH":
        dfsph_solver.finalize(
            w.sorted_position[:n],
            vel,
            w.sorted_density[:n],
            w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            w.sorted_health[:n],
            w.sorted_dTdt[:n],
            w.sorted_sleep_counter[:n],
            cp.zeros(n, dtype=cp.float32),
            sim._sort_perm[:n],
            sim._cell_start,
            sim._cell_end,
            w.position,
            w.velocity,
            w.color,
            w.packed_info,
            w.sleep_counter,
            w.temperature,
            cp.zeros(n, dtype=cp.float32),
            **common,
        )
    else:
        # PBF derives velocity from positions: v = (pred - orig) / dt.
        orig = w.sorted_position[:n].copy()
        pred = orig.copy()
        pred[:, 0] += cp.float32(v_now * dt)
        pbf_solver.pbf_finalize(
            pred,
            orig,
            vel,
            w.sorted_density[:n],
            w.sorted_mass[:n],
            w.sorted_packed_info[:n],
            w.sorted_temperature[:n],
            w.sorted_health[:n],
            w.sorted_dTdt[:n],
            w.sorted_sleep_counter[:n],
            sim._sort_perm[:n],
            sim._cell_start,
            sim._cell_end,
            w.position,
            w.velocity,
            w.color,
            w.packed_info,
            w.sleep_counter,
            w.temperature,
            **common,
        )
    cp.cuda.Device().synchronize()
    return float(w.velocity[0, 0].get())


def test_accelerating_grain_keeps_velocity(single):
    vx = _finalize(single, v_now=0.005, v_prev=0.0, rho=RHO0)
    # PBF applies a packed-grain damping of (1 - 2 dt) before anti-creep.
    assert vx > 0.9 * 0.005, f"{single[0]}: accelerating grain was zeroed (vx={vx})"


def test_grain_escapes_from_rest_on_slope(single):
    solver, w, sim, n = single
    dt = sim.dt
    v = 0.0
    for _ in range(6):
        v = _finalize(single, v_now=v + A_SLOPE * dt, v_prev=v, rho=RHO0)
    print(f"{solver}: dt={dt:.5f}, v after 6 substeps = {v:.4f} m/s")
    assert v > 0.01, f"{solver}: grain stayed frozen on a 45 degree slope (v={v})"


def test_decelerating_creep_is_zeroed(single):
    vx = _finalize(single, v_now=0.005, v_prev=0.008, rho=RHO0)
    assert vx == 0.0, f"{single[0]}: decelerating creep not zeroed (vx={vx})"


def test_loose_grain_is_not_zeroed(single):
    vx = _finalize(single, v_now=0.005, v_prev=0.008, rho=0.5 * RHO0)
    assert vx > 0.9 * 0.005, f"{single[0]}: loose grain was zeroed (vx={vx})"
