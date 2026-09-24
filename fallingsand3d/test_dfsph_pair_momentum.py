"""DFSPH pressure and divergence corrections conserve linear momentum (bd-r5-review-n28.100).

K_DFSPH_ComputePressureAccel and K_DFSPH_CorrectVelocityDiv use the symmetric
pair form m_j * (k_i/rho_i + k_j/rho_j), so the pair terms on i and j cancel
even where rho_i != rho_j. The old V_j * (k_i + k_j) form leaked momentum at
every density jump.

Scene: a WATER block touching a MUD block (rest density 4500 vs 2500), zero
gravity, spawn damping off, far from the walls, 300 DFSPH substeps. Only the
pressure solvers move the fluid: the non-pressure force pass is replaced by a
copy and reactions are skipped. The clamps in the correction path would also
break conservation, so the test tracks on the GPU how close each one came to
firing and reads everything back once at the end.
"""

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
from world import World  # noqa: E402
from simulation import Simulation  # noqa: E402
from solver_profiles import PROFILES  # noqa: E402
from materials import WATER, MUD  # noqa: E402

MAX_PARTICLES = 20_000
SUBSTEPS = 300


def _norm(v):
    return cp.sqrt(cp.sum(v[:, :3] * v[:, :3], axis=1))


def run_two_density_block(substeps=SUBSTEPS, monkeypatch=None):
    """Run the scene and return (momentum drift ratio, clamp ratios, KE)."""
    w = World(max_particles=MAX_PARTICLES)
    w.packed_info[:] = 0
    w._high_water = 0
    # 10 x 10 x 10 of each material, touching along x at x = 0.
    n = w.spawn_cube((-0.2, -0.1, -0.1), (0.0, 0.1, 0.1), WATER, spacing=0.02)
    n += w.spawn_cube((0.0, -0.1, -0.1), (0.2, 0.1, 0.1), MUD, spacing=0.02)
    profile = PROFILES["DFSPH"]
    s = Simulation(w, dt=profile.dt, speed=1.0, fixed_dt=True, max_substeps=1)
    s.set_solver_profile(profile)
    s.set_gravity(0.0)
    s._spawn_substep = s._damping_duration

    dt = float(s._sim_params[0]["dt"])
    h = float(s._sim_params[0]["smoothing_length"])
    v_limit = float(s._sim_params[0]["velocity_limit"])

    # Running maxima on the GPU; one readback at the end.
    max_div_dv = cp.zeros(
        (), dtype=cp.float32
    )  # |dt * dv| in CorrectVelocityDiv, clamp at h/dt
    max_a_press = cp.zeros(
        (), dtype=cp.float32
    )  # |a_press| in ApplyPressureVelocity, clamp at h/dt^2
    max_speed = cp.zeros(
        (), dtype=cp.float32
    )  # |v| after the density solve, clamp at v_limit

    orig_correct = dfsph_solver.correct_velocity_div
    orig_apply = dfsph_solver.apply_pressure_velocity

    def correct(velocity, *args, **kwargs):
        before = velocity.copy()
        orig_correct(velocity, *args, **kwargs)
        cp.maximum(max_div_dv, cp.max(_norm(before - velocity)), out=max_div_dv)

    def apply(velocity, accel, *args, **kwargs):
        cp.maximum(max_a_press, cp.max(_norm(accel)), out=max_a_press)
        orig_apply(velocity, accel, *args, **kwargs)
        cp.maximum(max_speed, cp.max(_norm(velocity)), out=max_speed)

    def no_forces(position, velocity, *args, **kwargs):
        args[7][...] = velocity  # the veleval output buffer

    monkeypatch.setattr(dfsph_solver, "correct_velocity_div", correct)
    monkeypatch.setattr(dfsph_solver, "apply_pressure_velocity", apply)
    monkeypatch.setattr(dfsph_solver, "compute_non_pressure_forces", no_forces)
    monkeypatch.setattr(s, "_run_reactions_spawn", lambda *a, **k: None)

    for _ in range(substeps):
        n = s._substep_prologue(n)
        s._count_substep_sort(n)
        s._run_graph_body(n)
        s._substep_epilogue(n)

    m = w.mass[:n]
    v = w.velocity[:n, :3]
    mv = m[:, None] * v
    p = cp.sqrt(cp.sum(cp.sum(mv, axis=0) ** 2))
    scale = cp.sum(m * _norm(w.velocity[:n]))
    ke = 0.5 * cp.sum(m * cp.sum(v * v, axis=1))
    p, scale, ke = float(p), float(scale), float(ke)
    div_dv, a_press, speed = float(max_div_dv), float(max_a_press), float(max_speed)
    ok = bool(cp.isfinite(v).all())
    return {
        "finite": ok,
        "drift": p / scale if scale > 0 else 0.0,
        "p": p,
        "scale": scale,
        "ke": ke,
        "div_clamp": div_dv / (h / dt),
        "press_clamp": a_press / (h / (dt * dt)),
        "speed_clamp": speed / v_limit,
        "n": n,
    }


def test_two_density_block_conserves_momentum(monkeypatch):
    r = run_two_density_block(monkeypatch=monkeypatch)
    print(
        f"  n={r['n']} |P|={r['p']:.3e} sum m|v|={r['scale']:.3e} drift={r['drift']:.2e} "
        f"KE={r['ke']:.3e} clamp ratios: div={r['div_clamp']:.3f} "
        f"press={r['press_clamp']:.3f} speed={r['speed_clamp']:.3f}"
    )
    assert r["finite"]
    # The interface must actually drive flow, or the test proves nothing.
    assert r["scale"] > 1e-4, "no motion: the density jump did not load the solvers"
    # No clamp came near firing, so the solvers alone set the velocity.
    assert r["div_clamp"] < 0.5
    assert r["press_clamp"] < 0.5
    assert r["speed_clamp"] < 0.5
    # Net momentum is float round-off against the total momentum magnitude.
    # Measured: 7e-9 with the symmetric form, 0.115 with the old V_j form.
    assert r["drift"] < 1e-5, f"momentum drift {r['drift']:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-s"])
