"""Simulation-level tests for subsystems no other test covers (bd-r5-review-n28.129).

1. Blast wave window: a FIRE particle whose lifetime is within half a substep
   of GUNPOWDER_FIRE_LIFETIME kicks its neighbours; one 1.5 substeps away does
   not.
2. set_world_size: after growing the world to half size 2.0, particles at
   x = 1.5 keep their neighbours and are not clamped to the old wall.
3. CUDA graph replay gives the same positions as direct launches of
   _run_graph_body, for every solver and for both graphs.
4. PBF and DFSPH run wake propagation: a moving particle wakes the sleeping
   SAND in its cell stencil.
5. Foam: a splash makes foam, and compaction removes it once it has expired.

Every test drives a real Simulation through _sim_step and reads the state back
once after the loop.
"""

import os
import re

import numpy as np
import pytest

import cupy as cp
import reactions
from world import World
from simulation import Simulation
from solver_profiles import PROFILES
from materials import FIRE, GUNPOWDER, MATERIALS, SAND, WATER

pytestmark = pytest.mark.gpu

H = 0.04
_SLEEP_BIT = 0x400  # SET_SLEEPING in physics/kernels/common.cuh

_worlds = {}


def _get_world(key, n_max=20_000):
    """One World per key for the whole module (allocation is slow)."""
    w = _worlds.get(key)
    if w is None:
        w = World(max_particles=n_max)
        _worlds[key] = w
    return w


def _make_sim(world, solver, spawn_fn, world_half_size=1.0):
    """Clean the world, spawn a scene and build a Simulation for it.

    Spawn damping is skipped: its ramp changes constants every substep and
    invalidates the CUDA graphs.
    """
    world.packed_info[:] = 0
    world._high_water = 0
    world.sorted_density[:] = 0.0
    world.sleep_counter[:] = 0
    world.velocity[:] = 0.0
    world.foam_count[:] = 0
    world.foam_enabled = False
    n = spawn_fn(world)
    assert n > 0
    sim = Simulation(
        world,
        dt=0.001,
        speed=1.0,
        accuracy=0.4,
        fixed_dt=False,
        max_substeps=20,
        world_half_size=world_half_size,
    )
    sim.set_solver_profile(PROFILES[solver])
    sim._spawn_substep = sim._damping_duration
    sim.sim_time = 0.0
    sim._last_frame_time = None
    return sim, n


def _unsorted_density(sim, n):
    """sorted_density of the last substep, mapped back to particle order."""
    perm = sim._sort_perm[:n].astype(cp.int64)
    out = cp.empty(n, dtype=cp.float32)
    out[perm] = sim.world.sorted_density[:n]
    return out


def _gunpowder_fire_lifetime():
    src = os.path.join(os.path.dirname(__file__), "physics", "kernels", "reactions.cu")
    with open(src, encoding="utf-8") as f:
        m = re.search(r"#define\s+GUNPOWDER_FIRE_LIFETIME\s+([0-9.]+)f?", f.read())
    assert m, "GUNPOWDER_FIRE_LIFETIME not found in reactions.cu"
    return float(m.group(1))


# ---------------------------------------------------------------------------
# 1. Blast wave fires only within half a substep of the fresh FIRE lifetime
# ---------------------------------------------------------------------------


def _spawn_blast_scene(world):
    tiny = ((-0.001, -0.001, -0.001), (0.001, 0.001, 0.001))
    # spawn_cube puts lattice points at min + spacing/2, so both cubes put a
    # point at the origin; the water one is excluded.
    n_w = world.spawn_cube(
        (-0.07, -0.07, -0.07), (0.07, 0.07, 0.07), WATER, spacing=0.02, exclude_box=tiny
    )
    n_g = world.spawn_cube(
        (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), GUNPOWDER, spacing=0.02
    )
    assert n_g == 1
    return n_w + n_g


@pytest.mark.parametrize(
    "offset,fires",
    [
        (+0.4, True),
        (-0.4, True),
        (+1.5, False),
        (-1.5, False),
    ],
)
def test_blast_wave_window(monkeypatch, offset, fires):
    """The blast source is a FIRE particle with lifetime GUNPOWDER_FIRE_LIFETIME
    + offset * dt when K_BlastWave runs. Only |offset| < 0.5 may kick."""
    world = _get_world("a")
    sim, n = _make_sim(world, "WCSPH", _spawn_blast_scene)
    g_idx = n - 1
    lifetime = _gunpowder_fire_lifetime() + offset * sim.dt

    # Turn the gunpowder slot into FIRE with the test lifetime right after
    # Reactions, so K_BlastWave sees exactly that value.
    orig = reactions.compute_reactions

    def compute_reactions(
        sorted_packed_info, temperature, health, sorted_lifetime, *args, **kwargs
    ):
        orig(sorted_packed_info, temperature, health, sorted_lifetime, *args, **kwargs)
        gp = (sorted_packed_info & cp.uint32(0xFF)) == cp.uint32(GUNPOWDER)
        sorted_packed_info[gp] = (
            sorted_packed_info[gp] & cp.uint32(0xFFFFFF00)
        ) | cp.uint32(FIRE)
        sorted_lifetime[gp] = cp.float32(lifetime)

    monkeypatch.setattr(reactions, "compute_reactions", compute_reactions)

    pos0 = cp.asnumpy(world.position[:n])[:, :3]
    sim._sim_step(n)
    vel = cp.asnumpy(world.velocity[:n])[:, :3]

    r = pos0[:g_idx] - pos0[g_idx]
    d = np.linalg.norm(r, axis=1)
    near = d < H
    assert near.sum() >= 20, f"only {near.sum()} water neighbours within h"
    radial = (vel[:g_idx][near] * (r[near] / d[near, None])).sum(axis=1)
    mean_radial = float(radial.mean())
    print(f"\n  offset {offset:+.1f} dt: mean radial water speed {mean_radial:.3f} m/s")
    if fires:
        assert mean_radial > 1.0, (
            f"no blast at lifetime L{offset:+.1f}dt: {mean_radial:.3f} m/s"
        )
    else:
        assert abs(mean_radial) < 0.2, (
            f"blast fired at lifetime L{offset:+.1f}dt: {mean_radial:.3f} m/s"
        )


# ---------------------------------------------------------------------------
# 2. set_world_size re-uploads grid and sim constants to every solver
# ---------------------------------------------------------------------------

_FAR_CENTER = (1.5, 0.0, 0.0)


def _spawn_far_block(world):
    cx, cy, cz = _FAR_CENTER
    return world.spawn_cube(
        (cx - 0.1, cy - 0.1, cz - 0.1),
        (cx + 0.1, cy + 0.1, cz + 0.1),
        WATER,
        spacing=0.02,
    )


@pytest.mark.parametrize("solver", ["WCSPH", "PBF", "DFSPH"])
def test_set_world_size_far_particles_keep_neighbours(solver):
    """Grow the world to half size 2.0 after the solver is set, then run one
    substep on a water block at x = 1.5 (outside the old world)."""
    world = _get_world("a")
    sim, n = _make_sim(world, solver, _spawn_far_block, world_half_size=1.0)
    sim.set_world_size(2.0)
    pos0 = cp.asnumpy(world.position[:n])[:, :3]
    sim._sim_step(n)

    rho = cp.asnumpy(_unsorted_density(sim, n))
    pos = cp.asnumpy(world.position[:n])[:, :3]
    rho0 = MATERIALS[WATER].rest_density
    interior = np.all(np.abs(pos0 - np.array(_FAR_CENTER)) < 0.05, axis=1)
    assert interior.sum() >= 50
    rho_med = float(np.median(rho[interior]))
    x_mean = float(pos[:, 0].mean())
    print(
        f"\n  {solver}: interior median density {rho_med:.1f} (rho0 {rho0}), "
        f"mean x {x_mean:.4f}"
    )
    assert rho_med > 0.5 * rho0, (
        f"{solver}: particles at x=1.5 lost their neighbours after set_world_size "
        f"(median density {rho_med:.1f}, rho0 {rho0})"
    )
    assert abs(x_mean - _FAR_CENTER[0]) < 0.02, (
        f"{solver}: block moved from x=1.5 to {x_mean:.4f} (old wall at 1.0?)"
    )


# ---------------------------------------------------------------------------
# 3. CUDA graph replay matches direct _run_graph_body launches
# ---------------------------------------------------------------------------


def _spawn_dam(world):
    # 20 x 20 x 20 water block near the floor (y = -1), 8000 particles.
    return world.spawn_cube(
        (-0.2, -1.0, -0.2),
        (0.2, -0.6, 0.2),
        WATER,
        spacing=0.02,
        velocity=(0.3, 0.0, 0.0),
    )


_GRAPH_SUBSTEPS = 5


def _run_substeps(sim, n, gather):
    for i in range(_GRAPH_SUBSTEPS):
        # Substep 0 always sorts, so the gather-only graph has a sort for n.
        sim._sort_skip_next = gather and i > 0
        sim._sim_step(n)


def _run_direct(world, solver, gather, n):
    """Run the scene with every graph capture replaced by a direct launch."""
    sim, n_b = _make_sim(world, solver, _spawn_dam)
    assert n_b == n

    def direct(n_, gather_only):
        sim._run_graph_body(n_)
        return None

    sim._capture_graph = direct
    _run_substeps(sim, n, gather)
    assert sim._graph_full_sort is None and sim._graph_gather_only is None
    return cp.asnumpy(world.position[:n])[:, :3]


@pytest.mark.parametrize("gather", [False, True], ids=["full_sort", "gather_only"])
@pytest.mark.parametrize("solver", ["WCSPH", "PBF", "DFSPH"])
def test_graph_matches_direct_launch(solver, gather):
    """Same scene twice: once through _sim_step with graph capture and replay,
    once with every graph capture replaced by a direct _run_graph_body call.
    Positions match to 1e-5 of the scene extent (not bitwise: atomics set the
    order within a cell)."""
    world_a = _get_world("a")
    sim_a, n = _make_sim(world_a, solver, _spawn_dam)
    _run_substeps(sim_a, n, gather)
    graph_a = sim_a._graph_gather_only if gather else sim_a._graph_full_sort
    assert graph_a is not None, "graph path never captured a graph"
    pos_a = cp.asnumpy(world_a.position[:n])[:, :3]

    pos_b = _run_direct(_get_world("b"), solver, gather, n)
    assert np.isfinite(pos_a).all() and np.isfinite(pos_b).all()
    scale = float(np.abs(pos_b).max())
    err = float(np.abs(pos_a - pos_b).max())
    tol = 1e-5 * scale
    label = f"{solver} {'gather' if gather else 'full'}"
    if solver == "DFSPH" and not gather:
        # After a full sort, K_DFSPH_DensityAlpha reads last substep's
        # sorted_density in the new sort order (known open issue, bead
        # 1-pbf-dfsph-density-prev). The within-cell order is atomic-driven,
        # so two direct runs already differ by about 1e-4 m after 2 substeps.
        # Compare against that run-to-run noise instead of 1e-5.
        pos_c = _run_direct(_get_world("c"), solver, gather, n)
        noise = float(np.abs(pos_b - pos_c).max())
        print(f"\n  {label}: direct run-to-run noise {noise:.3g} m")
        tol = max(tol, 5.0 * noise)
    print(
        f"\n  {label}: max |graph - direct| {err:.3g} m (scale {scale:.3g} m, tol {tol:.3g} m)"
    )
    assert err <= tol, (
        f"{label}: graph replay differs from direct launches by {err:.3g} m"
    )


# ---------------------------------------------------------------------------
# 4. PBF and DFSPH wake sleeping SAND near a moving particle
# ---------------------------------------------------------------------------

# Sleepers sit 0.05 m apart (more than h), so no two interact and nothing
# but the wake pass can wake them. Gravity is off, so they never move.
_SLEEPER_Y = 0.02
# The 0.015 offset keeps every particle off a cell boundary (grid_min = -1,
# cell size h), so the host-side cell check below matches the kernel.
_SLEEPER_XZ = -0.185 + 0.05 * np.arange(8)
_MOVER = (0.015, 0.07, 0.015)  # 0.05 m above the nearest sleeper, one cell up


def _spawn_sleepers_and_mover(world):
    xs, zs = np.meshgrid(_SLEEPER_XZ, _SLEEPER_XZ, indexing="ij")
    count = 0
    for x, z in zip(xs.ravel(), zs.ravel()):
        c = (float(x), _SLEEPER_Y, float(z))
        count += world.spawn_cube(
            tuple(v - 0.001 for v in c),
            tuple(v + 0.001 for v in c),
            SAND,
            spacing=0.002,
        )
    assert count == xs.size
    k = count
    world.packed_info[:k] |= cp.uint32(_SLEEP_BIT)
    world.sleep_counter[:k] = 200
    mv = world.spawn_cube(
        tuple(v - 0.001 for v in _MOVER),
        tuple(v + 0.001 for v in _MOVER),
        SAND,
        spacing=0.002,
        velocity=(0.0, -0.1, 0.0),
    )
    assert mv == 1
    return k + 1


def _cell(p, grid_params):
    """calcGridCell in float32, as the kernels compute it."""
    gmin = np.asarray(grid_params[0]["grid_min"], dtype=np.float32)
    delta = np.asarray(grid_params[0]["grid_delta"], dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    return np.floor((p - gmin) * delta).astype(np.int64)


@pytest.mark.parametrize("solver", ["PBF", "DFSPH"])
def test_solver_wakes_sleeping_sand(solver):
    """After 3 substeps the sleepers in the mover's 27-cell stencil are awake
    with a reset sleep_counter; the rest still sleep with their counter kept."""
    world = _get_world("a")
    sim, n = _make_sim(world, solver, _spawn_sleepers_and_mover)
    sim.set_gravity(0.0)
    k = n - 1
    pos0 = cp.asnumpy(world.position[:n])[:, :3]
    for _ in range(3):
        sim._sim_step(n)

    pi = cp.asnumpy(world.packed_info[:n])
    sc = cp.asnumpy(world.sleep_counter[:n])
    pos = cp.asnumpy(world.position[:n])[:, :3]

    gp = sim._grid_params
    mover_cell = _cell(pos[k], gp)
    assert (mover_cell == _cell(pos0[k], gp)).all(), "mover left its cell"
    assert (pos[:k] == pos0[:k]).all(), "a sleeper moved"
    in_stencil = np.all(np.abs(_cell(pos0[:k], gp) - mover_cell) <= 1, axis=1)
    assert in_stencil.sum() >= 2 and (~in_stencil).sum() >= 10

    asleep = (pi[:k] & _SLEEP_BIT) != 0
    print(
        f"\n  {solver}: {in_stencil.sum()} sleepers in stencil, "
        f"awake there {(~asleep[in_stencil]).sum()}, "
        f"awake elsewhere {(~asleep[~in_stencil]).sum()}"
    )
    assert not asleep[in_stencil].any(), f"{solver}: impact-cell sand still asleep"
    # Woken on substep 1 (counter 0), then counted up by at most one per
    # later substep while at rest.
    assert (sc[:k][in_stencil] <= 3).all(), (
        f"{solver}: sleep_counter not reset in the impact cells: {sc[:k][in_stencil]}"
    )
    assert asleep[~in_stencil].all(), f"{solver}: far sand woke"
    assert (sc[:k][~in_stencil] == 200).all()


# ---------------------------------------------------------------------------
# 5. Foam: a splash makes foam; compaction removes expired foam
# ---------------------------------------------------------------------------


def _spawn_splash(world):
    # A water block thrown at the floor (y = -1) at 3 m/s.
    return world.spawn_cube(
        (-0.1, -0.95, -0.1),
        (0.1, -0.75, 0.1),
        WATER,
        spacing=0.02,
        velocity=(0.0, -3.0, 0.0),
    )


def test_foam_splash_then_compaction():
    world = _get_world("a")
    sim, n = _make_sim(world, "WCSPH", _spawn_splash)
    world.foam_enabled = True
    for _ in range(60):
        sim._sim_step(n)
    count_splash = int(world.foam_count[0])
    print(f"\n  foam after 60 substeps: {count_splash}")
    assert count_splash > 0, "splash made no foam"

    # Expire every foam particle: lifetime (vel.w) below one dt, so the next
    # K_FoamPhysics marks it dead. The substep does not generate new foam.
    live = count_splash
    world.foam_velocity[:live, 3] = cp.minimum(
        world.foam_velocity[:live, 3], cp.float32(0.5 * sim.dt)
    )
    sim._sim_step(n, last_substep=False)
    # _maybe_compact_foam runs on frames where _frame_counter % 8 == 0.
    sim._frame_counter = 8
    sim._maybe_compact_foam()
    count_after = int(world.foam_count[0])
    print(f"  foam after expiry and compaction: {count_after}")
    assert count_after == 0, f"{count_after} expired foam particles survived compaction"
