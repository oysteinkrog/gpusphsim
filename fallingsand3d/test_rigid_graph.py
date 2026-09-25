"""CUDA graphs stay on with a rigid body present (bd-r5-review-n28.182).

_sim_step used to run every substep as direct launches whenever a body
existed. It now captures and replays the graphs with a body, and recaptures
when the body count or the prologue's max_displacement clear changes
(Simulation._rigid_graph_key).

Each case runs the same scene twice: once through the graphs and once with
capture turned off (every substep direct). A dynamic box drops into a water
pool, and the grid-reuse decision runs every 4 substeps so the gather graph
is used too. The only difference between the runs is float atomic order, so
the body state after 80 substeps must agree within the spread of two direct
runs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

SOLVERS = ["WCSPH", "PBF", "DFSPH"]
STEPS = 80
SUBSTEPS_PER_FRAME = 4


def _add_box(sim):
    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(position=(0.0, -0.03, 0.0),
                                      half_extents=(0.05, 0.05, 0.05), mass=0.5,
                                      velocity=(0.0, -1.0, 0.0))
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    sim.notify_scene_changed()


def _run(profile_name: str, graphs: bool, add_body_at: int = 0):
    from materials import WATER
    from simulation import Simulation
    from solver_profiles import PROFILES
    from world import World

    world = World(60_000)
    world.spawn_cube((-0.2, -0.3, -0.2), (0.2, -0.1, 0.2), WATER, 0.02)
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=0.4)
    sim.set_solver_profile(PROFILES[profile_name])
    sim._spawn_substep = sim._damping_duration
    rbm = sim.rigid_body_manager
    if add_body_at == 0:
        _add_box(sim)

    captures = []
    if graphs:
        real_capture = sim._capture_graph

        def counting_capture(n, gather_only):
            g = real_capture(n, gather_only)
            captures.append((gather_only, g is not None))
            return g
        sim._capture_graph = counting_capture
    else:
        def no_capture(n, gather_only):
            sim._run_graph_body(n)
            return None
        sim._capture_graph = no_capture

    # "S" full sort, "g" gather-only substep.
    seq = []
    for k in range(STEPS):
        if add_body_at and k == add_body_at:
            _add_box(sim)
        seq.append("g" if sim._sort_skip_next else "S")
        sim._sim_step(world._high_water)
        # The decision latches the newest displacement readback that has
        # landed, which depends on how far the host runs ahead of the GPU.
        # Syncing every substep makes it the same in both runs.
        cp.cuda.Device().synchronize()
        if (k + 1) % SUBSTEPS_PER_FRAME == 0:
            sim._frame_counter += 1
            sim._update_sort_skip_decision()
            # Let its readback copy land before the next substep writes.
            cp.cuda.Device().synchronize()
    body = rbm.d_rigid_bodies.view(cp.float32).reshape(-1, 24)[0].get()
    return body, captures, "".join(seq), sim


@pytest.mark.parametrize("profile_name", SOLVERS)
def test_rigid_body_graph_matches_direct(profile_name):
    body_g, captures, seq_g, sim_g = _run(profile_name, graphs=True)
    body_d, _, seq_d, _ = _run(profile_name, graphs=False)
    body_d2, _, _, _ = _run(profile_name, graphs=False)

    # The graphs were captured with the body present and then replayed.
    assert captures, "no graph was captured with a rigid body present"
    assert sim_g._graph_rigid_key == (1, True)
    assert sim_g._graph_full_sort is not None
    assert len(captures) <= 4, captures  # captured once per path, then replayed

    # The splash is chaotic: float atomic order alone moves the body between
    # two direct runs. The graph run must differ from a direct run by no
    # more than that, plus a small floor.
    def dist(a, b):
        return (float(np.linalg.norm(a[0:3] - b[0:3])),
                float(np.linalg.norm(a[8:11] - b[8:11])))
    dp_gd, dv_gd = dist(body_g, body_d)
    dp_dd, dv_dd = dist(body_d2, body_d)
    print(f"{profile_name}: {seq_g} / {seq_d} "
          f"graph-direct dpos {dp_gd:.2e} dvel {dv_gd:.2e}; "
          f"direct-direct dpos {dp_dd:.2e} dvel {dv_dd:.2e}; y {body_g[1]:.4f} {body_d[1]:.4f}")
    assert abs(body_d[1] - (-0.03)) > 0.005  # the box moved
    if profile_name == "WCSPH":
        # WCSPH repeats closely: both graph paths ran, the sort/gather
        # pattern is the same and the body agrees to direct-run noise.
        assert "g" in seq_g and "S" in seq_g, seq_g
        assert seq_g == seq_d, (seq_g, seq_d)
        assert dp_gd <= 4.0 * dp_dd + 1e-5, (dp_gd, dp_dd)
        assert dv_gd <= 4.0 * dv_dd + 1e-4, (dv_gd, dv_dd)
    else:
        # The PBF and DFSPH splashes are chaotic: two direct runs already
        # differ by millimetres and cross the grid-reuse threshold on
        # different substeps. Check the graph run stays in that spread.
        assert dp_gd <= max(4.0 * dp_dd, 0.02), (dp_gd, dp_dd)
        assert dv_gd <= max(4.0 * dv_dd, 0.2), (dv_gd, dv_dd)


def test_rigid_graph_key_tracks_body_count():
    from simulation import Simulation
    from world import World

    world = World(10_000)
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=0.4)
    assert sim._rigid_graph_key() == (0, False)
    rbm = sim.rigid_body_manager
    rbm.add_rigid_body_with_particles(position=(0.0, 0.0, 0.0),
                                      half_extents=(0.05, 0.05, 0.05), mass=0.5)
    rbm.finalize_boundary_data()
    rbm.upload(sim.get_all_modules())
    # Before injection the boundary block has no offset yet.
    assert sim._rigid_graph_key()[0] == 1
    sim._sim_step(world._high_water)
    assert sim._rigid_graph_key() == (1, True)


def test_body_added_while_graphs_replay():
    """A body added mid-run (graphs already captured without one) couples."""
    body_g, captures, seq_g, sim_g = _run("WCSPH", graphs=True, add_body_at=16)
    body_d, _, seq_d, _ = _run("WCSPH", graphs=False, add_body_at=16)
    body_d2, _, _, _ = _run("WCSPH", graphs=False, add_body_at=16)
    assert sim_g._graph_rigid_key == (1, True)
    assert seq_g == seq_d, (seq_g, seq_d)
    # Two direct runs already differ by 1e-5 to 5e-5 m (float atomic order
    # in the sort and the sums), so compare against that spread plus a floor.
    dp_gd = float(np.linalg.norm(body_g[0:3] - body_d[0:3]))
    dv_gd = float(np.linalg.norm(body_g[8:11] - body_d[8:11]))
    dp_dd = float(np.linalg.norm(body_d2[0:3] - body_d[0:3]))
    dv_dd = float(np.linalg.norm(body_d2[8:11] - body_d[8:11]))
    assert dp_gd <= 4.0 * dp_dd + 1e-4, (dp_gd, dp_dd)
    assert dv_gd <= 4.0 * dv_dd + 1e-3, (dv_gd, dv_dd)
    # The fluid slowed the box: free fall from -1 m/s would reach -1.26 m/s.
    assert body_d[9] > -1.2, body_d[8:11]
