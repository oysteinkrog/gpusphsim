"""Graph capture waits until n has held for a frame (bd-r5-review-n28.204).

A frame that paints or spawns particles changes n. The graphs are keyed on
the exact n, so a capture on such a frame is thrown away on the next one.
step_frame now allows a capture only when the frame-start n matches the
previous frame's, so painting runs direct launches and never instantiates.
Once spawning stops, the first stable frame captures each slot it uses once.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

SUBSTEPS_PER_FRAME = 4
DT = 0.001


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self):
        # A little over 4 substeps per frame so rounding never drops one.
        self.t += DT * SUBSTEPS_PER_FRAME * 1.01


def _make_sim():
    from materials import WATER
    from simulation import Simulation
    from world import World

    world = World(40_000)
    world.spawn_cube((-0.2, -0.3, -0.2), (0.2, -0.1, 0.2), WATER, 0.02)
    clock = _FakeClock()
    sim = Simulation(world, dt=DT, fixed_dt=True, world_half_size=0.4,
                     compact_interval=0, clock=clock)
    sim._spawn_substep = sim._damping_duration
    sim.max_substeps = SUBSTEPS_PER_FRAME

    captures: list = []
    real_capture = sim._capture_graph

    def counting_capture(n, gather_only):
        key = (gather_only, sim._reorder_skip, sim._integrate_in_place)
        captures.append((sim._frame_counter, key, n))
        return real_capture(n, gather_only)
    sim._capture_graph = counting_capture
    return world, sim, clock, captures


def _frame(sim, clock):
    clock.tick()
    return sim.step_frame()


def test_no_capture_while_spawning_then_one_per_slot():
    from materials import WATER

    world, sim, clock, captures = _make_sim()
    sim.step_frame()  # first call only latches the clock

    # Spawn 100 particles before each of 10 frames: n changes every frame.
    for _ in range(10):
        n_before = world._high_water
        assert world.spawn_sphere((0.0, 0.2, 0.0), 0.05, WATER, 100) == 100
        assert world._high_water == n_before + 100
        assert _frame(sim, clock) == SUBSTEPS_PER_FRAME
    assert captures == [], f"captured while spawning: {captures}"

    # Stop spawning. The first stable frame captures each slot it uses.
    stable_start = sim._frame_counter
    for _ in range(2):
        assert _frame(sim, clock) == SUBSTEPS_PER_FRAME
    first_two = [c for c in captures if c[0] >= stable_start]
    assert first_two, "no capture in the first two stable frames"
    n = world._high_water
    assert all(c[2] == n for c in captures)

    # Later stable frames replay: no slot is ever captured twice.
    for _ in range(8):
        _frame(sim, clock)
    per_slot = Counter(c[1] for c in captures)
    assert all(v == 1 for v in per_slot.values()), per_slot


def test_direct_sim_step_still_captures():
    """Direct _sim_step callers (no step_frame) keep the old behaviour."""
    world, sim, _clock, captures = _make_sim()
    n = world._high_water
    for _ in range(4):
        sim._sim_step(n)
    assert captures, "direct _sim_step never captured"
