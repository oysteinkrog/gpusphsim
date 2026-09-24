"""Tests for preset scenes: spawn boxes must not overlap (bd-r5-review-n28.28).

Two spawn boxes of different materials that share volume put two particle
lattices on top of each other. SPH density counts both, so the overlap zone
starts at about 2x rest density and the Tait EOS (gamma 7) blows it apart on
the first substep.

The overlap test is CPU-only: it runs every preset against a stub world that
records `spawn_cube` calls. The GPU smoke test steps the two presets fixed by
this bead once and checks the water density.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from presets import PRESETS, _check_no_overlap
from materials import MATERIALS, SAND, STONE, WATER

# Presets that still nest one spawn box inside another. Bead
# bd-r5-review-n28.47 (spawn_cube exclude_box filter) fixes these and must
# remove them from this set.
KNOWN_NESTED_SPAWNS = {"Lava Lamp", "Debris Flow", "SDF Hourglass"}


class RecordingWorld:
    """Stub World that records spawn_cube boxes instead of spawning particles."""

    def __init__(self) -> None:
        self.packed_info = np.zeros(1, dtype=np.uint32)
        self._high_water = 0
        self.boxes: list = []
        self.sdf_manager = MagicMock()
        self.rigid_body_manager = MagicMock()

    def spawn_cube(self, min_corner, max_corner, material_id, spacing=0.02,
                   velocity=(0.0, 0.0, 0.0)) -> int:
        self.boxes.append((tuple(min_corner), tuple(max_corner), material_id))
        return 0

    # Non-box spawners (bowl, ramp, wall with gap) are not axis-aligned boxes.
    # They are not recorded.
    def spawn_bowl(self, *args, **kwargs) -> int:
        return 0

    def spawn_ramp(self, *args, **kwargs) -> int:
        return 0

    def spawn_wall_with_gap(self, *args, **kwargs) -> int:
        return 0


def _record(name: str) -> list:
    world = RecordingWorld()
    PRESETS[name](world)
    return world.boxes


# ---------------------------------------------------------------------------
# _check_no_overlap helper
# ---------------------------------------------------------------------------

def test_check_no_overlap_detects_shared_volume():
    boxes = [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), SAND),
        ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5), WATER),
    ]
    assert _check_no_overlap(boxes) == [(0, 1)]


def test_check_no_overlap_ignores_touching_faces():
    boxes = [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), SAND),
        ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0), WATER),
    ]
    assert _check_no_overlap(boxes) == []


def test_check_no_overlap_ignores_same_material():
    boxes = [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), SAND),
        ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5), SAND),
    ]
    assert _check_no_overlap(boxes) == []


def test_check_no_overlap_needs_overlap_on_all_axes():
    boxes = [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), STONE),
        ((0.5, 0.5, 2.0), (1.5, 1.5, 3.0), WATER),
    ]
    assert _check_no_overlap(boxes) == []


# ---------------------------------------------------------------------------
# Every preset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_spawn_boxes_do_not_overlap(name):
    if name in KNOWN_NESTED_SPAWNS:
        pytest.skip("nested spawn fixed by bd-r5-review-n28.47")
    boxes = _record(name)
    pairs = _check_no_overlap(boxes)
    assert pairs == [], (
        f"{name}: overlapping spawn boxes "
        + ", ".join(f"{boxes[i]} x {boxes[j]}" for i, j in pairs)
    )


@pytest.mark.parametrize("name", ["Sand Castle", "Waterfall"])
def test_fixed_presets_record_boxes(name):
    """Guard against the stub silently recording nothing for the fixed presets."""
    assert len(_record(name)) >= 5


def test_known_nested_list_is_still_needed():
    """Fails once bd-r5-review-n28.47 lands, as a reminder to empty the set."""
    for name in KNOWN_NESTED_SPAWNS:
        assert name in PRESETS
        assert _check_no_overlap(_record(name)) != [], (
            f"{name} no longer overlaps: remove it from KNOWN_NESTED_SPAWNS"
        )


# ---------------------------------------------------------------------------
# GPU smoke test: water density after one substep
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_sim():
    cp = pytest.importorskip("cupy")
    from world import World
    from simulation import Simulation

    world = World(max_particles=500_000)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=True, max_substeps=20)
    yield cp, world, sim
    sim.close()


@pytest.mark.parametrize("name", ["Sand Castle", "Waterfall"])
def test_fixed_preset_water_density_after_one_substep(gpu_sim, name):
    cp, world, sim = gpu_sim
    PRESETS[name](world)
    sim.notify_scene_changed()
    n = world._high_water
    sim._sim_step(n)
    cp.cuda.Device().synchronize()

    mat = world.sorted_packed_info[:n].get() & 0xFF
    rho = world.sorted_density[:n].get()
    water = mat == WATER
    assert water.sum() > 10_000
    rho0 = MATERIALS[WATER].rest_density
    ratio = float(rho[water].max()) / rho0
    assert ratio < 1.3, f"{name}: max water density {ratio:.3f}x rest density"
