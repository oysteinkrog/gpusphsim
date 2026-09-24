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
from materials import MATERIALS, MUD, SAND, STONE, WATER


class RecordingWorld:
    """Stub World that records spawn_cube boxes instead of spawning particles."""

    def __init__(self) -> None:
        self.packed_info = np.zeros(1, dtype=np.uint32)
        self._high_water = 0
        self.boxes: list = []
        self.sdf_manager = MagicMock()
        self.rigid_body_manager = MagicMock()

    def spawn_cube(self, min_corner, max_corner, material_id, spacing=0.02,
                   velocity=(0.0, 0.0, 0.0), exclude_box=None,
                   inside_fn=None) -> int:
        self.boxes.append((tuple(min_corner), tuple(max_corner), material_id,
                           exclude_box))
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


def test_check_no_overlap_allows_nested_box_in_exclude_box():
    inner = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))
    boxes = [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), WATER, inner),
        (inner[0], inner[1], SAND),
    ]
    assert _check_no_overlap(boxes) == []


def test_check_no_overlap_flags_box_larger_than_exclude_box():
    boxes = [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), WATER,
         ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))),
        ((0.25, 0.25, 0.25), (0.9, 0.75, 0.75), SAND),
    ]
    assert _check_no_overlap(boxes) == [(0, 1)]


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


@pytest.mark.parametrize("name", ["Lava Lamp", "Debris Flow", "SDF Hourglass"])
def test_nested_presets_use_exclude_box(name):
    """These presets put one box inside another and rely on exclude_box
    (bd-r5-review-n28.47). Without it the overlap check would flag them."""
    boxes = _record(name)
    assert any(len(b) > 3 and b[3] is not None for b in boxes)
    stripped = [b[:3] for b in boxes]
    assert _check_no_overlap(stripped) != []


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


# ---------------------------------------------------------------------------
# Spawn density of the nested presets (bd-r5-review-n28.47)
# ---------------------------------------------------------------------------

def _poly6_density(pos: np.ndarray, mass: float, h: float) -> np.ndarray:
    """SPH poly6 density of every point over all points, on the CPU."""
    from scipy.spatial import cKDTree
    coeff = 315.0 / (64.0 * np.pi * h ** 9)
    pairs = cKDTree(pos).query_pairs(h, output_type="ndarray")
    r2 = ((pos[pairs[:, 0]] - pos[pairs[:, 1]]) ** 2).sum(axis=1)
    w = coeff * (h * h - r2) ** 3
    rho = np.full(len(pos), coeff * h ** 6)
    np.add.at(rho, pairs[:, 0], w)
    np.add.at(rho, pairs[:, 1], w)
    return rho * mass


@pytest.mark.parametrize("name,material", [
    ("Lava Lamp", WATER),
    ("Debris Flow", MUD),
    ("SDF Hourglass", SAND),
])
def test_nested_preset_spawn_density(name, material):
    """The outer material must not start with a second lattice inside it.

    Before the exclude_box fix the outer lattice filled the inner box too,
    so its peak spawn density was 1.5x to 2.0x its own median. The check uses
    the spawn positions (h = 0.04, poly6, all neighbours) rather than a GPU
    substep, so it measures the lattice and nothing else.
    """
    pytest.importorskip("scipy")
    pytest.importorskip("cupy")
    from world import World, PARTICLE_MASS
    world = World(max_particles=500_000)
    world.sdf_manager = MagicMock()
    world.rigid_body_manager = MagicMock()
    PRESETS[name](world)
    n = world._high_water
    pos = world.position[:n, :3].get().astype(np.float64)
    mat = world.packed_info[:n].get() & 0xFF
    rho = _poly6_density(pos, PARTICLE_MASS, 0.04)[mat == material]
    assert len(rho) > 1_000
    peak = float(rho.max())
    assert peak / float(np.median(rho)) < 1.4, (
        f"{name}: {MATERIALS[material].name} peak {peak:.0f} is "
        f"{peak / np.median(rho):.2f}x its median")
    assert peak / MATERIALS[material].rest_density < 1.3
