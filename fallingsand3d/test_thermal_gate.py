"""Thermal gate for K_Step1 (bd-r5-review-n28.172).

c_sim.thermal_enabled = 0 makes K_Step1 skip the neighbour temperature load
and the heat and exposure sums. Simulation clears it only for a scene where
every spawned material starts at T_AMBIENT and no pair of them has a non-zero
interaction entry, where those sums are exactly 0 anyway.

Kernel checks: with the flag off, dTdt and both exposures are 0 and density
and pressure are bitwise the same as with the flag on. Host checks: the rule
for water only, water and stone, water and lava, sand and water, an empty
world and a replaced material set, and the flip to on when lava is spawned
into a running water scene.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

cp = pytest.importorskip("cupy")

import step1  # noqa: E402
from conftest import counting_sort_scene, default_grid_params  # noqa: E402
from materials import (  # noqa: E402
    ACID, FLUID, LAVA, METAL, SAND, STATIC, STONE, WATER,
    build_interaction_matrix, build_material_array,
)

H = 0.04
SPACING = 0.02
MASS = 0.008
C_SIM_FLAG_OFFSET = 72  # byte offset of thermal_enabled in SimParams


def _upload(thermal: bool) -> None:
    step1.upload_grid_params(default_grid_params())
    step1.upload_sim_params(step1.build_sim_params(
        smoothing_length=H, particle_mass=MASS, particle_spacing=SPACING,
        thermal_enabled=thermal))
    step1.upload_precalc_params(step1.build_precalc_params(smoothing_length=H))
    step1.upload_materials(build_material_array())
    step1.upload_interactions(build_interaction_matrix())


def _read_flag(module) -> int:
    buf = np.zeros(1, dtype=np.int32)
    d_sim = int(module.get_global("c_sim"))
    cp.cuda.runtime.memcpy(buf.ctypes.data, d_sim + C_SIM_FLAG_OFFSET, 4, 2)
    return int(buf[0])


def _hot_acid_scene():
    """7x7x7 METAL block with a 1000 K centre and one ACID layer on top."""
    coords = np.arange(-0.06, 0.06 + SPACING * 0.5, SPACING, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    n = xx.size
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, 0], pos[:, 1], pos[:, 2] = xx.ravel(), yy.ravel(), zz.ravel()
    packed = np.full(n, METAL | (FLUID << 8), dtype=np.uint32)
    packed[pos[:, 1] > 0.05] = ACID | (FLUID << 8)
    temp = np.full(n, 293.0, dtype=np.float32)
    temp[np.argmin((pos[:, :3] ** 2).sum(axis=1))] = 1000.0
    return pos, packed, temp


def _step1_outputs(pos, packed, temp, thermal: bool) -> dict:
    _upload(thermal)
    n = pos.shape[0]
    scene = counting_sort_scene(
        pos, velocity=np.zeros((n, 4), dtype=np.float32),
        mass=np.full(n, MASS, dtype=np.float32), packed_info=packed,
        temperature=temp, grid_params=default_grid_params())
    scene["position"][:, 3] = 0.0  # neighbour density falls back to rest density
    density, _, dTdt, exp_heat, exp_corrode, pressure = step1.compute_step1(
        scene["position"], scene["velocity"], scene["mass"],
        scene["packed_info"], scene["temperature"],
        scene["cell_start"], scene["cell_end"], velocity_h=scene["velocity_h"])
    cp.cuda.Device().synchronize()
    return {"density": density.get(), "pressure": pressure.get(),
            "dTdt": dTdt.get(), "exposure_heat": exp_heat.get(),
            "exposure_corrode": exp_corrode.get()}


def test_flag_off_zeroes_heat_and_exposure():
    pos, packed, temp = _hot_acid_scene()
    on = _step1_outputs(pos, packed, temp, thermal=True)
    off = _step1_outputs(pos, packed, temp, thermal=False)
    # The scene really exchanges heat and corrodes with the gate on.
    assert np.abs(on["dTdt"]).max() > 0.0
    assert on["exposure_heat"].max() > 0.0
    assert on["exposure_corrode"].max() > 0.0
    for key in ("dTdt", "exposure_heat", "exposure_corrode"):
        assert np.all(off[key] == 0.0), key
    # Density does not read position.w, so it is bitwise the same.
    for key in ("density", "pressure"):
        assert np.array_equal(on[key].view(np.uint32), off[key].view(np.uint32)), key


def test_flag_off_matches_on_for_cold_water():
    pos, _, _ = _hot_acid_scene()
    n = pos.shape[0]
    packed = np.full(n, WATER | (FLUID << 8), dtype=np.uint32)
    temp = np.full(n, 293.0, dtype=np.float32)
    on = _step1_outputs(pos, packed, temp, thermal=True)
    off = _step1_outputs(pos, packed, temp, thermal=False)
    for key in ("dTdt", "exposure_heat", "exposure_corrode"):
        assert np.all(on[key] == 0.0) and np.all(off[key] == 0.0), key
    for key in ("density", "pressure"):
        assert np.array_equal(on[key].view(np.uint32), off[key].view(np.uint32)), key


def _sim_with(materials):
    from simulation import Simulation
    from world import World

    world = World(40_000)
    x = -0.4
    for mat in materials:
        world.spawn_cube((x, -0.5, -0.1), (x + 0.12, -0.38, 0.1), mat, SPACING)
        x += 0.2
    sim = Simulation(world, dt=0.001, fixed_dt=True, world_half_size=0.5)
    return world, sim


@pytest.mark.parametrize("materials, expected", [
    ((WATER,), 0),
    ((WATER, STONE), 0),
    ((WATER, LAVA), 1),
    ((SAND, WATER), 1),
    ((), 1),
])
def test_host_rule(materials, expected):
    world, sim = _sim_with(materials)
    assert int(sim._sim_params[0]["thermal_enabled"]) == expected
    assert _read_flag(step1.get_module()) == expected


def test_replaced_material_set_keeps_gate_on():
    world, sim = _sim_with((WATER,))
    assert int(sim._sim_params[0]["thermal_enabled"]) == 0
    # A load or undo rebuilds the set from the restored particles.
    world._spawned_material_ids = set(world._spawned_material_ids)
    sim._update_thermal_gate()
    assert int(sim._sim_params[0]["thermal_enabled"]) == 1


def test_spawning_lava_turns_gate_on():
    world, sim = _sim_with((WATER,))
    for _ in range(3):
        sim._sim_step(world._high_water)
    cp.cuda.Device().synchronize()
    assert _read_flag(step1.get_module()) == 0
    assert float(cp.abs(world.sorted_dTdt).max()) == 0.0

    # Lava right next to the water block.
    world.spawn_cube((-0.26, -0.5, -0.1), (-0.14, -0.38, 0.1), LAVA, SPACING)
    sim.notify_scene_changed()
    for _ in range(3):
        sim._sim_step(world._high_water)
    cp.cuda.Device().synchronize()
    assert int(sim._sim_params[0]["thermal_enabled"]) == 1
    assert _read_flag(step1.get_module()) == 1
    assert float(cp.abs(world.sorted_dTdt).max()) > 0.0


def test_static_material_counts_in_rule():
    """A STATIC stone floor under water keeps the gate off; STONE is STATIC."""
    from materials import MATERIALS
    assert MATERIALS[STONE].behavior_class == STATIC
    world, sim = _sim_with((STONE,))
    assert int(sim._sim_params[0]["thermal_enabled"]) == 0
