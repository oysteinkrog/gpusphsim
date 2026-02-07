"""Preset scenes for Falling Sand 3D.

Each preset function clears the world and spawns a themed configuration.
Returns (total_spawned, spawner_config_or_None).

World domain: (-1,-1,-1) to (1,1,1). Spacing default=0.02.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

from world import World
from materials import (
    STONE, SAND, WATER, LAVA, ACID, METAL, ICE, FIRE,
)


def _clear_world(world: World) -> None:
    """Kill all particles and reset high water mark."""
    world.packed_info[:] = 0
    world._high_water = 0


def load_sand_castle(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Sand Castle: tall sand pile with water around the base.

    ~100K sand as a tall conical pile approximated by stacked cubes,
    ~30K water around the base.
    """
    _clear_world(world)
    total = 0

    # Sand pile: tall structure with wide base, narrowing upward
    # Main body - wide base (spacing=0.02 for ~100K total sand)
    n = world.spawn_cube(
        min_corner=(-0.5, -0.9, -0.5),
        max_corner=(0.5, -0.2, 0.5),
        material_id=SAND,
        spacing=0.02,
    )
    total += n

    # Narrower middle section
    n = world.spawn_cube(
        min_corner=(-0.3, -0.2, -0.3),
        max_corner=(0.3, 0.3, 0.3),
        material_id=SAND,
        spacing=0.02,
    )
    total += n

    # Tower top
    n = world.spawn_cube(
        min_corner=(-0.15, 0.3, -0.15),
        max_corner=(0.15, 0.7, 0.15),
        material_id=SAND,
        spacing=0.02,
    )
    total += n

    # Water around the base (4 slabs surrounding the pile)
    # Front
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, 0.4),
        max_corner=(0.8, -0.4, 0.8),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    # Back
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.8),
        max_corner=(0.8, -0.4, -0.4),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    # Left
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.4),
        max_corner=(-0.4, -0.4, 0.4),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    # Right
    n = world.spawn_cube(
        min_corner=(0.4, -0.9, -0.4),
        max_corner=(0.8, -0.4, 0.4),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    print(f"  Sand Castle: {total:,} particles")
    return total, None


def load_volcano(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Volcano: stone crater walls with lava pool inside, fire above.

    Stone ring/crater walls, ~50K lava inside, fire particles above.
    """
    _clear_world(world)
    total = 0

    # Stone crater walls - 4 thick walls forming a ring
    wall_thick = 0.12
    wall_spacing = 0.03

    # Front wall
    n = world.spawn_cube(
        min_corner=(-0.6, -0.9, 0.4),
        max_corner=(0.6, 0.1, 0.4 + wall_thick),
        material_id=STONE,
        spacing=wall_spacing,
    )
    total += n

    # Back wall
    n = world.spawn_cube(
        min_corner=(-0.6, -0.9, -0.4 - wall_thick),
        max_corner=(0.6, 0.1, -0.4),
        material_id=STONE,
        spacing=wall_spacing,
    )
    total += n

    # Left wall
    n = world.spawn_cube(
        min_corner=(-0.6 - wall_thick, -0.9, -0.4 - wall_thick),
        max_corner=(-0.6, 0.1, 0.4 + wall_thick),
        material_id=STONE,
        spacing=wall_spacing,
    )
    total += n

    # Right wall
    n = world.spawn_cube(
        min_corner=(0.6, -0.9, -0.4 - wall_thick),
        max_corner=(0.6 + wall_thick, 0.1, 0.4 + wall_thick),
        material_id=STONE,
        spacing=wall_spacing,
    )
    total += n

    # Lava pool inside the crater (~50K particles)
    n = world.spawn_cube(
        min_corner=(-0.55, -0.9, -0.35),
        max_corner=(0.55, 0.0, 0.35),
        material_id=LAVA,
        spacing=0.023,
    )
    total += n

    # Fire particles above the lava
    n = world.spawn_cube(
        min_corner=(-0.3, -0.1, -0.2),
        max_corner=(0.3, 0.3, 0.2),
        material_id=FIRE,
        spacing=0.04,
    )
    total += n

    print(f"  Volcano: {total:,} particles")
    return total, None


def load_dam_break(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Dam Break: water behind ice wall, lava heat source on one side.

    ~100K water held behind an ice wall. Lava on the opposite side provides
    heat to gradually melt the ice, releasing the water.
    """
    _clear_world(world)
    total = 0

    # Ice wall (thin, in the middle)
    n = world.spawn_cube(
        min_corner=(-0.05, -0.9, -0.8),
        max_corner=(0.05, 0.5, 0.8),
        material_id=ICE,
        spacing=0.025,
    )
    total += n

    # Water behind the ice wall (left side, large volume)
    n = world.spawn_cube(
        min_corner=(-0.9, -0.9, -0.8),
        max_corner=(-0.05, 0.5, 0.8),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    # Lava heat source on the right side (close to the ice wall)
    n = world.spawn_cube(
        min_corner=(0.05, -0.9, -0.3),
        max_corner=(0.35, -0.3, 0.3),
        material_id=LAVA,
        spacing=0.03,
    )
    total += n

    print(f"  Dam Break: {total:,} particles")
    return total, None


def load_acid_rain(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Acid Rain: metal pillars (STATIC) with acid dropping from the top.

    Metal pillars as static structures. Returns a spawner config dict so
    the main loop can periodically spawn acid particles at the top.
    """
    _clear_world(world)
    total = 0

    pillar_spacing = 0.025

    # Metal pillar 1 (center)
    n = world.spawn_cube(
        min_corner=(-0.08, -0.9, -0.08),
        max_corner=(0.08, 0.3, 0.08),
        material_id=METAL,
        spacing=pillar_spacing,
    )
    total += n

    # Metal pillar 2 (front-left)
    n = world.spawn_cube(
        min_corner=(-0.5, -0.9, 0.3),
        max_corner=(-0.34, 0.1, 0.46),
        material_id=METAL,
        spacing=pillar_spacing,
    )
    total += n

    # Metal pillar 3 (front-right)
    n = world.spawn_cube(
        min_corner=(0.34, -0.9, 0.3),
        max_corner=(0.5, 0.1, 0.46),
        material_id=METAL,
        spacing=pillar_spacing,
    )
    total += n

    # Metal pillar 4 (back-left)
    n = world.spawn_cube(
        min_corner=(-0.5, -0.9, -0.46),
        max_corner=(-0.34, 0.1, -0.3),
        material_id=METAL,
        spacing=pillar_spacing,
    )
    total += n

    # Metal pillar 5 (back-right)
    n = world.spawn_cube(
        min_corner=(0.34, -0.9, -0.46),
        max_corner=(0.5, 0.1, -0.3),
        material_id=METAL,
        spacing=pillar_spacing,
    )
    total += n

    # Stone floor slab
    n = world.spawn_cube(
        min_corner=(-0.8, -0.95, -0.8),
        max_corner=(0.8, -0.85, 0.8),
        material_id=STONE,
        spacing=0.04,
    )
    total += n

    # Initial acid batch at top
    n = world.spawn_cube(
        min_corner=(-0.6, 0.7, -0.6),
        max_corner=(0.6, 0.85, 0.6),
        material_id=ACID,
        spacing=0.04,
    )
    total += n

    print(f"  Acid Rain: {total:,} particles")

    # Spawner config: periodically spawn acid at top of world
    spawner = {
        "type": "acid_rain",
        "material_id": ACID,
        "min_corner": (-0.5, 0.8, -0.5),
        "max_corner": (0.5, 0.9, 0.5),
        "spacing": 0.06,
        "interval_frames": 30,  # spawn every 30 frames
    }

    return total, spawner


def load_water_drop(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Water Drop: suspended water block falling under gravity.

    ~75K water particles in a block suspended in the upper half of the
    domain, dropping onto a stone floor. Inspired by the parent project's
    simple water demo.

    NOTE: Water spacing must be <= 0.02 (the default particle spacing).
    Larger spacing produces SPH density below rest_density, giving negative
    Tait EOS pressure that causes particles to attract instead of repel.
    """
    _clear_world(world)
    total = 0

    # Water block suspended high up
    n = world.spawn_cube(
        min_corner=(-0.5, 0.1, -0.5),
        max_corner=(0.5, 0.7, 0.5),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    print(f"  Water Drop: {total:,} particles")
    return total, None


def load_water_max(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Water Max: fill the world with as many water particles as possible.

    Spawns a large water cube that fills up to max_particles. The cube
    extends across most of the domain, leaving a small margin from the walls.
    """
    _clear_world(world)

    n = world.spawn_cube(
        min_corner=(-0.9, -0.9, -0.9),
        max_corner=(0.9, 0.9, 0.9),
        material_id=WATER,
        spacing=0.02,
    )

    print(f"  Water Max: {n:,} particles")
    return n, None


# Registry of all presets for UI access
PRESETS = {
    "Sand Castle": load_sand_castle,
    "Volcano": load_volcano,
    "Dam Break": load_dam_break,
    "Acid Rain": load_acid_rain,
    "Water Drop": load_water_drop,
    "Water Max": load_water_max,
}
