"""Preset scenes for Falling Sand 3D.

Each preset function clears the world and spawns a themed configuration.
Returns (total_spawned, spawner_config_or_None).

World domain: (-1,-1,-1) to (1,1,1). Spacing default=0.02.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

from world import World
from materials import (
    STONE, SAND, DIRT, GRAVEL, WATER, OIL, LAVA, ACID,
    WOOD, METAL, ICE, FIRE, GUNPOWDER, SMOKE, WET_SAND, MUD,
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


def load_lava_meets_ice(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Lava Meets Ice: head-on collision between lava and ice.

    Lava wall on the left, ice wall on the right, with a gap between.
    Heat transfer melts the ice and cools the lava to stone at the contact zone.
    """
    _clear_world(world)
    total = 0

    # Lava block (left side)
    n = world.spawn_cube(
        min_corner=(-0.9, -0.9, -0.6),
        max_corner=(-0.1, 0.3, 0.6),
        material_id=LAVA,
        spacing=0.023,
    )
    total += n

    # Ice block (right side)
    n = world.spawn_cube(
        min_corner=(0.1, -0.9, -0.6),
        max_corner=(0.9, 0.3, 0.6),
        material_id=ICE,
        spacing=0.023,
    )
    total += n

    print(f"  Lava Meets Ice: {total:,} particles")
    return total, None


def load_oil_fire(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Oil Fire: oil pool with fire dropped on top.

    Shallow oil lake at the bottom with fire particles falling in from above.
    Oil ignites and burns, producing smoke. Demonstrates combustion chain reaction.
    """
    _clear_world(world)
    total = 0

    # Oil pool (wide, shallow)
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.8),
        max_corner=(0.8, -0.4, 0.8),
        material_id=OIL,
        spacing=0.022,
    )
    total += n

    # Fire dropped from above (small cluster)
    n = world.spawn_cube(
        min_corner=(-0.15, 0.3, -0.15),
        max_corner=(0.15, 0.6, 0.15),
        material_id=FIRE,
        spacing=0.035,
    )
    total += n

    print(f"  Oil Fire: {total:,} particles")
    return total, None


def load_waterfall(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Waterfall: water pouring over stone steps into a pool.

    Stone steps on one side, water flowing over them. Demonstrates
    fluid-solid interaction and SSFR rendering of both materials.
    """
    _clear_world(world)
    total = 0

    # Stone steps (3 tiers descending left to right)
    # Top step
    n = world.spawn_cube(
        min_corner=(-0.9, -0.1, -0.7),
        max_corner=(-0.3, 0.05, 0.7),
        material_id=STONE,
        spacing=0.03,
    )
    total += n

    # Middle step
    n = world.spawn_cube(
        min_corner=(-0.3, -0.45, -0.7),
        max_corner=(0.2, -0.3, 0.7),
        material_id=STONE,
        spacing=0.03,
    )
    total += n

    # Bottom step
    n = world.spawn_cube(
        min_corner=(0.2, -0.75, -0.7),
        max_corner=(0.9, -0.6, 0.7),
        material_id=STONE,
        spacing=0.03,
    )
    total += n

    # Water on the top step
    n = world.spawn_cube(
        min_corner=(-0.85, 0.05, -0.6),
        max_corner=(-0.35, 0.5, 0.6),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    print(f"  Waterfall: {total:,} particles")
    return total, None


def load_gunpowder_trail(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Gunpowder Trail: a trail of gunpowder leading to a pile, with fire at one end.

    Fire ignites the trail, which burns along its length and detonates the pile.
    Demonstrates chain-reaction combustion.
    """
    _clear_world(world)
    total = 0

    # Gunpowder trail (thin line along the ground)
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.05),
        max_corner=(0.3, -0.82, 0.05),
        material_id=GUNPOWDER,
        spacing=0.02,
    )
    total += n

    # Gunpowder pile at the end of the trail
    n = world.spawn_cube(
        min_corner=(0.3, -0.9, -0.25),
        max_corner=(0.7, -0.5, 0.25),
        material_id=GUNPOWDER,
        spacing=0.02,
    )
    total += n

    # Sand walls flanking the pile (to see the blast push them)
    n = world.spawn_cube(
        min_corner=(0.25, -0.9, -0.55),
        max_corner=(0.75, -0.2, -0.30),
        material_id=SAND,
        spacing=0.022,
    )
    total += n
    n = world.spawn_cube(
        min_corner=(0.25, -0.9, 0.30),
        max_corner=(0.75, -0.2, 0.55),
        material_id=SAND,
        spacing=0.022,
    )
    total += n

    # Fire at the start of the trail
    n = world.spawn_cube(
        min_corner=(-0.85, -0.85, -0.1),
        max_corner=(-0.7, -0.6, 0.1),
        material_id=FIRE,
        spacing=0.03,
    )
    total += n

    print(f"  Gunpowder Trail: {total:,} particles")
    return total, None


def load_hourglass(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Hourglass: sand pouring through a narrow gap between stone slabs.

    Two stone platforms with a narrow gap, sand on top flowing down.
    Shows granular flow and pile formation.
    """
    _clear_world(world)
    total = 0

    # Stone shelf (left half)
    n = world.spawn_cube(
        min_corner=(-0.9, -0.05, -0.7),
        max_corner=(-0.06, 0.08, 0.7),
        material_id=STONE,
        spacing=0.03,
    )
    total += n

    # Stone shelf (right half) — gap in the middle
    n = world.spawn_cube(
        min_corner=(0.06, -0.05, -0.7),
        max_corner=(0.9, 0.08, 0.7),
        material_id=STONE,
        spacing=0.03,
    )
    total += n

    # Sand on top of the shelf
    n = world.spawn_cube(
        min_corner=(-0.85, 0.08, -0.65),
        max_corner=(0.85, 0.7, 0.65),
        material_id=SAND,
        spacing=0.02,
    )
    total += n

    print(f"  Hourglass: {total:,} particles")
    return total, None


def load_lava_lamp(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Lava Lamp: water with lava blobs rising and falling via buoyancy.

    Water fills the domain. Small lava blobs at the bottom heat up and
    rise (thermal expansion), cool at the top, and sink back down.
    Demonstrates Boussinesq buoyancy and thermal convection.
    """
    _clear_world(world)
    total = 0

    # Water filling most of the domain
    n = world.spawn_cube(
        min_corner=(-0.7, -0.9, -0.7),
        max_corner=(0.7, 0.6, 0.7),
        material_id=WATER,
        spacing=0.022,
    )
    total += n

    # Lava blob at the bottom center
    n = world.spawn_cube(
        min_corner=(-0.25, -0.85, -0.25),
        max_corner=(0.25, -0.5, 0.25),
        material_id=LAVA,
        spacing=0.022,
    )
    total += n

    print(f"  Lava Lamp: {total:,} particles")
    return total, None


def load_erosion(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Erosion: water flowing over a dirt hillside.

    Dirt terrain on one side with water above it. Water erodes and
    carries dirt particles. Shows fluid-granular interaction.
    """
    _clear_world(world)
    total = 0

    # Dirt hill (sloped via stacked layers decreasing in width)
    # Base layer
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.7),
        max_corner=(0.5, -0.5, 0.7),
        material_id=DIRT,
        spacing=0.023,
    )
    total += n

    # Middle layer
    n = world.spawn_cube(
        min_corner=(-0.8, -0.5, -0.7),
        max_corner=(0.1, -0.1, 0.7),
        material_id=DIRT,
        spacing=0.023,
    )
    total += n

    # Top layer
    n = world.spawn_cube(
        min_corner=(-0.8, -0.1, -0.7),
        max_corner=(-0.3, 0.2, 0.7),
        material_id=DIRT,
        spacing=0.023,
    )
    total += n

    # Water source above the hilltop
    n = world.spawn_cube(
        min_corner=(-0.75, 0.2, -0.5),
        max_corner=(-0.35, 0.7, 0.5),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    print(f"  Erosion: {total:,} particles")
    return total, None


def load_forge(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Forge: metal block heated by lava, quenched by water.

    Metal pillar in the center, lava on one side heating it,
    water on the other side cooling it. Demonstrates three-way heat transfer.
    """
    _clear_world(world)
    total = 0

    # Metal block (center)
    n = world.spawn_cube(
        min_corner=(-0.15, -0.9, -0.5),
        max_corner=(0.15, 0.2, 0.5),
        material_id=METAL,
        spacing=0.025,
    )
    total += n

    # Lava pool (left side, touching metal)
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.5),
        max_corner=(-0.15, 0.0, 0.5),
        material_id=LAVA,
        spacing=0.025,
    )
    total += n

    # Water pool (right side, touching metal)
    n = world.spawn_cube(
        min_corner=(0.15, -0.9, -0.5),
        max_corner=(0.8, 0.0, 0.5),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    print(f"  Forge: {total:,} particles")
    return total, None


def load_avalanche(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Avalanche: gravel and sand sliding down a stone ramp onto ice.

    Steep stone ramp with gravel and sand piled at the top. When released,
    the granular mass slides down and hits a frozen lake below.
    """
    _clear_world(world)
    total = 0

    # Stone ramp (angled slab)
    n = world.spawn_cube(
        min_corner=(-0.9, -0.2, -0.7),
        max_corner=(0.0, -0.05, 0.7),
        material_id=STONE,
        spacing=0.03,
    )
    total += n

    # Gravel on top of the ramp
    n = world.spawn_cube(
        min_corner=(-0.85, -0.05, -0.6),
        max_corner=(-0.2, 0.5, 0.6),
        material_id=GRAVEL,
        spacing=0.022,
    )
    total += n

    # Sand layer on top of gravel
    n = world.spawn_cube(
        min_corner=(-0.75, 0.5, -0.5),
        max_corner=(-0.25, 0.8, 0.5),
        material_id=SAND,
        spacing=0.022,
    )
    total += n

    # Frozen lake (ice floor on the right)
    n = world.spawn_cube(
        min_corner=(0.0, -0.9, -0.7),
        max_corner=(0.9, -0.7, 0.7),
        material_id=ICE,
        spacing=0.025,
    )
    total += n

    # Water on the ice
    n = world.spawn_cube(
        min_corner=(0.0, -0.7, -0.6),
        max_corner=(0.85, -0.4, 0.6),
        material_id=WATER,
        spacing=0.022,
    )
    total += n

    print(f"  Avalanche: {total:,} particles")
    return total, None


def load_acid_bath(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Acid Bath: wood and metal objects dissolving in an acid pool.

    Acid fills the bottom, wood blocks float on top, metal sinks and corrodes.
    """
    _clear_world(world)
    total = 0

    # Acid pool
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.8),
        max_corner=(0.8, -0.2, 0.8),
        material_id=ACID,
        spacing=0.022,
    )
    total += n

    # Wood block (will float)
    n = world.spawn_cube(
        min_corner=(-0.5, 0.0, -0.2),
        max_corner=(-0.1, 0.3, 0.2),
        material_id=WOOD,
        spacing=0.025,
    )
    total += n

    # Metal block (will sink and corrode)
    n = world.spawn_cube(
        min_corner=(0.1, 0.0, -0.2),
        max_corner=(0.5, 0.3, 0.2),
        material_id=METAL,
        spacing=0.025,
    )
    total += n

    print(f"  Acid Bath: {total:,} particles")
    return total, None


def load_multi_fluid(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Multi-Fluid: water, oil, and acid layered by density.

    Three fluids stacked vertically. Oil (lighter) should float,
    acid and water settle below. Demonstrates fluid stratification.
    """
    _clear_world(world)
    total = 0

    # Water (bottom layer)
    n = world.spawn_cube(
        min_corner=(-0.7, -0.9, -0.7),
        max_corner=(0.7, -0.3, 0.7),
        material_id=WATER,
        spacing=0.022,
    )
    total += n

    # Oil (middle — should float up)
    n = world.spawn_cube(
        min_corner=(-0.7, -0.3, -0.7),
        max_corner=(0.7, 0.1, 0.7),
        material_id=OIL,
        spacing=0.022,
    )
    total += n

    # Acid (top — denser, should sink)
    n = world.spawn_cube(
        min_corner=(-0.7, 0.1, -0.7),
        max_corner=(0.7, 0.5, 0.7),
        material_id=ACID,
        spacing=0.022,
    )
    total += n

    print(f"  Multi-Fluid: {total:,} particles")
    return total, None


def load_fire_and_ice(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Fire and Ice: fire raining down on a large ice block.

    Ice platform fills the bottom. Fire spawns continuously from above.
    Ice melts into water, water boils into steam from the fire's heat.
    Triple phase transition: ice → water → steam.
    """
    _clear_world(world)
    total = 0

    # Large ice block
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.8),
        max_corner=(0.8, 0.0, 0.8),
        material_id=ICE,
        spacing=0.023,
    )
    total += n

    # Initial fire batch above
    n = world.spawn_cube(
        min_corner=(-0.5, 0.3, -0.5),
        max_corner=(0.5, 0.7, 0.5),
        material_id=FIRE,
        spacing=0.04,
    )
    total += n

    print(f"  Fire and Ice: {total:,} particles")

    # Spawner: continuous fire rain
    spawner = {
        "type": "fire_rain",
        "material_id": FIRE,
        "min_corner": (-0.4, 0.7, -0.4),
        "max_corner": (0.4, 0.85, 0.4),
        "spacing": 0.06,
        "interval_frames": 20,
    }

    return total, spawner


def load_sandbox(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Sandbox: a variety of materials placed for free experimentation.

    Small blocks of each material type arranged in a grid. Use the brush
    to push them together and see what happens!
    """
    _clear_world(world)
    total = 0

    # Layout: 4x3 grid of material cubes, slightly elevated
    materials_grid = [
        (WATER, -0.6, -0.4),   (OIL, -0.2, -0.4),    (LAVA, 0.2, -0.4),   (ACID, 0.6, -0.4),
        (SAND, -0.6, 0.0),     (DIRT, -0.2, 0.0),     (GRAVEL, 0.2, 0.0),  (ICE, 0.6, 0.0),
        (WOOD, -0.6, 0.4),     (METAL, -0.2, 0.4),    (FIRE, 0.2, 0.4),    (GUNPOWDER, 0.6, 0.4),
    ]

    size = 0.14  # half-size of each cube
    for mat_id, cx, cz in materials_grid:
        sp = 0.03 if mat_id in (FIRE, SMOKE) else 0.025
        n = world.spawn_cube(
            min_corner=(cx - size, -0.9, cz - size),
            max_corner=(cx + size, -0.9 + size * 3, cz + size),
            material_id=mat_id,
            spacing=sp,
        )
        total += n

    print(f"  Sandbox: {total:,} particles")
    return total, None


def load_mudslide(world: World) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Mudslide: sand hillside with water pouring from above.

    Demonstrates SAND -> WET_SAND -> MUD transitions. Water pours onto
    a sand slope; contact zone darkens (WET_SAND), saturated zone flows
    as brown slurry (MUD), edges dry back to tan sand.
    """
    _clear_world(world)
    total = 0

    # Sand hillside (stacked layers for slope)
    # Base layer (wide)
    n = world.spawn_cube(
        min_corner=(-0.8, -0.9, -0.7),
        max_corner=(0.8, -0.3, 0.7),
        material_id=SAND,
        spacing=0.02,
    )
    total += n

    # Middle layer (narrower)
    n = world.spawn_cube(
        min_corner=(-0.6, -0.3, -0.5),
        max_corner=(0.3, 0.1, 0.5),
        material_id=SAND,
        spacing=0.02,
    )
    total += n

    # Top layer (narrow peak)
    n = world.spawn_cube(
        min_corner=(-0.4, 0.1, -0.3),
        max_corner=(0.0, 0.4, 0.3),
        material_id=SAND,
        spacing=0.02,
    )
    total += n

    # Water block above the hill peak
    n = world.spawn_cube(
        min_corner=(-0.3, 0.5, -0.25),
        max_corner=(0.1, 0.9, 0.25),
        material_id=WATER,
        spacing=0.02,
    )
    total += n

    print(f"  Mudslide: {total:,} particles")
    return total, None


# Registry of all presets for UI access
PRESETS = {
    "Sand Castle": load_sand_castle,
    "Volcano": load_volcano,
    "Dam Break": load_dam_break,
    "Acid Rain": load_acid_rain,
    "Water Drop": load_water_drop,
    "Water Max": load_water_max,
    "Lava Meets Ice": load_lava_meets_ice,
    "Oil Fire": load_oil_fire,
    "Waterfall": load_waterfall,
    "Gunpowder Trail": load_gunpowder_trail,
    "Hourglass": load_hourglass,
    "Lava Lamp": load_lava_lamp,
    "Erosion": load_erosion,
    "Forge": load_forge,
    "Avalanche": load_avalanche,
    "Acid Bath": load_acid_bath,
    "Multi-Fluid": load_multi_fluid,
    "Fire and Ice": load_fire_and_ice,
    "Sandbox": load_sandbox,
    "Mudslide": load_mudslide,
}
