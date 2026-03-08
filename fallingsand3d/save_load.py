"""Save/load particle scenes to .fs3d files (compressed NumPy .npz)."""

from __future__ import annotations

import json
import os

import cupy as cp
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saves")
VERSION = 1


def _ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def list_saves() -> list[str]:
    """Return sorted list of .fs3d filenames in the saves directory."""
    _ensure_save_dir()
    return sorted(f for f in os.listdir(SAVE_DIR) if f.endswith(".fs3d"))


def save_scene(world, sim, camera, filename: str) -> str:
    """Save current scene state to saves/<filename>.fs3d.

    Performs GPU->CPU sync (user-initiated, acceptable per CLAUDE.md).
    Returns the full path written.
    """
    _ensure_save_dir()
    if not filename.endswith(".fs3d"):
        filename += ".fs3d"
    path = os.path.join(SAVE_DIR, filename)

    n = world._high_water

    header = {
        "version": VERSION,
        "particle_count": int(n),
        "solver": sim._profile.name if hasattr(sim._profile, "name") else str(sim._profile.solver_type),
        "world_half_size": float(sim.world_half_size),
        "gravity_y": float(sim.gravity_y),
        "camera": {
            "target": camera.target.tolist(),
            "distance": float(camera.distance),
            "azimuth": float(camera.azimuth),
            "elevation": float(camera.elevation),
        },
    }

    np.savez_compressed(
        path,
        header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
        position=world.position[:n].get(),
        velocity=world.velocity[:n].get(),
        packed_info=world.packed_info[:n].get(),
        temperature=world.temperature[:n].get(),
        health=world.health[:n].get(),
        lifetime=world.lifetime[:n].get(),
        particle_dye=world.particle_dye[:n].get(),
        density=world.density[:n].get(),
    )
    return path


def load_scene(world, sim, camera, filename: str) -> int:
    """Load scene from saves/<filename>. Returns particle count.

    Resets world, restores all particle arrays and camera state.
    """
    if not filename.endswith(".fs3d"):
        filename += ".fs3d"
    path = os.path.join(SAVE_DIR, filename)

    data = np.load(path, allow_pickle=False)

    header = json.loads(data["header"].tobytes())
    n = header["particle_count"]

    max_p = world.position.shape[0]
    if n > max_p:
        raise ValueError(
            f"Save has {n:,} particles but world capacity is {max_p:,}. "
            f"Increase max_particles to at least {n:,} before loading."
        )

    # Reset world
    world.packed_info[:] = 0
    world._high_water = 0

    # Restore arrays
    world.position[:n] = cp.asarray(data["position"])
    world.velocity[:n] = cp.asarray(data["velocity"])
    world.packed_info[:n] = cp.asarray(data["packed_info"])
    world.temperature[:n] = cp.asarray(data["temperature"])
    world.health[:n] = cp.asarray(data["health"])
    world.lifetime[:n] = cp.asarray(data["lifetime"])
    world.particle_dye[:n] = cp.asarray(data["particle_dye"])
    world.density[:n] = cp.asarray(data["density"])
    world.veleval[:n] = world.velocity[:n]
    world._high_water = n

    # Restore simulation params
    if "world_half_size" in header:
        sim.set_world_size(header["world_half_size"])
    if "gravity_y" in header:
        sim.set_gravity(header["gravity_y"])

    # Restore camera
    cam = header.get("camera")
    if cam:
        camera.target = np.array(cam["target"], dtype=np.float32)
        camera.distance = cam["distance"]
        camera.azimuth = cam["azimuth"]
        camera.elevation = cam["elevation"]

    sim.sim_time = 0.0
    sim._last_frame_time = None
    sim.reset_spawn_damping()

    return n
