"""Save/load particle scenes to .fs3d files (compressed NumPy .npz)."""

from __future__ import annotations

import json
import os

import cupy as cp
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saves")
VERSION = 2  # incremented from 1 to include SDF + rigid body state


def _ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def list_saves() -> list[str]:
    """Return sorted list of .fs3d filenames in the saves directory."""
    _ensure_save_dir()
    return sorted(f for f in os.listdir(SAVE_DIR) if f.endswith(".fs3d"))


def _serialize_sdf_manager(sdf_manager) -> dict:
    """Serialize an SDFManager to a JSON-safe dict.

    Reads sdf_manager._objects (list of dicts) and sdf_manager._motions
    (dict mapping str(obj_id) -> motion config) read-only.
    """
    objects = [dict(obj) for obj in sdf_manager._objects]
    # Ensure all values are JSON-serialisable (lists not tuples)
    for obj in objects:
        for key, val in obj.items():
            if isinstance(val, (list, tuple)):
                obj[key] = list(val)
    # Motions: keys are int object IDs, values are dicts with "type", "params", "center"
    motions = {}
    for obj_id, motion in sdf_manager._motions.items():
        motions[str(obj_id)] = {
            "type": motion["type"],
            "params": dict(motion["params"]),
            "center": list(motion["center"]),
        }
    return {"objects": objects, "motions": motions}


def _serialize_rigid_body_manager(rbm) -> dict:
    """Serialize a RigidBodyManager's CPU-side body state to a JSON-safe dict.

    Downloads the current GPU state so dynamic bodies' positions/velocities
    are captured, then serialises the RIGID_BODY_DTYPE structured array.
    Each body is stored as a dict with all 6 float4 fields.
    """
    if rbm is None or rbm.num_bodies == 0:
        return {"count": 0, "bodies": []}

    # Download current GPU state so dynamic bodies are up to date
    cpu_bodies = rbm.d_rigid_bodies[:rbm.num_bodies].get()

    bodies = []
    for i in range(rbm.num_bodies):
        b = cpu_bodies[i]
        bodies.append({
            "position":     b["position"].tolist(),
            "rotation":     b["rotation"].tolist(),
            "lin_vel":      b["lin_vel"].tolist(),
            "ang_vel":      b["ang_vel"].tolist(),
            "half_extents": b["half_extents"].tolist(),
            "inertia_inv":  b["inertia_inv"].tolist(),
        })
    return {"count": rbm.num_bodies, "bodies": bodies}


def save_scene(world, sim, camera, filename: str) -> str:
    """Save current scene state to saves/<filename>.fs3d.

    Serialises particle arrays, SDF collision objects (including kinematic
    motion configs), and rigid body state.  Performs GPU->CPU sync
    (user-initiated, acceptable per CLAUDE.md).
    Returns the full path written.
    """
    _ensure_save_dir()
    if not filename.endswith(".fs3d"):
        filename += ".fs3d"
    path = os.path.join(SAVE_DIR, filename)

    n = world._high_water

    # Collect SDF and rigid-body state (read-only access to sim internals)
    sdf_data = {}
    if hasattr(sim, "sdf_manager"):
        sdf_data = _serialize_sdf_manager(sim.sdf_manager)

    rb_data = {}
    if hasattr(sim, "rigid_body_manager"):
        rb_data = _serialize_rigid_body_manager(sim.rigid_body_manager)

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
        "sdf": sdf_data,
        "rigid_bodies": rb_data,
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
        mass=world.mass[:n].get(),
        particle_dye=world.particle_dye[:n].get(),
        density=world.density[:n].get(),
    )
    return path


def _restore_sdf_manager(sdf_manager, sdf_data: dict) -> None:
    """Restore an SDFManager from the serialised dict produced by _serialize_sdf_manager.

    Clears any existing SDF state, then re-adds all objects and kinematic
    motions.  Uploads to GPU constant memory at the end.
    """
    if not sdf_data:
        return

    sdf_manager.clear()

    objects = sdf_data.get("objects", [])
    for obj in objects:
        sdf_manager.add_sdf_object(
            sdf_type=int(obj["type"]),
            position=tuple(obj.get("position", [0.0, 0.0, 0.0])),
            size=tuple(obj.get("size", [0.1, 0.1, 0.1])),
            rotation=tuple(obj.get("rotation", [0.0, 0.0, 0.0, 1.0])),
            velocity=tuple(obj.get("velocity", [0.0, 0.0, 0.0])),
            restitution=float(obj.get("restitution", 0.5)),
            friction=float(obj.get("friction", 0.3)),
            angular_speed=float(obj.get("angular_speed", 0.0)),
            angular_axis=tuple(obj.get("angular_axis", [0.0, 1.0, 0.0])),
        )

    motions = sdf_data.get("motions", {})
    for obj_id_str, motion in motions.items():
        obj_id = int(obj_id_str)
        sdf_manager.add_kinematic_motion(
            obj_id,
            motion_type=motion["type"],
            params=motion["params"],
        )
        # Restore the kinematic center snapshot that was saved
        if obj_id in sdf_manager._motions and "center" in motion:
            sdf_manager._motions[obj_id]["center"] = list(motion["center"])

    sdf_manager.upload_if_dirty()


def _restore_rigid_body_manager(rbm, rb_data: dict, modules: list) -> None:
    """Restore a RigidBodyManager's GPU state from the serialised dict.

    Only restores the raw body struct data (position, rotation, velocities,
    half_extents, inertia_inv) directly into the GPU array and calls
    rbm.upload() to sync count to constant memory.  The boundary particle
    psi_b data is topology (not state) and is preserved from the existing
    rbm setup -- loading only updates dynamic state.

    Parameters
    ----------
    rbm : RigidBodyManager
        The rigid body manager to restore into.
    rb_data : dict
        Dict produced by _serialize_rigid_body_manager.
    modules : list
        CuPy RawModule objects for uploading the body count to constant memory.
    """
    if not rb_data:
        return

    from rigid_bodies import RIGID_BODY_DTYPE

    count = rb_data.get("count", 0)
    bodies = rb_data.get("bodies", [])

    # Only restore state for bodies that already exist in the manager.
    # If the save has more bodies than the current manager (e.g. different
    # preset), we skip the extra bodies to stay within capacity.
    restore_count = min(count, rbm.num_bodies, len(bodies))
    if restore_count == 0:
        return

    for i in range(restore_count):
        b_data = bodies[i]
        rbm._bodies_cpu[i]["position"]     = np.array(b_data["position"],     dtype=np.float32)
        rbm._bodies_cpu[i]["rotation"]     = np.array(b_data["rotation"],     dtype=np.float32)
        rbm._bodies_cpu[i]["lin_vel"]      = np.array(b_data["lin_vel"],      dtype=np.float32)
        rbm._bodies_cpu[i]["ang_vel"]      = np.array(b_data["ang_vel"],      dtype=np.float32)
        rbm._bodies_cpu[i]["half_extents"] = np.array(b_data["half_extents"], dtype=np.float32)
        rbm._bodies_cpu[i]["inertia_inv"]  = np.array(b_data["inertia_inv"],  dtype=np.float32)

    rbm.upload(modules)


def load_scene(world, sim, camera, filename: str) -> int:
    """Load scene from saves/<filename>. Returns particle count.

    Resets world, restores all particle arrays, SDF collision geometry
    (including kinematic motions), rigid body dynamic state, and camera.
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

    # Validate all required keys exist before modifying world state
    required = ["position", "velocity", "packed_info", "temperature",
                "health", "lifetime", "mass", "particle_dye", "density"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Save file missing arrays: {missing}")

    # Reset world
    world.packed_info[:] = 0
    world._high_water = 0

    # Restore SDF geometry BEFORE setting _high_water so GPU kernels
    # dispatched during the same frame see the correct constant memory state.
    sdf_data = header.get("sdf", {})
    if sdf_data and hasattr(sim, "sdf_manager"):
        _restore_sdf_manager(sim.sdf_manager, sdf_data)

    # Restore arrays
    world.position[:n] = cp.asarray(data["position"])
    world.velocity[:n] = cp.asarray(data["velocity"])
    world.packed_info[:n] = cp.asarray(data["packed_info"])
    world.temperature[:n] = cp.asarray(data["temperature"])
    world.health[:n] = cp.asarray(data["health"])
    world.lifetime[:n] = cp.asarray(data["lifetime"])
    world.mass[:n] = cp.asarray(data["mass"])
    world.particle_dye[:n] = cp.asarray(data["particle_dye"])
    world.density[:n] = cp.asarray(data["density"])
    world.veleval[:n] = world.velocity[:n]
    world._high_water = n

    # Rebuild _spawned_material_ids from loaded packed_info
    unique_mats = cp.unique(world.packed_info[:n] & 0xFF).get()
    world._spawned_material_ids = set(int(m) for m in unique_mats if m != 0)

    # Restore rigid body dynamic state
    rb_data = header.get("rigid_bodies", {})
    if rb_data and hasattr(sim, "rigid_body_manager"):
        import integrate
        import pbf_solver
        import dfsph_solver
        modules = []
        for mod_getter in [
            lambda: integrate.get_module(),
            lambda: pbf_solver.get_module(),
            lambda: dfsph_solver.get_module(),
        ]:
            try:
                modules.append(mod_getter())
            except Exception:
                pass
        _restore_rigid_body_manager(sim.rigid_body_manager, rb_data, modules)

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
