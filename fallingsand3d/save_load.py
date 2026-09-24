"""Save/load particle scenes to .fs3d files (compressed NumPy .npz)."""

from __future__ import annotations

import json
import os

import cupy as cp
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saves")
VERSION = 3  # v3: added sim_time to header (bd-unl.14)


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

    # Rigid boundary particles are topology owned by the RigidBodyManager and
    # are rebuilt from the bodies on load, so they are not saved.
    # (bd-r5-review-n28.22)
    from materials import MAT_RIGID
    packed_host = world.packed_info[:n].get()
    keep = (packed_host & 0xFF) != MAT_RIGID
    n_saved = int(np.count_nonzero(keep))

    def _host(arr):
        return arr[:n].get()[keep]

    # Collect SDF and rigid-body state (read-only access to sim internals)
    sdf_data = {}
    if hasattr(sim, "sdf_manager"):
        sdf_data = _serialize_sdf_manager(sim.sdf_manager)

    rb_data = {}
    if hasattr(sim, "rigid_body_manager"):
        rb_data = _serialize_rigid_body_manager(sim.rigid_body_manager)

    header = {
        "version": VERSION,
        "particle_count": n_saved,
        "solver": sim._profile.name if hasattr(sim._profile, "name") else str(sim._profile.solver_type),
        "world_half_size": float(sim.world_half_size),
        "gravity_y": float(sim.gravity_y),
        "sim_time": float(sim.sim_time),  # bd-unl.14: preserve kinematic SDF positions on load
        "camera": {
            "target": camera.target.tolist(),
            "distance": float(camera.distance),
            "azimuth": float(camera.azimuth),
            "elevation": float(camera.elevation),
        },
        "sdf": sdf_data,
        "rigid_bodies": rb_data,
    }

    arrays = {
        "header": np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
        "position": _host(world.position),
        "velocity": _host(world.velocity),
        "packed_info": packed_host[keep],
        "temperature": _host(world.temperature),
        "health": _host(world.health),
        "lifetime": _host(world.lifetime),
        "mass": _host(world.mass),
        "particle_dye": _host(world.particle_dye),
        "density": _host(world.density),
    }
    # Write through a file object: given a path, NumPy appends ".npz", so the
    # file became "<name>.fs3d.npz" and list_saves/load_scene never found it.
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
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


# Expected per-particle arrays in a .fs3d file: name -> (dtype, trailing shape).
_SCENE_ARRAYS = {
    "position":     (np.float32, (4,)),
    "velocity":     (np.float32, (4,)),
    "packed_info":  (np.uint32,  ()),
    "temperature":  (np.float32, ()),
    "health":       (np.float32, ()),
    "lifetime":     (np.float32, ()),
    "mass":         (np.float32, ()),
    "particle_dye": (np.float32, (4,)),
    "density":      (np.float32, ()),
}


def _validate_scene_arrays(data, n: int) -> dict:
    """Check every particle array in a loaded .fs3d before the world is touched.

    Each array must exist, have shape (n,) or (n, 4) and the dtype save_scene
    writes. Material ids (packed_info bits [7:0]) must be below MAX_MATERIALS,
    because kernels index the 32-entry c_materials table with them. The
    behavior class (bits [9:8]) is 2 bits wide, so it is always in range.

    Returns the arrays as NumPy arrays. Raises ValueError naming the bad key.
    """
    from materials import MAX_MATERIALS

    missing = [k for k in _SCENE_ARRAYS if k not in data]
    if missing:
        raise ValueError(f"Save file missing arrays: {missing}")

    arrays = {}
    for key, (dtype, tail) in _SCENE_ARRAYS.items():
        arr = data[key]
        expected = (n,) + tail
        if arr.shape != expected:
            raise ValueError(
                f"Save file array '{key}' has shape {arr.shape}, expected {expected}"
            )
        if arr.dtype != dtype:
            raise ValueError(
                f"Save file array '{key}' has dtype {arr.dtype}, "
                f"expected {np.dtype(dtype)}"
            )
        arrays[key] = arr

    mat_ids = arrays["packed_info"] & 0xFF
    bad = mat_ids >= MAX_MATERIALS
    if np.any(bad):
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"Save file array 'packed_info' has material id {int(mat_ids[first])} "
            f"at particle {first}; ids must be below {MAX_MATERIALS}"
        )
    return arrays


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
    n = header.get("particle_count")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"Save file header has invalid particle_count: {n!r}")

    max_p = world.position.shape[0]
    if n > max_p:
        raise ValueError(
            f"Save has {n:,} particles but world capacity is {max_p:,}. "
            f"Increase max_particles to at least {n:,} before loading."
        )

    # bd-r5-review-n28.12: validate every array (key, shape, dtype, material
    # ids) before touching the world, so a corrupt or hand-edited file raises
    # a clean ValueError and leaves the current scene intact.
    arrays = _validate_scene_arrays(data, n)

    # Drop any rigid boundary particles in the file: the RigidBodyManager
    # rebuilds its block from the bodies on the next substep, so keeping them
    # would leave a second, orphaned block. (bd-r5-review-n28.22)
    from materials import MAT_RIGID
    keep = (arrays["packed_info"] & 0xFF) != MAT_RIGID
    if not np.all(keep):
        arrays = {k: v[keep] for k, v in arrays.items()}
        n = int(np.count_nonzero(keep))

    # Reset world
    world.packed_info[:] = 0
    world._high_water = 0

    # Restore SDF geometry BEFORE setting _high_water so GPU kernels
    # dispatched during the same frame see the correct constant memory state.
    sdf_data = header.get("sdf", {})
    if sdf_data and hasattr(sim, "sdf_manager"):
        _restore_sdf_manager(sim.sdf_manager, sdf_data)

    # Restore arrays
    world.position[:n] = cp.asarray(arrays["position"])
    world.velocity[:n] = cp.asarray(arrays["velocity"])
    world.packed_info[:n] = cp.asarray(arrays["packed_info"])
    world.temperature[:n] = cp.asarray(arrays["temperature"])
    world.health[:n] = cp.asarray(arrays["health"])
    world.lifetime[:n] = cp.asarray(arrays["lifetime"])
    world.mass[:n] = cp.asarray(arrays["mass"])
    world.particle_dye[:n] = cp.asarray(arrays["particle_dye"])
    world.density[:n] = cp.asarray(arrays["density"])
    world.veleval[:n] = world.velocity[:n]

    # bd-unl.13: zero warm-start and sleep fields so loaded particles start
    # fresh -- stale kappa/kappa_v/lambda_pbf produce wrong first-frame forces;
    # stale sleep_counter (up to 255) keeps particles asleep for ~255 frames;
    # stale angular_velocity carries micropolar spin from the saved session.
    world.kappa[:n] = 0.0
    world.kappa_v[:n] = 0.0
    world.lambda_pbf[:n] = 0.0
    world.sleep_counter[:n] = 0
    world.angular_velocity[:n] = 0.0

    world._high_water = n

    # Rebuild _spawned_material_ids from loaded packed_info
    unique_mats = cp.unique(world.packed_info[:n] & 0xFF).get()
    world._spawned_material_ids = set(int(m) for m in unique_mats if m != 0)

    # Restore rigid body dynamic state
    # bd-unl.15: use sim.get_all_modules() so every compiled module receives the
    # updated c_num_rigid_bodies constant -- the old 3-module list was a partial
    # subset that would silently leave any other module with a stale count.
    rb_data = header.get("rigid_bodies", {})
    if rb_data and hasattr(sim, "rigid_body_manager"):
        _restore_rigid_body_manager(
            sim.rigid_body_manager, rb_data, sim.get_all_modules()
        )

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

    # bd-unl.14: restore sim_time from save so kinematic SDF objects resume at
    # their saved positions rather than snapping to t=0.  Falls back to 0.0 for
    # older saves (version < 3) that don't carry the field.
    sim.sim_time = float(header.get("sim_time", 0.0))
    sim._last_frame_time = None
    sim.reset_spawn_damping()

    # The world arrays were replaced, so the boundary block is gone. Keep the
    # bodies (reset() would drop them) and rebuild the block on the next
    # substep from the restored body state. (bd-r5-review-n28.22)
    if hasattr(sim, "rigid_body_manager"):
        sim.rigid_body_manager.mark_for_reinjection()

    return n
