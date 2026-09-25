"""Save/load particle scenes to .fs3d files (compressed NumPy .npz)."""

from __future__ import annotations

import json
import os

import cupy as cp
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saves")
# v3: added sim_time to header (bd-unl.14)
# v4: each rigid body carries its constructor arguments under "build", so
#     load rebuilds the bodies instead of patching the session's
#     (bd-r5-review-n28.57)
VERSION = 4


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
    Each body is stored as a dict with all 6 float4 fields, plus "build": the
    arguments it was created with (rbm.build_params), so load can rebuild it
    and its boundary samples in a session that does not have it.
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
            "build":        rbm.build_params(i),
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


_RB_STATE_FIELDS = ("position", "rotation", "lin_vel", "ang_vel",
                    "half_extents", "inertia_inv")


def _write_body_state(rbm, i: int, b_data: dict) -> None:
    """Copy the six saved float4 fields of one body into rbm._bodies_cpu[i]."""
    for key in _RB_STATE_FIELDS:
        rbm._bodies_cpu[i][key] = np.array(b_data[key], dtype=np.float32)


def _restore_rigid_body_manager(rbm, rb_data: dict, modules: list) -> None:
    """Rebuild a RigidBodyManager from the serialised dict.

    Saves from VERSION 4 on carry each body's constructor arguments under
    "build". For those the manager is reset and every body is re-added with
    add_rigid_body_with_particles (or add_rigid_body when it had no boundary
    samples), then the saved dynamic state (pose, velocities, and the packed
    mass, restitution, friction and type words) overwrites the fresh bodies.
    A save with no bodies leaves the manager empty. The boundary block is
    rebuilt from the new bodies on the next substep.

    Older saves lack "build". They keep the old behaviour: only bodies that
    already exist in the session get the saved state, and a warning is
    printed when the counts differ.

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

    count = int(rb_data.get("count", 0))
    bodies = rb_data.get("bodies", [])[:count]

    if count == 0 or all("build" in b for b in bodies):
        if len(bodies) > rbm.max_bodies:
            print(f"WARNING: save has {len(bodies)} rigid bodies but the "
                  f"manager holds at most {rbm.max_bodies}; extra bodies dropped")
            bodies = bodies[:rbm.max_bodies]
        rbm.reset()
        for i, b_data in enumerate(bodies):
            build = b_data["build"]
            pos = b_data["position"]
            kwargs = dict(
                position=(pos[0], pos[1], pos[2]),
                half_extents=tuple(build["half_extents"]),
                mass=float(build["mass"]),
                sdf_type=int(build["sdf_type"]),
                restitution=float(build["restitution"]),
                friction=float(build["friction"]),
                is_kinematic=bool(build["is_kinematic"]),
            )
            if build.get("with_particles", False):
                idx = rbm.add_rigid_body_with_particles(
                    spacing=float(build["spacing"]),
                    smoothing_length=float(build["smoothing_length"]),
                    rho0=float(build["rho0"]),
                    **kwargs,
                )
            else:
                idx = rbm.add_rigid_body(**kwargs)
            _write_body_state(rbm, idx, b_data)
        rbm.finalize_boundary_data()
        rbm.upload(modules)
        return

    # Pre-v4 save: no topology, so patch the bodies the session already has.
    restore_count = min(count, rbm.num_bodies, len(bodies))
    if count != rbm.num_bodies:
        print(f"WARNING: old save format has {count} rigid bodies without "
              f"build data; the session has {rbm.num_bodies}. Restored state "
              f"for {restore_count} bodies only. Re-save to fix.")
    if restore_count == 0:
        return

    for i in range(restore_count):
        _write_body_state(rbm, i, bodies[i])

    rbm.upload(modules)


# Expected per-particle arrays in a .fs3d file: name -> (dtype, trailing shape).
# Only these keys are read, so an older save that carries an extra array
# (for example the float4 colour state removed in bd-r5-review-n28.144, or
# the spawn-time "density" array removed in bd-r5-review-n28.152) still
# loads. Density is not saved: K_Step1 recomputes it on the first substep.
_SCENE_ARRAYS = {
    "position":     (np.float32, (4,)),
    "velocity":     (np.float32, (4,)),
    "packed_info":  (np.uint32,  ()),
    "temperature":  (np.float32, ()),
    "health":       (np.float32, ()),
    "lifetime":     (np.float32, ()),
    "mass":         (np.float32, ()),
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


def _validate_header(header: dict) -> None:
    """Check the header values load_scene applies after the world is written.

    world_half_size, gravity_y and sim_time are optional (old test files and
    hand-made saves may omit them), but when present they must be finite
    numbers, and world_half_size must be positive. Raises ValueError naming
    the bad key, before any world array is modified. (bd-r5-review-n28.78)
    """
    import math

    for key in ("world_half_size", "gravity_y", "sim_time"):
        if key not in header:
            continue
        v = header[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Save file header '{key}' is not a finite number: {v!r}")
    if "world_half_size" in header and header["world_half_size"] <= 0:
        raise ValueError(
            f"Save file header 'world_half_size' must be positive, "
            f"got {header['world_half_size']!r}"
        )


def _saved_solver_profile(header: dict, sim, warnings: list):
    """Return the profile to switch to for header['solver'], or None.

    None when the save names no solver, names the active one, or the sim
    cannot switch. An unknown name adds a warning and keeps the active
    solver. (bd-r5-review-n28.117)
    """
    saved = header.get("solver")
    if not isinstance(saved, str) or not hasattr(sim, "set_solver_profile"):
        return None
    from solver_profiles import PROFILES
    if saved not in PROFILES:
        warnings.append(f"unknown solver {saved!r} in save; kept the active solver")
        return None
    if saved == getattr(getattr(sim, "_profile", None), "name", None):
        return None
    return PROFILES[saved]


def load_scene(world, sim, camera, filename: str, report: dict | None = None) -> int:
    """Load scene from saves/<filename>. Returns particle count.

    Switches to the saved solver profile when it differs from the active
    one, then resets world, restores all particle arrays, SDF collision
    geometry (including kinematic motions), rigid bodies, and camera.

    Header values and particle arrays are checked before the world is
    touched, so a bad file raises and leaves the current scene intact. The
    optional parts (SDF geometry, rigid bodies, camera) are applied in their
    own try/except: a failure there adds a message to report["warnings"] and
    the particles still load.

    report, when given, is filled in place:
      report["world_modified"]: True once any world array was written. The
        caller must then run its scene-loaded handling even if this function
        raised afterwards.
      report["warnings"]: list of messages for optional parts that failed.
    """
    if report is None:
        report = {}
    report["world_modified"] = False
    warnings = report.setdefault("warnings", [])
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
    _validate_header(header)
    target_profile = _saved_solver_profile(header, sim, warnings)

    # Drop any rigid boundary particles in the file: the RigidBodyManager
    # rebuilds its block from the bodies on the next substep, so keeping them
    # would leave a second, orphaned block. (bd-r5-review-n28.22)
    from materials import MAT_RIGID
    keep = (arrays["packed_info"] & 0xFF) != MAT_RIGID
    if not np.all(keep):
        arrays = {k: v[keep] for k, v in arrays.items()}
        n = int(np.count_nonzero(keep))

    # Reset world
    report["world_modified"] = True
    world.packed_info[:] = 0
    world._high_water = 0

    # Switch to the solver the save was made with before the particles go
    # in, so a DFSPH save does not run under WCSPH with the wrong dt. The
    # switch only changes solver params and constants, not particle arrays.
    # Rigid body restore below re-uploads the body count; re-upload here too
    # so a save without bodies still reaches the new modules.
    # (bd-r5-review-n28.117)
    if target_profile is not None:
        print(f"Load: switching solver to {target_profile.name}")
        sim.set_solver_profile(target_profile)
        if hasattr(sim, "rigid_body_manager"):
            sim.rigid_body_manager.upload(sim.get_all_modules())

    # Restore SDF geometry BEFORE setting _high_water so GPU kernels
    # dispatched during the same frame see the correct constant memory state.
    sdf_data = header.get("sdf", {})
    if sdf_data and hasattr(sim, "sdf_manager"):
        try:
            _restore_sdf_manager(sim.sdf_manager, sdf_data)
        except Exception as e:
            warnings.append(f"SDF geometry not restored: {e!r}")

    # Restore arrays
    world.position[:n] = cp.asarray(arrays["position"])
    world.velocity[:n] = cp.asarray(arrays["velocity"])
    world.packed_info[:n] = cp.asarray(arrays["packed_info"])
    world.temperature[:n] = cp.asarray(arrays["temperature"])
    world.health[:n] = cp.asarray(arrays["health"])
    world.lifetime[:n] = cp.asarray(arrays["lifetime"])
    world.mass[:n] = cp.asarray(arrays["mass"])

    # bd-unl.13: zero warm-start and sleep fields so loaded particles start
    # fresh -- stale kappa/kappa_v produce wrong first-frame forces;
    # stale sleep_counter (up to 255) keeps particles asleep for ~255 frames.
    world.kappa[:n] = 0.0
    world.kappa_v[:n] = 0.0
    world.sleep_counter[:n] = 0

    world._high_water = n

    # The particle set was replaced: force a full sort and a graph recapture
    # on the next substep, and drop stale foam, before any later step can
    # raise. The UI also calls these on success; calling them twice is
    # harmless. (bd-r5-review-n28.78)
    if hasattr(sim, "notify_scene_changed"):
        sim.notify_scene_changed()
    if hasattr(sim, "_invalidate_graphs"):
        sim._invalidate_graphs()
    if hasattr(world, "foam_count"):
        world.foam_count.fill(0)

    # No color to rebuild: K_ComputeColor colors the loaded state on the
    # next frame, paused or not (bd-r5-review-n28.145).

    # Rebuild _spawned_material_ids from loaded packed_info
    unique_mats = cp.unique(world.packed_info[:n] & 0xFF).get()
    world._spawned_material_ids = set(int(m) for m in unique_mats if m != 0)

    # Rebuild rigid bodies from the save (bd-r5-review-n28.57).
    # bd-unl.15: use sim.get_all_modules() so every compiled module receives the
    # updated c_num_rigid_bodies constant -- the old 3-module list was a partial
    # subset that would silently leave any other module with a stale count.
    rb_data = header.get("rigid_bodies", {})
    if rb_data and hasattr(sim, "rigid_body_manager"):
        try:
            _restore_rigid_body_manager(
                sim.rigid_body_manager, rb_data, sim.get_all_modules()
            )
        except Exception as e:
            warnings.append(f"Rigid bodies not restored: {e!r}")
            # A half-built manager is worse than none: clear it so the
            # loaded fluid runs without bodies.
            try:
                sim.rigid_body_manager.reset()
                sim.rigid_body_manager.upload(sim.get_all_modules())
            except Exception as e2:
                warnings.append(f"Rigid body reset failed: {e2!r}")

    # Restore simulation params
    if "world_half_size" in header:
        sim.set_world_size(header["world_half_size"])
    if "gravity_y" in header:
        sim.set_gravity(header["gravity_y"])

    # Restore camera
    cam = header.get("camera")
    if cam:
        try:
            target = np.array(cam["target"], dtype=np.float32).reshape(3)
            distance = float(cam["distance"])
            azimuth = float(cam["azimuth"])
            elevation = float(cam["elevation"])
        except Exception as e:
            warnings.append(f"Camera not restored: {e!r}")
        else:
            camera.target = target
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation

    # bd-unl.14: restore sim_time from save so kinematic SDF objects resume at
    # their saved positions rather than snapping to t=0.  Falls back to 0.0 for
    # older saves (version < 3) that don't carry the field.
    sim.sim_time = float(header.get("sim_time", 0.0))
    sim._last_frame_time = None
    sim.reset_spawn_damping()

    # The world arrays were replaced, so the boundary block is gone. Rebuild
    # the block on the next substep from the restored body state. A v4 load
    # already cleared the flag through reset(); old saves still need this.
    # (bd-r5-review-n28.22)
    if hasattr(sim, "rigid_body_manager"):
        sim.rigid_body_manager.mark_for_reinjection()

    return n
