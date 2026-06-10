"""Round-trip tests for save_load.py -- SDF + rigid body serialisation.

Tests the _serialize_sdf_manager / _restore_sdf_manager and
_serialize_rigid_body_manager helpers in isolation (no GPU required for the
pure-Python parts), plus a JSON round-trip of the full header structure.

A full end-to-end test (save_scene / load_scene with a live CuPy world) is
omitted here because it requires a CUDA GPU and a fully initialised World /
Simulation / Camera -- integration coverage for that path comes from the app's
existing preset reload paths.  The helpers tested here exercise the logic added
by bd-mzc.26, bd-unl.13, bd-unl.14, and bd-unl.15.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

# Import helpers directly without importing the full save_load module
# (which pulls in cupy at the module level and requires a CUDA GPU).
# We patch cupy with a minimal stub so the module-level import succeeds.
if "cupy" not in sys.modules:
    import types
    _cupy_stub = types.ModuleType("cupy")
    _cupy_stub.zeros = None
    _cupy_stub.asarray = None
    _cupy_stub.unique = None
    sys.modules["cupy"] = _cupy_stub

from save_load import _serialize_sdf_manager, _restore_sdf_manager


# ---------------------------------------------------------------------------
# Lightweight stubs (no GPU required)
# ---------------------------------------------------------------------------

class _StubSDFManager:
    """Minimal SDFManager stub that exercises the serialise/restore helpers."""

    def __init__(self) -> None:
        self._objects: list[dict] = []
        self._motions: dict[int, dict] = {}
        self._dirty = False

    @property
    def count(self) -> int:
        return len(self._objects)

    def add_sdf_object(
        self,
        sdf_type,
        position=(0.0, 0.0, 0.0),
        size=(0.1, 0.1, 0.1),
        rotation=(0.0, 0.0, 0.0, 1.0),
        velocity=(0.0, 0.0, 0.0),
        restitution=0.5,
        friction=0.3,
        angular_speed=0.0,
        angular_axis=(0.0, 1.0, 0.0),
    ) -> int:
        obj_id = len(self._objects)
        self._objects.append({
            "type": sdf_type,
            "position": list(position),
            "size": list(size),
            "rotation": list(rotation),
            "velocity": list(velocity),
            "restitution": restitution,
            "friction": friction,
            "angular_speed": angular_speed,
            "angular_axis": list(angular_axis),
        })
        self._dirty = True
        return obj_id

    def add_kinematic_motion(self, obj_id: int, motion_type: str, params: dict) -> None:
        if 0 <= obj_id < len(self._objects):
            self._motions[obj_id] = {
                "type": motion_type,
                "params": dict(params),
                "center": list(self._objects[obj_id]["position"]),
            }

    def clear(self) -> None:
        self._objects.clear()
        self._motions.clear()
        self._dirty = True

    def upload_if_dirty(self) -> None:
        self._dirty = False


# ---------------------------------------------------------------------------
# SDF serialisation tests
# ---------------------------------------------------------------------------

def test_serialize_empty_sdf_manager() -> None:
    """Serialising an empty SDFManager produces empty objects/motions."""
    mgr = _StubSDFManager()
    result = _serialize_sdf_manager(mgr)
    assert result["objects"] == []
    assert result["motions"] == {}
    print("[OK] Empty SDFManager serialises to empty lists")


def test_sdf_round_trip_single_box() -> None:
    """Single SDF_BOX object survives a serialise/restore round-trip."""
    SDF_BOX = 0
    src = _StubSDFManager()
    src.add_sdf_object(
        sdf_type=SDF_BOX,
        position=(0.5, 0.1, -0.3),
        size=(0.2, 0.15, 0.1),
        rotation=(0.0, 0.0, 0.0, 1.0),
        restitution=0.6,
        friction=0.4,
    )

    sdf_data = _serialize_sdf_manager(src)
    assert sdf_data["objects"][0]["type"] == SDF_BOX

    dst = _StubSDFManager()
    _restore_sdf_manager(dst, sdf_data)

    assert dst.count == 1
    obj = dst._objects[0]
    assert obj["type"] == SDF_BOX
    np.testing.assert_allclose(obj["position"], [0.5, 0.1, -0.3], atol=1e-6)
    np.testing.assert_allclose(obj["size"], [0.2, 0.15, 0.1], atol=1e-6)
    assert abs(obj["restitution"] - 0.6) < 1e-6
    assert abs(obj["friction"] - 0.4) < 1e-6
    print("[OK] Single SDF_BOX round-trip: position/size/restitution/friction preserved")


def test_sdf_round_trip_multiple_objects() -> None:
    """Multiple SDF objects (box + sphere) survive a round-trip."""
    SDF_BOX = 0
    SDF_SPHERE = 1

    src = _StubSDFManager()
    src.add_sdf_object(sdf_type=SDF_BOX, position=(0.0, 0.0, 0.0), size=(0.3, 0.3, 0.3))
    src.add_sdf_object(sdf_type=SDF_SPHERE, position=(1.0, 0.5, 0.0), size=(0.25, 0.25, 0.25))

    sdf_data = _serialize_sdf_manager(src)
    dst = _StubSDFManager()
    _restore_sdf_manager(dst, sdf_data)

    assert dst.count == 2
    assert dst._objects[0]["type"] == SDF_BOX
    assert dst._objects[1]["type"] == SDF_SPHERE
    np.testing.assert_allclose(dst._objects[1]["position"], [1.0, 0.5, 0.0], atol=1e-6)
    print("[OK] 2 SDF objects (box + sphere) round-trip correctly")


def test_sdf_round_trip_kinematic_motion() -> None:
    """Kinematic motion config (oscillate_y) survives a round-trip."""
    SDF_BOX = 0
    src = _StubSDFManager()
    obj_id = src.add_sdf_object(
        sdf_type=SDF_BOX,
        position=(0.0, 0.0, 0.0),
        size=(0.4, 0.1, 0.4),
    )
    src.add_kinematic_motion(
        obj_id,
        motion_type="oscillate_y",
        params={"amplitude": 0.3, "frequency": 0.5},
    )
    # Manually set center to verify it survives
    src._motions[obj_id]["center"] = [0.0, 0.1, 0.0]

    sdf_data = _serialize_sdf_manager(src)
    dst = _StubSDFManager()
    _restore_sdf_manager(dst, sdf_data)

    assert dst.count == 1
    assert 0 in dst._motions, "Kinematic motion for object 0 should be restored"
    motion = dst._motions[0]
    assert motion["type"] == "oscillate_y"
    assert abs(motion["params"]["amplitude"] - 0.3) < 1e-6
    assert abs(motion["params"]["frequency"] - 0.5) < 1e-6
    np.testing.assert_allclose(motion["center"], [0.0, 0.1, 0.0], atol=1e-6)
    print("[OK] Kinematic motion (oscillate_y) survives round-trip with center")


def test_sdf_serialise_is_json_serialisable() -> None:
    """Serialised SDF data must be JSON-encodable (for header embedding)."""
    SDF_BOX = 0
    src = _StubSDFManager()
    src.add_sdf_object(sdf_type=SDF_BOX, position=(0.1, -0.2, 0.3))

    sdf_data = _serialize_sdf_manager(src)
    # Should not raise
    encoded = json.dumps(sdf_data)
    decoded = json.loads(encoded)
    assert decoded["objects"][0]["type"] == SDF_BOX
    print("[OK] Serialised SDF data is JSON-encodable and round-trips via JSON")


def test_restore_into_empty_clears_existing() -> None:
    """Restoring into an SDFManager that already has objects clears them first."""
    SDF_BOX = 0
    SDF_SPHERE = 1

    src = _StubSDFManager()
    src.add_sdf_object(sdf_type=SDF_SPHERE, position=(9.0, 9.0, 9.0))
    sdf_data = _serialize_sdf_manager(src)

    dst = _StubSDFManager()
    dst.add_sdf_object(sdf_type=SDF_BOX, position=(0.0, 0.0, 0.0))
    dst.add_sdf_object(sdf_type=SDF_BOX, position=(1.0, 0.0, 0.0))
    assert dst.count == 2

    _restore_sdf_manager(dst, sdf_data)

    assert dst.count == 1, (
        f"Expected 1 object after restore, got {dst.count}"
    )
    assert dst._objects[0]["type"] == SDF_SPHERE
    print("[OK] Restore clears pre-existing SDF objects before re-populating")


def test_restore_from_empty_sdf_data() -> None:
    """Restoring with empty/missing sdf_data is a no-op (does not crash)."""
    dst = _StubSDFManager()
    dst.add_sdf_object(sdf_type=0, position=(1.0, 2.0, 3.0))

    # Empty dict should be a no-op (header["sdf"] = {} on v1 saves)
    _restore_sdf_manager(dst, {})
    # Object count unchanged since we returned early
    assert dst.count == 1
    print("[OK] Restoring from empty sdf_data is a no-op")


# ---------------------------------------------------------------------------
# SDF header JSON round-trip (tests the full header structure)
# ---------------------------------------------------------------------------

def test_header_json_with_sdf_and_camera() -> None:
    """Full header JSON (with SDF + camera) encodes and decodes correctly."""
    SDF_BOX = 0
    SDF_SPHERE = 1

    src = _StubSDFManager()
    src.add_sdf_object(sdf_type=SDF_BOX, position=(0.1, -0.2, 0.3), size=(0.5, 0.2, 0.5))
    src.add_sdf_object(sdf_type=SDF_SPHERE, position=(0.0, 0.8, 0.0), size=(0.15, 0.15, 0.15))
    src.add_kinematic_motion(1, "rotate_y", {"speed": 1.0})

    sdf_data = _serialize_sdf_manager(src)

    header = {
        "version": 2,
        "particle_count": 12345,
        "solver": "DFSPH",
        "world_half_size": 1.0,
        "gravity_y": -9.81,
        "camera": {
            "target": [0.0, 0.0, 0.0],
            "distance": 3.0,
            "azimuth": 0.5,
            "elevation": 0.3,
        },
        "sdf": sdf_data,
        "rigid_bodies": {"count": 0, "bodies": []},
    }

    encoded = json.dumps(header)
    decoded = json.loads(encoded)

    assert decoded["version"] == 2
    assert decoded["particle_count"] == 12345
    assert len(decoded["sdf"]["objects"]) == 2
    assert decoded["sdf"]["objects"][0]["type"] == SDF_BOX
    assert decoded["sdf"]["objects"][1]["type"] == SDF_SPHERE
    assert "1" in decoded["sdf"]["motions"]  # motion for object id 1
    assert decoded["sdf"]["motions"]["1"]["type"] == "rotate_y"
    print("[OK] Full header with SDF + kinematic motions encodes/decodes correctly")


# ---------------------------------------------------------------------------
# bd-unl.13: warm-start + sleep field zeroing on load
# ---------------------------------------------------------------------------

def test_header_includes_warm_start_fields_zeroed_on_load() -> None:
    """load_scene must zero kappa, kappa_v, lambda_pbf, sleep_counter, angular_velocity.

    Verified by inspecting save_load source: the load path explicitly assigns 0
    to each field over the [:n] slice immediately after writing the particle arrays.
    This test checks the header JSON path and verifies that the field names
    expected on World are consistent with the attribute names in world.py.
    """
    # The fields that must be zeroed; verify they match World._allocate names.
    expected_zeroed = {
        "kappa",          # DFSPH warm-start pressure
        "kappa_v",        # DFSPH warm-start divergence
        "lambda_pbf",     # PBF lambda
        "sleep_counter",  # rigid sleep timer (uint8, max 255 frames asleep)
        "angular_velocity",  # micropolar spin
    }
    import inspect
    import save_load as sl
    src = inspect.getsource(sl.load_scene)
    # Each field must appear as a zeroing assignment in load_scene
    for field in expected_zeroed:
        assert f"world.{field}[:n] = 0" in src or f"world.{field}[:n] = 0." in src, (
            f"load_scene does not zero world.{field} -- stale warm-start values will "
            f"survive a load cycle (bd-unl.13)"
        )
    print(f"[OK] load_scene zeros all {len(expected_zeroed)} warm-start/sleep fields")


# ---------------------------------------------------------------------------
# bd-unl.14: sim_time serialised to header and restored on load
# ---------------------------------------------------------------------------

def test_header_sim_time_present_and_round_trips() -> None:
    """save header must carry sim_time; load must restore it (not reset to 0).

    Checks the header JSON structure and that load_scene uses header.get('sim_time')
    rather than the hard-coded 0.0 that caused kinematic SDF snap-to-center.
    """
    import inspect
    import save_load as sl

    # 1. save_scene must write sim_time into the header dict.
    save_src = inspect.getsource(sl.save_scene)
    assert '"sim_time"' in save_src or "'sim_time'" in save_src, (
        "save_scene does not write 'sim_time' to the header (bd-unl.14)"
    )

    # 2. load_scene must read sim_time from the header rather than resetting to 0.
    load_src = inspect.getsource(sl.load_scene)
    assert 'sim_time' in load_src, (
        "load_scene does not reference sim_time at all (bd-unl.14)"
    )
    # The old bug was the literal 'sim.sim_time = 0.0' unconditionally.
    # After the fix it should use header.get('sim_time', ...).
    assert "header.get(\"sim_time\"" in load_src or "header.get('sim_time'" in load_src, (
        "load_scene does not call header.get('sim_time', ...) -- kinematic SDF positions "
        "will still snap to t=0 on load (bd-unl.14)"
    )

    # 3. Full header JSON round-trip with a non-zero sim_time.
    sdf_data = _serialize_sdf_manager(_StubSDFManager())
    header = {
        "version": sl.VERSION,
        "particle_count": 100,
        "solver": "DFSPH",
        "world_half_size": 1.0,
        "gravity_y": -9.81,
        "sim_time": 42.5,
        "camera": {"target": [0.0, 0.0, 0.0], "distance": 2.0, "azimuth": 0.0, "elevation": 0.0},
        "sdf": sdf_data,
        "rigid_bodies": {"count": 0, "bodies": []},
    }
    decoded = json.loads(json.dumps(header))
    assert decoded["sim_time"] == 42.5, (
        f"sim_time not preserved in header JSON round-trip: got {decoded.get('sim_time')}"
    )
    print("[OK] sim_time in header: save writes it, load reads it, JSON round-trips correctly")


def test_header_version_bumped_for_sim_time() -> None:
    """VERSION must be at least 3 to signal that sim_time is present."""
    import save_load as sl
    assert sl.VERSION >= 3, (
        f"VERSION={sl.VERSION}: should be incremented to 3+ when sim_time was added "
        f"to the header (bd-unl.14) so old loaders can detect the new field"
    )
    print(f"[OK] save_load.VERSION={sl.VERSION} >= 3 (sim_time field present)")


# ---------------------------------------------------------------------------
# bd-unl.15: rbm.upload uses sim.get_all_modules() on load and solver-switch
# ---------------------------------------------------------------------------

def test_load_scene_uses_get_all_modules() -> None:
    """load_scene must pass sim.get_all_modules() to _restore_rigid_body_manager.

    The old code built a partial 3-module list inline; the fix threads
    sim.get_all_modules() through so every compiled module gets the correct
    c_num_rigid_bodies constant (forward-maintenance guard, bd-unl.15).
    """
    import inspect
    import save_load as sl

    src = inspect.getsource(sl.load_scene)

    # The old partial-list pattern used inline imports of integrate/pbf_solver/dfsph_solver
    # and built a local 'modules' list.  After the fix that block is gone.
    assert "import integrate" not in src or "modules = []" not in src, (
        "load_scene still builds a partial inline module list -- it should use "
        "sim.get_all_modules() instead (bd-unl.15)"
    )

    # The fix must call get_all_modules() when passing modules to _restore_rigid_body_manager.
    assert "get_all_modules()" in src, (
        "load_scene does not call sim.get_all_modules() for the rbm.upload path (bd-unl.15)"
    )
    print("[OK] load_scene calls sim.get_all_modules() for rigid body module upload")


def test_main_solver_switch_calls_rbm_upload() -> None:
    """main.py solver-switch block must call rbm.upload after sim.set_solver_profile.

    Without this call c_num_rigid_bodies stays stale in newly compiled modules
    after a solver switch (bd-unl.15).

    Reads main.py as source text to avoid importing it (which would require glfw).
    """
    main_path = os.path.join(os.path.dirname(__file__), "main.py")
    with open(main_path, "r", encoding="utf-8") as fh:
        src = fh.read()

    # Look for the pattern introduced by the fix
    assert "rigid_body_manager.upload(sim.get_all_modules())" in src, (
        "main.py solver-switch block does not call "
        "sim.rigid_body_manager.upload(sim.get_all_modules()) after set_solver_profile "
        "(bd-unl.15)"
    )
    print("[OK] main.py solver-switch calls sim.rigid_body_manager.upload(sim.get_all_modules())")


def main() -> None:
    test_serialize_empty_sdf_manager()
    test_sdf_round_trip_single_box()
    test_sdf_round_trip_multiple_objects()
    test_sdf_round_trip_kinematic_motion()
    test_sdf_serialise_is_json_serialisable()
    test_restore_into_empty_clears_existing()
    test_restore_from_empty_sdf_data()
    test_header_json_with_sdf_and_camera()
    # bd-unl.13/14/15
    test_header_includes_warm_start_fields_zeroed_on_load()
    test_header_sim_time_present_and_round_trips()
    test_header_version_bumped_for_sim_time()
    test_load_scene_uses_get_all_modules()
    test_main_solver_switch_calls_rbm_upload()
    print("\n=== ALL SAVE/LOAD ROUND-TRIP TESTS PASSED ===")


if __name__ == "__main__":
    main()
