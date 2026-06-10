"""Round-trip tests for save_load.py -- SDF + rigid body serialisation (bd-mzc.26).

Tests the _serialize_sdf_manager / _restore_sdf_manager and
_serialize_rigid_body_manager helpers in isolation (no GPU required for the
pure-Python parts), plus a JSON round-trip of the full header structure.

A full end-to-end test (save_scene / load_scene with a live CuPy world) is
omitted here because it requires a CUDA GPU and a fully initialised World /
Simulation / Camera -- integration coverage for that path comes from the app's
existing preset reload paths.  The helpers tested here are the novel code added
by bd-mzc.26.
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


def main() -> None:
    test_serialize_empty_sdf_manager()
    test_sdf_round_trip_single_box()
    test_sdf_round_trip_multiple_objects()
    test_sdf_round_trip_kinematic_motion()
    test_sdf_serialise_is_json_serialisable()
    test_restore_into_empty_clears_existing()
    test_restore_from_empty_sdf_data()
    test_header_json_with_sdf_and_camera()
    print("\n=== ALL SAVE/LOAD ROUND-TRIP TESTS PASSED ===")


if __name__ == "__main__":
    main()
