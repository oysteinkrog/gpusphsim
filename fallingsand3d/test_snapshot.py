"""Tests for SnapshotRing undo ring-buffer correctness.

Verifies that:
1. undo(1) restores the state written before the most recent capture (not the
   same slot we just wrote).
2. Successive undo calls walk backward through distinct prior states.
3. Restored _write_idx moves back so a second undo reads a different (older)
   slot rather than repeating the same slot.

No CUDA GPU required -- stubs out the CuPy dependency with numpy arrays.
"""

import sys
import types
import unittest
import numpy as np


# ---------------------------------------------------------------------------
# Minimal CuPy stub so snapshot.py can be imported without a GPU
# ---------------------------------------------------------------------------

_cupy_stub = types.ModuleType("cupy")


def _empty(shape, dtype=np.float32):
    if isinstance(shape, int):
        shape = (shape,)
    return np.zeros(shape, dtype=dtype)


def _zeros(shape, dtype=np.float32):
    if isinstance(shape, int):
        shape = (shape,)
    return np.zeros(shape, dtype=dtype)


def _copyto(dst, src):
    np.copyto(dst, src)


class _GetArray(np.ndarray):
    """Numpy array subclass with a .get() method (mirrors CuPy interface)."""

    def get(self):
        return np.asarray(self)


def _unique(arr):
    result = np.unique(arr)
    return result.view(_GetArray)


_cupy_stub.empty = _empty
_cupy_stub.zeros = _zeros
_cupy_stub.copyto = _copyto
_cupy_stub.unique = _unique
_cupy_stub.float32 = np.float32
_cupy_stub.uint32 = np.uint32

# Import snapshot bound to whatever cupy is present. The tests patch
# ``snapshot.cp`` per-test (setUp/tearDown) so SnapshotRing uses the numpy stub
# regardless of import order, and WITHOUT replacing sys.modules['cupy'] (which
# previously polluted other test files when run in the full suite: a stale
# `setdefault` was a no-op once the real cupy was loaded, leaving snapshot bound
# to real cupy and `cp.copyto(numpy, numpy)` failing).
import snapshot  # noqa: E402
from snapshot import SnapshotRing  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal world stub
# ---------------------------------------------------------------------------

MAX_P = 16


class _World:
    """Minimal world object that SnapshotRing.capture / restore needs."""

    def __init__(self, n: int = MAX_P):
        self.max_particles = n
        self._high_water = n
        self.position = np.zeros((n, 4), dtype=np.float32)
        self.velocity = np.zeros((n, 4), dtype=np.float32)
        self.veleval = np.zeros((n, 4), dtype=np.float32)
        self.packed_info = np.zeros(n, dtype=np.uint32)
        self.temperature = np.zeros(n, dtype=np.float32)
        self.health = np.ones(n, dtype=np.float32)
        self.lifetime = np.zeros(n, dtype=np.float32)
        self.mass = np.ones(n, dtype=np.float32)
        self.particle_dye = np.zeros((n, 4), dtype=np.float32)
        self._spawned_material_ids: set = set()

    def set_marker(self, value: float) -> None:
        """Write a unique value into position[0] so states are distinguishable."""
        self.position[0, 0] = value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSnapshotRingUndo(unittest.TestCase):

    def setUp(self):
        # Force SnapshotRing to use the numpy stub regardless of whether the
        # real cupy is already imported. Scoped per-test so it never leaks.
        self._real_cp = snapshot.cp
        snapshot.cp = _cupy_stub

    def tearDown(self):
        snapshot.cp = self._real_cp

    def _make_ring(self, max_snapshots: int = 6) -> SnapshotRing:
        return SnapshotRing(max_snapshots=max_snapshots, max_particles=MAX_P)

    # ------------------------------------------------------------------
    # Test 1: undo(1) restores the state from one capture ago (not the
    # current state or an even older one).
    # ------------------------------------------------------------------
    def test_undo_restores_previous_state(self):
        ring = self._make_ring()
        world = _World()

        world.set_marker(1.0)
        ring.capture(world)  # snapshot A

        world.set_marker(2.0)
        ring.capture(world)  # snapshot B  ← current

        # Mutate world so it differs from both snapshots
        world.set_marker(99.0)

        ok = ring.restore(world, steps_back=1)
        self.assertTrue(ok, "restore should succeed when snapshots exist")
        self.assertAlmostEqual(float(world.position[0, 0]), 2.0,
                               msg="undo(1) should restore snapshot B (marker=2)")

    # ------------------------------------------------------------------
    # Test 2: consecutive undo calls produce distinct (progressively older)
    # states -- the ring pointer actually walks back.
    # ------------------------------------------------------------------
    def test_successive_undos_walk_back(self):
        ring = self._make_ring(max_snapshots=8)
        world = _World()

        markers = [1.0, 2.0, 3.0, 4.0]
        for m in markers:
            world.set_marker(m)
            ring.capture(world)

        # After 4 captures, undo to most recent (4.0)
        ring.restore(world, steps_back=1)
        v_first = float(world.position[0, 0])
        self.assertAlmostEqual(v_first, 4.0,
                               msg="first undo should yield marker 4.0")

        # Second undo — must yield 3.0, not 4.0 again
        ring.restore(world, steps_back=1)
        v_second = float(world.position[0, 0])
        self.assertAlmostEqual(v_second, 3.0,
                               msg="second undo should yield marker 3.0, not repeat 4.0")

        # Third undo — must yield 2.0
        ring.restore(world, steps_back=1)
        v_third = float(world.position[0, 0])
        self.assertAlmostEqual(v_third, 2.0,
                               msg="third undo should yield marker 2.0")

    # ------------------------------------------------------------------
    # Test 3: _write_idx decrements after each restore so the ring pointer
    # doesn't stay frozen in place.
    # ------------------------------------------------------------------
    def test_write_idx_moves_after_restore(self):
        ring = self._make_ring(max_snapshots=6)
        world = _World()

        world.set_marker(10.0)
        ring.capture(world)
        world.set_marker(20.0)
        ring.capture(world)

        before = ring._write_idx
        ring.restore(world, steps_back=1)
        after = ring._write_idx

        # The pointer must move (modulo ring size); it must not stay the same.
        self.assertNotEqual(
            before % ring.max_snapshots,
            after % ring.max_snapshots,
            msg="_write_idx must change after restore so a second undo reads a different slot",
        )

    # ------------------------------------------------------------------
    # Test 4: restore returns False when no snapshots exist.
    # ------------------------------------------------------------------
    def test_restore_returns_false_when_empty(self):
        ring = self._make_ring()
        world = _World()
        self.assertFalse(ring.restore(world, steps_back=1))

    # ------------------------------------------------------------------
    # Test 5: ring wraps around correctly (more captures than max_snapshots).
    # ------------------------------------------------------------------
    def test_ring_wrap_around(self):
        ring = self._make_ring(max_snapshots=3)
        world = _World()

        for i in range(5):
            world.set_marker(float(i + 1))
            ring.capture(world)

        # After 5 captures in a ring-3, the last 3 are 3, 4, 5.
        # undo(1) should give 5.0 (most recent).
        ring.restore(world, steps_back=1)
        self.assertAlmostEqual(float(world.position[0, 0]), 5.0,
                               msg="undo in wrapped ring should give most recent capture")


if __name__ == "__main__":
    unittest.main()
