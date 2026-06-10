"""GPU-resident ring buffer for undo/replay snapshots.

All copies are GPU-GPU (cupy.copyto), ~0.5ms at 500K particles.
No CPU transfer in the hot path.
"""

from __future__ import annotations

import cupy as cp


class SnapshotRing:
    """Fixed-size ring buffer of GPU snapshots for undo/replay."""

    def __init__(self, max_snapshots: int = 12, max_particles: int = 500_000):
        self.max_snapshots = max_snapshots
        self.max_particles = max_particles
        self._slots: list[dict | None] = [None] * max_snapshots
        self._write_idx = 0  # next slot to write
        self._count = 0      # how many valid snapshots
        self._frame_interval = 300  # capture every N frames
        self._frame_counter = 0

    def _alloc_slot(self) -> dict:
        n = self.max_particles
        return {
            "position": cp.empty((n, 4), dtype=cp.float32),
            "velocity": cp.empty((n, 4), dtype=cp.float32),
            "packed_info": cp.empty(n, dtype=cp.uint32),
            "temperature": cp.empty(n, dtype=cp.float32),
            "health": cp.empty(n, dtype=cp.float32),
            "lifetime": cp.empty(n, dtype=cp.float32),
            "mass": cp.empty(n, dtype=cp.float32),
            "particle_dye": cp.empty((n, 4), dtype=cp.float32),
            "high_water": 0,
        }

    def capture(self, world) -> None:
        """GPU-GPU snapshot of current world state into next ring slot."""
        n = world._high_water
        if n == 0:
            return

        slot = self._slots[self._write_idx]
        if slot is None:
            slot = self._alloc_slot()
            self._slots[self._write_idx] = slot

        cp.copyto(slot["position"][:n], world.position[:n])
        cp.copyto(slot["velocity"][:n], world.velocity[:n])
        cp.copyto(slot["packed_info"][:n], world.packed_info[:n])
        cp.copyto(slot["temperature"][:n], world.temperature[:n])
        cp.copyto(slot["health"][:n], world.health[:n])
        cp.copyto(slot["lifetime"][:n], world.lifetime[:n])
        cp.copyto(slot["mass"][:n], world.mass[:n])
        cp.copyto(slot["particle_dye"][:n], world.particle_dye[:n])
        slot["high_water"] = n

        self._write_idx = (self._write_idx + 1) % self.max_snapshots
        self._count = min(self._count + 1, self.max_snapshots)

    def tick(self, world) -> None:
        """Call every frame. Captures snapshot every _frame_interval frames."""
        self._frame_counter += 1
        if self._frame_counter >= self._frame_interval:
            self._frame_counter = 0
            self.capture(world)

    @property
    def num_snapshots(self) -> int:
        return self._count

    def restore(self, world, steps_back: int = 1) -> bool:
        """Restore world state from `steps_back` snapshots ago.

        Returns True if successful, False if no snapshot available.
        """
        if steps_back < 1 or steps_back > self._count:
            return False

        idx = (self._write_idx - steps_back) % self.max_snapshots
        slot = self._slots[idx]
        if slot is None:
            return False

        n = slot["high_water"]

        # Clear world then restore
        world.packed_info[:] = 0
        world._high_water = 0

        cp.copyto(world.position[:n], slot["position"][:n])
        cp.copyto(world.velocity[:n], slot["velocity"][:n])
        cp.copyto(world.packed_info[:n], slot["packed_info"][:n])
        cp.copyto(world.temperature[:n], slot["temperature"][:n])
        cp.copyto(world.health[:n], slot["health"][:n])
        cp.copyto(world.lifetime[:n], slot["lifetime"][:n])
        cp.copyto(world.mass[:n], slot["mass"][:n])
        cp.copyto(world.particle_dye[:n], slot["particle_dye"][:n])
        cp.copyto(world.veleval[:n], world.velocity[:n])
        world._high_water = n

        # Rebuild _spawned_material_ids from restored packed_info
        # NOTE: .get() is acceptable here — undo is user-initiated, not hot path
        unique_mats = cp.unique(world.packed_info[:n] & 0xFF).get()
        world._spawned_material_ids = set(int(m) for m in unique_mats if m != 0)

        # Rewind ring buffer so future snapshots don't contain stale "future" state
        # Drop the +1: without it, each successive undo steps back one more slot.
        # With +1, _write_idx doesn't move for steps_back=1, so every undo reads
        # the same slot and multi-undo is broken.
        self._write_idx = (self._write_idx - steps_back) % self.max_snapshots
        self._count = max(0, self._count - steps_back)

        return True

    def clear(self) -> None:
        """Reset ring buffer (keeps allocated GPU memory)."""
        self._count = 0
        self._write_idx = 0
        self._frame_counter = 0
