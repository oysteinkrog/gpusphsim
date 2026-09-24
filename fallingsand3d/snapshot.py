"""GPU-resident ring buffer for undo/replay snapshots.

All copies are GPU-GPU (cupy.copyto), ~0.5ms at 500K particles.
No CPU transfer in the hot path.

When a Simulation is passed in, each slot also holds the rigid body state
(the GPU body array and the body count), so undo rewinds the bodies together
with the fluid (bd-r5-review-n28.39).
"""

from __future__ import annotations

import cupy as cp
import numpy as np


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
            # Rigid body state, filled when capture() gets a Simulation.
            # num_bodies < 0 means the slot holds no rigid state.
            "rigid_bodies": None,
            "num_bodies": -1,
        }

    def capture(self, world, sim=None) -> None:
        """GPU-GPU snapshot of current world state into next ring slot.

        With ``sim``, also copies the rigid body array on the GPU. No host
        sync: the body count is a Python int and the copy is device to device.
        """
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
        _capture_rigid(slot, sim)

        self._write_idx = (self._write_idx + 1) % self.max_snapshots
        self._count = min(self._count + 1, self.max_snapshots)

    def tick(self, world, sim=None) -> None:
        """Call every frame. Captures snapshot every _frame_interval frames."""
        self._frame_counter += 1
        if self._frame_counter >= self._frame_interval:
            self._frame_counter = 0
            self.capture(world, sim=sim)

    @property
    def num_snapshots(self) -> int:
        return self._count

    def restore(self, world, steps_back: int = 1, sim=None) -> bool:
        """Restore world state from `steps_back` snapshots ago.

        With ``sim``, also restores the rigid body state captured in the slot.
        The caller must still call ``rigid_body_manager.mark_for_reinjection()``
        so the boundary block is rebuilt at the restored poses.

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

        # bd-r4-epic-x2j.6: zero warm-start fields so restored particles start fresh
        # (matches load_scene in save_load.py lines ~272-276).
        # Guard with hasattr for optional arrays that may not exist in all world configs.
        if hasattr(world, 'kappa'):
            world.kappa[:n] = 0
        if hasattr(world, 'kappa_v'):
            world.kappa_v[:n] = 0
        if hasattr(world, 'lambda_pbf'):
            world.lambda_pbf[:n] = 0
        if hasattr(world, 'sleep_counter'):
            world.sleep_counter[:n] = 0
        if hasattr(world, 'angular_velocity'):
            world.angular_velocity[:n] = 0

        # Rebuild _spawned_material_ids from restored packed_info
        # NOTE: .get() is acceptable here — undo is user-initiated, not hot path
        unique_mats = cp.unique(world.packed_info[:n] & 0xFF).get()
        world._spawned_material_ids = set(int(m) for m in unique_mats if m != 0)

        _restore_rigid(slot, sim)

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


def _copy_device_bytes(dst, src, nbytes: int) -> None:
    """Device-to-device byte copy on the current stream (no host sync).

    The rigid body array has a structured dtype, which cupy.copyto does not
    support, so copy the raw bytes instead.
    """
    if nbytes > 0:
        dst.data.copy_from_device_async(src.data, nbytes, cp.cuda.get_current_stream())


def _capture_rigid(slot: dict, sim) -> None:
    """Copy the rigid body state into ``slot`` (GPU to GPU)."""
    rbm = getattr(sim, "rigid_body_manager", None)
    if rbm is None:
        slot["num_bodies"] = -1
        return
    if slot["rigid_bodies"] is None or slot["rigid_bodies"].shape != rbm.d_rigid_bodies.shape:
        slot["rigid_bodies"] = cp.empty_like(rbm.d_rigid_bodies)
    nb = rbm.num_bodies
    _copy_device_bytes(slot["rigid_bodies"], rbm.d_rigid_bodies,
                       nb * rbm.d_rigid_bodies.dtype.itemsize)
    slot["num_bodies"] = nb


def _restore_rigid(slot: dict, sim) -> None:
    """Write the slot's rigid body state back into the manager.

    The slot holds only the dynamic body state, not the body list or boundary
    samples. If the body count changed since capture (a different preset),
    the rigid restore is skipped with a warning. The GPU copy is the source of
    truth: it is copied back on the device, then read once into the CPU body
    list so the CPU list and the next boundary injection use the restored
    poses. Undo is user initiated, so this one host sync is fine.
    """
    rbm = getattr(sim, "rigid_body_manager", None)
    if rbm is None or slot.get("num_bodies", -1) < 0:
        return
    nb = slot["num_bodies"]
    if nb != rbm.num_bodies:
        print(f"WARNING: snapshot has {nb} rigid bodies but the scene has "
              f"{rbm.num_bodies}; rigid bodies were not restored")
        return
    if nb == 0:
        return
    nbytes = nb * rbm.d_rigid_bodies.dtype.itemsize
    _copy_device_bytes(rbm.d_rigid_bodies, slot["rigid_bodies"], nbytes)
    host = np.empty(nb, dtype=rbm._bodies_cpu.dtype)
    cp.cuda.get_current_stream().synchronize()
    rbm.d_rigid_bodies.data.copy_to_host(host.ctypes.data, nbytes)
    rbm._bodies_cpu[:nb] = host
    rbm.upload(sim.get_all_modules())
