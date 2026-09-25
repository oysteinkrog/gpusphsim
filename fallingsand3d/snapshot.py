"""GPU-resident ring buffer for undo/replay snapshots.

All copies are GPU-GPU (cupy.copyto), ~0.5ms at 500K particles.
No CPU transfer in the hot path. The ring is sized to a VRAM budget
(256 MB by default) and all slots are allocated up front
(bd-r5-review-n28.113).

When a Simulation is passed in, each slot also holds the rigid body state
(the GPU body array and the body count), so undo rewinds the bodies together
with the fluid (bd-r5-review-n28.39). It also holds a CPU copy of the SDF
object list and its kinematic motions, so undo removes SDF objects added
after the capture and puts back moved ones (bd-r5-review-n28.61).
"""

from __future__ import annotations

import cupy as cp
import numpy as np


# Bytes per particle held in one slot. position, velocity and particle_dye are
# float4 (16 B each), packed_info and temperature are 4 B, and health,
# lifetime and mass are stored as float16 (2 B each). Temperature stays
# float32: fp16 has a 1 K step above 1024 K and 4 K near T_MAX = 5000 K, which
# would break the 0.5 K round trip for lava and fire (bd-r5-review-n28.113).
SLOT_BYTES_PER_PARTICLE = 16 + 16 + 16 + 4 + 4 + 2 + 2 + 2  # = 62

# Per-slot allowance for state that does not scale with particle count: the
# rigid body array (MAX_RIGID_BODIES = 8 bodies, well under 1 KB) and the CPU
# SDF copy (not in VRAM). 64 KB is a safe upper bound.
SLOT_FIXED_BYTES = 64 << 10

# Default VRAM budget for the whole ring.
DEFAULT_BUDGET_BYTES = 256 << 20


def slots_for_budget(max_snapshots: int, max_particles: int,
                     budget_bytes: int = DEFAULT_BUDGET_BYTES) -> int:
    """Number of ring slots that fit in ``budget_bytes``.

    Clamped to [2, max_snapshots] (or to max_snapshots when that is below 2),
    so undo always has at least two slots even when one slot alone is over
    budget. At 1M particles and 256 MB this gives 4 slots; at 250K, 12.
    """
    per_slot = SLOT_BYTES_PER_PARTICLE * max(int(max_particles), 0) + SLOT_FIXED_BYTES
    fit = int(budget_bytes) // per_slot
    lo = min(2, max_snapshots)
    return max(lo, min(fit, max_snapshots))


class SnapshotRing:
    """Fixed-size ring buffer of GPU snapshots for undo/replay.

    ``max_snapshots`` is the most slots wanted; the ring holds fewer when
    ``budget_bytes`` of VRAM cannot fit that many at ``max_particles``. After
    construction ``max_snapshots`` is the slot count actually used. All slots
    are allocated in ``__init__``, so ``capture()`` never allocates particle
    arrays inside the render loop.
    """

    def __init__(self, max_snapshots: int = 12, max_particles: int = 500_000,
                 budget_bytes: int = DEFAULT_BUDGET_BYTES):
        self.requested_snapshots = max_snapshots
        self.budget_bytes = budget_bytes
        self.max_snapshots = slots_for_budget(max_snapshots, max_particles, budget_bytes)
        self.max_particles = max_particles
        self._slots: list[dict | None] = [self._alloc_slot() for _ in range(self.max_snapshots)]
        self._write_idx = 0  # next slot to write
        self._count = 0      # how many valid snapshots
        # Cadence. With a sim time, capture every snapshot_interval_s seconds
        # of simulated time, so paused frames add nothing and the spacing is
        # the same at any speed. Without one (tests), fall back to counting
        # frames (bd-r5-review-n28.85).
        self.snapshot_interval_s = 5.0  # = 300 frames at 60 fps and 1x speed
        self._frame_interval = 300  # fallback: capture every N frames
        self._frame_counter = 0
        self._time_accum = 0.0
        self._last_tick_time: float | None = None
        self._last_capture_time: float | None = None

    def _alloc_slot(self) -> dict:
        n = self.max_particles
        return {
            "position": cp.empty((n, 4), dtype=cp.float32),
            "velocity": cp.empty((n, 4), dtype=cp.float32),
            "packed_info": cp.empty(n, dtype=cp.uint32),
            "temperature": cp.empty(n, dtype=cp.float32),
            # float16: health is 0..1, lifetime a few seconds, mass about
            # 0.01; fp16 keeps them within 0.05% (see SLOT_BYTES_PER_PARTICLE).
            "health": cp.empty(n, dtype=cp.float16),
            "lifetime": cp.empty(n, dtype=cp.float16),
            "mass": cp.empty(n, dtype=cp.float16),
            "particle_dye": cp.empty((n, 4), dtype=cp.float32),
            "high_water": 0,
            # Rigid body state, filled when capture() gets a Simulation.
            # num_bodies < 0 means the slot holds no rigid state.
            "rigid_bodies": None,
            "num_bodies": -1,
            # SDF objects and motions (CPU deep copy), None without a sim.
            "sdf_state": None,
        }

    def capture(self, world, sim=None, sim_time: float | None = None) -> None:
        """GPU-GPU snapshot of current world state into next ring slot.

        With ``sim``, also copies the rigid body array on the GPU and deep
        copies the SDF object list on the CPU. No host sync: the body count is
        a Python int and the rigid copy is device to device.

        With ``sim_time``, the capture is skipped when it equals the time of
        the last capture, so a paused sim never fills the ring with copies of
        one state.
        """
        n = world._high_water
        if n == 0:
            return
        if sim_time is not None:
            if sim_time == self._last_capture_time:
                return
            self._last_capture_time = sim_time

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
        sdf = getattr(sim, "sdf_manager", None)
        slot["sdf_state"] = sdf.snapshot_state() if sdf is not None else None

        self._write_idx = (self._write_idx + 1) % self.max_snapshots
        self._count = min(self._count + 1, self.max_snapshots)

    def tick(self, world, sim=None, sim_time: float | None = None) -> None:
        """Call on frames that ran a substep.

        ``sim_time`` defaults to ``sim.sim_time`` when ``sim`` has one. With a
        time, a snapshot is taken every ``snapshot_interval_s`` of simulated
        time; a tick with the same time as the last one (paused) adds nothing,
        and a time that went backwards (scene reset) restarts the count.
        Without a time, a snapshot is taken every ``_frame_interval`` ticks.
        """
        if sim_time is None:
            sim_time = getattr(sim, "sim_time", None)
        if sim_time is None:
            self._frame_counter += 1
            if self._frame_counter >= self._frame_interval:
                self._frame_counter = 0
                self.capture(world, sim=sim)
            return
        sim_time = float(sim_time)
        last = self._last_tick_time
        self._last_tick_time = sim_time
        if last is None or sim_time <= last:
            return
        self._time_accum += sim_time - last
        if self._time_accum >= self.snapshot_interval_s:
            self._time_accum = 0.0
            self.capture(world, sim=sim, sim_time=sim_time)

    @property
    def num_snapshots(self) -> int:
        return self._count

    def restore(self, world, steps_back: int = 1, sim=None) -> bool:
        """Restore world state from `steps_back` snapshots ago.

        With ``sim``, also restores the rigid body state and the SDF objects
        captured in the slot. The SDF list is marked dirty and uploads before
        the next frame step.
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

        # The slot holds no color, so rebuild it from the material table; a
        # paused undo would otherwise show the current colors
        # (bd-r5-review-n28.63).
        world.refresh_color(n)

        # bd-r4-epic-x2j.6: zero warm-start fields so restored particles start fresh
        # (matches load_scene in save_load.py lines ~272-276).
        # Guard with hasattr for optional arrays that may not exist in all world configs.
        if hasattr(world, 'kappa'):
            world.kappa[:n] = 0
        if hasattr(world, 'kappa_v'):
            world.kappa_v[:n] = 0
        if hasattr(world, 'sleep_counter'):
            world.sleep_counter[:n] = 0

        # Rebuild _spawned_material_ids from restored packed_info
        # NOTE: .get() is acceptable here — undo is user-initiated, not hot path
        unique_mats = cp.unique(world.packed_info[:n] & 0xFF).get()
        world._spawned_material_ids = set(int(m) for m in unique_mats if m != 0)

        _restore_rigid(slot, sim)
        sdf = getattr(sim, "sdf_manager", None)
        if sdf is not None and slot.get("sdf_state") is not None:
            sdf.restore_state(slot["sdf_state"])

        # Rewind ring buffer so future snapshots don't contain stale "future" state
        # Drop the +1: without it, each successive undo steps back one more slot.
        # With +1, _write_idx doesn't move for steps_back=1, so every undo reads
        # the same slot and multi-undo is broken.
        self._write_idx = (self._write_idx - steps_back) % self.max_snapshots
        self._count = max(0, self._count - steps_back)
        # Start the next interval from the restore, not from the old capture.
        self._time_accum = 0.0
        self._frame_counter = 0

        return True

    def clear(self) -> None:
        """Reset ring buffer (keeps allocated GPU memory)."""
        self._count = 0
        self._write_idx = 0
        self._frame_counter = 0
        self._time_accum = 0.0
        self._last_tick_time = None
        self._last_capture_time = None


def after_restore(world, sim, renderer) -> None:
    """Bring the sim and renderer in line with a world just restored from the ring.

    Shared by Ctrl+Z and the timeline Restore button. It keeps the snapshot
    ring: restore() already rewound it, so older slots stay reachable
    (bd-r5-review-n28.85). The caller resets its own spawner state.
    """
    # The restored arrays may hold the boundary block at an old offset;
    # rebuild it next substep (bd-r5-review-n28.22).
    sim.rigid_body_manager.mark_for_reinjection()
    renderer.num_active = world._high_water
    sim._invalidate_graphs()
    sim.notify_scene_changed()  # stale sort perm after restore (CRIT-5)
    sim.reset_spawn_damping()
    # Foam is not in the snapshot; clear it so no stale spray floats over the
    # rewound scene (bd-r5-review-n28.61).
    world.foam_count.fill(0)
    sim._last_foam_count = 0
    with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf, renderer.cuda_vel as vel_buf:
        sim.copy_to_vbos(pos_buf, col_buf, vel_buf)


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
