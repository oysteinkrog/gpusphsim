"""
test_odd_viscosity.py -- Falsification of the "Scan-Order Hall Fluid" theory
against the REAL fallingsand3d SPH sim.

Background: see odd_viscosity_probe.py (a GPU-free CPU rig that passes all five
signatures). This file runs the analogous measurement inside the actual
weakly-compressible SPH solver on a CUDA GPU, so the theory can be confirmed or
falsified on the code that ships.

THE CLAIM (as it applies to this repo)
--------------------------------------
The vorticity-confinement force in step2.cu uses `omega_j` rather than
`(omega_j - omega_i)` (documented as PROBLEMS.md item G3). That makes it a
PARITY-BROKEN stress. The theory says a parity-broken stress is, at leading
order, ODD (Hall) VISCOSITY:  eta_o * lap(R v). Its fingerprint is that it
converts a symmetric, zero-net-angular-momentum flow into one with a
SIGN-DEFINITE net angular momentum L_z -- a spontaneous chiral spin-up that a
truly parity-symmetric solver cannot produce.

We initialize a Taylor-Green velocity field in a gravity-free water block. By
construction it has:
    * zero net linear momentum
    * zero net angular momentum L_z
    * nonzero local vorticity everywhere (so the confinement term is active)

A parity-symmetric method must keep net L_z at the noise floor forever. The
theory predicts net L_z DRIFTS AWAY FROM ZERO once the parity-broken
confinement term is switched on, and that the drift grows with the confinement
coefficient (which plays the role of eta_o).

FALSIFICATION CONDITIONS (any one kills the theory):
  F1. |L_z| at vorticity_epsilon = 0 is NOT at the noise floor
      (=> the spin-up is not caused by the confinement term).
  F2. |L_z| does not increase monotonically with vorticity_epsilon
      (=> the effect does not scale like a viscosity coefficient).
  F3. The transverse shear-coupling test shows NO induced perpendicular
      momentum flux, or one that does not grow with the confinement coefficient.

The absolute thresholds below (NOISE_FLOOR, MIN_DRIFT) are CALIBRATION CONSTANTS
to be pinned on the first real GPU run; the STRUCTURAL asserts (null control,
monotonicity) are the actual physics and are threshold-light.

Requirements: cupy, numpy, an NVIDIA GPU with CUDA 12.x. Skips cleanly otherwise.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
try:
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device", allow_module_level=True)
except Exception:
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

import step2
from world import World, DEFAULT_SPACING
from materials import WATER
from simulation import Simulation
from solver_profiles import PROFILES


# ------------------------------- calibration -------------------------------
# Pin these on the first real GPU run (they are printed by every test).
NOISE_FLOOR = 5e-6      # |L_z| a parity-symmetric run may reach from FP noise
MIN_DRIFT = 5e-5        # |L_z| the parity-broken run must exceed to count as real
BLOCK = 0.4             # side length of the fluid cube (m)
FRAMES = 30             # frames to integrate


def _build_sim(vorticity_epsilon: float):
    """Gravity-free water cube with a chosen vorticity-confinement coefficient."""
    world = World(max_particles=200_000)
    lo = 0.5 - BLOCK / 2
    hi = 0.5 + BLOCK / 2
    n = world.spawn_cube((lo, lo, lo), (hi, hi, hi), WATER, spacing=DEFAULT_SPACING)
    assert n > 2000, f"expected a dense block, got {n} particles"

    sim = Simulation(world, world_half_size=1.0)
    sim.set_solver_profile(PROFILES["WCSPH"])
    sim.set_gravity(0.0)  # isolate the confinement term from buoyancy/settling
    # Override the confinement coefficient (this is the eta_o dial).
    step2.upload_granular_params(step2.build_granular_params(
        vorticity_epsilon=np.float32(vorticity_epsilon),
    ))
    return world, sim, n


def _seed_taylor_green(world: World, n: int, amp: float = 0.3):
    """Symmetric divergence-free field: zero net linear & angular momentum."""
    pos = cupy.asnumpy(world.position[:n, :3])
    c = 0.5
    x = pos[:, 0] - c
    y = pos[:, 1] - c
    kk = 2.0 * np.pi / BLOCK
    # u =  A sin(kx) cos(ky),  v = -A cos(kx) sin(ky)  -> div-free, L_z net = 0
    u = amp * np.sin(kk * x) * np.cos(kk * y)
    v = -amp * np.cos(kk * x) * np.sin(kk * y)
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, 0] = u
    vel[:, 1] = v
    world.velocity[:n] = cupy.asarray(vel)
    world.veleval[:n] = cupy.asarray(vel)


def _net_Lz(world: World, n: int) -> float:
    """Net z-angular-momentum about the block centre (the chirality observable)."""
    pos = cupy.asnumpy(world.position[:n, :3]).astype(np.float64)
    vel = cupy.asnumpy(world.velocity[:n, :3]).astype(np.float64)
    mass = cupy.asnumpy(world.mass[:n]).astype(np.float64)
    x = pos[:, 0] - 0.5
    y = pos[:, 1] - 0.5
    Lz = np.sum(mass * (x * vel[:, 1] - y * vel[:, 0]))
    return float(Lz)


def _run(vorticity_epsilon: float) -> float:
    world, sim, n = _build_sim(vorticity_epsilon)
    _seed_taylor_green(world, n)
    L0 = _net_Lz(world, n)
    for _ in range(FRAMES):
        sim.step_frame()
    n = world._high_water
    L1 = _net_Lz(world, n)
    print(f"  eps={vorticity_epsilon:<7.4f}  L_z: {L0:+.3e} -> {L1:+.3e}  "
          f"(drift {L1 - L0:+.3e})")
    return L1 - L0


def test_null_control_parity_symmetric():
    """F1: with confinement off, the symmetric flow must NOT spontaneously spin."""
    drift0 = abs(_run(0.0))
    print(f"  null-control |drift| = {drift0:.3e}  (noise floor {NOISE_FLOOR:.1e})")
    assert drift0 < NOISE_FLOOR, (
        "parity-symmetric run spun up on its own -> the spin-up is NOT explained "
        "by the confinement term; theory's causal claim is falsified (F1)."
    )


def test_chiral_spinup_scales_with_confinement():
    """F2: parity-broken confinement induces net L_z that grows with eta_o."""
    eps = [0.0, 0.05, 0.10]
    drifts = [abs(_run(e)) for e in eps]
    print(f"  |drift| vs eps: {list(zip(eps, [f'{d:.3e}' for d in drifts]))}")

    # structural (threshold-light) physics:
    assert drifts[0] < NOISE_FLOOR, "null control failed (F1)"
    assert drifts[-1] > MIN_DRIFT, (
        "no measurable chiral spin-up even at max confinement -> theory falsified (F3)."
    )
    assert drifts[1] < drifts[2], (
        "chiral spin-up does not increase with the confinement coefficient -> the "
        "effect does not scale like a viscosity; theory falsified (F2)."
    )


def test_transverse_shear_coupling():
    """
    F3 (SPH analogue of CPU-rig TEST 1): a pure shear (v_y varying in x) should,
    under a parity-broken stress, induce a perpendicular momentum component whose
    magnitude grows with the confinement coefficient. A parity-symmetric run
    should induce ~none.
    """
    def shear_response(vorticity_epsilon):
        world, sim, n = _build_sim(vorticity_epsilon)
        pos = cupy.asnumpy(world.position[:n, :3])
        kk = 2.0 * np.pi / BLOCK
        vel = np.zeros((n, 4), dtype=np.float32)
        vel[:, 1] = 0.3 * np.sin(kk * (pos[:, 0] - 0.5))  # v_y(x): pure shear
        world.velocity[:n] = cupy.asarray(vel)
        world.veleval[:n] = cupy.asarray(vel)
        for _ in range(5):
            sim.step_frame()
        n2 = world._high_water
        v2 = cupy.asnumpy(world.velocity[:n2, :3])
        # transverse (perpendicular) momentum magnitude induced in v_x
        return float(np.sqrt(np.mean(v2[:, 0] ** 2)))

    r0 = shear_response(0.0)
    r1 = shear_response(0.10)
    print(f"  transverse RMS v_x: symmetric={r0:.3e}  parity-broken={r1:.3e}")
    assert r1 > r0, (
        "parity-broken stress did not enhance the transverse (odd-viscosity) "
        "response relative to the symmetric control -> theory falsified (F3)."
    )


if __name__ == "__main__":
    # Allow running directly on the CUDA box: `python test_odd_viscosity.py`
    print("== null control ==");            test_null_control_parity_symmetric()
    print("== spin-up scaling ==");         test_chiral_spinup_scales_with_confinement()
    print("== transverse coupling ==");     test_transverse_shear_coupling()
    print("\nAll structural predictions held on the real SPH sim.")
