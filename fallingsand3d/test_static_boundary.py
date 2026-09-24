"""GPU tests for static_particle_boundary (sph_shared.cuh), bd-r5-review-n28.43.

The STATIC repulsion used to sum one push per STATIC neighbour inside 0.8h
and apply the raw sum. A particle 0.02 m over a flat floor moved about
0.028 m in one call. The fixed version uses a 0.5h band (one rest spacing)
and applies one correction: the deficit-weighted mean normal times the
largest single deficit.

Each test builds a small hashed grid on the host, calls the device function
through a one-thread wrapper kernel and reads back the result.
"""

import math
import os

import cupy
import numpy as np
import pytest

from materials import FLUID, STATIC, STONE, WATER

H = 0.04
DX = 0.02
MIN_DIST = 0.5 * H
TABLE_SIZE = 1 << 18
GRID_MIN = (-1.0, -1.0, -1.0)
RESTITUTION = 0.3

WRAPPER_SRC = r"""
#include "sph_shared.cuh"

extern "C" __global__
void k_cell_hash(const float4* pos, uint* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = pos[i];
    out[i] = spatialHash(calcGridCell(make_float3(p.x, p.y, p.z)));
}

extern "C" __global__
void k_static_boundary(
    const uint* cell_start, const uint* cell_end,
    const uint* packed_info, const float4* position,
    uint self_idx, float restitution,
    const float* vel_in, float* out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float4 p4 = position[self_idx];
    float3 pos = make_float3(p4.x, p4.y, p4.z);
    float3 vel = make_float3(vel_in[0], vel_in[1], vel_in[2]);
    static_particle_boundary(pos, vel, cell_start, cell_end,
                             packed_info, position, self_idx, restitution);
    out[0] = pos.x; out[1] = pos.y; out[2] = pos.z;
    out[3] = vel.x; out[4] = vel.y; out[5] = vel.z;
}
"""


def _packed(mat, behavior):
    return (mat & 0xFF) | ((behavior & 0x3) << 8)


@pytest.fixture(scope="module")
def module():
    kernel_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "physics", "kernels"
    )
    mod = cupy.RawModule(code=WRAPPER_SRC, options=("--std=c++14", f"-I{kernel_dir}"))

    grid = np.zeros(8, dtype=np.float32)
    grid[0:3] = GRID_MIN
    grid[3:6] = 1.0 / H
    grid_u = grid.view(np.uint32)
    grid_u[6] = TABLE_SIZE
    grid_u[7] = TABLE_SIZE - 1
    cupy.cuda.runtime.memcpy(
        mod.get_global("c_grid").ptr,
        grid.ctypes.data,
        grid.nbytes,
        cupy.cuda.runtime.memcpyHostToDevice,
    )
    # smoothing_length is the first SimParams field; the rest stay zero.
    sim_h = np.array([H, H * H], dtype=np.float32)
    cupy.cuda.runtime.memcpy(
        mod.get_global("c_sim").ptr,
        sim_h.ctypes.data,
        sim_h.nbytes,
        cupy.cuda.runtime.memcpyHostToDevice,
    )
    return mod


def _floor(n_side=7, y=0.0):
    """One-layer STATIC floor at height y, spacing DX, centred on x=z=0."""
    half = n_side // 2
    pts = [
        (ix * DX, y, iz * DX)
        for ix in range(-half, half + 1)
        for iz in range(-half, half + 1)
    ]
    return pts


def _wall_x(n_side=7, x=0.0):
    """One-layer STATIC wall in the plane x = const."""
    half = n_side // 2
    return [
        (x, iy * DX, iz * DX)
        for iy in range(-half, half + 1)
        for iz in range(-half, half + 1)
    ]


def _run(module, probe, statics, vel=(0.0, 0.0, 0.0)):
    """Call static_particle_boundary for one FLUID probe among STATIC particles."""
    pts = [probe] + list(statics)
    packed = [_packed(WATER, FLUID)] + [_packed(STONE, STATIC)] * len(statics)
    n = len(pts)
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, :3] = pts
    pos[:, 3] = 1.0

    d_pos = cupy.asarray(pos)
    d_hash = cupy.zeros(n, dtype=cupy.uint32)
    module.get_function("k_cell_hash")((1,), (256,), (d_pos, d_hash, np.int32(n)))
    hashes = d_hash.get()

    order = np.argsort(hashes, kind="stable")
    pos_s = pos[order]
    packed_s = np.asarray(packed, dtype=np.uint32)[order]
    hashes_s = hashes[order]
    self_idx = int(np.nonzero(order == 0)[0][0])

    cell_start = np.full(TABLE_SIZE, 0xFFFFFFFF, dtype=np.uint32)
    cell_end = np.zeros(TABLE_SIZE, dtype=np.uint32)
    for i, hsh in enumerate(hashes_s):
        if cell_start[hsh] == 0xFFFFFFFF:
            cell_start[hsh] = i
        cell_end[hsh] = i + 1

    out = cupy.zeros(6, dtype=cupy.float32)
    module.get_function("k_static_boundary")(
        (1,),
        (1,),
        (
            cupy.asarray(cell_start),
            cupy.asarray(cell_end),
            cupy.asarray(packed_s),
            cupy.asarray(pos_s),
            np.uint32(self_idx),
            np.float32(RESTITUTION),
            cupy.asarray(np.asarray(vel, dtype=np.float32)),
            out,
        ),
    )
    res = out.get()
    return res[:3] - np.asarray(probe, dtype=np.float32), res[3:]


def test_rest_spacing_over_floor_is_not_pushed(module):
    """A particle one rest spacing (0.02 m) over a flat floor stays put.

    The old 0.8h band summed five floor neighbours and moved it about 0.028 m.
    """
    disp, vel = _run(module, (0.0, DX, 0.0), _floor(), vel=(0.0, -0.5, 0.0))
    assert np.all(np.abs(disp) < 1e-6), f"moved by {disp}"
    assert np.allclose(vel, (0.0, -0.5, 0.0)), f"velocity changed to {vel}"


@pytest.mark.parametrize("y", [0.018, 0.015, 0.012, 0.005])
def test_push_over_floor_bounded_by_penetration(module, y):
    """Over a floor point, the push is exactly min_dist - y, straight up."""
    disp, _ = _run(module, (0.0, y, 0.0), _floor())
    expected = MIN_DIST - y
    assert disp[1] == pytest.approx(expected, abs=2e-6), (
        f"dy={disp[1]} expected {expected}"
    )
    assert disp[1] <= expected + 2e-6
    assert abs(disp[0]) < 1e-6 and abs(disp[2]) < 1e-6


def test_four_neighbours_give_one_push_not_four(module):
    """Probe above the centre of a floor square touches 4 floor particles.

    Each has the same deficit; the push is that deficit once, not the sum.
    """
    probe = (0.5 * DX, 0.5 * DX, 0.5 * DX)
    r = math.sqrt(3.0) * 0.5 * DX
    deficit = MIN_DIST - r
    disp, _ = _run(module, probe, _floor())
    assert disp[1] == pytest.approx(deficit, abs=2e-6), (
        f"dy={disp[1]} expected {deficit}"
    )
    assert abs(disp[0]) < 1e-6 and abs(disp[2]) < 1e-6


def test_velocity_reflected_only_into_wall(module):
    """Velocity into the floor is reflected with restitution; tangential kept."""
    _, vel = _run(module, (0.0, 0.012, 0.0), _floor(), vel=(0.3, -1.0, 0.0))
    assert vel[1] == pytest.approx(RESTITUTION * 1.0, abs=1e-5)
    assert vel[0] == pytest.approx(0.3, abs=1e-6)

    _, vel = _run(module, (0.0, 0.012, 0.0), _floor(), vel=(0.3, 1.0, 0.0))
    assert np.allclose(vel, (0.3, 1.0, 0.0), atol=1e-6), (
        f"outgoing velocity changed: {vel}"
    )


def _reference_push(probe, statics):
    """Host reference: deficit-weighted mean normal times the largest deficit."""
    p = np.asarray(probe, dtype=np.float64)
    q = np.asarray(statics, dtype=np.float64)
    r = p - q
    rl = np.linalg.norm(r, axis=1)
    inside = (rl < MIN_DIST) & (rl > 1e-6)
    deficit = MIN_DIST - rl[inside]
    d = (deficit[:, None] * r[inside] / rl[inside][:, None]).sum(axis=0)
    return deficit.max() * d / np.linalg.norm(d), deficit.max(), int(inside.sum())


def test_corner_push_never_exceeds_deepest_penetration(module):
    """Floor plus wall: one correction, length equal to the deepest deficit."""
    statics = _floor(y=0.0) + [p for p in _wall_x(x=-0.06) if p[1] > 0.0]
    probe = (-0.047, 0.013, 0.0)
    expected, deepest, n_inside = _reference_push(probe, statics)
    assert n_inside >= 3, "probe should touch both floor and wall"
    disp, _ = _run(module, probe, statics)
    length = float(np.linalg.norm(disp))
    assert length == pytest.approx(deepest, abs=2e-6), (
        f"|push|={length} expected {deepest}"
    )
    assert np.allclose(disp, expected, atol=2e-6), f"push {disp} expected {expected}"
    assert disp[0] > 0.0 and disp[1] > 0.0, (
        f"push {disp} should point out of the corner"
    )
