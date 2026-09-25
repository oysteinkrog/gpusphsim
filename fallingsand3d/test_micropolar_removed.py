"""The micropolar angular velocity is gone (bd-r5-review-n28.140).

Nothing read it, yet the sort, integrate, PBF/DFSPH finalize and spawn kernels
moved 64 B per particle per substep for it. These checks stop it from coming
back as dead traffic.
"""

import pathlib

import pytest

KERNEL_DIR = pathlib.Path(__file__).parent / "physics" / "kernels"


def test_no_kernel_moves_particle_angular_velocity():
    hits = []
    for path in sorted(KERNEL_DIR.glob("*.cu")) + sorted(KERNEL_DIR.glob("*.cuh")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "angular_velocity" in line or "MICROPOLAR" in line:
                hits.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not hits, (
        "micropolar angular velocity is back in the kernels:\n" + "\n".join(hits)
    )


@pytest.mark.gpu
def test_world_allocates_no_angular_velocity_storage():
    from world import World

    w = World(10_000)
    assert w.angular_velocity.size == 0
    assert w.sorted_angular_velocity.size == 0
