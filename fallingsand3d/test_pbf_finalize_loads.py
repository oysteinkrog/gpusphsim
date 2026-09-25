"""bd-r5-review-n28.175: K_PBF_Finalize loads each neighbour's mass and density once.

The vorticity eta and XSPH blocks share one mass[j] and one density[j] load,
and the vorticity block reads only |omega| (4 B), not the whole float4.
Checked on the source, so no GPU is needed.
"""

import os
import re

KERNEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "physics", "kernels", "pbf_solver.cu"
)


def _finalize_body():
    with open(KERNEL, encoding="utf-8") as f:
        src = f.read()
    start = src.index("void K_PBF_Finalize(")
    nxt = src.find('extern "C" __global__', start)
    return src[start : nxt if nxt != -1 else len(src)]


def test_one_mass_and_density_load_per_neighbour():
    body = _finalize_body()
    assert len(re.findall(r"mass\[j\]", body)) == 1, "expected one mass[j] load"
    assert len(re.findall(r"density\[j\]", body)) == 1, "expected one density[j] load"


def test_vorticity_reads_magnitude_only():
    body = _finalize_body()
    assert "vorticity_in[j]" not in body, "full float4 vorticity_in[j] load is back"
    assert "(const float*)vorticity_in)[4 * j + 3]" in body
