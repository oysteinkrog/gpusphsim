"""Measure the host (Python) time spent inside Renderer.draw.

Usage (Windows Python):
    python bench_render_host.py [--particles N] [--frames F] [--ssfr 0|1] [--errcheck 0|1]

--errcheck 0 sets OpenGL.ERROR_CHECKING = False before any OpenGL import,
matching what main.py does by default. The GPU is drained with glFinish
outside the timed region, so the number is the Python + driver submission
cost of one draw call sequence, not GPU time.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=100_000)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--ssfr", type=int, default=1)
    ap.add_argument("--errcheck", type=int, default=0)
    args = ap.parse_args()

    import OpenGL
    OpenGL.ERROR_CHECKING = bool(args.errcheck)
    OpenGL.ERROR_LOGGING = bool(args.errcheck)

    import glfw
    import numpy as np
    from OpenGL.GL import (
        glBindBuffer, glBufferSubData, glFinish, glGetError,
        GL_ARRAY_BUFFER, GL_NO_ERROR,
    )

    if not glfw.init():
        raise SystemExit("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(1280, 720, "bench", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(0)

    from camera import OrbitCamera
    from renderer import Renderer

    n = args.particles
    r = Renderer(n, point_scale=20.0, width=1280, height=720)
    rng = np.random.default_rng(0)
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, :3] = rng.uniform(-0.5, 0.5, size=(n, 3))
    pos[:, 3] = 1000.0
    col = np.ones((n, 4), dtype=np.float32)
    col[:, 3] = 5.0  # WATER material id in col.w (FLUID)
    glBindBuffer(GL_ARRAY_BUFFER, r._vbo_pos)
    glBufferSubData(GL_ARRAY_BUFFER, 0, pos.nbytes, pos)
    glBindBuffer(GL_ARRAY_BUFFER, r._vbo_col)
    glBufferSubData(GL_ARRAY_BUFFER, 0, col.nbytes, col)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    r.num_active = n
    r.ssfr_enabled = bool(args.ssfr)

    cam = OrbitCamera(distance=2.0, elevation=20.0)
    cam.set_aspect(1280, 720)
    view = cam.view_matrix()
    proj = cam.projection_matrix()

    times = []
    for i in range(args.frames + 30):
        glFinish()
        t0 = time.perf_counter()
        r.draw(view, proj)
        t1 = time.perf_counter()
        glfw.swap_buffers(win)
        if i >= 30:
            times.append(t1 - t0)
    glFinish()
    err = glGetError()
    t = np.array(times) * 1e3
    print(f"particles={n} ssfr={args.ssfr} errcheck={args.errcheck} "
          f"frames={len(t)} mean={t.mean():.3f}ms median={np.median(t):.3f}ms "
          f"p90={np.percentile(t, 90):.3f}ms gl_error={'none' if err == GL_NO_ERROR else err}")
    r.close()
    glfw.destroy_window(win)
    glfw.terminate()


if __name__ == "__main__":
    main()
