"""Regression tests for bd-r5-review-n28.5: GL error checking and static SSFR uniforms.

- main.py turns PyOpenGL per-call error checking off before the first
  OpenGL.GL import, and FS3D_GL_DEBUG=1 keeps it on.
- Renderer sets sampler unit indices and texel size once (init and resize),
  and an SSFR frame still renders with no GL error.
"""

import ctypes
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

_PROBE = (
    "import sys; sys.path.insert(0, r'{here}'); import main, OpenGL; "
    "print('EC', OpenGL.ERROR_CHECKING, OpenGL.ERROR_LOGGING)"
)


def _probe_main(env_extra: dict) -> str:
    env = dict(os.environ)
    env.pop("FS3D_GL_DEBUG", None)
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(here=HERE)],
        capture_output=True, text=True, env=env, cwd=HERE, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("EC ")]
    assert line, out.stdout
    return line[-1]


def test_main_disables_pyopengl_error_checking():
    assert _probe_main({}) == "EC False False"


def test_main_gl_debug_env_keeps_error_checking():
    assert _probe_main({"FS3D_GL_DEBUG": "1"}) == "EC True True"


@pytest.fixture(scope="module")
def gl_context():
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(320, 240, "test", None, None)
    if not win:
        glfw.terminate()
        pytest.skip("no GL 4.1 context")
    glfw.make_context_current(win)
    yield win
    glfw.destroy_window(win)
    glfw.terminate()


def _get_int(prog, loc):
    from OpenGL.GL import glGetUniformiv
    v = (ctypes.c_int * 1)()
    glGetUniformiv(prog, loc, v)
    return v[0]


def _get_vec2(prog, loc):
    from OpenGL.GL import glGetUniformfv
    v = (ctypes.c_float * 2)()
    glGetUniformfv(prog, loc, v)
    return v[0], v[1]


def test_static_uniforms_set_on_init_and_resize(gl_context):
    import numpy as np
    from OpenGL.GL import glGetError, GL_NO_ERROR
    from renderer import Renderer
    from camera import OrbitCamera

    r = Renderer(1000, width=320, height=240)
    try:
        assert _get_int(r._prog_skybox, r._u_sky_tex) == 0
        assert _get_int(r._prog_blur, r._u_blur_depth) == 0
        assert _get_int(r._prog_normal, r._u_norm_depth) == 0
        assert _get_int(r._prog_depth_write, r._u_dw_depth_tex) == 0
        units = [
            _get_int(r._prog_composite, loc)
            for loc in (r._u_comp_depth, r._u_comp_normal, r._u_comp_thickness,
                        r._u_comp_scene, r._u_comp_skybox)
        ]
        assert units == [0, 1, 2, 3, 4]
        for prog, loc in ((r._prog_blur, r._u_blur_texel),
                          (r._prog_normal, r._u_norm_texel),
                          (r._prog_composite, r._u_comp_texel)):
            assert _get_vec2(prog, loc) == pytest.approx((1 / 320, 1 / 240))

        r.resize(640, 480)
        for prog, loc in ((r._prog_blur, r._u_blur_texel),
                          (r._prog_normal, r._u_norm_texel),
                          (r._prog_composite, r._u_comp_texel)):
            assert _get_vec2(prog, loc) == pytest.approx((1 / 640, 1 / 480))

        # One SSFR frame and one point-sprite frame render without GL errors.
        cam = OrbitCamera(distance=2.0, elevation=20.0)
        cam.set_aspect(640, 480)
        view, proj = cam.view_matrix(), cam.projection_matrix()
        while glGetError() != GL_NO_ERROR:
            pass
        r.ssfr_enabled = True
        r.draw(view, proj)
        r.ssfr_enabled = False
        r.draw(view, proj)
        assert glGetError() == GL_NO_ERROR
        assert np.isfinite(view).all()
    finally:
        r.close()
