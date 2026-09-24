"""Regression tests for bd-r5-review-n28.115: velocity VBO and motion blur VAO.

- The velocity VBO holds zeros right after the renderer is built, so the
  point shader never reads undefined memory.
- With motion blur off, the point pass uses a VAO where attribute 2
  (velocity) is disabled. With blur on, it uses a VAO that binds the
  velocity VBO at attribute 2.
- Both modes draw with no GL error.
"""

import ctypes

import numpy as np
import pytest


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


def _attrib(vao: int, index: int, pname: int) -> int:
    from OpenGL.GL import glBindVertexArray, glGetVertexAttribiv
    glBindVertexArray(vao)
    v = (ctypes.c_int * 4)()
    glGetVertexAttribiv(index, pname, v)
    glBindVertexArray(0)
    return v[0]


def test_velocity_vbo_zero_and_blur_vao(gl_context):
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_NO_ERROR, GL_VERTEX_ATTRIB_ARRAY_ENABLED,
        GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING,
        glBindBuffer, glGetBufferSubData, glGetError,
    )
    from renderer import Renderer
    from camera import OrbitCamera

    n = 4096
    r = Renderer(n, width=320, height=240)
    try:
        nbytes = n * 16
        out = np.full(nbytes, 0xAB, dtype=np.uint8)
        glBindBuffer(GL_ARRAY_BUFFER, r._vbo_vel)
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, nbytes, out.ctypes.data_as(ctypes.c_void_p))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        assert not out.any(), "velocity VBO must start zeroed"

        # Blur off: the point-pass VAO has no velocity attribute.
        r.motion_blur_enabled = False
        vao_off = r.points_vao()
        assert vao_off == r._vao
        assert _attrib(vao_off, 2, GL_VERTEX_ATTRIB_ARRAY_ENABLED) == 0
        assert _attrib(vao_off, 0, GL_VERTEX_ATTRIB_ARRAY_ENABLED) == 1
        assert _attrib(vao_off, 1, GL_VERTEX_ATTRIB_ARRAY_ENABLED) == 1

        # Blur on: the point-pass VAO reads velocity from the velocity VBO.
        r.motion_blur_enabled = True
        vao_on = r.points_vao()
        assert vao_on == r._vao_blur and vao_on != vao_off
        assert _attrib(vao_on, 2, GL_VERTEX_ATTRIB_ARRAY_ENABLED) == 1
        assert _attrib(vao_on, 2, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING) == r._vbo_vel

        cam = OrbitCamera(distance=2.0, elevation=20.0)
        cam.set_aspect(320, 240)
        view, proj = cam.view_matrix(), cam.projection_matrix()
        r.num_active = n
        r.ssfr_enabled = False
        while glGetError() != GL_NO_ERROR:
            pass
        for blur in (False, True, False):
            r.motion_blur_enabled = blur
            r.draw(view, proj)
        assert glGetError() == GL_NO_ERROR
    finally:
        r.close()
