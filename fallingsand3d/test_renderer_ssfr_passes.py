"""Regression tests for bd-r5-review-n28.99: no redundant SSFR draws.

In SSFR mode every particle goes through the SSFR depth and thickness
passes only. The scene texture holds just the skybox (or the clear color),
the skybox is drawn once per frame, and the default framebuffer gets a
depth-only clear because the composite writes every pixel.
"""

import ctypes

import numpy as np

from test_renderer_gl import gl_context  # noqa: F401  (shared GL fixture)

W, H = 160, 120


def _make_renderer(n, behavior):
    from OpenGL.GL import glBindBuffer, glBufferSubData, GL_ARRAY_BUFFER
    from renderer import Renderer

    r = Renderer(n, point_scale=20.0, width=W, height=H)
    rng = np.random.default_rng(1)
    pos = np.zeros((n, 4), np.float32)
    pos[:, :3] = rng.uniform(-0.2, 0.2, (n, 3))
    col = np.zeros((n, 4), np.float32)
    col[:, 0] = 1.0  # bright red sprites
    col[:, 3] = behavior
    glBindBuffer(GL_ARRAY_BUFFER, r._vbo_pos)
    glBufferSubData(GL_ARRAY_BUFFER, 0, pos.nbytes, pos)
    glBindBuffer(GL_ARRAY_BUFFER, r._vbo_col)
    glBufferSubData(GL_ARRAY_BUFFER, 0, col.nbytes, col)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    r.num_active = n
    r.ssfr_enabled = True
    # draw() builds the SSFR FBOs lazily (bd-r5-review-n28.137); tests that
    # call _draw_ssfr directly need them up front.
    r._create_ssfr_fbos(W, H)
    return r


def _view_proj():
    from camera import OrbitCamera
    cam = OrbitCamera(distance=1.5, elevation=20.0)
    cam.set_aspect(W, H)
    return cam.view_matrix(), cam.projection_matrix()


def test_ssfr_rasterises_each_particle_once_and_skybox_once(gl_context, monkeypatch):  # noqa: F811
    import renderer as rmod
    from OpenGL.GL import GL_POINTS, GL_TRIANGLES, GL_COLOR_BUFFER_BIT

    r = _make_renderer(500, 0.25)  # GRANULAR
    try:
        view, proj = _view_proj()
        draws, clears = [], []
        real_draw, real_clear = rmod.glDrawArrays, rmod.glClear
        real_bind = rmod.glBindFramebuffer
        bound = {"fbo": 0}

        def rec_bind(target, fbo):
            bound["fbo"] = int(fbo)
            real_bind(target, fbo)

        def rec_draw(mode, first, count):
            draws.append((int(mode), int(count)))
            real_draw(mode, first, count)

        def rec_clear(mask):
            clears.append((bound["fbo"], int(mask)))
            real_clear(mask)

        monkeypatch.setattr(rmod, "glDrawArrays", rec_draw)
        monkeypatch.setattr(rmod, "glClear", rec_clear)
        monkeypatch.setattr(rmod, "glBindFramebuffer", rec_bind)
        r._draw_ssfr(view, proj)

        particle_draws = [d for d in draws if d == (int(GL_POINTS), 500)]
        assert len(particle_draws) == 2  # depth pass + thickness pass
        skybox_draws = [d for d in draws if d == (int(GL_TRIANGLES), 36)]
        assert len(skybox_draws) == 1
        default_clears = [m for fbo, m in clears if fbo == 0]
        assert all(not (m & int(GL_COLOR_BUFFER_BIT)) for m in default_clears)
    finally:
        r.close()


def test_ssfr_scene_texture_has_no_particle_sprites(gl_context):  # noqa: F811
    from OpenGL.GL import (glBindTexture, glGetTexImage, GL_TEXTURE_2D, GL_RGBA,
                           GL_FLOAT)

    r = _make_renderer(2000, 0.25)  # GRANULAR, would be drawn by an old pass 0
    try:
        r.skybox_enabled = False  # scene texture should stay the clear color
        view, proj = _view_proj()
        r.draw(view, proj)
        glBindTexture(GL_TEXTURE_2D, r._tex_scene)
        buf = (ctypes.c_float * (W * H * 4))()
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buf)
        glBindTexture(GL_TEXTURE_2D, 0)
        img = np.frombuffer(buf, np.float32).reshape(H, W, 4)
        assert np.allclose(img[..., :3], 0.15, atol=2e-3)
    finally:
        r.close()
