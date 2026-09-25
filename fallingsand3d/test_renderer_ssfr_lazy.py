"""Regression tests for bd-r5-review-n28.137: lazy SSFR framebuffers.

With SSFR off the renderer holds no SSFR FBOs and resize() never builds
them. The first SSFR draw builds them once at the current window size,
and turning SSFR off frees them again.
"""

import numpy as np

from test_renderer_gl import gl_context  # noqa: F401  (shared GL fixture)


def _view_proj(w, h):
    from camera import OrbitCamera
    cam = OrbitCamera(distance=2.0, elevation=20.0)
    cam.set_aspect(w, h)
    return cam.view_matrix(), cam.projection_matrix()


def _texture_size(tex):
    from OpenGL.GL import (glBindTexture, glGetTexLevelParameteriv,
                           GL_TEXTURE_2D, GL_TEXTURE_WIDTH, GL_TEXTURE_HEIGHT)
    glBindTexture(GL_TEXTURE_2D, tex)
    w = int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH))
    h = int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT))
    glBindTexture(GL_TEXTURE_2D, 0)
    return w, h


def test_ssfr_fbos_built_only_when_ssfr_draws(gl_context, monkeypatch):  # noqa: F811
    from OpenGL.GL import glGetError, glIsTexture, GL_NO_ERROR
    import renderer as rmod

    calls = []
    real_create = rmod.Renderer._create_ssfr_fbos

    def counting_create(self, w, h):
        calls.append((w, h))
        real_create(self, w, h)

    monkeypatch.setattr(rmod.Renderer, "_create_ssfr_fbos", counting_create)

    r = rmod.Renderer(1000, width=320, height=240)
    try:
        # Startup with SSFR off allocates nothing.
        assert calls == []
        assert r._tex_depth == 0 and r._fbo_scene == 0

        # A window drag with SSFR off builds nothing.
        for i in range(20):
            r.resize(320 + 8 * i, 240 + 6 * i)
        assert calls == []
        w, h = 320 + 8 * 19, 240 + 6 * 19
        view, proj = _view_proj(w, h)
        r.draw(view, proj)
        assert calls == []

        # First SSFR frame builds the FBOs once, at the current size.
        while glGetError() != GL_NO_ERROR:
            pass
        r.ssfr_enabled = True
        r.draw(view, proj)
        assert calls == [(w, h)]
        assert _texture_size(r._tex_scene) == (w, h)
        assert _texture_size(r._tex_depth) == (w, h)
        assert glGetError() == GL_NO_ERROR

        # Later SSFR frames at the same size do not rebuild.
        r.draw(view, proj)
        assert len(calls) == 1

        # Resizes with SSFR on: at most one rebuild per rendered frame,
        # and the textures follow the window size.
        for i in range(5):
            r.resize(400 + i, 300 + i)
        assert len(calls) == 1
        r.draw(view, proj)
        assert calls[-1] == (404, 304) and len(calls) == 2
        assert _texture_size(r._tex_scene) == (404, 304)

        # Turning SSFR off frees the FBOs on the next frame.
        old_tex = r._tex_scene
        r.ssfr_enabled = False
        r.draw(view, proj)
        assert r._tex_scene == 0 and r._fbo_depth == 0
        assert not r._ssfr_fbos_valid
        assert not glIsTexture(old_tex)
        assert glGetError() == GL_NO_ERROR

        # Turning it back on rebuilds once.
        r.ssfr_enabled = True
        r.draw(view, proj)
        assert len(calls) == 3
        assert np.isfinite(view).all()
    finally:
        r.close()
