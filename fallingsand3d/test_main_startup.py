"""Startup sizing regression tests for main.py (bd-r5-review-n28.14).

GLFW does not fire the framebuffer size callback at window creation, so
main.py must size the camera, the default viewport and the SSFR FBOs from
glfw.get_framebuffer_size, not from the WINDOW_WIDTH x WINDOW_HEIGHT
constants it asks for.
"""
import ast
import ctypes
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(HERE, "main.py")


def _main_tree():
    with open(MAIN_PY, encoding="utf-8") as f:
        return ast.parse(f.read())


def _call_name(node):
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def test_window_constants_only_used_for_create_window():
    tree = _main_tree()
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "create_window":
            for sub in ast.walk(node):
                allowed.add(id(sub))
    bad = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id in ("WINDOW_WIDTH", "WINDOW_HEIGHT")
                and id(node) not in allowed):
            bad.append(node.lineno)
    assert bad == [], f"WINDOW_WIDTH/HEIGHT used outside create_window at lines {bad}"


def test_renderer_and_camera_sized_from_framebuffer():
    tree = _main_tree()
    renderer_calls = [n for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and _call_name(n) == "Renderer"]
    assert len(renderer_calls) >= 2  # startup and Max particles path
    for call in renderer_calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "width" in kw and ast.unparse(kw["width"]) == "fb_w"
        assert "height" in kw and ast.unparse(kw["height"]) == "fb_h"


def _import_main():
    """Import main.py without changing PyOpenGL error checking for other
    tests: main sets OpenGL.ERROR_CHECKING at import time."""
    OpenGL = pytest.importorskip("OpenGL")
    import OpenGL.GL  # noqa: F401  (import before main so its flag has no effect)
    saved = (OpenGL.ERROR_CHECKING, OpenGL.ERROR_LOGGING)
    try:
        import main
    finally:
        OpenGL.ERROR_CHECKING, OpenGL.ERROR_LOGGING = saved
    return main


@pytest.fixture(scope="module")
def glfw_mod():
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("glfw init failed")
    yield glfw
    glfw.terminate()


def test_first_ssfr_frame_matches_resized_framebuffer(glfw_mod):
    """Create at 1280x720, resize before the first frame, then build the
    renderer the way main.py does: FBOs and viewport must match the real
    framebuffer, not the requested size."""
    glfw = glfw_mod
    from OpenGL.GL import (glGetError, GL_NO_ERROR, glGetIntegerv, GL_VIEWPORT,
                           glBindTexture, GL_TEXTURE_2D, glGetTexLevelParameteriv,
                           GL_TEXTURE_WIDTH, GL_TEXTURE_HEIGHT, glViewport)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(1280, 720, "test", None, None)
    if not win:
        pytest.skip("no GL 4.1 context")
    try:
        glfw.make_context_current(win)
        glfw.set_window_size(win, 1000, 600)
        glfw.poll_events()

        main = _import_main()
        from renderer import Renderer
        from camera import OrbitCamera

        fb_w, fb_h = main._framebuffer_size(win)
        assert (fb_w, fb_h) == tuple(glfw.get_framebuffer_size(win))
        assert (fb_w, fb_h) != (1280, 720)

        glViewport(0, 0, fb_w, fb_h)
        cam = OrbitCamera(distance=2.0, elevation=20.0)
        cam.set_aspect(fb_w, fb_h)
        r = Renderer(1000, width=fb_w, height=fb_h)
        try:
            while glGetError() != GL_NO_ERROR:
                pass
            r.ssfr_enabled = True
            r.draw(cam.view_matrix(), cam.projection_matrix())
            assert glGetError() == GL_NO_ERROR

            vp = (ctypes.c_int * 4)()
            glGetIntegerv(GL_VIEWPORT, vp)
            assert tuple(vp) == (0, 0, fb_w, fb_h)

            glBindTexture(GL_TEXTURE_2D, r._tex_blur1)
            tw = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
            th = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
            glBindTexture(GL_TEXTURE_2D, 0)
            assert (int(tw), int(th)) == (fb_w, fb_h)
        finally:
            r.close()
    finally:
        glfw.destroy_window(win)


def test_framebuffer_size_clamps_minimized(monkeypatch):
    main = _import_main()
    monkeypatch.setattr(main.glfw, "get_framebuffer_size", lambda _w: (0, 0))
    assert main._framebuffer_size(None) == (1, 1)
