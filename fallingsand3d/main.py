"""Falling Sand 3D - Main entry point with GLFW window and orbit camera."""

import time
import glfw
from OpenGL.GL import (
    glClearColor, glClear, glGetString, glGetError,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_VERSION, GL_RENDERER, GL_NO_ERROR,
)
from camera import OrbitCamera

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Falling Sand 3D"


def main():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # Request OpenGL 4.1+ core profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync

    # Verify OpenGL version
    gl_version = glGetString(GL_VERSION)
    gl_renderer = glGetString(GL_RENDERER)
    if gl_version:
        print(f"OpenGL: {gl_version.decode()}")
    if gl_renderer:
        print(f"Renderer: {gl_renderer.decode()}")

    # Dark gray background
    glClearColor(0.15, 0.15, 0.15, 1.0)

    camera = OrbitCamera()
    camera.set_aspect(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Input state
    right_pressed = False
    middle_pressed = False
    last_mx, last_my = 0.0, 0.0

    def key_callback(_win, key, _scancode, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(_win, True)

    def mouse_button_callback(_win, button, action, _mods):
        nonlocal right_pressed, middle_pressed, last_mx, last_my
        if button == glfw.MOUSE_BUTTON_RIGHT:
            right_pressed = action == glfw.PRESS
            if right_pressed:
                last_mx, last_my = glfw.get_cursor_pos(_win)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            middle_pressed = action == glfw.PRESS
            if middle_pressed:
                last_mx, last_my = glfw.get_cursor_pos(_win)

    def cursor_pos_callback(_win, xpos, ypos):
        nonlocal last_mx, last_my
        dx = xpos - last_mx
        dy = ypos - last_my
        last_mx, last_my = xpos, ypos

        if right_pressed:
            camera.orbit(dx, dy)
        elif middle_pressed:
            camera.pan(dx, dy)

    def scroll_callback(_win, _xoffset, yoffset):
        camera.zoom(yoffset)

    def framebuffer_size_callback(_win, width, height):
        camera.set_aspect(width, height)

    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    # FPS tracking
    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Camera matrices available for rendering
        _view = camera.view_matrix()
        _proj = camera.projection_matrix()

        glfw.swap_buffers(window)

        # FPS counter
        frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = now
            glfw.set_window_title(window, f"{WINDOW_TITLE} | FPS: {fps:.0f}")

        # Check GL errors
        err = glGetError()
        if err != GL_NO_ERROR:
            print(f"GL error: {err}")

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
