"""Falling Sand 3D - Main entry point with GLFW window and orbit camera."""

import time
import glfw
import numpy as np
from OpenGL.GL import (
    glClearColor, glClear, glGetString, glGetError, glViewport,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_VERSION, GL_RENDERER, GL_NO_ERROR,
)
from camera import OrbitCamera
from renderer import Renderer

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Falling Sand 3D"
NUM_PARTICLES = 500_000


def _apply_ptx_workaround():
    """Force CuPy to emit PTX instead of cubin for forward-compat with newer GPUs."""
    try:
        import cupy.cuda.compiler as _compiler
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch, '_cache'):
            _compiler._get_arch._cache = {}
        if hasattr(_compiler, '_get_arch_for_options_for_nvrtc'):
            fn = _compiler._get_arch_for_options_for_nvrtc
            if hasattr(fn, '_cache'):
                fn._cache = {}
    except Exception:
        pass


def _fill_particles_dummy(renderer: Renderer):
    """Fill VBOs with 500K particles in a cube with random colors via CuPy."""
    import cupy

    _apply_ptx_workaround()

    n = renderer.num_particles

    # Fill position VBO: random positions in [-0.5, 0.5]^3 cube
    with renderer.cuda_pos as buf:
        pos_arr = buf.device_pointer_as_cupy_array((n, 4), np.float32)
        pos_arr[:, 0] = cupy.random.uniform(-0.5, 0.5, n, dtype=cupy.float32)
        pos_arr[:, 1] = cupy.random.uniform(-0.5, 0.5, n, dtype=cupy.float32)
        pos_arr[:, 2] = cupy.random.uniform(-0.5, 0.5, n, dtype=cupy.float32)
        pos_arr[:, 3] = 1.0
        cupy.cuda.Device().synchronize()

    # Fill color VBO: random bright colors
    with renderer.cuda_col as buf:
        col_arr = buf.device_pointer_as_cupy_array((n, 4), np.float32)
        col_arr[:, 0] = cupy.random.uniform(0.2, 1.0, n, dtype=cupy.float32)
        col_arr[:, 1] = cupy.random.uniform(0.2, 1.0, n, dtype=cupy.float32)
        col_arr[:, 2] = cupy.random.uniform(0.2, 1.0, n, dtype=cupy.float32)
        col_arr[:, 3] = 1.0
        cupy.cuda.Device().synchronize()


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
    glfw.swap_interval(0)  # no vsync for FPS measurement

    # Verify OpenGL version
    gl_version = glGetString(GL_VERSION)
    gl_renderer = glGetString(GL_RENDERER)
    if gl_version:
        print(f"OpenGL: {gl_version.decode()}")
    if gl_renderer:
        print(f"Renderer: {gl_renderer.decode()}")

    # Dark gray background
    glClearColor(0.15, 0.15, 0.15, 1.0)

    camera = OrbitCamera(distance=2.0, elevation=20.0)
    camera.set_aspect(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Create renderer and fill with dummy particles
    renderer = Renderer(NUM_PARTICLES, point_scale=20.0)
    _fill_particles_dummy(renderer)
    print(f"Initialized {NUM_PARTICLES:,} particles")

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
        if width > 0 and height > 0:
            glViewport(0, 0, width, height)
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

        view = camera.view_matrix()
        proj = camera.projection_matrix()
        renderer.draw(view, proj)

        glfw.swap_buffers(window)

        # FPS counter
        frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = now
            glfw.set_window_title(
                window,
                f"{WINDOW_TITLE} | {NUM_PARTICLES // 1000}K particles | FPS: {fps:.0f}",
            )

        # Check GL errors
        err = glGetError()
        if err != GL_NO_ERROR:
            print(f"GL error: {err}")

    renderer.close()
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
