"""Falling Sand 3D - Main entry point with GLFW window, orbit camera, and ImGui UI."""

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
MAX_PARTICLES = 30_000  # budget for initial scene (~20K used)


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


def _spawn_initial_scene(world):
    """Spawn the initial scene: water cube + sand bed."""
    from materials import WATER, SAND

    # 10K water particles in cube at (0, 0.5, 0) size 0.4
    n_water = world.spawn_cube(
        min_corner=(-0.2, 0.3, -0.2),
        max_corner=(0.2, 0.7, 0.2),
        material_id=WATER,
        spacing=0.02,
    )
    print(f"  Water: {n_water:,} particles")

    # 10K sand particles in flat bed y=-0.5 to y=-0.3, x/z spanning -0.8 to 0.8
    n_sand = world.spawn_cube(
        min_corner=(-0.8, -0.5, -0.8),
        max_corner=(0.8, -0.3, 0.8),
        material_id=SAND,
        spacing=0.04,
    )
    print(f"  Sand:  {n_sand:,} particles")

    total = n_water + n_sand
    print(f"  Total: {total:,} particles")
    return total


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

    # Apply PTX workaround before any CuPy compilation
    _apply_ptx_workaround()

    # Create world and spawn initial scene
    from world import World
    from simulation import Simulation
    from ui import UI

    world = World(max_particles=MAX_PARTICLES)
    print("Spawning initial scene...")
    num_active = _spawn_initial_scene(world)

    # Create renderer sized to max_particles
    renderer = Renderer(MAX_PARTICLES, point_scale=20.0)
    renderer.num_active = num_active

    # Create simulation orchestrator (adaptive timestep, accuracy=0.4 CFL)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20)
    print("Simulation initialized -- kernels compiled and constants uploaded")

    # Create ImGui UI
    ui = UI(window)

    # Copy initial state to VBOs
    with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf:
        sim.copy_to_vbos(pos_buf, col_buf)
    import cupy
    cupy.cuda.Device().synchronize()

    # Input state for camera
    right_pressed = False
    middle_pressed = False
    last_mx, last_my = 0.0, 0.0

    def key_callback(_win, key, _scancode, action, _mods):
        nonlocal num_active
        # Let UI handle material shortcuts first
        if ui.handle_key_shortcuts(key, action, sim):
            return
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_win, True)
        elif key == glfw.KEY_SPACE:
            sim.toggle_pause()
        elif key == glfw.KEY_R:
            # Reset to initial configuration
            world.packed_info[:] = 0
            world._high_water = 0
            num_active = _spawn_initial_scene(world)
            renderer.num_active = num_active
            sim.sim_time = 0.0
            sim._last_frame_time = None
        elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
            sim.adjust_speed(0.2)
        elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
            sim.adjust_speed(-0.2)
        elif key == glfw.KEY_LEFT_BRACKET:
            sim.adjust_accuracy(-0.1)
        elif key == glfw.KEY_RIGHT_BRACKET:
            sim.adjust_accuracy(0.1)
        elif key == glfw.KEY_F:
            sim.toggle_fixed_dt()

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

    # Install callbacks through UI (chains imgui + user callbacks)
    ui.set_callbacks(
        key_cb=key_callback,
        mouse_button_cb=mouse_button_callback,
        cursor_pos_cb=cursor_pos_callback,
        scroll_cb=scroll_callback,
        framebuffer_cb=framebuffer_size_callback,
    )

    # FPS tracking
    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # --- ImGui frame start ---
        ui.begin_frame()

        # --- Process brush spawn/kill ---
        brush_delta = ui.process_brush_actions(world, camera)
        if brush_delta != 0:
            renderer.num_active = world._high_water

        # --- Simulation substeps ---
        substeps = sim.step_frame()

        # --- Copy to VBOs (once per frame) ---
        with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf:
            sim.copy_to_vbos(pos_buf, col_buf)

        # Update active count for renderer
        renderer.num_active = world._high_water

        # --- Render 3D scene ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = camera.view_matrix()
        proj = camera.projection_matrix()
        renderer.draw(view, proj)

        # --- Draw ImGui UI ---
        new_max = ui.draw(sim, world, fps)

        # Handle max_particles change
        if new_max is not None:
            world.resize(new_max)
            renderer.close()
            renderer = Renderer(new_max, point_scale=20.0)
            renderer.num_active = 0
            num_active = _spawn_initial_scene(world)
            renderer.num_active = num_active
            sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20)
            with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf:
                sim.copy_to_vbos(pos_buf, col_buf)

        # --- ImGui frame end (render ImGui draw data) ---
        ui.end_frame()

        glfw.swap_buffers(window)

        # FPS counter
        frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = now
            glfw.set_window_title(window, WINDOW_TITLE)

        # Check GL errors
        err = glGetError()
        if err != GL_NO_ERROR:
            print(f"GL error: {err}")

    ui.shutdown()
    renderer.close()
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
