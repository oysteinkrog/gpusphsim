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
MAX_PARTICLES = 500_000  # budget for preset scenes (up to ~200K used)


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
    glfw.swap_interval(1)  # vsync ON by default (stable substep counts)

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

    # Create renderer sized to max_particles (pass initial window dimensions for FBO allocation)
    renderer = Renderer(MAX_PARTICLES, point_scale=20.0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    renderer.num_active = num_active

    # Create simulation orchestrator (adaptive timestep, accuracy=0.4 CFL)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20)
    renderer.sdf_manager = sim.sdf_manager
    print("Simulation initialized -- kernels compiled and constants uploaded")

    # Create ImGui UI
    ui = UI(window)
    ui.set_sdf_drag_refs(camera, sim.sdf_manager)

    # Snapshot ring buffer for undo
    from snapshot import SnapshotRing
    snapshots = SnapshotRing(max_snapshots=12, max_particles=MAX_PARTICLES)
    ui.snapshots = snapshots  # expose to UI for timeline display

    # Copy initial state to VBOs
    with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf, renderer.cuda_vel as vel_buf:
        sim.copy_to_vbos(pos_buf, col_buf, vel_buf)
    import cupy
    cupy.cuda.Device().synchronize()

    # Input state for camera
    left_pressed = False
    right_pressed = False
    middle_pressed = False
    last_mx, last_my = 0.0, 0.0

    def key_callback(_win, key, _scancode, action, _mods):
        nonlocal num_active, active_spawner, spawner_frame_counter
        # Let UI handle material shortcuts first
        if ui.handle_key_shortcuts(key, action, sim):
            return
        if action != glfw.PRESS:
            return
        # Ctrl+Z = undo (restore previous snapshot)
        if key == glfw.KEY_Z and (_mods & glfw.MOD_CONTROL):
            if snapshots.restore(world):
                renderer.num_active = world._high_water
                sim._invalidate_graphs()
                sim.reset_spawn_damping()
                active_spawner = None
                spawner_frame_counter = 0
                with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf, renderer.cuda_vel as vel_buf:
                    sim.copy_to_vbos(pos_buf, col_buf, vel_buf)
                print(f"Undo: restored to {world._high_water:,} particles")
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_win, True)
        elif key == glfw.KEY_SPACE:
            sim.toggle_pause()
        elif key == glfw.KEY_R:
            # Reset to initial configuration
            world.packed_info[:] = 0
            world._high_water = 0
            world.foam_count.fill(0)  # reset foam pool
            num_active = _spawn_initial_scene(world)
            renderer.num_active = num_active
            sim.sim_time = 0.0
            sim._last_frame_time = None
            sim.reset_spawn_damping()  # clear damping ramp so restored particles aren't drag-glitched
            active_spawner = None
            spawner_frame_counter = 0
            snapshots.clear()
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
        nonlocal left_pressed, right_pressed, middle_pressed, last_mx, last_my
        if button == glfw.MOUSE_BUTTON_LEFT:
            left_pressed = action == glfw.PRESS
            if left_pressed:
                last_mx, last_my = glfw.get_cursor_pos(_win)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
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

        if left_pressed or right_pressed:
            camera.orbit(dx, dy)
        elif middle_pressed:
            camera.pan(dx, dy)

    def scroll_callback(_win, _xoffset, yoffset):
        camera.zoom(yoffset)

    def framebuffer_size_callback(_win, width, height):
        if width > 0 and height > 0:
            glViewport(0, 0, width, height)
            camera.set_aspect(width, height)
            renderer.resize(width, height)

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

    # Periodic spawner state (used by Acid Rain preset)
    active_spawner = None
    spawner_frame_counter = 0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # --- ImGui frame start ---
        ui.begin_frame()

        # --- Process brush spawn/kill ---
        brush_delta = ui.process_brush_actions(world, camera)
        if brush_delta != 0:
            renderer.num_active = world._high_water

        # --- Periodic spawner (Acid Rain) ---
        if active_spawner is not None and not sim.paused:
            spawner_frame_counter += 1
            if spawner_frame_counter >= active_spawner["interval_frames"]:
                spawner_frame_counter = 0
                world.spawn_cube(
                    min_corner=tuple(active_spawner["min_corner"]),
                    max_corner=tuple(active_spawner["max_corner"]),
                    material_id=active_spawner["material_id"],
                    spacing=active_spawner["spacing"],
                )
                renderer.num_active = world._high_water

        # --- Simulation substeps ---
        substeps = sim.step_frame()

        # --- Snapshot tick (auto-capture every N frames) ---
        snapshots.tick(world)

        # --- Copy to VBOs (once per frame) ---
        vel_ctx = renderer.cuda_vel if renderer.motion_blur_enabled else None
        with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf:
            if vel_ctx is not None:
                with vel_ctx as vel_buf:
                    sim.copy_to_vbos(pos_buf, col_buf, vel_buf)
            else:
                sim.copy_to_vbos(pos_buf, col_buf)

        # Update active count for renderer
        renderer.num_active = world._high_water

        # Sync foam enable state from renderer (UI) to world (sim)
        world.foam_enabled = renderer.foam_enabled

        # Copy foam particles to VBO (if enabled)
        if renderer.foam_enabled:
            with renderer.cuda_foam as foam_buf:
                renderer.num_foam = sim.copy_foam_to_vbo(foam_buf)
        else:
            renderer.num_foam = 0

        # --- Update 3D cursor ---
        cursor_pos = ui.get_cursor_world_pos(camera)
        if cursor_pos is not None:
            renderer.cursor_visible = True
            renderer.cursor_center = cursor_pos
            renderer.cursor_radius = ui.brush_radius
        else:
            renderer.cursor_visible = False

        # --- Update SDF selection highlight ---
        renderer.selected_sdf_id = ui.selected_sdf_id

        # --- Render 3D scene ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = camera.view_matrix()
        proj = camera.projection_matrix()
        renderer.draw(view, proj)

        # --- Draw ImGui UI ---
        ui_changes = ui.draw(sim, world, fps, renderer)

        # --- Handle solver switching ---
        solver_name = ui.pending_solver
        if solver_name is not None:
            from solver_profiles import PROFILES as SOLVER_PROFILES
            profile = SOLVER_PROFILES[solver_name]
            print(f"Switching solver: {solver_name}")
            # Clear world and reload current preset (avoids physics shock)
            world.packed_info[:] = 0
            world._high_water = 0
            world.foam_count.fill(0)  # reset foam pool
            sim.rigid_body_manager.reset()
            if hasattr(sim, '_rigid_boundary_initialized'):
                del sim._rigid_boundary_initialized
            num_active = _spawn_initial_scene(world)
            renderer.num_active = num_active
            sim.set_solver_profile(profile)
            sim.sim_time = 0.0
            sim._last_frame_time = None
            active_spawner = None
            spawner_frame_counter = 0
            with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf, renderer.cuda_vel as vel_buf:
                sim.copy_to_vbos(pos_buf, col_buf, vel_buf)

        # --- Handle preset loading ---
        preset_name = ui.pending_preset
        if preset_name is not None:
            from presets import PRESETS
            load_fn = PRESETS[preset_name]
            print(f"Loading preset: {preset_name}")
            sim.sdf_manager.clear()  # reset SDF objects before preset
            sim.rigid_body_manager.reset()  # reset rigid bodies before preset
            if hasattr(sim, '_rigid_boundary_initialized'):
                del sim._rigid_boundary_initialized
            n_spawned, spawner_cfg = load_fn(world)
            sim.rigid_body_manager.finalize_boundary_data()  # wire up rigid body pipeline
            sim.rigid_body_manager.upload(sim.get_all_modules())  # upload body data + count to GPU constant memory
            sim.sdf_manager.upload_if_dirty()  # upload any SDF objects preset added
            world.foam_count.fill(0)  # reset foam pool on preset load
            active_spawner = spawner_cfg
            spawner_frame_counter = 0
            renderer.num_active = world._high_water
            sim.sim_time = 0.0
            sim._last_frame_time = None
            sim.reset_spawn_damping()
            snapshots.clear()
            with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf, renderer.cuda_vel as vel_buf:
                sim.copy_to_vbos(pos_buf, col_buf, vel_buf)

        # Handle max_particles change
        if 'new_max' in ui_changes:
            new_max = ui_changes['new_max']
            world.resize(new_max)
            renderer.close()
            sim.close()  # drain readback stream + free pinned buffers/graphs before reassign
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            renderer = Renderer(new_max, point_scale=20.0, width=fb_w, height=fb_h)
            renderer.num_active = 0
            num_active = _spawn_initial_scene(world)
            renderer.num_active = num_active
            sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4, fixed_dt=False, max_substeps=20)
            renderer.sdf_manager = sim.sdf_manager
            ui.set_sdf_drag_refs(camera, sim.sdf_manager)
            active_spawner = None
            spawner_frame_counter = 0
            snapshots = SnapshotRing(max_snapshots=12, max_particles=new_max)
            ui.snapshots = snapshots
            with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf, renderer.cuda_vel as vel_buf:
                sim.copy_to_vbos(pos_buf, col_buf, vel_buf)

        # Handle scene loaded from save file or undo restore
        if '_scene_loaded' in ui_changes:
            renderer.num_active = world._high_water
            sim._invalidate_graphs()
            sim.reset_spawn_damping()  # clear damping ramp to avoid velocity glitch on first post-load frames
            with renderer.cuda_pos as pos_buf, renderer.cuda_col as col_buf, renderer.cuda_vel as vel_buf:
                sim.copy_to_vbos(pos_buf, col_buf, vel_buf)

        # Handle world size change
        if 'new_world_size' in ui_changes:
            sim.set_world_size(ui_changes['new_world_size'])

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
