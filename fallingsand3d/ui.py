"""ImGui-based user interface for Falling Sand 3D.

Provides:
- Material picker panel (15 materials as colored buttons in 3x5 grid)
- Brush size slider
- Simulation controls (speed, accuracy, fixed_dt, max_particles)
- Status bar (particle count, FPS, dt, substeps, speed, paused)
- Left-click spawns selected material; Shift+click kills particles
- Keyboard shortcuts: 1-9 for material select, Space=pause, R=reset, +/- speed

Uses imgui-bundle (imgui.backends.glfw_opengl3) integrated into the GLFW event loop.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import glfw

from materials import (
    MATERIALS, DEAD, STONE, SAND, DIRT, GRAVEL, WATER, OIL, LAVA, ACID,
    WOOD, METAL, ICE, STEAM, SMOKE, FIRE, GUNPOWDER, WET_SAND, MUD,
)
from presets import PRESETS
from solver_profiles import PROFILES, PROFILE_NAMES

# Material IDs for the picker (exclude DEAD=0)
_PICKER_MATERIALS = [
    STONE, SAND, DIRT, GRAVEL, WATER,
    OIL, LAVA, ACID, WOOD, METAL,
    ICE, STEAM, SMOKE, FIRE, GUNPOWDER,
    WET_SAND, MUD,
]

# Quick-select mapping: keys 1-9 map to first 9 materials
_QUICK_SELECT = {
    glfw.KEY_1: STONE,
    glfw.KEY_2: SAND,
    glfw.KEY_3: DIRT,
    glfw.KEY_4: GRAVEL,
    glfw.KEY_5: WATER,
    glfw.KEY_6: OIL,
    glfw.KEY_7: LAVA,
    glfw.KEY_8: ACID,
    glfw.KEY_9: WOOD,
}

# Max particles options
_MAX_PARTICLES_OPTIONS = [
    (100_000, "100K"),
    (250_000, "250K"),
    (500_000, "500K"),
    (1_000_000, "1M"),
]


def _unproject_mouse_ray(
    mx: float, my: float,
    viewport_w: int, viewport_h: int,
    view: np.ndarray, proj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject screen coordinates to a world-space ray.

    Returns (ray_origin, ray_direction) both as float32[3].
    """
    # Normalized device coords [-1, 1]
    ndc_x = (2.0 * mx / viewport_w) - 1.0
    ndc_y = 1.0 - (2.0 * my / viewport_h)  # flip Y

    # Inverse view-projection
    vp = proj @ view
    try:
        inv_vp = np.linalg.inv(vp)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float32), np.array([0, 0, -1], dtype=np.float32)

    # Near and far points in clip space
    near_clip = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
    far_clip = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float32)

    near_world = inv_vp @ near_clip
    far_world = inv_vp @ far_clip

    # Perspective divide
    near_world = near_world[:3] / near_world[3]
    far_world = far_world[:3] / far_world[3]

    direction = far_world - near_world
    length = np.linalg.norm(direction)
    if length > 1e-8:
        direction /= length

    return near_world.astype(np.float32), direction.astype(np.float32)


def _ray_plane_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    plane_y: float = 0.0,
) -> Optional[np.ndarray]:
    """Intersect ray with horizontal plane y=plane_y. Returns hit point or None."""
    if abs(ray_dir[1]) < 1e-8:
        return None
    t = (plane_y - ray_origin[1]) / ray_dir[1]
    if t < 0:
        return None
    return (ray_origin + t * ray_dir).astype(np.float32)


class UI:
    """ImGui user interface manager.

    Parameters
    ----------
    window : GLFW window handle
        The GLFW window to attach ImGui to.
    """

    def __init__(self, window) -> None:
        self._window = window
        self._selected_material: Optional[int] = None  # None = camera mode (no brush)
        self._brush_radius: float = 0.1
        self._speed_log: float = 0.0  # log10(speed), maps to 0.1-10.0
        self._max_particles_idx: int = 0  # index into _MAX_PARTICLES_OPTIONS
        self._world_half_size: float = 1.0  # current world half-extent

        # ImGui context + renderer
        imgui.create_context()
        self._imgui_renderer = GlfwRenderer(window, attach_callbacks=False)

        # We manually forward events to imgui renderer
        # Store references to user callbacks so we can chain
        self._user_key_cb = None
        self._user_mouse_button_cb = None
        self._user_cursor_pos_cb = None
        self._user_scroll_cb = None

        # Brush action state
        self._pending_spawn: Optional[Tuple[float, float]] = None
        self._pending_kill: Optional[Tuple[float, float]] = None

        # Preset loading state (set by Scenes panel, consumed by main loop)
        self._pending_preset: Optional[str] = None

        # Solver switching state (consumed by main loop)
        self._pending_solver: Optional[str] = None
        self._solver_idx: int = 0  # current index into PROFILE_NAMES

        # Cached viewport size
        w, h = glfw.get_framebuffer_size(window)
        self._viewport_w = w
        self._viewport_h = h

        # Track whether resize callback needs chaining
        self._user_framebuffer_cb = None

    def set_callbacks(
        self,
        key_cb=None,
        mouse_button_cb=None,
        cursor_pos_cb=None,
        scroll_cb=None,
        framebuffer_cb=None,
    ) -> None:
        """Store user callbacks and install combined GLFW callbacks."""
        self._user_key_cb = key_cb
        self._user_mouse_button_cb = mouse_button_cb
        self._user_cursor_pos_cb = cursor_pos_cb
        self._user_scroll_cb = scroll_cb
        self._user_framebuffer_cb = framebuffer_cb

        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor_pos)
        glfw.set_scroll_callback(self._window, self._on_scroll)
        glfw.set_char_callback(self._window, self._on_char)
        glfw.set_framebuffer_size_callback(self._window, self._on_framebuffer_size)

    def _on_key(self, window, key, scancode, action, mods):
        self._imgui_renderer.keyboard_callback(window, key, scancode, action, mods)
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return
        if self._user_key_cb:
            self._user_key_cb(window, key, scancode, action, mods)

    def _on_mouse_button(self, window, button, action, mods):
        self._imgui_renderer.mouse_button_callback(window, button, action, mods)
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        # Left click: spawn/kill only when a brush is selected
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if self._selected_material is not None:
                mx, my = glfw.get_cursor_pos(window)
                if mods & glfw.MOD_SHIFT:
                    self._pending_kill = (mx, my)
                else:
                    self._pending_spawn = (mx, my)
                return

        # Pass through to user callbacks (camera controls)
        if self._user_mouse_button_cb:
            self._user_mouse_button_cb(window, button, action, mods)

    def _on_cursor_pos(self, window, xpos, ypos):
        self._imgui_renderer.mouse_callback(window, xpos, ypos)
        io = imgui.get_io()
        if io.want_capture_mouse:
            return
        if self._user_cursor_pos_cb:
            self._user_cursor_pos_cb(window, xpos, ypos)

    def _on_scroll(self, window, xoffset, yoffset):
        self._imgui_renderer.scroll_callback(window, xoffset, yoffset)
        io = imgui.get_io()
        if io.want_capture_mouse:
            return
        if self._user_scroll_cb:
            self._user_scroll_cb(window, xoffset, yoffset)

    def _on_char(self, window, char):
        self._imgui_renderer.char_callback(window, char)

    def _on_framebuffer_size(self, window, width, height):
        self._viewport_w = width
        self._viewport_h = height
        if self._user_framebuffer_cb:
            self._user_framebuffer_cb(window, width, height)

    @property
    def selected_material(self) -> int:
        return self._selected_material

    @property
    def brush_radius(self) -> float:
        return self._brush_radius

    @property
    def pending_preset(self) -> Optional[str]:
        """Return and clear the pending preset name, if any."""
        name = self._pending_preset
        self._pending_preset = None
        return name

    @property
    def pending_solver(self) -> Optional[str]:
        """Return and clear the pending solver profile name, if any."""
        name = self._pending_solver
        self._pending_solver = None
        return name

    @property
    def want_capture_mouse(self) -> bool:
        return imgui.get_io().want_capture_mouse

    def process_brush_actions(self, world, camera) -> int:
        """Process pending spawn/kill brush actions.

        Returns number of particles spawned (negative = killed).
        """
        delta = 0
        view = camera.view_matrix()
        proj = camera.projection_matrix()

        if self._pending_spawn is not None:
            mx, my = self._pending_spawn
            self._pending_spawn = None
            origin, direction = _unproject_mouse_ray(
                mx, my, self._viewport_w, self._viewport_h, view, proj,
            )
            hit = _ray_plane_intersect(origin, direction, plane_y=0.0)
            if hit is not None:
                n = world.spawn_sphere(
                    center=(float(hit[0]), float(hit[1]), float(hit[2])),
                    radius=self._brush_radius,
                    material_id=self._selected_material,
                    count=int(500 * (self._brush_radius / 0.1) ** 3),
                )
                delta += n

        if self._pending_kill is not None:
            mx, my = self._pending_kill
            self._pending_kill = None
            origin, direction = _unproject_mouse_ray(
                mx, my, self._viewport_w, self._viewport_h, view, proj,
            )
            hit = _ray_plane_intersect(origin, direction, plane_y=0.0)
            if hit is not None:
                killed = world.kill_in_sphere(
                    center=(float(hit[0]), float(hit[1]), float(hit[2])),
                    radius=self._brush_radius,
                )
                delta -= killed

        return delta

    def begin_frame(self) -> None:
        """Start a new ImGui frame. Call after glfw.poll_events()."""
        self._imgui_renderer.process_inputs()
        imgui.new_frame()

    def end_frame(self) -> None:
        """Finalize and render ImGui. Call before glfw.swap_buffers()."""
        imgui.render()
        self._imgui_renderer.render(imgui.get_draw_data())

    def draw(self, sim, world, fps: float, renderer=None) -> dict:
        """Draw all UI panels. Returns dict of changes (empty if nothing changed).

        Possible keys: 'new_max', 'new_world_size'.

        Parameters
        ----------
        sim : Simulation
            The simulation orchestrator (for speed, accuracy, dt, etc.)
        world : World
            The particle world (for particle count, max_particles)
        fps : float
            Current frames per second
        renderer : Renderer, optional
            The renderer (for SSFR controls)
        """
        changes = {}

        # --- Material Picker Panel ---
        imgui.set_next_window_pos(imgui.ImVec2(10, 10), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(220, 0), imgui.Cond_.first_use_ever)

        if imgui.begin("Materials", None, imgui.WindowFlags_.always_auto_resize)[0]:
            # 3x5 grid of material buttons
            btn_size = imgui.ImVec2(60, 30)
            for i, mat_id in enumerate(_PICKER_MATERIALS):
                mat = MATERIALS[mat_id]
                col = imgui.ImVec4(mat.color_r, mat.color_g, mat.color_b, 1.0)

                # Highlight selected material
                is_selected = (mat_id == self._selected_material)
                if is_selected:
                    imgui.push_style_color(imgui.Col_.button, col)
                    imgui.push_style_color(
                        imgui.Col_.button_hovered,
                        imgui.ImVec4(
                            min(1.0, col.x + 0.2),
                            min(1.0, col.y + 0.2),
                            min(1.0, col.z + 0.2),
                            1.0,
                        ),
                    )
                    imgui.push_style_color(imgui.Col_.button_active, col)
                else:
                    # Dim version of color
                    dim = imgui.ImVec4(col.x * 0.6, col.y * 0.6, col.z * 0.6, 1.0)
                    imgui.push_style_color(imgui.Col_.button, dim)
                    imgui.push_style_color(
                        imgui.Col_.button_hovered, col,
                    )
                    imgui.push_style_color(
                        imgui.Col_.button_active, col,
                    )

                # Short name for button label
                label = mat.name[:6]
                if imgui.button(f"{label}##{mat_id}", btn_size):
                    # Toggle: click again to deselect (return to camera mode)
                    if self._selected_material == mat_id:
                        self._selected_material = None
                    else:
                        self._selected_material = mat_id

                imgui.pop_style_color(3)

                # 3 columns: same_line after 1st and 2nd in each row
                col_idx = i % 3
                if col_idx < 2 and i < len(_PICKER_MATERIALS) - 1:
                    imgui.same_line()

            imgui.separator()

            # Brush size slider
            changed, new_val = imgui.slider_float(
                "Brush", self._brush_radius, 0.02, 0.3, "%.2f",
            )
            if changed:
                self._brush_radius = new_val

            if self._selected_material is not None:
                imgui.text(f"Brush: {MATERIALS[self._selected_material].name}")
                imgui.text("LClick=Spawn  Shift+LClick=Kill")
            else:
                imgui.text("Brush: None (camera mode)")
                imgui.text("LDrag=Orbit  MDrag=Pan  Scroll=Zoom")

        imgui.end()

        # --- Scenes Panel ---
        imgui.set_next_window_pos(imgui.ImVec2(10, 340), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(220, 0), imgui.Cond_.first_use_ever)

        if imgui.begin("Scenes", None, imgui.WindowFlags_.always_auto_resize)[0]:
            btn_size = imgui.ImVec2(100, 28)
            for i, name in enumerate(PRESETS):
                if imgui.button(f"{name}##{i}", btn_size):
                    self._pending_preset = name
                if i % 2 == 0 and i < len(PRESETS) - 1:
                    imgui.same_line()
        imgui.end()

        # --- Simulation Controls Panel ---
        imgui.set_next_window_pos(imgui.ImVec2(10, 440), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(220, 0), imgui.Cond_.first_use_ever)

        if imgui.begin("Simulation", None, imgui.WindowFlags_.always_auto_resize)[0]:
            # Speed slider (log scale 0.1 to 10.0)
            # Map log10(speed) from log10(0.1)=-1 to log10(10)=1
            log_speed = math.log10(max(0.1, sim.speed))
            changed, new_log = imgui.slider_float(
                "Speed", log_speed, -1.0, 1.0, f"{sim.speed:.2f}x",
            )
            if changed:
                sim.speed = max(0.1, min(10.0, 10.0 ** new_log))

            # Accuracy slider
            changed, new_acc = imgui.slider_float(
                "Accuracy", sim.accuracy, 0.1, 1.0, "%.1f",
            )
            if changed:
                sim.accuracy = max(0.1, min(1.0, new_acc))

            # Fixed dt checkbox + value
            changed, new_fixed = imgui.checkbox("Fixed dt", sim.fixed_dt)
            if changed:
                sim.toggle_fixed_dt()

            if sim.fixed_dt:
                imgui.same_line()
                changed, new_fdt = imgui.input_float(
                    "##fdt", sim.fixed_dt_value * 1000.0, 0.1, 1.0, "%.2f ms",
                )
                if changed:
                    sim.fixed_dt_value = max(0.01, new_fdt) / 1000.0
                    sim._upload_dt(sim.fixed_dt_value)

            # Max particles dropdown
            imgui.separator()
            # Find current index
            cur_max = world.max_particles
            cur_idx = 0
            for idx, (val, _) in enumerate(_MAX_PARTICLES_OPTIONS):
                if val == cur_max:
                    cur_idx = idx
                    break

            labels = [label for _, label in _MAX_PARTICLES_OPTIONS]
            preview = labels[cur_idx]
            if imgui.begin_combo("Max particles", preview):
                for idx, (val, label) in enumerate(_MAX_PARTICLES_OPTIONS):
                    is_sel = (idx == cur_idx)
                    clicked, _ = imgui.selectable(label, is_sel)
                    if clicked and val != cur_max:
                        changes['new_max'] = val
                    if is_sel:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            # World size slider
            changed, new_hs = imgui.slider_float(
                "World size", self._world_half_size, 0.5, 3.0, "%.1f",
            )
            if changed:
                self._world_half_size = new_hs
            # Only apply on release (avoid re-upload every drag frame)
            if imgui.is_item_deactivated_after_edit():
                changes['new_world_size'] = self._world_half_size

            # Pause button
            if imgui.button("Pause [Space]" if not sim.paused else "Resume [Space]"):
                sim.toggle_pause()

            imgui.same_line()

            # Timing toggle
            changed, new_timing = imgui.checkbox("Timing", sim.timing_enabled)
            if changed:
                sim.timing_enabled = new_timing

            # Solver dropdown
            imgui.separator()
            preview = PROFILE_NAMES[self._solver_idx]
            if imgui.begin_combo("Solver", preview):
                for idx, name in enumerate(PROFILE_NAMES):
                    is_sel = (idx == self._solver_idx)
                    clicked, _ = imgui.selectable(name, is_sel)
                    if clicked and idx != self._solver_idx:
                        self._solver_idx = idx
                        self._pending_solver = name
                    if is_sel:
                        imgui.set_item_default_focus()
                imgui.end_combo()

        imgui.end()

        # --- Rendering Panel (SSFR controls) ---
        if renderer is not None:
            imgui.set_next_window_pos(imgui.ImVec2(10, 560), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(imgui.ImVec2(220, 0), imgui.Cond_.first_use_ever)

            if imgui.begin("Rendering", None, imgui.WindowFlags_.always_auto_resize)[0]:
                changed, new_ssfr = imgui.checkbox("SSFR (fluid surface)", renderer.ssfr_enabled)
                if changed:
                    renderer.ssfr_enabled = new_ssfr

                if renderer.ssfr_enabled:
                    changed, val = imgui.slider_float(
                        "Blur radius", renderer.ssfr_blur_radius, 1.0, 30.0, "%.0f",
                    )
                    if changed:
                        renderer.ssfr_blur_radius = val

                    changed, val = imgui.slider_float(
                        "Depth range", renderer.ssfr_depth_range, 0.01, 0.5, "%.2f",
                    )
                    if changed:
                        renderer.ssfr_depth_range = val

                    changed, val = imgui.slider_float(
                        "Absorption", renderer.ssfr_absorption_scale, 1.0, 100.0, "%.0f",
                    )
                    if changed:
                        renderer.ssfr_absorption_scale = val

                    changed, val = imgui.slider_float(
                        "Specular", renderer.ssfr_specular_power, 8.0, 256.0, "%.0f",
                    )
                    if changed:
                        renderer.ssfr_specular_power = val

                imgui.separator()

                # Foam controls
                changed, new_foam = imgui.checkbox("Foam / Spray", renderer.foam_enabled)
                if changed:
                    renderer.foam_enabled = new_foam

                if renderer.foam_enabled:
                    imgui.text(f"Foam particles: {renderer.num_foam:,}")

                imgui.separator()

                # Implicit surface tension controls (WCSPH only, < 100K)
                from solver_profiles import SolverType
                is_wcsph = sim._profile.solver_type == SolverType.WCSPH
                n_active = sim.world._high_water
                if not is_wcsph:
                    imgui.begin_disabled()
                changed, new_ist = imgui.checkbox(
                    "Implicit ST (< 100K)", sim.ist_enabled,
                )
                if changed:
                    sim.ist_enabled = new_ist
                if not is_wcsph:
                    imgui.end_disabled()

                if sim.ist_enabled and is_wcsph:
                    if n_active >= sim._IST_MAX_PARTICLES:
                        imgui.text_colored(
                            imgui.ImVec4(1.0, 0.5, 0.0, 1.0),
                            f"Disabled: {n_active:,} > {sim._IST_MAX_PARTICLES:,}",
                        )
                    changed, val = imgui.slider_float(
                        "ST Sigma", sim.ist_sigma, 0.01, 2.0, "%.2f",
                    )
                    if changed:
                        sim.ist_sigma = val
                        sim.update_ist_params()
                    changed, val = imgui.slider_int(
                        "ST Iters", sim.ist_iterations, 1, 20,
                    )
                    if changed:
                        sim.ist_iterations = val
                        sim.update_ist_params()

            imgui.end()

        # --- Kernel Timing Panel ---
        if sim.timing_enabled and sim._timing_ema:
            vp_timing = imgui.get_main_viewport()
            imgui.set_next_window_pos(
                imgui.ImVec2(vp_timing.pos.x + vp_timing.size.x - 310, 10),
                imgui.Cond_.first_use_ever,
            )
            imgui.set_next_window_size(
                imgui.ImVec2(300, 0), imgui.Cond_.first_use_ever,
            )
            imgui.set_next_window_bg_alpha(0.8)
            if imgui.begin(
                "Kernel Timings",
                None,
                imgui.WindowFlags_.always_auto_resize,
            )[0]:
                total = 0.0
                order = [
                    "hash", "sort", "reorder", "step1",
                    "reactions", "spawn", "step2", "integrate", "wake",
                ]
                for name in order:
                    ms = sim._timing_ema.get(name, 0.0)
                    total += ms
                    bar_frac = min(ms / 2.0, 1.0)  # 2ms = full bar
                    imgui.text(f"{name:>9s}")
                    imgui.same_line()
                    imgui.progress_bar(bar_frac, imgui.ImVec2(120, 14), "")
                    imgui.same_line()
                    imgui.text(f"{ms:6.2f}ms")
                imgui.separator()
                imgui.text(f"{'total':>9s}             {total:6.2f}ms")
            imgui.end()

        # --- Status Bar ---
        vp = imgui.get_main_viewport()
        status_h = 30.0
        imgui.set_next_window_pos(
            imgui.ImVec2(vp.pos.x, vp.pos.y + vp.size.y - status_h),
        )
        imgui.set_next_window_size(imgui.ImVec2(vp.size.x, status_h))
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_focus_on_appearing
            | imgui.WindowFlags_.no_nav
            | imgui.WindowFlags_.no_inputs
        )
        imgui.set_next_window_bg_alpha(0.65)

        if imgui.begin("##StatusBar", None, flags)[0]:
            active = world._high_water
            max_p = world.max_particles
            dt_ms = sim.dt * 1000.0
            substeps = sim.last_substeps
            speed = sim.speed
            paused_str = "PAUSED" if sim.paused else ""
            dt_mode = "FIXED" if sim.fixed_dt else "CFL"

            imgui.text(
                f"  Particles: {active:,}/{max_p:,}  |  "
                f"FPS: {fps:.0f}  |  "
                f"dt: {dt_ms:.2f}ms ({dt_mode})  |  "
                f"substeps: {substeps}  |  "
                f"speed: {speed:.1f}x  "
                f"{paused_str}"
            )

        imgui.end()

        return changes

    def get_cursor_world_pos(self, camera) -> Optional[np.ndarray]:
        """Compute 3D cursor position from current mouse pos when brush is active.

        Returns world-space intersection with y=0 plane, or None if brush
        mode is off or mouse is over ImGui.
        """
        if self._selected_material is None:
            return None
        if imgui.get_io().want_capture_mouse:
            return None

        mx, my = glfw.get_cursor_pos(self._window)
        view = camera.view_matrix()
        proj = camera.projection_matrix()
        origin, direction = _unproject_mouse_ray(
            mx, my, self._viewport_w, self._viewport_h, view, proj,
        )
        return _ray_plane_intersect(origin, direction, plane_y=0.0)

    def handle_key_shortcuts(self, key: int, action: int, sim) -> bool:
        """Handle keyboard shortcuts for material selection etc.

        Returns True if the key was consumed.
        """
        if action != glfw.PRESS:
            return False

        # Quick material select: 1-9 (toggle off if already selected)
        if key in _QUICK_SELECT:
            mat = _QUICK_SELECT[key]
            if self._selected_material == mat:
                self._selected_material = None
            else:
                self._selected_material = mat
            return True

        return False

    def shutdown(self) -> None:
        """Clean up ImGui resources."""
        self._imgui_renderer.shutdown()
        imgui.destroy_context()
