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
import os
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
from rigid_bodies import SDF_BOX, SDF_SPHERE, SDF_CYLINDER, SDF_PLANE

_SDF_TYPE_NAMES = ["Box", "Sphere", "Cylinder", "Plane"]
_MOTION_TYPE_NAMES = ["rotate_y", "oscillate_x", "oscillate_y", "oscillate_z"]

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


def _euler_to_quat(pitch: float, yaw: float, roll: float) -> tuple:
    """Convert euler angles (degrees) to quaternion (x,y,z,w)."""
    p = math.radians(pitch) * 0.5
    y = math.radians(yaw) * 0.5
    r = math.radians(roll) * 0.5
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    sr, cr = math.sin(r), math.cos(r)
    qx = sp * cy * cr - cp * sy * sr
    qy = cp * sy * cr + sp * cy * sr
    qz = cp * cy * sr - sp * sy * cr
    qw = cp * cy * cr + sp * sy * sr
    return (qx, qy, qz, qw)


def _quat_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple:
    """Convert quaternion (x,y,z,w) to euler angles (degrees): pitch, yaw, roll."""
    # pitch (X)
    sinp = 2.0 * (qw * qx + qy * qz)
    cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    pitch = math.atan2(sinp, cosp)
    # yaw (Y)
    siny = 2.0 * (qw * qy - qz * qx)
    siny = max(-1.0, min(1.0, siny))
    yaw = math.asin(siny)
    # roll (Z)
    sinr = 2.0 * (qw * qz + qx * qy)
    cosr = 1.0 - 2.0 * (qy * qy + qz * qz)
    roll = math.atan2(sinr, cosr)
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


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


def _ray_sphere_intersect(
    origin: np.ndarray, direction: np.ndarray,
    center: np.ndarray, radius: float,
) -> float:
    """Ray-sphere intersection. Returns t >= 0 or inf if no hit."""
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - c
    if disc < 0:
        return float('inf')
    sqrt_disc = math.sqrt(disc)
    t = -b - sqrt_disc
    if t < 0:
        t = -b + sqrt_disc
    return t if t >= 0 else float('inf')


def _ray_box_intersect(
    origin: np.ndarray, direction: np.ndarray,
    center: np.ndarray, half_ext: np.ndarray,
    quat: tuple,
) -> float:
    """Ray-oriented box intersection. Returns t >= 0 or inf if no hit."""
    # Transform ray to box local space
    rel = origin - center
    qx, qy, qz, qw = quat
    # Inverse quaternion rotation (conjugate)
    local_origin = _quat_rotate_inv(qx, qy, qz, qw, rel)
    local_dir = _quat_rotate_inv(qx, qy, qz, qw, direction)

    # AABB slab test in local space
    tmin = -1e30
    tmax = 1e30
    for i in range(3):
        if abs(local_dir[i]) < 1e-8:
            if abs(local_origin[i]) > half_ext[i]:
                return float('inf')
        else:
            inv_d = 1.0 / local_dir[i]
            t1 = (-half_ext[i] - local_origin[i]) * inv_d
            t2 = (half_ext[i] - local_origin[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return float('inf')
    return tmin if tmin >= 0 else (tmax if tmax >= 0 else float('inf'))


def _ray_cylinder_intersect(
    origin: np.ndarray, direction: np.ndarray,
    center: np.ndarray, radius: float, half_height: float,
    quat: tuple,
) -> float:
    """Ray-oriented cylinder intersection (Y-axis aligned in local space)."""
    rel = origin - center
    qx, qy, qz, qw = quat
    lo = _quat_rotate_inv(qx, qy, qz, qw, rel)
    ld = _quat_rotate_inv(qx, qy, qz, qw, direction)

    # Infinite cylinder: x^2 + z^2 = r^2
    a = ld[0] * ld[0] + ld[2] * ld[2]
    b = 2.0 * (lo[0] * ld[0] + lo[2] * ld[2])
    c = lo[0] * lo[0] + lo[2] * lo[2] - radius * radius

    best_t = float('inf')

    if abs(a) > 1e-10:
        disc = b * b - 4.0 * a * c
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            for sign in (-1, 1):
                t = (-b + sign * sqrt_disc) / (2.0 * a)
                if t >= 0:
                    y = lo[1] + t * ld[1]
                    if abs(y) <= half_height:
                        best_t = min(best_t, t)

    # Cap intersections (y = +/- half_height)
    if abs(ld[1]) > 1e-8:
        for cap_y in (-half_height, half_height):
            t = (cap_y - lo[1]) / ld[1]
            if t >= 0:
                hx = lo[0] + t * ld[0]
                hz = lo[2] + t * ld[2]
                if hx * hx + hz * hz <= radius * radius:
                    best_t = min(best_t, t)

    return best_t


def _ray_sdf_plane_intersect(
    origin: np.ndarray, direction: np.ndarray,
    center: np.ndarray, normal: np.ndarray,
) -> float:
    """Ray-plane intersection for SDF planes. Returns t or inf."""
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-8:
        return float('inf')
    t = float(np.dot(center - origin, normal)) / denom
    return t if t >= 0 else float('inf')


def _quat_rotate_inv(qx, qy, qz, qw, v):
    """Rotate vector v by inverse (conjugate) of quaternion (qx,qy,qz,qw)."""
    # Conjugate = (-qx, -qy, -qz, qw)
    # q * v * q_inv, but for inverse rotation we use conjugate * v * q
    # Using the standard formula: v' = v + 2*q_w*(q_v x v) + 2*(q_v x (q_v x v))
    # For inverse: negate q_v
    nqx, nqy, nqz = -qx, -qy, -qz
    # cross(q_v, v)
    cx = nqy * v[2] - nqz * v[1]
    cy = nqz * v[0] - nqx * v[2]
    cz = nqx * v[1] - nqy * v[0]
    # cross(q_v, cross_result)
    cx2 = nqy * cz - nqz * cy
    cy2 = nqz * cx - nqx * cz
    cz2 = nqx * cy - nqy * cx
    return np.array([
        v[0] + 2.0 * (qw * cx + cx2),
        v[1] + 2.0 * (qw * cy + cy2),
        v[2] + 2.0 * (qw * cz + cz2),
    ], dtype=np.float32)


def _raycast_sdf_objects(origin, direction, sdf_manager):
    """Raycast against all SDF objects. Returns (hit_id, hit_t) or (None, inf)."""
    objects = sdf_manager.get_sdf_objects()
    best_id = None
    best_t = float('inf')

    for obj in objects:
        center = np.array(obj["position"], dtype=np.float32)
        rot = tuple(obj["rotation"])
        sdf_type = obj["type"]

        if sdf_type == SDF_SPHERE:
            t = _ray_sphere_intersect(origin, direction, center, obj["size"][0])
        elif sdf_type == SDF_BOX:
            t = _ray_box_intersect(origin, direction, center,
                                   np.array(obj["size"], dtype=np.float32), rot)
        elif sdf_type == SDF_CYLINDER:
            t = _ray_cylinder_intersect(origin, direction, center,
                                        obj["size"][0], obj["size"][1], rot)
        elif sdf_type == SDF_PLANE:
            normal = np.array(obj["size"], dtype=np.float32)
            nlen = np.linalg.norm(normal)
            if nlen > 1e-8:
                normal /= nlen
            t = _ray_sdf_plane_intersect(origin, direction, center, normal)
        else:
            continue

        if t < best_t:
            best_t = t
            best_id = obj["id"]

    return best_id, best_t


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
        self._vsync_enabled: bool = True  # matches main.py swap_interval(1)

        # SDF object drag state
        self._selected_sdf_id: Optional[int] = None
        self._dragging_sdf: bool = False
        self._drag_plane_origin: Optional[np.ndarray] = None  # point on drag plane
        self._drag_plane_normal: Optional[np.ndarray] = None  # drag plane normal (camera forward)
        self._drag_offset: Optional[np.ndarray] = None  # offset from hit point to object center
        self._sdf_manager_ref = None  # set via set_sdf_drag_refs()
        self._camera_ref = None  # set via set_sdf_drag_refs()
        self._drag_shift: bool = False  # shift held = Y-axis constraint

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

        # Save/load state
        self._save_filename: str = "scene"
        self._save_status: str = ""
        self._load_files: list = []
        self._load_selected: int = 0

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

    def set_sdf_drag_refs(self, camera, sdf_manager):
        """Store references needed for SDF object drag handles."""
        self._camera_ref = camera
        self._sdf_manager_ref = sdf_manager

    @property
    def selected_sdf_id(self) -> Optional[int]:
        """Currently selected SDF object ID, or None."""
        return self._selected_sdf_id

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

            # No brush selected: try to pick SDF objects
            if self._sdf_manager_ref is not None and self._camera_ref is not None:
                mx, my = glfw.get_cursor_pos(window)
                cam = self._camera_ref
                view = cam.view_matrix()
                proj = cam.projection_matrix()
                origin, direction = _unproject_mouse_ray(
                    mx, my, self._viewport_w, self._viewport_h, view, proj,
                )
                hit_id, hit_t = _raycast_sdf_objects(origin, direction, self._sdf_manager_ref)
                if hit_id is not None and hit_t < float('inf'):
                    self._selected_sdf_id = hit_id
                    self._dragging_sdf = True
                    self._drag_shift = bool(mods & glfw.MOD_SHIFT)

                    # Compute drag plane: perpendicular to camera forward, through object center
                    obj_center = np.array(
                        self._sdf_manager_ref.get_sdf_objects()[hit_id]["position"],
                        dtype=np.float32,
                    )
                    cam_fwd = cam.target - cam.position
                    cam_fwd = cam_fwd / np.linalg.norm(cam_fwd)
                    self._drag_plane_normal = cam_fwd
                    self._drag_plane_origin = obj_center.copy()

                    # Offset from hit point to object center (so dragging feels natural)
                    hit_point = origin + hit_t * direction
                    self._drag_offset = obj_center - hit_point
                    return
                else:
                    self._selected_sdf_id = None

        # Left release: finish drag, upload to GPU
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            if self._dragging_sdf:
                self._dragging_sdf = False
                if self._sdf_manager_ref is not None:
                    self._sdf_manager_ref.upload_if_dirty()
                # Don't pass to camera
                return

        # Right click: deselect
        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            if self._selected_sdf_id is not None:
                self._selected_sdf_id = None
                self._dragging_sdf = False
                return

        # Pass through to user callbacks (camera controls)
        if self._user_mouse_button_cb:
            self._user_mouse_button_cb(window, button, action, mods)

    def _on_cursor_pos(self, window, xpos, ypos):
        self._imgui_renderer.mouse_callback(window, xpos, ypos)
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        # SDF object dragging
        if self._dragging_sdf and self._sdf_manager_ref is not None and self._camera_ref is not None:
            cam = self._camera_ref
            view = cam.view_matrix()
            proj = cam.projection_matrix()
            origin, direction = _unproject_mouse_ray(
                xpos, ypos, self._viewport_w, self._viewport_h, view, proj,
            )

            # Intersect ray with drag plane
            denom = float(np.dot(direction, self._drag_plane_normal))
            if abs(denom) > 1e-8:
                t = float(np.dot(self._drag_plane_origin - origin, self._drag_plane_normal)) / denom
                if t >= 0:
                    hit = origin + t * direction + self._drag_offset
                    if self._drag_shift:
                        # Shift: constrain to Y axis only
                        old_pos = self._sdf_manager_ref.get_sdf_objects()[self._selected_sdf_id]["position"]
                        new_pos = [old_pos[0], float(hit[1]), old_pos[2]]
                    else:
                        new_pos = [float(hit[0]), float(hit[1]), float(hit[2])]
                    self._sdf_manager_ref.update_sdf_object(self._selected_sdf_id, position=new_pos)
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

        # --- Save/Load Panel ---
        imgui.set_next_window_pos(imgui.ImVec2(240, 340), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(220, 0), imgui.Cond_.first_use_ever)

        if imgui.begin("Save / Load", None, imgui.WindowFlags_.always_auto_resize)[0]:
            from save_load import save_scene, load_scene, list_saves

            # Save section
            changed, self._save_filename = imgui.input_text(
                "Name", self._save_filename, 64,
            )
            if imgui.button("Save", imgui.ImVec2(100, 28)):
                camera = self._camera_ref
                if camera is not None:
                    path = save_scene(world, sim, camera, self._save_filename)
                    self._save_status = f"Saved: {os.path.basename(path)}"
                    self._load_files = list_saves()
                else:
                    self._save_status = "Error: no camera ref"
            if self._save_status:
                imgui.text(self._save_status)

            imgui.separator()

            # Load section
            if imgui.button("Refresh", imgui.ImVec2(100, 28)):
                self._load_files = list_saves()
            if not self._load_files:
                self._load_files = list_saves()
            if self._load_files:
                file_list = self._load_files
                changed, self._load_selected = imgui.combo(
                    "File", self._load_selected,
                    file_list,
                )
                if imgui.button("Load", imgui.ImVec2(100, 28)):
                    camera = self._camera_ref
                    if camera is not None and self._load_selected < len(file_list):
                        n = load_scene(world, sim, camera, file_list[self._load_selected])
                        self._save_status = f"Loaded: {n:,} particles"
                        changes['_scene_loaded'] = True
            else:
                imgui.text("No saves found")
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

            # Gravity slider
            changed, new_gy = imgui.slider_float(
                "Gravity", sim.gravity_y, -20.0, 5.0, "%.1f m/s²",
            )
            if changed:
                sim.set_gravity(new_gy)

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

            imgui.same_line()

            # Vsync toggle
            changed, new_vsync = imgui.checkbox("Vsync", self._vsync_enabled)
            if changed:
                self._vsync_enabled = new_vsync
                glfw.swap_interval(1 if new_vsync else 0)

            # Snapshot / Undo info
            snaps = getattr(self, 'snapshots', None)
            if snaps is not None:
                imgui.separator()
                imgui.text(f"Snapshots: {snaps.num_snapshots}/{snaps.max_snapshots}")
                imgui.text("Ctrl+Z to undo")
                if snaps.num_snapshots > 0:
                    old_steps = getattr(self, '_undo_steps', 1)
                    changed, new_steps = imgui.slider_int(
                        "Rewind", old_steps, 1, snaps.num_snapshots,
                    )
                    if changed:
                        self._undo_steps = new_steps
                    if imgui.button("Restore##undo", imgui.ImVec2(100, 28)):
                        steps = getattr(self, '_undo_steps', 1)
                        if snaps.restore(world, steps_back=steps):
                            changes['_scene_loaded'] = True

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

                # Motion blur controls
                changed, new_mb = imgui.checkbox("Motion blur", renderer.motion_blur_enabled)
                if changed:
                    renderer.motion_blur_enabled = new_mb
                if renderer.motion_blur_enabled:
                    changed, new_ts = imgui.slider_float(
                        "Trail scale", renderer.trail_scale, 0.1, 5.0, "%.1f",
                    )
                    if changed:
                        renderer.trail_scale = new_ts

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

        # --- SDF Objects Panel ---
        if renderer is not None and hasattr(sim, 'sdf_manager'):
            sdf_mgr = sim.sdf_manager
            imgui.set_next_window_pos(imgui.ImVec2(240, 10), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(imgui.ImVec2(260, 0), imgui.Cond_.first_use_ever)

            if imgui.begin("Objects", None, imgui.WindowFlags_.always_auto_resize)[0]:
                # Show/hide toggle
                changed, vis = imgui.checkbox("Show Objects", renderer.sdf_objects_visible)
                if changed:
                    renderer.sdf_objects_visible = vis

                imgui.text(f"Objects: {sdf_mgr.count}/16")

                # Add object combo
                if sdf_mgr.count < 16:
                    if imgui.button("+ Add Object"):
                        imgui.open_popup("add_sdf_popup")
                    if imgui.begin_popup("add_sdf_popup"):
                        for type_idx, type_name in enumerate(_SDF_TYPE_NAMES):
                            if imgui.selectable(type_name)[0]:
                                default_size = (0.2, 0.2, 0.2) if type_idx != SDF_SPHERE else (0.2, 0, 0)
                                sdf_mgr.add_sdf_object(sdf_type=type_idx, position=(0, 0, 0), size=default_size)
                        imgui.end_popup()

                imgui.separator()

                # Per-object controls
                objects = sdf_mgr.get_sdf_objects()
                to_remove = None
                for obj in objects:
                    oid = obj["id"]
                    type_name = _SDF_TYPE_NAMES[obj["type"]] if obj["type"] < len(_SDF_TYPE_NAMES) else "?"
                    expanded = imgui.collapsing_header(f"{type_name} #{oid}")
                    if expanded:
                        # Delete button
                        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.7, 0.1, 0.1, 1))
                        if imgui.button(f"Delete##{oid}"):
                            to_remove = oid
                        imgui.pop_style_color()

                        # Position
                        pos = obj["position"]
                        ch, vals = imgui.slider_float3(f"Pos##{oid}", [pos[0], pos[1], pos[2]], -3.0, 3.0, "%.2f")
                        if ch:
                            sdf_mgr.update_sdf_object(oid, position=list(vals))

                        # Size
                        sz = obj["size"]
                        if obj["type"] == SDF_SPHERE:
                            ch, val = imgui.slider_float(f"Radius##{oid}", sz[0], 0.01, 1.0, "%.2f")
                            if ch:
                                sdf_mgr.update_sdf_object(oid, size=[val, 0, 0])
                        else:
                            ch, vals = imgui.slider_float3(f"Size##{oid}", [sz[0], sz[1], sz[2]], 0.01, 1.0, "%.2f")
                            if ch:
                                sdf_mgr.update_sdf_object(oid, size=list(vals))

                        # Rotation (euler angles)
                        rot = obj["rotation"]
                        euler = _quat_to_euler(rot[0], rot[1], rot[2], rot[3])
                        ch, vals = imgui.slider_float3(f"Rot##{oid}", [euler[0], euler[1], euler[2]], -180.0, 180.0, "%.0f")
                        if ch:
                            q = _euler_to_quat(vals[0], vals[1], vals[2])
                            sdf_mgr.update_sdf_object(oid, rotation=list(q))

                        # Restitution & friction
                        ch, val = imgui.slider_float(f"Bounce##{oid}", obj["restitution"], 0.0, 1.0, "%.2f")
                        if ch:
                            sdf_mgr.update_sdf_object(oid, restitution=val)
                        ch, val = imgui.slider_float(f"Friction##{oid}", obj["friction"], 0.0, 1.0, "%.2f")
                        if ch:
                            sdf_mgr.update_sdf_object(oid, friction=val)

                        # Kinematic motion
                        has_motion = oid in sdf_mgr._motions
                        ch, kin = imgui.checkbox(f"Kinematic##{oid}", has_motion)
                        if ch:
                            if kin:
                                sdf_mgr.add_kinematic_motion(oid, "rotate_y", {"speed": 1.0})
                            else:
                                sdf_mgr.remove_kinematic_motion(oid)

                        if oid in sdf_mgr._motions:
                            motion = sdf_mgr._motions[oid]
                            mtype = motion["type"]
                            cur_idx = _MOTION_TYPE_NAMES.index(mtype) if mtype in _MOTION_TYPE_NAMES else 0
                            ch, new_idx = imgui.combo(f"Motion##{oid}", cur_idx, _MOTION_TYPE_NAMES)
                            if ch and new_idx != cur_idx:
                                new_type = _MOTION_TYPE_NAMES[new_idx]
                                if new_type == "rotate_y":
                                    sdf_mgr.add_kinematic_motion(oid, new_type, {"speed": 1.0})
                                else:
                                    sdf_mgr.add_kinematic_motion(oid, new_type, {"amplitude": 0.3, "frequency": 0.5})

                            p = motion["params"]
                            if mtype == "rotate_y":
                                ch, val = imgui.slider_float(f"Speed##{oid}", p.get("speed", 1.0), 0.1, 10.0, "%.1f rad/s")
                                if ch:
                                    p["speed"] = val
                            else:
                                ch, val = imgui.slider_float(f"Amp##{oid}", p.get("amplitude", 0.3), 0.01, 2.0, "%.2f m")
                                if ch:
                                    p["amplitude"] = val
                                ch, val = imgui.slider_float(f"Freq##{oid}", p.get("frequency", 0.5), 0.1, 5.0, "%.1f Hz")
                                if ch:
                                    p["frequency"] = val

                if to_remove is not None:
                    sdf_mgr.remove_kinematic_motion(to_remove)
                    sdf_mgr.remove_sdf_object(to_remove)

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
