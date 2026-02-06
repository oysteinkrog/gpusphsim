"""Orbit camera with spherical coordinates, view/projection matrices."""

import math
import numpy as np


class OrbitCamera:
    """Orbit camera around a target point using spherical coordinates.

    Controls: right-drag=orbit, scroll=zoom, middle-drag=pan.
    """

    def __init__(
        self,
        target: np.ndarray = None,
        distance: float = 3.0,
        azimuth: float = 0.0,
        elevation: float = 30.0,
        fov: float = 45.0,
        near: float = 0.01,
        far: float = 100.0,
    ):
        self.target = np.array(target if target is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = distance
        self.azimuth = azimuth      # degrees, horizontal rotation
        self.elevation = elevation  # degrees, vertical rotation
        self.fov = fov              # vertical field of view in degrees
        self.near = near
        self.far = far
        self.aspect = 16.0 / 9.0

        # Zoom limits
        self.min_distance = 0.1
        self.max_distance = 50.0

        # Elevation limits (avoid gimbal lock)
        self.min_elevation = -89.0
        self.max_elevation = 89.0

        # Sensitivity
        self.orbit_sensitivity = 0.3
        self.zoom_sensitivity = 0.15
        self.pan_sensitivity = 0.003

    @property
    def position(self) -> np.ndarray:
        """Camera position in world space computed from spherical coords."""
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        cos_el = math.cos(el)
        x = self.target[0] + self.distance * cos_el * math.sin(az)
        y = self.target[1] + self.distance * math.sin(el)
        z = self.target[2] + self.distance * cos_el * math.cos(az)
        return np.array([x, y, z], dtype=np.float32)

    def orbit(self, dx: float, dy: float):
        """Rotate camera around target. dx/dy in pixels."""
        self.azimuth -= dx * self.orbit_sensitivity
        self.elevation += dy * self.orbit_sensitivity
        self.elevation = max(self.min_elevation, min(self.max_elevation, self.elevation))

    def zoom(self, delta: float):
        """Zoom in/out. Positive delta = zoom in."""
        factor = 1.0 - delta * self.zoom_sensitivity
        self.distance *= factor
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))

    def pan(self, dx: float, dy: float):
        """Pan the camera (move target). dx/dy in pixels."""
        az = math.radians(self.azimuth)
        # Right vector (horizontal)
        right = np.array([math.cos(az), 0.0, -math.sin(az)], dtype=np.float32)
        # Up vector (world up projected)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        scale = self.distance * self.pan_sensitivity
        self.target -= right * dx * scale
        self.target += up * dy * scale

    def view_matrix(self) -> np.ndarray:
        """Compute 4x4 view matrix (look-at)."""
        eye = self.position
        center = self.target
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return _look_at(eye, center, up)

    def projection_matrix(self) -> np.ndarray:
        """Compute 4x4 perspective projection matrix."""
        return _perspective(self.fov, self.aspect, self.near, self.far)

    def set_aspect(self, width: int, height: int):
        if height > 0:
            self.aspect = width / height


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute OpenGL-style look-at view matrix."""
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0]; m[0, 1] = s[1]; m[0, 2] = s[2]
    m[1, 0] = u[0]; m[1, 1] = u[1]; m[1, 2] = u[2]
    m[2, 0] = -f[0]; m[2, 1] = -f[1]; m[2, 2] = -f[2]
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Compute OpenGL-style perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m
