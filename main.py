"""SPH simulation with OpenGL rendering.

Usage: python main.py [--particles N] [--speed S] [--dt DT]

Controls:
  Space    -- pause/resume simulation
  R        -- reset to initial configuration
  +/=      -- increase simulation speed by 0.2
  -        -- decrease simulation speed by 0.2
  Escape   -- quit
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# OpenGL / GLFW imports (pyglet-free: raw GLFW + OpenGL via ctypes or PyOpenGL)
# ---------------------------------------------------------------------------

try:
    import glfw  # type: ignore[import-untyped]
    from OpenGL import GL  # type: ignore[import-untyped]
except ImportError:
    print(
        "Missing dependencies. Install with:\n"
        "  pip install glfw PyOpenGL PyOpenGL_accelerate\n"
        "On Windows you may also need the GLFW DLL on PATH."
    )
    sys.exit(1)

import cupy  # type: ignore[import-untyped]

from gl_cuda_interop import CudaGLBuffer
from simulation import MAX_SUBSTEPS, SPHSimulation, setup_initial_scene

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_W = 1280
WINDOW_H = 720
WINDOW_TITLE = "SPH Simulation (US-014)"

DEFAULT_PARTICLES = 20_000
DEFAULT_SPEED = 1.0
DEFAULT_DT = 0.001


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """OpenGL point-sprite renderer with CUDA-GL interop."""

    def __init__(self, num_particles: int) -> None:
        self.num_particles = num_particles
        self.vbo_pos: int = 0
        self.vbo_color: int = 0
        self.cuda_pos: Optional[CudaGLBuffer] = None
        self.cuda_color: Optional[CudaGLBuffer] = None
        self._init_gl()

    def _init_gl(self) -> None:
        """Create OpenGL VBOs and register with CUDA."""
        n = self.num_particles

        # Position VBO: (N, 4) float32
        pos_bytes = n * 4 * 4  # N * 4 floats * 4 bytes
        self.vbo_pos = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, pos_bytes, None, GL.GL_DYNAMIC_DRAW)

        # Color VBO: (N, 4) float32
        color_bytes = n * 4 * 4
        self.vbo_color = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_color)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color_bytes, None, GL.GL_DYNAMIC_DRAW)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        # Register with CUDA
        self.cuda_pos = CudaGLBuffer(self.vbo_pos, pos_bytes)
        self.cuda_color = CudaGLBuffer(self.vbo_color, color_bytes)

    def update_from_simulation(self, sim: SPHSimulation) -> None:
        """Copy unsorted pos/color from simulation to VBOs via CUDA-GL interop."""
        n = self.num_particles

        # Map both VBOs
        self.cuda_pos.map()
        self.cuda_color.map()

        try:
            # Get device pointers as CuPy arrays
            pos_arr = self.cuda_pos.device_pointer_as_cupy_array(
                (n, 4), dtype=np.float32
            )
            color_arr = self.cuda_color.device_pointer_as_cupy_array(
                (n, 4), dtype=np.float32
            )

            # Copy unsorted pos/color to VBO (device-to-device, zero-copy write)
            pos_arr[:] = sim.position
            color_arr[:] = sim.color
        finally:
            self.cuda_color.unmap()
            self.cuda_pos.unmap()

    def draw(self) -> None:
        """Render particles as GL_POINTS."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)

        # Bind position VBO
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_pos)
        GL.glVertexPointer(4, GL.GL_FLOAT, 0, None)

        # Bind color VBO
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_color)
        GL.glColorPointer(4, GL.GL_FLOAT, 0, None)

        GL.glDrawArrays(GL.GL_POINTS, 0, self.num_particles)

        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def cleanup(self) -> None:
        """Unregister CUDA resources and delete VBOs."""
        if self.cuda_pos is not None:
            self.cuda_pos.close()
        if self.cuda_color is not None:
            self.cuda_color.close()
        if self.vbo_pos:
            GL.glDeleteBuffers(1, [self.vbo_pos])
        if self.vbo_color:
            GL.glDeleteBuffers(1, [self.vbo_color])


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class Application:
    """Main application managing window, simulation, and rendering."""

    def __init__(
        self,
        num_particles: int = DEFAULT_PARTICLES,
        speed: float = DEFAULT_SPEED,
        dt: float = DEFAULT_DT,
    ) -> None:
        self.num_particles = num_particles
        self.speed = speed
        self.dt = dt
        self.window = None
        self.sim: Optional[SPHSimulation] = None
        self.renderer: Optional[Renderer] = None
        self.last_time = 0.0
        self.frame_count = 0
        self.fps_time = 0.0
        self.fps = 0.0

    def _init_window(self) -> None:
        """Initialize GLFW window and OpenGL context."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        self.window = glfw.create_window(
            WINDOW_W, WINDOW_H, WINDOW_TITLE, None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self._key_callback)

        # OpenGL setup
        GL.glClearColor(0.05, 0.05, 0.1, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPointSize(2.0)

        # Simple orthographic-like projection for the [-1,1]^3 grid
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        # Perspective with reasonable FOV
        aspect = WINDOW_W / WINDOW_H
        _gluPerspective(45.0, aspect, 0.1, 100.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        # Camera looking at origin from (0, 0.5, 3)
        _gluLookAt(0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def _key_callback(
        self,
        window: object,
        key: int,
        scancode: int,
        action: int,
        mods: int,
    ) -> None:
        """Handle keyboard input."""
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
        elif key == glfw.KEY_SPACE:
            self.sim.toggle_pause()
        elif key == glfw.KEY_R:
            self._reset()
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            # + key
            self.sim.adjust_speed(0.2)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            # - key
            self.sim.adjust_speed(-0.2)

    def _reset(self) -> None:
        """Reset simulation to initial configuration."""
        self.sim = SPHSimulation(
            self.num_particles, dt=self.dt, speed=self.sim.speed
        )
        setup_initial_scene(self.sim)

    def _update_title(self, wall_dt: float, substeps: int) -> None:
        """Update window title with FPS and sim info."""
        self.frame_count += 1
        self.fps_time += wall_dt
        if self.fps_time >= 1.0:
            self.fps = self.frame_count / self.fps_time
            self.frame_count = 0
            self.fps_time = 0.0

        paused_str = " [PAUSED]" if self.sim.paused else ""
        title = (
            f"SPH {self.num_particles // 1000}K | "
            f"FPS: {self.fps:.0f} | "
            f"Steps: {substeps} | "
            f"Speed: {self.sim.speed:.1f}x | "
            f"t={self.sim.sim_time:.3f}{paused_str}"
        )
        glfw.set_window_title(self.window, title)

    def run(self) -> None:
        """Main loop."""
        self._init_window()

        # Create simulation
        self.sim = SPHSimulation(
            self.num_particles, dt=self.dt, speed=self.speed
        )
        setup_initial_scene(self.sim)

        # Create renderer (needs GL context)
        self.renderer = Renderer(self.num_particles)

        self.last_time = time.perf_counter()

        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()

                now = time.perf_counter()
                wall_dt = now - self.last_time
                self.last_time = now

                # Clamp wall_dt to avoid huge jumps (e.g., after window drag)
                wall_dt = min(wall_dt, 0.1)

                # Run simulation substeps
                substeps = self.sim.step_frame(wall_dt)

                # Render: copy sim data to VBOs, then draw
                self.renderer.update_from_simulation(self.sim)
                self.renderer.draw()

                glfw.swap_buffers(self.window)

                self._update_title(wall_dt, substeps)

        finally:
            if self.renderer:
                self.renderer.cleanup()
            glfw.terminate()


# ---------------------------------------------------------------------------
# GLU replacements (avoid dependency on PyOpenGL_accelerate for glu)
# ---------------------------------------------------------------------------


def _gluPerspective(fovy: float, aspect: float, znear: float, zfar: float) -> None:
    """Replacement for gluPerspective using raw GL calls."""
    import math

    fH = math.tan(fovy / 360.0 * math.pi) * znear
    fW = fH * aspect
    GL.glFrustum(-fW, fW, -fH, fH, znear, zfar)


def _gluLookAt(
    eyeX: float,
    eyeY: float,
    eyeZ: float,
    centerX: float,
    centerY: float,
    centerZ: float,
    upX: float,
    upY: float,
    upZ: float,
) -> None:
    """Replacement for gluLookAt using raw GL calls."""
    import math

    # Forward vector
    fx = centerX - eyeX
    fy = centerY - eyeY
    fz = centerZ - eyeZ
    flen = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx /= flen
    fy /= flen
    fz /= flen

    # Side = forward x up
    sx = fy * upZ - fz * upY
    sy = fz * upX - fx * upZ
    sz = fx * upY - fy * upX
    slen = math.sqrt(sx * sx + sy * sy + sz * sz)
    sx /= slen
    sy /= slen
    sz /= slen

    # Recompute up = side x forward
    ux = sy * fz - sz * fy
    uy = sz * fx - sx * fz
    uz = sx * fy - sy * fx

    m = [
        sx, ux, -fx, 0.0,
        sy, uy, -fy, 0.0,
        sz, uz, -fz, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    GL.glMultMatrixf(m)
    GL.glTranslatef(-eyeX, -eyeY, -eyeZ)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SPH Simulation Renderer")
    parser.add_argument(
        "--particles", type=int, default=DEFAULT_PARTICLES,
        help=f"Number of particles (default: {DEFAULT_PARTICLES})",
    )
    parser.add_argument(
        "--speed", type=float, default=DEFAULT_SPEED,
        help=f"Simulation speed multiplier (default: {DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--dt", type=float, default=DEFAULT_DT,
        help=f"Fixed timestep (default: {DEFAULT_DT})",
    )
    args = parser.parse_args()

    app = Application(
        num_particles=args.particles,
        speed=args.speed,
        dt=args.dt,
    )
    app.run()


if __name__ == "__main__":
    main()
