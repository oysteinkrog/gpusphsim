"""OpenGL point sprite renderer with CUDA-GL interop VBOs."""

from __future__ import annotations

import os
import numpy as np
from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
    glGetShaderInfoLog, glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glDeleteShader, glUseProgram,
    glGetUniformLocation, glUniformMatrix4fv, glUniform1f,
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
    glDrawArrays, glEnable, glDeleteBuffers, glDeleteVertexArrays,
    glDeleteProgram, glDepthFunc,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS,
    GL_LINK_STATUS, GL_TRUE, GL_FALSE, GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW, GL_FLOAT, GL_POINTS, GL_PROGRAM_POINT_SIZE,
    GL_DEPTH_TEST, GL_LESS,
)
from OpenGL.GL import shaders as _gl_shaders  # noqa: F401 -- just in case

from gl_cuda_interop import CudaGLBuffer

# Bytes per element
_FLOAT4_BYTES = 4 * 4  # 16 bytes per float4


def _read_shader_source(filename: str) -> str:
    """Read shader source from the shaders/ directory next to this module."""
    shader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
    path = os.path.join(shader_dir, filename)
    with open(path, "r") as f:
        return f.read()


def _compile_shader(source: str, shader_type: int) -> int:
    """Compile a GLSL shader; raise on error."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(shader)
        if isinstance(log, bytes):
            log = log.decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader


def _link_program(vert: int, frag: int) -> int:
    """Link vertex + fragment shaders into a program; raise on error."""
    program = glCreateProgram()
    glAttachShader(program, vert)
    glAttachShader(program, frag)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        log = glGetProgramInfoLog(program)
        if isinstance(log, bytes):
            log = log.decode()
        raise RuntimeError(f"Program link error:\n{log}")
    return program


class Renderer:
    """Point-sprite renderer backed by two CUDA-GL interop VBOs.

    Parameters
    ----------
    num_particles : int
        Maximum number of particles.
    point_scale : float
        Base point size divisor for distance-based sizing.
    """

    def __init__(self, num_particles: int, point_scale: float = 20.0):
        self.num_particles = num_particles
        self.num_active = num_particles
        self.point_scale = point_scale

        # Compile shaders
        vert_src = _read_shader_source("particle.vert")
        frag_src = _read_shader_source("particle.frag")
        vs = _compile_shader(vert_src, GL_VERTEX_SHADER)
        fs = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
        self._program = _link_program(vs, fs)
        glDeleteShader(vs)
        glDeleteShader(fs)

        # Uniform locations
        self._u_mvp = glGetUniformLocation(self._program, "uMVP")
        self._u_mv = glGetUniformLocation(self._program, "uMV")
        self._u_point_scale = glGetUniformLocation(self._program, "uPointScale")

        # Enable required GL state
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Create VBOs
        pos_nbytes = num_particles * _FLOAT4_BYTES
        col_nbytes = num_particles * _FLOAT4_BYTES

        self._vbo_pos = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, pos_nbytes, None, GL_DYNAMIC_DRAW)

        self._vbo_col = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_col)
        glBufferData(GL_ARRAY_BUFFER, col_nbytes, None, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Create VAO
        self._vao = int(glGenVertexArrays(1))
        glBindVertexArray(self._vao)

        # Position attribute (location=0): float4
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, _FLOAT4_BYTES, None)

        # Color attribute (location=1): float4
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_col)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, _FLOAT4_BYTES, None)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Register VBOs with CUDA
        self.cuda_pos = CudaGLBuffer(self._vbo_pos, pos_nbytes)
        self.cuda_col = CudaGLBuffer(self._vbo_col, col_nbytes)

    def draw(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Render point sprites.

        Parameters
        ----------
        view : np.ndarray
            4x4 view matrix (float32, row-major as from OrbitCamera).
        proj : np.ndarray
            4x4 projection matrix (float32, row-major).
        """
        mvp = proj @ view
        mv = view

        glUseProgram(self._program)
        glUniformMatrix4fv(self._u_mvp, 1, GL_TRUE, mvp)
        glUniformMatrix4fv(self._u_mv, 1, GL_TRUE, mv)
        glUniform1f(self._u_point_scale, self.point_scale)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self.num_active)
        glBindVertexArray(0)
        glUseProgram(0)

    def close(self) -> None:
        """Release GPU resources."""
        self.cuda_col.close()
        self.cuda_pos.close()
        glDeleteBuffers(2, [self._vbo_pos, self._vbo_col])
        glDeleteVertexArrays(1, [self._vao])
        glDeleteProgram(self._program)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
