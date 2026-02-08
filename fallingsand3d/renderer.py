"""OpenGL renderer with CUDA-GL interop VBOs.

Supports two modes:
- Point sprites (default): simple round point sprites for all particles
- SSFR (Screen-Space Fluid Rendering): multi-pass fluid surface rendering
  with Beer-Lambert absorption, Fresnel reflections, and specular highlights.
  Non-FLUID particles still rendered as point sprites on top.
"""

from __future__ import annotations

import os
import ctypes
import numpy as np
from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
    glGetShaderInfoLog, glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glDeleteShader, glUseProgram,
    glGetUniformLocation, glUniformMatrix4fv, glUniformMatrix3fv,
    glUniform1f, glUniform1i,
    glUniform2f, glUniform3f, glUniform4f,
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glBufferSubData, glEnableVertexAttribArray, glVertexAttribPointer,
    glDrawArrays, glEnable, glDisable, glDeleteBuffers, glDeleteVertexArrays,
    glDeleteProgram, glDepthFunc, glDepthMask, glBlendFunc, glClear,
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    glCheckFramebufferStatus, glDeleteFramebuffers,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glDeleteTextures, glActiveTexture,
    glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage,
    glFramebufferRenderbuffer, glDeleteRenderbuffers,
    glViewport, glClearColor,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS,
    GL_LINK_STATUS, GL_TRUE, GL_FALSE, GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW, GL_STATIC_DRAW, GL_FLOAT, GL_POINTS, GL_TRIANGLES,
    GL_LINE_LOOP,
    GL_PROGRAM_POINT_SIZE,
    GL_DEPTH_TEST, GL_LESS, GL_LEQUAL, GL_BLEND, GL_ONE, GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT,
    GL_FRAMEBUFFER_COMPLETE,
    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
    GL_NEAREST, GL_LINEAR, GL_CLAMP_TO_EDGE,
    GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE4,
    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_R32F, GL_R16F,
    GL_RED, GL_RGB, GL_RGBA16F, GL_RGBA, GL_UNSIGNED_BYTE,
)
from OpenGL.GL import shaders as _gl_shaders  # noqa: F401

from gl_cuda_interop import CudaGLBuffer

_FLOAT4_BYTES = 4 * 4


def _read_shader_source(filename: str) -> str:
    shader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
    path = os.path.join(shader_dir, filename)
    with open(path, "r") as f:
        return f.read()


def _compile_shader(source: str, shader_type: int) -> int:
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


def _build_program(vert_file: str, frag_file: str) -> int:
    """Compile and link a shader program from filenames."""
    vs = _compile_shader(_read_shader_source(vert_file), GL_VERTEX_SHADER)
    fs = _compile_shader(_read_shader_source(frag_file), GL_FRAGMENT_SHADER)
    prog = _link_program(vs, fs)
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


def _create_texture(width: int, height: int, internal_fmt, fmt, dtype) -> int:
    """Create an empty 2D texture with nearest filtering and clamp-to-edge."""
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, width, height, 0, fmt, dtype, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def _load_cubemap(face_paths: dict) -> int:
    """Load a cubemap texture from 6 face image files.

    face_paths: dict mapping GL face enum to file path.
    Returns the GL texture ID.
    """
    from PIL import Image
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex)
    for face_enum, path in face_paths.items():
        img = Image.open(path).convert("RGB")
        # Cubemap faces: no vertical flip (unlike regular 2D textures)
        data = np.asarray(img, dtype=np.uint8)
        h, w = data.shape[:2]
        glTexImage2D(face_enum, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    return tex


# Skybox cube vertices (unit cube centered at origin)
_SKYBOX_VERTICES = np.array([
    # Back face (-Z)
    -1, -1, -1,   1, -1, -1,   1,  1, -1,
     1,  1, -1,  -1,  1, -1,  -1, -1, -1,
    # Front face (+Z)
    -1, -1,  1,   1,  1,  1,   1, -1,  1,
    -1, -1,  1,  -1,  1,  1,   1,  1,  1,
    # Left face (-X)
    -1, -1, -1,  -1,  1,  1,  -1, -1,  1,
    -1, -1, -1,  -1,  1, -1,  -1,  1,  1,
    # Right face (+X)
     1, -1, -1,   1, -1,  1,   1,  1,  1,
     1,  1,  1,   1,  1, -1,   1, -1, -1,
    # Bottom face (-Y)
    -1, -1, -1,   1, -1,  1,   1, -1, -1,
    -1, -1, -1,  -1, -1,  1,   1, -1,  1,
    # Top face (+Y)
    -1,  1, -1,   1,  1, -1,   1,  1,  1,
     1,  1,  1,  -1,  1,  1,  -1,  1, -1,
], dtype=np.float32)


class Renderer:
    """Multi-mode renderer: point sprites + optional SSFR fluid rendering.

    Parameters
    ----------
    num_particles : int
        Maximum number of particles.
    point_scale : float
        Base point size divisor for distance-based sizing.
    width, height : int
        Initial framebuffer dimensions for FBO allocation.
    """

    def __init__(self, num_particles: int, point_scale: float = 20.0,
                 width: int = 1280, height: int = 720):
        self.num_particles = num_particles
        self.num_active = num_particles
        self.point_scale = point_scale
        self._width = width
        self._height = height

        # Foam state
        self.foam_enabled = False
        self.num_foam = 0           # number of active foam particles
        self._max_foam = 200_000    # max foam pool size (matches world._max_foam)

        # SSFR state
        self.ssfr_enabled = False
        self.ssfr_blur_radius = 15.0
        self.ssfr_depth_range = 0.15
        self.ssfr_absorption = np.array([3.0, 1.5, 0.2], dtype=np.float32)  # absorb red most, blue least = blue tint
        self.ssfr_absorption_scale = 5.0
        self.ssfr_fresnel_power = 5.0
        self.ssfr_fresnel_bias = 0.02
        self.ssfr_specular_power = 64.0
        self.ssfr_particle_radius = 0.025  # world-space particle radius for depth correction
        self.ssfr_point_scale_mult = 1.8   # point size multiplier for SSFR depth/thickness (more overlap)

        # --- Compile shaders ---

        # Point sprite: all particles (SSFR off)
        self._prog_points = _build_program("particle.vert", "particle.frag")
        self._u_points_mvp = glGetUniformLocation(self._prog_points, "uMVP")
        self._u_points_mv = glGetUniformLocation(self._prog_points, "uMV")
        self._u_points_ps = glGetUniformLocation(self._prog_points, "uPointScale")

        # Non-FLUID point sprite (SSFR on): culls FLUID particles
        self._prog_nonfluid = _build_program("particle_nonfluid.vert", "particle.frag")
        self._u_nf_mvp = glGetUniformLocation(self._prog_nonfluid, "uMVP")
        self._u_nf_mv = glGetUniformLocation(self._prog_nonfluid, "uMV")
        self._u_nf_ps = glGetUniformLocation(self._prog_nonfluid, "uPointScale")

        # SSFR depth pass
        self._prog_depth = _build_program("ssfr_depth.vert", "ssfr_depth.frag")
        self._u_depth_mvp = glGetUniformLocation(self._prog_depth, "uMVP")
        self._u_depth_mv = glGetUniformLocation(self._prog_depth, "uMV")
        self._u_depth_ps = glGetUniformLocation(self._prog_depth, "uPointScale")
        self._u_depth_proj = glGetUniformLocation(self._prog_depth, "uProj")
        self._u_depth_radius = glGetUniformLocation(self._prog_depth, "uParticleRadius")

        # SSFR thickness pass
        self._prog_thick = _build_program("ssfr_thickness.vert", "ssfr_thickness.frag")
        self._u_thick_mvp = glGetUniformLocation(self._prog_thick, "uMVP")
        self._u_thick_mv = glGetUniformLocation(self._prog_thick, "uMV")
        self._u_thick_ps = glGetUniformLocation(self._prog_thick, "uPointScale")

        # Fullscreen quad vertex shader (shared by blur, normal, composite)
        fs_vert = _read_shader_source("fullscreen_quad.vert")

        # SSFR blur pass
        vs_blur = _compile_shader(fs_vert, GL_VERTEX_SHADER)
        fs_blur = _compile_shader(_read_shader_source("ssfr_blur.frag"), GL_FRAGMENT_SHADER)
        self._prog_blur = _link_program(vs_blur, fs_blur)
        glDeleteShader(vs_blur)
        glDeleteShader(fs_blur)
        self._u_blur_depth = glGetUniformLocation(self._prog_blur, "uDepthTex")
        self._u_blur_texel = glGetUniformLocation(self._prog_blur, "uTexelSize")
        self._u_blur_dir = glGetUniformLocation(self._prog_blur, "uBlurDir")
        self._u_blur_radius = glGetUniformLocation(self._prog_blur, "uFilterRadius")
        self._u_blur_range = glGetUniformLocation(self._prog_blur, "uDepthRange")

        # SSFR normal pass
        vs_norm = _compile_shader(fs_vert, GL_VERTEX_SHADER)
        fs_norm = _compile_shader(_read_shader_source("ssfr_normal.frag"), GL_FRAGMENT_SHADER)
        self._prog_normal = _link_program(vs_norm, fs_norm)
        glDeleteShader(vs_norm)
        glDeleteShader(fs_norm)
        self._u_norm_depth = glGetUniformLocation(self._prog_normal, "uDepthTex")
        self._u_norm_texel = glGetUniformLocation(self._prog_normal, "uTexelSize")
        self._u_norm_proj_inv = glGetUniformLocation(self._prog_normal, "uProjInv")

        # SSFR composite pass
        vs_comp = _compile_shader(fs_vert, GL_VERTEX_SHADER)
        fs_comp = _compile_shader(_read_shader_source("ssfr_composite.frag"), GL_FRAGMENT_SHADER)
        self._prog_composite = _link_program(vs_comp, fs_comp)
        glDeleteShader(vs_comp)
        glDeleteShader(fs_comp)
        self._u_comp_depth = glGetUniformLocation(self._prog_composite, "uDepthTex")
        self._u_comp_normal = glGetUniformLocation(self._prog_composite, "uNormalTex")
        self._u_comp_thickness = glGetUniformLocation(self._prog_composite, "uThicknessTex")
        self._u_comp_scene = glGetUniformLocation(self._prog_composite, "uSceneTex")
        self._u_comp_proj_inv = glGetUniformLocation(self._prog_composite, "uProjInv")
        self._u_comp_texel = glGetUniformLocation(self._prog_composite, "uTexelSize")
        self._u_comp_absorption = glGetUniformLocation(self._prog_composite, "uAbsorption")
        self._u_comp_abs_scale = glGetUniformLocation(self._prog_composite, "uAbsorptionScale")
        self._u_comp_fluid_color = glGetUniformLocation(self._prog_composite, "uFluidColor")
        self._u_comp_fresnel_pow = glGetUniformLocation(self._prog_composite, "uFresnelPower")
        self._u_comp_fresnel_bias = glGetUniformLocation(self._prog_composite, "uFresnelBias")
        self._u_comp_spec_pow = glGetUniformLocation(self._prog_composite, "uSpecularPower")

        # Foam shader
        self._prog_foam = _build_program("foam.vert", "foam.frag")
        self._u_foam_mvp = glGetUniformLocation(self._prog_foam, "uMVP")
        self._u_foam_mv = glGetUniformLocation(self._prog_foam, "uMV")
        self._u_foam_ps = glGetUniformLocation(self._prog_foam, "uPointScale")

        # --- 3D Cursor (wireframe circle for brush placement) ---
        _CURSOR_VERT = """
#version 410 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform vec3 uCenter;
uniform float uRadius;
void main() {
    vec3 world = aPos * uRadius + uCenter;
    gl_Position = uMVP * vec4(world, 1.0);
}
"""
        _CURSOR_FRAG = """
#version 410 core
out vec4 fragColor;
uniform vec4 uColor;
void main() {
    fragColor = uColor;
}
"""
        vs_cursor = _compile_shader(_CURSOR_VERT, GL_VERTEX_SHADER)
        fs_cursor = _compile_shader(_CURSOR_FRAG, GL_FRAGMENT_SHADER)
        self._prog_cursor = _link_program(vs_cursor, fs_cursor)
        glDeleteShader(vs_cursor)
        glDeleteShader(fs_cursor)
        self._u_cursor_mvp = glGetUniformLocation(self._prog_cursor, "uMVP")
        self._u_cursor_center = glGetUniformLocation(self._prog_cursor, "uCenter")
        self._u_cursor_radius = glGetUniformLocation(self._prog_cursor, "uRadius")
        self._u_cursor_color = glGetUniformLocation(self._prog_cursor, "uColor")

        # Pre-compute unit circle vertices (XZ plane, Y=0)
        _CURSOR_SEGMENTS = 64
        angles = np.linspace(0, 2.0 * np.pi, _CURSOR_SEGMENTS, endpoint=False, dtype=np.float32)
        circle_verts = np.zeros((_CURSOR_SEGMENTS, 3), dtype=np.float32)
        circle_verts[:, 0] = np.cos(angles)  # X
        circle_verts[:, 2] = np.sin(angles)  # Z
        self._cursor_num_verts = _CURSOR_SEGMENTS

        self._vbo_cursor = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_cursor)
        glBufferData(GL_ARRAY_BUFFER, circle_verts.nbytes, circle_verts, GL_STATIC_DRAW)

        self._vao_cursor = int(glGenVertexArrays(1))
        glBindVertexArray(self._vao_cursor)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_cursor)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Cursor state (set externally each frame)
        self.cursor_visible = False
        self.cursor_center = np.zeros(3, dtype=np.float32)
        self.cursor_radius = 0.1

        # --- Skybox ---
        self._prog_skybox = _build_program("skybox.vert", "skybox.frag")
        self._u_sky_viewrot = glGetUniformLocation(self._prog_skybox, "uViewRot")
        self._u_sky_proj = glGetUniformLocation(self._prog_skybox, "uProj")
        self._u_sky_tex = glGetUniformLocation(self._prog_skybox, "uSkybox")

        # Skybox cube geometry
        self._vbo_skybox = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_skybox)
        glBufferData(GL_ARRAY_BUFFER, _SKYBOX_VERTICES.nbytes, _SKYBOX_VERTICES, GL_STATIC_DRAW)

        self._vao_skybox = int(glGenVertexArrays(1))
        glBindVertexArray(self._vao_skybox)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_skybox)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Load cubemap texture from parent project resources
        skybox_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "build", "resources", "vue_sky")
        sky_prefix = "threatening"  # dramatic clouds
        face_map = {
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0: os.path.join(skybox_dir, f"{sky_prefix}_RT.png"),  # +X = right
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1: os.path.join(skybox_dir, f"{sky_prefix}_LF.png"),  # -X = left
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2: os.path.join(skybox_dir, f"{sky_prefix}_UP.png"),  # +Y = up
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3: os.path.join(skybox_dir, f"{sky_prefix}_DN.png"),  # -Y = down
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4: os.path.join(skybox_dir, f"{sky_prefix}_BK.png"),  # +Z = back
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5: os.path.join(skybox_dir, f"{sky_prefix}_FR.png"),  # -Z = front
        }
        self._tex_skybox = _load_cubemap(face_map)
        self.skybox_enabled = True

        # SSFR composite: additional uniforms for skybox
        self._u_comp_skybox = glGetUniformLocation(self._prog_composite, "uSkybox")
        self._u_comp_viewrot_inv = glGetUniformLocation(self._prog_composite, "uViewRotInv")

        # --- GL state ---
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # --- VBOs ---
        pos_nbytes = num_particles * _FLOAT4_BYTES
        col_nbytes = num_particles * _FLOAT4_BYTES

        self._vbo_pos = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, pos_nbytes, None, GL_DYNAMIC_DRAW)

        self._vbo_col = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_col)
        glBufferData(GL_ARRAY_BUFFER, col_nbytes, None, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # --- VAO for particle drawing ---
        self._vao = int(glGenVertexArrays(1))
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, _FLOAT4_BYTES, None)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_col)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, _FLOAT4_BYTES, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # --- Foam VBO + VAO (separate from main particles) ---
        foam_nbytes = self._max_foam * _FLOAT4_BYTES
        self._vbo_foam = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_foam)
        glBufferData(GL_ARRAY_BUFFER, foam_nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._vao_foam = int(glGenVertexArrays(1))
        glBindVertexArray(self._vao_foam)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_foam)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, _FLOAT4_BYTES, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # --- Empty VAO for fullscreen triangle ---
        self._vao_fs = int(glGenVertexArrays(1))

        # --- CUDA interop ---
        self.cuda_pos = CudaGLBuffer(self._vbo_pos, pos_nbytes)
        self.cuda_col = CudaGLBuffer(self._vbo_col, col_nbytes)
        self.cuda_foam = CudaGLBuffer(self._vbo_foam, foam_nbytes)

        # --- SSFR FBOs (created lazily or on resize) ---
        self._ssfr_fbos_valid = False
        self._fbo_depth = 0
        self._tex_depth = 0
        self._rbo_depth_hw = 0
        self._fbo_thickness = 0
        self._tex_thickness = 0
        self._fbo_blur1 = 0
        self._tex_blur1 = 0
        self._fbo_blur2 = 0
        self._tex_blur2 = 0
        self._fbo_normal = 0
        self._tex_normal = 0
        self._fbo_scene = 0
        self._tex_scene = 0
        self._rbo_scene_depth = 0

        self._create_ssfr_fbos(width, height)

    # -----------------------------------------------------------------
    # SSFR FBO management
    # -----------------------------------------------------------------

    def _create_ssfr_fbos(self, w: int, h: int) -> None:
        """Create/recreate all SSFR FBOs at the given resolution."""
        self._destroy_ssfr_fbos()
        self._width = w
        self._height = h

        # 1. Depth FBO: R32F color + depth24 renderbuffer
        self._tex_depth = _create_texture(w, h, GL_R32F, GL_RED, GL_FLOAT)
        self._rbo_depth_hw = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, self._rbo_depth_hw)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
        self._fbo_depth = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_depth)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex_depth, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._rbo_depth_hw)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        # 2. Thickness FBO: R16F
        self._tex_thickness = _create_texture(w, h, GL_R16F, GL_RED, GL_FLOAT)
        self._fbo_thickness = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_thickness)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex_thickness, 0)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        # 3-4. Blur ping-pong: R32F x2
        self._tex_blur1 = _create_texture(w, h, GL_R32F, GL_RED, GL_FLOAT)
        self._fbo_blur1 = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_blur1)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex_blur1, 0)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        self._tex_blur2 = _create_texture(w, h, GL_R32F, GL_RED, GL_FLOAT)
        self._fbo_blur2 = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_blur2)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex_blur2, 0)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        # 5. Normal FBO: RGBA16F
        self._tex_normal = _create_texture(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT)
        self._fbo_normal = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_normal)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex_normal, 0)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        # 6. Scene capture FBO: RGBA16F + depth (for background capture before SSFR)
        self._tex_scene = _create_texture(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT)
        self._rbo_scene_depth = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, self._rbo_scene_depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
        self._fbo_scene = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_scene)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex_scene, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._rbo_scene_depth)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        self._ssfr_fbos_valid = True

    def _destroy_ssfr_fbos(self) -> None:
        """Delete all SSFR FBO resources."""
        for fbo in (self._fbo_depth, self._fbo_thickness, self._fbo_blur1,
                     self._fbo_blur2, self._fbo_normal, self._fbo_scene):
            if fbo:
                glDeleteFramebuffers(1, [fbo])
        for tex in (self._tex_depth, self._tex_thickness, self._tex_blur1,
                     self._tex_blur2, self._tex_normal, self._tex_scene):
            if tex:
                glDeleteTextures(1, [tex])
        for rbo in (self._rbo_depth_hw, self._rbo_scene_depth):
            if rbo:
                glDeleteRenderbuffers(1, [rbo])
        self._ssfr_fbos_valid = False

    def resize(self, width: int, height: int) -> None:
        """Recreate FBOs when window size changes."""
        if width > 0 and height > 0 and (width != self._width or height != self._height):
            self._create_ssfr_fbos(width, height)

    # -----------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------

    def draw(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Render the scene. Uses SSFR if enabled, otherwise point sprites."""
        if self.ssfr_enabled and self._ssfr_fbos_valid:
            self._draw_ssfr(view, proj)
        else:
            self._draw_points(view, proj)
        # Draw foam on top with additive blending
        if self.foam_enabled and self.num_foam > 0:
            self._draw_foam(view, proj)
        # Draw 3D cursor (brush placement indicator)
        if self.cursor_visible:
            self._draw_cursor(view, proj)

    def _draw_points(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Simple point sprite rendering (all particles)."""
        # Draw skybox as background
        self._draw_skybox(view, proj)

        mvp = proj @ view
        glUseProgram(self._prog_points)
        glUniformMatrix4fv(self._u_points_mvp, 1, GL_TRUE, mvp)
        glUniformMatrix4fv(self._u_points_mv, 1, GL_TRUE, view)
        glUniform1f(self._u_points_ps, self.point_scale)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self.num_active)
        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_foam(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Render foam particles as additive-blended small point sprites."""
        mvp = proj @ view
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)  # additive
        glDepthMask(GL_FALSE)  # don't write to depth buffer

        glUseProgram(self._prog_foam)
        glUniformMatrix4fv(self._u_foam_mvp, 1, GL_TRUE, mvp)
        glUniformMatrix4fv(self._u_foam_mv, 1, GL_TRUE, view)
        glUniform1f(self._u_foam_ps, self.point_scale)
        glBindVertexArray(self._vao_foam)
        glDrawArrays(GL_POINTS, 0, self.num_foam)
        glBindVertexArray(0)
        glUseProgram(0)

        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)

    def _draw_cursor(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Draw the 3D brush cursor as a wireframe circle on the y=0 plane."""
        mvp = proj @ view

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        glUseProgram(self._prog_cursor)
        glUniformMatrix4fv(self._u_cursor_mvp, 1, GL_TRUE, mvp)
        glUniform3f(self._u_cursor_center,
                    float(self.cursor_center[0]),
                    float(self.cursor_center[1]),
                    float(self.cursor_center[2]))
        glUniform1f(self._u_cursor_radius, self.cursor_radius)
        glUniform4f(self._u_cursor_color, 1.0, 1.0, 1.0, 0.8)

        glBindVertexArray(self._vao_cursor)
        glDrawArrays(GL_LINE_LOOP, 0, self._cursor_num_verts)
        glBindVertexArray(0)
        glUseProgram(0)

        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)

    def _draw_skybox(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Draw skybox cube behind everything (depth = 1.0)."""
        if not self.skybox_enabled or self._tex_skybox == 0:
            return
        # View matrix with translation removed (rotation only)
        view_rot = view.copy()
        view_rot[0, 3] = 0.0
        view_rot[1, 3] = 0.0
        view_rot[2, 3] = 0.0

        glDepthFunc(GL_LEQUAL)  # skybox writes at depth=1.0
        glDepthMask(GL_FALSE)

        glUseProgram(self._prog_skybox)
        glUniformMatrix4fv(self._u_sky_viewrot, 1, GL_TRUE, view_rot)
        glUniformMatrix4fv(self._u_sky_proj, 1, GL_TRUE, proj)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._tex_skybox)
        glUniform1i(self._u_sky_tex, 0)

        glBindVertexArray(self._vao_skybox)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)

        glUseProgram(0)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)

    def _draw_ssfr(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Multi-pass SSFR rendering pipeline."""
        w, h = self._width, self._height
        mvp = proj @ view
        texel = np.array([1.0 / w, 1.0 / h], dtype=np.float32)

        try:
            proj_inv = np.linalg.inv(proj)
        except np.linalg.LinAlgError:
            proj_inv = np.eye(4, dtype=np.float32)

        # ---- Pass 0: Capture background (skybox + non-fluid particles) ----
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_scene)
        glViewport(0, 0, w, h)
        glClearColor(0.15, 0.15, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

        # Draw skybox into scene FBO (background for refraction)
        self._draw_skybox(view, proj)

        # Draw non-FLUID particles as point sprites
        glUseProgram(self._prog_nonfluid)
        glUniformMatrix4fv(self._u_nf_mvp, 1, GL_TRUE, mvp)
        glUniformMatrix4fv(self._u_nf_mv, 1, GL_TRUE, view)
        glUniform1f(self._u_nf_ps, self.point_scale)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self.num_active)
        glBindVertexArray(0)
        glUseProgram(0)

        # ---- Pass 1: SSFR Depth (FLUID only) ----
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_depth)
        glViewport(0, 0, w, h)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

        ssfr_ps = self.point_scale * self.ssfr_point_scale_mult
        glUseProgram(self._prog_depth)
        glUniformMatrix4fv(self._u_depth_mvp, 1, GL_TRUE, mvp)
        glUniformMatrix4fv(self._u_depth_mv, 1, GL_TRUE, view)
        glUniform1f(self._u_depth_ps, ssfr_ps)
        glUniformMatrix4fv(self._u_depth_proj, 1, GL_TRUE, proj)
        glUniform1f(self._u_depth_radius, self.ssfr_particle_radius)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self.num_active)
        glBindVertexArray(0)
        glUseProgram(0)

        # ---- Pass 2: Thickness (FLUID, additive blend) ----
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_thickness)
        glViewport(0, 0, w, h)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)  # additive

        glUseProgram(self._prog_thick)
        glUniformMatrix4fv(self._u_thick_mvp, 1, GL_TRUE, mvp)
        glUniformMatrix4fv(self._u_thick_mv, 1, GL_TRUE, view)
        glUniform1f(self._u_thick_ps, ssfr_ps)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self.num_active)
        glBindVertexArray(0)
        glUseProgram(0)

        glDisable(GL_BLEND)

        # ---- Pass 3: Bilateral blur (horizontal) ----
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_blur1)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        glUseProgram(self._prog_blur)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_depth)
        glUniform1i(self._u_blur_depth, 0)
        glUniform2f(self._u_blur_texel, texel[0], texel[1])
        glUniform2f(self._u_blur_dir, 1.0, 0.0)  # horizontal
        glUniform1f(self._u_blur_radius, self.ssfr_blur_radius)
        glUniform1f(self._u_blur_range, self.ssfr_depth_range)
        glBindVertexArray(self._vao_fs)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
        glUseProgram(0)

        # ---- Pass 4: Bilateral blur (vertical) ----
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_blur2)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._prog_blur)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_blur1)
        glUniform1i(self._u_blur_depth, 0)
        glUniform2f(self._u_blur_texel, texel[0], texel[1])
        glUniform2f(self._u_blur_dir, 0.0, 1.0)  # vertical
        glUniform1f(self._u_blur_radius, self.ssfr_blur_radius)
        glUniform1f(self._u_blur_range, self.ssfr_depth_range)
        glBindVertexArray(self._vao_fs)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
        glUseProgram(0)

        # ---- Pass 5: Normal reconstruction ----
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_normal)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._prog_normal)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_blur2)  # smoothed depth
        glUniform1i(self._u_norm_depth, 0)
        glUniform2f(self._u_norm_texel, texel[0], texel[1])
        glUniformMatrix4fv(self._u_norm_proj_inv, 1, GL_TRUE, proj_inv)
        glBindVertexArray(self._vao_fs)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
        glUseProgram(0)

        # ---- Pass 6: Composite (to default framebuffer) ----
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

        # Draw skybox as background (non-fluid pixels pass through in composite)
        self._draw_skybox(view, proj)

        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        glUseProgram(self._prog_composite)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_blur2)  # smoothed depth
        glUniform1i(self._u_comp_depth, 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._tex_normal)
        glUniform1i(self._u_comp_normal, 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self._tex_thickness)
        glUniform1i(self._u_comp_thickness, 2)

        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, self._tex_scene)  # background with non-fluid particles + skybox
        glUniform1i(self._u_comp_scene, 3)

        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._tex_skybox)
        glUniform1i(self._u_comp_skybox, 4)

        glUniformMatrix4fv(self._u_comp_proj_inv, 1, GL_TRUE, proj_inv)
        glUniform2f(self._u_comp_texel, texel[0], texel[1])
        glUniform3f(self._u_comp_absorption, *self.ssfr_absorption)
        glUniform1f(self._u_comp_abs_scale, self.ssfr_absorption_scale)
        glUniform3f(self._u_comp_fluid_color, 0.8, 0.9, 1.0)  # white-blue tint
        glUniform1f(self._u_comp_fresnel_pow, self.ssfr_fresnel_power)
        glUniform1f(self._u_comp_fresnel_bias, self.ssfr_fresnel_bias)
        glUniform1f(self._u_comp_spec_pow, self.ssfr_specular_power)

        # Inverse view rotation for eye->world reflection transform
        view_rot_3x3 = view[:3, :3].astype(np.float32)
        view_rot_inv = np.linalg.inv(view_rot_3x3).astype(np.float32)
        glUniformMatrix3fv(self._u_comp_viewrot_inv, 1, GL_TRUE, view_rot_inv)

        glBindVertexArray(self._vao_fs)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
        glUseProgram(0)

        # Restore GL state for ImGui
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Release GPU resources."""
        if self._prog_points == 0:
            return
        self.cuda_col.close()
        self.cuda_pos.close()
        self.cuda_foam.close()
        glDeleteBuffers(5, [self._vbo_pos, self._vbo_col, self._vbo_foam, self._vbo_cursor, self._vbo_skybox])
        glDeleteVertexArrays(1, [self._vao])
        glDeleteVertexArrays(1, [self._vao_foam])
        glDeleteVertexArrays(1, [self._vao_fs])
        glDeleteVertexArrays(1, [self._vao_cursor])
        glDeleteVertexArrays(1, [self._vao_skybox])
        if self._tex_skybox:
            glDeleteTextures(1, [self._tex_skybox])
        for prog in (self._prog_points, self._prog_nonfluid, self._prog_depth,
                      self._prog_thick, self._prog_blur, self._prog_normal,
                      self._prog_composite, self._prog_foam, self._prog_cursor,
                      self._prog_skybox):
            if prog:
                glDeleteProgram(prog)
        self._destroy_ssfr_fbos()
        self._prog_points = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
