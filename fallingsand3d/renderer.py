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
import math as _math

_FLOAT4_BYTES = 4 * 4


# ---------------------------------------------------------------------------
# SDF Object mesh generation (unit primitives with normals)
# ---------------------------------------------------------------------------

def _generate_box_mesh() -> np.ndarray:
    """Unit box (+-0.5), 36 verts with normals. Returns (36, 6) float32."""
    faces = [
        ((0.5,-0.5,-0.5),(0.5,0.5,-0.5),(0.5,0.5,0.5),(0.5,-0.5,-0.5),(0.5,0.5,0.5),(0.5,-0.5,0.5),(1,0,0)),
        ((-0.5,-0.5,0.5),(-0.5,0.5,0.5),(-0.5,0.5,-0.5),(-0.5,-0.5,0.5),(-0.5,0.5,-0.5),(-0.5,-0.5,-0.5),(-1,0,0)),
        ((-0.5,0.5,-0.5),(0.5,0.5,-0.5),(0.5,0.5,0.5),(-0.5,0.5,-0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5),(0,1,0)),
        ((-0.5,-0.5,0.5),(-0.5,-0.5,-0.5),(0.5,-0.5,-0.5),(-0.5,-0.5,0.5),(0.5,-0.5,-0.5),(0.5,-0.5,0.5),(0,-1,0)),
        ((-0.5,-0.5,0.5),(0.5,-0.5,0.5),(0.5,0.5,0.5),(-0.5,-0.5,0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5),(0,0,1)),
        ((0.5,-0.5,-0.5),(-0.5,-0.5,-0.5),(-0.5,0.5,-0.5),(0.5,-0.5,-0.5),(-0.5,0.5,-0.5),(0.5,0.5,-0.5),(0,0,-1)),
    ]
    verts = []
    for face in faces:
        *tri_verts, normal = face
        for v in tri_verts:
            verts.append((*v, *normal))
    return np.array(verts, dtype=np.float32)


def _generate_sphere_mesh(stacks: int = 16, slices: int = 16) -> np.ndarray:
    """Unit sphere triangle mesh with normals. Returns (N, 6) float32."""
    verts = []
    for i in range(stacks):
        t0 = _math.pi * i / stacks
        t1 = _math.pi * (i + 1) / stacks
        for j in range(slices):
            p0 = 2 * _math.pi * j / slices
            p1 = 2 * _math.pi * (j + 1) / slices
            v00 = (_math.sin(t0)*_math.cos(p0), _math.cos(t0), _math.sin(t0)*_math.sin(p0))
            v10 = (_math.sin(t1)*_math.cos(p0), _math.cos(t1), _math.sin(t1)*_math.sin(p0))
            v01 = (_math.sin(t0)*_math.cos(p1), _math.cos(t0), _math.sin(t0)*_math.sin(p1))
            v11 = (_math.sin(t1)*_math.cos(p1), _math.cos(t1), _math.sin(t1)*_math.sin(p1))
            for tri in [(v00,v10,v11),(v00,v11,v01)]:
                for v in tri:
                    verts.append((*v, *v))
    return np.array(verts, dtype=np.float32)


def _generate_cylinder_mesh(segments: int = 16) -> np.ndarray:
    """Unit cylinder (r=0.5, h=1) with normals. Returns (N, 6) float32."""
    verts = []
    r, h = 0.5, 0.5
    for i in range(segments):
        a0 = 2*_math.pi*i/segments; a1 = 2*_math.pi*(i+1)/segments
        c0, s0, c1, s1 = _math.cos(a0), _math.sin(a0), _math.cos(a1), _math.sin(a1)
        # Side
        for v in [(r*c0,-h,r*s0,c0,0,s0),(r*c0,h,r*s0,c0,0,s0),(r*c1,h,r*s1,c1,0,s1),
                   (r*c0,-h,r*s0,c0,0,s0),(r*c1,h,r*s1,c1,0,s1),(r*c1,-h,r*s1,c1,0,s1)]:
            verts.append(v)
        # Top cap
        for v in [(0,h,0,0,1,0),(r*c0,h,r*s0,0,1,0),(r*c1,h,r*s1,0,1,0)]:
            verts.append(v)
        # Bottom cap
        for v in [(0,-h,0,0,-1,0),(r*c1,-h,r*s1,0,-1,0),(r*c0,-h,r*s0,0,-1,0)]:
            verts.append(v)
    return np.array(verts, dtype=np.float32)


def _generate_plane_mesh(size: float = 10.0) -> np.ndarray:
    """Large XZ quad (normal +Y). Returns (6, 6) float32."""
    s = size * 0.5
    return np.array([
        (-s,0,-s,0,1,0),(s,0,-s,0,1,0),(s,0,s,0,1,0),
        (-s,0,-s,0,1,0),(s,0,s,0,1,0),(-s,0,s,0,1,0),
    ], dtype=np.float32)


def _quat_to_mat4(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) to 4x4 rotation matrix (row-major)."""
    x2, y2, z2 = qx+qx, qy+qy, qz+qz
    xx, xy, xz = qx*x2, qx*y2, qx*z2
    yy, yz, zz = qy*y2, qy*z2, qz*z2
    wx, wy, wz = qw*x2, qw*y2, qw*z2
    m = np.eye(4, dtype=np.float32)
    m[0,0]=1-(yy+zz); m[0,1]=xy-wz;     m[0,2]=xz+wy
    m[1,0]=xy+wz;     m[1,1]=1-(xx+zz);  m[1,2]=yz-wx
    m[2,0]=xz-wy;     m[2,1]=yz+wx;      m[2,2]=1-(xx+yy)
    return m


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
        self.ssfr_blur_radius = 10.0
        self.ssfr_depth_range = 0.15
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
        self._u_comp_abs_scale = glGetUniformLocation(self._prog_composite, "uAbsorptionScale")
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

        # --- SDF Object rendering ---
        _SDF_VERT = """
#version 410 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform mat4 uModel;
void main() {
    gl_Position = uMVP * uModel * vec4(aPos, 1.0);
}
"""
        _SDF_FRAG = """
#version 410 core
out vec4 fragColor;
uniform vec4 uColor;
void main() {
    fragColor = uColor;
}
"""
        vs_sdf = _compile_shader(_SDF_VERT, GL_VERTEX_SHADER)
        fs_sdf = _compile_shader(_SDF_FRAG, GL_FRAGMENT_SHADER)
        self._prog_sdf = _link_program(vs_sdf, fs_sdf)
        glDeleteShader(vs_sdf)
        glDeleteShader(fs_sdf)
        self._u_sdf_mvp = glGetUniformLocation(self._prog_sdf, "uMVP")
        self._u_sdf_model = glGetUniformLocation(self._prog_sdf, "uModel")
        self._u_sdf_color = glGetUniformLocation(self._prog_sdf, "uColor")

        # Generate mesh data for each SDF primitive type
        self._sdf_meshes = {}  # type -> (vao, vbo, num_verts)
        self._sdf_meshes[0] = self._create_box_mesh()      # SDF_BOX
        self._sdf_meshes[1] = self._create_sphere_mesh()    # SDF_SPHERE
        self._sdf_meshes[2] = self._create_cylinder_mesh()  # SDF_CYLINDER
        self._sdf_meshes[3] = self._create_plane_mesh()     # SDF_PLANE

        # SDF object visibility
        self.sdf_objects_visible = True
        self.sdf_manager = None  # set externally by main loop
        self.selected_sdf_id = None  # set by UI for highlight

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

        # 2. Thickness FBO: RGBA16F (color-weighted thickness for per-pixel material color)
        self._tex_thickness = _create_texture(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT)
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
        # Draw SDF objects as semi-transparent meshes (after particles, before foam)
        self._draw_sdf_objects(view, proj)
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
        # All particles (including non-FLUID) go through SSFR depth/thickness passes
        self._draw_skybox(view, proj)

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
        glUniform1f(self._u_comp_abs_scale, self.ssfr_absorption_scale)
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
    # SDF object mesh generation and rendering
    # -----------------------------------------------------------------

    def _upload_mesh(self, verts: np.ndarray) -> tuple:
        """Upload vertex data to a VBO+VAO, return (vao, vbo, num_verts)."""
        vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        vao = int(glGenVertexArrays(1))
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return (vao, vbo, len(verts) // 3)

    def _create_box_mesh(self) -> tuple:
        """Unit cube [-1,1]^3 as 12 triangles (36 verts)."""
        # Same as skybox vertices but that's a unit cube too
        verts = np.array([
            -1,-1,-1,  1,-1,-1,  1,1,-1,   1,1,-1, -1,1,-1, -1,-1,-1,
            -1,-1,1,   1,1,1,    1,-1,1,   -1,-1,1, -1,1,1,   1,1,1,
            -1,-1,-1, -1,1,1,   -1,-1,1,   -1,-1,-1,-1,1,-1, -1,1,1,
             1,-1,-1,  1,-1,1,   1,1,1,     1,1,1,   1,1,-1,  1,-1,-1,
            -1,-1,-1,  1,-1,1,   1,-1,-1,  -1,-1,-1,-1,-1,1,  1,-1,1,
            -1,1,-1,   1,1,-1,   1,1,1,     1,1,1,  -1,1,1,  -1,1,-1,
        ], dtype=np.float32)
        return self._upload_mesh(verts)

    def _create_sphere_mesh(self, stacks: int = 16, slices: int = 16) -> tuple:
        """Unit sphere as triangle mesh."""
        verts = []
        for i in range(stacks):
            t0 = i / stacks
            t1 = (i + 1) / stacks
            phi0 = np.pi * t0
            phi1 = np.pi * t1
            for j in range(slices):
                s0 = j / slices
                s1 = (j + 1) / slices
                theta0 = 2.0 * np.pi * s0
                theta1 = 2.0 * np.pi * s1

                def sv(phi, theta):
                    return [np.sin(phi)*np.cos(theta), np.cos(phi), np.sin(phi)*np.sin(theta)]

                p00 = sv(phi0, theta0)
                p10 = sv(phi1, theta0)
                p01 = sv(phi0, theta1)
                p11 = sv(phi1, theta1)
                verts.extend(p00 + p10 + p11)
                verts.extend(p00 + p11 + p01)
        return self._upload_mesh(np.array(verts, dtype=np.float32))

    def _create_cylinder_mesh(self, segments: int = 16) -> tuple:
        """Unit cylinder (radius=1, height=2, centered at origin) as triangle mesh."""
        verts = []
        for i in range(segments):
            a0 = 2.0 * np.pi * i / segments
            a1 = 2.0 * np.pi * (i + 1) / segments
            c0, s0 = np.cos(a0), np.sin(a0)
            c1, s1 = np.cos(a1), np.sin(a1)
            # Side wall
            verts.extend([c0,1,s0, c0,-1,s0, c1,-1,s1])
            verts.extend([c0,1,s0, c1,-1,s1, c1,1,s1])
            # Top cap
            verts.extend([0,1,0, c0,1,s0, c1,1,s1])
            # Bottom cap
            verts.extend([0,-1,0, c1,-1,s1, c0,-1,s0])
        return self._upload_mesh(np.array(verts, dtype=np.float32))

    def _create_plane_mesh(self) -> tuple:
        """Large quad [-1,1]x[-1,1] in XZ plane (scaled to 10m at draw time)."""
        verts = np.array([
            -1,0,-1,  1,0,-1,  1,0,1,
            -1,0,-1,  1,0,1,  -1,0,1,
        ], dtype=np.float32)
        return self._upload_mesh(verts)

    def _draw_sdf_objects(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Render SDF collision objects as semi-transparent meshes."""
        if not self.sdf_objects_visible or self.sdf_manager is None:
            return
        objects = self.sdf_manager.get_sdf_objects()
        if not objects:
            return

        mvp = proj @ view
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        glUseProgram(self._prog_sdf)
        glUniformMatrix4fv(self._u_sdf_mvp, 1, GL_TRUE, mvp)

        for obj in objects:
            sdf_type = obj["type"]
            if sdf_type not in self._sdf_meshes:
                continue
            vao, _, num_verts = self._sdf_meshes[sdf_type]

            # Build model matrix: translate * rotate * scale
            pos = obj["position"]
            rot = obj["rotation"]  # quaternion (x,y,z,w)
            size = obj["size"]

            model = self._build_sdf_model_matrix(sdf_type, pos, rot, size)
            glUniformMatrix4fv(self._u_sdf_model, 1, GL_TRUE, model)

            # Color: selected = bright yellow, kinematic = blue, static = gray
            is_selected = (obj["id"] == self.selected_sdf_id)
            has_motion = obj["id"] in self.sdf_manager._motions
            if is_selected:
                glUniform4f(self._u_sdf_color, 1.0, 0.9, 0.3, 0.5)
            elif has_motion:
                glUniform4f(self._u_sdf_color, 0.3, 0.4, 0.8, 0.3)
            else:
                glUniform4f(self._u_sdf_color, 0.5, 0.5, 0.5, 0.3)

            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, num_verts)

        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)

    @staticmethod
    def _build_sdf_model_matrix(sdf_type: int, pos: list, rot: list, size: list) -> np.ndarray:
        """Build 4x4 model matrix from position, quaternion rotation, and size."""
        # Translation
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = pos[0]
        T[1, 3] = pos[1]
        T[2, 3] = pos[2]

        # Rotation from quaternion (x,y,z,w)
        qx, qy, qz, qw = rot
        R = np.eye(4, dtype=np.float32)
        R[0,0] = 1 - 2*(qy*qy + qz*qz)
        R[0,1] = 2*(qx*qy - qw*qz)
        R[0,2] = 2*(qx*qz + qw*qy)
        R[1,0] = 2*(qx*qy + qw*qz)
        R[1,1] = 1 - 2*(qx*qx + qz*qz)
        R[1,2] = 2*(qy*qz - qw*qx)
        R[2,0] = 2*(qx*qz - qw*qy)
        R[2,1] = 2*(qy*qz + qw*qx)
        R[2,2] = 1 - 2*(qx*qx + qy*qy)

        # Scale based on SDF type
        S = np.eye(4, dtype=np.float32)
        if sdf_type == 0:  # BOX: half_extents map to unit cube [-1,1]
            S[0,0] = size[0]
            S[1,1] = size[1]
            S[2,2] = size[2]
        elif sdf_type == 1:  # SPHERE: radius (uniform scale)
            S[0,0] = S[1,1] = S[2,2] = size[0]
        elif sdf_type == 2:  # CYLINDER: radius for XZ, half_height for Y
            S[0,0] = size[0]  # radius
            S[1,1] = size[2]  # half_height (stored in size.z)
            S[2,2] = size[0]  # radius
        elif sdf_type == 3:  # PLANE: large flat quad
            S[0,0] = S[2,2] = 5.0  # 10m across

        return T @ R @ S

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
        # Clean up SDF mesh resources
        for vao, vbo, _ in self._sdf_meshes.values():
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
        self._sdf_meshes.clear()

        for prog in (self._prog_points, self._prog_nonfluid, self._prog_depth,
                      self._prog_thick, self._prog_blur, self._prog_normal,
                      self._prog_composite, self._prog_foam, self._prog_cursor,
                      self._prog_skybox, self._prog_sdf):
            if prog:
                glDeleteProgram(prog)
        self._destroy_ssfr_fbos()
        self._prog_points = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
