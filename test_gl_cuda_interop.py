"""Integration test for gl_cuda_interop.py.

Creates an OpenGL VBO of 1 M float4s, registers it with CUDA via the
ctypes wrapper, writes a grid pattern using a CuPy RawKernel, then
draws the points with glDrawArrays(GL_POINTS) and verifies no errors.

Requirements: glfw, PyOpenGL, cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import ctypes
import sys
import numpy as np

# ---------------------------------------------------------------------------
# GLFW / OpenGL bootstrap
# ---------------------------------------------------------------------------

import glfw  # type: ignore[import-untyped]
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DYNAMIC_DRAW,
    GL_FLOAT,
    GL_NO_ERROR,
    GL_POINTS,
    glBindBuffer,
    glBufferData,
    glClear,
    glClearColor,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGetError,
    glPointSize,
    glVertexAttribPointer,
    glViewport,
)

import cupy  # type: ignore[import-untyped]
from gl_cuda_interop import CudaGLBuffer, check_last_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PARTICLES = 1_000_000  # 1 M particles
FLOAT4_BYTES = 4 * 4  # 4 floats * 4 bytes each
BUFFER_SIZE = NUM_PARTICLES * FLOAT4_BYTES

GRID_DIM = 1000  # 1000 x 1000 = 1 M points

# ---------------------------------------------------------------------------
# CuPy RawKernel -- writes a flat grid pattern into float4 positions
# ---------------------------------------------------------------------------

_grid_kernel = cupy.RawKernel(
    r"""
extern "C" __global__
void write_grid(float4* positions, int grid_dim, float spacing) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid_dim * grid_dim;
    if (idx >= total) return;

    int row = idx / grid_dim;
    int col = idx % grid_dim;

    // Center the grid around origin, z = 0, w = 1
    float x = (col - grid_dim / 2) * spacing;
    float y = (row - grid_dim / 2) * spacing;

    positions[idx] = make_float4(x, y, 0.0f, 1.0f);
}
""",
    "write_grid",
)


def _check_gl(context: str = "") -> None:
    """Raise if glGetError() reports an error."""
    err = glGetError()
    if err != GL_NO_ERROR:
        raise RuntimeError(f"OpenGL error 0x{err:04X} after {context}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def main() -> None:
    # -- 1. GLFW / GL context ------------------------------------------------
    if not glfw.init():
        sys.exit("GLFW init failed")

    # Request an OpenGL 3.3+ core profile so CUDA-GL interop is happy
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "CUDA-GL Interop Test", None, None)
    if not window:
        glfw.terminate()
        sys.exit("Failed to create GLFW window")

    glfw.make_context_current(window)

    # -- 2. Create VBO -------------------------------------------------------
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    _check_gl("VBO creation")

    print(f"[OK] Created GL VBO {vbo}  ({NUM_PARTICLES} float4s, {BUFFER_SIZE} bytes)")

    # -- 3. Register with CUDA & map -----------------------------------------
    buf = CudaGLBuffer(vbo, BUFFER_SIZE)
    print(f"[OK] Registered VBO with CUDA  (resource handle 0x{buf._resource:016X})")

    with buf:
        dev_ptr, mapped_size = buf.mapped_pointer()
        print(f"[OK] Mapped buffer -> device ptr 0x{dev_ptr:016X}, size {mapped_size}")
        assert mapped_size == BUFFER_SIZE, (
            f"Mapped size mismatch: expected {BUFFER_SIZE}, got {mapped_size}"
        )

        # -- 4. Write grid pattern with CuPy --------------------------------
        arr = buf.device_pointer_as_cupy_array(
            shape=(NUM_PARTICLES, 4), dtype=np.float32
        )

        # float4* view for the kernel
        threads = 256
        blocks = (NUM_PARTICLES + threads - 1) // threads
        spacing = np.float32(0.002)

        _grid_kernel(
            (blocks,),
            (threads,),
            (arr, np.int32(GRID_DIM), spacing),
        )
        cupy.cuda.Device().synchronize()
        check_last_error("CuPy RawKernel write_grid")

        # Quick sanity check: read back first few values
        first = cupy.asnumpy(arr[:3])
        print(f"[OK] First 3 positions:\n{first}")

    # buf is now unmapped
    print("[OK] Unmapped buffer -- GL can draw")

    # -- 5. Render one frame with GL_POINTS ---------------------------------
    from OpenGL.GL import (  # type: ignore[import-untyped]
        glGenVertexArrays,
        glBindVertexArray,
    )

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # float4 -> location 0, 4 components
    glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    _check_gl("VAO setup")

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glViewport(0, 0, 800, 600)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glPointSize(1.0)
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES)
    _check_gl("glDrawArrays(GL_POINTS)")

    glfw.swap_buffers(window)
    print(f"[OK] Drew {NUM_PARTICLES:,} points via glDrawArrays(GL_POINTS)")

    # -- 6. Final error checks -----------------------------------------------
    check_last_error("final CUDA check")
    _check_gl("final GL check")

    # -- 7. Cleanup ----------------------------------------------------------
    buf.close()
    print("[OK] Unregistered CUDA resource")

    glfw.destroy_window(window)
    glfw.terminate()

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
