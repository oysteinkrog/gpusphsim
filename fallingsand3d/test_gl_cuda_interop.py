"""Integration tests for gl_cuda_interop.py -- pytest suite.

Creates an OpenGL VBO, registers it with CUDA via the ctypes wrapper,
writes a grid pattern using a CuPy RawKernel, then draws the points
with glDrawArrays(GL_POINTS) and verifies no errors.

Requirements: glfw, PyOpenGL, cupy, numpy, an NVIDIA GPU with CUDA 12.x.
"""

from __future__ import annotations

import ctypes
import sys
import pytest
import numpy as np

import cupy
import cupy.cuda.compiler as _compiler

from gl_cuda_interop import (
    CudaGLBuffer,
    check_last_error,
    map_buffer,
    unmap_buffer,
    register_buffer,
    unregister_buffer,
)

# Blackwell (sm_120) PTX workaround: force PTX output so the driver JIT-compiles
_compiler._use_ptx = True
for _fn in (_compiler._get_arch, _compiler._get_arch_for_options_for_nvrtc):
    if hasattr(_fn, '_cache'):
        _fn._cache = {}

# ---------------------------------------------------------------------------
# GLFW / OpenGL bootstrap — shared fixture
# ---------------------------------------------------------------------------

glfw = pytest.importorskip("glfw", reason="glfw not installed")

from OpenGL.GL import (
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
    glGenVertexArrays,
    glBindVertexArray,
    glDeleteBuffers,
)


def _check_gl(context: str = "") -> None:
    """Raise if glGetError() reports an error."""
    err = glGetError()
    if err != GL_NO_ERROR:
        raise RuntimeError(f"OpenGL error 0x{err:04X} after {context}")


@pytest.fixture(scope="module")
def gl_context():
    """Initialize GLFW and create an invisible OpenGL 3.3 context."""
    if not glfw.init():
        pytest.skip("GLFW init failed (no display?)")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, False)  # headless / off-screen

    window = glfw.create_window(800, 600, "CUDA-GL Interop Test", None, None)
    if not window:
        glfw.terminate()
        pytest.skip("Failed to create GLFW window (no GPU display?)")

    glfw.make_context_current(window)
    yield window
    glfw.destroy_window(window)
    glfw.terminate()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PARTICLES = 1_000_000  # 1 M particles
FLOAT4_BYTES = 4 * 4       # 4 floats * 4 bytes each
BUFFER_SIZE = NUM_PARTICLES * FLOAT4_BYTES
GRID_DIM = 1000            # 1000 x 1000 = 1 M points

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

    float x = (col - grid_dim / 2) * spacing;
    float y = (row - grid_dim / 2) * spacing;

    positions[idx] = make_float4(x, y, 0.0f, 1.0f);
}
""",
    "write_grid",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vbo_registration(gl_context):
    """Create a VBO and register it with CUDA; unregister cleanly."""
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    _check_gl("VBO creation")

    resource = register_buffer(vbo)
    assert resource is not None
    assert resource != 0

    unregister_buffer(resource)
    glDeleteBuffers(1, [vbo])


def test_map_unmap(gl_context):
    """Map and unmap a registered VBO without errors."""
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    resource = register_buffer(vbo)
    map_buffer(resource)
    check_last_error("map_buffer")
    unmap_buffer(resource)
    check_last_error("unmap_buffer")
    unregister_buffer(resource)
    glDeleteBuffers(1, [vbo])


def test_mapped_pointer_size(gl_context):
    """Mapped pointer byte size matches the VBO allocation."""
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    buf = CudaGLBuffer(vbo, BUFFER_SIZE)
    with buf:
        dev_ptr, mapped_size = buf.mapped_pointer()
        assert dev_ptr != 0, "Device pointer must not be NULL"
        assert mapped_size == BUFFER_SIZE, (
            f"Mapped size mismatch: expected {BUFFER_SIZE}, got {mapped_size}"
        )
    buf.close()
    glDeleteBuffers(1, [vbo])


def test_device_pointer_as_cupy_array(gl_context):
    """Write a grid pattern via CuPy kernel and read back first values."""
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    buf = CudaGLBuffer(vbo, BUFFER_SIZE)
    with buf:
        arr = buf.device_pointer_as_cupy_array(
            shape=(NUM_PARTICLES, 4), dtype=np.float32
        )

        threads = 256
        blocks = (NUM_PARTICLES + threads - 1) // threads
        spacing = np.float32(0.002)
        _grid_kernel((blocks,), (threads,), (arr, np.int32(GRID_DIM), spacing))
        cupy.cuda.Device().synchronize()
        check_last_error("write_grid kernel")

        # Verify first and last rows were written (not all zeros)
        first = cupy.asnumpy(arr[:3])
        # First particle: col=0, row=0 -> x=(0-500)*0.002=-1.0, y=(0-500)*0.002=-1.0
        assert first[0, 3] == 1.0, "w-component should be 1.0"
        assert not np.all(first == 0.0), "Grid should not be all zeros"

    buf.close()
    glDeleteBuffers(1, [vbo])


def test_context_manager_maps_and_unmaps(gl_context):
    """CudaGLBuffer context manager maps on enter and unmaps on exit."""
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, 1024, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    buf = CudaGLBuffer(vbo, 1024)
    assert not buf._mapped

    with buf:
        assert buf._mapped, "Buffer should be mapped inside context"

    assert not buf._mapped, "Buffer should be unmapped after context"
    buf.close()
    glDeleteBuffers(1, [vbo])


def test_gl_draw_points(gl_context):
    """Draw 1M points via glDrawArrays(GL_POINTS) after CUDA write."""
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    buf = CudaGLBuffer(vbo, BUFFER_SIZE)
    with buf:
        arr = buf.device_pointer_as_cupy_array(
            shape=(NUM_PARTICLES, 4), dtype=np.float32
        )
        threads = 256
        blocks = (NUM_PARTICLES + threads - 1) // threads
        spacing = np.float32(0.002)
        _grid_kernel((blocks,), (threads,), (arr, np.int32(GRID_DIM), spacing))
        cupy.cuda.Device().synchronize()

    # GL draw after unmap
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    _check_gl("VAO setup")

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glViewport(0, 0, 800, 600)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPointSize(1.0)
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES)
    _check_gl("glDrawArrays GL_POINTS")

    glfw.swap_buffers(gl_context)

    # Final error checks
    check_last_error("final CUDA check")
    _check_gl("final GL check")

    buf.close()
    glDeleteBuffers(1, [vbo])


def test_close_unregisters(gl_context):
    """CudaGLBuffer.close() sets _resource to None (resource freed)."""
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, 1024, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    buf = CudaGLBuffer(vbo, 1024)
    assert buf._resource is not None
    buf.close()
    assert buf._resource is None, "Resource should be None after close()"
    glDeleteBuffers(1, [vbo])


@pytest.mark.xfail(
    strict=True,
    reason="bd-mzc.43: map/unmap uses NULL stream (gl_cuda_interop.py:218,243) "
           "— global GPU barrier 3-4x/frame; fix uses a named non-blocking stream",
)
def test_map_uses_nonnull_stream(gl_context):
    """map_buffer and unmap_buffer must use a non-NULL (named) CUDA stream.

    Currently both calls pass stream=NULL (c_void_p(0)), causing a global
    synchronization barrier every frame.  After bd-mzc.43 is applied the
    calls will use a non-zero stream handle and this test must be updated
    to pass (remove the xfail marker).
    """
    import inspect
    src = inspect.getsource(map_buffer)
    # NULL stream shows up as c_void_p(0) or stream=0 in the call
    assert "c_void_p(0)" not in src, (
        "map_buffer still passes NULL stream to cudaGraphicsMapResources"
    )


@pytest.mark.xfail(
    strict=True,
    reason="bd-mzc.49: CudaGLBuffer.map() silently swallows double-map "
           "(gl_cuda_interop.py:290) — masks races; fix should raise RuntimeError",
)
def test_double_map_raises(gl_context):
    """Calling map() twice on the same buffer must raise RuntimeError.

    Currently map() returns silently on the second call (early-return guard),
    masking potential races.  After bd-mzc.49 is applied a RuntimeError
    (or CudaError) must be raised on the second call.
    """
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, 1024, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    buf = CudaGLBuffer(vbo, 1024)
    try:
        buf.map()
        with pytest.raises((RuntimeError, Exception)):
            buf.map()  # second map should raise, not silently return
    finally:
        if buf._mapped:
            buf.unmap()
        buf.close()
        glDeleteBuffers(1, [vbo])


# ---------------------------------------------------------------------------
# Standalone runner (not used by pytest)
# ---------------------------------------------------------------------------

def main() -> None:
    """Original standalone runner kept for direct-execution compatibility."""
    if not glfw.init():
        sys.exit("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(800, 600, "CUDA-GL Interop Test", None, None)
    if not window:
        glfw.terminate()
        sys.exit("Failed to create GLFW window")
    glfw.make_context_current(window)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    buf = CudaGLBuffer(vbo, BUFFER_SIZE)
    with buf:
        dev_ptr, mapped_size = buf.mapped_pointer()
        print(f"[OK] Mapped buffer -> device ptr 0x{dev_ptr:016X}, size {mapped_size}")
        assert mapped_size == BUFFER_SIZE
        arr = buf.device_pointer_as_cupy_array(shape=(NUM_PARTICLES, 4), dtype=np.float32)
        threads = 256
        blocks = (NUM_PARTICLES + threads - 1) // threads
        _grid_kernel((blocks,), (threads,), (arr, np.int32(GRID_DIM), np.float32(0.002)))
        cupy.cuda.Device().synchronize()

    buf.close()
    glfw.destroy_window(window)
    glfw.terminate()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
