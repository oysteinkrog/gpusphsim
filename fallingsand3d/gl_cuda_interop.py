"""CUDA-GL interop via ctypes bindings for cudart64_12.dll (Windows/NVIDIA).

Provides Python wrappers around CUDA runtime GL interop functions so that
an OpenGL VBO can be shared with CUDA (and written to by CuPy kernels)
without any CPU round-trip.

Exported API
------------
- register_buffer(vbo_id, flags) -> cudaGraphicsResource*
- map_buffer(resource)
- get_mapped_pointer(resource) -> (dev_ptr, size)
- unmap_buffer(resource)
- CudaGLBuffer  -- convenience class wrapping the lifecycle
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
from ctypes import (
    POINTER,
    byref,
    c_int,
    c_size_t,
    c_uint,
    c_void_p,
)
from typing import Optional

# ---------------------------------------------------------------------------
# Load the CUDA runtime library
# ---------------------------------------------------------------------------

# Opaque handle -- cudaGraphicsResource*
_cudaGraphicsResource_p = c_void_p

# cudaError_t is an int enum in the CUDA headers
_cudaError_t = c_int

# cudaGraphicsRegisterFlags / cudaGraphicsMapFlags
cudaGraphicsRegisterFlagsNone = 0
cudaGraphicsRegisterFlagsWriteDiscard = 2
cudaGraphicsMapFlagsNone = 0
cudaGraphicsMapFlagsReadOnly = 1
cudaGraphicsMapFlagsWriteDiscard = 2

cudaSuccess = 0


def _load_cudart() -> ctypes.CDLL:
    """Locate and load the CUDA runtime shared library."""
    if platform.system() == "Windows":
        # Try CUDA 12.x first, then 11.x
        for ver in ("12", "11"):
            name = f"cudart64_{ver}"
            try:
                return ctypes.CDLL(name)
            except OSError:
                pass
        # Fallback: let ctypes.util find it
        path = ctypes.util.find_library("cudart64_12") or ctypes.util.find_library(
            "cudart"
        )
        if path:
            return ctypes.CDLL(path)
        raise OSError(
            "Cannot find cudart64_12.dll (or cudart64_11.dll). "
            "Make sure the NVIDIA CUDA Toolkit bin directory is on PATH."
        )
    else:
        # Linux / other
        for name in ("libcudart.so.12", "libcudart.so.11", "libcudart.so"):
            try:
                return ctypes.CDLL(name)
            except OSError:
                pass
        path = ctypes.util.find_library("cudart")
        if path:
            return ctypes.CDLL(path)
        raise OSError("Cannot find libcudart.so. Is the CUDA Toolkit installed?")


_cudart = _load_cudart()

# ---------------------------------------------------------------------------
# Declare function signatures
# ---------------------------------------------------------------------------

# cudaError_t cudaGraphicsGLRegisterBuffer(
#     cudaGraphicsResource **resource, GLuint buffer, unsigned int flags);
_cudart.cudaGraphicsGLRegisterBuffer.restype = _cudaError_t
_cudart.cudaGraphicsGLRegisterBuffer.argtypes = [
    POINTER(c_void_p),  # cudaGraphicsResource **resource
    c_uint,  # GLuint buffer
    c_uint,  # unsigned int flags
]

# cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource *resource);
_cudart.cudaGraphicsUnregisterResource.restype = _cudaError_t
_cudart.cudaGraphicsUnregisterResource.argtypes = [c_void_p]

# cudaError_t cudaGraphicsMapResources(
#     int count, cudaGraphicsResource **resources, cudaStream_t stream);
_cudart.cudaGraphicsMapResources.restype = _cudaError_t
_cudart.cudaGraphicsMapResources.argtypes = [
    c_int,  # int count
    POINTER(c_void_p),  # cudaGraphicsResource **resources
    c_void_p,  # cudaStream_t stream (0 = default)
]

# cudaError_t cudaGraphicsUnmapResources(
#     int count, cudaGraphicsResource **resources, cudaStream_t stream);
_cudart.cudaGraphicsUnmapResources.restype = _cudaError_t
_cudart.cudaGraphicsUnmapResources.argtypes = [
    c_int,
    POINTER(c_void_p),
    c_void_p,
]

# cudaError_t cudaGraphicsResourceGetMappedPointer(
#     void **devPtr, size_t *size, cudaGraphicsResource *mappedResource);
_cudart.cudaGraphicsResourceGetMappedPointer.restype = _cudaError_t
_cudart.cudaGraphicsResourceGetMappedPointer.argtypes = [
    POINTER(c_void_p),  # void **devPtr
    POINTER(c_size_t),  # size_t *size
    c_void_p,  # cudaGraphicsResource *mappedResource
]

# cudaError_t cudaGetLastError(void);
_cudart.cudaGetLastError.restype = _cudaError_t
_cudart.cudaGetLastError.argtypes = []

# const char* cudaGetErrorString(cudaError_t error);
_cudart.cudaGetErrorString.restype = ctypes.c_char_p
_cudart.cudaGetErrorString.argtypes = [_cudaError_t]

# cudaError_t cudaSetDevice(int device);
_cudart.cudaSetDevice.restype = _cudaError_t
_cudart.cudaSetDevice.argtypes = [c_int]

# ---------------------------------------------------------------------------
# Error checking
# ---------------------------------------------------------------------------


class CudaError(RuntimeError):
    """Raised when a CUDA runtime call returns a non-success error code."""

    def __init__(self, err_code: int, context: str = ""):
        err_name = _cudart.cudaGetErrorString(err_code)
        if isinstance(err_name, bytes):
            err_name = err_name.decode("utf-8", errors="replace")
        msg = f"CUDA error {err_code}: {err_name}"
        if context:
            msg = f"{context} -- {msg}"
        super().__init__(msg)
        self.err_code = err_code


def _check(err: int, context: str = "") -> None:
    """Raise CudaError if *err* != cudaSuccess."""
    if err != cudaSuccess:
        raise CudaError(err, context)


def check_last_error(context: str = "") -> None:
    """Call cudaGetLastError and raise on failure."""
    _check(_cudart.cudaGetLastError(), context)


# ---------------------------------------------------------------------------
# Public thin wrappers
# ---------------------------------------------------------------------------


def register_buffer(
    vbo_id: int,
    flags: int = cudaGraphicsRegisterFlagsWriteDiscard,
) -> int:
    """Register an OpenGL buffer object for access by CUDA.

    Parameters
    ----------
    vbo_id : int
        The OpenGL VBO name (GLuint).
    flags : int
        Registration flags.  Default is ``cudaGraphicsRegisterFlagsWriteDiscard``.

    Returns
    -------
    int
        Opaque handle (cudaGraphicsResource pointer as integer).
    """
    resource = c_void_p(0)
    err = _cudart.cudaGraphicsGLRegisterBuffer(
        byref(resource), c_uint(vbo_id), c_uint(flags)
    )
    _check(err, "cudaGraphicsGLRegisterBuffer")
    check_last_error("cudaGraphicsGLRegisterBuffer (post)")
    return resource.value


def unregister_buffer(resource: int) -> None:
    """Unregister a previously registered GL buffer."""
    err = _cudart.cudaGraphicsUnregisterResource(c_void_p(resource))
    _check(err, "cudaGraphicsUnregisterResource")
    check_last_error("cudaGraphicsUnregisterResource (post)")


def map_buffer(resource: int) -> None:
    """Map a registered resource for access by CUDA kernels.

    Must be followed by :func:`unmap_buffer` before OpenGL can use the
    buffer again.
    """
    res_ptr = c_void_p(resource)
    err = _cudart.cudaGraphicsMapResources(1, byref(res_ptr), c_void_p(0))
    _check(err, "cudaGraphicsMapResources")
    check_last_error("cudaGraphicsMapResources (post)")


def get_mapped_pointer(resource: int) -> tuple[int, int]:
    """Retrieve the device pointer and byte size of a mapped resource.

    Returns
    -------
    (dev_ptr, nbytes) : tuple[int, int]
    """
    dev_ptr = c_void_p(0)
    size = c_size_t(0)
    err = _cudart.cudaGraphicsResourceGetMappedPointer(
        byref(dev_ptr), byref(size), c_void_p(resource)
    )
    _check(err, "cudaGraphicsResourceGetMappedPointer")
    check_last_error("cudaGraphicsResourceGetMappedPointer (post)")
    return dev_ptr.value, size.value


def unmap_buffer(resource: int) -> None:
    """Unmap a previously mapped resource so OpenGL can access it again."""
    res_ptr = c_void_p(resource)
    err = _cudart.cudaGraphicsUnmapResources(1, byref(res_ptr), c_void_p(0))
    _check(err, "cudaGraphicsUnmapResources")
    check_last_error("cudaGraphicsUnmapResources (post)")


# ---------------------------------------------------------------------------
# CudaGLBuffer -- high-level RAII-style wrapper
# ---------------------------------------------------------------------------


class CudaGLBuffer:
    """Manage the lifecycle of a CUDA-registered OpenGL buffer.

    Usage::

        buf = CudaGLBuffer(vbo_id, nbytes)
        with buf:
            ptr, size = buf.mapped_pointer()
            # ... write to ptr via CuPy ...
        # buffer is unmapped; GL can draw it

    When the object is deleted (or :meth:`close` is called), the buffer
    is unregistered from CUDA.
    """

    def __init__(
        self,
        vbo_id: int,
        nbytes: int,
        flags: int = cudaGraphicsRegisterFlagsWriteDiscard,
    ) -> None:
        self.vbo_id = vbo_id
        self.nbytes = nbytes
        self._resource: Optional[int] = register_buffer(vbo_id, flags)
        self._mapped = False

    # -- context manager for map / unmap ----------------------------------

    def __enter__(self) -> "CudaGLBuffer":
        self.map()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unmap()

    # -- core operations ---------------------------------------------------

    def map(self) -> None:
        if self._mapped:
            return
        if self._resource is None:
            raise RuntimeError("CudaGLBuffer: resource already closed")
        map_buffer(self._resource)
        self._mapped = True

    def unmap(self) -> None:
        if not self._mapped:
            return
        if self._resource is None:
            raise RuntimeError("CudaGLBuffer: resource already closed")
        unmap_buffer(self._resource)
        self._mapped = False

    def mapped_pointer(self) -> tuple[int, int]:
        """Return ``(device_ptr, byte_size)`` of the mapped buffer."""
        if not self._mapped:
            raise RuntimeError("CudaGLBuffer: buffer is not mapped")
        return get_mapped_pointer(self._resource)

    def device_pointer_as_cupy_array(
        self,
        shape: tuple[int, ...],
        dtype: "object" = None,
    ) -> "object":
        """Wrap the mapped device pointer as a CuPy ndarray (zero-copy).

        Parameters
        ----------
        shape : tuple[int, ...]
            Desired array shape.
        dtype : numpy/cupy dtype
            Element type.  Defaults to ``float32``.

        Returns
        -------
        cupy.ndarray
        """
        import cupy
        import numpy as np

        if dtype is None:
            dtype = np.float32

        dev_ptr, size = self.mapped_pointer()
        mem = cupy.cuda.UnownedMemory(dev_ptr, size, owner=None)
        memptr = cupy.cuda.MemoryPointer(mem, 0)
        return cupy.ndarray(shape, dtype=dtype, memptr=memptr)

    # -- cleanup -----------------------------------------------------------

    def close(self) -> None:
        """Unmap (if needed) and unregister the buffer."""
        if self._mapped:
            self.unmap()
        if self._resource is not None:
            unregister_buffer(self._resource)
            self._resource = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
