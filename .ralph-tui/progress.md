## Codebase Patterns

- **CUDA-GL interop pattern (Python ctypes):** Load `cudart64_12.dll` (Windows) or `libcudart.so.12` (Linux) via `ctypes.CDLL`. Declare `restype`/`argtypes` for each CUDA runtime function. Wrap `cudaGraphicsResource*` as opaque `c_void_p` handles. Always check `cudaGetLastError` after each interop call.
- **CuPy zero-copy from device pointer:** `cupy.cuda.UnownedMemory(dev_ptr, nbytes, owner=None)` -> `MemoryPointer` -> `cupy.ndarray(shape, dtype, memptr)`. This avoids any host<->device copy when writing to GL-mapped buffers.
- **Existing C++ interop reference:** `SPHSimLib/SimCudaHelper.cpp` uses a `std::map<GLuint, cudaGraphicsResource*>` to track registered buffers -- the Python module mirrors this pattern with the `CudaGLBuffer` class.

---

## 2026-02-06 - US-003
- What was implemented:
  - `gl_cuda_interop.py` -- ctypes bindings for CUDA-GL interop (register, map, get_mapped_pointer, unmap, CudaGLBuffer class)
  - `test_gl_cuda_interop.py` -- integration test that creates a GL VBO of 1M float4s, registers with CUDA, writes a grid pattern via CuPy RawKernel, renders with glDrawArrays(GL_POINTS), checks CUDA+GL errors
- Files changed:
  - `gl_cuda_interop.py` (new)
  - `test_gl_cuda_interop.py` (new)
  - `.ralph-tui/progress.md` (new)
- **Learnings:**
  - `cudaGraphicsResource*` is an opaque pointer -- use `c_void_p` and pass by reference (`byref(c_void_p(handle))`) for map/unmap calls that take `cudaGraphicsResource**`
  - On modern CUDA (5.0+), `cudaGLSetGLDevice` is deprecated; plain `cudaSetDevice` works for GL interop as long as the GL context is current when registering buffers (confirmed in existing C++ code at `SimCudaHelper.cpp:86`)
  - `cudaGraphicsMapFlagsWriteDiscard` (value 2) is the right flag for write-only access from CUDA to a GL buffer -- matches the C++ code at `SimCudaHelper.cpp:178`
  - CuPy's `RawKernel` with `float4*` parameter works via the ndarray's device pointer; the kernel sees contiguous float4 memory
  - The project had no Python files previously -- this is the first Python module in the codebase
---
