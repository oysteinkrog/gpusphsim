## Codebase Patterns

- **CUDA-GL interop pattern (Python ctypes):** Load `cudart64_12.dll` (Windows) or `libcudart.so.12` (Linux) via `ctypes.CDLL`. Declare `restype`/`argtypes` for each CUDA runtime function. Wrap `cudaGraphicsResource*` as opaque `c_void_p` handles. Always check `cudaGetLastError` after each interop call.
- **CuPy zero-copy from device pointer:** `cupy.cuda.UnownedMemory(dev_ptr, nbytes, owner=None)` -> `MemoryPointer` -> `cupy.ndarray(shape, dtype, memptr)`. This avoids any host<->device copy when writing to GL-mapped buffers.
- **Existing C++ interop reference:** `SPHSimLib/SimCudaHelper.cpp` uses a `std::map<GLuint, cudaGraphicsResource*>` to track registered buffers -- the Python module mirrors this pattern with the `CudaGLBuffer` class.
- **CUDA constant memory upload pattern (CuPy):** Use `cupy.RawModule(code=KERNEL_SOURCE)` to compile CUDA source containing `__constant__` arrays. Get the device pointer via `module.get_global('c_symbol_name')`, then upload with `cupy.cuda.runtime.memcpy(int(d_ptr), host_array.ctypes.data, nbytes, 1)` where 1=cudaMemcpyHostToDevice. Do NOT use `memcpyToSymbol` -- it's unreliable with NVRTC modules.
- **Blackwell (sm_120) PTX workaround:** CuPy's NVRTC backend defaults to generating sm_NN cubin, but if the GPU (e.g. sm_120) exceeds the bundled NVRTC's max capability (e.g. sm_90 for CUDA 12.5), the cubin won't load. Fix: set `cupy.cuda.compiler._use_ptx = True` and clear the memoized `_get_arch`/`_get_arch_for_options_for_nvrtc` caches before first compilation. This forces `compute_NN` PTX output which the driver JIT-compiles.
- **Numpy structured arrays for C struct packing:** Define `np.dtype([("field", np.float32), ...], align=True)` matching the C struct layout exactly. Assert `dtype.itemsize == sizeof(CStruct)` to catch padding mismatches. Use `array.ctypes.data` for the host pointer when uploading to GPU.

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

## 2026-02-06 - US-006
- What was implemented:
  - `materials.py` -- Material property table (16 materials + 16 reserved) and 32x32 interaction matrix with GPU upload via CuPy RawModule constant memory
  - `test_materials.py` -- Integration test verifying host-side data, struct sizes, GPU upload, and constant memory readback via test kernel
- Files changed:
  - `materials.py` (new)
  - `test_materials.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - `module.get_global('c_symbol')` returns a `cupy.cuda.MemoryPointer`; cast with `int()` to get the raw device address for `cupy.cuda.runtime.memcpy`
  - On Blackwell (sm_120) with CuPy 13.6.0 / CUDA 12.5 NVRTC: NVRTC only supports up to sm_90 cubin generation, causing `CUDA_ERROR_NO_BINARY_FOR_GPU`. The fix is to force PTX mode (`compiler._use_ptx = True`) before first compilation, which produces forward-compatible `compute_90` PTX that the driver JIT-compiles for sm_120
  - CuPy's `_get_arch_for_options_for_nvrtc` and `_get_arch` are memoized -- clearing their `._cache = {}` is necessary after changing `_use_ptx` for it to take effect
  - MaterialProps struct padded to 64 bytes (16 fields * 4 bytes each) -- 13 floats + 3 ints fits exactly without extra padding needed
  - Interaction struct is 8 bytes (2 floats) -- total constant memory usage for 32 materials + 32x32 interactions = 10,240 bytes (well under the 64 KB limit)
  - PHYSICS.md doesn't exist in the repo yet -- material property values were defined from the acceptance criteria (water density=1000, acid-metal reaction_rate=0.3) plus reasonable physical constants for a falling-sand game
---
