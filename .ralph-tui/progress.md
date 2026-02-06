## Codebase Patterns

- **CUDA-GL interop pattern (Python ctypes):** Load `cudart64_12.dll` (Windows) or `libcudart.so.12` (Linux) via `ctypes.CDLL`. Declare `restype`/`argtypes` for each CUDA runtime function. Wrap `cudaGraphicsResource*` as opaque `c_void_p` handles. Always check `cudaGetLastError` after each interop call.
- **CuPy zero-copy from device pointer:** `cupy.cuda.UnownedMemory(dev_ptr, nbytes, owner=None)` -> `MemoryPointer` -> `cupy.ndarray(shape, dtype, memptr)`. This avoids any host<->device copy when writing to GL-mapped buffers.
- **Existing C++ interop reference:** `SPHSimLib/SimCudaHelper.cpp` uses a `std::map<GLuint, cudaGraphicsResource*>` to track registered buffers -- the Python module mirrors this pattern with the `CudaGLBuffer` class.
- **CUDA constant memory upload pattern (CuPy):** Use `cupy.RawModule(code=KERNEL_SOURCE)` to compile CUDA source containing `__constant__` arrays. Get the device pointer via `module.get_global('c_symbol_name')`, then upload with `cupy.cuda.runtime.memcpy(int(d_ptr), host_array.ctypes.data, nbytes, 1)` where 1=cudaMemcpyHostToDevice. Do NOT use `memcpyToSymbol` -- it's unreliable with NVRTC modules.
- **Blackwell (sm_120) PTX workaround:** CuPy's NVRTC backend defaults to generating sm_NN cubin, but if the GPU (e.g. sm_120) exceeds the bundled NVRTC's max capability (e.g. sm_90 for CUDA 12.5), the cubin won't load. Fix: set `cupy.cuda.compiler._use_ptx = True` and clear the memoized `_get_arch`/`_get_arch_for_options_for_nvrtc` caches before first compilation. This forces `compute_NN` PTX output which the driver JIT-compiles.
- **Numpy structured arrays for C struct packing:** Define `np.dtype([("field", np.float32), ...], align=True)` matching the C struct layout exactly. Assert `dtype.itemsize == sizeof(CStruct)` to catch padding mismatches. Use `array.ctypes.data` for the host pointer when uploading to GPU.
- **CuPy RawModule with external .cu files:** Read the .cu source from disk and pass to `cupy.RawModule(code=source, options=("--std=c++11", f"-I{kernel_dir}"))`. The `-I` flag lets NVRTC find `#include "common.cuh"` in the same directory. This is better than embedding CUDA source as Python strings for non-trivial kernels.
- **CUDA float3 struct packing (no trailing pad):** CUDA `float3` is exactly 12 bytes (3 x float32) with no trailing padding. Use `np.dtype([("field", np.float32, (3,))], align=False)` to match. Do NOT use `align=True` for float3 fields -- that would add 4 bytes of padding per field.
- **Grid neighbor iteration pattern (inline 27-cell loop):** For force kernels, iterate 27 neighbor cells with 3 nested loops (dx,dy,dz in [-1,1]), boundary-check each cell, look up cell_start/cell_end, skip self-interaction (j==i), and check distance within h^2. Use 0xFFFFFFFF sentinel for empty cells. This replaces the C++ template-based IterateParticlesInNearbyCells approach.
- **Tait EOS pressure with behavior classes:** Compute `p_raw = k * (pow(rho/rho0, 7) - 1)` then clamp per behavior: FLUID allows small tensile `max(p_raw, -0.5*k)`, GRANULAR clamps `max(p_raw, 0)`, GAS uses linear `k_gas * max(rho - rho0, 0)`. The different clamping prevents tensile instability in granular materials.
- **SPH precalc sign convention:** `pressure_precalc = +45/(pi*h^6)` is POSITIVE because it absorbs the double negative from the SPH momentum equation (negative gradient) and the spiky gradient constant (-45/(pi*h^6)). The force formula `f += pressure_precalc * m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky_variable(r)` produces correct repulsive forces for positive pressure.

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

## 2026-02-06 - US-008
- What was implemented:
  - `physics/kernels/common.cuh` -- Shared CUDA header with `GridParams` struct definition and `uint` typedef
  - `physics/kernels/hash_sort.cu` -- K_CalcHash kernel ported from `SPHSimLib/K_UniformGrid_Utils.inl` (calcGridCell + calcGridHash non-Morton) with boundary clamping
  - `hash_sort.py` -- Python module: GridParams numpy dtype, constant memory upload, CuPy RawModule compilation from external .cu file, `calc_hash()` kernel launch wrapper
  - `test_hash_sort.py` -- Integration tests: grid constants, struct layout, compilation, known-position hash, 100K uniform particles, boundary clamping, 500K stress test, CPU reference cross-validation
- Files changed:
  - `physics/kernels/common.cuh` (new)
  - `physics/kernels/hash_sort.cu` (new)
  - `hash_sort.py` (new)
  - `test_hash_sort.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - CuPy RawModule can compile external .cu files by reading source from disk and passing `-I{dir}` for includes -- cleaner than inline strings for multi-file kernels
  - CUDA `float3` is 12 bytes with no trailing pad; numpy dtype must use `align=False` with `(np.float32, (3,))` sub-arrays to match the layout (5 float3 = 60 bytes, not 80)
  - The C++ `calcGridCell` uses `make_int3(float3)` which truncates toward zero; porting to CUDA `(int)` cast preserves this behavior; the `np.floor` CPU reference differs for negative inputs but clamping makes them equivalent
  - Grid delta is `grid_res / grid_size` = 50/2 = 25 (not `1/cell_size` which would also be 25 for cell_size=0.04)
  - The original C++ hash kernel uses `wrapEdges=true` (modulo wrapping); the Python port uses clamping instead per acceptance criteria, which is simpler and avoids negative modulo edge cases
---

## 2026-02-06 - US-012
- What was implemented:
  - `physics/kernels/step2.cu` -- K_Step2 kernel: Tait EOS pressure, pressure force (spiky gradient, viscoplastic symmetrization), viscosity force (viscosity Laplacian), XSPH velocity correction (FLUID only). Skips STATIC and SLEEPING particles. Inline 27-cell neighbor iteration with grid cell_start/cell_end.
  - `step2.py` -- Python module: FluidParams/PrecalcParams numpy dtypes, constant memory upload (c_grid, c_fluid, c_precalc), CuPy RawModule compilation from external .cu file, `compute_step2()` kernel launch wrapper with block size 128.
  - `test_step2.py` -- Integration tests: compilation, struct sizes, precalc coefficients, rest-density zero-force, compressed repulsive forces (Newton's 3rd law), STATIC skip, SLEEPING skip, XSPH for FLUID, no XSPH for GRANULAR, viscosity opposing relative motion, GRANULAR pressure clamp, 500K particle stress test.
- Files changed:
  - `physics/kernels/step2.cu` (new)
  - `step2.py` (new)
  - `test_step2.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The C++ Step2 uses template-based neighbor iteration (SPHNeighborCalc + IterateParticlesInNearbyCells) which is elegant but overkill for Python/CuPy; an inline 27-cell loop is simpler and equally performant
  - The C++ `kernel_viscosity_precalc` includes the viscosity coefficient mu: `viscosity_precalc = mu * 45/(pi*h^6)`, not just the raw Laplacian constant. This is because the viscosity mu varies per material and needs to be baked into the precalc on the host side
  - Tait EOS with gamma=7 is much stiffer than the linear EOS in the original C++ code (`rest_pressure + gas_stiffness * (rho - rho0)`). At rest density, `pow(1.0, 7) - 1 = 0` so pressure is exactly zero, making rest-density tests straightforward
  - Grid cell functions had to be renamed (`calcGridCell_step2`) to avoid symbol conflicts since each .cu file is compiled as a separate NVRTC module -- unlike C++ where they share translation units via #include
  - The `__ldg()` intrinsic for read-only cache loads works with `const*` pointers in the kernel signature. CuPy passes device pointers directly from `cupy.ndarray` objects
  - For 2-particle test setups, particles must be sorted by grid hash and placed into cell_start/cell_end arrays matching the sorted order, otherwise the kernel won't find neighbors
---
