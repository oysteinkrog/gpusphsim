## Codebase Patterns

- **CUDA-GL interop pattern (Python ctypes):** Load `cudart64_12.dll` (Windows) or `libcudart.so.12` (Linux) via `ctypes.CDLL`. Declare `restype`/`argtypes` for each CUDA runtime function. Wrap `cudaGraphicsResource*` as opaque `c_void_p` handles. Always check `cudaGetLastError` after each interop call.
- **CuPy zero-copy from device pointer:** `cupy.cuda.UnownedMemory(dev_ptr, nbytes, owner=None)` -> `MemoryPointer` -> `cupy.ndarray(shape, dtype, memptr)`. This avoids any host<->device copy when writing to GL-mapped buffers.
- **Point sprite renderer pattern:** Vertex shader uses `uMV` (model-view) to compute eye-space distance, then `gl_PointSize = clamp(uPointScale / dist, 1, 64)`. Fragment shader uses `gl_PointCoord` mapped to [-1,1] to discard outside unit circle. Pass both `uMVP` and `uMV` as separate uniforms (MVP for position, MV for distance). Use `GL_TRUE` for transpose in `glUniformMatrix4fv` when matrices are row-major (numpy default).
- **Existing C++ interop reference:** `SPHSimLib/SimCudaHelper.cpp` uses a `std::map<GLuint, cudaGraphicsResource*>` to track registered buffers -- the Python module mirrors this pattern with the `CudaGLBuffer` class.
- **CUDA constant memory upload pattern (CuPy):** Use `cupy.RawModule(code=KERNEL_SOURCE)` to compile CUDA source containing `__constant__` arrays. Get the device pointer via `module.get_global('c_symbol_name')`, then upload with `cupy.cuda.runtime.memcpy(int(d_ptr), host_array.ctypes.data, nbytes, 1)` where 1=cudaMemcpyHostToDevice. Do NOT use `memcpyToSymbol` -- it's unreliable with NVRTC modules.
- **Blackwell (sm_120) PTX workaround:** CuPy's NVRTC backend defaults to generating sm_NN cubin, but if the GPU (e.g. sm_120) exceeds the bundled NVRTC's max capability (e.g. sm_90 for CUDA 12.5), the cubin won't load. Fix: set `cupy.cuda.compiler._use_ptx = True` and clear the memoized `_get_arch`/`_get_arch_for_options_for_nvrtc` caches before first compilation. This forces `compute_NN` PTX output which the driver JIT-compiles.
- **Numpy structured arrays for C struct packing:** Define `np.dtype([("field", np.float32), ...], align=True)` matching the C struct layout exactly. Assert `dtype.itemsize == sizeof(CStruct)` to catch padding mismatches. Use `array.ctypes.data` for the host pointer when uploading to GPU.
- **CuPy RawModule with external .cu files:** Read the .cu source from disk and pass to `cupy.RawModule(code=source, options=("--std=c++11", f"-I{kernel_dir}"))`. The `-I` flag lets NVRTC find `#include "common.cuh"` in the same directory. This is better than embedding CUDA source as Python strings for non-trivial kernels.
- **CUDA float3 struct packing (no trailing pad):** CUDA `float3` is exactly 12 bytes (3 x float32) with no trailing padding. Use `np.dtype([("field", np.float32, (3,))], align=False)` to match. Do NOT use `align=True` for float3 fields -- that would add 4 bytes of padding per field.
- **CuPy memset for 0xFFFFFFFF sentinel:** `cupy_array.data.memset(0xFF, cupy_array.nbytes)` fills each byte with 0xFF, which produces 0xFFFFFFFF for uint32 elements. This is the standard pattern for initializing cell_indexes_start to the empty-cell sentinel before each frame.
- **Grid neighbor iteration pattern (inline 27-cell loop):** For force kernels, iterate 27 neighbor cells with 3 nested loops (dx,dy,dz in [-1,1]), boundary-check each cell, look up cell_start/cell_end, skip self-interaction (j==i), and check distance within h^2. Use 0xFFFFFFFF sentinel for empty cells. This replaces the C++ template-based IterateParticlesInNearbyCells approach.
- **Tait EOS pressure with behavior classes:** Compute `p_raw = k * (pow(rho/rho0, 7) - 1)` then clamp per behavior: FLUID allows small tensile `max(p_raw, -0.5*k)`, GRANULAR clamps `max(p_raw, 0)`, GAS uses linear `k_gas * max(rho - rho0, 0)`. The different clamping prevents tensile instability in granular materials.
- **SPH precalc sign convention:** `pressure_precalc = +45/(pi*h^6)` is POSITIVE because it absorbs the double negative from the SPH momentum equation (negative gradient) and the spiky gradient constant (-45/(pi*h^6)). The force formula `f += pressure_precalc * m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky_variable(r)` produces correct repulsive forces for positive pressure.
- **Sim/render decoupling pattern:** Each render frame computes `sim_steps = clamp(round(speed * wall_dt / sim_dt), 0, max_substeps)`, runs that many simulation steps, then renders ONCE. The `speed` parameter controls how many sim-seconds elapse per wall-second. `max_substeps` (default 20) prevents spiral-of-death when sim_dt is very small relative to wall_dt.
- **Sorted vs unsorted array lifecycle:** Unsorted arrays (indexed by original particle ID) persist across frames. Sorted arrays (indexed by grid-sorted slot) are ephemeral within a single step. The reorder kernel gathers unsorted->sorted, and the integrate kernel scatters sorted->unsorted using `sort_indexes[sorted_slot] = original_id`.
- **mu(I) rheology two-pass neighbor loop:** For GRANULAR particles, the Step2 kernel does TWO neighbor traversals: (1) first pass computes gamma_dot_i (strain rate magnitude) from SPH velocity gradient, then derives eta_i via mu(I); (2) second pass accumulates viscosity force using harmonic mean eta_ij per pair. For eta_j, the pair-wise strain rate approximation `|v_ij|/|r_ij|` is used since we can't access j's full SPH-computed gamma_dot without an extra array. This avoids storing gamma_dot per particle at the cost of one extra neighbor traversal for GRANULAR only.
- **Mixed viscosity accumulator pattern:** When GRANULAR particles use per-pair variable viscosity (eta_ij * lap_const baked in), the f_viscosity accumulator has full coefficients already. For FLUID/GAS particles, the accumulator stores raw sums that get multiplied by viscosity_precalc in PostCalc. The `is_granular_i` branch in PostCalc handles this split: GRANULAR adds f_viscosity directly, FLUID/GAS multiplies by viscosity_precalc.

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

## 2026-02-06 - US-010
- What was implemented:
  - `physics/kernels/build_grid.cu` -- K_BuildDataStruct kernel that detects cell boundaries in the sorted hash array and writes cell_indexes_start/cell_indexes_end tables
  - `build_grid.py` -- Python module: CuPy RawModule compilation, constant memory upload, array allocation, `build_data_struct()` kernel launch wrapper with automatic memset of cell_start to 0xFFFFFFFF
  - `test_build_grid.py` -- Integration tests: compilation, allocation, block size, empty cell sentinel, known 8-particle/2-cell config, particle count sum consistency, boundary validation against sorted hashes, memset between frames, 500K stress test
- Files changed:
  - `physics/kernels/build_grid.cu` (new)
  - `build_grid.py` (new)
  - `test_build_grid.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The C++ reference `K_Grid_UpdateSorted` uses shared memory to cache neighboring hashes for the boundary check; the Python port uses direct global memory reads instead (`sorted_hashes[idx-1]`) since modern GPUs have L1/L2 caches that make the shared memory optimization less critical for correctness, and the simpler code is easier to verify
  - `cupy_array.data.memset(0xFF, nbytes)` sets each byte to 0xFF, producing 0xFFFFFFFF for uint32 -- this is the correct way to initialize the empty-cell sentinel before each frame
  - The kernel doesn't need `__constant__` GridParams for its core logic (it only reads sorted_hashes), but we include it for consistency with the hash_sort.cu pattern since downstream neighbor-search kernels will need it
  - Each .cu file compiled via CuPy RawModule gets its own `__constant__` symbol space -- the `c_grid` in build_grid.cu is separate from the one in hash_sort.cu, so both need their own `upload_grid_params()` call
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

## 2026-02-06 - US-014
- What was implemented:
  - `physics/kernels/step1.cu` -- K_Step1 density summation kernel (Poly6): inline 27-cell neighbor iteration, self-interaction included, density clamped to >= 1.0
  - `step1.py` -- Python module: CuPy RawModule compilation from external .cu file, constant memory upload (c_grid, c_fluid, c_precalc), `compute_step1()` kernel launch wrapper
  - `physics/kernels/integrate.cu` -- K_Integrate kernel: leapfrog velocity/position update, wall boundary penalty forces (6 walls), velocity limit clamping, velocity-based HSV coloring, writeback to unsorted arrays via sort_indexes permutation, position clamping to grid bounds
  - `integrate.py` -- Python module: IntegrateParams dtype and builder, CuPy RawModule compilation, constant memory upload (c_grid, c_integrate), `integrate()` kernel launch wrapper
  - `physics/kernels/fused_reorder.cu` -- K_FusedReorder kernel: gathers position, velocity, veleval, behavior_class, flags from unsorted to sorted order using sort_indexes permutation
  - `fused_reorder.py` -- Python module: CuPy RawModule compilation, `fused_reorder()` kernel launch wrapper
  - `simulation.py` -- SPHSimulation orchestrator: full pipeline per step (hash -> argsort -> fused_reorder -> build -> step1 -> step2 -> integrate), sim/render decoupling with speed parameter, max_substeps=20, pause/reset controls, constant memory upload to all 6 kernel modules at init (including materials)
  - `main.py` -- OpenGL renderer: GLFW window with perspective camera, CUDA-GL interop VBOs for position and color, keyboard controls (Space=pause, R=reset, +/-=speed, Esc=quit), FPS/substeps/speed display in title bar
  - Updated `hash_sort.py`, `build_grid.py`, `step2.py` to include `--use_fast_math` in CuPy RawModule compilation options
- Files changed:
  - `physics/kernels/step1.cu` (new)
  - `physics/kernels/integrate.cu` (new)
  - `physics/kernels/fused_reorder.cu` (new)
  - `step1.py` (new)
  - `integrate.py` (new)
  - `fused_reorder.py` (new)
  - `simulation.py` (new)
  - `main.py` (new)
  - `hash_sort.py` (modified -- added --use_fast_math)
  - `build_grid.py` (modified -- added --use_fast_math)
  - `step2.py` (modified -- added --use_fast_math)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The C++ code uses scale_to_simulation to convert between world and sim space during reorder (pre-scaling positions for cheaper neighbor distance checks). The Python port eliminates this complexity by keeping everything in world space, since the grid params already define the correct cell sizes for neighbor iteration
  - Step1 density summation INCLUDES self-interaction (particle i contributes to its own density with rlen_sq=0 -> (h^2)^3), unlike Step2 which skips self. This is because a particle should contribute to its own density
  - The C++ code stores velocity in two arrays: vel (half-step leapfrog velocity) and veleval (full-step average for evaluation/display). Leapfrog: vnext = vel + force*dt, vel_eval = (vel+vnext)/2, pos += vnext*dt
  - The integrate kernel writes back to UNSORTED arrays using sort_indexes[sorted_slot] = original_id. This scatter-write pattern means sorted arrays are ephemeral within a single step; unsorted arrays persist across frames
  - The fused_reorder is a separate kernel from build_data_struct (unlike the C++ K_Grid_UpdateSorted which fuses both). This simplifies the code at the cost of one extra kernel launch per step
  - Wall boundary forces use a spring-damper model: force = stiffness * penetration_depth - dampening * velocity_component. The boundary_distance parameter controls how far from the wall the force activates
  - Each CuPy RawModule compilation creates its own __constant__ symbol space, so constant memory must be uploaded separately to each module (hash, build, step1, step2, integrate, materials = 6 modules total)
---

## 2026-02-06 - US-016
- What was implemented:
  - mu(I) frictional yield viscosity for GRANULAR particles in Step2 kernel
  - `GranularParams` struct in step2.cu with mu_s=0.36, mu_2=0.70, I0=0.3, mu_max=10000, particle_spacing=0.02, mu0 constants
  - Two-pass neighbor loop for GRANULAR: first pass computes gamma_dot_i (SPH velocity gradient strain rate), second pass accumulates viscosity using harmonic mean eta_ij
  - `compute_muI_eta()` device function: I = gamma_dot * spacing / sqrt(p_eff / rho), mu_I = mu_s + (mu_2 - mu_s) / (1 + I0/max(I,1e-8)), eta = min(mu_max, mu0 + mu_I * p_eff / (gamma_dot + 1e-6))
  - Harmonic mean eta_ij = 2*eta_i*eta_j / (eta_i + eta_j + 1e-8) for GRANULAR-GRANULAR pairs
  - FLUID and GAS particles unchanged (constant mu0 viscosity via viscosity_precalc)
  - Python wrapper: GranularParams dtype, build_granular_params(), upload_granular_params()
  - Tests: mu(I) viscosity differs from constant mu0, harmonic mean produces finite forces, FLUID unaffected by granular params, no NaN at near-zero shear rate, 500K mixed particle stress test
- Files changed:
  - `physics/kernels/step2.cu` (modified -- added GranularParams, compute_muI_eta, two-pass neighbor loop for GRANULAR, per-pair viscosity branching)
  - `step2.py` (modified -- added GRANULAR_PARAMS_DTYPE, build_granular_params, upload_granular_params, DEFAULT_MU_S/MU_2/I0/MU_MAX/PARTICLE_SPACING)
  - `simulation.py` (modified -- imports and uploads granular_params to step2 module)
  - `test_step2.py` (modified -- added test_granular_muI_viscosity, test_granular_muI_harmonic_mean, test_fluid_unchanged_by_muI, test_granular_no_nan; updated 500K test with mixed particles; added c_granular symbol check and GranularParams struct size check)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - mu(I) rheology requires gamma_dot (strain rate) which is a field quantity computed from the velocity gradient tensor via SPH. This creates a chicken-and-egg problem: eta depends on gamma_dot which depends on neighbor sums. Solution: two-pass neighbor loop -- first pass computes gamma_dot, second pass uses it for viscosity
  - For eta_j in the harmonic mean, we can't access particle j's full SPH-computed gamma_dot (would need an extra per-particle array). The pair-wise approximation gamma_dot_j = |v_ij|/|r_ij| is a common simplification in granular SPH literature
  - When mixing per-pair variable viscosity (GRANULAR, with eta_ij * lap_const baked in) and constant viscosity (FLUID, with raw accumulation * viscosity_precalc in PostCalc), the f_viscosity accumulator must be handled differently in PostCalc. The cleanest approach: bake all coefficients per-pair for GRANULAR (including viscosity_precalc for GRANULAR-nonGRANULAR pairs), and keep the deferred pattern for FLUID/GAS
  - The mu_max clamp (10000 Pa·s) prevents infinite viscosity when gamma_dot approaches zero. The additional 1e-6 epsilon in the denominator (gamma_dot + 1e-6) provides a second safety net
  - GranularParams struct needs explicit padding fields (_pad0, _pad1) to reach 32 bytes (8 floats) for alignment compatibility with CUDA constant memory
---

## 2026-02-06 - US-001
- What was implemented:
  - `fallingsand3d/` project skeleton with all required Python modules, CUDA headers, GLSL shaders, requirements.txt, and run.bat
  - Placeholder files: main.py, world.py, materials.py, renderer.py, gl_cuda_interop.py, camera.py, ui.py, physics/__init__.py, physics/simulation.py
  - physics/kernels/common.cuh with uint typedef and include guard
  - shaders/particle.vert and particle.frag with basic pass-through MVP + point size + color
  - requirements.txt: cupy-cuda12x, glfw, PyOpenGL, PyOpenGL-accelerate, imgui-bundle, numpy
  - run.bat: pip install + python main.py
- Files changed:
  - All files under `fallingsand3d/` (new, 14 files total)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - Windows Python (via cmd.exe) cannot access WSL /tmp paths; use /c/ mounted paths for test scripts
  - Previous stories (US-003 through US-016) created Python files at the repo root level, not inside fallingsand3d/; those will need to be moved in a future restructuring task
  - All required packages (cupy, glfw, OpenGL, imgui_bundle) are already installed in the Windows Python environment
---

## 2026-02-06 - US-002
- What was implemented:
  - `camera.py` -- OrbitCamera class with spherical coordinate math (azimuth/elevation/distance), view matrix (look-at), perspective projection matrix, orbit/zoom/pan controls with sensitivity and clamping
  - `main.py` -- GLFW window (1280x720) with OpenGL 4.1 core profile context, orbit camera integration (right-drag=orbit, scroll=zoom, middle-drag=pan), ESC to close, dark gray background (glClearColor 0.15), FPS counter in window title, GL error checking
- Files changed:
  - `fallingsand3d/camera.py` (implemented from placeholder)
  - `fallingsand3d/main.py` (implemented from placeholder)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - GLFW on Windows with NVIDIA driver reports "4.1.0 NVIDIA 591.74" when requesting 4.1 core profile -- the driver delivers exactly the requested version, not the max supported
  - PyOpenGL `glGetString(GL_VERSION)` returns bytes on Windows -- must `.decode()` before use
  - The orbit camera uses the convention: azimuth=0 looks down +Z axis, elevation=0 is horizontal, positive elevation looks up. Pan uses world-up vector for vertical movement
  - `_look_at` and `_perspective` are module-level helper functions (not methods) since they're pure math with no state dependency
---

## 2026-02-06 - US-003 (moved into fallingsand3d/)
- What was implemented:
  - `fallingsand3d/gl_cuda_interop.py` -- Full ctypes bindings for CUDA-GL interop (register, map, get_mapped_pointer, unmap, CudaGLBuffer class with context manager and CuPy zero-copy helper)
  - `fallingsand3d/test_gl_cuda_interop.py` -- Integration test: creates GL VBO of 1M float4s, registers with CUDA, writes grid pattern via CuPy RawKernel, renders with glDrawArrays(GL_POINTS), checks all CUDA+GL errors
- Files changed:
  - `fallingsand3d/gl_cuda_interop.py` (implemented from placeholder)
  - `fallingsand3d/test_gl_cuda_interop.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The original US-003 implementation was at repo root level; US-001 created the `fallingsand3d/` project skeleton with placeholders. This iteration moved the implementation into the correct location
  - Blackwell (sm_120) PTX workaround is needed for CuPy RawKernel too, not just RawModule -- set `cupy.cuda.compiler._use_ptx = True` and clear memoized caches before first kernel compilation
  - `cudaGraphicsRegisterFlagsWriteDiscard` (value 2) is correct for write-only CUDA access to GL buffers
  - CuPy `UnownedMemory` + `MemoryPointer` + `ndarray` is the zero-copy pattern to wrap a mapped GL buffer as a CuPy array
---

## 2026-02-06 - US-004
- What was implemented:
  - `renderer.py` -- Point sprite renderer: shader compilation from .vert/.frag files, VAO with two VBOs (position float4, color float4), CUDA-GL interop registration via CudaGLBuffer, distance-based gl_PointSize, depth testing
  - `particle.vert` -- Updated vertex shader: MVP transform for position, MV transform for eye-space distance, gl_PointSize = clamp(uPointScale/dist, 1, 64)
  - `particle.frag` -- Updated fragment shader: gl_PointCoord circle discard (r^2 > 1.0), depth-based shading (shade = 1 - 0.4*r^2)
  - `main.py` -- Updated: integrates Renderer, dummy CuPy fill of 500K particles in [-0.5,0.5]^3 cube with random colors, disabled vsync for FPS measurement, glViewport in framebuffer resize callback
- Files changed:
  - `fallingsand3d/renderer.py` (implemented from placeholder)
  - `fallingsand3d/shaders/particle.vert` (updated -- float4 input, distance-based point size)
  - `fallingsand3d/shaders/particle.frag` (updated -- circle discard, depth shading)
  - `fallingsand3d/main.py` (updated -- renderer integration, dummy particle fill)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - PyOpenGL's `glUniformMatrix4fv(loc, 1, GL_TRUE, matrix)` with `GL_TRUE` for transpose handles numpy's row-major layout correctly -- OpenGL expects column-major, so the transpose flag converts on upload
  - VBO position must be float4 (not float3) for CUDA interop compatibility -- the w=1.0 component is set by the fill kernel and the vertex shader reads .xyz
  - `glGenBuffers(1)` in PyOpenGL returns a numpy scalar, not an int -- cast with `int()` before passing to CudaGLBuffer which expects a plain Python int for ctypes
  - `glSwapInterval(0)` disables vsync, allowing uncapped FPS measurement; the original US-002 used `glSwapInterval(1)` which caps at monitor refresh rate
  - The dummy particle fill uses CuPy's `cupy.random.uniform()` column-by-column on the mapped VBO -- this is simpler than a RawKernel for the initial test and still runs entirely on GPU
---

## 2026-02-06 - US-005
- What was implemented:
  - `fallingsand3d/physics/kernels/common.cuh` -- Complete CUDA common header with all shared definitions: BehaviorClass enum, MaterialProps struct (64 bytes), Interaction struct (8 bytes), GridParams struct, SimParams struct, PrecalcParams struct, packed_info bitfield macros, constant memory declarations (c_materials[32], c_interactions[32][32], c_grid, c_sim, c_precalc), and array type convention documentation
  - `fallingsand3d/test_common_cuh.py` -- Compilation test verifying struct sizes, enum values, packed_info macro correctness, constant memory symbol resolution, and kernel execution via CuPy RawModule
- Files changed:
  - `fallingsand3d/physics/kernels/common.cuh` (rewritten from placeholder)
  - `fallingsand3d/test_common_cuh.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - CuPy RawModule uses `get_function()` not `get_kernel()` to retrieve kernel handles -- the API differs from the `RawKernel` class
  - The new MaterialProps struct (US-005 spec) has different fields from the root-level prototype's MaterialProps (US-006): the new one uses eos_stiffness/eos_gamma/behavior_class/color_rgb instead of the old id/gas_stiffness/is_solid/is_flammable/friction_static/friction_kinetic. Both are 64 bytes but serve different design goals (game-oriented material system vs. direct physical properties)
  - GridParams in fallingsand3d uses `int3 grid_res` (correct for integer arithmetic in hash) vs the root prototype's `float3 grid_res` (which required casts). This is an improvement for the production codebase
  - SimParams struct naturally aligns to 64 bytes on the GPU (float3 members get 16-byte alignment) -- the actual GPU sizeof is 64 bytes for SimParams
  - PrecalcParams is kept separate from SimParams to clearly distinguish config parameters (change per scenario) from derived coefficients (recomputed from smoothing_length)
  - Total constant memory usage: materials 2048 + interactions 8192 + grid 52 + sim 64 + precalc 20 = ~10,376 bytes, well under the 64 KB limit
---

## 2026-02-06 - US-006 (fallingsand3d/)
- What was implemented:
  - `fallingsand3d/materials.py` -- Material property table (16 materials + 16 reserved) and 32x32 interaction matrix with GPU upload via CuPy RawModule constant memory. Struct layout matches `fallingsand3d/physics/kernels/common.cuh` (MaterialProps with eos_stiffness/eos_gamma/behavior_class/color_rgb fields, not the root-level prototype's id/gas_stiffness/is_solid fields)
  - `fallingsand3d/test_materials.py` -- Host-side and GPU integration tests: struct sizes, material count, water rest_density=1000.0, acid-metal reaction_rate=0.3, reserved slots zeroed, constant memory readback via test kernel
- Files changed:
  - `fallingsand3d/materials.py` (implemented from placeholder)
  - `fallingsand3d/test_materials.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The fallingsand3d `MaterialProps` struct has different fields from the root-level prototype (US-005/US-006 root): rest_density, eos_stiffness, eos_gamma, base_viscosity, friction_coeff, cohesion, buoyancy_extra, thermal_conductivity, heat_capacity, temp_melt, temp_boil, temp_ignite, behavior_class(int), color_r, color_g, color_b. All 16 fields x 4 bytes = 64 bytes with no extra padding needed
  - The `upload_to_gpu()` function accepts an optional `module` parameter to upload to any CuPy RawModule's constant memory symbols (needed because each compiled .cu file has its own __constant__ symbol space)
  - `np.float32(0.3)` is `0.30000001192092896` due to IEEE 754 -- test comparisons use `abs(x - 0.3) < 1e-6` for GPU readback
  - Behavior classes (FLUID=0, GRANULAR=1, GAS=2, STATIC=3) are stored as int in MaterialProps and used to assign physics behavior (EOS type, mu(I) vs constant viscosity, buoyancy)
---

## 2026-02-06 - US-007
- What was implemented:
  - `world.py` -- World class managing CuPy SoA arrays for all per-particle data: position(float4), velocity(float4), veleval(float4), sph_force(float4), density(float), mass(float), packed_info(uint32), temperature(float), health(float), lifetime(float), shear_rate(float), exposure_heat(float), exposure_corrode(float), color(float4), sleep_counter(uint8)
  - Constructor takes max_particles (default 500K), allocates all arrays on GPU
  - `resize(new_max)` method: destructive reallocation, kills all particles
  - `spawn_sphere(center, radius, material_id, count)`: rejection-sampled random positions in sphere, sets mass=rho0*spacing^3, packed_info via MAKE_PACKED, material-specific temperatures (LAVA=1500K, FIRE=1200K, STEAM=373K, SMOKE=500K), health=1.0
  - `spawn_cube(min_corner, max_corner, material_id, spacing)`: regular grid positions via numpy meshgrid, transferred to GPU
  - `kill_in_sphere(center, radius)`: marks particles as DEAD (packed_info=0) using squared-distance check on GPU
  - `num_active` property: on-demand GPU count of non-zero packed_info entries
  - `test_world.py` -- Integration tests covering all acceptance criteria
- Files changed:
  - `fallingsand3d/world.py` (implemented from placeholder)
  - `fallingsand3d/test_world.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - Blackwell PTX workaround is needed even for CuPy's internal elementwise kernels (array slice assignment like `arr[sl] = value`), not just user-compiled RawModule/RawKernel code. Any test script using CuPy on sm_120 must apply the workaround before first GPU operation
  - `_high_water` (allocation high-water mark) must be kept separate from `num_active` (count of live particles). After `kill_in_sphere`, dead particles create scattered holes but the high-water mark doesn't change. New spawns always go at the end (after `_high_water`). Compaction (US-029) will later reclaim holes
  - Rejection sampling for sphere spawn: generate 2x+128 candidates per batch in [-1,1]^3, filter by r^2<=1. About 52.4% of unit-cube points fall inside unit sphere, so 2x over-generation is usually sufficient in one batch
  - spawn_cube builds the meshgrid on CPU (numpy) then transfers to GPU (cupy.asarray) -- this is fine because grid generation is cheap and the transfer is a single bulk H2D copy
  - DEAD particles have packed_info=0, which means material_id=0 (DEAD) and behavior_class=0 (FLUID). The DEAD material in the table has all-zero properties, so even if a dead particle is accidentally processed, it produces zero forces
---

## 2026-02-06 - US-008 (fallingsand3d/)
- What was implemented:
  - `fallingsand3d/physics/kernels/hash_sort.cu` -- K_CalcHash kernel ported from SPHSimLib/K_UniformGrid_Utils.inl (calcGridCell + calcGridHash non-Morton) with boundary clamping, using the fallingsand3d common.cuh GridParams struct (int3 grid_res, int num_cells)
  - `fallingsand3d/hash_sort.py` -- Python module: GridParams numpy dtype (52 bytes matching common.cuh), constant memory upload, CuPy RawModule compilation from external .cu file with --use_fast_math, `calc_hash()` kernel launch wrapper
  - `fallingsand3d/test_hash_sort.py` -- Integration tests: grid constants, struct layout (52 bytes), compilation, known-position hash, 100K uniform particles, boundary clamping, 500K stress test, CPU reference cross-validation
- Files changed:
  - `fallingsand3d/physics/kernels/hash_sort.cu` (new)
  - `fallingsand3d/hash_sort.py` (new)
  - `fallingsand3d/test_hash_sort.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The fallingsand3d GridParams uses `int3 grid_res` (not `float3` like the root-level prototype) and has a `num_cells` field -- total 52 bytes vs 60 bytes in the prototype. This means the hash kernel can use integer arithmetic directly for grid_res without casting
  - The fallingsand3d common.cuh already declares `__constant__ GridParams c_grid` (along with c_materials, c_interactions, c_sim, c_precalc), so hash_sort.cu does NOT redeclare it -- unlike the root-level prototype where each .cu file declared its own c_grid
  - numpy dtype with `align=False` and `(np.int32, (3,))` for int3 fields matches CUDA int3 exactly (12 bytes, no trailing pad), same pattern as float3
---

## 2026-02-06 - US-009
- What was implemented:
  - `fallingsand3d/physics/kernels/fused_reorder.cu` -- K_FusedReorder kernel: single-pass gather of ALL SoA particle arrays (position, velocity, veleval, mass, packed_info, temperature, health, lifetime, color, sleep_counter, shear_rate) from unsorted to sorted order using sorted_index permutation, with `__ldg()` for read-only cache loads
  - `fallingsand3d/fused_reorder.py` -- Python module: CuPy RawModule compilation from external .cu file with --use_fast_math, `fused_reorder()` kernel launch wrapper with block size 256
  - `fallingsand3d/hash_sort.py` -- Added `sort_by_hash()` function: CuPy argsort (Thrust radix sort) on hash array, gathers sorted hashes and sorted original indices, supports pre-allocated output buffers to avoid per-frame allocations
  - `fallingsand3d/world.py` -- Added sorted_* temporary buffer pre-allocation to World._allocate(): sorted_position, sorted_velocity, sorted_veleval, sorted_sph_force, sorted_color, sorted_density, sorted_mass, sorted_temperature, sorted_health, sorted_lifetime, sorted_shear_rate, sorted_packed_info, sorted_sleep_counter, plus hashes/indices/sorted_hashes/sorted_indices arrays
  - `fallingsand3d/test_sort_reorder.py` -- Integration tests: sort produces non-decreasing hashes, sorted_index maps correctly (reconstruction test), pre-allocated buffer sort, kernel compilation, mass/temperature/health sum preservation, position gather correctness, World sorted buffer pre-allocation and resize, end-to-end World pipeline, 500K stress test, zero-particle edge case
- Files changed:
  - `fallingsand3d/physics/kernels/fused_reorder.cu` (new)
  - `fallingsand3d/fused_reorder.py` (new)
  - `fallingsand3d/hash_sort.py` (modified -- added sort_by_hash function)
  - `fallingsand3d/world.py` (modified -- added sorted_* buffer pre-allocation)
  - `fallingsand3d/test_sort_reorder.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - CuPy's `argsort()` uses Thrust radix sort internally for integer dtypes (uint32), making it an efficient single-call sort for grid hashes
  - A fused CUDA reorder kernel that gathers all SoA arrays in one pass (one read of sorted_index per thread) is more bandwidth-efficient than N separate CuPy fancy-indexing calls (`arr[sorted_index]`), because sorted_index is read from global memory only once instead of N times
  - `__ldg()` intrinsic in the reorder kernel provides read-only cache path for the scattered reads from unsorted arrays, which helps since the gather pattern has poor spatial locality
  - Pre-allocating sorted_* buffers at World init time avoids CuPy memory pool fragmentation in the hot simulation loop; the buffers are reused every frame
  - The `sort_by_hash()` function accepts optional pre-allocated output buffers (`sorted_hashes_out`, `sorted_indices_out`) to avoid per-frame allocations; when provided, it writes directly into the caller's buffers
  - Zero-particle edge cases are handled gracefully: `sort_by_hash` returns empty arrays, `fused_reorder` early-returns when `num_particles == 0`
---

## 2026-02-06 - US-010 (fallingsand3d/)
- What was implemented:
  - `fallingsand3d/physics/kernels/build_grid.cu` -- K_BuildDataStruct kernel: detects cell boundaries in sorted hash array and writes cell_indexes_start/cell_indexes_end tables. Uses common.cuh's `__constant__ GridParams c_grid` (does NOT redeclare it like the root-level prototype)
  - `fallingsand3d/build_grid.py` -- Python module: CuPy RawModule compilation from external .cu file with --use_fast_math, constant memory upload, array allocation, `build_data_struct()` kernel launch wrapper with automatic memset of cell_start to 0xFFFFFFFF
  - `fallingsand3d/test_build_grid.py` -- Integration tests: compilation, allocation, block size, empty cell sentinel, known 8-particle/2-cell config, particle count sum consistency, boundary validation against sorted hashes, memset between frames, 500K stress test
- Files changed:
  - `fallingsand3d/physics/kernels/build_grid.cu` (new)
  - `fallingsand3d/build_grid.py` (new)
  - `fallingsand3d/test_build_grid.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The fallingsand3d common.cuh already declares `__constant__ GridParams c_grid`, so build_grid.cu must NOT redeclare it (unlike the root-level prototype which had its own `__constant__` declaration per .cu file). This is a key architectural difference -- all fallingsand3d kernels share the same constant memory symbol space through common.cuh
  - The kernel is straightforward global-memory-only (no shared memory optimization) since modern GPUs' L1/L2 caches handle the `sorted_hashes[idx-1]` reads efficiently
  - `cell_end` is zeroed (memset 0x00) for safety even though only written cells matter -- prevents stale data from previous frames if code ever reads cell_end without checking cell_start sentinel first
---

## 2026-02-06 - US-011
- What was implemented:
  - `fallingsand3d/physics/kernels/step1.cu` -- K_Step1 SPH density summation kernel (Poly6): inline 27-cell neighbor iteration, self-interaction included, per-particle mass m_j for multi-material support, density clamped to >= 1.0, neighbor positions loaded via `__ldg()`, uses `c_grid`/`c_sim`/`c_precalc` from common.cuh (no redeclaration)
  - `fallingsand3d/step1.py` -- Python module: SimParams/PrecalcParams numpy dtypes, `build_sim_params()`/`build_precalc_params()` builders, CuPy RawModule compilation from external .cu file with --use_fast_math, constant memory upload (c_grid, c_sim, c_precalc), `compute_step1()` kernel launch wrapper with block size 128, optional pre-allocated density output buffer
  - `fallingsand3d/test_step1.py` -- Integration tests: compilation, block size, struct sizes (SimParams=64, PrecalcParams=20), precalc coefficients, single isolated particle density (mass * poly6_coeff * h^6), two particles within/beyond h, per-particle mass asymmetry, density clamp >= 1.0, uniform field rest density within 10% of rho0, self-contribution verified, 500K particle stress test
- Files changed:
  - `fallingsand3d/physics/kernels/step1.cu` (new)
  - `fallingsand3d/step1.py` (new)
  - `fallingsand3d/test_step1.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The fallingsand3d step1.cu uses `c_sim.smoothing_length_sq` and `c_precalc.poly6_coeff` from common.cuh's shared constant memory, unlike the root-level prototype which declared its own `FluidParams_Step1`/`PrecalcParams_Step1` structs with their own `__constant__` symbols. This simplifies the kernel and avoids symbol duplication
  - Per-particle mass (`__ldg(&mass[index_j])`) inside the neighbor loop enables multi-material density computation. The old prototype used a single constant mass from `c_fluid.particle_mass`, which doesn't support mass splitting or different material densities
  - For a uniform grid of water particles (spacing=0.02, h=0.04, rho0=1000), interior particles produce density ~1009.78 (0.98% error from rho0=1000), which is well within the 10% acceptance criterion. The slight overestimate is typical for SPH poly6 density at exactly the kernel support boundary
  - SimParams dtype with `align=False` totals exactly 64 bytes, matching the GPU sizeof. CUDA float3 fields have 4-byte alignment (not 16 like float4), so no padding is inserted between consecutive float3 and float fields
  - Grid cell computation is inlined directly in K_Step1 (no helper function call like the root prototype's `calcGridCell_step1`) since each kernel module is compiled independently via NVRTC -- there's no benefit to factoring out device functions that can't be shared across modules
---

## 2026-02-06 - US-012 (fallingsand3d/)
- What was implemented:
  - `fallingsand3d/physics/kernels/step2.cu` -- K_Step2 kernel: Tait EOS pressure (per-material via c_materials), pressure force (spiky gradient, viscoplastic symmetrization), viscosity force (viscosity Laplacian), XSPH velocity correction (FLUID only), mu(I) rheology two-pass neighbor loop for GRANULAR. Skips STATIC and SLEEPING particles. Uses c_grid/c_sim/c_precalc/c_materials from common.cuh plus local c_granular for mu(I) params and xsph_epsilon.
  - `fallingsand3d/step2.py` -- Python module: GranularParams numpy dtype (32 bytes), build_granular_params(), CuPy RawModule compilation from external .cu file with --use_fast_math, constant memory upload (c_grid, c_sim, c_precalc, c_materials, c_granular), compute_step2() kernel launch wrapper with block size 128, per-particle mass support via mass array argument.
  - `fallingsand3d/test_step2.py` -- Integration tests: compilation, struct sizes, precalc coefficients, rest-density zero-force, compressed repulsive forces (Newton's 3rd law), STATIC skip, SLEEPING skip, XSPH for FLUID, no XSPH for GRANULAR, viscosity opposing relative motion, GRANULAR pressure clamp, mu(I) viscosity differs from constant, harmonic mean finite forces, FLUID unchanged by mu(I), no NaN at near-zero shear rate, 500K mixed particle stress test.
- Files changed:
  - `fallingsand3d/physics/kernels/step2.cu` (new)
  - `fallingsand3d/step2.py` (new)
  - `fallingsand3d/test_step2.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The fallingsand3d step2.cu uses `c_materials[mat_id].rest_density` / `eos_stiffness` for per-material Tait EOS instead of the root prototype's single `c_fluid.rest_density` / `c_fluid.gas_stiffness`. This enables multi-material pressure computation without branching on material type.
  - The root prototype used separate `behavior_class` (int32 array) and `flags` (uint32 array) kernel arguments. The fallingsand3d version consolidates these into `packed_info` (uint32) using GET_BEHAVIOR() and IS_SLEEPING() macros from common.cuh, reducing kernel argument count and memory reads.
  - XSPH epsilon was stored in `c_fluid.xsph_epsilon` in the root prototype's FluidParams. In fallingsand3d, it's tucked into `c_granular.xsph_epsilon` (repurposing a padding slot) since there's no separate FluidParams struct -- c_sim/c_precalc handle the shared SPH params.
  - Per-particle mass (`__ldg(&mass[index_j])`) is used throughout the neighbor loop instead of the root prototype's constant `c_fluid.particle_mass`. This supports multi-material simulations where different materials have different rest densities and therefore different particle masses.
  - The `viscosity_lap_coeff` field in PrecalcParams (= 45/(pi*h^6)) is used for GRANULAR mu(I) per-pair viscosity baking, while `viscosity_precalc` (= mu * 45/(pi*h^6)) is used for FLUID/GAS constant-mu paths. These are separate fields in common.cuh's PrecalcParams to keep the coefficients clear.
---

## 2026-02-06 - US-013
- What was implemented:
  - `fallingsand3d/physics/kernels/integrate.cu` -- K_Integrate kernel: symplectic Euler velocity/position update, impulse-style SDF box boundary collisions (6 planes, restitution + Coulomb friction), GAS buoyancy (beta*(T-293)*g_y) and drag (1-c_drag*dt), velocity magnitude clamp at 50.0, particle color from material base color + temperature red tint + health fade, writeback to UNSORTED arrays via sort_indexes permutation. Skips STATIC particles.
  - `fallingsand3d/integrate.py` -- Python module: CuPy RawModule compilation from external .cu file with --use_fast_math, constant memory upload (c_sim, c_materials), `integrate()` kernel launch wrapper with block size 256, pre-allocated output buffer support.
  - `fallingsand3d/test_integrate.py` -- Integration tests (16 tests): compilation, block size, STATIC skip, gravity freefall, floor bounce with restitution, multi-step energy dissipation, GAS buoyancy, GAS drag, velocity clamp, color computation (base/hot/health), sort_indexes writeback, all 6 walls containment, Coulomb friction, XSPH position update (FLUID vs GRANULAR), 10K water pool no-NaN (1000 steps), 500K mixed-particle stress test.
- Files changed:
  - `fallingsand3d/physics/kernels/integrate.cu` (new)
  - `fallingsand3d/integrate.py` (new)
  - `fallingsand3d/test_integrate.py` (new)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - Impulse-style SDF boundary is simpler than spring-damper penalty forces: check penetration, project out, reflect normal velocity with restitution, apply Coulomb friction to tangential. No stiffness/damping constants to tune, no boundary_distance parameter needed.
  - XSPH position correction for FLUID: Step2 writes veleval = vel + eps*xsph_sum. The integrate kernel computes xsph_correction = veleval - vel (extracting just the eps*xsph_sum part), then uses pos_new = pos + dt * (vel_new + xsph_correction). This avoids double-counting the base velocity.
  - Coulomb friction model: friction_impulse = mu_wall * |v_normal|; reduction = min(friction_impulse / |v_tangential|, 1.0); v_tangential *= (1 - reduction). The min clamp prevents friction from reversing tangential direction.
  - The integrate kernel only needs c_sim and c_materials in constant memory (no c_grid, c_precalc, or c_granular), making it the simplest kernel to set up for standalone testing.
  - GAS buoyancy and drag are applied in the integrate kernel (not Step2) because they're integration-level effects, not SPH neighbor-dependent forces. This keeps Step2 focused on neighbor-based force computation.
---

## 2026-02-06 - US-014 (fallingsand3d/ orchestrator)
- What was implemented:
  - `fallingsand3d/simulation.py` -- SPH simulation orchestrator: full pipeline per step (hash -> argsort -> fused_reorder -> build -> step1 -> step2 -> integrate), sim/render decoupling with speed parameter, max_substeps=20, pause/toggle_pause, adjust_speed (+/- with clamp [0.1, 10.0]), constant memory upload to all 6 kernel modules at init (grid, sim, precalc, materials, granular), pre-allocated cell tables
  - `fallingsand3d/main.py` -- Updated: integrates Simulation orchestrator, spawns initial scene (8K water cube at y=0.3-0.7 + 8K sand bed at y=-0.5 to -0.3), CUDA-GL interop VBO copy once per frame, keyboard controls (Space=pause, R=reset, +/-=speed, Esc=quit), FPS/substeps/speed display in title bar
- Files changed:
  - `fallingsand3d/simulation.py` (new -- orchestrator)
  - `fallingsand3d/main.py` (rewritten -- simulation integration)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The simulation orchestrator touches 6 separate CuPy RawModule compilations, each with its own `__constant__` symbol space. Constant memory must be uploaded separately to each module: hash_sort (c_grid), build_grid (c_grid), step1 (c_grid, c_sim, c_precalc), step2 (c_grid, c_sim, c_precalc, c_materials, c_granular), integrate (c_sim, c_materials), materials (c_materials, c_interactions)
  - The sim/render decoupling pattern uses wall clock delta for substep calculation: `sim_steps = clamp(round(speed * wall_dt / sim_dt), 0, max_substeps)`. At speed=1.0, dt=0.001, 60fps: ~16 substeps/frame. First frame returns 0 substeps (no valid wall_dt yet)
  - The fused_reorder gathers veleval from unsorted to sorted, but Step2 overwrites sorted_veleval entirely each step (XSPH correction). So veleval doesn't need to persist across steps -- it's ephemeral within a single step
  - VBO copy uses CuPy slice assignment (`pos_arr[:n] = world.position[:n]`) which is a device-to-device memcpy (both sides are CuPy arrays on the same GPU). No host round-trip needed
  - For the initial scene, sand particles use spacing=0.04 (larger than water's 0.02) to get a coarser bed that still looks reasonable and keeps particle count manageable
---

## 2026-02-06 - US-015
- What was implemented:
  - Extended `K_Step1` kernel in `physics/kernels/step1.cu` to compute the symmetric strain-rate tensor D for GRANULAR particles using SPH velocity gradient with spiky kernel weighting
  - 6 symmetric components (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz) accumulated in the existing neighbor loop alongside density
  - PostCalc computes gamma_dot = sqrt(2 * D:D) and writes to `shear_rate_out` array; non-GRANULAR particles get 0
  - Updated `step1.py` with new `compute_step1()` signature: added velocity, density_in, packed_info inputs and shear_rate_out output; returns (density, shear_rate) tuple
  - Updated `simulation.py` to pass new arguments to `compute_step1()`, including density_in from previous step (None on first frame)
  - Updated `test_step1.py` with 3 new strain-rate tests (stationary=0, shear flow>0, non-GRANULAR=0) plus updated all existing density tests for new signature
- Files changed:
  - `fallingsand3d/physics/kernels/step1.cu` (modified -- added strain-rate tensor accumulation for GRANULAR)
  - `fallingsand3d/step1.py` (modified -- new compute_step1 signature with velocity, density_in, packed_info, shear_rate_out)
  - `fallingsand3d/simulation.py` (modified -- updated step1 call with new arguments)
  - `fallingsand3d/test_step1.py` (modified -- updated for new signature, added 3 strain-rate tests)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The strain-rate tensor D and density can be computed in a single neighbor loop pass without splitting into separate kernels -- no register pressure issues observed (the kernel compiles and runs fine for 500K particles with 6 extra float accumulators for GRANULAR)
  - The `density_in` parameter for m_j/rho_j weighting uses the previous step's density. On the first frame, the kernel falls back to rho_j=1000.0f, which is close enough for the initial strain-rate estimate
  - Using `cupy.ndarray(0, dtype=cupy.float32)` as a null-like placeholder for density_in when it's None works correctly -- the kernel checks `density_in != 0` (pointer check) and uses the fallback
  - SPH strain-rate approximation with spiky gradient is quite accurate: for a linear shear flow with applied_SR=10.0, interior particles produce gamma_dot=9.56 (4.4% error), well within expected SPH discretization error
  - The spiky gradient coefficient (`c_precalc.spiky_grad_coeff = -45/(pi*h^6)`) is NEGATIVE, so the gradient vector `gradW = spiky_grad_coeff * (h-r)^2/r * r_vec` points from i toward j (opposite to r = pos_i - pos_j). This is correct for the strain-rate computation since D_ab = sum (m_j/rho_j) * dv_a * gradW_b where dv = v_i - v_j
---

## 2026-02-06 - US-017
- What was implemented:
  - Velocity clamp and anti-creep for GRANULAR particles in the K_Integrate kernel
  - After velocity magnitude clamp and before position update: if behavior==GRANULAR AND |vel| < 0.01 AND rho_i > 0.95*rho0 AND shear_rate < 0.05, velocity is zeroed to (0,0,0)
  - Uses per-material rest_density from `c_materials[mat_id]` so different granular materials (sand=1600, dirt=1500, gravel=1800) each use their own rho0 threshold
  - Two new kernel inputs: `sorted_density` and `sorted_shear_rate` (from Step1)
  - Python wrapper accepts optional `sorted_density`/`sorted_shear_rate` params; defaults to zeros (anti-creep disabled) for backward compatibility
  - simulation.py passes sorted_density and sorted_shear_rate through to integrate
  - 6 new tests: settled pile zero velocity, flowing sand preserved, low density bypass, high shear rate bypass, 5000-step no-jitter, FLUID unaffected
- Files changed:
  - `fallingsand3d/physics/kernels/integrate.cu` (modified -- added anti-creep constants and GRANULAR velocity zeroing logic, added sorted_density/sorted_shear_rate kernel params)
  - `fallingsand3d/integrate.py` (modified -- added sorted_density/sorted_shear_rate optional params, passed to kernel)
  - `fallingsand3d/simulation.py` (modified -- passes sorted_density/sorted_shear_rate to integrate call)
  - `fallingsand3d/test_integrate.py` (modified -- added 6 anti-creep tests, updated 500K stress test with density/shear_rate data)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The anti-creep check is placed AFTER the velocity magnitude clamp and BEFORE the position update. This ordering ensures: (1) huge forces are clamped first, (2) then small residual velocities are checked for zeroing, (3) position update sees the final velocity
  - Making sorted_density/sorted_shear_rate optional with None defaults (mapped to zero arrays) preserves backward compatibility -- all 16 existing tests pass without modification since anti-creep won't trigger when density=0 (below 0.95*rho0)
  - Using per-material `c_materials[mat_id].rest_density` rather than a single global rho0 constant means each granular material type has its own appropriate anti-creep density threshold
  - The velocity check uses squared magnitude (`vel_sq < threshold_sq`) to avoid a sqrt, consistent with the existing velocity clamp pattern in the kernel
  - The 5000-step no-jitter test confirms that once anti-creep kicks in (first step: tiny velocity + high density + low shear), the particle stays at rest permanently -- no oscillation between clamped/unclamped states
---

## 2026-02-06 - US-018
- What was implemented:
  - Sleep system with hysteresis in K_Integrate kernel: sleep_counter (uint8) increments when |vel| < v_sleep (0.005) AND shear_rate < gamma_sleep (0.01), resets to 0 when either condition fails
  - SLEEPING flag set in packed_info when sleep_counter >= sleep_threshold (10), counter saturates at 255
  - Hysteresis wake condition: sleeping particles only wake when |vel| > v_wake (0.02), not v_sleep -- prevents oscillation between sleep/wake states
  - Sleeping particles: position written unchanged, velocity zeroed, skip force integration entirely (early return before acceleration/boundary computation)
  - Sleeping particles still participate in hash/sort/reorder/density (Step1) and contribute to neighbors' force calculations -- only their OWN force integration is skipped
  - Step2 already had IS_SLEEPING early return (from US-012) -- just never triggered until this story sets the flag
  - K_Integrate now outputs packed_info and sleep_counter to UNSORTED arrays (2 new output parameters) for persistence across frames
  - Python wrapper updated: integrate() returns 5-tuple (position, velocity, color, packed_info, sleep_counter) instead of 3-tuple
  - simulation.py updated to pass sorted_sleep_counter through and write back packed_info/sleep_counter
  - 7 new tests: counter_increments, counter_resets, hysteresis_wake, skip_force_integration, counter_saturates, density_contribution, sleep_wake_cycle
- Files changed:
  - `fallingsand3d/physics/kernels/integrate.cu` (modified -- sleep constants, sleep_counter/packed_info I/O, sleeping early return with hysteresis, sleep counter update logic)
  - `fallingsand3d/integrate.py` (modified -- sorted_sleep_counter input, packed_info_out/sleep_counter_out outputs, 5-tuple return)
  - `fallingsand3d/simulation.py` (modified -- pass sorted_sleep_counter and write back packed_info/sleep_counter)
  - `fallingsand3d/test_integrate.py` (modified -- updated all 22 existing tests for 5-tuple return, added 7 new sleep system tests)
  - `.ralph-tui/progress.md` (updated)
- **Learnings:**
  - The sleeping early return in K_Integrate must happen AFTER reading velocity (to check wake condition) but BEFORE reading veleval/sph_force/mass (to skip unnecessary global memory reads for sleeping particles). This ordering gives both correct hysteresis behavior and optimal memory bandwidth
  - When a sleeping particle wakes, the counter must be reset to 0 immediately AND the SLEEPING flag cleared from packed_info before continuing to the normal integration path. If the flag isn't cleared in the same frame, Step2 would still skip the particle next frame even though it's supposed to be awake
  - The sleep counter update happens AFTER position update and boundary collision, using the FINAL post-boundary velocity for the sleep check. This prevents a particle from being marked sleeping while it's actively bouncing off a wall
  - Adding packed_info_out and sleep_counter_out to K_Integrate changes the return type from 3-tuple to 5-tuple, requiring updates to ALL callers. The optional parameters with None defaults preserve backward compatibility for tests that don't care about sleep state
  - For multi-step loop tests, packed_info must be a mutable GPU array (not recreated each step from np constants) so that the SLEEPING flag persists across frames. Same for sleep_counter
---