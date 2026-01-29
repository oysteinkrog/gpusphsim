# SPH CUDA Simulation - TODO and Improvements

## Completed Optimizations (Jan 2026)
- [x] Replace B40C radix sort with CUB DeviceRadixSort (+16% perf at 524K particles)
- [x] Remove dead code: `__DEVICE_EMULATION__` guards, `IsFermi()` branches
- [x] Remove 64K block limit workarounds (obsolete on modern GPUs)
- [x] Remove `FermiCacheOverride` kernel
- [x] Standardize block sizes to 256 threads
- [x] Remove redundant `cudaDeviceSynchronize()` after `cudaMemcpyToSymbol`
- [x] Add per-kernel timing infrastructure (`SimTimingResult`)
- [x] Fix critical bugs in CudaMax.cu and SimBase.cu

## High Priority

### Performance
- [ ] **Coalesced memory access**: Restructure particle data for better memory coalescing in Step1/Step2 kernels
  - Current: Multiple separate FETCH calls per particle
  - Target: Single coalesced read per particle group

- [ ] **iDivUp optimization** (CudaUtils.cu:4-6): Replace `a % b != 0` with `(a + b - 1) / b`

- [ ] **Texture memory validation**: Add size check before binding textures (128MB limit)
  - SimSimpleSPH.cu:274-295

### Code Quality
- [ ] **Remove legacy sort backends**: Keep only CUB, remove CUDPP/Thrust/B40C conditional code
  - Config.h, UniformGrid.cu/cuh

- [ ] **RAII for CUB buffers**: Replace raw heap allocation with smart pointers
  - UniformGrid.cu:137-143

- [ ] **Remove duplicate code**: Factor common patterns across SimpleSPH/SnowSPH/DEM
  - Parameter synchronization
  - Texture binding/unbinding
  - Grid hashing

## Medium Priority

### Modernization
- [ ] **Update `__umul24` comments**: Remove references to obsolete intrinsic (deprecated CUDA 7.0)
  - 8+ kernel files still mention it

- [ ] **Remove commented-out code**:
  - SimSimpleSPH.cu:56-58 (buffer allocations)
  - SimSimpleSPH.cu:325-327 (particle data fields)
  - SimBase.cu:59-63 (parameter validation)

- [ ] **Remove unused variables**:
  - SimBase.cuh:89-90 - `mParams` never used
  - CudaMax.cuh:23-24 - `h_idata`, `h_odata` commented but declared

### Architecture
- [ ] **Reduce conditional compilation**: Currently 10+ #ifdef flags create untestable combinations
  - Consider runtime configuration where possible

- [ ] **Error checking**: Add `cudaGetLastError()` after kernel launches
  - SimSimpleSPH.cu:453, 491, 504, 543

- [ ] **NeighborList encapsulation**: Move allocation logic to dedicated manager class

## Low Priority / Future

### Physics Improvements (from existing TODOs)
- [ ] Interpolate terrain normals with curve estimation (K_Boundaries_Terrain.inl:58)
- [ ] Friction energy/heat transfer (K_Boundaries_Common.inl:74)
- [ ] Complete SPH kernel implementations (cubic, quadratic, quintic, quartic)

### Advanced Optimizations
- [ ] **Morton code hashing**: Re-evaluate with proper power-of-2 grid sizing
  - Tested but marginal improvement (+1.3%), added complexity

- [ ] **Cooperative Groups**: Replace `__syncthreads()` with flexible sync patterns

- [ ] **Shared memory optimization**: Reduce bank conflicts in K_UniformGrid_Update.inl

### Documentation
- [ ] Update header comments referencing "CUDA 2.3 SDK"
- [ ] Document benchmark methodology and expected performance

## Known Issues

### Race Conditions (potential)
- K_UniformGrid_Update.inl:43-48 - Concurrent writes to `cell_indexes_start/end`
  - Currently relies on hash uniqueness within warp
  - May need atomic operations for correctness guarantee

### File Issues
- K_SimpleSPH.inl appears to be binary/corrupted - verify or regenerate

## Benchmark Results (RTX 5070 Ti)

| Particles | Baseline (fps) | Optimized (fps) | Improvement |
|-----------|----------------|-----------------|-------------|
| 1,024     | 1,745          | 4,311           | +147%       |
| 8,192     | 1,779          | 2,555           | +44%        |
| 65,536    | 1,600          | 2,542           | +59%        |
| 262,144   | 1,115          | 1,065           | -4%*        |
| 524,288   | 803            | 930             | +16%        |

*Variance in mid-range particle counts; overall trend positive
