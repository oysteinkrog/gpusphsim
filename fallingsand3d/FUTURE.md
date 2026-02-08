# Future Roadmap -- Falling Sand 3D

Comprehensive vision document for "alien artifact" quality simulation.
Synthesized from frontier model oracles (GPT-5.2 Pro, Gemini 3 Pro),
deep research agents, and architecture analysis.

Organized by impact tier. Each item includes formulas, expected perf
impact, implementation complexity, and dependencies.

---

## Tier 1: Transformative (do these first)

### 1.1 Screen-Space Fluid Rendering (SSFR)

**The single biggest visual upgrade.** Transforms colored dots into a
continuous liquid surface with refraction, reflection, and volumetric
absorption. Both oracles and all research agents unanimously agree.

**Pipeline (5 passes, ~1.4ms at 1080p on RTX 5070 Ti):**

```
Pass 1: Depth         -- Render particles as view-space spheres, write gl_FragDepth
Pass 2: Thickness     -- Additive blend particle radii (no depth write) -> R16F
Pass 3: Bilateral blur -- Narrow-range filter on depth (separable H+V) -> smoothed depth
Pass 4: Normals       -- Reconstruct normals from smoothed depth (5-tap cross pattern)
Pass 5: Compositing   -- Fresnel + Beer-Lambert + refraction + specular -> final image
```

**Key formulas:**

Depth (sphere splatting):
```glsl
// Per-fragment: compute sphere intersection in view space
vec3 normal = vec3(pointCoord * 2.0 - 1.0, 0.0);
float r2 = dot(normal.xy, normal.xy);
if (r2 > 1.0) discard;
normal.z = sqrt(1.0 - r2);
float fragDepth = viewPos.z + normal.z * particleRadius;
gl_FragDepth = (proj[2][2] * fragDepth + proj[3][2]) / (-fragDepth);  // linearZ -> NDC
```

Narrow-range filter (Truong & Yuksel 2018, better than bilateral):
```glsl
// Only smooth depths within a narrow range of the center sample
float depthCenter = texture(depthTex, uv).r;
float rangeHalf = particleRadius * 2.0;  // narrow range
for each tap (dx, dy):
    float d = texture(depthTex, uv + offset).r;
    float diff = d - depthCenter;
    if (abs(diff) < rangeHalf):
        weight = gaussian(length(offset)) * range_weight(diff, rangeHalf);
        sum += d * weight; totalWeight += weight;
result = sum / totalWeight;
```

Compositing (Fresnel + Beer-Lambert):
```glsl
// Fresnel: Schlick approximation
float F0 = 0.02;  // water IOR ~1.33
float fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);

// Beer-Lambert absorption (volumetric color)
vec3 waterColor = vec3(0.1, 0.4, 0.7);      // absorption coefficients
vec3 absorption = exp(-thickness * 2.0 * (1.0 - waterColor));

// Refraction
vec2 refractUV = screenUV + N.xy * refractionStrength / fragDepth;
vec3 refractColor = texture(backgroundTex, refractUV).rgb * absorption;

// Combine
vec3 reflectColor = textureLod(envMap, reflect(-V, N), 0.0).rgb;
vec3 specular = pow(max(dot(reflect(-L, N), V), 0.0), 64.0) * vec3(1.0);
fragColor = mix(refractColor, reflectColor, fresnel) + specular * 0.3;
```

**FBO setup (~30MB VRAM at 1080p):**
- fboDepth: R32F (4 bytes/pixel)
- fboThickness: R16F (2 bytes/pixel)
- fboSmooth1, fboSmooth2: R32F each (ping-pong for separable blur)
- fboNormal: RGB16F (6 bytes/pixel)

**Multi-material rendering:**
- FLUID: Full SSFR pipeline (refraction, reflection, absorption)
- GRANULAR: Point sprites with normal-mapped lighting (skip fluid passes)
- GAS: Additive alpha-blended billboard particles
- Render each material type in a separate draw call to separate FBOs

**Implementation complexity:** Medium. ~500 lines GLSL + ~300 lines Python FBO setup.
All in renderer.py. No physics changes needed.

**Reference:** Truong & Yuksel, "A Narrow-Range Filter for Screen-Space Fluid
Rendering", i3D 2018. GitHub: ttnghia/RealTimeFluidRendering

---

### 1.2 Counting Sort (replace cupy.argsort)

**30-50% speedup per substep.** Current `cupy.argsort()` uses Thrust
radix sort which forces a GPU sync and cannot be captured in CUDA graphs.
Counting sort is O(N+K) where K = grid cells, fully graph-capturable,
and eliminates the single biggest performance bottleneck.

**Algorithm (3 kernels):**

```
Phase 1: Histogram (atomicAdd per cell)
    // One thread per particle
    cell = hash[particle_i]
    atomicAdd(&histogram[cell], 1)

Phase 2: Prefix sum (exclusive scan)
    // CUB DeviceScan::ExclusiveSum or custom Blelloch scan
    cell_start[c] = prefix_sum(histogram, c)

Phase 3: Scatter (write particles to sorted position)
    // One thread per particle
    cell = hash[particle_i]
    dest = atomicAdd(&write_offset[cell], 1)   // offset within cell
    sorted_index[cell_start[cell] + dest] = particle_i
```

**Why this matters:**
- cupy.argsort: ~0.8ms at 500K particles, forces synchronization
- Counting sort: ~0.15ms at 500K, fully async, graph-capturable
- Eliminates the Thrust dependency that blocks CUDA graph capture
- cell_start/cell_end arrays are produced as a byproduct (free BuildDataStruct)

**Fuses with existing pipeline:** Replace hash_sort.py + build_grid.py +
fused_sort_reorder_build.py with a single counting-sort-based pipeline.

**Implementation complexity:** Medium. ~200 lines CUDA + ~100 lines Python.
Need CUB prefix scan (available through CuPy's `cupyx.scipy.ndimage` or
inline PTX via `__shfl_xor_sync`).

---

### 1.3 Fix `__launch_bounds__` for Step2

**15-30% speedup on Step2. Trivial change.**

Current step2.cu uses `__launch_bounds__(128)` which only specifies block
size. The compiler targets minimum blocks per SM = 1, wasting occupancy.

```cuda
// Current (suboptimal):
extern "C" __global__ __launch_bounds__(128)
void K_Step2(...)

// Fixed:
extern "C" __global__ __launch_bounds__(128, 4)
void K_Step2(...)
```

With `(128, 4)`: compiler targets 4 blocks/SM = 512 threads/SM = 33%
occupancy on sm_89. This forces register usage down from 53 to ~40,
enabling more concurrent warps to hide memory latency.

Step2 is memory-bound (27-cell neighbor loop with scattered reads), so
occupancy directly translates to performance. The compiler will spill
excess registers to local memory, but L1 cache absorbs most spills.

**Apply to all heavy kernels:** Step1, Step2, Integrate, all PBF/DFSPH
kernels. Use `(128, 4)` or `(256, 2)` depending on block size preference.

**Verification:** Check register count with `--ptxas-options=-v` flag in
CuPy compilation. Target: Step2 <= 42 regs (down from 53).

---

## Tier 2: High Impact

### 2.1 Vorticity Confinement

**Makes fluid "alive" instead of "thick oil."** SPH's numerical viscosity
kills rotational energy at every substep. Vorticity confinement injects
energy back into existing vortices, producing swirls, splashes, and
turbulent detail that SPH naturally loses.

Gemini 3 Pro rated this even higher than surface tension: "Without it,
the fluid will always look like thick oil no matter what renderer you use."

**Formulas (per particle, in Step2 or NonPressureForces):**

```
1. Compute vorticity (curl of velocity):
   omega_i = curl(v) = SUM_j m_j/rho_j * (v_j - v_i) x grad_W_ij

2. Compute normalized gradient of vorticity magnitude:
   eta_i = grad(|omega|) = SUM_j m_j/rho_j * |omega_j| * grad_W_ij
   N_i = eta_i / (|eta_i| + epsilon)

3. Confinement force:
   f_conf_i = epsilon_conf * (N_i x omega_i)
```

Where `epsilon_conf` is a user-tunable strength parameter (start ~0.01-0.1).

**Performance cost:** One additional neighbor pass (~0.3ms at 500K). Can
be fused into the existing Step2 neighbor loop to avoid a separate pass:
accumulate omega alongside force computation.

**Implementation:** Add to Step2 kernel (WCSPH) or NonPressureForces
(DFSPH). Write omega to a per-particle buffer. In a second pass (or
fused), compute N and f_conf. Add to velocity.

For PBF: apply after finalize velocity update as a velocity correction.

---

### 2.2 Akinci Surface Tension

**Produces droplets, tendrils, and cohesion.** Without surface tension,
fluid particles have no attraction to each other — water looks like a
gas that happens to fall. Surface tension creates the characteristic
behaviors: droplet formation, thin sheets, dripping, and meniscus effects.

**Key insight from GPT-5.2 Pro:** Apply ONLY to surface particles to
control cost. Surface detection: particles with < N_threshold neighbors
(e.g., < 20) are surface particles. This reduces the neighbor pass cost
from O(N) to O(N_surface) ~ O(N^{2/3}).

**Akinci et al. 2013 formulas:**

```
1. Per-particle normal (during density pass):
   n_i = h * SUM_j m_j/rho_j * grad_W_ij

2. Cohesion force (attractive, creates surface tension):
   f_cohesion = -gamma * m_i * SUM_j m_j * C(|r_ij|) * r_ij/|r_ij|

   where C(r) is a piecewise polynomial:
     C(r) = (32/(pi*h^9)) * (h-r)^3 * r^3          if 0 < r <= h/2
     C(r) = (32/(pi*h^9)) * 2*(h-r)^3 * r^3 - h^6/64   if h/2 < r <= h

3. Curvature force (smooths surface):
   f_curvature = -gamma * m_i * (n_i - n_j)

4. Total surface tension:
   f_st = f_cohesion + f_curvature
```

`gamma` is the surface tension coefficient (~0.1-1.0 for water-like behavior).

**Optimization:** Only compute for particles flagged as "surface" in the
density pass. Flag: `is_surface = (neighbor_count < surface_threshold)`.

---

### 2.3 Beer-Lambert Absorption & Fresnel (Part of SSFR)

Listed separately because it can be added incrementally after basic SSFR.
The thickness pass + Beer-Lambert absorption is what makes water look like
WATER instead of tinted glass. Without it, even the best SSFR looks flat.

**Key parameters per material:**
```
FLUID (water):  absorption = vec3(0.2, 0.05, 0.01), tint = vec3(0.1, 0.4, 0.8)
FLUID (lava):   absorption = vec3(0.01, 0.3, 0.5),  tint = vec3(1.0, 0.3, 0.05)
GAS (steam):    scatter = vec3(0.8, 0.8, 0.85),     density_scale = 0.5
```

Deep water appears dark blue-green. Shallow water appears nearly transparent.
This depth-dependent coloring is impossible with per-particle colors alone.

---

## Tier 3: Signature Features

### 3.1 Ferrofluid Mode

**The "alien artifact" feature.** A magnetic fluid that forms Rosensweig
instability spikes when exposed to a magnetic field. Mouse position
controls a magnetic dipole; fluid grows spikes toward the cursor.

Both oracles independently identified this as the highest-wow-factor feature.

**Physics (Kelvin body force):**

```
f_mag = (chi / (2 * mu_0)) * grad(|H|^2)

where:
  chi   = magnetic susceptibility of the fluid (~1.0-5.0)
  mu_0  = permeability of free space (4*pi*1e-7)
  H     = magnetic field intensity
  grad(|H|^2) = gradient of squared field magnitude

For a point dipole at position p with moment m:
  H(r) = (1/(4*pi)) * (3*(m.r_hat)*r_hat - m) / |r-p|^3
  |H|^2 and its gradient can be computed analytically
```

**Implementation:**
- Add `magnetic_susceptibility` to material properties
- Add a `K_MagneticForce` kernel: per-particle, compute grad(|H|^2)
  from mouse-driven dipole position, add to velocity
- No neighbor loop needed (external field only, no particle-particle
  magnetic interaction) -- essentially free from a performance standpoint
- Run after forces, before integration
- Only applies to particles with `magnetic_susceptibility > 0`

**Visual:** Black ferrofluid with metallic specular highlights. The SSFR
pipeline renders it beautifully — the spikes create dramatic depth
discontinuities that the narrow-range filter preserves.

**Complexity:** Low-Medium. ~100 lines CUDA kernel + material property.
The magnetic field computation is per-particle (no neighbor loop).

---

### 3.2 Foam / Spray / Bubble Secondary Particles

**Transforms splashes from blobs into realistic spray.** When fluid
particles undergo violent motion (high velocity, high curvature, trapped
air), spawn lightweight secondary particles that follow simplified
physics (ballistic for spray, buoyant for bubbles, surface-following
for foam).

**Generation criteria (Ihmsen et al. 2012):**

```
// Three potentials, computed per fluid particle:
trapped_air = |v_i - v_avg_neighbors|        // velocity difference from neighbors
wave_crest  = max(0, -dot(v_i, n_surface))   // velocity into surface
kinetic     = 0.5 * |v_i|^2                  // raw kinetic energy

// Combined potential:
phi = k_ta * trapped_air + k_wc * wave_crest + k_ke * kinetic

// Spawn secondary particles when phi > threshold
if (phi > phi_threshold):
    spawn N_secondary = floor(phi / phi_per_particle) particles
    classify: SPRAY if above surface, BUBBLE if below, FOAM if at surface
```

**Secondary particle physics (very cheap, no neighbor loops):**
- SPRAY: ballistic (gravity + air drag), render as GL_POINTS
- BUBBLE: buoyancy + drag toward surface, dissolve after lifetime
- FOAM: float on surface, diffuse along surface velocity, fade out

**Performance:** Secondary particles are 10-100x cheaper than SPH
particles (no neighbor search). Can support 2-5M secondaries alongside
500K SPH particles. Render as point sprites with additive blending.

**Complexity:** Medium-High. New buffer management for secondary particle
pool, generation kernel, physics kernel, render pass. ~400 lines CUDA +
200 lines Python.

---

### 3.3 Color Mixing & Diffusion

**Colored fluids that blend when mixed.** Currently each particle has a
fixed material color. With diffusion, a red fluid and blue fluid mixing
produces purple at the interface, gradually spreading.

**Formula (SPH diffusion):**

```
dC_i/dt = D * SUM_j m_j/rho_j * (C_j - C_i) * lap_W_ij

where:
  C_i = color (RGB) of particle i
  D   = diffusion coefficient (~0.001 for slow, ~0.1 for fast)
  lap_W = Laplacian of kernel (viscosity kernel)
```

**Implementation:** Add `float4 particle_color` buffer (separate from
material color). During Step2/NonPressureForces neighbor loop, accumulate
color diffusion. In Finalize, blend material color with diffused color.

**Visual impact:** High for multi-fluid scenes. Ink-in-water, lava mixing,
chemical reactions that produce new colors.

**Complexity:** Low. ~30 lines added to existing neighbor loops + a new
color buffer.

---

## Tier 4: GPU Architecture Optimizations

### 4.1 Pack Scalar Fields into float4.w

**10-20% memory bandwidth reduction.** Many float4 arrays (position,
velocity) have unused .w components. Pack scalar fields into these:

```
position.w  = density (read-only in Step2, written in Step1)
velocity.w  = pressure (computed inline, but useful for debug/visualization)
predicted_position.w = lambda (PBF) or kappa (DFSPH)
```

**Impact:** Reduces memory transactions in neighbor loops. Every neighbor
access loads position (float4) anyway; getting density for free eliminates
a separate `__ldg(&density[j])` load.

**Risk:** Complicates code readability. Must maintain .w through all
kernels that write these arrays.

---

### 4.2 Speculative ILP Loads in Neighbor Loop

**5-15% speedup on neighbor-heavy kernels.** Current inner loop:

```cuda
// Current: serial load-then-test
for each cell in 27 neighbors:
    for j = cell_start[cell] to cell_end[cell]:
        float4 pos_j = __ldg(&position[j]);
        float3 diff = pos_i - make_float3(pos_j);
        float dist2 = dot(diff, diff);
        if (dist2 < h2 && dist2 > 0):
            // use pos_j for computation
```

Optimized: issue loads speculatively before the distance check:

```cuda
// Optimized: overlap loads with computation
for j = cell_start[cell] to cell_end[cell]:
    float4 pos_j = __ldg(&position[j]);
    float  mass_j = __ldg(&mass[j]);        // speculative load
    uint   info_j = __ldg(&packed_info[j]);  // speculative load
    float3 diff = pos_i - make_float3(pos_j);
    float dist2 = dot(diff, diff);
    if (dist2 < h2 && dist2 > 0):
        // mass_j and info_j already in registers -- no stall
```

**Benefit:** Memory loads have ~300 cycle latency. By issuing them before
the branch, they execute in parallel with the distance computation.
Wasted loads (particles outside h) cost bandwidth but save latency.

**When to apply:** Only when occupancy is low (register-bound kernels).
At high occupancy, enough warps exist to hide latency naturally.

---

### 4.3 FP16 Auxiliary Attributes

**~40% bandwidth reduction for non-critical data.** Store velocity,
color, normals, and other auxiliary data as `half4` instead of `float4`.
Position and density remain float32 for accuracy.

```cuda
// Read half4, convert to float4 for computation
half4 vel_h = __ldg((half4*)&velocity[j]);
float4 vel_j = __half42float4(vel_h);
```

**Applicable buffers:** velocity, color, shear_rate, temperature, dTdt.
NOT position (needs full precision for neighbor distance).

**sm_89 support:** Native FP16 load/store, 2x throughput for conversions.

**Risk:** Precision loss in velocity accumulation over many substeps.
Mitigate by doing accumulation in FP32 and only storing result as FP16.

---

### 4.4 Shared Memory X-Pencil Tiling

**20-40% speedup, but HIGH implementation effort.** Load one "pencil"
(row of cells along X) into shared memory, then all particles in those
cells access neighbors from shared memory instead of global memory.

```cuda
// Conceptual: cooperative loading
__shared__ float4 s_pos[MAX_PENCIL_PARTICLES];
__shared__ float  s_mass[MAX_PENCIL_PARTICLES];

// Phase 1: Load pencil into shared memory
for i in my_share_of_pencil: s_pos[i] = position[global_i];
__syncthreads();

// Phase 2: Compute using shared memory for X-direction neighbors
for j = 0 to pencil_count:
    float3 diff = pos_i - make_float3(s_pos[j]);
    ...
```

**Benefit:** 48KB shared memory at ~19 TB/s vs global memory at ~504 GB/s.
For the 9 cells in the X-row (3x3x1 slice), all accesses become shared
memory reads. The remaining 18 cells (Y/Z neighbors) still use global.

**Complexity:** Very high. Requires restructuring the entire neighbor loop
and grid traversal. Thread block geometry must match cell pencils.
Only worth pursuing after all other optimizations are exhausted.

---

### 4.5 Counting Sort Details (extends 1.2)

**Three-phase implementation with cell_start/cell_end as byproducts:**

```
Phase 1: K_Histogram (1 thread/particle, 256 threads/block)
    __shared__ uint s_hist[NUM_CELLS_PER_BLOCK];
    // Initialize shared histogram to 0
    // Each thread: atomicAdd(&s_hist[cell[i] % CELLS_PER_BLOCK], 1)
    // Reduce shared -> global histogram via atomicAdd

Phase 2: Prefix Sum (CUB DeviceScan::ExclusiveSum on histogram)
    // Output: cell_start[c] = sum of histogram[0..c-1]
    // cell_end[c] = cell_start[c] + histogram[c]

Phase 3: K_Scatter (1 thread/particle, 256 threads/block)
    __shared__ uint s_offset[NUM_CELLS_PER_BLOCK];
    // Initialize s_offset from cell_start
    // Each thread: dest = atomicAdd(&s_offset[cell[i]], 1)
    // sorted_index[dest] = i
    // Can also scatter position, velocity, etc. here (fused reorder)
```

**Replaces:** hash_sort.py + build_grid.py + fused_sort_reorder_build.py
**Produces:** sorted arrays + cell_start + cell_end in one pipeline

---

## Tier 5: Advanced Physics

### 5.1 XPBI Granular-Fluid Coupling

**Better sand behavior.** Current mu(I) rheology works but has
limitations. XPBI (Extended Position Based Interactions) uses a
Drucker-Prager yield surface for granular materials, naturally handling
friction angles, dilation, and cohesion.

**Key idea:** In the PBF constraint loop, add a Drucker-Prager constraint
for GRANULAR particles alongside the density constraint:

```
f_yield = |sigma_dev| - alpha_DP * tr(sigma) - c_DP <= 0

where:
  sigma_dev = deviatoric stress tensor
  alpha_DP  = sin(friction_angle) / sqrt(3)
  c_DP      = cohesion * cos(friction_angle)
```

Particles that violate the yield surface get position corrections that
enforce granular friction. This replaces the current tangential clamp.

**Complexity:** High. Requires per-particle stress tensor computation
and a new constraint solver. Best attempted after PBF/DFSPH are stable.

---

### 5.2 Micropolar SPH (Turbulence)

**Per-particle angular velocity for turbulent detail.** Standard SPH
only tracks linear velocity, losing rotational information. Micropolar
SPH (Bender et al. 2017) adds an angular velocity field that couples
with linear velocity through a transfer coefficient.

```
d(omega_micro)/dt = (nu_t / I) * SUM_j m_j/rho_j *
    (omega_j - omega_i) * lap_W_ij + torque_transfer

torque_transfer = nu_t * (0.5 * curl(v) - omega_micro)
```

**Visual effect:** Small-scale eddies and turbulent mixing that survive
longer than in standard SPH. Particularly visible in dam break splashes
and mixing flows.

**Cost:** One additional float3 per particle (angular velocity) + ~20%
more computation in neighbor loop. Can be added to existing Step2 or
NonPressureForces kernel.

---

### 5.3 Implicit Surface Tension (Jeske & Bender, SIGGRAPH 2024)

**Most accurate surface tension, but expensive.** Uses an implicit
formulation that handles topology changes (droplet breakup/merger)
correctly. Produces publication-quality surface tension behavior.

**Trade-off:** Requires 5-20 iterations of a linear solver per substep,
adding ~2-5ms at 500K particles. Too expensive for real-time at high
particle counts, but could be offered as a "quality mode" option.

**When to use:** Offline rendering, small scenes (< 100K particles),
or combined with temporal supersampling for cinematic quality.

---

### 5.4 Thermal Convection (Boussinesq Buoyancy)

**Already partially implemented** via temperature and dTdt buffers.
Extend with proper Boussinesq approximation for buoyancy-driven flow:

```
f_buoyancy = -beta * (T - T_ref) * g * rho_0

where:
  beta  = thermal expansion coefficient
  T_ref = reference temperature
  g     = gravity vector
```

**Heat transfer (already have buffers, just need the kernel):**

```
dT_i/dt = k_th / (rho_i * c_p) * SUM_j m_j/rho_j * (T_j - T_i) * lap_W_ij
```

**Visual:** Hot fluid rises, cold fluid sinks. Combined with color
mapping (blue=cold, red=hot), creates convection cell visualization.

---

## Tier 6: World & Infrastructure

### 6.1 Sparse/Paged Grid

Current dense grid: 50^3 = 125K cells. Fine for +/-1m worlds. Scaling
to 10m requires 250^3 = 15.6M cells (125 MB).

**Best option for our architecture:** Device hash map using open
addressing with linear probing. O(N) memory, ~10% overhead vs dense grid.

**Alternative:** Keep dense grid but use a sliding window that tracks
the AABB of active particles. Simpler, but wastes memory on sparse
distributions (e.g., explosion).

---

### 6.2 Adaptive Substep Grid Reuse

At dt=0.001, particles move < 0.01*h per step. The grid barely changes.

**Approach:** Track max displacement since last sort. Re-sort only when
`max_displacement > 0.25 * h`. Reuse cell_start/cell_end for 2-5
consecutive substeps.

**Expected:** 30-40% reduction in grid overhead. The fused sort-reorder-
build is ~15% of substep time; skipping it 3 out of 4 steps saves ~11%.

**Risk:** Not compatible with CUDA graph capture (variable topology).
Must use non-graph path when grid reuse is active.

---

### 6.3 Morton Code Sorting (Revisited)

Previous attempt (OPT-4) caused 14% regression at 524K (uniform fill).
However, for non-uniform density (compression, dam break bottom layer),
Morton codes provide better 3D spatial locality.

**Verdict from oracle consensus:** Skip. Counting sort (1.2) provides
more reliable benefits. Morton codes help cache locality but the benefit
is offset by degraded particle-data locality in neighbor loops. Revisit
only if profiling shows cache miss rates > 50% in neighbor loops under
compression.

---

## Implementation Priority Order

```
Priority  Item                          Effort   Impact    Depends On
--------  ----------------------------  -------  --------  ----------
   1      1.3 __launch_bounds__ fix      1 hour   15-30%   nothing
   2      1.1 SSFR rendering            2-3 days  visual   nothing
   3      1.2 Counting sort             1-2 days  30-50%   nothing
   4      2.1 Vorticity confinement     0.5 day   visual   nothing
   5      2.2 Surface tension           1 day     visual   nothing
   6      3.1 Ferrofluid                0.5 day   wow      2.2 (optional)
   7      3.3 Color mixing              0.5 day   visual   nothing
   8      4.1 Pack float4.w             1 day     10-20%   nothing
   9      3.2 Foam/spray/bubbles        2 days    visual   SSFR
  10      4.2 Speculative ILP loads     1 day     5-15%    nothing
  11      5.2 Micropolar turbulence     1 day     visual   2.1
  12      4.3 FP16 attributes           1 day     ~40% BW  nothing
  13      5.4 Thermal convection        0.5 day   visual   nothing
  14      5.1 XPBI granular             3 days    quality  PBF stable
  15      6.1 Sparse grid               2 days    scale    counting sort
  16      4.4 Shared memory tiling      3 days    20-40%   counting sort
  17      5.3 Implicit surface tension  2 days    quality  DFSPH stable
  18      6.2 Grid reuse                1 day     11%      counting sort
```

Items 1-3 alone would give ~50% performance improvement + photorealistic water.
Items 1-7 would produce the "alien artifact" visual quality target.
Items 1-12 represent the full vision for a state-of-the-art real-time SPH demo.

---

## Oracle Consensus Notes

**GPT-5.2 Pro** (stance: aggressive, 8/10 confidence):
- SSFR is THE visual breakthrough, everything else is secondary
- Counting sort > Morton codes for our architecture
- Ferrofluid is the signature "wow" feature
- Surface tension should be surface-particles-only for perf

**Gemini 3 Pro** (stance: balanced, 9/10 confidence):
- Agrees on SSFR as #1 priority
- Vorticity confinement MORE important than surface tension
- Current GPU significantly underutilized by architecture
- FP16 attributes worth the precision trade-off

**Key disagreement:** Morton sorting. Gemini recommends it (20-30%),
our benchmarks show 14% regression, GPT-5.2 says skip it. Resolution:
skip in favor of counting sort (1.2), which provides more reliable gains.
