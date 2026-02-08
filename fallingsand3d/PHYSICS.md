# SPH Physics Reference -- Falling Sand 3D

All mathematical formulas extracted from CUDA kernels and Python parameter builders.
Used to verify correctness against standard SPH literature and the working parent project.

## 1. SPH Kernel Functions

All kernels use smoothing length `h = 0.04 m`.

### 1.1 Poly6 Kernel (density, XSPH)

```
W_poly6(r, h) = C_poly6 * (h^2 - |r|^2)^3      for |r| <= h
              = 0                                  for |r| > h

C_poly6 = 315 / (64 * pi * h^9)
```

With h=0.04: `C_poly6 = 315 / (64 * pi * 0.04^9) = 5.968e12`

**Source**: `step1.py:132`, `step2.cu:114-117`

### 1.2 Spiky Gradient (pressure force)

```
grad_W_spiky(r, h) = C_spiky * (r / |r|) * (h - |r|)^2     for |r| <= h

C_spiky = -45 / (pi * h^6)    [full kernel, negative]
```

The code splits this into variable part and precalc coefficient:
```
grad_spiky_variable(r) = (r / |r|) * (h - |r|)^2     [positive, direction = away from neighbor]
pressure_precalc = +45 / (pi * h^6)                    [positive, absorbs negative from SPH eq.]
```

With h=0.04: `pressure_precalc = 45 / (pi * 0.04^6) = 3.497e9`

**Source**: `step2.cu:93-101`, `step1.py:135`

### 1.3 Viscosity Laplacian (viscosity force)

```
lap_W_visc(r, h) = C_visc * (h - |r|)    for |r| <= h

C_visc = 45 / (pi * h^6)
```

Code: `lap_visc_variable(r) = h - |r|`, coefficient applied in PostCalc.

```
viscosity_lap_coeff = 45 / (pi * h^6) = 3.497e9
viscosity_precalc = mu_0 * 45 / (pi * h^6)
```

**Source**: `step2.cu:106-108`, `step1.py:134,136`

---

## 2. Step 1: Density Summation

### 2.1 Formula

```
rho_i = C_poly6 * SUM_j [ m_j * (h^2 - |r_ij|^2)^3 ]
rho_i = max(rho_i, 1.0)
```

- Self-interaction included (j == i contributes, with r=0)
- **STATIC neighbors skipped** (prevents wall density inflation / sticking)
- Per-particle mass `m_j` from sorted mass array (supports multi-material)
- Density clamped to minimum 1.0
- Heat diffusion and exposure also skip STATIC neighbors

**Source**: `step1.cu:138-150` (inner loop), `step1.cu:224-225` (PostCalc)

### 2.2 Parent comparison

Parent uses constant mass pulled OUTSIDE the loop:
```
rho_i = m * C_poly6 * SUM_j [ (h^2 - |r_ij|^2)^3 ]
```

Both are mathematically equivalent when all particles have the same mass.

**Source**: parent `step1.cu:131-143`

### 2.3 Expected density at rest

For cubic lattice at spacing `dx = 0.02`, mass `m = 0.02`, h=0.04:
- ~30+ neighbors within h (all at distance <= h from particle i)
- SPH kernel-sum density ≈ 2500 kg/m³ (with m=0.02, Poly6 at h=0.04)
- rest_density is set to 2500 to match, giving rho/rho0 ≈ 1.0 at rest

---

## 3. Step 2: Pressure, Viscosity, XSPH

### 3.1 EOS Pressure (per-material gamma)

Per-material parameters from `c_materials[mat_id]`:
- `rho_0 = rest_density`
- `k = eos_stiffness`
- `gamma = eos_gamma`

```
gamma==1 (Linear, "Game SPH"):
  p = k * max(rho/rho_0 - 1, 0)

gamma!=1 (Tait):
  p = k * (pow(rho/rho_0, gamma) - 1)

GAS:      p = k * max(rho - rho_0, 0)
```

All FLUID and GRANULAR pressure clamped >= 0 (no tensile pressure).

| Material | rho_0 | k | gamma | EOS |
|----------|-------|-----|-------|------|
| WATER | 2500.0 | 500.0 | 7.0 | Tait |
| OIL | 2500.0 | 400.0 | 7.0 | Tait |
| ACID | 2500.0 | 450.0 | 7.0 | Tait |
| LAVA | 2500.0 | 30.0 | 7.0 | Tait |
| SAND | 2500.0 | 20.0 | 7.0 | Tait |
| DIRT | 2500.0 | 18.0 | 7.0 | Tait |

**Source**: `step2.cu:63-82`

### 3.2 Pressure Force (SPH momentum equation)

**STATIC neighbors skipped** in force loop (combined with density skipping, eliminates wall-sticking artifacts).

Standard SPH:
```
a_pressure_i = -SUM_j m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W_spiky(r_ij)
```

Code accumulation (inner loop):
```
f_pressure += m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky_variable(r_ij)
```

PostCalc applies coefficient:
```
a_pressure = pressure_precalc * f_pressure
           = [+45/(pi*h^6)] * SUM_j [ m_j * (p_i/rho_i^2 + p_j/rho_j^2) * (r/|r|)*(h-|r|)^2 ]
```

Sign: pressure_precalc is positive (+45/...), absorbing the double negative from
-1 * C_spiky (where C_spiky = -45/...). Net effect: positive pressure = repulsion.

**Source**: `step2.cu:315-321` (accumulation), `step2.cu:379-390` (PostCalc)

### 3.3 Viscosity Force

For FLUID/GAS particles:
```
f_viscosity += m_j * (v_j - v_i) / rho_j * (h - |r|)     [variable part]
```

PostCalc:
```
a_viscosity = viscosity_precalc * f_viscosity
            = mu_0 * [45/(pi*h^6)] * SUM_j [ m_j * (v_j - v_i) / rho_j * (h - |r|) ]
```

For GRANULAR-GRANULAR pairs: uses mu(I) effective viscosity (see Section 5).

**Source**: `step2.cu:353-359` (FLUID), `step2.cu:326-345` (GRANULAR)

### 3.4 XSPH Velocity Correction (FLUID only)

```
xsph_sum_i = SUM_j (m_j / rho_avg_ij) * (v_j - v_i) * W_poly6(|r_ij|)

where rho_avg_ij = 0.5 * (rho_i + rho_j)

veleval_out_i = v_i + epsilon * xsph_sum_i
```

`epsilon = 0.8` (from `c_granular.xsph_epsilon`)

**Source**: `step2.cu:362-369` (accumulation), `step2.cu:395-402` (output)

### 3.5 Step2 Output

```
sph_force_out = total_force    [acceleration, m/s^2]
```

**NO mass multiplication at output.** This is acceleration directly.

**Source**: `step2.cu:392`

### 3.6 Parent Step2 Output (CRITICAL DIFFERENCE)

The parent project multiplies by mass at output:
```
sph_force_out = total_force * m_j    [parent, line 453-457]
```

This means the parent's `sph_force` has an extra factor of `m = 0.02` compared to
pure acceleration. The parent's integrate.cu then does NOT divide by mass:
```
force = sph_force + gravity    [parent integrate.cu:172-176]
vnext = vel + force * dt       [parent integrate.cu:191-195]
```

**This is dimensionally inconsistent** but the parameters were tuned for it.

Net effect on SPH-to-gravity ratio:
```
Parent:  effective_accel = m^2 * precalc * SUM(press_sym * grad) + gravity
                         = 0.0004 * precalc * SUM(...) + gravity

Child:   effective_accel = m * precalc * SUM(press_sym * grad) + gravity
                         = 0.008 * precalc * SUM(...) + gravity
```

**Ratio: fallingsand3d SPH forces are 20x stronger relative to gravity than parent.**

---

## 4. Integration (Symplectic Euler)

### 4.1 Acceleration

```
accel = sph_force + gravity                    [sph_force is acceleration]
```

For GAS: `accel.y += 0.01 * (T - 293) * 9.81`  (buoyancy)

Clamped: `|accel| <= 30 m/s^2`

**Source**: `integrate.cu:316-336`

### 4.2 Velocity Update

```
v_new = v + dt * accel
```

GAS drag: `v_new *= (1 - 2.0 * dt)`

Clamped: `|v| <= 10 m/s`

GRANULAR anti-creep: if `|v| < 0.01` AND `rho > 0.95*rho_0` AND `gamma_dot < 0.05`:
set `v_new = 0`

**Source**: `integrate.cu:338-376`

### 4.3 Position Update

```
FLUID:   pos_new = pos + dt * veleval_xsph     (XSPH-smoothed velocity)
Others:  pos_new = pos + dt * v_new
```

FLUID uses XSPH-corrected velocity (`veleval_xsph = v_old + epsilon * xsph_sum`)
for position advection. This smooths compression artifacts at larger dt (Game SPH).
All other behaviors use `v_new` directly.

**Source**: `integrate.cu:378-392`

### 4.4 Parent Integration (Leapfrog)

```
vnext = vel + force * dt         [force = m*accel_sph + gravity, dimensionally wrong]
vel_eval = (vel + vnext) / 2
pos += vnext * dt
```

Parent uses penalty-force walls (stiffness=20000, dampening=256, distance=0.05).
Fallingsand3d uses impulse SDF boundaries (restitution=0.3, friction=0.5).

**Source**: parent `integrate.cu:189-207`

### 4.5 Impulse SDF Boundary (fallingsand3d)

For each of 6 box planes (world_min/max):
```
if pos outside plane:
    pos = project onto plane
    if velocity into wall:
        v_normal = -restitution * v_normal     [bounce]
        v_tangent *= (1 - min(mu * |v_n| / |v_t|, 1))   [Coulomb friction]
```

Wall friction: `mu = 0` for FLUID (prevents wall sticking), `c_sim.wall_friction` for others.

**Source**: `integrate.cu:70-174`, `integrate.cu:395-400`

---

## 5. Granular Rheology: mu(I) Model

### 5.1 Strain-Rate Tensor (computed in Step1, passed to Step2)

```
D_ij = 0.5 * SUM_k (m_k/rho_k) * (dv_i * gradW_j + dv_j * gradW_i)

gamma_dot = sqrt(2 * D:D)
          = sqrt(2 * (Dxx^2 + Dyy^2 + Dzz^2 + 2*(Dxy^2 + Dxz^2 + Dyz^2)))
```

Step1 computes `gamma_dot` from the full symmetric strain-rate tensor and writes it
to `shear_rate_out`. Step2 reads this as `shear_rate_in` via `__ldg()` — no redundant
recomputation. Previously Step2 had its own neighbor pass for gamma_dot (simplified
scalar formula); this was removed as it was redundant and wasted ~40-50% of Step2 time.

**Source**: `step1.cu:168-236`, `step2.cu` (reads `shear_rate_in`)

### 5.2 Inertial Number

```
I = gamma_dot * d / sqrt(p_eff / rho)
```

where `d = particle_spacing = 0.02`, `p_eff = max(p, 1.0)`

### 5.3 Friction Coefficient

```
mu(I) = mu_s + (mu_2 - mu_s) / (1 + I0 / I)
```

Default: mu_s=0.36, mu_2=0.70, I0=0.3

### 5.4 Effective Viscosity

```
eta = min(mu_max, mu0 + mu(I) * p_eff / (gamma_dot + 1e-6))
```

Default: mu_max=10000, mu0=1.0

### 5.5 Viscosity Force (GRANULAR-GRANULAR)

```
eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j)     [harmonic mean]
f_visc += eta_ij * [45/(pi*h^6)] * m_j * (v_j - v_i) / rho_j * (h - |r|)
```

**Source**: `step2.cu:122-130` (mu(I)), `step2.cu:326-345` (force)

---

## 6. Parameter Comparison Table

| Parameter | fallingsand3d | Parent | Notes |
|-----------|--------------|--------|-------|
| h (smoothing length) | 0.04 | 0.04 | Same |
| m (particle mass) | 0.02 | 0.02 | Matched to parent (was 0.008 = rho0*dx^3) |
| dx (particle spacing) | 0.02 | ~0.02 | Same |
| rho_0 (rest density) | 2500 | 1000 | Matches actual SPH kernel-sum density at rest (see §7) |
| k (EOS stiffness) | 500.0 (Water) | 3.0 | High k needed to resist gravity with force_scale=0.02 |
| gamma (EOS exponent) | 7.0 (all) | 7 | Tait EOS for all materials |
| mu_0 (viscosity) | 1.0 | 3.5 | Lowered; effective viscosity = mu0 * force_scale |
| epsilon (XSPH) | 0.8 | 0.5 | Higher for smoother FLUID advection |
| gravity | -9.8 | -9.8 | Same |
| dt | adaptive | 0.001 (fixed) | CFL: min(acoustic, viscous) ∈ [1e-5, 0.001]; DT_MAX=0.001 |
| force_scale | 0.02 | N/A | Matches parent's `output * m_j` convention |
| step2 output | accel * force_scale | accel * m_j | Both effectively multiply by 0.02 |
| integrate | accel = sph + g | accel = sph + g | Both treat step2 output as acceleration |
| position update | FLUID: pos += veleval_xsph * dt; others: pos += vel_new * dt | pos += vel_new * dt | FLUID uses XSPH for smooth advection |
| boundaries | impulse SDF | penalty springs | Fundamentally different |
| velocity_limit | 10 | 200 | Tight clamp prevents particles escaping grid cells |
| accel_max | 30 | N/A | Prevents shockwave-level accelerations |
| boundary_stiffness | N/A | 20000 | Only in parent |
| boundary_dampening | N/A | 256 | Only in parent |
| restitution | 0.3 | N/A | Only in fallingsand3d |
| wall_friction | 0.5 | N/A | Only in fallingsand3d |

---

## 7. Dimensional Analysis: The 20x Factor

### 7.1 SPH Pressure Acceleration

The standard SPH pressure acceleration per particle i:
```
a_i = pressure_precalc * SUM_j [ m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_var ]
```

The `m_j` inside the sum is part of the standard SPH formula. With uniform mass:
```
a_i = m * pressure_precalc * SUM_j [ (p_i/rho_i^2 + p_j/rho_j^2) * grad_var ]
```

### 7.2 Convention matching (RESOLVED)

Both projects now use equivalent force pipelines:

```
step2 inner:     f_pressure += m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W
                 f_viscosity += m_j * (v_j - v_i) / rho_j * lap_W
step2 output:    sph_force = (pp * f_pressure + vp * f_viscosity) * force_scale
integrate:       accel = sph_force + gravity
                 v += accel * dt
                 pos += v * dt
```

With m_j = 0.02 (matching parent) and force_scale = 0.02 (matching parent's `output * m_j`):
```
Effective SPH scaling = m_j * force_scale = 0.02 * 0.02 = 0.0004
Parent SPH scaling    = m_j * m_j         = 0.02 * 0.02 = 0.0004  (identical)
```

### 7.3 Why mass = 0.02 and rest_density = 2500

With m = rho0 * dx^3 = 0.008, the SPH density at rest ≈ 1000 (physical).
With m = 0.02, the SPH density at rest ≈ 2500 (inflated by particle mass choice).

Setting `rest_density = 2500` to MATCH the actual SPH kernel-sum density means:
- At rest: rho/rho0 ≈ 1.0 → p ≈ 0 (correct zero pressure in equilibrium)
- Under compression: rho/rho0 > 1.0 → Tait EOS provides restoring force

If rest_density were left at 1000 (physical), the ratio rho/rho0 ≈ 2.5 at rest,
causing massive baseline pressure even in equilibrium. With gamma=7:
p = k * (2.5^7 - 1) ≈ 610*k, leading to immediate explosion.

---

## 8. Sleep/Wake System

### 8.1 Sleep trigger
```
if |v| < 0.005 AND gamma_dot < 0.01:
    sleep_counter++
if sleep_counter >= 10:
    set SLEEPING flag
```

### 8.2 Wake trigger
```
if |v| > 0.02:
    clear SLEEPING, set JUST_WOKE
```

Hysteresis: 4x ratio between sleep (0.005) and wake (0.02) velocity thresholds.

**Source**: `integrate.cu:54-59, 280-302, 407-418`

---

## 9. Temperature Integration

```
T_new = T + dTdt * dt - COOL_RATE * (T - T_AMBIENT) * dt
T_final = clamp(T_new, 0, 5000)
```

Where:
- `dTdt` from Step1 heat diffusion: `kappa_i * [45/(pi*h^6)] * SUM_j (m_j/rho_j) * (T_j - T_i) * (h - |r|)`
- `COOL_RATE = 0.1`
- `T_AMBIENT = 293 K`

STATIC particles still conduct heat (but don't move).

**Source**: `integrate.cu:420-424`, `step1.cu:145-153, 221-222`

---

## 10. Known Issues and Discrepancies

### 10.1 FIXED: integrate.cu divided sph_force by mass
Step2 outputs acceleration. Previous code divided by mass again (125x amplification).
**Fixed**: removed `inv_mass` multiplication.

### 10.2 FIXED: SPH force scaling and particle mass
The parent multiplies step2 output by `m_j = 0.02` and uses mass=0.02 for density.
**Fixed** in two parts:
1. Particle mass changed from 0.008 (rho0*dx^3) to 0.02 (matching parent).
   This puts the Tait EOS in the same inflated-density regime (rho≈2500, p≈1828).
2. Added `force_scale = 0.02` in GranularParams, applied at step2 output.
   This matches the parent's `output * m_j` convention.
**Source**: `world.py:27`, `step2.cu:48,393`, `step2.py:49,101`

### 10.3 CHANGED: XSPH re-enabled for FLUID position update (Game SPH)
Previously XSPH was removed from position update to match parent convention.
Re-enabled for FLUID only as part of Game SPH approach: FLUID particles use
`pos += veleval_xsph * dt` which smooths compression artifacts at larger dt.
Other behaviors still use `pos += vel_new * dt`.
**Source**: `integrate.cu:378-392`

### 10.4 FIXED: Viscous CFL using mu_max from inactive materials
The adaptive timestep computed viscous CFL from ALL materials in the table,
including GRANULAR mu_max=10000, even when no GRANULAR particles existed.
This gave dt ≈ 6.4e-4 instead of dt ≈ 0.005 (DT_MAX), an 8x penalty.
**Fixed**: viscous CFL now uses base_viscosity only, not mu_max.
**Source**: `simulation.py:131-155`

### 10.5 OPEN: Boundary handling differences
Parent uses penalty-force walls (smooth, damped).
fallingsand3d uses impulse SDF (sharp reflection + friction).
May cause different settling behavior.

### 10.6 STATIC neighbor skipping + FLUID boundary improvements
STATIC particles inflated density at interfaces, creating pressure spikes and
wall-sticking.

Changes:
1. **Skip STATIC neighbors** in step1 density/heat/exposure and step2 force loops.
   Prevents wall density inflation. Self-interaction (i==j) preserved for density.
2. **XSPH position advection for FLUID**: Smooths compression artifacts.
3. **Zero wall friction for FLUID**: Prevents domain-wall sticking.
4. **All FLUID pressure clamped >= 0**: No tensile/negative pressure.

**Source**: step1.cu, step2.cu, integrate.cu

### 10.7a FIXED: SPH stability overhaul (Tait EOS + safety clamps)
With force_scale=0.02, SPH pressure forces are 50x weaker than gravity.
For deep water columns (500K particles), this causes CFL violations at
compressed densities, leading to shockwave cycles and explosions.

Root cause: CFL computed from rest-state speed of sound, but actual speed
of sound at compressed bottom (ratio 2-4x) is much higher, making dt too large.

Changes:
1. **All materials use Tait EOS (gamma=7)**: Nonlinear pressure response provides
   strong compression resistance. Linear EOS (gamma=1) was too soft for deep columns.
2. **rest_density=2500 for FLUID**: Matches actual SPH kernel-sum density at rest
   (m=0.02, dx=0.02, h=0.04), giving rho/rho0≈1.0 and near-zero pressure at rest.
3. **DT_MAX 0.005 → 0.001**: Respects CFL at compressed densities.
   ~17 substeps/frame at 60 FPS with speed=1.0.
4. **ACCEL_MAX 200 → 30**: Prevents shockwave-level accelerations.
5. **VELOCITY_LIMIT 50 → 10**: Prevents particles jumping multiple grid cells.
6. **mu0 0.1 → 1.0**: Higher viscous damping prevents oscillations.
7. **XSPH epsilon 0.5 → 0.8**: Smoother velocity field for FLUID.
8. **max_substeps 20 → 40**: Accommodates smaller dt at higher speeds.

**Source**: simulation.py, integrate.cu, step2.py, materials.py

### 10.8 OPTIMIZATION: Fused sort-reorder-build pipeline
The per-substep pipeline was restructured for performance (no physics changes):
1. **Removed redundant indices array**: K_CalcHash no longer writes `indices[idx]=idx`
   (always identity). sort_perm from argsort is used directly as sorted_indices.
2. **Reduced reorder from 11→8 arrays**: veleval, color, shear_rate removed from
   reorder since they're overwritten by K_Step2, K_Integrate, K_Step1 respectively.
3. **Fused kernel**: K_FusedSortReorderBuild combines hash gather, cell boundary
   detection, and 8-array reorder into a single kernel launch, replacing
   K_FusedReorder + K_BuildDataStruct + 2 CuPy fancy-index gathers.
4. **Reactions/spawn always run**: Previously conditionally skipped for non-reactive
   materials; now always included for CUDA graph fixed topology (§10.9).
   Non-reactive threads early-exit with negligible GPU cost.
5. **Fused wake kernels**: K_WakeSleepers + K_ClearJustWoke merged into
   K_WakeSleepersAndClearJustWoke (3→2 kernel launches).
6. **Block size 128→256** for Step1/Step2 kernels.

**Source**: simulation.py, fused_sort_reorder_build.py/.cu, wake.py/.cu

### 10.9 FIXED: GPU→CPU sync every frame in adaptive timestep
`_compute_adaptive_dt()` used `float(cp.max(...))` on GPU velocity arrays to
compute advection CFL, forcing a GPU→CPU sync (device→host transfer) every frame.
With current parameters, the acoustic CFL (dt≈0.05) always exceeds DT_MAX (0.005)
by 10x, and the velocity-based advection CFL was never binding (velocity_limit=50
and accel_max=5000 prevent supersonic particles). Removed the advection CFL
entirely; adaptive dt now uses only acoustic and viscous CFL, both precomputed
from the materials table at init time — zero GPU sync in the hot path.
**Source**: `simulation.py:151-175`

### 10.10 OPTIMIZATION: CUDA graph capture for SPH pipeline
The per-substep pipeline launches ~14 GPU operations (memsets + kernel calls) with
Python→CUDA round-trip overhead for each. CUDA graph capture records ops 3-14 into
a single replayable graph, reducing CPU→GPU dispatches from 14 to 3.

**Split point**: `cupy.argsort()` uses Thrust internally which calls
`cudaStreamSynchronize` — incompatible with graph capture. The pipeline splits:
1. `K_CalcHash` — normal launch
2. `cupy.argsort + copy` — normal launch (Thrust can't be captured)
3. Ops 3-14 as single CUDA graph launch (memsets, K_FusedSortReorderBuild,
   K_Step1, reset_freelist, K_Reactions, K_SpawnGas, K_Step2, K_Integrate,
   cell_wake_flags memset, K_MarkWakeCells, K_WakeSleepersAndClearJustWoke)

**Design details**:
- Pre-allocated `_sort_perm` buffer: graph bakes pointer addresses, so argsort
  result is copied into a stable buffer before graph launch.
- Device-side frame counter: `K_Reactions` reads `frame` from a `const uint*`
  device buffer (pointer stable, value updated via `fill()` before each launch).
- Reactions/spawn always included: graph requires fixed topology. For non-reactive
  scenes, every thread early-exits in the kernel — negligible GPU time.
- Graph invalidation: re-captured when `n` (particle count) changes (compaction,
  preset switch). Grid dimensions and `np.uint32(n)` args are baked.
- Capture uses a non-default stream (`cupy.cuda.Stream(non_blocking=True)`);
  replay launches on the default stream.

**Source**: `simulation.py:256-426`, `reactions.cu:80-81`, `reactions.py:106-186`,
`spawn.py:118-120`

### 10.11 VISUAL: FLUID particle coloring (depth + foam + density)
FLUID particles use `compute_fluid_color()` instead of the generic `compute_color()`.
Three visual effects combined:

1. **Depth gradient**: Y-position normalized within world bounds. Linear blend from
   dark blue (bottom) to bright cyan-blue (surface). Deep multipliers: 0.45-0.65x
   base color. Shallow multipliers: 1.05-1.15x base color.
2. **Density darkening**: Compressed regions (rho > rho0) darken slightly via
   `1 / (1 + 0.5 * max(rho/rho0 - 1, 0))`. Subtle depth cue reinforcement.
3. **Velocity foam**: Speed mapped to white blend via quadratic ramp (full at speed>=3).
   Fast-moving splash particles appear as white foam/spray.

Hot tint and health fade applied on top (same as base color function).

**Source**: `integrate.cu:207-280`

### 10.12 OPTIMIZATION: Spatial hash grid (replaces dense linear grid)

Replaced the dense cell_start/cell_end arrays (sized `grid_res^3`) with a
spatial hash using a fixed-size table (2^18 = 262144 entries). This enables
arbitrarily large world sizes without memory blowup.

**Old approach** (dense linear hash):
- `hash = z * ry * rx + y * rx + x` → array size = `grid_res^3`
- For world_half_size=1.0, h=0.04: grid_res=50, num_cells=125K (500KB)
- For world_half_size=10.0: grid_res=500, num_cells=125M (500MB!) → OOM

**New approach** (spatial hash):
```
hash = (cx * 73856093 ^ cy * 19349669 ^ cz * 83492791) & TABLE_MASK
```
- TABLE_SIZE = 262144 (2^18), fixed regardless of world size
- cell_start/cell_end always 262K entries (~2MB total)
- Hash collisions cause extra (harmless) distance checks, never missed neighbors
- At ~17K active cells, load factor ~6.5%, collisions rare

**GridParams struct** (common.cuh, 32 bytes):
```
grid_min   (float3, 12B) — world-space minimum corner for pos→cell
grid_delta (float3, 12B) — 1/cell_size per axis (= 1/h)
table_size (uint,    4B) — hash table size (power of 2)
table_mask (uint,    4B) — table_size - 1 (for & masking)
```

Shared helper functions in common.cuh (used by all kernels):
- `calcGridCell(pos)` — position to integer cell coordinates
- `spatialHash(cell)` — cell coordinates to hash table index

Neighbor loops no longer need bounds checking (`cx < 0 || cx >= rx`).
Any integer cell coordinates produce a valid hash in [0, table_size).

**Source**: `common.cuh`, `counting_sort.cu`, `step1.cu`, `step2.cu`,
`pbf_solver.cu`, `dfsph_solver.cu`, `wake.cu`, `hash_sort.py`, `simulation.py`

---

## 11. Multi-Solver Architecture

Three solver backends are available, selectable at runtime via UI dropdown.
Switching solvers resets the scene (avoids "physics shock" from different density packing).

| Solver | Algorithm | Substeps/frame | Inner iterations | Best for |
|--------|-----------|---------------|-----------------|----------|
| WCSPH | Weakly Compressible SPH | 10-25 at dt=0.001 | N/A | Accuracy, multi-material |
| PBF | Position Based Fluids | 1-2 at dt=1/60 | 4 constraint iters | Fast fluids |
| DFSPH | Divergence-Free SPH | 5 at dt=1/300 | 2 div + 8 dens iters | Quality + speed |

Each solver has its own CuPy RawModule with separate constant memory.
Shared constants (`c_grid`, `c_sim`, `c_precalc`, `c_materials`) are uploaded
to each module independently.

**Source**: `solver_profiles.py`, `simulation.py`

---

## 12. PBF Solver (Macklin & Muller, SIGGRAPH 2013)

### 12.1 Algorithm

```
Per substep:
  1. Predict: v* = v + dt * g,  x* = x + dt * v*
  2. Hash + Sort + Build grid from x*
  3. For iter = 1..N:
     a. Compute density rho_i from x* (Poly6 kernel)
     b. Compute lambda_i = -C_i / (SUM_j |grad_pj C_i|^2 + epsilon)
        where C_i = rho_i/rho_0 - 1
     c. Compute dx_i = (1/rho_0) SUM_j (lambda_i + lambda_j + s_corr) grad_W_spiky(x*_ij)
     d. x* += dx
  4. v_new = (x* - x_old) / dt
  5. XSPH viscosity: v_new += c SUM (m_j/rho_j) (v_j - v_i) W_poly6
  6. Boundary handling + writeback
```

### 12.2 Density Constraint

```
C_raw = rho_i / rho_0 - 1
C_i   = C_raw           if C_raw >= 0    (repulsive: enforce incompressibility)
      = C_raw * 0.05    if C_raw < 0     (weak attractive: surface cohesion)

grad_pk C_i = (m_k/rho_0) * grad_W_spiky(x*_i - x*_k)       for k != i
grad_pi C_i = (1/rho_0) * SUM_j m_j * grad_W_spiky(x*_i - x*_j)  for k == i

lambda_i = -C_i / (SUM_j |grad_pj C_i|^2 + epsilon)
```

`epsilon = pbf_relaxation` (default 0.01) stabilizes the denominator.
The 5% attractive factor for under-dense particles prevents fluid surface expansion
while avoiding the explosion that full negative C would cause.

**Source**: `pbf_solver.cu:K_PBF_ComputeLambda`

### 12.3 Position Correction (Mass-Weighted Delta)

```
dx_i = (m_j / rho_0) * SUM_j (lambda_i + lambda_j + s_corr) * grad_W_spiky(x*_ij)

|dx_i| = min(|dx_i|, 0.5 * h)    [safety clamp, rarely hit]
```

**Critical: the mass factor m_j is required.** The original Macklin & Muller paper
omits mass (assuming m=1 convention where density = SUM W). Our density uses
mass (rho = SUM m_j W), so the lambda denominator includes mass-weighted
gradients. To maintain dimensional consistency, the delta must also include
mass. Without m_j, corrections are 1/m ~ 50x too large, causing violent
overcorrection and instability.

With mass-weighted corrections, typical delta magnitudes are ~0.2mm per
iteration (for 10% density error), which is small enough that the 0.5*h
safety clamp is rarely activated. The solver converges naturally in 4
iterations to ~1-2% mean density error at dt=1/120.

Artificial pressure (`s_corr`) is disabled by default (`k=0`). At our smoothing
length (h=0.04), the Spiky gradient magnitudes are ~100x larger than in the
original paper (h~0.1), making the fixed-coefficient s_corr term dominate lambda
and cause instability. The weak attractive C provides surface cohesion instead.

**Source**: `pbf_solver.cu:K_PBF_ComputeDelta`

### 12.4 Velocity Update

```
v_new = (x* - x_old) / dt
```

Standard PBF velocity derivation from the corrected position. No artificial
damping is needed when the mass factor is correct — the constraint solver
produces appropriately-scaled corrections that don't inject excess energy.

### 12.5 XSPH Viscosity (PBF)

```
v_new += c * SUM_j (m_j / rho_j) * (v_j - v_i) * W_poly6(x*_ij)
```

`c = pbf_xsph_c` (default 0.05). Applied in K_PBF_Finalize.

**Source**: `pbf_solver.cu:K_PBF_Finalize`

### 12.5 PBF Multi-Material Handling

- **FLUID**: Full PBF constraint solve (density constraint + XSPH)
- **GRANULAR**: Participates in density constraint (lambda, dx) for volume exclusion.
  In K_PBF_ApplyDelta, Drucker-Prager friction is applied using pressure normals
  from K_PBF_ComputeLambda (see §12.7). This replaces the old axis-aligned Coulomb clamp.
  High artificial viscosity prevents granular from flowing like water.
- **GAS**: Skips PBF constraints. Predict uses gravity + drag only.
- **STATIC**: Skips all PBF kernels but contributes to neighbor density sums.

### 12.6 PBF Sleep System

PBF/DFSPH don't compute `shear_rate`, so sleep uses velocity-based activity:
```
activity = length(v_new)
sleep_counter = (activity < threshold) ? sleep_counter + 1 : 0
```

**Source**: `pbf_solver.cu:K_PBF_Finalize`

### 12.7 Drucker-Prager Friction for GRANULAR (PBF)

Replaces the old axis-aligned Coulomb friction clamp (which decomposed into
horizontal XZ / vertical Y) with a proper 3D Drucker-Prager yield surface
using the density gradient as the pressure normal direction.

#### 12.7.1 Pressure Normal Computation (K_PBF_ComputeLambda)

The density constraint gradient `grad_ci` already computed for lambda gives
the pressure direction. For GRANULAR particles, this is normalized and stored:

```
grad_ci = (1/rho0) * SUM_j m_j * grad_W_spiky(x*_i - x*_j)
grad_len = |grad_ci|

if grad_len > 1e-8:
    pressure_normal = grad_ci / grad_len    (unit vector)
else:
    pressure_normal = (0, 1, 0)             (fallback to gravity direction)
```

The `.w` component stores `grad_len` (magnitude of density gradient).
For FLUID, GAS, and STATIC particles, pressure_normal is zero (not used).

**Key insight**: `grad_ci` points in the direction of maximum density increase
(into the pile interior). This IS the pressure normal direction — the direction
along which compressive forces act in the granular material.

#### 12.7.2 Drucker-Prager Friction (K_PBF_ApplyDelta)

The PBF position correction `delta` is decomposed along the pressure normal:

```
n = pressure_normal.xyz         (unit normal from ComputeLambda)
d_dot_n = dot(delta, n)
delta_n = d_dot_n * n           (normal component: compression/expansion)
delta_t = delta - delta_n       (tangential component: sliding)

// Drucker-Prager yield criterion
max_tang = tan(phi_f) * |d_dot_n| + cohesion

if |delta_t| > max_tang:
    delta_t = delta_t * (max_tang / |delta_t|)

pos_new = pos_old + delta_n + delta_t
```

The normal component passes through unchanged (enforces incompressibility).
The tangential component is clamped to the Drucker-Prager friction cone.

#### 12.7.3 Parameters

Parameters are read from `c_granular` constant memory (uploaded per-solver with
different values for PBF vs WCSPH):

| Parameter | PBF Default | WCSPH Default | Description |
|-----------|-------------|---------------|-------------|
| `tan_phi_f` | 0.25 (= tan(14°)) | 0.781 (= tan(38°)) | Friction cone slope |
| `cohesion` | 0.0 | 0.002 | Free tangential (meters) |

**Why PBF needs a lower ratio than WCSPH**: In WCSPH, `tan_phi_f` relates
normal force to tangential force limit (force-space). In PBF, the corrections
are position deltas — each iteration directly moves particles. With 4 iterations
per frame, `tan(38°)=0.781` in position space allows catastrophic tangential
spreading (sand behaves like water). Position-space `tan(14°)=0.25` produces
realistic 25-35° angle of repose.

A static friction dead zone (`norm_mag < 5e-5`) zeroes all tangential correction
when the normal compression is negligible (< 0.05mm), preventing numerical creep.

#### 12.7.4 Comparison with XPBI

Full XPBI (eXtended Position-Based Dynamics for Inelastic materials) tracks a
deformation gradient per particle and uses SVD for yield surface projection.
This implementation achieves similar angle-of-repose results without the
deformation gradient overhead by leveraging the density gradient already
available from the PBF constraint computation.

**Source**: `pbf_solver.cu:K_PBF_ApplyDelta` (Drucker-Prager friction),
`pbf_solver.cu:K_PBF_ComputeLambda` (pressure normal output),
`solver_profiles.py` (`pbf_friction_ratio`, `pbf_friction_cohesion`),
`step2.py:build_granular_params()` (builds params uploaded to PBF module)

---

## 12.8 Implicit Surface Tension (Quality Mode)

Iterative Jacobi smoothing of surface particle velocities for smoother, more
cohesive fluid surfaces. WCSPH only, limited to < 100K particles.

### Algorithm

For each Jacobi iteration (5-20 per substep):
```
For each FLUID particle i with neighbor_count < surface_threshold:
  w_surface = 1 - neighbor_count / surface_threshold
  dv_i = sigma * w_surface * SUM_j (m_j/rho_j) * (v_j - v_i) * W_poly6(r_ij)
  v_new_i = v_old_i + dv_i
```

Surface particles (fewer neighbors) receive stronger smoothing. Interior
particles pass through unchanged. Uses ping-pong velocity buffers for proper
Jacobi iteration (read from old, write to new).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma` | 0.5 | Surface tension strength (higher = stronger smoothing) |
| `surface_threshold` | 25.0 | Neighbor count below which particle is "surface" |
| `num_iterations` | 5 | Jacobi iterations per substep |

### Pipeline Position

Runs between Step2 (force computation) and Integrate in the WCSPH pipeline.
Modifies `sorted_velocity` in-place before integration uses it. Uses
`sorted_veleval` as scratch buffer for ping-pong (overwritten by integrate).

**Source**: `implicit_st.cu:K_IST_Iterate`, `implicit_st.py`, `simulation.py`

---

## 13. DFSPH Solver (Bender & Koschier, SCA 2015)

### 13.1 Algorithm

```
Per substep:
  1. Hash + Sort + Build grid
  2. Compute density rho_i + precompute alpha_i factor
  3. Compute non-pressure forces: f_visc, f_gravity
  4. Predict velocity: v* = v + dt * a_nonpressure   (gravity + viscosity/rho_i)
  5. Divergence-free solver (N_div iterations):
     kappa_v_i = -div(v*_i) * alpha_i / dt
     v*_i -= dt * SUM_j m_j * (kappa_v_i/rho_i + kappa_v_j/rho_j) * grad_W_ij
  6. Density solver (Jacobi iteration on pressure, N_dens iterations):
     Initialize p_rho2 from warm-start (previous frame, scaled by 0.5)
     Compute initial a_press from warm-started p_rho2
     For each iteration:
       Predict density with v_total = v + dt*a_press
       Jacobi-update p_rho2 from density residual
       Recompute a_press from updated p_rho2
     Apply final: v* += dt * a_press
  7. Final position: x_final = x + dt * v*_final
  8. Boundary handling, color, writeback (kappa written back for warm-start)
```

### 13.2 Alpha Factor (Diagonal Preconditioner)

The alpha factor encodes how much a particle's density responds to pressure corrections:

```
grad_sum = SUM_j (m_j/rho_j) * grad_W_spiky(x_ij)     [vector sum]
grad_norm_sum = SUM_j (m_j/rho_j)^2 * |grad_W_spiky(x_ij)|^2    [scalar sum]

alpha_i = 1 / max(|grad_sum|^2 + grad_norm_sum, 1e-6)
```

The first term (`|grad_sum|^2`) is the squared norm of the vector sum of gradients.
The second term is the sum of squared norms. Both are accumulated in a single neighbor loop.

For GRANULAR particles, alpha is multiplied by a compliance factor (0.7) to allow
some compression for granular packing without destabilizing the solver.

**Source**: `dfsph_solver.cu:K_DFSPH_ComputeDensityAlpha`

### 13.3 Divergence Solver

Enforces divergence-free velocity field:
```
div_v_i = SUM_j (m_j/rho_j) * (v*_i - v*_j) . grad_W_ij
kappa_v_i = -div_v_i * alpha_i / dt
dv_i = -dt * SUM_j m_j * (kappa_v_i/rho_i + kappa_v_j/rho_j) * grad_W_ij
v*_i += dv_i
```

**Important**: The correction uses `kappa/rho` (not `kappa/rho^2`). This is the SPlisHSPlasH
formulation where alpha already absorbs one factor of 1/rho via the (m_j/rho_j) weighting
in its computation. Using the original paper's `kappa/rho^2` with this alpha would make
corrections ~rho times too weak (a critical bug we fixed).

Note: kappa_v **multiplies** by alpha (not divides). Since alpha = 1/denom, dividing
by alpha would multiply by the full denominator, producing astronomical corrections
for well-connected particles. The negative sign ensures positive divergence (expanding)
produces negative kappa, which drives an inward velocity correction.

Default: 2 iterations. Each iteration requires two neighbor passes (compute kappa_v,
then correct velocity).

**Source**: `dfsph_solver.cu:K_DFSPH_ComputeKappaV`, `K_DFSPH_CorrectVelocityDiv`

### 13.4 Density Solver (Jacobi Pressure Iteration)

The density solver uses a SPlisHSPlasH-style Jacobi iteration on the **pressure variable**
(`p/rho^2`), not velocity. This allows multiple iterations to converge because the pressure
field carries information between iterations through the `A*p` feedback term.

**Three kernels per iteration:**

1. **K_DFSPH_ComputePressureAccel** — Compute pressure acceleration from current p/rho^2:
```
a_press_i = -SUM_j V_j * (p_rho2_i + p_rho2_j) * grad_W_ij
where V_j = m_j / rho_j
```

2. **K_DFSPH_DensitySolverUpdate** — Predict density using total velocity, Jacobi-update pressure:
```
v_total_i = v_i + dt * a_press_i        [velocity including current pressure effect]
drho = SUM_j (m_j/rho_j) * (v_total_i - v_total_j) . grad_W_ij
density_adv = rho_i/rho_0 + dt * drho   [predicted density ratio]
residual = density_adv - 1.0             [positive = compressed]
p_rho2 = max(p_rho2 + omega * residual * alpha / dt^2, 0)
```
The `A*p` feedback comes from including `a_press` in the density prediction, which
accounts for the effect of current pressure on neighbors. This is what makes multiple
iterations converge — pure velocity correction without pressure feedback would repeat
the same correction.

3. **K_DFSPH_ApplyPressureVelocity** — Apply final pressure to velocity (once after convergence):
```
v*_i += dt * a_press_i
```

**Warm-starting:** Pressure values (`p_rho2`) are carried from the previous frame
through the sort-reorder pipeline (`kappa` → `sorted_kappa`). At the start of each
density solve, the warm-started values are scaled by a decay factor:
```
p_rho2 *= warm_start     (default 0.5 = carry 50% from previous frame)
```
Warm-starting reduces the number of iterations needed: 2 iters with warm-start ≈ 4 iters
without. Values above 0.8 cause pressure accumulation (particles ejected to boundaries).

**Alpha cap as Jacobi relaxation:** Interior particles have actual alpha ≈ 1.1e-3
(denom ≈ 915), but alpha is capped at 1e-5 in K_DFSPH_ComputeDensityAlpha. This cap
acts as an implicit relaxation controller:
```
correction factor k = omega * alpha_cap / dt^2 ≈ 1.0 * 1e-5 / (1/300)^2 ≈ 0.9
```
Without the cap, k ≈ 90 (explosive overshoot because actual alpha is 100x larger).
The spectral radius of the Jacobi system is close to 1, giving fundamentally slow
convergence (~7% error reduction per iteration).

**Convergence rates** (dt=1/300, warm_start=0.5, 36K particles):

| Iterations | Mean density error | ms/frame (5 sub/frame) |
|------------|-------------------|----------------------|
| 4          | 29.6%             | 9.2                  |
| 8          | 17.8%             | 11.7                 |
| 12         | 11.7%             | 14.6                 |
| 16         | ~10%              | ~16                  |
| 32         | ~7%               | ~27                  |

Default: 8 iterations with warm_start=0.5 at dt=1/300 (5 substeps/frame). This gives
18% mean density error at 11.7ms/frame — a reasonable quality/performance tradeoff for
real-time use.

**Source**: `dfsph_solver.cu:K_DFSPH_ComputePressureAccel`, `K_DFSPH_DensitySolverUpdate`,
`K_DFSPH_ApplyPressureVelocity`, `simulation.py:615-655`

### 13.5 Non-Pressure Forces

```
FLUID:    a_i = gravity + (mu_0 / rho_i) * [45/(pi*h^6)] * SUM_j m_j*(v_j-v_i)/rho_j*(h-|r|)
GRANULAR: a_i = gravity + (mu_0 / rho_i) * [45/(pi*h^6)] * SUM_j m_j*(v_j-v_i)/rho_j*(h-|r|)
GAS:      a_i = gravity + buoyancy - drag * v_i + viscosity (same as FLUID)
```

The `1/rho_i` factor is critical: without it, viscosity is ~2500x too strong (at rho≈2500),
making the fluid behave like a solid. With `mu_0 = 1.0` and `rho_i ≈ 2500`, the effective
kinematic viscosity is `nu = mu_0/rho_i = 0.0004 m^2/s` (400x physical water, giving nice
smooth game fluid behavior).

Velocity updated: `v*_i = v_i + dt * a_i`

**Source**: `dfsph_solver.cu:K_DFSPH_NonPressureForces`

### 13.6 DFSPH Multi-Material Handling

- **FLUID**: Full DFSPH (divergence + density solvers)
- **GRANULAR**: Participates in both solvers. Alpha multiplied by 0.7 compliance factor.
  mu(I) viscosity in non-pressure forces. Shear_rate computed in ComputeDensityAlpha.
- **GAS**: Skips pressure solvers. Non-pressure forces include buoyancy + drag.
- **STATIC**: Skips all solver kernels, contributes to neighbor density sums.

### 13.7 DFSPH Sleep System

Same velocity-based activity metric as PBF (§12.6).

**Source**: `dfsph_solver.cu:K_DFSPH_Finalize`

---

## 14. Thermal Convection (Boussinesq Buoyancy)

### 14.1 Formula

For FLUID particles with non-zero `thermal_expansion` (β):
```
a_buoy_y = +β * (T - T_ambient) * g
```

Where:
- β = `c_materials[mat_id].thermal_expansion` (1/K)
- T_ambient = 293 K
- g = 9.81 m/s²

From the Boussinesq approximation: `ρ_eff = ρ_0 * (1 - β*(T-T_0))`.
Hot fluid (T > T_0) has lower effective density → buoyancy force pushes it upward.
With gravity pointing down (negative y), the buoyancy acceleration is upward (positive y)
when T > T_ambient. The formula `accel.y += β * (T - T_ambient) * 9.81` achieves this:
- T > 293: positive contribution → upward (hot rises)
- T < 293: negative contribution → downward (cold sinks)

### 14.2 Material Properties

| Material | thermal_expansion (β) | Notes |
|----------|----------------------|-------|
| WATER    | 0.0003               | ~3×10⁻⁴ 1/K (physical water ≈ 2.1×10⁻⁴) |
| OIL      | 0.0007               | Higher expansion than water |
| LAVA     | 0.0001               | Low expansion (dense fluid) |
| ACID     | 0.0003               | Similar to water |
| Others   | 0.0                  | No thermal convection |

### 14.3 Solver Support

- **WCSPH**: Applied in `K_Integrate` after gravity
- **PBF**: Applied in `K_PBF_Finalize` as velocity correction (v += β*(T-T₀)*g*dt)
- **DFSPH**: Applied in `K_DFSPH_NonPressureForces` alongside gravity

**Source**: `integrate.cu`, `pbf_solver.cu:K_PBF_Finalize`, `dfsph_solver.cu:K_DFSPH_NonPressureForces`

---

## 15. Kernel Launch Bounds

All heavy kernels use `__launch_bounds__(256, 4)`:
- Block size: 256 threads
- Minimum blocks per SM: 4 (targets ~67% occupancy)
- Previous setting was `(256, 2)` which only targeted 33% occupancy

This forces the compiler to use fewer registers per thread, enabling more concurrent
blocks per SM. The tradeoff is potential register spilling, but for memory-bound kernels
(Step1/Step2/Integrate) the increased occupancy hides memory latency better.

Affected kernels: K_Step1, K_Step2, K_Integrate, all K_PBF_* (5), all K_DFSPH_* (13).

**Source**: all .cu files in physics/kernels/

---

## 16. Vorticity Confinement

### 16.1 Vorticity Computation (Step1)

Curl of velocity computed in the Step1 neighbor loop for FLUID particles:
```
omega_i = SUM_j (m_j / rho_j) * (v_j - v_i) × grad_W_spiky(r_ij)
```

Output: `vorticity_out[i] = float4(omega_x, omega_y, omega_z, |omega|)`

The magnitude `|omega|` is stored in `.w` for use by the Step2 confinement force.

**Source**: `step1.cu` (neighbor loop, FLUID branch)

### 16.2 Confinement Force (Step2)

The vorticity confinement force re-injects energy lost to numerical dissipation:

```
eta_i = SUM_j (m_j / rho_j) * |omega_j| * grad_W_spiky(r_ij)
N_i = eta_i / |eta_i|                          [normalized vorticity gradient]
f_conf = epsilon * (N_i × omega_i)              [confinement force]
```

Where `epsilon = c_granular.vorticity_epsilon` (default 0.05).

The confinement force is perpendicular to the vorticity axis and aligned with the
gradient of vorticity magnitude, amplifying existing rotational structures without
creating spurious rotation.

**Source**: `step2.cu` (second neighbor loop after main force computation)

### 16.3 Solver Support

- **WCSPH**: Full vorticity computation in Step1 + confinement in Step2
- **PBF/DFSPH**: Vorticity buffer allocated but not computed (no neighbor loop additions).
  Dye passthrough only.

---

## 17. Akinci Surface Tension

### 17.1 Surface Normal Computation (Step1)

Surface normal accumulated in Step1 neighbor loop for FLUID particles:
```
n_i = SUM_j (m_j / rho_j) * grad_W_spiky(r_ij)
neighbor_count_i = count of neighbors within h
```

Output: `normal_out[i] = float4(n_x, n_y, n_z, as_float(neighbor_count))`

The neighbor count is packed into `.w` as a reinterpreted float for efficient storage.

**Source**: `step1.cu` (neighbor loop, FLUID branch)

### 17.2 Surface Detection

A particle is classified as a surface particle when:
```
neighbor_count < 25
```

Interior FLUID particles typically have 30+ neighbors. Particles with fewer neighbors
are near the free surface and receive surface tension forces.

### 17.3 Curvature-Based Surface Tension (Step2)

For surface particles (neighbor_count < 25), the Akinci cohesion/curvature model:
```
f_st = -gamma * (n_i - n_j)     [curvature force, summed over neighbors]
```

Where `gamma = c_granular.surface_tension_gamma` (default 1.0).

The curvature force drives particles toward minimizing surface area by aligning
surface normals. The negative sign ensures convex surfaces (normals pointing outward)
produce inward forces (cohesion).

**Note**: The full Akinci model also includes a cohesion kernel C(r) for particle-pair
attraction. Currently only the curvature term is implemented — it provides the dominant
surface-minimizing behavior for our particle spacing (h=0.04, dx=0.02).

**Source**: `step2.cu` (surface tension block after vorticity confinement)

### 17.4 Solver Support

- **WCSPH**: Full surface tension in Step1 (normals) + Step2 (curvature force)
- **PBF/DFSPH**: Not yet integrated into solver-specific neighbor loops

---

## 18. Particle Dye Diffusion

### 18.1 Per-Particle Dye

Each particle carries a `float4` dye color `(r, g, b, unused)`. Initialized from
material base color on spawn. Stored in `particle_dye` / `sorted_particle_dye` arrays.

### 18.2 SPH Laplacian Diffusion (Step1)

Dye diffusion rate computed in Step1 neighbor loop for FLUID particles:
```
dC/dt_i = D * SUM_j (m_j / rho_j) * (C_j - C_i) * lap_W_visc(r_ij)
```

Where:
- D = 0.01 (diffusion coefficient, hardcoded)
- `lap_W_visc = (h - |r|)` (variable part of viscosity Laplacian kernel)
- `C_i`, `C_j` are dye colors (RGB) of particles i, j

Output: `dye_rate_out[i] = float4(dR/dt, dG/dt, dB/dt, 0)`

**Source**: `step1.cu` (neighbor loop, FLUID branch)

### 18.3 Integration (Integrate)

Dye updated in the integrate kernel:
```
dye_new = dye + dye_rate * dt
dye_final = clamp(dye_new, 0, 1)     [per-component clamp]
```

**Source**: `integrate.cu` (dye update block)

### 18.4 Sort Pipeline

Dye arrays flow through the counting sort / fused reorder pipeline:
```
particle_dye → [sort scatter] → sorted_particle_dye → [Step1 reads] → sorted_dye_rate
sorted_particle_dye + sorted_dye_rate → [Integrate] → particle_dye (unsorted writeback)
```

PBF and DFSPH Finalize kernels pass `sorted_particle_dye` through to `particle_dye_out`
via their writeback paths (STATIC, sleeping, and main return paths all copy dye).

### 18.5 Solver Support

- **WCSPH**: Full diffusion (Step1 computes rate, Integrate applies update)
- **PBF**: Dye preserved through Finalize writeback (no diffusion computation)
- **DFSPH**: Dye preserved through Finalize writeback (no diffusion computation)

**Source**: `step1.cu`, `integrate.cu`, `pbf_solver.cu:K_PBF_Finalize`,
`dfsph_solver.cu:K_DFSPH_Finalize`, `counting_sort.cu` / `fused_sort_reorder_build.cu`

---

## 19. Secondary Particles: Foam / Spray / Bubbles

Lightweight secondary particle system for visual effects. Foam particles are
separate from the main SPH system — no neighbor loops, no density computation.
Generated from FLUID particles and simulated with simple ballistic/advection physics.

### 19.1 Foam Pool

Separate GPU buffers (not part of the SPH particle arrays):
- `foam_position` (float4): xyz position + type in .w
- `foam_velocity` (float4): xyz velocity + remaining lifetime in .w
- `foam_count` (uint32): atomic counter for active foam particles
- Double-buffered for stream compaction (`foam_position_b`, `foam_velocity_b`)

Maximum pool size: 200,000 particles (configurable via `MAX_FOAM_PARTICLES`).

### 19.2 Foam Types

Encoded in `foam_position.w`:
| Type   | Value | Physics |
|--------|-------|---------|
| SPRAY  | 0.0   | Gravity + air drag (ballistic) |
| FOAM   | 1.0   | Weak gravity + strong horizontal damping |
| BUBBLE | 2.0   | Buoyancy + gentle drag |

### 19.3 Generation Criteria (K_FoamGenerate)

Per FLUID particle, after integrate:
```
trapped_air = |v_i|                          (speed → splash indicator)
wave_crest  = max(0, -dot(v_i, n_hat))      (moving away from fluid bulk)
kinetic     = 0.5 * |v_i|^2

phi = k_ta * trapped_air + k_wc * wave_crest + k_ke * kinetic
```

Generation occurs when `phi > threshold` AND `neighbor_count < 25` (surface proximity check).

Default parameters: `k_ta=0.5`, `k_wc=1.0`, `k_ke=0.2`, `threshold=2.0`

Type selection:
- SPRAY: `v_y > 0.5` and `speed > 1.0` (fast upward motion)
- FOAM: `neighbor_count < 15` (surface particle with few neighbors)
- BUBBLE: otherwise (near-surface interior)

Spawn position: parent position + random jitter (±0.01).
Spawn velocity: parent velocity × scale (1.2 for spray, 0.5 for others).
Lifetime: type-dependent base × random(0.5, 1.5).

### 19.4 Physics (K_FoamPhysics)

Simple per-particle physics, no neighbor interaction:

**SPRAY**: `a = -v * drag + gravity`
- `drag_coeff = 2.0`
- Full gravity

**FOAM**: `a = (-vx*2, gravity*0.1, -vz*2)`
- Strong horizontal damping (sits on surface)
- Weak gravity (10% of full)

**BUBBLE**: `a = -v*drag + (0, buoyancy, 0)`
- `buoyancy = 15.0` (rises toward surface)
- Half drag on vertical axis

Integration: symplectic Euler. World boundary clamp kills particles at floor.

### 19.5 Compaction (K_FoamCompact)

Stream compaction every 8 frames (amortized cost):
1. Atomic scatter alive particles to output buffer
2. Swap input/output buffers
3. Device-to-device copy of alive_count → foam_count (no GPU→CPU sync)

### 19.6 Rendering

Foam rendered as separate draw call after main scene (both point sprite and SSFR modes):
- Additive blending (`GL_ONE, GL_ONE`)
- No depth writes (`glDepthMask(GL_FALSE)`)
- Small white point sprites with soft circular falloff
- Alpha varies by type: SPRAY=0.8, FOAM=0.5, BUBBLE=0.3

**Source**: `foam.cu`, `foam.py`, `world.py` (foam pool), `renderer.py` (foam draw),
`simulation.py:_run_foam_step`

## 20. GPU Data Optimizations

### 20.1 Pack Density into position.w (OPT-4.1)

After Step1 computes density, `K_PackDensity` writes `density[i]` into
`sorted_position[i].w`. Step2 then reads `rho_j = position[j].w` instead of
issuing a separate `__ldg(&density[j])` load per neighbor. This eliminates one
global memory load per neighbor interaction in the Step2 inner loop.

Pipeline order: Step1 → K_PackDensity → Step2

Applied to WCSPH path only. PBF and DFSPH use their own density pipelines
(PBF computes density from predicted positions; DFSPH reads density from a
separate buffer across multiple correction kernels).

**Source**: `step1.cu:K_PackDensity`, `step2.cu:K_Step2` (reads `pos4_j.w`),
`step1.py:pack_density()`, `simulation.py:_run_wcsph_body`

### 20.2 Speculative ILP (OPT-4.2)

In all neighbor loop inner loops, `__ldg()` loads for neighbor data (position,
packed_info, velocity, mass, density, kappa) are issued BEFORE the distance
check. All loads go through the texture cache with ~200 cycle L2 latency.
By issuing them in parallel before the branch, the loads overlap and latency
is hidden. Approximately 30% of loads are wasted (out-of-range neighbors),
but the latency savings on in-range neighbors more than compensate.

Pattern:
```cuda
for (uint j = start; j < end_idx; j++) {
    // Issue ALL loads before distance check
    float4 pos4_j = __ldg(&position[j]);
    uint pi_j = __ldg(&packed_info[j]);
    float4 vel4_j = __ldg(&velocity[j]);
    float m_j = __ldg(&mass[j]);
    // ... distance check follows ...
}
```

Applied to: `step1.cu` (K_Step1), `step2.cu` (K_Step2 + vorticity loop),
`pbf_solver.cu` (K_PBF_ComputeLambda, K_PBF_ComputeDelta),
`dfsph_solver.cu` (NonPressureForces, ComputeKappaV, CorrectVelocityDiv,
ComputeKappaFromVelocity, ComputePressureAccel, DensitySolverUpdate,
CorrectVelocityDens)

## 10.13 Micropolar SPH (Phase 8, item 5.2)

Per-particle angular velocity tracks local rotational flow. Coupled to vorticity
via relaxation (no additional neighbor loop needed).

### Angular velocity update (integrate.cu, FLUID only)

```
omega_new = omega + dt * nu_t * (0.5 * curl_v - omega)
```

Where:
- `omega` = per-particle angular velocity (float4: wx, wy, wz, unused), persistent
- `curl_v` = vorticity from step1 (sorted_vorticity.xyz)
- `nu_t` = micropolar coupling viscosity (default 0.1, in solver_profiles.py)
- `dt` = simulation timestep

The formula relaxes angular velocity toward half the fluid vorticity:
- When `omega = 0.5 * curl_v`: equilibrium (no change)
- When `omega < 0.5 * curl_v`: angular velocity increases (spin-up)
- When `omega > 0.5 * curl_v`: angular velocity decreases (dissipation)

### Pipeline integration

- **Sorting**: `angular_velocity` scattered through counting sort (K_ScatterReorder/K_GatherReorder)
- **Computation**: Relaxation update in K_Integrate (WCSPH), passthrough in K_PBF_Finalize and K_DFSPH_Finalize
- **Persistence**: Written back to unsorted `angular_velocity` via sort_indexes, compacted with other arrays
- **Active**: Only for FLUID behavior class (GRANULAR/GAS/STATIC skip micropolar)
- **Buffer**: `world.angular_velocity` (N, 4) float32, `world.sorted_angular_velocity` (N, 4) float32

### Typical values (WCSPH Dam Break, 253K particles, 200 steps)

- max |omega| ≈ 0.23 rad/s (at splash zone)
- mean |omega| ≈ 0.003 rad/s (bulk average)
- Spatially coherent (std/mean ≈ 2.8)

**Source**: `integrate.cu:550-560`, `solver_profiles.py:43`, `counting_sort.cu` (K_ScatterReorder, K_GatherReorder)
