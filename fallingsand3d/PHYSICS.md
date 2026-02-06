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

For cubic lattice at spacing `dx = 0.02`, mass `m = 0.008`, h=0.04:
- 27 neighbors within h (3x3x3 cube, all at distance <= sqrt(3)*dx = 0.0346 < h)
- rho ≈ 1000 kg/m^3 (by construction: m = rho_0 * dx^3)

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

| Material | k | gamma | EOS |
|----------|-----|-------|------|
| WATER | 100.0 | 1.0 | Linear |
| OIL | 80.0 | 1.0 | Linear |
| ACID | 90.0 | 1.0 | Linear |
| LAVA | 30.0 | 7.0 | Tait |
| SAND | 20.0 | 7.0 | Tait |
| DIRT | 18.0 | 7.0 | Tait |

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

`epsilon = 0.5` (from `c_granular.xsph_epsilon`)

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

Clamped: `|accel| <= 5000 m/s^2`

**Source**: `integrate.cu:316-336`

### 4.2 Velocity Update

```
v_new = v + dt * accel
```

GAS drag: `v_new *= (1 - 2.0 * dt)`

Clamped: `|v| <= 50 m/s`

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

### 5.1 Strain-Rate Tensor (computed in Step1)

```
D_ij = 0.5 * SUM_k (m_k/rho_k) * (dv_i * gradW_j + dv_j * gradW_i)

gamma_dot = sqrt(2 * D:D)
          = sqrt(2 * (Dxx^2 + Dyy^2 + Dzz^2 + 2*(Dxy^2 + Dxz^2 + Dyz^2)))
```

**Source**: `step1.cu:168-236`

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

Default: mu_max=10000, mu0=3.5

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
| rho_0 (rest density) | 1000 | 1000 | Same; SPH density ≈ 2500 at rest (see §7) |
| k (EOS stiffness) | 100.0 (Water) | 3.0 | Game SPH: higher k with linear EOS |
| gamma (EOS exponent) | 1.0 (FLUID), 7 (GRANULAR) | 7 | Per-material; FLUID uses linear EOS |
| mu_0 (viscosity) | 3.5 | 3.5 | Same |
| epsilon (XSPH) | 0.5 | 0.5 | Same (computed but not used in position update) |
| gravity | -9.8 | -9.8 | Same |
| dt | adaptive | 0.001 (fixed) | CFL: min(acoustic, viscous) ∈ [1e-5, 0.005]; DT_MAX=0.005; CPU-only (no GPU sync) |
| force_scale | 0.02 | N/A | Matches parent's `output * m_j` convention |
| step2 output | accel * force_scale | accel * m_j | Both effectively multiply by 0.02 |
| integrate | accel = sph + g | accel = sph + g | Both treat step2 output as acceleration |
| position update | FLUID: pos += veleval_xsph * dt; others: pos += vel_new * dt | pos += vel_new * dt | FLUID uses XSPH for smooth advection |
| boundaries | impulse SDF | penalty springs | Fundamentally different |
| velocity_limit | 50 | 200 | Different |
| accel_max | 5000 | N/A | Only in fallingsand3d |
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

### 7.3 Why mass = 0.02 (not rho0 * dx^3 = 0.008)

With m = rho0 * dx^3 = 0.008, the SPH density at rest ≈ 1000 (physical).
With m = 0.02, the SPH density at rest ≈ 2500 (inflated, ~2.5x rho0).

The Tait EOS pressure depends on rho/rho0:
- m=0.008: rho/rho0 ≈ 1.0 → p ≈ 0 (near zero pressure at rest)
- m=0.02:  rho/rho0 ≈ 2.5 → p = k * (2.5^7 - 1) = 3 * 609 = 1828

The inflated density keeps the EOS in a stable high-pressure regime where
small density changes produce large pressure gradients (strong restoring force).
With the "correct" mass (0.008), k would need to be ~128,000 to achieve the
same stiffness, requiring a much smaller timestep for stability.

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

### 10.6 FIXED: Game SPH overhaul (performance + visual quality)
Water physics had two problems: (1) DT_MAX=0.001 forced ~17 substeps/frame due to
stiff Tait EOS (gamma=7), and (2) STATIC particles inflated density at interfaces,
creating pressure spikes and wall-sticking.

Changes:
1. **Skip STATIC neighbors** in step1 density/heat/exposure and step2 force loops.
   Prevents wall density inflation. Self-interaction (i==j) preserved for density.
2. **Per-material EOS gamma**: gamma=1 → linear EOS for FLUID (Water, Oil, Acid).
   gamma=7 → Tait EOS for GRANULAR (Sand, Dirt, Lava). Removes `powf()` for FLUID.
3. **FLUID EOS stiffness increased**: Water k=100, Oil k=80, Acid k=90 (was 3-10).
   Higher k compensates for linear EOS being less stiff than Tait at high compression.
4. **DT_MAX 0.001 → 0.005**: Linear EOS has constant speed of sound, CFL allows
   larger timesteps. Expected 3-5 substeps/frame instead of 17.
5. **XSPH position advection for FLUID**: Smooths compression artifacts at larger dt.
6. **Zero wall friction for FLUID**: Prevents domain-wall sticking.
7. **All FLUID pressure clamped >= 0**: No tensile/negative pressure.

**Source**: step1.cu, step2.cu, integrate.cu, materials.py, simulation.py

### 10.7 OPTIMIZATION: Fused sort-reorder-build pipeline
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

### 10.8 FIXED: GPU→CPU sync every frame in adaptive timestep
`_compute_adaptive_dt()` used `float(cp.max(...))` on GPU velocity arrays to
compute advection CFL, forcing a GPU→CPU sync (device→host transfer) every frame.
With current parameters, the acoustic CFL (dt≈0.05) always exceeds DT_MAX (0.005)
by 10x, and the velocity-based advection CFL was never binding (velocity_limit=50
and accel_max=5000 prevent supersonic particles). Removed the advection CFL
entirely; adaptive dt now uses only acoustic and viscous CFL, both precomputed
from the materials table at init time — zero GPU sync in the hot path.
**Source**: `simulation.py:151-175`

### 10.9 OPTIMIZATION: CUDA graph capture for SPH pipeline
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
