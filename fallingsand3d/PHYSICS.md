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
- **STATIC neighbors included** in density (Akinci boundary model, Phase 13)
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
| OIL | 2125.0 | 400.0 | 7.0 | Tait |
| ACID | 2500.0 | 450.0 | 7.0 | Tait |
| LAVA | 6500.0 | 800.0 | 7.0 | Tait |
| SAND | 4000.0 | 5000.0 | 7.0 | Tait |
| DIRT | 3750.0 | 3000.0 | 7.0 | Tait |

**Source**: `sph_shared.cuh:compute_pressure`

### 3.2 Pressure Force (SPH momentum equation)

**STATIC neighbors included** in force loop (Akinci boundary: pressure mirroring `2*p_i/rho_i^2` + viscous drag with vel_j=0).

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

**IMPORTANT: FLUID vs GRANULAR use different viscosity formulations (design choice).**

**FLUID/GAS: Kinematic viscosity form (divides by rho_j only)**
```
mu_ij = 2 * mu_i * mu_j / (mu_i + mu_j)     [harmonic mean of base_viscosity]
a_visc += mu_ij * [45/(pi*h^6)] * m_j * (v_j - v_i) / rho_j * (h - |r|)
```

This is NOT the standard SPH dynamic-viscosity form (which would divide by `rho_i * rho_j`).
The `base_viscosity` values (WATER=1.0, OIL=5.0, LAVA=100.0, ACID=2.0) are tuned as
kinematic-like coefficients, not physical dynamic viscosities in Pa*s.

This choice is coupled with `force_scale=0.02` which globally scales all FLUID SPH
forces. Changing to the dynamic form `/ (rho_i * rho_j)` requires simultaneously
retuning `force_scale` and all per-material `base_viscosity` values.

**GRANULAR-GRANULAR: Dynamic viscosity form (divides by rho_i * rho_j)**
```
eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j)     [harmonic mean of mu(I) viscosity]
a_visc += eta_ij * [45/(pi*h^6)] * m_j * (v_j - v_i) / (rho_j * rho_i) * (h - |r|)
```

GRANULAR uses the standard dynamic form because `eta_i` from mu(I) rheology (see Section 5)
has true Pa*s units and `force_scale=1.0` (no global scaling).

**Source**: `step2.cu:348-357` (FLUID), `step2.cu:321-340` (GRANULAR), `dfsph_solver.cu:419-425`

### 3.4 XSPH Velocity Correction (FLUID only)

```
xsph_sum_i = SUM_j (m_j / rho_avg_ij) * (v_j - v_i) * W_poly6(|r_ij|)

where rho_avg_ij = 0.5 * (rho_i + rho_j)

veleval_out_i = v_i + epsilon * xsph_sum_i     [FLUID only]
GRANULAR/GAS:  veleval_out_i = v_i             [plain velocity, no XSPH correction]
```

`epsilon = 0.8` (from `c_granular.xsph_epsilon`)

**FLUID only**: XSPH accumulation in the inner loop is guarded by `is_fluid_i` (bd-mzc.34).
GRANULAR position is advected with `vel_new` in `integrate.cu`, not `veleval_xsph`, so
computing XSPH for GRANULAR would produce a correction that is immediately discarded.
Non-FLUID particles write plain velocity to `veleval_out`.

**Source**: `step2.cu:370-381` (accumulation, FLUID guard), `step2.cu:448-461` (output)

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

Clamped: `|accel| <= 200 m/s^2`

**Source**: `integrate.cu:196-206`

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

where `d = particle_spacing = 0.02`, `p_eff = max(p, rho0 * |g| * particle_spacing)`

**Note**: `p_eff` uses Tait EOS pressure (from `compute_pressure()`), not the
lithostatic pressure `p = rho * g * depth` assumed by the original mu(I) model.
At h=0.04 with Tait gamma=7, Tait pressure is much higher than lithostatic for
typical grain depths, making the inertial number I artificially large. This pushes
mu(I) toward mu_2/mu_max everywhere, effectively making friction coefficient
nearly constant. The practical effect is minor since mu(I) rheology mainly
controls flow initiation, and the simulation uses explicit Drucker-Prager friction
as the primary granular mechanism.

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
f_visc += eta_ij * [45/(pi*h^6)] * m_j * (v_j - v_i) / (rho_j * rho_i) * (h - |r|)
```

This uses the **dynamic viscosity form** dividing by `rho_i * rho_j` (both densities),
consistent with §3.3 which states the same. `eta_i` from mu(I) has Pa*s units and
`force_scale=1.0` for GRANULAR (no global scaling), so the full dynamic form is correct.

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
| gravity | -4.0 | -9.8 | Reduced for miniature-world aesthetic |
| dt | adaptive | 0.001 (fixed) | CFL: min(acoustic, viscous) ∈ [1e-5, 0.001]; DT_MAX=0.001 |
| force_scale | 0.02 | N/A | Matches parent's `output * m_j` convention |
| step2 output | accel * force_scale | accel * m_j | Both effectively multiply by 0.02 |
| integrate | accel = sph + g | accel = sph + g | Both treat step2 output as acceleration |
| position update | FLUID: pos += veleval_xsph * dt; others: pos += vel_new * dt | pos += vel_new * dt | FLUID uses XSPH for smooth advection |
| boundaries | impulse SDF | penalty springs | Fundamentally different |
| velocity_limit | 10 | 200 | Tight clamp prevents particles escaping grid cells |
| accel_max | 200 | N/A | Prevents shockwave-level accelerations (FLUID + GRANULAR) |
| boundary_stiffness | N/A | 20000 | Only in parent |
| boundary_dampening | N/A | 256 | Only in parent |
| restitution | 0.3 | N/A | Only in fallingsand3d |
| wall_friction | 0.5 | N/A | Only in fallingsand3d |
| MAX_RIGID_BODIES | 8 | N/A | Dynamic Akinci bodies (global memory) |
| MAX_SDF_OBJECTS | 16 | N/A | Analytical SDF collision primitives (constant memory) |
| rigid lin_damp | 1 - 0.01*dt | N/A | Linear velocity damping |
| rigid ang_damp | 1 - 0.05*dt | N/A | Angular velocity damping |
| rigid omega_max | 20 rad/s | N/A | Angular velocity clamp |
| rigid F_max | 1000*mass | N/A | Force magnitude clamp |

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
- `dTdt` from Step1 heat diffusion (see §9.1)
- `COOL_RATE = 0.02` (ambient cooling toward T_AMBIENT, reduced from 0.1 to allow
  cross-material heating to dominate)
- `T_AMBIENT = 293 K`

STATIC particles still conduct heat (but don't move).

**Source**: `integrate.cu:45-46`, `step1.cu:166-187`

### 9.1 Cross-Material Heat Boost

SPH heat diffusion uses the particle's `thermal_conductivity` (`kappa_i`),
divided by `rho_i * cp_i` for physically correct thermal diffusivity:
```
dTdt = [kappa_i / (rho_i * cp_i)] * [45/(pi*h^6)] * SUM_j (m_j/rho_j) * (T_j - T_i) * (h - |r|)
```

Where `cp_i = c_materials[mat_id].heat_capacity` and `rho_i` is the computed SPH density.
This means `thermal_conductivity` values must be tuned for the `rho*cp` divisor to produce
visible heat exchange rates (see updated values in materials.py).

For cross-material pairs, a `heat_boost` multiplier from the interaction table
accelerates heat transfer:
```
heat_boost = max(1.0, c_interactions[mat_i][mat_j].heat_exchange)
dTdt += ... * (T_j - T_i) * lap_var * heat_boost
```

Same-material pairs (heat_exchange=0) get boost=1.0 (unchanged).
Cross-material pairs get boost=heat_exchange (e.g., WATER+LAVA=50.0).

This is bidirectional: `T_j - T_i` can be positive or negative, so lava
near water both cools the lava AND heats the water. This enables visible
phase transitions: lava→stone (T<900K), water→steam (T>373K).

### 9.2 Key Heat Exchange Values

| Pair | heat_exchange | Effect |
|------|---------------|--------|
| WATER+LAVA | 50.0 | Rapid cooling/boiling at contact |
| LAVA+ICE | 60.0 | Fastest melting |
| WATER+FIRE | 30.0 | Fire extinguished, water heated |
| LAVA+WOOD | 40.0 | Wood ignition |
| WATER+ICE | 20.0 | Melting/freezing |
| FIRE+ICE | 25.0 | Moderate melting |

**Source**: `materials.py:351-384`, `step1.cu:174-180`

### 9.3 Material Default Spawn Temperatures

Most particles spawn at `T_AMBIENT = 293 K`. Several materials use non-ambient defaults
(set in `_DEFAULT_TEMPS` in `world.py`):

| Material | Default spawn temp | Notes |
|----------|-------------------|-------|
| LAVA | 1500 K | Above solidification threshold |
| FIRE | 1200 K | Active combustion temperature |
| STEAM | 373 K | At boiling point |
| SMOKE | 500 K | Hot exhaust |
| ICE | 253 K | **20 K below the 273 K melt threshold (bd-mzc.18)** |

ICE spawns at 253 K so it does not immediately melt on frame 1 when the ambient
cooling term (293 K environment) would otherwise push a freshly-spawned 293 K ICE
particle above its `temp_melt=273 K` threshold in the first substep.

**Source**: `world.py:_DEFAULT_TEMPS`

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

### 10.6 FIXED: Akinci-style one-way STATIC boundary enforcement
Previously, STATIC particles (STONE, METAL, ICE) were skipped in all neighbor
loops, making them invisible to SPH — only the world-box SDF provided containment.
Fluids passed through stone walls in presets like Volcano and Waterfall.

**Phase 13 fix**: Akinci-style one-way boundary (Akinci et al., 2012):
1. **STATIC neighbors contribute to density** in all 3 solvers (step1, PBF lambda,
   DFSPH density_alpha). This fixes density deficiency near walls.
2. **WCSPH step2**: Pressure mirroring `p_boundary = 2 * p_i / rho_i^2` and
   viscous boundary friction (drag toward zero, STATIC has vel=0). XSPH skipped.
3. **PBF**: STATIC lambda_j=0, so delta naturally pushes fluid away from walls.
   XSPH includes STATIC (vel_j=0 → boundary friction).
4. **DFSPH**: STATIC kappa=0, p_rho2=0, vel=0 set by early returns. Correction
   kernels include STATIC neighbors for one-sided corrections.
5. **Dye diffusion**: STATIC neighbors skipped to prevent color bleeding from stone.
6. **GAS phase separation preserved**: GAS particles still skip STATIC neighbors.

Additional FLUID boundary improvements (unchanged):
- **XSPH position advection for FLUID**: Smooths compression artifacts.
- **Zero wall friction for FLUID**: Prevents domain-wall sticking.
- **All FLUID pressure clamped >= 0**: No tensile/negative pressure.

**Source**: step1.cu, step2.cu, pbf_solver.cu, dfsph_solver.cu

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
4. **ACCEL_MAX 200 → 200 (FLUID+GRANULAR)**: Both use 200 for dramatic dynamics.
5. **VELOCITY_LIMIT**: CFL-derived per solver (factor*h/dt), factor=0.9 for WCSPH/PBF,
   0.4 for DFSPH. Replaces hardcoded 10. DFSPH uses lower limit to prevent Jacobi
   solver oscillation at boundaries.
6. **mu0 0.1 → 1.0**: Higher viscous damping prevents oscillations.
7. **XSPH epsilon 0.5 → 0.1**: Per-material viscosity now dominates velocity smoothing.
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

### 10.13 Spawn velocity damping (PBF/DFSPH stabilization)

PBF and DFSPH aggressively correct incompressibility, which causes violent outward
expansion when particles are spawned on a cubic lattice (surface particles have ~50%
density deficiency due to fewer neighbors).

**Fix**: A `velocity_damping` field in SimParams applies `vel *= (1 - damping)` in
all three finalize/integrate kernels (K_Integrate, K_PBF_Finalize, K_DFSPH_Finalize).

Ramp schedule (Python-side, in simulation.py):
- Duration: 30 substeps after preset load or solver switch
- Linear ramp: `damping = 0.8 * (1 - t/30)` → 0.8 at step 0, 0.0 at step 30
- CUDA graph invalidated during ramp (constant memory changes each step)
- Graph captures normally at step 31 onward

**Source**: `common.cuh` (SimParams), `integrate.cu`, `pbf_solver.cu`,
`dfsph_solver.cu`, `simulation.py`

### 10.14 FIXED: GPT 5.2 Pro Physics Review (2026-02-27)

External algorithmic review by GPT 5.2 Pro identified 9 issues. Fixes applied:

1. **DFSPH granular compliance inverted** (CRITICAL → FIXED): `denom * 0.7f` made alpha
   *larger* (stiffer). Fixed to `alpha * 0.7f` so alpha gets smaller (more compliant).
2. **PBF double Boussinesq buoyancy** (HIGH → FIXED): Thermal convection was applied in
   both K_PBF_Predict and K_PBF_Finalize. Removed from Finalize (PBF derives velocity
   from corrected positions, so external forces belong only in prediction).
3. **PBF STATIC lambda non-zero on do_heat path** (HIGH → FIXED): When STATIC fell through
   for heat/exposure accumulation, it computed a non-zero lambda. Now force lambda=0
   after the neighbor loop for STATIC (Akinci one-way boundary convention).
4. **PBF GAS doesn't skip constraints** (HIGH → FIXED): GAS was going through full
   lambda+delta computation. Now early-returns with lambda=0, delta=0 in both
   ComputeLambda and ComputeDelta.
5. **PBF s_corr_n hardcoded to 4** (MEDIUM): Documented in code. At h=0.04, s_corr is
   disabled (k=0) so this has no practical impact.
6. **DFSPH K_DFSPH_ComputeKappaFromVelocity uses m_j not m_j/rho_j** (HIGH if used):
   Legacy kernel, not called from Python. Marked as LEGACY/UNUSED.
7. **DFSPH K_DFSPH_ComputeDensityAdv uses stale grid** (MEDIUM): Legacy kernel, not
   called from Python. Marked as LEGACY/UNUSED.
8. **W_poly6 can go negative from FP rounding** (LOW → HARDENED): Added `fmaxf(diff, 0)`
   clamp for safety.
9. **EOS powf overflow at extreme densities** (MEDIUM → HARDENED): Added `ratio` cap at
   10.0 to prevent NaN/Inf from extreme density ratios.

**Source**: `sph_shared.cuh`, `pbf_solver.cu`, `dfsph_solver.cu`

### 10.14b FIXED: Physics review fixes (PROBLEMS.md batch)

Comprehensive fixes from GPT/Gemini cross-review of all physics kernels:

**Material properties (Phase 1):**
- B1: Density scaling to force_scale=0.02 convention (OIL=2125, LAVA=6500, etc.)
- B2: EOS stiffness rebalance for denser materials (LAVA=800, STONE=1500, etc.)
- C1: XSPH epsilon 0.8→0.1 (stable), 0.05 (fast) to let per-material viscosity dominate
- C2: ACCEL_MAX_FLUID 30→200 for dramatic splashes/explosions
- E1: Gravity 9.8→4.0 for miniature-world aesthetic

**Reaction system (Phase 2):**
- J1: Acid corrosion switched from exclusion to inclusion whitelist
- J2: Drying math inversion fixed (division→multiplication)
- J3: WATER added to REACTIVE_MATERIAL_IDS
- J6: LAVA solidify temp uses c_materials[MAT_LAVA].temp_melt instead of hardcoded 900K
- D1: Steam condense hysteresis (373→360K, 13K band below boiling)
- D2: STONE→LAVA and SAND→LAVA melting reactions added
- D3: Fire→Smoke on lifetime expiry (smoke trails with 3s lifetime)
- D5: Explosion speed 5→25
- D6: Missing interaction pairs added (DIRT+WATER, LAVA+SAND/DIRT/GRAVEL)
- D7: Acid consumption (health -= 0.5 * damage dealt)

**Kernel physics (Phase 3):**
- G1: Heat diffusion corrected: dTdt = kappa/(rho*cp) * lap(T). thermal_conductivity
  values retuned for rho*cp divisor.
- G2: WCSPH mu(I) granular viscosity: added 1/rho_i divisor. mu_max 10000→25000.
- G3: Vorticity gradient uses (omega_j - omega_i) instead of omega_j
- H1: DFSPH STATIC density overridden to rest_density (prevents boundary gaps)
- H2: PBF XSPH coefficient clamped to 0.5 max (prevents instability with LAVA)
- H3: PBF Drucker-Prager friction applied to velocity correction (not absolute velocity)

**Integration pipeline (Phase 4):**
- I2: CFL-derived velocity limit (v_max = factor*h/dt, factor per solver) replaces hardcoded VELOCITY_LIMIT=10
- I4: frame_counter incremented per-frame (not per-substep); substep_counter for RNG
- I5: BOUNDARY_MARGIN=1e-4 prevents particles sitting exactly at wall positions

**Design decisions (Phase 6, intentionally kept):**
- I1: XSPH pre-integration lag (standard Euler, <0.008s lag)
- I3: Grid reuse: WCSPH (0.25*h threshold, 4-frame cap), PBF/DFSPH (0.15*h threshold, 2-frame cap)
- F1: force_scale=0.02 (co-adapted WCSPH convention)
- F2: WCSPH viscosity convention (force_scale absorbs 1/rho for FLUID)
- F3: Tait EOS for mu(I) (lithostatic needs expensive column height)
- G4: Surface tension is "surface normal cohesion" (not full Akinci model)
- J10: Sorted array transitions handled by per-particle branching (safe invariant)

**Reaction fixes (2026-06-10 batch, bd-mzc.27):**
- DIRT + corrosion-exposure (driven by WATER contact reaction_rate) now converts DIRT to
  MUD (wetting transition), not corrosion damage. This check runs before the acid
  corrosion whitelist so DIRT is redirected to the wetting path rather than being
  "corroded" by water.
- DIRT was removed from the acid-corrosion whitelist. The corrosion path (`health -= damage`)
  now applies only to: STONE, METAL, WOOD, GRAVEL, ICE, OIL, GUNPOWDER. DIRT reaches
  MUD via the wetting path (`exp_corrode > SAND_WET_THRESHOLD`) instead.

**Source**: `reactions.cu:225-261`

### 10.15 FIXED: Rigid body force/acceleration unit mismatch

Step2 accumulates **acceleration** (not force) into `d_rigid_forces[]` because the
step2 output is acceleration for integrate.cu. But `K_IntegrateRigidBodies` applies
`F * inv_mass`, giving jerk (m/s³) instead of acceleration.

**Fix**: Before accumulating into `d_rigid_forces`, multiply by `m_i` (fluid particle
mass) to convert acceleration to force:
```
a_on_fluid = (pressure + viscous) * force_scale
F_on_body = -m_i * a_on_fluid          // Newton's 3rd law, in force units
```
This ensures the rigid body integrator correctly computes `a = F/M_body`.

**DFSPH non-pressure viscous reaction force (bd-mzc.23)**: The DFSPH
`K_DFSPH_NonPressureForces` viscous boundary coupling previously used
`m_j / rho_i` as the mass factor when accumulating the reaction force, making
it scale with neighbor mass rather than the fluid particle's own mass. Fixed to
`F_on_body = -F_visc * particle_mass` (multiply viscous acceleration by `m_i =
c_sim.particle_mass`), consistent with the WCSPH convention.

**Source**: `step2.cu:K_Step2` (rigid body force accumulation block),
`dfsph_solver.cu:K_DFSPH_NonPressureForces:486-494` (viscous reaction fix)

### 10.16 FIXED: Surface tension sign inversion (repulsion instead of cohesion)

The curvature-based surface tension force had a sign error:
```
f_st = -gamma * n_i     [WRONG: pushes outward, repels surface particles]
f_st = +gamma * n_i     [CORRECT: pushes inward, cohesion]
```

Sign chain analysis: `spiky_grad_coeff` is **negative** (-45/π/h⁶), so the
accumulated surface normal `n_i = SUM (m_j/rho_j) * grad_W` points **into the
bulk** (toward higher density). Multiplying by `+gamma` gives inward force
(cohesion). The previous `-gamma` reversed this to outward repulsion.

All three solvers now apply `+gamma * n_i` (bd-mzc.21):
- **WCSPH** (`step2.cu`): `f_surface_tension = gamma * norm_i` (acceleration)
- **DFSPH** (`dfsph_solver.cu:K_DFSPH_NonPressureForces:613-615`): `accel += +gamma * norm_i`
- **PBF** (`pbf_solver.cu:K_PBF_Finalize:902-904`): `vel_new += dt * (+gamma * norm_i)`

Note: PBF uses `gamma=0.0` by default (disabled; relies on weak attractive density
constraint instead), but the sign convention in the code is correct.

**Source**: `step2.cu` (surface tension block), `dfsph_solver.cu:K_DFSPH_NonPressureForces`,
`pbf_solver.cu:K_PBF_Finalize`

### 10.17 FIXED: GAS density floor clamped above rest density

The density output floor `fmaxf(density, 1.0f)` was applied uniformly to all
behaviors. GAS materials have `rest_density` 0.2–0.6, so this floor prevented
GAS from ever reaching its natural rest density, keeping it permanently
"compressed" with artificially high pressure.

**Fix**: Behavior-aware density floor:
```
float rho_floor = is_gas_i ? RHO_EPSILON : 1.0f;
density_out[i] = fmaxf(density, rho_floor);
```
GAS uses `RHO_EPSILON` (0.01) as a minimal safety floor while allowing natural
density values. FLUID/GRANULAR keep the 1.0 floor (rest densities 500–2500).

Additionally, all denominator guards `fmaxf(rho_j, 1.0f)` across all solvers
were changed to `fmaxf(rho_j, RHO_EPSILON)` for consistency with GAS densities.

**Source**: `sph_shared.cuh` (RHO_EPSILON), `step1.cu`, `step2.cu`,
`dfsph_solver.cu`, `pbf_solver.cu`, `implicit_st.cu`

### 10.18 Rigid Body System Limitations

- **Bounding sphere collisions only**: Body-body and body-SDF collision uses bounding sphere (max of half_extents), not true shape. Elongated bodies leave gaps at corners.
- **No rigid-rigid stacking**: Single push-apart pass per substep; deep piles of bodies may interpenetrate.
- **Max 8 dynamic bodies, 16 SDF objects**: Hardcoded limits in constant memory (`MAX_RIGID_BODIES`, `MAX_SDF_OBJECTS`).
- **CUDA graph disabled**: Rigid body forces vary per substep, so the CUDA graph pipeline is disabled when any rigid body is present.
- **No body-frame friction/rolling**: Only linear velocity is reflected in collision; angular velocity is not affected by rigid-rigid contacts.

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

### 11.1 Cross-Solver Feature Parity

All three solvers now support the complete material behavior set:

| Feature | WCSPH | PBF | DFSPH |
|---------|-------|-----|-------|
| Pressure solve | Tait EOS | Density constraint | Div+Dens iters |
| Viscosity (FLUID) | Laplacian (per-material) | XSPH (per-material scaled) | Laplacian (per-material)/rho_i |
| Viscosity (GRANULAR) | mu(I) in Step2 | XSPH*10 + DP friction | mu(I) in NonPressure |
| XSPH smoothing | Step2, FLUID only (eps=0.8) | Finalize (c=0.05) | NonPressure (eps=0.5) |
| GAS buoyancy | Integrate | Predict (v+=) | NonPressure (accel) |
| GAS phase separation | Step1 (skip GAS) | ComputeLambda (skip GAS) | DensityAlpha (skip GAS) |
| Heat diffusion | Step1 | ComputeLambda | DensityAlpha |
| Cross-material heat | Step1 (heat_boost) | ComputeLambda (heat_boost) | DensityAlpha (heat_boost) |
| Exposure (fire/acid) | Step1 | ComputeLambda | DensityAlpha |
| Dye diffusion | Step1 | ComputeLambda | DensityAlpha |
| Vorticity computation | Step1 | ComputeLambda | DensityAlpha |
| Vorticity confinement | Step2 (eps=0.05) | Finalize (eps=0.001) | NonPressure (eps=0.05) |
| Surface normals | Step1 | ComputeLambda | DensityAlpha |
| Surface tension | Step2 (gamma=1.0) | Finalize (gamma=0.0) | NonPressure (gamma=1.0) |
| Micropolar angular vel | Integrate | Finalize | Finalize |
| Drucker-Prager friction | Step2 (force-space) | Finalize (vel-space) | NonPressure (force-space) |
| Sleep/wake system | Integrate | Finalize | Finalize |
| Temperature/reactions | Integrate | Finalize | Finalize |

**Key solver-specific differences:**
- PBF vorticity epsilon is 50x lower (0.001 vs 0.05) to prevent energy injection at fixed dt
- PBF surface tension is disabled (gamma=0) — relies on weak attractive constraint instead
- PBF Drucker-Prager uses velocity-space friction in Finalize (once, iteration-independent)
- WCSPH uses adaptive CFL timestep; PBF/DFSPH use fixed dt

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

**Boundary (STATIC) neighbor density (bd-mzc.25)**: STATIC neighbors contribute
to the density sum with `psi_b = m_j` (standard Akinci weight — particle mass field).
A previous `2x boundary_scale` multiplier was removed; it over-inflated near-wall
density to roughly `2*rho_0`, causing lambda to drive particles away from walls
(unphysical bounce). With the 2x factor gone, near-wall density is correct and
wall contact is stable. Lambda and delta gradients also use `m_j` without scaling.

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

**Lambda persistence:** Converged lambda values are written back to unsorted
buffer in K_PBF_Finalize and carried through the sort-reorder pipeline for
warm-starting. Scaled by `pbf_warm_start` (default 0.5). Since PBF lambda is
computed analytically (not iteratively), warm-starting serves as infrastructure
for future iterative PBF variants.

### 12.5 XSPH Viscosity (PBF)

```
v_new += c_scaled * SUM_j (m_j / rho_avg_ij) * (v_j - v_i) * W_poly6(x*_ij)

c_scaled = c_pbf.xsph_c * c_materials[mat_id].base_viscosity
```

`c_pbf.xsph_c` (default 0.05) is scaled by per-material `base_viscosity` so that
viscous fluids (OIL=5.0, LAVA=10.0) get proportionally more XSPH smoothing than
WATER (1.0). GRANULAR additionally multiplies by 10x for strong artificial viscosity.
Applied in K_PBF_Finalize.

**Source**: `pbf_solver.cu:K_PBF_Finalize`

### 12.5 PBF Multi-Material Handling

- **FLUID**: Full PBF constraint solve (density constraint + XSPH)
- **GRANULAR**: Participates in density constraint (lambda, dx) for volume exclusion.
  Drucker-Prager friction is applied in K_PBF_Finalize (velocity-space, once per
  substep) using pressure normals from K_PBF_ComputeLambda (see §12.7).
  High artificial viscosity prevents granular from flowing like water.
- **GAS**: Skips PBF constraints (lambda=0, delta=0) in all kernels.
  Predict uses gravity + buoyancy + drag. GAS is a compressible phase.
- **STATIC**: Lambda always forced to 0 (Akinci one-way boundary). STATIC
  contributes to neighbor density sums. On do_heat path, exposure/heat/dye
  are accumulated but lambda is still 0.

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

#### 12.7.2 Velocity-Space Drucker-Prager Friction (K_PBF_Finalize)

After PBF constraint iterations, friction is applied once in velocity space
(iteration-independent). The final velocity `v_new = (x* - x_old) / dt` is
decomposed along the pressure normal:

```
n = pressure_normal.xyz         (unit normal from ComputeLambda, points into pile)
v_dot_n = dot(v_new, n)

// Only apply friction when compressing (v_dot_n > 0).
// Separating particles (v_dot_n <= 0) are in tension -- no friction.
if v_dot_n > 0:
    v_n = v_dot_n * n           (normal component: into pile)
    v_t = v_new - v_n           (tangential component: sliding)

    max_tang = tan(phi_f) * v_dot_n + cohesion / dt

    if |v_t| > max_tang:
        v_t = v_t * (max_tang / |v_t|)

    v_new = v_n + v_t
```

The expansion guard (`v_dot_n > 0` check) prevents friction from freezing
particles during free flight or avalanche separation. Without it, `fabsf(v_dot_n)`
would apply friction symmetrically, zeroing tangential velocity on expansion.

#### 12.7.3 Parameters

Parameters are read from `c_granular` constant memory (uploaded per-solver with
different values for PBF vs WCSPH):

| Parameter | PBF Default | WCSPH Default | Description |
|-----------|-------------|---------------|-------------|
| `tan_phi_f` | 0.50 | 0.781 (= tan(38°)) | Friction cone slope |
| `cohesion` | 0.0 | 0.002 | Cohesive tangential allowance |

**Why PBF uses velocity-space friction**: Position-space friction (per-iteration)
causes iteration-dependent behavior -- changing from 4 to 12 iterations would apply
friction 3x more. Velocity-space friction applied once in Finalize gives consistent
behavior regardless of iteration count (7.8% variation across 4/6/8/12 iters).

A static friction dead zone (`v_dot_n < 5e-4`) zeroes all tangential velocity
when the normal compression is negligible, preventing numerical creep.

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
  5. Divergence-free solver (warm-started kappa_v from prev frame, scaled by 0.5):
     Apply warm-started kappa_v correction, then N_div iterations:
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
  8. Boundary handling, color, writeback (kappa + kappa_v written back for warm-start)
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

**Warm-starting:** kappa_v values are persisted across substeps through the sort-reorder
pipeline (kappa_v → sorted_kappa_v). At the start of each divergence solve, the
warm-started values are scaled by `dfsph_div_warm_start` (default 0.5) and an initial
velocity correction is applied before the iteration loop. Written back to unsorted
buffer in K_DFSPH_Finalize.

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
v_total_i = clamp(v_total_i, VELOCITY_LIMIT)   [bd-mzc.33: prevent Jacobi runaway]
drho = SUM_j (m_j/rho_j) * (v_total_i - v_total_j) . grad_W_ij
density_adv = rho_i/rho_0 + dt * drho/rho_0    [bd-mzc.22: both terms normalized by rho_0]
residual = density_adv - 1.0             [positive = compressed]
p_rho2 = max(p_rho2 + omega * residual * alpha / dt^2, 0)
```

**Density advection normalization (bd-mzc.22)**: Both terms in `density_adv` are divided
by `rho_0`, making the sum dimensionless (`rho_i/rho_0 + dt*drho/rho_0`). The `drho/rho_0`
normalization was previously missing, causing the density-change term to be ~rho_0 times
too large and the solver to converge to a fixed point with a persistent 20-40% density error.

**Velocity clamping before density prediction (bd-mzc.33)**: `v_total = v + dt*a_press` is
clamped to `VELOCITY_LIMIT` before computing `drho`. Without this clamp, a large `a_press`
overshoot in one Jacobi iteration can produce a huge `drho` that drives the next iteration
further out — runaway Jacobi oscillation. The clamp prevents this feedback from diverging.

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
(denom ≈ 915), but alpha is capped at `alpha_limit * dt^2` in K_DFSPH_ComputeDensityAlpha.
The `alpha_limit` parameter (default 1.0) controls the maximum correction per iteration:
```
alpha_cap = alpha_limit * dt^2
correction factor k = omega * alpha_cap / dt^2 = omega * alpha_limit
```
With alpha_limit=1.0, dt=1/300: alpha_cap ≈ 1.1e-5, k ≈ 1.0 (conservative).
Higher alpha_limit allows faster convergence: alpha_limit=10.0 reduces density error
from 35% to 23% at 8 iterations (300 steps, 36K particles).
Without any cap, k ≈ 90 (explosive overshoot because actual alpha is 100x larger).

**Alpha sweep results** (dt=1/300, omega=1.0, warm_start=0.5, 8 dens iters, 36K particles, 300 steps):

| alpha_limit | omega=0.3 | omega=0.5 | omega=0.7 | omega=1.0 |
|-------------|-----------|-----------|-----------|-----------|
| 0.5         | 41.2%     | 39.1%     | 37.7%     | 36.7%     |
| 1.0         | 38.2%     | 36.8%     | 35.8%     | 35.0%     |
| 2.0         | 36.2%     | 35.2%     | 33.9%     | 33.5%     |
| 5.0         | 34.2%     | 33.0%     | 33.0%     | 27.7%     |
| 10.0        | 32.9%     | 27.9%     | 23.3%     | 22.6%     |

Best density error: alpha_limit=10.0, omega=1.0 → 22.6% (35% improvement over baseline).
However, omega=1.0 causes persistent velocity oscillation (v_max 7-9 m/s at steady state)
due to Jacobi overcorrection of boundary/surface particles.

**Stability fix (2026-03-08)**: omega reduced to 0.7 (under-relaxation), velocity limit
lowered to 0.4*h/dt (was 0.9*h/dt), and pressure acceleration clamped to h/dt^2 in
K_DFSPH_ApplyPressureVelocity. Result: v_max drops from 7-9 to 1.4 at 5s steady state,
KE matches WCSPH (1.2 vs 1.6). Density error slightly higher (~23→~26%) but visually stable.

Default stable profile: 12 iterations, alpha_limit=10.0, omega=0.7, warm_start=0.5,
velocity_limit_factor=0.4 at dt=1/300.
Default fast profile: 8 iterations, alpha_limit=5.0, omega=0.7, velocity_limit_factor=0.4.

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
eta_i = SUM_j (m_j / rho_j) * (omega_j - omega_i) * grad_W_spiky(r_ij)
N_i = eta_i / |eta_i|                          [normalized vorticity gradient]
f_conf = epsilon * (N_i × omega_i)              [confinement force]
```

Where `epsilon = c_granular.vorticity_epsilon` (default 0.05).

The gradient uses the **difference form** `(omega_j - omega_i)` (not `omega_j`
alone). This is the SPH approximation of the gradient of a scalar field, which
vanishes identically when the vorticity field is uniform -- preventing spurious
confinement forces in solid-body rotation. The difference form is now unified
across all three solvers (WCSPH, DFSPH, PBF).

The confinement force is perpendicular to the vorticity axis and aligned with the
gradient of vorticity magnitude, amplifying existing rotational structures without
creating spurious rotation.

**Source**: `step2.cu` (second neighbor loop after main force computation)

### 16.3 Solver Support

- **WCSPH**: Full vorticity computation in Step1 + confinement in Step2 (epsilon=0.05)
- **PBF**: Vorticity computed in K_PBF_ComputeLambda, confinement applied in K_PBF_Finalize
  as velocity correction `v += dt * eps * (N x omega)`. PBF epsilon=0.001 (50x lower than
  WCSPH) because PBF's fixed dt=1/120 lacks CFL safety; higher values inject energy.
- **DFSPH**: Vorticity computed in K_DFSPH_ComputeDensityAlpha, confinement applied in
  K_DFSPH_NonPressureForces as acceleration. Uses same epsilon as WCSPH (0.05).

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

For surface particles (neighbor_count < 25), the Akinci curvature model:
```
f_st = +gamma * n_i     [cohesion force toward bulk]
```

Where `gamma = c_granular.surface_tension_gamma` (default 1.0).

**Sign convention**: The surface normal `n_i = SUM (m_j/rho_j) * grad_W_spiky`
points **into the bulk** (toward higher density) because `spiky_grad_coeff` is
negative (-45/π/h⁶). Multiplying by `+gamma` gives an inward (cohesive) force
that drives surface particles toward the fluid interior, minimizing surface area.

**Note**: The full Akinci model also includes a cohesion kernel C(r) for particle-pair
attraction. Currently only the curvature term is implemented — it provides the dominant
surface-minimizing behavior for our particle spacing (h=0.04, dx=0.02).

**Source**: `step2.cu` (surface tension block after vorticity confinement)

### 17.4 Solver Support

- **WCSPH**: Full surface tension in Step1 (normals) + Step2 (curvature force), gamma=1.0
- **PBF**: Normals computed in K_PBF_ComputeLambda, force applied in K_PBF_Finalize
  as velocity correction `v += dt * (-gamma * n_i)`. PBF gamma=0.0 (disabled) because
  PBF's fixed dt causes energy injection. PBF relies on weak attractive constraint (5%
  negative C) for cohesion instead.
- **DFSPH**: Normals computed in K_DFSPH_ComputeDensityAlpha, force applied in
  K_DFSPH_NonPressureForces as acceleration. Uses gamma=1.0 (same as WCSPH).

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
- **PBF**: Dye rate computed in K_PBF_ComputeLambda (first call only), applied in
  K_PBF_Finalize. Same SPH Laplacian formula as WCSPH.
- **DFSPH**: Dye rate computed in K_DFSPH_ComputeDensityAlpha, applied in
  K_DFSPH_Finalize. Same SPH Laplacian formula as WCSPH.

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

### 20.3 Pre-computed Pressure Array (PERF-007)

WCSPH Step2 no longer calls `compute_pressure()` per neighbor (~50 calls/particle).
Instead, `K_ComputePressure` runs once after Step1 to write per-particle pressure
to `sorted_pressure[]`. Step2 reads `p_i` and `p_j` via `__ldg(&pressure_in[i])`.

This eliminates the per-neighbor `powf()` call (Tait EOS) from the Step2 inner loop.

Pipeline: Step1 → K_ComputePressure → Step2

**Source**: `step2.cu:K_ComputePressure`, `step2.py:compute_pressure()`,
`simulation.py:_run_wcsph_body`

### 20.4 Warm-start kappa_v and lambda_pbf (PERF-008)

DFSPH divergence solver kappa_v and PBF lambda are now carried through the
counting sort pipeline (K_ScatterReorder, K_GatherReorder), persisted via
finalize kernel writeback, and warm-started at the beginning of each substep.

- **DFSPH kappa_v**: Scaled by `dfsph_div_warm_start` (default 0.5), then applied
  as an initial velocity correction before the divergence solver iterations.
- **PBF lambda**: Written back through K_PBF_Finalize for future use.

**Source**: `counting_sort.cu` (kappa_v_in/out, lambda_pbf_in/out),
`dfsph_solver.cu:K_DFSPH_Finalize`, `pbf_solver.cu:K_PBF_Finalize`,
`simulation.py:_run_dfsph_body` (warm-start application)

### 20.5 Grid Reuse for PBF/DFSPH (PERF-009)

Extended the WCSPH grid-reuse optimization to PBF and DFSPH. Both finalize
kernels now track max_displacement via atomicMax. Sort-skip decision uses
solver-specific thresholds:

- **WCSPH**: (0.25*h)² threshold, max 4 consecutive skips
- **PBF/DFSPH**: (0.15*h)² threshold, max 2 consecutive skips (tighter for
  constraint solver convergence quality)

**Source**: `pbf_solver.cu:K_PBF_Finalize` (max_displacement_out),
`dfsph_solver.cu:K_DFSPH_Finalize` (max_displacement_out),
`simulation.py:step_frame` (solver-specific threshold logic)

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
- **Computation**: Relaxation update in K_Integrate (WCSPH), K_PBF_Finalize (PBF), and K_DFSPH_Finalize (DFSPH)
- **Persistence**: Written back to unsorted `angular_velocity` via sort_indexes, compacted with other arrays
- **Active**: Only for FLUID behavior class (GRANULAR/GAS/STATIC skip micropolar)
- **Buffer**: `world.angular_velocity` (N, 4) float32, `world.sorted_angular_velocity` (N, 4) float32

### Typical values (WCSPH Dam Break, 253K particles, 200 steps)

- max |omega| ≈ 0.23 rad/s (at splash zone)
- mean |omega| ≈ 0.003 rad/s (bulk average)
- Spatially coherent (std/mean ≈ 2.8)

**Source**: `integrate.cu:550-560`, `solver_profiles.py:43`, `counting_sort.cu` (K_ScatterReorder, K_GatherReorder)

## 21. Sand Wetness System (Phase 14)

Three discrete sand states driven by water exposure:

| Material | ID | Behavior | Viscosity | Friction | Cohesion | Color |
|----------|-----|----------|-----------|----------|----------|-------|
| SAND | 2 | GRANULAR | 0.5 | 0.6 | 0.0 | tan (0.76, 0.70, 0.50) |
| WET_SAND | 16 | GRANULAR | 2.0 | 0.75 | 0.15 | dark tan (0.55, 0.48, 0.30) |
| MUD | 17 | FLUID | 50.0 | 0.0 | 0.0 | dark brown (0.35, 0.22, 0.10) |

### 21.1 State Transitions (reactions.cu)

Transitions are driven by `exposure_corrode`, accumulated via the interaction
matrix reaction_rate for SAND-WATER and WET_SAND-WATER pairs:

```
SAND  --[exp_corrode > 0.2]--> WET_SAND --[exp_corrode > 0.5]--> MUD
SAND  <--[exp_corrode < 0.02]-- WET_SAND <--[exp_corrode < 0.1]-- MUD
          (drying)                          (drying)
```

Drying is accelerated by temperature above 293K:
```
effective_threshold = threshold / (1 + 0.01 * max(temp - 293, 0))
```

Hysteresis prevents oscillation (wet threshold >> dry threshold).

### 21.2 Interaction Matrix Entries

| Pair | reaction_rate | heat_exchange | Purpose |
|------|--------------|---------------|---------|
| SAND + WATER | 0.4 | 5.0 | Wetting signal |
| WET_SAND + WATER | 0.6 | 5.0 | Saturation -> MUD |
| WET_SAND + FIRE | 0.0 | 25.0 | Fire dries wet sand |
| WET_SAND + LAVA | 0.0 | 40.0 | Lava dries wet sand |
| MUD + FIRE | 0.0 | 25.0 | Fire dries mud |
| MUD + LAVA | 0.0 | 40.0 | Lava dries mud |
| MUD + WATER | 0.3 | 3.0 | Mud stays wet near water |
| MUD + ACID | 0.05 | 3.0 | Acid corrodes mud |

### 21.3 Corrosion Guard

Sand family materials (SAND, WET_SAND, MUD) are excluded from the corrosion
health damage path. Their `exposure_corrode` signal is used only for wetting
transitions, not health reduction.

### 21.4 Cross-Solver Behavior

No kernel code changes needed — existing per-material property lookups handle everything:

- **WCSPH**: step2 reads `c_materials[mat_id].friction_coeff`; WET_SAND (0.75) > SAND (0.6). MUD uses FLUID viscosity path.
- **PBF**: Drucker-Prager uses global `c_granular.tan_phi_f`. Differentiation via XSPH damping from base_viscosity.
- **DFSPH**: MUD transitions to FLUID → incompressibility corrections + high viscosity.

**Source**: `materials.py`, `reactions.cu`

---

## 22. Rigid Body System

The rigid body system provides solid-fluid interaction via two mechanisms:
- **SDF collision** (Phase A): Analytical signed distance fields for particle-object collision
- **Akinci boundary coupling** (Phase B): Boundary particles for two-way hydrodynamic forces

### 22.1 SDF Primitive Distance Functions

All primitives return signed distance (negative = inside):

**Box** (half-extents h):
```
d_box(p, h) = length(max(|p| - h, 0)) - min(max(|p.x|-h.x, |p.y|-h.y, |p.z|-h.z), 0)
```

**Sphere** (radius r):
```
d_sphere(p, r) = length(p) - r
```

**Cylinder** (radius r, half-height h, Y-axis aligned):
```
d_cylinder(p, r, h) = min(max(sqrt(p.x² + p.z²) - r, |p.y| - h), 0) + length(max(d2, 0))
where d2 = (sqrt(p.x² + p.z²) - r, |p.y| - h)
```

**Plane** (normal n, point p0):
```
d_plane(p, n, p0) = dot(p - p0, n)
```

Oriented primitives (box, cylinder) transform to local space via inverse quaternion rotation before evaluation.

**Source**: `sph_shared.cuh:sdf_box()`, `sdf_sphere()`, `sdf_cylinder()`, `sdf_plane()`

### 22.2 SDF Collision Response

Applied in `K_Integrate` for FLUID/GRANULAR/GAS particles:

```
if d(pos, obj) < 0:
    pos += normal * |d|                          // push-out
    v_n = dot(vel, normal)
    if v_n < 0:
        vel -= (1 + e) * v_n * normal            // reflect normal component
        v_t = vel - dot(vel, normal) * normal     // tangential component
        vel -= min(mu * |v_n| * (1+e), |v_t|) * normalize(v_t)  // Coulomb friction
```

Parameters per SDF object: restitution (e), friction (mu).

**Source**: `sph_shared.cuh:sdf_object_boundary()`, `integrate.cu:K_Integrate`

### 22.3 Akinci Boundary Particle Sampling

Dynamic rigid bodies use boundary particles for hydrodynamic coupling (Akinci et al. 2012).

**Volume precomputation** (psi_b): Each boundary particle's contribution volume is:
```
psi_b = rho_0 / SUM_k W(r_b - r_k, h)
```
where the sum is over neighboring boundary particles of the same body. In practice, `psi_b` is stored directly as the particle's mass field.

**Material**: Boundary particles use `MAT_RIGID = 18` with `behavior_class = STATIC`.

**Source**: `rigid_bodies.py:_compute_psi()`, `common.cuh:MAT_RIGID`

### 22.4 Akinci Density Contribution

In Step1, boundary particles contribute to fluid density like regular particles:
```
rho_i += SUM_b psi_b * W(r_i - r_b, h)
```
No special code needed — boundary particles have mass = psi_b and participate in the standard density summation. MAT_RIGID particles skip their own density computation (early return).

**Source**: `step1.cu:K_Step1`

### 22.5 Akinci Pressure Force (Two-Way Coupling)

For a fluid particle i near a MAT_RIGID boundary particle b:

```
F_pressure = -m_i * (p_i/rho_i² + p_i/rho_0²) * psi_b * grad_W(r_i - r_b)
```

The second term uses rest density rho_0 (not rho_b) because boundary particles have no meaningful density of their own. This mirrors the fluid's own pressure onto the boundary.

**Viscous coupling**: Boundary velocity computed from rigid body state:
```
v_b = v_body + omega × (r_b - COM)
F_visc = mu * (v_b - v_i) / rho_avg * lap_W
```

**Reaction forces** (Newton's 3rd law):

Step2 computes acceleration `a_on_fluid` (pressure + viscosity, scaled by force_scale).
To get proper force for rigid body integration, multiply by the fluid particle's mass:
```
F_on_body = -m_i * a_on_fluid     // convert acceleration → force, flip sign
tau_on_body = (r_b - COM) × F_on_body
```

This ensures `K_IntegrateRigidBodies` correctly computes `a_body = F_total / M_body`.

Forces accumulated via `warp_reduce_accumulate()` then `atomicAdd` to per-body accumulators.

**Source**: `step2.cu:K_Step2` (WCSPH), `dfsph_solver.cu:K_DFSPH_NonPressureForces` (DFSPH)

### 22.6 Force Accumulation with Warp-Level Reduction

To reduce atomic contention (~32x), forces are summed within each warp before a single atomic write:

```c
// Warp-level reduction (butterfly)
for (int offset = 16; offset > 0; offset >>= 1) {
    fx += __shfl_down_sync(0xFFFFFFFF, fx, offset);
    // ... fy, fz, tx, ty, tz
}
if (lane_id == 0) atomicAdd(&d_rigid_forces[body_id*4 + k], f_k);
```

**Source**: `sph_shared.cuh:warp_reduce_accumulate()`

### 22.7 Rigid Body Integration

Symplectic Euler with world-frame inertia tensor:

```
// Linear
v += (F/m + g) * dt
x += v * dt

// Angular (world-frame inertia)
I_world_inv = R * diag(I_body_inv) * R^T
alpha = I_world_inv * tau
omega += alpha * dt

// Quaternion update
q += 0.5 * dt * (0, omega) * q
q = normalize(q)
```

Velocity damping: linear 1-0.01*dt, angular 1-0.05*dt. Angular velocity clamped to 20 rad/s. Force clamped to 1000*mass.

**Box inverse inertia (bd-mzc.24)**: For a box with half-extents `(hx, hy, hz)` and mass `m`,
the body-frame principal moments use divisor **3** (not 12):
```
I_xx = m * (hy^2 + hz^2) / 3
I_yy = m * (hx^2 + hz^2) / 3
I_zz = m * (hx^2 + hy^2) / 3
```
The divisor 3 applies to half-extents; divisor 12 applies only when the formula uses
full side lengths (e.g., `l = 2*hx`). Using 12 with half-extents would underestimate
inertia by 4x, making boxes spin unrealistically fast under torques.

**Source**: `rigid_bodies.py:480-487`, `integrate.cu:K_IntegrateRigidBodies`

### 22.8 Rigid-Rigid and Rigid-SDF Collision

After integration, `K_RigidBodyCollisions` performs simple push-apart:

**Body vs SDF objects**: Uses bounding sphere radius = |half_extents|. If SDF distance < radius, push out along SDF normal and reflect velocity with restitution/friction.

**Body vs body**: Bounding sphere overlap test. Push apart weighted by inverse mass. Velocity reflection along contact normal.

**Source**: `integrate.cu:K_RigidBodyCollisions`

### 22.9 Boundary Particle State Sync

After rigid body integration, boundary particles must be updated to match the new body state:

```
r_world = COM + R * r_local          // position
v_world = v_body + omega × r_world   // velocity
```

Runs as `K_UpdateBoundaryParticles`, one thread per boundary particle.

**Source**: `integrate.cu:K_UpdateBoundaryParticles`

### 22.10 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Body state storage | Global memory (RigidBody struct) | Max 8 bodies; constant memory reserved for SDF objects |
| psi_b storage | mass field directly | Avoids extra buffer; density sum naturally includes psi_b |
| Force accumulation | Warp reduction + atomicAdd | 32x less contention than per-thread atomics |
| SDF vs boundary | Both coexist | SDF for collision, boundary particles for pressure/viscosity |
| Quaternion convention | (x,y,z,w) where w=scalar | Matches CUDA float4 layout |
| CUDA graph | Disabled when bodies present | Rigid body forces vary per substep (not capturable) |
