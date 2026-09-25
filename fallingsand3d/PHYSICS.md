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
- Heat diffusion and exposure skip MAT_RIGID neighbors (exposure still counts
  them in its normalising weight, see §9.4). This holds in the cross-phase
  branch too: a GAS particle gets no exposure from a MAT_RIGID neighbour.
- The MAT_RIGID early-out writes density 2500 into both `density_out` and
  `position.w`, like every other particle. `K_UpdateBoundaryParticles` and
  `inject_boundary_particles` also write `.w = 2500`, so no reader sees 0.

**Source**: `step1.cu:138-150` (inner loop), `step1.cu:224-225` (PostCalc)

### 2.2 Parent comparison

Parent uses constant mass pulled OUTSIDE the loop:
```
rho_i = m * C_poly6 * SUM_j [ (h^2 - |r_ij|^2)^3 ]
```

Both are mathematically equivalent when all particles have the same mass.

**Source**: parent `step1.cu:131-143`

### 2.3 Expected density at rest

For cubic lattice at spacing `dx = 0.02`, h=0.04:
- 26 neighbours strictly inside h (32 at distance <= h); a settled pile has about 27-35
- SPH kernel-sum density ≈ 1.01 * 2500 * (m / 0.02) kg/m³ (Poly6 at h=0.04)
- FLUID and GRANULAR particles spawn with `m = 0.02 * rest_density / 2500`
  (`world.spawn_mass`), so every such material starts at rho/rho0 ≈ 1.01 (see 7.3)
- STATIC and GAS particles spawn with `m = 0.02`

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

| Material | rho_0 | k | gamma | EOS | spawn mass |
|----------|-------|-----|-------|------|-----------|
| WATER | 2500.0 | 500.0 | 7.0 | Tait | 0.020 |
| OIL | 2125.0 | 400.0 | 7.0 | Tait | 0.017 |
| ACID | 2500.0 | 450.0 | 7.0 | Tait | 0.020 |
| LAVA | 6500.0 | 1300.0 | 7.0 | Tait | 0.052 |
| SAND | 4000.0 | 5000.0 | 7.0 | Tait | 0.032 |
| DIRT | 3750.0 | 3000.0 | 7.0 | Tait | 0.030 |
| GRAVEL | 4250.0 | 4000.0 | 7.0 | Tait | 0.034 |
| WET_SAND, MUD, GUNPOWDER | 4500.0 | 5000 / 300 / 3000 | 7.0 | Tait | 0.036 |
| STEAM | 1200.0 | 3.0 | 1.0 | GAS | 0.020 (0.02/3 boiling child) |
| SMOKE | 1100.0 | 2.5 | 1.0 | GAS | 0.020 |
| FIRE | 1000.0 | 4.0 | 1.0 | GAS | 0.020 |

LAVA k = 500 * 6500 / 2500 gives it WATER's sound speed `sqrt(gamma * k / rho_0)` (1.18),
so under the same head both compress by the same fraction. The GRANULAR k values are
unchanged: at the spawn mass a settled 0.4 m WCSPH pile sits 2-4% above rho_0 with
v_rms about 0.01 m/s.

GAS `rho_0` is in kernel-sum units, not real gas density (bd-r5-review-n28.80). GAS
particles keep mass 0.02 (K_SpawnGas children 0.02/3), so the Poly6 self term alone is
`m * 315 / (64 pi h^3)` = 490 (163 for a child) at h = 0.04. Rule: `rho_0` for a gas is
the kernel-sum density of a mass-0.02 cubic lattice at the target gas spacing, and at least
2x the 490 self term: STEAM 1200 (spacing about 0.025), SMOKE 1100 (0.026), FIRE 1000
(0.027). Consequences:
- An isolated gas particle of either mass has `rho < rho_0` and `p = 0`.
- A cloud at the spawn spacing (0.02 brush, or three children per boiling water particle)
  reads `rho` about 2525, so `rho/rho_0` is 2.1-2.5 and pressure pushes it apart until it
  thins to `rho_0`.
- `k` and `rho_0` for gas now shape the cloud. In a WCSPH eval a 1000-particle STEAM cube
  rising for 1.2 s keeps every particle in contact (xz RMS radius 0.081 to 0.128 m, median
  `rho/rho_0` 0.58); with a real-world `rho_0` of 0.6 all particles ended isolated and the
  radius reached 0.225 m. The steam from 500 boiling water particles behaves the same way
  (radius 0.120 m, median `rho/rho_0` 0.53, no isolated particles).
- `simulation._compute_c_sound` reads `sqrt(k * gamma / rho_0)`, which for gas is now
  about 0.06, so SAND (2.96) sets the acoustic bound. The gas EOS sound speed is
  `sqrt(k)`, 1.6-2.0, which stays below it. DT_MAX (1 ms) binds either way.

**Source**: `sph_shared.cuh:compute_pressure`, `materials.py` (GAS rest_density)

### 3.2 Pressure Force (SPH momentum equation)

**STATIC neighbors included** in force loop (Akinci boundary: pressure mirroring `2*p_i/rho_i^2` + viscous drag with vel_j=0).

The two boundary types use different pressure mirrors:

- Regular STATIC wall: `m_j * 2*p_i/rho_i^2 * grad_W`.
- MAT_RIGID body particle: `psi_b * (p_i/rho_i^2 + p_i/rho0_i^2) * grad_W`, with
  `psi_b = rho0_i * V_b` (22.3, 22.5).

They agree only at `rho_i = rho0_i`. In the WCSPH Archimedes Pool the water sits
at about 2.1 rho0 (median `rho/rho0` 2.11 after 8 s), where the rigid form is
about 2.5x stronger. Switching rigid bodies to the wall form was measured and
kept out (bd-r5-review-n28.106): over 8 s the medium (4.3 kg) body sank 4 cm
deeper and the heavy (12 kg) body 2 cm deeper, and water inside the medium and
heavy boxes went from 1 and 90 particles to 153 and 298. Unifying the two needs
the WCSPH over-compression fixed first, or both boundary types moved to the
rigid form and walls retuned.

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

**Output scale per pair (bd-r5-review-n28.41).** Step2 multiplies particle i's sums by
`fs_i` (1.0 for GRANULAR, `force_scale` = 0.02 for FLUID and GAS). A same-behaviour pair
therefore gets the same scale on both sides. A FLUID-GRANULAR pair uses one scale on both
sides:
```
fs_cross = min(fs_FLUID, fs_GRANULAR) = force_scale
a_pressure_i += fs_cross * pressure_precalc * m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_var
```
The kernel multiplies the cross term by `fs_cross / fs_i` in the loop, so the final `fs_i`
leaves exactly `fs_cross`. Then `m_i a_i = -m_j a_j` for every pair, and a water-sand
interface conserves momentum. `min` (not the geometric mean 0.141) is used so that sand
under water feels the same pressure acceleration as the water it displaces: at hydrostatic
balance its upward push is about |g|, not 7x or 50x |g|
(`test_cross_force_scale.py::test_submerged_sand_not_ejected`, ratio sand/water = 1.0).

**Source**: `step2.cu:315-321` (accumulation), `step2.cu:379-390` (PostCalc)

### 3.3 Viscosity Force

All pair terms use the dynamic form `/ (rho_i * rho_j)`, so each pair is symmetric in
i and j and conserves momentum even where the densities differ (free surface).

**FLUID/GAS: dynamic viscosity form, same as DFSPH (bd-r5-review-n28.79)**
```
mu_ij = 2 * mu_i * mu_j / (mu_i + mu_j)     [harmonic mean of base_viscosity, Pa*s]
a_visc += mu_ij * [45/(pi*h^6)] * m_j * (v_j - v_i) / (rho_i * rho_j) * (h - |r|)
```
`force_scale` does not apply: the kernel divides the term by `fs_i` before the final
multiply. WCSPH and DFSPH (13.5) read the same `base_viscosity` and use this same pair
formula, so the kinematic viscosity is `nu = mu / rho0` in both. WATER: 1.0 / 2500 =
4e-4 m^2/s. OIL 5.0 / 2125 = 2.4e-3, LAVA 100 / 6500 = 0.015, MUD 50 / 4500 = 0.011,
ACID 2.0 / 2500 = 8e-4. GAS (mu 0.01, rho0 0.2 to 0.6) gets 0.017 to 0.05.

The `base_viscosity` values were not changed. The old WCSPH form divided by `rho_j`
only and was scaled by `force_scale`, which gave WATER nu = 0.02 m^2/s (50x DFSPH) and
F_ij / F_ji = rho_i / rho_j. WCSPH liquids are therefore thinner than before and now
match DFSPH. PBF uses `base_viscosity` only as its XSPH factor (12.5), so it is not
affected.

Dam Break check (18000 WATER particles, column x in [-0.95, -0.55], y in [-0.95, -0.35],
z in [-0.3, 0.3], world [-1, 1], front = 99th percentile of x):

| t | WCSPH front | DFSPH front | difference |
|---|---|---|---|
| 0.25 s | -0.568 | -0.514 | 0.054 m |
| 0.50 s | -0.285 | -0.148 | 0.137 m |
| 1.00 s | +0.597 | +0.694 to +0.717 | 0.10 to 0.12 m |

With WATER mu scaled to 50 in both solvers (the old WCSPH strength) the fronts at 1 s
were +0.435 (WCSPH) and +0.428 (DFSPH). The remaining gap at mu = 1 comes from the
pressure solve, not from viscosity. WCSPH at mu = 1 shows single splash particles up to
8 m/s near 1 s (DFSPH 3.3 m/s); sum of v^2/2 still decays from 11100 at 1 s to 1200 at 4 s.

**FLUID-GRANULAR pairs: one dynamic form on both sides (bd-r5-review-n28.41)**
```
mu_F    = base_viscosity_F                             [FLUID dynamic viscosity, WATER = 1.0]
mu_G    = mu (step1 `viscosity`, 1.0 in the app)       [same value as viscosity_precalc]
mu_pair = 2 * mu_F * mu_G / (mu_F + mu_G)              [WATER-SAND: 1.0]
a_visc += mu_pair * [45/(pi*h^6)] * m_j * (v_j - v_i) / (rho_i * rho_j) * (h - |r|)
```
F is whichever particle of the pair is FLUID. This term already carries its scale, so the
kernel divides it by `fs_i` before the final multiply. It is symmetric in i and j, so
viscous forces between water and sand are equal and opposite.

**GRANULAR-GRANULAR: Dynamic viscosity form (divides by rho_i * rho_j)**
```
eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j)     [harmonic mean of mu(I) viscosity]
a_visc += eta_ij * [45/(pi*h^6)] * m_j * (v_j - v_i) / (rho_j * rho_i) * (h - |r|)
```

GRANULAR uses the standard dynamic form because `eta_i` from mu(I) rheology (see Section 5)
has true Pa*s units and `force_scale=1.0` (no global scaling).

**STATIC and MAT_RIGID boundary neighbours (WCSPH step2, `boundary_visc_factor`)**

The boundary particle b has mass (or psi_b) m_b and velocity v_b (0 for regular STATIC,
the body surface velocity for MAT_RIGID).
```
FLUID/GAS i:  a_visc += base_viscosity_i * [45/(pi*h^6)] * m_b * (v_b - v_i) / (rho_i * rho0_i) * (h - |r|)
GRANULAR i:   a_visc += mu0 * [45/(pi*h^6)] * m_b * (v_b - v_i) / (rho_i * rho0_i) * (h - |r|)
```
The FLUID/GAS term, like the FLUID pair, is divided by `fs_i` before the final multiply.
K_Step2_NL uses the same helper. Both behaviours use the same dynamic form as their pair code, with the boundary density taken as
the rest density `rho0_i` of i's own material (Akinci mirror: the wall stands in for more
sand). It does not read the STATIC particle's own density. The drag rate per boundary
neighbour therefore equals the FLUID-GRANULAR pair drag at rest density rescaled from
mu_pair to mu0 (checked by `test_step2.py::test_granular_static_drag_matches_pair` and
`test_granular_rigid_drag_matches_pair`). At h=0.04, m=0.02, rho0_sand=4000 and
r=h/2 this is about 0.12 1/s per neighbour, well inside the explicit limit and far
below `ACCEL_MAX_GRANULAR = 200`.

**Implicit viscous self-term (WCSPH step2, bd-r5-review-n28.52)**

Every term above has the shape `c_j * (v_j - v_i)`. Step2 sums the coefficients that
multiply `-v_i` over all neighbours, including STATIC and MAT_RIGID ones:
```
c_tot   = sum_j c_j
x       = dt * fs_i * c_tot          [fs_i = force_scale for FLUID/GAS, 1 for GRANULAR]
a_visc  = fs_i * sum_j c_j (v_j - v_i) / (1 + x)
```
This is backward Euler on `v_i` in the viscous term:
`v' = (v + dt * fs_i * sum_j c_j v_j) / (1 + x)`. Explicit Euler needs `x < 2`. For
an interior lattice particle at rho = 2500 and dt = 1 ms, x is about 0.375 for LAVA
(base_viscosity 100), 0.19 for MUD (50), 0.019 for OIL, 0.0038 for WATER, and about 1.9
for resting SAND at `mu_max = 500`. At their own rest densities LAVA is 0.14 and MUD
0.10. With the older FLUID form (before bd-r5-review-n28.79) x was 50x larger: 18.75 for
LAVA, 9.4 for MUD. Before the implicit scale, LAVA and MUD velocities overshot the
neighbour mean by up to 18x per substep, held back only by `ACCEL_MAX`. That left a
velocity checkerboard in resting lava and made resting sand flip sign relative to its
neighbours every substep.

With the scale, a mode that alternates between neighbours changes by
`1 - lambda * x / (1 + x)` per substep, where `lambda * c_tot` is the mode's Laplacian
eigenvalue (at most 2 c_tot). This is always between -1 and 1. On the lattice
(`test_implicit_viscosity.py`, rho = 2500 for every material) one substep gives
v'/v = +0.73 (3D checkerboard) and +0.71 (alternating planes) for LAVA, +0.84 and +0.83
for MUD, +0.98 for OIL, +0.996 for WATER, +0.32 and +0.27 for SAND.

Side effects:
- Smooth flows lose some viscous drag: about x/(1+x) of the accel on the stiffest
  mode, under 2% for WATER and OIL and about 27% for LAVA at rho = 2500.
  Dam Break WATER (21504 particles, dt 1 ms) peak KE went from 872 to 878 J
  (+0.7%) and KE at 1 s from 256 to 265 J (+3.4%).
- A falling LAVA block used to be held back by the clamped checkerboard. It now falls
  under gravity like any fluid and only resists deformation: `eval_solvers.py`
  `b1_lava_slow` mean speed after 200 steps went from 0.12 to 0.57 m/s.
- MAT_RIGID: the viscous reaction on the first rigid body a particle touches is added
  after the loop with the same `1 / (1 + x)` scale, so action equals reaction. A second
  body in the same particle's neighbourhood stays explicit (not in `c_tot`, reaction
  added at once).
- DFSPH `NonPressureForces` and PBF have no implicit scale.

**FP16 pairing rule (bd-r5-review-n28.69)**

Neighbour velocities are read from the FP16 mirror `velocity_h` (OPT-4.3). Every pair
difference `v_j - v_i` (viscosity and XSPH in step2; vorticity and strain rate in step1)
also takes `v_i` from `velocity_h`, and the step1 dye difference takes `dye_i` from
`dye_h`. So two particles with the same true velocity give exactly zero difference.
Step2 keeps the FP32 `v_i` for `veleval = v_i + eps * xsph` and for the STATIC and
MAT_RIGID boundary terms, whose other side (`v_b`) is FP32. When `velocity_h` is NULL
both sides are FP32.

With FP32 `v_i` against FP16 `v_j`, every pair gave `fp16(v) - v`, a half-ulp of up to
4.9e-4 m/s at |v| in [1, 2), summed into a body force toward the FP16 grid. A 10^3
lattice moving at (1.3, -0.7, 2.1) m/s got up to 3.3e-3 extra force per particle for
WATER and 0.18 for LAVA, and vorticity up to 0.012 1/s. It now gets exactly zero.
Dam Break KE (WCSPH, 0.25 to 1.5 s) is unchanged within 1%.

Tests: `test_step2.py::test_translating_block_no_fp16_viscous_bias`,
`test_step1.py::test_translating_block_zero_vorticity_strain_dye` (K_Step1 and
K_Step1_BuildNL). `neighbor_list.cu` K_Step1_NL and K_Step2_NL still use FP32 `v_i`.

Tests: `test_visc_dynamic_form.py` (pair and free-surface block momentum at round-off,
pair value equals the formula above, K_Step2 and K_Step2_NL), `test_implicit_viscosity.py`.

**Source**: `step2.cu` K_Step2 FLUID/GAS pair branch, GRANULAR branch, `cross_pair_visc`,
`boundary_visc_factor`, implicit scale before PostCalc; `neighbor_list.cu` K_Step2_NL
(same formulas and scale); `dfsph_solver.cu` K_DFSPH_NonPressureForces

### 3.4 XSPH Velocity Correction (FLUID only)

```
xsph_sum_i = SUM_j (m_j / rho_avg_ij) * (v_j - v_i) * W_poly6(|r_ij|)   [j FLUID only]

where rho_avg_ij = 0.5 * (rho_i + rho_j)

veleval_out_i = v_i + epsilon * xsph_sum_i     [FLUID only]
GRANULAR/GAS:  veleval_out_i = v_i             [plain velocity, no XSPH correction]
```

`epsilon` = `c_granular.xsph_epsilon`, built from the active WCSPH profile's
`xsph_epsilon` (with `force_scale` and `mu0`) at init and on every switch to a WCSPH
profile (`Simulation._wcsph_granular_params`). Both WCSPH profiles use 0.8, the same as
`step2.DEFAULT_XSPH_EPSILON`. XSPH only changes the FLUID advection velocity, not the
stored velocity. Dam Break (21.5K WATER, 5 s) with 0.1 in place of 0.8 gave the same
KE decay (final/peak 0.003 to 0.004) and v_max at 5 s of 1.1 to 1.5 m/s for both values
at dt 1 ms and 0.5 ms (bd-r5-review-n28.25), so the value was left at 0.8.

**FLUID only**: XSPH accumulation in the inner loop is guarded by `is_fluid_i` (bd-mzc.34).
`integrate.cu` adds the XSPH delta `veleval - v` only for FLUID. GRANULAR is advected
with plain `vel_new`, so computing XSPH for GRANULAR would produce a correction that is
immediately discarded.
Non-FLUID particles write plain velocity to `veleval_out`.

**FLUID neighbours only**: the sum runs over FLUID neighbours only (`behavior_j == FLUID`,
bd-r5-review-n28.104). STATIC and GAS neighbours are skipped earlier in the loop.
GRANULAR neighbours are skipped too: a settled sand bed is asleep at v = 0, so counting it
pulled water's advection velocity toward 0 while its stored velocity stayed, and water
crawled over sand. Water over sand or stone now advects at its own velocity plus the
average over nearby water. Test: `test_step2.py::test_xsph_skips_granular_neighbour`.
`neighbor_list.cu` K_Step2_NL still sums over every non-STATIC neighbour.

**Source**: `step2.cu` K_Step2 XSPH block in the neighbour loop (accumulation, FLUID guards),
`veleval_out` write after the loop (output)

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

For GAS (ideal-gas buoyancy along the live gravity vector `g = c_sim.gravity`):
```
accel += -g * (T / T_amb + buoyancy_extra)
net   = -g * ((T - T_amb) / T_amb + buoyancy_extra)     [with the +g already in accel]
```
- `T_amb = GAS_AMBIENT_TEMP = 293 K`; `buoyancy_extra` is per material
  (STEAM 0.01, SMOKE 0.005, FIRE 0.02).
- Archimedes for a gas parcel in ambient air: the air pushes back with
  `-g * rho_air / rho_gas = -g * T / T_amb`. Gas at `T_amb` is weightless apart
  from `buoyancy_extra`; hot gas moves against gravity, whatever its sign or direction.
- Examples at `g = (0, -4, 0)`: STEAM at 373 K nets +1.13 m/s^2; FIRE at 1200 K
  nets +12.5 m/s^2, and with `GAS_DRAG_COEFF = 2/s` its terminal speed is about
  6.2 m/s (well under the velocity clamp). STEAM rises at `g = -9.81` too.
- Same formula in all three solvers: `gas_buoyancy_accel()` in `sph_shared.cuh`,
  used by `K_Integrate` (WCSPH), `K_PBF_Predict` and `K_DFSPH_NonPressureForces`.

For FLUID with `thermal_expansion > 0`: Boussinesq term `-g * beta * (T - T_amb)`
(see section 14).

Clamped: `|accel| <= 200 m/s^2`

**Source**: `integrate.cu` (K_Integrate acceleration block), `sph_shared.cuh:gas_buoyancy_accel`

### 4.2 Velocity Update

```
v_new = v + dt * accel
```

GAS drag: `v_new *= (1 - 2.0 * dt)`

Clamped: `|v| <= velocity_limit = velocity_limit_factor * h / dt`, recomputed in
`_upload_dt` whenever dt changes. With h = 0.04:

| Solver | factor | dt | velocity_limit |
|--------|--------|----|----------------|
| WCSPH (Fast), accuracy 0.4 | 0.9 | 1 ms | 36 m/s |
| WCSPH, accuracy 0.2 | 0.9 | 0.5 ms | 72 m/s |
| PBF | 0.9 | 1/120 s | 4.3 m/s |
| DFSPH | 0.4 | 1/300 s | 4.8 m/s |
| DFSPH (Fast) | 0.4 | 1/200 s | 3.2 m/s |

**WCSPH timestep** (`Simulation._compute_adaptive_dt`, adaptive mode only):

```
dt_accuracy = DT_MAX * accuracy / 0.4
dt_acoustic = 0.25 * h / c_sound
dt = clamp(min(dt_accuracy, dt_acoustic), DT_MIN = 1e-5, DT_MAX = 1e-3)
```

dt_acoustic is about 2.2 ms (FIRE sets c_sound = 4.47 m/s), so dt_accuracy is the
bound that acts: the Accuracy slider and the `[` `]` keys give 0.25 ms at 0.1, 0.5 ms
at 0.2 and 1 ms at 0.4. Values above 0.4 are capped by DT_MAX. There is no viscous CFL:
step2 treats the viscous self-term implicitly (see 3.3 and 10.4).

GRANULAR anti-creep: if `|v| < 0.01` AND `rho > 0.95*rho_0` AND `gamma_dot < 0.05`:
set `v_new = 0`

**Source**: `integrate.cu:338-376`

### 4.3 Position Update

```
FLUID:   pos_new = pos + dt * (v_new + (veleval_xsph - v_old))
                 = pos + dt * (v_new + epsilon * xsph_sum)
Others:  pos_new = pos + dt * v_new
```

All behaviors use symplectic Euler: the position moves with `v_new`, after gravity,
SPH forces, the velocity clamp, spawn damping, GAS drag and anti-creep. FLUID adds the
XSPH correction on top. Step2 writes `veleval_xsph = v_old + epsilon * xsph_sum`, so
`veleval_xsph - v_old` is exactly the XSPH delta and no extra buffer is needed. This
smooths compression artifacts at larger dt (Game SPH). A FLUID and a GRANULAR particle
in free fall stay at the same height (`test_fluid_granular_freefall_match`,
bd-r5-review-n28.44).

PBF and DFSPH do their own position update in their Finalize kernels (PBF from the
predicted position, DFSPH from the post-solve velocity plus XSPH), so they do not read
`sorted_veleval` for advection.

**Source**: `integrate.cu:279-294`

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
All three solvers use this rule: WCSPH in `K_Integrate`, PBF in `K_PBF_Finalize`,
DFSPH in `K_DFSPH_Finalize`. PBF uses its own copy, `pbf_box_boundary`, where a
particle exactly on the margin plane counts as contact (see 12.4).

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

**Note**: under WCSPH, `p_eff` uses Tait EOS pressure (from `compute_pressure()`),
not the
lithostatic pressure `p = rho * g * depth` assumed by the original mu(I) model.
At h=0.04 with Tait gamma=7, Tait pressure is much higher than lithostatic for
typical grain depths, making the inertial number I artificially large. This pushes
mu(I) toward mu_2/mu_max everywhere, effectively making friction coefficient
nearly constant. The practical effect is minor since mu(I) rheology mainly
controls flow initiation, and the simulation uses explicit Drucker-Prager friction
as the primary granular mechanism. Under DFSPH, `p` is the density solver's
own pressure instead (13.5).

### 5.3 Friction Coefficient

```
mu(I) = mu_s + (mu_2 - mu_s) / (1 + I0 / I)
```

Globals: mu_s=0.55, mu_2=0.85, I0=0.3.

`mu_s` is per material: WCSPH Step2 uses `c_materials[mat].friction_coeff`, or the
global `mu_s` when that is 0. The gap `mu_2 - mu_s = 0.30` stays global, so each
material's `mu_2 = friction_coeff + 0.30`. Each particle's `eta` uses its own
material, and the pair uses the harmonic mean of the two (5.5). Table values:
DIRT 0.5, GUNPOWDER 0.5, SAND 0.6, GRAVEL 0.65, WET_SAND 0.75 (bd-r5-review-n28.89).
DFSPH and the neighbour-list Step2 still use the global `mu_s`.

Measured effect (WCSPH, 0.2 x 0.6 x 0.2 m column on the floor, 2 s, pile height
over 95th-percentile radius): DIRT 0.102, SAND 0.119, WET_SAND 0.121, GRAVEL 0.131.
With the global mu_s all four were 0.111 to 0.114. The effect is small because
resting sand sits at the `mu_max` cap (5.4), where `mu_s` has no effect.

### 5.4 Effective Viscosity

```
eta = min(mu_max, mu0 + mu(I) * p_eff / max(gamma_dot, 0.1))
```

Default: mu_max=500, mu0=1.0

The strain-rate floor `MUI_GAMMA_DOT_FLOOR = 0.1 1/s` keeps eta finite and linear
in `p_eff` in slow flow, instead of growing as `1 / gamma_dot`. The cap acts when
`gamma_dot < mu(I) * p_eff / mu_max`, about 0.54 1/s at the SAND pressure floor
(490 Pa), so resting sand sits at `mu_max`. At rest the explicit viscous factor
would be `dt * mu_max * 15 / (h^2 * rho) = 1.88` at dt 1 ms, rho 2500, h 0.04:
under 2 but sign-alternating. Step2 treats the viscous self-term implicitly
(3.3, 10.4), so the relaxation is monotone at any `mu_max` or dt and no viscous
CFL term is needed for `mu_max`.

**Source**: `sph_shared.cuh:compute_muI_eta`, `step2.py:DEFAULT_MU_MAX`

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
| m (particle mass) | 0.02 * rho_0 / 2500 (FLUID, GRANULAR); 0.02 (STATIC, GAS) | 0.02 | WATER 0.02, SAND 0.032, LAVA 0.052, OIL 0.017 (see 7.3) |
| dx (particle spacing) | 0.02 | ~0.02 | Same |
| rho_0 (rest density) | 2500 (WATER), per material | 1000 | Each material rests at its own rho_0 at spacing 0.02 (see 7.3) |
| k (EOS stiffness) | 500.0 (Water) | 3.0 | High k needed to resist gravity with force_scale=0.02 |
| gamma (EOS exponent) | 7.0 (all) | 7 | Tait EOS for all materials |
| mu_0 (viscosity) | 1.0 | 3.5 | GRANULAR base viscosity. FLUID/GAS use per-material base_viscosity as dynamic mu, nu = mu / rho0 (3.3) |
| epsilon (XSPH) | 0.8 (WCSPH profile `xsph_epsilon`) | 0.5 | Higher for smoother FLUID advection |
| gravity | -4.0 | -9.8 | Reduced for miniature-world aesthetic |
| dt | adaptive | 0.001 (fixed) | DT_MAX * accuracy / 0.4, capped by acoustic CFL, in [1e-5, 0.001] (see 4.2) |
| force_scale | 0.02 | N/A | Matches parent's `output * m_j` convention. FLUID and GAS pairs; GRANULAR pairs use 1.0 |
| FLUID-GRANULAR pair scale | pressure: force_scale; viscosity: harmonic mean of base_viscosity and mu | N/A | Same value on both sides, so pair forces are antisymmetric (3.2, 3.3) |
| step2 output | accel * force_scale | accel * m_j | Both effectively multiply by 0.02 |
| integrate | accel = sph + g | accel = sph + g | Both treat step2 output as acceleration |
| position update | FLUID: pos += (vel_new + eps*xsph_sum) * dt; others: pos += vel_new * dt | pos += vel_new * dt | FLUID uses XSPH for smooth advection |
| boundaries | impulse SDF | penalty springs | Fundamentally different |
| velocity_limit | 0.9 * h / dt (36 m/s at 1 ms, 72 m/s at 0.5 ms) | 200 | CFL-derived per solver, see 4.2 |
| accel_max | 200 | N/A | Prevents shockwave-level accelerations (FLUID + GRANULAR) |
| boundary_stiffness | N/A | 20000 | Only in parent |
| boundary_dampening | N/A | 256 | Only in parent |
| restitution | 0.3 | N/A | Only in fallingsand3d |
| wall_friction | 0.5 | N/A | Only in fallingsand3d |
| MAX_RIGID_BODIES | 8 | N/A | Dynamic Akinci bodies (global memory) |
| RIGID_PSI_RHO0 | 2500 | N/A | Rigid boundary mass = RIGID_PSI_RHO0 * V_b; kernels rescale to psi_b = rho0_i * V_b (22.3) |
| MAX_SDF_OBJECTS | 16 | N/A | Analytical SDF collision primitives (constant memory) |
| rigid lin_damp | 1 - 0.01*dt | N/A | Linear velocity damping |
| rigid ang_damp | 1 - 0.05*dt | N/A | Angular velocity damping |
| rigid omega_max | 20 rad/s | N/A | Angular velocity clamp |
| rigid F_max | 1000*mass | N/A | Force magnitude clamp |
| mu_s (GRANULAR) | per material `friction_coeff`, global 0.55 if 0 | N/A | WCSPH mu(I) and PBF Drucker-Prager slope (5.3, 12.7.3) |
| phase points | from the material table | N/A | ICE/STONE/SAND/LAVA `temp_melt`, WATER `temp_boil` (9.5) |

MaterialProps fields no kernel reads: `cohesion` (PBF cohesion is the global
`c_granular.cohesion`, a length in metres, and the table column is not a length),
`temp_ignite` (ignition is exposure-based, 9.4), and `eos_stiffness`/`eos_gamma`
for STATIC materials (Akinci mirrors the fluid's own pressure, 3.2).
`test_materials.py::test_every_materialdef_field_is_read_by_a_kernel` checks this list.

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

This holds for FLUID and GAS pairs. GRANULAR pairs use scale 1.0, and FLUID-GRANULAR
pairs use one shared value on both sides (3.2, 3.3).

### 7.3 Particle mass and rest density

With m = 0.02 at spacing 0.02 the SPH kernel-sum density at rest is about 2525
(m = rho0 * dx^3 = 0.008 would give the physical 1000). WATER and ACID use
`rest_density = 2500` to match, so they rest at rho/rho0 ≈ 1.0 with p ≈ 0 and the Tait
EOS pushes back only under compression. A rest_density of 1000 with m = 0.02 would
start every particle at ratio 2.5, p = k * (2.5^7 - 1) ≈ 610*k, and explode.

Other FLUID and GRANULAR materials have their own rest_density (SAND 4000, LAVA 6500,
OIL 2125). Each spawns with `m = 0.02 * rest_density / 2500` (`world.spawn_mass`), so
the kernel sum at spacing 0.02 is its own rho0 and it starts at p ≈ 0 too. With one
mass for everything, SAND spawned at 0.63 rho0 and LAVA at 0.39 with zero pressure
until gravity compacted them 1.6x and 2.6x, and OIL spawned at 1.19 and puffed up.

Measured on 0.6 m cubes after 300 substeps (mean rho/rho0 over full-support particles,
`test_rest_density.py`):

| Solver | WATER | OIL | SAND | LAVA |
|--------|-------|-----|------|------|
| WCSPH, one mass | 1.08 | 1.26 | 0.67 | 0.40 |
| WCSPH, spawn mass | 1.08 | 1.07 | 1.02 | 1.04 |
| PBF, spawn mass | 1.00 | 1.00 | 0.99 | 1.00 |
| DFSPH, spawn mass | 0.94 | 0.95 | 0.97 | 0.99 |

With one mass the LAVA piles packed to 55-59 neighbours and DFSPH SAND to 45; with the
spawn mass every pile keeps 27-33. DFSPH at 300 substeps is 1 s, when the unconfined
WATER and OIL cubes are splashing off the walls, so they read 0.94-0.95.

STATIC keeps m = 0.02: STONE (6500) or METAL (19500) at their own rho0 would make
wall-adjacent fluid read far above its rho0. GAS keeps m = 0.02 (its units are a
separate issue). MAT_RIGID boundary particles keep their psi_b. K_Reactions applies the
same rule when a particle changes material (9.5).

Interfaces: density is `rho_i = sum_j m_j W_ij`, so a light particle among heavy
neighbours measures the heavy density. OIL inside WATER reads about 1.19 * its rho0,
and a heavy particle among light ones reads low. The number-density form of
Solenthaler and Pajarola 2008, `rho_i = rho0_i * sum_j V_j W_ij` with `V_j = m_j / rho0_j`,
would remove that; it is the follow-up.

### 7.4 Spawn spacing versus rest density

`spawn_mass` puts a FLUID or GRANULAR particle at its rho0 only on the 0.02 lattice.
The mass is not scaled with spacing, so particle size stays the same across a scene,
and the kernel sum scales as `(0.02 / s)^3`: a liquid box at spacing 0.022 spawns at
0.74 rho0 and one at 0.025 at 0.49 rho0 (measured median, poly6, 0.4 m cube).
`compute_pressure` clamps `p >= 0`, so such a block has no pressure until it has
compressed back to rho0, and PBF/DFSPH apply large first-frame corrections.

Every preset FLUID spawn box therefore uses spacing 0.02. Where a preset used to spawn
at 0.022 to 0.025, the box height shrinks (Sandbox: the whole cube) so the particle
count stays within 11% of before. GRANULAR and STATIC spawns keep their own spacings.
The one exception is the Acid Rain drop layer at spacing 0.04 = h, which is separate
drops, not a liquid block. `presets._check_fluid_spacing` finds FLUID spawns above
`DEFAULT_SPACING`, and `test_presets.py` runs it on every preset. One substep after
loading Lava Lamp and Multi-Fluid, every liquid reads a median of 1.01 rho0 (0.74 with
the old 0.022 spacing).

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

Wake propagation (`wake.run_wake_propagation`, a memset and 2 kernels on the
unsorted arrays) runs at the end of every substep for all three solvers.
`K_MarkWakeCells` flags the 3x3x3 cells around each particle that has
JUST_WOKE set, and around each awake particle faster than 0.02 m/s. The
velocity trigger skips STATIC and GAS particles. GAS is skipped because step1
and step2 apply no forces between GAS and non-GAS, so smoke or fire over a
sleeping pile would only wake it, let it re-sleep 10 substeps later, and wake
it again. A GAS particle that has JUST_WOKE set still marks its cells. Until
2026-07-11 it ran only for WCSPH, so a sleeping PBF or DFSPH pile never woke
when something hit it.

Rigid bodies wake sleepers through `K_MarkWakeCellsRigid`, one block per body,
launched from `run_wake_propagation` when the scene has bodies. Rigid boundary
particles are STATIC, so the particle trigger never fires for them, and a
sleeping particle returns from the solver before the rigid coupling. Without
this kernel a box dropped on a sleeping pile got no force and fell through it
to the floor. A body marks when the fastest point on it can exceed V_WAKE:
```
r = |half_extents|                 (bounds a box of any rotation, or a sphere)
if |v_body| + |omega_body| * r > 0.02:
    mark every cell of the cube [c - r - cell, c + r + cell]
```
One cell (about h) of padding lets sand within one smoothing length of the
surface wake. A body at rest marks nothing, so a box resting on a pile lets
the pile sleep; if the box then starts to sink, it wakes the pile again within
a few substeps. For WCSPH, `K_Integrate` keeps the MAT_RIGID boundary velocity
(written by `K_UpdateBoundaryParticles`) in the unsorted velocity array, where
other STATIC particles get zero.

**Source**: `integrate.cu:54-59, 115-127, 280-302, 407-418`, `wake.cu`
(`K_MarkWakeCells`, `K_MarkWakeCellsRigid`), `wake.py`, `simulation.py`
(`_run_wcsph_body`, `_run_pbf_body`, `_run_dfsph_body`, `_wake_rigid_kwargs`)

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

SPH heat diffusion uses a pairwise conductivity `k_ij`, the harmonic mean of the two
particles' `thermal_conductivity` values (Cleary and Monaghan 1999), divided by
`rho_i * cp_i` for physically correct thermal diffusivity:
```
k_ij = 2 * kappa_i * kappa_j / max(kappa_i + kappa_j, 1e-6)
dTdt = [1 / (rho_i * cp_i)] * [45/(pi*h^6)] * SUM_j k_ij * (m_j/rho_j) * (T_j - T_i) * (h - |r|) * heat_boost
```

Where `kappa = c_materials[mat_id].thermal_conductivity`,
`cp_i = c_materials[mat_id].heat_capacity` and `rho_i` is the density the solver
stores for particle i: the computed SPH density, or `max(rest_density, 1)` for a
STATIC wall particle (11.1).

`k_ij` and `heat_boost` are the same seen from i and from j, so the energy rate
`m_i cp_i dTdt_i` of a pair is equal and opposite: heat lost by i is gained by j and
`SUM_i m_i cp_i T_i` is conserved, as long as the density a particle divides by matches
the `density_in` its neighbours read. Same-material pairs get `k_ij = kappa_i`
(unchanged). Mixed pairs are limited by the poorer conductor: LAVA (500) and WATER (50)
give `k_ij = 91`, LAVA and METAL (2000) give 800. All five loop copies use this form:
`K_Step1`, `K_Step1_BuildNL`, the neighbour-list step1 (`neighbor_list.cu`),
`K_PBF_ComputeLambda` and `K_DFSPH_ComputeDensityAlpha`.

`test_step1.py::test_heat_energy_conserved_lava_water` checks the step-0 rate balance
(`|SUM m cp dTdt| / SUM |m cp dTdt| < 1e-4`, measured 3e-8) and the drift of
`SUM m cp T` over 100 fixed-geometry Euler steps (`< 1e-5`, measured 2e-8). With
`kappa_i` applied per particle the imbalance was 0.82.
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
phase transitions: lava→gravel (T<1000K), water→steam (T>373K). See §9.5.

**Cross-phase pairs (GAS with non-GAS).** The GAS/non-GAS branch skips density
and forces, but it does conduct heat, so FIRE, STEAM and SMOKE heat or cool
WATER, ICE, METAL, STONE, WOOD and the rest (bd-r5-review-n28.94). It uses the
same pair term as above, with the pair coefficient capped:
```
a_ij    = k_ij * (m_j/rho_j) * (h - |r|) * heat_boost
rho_lb  = max(C_poly6 * m_i * h^6, is_gas_i ? RHO_EPSILON : 1)   (own self term)
a_x_max = 0.5 * rho_lb * cp_i / ([45/(pi*h^6)] * dt)
sum    += min(a_ij, a_x_max) * (T_j - T_i)
```
`rho_lb` is a lower bound on the `rho_i` that PostCalc divides by, because the
self term is always in the density sum. So the cap holds one neighbour's
contribution to `|dTdt_i| * dt` at or below `0.5 * |T_j - T_i|`, and the explicit
step cannot overshoot that pair. When the cap is not hit the pair stays
symmetric and conserves energy like a same-phase pair. In normal scenes it is
not hit: WATER at 293 K with one FIRE neighbour at 1200 K at r = h/2 (both mass
0.02, isolated) gives WATER +1688 K/s and FIRE -2827 K/s, about 1.7 K and
2.8 K per 1 ms substep against a 454 K cap. MAT_RIGID neighbours never exchange
heat, in either branch. The cap reads `c_sim.dt` of the solver's own module.

This is in `K_Step1`, `K_Step1_BuildNL`, `K_PBF_ComputeLambda` (do_heat call) and
`K_DFSPH_ComputeDensityAlpha`. The neighbour-list step1 in `neighbor_list.cu`
(used only by `benchmark.py` and tests, not by `simulation.py`) still skips
cross-phase heat. Tests: `test_step1.py::test_fire_heats_water_across_phase`,
`test_step1.py::test_cross_phase_heat_capped_at_half_gap` and
`test_solvers.py::test_cross_phase_heat_matches_across_solvers` (all three
solvers give identical dTdt in a WATER/SAND/STONE/FIRE/STEAM block).

Fire under water: WATER turns FIRE into SMOKE once water holds more than 37.5%
of the fire particle's support (§9.4, `FIRE_EXTINGUISH_EXPOSURE`). GAS passes
through non-GAS, so a FIRE layer under a 5 cm WATER column rises into the water
and is extinguished within about 0.1 s. In that time it heats the bottom water
from 293 K to about 345 K (WCSPH, 200 FIRE and 300 WATER particles). With the
extinguish rule turned off, the same scene takes the bottom water past 373 K in
under 0.1 s under all three solvers.

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

### 9.4 Reaction Exposure (Step1 → K_Reactions)

Step1 gives each particle two exposure values that K_Reactions compares with
thresholds. Both are normalised by the kernel support weight of the whole
neighbourhood, so they do not depend on `h` or on how many neighbours there are:

```
W_ij = (h^2 - r_ij^2)^3          (poly6 shape; poly6_coeff cancels)
S_i  = SUM_j W_ij                over every non-self, non-DEAD neighbour in range,
                                 including cross-phase GAS and MAT_RIGID
exposure_corrode_i = SUM_j reaction_rate_ij * W_ij / S_i                    [1/s]
exposure_heat_i    = SUM_j heat_exchange_ij * max(T_j - T_i, 0) * W_ij / S_i  [K]
```

(0 when the particle has no neighbours.) `exposure_corrode` is the reaction rate
times the share of the kernel support held by reacting neighbours, so it lies in
[0, max reaction_rate] = [0, 1.0]. `exposure_heat` is `heat_exchange` times a
temperature difference, weighted the same way.

PBF (`K_PBF_ComputeLambda`, first call of the substep) and DFSPH
(`K_DFSPH_ComputeDensityAlpha`) compute the same two values with the same `S_i`,
so all three solvers give the same exposure for the same particle configuration
(`test_solvers.py::test_exposure_matches_across_solvers`).

Magnitudes at `h = 0.04`, spacing `0.5h` (26 neighbours, `S = 4.16 W(0)`):
- one nearest neighbour holds about 10% of the support; a neighbour at 0.85h
  holds about 0.5%.
- one FIRE neighbour (1200 K) next to 293 K WOOD at `h/2`: `25 * 907 * 0.10 = 2300 K`.
  At 0.85h: about 115 K.

Thresholds in `reactions.cu`:

| Reaction | Threshold | Meaning |
|----------|-----------|---------|
| WOOD → FIRE | exposure_heat > 500 K | one FIRE neighbour within about 0.75h |
| OIL → FIRE | exposure_heat > 300 K | |
| GUNPOWDER → FIRE | exposure_heat > 100 K | almost any FIRE neighbour in range |
| Corrosion | health -= exposure_corrode * dt | full acid contact kills in 1/rate s (STONE 6.7 s, METAL 3.3 s, WOOD 5 s) |
| ACID self-consumption | health -= 0.5 * exposure_corrode * dt | |
| Sand wetting | see §21.1 | thresholds are rate × water fraction |

K_Step1 and K_Step1_BuildNL use this definition. The PBF and DFSPH density
kernels and the benchmark-only `K_Step1_NL` (`neighbor_list.cu`) still output
the older unnormalised `poly6_coeff * SUM_j rate_ij * W_ij`.

**Source**: `step1.cu` (K_Step1, K_Step1_BuildNL PostCalc), `reactions.cu` thresholds

### 9.5 Temperature Phase Transitions (K_Reactions)

| Transition | Condition | Temperature after |
|------------|-----------|-------------------|
| ICE → WATER (FLUID) | T > ICE `temp_melt` (273 K) | 273 K |
| WATER → ICE (STATIC) | T < WATER `temp_melt` - 10 K (263 K) and (sleeping or \|v\| < 0.02 m/s) | 263 K; velocity set to 0 |
| STONE → LAVA (FLUID) | T > STONE `temp_melt` (1500 K) | 1500 K |
| SAND → LAVA (FLUID) | T > SAND `temp_melt` (1700 K) | 1700 K |
| LAVA → GRAVEL (GRANULAR) | T < LAVA `temp_melt` (1000 K) | 1000 K |
| STEAM → WATER (FLUID) | T < WATER `temp_boil` - 13 K (360 K) | unchanged; mass rule below |
| WATER boils | T > WATER `temp_boil` (373 K) | unchanged; sets the spawn-gas flag and spawn.cu makes STEAM |

All thresholds come from `c_materials` (bd-r5-review-n28.89). STEAM has no
`temp_boil` of its own, so condensation is keyed to WATER's. The 10 K and 13 K
gaps are the defines `WATER_FREEZE_HYSTERESIS` and `STEAM_CONDENSE_HYSTERESIS`.

Every material change also rescales mass by the spawn-mass rule (7.3):
`m *= s(new) / s(old)` with `s = rest_density / 2500` for FLUID and GRANULAR and 1 for
STATIC and GAS. STONE → LAVA multiplies by 2.6, LAVA → GRAVEL multiplies by 4250/6500 (0.65),
SAND → WET_SAND multiplies by 1.125, DIRT → MUD by 1.2, and WOOD, OIL or GUNPOWDER →
FIRE go back to 0.02. Mass is not conserved across a transition; the rest volume is.

STEAM → WATER is the exception: it conserves mass in expectation
(bd-r5-review-n28.95). Boiled steam carries `m_src / 3`, and water at that mass
sits at 1/3 of rho0, where Tait pressure clamps to 0 and the drops clump. So a
condensing particle survives as WATER with the full water spawn mass
`m_w = particle_mass * s(WATER)` (0.02) with probability `m_steam / m_w`, drawn
from `wang_hash(i, frame)`. Otherwise it becomes DEAD (STATIC, mass 0, lifetime 0)
and goes on the spawn freelist. Boiled steam survives with probability 1/3, so
3 children give one water particle on average. Brush steam has mass 0.02 and
always survives. With no mass buffer the particle simply becomes WATER.

Condensation is at 360 K, 13 K below boiling. Steam in still air cools only by
ambient relaxation, `T(t) = 293 + 80 exp(-0.02 t)`, so boiled steam (373 K)
reaches 360 K after 8.9 s. `K_SpawnGas` gives it a 12 s lifetime so it
condenses before it expires. Steam kept hot by its neighbours can still expire.

WATER freezes 10 K below the ICE melt point, so water between 263 K and 273 K
stays liquid and ice at 273 K does not flip back and forth. ICE is STATIC, so
only water that is sleeping or slower than V_WAKE (0.02 m/s) freezes. Faster cold
water keeps flowing until it slows, so a stream does not stop in mid-air.

Cooled LAVA becomes GRAVEL (GRANULAR, rest density 4250), not STATIC STONE. A
STATIC particle never moves, so lava that cooled in flight or inside a water
pool would hang where it cooled. GRAVEL falls until something supports it, and
the sleep system then pins it. GRAVEL has no melt transition, so solidified lava
does not melt back.

Melt and solidify set the temperature to the threshold. The heat above it (or
missing below it) is taken as consumed latent heat, so melted ice starts as
273 K water rather than at the overshoot temperature. The particle has to gain or lose heat
again before any reverse transition. The removed heat is not stored, so a
particle does not sit at the threshold while it absorbs a full latent heat. A
per-particle latent-energy accumulator (4 B per particle) for that
is not worth it here (PROBLEMS.md J4).

**Source**: `reactions.cu` (K_Reactions, phase transitions block)

### 9.6 Spawn Slot Initial State and GAS Lifetimes

Every host spawn (`spawn_sphere`, `spawn_cube`, `spawn_ramp`, `spawn_bowl`,
`spawn_cylinder_shell`, and `spawn_wall_with_gap` through `spawn_cube`) writes
all per-particle fields of its new slots through one helper,
`World._init_slot_state`. A reused slot keeps nothing from its previous
occupant:

| Field | Initial value |
|-------|---------------|
| position.xyz | spawn point |
| position.w | material `rest_density` (read as the previous-substep density by step1) |
| density | material `rest_density` |
| velocity, veleval | spawn velocity, w = 0 |
| sph_force, angular_velocity | 0 |
| kappa, kappa_v (DFSPH warm start) | 0 |
| mass | `spawn_mass(material)` (§7.4) |
| temperature | §9.3 default |
| health | 1 |
| lifetime | GAS default below, else 0 |
| shear_rate, exposure_heat, exposure_corrode | 0 |
| color, particle_dye | material color |

`presets._clear_world` also zeroes `kappa`, `kappa_v` and `sorted_lambda_pbf`
for the whole pool, so no DFSPH or PBF warm-start pressure from the previous
scene reaches the first substep of the next one.

K_Reactions counts `lifetime` down only while it is above 0. A spawned GAS
particle gets a default lifetime from `_DEFAULT_LIFETIMES` in `world.py`:

| Material | Default lifetime | On expiry |
|----------|------------------|-----------|
| FIRE | 1.5 s | becomes SMOKE with lifetime 3.0 s |
| SMOKE | 3.0 s | becomes DEAD |
| STEAM | 5.0 s | becomes DEAD |

Steam from boiling (`K_SpawnGas`) gets 12 s instead (§9.5). Brush STEAM at
373 K with 5.0 s only cools to 365 K, so it expires before it can condense.

So brush and preset FIRE turns to SMOKE after 1.5 s and dies after 4.5 s,
like fire from combustion. A caller can pass `lifetime=` to any `spawn_*`;
`lifetime=0` makes a GAS particle permanent.

Periodic preset spawners run through `presets.run_spawner`, which skips a
spawn once `_high_water` is above the spawner's `budget` (default
0.9 × `max_particles`; Fire and Ice and Water Mill set it explicitly). The
last batch can pass the budget by at most its own size.

**Source**: `world.py` (`_init_slot_state`, `_DEFAULT_LIFETIMES`),
`presets.py` (`_clear_world`, `run_spawner`), `reactions.cu` (GAS lifetime decay)

---

## 10. Known Issues and Discrepancies

### 10.1 FIXED: integrate.cu divided sph_force by mass
Step2 outputs acceleration. Previous code divided by mass again (125x amplification).
**Fixed**: removed `inv_mass` multiplication.

### 10.2 FIXED: SPH force scaling and particle mass
The parent multiplies step2 output by `m_j = 0.02` and uses mass=0.02 for density.
**Fixed** in two parts:
1. Particle mass changed from 0.008 (rho0*dx^3) to 0.02 (matching parent) for a
   rest_density of 2500. Other FLUID and GRANULAR materials scale it by
   rest_density / 2500 (7.3).
2. Added `force_scale = 0.02` in GranularParams, applied at step2 output.
   This matches the parent's `output * m_j` convention.
**Source**: `world.py` (`PARTICLE_MASS`, `spawn_mass`), `step2.cu:48,393`, `step2.py:49,101`

### 10.3 CHANGED: XSPH in the FLUID position update (Game SPH)
The parent advects with the new velocity only. Here FLUID adds the XSPH delta:
`pos += (vel_new + epsilon * xsph_sum) * dt`, which smooths compression artifacts at
larger dt. Other behaviors use `pos += vel_new * dt`. Both are symplectic Euler (4.3).
**Source**: `integrate.cu:279-294`

### 10.4 FIXED: Viscous CFL did not match the viscosity form
The adaptive timestep had a viscous CFL `accuracy * rho_min * h^2 / max_eta` built
from `base_viscosity`. It assumed `nu = mu / rho`, but the FLUID viscosity has no
`1/rho_i` (see 3.3), so the real explicit rate is `fs * mu * 15 / h^2` per second.
The CFL gave about 7 ms and never bound, while LAVA needed about 0.1 ms. The corrected
explicit bound (1.1e-5 s for LAVA) would have made WCSPH about 20x slower.
**Fixed** (bd-r5-review-n28.52): step2 treats the viscous self-term implicitly, which
is stable at any dt, and the viscous CFL is removed. GRANULAR at `mu_max` is covered by
the same scale: its explicit factor at dt 1 ms is 1.88 (rho 2500), which the implicit
self-term turns into a monotone decay, so `mu_max` needs no CFL term either (5.4).
**Source**: `simulation.py:_compute_adaptive_dt`, `step2.cu` (implicit scale)

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
6. **GAS phase separation**: GAS particles skip regular STATIC neighbours in
   the SPH loops. In WCSPH step2 a GAS particle does take the MAT_RIGID Akinci
   branch (pressure mirror with the gas EOS pressure and gas rest density,
   viscous drag toward the body surface velocity, reaction force and torque on
   the body), so moving bodies push and drag smoke and steam. Gas density in
   step1 still ignores the body. (bd-r5-review-n28.101)

7. **Position push** (`static_particle_boundary`, `sph_shared.cuh`): after
   integration, a FLUID, GRANULAR or GAS particle closer than
   `min_dist = 0.5 h` (one rest spacing, 0.02 m) to a STATIC particle gets one
   correction:
   `n = normalise(sum_j (min_dist - r_j) * r_hat_j)`,
   `pos += max_j (min_dist - r_j) * n`. The push length is the deepest
   penetration, not a sum over neighbours, so a particle over a flat floor
   moves at most `min_dist - y`. Velocity is reflected with restitution along
   `n` only, and only when moving into the wall. A particle resting one
   spacing from the wall is not touched. Called from WCSPH `K_Integrate`, PBF
   Finalize and DFSPH Finalize. (bd-r5-review-n28.43)
   MAT_RIGID particles push only GAS: FLUID and GRANULAR get their rigid
   response from the solver's Akinci coupling, but gas density ignores the
   body, so without the push gas would cross the body surface
   (bd-r5-review-n28.101).

   Measured on a 1500-particle water column in a closed STONE box (bottom
   layer after 5 s, old 0.8 h summed push vs this version): PBF gap
   1.58 dx to 0.96 dx, bottom vy std 0.114 to 0.030 m/s, max per-substep
   displacement 0.036 m to 0.0044 m (under the 0.15 h grid-reuse threshold in
   100% of substeps, was 1.2%). DFSPH gap 1.49 dx to 0.77 dx, jitter unchanged.
   No leaks in either.

   Known gap: the integrate module never receives `c_grid` (simulation.py
   uploads only `c_sim` and `c_materials` to it), so in WCSPH `calcGridCell`
   and `spatialHash` see zeros and this push scans one wrong bucket. It has
   no effect in WCSPH until that upload is added.

Additional FLUID boundary improvements (unchanged):
- **XSPH position advection for FLUID**: Smooths compression artifacts.
- **Zero wall friction for FLUID**: Prevents domain-wall sticking.
- **All FLUID pressure clamped >= 0**: No tensile/negative pressure.

**Source**: step1.cu, step2.cu, pbf_solver.cu, dfsph_solver.cu,
sph_shared.cuh:static_particle_boundary

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
7. **XSPH epsilon**: set per WCSPH profile, 0.8 for both (see 3.4).
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
The velocity-based advection CFL was never binding (velocity_limit and accel_max
prevent supersonic particles), so it was removed. Adaptive dt now uses the accuracy
bound plus the acoustic CFL, both computed on the CPU from the materials
table and `accuracy` (see 4.2). There is no GPU sync in the hot path.
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
hash = ((uint)cx * 73856093u + (uint)cy * 19349669u + (uint)cz * 83492791u) & TABLE_MASK
```
- TABLE_SIZE = 262144 (2^18) by default; `compute_table_size` picks 2^18 to 2^20
  from the particle count
- cell_start/cell_end always 262K entries (~2MB total) at 2^18
- The products and the sum wrap in 32-bit unsigned arithmetic, so negative
  cell coordinates are well defined
- **Stencil cells never share a bucket.** The combiner is addition, so
  `hash(c + d1) - hash(c + d2) = (d1 - d2) . P (mod table_size)` does not
  depend on the cell `c`. No difference `d1 - d2` in {-2..2}^3 is 0 mod 2^18,
  2^19 or 2^20, so the 27 cells of every 3x3x3 neighbour stencil land in 27
  distinct buckets and no particle is counted twice. `test_hash_sort.py`
  checks this for world edges, faces, interior and negative cells.
- Why not XOR: `P ^ (-P)` has the same low 18 bits (0x3FFFE) for all three
  primes, so for a cell with two coordinates at 0 (the first in-world layer,
  since `grid_min = world_min`) offsets (a,-1,-1) and (a,+1,+1) shared a
  bucket. Particles in the 0.04 m strip along those world edges scanned cell
  (a,1,1) twice: +3 to +12% density and +23 to +120% Tait pressure.
- Far-apart cells can still share a bucket. That costs extra distance checks
  but never misses or double-counts a neighbour. The hash is linear, so two
  cells collide only when their offset `d` has `d . P = 0 mod table_size`;
  at 2^18 the shortest such offset is (18, 47, 45) cells.
- CuPy's kernel disk cache keys on the `.cu` source text and options, not on
  included headers. After changing `spatialHash` or anything else in
  `common.cuh`, clear `~/.cupy/kernel_cache`, or modules whose `.cu` file did
  not change keep the old code, and the hash then disagrees between the sort
  and the neighbour loops.

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
   lambda+delta computation. GAS now has no density constraint. Its lambda slot
   holds the gas EOS term p/rho^2 and ComputeDelta applies only the GAS-GAS
   pressure delta (see 12.5).
5. **PBF s_corr_n hardcoded to 4** (MEDIUM): Documented in code. At h=0.04, s_corr is
   disabled (k=0) so this has no practical impact.
6. **DFSPH K_DFSPH_ComputeKappaFromVelocity uses m_j not m_j/rho_j** (HIGH if used):
   Kernel deleted (it was never launched).
7. **DFSPH K_DFSPH_ComputeDensityAdv uses stale grid** (MEDIUM): Kernel deleted (it was
   never launched).
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
- C1: XSPH epsilon comes from the WCSPH profile; both WCSPH profiles use 0.8 (see 3.4)
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
- I1: FLUID advects with `vel_new` plus the XSPH delta (symplectic Euler, 4.3); there is no pre-integration lag
- I3: Grid reuse: WCSPH (0.25*h threshold, 4-frame cap), PBF/DFSPH (0.15*h threshold, 2-frame cap)
- F1: force_scale=0.02 (co-adapted WCSPH convention). It applies per pair: FLUID-GRANULAR pairs use force_scale on both sides (3.2)
- F2: WCSPH viscosity convention (force_scale absorbs 1/rho for FLUID-FLUID). FLUID-GRANULAR pairs use the symmetric dynamic form of 3.3
- F3: Tait EOS for mu(I) (lithostatic needs expensive column height)
- G4: Surface tension is "surface normal cohesion" (not full Akinci model)
- J10: Sorted array transitions handled by per-particle branching (safe invariant)

**Reaction fixes (2026-06-10 batch, bd-mzc.27):**
- DIRT + corrosion-exposure (driven by WATER contact reaction_rate) now converts DIRT to
  MUD (wetting transition), not corrosion damage. This check runs before the acid
  corrosion whitelist so DIRT is redirected to the wetting path rather than being
  "corroded" by water.
- DIRT is not on the acid-corrosion whitelist. The corrosion path (`health -= damage`)
  applies only to: STONE, METAL, WOOD, GRAVEL, OIL, GUNPOWDER (ICE is excluded too,
  see 21.3). DIRT reaches MUD via the wetting path (`exp_corrode > SAND_WET_THRESHOLD`)
  instead.

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

### 10.17 FIXED: GAS rest density was unreachable

The density output floor is behavior-aware:
```
float rho_floor = is_gas_i ? RHO_EPSILON : 1.0f;
density_out[i] = fmaxf(density, rho_floor);
```
Neither floor is reached by a live particle. Its own Poly6 self term is 163-490
for GAS (mass 0.02/3 to 0.02) and more for every other behavior, so the floors
are only a guard for massless slots. Denominator guards across the solvers use
`fmaxf(rho_j, RHO_EPSILON)`.

With real-world GAS rest densities (0.2-0.6) the self term alone kept the gas
EOS `p = k * max(rho - rho0, 0)` on for every particle, so gas clouds scattered
until no neighbour was within h. GAS `rest_density` is now in kernel-sum units
(STEAM 1200, SMOKE 1100, FIRE 1000; see section 3.1), so an isolated gas
particle has `p = 0` and a cloud stops pushing apart at `rho0`
(bd-r5-review-n28.80). Test: `test_gas_physics.py::test_gas_rest_density_reachable`.

**Source**: `sph_shared.cuh` (RHO_EPSILON), `step1.cu`, `step2.cu`,
`dfsph_solver.cu`, `pbf_solver.cu`, `implicit_st.cu`

### 10.17b FIXED: Round-4 review fixes (2026-07-11, branch fix/review-2026-07-11)

Each item names its bead (`bd-r4fix-uup.N`).

- **Akinci rigid-boundary pressure sign (.1).** DFSPH and PBF used `grad_spiky`
  (which bakes in the negative coefficient `-45/(pi h^6)`) with a `+` sign, so the
  fluid pulled on rigid bodies instead of pushing. Both now use the step2.cu form:
  `grad_spiky_variable` times the positive `pressure_precalc`. The magnitude is
  unchanged; only the sign flips. In DFSPH and PBF this term feeds only the
  reaction force on the body.
- **Per-particle mass in rigid coupling (.4).** Three coupling sites used
  `c_sim.particle_mass` instead of the particle's own mass.
- **Reaction state carried back to unsorted arrays (.9).** Reactions and spawn
  edit `health`, `lifetime` and `mass` in sorted space. The next substep's
  gather re-read the unsorted copies, so every edit was lost: fire never burned
  out, acid never killed, and gas-spawn mass transfer did nothing. `K_Integrate`,
  `K_PBF_Finalize` and `K_DFSPH_Finalize` now scatter all three back.
- **Rigid-body angular acceleration (.10).** The world-frame inverse inertia is
  `R diag(I_body_inv) R^T`. The kernel applied `R diag(I_body_inv)` and dropped
  `R^T`, which gave wrong spin for any rotated body that is not a cube. It now
  computes `R * (diag(I_body_inv) * (R^T * tau))`.
- **Solver grid constants (.11).** PBF and DFSPH modules got grid constants
  built separately from the ones WCSPH uses, and `set_world_size` did not
  re-upload them. All solvers now share one cached `GridParams`.
- **Stale sort permutation (.12).** Grid reuse re-gathers through the last sort
  permutation, which is only valid for the particle count that built it. A full
  sort is now forced whenever `n` changes and on every scene change (brush,
  spawner, reset, preset, undo, load).
- **Density buffer lifecycle (.14).** The density kernels read neighbor density
  (`density_in`, used as `m_j / rho_j` weighting in heat, strain rate,
  vorticity and normals) from the same buffer they wrote (`density_out`). That
  was a `__restrict__` read/write race, the buffer was not re-ordered by the
  sort, and the first substep read zeros, giving `m_j / 0`. Now:
  - `density_in` is a separate buffer, `sorted_density_prev`.
  - WCSPH fills it from `position.w`, which holds the previous density and is
    re-ordered with the position, so the values match the current sort order.
    K_Step1 writes the density into sorted `.w`, and `K_Integrate` carries it to
    the unsorted `position.w` for awake particles (STATIC and sleeping particles
    pass `.w` through unchanged). Until 2026-09-24 `K_Integrate` wrote `.w = 1.0`
    for awake particles, so from the second substep every awake neighbour had
    `rho_j = 1` and every `m_j / rho_j` term in step1 was about 1000 to 2500
    times too large. `test_wcsph_density_in_carries_previous_density` covers it.
  - PBF and DFSPH fill it from `sorted_density`. This removes the race, but the
    values are still in the previous sort order after a full re-sort. The
    effect is bounded noise in the weighting, not a blow-up.
  - Every read site (K_Step1, K_Step1_BuildNL, K_DFSPH_ComputeDensityAlpha,
    K_PBF_ComputeLambda; same-phase and cross-phase heat, and the PBF
    vorticity/normal sum) replaces a value `<= 0` with the
    neighbour's own `rest_density`, floored at `RHO_EPSILON` so GAS keeps its
    low rest density (bd-r5-review-n28.110). A never-computed or freshly
    spawned slot therefore gets `m_j / rho_j` equal to its rest volume and
    cannot produce `inf`. See §13.2 for the effect on DFSPH alpha.
- **Wake for PBF/DFSPH, foam seed, spawn freelist (.15).** See §8.2 and §19.3.
  The spawn freelist is reset every substep because it stores sorted indices.
  `K_Reactions` used to add only particles that died in that substep, so dead
  slots that were not claimed at once were lost for good, and boiling water
  almost never found a slot for steam. It now adds every DEAD particle each
  substep. `K_SpawnGas` claims its slots all at once with one `atomicSub`, so a
  partial claim can no longer corrupt the stack (.8).
- **Blast wave fires once (.8).** The trigger window around
  `GUNPOWDER_FIRE_LIFETIME` is now `±0.5*dt`, one substep. It was `±0.01`,
  about 10 substeps at `dt = 0.001`.
- **Gravity kept on resize (.15).** `set_world_size` rebuilt the sim constants
  with gravity `(0,-4,0)` and dropped the user's setting.

The vorticity sign is fixed (bd-r5-review-n28.36). All three solvers now add
up `+curl(v)`, see §16.1.

### 10.18 Rigid Body System Limitations

- **Bounding sphere body-body collisions**: Body-body collision uses the bounding sphere `|half_extents|`, not the true shape, so elongated bodies leave gaps at corners. Body-SDF contact and the world wall clamp use the box support distance (22.7, 22.8).
- **No rigid-rigid stacking**: Single push-apart pass per substep; deep piles of bodies may interpenetrate.
- **Max 8 dynamic bodies, 16 SDF objects**: Hardcoded limits in constant memory (`MAX_RIGID_BODIES`, `MAX_SDF_OBJECTS`).
- **CUDA graph disabled**: Rigid body forces vary per substep, so the CUDA graph pipeline is disabled when any rigid body is present.
- **No body-frame friction/rolling**: Only linear velocity is reflected in collision; angular velocity is not affected by rigid-rigid contacts.
- **Gas against bodies** (bd-r5-review-n28.101): GAS is kept out of a body by the `static_particle_boundary` push (all solvers) plus, in WCSPH, the step2 Akinci term. `test_gas_rigid.py` drops a 12 cm, 50 kg box at 1 m/s through an ambient-temperature SMOKE block for 0.2 s. DFSPH: no smoke inside the box at any check. WCSPH: 216 inside, because the push is dead there until the integrate module gets `c_grid` (10.6, bd-r5-review-n28.214); with `c_grid` uploaded by hand it is 0 inside and the box slows from 1.0 to 0.86 m/s (0.998 m/s with no smoke in its path), and peak gas pressure under the box is about 30000 against 3600 at rest. PBF: up to about 90 inside, because at dt 1/120 the box moves 8 mm per substep, the PBF gas density solve does not see the body, and one push per substep lags behind.

---

### 10.19 Spawn freelist and the parked DEAD reserve

Boiling water (T > 373 K) turns into `SPAWN_N = 3` steam particles only if
`K_SpawnGas` can claim 3 DEAD slots from the freelist. `K_Reactions` rebuilds
the freelist every substep from the DEAD slots in `[0, _high_water)`. A scene
where nothing has died has no DEAD slots, so without a reserve its water never
boils. (bd-r5-review-n28.27)

- **Reserve.** While WATER or ICE has been spawned (`World.needs_dead_reserve`),
  the world keeps `World.dead_reserve` DEAD slots (default 15000, 0 turns it
  off). `World.add_dead_reserve()` appends them at `_high_water` with packed
  info `MAKE_PACKED(DEAD, STATIC)`, mass 0, zero velocity and T = 293 K. Every
  kernel early-returns on them (STATIC skips integrate and step2; mass 0 adds
  nothing to density). Plain `packed_info = 0` would read as behavior FLUID
  and fall under gravity.
- **Parking.** Slot index `s` maps to its own site on a cube lattice of side
  `ceil(cbrt(max_particles))` at spacing `h`, about 1000 m below the world box
  (past the camera far plane of 100). The lattice is snapped to the grid that
  starts at `-world_half_size`, and each site is at a cell centre, so every
  parked slot has its own grid cell. For 15000 slots, no hash bucket holds
  more than 8 of them at table sizes 2^18 to 2^20.
- **When it is filled.** `Simulation._maybe_compact` runs every
  `compact_interval` frames (60) and on the first frame after
  `notify_scene_changed()`. It counts dead slots once (`num_active`, one sync)
  and tops the reserve up to `dead_reserve` DEAD slots.
- **Compaction.** With the reserve on, compaction runs only when there are more
  than `dead_reserve + compact_threshold` dead slots. `World.compact()` drops
  every dead slot, parked or not, and `_maybe_compact` then re-creates the
  reserve at the new tail. Without WATER or ICE, compaction still runs at
  `compact_threshold` (100) dead slots.
- A steam particle claims the slot where it is written, so a parked slot jumps
  from the park lattice to the boiling water. The displacement tracker then
  forces a full sort, as for any reused DEAD slot.

**Source**: `world.py:add_dead_reserve`, `world.py:park_positions`,
`simulation.py:_maybe_compact`, `reactions.cu:K_Reactions`, `spawn.cu:K_SpawnGas`

### 10.20 Dead slots have mass 0 and are not boundary particles

A dead slot keeps its position until compaction or until a spawn reuses it.
Since 10.6, every STATIC neighbour counts as an Akinci boundary particle, so a
dead slot must add nothing to its neighbours. (bd-r5-review-n28.23)

- **Mass 0.** `K_Reactions` takes the sorted mass buffer and writes
  `mass[i] = 0` when acid is consumed, when a gas other than FIRE expires, and
  for every slot that is already DEAD. `K_SpawnGas` and `add_dead_reserve` do
  the same. With `m_j = 0`, every neighbour term is zero: density
  `m_j W`, the WCSPH Akinci pressure mirror and boundary drag, the PBF lambda
  mirror `m_j (lambda_i + lambda_i) / rho0`, DFSPH volumes and heat diffusion.
  Integrate and the PBF and DFSPH Finalize kernels scatter sorted mass back, so
  the zero survives the next sort.
- **No position push.** `static_particle_boundary` (`sph_shared.cuh`) pushes a
  moving particle out to `0.5 h` from STATIC neighbours (see 10.6),
  and that push has no mass factor. It skips `MAT_DEAD`, and it skips
  `MAT_RIGID` for every particle except GAS (see 10.6).
- `MAT_DEAD = 0` is defined in `common.cuh` next to `MAT_RIGID`.
- **Brush kill.** `World.kill_in_sphere` runs one kernel (`K_KillSphere`, no
  host sync). Each live, non-rigid particle in the sphere gets
  `MAKE_PACKED(DEAD, STATIC)` (`world.DEAD_PACKED`, 0x300), zero mass,
  health and velocity, and color `(0, 0, 0, -1)`. As STATIC it skips every
  solver's fluid path.
  `World.compact()` fills the dead tail with the same encoding. The alive test
  everywhere is `material_id != 0`, so a bare `packed_info = 0` also counts as
  dead. (bd-r5-review-n28.34)
- **Rest density floor.** DEAD has `rest_density = 0`. DFSPH
  (`K_DFSPH_DensitySolverUpdate`) and PBF (`K_PBF_ComputeLambda`,
  `K_PBF_ComputeDelta`) use `rho0 = max(rest_density, 1)`, so a DEAD slot that
  still has FLUID behavior cannot divide by zero. The density all three solvers
  write for STATIC particles is floored at 1 too, so `V_j = m_j / rho_j` stays
  finite. The Akinci rigid pressure term floors `rho0_i` at `RHO_EPSILON`
  (0.01), which leaves GAS (0.2 to 0.6) unchanged.
- A fully corroded particle becomes DEAD/STATIC with mass 0 and joins the
  freelist, the same as consumed acid (see 21.3).
- **Rendering.** Every draw covers `[0, _high_water)`, dead slots included.
  `compute_color` and `compute_fluid_color` (`sph_shared.cuh`) return
  `(0, 0, 0, -1)` for `MAT_DEAD`, so Integrate and the PBF and DFSPH Finalize
  kernels mark every dead slot, whatever its behavior bits. `K_KillSphere` and
  `add_dead_reserve` write `color.w = -1` directly. `particle.vert`,
  `particle_nonfluid.vert`, `ssfr_depth.vert` and `ssfr_thickness.vert` move
  any vertex with `color.w < 0` off screen with point size 0, so a killed
  particle vanishes at once and does not wait for compaction (every 60 frames,
  at least 100 dead). Live particles keep the behavior code in `color.w`
  (FLUID 0.0, GRANULAR 0.25, GAS 0.5, STATIC 0.75). The dead vertices are
  still fetched, so they cost vertex work but no fragment work.
  (bd-r5-review-n28.60)
- **Color outside a substep.** The World spawn paths and `K_SpawnGas` write
  the behavior code into `color.w` at spawn. Save files and undo snapshots
  hold no color, so `load_scene` and `SnapshotRing.restore` call
  `World.refresh_color(n)`: one GPU gather that writes each slot's base
  material color and `behavior * 0.25` (DEAD `(0, 0, 0, -1)`). Heat glow
  comes back on the next substep. A paused scene therefore shows the right
  colors right after spawn, load or undo. (bd-r5-review-n28.63)

**Source**: `reactions.cu:K_Reactions`, `sph_shared.cuh:static_particle_boundary`,
`sph_shared.cuh:compute_color`, `shaders/particle.vert`,
`simulation.py:_run_reactions_spawn`, `world.py:kill_in_sphere`,
`dfsph_solver.cu:K_DFSPH_DensitySolverUpdate`, `pbf_solver.cu:K_PBF_ComputeLambda`

### 10.21 Spawned steam: child placement and slot state

`K_SpawnGas` runs after the density pass and before the force or solver pass,
on sorted arrays, so every per-slot value a later kernel reads in the same
substep must be written for the claimed slot. Otherwise the child inherits the
dead occupant's values. (bd-r5-review-n28.42)

- **Placement.** The `SPAWN_N = 3` children sit on a horizontal ring of radius
  `0.3 * particle_spacing` around the source, 120 degrees apart, so pairwise
  distance is `0.52 * particle_spacing`. The ring rotation is hashed from the
  source index and position. Coincident children would never separate, because
  every neighbour loop skips pairs with `r^2 < 1e-12`.
- **Velocity.** Each child gets `v_src + (0, 2, 0) m/s` plus `0.1 m/s` along its
  ring direction. The three spreads sum to zero, so total momentum is
  `m_src * (v_src + (0, 2, 0))`. Mass is `m_src / 3` each.
- **Temperature and lifetime.** 373 K and 12 s, long enough to cool to the
  360 K condensation point in still air (§9.5).
- **Density.** `position.w` and `density` get the steam rest density. WCSPH
  step2 reads `position.w` as `rho_i`; before, it held the source water density.
- **WCSPH.** `pressure[slot] = 0` (gas EOS `k * max(rho - rho0, 0)` at rest
  density) and the FP16 mirror `velocity_h[slot]` gets the child velocity, so
  step2 in the spawn substep sees the child's real velocity.
- **PBF.** `K_PBF_Predict` runs before spawn and leaves the dead slot's old
  position in `predicted_position[slot]`. `K_SpawnGas` writes
  `x* = x_child + dt * v_child` (clamped to the world box) there, so
  `K_PBF_Finalize` gives the child `v = (x* - x) / dt = v_child` and keeps it at
  the source. It also zeroes `lambda`.
- **DFSPH.** `kappa` and `kappa_v` are zeroed. DFSPH skips GAS in its pressure
  solvers, so the slot's `alpha` from the pre-spawn density pass is not used.
  `predicted_position` is not written: DFSPH reuses it as `a_press` scratch.
- **All solvers.** Dye and angular velocity are zeroed. The spawn writes the
  `+inf` bits into `max_displacement`, so the next grid-reuse decision forces a
  full sort (see 20.5 on the readback lag).
- **Blast wave.** `K_BlastWave` adds impulses to `velocity` between step1 and
  step2 but does not refresh `velocity_h`, so WCSPH neighbours see a blasted
  particle's pre-blast velocity for one substep.

**Source**: `spawn.cu:K_SpawnGas`, `spawn.py:compute_spawn`,
`simulation.py:_run_reactions_spawn`, `simulation.py:_run_pbf_body`

### 10.22 OPEN: Residual alternating viscous mode at large x

The implicit viscous scale (3.3) makes every mode decay, but a mode that alternates
between neighbours decays by `(1 - (lambda - 1) x) / (1 + x)`, which tends to
`1 - lambda` for large x. For LAVA (x about 18.75) a pure alternating-plane mode keeps
10% of its amplitude with flipped sign per substep; the worst possible mode
(lambda = 2) would keep 90%. It is bounded and damped, far better than the old
18x growth. In a 20x20x20 LAVA block in a STONE box (500 substeps at 1 ms) the
fraction of particles whose deviation from the neighbour mean flips sign between
substeps is at most 0.3% (it was 99.9% before), and the largest deviation at step 500
is 0.028 m/s against 0.108 m/s for WATER in the same box (0.23 m/s before). A full
implicit solve (Jacobi or CG on the viscous system) would remove the residual.
**Source**: `step2.cu` (implicit scale), `test_implicit_viscosity.py`

## 11. Multi-Solver Architecture

Three solver backends are available, selectable at runtime via UI dropdown.
Switching solvers resets the scene (avoids "physics shock" from different density packing).

| Solver | Algorithm | Substeps/frame | Inner iterations | Best for |
|--------|-----------|---------------|-----------------|----------|
| WCSPH | Weakly Compressible SPH | 10-25 at dt=0.001 | N/A | Accuracy, multi-material |
| PBF | Position Based Fluids | 1-2 at dt=1/60 | 4 constraint iters | Fast fluids |
| DFSPH | Divergence-Free SPH | 5 at dt=1/300 | 3 div + 4 dens iters | Quality + speed |

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
| XSPH smoothing | Step2, FLUID only (eps=0.8, profile) | Finalize (c=0.05) | NonPressure (eps=0.1) |
| GAS buoyancy | Integrate | Predict (v+=) | NonPressure (accel) |
| GAS pressure (gas EOS, GAS pairs, x force_scale) | Step2 | ComputeDelta (dx = dt^2 a / iters) | NonPressure (accel) |
| GAS phase separation | Step1 (skip GAS) | ComputeLambda (skip GAS) | DensityAlpha (skip GAS) |
| Heat diffusion | Step1 | ComputeLambda | DensityAlpha |
| Cross-material heat | Step1 (heat_boost) | ComputeLambda (heat_boost) | DensityAlpha (heat_boost) |
| Cross-phase heat (GAS with non-GAS, capped at 0.5 dT per pair) | Step1 | ComputeLambda | DensityAlpha |
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

**STATIC wall volume convention** (bd-r5-review-n28.116). A regular STATIC
particle (STONE, METAL, WOOD, ...; not MAT_RIGID) has mass 0.02 whatever its
material, and every solver stores `max(rest_density, 1)` as its density:
K_Step1 and K_Step1_BuildNL (WCSPH), K_PBF_ComputeLambda and
K_DFSPH_ComputeDensityAlpha. So:

- Heat, normal, vorticity and dye sums weight a wall neighbour by
  `m_j / rest_density_j` in all three solvers. The wall's own dTdt divides by the
  same `rest_density * cp`, which keeps the pair term conservative (9).
- WCSPH forces do not read the wall density: step2 mirrors the fluid's own
  `p_i` (3.2) and the boundary drag uses `m_j`.
- DFSPH solver terms (alpha `grad_sum`, `div_v` in K_DFSPH_ComputeKappaV,
  `drho` in K_DFSPH_DensitySolverUpdate, XSPH in K_DFSPH_Finalize) use the Akinci
  boundary volume with the fluid's own rest density, `V_j = m_j / rho0_i`
  (`dfsph_vol_rho`). A STONE, METAL or WOOD floor then gives WATER the same
  alpha to 3e-7 (`test_static_wall_volume.py`). With `m_j / rest_density_j` the
  boundary gradient was 2.6x too weak against STONE, 7.8x against METAL and
  1.7x too strong against WOOD, and alpha of the bottom WATER layer differed by
  up to 5% between STONE and WOOD. MAT_RIGID keeps density `RIGID_PSI_RHO0`, so
  its volume stays `V_b` (22).

A hot two-layer STONE wall (1000 K) next to a 10 x 10 x 10 WATER block raises the
mean water temperature by 0.00247 K (WCSPH), 0.00233 K (PBF) and 0.00248 K
(DFSPH) in 5 substeps of 1 ms, and 0.0245, 0.0183 and 0.0242 K in 50. Before this
convention WCSPH stored the kernel sum (1300 to 2540 for that wall) and heated
the water 2.8x faster. A one-layer STONE wall next to LAVA heats 17.7 K (WCSPH),
15.9 K (PBF) and 17.3 K (DFSPH) in 5 substeps (`test_static_heat_cross_solver.py`).
PBF runs about 10% low at the start and drifts lower with time; that is a PBF
effect, not the wall convention.

**Key solver-specific differences:**
- PBF vorticity epsilon is 50x lower (0.001 vs 0.05) to prevent energy injection at fixed dt
- PBF surface tension is disabled (gamma=0) — relies on weak attractive constraint instead
- PBF Drucker-Prager uses velocity-space friction in Finalize (once, iteration-independent)
- WCSPH uses adaptive dt set by `accuracy` (0.5 ms for WCSPH, 1 ms for WCSPH (Fast), see 4.2); PBF/DFSPH use fixed dt

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
     d. x* += dx, then clamp x* to the world box and push it out of SDF objects
  4. v_new = (x* - x_old) / dt
  5. XSPH viscosity: v_new += c SUM (m_j/rho_j) (v_j - v_i) W_poly6
  6. Boundary handling + writeback
```

**Grid from predicted positions.** Steps 1 and 2 run in this order on a full
sort: `_run_grid_setup` runs `K_PBF_Predict` on the unsorted arrays into
`world.predicted_position` and hashes that, so the cell table is bucketed by
x*. After the sort, `K_PBF_Predict` runs again on the sorted arrays. It gives
the same x* per particle, so every x*_i lies in the cell it was sorted into and
every pair with |x*_i - x*_j| < h is inside the 27-cell stencil of cell(x*_i).
The extra predict reads about 40 B and writes 16 B per particle, on full-sort
substeps only. WCSPH and DFSPH still hash x.

What this does not cover:
- The N solver iterations move x* further (each dx is clamped at 0.5h, typical
  corrections are far smaller). The table is not rebuilt between iterations.
- Grid-reuse substeps (`K_GatherReorder`) keep the permutation and cell table
  of the last full sort, so their x* is compared against cells built from an
  older x*. The skip test uses the displacement `K_PBF_Finalize` measures
  (final x* vs x), under the 0.15h PBF threshold.
- `compute_blast_wave` (GUNPOWDER) walks the same table with the current
  position x, which is up to one substep of motion (at most 0.9h) away from
  the x* the cells were built from.

**SDF projection every iteration.** `K_PBF_Predict` and every
`K_PBF_ApplyDelta` pass call `push_out_of_sdf`: any x* closer than
`BOUNDARY_MARGIN` to an SDF object (or inside it) moves along the SDF normal to
`BOUNDARY_MARGIN` outside. So the density constraint near an SDF collider is
solved with the collider in place, and Finalize derives v_new from an x* that
is already outside. Finalize still applies the SDF restitution and friction
impulse. The loop bound is `c_num_sdf_objects`, so it costs nothing in a scene
without SDF objects. `test_pbf_sdf_push.py` checks that no iteration leaves
water more than `BOUNDARY_MARGIN` inside an SDF box.

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
MAT_RIGID neighbours use `psi_b = rho0_i * V_b` (22.3).
A previous `2x boundary_scale` multiplier was removed; it over-inflated near-wall
density to roughly `2*rho_0`, causing lambda to drive particles away from walls
(unphysical bounce). With the 2x factor gone, near-wall density is correct and
wall contact is stable. Lambda and delta gradients also use `m_j` without scaling.

**STATIC and GAS density per substep (bd-r5-review-n28.62)**: STATIC and GAS take
no part in the constraint (lambda = 0). GAS density is computed once per substep,
in the first ComputeLambda call (the one that also does heat, exposure and dye),
as `rho = SUM_j m_j W_poly6`. Later iterations take an early return and leave that
value in `sorted_density`, so next substep's heat and dye terms use the real
`vol_j = m_j / rho_j` for GAS neighbours (an isolated STEAM particle has
`rho = m W(0)`, about 163, where rest_density is 0.6). Those later iterations
write `max(rest_density, 1)` for STATIC.

**Source**: `pbf_solver.cu:K_PBF_ComputeLambda`

### 12.3 Position Correction (Mass-Weighted Delta)

```
dx_i = SUM_j m_j * (lambda_i / rho0_i + lambda_j / rho0_j + s_corr / rho0_i) * grad_W_spiky(x*_ij)

|dx_i| = min(|dx_i|, 0.5 * h)    [safety clamp, rarely hit]
```

**Each lambda uses its own rest density (bd-r5-review-n28.81).** `lambda_j`
belongs to `C_j = rho_j / rho0_j - 1`, so its term divides by `rho0_j`. The
kernel computes it as `m_j * (lambda_i + lambda_j * rho0_i / rho0_j + s_corr) / rho0_i`.
For each pair, `m_i * dx_i + m_j * dx_j = 0` for any masses, so a WATER/SAND or
LAVA/WATER contact gets no net momentum from the correction. Dividing both
terms by `rho0_i` left `m_i m_j (lambda_i + lambda_j) (1/rho0_i - 1/rho0_j) gradW`
per pair: in one ComputeDelta on a compressed WATER/SAND block, `|SUM m_i dx_i|`
was 4.6% of `SUM |m_i dx_i|`, and it is now about 1e-9. Pairs with equal rest
density keep the ratio at exactly 1, so same-material scenes are unchanged. A
STATIC neighbour mirrors `lambda_i` and uses `rho0_i` for both terms. The
`s_corr` term still divides by `rho0_i` only, which is not momentum-free across
materials, but `s_corr_k = 0` by default. Test: `test_pbf_delta_rho0.py`.

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

**Reaction kicks go into x\*.** Under PBF, reactions run after `K_PBF_Predict`
has consumed `velocity`, and this step rebuilds velocity from positions only.
So `K_Reactions` (25 m/s gunpowder burst) and `K_BlastWave` (up to 15 m/s
radial push within h) add `dt * dv` to `predicted_position` wherever they add
`dv` to `velocity`. The burst is a plain store on the exploding particle. The
blast uses `atomicAdd` on x, y and z of each non-STATIC target, so STATIC keeps
x\* = x. The constraint iterations then start from the kicked x\*, and
`v_new = (x* - x_old) / dt` carries the kick. The Drucker-Prager `v_pred` (12.7)
reads `velocity` after the kick, so the kick is not treated as a constraint
correction. WCSPH and DFSPH pass NULL and are unchanged. Two limits:
- The final velocity is still clamped at `0.9 h / dt` (4.3 m/s at dt = 1/120),
  so the 25 m/s burst leaves the FIRE particle at the clamp.
- A kicked x\* can move several cells from where the grid put it, so the
  iterations of that substep miss some of its neighbours. The next full sort
  fixes it.

**World box walls in Finalize.** `K_PBF_Predict` and `K_PBF_ApplyDelta` clamp
x\* to `world_min + BOUNDARY_MARGIN` (and `world_max - BOUNDARY_MARGIN`), so a
particle that hits a wall arrives in `K_PBF_Finalize` sitting exactly on the
margin plane with `v_new` pointing into the wall. `pbf_box_boundary` applies the
4.5 impulse (restitution 0.3 on the normal velocity, Coulomb `wall_friction` 0.5
on the tangential velocity, 0 for FLUID) when `pos <= margin` and the velocity
points into the wall. Example: WATER arriving at the floor with
`v_new.y = -1.8 m/s` leaves with `+0.54 m/s`. A resting particle has
`v_new = 0` along the normal, so it is unchanged.

**No lambda warm start:** K_PBF_ComputeLambda computes lambda in closed form
from the current predicted positions every iteration and never reads a previous
value. `sorted_lambda_pbf` is a scratch buffer that carries lambda from
ComputeLambda to ComputeDelta within one iteration. It is not reordered by the
sort and not scattered back to unsorted order.

### 12.5 XSPH Viscosity (PBF)

```
v_new += c_scaled * SUM_j (m_j / rho_avg_ij) * (v_j - v_i) * W_poly6(x*_ij)

c_scaled = c_pbf.xsph_c * c_materials[mat_id].base_viscosity
```

`c_pbf.xsph_c` (default 0.05) is scaled by per-material `base_viscosity` so that
viscous fluids (OIL=5.0, LAVA=10.0) get proportionally more XSPH smoothing than
WATER (1.0). GRANULAR additionally multiplies by 10x for strong artificial viscosity.
Applied in K_PBF_Finalize.

GAS neighbours are skipped. A STONE (fixed wall) neighbour has `v_j = 0`, which
gives wall friction. A MAT_RIGID neighbour uses the body surface velocity
`v_b = v_body + omega x r` stored in the sorted velocity, with weight
`psi_b / rho_i` in place of `m_j / rho_avg`; the body gets the reaction
(22.5, PBF fluid-rigid drag).

**Source**: `pbf_solver.cu:K_PBF_Finalize`

### 12.5 PBF Multi-Material Handling

- **FLUID**: Full PBF constraint solve (density constraint + XSPH)
- **GRANULAR**: Participates in density constraint (lambda, dx) for volume exclusion.
  Drucker-Prager friction is applied in K_PBF_Finalize (velocity-space, once per
  substep) using pressure normals from K_PBF_ComputeLambda (see §12.7).
  High artificial viscosity prevents granular from flowing like water.
- **GAS**: No density constraint. Predict uses gravity + buoyancy + drag.
  GAS is a compressible phase and gets the WCSPH gas EOS pressure inside its
  own phase (bd-r5-review-n28.91). The first (do_heat) K_PBF_ComputeLambda
  call writes `lambda_i = p_i / rho_i^2`, with `p_i = k * max(rho_i - rho0, 0)`
  from `compute_pressure`, into the GAS lambda slot; later iterations keep it.
  K_PBF_ComputeDelta spreads the step2 acceleration over the iterations:
  ```
  a_i  = -fs * SUM_j(GAS) m_j * (lambda_i + lambda_j) * grad_spiky(x*_ij)
  dx_i = dt^2 * a_i / num_iterations        (fs = c_granular.force_scale = 0.02)
  ```
  so Finalize's `(x* - x) / dt` adds `dt * a_i` to the velocity, as WCSPH
  does. The 0.5h delta clamp still applies. GAS never pairs with FLUID,
  GRANULAR or STATIC here, so their lambda and delta are unchanged.
  Steam spawned by `K_SpawnGas` after Predict gets its predicted position
  written by the spawn (`x + dt * v`, see 10.21).
- **STATIC**: Lambda always forced to 0 (Akinci one-way boundary). STATIC
  contributes to neighbor density sums. On do_heat path, exposure/heat/dye
  are accumulated but lambda is still 0.

### 12.6 PBF Sleep System

PBF/DFSPH don't compute `shear_rate`, so sleep uses velocity-based activity:
```
activity = length(v_new)
sleep_counter = (activity < threshold) ? sleep_counter + 1 : 0
```

Only GRANULAR particles can sleep, matching WCSPH in `integrate.cu`. FLUID and
GAS reset their counter every substep. Before 2026-07-11 the PBF sleep block had
no behavior gate, so smoke froze mid-air and patches of a still water surface
froze.

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

// n points INTO the pile and the PBF corrections push OUT, so a load-bearing
// (compressed) particle has v_dot_n < 0.  Separating particles skip friction.
if v_dot_n < 0:
    v_n = v_dot_n * n           (normal component)
    v_t = v_new - v_n           (tangential component: sliding)

    max_tang = tan(phi_f) * |v_dot_n| + cohesion / dt

    if |v_t| > max_tang:
        v_t = v_t * (max_tang / |v_t|)

    v_new = v_n + v_t
```

The sign gate stops friction from freezing particles during free flight or
avalanche separation. Until 2026-07-11 the gate was `v_dot_n > 0`, which is the
separating case for this normal direction, so friction almost never fired and
`tan_phi_f` and `cohesion` had no effect in PBF.

Particles with no usable density gradient get the fallback normal `(0,0,0)`.
That fails the `n_len_sq > 0.5` validity check, so they skip friction. The old
fallback `(0,1,0)` passed the check and split velocity against an invented
up-normal.

#### 12.7.3 Parameters

Parameters are read from `c_granular` constant memory (uploaded per-solver with
different values for PBF vs WCSPH):

| Parameter | PBF Default | WCSPH Default | Description |
|-----------|-------------|---------------|-------------|
| `tan_phi_f` | 0.50 | 0.781 (= tan(38°)) | Friction cone slope when the material's `friction_coeff` is 0 |
| `cohesion` | 0.0 | 0.002 | Cohesive tangential allowance (metres per step) |

The friction slope in K_PBF_Finalize is the material's `friction_coeff` (SAND 0.6,
WET_SAND 0.75), with `tan_phi_f` as the fallback (bd-r5-review-n28.89). Cohesion
stays global. In the 5.3 column test the PBF piles spread to one layer for every
material, before and after this change, so the slope has no visible effect there.

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
Modifies `sorted_velocity` before integration uses it. The Jacobi ping-pong
uses its own buffer, `World.sorted_ist_scratch` (min(100K, max_particles) x float4,
allocated in the substep prologue when IST is on, outside any CUDA graph capture).
With an odd iteration count the result is copied back into `sorted_velocity`.
`sorted_veleval` is left as step2 wrote it (`v_pre + eps * xsph_sum`, from the
pre-IST velocity). K_Integrate (4.3) advects FLUID with
`vel_new + (veleval - vel)`, where `vel` is the post-IST velocity, so the IST
change cancels out of this substep's position update: positions move by
`v_pre + a*dt + eps * xsph_sum`. The IST change is kept in the velocity and
reaches positions from the next substep. With `sigma = 0`, IST on matches IST off.

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
  5. Divergence-free solver (warm-started from the previous substep's
     accumulated kappa_v, scaled by 0.5):
     Apply the warm-start correction, then N_div iterations:
     dkappa_v_i = max(div(v*_i), 0) * omega * alpha_i / dt   [unilateral]
     v*_i -= dt * SUM_j m_j * (dkappa_v_i/rho_i + dkappa_v_j/rho_j) * grad_W_ij
     kappa_v_i += dkappa_v_i                                  [warm-start total]
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

Kernels (`dfsph_solver.cu`, 8 in total): ComputeDensityAlpha (step 2),
NonPressureForces (3-4), ComputeKappaV and CorrectVelocityDiv (5),
ComputePressureAccel, DensitySolverUpdate and ApplyPressureVelocity (6), Finalize (7-8).
The Jacobi loop on p/rho^2 is the only density solver; the velocity-space density
solver (ComputeDensityAdv, ComputeKappa, ComputeKappaFromVelocity, CorrectVelocityDens,
PredictPosition) was removed. Iteration counts N_div and N_dens are host-side loop
counts in simulation.py and are not uploaded to `c_dfsph`.

### 13.2 Alpha Factor (Diagonal Preconditioner)

The alpha factor encodes how much a particle's density responds to pressure corrections:

```
grad_sum = SUM_j (m_j/rho_j) * grad_W_spiky(x_ij)     [vector sum]
grad_norm_sum = SUM_{j not STATIC} (m_j/rho_j)^2 * |grad_W_spiky(x_ij)|^2    [scalar sum]

denom   = |grad_sum|^2 + grad_norm_sum
alpha_i = denom > 1e-6 ? min(1 / denom, alpha_limit * h^2) : 0
```

**Alpha cap (bd-r5-review-n28.37).** `alpha_limit` is in units of `h^2` and does not
depend on dt: the Jacobi displacement per iteration is `residual * alpha * |gradW|`, which
has no dt in it. Interior alpha is `1.09e-3 = 0.68 h^2` (denom about 915 at h = 0.04,
spacing 0.02), and both shipped profiles use `alpha_limit = 1.0`, so the cap sits at about
1.5x the interior value. It binds only on particles with few neighbours (small denom).
A cap of 3 h^2 or more lets those surface particles take full-size Jacobi steps and the
free surface boils (KE of a settling 36K block at 3 s: 100 at 3 h^2, 23 at 1 h^2).
`test_dfsph_retune.py::test_dfsph_interior_alpha_not_capped` checks that interior alpha
equals 1/denom at both profiles.

The first term (`|grad_sum|^2`) is the squared norm of the vector sum of gradients.
The second term is the sum of squared norms. Both are accumulated in a single neighbor loop.

**Boundary neighbours (bd-r5-review-n28.108).** Every non-GAS neighbour enters `grad_sum`,
STATIC ones (STONE, METAL, MAT_RIGID) included. STATIC neighbours are left out of
`grad_norm_sum`: that term is neighbour j's own density response to kappa_i, and a STATIC
particle never moves under pressure. This matches Bender and Koschier 2015 and
SPlisHSPlasH `computeDFSPHFactor`. Interior alpha is unchanged. A WATER particle in contact
with a STONE floor gets about 2% larger alpha. In a settled 4000-particle pool in a
two-layer STONE box (default profile, 4 s), mean |rho/rho0 - 1| within 2h of the floor
moved from 0.281% to 0.267% and at the side walls from 0.32% to 0.28%, which is within
run-to-run noise. `test_dfsph_alpha_static.py` checks the GPU alpha against a host
recompute of this formula.

**Zero `density_in` (bd-r5-review-n28.110).** `rho_j` comes from `density_in`, which is
zero at program start and for slots that were never computed. A zero value falls back to
the neighbour's `rest_density` (floored at `RHO_EPSILON`), so `m_j / rho_j` is the rest
volume and the first-substep alpha matches a run with density pre-filled at rest. The
fallback used to be 1000 kg/m^3, which for WATER (2500) made `V_j` 2.5x too large and
alpha about 6.25x too small on that substep. `test_dfsph_density_fallback.py` checks that
alpha from zero or null `density_in` matches alpha from `density_in = 2500` on a WATER
block within 5% (measured difference 0).

For GRANULAR particles, alpha is multiplied by a compliance factor (0.7) to allow
some compression for granular packing without destabilizing the solver.

**Source**: `dfsph_solver.cu:K_DFSPH_ComputeDensityAlpha`

### 13.3 Divergence Solver

Enforces divergence-free velocity field:
```
div_v_i = SUM_j (m_j/rho_j) * (v*_i - v*_j) . grad_W_ij     [> 0 = compressing]
kappa_v_i = max(div_v_i, 0) * omega * alpha_i / dt      [0 if N_i < 20]
dv_i = -dt * SUM_j V_j * (kappa_v_i + kappa_v_j) * grad_W_ij,  V_j = m_j/rho_j
v*_i += dv_i
```

kappa_v is never negative, and a positive kappa_v pushes the pair apart.

**Pair form and momentum (bd-r5-review-n28.100, open).** The `V_j * (k_i + k_j)` pair
term here and in 13.4 gives j the reaction weight `V_i` instead of `V_j`, so linear
momentum is not conserved where rho_i != rho_j (free surfaces, material interfaces).
The symmetric form `m_j * (k_i/rho_i + k_j/rho_j)` conserves momentum, but with it
OIL sinks below WATER in `test_rest_density.py::test_oil_stays_on_water` (OIL COM_y
-0.982 against WATER -0.980 after 3 s; with the current form OIL -0.989 sits above
WATER -0.994), and the full-support DFSPH WATER cube density drops from 0.931 to
0.927 rho0. The form `2 m_j (k_i + k_j) / (rho_i + rho_j)` gives the same failures.
So the current form stays until the OIL/WATER interface is handled.

**Important**: The correction uses `kappa/rho` (not `kappa/rho^2`). This is the SPlisHSPlasH
formulation where alpha already absorbs one factor of 1/rho via the (m_j/rho_j) weighting
in its computation. Using the original paper's `kappa/rho^2` with this alpha would make
corrections ~rho times too weak (a critical bug we fixed).

Note: kappa_v **multiplies** by alpha (not divides). Since alpha = 1/denom, dividing
by alpha would multiply by the full denominator, producing astronomical corrections
for well-connected particles. `div_v` has the same sign as `drho` in 13.4: positive
means the particle is being compressed, and a positive kappa_v pushes the pair apart.

**Unilateral (bd-r5-review-n28.37).** `K_DFSPH_ComputeKappaV` clamps `div_v` to `>= 0`, as
SPlisHSPlasH does (`densityAdv = max(densityAdv, 0)`). Only compressing flow is corrected;
separating particles (splashes, a thinning sheet, the free surface) are left alone.
With the bilateral form and alpha at 1/denom the Jacobi iteration diverged: on a settling
36K WATER block the mean |div_v| rose by 50 to 170 percent per substep and KE stayed above
1400 (a boiling surface).

**Neighbour-count gate (bd-r5-review-n28.71).** `K_DFSPH_ComputeKappaV` takes
`normal_in` (sorted_normal from 13.2) and sets `kappa_v = 0` when its `.w` neighbour count
`N_i` is below 20 (`DFSPH_DIV_MIN_NEIGHBORS`), as SPlisHSPlasH does in 3D. With a truncated
kernel the divergence estimate is biased, so the solve is skipped. `N_i` counts every
non-GAS neighbour with mass, MAT_RIGID included (bd-r5-review-n28.68), so fluid pressed
against a rigid body keeps its solve. At spacing 0.02 and h = 0.04 an interior particle has
about 30 neighbours and a face particle about 19, so the outer layer of a block is gated;
in an 11^3 block 566 of 1331 particles are below 20. `normal_in = None` turns the gate off,
and `Simulation._run_dfsph_body` does not pass `normal_in` yet, so the gate is off in the running
simulation until that call is wired. The density solver (13.4) is not gated.

Iterations: 3 (Stable) or 2 (Fast). Each iteration requires two neighbor passes (compute
kappa_v, then correct velocity). On a settling 36K block at the Stable profile the
positive divergence drops by 72 to 81 percent per substep (warm start plus 3 iterations).
Jacobi removes local modes fast but the block-wide mode slowly: 8 iterations reach 91
percent and 16 reach 94 percent.

**Warm-starting (bd-r5-review-n28.83):** the warm start is the accumulated stiffness,
as in SPlisHSPlasH. `sorted_kappa_v` holds the running total and is persisted across
substeps through the sort-reorder pipeline (kappa_v → sorted_kappa_v). At the start of
each divergence solve the total is scaled by `dfsph_div_warm_start` (default 0.5) and
applied once as a velocity correction (with warm start 0 it is zeroed instead). Each
iteration then writes its increment `dkappa_v` to a scratch buffer
(`Simulation._kappa_v_delta`, allocated outside the CUDA graph body),
`K_DFSPH_CorrectVelocityDiv` applies only that increment, and `K_DFSPH_ComputeKappaV`
adds it to the total at the particle's own index (no race, one extra 4 B write per
particle per iteration). STATIC, GAS and sleeping particles get a total of 0.
K_DFSPH_Finalize writes the total back to the unsorted buffer.

Measured on a 36K WATER block dropped to the floor, Stable profile, 2 s, mean positive
`div_v` left after the warm-start pass (t >= 1 s): 3.54 with no warm start, 3.36 when the
warm start was the last increment only (the previous behaviour), 3.01 / 2.69 / 2.38 with
the accumulated total at warm start 0.25 / 0.35 / 0.5. No NaN and no v_max growth at any
of these; 0.5 stays the default. In `eval_solvers.py` the DFSPH Dam Break spread moved
from max_x = -0.157 to -0.180 (WCSPH -0.226); the pass count is unchanged.

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
density_adv = (rho_i/rho_0) * (1 + dt * drho)  [predicted rho*/rho_0]
residual = density_adv - 1.0             [positive = compressed]
p_rho2 = max(p_rho2 + omega * residual * alpha / dt^2, 0)
```

**Density prediction units (bd-r5-review-n28.26)**: `drho` is volume weighted
(`V_j = m_j/rho_j`), so its unit is 1/s and it is already the relative rate
`(1/rho) Drho/Dt`. `dt * drho` is therefore the dimensionless relative density change,
and `rho*/rho_0 = (rho_i/rho_0) * (1 + dt * drho)`. `drho` must not be divided by
`rho_0` again: that makes the `A*p` feedback `rho_0` times too small (2500x for WATER),
the solver then ignores both the velocity divergence and the pressure it has already
applied, and `p_rho2` grows linearly with the iteration count. The divergence solver
(13.3) uses the same volume form with no `/rho_0`.

`alpha_i` (13.2) is the inverse diagonal of this residual: a unit `p_rho2` on one
interior particle lowers its own predicted residual by `1.02 * dt^2 / alpha_i`
(`test_solvers.py::test_dfsph_density_convergence`).

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

**Alpha is not capped in the interior** (13.2): each Jacobi iteration takes `omega` of the
exact diagonal step. Before bd-r5-review-n28.37 the cap was `alpha_limit * dt^2`, which at
dt = 1/300 held interior alpha at 0.1x of 1/denom.

**Uncapped behaviour** (alpha_limit=1e9, 9x9x9 WATER block, uniform compression
`v = -k (x - c)` with `3 k dt = 0.1`, mean complementarity residual over all particles):

| iteration   | 0     | 1     | 2     | 3     | 4     | 8     |
|-------------|-------|-------|-------|-------|-------|-------|
| omega = 1.0 | 5.2%  | 1.6%  | 1.1%  | 0.8%  | 0.9%  | 2.0%  |
| omega = 0.7 | 5.2%  | 2.6%  | 1.5%  | 0.8%  | 0.5%  | 0.11% |

omega=1.0 with the exact diagonal overshoots the short-wavelength modes and the
residual grows back after about 4 iterations. omega=0.7 converges monotonically.
On a 20x20x20 block the smooth (block-wide) mode converges more slowly, because a
uniform `p_rho2` produces no net acceleration inside the fluid: at omega=0.7 the
residual falls from 7.8% to 2.1% in 8 iterations.

**Measured effect of the density formula fix** (bd-r5-review-n28.26, with the old
dt^2 alpha cap and 12 / 8 iterations of the profiles at that time, 36K WATER cube
from `eval_review_fixes.py` test A, mean `|rho/rho0 - 1|` over FLUID particles):

| profile      | formula              | 1 s (impact) | 2 s   | 3 s  | 5 s (settled) |
|--------------|----------------------|--------------|-------|------|---------------|
| DFSPH        | `(rho_i/rho0)(1+dt*drho)` | 9.6%    | 11.7% | 5.2% | 4.2%          |
| DFSPH        | extra `/rho0` on drho | 29.1%       | 14.7% | 6.6% | 4.9%          |
| DFSPH (Fast) | `(rho_i/rho0)(1+dt*drho)` | 10.2%   | 10.8% | 6.6% | 4.1%          |
| DFSPH (Fast) | extra `/rho0` on drho | 27.3%       | 13.9% | 7.9% | 6.2%          |

No NaN in either case. `eval_solvers.py` gives 67/72 with both formulas; the same five
checks fail with both (WCSPH Fluid Settles, PBF and DFSPH Sand Friction, Dam Break COM_y,
KE Decay Ratio).

**Profiles (bd-r5-review-n28.37).** omega 0.7 for both solvers (omega 1.0 still overshoots
the short-wavelength modes, 13.4 table above), velocity limit 0.4*h/dt (0.9 left surface
particles at 9.9 m/s after 3 s), pressure acceleration clamped to h/dt^2 in
K_DFSPH_ApplyPressureVelocity, warm start 0.5 for both solvers, alpha_limit 1.0.

| profile      | dt    | div + dens iters | sweeps / substep | sweeps / sim s | ms / sim s at 1M |
|--------------|-------|------------------|------------------|----------------|------------------|
| DFSPH        | 1/300 | 3 + 4            | 18               | 5400           | 11600            |
| DFSPH (Fast) | 1/200 | 2 + 3            | 14               | 2800           | 6500             |
| old DFSPH    | 1/300 | 3 + 12, dt^2 cap | 34               | 10200          | 20100            |
| old Fast     | 1/300 | 2 + 8, dt^2 cap  | 24               | 7200           | 13400            |

Sweeps count density/alpha, non-pressure forces, the div warm-start pass, 2 per div
iteration, the initial a_press pass and 2 per density iteration. The ms column is the
median substep time from `bench_substep.py` at 992K particles divided by dt, on a GPU shared
with other jobs, so treat it as about +-10 percent. `SolverProfile(solver_type=DFSPH)`
defaults to the Stable values, and `velocity_limit_factor` defaults to 0.4 for DFSPH.

Settling 36K WATER block (`run_dfsph_retune.py`, 0.6 x 0.8 x 0.6 m dropped 0.1 m).
comp is the mean of `max(rho/rho0 - 1, 0)` over FLUID; KE in J:

| profile      | comp 1 s | comp 3 s | KE 1 s | KE 3 s | div reduction 3 s |
|--------------|----------|----------|--------|--------|-------------------|
| DFSPH        | 0.89%    | 0.35%    | 284    | 23     | 81%               |
| DFSPH (Fast) | 3.51%    | 0.89%    | 292    | 24     | 77%               |
| old DFSPH    | 0.49%    | 0.40%    | 432    | 98     | 62%               |
| old Fast     | 1.41%    | 0.70%    | 447    | 127    | 37%               |

The sweep over dt {1/300, 1/200, 1/150}, div {2, 3, 4} and dens {3, 4, 6} showed that dt
sets the compression error: 0.3 to 0.4% at 1/300, 0.8 to 1.2% at 1/200, 1.7 to 2.5% at
1/150, with little change from more density iterations. dt = 1/120 to 1/150 (PROBLEMS.md C3)
needs more than 6 iterations to stay under 1%. Fast trades a 3.5% peak at impact for half
the cost.

`eval_solvers.py` goes from 67/72 to 69/72: DFSPH Sand Friction and Dam Break COM_y now
pass (DFSPH COM_y -0.498 to -0.603, WCSPH -0.601). Still failing: WCSPH Fluid Settles,
PBF Sand Friction and KE Decay Ratio (DFSPH 5.71 to 2.89, PBF 2.58; the ratio compares
final KE with the peak over the first 0.3 s, before the block lands).

**Source**: `dfsph_solver.cu:K_DFSPH_ComputePressureAccel`, `K_DFSPH_DensitySolverUpdate`,
`K_DFSPH_ApplyPressureVelocity`, `simulation.py` (step 5, density solver loop)

### 13.5 Non-Pressure Forces

```
FLUID:    a_i = gravity + (1/rho_i) * [45/(pi*h^6)] * SUM_j mu_ij * m_j*(v_j-v_i)/rho_j*(h-|r|)
          mu_ij = 2*mu_i*mu_j/(mu_i+mu_j)   (per-material base_viscosity)
GRANULAR: a_i = gravity + (1/rho_i) * [45/(pi*h^6)] * SUM_j eta_ij * m_j*(v_j-v_i)/rho_j*(h-|r|)
          GRANULAR neighbor:     eta_ij = 2*eta_i*eta_j/(eta_i+eta_j)
                                 eta_k = compute_muI_eta(shear_rate[k], p_eff_k, rho_k)
                                 p_eff_k = max(p_prev[k] * rho_k, rho0_k * |g| * spacing)
          non-GRANULAR neighbor: eta_ij = mu_0   (c_precalc.viscosity_precalc)
GAS:      a_i = gravity + buoyancy - drag * v_i + viscosity (same as FLUID)
          - fs * SUM_j(GAS) m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_spiky(r_ij)
          p = k * max(rho - rho0, 0)   (gas EOS, compute_pressure)
          fs = c_granular.force_scale = 0.02
```

**GRANULAR confining pressure** (bd-r5-review-n28.92). `p_prev` is
`sorted_kappa`: the density solver's converged value from the previous
substep, read before the warm-start scale. The solver stores p/rho, not
p/rho^2: ComputePressureAccel uses `V_j * (p_i + p_j)` with `V_j = m_j/rho_j`,
which matches `m_j * (p_i/rho_i^2 + p_j/rho_j^2)` when the stored value is
p/rho. So `p = p_prev * rho` (Pa); `p_prev * rho^2` would be about 4000x too
large and pin eta at mu_max. Tait EOS is not used here, because the density
solver drives rho to rho0 and Tait pressure would then measure solver error
instead of load. The floor is the WCSPH step2 one-layer lithostatic pressure
(SAND at g = 4: 320 Pa). It also covers the first substep and woken
particles, where `p_prev = 0`. The current substep's pressure is not known
yet when this kernel runs.

Each particle's eta uses its own stored shear rate (from ComputeDensityAlpha)
and its own pressure, so eta_ij computed by i equals eta_ji computed by j and
the GRANULAR pair force is antisymmetric (`test_dfsph_mui_friction.py`).

Sand Castle, 5 s, g = 4 (`eval_dfsph_mui.py`): mean GRANULAR eta rises with
depth, 300 Pa.s in the top 0.14 m to 450 Pa.s at 0.64 m depth, with mean
p_eff 970 Pa to 8.4 kPa. The pile height is 0.71 m (0.73 m with the Tait
feed). WCSPH spreads the same preset to 0.17 m height in 5 s at eta = mu_max,
so the DFSPH and WCSPH piles do not match; that gap is outside this term.

The GAS pressure term is the WCSPH step2 gas pressure (bd-r5-review-n28.91).
`grad_spiky` carries the negative coefficient, so `-grad_spiky` points away
from j and the term is repulsive. `fs` keeps the magnitude equal to WCSPH;
without it the repulsion would be 50x stronger. The ACCEL_MAX clamp applies.

The `1/rho_i` factor is critical: without it, viscosity is ~2500x too strong (at rho≈2500),
making the fluid behave like a solid. With WATER `base_viscosity = 1.0` and `rho_i ≈ 2500`,
the effective kinematic viscosity is `nu = mu/rho_i = 0.0004 m^2/s` (400x physical water,
giving smooth game fluid behavior).

The FLUID/GAS term is the same number and pair formula as WCSPH step2 (3.3,
bd-r5-review-n28.79): `mu_ij * lap * m_j * (v_j - v_i) / (rho_i * rho_j)`.

Velocity updated: `v*_i = v_i + dt * a_i`

**Source**: `dfsph_solver.cu:K_DFSPH_NonPressureForces`

### 13.6 DFSPH Multi-Material Handling

- **FLUID**: Full DFSPH (divergence + density solvers)
- **GRANULAR**: Participates in both solvers. Alpha multiplied by 0.7 compliance factor.
  mu(I) viscosity against GRANULAR neighbours in non-pressure forces, with the
  solver pressure as confining pressure (13.5). Shear_rate computed in
  ComputeDensityAlpha.
- **GAS**: Skips the divergence and density solvers. Non-pressure forces
  include buoyancy, drag and the WCSPH gas EOS pressure from GAS neighbours
  (13.5). A 1000-particle SMOKE cloud at zero gravity reaches the same density
  and radius under WCSPH, DFSPH and PBF within 2% after 2 s
  (`eval_gas_repulsion.py --free`).
- **STATIC**: Skips all solver kernels, contributes to neighbor density sums.
  `K_DFSPH_ComputeDensityAlpha` writes `max(rest_density, 1)` as its density
  and divides its heat rate by that value, as WCSPH and PBF do (11.1). The
  solver terms (alpha `grad_sum`, `div_v`, `drho`, XSPH) give a regular STATIC
  neighbour the Akinci volume `m_j / rho0_i` instead (`dfsph_vol_rho`).

### 13.7 DFSPH Sleep System

Same velocity-based activity metric as PBF (§12.6). Only GRANULAR particles can
sleep, matching WCSPH and PBF. FLUID and GAS reset their counter every substep.
A FLUID or GAS particle that arrives with the SLEEPING bit set (older build or
loaded snapshot) wakes in the same substep; a sleeping GRANULAR particle stays
asleep while `|v| <= V_WAKE`.

Box-wall friction is 0 for FLUID and `c_sim.wall_friction` for other behaviours,
as in the other two solvers (§4.5).

Before 2026-09-24 the DFSPH sleep block had no behavior gate and FLUID used the
full wall friction. A 2 s resting pool of 19200 water particles was then fully
asleep; now none are. Dam Break COM_y is unchanged (-0.603).

**Source**: `dfsph_solver.cu:K_DFSPH_Finalize`

---

## 14. Thermal Convection (Boussinesq Buoyancy)

### 14.1 Formula

For FLUID particles with non-zero `thermal_expansion` (β), along the live gravity
vector `g = c_sim.gravity`:
```
a_buoy = -g * β * (T - T_ambient)
```

Where:
- β = `c_materials[mat_id].thermal_expansion` (1/K)
- T_ambient = 293 K
- g = `c_sim.gravity` (default (0, -4, 0), UI slider -20 to 5 on y)

From the Boussinesq approximation: `ρ_eff = ρ_0 * (1 - β*(T-T_0))`.
Hot fluid (T > T_0) has lower effective density, so it moves against gravity;
cold fluid (T < T_0) moves with it. The size scales with |g|, and the direction
follows the gravity vector, including a flipped sign.

Code: `boussinesq_accel(temp, beta)` in `sph_shared.cuh`.

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
- **PBF**: Applied in `K_PBF_Predict` as a velocity term (v* += a_buoy*dt), once per substep
- **DFSPH**: Applied in `K_DFSPH_NonPressureForces` alongside gravity

**Source**: `integrate.cu`, `pbf_solver.cu:K_PBF_Predict`, `dfsph_solver.cu:K_DFSPH_NonPressureForces`

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
omega_i = SUM_j (m_j / rho_j) * grad_i W_spiky(r_ij) × (v_j - v_i)
```

`r_ij = pos_i - pos_j` and `spiky_grad_coeff < 0`, so `grad_i W` is the true
gradient with respect to `pos_i` and points toward the neighbour. The operand
order matters: `(v_j - v_i) × grad_i W` is `-curl(v)`. For rigid rotation
`v = Omega × r` the result is `omega = 2 * Omega`. `test_vorticity_sign.py`
checks this on a 7^3 lattice (spacing 0.02, h 0.04): interior `omega_z` is
1.90 to 1.91 (WCSPH), 1.92 to 2.31 (PBF) and 1.91 to 2.03 (DFSPH) for
`Omega_z = 1`.

PBF and DFSPH use `dv = v_i - v_j` in their loops, so they compute the same sum
as `dv × grad_i W`.

Output: `vorticity_out[i] = float4(omega_x, omega_y, omega_z, |omega|)`

The magnitude `|omega|` is stored in `.w` for use by the Step2 confinement force.

**Source**: `step1.cu` (K_Step1 and K_Step1_BuildNL, FLUID branch),
`pbf_solver.cu` (K_PBF_ComputeLambda), `dfsph_solver.cu`
(K_DFSPH_ComputeDensityAlpha).

### 16.2 Confinement Force (Step2)

The vorticity confinement force re-injects energy lost to numerical dissipation:

```
eta_i = SUM_j (m_j / rho_j) * (omega_j - omega_i) * grad_W_spiky(r_ij)
N_i = eta_i / |eta_i|                          [normalized vorticity gradient]
f_conf = epsilon * (N_i × omega_i)              [confinement force]
```

Where `epsilon = c_granular.vorticity_epsilon` (default 0.05).

The sum for eta runs over FLUID neighbours only, in WCSPH (K_Step2) and PBF
(K_PBF_Finalize). STATIC (including MAT_RIGID), GAS and GRANULAR neighbours
have `omega_j = 0` and a `rho_j` that is not a fluid density: a MAT_RIGID
neighbour had weight `psi_b / 0.01` (about 5e4 times a water neighbour) and a
GAS neighbour about 2500 times. Included, they point N at the body or gas and
turn the force into a tangential push along the boundary.
`test_vorticity_eta_fluid_only.py` checks that an extra rigid, gas or sand
neighbour leaves the force unchanged.

The gradient uses the **difference form** `(omega_j - omega_i)` (not `omega_j`
alone). This is the SPH approximation of the gradient of a scalar field, which
vanishes identically when the vorticity field is uniform -- preventing spurious
confinement forces in solid-body rotation. The difference form is now unified
across all three solvers (WCSPH, DFSPH, PBF).

The confinement force is perpendicular to the vorticity axis and aligned with the
gradient of vorticity magnitude, amplifying existing rotational structures without
creating spurious rotation.

**Source**: `step2.cu` (K_Step2 main neighbour loop, after the behaviour
read), `pbf_solver.cu` (K_PBF_Finalize XSPH loop)

### 16.3 Solver Support

- **WCSPH**: Full vorticity computation in Step1 + confinement in Step2 (epsilon=0.05)
- **PBF**: Vorticity computed in K_PBF_ComputeLambda, confinement applied in K_PBF_Finalize
  as velocity correction `v += dt * eps * (N x omega)`. PBF epsilon=0.001 (50x lower than
  WCSPH) because PBF's fixed dt=1/120 lacks CFL safety; higher values inject energy.
- **DFSPH**: Vorticity computed in K_DFSPH_ComputeDensityAlpha, confinement applied in
  K_DFSPH_NonPressureForces as acceleration. Uses same epsilon as WCSPH (0.05).

The epsilon values were kept when the vorticity sign was fixed. Before the fix
the force pointed against the flow and drained vortex cores. With the right sign
it amplifies them. After the fix, the `test_odd_viscosity.py` Taylor-Green run
(WCSPH, 300 substeps) shows net L_z drift of 1.3e-7 (eps 0), 2.2e-7 (eps 0.05)
and 1.5e-7 (eps 0.10), against a noise floor of 5e-4. Transverse RMS v_x is
2.822e-3 with and without confinement. So these values do not inject
measurable angular momentum. A visual A/B at eps 0.01 and 0.05 has not been
done yet.

---

## 17. Akinci Surface Tension

### 17.1 Surface Normal Computation (Step1)

Surface normal accumulated in Step1 neighbor loop for FLUID particles:
```
n_i = SUM_j V_j * grad_W_spiky(r_ij)
      V_j = m_j / rho_j          for fluid, granular and STONE neighbours
      V_j = m_j / RIGID_PSI_RHO0 for MAT_RIGID neighbours (= V_b, see 22.3)
neighbor_count_i = count of non-self neighbours within h with m_j > 0
```

MAT_RIGID neighbours count in both sums, so fluid resting on a rigid body is
interior like fluid resting on STONE (bd-r5-review-n28.68). DEAD slots (mass 0)
are not counted. For a water block on a wall at spacing 0.02, h = 0.04, the
interior contact layer has 31 neighbours on a three-plane STONE wall, 30 on a
one-plane rigid shell and 31 on a three-plane rigid slab; with the rigid wall
left out it had 21, so it was classed as surface and its normal pointed away
from the body at free-surface strength. The rigid contact normal is not zero:
psi_b of a one-plane shell is about 1.6 V, so n points into the body at about
0.6 of a free-surface normal (0.2 for a three-plane slab). No force follows
from it while nc >= 25, because every consumer gates on nc < 25.
Vorticity, strain rate, heat, exposure and dye still skip MAT_RIGID.
PBF (K_PBF_ComputeLambda) and DFSPH (K_DFSPH_ComputeDensityAlpha) use the same rule.

Output: `normal_out[i] = float4(n_x, n_y, n_z, as_float(neighbor_count))`

The neighbor count is packed into `.w` as a reinterpreted float for efficient storage.

**Source**: `step1.cu` (K_Step1 and K_Step1_BuildNL neighbor loops),
test: `test_step1.py::test_rigid_wall_contact_layer_matches_stone`

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

The RNG seed is `Simulation._substep_counter`, which is unique per substep.
Foam generation runs every substep, and the old seed (`_frame_counter`) repeated
across all substeps of a frame, so each frame produced correlated, duplicated
foam at a multiple of the intended rate.

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

Launch size: `min(2 * last_foam_count + 1024, max_foam)` threads, where
`last_foam_count` is the deferred readback (section 20.5). The kernel reads the
live count from device memory. Foam that grows faster than the 2x headroom in
the readback lag gets no physics for those frames (seen for 2 frames when the
Dam Break splash fills the 200K pool in one frame).

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
ComputePressureAccel, DensitySolverUpdate)

### 20.3 Pre-computed Pressure Array (PERF-007)

WCSPH Step2 no longer calls `compute_pressure()` per neighbor (~50 calls/particle).
Instead, `K_ComputePressure` runs once after Step1 to write per-particle pressure
to `sorted_pressure[]`. Step2 reads `p_i` and `p_j` via `__ldg(&pressure_in[i])`.

This eliminates the per-neighbor `powf()` call (Tait EOS) from the Step2 inner loop.

Pipeline: Step1 → K_ComputePressure → Step2

**Source**: `step2.cu:K_ComputePressure`, `step2.py:compute_pressure()`,
`simulation.py:_run_wcsph_body`

### 20.4 Warm-start kappa_v (PERF-008) and PBF reorder skips

The DFSPH divergence solver's kappa_v is carried through the counting sort
pipeline (K_ScatterReorder, K_GatherReorder), persisted via K_DFSPH_Finalize
writeback, and warm-started at the beginning of each substep. The stored value is
the sum of the scaled warm start and every divergence iteration's increment
(13.3). It is scaled by `dfsph_div_warm_start` (default 0.5), then applied as an
initial velocity correction before the divergence solver iterations.

PBF has no warm-start state (see 12.4). PBF also never reads kappa, kappa_v or
the FP16 mirrors (sorted_velocity_h, sorted_temperature_h, sorted_dye_h), so
`_run_grid_setup` passes None for all five when the active solver is PBF, and
both reorder kernels skip the gather and the store for a NULL output. The
pointer set depends only on the solver, and a solver switch recaptures the
CUDA graph. Measured at 992K particles (PBF Dam Break, full sort each
substep, shared GPU): the sort stage median went from about 4.0 ms to about
2.4 ms.

**Source**: `counting_sort.cu` (kappa_v_in/out, NULL-guarded outputs),
`dfsph_solver.cu:K_DFSPH_Finalize`, `simulation.py:_run_grid_setup`,
`simulation.py:_run_dfsph_body` (warm-start application)

### 20.5 Grid Reuse (all solvers, PERF-009)

A gather-only substep (`K_GatherReorder`) re-gathers the unsorted arrays
through the last sort permutation and reuses `cell_start`/`cell_end`, so
particles stay binned by their position at the last full sort. K_Integrate,
K_PBF_Finalize and K_DFSPH_Finalize track the drift since that sort in
`max_displacement` via atomicMax. The sort-skip decision uses solver-specific
thresholds:

- **WCSPH**: (0.25*h)² threshold, max 4 consecutive skipped frames
- **PBF/DFSPH**: (0.15*h)² threshold, max 2 consecutive skipped frames
  (tighter for constraint solver convergence quality)

**Drift anchor (bd-r5-review-n28.54).** On each full sort `K_ScatterReorder`
writes the sorted position into `sorted_anchor` (float4, N). `K_GatherReorder`
never writes it, and the sort permutation is frozen between full sorts, so
`sorted_anchor[i]` stays the sort-time position of the particle in sorted slot
i. Any change of n or of the particle set forces a full sort, which rewrites
the anchor. The finalize kernels compute

```
disp_sq = |x_new - sorted_anchor[i]|^2
max_displacement = atomicMax over particles and substeps since the last full sort
```

so `max_displacement` is the largest drift of any particle from the position
it was binned at. A full sort zeroes it; resting particles read exactly 0.
PBF bins by the predicted position x* but measures drift on x; the two differ
by dt*(v - v_sort), which is small next to the threshold. Cost: one coalesced
16 B read per particle per substep (about +23 us in integrate at 1M) and one
16 B store per particle per full sort.

**Rigid boundary drift (bd-r5-review-n28.65).** Rigid boundary particles are
STATIC, so the finalize kernels skip them, but `K_UpdateBoundaryParticles`
moves them every substep. That kernel records their drift instead, against a
per-boundary-particle anchor in unsorted order (`rbm.boundary_anchor`, float4
per boundary particle, same index as `boundary_data`). The body moves after
the solver, so the kernel extrapolates one substep to match the fluid, which
records its drift after it moves:

```
x_end   = 2 * x - x_old          (x_old: the slot's position from the last substep)
disp_sq = |x_end - anchor[i]|^2  -> atomicMax into max_displacement
```

On a substep that runs a full sort, `_update_rigid_boundary` clears
`max_displacement` before the kernel and sets `anchor[i] = x`, and
`_run_grid_setup` skips its own clear, so the boundary drift of that substep
is kept. Rigid bodies disable CUDA graphs, so a captured graph always keeps
the clear in `_run_grid_setup`. A block with no anchor yet (after injection,
reinjection or reset) makes the substep a full sort. A body at 2 m/s with
dt = 1e-3 and 10 substeps per frame moves 20 mm (0.5h) per frame and sorts
every frame; a body at rest adds zero drift and calm water keeps skipping.
Cost: one extra 16 B read of the old position and one 16 B anchor access per
boundary particle per substep.

**Miss bound.** Cell size equals h and the search covers the 27 cells around
the current position of particle i, which reaches at least h from x_i. A
neighbour j is binned at its anchor, within d of x_j, so it is found whenever
r_ij + d <= h. With d < 0.25h only pairs with r_ij > 0.75h can be missed; W_poly6
there is at most (1 - 0.75²)³ = 8.4% of its peak.

**Prediction.** The latched drift is up to two frames old: it was copied at the
end of the previous frame, and the decision covers the next frame. The host
counts substeps since the last full sort (`_count_substep_sort`, from the skip
flag and the n guard, so graph replays are covered) and stores the count with
each copy. The decision scales the drift to the end of the next frame:

```
k_copy = substeps since the last sort when the copy was taken
k_next = substeps since the last sort now + substeps of this frame
predicted_sq = max_disp_sq * (k_next / k_copy)^2
skip if predicted_sq < threshold and the frame cap is not reached
```

A settled scene keeps a near-zero rate and skips. A front moving faster than
about 0.25h per frame (0.6 m/s at 60 fps and dt = 1e-3) sorts every frame.
Measured with `bench_substep.py` (WCSPH, 992K dam break, 17 substeps per
frame, first 300 substeps): skip rate 55% (83% before the anchor, when only
the frame cap ended a skip run), and the final position checksum matches the
sort-every-substep run.

**Decision data (bd-r5-review-n28.46).** `max_displacement` holds the float
bits of the largest squared drift since the last full sort. At the end of each
frame `_update_sort_skip_decision` does three things in this order, with no
host sync:

1. Latch. Each frame's D2H copy of `max_displacement` and `foam_count` has its
   own event, queued with a scene epoch. The host pops every event that is done
   (non-blocking query) and reads the pinned buffers if the newest landed copy
   is from the current epoch. When the host keeps pace with the GPU this is the
   previous frame's copy, so the lag is exactly one frame. When the host runs
   ahead it is the newest copy that has landed. If none has landed, the
   previous value is kept.
2. Decide, using the prediction above. A value at or above the +inf bits
   (`0x7F800000`) never skips. The
   startup value and the value after `notify_scene_changed()` is the sentinel
   `0xFFFFFFFF`, and `notify_scene_changed()` bumps the epoch, so copies
   launched before a scene change are never used.
3. Launch this frame's copies on the readback stream and queue their event.

At the top of `step_frame` the compute stream waits on the newest copy's event
(`wait_event`, device side), so the next full-sort memset of
`max_displacement` and the foam kernels cannot run before or during the copy.
`K_SpawnGas` writes the +inf bits into `max_displacement` when it spawns steam,
because the spawned slots jump to the source position without Integrate
seeing it, so the next frame does a full sort.

**Source**: `counting_sort.cu:K_ScatterReorder` (anchor_out),
`integrate.cu:K_Integrate`, `pbf_solver.cu:K_PBF_Finalize`,
`dfsph_solver.cu:K_DFSPH_Finalize` (max_displacement_out, displacement_anchor),
`world.py` (`sorted_anchor`),
`integrate.cu:K_UpdateBoundaryParticles` (rigid boundary drift),
`rigid_bodies.py` (`boundary_anchor`, `boundary_anchor_valid`),
`simulation.py:_update_rigid_boundary`, `_count_substep_sort` and
`_update_sort_skip_decision` (latch, prediction and threshold logic),
`spawn.cu:K_SpawnGas` (+inf on spawn)

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
- `nu_t` = micropolar coupling viscosity, 0.1, hardcoded in integrate.cu, pbf_solver.cu and
  dfsph_solver.cu. `SolverProfile.micropolar_nu_t` exists but is not read yet.
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

Transitions are driven by `exposure_corrode` (§9.4), which is the pair's
reaction_rate times the share of the kernel support held by WATER neighbours:

```
SAND  --[exp_corrode > 0.2]--> WET_SAND --[exp_corrode > 0.5]--> MUD
SAND  <--[exp_corrode < 0.02]-- WET_SAND <--[exp_corrode < 0.1]-- MUD
          (drying)                          (drying)
```

With the rates in §21.2 these are water fractions:

| Transition | Rate | Water fraction |
|------------|------|----------------|
| SAND → WET_SAND | 0.4 | above 50% |
| WET_SAND → MUD | 0.6 | above 83% |
| MUD → WET_SAND (drying) | 0.3 | below 33% |
| WET_SAND → SAND (drying) | 0.6 | below 3.3% |
| DIRT → MUD | 0.5 | above 40% |

A sand castle touched by a few water particles stays SAND; only sand that
water mostly surrounds turns wet, and only near-submerged wet sand turns to MUD.

Drying is accelerated by temperature above 293K:
```
effective_threshold = threshold / (1 + 0.01 * max(temp - 293, 0))
```

Hysteresis prevents oscillation (wet threshold >> dry threshold).

Temperature-driven transitions, including WATER → ICE below 263 K and LAVA →
GRAVEL below 1000 K, are listed in §9.5.

FIRE is put out the same way. When FIRE's `exposure_corrode` is above
`FIRE_EXTINGUISH_EXPOSURE = 0.3`, K_Reactions turns it into SMOKE/GAS with
lifetime 1 s and temperature 373 K, before the GAS lifetime decay. The
WATER/FIRE rate is 0.8, so this happens when more than 37.5% of the fire's
kernel support is water. Only WATER feeds FIRE's `exposure_corrode` (see the
one-way overrides in §21.2), so fire on wood with no water burns for its full
lifetime.

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
| WATER + ICE | 0.0 | 20.0 | Heat only: water warms ice |
| LAVA + ICE | 0.0 | 60.0 | Heat only: lava melts ice |
| FIRE + ICE | 0.0 | 25.0 | Heat only: fire melts ice |
| ICE + STEAM | 0.0 | 15.0 | Heat only: steam melts ice |
| LAVA + METAL | 0.0 | 30.0 | Heat only: lava heats metal |
| WATER + ACID | 0.0 | 5.0 | Heat only: acid is not consumed by water |

Rule: `exposure_corrode` is the W-weighted mean of `reaction_rate` over every neighbour (§9.4), and
K_Reactions reads it as acid damage, wetting and acid self-consumption. So a
pair has a non-zero `reaction_rate` only when ACID, WATER, MUD or a combustible
(WOOD, OIL, GUNPOWDER) is on one side. Heat-only pairs use rate 0 and a
`heat_exchange`. `test_reactions.py` checks this rule on the full matrix.

The matrix is symmetric except for `_ONE_WAY_RATE_OVERRIDES` in `materials.py`,
which set one direction's `reaction_rate` only. Row `i` is what particle `i`
reads. The FIRE row has rate 0 for WOOD, OIL, GUNPOWDER and ICE, so those
neighbours do not count as wetting fire. The reverse rows (WOOD, OIL and
GUNPOWDER reading FIRE) keep 0.9, 1.0 and 0.9, and `heat_exchange` stays
symmetric, so ignition is unchanged.

### 21.3 Corrosion Guard

Sand family materials (SAND, WET_SAND, MUD) are excluded from the corrosion
health damage path. Their `exposure_corrode` signal is used only for wetting
transitions, not health reduction.

Corrosion damage (`health -= exposure_corrode * dt`) applies only to STONE,
METAL, WOOD, GRAVEL, OIL and GUNPOWDER. `exposure_corrode` is rate × acid
support fraction, so a particle fully surrounded by acid dies after 1/rate
seconds (STONE 0.15 → 6.7 s, METAL 0.3 → 3.3 s). ICE is not on the list: it changes
only by melting to WATER above 273 K. With the heat-only pairs above at rate 0,
these materials get corrode exposure only from ACID (and, for the combustibles,
from FIRE and LAVA, whose ignition check runs first).

When health reaches 0 the particle becomes DEAD/STATIC with health 0, mass 0
and lifetime 0, and its index goes on the spawn freelist. It does not turn
into FIRE: `exposure_heat` has no temperature threshold, so any hot FIRE
particle next to WOOD, OIL or GUNPOWDER ignites it. A DEAD slot has an
all-zero interaction row, so it gives neighbours no heat exposure.

Known limit: the corrosion branch still depends on the table keeping rate 0
for non-acid pairs on the whitelist. A separate acid exposure channel would
remove that dependence.

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

**Volume precomputation** (V_b): Each boundary particle's volume is:
```
V_b = 1 / SUM_k W(r_b - r_k, h)
```
where the sum is over neighboring boundary particles of the same body. Akinci's
boundary weight is `psi_b = rho0_i * V_b`, with the rest density of the fluid
particle i that sees the boundary, so it differs per material.

**Storage** (bd-r5-review-n28.77): the mass field holds
`m_b = RIGID_PSI_RHO0 * V_b` with `RIGID_PSI_RHO0 = 2500` (`common.cuh`,
`rigid_bodies.py`), and MAT_RIGID particles read density `RIGID_PSI_RHO0`. So:

- volume terms `m_b / rho_b` give `V_b` directly (DFSPH, Jacobi).
- the PBF rigid XSPH drag (22.5) uses `psi_b / rho_i = V_b * rho0_i / rho_i`.
- density sums and the Akinci coupling use
  `psi_b = m_b * (rho0_i / RIGID_PSI_RHO0) = rho0_i * V_b`.
- the colour-field normal uses `V_b = m_b / RIGID_PSI_RHO0`.

The scale `rho0_i / RIGID_PSI_RHO0` is computed with `__fdiv_rn`, so it is
exactly 1 for WATER (2500) under `--use_fast_math`. OIL gets 0.85, SAND 1.6,
MUD 1.8, LAVA 2.6. `add_rigid_body_with_particles` rejects any `rho0` other
than `RIGID_PSI_RHO0`.

**Material**: Boundary particles use `MAT_RIGID = 18` with `behavior_class = STATIC`.

**Source**: `rigid_bodies.py:_compute_psi()`, `common.cuh:MAT_RIGID`

### 22.4 Akinci Density Contribution

In Step1, boundary particles contribute to fluid density like regular particles:
```
rho_i += SUM_b psi_b * W(r_i - r_b, h),   psi_b = rho0_i * V_b = m_b * (rho0_i / RIGID_PSI_RHO0)
```
The neighbour loop multiplies a MAT_RIGID neighbour's mass by `rho0_i / RIGID_PSI_RHO0`
(one multiply per rigid pair). Every other neighbour uses `m_j` unchanged. The same
term is in K_Step1, K_Step1_BuildNL, K_DFSPH_ComputeDensityAlpha and
K_PBF_ComputeLambda. MAT_RIGID particles skip their own density computation (early return).

Measured with a kinematic 12 cm box submerged in a 0.6 m pool for 1.5 s
(before = every fluid sees `2500 * V_b`): particles inside the box drop from
156 to 0 (DFSPH LAVA), 160 to 0 (PBF LAVA), 64 to 0 (WCSPH LAVA), 58 to 0
(PBF SAND) and 6 to 0 (DFSPH SAND). WATER is unchanged. WCSPH OIL gets a
15% weaker wall and more particles inside (38 to 88), because the WCSPH rigid
wall is already too weak (WCSPH WATER also leaks 10 to 40 particles).

**Source**: `step1.cu:K_Step1`

### 22.5 Akinci Pressure Force (Two-Way Coupling)

For a fluid particle i near a MAT_RIGID boundary particle b:

```
F_pressure = -m_i * (p_i/rho_i² + p_i/rho_0²) * psi_b * grad_W(r_i - r_b)
```

The second term uses rest density rho_0 (not rho_b) because boundary particles have no meaningful density of their own. This mirrors the fluid's own pressure onto the boundary. Here `psi_b = rho0_i * V_b` as in 22.3, and the viscous coupling below uses the same `psi_b` (WCSPH `boundary_visc_factor`, DFSPH `K_DFSPH_NonPressureForces`).

**Viscous coupling**: Boundary velocity computed from rigid body state:
```
v_b = v_body + omega × (r_b - COM)
```
The viscous acceleration on i uses the boundary form of Section 3.3
(`boundary_visc_factor` in `step2.cu`): FLUID/GAS divide by rho_i, GRANULAR
divides by rho_i * rho0_i with mu0.

**Reaction forces** (Newton's 3rd law):

Step2 computes acceleration `a_on_fluid` (pressure + viscosity, scaled by force_scale).
To get proper force for rigid body integration, multiply by the fluid particle's mass:
```
F_on_body = -m_i * a_on_fluid     // convert acceleration → force, flip sign
tau_on_body = (r_b - COM) × F_on_body
```

This ensures `K_IntegrateRigidBodies` correctly computes `a_body = F_total / M_body`.

Forces accumulated via `warp_reduce_accumulate()` then `atomicAdd` to per-body accumulators.

**Source**: `step2.cu:K_Step2` (WCSPH), `dfsph_solver.cu:K_DFSPH_NonPressureForces` (DFSPH viscous part)

The pressure form above is WCSPH only. PBF and DFSPH do not push the fluid off a
rigid body with Tait pressure, so they do not give the body a Tait force either.

**PBF and DFSPH pressure reaction.** The body gets exactly minus what each
fluid-rigid pair gives the fluid, in the kernel that applies that push. Here
`gW = grad_i W(r_i - r_b)` (spiky, points from i toward b), `V_b = m_b / rho_b`
with `rho_b = RIGID_PSI_RHO0 = 2500`, and `m_b = RIGID_PSI_RHO0 * V_b` (22.3). The PBF
row uses `m_b` without the `rho0_i` scale.

| Solver | Kernel | Push on fluid i from b | Force on body |
|---|---|---|---|
| PBF | `K_PBF_ComputeDelta`, every iteration | `dx_i += (m_b/rho0_i) * (2*lambda_i + s_corr) * gW` | `F = -m_i * dx_ib / dt^2` |
| DFSPH divergence | `K_DFSPH_CorrectVelocityDiv`, every pass incl. warm start | `dv_i = -dt * V_b * kv_i * gW` | `F = +m_i * V_b * kv_i * gW` |
| DFSPH density | `K_DFSPH_ComputePressureAccel`, final call only | `a_i += -V_b * p_i * gW` (p = p_rho2) | `F = +m_i * V_b * p_i * gW` |

The PBF position correction over a substep is an impulse `m_i * dx / dt` and a
force `m_i * dx / dt^2`, summed over the iterations. The DFSPH density solver
runs ComputePressureAccel `dens_iters + 1` times but only the last `a_press` is
applied (ApplyPressureVelocity), so `simulation.py` passes the rigid buffers on
that call only. Torque is `(r_b - COM) x F` in every case. DFSPH Finalize computes no rigid
force. PBF Finalize adds the drag below.

**PBF fluid-rigid drag** (bd-r5-review-n28.97). K_PBF_Finalize runs XSPH
against MAT_RIGID neighbours with the body surface velocity, which
K_UpdateBoundaryParticles writes into the velocity array and the sort carries
into `original_vel`:

```
factor_ib = (psi_b / rho_i) * W_poly6(r_ib),   psi_b = m_b * rho0_i / RIGID_PSI_RHO0
dv_i     += c_scaled * factor_ib * (v_b - v_i)     // c_scaled as in 12.5
F_on_body = -m_i * c_scaled * factor_ib * (v_b - v_i) / dt
tau       = (r_b - COM) x F_on_body
```

`v_i` is the post-constraint velocity before XSPH, the same value the
fluid-fluid terms use, so the fluid momentum change and the body impulse cancel
exactly. Cost is one extra 16 B load per fluid-rigid pair. This is the only
shear coupling between PBF fluid and a body: without it a moving or spinning
body slid through PBF water with no drag. Measured (`test_rigid_coupling.py`,
dt = 1 ms): fluid within h of the side faces of a kinematic box with surface
velocity 0.5 m/s reaches 0.046 m/s after 100 substeps (0 before); a 6 cm WATER
layer on a plate moving at 0.5 m/s reaches 0.126 m/s mean after 60 substeps; a
free 6 cm sphere spinning at 5 rad/s drops to 0.64 rad/s after 200 substeps
(4.33 rad/s before, 4.60 from body damping alone); with zero gravity the fluid
momentum gain equals minus the body impulse. A kinematic body moves with its
lin_vel (22.7), so these tests reset its position after every step to get a
moving surface on a body that stays put, like a conveyor belt.

The reaction is taken before the per-particle clamps (PBF `|dx| <= 0.5h`, DFSPH
`|dv_div| <= h/dt`, `|a_press| <= h/dt^2` and the velocity limit). When a clamp
is hit, the body gets slightly more than the fluid lost. The clamps are meant
as safety nets; the momentum test below stays within 2%.

Measured (`test_rigid_coupling.py`): with zero gravity, the fluid momentum gained
equals minus the body impulse exactly in PBF and within 2% in DFSPH (DFSPH
fluid-fluid pairs use `V_j = m_j/rho_j`, which is only symmetric when
`rho_i = rho_j`). A kinematic 12 cm box 0.25 m deep in a settled WATER pool
(g = 4) gets F_y of about 29.9 N (PBF) and 27.3 N (DFSPH). The expected
`rho0 * V_eff * g` is 27.4 N, where `V_eff = (0.12 + 0.02)^3` because fluid
particle centres keep about one spacing from the boundary particles, so the
displaced volume reaches half a spacing past each face. So the 4.32 kg
"neutral" body in Archimedes Pool (sized for the bare 0.12 m cube, 17.3 N weight)
gets about 27 N of support and should float in PBF and DFSPH (inferred from the
force, not run as a preset).

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

Symplectic Euler. Rotation is integrated through the world-frame angular
momentum `L`, so Euler's equation `I dω/dt + ω × (I ω) = τ` keeps its
gyroscopic term (bd-r5-review-n28.114):

```
// Linear
v += (F/m + g) * dt
x += v * dt

// Angular (momentum form, world-frame inertia)
I_world     = R * diag(I_body)     * R^T      // I_body = 1/I_body_inv, 0 where I_body_inv = 0
I_world_inv = R * diag(I_body_inv) * R^T
L      = I_world(q) * omega
L      = (L + tau * dt) * ang_damp
omega  = I_world_inv(q) * L                   // drives the pose update

// Quaternion update
q' = normalize(q + 0.5 * dt * (0, omega) * q)

// Stored angular velocity, from the same L at the new rotation
omega' = I_world_inv(q') * L
```

With no torque `L` is conserved, and re-deriving `omega` from `L` at the new
rotation gives the `-I_world_inv * (omega × I_world omega)` term. A non-cubic
body therefore tumbles correctly. The Submarine box (half extents 0.15, 0.04,
0.06) has `Ixx < Izz < Iyy`, so z is its intermediate axis. A spin about z with a
small perturbation flips, and a spin about y stays stable. Over 1000 substeps
at dt = 1e-3, `|L|` matches `|L0| * ang_damp^1000` within 0.1 percent and the
direction of `L` does not change (`test_rigid_gyro.py`). For a cube `I_world`
is isotropic and the update reduces to `omega = (omega + I_inv * tau * dt) * ang_damp`.

Velocity damping: linear 1-0.01*dt, angular 1-0.05*dt (applied to L). Angular velocity clamped to 20 rad/s (L is scaled by the same factor). Force clamped to 1000*mass.

**World wall clamp (bd-r5-review-n28.98)**: per axis, the COM stays `s(axis) + BOUNDARY_MARGIN` (1e-4) inside `world_min/max`, where `s` is the support distance from 22.8 at the updated rotation. So the body surface, not the COM, stops at the wall. Velocity into the wall is reflected with factor -0.3.

**Kinematic bodies (bd-r5-review-n28.105)**: a body with `is_kinematic` (or
`inv_mass <= 0`) skips gravity, fluid force and torque, damping, the clamps and
the world-wall bounce, but its pose still advances every substep with the same
update: `x += v * dt` and `q += 0.5 * dt * (0, omega) * q` from the stored
`lin_vel` and `ang_vel`. The velocities stay as set. The boundary particles and
the fluid coupling use `lin_vel` as the surface velocity, so the body now moves
at the velocity the fluid sees. A kinematic body with `lin_vel = (1, 0, 0)`
moves 1 m in 1 s (`test_integrate.py::test_kinematic_body_moves_with_its_velocity`).
The force and torque accumulators are not cleared by the kernel for a kinematic
body, so the fluid reaction stays readable until the next substep prologue
clears them. Rigid-rigid collision still skips kinematic bodies as movers.

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

**Body vs SDF objects** (bd-r5-review-n28.98): contact depth = `s(n) - sdf(COM)`, where `n` is the SDF normal at the COM and `s(n) = sum_k he_k * |dot(n, R e_k)|` for a box (radius `he.x` for a sphere), from the device helper `rigid_support_distance`. Bodies with `sdf(COM) >= |half_extents|` are rejected early. The body is pushed out along `n`, then its velocity is reflected with restitution/friction. A box therefore rests on its face, not on its bounding sphere, and a sphere rests at its radius `he.x`.

**Body vs body**: Bounding sphere (`|half_extents|`) overlap test. Push apart weighted by inverse mass. Velocity reflection along contact normal.

**Source**: `integrate.cu:K_RigidBodyCollisions`

### 22.9 Boundary Particle State Sync

After rigid body integration, boundary particles must be updated to match the new body state:

```
r_world = COM + R * r_local          // position
v_world = v_body + omega × r_world   // velocity
```

Runs as `K_UpdateBoundaryParticles`, one thread per boundary particle.

**Slot layout**: all boundary particles of all bodies form one contiguous block
`[boundary_offset, boundary_offset + N_b)` in the unsorted arrays, inside
`world._high_water`. The kernel writes position, velocity and mass (psi_b) at
`rbm.boundary_offset + i` every substep.

- `inject_boundary_particles()` writes the block once, at the current
  `_high_water`, then raises `_high_water` by `N_b`. It also sets packed_info
  (MAT_RIGID, STATIC, body id in bits [20:13]), color, density, temperature,
  health, lifetime and sleep_counter. It syncs, so it runs only on scene events.
- `rbm.boundary_injected` is the one flag that says the block exists.
  `reset()` and `mark_for_reinjection()` clear it. Before writing, injection
  kills and compacts away any MAT_RIGID particles already in the world, so
  there is only ever one block.
- Brush spawns append after the block, so they never move it. Brush kill skips
  MAT_RIGID.
- `World.compact()` moves `boundary_offset` to the number of alive slots below
  it (the gather keeps order, so the block stays contiguous). If any block slot
  is dead, it calls `mark_for_reinjection()` instead.
- Scene load and undo call `mark_for_reinjection()`. `save_scene()`
  does not write MAT_RIGID particles, and `load_scene()` drops any it finds.
  The block is rebuilt from the body state on the next substep.
- Undo snapshots also hold the GPU rigid body array and the body count
  (`snapshot.py`). Restore copies the array back, reads it once into the CPU
  body list and re-uploads the count, so the block is rebuilt at the restored
  poses. If the body count changed since capture, the rigid state is not
  restored and a warning is printed.
- The undo ring (`SnapshotRing`) is sized to a 256 MB VRAM budget, 62 B per
  particle per slot, clamped to 2..12 slots (4 slots at 1M particles, 8 at
  500K). All slots are allocated at construction. Health, lifetime and mass
  are stored as float16 in a slot (relative error under 0.05% after
  restore); position, velocity, packed_info, temperature and dye stay
  float32, so temperature comes back exact.
- The R key, the solver switch and preset load call
  `Simulation.reset_scene_objects()`. It clears SDF objects, calls
  `rbm.reset()` and uploads a body count of 0. The default scene that R and
  the solver switch spawn has no rigid bodies.
- Boundary particles are drawn, because they sit inside `_high_water`.

**Capacity**: from `finalize_boundary_data()` until the block is injected,
`world.reserved_tail = N_b`. `mark_for_reinjection()` sets it again. Every
spawn path and the DEAD reserve top-up use
`world.free_capacity() = max_particles - reserved_tail - _high_water`, so fluid
spawned while the block waits cannot take its slots. A successful injection
sets `reserved_tail = 0`, because the block now counts inside `_high_water`.
`reset()` also sets it to 0.

If the block still does not fit (presets spawn their fluid before adding the
bodies, so a preset at the cap can hit this), injection prints one warning and
sets `rbm.boundary_dropped`. While the flag is set, the substep skips
`integrate_rigid_bodies` and `rigid_body_collisions`, so the bodies stay frozen
instead of falling through fluid that cannot push them. After a compaction
that leaves room for `N_b` slots, `_maybe_compact` calls
`mark_for_reinjection()`, and the next substep injects the block and clears
the flag.

**Source**: `integrate.cu:K_UpdateBoundaryParticles`, `rigid_bodies.py:inject_boundary_particles`,
`simulation.py:_update_rigid_boundary`, `simulation.py:_maybe_compact`,
`world.py:_shift_rigid_boundary_block`, `world.py:free_capacity`

### 22.10 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Body state storage | Global memory (RigidBody struct) | Max 8 bodies; constant memory reserved for SDF objects |
| psi_b storage | mass field holds RIGID_PSI_RHO0 * V_b | Avoids extra buffer; volume terms read V_b, density sums and Akinci coupling scale by rho0_i / RIGID_PSI_RHO0 |
| Force accumulation | Warp reduction + atomicAdd | 32x less contention than per-thread atomics |
| SDF vs boundary | Both coexist | SDF for collision, boundary particles for pressure/viscosity |
| Quaternion convention | (x,y,z,w) where w=scalar | Matches CUDA float4 layout |
| CUDA graph | Disabled when bodies present | Rigid body forces vary per substep (not capturable) |
