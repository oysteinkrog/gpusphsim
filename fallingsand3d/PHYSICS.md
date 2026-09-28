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
- Per-particle mass `m_j` (supports multi-material). When the FP16 velocity
  mirror is present, K_Step1 reads `m_j` from its `.w` lane: the counting sort
  stores `fp16(m / MASS_H_UNIT)` there, with `MASS_H_UNIT = 0.02` (= `PARTICLE_MASS`),
  and step1 multiplies back. One 8 B load then gives both neighbour velocity and
  mass (bd-r5-review-n28.161). A 0.02 kg particle stores exactly 1.0, so uniform
  water density is bit-identical to the float path. Other masses round with
  relative error up to 2^-11 (4.9e-4); on a mixed-mass Dam Break block density
  differs from the float path by at most 2.9e-4 (mean 1e-4). The self term uses
  the same mirror mass. Nothing changes `sorted_mass` between the sort and step1.
  Step2 still reads the float mass array, because reactions and spawn change
  `sorted_mass` between step1 and step2 without refreshing the mirror. Measured
  at 1M WCSPH (shared GPU): step1 2.69/2.64 ms to 2.57/2.58 ms avg over two runs
  (min 2.32 to 2.23 ms); step2 unchanged. Test: `test_step1_mass_mirror.py`.
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

### 2.4 FLUID EOS input: Shepard blend (WCSPH)

A particle at a free surface has only part of its kernel support filled, so
the density sum reads low there. K_Step1 also sums the Shepard weight and,
for FLUID particles only, feeds the EOS a blend of the raw and the
Shepard-normalised density (bd-r5-review-n28.205):
```
S_i       = C_poly6 * SUM_j [ (m_j / rho_j) * (h^2 - |r_ij|^2)^3 ]   (self term included)
rho_eos_i = max(0.5 * (rho_i + rho_i / max(S_i, 1e-6)), 1.0)
p_i       = EOS(rho_eos_i)                                          (section 3.1)
```
- `rho_j` is the neighbour's `position.w` (last substep's density, or this
  substep's if that neighbour already finished); the self term uses the
  particle's own `position.w`.
- S includes STATIC neighbours (`rho_j = rest_density`) and MAT_RIGID
  neighbours (`V_b = m_j / RIGID_PSI_RHO0`). Cross-phase GAS neighbours are
  left out, as they are for density.
- With full support S is about 1, so `rho_eos = rho`; at a free surface S is
  about 0.5-0.6 and `rho_eos` lies halfway to the corrected value.
- Only `pressure` and `p / rho^2` use `rho_eos`. `density_out` and
  `position.w` keep the raw `rho_i`, so step2 pair weights, heat, and all
  other readers see the raw density. GRANULAR, GAS and STATIC pressure use the
  raw density. FLUID never sleeps, so the inactive skip path (section 8.3)
  never recomputes a FLUID pressure without S.
- The surface tension (`nc < 25`) test and `surface_tension_gamma` use the
  neighbour count and normal, which this does not change, and were not retuned.

Measured on a WCSPH WATER column in a closed 0.6 m box (`test_step1_shepard.py`,
24k particles, 3 s settle at dt 0.5 ms). The reference is the weight of the
water above each particle, scaled by a slope fitted below 3h depth. The settled
water is compressed: top layer `rho/rho0` about 1.24-1.28, 2.16 in the
interior, so the top layer already has positive pressure. Top-2h error, RMS of
depth-bin means over p_ref at 2h: 0.0185 before, 0.0109 after (the raw EOS on
the same settled state gives 0.0367). Top 1 cm bin: -31% of p_ref before, -5% to -7%
after. Interior EOS density differs from the raw density by 0.4% on average
(single particles up to 2%). The free surface rose 0.3 mm (0.02 of a spacing)
and interior `rho/rho0` stayed within 0.5%.

**Source**: `step1.cu` (sum_shepard in the neighbour loop, PostCalc pressure)

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
For FLUID, `rho` here is the Shepard-blended `rho_eos` of section 2.4.

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
- `simulation._compute_c_sound` reads `sqrt(k * gamma / rho_0)` over FLUID and GRANULAR
  materials only (GAS is left out), so SAND (2.96 m/s) sets the CFL sound speed. The gas
  EOS sound speed is `sqrt(k)`, 1.6-2.0, which stays below it.

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
Both behaviours use the same dynamic form as their pair code, with the boundary density taken as
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
also takes `v_i` from `velocity_h`. So two particles with the same true velocity give exactly zero difference.
Step2 keeps the FP32 `v_i` for `veleval = v_i + eps * xsph` and for the STATIC and
MAT_RIGID boundary terms, whose other side (`v_b`) is FP32. When `velocity_h` is NULL
both sides are FP32.

With FP32 `v_i` against FP16 `v_j`, every pair gave `fp16(v) - v`, a half-ulp of up to
4.9e-4 m/s at |v| in [1, 2), summed into a body force toward the FP16 grid. A 10^3
lattice moving at (1.3, -0.7, 2.1) m/s got up to 3.3e-3 extra force per particle for
WATER and 0.18 for LAVA, and vorticity up to 0.012 1/s. It now gets exactly zero.
Dam Break KE (WCSPH, 0.25 to 1.5 s) is unchanged within 1%.

Tests: `test_step2.py::test_translating_block_no_fp16_viscous_bias`,
`test_step1.py::test_translating_block_zero_vorticity_strain`.

Tests: `test_visc_dynamic_form.py` (pair and free-surface block momentum at round-off,
pair value equals the formula above), `test_implicit_viscosity.py`.

**Source**: `step2.cu` K_Step2 FLUID/GAS pair branch, GRANULAR branch, `cross_pair_visc`,
`boundary_visc_factor`, implicit scale before PostCalc; `dfsph_solver.cu` K_DFSPH_NonPressureForces

### 3.4 XSPH Velocity Correction (FLUID only)

```
xsph_sum_i = SUM_j (m_j / rho_avg_ij) * (v_j - v_i) * W_poly6(|r_ij|)   [j FLUID only]

where rho_avg_ij = 0.5 * (rho_i + rho_j)

veleval_out_i = v_i + epsilon * xsph_sum_i     [awake FLUID only]
Other rows of veleval_out are not written.
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
Step2 writes `veleval_out` for awake FLUID only, and `integrate.cu` reads it only for
FLUID, so the plain velocity copy for other behaviours is gone (bd-r5-review-n28.170).

**FLUID neighbours only**: the sum runs over FLUID neighbours only (`behavior_j == FLUID`,
bd-r5-review-n28.104). STATIC and GAS neighbours are skipped earlier in the loop.
GRANULAR neighbours are skipped too: a settled sand bed is asleep at v = 0, so counting it
pulled water's advection velocity toward 0 while its stored velocity stayed, and water
crawled over sand. Water over sand or stone now advects at its own velocity plus the
average over nearby water. Test: `test_step2.py::test_xsph_skips_granular_neighbour`.

**Source**: `step2.cu` K_Step2 XSPH block in the neighbour loop (accumulation, FLUID guards),
`veleval_out` write after the loop (output)

### 3.5 Step2 Output

```
sph_force_out = total_force    [acceleration, m/s^2]
```

**NO mass multiplication at output.** This is acceleration directly.

STATIC and sleeping particles take an early return and write neither `sph_force_out`
nor `veleval_out` (bd-r5-review-n28.170). `integrate.cu` never reads those rows: STATIC
and still-sleeping particles return first, and a particle that wakes inside integrate
(reactions, the blast wave, IST or a geometry push) uses zero SPH force and no XSPH delta
for that substep, which are the values step2 used to store for it. No kernel between
step2 and integrate changes the sleeping bit, so the two kernels agree on which rows
were skipped. On a settled 1M SAND block this cut step2 from 0.045 to 0.008 ms.
Test: `test_integrate.py::test_unwritten_force_and_veleval_rows_are_not_read`.

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

The PBF and DFSPH rows are for one substep per fixed dt; the substep split below raises
the limit n times.

**Lagged max velocity** (`Simulation._latch_v_max`, no GPU sync). The host already
latches `max_displacement` (the largest drift since the last full sort) from the async
readback each frame. It also records the sim time since that sort, so

```
v_max = sqrt(max_displacement) / sim_time_since_sort
```

While the scene moves every substep sorts, so this is the largest one-substep move
divided by dt. While sorts are skipped it is the fastest average speed over the skip
window. The value is 1-2 frames old. The +inf sentinel (K_SpawnGas teleport, or no copy
since a scene change) keeps the previous value; a drift of exactly 0 gives 0.

**WCSPH timestep** (`Simulation._compute_adaptive_dt`, adaptive mode only):

```
dt_accuracy = dt_max * accuracy / 0.4
dt_cfl      = 0.25 * h / (c_sound + v_max)
dt_target   = clamp(min(dt_accuracy, dt_cfl), DT_MIN = 1e-5, dt_max)
```

`dt_max` is the solver profile's `dt_max` when it has one, else `DT_MAX` = 1 ms (no
shipped profile sets it yet). At v_max = 0, dt_cfl is the acoustic bound, about 3.4 ms
(c_sound = 2.96 m/s), so dt_accuracy acts in a calm scene: the Accuracy slider and the
`[` `]` keys give 0.25 ms at 0.1, 0.5 ms at 0.2 and 1 ms at 0.4. With a 1 ms cap the
velocity CFL binds above about 7 m/s (10 m/s gives 0.77 ms); with the 0.5 ms cap of the
WCSPH profile, above about 17 m/s. There is no viscous CFL: step2 treats the viscous
self-term implicitly (see 3.3 and 10.4).

`step_frame` lowers dt to dt_target at once and raises it by at most 10% per frame. It
uploads c_sim only when dt changes by more than 5% or reaches dt_accuracy. A smaller dt
means more substeps per frame, up to `max_substeps`; past that the sim runs slower than
real time instead of taking larger steps. Grid-reuse limits and the 30-substep spawn
damping ramp count substeps, so they cover less sim time when dt is small.

Measured (WCSPH (Fast), Water Drop, 75K, `dt_max` = 2 ms set by hand): dt holds 2 ms in
free fall, falls to 0.81 ms at impact (v_max 13 m/s) and climbs back to 1.3-1.5 ms as
the splash calms. Sand Castle (364K, frames 200-300, `dt_max` = 2 ms against 1 ms): 594
against 1000 substeps per sim second, 3.3 against 5.5 wall seconds per sim second on a
shared GPU. The scene keeps v_max near 3 m/s, so dt sits at 1.6-1.7 ms. WCSPH B.2 eval
scenes at fixed dt 1.0, 1.5, 2.0 and 2.2 ms: Dam Break COM_y -0.601 to -0.602, granular
spread 1.88x to 1.91x, settled top 5% y -0.707 to -0.726, no NaN.

**PBF/DFSPH substep split** (`Simulation._next_fixed_subdiv`, fixed dt mode). With
`v_cap = factor * h / fixed_dt` (4.32 m/s for PBF at 1/120 s):

```
n  = clamp(ceil(v_max / (0.5 * v_cap)), 1, 8)
dt = fixed_dt / n        -> velocity_limit = n * v_cap
```

n rises at once and falls by one per frame; c_sim is uploaded only when n changes. A
PBF Water Drop (g = 9.81) reaches 8.6 m/s at impact with n = 2 to 5, where one 1/120 s
substep held it at 4.32 m/s. With PBF `max_substeps` = 4, frames with n > 2 run slower
than real time. WCSPH in fixed mode is not split.

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

A FLUID particle that woke in integrate this substep gets no XSPH delta: step2 skipped
it while it was asleep and did not write its `veleval` row.

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
DFSPH still uses the global `mu_s`.

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

`eta` depends only on the particle, so K_Step1 PostCalc computes it once per GRANULAR
particle from its own `gamma_dot`, pressure, density and material (5.2 to 5.4), and
writes it to `eta_out` (`world.sorted_eta`, 0 for non-GRANULAR). K_Step2 reads
`eta_i` and each `eta_j` from `eta_in` (4 B per GRANULAR pair) and does only the
harmonic mean per pair. The inputs are the ones Step2 used, so the forces match the
per-pair path exactly. The WCSPH substep in `simulation.py` passes `world.sorted_eta`
as both. When `eta_in` is NULL, K_Step2 recomputes eta per pair from
`shear_rate_in`. `step2.upload_granular_params` also uploads `c_granular` to the step1
module for this (bd-r5-review-n28.169). A particle whose material turns GRANULAR in
the reactions pass between Step1 and Step2 (LAVA to GRAVEL, MUD to WET_SAND) carries
eta 0, and K_Step2 recomputes eta from `shear_rate_in` for any GRANULAR particle whose
carried eta is not above 0, so it gets the same value as the per-pair path.
A GRANULAR to GRANULAR change in that pass (SAND to WET_SAND) keeps the eta of the old
material's `mu_s` for that one substep.
Measured on a resting sand block, WCSPH, K_Step2 per-pair eta to carried eta:
0.88 to 0.81 ms at 340K particles and 2.11 to 1.97 ms at 830K (shared GPU);
K_Step1 time unchanged.
DFSPH NonPressureForces still computes eta per pair.

**Source**: `step1.cu` (PostCalc eta), `step2.cu` (mu(I) eta_i, GRANULAR pair force)

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
| dt | adaptive | 0.001 (fixed) | min(dt_max * accuracy / 0.4, 0.25 h / (c_sound + v_max)), in [1e-5, dt_max], dt_max = 1 ms (see 4.2) |
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
if |v| > 0.02 or sdf_k(x) < 0 for any SDF object k or x outside [world_min, world_max]:
    clear SLEEPING, set JUST_WOKE
```

Hysteresis: 4x ratio between sleep (0.005) and wake (0.02) velocity thresholds.

Geometry wake (WCSPH, `K_Integrate` sleeping branch): a sleeping particle
inside an SDF object or outside the world box wakes and takes the normal path
in the same substep, so the SDF push-out, the object velocity impulse and the
world box clamp apply to it. This covers a kinematic SDF box or cylinder
(`SDFManager.update_kinematics`) moving into a settled pile, as in the Water
Mill and Wrecking Ball presets, and a world resize onto a sleeping pile. The
test is strict penetration (dist < 0, the unshrunk box), not the
BOUNDARY_MARGIN band: a particle resting on a surface sits at
dist = BOUNDARY_MARGIN after its last push-out, so a band test would wake it
on rounding noise and sand on static objects could not sleep. The SDF loop
costs nothing when there are no SDF objects. PBF and DFSPH finalize kernels do
not have this test yet.

Wake propagation runs at the end of every substep for all three solvers, on
the unsorted arrays. A marker is a particle that has JUST_WOKE set, or an
awake particle faster than 0.02 m/s that is not STATIC or GAS; it writes a
stamp into its own grid cell (one plain store). The solver's last kernel
writes the mark from the final position, velocity and packed_info it already
holds, through `wake_mark_own_cell` in `sph_shared.cuh`: `K_Integrate` for
WCSPH (bd-r5-review-n28.166), `K_PBF_Finalize` and `K_DFSPH_Finalize` for PBF
and DFSPH (bd-r5-review-n28.167). `run_wake_propagation` then runs with
`mark=False`, so the only wake launches are the rigid body marks (when the
scene has bodies) and the sleeper probe; `K_MarkWakeCells` is not launched and
its 36 B per particle reread is gone. `K_MarkWakeCells` stays as the reference
path (`mark=True`) for direct callers and the unit tests, and uses the same
test, so the marks are the same: `test_integrate_wake_mark.py` checks the
`K_Integrate` table equals the `K_MarkWakeCells` table for the same state, and
on Sand Castle with all sand starting asleep DFSPH sleeping counts matched the
`K_MarkWakeCells` path exactly at every checked substep. The integrate module
gets `c_grid` from `Simulation._upload_constants` for this; before, it had
none and every particle mapped to hash 0. At 1M particles (WCSPH Dam Break,
`bench_substep.py`, shared GPU, 2 interleaved runs) the wake stage went from
0.045 ms to 0.008 ms and integrate from 0.483 ms to 0.488 ms per substep.
`K_WakeSleepersAndClearJustWoke` wakes a sleeping
particle when any of its 27 cells (own cell and 26 neighbours) holds the
stamp. This is the same rule as marking the 3x3x3 block around each marker
and probing one cell, because the neighbourhood is symmetric; it replaces 27
scattered atomics per marker with one store (bd-r5-review-n28.141). Hash
collisions can only add wakes, never drop one.

The table holds a generation stamp, not a 0/1 flag:
```
stamp = frame_counter_d[0] + 1     (never 0, the value of a fresh table)
mark:  cell_wake_flags[hash(own cell)] = stamp
wake:  any(cell_wake_flags[hash(c)] == stamp for c in 27 cells)
```
`frame_counter_d` is advanced by one on the device by `K_IncrementCounter`, the
first node of every substep body (graph replay or direct launch). It never resets, so a mark from an earlier substep no longer matches and the
table is never cleared per substep (the 1 MB memset is gone). The table is
zeroed only when it is allocated. The uint32 counter would wrap after about
4e9 substeps. A caller that passes no frame counter gets stamp 1 and
`run_wake_propagation` clears the table first. At 1M particles (WCSPH Dam
Break, `bench_substep.py`) the wake stage went from 0.58 ms to 0.06 ms per
substep.

The velocity trigger skips STATIC and GAS particles. GAS is skipped because step1
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
    stamp every cell of the cube [c - r, c + r]
```
The sleeper's 27-cell probe adds one cell (about h) of reach, so sand within
one smoothing length of the cube wakes. A body at rest marks nothing, so a box resting on a pile lets
the pile sleep; if the box then starts to sink, it wakes the pile again within
a few substeps. For WCSPH, `K_Integrate` keeps the MAT_RIGID boundary velocity
(written by `K_UpdateBoundaryParticles`) in the unsorted velocity array, where
other STATIC particles get zero.

**Source**: `integrate.cu:54-59, 115-127, 280-302, 407-418`, `wake.cu`
(`K_MarkWakeCells`, `K_MarkWakeCellsRigid`), `sph_shared.cuh`
(`wake_stamp`, `wake_mark_own_cell`), `pbf_solver.cu` and `dfsph_solver.cu`
(finalize kernels), `wake.py`, `simulation.py`
(`_run_wcsph_body`, `_run_pbf_body`, `_run_dfsph_body`, `_wake_rigid_kwargs`)

### 8.3 Step1 skip for inactive particles (WCSPH)

`K_Step1` treats a particle as inactive when it is SLEEPING or non-rigid
STATIC. Step2 skips both as particle i and `K_Integrate` reads only their
`dTdt`, and their density cannot change while they stay inactive
(bd-r5-review-n28.159). An inactive particle runs the neighbour loop only on
its refresh substep:
```
refresh = force_refresh or (substep + index_i / 32) % 8 == 0
substep = frame_counter_d[0]       (the wake-stamp counter, graph-safe)
force_refresh = 1 on every full-sort substep
```
On other substeps it writes `density = position.w` (STATIC: `rest_density`;
a sleeper with `.w <= 0`: `rest_density`), recomputes `pressure` and
`p / rho^2` from it, and leaves shear rate, eta, dTdt, both exposures,
vorticity and normal as the last full compute wrote them. Under grid reuse
the sorted order does not change, so those values belong to the same
particle; a full sort reorders the buffers, so it forces a refresh. The
phase is per warp of 32 sorted indices: a per-index phase put 4 refreshing
lanes in every warp, and the warp then ran the loop every substep.

On a refresh the inactive particle's heat rate is backward Euler over the
8-substep hold, with the neighbour temperatures fixed:
```
S = C * SUM_j a_ij (T_j - T_i),  A = C * SUM_j a_ij,  C = viscosity_lap_coeff / (rho_i cp_i)
dTdt_i = S / (1 + A * 8 * dt)
```
`K_Integrate` applies this rate on every substep until the next refresh, so
the held rate cannot carry T_i past its neighbours' weighted mean. Active
particles keep the explicit `dTdt = S`. Staleness bounds: exposure and heat
lag by at most 8 substeps; a particle that just fell asleep holds its last
explicit rate until its first refresh. In a 181-particle test (hot lava
under a sleeping sand block, dt = 1/240), the sand's heat gain over 1 s was
within 3% of the always-compute path.

MAT_DEAD slots skip the loop every substep and write fixed outputs: density
1 (the STATIC floor they had before, not 0, because K_Step2 divides by a
neighbour's `position.w`), pressure from it, and zero for everything else.

The saving is per warp. The spatial-hash sort order puts about 4 cells in a
warp, so a warp skips only when all its particles are inactive. Measured on
884K SAND in a lattice (`K_Step1` alone, RTX 5070 Ti, GPU shared with other
jobs): 25/50/75/100% asleep gave 0.97/0.92/0.81/0.29 of the always-compute
time. Fluid-only scenes are unchanged (1.00). Callers that pass no frame
counter (tests, `benchmark.py`) compute every particle every call.

**Source**: `step1.cu` (K_Step1 inactive gate and PostCalc dTdt),
`step1.py` (`compute_step1` `frame_counter`, `force_refresh`),
`simulation.py` (`_run_wcsph_body`)

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
the `rho_j` its neighbours read (`position.w` in WCSPH, `density_in` in PBF and DFSPH). Same-material pairs get `k_ij = kappa_i`
(unchanged). Mixed pairs are limited by the poorer conductor: LAVA (500) and WATER (50)
give `k_ij = 91`, LAVA and METAL (2000) give 800. All three loop copies use this form:
`K_Step1`, `K_PBF_ComputeLambda` and `K_DFSPH_ComputeDensityAlpha`.

`test_step1.py::test_heat_energy_conserved_lava_water` checks the step-0 rate balance
(`|SUM m cp dTdt| / SUM |m cp dTdt| < 1e-4`, measured 3e-8) and the drift of
`SUM m cp T` over 100 fixed-geometry Euler steps (`< 1e-5`, measured 2e-8). With
`kappa_i` applied per particle the imbalance was 0.82.

**Temperature mirror precision (bd-r5-review-n28.132).** WCSPH step1 reads
neighbour temperatures from `sorted_temperature_h`, a 2-byte mirror the counting
sort writes every substep. It holds `uint16 = round(T * 8)`, so it has uniform
0.125 K steps, covers 0 to 8191.875 K (T_MAX = 5000) and is exact for every
multiple of 0.125 K, T_AMBIENT = 293 included. `store_t16` / `load_t16` in
`common.cuh` encode and decode it. The particle's own `T_i` in every pair
difference (`T_j - T_i` in `sum_dTdt` and `max(T_j - T_i, 0)` in
`exposure_heat`) is read from the same mirror, as for velocity (3.3).
The pair flux is therefore exactly antisymmetric and a uniform field gives
exactly zero. The cost is a 0.125 K resolution in the conduction gradient:
pairs whose temperatures round to the same step exchange no heat. With FP16
(0.25 K steps near ambient, 1 K from 1024 K, 2 K from 2048 K) against an FP32
`T_i`, each pair gave `fp16(T_j) - T_i`, which pulled particles toward the FP16
grid and did not conserve `SUM m cp T`. The integrator still updates the FP32
`temperature`; only the step1 neighbour reads see the mirror. With a NULL mirror
step1 reads FP32 for both. The PBF and DFSPH heat loops read FP32 only. Tests:
`test_step1.py::test_uniform_off_grid_temperature_zero_dTdt` (WATER at 293.2 K
and LAVA at 1499.7 K in STATIC STONE walls: every `dTdt` and `exposure_heat`
is 0, both step1 kernels), `test_temp_mirror_is_fixed_point_eighth_kelvin` and
`test_heat_energy_conserved_with_temp_mirror` (LAVA/WATER slab, 1000 substeps
through the mirror, drift `< 1e-6`, measured 4.7e-7).
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

This is in `K_Step1`, `K_PBF_ComputeLambda` (do_heat call) and
`K_DFSPH_ComputeDensityAlpha`. Tests: `test_step1.py::test_fire_heats_water_across_phase`,
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

K_Step1 uses this definition. The PBF and DFSPH density kernels still
output the older unnormalised `poly6_coeff * SUM_j rate_ij * W_ij`.

For sleeping and non-rigid STATIC particles, WCSPH recomputes both
exposures only on their refresh substep and holds them in between, so a
threshold or `exposure_corrode * dt` sees a value up to 8 substeps old
(§8.3).

**Source**: `step1.cu` (K_Step1 PostCalc), `reactions.cu` thresholds

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
| velocity | spawn velocity, w = 0 |
| kappa, kappa_v (DFSPH warm start) | 0 |
| mass | `spawn_mass(material)` (§7.4) |
| temperature | §9.3 default |
| health | 1 |
| lifetime | GAS default below, else 0 |

World holds no unsorted veleval, sph_force, density, shear_rate or exposure
arrays (bd-r5-review-n28.152). Kernels only read the `sorted_*` versions,
which step1 and step2 write for every particle each substep, so a spawn has
nothing to reset there. There is no color field to set: `K_ComputeColor` computes the render color
from the state every frame (see 10.11).

`presets._clear_world` also zeroes `kappa`, `kappa_v` and `sorted_lambda_pbf`
for the whole pool, so no DFSPH or PBF warm-start pressure from the previous
scene reaches the first substep of the next one.

Two other paths reuse slots, and both clear the DFSPH warm start too
(bd-r5-review-n28.121):

- `World.compact()` zeroes `kappa` and `kappa_v` in the freed tail
  `[num_alive, old _high_water)`, along with `packed_info`. It also zeroes
  all of `sorted_density`: compaction moves every slot, so the PBF and DFSPH
  previous-density read (see 10.17b, density buffer lifecycle) would otherwise
  get other particles' values. A zero falls back to rest density, as on a
  fresh scene.
- `RigidBodyManager.inject_boundary_particles` zeroes `kappa` and `kappa_v`
  in the boundary block.

The DFSPH neighbour loops also ignore whatever a STATIC neighbour holds: in
K_DFSPH_CorrectVelocityDiv `kv_j = 0` and in K_DFSPH_ComputePressureAccel
`p_j = 0` when j is STATIC. So a wall particle adds only its `kv_i` or `p_i`
mirror term (§13), even on the first substep after a slot was reused. PBF has
no unsorted lambda array (§12.4), so there is no PBF warm start to clear.

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

**Source**: `world.py` (`_init_slot_state`, `_DEFAULT_LIFETIMES`, `compact`),
`rigid_bodies.py` (`inject_boundary_particles`),
`dfsph_solver.cu` (K_DFSPH_CorrectVelocityDiv, K_DFSPH_ComputePressureAccel),
`presets.py` (`_clear_world`, `run_spawner`), `reactions.cu` (GAS lifetime decay)

---

### 9.7 Thermal gate for cold scenes (WCSPH)

`c_sim.thermal_enabled` (int, last field of SimParams, 76 bytes) lets
K_Step1 skip heat diffusion (§9.1) and exposure (§9.4). With the flag 0 the
kernel does not load T_j and writes `dTdt = exposure_heat = exposure_corrode
= 0`. Density, pressure, vorticity, normal and strain rate do not change.
K_Step1 is one kernel whose body is a template on the gate, so the thermal
path carries no extra branches (bd-r5-review-n28.172).

Simulation sets the flag on the host from `World._spawned_material_ids`, with
no GPU sync. The flag is 0 only when all of these hold:

- at least one material was spawned;
- every spawned material starts at T_AMBIENT (none of LAVA, FIRE, STEAM,
  SMOKE, ICE, see §9.3);
- no pair of spawned materials, a material with itself included, has a
  non-zero `reaction_rate` or `heat_exchange`.

In such a scene every temperature stays at T_AMBIENT and every heat and
exposure term is exactly 0, so the gate changes no result. Water alone and
water with stone qualify; sand with water does not (wetting pair). The rule
runs once per substep and re-uploads c_sim only when the flag changes. A
load or undo replaces the material set with one rebuilt from the restored
particles, whose temperatures are unknown, so the flag then stays 1. The
set only grows while a World lives, so after a hot preset the flag stays 1
for later presets. PBF and DFSPH ignore the flag.

Measured with `benchmark.py` on the water-only dam break at 992K particles
(RTX 5070 Ti, shared GPU): step1 2.34 to 1.95 ms, substep 8.35 to 7.85 ms.
The Volcano preset (flag 1) gives the same material counts, temperatures
and centre of mass after 600 substeps, and its step1 time did not go up.

**Source**: `step1.cu` (step1_body, K_Step1), `simulation.py`
(`_thermal_rule`, `_update_thermal_gate`), `step1.py` (`SIM_PARAMS_DTYPE`)

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
5. **GAS phase separation**: GAS particles skip regular STATIC neighbours in
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
   spacing from the wall is not touched. Called from PBF Finalize and DFSPH
   Finalize, and from WCSPH `K_Integrate` when it is given the cell arrays
   (see the known gap below). (bd-r5-review-n28.43)
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

   Single neighbour scan in PBF and DFSPH Finalize (bd-r5-review-n28.164).
   The push shares the XSPH loop's 27-cell scan through the `StaticPush`
   helpers (`sph_shared.cuh`) instead of running a second one. The result
   equals the separate scan to float rounding in every case:
   - PBF FLUID and GRANULAR: the push is summed in the XSPH loop.
     `final_pos == pred`, so the distances are the ones the scan would use.
   - DFSPH: the push is measured at `pos_new = pos + dt * v_advect`, which
     needs the XSPH sum. The loop tests walls against
     `pos_adv = pos + dt * v` instead. GRANULAR has no XSPH, so
     `pos_adv == pos_new` and the loop sums the push. FLUID only flags a
     wall particle within `0.5 h + 0.1 h` of `pos_adv`. With no flag and
     `|pos_new - pos_adv| < 0.1 h` there is no push; otherwise the particle
     runs the separate scan at `pos_new`. The loop path needs
     `dt * |v| < 0.4 h` so the 27 cells around `pos` (cell size h) hold
     every wall particle in reach; faster particles run the separate scan.
   - GAS skips the XSPH loop and always runs the separate scan.

   Measured with `bench_substep.py` at 500K (dam break, no STONE, shared
   GPU, 2 interleaved runs each): Finalize median PBF 2.04 to 1.74 ms (-15%),
   DFSPH 1.26 to 0.96 ms (-24%). `test_static_push_fold.py` replays every
   Finalize of a water, sand and smoke drop on STONE with the fold switched
   off: max difference 6e-8 m, 8e-7 m/s.

   Known gap: the push does not run in WCSPH. The integrate module has
   `c_grid` (added for the wake mark, bd-r5-review-n28.166), but the WCSPH
   body passes no `cell_start`/`cell_end` to `K_Integrate`, which skips the
   scan. Before the `c_grid` upload the scan hit hash bucket 0 only, so this
   keeps WCSPH physics as it was. Passing the cell arrays turns the push on
   and costs about 0.16 ms per substep at 1M (Dam Break, integrate 0.48 to
   0.65 ms).

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

### 10.8 OPTIMIZATION: Counting sort pipeline
The per-substep grid build is `counting_sort.counting_sort_full` (no physics changes):
1. **One sort path, five launches** (bd-r5-review-n28.189): K_HashHistogram
   (hash and count per cell), K_ScanReduce and K_ScanDownsweep (our own
   inclusive prefix sum into `cell_end`), K_ScatterPerm (slot claim, writes
   `sort_perm` only) and K_GatherReorder (gathers every particle array).
   `sort_perm` maps a sorted slot to its original index. The argsort pipeline (hash_sort,
   build_grid, fused_reorder, fused_sort_reorder_build) was removed
   (bd-r5-review-n28.153).
2. **Reorder only what is read**: veleval, shear_rate and similar per-substep
   outputs are not reordered, since K_Step1, K_Step2 and K_Integrate overwrite them.
3. **Perm-only scatter, then gather**: K_ScatterPerm claims each particle's
   sorted slot with `k = atomicSub(&histogram[hash], 1)`, slot
   `cell_end[hash] - k`, and stores one uint32 per particle. The atomicSub
   drains the histogram back to zero, so the next full sort needs no memset
   and there is no write_offset buffer. Within-cell order is not
   deterministic. The particle data then moves in K_GatherReorder, which has
   coalesced stores (the grid-reuse path uses the same kernel). The scan
   takes nothing from the CuPy memory pool: `cupy.cumsum` took a CUB
   workspace on every call, and a captured graph kept a raw pointer to it
   after the pool had taken the block back. Measured at 1M particles, 2^18
   cells, random positions (shared GPU, 3 runs): full sort as a graph 0.74 to
   0.78 ms, was 1.92 to 2.04 ms; host time per direct call 80 to 320 us, was
   700 to 980 us.
4. **Reactions/spawn always run**: Previously conditionally skipped for non-reactive
   materials; now always included for CUDA graph fixed topology (§10.9).
   Non-reactive threads early-exit with negligible GPU cost.
5. **Fused wake kernels**: K_WakeSleepers + K_ClearJustWoke merged into
   K_WakeSleepersAndClearJustWoke (3→2 kernel launches).
6. **Block size 128→256** for Step1/Step2 kernels.
7. **One cell offsets table** (bd-r5-review-n28.160): the grid is one uint32
   array of `table_size + 1` entries with `offsets[0] = 0`; the prefix sum
   writes the inclusive sum into `offsets[1:]`. Kernels get
   `cell_start = offsets[:-1]` and `cell_end = offsets[1:]` (views of the same
   memory), so cell `h` holds sorted slots `cell_start[h] .. cell_end[h] - 1`
   and an empty cell has `cell_start[h] == cell_end[h]`. There is no empty-cell
   sentinel and no neighbour loop tests for one, and a full sort needs no
   separate cell_end pass or memset. Measured against the two-array layout at
   1M WCSPH particles (shared GPU, 3 A/B runs): step1 2.64 vs 2.72 ms, step2
   4.03 vs 4.38 ms per substep.

**Source**: simulation.py, counting_sort.py/.cu, wake.py/.cu

### 10.9 FIXED: GPU→CPU sync every frame in adaptive timestep
`_compute_adaptive_dt()` used `float(cp.max(...))` on GPU velocity arrays to
compute advection CFL, forcing a GPU→CPU sync (device→host transfer) every frame.
Adaptive dt now takes v_max from the drift value the per-frame async readback already
copies (one frame late), and computes the accuracy bound and the velocity CFL on the
CPU (see 4.2). There is no GPU sync in the hot path.
**Source**: `simulation.py:151-175`

### 10.10 OPTIMIZATION: CUDA graph capture for SPH pipeline
The per-substep pipeline launches ~14 GPU operations (memsets + kernel calls) with
Python→CUDA round-trip overhead for each. CUDA graph capture records ops 3-14 into
a single replayable graph, reducing CPU→GPU dispatches from 14 to 3.

**No split point**: the counting sort uses only its own kernels and makes no
memory-pool allocation (10.8), so the whole substep, sort included, is
captured. There
are two graphs: one with the full counting sort and one with the gather-only
reorder (grid reuse, see `_capture_graph`).

**Design details**:
- Pre-allocated `_sort_perm` buffer: graph bakes pointer addresses, so the
  sort writes the permutation into one stable buffer.
- Device-side frame counter: `K_Reactions` reads `frame` from a `const uint*`
  device buffer (pointer stable, value updated via `fill()` before each launch).
- Reactions/spawn always included: graph requires fixed topology. For non-reactive
  scenes, every thread early-exits in the kernel — negligible GPU time.
- Graph invalidation: re-captured when `n` (particle count) changes (compaction,
  preset switch). Grid dimensions and `np.uint32(n)` args are baked.
  `World.compact()` runs one `K_Compact` pass (`compact.cu`) into the
  `sorted_*` scratch arrays and then swaps them with the primary arrays, so
  the device pointers change. It bumps `World.buffer_generation`, and
  `_sim_step` drops both graphs when that value changed, even if `n` came
  back to the same count (bd-r5-review-n28.149).
- Capture uses a non-default stream (`cupy.cuda.Stream(non_blocking=True)`);
  replay launches on the default stream.
- Constant memory is not part of a graph: kernel nodes read `__constant__`
  data when they run. Per-substep and per-frame constant writes (kinematic
  SDF poses, the spawn damping ramp, dt and gravity changes) go through
  `rigid_bodies.PinnedUploadRing`: the bytes are copied into a pinned slot
  and pushed with `cudaMemcpyAsync` on the current stream, so the copy is
  ordered before the next graph launch and the host never waits for the GPU
  (a pageable `cudaMemcpy` drains the queue first). These writes do not
  invalidate the graphs (bd-r5-review-n28.168).

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

**Where the color is computed.** No solver stores a color. `K_ComputeColor`
(`integrate.cu`) runs from `Simulation.copy_to_vbos`, at most once per frame, one
thread per unsorted slot in `[0, _high_water)`, and writes straight into
the mapped color VBO. It reads `packed_info`, `temperature`, `health`,
`velocity` and `position`, with this substep's density in `position.w`.
K_Integrate, K_PBF_Finalize and K_DFSPH_Finalize all write the density
there. FLUID slots get `compute_fluid_color(mat, T, health, pos.y, |v|^2,
rho)`, every other behavior gets `compute_color(mat, T, health, behavior)`.
The formulas are the ones the solvers used to apply every substep, and the
last substep's inputs are what the unsorted arrays hold at frame end, so
the rendered color is the same. At 1M particles the kernel takes about
0.1 ms per frame and replaces a 16 B per particle scatter on every
substep (WCSPH integrate 0.96 to 0.72 ms per substep) and the 32 MB of
`color` and `sorted_color` buffers. (bd-r5-review-n28.145)

main.py skips the whole VBO fill (position, color, velocity and foam) on
frames where no substep ran, unless `Simulation.scene_dirty` is set.
`notify_scene_changed()` and `set_world_size()` set it and `copy_to_vbos`
clears it, so a paused scene costs no device copies and no interop map
calls, and a brush spawn while paused still shows on that frame. Position,
color and velocity are mapped with one `cudaGraphicsMapResources` call
(`gl_cuda_interop.map_all`). (bd-r5-review-n28.150)

**Source**: `sph_shared.cuh:compute_fluid_color`, `integrate.cu:K_ComputeColor`

### 10.12 OPTIMIZATION: Spatial hash grid (replaces dense linear grid)

Replaced the dense cell_start/cell_end arrays (sized `grid_res^3`) with a
spatial hash table whose size is a power of 2 from 2^18 to 2^20. This enables
arbitrarily large world sizes without memory blowup.

**Old approach** (dense linear hash):
- `hash = z * ry * rx + y * rx + x` → array size = `grid_res^3`
- For world_half_size=1.0, h=0.04: grid_res=50, num_cells=125K (500KB)
- For world_half_size=10.0: grid_res=500, num_cells=125M (500MB!) → OOM

**New approach** (Morton order inside 8x8x8-cell blocks):
```
lo   = morton3(cx & 7, cy & 7, cz & 7)                       // 9 bits
hi   = (cx>>3)*73856093u + (cy>>3)*19349669u + (cz>>3)*83492791u
hash = (lo + (hi << 9)) & TABLE_MASK
```
- The counting sort orders particles by hash, so inside a block the
  particles lie in Z-order: the cells a warp reads are spatial neighbours
  and their 27-cell stencils share cache lines. A/B on `bench_substep.py`
  (WCSPH, full sort every substep, RTX 5070 Ti): step1+step2 at 1M
  3.98 -> 2.73 ms, at 524K 1.86 -> 1.23 ms, at 262K 0.95 -> 0.75 ms,
  against the plain prime hash of the whole cell. PBF and DFSPH totals at
  1M drop about 35%. (bd-r5-review-n28.201)
- The block is 8 cells wide, not 64. Part-filled blocks waste table
  slots, since every cell of a block with the same `hi` bits aliases: a
  64-cell block made a wide 2-layer water slab (480K particles, world
  half size 10) 2.7x slower in step1+step2, where the 8-cell block is
  4% faster than the plain hash.
- `hi` stays a linear prime sum. A nonlinear mixer spread lattice-shaped
  scenes worse in a host model of bucket loads.
- `grid_params.compute_table_size(N)` rounds `max(N/2, 2^18)` up to a power
  of 2 and caps it at 2^20: 2^18 up to about 524K particles, 2^19 up to about 1M,
  2^20 above. `TABLE_SIZE = 2^18` is the default for tests.
- The cell offsets table (§10.8) has table_size + 1 entries (~1MB at 2^18)
- The block products and sum wrap in 32-bit unsigned arithmetic and `>> 3`
  is an arithmetic shift, so negative cell coordinates are well defined
- **Stencil cells never share a bucket.** Along each axis the 3 cells of a
  stencil have 3 distinct values mod 8, so the 27 cells have 27 distinct
  in-block triples and 27 distinct `lo` codes, and `hi << 9` never touches
  the low 9 bits. This holds for any table of 2^9 or more, so no particle
  is counted twice. `test_hash_sort.py` checks this on K_HashHistogram for
  world edges, faces, block corners, interior and negative cells.
- Why the block hash is a sum and not XOR: `P ^ (-P)` has the same low 18
  bits (0x3FFFE) for all three primes, so an XOR combiner of whole-cell
  products put two stencil cells of a world-edge cell in one bucket, and
  particles along those edges were counted twice (+3 to +12% density).
- Far-apart cells can still share a bucket: two cells collide when they
  have the same place in their block and their blocks agree in the hash
  bits above bit 9. That costs extra distance checks but never misses or
  double-counts a neighbour.
- `grid_params.spatial_hash_np` is the host copy of `spatialHash` that
  tests use.
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
`pbf_solver.cu`, `dfsph_solver.cu`, `wake.cu`, `grid_params.py`, `simulation.py`

### 10.13 Spawn velocity damping (PBF/DFSPH stabilization)

PBF and DFSPH aggressively correct incompressibility, which causes violent outward
expansion when particles are spawned on a cubic lattice (surface particles have ~50%
density deficiency due to fewer neighbors).

**Fix**: A `velocity_damping` field in SimParams applies `vel *= (1 - damping)` in
all three finalize/integrate kernels (K_Integrate, K_PBF_Finalize, K_DFSPH_Finalize).

Ramp schedule (Python-side, in simulation.py):
- Duration: 30 substeps after preset load or solver switch
- Linear ramp: `damping = 0.8 * (1 - t/30)` → 0.8 at step 0, 0.0 at step 30
- Each ramp substep writes `c_sim` to every module that reads it with an
  async pinned copy (`Simulation._upload_sim_params_async`), queued before
  that substep's kernels on the same stream
- CUDA graphs stay valid during the ramp: the kernels read
  `c_sim.velocity_damping` when they run, so a graph captured on ramp step 1
  replays with each later step's value (bd-r5-review-n28.168)

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
  - For PBF and DFSPH, `density_in` is a separate buffer, `sorted_density_prev`.
  - WCSPH has no `density_in`: K_Step1 reads `rho_j = pos4_j.w` from the
    neighbour position it already loads (bd-r5-review-n28.163, see 20.1).
    `position.w` holds the previous density and is re-ordered with the
    position, so the values match the current sort order.
    K_Step1 writes the density into sorted `.w`, and `K_Integrate` carries it to
    the unsorted `position.w` for awake particles (STATIC and sleeping particles
    pass `.w` through unchanged). Until 2026-09-24 `K_Integrate` wrote `.w = 1.0`
    for awake particles, so from the second substep every awake neighbour had
    `rho_j = 1` and every `m_j / rho_j` term in step1 was about 1000 to 2500
    times too large. `test_wcsph_density_in_carries_previous_density` covers it.
  - PBF and DFSPH fill it from `sorted_density`. This removes the race, but the
    values are still in the previous sort order after a full re-sort. The
    effect is bounded noise in the weighting, not a blow-up.
  - Every read site (K_Step1, K_DFSPH_ComputeDensityAlpha,
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
  `notify_scene_changed()`. On an interval frame it first reads the freelist
  size (`_dead_count`) that the async readback copied at the end of the
  previous frame. That is the DEAD count one frame ago, with no host sync. If
  it shows no compaction due and at least `dead_reserve` dead slots, the frame
  does nothing more. Otherwise, and on the first frame after a scene change
  (when that value is unknown), it counts dead slots once (`num_active`, one
  sync) and tops the reserve up to `dead_reserve` DEAD slots. Deaths after the
  copy (brush kills, the last frame's substeps) show up in the next copy.
- **Compaction.** With the reserve on, compaction runs only when there are more
  than `dead_reserve + compact_threshold` dead slots. `World.compact()` drops
  every dead slot, parked or not, and `_maybe_compact` then re-creates the
  reserve at the new tail. Without WATER or ICE, compaction still runs at
  `compact_threshold` (100) dead slots.
- A steam particle claims the slot where it is written, so a parked slot jumps
  from the park lattice to the boiling water. The displacement tracker then
  forces a full sort, as for any reused DEAD slot.
- **Waiting for slots.** `K_Reactions` sets the spawn flag on WATER above
  `temp_boil` and writes `packed_info` only when the flag changes. When the
  freelist has fewer than 3 entries, `K_SpawnGas` leaves the flag set and
  writes nothing. A volatile read of `dead_count` skips the claim atomics
  while the freelist is visibly short; the `atomicSub` claim stays the
  authoritative check. WATER that cools to `temp_boil` or below while flagged
  has the flag cleared by `K_Reactions`, so it never spawns steam late.
  (bd-r5-review-n28.142)
- **Rebuild push.** `K_Reactions` pushes DEAD slots with one `atomicAdd` per
  warp (`__ballot_sync`, `__popc`, `__shfl_sync`). Each DEAD index is listed
  exactly once; the order inside the freelist is not defined.

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
  health and velocity. As STATIC it skips every solver's fluid path.
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
  `(0, 0, 0, -1)` for `MAT_DEAD`, so `K_ComputeColor` (see 10.11) marks
  every dead slot on the next frame, whatever its behavior bits, including
  slots killed by the brush or parked by `add_dead_reserve`. `particle.vert`,
  `particle_nonfluid.vert`, `ssfr_depth.vert` and `ssfr_thickness.vert` move
  any vertex with `color.w < 0` off screen with point size 0, so a killed
  particle vanishes at once and does not wait for compaction (every 60 frames,
  at least 100 dead). Live particles keep the behavior code in `color.w`
  (FLUID 0.0, GRANULAR 0.25, GAS 0.5, STATIC 0.75). The dead vertices are
  still fetched, so they cost vertex work but no fragment work.
  (bd-r5-review-n28.60)
- **Color outside a substep.** Nothing stores a color, so spawn, load,
  undo and kill need no color write. `K_ComputeColor` computes the color
  from the state on every frame, paused or not, so a paused scene shows the
  right colors, heat glow included, on the first frame after any of them.
  (bd-r5-review-n28.63, bd-r5-review-n28.145)

**Source**: `reactions.cu:K_Reactions`, `sph_shared.cuh:static_particle_boundary`,
`sph_shared.cuh:compute_color`, `integrate.cu:K_ComputeColor`, `shaders/particle.vert`,
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
  the source. The x* `.w` gets the child mass (12.1, mass carrier). It also
  zeroes `lambda`.
- **DFSPH.** `kappa` and `kappa_v` are zeroed. DFSPH skips GAS in its pressure
  solvers, so the slot's `alpha` from the pre-spawn density pass is not used.
  `predicted_position` is not written: DFSPH reuses it as `vt` scratch.
- **All solvers.** The spawn writes the
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

### 10.23 Particle dye diffusion (removed)

Particles carry no dye colour, and no kernel diffuses, integrates or sorts one
(bd-r5-review-n28.144). The old pipeline diffused a per-particle `float4` dye
with an SPH Laplacian in step1 (and in PBF ComputeLambda and DFSPH
DensityAlpha) and integrated it in the integrate and finalize kernels. It was
gated on a `dye_enabled` flag that no caller ever set, and no renderer read the
dye, so it never ran and never showed. It still moved 88 B per particle per
substep through the counting sort and the integrate/finalize kernels and held
56 MB of buffers at 1M particles. `SimParams` lost the flag and is 72 bytes.

Removing it cut the WCSPH substep at 1M particles from 11.27 ms to 10.48 ms
median (sort 1.80 to 1.54 ms, step1 3.02 to 2.68 ms, integrate 1.42 to
0.96 ms; `bench_substep.py`, RTX 5070 Ti shared with other jobs) with the same
position checksum. Dam Break after 10 substeps matches the old code within
the run-to-run spread of the atomics for all three solvers. Save files written
before the removal still load: `load_scene` ignores the extra array.

A real dye feature would need the painting brush to write it and
`compute_fluid_color` to blend it into the particle colour.

**Source**: `test_state_roundtrip.py::test_load_ignores_removed_dye_array`,
`test_common_cuh.py::test_struct_sizes`

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
| Vorticity computation | Step1 | ComputeLambda | DensityAlpha |
| Vorticity confinement | Step2 (eps=0.05) | Finalize (eps=0.001) | NonPressure (eps=0.05) |
| Surface normals | Step1 | ComputeLambda | DensityAlpha |
| Surface tension | Step2 (gamma=1.0) | Finalize (gamma=0.0) | NonPressure (gamma=1.0) |
| Drucker-Prager friction | Step2 (force-space) | Finalize (vel-space) | NonPressure (force-space) |
| Sleep/wake system | Integrate | Finalize | Finalize |
| Temperature/reactions | Integrate | Finalize | Finalize |

**STATIC wall volume convention** (bd-r5-review-n28.116). A regular STATIC
particle (STONE, METAL, WOOD, ...; not MAT_RIGID) has mass 0.02 whatever its
material, and every solver stores `max(rest_density, 1)` as its density:
K_Step1 (WCSPH), K_PBF_ComputeLambda and
K_DFSPH_ComputeDensityAlpha. So:

- Heat, normal and vorticity sums weight a wall neighbour by
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
- WCSPH uses adaptive dt set by `accuracy` and the velocity CFL (0.5 ms cap for WCSPH, 1 ms for WCSPH (Fast), see 4.2); PBF/DFSPH use fixed dt, split into more substeps when v_max nears the velocity clamp (4.2)

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
     d. x*_out = x* + dx, clamped to the world box and pushed out of SDF objects,
        written to the other predicted-position buffer (same kernel as c)
  4. v_new = (x* - x_old) / dt
  5. XSPH viscosity: v_new += c SUM (m_j/rho_j) (v_j - v_i) W_poly6
  6. Boundary handling + writeback
```

**Ping-pong buffers and mass carrier.** Steps 3c and 3d are one kernel,
`K_PBF_ComputeDelta`. It reads x* from one buffer and writes x*_out into the
other (`sorted_predicted_position` and `sorted_delta_position`), so every
particle in an iteration sees the same x* (Jacobi) without a separate apply
pass. STATIC copies its x* through. After N iterations the result is in
`sorted_predicted_position` for even N and `sorted_delta_position` for odd N;
N is fixed per graph capture and `_run_pbf_body` hands the right one to
Finalize. `.w` of x* holds the particle mass: `K_PBF_Predict` writes
`mass[i]`, and the neighbour loops of ComputeLambda, ComputeDelta and Finalize
read `m_j` from the float4 they already load. Reactions and `K_SpawnGas` run
after the first lambda pass and can change `mass` (material rescale, dead
slots, steam children), so the first ComputeDelta reads `mass[]` and writes it
into `.w` for the later passes. Only the last lambda call of the substep writes
the GRANULAR pressure normal, the one Finalize reads. The result is bitwise the
same as the old delta-then-apply pipeline (checked at 4 and 5 iterations).

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
`K_PBF_ComputeDelta` pass call `push_out_of_sdf`: any x* closer than
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
in the first ComputeLambda call (the one that also does heat and exposure),
as `rho = SUM_j m_j W_poly6`. Later iterations take an early return and leave that
value in `sorted_density`, so next substep's heat terms use the real
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
When `k=0`, `upload_pbf_params` uploads `s_corr_W_dq = 0`, so ComputeDelta
skips the s_corr ratio for every neighbour (no W/W_dq division, no ratio^4).

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
blast uses `atomicAdd` on x, y and z of each non-STATIC target. STATIC targets
get no velocity or x\* write, so STATIC keeps x\* = x (rigid boundary particles
push their body instead, see 22.6.1). The constraint iterations then start from the kicked x\*, and
`v_new = (x* - x_old) / dt` carries the kick. The Drucker-Prager `v_pred` (12.7)
reads `velocity` after the kick, so the kick is not treated as a constraint
correction. WCSPH and DFSPH pass NULL and are unchanged. Two limits:
- The final velocity is still clamped at `0.9 h / dt` (4.3 m/s at dt = 1/120),
  so the 25 m/s burst leaves the FIRE particle at the clamp.
- A kicked x\* can move several cells from where the grid put it, so the
  iterations of that substep miss some of its neighbours. The next full sort
  fixes it.

**World box walls in Finalize.** `K_PBF_Predict` and `K_PBF_ComputeDelta` clamp
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
  contributes to neighbor density sums. On do_heat path, exposure and heat
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

**Source**: `pbf_solver.cu:K_PBF_Finalize` (Drucker-Prager friction),
`pbf_solver.cu:K_PBF_ComputeLambda` (pressure normal output),
`solver_profiles.py` (`pbf_friction_ratio`, `pbf_friction_cohesion`),
`step2.py:build_granular_params()` (builds params uploaded to PBF module)

---

## 12.8 Surface Velocity Smoothing (Quality Mode)

Iterative Jacobi smoothing of FLUID surface particle velocities. This is
XSPH-style smoothing restricted to surface particles. It is not surface
tension: it has no cohesion or curvature term, so it cannot pull a surface
inward or form droplets. It only damps relative velocity at the free surface.
The code keeps the old "IST" (implicit surface tension) names
(`implicit_st.cu`, `sim.ist_*`). The UI calls it "Surface velocity smoothing".
WCSPH only, limited to < 100K particles.

Real implicit surface tension (Jeske et al. 2023, which makes the explicit
Akinci curvature force of §17.3 in `step2.cu` implicit) is a separate
feature that does not exist yet.

### Algorithm

For each Jacobi iteration (1-20 per substep):
```
For each FLUID particle i with neighbor_count < surface_threshold:
  w_surface = 1 - neighbor_count / surface_threshold
  dv_i = SUM_j (m_j/rho_j) * W_poly6(r_ij) * (v_j - v_i)
  S_i  = SUM_j (m_j/rho_j) * W_poly6(r_ij)
  a_i  = min(sigma * w_surface, 1 / S_i)
  v_new_i = v_old_i + a_i * dv_i
```

The sums run over FLUID neighbours j != i. The clamp keeps `a_i * S_i <= 1`,
so `v_new_i` is a convex combination of `v_i` and its neighbours' velocities
and the largest speed cannot grow between iterations (Jacobi stability). S is
about 1 at rest and can exceed 1 when a surface particle's own density is below
its neighbours'. Without the clamp, sigma 50 on a still 1000-particle droplet
grows a 1 m/s kick to 2e9 m/s in 7 iterations. With it, speed falls every
iteration (`test_implicit_st.py::test_surface_speed_never_grows`).

The clamp bounds each particle on its own, so it does not stop two particles
from overshooting each other: for an isolated pair the relative velocity is
multiplied by `1 - a_1 S_1 - a_2 S_2` per iteration. At sigma 2.0 that was
about -0.73 (a damped sign flip each iteration). At sigma 1.0 it was about
+0.13. The UI caps sigma at 1.0.

Surface particles (fewer neighbors) receive stronger smoothing. Interior,
sleeping and non-FLUID particles keep their velocity. Uses ping-pong velocity
buffers for proper Jacobi iteration (read from old, write to new).
`run_implicit_st` copies the input velocity into the scratch buffer once before
the first iteration, and `K_IST_Iterate` writes only the particles it smooths.
The unsmoothed particles then hold the same value in both buffers for the whole
pass, so the kernel skips their 16 B pass-through write each iteration. The
result is bit-identical to writing every particle
(`test_implicit_st.py::test_surface_only_writes_match_old_path`).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma` | 0.5 | Smoothing strength, UI range 0.01-1.0 (higher = stronger smoothing; clamped per particle) |
| `surface_threshold` | 25.0 | Neighbor count below which particle is "surface" |
| `num_iterations` | 4 | Jacobi iterations per substep, UI range 1-20. Even counts end in `sorted_velocity` and skip the copy-back |

### Pipeline Position

Runs between Step2 (force computation) and Integrate in the WCSPH pipeline.
Modifies `sorted_velocity` before integration uses it. The Jacobi ping-pong
uses its own buffer, `World.sorted_ist_scratch` (min(100K, max_particles) x float4,
allocated in the substep prologue when IST is on, outside any CUDA graph capture).
With an odd iteration count the result ends in the scratch buffer and is copied
back into `sorted_velocity` (one extra device copy); the default of 4 avoids it.
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
     (reactions and spawn run here)
     Pack the volume carrier into sorted position.w (see below)
  3. Compute non-pressure forces: f_visc, f_gravity
  4. Predict velocity: v* = v + dt * a_nonpressure   (gravity + viscosity/rho_i)
  5. Divergence-free solver (warm-started from the previous substep's
     accumulated kappa_v, scaled by 0.5):
     Apply the warm-start correction, then N_div iterations:
     dkappa_v_i = max(div(v*_i), 0) * omega * alpha_i / dt   [unilateral]
     v*_i -= dt * SUM_j m_j * (dkappa_v_i/rho_i + dkappa_v_j/rho_j) * grad_W_ij
            [fluid j; a STATIC j adds V_j * dkappa_v_i, see 13.3]
     kappa_v_i += dkappa_v_i                                  [warm-start total]
  6. Density solver (Jacobi iteration on pressure, N_dens iterations):
     Initialize p_rho2 from warm-start (previous frame, scaled by 0.5)
     Compute initial vt = clamp_v(v + dt * clamp_a(a_press)) from warm-started p_rho2
     For each iteration:
       Predict density from vt
       Jacobi-update p_rho2 from density residual
       Recompute vt from updated p_rho2
     The last vt pass writes the velocity itself: v* = vt
  7. Final position: x_final = x + dt * v*_final
  8. Boundary handling, writeback (kappa + kappa_v written back for warm-start,
     already scaled by their warm-start factors; position.w gets the density
     for the per-frame K_ComputeColor)
```

**Velocity buffer routing (bd-r5-review-n28.173):** NonPressureForces reads
`sorted_velocity` and writes `sorted_veleval` (the two must not alias). From then on
`sorted_veleval` is the substep velocity: CorrectVelocityDiv, ComputeKappaV,
ComputePressureAccel and Finalize read or update it, and foam generation reads it
after the substep. DensitySolverUpdate reads the predicted velocity `vt` from
`sorted_predicted_position` instead (13.4). `sorted_velocity` keeps the reordered start
velocity. The routing is fixed, never swapped per substep, because CUDA graphs bake
buffer pointers.

**Volume carrier in position.w (bd-r5-review-n28.184):** `K_DFSPH_PackVolume` runs
after reactions and spawn (they change mass and material) and writes into the sorted
`position.w`:
```
regular STATIC wall (not MAT_RIGID):  .w = m_j
every other particle:                 .w = V_j = m_j / max(rho_j, RHO_EPSILON)
```
The four solver loops (ComputeKappaV, CorrectVelocityDiv, ComputePressureAccel,
DensitySolverUpdate) read the neighbour volume from the position float4 they already
load. ComputeKappaV and DensitySolverUpdate load no `mass[j]` or `density[j]`;
CorrectVelocityDiv and ComputePressureAccel load `mass[j]` for a fluid neighbour inside
h, for the symmetric pair form (13.3). A wall stores its mass because
the loops use two divisors for it: the divergence and density-prediction sums use the
Akinci volume `m_j / rho0_i` (13.3), the pressure and divergence-correction sums use
`m_j / rho_j` with the wall's own stored density, which those two kernels load only for
a wall neighbour. The volumes are the same as before the carrier, except
for a particle that a reaction turned into a wall in this substep, whose stored density
is still its fluid density. The next reorder overwrites `.w`, so the carrier lives only
inside the substep; Finalize writes this substep's density into the unsorted
`position.w` for K_ComputeColor.

Kernels (`dfsph_solver.cu`, 9 in total): ComputeDensityAlpha (step 2), PackVolume (2),
NonPressureForces (3-4), ComputeKappaV and CorrectVelocityDiv (5),
CorrectDivAndPressureAccel (the last pass of 5 fused with the first sweep of 6, 13.4),
ComputePressureAccel and DensitySolverUpdate (6), Finalize (7-8).
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
dv_i = -dt * SUM_j c_ij * grad_W_ij
  fluid j:   c_ij = m_j * (kappa_v_i/rho_i + kappa_v_j/rho_j)
  STATIC j:  c_ij = V_j * kappa_v_i,  V_j = m_j/rho_j   (kappa_v_j = 0)
v*_i += dv_i
```

kappa_v is never negative, and a positive kappa_v pushes the pair apart.

**Pair form and momentum (bd-r5-review-n28.100).** Between two non-STATIC particles the
pair term here and in 13.4 is `m_j * (k_i/rho_i + k_j/rho_j)`. The impulse on i is
`m_i * m_j * (k_i/rho_i + k_j/rho_j) * grad_W_ij` and the one on j is the same with
`grad_W_ji = -grad_W_ij`, so the pair conserves linear momentum even where
rho_i != rho_j (free surfaces, material interfaces). The earlier `V_j * (k_i + k_j)` form
gave j the reaction weight `V_i`. A STATIC neighbour (wall or MAT_RIGID) keeps
`V_j * k_i`: it has no kappa of its own, the wall push is the same as before, and the
MAT_RIGID reaction `+m_i * V_j * k_i * grad_W_ij` still cancels the fluid impulse.
`k_i/rho_i` is computed once per particle; `m_j` is loaded only for a fluid neighbour
inside h.

`test_dfsph_pair_momentum.py`: a WATER block touching a MUD block, zero gravity, spawn
damping off, non-pressure pass and reactions stubbed out, 300 substeps. `|sum m v| /
sum m|v|` is 2.4e-8 with the symmetric form and 0.107 with the old one. The test also
checks that the divergence, pressure and speed clamps stay below half their limit, since
a clamp would break conservation (measured 0.007, 0.090 and 0.268 of the limit).
`test_rigid_coupling.py::test_fluid_body_momentum_balance[DFSPH]` passes with the
symmetric form (dp_fluid -0.0457 against a body impulse of +0.0457; the old form gave
-0.233 against 0.120).

Effects of the change:
- OIL on WATER in a closed box (half size 0.3, 0.2 m layers) stays stratified with both
  forms: after 3 s OIL COM_y -0.062, WATER -0.231, no OIL below the WATER COM.
- In the open 2 m domain the same blocks spread into a puddle one or two particles thick
  within 3 s. There, every conserving form tried (this one, rest-volume weights, and
  rest-volume density at material interfaces) leaves more WATER on top of OIL: 40% of
  OIL has WATER just above it at 3 s and 26% at 9 s, against 14% and 10% with the old
  form. A layer that thin has no hydrostatic gradient to sort it.
- An unconfined WATER cube rebounds harder off the domain walls: full-support
  rho/rho0 at 1 s is 0.927 (old form 0.930), OIL 0.930 (0.941).
- CorrectVelocityDiv and ComputePressureAccel read 4 B more per in-range fluid pair.
  `bench_substep.py`, DFSPH, 500K particles, GPU shared: divergence 4.31 to 4.49 ms,
  density_solve 5.46 to 5.69 ms, substep 13.8 to 14.0-14.2 ms (about 2 percent).

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
in an 11^3 block 566 of 1331 particles are below 20. `normal_in = None` turns the gate off.
`Simulation._run_dfsph_body` passes no `normal_in`, so the gate is off in the running
simulation, because with it on the flow spreads too much. A free-falling WATER droplet
(9^3 or 13^3 particles at spacing 0.02) after 0.3 s has a mean pairwise separation of
1.23 to 1.57 times the WCSPH value with the gate, and 1.04 to 1.18 times without it
(spawn damping on or off). The gated share of a dam-break splash grows from 8 to 67
percent of FLUID over 1 s as the flow thins, and the splash ends with 46 percent more KE.
The density solver (13.4) is not gated.

Iterations: 3 (Stable) or 2 (Fast). Each iteration requires two neighbor passes (compute
kappa_v, then correct velocity); the last correction shares its pass with the first
pressure sweep of the density solver (13.4). On a settling 36K block at the Stable profile the
positive divergence drops by 72 to 81 percent per substep (warm start plus 3 iterations).
Jacobi removes local modes fast but the block-wide mode slowly: 8 iterations reach 91
percent and 16 reach 94 percent.

**Warm-starting (bd-r5-review-n28.83):** the warm start is the accumulated stiffness,
as in SPlisHSPlasH. `sorted_kappa_v` holds the running total and is persisted across
substeps through the sort-reorder pipeline (kappa_v → sorted_kappa_v). K_DFSPH_Finalize
writes the total back times `dfsph_div_warm_start` (default 0.5; `c_dfsph.kappa_v_carry`),
so the next divergence solve starts from the scaled value and applies it once as a
velocity correction. With warm start 0 the carry is 0 and the host skips that pass and
zeroes the total (which also clears values from before a profile change). Each
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

**Two kernels per iteration** (bd-r5-review-n28.184):

1. **K_DFSPH_ComputePressureAccel**: pressure acceleration from current p/rho^2,
written as the clamped predicted velocity:
```
a_press_i = -SUM_j c_ij * grad_W_ij
  fluid j:   c_ij = m_j * (p_rho2_i/rho_i + p_rho2_j/rho_j)     [symmetric, 13.3]
  STATIC j:  c_ij = V_j * p_rho2_i,  V_j = m_j / rho_j (carrier, 13.1)
a_press_i = clamp(a_press_i, h/dt^2)     [velocity change at most h/dt per substep]
vt_i = clamp(v_i + dt * a_press_i, VELOCITY_LIMIT)   [bd-mzc.33: prevent Jacobi runaway]
```
STATIC, GAS and sleeping particles write `vt_i = v_i`. `vt` goes to
`sorted_predicted_position`; the last call of the substep writes it into the velocity
buffer instead, which applies the converged pressure. Each thread reads and writes only
its own velocity, so the in-place write is safe. With `dens_iters = 0` the initial call
is the last one.

**Fused first sweep (bd-r5-review-n28.192):** the initial call runs as
K_DFSPH_CorrectDivAndPressureAccel together with the last divergence correction (13.3):
the last warm-start pass when `div_iters = 0`, else the last iteration's increment.
CorrectVelocityDiv writes only `v_i` and ComputePressureAccel reads only its own `v_i`,
so one neighbour sweep accumulates both sums over the same `r`, `grad_W` and `V_j`:
```
v'_i = v_i - clamp(dt * SUM_j c_div_ij * grad_W_ij, h/dt)    [written to the velocity]
vt_i = clamp(v'_i + dt * clamp(a_press_i, h/dt^2), VELOCITY_LIMIT)
```
The result equals the two kernels in sequence
(`test_dfsph_fused_div_accel.py`). Its divergence rigid reaction is always added, the
pressure one only when `dens_iters = 0`; both go to the body in one atomic per pair.
With no divergence pass at all (warm start 0 and `div_iters = 0`) the host launches
plain ComputePressureAccel. 62 registers at (256, 4). This removes one sweep and one
launch per substep: at 1M particles (dam break, RTX 5070 Ti shared with other jobs)
the divergence stage went from 10.15 to 8.39-8.46 ms and the substep from 33.96 to
32.31-32.39 ms, with density_solve unchanged at 12.97 ms.

2. **K_DFSPH_DensitySolverUpdate**: predict density from `vt`, Jacobi-update pressure:
```
drho = SUM_j V_j * (vt_i - vt_j) . grad_W_ij   [V_j: Akinci volume, m_j/rho0_i for a wall]
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

**Velocity clamping before density prediction (bd-mzc.33)**: `vt = v + dt*a_press` is
clamped to `VELOCITY_LIMIT` before computing `drho`. Without this clamp, a large `a_press`
overshoot in one Jacobi iteration can produce a huge `drho` that drives the next iteration
further out: runaway Jacobi oscillation. The clamp prevents this feedback from diverging.
The `h/dt^2` clamp on `a_press` is applied inside the prediction too, so the density the
solver predicts comes from the same velocity the fluid gets. Both clamps run once per
particle in ComputePressureAccel instead of once per pair in DensitySolverUpdate.

The `A*p` feedback comes from including `a_press` in the density prediction, which
accounts for the effect of current pressure on neighbors. This is what makes multiple
iterations converge — pure velocity correction without pressure feedback would repeat
the same correction.

**Warm-starting:** Pressure values (`p_rho2`) are carried from the previous frame
through the sort-reorder pipeline (`kappa` → `sorted_kappa`). K_DFSPH_Finalize applies
the decay factor as it writes them back (bd-r5-review-n28.173), so the next density
solve starts from the scaled value:
```
kappa_out = p_rho2 * kappa_carry     (kappa_carry = warm_start, default 0.5 = carry 50%)
```
With `dfsph_warm_start <= 0` the carry is 1 (mu(I) still needs the unscaled value) and
the host zeroes `p_rho2` before the density solver instead. A profile change takes
effect on the writeback, so the first substep after it starts from the old factor.
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
K_DFSPH_ComputePressureAccel, warm start 0.5 for both solvers, alpha_limit 1.0.

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

**Neighbour bytes (bd-r5-review-n28.184).** Per neighbour, with the volume carrier and
`vt`: DensitySolverUpdate 60 B to 36 B, ComputeKappaV 44 to 36, ComputePressureAccel 32 to
24, CorrectVelocityDiv 32 to 24, and the ApplyPressureVelocity launch is gone (PackVolume
adds one 16 B per particle launch). `bench_substep.py` at 500K particles, Stable profile,
full sort every substep, GPU shared with other jobs, three runs each (median ms):

| stage         | before      | after       |
|---------------|-------------|-------------|
| divergence    | 5.22-5.39   | 4.97-5.04   |
| density_solve | 7.90-8.08   | 6.31-6.39   |
| substep total | 17.92-18.42 | 16.16-16.47 |

The substep is about 10 percent faster, the density solve about 20 percent.
Settling 36K WATER block and 48K dam break at 1, 2 and 5 s: compression error, KE,
99th-percentile speed and COM_y stay inside the run-to-run spread of the old code
(for example block at 5 s: comp 0.32-0.36 % before, 0.32-0.33 % after; KE 12.6-13.2 J
before, 12.9-13.3 J after).

**Source**: `dfsph_solver.cu:K_DFSPH_ComputePressureAccel`, `K_DFSPH_DensitySolverUpdate`,
`K_DFSPH_PackVolume`, `simulation.py` (step 5, density solver loop)

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
substep, which Finalize stored times `kappa_carry`; the kernel multiplies by
`kappa_uncarry = 1/kappa_carry` to get the unscaled value. The solver stores p/rho, not
p/rho^2: ComputePressureAccel uses `m_j * (p_i/rho_i + p_j/rho_j)` on the
stored values, which is `m_j * (p_i/rho_i^2 + p_j/rho_j^2)` when the stored
value is p/rho. So `p = p_prev * rho` (Pa); `p_prev * rho^2` would be about 4000x too
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

Heavy kernels use a 256-thread block with `__launch_bounds__(256, B)`. B is the
minimum number of resident blocks per SM the compiler must allow, so it caps
registers per thread: B = 4 caps at 64 registers (32 warps per SM), B = 3 at 80
(24 warps). A lower cap raises occupancy but can spill to local memory, and a
spill in a neighbour loop costs more than the lost occupancy.

### 15.1 Measured register and local-memory counts

Measured on sm_120 (RTX 5070 Ti, PTX JIT, `--use_fast_math`) with CuPy kernel
attributes `num_regs` and `local_size_bytes` (bd-r5-review-n28.191). Local
memory above 0 B is a spill or a stack array.

| Kernel | Bounds | Regs | Local | Other bounds tried |
|---|---|---|---|---|
| K_Step1 | (256, 4) | 64 | 0 B | (256, 3): 69 regs, 0 B |
| K_Step2 (rigid, vorticity) | (256, 3) | 80 | 48 B | (256, 4): 64 regs, 112 B |
| K_Step2_NoRigid | (256, 3) | 80 | 0 B | (256, 4): 64 regs, 24 B |
| K_Step2_NoVort | (256, 3) | 80 | 24 B | (256, 4): 64 regs, 80 B |
| K_Step2_NoRigidNoVort | (256, 4) | 64 | 0 B | (256, 3): 72 regs, 0 B |
| K_ComputePressure | (256) | 12 | 0 B | |
| K_PBF_Predict | (256, 4) | 29 | 0 B | |
| K_PBF_ComputeLambda | (256, 3) | 80 | 0 B | (256, 4): 64 regs, 16 B |
| K_PBF_ComputeDelta | (256, 4) | 53 | 0 B | |
| K_PBF_Finalize | (256, 4) | 64 | 8 B | |
| K_DFSPH_ComputeDensityAlpha | (256, 3) | 80 | 0 B | (256, 4): 64 regs, 40 B |
| K_DFSPH_PackVolume | (256) | 14 | 0 B | |
| K_DFSPH_NonPressureForces | (256, 4) | 64 | 8 B | |
| K_DFSPH_ComputeKappaV | (256, 4) | 40 | 0 B | |
| K_DFSPH_CorrectVelocityDiv | (256, 4) | 56 | 0 B | |
| K_DFSPH_ComputePressureAccel | (256, 4) | 56 | 0 B | |
| K_DFSPH_CorrectDivAndPressureAccel | (256, 4) | 62 | 0 B | |
| K_DFSPH_DensitySolverUpdate | (256, 4) | 43 | 0 B | |
| K_DFSPH_Finalize | (256, 4) | 56 | 0 B | |

K_Step1 does not spill at (256, 4), so its GRANULAR strain and FLUID
omega/normal accumulators stay separate. K_PBF_ComputeLambda and
K_DFSPH_ComputeDensityAlpha stay at (256, 3), because (256, 4) makes them spill.

### 15.2 K_Step2 variants

K_Step2's body is a template on two host-known switches, compiled into four
`extern "C"` kernels:

- `HAS_RIGID`: the MAT_RIGID two-way coupling branch. Its per-body sums stay
  live across the whole neighbour loop, which is what spills. Without it a
  MAT_RIGID particle takes the regular STATIC branch, the same as the runtime
  `d_rigid_bodies != NULL` check does.
- `HAS_VORT`: the vorticity-confinement eta sum. Off only when
  `vorticity_epsilon == 0`.

`step2.step2_kernel_name` picks the kernel: rigid bodies on when
`d_rigid_bodies` is passed, vorticity on unless the last
`step2.upload_granular_params` had `vorticity_epsilon == 0` (host-side value,
no GPU read). The live WCSPH profile has vorticity on (epsilon 0.05), so a
scene without rigid bodies runs K_Step2_NoRigid.

1M-particle WCSPH dam break (992,319 particles), dt 1 ms, full sort every
substep, shared GPU so timings are noisy:

| Measurement | K_Step2 (old path) | K_Step2_NoRigid |
|---|---|---|
| step2 stage, bench_substep median (2 runs each) | 3.85, 4.06 ms | 2.62, 2.56 ms |
| whole substep, bench_substep median | 7.19, 7.33 ms | 5.84, 5.80 ms |
| kernel alone, same inputs, interleaved | 3.89 ms | 2.55 ms |

With `vorticity_epsilon = 0` on the same inputs: K_Step2 3.62 ms, NoRigid
2.31 ms, NoVort 3.44 ms, NoRigidNoVort 1.82 ms. On the dam break every variant
gives bit-identical output to K_Step2. On a mixed FLUID, GRANULAR, GAS and
STATIC scene K_Step2_NoRigid's force differs by at most 3e-7 (largest value
1.2) from float rounding (`test_step2_variants.py`).

**Source**: `physics/kernels/step2.cu` (step2_body and the four kernels),
`step2.py:step2_kernel_name`, all .cu files in physics/kernels/ for the bounds.

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

The magnitude `|omega|` is stored in `.w`. WCSPH `K_Step1` also writes it into
the `.y` lane of the packed `sorted_pr2_omega` carrier (section 20.3b), which is
where `K_Step2` reads `|omega_j|` for FLUID neighbours. Particle i's own
`vort_i` still comes from the float4.

**Source**: `step1.cu` (K_Step1, FLUID branch),
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
it amplifies them. `eval_vorticity.py` measures this on a gravity-free 0.4 m
water cube with a Taylor-Green field (A = 0.3 m/s, WCSPH, 300 substeps). On
2026-09-25 net L_z drift was +1.5e-6 (eps 0), -5.1e-6 (eps 0.05) and -2.7e-6
(eps 0.10), with no trend in eps. Transverse RMS v_x under a pure shear is
7.3009e-3 (eps 0) and 7.3011e-3 (eps 0.10). So these values do not inject
measurable angular momentum. KE/KE0 is 0.836, 0.841 and 0.846 for the three
eps values: confinement feeds the vortex. `test_vorticity_confinement.py`
checks this over 100 substeps (measured gap 2.87e-3 of KE0, required 1e-3). A visual A/B at eps 0.01 and 0.05 has not been
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
Vorticity, strain rate, heat and exposure still skip MAT_RIGID.
PBF (K_PBF_ComputeLambda) and DFSPH (K_DFSPH_ComputeDensityAlpha) use the same rule.

Output: `normal_out[i] = float4(n_x, n_y, n_z, as_float(neighbor_count))`

The neighbor count is packed into `.w` as a reinterpreted float for efficient storage.

**Source**: `step1.cu` (K_Step1 neighbor loop),
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
| FOAM   | 1.0   | Strong horizontal damping; weak gravity in air, buoyancy inside fluid |
| BUBBLE | 2.0   | Buoyancy + gentle drag; becomes FOAM when it leaves the fluid |

### 19.3 Generation Criteria (K_FoamGenerate)

Per FLUID particle, after integrate:
```
trapped_air = |v_i|                          (speed → splash indicator)
wave_crest  = max(0, -dot(v_i, n_hat))      (moving away from fluid bulk)
kinetic     = 0.5 * |v_i|^2

phi = k_ta * trapped_air + k_wc * wave_crest + k_ke * kinetic
```

A particle qualifies when `phi > threshold` AND `neighbor_count < 25` (surface
proximity check). A qualifying particle emits one foam particle with probability

```
p = min(1, k_rate * (phi - threshold) * dt_emit)
```

so the expected rate is `k_rate * (phi - threshold)` per second (Ihmsen et al. 2012
use `n = dt * phi`). `dt_emit` is the sim time the launch covers, passed as a kernel
argument. The draw is `hash_rand(i, seed + 4)`.

Default parameters: `k_ta=0.5`, `k_wc=1.0`, `k_ke=0.2`, `threshold=2.0`,
`k_rate=5.0` 1/s (a surface particle with `phi = 4` emits about 10 per second).

Cadence: `K_FoamGenerate` runs once per frame, on the last substep. `step_frame`
passes `last_substep` to the step function, and `dt_emit` is the sim time since the
previous launch (`sim_steps * dt` in normal frames). Direct `_sim_step` callers keep
`last_substep=True` and emit every substep with `dt_emit = dt`. The emitted amount
therefore does not depend on the substep count: Dam Break peaks at about 37K foam
particles at 17 substeps per frame and about 41K at 34, well under the 200K pool.

Type selection:
- SPRAY: `v_y > 0.5` and `speed > 1.0` (fast upward motion)
- FOAM: `neighbor_count < 15` (surface particle with few neighbors)
- BUBBLE: otherwise (near-surface interior)

The RNG seed is `frame_ptr[0]`, the device counter `Simulation._frame_counter_d`
that `K_IncrementCounter` advances once per substep (the same counter `K_Reactions`
and the wake kernels read). It is unique per launch. `_frame_counter` would repeat
for launches within one frame. Generation runs outside the CUDA graph, because
`dt_emit` changes with the frame's substep count.

Spawn position: parent position + random jitter (±0.01).
Spawn velocity: parent velocity × scale (1.2 for spray, 0.5 for others).
Lifetime: type-dependent base × random(0.5, 1.5).

### 19.4 Physics (K_FoamPhysics)

Simple per-particle physics, no neighbor interaction.

In-fluid test: FOAM and BUBBLE read their own cell of the SPH hash grid built this
substep, `h = spatialHash(calcGridCell(p))`, and count as inside fluid when
`cell_start[h] < cell_end[h]` (two 4 B reads, no neighbour loop; the foam module
gets `c_grid` from `Simulation._upload_constants`). Any SPH particle in the cell
counts, including solids, and a hash collision with a far occupied cell also reads
as inside. `foam.foam_physics` called without cell tables passes NULL, and the
kernel then treats every particle as outside for FOAM and never converts BUBBLE.

**SPRAY**: `a = -v * drag + gravity`
- `drag_coeff = 2.0`
- Full gravity

**FOAM**: `a = (-vx*2, ay, -vz*2)`
- Strong horizontal damping (sits on surface)
- Outside fluid: `ay = gravity*0.1` (weak gravity, no vertical drag)
- Inside fluid: `ay = foam_buoyancy - vy*foam_water_drag`, with
  `foam_buoyancy = 1.5` m/s^2 and `foam_water_drag = 10` 1/s. Terminal rise speed
  is 0.15 m/s, so foam that leaves the top cell overshoots by about 1 cm under
  0.1 g and then settles back. FOAM therefore rests in the top occupied cell of
  the fluid.

**BUBBLE**: `a = -v*drag + (0, buoyancy, 0)`
- `buoyancy = 15.0` (rises toward surface)
- Half drag on vertical axis
- In an empty cell it becomes FOAM (`pos.w = 1`) with `vy = 0` before the step,
  so bubbles stop at the free surface instead of rising to the world ceiling.
  Dam Break at 3 s: 22 foam particles at the ceiling and 12 bubbles more than one
  cell above the local water top, against 7577 and 8844 without the test. The
  remaining ones sit in cells held by ice or lava, or in colliding hash cells.

Integration: symplectic Euler. World boundary clamp kills particles at floor.

SDF object collision: after the world clamp, each live particle is checked
against every analytic SDF object in `c_sdf_objects` (at most 16). When
`eval_sdf(p) < BOUNDARY_MARGIN` the particle moves to `BOUNDARY_MARGIN` outside
along `sdf_normal(p)`, and the inward part of its velocity relative to the
object surface (`v_obj + omega x r`) is removed: `v -= min(0, (v - v_obj).n) * n`.
There is no bounce and no friction term. A SPRAY particle that touches an object
becomes FOAM (type 1.0) and keeps its remaining lifetime, so splash that lands on
an obstacle rests on its surface. The objects are uploaded by
`rigid_bodies.upload_sdf_objects`, which includes the foam module; a foam module
compiled after the last upload gets it replayed on compile. Rigid body SDFs
(`d_rigid_bodies`) are not checked, so foam still passes through rigid bodies.

Death: when the lifetime runs out, or the floor clamp kills a particle, the
kernel writes -1 to both `foam_velocity.w` (lifetime, read by compaction) and
`foam_position.w` (type, read by `foam.vert`). The particle stops drawing on
the substep it dies.

Launch size: `max_foam` threads. The kernel reads the live count from device
memory and threads past it exit at once, so foam made since the deferred
readback (section 20.5) gets physics on its next substep.

With foam on, `K_FoamPhysics` is the last node of the substep body, so it is
captured in both CUDA graphs (full sort and gather only). Turning foam on or off
drops the graphs.

### 19.5 Compaction (K_FoamCompact)

Stream compaction runs once per frame at most, from `step_frame` after the
substep loop (`_maybe_compact_foam`). It runs every 8th frame, and every frame
while the deferred foam count is above 80% of the pool:
1. Atomic scatter alive particles to output buffer
2. `K_FoamClearTail` sets `pos.w = -1` for slots `[alive_count, max_foam)` of
   the output buffer. Those slots hold positions from an earlier cycle, and the
   draw count is the deferred readback, which can be above alive_count.
3. Device-to-device copy of the whole output pool back into `foam_position` and
   `foam_velocity` (2 x max_foam x 16 B, 6.4 MB at 200K). The buffers are not
   swapped, because the captured graphs hold the `foam_position`/`foam_velocity`
   pointers for `K_FoamPhysics`.
4. Device-to-device copy of alive_count → foam_count (no GPU→CPU sync)

### 19.6 Rendering

Foam rendered as separate draw call after main scene (both point sprite and SSFR modes):
- Additive blending (`GL_ONE, GL_ONE`)
- No depth writes (`glDepthMask(GL_FALSE)`)
- Small white point sprites with soft circular falloff
- Alpha varies by type: SPRAY=0.8, FOAM=0.5, BUBBLE=0.3
- `foam.vert` culls dead slots (`aPos.w < 0`) by placing them outside clip space
  with point size 0. The draw count is still the deferred readback, so new foam
  appears one frame late.

**Source**: `foam.cu`, `foam.py`, `world.py` (foam pool), `renderer.py` (foam draw),
`simulation.py:_run_foam_step`, `simulation.py:_maybe_compact_foam`

## 20. GPU Data Optimizations

### 20.1 Pack Density into position.w (OPT-4.1)

At the end of each particle, `K_Step1` writes `density[i]` into
`sorted_position[i].w`. Step2 then reads `rho_j = position[j].w` instead of
issuing a separate `__ldg(&density[j])` load per neighbor. This eliminates one
global memory load per neighbor interaction in the Step2 inner loop.

Pipeline order: Step1 (writes density and `.w`) → Step2

Applied to WCSPH path only. PBF computes density from predicted positions. DFSPH
keeps density in a separate buffer and puts the neighbour volume (not the density)
into the sorted `position.w` for its four solver loops (13.1, volume carrier).

K_Step1 also reads its own neighbour density from `.w` (bd-r5-review-n28.163).
`K_Integrate` writes this substep's density into the unsorted `position.w` and the
reorder gathers it, so at the start of K_Step1 `pos4_j.w` is the previous substep's
density of neighbour j. The heat, vorticity, normal and strain-rate weights
`m_j / rho_j` use it; `.w <= 0` falls back to the neighbour's `rest_density`. There is
no `density_in` buffer and no per-substep copy into `sorted_density_prev` for WCSPH.

K_Step1 writes `.w` for particle i in the same launch, so a neighbour whose thread has
already finished gives this substep's density instead of the previous one. Both are the
same particle's density one substep apart, and only `.w` changes, so `.xyz` is never
torn. The sort order within a cell already varies between runs, so WCSPH was not bitwise
reproducible before this either. A test that needs an exact `rho_j` must put the
density that K_Step1 will write into `.w` first (one warm-up call), because any other
starting `.w` is read as either value.

Measured at 992K particles (dam break, full sort every substep, shared GPU, two runs
each): step1 2.62 ms before, 2.32 ms after; whole substep 8.88 ms before, 8.58 ms
after. In a 250K dam break after 400 substeps, KE and mean density error of three
1e-6 m jittered starts overlap before and after (KE 6516 to 6517 vs 6514 to 6517,
error 0.4119 to 0.4122 vs 0.4113 to 0.4120).

**Source**: `step1.cu:K_Step1` (end of kernel, `rho_j = pos4_j.w` in the loop),
`step2.cu:K_Step2` (reads `pos4_j.w`), `simulation.py:_run_wcsph_body`

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
Instead, `K_Step1` writes per-particle pressure to `sorted_pressure[]`
(`pressure_out`) right after it computes density. Step2 reads `p_i` and `p_j`
via `__ldg(&pressure_in[i])`.

This eliminates the per-neighbor `powf()` call (Tait EOS) from the Step2 inner loop.

Pipeline: Step1 (writes density and pressure) → Step2

`K_ComputePressure` (`step2.cu`, `step2.py:compute_pressure()`) computes the same
pressure from a density array. It is test-only: the simulation never launches
it, and tests that set density by hand use it to build `pressure_in`.

**Source**: `step1.cu:K_Step1` (`pressure_out`), `step2.cu:K_Step2`,
`simulation.py:_run_wcsph_body`

### 20.3b Packed neighbour carrier (p / rho^2, |omega|)

`K_Step1` writes `sorted_pr2_omega[i] = float2(p_i / rho_i^2, |omega_i|)`
right after pressure and vorticity (rigid particles write `(0, 0)`).
`K_Step2` reads this 8 B pair for each FLUID neighbour j instead of
`pressure_in[j]` (4 B) and `vorticity_in[j]` (16 B, of which only `.w` was used),
and forms the pressure term as `pr2_i + pr2_j` with no per-pair divide.
`p / (rho * rho)` is the same expression K_Step2 used before, so the forces are
unchanged to float rounding (`test_step2_packed_carrier.py`, rtol 1e-5).

GRANULAR and GAS neighbours keep the `pressure_in[j]` read: GRANULAR needs
`p_j` for the mu(I) pressure floor, and `K_SpawnGas` rewrites the pressure of
new GAS slots between Step1 and Step2 without touching the carrier. The carrier
is valid only between Step1 and Step2 of one substep and is not reordered.
Passing no carrier (`pr2_omega_in=None`) makes K_Step2 use `pressure_in` and
`vorticity_in` for every neighbour, which tests that build inputs by hand rely on.
The `pressure_in[j]` read is issued before the distance check, together with
the position, packed_info, velocity and mass loads, and only for neighbours
that do not take the carrier. On frozen 1M inputs this cut K_Step2 by about
5% (water 4.87 to 4.59 ms, water plus sand 4.88 to 4.64 ms), with bitwise-equal
forces (bd-r5-review-n28.171).

At 1M particles (Dam Break, RTX 5070 Ti, shared GPU) K_Step2 median went from
about 5.1-5.2 ms to 4.6-4.8 ms in uncontended runs (bench_substep.py).

**Source**: `step1.cu:K_Step1` (`pr2_omega_out`), `step2.cu:K_Step2`
(`pr2_omega_in`), `world.py` (`sorted_pr2_omega`), `simulation.py:_run_wcsph_body`

### 20.3a No neighbour-list path

Every WCSPH neighbour loop scans the 27 grid cells directly; there are no
neighbour lists. A neighbour-list variant (`K_Step1_BuildNL`, `K_Step1_NL`,
`K_Step2_NL` in `neighbor_list.cu`) netted only about 6% at 1M particles
(`perf_benchmark.md`), so it was retired (bd-r5-review-n28.154). Its copy of the
step1 loop had to be kept in sync with `K_Step1` by hand.

### 20.4 Warm-start kappa_v (PERF-008) and per-solver reorder skips

The DFSPH divergence solver's kappa_v is carried through the counting sort
pipeline (K_GatherReorder, on full sorts and grid reuse), persisted via K_DFSPH_Finalize
writeback, and warm-started at the beginning of each substep. The stored value is
the sum of the scaled warm start and every divergence iteration's increment
(13.3). It is scaled by `dfsph_div_warm_start` (default 0.5), then applied as an
initial velocity correction before the divergence solver iterations.

The reorder fills only the sorted buffers the active solver reads.
`_run_grid_setup` passes None for the rest, and both reorder kernels skip the
gather and the store for a NULL output:

| Buffer | WCSPH | PBF | DFSPH | Reader |
|---|---|---|---|---|
| `sorted_kappa`, `sorted_kappa_v` | skipped | skipped | reordered | DFSPH pressure solvers |
| `sorted_velocity_h`, `sorted_temperature_h` | reordered | skipped | skipped | WCSPH step1, step2, spawn |

PBF has no warm-start state (see 12.4). Spawn zeroes the child's kappa and
kappa_v under DFSPH only. A skipped buffer keeps its unsorted values: only
K_DFSPH_Finalize writes the unsorted kappa and kappa_v, so a switch
DFSPH -> WCSPH -> DFSPH gathers the same warm start again. The pointer set
depends only on the solver, and a solver switch invalidates the CUDA graphs.

Measured sort stage medians (1M-particle Dam Break, full sort each substep,
shared GPU, same kernels with the old and new pointer sets): WCSPH about
1.33 ms to 1.11 ms, DFSPH about 1.51 ms to 1.25 ms. PBF at 992K went from about
4.0 ms to about 2.4 ms when it first skipped these buffers.

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

**Drift anchor (bd-r5-review-n28.54).** On each full sort `K_GatherReorder`
also writes the sorted position into `sorted_anchor` (float4, N). A grid-reuse
gather passes NULL and never writes it, and the sort permutation is frozen between full sorts, so
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
is kept. A captured graph bakes in which branch `_run_grid_setup` took, so
`Simulation._rigid_graph_key` (body count, and whether this clear runs) drops
the graphs when it changes (bd-r5-review-n28.182). A block with no anchor yet (after injection,
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

**Source**: `counting_sort.cu:K_GatherReorder` (anchor_out),
`integrate.cu:K_Integrate`, `pbf_solver.cu:K_PBF_Finalize`,
`dfsph_solver.cu:K_DFSPH_Finalize` (max_displacement_out, displacement_anchor),
`world.py` (`sorted_anchor`),
`integrate.cu:K_UpdateBoundaryParticles` (rigid boundary drift),
`rigid_bodies.py` (`boundary_anchor`, `boundary_anchor_valid`),
`simulation.py:_update_rigid_boundary`, `_count_substep_sort` and
`_update_sort_skip_decision` (latch, prediction and threshold logic),
`spawn.cu:K_SpawnGas` (+inf on spawn)

**In-place writeback (WCSPH, bd-r5-review-n28.196).** Within one frame every
substep takes the same branch, so on a grid-reuse frame the sort permutation
is the same for all its substeps. `step_frame` passes `in_place_ok` for every
substep but the last, and `_plan_substep_writeback` then runs K_Integrate with
`write_mode` 1: it writes position, velocity, packed_info, sleep counter and
temperature back to the sorted arrays at slot i, and rewrites the FP16
velocity and temperature mirrors. The next substep skips K_GatherReorder
(`_reorder_skip`), since the sorted arrays are already current. Wake runs on
the sorted arrays on those substeps; it is per particle and order-free. The
frame's last substep scatters to the unsorted arrays as before (`write_mode`
2, see 21.5), so the renderer, compaction, snapshots, save and the brush
always read current unsorted state after `step_frame`. Nothing else changes
the physics: each substep reads the same sorted values the gather would have
produced.

K_Integrate reads no neighbour slots on this path (the STATIC push-out needs
`cell_start`, which WCSPH does not pass), so writing slot i in place has no
race. The aliased pointers carry no `__restrict__`. In-place writeback is off
with rigid bodies, because `K_UpdateBoundaryParticles` moves the boundary
block in the unsorted arrays before each substep and only a gather carries
that into sorted space. It is off for PBF and DFSPH, whose finalize kernels
always scatter. Direct `_sim_step` callers keep the default
`in_place_ok=False`. A reuse frame uses three extra CUDA graph variants,
keyed by (skip reorder, in place) in `_graph_reuse_variants`.

At 992K particles (WCSPH, zero-gravity block, shared GPU), a reuse substep
went from 5.10 ms to 4.42 ms in the middle of a frame (no gather, K_Integrate
0.48 to 0.16 ms), 4.77 ms on the first and 4.73 ms on the last. A 4-substep
reuse frame on the CUDA graphs went from 22.7 ms to 19.8 ms (median).
Full-sort substeps are unchanged.

**Source**: `integrate.cu:K_Integrate` (`write_mode`),
`simulation.py:_plan_substep_writeback`, `_run_grid_setup`,
`_run_wcsph_body`, `step_frame`

## 10.13 Micropolar SPH (removed)

The simulation has no micropolar model. Particles carry no angular velocity,
and no kernel reads, relaxes or sorts one (bd-r5-review-n28.140).

An earlier version relaxed a per-particle `omega` toward `0.5 * curl_v` with
`nu_t = 0.1` in K_Integrate, K_PBF_Finalize and K_DFSPH_Finalize. No force
term read `omega`, so it had no effect on the flow. It still cost 64 B per
particle per substep in the counting sort and the integrate/finalize kernels,
and 32 MB of buffers at 1M particles. Removing it cut the WCSPH substep at
1M particles from 11.57 ms to 11.16 ms (sort 1.91 to 1.74 ms, integrate 1.68
to 1.41 ms, `bench_substep.py`, RTX 5070 Ti shared with other jobs) with the
same position checksum.

A real micropolar model would need the coupling term in step2 (Bender et al.
2017), which exchanges momentum between `omega` and the linear velocity, and
a relaxation time under one frame.

Until `simulation.py` and `benchmark.py` stop passing them, `World` keeps
empty `(0, 4)` placeholders named `angular_velocity` and
`sorted_angular_velocity`, and the Python wrappers of the sort, integrate,
finalize and spawn kernels accept and ignore the matching keywords.

**Source**: `test_micropolar_removed.py`

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

### 21.5 Reaction State Carry-Back (state-dirty bit)

K_Reactions and K_SpawnGas change health, lifetime and mass in sorted space.
Every substep re-gathers those arrays from the unsorted copies, so each change
must be scattered back to unsorted or it is lost. packed_info bit 21
(`STATE_DIRTY_BIT` in `state_dirty.cuh`) marks the particles those two kernels
wrote: every material transition, death, spawned steam child, the source water
it replaces, non-lethal corrosion and acid health loss, GAS lifetime decay, and
a DEAD slot whose leftover mass is cleared to 0.

WCSPH `K_Integrate` scatters health, lifetime and mass only when the bit is
set, and clears it in the packed_info it writes back, so the unsorted
packed_info never keeps it. Every other particle's unsorted copy already holds
the value it was gathered from. The PBF and DFSPH finalize kernels still
scatter the three arrays for every particle and pass the bit through; all
readers mask packed_info fields, so a set bit has no effect there.

In the same kernel, a particle that stays asleep stores its zero velocity only
when the gathered velocity is not all-zero bits: on its first sleeping substep,
or after a kick below the wake threshold. Later sleeping substeps skip the
store because the unsorted velocity is already zero.

On an in-place substep (20.5) K_Integrate leaves the bit set, since health,
lifetime and mass are already current in sorted space, and the scatter that
ends the frame carries them out. That scatter (`write_mode` 2) also stores
every sleeper's zero velocity, because the unsorted copy is older than the
sorted input there. The position store stays,
because `.w` carries the density.

At 1M particles (dam break, no reactions) the K_Integrate median went from
0.73 ms to 0.48 ms (bd-r5-review-n28.162).

**Source**: `reactions.cu`, `spawn.cu`, `integrate.cu`, `state_dirty.cuh`

---

## 22. Rigid Body System

The rigid body system provides solid-fluid interaction via two mechanisms:
- **SDF collision** (Phase A): Analytical signed distance fields for particle-object collision
- **Akinci boundary coupling** (Phase B): Boundary particles for two-way hydrodynamic forces

SDF objects live in `__constant__ c_sdf_objects[16]` and `c_num_sdf_objects`
in the integrate, PBF and DFSPH modules. With a kinematic motion,
`SDFManager.update_kinematics` sets the pose for the current `sim_time`
before each substep and `upload_if_dirty` writes the table with async pinned
copies (`PinnedUploadRing`), so the particles see the pose of the substep
they run in. The count is written only when it changes (bd-r5-review-n28.168).

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

### 22.2.1 Positional Collision Against Rigid Bodies

The Akinci force from a body's single boundary layer (one 2 cm spacing)
cannot stop a body that moves more than about one spacing per substep. A
12 cm box driven down into still water at 3 m/s left up to 444 (WCSPH) and
36 (PBF) water particles inside the hollow box. So every non-STATIC particle
also collides with each rigid body's own SDF (bd-r5-review-n28.123).

`rigid_body_boundary` runs in `K_Integrate`, `K_PBF_Finalize` and
`K_DFSPH_Finalize`, after the SDF object loop. For each body b:

```
(c, q) = pose of b advanced by one dt with its lin_vel v_b and ang_vel w_b
         (rigid_integrate_pose, the same update K_IntegrateRigidBodies uses)
r      = pos - c
skip b if |r| >= R_bound + BOUNDARY_MARGIN      (R_bound = |half_extents| or radius)
d, n   = sdf_box(q^-1 r, half_extents), q * sdf_box_normal(...)    (box)
         sdf_sphere(r, radius), r / |r|                             (sphere)
if d < BOUNDARY_MARGIN:
    pos += (BOUNDARY_MARGIN - d) * n
    v_s   = v_b + w_b x (pos - c)
    v_rel = vel - v_s,  v_n = dot(v_rel, n)
    if v_n < 0:
        vel -= (1 + e_b) * v_n * n
        v_t  = v_rel - v_n * n
        vel -= min(mu_b * |v_n| / |v_t|, 1) * v_t
```

e_b is the body's restitution (lin_vel.w) and mu_b its friction (ang_vel.w).

- **End-of-substep pose.** The fluid kernels run before
  `K_IntegrateRigidBodies`. Testing the current pose would leave a particle
  in front of a moving face up to v_b * dt inside once the body moves. For a
  kinematic body the advanced pose is exactly the pose the body gets; for a
  dynamic body it misses only this substep's velocity change.
- **No reaction on the body.** The push is a positional safety net. The
  fluid force on the body is still the Akinci pressure coupling (22.5).
- **Sleeping particles.** `K_Integrate`'s geometry wake test also wakes a
  sleeping particle that is strictly inside a body (d < 0) at that pose.
- **Body array access.** `K_PBF_Finalize` and `K_DFSPH_Finalize` use their
  `d_rigid_bodies` argument (NULL when there are no bodies). `K_Integrate`
  reads the `c_rigid_bodies` constant, which `RigidBodyManager.upload()`
  writes with `c_num_rigid_bodies`. A new `RigidBodyManager` writes a zero
  count and its own array address to the integrate module, so a count left by
  an earlier manager never points `K_Integrate` at a freed array.
- **Cost.** One bounding-sphere test per body per particle, and the SDF only
  near a body. WCSPH at 1M particles with one body: substep median 13.26 ms
  with the push against 13.28 ms without (inside noise on a shared GPU).

**Source**: `sph_shared.cuh:rigid_body_boundary()`, `rigid_body_sdf()`,
`rigid_body_penetrates()`; test `test_rigid_fluid_sdf_push.py`

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
term is in K_Step1, K_DFSPH_ComputeDensityAlpha and
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

**Viscous coupling**: the boundary velocity is
```
v_b = v_body + omega × (r_b - COM)
```
`K_UpdateBoundaryParticles` writes it into the boundary particle's velocity
each substep and the sort carries it into `sorted_velocity` and its FP16
mirror, so K_Step2 and K_DFSPH_NonPressureForces read it as the neighbour's
velocity `v_j` and load only the body `position` (the torque arm) per pair
(bd-r5-review-n28.195). K_Step2 reads `v_j` from the FP16 mirror and pairs it
with the mirrored `v_i` (FP16 pairing rule, Section 3.3); DFSPH uses FP32 for
both. Measured with a spinning, moving kinematic box: the sorted FP32 velocity
matches the recomputed `v_b` to 7e-8 m/s, the FP16 mirror to 2.4e-4 m/s.
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

Accumulation (bd-r5-review-n28.182): K_Step2 sums the reaction on the first
body a particle touches in registers and writes it once after the neighbour
loop, 6 atomics per particle. The viscous part is scaled by the implicit
factor there. The pressure part `F_p = -m_i * fs_i * pressure_precalc * sum F_on_fluid`
is summed as a force only: every Akinci term is parallel to `r = x_i - x_j`, so
```
sum_j (x_j - COM) x F_j = sum_j ((x_i - COM) - r_j) x F_j = (x_i - COM) x F_p
```
A pair on any other body is written at once with `atomic_accumulate3`. A second
register sum of torque as well spilled K_Step2 (local memory 40 to 88 B) and
cost 0.8 ms per substep at 1M particles, so there is none.

Torque is summed about the COM, not about the world origin with `COM x F`
subtracted later. The pair forces on a body are large and cancel, so the
float32 error of an origin sum scales with `|COM| * sum |F_pair|`, not
`|COM| * |F|`. For a box spun at 1 rad/s 2 m from the origin, an origin sum
was off by 5e-2 (PBF), 2.5e-3 (DFSPH) and 5e-5 (WCSPH) on the first substep;
the COM sum matches the same scene at the origin to 1.1e-3, 3e-5 and 4e-6
(`test_rigid_coupling.py::test_offset_spinning_box_torque`).

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
| DFSPH divergence | `K_DFSPH_CorrectVelocityDiv` (last pass in `K_DFSPH_CorrectDivAndPressureAccel`), every pass incl. warm start | `dv_i = -dt * V_b * kv_i * gW` | `F = +m_i * V_b * kv_i * gW` |
| DFSPH density | `K_DFSPH_ComputePressureAccel` (first call in `K_DFSPH_CorrectDivAndPressureAccel`), final call only | `a_i += -V_b * p_i * gW` (p = p_rho2) | `F = +m_i * V_b * p_i * gW` |

The PBF position correction over a substep is an impulse `m_i * dx / dt` and a
force `m_i * dx / dt^2`, summed over the iterations. The DFSPH density solver
runs ComputePressureAccel `dens_iters + 1` times but only the last `a_press` is
applied (that call writes the velocity), so `simulation.py` passes the rigid buffers on
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

### 22.6 Force Accumulation

`atomic_accumulate3(target, val, body_id)` adds a float3 to one body's slot of
the `[MAX_RIGID_BODIES * 4]` force or torque accumulator with three global
`atomicAdd`. There is no warp reduction: callers sit in divergent
neighbour-loop branches where lanes may hold different bodies, and a
full-warp shuffle would blend them (bd-r4-epic-x2j.1).

- WCSPH K_Step2 sums a particle's reaction on its first body in registers and
  writes it once (22.5, reaction forces): 6 atomics per particle
  instead of 6 per pair plus 6.
- PBF (K_PBF_ComputeDelta, K_PBF_Finalize) and DFSPH (K_DFSPH_NonPressureForces,
  K_DFSPH_CorrectVelocityDiv, K_DFSPH_ComputePressureAccel) write each pair at
  once. A per-thread register or `__shared__` sum there made those kernels 3 to
  5 percent slower at 1M particles with a submerged 0.4 m box, more than the
  atomics cost (bd-r5-review-n28.182).

Both forms give the same per-body sums up to float summation order: on
Archimedes Pool the per-substep force and torque match the per-pair form
within 1e-6 relative (WCSPH), the same size as the run-to-run atomic noise.

**Source**: `sph_shared.cuh:atomic_accumulate3()`, `step2.cu:K_Step2`

### 22.6.1 Blast Wave Force on Rigid Bodies

`K_BlastWave` gives every neighbour j within h of a fresh gunpowder blast the
radial kick `dv = 15 m/s * (1 - r/h) * r_hat`. It writes no velocity to STATIC
targets: `K_Integrate` zeroes STATIC velocity and `K_UpdateBoundaryParticles`
rewrites rigid boundary velocity from the body state, so the kick would be
lost. A MAT_RIGID boundary particle instead adds a force and torque to its
body:

```
F_body   += m_j * dv / dt          // m_j = RIGID_PSI_RHO0 * V_b (mass slot)
tau_body += (x_j - x_COM) x (m_j * dv / dt)
```

This is the impulse `m_j * dv` spread over one substep, so the body gains the
momentum a particle of that mass would have gained. The accumulators are zeroed
at the start of the solver body (`_run_graph_body`), step2 adds the fluid forces to them, and
`K_IntegrateRigidBodies` applies the sum at the end of the same substep. Blasts
are rare, so the kernel uses plain `atomicAdd` without warp reduction. All three
solvers pass the rigid buffers when bodies exist; with no bodies they pass NULL
and rigid particles in range are skipped.

**Source**: `reactions.cu:K_BlastWave`, `reactions.py:compute_blast_wave`,
`simulation.py:_run_reactions_spawn`

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
body, so the fluid reaction stays readable until the next substep's solver
body clears them. Rigid-rigid collision still skips kinematic bodies as movers.

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

The kernel runs one thread per body in a single block. It first copies all bodies into shared memory and syncs, and every read of another body comes from that copy, so each body-body contact sees the pre-step state of both bodies even though each thread writes its own body back to global memory (bd-r5-review-n28.131). Each thread still applies its own contacts one after another to its own velocity, so a body with several contacts does not see a strictly pairwise-symmetric result.

**Source**: `integrate.cu:K_RigidBodyCollisions`

### 22.9 Boundary Particle State Sync

After rigid body integration, boundary particles must be updated to match the new body state:

```
r_world = COM + R * r_local          // position
v_world = v_body + omega × r_world   // velocity
```

Runs as `K_UpdateBoundaryParticles`, one thread per boundary particle.

**Record layout**: `rbm.boundary_data` is `(N_b, 4)` float32, one float4
(16 B) per boundary particle: `(x_local, y_local, z_local, psi_b)`.
`rbm.boundary_body_id` is `(N_b,)` int32 with each particle's parent body
index. The kernel does one 16 B load of the record and one 4 B load of the
body id, then reads only `position`, `rotation`, `lin_vel` and `ang_vel` of
the body (64 of its 96 B). Surface normals are not stored, because no kernel
uses them. Records are grouped by body, in body order. Save files do not
store these arrays: `load_scene` rebuilds them from each body's build
parameters.

**Slot layout**: all boundary particles of all bodies form one contiguous block
`[boundary_offset, boundary_offset + N_b)` in the unsorted arrays, inside
`world._high_water`. The kernel writes position, velocity and mass (psi_b) at
`rbm.boundary_offset + i` every substep.

- `inject_boundary_particles()` writes the block once, at the current
  `_high_water`, then raises `_high_water` by `N_b`. It also sets packed_info
  (MAT_RIGID, STATIC, body id in bits [20:13]), density, temperature,
  health, lifetime and sleep_counter. The block's color comes from the
  MAT_RIGID row of the material table through `K_ComputeColor`. It syncs, so it runs only on scene events.
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
- The undo ring (`SnapshotRing`) is sized to a 256 MB VRAM budget, 46 B per
  particle per slot, clamped to 2..12 slots (5 slots at 1M particles, 11 at
  500K). All slots are allocated at construction. Health, lifetime and mass
  are stored as float16 in a slot (relative error under 0.05% after
  restore); position, velocity, packed_info and temperature stay
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
| Force accumulation | Per-particle register sum in K_Step2, per-pair atomicAdd in PBF/DFSPH | See 22.6: a register sum spills or slows the PBF/DFSPH kernels |
| SDF vs boundary | Both coexist | SDF for collision, boundary particles for pressure/viscosity |
| Quaternion convention | (x,y,z,w) where w=scalar | Matches CUDA float4 layout |
| CUDA graph | On with bodies present | The accumulators are zeroed by two memset nodes at the start of the graph body; boundary moves and body integration run outside the graph. `_rigid_graph_key` drops the graphs when the body count or the boundary drift clear changes (bd-r5-review-n28.182), and `reset_scene_objects` and preset load drop them on every scene change (bd-r5-review-n28.190) |
