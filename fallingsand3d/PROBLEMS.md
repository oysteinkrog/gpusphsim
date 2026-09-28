# Known Problems and Improvement Opportunities: fallingsand3d

Compiled from multi-model reviews (Gemini 3.1 Pro, GPT 5.2 Pro, Claude) of
all physics kernels, material parameters, solver profiles, reaction system,
and scene presets: 8 AI reviews across 2 rounds in February 2026, then the
round-5 codebase review of 2026-09-24 (beads `bd-r5-review-n28.*`).

Last updated 2026-09-28. Total: 57 items (47 fixed, 2 confirmed, 3 investigate,
2 deferred, 3 refuted), plus 3 architectural notes in section F.

---

## Status Key

- **FIXED**: code change applied and committed
- **CONFIRMED**: verified as a real issue, fix planned
- **INVESTIGATE**: needs testing to confirm impact
- **DEFERRED**: known limitation, acceptable for now
- **REFUTED**: the claim was wrong after code verification

---

## A. Physics Kernel Bugs (from GPT 5.2 Pro review, 2026-02-27)

### A1. FIXED: DFSPH granular compliance inverted
- **File:** `dfsph_solver.cu:259`
- **Was:** `denom * 0.7f` (makes alpha *larger* = stiffer, opposite of intent)
- **Fix:** `alpha * 0.7f` (makes alpha smaller = more compliant)
- **Commit:** d4727ba

### A2. FIXED: PBF double Boussinesq buoyancy
- **File:** `pbf_solver.cu` K_PBF_Predict + K_PBF_Finalize
- **Was:** Thermal convection applied in both predict and finalize (2x buoyancy)
- **Fix:** Removed from Finalize; PBF derives v from corrected positions
- **Commit:** d4727ba

### A3. FIXED: PBF STATIC lambda non-zero on do_heat path
- **File:** `pbf_solver.cu` K_PBF_ComputeLambda
- **Was:** STATIC fell through for heat accumulation and computed lambda != 0
- **Fix:** Force lambda=0, density=rho0 for STATIC after neighbor loop
- **Commit:** d4727ba

### A4. FIXED: PBF GAS doesn't skip constraints
- **File:** `pbf_solver.cu` K_PBF_ComputeLambda + K_PBF_ComputeDelta
- **Was:** GAS went through full constraint solve (incompressible bubble)
- **Fix:** Early return with lambda=0, delta=0 for GAS
- **Commit:** d4727ba

### A5. FIXED: W_poly6 FP safety
- **File:** `sph_shared.cuh:83`
- **Fix:** Added `fmaxf(diff, 0)` clamp for FP rounding safety

### A6. FIXED: EOS powf overflow at extreme densities
- **File:** `sph_shared.cuh:132`
- **Fix:** Capped density ratio at 10.0 to prevent NaN/Inf from powf

### A7. DEFERRED: PBF s_corr_n hardcoded to 4
- **File:** `pbf_solver.cu` K_PBF_ComputeDelta (artificial pressure block)
- **Status:** DEFERRED. s_corr is off in every profile (`pbf_s_corr_k = 0.0`), and
  `pbf_solver.py` then uploads `W_dq = 0`, so the kernel skips the s_corr branch.
  No practical impact at h = 0.04.
- **Note:** `c_pbf.s_corr_n` is uploaded from `SolverProfile.pbf_s_corr_n`, but the
  kernel computes `ratio^4` and never reads it. A profile that sets s_corr_n to a
  value other than 4 would be silently ignored.

### A8. FIXED: Legacy DFSPH kernels deleted
- **Files:** `dfsph_solver.cu`, `dfsph_solver.py`
- **Fix:** Deleted the five kernels that were never launched (K_DFSPH_PredictPosition, ComputeDensityAdv, ComputeKappa, ComputeKappaFromVelocity, CorrectVelocityDens) and their Python wrappers. The Jacobi solver on p/rho^2 is the only DFSPH density solver (bd-r5-review-n28.11).

---

## B. Material Property Issues (from Gemini 3.1 Pro review, 2026-02-28)

### B1. FIXED: Density scaling inconsistent across materials
- **File:** `materials.py`
- **Severity:** CRITICAL (breaks buoyancy behavior)
- **Problem:** Water uses rest_density=2500 (2.5x real-world for force_scale convention) but several materials used real-world densities, creating wrong buoyancy ratios.

| Material | Was | Should Be | Ratio vs Water (was) | Ratio vs Water (correct) |
|----------|-----|-----------|----------------------|--------------------------|
| WATER    | 2500 | 2500      | 1.00                 | 1.00                     |
| OIL      | 2500 | 2000-2200 | 1.00 (won't float)   | 0.80-0.88                |
| LAVA     | 2500 | 6500-7000 | 1.00 (won't sink)    | 2.60-2.80                |
| SAND     | 2500 | 3500-4500 | 1.00                 | 1.40-1.80                |
| ICE      | 917  | 2300      | 0.37 (rockets out)   | 0.92                     |
| WOOD     | 600  | 1250-1500 | 0.24 (rockets out)   | 0.50-0.60                |

- **Fix:** rest densities now follow the "Should Be" column. Particle mass follows
  them too: FLUID and GRANULAR particles spawn with `0.02 * rest_density / 2500`
  (`world.spawn_mass`, bd-r5-review-n28.53), so each material starts at its own
  rho0 instead of at 2500 (SAND had spawned at 0.63 rho0, LAVA at 0.39). STATIC and
  GAS keep 0.02. The remaining gap is at interfaces: a light particle among heavy
  ones still reads the heavy kernel density (PHYSICS.md 7.3).

### B2. FIXED: LAVA EOS stiffness too low
- **File:** `materials.py`
- **Problem:** LAVA k=30 vs WATER k=500. Lava was 16x more compressible and collapsed under its own weight.
- **Suggested:** k=1500-2000

### B3. FIXED: ICE/WOOD EOS stiffness too low
- **Problem:** k=15 for both. They behave like jello if they become dynamic particles.
- **Suggested:** k=1000

### B4. FIXED: Metal thermal conductivity retuned
- **File:** `materials.py` (METAL thermal_conductivity=50)
- **Problem:** May violate thermal CFL with explicit Euler integration
- **Suggested:** 5-10

### B5. INVESTIGATE: Fire heat capacity too high
- **File:** `materials.py` (FIRE heat_capacity=1000, unchanged)
- **Problem:** Fire retains heat too long and does not dump energy dramatically
- **Suggested:** 100

### B6. FIXED: Viscosity units comment misleading
- **File:** `common.cuh` (MaterialProps.base_viscosity)
- **Was:** `// Pa*s`
- **Fix:** The comment now describes the per-solver convention (dynamic viscosity mu
  for FLUID/GAS, used as `mu*m_j*lap/(rho_i*rho_j)` in WCSPH and DFSPH).
- **Commit:** d4727ba

---

## C. Solver Tuning Issues (from Gemini 3.1 Pro review, 2026-02-28)

### C1. FIXED: WCSPH XSPH epsilon too high
- **File:** `solver_profiles.py`
- **Problem:** xsph_epsilon=0.8 applies massive velocity smoothing, drowning material viscosity differences. Water looks like syrup, oil/lava indistinguishable.
- **Suggested:** 0.05-0.1

### C2. FIXED: Acceleration clamp too restrictive
- **File:** `simulation.py` (accel_max in SimParams)
- **Problem:** accel_max=30 m/s^2 (about 3 g) mutes collision splashes and explosive interactions
- **Suggested:** 150-300 m/s^2 (velocity_limit=10 already prevents escape)

### C3. INVESTIGATE: DFSPH timestep overly conservative
- **File:** `solver_profiles.py`
- **Problem:** The Stable DFSPH profile runs dt=1/300. DFSPH is implicit and designed for larger dt.
- **Suggested:** dt=1/120-1/150 (2 substeps/frame), saving GPU budget
- **Risk:** May need more density iterations to compensate
- **Current state:** bd-r5-review-n28.37 (commits 9ae8ec5, 731a2ae, 06053cf) cut the
  Stable profile to 3 div + 4 density iterations at dt = 1/300 (5400 neighbour
  sweeps per simulated second, was 10200) and set the Fast profile to 2 + 3 at
  dt = 1/200. dt = 1/120 and 1/150 were in the sweep plan, but no result for them
  is recorded, so the question stays open.

### C4. INVESTIGATE: PBF iteration count insufficient for deep columns
- **File:** `solver_profiles.py`
- **Problem:** 6 iterations (Stable) or 4 (Fast) at dt=1/120 may not propagate corrections through 100-particle columns
- **Suggested:** 8-10 iterations for stable profile
- **Current state:** no deep-column measurement exists yet.

---

## D. Reaction System Issues (from Gemini 3.1 Pro review, 2026-02-28)

### D1. FIXED: Steam/water phase transition flickering
- **File:** `reactions.cu:45-46`
- **Problem:** STEAM_CONDENSE_TEMP=373 and WATER_BOIL_TEMP=373 gave zero hysteresis. Particles at the boundary temperature flickered between FLUID and GAS, causing pressure spikes.
- **Fix:** Lower STEAM_CONDENSE_TEMP to 360 (13K hysteresis band)

### D2. FIXED: Missing melting reactions
- **File:** `reactions.cu`
- **Problem:** Stone/sand heat up from lava contact but never melt
- **Fix:** Add STONE to LAVA at temp>1500K, SAND to LAVA at temp>1700K

### D3. FIXED: Fire vanishes instead of becoming smoke
- **File:** `reactions.cu:243`
- **Problem:** MAT_FIRE lifetime expiry went to MAT_DEAD (invisible). Combustion should leave a trail.
- **Fix:** Fire becomes MAT_SMOKE with a new lifetime (see J8)

### D4. REFUTED for condensed materials: Heat exchange values too high
- **File:** `materials.py` _INTERACTION_PAIRS
- **Claim was:** heat_exchange up to 60 (LAVA+ICE) can overshoot 100% of the
  temperature gap per substep and make temperatures oscillate.
- **Actual:** the heat term is `dTdt_i = [45/(pi h^6)] / (rho_i cp_i) * SUM_j k_ij
  (m_j/rho_j) (T_j - T_i) (h - r) * heat_boost` (PHYSICS.md 9.1). Upper bound for a
  particle whose whole support (spacing 0.02, h = 0.04, SUM (h - r) = 0.303 m) is the
  other material, with m_j/rho_j = 0.02^3 and the longest substep (PBF, 1/120 s):
  LAVA next to METAL (heat_exchange 30) moves 0.26 of the gap per substep, METAL
  next to LAVA 0.19, ICE next to LAVA (60) 0.15, LAVA next to ICE 0.11, every other
  condensed pair under 0.08. All are well under 1, so there is no overshoot. METAL
  and ICE are STATIC with mass 0.02, so their real m_j/rho_j is smaller and the true
  numbers are lower still.
- **GAS pairs:** GAS with non-GAS neighbours uses a per-pair cap of half the gap
  (bd-r5-review-n28.94, PHYSICS.md 9.1 "Cross-phase pairs").

### D5. FIXED: Explosion speed too low
- **File:** `reactions.cu:57`
- **Problem:** EXPLOSION_SPEED=5.0 was damped by viscosity and the pressure solver to a soft "puff"
- **Suggested:** 25-30 m/s for visually dramatic explosions

### D6. FIXED: Missing interactions
- **Missing:** DIRT+WATER wetting (like sand), LAVA+SAND heating toward melt
- **Fix:** Add `(DIRT, WATER, 0.5, 5.0)` and `(LAVA, SAND, 0.0, 25.0)` to interaction pairs

### D7. FIXED: Acid not consumed during reactions
- **Problem:** A single acid particle corroded unlimited stone, with no mass loss mechanism
- **Fix:** Acid health drops by `exposure_corrode * dt * 0.5` while it corrodes (`reactions.cu`, acid consumption block)

---

## E. World and Preset Issues (from Gemini 3.1 Pro review, 2026-02-28)

### E1. FIXED: Gravity too high for "miniature world" feel
- **Problem:** g=9.81 in a 2m cube: objects fall top-to-bottom in 0.63s, feels frantic/small
- **Suggested:** g=3.0-4.0, so splashes linger and avalanches feel massive

### E2. FIXED: Presets underutilize particle budget
- **Problem:** Most presets spawned about 130K of 500K capacity and looked sparse
- **Suggested:** Flagship presets should target 300K+ particles (leave 150K for reactions)
- **Fix:** f9a28d0 (2026-03-08) made Sand Castle, Dam Break, Volcano, Waterfall and
  Erosion denser. Lattice upper bounds per preset (before exclude and shape filters):
  Sand Castle 381K, Dam Break 336K, Lava Lamp 296K, Multi-Fluid 277K, Volcano 202K.
  Small demo scenes (SDF Hourglass 6K, Acid Rain 19K plus its spawner) stay small by
  design.

### E3. FIXED: No initial velocity support for spawned blocks
- **Problem:** All spawned material started at rest, which limits dramatic openings
- **Fix:** f9a28d0 added `velocity=` to `World.spawn_cube` and `spawn_sphere`. Presets
  use it: Dam Break, Waterfall, Erosion, Submarine, Canyon and Whirlpool.

### E4. FIXED: No internal terrain features
- **Problem:** Flat invisible walls made fluids settle into uniform pools
- **Fix:** f9a28d0 added STATIC terrain helpers (`spawn_ramp`, `spawn_bowl`,
  `spawn_wall_with_gap`, `spawn_cylinder_shell`) and the Canyon, Whirlpool and
  Avalanche Ramp presets. SDF presets add ledges, cylinders and boxes.

---

## F. Architectural Notes (not bugs, documented for context)

### F1. WCSPH force_scale=0.02 convention
- SPH forces are 50x weaker than standard formulation
- All parameters (k, mu, XSPH) are tuned to this convention
- Inherited from parent gpusphsim project (see ../PROBLEMS.md #1)
- Changing would require retuning all WCSPH parameters

### F2. WCSPH viscosity lacks 1/rho_i division
- WCSPH step2 doesn't divide viscosity by rho_i (unlike DFSPH)
- Uses force_scale=0.02 instead, a different convention tuned separately
- "Pa*s" unit label was misleading (fixed in d4727ba)

### F3. PBF Tait EOS used for mu(I) instead of lithostatic pressure
- Drucker-Prager I-number uses Tait pressure, not lithostatic (rho*g*h)
- Makes I artificially large, pushing mu(I) toward mu_2/mu_max everywhere
- Acceptable for game physics. Full lithostatic pressure would need column height estimation.

---

## G. Cross-Kernel Physics Issues (from GPT 5.2 Pro cross-review, 2026-02-28)

### G1. FIXED: Heat diffusion missing rho*cp divisor
- **File:** `step1.cu:87,294`, `integrate.cu:311`
- **Severity:** HIGH (units were wrong in the dimensional chain)
- **Problem:** Physical heat equation is `dT/dt = (k / (rho * cp)) * lap(T)`.
  Code used `dTdt = kappa * visc_lap_coeff * SUM...` where kappa is `thermal_conductivity`
  (W/(m*K)), giving dTdt in W/m^3 not K/s. The `heat_capacity` field was unused.
- **Impact:** `thermal_conductivity` values (0.02-2.0) happened to work as thermal diffusivity
  (m^2/s). But METAL at k=50 gave thermal CFL = 260 (wildly unstable with explicit Euler).
  Adding the proper `1/(rho*cp)` divisor gives METAL effective diffusivity = 4.4e-5,
  CFL = 0.00023 (stable). This also fixed B4.
- **Fix:** Divide by `density * cp_i` in the step1 heat term (PHYSICS.md 9.1).

### G2. FIXED: GRANULAR mu(I) missing 1/rho_i division (WCSPH only)
- **File:** `sph_shared.cuh:154-162`, `step2.cu:209-213`
- **Severity:** MEDIUM
- **Problem:** `compute_muI_eta()` returns Pa*s (dynamic viscosity) but WCSPH step2 divided by
  `rho_j` only (kinematic form), not `rho_i * rho_j` (dynamic form). Off by a factor of about rho_i
  (about 2500). Masked by the `mu_max=10000` clamp dominating the result.
- **Note:** DFSPH correctly applies `1/rho_i` at `dfsph_solver.cu:441`. This was WCSPH-specific.
- **Fix:** Add /rho_i in step2.

### G3. FIXED: Vorticity confinement gradient uses |omega_j| not (|omega_j| - |omega_i|)
- **File:** `step2.cu:277`
- **Severity:** LOW-MEDIUM
- **Problem:** SPH gradient of scalar A should use `(A_j - A_i) * grad_W`. Using `A_j` alone
  relies on `SUM (m_j/rho_j) * grad_W = 0` which fails near free surfaces. Produces spurious
  confinement force at surfaces. Mitigated by small epsilon=0.05.
- **Fix:** step2.cu (K_Step2 confinement block) uses `omega_diff = omega_j - omega_mag_i`.
- **Measured (2026-09-25, `eval_vorticity.py`):** gravity-free 0.4 m WATER cube, Taylor-Green
  field A = 0.3 m/s, WCSPH, 300 substeps. Net L_z drift is +1.5e-6 (eps 0), -5.1e-6 (eps 0.05)
  and -2.7e-6 (eps 0.10), with no trend in eps. KE/KE0 is 0.836, 0.841 and 0.846, so confinement
  feeds the vortex. Transverse RMS v_x under a pure shear is 7.3009e-3 (eps 0) and 7.3011e-3
  (eps 0.10). Confinement injects no measurable net angular momentum.
  `test_vorticity_confinement.py` guards the KE result.

### G4. CONFIRMED: Surface tension is a one-body push, not the Akinci model
- **File:** `step2.cu` (K_Step2 surface tension block), `dfsph_solver.cu` NonPressureForces
- **Severity:** MEDIUM
- **Bead:** bd-r5-review-n28.125 (open)
- **Problem:** Code applies `gamma * normal_i` to surface particles (neighbour count
  below 25), where the normal is the SPH gradient of the color field. There is no
  cohesion pair term, no curvature term and no reaction on the neighbours, so it
  does not conserve momentum and is resolution-dependent. DFSPH applies it without
  force_scale, so its push is 50x the WCSPH one. At gamma 1 a 2.5:1 zero-g WCSPH
  water block compresses by 12% and gets longer, not rounder.
- **Tried:** two pairwise Akinci attempts landed and were reverted (reverts 0c694e5
  and 4352c79). Any cohesion strong enough to round the block raised density by 20%
  or more against the soft force_scale=0.02 WCSPH pressure, and step2 at 1M
  particles rose from 3.5 to 5.9 ms. The bead comments list the design options.

---

## H. Cross-Solver Issues (from Gemini 3.1 Pro cross-review, 2026-02-28)

### H1. FIXED: DFSPH STATIC density not forced to rest_density
- **File:** `dfsph_solver.cu:246` (K_DFSPH_ComputeDensityAlpha)
- **Severity:** HIGH
- **Problem:** PBF correctly forces `density_out[i] = rest_density` for STATIC. DFSPH had no
  STATIC override and wrote raw Poly6 density (about 0.5*rho0 due to the truncated neighborhood). Since
  DFSPH uses `V_j = m_j/rho_j` everywhere, this doubled effective STATIC volume, causing
  excess boundary repulsion.
- **Fix:** After Poly6 density, override: `if (STATIC) density_out[i] = rest_density`

### H2. FIXED: PBF XSPH coefficient exceeds stability for viscous materials
- **File:** `pbf_solver.cu:753`
- **Severity:** HIGH (caused LAVA to explode in PBF)
- **Problem:** XSPH coefficient = `xsph_c * base_viscosity`. For LAVA (base_viscosity=100),
  c_xsph = 0.05 * 100 = 5.0. XSPH requires c in [0, 0.5] for stability. At c=5.0, velocity
  correction is 5x the difference, so particles reverse direction and oscillate explosively.
- **Fix:** Clamp: `c_xsph = fminf(xsph_c * base_viscosity, 0.5f)`

### H3. FIXED: PBF Drucker-Prager friction uses absolute velocity
- **File:** `pbf_solver.cu:668`
- **Severity:** MEDIUM (Galilean invariance violation)
- **Problem:** Friction decomposed absolute `vel_new` along the pressure normal. For free-falling
  sand, `v_dot_n` can be large from rigid-body motion, causing artificial internal friction.
  Expansion guard (`v_dot_n > 0`) partially mitigated vertical free-fall.
- **Fix:** Apply friction to velocity correction `(vel_new - v_predicted)` instead.

---

## I. Integration and Pipeline Issues (from Gemini 3.1 Pro cross-review, 2026-02-28)

### I1. FIXED: XSPH position update uses pre-integration velocity
- **File:** `integrate.cu` (K_Integrate advection)
- **Severity:** HIGH
- **Problem:** FLUID position used `veleval_xsph` from step2, built from the old velocity,
  so positions advanced with v_n while velocity advanced to v_{n+1}. That is forward
  Euler in position, not the symplectic Euler the header and PHYSICS.md 4 claimed.
  The velocity clamp, spawn damping, GAS drag and boundary impulses reached the FLUID
  position one substep late.
- **Fix (bd-r5-review-n28.44, 5cd6485 6f3c6ff c8d78c7):** `advect_vel = vel_new`, plus the
  XSPH correction `veleval - vel` for FLUID.

### I2. FIXED: Advection CFL missing, PBF particles can tunnel
- **File:** `simulation.py`
- **Severity:** HIGH for PBF
- **Problem:** With velocity_limit=10 and PBF dt=1/120, particles travel 0.083m = 2.08*h per
  substep, exceeding cell size. Can skip cells in spatial hash, missing neighbors.
  DFSPH at dt=1/300 gives 0.83*h (barely safe). WCSPH at DT_MAX=0.001 gives 0.25*h (safe).
- **Fix:** Either reduce PBF velocity_limit, reduce PBF dt, or add CFL check.

### I3. FIXED: Grid reuse measures drift since the last full sort
- **File:** `counting_sort.cu:K_ScatterReorder`, `integrate.cu:K_Integrate`,
  `pbf_solver.cu:K_PBF_Finalize`, `dfsph_solver.cu:K_DFSPH_Finalize`,
  `simulation.py:_update_sort_skip_decision`
- **Severity:** MEDIUM-HIGH (was)
- **Problem:** `max_displacement` held the largest single-substep move since the last sort, so
  the (0.25h)^2 WCSPH test needed about 10 m/s at dt = 1e-3 to trip. Only the 4-frame cap
  forced a sort. A 2 m/s block drifted about 3 cells on a stale grid, lost its neighbours and
  fell to the 1.0 kg/m^3 density floor before the forced sort.
- **Fix (bd-r5-review-n28.54):** `K_ScatterReorder` writes `sorted_anchor[i]` (the position at
  the full sort) and `K_GatherReorder` leaves it alone. The three finalize kernels measure
  `|x_new - anchor[i]|^2`. The host scales the latched drift, which is up to two frames old,
  to the end of the next frame before it allows a skip (PHYSICS.md 20.5).
  Measured with `bench_substep.py` (WCSPH, 992K dam break, 17 substeps per frame): skip rate
  83% to 55%, total 9.18 to 9.84 ms per substep. With reuse on, the final position checksum
  now matches the sort-every-substep run (328179.85 vs 328179.31; it was 326174.88).
- **Follow-up:** coherent flow still loses reuse. Computing the cell of particle i from
  `anchor[i]` in step1/step2 (Verlet style, one extra 16 B read per kernel) would let pure
  translation keep the grid. Rigid boundary particles feed the tracker since
  bd-r5-review-n28.65.

### I4. FIXED: Frame counter increments per substep, not per frame
- **File:** `simulation.py:984`
- **Severity:** MEDIUM
- **Problem:** `_frame_counter += 1` inside `_sim_step()` (per substep). WCSPH runs about 40
  substeps/frame, so foam compaction (`% 8`) fired every 0.2 frames and particle compaction
  (`% 60`) every 1.5 frames, much more often than intended.
- **Fix:** `_frame_counter` increments once per frame in `step_frame()`. The foam
  compaction gate was still checked inside the per-substep foam step, so every substep of
  every 8th frame compacted. Compaction now runs from `step_frame()` after the substep loop,
  once per frame at most, and foam generation runs once per frame with a dt-scaled
  emission probability (bd-r5-review-n28.122).

### I5. FIXED: Boundary clamp can create r=0 with STATIC particles
- **File:** `sph_shared.cuh:187-273`
- **Severity:** MEDIUM
- **Problem:** `sdf_box_boundary` clamped to exact `world_min/max`. If STATIC particles sit at
  those coordinates, distance=0 gives a division by zero in `grad_spiky`. In practice STATIC
  particles are placed at spacing=0.02 from the wall (safe), but fragile.
- **Fix:** Add small margin: `world_min + 1e-4f`

---

## J. Reaction System Issues (from GPT 5.2 Pro cross-review, 2026-02-28)

### J1. FIXED: Corrosion and wetness share one accumulator, water dies from acid
- **File:** `reactions.cu:222-236`
- **Severity:** CRITICAL
- **Problem:** `exposure_corrode` drives both sand wetting AND health damage. WATER was not
  excluded from the damage path. The `(WATER, ACID, 0.05, 5.0)` interaction made water
  particles near acid accumulate corrode exposure and drain health to MAT_DEAD.
- **Fix:** Only whitelisted materials take acid damage (`CORRODIBLE_MATS`: STONE, METAL,
  WOOD, GRAVEL, OIL, GUNPOWDER), and WATER/ACID is heat-only (rate 0).

### J2. FIXED: Drying temperature acceleration is mathematically inverted
- **File:** `reactions.cu:211-218`
- **Severity:** HIGH
- **Problem:** Drying threshold: `exp_corrode < THRESHOLD / (1 + 0.01 * max(temp-293, 0))`.
  As temperature rises, the right side decreases, so hotter material dried SLOWER (inverted).
- **Fix:** Change to multiplication: `exp_corrode < THRESHOLD * (1 + 0.01 * max(temp-293, 0))`

### J3. FIXED: WATER missing from REACTIVE_MATERIAL_IDS
- **File:** `materials.py:344-347`
- **Severity:** HIGH
- **Problem:** `REACTIVE_MATERIAL_IDS` gates reaction kernel dispatch. WATER was absent, so in
  a water-only scene, pre-heated water could never boil into steam.
- **Fix:** Add WATER to the set.

### J4. FIXED: No latent heat at phase transitions
- **File:** `reactions.cu` (K_Reactions phase transitions)
- **Severity:** HIGH
- **Problem:** Phase transitions copied temperature unchanged. ICE at 280K became WATER at
  280K, but cp doubles (2090 to 4186), so internal energy jumped 2x. Combustion hard-sets 1200K
  regardless.
- **Fix (bd-r5-review-n28.64):** ICE to WATER, STONE/SAND to LAVA and LAVA to STONE now set the
  temperature to the threshold (273K for ICE, `c_materials[mat].temp_melt` otherwise). The
  excess is treated as consumed latent heat. STEAM to WATER and combustion are unchanged.
- **Decision:** No per-particle latent-energy accumulator (4 B per particle). A real melt
  plateau is not worth the memory and bandwidth in a sandbox; the clamp gives a visible melt
  front at no cost.

### J5. CONFIRMED: Acid+mud interaction cannot corrode mud
- **File:** `materials.py` (`(MUD, ACID, 0.05, 3.0)`, commented "acid corrodes mud"),
  `reactions.cu` (`CORRODIBLE_MATS` whitelist)
- **Severity:** MEDIUM
- **Problem:** MUD is not in `CORRODIBLE_MATS`, so MUD never loses health to acid. The pair's
  reaction_rate only adds to MUD's `exposure_corrode`, which MUD reads as wetness: acid
  neighbours slow MUD drying back to WET_SAND. The pair is symmetric, so ACID next to MUD
  is consumed (acid consumption block) while the mud is untouched.
- **Fix options:** add MUD to the whitelist (acid then dissolves mud), or set the pair to
  rate 0 (heat only) and fix the comment. No bead yet.

### J6. FIXED: Reaction thresholds duplicated and disagree
- **File:** `reactions.cu:44` vs `materials.py:243`
- **Severity:** MEDIUM
- **Problem:** `LAVA_SOLIDIFY_TEMP = 900.0f` in kernel, but LAVA `temp_melt = 1000.0` in
  materials.py. Conceptually the same threshold, differs by 100K.
- **Fix:** Read from `c_materials[].temp_melt` instead of hardcoded #define.

### J7. REFUTED: Water boiling spawn flag, source IS killed
- **File:** `reactions.cu` (boil sets the spawn flag), `spawn.cu` K_SpawnGas
- **Severity:** N/A (not a bug)
- **Claim was:** Boiling water sets SPAWN flag but water particle persists, risking mass increase.
- **Actual:** `K_SpawnGas` explicitly kills the source particle:
  `packed_info[i] = SET_STATE_DIRTY(MAKE_PACKED(MAT_DEAD, STATIC)); health[i] = 0.0f; mass[i] = 0.0f;`
  STATIC keeps the dead slot out of the fluid neighbour loop until compaction. Mass is
  conserved across the boiling transition.

### J8. FIXED: Steam and smoke get a lifetime
- **File:** `spawn.cu` K_SpawnGas, `reactions.cu` (GAS lifetime decay, fire extinguish)
- **Severity:** LOW
- **Claim was:** STEAM/SMOKE created without setting lifetime, accumulate indefinitely.
- **Actual (STEAM):** K_SpawnGas sets `lifetime[slot] = SPAWN_LIFETIME` (12.0 s).
- **Actual (SMOKE):** FIRE whose lifetime runs out becomes SMOKE with `lifetime = 3.0 s` and
  temperature capped at 500 K. FIRE put out by water becomes SMOKE with
  `EXTINGUISHED_SMOKE_LIFETIME` (1.0 s) at 373 K (bd-r5-review-n28.73). Expired SMOKE and
  STEAM go to MAT_DEAD with zero mass.

### J9. REFUTED: Interaction matrix sparsity makes materials insulators
- **File:** `materials.py` (_INTERACTION_PAIRS, 36 pairs), `step1.cu` heat term
- **Claim was:** Unpaired materials default to zero heat_exchange, so most materials are
  perfect insulators to each other, and thresholds such as GRAVEL melting are unreachable.
- **Actual:** conduction runs for every pair through `k_ij`, the harmonic mean of the two
  conductivities (bd-r5-review-n28.67). `heat_exchange` only boosts it:
  `heat_boost = max(1, heat_exchange)`, so an unpaired pair conducts at boost 1.
  GRAVEL heats from LAVA (and has a LAVA+GRAVEL pair). What stays pair-only is
  `exposure_heat`, which drives WOOD, OIL and GUNPOWDER ignition.

### J10. DEFERRED: Behavior-class transitions in sorted arrays
- **File:** `reactions.cu` (K_Reactions)
- **Severity:** MEDIUM (risk)
- **Problem:** Reactions change behavior class in place (for example ICE/STATIC to WATER/FLUID)
  on sorted arrays between Step1 and Step2. Safe only if downstream kernels branch per
  particle (they do). This is an invariant to keep, not a bug.

---

## K. Round-5 Review Findings (2026-09-24)

Items from the round-5 review that belong on this list. All are fixed; the bead
holds the full analysis.

### K1. FIXED: WCSPH neighbour density read as 1.0
- **Bead:** bd-r5-review-n28.21 (d2b75f2), then bd-r5-review-n28.163
- **Problem:** K_Integrate wrote `position.w = 1.0` for every awake particle, so from
  substep 2 on step1 read `rho_j = 1.0` for every awake neighbour through `density_in`.
- **Fix:** K_Integrate carries the step1 density in `position.w`; step1 now reads `rho_j`
  from `position.w` and `density_in` is gone for WCSPH.

### K2. FIXED: Rigid boundary particles written at a moving offset
- **Bead:** bd-r5-review-n28.22 (d1b16ae e31593e 7bf0eca ebd12ce)
- **Problem:** K_UpdateBoundaryParticles used `offset = world._high_water`, which brush spawn
  and compaction move, so body data landed in slots owned by other particles.
- **Fix:** The boundary block sits inside `_high_water` at `rbm.boundary_offset`, and
  compaction tracks it.

### K3. FIXED: load_scene tripped the rigid injection assert
- **Bead:** bd-r5-review-n28.22 (same commits as K2)
- **Problem:** save_load never reset the RigidBodyManager, so re-injection after a scene load
  hit `assert not self._boundary_particles_injected` and the sim loop died.
- **Fix:** One flag, `rbm.boundary_injected`, with `reset()` and `mark_for_reinjection()`;
  load_scene marks for reinjection.

### K4. FIXED: Dead particles kept their mass
- **Beads:** bd-r5-review-n28.23 (039e4f3 d89669a 68f0535), bd-r5-review-n28.34
  (f56e8b9 5e43266 8c76478 c089001)
- **Problem:** Acid-consumed and expired-gas particles, and brush-killed particles, became
  MAT_DEAD with their mass intact and kept acting as boundary mass until compaction.
- **Fix:** Every death path writes `MAKE_PACKED(MAT_DEAD, STATIC)` with zero mass; boundary
  loops skip MAT_DEAD (PHYSICS.md 10.20).

### K5. FIXED: The Timing panel ran a drifted copy of the WCSPH substep
- **Bead:** bd-r5-review-n28.2 (0a91045 6015fb8)
- **Problem:** `_sim_step_timed` was a hand-copied pipeline that skipped the STATIC boundary,
  surface velocity smoothing, the blast wave and the SDF kinematics update.
- **Fix:** The timed path runs the shared substep body with per-stage marks.

### K6. FIXED: Rigid particles had position.w = 0
- **Bead:** bd-r5-review-n28.66 (05610b0); bd-r5-review-n28.102 was closed as its duplicate
- **Problem:** Rigid boundary particles kept `.w = 0`, and step2 still summed them into the
  vorticity-confinement direction.
- **Fix:** K_Step1, K_UpdateBoundaryParticles and the injection write `.w = 2500`
  (`RIGID_PSI_RHO0`), and the step2 eta sum uses FLUID neighbours only.
