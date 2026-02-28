# Known Problems & Improvement Opportunities — fallingsand3d

Compiled from multi-model review (Gemini 3.1 Pro, GPT 5.2 Pro, Claude) of
all physics kernels, material parameters, solver profiles, reaction system,
and scene presets. 8 independent AI reviews across 2 rounds.

Last updated 2026-02-28. Total: 48 items (35 fixed, 6 investigate, 5 deferred, 2 refuted).

---

## Status Key

- **FIXED** — Code change applied and committed
- **CONFIRMED** — Verified as real issue, fix planned
- **INVESTIGATE** — Needs testing to confirm impact
- **DEFERRED** — Known limitation, acceptable for now
- **REFUTED** — Claim was incorrect after code verification

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

### A7. Documented: PBF s_corr_n hardcoded to 4
- **File:** `pbf_solver.cu:465`
- **Status:** DEFERRED — s_corr is disabled (k=0) at h=0.04, no practical impact
- **Note:** `c_pbf.s_corr_n` parameter exists but kernel uses hardcoded n=4

### A8. Documented: Legacy DFSPH kernels (3 kernels)
- **Files:** `dfsph_solver.cu` — K_DFSPH_ComputeDensityAdv, K_DFSPH_ComputeKappaFromVelocity, K_DFSPH_CorrectVelocityDens
- **Status:** DEFERRED — Marked as LEGACY/UNUSED, not called from Python
- **Risk:** ComputeKappaFromVelocity uses m_j (not m_j/rho_j), inconsistent with active Jacobi solver

---

## B. Material Property Issues (from Gemini 3.1 Pro review, 2026-02-28)

### B1. FIXED: Density scaling inconsistent across materials
- **File:** `materials.py`
- **Severity:** CRITICAL — breaks buoyancy behavior
- **Problem:** Water uses rest_density=2500 (2.5x real-world for force_scale convention) but several materials use real-world densities, creating wrong buoyancy ratios.

| Material | Current | Should Be | Ratio vs Water (current) | Ratio vs Water (correct) |
|----------|---------|-----------|--------------------------|--------------------------|
| WATER    | 2500    | 2500      | 1.00                     | 1.00                     |
| OIL      | 2500    | 2000-2200 | 1.00 (won't float!)      | 0.80-0.88                |
| LAVA     | 2500    | 6500-7000 | 1.00 (won't sink!)       | 2.60-2.80                |
| SAND     | 2500    | 3500-4500 | 1.00                     | 1.40-1.80                |
| ICE      | 917     | 2300      | 0.37 (rockets out!)      | 0.92                     |
| WOOD     | 600     | 1250-1500 | 0.24 (rockets out!)      | 0.50-0.60                |

### B2. FIXED: LAVA EOS stiffness too low
- **File:** `materials.py`
- **Problem:** LAVA k=30 vs WATER k=500 — lava is 16x more compressible, collapses under own weight
- **Suggested:** k=1500-2000

### B3. FIXED: ICE/WOOD EOS stiffness too low
- **Problem:** k=15 for both — behaves like jello if they become dynamic particles
- **Suggested:** k=1000

### B4. FIXED: Metal thermal conductivity retuned
- **File:** `materials.py` — METAL thermal_conductivity=50
- **Problem:** May violate thermal CFL with explicit Euler integration
- **Suggested:** 5-10

### B5. INVESTIGATE: Fire heat capacity too high
- **File:** `materials.py` — FIRE heat_capacity=1000
- **Problem:** Fire retains heat too long, doesn't dump energy dramatically
- **Suggested:** 100

### B6. CONFIRMED: Viscosity units comment misleading
- **File:** `common.cuh:38`
- **Was:** `// Pa*s`
- **Fix:** Updated to describe actual per-solver convention
- **Commit:** d4727ba

---

## C. Solver Tuning Issues (from Gemini 3.1 Pro review, 2026-02-28)

### C1. FIXED: WCSPH XSPH epsilon too high
- **File:** `solver_profiles.py`
- **Problem:** xsph_epsilon=0.8 applies massive velocity smoothing, drowning material viscosity differences. Water looks like syrup, oil/lava indistinguishable.
- **Suggested:** 0.05-0.1

### C2. FIXED: Acceleration clamp too restrictive
- **File:** `simulation.py` (accel_max in SimParams)
- **Problem:** accel_max=30 m/s^2 (~3G) mutes collision splashes and explosive interactions
- **Suggested:** 150-300 m/s^2 (velocity_limit=10 already prevents escape)

### C3. INVESTIGATE: DFSPH timestep overly conservative
- **File:** `solver_profiles.py`
- **Problem:** dt=1/300 (5 substeps/frame) — DFSPH is implicit, designed for larger dt
- **Suggested:** dt=1/120-1/150 (2 substeps/frame), saving significant GPU budget
- **Risk:** May need more density iterations to compensate

### C4. INVESTIGATE: PBF iteration count insufficient for deep columns
- **File:** `solver_profiles.py`
- **Problem:** 6 iterations at dt=1/120 can't propagate corrections through 100-particle columns
- **Suggested:** 8-10 iterations for stable profile

---

## D. Reaction System Issues (from Gemini 3.1 Pro review, 2026-02-28)

### D1. FIXED: Steam/water phase transition flickering
- **File:** `reactions.cu:45-46`
- **Problem:** STEAM_CONDENSE_TEMP=373 and WATER_BOIL_TEMP=373 — zero hysteresis. Particles at boundary temperature flicker between FLUID/GAS, causing pressure spikes.
- **Fix:** Lower STEAM_CONDENSE_TEMP to 360 (13K hysteresis band)

### D2. FIXED: Missing melting reactions
- **File:** `reactions.cu`
- **Problem:** Stone/sand heat up from lava contact but never melt — breaks sandbox immersion
- **Fix:** Add STONE→LAVA at temp>1500K, SAND→LAVA at temp>1700K

### D3. FIXED: Fire vanishes instead of becoming smoke
- **File:** `reactions.cu:243`
- **Problem:** MAT_FIRE lifetime expiry → MAT_DEAD (invisible). Combustion should leave a trail.
- **Fix:** Fire → MAT_SMOKE with lifetime=2.0s

### D4. INVESTIGATE: Heat exchange values too high
- **File:** `materials.py` _INTERACTION_PAIRS
- **Problem:** heat_exchange up to 60 (LAVA+ICE). At dt~0.008s with multiple neighbors, can overshoot 100% per substep causing temperature oscillation.
- **Suggested caps:** LAVA+ICE: 30, WATER+LAVA: 25, LAVA+OIL: 25, LAVA+GUNPOWDER: 25

### D5. FIXED: Explosion speed too low
- **File:** `reactions.cu:57`
- **Problem:** EXPLOSION_SPEED=5.0 gets dampened by viscosity+pressure solver to a soft "puff"
- **Suggested:** 25-30 m/s for visually dramatic explosions

### D6. FIXED: Missing interactions
- **Missing:** DIRT+WATER wetting (like sand), LAVA+SAND heating toward melt
- **Fix:** Add `(DIRT, WATER, 0.5, 5.0)` and `(LAVA, SAND, 0.0, 25.0)` to interaction pairs

### D7. FIXED: Acid not consumed during reactions
- **Problem:** A single acid particle corrodes unlimited stone — no mass loss mechanism
- **Potential fix:** Deplete acid health proportional to damage dealt

---

## E. World & Preset Issues (from Gemini 3.1 Pro review, 2026-02-28)

### E1. FIXED: Gravity too high for "miniature world" feel
- **Problem:** g=9.81 in a 2m cube: objects fall top-to-bottom in 0.63s, feels frantic/small
- **Suggested:** g=3.0-4.0 — splashes linger, avalanches feel massive
- **Impact:** Single biggest change for "world in a box" aesthetic

### E2. CONFIRMED: Presets underutilize particle budget
- **Problem:** Most presets spawn ~130K of 500K capacity — looks sparse
- **Suggested:** Flagship presets should target 300K+ particles (leave 150K for reactions)

### E3. INVESTIGATE: No initial velocity support for spawned blocks
- **Problem:** All spawned material starts at rest — limits dramatic openings
- **Suggested:** Add initial_velocity parameter to spawn_cube

### E4. INVESTIGATE: No internal terrain features
- **Problem:** Flat invisible walls cause fluids to settle into boring uniform pools
- **Suggested:** Use STATIC materials (stone/metal) to build ramps, pedestals, channels

---

## F. Architectural Notes (not bugs, documented for context)

### F1. WCSPH force_scale=0.02 convention
- SPH forces are 50x weaker than standard formulation
- All parameters (k, mu, XSPH) are tuned to this convention
- Inherited from parent gpusphsim project (see ../PROBLEMS.md #1)
- Changing would require retuning all WCSPH parameters

### F2. WCSPH viscosity lacks 1/rho_i division
- WCSPH step2 doesn't divide viscosity by rho_i (unlike DFSPH)
- Uses force_scale=0.02 instead — different convention, tuned separately
- "Pa*s" unit label was misleading (fixed in d4727ba)

### F3. PBF Tait EOS used for mu(I) instead of lithostatic pressure
- Drucker-Prager I-number uses Tait pressure, not lithostatic (rho*g*h)
- Makes I artificially large, pushing mu(I) toward mu_2/mu_max everywhere
- Acceptable for game physics — full lithostatic would need column height estimation

---

## G. Cross-Kernel Physics Issues (from GPT 5.2 Pro cross-review, 2026-02-28)

### G1. FIXED: Heat diffusion missing rho*cp divisor
- **File:** `step1.cu:87,294`, `integrate.cu:311`
- **Severity:** HIGH — units are wrong in the dimensional chain
- **Problem:** Physical heat equation is `dT/dt = (k / (rho * cp)) * lap(T)`.
  Code uses `dTdt = kappa * visc_lap_coeff * SUM...` where kappa is `thermal_conductivity`
  (W/(m*K)), giving dTdt in W/m^3 not K/s. The `heat_capacity` field is completely unused.
- **Impact:** `thermal_conductivity` values (0.02-2.0) happen to work as thermal diffusivity
  (m^2/s). But METAL at k=50 gives thermal CFL = 260 (wildly unstable with explicit Euler).
  Adding the proper `1/(rho*cp)` divisor would give METAL effective diffusivity = 4.4e-5,
  CFL = 0.00023 (stable). This also fixes B4 automatically.
- **Fix:** Either (A) divide by `density * cp_i` in step1 postcalc, or (B) rename field
  to `thermal_diffusivity` and remove `heat_capacity`.

### G2. FIXED: GRANULAR mu(I) missing 1/rho_i division (WCSPH only)
- **File:** `sph_shared.cuh:154-162`, `step2.cu:209-213`
- **Severity:** MEDIUM
- **Problem:** `compute_muI_eta()` returns Pa*s (dynamic viscosity) but WCSPH step2 divides by
  `rho_j` only (kinematic form), not `rho_i * rho_j` (dynamic form). Off by factor ~rho_i
  (~2500). Masked by `mu_max=10000` clamp dominating the result.
- **Note:** DFSPH correctly applies `1/rho_i` at `dfsph_solver.cu:441`. This is WCSPH-specific.
- **Fix:** Either make mu(I) return kinematic viscosity (divide by rho), or add /rho_i in step2.

### G3. FIXED: Vorticity confinement gradient uses |omega_j| not (|omega_j| - |omega_i|)
- **File:** `step2.cu:277`
- **Severity:** LOW-MEDIUM
- **Problem:** SPH gradient of scalar A should use `(A_j - A_i) * grad_W`. Using `A_j` alone
  relies on `SUM (m_j/rho_j) * grad_W = 0` which fails near free surfaces. Produces spurious
  confinement force at surfaces. Mitigated by small epsilon=0.05.
- **Fix:** Replace `omega_j` with `(omega_j - omega_mag_i)` at step2.cu:277-279.

### G4. Documented: Surface tension is not Akinci model (naming)
- **File:** `step2.cu:298-315`
- **Severity:** LOW (documentation only)
- **Problem:** Code applies `-gamma * normal_i` where normal is SPH gradient of color field.
  This is a surface normal cohesion force, not the full Akinci model (which includes curvature
  weighting and density factors). Resolution-dependent.
- **Note:** PHYSICS.md already partially acknowledges this.

---

## H. Cross-Solver Issues (from Gemini 3.1 Pro cross-review, 2026-02-28)

### H1. FIXED: DFSPH STATIC density not forced to rest_density
- **File:** `dfsph_solver.cu:246` (K_DFSPH_ComputeDensityAlpha)
- **Severity:** HIGH
- **Problem:** PBF correctly forces `density_out[i] = rest_density` for STATIC. DFSPH has no
  STATIC override — writes raw Poly6 density (~0.5*rho0 due to truncated neighborhood). Since
  DFSPH uses `V_j = m_j/rho_j` everywhere, this doubles effective STATIC volume, causing
  excess boundary repulsion.
- **Fix:** After Poly6 density at line 246, override: `if (STATIC) density_out[i] = rest_density`

### H2. FIXED: PBF XSPH coefficient exceeds stability for viscous materials
- **File:** `pbf_solver.cu:753`
- **Severity:** HIGH — causes LAVA to explode in PBF
- **Problem:** XSPH coefficient = `xsph_c * base_viscosity`. For LAVA (base_viscosity=100),
  c_xsph = 0.05 * 100 = 5.0. XSPH requires c in [0, 0.5] for stability. At c=5.0, velocity
  correction is 5x the difference — particles reverse direction and oscillate explosively.
- **Fix:** Clamp: `c_xsph = fminf(xsph_c * base_viscosity, 0.5f)`

### H3. FIXED: PBF Drucker-Prager friction uses absolute velocity
- **File:** `pbf_solver.cu:668`
- **Severity:** MEDIUM (Galilean invariance violation)
- **Problem:** Friction decomposes absolute `vel_new` along pressure normal. For free-falling
  sand, `v_dot_n` can be large from rigid-body motion, causing artificial internal friction.
  Expansion guard (`v_dot_n > 0`) partially mitigates for vertical free-fall.
- **Fix:** Apply friction to velocity correction `(vel_new - v_predicted)` instead.

---

## I. Integration & Pipeline Issues (from Gemini 3.1 Pro cross-review, 2026-02-28)

### I1. INVESTIGATE: XSPH position update uses pre-integration velocity
- **File:** `integrate.cu:258-271`
- **Severity:** HIGH (if confirmed — needs code verification)
- **Problem:** FLUID position uses `veleval_xsph` from step2 (computed from old velocity).
  The newly integrated `vel_new` (with this substep's forces) is only written to the velocity
  buffer, not used for position advection. This creates a one-substep lag.
- **Note:** This may be intentional (Euler vs symplectic Euler convention). Needs verification
  against the actual code path and whether `veleval_xsph` already includes force contributions.

### I2. FIXED: Advection CFL missing — PBF particles can tunnel
- **File:** `simulation.py`
- **Severity:** HIGH for PBF
- **Problem:** With velocity_limit=10 and PBF dt=1/120, particles travel 0.083m = 2.08*h per
  substep, exceeding cell size. Can skip cells in spatial hash, missing neighbors.
  DFSPH at dt=1/300 gives 0.83*h (barely safe). WCSPH at DT_MAX=0.001 gives 0.25*h (safe).
- **Fix:** Either reduce PBF velocity_limit, reduce PBF dt, or add CFL check.

### I3. INVESTIGATE: Grid reuse tracks max single-step displacement, not cumulative drift
- **File:** `integrate.cu:346-356`, `simulation.py:1216-1238`
- **Severity:** MEDIUM-HIGH
- **Problem:** `atomicMax(max_displacement, disp_sq)` gives the largest single-substep movement
  since last sort. A particle drifting 0.009m/substep (below 0.01m threshold) for 40 substeps
  could drift 0.36m = 9*h without triggering a resort.
- **Mitigating factor:** Safety limit of 4 consecutive skipped frames caps total drift.
  At 0.009m/substep * ~20 substeps * 4 frames = 0.72m worst case.
- **Fix:** Store reference position at last sort, compute drift against reference.

### I4. FIXED: Frame counter increments per substep, not per frame
- **File:** `simulation.py:984`
- **Severity:** MEDIUM
- **Problem:** `_frame_counter += 1` inside `_sim_step()` (per substep). WCSPH runs ~40
  substeps/frame, so foam compaction (`% 8`) fires every 0.2 frames, particle compaction
  (`% 60`) fires every 1.5 frames — much more often than intended.
- **Fix:** Move increment to `step_frame()` or use separate substep counter.

### I5. FIXED: Boundary clamp can create r=0 with STATIC particles
- **File:** `sph_shared.cuh:187-273`
- **Severity:** MEDIUM
- **Problem:** `sdf_box_boundary` clamps to exact `world_min/max`. If STATIC particles sit at
  those coordinates, distance=0 → division by zero in `grad_spiky`. In practice, STATIC
  particles are placed at spacing=0.02 from wall (safe), but fragile.
- **Fix:** Add small margin: `world_min + 1e-4f`

---

## J. Reaction System Issues (from GPT 5.2 Pro cross-review, 2026-02-28)

### J1. FIXED: Corrosion and wetness share one accumulator — water dies from acid
- **File:** `reactions.cu:222-236`
- **Severity:** CRITICAL
- **Problem:** `exposure_corrode` drives both sand wetting AND health damage. WATER is not
  excluded from the damage path. The `(WATER, ACID, 0.05, 5.0)` interaction causes water
  particles near acid to accumulate corrode exposure, draining health → MAT_DEAD.
  Acid literally dissolves water into nothing.
- **Fix:** Either split into `exposure_wet` / `exposure_corrode`, or whitelist only corrodible
  materials (STONE, METAL, WOOD, DIRT, GRAVEL) in the damage block.

### J2. FIXED: Drying temperature acceleration is mathematically inverted
- **File:** `reactions.cu:211-218`
- **Severity:** HIGH
- **Problem:** Drying threshold: `exp_corrode < THRESHOLD / (1 + 0.01 * max(temp-293, 0))`.
  As temperature rises, RHS decreases → harder to satisfy → hotter = dries SLOWER (inverted).
- **Fix:** Change to multiplication: `exp_corrode < THRESHOLD * (1 + 0.01 * max(temp-293, 0))`

### J3. FIXED: WATER missing from REACTIVE_MATERIAL_IDS
- **File:** `materials.py:344-347`
- **Severity:** HIGH
- **Problem:** `REACTIVE_MATERIAL_IDS` gates reaction kernel dispatch. WATER is absent, so in
  a water-only scene, pre-heated water can never boil into steam.
- **Fix:** Add WATER to the set.

### J4. CONFIRMED: No latent heat at phase transitions
- **File:** `reactions.cu:123-171`
- **Severity:** HIGH
- **Problem:** Phase transitions copy temperature unchanged. ICE at 280K → WATER at 280K, but
  cp doubles (2090→4186), so internal energy jumps 2x. Combustion hard-sets 1200K regardless.
  No latent heat modeled.
- **Fix (minimal):** Clamp temperature to transition point on phase change (e.g., ICE→WATER
  clamps to 273K). Excess heat treated as consumed latent heat.

### J5. INVESTIGATE: Acid+mud interaction defined but cannot fire (dead code)
- **File:** `materials.py:413`, `reactions.cu:223`
- **Severity:** MEDIUM
- **Problem:** `(MUD, ACID, 0.05, 3.0)` interaction exists but MUD is excluded from the
  corrosion health-damage path. The reaction_rate feeds the wetting state machine, not damage.

### J6. FIXED: Reaction thresholds duplicated and disagree
- **File:** `reactions.cu:44` vs `materials.py:243`
- **Severity:** MEDIUM
- **Problem:** `LAVA_SOLIDIFY_TEMP = 900.0f` in kernel, but LAVA `temp_melt = 1000.0` in
  materials.py. Conceptually the same threshold, differs by 100K.
- **Fix:** Read from `c_materials[].temp_melt` instead of hardcoded #define.

### J7. REFUTED: Water boiling spawn flag — source IS killed
- **File:** `reactions.cu:139-144`, `spawn.cu:168-170`
- **Severity:** N/A — not a bug
- **Claim was:** Boiling water sets SPAWN flag but water particle persists, risking mass increase.
- **Actual:** `K_SpawnGas` (`spawn.cu:168-170`) explicitly kills the source particle:
  `packed_info[i] = MAKE_PACKED(MAT_DEAD, FLUID); health[i] = 0.0f; mass[i] = 0.0f;`
  Mass is conserved across the boiling transition.

### J8. PARTIALLY REFUTED: Steam DOES get lifetime; smoke status unclear
- **File:** `spawn.cu:148`, `reactions.cu:238-254`
- **Severity:** LOW (reduced from MEDIUM)
- **Claim was:** STEAM/SMOKE created without setting lifetime, accumulate indefinitely.
- **Actual (STEAM):** `spawn.cu:148` sets `lifetime[slot] = SPAWN_LIFETIME` (5.0s). STEAM
  correctly decays via the GAS lifetime path. REFUTED for STEAM.
- **Actual (SMOKE):** Combustion transitions (WOOD/OIL/GUNPOWDER → FIRE) set lifetime for FIRE,
  but the FIRE→SMOKE transition path needs verification. If SMOKE inherits FIRE's remaining
  lifetime, it will also decay. Needs code tracing.
- **Note:** GAS decay condition `lt > 0.0f` (reactions.cu:240) is confirmed correct.

### J9. INVESTIGATE: Interaction matrix only ~20% populated
- **File:** `materials.py:372-414`
- **Severity:** MEDIUM
- **Problem:** 32/153 unique pairs have explicit interactions. All others default to zero
  heat_exchange — most materials are perfect insulators to each other. Some temperature
  thresholds are unreachable (e.g., GRAVEL can't be heated by anything except self-conduction).
- **Suggested:** Derive baseline heat exchange from `thermal_conductivity` for unpaired materials.

### J10. Documented: Behavior-class transitions in sorted arrays
- **File:** `reactions.cu:125-137, 158-171`
- **Severity:** MEDIUM (risk)
- **Problem:** Reactions change behavior class in-place (ICE/STATIC→WATER/FLUID) on sorted
  arrays between Step1 and Step2. Safe only if downstream kernels branch per-particle (they do).
  Should be documented as an invariant.
