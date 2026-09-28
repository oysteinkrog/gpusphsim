# Future Roadmap: Falling Sand 3D

Open future work for the simulation. The original roadmap (2026-02-08) came from
frontier model oracles (GPT-5.2 Pro, Gemini 3 Pro), research agents and architecture
analysis. Most of it has shipped; the table below says where each item went, and the
sections after it describe only the items still open.

Last checked against the code 2026-09-28 (bd-r5-review-n28.207).

---

## Status of the original roadmap items

"9a166d9" is the 2026-02-08 commit that added this file; several items shipped in
that same commit.

| Item | Status | Where it landed | Documented in |
|------|--------|-----------------|---------------|
| 1.1 Screen-space fluid rendering | Done | 9a166d9, improved in d4727ba | `renderer.py`, `shaders/ssfr_*` |
| 1.2 Counting sort | Done | 9a166d9 (`counting_sort.py`, `counting_sort.cu`) | PHYSICS.md 10.8 |
| 1.3 `__launch_bounds__` min blocks | Done | 9a166d9: K_Step1 `(256, 4)`; K_Step2 now `(256, 3)` | PHYSICS.md 15 |
| 2.1 Vorticity confinement | Done | 9a166d9; gradient fix G3 | PHYSICS.md 16 |
| 2.2 Akinci surface tension | Open (one-body form only) | see "Pairwise Akinci surface tension" below | PHYSICS.md 17 |
| 2.3 Beer-Lambert and Fresnel | Done | part of the SSFR composite pass | `renderer.py`, `shaders/ssfr_composite.frag` |
| 3.1 Ferrofluid mode | Open | see below | |
| 3.2 Foam, spray and bubbles | Done | 9a166d9 (`foam.py`, `foam.cu`) | PHYSICS.md 19 |
| 3.3 Color mixing and diffusion | Dropped | shipped as particle dye in 9a166d9, never enabled or rendered, removed by bd-r5-review-n28.144 | PHYSICS.md 10.23 |
| 4.1 Scalars packed into float4.w | Done | 9a166d9, extended by bd-r5-review-n28.21, n28.161, n28.163, n28.184, n28.185 | PHYSICS.md 20.1, 20.3b |
| 4.2 Speculative ILP loads | Done | 9a166d9, extended by bd-r5-review-n28.171 | PHYSICS.md 20.2 |
| 4.3 FP16 auxiliary attributes | Done | d09578b (velocity), bd-r5-review-n28.132 (temperature mirror) | PHYSICS.md 2, 9.1 |
| 4.4 Shared-memory X-pencil tiling | Open | see below | |
| 4.5 Counting sort details | Done | same as 1.2 | PHYSICS.md 10.8 |
| 5.1 XPBI granular-fluid coupling | Open | see below | |
| 5.2 Micropolar SPH | Dropped | shipped in 9a166d9 with no reader in step1/step2, removed by bd-r5-review-n28.140 | PHYSICS.md 10.13 |
| 5.3 Implicit surface tension (Jeske and Bender) | Open | `implicit_st.cu` is surface velocity smoothing, not this model (bd-r5-review-n28.133) | PHYSICS.md 12.8 |
| 5.4 Thermal convection (Boussinesq) | Done | 9a166d9 | PHYSICS.md 14 |
| 6.1 Sparse/paged grid | Done | 9a166d9 (spatial hash, 2^18 table) | PHYSICS.md 10.12 |
| 6.2 Adaptive substep grid reuse | Done | 9a166d9; drift anchor bd-r5-review-n28.54 | PHYSICS.md 20.5 |
| 6.3 Morton code sorting | Dropped | 14% slower at 524K in benchmarks; counting sort kept | |

---

## Pairwise Akinci surface tension (roadmap 2.2, PROBLEMS.md G4)

**Bead:** bd-r5-review-n28.125 (open).

**Current state.** WCSPH step2 and DFSPH NonPressureForces push each surface particle
(neighbour count below 25) along its color-field normal, `gamma * n_i`. There is no
pair term, so the force has no reaction on the neighbours and does not conserve
momentum. DFSPH applies it without force_scale, so its push is 50x the WCSPH one.

**Target model (Akinci et al. 2013):**

```
1. Per-particle normal (during density pass):
   n_i = h * SUM_j m_j/rho_j * grad_W_ij

2. Cohesion force (attractive):
   f_cohesion = -gamma * m_i * SUM_j m_j * C(|r_ij|) * r_ij/|r_ij|

   where C(r) is a piecewise polynomial:
     C(r) = (32/(pi*h^9)) * (h-r)^3 * r^3              if 0 < r <= h/2
     C(r) = (32/(pi*h^9)) * 2*(h-r)^3 * r^3 - h^6/64   if h/2 < r <= h

3. Curvature force (smooths surface):
   f_curvature = -gamma * m_i * (n_i - n_j)

4. Total surface tension:
   f_st = f_cohesion + f_curvature
```

**What blocked it.** Two attempts landed and were reverted (0c694e5, 4352c79). On a
2.5:1 zero-g water block, any cohesion strong enough to round the block raised density
by 20% or more, because WCSPH pressure with force_scale 0.02 is too soft to resist it.
Curvature-only either clumped the fluid or expanded it, depending on sign. Step2 at 1M
particles rose from 3.5 to 5.9 ms, because K_Step2 is capped at 80 registers and the
`n_j` read spilled. A zero-g DFSPH block disperses even at gamma 0, so a DFSPH rounding
test proves nothing until that is fixed.

**Design options (from the bead):** (a) stiffer WCSPH pressure for FLUID before adding
cohesion; (b) an implicit or stabilised curvature term; (c) keep the one-body form, add
a momentum-conserving reaction, and give DFSPH force_scale. For cost, `n_j` needs a
packed carrier, for example `h n` folded into the K_Step1 `pr2_omega` record with a
surface flag so interior-interior pairs skip the load.

---

## Ferrofluid mode (roadmap 3.1)

A magnetic fluid that forms Rosensweig instability spikes in a magnetic field. The mouse
position controls a magnetic dipole, and the fluid grows spikes toward the cursor.

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
- Add `magnetic_susceptibility` to material properties.
- Add a `K_MagneticForce` kernel: per particle, compute grad(|H|^2) from the
  mouse-driven dipole position and add it to the velocity.
- No neighbour loop (external field only, no particle-particle magnetic interaction),
  so the cost is one light per-particle pass.
- Run after forces, before integration, only for particles with
  `magnetic_susceptibility > 0`.

**Visual:** black fluid with metallic specular highlights. The spikes create depth
discontinuities that the SSFR narrow-range filter preserves.

**Complexity:** about 100 lines of CUDA plus a material property.

---

## Shared-memory X-pencil tiling (roadmap 4.4)

Estimated 20-40% on the neighbour kernels, at high implementation cost. Load one
"pencil" (row of cells along X) into shared memory, so all particles in those cells read
X-direction neighbours from shared memory instead of global memory.

```cuda
// Conceptual: cooperative loading
__shared__ float4 s_pos[MAX_PENCIL_PARTICLES];
__shared__ float  s_mass[MAX_PENCIL_PARTICLES];

// Phase 1: load pencil into shared memory
for i in my_share_of_pencil: s_pos[i] = position[global_i];
__syncthreads();

// Phase 2: use shared memory for X-direction neighbours
for j = 0 to pencil_count:
    float3 diff = pos_i - make_float3(s_pos[j]);
    ...
```

Shared memory runs at about 19 TB/s against about 670 GB/s for global memory on the
RTX 5070 Ti. The 9 cells in the X row (3x3x1 slice) become shared reads; the other 18
cells still go to global memory.

**Complexity:** very high. It needs a restructured neighbour loop and grid traversal,
and the thread block geometry must match the cell pencils. The spatial hash (not a dense
grid) makes pencil lookup harder: consecutive X cells are not adjacent in the hash
table. The 1M-particle substep already runs at about 82% of theoretical bandwidth, so
the real gain is likely below the estimate.

---

## XPBI granular-fluid coupling (roadmap 5.1)

The current granular model is mu(I) rheology in WCSPH and DFSPH, and a velocity-space
Drucker-Prager friction clamp in K_PBF_Finalize. XPBI (Extended Position Based
Interactions) would instead add a Drucker-Prager yield constraint for GRANULAR
particles inside the PBF constraint loop:

```
f_yield = |sigma_dev| - alpha_DP * tr(sigma) - c_DP <= 0

where:
  sigma_dev = deviatoric stress tensor
  alpha_DP  = sin(friction_angle) / sqrt(3)
  c_DP      = cohesion * cos(friction_angle)
```

Particles that violate the yield surface get position corrections that enforce granular
friction, replacing the tangential clamp.

**Complexity:** high. It needs a per-particle stress tensor and a new constraint solver.

---

## Implicit surface tension (roadmap 5.3)

Jeske and Bender (SIGGRAPH 2024) formulate surface tension implicitly, which handles
droplet breakup and merging correctly. It needs 5-20 linear-solver iterations per
substep, estimated at 2-5 ms at 500K particles, so it only fits as a quality mode for
small scenes (under 100K particles) or offline rendering.

`implicit_st.cu` is not this model. It is XSPH-style velocity smoothing on surface
particles with no cohesion or curvature term (PHYSICS.md 12.8), renamed "surface
velocity smoothing" by bd-r5-review-n28.133.

Do this after the pairwise Akinci item, which fixes the explicit baseline first.
