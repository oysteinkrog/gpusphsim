# Known Problems in gpusphsim (Parent Project)

Documented during physics review comparing gpusphsim (parent SimpleSPH port)
with fallingsand3d (child multi-material extension).

---

## 1. Dimensional Inconsistency in SPH Force Pipeline

**Severity:** Architectural (tuned around, not a runtime bug)

The step2 kernel computes SPH acceleration (force/mass) in its inner loop,
then multiplies the output by `m_j` (particle mass) before writing:

```cuda
// step2.cu line ~453
sph_force_out = total_force * m_j;  // total_force is acceleration, output is mass*accel
```

The integrate kernel then treats this as pure acceleration:

```cuda
// integrate.cu
vel += (sph_force + gravity) * dt;  // sph_force has units of m*a, gravity has units of a
```

This means SPH forces are effectively scaled by `m_j` relative to gravity.
With `m_j = 0.02`, SPH forces contribute at `0.02x` their correct magnitude
relative to gravity, and all EOS/viscosity parameters (k=3.0, mu=3.5) were
tuned to compensate for this scaling.

**Impact:** The simulation produces visually correct results because parameters
are tuned to this convention. However, changing particle mass requires retuning
all SPH parameters, and the force magnitudes have no physically meaningful units.

---

## 2. Single-Material Assumption

**Severity:** Design limitation

The parent project uses a single global `particle_mass` constant
(`c_fluid.particle_mass = 0.02`) for all particles. The density kernel
pulls mass from constant memory rather than per-particle:

```cuda
// step2.cu
float m_j = c_fluid.particle_mass;  // same mass for every particle
```

This prevents multi-material simulations where different materials have
different densities or masses. The fallingsand3d child project addressed
this by reading per-particle mass from a sorted mass array and storing
per-material EOS parameters in a material property table.

---

## 3. No Per-Material EOS Parameters

**Severity:** Design limitation

Pressure computation uses global `gas_stiffness` and `rest_density` from
`FluidParams` constant memory. All particles share the same EOS, making it
impossible to simulate different fluids (e.g., water and oil) with different
compressibilities or rest densities in the same scene.

---

## 4. XSPH Epsilon Source Mismatch with Documentation

**Severity:** Minor / cosmetic

The XSPH blending factor `epsilon` is read from `c_fluid.xsph_epsilon`
(the FluidParams struct), but conceptually it belongs with the velocity
smoothing parameters. This is not a bug (the value is set correctly to 0.5),
but it creates a confusing API where the XSPH parameter lives in a struct
named for fluid EOS properties.

---

## 5. No Sleep/Wake System

**Severity:** Performance limitation

All particles are processed every frame regardless of whether they are
moving. Static or settled particles still go through the full neighbor
search, pressure computation, and force accumulation. A sleep/wake system
(as implemented in fallingsand3d) could skip settled particles entirely.

---

## 6. Fixed Grid Resolution

**Severity:** Performance limitation

The uniform grid cell size is fixed at compile time (tied to smoothing
length). For scenes with large empty regions, this wastes memory on empty
cells. The grid dimensions and bounds are hardcoded, limiting the effective
simulation domain.

---

## 7. No Boundary Friction Model

**Severity:** Physics limitation

The parent project uses simple position clamping at domain boundaries
without restitution or friction:

```cuda
// Boundary: clamp position, zero normal velocity
if (pos.x < world_min.x) { pos.x = world_min.x; vel.x = 0; }
```

This causes particles to "stick" to walls unnaturally. A proper impulse-based
boundary with restitution and Coulomb friction (as in fallingsand3d) produces
more realistic wall interactions.

---

## 8. Precalc Coefficient Sign Convention

**Severity:** Cosmetic / maintenance risk

The `pressure_precalc` constant absorbs the negative sign from the spiky
gradient kernel:

```
pressure_precalc = +45 / (pi * h^6)   // positive (double negative absorbed)
```

The actual spiky gradient is negative (`-45/(pi*h^6) * (h-r)^2 * r_hat`),
and the pressure formula has another negative
(`-m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W`). The double negative
cancels to positive, which is baked into `pressure_precalc`. While
mathematically correct, this makes the code harder to audit against
textbook SPH formulations where the signs are explicit.
