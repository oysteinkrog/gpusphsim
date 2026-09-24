# The Scan-Order Hall Fluid

*A falsifiable connection between falling-sand/SPH update artifacts and odd (Hall) viscosity, applied to `fallingsand3d`.*

## The idea in one sentence

The "cheating" every particle/CA fluid does for speed — updating in a fixed sweep
order, or using an antisymmetric force such as vorticity confinement — breaks
parity, and the **leading truncation error of a parity-broken fluid update is
mathematically identical to odd (Hall) viscosity**:

```
rho Dv/Dt = -grad p + nu lap(v) + eta_o lap(R v),   R = 90-degree rotation
```

Odd viscosity is the non-dissipative, parity-breaking transport coefficient
studied in quantum-Hall fluids, rotating colloids, and chiral active matter. In
the **incompressible bulk it hides inside the pressure** (invisible), and only
becomes observable at **free surfaces** and in the **weakly-compressible bulk** —
which is exactly what `fallingsand3d` is made of.

## Why this repo is the natural test bed

Three items already in `PROBLEMS.md` are, in this language, the same phenomenon:

| PROBLEMS.md | Called | Reinterpreted as |
|---|---|---|
| **G3** (`step2.cu:277`) — confinement uses `omega_j`, not `(omega_j - omega_i)` | "spurious surface force" | a deterministic **odd-viscosity source at the free surface** |
| **I1** (`integrate.cu:258`) — advection uses pre-integration velocity | "one-substep lag" | **broken time-reversal symmetry** (the T-breaking odd viscosity needs) |
| **G2/F2** — asymmetric `1/rho` viscosity conventions | "unit mismatch" | asymmetric momentum stencil → nonzero parity-odd part |

Plus the ULP-level seed in the "clean" path: gather kernels sum a fixed 27-cell
traversal with a non-associative float reduction (`common.cuh:119`), so `F_ij`
and `F_ji` don't cancel exactly — a tiny sign-definite `eta_o`. Meanwhile **PBF
is Jacobi** (`K_PBF_ComputeDelta` double-buffers), i.e. parity-symmetric — a
built-in null control.

## Falsifiable predictions

1. **k² fingerprint.** A shear (velocity varying in x) drives a *transverse*
   response `~ k^2` (a viscosity), not `~ k^0` (a Coriolis-like body force).
   This single measurement separates the theory from the mundane explanation.
2. **Chirality antisymmetry.** Reverse the sweep / flip `eta_o` → the transverse
   response flips sign with equal magnitude.
3. **Null control.** A genuinely parity-symmetric update (`eta_o = 0`, symmetric
   stencil, or PBF-Jacobi) shows *no* transverse response.
4. **Emergence.** A biased *stencil* alone (no explicit odd term) reproduces the
   same k² transverse fingerprint — the artifact **is** the term.
5. **Incompressible invisibility.** The induced response is dilatational
   (pressure channel); an incompressibility constraint erases it.

## Status: CPU rig passes all five

`odd_viscosity_probe.py` is a GPU-free finite-difference measurement rig. Result:

```
[PASS] T1 k^2 vs Coriolis discriminator     (exponent +1.976 vs -0.001; eta_o recovered 0.0505 from input 0.05)
[PASS] T2 chirality antisymmetry            (residual 0.0e+00)
[PASS] T3 null control (parity-symmetric)   (a = 0 exactly)
[PASS] T4 emergent stencil = odd viscosity  (exponent +1.977, sign flips with bias)
[PASS] T5 incompressible invisibility       (induced response 100% dilatational)
```

Run it: `python odd_viscosity_probe.py` (needs only numpy).

## Next: confirm on the real SPH sim

`test_odd_viscosity.py` runs the analogous probe inside the actual
weakly-compressible SPH solver (skips cleanly without a CUDA GPU). It seeds a
symmetric, zero-net-angular-momentum Taylor-Green field in a gravity-free water
block and checks whether the parity-broken confinement term spins up a
sign-definite net `L_z` that **scales with the confinement coefficient** and
vanishes when it is switched off. Divergence there (F1–F3 in the file) falsifies
the claim that `fallingsand3d`'s spurious rotation is odd-viscosity-driven.

Run on the CUDA box: `python test_odd_viscosity.py` or `pytest test_odd_viscosity.py`.

## Turning the bug into a feature

Odd viscosity is one cross-product away from the existing viscosity loop: apply
the velocity Laplacian to the 90°-rotated velocity. In 3D with up-axis `k̂`:

```
f_odd = eta_o * lap( k̂ × v )
```

i.e. in the `step2.cu` viscosity neighbor loop, replace `v_ij` with
`cross(k_hat, v_ij)` and scale by a per-material `eta_odd`. That buys a genuinely
new material class no falling-sand game has — self-spinning "spinner" fluids,
chiral potions, lava that curls its own boundary — with real, filmable
signatures (edge currents hugging the walls in one handedness, unidirectional
surface ripples, a droplet that spins itself up). The artifact everyone
symmetrizes away becomes a dial.
