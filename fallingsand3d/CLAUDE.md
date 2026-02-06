# Claude Code Notes -- fallingsand3d

## Important: Keep physics documentation up to date

Whenever you modify any physics-related code (kernels, simulation parameters,
integration, force computation, EOS, boundary handling, etc.), you MUST update
`PHYSICS.md` to reflect the changes. This document is the single source of
truth for understanding the mathematical formulas and physical conventions
used in the simulation.

Specifically, update PHYSICS.md when changing:
- SPH kernel functions or coefficients (step1.cu, step2.cu)
- Tait EOS parameters or pressure computation
- Force accumulation formulas (pressure, viscosity, XSPH)
- Integration scheme (integrate.cu)
- Boundary handling
- Material properties (materials.py)
- Simulation constants (simulation.py, step2.py)
- The parameter comparison table (section 6)
- Known issues list (section 10)

Also keep `../PROBLEMS.md` (parent project issues) updated if you discover
new issues in the parent gpusphsim project.
