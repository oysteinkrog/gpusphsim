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

## How to run

```
/c/Windows/System32/cmd.exe /c "cd /d C:\WORK\gpusphsim\fallingsand3d && python main.py"
```

This is a WSL-like environment — use `cmd.exe` to launch via Windows Python,
which has access to the CUDA GPU and native OpenGL/GLFW.

Requires: Python 3.13+, CUDA GPU, and these packages:
- cupy (CUDA array library)
- glfw (windowing)
- PyOpenGL (rendering)
- imgui[glfw] (UI)

Presets (Sand Castle, Volcano, Dam Break, Acid Rain, Water Drop) are loaded
via the ImGui UI at runtime. There are no command-line arguments.
