"""bd-r5-review-n28.164: the STATIC push is folded into the Finalize XSPH loops.

K_PBF_Finalize and K_DFSPH_Finalize used to run static_particle_boundary, a
second 27-cell scan, after the XSPH loop. PBF FLUID and GRANULAR and DFSPH
GRANULAR now accumulate the push inside the XSPH loop. DFSPH FLUID uses the
loop only to prove there is no push, and runs the separate scan near a wall.
GAS and fast DFSPH particles still take the separate scan.

Each test runs water, sand and smoke falling onto a STONE slab. Every
Finalize call is replayed on the same inputs with two variants of the
kernel, built from the current source with small text changes:

  two-scan  the fold switched off, so every particle runs
            static_particle_boundary (the behaviour before this bead)
  no-push   the STATIC push removed entirely

The folded kernel must match two-scan to float rounding (1e-6 m, 1e-4 m/s)
and must differ from no-push, which shows the push acts in the scene.
"""

import ctypes
import inspect
import os
import sys

import cupy as cp
import numpy as np
import pytest

import dfsph_solver
import pbf_solver
from materials import SAND, SMOKE, STONE, WATER, GRANULAR, FLUID, GAS
from simulation import Simulation
from solver_profiles import PROFILES
from world import World

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physics", "kernels")
STEPS = 40
CONST_SYMBOLS = ("c_grid", "c_sim", "c_precalc", "c_materials", "c_interactions",
                 "c_granular", "c_num_rigid_bodies", "c_sdf_objects",
                 "c_num_sdf_objects", "c_pbf", "c_dfsph")

_cuda = None

PBF_FOLD_ON = "spush_in_loop = true;"
DFSPH_FOLD_ON = "spush_in_loop = (step_sq < step_max * step_max);"
NO_PUSH = ("#define static_particle_boundary(...) ((void)0)\n"
           "#define static_push_apply(...) ((void)0)\n")


def _variant(filename, fold_on, no_push):
    with open(os.path.join(KERNEL_DIR, filename), encoding="utf-8") as f:
        src = f.read()
    assert src.count(fold_on) == 1, f"{filename}: fold switch not found"
    if no_push:
        last = src.rindex("#include")
        eol = src.index("\n", last) + 1
        src = src[:eol] + NO_PUSH + src[eol:]
    else:
        src = src.replace(fold_on, "spush_in_loop = false;")
    return cp.RawModule(code=src, options=("--std=c++11", "--use_fast_math",
                                            f"-I{KERNEL_DIR}"))


def _symbol_size(mod, name):
    """Byte size of a __constant__ symbol (CuPy's get_global reports 0)."""
    global _cuda
    if _cuda is None:
        lib = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32"
               else ctypes.CDLL("libcuda.so.1"))
        _cuda = lib.cuModuleGetGlobal_v2
        _cuda.argtypes = [ctypes.POINTER(ctypes.c_uint64),
                          ctypes.POINTER(ctypes.c_size_t),
                          ctypes.c_void_p, ctypes.c_char_p]
    ptr, size = ctypes.c_uint64(), ctypes.c_size_t()
    rc = _cuda(ctypes.byref(ptr), ctypes.byref(size),
               ctypes.c_void_p(mod.module.ptr), name.encode())
    return size.value if rc == 0 else 0


def _copy_constants(src_mod, dst_mod):
    for name in CONST_SYMBOLS:
        size = _symbol_size(src_mod, name)
        if size == 0:
            continue
        s = src_mod.get_global(name)
        d = dst_mod.get_global(name)
        assert _symbol_size(dst_mod, name) == size
        cp.cuda.runtime.memcpy(d.ptr, s.ptr, size,
                               cp.cuda.runtime.memcpyDeviceToDevice)


def _run(solver, monkeypatch):
    """Run the scene; return per-call lists of (new, two-scan, no-push, behaviour)."""
    world = World(max_particles=16384)
    sim = Simulation(world, dt=0.001, speed=1.0, accuracy=0.4,
                     fixed_dt=False, max_substeps=20)
    sim.set_solver_profile(PROFILES[solver])
    monkeypatch.setattr(
        sim, "_capture_graph",
        lambda n, gather_only=False: (sim._run_graph_body(n), None)[1])

    base = -sim.world_half_size
    top = base + 0.08
    n = world.spawn_cube((-0.3, base, -0.2), (0.3, top, 0.2), STONE, spacing=0.02)
    n += world.spawn_cube((-0.26, top + 0.005, -0.1), (-0.02, top + 0.2, 0.1),
                          WATER, spacing=0.02, velocity=(0.0, -1.5, 0.0))
    n += world.spawn_cube((0.02, top + 0.005, -0.1), (0.26, top + 0.14, 0.1),
                          SAND, spacing=0.02, velocity=(0.0, -1.5, 0.0))
    n += world.spawn_cube((-0.1, top + 0.005, 0.12), (0.1, top + 0.05, 0.18),
                          SMOKE, spacing=0.02, velocity=(0.0, -1.0, 0.0), lifetime=0)
    sim.notify_scene_changed()
    sim._spawn_substep = sim._damping_duration

    if solver == "PBF":
        mod_py, fname, fold_on, fn_name = pbf_solver, "pbf_solver.cu", PBF_FOLD_ON, "pbf_finalize"
        out_names = ("position_out", "velocity_out", "packed_info_out",
                     "sleep_counter_out", "temperature_out",
                     "health_out", "lifetime_out", "mass_out")
    else:
        mod_py, fname, fold_on, fn_name = dfsph_solver, "dfsph_solver.cu", DFSPH_FOLD_ON, "finalize"
        out_names = ("position_out", "velocity_out", "packed_info_out",
                     "sleep_counter_out", "temperature_out", "kappa_out",
                     "kappa_v_out", "health_out", "lifetime_out", "mass_out")

    real_mod = mod_py._get_module()
    two_scan = _variant(fname, fold_on, no_push=False)
    no_push = _variant(fname, fold_on, no_push=True)
    orig = getattr(mod_py, fn_name)
    sig = inspect.signature(orig)
    records = []

    def replay(module, bound):
        args = dict(bound.arguments)
        for k in out_names:
            if args.get(k) is not None:
                args[k] = args[k].copy()
        args["max_displacement"] = None
        _copy_constants(real_mod, module)
        mod_py._module = module
        try:
            orig(**args)
        finally:
            mod_py._module = real_mod
        return args["position_out"], args["velocity_out"]

    def spy(*a, **k):
        orig(*a, **k)
        bound = sig.bind(*a, **k)
        bound.apply_defaults()
        new = (bound.arguments["position_out"].copy(), bound.arguments["velocity_out"].copy())
        ref = replay(two_scan, bound)
        nop = replay(no_push, bound)
        # Behaviour per unsorted slot (packed_info_out is unsorted).
        beh = (bound.arguments["packed_info_out"] >> 8) & 0x3
        records.append((new, ref, nop, beh))

    monkeypatch.setattr(mod_py, fn_name, spy)
    for _ in range(STEPS):
        sim._sim_step(n)
    cp.cuda.Device().synchronize()
    monkeypatch.undo()

    out = []
    for new, ref, nop, beh in records:
        out.append(tuple(cp.asnumpy(x[:n]) for x in (*new, *ref, *nop, beh)))
    return out


def _diffs(records, behaviour):
    """Max |new - two_scan| (pos, vel), max |new - no_push| (pos), and counts.

    pushed: particle-substeps the two-scan push moved.
    vel_mismatch: particle-substeps whose velocity differs from two-scan by
    more than 1 mm/s.
    """
    d_ref_p, d_ref_v, d_nop_p = 0.0, 0.0, 0.0
    pushed, vel_mismatch = 0, 0
    for p, v, rp, rv, np_, nv, beh in records:
        m = beh == behaviour
        if not m.any():
            continue
        d_ref_p = max(d_ref_p, float(np.abs(p[m, :3] - rp[m, :3]).max()))
        d_ref_v = max(d_ref_v, float(np.abs(v[m, :3] - rv[m, :3]).max()))
        d_nop_p = max(d_nop_p, float(np.abs(p[m, :3] - np_[m, :3]).max()))
        pushed += int((np.abs(rp[m, :3] - np_[m, :3]).max(axis=1) > 1e-7).sum())
        vel_mismatch += int((np.abs(v[m, :3] - rv[m, :3]).max(axis=1) > 1e-3).sum())
    return d_ref_p, d_ref_v, d_nop_p, pushed, vel_mismatch


@pytest.fixture(scope="module")
def pbf_records():
    mp = pytest.MonkeyPatch()
    try:
        return _run("PBF", mp)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def dfsph_records():
    mp = pytest.MonkeyPatch()
    try:
        return _run("DFSPH", mp)
    finally:
        mp.undo()


@pytest.mark.parametrize("behaviour,name", [(FLUID, "FLUID"), (GRANULAR, "GRANULAR"),
                                            (GAS, "GAS")])
def test_pbf_fold_matches_two_scan(pbf_records, behaviour, name):
    assert len(pbf_records) == STEPS
    dp, dv, dnop, pushed, vmis = _diffs(pbf_records, behaviour)
    print(f"  PBF {name}: |new-two_scan| pos={dp:.3e} vel={dv:.3e}, "
          f"|new-no_push| pos={dnop:.3e}, pushed={pushed}, vel_mismatch={vmis}")
    assert dp <= 1e-6, f"PBF {name} position differs from two-scan by {dp:.3e} m"
    assert dv <= 1e-4, f"PBF {name} velocity differs from two-scan by {dv:.3e} m/s"
    assert dnop > 1e-5, f"PBF {name}: STATIC push never acted in the scene"


@pytest.mark.parametrize("behaviour,name", [(GRANULAR, "GRANULAR"), (FLUID, "FLUID"),
                                            (GAS, "GAS")])
def test_dfsph_fold_matches_two_scan(dfsph_records, behaviour, name):
    """GRANULAR has no XSPH, so pos + dt*vel is pos_new and the loop push is exact.

    FLUID near a wall and GAS take the separate scan, so they match too.
    """
    assert len(dfsph_records) == STEPS
    dp, dv, dnop, pushed, vmis = _diffs(dfsph_records, behaviour)
    print(f"  DFSPH {name}: |new-two_scan| pos={dp:.3e} vel={dv:.3e}, "
          f"|new-no_push| pos={dnop:.3e}, pushed={pushed}, vel_mismatch={vmis}")
    assert dp <= 1e-6, f"DFSPH {name} position differs from two-scan by {dp:.3e} m"
    assert dv <= 1e-4, f"DFSPH {name} velocity differs from two-scan by {dv:.3e} m/s"
    assert dnop > 1e-5, f"DFSPH {name}: STATIC push never acted in the scene"
