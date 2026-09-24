"""Shared loader for the CuPy RawModules built from physics/kernels/*.cu.

Every wrapper module (step1.py, step2.py, ...) used to carry its own copy of
the NVRTC arch check and the RawModule construction. This module owns both,
so the compile options and the sm_120 PTX workaround live in one place.

Compile options for every module are BASE_OPTIONS plus ``-I<kernel_dir>``,
then any ``extra_options`` the caller passes. A module that needs a different
define or flag must pass it through ``extra_options``.
"""

from __future__ import annotations

import os

import cupy

KERNEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "physics", "kernels"
)

# Shared by all kernels. --use_fast_math affects numerics, so it must stay the
# same across modules that share common.cuh.
BASE_OPTIONS: tuple[str, ...] = ("--std=c++11", "--use_fast_math")

_cache: dict[tuple[str, tuple[str, ...]], cupy.RawModule] = {}
_ptx_checked = False


def ensure_ptx_if_needed() -> None:
    """Force PTX compilation when the GPU arch is newer than NVRTC's max target.

    CuPy has no public switch for this, so it sets private ``_compiler``
    attributes and clears the arch caches. Runs its check once per process.
    """
    global _ptx_checked
    if _ptx_checked:
        return

    from cupy.cuda import compiler as _compiler
    from cupy.cuda import device as _device

    gpu_cc = _device.Device().compute_capability
    nvrtc_max = _compiler._get_max_compute_capability()
    if int(gpu_cc) > int(nvrtc_max):
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
            _compiler._get_arch_for_options_for_nvrtc._cache = {}
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}
    _ptx_checked = True


def compile_options(extra_options: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return the full NVRTC option tuple used for a kernel file."""
    return BASE_OPTIONS + (f"-I{KERNEL_DIR}",) + tuple(extra_options)


def load_module(cu_name: str, extra_options: tuple[str, ...] = ()) -> cupy.RawModule:
    """Compile ``physics/kernels/<cu_name>`` once and return the cached RawModule.

    The cache key is the file name plus ``extra_options``, so the same file
    built with different defines gives separate modules (each with its own
    constant memory).

    Raises FileNotFoundError naming the full path when the .cu file is missing.
    """
    extra = tuple(extra_options)
    key = (cu_name, extra)
    module = _cache.get(key)
    if module is not None:
        return module

    cu_path = os.path.join(KERNEL_DIR, cu_name)
    if not os.path.isfile(cu_path):
        raise FileNotFoundError(f"kernel_loader: CUDA source not found: {cu_path}")

    ensure_ptx_if_needed()

    with open(cu_path) as f:
        source = f.read()

    module = cupy.RawModule(code=source, options=compile_options(extra))
    _cache[key] = module
    return module
