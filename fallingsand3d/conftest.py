"""Pytest configuration: ensure CuPy's NVRTC can locate its builtins DLL.

CuPy 13.x bundles ``nvrtc64_120_0.dll`` but loads ``nvrtc-builtins64_<ver>.dll``
lazily on the *second* on-the-fly compile. If the matching CUDA toolkit ``bin``
directory is not on the DLL search path, that load fails with::

    nvrtc: error: failed to open nvrtc-builtins64_128.dll

The interactive app works because Windows happens to resolve the DLL from the
process PATH, but pytest subprocesses do not always inherit a PATH that points
at the right toolkit version. We add every installed CUDA ``bin`` directory that
contains an ``nvrtc-builtins`` DLL to the DLL search path at import time.
"""
import glob
import os


def _add_cuda_nvrtc_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    roots = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        os.environ.get("CUDA_PATH", ""),
    ]
    dirs = []
    seen = set()
    for root in roots:
        if not root:
            continue
        for binary in glob.glob(os.path.join(root, "**", "nvrtc-builtins*.dll"), recursive=True):
            d = os.path.dirname(binary)
            if d in seen:
                continue
            seen.add(d)
            dirs.append(d)
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    # cupy's NVRTC is loaded via plain LoadLibrary, which consults the process
    # PATH at load time (add_dll_directory is not honored by that loader), so
    # prepend the toolkit bin dirs to PATH as well.
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_nvrtc_dirs()
