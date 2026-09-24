"""Tests for kernel_loader.load_module (shared RawModule compile and cache)."""

from __future__ import annotations

import pytest

import kernel_loader
import step1
import step2


def test_same_object_on_second_call():
    a = kernel_loader.load_module("step1.cu")
    b = kernel_loader.load_module("step1.cu")
    assert a is b


def test_step_modules_use_shared_cache():
    assert step1.get_module() is kernel_loader.load_module("step1.cu")
    assert step2.get_module() is kernel_loader.load_module("step2.cu")
    assert step1.get_module() is not step2.get_module()


def test_extra_options_are_honoured():
    extra = ("-DKERNEL_LOADER_TEST_DEFINE=1",)
    opts = kernel_loader.compile_options(extra)
    assert opts[: len(kernel_loader.BASE_OPTIONS)] == kernel_loader.BASE_OPTIONS
    assert f"-I{kernel_loader.KERNEL_DIR}" in opts
    assert opts[-1] == extra[0]

    plain = kernel_loader.load_module("step1.cu")
    with_define = kernel_loader.load_module("step1.cu", extra)
    assert with_define is not plain
    assert with_define.options[-1] == extra[0]
    assert kernel_loader.load_module("step1.cu", extra) is with_define
    # Compiles and exposes the same kernels as the plain build.
    assert with_define.get_function("K_Step1") is not None


def test_missing_cu_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="no_such_kernel.cu"):
        kernel_loader.load_module("no_such_kernel.cu")
