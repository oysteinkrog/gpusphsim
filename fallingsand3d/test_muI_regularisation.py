"""mu(I) viscosity regularisation (bd-r5-review-n28.107).

compute_muI_eta divides by max(gamma_dot, MUI_GAMMA_DOT_FLOOR) with a floor of
0.1 1/s. With the cap lifted, eta at gamma_dot = 0 must be the finite value
mu0 + mu_s * p_eff / 0.1 and must scale with pressure. With the default
mu_max, resting sand at the SAND pressure floor must still sit at the cap.
"""

import cupy
import numpy as np

from kernel_loader import compile_options, ensure_ptx_if_needed
from step2 import DEFAULT_MU_MAX, build_granular_params

SRC = r"""
#include "sph_shared.cuh"
extern "C" __global__
void eval_eta(const float* gd, const float* p, const float* rho, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = compute_muI_eta(gd[i], p[i], rho[i]);
}
"""

FLOOR = 0.1


def _eval(params, gd, p, rho):
    ensure_ptx_if_needed()
    mod = cupy.RawModule(code=SRC, options=compile_options())
    ptr = mod.get_global("c_granular")
    cupy.ndarray(params.nbytes, dtype=cupy.uint8, memptr=ptr).set(params.view(np.uint8))
    gd = cupy.asarray(gd, dtype=cupy.float32)
    p = cupy.asarray(p, dtype=cupy.float32)
    rho = cupy.asarray(rho, dtype=cupy.float32)
    out = cupy.zeros_like(gd)
    n = gd.size
    mod.get_function("eval_eta")(((n + 127) // 128,), (128,), (gd, p, rho, out, np.int32(n)))
    return cupy.asnumpy(out)


def _ref(params, gd, p, rho):
    g = params[0]
    I = gd * g["particle_spacing"] / np.sqrt(p / rho)
    mu_I = g["mu_s"] + (g["mu_2"] - g["mu_s"]) / (1.0 + g["I0"] / np.maximum(I, 1e-8))
    return np.minimum(g["mu_max"], g["mu0"] + mu_I * p / np.maximum(gd, FLOOR))


def test_eta_finite_and_pressure_scaled_below_floor():
    params = build_granular_params(mu_max=1e9)
    gd = np.array([0.0, 1e-4, 0.05, 0.0, 0.0], dtype=np.float32)
    p = np.array([490.0, 490.0, 490.0, 980.0, 50.0], dtype=np.float32)
    rho = np.full_like(gd, 2500.0)
    eta = _eval(params, gd, p, rho)
    assert np.all(np.isfinite(eta))
    # Old form gave mu_s * 490 / 1e-6 = 2.7e8 at gamma_dot 0.
    np.testing.assert_allclose(eta, _ref(params, gd, p, rho), rtol=2e-3)
    mu0, mu_s = float(params[0]["mu0"]), float(params[0]["mu_s"])
    np.testing.assert_allclose(eta[0], mu0 + mu_s * 490.0 / FLOOR, rtol=2e-3)
    # Below the floor, eta - mu0 is linear in pressure.
    np.testing.assert_allclose((eta[3] - mu0) / (eta[0] - mu0), 2.0, rtol=2e-3)


def test_eta_unchanged_above_floor():
    params = build_granular_params(mu_max=1e9)
    gd = np.array([0.2, 1.0, 5.0, 50.0], dtype=np.float32)
    p = np.full_like(gd, 800.0)
    rho = np.full_like(gd, 2500.0)
    eta = _eval(params, gd, p, rho)
    np.testing.assert_allclose(eta, _ref(params, gd, p, rho), rtol=2e-3)


def test_resting_sand_stays_at_default_cap():
    params = build_granular_params()
    gd = np.array([0.0, 0.05, 0.5], dtype=np.float32)
    p = np.full_like(gd, 2500.0 * 9.81 * 0.02)  # SAND pressure floor
    rho = np.full_like(gd, 2500.0)
    eta = _eval(params, gd, p, rho)
    np.testing.assert_allclose(eta, float(DEFAULT_MU_MAX), rtol=1e-6)
