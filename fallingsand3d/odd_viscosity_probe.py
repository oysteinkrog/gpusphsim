"""
odd_viscosity_probe.py -- Falsification battery for the "Scan-Order Hall Fluid"
theory as it applies to fallingsand3d.

THEORY (short form)
-------------------
Any fluid update that breaks parity (a fixed sweep direction, a diagonally
biased move rule, or an antisymmetric force such as vorticity confinement that
uses omega_j instead of omega_j - omega_i) injects, at leading truncation
order, a term identical to ODD (Hall) VISCOSITY:

    rho Dv/Dt = -grad p + nu lap(v) + eta_o lap(R v),   R = 90-deg rotation.

In 2D, R v = (v_y, -v_x). Odd viscosity is non-dissipative and, in the
INCOMPRESSIBLE BULK, is absorbed into the pressure (invisible). It only becomes
observable where fallingsand3d actually lives: at FREE SURFACES and in the
WEAKLY-COMPRESSIBLE bulk. Its fingerprint is a *transverse* response: a shear
(velocity varying in x) drives a perpendicular velocity/stress, with magnitude
proportional to k^2 (it is a viscosity, ~lap) -- NOT proportional to k^0 (which
is what a mundane Coriolis-like body force would give).

This script does NOT need a GPU. It is a minimal periodic weakly-compressible
linearized flow solver (finite differences, np.roll stencils) used purely as a
measurement rig. It runs five decisive tests. Each prints PASS/FAIL against the
theory so the theory can be *falsified*, not just illustrated.

The same five observables are what the GPU pytest (test_odd_viscosity.py) probes
in the real SPH sim. If the CPU rig and the SPH sim disagree in SIGN or SCALING,
the theory is wrong.

Run:  python odd_viscosity_probe.py
"""

import numpy as np


# --------------------------------------------------------------------------
# Minimal periodic 2D weakly-compressible linearized solver (measurement rig)
# --------------------------------------------------------------------------
# Fields: u, v (velocity), rho (density perturbation about rho0=1).
# Linearized acoustics + Newtonian viscosity nu + odd viscosity eta_o.
#
#   rho_t = -div(u)
#   u_t   = -c^2 dx(rho) + nu lap(u) + eta_o lap(v)        # +R applied to v
#   v_t   = -c^2 dy(rho) + nu lap(v) - eta_o lap(u)        #  (v_y,-v_x)
#
# We deliberately keep it linear so the leading-order odd-viscosity coupling is
# not contaminated by advective nonlinearity -- this is a clean coefficient rig.

class Rig:
    def __init__(self, N=64, L=2.0 * np.pi, c=1.0, nu=0.0, eta_o=0.0,
                 coriolis=0.0, chirality_stencil=0.0):
        self.N = N
        self.dx = L / N
        self.c = c
        self.nu = nu
        self.eta_o = eta_o          # explicit odd-viscosity coefficient
        self.coriolis = coriolis    # control: a k-independent body force omega*Rv
        self.chi = chirality_stencil  # emergent: asymmetric (parity-broken) stencil
        x = (np.arange(N) + 0.5) * self.dx
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")

    # central first derivatives (periodic)
    def dxf(self, f):
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * self.dx)

    def dyf(self, f):
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * self.dx)

    def lap(self, f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f) / self.dx ** 2

    def odd_operator(self, u, v):
        """Return (fx, fy): the parity-odd contribution to (u_t, v_t)."""
        fx = np.zeros_like(u)
        fy = np.zeros_like(v)
        # (1) EXPLICIT odd viscosity: eta_o * lap(R v),  R v = (v, -u)
        if self.eta_o != 0.0:
            fx += self.eta_o * self.lap(v)
            fy += -self.eta_o * self.lap(u)
        # (2) CORIOLIS control (a body force, NOT a viscosity): omega * R v
        if self.coriolis != 0.0:
            fx += self.coriolis * v
            fy += -self.coriolis * u
        # (3) EMERGENT parity-broken stencil: a fixed-direction (scan-order)
        #     asymmetric coupling between components. This mimics what a biased
        #     sequential update / asymmetric neighbor traversal does. It couples
        #     u and v through a one-sided (forward-difference) cross term, whose
        #     antisymmetric part is exactly an odd-viscosity Laplacian.
        if self.chi != 0.0:
            # one-sided (forward) second difference of the rotated field.
            # forward - centered differs by a term ~ dx * d^3, but the
            # component-coupling structure is the odd-viscosity one.
            fwd = lambda f: (np.roll(f, -1, 0) - 2 * f + np.roll(f, -1, 1)) / self.dx ** 2
            fx += self.chi * fwd(v)
            fy += -self.chi * fwd(u)
        return fx, fy

    def rhs(self, u, v, rho):
        ox, oy = self.odd_operator(u, v)
        u_t = -self.c ** 2 * self.dxf(rho) + self.nu * self.lap(u) + ox
        v_t = -self.c ** 2 * self.dyf(rho) + self.nu * self.lap(v) + oy
        rho_t = -(self.dxf(u) + self.dyf(v))
        return u_t, v_t, rho_t

    def step(self, u, v, rho, dt):
        # RK2 (midpoint) -- enough for a short leading-order measurement.
        u1, v1, r1 = self.rhs(u, v, rho)
        um, vm, rm = u + 0.5 * dt * u1, v + 0.5 * dt * v1, rho + 0.5 * dt * r1
        u2, v2, r2 = self.rhs(um, vm, rm)
        return u + dt * u2, v + dt * v2, rho + dt * r2


# --------------------------------------------------------------------------
# Measurement: induced transverse velocity from a pure shear mode
# --------------------------------------------------------------------------
def measure_transverse_coupling(rig, k_index, V=1e-3, steps=8, dt=None):
    """
    Initialize a pure shear mode  v(x) = V cos(k x),  u = 0, rho = 0,
    with k = k_index * (2 pi / L).  Integrate a few tiny steps and read off the
    induced longitudinal velocity u projected onto cos(k x).

    Returns coefficient  a(k) = <induced u, cos(kx)> / (V * t).
    Leading-order theory:  a(k) = -eta_o * k^2   (odd viscosity)
                           a(k) = -omega         (Coriolis body force, k-indep.)
    """
    N, dx = rig.N, rig.dx
    L = N * dx
    k = k_index * (2 * np.pi / L)
    if dt is None:
        dt = 0.02 * dx / rig.c  # very small, stay in leading order
    u = np.zeros((N, N))
    rho = np.zeros((N, N))
    v = V * np.cos(k * rig.X)

    for _ in range(steps):
        u, v, rho = rig.step(u, v, rho, dt)

    t = steps * dt
    basis = np.cos(k * rig.X)
    amp_u = 2.0 * np.mean(u * basis)        # projection amplitude of induced u
    return amp_u / (V * t), k


def fit_power(ks, a):
    """Fit |a| = C * k^p ; return (p, C) via log-log least squares."""
    ks = np.asarray(ks)
    a = np.abs(np.asarray(a))
    m = a > 0
    p, logC = np.polyfit(np.log(ks[m]), np.log(a[m]), 1)
    return p, np.exp(logC)


def banner(s):
    print("\n" + "=" * 74 + f"\n{s}\n" + "=" * 74)


def main():
    np.random.seed(0)  # determinism (no Date/random dependence)
    N = 64
    k_indices = [1, 2, 3, 4, 6, 8]

    banner("TEST 1  -- k^2 scaling discriminator (odd viscosity vs Coriolis)")
    print("Odd viscosity is a LAPLACIAN term -> transverse coupling ~ k^2.")
    print("A Coriolis-like body force is k-independent      -> ~ k^0.\n")

    # (a) explicit odd viscosity
    ks, a_odd = [], []
    for ki in k_indices:
        rig = Rig(N=N, eta_o=0.05)
        a, k = measure_transverse_coupling(rig, ki)
        ks.append(k); a_odd.append(a)
    p_odd, C_odd = fit_power(ks, a_odd)

    # (b) Coriolis control
    a_cor = []
    for ki in k_indices:
        rig = Rig(N=N, coriolis=0.05)
        a, k = measure_transverse_coupling(rig, ki)
        a_cor.append(a)
    p_cor, _ = fit_power(ks, a_cor)

    print(f"  odd-viscosity term : fitted exponent p = {p_odd:+.3f}  (theory: +2.000)")
    print(f"  Coriolis control   : fitted exponent p = {p_cor:+.3f}  (theory:  0.000)")
    print(f"  recovered eta_o from slope C = {C_odd:.4f}  (input: 0.0500)")
    t1 = abs(p_odd - 2.0) < 0.15 and abs(p_cor) < 0.15
    print(f"  --> {'PASS' if t1 else 'FAIL'}: k^2 fingerprint distinguishes odd viscosity from Coriolis")

    banner("TEST 2  -- sign antisymmetry (chirality flip)")
    print("Reversing the sign of eta_o must flip the transverse response exactly.\n")
    ap, _ = measure_transverse_coupling(Rig(N=N, eta_o=+0.05), 3)
    am, _ = measure_transverse_coupling(Rig(N=N, eta_o=-0.05), 3)
    asym = abs(ap + am) / (abs(ap) + abs(am) + 1e-30)
    print(f"  a(+eta_o) = {ap:+.6e}")
    print(f"  a(-eta_o) = {am:+.6e}")
    print(f"  residual |a+ + a-| / (|a+|+|a-|) = {asym:.2e}  (theory: 0)")
    t2 = asym < 1e-6
    print(f"  --> {'PASS' if t2 else 'FAIL'}: response is exactly antisymmetric in chirality")

    banner("TEST 3  -- null control (parity-symmetric update => no odd stress)")
    print("With eta_o = 0 and a symmetric stencil, transverse coupling must vanish.\n")
    a0, _ = measure_transverse_coupling(Rig(N=N, eta_o=0.0), 3)
    aref, _ = measure_transverse_coupling(Rig(N=N, eta_o=0.05), 3)
    print(f"  symmetric update  a = {a0:+.3e}")
    print(f"  odd-visc update   a = {aref:+.3e}")
    t3 = abs(a0) < 1e-9 and abs(aref) > 1e-6
    print(f"  --> {'PASS' if t3 else 'FAIL'}: the effect is absent unless parity is broken")

    banner("TEST 4  -- EMERGENT: a parity-broken *stencil* reproduces odd viscosity")
    print("No explicit eta_o term. Just a fixed-direction asymmetric coupling")
    print("stencil (a proxy for scan-order / biased neighbor traversal).")
    print("Prediction: same k^2 transverse fingerprint, sign set by the bias.\n")
    ks2, a_emg = [], []
    for ki in k_indices:
        rig = Rig(N=N, chirality_stencil=0.05)
        a, k = measure_transverse_coupling(rig, ki)
        ks2.append(k); a_emg.append(a)
    p_emg, _ = fit_power(ks2, a_emg)
    ap_e, _ = measure_transverse_coupling(Rig(N=N, chirality_stencil=+0.05), 3)
    am_e, _ = measure_transverse_coupling(Rig(N=N, chirality_stencil=-0.05), 3)
    flip_ok = (ap_e * am_e) < 0
    print(f"  emergent stencil : fitted exponent p = {p_emg:+.3f}  (theory ~ +2)")
    print(f"  bias +/-         : a(+)= {ap_e:+.3e}  a(-)= {am_e:+.3e}  (opposite sign: {flip_ok})")
    t4 = abs(p_emg - 2.0) < 0.35 and flip_ok
    print(f"  --> {'PASS' if t4 else 'FAIL'}: a biased stencil IS an odd-viscosity source")

    banner("TEST 5  -- incompressible invisibility (the 'hides in pressure' claim)")
    print("Odd viscosity in the INCOMPRESSIBLE bulk is a pressure redefinition. Its")
    print("contribution to the VORTICITY (curl of momentum eq) is eta_o*lap(div u),")
    print("so it drives real (solenoidal) flow ONLY through compressibility. The")
    print("INDUCED velocity response should therefore be almost entirely")
    print("DILATATIONAL (curl-free = pressure/compression channel), not solenoidal.\n")

    def induced_dilatational_fraction(c):
        rig = Rig(N=N, eta_o=0.05, c=c)
        N_, dx = rig.N, rig.dx
        L = N_ * dx
        ki = 3
        k = ki * (2 * np.pi / L)
        u0 = np.zeros((N_, N_)); rho = np.zeros((N_, N_))
        v0 = 1e-3 * np.cos(k * rig.X)
        u, v = u0.copy(), v0.copy()
        dt = 0.02 * dx / rig.c
        for _ in range(8):
            u, v, rho = rig.step(u, v, rho, dt)
        # INDUCED field only (subtract the initial shear mode)
        du, dv = u - u0, v - v0
        uh = np.fft.fftn(du); vh = np.fft.fftn(dv)
        kx = np.fft.fftfreq(N_, d=dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        K2 = KX ** 2 + KY ** 2; K2[0, 0] = 1.0
        div = (KX * uh + KY * vh)
        uh_d = KX * div / K2                  # dilatational (curl-free) part
        vh_d = KY * div / K2
        dil = np.sum(np.abs(uh_d) ** 2 + np.abs(vh_d) ** 2)
        tot = np.sum(np.abs(uh) ** 2 + np.abs(vh) ** 2) + 1e-30
        return dil / tot

    f1 = induced_dilatational_fraction(c=1.0)
    f2 = induced_dilatational_fraction(c=4.0)
    print(f"  dilatational (pressure-channel) fraction of INDUCED response, c=1 : {f1:.4f}")
    print(f"  dilatational (pressure-channel) fraction of INDUCED response, c=4 : {f2:.4f}")
    print("  (theory: ~1.0 -- the odd response lives in the compression/pressure")
    print("   channel, so an incompressibility constraint would erase it entirely)")
    t5 = f1 > 0.98 and f2 > 0.98
    print(f"  --> {'PASS' if t5 else 'FAIL'}: odd-viscosity response is a pressure effect,")
    print(f"      exposed here only because the flow is (weakly) compressible")

    banner("VERDICT")
    results = {
        "T1 k^2 vs Coriolis discriminator": t1,
        "T2 chirality antisymmetry": t2,
        "T3 null control (parity-symmetric)": t3,
        "T4 emergent stencil = odd viscosity": t4,
        "T5 incompressible invisibility": t5,
    }
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allpass = all(results.values())
    print("\n  " + ("ALL PASS -- theory survives every falsification test on the CPU rig."
                     if allpass else
                     "SOME FAIL -- theory is (at least partly) falsified; inspect above."))
    print("  Next: run test_odd_viscosity.py on the CUDA box to check the *real* SPH")
    print("  sim shows the same sign + k^2 scaling. Divergence there falsifies the")
    print("  claim that fallingsand3d's spurious rotation is odd-viscosity-driven.")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
