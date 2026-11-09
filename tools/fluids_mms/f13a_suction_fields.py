"""Inductionless MHD with suction: derivation and 40-digit references of analytic_hartmann_suction_2d, the non-MMS case
of the convection row.

  1. ansatz u = (U(y), V, 0) (in-plane) and u = (0, V, W(y)) (duct), phi = -E x (E = 0 here), p = -G_x x + P0,
     substituted into the CARTESIAN inductionless MHD equations WITH rho (u.grad)u: the residuals are
     functions of y only and reduce to  mu F'' - rho V F' - sigma B0^2 F + G = 0  (checked symbolically);
  2. dsolve of that ODE, constants from F(-1) = F(1) = 0;
  3. back-substitution into the Cartesian equations at rational points (40 digits);
  4. the fixed numpy closed form (benchmarks/fluids/cases/f13a_suction_check.py) against mpmath: F at 41 points and
     Q = int F dy (quadrature), written to tools/fluids_mms/f13a_reference/mhd_suction_reference.json.

    python3 -B tools/fluids_mms/f13a_suction_fields.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import f13a_suction_check as SC  # noqa: E402

OUT = HERE / "f13a_reference" / "mhd_suction_reference.json"
x, y = sp.symbols("x y", real=True)
mu, sig, rho, G, B0, Vs = sp.symbols("mu sigma rho G B0 V", positive=True)


def residuals(u, phi, p, B, Gz, Ez=0):
    uxB = sp.Matrix([u[1] * B[2] - u[2] * B[1], u[2] * B[0] - u[0] * B[2], u[0] * B[1] - u[1] * B[0]])
    J = sig * sp.Matrix([-sp.diff(phi, x) + uxB[0], -sp.diff(phi, y) + uxB[1], Ez + uxB[2]])
    F = sp.Matrix([J[1] * B[2] - J[2] * B[1], J[2] * B[0] - J[0] * B[2], J[0] * B[1] - J[1] * B[0]])
    lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)  # noqa: E731
    conv = [rho * (u[0] * sp.diff(u[i], x) + u[1] * sp.diff(u[i], y)) for i in range(3)]
    gp = [sp.diff(p, x), sp.diff(p, y), 0]
    mom = [sp.simplify(conv[i] - mu * lap(u[i]) + gp[i] - (Gz if i == 2 else 0) - F[i]) for i in range(3)]
    return mom + [sp.simplify(sp.diff(u[0], x) + sp.diff(u[1], y)), sp.simplify(sp.diff(J[0], x) + sp.diff(J[1], y))]


def main():
    Ff = sp.Function("F")
    B = sp.Matrix([0, B0, 0])
    out = {"derivation": {}, "variants": {}}
    # in-plane: u = (F, V, 0), p = G (LX - x), Gz = 0;  duct: u = (0, V, F), p = 0, Gz = G, phi = 0
    LX = sp.Rational(1, 4)
    for mode, u, p, Gz in (("inplane", sp.Matrix([Ff(y), Vs, 0]), G * (LX - x), 0),
                           ("duct", sp.Matrix([0, Vs, Ff(y)]), sp.Integer(0), G)):
        res = residuals(u, sp.Integer(0), p, B, Gz)
        idx = 0 if mode == "inplane" else 2
        ode = sp.expand(res[idx])
        target = sp.expand(-(mu * Ff(y).diff(y, 2) - rho * Vs * Ff(y).diff(y) - sig * B0 ** 2 * Ff(y) + G))
        assert sp.simplify(ode - target) == 0, (mode, ode)
        others = [sp.simplify(r) for i, r in enumerate(res) if i != idx]
        assert all(o == 0 for o in others), (mode, others)
        out["derivation"][mode] = {"ode": str(ode), "other_residuals": [str(o) for o in others]}
        print(f"[{mode}] section-1 residual = {ode}; the other four vanish identically")
    gen = sp.dsolve(sp.Eq(mu * Ff(y).diff(y, 2) - rho * Vs * Ff(y).diff(y) - sig * B0 ** 2 * Ff(y) + G, 0), Ff(y)).rhs
    out["derivation"]["general"] = str(gen)
    print("[general]", gen)
    mp.mp.dps = 40
    for key, var in SC.VARIANTS.items():
        ex = SC.exact(key)
        vals = {mu: sp.Rational(1, 2), sig: 2, rho: 1, G: 1, B0: sp.nsimplify(ex.B0), Vs: sp.nsimplify(var["V"])}
        g = gen.subs(vals)
        c = sorted(g.free_symbols - {y}, key=str)
        sol = sp.solve([g.subs(y, -1), g.subs(y, 1)], c, dict=True)[0]
        Fy = g.subs(sol)
        # back-substitution in the Cartesian equations (40 digits)
        u = sp.Matrix([Fy, vals[Vs], 0]) if var["mode"] == "inplane" else sp.Matrix([0, vals[Vs], Fy])
        p = vals[G] * (LX - x) if var["mode"] == "inplane" else sp.Integer(0)
        res = [r.subs(vals) for r in residuals(u, sp.Integer(0), p, B, 0 if var["mode"] == "inplane" else G)]
        worst = max(abs(sp.N(r.subs({x: sp.Rational(1, 7), y: yy}), 40)) for r in res
                    for yy in (sp.Rational(-9, 10), sp.Rational(1, 3), sp.Rational(7, 8)))
        assert worst < 1e-30, worst
        f = sp.lambdify(y, Fy, "mpmath")
        Q = mp.quad(f, [-1, -0.5, 0, 0.5, 1])
        ys = np.linspace(-1, 1, 41)
        dF = max(abs(float(f(mp.mpf(float(t)))) - float(ex.F(np.array(t)))) for t in ys) / max(abs(float(f(mp.mpf(float(t))))) for t in ys)
        dQ = abs(float(Q) - ex.Q) / abs(float(Q))
        print(f"[{key}] residual {float(worst):.1e}; numpy vs 40-digit: F {dF:.1e}, Q {dQ:.1e} (Q = {float(Q):.15g})")
        assert dF < 1e-13 and dQ < 1e-13
        out["variants"][key] = {"Q": str(Q), "residual_40digit": float(worst), "numpy_vs_mpmath": {"F": dF, "Q": dQ},
                                "B0": ex.B0, "V": var["V"], "Ha": var["Ha"], "mode": var["mode"]}
    out["check_module_sha256"] = hashlib.sha256((ROOT / "benchmarks/fluids/cases/f13a_suction_check.py").read_bytes()).hexdigest()
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("wrote", OUT.relative_to(ROOT), hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
