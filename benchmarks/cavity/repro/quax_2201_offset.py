"""The 1 % absolute-f offset of QUAX 2201.04223 (beta).

    python benchmarks/cavity/repro/quax_2201_offset.py

The reproduction on docs/validation/cavity_reproductions.md (QUAX two-shell) finds: at the stated dimensions (4.2 K, eps 11.2, 2004
cones) axionFEM gives f = 10.3685 GHz vs Table II 10.4745 GHz (+1.02 %), while every hard target (Delta f(eps),
Delta f(T), p_e, C trend) agrees. Cones, end gaps, contraction set, D_cav and the stated tube tolerances were swept
and cannot supply 106 MHz.

Systematic question here: which single input, changed by how much, moves f onto 10.4745 GHz, and does that change
keep the hard targets (p_e = 0.170, Delta f(eps 11.44 -> 11.2) = 18.9 MHz, C030 = 0.0321 up to the cone factor)?
Each candidate is judged against its plausibility (stated tolerance, a plausible typo, a modelling convention).
Method: exact layered Bessel model of the z-invariant cavity (flat ends; the cones add a constant
Delta_cone = f_cones - f_flat taken from the round-1 axionFEM records, +4.5 MHz, and multiply C by the round-1
ratio C_cones/C_flat), with C and p_e by quadrature; spot checks against the axionFEM 2D solver (flat ends).
Writes benchmarks/records/cavity_repro_quax_2201_offset.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

MM = 1e-3
C0 = cm.C_LIGHT
CU, SA = 3.26e-3, 8e-4  # contraction set A (293 -> 4.2 K)
NOM = dict(R=28.55, r1=10.875, t1=2.35, r2=20.90, t2=1.94, eps1=11.2, eps2=11.2, cu=CU, sa=SA)
F_TAB = 10.4745e9
DF_EPS_TAB = 18.9e6
PE_TAB = 0.170
C_TAB = 0.0321


def layers(p):
    s = lambda x: x * (1 - p["sa"]) * MM  # noqa: E731
    R = p["R"] * (1 - p["cu"]) * MM
    return [(s(p["r1"]), 1.0), (s(p["r1"] + p["t1"]), p["eps1"]), (s(p["r2"]), 1.0), (s(p["r2"] + p["t2"]), p["eps2"]),
            (R, 1.0)], R


def coeffs(k0, lay):
    """(a, b) of E_z = a J0 + b Y0 in each layer, starting from J0 on the axis."""
    out = []
    a, b = 1.0, 0.0
    k = k0 * math.sqrt(lay[0][1])
    out.append((a, b, k))
    for (ro, _), (_, eps2) in zip(lay[:-1], lay[1:]):
        e = a * j0(k * ro) + b * y0(k * ro)
        de = -k * (a * j1(k * ro) + b * y1(k * ro))
        k2 = k0 * math.sqrt(eps2)
        M = np.array([[j0(k2 * ro), y0(k2 * ro)], [-k2 * j1(k2 * ro), -k2 * y1(k2 * ro)]])
        a, b = np.linalg.solve(M, [e, de])
        k = k2
        out.append((a, b, k))
    return out


def ez_wall(f, lay, R):
    a, b, k = coeffs(2 * math.pi * f / C0, lay)[-1]
    return a * j0(k * R) + b * y0(k * R)


def roots(lay, R, fmin=9.8e9, fmax=11.0e9, n=1200):
    fs = np.linspace(fmin, fmax, n)
    v = [ez_wall(f, lay, R) for f in fs]
    return [brentq(ez_wall, fs[i], fs[i + 1], args=(lay, R), xtol=1e-3) for i in range(n - 1) if v[i] * v[i + 1] < 0]


def props(f, lay, R):
    cs = coeffs(2 * math.pi * f / C0, lay)
    S = D = De = 0.0
    r0 = 0.0
    for (ro, eps), (a, b, k) in zip(lay, cs):
        u = (lambda r, a=a, b=b, k=k: a * j0(k * r) + b * y0(k * r)) if r0 > 0 else (lambda r, a=a, k=k: a * j0(k * r))
        S += quad(lambda r: r * u(r), r0, ro, limit=200)[0]
        d = eps * quad(lambda r: r * u(r) ** 2, r0, ro, limit=200)[0]
        D += d
        if eps > 1:
            De += d
        r0 = ro
    return {"C_flat": 2 * S ** 2 / (R ** 2 * D), "p_e": De / D}


def mode030(p):
    """The pseudo-TM030 (largest C of the z-uniform TM0n0 between 9.8 and 11 GHz)."""
    lay, R = layers(p)
    best = None
    for f in roots(lay, R):
        pr = props(f, lay, R)
        if best is None or pr["C_flat"] > best["C_flat"]:
            best = {"f_Hz": f, **pr}
    return best


def main():
    rec1 = json.loads((cm.RECS / "cavity_repro_quax_two_shell_2201.json").read_text())
    nom = mode030(NOM)
    # cone correction from the round-1 axionFEM records (baseline cones vs flat ends, same dimensions)
    runs = rec1["runs"]
    f_cones, C_cones = runs["base_o3_h0.7"]["f_Hz"], runs["base_o3_h0.7"]["C"]
    out_flat_fem = runs["cone_h0.0_o2_h0.5"]  # axionFEM flat ends (cone height 0), for the layered cross-check
    d_cone = f_cones - nom["f_Hz"]
    c_fac = C_cones / nom["C_flat"]
    target_flat = F_TAB - d_cone
    out = {"nominal_flat": nom, "cone_shift_Hz": d_cone, "cone_C_factor": c_fac, "target_f_flat_Hz": target_flat,
           "axionFEM_flat_round1": {"f_Hz": out_flat_fem["f_Hz"], "C": out_flat_fem["C"],
                                    "p_e": out_flat_fem.get("p_e")},
           "candidates": []}

    def df_eps(p):
        q = dict(p, eps1=11.44 * p["eps1"] / 11.2, eps2=11.44 * p["eps2"] / 11.2)
        return mode030(p)["f_Hz"] - mode030(q)["f_Hz"]
    out["nominal_flat"]["df_eps_Hz"] = df_eps(NOM)

    def solve_for(key, lo, hi, transform=None):
        def g(x):
            p = dict(NOM)
            if transform:
                p.update(transform(x))
            else:
                p[key] = x
            return mode030(p)["f_Hz"] - target_flat
        try:
            x = brentq(g, lo, hi, xtol=1e-6)
        except ValueError:
            return None
        p = dict(NOM)
        if transform:
            p.update(transform(x))
        else:
            p[key] = x
        m = mode030(p)
        return x, p, m, df_eps(p)
    cands = [
        ("R_cav [mm]", "R", 23.5, 26.0, None, "D_cav 57.1 mm (Fig. 1a) / 57 mm (text)"),
        ("tube-1 inner radius [mm] (t1 fixed)", "r1", 10.0, 10.875, None, "ID 21.75 +- 0.15 mm (r +- 0.075)"),
        ("tube-1 thickness [mm] (r1 fixed)", "t1", 1.5, 2.35, None, "OD tolerance +- 0.05 mm (t +- 0.025)"),
        ("tube-2 inner radius [mm] (t2 fixed)", "r2", 19.0, 21.5, None, "not stated"),
        ("tube-2 thickness [mm] (r2 fixed)", "t2", 0.3, 1.94, None, "not stated"),
        ("eps_r both tubes", "eps", 9.0, 11.2, lambda x: {"eps1": x, "eps2": x}, "11.2 measured at 4.2 K (text)"),
        ("eps_r tube 1 only", "eps1", 9.0, 11.2, None, "same sapphire"),
        ("eps_r tube 2 only", "eps2", 5.0, 11.2, None, "same sapphire"),
        ("uniform scale of all dimensions", "scale", 0.97, 1.0,
         lambda x: {k: NOM[k] * x for k in ("R", "r1", "t1", "r2", "t2")}, "a common contraction/units factor"),
        ("uniform scale of the tube radii (cavity fixed)", "tscale", 0.97, 1.0,
         lambda x: {k: NOM[k] * x for k in ("r1", "t1", "r2", "t2")}, "tube drawing scale"),
    ]
    for name, key, lo, hi, tr, note in cands:
        r = solve_for(key, lo, hi, tr)
        if r is None:
            out["candidates"].append({"parameter": name, "required": None, "note": note,
                                      "result": "no solution in the scanned range"})
            continue
        x, p, m, dfe = r
        nomv = (NOM[key] if key in NOM else (11.2 if key == "eps" else 1.0))
        out["candidates"].append({
            "parameter": name, "nominal": nomv, "required": x, "change": x - nomv, "note": note,
            "f_flat_Hz": m["f_Hz"], "f_with_cones_Hz": m["f_Hz"] + d_cone, "C_with_cones": m["C_flat"] * c_fac,
            "C_rel_to_table": m["C_flat"] * c_fac / C_TAB - 1, "p_e": m["p_e"], "p_e_rel_to_table": m["p_e"] / PE_TAB - 1,
            "df_eps_Hz": dfe, "df_eps_rel_to_table": dfe / DF_EPS_TAB - 1})
    p = cm.write_record("quax_2201_offset", out)
    print(p)
    print("nominal flat", nom, "df_eps", out["nominal_flat"]["df_eps_Hz"], "cone shift", d_cone, "C factor", c_fac)
    for c in out["candidates"]:
        if c.get("required") is None:
            print(c["parameter"], c["result"])
            continue
        print(f"{c['parameter']:48s} {c['nominal']:.4g} -> {c['required']:.5g} ({c['change']:+.4g}); C {c['C_with_cones']:.4f} "
              f"({c['C_rel_to_table']:+.1%}) p_e {c['p_e']:.4f} ({c['p_e_rel_to_table']:+.1%}) "
              f"df_eps {c['df_eps_Hz'] / 1e6:.2f} MHz ({c['df_eps_rel_to_table']:+.1%})")


if __name__ == "__main__":
    main()
