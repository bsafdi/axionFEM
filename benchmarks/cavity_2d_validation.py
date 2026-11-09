"""Cavity haloscopes, axisymmetric (beta): convergence against closed forms.

    python benchmarks/cavity_2d_validation.py [workdir] [--only mc1,mc2,mc3,mc4,mf]

Writes benchmarks/records/cavity_mc{1..4}_*.json (one per check), cavity_mf_fields.json (per-mode field
export: pointwise accuracy of Results.cavity_mode_field / cavity_mode_sample) and regenerates
docs/validation/cavity_2d.md from every record present. References:
closed forms derived independently of the code, and axionfem.cavity.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from axionfem import cavity as cv

ROOT = Path(__file__).resolve().parents[1]
RECS = ROOT / "benchmarks" / "records"
OUT = ROOT / "docs" / "validation" / "cavity_2d.md"
C = cv.C_LIGHT
Z0 = cv.Z0
A, L = 0.1, 0.2
CHI0 = [2.404825557695773, 5.520078110286311, 8.653727912911013]
CHI1P = 3.831705970207512
HS = [0.02, 0.01, 0.005, 0.0025]
RHO = 1.7e-8
WALL = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": RHO, "T_K": 4.0}}
FILES = {"mc1": "cavity_mc1_modes.json", "mc2": "cavity_mc2_target.json", "mc3": "cavity_mc3_form_factor.json",
         "mc4": "cavity_mc4_q0.json", "mf": "cavity_mf_fields.json"}


def _f(k):
    return C * k / (2 * math.pi)


def _near(modes, f):
    return min(modes, key=lambda m: abs(m["f_Hz"] - f))


def _solve(d, wd):
    t0 = time.time()
    r = d.solve(wd, verbose=0)
    return r, time.time() - t0


REF = {"TM010": ("TM", _f(CHI0[0] / A)), "TM011": ("TM", _f(math.hypot(CHI0[0] / A, math.pi / L))),
       "TE011": ("TE", _f(math.hypot(CHI1P / A, math.pi / L)))}


def mc1(wd: Path):
    rows = []
    for order in (2, 3):
        for h in (HS if order == 2 else HS[:3]):
            d = cv.pillbox_design(A, L, h=h, order=order, modes=4, domain_radius=1.0)
            r, dt = _solve(d, wd / f"mc1_{order}_{h}")
            fw = r.data["fullwave"]
            row = {"order": order, "h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt,
                   "excluded_components": fw["cavity"]["excluded_components"]}
            for lab, (fam, f) in REF.items():
                m = _near(fw["eigenmodes"][fam]["modes"], f)
                row[lab] = {"f_Hz": m["f_Hz"], "f_exact_Hz": f, "rel": m["f_Hz"] / f - 1,
                            "residual_true": m["residual_true"], "energy_fraction": m["energy_fraction"]}
            rows.append(row)
    return {"case": "copper pillbox a = 0.1 m, L = 0.2 m (wall t = 1 cm), closed sphere R = 1 m, cavity restricted",
            "rows": rows}


def mc2(wd: Path):
    ft = 2.5 * _f(CHI0[0] / A)
    ref = [m for m in cv.pillbox_modes(A, L, 1.3 * ft) if abs(m["f_Hz"] / ft - 1) < 0.2]
    out = {"case": "pillbox, target_f_Hz = 2.5 f(TM010), 12 modes per family, window +-20 %",
           "target_f_Hz": ft, "runs": []}
    for h in (0.005, 0.0025):
        d = cv.pillbox_design(A, L, h=h, modes=12, target_f_Hz=ft, domain_radius=1.0)
        r, dt = _solve(d, wd / f"mc2_{h}")
        fw = r.data["fullwave"]
        run = {"h_m": h, "seconds": dt, "solver": {k: fw["eigenmodes"][k]["solver"] for k in ("TM", "TE")}, "modes": []}
        for m in ref:
            g = _near(fw["eigenmodes"][m["family"]]["modes"], m["f_Hz"])
            run["modes"].append({"mode": f"{m['family']}0{m['n']}{m['p']}", "f_exact_Hz": m["f_Hz"], "f_Hz": g["f_Hz"],
                                 "rel": g["f_Hz"] / m["f_Hz"] - 1, "residual_true": g["residual_true"]})
        run["complete"] = all(abs(x["rel"]) < 1e-4 for x in run["modes"])
        out["runs"].append(run)
    return out


def mc3(wd: Path):
    exact = [0.6916602761, 0.1312712271, 0.0534138053]
    rows = []
    for h in HS:
        d = cv.pillbox_design(A, L, h=h, modes=12, families=("TM", "TE"), domain_radius=1.0)
        r, _ = _solve(d, wd / f"mc3_{h}")
        fw = r.data["fullwave"]
        tm = fw["eigenmodes"]["TM"]["modes"]
        row = {"h_m": h}
        for n, (chi, Cx) in enumerate(zip(CHI0, exact), start=1):
            m = _near(tm, _f(chi / A))
            row[f"C0{n}0"] = {"C": m["form_factor"]["C"], "exact": Cx, "rel": m["form_factor"]["C"] / Cx - 1}
        c010 = row["C010"]["C"]
        ref = [x for x in cv.pillbox_modes(A, L, 6e9) if x["family"] == "TM"]
        p_of = [_near(ref, m["f_Hz"])["p"] for m in tm]  # label each computed mode by the nearest analytic one
        row["max_C_TM0np_p_gt_0_over_C010"] = max(m["form_factor"]["C"] for m, p in zip(tm, p_of) if p > 0) / c010
        row["max_C_TE"] = max(m["form_factor"]["C"] for m in fw["eigenmodes"]["TE"]["modes"])
        rows.append(row)
    ref = cv.coax_tm_mode(0.05, 0.2)
    coax = []
    for h in HS:
        d = cv.coax_design(0.05, 0.2, 0.2, h=h, modes=4, domain_radius=1.0)
        r, _ = _solve(d, wd / f"mc3_coax_{h}")
        m = _near(r.data["fullwave"]["eigenmodes"]["TM"]["modes"], ref["f_Hz"])
        coax.append({"h_m": h, "f_Hz": m["f_Hz"], "rel_f": m["f_Hz"] / ref["f_Hz"] - 1, "C": m["form_factor"]["C"],
                     "rel_C": m["form_factor"]["C"] / 0.7949373125 - 1})
    sref = cv.sleeve_tm_mode(A, 0.03, 0.05, 4.0)
    sleeve = []
    for h in HS[1:]:
        d = cv.pillbox_design(A, L, h=h, modes=2, families=("TM",), domain_radius=1.0)
        d.doc["materials"]["sleeve"] = {"kind": "linear", "eps_r": 4.0}
        d.doc["regions"].insert(1, {"name": "sleeve", "shape": {"rect": {"r": [0.03, 0.05], "z": [-L / 2, L / 2]}},
                                    "material": "sleeve", "mesh_size": h})
        r, _ = _solve(d, wd / f"mc3_sleeve_{h}")
        m = r.data["fullwave"]["eigenmodes"]["TM"]["modes"][0]
        sleeve.append({"h_m": h, "f_Hz": m["f_Hz"], "rel_f": m["f_Hz"] / sref["f_Hz"] - 1, "C": m["form_factor"]["C"],
                       "rel_C": m["form_factor"]["C"] / sref["C"] - 1})
    return {"case": "pillbox (axial uniform background B0), coax b = 0.2, a = 0.05, L = 0.2 (rod spans both caps), "
                    "pillbox with a dielectric sleeve 0.03 < r < 0.05, eps_r = 4, over the full length",
            "pillbox": rows, "coax_exact": {"kb": ref["kb"], "f_Hz": ref["f_Hz"], "C": ref["C"]}, "coax": coax,
            "sleeve_exact": {"f_Hz": sref["f_Hz"], "C": sref["C"]}, "sleeve": sleeve}


def mc4(wd: Path):
    rows = []
    kc, beta = CHI1P / A, math.pi / L
    k = math.hypot(kc, beta)
    G = {"TM010": Z0 * CHI0[0] * L / (2 * (A + L)),
         "TM011": Z0 * _f(math.hypot(CHI0[0] / A, math.pi / L)) * 2 * math.pi / C * A * L / (2 * (L + 2 * A)),
         "TE011": Z0 * k ** 3 * A * L / (2 * (kc ** 2 * L + 2 * A * beta ** 2))}
    for h in HS:
        d = cv.pillbox_design(A, L, h=h, modes=4, wall_material=WALL, domain_radius=1.0)
        r, _ = _solve(d, wd / f"mc4_{h}")
        fw = r.data["fullwave"]
        row = {"h_m": h}
        for lab, (fam, f) in REF.items():
            m = _near(fw["eigenmodes"][fam]["modes"], f)
            lo = m["losses"]
            Rs = cv.skin_Rs(m["f_Hz"], RHO)
            row[lab] = {"G_ohm": lo["G_ohm"], "G_exact_ohm": G[lab], "rel": lo["G_ohm"] / G[lab] - 1,
                        "Q0_Rs_rel": lo["Q0"] * Rs / G[lab] - 1,
                        "rel_pointwise": lo["walls"]["wall"].get("G_ohm_pointwise", lo["G_ohm"]) / G[lab] - 1}
        rows.append(row)
    d = cv.pillbox_design(A, L, h=0.005, modes=2, families=("TM",), domain_radius=1.0,
                          fill={"kind": "linear", "eps_r": 2.0, "tan_delta": 1e-4}, wall_material=WALL)
    r, _ = _solve(d, wd / "mc4_diel")
    m = r.data["fullwave"]["eigenmodes"]["TM"]["modes"][0]
    diel = {"eps_r": 2.0, "tan_delta": 1e-4, "f_rel": m["f_Hz"] * math.sqrt(2.0) / _f(CHI0[0] / A) - 1,
            "Q_dielectric": m["losses"]["Q_dielectric"], "Q_d_rel": m["losses"]["Q_dielectric"] * 1e-4 - 1,
            "C": m["form_factor"]["C"], "C_rel_vs_C010_over_eps": m["form_factor"]["C"] * 2.0 / (4 / CHI0[0] ** 2) - 1,
            "Q0": m["losses"]["Q0"], "Q_wall": m["losses"]["Q_wall"]}
    return {"case": "pillbox, copper wall rf skin model (rho = 1.7e-8 ohm m, any rho), geometry factor G = Q0 R_s",
            "rows": rows, "dielectric_fill": diel}


def mf(wd: Path):
    """Per-mode field export: pointwise errors of the exported fields against the closed forms (normalized to
    int |E|^2 dV = 1) on a 41 x 21 probe grid r <= 0.995 a, |z| <= 0.098 m, relative to the field scale; and the
    normalization / energy balance by 48 x 48 Gauss-Legendre quadrature of cavity_mode_sample over the cavity."""
    import numpy as np
    from scipy.special import j0, j1, roots_legendre

    E10 = 1 / math.sqrt(math.pi * A * A * L * j1(CHI0[0]) ** 2)
    E11 = math.sqrt(2 / (math.pi * A * A * L * j0(CHI1P) ** 2))
    x, w = roots_legendre(48)
    R, Z = np.meshgrid(0.5 * A * (x + 1), 0.5 * L * x)
    W = np.outer(w, w) * 0.25 * A * L * 2 * math.pi * R
    rows = []
    for h in HS:
        d = cv.pillbox_design(A, L, h=h, modes=2, domain_radius=1.0)
        r, dt = _solve(d, wd / f"mf_{h}")
        tm = _near([m for m in r.cavity_modes() if m["family"] == "TM"], REF["TM010"][1])
        te = _near([m for m in r.cavity_modes() if m["family"] == "TE"], REF["TE011"][1])
        P = r.cavity_mode_probe(tm, (0.0, 0.995 * A, 41), (-0.098, 0.098, 21))
        k = CHI0[0] / A
        row = {"h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt,
               "TM010_Ez": float(np.max(np.abs(P["E_z"] - E10 * j0(k * P["R"]))) / E10),
               "TM010_Er": float(np.max(np.abs(P["E_r"])) / E10),
               "TM010_Hphi": float(np.max(np.abs(P["H_phi"] - E10 / Z0 * j1(k * P["R"]))) / (E10 / Z0))}
        P = r.cavity_mode_probe(te, (0.0, 0.995 * A, 41), (-0.098, 0.098, 21))
        kc, beta, om = CHI1P / A, math.pi / L, 2 * math.pi * te["f_Hz"]
        sn, cs = np.sin(beta * (P["Z"] + L / 2)), np.cos(beta * (P["Z"] + L / 2))
        Hs = E11 / (om * cv.MU0)
        row["TE011_Ephi"] = float(np.max(np.abs(P["E_phi"] - E11 * j1(kc * P["R"]) * sn)) / E11)
        row["TE011_H"] = float(max(np.max(np.abs(P["H_r"] + Hs * beta * j1(kc * P["R"]) * cs)),
                                   np.max(np.abs(P["H_z"] - Hs * kc * j0(kc * P["R"]) * sn))) / (Hs * kc))
        for name, m in (("TM010", tm), ("TE011", te)):
            S = r.cavity_mode_sample(m, R, Z)
            e2 = S["E_r"] ** 2 + S["E_phi"] ** 2 + S["E_z"] ** 2
            h2 = S["H_r"] ** 2 + S["H_phi"] ** 2 + S["H_z"] ** 2
            row[name + "_norm"] = float(np.sum(W * e2) - 1)
            row[name + "_energy"] = float(np.sum(W * h2) * Z0 ** 2 - 1)
        rows.append(row)
    return {"case": "pillbox a = 0.1 m, L = 0.2 m, order 2; exported fields vs closed forms", "rows": rows}


def _fmt(x):
    return f"{x:.2e}"


def page():
    rec = {k: json.loads((RECS / v).read_text()) for k, v in FILES.items() if (RECS / v).exists()}
    s = ["# Cavity haloscopes, axisymmetric — validation", "",
         "**Beta.** Generated by `benchmarks/cavity_2d_validation.py` from `benchmarks/records/cavity_mc*.json`; references",
         "are closed forms (pillbox and coaxial modes, form factors and geometry factors of `axionfem.cavity`, derived",
         "independently of the solver; tests `tests/test_cavity.py`). Order-2",
         "elements unless stated; `h` is the cavity element size (closed sphere R = 1 m, so the mesher's far-field",
         "cap pi R/96 does not interfere). Design section: docs/design_format.md, \"Cavity haloscopes\".", ""]
    if "mc1" in rec:
        s += ["## 1. Cavity domain, frequencies (pillbox a = 0.1 m, L = 0.2 m)", "",
              "Relative frequency error. All energy lies in the cavity region (`energy_fraction`), the room outside",
              "the wall is one excluded component.", "",
              "| order | h [m] | DOFs | TM010 | TM011 | TE011 |", "|---|---|---|---|---|---|"]
        for r in rec["mc1"]["rows"]:
            s.append(f"| {r['order']} | {r['h_m']} | {r['dofs']} | {_fmt(r['TM010']['rel'])} | {_fmt(r['TM011']['rel'])} | "
                     f"{_fmt(r['TE011']['rel'])} |")
        s += ["", "Order 2 converges as h^4 (eigenvalues); 1e-5 is reached at h = a/5 = 2 cm for TM010/TM011 and at",
              "h = 1 cm for TE011.", ""]
    if "mc2" in rec:
        s += ["## 2. Targeted shift-invert", "",
              f"`fullwave.target_f_Hz` = 2.5 f(TM010) = {rec['mc2']['target_f_Hz'] / 1e9:.4f} GHz, 12 modes per family.",
              "Every analytic TM0np/TE0np mode within +-20 % of the target is returned (the nearest in omega^2).", "",
              "| mode | f exact [GHz] | rel (h = 5 mm) | rel (h = 2.5 mm) |", "|---|---|---|---|"]
        r0, r1 = rec["mc2"]["runs"]
        for a, b in zip(r0["modes"], r1["modes"]):
            s.append(f"| {a['mode']} | {a['f_exact_Hz'] / 1e9:.6f} | {_fmt(a['rel'])} | {_fmt(b['rel'])} |")
        s += ["", f"Factorization: {r1['solver']['TM'].get('factorization')} (probe-checked); Lanczos steps "
                  f"{r1['solver']['TM'].get('lanczos_steps')}.", ""]
    if "mc3" in rec:
        s += ["## 3. Form factor (uniform axial B0)", "",
              "| h [m] | C010 rel | C020 rel | C030 rel | max C(TM0np, p>0)/C010 | max C(TE) |",
              "|---|---|---|---|---|---|"]
        for r in rec["mc3"]["pillbox"]:
            s.append(f"| {r['h_m']} | {_fmt(r['C010']['rel'])} | {_fmt(r['C020']['rel'])} | {_fmt(r['C030']['rel'])} | "
                     f"{_fmt(r['max_C_TM0np_p_gt_0_over_C010'])} | {r['max_C_TE']} |")
        ce = rec["mc3"]["coax_exact"]
        s += ["", f"Coaxial cavity b = 0.2 m, a = 0.05 m: exact kb = {ce['kb']:.10f}, f = {ce['f_Hz'] / 1e6:.6f} MHz, "
                  f"C = {ce['C']:.10f} (vacuum volume).", "",
              "| h [m] | f rel | C rel |", "|---|---|---|"]
        for r in rec["mc3"]["coax"]:
            s.append(f"| {r['h_m']} | {_fmt(r['rel_f'])} | {_fmt(r['rel_C'])} |")
        se = rec["mc3"]["sleeve_exact"]
        s += ["", "Dielectric sleeve (eps_r = 4, 0.03 < r < 0.05 m, full length) in the pillbox: 1D Bessel matching",
              f"(`axionfem.cavity.sleeve_tm_mode`) f = {se['f_Hz'] / 1e9:.9f} GHz, C = {se['C']:.10f} (eps-weighted).", "",
              "| h [m] | f rel | C rel |", "|---|---|---|"]
        for r in rec["mc3"]["sleeve"]:
            s.append(f"| {r['h_m']} | {_fmt(r['rel_f'])} | {_fmt(r['rel_C'])} |")
        s.append("")
    if "mc4" in rec:
        s += ["## 4. Geometry factor G = Q0 R_s and dielectric loss", "",
              "G(TM010) = Z0 chi01 L/(2(a + L)); G(TM011) = Z0 k a L/(2(L + 2a)); G(TE011) from",
              "Pozar, Microwave Engineering, Sec. 6.4 (the TE_nml wall Q), with n = 0. TE: wall current from the consistent-flux recovery; the pointwise derivative",
              "(`G_ohm_pointwise`) is shown for comparison.", "",
              "| h [m] | TM010 | TM011 | TE011 | TE011 pointwise |", "|---|---|---|---|---|"]
        for r in rec["mc4"]["rows"]:
            s.append(f"| {r['h_m']} | {_fmt(r['TM010']['rel'])} | {_fmt(r['TM011']['rel'])} | {_fmt(r['TE011']['rel'])} | "
                     f"{_fmt(r['TE011']['rel_pointwise'])} |")
        dz = rec["mc4"]["dielectric_fill"]
        s += ["", f"Fully filled dielectric (eps_r = 2, tan delta = 1e-4, h = 5 mm): f/f_vac*sqrt(2) - 1 = "
                  f"{_fmt(dz['f_rel'])}, Q_d tan delta - 1 = {_fmt(dz['Q_d_rel'])}, C eps_r/C010 - 1 = "
                  f"{_fmt(dz['C_rel_vs_C010_over_eps'])} (eps-weighted C).", ""]
    if "mf" in rec:
        s += ["## Per-mode field export (Results.cavity_mode_field)", "",
              "Exported fields of the pillbox (normalized to int eps_r |E|^2 dV = 1) against the closed forms",
              "TM010: E_z = E0 J0(chi01 r/a), H_phi = (E0/Z0) J1(chi01 r/a); TE011: E_phi = E0 J1(kc r) sin(pi (z + L/2)/L)",
              "and its H_r, H_z. Max pointwise error on a 41 x 21 grid (r <= 0.995 a, |z| <= 0.098 m) relative to the",
              "field scale (E0, E0/Z0; H of TE011: E0 kc/(omega mu0)). E of TM and H of TE are derivatives of the order-2",
              "unknown and converge as h^2; H_phi of TM and E_phi of TE (the unknown itself) as h^3. `norm`, `energy`:",
              "int eps_r |E|^2 dV - 1 and Z0^2 int mu_r |H|^2 dV - 1 by 48 x 48 Gauss quadrature of the sampled fields",
              "(the second checks that E, H and f belong to the same eigenpair). Both identities hold exactly with",
              "element quadrature (the Rayleigh quotient of the discrete eigenpair; element-wise quadrature of the sampled",
              "fields reproduces both to < 1e-9, tests/test_cavity_modefields.py); the residuals here are those of",
              "the tensor Gauss rule on fields that are only piecewise smooth (kinks at element edges).", "",
              "| h [m] | TM010 E_z | TM010 E_r | TM010 H_phi | TE011 E_phi | TE011 H | TM010 norm | TM010 energy | "
              "TE011 norm | TE011 energy |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rec["mf"]["rows"]:
            s.append(f"| {r['h_m']} | {_fmt(r['TM010_Ez'])} | {_fmt(r['TM010_Er'])} | {_fmt(r['TM010_Hphi'])} | "
                     f"{_fmt(r['TE011_Ephi'])} | {_fmt(r['TE011_H'])} | {_fmt(r['TM010_norm'])} | "
                     f"{_fmt(r['TM010_energy'])} | {_fmt(r['TE011_norm'])} | {_fmt(r['TE011_energy'])} |")
        s += ["", "![Cavity eigenmode gallery](../images/cavity_modes.png)", "",
              "*|E| of pillbox TM010, TM020, TE011 and of the coaxial (centred rod) TM010-like mode from",
              "`Results.plot_mode` (C++ renderer); f, C and Q0 (300 K copper) from the solve. Regenerate with",
              "[docs/images/make_cavity_figure.py](../images/make_cavity_figure.py).*", ""]
    s += ["## Limitations", "",
          "- m = 0 modes only (TM0np, TE0np); hybrid and m != 0 modes need the 3D path ([cavity_3d.md](cavity_3d.md))."]
    if "mc4" in rec:
        s += ["- Losses are perturbative (lossless mode, Q >> 1); these cases use the `skin`, `power` and `table` R_s models",
              "  (all models: [surface_resistance.md](../physics/surface_resistance.md)).",
              "- Sharp re-entrant metal edges carry integrable field singularities; the wall-loss integral converges more",
              "  slowly there (not exercised by these cases)."]
    s.append("")
    OUT.write_text("\n".join(s))


def main():
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path("cavity_2d_runs")
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    for key, fn in (("mc1", mc1), ("mc2", mc2), ("mc3", mc3), ("mc4", mc4), ("mf", mf)):
        if only and key not in only:
            continue
        t0 = time.time()
        res = fn(wd)
        res["seconds"] = time.time() - t0
        res["status"] = "beta"
        (RECS / FILES[key]).write_text(json.dumps(res, indent=1))
        print(key, f"{res['seconds']:.1f} s")
    page()


if __name__ == "__main__":
    main()
