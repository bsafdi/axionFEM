#!/usr/bin/env python
"""COMSOL vs axionFEM cross-check on the arXiv:2211.00008 fiducial toroid (validation plot).

Designs (this directory): ``comsol_toroid_lossless.json`` and ``comsol_toroid_lossy.json``. The toroid has
r = 0.5-1 m, h = 0.5 m and 2.5 cm PEC walls. A planar 2 cm slit is read out through a small box port at the
bottom of the outer wall, and the whole is inside a closed PEC sphere of radius 1.5 m. The axion source is
B_phi = 1 T filling the cavity (``fullwave.source`` kind ``toroidal_box``). The lossy variant adds a dielectric
support ring (eps_r 3, tan_delta 2e-3).

Subcommands (run from the repository root, ``PYTHONPATH=python``):

    check_toroid.py export OUTDIR [--dense]   # COMSOL Java models (+ a dense-frequency variant); needs only Python
    check_toroid.py solve RUNDIR [--ranks N]  # axionFEM solves at two mesh levels -> axionfem_reference.json
                                              # (needs the C++ build; run as a batch job)
    check_toroid.py merge ref_*.json          # per-job results -> axionfem_reference.json
    check_toroid.py compare COMSOL_DIR [-o validation_toroid.pdf]   # the cross-check: tables, criteria, plot

Conventions (docs/comsol_export.md section 4): axionFEM uses exp(-i w t), COMSOL exp(+j w t), so COMSOL's
numbers are the complex conjugates of axionFEM's: Zport_1 = conj(Z) = R + j w L. axionFEM's open-circuit voltage
is V = -i w Psi (kappa = 1 A m^-2 T^-1 = Je_scale). COMSOL's 'volt' run loads the port with Z_ref = 50 ohm;
``compare`` converts it to the open-circuit (Thevenin) voltage with the 'imp' run's Zport_1. The port polarity
is arbitrary in COMSOL, so voltages are compared in magnitude.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = ("lossless", "lossy")
CLASS = {"lossless": "ComsolToroidLossless", "lossy": "ComsolToroidLossy"}
Z_REF = 50.0
DENSE_MHZ = [round(0.5 + 0.25 * k, 4) for k in range(238)]          # 0.5 ... 59.75 MHz
# axionFEM mesh levels: "A" is the committed design; "B" refines it to estimate axionFEM's discretization error
LEVELS = {"A": {}, "B": {"size": 0.08, "surface_size": 0.02, "port_surface_size": 0.0035}}

# Acceptance criteria (COMSOL at its finest mesh vs axionFEM level B); see README.md
CRIT = {"L_lowf_rel": 0.02,        # |L_C/L_A - 1| at f <= 5 MHz
        "Voc_lowf_rel": 0.03,      # | |V_C|/|V_A| - 1 | at f <= 5 MHz
        "X_rel": 0.05,             # Im Z away from poles (|f - f_pole| > 10 % f): relative
        "Voc_rel": 0.05,           # |V_oc| away from poles
        "R_rel": 0.15,             # Re Z (lossy case), away from poles, where R > R_min_over_absX |X|
        "R_min_over_absX": 1e-6,   # (the support ring gives R/|X| ~ 1e-5 to 1e-3: a 1e-3 cut would test nothing)
        "f_srf_rel": 0.02}         # first self-resonance (sign change of Im Z) location


def design(case: str, level: str = "A") -> dict:
    d = json.loads((HERE / f"comsol_toroid_{case}.json").read_text())
    lv = LEVELS[level]
    if lv:
        d["mesh"]["size"], d["mesh"]["surface_size"] = lv["size"], lv["surface_size"]
        for r in d["regions"]:
            if r["name"] == "port":
                r["surface_size"] = lv["port_surface_size"]
    return d


# --------------------------------------------------------------------------------------------- export
def export(out: Path, dense: bool) -> dict:
    from axionfem import comsol

    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for case in CASES:
        d = design(case)
        res[case] = comsol.export_comsol(d, out / f"{CLASS[case]}.java", format="java", physics="fullwave")
        if dense:
            dd = json.loads(json.dumps(d))
            dd["name"] += "_dense"
            dd["fullwave"]["frequencies"] = [f * 1e6 for f in DENSE_MHZ]
            res[case + "_dense"] = comsol.export_comsol(dd, out / f"{CLASS[case]}Dense.java", format="java",
                                                        physics="fullwave")
    return {k: {"file": str(v.get("file")), "warnings": v["physics"].get("warnings", [])} for k, v in res.items()}


# --------------------------------------------------------------------------------------------- solve
def _rows(r) -> list[dict]:
    rows = []
    for s in r.data["fullwave"]["sweep"]:
        f = s["f_Hz"]
        w = 2 * math.pi * f
        Z = complex(s["Z_matrix_re_ohm"][0][0], s["Z_matrix_im_ohm"][0][0])
        Psi = complex(s["Psi_vector"]["re"][0], s["Psi_vector"]["im"][0])
        V = -1j * w * Psi
        row = {"f_MHz": f / 1e6, "Z_ohm": [Z.real, Z.imag], "L_H": s.get("L_H"), "Psi": [Psi.real, Psi.imag],
               "V_open_V": [V.real, V.imag],
               "COMSOL_expected": {"Zport_1": [Z.real, -Z.imag], "abs_V_open_V": abs(V)},
               "residual": (s.get("solver") or {}).get("relative_residual")}
        if "direct_source_voltage" in s:
            row["V_open_direct_solve_V"] = [s["direct_source_voltage"]["re"][0], s["direct_source_voltage"]["im"][0]]
        rows.append(row)
    return rows


def solve(rundir: Path, ranks: int, levels: list[str], cases=CASES, freqs_MHz=None, ref_name="axionfem_reference.json") -> dict:
    import axionfem as af

    path = HERE / ref_name
    ref = json.loads(path.read_text()) if path.is_file() else {}
    ref.update({"Z_ref_ohm": Z_REF,
                "conventions": {"axionFEM": "exp(-i w t); Z = R - i w L; V_open = -i w Psi (kappa = Je_scale = 1)",
                                "COMSOL": "exp(+j w t): Zport_1 = conj(Z); |V_open| equal; volt run is loaded by Z_ref"}})
    ref.setdefault("cases", {})
    for case in cases:
        for level in levels:
            doc = design(case, level)
            if freqs_MHz:
                doc["fullwave"]["frequencies"] = [f * 1e6 for f in freqs_MHz]
            d = af.Design(doc)
            r = d.solve(rundir / f"{case}_{level}", verbose=0, mpi_ranks=ranks)
            mesh = {k: v for k, v in (r.data.get("mesh") or {}).items() if isinstance(v, (int, float, str))}
            ref["cases"].setdefault(case, {})[level] = {
                "mesh_settings": doc["mesh"], "frequencies_MHz": [f / 1e6 for f in doc["fullwave"]["frequencies"]], "mesh": mesh, "rows": _rows(r),
                "self_resonances": r.data["fullwave"].get("self_resonances"),
                "L_MQS_H": (r.data.get("pickup") or {}).get("L_p_H"), "warnings": r.data.get("warnings", [])}
            path.write_text(json.dumps(ref, indent=1) + "\n")
    return ref


def merge(parts: list[Path], out: Path | None = None) -> dict:
    """Combine per-job reference files (solve --ref-name ref_<case>_<level>[_<tag>].json) into
    axionfem_reference.json. Several files may hold the same case and level at different frequencies (one job per
    frequency): their rows are merged by frequency, and the self-resonance brackets are recomputed from L_H."""
    ref = {}
    for p in parts:
        r = json.loads(p.read_text())
        for k, v in r.items():
            if k != "cases":
                ref[k] = v
        for case, levels in r.get("cases", {}).items():
            for level, entry in levels.items():
                cur = ref.setdefault("cases", {}).setdefault(case, {}).get(level)
                if cur is None:
                    ref["cases"][case][level] = cur = json.loads(json.dumps(entry))
                    cur["source_files"] = [p.name]
                    continue
                size = {k: v for k, v in entry["mesh"].items() if "element" in k or "vert" in k or "dof" in k.lower()}
                if any(cur["mesh"].get(k) != v for k, v in size.items()):
                    cur.setdefault("warnings", []).append(f"merged rows from different meshes: {p.name} has {size}")
                rows = {round(x["f_MHz"], 9): x for x in cur["rows"]}
                for x in entry["rows"]:
                    if round(x["f_MHz"], 9) in rows:
                        raise SystemExit(f"{p}: {case}/{level} at {x['f_MHz']} MHz is already merged")
                    rows[round(x["f_MHz"], 9)] = x
                cur["rows"] = [rows[k] for k in sorted(rows)]
                cur["frequencies_MHz"] = sorted(set(cur["frequencies_MHz"]) | set(entry["frequencies_MHz"]))
                cur["warnings"] = list(dict.fromkeys(cur.get("warnings", []) + entry.get("warnings", [])))
                cur["source_files"].append(p.name)
    for levels in ref.get("cases", {}).values():
        for e in levels.values():
            rs = e["rows"]
            e["self_resonances"] = [{"between_Hz": [a["f_MHz"] * 1e6, b["f_MHz"] * 1e6]} for a, b in zip(rs, rs[1:])
                                    if a.get("L_H") is not None and b.get("L_H") is not None
                                    and a["L_H"] > 0 >= b["L_H"]]
    (out or HERE / "axionfem_reference.json").write_text(json.dumps(ref, indent=1) + "\n")
    return {c: {lv: [r["f_MHz"] for r in e["rows"]] for lv, e in v.items()} for c, v in ref.get("cases", {}).items()}


# --------------------------------------------------------------------------------------------- compare
def _comsol(comsol_dir: Path, case: str):
    """COMSOL's tables for one case (the reference-frequency run and, if present, the dense one), converted to
    axionFEM's convention: Z = conj(Zport_1), |V_open|."""
    try:
        from axionfem.comsol_physics import open_circuit_voltage, read_eval_table
    except ImportError:  # no axionFEM C++ build (e.g. on the COMSOL machine): load the pure-numpy module directly
        import importlib.util

        spec = importlib.util.spec_from_file_location("comsol_physics", ROOT / "python/axionfem/comsol_physics.py")
        cph = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cph)
        open_circuit_voltage, read_eval_table = cph.open_circuit_voltage, cph.read_eval_table

    out = {}
    for tag, stem in (("ref", CLASS[case]), ("dense", CLASS[case] + "Dense")):
        zi, vi = comsol_dir / f"{stem}_impedance.txt", comsol_dir / f"{stem}_voltage.txt"
        if not zi.is_file():
            continue
        Zt = read_eval_table(zi)
        Zc = Zt[1] + 1j * Zt[2]
        d = {"f_MHz": Zt[0], "Z": np.conj(Zc)}
        if vi.is_file():
            Vt = read_eval_table(vi)
            if Vt.shape != Zt.shape or not np.allclose(Vt[0], Zt[0]):
                raise SystemExit(f"{vi}: frequencies differ from {zi}")
            d["Voc"] = np.abs(open_circuit_voltage(Vt[1] + 1j * Vt[2], Zc, Z_REF))
        out[tag] = d
    return out


def _first_srf(f, X):
    """First sign change of X = Im Z from - to + (axionFEM convention X = -w L < 0 below it). Near a pole (parallel
    resonance, |X| growing into the crossing) 1/X is interpolated linearly, which is smooth through the pole; near a
    zero (series resonance) X itself is."""
    f, X = np.asarray(f, float), np.asarray(X, float)
    for k in range(len(f) - 1):
        if X[k] < 0 <= X[k + 1]:
            if X[k + 1] == 0:
                return float(f[k + 1])
            pole = k == 0 or max(abs(X[k]), abs(X[k + 1])) > abs(X[k - 1])
            y0, y1 = (1 / X[k], 1 / X[k + 1]) if pole else (X[k], X[k + 1])
            return float(f[k] + (f[k + 1] - f[k]) * (0 - y0) / (y1 - y0))
    return None


def _crossings(f, X) -> list[float]:
    """Every sign change of X = Im Z (poles and zeros, either direction), located as in _first_srf."""
    f, X = np.asarray(f, float), np.asarray(X, float)
    out = []
    for k in range(len(f) - 1):
        if X[k] == 0:
            out.append(float(f[k]))
        elif X[k] * X[k + 1] < 0:
            pole = k == 0 or max(abs(X[k]), abs(X[k + 1])) > abs(X[k - 1])
            y0, y1 = (1 / X[k], 1 / X[k + 1]) if pole else (X[k], X[k + 1])
            out.append(float(f[k] + (f[k + 1] - f[k]) * (0 - y0) / (y1 - y0)))
    return out


def _match(f_from, f_to, rtol=1e-6) -> np.ndarray:
    """For each frequency of f_to, its index in f_from (-1 if absent)."""
    f_from = np.asarray(f_from, float)
    out = []
    for f in f_to:
        k = np.flatnonzero(np.abs(f_from - f) <= rtol * max(abs(f), 1.0))
        out.append(int(k[0]) if len(k) else -1)
    return np.array(out, int)


def _rel_panel(a, curves, label):
    """Relative-difference panel: y range +-0.1, widened up to +-1 to show larger misfits (clipped beyond)."""
    vals = np.concatenate([np.asarray(c, float) for c in curves]) if curves else np.array([])
    vals = np.abs(vals[np.isfinite(vals)])
    lim = 0.1 if not vals.size else min(1.0, max(0.1, 1.15 * vals.max()))
    a.set_ylim(-lim, lim)
    a.axhline(0, color="0.6", lw=0.6)
    a.set_ylabel(label)
    if vals.size and vals.max() > 1:
        a.text(0.98, 0.05, "clipped at $\\pm$1", transform=a.transAxes, ha="right", fontsize=7, color="maroon")


def compare(comsol_dir: Path, pdf: Path, ref_path: Path | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = json.loads((ref_path or HERE / "axionfem_reference.json").read_text())
    summary = {"criteria": CRIT, "cases": {}}
    fig, ax = plt.subplots(4, 2, figsize=(11, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 3, 1.3]})
    for j, case in enumerate(CASES):
        a0, a1, a2, a3 = ax[:, j]
        res = {"points": [], "pass": True, "notes": []}
        summary["cases"][case] = res
        A = ref.get("cases", {}).get(case)
        if not A:
            res["pass"] = False
            res["notes"].append("no axionFEM reference for this case")
            a0.text(0.5, 0.5, f"{case}: no axionFEM reference", transform=a0.transAxes, ha="center")
            for a in (a0, a1, a2, a3):
                a.set_xticks([]); a.set_yticks([])
            continue
        lvl = "B" if "B" in A else "A"
        res["axionFEM_level"] = lvl
        best = A[lvl]
        fa = np.array([r["f_MHz"] for r in best["rows"]], float)
        Za = np.array([complex(*r["Z_ohm"]) for r in best["rows"]])
        Va = np.array([math.hypot(*r["V_open_V"]) for r in best["rows"]])
        # axionFEM coarse (A) vs fine (B), matched by frequency: axionFEM's own discretization error
        Zco = Vco = None
        if lvl == "B" and "A" in A:
            fco = [r["f_MHz"] for r in A["A"]["rows"]]
            ic = _match(fco, fa)
            Zco = np.array([complex(*A["A"]["rows"][i]["Z_ohm"]) if i >= 0 else complex(np.nan, np.nan) for i in ic])
            Vco = np.array([math.hypot(*A["A"]["rows"][i]["V_open_V"]) if i >= 0 else np.nan for i in ic])
        C = _comsol(comsol_dir, case)
        # --- curves
        if "dense" in C:
            dn = C["dense"]
            a0.plot(dn["f_MHz"], np.abs(dn["Z"].imag), "-", color="0.55", lw=1, label="COMSOL |Im Z| (dense)")
            if case == "lossy":
                a0.plot(dn["f_MHz"], np.abs(dn["Z"].real), ":", color="0.55", lw=1, label="COMSOL Re Z (dense)")
            if "Voc" in dn:
                a2.plot(dn["f_MHz"], dn["Voc"], "-", color="0.55", lw=1, label="COMSOL |V_oc| (dense)")
        a0.plot(fa, np.abs(Za.imag), "o", color="k", ms=5, label=f"axionFEM |Im Z| (level {lvl})")
        if case == "lossy":
            a0.plot(fa, np.abs(Za.real), "s", color="k", ms=4, mfc="none", label=f"axionFEM Re Z (level {lvl})")
        a2.plot(fa, Va, "o", color="k", ms=5, label=f"axionFEM |V_oc| (level {lvl})")
        srf_a = _first_srf(fa, Za.imag)
        res["f_srf_MHz"] = {"axionFEM": srf_a}
        rel_X, rel_V = [], []
        if "ref" in C:
            rc = C["ref"]
            a0.plot(rc["f_MHz"], np.abs(rc["Z"].imag), "x", color="maroon", ms=7, label="COMSOL |Im Z|")
            if case == "lossy":
                a0.plot(rc["f_MHz"], np.abs(rc["Z"].real), "+", color="maroon", ms=7, label="COMSOL Re Z")
            if "Voc" in rc:
                a2.plot(rc["f_MHz"], rc["Voc"], "x", color="maroon", ms=7, label="COMSOL |V_oc|")
            else:
                res["pass"] = False
                res["notes"].append("no COMSOL voltage table for the reference run")
            idx = _match(rc["f_MHz"], fa)
            if (idx < 0).any():
                res["pass"] = False
                res["notes"].append(f"COMSOL reference run lacks the axionFEM frequencies {fa[idx < 0].tolist()} MHz")
            m = idx >= 0
            fm, Zam, Vam = fa[m], Za[m], Va[m]
            Zc = rc["Z"][idx[m]]
            Vc = rc["Voc"][idx[m]] if "Voc" in rc else np.full(m.sum(), np.nan)
            if Zc.size and Zc[0].imag > 0:
                res["notes"].append("COMSOL Im(Zport_1) < 0 at the lowest frequency (capacitive): check the time/sign "
                                    "convention and the port before anything else")
            # SRFs: the criterion compares like with like (both from the reference frequencies, same interpolation);
            # the dense COMSOL sweep, if present, is reported as the more precise COMSOL value
            srf_c = _first_srf(fm, Zc.imag)
            srf_cd = _first_srf(C["dense"]["f_MHz"], C["dense"]["Z"].imag) if "dense" in C else None
            res["f_srf_MHz"].update({"COMSOL": srf_c, "COMSOL_dense": srf_cd})
            if srf_a and srf_c:
                res["f_srf_MHz"]["rel"] = srf_c / srf_a - 1
                if abs(srf_c / srf_a - 1) > CRIT["f_srf_rel"]:
                    res["pass"] = False
                    res["notes"].append("SRF differs by more than the criterion")
            elif bool(srf_a) != bool(srf_c):
                res["pass"] = False
                res["notes"].append("a self-resonance is found in one code's sweep but not in the other's")
            if srf_a and srf_cd:
                res["f_srf_MHz"]["rel_dense"] = srf_cd / srf_a - 1
            # every pole/zero crossing of Im Z (either code), not only the first SRF: pointwise ratios are
            # meaningless within 10 % of any of them
            cross = sorted(set(_crossings(fa, Za.imag) + _crossings(fm, Zc.imag)
                               + (_crossings(C["dense"]["f_MHz"], C["dense"]["Z"].imag) if "dense" in C else [])))
            res["crossings_MHz"] = cross
            pole = np.zeros(len(fm), bool)
            for fp in cross:
                pole |= np.abs(fm - fp) <= 0.10 * fm
            rX = Zc.imag / Zam.imag - 1
            rV = Vc / Vam - 1
            for k, f in enumerate(fm):
                p = {"f_MHz": f, "dX_rel": rX[k], "dVoc_rel": rV[k], "near_pole": bool(pole[k])}
                lowf = f <= 5
                tol_X = CRIT["L_lowf_rel"] if lowf else CRIT["X_rel"]
                tol_V = CRIT["Voc_lowf_rel"] if lowf else CRIT["Voc_rel"]
                p["X_ok"] = bool(pole[k] or abs(rX[k]) <= tol_X)
                p["V_ok"] = bool(pole[k] or (np.isfinite(rV[k]) and abs(rV[k]) <= tol_V))
                if case == "lossy" and not pole[k] and Zam[k].real > CRIT["R_min_over_absX"] * abs(Zam[k].imag):
                    p["dR_rel"] = Zc[k].real / Zam[k].real - 1
                    p["R_ok"] = bool(abs(p["dR_rel"]) <= CRIT["R_rel"])
                res["points"].append(p)
                res["pass"] &= p["X_ok"] and p["V_ok"] and p.get("R_ok", True)
            # relative differences, points within 10 % of a crossing left out (pointwise ratios meaningless there)
            hide = lambda r: np.where(pole, np.nan, r)  # noqa: E731
            rel_X.append(hide(rX)); rel_V.append(hide(rV))
            a1.plot(fm, hide(rX), "x-", color="maroon", label="COMSOL / axionFEM − 1: Im Z")
            if case == "lossy":
                rR = np.array([p.get("dR_rel", np.nan) for p in res["points"]])
                rel_X.append(rR)
                a1.plot(fm, rR, "+:", color="maroon", ms=7, label="COMSOL / axionFEM − 1: Re Z")
            a3.plot(fm, hide(rV), "x-", color="maroon", label="COMSOL / axionFEM − 1")
            for fp in cross:
                for a in (a0, a1, a2, a3):
                    a.axvline(fp, color="0.85", lw=0.7, ls="--", zorder=0)
            for fp in (x for x in (srf_a, srf_c, srf_cd) if x):
                for a in (a0, a1, a2, a3):
                    a.axvline(fp, color="0.7", lw=0.8, zorder=0)
        else:
            res["pass"] = False
            res["notes"].append("no COMSOL reference table found")
        if Zco is not None:
            far = np.array([not any(abs(f - fp) <= 0.10 * f for fp in _crossings(fa, Za.imag)) for f in fa])
            rXc, rVc = np.where(far, Zco.imag / Za.imag - 1, np.nan), np.where(far, Vco / Va - 1, np.nan)
            rel_X.append(rXc); rel_V.append(rVc)
            a1.plot(fa, rXc, "o--", color="0.5", ms=3, label="axionFEM level A / B − 1")
            a3.plot(fa, rVc, "o--", color="0.5", ms=3, label="axionFEM level A / B − 1")
        _rel_panel(a1, rel_X, r"rel. diff. $Z$")
        _rel_panel(a3, rel_V, r"rel. diff. $|V_{\rm oc}|$")
        # acceptance bands (f <= 5 MHz: L 2 %, |V| 3 %; above: 5 %)
        f_hi = max(fa.max(), C["dense"]["f_MHz"].max() if "dense" in C else 0)
        for a, lo, hi in ((a1, CRIT["L_lowf_rel"], CRIT["X_rel"]), (a3, CRIT["Voc_lowf_rel"], CRIT["Voc_rel"])):
            for sgn in (-1, 1):
                a.hlines(sgn * lo, 0, 5, color="tab:green", lw=0.8, ls=":")
                a.hlines(sgn * hi, 5, f_hi, color="tab:green", lw=0.8, ls=":")
        a0.set_yscale("log"); a2.set_yscale("log")
        a0.set_ylabel(r"$|{\rm Im}\,Z|$, Re $Z$ [$\Omega$]")
        a2.set_ylabel(r"$|V_{\rm oc}|$ [V]  ($B_0$ = 1 T, $\kappa$ = 1)")
        for a in (a0, a1, a2, a3):
            a.legend(fontsize=7, frameon=False, loc="best")
        verdict = "PASS" if res["pass"] else "FAIL"
        a0.set_title(f"{case}: {verdict}" + (f"  (SRF axionFEM {srf_a:.2f} MHz)" if srf_a else ""), fontsize=10,
                     color="k" if res["pass"] else "maroon")
        a3.set_xlabel("frequency [MHz]")
    fig.text(0.5, 0.995, "COMSOL vs axionFEM, arXiv:2211.00008 fiducial toroid (axionFEM convention; vertical lines: "
             "first SRF (solid), other Im Z sign changes (dashed); rel. diffs omit points within 10 % of any of them)", ha="center", va="top", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(pdf)
    summary["pass"] = all(c["pass"] for c in summary["cases"].values())
    (pdf.with_suffix(".json")).write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("out", type=Path); e.add_argument("--dense", action="store_true")
    s = sub.add_parser("solve"); s.add_argument("rundir", type=Path); s.add_argument("--ranks", type=int, default=1)
    s.add_argument("--levels", default="A,B")
    s.add_argument("--cases", default=",".join(CASES))
    s.add_argument("--freqs-MHz", default="", help="override the frequencies (pilot runs)")
    s.add_argument("--ref-name", default="axionfem_reference.json")
    m = sub.add_parser("merge"); m.add_argument("parts", type=Path, nargs="+")
    m.add_argument("-o", "--out", type=Path, default=None, help="default: axionfem_reference.json here")
    c = sub.add_parser("compare"); c.add_argument("comsol_dir", type=Path)
    c.add_argument("-o", "--out", type=Path, default=Path("validation_toroid.pdf"))
    c.add_argument("--ref", type=Path, default=None, help="axionFEM reference JSON (default: axionfem_reference.json here)")
    a = ap.parse_args()
    if a.cmd == "export":
        print(json.dumps(export(a.out, a.dense), indent=1))
    elif a.cmd == "solve":
        fr = [float(x) for x in a.freqs_MHz.split(",")] if a.freqs_MHz else None
        solve(a.rundir, a.ranks, a.levels.split(","), a.cases.split(","), fr, a.ref_name)
    elif a.cmd == "merge":
        print(json.dumps(merge(a.parts, a.out), indent=1))
    else:
        print(json.dumps(compare(a.comsol_dir, a.out, a.ref), indent=1, default=float))


if __name__ == "__main__":
    main()
