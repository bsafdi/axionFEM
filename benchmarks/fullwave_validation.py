"""Validation of the axisymmetric full-wave solver (beyond MQS) against exact references.

    python benchmarks/fullwave_validation.py [workdir]

Writes benchmarks/records/fullwave_validation.json and docs/validation/fullwave.md.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "benchmarks" / "records" / "fullwave_validation.json"
OUT = ROOT / "docs" / "validation" / "fullwave.md"
C = 299792458.0
ETA = 376.730313668


def sphere(wd: Path):
    doc = {"schema": "axionfem.design/1", "name": "sphere",
           "regions": [{"name": "box", "shape": {"rect": {"r": [0.0, 0.6], "z": [-0.6, 0.6]}}, "material": "vacuum",
                        "mesh_size": 0.04},
                       {"name": "coil", "shape": {"rect": {"r": [0.62, 0.63], "z": [-0.01, 0.01]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1}],
           "fullwave": {"modes": 5, "families": ["TE", "TM"]},
           "mesh": {"order": 2, "size": 0.04, "surface_size": 0.04, "domain_radius": 1.0}}
    r = af.Design(doc).solve(wd / "sphere", verbose=0)
    exact = {"TE": [("j_1", 4.493409458), ("j_2", 5.763459197), ("j_3", 6.987932000), ("j_1 (2nd)", 7.725251837),
                    ("j_4", 8.182561453)],
             "TM": [("n=1", 2.743706800), ("n=2", 3.870238669), ("n=3", 4.973420194), ("n=4", 6.061998019),
                    ("n=1 (2nd)", 6.116763380)]}
    rows = []
    for fam, ref in exact.items():
        got = [m["f_Hz"] * 2 * math.pi / C for m in r.data["fullwave"]["eigenmodes"][fam]["modes"]]
        for (lab, x), g in zip(ref, got):
            rows.append({"family": fam, "mode": lab, "kR_exact": x, "kR": g, "rel": g / x - 1})
    return rows


def loop(wd: Path, b: float):
    from scipy.integrate import quad
    from scipy.special import jv
    a = 0.5
    kas = [0.05, 0.2, 0.5, 1.0, 1.5, 2.0]
    doc = {"schema": "axionfem.design/1", "name": "loop",
           "regions": [{"name": "ring", "shape": {"disk": {"center": [a, 0.0], "radius": b}}, "material": "superconductor"},
                       {"name": "coil", "shape": {"rect": {"r": [0.1, 0.11], "z": [-0.01, 0.01]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1}],
           "superconductors": [{"body": "ring", "azimuthal_slit": True}],
           "pickup": {"type": "azimuthal", "conductor": "ring"},
           "fullwave": {"frequencies": [ka * C / (2 * math.pi * a) for ka in kas]},
           "mesh": {"order": 2, "size": 0.05, "surface_size": b / 2.5, "domain_radius": 1.0}}
    r = af.Design(doc).solve(wd / f"loop_{b}", verbose=0)
    rows = []
    for ka, row in zip(kas, r.data["fullwave"]["sweep"]):
        Rex = ETA * math.pi / 2 * ka * quad(lambda x: jv(2, x), 0, 2 * ka)[0]
        rows.append({"b_over_a": b / a, "ka": ka, "R": row["R_rad_ohm"], "R_exact": Rex, "rel": row["R_rad_ohm"] / Rex - 1,
                     "L_over_MQS": row["L_over_MQS"], "balance": row["power_balance_rel"]})
    L0 = r.coupling["L_p_H"]
    Lth = 4e-7 * math.pi * a * (math.log(8 * a / b) - 2)
    return rows, {"L_MQS": L0, "L_thin_ring": Lth, "rel": L0 / Lth - 1}


def coax(wd: Path):
    a1, b1, t, g, l1, l2 = 0.05, 0.15, 0.02, 0.005, 0.6, 0.3
    freqs = [1e6, 5e6, 20e6, 50e6, 100e6, 150e6]
    body = {"union": [{"rect": {"r": [0.0, a1], "z": [-l2 - t, g + l1 + t]}},
                      {"rect": {"r": [b1, b1 + t], "z": [-l2 - t, g + l1 + t]}},
                      {"rect": {"r": [a1, b1], "z": [g + l1, g + l1 + t]}},
                      {"rect": {"r": [a1, b1], "z": [-l2 - t, -l2]}}]}
    doc = {"schema": "axionfem.design/1", "name": "coax",
           "regions": [{"name": "coax", "shape": body, "material": "superconductor"},
                       {"name": "port", "shape": {"rect": {"r": [a1, b1], "z": [0.0, g]}}, "material": "vacuum"},
                       {"name": "coil", "shape": {"rect": {"r": [0.3, 0.32], "z": [-0.1, 0.1]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
           "pickup": {"type": "poloidal", "port": "port", "inside_point": [0.1, 0.3]},
           "fullwave": {"frequencies": freqs, "modes": 6, "families": ["TM"]},
           "mesh": {"order": 2, "size": 0.02, "surface_size": 0.005}}
    r = af.Design(doc).solve(wd / "coax", verbose=0)
    fw = r.data["fullwave"]
    Z0 = ETA / (2 * math.pi) * math.log(b1 / a1)
    L1, L2 = l1 + g / 2, l2 + g / 2
    rows = []
    for f, row in zip(freqs, fw["sweep"]):
        k = 2 * math.pi * f / C
        t1, t2 = math.tan(k * L1), math.tan(k * L2)
        tl = (t1 * t2 / (t1 + t2)) / (k * L1 * L2 / (L1 + L2))
        rows.append({"f_Hz": f, "L_over_MQS": row["L_over_MQS"], "TL": tl, "rel": row["L_over_MQS"] / tl - 1,
                     "R": row["R_rad_ohm"]})
    f_tem = C / (2 * (l1 + l2 + g))
    best = min(fw["eigenmodes"]["TM"]["modes"], key=lambda m: abs(m["f_Hz"] / f_tem - 1))
    L0 = r.coupling["L_p_H"]
    return rows, {"f_TEM_exact": f_tem, "f_TEM": best["f_Hz"], "rel": best["f_Hz"] / f_tem - 1,
                  "L_MQS": L0, "L_TL": Z0 / C * L1 * L2 / (L1 + L2)}


def benabou(wd: Path):
    """Fiducial toroid of Benabou et al. 2022 (arXiv:2211.00008): r_in 0.5 m, r_out 1 m, h 0.5 m, 2.5 cm sheath,
    unshielded. Their dipole estimate: Q_rad = (54/49) (k h)^-1 (k r_in)^-2."""
    d = af.template("toroid", R=1.0, y=0.5, H=0.5, tw=0.02, g=0.01, ts=0.025, f=0.03, NI=2.5e7,
                    mesh={"order": 2, "size": 0.05, "surface_size": 0.01})
    freqs = [1e6, 3e6, 10e6, 25e6, 50e6, 100e6]
    d.doc["fullwave"] = {"frequencies": freqs}
    r = d.solve(wd / "benabou", verbose=0)
    rows = []
    for row in r.data["fullwave"]["sweep"]:
        k = 2 * math.pi * row["f_Hz"] / C
        qd = 54 / 49 / (k * 0.5) / (k * 0.5) ** 2
        rows.append({"f_MHz": row["f_Hz"] / 1e6, "Q_rad": row.get("Q_rad"), "Q_dipole": qd,
                     "L_over_MQS": row.get("L_over_MQS"), "R_rad": row["R_rad_ohm"]})
    return {"L_MQS": r.coupling["L_p_H"], "rows": rows}


def template(wd: Path, name: str):
    kRs = [1e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5]
    d = af.template(name)
    r0 = d.solve(wd / f"{name}_mqs", verbose=0)
    R = r0.coupling["device_radius_m"]
    d.doc["fullwave"] = {"frequencies": [k * C / (2 * math.pi * R) for k in kRs], "modes": 3}
    t = time.time()
    r = d.solve(wd / f"{name}_fw", verbose=0)
    fw = r.data["fullwave"]
    rows = []
    for kR, row in zip(kRs, fw["sweep"]):
        rows.append({"kR": kR, "f_Hz": row["f_Hz"], "Psi_over_MQS": row.get("Psi_over_MQS"),
                     "L_over_MQS": row.get("L_over_MQS"), "R_rad": row["R_rad_ohm"], "Q_rad": row.get("Q_rad"),
                     "balance": row.get("power_balance_rel")})
    modes = [{"f_Hz": m["f_Hz"], "kR": m["k_R"]} for m in fw["eigenmodes"][list(fw["eigenmodes"])[0]]["modes"]]
    return {"family": fw["family"], "R_m": R, "Psi_MQS": r.coupling["Psi_SI_H_T_m2"], "L_MQS": r.coupling["L_p_H"],
            "dofs": fw["sweep"][0]["solver"]["dofs"], "seconds_per_frequency": fw["seconds"] / len(kRs),
            "sweep": rows, "closed_modes": modes, "warnings": r.data["warnings"]}


def fmt(x, f="{:+.2e}"):
    return "–" if x is None else f.format(x)


def report(rec):
    L = ["# Full-wave solver validation (beyond MQS)", "",
         "Generated by `benchmarks/fullwave_validation.py` (axisymmetric, order 2). Time dependence e^{−iωt};",
         "the readout is driven with I = 1 A. Formulation: `cpp/include/axionfem/fullwave.hpp`.", ""]
    L += ["## Perfectly conducting sphere (R = 1 m): eigenmodes", "",
          "| family | mode | kR exact | kR computed | rel. error |", "|---|---|---|---|---|"]
    for r in rec["sphere"]:
        L.append(f"| {r['family']} | {r['mode']} | {r['kR_exact']:.6f} | {r['kR']:.6f} | {r['rel']:+.1e} |")
    L += ["", "## Superconducting loop (a = 0.5 m): radiation resistance", "",
          "Exact reference: uniform-current loop, R = η (π/2) ka ∫₀^{2ka} J₂(x) dx (thin wire). The residual",
          "scales as (b/a)²: the current on a wire of radius b is not a filament.", ""]
    for key in ("loop_b0.005", "loop_b0.0025"):
        rows, l0 = rec[key]
        L += [f"Wire radius b/a = {rows[0]['b_over_a']:.3f}; MQS L = {l0['L_MQS']:.6e} H vs thin ring μ0 a (ln 8a/b − 2) "
              f"= {l0['L_thin_ring']:.6e} H ({l0['rel']:+.1e}).", "",
              "| ka | R_rad [Ω] | exact [Ω] | rel. | L(ω)/L_MQS | power balance |", "|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['ka']} | {r['R']:.5e} | {r['R_exact']:.5e} | {r['rel']:+.1e} | {r['L_over_MQS']:.5f} | "
                     f"{r['balance']:.0e} |")
        L.append("")
    rows, tem = rec["coax"]
    L += ["## Sealed coaxial cavity with the readout port between two shorted stubs (TM0)", "",
          "Inner radius 0.05 m, outer 0.15 m, stubs 0.6 m and 0.3 m; the port gap is 5 mm. Transmission-line reference:",
          "Z = (−iZ₀ tan kℓ₁) ∥ (−iZ₀ tan kℓ₂); the table compares Z/(−iωL_MQS).", "",
          f"MQS L = {tem['L_MQS']:.5e} H, transmission line {tem['L_TL']:.5e} H (port and end effects).", "",
          "| f [MHz] | Z/(−iωL) computed | transmission line | rel. | R [Ω] |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['f_Hz']/1e6:g} | {r['L_over_MQS']:.6f} | {r['TL']:.6f} | {r['rel']:+.1e} | {r['R']:.0e} |")
    L += ["", f"TEM resonance with the port open: {tem['f_TEM']/1e6:.4f} MHz vs c/(2ℓ) = {tem['f_TEM_exact']/1e6:.4f} MHz "
          f"({tem['rel']:+.1e}).", ""]
    b = rec.get("benabou")
    if b:
        L += ["## Comparison with Benabou et al. 2022 (arXiv:2211.00008): fiducial toroid, unshielded", "",
              "r_in = 0.5 m, r_out = 1 m, h = 0.5 m, 2.5 cm sheath. The paper's dipole estimate",
              "Q_rad = (54/49)(k h)⁻¹(k r_in)⁻² follows their COMSOL model at low frequency; the paper notes it breaks",
              "down towards its first self-resonance (~25 MHz). That self-resonance is a standing wave around the",
              "circumference of a slit fed at one point (m ≠ 0), which the axisymmetric (uniform-feed) model excludes;",
              "it needs the 3D full-wave solver.", "",
              f"MQS L_p = {b['L_MQS']:.4e} H.", "",
              "| f [MHz] | Q_rad (axionFEM) | Q_rad (dipole formula) | ratio | L(ω)/L_MQS |", "|---|---|---|---|---|"]
        for r in b["rows"]:
            L.append(f"| {r['f_MHz']:g} | {r['Q_rad']:.4g} | {r['Q_dipole']:.4g} | {r['Q_rad'] / r['Q_dipole']:.3f} | "
                     f"{r['L_over_MQS']:.4f} |")
        L.append("")
    L += ["## Templates: from the MQS limit to kR ~ 1", "",
          "kR uses R = the outer radius of all regions. Deviations from MQS grow as (kR)²; the radiation resistance of the",
          "toroid's azimuthal readout grows as (kR)⁴ (magnetic dipole). Poloidal readouts (coax, CORE) are",
          "well shielded and radiate little; they self-resonate near kR ≈ 0.9 (L(ω) changes sign). The closed-",
          "sphere eigenmodes include modes of the surrounding sphere. For TM readouts Ψ(ω) includes the reciprocity",
          "integral outside the domain sphere (static multipoles × outgoing readout wave), which is up to 10 % of the",
          "unshielded solenoid's Ψ at kR ≈ 0.1–0.6; with it Ψ does not depend on the domain radius (tests/test_fullwave.py).", ""]
    for name in ("toroid", "solenoid", "core"):
        t = rec[name]
        L += [f"### {name} ({t['family']}), R = {t['R_m']:.3f} m, {t['dofs']:,} DOFs, "
              f"{t['seconds_per_frequency']:.1f} s per frequency", "",
              "| kR | f [MHz] | \\|Ψ\\|/Ψ_MQS − 1 | L/L_MQS − 1 | R_rad [Ω] | Q_rad | power balance |",
              "|---|---|---|---|---|---|---|"]
        for r in t["sweep"]:
            L.append(f"| {r['kR']:g} | {r['f_Hz']/1e6:.4g} | {fmt(None if r['Psi_over_MQS'] is None else r['Psi_over_MQS']-1)} | "
                     f"{fmt(None if r['L_over_MQS'] is None else r['L_over_MQS']-1)} | {r['R_rad']:.2e} | "
                     f"{fmt(r['Q_rad'], '{:.2e}')} | {fmt(r['balance'], '{:.0e}')} |")
        L.append("")
        L.append("Closed-sphere eigenmodes (kR): " + ", ".join(f"{m['kR']:.3f}" for m in t["closed_modes"]))
        L.append("")
    return "\n".join(L) + "\n"


def main():
    wd = Path(sys.argv[1] if len(sys.argv) > 1 else "fullwave_validation_runs")
    wd.mkdir(parents=True, exist_ok=True)
    rec = {"sphere": sphere(wd)}
    rec["loop_b0.005"] = loop(wd, 0.005)
    rec["loop_b0.0025"] = loop(wd, 0.0025)
    rec["coax"] = coax(wd)
    rec["benabou"] = benabou(wd)
    for name in ("toroid", "solenoid", "core"):
        rec[name] = template(wd, name)
    REC.write_text(json.dumps(rec, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report(rec))
    print(OUT)


if __name__ == "__main__":
    main()
