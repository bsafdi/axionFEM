#!/usr/bin/env python
"""axionFEM vs the 2022 COMSOL run of the shielded lossless solenoid (arXiv:2211.00008, Model_File.mph).

COMSOL data (exp(+jwt)), in the directory given by --comsol: {Source_Impedance,Port_Voltage}_{5_50,50_150,150_160}.txt,
5-160 MHz at 1 kHz: Zport_1 (port excited, Je off) and Vport_1 (Je = (0,0,1) A/m^2 in the bore, port loaded by
Zref = 50 ohm), so V_oc = Vport (Zport + 50)/50.  Everything else (conventions, pole/zero estimates, eigenmode matching,
plot) is shared with compare_toroid_comsol.py.

COMSOL poles are classified by their strength: the residue a = -X (f - f_p) [ohm Hz] measured 20 kHz on either side
(X ~ a/(f_p - f)).  'narrow' poles (a < --narrow, default 1e8 ohm Hz; the three strong poles have a > 2.7e8, all others < 7.3e7) are weakly
port-coupled features (pole-zero pairs a few 100 kHz wide) that a sweep cannot bracket; they are compared with
axionFEM eigenmodes only.

    python compare_solenoid_comsol.py --comsol DIR --runs RUN [RUN ...] [--modes RUN] [--control LABEL=DIR,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import compare_toroid_comsol as ct

HERE = Path(__file__).resolve().parent
ZREF = 50.0


def load_comsol_txt(d: Path):
    def load(stem):
        a = np.concatenate([np.loadtxt(d / f"{stem}_{b}.txt", comments="%") for b in ("5_50", "50_150", "150_160")])
        a = a[np.argsort(a[:, 0], kind="stable")]
        _, i = np.unique(np.round(a[:, 0] * 1e3), return_index=True)     # 50 MHz appears twice
        a = a[i]
        return a[:, 0] * 1e6, a[:, 1] + 1j * a[:, 2]
    fz, z = load("Source_Impedance")
    fv, vl = load("Port_Voltage")
    assert np.array_equal(fz, fv)
    return dict(f=fz, Z=z, fV=fv, Vload=vl, Voc=vl * (z + ZREF) / ZREF)


def pole_strengths(c, poles, df=20e3):
    f, x = c["f"], c["Z"].imag
    out = []
    for p in poles:
        lo, hi = np.interp(p - df, f, x), np.interp(p + df, f, x)
        out.append(float(0.5 * (lo * df + (-hi) * df)))
    return np.array(out)


def resonant_figure(c, tab, out_dir: Path):
    """Fig. S5 ('Lossless' curve) of the paper from axionFEM's sweep, formalism of compare_toroid_comsol."""
    stem = "Solenoid_Relative_Sensitivity_Eta20"
    f, X = c["f"], c["Z"].imag
    w = 2 * np.pi * f
    m = f <= 150.5e6
    Vc = np.abs(c["Voc"])
    qb_c, g_c = ct.resonant_sensitivity(f[m], X[m], np.gradient(X, w)[m], Vc[m], X[0] / w[0], Vc[0] / w[0])
    t = [x for x in tab if x["accepted"] and x["dX_axf_H"] and x["f_Hz"] <= 150.5e6]
    fa = np.array([x["f_Hz"] for x in t])
    t0 = min(t, key=lambda x: x["f_Hz"])
    w0 = 2 * np.pi * t0["f_Hz"]
    qb_a, g_a = ct.resonant_sensitivity(fa, [x["ImZ_axf_ohm"] for x in t], [x["dX_axf_H"] for x in t],
                                        [x["absVoc_axf_V"] for x in t], t0["ImZ_axf_ohm"] / w0, t0["absVoc_axf_V"] / w0)
    rec = json.loads((HERE / "paper_curves" / f"{stem}.json").read_text())
    panels, val = [], []
    for p, ya, yc in zip(rec["panels"], (g_a, qb_a), (g_c, qb_c)):
        ref = [k for k in p["curves"] if k.get("label") == "Lossless"][0]
        x, y = np.array(ref["x"]) * 1e6, np.array(ref["y"])
        d = np.abs(np.log(np.interp(x, f[m], yc) / y))
        val.append(dict(median_abs_log=float(np.nanmedian(d)), p90_abs_log=float(np.nanpercentile(d, 90))))
        yp = np.interp(fa, x, y)
        panels.append(dict(curves=[dict(x=(fa / 1e6).tolist(), y=np.asarray(ya).tolist(), marker="o", ls="none",
                                        label="axionFEM, lossless (shielded)")],
                           rel_vs_paper=((np.asarray(ya) - yp) / yp).tolist()))
    valid = dict(g_over_gMQS=val[0], Q_B=val[1])
    (out_dir / f"{stem}.json").write_text(json.dumps(dict(
        source="compare_solenoid_comsol.py: axionFEM shielded lossless solenoid sweep through resonant_sensitivity()",
        formalism=ct.resonant_sensitivity.__doc__, panels=panels, formalism_check_comsol_vs_paper=valid), indent=1) + "\n")
    return valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comsol", type=Path, required=True, help="dir with the Shielded_Solenoid *.txt tables")
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--control", nargs="*", default=[], metavar="LABEL=DIR[,DIR]")
    ap.add_argument("--modes", type=Path, nargs="*", default=[])
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--narrow", type=float, default=1e8, help="pole residue [ohm Hz] below which a pole is 'narrow'")
    ap.add_argument("--tag", default="solenoid_shielded_lossless_vs_comsol")
    ap.add_argument("--fmax", type=float, default=160e6)
    a = ap.parse_args()
    c = load_comsol_txt(a.comsol)
    poles, zeros = ct.comsol_poles_zeros(c)
    strength = pole_strengths(c, poles)
    w = c["fV"] <= a.fmax
    ph = np.abs(np.abs(np.angle(c["Voc"][w], deg=True)) - 90.0)
    rows, infos = [], []
    for d in a.runs:
        r, _, i = ct.load_run(d)
        rows += r
        infos.append(dict(run=str(d), **i))
    rows = sorted({round(s["f"]): s for s in rows}.values(), key=lambda s: s["f"])
    eig = []
    for d in (a.modes or a.runs):
        eig += ct.load_run(d)[1]
    eig = sorted({round(e["f"], 0): e for e in eig}.values(), key=lambda e: e["f"])
    r0 = rows[0]
    v0 = ct.interp_c(np.array([r0["f"]]), c["fV"], c["Voc"])[0]
    psi_sign = float(np.sign(r0["Psi"].real * (v0 / 1j).real)) or 1.0
    tab = ct.sweep_table(rows, c, psi_sign)
    ctrl, cinfo, disc = {}, [], []
    for spec in a.control:
        lab, dirs = spec.split("=", 1)
        crow = []
        for d in dirs.split(","):
            r, _, i = ct.load_run(Path(d))
            crow += r
            cinfo.append(dict(label=lab, run=d, **i))
        ctrl[lab] = ct.sweep_table(sorted(crow, key=lambda s: s["f"]), c, psi_sign)
        for t in ctrl[lab]:
            b = [b for b in tab if abs(b["f_Hz"] - t["f_Hz"]) < 1]
            if b:
                b = b[0]
                disc.append(dict(label=lab, f_Hz=t["f_Hz"],
                                 dImZ_rel=(t["ImZ_axf_ohm"] - b["ImZ_axf_ohm"]) / abs(b["ImZ_axf_ohm"]),
                                 dVoc_rel=(t["absVoc_axf_V"] - b["absVoc_axf_V"]) / b["absVoc_axf_V"]))
    sel = poles <= a.fmax
    pz, zz, st = poles[sel], zeros[zeros <= a.fmax], strength[sel]
    strong = pz[st >= a.narrow]
    # modes: match every COMSOL pole (strong first so they get the best-coupled modes)
    order = np.argsort(-st)
    modes, extra = ct.match_modes([e for e in eig if e["f"] <= 1.05 * a.fmax], pz[order])
    for m in modes:
        k = int(np.argmin(abs(pz - m["f_comsol_pole_Hz"])))
        m["comsol_residue_ohm_Hz"] = float(st[k])
        m["narrow"] = bool(st[k] < a.narrow)
    modes.sort(key=lambda m: m["f_comsol_pole_Hz"])
    sp, sz = ct.axf_poles_zeros(rows)
    sweep_poles, sweep_zeros = ct.match_points(sp, pz), ct.match_points(sz, zz)
    acc = [t for t in tab if t["accepted"] and np.isfinite(t["rel_ImZ"])]
    far = [t for t in acc if all(abs(t["f_Hz"] - p) / p > 0.03 for p in strong)]
    stats = dict(n=len(tab), n_accepted=len(acc),
                 median_abs_rel_ImZ=float(np.median([abs(t["rel_ImZ"]) for t in acc])) if acc else None,
                 median_abs_rel_Voc=float(np.median([abs(t["rel_absVoc"]) for t in acc])) if acc else None,
                 max_abs_rel_ImZ=max([abs(t["rel_ImZ"]) for t in acc], default=None),
                 max_abs_rel_Voc=max([abs(t["rel_absVoc"]) for t in acc], default=None),
                 far_from_strong_poles=dict(n=len(far),
                                            median_abs_rel_ImZ=float(np.median([abs(t["rel_ImZ"]) for t in far])) if far else None,
                                            max_abs_rel_ImZ=max([abs(t["rel_ImZ"]) for t in far], default=None),
                                            max_abs_rel_Voc=max([abs(t["rel_absVoc"]) for t in far], default=None)))
    out = dict(
        what="axionFEM vs COMSOL (2022), shielded lossless solenoid of arXiv:2211.00008 (Model_File.mph)",
        conventions="COMSOL exp(+jwt): Im Z = +w L; axionFEM values conjugated. V_oc = Vport (Zport+50)/50. Psi sign "
                    "fixed by the port orientation at the lowest sweep frequency. rel = (axionFEM - COMSOL)/|COMSOL|. "
                    f"COMSOL pole 'narrow' if its residue < {a.narrow:g} ohm Hz.",
        comsol=dict(L_5MHz_H=float(c["Z"].imag[0] / (2 * np.pi * c["f"][0])),
                    Psi_5MHz_Wb=float((c["Voc"][0] / 1j).real / (2 * np.pi * c["f"][0])),
                    poles_Hz=pz.tolist(), pole_residue_ohm_Hz=st.tolist(), strong_poles_Hz=strong.tolist(),
                    zeros_Hz=zz.tolist(),
                    Voc_conversion_max_phase_residual_deg=float(ph.max()),
                    Voc_conversion_median_phase_residual_deg=float(np.median(ph))),
        axionfem_runs=infos, control_runs=cinfo, eigenmodes_from=[str(d) for d in (a.modes or a.runs)],
        eigenmodes_all_converged=bool(eig) and all(e["converged"] for e in eig), psi_orientation_sign=psi_sign,
        summary=stats, sweep=tab, control=ctrl, discretization=disc, eigenmodes_matched=modes,
        eigenmodes_unmatched=extra, sweep_poles=sweep_poles, sweep_zeros=sweep_zeros)
    (HERE / "axionfem_results").mkdir(exist_ok=True)
    jp = HERE / "axionfem_results" / f"{a.tag}.json"
    (HERE / "build").mkdir(exist_ok=True)
    pp = HERE / "build" / f"{a.tag}.pdf"
    ct.plot(pp, c, strong, tab, ctrl, eig, a.fmax)
    if a.figures:
        out["paper_figures_formalism_check"] = resonant_figure(c, tab, HERE / "axionfem_results")
        print("formalism check (COMSOL data vs paper S5 curves):", json.dumps(out["paper_figures_formalism_check"]))
    jp.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {jp}\nwrote {pp}")
    print(f"COMSOL: L(5 MHz) = {out['comsol']['L_5MHz_H']*1e9:.2f} nH, Psi(5 MHz) = {out['comsol']['Psi_5MHz_Wb']:.4e} Wb")
    print("COMSOL poles [MHz] (residue ohm Hz): " + ", ".join(f"{p/1e6:.3f} ({s:.1e})" for p, s in zip(pz, st)))
    print("COMSOL zeros [MHz]: " + ", ".join(f"{z/1e6:.3f}" for z in zz))
    print(f"V_oc phase residual median {np.median(ph):.1e} deg")
    print(" f[MHz]   ImZ_axf    ImZ_com    relZ       |V|axf    |V|com    relV     Psi ratio  acc")
    for t in tab:
        print(f"{t['f_Hz']/1e6:8.3f} {t['ImZ_axf_ohm']:10.3f} {t['ImZ_comsol_ohm']:10.3f} {t['rel_ImZ']:+.2e} "
              f"{t['absVoc_axf_V']:9.4f} {t['absVoc_comsol_V']:9.4f} {t['rel_absVoc']:+.2e} {t['Psi_ratio']:+.4f} {t['accepted']}")
    print("summary:", json.dumps(stats))
    for t in disc:
        print(f"control {t['label']} @ {t['f_Hz']/1e6:.3f} MHz: ImZ {t['dImZ_rel']:+.2e}, |V| {t['dVoc_rel']:+.2e}")
    print(" COMSOL pole   axionFEM mode   rel       port E-frac  residual  residue")
    for m in modes:
        print(f"{m['f_comsol_pole_Hz']/1e6:10.3f} {m['f_axf_Hz']/1e6:12.3f}  {m['rel']:+.2e}  {m['port_fraction']:.1e}  "
              f"{m['residual']:.1e}  {m['comsol_residue_ohm_Hz']:.1e}" + (" narrow" if m["narrow"] else "")
              + ("" if m["converged"] else "  not converged"))
    for e in extra:
        print(f"   (no pole) {e['f_axf_Hz']/1e6:12.3f}             {e['port_fraction']:.1e}  {e['residual']:.1e}")
    for nm, lst in (("pole", sweep_poles), ("zero", sweep_zeros)):
        for e in lst:
            print(f"sweep {nm}: axionFEM {e['f_Hz']/1e6:.3f} (bracket {e['bracket_Hz'][0]/1e6:.2f}-"
                  f"{e['bracket_Hz'][1]/1e6:.2f}) vs COMSOL {e['f_comsol_Hz']/1e6:.3f} MHz, rel {e['rel']:+.2e}")


if __name__ == "__main__":
    main()
