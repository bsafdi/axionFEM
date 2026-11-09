#!/usr/bin/env python
"""axionFEM vs the 2022 COMSOL run of the shielded lossless toroid (arXiv:2211.00008, Small_Shield.mph).

COMSOL data (exp(+jwt), rows: f [GHz], Re, Im):
  source_impedance.npy  Zport (port excited, Je off), 5-500 MHz
  source_voltage.npy    Vport with Je = 1 A/m^2 on and the port loaded by its Zref = 50 ohm.  The open-circuit
                        voltage is V_oc = Vport (Zport + Zref)/Zref.  The conversion is checked here: for a lossless
                        model V_oc = j w Psi must be purely imaginary, so the residual phase of the converted V_oc
                        is reported (|arg V_oc| - 90 deg).
axionFEM (exp(-iwt)): results.json of each run dir, fullwave.sweep[] rows (Z_re_ohm, Z_im_ohm, Psi_re, Psi_im) and
fullwave.eigenmodes.  Z_axf = conj(Z_comsol), V_oc,axf = -i w Psi (B0_T = 1 <-> Je = 1 A/m^2).  We compare
Im Z_comsol = w L with -Z_im_ohm, and |V_oc| with w |Psi|.  The sign of Psi depends on the port orientation
("toward"); Psi_comsol = V_oc/(j w) is real, and the ratio Psi_axf/Psi_comsol is reported with its sign.

Poles of the COMSOL Zport (Im Z jumping from + to -, |Z| large) are the eigenmodes of the open-port structure that
couple to the port; they are matched to the closed-domain eigenmodes of axionFEM.  An axionFEM mode's coupling is
gauged by the electric-energy fraction in the port volume (energy_fraction["port"]).

    python compare_toroid_comsol.py --comsol DIR --runs RUN [RUN ...] [--control RUN ...] [--tag NAME]

Writes axionfem_results/<tag>.json and build/<tag>.pdf.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ZREF = 50.0


def load_comsol(d: Path):
    zf = np.load(d / "source_impedance.npy", allow_pickle=False)
    vf = np.load(d / "source_voltage.npy", allow_pickle=False)
    fz, z = zf[0] * 1e9, zf[1] + 1j * zf[2]
    fv, vl = vf[0] * 1e9, vf[1] + 1j * vf[2]
    m = fv <= fz[-1]                       # V_oc needs Zport: restrict to the common band
    fv, vl = fv[m], vl[m]
    zv = np.interp(fv, fz, z.real) + 1j * np.interp(fv, fz, z.imag) if not np.array_equal(fv, fz) else z
    voc = vl * (zv + ZREF) / ZREF
    return dict(f=fz, Z=z, fV=fv, Vload=vl, Voc=voc)


def comsol_poles_zeros(c):
    f, x = c["f"], c["Z"].imag
    s = np.sign(x)
    i = np.nonzero(s[:-1] != s[1:])[0]
    poles, zeros = [], []
    for k in i:
        # linear interpolation of the crossing; a pole jumps + -> - with large |Im Z|
        if s[k] > 0 > s[k + 1] and min(abs(x[k]), abs(x[k + 1])) > 10.0:
            # pole: estimate from 1/Im Z, which crosses zero linearly
            y0, y1 = 1 / x[k], 1 / x[k + 1]
            poles.append(f[k] + (f[k + 1] - f[k]) * y0 / (y0 - y1))
        elif s[k] < 0 < s[k + 1]:
            zeros.append(f[k] + (f[k + 1] - f[k]) * x[k] / (x[k] - x[k + 1]))
    return np.array(poles), np.array(zeros)


def interp_c(f0, f, y):
    """Linear interpolation of a dense (1 kHz) complex curve; NaN outside its band (never extrapolate)."""
    out = np.interp(f0, f, y.real) + 1j * np.interp(f0, f, y.imag)
    return np.where((f0 < f[0]) | (f0 > f[-1]), np.nan + 0j, out)


def load_run(d: Path):
    r = json.loads((d / "results.json").read_text())
    fw = r["fullwave"]
    rows = []
    for s in fw.get("sweep", []):
        dz = (s.get("d_ImZ_domega_from_energy_H") or {}).get("re")
        rows.append(dict(f=s["f_Hz"], Z=complex(s["Z_re_ohm"], s["Z_im_ohm"]), Psi=complex(s["Psi_re"], s["Psi_im"]),
                         dX=-float(dz[0][0]) if dz else None,   # d(w L)/dw, COMSOL sign (Foster: > 0)
                         accepted=bool(s.get("validation_accepted", (s.get("solver") or {}).get("accepted", True)))))
    eig = []
    em = fw.get("eigenmodes") or {}
    groups = em.values() if em and "modes" not in em else [em]
    for g in groups:
        for m in (g or {}).get("modes", []):
            eig.append(dict(f=m["f_Hz"], port_fraction=float(m.get("energy_fraction", {}).get("port", 0.0)),
                            converged=bool(m.get("converged", False)), residual=m.get("relative_residual")))
    mesh = r.get("mesh", {})
    info = dict(dofs=mesh.get("dofs") or mesh.get("ndofs"), elements=mesh.get("elements") or mesh.get("elements_3d"),
                seconds=(r.get("timing") or {}))
    return rows, eig, info


def axf_poles_zeros(rows):
    """Poles and zeros of X = Im Z (COMSOL sign) bracketed by consecutive sweep points.

    Pole between w1 < w2 (X: + -> -): with X ~ A/(w_p - w) + smooth, dX/dw ~ A/(w_p - w)^2 at both points, so
    (w_p - w1)/(w2 - w_p) = sqrt(X'2/X'1), independent of A and the background.  Zero (X: - -> +): Newton step from
    the point with the smaller |X| (X' from the energy identity).  The bracket width is the uncertainty bound."""
    poles, zeros = [], []
    for a, b in zip(rows[:-1], rows[1:]):
        xa, xb = -a["Z"].imag, -b["Z"].imag
        wa, wb = 2 * np.pi * a["f"], 2 * np.pi * b["f"]
        if xa > 0 > xb:
            if a["dX"] and b["dX"] and a["dX"] > 0 and b["dX"] > 0:
                sa, sb = np.sqrt(a["dX"]), np.sqrt(b["dX"])
                wp = (wa * sa + wb * sb) / (sa + sb)
            else:
                wp = wa + (wb - wa) * (1 / xa) / (1 / xa - 1 / xb)
            poles.append(dict(f_Hz=wp / (2 * np.pi), bracket_Hz=[a["f"], b["f"]]))
        elif xa < 0 < xb:
            c = a if abs(xa) < abs(xb) else b
            if c["dX"] and c["dX"] > 0:
                w0 = 2 * np.pi * c["f"] - (-c["Z"].imag) / c["dX"]
            else:
                w0 = wa + (wb - wa) * xa / (xa - xb)
            zeros.append(dict(f_Hz=w0 / (2 * np.pi), bracket_Hz=[a["f"], b["f"]]))
    return poles, zeros


def sweep_table(rows, c, psi_sign):
    out = []
    for s in rows:
        f, w = s["f"], 2 * np.pi * s["f"]
        zc = interp_c(np.array([f]), c["f"], c["Z"])[0]
        vc = interp_c(np.array([f]), c["fV"], c["Voc"])[0]
        imz_a = -s["Z"].imag                                  # COMSOL convention Im Z = w L
        voc_a = w * abs(s["Psi"])
        psi_c = (vc / (1j * w)).real
        out.append(dict(f_Hz=f, ImZ_axf_ohm=imz_a, ImZ_comsol_ohm=zc.imag, L_axf_H=imz_a / w, L_comsol_H=zc.imag / w,
                        rel_ImZ=(imz_a - zc.imag) / abs(zc.imag),
                        absVoc_axf_V=voc_a, absVoc_comsol_V=abs(vc), rel_absVoc=(voc_a - abs(vc)) / abs(vc),
                        Psi_axf_Wb=psi_sign * s["Psi"].real, Psi_axf_im_Wb=s["Psi"].imag, Psi_comsol_Wb=psi_c,
                        Psi_ratio=psi_sign * s["Psi"].real / psi_c if psi_c else None,
                        ReZ_axf_ohm=s["Z"].real, dX_axf_H=s["dX"], accepted=s["accepted"]))
    return out


def match_modes(eig, poles):
    """Each COMSOL pole -> the most port-coupled unused axionFEM eigenmode within 0.1%, else the nearest."""
    table, used = [], set()
    for p in poles:
        if len(used) == len(eig):
            break
        free = [i for i in range(len(eig)) if i not in used]
        near = [i for i in free if abs(eig[i]["f"] - p) <= 1e-3 * p]     # several modes within 0.1%: the
        k = (max(near, key=lambda i: eig[i]["port_fraction"]) if near     # port-coupled one is the pole
             else min(free, key=lambda i: abs(eig[i]["f"] - p)))
        used.add(k)
        e = eig[k]
        table.append(dict(f_comsol_pole_Hz=p, f_axf_Hz=e["f"], rel=(e["f"] - p) / p, port_fraction=e["port_fraction"],
                          residual=e["residual"], converged=e["converged"]))
    extra = [dict(f_axf_Hz=e["f"], port_fraction=e["port_fraction"], residual=e["residual"])
             for i, e in enumerate(eig) if i not in used]
    return table, extra


def match_points(est, ref):
    """Sweep-bracketed pole/zero estimates -> nearest COMSOL value."""
    out = []
    for e in est:
        if len(ref):
            r = ref[np.argmin(abs(ref - e["f_Hz"]))]
            out.append(dict(e, f_comsol_Hz=float(r), rel=(e["f_Hz"] - r) / r))
    return out


Q_MQS = 1e6   # paper: intrinsic resistance r = w L_MQS / Q_MQS


def resonant_sensitivity(f, X, dX, absV, L_ref, psi_ref):
    """Fig. 3 / S1 of arXiv:2211.00008 for a lossless source (formalism identified from the paper's curves).

    Tuned LC readout (capacitor if X > 0, inductor if X < 0) with r = w L_ref / Q_MQS:
      Q_B = Q_MQS (dX/dw + |X|/w) / (2 L_ref)               (FWHM of the tuned response; = Q_MQS in the MQS limit)
      g/g_MQS = (w psi_ref / |V_oc|) (Q_B / Q_MQS)^(1/4)
    L_ref, psi_ref: the 5 MHz values X/w and |V_oc|/w (the paper normalises to its lowest frequency).  Applied to the
    COMSOL data this reproduces the paper's lossless g/g_MQS to ~0.1% (median) for both eta_A = 20 and 1 (the lossless
    curve does not depend on eta_A); Q_B agrees to <~1% except within a few % of a pole, where the paper's curve is
    higher (by up to x2-4 at the peaks), consistent with a coarse finite-difference dX/dw in the paper."""
    w = 2 * np.pi * np.asarray(f)
    qb = Q_MQS * (np.asarray(dX) + np.abs(X) / w) / (2 * L_ref)
    g = (w * psi_ref / np.asarray(absV)) * (qb / Q_MQS) ** 0.25
    return qb, g


def paper_figures(c, tab, curves_dir: Path, out_dir: Path):
    """axionFEM curves for the paper figures that involve the shielded lossless toroid (Figs. 3 and S1, 'Lossless').

    Writes out_dir/<figure>.json (read by make_note.py) and returns the validation of the formalism (COMSOL data
    pushed through resonant_sensitivity vs the curves extracted from the paper)."""
    f, X = c["f"], c["Z"].imag
    w = 2 * np.pi * f
    Vc = np.interp(f, c["fV"], np.abs(c["Voc"]))
    qb_c, g_c = resonant_sensitivity(f, X, np.gradient(X, w), Vc, X[0] / w[0], Vc[0] / w[0])
    t = [x for x in tab if x["accepted"] and x["dX_axf_H"]]
    fa = np.array([x["f_Hz"] for x in t])
    t0 = min(t, key=lambda x: x["f_Hz"])
    w0 = 2 * np.pi * t0["f_Hz"]
    qb_a, g_a = resonant_sensitivity(fa, [x["ImZ_axf_ohm"] for x in t], [x["dX_axf_H"] for x in t],
                                     [x["absVoc_axf_V"] for x in t], t0["ImZ_axf_ohm"] / w0, t0["absVoc_axf_V"] / w0)
    valid = {}
    for stem in ("Relative_Sensitivity_Eta20", "Relative_Sensitivity_Eta1"):
        rec = json.loads((curves_dir / f"{stem}.json").read_text())
        panels, val = [], []
        for p, ya, yc in zip(rec["panels"], (g_a, qb_a), (g_c, qb_c)):
            ref = [k for k in p["curves"] if k.get("label") == "Lossless"][0]
            x, y = np.array(ref["x"]) * 1e6, np.array(ref["y"])
            d = np.abs(np.log(np.interp(x, f, yc) / y))
            val.append(dict(median_abs_log=float(np.nanmedian(d)), p90_abs_log=float(np.nanpercentile(d, 90))))
            # compare axionFEM with the paper curve where both exist (paper interpolated at the axionFEM points)
            yp = np.interp(fa, x, y)
            panels.append(dict(curves=[dict(x=(fa / 1e6).tolist(), y=np.asarray(ya).tolist(), marker="o", ls="none",
                                            label="axionFEM, lossless (shielded)")],
                               rel_vs_paper=((np.asarray(ya) - yp) / yp).tolist()))
        (out_dir / f"{stem}.json").write_text(json.dumps(dict(
            source="compare_toroid_comsol.py: axionFEM shielded lossless toroid sweep through resonant_sensitivity()",
            formalism=resonant_sensitivity.__doc__, panels=panels,
            formalism_check_comsol_vs_paper=dict(g_over_gMQS=val[0], Q_B=val[1])), indent=1) + "\n")
        valid[stem] = dict(g_over_gMQS=val[0], Q_B=val[1])
    return valid


def plot(path, c, poles, tab, ctrl, eig, fmax):
    """Im Z and |V_oc| vs f (COMSOL line, axionFEM points, eigenfrequency ticks) and |relative difference|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MAROON, GRAY = "#800000", "0.5"
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 8.8), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 2]))
    m = c["f"] <= fmax
    ax[0].plot(c["f"][m] / 1e6, c["Z"].imag[m], color="k", lw=0.8, label="COMSOL (2022)")
    ax[0].set_yscale("symlog", linthresh=10)
    ax[0].set_ylabel(r"Im $Z = \omega L$ [$\Omega$]")
    mv = c["fV"] <= fmax
    ax[1].plot(c["fV"][mv] / 1e6, np.abs(c["Voc"][mv]), color="k", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$|V_{\rm oc}|$ [V]  ($J_e = 1$ A/m$^2$)")
    sets = [(tab, MAROON, "o", True, "axionFEM")]
    sets += [(t, GRAY, "s", False, f"axionFEM, {lab}") for lab, t in ctrl.items()]
    for t, col, mk, fill, lab in sets:
        if not t:
            continue
        f = np.array([x["f_Hz"] for x in t]) / 1e6
        kw = dict(color=col, ms=3.5, ls="none", mfc=col if fill else "none")
        ax[0].plot(f, [x["ImZ_axf_ohm"] for x in t], mk, label=lab, **kw)
        ax[1].plot(f, [x["absVoc_axf_V"] for x in t], mk, **kw)
        ax[2].plot(f, np.abs([x["rel_ImZ"] for x in t]), mk, **kw)
        ax[2].plot(f, np.abs([x["rel_absVoc"] for x in t]), "^", color=col, ms=3.5, ls="none", mfc="none")
    ax[2].plot([], [], "o", color="k", ms=3.5, ls="none", label=r"Im $Z$")
    ax[2].plot([], [], "^", color="k", ms=3.5, ls="none", mfc="none", label=r"$|V_{\rm oc}|$")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("|axionFEM / COMSOL - 1|")
    ax[2].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    for a in ax:
        for p in poles:
            a.axvline(p / 1e6, color=GRAY, lw=0.5, ls=":")
    tr = ax[0].get_xaxis_transform()
    for e in eig:
        if e["f"] <= fmax:
            col = MAROON if e["port_fraction"] > 1e-6 else GRAY
            ax[0].plot([e["f"] / 1e6] * 2, [0.93, 1.0], transform=tr, color=col, lw=1.2, clip_on=False)
    ax[0].plot([], [], color=MAROON, lw=1.2, label="axionFEM eigenmode (port-coupled)")
    ax[0].plot([], [], color=GRAY, lw=1.2, label="axionFEM eigenmode (not coupled)")
    ax[0].plot([], [], color=GRAY, lw=0.5, ls=":", label="COMSOL pole")
    ax[0].set_ylim(-1e5, 1e5)
    ax[2].set_xlabel("frequency [MHz]")
    ax[2].set_xlim(0, fmax / 1e6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comsol", type=Path, required=True, help="dir with source_impedance.npy, source_voltage.npy")
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="axionFEM run dirs (results.json) of the production mesh: sweeps and eigenmodes, merged")
    ap.add_argument("--control", nargs="*", default=[], metavar="LABEL=DIR[,DIR]",
                    help="other meshes/orders at a few frequencies (discretization control), e.g. refined=runs/ctrlD")
    ap.add_argument("--modes", type=Path, nargs="*", default=[],
                    help="run dirs whose eigenmodes are used (their sweep rows are ignored); default: the --runs")
    ap.add_argument("--figures", action="store_true",
                    help="also write the paper-figure overlays (axionfem_results/Relative_Sensitivity_Eta*.json)")
    ap.add_argument("--tag", default="toroid_small_shield_lossless_vs_comsol")
    ap.add_argument("--fmax", type=float, default=150e6)
    a = ap.parse_args()
    c = load_comsol(a.comsol)
    poles, zeros = comsol_poles_zeros(c)
    pz = poles[poles <= a.fmax]
    zz = zeros[zeros <= a.fmax]
    # residual phase of the converted V_oc (0 for a lossless model if the 50-ohm-load reading is right)
    w = c["fV"] <= a.fmax
    ph = np.abs(np.abs(np.angle(c["Voc"][w], deg=True)) - 90.0)
    rows, eig, infos = [], [], []
    for d in a.runs:
        r, e, i = load_run(d)
        rows += r
        eig += e
        infos.append(dict(run=str(d), **i))
    rows.sort(key=lambda s: s["f"])
    modes_from = [str(d) for d in a.runs]
    if a.modes:
        eig, modes_from = [], [str(d) for d in a.modes]
        for d in a.modes:
            eig += load_run(d)[1]
    eig = sorted({round(e["f"], 0): e for e in eig}.values(), key=lambda e: e["f"])
    # port orientation: fix the sign of Psi at the lowest frequency
    r0 = rows[0]
    v0 = interp_c(np.array([r0["f"]]), c["fV"], c["Voc"])[0]
    psi_sign = float(np.sign(r0["Psi"].real * (v0 / 1j).real)) or 1.0
    tab = sweep_table(rows, c, psi_sign)
    ctrl, cinfo, disc = {}, [], []
    for spec in a.control:
        lab, dirs = spec.split("=", 1)
        crow = []
        for d in dirs.split(","):
            r, _, i = load_run(Path(d))
            crow += r
            cinfo.append(dict(label=lab, run=d, **i))
        ctrl[lab] = sweep_table(sorted(crow, key=lambda s: s["f"]), c, psi_sign)
        for t in ctrl[lab]:
            b = [b for b in tab if abs(b["f_Hz"] - t["f_Hz"]) < 1]
            if b:
                b = b[0]
                disc.append(dict(label=lab, f_Hz=t["f_Hz"],
                                 dImZ_rel=(t["ImZ_axf_ohm"] - b["ImZ_axf_ohm"]) / abs(b["ImZ_axf_ohm"]),
                                 dVoc_rel=(t["absVoc_axf_V"] - b["absVoc_axf_V"]) / b["absVoc_axf_V"]))
    modes, extra = match_modes([e for e in eig if e["f"] <= 1.1 * a.fmax], poles[poles <= 1.1 * a.fmax])
    sp, sz = axf_poles_zeros(rows)
    sweep_poles, sweep_zeros = match_points(sp, pz), match_points(sz, zz)
    acc = [t for t in tab if t["accepted"]]
    far = [t for t in acc if all(abs(t["f_Hz"] - p) / p > 0.03 for p in pz)]   # >3% from any COMSOL pole
    stats = dict(n=len(tab), n_accepted=len(acc),
                 median_abs_rel_ImZ=float(np.median([abs(t["rel_ImZ"]) for t in acc])) if acc else None,
                 median_abs_rel_Voc=float(np.median([abs(t["rel_absVoc"]) for t in acc])) if acc else None,
                 far_from_poles=dict(n=len(far),
                                     max_abs_rel_ImZ=max([abs(t["rel_ImZ"]) for t in far], default=None),
                                     max_abs_rel_Voc=max([abs(t["rel_absVoc"]) for t in far], default=None)))
    out = dict(
        what="axionFEM vs COMSOL (2022), shielded lossless toroid of arXiv:2211.00008 (Small_Shield.mph)",
        conventions="COMSOL exp(+jwt): Im Z = +w L; axionFEM values conjugated. V_oc = Vport (Zport+50)/50 (COMSOL "
                    "source run loads the port with Zref = 50 ohm). Psi sign fixed by the port orientation at the "
                    "lowest sweep frequency. rel = (axionFEM - COMSOL)/|COMSOL|.",
        comsol=dict(L_5MHz_H=float(np.interp(5e6, c["f"], c["Z"].imag) / (2 * np.pi * 5e6)),
                    poles_Hz=pz.tolist(), zeros_Hz=zz.tolist(),
                    Voc_conversion_max_phase_residual_deg=float(ph.max()),
                    Voc_conversion_median_phase_residual_deg=float(np.median(ph))),
        axionfem_runs=infos, control_runs=cinfo, eigenmodes_from=modes_from,
        eigenmodes_all_converged=bool(eig) and all(e["converged"] for e in eig), psi_orientation_sign=psi_sign, summary=stats,
        sweep=tab, control=ctrl, discretization=disc, eigenmodes_matched=modes, eigenmodes_unmatched=extra,
        sweep_poles=sweep_poles, sweep_zeros=sweep_zeros)
    (HERE / "axionfem_results").mkdir(exist_ok=True)
    jp = HERE / "axionfem_results" / f"{a.tag}.json"
    jp.write_text(json.dumps(out, indent=1, default=float) + "\n")
    (HERE / "build").mkdir(exist_ok=True)
    pp = HERE / "build" / f"{a.tag}.pdf"
    plot(pp, c, pz, tab, ctrl, eig, a.fmax)
    if a.figures:
        out["paper_figures_formalism_check"] = paper_figures(c, tab, HERE / "paper_curves", HERE / "axionfem_results")
        jp.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print("formalism check (COMSOL data vs paper curves):", json.dumps(out["paper_figures_formalism_check"]))
    print(f"wrote {jp}\nwrote {pp}")
    print(f"COMSOL: L(5 MHz) = {out['comsol']['L_5MHz_H']*1e9:.1f} nH; poles [MHz] "
          + ", ".join(f"{p/1e6:.3f}" for p in pz) + "; zeros " + ", ".join(f"{z/1e6:.3f}" for z in zz)
          + f"; V_oc phase residual median {np.median(ph):.1e} deg")
    print(" f[MHz]   ImZ_axf    ImZ_com    relZ       |V|axf    |V|com    relV     Psi ratio  acc")
    for t in tab:
        print(f"{t['f_Hz']/1e6:8.3f} {t['ImZ_axf_ohm']:10.3f} {t['ImZ_comsol_ohm']:10.3f} {t['rel_ImZ']:+.2e} "
              f"{t['absVoc_axf_V']:9.4f} {t['absVoc_comsol_V']:9.4f} {t['rel_absVoc']:+.2e} {t['Psi_ratio']:+.4f} {t['accepted']}")
    print("summary:", json.dumps(stats))
    for t in disc:
        print(f"control {t['label']} @ {t['f_Hz']/1e6:.3f} MHz: (control - main)/main ImZ {t['dImZ_rel']:+.2e}, "
              f"|V| {t['dVoc_rel']:+.2e}")
    print(" COMSOL pole   axionFEM mode   rel       port E-frac  residual")
    for m in modes:
        print(f"{m['f_comsol_pole_Hz']/1e6:10.3f} {m['f_axf_Hz']/1e6:12.3f}  {m['rel']:+.2e}  {m['port_fraction']:.1e}  "
              f"{m['residual']:.1e}" + ("" if m["converged"] else "  unconverged (eigensolver residual above tolerance)"))
    for e in extra:
        print(f"   (no pole) {e['f_axf_Hz']/1e6:12.3f}             {e['port_fraction']:.1e}  {e['residual']:.1e}")
    for nm, lst in (("pole", sweep_poles), ("zero", sweep_zeros)):
        for e in lst:
            print(f"sweep {nm}: axionFEM {e['f_Hz']/1e6:.3f} (bracket {e['bracket_Hz'][0]/1e6:.2f}-"
                  f"{e['bracket_Hz'][1]/1e6:.2f}) vs COMSOL {e['f_comsol_Hz']/1e6:.3f} MHz, rel {e['rel']:+.2e}")


if __name__ == "__main__":
    main()
