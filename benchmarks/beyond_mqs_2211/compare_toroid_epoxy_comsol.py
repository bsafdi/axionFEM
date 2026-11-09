#!/usr/bin/env python
"""axionFEM vs the 2022 COMSOL run of the shielded LOSSY toroid (TACHYON 100G lining) of arXiv:2211.00008.

COMSOL data (Toroid_With_Epoxy/, exp(+jwt); rows f [MHz] (not GHz as in the lossless files), Re, Im):
  source_impedance.npy  Zport (port excited, Je off), 5-650 MHz, 1 kHz grid
  source_voltage.npy    Vport with Je = 1 A/m^2 on and the port loaded by Zref = 50 ohm; V_oc = Vport (Zport+50)/50
The model holds 19.0 % TACHYON by volume (the paper's ~20 %, Fig. S3); see MODEL_SETTINGS_epoxy.md.

axionFEM (exp(-iwt)): one results.json per run dir (one frequency per AMS job).  Z_axf = conj(Z_comsol): we compare
X = Im Z_comsol with -Z_im_ohm, R = Re Z (convention independent) and |V_oc| with w |Psi|.  dX/dw for the lossy model
is taken from the Foster energy identity dX/dw = 4 (W_e + W_m)/|I|^2 (energy_total_J_per_A2), exact for the lossless
operator and correct to O(tan_delta) here (flagged); it is checked against finite differences of close sweep pairs.

Poles of Z (X: + -> -) are located from the sweep bracket (1/X linear) and their quality factors from the Lorentzian
tails: near a pole Z ~ R_p / (1 - 2jQ d), d = f/f_p - 1, so Q = |X| / (2 |d| R) and R_p = X^2/R at any point a few
line widths away (the COMSOL 1 kHz grid does not resolve the <=2 kHz FWHM either; the same estimator is applied to the
COMSOL data at the same relative offsets).

Paper quantities (Figs. 2, 3, S1, S2, S3), same formulas for axionFEM and COMSOL:
  Q_B = w (dX/dw + |X|/w) / (2 (R + r)),  r = w L_MQS / Q_MQS   (FWHM of the tuned response; reduces to the note's
        lossless Q_B = Q_MQS (dX/dw+|X|/w)/(2 L_MQS) for R = 0)
  g/g_MQS: scan-integrated SNR^2 = int dd |V|^4 / N(d)^2 with N = 2 T r + S_V + S_I |Z_tot|^2, S_V S_I = (eta w/2)^2,
        S_I = eta w/(2 R_tot) (readout tuned to minimise the on-resonance noise), T = 20 mK; g/g_MQS =
        (FOM_MQS/FOM)^(1/4) (V_MQS = w Psi(5 MHz), L_MQS = X(5 MHz)/w).  This reduces exactly to the note's lossless
        formula (w Psi_MQS/|V|)(Q_B/Q_MQS)^(1/4) for R = 0, but for the lossy case it is a CANDIDATE: through it, the
        COMSOL tables reproduce the paper's Fig. S3 single-pole reach/DFSZ only to ~4 % (eta_A = 20) and ~20 %
        (eta_A = 1) (median), so the paper's lossy noise model is not identified exactly.  The axionFEM-vs-COMSOL
        comparison through it is formula independent.

    python compare_toroid_epoxy_comsol.py --comsol DIR --runs RUN [RUN ...] [--control LABEL=DIR[,DIR] ...]
Writes axionfem_results/toroid_small_shield_epoxy_vs_comsol.json, axionfem_results/<paper figure>_epoxy.json and
build/toroid_small_shield_epoxy_vs_comsol.pdf.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ZREF, Q_MQS, T_LC_MK = 50.0, 1e6, 20.0
MK_PER_MHZ = 0.0479924            # h f / k_B
NEV_PER_MHZ = 1 / 0.241798935     # m_a = 2 pi hbar f
TAG = "toroid_small_shield_epoxy_vs_comsol"


def load_comsol(d: Path):
    zf = np.load(d / "source_impedance.npy", allow_pickle=False)
    vf = np.load(d / "source_voltage.npy", allow_pickle=False)
    assert np.array_equal(zf[0], vf[0])
    f, z = zf[0] * 1e6, zf[1] + 1j * zf[2]
    voc = (vf[1] + 1j * vf[2]) * (z + ZREF) / ZREF
    return dict(f=f, Z=z, Voc=voc)


def interp_c(f0, f, y):
    out = np.interp(f0, f, y.real) + 1j * np.interp(f0, f, y.imag)
    return np.where((f0 < f[0]) | (f0 > f[-1]), np.nan + 0j, out)


def load_rows(dirs):
    rows, info = [], []
    for d in dirs:
        d = Path(d)
        if not (d / "results.json").is_file():
            continue
        r = json.loads((d / "results.json").read_text())
        for s in r["fullwave"].get("sweep", []):
            sol = s.get("solver") or {}
            W = (s.get("energy_total_J_per_A2") or {}).get("re")
            rows.append(dict(f=s["f_Hz"], Z=complex(s["Z_re_ohm"], s["Z_im_ohm"]), Psi=complex(s["Psi_re"], s["Psi_im"]),
                             dX=4 * float(W[0][0]) if W else None, R_bulk=s.get("R_bulk_ohm"),
                             accepted=bool(s.get("validation_accepted", sol.get("accepted", False))),
                             residual=sol.get("relative_residual"), iters=sol.get("gmres_iterations"), run=str(d)))
        info.append(dict(run=str(d), dofs=r["mesh"].get("dofs")))
    rows.sort(key=lambda s: s["f"])
    return rows, info


TAN_DELTA, PERT_STEP = 0.0021, 0.01


def load_pert(m_dirs, o_dirs, p_dirs):
    """First-order loss from LOSSLESS direct (STRUMPACK) solves at eps' (1 - s, 1, 1 + s) in the TACHYON region.

    Reciprocity: a change d(eps) of the epoxy permittivity changes the port impedance by dZ = -j w d(eps) int E.E / I^2
    (COMSOL sign).  A real change d(eps') = s eps' gives dX; the loss d(eps) = -j eps' tan_delta gives
    dR = eps' tan_delta dX/d(eps') = tan_delta (X(+s) - X(-s)) / (2 s).  X, V_oc and dX/dw are the lossless values at
    eps'; all corrections are O(tan_delta^2), valid a few line widths (f_p/Q) away from a pole."""
    def by_f(dirs):
        rows, _ = load_rows(dirs)
        return {round(s["f"]): s for s in rows if s["accepted"]}
    m, o, p = by_f(m_dirs), by_f(o_dirs), by_f(p_dirs)
    out = []
    for f in sorted(set(m) & set(o) & set(p)):
        s = dict(o[f])
        xm, xp = -m[f]["Z"].imag, -p[f]["Z"].imag
        s["Z"] = complex(TAN_DELTA * (xp - xm) / (2 * PERT_STEP), o[f]["Z"].imag)
        s["method"] = "perturbative (lossless STRUMPACK at eps' and eps'(1+-1%), first-order loss)"
        out.append(s)
    return out


def comsol_poles(c, fmax):
    f, x = c["f"], c["Z"].imag
    s = np.sign(x)
    out = []
    for k in np.nonzero(s[:-1] != s[1:])[0]:
        if f[k] > fmax:
            break
        if s[k] > 0 > s[k + 1] and min(abs(x[k]), abs(x[k + 1])) > 1e4:      # strong, port-coupled poles only
            y0, y1 = 1 / x[k], 1 / x[k + 1]
            out.append(f[k] + (f[k + 1] - f[k]) * y0 / (y0 - y1))
    return np.array(out)


def tail_q(f, Z, fp, dmin=3e-4, dmax=1.2e-2):
    """Lorentzian-tail Q and R_p from points with dmin <= |f/fp-1| <= dmax (X = Im Z in the COMSOL sign)."""
    d = f / fp - 1
    m = (np.abs(d) >= dmin) & (np.abs(d) <= dmax) & (Z.real > 0)
    q = np.abs(Z.imag[m]) / (2 * np.abs(d[m]) * Z.real[m])
    rp = Z.imag[m] ** 2 / Z.real[m]
    return q, rp, d[m]


def axf_poles(rows):
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        xa, xb = -a["Z"].imag, -b["Z"].imag
        if xa > 0 > xb and min(abs(xa), abs(xb)) > 100:
            fp = a["f"] + (b["f"] - a["f"]) * (1 / xa) / (1 / xa - 1 / xb)
            out.append(dict(f_Hz=fp, bracket_Hz=[a["f"], b["f"]]))
    return out


def fom(R, Xp, r, eta, f):
    """Scan-integrated SNR^2 (up to |V|^4 and constants), readout tuned for minimum on-resonance noise."""
    wmk = f / 1e6 * MK_PER_MHZ
    s = eta * wmk / (2 * R)
    A = 2 * T_LC_MK * r + (eta * wmk / 2) ** 2 / s + s * R * R
    return A ** -1.5 * (s * Xp * Xp) ** -0.5


def paper_quantities(f, X, R, Xp, absV, L_ref, psi_ref, eta):
    f, X, R, Xp, absV = map(np.asarray, (f, X, R, Xp, absV))
    w = 2 * np.pi * f
    r = w * L_ref / Q_MQS
    qb = w * Xp / (2 * (R + r))
    g = (w * psi_ref / absV) * (fom(r, 2 * L_ref, r, eta, f) / fom(R + r, Xp, r, eta, f)) ** 0.25
    return qb, g


def paper_curve(stem, label=None):
    return json.loads((HERE / "paper_curves" / f"{stem}.json").read_text())


def single_pole_over_dfsz(rec):
    """Fig. 2/S2/S3: the 'Single Pole' regions' lower edge divided by the DFSZ line (= g/g_MQS, MQS-DFSZ scan)."""
    p = rec["panels"][0]
    lines = [c for c in p["curves"] if "'width': 2.0" in str(c.get("style"))]
    dl = min(lines, key=lambda c: min(c["y"]))
    k = np.polyfit(np.log(dl["x"]), np.log(dl["y"]), 1)
    xs, gs = [], []
    for reg in p["regions"]:
        if "0.8039" not in str(reg["style"]):
            continue
        x, y = np.array(reg["x"]), np.array(reg["y"])
        m = y < 0.5
        xs += list(x[m]); gs += list(y[m] / np.exp(np.polyval(k, np.log(x[m]))))
    o = np.argsort(xs)
    return np.array(xs)[o] / NEV_PER_MHZ * 1e6, np.array(gs)[o], k


def plot(path, c, tab, ctrl, poles, fmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MAROON, GRAY = "#800000", "0.5"
    fig, ax = plt.subplots(4, 1, figsize=(6.4, 10.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 3, 2]))
    m = c["f"] <= fmax
    fc = c["f"][m] / 1e6
    ax[0].plot(fc, c["Z"].imag[m], color="k", lw=0.8, label="COMSOL (2022), 19% TACHYON")
    ax[0].set_yscale("symlog", linthresh=10)
    ax[0].set_ylim(-1e5, 1e5)
    ax[0].set_ylabel(r"Im $Z = X$ [$\Omega$]")
    ax[1].plot(fc, c["Z"].real[m], color="k", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"Re $Z$ [$\Omega$]")
    ax[2].plot(fc, np.abs(c["Voc"][m]), color="k", lw=0.8)
    ax[2].set_yscale("log")
    ax[2].set_ylabel(r"$|V_{\rm oc}|$ [V]  ($J_e=1$ A/m$^2$)")
    sets = [(tab, MAROON, "o", True, "axionFEM")] + [(t, GRAY, "s", False, f"axionFEM, {lab}") for lab, t in ctrl.items()]
    for t, col, mk, fill, lab in sets:
        if not t:
            continue
        ff = np.array([x["f_Hz"] for x in t]) / 1e6
        kw = dict(color=col, ms=3.2, ls="none", mfc=col if fill else "none")
        ax[0].plot(ff, [x["X_axf_ohm"] for x in t], mk, label=lab, **kw)
        ax[1].plot(ff, [x["R_axf_ohm"] for x in t], mk, **kw)
        ax[2].plot(ff, [x["absVoc_axf_V"] for x in t], mk, **kw)
        ax[3].plot(ff, np.abs([x["rel_X"] for x in t]), mk, **kw)
        ax[3].plot(ff, np.abs([x["rel_R"] for x in t]), "D", color=col, ms=2.8, ls="none", mfc="none")
        ax[3].plot(ff, np.abs([x["rel_absVoc"] for x in t]), "^", color=col, ms=3.2, ls="none", mfc="none")
    ax[3].plot([], [], "o", color="k", ms=3, ls="none", label=r"$X$")
    ax[3].plot([], [], "D", color="k", ms=3, ls="none", mfc="none", label=r"Re $Z$")
    ax[3].plot([], [], "^", color="k", ms=3, ls="none", mfc="none", label=r"$|V_{\rm oc}|$")
    ax[3].set_yscale("log")
    ax[3].set_ylabel("|axionFEM/COMSOL$-$1|")
    ax[3].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    for a in ax:
        for p in poles:
            a.axvline(p / 1e6, color=GRAY, lw=0.5, ls=":")
    ax[3].set_xlabel("frequency [MHz]")
    ax[3].set_xlim(0, fmax / 1e6)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.savefig(path)
    plt.close(fig)


def assumed10(dirs):
    """Paper Figs. 2, 3, S1, S2 (10 % lossy) from the ASSUMED-geometry 10 % replica (toroid_small_shield_epoxy10.json;
    no COMSOL data exist for it): Q_B and g/g_MQS through the same formulas, compared with the paper's curves."""
    rows, info = load_rows(dirs)
    ta = [s for s in rows if s["accepted"] and s["dX"] and s["f"] <= 150.1e6]
    f = np.array([s["f"] for s in ta])
    X = np.array([-s["Z"].imag for s in ta])
    R = np.array([s["Z"].real for s in ta])
    V = np.array([2 * np.pi * s["f"] * abs(s["Psi"]) for s in ta])
    Xp = np.array([s["dX"] for s in ta]) + np.abs(X) / (2 * np.pi * f)
    w0 = 2 * np.pi * f[0]
    ok = Xp > 0
    rec, cmp_ = {}, {}
    for eta, stems in ((20, ("Relative_Sensitivity_Eta20", "A_Toroid_Projection_Eta20")),
                       (1, ("Relative_Sensitivity_Eta1", "Toroid_Projection_Eta1"))):
        qb, g = paper_quantities(f, X, R, Xp, V, X[0] / w0, V[0] / w0, eta)
        pr = paper_curve(stems[0])
        ref = [[c for c in p["curves"] if c.get("label") == "Lossy"][0] for p in pr["panels"]]
        rg = np.interp(f[ok], np.array(ref[0]["x"]) * 1e6, ref[0]["y"])
        rq = np.interp(f[ok], np.array(ref[1]["x"]) * 1e6, ref[1]["y"])
        xs, gs, kd = single_pole_over_dfsz(paper_curve(stems[1]))
        cap = ok & (X > 0) & (f >= xs.min()) & (f <= xs.max())
        cmp_[f"eta{eta}"] = dict(QB_vs_paper_Fig3_lossy_median_rel=float(np.median(qb[ok] / rq - 1)),
                                 g_vs_paper_Fig3_lossy_median_abs_log=float(np.median(np.abs(np.log(g[ok] / rg)))),
                                 single_pole_vs_paper_median_abs_log=float(np.median(np.abs(np.log(
                                     g[cap] / np.interp(f[cap], xs, gs))))) if cap.any() else None)
        lab = "axionFEM, ASSUMED 10% geometry"
        rec[stems[0]] = dict(panels=[dict(curves=[dict(x=(f[ok] / 1e6).tolist(), y=g[ok].tolist(), label=lab)]),
                                     dict(curves=[dict(x=(f[ok] / 1e6).tolist(), y=qb[ok].tolist(), label=lab)])])
        m = f[cap] / 1e6 * NEV_PER_MHZ
        rec[stems[1]] = dict(panels=[dict(curves=[dict(x=m.tolist(), y=(g[cap] * np.exp(np.polyval(kd, np.log(m)))).tolist(),
                                                       label="axionFEM single pole, ASSUMED 10% geometry")])])
    for stem, d in rec.items():
        d.update(source="compare_toroid_epoxy_comsol.py --assumed10 (axionFEM, ASSUMED 1.125 cm / 9.75% lining)")
        (HERE / "axionfem_results" / f"{stem}_epoxy10.json").write_text(json.dumps(d, indent=1) + "\n")
    out = dict(what="axionFEM, ASSUMED-geometry 10% TACHYON lining (no COMSOL data) vs the paper's 10% curves",
               runs=info, n=len(rows), n_accepted=len(ta),
               L_5MHz_H=float(X[0] / w0), comparison=cmp_,
               sweep=[dict(f_Hz=float(a), X_ohm=float(b), R_ohm=float(c), absVoc_V=float(d)) for a, b, c, d in zip(f, X, R, V)])
    (HERE / "axionfem_results" / "toroid_small_shield_epoxy10_assumed.json").write_text(json.dumps(out, indent=1) + "\n")
    print("assumed 10%:", json.dumps(cmp_))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comsol", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--control", nargs="*", default=[], metavar="LABEL=DIR[,DIR]")
    ap.add_argument("--fmax", type=float, default=150e6)
    ap.add_argument("--pert", nargs=3, metavar=("MINUS", "NOMINAL", "PLUS"), default=None,
                    help="comma-separated run dirs of the lossless eps'(1-s), eps', eps'(1+s) STRUMPACK solves")
    ap.add_argument("--assumed10", nargs="*", default=[], help="run dirs of the ASSUMED 10%% geometry replica")
    a = ap.parse_args()
    if a.assumed10:
        assumed10(a.assumed10)
    c = load_comsol(a.comsol)
    rows, info = load_rows(a.runs)
    for s in rows:
        s.setdefault("method", "AMS-GMRES, lossy operator")
    pert, pert_check = [], []
    if a.pert:
        pert = load_pert(*[d.split(",") for d in a.pert])
        ams = {round(s["f"]): s for s in rows if s["accepted"]}
        for s in pert:
            q = ams.get(round(s["f"]))
            if q:   # validation of the perturbative method against the direct lossy solve
                pert_check.append(dict(f_Hz=s["f"], dX_rel=s["Z"].imag / q["Z"].imag - 1, dR_rel=s["Z"].real / q["Z"].real - 1,
                                       dV_rel=abs(s["Psi"]) / abs(q["Psi"]) - 1))
            else:
                rows.append(s)
        rows.sort(key=lambda s: s["f"])
    rows = [s for s in rows if s["f"] <= a.fmax * 1.001]
    rejected = sorted(str(Path(d).name) for d in a.runs if (Path(d) / "fullwave_rejected.json").is_file()
                      and not (Path(d) / "results.json").is_file())
    cp = comsol_poles(c, a.fmax * 1.001)
    v0 = interp_c(np.array([rows[0]["f"]]), c["f"], c["Voc"])[0]
    psi_sign = float(np.sign(rows[0]["Psi"].real * (v0 / 1j).real)) or 1.0

    def table(rr):
        out = []
        for s in rr:
            f, w = s["f"], 2 * np.pi * s["f"]
            zc = interp_c(np.array([f]), c["f"], c["Z"])[0]
            vc = interp_c(np.array([f]), c["f"], c["Voc"])[0]
            x, rz, va = -s["Z"].imag, s["Z"].real, w * abs(s["Psi"])
            out.append(dict(f_Hz=f, X_axf_ohm=x, X_comsol_ohm=zc.imag, rel_X=(x - zc.imag) / abs(zc.imag),
                            R_axf_ohm=rz, R_comsol_ohm=zc.real, rel_R=(rz - zc.real) / zc.real,
                            R_bulk_axf_ohm=s["R_bulk"], absVoc_axf_V=va, absVoc_comsol_V=abs(vc),
                            rel_absVoc=(va - abs(vc)) / abs(vc), Psi_ratio=psi_sign * s["Psi"].real / (vc / (1j * w)).real,
                            dX_energy_H=s["dX"], accepted=s["accepted"], residual=s["residual"], gmres_iterations=s["iters"],
                            method=s.get("method", "AMS-GMRES, lossy operator"),
                            run=s["run"]))
        return out

    tab = table(rows)
    acc = [t for t in tab if t["accepted"]]
    # finite-difference check of the energy-identity dX/dw on close pairs (|df/f| < 2e-3)
    fd = []
    for p, q in zip(acc[:-1], acc[1:]):
        if (q["f_Hz"] - p["f_Hz"]) / p["f_Hz"] < 2e-3 and p["dX_energy_H"] and q["dX_energy_H"]:
            d = (q["X_axf_ohm"] - p["X_axf_ohm"]) / (2 * np.pi * (q["f_Hz"] - p["f_Hz"]))
            fd.append(dict(f_Hz=0.5 * (p["f_Hz"] + q["f_Hz"]), fd_H=d,
                           energy_H=0.5 * (p["dX_energy_H"] + q["dX_energy_H"])))
    ctrl, disc = {}, []
    for spec in a.control:
        lab, dirs = spec.split("=", 1)
        crow, _ = load_rows(dirs.split(","))
        ctrl[lab] = table(crow)
        for t in ctrl[lab]:
            b = [b for b in tab if abs(b["f_Hz"] - t["f_Hz"]) < 1]
            if b:
                b = b[0]
                disc.append(dict(label=lab, f_Hz=t["f_Hz"], dX_rel=(t["X_axf_ohm"] - b["X_axf_ohm"]) / abs(b["X_axf_ohm"]),
                                 dR_rel=(t["R_axf_ohm"] - b["R_axf_ohm"]) / b["R_axf_ohm"],
                                 dV_rel=(t["absVoc_axf_V"] - b["absVoc_axf_V"]) / b["absVoc_axf_V"]))
    # poles and Q
    ap_ = axf_poles([s for s in rows if s["accepted"]])
    fa = np.array([t["f_Hz"] for t in acc])
    za = np.array([complex(t["R_axf_ohm"], t["X_axf_ohm"]) for t in acc])
    pole_tab = []
    for p in ap_:
        fc_ = cp[np.argmin(abs(cp - p["f_Hz"]))] if len(cp) else np.nan
        q_a, rp_a, d_a = tail_q(fa, za, p["f_Hz"])
        # COMSOL at the same relative offsets from its own pole
        zc = interp_c(fc_ * (1 + d_a), c["f"], c["Z"])
        q_c = np.abs(zc.imag) / (2 * np.abs(d_a) * zc.real)
        q_c_all, rp_c_all, _ = tail_q(c["f"], c["Z"], fc_, 1e-3, 1.2e-2)
        pole_tab.append(dict(f_axf_Hz=p["f_Hz"], bracket_Hz=p["bracket_Hz"], f_comsol_Hz=float(fc_),
                             rel=(p["f_Hz"] - fc_) / fc_, n_tail_points=int(len(q_a)), tail_offsets=d_a.tolist(),
                             Q_axf=float(np.median(q_a)) if len(q_a) else None,
                             Q_axf_points=q_a.tolist(), Q_comsol_same_offsets=q_c.tolist(),
                             Q_comsol=float(np.median(q_c)) if len(q_c) else float(np.median(q_c_all)),
                             Rp_axf_ohm=float(np.median(rp_a)) if len(rp_a) else None,
                             Rp_comsol_ohm=float(np.median(rp_c_all))))
    far = [t for t in acc if all(abs(t["f_Hz"] - p) / p > 0.02 for p in cp)]
    med = lambda k, L: float(np.median([abs(t[k]) for t in L])) if L else None   # noqa: E731
    stats = dict(n=len(tab), n_accepted=len(acc), median_abs_rel_X=med("rel_X", acc), median_abs_rel_R=med("rel_R", acc),
                 median_abs_rel_Voc=med("rel_absVoc", acc),
                 far_from_poles=dict(n=len(far), median_abs_rel_X=med("rel_X", far), median_abs_rel_R=med("rel_R", far),
                                     median_abs_rel_Voc=med("rel_absVoc", far),
                                     max_abs_rel_X=max([abs(t["rel_X"]) for t in far], default=None),
                                     max_abs_rel_R=max([abs(t["rel_R"]) for t in far], default=None),
                                     max_abs_rel_Voc=max([abs(t["rel_absVoc"]) for t in far], default=None)))
    # paper quantities: COMSOL (dense, numerical dX/dw) and axionFEM (energy-identity dX/dw), same formulas
    mc = c["f"] <= a.fmax * 1.01
    fcs, Xc, Rc, Vc = c["f"][mc], c["Z"].imag[mc], c["Z"].real[mc], np.abs(c["Voc"][mc])
    wc = 2 * np.pi * fcs
    Xpc = np.gradient(Xc, wc) + np.abs(Xc) / wc
    t0 = min(acc, key=lambda t: t["f_Hz"])
    w0 = 2 * np.pi * t0["f_Hz"]
    ta = [t for t in acc if t["dX_energy_H"]]
    fa2 = np.array([t["f_Hz"] for t in ta])
    Xpa = np.array([t["dX_energy_H"] + abs(t["X_axf_ohm"]) / (2 * np.pi * t["f_Hz"]) for t in ta])
    figs, check = {}, {}
    for eta, stems in ((20, ("Relative_Sensitivity_Eta20", "Lossier_Toroid_Projection_Eta20")),
                       (1, ("Relative_Sensitivity_Eta1", "Lossier_Toroid_Projection_Eta1"))):
        qb_c, g_c = paper_quantities(fcs, Xc, Rc, Xpc, Vc, Xc[0] / wc[0], Vc[0] / wc[0], eta)
        qb_a, g_a = paper_quantities(fa2, [t["X_axf_ohm"] for t in ta], [t["R_axf_ohm"] for t in ta], Xpa,
                                     [t["absVoc_axf_V"] for t in ta], t0["X_axf_ohm"] / w0, t0["absVoc_axf_V"] / w0, eta)
        gi = np.interp(fa2, fcs, g_c)
        qi = np.interp(fa2, fcs, qb_c)
        ok = Xpa > 0
        rec = paper_curve(stems[1])
        xs, gs, kd = single_pole_over_dfsz(rec)
        sel = (xs > fcs[0]) & (xs < fcs[-1])
        dlog = np.log(np.interp(xs[sel], fcs, g_c) / gs[sel])
        check[f"eta{eta}"] = dict(comsol_formula_vs_paper_S3_median_log=float(np.median(dlog)),
                                  comsol_formula_vs_paper_S3_median_abs_log=float(np.median(np.abs(dlog))))
        comp = dict(axf_vs_comsol_QB_median_abs_rel=float(np.median(np.abs(qb_a[ok] / qi[ok] - 1))),
                    axf_vs_comsol_g_median_abs_rel=float(np.median(np.abs(g_a[ok] / gi[ok] - 1))))
        check[f"eta{eta}"].update(comp)
        figs[stems[0]] = dict(panels=[
            dict(curves=[dict(x=(fa2[ok] / 1e6).tolist(), y=g_a[ok].tolist(), marker="o", ls="none",
                              label="axionFEM, 19% TACHYON (paper curve: 10%)")]),
            dict(curves=[dict(x=(fa2[ok] / 1e6).tolist(), y=qb_a[ok].tolist(), marker="o", ls="none",
                              label="axionFEM, 19% TACHYON (paper curve: 10%)")])])
        # S3: single-pole reach = DFSZ x g/g_MQS, in the projection's (m_a [neV], g [GeV^-1]) units
        cap = ok & (np.array([t["X_axf_ohm"] for t in ta]) > 0)   # single pole: capacitive tuning only (X > 0)
        mnev = fa2[cap] / 1e6 * NEV_PER_MHZ
        figs[stems[1]] = dict(panels=[dict(curves=[dict(x=mnev.tolist(),
                                                        y=(g_a[cap] * np.exp(np.polyval(kd, np.log(mnev)))).tolist(),
                                                        marker="o", ls="none",
                                                        label="axionFEM single pole (DFSZ x g/g_MQS)")])])
    out = dict(
        what="axionFEM vs COMSOL (2022), shielded LOSSY toroid of arXiv:2211.00008 (Toroid_With_Epoxy/Small_Shield.mph, "
             "19.0% TACHYON 100G lining)",
        conventions="COMSOL exp(+jwt): X = Im Z = +w L; axionFEM Z conjugated. R = Re Z. V_oc = Vport (Zport+50)/50. "
                    "rel = (axionFEM - COMSOL)/|COMSOL|. dX/dw (axionFEM) from the energy identity (O(tan_delta) "
                    "approximation for the lossy operator).",
        comsol=dict(L_5MHz_H=float(c["Z"].imag[0] / (2 * np.pi * c["f"][0])), R_5MHz_ohm=float(c["Z"].real[0]),
                    strong_poles_Hz=cp.tolist()),
        axionfem_runs=info, psi_orientation_sign=psi_sign, summary=stats, sweep=tab, control=ctrl, discretization=disc,
        poles=pole_tab, energy_identity_vs_finite_difference=fd, perturbative_vs_direct_lossy=pert_check,
        rejected_runs=rejected, paper_quantities=check,
        paper_formalism=paper_quantities.__doc__ or "see module docstring")
    (HERE / "axionfem_results").mkdir(exist_ok=True)
    jp = HERE / "axionfem_results" / f"{TAG}.json"
    jp.write_text(json.dumps(out, indent=1, default=float) + "\n")
    for stem, d in figs.items():
        d.update(source="compare_toroid_epoxy_comsol.py (axionFEM, shielded toroid with 19% TACHYON lining)",
                 formalism=__doc__.split("Paper quantities")[1].split("python compare")[0])
        (HERE / "axionfem_results" / f"{stem}_epoxy.json").write_text(json.dumps(d, indent=1) + "\n")
    (HERE / "build").mkdir(exist_ok=True)
    plot(HERE / "build" / f"{TAG}.pdf", c, tab, ctrl, cp, a.fmax)
    print(f"wrote {jp}")
    print(f"L(5 MHz): COMSOL {out['comsol']['L_5MHz_H']*1e9:.2f} nH, axionFEM {t0['X_axf_ohm']/w0*1e9:.2f} nH")
    print(" f[MHz]     X_axf      X_com    relX      R_axf      R_com     relR     |V|axf  |V|com   relV   res  acc")
    for t in tab:
        print(f"{t['f_Hz']/1e6:9.4f} {t['X_axf_ohm']:10.3f} {t['X_comsol_ohm']:10.3f} {t['rel_X']:+.2e} {t['R_axf_ohm']:.3e} "
              f"{t['R_comsol_ohm']:.3e} {t['rel_R']:+.2e} {t['absVoc_axf_V']:8.3f} {t['absVoc_comsol_V']:8.3f} "
              f"{t['rel_absVoc']:+.2e} {t['residual'] or 0:.0e} {t['accepted']}")
    print("summary:", json.dumps(stats))
    for d in disc:
        print(f"control {d['label']} @ {d['f_Hz']/1e6:.3f}: dX {d['dX_rel']:+.2e} dR {d['dR_rel']:+.2e} dV {d['dV_rel']:+.2e}")
    for p in pole_tab:
        print(f"pole axionFEM {p['f_axf_Hz']/1e6:.4f} COMSOL {p['f_comsol_Hz']/1e6:.4f} rel {p['rel']:+.2e}  "
              f"Q axf {p['Q_axf']} COMSOL {p['Q_comsol']:.3g} (n={p['n_tail_points']})  R_p axf {p['Rp_axf_ohm']} COMSOL {p['Rp_comsol_ohm']:.3g}")
    for e in pert_check:
        print(f"perturbative vs direct lossy @ {e['f_Hz']/1e6:.3f}: dX {e['dX_rel']:+.2e} dR {e['dR_rel']:+.2e} dV {e['dV_rel']:+.2e}")
    print("rejected (unconverged) direct lossy runs:", rejected)
    for e in fd:
        print(f"dX/dw @ {e['f_Hz']/1e6:.3f}: finite diff {e['fd_H']:.4e}  energy {e['energy_H']:.4e}")
    print("paper quantities:", json.dumps(check))


if __name__ == "__main__":
    main()
