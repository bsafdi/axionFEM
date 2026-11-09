"""Validation-note section for the shielded LOSSY toroid (compare_toroid_epoxy_comsol.py); used by make_note.py."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAG = "toroid_small_shield_epoxy_vs_comsol"

LOSSY_NOTE = (r"Lossy overlay (black squares, ``19\% TACHYON''): axionFEM replica of the recovered COMSOL lossy model, which "
              r"holds 19\% TACHYON (the paper's 20\% case of Fig.~S3), not the 10\% of this figure's grey ``Lossy'' "
              r"curve; its $Q_B$ is accordingly $\approx$0.53$\times$ the 10\% curve. $Q_B=\omega(dX/d\omega+|X|/\omega)/"
              r"(2(R+r))$; $g/g_{\rm MQS}$ through a candidate lossy noise model (see the lossy-toroid section), "
              r"tentative.")
S3_NOTE = (r"axionFEM (black squares): single-pole reach $= g_{\rm DFSZ}\times g/g_{\rm MQS}$ of the 19\% TACHYON replica "
           r"(the scan collects data until the MQS DFSZ sensitivity is reached; the paper's single-pole edge divided by its "
           r"DFSZ line equals its Fig.~3 $g/g_{\rm MQS}$). $g/g_{\rm MQS}$ uses a candidate lossy noise model, tentative: "
           r"through it the COMSOL tables themselves reproduce this edge only to the accuracy quoted in the lossy-toroid "
           r"section. The multipole curve is not reproduced.")
TEN_NOTE = (r"Grey triangles: axionFEM with an ASSUMED 10\% geometry (the same lining at half thickness, 1.125\,cm, "
            r"9.75\% of the interior; the paper's 10\% COMSOL model was not provided), through the same candidate "
            r"formulas (tentative); single pole = DFSZ $\times$ $g/g_{\rm MQS}$ at capacitive tunings ($X>0$).")
NOTES = {"A_Toroid_Projection_Eta20": TEN_NOTE, "Toroid_Projection_Eta1": TEN_NOTE,
         "Relative_Sensitivity_Eta20": LOSSY_NOTE + " " + TEN_NOTE, "Relative_Sensitivity_Eta1": LOSSY_NOTE + " " + TEN_NOTE,
         "Lossier_Toroid_Projection_Eta20": S3_NOTE, "Lossier_Toroid_Projection_Eta1": S3_NOTE}


def section(results: Path, build: Path) -> str:
    jp, pp = results / f"{TAG}.json", build / f"{TAG}.pdf"
    if not (jp.is_file() and pp.is_file()):
        return "Lossy toroid vs COMSOL: pending (run \\texttt{compare\\_toroid\\_epoxy\\_comsol.py}).\n"
    r = json.loads(jp.read_text())
    sm, cm = r["summary"], r["comsol"]
    s0 = min(r["sweep"], key=lambda t: t["f_Hz"])
    fr = sm["far_from_poles"]
    prov = sm["n_accepted"] < sm["n"]
    rows = [f"{p['f_comsol_Hz']/1e6:.3f} & {p['f_axf_Hz']/1e6:.3f} & {100*p['rel']:+.3f} & "
            f"{p['Q_comsol']:.3g} & {(p['Q_axf'] or float('nan')):.3g} & {p['n_tail_points']} \\\\" for p in r["poles"]]
    disc = "; ".join(f"{d['f_Hz']/1e6:.1f}\\,MHz: $\\Delta X/X={100*d['dX_rel']:+.2f}\\%$, "
                     f"$\\Delta R/R={100*d['dR_rel']:+.2f}\\%$, $\\Delta|V|/|V|={100*d['dV_rel']:+.2f}\\%$"
                     for d in r["discretization"]) or "pending"
    pq = r["paper_quantities"]
    dofs = r["axionfem_runs"][0].get("dofs") or 0
    return "\n".join([
        "\\section{Shielded lossy toroid (TACHYON 100G): axionFEM vs the original COMSOL data}",
        "Recovered COMSOL model \\texttt{Toroid\\_With\\_Epoxy/Small\\_Shield.mph} (settings: "
        "\\texttt{MODEL\\_SETTINGS\\_epoxy.md}): the lossless model plus a TACHYON 100G lining, a hollow square 2.25\\,cm "
        "thick inside the walls (19.0\\% of the interior; the paper's ``$\\approx$20\\%'' case of Fig.~S3 --- the 10\\% "
        "model of Figs.~2, 3, S1, S2 is not in the archive), $\\varepsilon'=3.0533-6.67\\times10^{-12}f$, "
        "$\\tan\\delta=0.0021$. Replica: \\texttt{designs/toroid\\_small\\_shield\\_epoxy.json} (lossless replica + the "
        f"lining), order 2, {dofs:,} DOFs, AMS-GMRES, one frequency per job, {sm['n']} frequencies "
        f"({sm['n_accepted']} accepted by the solver gate).",
        f"$L(5\\,{{\\rm MHz}})$: axionFEM {s0['X_axf_ohm']/(2*3.141592653589793*s0['f_Hz'])*1e9:.1f}\\,nH, COMSOL "
        f"{cm['L_5MHz_H']*1e9:.1f}\\,nH (the lossless COMSOL file: 775.2\\,nH; the dielectric cannot change $L$ there, so "
        "$\\approx$1\\% is COMSOL's own mesh difference between its two files). "
        f"Median over all points: $|\\Delta X/X|$ {100*sm['median_abs_rel_X']:.2f}\\%, $|\\Delta R/R|$ "
        f"{100*sm['median_abs_rel_R']:.2f}\\%, $|\\Delta|V_{{\\rm oc}}|/|V_{{\\rm oc}}||$ {100*sm['median_abs_rel_Voc']:.2f}\\%; "
        f"more than 2\\% from a pole ({fr['n']} points): medians {100*(fr['median_abs_rel_X'] or 0):.2f} / "
        f"{100*(fr['median_abs_rel_R'] or 0):.2f} / {100*(fr['median_abs_rel_Voc'] or 0):.2f}\\%, maxima "
        f"{100*(fr['max_abs_rel_X'] or 0):.1f} / {100*(fr['max_abs_rel_R'] or 0):.1f} / {100*(fr['max_abs_rel_Voc'] or 0):.1f}\\%. "
        f"Mesh control (finer mesh, same order): {disc}." + (" Points rejected by the solver gate are provisional." if prov else ""),
        "",
        "Poles of $Z$ and their quality factors (Lorentzian tails, $Q=|X|/(2|f/f_p-1|R)$, at the axionFEM points and at "
        "the same relative offsets in the COMSOL data; the $\\le$2\\,kHz line widths are below both the COMSOL grid and "
        "the axionFEM pole-position accuracy):",
        "\\begin{center}\\begin{tabular}{rrrrrr}\\toprule",
        "COMSOL pole [MHz] & axionFEM [MHz] & diff [\\%] & $Q$ COMSOL & $Q$ axionFEM & tail points \\\\ \\midrule", *rows,
        "\\bottomrule\\end{tabular}\\end{center}",
        "In the lossless replica all poles matched COMSOL to $\\le$0.1\\%; with the lining, mode 1 still matches "
        "(+0.04\\%) but modes 2, 3 and 4 do not: the epoxy shifts them by $-1.5$\\% / $-1.2$\\% / $-1.0$\\% in COMSOL "
        "and by only $-0.6$\\% / $-0.3$\\% / $-0.4$\\% in axionFEM (pole 3 bracketed by 115.72 and 116.3\\,MHz: "
        "116.06\\,MHz vs COMSOL's 115.03\\,MHz; pole 4 at 149.37\\,MHz vs 148.44\\,MHz, from perturbative points at "
        "149.0/149.33/150\\,MHz since the direct lossy solve at 149.33\\,MHz did not converge). The finer mesh D moves axionFEM's pole 2 by a further +0.04\\%, so this is not axionFEM "
        "discretization. UNRESOLVED; candidates: the COMSOL tables (dated 2022-05-04) were computed with a different "
        "lining than the saved \\texttt{.mph} (2022-04-25), or COMSOL's coarse epoxy mesh ($h_{\\max}$ 5.3\\,cm in a "
        "2.25\\,cm layer). Near poles 2 and 3 the point-wise differences therefore reach tens of percent. "
        "Independent check of the lossy solves: a first-order perturbative loss from three lossless STRUMPACK solves at "
        "$\\varepsilon'(1+\\{-1,0,+1\\}\\%)$, $R=\\tan\\delta\\,(X_+-X_-)/0.02$, agrees with the direct AMS-GMRES lossy "
        "solves at 17 common frequencies (incl. the near-pole 26.1--26.73 and 73--81\\,MHz points) to "
        "$\\le 2.4\\times10^{-4}$ in $X$ and $|V_{\\rm oc}|$ and $\\le 1.1\\times10^{-3}$ in $R$.",
        "Paper quantities (same formulas for both codes): $Q_B=\\omega(dX/d\\omega+|X|/\\omega)/(2(R+r))$ (FWHM "
        "definition; the note's lossless formula for $R=0$) and $g/g_{\\rm MQS}$ from the scan-integrated SNR with "
        "thermal noise of the intrinsic $r$ and an amplifier tuned for minimum on-resonance noise (reduces to the "
        "lossless formula for $R=0$). axionFEM vs COMSOL: $Q_B$ median "
        f"{100*pq['eta20']['axf_vs_comsol_QB_median_abs_rel']:.1f}\\%, $g/g_{{\\rm MQS}}$ median "
        f"{100*pq['eta20']['axf_vs_comsol_g_median_abs_rel']:.1f}\\% ($\\eta_A=20$), "
        f"{100*pq['eta1']['axf_vs_comsol_g_median_abs_rel']:.1f}\\% ($\\eta_A=1$). The lossy $g/g_{{\\rm MQS}}$ formalism "
        "is NOT identified exactly: through it the COMSOL tables reproduce the paper's Fig.~S3 single-pole edge / DFSZ "
        f"to a median $|\\log|$ of {pq['eta20']['comsol_formula_vs_paper_S3_median_abs_log']:.2f} ($\\eta_A=20$) and "
        f"{pq['eta1']['comsol_formula_vs_paper_S3_median_abs_log']:.2f} ($\\eta_A=1$) --- tentative.",
        ten_text(results),        "",
        "\\begin{center}",
        f"\\includegraphics[width=0.78\\textwidth]{{{pp.name}}}\\\\",
        "\\small COMSOL 2022 (black line) and axionFEM (maroon points; grey: refined mesh). Bottom: $|$axionFEM/COMSOL$-1|$ "
        "for $X$ (circles), Re\\,$Z$ (diamonds), $|V_{\\rm oc}|$ (triangles); dotted: COMSOL poles.",
        "\\end{center}", "\\clearpage"])


def ten_text(results: Path) -> str:
    p = results / "toroid_small_shield_epoxy10_assumed.json"
    if not p.is_file():
        return ""
    r = json.loads(p.read_text())
    c = r["comparison"]
    return ("\n\\paragraph{Assumed 10\\% geometry (Figs.~2, 3, S1, S2).} The paper's 10\\% COMSOL model is not in the "
            "archive; \\texttt{designs/toroid\\_small\\_shield\\_epoxy10.json} assumes the same lining at half thickness "
            f"(1.125\\,cm, 9.75\\%), {r['n_accepted']} accepted frequencies. Against the paper's 10\\% curves: $Q_B$ median "
            f"{100*c['eta20']['QB_vs_paper_Fig3_lossy_median_rel']:+.0f}\\%; $g/g_{{\\rm MQS}}$ median $|\\log|$ "
            f"{c['eta20']['g_vs_paper_Fig3_lossy_median_abs_log']:.2f} ($\\eta_A=20$), "
            f"{c['eta1']['g_vs_paper_Fig3_lossy_median_abs_log']:.2f} ($\\eta_A=1$). Geometry and $g$ formalism are both "
            "assumptions here, so this comparison is tentative.\n")
