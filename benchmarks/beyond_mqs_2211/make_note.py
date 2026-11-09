#!/usr/bin/env python
"""Build the validation note: arXiv:2211.00008 figures vs their axionFEM reproduction (PDF).

For every figure of the paper the note shows the original (from the arXiv source) next to a panel that redraws the
curves extracted from it (extract_paper_curves.py) and overlays axionFEM's reproduction where one exists
(``axionfem_results/<figure>.json``; see REPRO below). Output: build/beyond_mqs_2211_validation.pdf.

    python make_note.py [--src PAPER_SOURCE_DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import curves  # noqa: E402
from extract_paper_curves import extract, fetch_source  # noqa: E402
import epoxy_note  # noqa: E402

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
RESULTS = HERE / "axionfem_results"

# figure order and captions (paper numbering); "needs" = the axionFEM inputs a reproduction requires
FIGS = [
    ("Detector_Geometry_Toroid.png", "Fig. 1", "Fiducial toroid (r 0.5--1 m, h 0.5 m, 5$^\\circ$ gap, readout bridge); "
     "Poynting flux at 50 MHz, unshielded.", "3D model render; far-field Poynting flux at 50 MHz (radiation boundary)"),
    ("A_Toroid_Projection_Eta20", "Fig. 2", "Projected reach of the toroid, shielded + 10\\% lossy, $\\eta_A=20$: MQS "
     "scan, single-pole and multipole strategies.", "Z(f), V(f) shielded+lossy, 5--1000 MHz; scan formalism"),
    ("Relative_Sensitivity_Eta20", "Fig. 3", "$g/g^{\\rm MQS}$ (top) and $Q_B$ (bottom) for the lossy, lossless and "
     "unshielded toroid, $\\eta_A=20$.", "Z(f), V(f) for the three scenarios, 5--150 MHz, dense near poles"),
    ("Relative_Sensitivity_Eta1", "Fig. S1", "As Fig. 3 for $\\eta_A=1$.", "as Fig. 3"),
    ("Toroid_Projection_Eta1", "Fig. S2", "As Fig. 2 for $\\eta_A=1$.", "as Fig. 2"),
    ("Lossier_Toroid_Projection_Eta20", "Fig. S3 (left)", "As Fig. 2 with 20\\% lossy volume, $\\eta_A=20$.",
     "Z(f), V(f) shielded + 20\\% lossy"),
    ("Lossier_Toroid_Projection_Eta1", "Fig. S3 (right)", "As Fig. 2 with 20\\% lossy volume, $\\eta_A=1$.", "as above"),
    ("Detector_Geometry_Solenoid.png", "Fig. S4 (left)", "Solenoid geometry (r 0.5--1 m, h 1.5 m, 1 cm annular gap).",
     "3D model render"),
    ("Solenoid_Cross_Section.png", "Fig. S4 (right)", "Solenoid cross-section: assumed uniform axial field in the bore.",
     "source-field map"),
    ("Solenoid_Relative_Sensitivity_Eta20", "Fig. S5", "As Fig. 3 for the solenoid.", "solenoid Z(f), V(f), three scenarios"),
    ("Solenoid_Projection_Eta20", "Fig. S6 (left)", "As Fig. 2 for the solenoid, $\\eta_A=20$.", "solenoid, shielded+lossy"),
    ("Solenoid_Projection_Eta1", "Fig. S6 (right)", "As Fig. 2 for the solenoid, $\\eta_A=1$.", "as above"),
    ("Toroid_Broadband_Rescale", "Fig. S7", "Broadband readout relative to MQS, three scenarios.", "toroid Z(f), V(f); broadband formalism"),
    ("Toroid_Broadband_20", "Fig. S8 (left)", "Toroid broadband reach, $\\eta_A=20$.", "as S7"),
    ("Toroid_Broadband_1", "Fig. S8 (right)", "Toroid broadband reach, $\\eta_A=1$.", "as S7"),
    ("Solenoid_Broadband_20", "Fig. S9 (left)", "Solenoid broadband reach, $\\eta_A=20$.", "solenoid; broadband"),
    ("Solenoid_Broadband_1", "Fig. S9 (right)", "Solenoid broadband reach, $\\eta_A=1$.", "solenoid; broadband"),
]


_FIG3 = (r"axionFEM: shielded lossless toroid only (maroon points at the production sweep frequencies), from "
         r"axionFEM's $X$, $dX/d\omega$ (energy identity) and $|V_{\rm oc}|$ through the paper's resonant formalism, "
         r"identified from the paper's own curves: $Q_B = Q_{\rm MQS}(dX/d\omega + |X|/\omega)/(2L_{\rm MQS})$, "
         r"$g/g_{\rm MQS} = (\omega\Psi_{\rm MQS}/|V_{\rm oc}|)(Q_B/Q_{\rm MQS})^{1/4}$, $Q_{\rm MQS}=10^6$, "
         r"$L_{\rm MQS}$ and $\Psi_{\rm MQS}$ taken at 5\,MHz. The COMSOL data through the same formulas reproduce "
         r"the paper's lossless $g/g_{\rm MQS}$ to 0.1\% (median), for both $\eta_A$. axionFEM vs the paper: "
         r"$g/g_{\rm MQS}$ median 0.7\%, at most 3.5\% next to poles; $Q_B$ median 1\%. Near the poles the "
         r"paper's $Q_B$ is higher (up to $\times$2--4 at the peaks), for the COMSOL data as well, consistent with "
         r"a coarse finite-difference $dX/d\omega$ in the paper. axionFEM points are shown for the shielded lossless "
         r"scenario only.")
OVERLAY_NOTES = {"Relative_Sensitivity_Eta20": _FIG3, "Relative_Sensitivity_Eta1": _FIG3}
OVERLAY_NOTES["Solenoid_Relative_Sensitivity_Eta20"] = (
    r"axionFEM: shielded lossless solenoid only (maroon points at the production sweep frequencies), same formalism "
    r"as Fig.~3. The COMSOL tables through it reproduce the paper's lossless $g/g_{\rm MQS}$ to 0.1\,\% (median). "
    r"axionFEM vs the paper: $g/g_{\rm MQS}$ median 3.6\,\%, $Q_B$ median 7\,\%, dominated by points next to the "
    r"poles, which axionFEM places 0.3--1.3\,\% higher (see the solenoid direct comparison: attributed to COMSOL's "
    r"coarse mesh at the slot). Lossy and unshielded scenarios: pending.")


for _k, _v in epoxy_note.NOTES.items():  # lossy-toroid notes, appended to the figure captions
    OVERLAY_NOTES[_k] = (OVERLAY_NOTES[_k] + " " if _k in OVERLAY_NOTES else "") + _v


def overlay(stem: str, rec: dict) -> Path:
    """Redraw the extracted curves (grey) and overlay axionFEM's reproduction, if any (maroon)."""
    n = len(rec["panels"])
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.9 * n + 0.5), squeeze=False, sharex=True)
    res = json.loads((RESULTS / f"{stem}.json").read_text()) if (RESULTS / f"{stem}.json").is_file() else None
    lp = RESULTS / f"{stem}_epoxy.json"
    lossy = json.loads(lp.read_text()) if lp.is_file() else None
    lp10 = RESULTS / f"{stem}_epoxy10.json"
    lossy10 = json.loads(lp10.read_text()) if lp10.is_file() else None
    for i, p in enumerate(rec["panels"]):
        ax = axes[i, 0]
        curves.draw_panel(ax, p, alpha=0.55, gray=True)
        if res:
            for c in res.get("panels", [{}] * n)[i].get("curves", []):
                ax.plot(c["x"], c["y"], color="maroon", lw=0.9, ls=c.get("ls", "-"), marker=c.get("marker"), ms=2.5,
                        label=c.get("label"))
            if any(c.get("label") for c in res["panels"][i].get("curves", [])):
                ax.legend(fontsize=6, frameon=False)
        if lossy10:  # ASSUMED-geometry 10% lossy toroid (no COMSOL file), compare_toroid_epoxy_comsol.py --assumed10
            for c in lossy10["panels"][i].get("curves", []):
                ax.plot(c["x"], c["y"], color="0.35", lw=0, marker="^", mfc="none", ms=2.8, label=c.get("label"))
            ax.legend(fontsize=6, frameon=False)
        if lossy:  # shielded lossy (TACHYON) toroid, compare_toroid_epoxy_comsol.py
            for c in lossy["panels"][i].get("curves", []):
                ax.plot(c["x"], c["y"], color="k", lw=0.9, ls=c.get("ls", "-"), marker="s", mfc="none", ms=2.5,
                        label=c.get("label"))
            ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=7)
    xl = "$m_a$ [neV]" if rec["panels"][-1]["x"]["scale"] == "log" else "frequency [MHz]"
    axes[-1, 0].set_xlabel(xl, fontsize=8)
    fig.tight_layout()
    out = BUILD / f"overlay_{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


DIRECT = "toroid_small_shield_lossless_vs_comsol"
DIRECT_SOL = "solenoid_shielded_lossless_vs_comsol"


def direct_section() -> str:
    """Section: axionFEM vs the original COMSOL data of the shielded lossless toroid (compare_toroid_comsol.py)."""
    jp, pp = RESULTS / f"{DIRECT}.json", BUILD / f"{DIRECT}.pdf"
    if not (jp.is_file() and pp.is_file()):
        return "Direct comparison with the COMSOL data: pending (run \\texttt{compare\\_toroid\\_comsol.py}).\n"
    r = json.loads(jp.read_text())
    sm, cm = r["summary"], r["comsol"]
    s0 = min(r["sweep"], key=lambda t: t["f_Hz"])
    conv = r.get("eigenmodes_all_converged", False)
    rows = []
    for m in r["eigenmodes_matched"]:
        rows.append(f"pole & {m['f_comsol_pole_Hz']/1e6:.3f} & {m['f_axf_Hz']/1e6:.3f} & {100*m['rel']:+.3f} & "
                    f"eigenmode{'' if m['converged'] else ' (provisional)'}, port $E$-fraction {m['port_fraction']:.0e} \\\\")
    for nm, lst in (("pole", r["sweep_poles"]), ("zero", r["sweep_zeros"])):
        for e in lst:
            rows.append(f"{nm} & {e['f_comsol_Hz']/1e6:.3f} & {e['f_Hz']/1e6:.3f} & {100*e['rel']:+.3f} & sweep bracket "
                        f"{e['bracket_Hz'][0]/1e6:.1f}--{e['bracket_Hz'][1]/1e6:.1f} MHz \\\\")
    disc = "; ".join(f"{d['f_Hz']/1e6:.0f} MHz: $\\Delta X/X={100*d['dImZ_rel']:+.2f}\\%$, "
                     f"$\\Delta|V|/|V|={100*d['dVoc_rel']:+.2f}\\%$" for d in r["discretization"])
    runs = r["axionfem_runs"]
    return "\n".join([
        "\\section{Shielded lossless toroid: axionFEM vs the original COMSOL data}",
        "The 2022 COMSOL model (\\texttt{Small\\_Shield.mph}) and its port impedance and source-voltage outputs were "
        "recovered; the axionFEM replica (\\texttt{designs/toroid\\_small\\_shield\\_lossless.json}) differs only where "
        "axionFEM requires it: the zero-thickness PEC bridge is 0.2\\,mm thick, the lumped port is a small vacuum volume "
        "in the bridge, the PEC walls are holes in the mesh and the PEC shield cylinder is the outer boundary. "
        f"Production: N\\'ed\\'elec order 2, {runs[0].get('dofs') or 0:,} DOFs, {sm['n']} frequencies 5--150\\,MHz. "
        f"$L(5\\,{{\\rm MHz}})$: axionFEM {s0['L_axf_H']*1e9:.1f}\\,nH, COMSOL {cm['L_5MHz_H']*1e9:.1f}\\,nH. "
        f"Median $|\\Delta X/X|$ = {100*sm['median_abs_rel_ImZ']:.2f}\\%, median $|\\Delta |V_{{\\rm oc}}|/|V_{{\\rm oc}}||$ = "
        f"{100*sm['median_abs_rel_Voc']:.2f}\\%; the largest differences ($\\le$6\\%) sit next to poles and zeros, "
        "where a 0.05--0.25\\% shift of the resonance moves the curve steeply. Mesh refinement (finer mesh D, same "
        f"order) changes the results by: {disc}.",
        "",
        "\\begin{center}\\begin{tabular}{lrrrl}\\toprule",
        " & COMSOL [MHz] & axionFEM [MHz] & diff [\\%] & from \\\\ \\midrule", *rows,
        "\\bottomrule\\end{tabular}\\end{center}",
        "" if conv else "{\\footnotesize Eigenmodes marked provisional come from an eigensolve that did not reach its "
        "residual tolerance (Rayleigh-quotient frequencies; accuracy limited by the residual). The sweep-bracketed "
        "poles and zeros come from converged driven solves and the energy-identity $dX/d\\omega$.}",
        "",
        "\\begin{center}",
        f"\\includegraphics[width=0.8\\textwidth]{{{pp.name}}}\\\\",
        "\\small COMSOL 2022 (black line) and axionFEM (maroon points; grey: refined mesh); ticks: axionFEM eigenmodes. "
        "Bottom: $|$axionFEM/COMSOL$-1|$.",
        "\\end{center}", "\\clearpage"])


def solenoid_section() -> str:
    """Section: axionFEM vs the original COMSOL data of the shielded lossless solenoid (compare_solenoid_comsol.py)."""
    jp, pp = RESULTS / f"{DIRECT_SOL}.json", BUILD / f"{DIRECT_SOL}.pdf"
    if not (jp.is_file() and pp.is_file()):
        return "Solenoid direct comparison: pending (run \\texttt{compare\\_solenoid\\_comsol.py}).\n"
    r = json.loads(jp.read_text())
    sm, cm = r["summary"], r["comsol"]
    s0 = min(r["sweep"], key=lambda t: t["f_Hz"])
    rows = []
    for m in r["eigenmodes_matched"]:
        rows.append(f"pole{' (narrow)' if m['narrow'] else ''} & {m['f_comsol_pole_Hz']/1e6:.3f} & {m['f_axf_Hz']/1e6:.3f} & "
                    f"{100*m['rel']:+.2f} & eigenmode{'' if m['converged'] else ' (provisional)'}, port $E$-frac. "
                    f"{m['port_fraction']:.0e} \\\\")
    for nm, lst in (("pole", r["sweep_poles"]), ("zero", r["sweep_zeros"])):
        for e in lst:
            rows.append(f"{nm} & {e['f_comsol_Hz']/1e6:.3f} & {e['f_Hz']/1e6:.3f} & {100*e['rel']:+.2f} & sweep bracket "
                        f"{e['bracket_Hz'][0]/1e6:.1f}--{e['bracket_Hz'][1]/1e6:.1f} MHz \\\\")
    disc = "; ".join(f"{d['label']} {d['f_Hz']/1e6:.0f} MHz: $\\Delta X/X={100*d['dImZ_rel']:+.2f}\\%$, "
                     f"$\\Delta|V|/|V|={100*d['dVoc_rel']:+.2f}\\%$" for d in r["discretization"])
    runs = r["axionfem_runs"]
    far = sm["far_from_strong_poles"]
    return "\n".join([
        "\\section{Shielded lossless solenoid: axionFEM vs the original COMSOL data}",
        "Recovered COMSOL model \\texttt{Model\\_File.mph} (5--160\\,MHz tables at 1\\,kHz). Replica "
        "\\texttt{designs/solenoid\\_shielded\\_lossless.json}: annular PEC can (25\\,mm walls, PEC holes in the mesh), "
        "1\\,cm slot through the top lid, bridge 0.2\\,mm thick (COMSOL: zero thickness) with a small vacuum-volume port, "
        "$J_e=\\hat z$\\,A/m$^2$ in the bore (\\texttt{fullwave.source} kind \\texttt{axial\\_box}), PEC shield cylinder "
        f"as the outer boundary. Production: N\\'ed\\'elec order 2, {runs[0].get('dofs') or 0:,} DOFs, {sm['n']} frequencies "
        f"5--160\\,MHz. $L(5\\,{{\\rm MHz}})$: axionFEM {s0['L_axf_H']*1e9:.1f}\\,nH, COMSOL {cm['L_5MHz_H']*1e9:.1f}\\,nH. "
        f"Median $|\\Delta X/X|$ = {100*sm['median_abs_rel_ImZ']:.1f}\\%, median $|\\Delta|V_{{\\rm oc}}|/|V_{{\\rm oc}}||$ = "
        f"{100*sm['median_abs_rel_Voc']:.1f}\\% (max {100*sm['max_abs_rel_ImZ']:.0f}\\% / {100*sm['max_abs_rel_Voc']:.0f}\\%, "
        f"next to poles); more than 3\\% from the three strong poles: median $|\\Delta X/X|$ "
        f"{100*far['median_abs_rel_ImZ']:.1f}\\%. Mesh controls: {disc}.",
        "", SOL_NOTE, "",
        "\\begin{center}\\begin{tabular}{lrrrl}\\toprule",
        " & COMSOL [MHz] & axionFEM [MHz] & diff [\\%] & from \\\\ \\midrule", *rows,
        "\\bottomrule\\end{tabular}\\end{center}",
        "",
        "\\begin{center}",
        f"\\includegraphics[width=0.8\\textwidth]{{{pp.name}}}\\\\",
        "\\small COMSOL 2022 (black line) and axionFEM (maroon points; grey: other meshes); ticks: axionFEM eigenmodes; "
        "dotted: the three strong COMSOL poles. Bottom: $|$axionFEM/COMSOL$-1|$.",
        "\\end{center}", "\\clearpage"])


SOL_NOTE = (
    r"\textbf{Reading.} The MQS end agrees: $L$ to 0.3\,\%, $\Psi$ to 1.4\,\% at 5\,MHz. Every resonance of axionFEM lies "
    r"0.3--1.3\,\% \emph{above} COMSOL's (largest for the first, slot-capacitance pole), and the steep curves around "
    r"the poles turn that shift into the large pointwise differences. The shift is not axionFEM discretization: "
    r"refining (K $\to$ A $\to$ B, 3.8/5.4/7.5\,M DOFs) changes $X(20\,{\rm MHz})$ by $-0.7$\,\% per step, i.e.\ "
    r"the first pole by $\approx +0.07$\,\%, away from COMSOL. A control on a mesh of COMSOL's size (237\,k tetrahedra "
    r"vs COMSOL's 188\,k; 2\,cm elements at the 1\,cm slot, as COMSOL's narrow-region setting gives) moves the "
    r"first pole to $\approx 22.0$\,MHz ($X(20\,{\rm MHz})$ +21\,\%), below COMSOL's 22.12\,MHz. The COMSOL "
    r"resonances are therefore consistent with its own (coarse, slot-limited) discretization error; axionFEM's "
    r"converged values are the more accurate ones. The narrow pole--zero pairs (weakly port-coupled modes) are also "
    r"mesh-sensitive in COMSOL itself: the paper's Fig.~S5 bands (e.g.\ 47.9--48.2, 116.2--116.8\,MHz) differ from "
    r"the recovered tables (48.9, 118.8\,MHz) by up to 2\,\%. "
    r"\textbf{Eigenmodes (mesh A, partly unconverged).} The 40-mode HypreAME solve on mesh A (5.44\,M DOFs, "
    r"22.3\,h) returned 39 modes (mass overlaps $\le 10^{-8}$), 29 below the $10^{-4}$ residual tolerance. The three strong modes stay "
    r"slightly above it (residuals $4\times10^{-3}$, $2\times10^{-4}$, $2\times10^{-4}$) but agree with the poles "
    r"bracketed by the driven sweep to $\le 0.01$\,\% (22.4037 / 70.2125 / 132.2133 vs 22.404 / 70.214 / 132.220\,MHz), "
    r"so they confirm the $+1.3$ / $+0.50$ / $+0.32$\,\% offsets from COMSOL. The narrow COMSOL poles are matched by "
    r"converged modes at $+0.01$ to $+0.15$\,\% (the 131.54\,MHz COMSOL pole has no converged partner; the nearest mode, "
    r"132.225\,MHz, is the degenerate twin of the strong mode). Unmatched modes are mostly the second member of "
    r"near-degenerate pairs ($\Delta f/f\sim10^{-5}$) and modes that do not couple to the port. The earlier mesh-K solve "
    r"(3.8\,M DOFs) did not converge at all (residuals 0.8--1.0).")


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=HERE / "paper")
    a = ap.parse_args()
    src = fetch_source(a.src)
    BUILD.mkdir(exist_ok=True)
    rows, pages = [], []
    for name, num, cap, needs in FIGS:
        orig = src / "Plots" / (name if name.endswith(".png") else name + ".pdf")
        shutil.copy(orig, BUILD / orig.name)
        if name.endswith(".png"):
            status = "pending (render)"
            right = None
        else:
            rec = extract(orig)
            right = overlay(name, rec)
            cal = "; ".join(f"x {p['x']['scale']}, y {p['y']['scale']}, max tick residual "
                            f"{max(p['x']['max_residual_pt'], p['y']['max_residual_pt']):.1e} pt" for p in rec["panels"])
            status = "overlay (lossless)" if (RESULTS / f"{name}.json").is_file() else "pending"
            if (RESULTS / f"{name}_epoxy.json").is_file() or (RESULTS / f"{name}_epoxy10.json").is_file():
                status = ("overlay (lossless, lossy)" if status != "pending" else "overlay (lossy)")
        rows.append(f"{num} & \\texttt{{{tex_escape(name)}}} & {needs} & {status} \\\\")
        body = [f"\\subsection*{{{num}: \\texttt{{{tex_escape(name)}}}}}", cap, "", "\\begin{center}",
                f"\\includegraphics[width=0.62\\textwidth,height={0.26 if name in OVERLAY_NOTES else 0.32}\\textheight,keepaspectratio]{{{orig.name}}}\\\\",
                "\\small original (arXiv source)\\\\[6pt]"]
        if right:
            body += [f"\\includegraphics[width=0.75\\textwidth,height={0.34 if name in OVERLAY_NOTES else 0.40}\\textheight,keepaspectratio]{{{right.name}}}\\\\",
                     "\\small curves extracted from the original (grey shades; paper colours not kept) and axionFEM (maroon, black)", "\\end{center}",
                     *([f"{{\\footnotesize {OVERLAY_NOTES[name]}}}", ""] if name in OVERLAY_NOTES else []),
                     f"{{\\footnotesize Extraction: {tex_escape(cal)}.}}"]
        else:
            body += ["\\fbox{\\parbox{0.6\\textwidth}{\\centering\\vspace{1.5cm}axionFEM render: pending\\vspace{1.5cm}}}",
                     "\\end{center}"]
        pages.append("\n".join(body) + "\n\\clearpage")
    tex = (HERE / "note_template.tex").read_text()
    tex = tex.replace("%%TABLE%%", "\n".join(rows)).replace("%%FIGURES%%", "\n\n".join(pages))
    tex = tex.replace("%%DIRECT%%", direct_section() + "\n\n" + solenoid_section() + "\n\n" + epoxy_note.section(RESULTS, BUILD))
    tex = tex.replace("\\begin{document}", "\\usepackage{array}\n\\begin{document}", 1)
    (BUILD / "beyond_mqs_2211_validation.tex").write_text(tex)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "beyond_mqs_2211_validation.tex"],
                   cwd=BUILD, check=True, capture_output=True)
    print(BUILD / "beyond_mqs_2211_validation.pdf")


if __name__ == "__main__":
    main()
