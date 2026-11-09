"""Assemble benchmarks/cavity/refs/admx_two_rod.json from the digitization and the 2D inference.

Usage:  python3 -I build_admx_two_rod_ref.py
Inputs: digitize/admx_two_rod_digitized.json  (admx_two_rod_digitize.py)
        digitize/admx_two_rod_infer.json      (admx_two_rod_infer.py all + positions)
Also writes admx_two_rod_geometry.png (inferred cross-section with the Fig. 2 rod positions) and
admx_two_rod_1a_fig8c_model_overlay.png (2D model f(theta) over the Run 1A mode map).
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.dirname(HERE)
DIG = json.load(open(os.path.join(HERE, "admx_two_rod_digitized.json")))
INF = json.load(open(os.path.join(HERE, "admx_two_rod_infer.json")))
R, RROD = 210.0, 25.4


def r2(x, n=2):
    return None if x is None else round(float(x), n)


def fig2_table():
    rows = []
    for src, lab in (("run1b_fig2", "1910.08638 Fig. 2"), ("run1b_analysis_fig16", "2010.06183 Fig. 16")):
        for pn in ("symmetric", "anti_symmetric"):
            m = DIG[src]["panels"][pn]
            for sc in ("outer_rim_edge_is_wall", "inner_rim_edge_is_wall"):
                g = m[sc]["rods"]
                for k, side in enumerate(("left", "right")):
                    q = g[k]
                    rows.append([lab, pn, sc.replace("_is_wall", ""), side,
                                 r2(q["rod_xy_mm"][0], 1), r2(q["rod_xy_mm"][1], 1), r2(q["rod_r_mm"], 1),
                                 r2(q["pivot_xy_mm"][0], 1), r2(q["pivot_xy_mm"][1], 1), r2(q["pivot_r_mm"], 1),
                                 r2(q["arm_mm"], 1), r2(q["theta_ccw_deg"], 1)])
    return rows


def model_overlay_8c():
    im = Image.open(os.path.join(REFS, "admx_two_rod_1a_fig8c.png")).convert("RGB")
    dr = ImageDraw.Draw(im)
    xp, xv = [59, 121, 184, 246, 308, 371, 433], [90, 100, 110, 120, 130, 140, 150]
    yp, yv = [15, 77, 138, 200, 262, 323, 385], [0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60]
    fx, fy = np.polyfit(xv, xp, 1), np.polyfit(yv, yp, 1)
    pts = [(2 * np.polyval(fx, t), 2 * np.polyval(fy, f / 1e3)) for t, f in INF["model_curve_rod1_wall_h1mm"]
           if 90 <= t <= 155]
    dr.line(pts, fill=(0, 255, 255), width=3)
    for t, f in INF["model_curve_rod1_wall_h1mm"]:
        if 90 <= t <= 155:
            X, Y = 2 * np.polyval(fx, t), 2 * np.polyval(fy, f / 1e3)
            dr.ellipse([X - 5, Y - 5, X + 5, Y + 5], outline=(0, 255, 255), width=2)
    im.save(os.path.join(REFS, "admx_two_rod_1a_fig8c_model_overlay.png"))


def geometry_png(rp, a):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, pn, title in ((axs[0], "symmetric", "Symmetric (Fig. 2 left)"),
                          (axs[1], "anti_symmetric", "Anti-symmetric (Fig. 2 right)")):
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(R * np.cos(t), R * np.sin(t), "k", lw=1.5)
        for sx in (-1, 1):
            ax.plot(sx * rp + a * np.cos(t), a * np.sin(t), ":", color="0.5", lw=1)
            ax.plot([sx * rp], [0], "k+", ms=10)
        g = DIG["run1b_analysis_fig16"]["panels"][pn]["outer_rim_edge_is_wall"]["rods"]
        for q in g:
            x, y = q["rod_xy_mm"]
            ax.add_patch(plt.Circle((x, y), RROD, fc="#f2d24b", ec="k", lw=1))
            ax.plot([q["pivot_xy_mm"][0], x], [q["pivot_xy_mm"][1], y], "r-", lw=1.5)
            ax.text(x, y - 38, f"{q['theta_ccw_deg']:.1f} deg ccw", ha="center", fontsize=8)
        ax.set_aspect("equal"); ax.set_xlim(-220, 220); ax.set_ylim(-220, 220)
        ax.set_title(title, fontsize=10); ax.set_xlabel("x [mm]")
    axs[0].set_ylabel("y [mm]")
    fig.suptitle(f"ADMX Run 1A/1B two-rod cross-section: R = 210 mm, rod d = 50.8 mm, INFERRED pivot r_p = {rp:g} mm, arm a = {a:g} mm\n"
                 "rods = positions measured from 2010.06183 Fig. 16 (outer-rim scale); dotted = arm circles", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REFS, "admx_two_rod_geometry.png"), dpi=110)


def main():
    fit, rep = INF["fit"], INF["reported"]
    rp, a = rep["r_p_mm"], rep["a_mm"]
    model_overlay_8c()
    geometry_png(rp, a)
    f2 = fig2_table()
    meas = {}
    for src in ("run1b_fig2", "run1b_analysis_fig16"):
        for pn in ("symmetric", "anti_symmetric"):
            for sc in ("outer_rim_edge_is_wall", "inner_rim_edge_is_wall"):
                for q in DIG[src]["panels"][pn][sc]["rods"]:
                    meas.setdefault(sc, {"rp": [], "a": []})
                    meas[sc]["rp"].append(q["pivot_r_mm"]); meas[sc]["a"].append(q["arm_mm"])
    m_out = {k: {"pivot_r_mm_mean": r2(np.mean(v["rp"]), 1), "pivot_r_mm_range": [r2(min(v["rp"]), 1), r2(max(v["rp"]), 1)],
                 "arm_mm_mean": r2(np.mean(v["a"]), 1), "arm_mm_range": [r2(min(v["a"]), 1), r2(max(v["a"]), 1)]}
             for k, v in meas.items()}
    pos = INF.get("position_checks", [])
    ref = INF["refined_table"]
    sym_curve = INF["model_curve_symmetric_h1mm"]
    th_sym = np.array([c[0] for c in sym_curve], float); f_sym = np.array([c[1] for c in sym_curve], float)
    th_680 = float(np.interp(680, f_sym, th_sym)); th_790 = float(np.interp(790, f_sym, th_sym))
    c8c = DIG["run1a_fig8c"]
    c4c = DIG["run1a_fig4c"]
    c8 = DIG["run1b_analysis_fig8"]
    dd = DIG["dd_fig5"]
    a8 = DIG["run1a_fig8a"]

    J = {
        "name": "admx_two_rod",
        "status": ("COMPLETE" if abs(a - 58) < 1e-9 else
                   "PARTIAL: 2D model tables/curves computed at r_p = %g, a = %g mm (rounded chi^2 fit); the final refine at the combined value a = 58 mm is pending" % (rp, a)),
        "title": "ADMX Run 1A/1B main cavity with two 50.8 mm copper tuning rods on alumina armatures: geometry (pivot INFERRED), Run 1B Fig. 2 symmetric / anti-symmetric rod positions, simulated mode map and C(f)",
        "benchmark_rank": "8",
        "citations": [
            {"arxiv": "1910.08638", "authors_short": "Braine et al. (ADMX)", "journal": "Phys. Rev. Lett. 124, 101303 (2020)",
             "items_used": ["p.1: 136 L copper-plated cavity, 7.6 T, two copper rods parallel to the axis movable 'between near the center of the cavity and the walls'",
                            "p.2: average C = 0.4; mode crossings with weakly tuning TE/TEM modes; alternative rod configurations shift 'the static mode by several MHz'",
                            "p.3 Fig. 2: top-down CST view, symmetric (left) and anti-symmetric (right) rod configurations with the SAME TM010 frequency, colour = |E_z|",
                            "p.3-4: symmetric configuration first (8 mode crossings, identified with simulations, confirmed by wide S21); anti-symmetric fills crossings, C ~ 0.35 there, unloaded Q x1.5 vs symmetric; 3 crossings filled",
                            "p.4 Table I: 5 unfilled mode crossings (f, width)", "p.4: search range 680-790 MHz"]},
            {"arxiv": "2010.06183", "authors_short": "Bartram et al. (ADMX)", "journal": "Phys. Rev. D 103, 032002 (2021) [journal ref UNVERIFIED here]",
             "items_used": ["Sec. II p.3: '140-liter cavity ... (136 liter when the tuning rod volume is subtracted). Two 50.8-mm diameter copper tuning rods'",
                            "Sec. VI p.9 + Fig. 8: form factor vs frequency simulated with CST at 'a few select frequencies', interpolated",
                            "Sec. VII p.13 + Fig. 16 (p.14): definitions: symmetric = 'rods, starting at the same position opposite each other next to the walls, were rotated in the same direction, at the same rate'; Fig. 16 caption: symmetric = both rods counter-clockwise; anti-symmetric = left rod counter-clockwise, right rod clockwise; both bring rods to the centre and increase f",
                            "Table IV: same 5 unfilled crossings as 1910.08638 Table I"]},
            {"arxiv": "2010.00169", "authors_short": "Khatiwada et al. (ADMX)", "journal": "Rev. Sci. Instrum. 92, 124502 (2021) [journal ref UNVERIFIED here]",
             "items_used": ["Sec. II.B p.4: 136 L, 'approximately 1 m in length and 0.4 m in diameter'; two 0.05 m copper rods on aluminum oxide rotary armatures; TM010-like 580-890 MHz; OFHC plating >= 0.08 mm, annealed 8 h at 400 C; Q 40,000-80,000 (Runs 1A/1B); C from simulation, average 0.4",
                            "Sec. II.C p.4: rods 'swept out the 0.09 m radius circular trajectories shown in Fig 5'; theta = 0 at the walls, theta = 180 at the centre (highest TM010 f); alumina armatures protrude through the end caps and offset the rods",
                            "Fig. 3 (p.5): CST TM010 'with each of the rods at 116 deg from the center'; CAD cutaway; photo of rods and alumina axles",
                            "Fig. 4: CST magnet field map used in C", "Fig. 5 top right (p.6): TM010 cross-section at 773 MHz with rods labelled 180 deg and 270 deg, pivots and arms drawn (vector annotation)"]},
            {"arxiv": "2312.16668", "authors_short": "ADMX Collaboration, Run 1A analysis details", "journal": "[journal ref not checked]",
             "items_used": ["Sec. II.A.2 p.3-5: knife-edge end caps; OFHC plating >= 0.075 mm; rods 0.05 m, 'copper-plated stainless steel', on alumina oxide rotary armatures; rods reduce Q by ~2x; TM010-like branch = 'TM010c' of Lyapustin; 580-890 MHz; Run 1A: one rod at the wall, the other free; C simulated with COMSOL",
                            "Eq. 5-6: C definition with the magnet field map", "Fig. 4c (p.5): simulated C010 (COMSOL, 6% band) and measured Q_L vs TM010c frequency, 645-682 MHz",
                            "Fig. 8a (p.10): |E_z| at 656.4 MHz, rods labelled 120 deg and 0 deg (pivots/arms drawn as vector annotation)",
                            "Fig. 8c (p.10): simulated S12 mode map 0.60-0.72 GHz vs 'Rod 2 Angle' 90-155 deg; TM010c bracketed (107-141 deg per caption)"]},
            {"arxiv": "2408.03444", "authors_short": "T. Braine (PhD thesis, U. Washington)", "journal": "thesis",
             "items_used": ["Sec. 4.x p.51-53 (PDF p.78-80): cavity 101 cm long, 42 cm inner diameter, 136 L; knife-edge clamp (Fig. 4.8); rods = hollow OFHC copper tubes with welded end caps, '~0.020 in clearance gap on top and bottom'; axles off-centre of the rod; alumina (earlier) / sapphire axles; Run 1A-B rods 5 cm diameter, Run 1C 11.1 cm, Run 1D single 20.5 cm; 3 antenna ports",
                            "Table 6.1 (PDF p.123): ADMX Main R = 21.0 cm, H = 101 cm (see analytic_admx_thesis_tab6_1.json)"]},
            {"arxiv": "1804.05750", "authors_short": "Du et al. (ADMX)", "journal": "Phys. Rev. Lett. 120, 151301 (2018)",
             "items_used": ["p.1: 136 L, two copper-plated rods moved 'from near the center to the perimeter'", "Table I: form-factor modelling systematic 6.0% at 655 MHz", "645-680 MHz search range"]},
            {"arxiv": "2110.06096", "authors_short": "Bartram et al. (ADMX Run 1C)", "journal": "Phys. Rev. Lett. 127, 261803 (2021) [UNVERIFIED here]",
             "items_used": ["context only: Run 1C used DIFFERENT (bulk copper, 11.1 cm per 2408.03444) rods moved symmetrically; Fig. 3 COMSOL TM010/TM011 with CST C = 0.455 / <0.001 - NOT this geometry"]},
            {"arxiv": "1603.06990", "authors_short": "Stern, Carosi, Rybka, et al.", "journal": "Rev. Sci. Instrum. 86, 123305 (2015) [as cited in stern_multipost_1603.json]",
             "items_used": ["Sec. II + Fig. 2 (schematic, not to scale): ADMX rods move along circular arcs from near the wall to near the centre; empty f0 ~ 550 MHz; methods only (no two-rod numbers)"]},
        ],
        "sources_searched_not_used": [
            "Lyapustin PhD thesis (U. Washington; cited as [27] in 2312.16668 for the 'TM010c' name): not found online.",
            "UW REU 2019 slides (archive.int.washington.edu/REU/2019/Man_Presentation.pdf): MEASURED warm/cold mode maps vs raw encoder 'rod position', 700-1050 MHz, apparently later (Run 1C-era) rods; no geometry; not used.",
            "UW REU 2012 report (swan_paper.pdf): older 25 cm radius ADMX cavity, end-plate ridges; not this geometry.",
        ],
        "software": {
            "code": "CST Microwave Studio (Run 1B: 2010.06183 Fig. 8/16, 2010.00169 Fig. 3/5); COMSOL per 2312.16668 text for Run 1A (but its Fig. 8a render has the CST colour legend)",
            "version": "not stated",
            "solver_type": "eigenmode (field plots, C); simulated S12 transmission spectra for the mode map (2312.16668 Fig. 8c)",
            "boundary": "copper; conductivity not stated",
            "mesh_notes": "not stated; Run 1B C computed 'at a few select frequencies' and interpolated (2010.06183 p.9)",
        },
        "geometry": {
            "units": "mm",
            "description": "Right circular copper-plated stainless-steel cylinder (flat end caps clamped on knife edges) with two parallel copper tuning rods of 50.8 mm diameter running the length of the cavity, ~0.5 mm clearance at each end cap. Each rod is carried by alumina armatures (top and bottom) that pivot about an axle through the end cap; the two pivots lie on one diameter, symmetric about the axis. theta = 0: arm pointing radially outward (rod at the wall); theta = 180: rod toward the centre. Pivot radius and arm length are NOT published and are INFERRED here (figure measurement + 2D frequency fit).",
            "parameters": {
                "R_cavity": {"value": 210.0, "unit": "mm", "source": "2408.03444 p.52 ('inner diameter of 42 cm'); Table 6.1 R = 21.0 cm",
                             "note": "2010.00169 p.4 'approximately ... 0.4 m in diameter'"},
                "H_cavity": {"value": 1010.0, "unit": "mm", "source": "2408.03444 p.52 ('101 cm'); Table 6.1 H = 101 cm", "note": "2010.00169: 'approximately 1 m'"},
                "volume": {"value": {"total_L": 140, "minus_rods_L": 136}, "unit": "L", "source": "2010.06183 p.3",
                           "note": "check: pi 0.21^2 1.01 = 139.93 L; minus 2 rods of 50.8 mm x 1008.98 mm = 135.84 L. The '136 L' quoted everywhere is the vacuum volume."},
                "rod_diameter": {"value": 50.8, "unit": "mm", "source": "2010.06183 p.3 ('50.8-mm diameter')", "note": "2 in; '0.05 m' in 2010.00169 / 2312.16668; '5 cm' in 2408.03444"},
                "rod_length": {"value": 1008.98, "unit": "mm", "source": "derived: H - 2 x end_gap_each", "note": "rod ends: welded end caps (2408.03444); edge rounding not given"},
                "end_gap_each": {"value": 0.508, "unit": "mm", "source": "2408.03444 p.52: '~0.020 in clearance gap on top and bottom'",
                                 "note": "approximate ('~'); top/bottom asymmetry unknown; alumina armatures/axles cross this region (geometry not given)"},
                "n_rods": {"value": 2, "unit": "", "source": "all"},
                "pivot_radius": {"value": rp, "unit": "mm", "status": "INFERRED",
                                 "uncertainty_mm": 2.0,
                                 "source": "INFERRED (two independent methods agree): (i) Run 1B Fig. 2 / 2010.06183 Fig. 16: pivot = centre of the ring at the far end of each alumina armature, both on one diameter; r_p = %.1f mm (outer-rim scale) / %.1f mm (inner-rim scale); (ii) chi^2 fit of the ideal 2D z-uniform model to 4 published simulated frequencies (580, 890, 773 MHz, 656.4 MHz at stated angles): r_p = %.1f +/- %.1f mm" % (
                                     m_out["outer_rim_edge_is_wall"]["pivot_r_mm_mean"], m_out["inner_rim_edge_is_wall"]["pivot_r_mm_mean"], fit["r_p_mm"], fit["sigma_r_p_mm"]),
                                 "method": "digitize/admx_two_rod_digitize.py (ring/dot circle fits, scale from the cavity outline: R = 210 mm at the outer or inner rim edge; CST render is near-orthographic, rim width 2.6%); digitize/admx_two_rod_infer.py (Shortley-Weller FD eigen-solver, h = 2/1/0.5 mm, validated vs empty-cylinder and coax closed forms to < 1e-5)",
                                 "note": "pivots diametrically opposite (measured angular offset < 1 deg, radii equal within 0.3 mm). 2010.00169 Fig. 5 hand-drawn pivot dots give 129.5 / 120.2 mm (asymmetric, not used). 2312.16668 Fig. 8a dots give ~130 mm (hand-placed, not used)."},
                "arm_length": {"value": a, "unit": "mm", "status": "INFERRED", "uncertainty_mm": 2.0,
                               "source": "INFERRED: Fig. 2/16 pivot-ring to rod-centre distance %.1f mm (outer-rim scale; all four rods within %s) / %.1f mm (inner-rim scale); 2D frequency fit a = %.1f +/- %.1f mm (weakly constrained)" % (
                                   m_out["outer_rim_edge_is_wall"]["arm_mm_mean"], m_out["outer_rim_edge_is_wall"]["arm_mm_range"], m_out["inner_rim_edge_is_wall"]["arm_mm_mean"], fit["a_mm"], fit["sigma_a_mm"]),
                               "note": "conflicts with 2010.00169 'rods swept out the 0.09 m radius circular trajectories' (see inconsistencies). Wall clearance at theta = 0: R - (r_p + a + 25.4) = %.1f mm; rod-centre radius at theta = 180: r_p - a = %.0f mm." % (R - (rp + a + RROD), rp - a)},
                "pivot_positions": {"value": [[-rp, 0.0], [rp, 0.0]], "unit": "mm", "status": "INFERRED", "source": "Fig. 2/16 measurement; frame: x along the pivot diameter, y up in the published top-down view"},
                "rod_angle_range": {"value": [0, 180], "unit": "deg", "source": "2010.00169 Sec. II.C (theta = 0 walls, 180 centre)",
                                    "note": "Run 1A: one rod at 0, the other 107-141 deg (2312.16668 Fig. 8 caption). Run 1B (680-790 MHz, symmetric) corresponds in the inferred 2D model to theta ~ %.0f-%.0f deg." % (th_680, th_790)},
                "armatures": {"value": None, "unit": "mm", "source": "alumina (Al2O3) 'rotary armatures' / 'ceramic axles' (2010.00169 Sec. II.C, Fig. 3; 2312.16668)",
                              "note": "dimensions not published. Fig. 16 render: a rounded bar from the pivot (ring radius ~5.5 mm) to under the rod top, at the end-cap level; omit in 2D; in 3D sweep a 6-10 mm wide, few-mm thick alumina bar (eps_r ~ 9.8) in each end gap."},
                "end_caps": {"value": "flat", "unit": "", "source": "2408.03444 Fig. 4.8 (knife-edge clamp); 2312.16668 p.4",
                             "note": "copper-plated stainless; port fixtures for antennae and rod axles. Port positions/sizes not published."},
                "ports": {"value": None, "unit": "", "source": "2312.16668 p.5: one fixed weak port on the bottom plate, two tunable major ports on the top plate (Run 1A); 2408.03444: 2 weak + 1 strong", "note": "positions not published; omit"},
            },
        },
        "materials": {
            "walls_endcaps": {"material": "stainless steel, OFHC copper plated (>= 0.08 mm, 2010.00169; >= 0.075 mm, 2312.16668), annealed 8 h at 400 C", "sigma_S_per_m": None, "temperature_K": "~0.13-0.15 K operation; simulations: not stated", "source": "2010.00169 Sec. II.B; 2312.16668 p.4"},
            "rods": {"material": "copper (1910.08638, 2010.06183); 'copper-plated stainless steel' (2312.16668); hollow OFHC copper tubes with welded end caps (2408.03444)", "sigma_S_per_m": None, "source": "see inconsistencies"},
            "armatures_axles": {"material": "alumina (Al2O3)", "eps_r": None, "tan_delta": None, "source": "2010.00169, 2312.16668, 2408.03444", "note": "use eps_r = 9.8 if modelled"},
        },
        "conditions": {"temperature_K": "simulations: not stated", "B0_T": "7.6 (Run 1B), Run 1A 6.8 T nominal [not re-checked]",
                       "B0_direction": "+z (cavity axis)",
                       "field_map": "C uses the simulated solenoid map (CST magnet solver, 2010.00169 Fig. 4; COMSOL map, 2312.16668 Fig. 3); NOT uniform. For 2D z-uniform comparisons use uniform B and expect C_2D > C_published."},
        "conventions": {
            "form_factor_definition": "C = |int E . B0 dV|^2 / (|B_max|^2 V int |E|^2 dV) (2312.16668 Eq. 5-6), B0 = magnet map. V convention not stated (136 L vacuum vs 140 L full); report both (factor 140/136 = 1.029).",
            "Q_definition": "2312.16668 Fig. 4c: measured LOADED Q_L (critically coupled antenna); 2010.00169: 'quality factor between 40,000 and 80,000' (type not stated)",
            "angle_convention": "theta measured at each pivot from the radially OUTWARD direction (0 = rod at the wall, 180 = rod nearest the centre) (2010.00169 Sec. II.C). In 1910.08638 Fig. 2 / 2010.06183 Fig. 16 both rods have theta = 115.5 deg counter-clockwise (as viewed) in the symmetric panel; in the anti-symmetric panel the left rod is at 129.9 deg CCW and the right rod at 129.4 deg CW (230.6 deg CCW). This matches 2010.00169 Fig. 3 ('each of the rods at 116 deg').",
            "symmetric_vs_antisymmetric": "symmetric: both rods start at the wall (theta = 0) on opposite sides and rotate in the SAME sense at the same rate -> configuration has C2 (180-deg rotation) symmetry about the axis. anti-symmetric: opposite senses -> mirror symmetry about the perpendicular bisector of the pivot line (both rods on the same side of the pivot diameter). (2010.06183 Sec. VII + Fig. 16 caption; 1910.08638 Fig. 2.) In the model frame: left rod centre = (-r_p - a cos th_L, -a sin th_L), right rod centre = (r_p + a cos th_R, s a sin th_R) with s = +1 symmetric, s = -1 anti-symmetric.",
            "notes": "frequencies in the 2D model are invariant under a global mirror, so the sense convention of a single labelled rod (e.g. 2010.00169 Fig. 5 '270 deg', measured 90.9 deg CCW = 269.1 deg CW) does not affect f.",
        },
        "reference_values": {
            "tables": [
                {"source": "1910.08638 Fig. 2 and 2010.06183 Fig. 16 (MEASURED from the figure by this work)",
                 "description": "Rod and pivot positions in the cavity frame (origin = cavity axis, x along the pivot diameter, y up as printed). Scale from the cavity outline with R = 210 mm at the outer rim edge (preferred; the inner-rim scale is 2.6-2.7% larger and puts the theta = 0 rod into the wall). Caption: 'The frequency of the TM010 mode is the same in both configurations shown' (value not published).",
                 "simulated": True,
                 "columns": ["figure", "panel", "scale (rim edge = R)", "rod", "rod_x", "rod_y", "rod_r", "pivot_x", "pivot_y", "pivot_r", "arm", "theta_ccw"],
                 "units": ["", "", "", "", "mm", "mm", "mm", "mm", "mm", "mm", "mm", "deg"],
                 "rows": f2,
                 "scale_choice_check": "2D f at the measured positions (analytic_crosschecks.two_rod_2d_model_this_work.at_measured_rod_positions): sym vs anti differ by 0.25% with the outer-rim scale but 0.8% with the inner-rim scale, so the caption's 'same frequency' also favours the outer-rim scale.",
                 "uncertainty": "positions +/-1.5 mm (circle-fit and perspective: rendered rod rims doubled by ~0.5%, rim width 2.6%), angles +/-1 deg; Fig. 2 (1188 px) and Fig. 16 (2540 px) agree within 0.5 mm / 0.2 deg"},
                {"source": "published simulated TM010-like frequencies at stated rod angles",
                 "description": "point values used for the inference (and as 3D targets)",
                 "simulated": True,
                 "columns": ["configuration (theta_L, theta_R)", "f_MHz", "code", "source", "note"],
                 "units": ["deg", "MHz", "", "", ""],
                 "rows": [["(0, 0) both at the wall", "~580", "sim (unspecified)", "2312.16668 p.5; 2010.00169 p.4", "'roughly 580 MHz to 890 MHz'"],
                          ["(180, 180) both at the centre", "~890", "sim (unspecified)", "same", "upper end"],
                          ["(180, 270)", "773", "CST", "2010.00169 Fig. 5 top right", "drawn rods at r = %.1f mm (left) and (%.1f, %.1f) mm (right)" % (dd["geometry"][0]["rod_r_mm"], *[(dd["rods_pt"][1][0] - dd["outline_pt"][0]) * dd["mm_per_pt"], -(dd["rods_pt"][1][1] - dd["outline_pt"][1]) * dd["mm_per_pt"]])],
                          ["(120, 0)", "656.4", "CST-style render (text: COMSOL)", "2312.16668 Fig. 8a", "raster rods at r = %.1f mm (120 deg) and %.1f mm (0 deg)" % (a8["geometry"][0]["rod_r_mm"], a8["geometry"][1]["rod_r_mm"])],
                          ["both at 116 (symmetric)", "not given", "CST", "2010.00169 Fig. 3", "same as Fig. 2 symmetric panel within 0.5 deg"],
                          ["Fig. 2 symmetric (115.5, 115.5) vs anti-symmetric (129.9 ccw, 129.4 cw)", "equal (value not given)", "CST", "1910.08638 Fig. 2 caption", "use as a consistency target: f_sym = f_anti"]]},
                {"source": "1910.08638 Table I = 2010.06183 Table IV",
                 "description": "Run 1B mode crossings where no limit could be set (operational; crossings 'identified with simulations and confirmed with wide-span S21'). Configuration: after anti-symmetric filling.",
                 "simulated": False,
                 "columns": ["f_MHz", "width_MHz"], "units": ["MHz", "MHz"],
                 "rows": [["704.659", "0.350"], ["715.064", "0.140"], ["717.025", "0.140"], ["726.624", "0.701"], ["753.844", "12.682"]]},
            ],
            "curves": [
                {"source": "2312.16668 Fig. 8c", "description": "Simulated S12 transmission mode map (code: COMSOL per text), rod 1 at the wall (theta = 0), rod 2 angle 90-155 deg, 0.60-0.72 GHz. Branches = per-column peaks of the inverted 'inferno' colour map, assigned to hand guides. The bracketed TM010c branch (Run 1A, 107/110-141 deg) is series 0; between ~136 and ~145 deg it passes an avoided crossing with flat intruders at 0.68-0.70 GHz (assignment uncertain +/-0.005 GHz there).",
                 "x": {"label": "rod 2 angle", "unit": "deg"}, "y": {"label": "frequency", "unit": "GHz"},
                 "series": [{"label": s["label"], "points": s["points"]} for s in c8c["series"]],
                 "digitization": {"method": "embedded 576x432 raster; inverse colour map (median RGB error %.1f); scipy find_peaks prominence 0.04 per pixel column, parabolic sub-pixel refinement; snapped to PCHIP guides (tolerance 2-2.5 MHz); 1-deg median bins" % c8c["axis_calibration"]["colormap_inverse_rms_rgb_err_median"],
                                  "page": 11, "axis_calibration": c8c["axis_calibration"], "uncertainty_x": 0.3, "uncertainty_y": 0.0005,
                                  "uncertainty_basis": "0.16 deg/px, 0.32 MHz/px; peak position +/-1 px; calibration residual < 0.12 MHz",
                                  "guides": {s["label"]: s["guide"] for s in c8c["series"]},
                                  "unassigned_peaks_GHz": "intruder dots near 0.68-0.70 GHz and a faint line 0.612-0.616 GHz are kept in digitize/admx_two_rod_digitized.json",
                                  "figure_png": "admx_two_rod_1a_fig8c.png", "overlay_png": "admx_two_rod_1a_fig8c_overlay.png",
                                  "model_overlay_png": "admx_two_rod_1a_fig8c_model_overlay.png", "script": "digitize/admx_two_rod_digitize.py"}},
                {"source": "2312.16668 Fig. 4c", "description": "Simulated C010 of the TM010c mode over the Run 1A tuning path (rod 1 at the wall, rod 2 ~107-141 deg) vs TM010c frequency, with the quoted 6% band (blue shading). Code per text: COMSOL; C uses the magnet field map.",
                 "x": {"label": "TM010c frequency", "unit": "MHz"}, "y": {"label": "C010", "unit": ""},
                 "series": [{"label": "C010 simulated (blue line)", "points": c4c["C_sim"]},
                            {"label": "band lower/upper edge (+/-6%)", "points": c4c["C_band_lo_hi"]}],
                 "digitization": {"method": "embedded 2048x1536 layer with the SOLID blue line (xref 416; the printed layer 417 is the dashed copy; max |dC| between layers %.4f); per-column centre of the pure-blue stroke; 1-MHz median bins; band = extreme light-blue pixels" % c4c["dashed_vs_solid_layer_max_abs_diff_C"],
                                  "page": 6, "axis_calibration": c4c["axis_calibration"], "uncertainty_x": 0.1, "uncertainty_y": 0.001,
                                  "figure_png": "admx_two_rod_1a_fig4c.png", "overlay_png": "admx_two_rod_1a_fig4c_overlay.png", "script": "digitize/admx_two_rod_digitize.py"}},
                {"source": "2010.06183 Fig. 8", "description": "Run 1B simulated (CST, 'a few select frequencies', linearly interpolated) form factor vs frequency, 677-812 MHz. Rod configuration not stated (presumably symmetric, the normal data-taking mode). Dips at 729, 743, 750-766 MHz are mode crossings (deepest 0.2245 at 754 MHz).",
                 "x": {"label": "frequency", "unit": "MHz"}, "y": {"label": "C", "unit": ""},
                 "series": [{"label": "polyline vertices (= CST sample points where the slope changes; collinear samples are not recoverable)", "points": c8["vertices"]},
                            {"label": "dense trace, 1-MHz bins", "points": c8["dense_1MHz"]}],
                 "digitization": {"method": "embedded 1922x1119 raster; per-column centre of the tab:blue stroke; Ramer-Douglas-Peucker (3 px) for vertices; local-extremum vertices corrected for stroke width (round joins): vertex = extreme stroke pixel -/+ lw/2",
                                  "page": 9, "axis_calibration": c8["axis_calibration"], "uncertainty_x": 0.5, "uncertainty_y": 0.001,
                                  "uncertainty_basis": "0.09 MHz/px, 0.0003/px; stroke 8 px; vertex abscissae +/-0.5 MHz; dip minimum +/-0.003",
                                  "extremum_refinement": c8["extremum_vertex_refinement_f_Cmedian_Crefined"],
                                  "figure_png": "admx_two_rod_1banalysis_fig8.png", "overlay_png": "admx_two_rod_1banalysis_fig8_overlay.png", "script": "digitize/admx_two_rod_digitize.py"}},
            ],
            "figures": {
                "run1b_fig2_crop": "admx_two_rod_1b_fig2.png", "run1b_fig2_overlay": "admx_two_rod_1b_fig2_overlay.png",
                "run1b_analysis_fig16_overlay": "admx_two_rod_1banalysis_fig16_overlay.png",
                "dd_fig5": "admx_two_rod_dd_fig5.png", "dd_fig5_overlay": "admx_two_rod_dd_fig5_overlay.png",
                "run1a_fig8a": "admx_two_rod_1a_fig8a.png", "run1a_fig8a_overlay": "admx_two_rod_1a_fig8a_overlay.png",
                "inferred_geometry": "admx_two_rod_geometry.png",
                "overlay_legend": "Fig. 2/16: magenta = outer-rim circle fit, black thin = inner-rim fit, black + = cavity axis, red + = pivot (ring fit), red line = arm, red circle = 50.8 mm rod at the outer-rim scale, white circle = pivot circle. DD Fig. 5: magenta = drawn outline, red = drawn rod circles and pivots, green = 50.8 mm rod at that scale. 1A Fig. 8a: magenta = fitted outline/rods, black + = annotation pivots. Fig. 8c: green = TM010c, cyan = upper parallel line, white = lowest line, magenta = top-left line, black dots = unassigned peaks; model overlay: cyan line = 2D model (this work)."},
        },
        "measured_values": {
            "tables": [
                {"source": "2312.16668 Fig. 4c red line (measured cryogenic loaded Q_L vs TM010c frequency)", "simulated": False,
                 "columns": ["f_MHz", "Q_L"], "units": ["MHz", ""], "rows": c4c["QL_measured"]},
                {"source": "text", "simulated": False, "columns": ["quantity", "value", "source"], "units": ["", "", ""],
                 "rows": [["Q (Runs 1A/1B)", "40,000-80,000", "2010.00169 p.4"], ["Q0 anti-symmetric / symmetric", "1.5", "1910.08638 p.4"],
                          ["C in filled crossings (anti-symmetric)", "~0.35", "1910.08638 p.4; 2010.06183 Sec. VII"],
                          ["average C (Run 1B)", "0.4", "1910.08638 p.2"], ["form-factor systematic", "6.0% at 655 MHz", "1804.05750 Table I"]]},
            ],
        },
        "analytic_crosschecks": {
            "empty_cylinder": {"TM010_MHz": 546.39, "TM011_MHz": 566.19, "TE111_MHz": 443.88, "C_TM010": 0.6917, "source": "analytic_admx_thesis_tab6_1.json (R = 210 mm, H = 1010 mm)"},
            "TM011_offset_note": "f_TM011-like ~ sqrt(f_TM010^2 + (c/2H)^2), c/2H = 148.4 MHz: at f_TM010c = 659 MHz this gives +16.5 MHz, matching the upper parallel line of Fig. 8c (~+17-19 MHz).",
            "two_rod_2d_model_this_work": {
                "description": "ideal 2D z-uniform TM model (no end gaps/armatures/ports), R = 210 mm, rods 50.8 mm, INFERRED r_p = %g mm, a = %g mm; Richardson-extrapolated from h = 1, 0.5 mm (h-convergence shown); C2D with uniform B: full-disk area and vacuum area. NOT published values - inference/consistency only." % (rp, a),
                "fit": fit,
                "rows": [{"label": r["label"], "theta_L": r["theta_L"], "theta_R": r["theta_R"], "sense_R": r["sense_R"], "target_MHz": r["target_MHz"],
                          "f_MHz_h2_h1_h05": [r2(v, 3) for v in r["f_MHz_h2_h1_h05"]], "f_MHz": r2(r["f_MHz_richardson"], 2),
                          "next_TM_modes_MHz": [r2(v, 1) for v in r["next_modes_MHz_h05"][1:]],
                          "C2D_full": r2(r["C2D_full_disk_h05"], 4), "C2D_vac": r2(r["C2D_vacuum_area_h05"], 4)} for r in ref],
                "at_measured_rod_positions": [{"label": p["label"], "rods_xy_mm": p["rods_xy_mm"], "published_f_MHz": p["published_f_MHz"],
                                               "f2D_MHz": r2(p["f2D_MHz_richardson"], 2), "C2D_full": r2(p["C2D_full_disk"], 4), "C2D_vac": r2(p["C2D_vacuum_area"], 4)} for p in pos],
                "curve_rod1_wall_theta2": INF["model_curve_rod1_wall_h1mm"],
                "curve_symmetric": [[c[0], r2(c[1], 2), r2(c[2], 4), r2(c[3], 4)] for c in INF["model_curve_symmetric_h1mm"]],
                "curve_antisymmetric": [[c[0], r2(c[1], 2), r2(c[2], 4), r2(c[3], 4)] for c in INF["model_curve_antisymmetric_h1mm"]],
                "curve_columns": "theta_deg, f_MHz[, C2D_full, C2D_vac] (h = 1 mm)",
                "script": "digitize/admx_two_rod_infer.py (solver digitize/admx_two_rod_2d.py); results digitize/admx_two_rod_infer.json",
            },
        },
        "gaps": [
            {"parameter": "pivot radius and arm length", "issue": "not published", "proposal": "use INFERRED r_p = %g +/- 2 mm, a = %g +/- 2 mm; for 3D, sweep a over 55-61 mm" % (rp, a)},
            {"parameter": "alumina armature/axle geometry, end-cap holes", "issue": "material only", "proposal": "2D: omit. 3D: bar in each 0.5 mm end gap + axle through the end cap at the pivot; sensitivity study"},
            {"parameter": "rod end gaps", "issue": "'~0.020 in' (thesis) only", "proposal": "3D sweep 0.25-1.0 mm; 2D ideal = gap 0"},
            {"parameter": "rod-angle origin in 2312.16668 Fig. 8c", "issue": "model overlay (admx_two_rod_1a_fig8c_model_overlay.png): the 2D model follows the TM010c branch within ~1%% for 110-155 deg (4-7 MHz low at 112-128 deg, on it at 135-155 deg); below ~108 deg the model (620 MHz at 90 deg) runs BETWEEN the TM010c branch (flat at 637 MHz) and the lowest line (617-624 MHz): the published map shows an avoided crossing with a flat (3D, non-TM0n0) mode near 625-635 MHz that the z-uniform model cannot contain. Fig. 8a: 656.4 MHz at 120 deg vs model %.1f MHz" % [r["f_MHz_richardson"] for r in ref if r["label"].startswith("T4")][0], "proposal": "compare shapes after fitting a theta offset and rod-1 wall angle; do not use Fig. 8c absolute angles at better than ~3 deg"},
            {"parameter": "magnet field map", "issue": "only plotted (2010.00169 Fig. 4, 2312.16668 Fig. 3)", "proposal": "uniform B for 2D; for 3D use the axionFEM solenoid model; expect published C < C_2D by ~20-30%"},
            {"parameter": "rod configuration of 2010.06183 Fig. 8", "issue": "not stated", "proposal": "assume symmetric (normal data taking); the three filled crossings used anti-symmetric"},
            {"parameter": "wall conductivity / Q model", "issue": "not stated", "proposal": "Q is not a benchmark target (measured Q_L only)"},
        ],
        "inconsistencies": [
            {"description": "2010.00169 Sec. II.C: rods 'swept out the 0.09 m radius circular trajectories'", "evidence": "Fig. 2/16 arm (pivot ring to rod centre) = 58-60 mm; with a = 90 mm and the wall clearance r_p + a + 25.4 <= 210 the two rods would collide at theta = 180 (r_p - a <= 5 mm); the 2D fit to 580/890/773/656 MHz gives a = %.0f +/- %.0f mm" % (fit["a_mm"], fit["sigma_a_mm"]), "proposal": "treat 0.09 m as an error or a different quantity; use a = %g mm" % a},
            {"description": "rod material", "evidence": "'copper tuning rods' (1910.08638, 2010.06183, 2010.00169); 'copper-plated stainless steel' (2312.16668 p.5); 'hollow OFHC copper tubes with the end-caps welded on' (2408.03444)", "proposal": "irrelevant for f and C (PEC); Q not targeted"},
            {"description": "2312.16668 says C was simulated with COMSOL, but Fig. 8a carries CST's legend/style; Fig. 8b says 'Rod 1 Position', Fig. 8c 'Rod 2 Angle'", "evidence": "visual", "proposal": "record code as uncertain for Run 1A"},
            {"description": "Run 1A 120-deg frequency", "evidence": "Fig. 8a label 656.4 MHz at (120, 0) vs the digitized Fig. 8c TM010c branch 660-661 MHz at 120-121 deg", "proposal": "angle labels good to ~3 deg"},
            {"description": "2010.00169 Fig. 5 annotation", "evidence": "hand-drawn pivots asymmetric (129.5 / 120.2 mm), drawn rod radius 26.7 mm, right rod labelled 270 deg but drawn at 90.9 deg CCW; the drawn rods are not reachable with one (r_p, a) at exactly 180/270 deg (needs r_p - a = 67.6 and r_p = 119.1, a = 61.4)", "proposal": "use the label frequency 773 MHz with the nominal angles (2D model: %.1f MHz) and, separately, with the drawn rod positions (see analytic_crosschecks)" % [r["f_MHz_richardson"] for r in ref if r["label"].startswith("T3")][0]},
            {"description": "cavity size wording", "evidence": "'approximately 1 m ... 0.4 m in diameter' (2010.00169) vs 101 cm / 42 cm (2408.03444); 136 L vs 140 L (2010.06183 explains: rods subtracted)", "proposal": "R = 210 mm, H = 1010 mm"},
            {"description": "plating thickness", "evidence": "0.08 mm (2010.00169) vs 0.075 mm (2312.16668)", "proposal": "irrelevant for f, C"},
        ],
        "solver_class": {
            "axisymmetric_2d_possible": False,
            "cross_section_2d_possible": True,
            "needs_3d": True,
            "reason": "Rods are off-axis: no axisymmetric reduction. The ideal geometry (rods touching both end caps, no armatures) is z-uniform, so TM010-like f, the TM mode map of the cross-section and C with uniform B are a 2D cross-section (Helmholtz, E_z Dirichlet) problem: this reproduces the published 580/890/773/656 MHz to <= 0.5% with the inferred pivot geometry. 3D is needed for the 0.5 mm end gaps, the alumina armatures, TE/TEM and TM01p intruder modes that cause the mode crossings (Fig. 8c, Fig. 8 dips, Table I), and C with the real solenoid field map.",
        },
        "priority": "P1 (2D cross-section: f(theta), symmetric = anti-symmetric frequency check, C2D trends) / P2 (3D: crossings, C with field map)",
        "priority_reason": "flagship haloscope; geometry nearly complete once the pivot is inferred; the Fig. 2 symmetric/anti-symmetric pair is a clean 2D target (equal f, different C).",
        "readme_row": {
            "quantities": "Fig. 2 rod positions (sym/anti, equal f); f at (180,270) = 773 MHz, (120,0) = 656.4 MHz, range 580-890 MHz; Run 1A S12 mode map vs angle; C(f) Run 1A (COMSOL, 6% band) and Run 1B (CST); crossings table",
            "geometry_complete": "nearly: pivot radius/arm INFERRED (r_p = %g, a = %g mm, +/-2), armatures and ports missing" % (rp, a),
            "axisym_2d": "no; 2D cross-section feasible (z-uniform, no end gaps)",
            "needs_3d": "yes for end gaps/armatures, mode crossings and C with the field map",
        },
    }
    json.dump(J, open(os.path.join(REFS, "admx_two_rod.json"), "w"), indent=1)
    print("wrote admx_two_rod.json; r_p, a =", rp, a, " Run 1B theta range", th_680, th_790)


if __name__ == "__main__":
    main()
