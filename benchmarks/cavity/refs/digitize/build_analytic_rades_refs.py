"""Build the reference JSONs for the analytic empty-cylinder tables and RADES benchmarks 1-2.

Usage: python3 -I build_analytic_rades_refs.py
Writes REFS/analytic_admx_thesis_tab6_1.json, analytic_klash_tab2.json, analytic_birme_tab1.json,
rades_5cav_1803.json, rades_6cav_alt_2002.json (keeping any digitized curves already present in the last).
Tabulated numbers were transcribed by hand from the arXiv PDFs (text layer checked with PyMuPDF);
closed-form checks are computed here.  Then run rades_6cav_alt_2002_digitize.py to (re)fill the curves.
"""
import json
import os

import numpy as np
from scipy.special import jn_zeros, jnp_zeros

REFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C0 = 299792458.0
MU0 = 4e-7 * np.pi
ETA = MU0 * C0
X01, X02 = jn_zeros(0, 2)
XP11 = jnp_zeros(1, 1)[0]


def f_tm0np(R, L, n, p):
    x = jn_zeros(0, n)[-1]
    return C0 / (2 * np.pi) * np.sqrt((x / R) ** 2 + (p * np.pi / L) ** 2)


def f_te111(R, L):
    return C0 / (2 * np.pi) * np.sqrt((XP11 / R) ** 2 + (np.pi / L) ** 2)


def skin(f, sigma):
    return 1.0 / np.sqrt(np.pi * f * MU0 * sigma)


def q_tm01p(R, L, p, sigma):
    """Wall-loss Q of TM01p in a closed PEC-like cylinder (perturbative, R_s = 1/(sigma*delta)).
    p = 0: (R/delta)/(1+R/L);  p >= 1: (R/delta)/(1+2R/L)."""
    f = f_tm0np(R, L, 1, p)
    d = skin(f, sigma)
    return (R / d) / (1 + (1 if p == 0 else 2) * R / L)


def q_te_nml(R, L, n, m, l, sigma):
    """Pozar (Microwave Engineering, Sec. 6.4, circular cavity) TE_nml wall-loss Q."""
    pp = jnp_zeros(n, m)[-1]
    beta = l * np.pi / L
    k = np.sqrt((pp / R) ** 2 + beta ** 2)
    f = C0 * k / (2 * np.pi)
    Rs = 1.0 / (sigma * skin(f, sigma))
    num = (k * R) ** 3 * ETA * R * L / (4 * pp ** 2 * Rs) * (1 - (n / pp) ** 2)
    den = R * L / 2 * (1 + (beta * R * n / pp ** 2) ** 2) + (beta * R ** 2 / pp) ** 2 * (1 - n ** 2 / pp ** 2)
    return num / den


def pct(a, b):
    return round(100.0 * (a - b) / b, 4)


C010 = 4 / X01 ** 2
C020 = 4 / X02 ** 2


def common_cyl_solver_class():
    return {"axisymmetric_2d_possible": True, "needs_3d": False,
            "reason": "empty closed cylinder: TM0np exactly axisymmetric (m=0); TE111 is an m=1 azimuthal mode, reachable "
                      "with an azimuthal-order-m 2D formulation or as a 3D cross-check"}


def dump(name, J):
    with open(os.path.join(REFS, name + ".json"), "w") as f:
        json.dump(J, f, indent=1)


# --------------------------------------------------------------------------------------------
# A. ADMX (Braine thesis) Table 6.1
# --------------------------------------------------------------------------------------------
def admx():
    rows = [  # name, R_cm, H_cm, TM010, TE111, TM011, TM020 (GHz) as printed
        ["ADMX Main", "21.0", "101", "0.547", "0.444", "0.567", "1.25"],
        ["ADMX Sidecar", "4.60", "12.1", "2.50", "2.28", "2.79", "5.73"],
        ["ADMX run 2", "8.5", "100", "1.35", "1.04", "1.36", "3.1"],
        ["ADMX EFR", "6.4", "100", "1.79", "1.38", "1.8", "4.12"],
    ]
    checks = []
    for r in rows:
        R, H = float(r[1]) / 100, float(r[2]) / 100
        an = {"TM010": f_tm0np(R, H, 1, 0), "TE111": f_te111(R, H), "TM011": f_tm0np(R, H, 1, 1), "TM020": f_tm0np(R, H, 2, 0)}
        tab = dict(zip(["TM010", "TE111", "TM011", "TM020"], [float(v) for v in r[3:]]))
        checks.append({"cavity": r[0], "R_m": R, "H_m": H, "_strings": dict(zip(["TM010", "TE111", "TM011", "TM020"], r[3:])),
                       "closed_form_GHz": {k: round(v / 1e9, 5) for k, v in an.items()},
                       "table_GHz": tab,
                       "pct_diff_table_vs_closed_form": {k: pct(tab[k], an[k] / 1e9) for k in tab},
                       "C_TM010_closed_form": round(C010, 5), "C_TM020_closed_form": round(C020, 5),
                       "C_TM011_closed_form": 0.0})
    J = {
        "name": "analytic_admx_thesis_tab6_1",
        "title": "ADMX empty-cylinder mode frequencies (Braine PhD thesis Table 6.1)",
        "benchmark_rank": "analytic",
        "citations": [{"arxiv": "2408.03444", "authors_short": "T. Braine (PhD thesis, U. Washington 2024)", "journal": "thesis",
                       "items_used": ["Table 6.1 (PDF page 123, printed p. 96)", "Eqs. 6.21-6.22"]}],
        "software": {"code": "closed form (Eq. 6.21); the thesis does not say whether the table is computed or measured, "
                             "the caption says 'measurements of a single empty cavity' for run 2 / EFR",
                     "version": None, "solver_type": "analytic eigenfrequency", "boundary": "PEC", "mesh_notes": None},
        "geometry": {"units": "cm", "description": "four empty right circular cylinders (closed, no rods, no ports)",
                     "parameters": {f"{r[0]}": {"value": {"R": float(r[1]), "H": float(r[2])}, "unit": "cm",
                                                "source": "Table 6.1 (column header 'Radius'; caption: all lengths in cm)", "note": None}
                                    for r in rows}},
        "materials": {"walls": {"material": "PEC (frequencies only)", "sigma_S_per_m": None, "temperature_K": None, "source": "n/a"}},
        "conditions": {"temperature_K": None, "B0_T": None, "B0_direction": "z (axial), for the C cross-check only", "field_map": "uniform"},
        "conventions": {"form_factor_definition": "C = |int E.z dV|^2 / (V int |E|^2 dV) (not tabulated here; closed form 4/x0n^2 for TM0n0, 0 for TM0np p>=1)",
                        "Q_definition": "not tabulated", "notes": "f = c/(2 pi) sqrt((x/R)^2 + (p pi/H)^2), x = j0n for TM0n, j'11 for TE11"},
        "reference_values": {
            "tables": [{"source": "Table 6.1", "description": "empty-cavity base frequencies", "simulated": None,
                        "columns": ["cavity", "R", "H", "f_TM010", "f_TE111", "f_TM011", "f_TM020"],
                        "units": ["", "cm", "cm", "GHz", "GHz", "GHz", "GHz"], "rows": rows}],
            "curves": []},
        "analytic_check": {"method": "closed form, c = 299792458 m/s, PEC; tabulated values are rounded to 2-4 significant figures", "rows": checks},
        "measured_values": {"tables": [], "curves": []},
        "gaps": [
            {"parameter": "provenance (computed vs measured)", "issue": "caption says the run-2/EFR entries are 'measurements of a single empty cavity'; "
             "the main/Sidecar rows look computed. Precision is 2-4 significant figures only.",
             "proposal": "use as an analytic sanity check only; the axionFEM reference is the closed form (exact), the table is checked to its rounding"},
            {"parameter": "Sidecar radius", "issue": "Table 6.1 gives Sidecar R = 4.60 cm, H = 12.1 cm, but the Sidecar paper 1901.00920 quotes 0.064 m inner DIAMETER (R = 3.2 cm). "
             "The 4.60 cm row may be the later clamshell Sidecar cavity of the thesis.",
             "proposal": "take Table 6.1 numbers at face value (they are self-consistent with Eq. 6.21, see analytic_check); do not mix with 1901.00920"}],
        "inconsistencies": [],
        "solver_class": common_cyl_solver_class(),
        "priority": "P0", "priority_reason": "closed-form, exact, 2D axisymmetric; first regression test",
        "readme_row": {"quantities": "f TM010/TE111/TM011/TM020 (4 cavities)", "geometry_complete": "yes", "axisym_2d": "yes (TE111 needs m=1)", "needs_3d": "no"},
    }
    # flag rows that disagree beyond rounding
    for c in checks:
        for k, v in c["pct_diff_table_vs_closed_form"].items():
            tab = c["table_GHz"][k]
            txt = c["_strings"][k]
            ndec = len(txt.split(".")[1]) if "." in txt else 0
            tol = 100 * 0.5 * 10 ** (-ndec) / tab
            if abs(v) > tol + 1e-9:
                J["inconsistencies"].append({"description": f"{c['cavity']} {k}: table {tab} GHz vs closed form {c['closed_form_GHz'][k]} GHz ({v:+.2f}%), beyond rounding (+-{tol:.2f}%)",
                                             "evidence": "Table 6.1 vs Eq. 6.21 evaluated with the tabulated R, H",
                                             "proposal": "trust the closed form; record the table value only as a rounded sanity check"})
    for c in checks:
        c.pop("_strings")
    dump(J["name"], J)
    return J


# --------------------------------------------------------------------------------------------
# B. KLASH Table 2
# --------------------------------------------------------------------------------------------
def klash():
    R = 2.0
    sigma_rt = 5.8e7
    rows = [["TM010", "57.37", "153,000", "57.37", "146,000"], ["TE111", "57.74", "162,000", "61.35", "159,000"]]
    checks = []
    for L, ci in ((4.0, (1, 2)), (3.5, (3, 4))):
        an = {"TM010": (f_tm0np(R, L, 1, 0), q_tm01p(R, L, 0, sigma_rt)), "TE111": (f_te111(R, L), q_te_nml(R, L, 1, 1, 1, sigma_rt))}
        for r in rows:
            ft, qt = float(r[ci[0]]), float(r[ci[1]].replace(",", ""))
            fa, qa = an[r[0]]
            checks.append({"L_m": L, "mode": r[0], "f_closed_form_MHz": round(fa / 1e6, 4), "f_table_MHz": ft, "pct_diff_f": pct(ft, fa / 1e6),
                           "Q_closed_form_sigma5.8e7": round(qa), "Q_table": qt, "pct_diff_Q": pct(qt, qa)})
    J = {
        "name": "analytic_klash_tab2",
        "title": "KLASH 4 m x 4 m empty copper cylinder: TM010 and TE111 f, Q0 at 300 K (KLASH proposal Table 2)",
        "benchmark_rank": "analytic",
        "citations": [{"arxiv": "1707.06010", "authors_short": "D. Alesini et al. (KLASH proposal)", "journal": "arXiv preprint (LNF)",
                       "items_used": ["Table 2", "Fig. 3 (D = 4 m, L = 4 m)", "Table 3 (R = 2 m, L = 4 m, OFHC, Q0(4 K) = 450,000)"]}],
        "software": {"code": "not stated (field plots in Figs. 3-4 look like HFSS/CST)", "version": None, "solver_type": "eigenmode (assumed)",
                     "boundary": "copper walls at 300 K (conductivity not stated)", "mesh_notes": None},
        "geometry": {"units": "m", "description": "empty closed copper cylinder, two lengths",
                     "parameters": {"R": {"value": 2.0, "unit": "m", "source": "Fig. 3 'D=4 m'; Table 3 R = 2", "note": None},
                                    "L": {"value": [4.0, 3.5], "unit": "m", "source": "Table 2 column headers; Fig. 3 'L=4 m'", "note": None}}},
        "materials": {"walls": {"material": "OFHC copper", "sigma_S_per_m": None, "temperature_K": 300,
                                "source": "Table 3 'Oxygen Free Copper'; sigma not given - 5.8e7 S/m reproduces Table 2 Q to <1%"}},
        "conditions": {"temperature_K": 300, "B0_T": 0.6, "B0_direction": "z (axial, KLOE solenoid)", "field_map": "uniform"},
        "conventions": {"form_factor_definition": "not tabulated (closed form C010 = 0.6917)", "Q_definition": "unloaded wall-loss Q0 at 300 K",
                        "notes": "TE111 Q closed form: Pozar, Microwave Engineering, Sec. 6.4; TM010: (R/delta)/(1+R/L)"},
        "reference_values": {"tables": [{"source": "Table 2", "description": "f and Q0(300 K) of TM010 and nearest mode TE111", "simulated": True,
                                         "columns": ["mode", "f (L=4 m)", "Q0 300K (L=4 m)", "f (L=3.5 m)", "Q0 300K (L=3.5 m)"],
                                         "units": ["", "MHz", "1", "MHz", "1"], "rows": rows}],
                             "curves": []},
        "analytic_check": {"method": "closed form, PEC eigenfrequency + perturbative wall loss with sigma = 5.8e7 S/m (assumed)", "rows": checks},
        "measured_values": {"tables": [], "curves": []},
        "gaps": [{"parameter": "copper conductivity at 300 K", "issue": "not stated", "proposal": "use 5.8e7 S/m (IACS); check shows Table 2 Q matches to <1%"},
                 {"parameter": "solver", "issue": "software and mesh not stated", "proposal": "compare to closed form; table is a 4-significant-figure sanity check"}],
        "inconsistencies": [],
        "solver_class": common_cyl_solver_class(),
        "priority": "P0", "priority_reason": "closed-form, very large aspect-scale test (R = 2 m) of unit handling and Q scaling",
        "readme_row": {"quantities": "f, Q0(300 K) TM010/TE111, L = 4 and 3.5 m", "geometry_complete": "yes (sigma assumed)", "axisym_2d": "yes (TE111 needs m=1)", "needs_3d": "no"},
    }
    dump(J["name"], J)
    return J


# --------------------------------------------------------------------------------------------
# C. BI-RME Table I (cylinder with coaxial probe)
# --------------------------------------------------------------------------------------------
def birme_tab1():
    R, L, sigma = 0.045, 1.0, 6e7
    rows = [[1, "TM010", "2.55920", "33069.1", "0.68505"], [2, "TM011", "2.56334", "33260.6", "0.00523"],
            [3, "TM012", "2.57646", "33119.7", "0.00027"], [4, "TM013", "2.59817", "33292.4", "5.305e-5"],
            [5, "TM014", "2.62827", "33493.8", "1.710e-5"], [6, "TM015", "2.66646", "33781.9", "7.247e-6"]]
    checks = []
    f0t = float(rows[0][2])
    for m, name, f, q, c in rows:
        p = m - 1
        fa = f_tm0np(R, L, 1, p) / 1e9
        qa = q_tm01p(R, L, p, sigma)
        ent = {"mode": name, "f_closed_form_GHz": round(fa, 5), "f_table_GHz": float(f), "pct_diff_f": pct(float(f), fa),
               "Q_closed_form": round(qa, 1), "Q_table": float(q), "pct_diff_Q": pct(float(q), qa),
               "C_closed_form": round(C010, 5) if p == 0 else 0.0, "C_table": float(c)}
        if p >= 1:
            # effective length implied by f_p^2 - f_0^2 = p^2 (c/2L)^2, using the table's own f_0
            df2 = (float(f) ** 2 - f0t ** 2) * 1e18
            ent["L_eff_from_table_spacing_m"] = round(C0 * p / (2 * np.sqrt(df2)), 4)
        checks.append(ent)
    J = {
        "name": "analytic_birme_tab1",
        "title": "D = 90 mm, L = 1 m copper cylinder with axial coaxial probe: TM01p f, Q, C (BI-RME paper Table I, CST)",
        "benchmark_rank": "analytic",
        "citations": [{"arxiv": "2107.03137", "authors_short": "J.M. Garcia-Barcelo et al. (BI-RME 3D for haloscopes)", "journal": "J. Comput. Phys. (2022), arXiv v-of-record unverified",
                       "items_used": ["Sec. 4.2.1 text (geometry, sigma, probe, B)", "Table I", "Eq. (23) form factor", "Figs. 5-10 (not digitized)"]}],
        "software": {"code": "CST Studio Suite (eigenmode/driven) + MATLAB post-processing; BI-RME 3D", "version": None,
                     "solver_type": "eigenmode values tabulated (CST), wide-band BI-RME response in figures", "boundary": "lossy metal sigma = 6e7 S/m",
                     "mesh_notes": "not stated"},
        "geometry": {"units": "mm", "description": "closed copper cylinder; 50-ohm coaxial probe inserted at the centre of the top cap, designed for critical coupling of TM010",
                     "parameters": {"D": {"value": 90.0, "unit": "mm", "source": "Sec. 4.2.1 text", "note": "diameter"},
                                    "L": {"value": 1000.0, "unit": "mm", "source": "Sec. 4.2.1 text", "note": None},
                                    "coax_inner_radius_b": {"value": 0.635, "unit": "mm", "source": "Sec. 4.2.1 text", "note": None},
                                    "coax_outer_radius_a": {"value": 2.11, "unit": "mm", "source": "Sec. 4.2.1 text", "note": None},
                                    "coax_eps_r": {"value": 2.08, "unit": "1", "source": "Sec. 4.2.1 text (PTFE)", "note": None},
                                    "probe_insertion_depth": {"value": None, "unit": "mm", "source": "not given", "note": "'designed under critical coupling for TM010'"}}},
        "materials": {"walls": {"material": "metal", "sigma_S_per_m": 6e7, "temperature_K": None, "source": "Sec. 4.2.1"},
                      "coax_dielectric": {"material": "PTFE", "eps_r": 2.08, "tan_delta": None, "source": "Sec. 4.2.1"}},
        "conditions": {"temperature_K": None, "B0_T": 8.0, "B0_direction": "z (axial)", "field_map": "uniform"},
        "conventions": {"form_factor_definition": "Eq. (23): C_m = |int_V E_m . B_e dV|^2 / int_V ||B_e||^2 dV with E_m normalized (int |E_m|^2 dV = 1 implied by the BI-RME basis); "
                                                  "equivalent to the standard C = |int E.z|^2/(V int |E|^2)",
                        "Q_definition": "unloaded Q_m (wall losses)", "notes": "Table I modes are those coupled to the axial probe"},
        "reference_values": {"tables": [{"source": "Table I", "description": "TM01p modes of the probed cylinder", "simulated": True,
                                         "columns": ["m", "mode", "f_m", "Q_m", "C_m"], "units": ["", "", "GHz", "1", "1"], "rows": rows}],
                             "curves": []},
        "analytic_check": {"method": "closed form for the EMPTY cylinder (no probe), PEC frequencies, perturbative wall Q with sigma = 6e7: "
                                     "Q = (R/delta)/(1+R/L) for p=0, (R/delta)/(1+2R/L) for p>=1; C = 4/x01^2 for p=0, 0 otherwise", "rows": checks},
        "measured_values": {"tables": [], "curves": []},
        "gaps": [{"parameter": "probe insertion depth / pin geometry", "issue": "not given (only 'critical coupling for TM010')",
                  "proposal": "benchmark the EMPTY cylinder against the closed form; treat Table I as a probe-perturbed reference: model the probe as a flush (zero-depth) coax aperture first, then sweep depth 0-10 mm and report which depth reproduces f1 = 2.55920 GHz"},
                 {"parameter": "temperature / anomalous skin", "issue": "sigma = 6e7 S/m quoted without temperature", "proposal": "use sigma = 6e7 S/m, classical skin effect"}],
        "inconsistencies": [
            {"description": "TM010 frequency is 0.36% ABOVE the empty-cylinder closed form (2.5592 vs 2.5501 GHz) although an axial pin probe in the high-E region should LOWER it",
             "evidence": "Table I vs f = c x01/(2 pi R), R = 45 mm",
             "proposal": "check whether CST used R = 44.84 mm or a different D; benchmark f only at the 0.5% level unless the probe is modelled; record the discrepancy"},
            {"description": "Q_m increases with p in Table I (33069 -> 33782), whereas the closed form predicts Q(TM011) ~4% BELOW Q(TM010) (end-cap losses double for p>=1)",
             "evidence": "Table I Q column vs closed form (analytic_check)",
             "proposal": "use only Q(TM010) as a benchmark (1.2% below closed form, plausibly probe loss / mesh); report TM01p Q discrepancies"},
            {"description": "C for TM01p (p>=1) is exactly 0 for an ideal cylinder in uniform axial B; Table I gives 5.2e-3 ... 7e-6 (probe-induced asymmetry and numerical noise)",
             "evidence": "Table I", "proposal": "benchmark only C(TM010) = 0.68505 vs 0.69166 (-0.96%); p>=1 values are noise-level"},
            {"description": "TM01p frequency spacing implies an effective length 1.001-1.029 m (not 1 m) relative to the table's own TM010",
             "evidence": "analytic_check L_eff_from_table_spacing_m", "proposal": "consistent with a probe perturbation of mainly TM010/TM011; no action"}],
        "solver_class": {"axisymmetric_2d_possible": True, "needs_3d": False,
                         "reason": "cylinder + centred axial coax probe is fully axisymmetric (TM0np); the 2D solver needs a coax port/PEC pin for the probed case"},
        "priority": "P0", "priority_reason": "empty case is closed form; probed case is a 2D-axisym test of a centred pin",
        "readme_row": {"quantities": "f, Q, C of TM010-TM015 (probed cylinder)", "geometry_complete": "nearly (probe depth missing)", "axisym_2d": "yes", "needs_3d": "no"},
    }
    dump(J["name"], J)
    return J


# --------------------------------------------------------------------------------------------
# D. RADES 5-subcavity all-inductive filter (1803.01243)
# --------------------------------------------------------------------------------------------
def te101(a, b, L, sigma):
    f = C0 / 2 * np.sqrt(1 / a ** 2 + 1 / L ** 2)
    d = skin(f, sigma)
    q = 1 / d * a * b * L * (L ** 2 + a ** 2) / (L * a * (L ** 2 + a ** 2) + 2 * b * (L ** 3 + a ** 3))
    return f, q


def rades5():
    a, b = 22.86e-3, 10.16e-3
    sig = 2e9
    fL1, qL1 = te101(a, b, 26.68e-3, sig)
    fL2, qL2 = te101(a, b, 25.00e-3, sig)
    tab2 = [[1, "+ + + + +", "8.428", "0.65", "40386"], [2, "+ + 0 - -", "8.454", "3.2e-7", "42033"], [3, "- + + + -", "8.528", "8.1e-5", "43654"],
            [4, "- + 0 - +", "8.625", "1.6e-12", "45882"], [5, "- + - + -", "8.710", "6.4e-6", "48048"]]
    tab3 = [[1, "8.379", "8.398", "8.428", "0.028", "0.030"], [2, "8.399", "8.429", "8.454", "0.031", "0.025"], [3, "8.474", "8.504", "8.528", "0.030", "0.024"],
            [4, "8.572", "8.602", "8.625", "0.030", "0.023"], [5, "8.658", "8.687", "8.710", "0.030", "0.023"]]
    # coupled-resonator model check: tridiagonal with Omega_q from isolated lengths and k = -0.0185
    J = {
        "name": "rades_5cav_1803",
        "title": "RADES 5-subcavity all-inductive WR-90 filter (CAST-RADES 2018)",
        "benchmark_rank": "1",
        "citations": [{"arxiv": "1803.01243", "authors_short": "A. Alvarez Melcon et al. (RADES)", "journal": "JCAP 05 (2018) 040",
                       "items_used": ["Table 1 (dimensions 2 K / 298 K)", "Table 2 (CST eigenmodes)", "Table 3 (measured vs CST f)", "Sec. 3-4 text", "Eq. 2.11", "Eq. A.5"]},
                      {"arxiv": "2107.03137", "authors_short": "J.M. Garcia-Barcelo et al. (BI-RME 3D)", "journal": "see analytic_birme_tab1",
                       "items_used": ["Sec. 4.2.2", "Table II (identical to 1803.01243 Table 2)"]}],
        "software": {"code": "CST Microwave Studio", "version": None, "solver_type": "eigenmode (Table 2); text also mentions time-domain FIT for the design iterations",
                     "boundary": "lossy metal, sigma = 2e9 S/m", "mesh_notes": "not stated"},
        "geometry": {"units": "mm",
                     "description": "Five WR-90 rectangular sub-cavities in a row along z, coupled by four inductive irises (vertical full-height walls of thickness t with a centred aperture of width W; aperture position assumed centred). "
                                    "Cross-section a (x) by b (y). Order: L1 | iris | L2 | iris | L3 | iris | L4 | iris | L5. Probes (only in the S-parameter simulations/measurements): SMA, centre of the top (y=b) face of cavities 1 and 5; port-1 tip flush with the wall, port-2 tip retracted 1 mm.",
                     "parameters": {
                         "a": {"value": 22.86, "unit": "mm", "source": "Table 1, 2 K", "note": "298 K (incl. 30 um Cu coating): 22.99"},
                         "b": {"value": 10.16, "unit": "mm", "source": "Table 1, 2 K", "note": "298 K: 10.25"},
                         "L1=L5": {"value": 26.68, "unit": "mm", "source": "Table 1, 2 K", "note": "298 K: 26.82"},
                         "L2=L3=L4": {"value": 25.00, "unit": "mm", "source": "Table 1, 2 K", "note": "298 K: 25.14"},
                         "W (iris aperture width, all 4)": {"value": 8.00, "unit": "mm", "source": "Table 1, 2 K", "note": "298 K: 8.14"},
                         "t (iris thickness)": {"value": 2.00, "unit": "mm", "source": "Table 1, 2 K", "note": "298 K: 1.95 (coating on both faces)"},
                         "iris aperture height": {"value": 10.16, "unit": "mm", "source": "inferred: inductive iris = full height b", "note": None},
                         "iris aperture x-position": {"value": "centred", "unit": None, "source": "inferred (standard symmetric inductive iris); Fig. 2 field plots consistent", "note": None},
                         "corner rounding radius": {"value": None, "unit": "mm", "source": "text p. 7: corners rounded to facilitate machining; radius not given", "note": None},
                         "total length": {"value": 26.68 * 2 + 25.0 * 3 + 4 * 2.0, "unit": "mm", "source": "inferred: sum of Table 1 lengths", "note": None}}},
        "materials": {"walls": {"material": "316L stainless steel with ~30 um Cu coating; simulated as Cu", "sigma_S_per_m": 2e9, "temperature_K": 2,
                                "source": "Sec. 3 text (Table 2 uses sigma = 2e9 S/m); RRR 30-200 expected"}},
        "conditions": {"temperature_K": 2, "B0_T": 9.0, "B0_direction": "y (along the narrow side b, CAST LHC dipole), parallel to TE101 E-field", "field_map": "uniform"},
        "conventions": {"form_factor_definition": "Eq. A.5: G_m = (1/(B_e V)) int_V B_e . E_m d^3x with E_m normalized so that int |E_m|^2 = V; Table 2 lists G^2 (= standard C). "
                                                  "Volume: total cavity volume (whether iris apertures are included is not stated; effect O(1%)). Single-cavity TE101 limit: G^2 = (8/pi^2)^2 = 0.657",
                        "Q_definition": "unloaded Q_u = omega U / P_c (Eq. 2.10), wall losses only",
                        "notes": "Table 2 eigenmodes presumably WITHOUT the coaxial probes (probes were added for separate S-parameter simulations, Sec. 4)"},
        "reference_values": {
            "tables": [{"source": "1803.01243 Table 2", "description": "CST eigenmodes at 2 K: field pattern (signs of eigenvector), f, G^2, Q_u", "simulated": True,
                        "columns": ["mode", "E-field pattern", "f", "G^2", "Q_u (2 K)"], "units": ["", "", "GHz", "1", "1"], "rows": tab2},
                       {"source": "2107.03137 Table II", "description": "BI-RME paper 'parameters of the RADES haloscope': numerically IDENTICAL to 1803.01243 Table 2 (see inconsistencies)",
                        "simulated": True, "columns": ["m", "pattern", "f_m", "Q_m", "C_m"], "units": ["", "", "GHz", "1", "1"],
                        "rows": [[r[0], r[1], r[2], r[4], r[3]] for r in tab2]}],
            "curves": [],
            "curves_note": "Fig. 5 is measured S12 + fitted analytic model + axion power (no CST curve) -> not digitized as a simulation reference; Fig. 2 is field maps only. Measured peak frequencies are tabulated in Table 3."},
        "analytic_check": {"method": "isolated TE101 cavities (closed form, sigma = 2e9 S/m, Eq. 2.11) - NOT the coupled filter; useful only for scale",
                           "rows": [{"cavity": "L = 26.68 mm", "f_TE101_GHz": round(fL1 / 1e9, 4), "Q_TE101_Eq2.11": round(qL1)},
                                    {"cavity": "L = 25.00 mm", "f_TE101_GHz": round(fL2 / 1e9, 4), "Q_TE101_Eq2.11": round(qL2)},
                                    {"cavity": "single TE101", "G2": round(64 / np.pi ** 4, 4)}],
                           "notes": "Irises load the cavities strongly: coupled-filter modes sit 0.2-0.5 GHz below the isolated-cavity TE101 frequencies; the paper says the k = -0.0185 coupling and L1/5 were tuned numerically in CST"},
        "measured_values": {"tables": [{"source": "1803.01243 Table 3", "description": "measured peak frequencies at 298 K and 2 K vs CST (Table 2)", "simulated": False,
                                        "columns": ["mode", "f_298K", "f_2K", "f_CST", "(f_298K - f_2K) as printed", "(f_CST - f_2K) as printed"],
                                        "units": ["", "GHz", "GHz", "GHz", "GHz", "GHz"], "rows": tab3},
                                       {"source": "1803.01243 Sec. 4 text", "description": "measured Q", "simulated": False,
                                        "columns": ["quantity", "value"], "units": ["", ""],
                                        "rows": [["Q_L at 8.4 GHz", "~6000"], ["Q_u if critically coupled", "~12000"], ["port-1 coupling (|S11|)", "~0.64"], ["Q_u implied", "~16000 (3-4x below CST)"]]}],
                            "curves": []},
        "gaps": [
            {"parameter": "corner rounding radius", "issue": "text says corners are rounded for machining; radius not given",
             "proposal": "baseline with sharp corners; sweep r = 0, 0.5, 1.0, 1.5 mm (vertical edges of cavities and iris apertures) and quote df/dr; expect MHz-level shifts, i.e. below the paper's 3-digit rounding (1 MHz)"},
            {"parameter": "iris aperture position and edge shape", "issue": "W and t given; centring not stated explicitly", "proposal": "assume centred, square-edged apertures"},
            {"parameter": "probes in Table 2", "issue": "not stated whether the eigenmode run included the SMA probes", "proposal": "simulate without probes (closed eigenproblem); optional run with flush port-1 pin and 1-mm retracted port-2 pin to bound the shift"},
            {"parameter": "G^2 integration volume", "issue": "whether irises are included in V", "proposal": "compute both; difference O(t W b / V) ~ 1%"},
            {"parameter": "B direction", "issue": "dipole field along the narrow side b (y) - stated implicitly (TE101 E along y, CAST dipole)", "proposal": "uniform B along y"}],
        "inconsistencies": [
            {"description": "Table 3 mode 1: f_298K = 8.379, f_2K = 8.398 give a difference of 0.019 GHz, but the column prints 0.028 (all other modes ~0.030). Either f_298K(mode 1) = 8.370 or the difference is mistyped. Column header is f_298K - f_2K but values are f_2K - f_298K (sign).",
             "evidence": "1803.01243 Table 3", "proposal": "irrelevant for the CST benchmark (Table 2); flag if measured values are used"},
            {"description": "Table 3 caption refers to 'column 3' for the 298 K - 2 K difference; it is column 4",
             "evidence": "1803.01243 Table 3 caption", "proposal": "none"},
            {"description": "2107.03137 Table II is described in the earlier literature survey as an 'independent BI-RME solution', but it is numerically identical to 1803.01243 Table 2 (same f to 1 MHz, same Q to 1 unit, same G^2 incl. 1.6e-12) and the text says it 'summarizes the most relevant parameters' of the structure. It is a copy, not an independent cross-check.",
             "evidence": "2107.03137 Sec. 4.2.2 and Table II vs 1803.01243 Table 2; source: the earlier literature survey", "proposal": "treat the CST Table 2 as the single reference; the BI-RME Figs. 12-14 (response, Pw) are BI-RME's own output but are probe-dependent"},
            {"description": "Text says Q_u ~ 5.5 mm/delta from Eq. 2.11 with delta = 0.1 um (Q ~ 55,000), while sigma = 2e9 S/m gives delta = 0.123 um at 8.43 GHz (Q ~ 45,000) and the CST Q is 40,386-48,048",
             "evidence": "1803.01243 Sec. 3 text; analytic_check", "proposal": "use sigma = 2e9 S/m directly, not delta = 0.1 um"}],
        "solver_class": {"axisymmetric_2d_possible": False, "needs_3d": True,
                         "reason": "rectangular multi-cavity with thin irises; no rotational symmetry (2D waveguide cross-section reasoning gives only the TE10 cutoff, not the coupled filter modes)"},
        "priority": "P1", "priority_reason": "fully specified, tabulated CST eigenmodes (f to 1 MHz, G^2, Q); first 3D benchmark after the 3D pillbox/rectangular TM110 checks",
        "readme_row": {"quantities": "5 modes: f, G^2 (C), Q_u at 2 K (CST); measured f (Table 3)", "geometry_complete": "nearly (corner radius missing)", "axisym_2d": "no", "needs_3d": "yes"},
    }
    dump(J["name"], J)
    return J


# --------------------------------------------------------------------------------------------
# E. RADES 6-subcavity alternating filter (2002.07639)
# --------------------------------------------------------------------------------------------
def rades6():
    jp = os.path.join(REFS, "rades_6cav_alt_2002.json")
    old_curves, old_meas = [], []
    if os.path.exists(jp):
        try:
            O = json.load(open(jp))
            old_curves = O.get("reference_values", {}).get("curves", [])
            old_meas = O.get("measured_values", {}).get("curves", [])
        except Exception:
            pass
    dims = [  # parameter, 2K design, 300K thermally expanded, 300K manufactured
        ["cavity width (a)", "22.860", "22.929", "22.860"], ["cavity height (b)", "10.160", "10.191", "10.272"],
        ["length external cavities (l_ext)", "29.400", "29.489", "29.567"], ["length internal cavities (l_int)", "27.000", "27.081", "27.167"],
        ["inductive iris width (w)", "8.900", "8.987", "8.986"], ["inductive iris thickness (t_ind)", "2.000", "2.006", "1.939"],
        ["capacitive iris height (h)", "1.000", "1.063", "1.019"], ["capacitive iris thickness (t_cap)", "3.500", "3.511", "3.436"],
        ["capacitive iris gap (g)", "1.100", "1.104", "1.192"]]
    exp = 22.929 / 22.860
    scale_check = []
    for p, d2, d3, _ in dims:
        pred = float(d2) * exp
        scale_check.append({"parameter": p, "2K": float(d2), "300K_printed": float(d3), "2K_times_(a300/a2)": round(pred, 3), "diff_mm": round(float(d3) - pred, 3)})
    tab2 = [[1, "8.298", "1.820e-9", "42878", "7300"], [2, "8.319", "1.257e-3", "43758", "7448"], [3, "8.395", "1.007e-8", "43626", "7426"],
            [4, "8.508", "0.534", "40657", "6957"], [5, "8.613", "1.511e-8", "44168", "7563"], [6, "8.698", "3.418e-5", "44454", "7597"]]
    J = {
        "name": "rades_6cav_alt_2002",
        "title": "RADES 6-subcavity alternating inductive/capacitive WR-90 filter (2020)",
        "benchmark_rank": "2",
        "citations": [{"arxiv": "2002.07639", "authors_short": "A. Alvarez Melcon et al. (RADES)", "journal": "JHEP 07 (2020) 084",
                       "items_used": ["Table 1 (design / expanded / as-built)", "Table 2 (CST eigenmodes 2 K and 300 K)", "Table 3 (circuit)", "Figs. 4, 6, 7 (geometry)",
                                      "Fig. 5 (k vs w, k vs h)", "Fig. 6 right (k vs h with protrusions)", "Fig. 10 (S21 CST 2 K vs ideal)", "Fig. 11 text (nominal/mean G^2)", "Fig. 13 (as-built CST vs measured, RT)"]}],
        "software": {"code": "CST Microwave Studio", "version": None, "solver_type": "eigenmode (Table 2, Figs. 5-6 via PEC/PMC symmetry planes); frequency-domain S-parameters (Figs. 10, 13)",
                     "boundary": "lossy Cu, sigma_2K = 2.008e9 S/m, sigma_RT = 5.8e7 S/m", "mesh_notes": "not stated; CST optimiser used on l_ext, l_int, w, h maximizing G4^2"},
        "geometry": {"units": "mm",
                     "description": "Six WR-90 sub-cavities along z (Fig. 7 axes: x along a, y along b, z along the filter). Iris sequence: "
                                    "l_ext | CAP | l_int | IND | l_int | CAP | l_int | IND | l_int | CAP | l_ext (3 capacitive, 2 inductive irises, k1 = +0.0248, k2 = -0.0248). "
                                    "Inductive iris: wall of thickness t_ind with a full-height aperture of width w (Fig. 4 top). "
                                    "Capacitive ('quasi-capacitive') iris: wall of thickness t_cap with a full-width horizontal slot of height h, with protrusions of width g (Fig. 6 left). "
                                    "SMA probes in the first and last cavity (top face); port 1 critically coupled at 2 K, port 2 minimal pin.",
                     "parameters": {r[0]: {"value": float(r[1]), "unit": "mm", "source": "Table 1, T = 2 K design",
                                           "note": f"300 K thermally expanded: {r[2]}; 300 K manufactured (measured +-3 um): {r[3]}"} for r in dims} | {
                         "total length (2K)": {"value": round(2 * 29.4 + 4 * 27.0 + 3 * 3.5 + 2 * 2.0, 3), "unit": "mm", "source": "inferred: 2 l_ext + 4 l_int + 3 t_cap + 2 t_ind", "note": None},
                         "capacitive slot vertical position": {"value": None, "unit": "mm", "source": "not given (Fig. 4 bottom / Fig. 7 inset suggest the slot is near the bottom wall)", "note": None},
                         "protrusion geometry (g)": {"value": 1.1, "unit": "mm", "source": "text p. 15 'protrusions of width 1.1 mm'; Fig. 6 left", "note": "shape/extent of the protrusions not dimensioned"},
                         "corner rounding radius": {"value": None, "unit": "mm", "source": "not given", "note": None}}},
        "materials": {"walls": {"material": "316LN stainless steel + 30 um Cu coating; simulated as solid Cu", "sigma_S_per_m": {"2K": 2.008e9, "300K": 5.8e7},
                                "temperature_K": [2, 300], "source": "Sec. 3.1 text; footnote 6: coated samples suggest sigma_2K ~ 1e9 in practice"}},
        "conditions": {"temperature_K": 2, "B0_T": None, "B0_direction": "y (dipole along the narrow side b), as for the 2018 RADES filter", "field_map": "uniform"},
        "conventions": {"form_factor_definition": "G_m^2 as in 1803.01243 Eq. A.5 (normalized |int E.B|^2/(B^2 V int |E|^2)); Table 2 col. 3", "Q_definition": "unloaded Q0, wall losses",
                        "notes": "coupling k from even/odd (PMC/PEC mid-plane) eigenfrequencies: |k| = |fe^2 - fo^2|/(fe^2 + fo^2) (Eq. 3.2); k2 < 0 inductive, k1 > 0 capacitive"},
        "reference_values": {
            "tables": [{"source": "Table 2", "description": "CST eigenmodes: f, G^2, Q0 at 2 K (design dims) and Q0 at 300 K (thermally expanded dims, sigma_RT)", "simulated": True,
                        "columns": ["mode", "f", "G^2", "Q0 (2 K)", "Q0 (300 K)"], "units": ["", "GHz", "1", "1", "1"], "rows": tab2},
                       {"source": "Fig. 11 caption", "description": "tolerance Monte Carlo (800 CST runs, +-30 um normal): G4^2 nominal and mean", "simulated": True,
                        "columns": ["nominal G4^2", "mean G4^2"], "units": ["1", "1"], "rows": [["0.5340", "0.5383"]]},
                       {"source": "Table 3", "description": "ideal lumped circuit (lossless) used for synthesis", "simulated": False,
                        "columns": ["component", "value"], "units": ["", ""],
                        "rows": [["L1", "45.89 nH"], ["C1", "7.82 fF"], ["k1", "+0.0248"], ["L2", "42.84 nH"], ["C2", "8.17 fF"], ["k2", "-0.0248"]]}],
            "curves": old_curves},
        "measured_values": {"tables": [{"source": "Sec. 3.3 text", "description": "measured RT quality factors of mode 4", "simulated": False,
                                        "columns": ["quantity", "value"], "units": ["", ""],
                                        "rows": [["Q_L (-3 dB)", "1625"], ["Q_ext", "5318"], ["Q0 (50 screws)", "2340"], ["Q0 (14 screws)", "1620"], ["Q0 simulated RT (Table 2)", "6957"]]}],
                            "curves": old_meas},
        "analytic_check": {"method": "uniform thermal-expansion check of Table 1 col. 2 (scale factor a_300/a_2 = %.6f)" % exp, "rows": scale_check},
        "gaps": [
            {"parameter": "capacitive iris: slot vertical position and protrusion shape", "issue": "h, t_cap, g are given but the slot's y-position and the exact protrusion outline (Fig. 6 left) are not dimensioned",
             "proposal": "build the iris from Fig. 6 left (U-shaped lip of width g around a full-width slot); place the slot as drawn; calibrate by reproducing Fig. 6 right k1(h) at h = 0.93-1.0 mm with the PEC/PMC symmetry-plane method before running the full filter"},
            {"parameter": "inductive aperture position", "issue": "centring not stated (Fig. 4 top drawing is ambiguous)", "proposal": "assume centred; verify against Fig. 5 left k2(w)"},
            {"parameter": "corner rounding", "issue": "not given", "proposal": "sharp corners; sweep 0-1.5 mm as for benchmark 1"},
            {"parameter": "probes", "issue": "SMA pin depths not dimensioned", "proposal": "Table 2 as closed eigenproblem without probes; S21 figures need ports (deferred)"},
            {"parameter": "B0 magnitude", "issue": "not needed for f, G^2, Q", "proposal": "n/a"}],
        "inconsistencies": [
            {"description": "Capacitive iris height: text gives the chosen h = 0.93 mm (k = 0.0248; Fig. 6 right marker digitized at h = 0.928), Table 1 design gives h = 1.000 mm. The text says h was re-optimized in the final CST stage, so 1.000 is presumably final; note the Fig. 6 right curve gives k1(1.000) = 0.0270, not 0.0248.",
             "evidence": "p. 15 text vs Table 1", "proposal": "use Table 1 (h = 1.000 mm) for Table 2 / Fig. 10; use 0.93 mm only for the Fig. 6 coupling check"},
            {"description": "Inductive iris width: text says w = 8.90 mm at k = -0.0248, but the Fig. 5 left marker sits at w = 9.06 mm (k = -0.0250); the digitized curve gives k2(8.90) = -0.0238",
             "evidence": "Fig. 5 left vector marker vs p. 15 text / Table 1", "proposal": "use Table 1 w = 8.900; note the 2% k mismatch as a figure-vs-text inconsistency"},
            {"description": "Table 1 'thermally expanded' column is not a uniform scaling: w and h are 0.060-0.087 mm larger than 2K x (a_300/a_2) (exactly +60 um = 2 x 30 um coating for w and h), while a, b, l, t are pure expansion. Coating is applied inconsistently.",
             "evidence": "analytic_check rows (diff_mm)", "proposal": "for the 300 K Q0 column of Table 2, reproduce with the printed col. 2 dimensions as-is"},
            {"description": "As-built a = 22.860 mm at 300 K equals the 2 K design value (expected ~22.93); b, l larger than expanded",
             "evidence": "Table 1 col. 3", "proposal": "use col. 3 as printed for the Fig. 13 as-built CST overlay"},
            {"description": "Fig. 10 annotation: axion coupling at 8.506 GHz vs Table 2 mode 4 = 8.508 GHz; CST S21 peaks (digitized) 8.3016, 8.3219, 8.3968, 8.5056, 8.6096, 8.6915 GHz vs Table 2 8.298, 8.319, 8.395, 8.508, 8.613, 8.698 (probe loading in the S-parameter model; mode 6 differs by 6.5 MHz)",
             "evidence": "Fig. 10 vs Table 2", "proposal": "benchmark eigenfrequencies against Table 2; S21 peaks only at the few-MHz level"},
            {"description": "the earlier literature survey lists 'Figs. 5-6: k(w), k(h)' for this paper; Fig. 5 contains both k2(w) and the plain capacitive k1(h), Fig. 6 right the protruded-iris k1(h) actually used",
             "evidence": "2002.07639 Figs. 5-6", "proposal": "use Fig. 6 right for the as-built iris"}],
        "solver_class": {"axisymmetric_2d_possible": False, "needs_3d": True, "reason": "rectangular multi-cavity with inductive and slotted capacitive irises"},
        "priority": "P1", "priority_reason": "fully dimensioned (design, expanded, as-built), tabulated eigenmodes at 2 K and 300 K, plus k-curves that isolate the iris model; 3D",
        "readme_row": {"quantities": "6 modes f, G^2, Q0 (2 K, 300 K); k2(w), k1(h) curves; S21 (CST 2 K, as-built RT vs measured)", "geometry_complete": "nearly (cap-iris slot position/protrusion outline)",
                       "axisym_2d": "no", "needs_3d": "yes"},
    }
    dump(J["name"], J)
    return J


if __name__ == "__main__":
    for fn in (admx, klash, birme_tab1, rades5, rades6):
        J = fn()
        print(J["name"], json.dumps(J.get("analytic_check", {}).get("rows", [])[:8])[:1500])
        print("  inconsistencies:", len(J["inconsistencies"]))
