"""Cavity haloscopes in 3D (beta): eigenmodes, targeted shift-invert, form factor, Q0, overlaps.

    python benchmarks/cavity_3d_validation.py [workdir] [--only mc5+mc6+mc7] [--ranks N] [--fine]

Run as a batch job (3D). Writes benchmarks/records/cavity_mc{5,6,7}_*.json and
regenerates docs/validation/cavity_3d.md from every record present. References: closed forms derived
independently of the code, axionfem.cavity (cylinder_modes_3d, box_modes,
coax_tm_mode, sleeve_tm_mode, eta_tm010_te01p) and the axisymmetric solver (2D) where no closed form exists.
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
OUT = ROOT / "docs" / "validation" / "cavity_3d.md"
A, L = 0.1, 0.2
RHO = 1.7e-8
WALL = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": RHO, "T_K": 4.0}}
FILES = {"mc5": "cavity_mc5_modes.json", "mc6": "cavity_mc6_shift_invert.json", "mc7": "cavity_mc7_form_factor_q0.json",
         "rod": "cavity_mc7_floating_rod.json"}
RANKS = 16
FINE = False


def _solve(d, wd, ranks=None):
    t0 = time.time()
    r = d.solve(wd, verbose=0, mpi_ranks=ranks or RANKS)
    return r, time.time() - t0


def _eig(r):
    return r.data["fullwave"]["eigenmodes"]


def _diag(r):
    e = _eig(r)
    return {k: e[k] for k in ("requested", "returned", "converged", "max_relative_residual", "max_mass_overlap")}


def _match_sorted(got, ref):
    """Pair sorted computed and analytic frequency lists of equal length (degenerate members counted)."""
    g = sorted(got, key=lambda m: m["f_Hz"])
    return list(zip(g, ref[:len(g)]))


def _near(modes, f):
    return min(modes, key=lambda m: abs(m["f_Hz"] - f))


# ----------------------------------------------------------------------------------------------------- lowest modes
def mc5(wd: Path):
    rec = {"case": "closed cylinder a = 0.1 m, L = 0.2 m (mesh.domain_solid wall, vacuum), AME, ND2", "pillbox": [],
           "box": [], "quarter": []}
    ref = cv.cylinder_modes_3d(A, L, 3e9)
    hs = [0.04, 0.02, 0.01] + ([0.0075] if FINE else [])
    for h in hs:
        r, dt = _solve(cv.cylinder_design_3d(A, L, h=h, modes=10), wd / f"mc5_pill_{h}")
        rows = []
        for m, x in _match_sorted(_eig(r)["modes"], ref):
            rows.append({"name": x["name"], "f_Hz": m["f_Hz"], "f_exact_Hz": x["f_Hz"], "rel": m["f_Hz"] / x["f_Hz"] - 1,
                         "residual_true": m["residual_true"], "divergence_fraction": m["divergence_fraction"]})
        rec["pillbox"].append({"h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt, "diag": _diag(r),
                               "solver_seconds": _eig(r)["solver"]["seconds"], "modes": rows})
        print("mc5 pillbox", h, max(abs(x["rel"]) for x in rows), flush=True)
    # rectangular box: TM110 (lowest mode)
    ab = (0.2, 0.15, 0.1)
    bref = cv.box_modes(*ab, 3e9)
    for h in [0.03, 0.015] + ([0.0075] if FINE else []):
        r, dt = _solve(cv.box_design_3d(*ab, h=h, modes=4), wd / f"mc5_box_{h}")
        rows = [{"name": x["name"], "f_Hz": m["f_Hz"], "f_exact_Hz": x["f_Hz"], "rel": m["f_Hz"] / x["f_Hz"] - 1}
                for m, x in _match_sorted(_eig(r)["modes"], bref)]
        rec["box"].append({"h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt, "diag": _diag(r), "modes": rows})
        print("mc5 box", h, rows[0], flush=True)
    # quarter models: x = 0 PEC / y = 0 PMC, and both PMC
    for sym in ({"x": "pec", "y": "pmc"}, {"x": "pmc", "y": "pmc"}):
        sref = cv.cylinder_modes_3d(A, L, 3e9, sym)
        for h in (0.02, 0.01):
            r, dt = _solve(cv.cylinder_design_3d(A, L, h=h, modes=6, symmetry=sym), wd / f"mc5_q_{sym['x']}_{h}")
            rows = [{"name": x["name"], "f_Hz": m["f_Hz"], "f_exact_Hz": x["f_Hz"], "rel": m["f_Hz"] / x["f_Hz"] - 1,
                     "C": m["form_factor"].get("C"), "C_exact": x["C"]}
                    for m, x in _match_sorted(_eig(r)["modes"], sref)]
            rec["quarter"].append({"symmetry": sym, "h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt,
                                   "diag": _diag(r), "modes": rows})
            print("mc5 quarter", sym, h, max(abs(x["rel"]) for x in rows), flush=True)
    return rec


# ----------------------------------------------------------------------------------------------------- shift-invert
def mc6(wd: Path):
    rec = {"case": "closed cylinder a = 0.1 m, L = 0.2 m; STRUMPACK shift-invert Lanczos, ND2", "window": [],
           "ranks": [], "exact_target": {}, "floating_rod": {}}
    ref = cv.cylinder_modes_3d(A, L, 4e9)
    ft, nev = 2.1e9, 17  # around the 15th/16th mode (TE212 pair, 2.0906 GHz); 17 closes the TE113 pair
    h = 0.02
    ame, _ = _solve(cv.cylinder_design_3d(A, L, h=h, modes=12), wd / "mc6_ame")
    for hh in [0.02, 0.01]:
        r, dt = _solve(cv.cylinder_design_3d(A, L, h=hh, modes=nev, target_f_Hz=ft), wd / f"mc6_si_{hh}")
        got = _eig(r)["modes"]
        dmax = max(abs(m["f_Hz"] ** 2 - ft ** 2) for m in got)
        inwin = [x for x in ref if abs(x["f_Hz"] ** 2 - ft ** 2) <= dmax * (1 + 1e-3)]
        rows = [{"name": x["name"], "f_Hz": m["f_Hz"], "f_exact_Hz": x["f_Hz"], "rel": m["f_Hz"] / x["f_Hz"] - 1,
                 "residual_true": m["residual_true"], "divergence_fraction": m["divergence_fraction"],
                 "rayleigh_ratio_minus_1": m["rayleigh_ratio"] - 1} for m, x in _match_sorted(got, inwin)]
        row = {"h_m": hh, "target_f_Hz": ft, "modes_requested": nev, "analytic_in_window": len(inwin),
               "dofs": r.data["mesh"]["dofs"], "seconds": dt, "diag": _diag(r), "solver": _eig(r)["solver"],
               "modes": rows}
        if hh == h:  # same mesh as the AME run: identical eigenvalues where the sets overlap
            low = _eig(ame)["modes"]
            row["vs_ame"] = [{"f_Hz": m["f_Hz"], "rel_vs_ame": m["f_Hz"] / _near(low, m["f_Hz"])["f_Hz"] - 1}
                             for m in got if abs(_near(low, m["f_Hz"])["f_Hz"] / m["f_Hz"] - 1) < 1e-5]
        rec["window"].append(row)
        print("mc6 window", hh, len(inwin), len(got), max(abs(x["rel"]) for x in rows), flush=True)
    # 1 vs 4 vs 16 MPI ranks on one mesh (the mesh is generated by rank 0 and identical)
    base = None
    for nr in (1, 4, 16):
        r, dt = _solve(cv.cylinder_design_3d(A, L, h=h, modes=nev, target_f_Hz=ft), wd / f"mc6_ranks_{nr}", ranks=nr)
        fs = sorted(m["f_Hz"] for m in _eig(r)["modes"])
        base = base or fs
        rec["ranks"].append({"ranks": nr, "seconds": dt, "solver_seconds": _eig(r)["solver"]["seconds"],
                             "factorization_seconds": _eig(r)["solver"]["factorization_seconds"],
                             "lanczos_seconds": _eig(r)["solver"]["lanczos_seconds"],
                             "max_rel_diff_vs_1_rank": max(abs(a / b - 1) for a, b in zip(fs, base))})
        print("mc6 ranks", rec["ranks"][-1], flush=True)
    # target exactly at a non-degenerate eigenfrequency (TM010) of the same mesh, copied at full precision from a
    # shift-invert run: the Lanczos breakdown restart must fire (a degenerate target keeps beta/alpha ~ 1)
    rb, _ = _solve(cv.cylinder_design_3d(A, L, h=h, modes=6, target_f_Hz=1.2e9), wd / "mc6_exact_base")
    fexact = min(m["f_Hz"] for m in _eig(rb)["modes"])
    r, dt = _solve(cv.cylinder_design_3d(A, L, h=h, modes=8, target_f_Hz=fexact), wd / "mc6_exact")
    got = _eig(r)["modes"]
    rec["exact_target"] = {"target_f_Hz": fexact, "mode": "TM010", "returned": len(got), "requested": 8, "diag": _diag(r),
                           "solver": _eig(r)["solver"],
                           "nearest_rel": min(abs(m["f_Hz"] / fexact - 1) for m in got),
                           "max_residual_true": max(m["residual_true"] for m in got)}
    print("mc6 exact", rec["exact_target"]["returned"], rec["exact_target"]["nearest_rel"], flush=True)
    # floating PEC rod (radius 2 cm on the axis, 2 cm gaps at both caps): a second boundary component
    rod = {"radius": 0.02, "gap": 0.02}
    ra, _ = _solve(cv.cylinder_design_3d(A, L, h=h, modes=6, rod=rod), wd / "mc6_float_ame")
    rs, _ = _solve(cv.cylinder_design_3d(A, L, h=h, modes=6, rod=rod, target_f_Hz=1.2e9), wd / "mc6_float_si")
    rw, _ = _solve(cv.cylinder_design_3d(A, L, h=h, modes=12, rod=rod, target_f_Hz=0.6e9), wd / "mc6_float_si_low")
    rec["floating_rod"] = {
        "rod": rod, "h_m": h,
        "ame": {"diag": _diag(ra), "f_Hz": [m["f_Hz"] for m in _eig(ra)["modes"]],
                "residual_true": [m["residual_true"] for m in _eig(ra)["modes"]]},
        "shift_invert": {"target_f_Hz": 1.2e9, "diag": _diag(rs), "solver": _eig(rs)["solver"],
                         "f_Hz": [m["f_Hz"] for m in _eig(rs)["modes"]],
                         "residual_true": [m["residual_true"] for m in _eig(rs)["modes"]],
                         "divergence_fraction": [m["divergence_fraction"] for m in _eig(rs)["modes"]]},
        "shift_invert_low": {"target_f_Hz": 0.6e9, "diag": _diag(rw), "solver": _eig(rw)["solver"],
                             "f_Hz": [m["f_Hz"] for m in _eig(rw)["modes"]],
                             "residual_true": [m["residual_true"] for m in _eig(rw)["modes"]]}}
    print("mc6 floating", rec["floating_rod"]["ame"]["f_Hz"][:2], rec["floating_rod"]["shift_invert_low"]["f_Hz"][:3],
          flush=True)
    return rec


# ----------------------------------------------------------------------------------------------- form factor, Q0
def mc7(wd: Path):
    rec = {"case": "form factor (uniform axial B0 = fullwave.source), G = Q0 R_s (skin copper walls), overlaps",
           "pillbox": [], "box": [], "coax": [], "sleeve": [], "quarter": [], "overlaps": []}
    Cx = cv.cylinder_modes_3d(A, L, 1.2e9)[0]
    for h in [0.04, 0.02, 0.01] + ([0.0075] if FINE else []):
        r, dt = _solve(cv.cylinder_design_3d(A, L, h=h, modes=1, target_f_Hz=Cx["f_Hz"] * 0.97, wall_material=WALL),
                       wd / f"mc7_pill_{h}")
        m = _eig(r)["modes"][0]
        rec["pillbox"].append({"h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt, "f_rel": m["f_Hz"] / Cx["f_Hz"] - 1,
                               "C": m["form_factor"]["C"], "C_exact": Cx["C"], "C_rel": m["form_factor"]["C"] / Cx["C"] - 1,
                               "C_rms": m["form_factor"]["C_rms"], "G_ohm": m["losses"]["G_ohm"], "G_exact": Cx["G_ohm"],
                               "G_rel": m["losses"]["G_ohm"] / Cx["G_ohm"] - 1,
                               "G_pointwise_rel": m["losses"]["walls"]["outer"]["G_ohm_pointwise"] / Cx["G_ohm"] - 1,
                               "Q0": m["losses"]["Q0"], "Q0_exact": Cx["G_ohm"] / cv.skin_Rs(m["f_Hz"], RHO)})
        print("mc7 pillbox", rec["pillbox"][-1], flush=True)
    ab = (0.2, 0.15, 0.1)
    bx = cv.box_modes(*ab, 1.3e9)[0]
    for h in [0.03, 0.015] + ([0.0075] if FINE else []):
        r, dt = _solve(cv.box_design_3d(*ab, h=h, modes=1, target_f_Hz=bx["f_Hz"] * 0.97, wall_material=WALL),
                       wd / f"mc7_box_{h}")
        m = _eig(r)["modes"][0]
        rec["box"].append({"h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt, "f_rel": m["f_Hz"] / bx["f_Hz"] - 1,
                           "C": m["form_factor"]["C"], "C_exact": bx["C"], "C_rel": m["form_factor"]["C"] / bx["C"] - 1,
                           "G_ohm": m["losses"]["G_ohm"], "G_exact": bx["G_ohm"], "G_rel": m["losses"]["G_ohm"] / bx["G_ohm"] - 1,
                           "G_pointwise_rel": m["losses"]["walls"]["outer"]["G_ohm_pointwise"] / bx["G_ohm"] - 1})
        print("mc7 box", rec["box"][-1], flush=True)
    # coax: rod a = 0.05 on the axis spanning both caps, outer b = 0.2, L = 0.2; 3D vs analytic (f, C) and 2D (G)
    ca, cb, cL = 0.05, 0.2, 0.2
    cx = cv.coax_tm_mode(ca, cb)
    r2 = cv.coax_design(ca, cb, cL, h=0.0025, modes=2, wall_material=WALL, domain_radius=1.0).solve(wd / "mc7_coax2d", verbose=0)
    m2 = _near(r2.data["fullwave"]["eigenmodes"]["TM"]["modes"], cx["f_Hz"])
    ref2 = {"f_Hz": m2["f_Hz"], "C": m2["form_factor"]["C"], "G_ohm": m2["losses"]["G_ohm"],
            "walls": {k: v["G_ohm"] for k, v in m2["losses"]["walls"].items()}}
    rec["coax_2d_reference"] = {"h_m": 0.0025, **ref2, "f_exact_Hz": cx["f_Hz"], "C_exact": cx["C"]}
    for h in [0.02, 0.01] + ([0.0075] if FINE else []):
        r, dt = _solve(cv.cylinder_design_3d(cb, cL, h=h, modes=1, rod={"radius": ca}, wall_material=WALL,
                                             target_f_Hz=cx["f_Hz"] * 0.98), wd / f"mc7_coax_{h}")
        m = _eig(r)["modes"][0]
        rec["coax"].append({"h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt,
                            "f_rel": m["f_Hz"] / cx["f_Hz"] - 1, "C": m["form_factor"]["C"],
                            "C_rel": m["form_factor"]["C"] / cx["C"] - 1, "G_ohm": m["losses"]["G_ohm"],
                            "G_rel_vs_2d": m["losses"]["G_ohm"] / ref2["G_ohm"] - 1,
                            "walls": {k: v["G_ohm"] for k, v in m["losses"]["walls"].items()}})
        print("mc7 coax", rec["coax"][-1], flush=True)
    # dielectric sleeve (eps_r 4, tan d 1e-4, 0.03 < r < 0.05) in the pillbox: f, eps-weighted C, Q_d vs 1D Bessel
    # matching (sleeve_tm_mode) and the 2D solver (electric energy fraction)
    sl = cv.sleeve_tm_mode(A, 0.03, 0.05, 4.0)
    for h in [0.02, 0.01]:
        d = cv.cylinder_design_3d(A, L, h=h, modes=1, target_f_Hz=sl["f_Hz"] * 0.97, wall_material=WALL,
                                  sleeve={"r1": 0.03, "r2": 0.05,
                                          "material": {"kind": "linear", "eps_r": 4.0, "tan_delta": 1e-4}})
        r, dt = _solve(d, wd / f"mc7_sleeve_{h}")
        m = _eig(r)["modes"][0]
        dl = m["losses"].get("dielectric", {}).get("sleeve", {})
        rec["sleeve"].append({"h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt,
                              "f_rel": m["f_Hz"] / sl["f_Hz"] - 1, "C": m["form_factor"]["C"],
                              "C_rel": m["form_factor"]["C"] / sl["C"] - 1,
                              "electric_energy_fraction": dl.get("electric_energy_fraction"),
                              "Q_dielectric": m["losses"].get("Q_dielectric")})
        print("mc7 sleeve", rec["sleeve"][-1], flush=True)
    # quarter model with two PMC planes (contains TM010): image-corrected C and G
    for h in (0.02, 0.01):
        r, dt = _solve(cv.cylinder_design_3d(A, L, h=h, modes=1, symmetry={"x": "pmc", "y": "pmc"}, wall_material=WALL,
                                             target_f_Hz=Cx["f_Hz"] * 0.97), wd / f"mc7_quarter_{h}")
        m = _eig(r)["modes"][0]
        rec["quarter"].append({"h_m": h, "symmetry": {"x": "pmc", "y": "pmc"}, "f_rel": m["f_Hz"] / Cx["f_Hz"] - 1,
                               "C_rel": m["form_factor"]["C"] / Cx["C"] - 1, "G_rel": m["losses"]["G_ohm"] / Cx["G_ohm"] - 1,
                               "walls": list(m["losses"]["walls"])})
        print("mc7 quarter", rec["quarter"][-1], flush=True)
    # overlaps: window 1.15 ... 1.98 GHz (TM010 ... TE011/TM111 triple)
    eta_x = cv.eta_tm010_te01p(A, L, 1)
    for h in [0.02, 0.01]:
        r, dt = _solve(cv.cylinder_design_3d(A, L, h=h, modes=14, target_f_Hz=1.58e9, mode_overlaps=True),
                       wd / f"mc7_eta_{h}")
        modes = _eig(r)["modes"]
        eta = _eig(r)["overlaps"]["eta"]
        f = [m["f_Hz"] for m in modes]
        ref = cv.cylinder_modes_3d(A, L, 2.2e9)
        names = [x["name"] for _, x in _match_sorted(modes, ref)]
        i0 = names.index("TM010")
        tri = [j for j, x in enumerate(f) if abs(x / ref[[y["name"] for y in ref].index("TE011")]["f_Hz"] - 1) < 2e-3]
        eta_te011 = math.sqrt(sum(eta[i0][j] ** 2 for j in tri))
        m011 = names.index("TM011")
        # symmetry-forbidden pairs (continuum zero): TM010 (m = 0, E poloidal) with every m != 0 mode and TM011
        forb = [j for j, n in enumerate(names) if j != i0 and not n.startswith("TE01") and j not in tri]
        rec["overlaps"].append({"h_m": h, "dofs": r.data["mesh"]["dofs"], "seconds": dt, "names": names,
                                "eta_TM010_TE011_cluster": eta_te011, "eta_exact": eta_x,
                                "eta_rel": eta_te011 / eta_x - 1, "cluster_size": len(tri),
                                "eta_TM010_TM011": eta[i0][m011],
                                "max_eta_TM010_symmetry_forbidden": max(max(eta[i0][j], eta[j][i0]) for j in forb),
                                "eta_TM010_self": eta[i0][i0], "eta": eta})
        print("mc7 eta", {k: v for k, v in rec["overlaps"][-1].items() if k != "eta"}, flush=True)
    return rec


# ----------------------------------------------------------------------------------------------------- floating rod
def rod(wd: Path):
    """On-axis floating PEC rod (r = 0.015, gaps 0.02 at both caps) in the a = 0.1, L = 0.2 pillbox, TM010-like mode:
    3D (PEC hole) vs the axisymmetric solver, refined globally and at the rod (gap regions + rod surface)."""
    ra, g = 0.015, 0.02
    rec = {"case": f"floating rod r = {ra} m, gaps {g} m, pillbox a = {A}, L = {L}; skin copper", "ref2d": [],
           "study": []}
    for h2 in (0.005, 0.0025, 0.00125):
        r2, dt = _solve(cv.floating_rod_design(A, L, ra, g, h=h2, wall_material=WALL), wd / f"rod2d_{h2}", ranks=1)
        m = min(r2.data["fullwave"]["eigenmodes"]["TM"]["modes"], key=lambda y: y["f_Hz"])
        rec["ref2d"].append({"h_m": h2, "rod_size_m": h2 / 4, "seconds": dt, "dofs": r2.data["mesh"].get("dofs"),
                             "f_Hz": m["f_Hz"], "C": m["form_factor"]["C"], "G_ohm": m["losses"]["G_ohm"],
                             "walls": {b: w["G_ohm"] for b, w in m["losses"]["walls"].items()},
                             "walls_pointwise": {b: w.get("G_ohm_pointwise") for b, w in m["losses"]["walls"].items()}})
        print("rod 2d", rec["ref2d"][-1], flush=True)
    ref = rec["ref2d"][-1]
    cases = [(0.02, None), (0.014, None), (0.02, 0.005), (0.014, 0.0035), (0.01, 0.0025)] + \
        ([(0.007, 0.00175)] if FINE else [])
    for h, gs in cases:
        rd = {"radius": ra, "gap": g}
        if gs:
            rd["gap_size"] = gs
        r, dt = _solve(cv.cylinder_design_3d(A, L, h=h, modes=1, rod=rd, target_f_Hz=0.6e9, wall_material=WALL),
                       wd / f"rod3d_{h}_{gs}")
        m = _eig(r)["modes"][0]
        w = m["losses"]["walls"]
        rec["study"].append({"h_m": h, "gap_size_m": gs, "dofs": r.data["mesh"]["dofs"], "seconds": dt,
                             "f_rel": m["f_Hz"] / ref["f_Hz"] - 1, "C_rel": m["form_factor"]["C"] / ref["C"] - 1,
                             "G_rel": m["losses"]["G_ohm"] / ref["G_ohm"] - 1,
                             "G_rod_rel": w["rod"]["G_ohm"] / ref["walls"]["rod"] - 1,
                             "G_outer_rel": w["outer"]["G_ohm"] / ref["walls"]["wall"] - 1,
                             "rod_disagreement": w["rod"]["G_flux_pointwise_disagreement"],
                             "outer_disagreement": w["outer"]["G_flux_pointwise_disagreement"],
                             "resolution_warning": any("disagree" in x for x in r.data.get("warnings", []))})
        print("rod 3d", rec["study"][-1], flush=True)
    # off-centre floating rod: PEC hole vs retained conductor volume (both consistent flux), and its indicator
    off = {"radius": ra, "gap": g, "x": 0.045, "y": 0.0}
    rec["offcentre"] = []
    for tag, rd in (("hole", off), ("retained", {**off, "hole": False}), ("hole_refined", {**off, "gap_size": 0.004})):
        r, dt = _solve(cv.cylinder_design_3d(A, L, h=0.02, modes=3, rod=rd, target_f_Hz=0.9e9, wall_material=WALL),
                       wd / f"rodoff_{tag}")
        rec["offcentre"].append({"model": tag, "dofs": r.data["mesh"]["dofs"], "seconds": dt,
                                 "modes": [{"f_Hz": m["f_Hz"], "C": m["form_factor"].get("C"), "G_ohm": m["losses"]["G_ohm"],
                                            "rod_G": m["losses"]["walls"]["rod"]["G_ohm"],
                                            "rod_G_pointwise": m["losses"]["walls"]["rod"]["G_ohm_pointwise"],
                                            "rod_disagreement": m["losses"]["walls"]["rod"]["G_flux_pointwise_disagreement"]}
                                           for m in sorted(_eig(r)["modes"], key=lambda y: y["f_Hz"])]})
        print("rod off", rec["offcentre"][-1], flush=True)
    return rec


# ----------------------------------------------------------------------------------------------------- report
def _fmt(x, f="{:.2e}"):
    return "-" if x is None else f.format(x)


def report():
    R = {k: json.loads((RECS / v).read_text()) for k, v in FILES.items() if (RECS / v).exists()}
    L_ = ["# Cavity haloscopes in 3D — validation", "",
          "**Beta.** Generated by",
          "`benchmarks/cavity_3d_validation.py` from `benchmarks/records/cavity_mc{5,6,7}_*.json`. Closed PEC",
          "cavities (`mesh.domain_solid` wall, `fullwave.rf_only`), ND order 2, 16 MPI ranks unless stated;",
          "`h` = `mesh.size` (= `surface_size`). References: the closed forms of `axionfem.cavity` (derived independently of",
          "the solver) and the 2D solver; tests `tests/test_cavity3d.py`.", ""]
    if "mc5" in R:
        r = R["mc5"]
        L_ += ["## 1. Lowest modes (AME, no pickup)", "", f"{r['case']}. Relative frequency error vs the analytic "
               "TM/TE_mnp list (degenerate members counted; sorted pairing).", ""]
        names = [m["name"] for m in r["pillbox"][0]["modes"]]
        L_ += ["| h [m] | DOFs | " + " | ".join(names) + " | gate | s |", "|" + "---|" * (len(names) + 4)]
        for row in r["pillbox"]:
            L_.append(f"| {row['h_m']} | {row['dofs']} | " + " | ".join(_fmt(m["rel"], "{:+.1e}") for m in row["modes"])
                      + f" | {row['diag']['converged']} | {row['seconds']:.0f} |")
        L_ += ["", "Note: for L = 2a the lowest modes are TM010, TE111 x2, TM011, TE211 x2, TE112 x2, TM110 x2 (TM110 is",
               "the 9th/10th mode, not among the lowest six as the survey's acceptance line assumed).", "",
               "Rectangular box 0.2 x 0.15 x 0.1 m:", "", "| h [m] | DOFs | mode | rel |", "|---|---|---|---|"]
        for row in r["box"]:
            for m in row["modes"][:2]:
                L_.append(f"| {row['h_m']} | {row['dofs']} | {m['name']} | {m['rel']:+.2e} |")
        L_ += ["", "Quarter models (mirror planes x = 0, y = 0 through the axis): returned modes vs the analytic subset of",
               "that parity.", "", "| planes (x, y) | h | modes (rel) |", "|---|---|---|"]
        for row in r["quarter"]:
            L_.append(f"| {row['symmetry']['x']}, {row['symmetry']['y']} | {row['h_m']} | " +
                      ", ".join(f"{m['name']} {m['rel']:+.1e}" for m in row["modes"]) + " |")
        L_.append("")
    if "mc6" in R:
        r = R["mc6"]
        L_ += ["## 2. Targeted shift-invert (STRUMPACK)", "", r["case"] + ".", ""]
        for w in r["window"]:
            L_ += [f"Target {w['target_f_Hz']/1e9:.3f} GHz, {w['modes_requested']} modes, h = {w['h_m']} ({w['dofs']} DOFs):"
                   f" {w['analytic_in_window']} analytic modes in the returned window; Lanczos steps "
                   f"{w['solver']['lanczos_steps']}, max true residual {w['diag']['max_relative_residual']:.1e}.", "",
                   "| mode | f exact [GHz] | rel | residual | div. fraction |", "|---|---|---|---|---|"]
            for m in w["modes"]:
                L_.append(f"| {m['name']} | {m['f_exact_Hz']/1e9:.6f} | {m['rel']:+.2e} | {m['residual_true']:.1e} | "
                          f"{m['divergence_fraction']:.1e} |")
            if "vs_ame" in w:
                L_.append("")
                L_.append(f"Same mesh, AME (12 lowest modes): {len(w['vs_ame'])} common modes, max |f_SI/f_AME - 1| = "
                          f"{max(abs(x['rel_vs_ame']) for x in w['vs_ame']):.1e}.")
            L_.append("")
        L_ += ["MPI ranks (h = 0.02, same target):", "", "| ranks | total s | factorization s | Lanczos s | max rel diff vs 1 rank |",
               "|---|---|---|---|---|"]
        for x in r["ranks"]:
            L_.append(f"| {x['ranks']} | {x['seconds']:.0f} | {x['factorization_seconds']:.1f} | {x['lanczos_seconds']:.1f} | "
                      f"{x['max_rel_diff_vs_1_rank']:.1e} |")
        e = r["exact_target"]
        L_ += ["", f"Target = the TM010 eigenfrequency of the same mesh ({e['target_f_Hz']:.9e} Hz, copied from a shift-invert run): "
               f"{e['returned']}/{e['requested']} modes, converged {e['diag']['converged']}, restarts "
               f"{e['solver'].get('restarts_after_invariant_subspace')}, nearest |f/f_t - 1| = {e['nearest_rel']:.1e}, "
               f"max residual {e['max_residual_true']:.1e}.", ""]
        fr = r["floating_rod"]
        L_ += [f"Floating PEC rod (radius {fr['rod']['radius']} m, gaps {fr['rod']['gap']} m at both caps; PEC hole): AME "
               f"returns f = {fr['ame']['f_Hz'][0]:.3g} Hz (lambda ~ 0, residual {fr['ame']['residual_true'][0]:.2f}), "
               f"gate converged = {fr['ame']['diag']['converged']}. Shift-invert at {fr['shift_invert']['target_f_Hz']/1e9} GHz: "
               f"modes {', '.join(f'{x/1e9:.4f}' for x in fr['shift_invert']['f_Hz'])} GHz, max residual "
               f"{max(fr['shift_invert']['residual_true']):.1e}, kernel Ritz values discarded "
               f"{fr['shift_invert']['solver']['kernel_ritz_values_discarded']}; at {fr['shift_invert_low']['target_f_Hz']/1e9} GHz "
               f"(lowest modes): {', '.join(f'{x/1e9:.4f}' for x in fr['shift_invert_low']['f_Hz'][:4])} ... GHz, discarded "
               f"{fr['shift_invert_low']['solver']['kernel_ritz_values_discarded']}.", ""]
    if "mc7" in R:
        r = R["mc7"]
        L_ += ["## 3. Form factor, Q0, overlaps", "", r["case"] + ".", "",
               "Pillbox TM010 (C = 4/chi01^2 = 0.6916602761, G = Z0 chi01 L/(2(a + L)) = 301.9902289 ohm):", "",
               "| h [m] | DOFs | f rel | C rel | G rel (flux) | G rel (pointwise) |", "|---|---|---|---|---|---|"]
        for x in r["pillbox"]:
            L_.append(f"| {x['h_m']} | {x['dofs']} | {x['f_rel']:+.1e} | {x['C_rel']:+.1e} | {x['G_rel']:+.1e} | {x['G_pointwise_rel']:+.1e} |")
        L_ += ["", "Rectangular box TM110 (C = 64/pi^4 = 0.6570228643; G from the closed-form Q0):", "",
               "| h [m] | DOFs | f rel | C rel | G rel (flux) | G rel (pointwise) |", "|---|---|---|---|---|---|"]
        for x in r["box"]:
            L_.append(f"| {x['h_m']} | {x['dofs']} | {x['f_rel']:+.1e} | {x['C_rel']:+.1e} | {x['G_rel']:+.1e} | {x['G_pointwise_rel']:+.1e} |")
        c2 = r.get("coax_2d_reference", {})
        L_ += ["", f"Coaxial cavity (rod a = 0.05 spanning both caps, b = 0.2, L = 0.2; kb = 4.0976855393, C = 0.7949373125;"
               f" G from the 2D solver at h = 2.5 mm: {c2.get('G_ohm', float('nan')):.6f} ohm):", "",
               "| h [m] | DOFs | f rel | C rel | G rel vs 2D |", "|---|---|---|---|---|"]
        for x in r["coax"]:
            L_.append(f"| {x['h_m']} | {x['dofs']} | {x['f_rel']:+.1e} | {x['C_rel']:+.1e} | {x['G_rel_vs_2d']:+.1e} |")
        if r.get("sleeve"):
            L_ += ["", "Dielectric sleeve (eps_r 4, tan d 1e-4, 0.03 < r < 0.05 m) in the pillbox vs 1D Bessel matching:", "",
                   "| h [m] | f rel | C rel (eps-weighted) | W_e fraction in sleeve | Q_d |", "|---|---|---|---|---|"]
            for x in r["sleeve"]:
                L_.append(f"| {x['h_m']} | {x['f_rel']:+.1e} | {x['C_rel']:+.1e} | {_fmt(x['electric_energy_fraction'], '{:.7f}')} | "
                          f"{_fmt(x['Q_dielectric'], '{:.1f}')} |")
            L_.append("")
            L_.append("Reference W_e fraction (2D solver / 1D Bessel): 0.7156375; "
                      "Q_d = 1/(tan d * fraction) = 13973.4.")
        L_ += ["", "Quarter model (PMC planes x = 0, y = 0; C with the mirror-image source sum, symmetry planes lossless):", "",
               "| h [m] | f rel | C rel | G rel | wall bodies |", "|---|---|---|---|---|"]
        for x in r["quarter"]:
            L_.append(f"| {x['h_m']} | {x['f_rel']:+.1e} | {x['C_rel']:+.1e} | {x['G_rel']:+.1e} | {', '.join(x['walls'])} |")
        L_ += ["", "Mode overlaps eta_ij = |int E_i.B_j| / sqrt(int|E_i|^2 int|B_j|^2) (14 modes around 1.58 GHz):", "",
               "| h [m] | eta(TM010, TE011) | exact | rel | eta(TM010, TM011) | max eta TM010 x symmetry-forbidden |",
               "|---|---|---|---|---|---|"]
        for x in r["overlaps"]:
            L_.append(f"| {x['h_m']} | {x['eta_TM010_TE011_cluster']:.6f} | {x['eta_exact']:.6f} | {x['eta_rel']:+.1e} | "
                      f"{x['eta_TM010_TM011']:.1e} | {x['max_eta_TM010_symmetry_forbidden']:.1e} |")
        L_ += ["", "TE011 is exactly degenerate with the TM111 pair (chi'01 = chi11); the mesh mixes the three, so",
               "eta(TM010, TE011) is the root sum of squares over that cluster (TM111's B is transverse, E_TM010 is axial).",
               "Symmetry-forbidden pairs (TM010 with m != 0 modes) vanish only to discretization accuracy on an unstructured",
               "mesh (the mesh is not mirror-symmetric); they decrease under refinement but do not reach 1e-10. Under",
               "mesh.symmetry the full-cavity eta of two modes of one model is identically zero (E.B is a pseudoscalar).", ""]
    if "rod" in R:
        r = R["rod"]
        L_ += ["## Floating tuning rod (3D vs axisymmetric reference)", "", r["case"] + ". Lowest (TM010-like) mode.", "",
               "Reference: the 2D solver (`cavity.floating_rod_design`), cavity size h, rod surface h/4:", "",
               "| h [m] | DOFs | f [GHz] | C | G [ohm] | G rod | G wall |", "|---|---|---|---|---|---|---|"]
        for x in r["ref2d"]:
            L_.append(f"| {x['h_m']} | {x['dofs']} | {x['f_Hz']/1e9:.7f} | {x['C']:.6f} | {x['G_ohm']:.4f} | "
                      f"{x['walls']['rod']:.3f} | {x['walls']['wall']:.3f} |")
        L_ += ["", "3D (PEC hole), relative to the finest 2D row; `gap` = `rod.gap_size` (element size in the two gap "
               "regions around the rod tips, also the rod's surface_size; `-`: none, the rod surface follows h); "
               "`disagr.` = |G_flux/G_pointwise - 1| of the rod (the output's resolution indicator, warned above 0.05):",
               "", "| h [m] | gap [m] | DOFs | f rel | C rel | G rel | G rod rel | G wall rel | disagr. rod | warned |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for x in r["study"]:
            L_.append(f"| {x['h_m']} | {x['gap_size_m'] or '-'} | {x['dofs']} | {x['f_rel']:+.1e} | {x['C_rel']:+.1e} | "
                      f"{x['G_rel']:+.1e} | {x['G_rod_rel']:+.1e} | {x['G_outer_rel']:+.1e} | {x['rod_disagreement']:.3f} | "
                      f"{'yes' if x['resolution_warning'] else 'no'} |")
        ref_rows = [x for x in r["study"] if x["gap_size_m"]]
        if len(ref_rows) >= 3:  # Richardson: e(s) = e_inf + c s^p from the three finest locally refined rows
            ref_rows = sorted(ref_rows, key=lambda x: -x["gap_size_m"])[-3:]
            s1, s2, s3 = (x["gap_size_m"] for x in ref_rows)
            ext = []
            for key in ("f_rel", "C_rel", "G_rel", "G_rod_rel", "G_outer_rel"):
                e1, e2, e3 = (x[key] for x in ref_rows)
                try:
                    pw = math.log((e1 - e2) / (e2 - e3)) / math.log(s1 / s2) if s1 / s2 == s2 / s3 else \
                        math.log((e1 - e2) / (e2 - e3)) / math.log(math.sqrt(s1 / s3))
                    einf = e3 - (e2 - e3) / ((s2 / s3) ** pw - 1)
                    ext.append(f"{key.replace('_rel', '')} {einf:+.1e} (order {pw:.2f})")
                except (ValueError, ZeroDivisionError):
                    ext.append(f"{key.replace('_rel', '')} n/a")
            L_ += ["", "Extrapolated to h -> 0 (Richardson from the three finest refined rows; residual difference from the "
                   "2D reference): " + ", ".join(ext) + ". The order ~4/3 is that of the re-entrant rod-tip edge."]
        L_ += ["", "Off-centre floating rod (x = 0.045 m), h = 0.02: PEC hole, retained conductor volume (both with the "
               "consistent-flux current), and the hole with gap_size 0.004:", "",
               "| model | DOFs | f [GHz] | G [ohm] | rod G flux / pointwise | rod disagr. |", "|---|---|---|---|---|---|"]
        for x in r["offcentre"]:
            for m in x["modes"]:
                L_.append(f"| {x['model']} | {x['dofs']} | {m['f_Hz']/1e9:.5f} | {m['G_ohm']:.3f} | {m['rod_G']:.2f} / "
                          f"{m['rod_G_pointwise']:.2f} | {m['rod_disagreement']:.3f} |")
        L_ += ["", "The rod tips (a 270-degree edge) and the gap capacitance converge slowly with a uniform mesh: at "
               "h = 0.02 C is a few % low. Refine the gaps (`rod.gap_size`, or regions with mesh_size around them) "
               "until the rod's flux/pointwise disagreement is small; the consistent-flux G converges from below, "
               "the pointwise one from above.", ""]
    OUT.write_text("\n".join(L_) + "\n")
    print("wrote", OUT)


def main():
    global RANKS, FINE
    args = sys.argv[1:]
    only = ["mc5", "mc6", "mc7", "rod"]
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1].split("+")
        del args[i:i + 2]
    if "--ranks" in args:
        i = args.index("--ranks")
        RANKS = int(args[i + 1])
        del args[i:i + 2]
    if "--fine" in args:
        FINE = True
        args.remove("--fine")
    wd = Path(args[0]) if args else Path("runs/cavity3d")
    RECS.mkdir(parents=True, exist_ok=True)
    for k in only:
        t0 = time.time()
        rec = {"mc5": mc5, "mc6": mc6, "mc7": mc7, "rod": rod}[k](wd)
        rec["status"] = "beta"
        rec["mpi_ranks"] = RANKS
        rec["total_seconds"] = time.time() - t0
        (RECS / FILES[k]).write_text(json.dumps(rec, indent=1) + "\n")
        print(k, "done", rec["total_seconds"], flush=True)
    report()


if __name__ == "__main__":
    main()
