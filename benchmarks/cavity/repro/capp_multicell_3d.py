"""3D: CAPP multi-cell cavities, ideal sector cells (beta).

    python benchmarks/cavity/repro/capp_multicell_3d.py

References (arXiv PDFs): 1710.06969 (Jeong et al. 2018) Table 2 p. 3 (Q of the N = 2, 4, 6, 8 cell cavity, Rc = 25 mm,
copper sigma = 5.96e7 S/m at room temperature, "no end caps"; analytical 17794 / 16151 / 14320 / 12907 and two
COMSOL columns), Table 3 p. 3 (omega/omega0, Q/Q0, P_rad/P_azi), Eq. 2 (C); 2205.01319 Table 2 p. 7 (closed-form
C = 0.64 / 0.65 / 0.63 / 0.61 for N = 2, 4, 6, 8, normalized f and Q). refs: benchmarks/cavity/refs/
capp_multicell_1710.json (its closed form, digitize/capp_sector_analytic.py, reproduces Table 3 and finds that the
analytical Table 2 column corresponds to sigma = 5.998e7 S/m, COMSOL's built-in copper, not the stated 5.96e7).

Model: the N cells are separated by zero-thickness conducting partitions that meet on the axis, so the cells are
decoupled and the in-phase N-cell mode has the f, Q and C of one cell. One cell is meshed in 3D as a sector of the
cylinder (`revolve` of r < Rc over 0 < phi < 360/N deg); the partitions are part of its lossy wall. "No end caps":
the TM_(nu,1,0) sector mode (E_z = J_nu(chi r/Rc) sin(nu phi), nu = N/2) is z-independent, so it is an exact mode of
the cell closed by flat end plates at any length; the plates are separate conductor bodies (PEC holes `cap_lo`,
`cap_hi`, same copper), and Q_no_caps = G_outer / R_s uses the barrel + partitions body only (Table 2); Q with caps
(L = Rc) is reported as well. sigma = 5.96e7 S/m (stated) and 5.998e7 (implied by the analytical column).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "capp_multicell_3d"
RC = 0.025
L = 0.025
TCAP = 0.004
SIG = 5.96e7
SIG_COMSOL = 5.998e7


def sector_design(tag, N, *, h, order=2, sigma=SIG):
    ang = [0.0, 360.0 / N]
    z0, z1 = -L / 2 - TCAP, L / 2 + TCAP

    def sec(za, zb):
        # center/axis/fillet spelled out: mesh.domain_solid reaches the mesher without the defaults that region
        # solids get (axf-mesher reads b["center"] for a revolve)
        return {"revolve": {"profile": {"rect": {"r": [0.0, RC], "z": [za, zb], "fillet": 0.0}}, "angles": ang,
                            "center": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]}}
    dom = sec(z0, z1)
    caps = [{"name": "cap_lo", "solid": sec(z0, -L / 2)}, {"name": "cap_hi", "solid": sec(L / 2, z1)}]
    phi = math.radians(180.0 / N)
    ip = (0.55 * RC * math.cos(phi), 0.55 * RC * math.sin(phi), 0.13 * L)
    ref = json.loads((cm.REFS / "capp_multicell_1710.json").read_text())
    rows = {r["N_cells"]: r for r in ref["reference_values"]["analytic_check"]["rows"]}
    d = cm.design3d(tag, dom, h=h, order=order, modes=4, target_f_Hz=rows[N]["f_GHz"] * 1e9,
                    wall=cm.rf_skin(sigma, 293.0), B_T=(0.0, 0.0, 1.0), conductors=caps, interior_point=ip,
                    remove_pec=True)
    return d


def closed_form(N, sigma):
    """TM_(nu,1,0) sector mode, nu = N/2 (N = 0: the full TM010)."""
    from scipy.special import jn_zeros, jv, jvp
    from scipy.integrate import quad
    nu = N / 2
    if N == 0:
        chi = jn_zeros(0, 1)[0]
    else:
        from scipy.optimize import brentq
        # first zero of J_nu
        chi = jn_zeros(int(nu), 1)[0] if float(nu).is_integer() else brentq(lambda x: jv(nu, x), nu + 0.5, nu + 4)
    f = cm.C_LIGHT * chi / (2 * math.pi * RC)
    w = 2 * math.pi * f
    k = w / cm.C_LIGHT
    Rs = cm.Rs_skin(f, sigma)
    th = 2 * math.pi / N if N else 2 * math.pi
    # E_z = J_nu(k r) sin(nu phi) on 0<phi<th (sin -> 1 for N=0)
    ang2 = th / 2 if N else th  # int sin^2
    ang1 = (2 / nu) if N else th  # int_0^th sin(nu phi) dphi = 2/nu for th = pi/nu
    Iez2 = ang2 * quad(lambda r: jv(nu, k * r) ** 2 * r, 0, RC, epsabs=1e-14)[0]
    Iez = ang1 * quad(lambda r: jv(nu, k * r) * r, 0, RC, epsabs=1e-14)[0]
    A = th * RC ** 2 / 2
    C = Iez ** 2 / (A * Iez2)
    # H_t on the barrel: H_phi = (1/(i w mu)) dE_z/dr; on the partitions: H_r = (1/(i w mu)) (1/r) dE_z/dphi
    # Q = w U / P, U = (eps/2) int |E|^2 dV (per unit length), P = (Rs/2) oint |H_t|^2 dl
    mu = cm.MU0
    eps = 1 / (mu * cm.C_LIGHT ** 2)
    Pbar = ang2 * RC * (k * jvp(nu, k * RC)) ** 2
    Ppart = 0.0
    if N:
        Ppart = 2 * quad(lambda r: (nu * jv(nu, k * r) / r) ** 2, 0, RC,
                         epsabs=1e-14, limit=200)[0]
    Pint = (Pbar + Ppart) / (w * mu) ** 2
    U = 0.5 * eps * Iez2
    Q = w * U / (0.5 * Rs * Pint)
    return {"N": N, "nu": nu, "chi": chi, "f_Hz": f, "C": C, "Q_no_caps": Q, "Rs_ohm": Rs,
            "P_rad_over_P_azi": Ppart / Pbar if N else 0.0}


def main():
    ref = json.loads((cm.REFS / "capp_multicell_1710.json").read_text())
    tab2 = {r[0]: r for r in ref["reference_values"]["tables"][1]["rows"]}
    jobs = []
    for N in (2, 4, 6, 8):
        for div in (6, 9, 12):
            jobs.append((f"N{N}_o2_h{div}", sector_design(f"N{N}_o2_h{div}", N, h=RC / div), dict(N=N, order=2, div=div)))
        jobs.append((f"N{N}_o3_h9", sector_design(f"N{N}_o3_h9", N, h=RC / 9, order=3), dict(N=N, order=3, div=9)))
        jobs.append((f"N{N}_o3_h9_s5998", sector_design(f"N{N}_o3_h9_s5998", N, h=RC / 9, order=3, sigma=SIG_COMSOL),
                     dict(N=N, order=3, div=9, sigma=SIG_COMSOL)))
    out = cm.solve_many3d([(t, d) for t, d, _ in jobs], NAME)
    rec = {"benchmark": "CAPP multi-cell (1710.06969 Tables 2-3; 2205.01319 Table 2): ideal sector cells in 3D",
           "model": __doc__.split("Model: ")[1].strip(), "cases": []}
    for N in (2, 4, 6, 8):
        cf = closed_form(N, SIG)
        cf2 = closed_form(N, SIG_COMSOL)
        runs = []
        for (tag, data, dt), (_, _, meta) in zip(out, jobs):
            if meta["N"] != N:
                continue
            ms = cm.modes3d(data)
            m = min(ms, key=lambda x: abs(x["f_Hz"] - cf["f_Hz"]))
            lo = m["losses"]
            walls = lo["walls"]
            Rs = walls["outer"]["Rs_ohm"]
            Gout = walls["outer"]["G_ohm"]
            runs.append({"tag": tag, **meta, "dofs": cm.dofs(data), "seconds": dt, "f_Hz": m["f_Hz"],
                         "C": m["form_factor"]["C"], "Q_no_caps": Gout / Rs, "Q_with_caps": lo.get("Q0"),
                         "G_outer_ohm": Gout, "G_outer_pointwise_disagreement": walls["outer"].get(
                             "G_flux_pointwise_disagreement"),
                         "walls": {k: {kk: v.get(kk) for kk in ("G_ohm", "Q", "P_frac")} for k, v in walls.items()},
                         "residual_true": m.get("residual_true"),
                         "other_modes_f_Hz": sorted(x["f_Hz"] for x in ms if x is not m)})
        best = [r for r in runs if r["order"] == 3 and r.get("sigma") is None][0]
        best2 = [r for r in runs if r.get("sigma") == SIG_COMSOL][0]
        t2 = tab2[N]
        num = float(str(t2[3]).split()[0].replace(",", ""))
        numax = float(str(t2[4]).split()[0].replace(",", ""))
        rec["cases"].append({
            "N": N, "closed_form_sigma_5.96e7": cf, "closed_form_sigma_5.998e7": cf2, "runs": runs,
            "converged": {"f_Hz": best["f_Hz"], "C": best["C"], "Q_no_caps": best["Q_no_caps"],
                          "rel_f": best["f_Hz"] / cf["f_Hz"] - 1, "rel_C": best["C"] / cf["C"] - 1,
                          "rel_Q": best["Q_no_caps"] / cf["Q_no_caps"] - 1,
                          "Q_no_caps_sigma_5.998e7": best2["Q_no_caps"]},
            "paper": {"Q_analytical": t2[2], "Q_numerical": num, "Q_numerical_ax": numax,
                      "analytical_minus_axionFEM_5.96e7_rel": t2[2] / best["Q_no_caps"] - 1,
                      "analytical_minus_axionFEM_5.998e7_rel": t2[2] / best2["Q_no_caps"] - 1,
                      "numerical_minus_axionFEM_5.96e7_rel": num / best["Q_no_caps"] - 1,
                      "numerical_ax_minus_axionFEM_5.96e7_rel": numax / best["Q_no_caps"] - 1}})
    # Table 3 ratios relative to the empty cylinder (N = 0) closed form
    c0 = closed_form(0, SIG)
    for c in rec["cases"]:
        c["ratios_vs_N0"] = {"w_over_w0": c["converged"]["f_Hz"] / c0["f_Hz"],
                             "Q_over_Q0": c["converged"]["Q_no_caps"] / c0["Q_no_caps"]}
    rec["closed_form_N0"] = c0
    p = cm.write_record("capp_multicell_3d", rec)
    print(p)
    for c in rec["cases"]:
        print(c["N"], json.dumps(c["converged"]), json.dumps(c["paper"]))


if __name__ == "__main__":
    main()
