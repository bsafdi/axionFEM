"""3D pickup surface losses (pickup.losses) against the 2D engine and the closed form.

    python benchmarks/rf_losses_3d.py [--ranks N]
    python benchmarks/rf_losses_3d.py hstudy [--ranks N]    (uniform h-study with Richardson extrapolation)

Solenoid template, copper coax with the builtin rf skin block (rho 1.7e-10 Ohm m, T 0.03 K); R = 1, H = 2,
x = 0.534, tw = 0.05, g = tc = 0.03, tp = 0.05. The pickup field in the coax is H = I / (2 pi r), so per unit current
    Gamma_inner = (H - tp - tc + tp / 3) / (2 pi x R)
    Gamma_outer = (H - tp - tc + tp / 3) / (2 pi (R - g - tc))
    Gamma_short = ln((R - g - tc) / (x R)) / (2 pi)
(tp / 3: under the uniform-feed port the wall current falls linearly to zero across the port height).
The 2D engine (order 2) and 3D at order 1 and order 2 on two meshes are compared per surface: Gamma, Q at the band
edges, and T/Q total. The formula chain R_eff = R_s(f) Gamma, Q = 2 pi f L_p / R_eff is checked against the copper
skin R_s = sqrt(pi f mu0 rho), with each band edge recovered as f = Q R_eff / (2 pi L_p).
Scope: a smooth 1/r surface field with no singular edge. The edge- and port-dominated case (rounded slit lips, partial
ports) is the slotted box of benchmarks/fullwave_surface_losses.py (the `abra` cases).
The 3D runs use the axf CLI (AXF_EXE) on N MPI ranks, the 2D run the in-process module.
Writes benchmarks/records/rf_losses_3d.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "python"))

import axionfem as af  # noqa: E402
from axionfem.convert import to_3d  # noqa: E402

RUNS = ROOT / "runs" / "rf_losses_3d"
OUT = ROOT / "benchmarks" / "records" / "rf_losses_3d.json"
MU0 = 4e-7 * math.pi
RHO_CU = 1.7e-10   # builtin copper rf block
T_CU = 0.03
PARAMS = {"R": 1.0, "H": 2.0, "x": 0.534, "tw": 0.05, "g": 0.03, "tc": 0.03, "tp": 0.05}
MESH2D = {"order": 2, "size": 0.05, "surface_size": 0.015}
MESH3D = {"o1": {"order": 1, "size": 0.4, "surface_size": 0.13},
          "o2": {"order": 2, "size": 0.4, "surface_size": 0.13},
          "o2_fine": {"order": 2, "size": 0.25, "surface_size": 0.07}}
H_FACTORS = [1.0, 0.75, 0.55]   # hstudy: every mesh size of the "o2" run times these


def closed_form() -> dict:
    p = PARAMS
    length = p["H"] - p["tp"] - p["tc"] + p["tp"] / 3
    ri = p["x"] * p["R"]
    ro = p["R"] - p["g"] - p["tc"]
    return {"coax_inner": length / (2 * math.pi * ri), "coax_outer": length / (2 * math.pi * ro),
            "coax_short": math.log(ro / ri) / (2 * math.pi)}


def chain_error(losses: dict, Lp: float) -> float:
    """Largest relative error of R_eff = R_s Gamma (copper skin R_s), Q = 2 pi f L_p / R_eff and T/Q = T / Q."""
    worst = 0.0
    tq = {"f_min": 0.0, "f_max": 0.0}
    for v in losses["surfaces"].values():
        for e in ("f_min", "f_max"):
            f = v["Q"][e] * v["R_eff_ohm"][e] / (2 * math.pi * Lp)
            Rs = math.sqrt(math.pi * f * MU0 * RHO_CU)
            worst = max(worst, abs(v["R_eff_ohm"][e] / (Rs * v["Gamma"]) - 1))
            worst = max(worst, abs(v["T_over_Q_K"][e] * v["Q"][e] / T_CU - 1))
            tq[e] += v["T_over_Q_K"][e]
    for e in tq:
        worst = max(worst, abs(losses["T_over_Q_total_K"][e] / tq[e] - 1))
    return worst


def band(losses: dict, Lp: float) -> dict:
    v = next(iter(losses["surfaces"].values()))
    return {e: v["Q"][e] * v["R_eff_ohm"][e] / (2 * math.pi * Lp) for e in ("f_min", "f_max")}


def main() -> None:
    a = sys.argv[1:]
    ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4"))
    RUNS.mkdir(parents=True, exist_ok=True)
    cf = closed_form()
    d2 = af.template("solenoid", **PARAMS, mesh=MESH2D)
    t0 = time.time()
    r2 = d2.solve(RUNS / "s2", verbose=0)
    l2 = r2.data["pickup"]["losses"]
    L2 = r2.coupling["L_p_H"]
    rec = {"check": "rf_losses_3d", "date": time.strftime("%Y-%m-%d"), "params": PARAMS, "closed_form_Gamma": cf,
           "2D": {"mesh": MESH2D, "L_p_H": L2, "seconds": time.time() - t0, "band_Hz": band(l2, L2),
                  "chain_max_rel_error": chain_error(l2, L2),
                  "Gamma": {k: v["Gamma"] for k, v in l2["surfaces"].items()},
                  "Gamma_over_closed_form": {k: l2["surfaces"][k]["Gamma"] / cf[k] for k in cf},
                  "T_over_Q_total_K": l2["T_over_Q_total_K"]},
           "3D": {}}
    print("2D/closed form:", {k: round(v, 7) for k, v in rec["2D"]["Gamma_over_closed_form"].items()}, flush=True)
    for tag, mesh in MESH3D.items():
        t0 = time.time()
        r3 = to_3d(d2, mesh=mesh).solve(RUNS / tag, verbose=0, mpi_ranks=ranks)
        l3 = r3.data["pickup"]["losses"]
        L3 = r3.coupling["L_p_H"]
        s3 = l3["surfaces"]
        row = {"mesh": mesh, "dofs": r3.data["mesh"]["dofs"], "mpi_ranks": ranks, "seconds": time.time() - t0,
               "L_p_over_2D": L3 / L2, "band_Hz": band(l3, L3), "chain_max_rel_error": chain_error(l3, L3),
               "Gamma": {k: v["Gamma"] for k, v in s3.items()},
               "Gamma_over_2D": {k: s3[k]["Gamma"] / l2["surfaces"][k]["Gamma"] for k in cf},
               "Gamma_over_closed_form": {k: s3[k]["Gamma"] / cf[k] for k in cf},
               "Q_f_min_over_2D": {k: s3[k]["Q"]["f_min"] / l2["surfaces"][k]["Q"]["f_min"] for k in cf},
               "T_over_Q_total_over_2D": {e: l3["T_over_Q_total_K"][e] / l2["T_over_Q_total_K"][e]
                                          for e in ("f_min", "f_max")},
               "Gamma_by_field_region": {k: v.get("Gamma_by_field_region") for k, v in s3.items()}}
        rec["3D"][tag] = row
        print(tag, "dofs", row["dofs"], "L_p/2D", f"{row['L_p_over_2D']:.6f}",
              "Gamma/2D", {k: round(v, 4) for k, v in row["Gamma_over_2D"].items()},
              "T/Q/2D", round(row["T_over_Q_total_over_2D"]["f_min"], 4), f"{row['seconds']:.0f} s", flush=True)
    c, f = rec["3D"]["o2"], rec["3D"]["o2_fine"]
    h = MESH3D["o2"]["surface_size"] / MESH3D["o2_fine"]["surface_size"]
    rec["observed_rate_o2"] = {k: math.log(abs(c["Gamma_over_closed_form"][k] - 1) /
                                           abs(f["Gamma_over_closed_form"][k] - 1)) / math.log(h) for k in cf}
    print("observed order-2 rate in surface_size:", {k: round(v, 2) for k, v in rec["observed_rate_o2"].items()})
    if OUT.exists() and "h_study" in json.loads(OUT.read_text()):   # kept from `hstudy`
        rec["h_study"] = json.loads(OUT.read_text())["h_study"]
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote", OUT)


def hstudy(ranks: int) -> None:
    """Uniform h-study of the order-2 3D run (axionfem.convergence.h_study, every mesh size times 1, 0.75, 0.55): the
    three-mesh Richardson extrapolation of each Gamma against the closed form and the finest mesh. Merged into the
    record as "h_study"."""
    from axionfem.convergence import h_study, richardson
    cf = closed_form()
    d3 = to_3d(af.template("solenoid", **PARAMS, mesh=MESH2D), mesh=MESH3D["o2"])
    t0 = time.time()
    out = h_study(d3, H_FACTORS, RUNS / "h_study", mpi_ranks=ranks)
    rows = out["rows"]
    summary = {}
    for k in cf:
        ex = out["extrapolation"][f"Gamma:{k}"]
        finest = min(rows, key=lambda row: row["factor"])["quantities"][f"Gamma:{k}"]
        conservative = richardson([row["factor"] for row in rows], [row["quantities"][f"Gamma:{k}"] for row in rows],
                                  formal_order=2)
        summary[k] = {"finest_over_closed_form": finest / cf[k],
                      "extrapolated_over_closed_form": ex["extrapolated"] / cf[k] if ex["extrapolated"] else None,
                      "observed_order": ex["order"], "gci_fine": ex["gci_fine"], "behaviour": ex["behaviour"],
                      "gci_fine_formal_order_2": conservative["gci_fine"]}
        print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in summary[k].items()}, flush=True)
    rec = json.loads(OUT.read_text()) if OUT.exists() else {}
    rec["h_study"] = {"base_mesh": MESH3D["o2"], "factors": H_FACTORS, "mpi_ranks": ranks,
                      "seconds": time.time() - t0, "dofs": [row["dofs"] for row in rows],
                      "rows": rows, "dof_ratios": out["dof_ratios"], "warnings": out["warnings"],
                      "Gamma": summary, "L_p_extrapolation": out["extrapolation"].get("L_p_H")}
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote h_study into", OUT)


if __name__ == "__main__":
    if sys.argv[1:2] == ["hstudy"]:
        args = sys.argv[2:]
        hstudy(int(args[args.index("--ranks") + 1]) if "--ranks" in args else int(os.environ.get("SLURM_NTASKS", "4")))
    else:
        main()
