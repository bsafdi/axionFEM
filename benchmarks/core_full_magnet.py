"""DMRadio-Core with its full 2.95 m magnet as an imposed background in the 3D full-wave sweep.

    python benchmarks/core_full_magnet.py run CONFIG [--ranks N]     CONFIG: n1 (the published 30-degree port) or c360
    python benchmarks/core_full_magnet.py axi                        the axisymmetric reference (4 and 61 points)
    python benchmarks/core_full_magnet.py report

The 3D Core studies (published_benchmarks.py core3d*, core_readout_ports.py) shortened the magnet to 0.6 m because a
full-wave domain around the 2.95 m magnet was out of reach, so their Psi is the reduced design's and only ratios carry
over. Here the magnet is a `background`: its analytic field (benchmarks/fullwave3d_background.coil_field, both halves
of the split solenoid of published_benchmarks.design_core_reduced at full height and current) on an (r, z) grid. The
reduced design's coil regions are kept, as vacuum, so the mesh is core_readout_ports' "fast" mesh, and the domain
stays the pickup's (R1 about 1.08 m): the magnet reaches z = +-1.475 m, and its field outside R1 enters through the
background's exterior term (docs/physics/fullwave_background.md). The axisymmetric engine solves the same full design
with the magnet as windings (uniform feed, the reference for c360), at the four points and on a dense 61-point sweep:
the dense sweep gives its L_eff by finite differences, then the scan time by the 3D runs' four-point trapezoid (the
same method) and by the dense trapezoid (the four-point quadrature error). The magnet's segmentation and iron are not
modelled (as in published_benchmarks section 5.1).

Writes benchmarks/records/core_full_magnet.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "python"))   # after PYTHONPATH, so that a package copy (runs/pkgN) can take precedence
sys.path.insert(0, str(HERE))

RUNS = ROOT / "runs" / "core_full_magnet"
OUT = ROOT / "benchmarks" / "records" / "core_full_magnet.json"


def magnet() -> dict:
    import published_benchmarks as P
    g = P.SRC["dmradio_core"]["geometry"]
    Rc = g["bore_diameter_m"][0] / 2
    return {"Rc": Rc, "tw": g["outer_diameter_m"][0] / 2 - Rc, "H": g["H_m"][0], "gc": g["plate_spacing_m"][0],
            "NI": 1.18943e7}


def magnet_grid() -> dict:
    """The split solenoid's field (two halves, NI/2 each) on an (r, z) grid out to r = 3 m, |z| = 4 m."""
    import fullwave3d_background as fb
    from axionfem.background import graded_nodes, grid_block
    m = magnet()
    a1 = m["Rc"]
    a2 = m["Rc"] + m["tw"]

    def field(r, z):
        br1, bz1 = fb.coil_field(r, z, a1=a1, a2=a2, z1=m["gc"] / 2, z2=m["H"] / 2, ampere_turns=m["NI"] / 2)
        br2, bz2 = fb.coil_field(r, z, a1=a1, a2=a2, z1=-m["H"] / 2, z2=-m["gc"] / 2, ampere_turns=m["NI"] / 2)
        return br1 + br2, bz1 + bz2
    rn = graded_nodes(0.0, 3.0, 0.02, fine=[(0.0, 0.8, 0.004)])
    zn = graded_nodes(-4.0, 4.0, 0.02, fine=[(-1.7, 1.7, 0.004)])
    return grid_block(field, rn, zn, source=f"analytic split solenoid {m}", notes="benchmarks/core_full_magnet.py")


def design(config: str):
    import core_readout_ports as CR
    d = CR.design(config, "fast", CR.band(4))
    doc = d.doc
    # the coil regions are kept, as a linear mu_r = 1 material (vacuum): the mesher refines them as it refines windings,
    # so the mesh is that of core_readout_ports' "fast" runs
    doc["windings"] = []
    doc.setdefault("materials", {})["coil_space"] = {"kind": "linear", "mu_r": 1.0}
    for region in doc["regions"]:
        if region["name"].startswith("coil"):
            region["material"] = "coil_space"
    doc["background"] = magnet_grid()
    doc["name"] = doc["name"].replace("reduced", "full_magnet")
    doc["description"] = ("DMRadio-Core pickup (120 cm) with the full 2.95 m magnet as a background field; "
                          f"readout configuration {config} of benchmarks/core_readout_ports.py.")
    return d


DENSE = 61   # the dense axisymmetric sweep: log-spaced, so every 20th point is one of band(4)'s four


def axi_design(points: int = 4):
    import core_readout_ports as CR
    import published_benchmarks as P
    m = magnet()
    d = P.design_core_reduced(H_coil=m["H"])
    d.doc["fullwave"] = CR.band(points)
    return d


def dense_rows(res: dict) -> list[dict]:
    """Rows of the dense axisymmetric sweep in core_readout_ports.rows' form, with L_eff = (w dX/dw + X) / (2 w) from a
    second-order finite difference over the dense sweep (the axisymmetric engine reports no exact dZ/dw)."""
    sweep = res["fullwave"]["sweep"]
    f = np.array([s["f_Hz"] for s in sweep], dtype=float)
    w = 2 * math.pi * f
    X = w * np.array([s["L_H"] for s in sweep], dtype=float)
    dX = np.gradient(X, w, edge_order=2)
    L_eff = (w * dX + X) / (2 * w)
    out = []
    for i, s in enumerate(sweep):
        psi = abs(complex(s["Psi_re"], s["Psi_im"]))
        out.append({"f_Hz": float(f[i]), "L_H": float(X[i] / w[i]), "L_eff_H": float(L_eff[i]), "Psi_abs": psi,
                    "F_eff": psi ** 4 / L_eff[i] ** 2 if L_eff[i] > 0 else None})
    return out


def run(config: str, ranks: int) -> None:
    d = design(config)
    t0 = time.time()
    try:
        d.solve(RUNS / config, verbose=1, mpi_ranks=ranks)
    except Exception as exc:  # noqa: BLE001  (results.json is written by the C++ solver)
        print("solve raised:", type(exc).__name__, exc, flush=True)
    print(f"{config}: {time.time() - t0:.0f} s", flush=True)


def run_axi() -> None:
    t0 = time.time()
    axi_design().solve(RUNS / "axi", verbose=1)
    print(f"axi: {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    axi_design(DENSE).solve(RUNS / "axi_dense", verbose=1)
    print(f"axi_dense: {time.time() - t0:.0f} s", flush=True)


def report() -> None:
    import core_readout_ports as CR
    import fullwave3d_background as fb
    rec = {"check": "core_full_magnet", "magnet": magnet(),
           "date": time.strftime("%Y-%m-%d"), "configs": {}}
    for name in ("n1", "c360", "axi"):
        path = RUNS / name / "results.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text())
        fw = res["fullwave"]
        checks = res.get("checks") if isinstance(res.get("checks"), dict) else {}   # a list in axisymmetric results
        psi_primary, psi_reciprocal = fb.mqs_forms(res, checks)   # primary and reciprocal 3D MQS forms, oriented
        entry = {"rows": CR.rows(res), "Psi_MQS": psi_primary, "L_MQS": res["coupling"]["L_p_H"],
                 "Psi_MQS_reciprocal": psi_reciprocal,
                 "dofs": res.get("mesh", {}).get("dofs"), "background": fw.get("background"),
                 "exterior_term_background": fw.get("exterior_term_background"),
                 "warnings": res.get("warnings", [])}
        for row, s in zip(entry["rows"], fw["sweep"]):
            bg = s.get("Psi_background")
            if bg:
                row["Psi_background_volume"] = [bg["volume"]["re"], bg["volume"]["im"]]
                ext = bg.get("exterior")   # null where no exterior term is computed
                row["Psi_background_exterior"] = [ext["re"], ext["im"]] if ext else None
            row["relative_residual"] = (s.get("solver") or {}).get("relative_residual")
        entry["scan_time_yr"] = CR.scan_time_yr(entry["rows"])
        rec["configs"][name] = entry
    c = rec["configs"]
    if "c360" in c and "axi" in c:
        rec["c360_vs_axi"] = [{"f_Hz": a["f_Hz"], "Psi": a["Psi_abs"] / b["Psi_abs"], "L": a["L_H"] / b["L_H"],
                               "F_eff": (a["F_eff"] / b["F_eff"]) if a["F_eff"] and b["F_eff"] else None}
                              for a, b in zip(c["c360"]["rows"], c["axi"]["rows"])]
    if "n1" in c and "c360" in c and c["n1"]["scan_time_yr"] and c["c360"]["scan_time_yr"]:
        rec["n1_over_c360_scan_time"] = c["n1"]["scan_time_yr"] / c["c360"]["scan_time_yr"]
    # the axisymmetric reference by the same integrand: its L_eff from the dense sweep, then the same four points and
    # trapezoid in ln f as the 3D runs; and the dense trapezoid, whose difference is the four-point quadrature error
    path = RUNS / "axi_dense" / "results.json"
    if path.exists():
        rows = dense_rows(json.loads(path.read_text()))
        step = (DENSE - 1) // 3
        four = rows[::step]
        dense = {"points": DENSE, "scan_time_yr_four_points": CR.scan_time_yr(four),
                 "scan_time_yr_all_points": CR.scan_time_yr(rows), "four_point_rows": four}
        if dense["scan_time_yr_four_points"] and dense["scan_time_yr_all_points"]:
            four_over_all = dense["scan_time_yr_four_points"] / dense["scan_time_yr_all_points"]
            dense["four_point_quadrature_error"] = four_over_all - 1
        if "c360" in c:
            dense["c360_F_eff_over_axi"] = [(a["F_eff"] / b["F_eff"]) if a["F_eff"] and b["F_eff"] else None
                                            for a, b in zip(c["c360"]["rows"], four)]
            if c["c360"]["scan_time_yr"] and dense["scan_time_yr_four_points"]:
                dense["c360_over_axi_scan_time_same_method"] = (c["c360"]["scan_time_yr"]
                                                                / dense["scan_time_yr_four_points"])
        rec["axi_dense"] = dense
    print(json.dumps({k: v for k, v in rec.items() if k != "configs"}, indent=1, default=float), flush=True)
    for name, e in c.items():
        print(name, "scan time", e["scan_time_yr"], [(r["f_Hz"], r["Psi_abs"], r["L_eff_H"]) for r in e["rows"]],
              flush=True)
    OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("wrote", OUT, flush=True)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "axi", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    import axionfem  # noqa: F401  (the package on PYTHONPATH, before core_readout_ports puts python/ first)
    RUNS.mkdir(parents=True, exist_ok=True)
    if a[0] == "run":
        ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "16"))
        run(a[1], ranks)
    elif a[0] == "axi":
        run_axi()
    report()


if __name__ == "__main__":
    main()
