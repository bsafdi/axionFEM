"""SRF cell, 2D reference with the smooth (exact elliptical) wall (beta; docs/srf/README.md).

The default cell (Project X 650 MHz beta = 0.9, axial factor 0.86, tubes 50 mm x 100 mm) with ``wall="smooth"``: the
wall is the exact contour (elliptical arcs; curved second-order elements on it), and the wall peaks are the largest
normal |E| / tangential |B| over the Gauss points of the wall faces -- the definition of the 3D solver. Mesh
convergence in h (R_eq/10 ... R_eq/160) and order (2, 3). This is the 2D column the 3D runs
(srf_3d_validation.py --wall smooth) are compared with.

Usage (the finest meshes take minutes; use a compute node): python benchmarks/srf/srf_2d_smooth_wall.py
    [--out benchmarks/srf/records/srf_2d_smooth.json] [--quick]
"""
import argparse
import json
import tempfile
import time

from axionfem import srf

KEYS = ("f_pump_Hz", "f_signal_Hz", "splitting_Hz", "xi_pump", "xi_signal", "eta_10", "Bpk_over_B0", "Epk_over_cB0",
        "B_equator_over_B0", "G_pump_ohm", "G_signal_ohm", "A_over_V23", "V_m3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="benchmarks/srf/records/srf_2d_smooth.json")
    a = ap.parse_args()
    Req = srf.PROJECTX_650_B09["equator_radius"]
    base = tempfile.mkdtemp(prefix="srf2ds_")
    rows = []
    hs = (10, 20, 40) if a.quick else (10, 20, 40, 80, 160)
    for order in (2, 3):
        for n in hs:
            if order == 3 and n > 80:
                continue
            t = time.time()
            d = srf.elliptical_cell_design(axial_factor=0.86, h=Req / n, order=order, modes=6, wall="smooth")
            r = d.solve(f"{base}/o{order}_n{n}", verbose=0)
            o = srf.figures_of_merit(r, subdiv=4)
            row = {k: o[k] for k in KEYS}
            pk = o["pump_peaks"]
            row.update(order=order, h_over_Req=1 / n, wall="smooth", seconds=time.time() - t,
                       Bpk_over_B0_face_rms=pk["Bpk_over_B0_face_rms"], Epk_over_cB0_face_rms=pk["Epk_over_cB0_face_rms"],
                       Et_max_over_cB0=pk["Et_max_over_cB0"], pump_peaks_at_rz={"B": pk["Bpk_at_rz"], "E": pk["Epk_at_rz"]},
                       checks=o["checks"], mesh={k: r.data.get("mesh", {}).get(k) for k in ("elements", "nodes", "order")})
            rows.append(row)
            print(order, n, {k: round(row[k], 6) for k in ("xi_pump", "eta_10", "Bpk_over_B0", "Epk_over_cB0",
                                                             "Epk_over_cB0_face_rms", "G_pump_ohm", "G_signal_ohm")},
                  flush=True)
    rec = {"schema": "axionfem.srf_2d_smooth/1", "status": "beta",
           "cell": "PROJECTX_650_B09 x axial factor 0.86, tubes 50 mm x 100 mm, wall='smooth'",
           "peak_definition": "largest normal |E| / tangential |B| over the Gauss points of the wall faces "
                              "(evaluated in the cavity element), B0 = volume rms",
           "rows": rows}
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
