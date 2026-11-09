"""SRF heterodyne cavities, axisymmetric validation (beta; docs/srf/README.md).

1. closed-cylinder pairs (TM020/TE011, TM030/TE021, TE012/TM013): 2D solver vs closed forms (axionfem.srf.cylinder_pair);
2. the default elliptical cell (Project X 650 MHz beta = 0.9, axial factor 0.86, beam tubes 50 mm x 100 mm):
   mesh convergence of f, xi, eta_10, B_pk/B0, E_pk/(c B0), G (pump TM020, signal TE011) in h and order;
3. the degeneracy tuner (axial factor at which f_TM020 = f_TE011);
4. uniform scaling (x10): f / 10, scale-free figures unchanged.

Usage: python benchmarks/srf/srf_2d_validation.py [--quick] [--out benchmarks/srf/records/srf_2d.json]
Run on a compute node (the finest meshes take minutes).

The cell runs use the faceted wall (wall="polygon", the record srf_2d.json).  The wall peaks are the solver's wall
Gauss points (no polygon vertices); the polygon_convergence rows of srf_2d.json were computed from lattice peaks that
include the vertices, so a rerun does not reproduce those rows exactly. The smooth-wall reference is benchmarks/srf/srf_2d_smooth_wall.py."""
import argparse
import json
import math
import sys
import tempfile
import time

from axionfem import srf

KEYS = ("f_pump_Hz", "f_signal_Hz", "splitting_Hz", "xi_pump", "xi_signal", "eta_10", "Bpk_over_B0", "Epk_over_cB0",
        "B_equator_over_B0", "G_pump_ohm", "G_signal_ohm", "A_over_V23", "V_m3")


def fom(d, wd, pump="TM020", signal="TE011", subdiv=4):
    t = time.time()
    r = d.solve(wd, verbose=0)
    o = srf.figures_of_merit(r, pump, signal, subdiv=subdiv)
    row = {k: o[k] for k in KEYS}
    row["checks"] = o["checks"]
    row["pump_peaks_at_rz"] = {"B": o["pump_peaks"]["Bpk_at_rz"], "E": o["pump_peaks"]["Epk_at_rz"]}
    row["seconds"] = time.time() - t
    row["mesh"] = {k: r.data.get("mesh", {}).get(k) for k in ("elements", "nodes", "order")}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="benchmarks/srf/records/srf_2d.json")
    a = ap.parse_args()
    base = tempfile.mkdtemp(prefix="srf2d_")
    rec = {"schema": "axionfem.srf_2d_validation/1", "status": "beta", "cylinders": {}, "cell": {}}
    # 1. cylinders
    for pair in srf.CYLINDER_PAIRS:
        ref = srf.cylinder_pair(pair)
        pu, si = pair.split("/")
        rows = []
        for h in ((0.01, 0.005) if a.quick else (0.01, 0.005, 0.0025)):
            d = srf.cylinder_pair_design(pair, 0.1, h=h, modes=8)
            row = fom(d, f"{base}/cyl_{pu}_{h}", pu, si)
            row["h_over_R"] = h / 0.1
            rows.append(row)
            print(pair, h, {k: row[k] for k in ("xi_pump", "eta_10", "Bpk_over_B0", "Epk_over_cB0", "G_pump_ohm",
                                                 "G_signal_ohm")}, flush=True)
        rec["cylinders"][pair] = {"analytic": {k: v for k, v in ref.items() if k not in ("pump", "signal")} |
                                  {"pump": ref["pump"], "signal": ref["signal"]}, "fem": rows}
    # 2. cell convergence
    Req = srf.PROJECTX_650_B09["equator_radius"]
    conv = []
    hs = (10, 20, 40) if a.quick else (10, 20, 40, 80, 160)
    for order in (2, 3):
        for n in hs:
            if order == 3 and n > 80:
                continue
            if n > 80 and a.quick:
                continue
            # the wall polygon is refined with the mesh: angle_tol = 0.4 / n (0.01 at the default h = Req/40)
            d = srf.elliptical_cell_design(axial_factor=0.86, h=Req / n, order=order, modes=6, angle_tol=0.4 / n,
                                           wall="polygon")
            row = fom(d, f"{base}/cell_o{order}_{n}")
            row.update(order=order, h_over_Req=1 / n, wall_size_m=Req / n / 3, angle_tol=0.4 / n,
                       n_wall_vertices=len(d.srf_contour["wall"]))
            conv.append(row)
            print("cell", order, n, {k: row[k] for k in KEYS}, flush=True)
    rec["cell"]["convergence"] = conv
    # wall polygon alone (h = Req/40): the polygon's convex corners are weak E singularities at the iris
    poly = []
    for order in (2, 3):
        for at in (0.05, 0.02, 0.01, 0.005, 0.0025):
            d = srf.elliptical_cell_design(axial_factor=0.86, h=Req / 40, order=order, modes=6, angle_tol=at, wall="polygon")
            row = fom(d, f"{base}/poly_o{order}_{at}")
            row.update(order=order, h_over_Req=1 / 40, angle_tol=at, n_wall_vertices=len(d.srf_contour["wall"]))
            poly.append(row)
            print("poly", order, at, {k: row[k] for k in ("Epk_over_cB0", "Bpk_over_B0", "eta_10")}, flush=True)
    rec["cell"]["polygon_convergence"] = poly
    # subdiv check of the integrals on the finest order-2 mesh
    d = srf.elliptical_cell_design(axial_factor=0.86, h=Req / hs[-1], order=2, modes=6, angle_tol=0.4 / hs[-1], wall="polygon")
    r = d.solve(f"{base}/cell_subdiv", verbose=0)
    rec["cell"]["subdiv_check"] = {s: {k: srf.figures_of_merit(r, subdiv=s)[k] for k in ("eta_10", "Bpk_over_B0",
                                                                                         "Epk_over_cB0")}
                                   for s in (2, 4, 8)}
    # 3. tuner
    for n in ((20,) if a.quick else (40, 80)):
        tu = srf.tune_degeneracy(design_kw={"h": Req / n, "order": 3, "modes": 6, "wall": "polygon"}, bracket=(0.82, 0.90), tol=1e-6,
                                 workdir=f"{base}/tune_{n}")
        rec["cell"].setdefault("tuner", []).append({"h_over_Req": 1 / n, "order": 3, **tu})
        print("tuner", n, tu["axial_factor"], tu["f_Hz"], flush=True)
        d = srf.elliptical_cell_design(axial_factor=tu["axial_factor"], h=Req / n, order=3, modes=6, wall="polygon")
        row = fom(d, f"{base}/cell_tuned_{n}")
        rec["cell"].setdefault("tuned", []).append({"h_over_Req": 1 / n, "order": 3, "axial_factor": tu["axial_factor"],
                                                    **row})
    # 4. scaling
    sc = []
    for s in (1.0, 10.0):
        d = srf.elliptical_cell_design(axial_factor=0.86, scale=s, h=s * Req / 40, order=3, modes=6, wall="polygon")
        row = fom(d, f"{base}/scale_{s}")
        row["scale"] = s
        sc.append(row)
    rec["cell"]["scaling"] = sc
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
