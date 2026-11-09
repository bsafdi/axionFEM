"""SRF heterodyne cell, 3D cross-check (beta; docs/srf/README.md).

The default elliptical cell (axial factor 0.86) as a full 3D model (no mirror planes), shift-invert about the
2D TM020/TE011 frequency: f, xi, eta_10, G, B_pk/B0, E_pk/(c B0) of the pump TM020 and signal TE011, identified by
their E_phi fraction (0 / 1) and frequency, for a sequence of element sizes (mesh convergence).

Usage (MPI; e.g. under a scheduler): srun python benchmarks/srf/srf_3d_validation.py --n 8 12 16 [--order 2] [--wall smooth] [--out ...]

--wall smooth (default) meshes the exact wall (elliptical arcs, curved second-order elements on it);
--wall polygon the faceted polyline (angle_tol 0.1; the records srf_3d_o2.json / srf_3d_o3.json).
Wall peaks (smooth wall): srf.azimuthal_wall_peaks, a quadratic fit in the meridional arclength through the wall
quadrature values near the peak (all azimuths), the 3D estimate of the meridional function the 2D solver samples at its
wall Gauss points (docs/srf/README.md, "Wall peaks"); the single-point maxima are kept as *_pointwise_max.
"""
import math
import argparse
import json
import os
import tempfile
import time

from axionfem import srf

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=float, nargs="+", default=[8.0], help="h = Req / n")
ap.add_argument("--order", type=int, default=2)
ap.add_argument("--modes", type=int, default=16)
ap.add_argument("--axial-factor", type=float, default=0.86)
ap.add_argument("--f2d", type=float, nargs=2, default=[1.41657e9, 1.41654e9], help="2D f_pump f_signal (Hz)")
ap.add_argument("--out", default="benchmarks/srf/records/srf_3d.json")
ap.add_argument("--workdir", default=None)
ap.add_argument("--ranks", type=int, default=1)
ap.add_argument("--wall", choices=("smooth", "polygon"), default="smooth")
ap.add_argument("--curvature-points", type=int, default=None, help="smooth wall: default max(16, 1.5 n)")
ap.add_argument("--repeak", action="store_true",
                help="recompute the wall peaks (srf.azimuthal_wall_peaks) of the rows of --out from the wall-sample "
                     "files in --workdir/n<n>_o<order> (no solve)")
a = ap.parse_args()
if a.repeak:
    con = srf.elliptical_cell_contour(axial_factor=a.axial_factor)
    rec = json.load(open(a.out))
    for o in rec["rows"]:
        wd = f"{a.workdir}/n{float(round(1 / o['h_over_Req']))}_o{o['order']}"
        modes = json.load(open(f"{wd}/results.json"))["fullwave"]["eigenmodes"]["modes"]
        pw = modes[o["i_pump"]]["peak_wall_fields"]
        az = srf.azimuthal_wall_peaks(pw, con)
        o["azimuthal"] = az
        o["Bpk_over_B0"], o["Epk_over_cB0"] = az["Bpk_over_B0"], az["Epk_over_cB0"]
        o["Bpk_at_rz"], o["Epk_at_rz"] = az["Bpk_at_rz"], az["Epk_at_rz"]
        o["peak_definition"] = "azimuthal"
        print(round(1 / o["h_over_Req"]), o["order"], o["Bpk_over_B0"], o["Epk_over_cB0"], flush=True)
    rec["repeak"] = "wall peaks recomputed from the saved wall samples with the srf.py of this commit"
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    raise SystemExit(0)
Req = srf.PROJECTX_650_B09["equator_radius"]
base = a.workdir or tempfile.mkdtemp(prefix="srf3d_")
rows = []
if os.path.exists(a.out):
    rows = json.load(open(a.out)).get("rows", [])
for n in a.n:
    t = time.time()
    d = srf.elliptical_cell_design_3d(axial_factor=a.axial_factor, h=Req / n, order=a.order, modes=a.modes,
                                      target_f_Hz=sum(a.f2d) / 2, wall=a.wall,
                                      curvature_points=a.curvature_points)
    r = d.solve(f"{base}/n{n}_o{a.order}", verbose=0, mpi_ranks=a.ranks)
    o = srf.figures_of_merit_3d(r, *a.f2d, contour=d.srf_contour)
    fw = r.fullwave
    modes = fw["eigenmodes"]["modes"]
    rz = lambda x: [math.hypot(x[0], x[1]), x[2]] if x else None  # noqa: E731
    o.update(h_over_Req=1 / n, order=a.order, wall=a.wall, seconds=time.time() - t,
             curvature_points=d.doc["mesh"].get("curvature_points"),
             Epk_pointwise_max_at_rz=rz(o.get("Epk_at_xyz")), Bpk_pointwise_max_at_rz=rz(o.get("Bpk_at_xyz")),
             curved_elements_invalid=r.data.get("mesh", {}).get("info", {}).get("curved_elements_invalid"),
             mesh_quality=r.data.get("mesh", {}).get("info", {}).get("quality"),
             warnings=[w for w in r.data.get("warnings", []) if "curved" in str(w) or "element" in str(w)],
             converged=fw["eigenmodes"].get("converged"),
             mesh={k: r.data.get("mesh", {}).get(k) for k in ("elements", "dofs", "order")},
             spectrum=[{"f_Hz": m["f_Hz"], "E_azimuthal_fraction": m.get("E_azimuthal_fraction"),
                        "G_ohm": m.get("losses", {}).get("G_ohm")} for m in modes])
    rows = [x for x in rows if not (x["h_over_Req"] == 1 / n and x["order"] == a.order
                                    and x.get("wall", "polygon") == a.wall
                                    and x.get("curvature_points") == o["curvature_points"])] + [o]
    print(json.dumps({k: v for k, v in o.items() if k != "spectrum"}, default=float), flush=True)
    with open(a.out, "w") as fh:
        json.dump({"schema": "axionfem.srf_3d_validation/1", "status": "beta", "rows": rows}, fh, indent=1,
                  default=float)
