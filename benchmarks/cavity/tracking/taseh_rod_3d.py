"""TASEH single copper tuning rod in 3D (arXiv:2205.01477, Ansys HFSS): f, Q0, C vs rod angle with mode tracking.
beta.

Geometry (2205.01477 p. 4, "Figure 4(c) shows the top view ..."; refs JSON benchmarks/cavity/refs/taseh_rod_2205.json):
inner D = 50 mm, L = 120 mm; inner corners rounded with a 3 mm radius (p. 5, Fabrication); copper rod 4 mm in
diameter and 114 mm long; rotation axis 19 mm from the centre; off-axis length (rod centre to rotation axis)
3.5 mm; 1 mm gap between the transverse part of the rod end and the top/bottom wall; Teflon axles at the rod ends
(Fig. 4(b); no dimensions). V = 0.234 L with the rod.

Modelling choices (gaps of the paper; the refs JSON lists them with proposals):
- rod ends (JSON baseline A): the 114 mm is the distance between the centrelines of 4 mm transverse arms that run
  from the rod to the rotation axis (modelled 4 mm high x 3.6 mm wide); arm surface 1 mm from the end walls; the
  vertical rod ends flat at the arms'
  outer surface; a 4 mm-diameter, 4 mm-high knuckle on the rotation axis joins the arm to the axle. Variant B:
  total copper height 114 mm (arm centrelines at
  +-55 mm, 3 mm gaps).
- Teflon axles: 3 mm diameter, eps_r = 2.1 (not given), on the rotation axis across the 1 mm gaps; no end-cap holes.
  Dielectric loss not modelled (tan delta not given).
- theta = 0: rod centre farthest from the axis (r = 22.5 mm; JSON inference from the monotonic f rise); rotation
  axis on +x; the probes are omitted (eigenmode benchmark; they couple weakly).
- walls and rod: skin copper sigma = 5.8e7 S/m at 293 K (HFSS "bulk copper", value not printed).
- 3 mm corner fillets (revolved profile) only as a variant (`fillet=True`, theta >= 60 deg): with the baseline-A rod
  end the rod (outer edge at r = 24.5 mm, top of the rod at z = 59 mm for theta = 0) would cut into a 3 mm fillet
  (wall radius 24.24 mm at z = 59 mm), so the published model cannot have both near theta = 0; the production model
  has square corners.
The model keeps z >= 0 with a PEC plane at z = 0 (mesh.symmetry; the geometry is mirror-symmetric in z): the
modes with E_z even in z (TM010-like). Stage `pmc` computes the odd family (PMC plane) for the full mode map.

Usage: $PY taseh_rod_3d.py conv | sweep | variants | analyze
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common3d as cm  # noqa: E402

MM = 1e-3
RC, LC = 25 * MM, 120 * MM
FIL = 3 * MM
ROD_R, ROD_LEN = 2 * MM, 114 * MM
AX_OFF, ARM = 19 * MM, 3.5 * MM
GAP = 1 * MM
AXLE_R, AXLE_EPS = 1.5 * MM, 2.1
SIGMA = 5.8e7
BENCH = "taseh_rod_3d"
PROD = {"h": 5e-3, "hs": 0.8e-3, "hg": 0.5e-3}
CONV = [{"h": 7e-3, "hs": 1.2e-3, "hg": 0.8e-3}, {"h": 5e-3, "hs": 0.8e-3, "hg": 0.5e-3},
        {"h": 3.5e-3, "hs": 0.55e-3, "hg": 0.35e-3}]


def rod_centre(theta_deg: float):
    t = math.radians(theta_deg)
    return AX_OFF + ARM * math.cos(t), ARM * math.sin(t)


_TARGET: dict = {}


def target_f(theta_deg: float) -> float:
    """Window centre: the axionFEM cross-section TM010-like f of the same rod position (rod spanning the length)."""
    key = round(theta_deg, 6)
    if key not in _TARGET:
        from axionfem import cavity_xsec as cx

        x, y = rod_centre(theta_deg)
        d = cx.cylinder_rods_design(RC, [{"radius": ROD_R, "center": [x, y]}], modes=4, h=RC / 30)
        res = d.solve(workdir=cm.WORK / BENCH / "xsec_target" / f"t{key:.4f}", verbose=0)
        tm = [r for r in res.cavity_modes() if r["family"] == "TM"]
        _TARGET[key] = 0.998 * max(tm, key=lambda r: r["C"] or 0)["f_Hz"]
    return _TARGET[key]


def barrel(fillet: bool):
    if not fillet:
        return cm.cyl(0, 0, -LC / 2, LC / 2, RC)
    pts = [[0.0, -LC / 2, 0.0], [RC, -LC / 2, FIL], [RC, LC / 2, FIL], [0.0, LC / 2, 0.0]]
    return {"revolve": {"profile": {"polygon": {"points": pts}}, "angles": [0, 360], "center": [0, 0, 0],
                        "axis": [0, 0, 1]}}


def rod_solid(theta_deg: float, variant: str = "A"):
    """The rod: the vertical rod (flat ends at the arms' outer surface), per end a transverse arm (4 mm high, from
    the rod axis to the rotation axis) and a knuckle cylinder on the rotation axis (the arm's end, joined to the
    Teflon axle). Returns (solid, arm centreline height)."""
    x, y = rod_centre(theta_deg)
    zc = ROD_LEN / 2 if variant == "A" else ROD_LEN / 2 - ROD_R  # arm centreline
    parts = [cm.cyl(x, y, -zc - ROD_R, zc + ROD_R, ROD_R)]
    for s in (1, -1):
        z = s * zc
        # arm: a box (4 mm high, 3.6 mm wide so that its corners stay inside the two cylinders: no tangent
        # contacts, which OCC fuses badly) from the rod axis to the rotation axis
        ang = math.degrees(math.atan2(-y, AX_OFF - x))
        parts.append({"box": {"min": [x, y - 0.9 * ROD_R, z - ROD_R], "max": [x + ARM, y + 0.9 * ROD_R, z + ROD_R]},
                      "rotate": {"axis": [0.0, 0.0, 1.0], "origin": [x, y, 0.0], "angle_deg": ang}})
        parts.append(cm.cyl(AX_OFF, 0.0, z - ROD_R, z + ROD_R, ROD_R))  # knuckle on the rotation axis
    return {"union": parts}, zc


def build(theta_deg: float, *, h: float, hs: float, hg: float, variant: str = "A", fillet: bool = False,
          axles: bool = True, modes: int = 10, order: int = 2, margin: float = 2.5e-3, zsym: str = "pec"):
    from axionfem import cavity as cv

    rod, zc = rod_solid(theta_deg, variant)
    dom = barrel(fillet)
    # refinement around both rod ends: a cylinder about the rotation axis that holds the arm, the knuckle and the
    # rod end at every angle (radius limited by the barrel: the rod comes within 0.5 mm of it at theta = 0)
    rr = min(ARM + ROD_R + margin, (RC - FIL if fillet else RC) - AX_OFF - 0.05 * MM)  # inside the (filleted) wall
    regions = [{"name": "cavity", "material": "vacuum", "solid": dom}]
    for nm, s in (("end_hi", 1), ("end_lo", -1)):
        z0, z1 = sorted((s * (zc - ROD_R - margin), s * LC / 2))
        regions.append({"name": nm, "material": "vacuum", "mesh_size": hg, "solid": cm.cyl(AX_OFF, 0, z0, z1, rr)})
    if axles:
        for nm, s in (("axle_hi", 1), ("axle_lo", -1)):
            z0, z1 = sorted((s * zc, s * LC / 2))
            regions.append({"name": nm, "material": "teflon", "mesh_size": hg, "solid": cm.cyl(AX_OFF, 0, z0, z1, AXLE_R)})
    regions.append({"name": "rod", "material": "wall_cu", "surface_size": hs, "solid": rod})
    d = cv._design3d(BENCH, regions, dom, (-0.012, -0.007, 0.37 * LC / 2 + 0.01), h=h, order=order, modes=modes,
                     target_f_Hz=target_f(theta_deg), eigensolver={"method": "shift_invert"}, symmetry={"z": zsym},
                     B_T=(0.0, 0.0, 1.0), wall_material=cm.skin(SIGMA), losses=True, mode_overlaps=False,
                     surface_size=hs * 2, remove_pec=True)
    if axles:
        d.doc["materials"]["teflon"] = {"kind": "linear", "eps_r": AXLE_EPS}
    return d


def probes():
    from axionfem.cavity_track import probe_points_3d

    p = probe_points_3d(bounds=((-RC, -RC, 0.0), (RC, RC, LC / 2)), n=(13, 13, 9), inset=0.03)
    return [q for q in p if math.hypot(q[0], q[1]) < 0.96 * RC]


def run_sweep(thetas, mesh, tag, *, refine=3, min_step=0.5, **kw):
    from axionfem import cavity as cv

    t0 = time.time()
    T = cv.sweep(lambda th: build(th, **mesh, **kw), thetas, workdir=cm.WORK / BENCH / tag, probes=probes(),
                 solve_kw={"mpi_ranks": cm.RANKS}, refine=refine, min_step=min_step,
                 track_kw={"degenerate_rtol": 1e-5})
    (cm.WORK / BENCH / tag / "tracking.json").write_text(json.dumps(T.to_dict(), indent=1, default=str))
    print(f"{tag}: {len(T.steps)} steps in {time.time() - t0:.0f} s")
    print(T.summary())
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["conv", "conv0", "sweep", "pmc", "variants", "analyze"])
    a = ap.parse_args()
    if a.stage == "conv":
        for k, mesh in enumerate(CONV):
            run_sweep([90.0], mesh, f"conv{k}", refine=0)
    elif a.stage == "conv0":  # theta = 0: the rod 0.5 mm from the barrel (narrow-gap currents, Q)
        for k, mesh in enumerate(CONV):
            run_sweep([0.0], mesh, f"conv0_{k}", refine=0)
        run_sweep([0.0], {"h": 5e-3, "hs": 0.3e-3, "hg": 0.5e-3}, "conv0_fine_rod", refine=0)
    elif a.stage == "sweep":
        run_sweep([float(t) for t in range(0, 181, 5)], PROD, "sweep", refine=3, min_step=0.6)
    elif a.stage == "pmc":
        run_sweep([float(t) for t in range(0, 181, 10)], PROD, "pmc", zsym="pmc", refine=0)
    elif a.stage == "variants":
        th = [0.0, 60.0, 120.0, 180.0]
        run_sweep(th, PROD, "variantB", variant="B", refine=0)
        run_sweep(th, PROD, "noaxles", axles=False, refine=0)
        run_sweep([60.0, 120.0, 180.0], PROD, "fillet", fillet=True, refine=0)
    else:
        from taseh_analyze import analyze  # noqa: E402

        analyze()


if __name__ == "__main__":
    main()
