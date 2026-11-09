"""HAYSTAC one-rod cavity in 3D with the alumina axles: rod-angle sweep with mode tracking (beta).

Reference: benchmarks/cavity/refs/haystac_rod_2006.json (2006.01248 Fig. 3 / Fig. 6, CST). Geometry (JSON
"geometry", each value with its source): barrel R = 2 in, H = 10 in; copper rod r = 1 in, 9.98 in long, 0.010 in
gaps at both end caps; the rod turns about an axle 0.475 in from the cavity axis and from the rod axis, so the rod
centre is at d(theta) = 2 (0.475 in) sin(theta/2) (theta = 0: concentric). Axles: alumina (eps_r 9.8, the JSON's
proposal; not given in the paper), OD 0.25 in, at the pivot, crossing both gaps (CST model: "0.25 in OD alumina
axles at the pivot point"). Variant `--holes`: the axles continue 10 mm into end-cap holes of their own diameter
(the JSON's proposed sensitivity case). Walls: skin copper sigma = 5.8e7 S/m at 293 K (not stated in the paper).

The model keeps z >= 0 with a PEC plane at z = 0 (mesh.symmetry): the geometry (gaps, axles) is mirror-symmetric
in z, and TM010-like modes (E_z even in z) have no tangential E on that plane. Modes odd in z (TM011-like) need a PMC plane: stage `pmc`.

Targets (refs README): df/dtheta and C at theta ~ 5-20 deg; f(theta = 0) is NOT decidable from Fig. 3 (do not
compare it). Fig. 6 points are given vs frequency only; they are compared at equal frequency and at the CST angle read off
the Fig. 3 lower line.

Usage (inside a batch allocation; AXF_TRACK_RANKS ranks per solve):
  $PY haystac_rod_3d.py conv              # mesh study at theta = 10 deg
  $PY haystac_rod_3d.py sweep [--holes]   # 0..32 deg with tracking and refinement
  $PY haystac_rod_3d.py analyze           # record + figure (login node, cached)
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

IN = 0.0254
R, H = 2 * IN, 10 * IN
A, HROD = 1 * IN, 9.98 * IN
G = (H - HROD) / 2  # 0.254 mm
P = 0.475 * IN
AXLE_R = 0.125 * IN
AXLE_EPS = 9.8
SIGMA = 5.8e7
BENCH = "haystac_rod_3d"
PROD = {"h": 6e-3, "hs": 3e-3, "hg": 0.7e-3}
CONV = [{"h": 9e-3, "hs": 4.5e-3, "hg": 1.0e-3}, {"h": 6e-3, "hs": 3e-3, "hg": 0.7e-3},
        {"h": 4.5e-3, "hs": 2.2e-3, "hg": 0.5e-3}]


def rod_centre(theta_deg: float):
    t = math.radians(theta_deg)
    return P - P * math.cos(t), P * math.sin(t)


_TARGET_CACHE: dict = {}


def target_f(theta_deg: float) -> float:
    """Shift for the shift-invert solve: the axionFEM cross-section (no gaps, no axles) TM010-like frequency of the
    same rod position, lowered by 0.1 % (the 2D gap shift at theta = 0). Only the centre of the window."""
    key = round(theta_deg, 6)
    if key not in _TARGET_CACHE:
        from axionfem import cavity_xsec as cx

        x, y = rod_centre(theta_deg)
        d = cx.cylinder_rods_design(R, [{"radius": A, "center": [x, y]}], modes=4, h=R / 30)
        res = d.solve(workdir=cm.WORK / BENCH / "xsec_target" / f"t{key:.4f}", verbose=0)
        tm = [r for r in res.cavity_modes() if r["family"] == "TM"]
        _TARGET_CACHE[key] = 0.999 * max(tm, key=lambda r: r["C"] or 0)["f_Hz"]
    return _TARGET_CACHE[key]


def build(theta_deg: float, *, h: float, hs: float, hg: float, axles: bool = True, holes: float = 0.0,
          modes: int = 16, order: int = 2, margin: float = 3e-3, zsym: str = "pec"):
    from axionfem import cavity as cv

    cx_, cy_ = rod_centre(theta_deg)
    zt = H / 2
    dom = cm.cyl(0, 0, -H / 2, H / 2, R)
    if axles and holes > 0:
        dom = {"union": [dom, cm.cyl(P, 0, zt, zt + holes, AXLE_R), cm.cyl(P, 0, -zt - holes, -zt, AXLE_R)]}
    regions = [{"name": "cavity", "material": "vacuum", "solid": dom},
               {"name": "gap_hi", "material": "vacuum", "mesh_size": hg,
                "solid": cm.cyl(cx_, cy_, zt - G - margin, zt, A + margin)},
               {"name": "gap_lo", "material": "vacuum", "mesh_size": hg,
                "solid": cm.cyl(cx_, cy_, -zt, -zt + G + margin, A + margin)}]
    if axles:
        for nm, z0, z1 in (("axle_hi", zt - G, zt + holes), ("axle_lo", -zt - holes, -zt + G)):
            regions.append({"name": nm, "material": "alumina", "mesh_size": hg, "solid": cm.cyl(P, 0, z0, z1, AXLE_R)})
    regions.append({"name": "rod", "material": "wall_cu", "surface_size": hs,
                    "solid": cm.cyl(cx_, cy_, -zt + G, zt - G, A)})
    d = cv._design3d(BENCH, regions, dom, (-0.040, -0.004, 0.37 * H), h=h, order=order, modes=modes,
                     target_f_Hz=target_f(theta_deg), eigensolver={"method": "shift_invert"},
                     symmetry={"z": zsym}, B_T=(0.0, 0.0, 1.0), wall_material=cm.skin(SIGMA),
                     losses=True, mode_overlaps=False, surface_size=hs, remove_pec=True)
    if axles:
        d.doc["materials"]["alumina"] = {"kind": "linear", "eps_r": AXLE_EPS}
    return d


def probes():
    """Probe lattice in the kept half (z > 0) of the barrel (the rod interior is invalid at each step)."""
    from axionfem.cavity_track import probe_points_3d

    return probe_points_3d(bounds=((-R, -R, 0.0), (R, R, H / 2)), n=(11, 11, 9), inset=0.03)


def keep(p):
    """Probe points inside the barrel."""
    return [q for q in p if math.hypot(q[0], q[1]) < 0.97 * R]


def run_sweep(thetas, mesh, tag, *, holes=0.0, refine=3, min_step=0.25, axles=True, **kw):
    from axionfem import cavity as cv

    t0 = time.time()
    T = cv.sweep(lambda th: build(th, holes=holes, axles=axles, **mesh, **kw), thetas, workdir=cm.WORK / BENCH / tag,
                 probes=keep(probes()), solve_kw={"mpi_ranks": cm.RANKS}, refine=refine, min_step=min_step,
                 track_kw={"degenerate_rtol": 1e-5})
    (cm.WORK / BENCH / tag / "tracking.json").write_text(json.dumps(T.to_dict(), indent=1, default=str))
    print(f"{tag}: {len(T.steps)} steps in {time.time() - t0:.0f} s")
    print(T.summary())
    return T


def tm010_like(step):
    """The TM010-like mode of a step: largest C."""
    return max(step.modes, key=lambda m: m.get("C") or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["conv", "sweep", "recheck", "pmc", "holes", "analyze"])
    a = ap.parse_args()
    if a.stage == "conv":
        for k, mesh in enumerate(CONV):
            run_sweep([10.0], mesh, f"conv{k}", refine=0)
    elif a.stage == "sweep":
        run_sweep([float(t) for t in range(0, 33, 1)], PROD, "sweep", refine=3, min_step=0.25)
    elif a.stage == "recheck":  # re-measure selected angles after a solver/mesher change (use a fresh AXF_TRACK_WORK)
        run_sweep([2.0, 6.0, 10.0, 14.0, 31.0], PROD, "recheck", refine=0)
    elif a.stage == "pmc":
        run_sweep([2.0, 6.0, 10.0, 14.0], PROD, "pmc", zsym="pmc", refine=0)
    elif a.stage == "holes":
        run_sweep([5.0, 10.0, 15.0], PROD, "holes", holes=10e-3, refine=0)
        run_sweep([5.0, 10.0, 15.0], PROD, "noaxles", axles=False, refine=0)
    else:
        from haystac_analyze import analyze  # noqa: E402

        analyze()


if __name__ == "__main__":
    main()
