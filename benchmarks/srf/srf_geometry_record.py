"""Cached heterodyne figures of merit of the default SRF cell, for the sensitivity model (beta).

Solves the 2D axisymmetric Project X 650 MHz beta = 0.9 cell shortened axially by 0.86 (beam tubes 50 mm x 100 mm;
at this axial factor the TM020 pump and the TE011 signal are degenerate) with :mod:`axionfem.srf` on two meshes and writes the scale-free figures of
merit that :class:`axionfem.heterodyne.SRFGeometry` needs (xi, eta_10, G pump/signal, B_pk/B0 on the wall and at the
equator, E_pk/(c B0), A/V^(2/3)) with their mesh-convergence differences, to

    python/axionfem/data/srf_projectx086_geometry.json

which :func:`axionfem.heterodyne.projectx_geometry` reads (preset ``PROJECTX_086`` = ``PROJECTX_086_WALL``, B_pk =
the largest wall |B|; diagnostic ``PROJECTX_086_EQUATOR_DIAGNOSTIC``).  The numbers are never
typed in by hand: rerun this script to refresh them.

    python benchmarks/srf/srf_geometry_record.py --update-package-data     # ~2 min (order 3, h = R_eq/80 and R_eq/40)
    python benchmarks/srf/srf_geometry_record.py --n 40 --out variant.json  # any other variant: explicit --out only

The record carries its inputs (cell, axial factor, tubes, mode pair, mesh) and a fingerprint (sha256 of srf.py,
``git describe --always --dirty``); :func:`axionfem.heterodyne.check_record` checks them on every load.  The package
file is written only with ``--update-package-data`` and only for the preset definition (axial factor 0.86, default
tubes, n >= 80, order >= 3).

Reference values: docs/srf/README.md (validation table) and benchmarks/srf/records/srf_2d.json.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import tempfile
import time
from pathlib import Path

from axionfem import __version__, srf
from axionfem import heterodyne as H

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "python" / "axionfem" / "data" / "srf_projectx086_geometry.json"
KEYS = ("f_pump_Hz", "f_signal_Hz", "splitting_Hz", "V_m3", "xi_pump", "xi_signal", "eta_10", "G_pump_ohm",
        "G_signal_ohm", "Bpk_over_B0", "B_equator_over_B0", "Epk_over_cB0", "A_over_V23")


def solve(n: int, order: int, axial_factor: float, work: str) -> dict:
    req = srf.PROJECTX_650_B09["equator_radius"]
    t = time.time()
    d = srf.elliptical_cell_design(axial_factor=axial_factor, h=req / n, order=order, modes=6,
                                   tube_radius=srf.PROJECTX_650_B09["iris_radius"], tube_length=0.100, wall="smooth")
    r = d.solve(f"{work}/cell_o{order}_n{n}", verbose=0)
    o = srf.figures_of_merit(r, "TM020", "TE011", subdiv=4)
    row = {k: float(o[k]) for k in KEYS}
    row.update(Bpk_at_rz_m=o["pump_peaks"]["Bpk_at_rz"], Epk_at_rz_m=o["pump_peaks"]["Epk_at_rz"],
               Bpk_over_B0_face_rms=o["pump_peaks"]["Bpk_over_B0_face_rms"],
               Epk_over_cB0_face_rms=o["pump_peaks"]["Epk_over_cB0_face_rms"],
               mesh={"h_over_Req": 1.0 / n, "order": order, "wall": "smooth",
                     "elements": r.data.get("mesh", {}).get("elements")},
               checks=o["checks"], seconds=time.time() - t)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="R_eq/h of the reference mesh")
    ap.add_argument("--n-check", type=int, default=40, help="R_eq/h of the convergence-check mesh")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--axial-factor", type=float, default=0.86)
    ap.add_argument("--out", default=None, help="output JSON (required unless --update-package-data)")
    ap.add_argument("--update-package-data", action="store_true",
                    help=f"overwrite the package record {OUT.relative_to(ROOT)} (preset definition only)")
    a = ap.parse_args()
    if a.update_package_data:
        if a.out is not None:
            ap.error("--update-package-data writes the package record; do not combine it with --out")
        mn = H.PROJECTX_RECORD_MIN_MESH
        if (a.n < mn["n"] or a.order < mn["order"]
                or abs(a.axial_factor - H.PROJECTX_RECORD_INPUTS["axial_factor"]) > 1e-12):
            ap.error(f"--update-package-data needs the preset definition: --axial-factor "
                     f"{H.PROJECTX_RECORD_INPUTS['axial_factor']}, --n >= {mn['n']}, --order >= {mn['order']}")
        out = OUT
    elif a.out is None:
        ap.error("give --out PATH (the package record is written only with --update-package-data)")
    else:
        out = Path(a.out).resolve()
        if out in (OUT.resolve(), Path(H.PROJECTX_RECORD).resolve()):   # the repo copy or the imported package's
            ap.error("--out points at the package record: use --update-package-data instead")
    with tempfile.TemporaryDirectory(prefix="srf_geo_") as work:
        ref = solve(a.n, a.order, a.axial_factor, work)
        chk = solve(a.n_check, a.order, a.axial_factor, work)

    def git(*args):
        try:
            return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(ROOT), *args], capture_output=True,
                                  text=True, timeout=120).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    sha = git("rev-parse", "--short", "HEAD")
    describe = git("describe", "--always", "--dirty", "--abbrev=12")
    tube_r = srf.PROJECTX_650_B09["iris_radius"]
    rec = {
        "schema": "axionfem.srf_geometry/1",
        "status": "beta",
        "cell": "Project X 650 MHz beta = 0.9 regular cell (srf.PROJECTX_650_B09), axial factor "
                f"{a.axial_factor}, beam tubes 50 mm x 100 mm",
        "pump": "TM020", "signal": "TE011",
        "definitions": "docs/srf/README.md; Bpk_over_B0 is the largest pump |B| on the wall (iris-side lobe), the "
                       "peak wall magnetic field to which a field limit applies. B_equator_over_B0 is the pump |B| at "
                       "the equator, a diagnostic only (not a field limit). B0 is the volume rms. The wall is the "
                       "exact elliptical contour (curved elements) and the wall peaks are taken at the Gauss points "
                       "of the wall faces, as in 3D",
        "inputs": {"cell": {k: list(v) if isinstance(v, tuple) else v for k, v in srf.PROJECTX_650_B09.items()},
                   "axial_factor": a.axial_factor, "tube_radius_m": tube_r, "tube_length_m": 0.100,
                   "pump": "TM020", "signal": "TE011", "n": a.n, "n_check": a.n_check, "order": a.order,
                   "wall": "smooth", "subdiv": 4},
        "fingerprint": {"srf_py_sha256": H.srf_source_sha256(), "git_describe": describe,
                        "dirty": describe.endswith("-dirty") or not describe},
        "figures_of_merit": ref,
        "convergence_check": chk,
        "rel_diff_check_vs_ref": {k: chk[k] / ref[k] - 1 for k in KEYS if k != "splitting_Hz" and ref[k]},
        "generated": {"by": "benchmarks/srf/srf_geometry_record.py", "axionfem": __version__, "git": sha,
                      "date": _dt.date.today().isoformat()},
    }
    if out == OUT:
        H.check_record(rec, path=str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps({k: ref[k] for k in KEYS}, indent=1))
    print("max |rel diff| vs check mesh:", max(abs(v) for v in rec["rel_diff_check_vs_ref"].values()))
    print("wrote", out)


if __name__ == "__main__":
    main()
