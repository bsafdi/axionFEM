"""Validation record for the force post-processing.

Four cases with known answers:

* an isolated current loop: the net axial force it exerts on itself is exactly zero, so the raw
  Lorentz integral measures the discretization error and `Fz_excluding_self_N` must vanish;
* that loop plus a small superconducting sphere above it: the force on the sphere is the
  point-dipole gradient force -(pi a^3/mu0) d(B^2)/dz of the thin-loop on-axis field, and the
  reaction on the loop must be equal and opposite (Newton's third law);
* the CORE template: up/down symmetric, so the exact Fz on every body is zero and the closure
  residual is the error of the Maxwell-pressure integral on the cavity;
* the toroid template: also symmetric, with a poloidal winding (no self-force estimate).

Each case also checks that the distributed loads integrate to the net force.

The solver's working directories are large and are not part of the record: they go to a temporary
directory, or to WORKDIR if one is given.

usage: force_validation.py OUT.json [order] [WORKDIR]
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import axionfem as af

MU0 = 4e-7 * math.pi
R_LOOP, NI, A_SPHERE, Z_SPHERE = 1.0, 1e4, 0.06, 0.4


def loop_design(with_sphere: bool, order: int):
    regions = [{"name": "coil", "material": "winding",
                "shape": {"rect": {"r": [R_LOOP - 0.01, R_LOOP + 0.01], "z": [-0.01, 0.01]}},
                "surface_size": 0.005}]
    doc = {"schema": "axionfem.design/1", "name": "loop_sphere" if with_sphere else "loop",
           "regions": regions,
           "windings": [{"name": "coil", "region": "coil", "type": "azimuthal", "ampere_turns": NI}],
           "mesh": {"order": order, "size": 0.05, "surface_size": 0.01}}
    if with_sphere:
        regions.append({"name": "sphere", "material": "nb", "mesh_size": A_SPHERE / 6,
                        "shape": {"disk": {"center": [0, Z_SPHERE], "radius": A_SPHERE}}})
        doc["superconductors"] = [{"body": "sphere", "azimuthal": {"flux": 0}}]
    return af.Design(doc)


def dipole_force() -> float:
    """-(pi a^3/mu0) d(B^2)/dz for the on-axis field of a thin loop: the force on a perfectly
    diamagnetic sphere of radius a, to leading order in a/z."""
    s = R_LOOP**2 + Z_SPHERE**2
    Bz = MU0 * NI * R_LOOP**2 / (2 * s**1.5)
    dBz = -3 * MU0 * NI * R_LOOP**2 * Z_SPHERE / (2 * s**2.5)
    return -(math.pi * A_SPHERE**3 / MU0) * 2 * Bz * dBz


def summarize(res) -> dict:
    f = res.forces
    out = {"bodies": {}, "closure": {k: v for k, v in f.get("closure", {}).items() if k != "bodies"},
           "distributions": {}}
    for k, v in f.items():
        if k in ("distributions", "closure"):
            continue
        out["bodies"][k] = v
    for name, d in f.get("distributions", {}).items():
        if d["geometry"] == "surface":
            integral = sum(fz * dl for fz, dl in zip(d["f_z_N_per_m"], d["dl_m"]))
        else:
            integral = sum(d["f_z_N"])
        out["distributions"][name] = {
            "kind": d["kind"], "geometry": d["geometry"], "Fz_N": d["Fz_N"], "samples": d["samples"],
            "integral_of_samples_N": integral,
            # the relative error is meaningless where Fz is itself zero by symmetry (the toroid):
            # judge those by the absolute error against the device's force scale
            "integral_absolute_error_N": integral - d["Fz_N"],
            "integral_relative_error": (integral - d["Fz_N"]) / d["Fz_N"] if d["Fz_N"] else 0.0,
            **({"length_m": d["length_m"]} if d["geometry"] == "surface" else {}),
            **({"Fz_lorentz_N": d["Fz_lorentz_N"]} if "Fz_lorentz_N" in d else {}),
        }
    out["checks"] = [c for c in res.data["checks"]
                     if "Newton's third law" in c["name"] or "B.n" in c["name"]]
    out["mesh"] = {k: res.data["mesh"][k] for k in ("elements", "dofs", "order")}
    return out


def main() -> None:
    out_path = Path(sys.argv[1])
    order = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    # never under benchmarks/records: these are ~20 MB of meshes and field dumps, not a record
    wd = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(tempfile.mkdtemp(prefix="axf_force_"))
    print(f"solver working directories: {wd}", flush=True)
    rec: dict = {"case": "force_validation", "order": order,
                 "reference": {"isolated_loop_Fz_N": 0.0,
                               "sphere_dipole_Fz_N": dipole_force(),
                               "core_Fz_per_body_N": 0.0, "toroid_Fz_per_body_N": 0.0},
                 "runs": {}}

    r = loop_design(False, order).solve(workdir=wd / "loop", verbose=0)
    rec["runs"]["loop"] = summarize(r)
    hoop = abs(r.forces["coil"]["radial_force_per_radian_N"]) * 2 * math.pi
    rec["runs"]["loop"]["hoop_force_N"] = hoop
    rec["runs"]["loop"]["self_force_over_hoop"] = r.forces["coil"]["Fz_self_N"] / hoop
    rec["runs"]["loop"]["corrected_over_hoop"] = r.forces["coil"]["Fz_excluding_self_N"] / hoop

    r = loop_design(True, order).solve(workdir=wd / "loop_sphere", verbose=0)
    s = summarize(r)
    ref = dipole_force()
    s["sphere_relative_error_vs_dipole"] = (r.forces["superconductor:sphere"]["Fz_N"] - ref) / ref
    s["third_law_raw"] = r.forces["coil"]["Fz_N"] + r.forces["superconductor:sphere"]["Fz_N"]
    s["third_law_self_corrected"] = (r.forces["coil"]["Fz_excluding_self_N"]
                                     + r.forces["superconductor:sphere"]["Fz_N"])
    rec["runs"]["loop_sphere"] = s

    for name in ("core", "toroid"):
        rec["runs"][name] = summarize(af.template(name).solve(workdir=wd / name, verbose=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: {"closure": v["closure"].get("relative")} for k, v in rec["runs"].items()}, indent=1))


if __name__ == "__main__":
    main()
