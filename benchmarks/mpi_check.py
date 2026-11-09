"""MPI consistency check of the 3D engine: the same designs on 1 and N ranks must agree to solver tolerance.

  python benchmarks/mpi_check.py write  runs/mpi        # write the designs
  (then run `srun -n K build/axf solve` for K = 1, 4, 8, or the site's MPI launcher)
  python benchmarks/mpi_check.py compare runs/mpi 1 4 8  # compare and write runs/mpi/report.json

benchmarks/records/mpi_check.json is the report.json of a run on 1, 4 and 8 ranks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import axionfem as af  # noqa: E402
from axionfem.convert import to_3d  # noqa: E402


def _ring(r0, r1, z0, z1, **kw):
    return {"revolve": {"profile": {"rect": {"r": [r0, r1], "z": [z0, z1]}}, **kw}}


def designs() -> dict[str, dict]:
    out = {}
    sol = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05)
    out["solenoid_port"] = to_3d(sol, mesh={"size": 0.4, "surface_size": 0.13, "order": 1}).doc
    tor = af.template("toroid", R=1.0, H=1.0, tw=0.06, g=0.04, ts=0.05, f=0.06)
    out["toroid_azimuthal"] = to_3d(tor, mesh={"size": 0.45, "surface_size": 0.18, "order": 1}).doc
    out["toroid_azimuthal_nd2"] = to_3d(tor, mesh={"size": 0.45, "surface_size": 0.18, "order": 2}).doc
    fe = {"schema": "axionfem.design/1", "name": "fe",
          "regions": [{"name": "coil", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.2, 0.2]}}, "material": "winding"},
                      {"name": "core", "shape": {"rect": {"r": [0.0, 0.4], "z": [-0.3, 0.3]}}, "material": "iron"}],
          "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 3e6}]}
    out["iron_newton"] = to_3d(af.Design(fe), mesh={"size": 0.3, "surface_size": 0.15, "order": 1}).doc
    circ = {"schema": "axionfem.design/1", "name": "circ",
            "regions": [{"name": "sol", "shape": {"rect": {"r": [1.0, 1.05], "z": [-1, 1]}}, "material": "winding"},
                        {"name": "pc", "shape": {"rect": {"r": [0.5, 0.55], "z": [0.3, 0.4]}}, "material": "winding",
                         "mesh_size": 0.05}],
            "windings": [{"name": "sol", "region": "sol", "type": "azimuthal", "ampere_turns": 1e5},
                         {"name": "pc", "region": "pc", "type": "azimuthal", "turns": 100, "circuit": "loop"}],
            "circuits": [{"name": "loop", "type": "persistent", "flux_linkage": 0}]}
    out["persistent_circuit"] = to_3d(af.Design(circ), mesh={"size": 0.4, "surface_size": 0.15, "order": 1}).doc
    return out


KEYS = [("coupling", "Psi_SI_H_T_m2"), ("coupling", "L_p_H"), ("coupling", "W_mag_J"), ("coupling", "F_SI_T4_H2_m8"),
        ("fields", "B_max_T"), ("static", "energy_J"), ("geometry", "iron_mass_kg"), ("coupling", "device_radius_m")]


def _flat(r: dict) -> dict[str, float]:
    v = {}
    for a, b in KEYS:
        if a in r and b in r[a]:
            v[f"{a}.{b}"] = r[a][b]
    for k, x in r.get("fields", {}).get("superconductor_surface_T", {}).items():
        v[f"sc_surface.{k}"] = x
    for k, x in r.get("fields", {}).get("by_region_T", {}).items():
        v[f"by_region.{k}"] = x
    for kind in ("windings", "superconductors"):
        for k, x in r.get("forces", {}).get(kind, {}).items():
            for i, f in enumerate(x["F_N"]):
                v[f"force.{kind}.{k}.{i}"] = f
    for c in r.get("static", {}).get("circuits", []):
        v[f"circuit.{c['name']}.current_A"] = c["current_A"]
    p = r.get("pickup") or {}
    if "L_p_from_flux_H" in p:
        v["pickup.L_p_from_flux_H"] = p["L_p_from_flux_H"]
    v["mesh.dofs"] = r["mesh"]["dofs"]
    return v


def compare(base: Path, ranks: list[int]) -> dict:
    rep = {}
    for name in designs():
        res = {}
        for k in ranks:
            f = base / name / f"np{k}" / "results.json"
            if f.exists():
                res[k] = json.loads(f.read_text())
        if not res or ranks[0] not in res:
            rep[name] = {"error": "missing results"}
            continue
        ref = _flat(res[ranks[0]])
        entry = {"timing_s": {k: r["timing"]["total_s"] for k, r in res.items()},
                 "pcg_iterations": {k: r["solver"]["pcg_iterations_total"] for k, r in res.items()},
                 "warnings": {k: r["warnings"] for k, r in res.items()}, "max_rel_diff": {}, "worst": {}}
        for k, r in res.items():
            cur = _flat(r)
            scale = max((abs(x) for key, x in ref.items() if key.startswith("force.")), default=1.0)
            worst, wk = 0.0, None
            for key, x in ref.items():
                if key not in cur:
                    worst, wk = float("inf"), key + " (missing)"
                    break
                den = scale if key.startswith("force.") else max(abs(x), 1e-300)
                d = abs(cur[key] - x) / den
                if d > worst:
                    worst, wk = d, key
            entry["max_rel_diff"][k] = worst
            entry["worst"][k] = wk
        rep[name] = entry
    return rep


def reload(base: Path, ranks: list[int]) -> dict:
    """Saved solutions load in one process; samples and scenes agree across rank counts."""
    import math

    out = {}
    pts = ([0.05, 0.3, 0.52, 0.7, 0.9], [0.0, 0.1, -0.2, 0.05, 0.0], [0.0, 0.1, 0.05, -0.3, 0.6])
    for name in designs():
        ref, rep = None, {}
        for k in ranks:
            p = base / name / f"np{k}" / "fields"
            if not (p / "state.json").exists():
                rep[k] = "missing"
                continue
            r = af.load_solution(p)
            s = r.sample(*pts)
            sc = r.scene()
            vals = [x for key in ("Bx", "By", "Bz") for x in s[key] if not math.isnan(x)]
            if ref is None:
                ref = vals
            scale = max(abs(x) for x in ref) or 1.0
            rep[k] = {"max_sample_diff_rel": max(abs(a - b) for a, b in zip(vals, ref)) / scale,
                      "scene_regions": len(sc.get("regions", [])), "Psi": r.Psi}
        out[name] = rep
        print(name, json.dumps(rep))
    return out


if __name__ == "__main__":
    cmd, base = sys.argv[1], Path(sys.argv[2])
    if cmd == "write":
        for name, doc in designs().items():
            (base / name).mkdir(parents=True, exist_ok=True)
            (base / name / "design.json").write_text(json.dumps(doc, indent=1))
        print(" ".join(designs()))
    elif cmd == "reload":
        reload(base, [int(x) for x in sys.argv[3:]])
    else:
        rep = compare(base, [int(x) for x in sys.argv[3:]])
        (base / "report.json").write_text(json.dumps(rep, indent=1))
        for name, e in rep.items():
            print(name, json.dumps({k: e[k] for k in ("max_rel_diff", "worst", "timing_s")} if "error" not in e else e))
