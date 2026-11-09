"""Bit-identity check for designs without thin layers (docs/validation/thin_magnetic_layer.md).

Solve a fixed set of designs that have no thin layer and dump their results, so that two builds
(e.g. main and a branch) can be compared exactly. Multithreaded assembly uses an OpenMP dynamic
schedule and is not bit-reproducible even on one build, so run with OMP_NUM_THREADS=1.

Usage:
    export OMP_NUM_THREADS=1
    PYTHONPATH=<build A>/python python benchmarks/thin_layer_bitid.py dump A.json DIR_A
    PYTHONPATH=<build B>/python python benchmarks/thin_layer_bitid.py dump B.json DIR_B
    python benchmarks/thin_layer_bitid.py compare A.json B.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# result keys that legitimately differ between runs: timings and file paths
VOLATILE_EXACT = {"info", "output", "path", "workdir", "timing"}
COARSE = {"order": 2, "size": "max(R, H)/10", "surface_size": "max(R, H)/30"}


def designs() -> dict:
    import axionfem as af

    out = {}
    for name in ("toroid", "solenoid"):
        d = af.template(name)
        d.doc["mesh"] = dict(COARSE)
        out[name] = d.doc
    core = af.template("core", iron_t=0.05)
    core.doc["mesh"] = {"order": 2, "size": 0.25, "surface_size": 0.05}
    out["core_iron"] = core.doc
    out["sheet_circuit_linear"] = {
        "schema": "axionfem.design/1", "name": "sheet",
        "materials": {"lin": {"kind": "linear", "mu_r": 50}},
        "regions": [
            {"name": "coil", "shape": {"rect": {"r": [0.5, 0.55], "z": [-0.2, 0.2]}},
             "material": "winding"},
            {"name": "lin", "shape": {"rect": {"r": [0.1, 0.2], "z": [-0.1, 0.1]}},
             "material": "lin"},
            {"name": "loop", "shape": {"disk": {"center": [0.3, 0.3], "radius": 0.02}},
             "material": "nb"}],
        "curves": [{"name": "sh", "material": "superconductor",
                    "path": {"arc": {"center": [0, 0], "radius": 0.8, "angles": [-1.2, 1.2]}}}],
        "windings": [{"name": "w", "region": "coil", "type": "azimuthal", "turns": 10,
                      "circuit": "c"}],
        "circuits": [{"name": "c", "type": "persistent", "flux_linkage": 0.01}],
        "pickup": {"type": "azimuthal", "conductor": "loop"},
        "mesh": {"order": 2, "size": 0.1, "surface_size": 0.02}}
    out["background_sc"] = {
        "schema": "axionfem.design/1", "name": "bgsc",
        "regions": [{"name": "ball", "shape": {"disk": {"center": [0.4, 0], "radius": 0.1}},
                     "material": "nb"}],
        "background": {"type": "uniform", "Bz": 0.5},
        "mesh": {"order": 2, "size": 0.05, "surface_size": 0.02}}
    out["sens_bg_pickup"] = {  # adjoint with a background
        "schema": "axionfem.design/1", "name": "s", "parameters": {"rc": 0.5},
        "regions": [
            {"name": "coil", "material": "winding",
             "shape": {"rect": {"r": ["rc", "rc + 0.05"], "z": [-0.2, 0.2]}}},
            {"name": "pick", "material": "nb",
             "shape": {"disk": {"center": [0.3, 0.0], "radius": 0.02}}},
            {"name": "sc", "material": "nb",
             "shape": {"disk": {"center": [0.7, 0.3], "radius": 0.03}}}],
        "windings": [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
        "pickup": {"type": "azimuthal", "conductor": "pick"},
        "background": {"type": "uniform", "Bz": 0.1},
        "mesh": {"order": 2, "size": 0.2, "surface_size": 0.02, "domain_radius": 2.0}}
    return out


def volatile(key: str) -> bool:
    return key in VOLATILE_EXACT or key.endswith("_s") or "second" in key


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not volatile(k)}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def dump(out_path: str, work: str) -> None:
    import axionfem as af

    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1: multithreaded assembly is not bit-reproducible")
    out = {}
    for name, doc in designs().items():
        res = af.Design(doc).solve(workdir=str(Path(work) / name), verbose=0)
        out[name] = strip(res.data)
        if name == "solenoid":
            sens = res.sensitivities(params=["R"], objectives=["Psi", "L_p", "W_mag"])
            out[name + "_sens"] = strip(sens)
        if name == "sens_bg_pickup":
            sens = res.sensitivities(params=["rc"], objectives=["L_p", "W_mag"])
            out[name + "_sens"] = strip(sens)
    Path(out_path).write_text(json.dumps(out, indent=0, sort_keys=True, default=str))
    print("wrote", out_path, list(out))


def compare(a: str, b: str) -> int:
    ja, jb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    if ja == jb:
        print(f"IDENTICAL: {len(ja)} entries")
        return 0
    for k in sorted(set(ja) | set(jb)):
        if ja.get(k) != jb.get(k):
            print("DIFFERENT:", k)
    return 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(__doc__)
