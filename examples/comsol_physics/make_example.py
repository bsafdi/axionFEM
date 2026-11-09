#!/usr/bin/env python
"""Worked example of the COMSOL physics export: the Stage 1 two-port copper frame.

Design: benchmarks/designs/stage1_twoport.json (axionFEM already solved it; docs/validation/multiport_stage1.md).
A copper (RF PEC) square frame of 0.6 m, read out across the box gap ``gap0`` (port 1; ``gap1`` is a second,
open port), inside a PEC sphere of radius 0.7 m, with the uniform prescribed source B0 = (1, 0.2, 0.3) T, at
1 and 12 MHz.

This script

1. exports the COMSOL models with ``--physics fullwave``: the parametric ``Stage1TwoPort.java`` and the
   mesh-import route ``stage1_twoport.mphtxt`` + ``Import_stage1_twoport.java``;
2. with ``--solve``, solves the design with axionFEM at mesh order 1 (the recorded mesh) and order 2, and writes
   ``axionfem_reference.json``: Z, L, Psi and the open-circuit voltage in axionFEM's convention, and the values
   COMSOL should report (complex conjugates; port 1 loaded by Z_ref = 50 ohm in the 'volt' run).

Run (repository root, package built; the solve takes a few minutes, run it as a batch job):

    PYTHONPATH=python AXF_MESHER=build/axf-mesher python examples/comsol_physics/make_example.py OUTDIR [--solve]

Then compare with COMSOL's output: ``python examples/comsol_physics/make_example.py OUTDIR --compare DIR`` where
DIR holds Stage1TwoPort_impedance.txt / Stage1TwoPort_voltage.txt (or the Import_* ones).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parents[1] / "benchmarks" / "designs" / "stage1_twoport.json"
Z_REF = 50.0


def export(out: Path) -> dict:
    from axionfem import comsol

    doc = json.loads(DESIGN.read_text())
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(DESIGN, out / "stage1_twoport.json")
    j = comsol.export_comsol(doc, out / "Stage1TwoPort.java", format="java", physics="fullwave")
    m = comsol.export_comsol(doc, out / "stage1_twoport.mphtxt", format="mphtxt", physics="fullwave",
                             workdir=out / "stage1_twoport_mesh")
    return {"java": j, "mphtxt": m}


def solve(out: Path) -> dict:
    import axionfem as af

    doc = json.loads(DESIGN.read_text())
    ref = {"design": str(DESIGN.relative_to(HERE.parents[1])), "Z_ref_ohm": Z_REF,
           "conventions": {
               "axionFEM": "exp(-i omega t); Z = R - i omega L; open-circuit V = -i omega Psi (kappa = 1 A/m^2/T)",
               "COMSOL": "exp(+j omega t): conjugate values; Zport_1 = conj(Z); open-circuit V = conj(V) up to the "
                         "port's polarity; 'volt' run reports V_loaded = V_oc Z_ref / (Z_ref + Zport_1)"},
           "runs": []}
    for order in (1, 2):
        d = af.Design(json.loads(json.dumps(doc)))
        d.doc["mesh"]["order"] = order
        r = d.solve(out / f"axionfem_order{order}", verbose=0)
        rows = []
        for s in r.data["fullwave"]["sweep"]:
            f = s["f_Hz"]
            w = 2 * math.pi * f
            Z = complex(s["Z_matrix_re_ohm"][0][0], s["Z_matrix_im_ohm"][0][0])
            Psi = complex(s["Psi_vector"]["re"][0], s["Psi_vector"]["im"][0])
            V = -1j * w * Psi
            Zc, Vc = Z.conjugate(), V.conjugate()
            Vl = Vc * Z_REF / (Z_REF + Zc)
            row = {"f_MHz": f / 1e6, "axionFEM": {"Z_ohm": [Z.real, Z.imag], "L_H": s["L_H"], "Psi_Wb_per_T": [Psi.real, Psi.imag],
                                                  "V_open_V": [V.real, V.imag]},
                   "COMSOL_expected": {"Zport_1": [Zc.real, Zc.imag], "abs_V_open_V": abs(Vc),
                                       "abs_Vport_1_loaded_50ohm_V": abs(Vl)}}
            if "direct_source_voltage" in s:
                row["axionFEM"]["V_open_direct_solve_V"] = [s["direct_source_voltage"]["re"][0], s["direct_source_voltage"]["im"][0]]
            rows.append(row)
        mesh = {k: v for k, v in (r.data.get("mesh") or {}).items() if isinstance(v, (int, float, str))}
        ref["runs"].append({"mesh_order": order, "mesh": mesh, "rows": rows, "warnings": r.data.get("warnings", [])})
    (HERE / "axionfem_reference.json").write_text(json.dumps(ref, indent=1) + "\n")
    return ref


def compare(ref_path: Path, comsol_dir: Path) -> None:
    from axionfem.comsol_physics import open_circuit_voltage, read_eval_table

    ref = json.loads(ref_path.read_text())
    best = ref["runs"][-1]["rows"]
    for stem in ("Stage1TwoPort", "Import_stage1_twoport"):
        zi, vi = comsol_dir / f"{stem}_impedance.txt", comsol_dir / f"{stem}_voltage.txt"
        if not zi.is_file():
            continue
        Z = read_eval_table(zi)
        V = read_eval_table(vi) if vi.is_file() else None
        print(f"== {stem}")
        for k, row in enumerate(best):
            zc = Z[1, k] + 1j * Z[2, k]
            za = complex(*row["COMSOL_expected"]["Zport_1"])
            line = f"f = {Z[0, k]:g} MHz: Zport_1 = {zc:.6g} (axionFEM conj: {za:.6g}, ratio Im {zc.imag / za.imag:.4f})"
            if V is not None:
                voc = open_circuit_voltage(V[1, k] + 1j * V[2, k], zc, ref["Z_ref_ohm"])
                line += f"; |V_open| = {abs(voc):.6g} V (axionFEM {row['COMSOL_expected']['abs_V_open_V']:.6g} V)"
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--solve", action="store_true", help="also solve with axionFEM and write axionfem_reference.json")
    ap.add_argument("--compare", type=Path, help="directory with COMSOL's *_impedance.txt / *_voltage.txt")
    a = ap.parse_args()
    if a.compare:
        compare(HERE / "axionfem_reference.json", a.compare)
        return
    res = export(a.out)
    print(json.dumps({k: {"file": v.get("file"), "import_java": v.get("import_java"), "physics": v["physics"]}
                      for k, v in res.items()}, indent=1))
    if a.solve:
        print(json.dumps(solve(a.out), indent=1))


if __name__ == "__main__":
    np.set_printoptions(precision=6)
    main()
