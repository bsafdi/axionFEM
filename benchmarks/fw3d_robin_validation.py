"""Analytic validation of the 3D full-wave Robin boundaries (impedance, scattering) and PML.  beta
(COMSOL import of IBC/SBC/PML, docs/comsol_import.md).

A parallel-plate TEM line (plates y = 0 and y = h: PEC or copper impedance walls; PMC side walls z = 0, W) of
length l, fed by a surface lumped port on x = 0 (the importer's surface port), terminated at x = l by
  pec         Z = -i Z0 tan(k l),
  scattering  Z = Z0 = eta h / W (plane-wave scattering condition: exact for the TEM wave),
  pml         a Cartesian PML x in [l, l + d] backed by PEC: Z ~ Z0;
and with impedance plates and a scattering end: attenuation alpha = R_s / (eta h) from the power reaching the end;
a thin resistive sheet (sheet group on interior faces) across the matched line: the shunt-admittance closed form;
the impedance case again on 2 MPI ranks (--mpi).
Structured Kuhn tetrahedra (straight geometry is exact here), written as an imported mesh (gmsh MSH 2.2).

    python benchmarks/fw3d_robin_validation.py [--out benchmarks/records/fw3d_robin_validation.json]
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

C0 = 299792458.0
MU0 = 1.25663706212e-6
ETA0 = MU0 * C0
SIGMA_CU = 5.8e7
_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def box_mesh(xs, ys, zs):
    """Kuhn tetrahedra of the tensor grid xs x ys x zs; returns points, tets, boundary triangles with face labels
    (0 x=min, 1 x=max, 2 y=min, 3 y=max, 4 z=min, 5 z=max) and interior faces on the planes x = x_i (label 10 + i)."""
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    P = np.array([[x, y, z] for x in xs for y in ys for z in zs], float)
    T = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                T += [[c[q] for q in t] for t in _KUHN]
    T = np.array(T)
    F = np.sort(np.concatenate([T[:, [1, 2, 3]], T[:, [0, 2, 3]], T[:, [0, 1, 3]], T[:, [0, 1, 2]]]), axis=1)
    uf, cnt = np.unique(F, axis=0, return_counts=True)
    tris, lab = [], []
    lo, hi = P.min(axis=0), P.max(axis=0)
    for f, c in zip(uf, cnt):
        X = P[f]
        if c == 1:
            for ax in range(3):
                for side, val in ((0, lo[ax]), (1, hi[ax])):
                    if np.all(np.abs(X[:, ax] - val) < 1e-12):
                        tris.append(f)
                        lab.append(2 * ax + side)
        else:
            for i, x in enumerate(xs):
                if np.all(np.abs(X[:, 0] - x) < 1e-12):
                    tris.append(f)
                    lab.append(10 + i)
    return P, T, np.array(tris), np.array(lab)


def write_msh(path, P, T, tet_attr, tris, tri_attr):
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(P)}\n")
        for k, p in enumerate(P):
            f.write(f"{k + 1} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(tris) + len(T)}\n")
        k = 1
        for t, a in zip(tris, tri_attr):
            f.write(f"{k} 2 2 {a} {a} {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
            k += 1
        for t, a in zip(T, tet_attr):
            f.write(f"{k} 4 2 {a} {a} " + " ".join(str(v + 1) for v in t) + "\n")
            k += 1
        f.write("$EndElements\n")


def build(wd: Path, *, termination="scattering", plates="pec", l=0.3, h=0.02, W=0.04, nx=30, pml_cells=10,
          d_pml=0.1, order=2, sheet=None):
    wd.mkdir(parents=True, exist_ok=True)
    pml = termination == "pml"
    xs = list(np.linspace(0, l, nx + 1)) + (list(l + d_pml * np.arange(1, pml_cells + 1) / pml_cells) if pml else [])
    P, T, tris, lab = box_mesh(np.array(xs), np.linspace(0, h, 3), np.linspace(0, W, 3))
    cen = P[T].mean(axis=1)
    tet_attr = np.where(cen[:, 0] > l, 3, 2)
    attr = {0: 10001, 1: 10002, 2: 10003, 3: 10004, 4: 10005, 5: 10006}
    if sheet:  # a thin resistive layer across the line on the grid plane x = xs[i_sheet] (interior faces)
        attr[10 + sheet["i"]] = 10008
    keep = (lab < 10) | (bool(sheet) & (lab == 10 + (sheet or {}).get("i", -1)))
    tri_attr = np.array([attr[int(b)] for b in lab[keep]])
    tris = tris[keep]
    if pml:  # plates inside the PML: separate PEC group
        c = P[tris].mean(axis=1)
        tri_attr = np.where((c[:, 0] > l) & np.isin(tri_attr, [10003, 10004]), 10007, tri_attr)
    ptype = {"type": "pec"} if plates == "pec" else {"type": "impedance", "material": "copper_ibc"}
    groups = [{"attr": 10001, "name": "lport1", "type": "port"},
              {"attr": 10003, "name": "plate_bottom", **ptype}, {"attr": 10004, "name": "plate_top", **ptype},
              {"attr": 10005, "name": "side_1", "type": "pmc"}, {"attr": 10006, "name": "side_2", "type": "pmc"}]
    groups.append({"attr": 10002, "name": "end", "type": "scattering", "wave": "plane"} if termination == "scattering"
                  else {"attr": 10002, "name": "end", "type": "pec"})
    if pml:
        groups.append({"attr": 10007, "name": "plates_pml", "type": "pec"})
    materials = {"copper_ibc": {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "sigma_S_m": SIGMA_CU}}
    if sheet:
        groups.append({"attr": 10008, "name": "film", "type": "sheet", "material": "film", "thickness_m": sheet["d"]})
        materials["film"] = {"kind": "linear", "eps_r": 1.0, "mu_r": 1.0, "sigma_S_m": sheet["sigma"]}
    mesh = wd / "plates3d.msh"
    write_msh(mesh, P, T, tet_attr, tris, tri_attr)
    regions = [{"name": "field", "material": "vacuum", "solid": {"imported": {}}}]
    if pml:
        regions.append({"name": "pml", "material": "vacuum", "solid": {"imported": {}}})
    info = wd / "plates3d_mesh_info.json"
    info.write_text(json.dumps({"ok": True, "boundary_groups": groups, "shell_attr": -1,
                                "groups": [{"dim": 3, "attr": 2 + k, "name": r["name"]} for k, r in enumerate(regions)],
                                "domain_center": [l / 2, h / 2, W / 2], "device_radius": l, "domain_radius": l}))
    fw = {"boundary": "pec", "rf_only": True}
    if pml:
        fw["pml"] = [{"regions": ["pml"], "type": "cartesian"}]
    d = {"schema": "axionfem.design/1", "name": "plates3d", "symmetry": "3d", "regions": regions,
         "materials": materials,
         "pickup": {"type": "port", "port_boundary": "lport1", "toward": [0, 1, 0]},
         "fullwave": fw, "mesh": {"order": order, "file": str(mesh), "info_file": str(info)}}
    return d, {"l": l, "h": h, "W": W, "x_sheet": xs[sheet["i"]] if sheet else None}


def run(out=None, mpi_ranks=1):
    import axionfem as af

    rec = {"schema": "axionfem.fw3d_robin_validation/1", "beta": True, "cases": []}
    tmp = Path(tempfile.mkdtemp(prefix="fw3d_robin_"))
    f = 3e8
    k = 2 * math.pi * f / C0
    for term, plates in (("pec", "pec"), ("scattering", "pec"), ("pml", "pec"), ("scattering", "impedance")):
        d, m = build(tmp / f"{term}_{plates}", termination=term, plates=plates)
        d["fullwave"]["frequencies"] = [f]
        res = af.Design(d).solve(workdir=str(tmp / f"{term}_{plates}"), verbose=0)
        row = res.data["fullwave"]["sweep"][0]
        Z = row["Z_re_ohm"] + 1j * row["Z_im_ohm"]
        Z0 = ETA0 * m["h"] / m["W"]
        case = {"termination": term, "plates": plates, "f_Hz": f, "Z": [Z.real, Z.imag], "Z0": Z0,
                "gmres_relative_residual": row["solver"]["relative_residual"]}
        if plates == "pec":
            Zx = -1j * Z0 * math.tan(k * m["l"]) if term == "pec" else Z0
            case.update(Z_exact=[Zx.real, Zx.imag], rel_err=abs(Z - Zx) / abs(Zx))
        else:
            Pin = 0.5 * Z.real
            Pend = row["boundary_power_W_per_A2"]["end"]
            alpha = -math.log(Pend / Pin) / (2 * m["l"])
            ax = math.sqrt(math.pi * f * MU0 / SIGMA_CU) / (ETA0 * m["h"])
            case.update(alpha_fem=alpha, alpha_exact=ax, rel_err=abs(alpha / ax - 1),
                        power_balance_rel=row.get("power_balance_rel"))
        rec["cases"].append(case)
        print(json.dumps(case))
    # thin resistive sheet across the line (sheet group on interior faces), matched end: shunt admittance
    # G = Y_s W / h, Y_s = -(2i/eta_m) tan(k_m d / 2), then the line from the sheet to the port
    sigma, dfilm = 1e4, 1e-6
    d, m = build(tmp / "sheet", termination="scattering", plates="pec", sheet={"i": 15, "sigma": sigma, "d": dfilm})
    d["fullwave"]["frequencies"] = [f]
    row = af.Design(d).solve(workdir=str(tmp / "sheet"), verbose=0).data["fullwave"]["sweep"][0]
    Z = row["Z_re_ohm"] + 1j * row["Z_im_ohm"]
    Z0 = ETA0 * m["h"] / m["W"]
    ec = 1 / (MU0 * C0 ** 2) + 1j * sigma / (2 * math.pi * f)
    km, etam = 2 * math.pi * f * np.sqrt(MU0 * ec), np.sqrt(MU0 / ec)
    G = -2j / etam * np.tan(km * dfilm / 2) * m["W"] / m["h"]
    ZL = Z0 / (1 + G * Z0)
    t = math.tan(k * m["x_sheet"])
    Zx = Z0 * (ZL - 1j * Z0 * t) / (Z0 - 1j * ZL * t)
    case = {"termination": "scattering", "plates": "pec", "sheet_x_m": m["x_sheet"], "f_Hz": f, "Z": [Z.real, Z.imag],
            "Z_exact": [Zx.real, Zx.imag], "rel_err": abs(Z - Zx) / abs(Zx), "power_balance_rel": row.get("power_balance_rel")}
    rec["cases"].append(case)
    print(json.dumps(case))
    # the impedance-plate case on 2 MPI ranks (Robin assembly and boundary power across a partition)
    if mpi_ranks > 1:
        d, m = build(tmp / "imp_mpi", termination="scattering", plates="impedance")
        d["fullwave"]["frequencies"] = [f]
        row = af.Design(d).solve(workdir=str(tmp / "imp_mpi"), verbose=0, mpi_ranks=mpi_ranks).data["fullwave"]["sweep"][0]
        Z = row["Z_re_ohm"] + 1j * row["Z_im_ohm"]
        ref = next(c for c in rec["cases"] if c.get("plates") == "impedance")
        case = {"case": "impedance_mpi", "ranks": mpi_ranks, "Z": [Z.real, Z.imag], "Z_serial": ref["Z"],
                "rel_diff_vs_serial": abs(Z - complex(*ref["Z"])) / abs(Z),
                "power_balance_rel": row.get("power_balance_rel")}
        rec["cases"].append(case)
        print(json.dumps(case))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "records" / "fw3d_robin_validation.json"))
    ap.add_argument("--mpi", type=int, default=2, help="also run the impedance case on this many ranks (1: skip)")
    a = ap.parse_args()
    run(a.out, a.mpi)
