"""Second cross-code reference for the 3D Stokes cavity (xcode_stokes_cavity_reg_3d): NGSolve, regularized-lid Stokes
cavity in the unit cube.

Written independently of the axionFEM code.  It checks the agreement between the two reference codes: the primary
values come from DOLFINx (fenicsx_cavity3d_reference.py); this script solves the same fixed discrete problem with an
unrelated code.

Discrete problem (X1):
  * mesh: f1c_stokes_check.cube_tet(N), the octant Kuhn mesh, N = 8, 16 (fixed builder, imported read-only);
  * Taylor-Hood P2/P1, both viscous forms, mu = 1, gamma = 0, f = 0, mean-zero pressure by a scalar multiplier;
  * Dirichlet: u = (256 x^2 (1-x)^2 y^2 (1-y)^2, 0, 0) on z = 1, 0 elsewhere, by P2 LAGRANGE NODAL interpolation:
    Set(dual=True) gives the vertex values, then each lid edge's bubble coefficient is corrected so that the midpoint
    value equals the data (max nodal defect stored);
  * sparse LU (UMFPACK); integrals with order-8 rules (exact for the degree-4 integrands).

    python -B tools/fluids_mms/xcode/ngsolve_cavity3d_reference.py OUT.json [NMAX]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import ngsolve as ngs
from netgen.meshing import Mesh as NGMesh, MeshPoint, Element3D, Element2D, FaceDescriptor, Pnt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import f1c_stokes_check as C  # noqa: E402  (fixed mesh builder only)

ngs.SetNumThreads(1)
ORD = 8


def lid(x, y):
    return 256 * x ** 2 * (1 - x) ** 2 * y ** 2 * (1 - y) ** 2


def ng_mesh(fm):
    V, T = fm.V, fm.E.copy()
    P = V[T]
    d = np.linalg.det(np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1))
    T[d < 0] = T[d < 0][:, [0, 2, 1, 3]]
    m = NGMesh(dim=3)
    pids = [m.Add(MeshPoint(Pnt(*map(float, v)))) for v in V]
    m.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))
    m.Add(FaceDescriptor(surfnr=2, domin=1, domout=0, bc=2))
    for t in T:
        m.Add(Element3D(1, [pids[int(k)] for k in t]))
    for f in fm.boundary_faces():
        pts = V[list(f)]
        idx = 1 if np.all(np.abs(pts[:, 2] - 1) < 1e-12) else 2
        m.Add(Element2D(idx, [pids[int(k)] for k in f]))
    m.SetBCName(0, "lid")
    m.SetBCName(1, "wall")
    return ngs.Mesh(m)


def solve(N, form):
    fm = C.cube_tet(N)
    mesh = ng_mesh(fm)
    Uc = ngs.H1(mesh, order=2, dirichlet="lid|wall")
    Q = ngs.H1(mesh, order=1)
    X = ngs.FESpace([Uc, Uc, Uc, Q, ngs.NumberSpace(mesh)])
    (ux, uy, uz, p, lam), (vx, vy, vz, q, mt) = X.TnT()
    gu = ngs.CF(tuple(ngs.grad(c)[j] for c in (ux, uy, uz) for j in range(3)), dims=(3, 3))
    gv = ngs.CF(tuple(ngs.grad(c)[j] for c in (vx, vy, vz) for j in range(3)), dims=(3, 3))
    divu = ngs.grad(ux)[0] + ngs.grad(uy)[1] + ngs.grad(uz)[2]
    divv = ngs.grad(vx)[0] + ngs.grad(vy)[1] + ngs.grad(vz)[2]
    visc = ngs.InnerProduct(gu, gv) if form == "lap" else \
        2 * ngs.InnerProduct(0.5 * (gu + gu.trans), 0.5 * (gv + gv.trans))
    a = ngs.BilinearForm(X)
    a += (visc - p * divv - q * divu + lam * q + mt * p) * ngs.dx
    a.Assemble()
    gf = ngs.GridFunction(X)
    gx = gf.components[0]
    gx.Set(ngs.CF(lid(ngs.x, ngs.y)), definedon=mesh.Boundaries("lid"), dual=True)
    lid_edges = set()
    for el in mesh.Elements(ngs.BND):
        if el.mat == "lid":
            lid_edges.update(el.edges)
    probe = ngs.GridFunction(Uc)
    phi_mid = None
    for e in lid_edges:
        vts = mesh[e].vertices
        vd = set(sum((list(Uc.GetDofNrs(v)) for v in vts), []))
        edge_dofs = [d for d in Uc.GetDofNrs(e) if d >= 0 and d not in vd]
        pa, pb = (np.array(mesh[v].point) for v in vts)
        pm = 0.5 * (pa + pb)
        if phi_mid is None:
            probe.vec[:] = 0
            probe.vec[edge_dofs[0]] = 1.0
            phi_mid = probe(mesh(*pm))
        cur = gx(mesh(*pm))
        gx.vec[edge_dofs[0]] += (lid(pm[0], pm[1]) - cur) / phi_mid
    defects = []
    for e in lid_edges:
        vts = mesh[e].vertices
        pa, pb = (np.array(mesh[v].point) for v in vts)
        for pt in (pa, pb, 0.5 * (pa + pb)):
            defects.append(abs(gx(mesh(*pt)) - lid(pt[0], pt[1])))
    free = X.FreeDofs()
    res = gf.vec.CreateVector()
    res.data = -a.mat * gf.vec
    r0 = np.linalg.norm(np.array(res.FV().NumPy())[np.array(free)])
    gf.vec.data += a.mat.Inverse(free, inverse="umfpack") * res
    r2 = gf.vec.CreateVector()
    r2.data = -a.mat * gf.vec
    relres = float(np.linalg.norm(np.array(r2.FV().NumPy())[np.array(free)]) / max(r0, 1e-300))
    uh = [gf.components[i] for i in range(3)]
    Gh = ngs.CF(tuple(ngs.grad(c)[j] for c in uh for j in range(3)), dims=(3, 3))
    ekin = 0.5 * ngs.Integrate(sum(c * c for c in uh), mesh, order=ORD)
    if form == "lap":
        diss = ngs.Integrate(ngs.InnerProduct(Gh, Gh), mesh, order=ORD)
    else:
        Sh = 0.5 * (Gh + Gh.trans)
        diss = 2 * ngs.Integrate(ngs.InnerProduct(Sh, Sh), mesh, order=ORD)
    return {"Ekin": float(ekin), "Diss": float(diss), "relres": relres, "nodal_defect": float(max(defects)),
            "ndof": int(sum(free))}


def main():
    out_path = Path(sys.argv[1])
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    t0 = time.time()
    q, diag = {}, {}
    for form in ("lap", "sym"):
        for N in (8, 16):
            if N > nmax:
                continue
            r = solve(N, form)
            q[f"{form}_Ekin_N{N}"] = r["Ekin"]
            q[f"{form}_Diss_N{N}"] = r["Diss"]
            diag[f"{form}_N{N}"] = {k: r[k] for k in ("relres", "nodal_defect", "ndof")}
            print(f"{form} N={N}: Ekin {r['Ekin']!r} Diss {r['Diss']!r} relres {r['relres']:.1e} "
                  f"nodal defect {r['nodal_defect']:.1e}", flush=True)
    out = {"format": "axionfem.fluids.xcode_output/1", "code": "NGSolve", "version": ngs.__version__,
           "case": "xcode_stokes_cavity_reg_3d", "quantities": q, "diagnostics": diag}
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
