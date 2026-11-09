"""Out-of-band cross-code reference for F1a: NGSolve, regularized-lid Stokes cavity.

Produced independently of the axionFEM code, before any axionFEM run.

Discrete problem = the fixed one of selfconv_stokes_cavity_reg_2d / xcode_stokes_cavity_reg_2d:
  * mesh: f1a_stokes_check.structured_mesh(N), N = 16, 32, 64, 128 (fixed builder, imported read-only);
  * Taylor-Hood P2/P1, both viscous forms, mu = 1, gamma = 0, mean-zero pressure by a scalar multiplier;
  * Dirichlet data: u = (16 x^2 (1-x)^2, 0) on y = 1, 0 elsewhere, by P2 LAGRANGE NODAL INTERPOLATION (exact at the vertices and edge midpoints).  NGSolve's hierarchical P2 basis is set by `Set(dual=True)`
    (exact vertex values) and then each lid edge's bubble coefficient is corrected so that the midpoint value equals
    the data (verified: max nodal defect printed and stored);
  * sparse LU (UMFPACK); integrals with order-12 rules (exact for the degree-4 integrands).
Outputs (JSON): quantities {"<form>_Ekin_N<N>": 0.5 ||u_h||^2, "<form>_Diss_N<N>": a_form(u_h, u_h)} and diagnostics.

    python -B tools/fluids_mms/xcode/ngsolve_cavity_reference.py OUT.json [NMAX]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import ngsolve as ngs
from netgen.meshing import Mesh as NGMesh, MeshPoint, Element2D, Element1D, FaceDescriptor, Pnt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import f1a_stokes_check as C  # noqa: E402  (fixed mesh builder only)

ngs.SetNumThreads(1)
ORD = 12


def lid(xx):
    return 16 * xx ** 2 * (1 - xx) ** 2


def ng_mesh(V, T):
    V = np.asarray(V, float)
    T = np.asarray(T).copy()
    P = V[T]
    d = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
    T[d < 0] = T[d < 0][:, [0, 2, 1]]
    m = NGMesh(dim=2)
    pids = [m.Add(MeshPoint(Pnt(float(a), float(b), 0.0))) for a, b in V]
    m.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    for t in T:
        m.Add(Element2D(1, [pids[int(k)] for k in t]))
    cnt, orient = {}, {}
    for t in T:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(t[i]), int(t[j])
            key = (min(a, b), max(a, b))
            cnt[key] = cnt.get(key, 0) + 1
            orient[key] = (a, b)
    for key, c in cnt.items():
        if c == 1:
            a, b = orient[key]
            idx = 1 if (abs(V[a][1] - 1) < 1e-12 and abs(V[b][1] - 1) < 1e-12) else 2
            m.Add(Element1D([pids[a], pids[b]], index=idx))
    m.SetBCName(0, "lid")
    m.SetBCName(1, "wall")
    return ngs.Mesh(m)


def solve(N, form):
    V, T = C.structured_mesh(N)
    mesh = ng_mesh(V, T)
    Uc = ngs.H1(mesh, order=2, dirichlet="lid|wall")
    Q = ngs.H1(mesh, order=1)
    X = ngs.FESpace([Uc, Uc, Q, ngs.NumberSpace(mesh)])
    (ux, uy, p, lam), (vx, vy, q, mt) = X.TnT()
    gu = ngs.CF((ngs.grad(ux)[0], ngs.grad(ux)[1], ngs.grad(uy)[0], ngs.grad(uy)[1]), dims=(2, 2))
    gv = ngs.CF((ngs.grad(vx)[0], ngs.grad(vx)[1], ngs.grad(vy)[0], ngs.grad(vy)[1]), dims=(2, 2))
    divu = ngs.grad(ux)[0] + ngs.grad(uy)[1]
    divv = ngs.grad(vx)[0] + ngs.grad(vy)[1]
    visc = ngs.InnerProduct(gu, gv) if form == "lap" else 2 * ngs.InnerProduct(0.5 * (gu + gu.trans), 0.5 * (gv + gv.trans))
    a = ngs.BilinearForm(X)
    a += (visc - p * divv - q * divu + lam * q + mt * p) * ngs.dx
    a.Assemble()
    gf = ngs.GridFunction(X)
    gx = gf.components[0]
    gx.Set(ngs.CF(16 * ngs.x ** 2 * (1 - ngs.x) ** 2), definedon=mesh.Boundaries("lid"), dual=True)
    # nodal correction of the lid edge bubbles: value at the edge midpoint := data
    lid_edges = []
    for el in mesh.Elements(ngs.BND):
        if el.mat == "lid":
            lid_edges.extend(el.edges)
    probe = ngs.GridFunction(Uc)
    phi_mid = None
    defects = []
    for e in lid_edges:
        dofs = [d for d in Uc.GetDofNrs(e) if d >= 0]
        vts = mesh[e].vertices
        pa, pb = (np.array(mesh[v].point) for v in vts)
        xm, ym = 0.5 * (pa + pb)
        edge_dofs = [d for d in dofs if d not in set(sum((list(Uc.GetDofNrs(v)) for v in vts), []))]
        if phi_mid is None:
            probe.vec[:] = 0
            probe.vec[edge_dofs[0]] = 1.0
            phi_mid = probe(mesh(xm, ym, 0.0))
        cur = gx(mesh(xm, ym, 0.0))
        gx.vec[edge_dofs[0]] += (lid(xm) - cur) / phi_mid
    for e in lid_edges:
        vts = mesh[e].vertices
        pa, pb = (np.array(mesh[v].point) for v in vts)
        for pt in (pa, pb, 0.5 * (pa + pb)):
            defects.append(abs(gx(mesh(pt[0], pt[1], 0.0)) - lid(pt[0])))
    free = X.FreeDofs()
    res = gf.vec.CreateVector()
    res.data = -a.mat * gf.vec
    r0 = np.linalg.norm(np.array(res.FV().NumPy())[np.array(free)])
    gf.vec.data += a.mat.Inverse(free, inverse="umfpack") * res
    r2 = gf.vec.CreateVector()
    r2.data = -a.mat * gf.vec
    relres = float(np.linalg.norm(np.array(r2.FV().NumPy())[np.array(free)]) / max(r0, 1e-300))
    uxh, uyh = gf.components[0], gf.components[1]
    Gh = ngs.CF((ngs.grad(uxh)[0], ngs.grad(uxh)[1], ngs.grad(uyh)[0], ngs.grad(uyh)[1]), dims=(2, 2))
    ekin = 0.5 * ngs.Integrate(uxh ** 2 + uyh ** 2, mesh, order=ORD)
    if form == "lap":
        diss = ngs.Integrate(ngs.InnerProduct(Gh, Gh), mesh, order=ORD)
    else:
        Sh = 0.5 * (Gh + Gh.trans)
        diss = 2 * ngs.Integrate(ngs.InnerProduct(Sh, Sh), mesh, order=ORD)
    return {"Ekin": float(ekin), "Diss": float(diss), "relres": relres, "nodal_defect": float(max(defects)),
            "ndof": int(sum(free))}


def main():
    out_path = Path(sys.argv[1])
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    t0 = time.time()
    q, diag = {}, {}
    for form in ("lap", "sym"):
        for N in (16, 32, 64, 128):
            if N > nmax:
                continue
            r = solve(N, form)
            q[f"{form}_Ekin_N{N}"] = r["Ekin"]
            q[f"{form}_Diss_N{N}"] = r["Diss"]
            diag[f"{form}_N{N}"] = {k: r[k] for k in ("relres", "nodal_defect", "ndof")}
            print(f"{form} N={N}: Ekin {r['Ekin']!r} Diss {r['Diss']!r} relres {r['relres']:.1e} "
                  f"nodal defect {r['nodal_defect']:.1e}", flush=True)
    out = {"format": "axionfem.fluids.xcode_output/1", "code": "NGSolve", "version": ngs.__version__,
           "case": "xcode_stokes_cavity_reg_2d", "quantities": q, "diagnostics": diag,
           "seconds": None}
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
