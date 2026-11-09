"""Out-of-band cross-code reference for F1b: NGSolve, axisymmetric Stokes cavity with a swirling lid.

Produced independently of the axionFEM code, before any axionFEM run.  NGSolve has no axisymmetric module: the r-weighted weak form of
tools/fluids_mms/axi_stokes.py (docstring) is written here directly with r = x.

Discrete problem = the fixed one of xcode_stokes_axi_cavity_swirl:
  * mesh: f1b_stokes_check.square_mesh(N) on (r, z) in [0,1]^2, N = 16, 32, 64, 128 (fixed builder, read-only);
  * Taylor-Hood P2/P1 for (u_r, u_z, u_theta; p), both viscous forms, mu = 1, gamma = 0; pressure mean by a scalar
    multiplier (irrelevant for the outputs);
  * axis r = 0: u_r = u_theta = 0 (Dirichlet on those components only), u_z and p natural;
  * Dirichlet data f1b_stokes_check.cavity_lid by P2 LAGRANGE NODAL INTERPOLATION: Set(dual=True) for the vertex
    values, then each Dirichlet edge's bubble coefficient corrected so that the midpoint value equals the data
    (max nodal defect printed and stored);
  * quadrature: bonus order 8.  On elements with an axis EDGE the free u_r, u_theta test functions vanish on it and
    the hoop integrands u v / r are polynomials; on elements that touch the axis at ONE vertex they are rational, so
    the discrete problem depends (slightly) on the rule: E_kin moves by up to ~1e-8 at N = 16 between rules of
    degree 4..10, by <= 6e-12 at N = 64, 128 (the compared levels);
  * sparse LU (UMFPACK).
Outputs (JSON): quantities {"<form>_Ekin_N<N>": (1/2) int |u_h|^2 r dr dz, "<form>_Diss_N<N>": a_form(u_h, u_h)}
(per radian) and diagnostics.

    python -B tools/fluids_mms/xcode/ngsolve_axi_cavity_reference.py OUT.json [NMAX]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import ngsolve as ngs
from netgen.meshing import Mesh as NGMesh, MeshPoint, Element2D, Element1D, FaceDescriptor, Pnt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import f1b_stokes_check as C  # noqa: E402  (fixed mesh builder and lid data only)

ngs.SetNumThreads(1)
ORD = 14


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
            if abs(V[a][0]) < 1e-12 and abs(V[b][0]) < 1e-12:
                idx = 1                                            # axis
            elif abs(V[a][1] - 1) < 1e-12 and abs(V[b][1] - 1) < 1e-12:
                idx = 2                                            # lid
            else:
                idx = 3                                            # wall r = 1 and bottom z = 0
            m.Add(Element1D([pids[a], pids[b]], index=idx))
    m.SetBCName(0, "axis")
    m.SetBCName(1, "lid")
    m.SetBCName(2, "wall")
    return ngs.Mesh(m)


def set_nodal(gfc, space, mesh, fn, bnd):
    """P2 Lagrange nodal interpolation of fn(r, z) on the boundary edges named in bnd."""
    gfc.Set(fn(ngs.x, ngs.y), definedon=mesh.Boundaries(bnd), dual=True)
    edges = []
    for el in mesh.Elements(ngs.BND):
        if el.mat in bnd.split("|"):
            edges.extend(el.edges)
    probe = ngs.GridFunction(space)
    phi_mid = None
    for e in edges:
        vts = mesh[e].vertices
        vd = set(sum((list(space.GetDofNrs(v)) for v in vts), []))
        edge_dofs = [d for d in space.GetDofNrs(e) if d >= 0 and d not in vd]
        pa, pb = (np.array(mesh[v].point) for v in vts)
        xm, ym = 0.5 * (pa + pb)
        if phi_mid is None:
            probe.vec[:] = 0
            probe.vec[edge_dofs[0]] = 1.0
            phi_mid = probe(mesh(xm, ym, 0.0))
        gfc.vec[edge_dofs[0]] += (fn(xm, ym) - gfc(mesh(xm, ym, 0.0))) / phi_mid
    worst = 0.0
    for e in edges:
        vts = mesh[e].vertices
        pa, pb = (np.array(mesh[v].point) for v in vts)
        for pt in (pa, pb, 0.5 * (pa + pb)):
            worst = max(worst, abs(gfc(mesh(pt[0], pt[1], 0.0)) - fn(pt[0], pt[1])))
    return worst


def solve(N, form):
    V, T = C.square_mesh(N)
    mesh = ng_mesh(V, T)
    Ur = ngs.H1(mesh, order=2, dirichlet="axis|lid|wall")
    Uz = ngs.H1(mesh, order=2, dirichlet="lid|wall")
    Q = ngs.H1(mesh, order=1)
    X = ngs.FESpace([Ur, Uz, Ur, Q, ngs.NumberSpace(mesh)])
    (ur, uz, ut, p, lam), (vr, vz, vt, q, mt) = X.TnT()
    r = ngs.x

    def G(a, b, c):  # 3x3 gradient, rows/cols (r, z, theta)
        return ngs.CF((ngs.grad(a)[0], ngs.grad(a)[1], -c / r,
                       ngs.grad(b)[0], ngs.grad(b)[1], 0,
                       ngs.grad(c)[0], ngs.grad(c)[1], a / r), dims=(3, 3))
    Gu, Gv = G(ur, uz, ut), G(vr, vz, vt)
    divu = ngs.grad(ur)[0] + ur / r + ngs.grad(uz)[1]
    divv = ngs.grad(vr)[0] + vr / r + ngs.grad(vz)[1]
    if form == "lap":
        visc = ngs.InnerProduct(Gu, Gv)
    else:
        visc = 2 * ngs.InnerProduct(0.5 * (Gu + Gu.trans), 0.5 * (Gv + Gv.trans))
    a = ngs.BilinearForm(X)
    a += ((visc - p * divv - q * divu) * r + lam * q + mt * p) * ngs.dx(bonus_intorder=8)
    a.Assemble()
    gf = ngs.GridFunction(X)
    lid_r = lambda R, Z: R * (1 - R ** 2) ** 2 * Z ** 2          # noqa: E731  (= cavity_lid components)
    lid_t = lambda R, Z: 2 * R * (1 - R ** 2) ** 2 * Z ** 2      # noqa: E731
    d1 = set_nodal(gf.components[0], Ur, mesh, lid_r, "lid|wall")
    d2 = set_nodal(gf.components[2], Ur, mesh, lid_t, "lid|wall")
    # check against the fixed callable at the nodes
    chk = C.cavity_lid(np.array([0.5]), np.array([1.0]))[0]
    assert abs(chk[0] - lid_r(0.5, 1.0)) < 1e-15 and abs(chk[2] - lid_t(0.5, 1.0)) < 1e-15
    free = X.FreeDofs()
    res = gf.vec.CreateVector()
    res.data = -a.mat * gf.vec
    r0 = np.linalg.norm(np.array(res.FV().NumPy())[np.array(free)])
    gf.vec.data += a.mat.Inverse(free, inverse="umfpack") * res
    r2 = gf.vec.CreateVector()
    r2.data = -a.mat * gf.vec
    relres = float(np.linalg.norm(np.array(r2.FV().NumPy())[np.array(free)]) / max(r0, 1e-300))
    urh, uzh, uth = gf.components[0], gf.components[1], gf.components[2]
    ekin = 0.5 * ngs.Integrate((urh ** 2 + uzh ** 2 + uth ** 2) * r, mesh, order=ORD)
    Gh = G(urh, uzh, uth)
    if form == "lap":
        diss = ngs.Integrate(ngs.InnerProduct(Gh, Gh) * r, mesh, order=ORD)
    else:
        Sh = 0.5 * (Gh + Gh.trans)
        diss = 2 * ngs.Integrate(ngs.InnerProduct(Sh, Sh) * r, mesh, order=ORD)
    return {"Ekin": float(ekin), "Diss": float(diss), "relres": relres, "nodal_defect": float(max(d1, d2)),
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
           "case": "xcode_stokes_axi_cavity_swirl", "quantities": q, "diagnostics": diag, "seconds": None}
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
