"""Generate the F1c MMS / closed-form data and run the self-checks.

    python3 -B tools/fluids_mms/generate_f1c.py [--only NAME ...]

Writes tools/fluids_mms/out/f1c_<field>.json and tools/fluids_mms/out/f1c_selfcheck_report.json; exit status 1 if
any self-check fails.  Checks: div u = 0; forcing by two independent derivations per form and with grad-div
(50-digit evaluation at random points); the EMITTED sympy string parsed back gives zero residual; the numpy spelling
evaluates to the same numbers; a 4th-order finite-difference residual per form from u, p, mu only, and the other
form's forcing must fail it (variable mu); face conditions: every slip/symmetry/do-nothing face satisfies its own
condition (and do-nothing only in its own form), and every face VIOLATES the homogeneous condition of every other
natural/slip kind of its field (cross-BC rule); zero net boundary flux of the enclosed fields;
sphere fields: each Stokes basis field solves the homogeneous Stokes equations, the boundary data, f = 0, and the
closed-form drag equals a 2-D Gauss quadrature of sigma n over r = a in both forms.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

import stokes3d as S  # noqa: E402
import f1c_fields as F  # noqa: E402

OUT = HERE / "out"


def constants(fld, em):
    P = S.sample_points(fld, 4000, seed=29)
    uf = [S.np_lambda(c) for c in fld.u]
    U = np.stack([np.broadcast_to(f(*P.T), (len(P),)) for f in uf], -1)
    pv = np.broadcast_to(S.np_lambda(fld.p)(*P.T), (len(P),))
    muv = np.broadcast_to(S.np_lambda(fld.mu)(*P.T), (len(P),))
    vol = S.integrate(fld, lambda X, Y, Z: np.ones_like(X))
    pmean = S.integrate(fld, lambda X, Y, Z: np.broadcast_to(S.np_lambda(fld.p)(X, Y, Z), X.shape)) / vol
    return {"volume": vol, "p_mean": pmean, "u_max_sampled": float(np.max(np.linalg.norm(U, axis=1))),
            "p_max_abs_sampled": float(np.max(np.abs(pv))),
            "p_minus_mean_max_abs_sampled": float(np.max(np.abs(pv - pmean))),
            "mu_min_sampled": float(np.min(muv)), "mu_max_sampled": float(np.max(muv))}


def face_checks(fld):
    rep, ok = {}, True
    kinds = {f.bc for f in fld.faces} - {"dirichlet", "traction"}
    for face in fld.faces:
        for form in fld.forms:
            if face.bc in ("slip", "symmetry"):
                v = S.face_condition_values(fld, face, "slip", form)
                rep[f"{face.name}_{face.bc}_{form}_own"] = v
                ok &= v < 1e-25
            if face.bc == "do_nothing":
                v = S.face_condition_values(fld, face, "do_nothing", form)
                own = fld.extra.get("do_nothing_form") in (form, "both")
                rep[f"{face.name}_do_nothing_{form}_{'own' if own else 'other_must_violate'}"] = v
                ok &= (v < 1e-25) if own else (v > 1e-3)
            for k in ([] if fld.extra.get("cross_bc_exempt") else kinds):   # cross-BC rule
                if k == face.bc or ({k, face.bc} <= {"slip", "symmetry"}):
                    continue
                if k == "do_nothing" and fld.extra.get("do_nothing_form") not in (None, form, "both"):
                    continue
                v = S.face_condition_values(fld, face, "slip" if k in ("slip", "symmetry") else k, form)
                rep[f"{face.name}_{face.bc}_violates_{k}_{form}"] = v
                ok &= v > 1e-3
    if fld.pressure == "mean_zero" and fld.box is not None:   # zero net flux through the boundary (exact u)
        tot = 0.0
        g, w = np.polynomial.legendre.leggauss(24)
        s = 0.5 * (g + 1)
        A, B = np.meshgrid(s, s, indexing="ij")
        W = np.outer(w, w) * 0.25
        uf = [S.np_lambda(c) for c in fld.u]
        for face in fld.faces:
            i, v = face.fixed
            cols = [None, None, None]
            others = [j for j in range(3) if j != i]
            cols[i] = np.full_like(A, float(v))
            cols[others[0]], cols[others[1]] = A, B
            un = sum(float(face.normal[j]) * np.broadcast_to(uf[j](*cols), A.shape) for j in range(3))
            tot += float(np.sum(W * un))
        rep["net_boundary_flux"] = tot
        ok &= abs(tot) < 1e-12
    return rep, ok


def sphere_checks(fld):
    rep, ok = {}, True
    for nm, (bu, bp) in F.BASIS.items():
        d = S.div_vec(bu)
        res = [sum(sp.diff(bu[i], S.X[j], 2) for j in range(3)) - sp.diff(bp, S.X[i]) for i in range(3)]
        rng = random.Random(41)
        pts = [[sp.Rational(rng.randint(-150, 150), 100) + sp.Rational(1, 7) for _ in range(3)] for _ in range(4)]
        m = max(float(abs(S.eval_mp(e, P))) for P in pts for e in [d] + res)
        rep[f"basis_{nm}_stokes_residual"] = m
        ok &= m < 1e-25
    rng = random.Random(43)
    pts = [[sp.Float(c, 30) for c in S.sample_points(fld, 1, seed=rng.randint(0, 10 ** 6))[0]] for _ in range(6)]
    for form in fld.forms:
        f = S.forcing_route_a(fld.u, fld.p, fld.mu, form)
        m = max(float(abs(S.eval_mp(f[i], P))) for P in pts for i in range(3))
        rep[f"forcing_zero_{form}"] = m
        ok &= m < 1e-25
    d = max(float(abs(S.eval_mp(S.div_vec(fld.u), P))) for P in pts)
    rep["div_u"] = d
    ok &= d < 1e-25
    # boundary data: u = 0 on r = 1; on r = 4 the unbounded field / uniform U e_z
    for rad, want in ((1, "zero"), (4, "outer")):
        worst = 0.0
        for k in range(6):
            th, ph = 0.3 + 0.4 * k, 1.1 * k + 0.2
            P = [sp.Float(rad * np.sin(th) * np.cos(ph), 30), sp.Float(rad * np.sin(th) * np.sin(ph), 30),
                 sp.Float(rad * np.cos(th), 30)]
            uv = [S.eval_mp(fld.u[i], P) for i in range(3)]
            if want == "zero" or "concentric" in fld.name:
                tgt = [0, 0, 0] if want == "zero" else [0, 0, 1]
                worst = max(worst, max(float(abs(uv[i] - tgt[i])) for i in range(3)))
        rep[f"bc_r{rad}"] = worst
        ok &= worst < 1e-14   # double-precision sample points on the spheres
    # drag: closed form vs quadrature of sigma n over r = 1 (n = x / r, pointing INTO the fluid = outward from the
    # body: the force exerted by the fluid on the body is int sigma n_body dS with n_body = x/r)
    gt, wt = np.polynomial.legendre.leggauss(40)
    nph = 80
    ph = np.arange(nph) * 2 * np.pi / nph
    CT, PH = np.meshgrid(gt, ph, indexing="ij")
    ST = np.sqrt(1 - CT ** 2)
    Xs, Ys, Zs = ST * np.cos(PH), ST * np.sin(PH), CT
    W = wt[:, None] * np.full(nph, 2 * np.pi / nph)[None, :]
    exact = float(fld.extra["drag_z_exact"])
    for form in fld.forms:
        Sg = S.stress(fld.u, fld.p, fld.mu, form)
        fz = sum(np.broadcast_to(S.np_lambda(Sg[2, j])(Xs, Ys, Zs), Xs.shape) * (Xs, Ys, Zs)[j] for j in range(3))
        q = float(np.sum(W * fz))
        rep[f"drag_quadrature_{form}"] = q
        rep[f"drag_quadrature_minus_exact_{form}"] = q - exact
        # the Cauchy (sym) traction gives the force; the lap pseudo-traction equals it on the no-slip sphere
        ok &= abs(q - exact) < 1e-10 * abs(exact)
    rep["drag_exact"] = exact
    rep["drag_exact_over_6pi"] = exact / (6 * np.pi)
    return rep, ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    rpath = OUT / "f1c_selfcheck_report.json"
    report = json.loads(rpath.read_text())["fields"] if (a.only and rpath.exists()) else {}
    allok = True
    for mk in F.ALL:
        fld = mk()
        if a.only and fld.name not in a.only:
            continue
        t0 = time.time()
        em = S.emit(fld)
        if fld.no_forcing:
            rep, ok = {"status": "data-only field (no PDE checks)"}, True
        elif fld.shell:
            rep, ok = sphere_checks(fld)
        else:
            rep = S.selfcheck(fld, em)
            ok = rep["status"] == "PASS"
            frep, fok = face_checks(fld)
            rep.update(frep)
            ok &= fok
        em["constants"] = constants(fld, em) if not fld.no_forcing else {"volume": 1.0}
        if fld.shell:
            em["constants"].update({"drag_z_exact": float(fld.extra["drag_z_exact"]),
                                    "drag_z_exact_sympy": str(fld.extra["drag_z_exact"])})
        rep["status"] = "PASS" if ok else "FAIL"
        rep["seconds"] = round(time.time() - t0, 1)
        em["selfcheck"] = {k: v for k, v in rep.items() if k != "seconds"}
        (OUT / f"{fld.name}.json").write_text(json.dumps(em, indent=1, sort_keys=True) + "\n")
        report[fld.name] = rep
        allok &= ok
        print(f"{fld.name}: {rep['status']} ({rep['seconds']} s)", flush=True)
    rpath.write_text(json.dumps({"fields": report}, indent=1, sort_keys=True) + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
