"""Transient Navier-Stokes cross-code instrument with FEniCSx/DOLFINx 0.11: a code independent of the NGSolve
reference (UFL forms, automatic differentiation for the Jacobian, PETSc SNES/TS, MUMPS LU).  Only the fixed mesh
builder and the field data loader of the check module (read only) are shared.

Modes (output JSON first):
  t1   OUT N...          fixed-step BDF1/BDF2 (BDF1 starter), Newton per step and semi-implicit, on the planar
                         space-exact temporal MMS: final-time velocity and pressure errors (inter-code check of the
                         NGSolve reference solver, same discrete problem)
  tsbdf OUT              PETSc TSBDF (order 2, adaptive, its own estimator and controller) on the error-control field:
                         global error at T over the fixed rtol sweep (cross-code check of the tolerance behaviour)
  x1   OUT N...          the start-up cavity (regularized lid, Re = 400): Ekin, Enst, Diss at t = 1 and 2 for lap/newton,
                         sym/newton, lap/semi (inter-code check of the X1 cross-code reference)

    UCX_TLS=self,sm fenicsx/bin/python -B tools/fluids_mms/xcode/dolfinx_tr.py MODE OUT [N ...]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
sys.path.insert(0, str(HERE.parent))
import f3a_transient_check as C  # noqa: E402
import f3a_spec as S  # noqa: E402

UFLFN = {"sin": ufl.sin, "cos": ufl.cos, "exp": ufl.exp, "tanh": ufl.tanh, "sqrt": ufl.sqrt, "pi": math.pi}


def field(name):
    p = ROOT / "tools" / "fluids_mms" / "out" / f"{name}.json"
    return C.load_field({"path": f"tools/fluids_mms/out/{name}.json",
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})


def ufl_from_sympy(s, X, t):
    """A sympy string in (x, y, t) as a UFL expression (no sympy at run time: parse with a tiny evaluator)."""
    expr = s.replace("^", "**")
    ns = dict(UFLFN)
    ns.update({"x": X[0], "y": X[1], "t": t, "Rational": lambda a, b: float(a) / float(b),
               "Integer": float, "Float": float})
    return eval(expr, {"__builtins__": {}}, ns)


class Setup:
    def __init__(self, N, fld, form, mean_pin=True):
        V0, T0 = C.structured_mesh(N)
        ce = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
        self.msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, np.asarray(T0, np.int64), ce, np.asarray(V0, np.float64))
        P2 = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
        P1 = basix.ufl.element("Lagrange", "triangle", 1)
        self.W = dolfinx.fem.functionspace(self.msh, basix.ufl.mixed_element([P2, P1]))
        self.Vs, self.Vmap = self.W.sub(0).collapse()
        self.Qs, self.Qmap = self.W.sub(1).collapse()
        fdim = 1
        bfac = dolfinx.mesh.locate_entities_boundary(self.msh, fdim, lambda x: np.full(x.shape[1], True))
        self.g = dolfinx.fem.Function(self.Vs)
        self.bdofs = dolfinx.fem.locate_dofs_topological((self.W.sub(0), self.Vs), fdim, bfac)
        self.bc_u = dolfinx.fem.dirichletbc(self.g, self.bdofs, self.W.sub(0))
        p0 = dolfinx.fem.Function(self.Qs)
        self.pdofs = dolfinx.fem.locate_dofs_geometrical((self.W.sub(1), self.Qs),
                                                         lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
        self.bc_p = dolfinx.fem.dirichletbc(p0, self.pdofs, self.W.sub(1))
        self.fld, self.form = fld, form
        self.X = ufl.SpatialCoordinate(self.msh)
        self.t = dolfinx.fem.Constant(self.msh, 0.0)

    def set_bc(self, t):
        self.g.interpolate(lambda x: np.asarray(self.fld.u(x[0], x[1], t), float).T.copy())

    def visc(self, u, v):
        mu = self.fld.mu
        if self.form == "lap":
            return mu * ufl.inner(ufl.grad(u), ufl.grad(v))
        return 2 * mu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v)))

    def forcing(self):
        return ufl.as_vector([ufl_from_sympy(s["sympy"], self.X, self.t) for s in self.fld.data["forms"][self.form]["f"]])


def fixed_step(st: Setup, ts, order, scheme):
    """BDF(order) with one BDF1 starter step on the grid ts; returns the final (u, p) as collapsed functions."""
    fld = st.fld
    rho = fld.rho
    w = dolfinx.fem.Function(st.W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(st.W)
    un1, un2, wad = (dolfinx.fem.Function(st.Vs) for _ in range(3))
    a = [dolfinx.fem.Constant(st.msh, 0.0) for _ in range(3)]
    st.set_bc(ts[0])
    if not fld.data.get("data_only"):
        un1.interpolate(lambda x: np.asarray(fld.u(x[0], x[1], ts[0]), float).T.copy())   # space-exact / zero: exact
    w.x.array[st.Vmap] = un1.x.array
    dudt = a[0] * u + a[1] * un1 + a[2] * un2
    f = st.forcing()
    adv = ufl.dot(ufl.grad(u), u) if scheme == "newton" else ufl.dot(ufl.grad(u), wad)
    F = (rho * ufl.inner(dudt, v) + st.visc(u, v) + rho * ufl.inner(adv, v) - p * ufl.div(v) - q * ufl.div(u)
         - ufl.inner(f, v)) * ufl.dx
    prob = dolfinx.fem.petsc.NonlinearProblem(F, w, bcs=[st.bc_u, st.bc_p], petsc_options={
        "snes_type": "newtonls", "snes_linesearch_type": "basic", "snes_rtol": 1e-14, "snes_atol": 1e-13,
        "snes_stol": 0.0, "snes_max_it": 30, "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}, petsc_options_prefix=f"fx{id(w)}_")
    hist = []
    its = 0
    for n in range(1, len(ts)):
        dt = ts[n] - ts[n - 1]
        k = order if n >= 2 else 1
        al = C.bdf_coefficients(k, [dt] + ([ts[n - 1] - ts[n - 2]] if k == 2 else []))
        for j in range(3):
            a[j].value = al[j] / dt if j < len(al) else 0.0
        st.t.value = ts[n]
        st.set_bc(ts[n])
        if scheme == "semi":
            if k == 1:
                wad.x.array[:] = un1.x.array
            else:
                be = C.extrapolation_coefficients(2, [dt, ts[n - 1] - ts[n - 2]])
                wad.x.array[:] = be[0] * un1.x.array + be[1] * un2.x.array
        prob.solve()
        snes = prob.solver
        if snes.getConvergedReason() <= 0:
            raise RuntimeError(f"SNES failed at step {n}: {snes.getConvergedReason()}")
        its += snes.getIterationNumber()
        hist.append(un1.x.array.copy())
        un2.x.array[:] = un1.x.array
        un1.x.array[:] = w.x.array[st.Vmap]
    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()
    return uh, ph, its


def errors_exact_space(st, uh, ph, t):
    """Velocity and mean-free pressure errors against the exact P2/P1 interpolants (space-exact field: exact)."""
    fld = st.fld
    ue = dolfinx.fem.Function(st.Vs)
    ue.interpolate(lambda x: np.asarray(fld.u(x[0], x[1], t), float).T.copy())
    pe = dolfinx.fem.Function(st.Qs)
    pe.interpolate(lambda x: np.asarray(fld.p(x[0], x[1], t), float))
    du = uh - ue
    eu = math.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(du, du) * ufl.dx)))
    area = 1.0
    mh = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ph * ufl.dx)) / area
    me = dolfinx.fem.assemble_scalar(dolfinx.fem.form(pe * ufl.dx)) / area
    dp = (ph - mh) - (pe - me)
    ep = math.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(dp * dp * ufl.dx)))
    return eu, ep


def mode_t1(out, Ns):
    F = field("f3a_tr_sq_poly")
    res = {"code": "DOLFINx", "version": dolfinx.__version__, "petsc": ".".join(map(str, PETSc.Sys.getVersion())),
           "field": F.name, "results": {}}
    for form in S.FORMS:
        for order, scheme, gk in S.T1_VARIANTS:
            key = f"{form}_bdf{order}_{scheme}_{gk}"
            rows = []
            for Nt in S.T1_NT:
                st = Setup(Ns[0], F, form)
                grid = {"kind": "uniform"} if gk == "uniform" else S.T1_CYCLIC
                ts = C.time_grid(grid, S.T1_T, Nt)
                t0 = time.time()
                uh, ph, its = fixed_step(st, ts, order, scheme)
                eu, ep = errors_exact_space(st, uh, ph, S.T1_T)
                rows.append({"Nt": Nt, "uL2_T": eu, "pL2_T": ep, "newton_its": its})
                print(f"dolfinx t1 {key} Nt={Nt}: u {eu:.6e} p {ep:.6e} ({time.time() - t0:.0f}s)", flush=True)
            res["results"][key] = rows
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


def mode_x1(out, Ns):
    F = field("f3a_tr_cavity_x1")
    res = {"code": "DOLFINx", "version": dolfinx.__version__, "petsc": ".".join(map(str, PETSc.Sys.getVersion())),
           "field": F.name, "quantities": {}}
    for form, scheme in (("lap", "newton"), ("sym", "newton"), ("lap", "semi")):
        for N in Ns:
            st = Setup(N, F, form)
            out_t = {}
            for T_end in (1.0, 2.0):
                ts = C.time_grid({"kind": "uniform"}, T_end, int(round(T_end * X1_NT_PER_UNIT)))
                t0 = time.time()
                uh, ph, its = fixed_step(st, ts, 2, scheme)
                G = ufl.grad(uh)
                om = uh[1].dx(0) - uh[0].dx(1)

                def integ(e):
                    return float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(e * ufl.dx)))
                dis = F.mu * integ(ufl.inner(G, G)) if form == "lap" else 2 * F.mu * integ(ufl.inner(ufl.sym(G),
                                                                                                    ufl.sym(G)))
                q = {"Ekin": 0.5 * integ(ufl.inner(uh, uh)), "Enst": integ(om * om), "Diss": dis}
                for k, v in q.items():
                    res["quantities"][f"{form}_{scheme}_{k}_t{int(T_end)}_N{N}"] = v
                print(f"dolfinx x1 {form}/{scheme} N={N} T={T_end}: {q} ({time.time() - t0:.0f}s)", flush=True)
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


X1_NT_PER_UNIT = 50


def mode_tsbdf(out):
    """PETSc TSBDF order 2 (adaptive, PETSc's own error estimate and basic controller) on the error-control field
    (planar, lap form, N = 8): IFunction F(t, U, Udot) of the mixed system with the Dirichlet rows U - g(t) and a
    pinned pressure vertex; error weights: velocity DOFs atol = rtol S_u, rtol = rtol; pressure and constrained DOFs
    excluded (huge tolerances).  Records the relative global velocity error at T over the fixed sweep."""
    F = field(S.T2_FIELD)
    Su = C.solution_scale(F, S.T2_T)
    st = Setup(S.T2_MESH["N"][0], F, "lap")
    rho = F.rho
    w = dolfinx.fem.Function(st.W)
    wd = dolfinx.fem.Function(st.W)
    u, p = ufl.split(w)
    ud, _ = ufl.split(wd)
    v, q = ufl.TestFunctions(st.W)
    shift = dolfinx.fem.Constant(st.msh, 0.0)
    f = st.forcing()
    Fform = (rho * ufl.inner(ud, v) + st.visc(u, v) + rho * ufl.inner(ufl.dot(ufl.grad(u), u), v) - p * ufl.div(v)
             - q * ufl.div(u) - ufl.inner(f, v)) * ufl.dx
    J = ufl.derivative(Fform, w) + shift * ufl.derivative(Fform, wd)
    Fc, Jc = dolfinx.fem.form(Fform), dolfinx.fem.form(J)
    cons = np.concatenate([np.asarray(st.bdofs[0], np.int32), np.asarray(st.pdofs[0], np.int32)])
    nb = len(st.bdofs[0])
    rows = []
    for rt in S.T2_RTOLS:
        st.set_bc(0.0)
        w.x.array[:] = 0.0
        w.x.array[st.Vmap] = np.asarray(st.g.x.array)
        u0 = dolfinx.fem.Function(st.Vs)
        u0.interpolate(lambda x: np.asarray(F.u(x[0], x[1], 0.0), float).T.copy())
        w.x.array[st.Vmap] = u0.x.array
        Uv = w.x.petsc_vec.copy()
        Fv = Uv.duplicate()
        A = dolfinx.fem.petsc.create_matrix(Jc)

        def ifunc(ts_, t, U, Ud, Fout):
            U.copy(w.x.petsc_vec)
            w.x.scatter_forward()
            Ud.copy(wd.x.petsc_vec)
            wd.x.scatter_forward()
            st.t.value = t
            st.set_bc(t)
            with Fout.localForm() as fl:
                fl.set(0.0)
            dolfinx.fem.petsc.assemble_vector(Fout, Fc)
            Fout.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            Fa = Fout.getArray()
            Ua = U.getArray(readonly=True)
            bd = np.asarray(st.bdofs[0])
            Fa[bd] = Ua[bd] - st.g.x.array[np.asarray(st.bdofs[1])]
            pdd = np.asarray(st.pdofs[0])
            Fa[pdd] = Ua[pdd]

        def ijac(ts_, t, U, Ud, a_, Jm, Pm):
            U.copy(w.x.petsc_vec)
            w.x.scatter_forward()
            shift.value = a_
            Jm.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(Jm, Jc)
            Jm.assemble()
            Jm.zeroRows(PETSc.IS().createGeneral(cons), diag=1.0)
            Jm.assemble()

        ts = PETSc.TS().create(MPI.COMM_SELF)
        ts.setType("bdf")
        ts.setIFunction(ifunc, Fv)
        ts.setIJacobian(ijac, A, A)
        ts.setTime(0.0)
        ts.setMaxTime(S.T2_T)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        ts.setTimeStep(S.T2_H0)
        ts.setMaxSteps(200000)
        atol = Uv.duplicate()
        rtv = Uv.duplicate()
        atol.set(1e10)
        rtv.set(1e10)
        va, ra = atol.getArray(), rtv.getArray()
        vm = np.asarray(st.Vmap)
        va[vm] = rt * Su / math.sqrt(1.0)
        ra[vm] = 0.0
        va[np.asarray(st.bdofs[0])] = 1e10
        ts.setTolerances(rtol=rtv, atol=atol)
        opts = PETSc.Options()
        pre = f"bdf{len(rows)}_"
        ts.setOptionsPrefix(pre)
        opts[pre + "ts_bdf_order"] = 2
        opts[pre + "ts_adapt_type"] = "basic"
        opts[pre + "ts_max_snes_failures"] = -1
        opts[pre + "snes_rtol"] = 1e-12
        opts[pre + "snes_atol"] = 1e-13
        opts[pre + "ksp_type"] = "preonly"
        opts[pre + "pc_type"] = "lu"
        opts[pre + "pc_factor_mat_solver_type"] = "mumps"
        ts.setFromOptions()
        t0 = time.time()
        ts.solve(Uv)
        Uv.copy(w.x.petsc_vec)
        w.x.scatter_forward()
        uh = w.sub(0).collapse()
        ue = dolfinx.fem.Function(st.Vs)
        ue.interpolate(lambda x: np.asarray(F.u(x[0], x[1], S.T2_T), float).T.copy())
        d = uh - ue
        err = math.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(d, d) * ufl.dx))) / Su
        rows.append({"rtol": rt, "err": err, "err_over_tol": err / rt, "steps": ts.getStepNumber(),
                     "rejected": ts.getStepRejections(), "seconds": round(time.time() - t0, 1)})
        print(f"tsbdf rtol={rt}: err {err:.3e} err/tol {err / rt:.3f} steps {ts.getStepNumber()} "
              f"rej {ts.getStepRejections()}", flush=True)
    Path(out).write_text(json.dumps({"code": "PETSc TSBDF (order 2, TSADAPTBASIC) via DOLFINx",
                                     "dolfinx": dolfinx.__version__,
                                     "petsc": ".".join(map(str, PETSc.Sys.getVersion())), "S_u": Su,
                                     "error_weights": "velocity DOFs: atol = rtol S_u, rtol 0 (WRMS over all DOFs, "
                                                      "constrained and pressure DOFs excluded by huge tolerances)",
                                     "sweep": rows}, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    Ns = [int(a) for a in sys.argv[3:]]
    if mode == "t1":
        mode_t1(out, Ns or [8])
    elif mode == "x1":
        mode_x1(out, Ns or [32, 64])
    elif mode == "tsbdf":
        mode_tsbdf(out)
    else:
        raise SystemExit(mode)
