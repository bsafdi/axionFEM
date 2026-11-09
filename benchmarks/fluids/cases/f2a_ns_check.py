"""Steady Navier-Stokes (group F2a): the check module's error computation (runner.check_module "f2a_ns_check").

The case runner (cases/f2a_ns.py) builds the fixed meshes with the builders here (and those of the planar Stokes
module f1a_stokes_check, imported read-only), solves through the public fluids API, wraps the discrete solution in an
Evaluator and calls the functions below.  Nothing here calls the solver: every norm, identity, Newton-history and
Taylor-test number is computed here from raw point evaluations and raw vectors.

Evaluator protocol:
  planar        ev.geometry(ref) -> (x[ne,nq,2], detJ[ne,nq]);  ev.velocity(ref) -> (u[ne,nq,2], G[ne,nq,2,2]),
                G[...,i,j] = d u_i / d x_j;  ev.pressure(ref) -> p[ne,nq]
  axisymmetric  ev.geometry(ref) -> (rz[ne,nq,2], detJ[ne,nq]) in the (r, z) half-plane (detJ of the planar map, NO r);
                ev.velocity(ref) -> (u[ne,nq,3], D[ne,nq,3,2]) with u = (u_r, u_theta, u_z) and
                D[...,i,j] = d u_i / d (r, z)_j (component derivatives);  ev.pressure(ref) -> p[ne,nq]
ref = points of the reference triangle (0,0), (1,0), (0,1).

Conventions: rho (u.grad)u - div sigma_form = f, div u = 0; sigma_sym = 2 mu eps - p I,
sigma_lap = mu G - p I.  Convective forms: "standard" c(w; u, v) = rho ((w.grad) u, v); "skew" (Temam)
c(w; u, v) = rho ((w.grad) u, v) + (rho/2) ((div w) u, v).  Axisymmetric measures carry the weight r (the physical
volume / 2 pi); "unweighted" norms drop it.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy.special as sc

import f1a_stokes_check as C1           # fixed F1a module (mesh builders, quadrature, nodal checks)

QUAD_DEGREE = 10
tri_rule = C1.tri_rule
structured_mesh = C1.structured_mesh
distorted_mesh = C1.distorted_mesh
red_refine = C1.red_refine
mesh_hash = C1.mesh_hash
verify_mesh = C1.verify_mesh
boundary_edges = C1.boundary_edges
all_boundary_triangles = C1.all_boundary_triangles
dirichlet_nodal_defect = C1.dirichlet_nodal_defect
dirichlet_shared_node_mismatch = C1.dirichlet_shared_node_mismatch


# ============================================================================================== meshes
def rect_mesh(nx0: int, ny0: int, k: int, x0: float, x1: float, y0: float, y1: float):
    """nx0 x ny0 squares on [x0,x1] x [y0,y1] split along the (1,1) diagonal except the cells at the corners (x1,y0)
    and (x0,y1) (split the other way: no triangle with all vertices on the boundary), refined uniformly (red) k
    times.  Nested in k.  Kovasznay: rect_mesh(6, 8, k, -1/2, 1, -1/2, 3/2), cells of side 1/4 / 2^k."""
    V, T = C1._structured(nx0, ny0, x0, x1, y0, y1)
    for _ in range(k):
        V, T = red_refine(V, T)
    return V, T


def distorted_unit(k: int):
    """The F1a distorted family on [0,1]^2 (f1a_stokes_check.distorted_mesh, seed 20261031, 0.2 perturbation)."""
    return distorted_mesh(k)


# ============================================================================================== fields
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compile(spec: dict, coords):
    code = compile(spec["numpy"], "<mms expression>", "eval")

    def f(A, B):
        v = eval(code, {"__builtins__": {}, "np": np, "sc": sc}, {coords[0]: A, coords[1]: B})
        return np.broadcast_to(np.asarray(v, dtype=float), np.shape(A))
    return f


class Field:
    """One generated field (tools/fluids_mms/out/ns_*.json), sha256-verified by load_field."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data["name"]
        self.geometry = data["geometry"]
        self.coords = data["coords"]
        self.rho = float(data["rho"]["value"])
        cc = self.coords
        self._u = [_compile(s, cc) for s in data["u"]]
        self._g = [[_compile(s, cc) for s in row] for row in data["grad_u"]]
        self.p = _compile(data["p"], cc)
        self.mu = _compile(data["mu"], cc)
        self._f = {form: [_compile(s, cc) for s in fd["f"]] for form, fd in data["forms"].items()}
        self.constants = data["constants"]
        self.exact = bool(data.get("exact_solution", True))
        self.ncomp = len(self._u)

    def u(self, A, B):
        return np.stack([g(A, B) for g in self._u], axis=-1)

    def grad_u(self, A, B):
        return np.stack([np.stack([g(A, B) for g in row], axis=-1) for row in self._g], axis=-2)

    def f(self, form):
        fs = self._f[form]
        return lambda A, B: np.stack([g(A, B) for g in fs], axis=-1)

    def traction(self, form, edge):
        b = next(b for b in self.data["boundaries"] if b["name"] == edge)
        gs = [_compile(s, self.coords) for s in b["traction"][form]]
        return lambda A, B: np.stack([g(A, B) for g in gs], axis=-1)


def load_field(entry: dict, root: Path | None = None) -> Field:
    """entry = {"path": "tools/fluids_mms/out/<name>.json", "sha256": ...} from the fixed run parameters."""
    root = Path(root) if root is not None else _repo_root()
    raw = (root / entry["path"]).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"MMS data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return Field(json.loads(raw))


# ============================================================================================== quadrature
def _weights(ev, deg=QUAD_DEGREE, axisymmetric=False):
    q, w = tri_rule(deg)
    xq, detJ = ev.geometry(q)
    if np.any(detJ <= 0):
        raise RuntimeError("non-positive Jacobian in the element map")
    W = w[None, :] * detJ
    if axisymmetric and np.any(xq[..., 0] < -1e-14):
        raise RuntimeError("axisymmetric mesh with r < 0")
    return q, xq, W


def cyl_gradient(u, D, rr):
    """The cylindrical gradient tensor G (rows r, theta, z; columns d_r, (1/r) d_theta, d_z) from the components
    u = (u_r, u_t, u_z) and their (r, z) derivatives D."""
    G = np.zeros(u.shape[:-1] + (3, 3))
    G[..., 0, 0], G[..., 0, 1], G[..., 0, 2] = D[..., 0, 0], -u[..., 1] / rr, D[..., 0, 1]
    G[..., 1, 0], G[..., 1, 1], G[..., 1, 2] = D[..., 1, 0], u[..., 0] / rr, D[..., 1, 1]
    G[..., 2, 0], G[..., 2, 2] = D[..., 2, 0], D[..., 2, 1]
    return G


# ============================================================================================== error norms
def norms(ev, fld: Field, pressure: str = "mean_free") -> dict:
    """Planar: uL2, uH1 (all gradient components), pL2 (mean-free or determined), divL2, as F1a.
    Axisymmetric: the r-WEIGHTED norms (the physical 3D norms / 2 pi):
        uL2  = (int |u - u_h|^2 r)^1/2,
        uH1  = (int |G(u) - G(u_h)|^2 r)^1/2 with the cylindrical gradient (hoop terms (u_r/r)^2, (u_t/r)^2 included),
        pL2  = (int ((p_h - <p_h>) - (p - <p>))^2 r)^1/2 with r-weighted means,
        divL2 = (int (d_r u_r + u_r / r + d_z u_z)^2 r)^1/2,
    and the UNWEIGHTED ones (suffix _unw): the same integrands without r, uH1_unw over the component derivatives
    only (no hoop terms), pressure means unweighted."""
    ax = fld.geometry == "axisymmetric"
    q, xq, W = _weights(ev, axisymmetric=ax)
    A, B = xq[..., 0], xq[..., 1]
    uh, Dh = ev.velocity(q)
    ph = ev.pressure(q)
    ue, De = fld.u(A, B), fld.grad_u(A, B)
    pe = fld.p(A, B)
    out = {"u_magnitude": float(np.max(np.linalg.norm(ue, axis=-1)))}
    weights = {"": W * A, "_unw": W} if ax else {"": W}
    for suf, Wt in weights.items():
        vol = Wt.sum()
        eu = uh - ue
        out["uL2" + suf] = math.sqrt(float((Wt[..., None] * eu ** 2).sum()))
        if ax and suf == "":
            Gh, Ge = cyl_gradient(uh, Dh, A), cyl_gradient(ue, De, A)
            out["uH1" + suf] = math.sqrt(float((Wt[..., None, None] * (Gh - Ge) ** 2).sum()))
            div = Dh[..., 0, 0] + uh[..., 0] / A + Dh[..., 2, 1]
        else:
            out["uH1" + suf] = math.sqrt(float((Wt[..., None, None] * (Dh - De) ** 2).sum()))
            div = (Dh[..., 0, 0] + Dh[..., 1, 1]) if not ax else (Dh[..., 0, 0] + uh[..., 0] / A + Dh[..., 2, 1])
        out["divL2" + suf] = math.sqrt(float((Wt * div ** 2).sum()))
        pmh, pm = float((Wt * ph).sum() / vol), float((Wt * pe).sum() / vol)
        if pressure == "mean_free":
            out["pL2" + suf] = math.sqrt(float((Wt * ((ph - pmh) - (pe - pm)) ** 2).sum()))
        elif pressure == "determined":
            out["pL2" + suf] = math.sqrt(float((Wt * (ph - pe) ** 2).sum()))
        else:
            raise ValueError(pressure)
        out["pmean_h" + suf], out["pmean" + suf] = pmh, pm
    pm = out["pmean"]
    out["p_magnitude"] = float(np.max(np.abs(pe - pm))) if pressure == "mean_free" else float(np.max(np.abs(pe)))
    return out


# ============================================================================================== discrete identities
def _forms(ev, form, conv, rho, mu, axisymmetric, deg=14):
    """a_form(u_h, u_h), c_conv(u_h; u_h, u_h) and the r-weighted measure pieces, by a degree-14 rule (exact for the
    polynomial integrands of P2 on affine elements up to degree 12 with variable mu evaluated pointwise)."""
    q, xq, W = _weights(ev, deg, axisymmetric)
    A, B = xq[..., 0], xq[..., 1]
    u, D = ev.velocity(q)
    m = mu(A, B) if callable(mu) else mu
    if axisymmetric:
        Wt = W * A
        G = cyl_gradient(u, D, A)
        div = G[..., 0, 0] + G[..., 1, 1] + G[..., 2, 2]
    else:
        Wt = W
        G = D
        div = G[..., 0, 0] + G[..., 1, 1]
    if form == "lap":
        a = float((Wt * m * (G ** 2).sum((-1, -2))).sum())
    elif form == "sym":
        E = 0.5 * (G + np.swapaxes(G, -1, -2))
        a = float((Wt * 2 * m * (E ** 2).sum((-1, -2))).sum())
    else:
        raise ValueError(form)
    adv = np.einsum("...ij,...j->...i", G, u)          # (u.grad)u in the orthonormal frame (G u)
    c_std = rho * float((Wt * (adv * u).sum(-1)).sum())
    c_div = 0.5 * rho * float((Wt * div * (u * u).sum(-1)).sum())
    c = {"standard": c_std, "skew": c_std + c_div}[conv]
    return {"a": a, "c": c, "c_standard": c_std, "c_skew": c_std + c_div, "div2": float((Wt * div ** 2).sum()),
            "q": q, "xq": xq, "Wt": Wt, "u": u}


def energy_identity(ev, form: str, conv: str, rho: float, mu, f, axisymmetric: bool = False) -> dict:
    """Homogeneous Dirichlet data (u_h = 0 on the Dirichlet boundary and, axisymmetric, u_r = u_t = 0 on the axis),
    exact load quadrature (linear f): testing the discrete momentum equation with v = u_h and the continuity
    equation with q = p_h gives
        a_form(u_h, u_h) + c_conv(u_h; u_h, u_h) = (f, u_h)          [r-weighted in axisymmetry]
    up to the Newton tolerance.  For "skew", c(u_h; u_h, u_h) = 0 identically; for "standard" it is
    -(rho/2) ((div u_h) u_h, u_h) != 0.  Returns the relative defect of the DECLARED (form, conv) and, as positive
    controls, the defects the other convective form and the other viscous form would show (they must be large)."""
    d = _forms(ev, form, conv, rho, mu, axisymmetric)
    fv = f(d["xq"][..., 0], d["xq"][..., 1])
    rhs = float((d["Wt"][..., None] * fv * d["u"]).sum())
    other_conv = "skew" if conv == "standard" else "standard"
    d_other_visc = _forms(ev, "sym" if form == "lap" else "lap", conv, rho, mu, axisymmetric)
    return {"defect": abs(d["a"] + d["c"] - rhs) / abs(rhs),
            "defect_other_convective_form": abs(d["a"] + d["c_" + other_conv] - rhs) / abs(rhs),
            "defect_other_viscous_form": abs(d_other_visc["a"] + d["c"] - rhs) / abs(rhs),
            "power_input": rhs, "a": d["a"], "c": d["c"], "c_standard": d["c_standard"], "c_skew": d["c_skew"]}


def swirl_pair_power(ev, rho: float) -> float:
    """int rho u_t^2 u_r dr dz = the power the centrifugal term -rho u_t^2 / r (r row) and the Coriolis term
    rho u_r u_t / r (theta row) each put into the energy identity, with opposite signs (they cancel exactly in a
    consistent discretization).  Reported with the axisymmetric energy identity as the size of the defect that
    dropping or flipping one of them produces."""
    q, xq, W = _weights(ev, 14, True)
    u, _ = ev.velocity(q)
    return float(rho * (W * u[..., 1] ** 2 * u[..., 0]).sum())


def functionals(ev, form: str, mu, axisymmetric: bool = False) -> dict:
    """Ekin = (1/2) int |u_h|^2, Diss = a_form(u_h, u_h), Enst = int |omega_h|^2 (planar)."""
    q, xq, W = _weights(ev, QUAD_DEGREE, axisymmetric)
    A = xq[..., 0]
    u, D = ev.velocity(q)
    Wt = W * A if axisymmetric else W
    out = {"Ekin": 0.5 * float((Wt[..., None] * u ** 2).sum())}
    m = mu(xq[..., 0], xq[..., 1]) if callable(mu) else mu
    G = cyl_gradient(u, D, A) if axisymmetric else D
    if form == "lap":
        out["Diss"] = float((Wt * m * (G ** 2).sum((-1, -2))).sum())
    else:
        E = 0.5 * (G + np.swapaxes(G, -1, -2))
        out["Diss"] = float((Wt * 2 * m * (E ** 2).sum((-1, -2))).sum())
    if not axisymmetric:
        out["Enst"] = float((Wt * (D[..., 1, 0] - D[..., 0, 1]) ** 2).sum())
    return out


def relative_difference(ev_a, ev_b, axisymmetric: bool = False) -> dict:
    """||u_a - u_b|| / ||u_a|| and the mean-free pressure analogue on the same mesh (path-independence checks:
    Picard vs Newton, Re continuation vs pseudo-transient continuation, two continuation schedules)."""
    q, xq, W = _weights(ev_a, QUAD_DEGREE, axisymmetric)
    Wt = W * xq[..., 0] if axisymmetric else W
    ua, _ = ev_a.velocity(q)
    ub, _ = ev_b.velocity(q)
    pa, pb = ev_a.pressure(q), ev_b.pressure(q)
    vol = Wt.sum()
    pa0, pb0 = pa - (Wt * pa).sum() / vol, pb - (Wt * pb).sum() / vol
    return {"u_rel": math.sqrt(float((Wt[..., None] * (ua - ub) ** 2).sum()) / float((Wt[..., None] * ua ** 2).sum())),
            "p_rel": math.sqrt(float((Wt * (pa0 - pb0) ** 2).sum()) / max(float((Wt * pa0 ** 2).sum()), 1e-300))}


# ============================================================================================== Newton diagnostics
def newton_order(residuals, floor: float = 1e-12) -> dict:
    """Observed local order of a nonlinear iteration from its residual norms r_0, r_1, ... (relative residuals of the steady system):  p_k = log(r_{k+1} / r_k) / log(r_k / r_{k-1}), over the k with
    r_{k+1} > floor (above round-off) and r_k < r_{k-1}.  Newton with an exact Jacobian is locally quadratic
    (p -> 2); a Picard/Oseen or a wrongly linearized Jacobian is linear (p ~ 1).  Returns the maximum, the MEDIAN
 (a Picard iteration can show one p_k up to 1.7, its median stays ~1) and the list.  The residuals are
    those the solver reports in the discrete dual norm."""
    r = [float(v) for v in residuals]
    ps = []
    for k in range(1, len(r) - 1):
        if r[k + 1] > floor and 0 < r[k] < r[k - 1] and r[k + 1] > 0:
            ps.append(math.log(r[k + 1] / r[k]) / math.log(r[k] / r[k - 1]))
    return {"p_max": max(ps) if ps else float("nan"), "p_median": float(np.median(ps)) if ps else float("nan"),
            "p": ps, "n": len(r) - 1}


def line_search_violations(residuals, steps, c1: float = 1e-4) -> int:
    """Number of accepted steps that violate the sufficient-decrease (Armijo) condition on the merit function,
    r_{k+1} <= (1 - c1 * lambda_k) r_k, lambda_k the accepted step length, r the residual in the discrete DUAL norm
 (basis-free).  A minimum-step (1/64) acceptance without sufficient decrease counts as a violation."""
    r = [float(v) for v in residuals]
    return int(sum(1 for k in range(len(r) - 1) if not r[k + 1] <= (1 - c1 * float(steps[k])) * r[k] * (1 + 1e-12)))


def taylor_test(R0, R_eps, Jd, eps) -> dict:
    """Jacobian Taylor test from raw vectors: R0 = R(x), R_eps[k] = R(x + eps_k d), Jd = J(x) d, all on
    the free degrees of freedom.  r1_k = ||R(x + e d) - R(x)||, r2_k = ||R(x + e d) - R(x) - e J d||.  For the exact
    Jacobian r1 = O(e) and r2 = O(e^2); the incompressible NS residual is QUADRATIC in the state, so r2 / e^2 is
    constant (= ||c(d; d, .)||) up to round-off.  Returns the least-squares slopes, the spread of r2 / e^2 and the
    ratio r2 / r1 at the smallest e."""
    R0, Jd = np.asarray(R0, float), np.asarray(Jd, float)
    e = np.asarray(eps, float)
    r1 = np.array([np.linalg.norm(np.asarray(R, float) - R0) for R in R_eps])
    r2 = np.array([np.linalg.norm(np.asarray(R, float) - R0 - ek * Jd) for R, ek in zip(R_eps, e)])
    s1 = float(np.polyfit(np.log(e), np.log(r1), 1)[0])
    s2 = float(np.polyfit(np.log(e), np.log(r2), 1)[0])
    q = r2 / e ** 2
    return {"slope_r1": s1, "slope_r2": s2, "r2_over_eps2_spread": float((q.max() - q.min()) / q.mean()),
            "r2_over_r1_min_eps": float(r2[-1] / r1[-1]), "r1": r1.tolist(), "r2": r2.tolist()}
