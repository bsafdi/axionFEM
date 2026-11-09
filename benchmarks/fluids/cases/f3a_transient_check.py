"""Transient Navier-Stokes (group F3a): the check module's error computation (runner.check_module "f3a_transient_check").

It imports f1a_stokes_check (the fixed mesh builders, the quadrature rule and the nodal Dirichlet checks).  The
case runner (cases/f3a_transient.py) builds the fixed meshes and time grids with the functions below,
solves through the public fluids API, wraps each stored time level in an Evaluator and calls the functions here.
Nothing in this module calls the solver: every norm, identity, effectivity and step statistic is computed here from
raw point evaluations and the step sizes the solver reports.

Evaluator protocol (as in f1a_stokes_check and f2a_ns_check):
    ev.n_elements                -> int
    ev.geometry(ref_pts)         -> (x[ne, nq, 2], detJ[ne, nq])         element map; (x, y) or (r, z)
    ev.velocity(ref_pts)         -> (u[ne, nq, nc], d[ne, nq, nc, 2])    nc = 2 planar (u_x, u_y), 3 axisymmetric
                                                                          (u_r, u_theta, u_z); d[..., i, j] = d u_i / d
                                                                          (x, y)_j, COMPONENT derivatives (no hoop terms)
    ev.pressure(ref_pts)         -> p[ne, nq]
Every evaluator of one run must use the same element order (checked: same geometry to 1e-14).

Conventions: rho (du/dt + (u.grad)u) - div sigma_form = f, div u = 0; sigma_lap = mu G - p I,
sigma_sym = 2 mu eps - p I; axisymmetric orthonormal-frame gradient G with the hoop entries; r-weighted integrals.
BDF of order k on the step sequence dt_n = t_n - t_{n-1}, omega_n = dt_n / dt_{n-1}:
    BDF1: (u^n - u^{n-1}) / dt_n
    BDF2: [ (1 + 2 w)/(1 + w) u^n - (1 + w) u^{n-1} + w^2/(1 + w) u^{n-2} ] / dt_n
Semi-implicit (linearly implicit) advecting velocity: BDF1 w = u^{n-1}; BDF2 w = (1 + omega) u^{n-1} - omega u^{n-2}.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import f1a_stokes_check as C1


def jv(n, z):
    """scipy.special.jv, imported on first use (only the Womersley pipe profile needs it)."""
    from scipy.special import jv as _jv
    return _jv(n, z)

QUAD_DEGREE = 10
tri_rule = C1.tri_rule
structured_mesh = C1.structured_mesh
verify_mesh = C1.verify_mesh
dirichlet_nodal_defect = C1.dirichlet_nodal_defect
dirichlet_shared_node_mismatch = C1.dirichlet_shared_node_mismatch
mesh_hash = C1.mesh_hash


# ============================================================================================== exact fields
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class TField:
    """Time-dependent exact solution / data of one generated field (tools/fluids_mms/out/f3a_tr_*.json)."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data["name"]
        self.geometry = data["geometry"]
        self.axi = self.geometry == "axisymmetric"
        self.coords = data["coords"]
        self.rho, self.mu = float(data["rho"]), float(data["mu"])
        self.nc = 3 if self.axi else 2
        self._u = [self._compile(s) for s in data["u"]]
        self._g = [[self._compile(s) for s in row] for row in data["grad_u"]]
        self._p = self._compile(data["p"])
        self._f = {k: [self._compile(s) for s in v["f"]] for k, v in data["forms"].items()}
        self._fv = {(k, var): [self._compile(s) for s in v[var]] for k, v in data["forms"].items()
                    for var in ("f_without_convection", "f_without_time") if var in v}
        self._sig = {k: [[self._compile(s) for s in row] for row in v["sigma"]] for k, v in data["forms"].items()
                     if "sigma" in v}

    def _compile(self, spec):
        code = compile(spec["numpy"], "<f3a expression>", "eval")
        names = self.coords

        def f(A, B, t):
            v = eval(code, {"__builtins__": {}, "np": np, "numpy": np, "jv": jv}, dict(zip(names, (A, B, t))))
            return np.broadcast_to(np.real(np.asarray(v, dtype=complex)).astype(float), np.shape(A))
        return f

    def u(self, A, B, t):
        return np.stack([c(A, B, t) for c in self._u], axis=-1)

    def grad_u(self, A, B, t):
        return np.stack([np.stack([g(A, B, t) for g in row], axis=-1) for row in self._g], axis=-2)

    def p(self, A, B, t):
        return self._p(A, B, t)

    def f(self, form, variant=None):
        fs = self._f[form] if variant is None else self._fv[(form, variant)]
        return lambda A, B, t: np.stack([c(A, B, t) for c in fs], axis=-1)

    def traction(self, form, normal):
        """g = sigma_form n (outward n, frame (x, y) or (r, theta, z) with n = (n_r, 0, n_z))."""
        S = self._sig[form]
        n = list(normal)
        if self.axi:
            n = [n[0], 0.0, n[1]]
        return lambda A, B, t: np.stack([sum(S[i][j](A, B, t) * n[j] for j in range(len(n)) if n[j] != 0)
                                         + 0.0 * A for i in range(len(n))], axis=-1)


def load_field(entry: dict, root: Path | None = None) -> TField:
    """entry = {"path": "tools/fluids_mms/out/f3a_tr_<name>.json", "sha256": ...} from the fixed run parameters."""
    root = Path(root) if root is not None else _repo_root()
    raw = (root / entry["path"]).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        raise RuntimeError(f"F3a data {entry['path']}: sha256 {got} != fixed {entry['sha256']}")
    return TField(json.loads(raw))


# ============================================================================================== meshes
def rect_mesh(nx: int, ny: int, k: int, x0: float, x1: float, y0: float, y1: float):
    """nx x ny squares on [x0, x1] x [y0, y1] split along the (1,1) diagonal except the cells at the corners (x1, y0)
    and (x0, y1) (no triangle with all vertices on the boundary), refined uniformly (red) k times.  Nested."""
    V, T = C1._structured(nx, ny, x0, x1, y0, y1)
    for _ in range(k):
        V, T = C1.red_refine(V, T)
    return np.asarray(V, float), np.asarray(T, np.int64)


def family_mesh(fam: dict, level: int):
    """The fixed families of run.parameters.meshes: {"builder": "structured", "N": [...]} or {"builder": "rect",
    "nx", "ny", "box", "k": [...]}; returns (V, T, h_nominal)."""
    if fam["builder"] == "structured":
        N = int(fam["N"][level])
        V, T = structured_mesh(N)
        return V, T, 1.0 / N
    if fam["builder"] == "rect":
        k = int(fam["k"][level])
        x0, x1, y0, y1 = fam["box"]
        V, T = rect_mesh(int(fam["nx"]), int(fam["ny"]), k, x0, x1, y0, y1)
        return V, T, float(fam["cell"]) / 2 ** k
    raise ValueError(fam)


def all_boundary_triangles(T) -> int:
    return C1.all_boundary_triangles(T)


# ============================================================================================== time grids
def time_grid(spec: dict, T_end: float, N: int) -> np.ndarray:
    """t_0 = 0 < t_1 < ... < t_N = T_end.  spec["kind"] = "uniform" or "cyclic" (step sizes proportional to the fixed
    pattern repeated N / len(pattern) times, scaled to sum to T_end: a step function)."""
    if spec["kind"] == "uniform":
        return np.linspace(0.0, T_end, N + 1)
    if spec["kind"] == "cyclic":
        pat = np.asarray(spec["pattern"], float)
        if N % len(pat):
            raise ValueError("N must be a multiple of the pattern length")
        h = np.tile(pat, N // len(pat))
        h = h * (T_end / h.sum())
        return np.concatenate([[0.0], np.cumsum(h)])
    raise ValueError(spec)


def step_stats(ts) -> dict:
    """Step statistics of a time grid (as the solver reports it): max ratio of successive steps, extremes."""
    ts = np.asarray(ts, float)
    h = np.diff(ts)
    if np.any(h <= 0):
        raise ValueError("time grid not increasing")
    rat = h[1:] / h[:-1] if len(h) > 1 else np.array([1.0])
    return {"n_steps": int(len(h)), "dt_max": float(h.max()), "dt_min": float(h.min()),
            "dt_max_ratio": float(rat.max()), "dt_min_ratio": float(rat.min()), "t_end": float(ts[-1])}


def bdf_coefficients(order: int, dts) -> list[float]:
    """alpha_0..alpha_k with  (sum_j alpha_j u^{n-j}) / dt_n  approximating du/dt(t_n); dts = (dt_n, dt_{n-1})."""
    if order == 1:
        return [1.0, -1.0]
    if order == 2:
        w = dts[0] / dts[1]
        return [(1 + 2 * w) / (1 + w), -(1 + w), w * w / (1 + w)]
    raise ValueError(order)


def extrapolation_coefficients(order: int, dts) -> list[float]:
    """Advecting velocity of the semi-implicit scheme: w = sum_j beta_j u^{n-j} (j >= 1)."""
    if order == 1:
        return [1.0]
    w = dts[0] / dts[1]
    return [1 + w, -w]


# ============================================================================================== quadrature helpers
def _weights(ev, axi: bool, deg=QUAD_DEGREE):
    q, w = tri_rule(deg)
    xq, detJ = ev.geometry(q)
    if np.any(detJ <= 0):
        raise RuntimeError("non-positive Jacobian in the element map")
    W = w[None, :] * detJ
    return q, xq, (W * xq[..., 0] if axi else W), W


def cyl_grad(u, d, R):
    """Orthonormal-frame gradient G (3x3) from component derivatives d[..., i, (r, z)] and values u (axisymmetric)."""
    ur, ut = u[..., 0], u[..., 1]
    G = np.zeros(u.shape[:-1] + (3, 3))
    G[..., 0, 0], G[..., 0, 1], G[..., 0, 2] = d[..., 0, 0], -ut / R, d[..., 0, 1]
    G[..., 1, 0], G[..., 1, 1], G[..., 1, 2] = d[..., 1, 0], ur / R, d[..., 1, 1]
    G[..., 2, 0], G[..., 2, 2] = d[..., 2, 0], d[..., 2, 1]
    return G


def _grad_full(u, d, xq, axi):
    return cyl_grad(u, d, xq[..., 0]) if axi else d


def _div(u, d, xq, axi):
    return d[..., 0, 0] + d[..., 2, 1] + u[..., 0] / xq[..., 0] if axi else d[..., 0, 0] + d[..., 1, 1]


# ============================================================================================== error norms
def errors(ev, fld: TField, t: float, pressure: str | None = None) -> dict:
    """Planar: uL2, uH1 (all gradient components), pL2 (mean-free or determined), divL2.  Axisymmetric:
    r-weighted uL2w, uH1w (CYLINDRICAL gradient, hoop terms included), pL2w, divL2w (the 3D norms / 2 pi) and the
    unweighted uL2_unw, uH1_unw (component derivatives), pL2_unw.  Mean-free pressures use the means of the same
    weight.  Exact solution evaluated at the quadrature points (never interpolated); degree-10 rule."""
    axi = fld.axi
    pressure = pressure or ("mean_free" if fld.data["pressure"] == "mean_zero" else "determined")
    q, xq, Wr, W1 = _weights(ev, axi)
    A, B = xq[..., 0], xq[..., 1]
    uh, dh = ev.velocity(q)
    ph = ev.pressure(q)
    ue, de = fld.u(A, B, t), fld.grad_u(A, B, t)
    pe = fld.p(A, B, t)
    Gh, Ge = _grad_full(uh, dh, xq, axi), _grad_full(ue, de, xq, axi)
    out = {"u_magnitude": float(np.max(np.linalg.norm(ue, axis=-1))), "t": float(t)}

    def pnorm(Wt):
        if pressure == "mean_free":
            a = Wt.sum()
            e = (ph - (Wt * ph).sum() / a) - (pe - (Wt * pe).sum() / a)
            mag = float(np.max(np.abs(pe - (Wt * pe).sum() / a)))
        else:
            e = ph - pe
            mag = float(np.max(np.abs(pe)))
        return math.sqrt(float((Wt * e ** 2).sum())), mag
    sfx = "w" if axi else ""
    out["uL2" + sfx] = math.sqrt(float((Wr[..., None] * (uh - ue) ** 2).sum()))
    out["uH1" + sfx] = math.sqrt(float((Wr[..., None, None] * (Gh - Ge) ** 2).sum()))
    out["pL2" + sfx], out["p_magnitude"] = pnorm(Wr)
    out["divL2" + sfx] = math.sqrt(float((Wr * _div(uh, dh, xq, axi) ** 2).sum()))
    out["uL2_norm" + sfx] = math.sqrt(float((Wr[..., None] * ue ** 2).sum()))
    if axi:
        out["uL2_unw"] = math.sqrt(float((W1[..., None] * (uh - ue) ** 2).sum()))
        out["uH1_unw"] = math.sqrt(float((W1[..., None, None] * (dh - de) ** 2).sum()))
        out["pL2_unw"], _ = pnorm(W1)
    return out


def same_mesh(ev_a, ev_b) -> bool:
    q, _ = tri_rule(4)
    xa, _ = ev_a.geometry(q)
    xb, _ = ev_b.geometry(q)
    return xa.shape == xb.shape and float(np.max(np.abs(xa - xb))) <= 1e-14


def l2_inner(ev_a, ev_b, axi: bool) -> float:
    """(u_a, u_b)_L2 (r-weighted in axisymmetry) of two velocity fields on the same mesh."""
    if not same_mesh(ev_a, ev_b):
        raise RuntimeError("evaluators on different meshes / element orders")
    q, xq, Wr, _ = _weights(ev_a, axi)
    ua, _ = ev_a.velocity(q)
    ub, _ = ev_b.velocity(q)
    return float((Wr[..., None] * ua * ub).sum())


# ============================================================================================== discrete divergence
def _p1_vertex_map(ev, V):
    """Global vertex ids of every element, matched by coordinates of the element map at the reference vertices."""
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    xv, _ = ev.geometry(ref)
    V = np.asarray(V, float)
    key = {(round(a, 10), round(b, 10)): i for i, (a, b) in enumerate(V)}
    ids = np.empty((xv.shape[0], 3), np.int64)
    for e in range(xv.shape[0]):
        for k in range(3):
            ids[e, k] = key[(round(float(xv[e, k, 0]), 10), round(float(xv[e, k, 1]), 10))]
    return ids


def discrete_divergence(ev, V, axi: bool) -> dict:
    """Dual norm of the discrete divergence of u_h against the P1 pressure space of the fixed mesh (vertices V):
    b_i = (div u_h, phi_i) (r-weighted, cylindrical divergence in axisymmetry), ||b||_* = (b^T M_p^-1 b)^1/2 = the L2
    norm of the L2 projection of div u_h onto P1 (consistent, exactly integrated r-weighted mass).  Zero (to
    round-off) iff u_h is discretely divergence-free, the property the projection of the initial data must give.
    Reported relative to |u_h|_H1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    ids = _p1_vertex_map(ev, V)
    q, xq, Wr, _ = _weights(ev, axi)
    uh, dh = ev.velocity(q)
    phi = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], axis=-1)              # [nq, 3]
    dv = _div(uh, dh, xq, axi)
    nv = len(V)
    b = np.zeros(nv)
    np.add.at(b, ids, np.einsum("eq,eq,qk->ek", Wr, dv, phi))
    Me = np.einsum("eq,qk,ql->ekl", Wr, phi, phi)
    rows = np.repeat(ids, 3, axis=1).ravel()
    cols = np.tile(ids, (1, 3)).ravel()
    M = coo_matrix((Me.ravel(), (rows, cols)), shape=(nv, nv)).tocsc()
    dual = math.sqrt(max(float(b @ spsolve(M, b)), 0.0))
    G = _grad_full(uh, dh, xq, axi)
    h1 = math.sqrt(float((Wr[..., None, None] * G ** 2).sum()))
    return {"div_dual": dual, "u_h1": h1, "div_dual_rel": dual / max(h1, 1e-300)}


# ============================================================================================== energies
def kinetic_energy(ev, rho: float, axi: bool) -> float:
    q, xq, Wr, _ = _weights(ev, axi)
    uh, _ = ev.velocity(q)
    return 0.5 * rho * float((Wr[..., None] * uh ** 2).sum())


def flow_functionals(ev, mu: float, form: str, axi: bool = False) -> dict:
    """Ekin = 1/2 ||u_h||^2, Enst = ||omega_h||^2 (planar vorticity), Diss = mu ||G||^2 (lap) or 2 mu ||eps||^2 (sym):
    the cross-code quantities (planar)."""
    q, xq, Wr, _ = _weights(ev, axi)
    uh, dh = ev.velocity(q)
    G = _grad_full(uh, dh, xq, axi)
    om = dh[..., 1, 0] - dh[..., 0, 1]
    if form == "lap":
        dis = mu * float((Wr[..., None, None] * G ** 2).sum())
    else:
        E = 0.5 * (G + np.swapaxes(G, -1, -2))
        dis = 2 * mu * float((Wr[..., None, None] * E ** 2).sum())
    return {"Ekin": 0.5 * float((Wr[..., None] * uh ** 2).sum()), "Enst": float((Wr * om ** 2).sum()), "Diss": dis}


def _visc_energy(G, mu, form):
    if form == "lap":
        return mu * (G ** 2).sum(axis=(-1, -2))
    E = 0.5 * (G + np.swapaxes(G, -1, -2))
    return 2 * mu * (E ** 2).sum(axis=(-1, -2))


def step_energy_identity(evs, ts, n: int, order: int, fld: TField, form: str, scheme: str,
                         alpha=None, f_shift: int = 0, rho_mass: float | None = None) -> dict:
    """Momentum equation of step n tested with v = u^n (homogeneous Dirichlet data, (p^n, div u^n) = 0 because
    p^n is a pressure test function):
        rho/dt_n (sum_j alpha_j u^{n-j}, u^n) + a_form(u^n, u^n) + c(w; u^n, u^n) - (f(t_n), u^n) = 0
    up to the solver tolerance (planar: algebraic; axisymmetric: up to the implementation's quadrature of the rational hoop terms).  c(w; u, v) = rho ((w.grad) u, v) (standard form, F2a), w = u^n (Newton per
    step) or the extrapolation (semi-implicit).  alpha, f_shift (f at t_{n - f_shift}) and rho_mass let the caller
    evaluate the WRONG schemes (positive controls).  Returns defect, scale (sum of |terms|) and the terms."""
    axi = fld.axi
    evn = evs[n]
    for j in range(1, order + 1):
        if not same_mesh(evn, evs[n - j]):
            raise RuntimeError("history on a different mesh")
    q, xq, Wr, _ = _weights(evn, axi)
    A, B = xq[..., 0], xq[..., 1]
    dts = [ts[n] - ts[n - 1]] + ([ts[n - 1] - ts[n - 2]] if order == 2 else [])
    al = alpha if alpha is not None else bdf_coefficients(order, dts)
    rho = fld.rho
    rm = rho if rho_mass is None else rho_mass
    U = [evs[n - j].velocity(q) for j in range(0, len(al))]
    un, dn = U[0]
    Gn = _grad_full(un, dn, xq, axi)
    dudt = sum(a * U[j][0] for j, a in enumerate(al)) / dts[0]
    t_time = rm * float((Wr[..., None] * dudt * un).sum())
    t_visc = float((Wr * _visc_energy(Gn, fld.mu, form)).sum())
    if scheme == "newton":
        w = un
    else:
        be = extrapolation_coefficients(order, dts)
        w = sum(b * evs[n - 1 - j].velocity(q)[0] for j, b in enumerate(be))
    conv = np.einsum("eqij,eqj->eqi", Gn, w)
    t_conv = rho * float((Wr[..., None] * conv * un).sum())
    fv = fld.f(form)(A, B, ts[n - f_shift])
    t_f = float((Wr[..., None] * fv * un).sum())
    d = t_time + t_visc + t_conv - t_f
    sc = abs(t_time) + abs(t_visc) + abs(t_conv) + abs(t_f)
    return {"defect": d, "scale": sc, "rel": abs(d) / max(sc, 1e-300), "time": t_time, "visc": t_visc,
            "conv": t_conv, "load": t_f}


def run_energy_identity(evs, ts, order: int, fld: TField, form: str, scheme: str, start_order1: bool = True) -> dict:
    """Max over the steps of the relative identity defect, with the positive controls the check must see: the same
    sequence judged with the other BDF order's coefficients, with f at t_{n-1}, and with rho = 1 in the mass term."""
    n0 = 2 if order == 2 else 1
    worst, ctrl = 0.0, {"other_order": math.inf, "f_lag": math.inf, "rho_one": math.inf}
    rows = []
    if order == 2 and start_order1:
        r1 = step_energy_identity(evs, ts, 1, 1, fld, form, scheme)
        worst = max(worst, r1["rel"])
    for n in range(n0, len(evs)):
        r = step_energy_identity(evs, ts, n, order, fld, form, scheme)
        worst = max(worst, r["rel"])
        rows.append(r)
        if r["scale"] <= 1e-12 * max([x["scale"] for x in rows] + [1e-300]):
            continue                    # a zero step (e.g. zero force and zero history) cannot show a control
        o2 = 1 if order == 2 else 2
        if n >= o2:
            ro = step_energy_identity(evs, ts, n, o2, fld, form, scheme)
            ctrl["other_order"] = min(ctrl["other_order"], ro["rel"])
        rf = step_energy_identity(evs, ts, n, order, fld, form, scheme, f_shift=1)
        ctrl["f_lag"] = min(ctrl["f_lag"], rf["rel"])
        if abs(fld.rho - 1.0) > 1e-12:
            rr = step_energy_identity(evs, ts, n, order, fld, form, scheme, rho_mass=1.0)
            ctrl["rho_one"] = min(ctrl["rho_one"], rr["rel"])
    ke = [kinetic_energy(e, fld.rho, fld.axi) for e in evs]
    return {"defect_rel_max": worst, "controls_min": ctrl, "kinetic_energy": ke, "steps": len(evs) - 1}


# ============================================================================================== error control
def effectivity(ev_est, ev_un, fld: TField, t_n: float) -> dict:
    """est = the solver's estimate of the local error of the ADVANCED velocity (one step from the exact history);
    e = u^n - u(t_n) (space-exact fields: u(t_n) is in the discrete space, so e is the true local error).
    effectivity = (est, e) / (e, e) in the (r-weighted) L2 inner product: 1 for an asymptotically exact estimator
 (T2); also ||est|| / ||e||, and est_L2 = ||est||_L2 ((r-weighted) L2 norm of the estimate,
    the number a solver's REPORTED error norm must equal: T2 `est_norm_ratio`)."""
    axi = fld.axi
    q, xq, Wr, _ = _weights(ev_un, axi)
    if not same_mesh(ev_est, ev_un):
        raise RuntimeError("estimate on another mesh")
    e = ev_un.velocity(q)[0] - fld.u(xq[..., 0], xq[..., 1], t_n)
    d = ev_est.velocity(q)[0]
    ee = float((Wr[..., None] * e * e).sum())
    de = float((Wr[..., None] * d * e).sum())
    dd = float((Wr[..., None] * d * d).sum())
    return {"effectivity": de / max(ee, 1e-300), "norm_ratio": math.sqrt(dd / max(ee, 1e-300)),
            "local_error_L2": math.sqrt(ee), "est_L2": math.sqrt(dd)}


def solution_scale(fld: TField, T_end: float, n_t: int = 41, n_xy: int = 33) -> float:
    """S_u = max over t in [0, T_end] of ||u(t)||_L2 (r-weighted), by a tensor midpoint rule: the fixed scale of the
    error-control tolerance (atol = rtol S_u) and of the relative global errors."""
    x0, x1, y0, y1 = fld.data["box"]
    a = x0 + (x1 - x0) * (np.arange(n_xy) + 0.5) / n_xy
    b = y0 + (y1 - y0) * (np.arange(n_xy) + 0.5) / n_xy
    A, B = np.meshgrid(a, b, indexing="ij")
    w = (x1 - x0) * (y1 - y0) / n_xy ** 2 * (A if fld.axi else np.ones_like(A))
    best = 0.0
    for tt in np.linspace(0, T_end, n_t):
        best = max(best, math.sqrt(float((w[..., None] * fld.u(A, B, tt) ** 2).sum())))
    return best


# ============================================================================================== order helper (diagnostic)
def pairwise_orders(x, e) -> list[float]:
    return [math.log(e[i] / e[i + 1]) / math.log(x[i] / x[i + 1]) for i in range(len(e) - 1)]
