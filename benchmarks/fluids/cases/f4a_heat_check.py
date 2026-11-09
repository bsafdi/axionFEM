"""Heat transfer in fluids (group F4a: scalar transport with SUPG; Boussinesq; axisymmetric): the check module's error
computation (runner.check_module "f4a_heat_check").

It imports scalar_transport_check (self-contained).  The
runner (cases/f4a_heat.py) builds the fixed meshes with the builders of scalar_transport_check, solves through the
public fluids API, wraps the discrete solution in an Evaluator and calls the functions here.  Nothing here calls the
solver.

Evaluator protocol, in addition to scalar_transport_check's (geometry, scalar = T):
  planar        ev.velocity(ref) -> (u[ne,nq,2], G[ne,nq,2,2]),  G[...,i,j] = d u_i / d x_j;  ev.pressure(ref) -> p[ne,nq]
  axisymmetric  ev.velocity(ref) -> (u[ne,nq,3], D[ne,nq,3,2]),  u = (u_r, u_theta, u_z), D[...,i,j] = d u_i / d(r,z)_j;
                ev.pressure(ref) -> p[ne,nq]
Boussinesq: rho0 (u.grad)u - div sigma_form = f_u + rho0 beta g (T - T_ref) e_up, div u = 0,
rho0 c_p u.grad T - div(k grad T) = Q; e_up = e_y (planar) or e_z (axisymmetric).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import scalar_transport_check as S

QUAD_DEGREE = S.QUAD_DEGREE
tri_rule = S.tri_rule
box_mesh = S.box_mesh
distorted_mesh = S.distorted_mesh
verify_mesh = S.verify_mesh
mesh_hash = S.mesh_hash
all_boundary_triangles = S.all_boundary_triangles
scalar_norms = S.scalar_norms
sampled_extrema = S.sampled_extrema
flux_from_residual = S.flux_from_residual
nodal_dirichlet_defect = S.nodal_dirichlet_defect
taylor_test = S.taylor_test
load_scalar_field = S.load_field
boundary_integral = S.boundary_integral
graetz_wall_flux_exact = S.graetz_wall_flux_exact
exact_boundary_flux = S.exact_boundary_flux


# ============================================================================================== Boussinesq fields
class BousField:
    """A generated Boussinesq field (kind boussinesq_mms | hydrostatic), sha256-verified by load_bous."""

    def __init__(self, data: dict):
        self.data = data
        self.name = data["name"]
        self.kind = data["kind"]
        self.geometry = data["geometry"]
        self.coords = data["coords"]
        cc = self.coords
        c = lambda s: S.compile_expr(s, cc)  # noqa: E731
        self._u = [c(s) for s in data["u"]]
        self._gu = [[c(s) for s in row] for row in data["grad_u"]]
        self.p = c(data["p"])
        self.mu = c(data["mu"])
        self._T = c(data["T"])
        self._gT = [c(s) for s in data["grad_T"]]
        self.k = c(data["k"])
        self.rho0 = float(data["rho0"]["value"])
        self.cp = float(data["cp"]["value"])
        self.beta_g = float(data["beta_g"]["value"])
        self.T_ref = float(data["T_ref"]["value"])
        self._f = {form: [c(s) for s in fd["f"]] for form, fd in data["forms"].items()}
        self.Q = c(data["Q"])
        self.T_boundaries = {bd["name"]: bd for bd in data["T_boundaries"]}
        self.ncomp = len(self._u)

    def u(self, A, B):
        return np.stack([g(A, B) for g in self._u], -1)

    def grad_u(self, A, B):
        return np.stack([np.stack([g(A, B) for g in row], -1) for row in self._gu], -2)

    def T(self, A, B):
        return self._T(A, B)

    def grad_T(self, A, B):
        return np.stack([g(A, B) for g in self._gT], -1)

    def f(self, form):
        fs = self._f[form]
        return lambda A, B: np.stack([g(A, B) for g in fs], -1)


def load_bous(entry: dict, root: Path | None = None) -> BousField:
    return BousField(S.load_json(entry, root))


def _cyl_grad(u, D, r):
    """Full cylindrical gradient (3 x 3, columns d/dr, (1/r) d/dtheta, d/dz) from components and their (r, z)
    derivatives: hoop entries -u_theta / r (r row) and u_r / r (theta row)."""
    G = np.zeros(u.shape[:-1] + (3, 3))
    G[..., :, 0] = D[..., :, 0]
    G[..., :, 2] = D[..., :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        G[..., 0, 1] = np.where(r > 0, -u[..., 1] / r, -D[..., 1, 0])
        G[..., 1, 1] = np.where(r > 0, u[..., 0] / r, D[..., 0, 0])
    return G


def flow_norms(ev, fld: BousField, pressure: str = "mean_free") -> dict:
    """Velocity, pressure and temperature errors of a Boussinesq solution.
    planar: uL2, uH1, pL2(m), divL2, TL2, TH1;  axisymmetric: r-weighted uL2w, uH1w (cylindrical gradient, hoop terms
    included), pL2(m)w, divL2w, TL2w, TH1w and unweighted uL2_unw, uH1_unw, TL2_unw, TH1_unw.  Mean-free pressures use
    the (r-weighted) means.  Exact solution at the quadrature points, never interpolated."""
    axi = fld.geometry == "axisymmetric"
    q, X, W, Wr = S._weights(ev, axi)
    A, B = X[..., 0], X[..., 1]
    uh, Dh = ev.velocity(q)
    ph = ev.pressure(q)
    Th, dTh = ev.scalar(q)
    ue, De = fld.u(A, B), fld.grad_u(A, B)
    pe, Te, dTe = fld.p(A, B), fld.T(A, B), fld.grad_T(A, B)
    out = {}
    if axi:
        Gh, Ge = _cyl_grad(uh, Dh, A), _cyl_grad(ue, De, A)
        divh = Gh[..., 0, 0] + Gh[..., 1, 1] + Gh[..., 2, 2]
    else:
        Gh, Ge = Dh, De
        divh = Dh[..., 0, 0] + Dh[..., 1, 1]
    meas = Wr.sum()
    sfx = "w" if axi else ""
    out["uL2" + sfx] = math.sqrt(float((Wr[..., None] * (uh - ue) ** 2).sum()))
    out["uH1" + sfx] = math.sqrt(float((Wr[..., None, None] * (Gh - Ge) ** 2).sum()))
    out["divL2" + sfx] = math.sqrt(float((Wr * divh ** 2).sum()))
    out["TL2" + sfx] = math.sqrt(float((Wr * (Th - Te) ** 2).sum()))
    out["TH1" + sfx] = math.sqrt(float((Wr[..., None] * (dTh - dTe) ** 2).sum()))
    pmh, pme = float((Wr * ph).sum() / meas), float((Wr * pe).sum() / meas)
    if pressure == "mean_free":
        out["pL2m" + sfx] = math.sqrt(float((Wr * ((ph - pmh) - (pe - pme)) ** 2).sum()))
    else:
        out["pL2" + sfx] = math.sqrt(float((Wr * (ph - pe) ** 2).sum()))
    if axi:
        out["uL2_unw"] = math.sqrt(float((W[..., None] * (uh - ue) ** 2).sum()))
        out["uH1_unw"] = math.sqrt(float((W[..., None, None] * (Dh - De) ** 2).sum()))
        out["TL2_unw"] = math.sqrt(float((W * (Th - Te) ** 2).sum()))
        out["TH1_unw"] = math.sqrt(float((W[..., None] * (dTh - dTe) ** 2).sum()))
    out["u_magnitude"] = float(np.max(np.linalg.norm(ue, axis=-1)))
    out["p_magnitude"] = float(np.max(np.abs(pe - pme)))
    out["T_magnitude"] = float(np.max(np.abs(Te)))
    out["gradu_magnitude"] = float(np.max(np.sqrt((Ge ** 2).sum(axis=(-2, -1)))))
    out["gradT_magnitude"] = float(np.max(np.linalg.norm(dTe, axis=-1)))
    out["measure"] = float(meas)
    # solution_magnitude of every series: max of the exact quantity measured
    base = {"uL2": "u_magnitude", "uH1": "gradu_magnitude", "divL2": "gradu_magnitude", "pL2m": "p_magnitude",
            "pL2": "p_magnitude", "TL2": "T_magnitude", "TH1": "gradT_magnitude"}
    out["magnitudes"] = {}
    for k in list(out):
        b = k[:-4] if k.endswith("_unw") else (k[:-1] if k.endswith("w") else k)
        if b in base:
            out["magnitudes"][k] = out[base[b]]
    return out


def hydrostatic_velocity_scale(fld: BousField, length: float = 1.0) -> float:
    """The declared velocity scale of the hydrostatic no-flow problem (its exact velocity is 0): U = rho0 beta g
    |Delta T| L^2 / mu, Delta T the imposed vertical temperature difference (the buoyancy-driven velocity the problem would have without the hydrostatic balance).  For the fields of this design
    (rho0 = mu = 1, beta g = 1000, Delta T = 1) U = 1000."""
    tb = float(fld.T(np.array([0.5]), np.array([0.0]))[0])
    tt = float(fld.T(np.array([0.5]), np.array([1.0]))[0])
    mu = float(fld.mu(np.array([0.5]), np.array([0.5]))[0])
    return fld.rho0 * fld.beta_g * abs(tt - tb) * length ** 2 / mu


def discrete_velocity_norms(ev, axisymmetric: bool, *, u_scale: float) -> dict:
    """||u_h|| and |u_h|_1 (the exact velocity is zero: hydrostatic no-flow) and the sampled max |u_h| at the P2
    nodes; axisymmetric: r-weighted (cylindrical gradient).  `magnitudes` reports u_scale for both series: it is a
    REQUIRED keyword, to be given as hydrostatic_velocity_scale(fld); the runner reports
    series.solution_magnitude = magnitudes[key] of the finest level."""
    q, X, W, Wr = S._weights(ev, axisymmetric)
    uh, Dh = ev.velocity(q)
    G = _cyl_grad(uh, Dh, X[..., 0]) if axisymmetric else Dh
    un, _ = ev.velocity(S.P2_NODES)
    return {"uL2": math.sqrt(float((Wr[..., None] * uh ** 2).sum())),
            "uH1": math.sqrt(float((Wr[..., None, None] * G ** 2).sum())),
            "u_nodal_max": float(np.max(np.linalg.norm(un, axis=-1))),
            "magnitudes": {"uL2": float(u_scale), "uH1": float(u_scale)}}


# ============================================================================================== cavity functionals
def cavity_functionals(ev, axisymmetric: bool = False, k: float = 1.0, a: float = 1.0) -> dict:
    """Integral functionals of a cavity solution (all by element quadrature of degree 10, exact for the polynomial
    integrands of P2 / P1 on affine elements):
      Ekin = 1/2 int |u_h|^2;  Diss = int |grad u_h|^2;  Tsq = int T_h^2;  vT = int v_h T_h (vertical heat
      transport, v = the e_up component);  vleft = int_{c0 < 1/2} v_h (sign of the circulation next to the c0 = 0 wall);
      adv = int a u_h.grad T_h (the discrete advection integral of the convective form: equals minus the sum of all boundary fluxes for the convective form);  umax_nodal (sampled at the P2 nodes)."""
    q, X, W, Wr = S._weights(ev, axisymmetric)
    uh, Dh = ev.velocity(q)
    Th, dTh = ev.scalar(q)
    up = 1 if not axisymmetric else 2
    G = _cyl_grad(uh, Dh, X[..., 0]) if axisymmetric else Dh
    left = (X.mean(axis=1)[:, 0] < 0.5)
    mer = uh[..., [0, 2]] if axisymmetric else uh
    un, _ = ev.velocity(S.P2_NODES)
    return {"Ekin": 0.5 * float((Wr[..., None] * uh ** 2).sum()),
            "Diss": float((Wr[..., None, None] * G ** 2).sum()),
            "Tsq": float((Wr * Th ** 2).sum()),
            "vT": float((Wr * uh[..., up] * Th).sum()),
            "vleft": float((Wr * uh[..., up])[left].sum()),
            "adv": float((Wr * a * (mer * dTh).sum(-1)).sum()),
            "umax_nodal": float(np.max(np.linalg.norm(un, axis=-1)))}


def nusselt(q_in_hot: float, k: float, dT: float, height: float = 1.0, length: float = 1.0) -> float:
    """Average Nusselt number of a wall: Nu = q_in / (k dT height / length), q_in the heat entering the fluid through
    the wall (per unit depth; = - outward flux from the residual)."""
    return q_in_hot / (k * dT * height / length)


def balance_identity(q_out_list, adv: float, source: float = 0.0, natural_out: float = 0.0) -> float:
    """Discrete energy identity of the CONVECTIVE form (test function v = 1): sum of the variational outward fluxes
    through the Dirichlet parts + the natural (Neumann/Robin) outward fluxes = int Q - int a u_h.grad T_h.  Returns
    the defect |sum q_out + natural_out - source + adv| / max(|adv|, sum |q_out|) (relative; a round-off quantity)."""
    s = float(sum(q_out_list)) + natural_out - source + adv
    scale = max(abs(adv), float(sum(abs(v) for v in q_out_list)), 1e-300)
    return abs(s) / scale


# ============================================================================================== Newton histories
def newton_iterations(history) -> int:
    """Number of Newton steps in a solver.history list of {kind, residual, ...}."""
    return int(sum(1 for h in history if h.get("kind") == "newton"))


def newton_order_median(history, floor: float = 1e-13) -> float:
    """Median of p_k = log(r_{k+1}/r_k) / log(r_k/r_{k-1}) over consecutive Newton residuals above the floor
    (quadratic convergence gives ~2; a Picard-quality Jacobian ~1)."""
    rs = [h["residual"] for h in history if h.get("kind") == "newton" and h["residual"] > floor]
    ps = []
    for i in range(1, len(rs) - 1):
        a, b = math.log(rs[i] / rs[i - 1]), math.log(rs[i + 1] / rs[i])
        if a < 0 and b < 0:
            ps.append(b / a)
    return float(np.median(ps)) if ps else float("nan")
