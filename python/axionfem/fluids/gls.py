"""Stabilized equal-order P1/P1 incompressible flow, planar (beta): residual-based GLS = SUPG + PSPG + LSIC.

The discrete problem:

    R_u,a = sum_K int [ rho (u_t + (w.grad)u).N_a e_i + a_visc(mu; u)(N_a e_i) - p d_i N_a - f_i N_a
                       + tau_supg rho ((w.grad N_a) R_i) + tau_c rho (div u) d_i N_a ]
    R_p,a = sum_K int [ (div u) N_a + (tau_pspg / rho) grad N_a . R ]
    R = rho (u_t + (w.grad)u) + grad p - div sigma_visc(u; mu) - f                    (strong residual; grad mu kept)
    tau_supg = [ (2/dt)^2 + CG w.G w + C_I (mu/rho)^2 G:G ]^(-1/2),  tau_pspg = [ CG w.G w + C_I (mu/rho)^2 G:G ]^(-1/2)
    tau_c = 1 / (tau_pspg tr G);  G of the EQUILATERAL reference triangle;  C_I = 36, CG = 1;  6-point degree-4 rule

DYNAMIC units (mu the dynamic viscosity, rho a constant density, p the physical pressure, f a force density). Arrays are in
the order of the caller's mesh (vertices, triangles are used as given). Coefficients are vectorized callables evaluated at
the fixed rule points of every element (`rule_points()`, followed by the centroid):

    mu(x, y) -> (mu, grad_mu)    arrays [ne, 7] and [ne, 7, 2]      (or mu a number, or mu_nodal = P1 nodal array [nv])
    force(x, y) -> [ne, 7, 2];  advect(x, y) -> [ne, 7, 2]  (mode "oseen")

A nodal viscosity is the P1 interpolant with its element-wise gradient.

    prob = gls_problem(V, T, mu=0.01, rho=1.0, force=f, dirichlet=(nodes, values), solve=True)
    prob.diagnostics["converged"], prob.u(), prob.p()          # u [nv, 2], p [nv] (mean-free)
    prob.residual_full(u, p)                                  # all rows, no elimination: (R_u [nv,2], R_p [nv])
    prob.tau_at(w_nodal)                                      # tau_supg, tau_pspg, tau_lsic [ne, 6]
    x = prob.state(); prob.residual(x); prob.jacobian_action(x, d)    # free-DOF API (Taylor tests)

Transient: one implicit step per problem (dt > 0, bdf 1 or 2, history = (u^{n-1}[, u^{n-2}]) nodal arrays). Planar only:
geometry="axisymmetric" is refused. Direct sparse LU. Enclosed flows ONLY: the constant pressure
is removed by a mean-value multiplier (pressure mean-free in the c-weighted sense, c_a = int N_a); the solve REFUSES (raises)
unless every boundary vertex is Dirichlet (open/natural boundaries are not implemented).
`stabilization={"supg": .., "pspg": .., "lsic": ..}` switches terms off for the NEGATIVE CONTROLS only (the
unstabilized P1/P1 pair is singular; a failed solve is reported, never rescued).
"""
from __future__ import annotations

import numpy as np

from .. import _core

_f = _core._fluids


def rule_points():
    """The fixed 6-point degree-4 rule on (0,0),(1,0),(0,1): (points [6,2], weights [6] summing to 1/2)."""
    pts, w = _f.gls_rule()
    return np.asarray(pts), np.asarray(w)


def _shape_values():
    pts, _ = rule_points()
    P = np.vstack([pts, [[1.0 / 3.0, 1.0 / 3.0]]])
    return np.stack([1 - P[:, 0] - P[:, 1], P[:, 0], P[:, 1]], axis=-1)          # [7, 3]


def quadrature_coordinates(V, T):
    """Physical coordinates [ne, 7, 2] of the 6 rule points (in the element's own vertex order) and the centroid."""
    V = np.asarray(V, float)
    T = np.asarray(T)
    return np.einsum("qa,eaj->eqj", _shape_values(), V[T])


class GlsSolution:
    """A GLS problem (and its solve, if requested): thin wrapper of the compiled object."""

    def __init__(self, core, V, T, mode, form, dt, bdf):
        self._c = core
        self.n_vertices, self.n_elements = len(V), len(T)
        self.mode, self.form, self.dt, self.bdf = mode, form, dt, bdf

    @property
    def diagnostics(self):
        return self._c.diagnostics

    @property
    def history(self):
        return self._c.history

    def u(self, allow_unconverged=False):
        if not allow_unconverged and not self.diagnostics["converged"]:
            raise _core.AxionFEMError("the solve did not converge: fields are refused "
                                      f"({self.diagnostics['failure']}); pass allow_unconverged=True to read them")
        return np.asarray(self._c.u())

    def p(self, allow_unconverged=False):
        if not allow_unconverged and not self.diagnostics["converged"]:
            raise _core.AxionFEMError("the solve did not converge: fields are refused "
                                      f"({self.diagnostics['failure']}); pass allow_unconverged=True to read them")
        return np.asarray(self._c.p())

    def dirichlet_nodes(self):
        return np.asarray(self._c.dirichlet_nodes())

    def residual_full(self, u, p, viscosity_scale=1.0):
        return self._c.residual_full(_f64(u), _f64(p), float(viscosity_scale))

    def layout(self):
        return tuple(self._c.layout())

    def state(self):
        return np.asarray(self._c.state())

    def residual(self, x, viscosity_scale=1.0):
        return np.asarray(self._c.residual(_f64(x), float(viscosity_scale)))

    def jacobian_action(self, x, d, viscosity_scale=1.0):
        return np.asarray(self._c.jacobian_action(_f64(x), _f64(d), float(viscosity_scale)))

    def tau_at(self, w_nodal):
        a, b, c = self._c.tau_at(_f64(w_nodal))
        return {"tau_supg": np.asarray(a), "tau_pspg": np.asarray(b), "tau_lsic": np.asarray(c)}

    def dual_norm(self, ru_free, rp):
        return self._c.dual_norm(_f64(ru_free), _f64(rp))


def _f64(a):
    return np.ascontiguousarray(a, dtype=float)


def gls_problem(vertices, triangles, *, mu=None, mu_nodal=None, rho=1.0, force=None, advect=None, mode="navier_stokes",
                form="stress", dt=0.0, bdf=1, history=(), dirichlet=None, stabilization=None, lu=None, lu_threads=1,
                newton_rtol=1e-12, newton_maxit=40, line_search=True, viscosity_scales=None, continuation_re=None,
                re=None, geometry="planar", solve=True):
    """See the module docstring. mode "stokes" (w = 0: no convective and no SUPG term), "oseen" (w = advect) or
    "navier_stokes" (w = u). form "stress" (2 mu eps:eps) | "laplacian" (mu grad u : grad v). dirichlet = (nodes [m],
    values [m, 2]): the nodal velocity data. Never raises on non-convergence (check diagnostics["converged"]); a singular
    or failed linear solve is reported as a failure, never rescued."""
    from . import default_lu
    if geometry != "planar":
        raise ValueError("axisymmetric stabilized P1/P1 is not implemented (the GLS solver is planar)")
    if mode not in ("stokes", "oseen", "navier_stokes"):
        raise ValueError(f"mode must be stokes, oseen or navier_stokes, got {mode!r}")
    if form not in ("stress", "laplacian"):
        raise ValueError(f"form must be stress or laplacian, got {form!r}")
    if dt and dt < 0:
        raise ValueError("dt must be >= 0")
    V = _f64(vertices)
    T = np.ascontiguousarray(triangles, dtype=np.int32)
    X = quadrature_coordinates(V, T)
    ne = len(T)
    mu_q = gmu_q = None
    if mu_nodal is not None:
        mu_nodal = _f64(mu_nodal)
        if mu_nodal.shape != (len(V),) or not (np.all(np.isfinite(mu_nodal)) and np.all(mu_nodal >= 0)):
            raise ValueError("mu_nodal must be a finite, non-negative array of one value per vertex")
        if mu is not None:
            raise ValueError("give mu or mu_nodal, not both")
    elif mu is None:
        raise ValueError("a viscosity is required (mu number or callable, or mu_nodal)")
    elif callable(mu):
        m, gm = mu(X[..., 0], X[..., 1])
        mu_q = _f64(np.broadcast_to(np.asarray(m, float), (ne, 7)))
        gmu_q = _f64(np.broadcast_to(np.asarray(gm, float), (ne, 7, 2)))
    else:
        mu_q = np.full((ne, 7), float(mu))
        gmu_q = np.zeros((ne, 7, 2))
    if mu_q is not None and not (np.all(np.isfinite(mu_q)) and np.all(mu_q >= 0)):
        raise ValueError("the viscosity must be finite and non-negative at every quadrature point")
    f_q = None
    if force is not None:
        f_q = _f64(np.broadcast_to(np.asarray(force(X[..., 0], X[..., 1]), float), (ne, 7, 2)))
    adv_q = None
    if mode == "oseen":
        if advect is None:
            raise ValueError("mode 'oseen' needs advect(x, y)")
        adv_q = _f64(np.broadcast_to(np.asarray(advect(X[..., 0], X[..., 1]), float), (ne, 7, 2)))
    elif advect is not None:
        raise ValueError("advect is only used in mode 'oseen'")
    hist = [_f64(h) for h in history]
    if dt and len(hist) < (1 if bdf == 1 else 2):
        raise ValueError(f"BDF{bdf} needs {1 if bdf == 1 else 2} history array(s)")
    dn = dv = None
    if dirichlet is not None:
        dn = np.ascontiguousarray(dirichlet[0], dtype=np.int32)
        dv = _f64(dirichlet[1])
    st = {"supg": True, "pspg": True, "lsic": True}
    unknown = sorted(set(stabilization or {}) - set(st))
    if unknown:
        raise ValueError(f"unknown stabilization key(s) {unknown}: the keys are 'supg', 'pspg' and 'lsic'")
    st.update(stabilization or {})
    if viscosity_scales is not None and continuation_re is None:
        vs = [float(x) for x in viscosity_scales]
        if vs and abs(vs[-1] - 1.0) > 1e-12:
            raise ValueError(f"viscosity_scales must end at 1 (the last stage solves the given mu), got {vs[-1]}")
    if re is not None and continuation_re is None:
        raise ValueError("re= is only meaningful with continuation_re")
    if continuation_re is not None:
        if viscosity_scales is not None:
            raise ValueError("give continuation_re or viscosity_scales, not both")
        res = [float(r) for r in continuation_re]
        if not res or any(not (r > 0) for r in res):
            raise ValueError("continuation_re must be a non-empty list of positive Reynolds numbers")
        if re is not None and abs(res[-1] - float(re)) > 1e-12 * abs(float(re)):
            raise ValueError(f"the last continuation Re ({res[-1]}) must be re ({re})")
        viscosity_scales = [res[-1] / r for r in res]
    core = _f.gls_problem(V, T, mu_q, gmu_q, mu_nodal, f_q, adv_q, hist[0] if len(hist) > 0 else None,
                          hist[1] if len(hist) > 1 else None, dn, dv, float(rho), mode, form, float(dt), int(bdf),
                          bool(st["supg"]), bool(st["pspg"]), bool(st["lsic"]), lu or default_lu(), int(lu_threads),
                          float(newton_rtol), int(newton_maxit), bool(line_search),
                          [float(s) for s in (viscosity_scales or [])], bool(solve))
    return GlsSolution(core, V, T, mode, form, float(dt), int(bdf))


def solve_gls(*args, **kw):
    """gls_problem(..., solve=True)."""
    kw["solve"] = True
    return gls_problem(*args, **kw)


__all__ = ["gls_problem", "solve_gls", "rule_points", "quadrature_coordinates", "GlsSolution"]
