"""Porous-electrode check module (chem_analytic_c3_porous_1d), numpy only.  The 3D cases use the current-distribution
check module c3_checks (its functions are dimension-generic).

The closed-form fields below are the linear porous-electrode solution in double precision (the case needs the exact
fields for its L2/H1 order series, as an MMS case reads its manufactured field).  The reference values (V, I) of the
case definition come from a 40-digit mpmath evaluation of the same closed form.

The 3D cases (chem_mms_c3_3d, chem_analytic_c3_concentric_3d) name this module as their check module: it re-exports the fixed C3a module c3_checks (dimension-generic) and adds the strong flux current and per-chunk
error sums.

The runner calls these functions with RAW numbers from the solver: per element the quadrature points and weights
(with the 1D Jacobian), the two discrete fields and their x-derivatives there, the full (unconstrained) residual of the
two field equations at the collector row and the separator row.
"""
from __future__ import annotations

import math

import numpy as np

from c3_checks import *                 # noqa: F401,F403  (the fixed C3a module next to this one)
from c3_checks import field_errors      # noqa: F401  the C3a (3D-capable) field_errors, explicitly

FORMAT = "axionfem.chem.c3b.checks/1"
F_C, R_C = 96485.33212, 8.314462618


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def kv(params: dict, T: float = 298.15) -> float:
    """Volumetric exchange coefficient a_v i0 (alpha_a + alpha_c) F/(R T) [A/(V m^3)] from the fixed parameters."""
    return _num(params["a_v"]) * _num(params["i0"]) * (_num(params["alpha_a"]) + _num(params["alpha_c"])) * \
        F_C / (R_C * T)


def exact(params: dict, I: float):
    """Closed-form phi1, phi1', phi2, phi2' (callables of x) and V = phi1(0) - phi2(l) for the cell current I."""
    l, sg, kp, k_v = _num(params["l"]), _num(params["sigma"]), _num(params["kappa"]), kv(params)
    k = math.sqrt(k_v * (1 / sg + 1 / kp))
    nu = k * l
    B = -I / (sg * k)
    A = I * (1 / kp + math.cosh(nu) / sg) / (k * math.sinh(nu))
    eta = lambda x: A * np.cosh(k * x) + B * np.sinh(k * x)                                  # noqa: E731
    deta = lambda x: k * (A * np.sinh(k * x) + B * np.cosh(k * x))                           # noqa: E731
    i2 = lambda x: k_v * (A * np.sinh(k * x) + B * (np.cosh(k * x) - 1)) / k                  # noqa: E731
    int_i2 = lambda x: k_v * (A * (np.cosh(k * x) - 1) + B * (np.sinh(k * x) - k * x)) / k ** 2  # noqa: E731
    phi1 = lambda x: -(I * x - int_i2(x)) / sg                                               # noqa: E731
    dphi1 = lambda x: -(I - i2(x)) / sg                                                      # noqa: E731
    phi2 = lambda x: phi1(x) - eta(x)                                                        # noqa: E731
    dphi2 = lambda x: dphi1(x) - deta(x)                                                     # noqa: E731
    V = float(phi1(0.0) - phi2(l))
    return phi1, dphi1, phi2, dphi2, V


def resistance(params: dict) -> float:
    """V/I of the closed form (linear problem)."""
    return exact(params, 1.0)[4]


def porous_field_errors(xq, wq, p1, dp1, p2, dp2, params: dict, I: float) -> dict:
    """L2 and H1-seminorm errors of both fields together: sqrt(|e1|^2 + |e2|^2) from raw quadrature data (the rule
    of degree >= 2k + 8 the case declares)."""
    e1, d1, e2, d2, _ = exact(params, I)
    xq, wq = np.asarray(xq, float), np.asarray(wq, float)
    L2 = np.sum(((np.asarray(p1) - e1(xq)) ** 2 + (np.asarray(p2) - e2(xq)) ** 2) * wq)
    H1 = np.sum(((np.asarray(dp1) - d1(xq)) ** 2 + (np.asarray(dp2) - d2(xq)) ** 2) * wq)
    return {"L2": float(math.sqrt(L2)), "H1": float(math.sqrt(H1))}


def exchange_total(xq, wq, p1, p2, params: dict) -> float:
    """int kv (phi1_h - phi2_h) dx from raw quadrature data and the fixed parameters (the case's own exchange term
    must reproduce it; a different kv is caught by the comparisons)."""
    return float(kv(params) * np.sum((np.asarray(p1, float) - np.asarray(p2, float)) * np.asarray(wq, float)))


def identities(exchange: float, collector_reaction: float, I: float, terms: float) -> dict:
    """Identities of the two-field system (sums of residual rows): int j_h = I (electrolyte equation tested with
    1) and the collector reaction current = int j_h (solid equation tested with 1).  `terms` = sum over all rows of
    |A||z| (+ |I|), the magnitudes that cancel."""
    return {"exchange_abs": abs(exchange - I), "collector_abs": abs(collector_reaction - I), "terms": float(terms)}


# ------------------------------------------------------------------------------------------------ 3D
def flux_current(x, w, grad_h, normal, kappa) -> float:
    """Strong flux current int kappa grad phi_h . n_out dA from raw facet quadrature: points x (n, 3), weights w (n,)
    with the surface Jacobian, the discrete gradient (n, 3) at the points, the OUTWARD unit normal (n, 3) of the
    domain there (on curved P2 facets: the normal of the isoparametric map at the point) and kappa: a callable of x or
    a number.  The normal is checked to be unit length; its orientation is what the strong-flux comparisons test."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    g, n = np.asarray(grad_h, float), np.asarray(normal, float)
    nn = np.linalg.norm(n, axis=1)
    if np.any(np.abs(nn - 1.0) > 1e-10):
        raise ValueError("flux_current: the normals are not unit vectors")
    k = kappa(x) if callable(kappa) else float(kappa) * np.ones(len(w))
    return float(np.sum(k * np.einsum("qi,qi->q", g, n) * w))


def field_error_sums(x, w, phi_h, grad_h, phi, grad_phi) -> dict:
    """Per-chunk (e.g. per rank) sums of the squared errors from raw domain quadrature at degree 2k + 8: returns
    {"L2_sq", "H1_sq"}; combine_error_sums adds chunks and takes the root (fixed arithmetic)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    e = np.asarray(phi_h, float) - phi(x)
    ge = np.asarray(grad_h, float) - grad_phi(x)
    return {"L2_sq": float(np.sum(e * e * w)), "H1_sq": float(np.sum(np.sum(ge * ge, axis=1) * w))}


def combine_error_sums(chunks) -> dict:
    return {"L2": math.sqrt(sum(c["L2_sq"] for c in chunks)), "H1": math.sqrt(sum(c["H1_sq"] for c in chunks))}


# ----------------------------------------------------------------------------------------------------- porous strips
def strip_spread(x_nodes, phi1_nodes, phi2_nodes, x_lattice, tol_rel=1e-9) -> float:
    """Max over the 1D lattice points x_i of the spread (max - min) of phi1 and of phi2 over the strip nodes with
    x = x_i (a 1D path: one node per x_i, spread 0).  A node that is not on the lattice is an error: simplex strips
    with interior nodes off the lattice are not allowed by the run block."""
    xn = np.asarray(x_nodes, float)
    xl = np.asarray(x_lattice, float)
    L = float(xl[-1] - xl[0])
    idx = np.rint((xn - xl[0]) / L * (len(xl) - 1)).astype(int)
    if np.any(np.abs(xl[np.clip(idx, 0, len(xl) - 1)] - xn) > tol_rel * L):
        raise ValueError("strip_spread: a node is off the fixed x-lattice (not a tensor-product strip)")
    out = 0.0
    for f in (np.asarray(phi1_nodes, float), np.asarray(phi2_nodes, float)):
        for i in np.unique(idx):
            v = f[idx == i]
            out = max(out, float(v.max() - v.min()))
    return out
