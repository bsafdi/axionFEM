"""Self-tests of the steady Navier-Stokes check module benchmarks/fluids/cases/f2a_ns_check.py (the check module
judges every solver result, so it is tested on its own).

No solver.  The discrete "solutions" are P2/P1 nodal INTERPOLANTS of the exact fields (planar and axisymmetric),
whose error orders on the fixed nested meshes are known (3 / 2 / 2); exact fields test the identities; synthetic
residual maps test the Newton-order, line-search and Taylor-test diagnostics.  Seconds: `pytest -m fast`.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True
import f2a_ns_check as C  # noqa: E402
from test_fluids_stokes_planar_check import _p2  # noqa: E402

fast = pytest.mark.fast


def entry(name):
    p = ROOT / "tools" / "fluids_mms" / "out" / f"{name}.json"
    return {"path": f"tools/fluids_mms/out/{name}.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


class InterpEval:
    """P2 velocity (any number of components) / P1 pressure nodal interpolant on an affine mesh; for axisymmetric
    fields the coordinates are (r, z) and the derivatives are component derivatives d u_i / d (r, z)_j."""

    def __init__(self, V, T, u, p):
        self.V, self.T = np.asarray(V, float), np.asarray(T)
        P = self.V[self.T]
        nodes = np.concatenate([P, 0.5 * (P[:, [0, 1, 2]] + P[:, [1, 2, 0]])], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = u(nodes[..., 0], nodes[..., 1])
        self.uv = np.nan_to_num(uv)        # axis nodes: u_r = u_t = 0 (the axis rule)
        self.pv = p(P[..., 0], P[..., 1])
        self.n_elements = len(self.T)

    def _J(self):
        P = self.V[self.T]
        return P, np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=-1)

    def geometry(self, q):
        P, J = self._J()
        x = P[:, None, 0, :] + np.einsum("eda,qa->eqd", J, q)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        return x, np.repeat(np.abs(det)[:, None], len(q), axis=1)

    def velocity(self, q):
        N, dN = _p2(q)
        _, J = self._J()
        G = np.einsum("qka,ead->eqkd", dN, np.linalg.inv(J))
        return np.einsum("qk,ekc->eqc", N, self.uv), np.einsum("eqkd,ekc->eqcd", G, self.uv)

    def pressure(self, q):
        Np = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], 1)
        return np.einsum("qk,ek->eq", Np, self.pv)


class ExactEval(InterpEval):
    def __init__(self, V, T, fld):
        super().__init__(V, T, fld.u, fld.p)
        self.fld = fld

    def velocity(self, q):
        x, _ = self.geometry(q)
        return self.fld.u(x[..., 0], x[..., 1]), self.fld.grad_u(x[..., 0], x[..., 1])

    def pressure(self, q):
        x, _ = self.geometry(q)
        return self.fld.p(x[..., 0], x[..., 1])


def _orders(e):
    e = np.asarray(e)
    return np.log2(e[:-1] / e[1:])


@fast
def test_load_field_refuses_wrong_hash():
    e = entry("ns_sq_dirichlet")
    C.load_field(e)
    with pytest.raises(RuntimeError):
        C.load_field({**e, "sha256": "0" * 64})


@fast
def test_exact_solutions_have_zero_forcing_and_exact_norms_vanish():
    for name, geo in (("ns_kovasznay_re40", "planar"), ("ns_axi_burgers", "axisymmetric")):
        fld = C.load_field(entry(name))
        assert fld.geometry == geo and not fld.data["forms"]["lap"]["f"][0]["numpy"].strip("0. ")
    fld = C.load_field(entry("ns_sq_dirichlet"))
    V, T = C.structured_mesh(8)
    nr = C.norms(ExactEval(V, T, fld), fld)
    assert nr["uL2"] == 0 and nr["uH1"] == 0 and nr["pL2"] < 1e-15
    assert nr["divL2"] < 1e-12                       # the exact field is divergence-free


@fast
def test_interpolant_orders_planar():
    fld = C.load_field(entry("ns_sq_dirichlet"))
    rows = [C.norms(InterpEval(*C.structured_mesh(N), fld.u, fld.p), fld) for N in (8, 16, 32)]
    assert np.allclose(_orders([r["uL2"] for r in rows]), 3, atol=0.1)
    assert np.allclose(_orders([r["uH1"] for r in rows]), 2, atol=0.1)
    assert np.allclose(_orders([r["pL2"] for r in rows]), 2, atol=0.1)


@fast
def test_interpolant_orders_axisymmetric_weighted_and_unweighted():
    fld = C.load_field(entry("ns_axi_dirichlet"))
    rows = [C.norms(InterpEval(*C.structured_mesh(N), fld.u, fld.p), fld) for N in (8, 16, 32)]
    for k, p in (("uL2", 3), ("uH1", 2), ("pL2", 2), ("uL2_unw", 3), ("uH1_unw", 2), ("pL2_unw", 2)):
        assert np.allclose(_orders([r[k] for r in rows]), p, atol=0.15), (k, _orders([r[k] for r in rows]))
    # the weighted H1 seminorm contains the hoop terms: it is NOT the unweighted one times a constant
    assert rows[-1]["uH1"] != pytest.approx(rows[-1]["uH1_unw"], rel=0.05)


@fast
def test_kovasznay_family_nested_and_taylor_hood_condition():
    prev = None
    for k in range(3):
        V, T = C.rect_mesh(6, 8, k, -0.5, 1, -0.5, 1.5)
        assert len(T) == 6 * 8 * 2 * 4 ** k
        assert C.all_boundary_triangles(T) == 0
        r = C.verify_mesh(V, T, V, T, V_coarser=prev)
        assert r["hash_match"] and (prev is None or r["nested"])
        assert abs(r["h_max"] - math.sqrt(2) * 0.25 / 2 ** k) < 1e-12
        prev = V


@fast
def test_cylindrical_gradient_against_cartesian_differences():
    fld = C.load_field(entry("ns_axi_dirichlet"))
    rr, zz, phi, h = np.array([0.37]), np.array([0.61]), 0.4, 1e-5
    G = C.cyl_gradient(fld.u(rr, zz), fld.grad_u(rr, zz), rr)[0]

    def U(X):
        r_, z_ = np.hypot(X[0], X[1]), X[2]
        a, b, w = fld.u(np.array([r_]), np.array([z_]))[0]
        c, s = X[0] / r_, X[1] / r_
        return np.array([a * c - b * s, a * s + b * c, w])
    X0 = np.array([0.37 * math.cos(phi), 0.37 * math.sin(phi), 0.61])
    Gc = np.stack([(U(X0 + h * e) - U(X0 - h * e)) / (2 * h) for e in np.eye(3)], -1)
    Q = np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]])
    assert np.allclose(Q @ Gc @ Q.T, G, atol=1e-8)


@fast
def test_convective_forms_differ_by_the_divergence_term_only():
    """c_skew - c_std = (rho/2) (div u_h, |u_h|^2): zero for a divergence-free field, not otherwise."""
    fld = C.load_field(entry("ns_sq_dirichlet"))
    V, T = C.structured_mesh(8)
    d = C._forms(ExactEval(V, T, fld), "lap", "standard", 1.25, 0.01, False)
    assert abs(d["c_skew"] - d["c_standard"]) < 1e-12 * abs(d["c_standard"]) + 1e-15
    ev = InterpEval(V, T, lambda A, B: np.stack([A * (1 - A) * B, 0 * A], -1), lambda A, B: 0 * A)
    d = C._forms(ev, "lap", "standard", 1.0, 0.01, False)
    # div u = (1 - 2x) y; (1/2) int (1 - 2x) y |u|^2 by an independent tensor Gauss rule
    g, w = np.polynomial.legendre.leggauss(20)
    s, W = 0.5 * (g + 1), 0.25 * np.outer(w, w)
    X, Y = np.meshgrid(s, s, indexing="ij")
    ref = 0.5 * float(np.sum(W * (1 - 2 * X) * Y * (X * (1 - X) * Y) ** 2))
    assert d["c_skew"] - d["c_standard"] == pytest.approx(ref, rel=1e-10)


@fast
def test_energy_identity_exact_for_a_consistent_synthetic_problem():
    """For u = 0 on the boundary and f := what the discrete identity needs, the declared form's defect is round-off
    and the other forms show their (computable) differences."""
    V, T = C.structured_mesh(8)
    # (the exp factor breaks the reflection symmetry that would make int div(u) |u|^2 vanish)
    ev = InterpEval(V, T, lambda A, B: np.exp(A + B / 2)[..., None] * np.stack(
        [(A * (1 - A)) ** 2 * B * (1 - B) * 3, A * (1 - A) * (B * (1 - B)) ** 2], -1), lambda A, B: 0 * A)
    d = C._forms(ev, "sym", "standard", 1.0, 0.02, False)
    # choose a constant body force f = (F, 0) scaled so that (f, u) = a + c_std exactly
    q, xq, W = C._weights(ev, 14)
    u, _ = ev.velocity(q)
    Fx = (d["a"] + d["c_standard"]) / float((W * u[..., 0]).sum())
    e = C.energy_identity(ev, "sym", "standard", 1.0, 0.02, lambda A, B: np.stack([Fx + 0 * A, 0 * A], -1))
    assert e["defect"] < 1e-13
    assert e["defect_other_convective_form"] > 1e-6          # div u_h != 0 for this field
    assert e["defect_other_viscous_form"] > 1e-6


@fast
def test_newton_order_and_line_search_diagnostics():
    quad = [1e-1, 1e-2, 1e-4, 1e-8]
    lin = [1e-1 * 0.3 ** k for k in range(10)]
    assert C.newton_order(quad)["p_max"] == pytest.approx(2.0)
    assert C.newton_order(lin)["p_max"] == pytest.approx(1.0)
    assert C.newton_order(quad)["p_median"] == pytest.approx(2.0)
    osc = [1e-1, 3e-2, 2e-2, 2e-3, 1e-3, 1e-4, 5e-5]           # Picard-like: one large p_k, median ~1
    assert C.newton_order(osc)["p_max"] > 1.6 and C.newton_order(osc)["p_median"] < 1.3
    assert C.newton_order([1e-2, 1e-6, 1e-13])["p"] == []          # below the floor: no estimate
    assert C.line_search_violations([1.0, 0.5, 0.6], [1.0, 1.0]) == 1
    assert C.line_search_violations([1.0, 0.5, 0.2], [1.0, 0.5]) == 0


@fast
def test_taylor_test_exact_and_wrong_jacobian():
    rng = np.random.default_rng(0)
    n = 30
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n, n))
    R = lambda x: A @ x + np.einsum("ijk,j,k->i", B, x, x)      # noqa: E731  quadratic like NS
    J = lambda x: A + np.einsum("ijk,k->ij", B, x) + np.einsum("ijk,j->ik", B, x)  # noqa: E731
    x, d = rng.standard_normal(n), rng.standard_normal(n)
    eps = [10.0 ** (-k / 2) for k in range(9)]
    t = C.taylor_test(R(x), [R(x + e * d) for e in eps], J(x) @ d, eps)
    assert t["slope_r2"] == pytest.approx(2.0, abs=0.01) and t["r2_over_eps2_spread"] < 1e-6
    assert t["slope_r1"] == pytest.approx(1.0, abs=0.05)
    Jw = A + np.einsum("ijk,j->ik", B, x)                          # one linearization term missing
    t = C.taylor_test(R(x), [R(x + e * d) for e in eps], Jw @ d, eps)
    assert t["slope_r2"] < 1.3
