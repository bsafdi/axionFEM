"""Self-tests of the planar Stokes check module benchmarks/fluids/cases/f1a_stokes_check.py.

No solver: the discrete "solutions" here are P2/P1 nodal INTERPOLANTS of the exact fields (evaluated by an
interpolation evaluator written in this file), whose error orders are known exactly (3 / 2 / 2 on the fixed nested
meshes), plus exact-field evaluators for the identities.  Seconds on a login node: `pytest -m fast`.
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
sys.dont_write_bytecode = True
import f1a_stokes_check as C  # noqa: E402

fast = pytest.mark.fast


def entry(name):
    p = ROOT / "tools" / "fluids_mms" / "out" / f"{name}.json"
    return {"path": f"tools/fluids_mms/out/{name}.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


# ---------------------------------------------------------------------------------------------- evaluators
def _p2(q):
    xi, eta = q[:, 0], q[:, 1]
    L = [1 - xi - eta, xi, eta]
    N = np.stack([L[0] * (2 * L[0] - 1), L[1] * (2 * L[1] - 1), L[2] * (2 * L[2] - 1),
                  4 * L[0] * L[1], 4 * L[1] * L[2], 4 * L[2] * L[0]], 1)
    dl = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    dN = np.zeros((len(xi), 6, 2))
    for i in range(3):
        dN[:, i] = (4 * L[i] - 1)[:, None] * dl[i]
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        dN[:, 3 + k] = 4 * (L[i][:, None] * dl[j] + L[j][:, None] * dl[i])
    return N, dN


class InterpEval:
    """P2 velocity / P1 pressure nodal interpolant of an exact field on an affine mesh (V, T)."""

    def __init__(self, V, T, u, p, p_shift=0.0):
        self.V, self.T = np.asarray(V, float), np.asarray(T)
        P = self.V[self.T]                                                       # (ne, 3, 2)
        nodes = np.concatenate([P, 0.5 * (P[:, [0, 1, 2]] + P[:, [1, 2, 0]])], axis=1)   # (ne, 6, 2)
        self.uv = u(nodes[..., 0], nodes[..., 1])                                 # (ne, 6, 2)
        self.pv = p(P[..., 0], P[..., 1]) + p_shift                               # (ne, 3)
        self.n_elements = len(self.T)

    def _J(self):
        P = self.V[self.T]
        J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=-1)            # J[e, d, a] = dx_d / dxi_a
        return P, J

    def geometry(self, q):
        P, J = self._J()
        x = P[:, None, 0, :] + np.einsum("eda,qa->eqd", J, q)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        return x, np.repeat(det[:, None], len(q), axis=1)

    def velocity(self, q):
        N, dN = _p2(q)
        _, J = self._J()
        Jinv = np.linalg.inv(J)                                                   # Jinv[e, a, d]
        G = np.einsum("qka,ead->eqkd", dN, Jinv)                                   # d phi_k / d x_d
        u = np.einsum("qk,ekc->eqc", N, self.uv)
        gu = np.einsum("eqkd,ekc->eqcd", G, self.uv)
        return u, gu

    def pressure(self, q):
        Np = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], 1)
        return np.einsum("qk,ek->eq", Np, self.pv)


class ExactEval(InterpEval):
    """The exact field itself at the quadrature points (no interpolation)."""

    def __init__(self, V, T, u, gu, p):
        self.V, self.T = np.asarray(V, float), np.asarray(T)
        self.u_, self.gu_, self.p_ = u, gu, p
        self.n_elements = len(self.T)

    def velocity(self, q):
        x, _ = self.geometry(q)
        return self.u_(x[..., 0], x[..., 1]), self.gu_(x[..., 0], x[..., 1])

    def pressure(self, q):
        x, _ = self.geometry(q)
        return self.p_(x[..., 0], x[..., 1])


def _orders(hs, es):
    hs, es = np.asarray(hs), np.asarray(es)
    return np.log(es[:-1] / es[1:]) / np.log(hs[:-1] / hs[1:])


# ---------------------------------------------------------------------------------------------- tests
@fast
def test_quadrature_exact_to_degree_12():
    q, w = C.tri_rule(10)
    assert abs(w.sum() - 0.5) < 1e-15
    for a in range(13):
        for b in range(13 - a):
            exact = math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
            assert abs((w * q[:, 0] ** a * q[:, 1] ** b).sum() - exact) < 1e-15, (a, b)


@fast
def test_field_hash_is_verified(tmp_path):
    e = entry("sq_dirichlet")
    C.load_field(e)
    with pytest.raises(RuntimeError):
        C.load_field({"path": e["path"], "sha256": "0" * 64})


@fast
@pytest.mark.parametrize("family", ["structured", "distorted"])
def test_interpolation_orders_3_2_2(family):
    """Interpolation errors on the fixed nested families converge at exactly 3 / 2 / 2 (uL2 / uH1 / pL2): the
    norms, the quadrature and the element map are consistent."""
    fld = C.load_field(entry("sq_dirichlet"))
    hs, rows = [], []
    for k in range(3):
        V, T = C.structured_mesh(8 * 2 ** k) if family == "structured" else C.distorted_mesh(k + 1)
        ev = InterpEval(V, T, fld.u, fld.p)
        rows.append(C.norms(ev, fld, "mean_free")); hs.append(1 / (8 * 2 ** k))
    for key, p in (("uL2", 3.0), ("uH1", 2.0), ("pL2", 2.0)):
        o = _orders(hs, [r[key] for r in rows])
        assert abs(o[-1] - p) < 0.12, (family, key, o)


@fast
def test_mean_free_pressure_and_pin_offset():
    fld = C.load_field(entry("sq_dirichlet"))
    V, T = C.structured_mesh(16)
    a = C.norms(InterpEval(V, T, fld.u, fld.p), fld, "mean_free")
    b = C.norms(InterpEval(V, T, fld.u, fld.p, p_shift=0.7), fld, "mean_free")
    assert abs(a["pL2"] - b["pL2"]) < 1e-13
    assert abs((b["pmean_h"] - a["pmean_h"]) - 0.7) < 1e-13
    d = C.norms(InterpEval(V, T, fld.u, fld.p, p_shift=0.7), fld, "determined")
    assert d["pL2"] > 0.69 * math.sqrt(a["area"])


@fast
def test_exact_field_has_zero_error_and_divergence():
    fld = C.load_field(entry("sq_dirichlet"))
    V, T = C.distorted_mesh(1)
    r = C.norms(ExactEval(V, T, fld.u, fld.grad_u, fld.p), fld, "determined")
    assert r["uL2"] == 0 and r["uH1"] == 0 and r["pL2"] == 0
    assert r["divL2"] < 1e-12
    assert abs(r["area"] - 1) < 1e-14


@fast
def test_flux_defect_definition_and_order():
    """Simpson definition: zero for data whose normal component is cubic along each edge; O(h^4) for sq_dirichlet,
    matching independently computed numbers (2.5e-7, 1.6e-8, 9.8e-10 relative at N = 8, 16, 32)."""
    def edges(N):
        V, T = C.structured_mesh(N)
        out = []
        for a, b in C.boundary_edges(T):
            Pa, Pb = V[a], V[b]
            d = Pb - Pa
            n = np.array([d[1], -d[0]]) / np.hypot(*d)
            if np.dot(n, 0.5 * (Pa + Pb) - 0.5) < 0:
                n = -n
            out.append((Pa, 0.5 * (Pa + Pb), Pb, n))
        return out
    cubic = lambda X, Y: np.stack([X ** 3 + Y, Y ** 2 * X], -1)  # noqa: E731
    d, r = C.flux_defect(edges(8), cubic)
    # Simpson is exact for the cubic: the value is the exact flux int div = int (3x^2 + 2xy) = 1 + 1/2
    assert abs(d - 1.5) < 1e-13
    fld = C.load_field(entry("sq_dirichlet"))
    rel = []
    for N in (8, 16, 32):
        d, r = C.flux_defect(edges(N), fld.u)
        rel.append(abs(d) / r)
    assert np.allclose(rel, [2.5e-7, 1.6e-8, 9.8e-10], rtol=0.1), rel
    fp = C.load_field(entry("sq_point"))          # flux-free data: exactly zero
    d, r = C.flux_defect(edges(16), fp.u)
    assert abs(d) < 1e-15 * max(r, 1)


@fast
def test_meshes_corner_condition_nesting_hash():
    for N in (8, 16, 32, 64):
        V, T = C.structured_mesh(N)
        assert len(T) == 2 * N * N and C.all_boundary_triangles(T) == 0
    Vb, Tb = C._structured(16, 16, flip=False)
    assert C.all_boundary_triangles(Tb) == 2
    V1, T1 = C.structured_mesh(16)
    V2, T2 = C.structured_mesh(32)
    v = C.verify_mesh(V2, T2, V2, T2, V_coarser=V1)
    assert v["nested"] and v["hash_match"] and v["counts_match"] and abs(v["h_max"] - math.sqrt(2) / 32) < 1e-14
    perm = np.random.default_rng(1).permutation(len(V2))
    inv = np.argsort(perm)
    assert C.mesh_hash(V2[perm], inv[T2][::-1]) == C.mesh_hash(V2, T2)
    # the non-nested alternative reading (only the two corner cells flipped at every level) differs
    Va, Ta = C._structured(32, 32)
    assert C.mesh_hash(Va, Ta) != C.mesh_hash(V2, T2)
    # distorted family: deterministic seeded displacement, nested, Jacobians vary
    Vd, Td = C.distorted_mesh(0)
    Vd2, _ = C.distorted_mesh(0)
    assert np.array_equal(Vd, Vd2) and not np.allclose(Vd, C._structured(4, 4)[0])
    Vk, Tk = C.distorted_mesh(2)
    assert C.verify_mesh(Vk, Tk, Vk, Tk, V_coarser=C.distorted_mesh(1)[0])["nested"]
    _, det = InterpEval(Vk, Tk, lambda X, Y: np.stack([X, Y], -1), lambda X, Y: X).geometry(np.array([[1 / 3, 1 / 3]]))
    assert det.std() / det.mean() > 0.05


@fast
def test_energy_identity_detects_wrong_gamma():
    """u = (b, 0), b = sin(pi x) sin(pi y) (zero on the boundary, div u != 0), constant nu, no pressure:
        lap: f = -nu lap u - gamma grad div u,   sym: f = -nu lap u - (nu + gamma) grad div u
    (derived by hand: lap b = -2 pi^2 b, grad div u = (-pi^2 b, pi^2 cos(pi x) cos(pi y))).  The identity
    a_form(u,u) + gamma ||div u||^2 = (f, u) then holds; on the exact field it is quadrature-exact, and a wrong gamma
    (scaled by nu) is detected."""
    nu, gamma, pi = 0.37, 10.0, math.pi
    b = lambda X, Y: np.sin(pi * X) * np.sin(pi * Y)  # noqa: E731
    U = lambda X, Y: np.stack([b(X, Y), 0 * X], -1)  # noqa: E731
    GU = lambda X, Y: np.stack([np.stack([pi * np.cos(pi * X) * np.sin(pi * Y), pi * np.sin(pi * X) * np.cos(pi * Y)], -1),  # noqa: E731
                                np.stack([0 * X, 0 * X], -1)], -2)
    def F(form):
        c = gamma + (nu if form == "sym" else 0.0)
        return lambda X, Y: np.stack([2 * nu * pi ** 2 * b(X, Y) + c * pi ** 2 * b(X, Y),
                                      -c * pi ** 2 * np.cos(pi * X) * np.cos(pi * Y)], -1)
    V, T = C.structured_mesh(16)
    ev = ExactEval(V, T, U, GU, lambda X, Y: 0 * X)
    for form in ("lap", "sym"):
        ok = C.energy_identity_defect(ev, nu, gamma, form, F(form))
        scaled = C.energy_identity_defect(ev, nu, gamma * nu, form, F(form))
        assert ok < 1e-9, (form, ok)
        assert scaled > 1e-2, (form, scaled)


@fast
def test_inf_sup_counts_zero_modes():
    A = np.eye(4)
    B = np.array([[1.0, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 0]])
    r = C.inf_sup(A, B, np.eye(3))
    assert r["zero_modes_beyond_constant"] == 0 and abs(r["beta"] - 0.5) < 1e-14
    B2 = np.array([[1.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert C.inf_sup(A, B2, np.eye(3))["zero_modes_beyond_constant"] == 1


@fast
def test_slip_nodal_check_and_polar_mesh():
    n = (0.5, -math.sqrt(3) / 2)
    assert C.slip_nodal_un([np.array([math.sqrt(3) / 2, 0.5])], [[n]], 1.0) < 1e-15
    assert C.slip_nodal_un([np.array([0.0, 1.0])], [[n]], 1.0) > 0.8
    V, T = C.polar_mesh(4)
    assert len(V) == 5 * 32 and len(T) == 2 * 4 * 32 and C.all_boundary_triangles(T) == 0


@fast
def test_polar_p2_nodes_and_dirichlet_defect():
    """The curved geometry is built and hashed; nodal Dirichlet data are checked."""
    X6 = C.polar_p2_nodes(4)
    V, T = C.polar_mesh(4)
    assert X6.shape == (len(T), 6, 2)
    r = np.hypot(X6[..., 0], X6[..., 1])
    on1 = np.isclose(r[:, :3], 1.0)
    # an edge with both end points on r = 1 has its midpoint on r = 1; interior edges are straight
    for e in range(len(T)):
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            if on1[e, i] and on1[e, j]:
                assert abs(r[e, 3 + k] - 1.0) < 1e-14
            elif not (np.isclose(r[e, i], 2.0) and np.isclose(r[e, j], 2.0)):
                assert np.allclose(X6[e, 3 + k], 0.5 * (X6[e, i] + X6[e, j]))
    straight = X6.copy()
    straight[:, 3:] = 0.5 * (X6[:, [0, 1, 2]] + X6[:, [1, 2, 0]])
    v = C.verify_mesh(V, T, V, T, X6=straight, X6_ref=X6)
    assert v["hash_match"] and not v["p2_hash_match"]
    assert C.verify_mesh(V, T, V, T, X6=X6[::-1], X6_ref=X6)["p2_hash_match"]
    g = lambda X, Y: np.stack([np.sin(X), X * Y], -1)  # noqa: E731
    P = np.random.default_rng(0).uniform(0, 1, (20, 2))
    assert C.dirichlet_nodal_defect(P, g(P[:, 0], P[:, 1]), g) == 0.0
    assert C.dirichlet_nodal_defect(P, g(P[:, 0], P[:, 1]) + 1e-7, g) > 1e-8
    n = (0.0, 1.0)
    assert C.slip_nodal_un([np.array([0.0, 1.0])], [[n]], 1.0, on_dirichlet=[True]) == 0.0


@fast
def test_shared_node_rule():
    """Cavity: lid data 16x^2(1-x)^2 and wall data 0 agree at the lid corners; an inconsistent lid (e.g. 1) is caught."""
    lid = lambda X, Y: np.stack([16 * X ** 2 * (1 - X) ** 2, 0 * X], -1)  # noqa: E731
    wall = lambda X, Y: np.stack([0 * X, 0 * X], -1)  # noqa: E731
    bad = lambda X, Y: np.stack([1 + 0 * X, 0 * X], -1)  # noqa: E731
    corners = [(0.0, 1.0), (1.0, 1.0)]
    assert C.dirichlet_shared_node_mismatch(corners, [[lid, wall], [lid, wall]]) == 0.0
    assert C.dirichlet_shared_node_mismatch(corners, [[bad, wall], [bad, wall]]) == 1.0
