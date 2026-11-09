"""Self-tests of the axisymmetric Stokes check module benchmarks/fluids/cases/f1b_stokes_check.py.

No solver: the discrete "solutions" are P2/P1 nodal INTERPOLANTS of exact fields (an interpolation evaluator written
in this file, isoparametric when 6 geometry nodes are given), whose error orders are known (3 / 2 / 2), and exact
polynomial fields for which every check value is known in closed form.  Seconds on a login node: `pytest -m fast`.
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
import f1b_stokes_check as C  # noqa: E402

fast = pytest.mark.fast


def entry(name):
    p = ROOT / "tools" / "fluids_mms" / "out" / "f1b" / f"{name}.json"
    return {"path": f"tools/fluids_mms/out/f1b/{name}.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


class InterpEval:
    """P2 (3 components) / P1 nodal interpolant of exact callables on a mesh with 6-node geometry X6."""

    def __init__(self, X6, u, p, p_shift=0.0):
        self.X6 = np.asarray(X6, float)
        self.uv = u(self.X6[..., 0], self.X6[..., 1])                         # (ne, 6, 3)
        self.pv = p(self.X6[:, :3, 0], self.X6[:, :3, 1]) + p_shift           # (ne, 3)
        self.n_elements = len(self.X6)

    def _map(self, q):
        N, dN = C._p2(q)
        x = np.einsum("qk,ekd->eqd", N, self.X6)
        J = np.einsum("qkj,ekd->eqdj", dN, self.X6)
        return N, dN, x, J

    def geometry(self, q):
        _, _, x, J = self._map(q)
        return x, np.abs(np.linalg.det(J))

    def velocity(self, q):
        N, dN, _, J = self._map(q)
        dNx = np.einsum("qkj,eqjd->eqkd", dN, np.linalg.inv(J))
        return np.einsum("qk,ekc->eqc", N, self.uv), np.einsum("eqkd,ekc->eqcd", dNx, self.uv)

    def pressure(self, q):
        L = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], -1)
        return np.einsum("qk,ek->eq", L, self.pv)


def affine_X6(V, T):
    V, T = np.asarray(V, float), np.asarray(T)
    P = V[T]
    return np.concatenate([P, 0.5 * (P[:, [0, 1, 2]] + P[:, [1, 2, 0]])], axis=1)


def orders(e):
    return [math.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]


@fast
def test_field_hash_is_enforced():
    e = entry("axi_dirichlet")
    C.load_field(e, ROOT)
    with pytest.raises(RuntimeError):
        C.load_field(dict(e, sha256="0" * 64), ROOT)


@fast
def test_interpolation_orders_weighted_and_plain():
    F = C.load_field(entry("axi_dirichlet"), ROOT)
    rows = []
    for N in (8, 16, 32):
        V, T = C.square_mesh(N)
        rows.append(C.norms(InterpEval(affine_X6(V, T), F.u, F.p), F))
    for k, p in (("uL2r", 3), ("uH1r", 2), ("pL2r", 2), ("uL2", 3), ("uH1", 2), ("pL2", 2), ("utL2r", 3), ("utH1r", 2)):
        o = orders([r[k] for r in rows])
        assert abs(o[-1] - p) < 0.12, (k, o)


@fast
def test_mean_free_pressure_ignores_constants_and_determined_does_not():
    F = C.load_field(entry("axi_dirichlet"), ROOT)
    V, T = C.square_mesh(8)
    X6 = affine_X6(V, T)
    a = C.norms(InterpEval(X6, F.u, F.p), F)
    b = C.norms(InterpEval(X6, F.u, F.p, p_shift=3.7), F)
    assert abs(a["pL2r"] - b["pL2r"]) < 1e-12 and abs(a["pL2"] - b["pL2"]) < 1e-12
    c = C.norms(InterpEval(X6, F.u, F.p, p_shift=3.7), F, pressure="determined")
    assert abs(c["pL2r"] - 3.7 * math.sqrt(a["area_r"])) < 1e-2


@fast
def test_weighted_area_and_quadrature():
    V, T = C.square_mesh(8)
    ev = InterpEval(affine_X6(V, T), lambda R, Z: np.zeros(R.shape + (3,)), lambda R, Z: 0 * R)
    assert abs(C.area_r(ev) - 0.5) < 1e-14
    # int r^5 z^4 r dr dz over the unit square = 1/7 * 1/5, through the norm of a known field
    q, w = C.tri_rule(10)
    x, d = ev.geometry(q)
    val = float((w[None] * d * x[..., 0] ** 6 * x[..., 1] ** 4).sum())
    assert abs(val - 1 / 35) < 1e-14


@fast
def test_sphere_mesh_geometry_isoparametric_order_4_affine_order_2():
    errs_iso, errs_aff = [], []
    for k in (1, 2, 3):
        V, T = C.sphere_mesh(k)
        assert C.all_boundary_triangles(T) == 0
        X6 = C.sphere_p2_nodes(k)
        z = lambda R, Z: np.zeros(R.shape + (3,))  # noqa: E731
        errs_iso.append(abs(C.area_r(InterpEval(X6, z, lambda R, Z: 0 * R)) - 42.0))
        errs_aff.append(abs(C.area_r(InterpEval(affine_X6(V, T), z, lambda R, Z: 0 * R)) - 42.0))
        # orientation and exact circles
        assert np.all(np.linalg.det(np.stack([V[T][:, 1] - V[T][:, 0], V[T][:, 2] - V[T][:, 0]], -1)) > 0)
        rad = np.hypot(X6[..., 0], X6[..., 1])
        on1 = np.abs(rad[:, :3] - 1) < 1e-12
        for kk, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            both = on1[:, i] & on1[:, j]
            assert np.all(np.abs(rad[both, 3 + kk] - 1) < 1e-12)
    assert abs(orders(errs_iso)[-1] - 4) < 0.1 and abs(orders(errs_aff)[-1] - 2) < 0.1
    # nested in the parameter space: every vertex of level k is a vertex of level k+1
    V1, _ = C.sphere_mesh(1)
    V2, _ = C.sphere_mesh(2)
    s2 = {tuple(np.round(v, 10) + 0.0) for v in V2}
    assert all(tuple(np.round(v, 10) + 0.0) in s2 for v in V1)


@fast
def test_mesh_families_counts_nesting_and_corner_condition():
    for N in (8, 16):
        for mk in (C.square_mesh, C.annulus_rect_mesh):
            V, T = mk(N)
            assert len(T) == 2 * N * N and C.all_boundary_triangles(T) == 0
    for v in ("coarse", "fine", "dist_a", "dist_b"):
        V, T = C.pipe_mesh(v)
        assert C.all_boundary_triangles(T) == 0
        assert np.all(V[:, 0] >= 0) and np.any(np.abs(V[:, 0]) < 1e-15)
    Va, _ = C.square_distorted(1)
    assert np.allclose(sorted(Va[np.abs(Va[:, 0]) < 1e-14, 1]), np.linspace(0, 1, 9))   # axis stays on r = 0


@fast
def test_axis_nodal_check():
    P = np.array([[0.0, 0.1], [0.0, 0.5], [0.3, 0.2]])
    U = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [5.0, 5.0, 5.0]])
    assert C.axis_nodal_max(P, U, 2.0) == 0.0
    U[1, 2] = 1e-3
    assert abs(C.axis_nodal_max(P, U, 2.0) - 5e-4) < 1e-15


@fast
def test_discrete_continuity_residual_zero_for_divergence_free_p2_field():
    V, T = C.square_mesh(8)
    X6 = affine_X6(V, T)
    u = lambda R, Z: np.stack([R * Z, -Z ** 2, R], -1)      # noqa: E731  div = z + z - 2z = 0, in P2
    r = C.discrete_continuity_residual(InterpEval(X6, u, lambda R, Z: 0 * R), V)
    assert r["rq_max_rel"] < 1e-14 and r["q_max_rel"] < 1e-14
    u2 = lambda R, Z: np.stack([R * Z, -Z ** 2 / 2, R], -1)  # noqa: E731  div = z/2 != 0
    r2 = C.discrete_continuity_residual(InterpEval(X6, u2, lambda R, Z: 0 * R), V)
    assert r2["rq_max_rel"] > 1e-3


@fast
def test_reaction_patch_values():
    # rigid rotation u_theta = r: zero Cauchy stress, torque 0 on the inner cylinder
    V, T = C.annulus_rect_mesh(8)
    X6 = affine_X6(V, T)
    rot = InterpEval(X6, lambda R, Z: np.stack([0 * R, 0 * R, R], -1), lambda R, Z: 0 * R)
    inner = lambda R, Z: np.abs(R - 1) < 1e-12   # noqa: E731
    assert abs(C.reaction(rot, lambda R, Z: 0.7 + 0 * R, inner, "torque_z")) < 1e-13
    # circular Couette interpolant: torque converges to the closed form -28/15 per radian
    F = C.load_field(entry("couette_circular"), ROOT)
    errs = []
    for N in (8, 16, 32):
        V, T = C.annulus_rect_mesh(N)
        ev = InterpEval(affine_X6(V, T), F.u, F.p)
        errs.append(abs(C.reaction(ev, F.mu, inner, "torque_z") + 28 / 15))
    assert errs[-1] < 1e-3 and errs[-1] < errs[0]
    # Hagen-Poiseuille (exact in P2 x P1): pressure-driven wall force = pressure drop x area = 2 per radian
    H = C.load_field(entry("hp_pipe"), ROOT)
    V, T = C.pipe_mesh("dist_a")
    ev = InterpEval(affine_X6(V, T), H.u, H.p)
    wall = lambda R, Z: np.abs(R - 1) < 1e-12   # noqa: E731

    def on_z(v):
        return lambda a, m, b: abs(a[1] - v) < 1e-12 and abs(b[1] - v) < 1e-12
    for form in ("lap", "sym"):
        tr = [(on_z(0.0), H.traction(form)), (on_z(2.0), H.traction(form))]
        assert abs(C.reaction(ev, H.mu, wall, "force_z", traction_edges=tr) - 2.0) < 1e-12


@fast
def test_energy_functionals_of_rigid_rotation():
    V, T = C.square_mesh(8)
    ev = InterpEval(affine_X6(V, T), lambda R, Z: np.stack([0 * R, 0 * R, R], -1), lambda R, Z: 0 * R)
    assert abs(C.dissipation_r(ev, "lap") - 1.0) < 1e-13          # mu int (1 + 1) r = 1
    assert abs(C.dissipation_r(ev, "sym")) < 1e-13               # zero strain
    assert abs(C.kinetic_energy_r(ev) - 0.125) < 1e-13           # 1/2 int r^2 r = 1/8


@fast
def test_cavity_lid_data():
    R = np.linspace(0, 1, 7)
    for Z, nz in ((np.ones(7), True), (np.zeros(7), False)):
        g = C.cavity_lid(R, Z)
        assert np.all(g[:, 1] == 0) and (np.any(g[:, 0] != 0) == nz)
    g = C.cavity_lid(np.array([0.0, 1.0]), np.array([0.7, 0.7]))
    assert np.all(g == 0)                                        # axis (odd components) and wall


@fast
def test_flux_defect_r_of_quadratic_data_is_exact():
    # g = (0, 1 - r^2, 0) on the pipe outlet: P2-exact, so the interpolated flux equals int_0^1 r (1 - r^2) dr = 1/4
    g = lambda R, Z: np.stack([0 * R, 1 - R ** 2, 0 * R], -1)   # noqa: E731
    edges = [(np.array([a, 2.0]), np.array([(a + b) / 2, 2.0]), np.array([b, 2.0]), np.array([0.0, 1.0]))
             for a, b in zip(np.linspace(0, 1, 5)[:-1], np.linspace(0, 1, 5)[1:])]
    d, ref = C.flux_defect_r(edges, g)
    assert abs(d - 0.25) < 1e-15 and abs(ref - 0.25) < 1e-15


@fast
def test_runner_independent_axis_and_dirichlet_checks():
    """The check module locates the axis and Dirichlet nodes itself (count asserted against the fixed mesh)."""
    V, T = C.square_mesh(8)
    assert C.expected_axis_nodes(V, T) == 17                      # 9 vertices + 8 edge nodes on r = 0
    F = C.load_field(entry("axi_dirichlet"), ROOT)
    ev = InterpEval(affine_X6(V, T), F.u, F.p)
    assert C.axis_nodal_max_ev(ev, V, T, 1.0) < 1e-15            # the exact field is odd: u_r = u_theta = 0 on r = 0
    nonax = lambda a, m, b: not (abs(a[0]) < 1e-12 and abs(b[0]) < 1e-12) and (   # noqa: E731
        abs(a[0] - 1) < 1e-12 and abs(b[0] - 1) < 1e-12 or abs(a[1]) < 1e-12 and abs(b[1]) < 1e-12
        or abs(a[1] - 1) < 1e-12 and abs(b[1] - 1) < 1e-12)
    assert C.dirichlet_nodal_defect_ev(ev, nonax, F.u) < 1e-15
    # a nonzero u_theta at the axis EDGE nodes only (vertices exact) is seen
    bad = InterpEval(affine_X6(V, T), F.u, F.p)
    on_ax_mid = np.abs(bad.X6[:, 3:, 0]) < 1e-12
    bad.uv[:, 3:, 2][on_ax_mid] = 1e-3
    assert C.axis_nodal_max_ev(bad, V, T, 1.0) >= 1e-3
    bad.uv[:, 3:, 2][on_ax_mid] = 0.0
    bad.uv[:, 1:2, 1] += 1e-4 * (np.abs(bad.X6[:, 1:2, 1] - 1) < 1e-12)      # perturb some z = 1 vertices
    assert C.dirichlet_nodal_defect_ev(bad, nonax, F.u) > 1e-6


@fast
def test_applied_load_and_lap_residual_on_poiseuille():
    H = C.load_field(entry("hp_pipe"), ROOT)
    V, T = C.pipe_mesh("dist_b")
    ev = InterpEval(affine_X6(V, T), H.u, H.p)
    inlet = lambda a, m, b: abs(a[1]) < 1e-12 and abs(b[1]) < 1e-12   # noqa: E731
    assert abs(C.applied_boundary_load(ev, inlet, H.traction("lap"), "force_z") - 2.0) < 1e-13   # p(0) * 1/2
    wall = lambda R, Z: np.abs(R - 1) < 1e-12   # noqa: E731
    tr = [(inlet, H.traction("lap")), (lambda a, m, b: abs(a[1] - 2) < 1e-12 and abs(b[1] - 2) < 1e-12, H.traction("lap"))]
    assert abs(C.reaction(ev, H.mu, wall, "force_z", traction_edges=tr, residual_form="lap") - 2.0) < 1e-12
