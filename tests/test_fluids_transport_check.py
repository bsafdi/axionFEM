"""Self-tests of the scalar-transport and Boussinesq check modules benchmarks/fluids/cases/scalar_transport_check.py
and benchmarks/fluids/cases/f4a_heat_check.py (the check modules judge every solver result, so they are tested on
their own).

No solver.  The discrete "solutions" are P2 (and P1 pressure) nodal INTERPOLANTS of the exact fields on the fixed
nested meshes, whose error orders are known (3 / 2 for P2 in L2 / H1, 2 for the P1 pressure); synthetic maps test the
Taylor-test and Newton-order diagnostics.  Seconds: `pytest -m fast`.
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
import scalar_transport_check as S  # noqa: E402
import f4a_heat_check as H  # noqa: E402

fast = pytest.mark.fast


def entry(name):
    p = ROOT / "tools" / "fluids_mms" / "out" / f"{name}.json"
    return {"path": f"tools/fluids_mms/out/{name}.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


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
    """P2 scalar T, P2 velocity (any number of components), P1 pressure nodal interpolants on an affine mesh."""

    def __init__(self, V, T, Tfun, u=None, p=None):
        self.V, self.T = np.asarray(V, float), np.asarray(T)
        P = self.V[self.T]
        self.nodes = np.concatenate([P, 0.5 * (P[:, [0, 1, 2]] + P[:, [1, 2, 0]])], axis=1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self.Tv = np.nan_to_num(Tfun(self.nodes[..., 0], self.nodes[..., 1]))
            self.uv = np.nan_to_num(u(self.nodes[..., 0], self.nodes[..., 1])) if u else None
            self.pv = p(P[..., 0], P[..., 1]) if p else None
        self.n_elements = len(self.T)

    def _J(self):
        P = self.V[self.T]
        return P, np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=-1)

    def geometry(self, q):
        P, J = self._J()
        x = P[:, None, 0, :] + np.einsum("eda,qa->eqd", J, q)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        return x, np.repeat(np.abs(det)[:, None], len(q), axis=1)

    def _grad(self, q):
        N, dN = _p2(q)
        _, J = self._J()
        return N, np.einsum("qka,ead->eqkd", dN, np.linalg.inv(J))

    def scalar(self, q):
        N, G = self._grad(q)
        return np.einsum("qk,ek->eq", N, self.Tv), np.einsum("eqkd,ek->eqd", G, self.Tv)

    def velocity(self, q):
        N, G = self._grad(q)
        return np.einsum("qk,ekc->eqc", N, self.uv), np.einsum("eqkd,ekc->eqcd", G, self.uv)

    def pressure(self, q):
        L = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], 1)
        return np.einsum("qk,ek->eq", L, self.pv)


def _orders(es, ratio=2.0):
    return [math.log(es[i] / es[i + 1]) / math.log(ratio) for i in range(len(es) - 1)]


@fast
def test_box_mesh_equals_the_planar_stokes_builder():
    import f1a_stokes_check as C1
    for k in range(3):
        V, T = S.box_mesh(8, 8, k)
        V1, T1 = C1.structured_mesh(8 * 2 ** k)
        assert np.array_equal(np.asarray(V), np.asarray(V1)) and np.array_equal(np.asarray(T), np.asarray(T1))
        assert S.all_boundary_triangles(T) == 0
    V, T = S.box_mesh(16, 8, 1, (0, 2, 0, 1))
    assert S.all_boundary_triangles(T) == 0 and len(T) == 16 * 8 * 2 * 4
    Vc, _ = S.distorted_mesh(1)
    Vf, Tf = S.distorted_mesh(2)
    assert S.verify_mesh(Vf, Tf, Vf, Tf, Vc)["nested"]


@fast
@pytest.mark.parametrize("name", ["heat_sq_pe1", "heat_axi_pe1", "heat_axi_pe1e6"])
def test_scalar_interpolant_orders(name):
    fld = S.load_field(entry(name))
    axi = fld.geometry == "axisymmetric"
    es = {"TL2": [], "TH1": []}
    for k in range(4):
        V, T = S.box_mesh(8, 8, k)
        nr = S.scalar_norms(InterpEval(V, T, fld.T), fld)
        es["TL2"].append(nr["TL2w" if axi else "TL2"])
        es["TH1"].append(nr["TH1w" if axi else "TH1"])
        if axi:
            assert nr["TL2_unw"] > 0
    assert abs(_orders(es["TL2"])[-1] - 3) < 0.1
    assert abs(_orders(es["TH1"])[-1] - 2) < 0.1


@fast
@pytest.mark.parametrize("name", ["heat_sq_patch_pe1e6", "heat_axi_patch_pe1e6"])
def test_patch_field_is_in_p2(name):
    fld = S.load_field(entry(name))
    V, T = S.distorted_mesh(1)
    nr = S.scalar_norms(InterpEval(V, T, fld.T), fld)
    assert max(nr["TL2w" if fld.geometry == "axisymmetric" else "TL2"],
               nr["TH1w" if fld.geometry == "axisymmetric" else "TH1"]) < 1e-12


@fast
@pytest.mark.parametrize("name", ["graetz_plates", "graetz_pipe"])
def test_graetz_modes_exact_and_derivatives(name):
    fld = S.load_field(entry(name))
    rng = np.random.default_rng(1)
    if fld.geometry == "planar":
        A, B = rng.uniform(0, 2, 20), rng.uniform(0, 1, 20)
        wall = fld.T(np.linspace(0, 2, 11), np.ones(11))
    else:
        A, B = rng.uniform(0, 1, 20), rng.uniform(0, 2, 20)
        wall = fld.T(np.ones(11), np.linspace(0, 2, 11))
    assert np.max(np.abs(wall)) < 1e-14
    g = fld.grad_T(A, B)
    h = 1e-6
    fdA = (fld.T(A + h, B) - fld.T(A - h, B)) / (2 * h)
    fdB = (fld.T(A, B + h) - fld.T(A, B - h)) / (2 * h)
    assert np.max(np.abs(g[..., 0] - fdA)) < 1e-8 and np.max(np.abs(g[..., 1] - fdB)) < 1e-8


@fast
def test_boundary_integral_and_extrema():
    V, T = S.box_mesh(8, 8, 1)
    ev = InterpEval(V, T, lambda A, B: A ** 2 + B)
    top = lambda X: np.abs(X[..., 1] - 1) < 1e-12  # noqa: E731
    # int_{y=1} (x^2 + 1) dx = 4/3; outward normal (0, 1): int dT/dn = int 1 dx = 1
    assert abs(S.boundary_integral(ev, top, lambda X, Tv, dT, n: Tv, False) - 4 / 3) < 1e-13
    assert abs(S.boundary_integral(ev, top, lambda X, Tv, dT, n: (dT * n).sum(-1), False) - 1) < 1e-13
    left = lambda X: np.abs(X[..., 0]) < 1e-12  # noqa: E731
    # outward normal (-1, 0): dT/dn = -2x = 0 on x = 0; axisymmetric r ds on r = 1 of T = z: int_0^1 z dz = 1/2
    assert abs(S.boundary_integral(ev, left, lambda X, Tv, dT, n: (dT * n).sum(-1), False)) < 1e-13
    ev2 = InterpEval(V, T, lambda A, B: B)
    r1 = lambda X: np.abs(X[..., 0] - 1) < 1e-12  # noqa: E731
    assert abs(S.boundary_integral(ev2, r1, lambda X, Tv, dT, n: Tv, True) - 0.5) < 1e-13
    ext = S.sampled_extrema(ev)
    assert abs(ext["max"] - 2) < 1e-14 and abs(ext["min"]) < 1e-14


@fast
def test_uniform_flux_nusselt_closed_form():
    for name, nx, ny, box, nu in (("flux_plates", 16, 8, (0, 2, 0, 1), 140 / 17), ("flux_pipe", 8, 16, (0, 1, 0, 2), 48 / 11)):
        fld = S.load_field(entry(name))
        assert abs(fld.data["nusselt_closed_form"]["value"] - nu) < 1e-12
        errs = []
        for k in (1, 2, 3):
            V, T = S.box_mesh(nx, ny, k, box)
            wall = "y1" if fld.geometry == "planar" else "r1"
            errs.append(abs(S.nusselt_uniform_flux(InterpEval(V, T, fld.T), fld, wall) - nu))
        assert errs[-1] < 1e-5 and errs[-1] < errs[0]


@fast
def test_flux_from_residual_and_balance():
    R = np.array([1.0, -2.0, 0.5, 3.0])
    assert S.flux_from_residual(R, [1, 3]) == -1.0
    assert S.flux_from_residual(R, weights=[0, 1, 0, 1]) == -1.0
    assert S.balance_defect(1.0, 2.0, 4.0, 1.0) == 0.0
    assert S.balance_defect(1.0, 2.0, 4.0, 1.1) > 1e-2


@fast
def test_natural_outflow_robin_equals_neumann_for_exact_T():
    fld = S.load_field(entry("heat_sq_pe1"))
    V, T = S.box_mesh(8, 8, 3)
    ev = InterpEval(V, T, fld.T)
    bd = fld.boundaries["y1"]
    on = S.edge_predicate("y1", (0, 1, 0, 1))
    rob = S.natural_outflow(ev, [(on, "robin", bd["h"]["value"], S.compile_expr(bd["T_ext"], fld.coords))], False)
    neu = S.natural_outflow(ev, [(on, "neumann", S.compile_expr(bd["q_out"], fld.coords))], False)
    assert abs(rob - neu) < 1e-6 * abs(neu)


@fast
def test_taylor_test_slopes():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    x0, d = rng.standard_normal(6), rng.standard_normal(6)
    F = lambda x: A @ x + 0.3 * x ** 3  # noqa: E731
    J = A + np.diag(0.9 * x0 ** 2)
    R0 = F(x0)
    eps = [10.0 ** (-k) for k in range(1, 7)]
    good = S.taylor_test(R0, lambda e: F(x0 + e * d), J @ d, eps)
    bad = S.taylor_test(R0, lambda e: F(x0 + e * d), A @ d, eps)
    assert abs(good["median_slope"] - 2) < 0.1 and abs(bad["median_slope"] - 1) < 0.1


@fast
def test_newton_order_median():
    hist = [{"kind": "newton", "residual": r} for r in (1e-1, 1e-2, 1e-4, 1e-8)]
    assert abs(H.newton_order_median(hist) - 2) < 1e-9
    lin = [{"kind": "newton", "residual": 10.0 ** -k} for k in range(1, 8)]
    assert abs(H.newton_order_median(lin) - 1) < 1e-9 and H.newton_iterations(lin) == 7


@fast
@pytest.mark.parametrize("name", ["bous_sq", "bous_axi"])
def test_flow_norms_interpolant_orders(name):
    fld = H.load_bous(entry(name))
    axi = fld.geometry == "axisymmetric"
    sfx = "w" if axi else ""
    es = {k: [] for k in ("uL2", "uH1", "TL2", "pL2m")}
    for k in range(4):
        V, T = S.box_mesh(8, 8, k)
        nr = H.flow_norms(InterpEval(V, T, fld.T, fld.u, fld.p), fld)
        for key in es:
            es[key].append(nr[key + sfx])
    for key, p in (("uL2", 3), ("uH1", 2), ("TL2", 3), ("pL2m", 2)):
        assert abs(_orders(es[key])[-1] - p) < 0.15, (key, es[key])


@fast
def test_hydrostatic_fields_and_cavity_functionals():
    fld = H.load_bous(entry("hydro_sq_strat"))
    V, T = S.box_mesh(8, 8, 1)
    ev = InterpEval(V, T, fld.T, fld.u, fld.p)
    dv = H.discrete_velocity_norms(ev, False, u_scale=H.hydrostatic_velocity_scale(fld))
    assert dv["magnitudes"]["uL2"] == 1000.0
    assert dv["uL2"] == 0 and dv["u_nodal_max"] == 0
    nr = H.flow_norms(ev, fld)
    assert nr["TL2"] < 1e-14                    # T linear: in P2
    cf = H.cavity_functionals(InterpEval(V, T, lambda A, B: 0.5 - A, lambda A, B: np.stack([0 * A, A * (1 - A) * B * (1 - B)], -1)))
    assert cf["vleft"] > 0 and abs(cf["Tsq"] - 1 / 12) < 1e-14


@fast
def test_magnitudes_and_exact_fluxes():
    fld = S.load_field(entry("heat_sq_pe1"))
    V, T = S.box_mesh(8, 8, 1)
    nr = S.scalar_norms(InterpEval(V, T, fld.T), fld)
    assert set(nr["magnitudes"]) == {"TL2", "TH1", "Tstream"}
    assert nr["magnitudes"]["TL2"] == nr["T_magnitude"] and nr["magnitudes"]["Tstream"] > 0
    g = S.load_field(entry("graetz_plates"))
    ex = sum(-g.kappa_value * amp * float(m["dphi_wall"]) * (1 - math.exp(-lm * 2.0)) / lm
             for (lm, amp, ch), m in zip(g.modes, g.data["modes"]))
    assert abs(S.graetz_wall_flux_exact(g) - ex) < 1e-12 * abs(ex)
    a = S.load_field(entry("cond_annulus"))
    Va, Ta = S.box_mesh(4, 8, 2, (0.5, 1, 0, 1))
    eva = InterpEval(Va, Ta, a.T)
    q = S.exact_boundary_flux(eva, a, S.edge_predicate("r0", (0.5, 1, 0, 1)), True)
    assert abs(q - 1.0) < 1e-12
    h = H.load_bous(entry("hydro_sq_strat"))
    assert abs(H.hydrostatic_velocity_scale(h) - 1000.0) < 1e-9
    b = H.load_bous(entry("bous_axi"))
    nb = H.flow_norms(InterpEval(V, T, b.T, b.u, b.p), b)
    assert {"uL2w", "uH1w", "pL2mw", "TL2w", "TH1w", "uL2_unw", "TL2_unw"} <= set(nb["magnitudes"])
