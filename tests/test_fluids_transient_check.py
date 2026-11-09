"""Self-tests of the transient Navier-Stokes check module benchmarks/fluids/cases/f3a_transient_check.py (the check
module judges every solver result, so it is tested on its own).

No solver.  Discrete "solutions" are P2/P1 nodal interpolants of the generated fields; time-level sequences are
interpolants of exact fields at the step times.  Seconds: `pytest -m fast tests/test_fluids_transient_check.py`.
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
import f3a_transient_check as C  # noqa: E402
from test_fluids_stokes_planar_check import _p2  # noqa: E402

fast = pytest.mark.fast


def entry(name):
    p = ROOT / "tools" / "fluids_mms" / "out" / f"{name}.json"
    return {"path": f"tools/fluids_mms/out/{name}.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


class InterpEval:
    """P2 velocity (2 or 3 components) / P1 pressure nodal interpolant on an affine mesh; axisymmetric: coordinates
    (r, z), component derivatives d u_i / d (r, z)_j; u_r = u_t = 0 at axis nodes."""

    def __init__(self, V, T, u, p):
        self.V, self.T = np.asarray(V, float), np.asarray(T)
        P = self.V[self.T]
        nodes = np.concatenate([P, 0.5 * (P[:, [0, 1, 2]] + P[:, [1, 2, 0]])], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.uv = np.nan_to_num(np.asarray(u(nodes[..., 0], nodes[..., 1]), float))
        self.pv = np.asarray(p(P[..., 0], P[..., 1]), float)
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


def interp(fld, V, T, t, scale=1.0):
    return InterpEval(V, T, lambda A, B: scale * fld.u(A, B, t), lambda A, B: scale * fld.p(A, B, t))


def _orders(e):
    e = np.asarray(e)
    return np.log2(e[:-1] / e[1:])


@fast
def test_load_field_refuses_wrong_hash():
    e = entry("f3a_tr_sq_poly")
    C.load_field(e)
    with pytest.raises(RuntimeError):
        C.load_field({**e, "sha256": "0" * 64})


@fast
@pytest.mark.parametrize("name", ["f3a_tr_sq_poly", "f3a_tr_axi_poly", "f3a_tr_sq_adapt"])
def test_space_exact_fields_are_reproduced_by_their_interpolant(name):
    """The temporal-MMS premise: u(t) in P2 and p(t) in P1 at every t, so the interpolant has zero error (and the
    semi-discrete Galerkin solution is the exact solution)."""
    fld = C.load_field(entry(name))
    V, T = C.structured_mesh(8)
    for t in (0.0, 0.37, 1.0):
        e = C.errors(interp(fld, V, T, t), fld, t)
        for k, v in e.items():
            if k.startswith(("uL2", "uH1", "pL2", "divL2")) and "norm" not in k:
                assert v < 2e-13, (name, t, k, v)


@fast
def test_interpolant_orders_taylor_green_and_womersley():
    tg = C.load_field(entry("f3a_tr_taylor_green"))
    rows = [C.errors(interp(tg, *C.structured_mesh(N), 0.1), tg, 0.1) for N in (8, 16, 32)]
    for k, p in (("uL2", 3), ("uH1", 2), ("pL2", 2)):
        assert np.allclose(_orders([r[k] for r in rows]), p, atol=0.12), (k, _orders([r[k] for r in rows]))
    w2 = C.load_field(entry("f3a_tr_womersley_2d"))
    rows = [C.errors(interp(w2, *C.rect_mesh(2, 4, k, 0, 1, -1, 1), 0.3), w2, 0.3) for k in (2, 3, 4)]
    assert np.allclose(_orders([r["uL2"] for r in rows]), 3, atol=0.12)
    wa = C.load_field(entry("f3a_tr_womersley_axi"))
    rows = [C.errors(interp(wa, *C.rect_mesh(2, 2, k, 0, 1, 0, 1), 0.3), wa, 0.3) for k in (2, 3, 4)]
    for k, p in (("uL2w", 3), ("uH1w", 2), ("uL2_unw", 3), ("uH1_unw", 2)):
        assert np.allclose(_orders([r[k] for r in rows]), p, atol=0.15), (k, _orders([r[k] for r in rows]))


@fast
def test_womersley_traction_is_sigma_n():
    """Planar Womersley at x = 1 (n = (1, 0)): lap g = (-p, 0); sym g = (-p, mu dU/dy)."""
    w2 = C.load_field(entry("f3a_tr_womersley_2d"))
    A, B, t = np.array([1.0, 1.0]), np.array([-0.3, 0.6]), 0.27
    gl = w2.traction("lap", (1, 0))(A, B, t)
    gs = w2.traction("sym", (1, 0))(A, B, t)
    assert np.allclose(gl[:, 0], -w2.p(A, B, t)) and np.allclose(gl[:, 1], 0)
    assert np.allclose(gs[:, 1], w2.mu * w2.grad_u(A, B, t)[:, 0, 1])


@fast
def test_bdf_and_extrapolation_coefficients_variable_step():
    rng = np.random.default_rng(1)
    for _ in range(20):
        h0, h1 = rng.uniform(0.2, 1.0, 2)
        ts = [0.0, h1, h1 + h0]
        for order in (1, 2):
            al = C.bdf_coefficients(order, [h0, h1])
            for deg in range(order + 1):          # exact for polynomials of degree <= order
                u = [t ** deg for t in ts[::-1]]
                du = deg * ts[-1] ** (deg - 1) if deg else 0.0
                assert sum(a * u[j] for j, a in enumerate(al)) / h0 == pytest.approx(du, abs=1e-12)
        be = C.extrapolation_coefficients(2, [h0, h1])
        assert be[0] * ts[1] + be[1] * ts[0] == pytest.approx(ts[2])   # exact for linear functions
    assert C.bdf_coefficients(2, [0.1, 0.1]) == pytest.approx([1.5, -2.0, 0.5])


@fast
def test_cyclic_grid_and_step_stats():
    ts = C.time_grid({"kind": "cyclic", "pattern": [1.0, 1.25, 0.8, 1.1, 0.9]}, 1.0, 40)
    st = C.step_stats(ts)
    assert st["n_steps"] == 40 and st["t_end"] == pytest.approx(1.0)
    assert st["dt_max_ratio"] == pytest.approx(1.1 / 0.8)
    assert C.step_stats(C.time_grid({"kind": "uniform"}, 1.0, 10))["dt_max_ratio"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        C.time_grid({"kind": "cyclic", "pattern": [1, 2, 3]}, 1.0, 10)


@fast
def test_rect_mesh_nested_and_no_all_boundary_triangle():
    for nx, ny, box in ((2, 4, (0, 1, -1, 1)), (2, 2, (0, 1, 0, 1)), (4, 4, (0, 1, 0, 1))):
        prev = None
        for k in range(4):
            V, T = C.rect_mesh(nx, ny, k, *box)
            assert C.all_boundary_triangles(T) == 0
            if prev is not None:
                assert len(T) == 4 * prev
            prev = len(T)


@fast
def test_discrete_divergence_zero_for_space_exact_positive_for_interpolant():
    V, T = C.structured_mesh(8)
    for name in ("f3a_tr_sq_poly", "f3a_tr_axi_poly"):
        fld = C.load_field(entry(name))
        d = C.discrete_divergence(interp(fld, V, T, 0.4), V, fld.axi)
        assert d["div_dual_rel"] < 1e-13, name
    pj = C.load_field(entry("f3a_tr_sq_proj"))
    rel = [C.discrete_divergence(interp(pj, *C.structured_mesh(N), 0.0), C.structured_mesh(N)[0], False)
           ["div_dual_rel"] for N in (8, 16)]
    assert rel[0] > 1e-4 and rel[1] < rel[0]          # the nodal interpolant is NOT discretely divergence-free
    # ... whereas the (separable) Taylor-Green interpolant is, on the structured family (why the projection case
    # uses its own field)
    tg = C.load_field(entry("f3a_tr_taylor_green"))
    assert C.discrete_divergence(interp(tg, V, T, 0.0), V, False)["div_dual_rel"] < 1e-12


@fast
def test_effectivity_definition():
    fld = C.load_field(entry("f3a_tr_sq_adapt"))
    V, T = C.structured_mesh(8)
    t = 0.4
    ex = fld.u
    pert = lambda A, B: ex(A, B, t) + 1e-3 * np.stack([A * B, A - B], -1)          # noqa: E731
    un = InterpEval(V, T, pert, lambda A, B: 0 * A)
    e_vec = lambda A, B: 1e-3 * np.stack([A * B, A - B], -1)                       # noqa: E731
    for s, want in ((1.0, 1.0), (2.0, 2.0), (-0.5, -0.5)):
        est = InterpEval(V, T, lambda A, B: s * e_vec(A, B), lambda A, B: 0 * A)
        assert C.effectivity(est, un, fld, t)["effectivity"] == pytest.approx(want, rel=1e-10)
    orth = InterpEval(V, T, lambda A, B: np.stack([-(A - B), A * B], -1) * 1e-3, lambda A, B: 0 * A)
    assert abs(C.effectivity(orth, un, fld, t)["effectivity"]) < 1e-12
    # est_L2 is the L2 norm of the estimate (the number a solver's REPORTED error norm must equal: T2 est_norm_ratio)
    est = InterpEval(V, T, lambda A, B: 2.0 * e_vec(A, B), lambda A, B: 0 * A)
    assert C.effectivity(est, un, fld, t)["est_L2"] == pytest.approx(
        math.sqrt(C.l2_inner(est, est, False)), rel=1e-12)


@fast
def test_energy_identity_terms_on_a_scaled_sequence():
    """u^n = c(t_n) U with U the planar energy field's... a fixed P2 field: the time term must be
    rho/dt sum alpha_j c_{n-j} c_n ||U||^2, exactly; BDF2 of a quadratic c(t) gives rho c'(t_n) c_n ||U||^2."""
    fld = C.load_field(entry("f3a_tr_sq_energy"))
    V, T = C.structured_mesh(8)
    U = lambda A, B: np.stack([A * (1 - A) * B * (1 - B), A * A * B * (1 - B) * (1 - A)], -1)  # noqa: E731
    c = lambda t: 1 + 2 * t - 3 * t * t                                                       # noqa: E731
    ts = [0.0, 0.07, 0.16, 0.22]
    evs = [InterpEval(V, T, lambda A, B, tt=tt: c(tt) * U(A, B), lambda A, B: 0 * A) for tt in ts]
    nrm2 = C.l2_inner(evs[-1], evs[-1], False) / c(ts[-1]) ** 2
    r = C.step_energy_identity(evs, ts, 3, 2, fld, "lap", "newton")
    assert r["time"] == pytest.approx(fld.rho * (2 - 6 * ts[3]) * c(ts[3]) * nrm2, rel=1e-12)
    rr = C.step_energy_identity(evs, ts, 3, 2, fld, "lap", "newton", rho_mass=1.0)
    assert rr["time"] == pytest.approx(r["time"] / fld.rho, rel=1e-12)
    # the semi-implicit advecting velocity is the variable-step extrapolation: linear c -> exact w
    cl = lambda t: 1 + 2 * t                                                                    # noqa: E731
    evl = [InterpEval(V, T, lambda A, B, tt=tt: cl(tt) * U(A, B), lambda A, B: 0 * A) for tt in ts]
    a = C.step_energy_identity(evl, ts, 3, 2, fld, "lap", "semi")
    b = C.step_energy_identity(evl, ts, 3, 2, fld, "lap", "newton")
    assert a["conv"] == pytest.approx(b["conv"], rel=1e-10, abs=1e-16)


@fast
def test_kinetic_energy_and_solution_scale_taylor_green():
    tg = C.load_field(entry("f3a_tr_taylor_green"))
    V, T = C.structured_mesh(32)
    ke = C.kinetic_energy(interp(tg, V, T, 0.0), tg.rho, False)
    assert ke == pytest.approx(tg.rho / 4, rel=1e-5)
    assert C.solution_scale(tg, 0.2) == pytest.approx(math.sqrt(0.5), rel=1e-3)
