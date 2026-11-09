"""Fast self-tests of the MHD check module (benchmarks/fluids/cases/f13a_mhd_check.py) and its series module.
No solver: exact fields wrapped in an Evaluator, synthetic discrete fields, and the fixed mesh builders."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import f13a_mhd_check as C   # noqa: E402
import f13a_mhd_series as S  # noqa: E402

pytestmark = pytest.mark.fast
MU, SIGMA, G = 0.5, 2.0, 1.0


class ExactEvaluator:
    """Evaluator protocol over closed-form fields on a fixed mesh (affine elements)."""

    def __init__(self, V, T, u, gu, p, phi, gphi):
        P = np.asarray(V, float)[np.asarray(T)]
        self.a0, self.e1, self.e2 = P[:, 0], P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        self.n_elements = len(T)
        self.f = (u, gu, p, phi, gphi)

    def geometry(self, ref):
        ref = np.asarray(ref, float)
        x = self.a0[:, None] + ref[None, :, :1] * self.e1[:, None] + ref[None, :, 1:2] * self.e2[:, None]
        det = np.abs(self.e1[:, 0] * self.e2[:, 1] - self.e1[:, 1] * self.e2[:, 0])
        return x, np.repeat(det[:, None], len(ref), axis=1)

    def _xy(self, ref):
        x, _ = self.geometry(ref)
        return x[..., 0], x[..., 1]

    def velocity(self, ref):
        X, Y = self._xy(ref)
        return self.f[0](X, Y), self.f[1](X, Y)

    def pressure(self, ref):
        return self.f[2](*self._xy(ref))

    def potential(self, ref):
        X, Y = self._xy(ref)
        return self.f[3](X, Y), self.f[4](X, Y)


def _zero_u(X, Y):
    return np.zeros(np.shape(X) + (3,))


def _zero_gu(X, Y):
    return np.zeros(np.shape(X) + (3, 2))


# ---------------------------------------------------------------------------------------------- meshes
def test_graded_nodes_first_cell_symmetry_and_uniform_limit():
    for Ha in (10.0, 100.0, 1000.0):
        y = C.graded_nodes(16, 1.0, 1.0 / Ha)
        assert len(y) == 17 and y[0] == -1.0 and y[-1] == 1.0 and y[8] == 0.0
        assert np.all(np.diff(y) > 0)
        assert np.allclose(y, -y[::-1], atol=1e-15)
        assert abs((y[1] - y[0]) - 1.0 / Ha) < 1e-9 * (1.0 / Ha) + 1e-15
    assert np.allclose(C.graded_nodes(16, 1.0, math.inf), np.linspace(-1, 1, 17))
    assert np.allclose(C.graded_nodes(16, 1.0, 0.5), np.linspace(-1, 1, 17))


@pytest.mark.parametrize("builder,Ha", [(C.duct_mesh, 1000.0), (C.duct_mesh, 0.0), (C.hartmann_strip_mesh, 100.0)])
def test_graded_families_nested_th_condition_and_counts(builder, Ha):
    prev = None
    for k in range(3):
        V, T = builder(Ha, k)
        rep = C.verify_mesh(V, T, V, T, V_coarser=prev)
        assert rep["hash_match"] and rep["all_boundary_triangles"] == 0
        if prev is not None:
            assert rep["nested"]
        prev = V
        n0 = len(builder(Ha, 0)[1])
        assert len(T) == n0 * 4 ** k
    # probe points used by the checks are vertices of every level
    V, _ = C.duct_mesh(Ha, 0) if builder is C.duct_mesh else C.hartmann_strip_mesh(Ha, 0)
    for P in ((0.0, 0.0), (1.0, 0.0)) if builder is C.duct_mesh else ((0.0, 0.0),):
        assert np.min(np.hypot(V[:, 0] - P[0], V[:, 1] - P[1])) < 1e-14


# ---------------------------------------------------------------------------------------------- fields
def _field_entry(name):
    p = ROOT / "tools/fluids_mms/out" / f"{name}.json"
    return {"path": f"tools/fluids_mms/out/{name}.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def test_field_sha_refused_when_tampered():
    e = _field_entry("mhd_mixed")
    C.load_field(e, ROOT)
    bad = dict(e, sha256="0" * 64)
    with pytest.raises(RuntimeError):
        C.load_field(bad, ROOT)


@pytest.mark.parametrize("name", ["mhd_mixed", "mhd_insulating"])
def test_norms_vanish_for_the_exact_field_and_see_a_perturbation(name):
    fld = C.load_field(_field_entry(name), ROOT)
    V, T = C.mms_mesh("structured", 8)
    ev = ExactEvaluator(V, T, fld.u, fld.grad_u, fld.p, fld.phi, fld.grad_phi)
    nr = C.norms(ev, fld, pressure="mean_free", phi="mean_free")
    for k in ("uL2", "uH1", "pL2", "phiL2", "phiH1", "JL2", "divL2"):
        assert nr[k] < 1e-12, (k, nr[k])
    ev2 = ExactEvaluator(V, T, lambda X, Y: fld.u(X, Y) + 1e-3, fld.grad_u, fld.p, fld.phi, fld.grad_phi)
    nr2 = C.norms(ev2, fld)
    assert abs(nr2["uL2"] - 1e-3 * math.sqrt(3)) < 1e-9
    assert nr2["JL2"] > 1e-4                      # J_h is recomputed from u_h by the check module (u x B)


def test_insulating_field_identities_hold_for_the_exact_solution():
    """Exact J of mhd_insulating: int J_perp = 0, J.grad phi integrates to 0 (div J = 0, J.n = 0), so the
    Lorentz-Joule defect is quadrature-small; the exact field is source free."""
    fld = C.load_field(_field_entry("mhd_insulating"), ROOT)
    V, T = C.mms_mesh("structured", 16)
    ev = ExactEvaluator(V, T, fld.u, fld.grad_u, fld.p, fld.phi, fld.grad_phi)
    I = C.integrals(ev, 1.6, [0.0, 1.2, 0.8], fld.Ez, fld.mu)
    assert C.net_current_rel(I) < 1e-12
    assert C.lorentz_joule_defect(I, fld.Ez) < 1e-11


# ---------------------------------------------------------------------------------------------- conservation tools
def test_weak_residual_zero_for_a_discrete_harmonic_field_and_has_teeth():
    """phi = x y (in P2), u = 0, sigma = 1: J = -grad phi, int J.grad N_i = -int grad(xy).grad N_i = 0 for every
    interior P2 node (Delta(xy) = 0 exactly, P2-exact quadrature); a non-harmonic phi = x^2 gives O(h^2) per node."""
    V, T = C.mms_mesh("structured", 8)
    onb = lambda P: min(P[0], P[1], 1 - P[0], 1 - P[1]) < 1e-12   # noqa: E731
    sig = lambda X, Y: np.ones_like(X)                              # noqa: E731
    B0 = lambda X, Y: np.zeros(np.shape(X) + (3,))                  # noqa: E731
    ev = ExactEvaluator(V, T, _zero_u, _zero_gu, lambda X, Y: 0 * X, lambda X, Y: X * Y,
                        lambda X, Y: np.stack([Y, X], axis=-1))
    r = C.weak_current_residual(ev, sig, B0, 0.0, dirichlet=onb)
    assert r["max_rel"] < 1e-13 and r["n_free"] == 15 ** 2
    ev2 = ExactEvaluator(V, T, _zero_u, _zero_gu, lambda X, Y: 0 * X, lambda X, Y: X * X,
                         lambda X, Y: np.stack([2 * X, 0 * X], axis=-1))
    assert C.weak_current_residual(ev2, sig, B0, 0.0, dirichlet=onb)["max_rel"] > 1e-3
    # with the matching source q = div J = -2 the residual vanishes again
    r3 = C.weak_current_residual(ev2, sig, B0, 0.0, q_fn=lambda X, Y: -2.0 + 0 * X, dirichlet=onb)
    assert r3["max_rel"] < 1e-13


def test_weak_residual_boundary_current_term():
    """phi = -x (J = (1, 0, 0) with sigma = 1), all nodes free, j_n = J.n on all edges: residual 0."""
    V, T = C.mms_mesh("structured", 8)
    sig = lambda X, Y: np.ones_like(X)                              # noqa: E731
    B0 = lambda X, Y: np.zeros(np.shape(X) + (3,))                  # noqa: E731
    ev = ExactEvaluator(V, T, _zero_u, _zero_gu, lambda X, Y: 0 * X, lambda X, Y: -X,
                        lambda X, Y: np.stack([-1 + 0 * X, 0 * X], axis=-1))
    edges = [((lambda P: abs(P[0]) < 1e-12), lambda X, Y: -1.0 + 0 * X, None),
             ((lambda P: abs(P[0] - 1) < 1e-12), lambda X, Y: 1.0 + 0 * X, None)]
    assert C.weak_current_residual(ev, sig, B0, 0.0, jn_edges=edges)["max_rel"] < 1e-13
    assert C.weak_current_residual(ev, sig, B0, 0.0, jn_edges=edges[:1])["max_rel"] > 1e-2


def test_wall_current_moment_direct_and_variational_agree_with_closed_form():
    """phi = sin(pi x/2) cosh(pi y/2) is harmonic with d phi/dx = 0 at x = +-1 (insulating side walls), u = 0,
    sigma = 2: M = int_{-1}^{1} J_y(x, 1) x dx = -sigma (pi/2) sinh(pi/2) 8/pi^2."""
    s = 2.0
    V, T = C.duct_mesh(10.0, 1)
    ph = lambda X, Y: np.sin(np.pi * X / 2) * np.cosh(np.pi * Y / 2)   # noqa: E731
    gph = lambda X, Y: np.stack([np.pi / 2 * np.cos(np.pi * X / 2) * np.cosh(np.pi * Y / 2),   # noqa: E731
                                 np.pi / 2 * np.sin(np.pi * X / 2) * np.sinh(np.pi * Y / 2)], axis=-1)
    ev = ExactEvaluator(V, T, _zero_u, _zero_gu, lambda X, Y: 0 * X, ph, gph)
    M = -s * (np.pi / 2) * np.sinh(np.pi / 2) * 8 / np.pi ** 2
    assert abs(C.wall_current_moment_direct(ev, s, [0, 1, 0]) - M) < 1e-12 * abs(M)
    assert abs(C.wall_current_moment_variational(ev, s, [0, 1, 0]) - M) < 1e-10 * abs(M)


def test_hartmann_exact_satisfies_its_ode_and_flow_rate():
    for Ha in (0.0, 1e-4, 1.0, 10.0, 100.0, 1000.0):
        B0 = Ha / math.sqrt(SIGMA / MU)
        for mode in ("inplane", "duct"):
            for circ in ("short", "open"):
                h = S.Hartmann(MU, SIGMA, B0, G, 1.0, mode, circ)
                y = np.linspace(-0.9, 0.9, 7)
                d = 1e-4 / max(1.0, Ha / 50)
                U2 = (h.U(y + d) - 2 * h.U(y) + h.U(y - d)) / d ** 2
                res = MU * U2 - SIGMA * B0 ** 2 * h.U(y) + h.Geff
                assert np.max(np.abs(res)) < 1e-5 * max(h.Geff, 1e-300) * (1 + Ha) ** 0.5
                assert abs(h.U(np.array(1.0))) < 1e-15 and abs(h.U(np.array(-1.0))) < 1e-15
                g, w = np.polynomial.legendre.leggauss(400)
                Q = float(np.sum(w * h.U(g)))
                if Ha <= 100:
                    assert abs(Q - h.Q) < 1e-10 * abs(h.Q)
                if circ == "open" and B0 > 0:
                    assert abs(h.net_current()) < 1e-12 * SIGMA * B0 * abs(h.Q) + 1e-15


def test_series_wide_duct_limits_and_core_current():
    """L -> infinity: Shercliff -> open-circuit Hartmann, Hunt -> short-circuit Hartmann (centre velocity); the fast
    core-current evaluators agree with the plain series."""
    for Ha in (1.0, 10.0):
        B0 = Ha / math.sqrt(SIGMA / MU)
        sh = S.Shercliff(MU, SIGMA, B0, G, L=40.0, K=2 ** 16).w_center()
        hu = S.Hunt(MU, SIGMA, B0, G, L=40.0, K=2 ** 16).w_center()
        assert abs(sh / S.Hartmann(MU, SIGMA, B0, G, 1.0, "duct", "open").U(0.0) - 1) < 1e-9
        assert abs(hu / S.Hartmann(MU, SIGMA, B0, G, 1.0, "duct", "short").U(0.0) - 1) < 1e-9
        X, Y = np.array([0.1, -0.3]), np.array([0.2, 0.4])
        h = S.Hunt(MU, SIGMA, B0, G, K=2 ** 17)
        assert np.allclose(S.hunt_J_fast(h, X, Y, 400), h.J(X, Y), rtol=0, atol=1e-11)
        s = S.Shercliff(MU, SIGMA, B0, G, K=2 ** 14)
        assert np.allclose(S.shercliff_J_fast(s, X, Y), s.J(X, Y), rtol=0, atol=1e-12)


def test_core_current_error_zero_for_exact_hunt_field():
    B0 = 100.0 / math.sqrt(SIGMA / MU)
    h = S.Hunt(MU, SIGMA, B0, G, K=4000)
    V, T = C.duct_mesh(100.0, 0)

    def u(X, Y):
        U = np.zeros(np.shape(X) + (3,))
        Jx, _ = S.hunt_J_fast(h, X, Y, 400)
        # w from Ohm's law is not needed: give the exact w through J_x = sigma(-phi_x - B0 w) with phi_x = 0 trick
        U[..., 2] = -Jx / (SIGMA * B0)
        return U

    def gphi(X, Y):
        _, Jy = S.hunt_J_fast(h, X, Y, 400)
        return np.stack([0 * X, -Jy / SIGMA], axis=-1)
    ev = ExactEvaluator(V, T, u, _zero_gu, lambda X, Y: 0 * X, lambda X, Y: 0 * X, gphi)
    r = C.core_current_error(ev, SIGMA, [0, B0, 0], lambda X, Y: S.hunt_J_fast(h, X, Y, 400))
    assert r["rel"] < 1e-12 and r["n_core_elements"] > 0


def test_reference_values_reproduce_from_the_series():
    """The stage-1 reference file (out of band) agrees with the fixed series module at K = 2^18 (spot check)."""
    ref = json.loads((ROOT / "tools/fluids_mms/f13a_reference/analytic_reference.json").read_text())
    p = ref["parameters"]
    B0 = 100.0 / math.sqrt(p["sigma"] / p["mu"])
    h = S.Hunt(p["mu"], p["sigma"], B0, p["G"], p["L"], p["a"], K=2 ** 18)
    assert abs(h.Q() - ref["hunt"]["Ha100"]["Q"]) < 1e-12 * ref["hunt"]["Ha100"]["Q"]
    s = S.Shercliff(p["mu"], p["sigma"], B0, p["G"], p["L"], p["a"], K=2 ** 18)
    assert abs(s.Q() - ref["shercliff"]["Ha100"]["Q"]) < 1e-12 * ref["shercliff"]["Ha100"]["Q"]


def test_identities_are_zero_when_the_current_vanishes():
    """Revision 2: B = 0 gives J_h = 0 identically; the identities return 0 instead of dividing by 0."""
    V, T = C.duct_mesh(0.0, 0)
    w = lambda X, Y: np.stack([0 * X, 0 * X, 0.5 * (1 - Y ** 2)], axis=-1)   # noqa: E731
    ev = ExactEvaluator(V, T, w, _zero_gu, lambda X, Y: 0 * X, lambda X, Y: 0 * X,
                        lambda X, Y: np.zeros(np.shape(X) + (2,)))
    I = C.integrals(ev, SIGMA, [0.0, 0.0, 0.0], 0.0, MU)
    assert C.lorentz_joule_defect(I, 0.0) == 0.0 and C.net_current_rel(I) == 0.0
    r = C.weak_current_residual(ev, lambda X, Y: np.full(np.shape(X), SIGMA),
                                lambda X, Y: np.zeros(np.shape(X) + (3,)), 0.0)
    assert r["max_rel"] == 0.0


def test_axial_momentum_identity_exact_for_p2_poiseuille_and_has_teeth():
    """B = 0, w = G(1 - y^2)/(2 mu) (in P2): mu Lap w + G = 0, so every interior r_i vanishes to round-off; a wrong
    G (or a missing Lorentz term when B != 0) gives O(1)."""
    V, T = C.duct_mesh(0.0, 1)
    wf = lambda X, Y: np.stack([0 * X, 0 * X, G * (1 - Y ** 2) / (2 * MU)], axis=-1)   # noqa: E731

    def gw(X, Y):
        g = np.zeros(np.shape(X) + (3, 2))
        g[..., 2, 1] = -G * Y / MU
        return g
    ev = ExactEvaluator(V, T, wf, gw, lambda X, Y: 0 * X, lambda X, Y: 0 * X,
                        lambda X, Y: np.zeros(np.shape(X) + (2,)))
    onwall = lambda P: abs(abs(P[0]) - 1) < 1e-12 or abs(abs(P[1]) - 1) < 1e-12   # noqa: E731
    assert C.weak_axial_momentum_residual(ev, MU, SIGMA, [0, 0, 0], G, dirichlet=onwall)["max_rel"] < 1e-13
    assert C.weak_axial_momentum_residual(ev, MU, SIGMA, [0, 0, 0], 2 * G, dirichlet=onwall)["max_rel"] > 1e-2
    assert C.weak_axial_momentum_residual(ev, MU, SIGMA, [0, 1.0, 0], G, dirichlet=onwall)["max_rel"] > 1e-2


def test_core_reference_is_fixed_in_the_module():
    f = C.core_J_exact("hunt", MU, SIGMA, 50.0, G)
    h = S.Hunt(MU, SIGMA, 50.0, G, K=2 ** 17)
    X, Y = np.array([0.1, -0.3]), np.array([0.2, 0.4])
    assert np.allclose(f(X, Y), h.J(X, Y), rtol=0, atol=1e-11)
    with pytest.raises(ValueError):
        C.core_J_exact("nope", MU, SIGMA, 1.0, G)
