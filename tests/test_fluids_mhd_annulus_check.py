"""Fast self-tests of the MHD variable-field annulus check module (benchmarks/fluids/cases/f13a_annulus_check.py;
case analytic_mhd_annulus_variable_b).  No solver: the exact fields on the fixed curved family,
through an Evaluator that uses the fixed isoparametric map.  Also the suction-duct check module."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import f13a_annulus_check as AC  # noqa: E402
import f13a_mhd_check as M  # noqa: E402

pytestmark = pytest.mark.fast


class IsoEvaluator:
    """The exact solution, sampled through the fixed isoparametric geometry (or a perturbation of it)."""

    def __init__(self, X6, ex, du=None, dphi=None):
        self.X6, self.ex, self.n_elements = X6, ex, len(X6)
        self.du, self.dphi = du, dphi

    def geometry(self, ref):
        x, _, det = AC.iso_map(self.X6, np.asarray(ref, float))
        return x, np.abs(det)

    def _xy(self, ref):
        x, _ = self.geometry(ref)
        return x[..., 0], x[..., 1]

    def velocity(self, ref):
        X, Y = self._xy(ref)
        u, g = self.ex.u(X, Y), self.ex.grad_u(X, Y)
        if self.du is not None:
            u = u + self.du(X, Y)
        return u, g

    def pressure(self, ref):
        X, Y = self._xy(ref)
        return self.ex.p(X, Y)

    def potential(self, ref):
        X, Y = self._xy(ref)
        ph, g = self.ex.phi(X, Y), self.ex.grad_phi(X, Y)
        if self.dphi is not None:
            ph = ph + self.dphi(X, Y)
        return ph, g


def test_family_nested_and_curved():
    for v in ("radial_ins_m20", "azimuthal_cond"):
        prev = None
        for k in range(3):
            V, T, X6 = AC.annulus_mesh(v, k)
            assert len(T) == 2 * AC.NR0 * AC.NT0 * 4 ** k
            r = np.hypot(V[:, 0], V[:, 1])
            assert abs(r.min() - 1) < 1e-14 and abs(r.max() - 2) < 1e-14
            if prev is not None:     # every coarse vertex is a fine vertex (nested in (r, theta))
                fine = {tuple(np.round(p, 10)) for p in V}
                assert all(tuple(np.round(p, 10)) in fine for p in prev)
            prev = V
            # boundary-edge midpoints on the circles, interior midpoints straight
            rm = np.hypot(X6[:, 3:, 0], X6[:, 3:, 1])
            on = np.isclose(rm, 1.0, atol=1e-13) | np.isclose(rm, 2.0, atol=1e-13)
            assert on.sum() == 2 * AC.NT0 * 2 ** k
    # the radial m = 20 base is graded to the layer thickness 1/m at the inner wall
    b = AC.radial_base_nodes("radial_ins_m20")
    assert abs(b[1] - b[0] - 1 / 20) < 1e-9 and abs(b[-1] - b[-2] - 1 / 20) < 1e-9
    assert np.allclose(np.diff(AC.radial_base_nodes("azimuthal_cond")), 1 / AC.NR0)


def test_mesh_record_stable():
    r = AC.mesh_record("radial_ins_m5", 1)
    assert r["n_elements"] == 2 * 16 * 64 and len(r["mesh_hash"]) == 64 and len(r["p2_nodes_hash"]) == 64


def test_iso_map_area_converges_at_order_4():
    errs = []
    for k in range(3):
        _, _, X6 = AC.annulus_mesh("azimuthal_cond", k)
        ev = IsoEvaluator(X6, AC.exact("azimuthal_cond"))
        errs.append(AC.geometry_area_error(ev))
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(3.7 < o < 4.3 for o in orders), orders


@pytest.mark.parametrize("v", list(AC.VARIANTS))
def test_exact_fields_satisfy_the_pde_and_walls(v):
    """FD residuals of the inductionless MHD equations at interior points; no-slip walls, potential walls, B
    div/curl-free."""
    ex = AC.exact(v)
    var = AC.VARIANTS[v]
    h = 1e-4
    rng = np.random.default_rng(3)
    rr = rng.uniform(1.1, 1.9, 6)
    tt = rng.uniform(0, 2 * np.pi, 6)
    X, Y = rr * np.cos(tt), rr * np.sin(tt)

    def lap(f, X, Y):
        return (f(X + h, Y) + f(X - h, Y) + f(X, Y + h) + f(X, Y - h) - 4 * f(X, Y)) / h ** 2

    def dx(f, X, Y, ax):
        return (f(X + h, Y) - f(X - h, Y)) / (2 * h) if ax == 0 else (f(X, Y + h) - f(X, Y - h)) / (2 * h)

    J = ex.J(X, Y)
    F = M.cross(J, ex.B(X, Y))
    gp = np.stack([dx(ex.p, X, Y, 0), dx(ex.p, X, Y, 1), 0 * X], -1)
    for i in range(3):
        L = lap(lambda a, b: ex.u(a, b)[..., i], X, Y)
        res = -ex.mu * L + gp[..., i] - (ex.G if i == 2 else 0) - F[..., i]
        assert np.max(np.abs(res)) < 1e-5 * max(1.0, np.max(np.abs(F))), (v, i, res)
    charge = dx(lambda a, b: ex.J(a, b)[..., 0], X, Y, 0) + dx(lambda a, b: ex.J(a, b)[..., 1], X, Y, 1)
    assert np.max(np.abs(charge)) < 1e-6 * max(1.0, np.max(np.abs(J)))
    divB = dx(lambda a, b: ex.B(a, b)[..., 0], X, Y, 0) + dx(lambda a, b: ex.B(a, b)[..., 1], X, Y, 1)
    curlB = dx(lambda a, b: ex.B(a, b)[..., 1], X, Y, 0) - dx(lambda a, b: ex.B(a, b)[..., 0], X, Y, 1)
    assert np.max(np.abs(divB)) < 1e-6 and np.max(np.abs(curlB)) < 1e-6
    # B is NOT uniform (the point of the case): |B| varies by the factor r2/r1 = 2
    Bm = np.linalg.norm(ex.B(np.array([1.0, 2.0]), np.array([0.0, 0.0])), axis=-1)
    assert abs(Bm[0] / Bm[1] - 2.0) < 1e-12
    # walls
    for rw in (1.0, 2.0):
        th = np.linspace(0, 2 * np.pi, 7)
        u = ex.u(rw * np.cos(th), rw * np.sin(th))
        u[..., 2] -= var.get("W", 0.0) if rw == 1.0 else 0.0          # azimuthal_slide_ins: w(1) = W
        assert np.max(np.abs(u)) < 1e-12 * max(1.0, float(np.max(np.abs(ex.w_r(np.linspace(1, 2, 11))))))
        Jw = ex.J(rw * np.cos(th), rw * np.sin(th))
        jn = Jw[:, 0] * np.cos(th) + Jw[:, 1] * np.sin(th)
        if var["field"] == "radial" or var["potential"] == "insulating":
            assert np.max(np.abs(jn)) < 1e-13          # insulating (and the radial grounded) walls: J.n = 0
        else:
            assert np.allclose(jn, ex.K / rw, rtol=1e-13)
    if var["field"] == "azimuthal" and var["potential"] == "dirichlet":
        assert abs(ex.phi_r(np.array(1.0))) < 1e-14 and abs(ex.phi_r(np.array(2.0)) - var["V"]) < 1e-13


@pytest.mark.parametrize("v", list(AC.VARIANTS))
def test_functionals_of_the_exact_field(v):
    """Discrete functionals of the exact field on the finest affordable level converge to the closed forms."""
    ex = AC.exact(v)
    _, _, X6 = AC.annulus_mesh(v, 2)
    ev = IsoEvaluator(X6, ex)
    assert abs(AC.flow_rate(ev) - ex.Q) < 1e-6 * abs(ex.Q)
    if AC.VARIANTS[v]["field"] == "radial":
        assert abs(AC.angular_momentum(ev) - ex.L_theta) < 1e-6 * abs(ex.L_theta)
        assert AC.potential_spread(ev)["phi_spread"] == 0.0
    elif AC.VARIANTS[v]["potential"] == "dirichlet":
        assert abs(AC.wall_current_variational(ev, ex) - ex.I_wall) < 1e-6 * abs(ex.I_wall)
        assert abs(AC.wall_current_direct(ev, ex, X6) - ex.I_wall) < 1e-9 * abs(ex.I_wall)
    nr = M.norms(ev, ex, pressure="mean_free", phi="determined")
    for q in ("uL2", "uH1", "pL2", "phiL2", "phiH1", "JL2", "divL2"):
        assert nr[q] < 1e-12 * max(1.0, nr["u_magnitude"]), (q, nr[q])


def test_norms_and_functionals_see_perturbations():
    """Teeth: a velocity perturbation shows in uL2 and Q; a potential perturbation in phiL2 and the wall current."""
    ex = AC.exact("azimuthal_cond")
    _, _, X6 = AC.annulus_mesh("azimuthal_cond", 1)
    bump = lambda X, Y: 1e-3 * (np.hypot(X, Y) - 1) * (2 - np.hypot(X, Y))  # noqa: E731
    ev = IsoEvaluator(X6, ex, du=lambda X, Y: np.stack([0 * X, 0 * X, bump(X, Y)], -1), dphi=bump)
    nr = M.norms(ev, ex, pressure="mean_free", phi="determined")
    assert nr["uL2"] > 1e-4 and nr["phiL2"] > 1e-4
    assert abs(AC.flow_rate(ev) - ex.Q) > 1e-4


def test_rejects_resonant_exponent():
    with pytest.raises(ValueError):
        AC.RadialExact(1.0, 0.5)          # m = 2


# ============================================================================================== suction duct
import f13a_suction_check as SC  # noqa: E402


@pytest.mark.parametrize("v", list(SC.VARIANTS))
def test_suction_exact_field(v):
    """FD residuals of the inductionless MHD equations WITH convection for the suction closed form; walls; Q
    against quadrature."""
    ex = SC.exact(v)
    h = 1e-4
    Y = np.linspace(-0.95, 0.95, 9)
    X = 0.1 + 0 * Y
    u = ex.u(X, Y)
    J = ex.J(X, Y)
    F = M.cross(J, ex.B(X, Y))
    gp = np.stack([(ex.p(X + h, Y) - ex.p(X - h, Y)) / (2 * h), 0 * Y, 0 * Y], -1)
    for i in range(3):
        f = lambda a, b: ex.u(a, b)[..., i]  # noqa: E731
        lap = (f(X + h, Y) + f(X - h, Y) + f(X, Y + h) + f(X, Y - h) - 4 * f(X, Y)) / h ** 2
        conv = ex.rho * (u[..., 0] * (f(X + h, Y) - f(X - h, Y)) + u[..., 1] * (f(X, Y + h) - f(X, Y - h))) / (2 * h)
        Gi = ex.G if (i == 2 and ex.mode == "duct") else 0.0
        res = conv - ex.mu * lap + gp[..., i] - Gi - F[..., i]
        assert np.max(np.abs(res)) < 1e-5, (v, i, res)
    assert np.allclose(ex.u(np.zeros(2), np.array([-1.0, 1.0])), [[0, ex.V, 0]] * 2, atol=1e-14) or \
        np.allclose(ex.u(np.zeros(2), np.array([-1.0, 1.0]))[:, [0, 2]], 0, atol=1e-14)
    yy = np.linspace(-1, 1, 20001)
    assert abs(np.trapezoid(ex.F(yy), yy) - ex.Q) < 1e-7 * ex.Q
    # convection matters: the profile is asymmetric (peak shifted towards the suction wall)
    assert yy[np.argmax(ex.F(yy))] > 0.1


def test_suction_flow_rate_functional():
    v = "duct_Ha10_V4"
    ex = SC.exact(v)
    V, T = SC.mesh(v, 2)
    P = V[T]
    X6 = np.concatenate([P, 0.5 * (P[:, [0, 1, 2]] + P[:, [1, 2, 0]])], axis=1)
    ev = IsoEvaluator(X6, ex)
    assert abs(SC.flow_rate(ev, "duct") - ex.Q) < 1e-7 * ex.Q
    nr = M.norms(ev, ex, pressure="mean_free", phi="determined")
    assert nr["uL2"] < 1e-14 and nr["JL2"] < 1e-12
