"""Steady Navier-Stokes through the fluids Python API (axionfem.fluids.solve_navier_stokes, MPI builds):
unit tests that run in seconds on one rank (formulation: docs/guides/fluids.md).

  U1  Taylor test of R(x) against J(x)d per (geometry x form x viscous form): slope 2, r2/eps^2 constant (R is quadratic)
  U5  discrete energy identity of the kernel on random homogeneous fields: c_skew(u; u, u) = 0 to round-off and
      c_standard(u; u, u) != 0 (planar and axisymmetric)
  U6  rho -> 0: the NS solution is the Stokes solution; swirl-free data keep u_theta = 0 exactly
  U7  circular Couette (planar annulus, exact NS solution at every Re): the velocity error and the Babuska-Miller
      torque including the convective term converge (torque order ~4)
  U8  driver: history semantics (the true Stokes start equals solve_stokes), Picard-only = Newton, continuation and
      PTC reach the same state, pin = multiplier, the PTC release branch, a failed solve reported (not raised) with
      its fields refused, and the refusals of unsupported options
"""
import math

import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)
if not hasattr(fluids, "solve_navier_stokes"):
    pytest.skip("this build has no Navier-Stokes solver", allow_module_level=True)

B, R, T, L = 1, 2, 3, 4  # attributes: bottom, right, top, left (the axis in axisymmetry)


def rect(nx, ny, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    V = np.array([[x, y] for y in ys for x in xs])
    vid = lambda i, j: j * (nx + 1) + i  # noqa: E731
    tri = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            flip = (i == nx - 1 and j == 0) or (i == 0 and j == ny - 1)
            tri += [[a, b, d], [b, c, d]] if flip else [[a, b, c], [a, c, d]]
    seg, att = [], []
    for i in range(nx):
        seg += [[vid(i, 0), vid(i + 1, 0)], [vid(i + 1, ny), vid(i, ny)]]
        att += [B, T]
    for j in range(ny):
        seg += [[vid(nx, j), vid(nx, j + 1)], [vid(0, j + 1), vid(0, j)]]
        att += [R, L]
    return fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32),
                                   np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32))


def lid(x, y):
    return (16 * x * x * (1 - x) ** 2, 0.0)  # regularized lid


def cavity_bcs():
    return [{"attribute": T, "kind": "dirichlet", "value": lid}] + \
           [{"attribute": a, "kind": "dirichlet"} for a in (B, R, L)]


def ns(mesh, mu, rho=1.0, **kw):
    kw.setdefault("lu", "eigen")
    return fluids.solve_navier_stokes(mesh, mu, rho, **kw)


def fields(sol, ref=((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3))):
    ref = np.array(ref, float)
    el = np.arange(sol.n_elements, dtype=np.int32)
    u, _ = sol.velocity(el, ref)
    return np.asarray(u), np.asarray(sol.pressure(el, ref))


# ------------------------------------------------------------------------------------------------ U1 Taylor test
@pytest.mark.parametrize("geometry", ["planar", "axisymmetric"])
@pytest.mark.parametrize("conv", ["standard", "skew"])
@pytest.mark.parametrize("form", ["lap", "sym"])
def test_u1_taylor(geometry, conv, form):
    axi = geometry == "axisymmetric"
    m = rect(4, 4)
    if axi:
        g = lambda x, y: (0.0, 0.0, 4 * x * (1 - x))  # noqa: E731  swirl driven by the top wall
        bcs = [{"attribute": T, "kind": "dirichlet", "value": g}, {"attribute": L, "kind": "axis"}] + \
              [{"attribute": a, "kind": "dirichlet"} for a in (B, R)]
    else:
        bcs = cavity_bcs()
    sol = ns(m, 0.05, 1.3, bcs=bcs, form={"lap": "laplacian", "sym": "stress"}[form], convection=conv,
             geometry=geometry)
    assert sol.diagnostics["converged"]
    rng = np.random.default_rng(7)
    x = np.asarray(sol.state()) + 0.1 * rng.standard_normal(len(sol.state()))
    d = rng.standard_normal(x.size)
    R0 = np.asarray(sol.residual(x))
    Jd = np.asarray(sol.jacobian_action(x, d))
    eps = 10.0 ** (-np.arange(0, 6) / 2)
    r2 = np.array([np.linalg.norm(np.asarray(sol.residual(x + e * d)) - R0 - e * Jd) for e in eps])
    q = r2 / eps ** 2
    assert (q.max() - q.min()) / q.mean() < 1e-6
    slope = np.polyfit(np.log(eps), np.log(r2), 1)[0]
    assert abs(slope - 2) < 0.02
    # the assembled Jacobian is the same operator as the action
    J = sol.jacobian(x)
    import scipy.sparse as sp
    Jm = sp.csr_matrix((J["data"], J["indices"], J["indptr"]), shape=J["shape"])
    assert np.linalg.norm(Jm @ d - Jd) <= 1e-12 * np.linalg.norm(Jd)


# ------------------------------------------------------------------------------------------------ U5 energy identity
@pytest.mark.parametrize("geometry", ["planar", "axisymmetric"])
def test_u5_kernel_energy(geometry):
    """With p = 0, no force and viscosity scale 0, R_u(x) = N(u) on the free rows: u . N(u) = c(u; u, u) for u = 0 on
    the Dirichlet boundary (and the axis), so skew gives 0 to round-off and standard does not."""
    axi = geometry == "axisymmetric"
    m = rect(4, 4)
    bcs = [{"attribute": a, "kind": "dirichlet"} for a in (B, R, T)] + \
          [{"attribute": L, "kind": "axis" if axi else "dirichlet"}]
    vals = {}
    for conv in ("standard", "skew"):
        sol = ns(m, 1.0, 1.0, bcs=bcs, convection=conv, geometry=geometry)  # zero data: converged at the lift
        nfu, npr = sol.layout()
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.standard_normal(nfu), np.zeros(npr)])
        Nu = np.asarray(sol.residual(x, 0.0))[:nfu]
        vals[conv] = (float(x[:nfu] @ Nu), float(np.linalg.norm(Nu) * np.linalg.norm(x[:nfu])))
    assert abs(vals["skew"][0]) <= 1e-13 * vals["skew"][1]
    assert abs(vals["standard"][0]) >= 1e-3 * vals["standard"][1]


# ------------------------------------------------------------------------------------------------ U6 small Re
def test_u6_stokes_limit_and_swirl_invariance():
    m = rect(6, 6)
    st = fluids.solve_stokes(m, 0.1, None, cavity_bcs(), lu="eigen")
    sol = ns(m, 0.1, 1e-8, bcs=cavity_bcs())
    assert sol.diagnostics["converged"]
    us, ps = fields(st)
    un, pn = fields(sol)
    assert np.max(np.abs(un - us)) <= 1e-7 * np.max(np.abs(us))
    # axisymmetric, meridional data only: u_theta stays identically zero
    g = lambda r, z: (0.0, 4 * r * (1 - r) * 0 + 0.0, 0.0)  # noqa: E731
    f = lambda r, z: (math.sin(3 * z), r * (1 - r), 0.0)  # noqa: E731
    bcs = [{"attribute": L, "kind": "axis"}] + [{"attribute": a, "kind": "dirichlet", "value": g} for a in (B, R, T)]
    sa = ns(m, 0.05, 1.0, force=f, bcs=bcs, geometry="axisymmetric")
    assert sa.diagnostics["converged"]
    ua, _ = fields(sa)
    assert np.max(np.abs(ua[..., 2])) == 0.0
    assert np.max(np.abs(ua[..., :2])) > 1e-3


# ------------------------------------------------------------------------------------------------ U7 Couette
def couette_solution(r1, r2, om1, rho):
    A = -om1 * r1 ** 2 / (r2 ** 2 - r1 ** 2)
    Bc = om1 * r1 ** 2 * r2 ** 2 / (r2 ** 2 - r1 ** 2)
    ut = lambda r: A * r + Bc / r  # noqa: E731
    p = lambda r: rho * (A * A * r * r / 2 + 2 * A * Bc * np.log(r) - Bc * Bc / (2 * r * r))  # noqa: E731
    return A, Bc, ut, p


def test_u7_couette_annulus_torque():
    r1, r2, om1, rho, mu = 0.5, 1.0, 2.0, 1.25, 0.05
    A, Bc, ut, p = couette_solution(r1, r2, om1, rho)
    tq_exact = -4 * math.pi * mu * Bc  # torque exerted by the fluid on the inner cylinder (per unit depth)
    errs, tqe = [], []
    for k in (1, 2, 3):
        m = fluids.Mesh.annulus(4 * 2 ** k, 16 * 2 ** k, r1, r2, 2)
        wall = lambda x, y: (-om1 * y, om1 * x)  # noqa: E731
        bcs = [{"attribute": 1, "kind": "dirichlet", "value": wall}, {"attribute": 2, "kind": "dirichlet"}]
        sol = ns(m, mu, rho, bcs=bcs)
        assert sol.diagnostics["converged"]
        ref = np.array([[1 / 3, 1 / 3]])
        el = np.arange(sol.n_elements, dtype=np.int32)
        x, _ = sol.geometry(el, ref)
        u, _ = sol.velocity(el, ref)
        rr = np.hypot(x[..., 0], x[..., 1])
        ue = np.stack([-ut(rr) * x[..., 1] / rr, ut(rr) * x[..., 0] / rr], -1)
        errs.append(float(np.max(np.abs(u - ue))))
        tqe.append(abs(sol.reaction([1], [0.0, 0.0])["torque"] - tq_exact))
    assert errs[-1] < errs[0] / 20
    order = math.log(tqe[-2] / tqe[-1], 2)
    assert order > 3.3, (tqe, order)


# ------------------------------------------------------------------------------------------------ U8 driver
def test_u8_stokes_start_equals_stokes_and_history():
    m = rect(6, 6)
    st = fluids.solve_stokes(m, 0.02, None, cavity_bcs(), lu="eigen")
    sol = ns(m, 0.02, 1.0, bcs=cavity_bcs(), strategy="newton", n_picard=0, newton_maxit=0)  # only the Stokes step
    h = sol.history
    assert h[0]["kind"] == "stokes" and len(h) == 1 and not sol.diagnostics["converged"]
    with pytest.raises(Exception):
        sol.velocity(np.arange(1, dtype=np.int32), np.array([[0.2, 0.2]]))
    full = ns(m, 0.02, 1.0, bcs=cavity_bcs())
    hk = [e["kind"] for e in full.history]
    assert hk[0] == "stokes" and hk[1:4] == ["picard"] * 3 and set(hk[4:]) <= {"newton"}
    assert full.diagnostics["iterations"] == len(hk) - 1
    assert full.history[-1]["residual"] == full.diagnostics["relative_residual"] <= 1e-12
    # the Stokes step of the NS driver is the Stokes solution: rho -> 0 with zero Newton steps equals solve_stokes
    s0 = ns(m, 0.02, 1e-300, bcs=cavity_bcs(), n_picard=0)
    us, _ = fields(st)
    u0, _ = fields(s0)
    assert np.max(np.abs(u0 - us)) <= 1e-12 * np.max(np.abs(us))


def test_u8_paths_agree():
    m = rect(8, 8)
    mu = 1.0 / 200
    ref = ns(m, mu, bcs=cavity_bcs())
    pic = ns(m, mu, bcs=cavity_bcs(), strategy="picard")
    cont = ns(m, mu, bcs=cavity_bcs(), continuation_re=[50, 100, 200], re=200)
    ptc = ns(m, mu, bcs=cavity_bcs(), strategy="ptc")
    rel = ns(m, mu, bcs=cavity_bcs(), strategy="ptc", ptc_release_dt=1.0)  # the release branch
    pin = ns(m, mu, bcs=cavity_bcs(), nullspace="pin")
    # U8 "pin = multiplier": the default backend's auto nullspace is the pin, so compare the two treatments
    # explicitly on the STRUMPACK + multiplier path, when that backend is built
    u0, p0 = fields(ref)
    if fluids.lu_backend_available("strumpack"):
        mult = ns(m, mu, bcs=cavity_bcs(), lu="strumpack", nullspace="multiplier")
        pinsp = ns(m, mu, bcs=cavity_bcs(), lu="strumpack", nullspace="pin")
        for s in (mult, pinsp):
            assert s.diagnostics["converged"], s.diagnostics["failure"]
        um, pm = fields(mult)
        up, pp = fields(pinsp)
        assert np.max(np.abs(um - up)) <= 1e-9 and np.max(np.abs(pm - pp)) <= 1e-8
        assert np.max(np.abs(um - u0)) <= 1e-9 and np.max(np.abs(pm - p0)) <= 1e-8
        assert len(mult.history) == len(pinsp.history)
    # the multiplier treatment is also available with Eigen (all four backend x treatment combinations agree)
    mult_eigen = ns(m, mu, bcs=cavity_bcs(), nullspace="multiplier")
    assert mult_eigen.diagnostics["converged"]
    um, pm = fields(mult_eigen)
    assert np.max(np.abs(um - fields(pin)[0])) <= 1e-9 and np.max(np.abs(pm - fields(pin)[1])) <= 1e-8
    for s in (pic, cont, ptc, rel, pin):
        assert s.diagnostics["converged"], s.diagnostics["failure"]
        u, p = fields(s)
        assert np.max(np.abs(u - u0)) <= 1e-9
        assert np.max(np.abs(p - p0)) <= 1e-8
    assert {e["kind"] for e in pic.history[1:]} == {"picard"}
    assert "ptc" in {e["kind"] for e in rel.history} and rel.history[-1]["kind"] == "newton"
    assert sorted({e["stage"] for e in cont.history}) == [0, 1, 2]


def test_u8_refusals():
    """Refusals at the Python boundary (grad-div, MINRES, a pressure point and swirl-free axisymmetry have no
    keyword here; the C++ refusals are covered by cpp/tests/test_fluids_navier_stokes.cpp)."""
    m = rect(3, 3)
    for kind in ("slip", "symmetry"):
        with pytest.raises(Exception, match="not offered with convection"):
            ns(m, 1.0, bcs=[{"attribute": T, "kind": kind}] + [{"attribute": a, "kind": "dirichlet"} for a in (B, R, L)])
    # the periodic and Ortho refusals
    per = [{"attribute": L, "kind": "periodic", "partner": R, "translation": (1.0, 0.0)}] + \
          [{"attribute": a, "kind": "dirichlet"} for a in (B, T)]
    with pytest.raises(Exception, match="not offered"):
        ns(m, 1.0, bcs=per)
    with pytest.raises(Exception, match="Ortho"):
        ns(m, 1.0, bcs=cavity_bcs(), nullspace="ortho")
    # a viscosity schedule must end at 1; re= needs continuation_re
    with pytest.raises(ValueError, match="end at 1"):
        ns(m, 1.0, bcs=cavity_bcs(), viscosity_scales=[2.0, 0.5])
    with pytest.raises(ValueError, match="continuation_re"):
        ns(m, 1.0, bcs=cavity_bcs(), re=100)


def test_u8_failed_solve_is_reported():
    m = rect(8, 8)
    sol = ns(m, 1.0 / 5000, bcs=cavity_bcs(), n_picard=0, line_search=False, newton_maxit=3)
    d = sol.diagnostics
    assert not d["converged"] and d["failure"]
    assert len(sol.history) >= 2
