"""Transient Navier-Stokes through the fluids Python API (axionfem.fluids.TransientNavierStokes, MPI builds):
the unit tests (seconds, one rank).

  P1  polynomial-in-time exactness: u = a(t) y (1 - y) e_x with a quadratic in t is reproduced to round-off by BDF2
      (Newton and semi-implicit) and NOT by BDF1; the load and the Dirichlet data are taken at t_n
  P2  initial projection: the projected state is discretely divergence-free (the interpolant of a non-divergence-free
      field is not) and carries the nodal Dirichlet data
  P3  the adaptive integration reaches T, keeps the step-ratio cap and rejects steps from a large first step
  P4  refusals: BDF order 3, slip boundaries, a missing history
"""
import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)
if not hasattr(fluids, "TransientNavierStokes"):
    pytest.skip("this build has no transient Navier-Stokes solver", allow_module_level=True)

B, R, T, L = 1, 2, 3, 4
RHO, MU = 1.25, 0.05


def rect(nx, ny):
    xs, ys = np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1)
    V = np.array([[x, y] for y in ys for x in xs])
    vid = lambda i, j: j * (nx + 1) + i  # noqa: E731
    tri = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tri += [[a, b, c], [a, c, d]]
    seg, att = [], []
    for i in range(nx):
        seg += [[vid(i, 0), vid(i + 1, 0)], [vid(i + 1, ny), vid(i, ny)]]
        att += [B, T]
    for j in range(ny):
        seg += [[vid(nx, j), vid(nx, j + 1)], [vid(0, j + 1), vid(0, j)]]
        att += [R, L]
    return fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32),
                                   np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32))


def a_of(t):
    return 1.0 + t + t * t


def exact(X, Y, t):
    return np.stack([a_of(t) * Y * (1 - Y), 0 * X], axis=-1)


def force(X, Y, t):
    # rho a' w + 2 mu a e_x  (w = y(1-y) e_x: (w.grad)w = 0, Laplacian w = -2 e_x, p = 0); same for both viscous forms
    return np.stack([RHO * (1 + 2 * t) * Y * (1 - Y) + 2 * MU * a_of(t) + 0 * X, 0 * X], axis=-1)


def make(order, scheme, form="laplacian", mesh=None, **kw):
    mesh = mesh or rect(4, 4)
    bcs = [{"attribute": a, "kind": "dirichlet", "value": exact} for a in (B, R, T, L)]
    return fluids.TransientNavierStokes(mesh, MU, RHO, force, bcs, form=form, scheme=scheme, max_order=order,
                                        lu="eigen", vectorized=True, **kw), mesh


def err_at(ts, t):
    snap = ts.snapshot()
    el = np.arange(snap.n_elements, dtype=np.int32)
    ref = np.array([[0.2, 0.3], [0.6, 0.1], [0.1, 0.8]])
    x, _ = snap.geometry(el, ref)
    u, _ = snap.velocity(el, ref)
    return float(np.max(np.abs(np.asarray(u) - exact(np.asarray(x)[..., 0], np.asarray(x)[..., 1], t))))


@pytest.mark.parametrize("scheme", ["newton", "semi"])
@pytest.mark.parametrize("form", ["laplacian", "stress"])
def test_bdf2_exact_for_quadratic_time_bdf1_not(scheme, form):
    out = {}
    for order in (1, 2):
        ts, _ = make(order, scheme, form)
        ts.set_history([0.05, 0.0], [lambda X, Y: exact(X, Y, 0.05), lambda X, Y: exact(X, Y, 0.0)])
        t = 0.05
        for _ in range(10):
            r = ts.advance(0.05)
            assert r["converged"] and r["relative_residual"] < 1e-11
            assert r["order"] == order
            t = r["t"]
        out[order] = err_at(ts, t)
    assert out[2] < 1e-9 * a_of(t)           # BDF2 is exact for a quadratic in t (exact starting levels)
    assert out[1] > 1e-4                     # BDF1 is first order


def test_bdf2_exact_with_exact_second_level_variable_steps():
    # variable steps: BDF2 with the variable-step coefficients stays exact once the first (BDF1) step error is gone
    ts, _ = make(2, "newton")
    ts.set_history([0.2, 0.1], [lambda X, Y: exact(X, Y, 0.2), lambda X, Y: exact(X, Y, 0.1)])
    t = 0.2
    for dt in (0.05, 0.08, 0.03, 0.06):
        r = ts.advance(dt)
        assert r["converged"] and r["order"] == 2
        t = r["t"]
    assert err_at(ts, t) < 1e-9


def test_initial_projection_divergence_free():
    mesh = rect(8, 8)

    def u0(X, Y):  # not divergence free, zero on the boundary
        s = X * (1 - X) * Y * (1 - Y)
        return np.stack([s * (1 + X), s * Y], axis=-1)
    bcs = [{"attribute": a, "kind": "dirichlet"} for a in (B, R, T, L)]
    divs = {}
    for proj in (True, False):
        ts = fluids.TransientNavierStokes(mesh, MU, RHO, None, bcs, scheme="newton", lu="eigen", vectorized=True)
        assert ts.initialize(u0, 0.0, proj)
        snap = ts.snapshot()
        el = np.arange(snap.n_elements, dtype=np.int32)
        q = np.array([[1 / 3, 1 / 3]])
        _, g = snap.velocity(el, q)
        g = np.asarray(g)
        divs[proj] = float(np.max(np.abs(g[..., 0, 0] + g[..., 1, 1])))
        P, U = snap.boundary_nodes([B, R, T, L])
        assert np.max(np.abs(np.asarray(U))) == 0.0     # nodal Dirichlet data (zero)
    assert divs[False] > 1e-3
    assert divs[True] < divs[False]


def test_adaptive_reaches_T_with_ratio_cap_and_rejections():
    ts, _ = make(2, "newton")
    ts.initialize(lambda X, Y: exact(X, Y, 0.0), 0.0)
    res = ts.adapt(0.5, 1e-4, 1.0, 0.2)
    assert res["reached"] and abs(res["times"][-1] - 0.5) < 1e-12
    assert res["rejected"] >= 1
    assert res["dt_max_ratio"] <= 2.2 + 1e-12
    assert ts.stats["all_converged"] and ts.stats["max_relative_residual"] < 1e-11


def test_refusals():
    mesh = rect(2, 2)
    with pytest.raises(ValueError):
        fluids.TransientNavierStokes(mesh, MU, RHO, None, [], max_order=3)
    with pytest.raises(Exception):
        fluids.TransientNavierStokes(mesh, MU, RHO, None, [{"attribute": a, "kind": "slip"} for a in (B, R, T, L)],
                                     lu="eigen")
    ts, _ = make(2, "newton")
    with pytest.raises(Exception):
        ts.advance(0.1)       # not initialized
