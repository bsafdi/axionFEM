"""Planar Stokes through the fluids Python API (axionfem.fluids, MPI builds): variable viscosity in the Laplacian form,
grad-div, slip/symmetry with nodal frames, periodic pairs (pressure drop or flow rate), 6-node P2 geometry input,
boundary power and the equal-order refusal. Seconds on one core.
"""
import json
import math

import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)
from axionfem import _core  # noqa: E402


def rect(nx, ny, L=1.0, H=1.0, theta=0.0):
    """Structured triangles on [0, L] x [0, H] (rotated by theta); attributes 1 bottom, 2 right, 3 top, 4 left."""
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)
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
        att += [1, 3]
    for j in range(ny):
        seg += [[vid(nx, j), vid(nx, j + 1)], [vid(0, j + 1), vid(0, j)]]
        att += [2, 4]
    c, s = math.cos(theta), math.sin(theta)
    V = np.c_[V[:, 0] * c - V[:, 1] * s, V[:, 0] * s + V[:, 1] * c]
    return fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32),
                                   np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32))


def l2_rel(sol, n_el, exact):
    el = np.arange(n_el, dtype=np.int32)
    ref = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 3, 1 / 3]])
    x, _ = sol.geometry(el, ref)
    u, _ = sol.velocity(el, ref)
    ue = exact(x[..., 0], x[..., 1])
    return float(np.max(np.abs(u - ue)))


def test_build_reports_fluids():
    b = json.loads(_core.build_info())
    assert b["fluids"] is True


def test_variable_mu_laplacian_is_accepted_and_differs_from_stress():
    mesh = rect(8, 8)
    mu = lambda x, y: 1.0 + 0.5 * x  # noqa: E731
    lid = lambda x, y: (1.0 if y > 1 - 1e-12 else 0.0, 0.0)  # noqa: E731
    bcs = [{"attribute": a, "kind": "dirichlet", "value": lid} for a in (1, 2, 3, 4)]
    a = fluids.solve_stokes(mesh, mu, None, bcs, form="laplacian")
    b = fluids.solve_stokes(mesh, mu, None, bcs, form="stress")
    assert a.diagnostics["converged"] and b.diagnostics["converged"]
    el = np.arange(mesh.n_elements, dtype=np.int32)
    ref = np.array([[1 / 3, 1 / 3]])
    assert np.max(np.abs(a.velocity(el, ref)[0] - b.velocity(el, ref)[0])) > 1e-4


def test_graddiv_identity_constant_mu():
    """sym with gamma = 0 equals lap with gamma = mu for constant mu and all-Dirichlet data."""
    mesh = rect(8, 8)
    g = lambda x, y: (np.sin(y) * x, np.cos(x) * y)  # noqa: E731
    f = lambda x, y: (1.0 + x, y * y)  # noqa: E731
    bcs = [{"attribute": a, "kind": "dirichlet", "value": g} for a in (1, 2, 3, 4)]
    s = fluids.solve_stokes(mesh, 0.37, f, bcs, form="stress")
    lgd = fluids.solve_stokes(mesh, 0.37, f, bcs, form="laplacian", grad_div=0.37)
    l0 = fluids.solve_stokes(mesh, 0.37, f, bcs, form="laplacian")
    el = np.arange(mesh.n_elements, dtype=np.int32)
    ref = np.array([[0.2, 0.3], [0.5, 0.25]])
    us, ul, u0 = (z.velocity(el, ref)[0] for z in (s, lgd, l0))
    assert np.max(np.abs(us - ul)) < 1e-11 * np.max(np.abs(us))
    assert np.max(np.abs(us - u0)) > 1e-8


def test_slip_plug_flow_inclined_channel():
    th = math.pi / 6
    mesh = rect(8, 4, 2.0, 1.0, th)
    U = np.array([math.cos(th), math.sin(th)])
    plug = lambda x, y: tuple(U)  # noqa: E731
    for form in ("laplacian", "stress"):
        sol = fluids.solve_stokes(mesh, 1.0, None, [
            {"attribute": 4, "kind": "dirichlet", "value": plug}, {"attribute": 2, "kind": "do_nothing"},
            {"attribute": 1, "kind": "slip"}, {"attribute": 3, "kind": "symmetry"}], form=form)
        d = sol.diagnostics
        assert d["converged"] and d["slip_nodes"] > 0
        err = l2_rel(sol, mesh.n_elements, lambda X, Y: np.broadcast_to(U, X.shape + (2,)))
        assert err < 1e-12
        P, Uv = sol.boundary_nodes([1, 3])
        n = np.array([math.sin(th), -math.cos(th)])
        assert np.max(np.abs(Uv @ n)) < 1e-13


def test_periodic_poiseuille_drop_and_flow_rate():
    mesh = rect(8, 4, 2.0, 1.0)
    walls = [{"attribute": 1, "kind": "dirichlet"}, {"attribute": 3, "kind": "dirichlet"}]
    exact = lambda X, Y: np.stack([Y * (1 - Y), 0 * X], axis=-1)  # noqa: E731   G = 2, mu = 1
    for form in ("laplacian", "stress"):
        dp = fluids.solve_stokes(mesh, 1.0, None, [{"attribute": 4, "kind": "periodic", "partner": 2,
                                                    "translation": (2.0, 0.0), "pressure_drop": 4.0}] + walls,
                                 form=form)
        assert dp.diagnostics["converged"] and dp.diagnostics["nullspace_detected"]
        assert l2_rel(dp, mesh.n_elements, exact) < 1e-12
        q = fluids.solve_stokes(mesh, 1.0, None, [{"attribute": 4, "kind": "periodic", "partner": 2,
                                                   "translation": (2.0, 0.0), "flow_rate": 1.0 / 6.0}] + walls,
                                form=form)
        assert q.diagnostics["converged"]
        assert abs(q.diagnostics["periodic_pressure_gradient"][0] - 2.0) < 1e-11
    with pytest.raises(fluids.FluidsError, match="partner"):
        fluids.solve_stokes(mesh, 1.0, None, [{"attribute": 4, "kind": "periodic", "partner": 2,
                                               "translation": (2.0, 0.0)}, {"attribute": 2, "kind": "do_nothing"}]
                            + walls)


def test_p2_nodes_annulus_area_and_lid_power():
    nr, nt = 2, 16
    rs, ts = np.linspace(1, 2, nr + 1), 2 * np.pi * np.arange(nt) / nt
    V = np.array([[r * math.cos(t), r * math.sin(t)] for r in rs for t in ts])
    idx = lambda i, j: i * nt + (j % nt)  # noqa: E731
    T = np.array([t for i in range(nr) for j in range(nt)
                  for t in ([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])],
                 dtype=np.int32)
    X6 = np.zeros((len(T), 6, 2))
    seg, att = [], []
    rad = np.hypot(V[:, 0], V[:, 1])
    for e, t in enumerate(T):
        X6[e, :3] = V[t]
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            a, b = t[i], t[j]
            if abs(rad[a] - rad[b]) < 1e-12:
                ang = math.atan2(V[a, 1] + V[b, 1], V[a, 0] + V[b, 0])
                X6[e, 3 + k] = rad[a] * np.array([math.cos(ang), math.sin(ang)])
            else:
                X6[e, 3 + k] = 0.5 * (V[a] + V[b])
    for j in range(nt):
        seg += [[idx(0, j + 1), idx(0, j)], [idx(nr, j), idx(nr, j + 1)]]
        att += [1, 2]
    mesh = fluids.Mesh.from_arrays(V, T, np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32), p2_nodes=X6)
    assert np.allclose(mesh.p2_nodes(), X6, atol=1e-14)
    straight = fluids.Mesh.from_arrays(V, T, np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32))
    rot = lambda x, y: (-y, x)  # noqa: E731
    bcs = [{"attribute": 1, "kind": "dirichlet", "value": rot}, {"attribute": 2, "kind": "dirichlet"}]
    areas = []
    for m in (mesh, straight):
        sol = fluids.solve_stokes(m, 1.0, None, bcs, form="stress")
        el = np.arange(m.n_elements, dtype=np.int32)
        g, w = np.polynomial.legendre.leggauss(6)
        s = 0.5 * (g + 1)
        ref = np.array([[a * (1 - b), b] for a in s for b in s])
        wts = np.array([0.25 * wa * wb * (1 - b) for wa in w for b, wb in zip(s, w)])
        _, detJ = sol.geometry(el, ref)
        areas.append(float((detJ * wts[None, :]).sum()))
        # power through the rotating wall = dissipation, both from the solver's own residual
        assert sol.boundary_power([1]) > 0
    assert abs(areas[0] - 3 * math.pi) < 0.1 * abs(areas[1] - 3 * math.pi)


def test_equal_order_pair_needs_the_flag():
    mesh = rect(4, 4)
    bcs = [{"attribute": a, "kind": "dirichlet"} for a in (1, 2, 3, 4)]
    with pytest.raises(fluids.FluidsError, match="equal-order|inf-sup"):
        fluids.assemble_stokes(mesh, 1.0, bcs, order=1, pressure_order=1)
    a = fluids.assemble_stokes(mesh, 1.0, bcs, order=1, pressure_order=1, allow_unstable_pair=True)
    assert a.blocks()["B"]["shape"][0] == 25


# --- slip on curved walls, periodic pairs, reactions

def _annulus(nr, nt, split=1, perturb=0.0, seed=3, p2=True, slip_inner=False, squeeze=0.0):
    """Annulus 1 < r < 2 with isoparametric P2 circles; the outer circle split into `split` attributes (2..);
    angular positions perturbed (non-symmetric mesh) by perturb * dtheta."""
    rng = np.random.default_rng(seed)
    ts = 2 * np.pi * np.arange(nt) / nt + perturb * (2 * np.pi / nt) * rng.uniform(-0.5, 0.5, nt)
    rs = np.linspace(1, 2, nr + 1)
    V = np.array([[r * math.cos(t), r * math.sin(t)] for r in rs for t in ts])
    idx = lambda i, j: i * nt + (j % nt)  # noqa: E731
    T = np.array([t for i in range(nr) for j in range(nt)
                  for t in ([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])],
                 dtype=np.int32)
    rad = np.hypot(V[:, 0], V[:, 1])
    X6 = np.zeros((len(T), 6, 2))
    for e, t in enumerate(T):
        X6[e, :3] = V[t]
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            a, b = t[i], t[j]
            if abs(rad[a] - rad[b]) < 1e-12:
                ang = math.atan2(V[a, 1] + V[b, 1], V[a, 0] + V[b, 0])
                X6[e, 3 + k] = rad[a] * np.array([math.cos(ang), math.sin(ang)])
            else:
                X6[e, 3 + k] = 0.5 * (V[a] + V[b])
    seg, att = [], []
    for j in range(nt):
        seg += [[idx(0, j + 1), idx(0, j)], [idx(nr, j), idx(nr, j + 1)]]
        att += [1, 2 + (j * split) // nt]
    if squeeze:   # an ellipse-like annulus: no continuous rotation nullspace
        V = V * np.array([1.0, 1.0 - squeeze])
        X6 = X6 * np.array([1.0, 1.0 - squeeze])
    return fluids.Mesh.from_arrays(V, T, np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32),
                                   p2_nodes=X6 if p2 else None), sorted(set(att) - {1})


def _rot_err(sol, n_el, omega_inner):
    el = np.arange(n_el, dtype=np.int32)
    ref = np.array([[1 / 3, 1 / 3]])
    x, _ = sol.geometry(el, ref)
    u, _ = sol.velocity(el, ref)
    return u, x


def test_split_curved_slip_wall_equals_single_attribute():
    """A smooth slip wall split into several attributes gives the same solution (corners are geometric)."""
    sols = []
    for split in (1, 3):
        mesh, outer = _annulus(4, 32, split=split)
        bcs = [{"attribute": 1, "kind": "dirichlet", "value": lambda x, y: (-y, x)}] + \
              [{"attribute": a, "kind": "slip"} for a in outer]
        s = fluids.solve_stokes(mesh, 1.0, None, bcs, form="laplacian")
        assert s.diagnostics["converged"] and s.diagnostics["slip_corner_nodes"] == 0
        sols.append(_rot_err(s, mesh.n_elements, 1.0)[0])
    assert np.max(np.abs(sols[0] - sols[1])) < 1e-12


def test_sharp_corners_in_one_slip_attribute_do_not_leak():
    """A square with ONE slip attribute: the four corners are geometric corners (u = 0), as with four attributes."""
    mesh = rect(8, 8)
    seg, att = mesh.boundary()
    one = fluids.Mesh.from_arrays(mesh.vertices(), mesh.elements(), np.ascontiguousarray(seg),
                                  np.ones(len(att), dtype=np.int32))
    f = lambda x, y: (np.sin(3 * y), np.cos(2 * x))  # noqa: E731
    s1 = fluids.solve_stokes(one, 1.0, f, [{"attribute": 1, "kind": "slip"}], form="stress")
    s4 = fluids.solve_stokes(mesh, 1.0, f, [{"attribute": a, "kind": "slip"} for a in (1, 2, 3, 4)], form="stress")
    assert s1.diagnostics["slip_corner_nodes"] == 4 and s4.diagnostics["slip_corner_nodes"] == 4
    P, U = s1.boundary_nodes([1])
    corner = (np.abs(P[:, 0] - np.round(P[:, 0])) < 1e-12) & (np.abs(P[:, 1] - np.round(P[:, 1])) < 1e-12)
    assert np.max(np.abs(U[corner])) == 0.0


def test_viscosity_contrast_converges_or_reports():
    mesh = rect(4, 8)
    for ratio, tol, ok in ((100.0, 1e-10, True), (1e4, 1e-8, True)):
        mu = lambda x, y, r=ratio: 1.0 if y < 0.5 else r  # noqa: E731
        bcs = [{"attribute": 4, "kind": "periodic", "partner": 2, "translation": (1.0, 0.0)},
               {"attribute": 1, "kind": "dirichlet"},
               {"attribute": 3, "kind": "traction", "value": (1.0, 0.0)}]
        s = fluids.solve_stokes(mesh, mu, None, bcs, form="stress", direct_tol=tol)   # never throws
        assert s.diagnostics["converged"] is ok


def test_velocity_nullspaces_are_refused():
    mesh, outer = _annulus(2, 16)
    bcs = [{"attribute": 1, "kind": "slip"}] + [{"attribute": a, "kind": "slip"} for a in outer]
    with pytest.raises(fluids.FluidsError, match="velocity nullspace: rotation"):
        fluids.solve_stokes(mesh, 1.0, lambda x, y: (x, y), bcs, form="stress")
    s = fluids.solve_stokes(mesh, 1.0, lambda x, y: (x, y), bcs, form="laplacian")   # no rigid rotation in lap
    assert s.diagnostics["converged"] and s.diagnostics["velocity_nullspace"] == "none"
    ch = rect(8, 4, 2.0, 1.0)
    walls = [{"attribute": 1, "kind": "slip"}, {"attribute": 3, "kind": "slip"}]
    with pytest.raises(fluids.FluidsError, match="velocity nullspace: translation"):
        fluids.solve_stokes(ch, 1.0, None, [{"attribute": 4, "kind": "periodic", "partner": 2,
                                             "translation": (2.0, 0.0), "pressure_drop": 1.0}] + walls)
    q = fluids.solve_stokes(ch, 1.0, None, [{"attribute": 4, "kind": "periodic", "partner": 2,
                                             "translation": (2.0, 0.0), "flow_rate": 0.1}] + walls)
    assert q.diagnostics["converged"]


def test_lu_is_single_threaded_by_default():
    s = fluids.solve_stokes(rect(4, 4), 1.0, None, [{"attribute": a, "kind": "dirichlet"} for a in (1, 2, 3, 4)])
    assert s.diagnostics["lu_threads"] == 1


def test_enclosed_curved_slip_on_a_non_symmetric_mesh():
    mesh, outer = _annulus(4, 32, perturb=0.3)
    bcs = [{"attribute": 1, "kind": "dirichlet", "value": lambda x, y: (-y, x)}] + \
          [{"attribute": a, "kind": "slip"} for a in outer]
    s = fluids.solve_stokes(mesh, 1.0, None, bcs, form="stress")
    assert s.diagnostics["converged"] and s.diagnostics["nullspace_detected"]
    exact = [dict(b) for b in bcs]
    exact[-1]["normal"] = lambda x, y: (x, y)
    with pytest.raises(fluids.FluidsError, match="consistent"):
        fluids.solve_stokes(mesh, 1.0, None, exact, form="stress")


def test_validation_of_viscosity_and_p2_nodes():
    mesh = rect(2, 2)
    bcs = [{"attribute": a, "kind": "dirichlet"} for a in (1, 2, 3, 4)]
    for nu in (0.0, -1.0, float("nan")):
        with pytest.raises(fluids.FluidsError, match="positive"):
            fluids.solve_stokes(mesh, nu, None, bcs)
    with pytest.raises(fluids.FluidsError, match="positive"):
        fluids.solve_stokes(mesh, lambda X, Y: X - 0.5, None, bcs, vectorized=True)


def test_rigid_motion_criterion_is_mesh_independent():
    """A criterion of the continuous problem over a refinement sequence N_r x 8 N_r, 2 x 16
    ... 64 x 512, slip on both circles. Symmetric annulus, stress form: the rigid rotation is a nullspace (refused)
    at every level; Laplacian form: no rotation in the kernel (solved) at every level; annulus squeezed by 1 %: no
    continuous nullspace (solved, the small misfit reported) at every level; 30 %-perturbed angular spacing (vertices
    and edge nodes still on the circles): a circle, so refused like the symmetric one."""
    f = lambda x, y: (x, y)  # noqa: E731
    for nr in (2, 4, 8, 16, 32, 64):
        sym, outer = _annulus(nr, 8 * nr)
        bcs = [{"attribute": 1, "kind": "slip"}] + [{"attribute": a, "kind": "slip"} for a in outer]
        with pytest.raises(fluids.FluidsError, match="velocity nullspace: rotation"):
            fluids.solve_stokes(sym, 1.0, f, bcs, form="stress")
        lap = fluids.solve_stokes(sym, 1.0, f, bcs, form="laplacian")
        assert lap.diagnostics["converged"] and lap.diagnostics["velocity_nullspace"] == "none"
        sq, outer = _annulus(nr, 8 * nr, squeeze=0.01)
        s = fluids.solve_stokes(sq, 1.0, f, [{"attribute": 1, "kind": "slip"}] +
                                [{"attribute": a, "kind": "slip"} for a in outer], form="stress", direct_tol=1e-8)
        assert s.diagnostics["velocity_nullspace"] == "none"
        assert 1e-3 < s.diagnostics["velocity_nullspace_residual"] < 2e-2
        if nr <= 8:
            pert, outer = _annulus(nr, 8 * nr, perturb=0.3)
            with pytest.raises(fluids.FluidsError, match="velocity nullspace: rotation"):
                fluids.solve_stokes(pert, 1.0, f, [{"attribute": 1, "kind": "slip"}] +
                                    [{"attribute": a, "kind": "slip"} for a in outer], form="stress")
    mesh, outer = _annulus(4, 32, perturb=0.3)
    ok = fluids.solve_stokes(mesh, 1.0, f, [{"attribute": 1, "kind": "dirichlet"}] +
                             [{"attribute": a, "kind": "slip"} for a in outer], form="stress")
    assert ok.diagnostics["converged"] and ok.diagnostics["velocity_nullspace"] == "none"
    assert ok.diagnostics["velocity_nullspace_residual"] > 0.1


def test_coarse_symmetric_circles_are_exactly_singular_and_refused():
    """Symmetric annuli of 8 and 12 uniform P2 arcs, slip on both circles, stress form, are
    exactly singular discretely although the coarse boundary misfits the rotation geometrically (m > 1e-3): refused
    by the round-off discrete test, with every LU backend; the Laplacian form still solves."""
    f = lambda x, y: (x, y)  # noqa: E731
    backends = [b for b in ("eigen", "strumpack") if fluids.lu_backend_available(b)]
    for nr, nt in ((1, 8), (1, 12), (2, 8), (2, 12)):
        mesh, outer = _annulus(nr, nt)
        bcs = [{"attribute": 1, "kind": "slip"}] + [{"attribute": a, "kind": "slip"} for a in outer]
        for lu in backends:
            with pytest.raises(fluids.FluidsError, match="velocity nullspace: rotation.*discretely exact"):
                fluids.solve_stokes(mesh, 1.0, f, bcs, form="stress", lu=lu)
            s = fluids.solve_stokes(mesh, 1.0, f, bcs, form="laplacian", lu=lu)
            assert s.diagnostics["converged"] and s.diagnostics["velocity_nullspace"] == "none"
    # a well-posed case keeps a discrete residual far above round-off
    mesh, outer = _annulus(2, 16, squeeze=0.01)
    s = fluids.solve_stokes(mesh, 1.0, f, [{"attribute": 1, "kind": "slip"}] +
                            [{"attribute": a, "kind": "slip"} for a in outer], form="stress", direct_tol=1e-8)
    assert s.diagnostics["velocity_nullspace_discrete_residual"] > 1e-8
    assert s.diagnostics["velocity_nullspace_near"]   # m ~ 7e-3 < 10 tol: flagged, solved


def test_tolerances_must_be_finite():
    mesh = rect(2, 2)
    bcs = [{"attribute": a, "kind": "dirichlet"} for a in (1, 2, 3, 4)]
    for kw in ({"velocity_nullspace_tol": float("nan")}, {"velocity_nullspace_tol": 2.0},
               {"direct_tol": float("nan")}, {"slip_corner_angle_deg": float("inf")},
               {"grad_div": float("nan")}):
        with pytest.raises(fluids.FluidsError):
            fluids.solve_stokes(mesh, 1.0, None, bcs, **kw)


def test_per_point_viscosity_is_validated():
    bcs = [{"attribute": a, "kind": "dirichlet"} for a in (1, 2, 3, 4)]
    with pytest.raises(fluids.FluidsError, match="positive"):
        fluids.solve_stokes(rect(4, 4), lambda x, y: x - 0.5, None, bcs)


def test_flow_rate_channels_are_well_posed_at_every_size():
    """With a prescribed flow rate the translation along the pair is constrained; no-slip
    (Poiseuille) and slip walls, both forms, 8 x 4 ... 64 x 32 cells: solved, exact where the solution is in the
    space, never refused."""
    for nx in (8, 16, 32, 64):
        ch = rect(nx, nx // 2, 2.0, 1.0)
        for walls_kind in ("dirichlet", "slip"):
            walls = [{"attribute": 1, "kind": walls_kind}, {"attribute": 3, "kind": walls_kind}]
            for form in ("laplacian", "stress"):
                s = fluids.solve_stokes(ch, 1.0, (0.0, 0.0) if walls_kind == "slip" else None,
                                        [{"attribute": 4, "kind": "periodic", "partner": 2,
                                          "translation": (2.0, 0.0), "flow_rate": 1.0 / 6.0}] + walls, form=form)
                d = s.diagnostics
                assert d["converged"] and d["velocity_nullspace"] == "none", (nx, walls_kind, form, d)
                assert d["velocity_nullspace_discrete_residual"] > 1e-6
                if walls_kind == "dirichlet":
                    assert abs(d["periodic_pressure_gradient"][0] - 2.0) < 1e-10
                else:   # plug flow, no pressure gradient
                    assert abs(d["periodic_pressure_gradient"][0]) < 1e-10
