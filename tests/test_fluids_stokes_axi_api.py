"""Axisymmetric Stokes through the fluids Python API (axionfem.fluids, MPI builds): the axisymmetric path with swirl
(geometry="axisymmetric"): patch tests exact in P2 x P1, the strong axis condition at every P2 node, grad-div in the
axisymmetric form (the constant-mu identity sym == lap + mu grad-div), the r q-tested continuity (symmetric
operator, discrete divergence theorem), swirl decoupling, normal-dependent traction data on a curved boundary, the
pressure point on an axis vertex and the refusals. Seconds on a login node.
"""
import math

import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)

AX, Z0, RMAX, ZMAX = 4, 1, 2, 3


def rect(nx, ny, r0=0.0, r1=1.0, z0=0.0, z1=1.0):
    """Structured triangles on [r0, r1] x [z0, z1] (corner cells flipped); attributes 1 z0, 2 rmax, 3 zmax, 4 rmin."""
    rs, zs = np.linspace(r0, r1, nx + 1), np.linspace(z0, z1, ny + 1)
    V = np.array([[r, z] for z in zs for r in rs])
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
        att += [Z0, ZMAX]
    for j in range(ny):
        seg += [[vid(nx, j), vid(nx, j + 1)], [vid(0, j + 1), vid(0, j)]]
        att += [RMAX, AX]
    return fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32),
                                   np.array(seg, dtype=np.int32), np.array(att, dtype=np.int32))


def nodes(sol):
    """(x, u) at the 6 P2 reference nodes of every element."""
    ref = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)
    el = np.arange(sol.n_elements, dtype=np.int32)
    x, _ = sol.geometry(el, ref)
    u, _ = sol.velocity(el, ref)
    return x.reshape(-1, 2), u.reshape(-1, 3)


def maxdiff(sol, exact_u, exact_p=None):
    ref = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 3, 1 / 3]])
    el = np.arange(sol.n_elements, dtype=np.int32)
    x, _ = sol.geometry(el, ref)
    u, _ = sol.velocity(el, ref)
    du = float(np.max(np.abs(u - exact_u(x[..., 0], x[..., 1]))))
    dp = float(np.max(np.abs(sol.pressure(el, ref) - exact_p(x[..., 0], x[..., 1])))) if exact_p else 0.0
    return du, dp


def axi(mesh, nu, f, bcs, form, **kw):
    return fluids.solve_stokes(mesh, nu, f, bcs, form=form, geometry="axisymmetric", vectorized=True, **kw)


def hp(R, Z):  # Hagen-Poiseuille u_z = 1 - r^2, p = 4 mu (2 - z), mu = 1/2
    return np.stack([0 * R, 1 - R ** 2, 0 * R], -1)


@pytest.mark.parametrize("form", ["laplacian", "stress"])
def test_hagen_poiseuille_exact_with_pressure_point_on_the_axis(form):
    m = rect(4, 8, z1=2.0)
    bcs = [{"attribute": AX, "kind": "axis"}] + [{"attribute": a, "kind": "dirichlet", "value": hp}
                                                 for a in (Z0, RMAX, ZMAX)]
    s = axi(m, 0.5, None, bcs, form, pressure_point={"x0": (0.0, 1.0), "value": 2.0})
    d = s.diagnostics
    assert d["converged"] and d["nullspace_detected"]
    du, dp = maxdiff(s, hp, lambda R, Z: 2.0 * (2 - Z))
    assert du < 1e-12 and dp < 1e-11


@pytest.mark.parametrize("form", ["laplacian", "stress"])
def test_pressure_driven_pipe_with_normal_dependent_traction(form):
    """Inlet/outlet traction sigma_form n of the exact Poiseuille field, given as g(x, y, n): the geometric normal."""
    m = rect(4, 8, z1=2.0)

    def g(R, Z, NR, NZ):
        p = 2.0 * (2 - Z)
        dur = -R  # mu d_r u_z = 0.5 * (-2 r)
        if form == "stress":   # sigma = 2 mu eps - p I: sigma_rz = sigma_zr = mu d_r u_z
            return np.stack([-p * NR + dur * NZ, dur * NR - p * NZ, 0 * R], -1)
        return np.stack([-p * NR, dur * NR - p * NZ, 0 * R], -1)   # lap: sigma = mu grad u - p I
    bcs = [{"attribute": AX, "kind": "axis"}, {"attribute": RMAX, "kind": "dirichlet", "value": hp},
           {"attribute": Z0, "kind": "traction", "value": g, "normal_dependent": True},
           {"attribute": ZMAX, "kind": "traction", "value": g, "normal_dependent": True}]
    s = axi(m, 0.5, None, bcs, form)
    assert s.diagnostics["converged"] and not s.diagnostics["nullspace_detected"]
    du, dp = maxdiff(s, hp, lambda R, Z: 2.0 * (2 - Z))
    assert du < 1e-12 and dp < 1e-11


def test_axis_condition_at_every_p2_node_and_swirl_rigid_rotation():
    m = rect(4, 4)
    rot = lambda R, Z: np.stack([0 * R, 0 * R, R], -1)  # noqa: E731
    bcs = [{"attribute": AX, "kind": "axis"}] + [{"attribute": a, "kind": "dirichlet", "value": rot}
                                                 for a in (Z0, RMAX, ZMAX)]
    for form in ("laplacian", "stress"):
        s = axi(m, 1.3, None, bcs, form)
        x, u = nodes(s)
        on = np.abs(x[:, 0]) < 1e-14
        assert on.sum() > 0 and np.all(u[on][:, [0, 2]] == 0.0)
        du, _ = maxdiff(s, rot)
        assert du < 1e-12


def test_constant_mu_identity_sym_equals_lap_plus_mu_graddiv():
    m = rect(6, 6)
    g = lambda R, Z: np.stack([R * Z * (1 - R) ** 2, np.cos(R ** 2 + Z), R * Z], -1)  # noqa: E731
    f = lambda R, Z: np.stack([np.sin(Z) * R, 1 + R ** 2, R * np.cos(Z)], -1)       # noqa: E731
    bcs = [{"attribute": AX, "kind": "axis"}] + [{"attribute": a, "kind": "dirichlet", "value": g}
                                                 for a in (Z0, RMAX, ZMAX)]
    mu = 0.37
    ss = axi(m, mu, f, bcs, "stress")
    sl = axi(m, mu, f, bcs, "laplacian", grad_div=mu)
    sl0 = axi(m, mu, f, bcs, "laplacian")
    el = np.arange(ss.n_elements, dtype=np.int32)
    q = np.array([[0.2, 0.3], [0.6, 0.1]])
    ua, ub, uc = ss.velocity(el, q)[0], sl.velocity(el, q)[0], sl0.velocity(el, q)[0]
    assert np.max(np.abs(ua - ub)) < 1e-12 * max(1.0, np.max(np.abs(ua)))
    assert np.max(np.abs(ua - uc)) > 1e-6   # without grad-div the forms differ (data with u.n != 0)


def test_swirl_and_meridional_blocks_decouple():
    m = rect(4, 4)
    g = lambda R, Z: np.stack([R * Z * (1 - Z), R ** 2 * 0 + Z ** 2, R * (1 + Z)], -1)   # noqa: E731
    gm = lambda R, Z: g(R, Z) * np.array([1.0, 1.0, 0.0])                                # noqa: E731
    gt = lambda R, Z: g(R, Z) * np.array([0.0, 0.0, 1.0])                                # noqa: E731
    mu = lambda R, Z: 1.5 + 0.3 * np.sin(R ** 2 - Z)                                   # noqa: E731
    mk = lambda gg: [{"attribute": AX, "kind": "axis"}] + [{"attribute": a, "kind": "dirichlet", "value": gg}  # noqa
                                                           for a in (Z0, RMAX, ZMAX)]
    for form in ("laplacian", "stress"):
        s, sm, st = (axi(m, mu, None, mk(gg), form) for gg in (g, gm, gt))
        el = np.arange(s.n_elements, dtype=np.int32)
        q = np.array([[0.2, 0.3], [0.6, 0.1]])
        u, um, ut = (x.velocity(el, q)[0] for x in (s, sm, st))
        assert np.max(np.abs(u[..., 2] - ut[..., 2])) < 1e-13 and np.max(np.abs(ut[..., :2])) < 1e-13
        assert np.max(np.abs(u[..., :2] - um[..., :2])) < 1e-12


def test_refusals():
    m = rect(2, 2)
    wall = [{"attribute": a, "kind": "dirichlet"} for a in (Z0, RMAX, ZMAX)]
    with pytest.raises(fluids.FluidsError, match="axis"):   # the axis attribute declared as a wall
        axi(m, 1.0, None, wall + [{"attribute": AX, "kind": "dirichlet"}], "stress")
    with pytest.raises(fluids.FluidsError, match="normal_dependent"):  # a constant cannot depend on the normal
        axi(m, 1.0, None, [{"attribute": AX, "kind": "axis"}, {"attribute": Z0, "kind": "dirichlet"},
                           {"attribute": RMAX, "kind": "dirichlet"},
                           {"attribute": ZMAX, "kind": "traction", "value": (0.0, 0.0, 0.0),
                            "normal_dependent": True}], "stress")


def test_circular_couette_torque_with_curved_free_traction():
    """Shear-driven annulus [1, 2] x [0, 1]: sigma_theta_r = 1/2 on r = 2 given through the normal (n = e_r there)."""
    m = rect(8, 8, r0=1.0, r1=2.0)
    ex = lambda R, Z: np.stack([0 * R, 0 * R, 10 * R / 7 - 10 / (7 * R)], -1)  # noqa: E731

    def g(R, Z, NR, NZ):
        return np.stack([0 * R, 0 * R, 0.5 * NR], -1)
    bcs = [{"attribute": AX, "kind": "dirichlet", "value": (0.0, 0.0, 0.0)},
           {"attribute": RMAX, "kind": "traction", "value": g, "normal_dependent": True},
           {"attribute": Z0, "kind": "do_nothing"}, {"attribute": ZMAX, "kind": "do_nothing"}]
    s = axi(m, 0.7, None, bcs, "stress")
    du, _ = maxdiff(s, ex)
    assert s.diagnostics["converged"] and du < 1e-3   # P2 on 8 x 8: 1.3e-4


def test_axis_point_without_axis_edge_is_constrained():
    """A domain touching r = 0 at ONE vertex (a diamond with its apex on the axis), natural conditions on
    both edges at the apex: u_r = u_theta = 0 there all the same (reported as axis_point_nodes)."""
    n = 6
    s, t = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    S, T = s.ravel(), t.ravel()
    V = np.c_[1 + S - T, S + T]                        # (s, t) = (0, 1) -> apex (0, 1)
    vid = lambda i, j: j * (n + 1) + i  # noqa: E731
    tri, seg, att = [], [], []
    for j in range(n):
        for i in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tri += [[a, b, d], [b, c, d]] if (i == n - 1 and j == 0) or (i == 0 and j == n - 1) else [[a, b, c], [a, c, d]]
    for i in range(n):
        seg += [[vid(i, 0), vid(i + 1, 0)], [vid(i + 1, n), vid(i, n)]]
        att += [1, 3]
    for j in range(n):
        seg += [[vid(n, j), vid(n, j + 1)], [vid(0, j + 1), vid(0, j)]]
        att += [2, 4]
    m = fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.array(tri, dtype=np.int32), np.array(seg, dtype=np.int32),
                                np.array(att, dtype=np.int32))
    bcs = [{"attribute": 1, "kind": "dirichlet"}, {"attribute": 2, "kind": "dirichlet"},
           {"attribute": 3, "kind": "do_nothing"}, {"attribute": 4, "kind": "do_nothing"}]
    f = lambda R, Z: np.stack([np.ones_like(R), np.ones_like(R), np.ones_like(R)], -1)  # noqa: E731
    for form in ("laplacian", "stress"):
        sol = axi(m, 1.0, f, bcs, form)
        assert sol.diagnostics["converged"] and sol.diagnostics["axis_point_nodes"] == 1
        x, u = nodes(sol)
        on = np.abs(x[:, 0]) < 1e-14
        assert on.sum() > 0 and np.all(u[on][:, [0, 2]] == 0.0) and np.max(np.abs(u[on][:, 1])) > 1e-6


# ------------------------------------------------------------------ reactions (grad-div, Laplacian form)
def _sphere_runs(field, form, gamma, ks, kind):
    """Relative errors of the reaction (3D, x 2 pi) on the sphere family of the axisymmetric check module, per residual
    form; body data on the sphere (rest / rigid rotation), the exact solution on rho = 4."""
    import hashlib
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "benchmarks" / "fluids" / "cases"))
    import f1b_stokes_check as C
    p = root / "tools" / "fluids_mms" / "out" / "f1b" / f"{field}.json"
    F = C.load_field({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}, root="/")
    case = "translation" if field == "sphere_translation" else "rotation"
    key = "drag_z_per_radian" if kind == "force" else "torque_z_per_radian"
    exact = 2 * math.pi * float(eval(F.data["extra"][key]["numpy"], {"np": np, "__builtins__": {}}))
    mu = float(F.mu(np.array(2.0), np.array(0.0)))
    errs = {rf: [] for rf in ("solved", "laplacian")}
    for k in ks:
        V, T = C.sphere_mesh(k)
        X6 = C.sphere_p2_nodes(k)
        edges = C.boundary_edges(T)
        att = []
        for a, b in edges:
            x, y = 0.5 * (V[a] + V[b])
            att.append((1 if y > 0 else 2) if abs(x) < 1e-12 else (3 if math.hypot(x, y) < 2.5 else 4))
        m = fluids.Mesh.from_arrays(np.ascontiguousarray(V), np.ascontiguousarray(T, dtype=np.int32),
                                    np.array(edges, dtype=np.int32), np.array(att, dtype=np.int32), p2_nodes=X6)
        bcs = [{"attribute": 1, "kind": "axis"}, {"attribute": 2, "kind": "axis"},
               {"attribute": 3, "kind": "dirichlet", "value": lambda R, Z: C.sphere_body_data(R, Z, case)},
               {"attribute": 4, "kind": "dirichlet", "value": F.u}]
        s = axi(m, mu, None, bcs, form, grad_div=gamma)
        for rf in errs:
            r = s.reaction([3], residual_form=rf)
            v = r["force"][2] if kind == "force" else r["torque"]
            errs[rf].append(abs(v - exact) / abs(exact))
    return errs


def test_reaction_includes_graddiv_drag_order_4():
    """gamma = 10: the drag of the solved residual converges at order 4 (without the grad-div term in the residual
    the order falls to ~2.4)."""
    for form in ("stress", "laplacian"):
        e = _sphere_runs("sphere_translation", form, 10.0, (1, 2, 3), "force")["solved"]
        assert math.log2(e[1] / e[2]) > 3.5 and e[2] < 1e-3


def test_laplacian_form_reaction_is_the_physical_torque_on_a_rotating_sphere():
    e = _sphere_runs("sphere_rotation", "laplacian", 0.0, (1, 2, 3), "torque")
    assert math.log2(e["solved"][1] / e["solved"][2]) > 3.5 and e["solved"][2] < 2e-5
    assert e["laplacian"][2] == pytest.approx(1 / 3, abs=1e-3)   # the raw pseudo-traction: 2/3 of the torque


def test_planar_laplacian_form_reaction_on_a_rotating_cylinder():
    mesh = fluids.Mesh.annulus(8, 48, 1.0, 2.0)
    exact = -16 * math.pi / 3
    for gamma in (0.0, 10.0):
        sol = fluids.solve_stokes(mesh, 1.0, None, [
            {"attribute": 1, "kind": "dirichlet", "value": lambda x, y: (-y, x)},
            {"attribute": 2, "kind": "dirichlet"}], form="laplacian", grad_div=gamma)
        t = sol.reaction([1])["torque"]
        tl = sol.reaction([1], residual_form="laplacian")["torque"]
        assert abs(t - exact) < 1e-5 * abs(exact)
        assert (tl - exact) / exact == pytest.approx(-0.375, abs=0.01)   # raw pseudo-traction: 5/8


def test_laplacian_form_physical_reaction_refuses_variable_viscosity():
    m = rect(4, 4, r0=1.0, r1=2.0)
    bcs = [{"attribute": AX, "kind": "dirichlet", "value": (0.0, 0.0, 1.0)}, {"attribute": RMAX, "kind": "dirichlet"},
           {"attribute": Z0, "kind": "dirichlet"}, {"attribute": ZMAX, "kind": "dirichlet"}]
    s = axi(m, lambda R, Z: 1.0 + 0.1 * R, None, bcs, "laplacian")
    with pytest.raises(fluids.FluidsError, match="VARIABLE viscosity"):
        s.reaction([AX])
    assert math.isfinite(s.reaction([AX], residual_form="laplacian")["torque"])


def test_no_nanobind_leak_report_with_callables_that_refer_back_to_the_solution(tmp_path):
    """The solution's Python callables are visible to the GC (tp_traverse/tp_clear): a module-level callable (whose
    globals hold the solution) and an explicit cycle no longer leak at interpreter exit."""
    import os
    import subprocess
    import sys
    script = tmp_path / "cycle.py"
    script.write_text(
        "import numpy as np\n"
        "from axionfem import fluids\n"
        "m = fluids.Mesh.unit_square(4)\n"
        "g = lambda x, y: np.stack([np.ones_like(x), 0 * x], -1)\n"
        "sol = fluids.solve_stokes(m, lambda x, y: 1.0 + 0 * x, None,\n"
        "    [{'attribute': a, 'kind': 'dirichlet', 'value': g} for a in (1, 2, 3, 4)], vectorized=True)\n"
        "class H:\n"
        "    def __call__(self, x, y): return np.stack([np.ones_like(x), 0 * x], -1)\n"
        "h = H()\n"
        "s2 = fluids.solve_stokes(m, 1.0, None, [{'attribute': a, 'kind': 'dirichlet', 'value': h} for a in (1, 2, 3, 4)],\n"
        "    vectorized=True)\n"
        "h.sol = s2\n"
        "del h, s2\n"
        "print('ok', sol.diagnostics['converged'])\n")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=dict(os.environ), timeout=300)
    assert r.returncode == 0 and "ok True" in r.stdout, r.stderr[-2000:]
    assert "leaked" not in r.stderr, r.stderr[-2000:]
