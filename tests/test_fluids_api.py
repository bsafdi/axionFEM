"""Python boundary of the fluids library (axionfem.fluids, built in MPI builds).

Seconds on a login node. Exact Poiseuille with the Laplacian do-nothing outlet, explicit-BC refusals, the raw
pressure point, diagnostics of the consistent system, point evaluation, blocks, and the Babuška–Miller torque.
"""
import math

import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)


def channel_mesh(nx=4, ny=2, L=2.0, H=1.0):
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)
    V = np.array([[x, y] for y in ys for x in xs])
    vid = lambda i, j: j * (nx + 1) + i
    tri = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tri += [[a, b, d], [b, c, d]] if (i + j) % 2 == 0 else [[a, b, c], [a, c, d]]
    seg, att = [], []
    for i in range(nx):
        seg.append([vid(i, 0), vid(i + 1, 0)]); att.append(3)          # bottom wall
        seg.append([vid(i + 1, ny), vid(i, ny)]); att.append(3)        # top wall
    for j in range(ny):
        seg.append([vid(0, j + 1), vid(0, j)]); att.append(1)          # inflow
        seg.append([vid(nx, j), vid(nx, j + 1)]); att.append(2)        # outflow
    return fluids.Mesh.from_arrays(V, np.array(tri, dtype=np.int32), np.array(seg, dtype=np.int32),
                                   np.array(att, dtype=np.int32))


def test_poiseuille_do_nothing_laplacian_is_exact():
    mesh = channel_mesh()
    nu = 0.5
    prof = lambda x, y: (4 * y * (1 - y) if x < 1e-12 else 0.0, 0.0)
    sol = fluids.solve_stokes(mesh, nu, None, [
        {"attribute": 1, "kind": "dirichlet", "value": prof},
        {"attribute": 2, "kind": "do_nothing"},
        {"attribute": 3, "kind": "dirichlet"}], form="laplacian")
    d = sol.diagnostics
    assert d["converged"] and d["relative_residual"] < 1e-10
    assert not d["nullspace_detected"] and d["flux_defect"] == 0.0
    el = np.arange(mesh.n_elements, dtype=np.int32)
    ref = np.array([[0.2, 0.3], [0.6, 0.1]])
    x, detJ = sol.geometry(el, ref)
    u, g = sol.velocity(el, ref)
    p = sol.pressure(el, ref)
    assert np.allclose(u[..., 0], 4 * x[..., 1] * (1 - x[..., 1]), atol=1e-10)
    assert np.allclose(u[..., 1], 0, atol=1e-10)
    assert np.allclose(g[..., 0, 1], 4 - 8 * x[..., 1], atol=1e-9)
    assert np.allclose(p, 8 * nu * (2.0 - x[..., 0]), atol=1e-9)   # -nu u'' = -dp/dx, p = 0 at the outlet
    assert np.all(detJ > 0)


def test_boundary_conditions_are_explicit():
    mesh = channel_mesh()
    with pytest.raises(fluids.FluidsError, match="no boundary condition"):
        fluids.solve_stokes(mesh, 1.0, None, [{"attribute": 1, "kind": "dirichlet"},
                                              {"attribute": 3, "kind": "dirichlet"}])
    with pytest.raises(fluids.FluidsError, match="more than one"):
        fluids.solve_stokes(mesh, 1.0, None, [{"attribute": a, "kind": "dirichlet"} for a in (1, 2, 3, 3)])
    with pytest.raises(fluids.FluidsError, match="no boundary element"):
        fluids.solve_stokes(mesh, 1.0, None, [{"attribute": a, "kind": "dirichlet"} for a in (1, 2, 3, 9)])


def test_pressure_point_is_raw_and_blocks():
    mesh = fluids.Mesh.unit_square(6)
    lid = lambda x, y: (1.0 if y > 1 - 1e-12 else 0.0, 0.0)
    bcs = [{"attribute": a, "kind": "dirichlet", "value": lid} for a in (1, 2, 3, 4)]
    s0 = fluids.solve_stokes(mesh, 1.0, None, bcs)
    s1 = fluids.solve_stokes(mesh, 1.0, None, bcs, pressure_point={"x0": (0.5, 0.5), "value": 2.5})
    assert s0.diagnostics["nullspace_detected"] and s1.diagnostics["converged"]
    assert s1.diagnostics["pressure_point_value"] == pytest.approx(2.5, abs=1e-12)
    el = np.arange(mesh.n_elements, dtype=np.int32)
    ref = np.array([[1 / 3, 1 / 3]])
    diff = s1.pressure(el, ref) - s0.pressure(el, ref)
    assert np.ptp(diff) < 1e-9   # differs from the mean-free pressure by a constant only
    with pytest.raises(fluids.FluidsError, match="not a pressure node"):
        fluids.solve_stokes(mesh, 1.0, None, bcs, pressure_point={"x0": (0.51, 0.5), "value": 0.0})
    b = s0.blocks()
    nu_, np_ = s0.diagnostics["dofs_velocity"], s0.diagnostics["dofs_pressure"]
    assert b["A"]["shape"] == (nu_, nu_) and b["B"]["shape"] == (np_, nu_) and b["Mp"]["shape"] == (np_, np_)
    assert len(b["essential_velocity_dofs"]) > 0


def test_couette_torque_babuska_miller():
    mesh = fluids.Mesh.annulus(4, 24, 1.0, 2.0)
    sol = fluids.solve_stokes(mesh, 1.0, None, [
        {"attribute": 1, "kind": "dirichlet", "value": lambda x, y: (-y, x)},
        {"attribute": 2, "kind": "dirichlet"}])
    assert sol.diagnostics["converged"]
    exact = -16 * math.pi / 3
    t = sol.reaction([1])["torque"]
    tl = sol.reaction([1], residual_form="laplacian")["torque"]
    assert abs(t - exact) < 1e-3 * abs(exact)
    assert (tl - exact) / exact == pytest.approx(-0.375, abs=0.02)   # negative control: L pseudo-traction
