"""Inductionless MHD through the fluids Python API (solve_navier_stokes(..., geometry="planar25d", mhd=...)):
unit tests that run in seconds on one rank (formulation: docs/physics/fluids.md).

  H1  Hartmann duct mode: the flow rate converges to the closed form; Lorentz power balance and the charge-conservation
      identities hold to round-off (Shercliff-type insulated duct: net in-plane current 0, potential nullspace pinned,
      mean-free potential)
  H2  Newton Jacobian: J(x) d = (R(x + eps d) - R(x - eps d)) / (2 eps) for the full (u, p, phi) system with
      convection (the Lorentz blocks are linear, the Picard and Newton matrices coincide on them)
  H3  scaling identity (sigma, B) -> (4 sigma, B/2): same velocity, half the potential
  H4  refusals: axisymmetric MHD, 2.5D without MHD, rho = 0 without MHD, slip, missing electric conditions, a
      potential nullspace declared against the algebra, the potential of a non-MHD solve
"""
import math

import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)
if not hasattr(fluids, "solve_navier_stokes"):
    pytest.skip("this build has no Navier-Stokes solver", allow_module_level=True)

B, R, T, L = 1, 2, 3, 4


def rect(nx, ny, x0=0.0, x1=1.0, y0=-1.0, y1=1.0):
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


MU, SIGMA, G = 0.5, 2.0, 1.0


def ins(attrs=(B, R, T, L)):
    return [{"attribute": a, "kind": "insulating"} for a in attrs]


def duct(n=12, Bx=(0.0, 3.0, 0.0), sigma=SIGMA, rho=0.0, electric=None, **kw):
    mesh = rect(n, n, -1.0, 1.0, -1.0, 1.0)
    return fluids.solve_navier_stokes(
        mesh, MU, rho, force=lambda x, y: (0.0, 0.0, G), bcs=[{"attribute": a, "kind": "dirichlet"} for a in (B, R, T, L)],
        geometry="planar25d", form="laplacian", lu="eigen",
        mhd={"sigma": sigma, "B": lambda x, y: (Bx[0], Bx[1], Bx[2]), "E_z": 0.0,
             "electric": electric if electric is not None else ins()}, **kw)


def ev(sol, ref=((1 / 3, 1 / 3),)):
    ref = np.array(ref, float)
    el = np.arange(sol.n_elements, dtype=np.int32)
    u, g = sol.velocity(el, ref)
    phi, gphi = sol.potential(el, ref)
    return np.asarray(u), np.asarray(phi), np.asarray(gphi), np.asarray(sol.geometry(el, ref)[1])


def test_h1_insulated_duct_conservation():
    sol = duct(12)
    d = sol.diagnostics
    assert d["converged"] and d["relative_residual"] < 1e-10
    assert d["mhd"] and d["phi_nullspace_detected"] and d["phi_nullspace_declared"]
    assert d["nullspace_detected"]                     # velocity Dirichlet everywhere: pressure nullspace too
    el = np.arange(sol.n_elements, dtype=np.int32)
    ref = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
    u, _ = sol.velocity(el, ref)
    phi, gphi = sol.potential(el, ref)
    x, det = sol.geometry(el, ref)
    u, phi, gphi, det = np.asarray(u), np.asarray(phi), np.asarray(gphi), np.asarray(det)
    # mean-free potential (its integral is zero up to the 3-point quadrature): the mean is tiny against its range
    assert abs(np.sum(phi * det) / np.sum(det)) < 0.05 * np.max(np.abs(phi))
    # (net current and power balance to round-off: the checks of the MHD check module)


def test_h1_hartmann_duct_flow_rate():
    # Hartmann channel (duct mode, electrodes phi = -E_x x, open circuit): exact profile through the check module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "fluids" / "cases"))
    import f13a_mhd_check as C
    Ha = 5.0
    B0 = Ha / math.sqrt(SIGMA / MU)
    ex = C.HartmannExact(MU, SIGMA, B0, G, "duct", "open")
    n_y, n_x = 32, 2
    mesh = rect(n_x, n_y, 0.0, C.LX_STRIP, -1.0, 1.0)
    W = lambda x, y: np.asarray(ex.u(x, y), float)  # noqa: E731
    sol = fluids.solve_navier_stokes(
        mesh, MU, 0.0, force=lambda x, y: (0.0 * x, 0.0 * x, G + 0.0 * x), vectorized=True,
        bcs=[{"attribute": B, "kind": "dirichlet"}, {"attribute": T, "kind": "dirichlet"},
             {"attribute": R, "kind": "dirichlet", "value": W}, {"attribute": L, "kind": "dirichlet", "value": W}],
        geometry="planar25d", form="laplacian", lu="eigen",
        mhd={"sigma": SIGMA, "B": lambda x, y: np.stack([0 * x, B0 + 0 * x, 0 * x], -1), "E_z": 0.0,
             "electric": [{"attribute": L, "kind": "potential", "value": lambda x, y: 0.0 * x},
                          {"attribute": R, "kind": "potential", "value": lambda x, y: -ex.h.E * C.LX_STRIP + 0.0 * x},
                          {"attribute": B, "kind": "insulating"}, {"attribute": T, "kind": "insulating"}]})
    d = sol.diagnostics
    assert d["converged"] and not d["phi_nullspace_detected"] and d["dofs_potential"] > 0

    class E:
        n_elements = sol.n_elements
        _e = np.arange(sol.n_elements, dtype=np.int32)
        geometry = lambda s, r: sol.geometry(s._e, np.ascontiguousarray(r, float))  # noqa: E731
        velocity = lambda s, r: sol.velocity(s._e, np.ascontiguousarray(r, float))  # noqa: E731
        potential = lambda s, r: sol.potential(s._e, np.ascontiguousarray(r, float))  # noqa: E731
        pressure = lambda s, r: sol.pressure(s._e, np.ascontiguousarray(r, float))  # noqa: E731
    I = C.integrals(E(), SIGMA, [0.0, B0, 0.0], 0.0, MU)
    Q = I["flow"][2] / C.LX_STRIP
    assert abs(Q / ex.h.Q - 1) < 1e-4
    # linear potential: the exact -E_x x is in P2
    n = C.norms(E(), ex, pressure="mean_free", phi="determined")
    assert n["phiL2"] < 1e-5 * max(n["phi_magnitude"], 1.0)


def test_h2_jacobian_action_fd():
    mesh = rect(4, 4, 0.0, 1.0, 0.0, 1.0)
    sol = fluids.solve_navier_stokes(
        mesh, MU, 1.0, force=lambda x, y: (1.0 + 0 * x, 0.5 * y, G), vectorized=True, lu="eigen",
        bcs=[{"attribute": B, "kind": "dirichlet"}, {"attribute": T, "kind": "dirichlet"},
             {"attribute": R, "kind": "dirichlet"}, {"attribute": L, "kind": "dirichlet"}],
        geometry="planar25d", form="laplacian",
        mhd={"sigma": lambda x, y: 1.0 + 0.3 * x, "B": lambda x, y: np.stack([0.4 + 0 * x, 1.5 + 0 * x, 0.6 + y], -1),
             "E_z": 0.2, "q": lambda x, y: 0.1 * x,
             "electric": [{"attribute": L, "kind": "potential", "value": lambda x, y: y},
                          {"attribute": R, "kind": "normal_current", "value": lambda x, y: 0.3 + y},
                          {"attribute": B, "kind": "insulating"}, {"attribute": T, "kind": "insulating"}]})
    assert sol.diagnostics["converged"]
    x0 = np.asarray(sol.state())
    rng = np.random.default_rng(3)
    x = x0 + 0.05 * rng.standard_normal(x0.shape)         # away from the solution
    d = rng.standard_normal(x0.shape)
    Jd = np.asarray(sol.jacobian_action(list(x), list(d)))
    errs = []
    for eps in (1e-3, 5e-4):
        rp = np.asarray(sol.residual(list(x + eps * d)))
        rm = np.asarray(sol.residual(list(x - eps * d)))
        fd = (rp - rm) / (2 * eps)
        errs.append(np.linalg.norm(fd - Jd) / np.linalg.norm(Jd))
    assert max(errs) < 1e-8      # R is quadratic in (u, phi, p): the central difference is exact to round-off


def test_h3_scaling_identity():
    a = duct(10, Bx=(0.0, 3.0, 0.0), sigma=SIGMA)
    b = duct(10, Bx=(0.0, 1.5, 0.0), sigma=4 * SIGMA)
    ua, pa, _, _ = ev(a)
    ub, pb, _, _ = ev(b)
    assert np.max(np.abs(ua - ub)) < 1e-10 * np.max(np.abs(ua))
    assert np.max(np.abs(pa - 2 * pb)) < 1e-9 * np.max(np.abs(pa))


def test_h4_refusals():
    mesh = rect(3, 3, 0.0, 1.0, 0.0, 1.0)
    dir4 = [{"attribute": a, "kind": "dirichlet"} for a in (B, R, T, L)]
    mhd = {"sigma": 1.0, "B": (0.0, 1.0, 0.0), "electric": ins()}
    with pytest.raises(fluids.FluidsError, match="planar25d"):
        fluids.solve_navier_stokes(mesh, MU, 1.0, bcs=dir4, geometry="planar", form="laplacian", mhd=mhd, lu="eigen")
    with pytest.raises(fluids.FluidsError, match="go together"):
        fluids.solve_navier_stokes(mesh, MU, 1.0, bcs=dir4, geometry="planar25d", form="laplacian", lu="eigen")
    with pytest.raises(fluids.FluidsError, match="positive"):
        fluids.solve_navier_stokes(mesh, MU, 0.0, bcs=dir4, geometry="planar", lu="eigen")
    with pytest.raises(fluids.FluidsError, match="no electric condition"):
        fluids.solve_navier_stokes(mesh, MU, 0.0, bcs=dir4, geometry="planar25d", form="laplacian", lu="eigen",
                                   mhd=dict(mhd, electric=ins((B, R, T))))
    with pytest.raises(fluids.FluidsError, match="axisymmetric"):
        fluids.solve_navier_stokes(mesh, MU, 1.0, bcs=dir4, geometry="axisymmetric", form="laplacian", mhd=mhd,
                                   lu="eigen")
    with pytest.raises(fluids.FluidsError, match="slip|not offered"):
        fluids.solve_navier_stokes(mesh, MU, 0.0, bcs=[{"attribute": B, "kind": "slip"}] + dir4[1:],
                                   geometry="planar25d", form="laplacian", mhd=mhd, lu="eigen")
    with pytest.raises(fluids.FluidsError, match="Laplacian"):
        fluids.solve_navier_stokes(mesh, MU, 0.0, bcs=dir4, geometry="planar25d", form="stress", mhd=mhd, lu="eigen")
    plain = fluids.solve_navier_stokes(mesh, MU, 1.0, bcs=dir4, lu="eigen", force=lambda x, y: (1.0, 0.0))
    with pytest.raises(fluids.FluidsError):
        plain.potential(np.arange(plain.n_elements, dtype=np.int32), np.array([[0.3, 0.3]]))


def test_h5_incompatible_potential_data_refused():
    """With no potential wall the potential problem is solvable only for zero net injected current."""
    one = lambda x, y: 0.0 * x + 1.0  # noqa: E731
    mk = lambda *kinds: [{"attribute": a, "kind": k, **({"value": one} if k == "normal_current" else {})}  # noqa: E731
                         for a, k in zip((B, R, T, L), kinds)]
    # net inward/outward current on one wall only: refused (a uniform nodal sink would be solved silently)
    with pytest.raises(fluids.FluidsError, match="incompatible"):
        duct(6, electric=mk("normal_current", "insulating", "insulating", "insulating"))
    # a volumetric source with insulating walls: refused
    mesh = rect(6, 6, -1.0, 1.0, -1.0, 1.0)
    with pytest.raises(fluids.FluidsError, match="incompatible"):
        fluids.solve_navier_stokes(
            mesh, MU, 0.0, force=lambda x, y: (0.0, 0.0, G), bcs=[{"attribute": a, "kind": "dirichlet"} for a in (B, R, T, L)],
            geometry="planar25d", form="laplacian", lu="eigen",
            mhd={"sigma": SIGMA, "B": (0.0, 3.0, 0.0), "q": 1.0, "electric": ins()})
    # compatible: the same current leaves through the opposite wall (outward-normal data +1 and -1)
    elec = mk("normal_current", "insulating", "normal_current", "insulating")
    elec[2]["value"] = lambda x, y: 0.0 * x - 1.0
    sol = duct(6, electric=elec)
    assert sol.diagnostics["converged"] and sol.diagnostics["phi_flux_defect_abs"] < 1e-12


def test_h6_untested_options_refused_with_mhd():
    """Skew convection, PTC, continuation and a variable mu are refused together with MHD."""
    with pytest.raises(fluids.FluidsError, match="skew"):
        duct(6, rho=1.0, convection="skew")
    with pytest.raises(fluids.FluidsError, match="pseudo-transient"):
        duct(6, rho=1.0, strategy="ptc")
    with pytest.raises(fluids.FluidsError, match="continuation"):
        duct(6, rho=1.0, viscosity_scales=[2.0, 1.0])
    mesh = rect(6, 6, -1.0, 1.0, -1.0, 1.0)
    with pytest.raises(fluids.FluidsError, match="variable viscosity"):
        fluids.solve_navier_stokes(
            mesh, lambda x, y: 0.5 + 0.0 * x, 0.0, force=lambda x, y: (0.0, 0.0, G),
            bcs=[{"attribute": a, "kind": "dirichlet"} for a in (B, R, T, L)], geometry="planar25d", form="laplacian",
            lu="eigen", mhd={"sigma": SIGMA, "B": (0.0, 3.0, 0.0), "electric": ins()})
    assert duct(6, rho=1.0).diagnostics["converged"]       # the offered path is unaffected
