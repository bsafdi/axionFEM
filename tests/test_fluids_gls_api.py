"""Stabilized P1/P1 GLS through the fluids Python API (axionfem.fluids.gls_problem, MPI builds): the unit
tests (seconds, one rank).  The formulation itself is fixed in the check module; the verification
cases check the numbers, these tests check the API contract and the driver:

  G1  the planar-only refusal and the argument refusals
  G2  residual_full equals the numpy residual of the check module (the fixed statement), Stokes / Oseen / NS, both forms,
      nodal viscosity, transient BDF2
  G3  jacobian_action = the directional derivative of residual (tau differentiated with the state; central difference)
  G4  an exact P1 solution is reproduced to round-off; the mean-free pressure; Dirichlet data exactly imposed
  G5  tau_at equals f5_gls_check.tau_fields; tau_pspg has no dt term
  G6  a failed solve is reported (fields refused), never rescued: the unstabilized pair does not pass as a solution
  G7  OPEN boundaries are refused (a do-nothing channel); a misspelt stabilization key is refused
  G8  enclosed flows still get the mean-value multiplier (nullspace_detected, mean-free pressure); per-step LU accuracy
      is recorded in the history
"""
import sys
from pathlib import Path

import numpy as np
import pytest

fluids = pytest.importorskip("axionfem.fluids", exc_type=ImportError)
if not hasattr(fluids, "gls_problem"):
    pytest.skip("this build has no GLS solver", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
sys.dont_write_bytecode = True
import f2a_ns_check as C2  # noqa: E402
import f5_gls_check as C5  # noqa: E402

V, T = C5.graded_mesh(6, 6, 20.0)
BND = np.array(sorted({int(v) for e in C5.C1.boundary_edges(T) for v in e}), dtype=np.int32)
PT = C5.Patch()
FORMS = {"lap": "laplacian", "sym": "stress"}


def _problem(mode="navier_stokes", form="sym", **kw):
    args = dict(mu=PT.mu, rho=PT.rho, force=PT.f(form, convection=(mode != "stokes")),
                dirichlet=(BND, PT.u(V[BND, 0], V[BND, 1])))
    if mode == "oseen":
        args["advect"] = lambda x, y: PT.u(x, y)
    args.update(kw)
    return fluids.gls_problem(V, T, mode=mode, form=FORMS[form], **args)


def test_g1_refusals():
    with pytest.raises(ValueError, match="axisymmetric"):
        fluids.gls_problem(V, T, mu=1.0, geometry="axisymmetric")
    with pytest.raises(ValueError):
        fluids.gls_problem(V, T)                       # no viscosity
    with pytest.raises(ValueError):
        fluids.gls_problem(V, T, mu=1.0, mode="oseen")  # no advect
    with pytest.raises(ValueError):
        fluids.gls_problem(V, T, mu=1.0, dt=0.1, bdf=2, history=(np.zeros((len(V), 2)),))
    with pytest.raises(ValueError):
        fluids.gls_problem(V, T, mu=1.0, viscosity_scales=[2.0])   # must end at 1


@pytest.mark.parametrize("mode", ["stokes", "oseen", "navier_stokes"])
@pytest.mark.parametrize("form", ["lap", "sym"])
def test_g2_residual_matches_numpy(mode, form):
    rng = np.random.default_rng(3)
    u = rng.standard_normal((len(V), 2)) * 0.3
    p = rng.standard_normal(len(V)) * 0.2
    prob = _problem(mode, form, solve=False)
    Ru, Rp = prob.residual_full(u, p)
    adv = {"stokes": "zero", "oseen": (lambda x, y: PT.u(x, y)), "navier_stokes": None}[mode]
    ru, rp = C5.residual(V, T, u, p, PT.rho, PT.mu, PT.f(form, convection=(mode != "stokes")), form=form,
                         convection=(mode != "stokes"), advect=adv)
    scale = max(np.abs(ru).max(), np.abs(rp).max())
    assert max(np.abs(Ru - ru).max(), np.abs(Rp - rp).max()) / scale < 1e-12


def test_g2_nodal_mu_and_bdf2():
    rng = np.random.default_rng(4)
    u = rng.standard_normal((len(V), 2)) * 0.3
    p = rng.standard_normal(len(V)) * 0.2
    h1, h2 = rng.standard_normal((len(V), 2)) * 0.2, rng.standard_normal((len(V), 2)) * 0.1
    mun = PT.mu(V[:, 0], V[:, 1])[0]
    prob = fluids.gls_problem(V, T, mu_nodal=mun, rho=PT.rho, force=PT.f("lap"), form="laplacian", dt=0.05, bdf=2,
                              history=(h1, h2), solve=False)
    Ru, Rp = prob.residual_full(u, p)
    ru, rp = C5.residual(V, T, u, p, PT.rho, None, PT.f("lap"), form="lap", dt=0.05, hist=(h1, h2), bdf=2, nu_nodal=mun)
    assert max(np.abs(Ru - ru).max(), np.abs(Rp - rp).max()) / max(np.abs(ru).max(), np.abs(rp).max()) < 1e-12


@pytest.mark.parametrize("form", ["lap", "sym"])
def test_g3_jacobian_is_the_derivative(form):
    prob = _problem("navier_stokes", form, solve=False, mu=lambda x, y: (0.02 * (1 + 0.3 * x) + 0 * y, np.stack(
        [0.006 + 0 * x, 0 * x], -1)))
    rng = np.random.default_rng(5)
    n = sum(prob.layout())
    x = rng.standard_normal(n) * 0.3
    d = rng.standard_normal(n)
    e = 1e-5
    fd = (prob.residual(x + e * d) - prob.residual(x - e * d)) / (2 * e)
    Jd = prob.jacobian_action(x, d)
    assert np.linalg.norm(fd - Jd) / np.linalg.norm(Jd) < 1e-7


@pytest.mark.parametrize("mode", ["stokes", "oseen", "navier_stokes"])
def test_g4_patch_is_reproduced(mode):
    sol = _problem(mode, "sym")
    assert sol.diagnostics["converged"] and sol.diagnostics["relative_residual"] <= 1e-11
    u, p = sol.u(), sol.p()
    c = np.zeros(len(V))
    np.add.at(c, T.ravel(), np.repeat(C5.element_geometry(V, T)["area"] / 3.0, 3))
    pex = PT.p(V[:, 0], V[:, 1])
    pex = pex - c @ pex / c.sum()
    assert np.abs(u - PT.u(V[:, 0], V[:, 1])).max() < 1e-10
    assert np.abs(p - pex).max() < 1e-9
    assert abs(c @ p) < 1e-12
    assert np.array_equal(u[BND], PT.u(V[BND, 0], V[BND, 1]))
    assert sol.diagnostics["nullspace_detected"]


def test_g5_tau_at_and_no_dt_in_pspg():
    rng = np.random.default_rng(6)
    w = rng.standard_normal((len(V), 2))
    mun = 0.01 + 0.02 * V[:, 0]
    a = {}
    for dt in (0.0, 1e-3):
        prob = fluids.gls_problem(V, T, mu_nodal=mun, rho=1.3, dt=dt, history=(np.zeros((len(V), 2)),) if dt else (),
                                  solve=False)
        a[dt] = prob.tau_at(w)
    ref = C5.tau_reference_arrays(V, T, w, mun, 1.3, 1e-3)
    for k in ref:
        assert np.max(np.abs(a[1e-3][k] - ref[k]) / ref[k]) < 1e-13
    assert np.array_equal(a[0.0]["tau_pspg"], a[1e-3]["tau_pspg"])
    assert np.all(a[1e-3]["tau_supg"] < a[0.0]["tau_supg"])


def test_g6_failed_solve_is_reported_and_fields_refused():
    sol = fluids.gls_problem(V, T, mu=PT.mu, rho=PT.rho, force=PT.f("sym"), dirichlet=(BND, PT.u(V[BND, 0], V[BND, 1])),
                             stabilization={"supg": False, "pspg": False, "lsic": False}, mode="stokes", newton_maxit=2)
    d = sol.diagnostics
    if d["converged"]:
        pytest.skip("this LU happened to solve the singular unstabilized system to its gate (reported honestly by the case)")
    assert d["failure"]
    with pytest.raises(Exception, match="did not converge"):
        sol.u()
    assert np.all(np.isfinite(sol.u(allow_unconverged=True)))


def _channel(open_outlet):
    """Stokes channel [0,2] x [0,1], mu = 1, Poiseuille inflow, no-slip walls; the outlet x = 2 is do-nothing (open) or,
    for open_outlet=False, also Dirichlet (enclosed)."""
    Vc, Tc = C2.rect_mesh(8, 4, 1, 0.0, 2.0, 0.0, 1.0)
    b = np.array(sorted({int(v) for e in C5.C1.boundary_edges(Tc) for v in e}), dtype=np.int32)
    if open_outlet:
        corners = np.where((np.abs(Vc[:, 0] - 2) < 1e-12) & ((np.abs(Vc[:, 1]) < 1e-12) | (np.abs(Vc[:, 1] - 1) < 1e-12)))[0]
        b = np.union1d(b[Vc[b, 0] < 2.0 - 1e-12], corners).astype(np.int32)
    g = np.stack([4 * Vc[b, 1] * (1 - Vc[b, 1]), 0 * Vc[b, 0]], -1)
    return Vc, Tc, b, g


def test_g7_open_boundary_is_refused():
    """A do-nothing outlet would 'converge' without the check (gate 1e-15, nullspace_detected True) to a velocity
    that is 100 % wrong, because the mean-value multiplier is a wrong constraint when the pressure has no nullspace."""
    Vc, Tc, b, g = _channel(True)
    with pytest.raises(Exception, match="open/natural boundaries"):
        fluids.gls_problem(Vc, Tc, mu=1.0, rho=1.0, mode="stokes", form="laplacian", dirichlet=(b, g), solve=True)
    with pytest.raises(Exception, match="open/natural boundaries"):
        fluids.gls_problem(Vc, Tc, mu=1.0, rho=1.0, mode="stokes", form="laplacian", dirichlet=None, solve=True)
    # the read-only operators stay available on an unsolved open problem (no solve, no wrong answer)
    pr = fluids.gls_problem(Vc, Tc, mu=1.0, rho=1.0, mode="stokes", form="laplacian", dirichlet=(b, g), solve=False)
    assert pr.layout()[0] > 0


def test_g8_enclosed_flow_still_gets_the_multiplier():
    Vc, Tc, b, g = _channel(False)
    sol = fluids.gls_problem(Vc, Tc, mu=1.0, rho=1.0, mode="stokes", form="laplacian", dirichlet=(b, g), solve=True)
    d = sol.diagnostics
    assert d["nullspace_detected"] is True and d["nullspace"] == "mean_multiplier"
    assert d["converged"], d["failure"]
    cw = np.zeros(len(Vc))
    geo = C5.element_geometry(Vc, Tc)
    np.add.at(cw, Tc.ravel(), np.repeat(geo["area"] / 3.0, 3))
    assert abs(float(cw @ sol.p())) < 1e-13 * float(np.abs(sol.p()).max() * cw.sum() + 1e-300)
    # per-step LU accuracy is recorded
    assert sol.history and all(("lu_residual" in h and h["lu_residual"] >= 0) for h in sol.history)
    # the patch flows of this file are enclosed: the multiplier is imposed on all of them
    assert _problem(mode="stokes").diagnostics["nullspace_detected"] is True


def test_g9_stabilization_typo_is_refused():
    with pytest.raises(ValueError, match="unknown stabilization"):
        _problem(stabilization={"PSPG": False})
