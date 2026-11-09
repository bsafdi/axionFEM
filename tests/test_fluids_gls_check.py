"""Self-tests of the GLS check module benchmarks/fluids/cases/f5_gls_check.py (no solver; seconds: `pytest -m fast`).

The numpy residual is the fixed discrete problem: its exact-patch consistency, tau limits, vertex-numbering invariance
of tau, the sensitivity of the residual to the stabilization and grad-mu terms, the stabilized inf-sup estimate and
the time field are tested here on synthetic nodal states.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
sys.dont_write_bytecode = True
import f5_gls_check as C  # noqa: E402

fast = pytest.mark.fast


@fast
def test_rule_exact_to_degree_4():
    from math import factorial as f
    q, w = C.rule_points()
    assert abs(w.sum() - 0.5) < 1e-14
    for i in range(5):
        for j in range(5 - i):
            assert abs((w * q[:, 0] ** i * q[:, 1] ** j).sum() - f(i) * f(j) / f(i + j + 2)) < 1e-14


@fast
def test_G_invariant_under_vertex_renumbering_and_equilateral_value():
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.8]])
    base = C.element_geometry(V, np.array([[0, 1, 2]]))["G"][0]
    for perm in ([1, 2, 0], [2, 1, 0], [0, 2, 1]):
        assert np.allclose(C.element_geometry(V, np.array([perm]))["G"][0], base, atol=1e-13)
    Ve = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]])                 # equilateral, edge h = 2
    G = C.element_geometry(Ve, np.array([[0, 1, 2]]))["G"][0]
    assert np.allclose(G, np.eye(2) / 4.0, atol=1e-13)                           # G = I / h^2


@fast
def test_tau_limits():
    V, T = C.structured_mesh(8)
    geo = C.element_geometry(V, T)
    ne, nq = len(T), 6
    z = np.zeros((ne, nq, 2))
    mu = np.full((ne, nq), 0.7)
    tm, tp, tc = C.tau_fields(V, T, z, mu, 1.25, None)
    GG = np.einsum("eij,eij->e", geo["G"], geo["G"])[:, None]
    assert np.allclose(tm, 1 / np.sqrt(36 * (0.7 / 1.25) ** 2 * GG))             # diffusive limit
    w = np.ones((ne, nq, 2))
    tm2, _, _ = C.tau_fields(V, T, w, np.zeros((ne, nq)), 1.0, None)
    wGw = np.einsum("eqi,eij,eqj->eq", w, geo["G"], w)
    assert np.allclose(tm2, 1 / np.sqrt(wGw))                                    # convective limit
    tm3, tp3, tc3 = C.tau_fields(V, T, w, mu, 1.0, 1e-6)
    assert np.allclose(tm3, 5e-7, rtol=1e-6)                                     # dt -> 0: tau_supg -> dt/2
    assert np.all(tp3 > 100 * tm3)                                               # PSPG tau does NOT collapse
    assert np.allclose(tc3, 1 / (tp3 * np.einsum("eii->e", geo["G"])[:, None]))


def _patch_state(V, patch, t=0.0):
    return patch.u(V[:, 0], V[:, 1], t), patch.p(V[:, 0], V[:, 1], t)


@fast
@pytest.mark.parametrize("form", ["lap", "sym"])
@pytest.mark.parametrize("advect", [None, "oseen", "zero"])
def test_exact_p1_patch_has_zero_residual_on_interior_rows(form, advect):
    """A P1-representable exact solution is a zero of the fixed residual on every interior row (Galerkin + stabilization)."""
    V, T = C.distorted_unit(2)
    P = C.Patch()
    u, p = _patch_state(V, P)
    conv = advect != "zero"
    adv = None if advect is None else ("zero" if advect == "zero" else (lambda x, y: P.u(x, y)))
    Ru, Rp = C.residual(V, T, u, p, P.rho, P.mu, P.f(form, convection=conv), form=form, convection=conv, advect=adv)
    interior = np.ones(len(V), bool)
    for a, b in C.C1.boundary_edges(T):
        interior[[a, b]] = False
    assert np.abs(Ru[interior]).max() < 1e-13
    assert np.abs(Rp).max() < 1e-13           # the continuity rows (also at boundary nodes: no Dirichlet condition on p)


@fast
def test_transient_patch_zero_residual_and_time_term_matters():
    V, T = C.distorted_unit(2)
    P = C.Patch()
    dt = 0.1
    u1, p1 = _patch_state(V, P, 0.1)
    for bdf, hist in ((1, (P.u(V[:, 0], V[:, 1], 0.0),)), (2, (P.u(V[:, 0], V[:, 1], 0.1), P.u(V[:, 0], V[:, 1], 0.0)))):
        u, p = _patch_state(V, P, 0.1 if bdf == 1 else 0.2)
        if bdf == 2:
            hist = (P.u(V[:, 0], V[:, 1], 0.1), P.u(V[:, 0], V[:, 1], 0.0))
        Ru, Rp = C.residual(V, T, u, p, P.rho, P.mu, P.f("sym", t=0.1 if bdf == 1 else 0.2, transient=True), dt=dt, hist=hist,
                            bdf=bdf)
        interior = np.ones(len(V), bool)
        for a, b in C.C1.boundary_edges(T):
            interior[[a, b]] = False
        assert np.abs(Ru[interior]).max() < 1e-12 and np.abs(Rp).max() < 1e-12
        # forgetting u_t in the body force breaks it
        Ru2, _ = C.residual(V, T, u, p, P.rho, P.mu, P.f("sym", t=0.1 if bdf == 1 else 0.2, transient=False), dt=dt, hist=hist, bdf=bdf)
        assert np.abs(Ru2[interior]).max() > 1e-4


@fast
def test_residual_is_sensitive_to_the_grad_mu_and_stabilization_terms():
    V, T = C.distorted_unit(2)
    P = C.Patch()
    u, p = _patch_state(V, P)
    f = P.f("sym")
    Ru, Rp = C.residual(V, T, u, p, P.rho, P.mu, f)
    # exact P1 fields: the continuity rows vanish with and without stabilization
    # perturb the state: the stabilized and Galerkin residuals differ
    rng = np.random.default_rng(0)
    u2, p2 = u + 0.01 * rng.standard_normal(u.shape), p + 0.01 * rng.standard_normal(p.shape)
    R_s = C.residual(V, T, u2, p2, P.rho, P.mu, f)
    R_g = C.residual(V, T, u2, p2, P.rho, P.mu, f, stab=False)
    assert np.abs(R_s[1] - R_g[1]).max() > 1e-6          # PSPG changes the continuity rows
    mu0 = lambda x, y: (P.mu(x, y)[0], 0 * P.mu(x, y)[1])  # noqa: E731
    R_n = C.residual(V, T, u2, p2, P.rho, mu0, f)
    assert np.abs(R_n[0] - R_s[0]).max() > 1e-6          # dropping grad mu changes the stabilized momentum rows


@fast
def test_inf_sup_p1p1_unstabilized_has_spurious_modes_and_pspg_removes_them():
    V, T = C.structured_mesh(8)
    g = C.inf_sup_p1p1(V, T, 1.0, stab=False)
    s = C.inf_sup_p1p1(V, T, 1.0, stab=True)
    assert g["zero_modes"] > 1
    assert s["zero_modes"] == 1 and s["beta"] > 0.05


@fast
def test_time_field_and_nodal_norms_order():
    import json
    d = json.loads((ROOT / "tools/fluids_mms/out/gls_tr_sq.json").read_text())
    F = C.TimeField(d, 0.0)
    errs = []
    for N in (8, 16, 32):
        V, T = C.structured_mesh(N)
        u = F.u(V[:, 0], V[:, 1])
        p = F.p(V[:, 0], V[:, 1])
        n = C.norms(C.NodalEval(V, T, u, p), F)
        errs.append((n["uL2"], n["uH1"]))
    assert 1.9 < np.log2(errs[1][0] / errs[2][0]) < 2.1          # P1 interpolant: L2 order 2
    assert 0.95 < np.log2(errs[1][1] / errs[2][1]) < 1.05        # H1 order 1


@fast
def test_tau_pin_defects_zero_for_own_values_and_see_a_wrong_ci():
    V, T = C.structured_mesh(8)
    w = np.stack([np.sin(3 * V[:, 0]), np.cos(2 * V[:, 1])], -1)
    q, _ = C.rule_points()
    N = np.stack([1 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1]], -1)
    wq = np.einsum("qa,eai->eqi", N, w[T])
    tm, tp, tc = C.tau_fields(V, T, wq, np.full((len(T), 6), 0.05), 1.0, 0.01)
    d = C.tau_pin_defects(V, T, w, 0.05, 1.0, 0.01, {"tau_supg": tm, "tau_pspg": tp, "tau_lsic": tc})
    assert max(d.values()) < 1e-14
    tm2, _, _ = C.tau_fields(V, T, wq, np.full((len(T), 6), 0.05), 1.0, 0.01, CI=3.0)
    assert C.tau_pin_defects(V, T, w, 0.05, 1.0, 0.01, {"tau_supg": tm2, "tau_pspg": tp, "tau_lsic": tc})["tau_supg"] > 1e-4


@fast
def test_tau_renumbering_invariance_and_states():
    s = C.tau_state("variable_viscosity")
    Tr = C.renumber_triangles(s["T"])
    a = C.tau_reference_arrays(s["V"], s["T"], s["w"], s["mu"], s["rho"], s["dt"])
    b = C.tau_reference_arrays(s["V"], Tr, s["w"], s["mu"], s["rho"], s["dt"])
    assert C.tau_renumber_defect(a, b) < 1e-12
    # the unit RIGHT triangle metric would NOT be invariant: emulate it and see the change
    for name in C.TAU_STATES:
        st = C.tau_state(name)
        assert C.element_geometry(st["V"], st["T"])["area"].min() > 0
    an = C.tau_state("anisotropic")
    aspect = np.ptp(an["V"][:, 1]) / 8 / (np.diff(np.unique(an["V"][:, 1]))[0])
    assert aspect > 100


@fast
def test_parity_state_is_deterministic_and_stokes_uses_f_without_convection():
    import json
    root = ROOT / "tools" / "fluids_mms" / "out"
    F = {n: C.GlsField(json.loads((root / f"{n}.json").read_text())) for n in ("gls_sq_mix", "gls_sq_conv")}
    s1, s2 = C.parity_state("steady_ns_distorted", F), C.parity_state("steady_ns_distorted", F)
    assert np.array_equal(s1["u"], s2["u"]) and np.array_equal(s1["p"], s2["p"])
    st = C.parity_state("steady_stokes", F)
    x = np.array([[0.3]]); y = np.array([[0.4]])
    assert np.abs(st["f"](x, y) - F["gls_sq_mix"].f("sym")(x, y)).max() > 1e-3     # convection absent
    for n in C.PARITY_STATES:
        Ru, Rp = C.parity_reference(n, F)
        assert C.parity_defect(n, F, Ru, Rp) == 0.0 and np.isfinite(Ru).all()
