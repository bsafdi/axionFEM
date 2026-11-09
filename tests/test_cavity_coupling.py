"""Receiver coupling beta of a coaxial antenna port (axionfem.cavity_coupling, beta; docs/cavity/coupling.md).

Pure-Python parts: the resonance fit and the one-port lemma on synthetic impedances with known poles, the mesh
geometry.  FE parts (2D imported-mesh engine, ~1-2 min on one core): the Balleyguier pillbox + coax monopole at
h = 10 mm and flush, driven fit vs open/short eigenmode expansion, the reference Q_ext (Balleyguier 1997 as
tabulated by Dhavale & Mittal 2007, Table 3), Q_ext independent of Q0, and the haloscope hook.
"""
import math

import numpy as np
import pytest

from axionfem import cavity_coupling as cc
from axionfem import haloscope as hs


# ------------------------------------------------------------------------------------------------ synthetic
def _foster(w, w1, v1, Q0, C0, w2, v2):
    """One-port: a capacitive static term, the resonance of interest (lossy) and one higher mode (lossless);
    Z = i/(w C0) + i w v^2 / (w^2 - wc^2) summed (e^{-i w t})."""
    wc = w1 * (1 - 1j / (2 * Q0))
    return 1j / (w * C0) + 1j * w * v1 ** 2 / (w ** 2 - wc ** 2) + 1j * w * v2 ** 2 / (w ** 2 - w2 ** 2)


def _foster_roots(w1, v1, Q0, C0, w2, v2, Zload, guess):
    """Root nearest ``guess`` of Z + Zload (Zload = inf: the poles of Z) for :func:`_foster`, from the rational
    form Z = N/D, N = i[(w^2 - wc^2)(w^2 - w2^2) + w^2 C0 v1^2 (w^2 - w2^2) + w^2 C0 v2^2 (w^2 - wc^2)],
    D = w C0 (w^2 - wc^2)(w^2 - w2^2)."""
    P = np.polynomial.polynomial
    wc = w1 * (1 - 1j / (2 * Q0))
    a = np.array([-wc ** 2, 0, 1], complex)
    b = np.array([-w2 ** 2, 0, 1], complex)
    w2p = np.array([0, 0, 1], complex)
    N = 1j * (P.polymul(a, b) + C0 * v1 ** 2 * P.polymul(w2p, b) + C0 * v2 ** 2 * P.polymul(w2p, a))
    D = C0 * P.polymul(np.array([0, 1], complex), P.polymul(a, b))
    poly = D if Zload is None else P.polyadd(N, Zload * D)
    r = P.polyroots(poly)
    return r[np.argmin(np.abs(r - guess))]


@pytest.mark.parametrize("v1", [3e3, 1e4, 3e4])
def test_fit_port_resonance_recovers_poles(v1):
    w1, Q0, C0, w2, v2, Z0 = 2 * math.pi * 1.1e9, 2e4, 3e-13, 2 * math.pi * 2.3e9, 2e4, 50.0
    Z = lambda w: _foster(w, w1, v1, Q0, C0, w2, v2)  # noqa: E731
    p = _foster_roots(w1, v1, Q0, C0, w2, v2, None, w1)                # open-port pole
    pL = _foster_roots(w1, v1, Q0, C0, w2, v2, Z0, w1)                 # loaded resonance
    g0, gL = -p.imag, -pL.imag
    # grid over both the pole and the loaded resonance
    f = np.concatenate([p.real * (1 + np.linspace(-3, 3, 13) / (2 * Q0)),
                        pL.real * (1 + np.linspace(-3, 3, 13) * gL / pL.real)]) / (2 * math.pi)
    fit = cc.fit_port_resonance(f, Z(2 * math.pi * f), Z0)
    assert fit["fit_residual_rel"] < 1e-6
    assert fit["Q0"] == pytest.approx(p.real / (2 * g0), rel=1e-6, abs=0)
    assert fit["Q_L"] == pytest.approx(pL.real / (2 * gL), rel=1e-6, abs=0)
    assert fit["beta"] == pytest.approx((gL - g0) / g0, rel=1e-5, abs=0)
    assert fit["Q_ext"] == pytest.approx(p.real / (2 * (gL - g0)), rel=1e-5, abs=0)
    assert fit["converged"] and fit["inside_grid"] and not fit["warnings"]
    assert fit["f_loaded_Hz"] == pytest.approx(pL.real / (2 * math.pi), rel=1e-10, abs=0)


@pytest.mark.parametrize("C0,v1", [(3e-13, 3e4), (3e-13, 1e5), (1e-13, 2e5)])
def test_fit_on_production_grid_large_background(C0, v1):
    """Strongly pulled resonance (|Y| up to ~10 Z0): sweep_grid centred on the loaded resonance and on the pole
    (as coupling_study places it) recovers Q0, Q_L, beta; a grid that misses the pole is flagged, not silent."""
    w1, Q0, w2, v2, Z0 = 2 * math.pi * 1.1e9, 2e4, 2 * math.pi * 2.3e9, 2e4, 50.0
    Z = lambda w: _foster(w, w1, v1, Q0, C0, w2, v2)  # noqa: E731
    p = _foster_roots(w1, v1, Q0, C0, w2, v2, None, w1)
    pL = _foster_roots(w1, v1, Q0, C0, w2, v2, Z0, w1)
    g0, gL = -p.imag, -pL.imag
    f = cc.sweep_grid(pL.real / (2 * math.pi), pL.real / (2 * gL), p.real / (2 * math.pi), p.real / (2 * g0))
    fit = cc.fit_port_resonance(f, Z(2 * math.pi * f), Z0)
    assert fit["inside_grid"] and not fit["warnings"]
    assert fit["Q0"] == pytest.approx(p.real / (2 * g0), rel=1e-4, abs=0)
    assert fit["Q_L"] == pytest.approx(pL.real / (2 * gL), rel=1e-4, abs=0)
    assert fit["beta"] == pytest.approx((gL - g0) / g0, rel=1e-3, abs=0)
    # loaded window only, pole far outside: must be flagged when extrapolating
    f2 = cc.sweep_grid(pL.real / (2 * math.pi), pL.real / (2 * gL))
    fit2 = cc.fit_port_resonance(f2, Z(2 * math.pi * f2), Z0)
    pull = abs(pL.real - p.real) / gL     # 3.4, 8.3, 21 loaded half-widths
    if pull > 5:                          # at 3.4 the pole is within one of its own half-widths of the last sample
        assert not fit2["inside_grid"] and fit2["warnings"]


def test_one_port_lemma_error_is_order_one_over_Q():
    """Q_ext from the open/short secant lemma vs the exact loaded pole of a lossless-mode Foster one-port: the
    relative error falls like 1/Q_ext (independent derivation of the lemma's error)."""
    w1, C0, w2, v2, Z0 = 2 * math.pi * 1.1e9, 3e-13, 2 * math.pi * 2.3e9, 2e4, 50.0
    errs, qs = [], []
    for v1 in (3e4, 1e4, 3e3):
        ws = _foster_roots(w1, v1, 1e30, C0, w2, v2, 0.0, w1).real        # short-circuit resonance
        lem = cc.one_port_coupling(w1, ws, v1, Z0)
        pL = _foster_roots(w1, v1, 1e30, C0, w2, v2, Z0, w1)
        Qx = w1 / (2 * -pL.imag)
        errs.append(abs(lem["Q_ext"] / Qx - 1))
        qs.append(Qx)
        assert (w1 + lem["d_omega_loaded"]) / pL.real - 1 == pytest.approx(0, abs=5 / Qx ** 2 + 1e-12)
    errs, qs = np.array(errs), np.array(qs)
    assert np.all(errs * qs < 5)               # O(1/Q) with an O(1) coefficient
    assert errs[-1] < errs[0] / 30             # Q_ext x ~100 -> error / ~100


def test_one_port_lemma_rejects_unresolved():
    with pytest.raises(ValueError):
        cc.one_port_coupling(1.0, 1.0, 1.0, 50.0)


# ------------------------------------------------------------------------------------------------ geometry
@pytest.mark.parametrize("h", [0.0, 0.01])
def test_mesh_areas_and_groups(h):
    g = cc.ProbePillbox(R=0.1, L=0.15, a=0.825e-3, b=2.675e-3, h=h, coax_len=0.03, refine=0.5)
    P, tri, reg, edges, groups, meta = cc.probe_pillbox_mesh(g)
    p = P[tri[:, :3]]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    assert np.all(area > 0)
    assert area[reg == 0].sum() == pytest.approx(g.R * g.L - g.a * h, rel=1e-12)
    assert area[reg == 1].sum() == pytest.approx((g.b - g.a) * g.coax_len, rel=1e-12)
    seg = np.linalg.norm(P[edges[:, 1]] - P[edges[:, 0]], axis=1)
    length = {k: seg[groups == k].sum() for k in set(groups)}
    assert length["port"] == pytest.approx(g.b - g.a, rel=1e-12)
    assert length["axis"] == pytest.approx(g.L - h, rel=1e-12)
    assert length["probe"] == pytest.approx(g.a + h, rel=1e-12)
    assert length["coax_wall"] == pytest.approx(2 * g.coax_len, rel=1e-12)
    assert length["wall"] == pytest.approx(g.L + 2 * g.R - g.b, rel=1e-12)   # side, top, bottom r > b


def test_geometry_validation():
    with pytest.raises(ValueError):
        cc.ProbePillbox(a=0.005, b=0.004)
    with pytest.raises(ValueError):
        cc.ProbePillbox(h=0.2, L=0.15)
    assert cc.coax_Z0(1.0, math.e, 4.0) == pytest.approx(cc.ETA0 / (4 * math.pi), rel=1e-14)


# ------------------------------------------------------------------------------------------------ FE
def _balleyguier(h, **kw):
    return cc.ProbePillbox(R=0.1, L=0.15, a=0.825e-3, b=2.675e-3, coax_eps_r=2.0, h=h, coax_len=0.03,
                           sigma_S_m=5.8e7, **kw)


@pytest.fixture(scope="module")
def study10(tmp_path_factory):
    return cc.coupling_study(_balleyguier(10e-3), tmp_path_factory.mktemp("cpl10"))


def test_driven_vs_expansion(study10):
    o = study10
    ex, dr = o["expansion"], o["driven"]
    assert dr["fit_residual_rel"] < 1e-6
    assert abs(o["comparison"]["Q0"]) < 1e-4                    # open-port pole vs perturbative wall Q
    assert abs(o["comparison"]["Q_ext"]) < 6e-3                 # 4.4e-3 at refine 1, 1.3e-3 at refine 2
    assert abs(o["comparison"]["f_pole"]) < 3e-6
    assert ex["v_TEM"] == pytest.approx(ex["v"], rel=2e-3)      # port voltage two ways
    assert ex["norm_int_eps_e2_raw"] == pytest.approx(1.0, rel=1e-6)
    # the fitted residue is the eigenmode's i v^2 / 2 (sign-sensitive: Z convention, v, the pole term)
    assert abs(o["comparison"]["residue_over_iv2_2"]) < 1e-2
    assert ex["Y_ohm"] == pytest.approx(ex["Y_from_fields_ohm"], rel=1e-3)       # two Y estimates
    assert ex["coax_energy_fraction"] < 1e-2 and not o["warnings"]
    # open-circuit Psi(omega) vs the single-mode expansion near the loaded resonance (2.2 % at this mesh;
    # a sign or (1 +- i) error in the expansion gives O(1))
    assert o["Psi_check"]["max_rel_diff_loaded_window"] < 0.1
    assert dr["converged"] and dr["inside_grid"]
    # loaded resonance of the two routes (includes the expansion's -(1+i) w/(2 Q0) wall-reactance shift: 1.2e-6 at
    # this mesh; without the shift ~1.6e-5)
    assert abs(o["comparison"]["f_loaded"]) < 3e-6
    # the probe is a lossy wall (probe_lossy default) and carries part of the wall loss
    assert ex["wall_power_W"]["probe"] > 0 and ex["wall_power_W"]["wall"] > 0
    assert not o["engine_warnings"]
    # loaded port power at the loaded resonance = single-mode beta/(1+beta) Q_L O^2/(2 w)
    assert o["loaded_power"]["ratio"] == pytest.approx(1.0, abs=1e-3)


def test_reference_Q_ext(study10):
    """Balleyguier (1997) computed Q_ext = 10536 at 10 mm (Dhavale & Mittal 2007 Table 3; their Kroll-Yu column
    12086); stated agreement with measurement 5-15 %."""
    assert study10["driven"]["Q_ext"] == pytest.approx(10536, rel=0.05)
    assert study10["expansion"]["C"] == pytest.approx(0.6917, rel=0.01)


def test_haloscope_hook(study10):
    """Consistency of the hook (not an independent check of power_coefficient, which test_haloscope pins to the
    blind reference): signal_power with the computed beta, Q_L and C equals the FE port power into the receiver
    load times the axion's effective-current factor kappa^2 = 2 w eps0 P_coef / (C B^2 V) (unit-source FE run),
    i.e. the C, V, Q0, beta handed over are the ones that reproduce the full-wave power."""
    o = study10
    cav = cc.to_haloscope_cavity(o, B_T=8.0)
    f = o["loaded_power"]["f_Hz"]
    g = 1e-15
    P_hs = hs.signal_power(g, f, cav.B_T, cav.V_m3, cav.C, Q0=cav.Q0, beta=cav.beta)
    kappa2 = 2 * (2 * math.pi * f) * cc.EPS0 * hs.power_coefficient(g, f, 1.0, cav.V_m3, cav.C) / (cav.C * cav.V_m3)
    P_fe = o["loaded_power"]["P_fe_W"] * kappa2 * 8.0 ** 2      # the FE run has B0 = 1 T
    assert P_hs == pytest.approx(P_fe, rel=2e-3)
    assert (cav.Q0, cav.beta, cav.C, cav.V_m3) == (o["driven"]["Q0"], o["driven"]["beta"], o["expansion"]["C"],
                                                   o["expansion"]["V_m3"])
    cav_e = cc.to_haloscope_cavity(o, B_T=8.0, source="expansion")
    assert cav_e.beta == pytest.approx(cav.beta, rel=1e-2)
    with pytest.raises(KeyError):
        cc.to_haloscope_cavity({"expansion": o["expansion"]}, B_T=8.0)
    with pytest.raises(ValueError):
        cc.to_haloscope_cavity(o, B_T=8.0, source="comparison")
    with pytest.raises(ValueError):
        cc.to_haloscope_cavity({**o, "warnings": ["x"]}, B_T=8.0)


def test_flush_probe(tmp_path):
    """Flush probe (h = 0): expansion only (two eigen solves); Balleyguier 6.07e6 (Kroll-Yu 3.55e6).  (The
    invariance of the driven Q_ext under the wall conductivity is in the benchmark record, case sigma.)"""
    g = _balleyguier(0.0, refine=0.75)
    ro, meta = cc.solve_probe_pillbox(g, tmp_path / "o", port="open", modes=2, target_f_Hz=g.f_tm010_empty())
    rs, _ = cc.solve_probe_pillbox(g, tmp_path / "s", port="short", modes=2, target_f_Hz=g.f_tm010_empty())
    out = [cc.expansion_coupling(g, ro, rs, meta)]
    assert out[0]["Q_ext"] == pytest.approx(6.068e6, rel=0.1)
    # absolute Q0 (wall wiring and conductivity): the flush probe barely perturbs TM010, Q0 R_s = p01 eta0/(2(1+R/L))
    f = out[0]["f_open_Hz"]
    Rs = math.sqrt(math.pi * f * 4e-7 * math.pi / g.sigma_S_m)
    assert out[0]["Q0"] == pytest.approx(cc.P01 * cc.ETA0 / (2 * (1 + g.R / g.L)) / Rs, rel=1e-3)
    assert not out[0]["warnings"]
    assert out[0]["Q_ext_hartung_haebel"] == pytest.approx(out[0]["Q_ext"], rel=1e-3)
    est = cc.probe_coupling_estimates(_balleyguier(0.0), out[0]["Q0"])
    assert est["Q_ext_flush_TEM_3c"] == pytest.approx(out[0]["Q_ext"], rel=0.25)   # approximate closed form


def test_line_resonance_guard(tmp_path):
    """A coax stub whose half-wave resonance sits near the mode (90 mm, eps_r = 2, plus the 10 mm probe: ~1.09 GHz
    vs 1.147 GHz) mixes
    the open-port mode with the line: coupling_study must flag it, and the hook refuse it."""
    g = _balleyguier(10e-3, refine=0.6)
    g.coax_len = 0.09
    assert any(abs(f / g.f_tm010_empty() - 1) < 0.1 for f in cc.line_resonances(g, 2e9)["open_open"])  # 1.09 GHz
    with pytest.warns(RuntimeWarning, match="coax stub half-wave resonance"):
        o = cc.coupling_study(g, tmp_path)
    # the independent consistency checks fire too (not only the geometric guard)
    assert any("loaded power" in w for w in o["warnings"]) and any("Q_L driven" in w for w in o["warnings"])
    with pytest.raises(ValueError):
        cc.to_haloscope_cavity(o, B_T=8.0)
