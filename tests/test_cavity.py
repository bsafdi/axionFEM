"""Cavity haloscopes, axisymmetric (axionfem.cavity, beta): cavity domain and eigenmodes, targeted shift-invert,
form factor, Q0 from wall and dielectric losses.

References are the closed forms of an independent (blind) analytic derivation:

- pillbox a = 0.1 m, L = 0.2 m: TM0np, TE0np frequencies; C(TM0n0) = 4/chi_0n^2 (0.6916602761, 0.1312712271,
  0.0534138053), C = 0 for TM0np (p > 0) and TE modes;
  Q0 R_s = Z0 chi01 L / (2 (R + L)) (TM010, normal skin);
- coaxial cavity b = 0.2, a = 0.05 (rod spanning both caps): kb = 4.0976855393 (977.573603 MHz),
  C = 0.7949373125 over the vacuum volume;
- fully filled dielectric: f = f_vac / sqrt(eps_r), Q_d = 1/tan(delta).
TE0np geometry factor: Pozar, Microwave Engineering, Sec. 6.4 (TE_nml) with n = 0, G = Z0 k^3 a L / (2 (kc^2 L + 2 a beta^2)).
Achieved accuracies and convergence: docs/validation/cavity_2d.md.
"""
import math

import pytest

import axionfem as af
from axionfem import cavity as cv

C = 299792458.0
Z0 = 1.25663706212e-6 * C
A, L = 0.1, 0.2
CHI0 = [2.404825557695773, 5.520078110286311, 8.653727912911013]  # zeros of J0
CHI1P = 3.831705970207512                                          # first zero of J1 (= J0')
RHO = 1.7e-8  # any resistivity: Q0 R_s is independent of it


def _f(k):
    return C * k / (2 * math.pi)


F_TM010 = _f(CHI0[0] / A)
F_TM011 = _f(math.hypot(CHI0[0] / A, math.pi / L))
F_TE011 = _f(math.hypot(CHI1P / A, math.pi / L))


def _by_freq(modes, f):
    return min(modes, key=lambda m: abs(m["f_Hz"] - f))


@pytest.fixture(scope="module")
def pillbox(tmp_path_factory):
    d = cv.pillbox_design(A, L, h=0.004, modes=12, wall_material={
        "kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": RHO, "T_K": 4.0}})
    return d.solve(tmp_path_factory.mktemp("pillbox"), verbose=0)


# ------------------------------------------------------------------ cavity domain, eigenmodes
def test_pillbox_frequencies(pillbox):
    fw = pillbox.data["fullwave"]
    cav = fw["cavity"]
    assert cav["regions"] == ["cavity"] and cav["excluded_components"] == 1
    assert cav["volume_m3"] == pytest.approx(math.pi * A * A * L, rel=1e-12)
    tm, te = fw["eigenmodes"]["TM"]["modes"], fw["eigenmodes"]["TE"]["modes"]
    assert tm[0]["f_Hz"] == pytest.approx(F_TM010, rel=1e-6)
    # the builder's uniform background field draws no W_mag/cost warning: an eigenmode design has neither
    assert not any("exclude the background magnet" in w for w in pillbox.data["warnings"]), pillbox.data["warnings"]
    assert tm[1]["f_Hz"] == pytest.approx(F_TM011, rel=1e-6)
    assert te[0]["f_Hz"] == pytest.approx(F_TE011, rel=1e-6)
    for m in tm + te:  # restricted to the interior: no room modes, all energy inside
        assert list(m["energy_fraction"]) == ["cavity"]
        assert m["residual_true"] < 1e-7  # Lanczos stops at a 1e-8 Ritz estimate


def test_cavity_complete_spectrum(pillbox):
    """The lowest 12 modes of each family are exactly the analytic TM0np / TE0np sequence."""
    fw = pillbox.data["fullwave"]
    ref = cv.pillbox_modes(A, L, 5e9)
    for fam in ("TM", "TE"):
        got = [m["f_Hz"] for m in fw["eigenmodes"][fam]["modes"]]
        exact = [m["f_Hz"] for m in ref if m["family"] == fam][:len(got)]
        assert len(got) == 12
        for g, x in zip(got, exact):
            assert g == pytest.approx(x, rel=2e-5), fam


def test_cavity_spec_errors(tmp_path):
    d = cv.pillbox_design(A, L)
    d.doc["cavity"]["interior_point"] = [A + 0.005, 0.0]  # inside the wall
    with pytest.raises(af.AxionFEMError, match="inside the conductor region 'wall'"):
        d.solve(tmp_path / "wall", verbose=0)
    d = cv.pillbox_design(A, L)
    d.doc["regions"][1]["shape"]["union"].pop(1)  # no top cap: the cavity opens to the room
    with pytest.raises(af.AxionFEMError, match="not enclosed"):
        d.solve(tmp_path / "open", verbose=0)
    for bad, what in [({"frequencies": [1e9]}, "frequency sweep"), ({"modes": 0}, "modes > 0")]:
        d = cv.pillbox_design(A, L)
        d.doc["fullwave"].update(bad)
        with pytest.raises(af.AxionFEMError, match=what):
            d.validate()
    d = cv.pillbox_design(A, L)
    d.doc["cavity"]["colour"] = "red"
    with pytest.raises(af.AxionFEMError, match="colour"):
        d.validate()
    d = cv.pillbox_design(A, L)
    d.doc["cavity"]["form_factor"] = {"B0": "meissner"}
    with pytest.raises(af.AxionFEMError, match="unknown form_factor.B0"):
        d.validate()


def test_interior_point_on_wall_surface(tmp_path):
    """A point on the cavity/wall interface selects the adjacent cavity element (with a warning)."""
    for p in ([A, 0.0], [A / 2, L / 2]):
        d = cv.pillbox_design(A, L, h=0.01, modes=1, families=("TM",))
        d.doc["cavity"]["interior_point"] = p
        r = d.solve(tmp_path / f"p{p[0]}_{p[1]}", verbose=0)
        assert r.data["fullwave"]["cavity"]["volume_m3"] == pytest.approx(math.pi * A * A * L, rel=1e-12)
        assert any("lies on the surface of the conductor region 'wall'" in w for w in r.data["warnings"])


# ------------------------------------------------------------------ targeted shift-invert
def test_targeted_shift_invert(pillbox, tmp_path):
    ft = 2.5 * F_TM010
    d = cv.pillbox_design(A, L, h=0.004, modes=12, target_f_Hz=ft)
    r = d.solve(tmp_path / "target", verbose=0)
    fw = r.data["fullwave"]
    ref = [m for m in cv.pillbox_modes(A, L, 1.3 * ft) if abs(m["f_Hz"] / ft - 1) < 0.2]
    assert len(ref) == 8  # TE012, TM013, TM020, TM021, TE013, TM022, TM014, TE021
    for fam in ("TM", "TE"):
        blk = fw["eigenmodes"][fam]
        assert blk["target_f_Hz"] == ft and blk["solver"]["converged"]
        got = blk["modes"]
        assert len(got) == 12  # the requested count
        for m in (x for x in ref if x["family"] == fam):  # complete in the window
            g = _by_freq(got, m["f_Hz"])
            assert g["f_Hz"] == pytest.approx(m["f_Hz"], rel=2e-6), (fam, m["n"], m["p"])
            assert g["residual_true"] < 1e-7
        # same mesh, lowest-mode run: identical eigenvalues where the two sets overlap
        low = pillbox.data["fullwave"]["eigenmodes"][fam]["modes"]
        for g in got:
            o = _by_freq(low, g["f_Hz"])
            if abs(o["f_Hz"] / g["f_Hz"] - 1) < 1e-4:
                assert g["f_Hz"] == pytest.approx(o["f_Hz"], rel=1e-10)


# ------------------------------------------------------------------ form factor
def test_form_factor_pillbox(pillbox):
    fw = pillbox.data["fullwave"]
    tm = fw["eigenmodes"]["TM"]["modes"]
    exact = [0.6916602761, 0.1312712271, 0.0534138053]  # C = 4/chi_0n^2 (independent closed form)
    for chi, Cx, tol in zip(CHI0, exact, [1e-6, 2e-6, 2e-5]):
        assert 4 / chi ** 2 == pytest.approx(Cx, rel=1e-9)
        m = _by_freq(tm, _f(chi / A))
        ff = m["form_factor"]
        assert ff["C"] == pytest.approx(Cx, rel=tol)
        assert ff["C_rms"] == pytest.approx(ff["C"], rel=1e-12)  # uniform B0
        assert ff["V_m3"] == pytest.approx(math.pi * A * A * L, rel=1e-12)
    C010 = tm[0]["form_factor"]["C"]
    for m in tm:
        if all(abs(m["f_Hz"] / _f(chi / A) - 1) > 1e-3 for chi in CHI0):  # TM0np, p > 0
            assert m["form_factor"]["C"] < 1e-10 * C010
    for m in fw["eigenmodes"]["TE"]["modes"]:  # E_phi has no overlap with an axial B0 (structural; the
        assert m["form_factor"]["C"] == 0.0     # E_phi.B_phi path is tested in test_te_form_factor_toroidal_field)
    rows = pillbox.cavity_modes()
    assert len(rows) == 24 and rows[0]["family"] == "TM" and rows[0]["C"] == pytest.approx(C010)
    assert all(rows[i]["f_Hz"] <= rows[i + 1]["f_Hz"] for i in range(len(rows) - 1))
    assert len(pillbox.cavity_modes("TE")) == 12


def test_coaxial_cavity(tmp_path):
    a, b = 0.05, 0.2
    ref = cv.coax_tm_mode(a, b)
    assert ref["kb"] == pytest.approx(4.0976855393, rel=1e-10)  # independent closed form
    assert ref["f_Hz"] == pytest.approx(977.573603e6, rel=1e-9)
    assert ref["C"] == pytest.approx(0.7949373125, rel=1e-9)
    r = cv.coax_design(a, b, 0.2, h=0.005, modes=4).solve(tmp_path / "coax", verbose=0)
    fw = r.data["fullwave"]
    assert fw["enclosed_cavities"] == 1  # the coax gap is sealed: zero-flux constraint removes the static mode
    tm = fw["eigenmodes"]["TM"]["modes"]
    tem = _by_freq(tm, C / (2 * 0.2))  # TEM_1 (E_z = 0)
    assert tem["f_Hz"] == pytest.approx(C / 0.4, rel=1e-6)  # H ~ 1/r: not polynomial, 4e-7 at h = 5 mm
    assert tem["form_factor"]["C"] < 1e-10
    m = _by_freq(tm, ref["f_Hz"])
    assert m["f_Hz"] == pytest.approx(ref["f_Hz"], rel=2e-7)
    assert m["form_factor"]["C"] == pytest.approx(0.7949373125, rel=1e-6)
    assert m["form_factor"]["V_m3"] == pytest.approx(math.pi * (b * b - a * a) * 0.2, rel=1e-12)


def test_dielectric_sleeve(tmp_path):
    """Inhomogeneous filling: a coaxial dielectric sleeve spanning the cavity length (the TM010-like mode
    stays z-independent) against 1D Bessel matching; tests the eps weighting of C."""
    r1, r2, eps = 0.03, 0.05, 4.0
    ref = cv.sleeve_tm_mode(A, r1, r2, eps)
    d = cv.pillbox_design(A, L, h=0.004, modes=2, families=("TM",))
    tand = 1e-4
    d.doc["materials"]["sleeve"] = {"kind": "linear", "eps_r": eps, "tan_delta": tand}
    d.doc["regions"].insert(1, {"name": "sleeve", "shape": {"rect": {"r": [r1, r2], "z": [-L / 2, L / 2]}},
                                "material": "sleeve", "mesh_size": 0.004})
    r = d.solve(tmp_path / "sleeve", verbose=0)
    fw = r.data["fullwave"]
    assert sorted(fw["cavity"]["regions"]) == ["cavity", "sleeve"]
    m = fw["eigenmodes"]["TM"]["modes"][0]
    assert m["f_Hz"] == pytest.approx(ref["f_Hz"], rel=1e-6)
    assert m["form_factor"]["C"] == pytest.approx(ref["C"], rel=1e-5)
    assert ref["C"] < 4 / CHI0[0] ** 2  # eps weighting lowers C
    # dielectric loss of a partial filling: 1/Q_d = tan(delta) x electric-energy fraction in the sleeve
    assert ref["fe_sleeve"] == pytest.approx(0.71563749, rel=1e-7)  # 1D Bessel matching
    lo = m["losses"]
    assert lo["dielectric"]["sleeve"]["electric_energy_fraction"] == pytest.approx(ref["fe_sleeve"], rel=1e-6)
    assert lo["Q_dielectric"] == pytest.approx(1 / (tand * ref["fe_sleeve"]), rel=1e-6)
    assert "cavity" not in lo["dielectric"]  # the lossless vacuum carries no dielectric loss


def test_sleeve_reference_limits():
    """cavity.sleeve_tm_mode with one or two layers (rod on the axis, lining on the wall, full filling)."""
    full = cv.sleeve_tm_mode(A, 0.0, A, 2.0)
    assert full["f_Hz"] == pytest.approx(F_TM010 / math.sqrt(2.0), rel=1e-12)
    assert full["C"] == pytest.approx(4 / CHI0[0] ** 2 / 2.0, rel=1e-10) and full["fe_sleeve"] == pytest.approx(1.0)
    lining = cv.sleeve_tm_mode(A, 0.05, A, 9.4)  # dielectric touching the barrel
    assert lining["f_Hz"] == pytest.approx(533822233.81969285, rel=1e-10)
    assert lining["C"] == pytest.approx(0.14769221142547698, rel=1e-9)
    assert lining["fe_sleeve"] == pytest.approx(0.9080268463535965, rel=1e-9)
    rod = cv.sleeve_tm_mode(A, 0.0, 0.03, 9.4)
    assert rod["f_Hz"] < F_TM010 and 0 < rod["fe_sleeve"] < 1
    with pytest.raises(ValueError, match="r1 < r2"):
        cv.sleeve_tm_mode(A, 0.05, 0.03, 4.0)


def test_lossy_dielectric_does_not_shift_f(tmp_path):
    """Lossy media enter the eigenproblem with eps' exactly (no -tan^2(delta)/2 shift)."""
    f = []
    for tand in (0.0, 0.1):
        d = cv.pillbox_design(A, L, h=0.01, modes=2, families=("TM", "TE"),
                              fill={"kind": "linear", "eps_r": 2.0, "tan_delta": tand})
        fw = d.solve(tmp_path / f"t{tand}", verbose=0).data["fullwave"]["eigenmodes"]
        f.append([m["f_Hz"] for fam in ("TM", "TE") for m in fw[fam]["modes"]])
    for a, b in zip(*f):
        assert b == pytest.approx(a, rel=1e-12)


# ------------------------------------------------------------------ Q0 (walls, dielectric)
def test_wall_losses_pillbox(pillbox):
    fw = pillbox.data["fullwave"]
    m = fw["eigenmodes"]["TM"]["modes"][0]
    lo = m["losses"]
    Rs = cv.skin_Rs(m["f_Hz"], RHO)
    G = Z0 * CHI0[0] * L / (2 * (A + L))  # independent closed form
    assert lo["Q0"] * Rs == pytest.approx(G, rel=1e-6)
    assert lo["G_ohm"] == pytest.approx(G, rel=1e-6)
    assert lo["Q_wall"] == lo["Q0"] and "Q_dielectric" not in lo
    w = lo["walls"]["wall"]
    assert w["Rs_ohm"] == pytest.approx(Rs, rel=1e-12) and w["P_frac"] == pytest.approx(1.0)
    assert lo["T_over_Q_K"] == pytest.approx(4.0 / lo["Q0"], rel=1e-12)
    # TE011: Pozar Sec. 6.4 (TE_nml), n = 0; H_t from the consistent-flux recovery
    te = fw["eigenmodes"]["TE"]["modes"][0]
    kc, beta = CHI1P / A, math.pi / L
    k = math.hypot(kc, beta)
    G_te = Z0 * k ** 3 * A * L / (2 * (kc ** 2 * L + 2 * A * beta ** 2))
    assert te["losses"]["G_ohm"] == pytest.approx(G_te, rel=1e-5)
    assert te["losses"]["walls"]["wall"]["G_ohm_pointwise"] == pytest.approx(G_te, rel=5e-3)  # second order in h
    # every mode against the closed forms
    ref = cv.pillbox_modes(A, L, 5e9)
    for fam in ("TM", "TE"):
        for mm in fw["eigenmodes"][fam]["modes"][:6]:
            x = _by_freq([q for q in ref if q["family"] == fam], mm["f_Hz"])
            assert mm["losses"]["G_ohm"] == pytest.approx(x["G_ohm"], rel=5e-5), (fam, x["n"], x["p"])


def test_dielectric_fill_and_rf_table(tmp_path):
    eps_r, tand = 2.0, 1e-4
    table = {"model": "table", "f_Hz": [5e8, 1e9, 2e9], "Rs_ohm": [1e-3, 2e-3, 3e-3], "T_K": 4.0}
    d = cv.pillbox_design(A, L, h=0.004, modes=4, fill={"kind": "linear", "eps_r": eps_r, "tan_delta": tand},
                          wall_material={"kind": "conductor", "rf": table})
    r = d.solve(tmp_path / "diel", verbose=0)
    fw = r.data["fullwave"]
    tm = fw["eigenmodes"]["TM"]["modes"]
    m = tm[0]
    assert m["f_Hz"] == pytest.approx(F_TM010 / math.sqrt(eps_r), rel=1e-6)
    lo = m["losses"]
    assert lo["Q_dielectric"] == pytest.approx(1 / tand, rel=1e-8)
    assert lo["dielectric"]["cavity"]["electric_energy_fraction"] == pytest.approx(1.0, rel=1e-12)
    assert m["form_factor"]["C"] == pytest.approx(4 / CHI0[0] ** 2 / eps_r, rel=1e-5)  # eps-weighted C
    # wall: R_s from the table by log-log interpolation; G is independent of the filling (vacuum G / sqrt(eps_r))
    f = m["f_Hz"]
    assert 5e8 < f < 1e9
    Rs = 1e-3 * (f / 5e8) ** (math.log(2e-3 / 1e-3) / math.log(1e9 / 5e8))  # log-log between the first two points
    assert lo["walls"]["wall"]["Rs_ohm"] == pytest.approx(Rs, rel=1e-12)
    G = Z0 * CHI0[0] * L / (2 * (A + L)) / math.sqrt(eps_r)
    assert lo["Q_wall"] * Rs == pytest.approx(G, rel=1e-5)
    assert 1 / lo["Q0"] == pytest.approx(1 / lo["Q_wall"] + tand, rel=1e-12)
    # modes above the table: no extrapolation, Q0 withheld with a warning
    hi = [x for x in tm + fw["eigenmodes"]["TE"]["modes"] if x["f_Hz"] > 2e9]
    assert hi and all("Q0" not in x["losses"] and "R_s not available" in x["losses"]["walls"]["wall"]["note"]
                      for x in hi)
    assert any("without R_s" in w for w in r.data["warnings"])


def test_rf_table_validation():
    base = cv.pillbox_design(A, L)
    for rf, what in [({"model": "table", "f_Hz": [1e9], "Rs_ohm": [1e-3], "T_K": 4}, "equal length >= 2"),
                     ({"model": "table", "f_Hz": [2e9, 1e9], "Rs_ohm": [1e-3, 2e-3], "T_K": 4}, "strictly increasing"),
                     ({"model": "table", "f_Hz": [1e9, 2e9], "Rs_ohm": [1e-3, -2e-3], "T_K": 4}, "positive"),
                     ({"model": "bcs", "Rs_ohm": 1e-3, "T_K": 4}, "does not belong to the 'bcs' model"),
                     ({"model": "superconducting", "T_K": 4}, "must be 'power', 'skin', 'table', 'anomalous' or 'bcs'"),
                     ({"Rs_ohm": [1e-3, 2e-3], "T_K": 4}, "did you mean")]:
        d = af.Design(dict(base.doc))
        d.doc["materials"] = {"wall_cu": {"kind": "conductor", "rf": rf}}
        with pytest.raises(af.AxionFEMError, match=what):
            d.validate()


# ------------------------------------------------------------------ edge cases: target on a mode, sheets, B0 choices
def test_target_at_eigenfrequency(tmp_path):
    """target_f_Hz equal to a converged eigenfrequency (same mesh): the shift-invert factorization is singular to
    round-off (pivoting LU fallback) and the Lanczos start vector collapses onto that mode; the deflated restart
    must still return the requested number of modes."""
    base = cv.pillbox_design(A, L, h=0.01, modes=4, families=("TM",)).solve(tmp_path / "base", verbose=0)
    f_low = [m["f_Hz"] for m in base.data["fullwave"]["eigenmodes"]["TM"]["modes"]]
    for ft in (f_low[0], f_low[0] * (1 + 1e-15), f_low[2]):
        r = cv.pillbox_design(A, L, h=0.01, modes=4, families=("TM",), target_f_Hz=ft).solve(
            tmp_path / f"t{ft}", verbose=0)
        blk = r.data["fullwave"]["eigenmodes"]["TM"]
        assert len(blk["modes"]) == 4
        sol = blk["solver"]
        assert sol["converged"] and sol["factorization"] == "LU" and sol["lanczos_restarts"] >= 1
        for m, f in zip(blk["modes"], f_low):  # the four lowest are also the four nearest these targets
            assert m["f_Hz"] == pytest.approx(f, rel=1e-10)
            assert m["residual_true"] < 1e-7


def _iris(tmp_path, tag, *, sheet=False, t=None, h=0.004):
    d = cv.pillbox_design(A, L, h=h, modes=4, wall_material={
        "kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": RHO, "T_K": 4.0}})
    if sheet:  # annular iris at z = 0, r in [a/2, a]: zero-thickness sheet with the cavity on both sides
        d.doc["curves"] = [{"name": "iris", "path": {"segment": [[A / 2, 0.0], [A, 0.0]]}, "material": "wall_cu",
                            "mesh_size": h}]
    elif t:  # the same iris as a thin conducting region
        d.doc["regions"].append({"name": "iris", "shape": {"rect": {"r": [A / 2, A], "z": [-t / 2, t / 2]}},
                                 "material": "wall_cu", "mesh_size": min(h, t)})
    d.doc["cavity"]["interior_point"] = [0.02, 0.05]
    return d.solve(tmp_path / tag, verbose=0)


def test_iris_sheet_two_sided(tmp_path):
    """Conducting sheet inside the cavity.

    TE: TE012 has E_phi = 0 on the plane z = 0, so the iris does not perturb it, but H_r is tangential there with
    equal and opposite currents on the two faces: G of the iris (both faces) and of the whole cavity are closed
    forms. TM: H_phi must jump across the iris (PEC on each face); TM011 has its E_r maximum at z = 0 and is
    lowered strongly; checked against the iris as a thin conducting region, t -> 0."""
    r = _iris(tmp_path, "sheet", sheet=True)
    fw = r.data["fullwave"]
    cav = fw["cavity"]
    assert cav["two_sided_sheets"] == ["iris"] and cav["TM_split_dofs"] > 0
    assert cav["sheets_with_free_edge"] == ["iris"] and any("free edge" in w for w in r.data["warnings"])
    # TE012 closed forms: E_phi = J1(kc r) sin(beta (z + L/2)), H_r ~ beta J1 cos, H_z ~ kc J0 sin
    from scipy.integrate import quad
    from scipy.special import j0, j1
    kc, beta = CHI1P / A, 2 * math.pi / L
    k = math.hypot(kc, beta)
    IJ1 = quad(lambda x: j1(kc * x) ** 2 * x, 0, A, epsabs=0, epsrel=1e-13)[0]
    IJ0 = quad(lambda x: j0(kc * x) ** 2 * x, 0, A, epsabs=0, epsrel=1e-13)[0]
    Ii = quad(lambda x: j1(kc * x) ** 2 * x, A / 2, A, epsabs=0, epsrel=1e-13)[0]
    vol = 2 * math.pi * (L / 2) * (beta ** 2 * IJ1 + kc ** 2 * IJ0)
    walls = 2 * math.pi * A * (L / 2) * kc ** 2 * j0(CHI1P) ** 2 + 4 * math.pi * beta ** 2 * IJ1
    iris = 4 * math.pi * beta ** 2 * Ii  # both faces
    G_iris, G_tot = Z0 * k * vol / iris, Z0 * k * vol / (walls + iris)
    te = _by_freq(fw["eigenmodes"]["TE"]["modes"], _f(k))
    assert te["f_Hz"] == pytest.approx(_f(k), rel=2e-6)
    lo = te["losses"]
    assert lo["walls"]["iris"]["G_ohm"] == pytest.approx(G_iris, rel=3e-4)  # 1.0e-4 at h = 4 mm, 2.2e-5 at 2 mm
    assert lo["walls"]["iris"]["G_ohm_pointwise"] == pytest.approx(G_iris, rel=3e-3)
    assert lo["G_ohm"] == pytest.approx(G_tot, rel=1e-4)
    assert lo["walls"]["wall"]["G_ohm"] == pytest.approx(Z0 * k * vol / walls, rel=1e-5)
    # TM: TM010 (E along z, tangential E_r = 0 at z = 0) is unperturbed; TM011 drops by ~10 %
    tm = [m["f_Hz"] for m in fw["eigenmodes"]["TM"]["modes"]]
    assert tm[0] == pytest.approx(F_TM010, rel=1e-6)
    f_sheet = tm[1]
    assert f_sheet / F_TM011 - 1 == pytest.approx(-0.104, abs=0.002)
    f_thin = []
    for t in (0.002, 0.001, 0.0005):
        f_thin.append(_iris(tmp_path, f"thin{t}", t=t).data["fullwave"]["eigenmodes"]["TM"]["modes"][1]["f_Hz"])
    assert f_thin[0] < f_thin[1] < f_thin[2] < f_sheet  # thinner iris -> closer to the sheet
    # t -> 0 by a power-law fit f = f0 - c t^p through the three thicknesses
    d1, d2 = f_thin[1] - f_thin[0], f_thin[2] - f_thin[1]
    q = d2 / d1  # = 2^-p
    f0 = f_thin[2] + d2 * q / (1 - q)
    shift = F_TM011 - f0
    assert abs(f_sheet - f0) < 0.015 * shift  # 0.8 % of the shift at h = 4 mm (0.3 % at 2 mm; edge singularity)


def test_form_factor_superconducting_walls(tmp_path):
    """Default B0 = the applied field: superconducting cavity walls do not expel it;
    B0 = "static" (Meissner) gives B0 ~ 0 inside, and C is then withheld with a warning."""
    res = {}
    for B0 in ("applied", "static"):
        d = cv.pillbox_design(A, L, h=0.01, modes=2, families=("TM",), wall_material={"kind": "superconductor"})
        d.doc["cavity"]["form_factor"] = {"B0": B0}
        res[B0] = d.solve(tmp_path / B0, verbose=0)
    cav = res["applied"].data["fullwave"]["cavity"]
    assert cav["B0_field"] == "applied" and cav["B0_transparent_walls"] == ["wall"] and not cav["B0_negligible"]
    assert cav["B0_rms_T"] == pytest.approx(1.0, rel=1e-10)
    ff = res["applied"].data["fullwave"]["eigenmodes"]["TM"]["modes"][0]["form_factor"]
    assert ff["C"] == pytest.approx(4 / CHI0[0] ** 2, rel=1e-5)
    cav = res["static"].data["fullwave"]["cavity"]
    assert cav["B0_negligible"] and cav["B0_rms_T"] < 1e-10 * cav["B0_max_design_T"]
    ff = res["static"].data["fullwave"]["eigenmodes"]["TM"]["modes"][0]["form_factor"]
    assert "C" not in ff and "C_rms" not in ff
    assert any("negligible" in w for w in res["static"].data["warnings"])


def test_form_factor_reference_field(tmp_path):
    """Non-uniform B0 (background + an external solenoid): by default C is normalised by the cavity rms of B0,
    so it does not depend on interior_point; an explicit B_ref_point uses |B0| there."""
    rows = []
    for p, bref in (([0.05, 0.0], None), ([0.09, 0.09], None), ([0.05, 0.0], [0.0, 0.0])):
        d = cv.pillbox_design(A, L, h=0.01, modes=1, families=("TM",))
        d.doc["regions"].append({"name": "coil", "shape": {"rect": {"r": [0.2, 0.22], "z": [-0.05, 0.05]}},
                                 "material": "copper"})
        d.doc["windings"] = [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 3e5}]
        d.doc["cavity"]["interior_point"] = p
        if bref:
            d.doc["cavity"]["form_factor"] = {"B_ref_point": bref}
        d.doc["mesh"]["domain_radius"] = 0.6
        r = d.solve(tmp_path / f"r{len(rows)}", verbose=0)
        rows.append((r.data["fullwave"]["cavity"], r.data["fullwave"]["eigenmodes"]["TM"]["modes"][0]["form_factor"]))
    (c0, f0), (c1, f1), (c2, f2) = rows
    assert c0["B_ref"].startswith("volume rms") and f0["C"] == pytest.approx(f0["C_rms"], rel=1e-12)
    assert f1["C"] == pytest.approx(f0["C"], rel=1e-12)  # independent of the selection point
    assert c2["B_ref"] == "point" and abs(c2["B_ref_over_B0_rms"] - 1) > 0.01  # non-uniform B0
    assert f2["C"] == pytest.approx(f0["C_rms"] / c2["B_ref_over_B0_rms"] ** 2, rel=1e-12)


def test_te_form_factor_toroidal_field(tmp_path):
    """E_phi . B_phi: an annular cavity (r in [0.1, 0.2], |z| < 0.1) inside a toroidal winding, B_phi = K/r, copper
    walls transparent. TE011 of the coaxial cavity: E_phi = R(r) sin(pi (z + L/2)/L) with
    R = J1(kc r) Y1(kc a) - Y1(kc r) J1(kc a), R(b) = 0, and
    C = C_rms = 8 (int R dr)^2 / (pi^2 ln(b/a) int R^2 r dr). A nonzero TE C that a wrong E_phi.B_phi term would
    miss."""
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import j1, y1
    ai, bo, Lc, t = 0.1, 0.2, 0.2, 0.01
    R = lambda x, kc: j1(kc * x) * y1(kc * ai) - y1(kc * x) * j1(kc * ai)  # noqa: E731
    ks = np.linspace(1.0, 80.0, 8000)
    i = next(i for i in range(len(ks) - 1) if R(bo, ks[i]) * R(bo, ks[i + 1]) < 0)
    kc = brentq(lambda x: R(bo, x), ks[i], ks[i + 1], xtol=1e-14, rtol=1e-15)
    I1 = quad(lambda x: R(x, kc), ai, bo, epsabs=0, epsrel=1e-13)[0]
    I2 = quad(lambda x: R(x, kc) ** 2 * x, ai, bo, epsabs=0, epsrel=1e-13)[0]
    f_ref, C_ref = _f(math.hypot(kc, math.pi / Lc)), 8 * I1 ** 2 / (math.pi ** 2 * math.log(bo / ai) * I2)
    wall = {"union": [{"rect": {"r": [ai - t, ai], "z": [-Lc / 2 - t, Lc / 2 + t]}},
                      {"rect": {"r": [bo, bo + t], "z": [-Lc / 2 - t, Lc / 2 + t]}},
                      {"rect": {"r": [ai, bo], "z": [Lc / 2, Lc / 2 + t]}},
                      {"rect": {"r": [ai, bo], "z": [-Lc / 2 - t, -Lc / 2]}}]}
    frame = {"difference": [{"rect": {"r": [0.05, 0.27], "z": [-0.17, 0.17]}},
                            {"rect": {"r": [0.07, 0.25], "z": [-0.15, 0.15]}}]}
    d = af.Design({"schema": "axionfem.design/1", "name": "annular",
                   "materials": {"wall_cu": {"kind": "conductor"}},
                   "regions": [{"name": "cavity", "shape": {"rect": {"r": [ai, bo], "z": [-Lc / 2, Lc / 2]}},
                                "material": "vacuum", "mesh_size": 0.005},
                               {"name": "wall", "shape": wall, "material": "wall_cu"},
                               {"name": "coil", "shape": frame, "material": "copper"}],
                   "windings": [{"name": "tor", "region": "coil", "type": "poloidal", "ampere_turns": 1e5}],
                   "fullwave": {"modes": 2, "families": ["TE", "TM"], "boundary": "pec"},
                   "cavity": {"interior_point": [0.15, 0.0]},
                   "mesh": {"order": 2, "size": 0.03, "surface_size": 0.005, "domain_radius": 0.6}})
    fw = d.solve(tmp_path / "tor", verbose=0).data["fullwave"]
    te = _by_freq(fw["eigenmodes"]["TE"]["modes"], f_ref)
    assert te["f_Hz"] == pytest.approx(f_ref, rel=1e-6)
    assert te["form_factor"]["C"] == pytest.approx(C_ref, rel=1e-5)  # 7e-7 at h = 5 mm
    assert te["form_factor"]["C"] == pytest.approx(te["form_factor"]["C_rms"], rel=1e-12)
    assert C_ref > 0.5
    for m in fw["eigenmodes"]["TM"]["modes"]:  # TM couples to (B_r, B_z) only: zero in a purely toroidal B0
        assert m["form_factor"]["C"] == 0.0


def test_lossless_walls_flagged(tmp_path):
    """Walls without rf data next to walls with rf data are listed and warned about."""
    d = cv.pillbox_design(A, L, h=0.01, modes=1, families=("TM",), wall_material={
        "kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": RHO, "T_K": 4.0}})
    d.doc["materials"]["pec"] = {"kind": "conductor"}
    caps = d.doc["regions"][1]["shape"]["union"]
    d.doc["regions"][1]["shape"] = caps[0]
    d.doc["regions"].append({"name": "caps", "shape": {"union": caps[1:]}, "material": "pec"})
    r = d.solve(tmp_path / "ll", verbose=0)
    lo = r.data["fullwave"]["eigenmodes"]["TM"]["modes"][0]["losses"]
    assert lo["lossless_walls"] == ["caps"] and "Q" not in lo["walls"]["caps"]
    assert any("caps have no rf data" in w for w in r.data["warnings"])


def test_lossless_walls_consistent_with_3d(tmp_path):
    """cavity.allow_lossless_walls as in 3D and xsec: the builder default (lossless wall) is flagged by request and
    reports Q0 = Q_dielectric; without the flag the wall is warned and Q0 is omitted (Q_dielectric kept)."""
    d = cv.pillbox_design(A, L, h=0.02, modes=1, families=("TM",),
                          fill={"kind": "linear", "eps_r": 2.0, "tan_delta": 1e-4})
    assert d.doc["cavity"]["allow_lossless_walls"] is True
    r = d.solve(tmp_path / "a", verbose=0)
    lo = r.data["fullwave"]["eigenmodes"]["TM"]["modes"][0]["losses"]
    assert lo["walls_lossless_by_request"] is True and lo["lossless_walls"] == ["wall"]
    assert lo["Q0"] == pytest.approx(lo["Q_dielectric"], rel=1e-12) and "Q_wall" not in lo
    assert not any("rf data" in w for w in r.data["warnings"])
    d.doc["cavity"].pop("allow_lossless_walls")
    r = d.solve(tmp_path / "b", verbose=0)
    lo = r.data["fullwave"]["eigenmodes"]["TM"]["modes"][0]["losses"]
    assert "Q0" not in lo and lo["Q_dielectric"] > 0 and lo["lossless_walls"] == ["wall"]
    assert any("Q0 is not reported" in w for w in r.data["warnings"])
    assert any("wall have no rf data" in w for w in r.data["warnings"])
