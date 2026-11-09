"""Per-mode fields of the axisymmetric cavity eigenmodes (Results.cavity_mode_field / _sample / _probe,
save_cavity_modes, mode_scene, plot_mode; part of the beta cavity module).

Analytic references (pillbox radius a, length L, centred on z = 0; normalization int |E|^2 dV = 1):

- TM010: E_z = E0 J0(chi01 r/a), H_phi = (E0/Z0) J1(chi01 r/a), E0^2 = 1/(pi a^2 L J1(chi01)^2);
- TE011: E_phi = E0 J1(kc r) sin(pi (z + L/2)/L), kc = chi'01/a, E0^2 = 2/(pi a^2 L J0(chi'01)^2),
  H_r = -(E0 beta/(omega mu0)) J1(kc r) cos(.), H_z = (E0 kc/(omega mu0)) J0(kc r) sin(.), beta = pi/L.
The exported arrays E, H are real amplitudes (complex fields E_c = i E, H_c = H) with the largest E component positive at max |E|.
"""
import json
import math

import numpy as np
import pytest

import axionfem as af
from axionfem import cavity as cv

j0 = pytest.importorskip("scipy.special").j0
from scipy.special import j1, roots_legendre, y0, y1  # noqa: E402

C = 299792458.0
MU0 = 1.25663706212e-6
Z0 = MU0 * C
A, L = 0.1, 0.2
CHI01, CHI1P = 2.404825557695773, 3.831705970207512


@pytest.fixture(scope="module")
def pillbox(tmp_path_factory):
    d = cv.pillbox_design(A, L, h=0.004, modes=4)
    return d.solve(tmp_path_factory.mktemp("modefields"), verbose=0)


def _row(res, fam, idx):
    return next(r for r in res.cavity_modes() if r["family"] == fam and r["index"] == idx)


def _gauss_rect(r0, r1, z0, z1, n=48):
    """Tensor Gauss-Legendre points and weights (2 pi r dr dz) on a rectangle of the meridional plane."""
    x, w = roots_legendre(n)
    r = 0.5 * (r1 - r0) * (x + 1) + r0
    z = 0.5 * (z1 - z0) * (x + 1) + z0
    R, Z = np.meshgrid(r, z)
    W = np.outer(w, w) * 0.25 * (r1 - r0) * (z1 - z0) * 2 * np.pi * R
    return R, Z, W


def _tm010_err(res):
    k = CHI01 / A
    E0 = 1 / math.sqrt(math.pi * A * A * L * j1(CHI01) ** 2)
    P = res.cavity_mode_probe(_row(res, "TM", 0), (0.0, 0.995 * A, 41), (-0.098, 0.098, 21))
    eEz = np.max(np.abs(P["E_z"] - E0 * j0(k * P["R"]))) / E0
    eEr = np.max(np.abs(P["E_r"])) / E0
    eH = np.max(np.abs(P["H_phi"] - E0 / Z0 * j1(k * P["R"]))) / (E0 / Z0)
    return eEz, eEr, eH, P


def test_tm010_matches_J0(pillbox):
    eEz, eEr, eH, P = _tm010_err(pillbox)
    assert eEz < 2e-3 and eEr < 2e-3  # E = curl(H_phi)/(omega eps): one derivative of the P2 unknown
    assert eH < 1e-4                  # H_phi is the unknown itself
    for k in ("E_phi", "H_r", "H_z"):
        assert np.all(P[k] == 0.0)
    assert np.all(P["eps_r"] == 1.0) and np.all(P["mu_r"] == pytest.approx(1.0, rel=1e-12))


def test_te011_matches_analytic(pillbox):
    row = _row(pillbox, "TE", 0)
    kc, beta = CHI1P / A, math.pi / L
    w = 2 * math.pi * row["f_Hz"]
    E0 = math.sqrt(2 / (math.pi * A * A * L * j0(CHI1P) ** 2))
    P = pillbox.cavity_mode_probe(row, (0.0, 0.995 * A, 41), (-0.098, 0.098, 21))
    s, c = np.sin(beta * (P["Z"] + L / 2)), np.cos(beta * (P["Z"] + L / 2))
    Hs = E0 / (w * MU0)
    assert np.max(np.abs(P["E_phi"] - E0 * j1(kc * P["R"]) * s)) / E0 < 1e-4
    assert np.max(np.abs(P["H_r"] + Hs * beta * j1(kc * P["R"]) * c)) / (Hs * kc) < 3e-3
    assert np.max(np.abs(P["H_z"] - Hs * kc * j0(kc * P["R"]) * s)) / (Hs * kc) < 3e-3
    for k in ("E_r", "E_z", "H_phi"):
        assert np.all(P[k] == 0.0)


def test_pointwise_convergence_order(tmp_path):
    """Order-2 elements: E of TM010 (a derivative of H_phi) converges as h^2 pointwise, H_phi faster."""
    errs = []
    for h in (0.02, 0.01):
        d = cv.pillbox_design(A, L, h=h, modes=1, families=("TM",), domain_radius=2.0)
        errs.append(_tm010_err(d.solve(tmp_path / f"h{h}", verbose=0)))
    rate_E = math.log2(errs[0][0] / errs[1][0])
    rate_H = math.log2(errs[0][2] / errs[1][2])
    assert 1.6 < rate_E < 3.5, errs
    assert rate_H > 2.5, errs


def _element_integrals(res, row, n=4):
    """int eps_r |E|^2 dV - 1 and Z0^2 int mu_r |H|^2 dV - 1 with a quadrature on each cavity element (collapsed
    Gauss, n^2 points per triangle; straight-sided elements). Both identities are exact for the discrete eigenpair
    (the normalization by construction, the energy balance as its Rayleigh quotient), so element-wise quadrature
    reproduces them to rounding, unlike a global rule across the kinks of the fields at element edges. Also returns
    the samples, their weights (2 pi r dA) and whether every point was found in its own element."""
    x, w = roots_legendre(n)
    x, w = 0.5 * (x + 1), 0.5 * w
    U, V = np.meshgrid(x, x)
    xi, eta, wq = U.ravel(), (V * (1 - U)).ravel(), (np.outer(w, w) * (1 - U)).ravel()
    F = res.cavity_mode_field(row, subdiv=1)  # 3 vertices per element
    P = np.column_stack([F["r"], F["z"]]).reshape(-1, 3, 2)
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    J = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    R = a[:, None, 0] + xi * (b[:, None, 0] - a[:, None, 0]) + eta * (c[:, None, 0] - a[:, None, 0])
    Z = a[:, None, 1] + xi * (b[:, None, 1] - a[:, None, 1]) + eta * (c[:, None, 1] - a[:, None, 1])
    S = res.cavity_mode_sample(row, R, Z)
    W = J[:, None] * wq * 2 * np.pi * R
    E2 = S["E_r"] ** 2 + S["E_phi"] ** 2 + S["E_z"] ** 2
    H2 = S["H_r"] ** 2 + S["H_phi"] ** 2 + S["H_z"] ** 2
    own = bool(np.all(S["elem"] == F["elem"][::3][:, None]))
    return np.sum(W * S["eps_r"] * E2) - 1, np.sum(W * S["mu_r"] * H2) * Z0 ** 2 - 1, S, W, own


@pytest.mark.parametrize("fam,idx", [("TM", 0), ("TM", 1), ("TE", 0), ("TE", 1)])
def test_normalization_and_energy_balance(pillbox, fam, idx):
    """int eps_r |E|^2 dV = 1 by construction; int mu_r |H|^2 dV = 1/Z0^2 holds only if E, H and omega belong to
    the same eigenpair with consistent omega/eps/mu factors (equal electric and magnetic energy). Element-wise
    quadrature: both to rounding. A global tensor Gauss rule (kinks at element edges) to its own error. The form
    factor C recomputed from the exported E_z (uniform B0 = 1 T axial) matches the solver's value (this also checks
    that the row of cavity_modes() and the exported eigenvector are the same mode)."""
    row = _row(pillbox, fam, idx)
    en, eh, _, _, own = _element_integrals(pillbox, row)
    assert own
    assert abs(en) < 1e-10 and abs(eh) < 1e-9
    R, Z, W = _gauss_rect(0.0, A, -L / 2, L / 2)
    S = pillbox.cavity_mode_sample(row, R, Z)
    E2 = S["E_r"] ** 2 + S["E_phi"] ** 2 + S["E_z"] ** 2
    H2 = S["H_r"] ** 2 + S["H_phi"] ** 2 + S["H_z"] ** 2
    assert np.sum(W * S["eps_r"] * E2) == pytest.approx(1.0, rel=2e-4, abs=0)
    assert np.sum(W * S["mu_r"] * H2) * Z0 ** 2 == pytest.approx(1.0, rel=2e-4, abs=0)
    C = np.sum(W * S["E_z"]) ** 2 / (math.pi * A * A * L)
    if row["C"] > 1e-6:
        assert C == pytest.approx(row["C"], rel=1e-4, abs=0)
    else:
        assert C < 1e-8


def test_dielectric_sleeve_fields(tmp_path):
    """Fields inside a dielectric (lossy sleeve eps_r = 4, tan delta = 1e-3, r in [0.03, 0.05]): eps_r of the
    exported points, the eps-weighted normalization and the energy balance (element-wise, to rounding), the
    electric-energy fraction in the sleeve and C from the exported E_z against the independent 1D Bessel matching
    (cavity.sleeve_tm_mode), and continuity of the tangential E_z and of H_phi across the interfaces."""
    r1, r2, eps = 0.03, 0.05, 4.0
    ref = cv.sleeve_tm_mode(A, r1, r2, eps)
    d = cv.pillbox_design(A, L, h=0.004, modes=2, families=("TM",))
    d.doc["materials"]["sleeve"] = {"kind": "linear", "eps_r": eps, "tan_delta": 1e-3}
    d.doc["regions"].insert(1, {"name": "sleeve", "shape": {"rect": {"r": [r1, r2], "z": [-L / 2, L / 2]}},
                                "material": "sleeve", "mesh_size": 0.004})
    res = d.solve(tmp_path / "sleeve", verbose=0)
    row = min(res.cavity_modes(), key=lambda r: abs(r["f_Hz"] - ref["f_Hz"]))
    assert row["f_Hz"] == pytest.approx(ref["f_Hz"], rel=1e-6, abs=0)
    en, eh, S, W, own = _element_integrals(res, row)
    assert own and abs(en) < 1e-10 and abs(eh) < 1e-9
    sl = (S["r"] > r1) & (S["r"] < r2)
    assert np.all(S["eps_r"][sl] == eps) and np.all(S["eps_r"][~sl] == 1.0)
    E2 = S["E_r"] ** 2 + S["E_z"] ** 2
    assert np.sum((W * S["eps_r"] * E2)[sl]) == pytest.approx(ref["fe_sleeve"], rel=1e-5, abs=0)
    assert np.sum(W * S["E_z"]) ** 2 / (math.pi * A * A * L) == pytest.approx(ref["C"], rel=1e-5, abs=0)
    P = res.cavity_mode_sample(row, [r1 - 1e-7, r1 + 1e-7, r2 - 1e-7, r2 + 1e-7], [0.01] * 4)
    assert list(P["eps_r"]) == [1.0, eps, eps, 1.0]
    for k, tol in (("E_z", 3e-3), ("H_phi", 1e-4)):
        sc = np.max(np.abs(S[k]))
        assert abs(P[k][0] - P[k][1]) < tol * sc and abs(P[k][2] - P[k][3]) < tol * sc, (k, P[k])


def test_two_sided_sheet_fields(tmp_path):
    """Perfectly conducting annular iris (zero-thickness sheet, z = 0, r in [a/2, a]) with the cavity on both
    sides: TM uses split DOFs there. Every mode keeps the normalization and the energy balance (element-wise); H_phi
    of the TM mode with E_r at z = 0 (TM011-like) is odd across the sheet (surface current), that of TM010 continuous;
    the normal H_z of TE modes vanishes on the sheet."""
    d = cv.pillbox_design(A, L, h=0.004, modes=2)
    d.doc["curves"] = [{"name": "iris", "path": {"segment": [[A / 2, 0.0], [A, 0.0]]}, "material": "wall_cu",
                        "mesh_size": 0.004}]
    d.doc["cavity"]["interior_point"] = [0.02, 0.05]
    res = d.solve(tmp_path / "iris", verbose=0)
    assert res.data["fullwave"]["cavity"]["TM_split_dofs"] > 0
    for row in res.cavity_modes():
        en, eh, S, _, own = _element_integrals(res, row)
        assert own and abs(en) < 1e-10 and abs(eh) < 1e-9, (row["family"], row["index"], en, eh)
        P = res.cavity_mode_sample(row, [0.075, 0.075], [1e-9, -1e-9])
        if row["family"] == "TE":
            assert np.max(np.abs(P["H_z"])) < 1e-5 * np.max(np.abs(S["H_z"]))
    tm0, tm1 = _row(res, "TM", 0), _row(res, "TM", 1)
    P0 = res.cavity_mode_sample(tm0, [0.075, 0.075], [1e-9, -1e-9])
    P1 = res.cavity_mode_sample(tm1, [0.075, 0.075], [1e-9, -1e-9])
    assert P0["H_phi"][0] == pytest.approx(P0["H_phi"][1], rel=1e-6, abs=0)
    assert abs(P1["H_phi"][0]) > 0.3 * np.max(np.abs(res.cavity_mode_field(tm1)["H_phi"]))
    assert P1["H_phi"][0] == pytest.approx(-P1["H_phi"][1], rel=1e-5, abs=0)


def test_fields_near_the_axis(pillbox):
    """r -> 0: the u/r term of curl(u phi^) is replaced by du/dr at r = 0 exactly; the fields are continuous into
    the axis (E_r, H_phi, E_phi, H_r -> 0 linearly; E_z, H_z finite)."""
    r = np.array([0.0, 1e-12, 1e-9, 1e-6])
    for fam, idx in (("TM", 0), ("TE", 0)):
        S = pillbox.cavity_mode_sample(_row(pillbox, fam, idx), r, np.full(4, 0.03))
        main = "E_z" if fam == "TM" else "H_z"
        assert np.all(np.isfinite(S[main])) and S[main][0] != 0
        assert np.max(np.abs(S[main] - S[main][0])) < 1e-5 * abs(S[main][0])
        for k in (("E_r", "H_phi") if fam == "TM" else ("E_phi", "H_r")):
            assert S[k][0] == 0 and np.all(np.abs(S[k][1:]) <= 1e3 * r[1:] * np.max(np.abs(S[main])) / A)


def test_index_mapping(pillbox):
    rows = pillbox.cavity_modes()
    assert len(rows) == 8
    for i, row in enumerate(rows):
        F = pillbox.cavity_mode_field(i, subdiv=1)
        G = pillbox.cavity_mode_field(row["index"], family=row["family"], subdiv=1)
        assert F["family"] == row["family"] and F["index"] == row["index"]
        assert F["f_Hz"] == row["f_Hz"] == F["mode"]["f_Hz"]
        assert np.array_equal(F["E_z"], G["E_z"]) and np.array_equal(F["E_phi"], G["E_phi"])
        zero = ("E_phi", "H_r", "H_z") if row["family"] == "TM" else ("E_r", "E_z", "H_phi")
        for k in zero:
            assert not np.any(F[k])
        assert F["tris"].shape == (len(F["r"]) // 3, 3)  # subdiv 1: 3 points and 1 triangle per element
        assert np.all(F["elem"] >= 0)
    with pytest.raises(IndexError):
        pillbox.cavity_mode_field(8)
    with pytest.raises(IndexError, match="TE"):
        pillbox.cavity_mode_field(7, family="TE")
    with pytest.raises(af.AxionFEMError, match="family 'TX'"):
        pillbox._model.cavity_mode_mesh("TX", 0, 2)


def test_sample_outside_cavity_is_nan(pillbox):
    S = pillbox.cavity_mode_sample(0, [0.105, 0.05, 0.3], [0.0, 0.0, 0.0])  # wall, cavity, outside the wall
    assert np.isnan(S["E_z"][0]) and np.isnan(S["E_z"][2]) and np.isfinite(S["E_z"][1])
    assert list(S["elem"][[0, 2]]) == [-1, -1]
    T = pillbox.cavity_mode_sample(0, 0.05, [0.0, 0.01])  # scalar r broadcast against z
    assert T["E_z"].shape == (2,) and T["E_z"][0] == S["E_z"][1]


def test_save_and_plot(pillbox, tmp_path):
    paths = pillbox.save_cavity_modes(tmp_path / "modes", modes=[0, 3])
    assert [p.name for p in paths] == ["TM_0.npz", "TE_0.npz"]
    z = np.load(paths[0])
    meta = json.loads(str(z["meta"]))
    assert meta["row"]["family"] == "TM" and "eps_r |E|^2" in meta["normalization"]
    F = pillbox.cavity_mode_field(0)
    assert np.array_equal(z["E_z"], F["E_z"]) and z["tris"].shape == F["tris"].shape
    p = pillbox.png(tmp_path / "m.png", mode=0, width=900, height=600)
    assert p.read_bytes()[:4] == b"\x89PNG"
    p = pillbox.plot_mode(3, tmp_path / "h.png", quantity="Hmag", annotate=False, width=700, height=500)
    assert p.read_bytes()[:4] == b"\x89PNG"
    sc = pillbox.mode_scene(1, contours=8)
    assert sc["field_lines"]["static"]["shape"][0] > 0 and "E lines" in sc["field_lines"]["static_label"]
    assert sc["summary"]["cavity_mode"]["f_Hz"] == pillbox.cavity_modes()[1]["f_Hz"]


def test_coax_mode_shape(tmp_path):
    """Coaxial cavity (rod a = 0.05 spanning the caps, wall b = 0.2): E_z of the lowest TM mode is
    J0(kr) Y0(ka) - Y0(kr) J0(ka) (z independent); the rod (r < a) is outside the cavity."""
    a, b = 0.05, 0.2
    res = cv.coax_design(a, b, L, h=0.006, modes=3).solve(tmp_path / "coax", verbose=0)
    ref = cv.coax_tm_mode(a, b)
    k = ref["k"]
    # the lowest modes are the coaxial-line (TEM, f = p c / 2L) resonances; select by frequency
    row = min(res.cavity_modes(), key=lambda r: abs(r["f_Hz"] - ref["f_Hz"]))
    assert row["f_Hz"] == pytest.approx(ref["f_Hz"], rel=1e-5) and row["C"] == pytest.approx(ref["C"], rel=1e-4)
    P = res.cavity_mode_probe(row, (0.0, 0.999 * b, 81), (-0.09, 0.09, 7))
    inside = P["R"] > 1.001 * a
    assert np.all(np.isnan(P["E_z"][P["R"] < 0.999 * a]))
    ex = j0(k * P["R"]) * y0(k * a) - y0(k * P["R"]) * j0(k * a)
    s = np.sum(P["E_z"][inside] * ex[inside]) / np.sum(ex[inside] ** 2)
    assert np.max(np.abs(P["E_z"][inside] - s * ex[inside])) / np.max(np.abs(P["E_z"][inside])) < 6e-3  # h = 6 mm
    hp = j1(k * P["R"]) * y0(k * a) - y1(k * P["R"]) * j0(k * a)  # H_phi ~ -dE_z/dr / k = this up to sign
    t = np.sum(P["H_phi"][inside] * hp[inside]) / np.sum(hp[inside] ** 2)
    assert np.max(np.abs(P["H_phi"][inside] - t * hp[inside])) / np.max(np.abs(P["H_phi"][inside])) < 1e-3
