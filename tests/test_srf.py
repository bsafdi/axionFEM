"""SRF heterodyne cavities (axionfem.srf; beta, docs/srf/README.md).

Fast: elliptical-cell geometry (V ~ 17 L, A ~ 6.2 V^(2/3)), closed-cylinder pair closed forms (TM020/TE011
xi ~ 7.5 at L/R ~ 0.79, eta ~ 0.44; TM030/TE021 xi ~ 10.8 at L/R ~ 0.62; TE012/TM013 xi ~ 9.1 at L/R ~ 2.35), one
2D cylinder solve vs the closed forms and one coarse 2D solve of the default cell.
Slow (AXF_SLOW=1): degeneracy tuner, 3D cross-check of the cell.
"""
import math
import os

import pytest

import numpy as np

import axionfem as af
from axionfem import cavity as cv
from axionfem import srf

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="SRF tuner / 3D cell (slow); set AXF_SLOW=1")
needs_strumpack = pytest.mark.skipif(not af.build_info().get("strumpack"),
                                     reason="the 3D shift-invert eigensolver needs a STRUMPACK build")


# ------------------------------------------------------------------------------------------------ geometry
def test_cell_contour_volume_and_area():
    c = srf.elliptical_cell_contour(axial_factor=0.86, ds=2e-4)
    assert c["V_m3"] == pytest.approx(0.01706, rel=2e-3)          # V ~ 17 L
    assert c["A_m2"] / c["V_m3"] ** (2 / 3) == pytest.approx(6.18, abs=0.01)
    assert c["length"] == pytest.approx(2 * (0.86 * 0.1038 + 0.1), rel=1e-12)
    W = c["wall"]
    assert W[0] == (0.0, -c["z_end"]) and W[-1] == (0.0, c["z_end"])
    assert max(p[0] for p in W) == pytest.approx(0.2003, rel=1e-12)
    # tangent points lie on both ellipses, between iris and equator
    (z1, r1), (z2, r2) = c["params"]["tangent_points_zr"]
    L = c["params"]["half_length"]
    assert ((z1 + L) / (0.86 * 0.018)) ** 2 + ((r1 - 0.088) / 0.038) ** 2 == pytest.approx(1, abs=1e-12)
    assert ((z2) / (0.86 * 0.0825)) ** 2 + ((r2 - 0.1163) / 0.084) ** 2 == pytest.approx(1, abs=1e-12)
    assert 0 < c["tangent_angle_deg"] < 15
    # the segment between them is tangent to both ellipses
    import numpy as np

    d = np.array([z2 - z1, r2 - r1])
    t1 = np.array([0.86 * 0.018 * (r1 - 0.088) / 0.038, -0.038 * (z1 + L) / (0.86 * 0.018)])  # grad-perp, iris
    t2 = np.array([0.86 * 0.0825 * (r2 - 0.1163) / 0.084, -0.084 * z2 / (0.86 * 0.0825)])
    for t in (t1, t2):
        assert abs(t[0] * d[1] - t[1] * d[0]) / (np.linalg.norm(t) * np.linalg.norm(d)) < 1e-9


def test_cell_contour_scales_uniformly():
    c1 = srf.elliptical_cell_contour(axial_factor=0.86, ds=5e-4)
    c2 = srf.elliptical_cell_contour(axial_factor=0.86, ds=5e-4, scale=24.5)
    assert c2["V_m3"] == pytest.approx(24.5 ** 3 * c1["V_m3"], rel=1e-9)
    assert c2["A_m2"] / c2["V_m3"] ** (2 / 3) == pytest.approx(c1["A_m2"] / c1["V_m3"] ** (2 / 3), rel=1e-9)
    # scaled by 24.5: V ~ 250 m^3, equator radius 4.9 m, length 9.2 m
    assert c2["V_m3"] == pytest.approx(250, rel=0.01)
    assert c2["equator_radius"] == pytest.approx(4.9, rel=0.01)
    assert c2["length"] == pytest.approx(9.2, rel=0.01)


def test_cell_rejects_overlapping_ellipses():
    bad = dict(srf.PROJECTX_650_B09, iris_axes=(0.09, 0.038))
    with pytest.raises(ValueError, match="common tangent"):
        srf.elliptical_cell_contour(bad, axial_factor=1.0)


# ------------------------------------------------------------------------------------- closed cylinders
@pytest.mark.parametrize("pair,LR,xi", [("TM020/TE011", 0.7906, 7.476), ("TM030/TE021", 0.6201, 10.808),
                                        ("TE012/TM013", 2.3549, 9.098)])
def test_cylinder_pair_degeneracy(pair, LR, xi):
    g = srf.cylinder_pair_geometry(*srf.CYLINDER_PAIRS[pair])
    assert g["L_over_R"] == pytest.approx(LR, abs=2e-4)
    assert g["xi"] == pytest.approx(xi, abs=2e-3)


def test_cylinder_pair_closed_forms():
    o = srf.cylinder_pair("TM020/TE011", n_wall=1501)
    assert o["eta_10"] == pytest.approx(0.43697, abs=1e-4)             # Thomson et al., arXiv:2301.06778: 0.44
    # G against the independent closed forms of axionfem.cavity.pillbox_modes (Pozar Sec. 6.4)
    a, L = 1.0, o["L_over_R"]
    ms = cv.pillbox_modes(a, L, 1.01 * cv.C_LIGHT * o["pump"]["kR"] / (2 * math.pi))
    g = {(m["family"], m["n"], m["p"]): m["G_ohm"] for m in ms}
    assert o["G_pump_ohm"] == pytest.approx(g[("TM", 2, 0)], rel=1e-5)
    assert o["G_signal_ohm"] == pytest.approx(g[("TE", 1, 1)], rel=1e-5)
    # TM0n0: B_pk = max|J1| on the caps, B_rms = |J1(chi_0n)|; E_pk = J0(0) on the caps
    from scipy.special import j1

    assert o["Bpk_over_B0"] == pytest.approx(0.581865 / abs(j1(5.520078110286311)), rel=1e-4)
    assert o["Epk_over_cB0"] == pytest.approx(1 / abs(j1(5.520078110286311)), rel=1e-4)


def test_eta_closed_forms_independent_derivation():
    """eta_10 of the closed-cylinder pairs against closed forms from an independent analytic derivation."""
    from scipy.special import jn_zeros

    a1, a2, a3 = jn_zeros(0, 3)
    b1, b2 = jn_zeros(1, 2)
    ref = {"TM020/TE011": 4 * math.sqrt(2) * b1 / (math.pi * (a2 ** 2 - b1 ** 2)),
           "TM030/TE021": 4 * math.sqrt(2) * b2 / (math.pi * (a3 ** 2 - b2 ** 2)),
           "TE012/TM013": 16 * b1 / (5 * math.pi * (b1 ** 2 - a1 ** 2))}
    for pair, eta in ref.items():
        pump, signal = srf.CYLINDER_PAIRS[pair]
        L = srf.cylinder_pair_geometry(pump, signal)["L_over_R"]
        assert srf.eta_cylinder(1.0, L, pump, signal) == pytest.approx(eta, rel=1e-10)


def test_eta_cylinder_matches_existing_tm010_te01p():
    # eta_tm010_te01p: E of TM010 (the signal here) against B of TE01p (the pump here)
    for p in (1, 2, 3):
        assert srf.eta_cylinder(0.1, 0.23, ("TE", 1, p), ("TM", 1, 0)) == pytest.approx(
            cv.eta_tm010_te01p(0.1, 0.23, p), rel=1e-10, abs=1e-14)
    assert srf.eta_cylinder(0.1, 0.2, ("TM", 1, 0), ("TM", 2, 0)) == 0.0


@pytest.mark.parametrize("pair", list(srf.CYLINDER_PAIRS))
def test_eta_reciprocity_at_degeneracy(pair):
    """int E1.curl E0 = int E0.curl E1 (PEC walls), so eta_10 = (omega_1/omega_0) eta_01: equal at degeneracy,
    different away from it."""
    pump, signal = srf.CYLINDER_PAIRS[pair]
    L = srf.cylinder_pair_geometry(pump, signal)["L_over_R"]
    assert srf.eta_cylinder(1.0, L, pump, signal) == pytest.approx(srf.eta_cylinder(1.0, L, signal, pump), rel=1e-9)
    k0, *_ = srf.cylinder_mode_fields(*pump, 1.0, 1.1 * L)
    k1, *_ = srf.cylinder_mode_fields(*signal, 1.0, 1.1 * L)
    assert srf.eta_cylinder(1.0, 1.1 * L, pump, signal) == pytest.approx(
        k1 / k0 * srf.eta_cylinder(1.0, 1.1 * L, signal, pump), rel=1e-9)


@pytest.mark.parametrize("pair", ["TM020/TE011", "TE012/TM013"])
def test_cylinder_pair_2d_matches_closed_forms(tmp_path, pair):
    """TE012/TM013 puts the pump's B_pk on the caps (r/R = 0.4805, 1.1675 B_rms vs 1.1606 on the barrel): checks the
    cap/wall detection; its E_pk on the wall is 0. Order 3: the wall peaks are taken at the Gauss points of the wall
    faces, and the TM020 cap maximum (on the axis) is the derivative of H_phi there (order 2 at this h: +0.5 %)."""
    pu, si = pair.split("/")
    d = srf.cylinder_pair_design(pair, 0.1, h=0.005, order=3, modes=8)
    r = d.solve(tmp_path, verbose=0)
    o = srf.figures_of_merit(r, pu, si)
    a = srf.cylinder_pair(pair, n_wall=1501, n_vol=601)
    assert abs(o["splitting_Hz"]) / o["f_pump_Hz"] < 1e-5
    assert o["xi_pump"] == pytest.approx(a["xi"], rel=1e-5)
    assert o["eta_10"] == pytest.approx(a["eta_10"], rel=1e-3)
    assert o["Bpk_over_B0"] == pytest.approx(a["Bpk_over_B0"], rel=3e-3)
    assert o["Epk_over_cB0"] == pytest.approx(a["Epk_over_cB0"], rel=5e-3, abs=1e-3)
    assert o["signal_peaks"]["Bpk_over_B0"] == pytest.approx(a["signal"]["Bpk_over_B0"], rel=3e-3)
    assert o["signal_peaks"]["Epk_over_cB0"] == pytest.approx(a["signal"]["Epk_over_cB0"], rel=5e-3, abs=1e-3)
    assert abs(o["checks"]["volume_contour_vs_solver"]) < 1e-9 and o["checks"]["n_wall_points"] > 100
    if pu == "TE012":  # the cap maximum, not the barrel one
        z = o["pump_peaks"]["Bpk_at_rz"]
        assert abs(abs(z[1]) - d.srf_contour["z_end"]) < 1e-9 and z[0] / 0.1 == pytest.approx(0.4805, abs=0.02)
    assert o["G_pump_ohm"] == pytest.approx(a["G_pump_ohm"], rel=1e-4)
    assert o["G_signal_ohm"] == pytest.approx(a["G_signal_ohm"], rel=1e-4)
    assert o["A_over_V23"] == pytest.approx(a["A_over_V23"], rel=1e-12)


# ------------------------------------------------------------------------------------------ the default cell
def test_cell_2d_coarse(tmp_path):
    """Default (coarse) mesh: the values of docs/srf/README.md to the accuracy of that mesh."""
    d = srf.elliptical_cell_design(axial_factor=0.86, modes=6)
    r = d.solve(tmp_path, verbose=0)
    lab = {m["name"]: m for m in srf.classify_modes(r, d.srf_contour)}
    assert {"TM010", "TM011", "TM020", "TE011"} <= set(lab)
    assert lab["TM010"]["f_Hz"] == pytest.approx(0.6496e9, rel=2e-3)
    o = srf.figures_of_merit(r)
    assert o["f_pump_Hz"] == pytest.approx(1.4166e9, rel=5e-4)
    assert abs(o["splitting_Hz"]) < 0.5e6
    assert o["xi_pump"] == pytest.approx(7.64, abs=0.02)
    assert o["eta_10"] == pytest.approx(0.449, abs=0.003)
    assert o["G_pump_ohm"] == pytest.approx(524.5, rel=5e-3)
    assert o["G_signal_ohm"] == pytest.approx(604.5, rel=5e-3)
    assert o["Bpk_over_B0"] == pytest.approx(1.46, abs=0.02)
    assert o["Epk_over_cB0"] == pytest.approx(1.85, abs=0.03)  # converged ~1.845 (docs/srf/README.md)
    assert o["A_over_V23"] == pytest.approx(6.18, abs=0.01)
    for k, v in o["checks"].items():
        if "expected" in k:
            assert v == pytest.approx(1, abs=2e-3), k
    # smooth wall: the curved (second-order) elements reproduce the exact volume to O(h^4)
    assert abs(o["checks"]["volume_contour_vs_solver"]) < 1e-6 and o["checks"]["n_wall_points"] > 100


# reference values of the smooth-wall 2D convergence (benchmarks/srf/records/srf_2d_smooth.json: order 2 at R_eq/160,
# order 3 at R_eq/80 agree to 2e-4); wall peaks at the Gauss points of the wall faces (the 3D definition)
EPK_2D, BPK_2D = 1.8430, 1.46401


def test_contour_shape_is_exact_and_validated():
    """The ``contour`` shape (lines + elliptical arcs) of the cell: its arcs pass through the tangent points, the
    exact V/A differ from the polygon's by the faceting only, and malformed contours are spec errors."""
    import numpy as np

    c = srf.elliptical_cell_contour(axial_factor=0.86)
    P = srf.contour_points(c["shape"], n_arc=50)
    (z1, r1), (z2, r2) = c["params"]["tangent_points_zr"]
    for q in ((r1, z1), (r2, z2), (r1, -z1), (r2, -z2)):
        assert np.min(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1])) < 1e-12
    assert np.max(P[:, 0]) == pytest.approx(0.2003, rel=1e-12)
    assert c["V_polygon_m3"] / c["V_m3"] - 1 == pytest.approx(0, abs=1e-3) and c["V_polygon_m3"] != c["V_m3"]
    d = srf.elliptical_cell_design(axial_factor=0.86, modes=4)
    assert d.doc["regions"][1]["shape"] == c["shape"]
    d.validate()
    k = next(i for i, sg in enumerate(c["shape"]["contour"]["segments"]) if "ellipse" in sg)
    bad = json_copy(d.doc)
    bad["regions"][1]["shape"]["contour"]["segments"][k]["ellipse"]["t"][0] += 0.01   # arc not joined
    with pytest.raises(Exception, match="must start where the previous one ends"):
        af.Design(bad).validate()
    bad = json_copy(d.doc)
    bad["regions"][1]["shape"]["contour"]["segments"][k]["ellipse"]["axes"][0] = -1.0
    with pytest.raises(Exception, match="semi-axes must be > 0"):
        af.Design(bad).validate()


def _sphere_design(a, ccw=True, order=3):
    from axionfem.cavity import _design
    seg = {"ellipse": {"center": [0.0, 0.0], "axes": [a, a], "t": [-math.pi / 2, math.pi / 2] if ccw
                       else [math.pi / 2, -math.pi / 2]}}
    shape = {"contour": {"start": [0.0, -a if ccw else a], "segments": [seg, {"line": [0.0, -a if ccw else a]}]}}
    regions = [{"name": "wall", "shape": {"rect": {"r": [0.0, 1.1 * a], "z": [-1.1 * a, 1.1 * a]}}, "material": "wall_cu"},
               {"name": "cavity", "shape": shape, "material": "vacuum", "mesh_size": a / 8, "surface_size": a / 16}]
    f = 2.743707269992269 * cv.C_LIGHT / (2 * math.pi * a)    # TM101: first zero of (x j1(x))'
    return _design("sphere", regions, wall_material=None, fill=None, h=a / 8, order=order, Bz=1.0, modes=4,
                   families=("TM",), target_f_Hz=f, interior_point=(0.5 * a, 0.0), domain_radius=3 * a,
                   surface_size=a / 16), f


@pytest.mark.parametrize("ccw", [True, False])
def test_contour_sphere_exact_volume_and_mode(tmp_path, ccw):
    """A contour made of one elliptical arc over pi (two B-spline spans), run in either direction: the meshed
    volume is the sphere's to 1e-7 and the TM101 frequency is the closed form to 1e-6 (the arc is exact, the curved
    elements second order)."""
    a = 0.1
    d, f = _sphere_design(a, ccw)
    r = d.solve(tmp_path, verbose=0)
    assert r.fullwave["cavity"]["volume_m3"] == pytest.approx(4 / 3 * math.pi * a ** 3, rel=1e-7)
    fm = min(r.cavity_modes(), key=lambda m: abs(m["f_Hz"] - f))
    assert fm["f_Hz"] == pytest.approx(f, rel=1e-6)
    V, A = srf._contour_volume_area(d.doc["regions"][1]["shape"])
    assert abs(V) == pytest.approx(4 / 3 * math.pi * a ** 3, rel=1e-13) and (V > 0) == ccw


def test_contour_spec_errors():
    """r < 0 along an arc (t through pi), a closing gap below the CAD tolerance, and a too-short arc are spec errors."""
    a = 0.1
    d, _ = _sphere_design(a)
    bad = json_copy(d.doc)
    bad["regions"][1]["shape"]["contour"]["segments"][0]["ellipse"]["center"] = [0.05, 0.0]   # r < 0 near t = pi?
    bad["regions"][1]["shape"]["contour"]["start"] = [0.05, -a]
    bad["regions"][1]["shape"]["contour"]["segments"][0]["ellipse"]["t"] = [-math.pi / 2, -3 * math.pi / 2]
    with pytest.raises(Exception, match="negative"):
        af.Design(bad).validate()
    bad = json_copy(d.doc)
    bad["regions"][1]["shape"]["contour"]["segments"][1]["line"] = [0.0, -a + 5e-8]
    with pytest.raises(Exception, match="within 1e-7 m of its start"):
        af.Design(bad).validate()
    bad = json_copy(d.doc)
    bad["regions"][1]["shape"]["contour"]["segments"][0]["ellipse"]["t"] = [-math.pi / 2, -math.pi / 2 + 1e-7]
    with pytest.raises(Exception, match="shorter than 1e-7 m"):
        af.Design(bad).validate()


def json_copy(x):
    import json
    return json.loads(json.dumps(x))


def test_cell_2d_wall_peaks_robust(tmp_path):
    """Smooth wall, order 3: E_pk (iris nose) and B_pk (iris-side wall) on three meshes (R_eq/10, /20, /40) within
    0.5 % and 1e-4 of the converged values; the peaks sit at the iris nose / iris-side wall."""
    Req = srf.PROJECTX_650_B09["equator_radius"]
    for n in (10, 20, 40):
        d = srf.elliptical_cell_design(axial_factor=0.86, h=Req / n, order=3, modes=6)
        o = srf.figures_of_merit(d.solve(tmp_path / f"n{n}", verbose=0))
        assert o["Epk_over_cB0"] == pytest.approx(EPK_2D, rel=5e-3), n
        assert o["Bpk_over_B0"] == pytest.approx(BPK_2D, rel=1e-4), n
        r, z = o["pump_peaks"]["Epk_at_rz"]
        assert 0.050 <= r < 0.057 and 0.078 < abs(z) <= 0.0893, (n, r, z)       # iris nose (tip at r = 50 mm)
        r, z = o["pump_peaks"]["Bpk_at_rz"]
        assert 0.07 < r < 0.09 and 0.07 < abs(z) < 0.08, (n, r, z)
        assert o["pump_peaks"]["Et_max_over_cB0"] < 0.05 * o["Epk_over_cB0"]   # PEC: tangential E small


@pytest.mark.parametrize("noise", [0.0, 0.04])
def test_azimuthal_wall_peaks_synthetic(tmp_path, noise):
    """srf.azimuthal_wall_peaks on synthetic wall samples of an m = 0 field: points scattered over the revolved exact
    contour (all azimuths), a Gaussian peak at the iris nose of each half (E) and on the iris-side wall (B), with
    and without independent point noise as in the 3D solutions (4 % on E, 1 % on B). The fit recovers the peak to 0.3 % (noise)
    and 1e-3 (none), unbiased by the extreme of the noisy points, which exceed it by > 10 %."""
    import numpy as np

    rng = np.random.default_rng(1)
    con = srf.elliptical_cell_contour(axial_factor=0.86)
    P = srf.contour_points(con["shape"], n_arc=4000)
    seg = np.hypot(*np.diff(P, axis=0).T)
    sl = np.concatenate([[0], np.cumsum(seg)])
    Ltot = sl[-1]
    n = 200000
    s = rng.uniform(0, Ltot, n)
    r, z = np.interp(s, sl, P[:, 0]), np.interp(s, sl, P[:, 1])
    keep = r > 1e-3
    s, r, z = s[keep], r[keep], z[keep]
    phi = rng.uniform(0, 2 * np.pi, len(s))
    # peaks at fixed arclengths (mirror pairs), widths 5 mm (E) and 15 mm (B)
    sEp = [0.157, Ltot - 0.157]
    sBp = [0.185, Ltot - 0.185]
    E = sum(1.843 * np.exp(-((s - q) / 5e-3) ** 2) for q in sEp)
    B = sum(1.464 * np.exp(-((s - q) / 15e-3) ** 2) for q in sBp)
    E = E * (1 + noise * rng.standard_normal(len(s)))
    B = B * (1 + 0.25 * noise * rng.standard_normal(len(s)))   # 3D: B scatters ~4x less than E
    D = np.column_stack([r * np.cos(phi), r * np.sin(phi), z, np.full(len(s), 1e-6), E * cv.C_LIGHT,
                         E * cv.C_LIGHT, B, B])
    f = tmp_path / "ws.f64"
    D.astype(np.float64).tofile(f)
    a = srf.azimuthal_wall_peaks({"samples_file": str(f), "B_rms": 1.0}, con)
    tol = 3e-3 if noise else 1e-3
    assert a["Epk_over_cB0"] == pytest.approx(1.843, rel=tol)
    assert a["Bpk_over_B0"] == pytest.approx(1.464, rel=tol)
    assert a["Epk_flux_over_cB0"] == pytest.approx(1.843, rel=tol)
    assert abs(a["Epk_over_cB0_halves"][0] - a["Epk_over_cB0_halves"][1]) < 2 * tol
    if noise:
        assert E.max() > 1.1 * 1.843      # the single-point maximum is biased high
    rz = a["Epk_at_rz"]
    i = int(np.argmin(np.abs(s - sEp[0]) + 10 * (z > 0)))
    assert np.hypot(rz[0] - r[i], abs(rz[1]) - abs(z[i])) < 1.5e-3


def test_azimuthal_wall_peaks_on_real_3d_samples():
    """srf.azimuthal_wall_peaks on a committed subsample of real 3D wall samples (h = R_eq/8, order 3, the points
    within 6 mm of the peaks; tests/reference/srf_wall_samples_o3n8_peaks.*) reproduces the full-sample values of the
    record (E 1.8359, B 1.4638) exactly, so a change of the estimator shows up here."""
    import json
    from pathlib import Path

    ref = Path(__file__).resolve().parent / "reference"
    meta = json.loads((ref / "srf_wall_samples_o3n8_peaks.json").read_text())
    pw = {"samples_file": str(ref / "srf_wall_samples_o3n8_peaks.f64"), "B_rms": meta["B_rms"],
          "samples_count": meta["samples_count"]}
    a = srf.azimuthal_wall_peaks(pw, srf.elliptical_cell_contour(axial_factor=0.86))
    for k, v in meta["expected"].items():
        assert a[k] == pytest.approx(v, rel=1e-12), k
    assert a["Epk_over_cB0"] == pytest.approx(EPK_2D, rel=5e-3) and a["Bpk_over_B0"] == pytest.approx(BPK_2D, rel=1e-3)
    pw["samples_count"] += 1
    with pytest.raises(ValueError, match="belongs to another solve"):
        srf.azimuthal_wall_peaks(pw, srf.elliptical_cell_contour(axial_factor=0.86))


def test_cell_3d_wall_peaks_converge_to_2d_records():
    """Regression of the 3D wall-peak robustness (records benchmarks/srf/records/srf_3d_smooth_o{2,3,4}.json, made by
    srf_3d_validation.py on the smooth wall): at orders >= 3, on every mesh (>= 3 meshes, R_eq/6 ... R_eq/12 and
    order 4), E_pk/(cB0) at the iris nose is within 1 % and B_pk/B0 within 0.2 % of the 2D values (1.8430,
    1.46401), xi to 3e-5, eta to 2e-4, G signal to 1e-4; at order 2 the peaks converge monotonically to the 2D values
    (E from below, B from above) and reach 2 % / 0.1 % at R_eq/16. The single-point maxima, which the 3D solutions'
    azimuthal scatter biases high, are not used (they exceed the 2D values by up to 35 %)."""
    import json
    from pathlib import Path

    rec = Path(__file__).resolve().parents[1] / "benchmarks/srf/records"
    hi = []
    for o in (3, 4):
        f = rec / f"srf_3d_smooth_o{o}.json"
        if f.exists():
            hi += json.loads(f.read_text())["rows"]
    assert len(hi) >= 3
    for r in hi:
        tag = (r["order"], round(1 / r["h_over_Req"]))
        assert r["peak_definition"] == "azimuthal" and r["wall"] == "smooth", tag
        assert r["Epk_over_cB0"] == pytest.approx(EPK_2D, rel=1e-2), tag
        assert r["Bpk_over_B0"] == pytest.approx(BPK_2D, rel=2e-3), tag
        assert r["xi_pump"] == pytest.approx(7.64298, rel=3e-5), tag
        assert r["eta_10"] == pytest.approx(0.449212, rel=2e-4), tag
        assert r["G_signal_ohm"] == pytest.approx(604.4785, rel=1e-4), tag
        rr, zz = r["Epk_at_rz"]
        assert 0.050 <= rr < 0.057 and 0.078 < abs(zz) <= 0.0893, tag
        a = r["azimuthal"]
        assert abs(a["Epk_over_cB0_halves"][0] / a["Epk_over_cB0_halves"][1] - 1) < 5e-3, tag
    lo = sorted(json.loads((rec / "srf_3d_smooth_o2.json").read_text())["rows"], key=lambda r: -r["h_over_Req"])
    E, B = [r["Epk_over_cB0"] for r in lo], [r["Bpk_over_B0"] for r in lo]
    assert len(lo) >= 4 and all(np.diff(E) > 0) and all(np.diff(B) < 0)
    assert E[-1] == pytest.approx(EPK_2D, rel=2e-2) and B[-1] == pytest.approx(BPK_2D, rel=1e-3)


@slow
def test_tuner_finds_default_axial_factor(tmp_path):
    Req = srf.PROJECTX_650_B09["equator_radius"]
    tu = srf.tune_degeneracy(design_kw={"h": Req / 30, "modes": 6}, bracket=(0.82, 0.90), tol=1e-5,
                             workdir=str(tmp_path))
    assert tu["axial_factor"] == pytest.approx(0.86, abs=0.005)
    assert tu["f_Hz"] == pytest.approx(1.4166e9, rel=1e-3)


@slow
@needs_strumpack
def test_cell_3d_coarse_matches_2d(tmp_path):
    """Coarse full 3D model (h = Req/6): pump/signal identified by their E_phi fraction, f, eta_10 and G close to
    the converged 2D values (docs/srf/README.md has the 3D convergence)."""
    Req = srf.PROJECTX_650_B09["equator_radius"]
    d = srf.elliptical_cell_design_3d(axial_factor=0.86, h=Req / 6, modes=12, target_f_Hz=1.4165e9)
    r = d.solve(tmp_path, verbose=0, mpi_ranks=int(os.environ.get("AXF_TEST_RANKS", "1")))
    o = srf.figures_of_merit_3d(r, 1.416511e9, 1.416531e9, contour=d.srf_contour)
    fp, fs = o["E_azimuthal_fraction"]
    assert fp < 0.01 and fs > 0.99
    assert o["f_pump_Hz"] == pytest.approx(1.416511e9, rel=3e-3)
    assert o["f_signal_Hz"] == pytest.approx(1.416531e9, rel=3e-3)
    assert o["eta_10"] == pytest.approx(0.4492, abs=0.01)
    assert o["eta_01_transpose"] == pytest.approx(o["eta_10"], rel=0.01)  # reciprocity near degeneracy
    assert o["G_pump_ohm"] == pytest.approx(524.5, rel=0.02)
    assert o["G_signal_ohm"] == pytest.approx(604.5, rel=0.05)
    # Smooth wall: the azimuthal-fit peaks of this coarse order-2 mesh are B 1.5055 (+2.8 %) and E 1.759
    # (-4.6 %) (srf_3d_smooth_o2.json); they converge to 2D with h and order (test_cell_3d_wall_peaks_converge_...).
    # The single-point maxima (1.588 / 2.49 here) are biased high by the azimuthal scatter.
    assert o["peak_definition"] == "azimuthal"
    assert 1.46 < o["Bpk_over_B0"] < 1.53 and 1.72 < o["Epk_over_cB0"] < 1.86
    assert o["Bpk_over_B0_pointwise_max"] > o["Bpk_over_B0"]


@slow
@needs_strumpack
def test_peak_fields_probes_3d_pillbox(tmp_path):
    """3D cavity keys of this branch on a closed cylinder with closed forms: E_azimuthal_fraction (TM010: 0,
    TE011: 1), peak_wall_fields of TM010 (B_pk/B_rms = max|J1|/|J1(chi01)| on the caps, E_pk/(c B_rms) =
    1/|J1(chi01)| at the cap centres) and the eigenmode field probes (E_z of TM010 ~ J0(chi01 r/a), B = curl E/omega)."""
    import numpy as np
    from scipy.special import j0, j1

    a, L = 0.1, 0.2
    pts = [[0.3 * a * t, 0.2 * a * t, 0.01] for t in (0.5, 1.0, 1.5, 2.0)]
    d = cv.cylinder_design_3d(a, L, h=a / 5, modes=4, target_f_Hz=1.147e9, eigensolver={"method": "shift_invert"})
    d.doc["fullwave"]["field_probes"] = pts
    d.doc["cavity"]["wall_samples"] = True
    r = d.solve(tmp_path, verbose=0, mpi_ranks=int(os.environ.get("AXF_TEST_RANKS", "2")))
    eig = r.fullwave["eigenmodes"]
    m0 = min(eig["modes"], key=lambda m: abs(m["f_Hz"] - 1.1474e9))
    assert m0["E_azimuthal_fraction"] < 1e-3
    te = [m for m in eig["modes"] if abs(m["f_Hz"] - 1.9759e9) < 0.01e9]
    if te:
        assert te[0]["E_azimuthal_fraction"] > 0.999
    pw = m0["peak_wall_fields"]
    x = 2.404825557695773
    assert pw["Bpk_over_Brms"] == pytest.approx(0.5818652242 / abs(j1(x)), rel=0.03)
    assert pw["Bpk_face_rms_over_Brms"] == pytest.approx(0.5818652242 / abs(j1(x)), rel=0.05)
    assert pw["cEpk_over_Brms"] == pytest.approx(1 / abs(j1(x)), rel=0.03)
    # consistent-flux (charge) recovery of the cap E_n, and the wall-sample dump (cavity.wall_samples)
    assert pw["cEpk_flux_over_Brms"] == pytest.approx(1 / abs(j1(x)), rel=0.03)
    assert abs(abs(pw["E_n_max_flux_at"][2]) - L / 2) < 1e-9
    S = np.fromfile(pw["samples_file"], dtype=np.float64).reshape(-1, 8)
    assert len(S) > 100 and pw["samples_columns"][5] == "E_n_flux"
    assert S[:, 3].sum() == pytest.approx(2 * math.pi * a * a + 2 * math.pi * a * L, rel=0.02)   # wall area
    assert S[:, 5].max() / (cv.C_LIGHT * pw["B_rms"]) == pytest.approx(pw["cEpk_flux_over_Brms"], rel=1e-9)
    assert pw["B_rms"] == pytest.approx(1 / (cv.C_LIGHT * math.sqrt(math.pi * a * a * L)), rel=0.01)
    assert np.hypot(*pw["E_max_at"][:2]) < 0.3 * a and abs(abs(pw["E_max_at"][2]) - L / 2) < 1e-9
    assert all(eig["field_probes"]["valid"])
    E = np.array(m0["probes"]["E"])
    rr = np.hypot(np.array(pts)[:, 0], np.array(pts)[:, 1])
    ez = E[:, 2] / E[0, 2] * j0(x * rr[0] / a)
    assert ez == pytest.approx(j0(x * rr / a), abs=0.02)
    B = np.array(m0["probes"]["B"])
    # |B_phi| / |E_z| = J1/J0 / c at each point (TM010, z-independent)
    ratio = np.hypot(B[:, 0], B[:, 1]) / np.abs(E[:, 2]) * cv.C_LIGHT
    assert ratio == pytest.approx(np.abs(j1(x * rr / a) / j0(x * rr / a)), rel=0.03)


def test_geometry_record_reproduced_by_a_coarse_solve(tmp_path):
    """The cached figures of merit behind heterodyne.PROJECTX_086 (order 3, h = R_eq/80) agree with a coarse
    solve (h = R_eq/20): integrals to 1e-4, B_pk to 3e-4, E_pk (polygon-corner sensitive) to 1e-2."""
    from axionfem import heterodyne as H

    rec = H._load_record(str(H.PROJECTX_RECORD))["figures_of_merit"]
    d = srf.elliptical_cell_design(axial_factor=0.86, h=0.2003 / 20, order=3, modes=6, angle_tol=0.02)
    o = srf.figures_of_merit(d.solve(tmp_path / "c", verbose=0))
    for k, tol in (("xi_pump", 1e-4), ("eta_10", 1e-4), ("G_pump_ohm", 1e-4), ("G_signal_ohm", 1e-4),
                   ("A_over_V23", 1e-4), ("Bpk_over_B0", 3e-4), ("B_equator_over_B0", 3e-4), ("Epk_over_cB0", 1e-2)):
        assert o[k] == pytest.approx(rec[k], rel=tol), k
    # the fresh figures of merit build a valid geometry (SRFGeometry validation) equal to the preset to the same tol
    g, w = H.SRFGeometry.from_modes(o), H.PROJECTX_086
    for f, tol in (("xi", 1e-4), ("eta", 1e-4), ("G_pump_ohm", 1e-4), ("G_signal_ohm", 1e-4), ("area_coeff", 1e-4),
                   ("Bpk_over_B0", 3e-4), ("Epk_over_cB0", 1e-2)):
        assert getattr(g, f) == pytest.approx(getattr(w, f), rel=tol), f


def test_ceff_regression_coarse(tmp_path):
    """c_eff of the vibration requirement (random figure error, c_eff = (ln 2)^(-1/2) R_eq L_c sqrt(pi I_4) with the
    2D wall fields, benchmarks/srf/srf_vibration_mixing.py) at h = R_eq/20, order 3: 3.7391 (3.7380 at R_eq/80, the
    recorded value).
    Also checks the I_4 / per-octave normalization through the G-average closed form (3.4088)."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "benchmarks/srf/srf_vibration_mixing.py"
    spec = importlib.util.spec_from_file_location("srf_vibration_mixing", p)
    vm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vm)
    row = vm.ceff_one(20, 3, str(tmp_path))
    assert row["c_eff"] == pytest.approx(3.7391, abs=2e-4)
    assert row["c_eff_G_average"] == pytest.approx(3.4088, abs=2e-4)
    assert row["A_wall_m2"] == pytest.approx(row["A_contour_m2"], rel=1e-9)
