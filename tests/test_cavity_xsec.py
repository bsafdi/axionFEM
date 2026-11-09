"""Cross-section cavity eigensolver ("symmetry": "xsec"; part of the beta cavity module).

References (docs/validation/cavity_xsec.md):
- empty disk R: TM_0n0 k = j_0n / R, C = 4 / j_0n^2 (0.6916602761 for TM010), G_2d = Z0 j_0n / 2 (barrel per unit
  length), with end caps G = Z0 j_01 L / (2 (R + L)) (the pillbox closed form; exact for every L);
  TE_mn (z-independent H_z): k = j'_mn / R, G_2d = Z0 j'_mn (1 - m^2 / j'_mn^2) / 2;
- coaxial cross-section b / a = 4: kb = 4.0976855393, C = 0.7949373125 (vacuum area), and the axisymmetric solver;
- off-centre rod: the method of particular solutions (cavity_xsec.eccentric_rod_tm_mode; independent of the FEM);
- centred dielectric rod: the 1D Bessel matching of cavity.sleeve_tm_mode (f, C, electric energy fraction);
- CAPP ideal sector cells (1710.06969 Tables 2-3): closed form of cavity_xsec.sector_cell_mode.
"""
import json
import math

import pytest

import axionfem as af
from axionfem import cavity as cv
from axionfem import cavity_xsec as cx

C0 = 299792458.0
Z0 = 1.25663706212e-6 * C0
J01, J02 = 2.404825557695773, 5.520078110286311
R = 0.05


def _tm(res):
    return [r for r in res.cavity_modes() if r["family"] == "TM"]


@pytest.fixture(scope="module")
def disk(tmp_path_factory):
    d = cx.cylinder_rods_design(R, [], wall_rf=cx.skin_rf(5.8e7), modes=6, families=("TM", "TE"), h=R / 16,
                                length=0.25)
    return d.solve(tmp_path_factory.mktemp("xs_disk"), verbose=0)


def test_disk_tm0n0(disk):
    tm = _tm(disk)
    ex = [m for m in cx.disk_tm_modes(R, 2 * C0 * J02 / (2 * math.pi * R)) if m["m"] == 0]
    t010, t020 = tm[0], min(tm, key=lambda r: abs(r["f_Hz"] - ex[1]["f_Hz"]))
    assert t010["f_Hz"] == pytest.approx(C0 * J01 / (2 * math.pi * R), rel=2e-7)
    assert t010["C"] == pytest.approx(4 / J01 ** 2, rel=2e-6)
    assert t010["C"] == pytest.approx(0.6916602761, rel=2e-6)
    assert t020["f_Hz"] == pytest.approx(ex[1]["f_Hz"], rel=1e-5)
    assert t020["C"] == pytest.approx(4 / J02 ** 2, rel=2e-5)
    assert t010["A_m2"] == pytest.approx(math.pi * R * R, rel=1e-6)
    assert t010["A_full_m2"] == pytest.approx(math.pi * R * R, rel=1e-6)
    assert t010["V_m3"] == pytest.approx(math.pi * R * R * 0.25, rel=1e-6)
    for r in tm:
        assert r["residual_true"] < 1e-8
    # the TM_1n0 doublets (C = 0)
    t110 = [r for r in tm if abs(r["f_Hz"] / (C0 * 3.831705970207512 / (2 * math.pi * R)) - 1) < 1e-5]
    assert len(t110) == 2 and all(abs(r["C"]) < 1e-8 for r in t110)


def test_disk_wall_loss_per_length_and_end_caps(disk):
    t010 = _tm(disk)[0]
    assert t010["G_ohm_2d"] == pytest.approx(Z0 * J01 / 2, rel=2e-6)
    L = 0.25
    assert t010["G_ohm"] == pytest.approx(Z0 * J01 * L / (2 * (R + L)), rel=2e-6)
    caps = t010["end_caps"]
    assert caps["G_ohm"] == pytest.approx(Z0 * J01 / R * L / 2, rel=2e-6)  # omega mu0 L / 2
    Rs = cv.skin_Rs(t010["f_Hz"], 1 / 5.8e7)
    assert t010["Q0"] == pytest.approx(t010["G_ohm"] / Rs, rel=2e-6)
    assert t010["Q0_2d"] == pytest.approx(Z0 * J01 / 2 / Rs, rel=2e-6)
    w = t010["walls"]["wall"]
    assert w["G_ohm_pointwise"] == pytest.approx(w["G_ohm"], rel=1e-3)  # second-order pointwise check
    assert w["P_frac"] + caps["P_frac"] == pytest.approx(1.0, rel=1e-12)


def test_disk_te_modes(disk):
    te = [r for r in disk.cavity_modes() if r["family"] == "TE"]
    ex = cx.disk_te_modes(R, 3e10)
    for m in ex[:3]:
        hits = [r for r in te if abs(r["f_Hz"] / m["f_Hz"] - 1) < 1e-5]
        assert len(hits) == m["degeneracy"]
        for r in hits:
            assert r["C"] == 0.0
            assert r["G_ohm_2d"] == pytest.approx(m["G2d_ohm"], rel=1e-4)
            assert r["end_caps"] is None  # z-independent TE modes do not satisfy end caps


def test_axisymmetric_pillbox_cross_check(disk, tmp_path):
    """TM010 with end caps: the cross-section result with cavity.length = L equals the axisymmetric pillbox."""
    L = 0.25
    pb = cv.pillbox_design(R, L, h=R / 16, modes=2, families=("TM",), wall_material={
        "kind": "conductor", "rf": cx.skin_rf(5.8e7)}).solve(tmp_path / "pb", verbose=0)
    a = [r for r in pb.cavity_modes() if r["family"] == "TM"][0]
    b = _tm(disk)[0]
    assert b["f_Hz"] == pytest.approx(a["f_Hz"], rel=1e-6)
    assert b["C"] == pytest.approx(a["C"], rel=1e-5)
    assert b["G_ohm"] == pytest.approx(a["G_ohm"], rel=1e-5)
    assert b["Q0"] == pytest.approx(a["Q0"], rel=1e-5)


def test_coax_b_over_a_4(tmp_path):
    a, b = 0.05, 0.2
    d = cx.xsec_design([{"name": "cav", "shape": {"disk": {"center": [0, 0], "radius": b}}, "material": "vacuum"},
                        {"name": "rod", "shape": {"disk": {"center": [0, 0], "radius": a}}, "material": "copper"}],
                       wall_material="copper", modes=2, h=(b - a) / 12)
    res = d.solve(tmp_path / "coax", verbose=0)
    t = _tm(res)[0]
    ref = cx.coax_tm_G2d(a, b)
    assert t["f_Hz"] * 2 * math.pi / C0 * b == pytest.approx(4.0976855393, rel=5e-6)
    assert t["C"] == pytest.approx(0.7949373125, rel=5e-6)
    assert t["C_full"] == pytest.approx(0.7949373125 * (1 - (a / b) ** 2), rel=1e-5)
    assert t["walls"]["rod"]["G_ohm"] == pytest.approx(ref["G2d_rod_ohm"], rel=1e-5)
    assert t["walls"]["wall"]["G_ohm"] == pytest.approx(ref["G2d_wall_ohm"], rel=1e-5)
    # the same rod/wall in the axisymmetric solver (rod spanning both caps): f and C
    ax = cv.coax_design(a, b, 0.2, h=(b - a) / 12, modes=3).solve(tmp_path / "axi", verbose=0)
    # nearest axisymmetric TM mode (the TEM-like TM001 at c / 2L = 749 MHz lies below)
    m = min((r for r in ax.cavity_modes() if r["family"] == "TM"), key=lambda r: abs(r["f_Hz"] - t["f_Hz"]))
    assert t["f_Hz"] == pytest.approx(m["f_Hz"], rel=1e-6)
    assert t["C"] == pytest.approx(m["C"], rel=1e-5)


def test_off_centre_rod_vs_particular_solutions(tmp_path):
    Rw, a, dx = 0.2, 0.05, 0.06
    ref = cx.eccentric_rod_tm_mode(Rw, a, dx)
    res = cx.cylinder_rods_design(Rw, [{"radius": a, "center": [dx, 0.0]}], wall_rf=cx.skin_rf(5.8e7), h=Rw / 20,
                                  modes=3).solve(tmp_path / "ecc", verbose=0)
    t = _tm(res)[0]
    assert t["f_Hz"] == pytest.approx(ref["f_Hz"], rel=2e-6)
    assert t["C"] == pytest.approx(ref["C"], rel=2e-5)
    assert t["C_full"] == pytest.approx(ref["C_full"], rel=2e-5)
    assert t["walls"]["rod0"]["G_ohm"] == pytest.approx(ref["G2d_rod_ohm"], rel=1e-4)
    assert t["walls"]["wall"]["G_ohm"] == pytest.approx(ref["G2d_wall_ohm"], rel=1e-4)
    # the same rod placed by pivot, arm and angle
    res2 = cx.cylinder_rods_design(Rw, [{"radius": a, "pivot": [0.03, 0.0], "arm": 0.03, "angle_deg": 0.0}],
                                   h=Rw / 20, modes=1).solve(tmp_path / "ecc2", verbose=0)
    assert _tm(res2)[0]["f_Hz"] == pytest.approx(t["f_Hz"], rel=1e-6)


def test_dielectric_rod_weak_form(tmp_path):
    """Centred dielectric rod (eps in the mass matrix): f, C (eps-weighted, full area) and the electric energy
    fraction (Q_d = 1 / (tan_delta fe)) against the 1D Bessel matching."""
    b, r2, eps, tand = 0.05, 0.015, 9.4, 1e-4
    ref = cv.sleeve_tm_mode(b, 0.0, r2, eps)
    res = cx.cylinder_rods_design(b, [{"radius": r2, "center": [0, 0], "name": "sapphire",
                                       "material": {"kind": "linear", "eps_r": eps, "tan_delta": tand}}],
                                  h=b / 16, modes=2).solve(tmp_path / "diel", verbose=0)
    t = _tm(res)[0]
    assert t["f_Hz"] == pytest.approx(ref["f_Hz"], rel=2e-6)
    assert t["C"] == pytest.approx(ref["C"], rel=1e-5)
    assert t["energy_fraction"]["sapphire"] == pytest.approx(ref["fe_sleeve"], rel=1e-5)
    assert t["Q_dielectric"] == pytest.approx(1 / (tand * ref["fe_sleeve"]), rel=1e-5)


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_capp_ideal_sector_cells(N, tmp_path):
    """1710.06969 Tables 2-3: ideal sector cells (Rc = 25 mm, sigma = 5.96e7 S/m, no end caps)."""
    Rc, sigma = 0.025, 5.96e7
    ref = cx.sector_cell_mode(N, Rc)
    res = cx.sector_cell_design(Rc, N, wall_rf=cx.skin_rf(sigma), h=Rc / 24).solve(tmp_path / f"s{N}", verbose=0)
    t = _tm(res)[0]
    assert t["f_Hz"] == pytest.approx(ref["f_Hz"], rel=2e-6)
    assert t["C"] == pytest.approx(ref["C"], rel=2e-5)
    assert t["G_ohm_2d"] == pytest.approx(ref["G2d_ohm"], rel=2e-4)
    Q = t["Q0_2d"]
    assert Q == pytest.approx(ref["G2d_ohm"] / cv.skin_Rs(ref["f_Hz"], 1 / sigma), rel=2e-4)
    paper = {2: 17794, 4: 16151, 6: 14320, 8: 12907}[N]  # analytic column; the paper used sigma ~ 5.998e7
    assert Q == pytest.approx(paper, rel=6e-3)


def test_capp_full_disk_with_sheets(tmp_path):
    """Two-cell cavity as a full disk with a diametral sheet (curves): same TM mode as the half-disk cell."""
    Rc = 0.025
    ref = cx.sector_cell_mode(2, Rc)
    res = cx.sector_cell_design(Rc, 2, full=True, wall_rf=cx.skin_rf(5.96e7), h=Rc / 24, modes=2,
                                ).solve(tmp_path / "full", verbose=0)
    t = _tm(res)[0]
    assert t["f_Hz"] == pytest.approx(ref["f_Hz"], rel=2e-6)
    assert t["C"] == pytest.approx(ref["C"], rel=2e-5)
    assert t["G_ohm_2d"] == pytest.approx(ref["G2d_ohm"], rel=2e-4)
    assert set(t["walls"]) == {"wall", "partition0", "partition1"}
    cav = res.data["fullwave"]["cavity"]
    assert cav["components"] == 2 and "two_sided_sheets" not in cav  # one cell selected: the sheets are its walls


def test_te_split_across_sheet_and_free_edge_warning(tmp_path):
    """A partition with a centre gap joins the cells: TE_z H_z is split across the sheet, and the free sheet edges
    are flagged."""
    Rc = 0.025
    res = cx.sector_cell_design(Rc, 2, full=True, gap=0.3 * Rc, h=Rc / 16, modes=3).solve(tmp_path / "gap", verbose=0)
    cav = res.data["fullwave"]["cavity"]
    assert cav["components"] == 1 and sorted(cav["sheets_with_free_edge"]) == ["partition0", "partition1"]
    assert any("free sheet edge" in w for w in res.data["warnings"])
    d = cx.sector_cell_design(Rc, 2, full=True, gap=0.3 * Rc, h=Rc / 16, modes=3)
    d.doc["fullwave"]["families"] = ["TE"]
    te = d.solve(tmp_path / "gap_te", verbose=0)
    assert te.data["fullwave"]["eigenmodes"]["TE"]["solver"]["split_dofs"] > 0
    assert te.data["fullwave"]["eigenmodes"]["TE"]["solver"]["dropped_static_modes"] == 1


def test_mode_fields_and_normalization(disk, tmp_path):
    import numpy as np

    pytest.importorskip("matplotlib")   # plot_mode draws with matplotlib (the `plots` extra)

    t010 = _tm(disk)[0]  # row 0 of cavity_modes() is TE11 (below TM010)
    F = disk.cavity_mode_field(t010, subdiv=3)
    # E_z(0) of the normalized TM010: J0 / sqrt(int J0^2 dA) = 1 / (sqrt(pi) R J1(j01)), positive (int E_z > 0)
    s = disk.cavity_mode_sample(t010, [0.0, 2 * R], [0.0, 0.0])
    J1 = 0.5191474972894669
    assert s["E_z"][0] == pytest.approx(1 / (math.sqrt(math.pi) * R * J1), rel=1e-5)
    assert math.isnan(s["E_z"][1])
    # H_phi = J1 / (Z0 ...) on the wall: |H| at the wall over E_z(0) = J1 / (Z0 J0(0)) for TM010
    w = disk.cavity_mode_sample(t010, [0.999999 * R], [0.0])
    assert w["Hmag"][0] / s["E_z"][0] == pytest.approx(J1 / Z0, rel=1e-3)  # pointwise derivative at the wall
    assert F["mode"]["f_Hz"] == t010["f_Hz"] and np.all(F["E_x"] == 0)
    paths = disk.save_cavity_modes(tmp_path / "npz", modes=[1])
    meta = json.loads(str(np.load(paths[0])["meta"]))
    assert meta["symmetry"] == "xsec"
    p = disk.plot_mode(t010, tmp_path / "m.png", quantity="Ez")
    assert p.exists() and p.stat().st_size > 1000


def test_targeted_and_spectrum(tmp_path):
    f020 = C0 * J02 / (2 * math.pi * R)
    res = cx.cylinder_rods_design(R, [], modes=1, target_f_Hz=1.001 * f020, h=R / 12).solve(tmp_path / "tg", verbose=0)
    t = _tm(res)
    assert len(t) == 1 and t[0]["f_Hz"] == pytest.approx(f020, rel=3e-5)
    sp = cx.z_dependent_spectrum(t, 0.25, p_max=1)
    assert sp[1]["f_Hz"] == pytest.approx(math.hypot(t[0]["f_Hz"], C0 / 0.5), rel=1e-12)


def test_rod_angle_sweep_and_map(tmp_path):
    pytest.importorskip("matplotlib")   # plot_mode_map draws with matplotlib (the `plots` extra)
    rows = cx.rod_angle_sweep(0.05, [{"radius": 0.008, "arm": 0.025, "angle_deg": 0.0}], [0, 90, 180],
                              workdir=tmp_path / "sw", h=0.05 / 12, modes=3)
    fs = [r["tracked"]["f_Hz"] for r in rows]
    assert fs[0] == pytest.approx(fs[1], rel=1e-4) and fs[1] == pytest.approx(fs[2], rel=1e-4)  # rotation invariance
    p = cx.plot_mode_map(rows, tmp_path / "map.png")
    assert p.exists()


@pytest.mark.parametrize("mutate,what", [
    (lambda d: d.update(windings=[]), "windings"),
    (lambda d: d["cavity"].update(wall_material="vacuum"), "conductor or superconductor"),
    (lambda d: d["cavity"].update(length=-1), "positive length"),
    (lambda d: d["fullwave"].update(frequencies=[1e9]), "unknown field"),
    (lambda d: d.pop("cavity"), "needs a 'cavity' section"),
    (lambda d: d["regions"][0]["shape"].update({"disk": {"center": [0, 0], "radius": 0.05, "arm": 0.01}}), "angle_deg"),
])
def test_spec_errors(mutate, what):
    d = cx.cylinder_rods_design(R, [], modes=2).doc
    mutate(d)
    with pytest.raises(af.AxionFEMError, match=what):
        af.Design(d).validate()


def test_planar_shapes_accept_negative_x():
    d = cx.cylinder_rods_design(R, [{"radius": 0.005, "center": [-0.02, -0.01]}], modes=2).doc
    d["regions"].append({"name": "box", "shape": {"rect": {"x": [-0.01, 0.0], "y": [0.0, 0.01]}}, "material": "copper"})
    af.Design(d).validate()


# ---------------------------------------------------------------- off-centre rods, end caps, walls, sweeps
def test_off_centre_dielectric_rod_vs_particular_solutions(tmp_path):
    """Adversarial: an OFF-centre dielectric rod (curved eps interface not aligned with any symmetry of the wall)
    against the independent method-of-particular-solutions reference (Bessel bases inside/outside, interface
    matching): f, C (eps-weighted, full area), the rod's electric energy fraction and the wall G."""
    R, a, d, eps = 0.05, 0.0075, 0.02, 9.4
    ref = cx.eccentric_dielectric_rod_tm_mode(R, a, d, eps)
    assert ref["boundary_residual"] < 1e-7
    res = cx.cylinder_rods_design(R, [{"radius": a, "center": [d, 0.0], "name": "sapph",
                                       "material": {"kind": "linear", "eps_r": eps, "tan_delta": 1e-4}}],
                                  wall_rf=cx.skin_rf(5.8e7), h=R / 24, modes=3).solve(tmp_path / "ecc_diel", verbose=0)
    t = min(_tm(res), key=lambda r: abs(r["f_Hz"] - ref["f_Hz"]))
    assert t["f_Hz"] == pytest.approx(ref["f_Hz"], rel=3e-6)
    assert t["C"] == pytest.approx(ref["C"], rel=1e-5)
    assert t["energy_fraction"]["sapph"] == pytest.approx(ref["fe_rod"], rel=1e-5)
    assert t["walls"]["wall"]["G_ohm"] == pytest.approx(ref["G2d_wall_ohm"], rel=1e-5)
    # a wrong weak form (eps_r in the stiffness instead of the mass matrix) would move f by O(1): sanity of the scale
    assert abs(ref["f_Hz"] / (C0 * J01 / (2 * math.pi * R)) - 1) > 0.1


def test_end_cap_flux_flow_orientation(tmp_path):
    """Flux-flow walls in an axial B0: TM_z barrel currents run along z (parallel to the vortices, s = 0), the end-cap
    currents are in the cap plane (perpendicular, s = 1)."""
    ff = {"Bc2_T": 10.0, "rho_n_ohm_m": 1.5e-7, "lambda_m": 1.1e-7, "f_depin_Hz": 1e9}
    rf = {"model": "power", "Rs_ohm": 1e-6, "T_K": 4.2, "flux_flow": ff}
    B0 = 2.0
    d = cx.cylinder_rods_design(R, [], wall_rf=rf, length=0.25, h=R / 10, modes=1, Bz=B0)
    t = _tm(d.solve(tmp_path / "ff", verbose=0))[0]
    w, caps = t["walls"]["wall"], t["end_caps"]
    Rs_tot, Rs_ff = cv.surface_resistance(dict(rf, flux_flow=dict(ff, B_T=B0)), t["f_Hz"], flux_flow_part=True)
    assert w["flux_flow_orientation_factor"] == 0.0 and w["Rs_ohm"] == pytest.approx(1e-6, rel=1e-12)
    assert caps["flux_flow_orientation_factor"] == 1.0
    assert caps["Rs_ohm"] == pytest.approx(Rs_tot, rel=1e-10) and Rs_ff > 1e-4


def test_post_touching_wall_is_reported(tmp_path):
    """A post overlapping the wall becomes part of the outer boundary: area_full (C_full) excludes it; reported."""
    a = 0.008
    res = cx.cylinder_rods_design(R, [{"radius": a, "center": [R - 0.7 * a, 0.0]}], h=R / 16,
                                  modes=1).solve(tmp_path / "touch", verbose=0)
    cav = res.data["fullwave"]["cavity"]
    assert sorted(cav["outer_boundary_bodies"]) == ["rod0", "wall"]
    assert any("outer boundary of the cavity is formed by several bodies" in w for w in res.data["warnings"])
    t = _tm(res)[0]
    assert t["C_full"] == pytest.approx(t["C"], rel=1e-9)
    # a free-standing post: no such report
    res2 = cx.cylinder_rods_design(R, [{"radius": a, "center": [0.02, 0.0]}], h=R / 16, modes=1).solve(tmp_path / "free",
                                                                                                       verbose=0)
    assert "outer_boundary_bodies" not in res2.data["fullwave"]["cavity"]
    assert res2.data["fullwave"]["cavity"]["area_full_m2"] == pytest.approx(math.pi * R * R, rel=1e-6)


def test_sweep_overlap_tracking(tmp_path):
    """track="overlap" follows the TM010-like branch (field overlap ~1 between steps); invalid track rejected."""
    rows = cx.sweep(lambda x: cx.cylinder_rods_design(R, [{"radius": 0.0125, "center": [x, 0.0]}], h=R / 10, modes=4),
                    [0.0, 0.008, 0.016, 0.024], workdir=tmp_path / "ov", track="overlap")
    ref = cx.sweep(lambda x: cx.cylinder_rods_design(R, [{"radius": 0.0125, "center": [x, 0.0]}], h=R / 10, modes=4),
                   [0.0, 0.008, 0.016, 0.024], workdir=tmp_path / "c", track="C")
    assert rows[0]["tracked_overlap"] is None
    for r, q in zip(rows[1:], ref[1:]):
        # the field changes strongly as the rod leaves the axis (overlap 0.79 for the first step, > 0.95 later)
        assert r["tracked_overlap"] > 0.7 and r["tracked"]["f_Hz"] == pytest.approx(q["tracked"]["f_Hz"], rel=1e-12)
    with pytest.raises(ValueError, match="track"):
        cx.sweep(lambda x: None, [0], track="max")


def test_lossless_walls_consistent_with_3d(tmp_path):
    """cavity.allow_lossless_walls as in 3D: a named wall material without rf data is a spec error unless allowed;
    the builders set the flag for their default lossless wall."""
    d = cx.cylinder_rods_design(0.05, h=0.05 / 8, modes=1, fill={"kind": "linear", "eps_r": 2.0, "tan_delta": 1e-4})
    assert d.doc["cavity"]["allow_lossless_walls"] is True
    r = d.solve(tmp_path / "a", verbose=0)
    lo = _tm(r)[0]
    assert lo["Q0"] == pytest.approx(lo["Q_dielectric"], rel=1e-12)
    assert not any("rf data" in w for w in r.data["warnings"])
    d.doc["cavity"].pop("allow_lossless_walls")
    with pytest.raises(af.AxionFEMError, match="no rf data"):
        d.validate()
