"""An imposed background in 3D full-wave sweeps (docs/physics/fullwave_background.md) on a small copy of the
`solenoid` template, a shorted coax read out across its bottom gap inside a solenoid (the full study and its record:
benchmarks/fullwave3d_background.py, docs/validation/fullwave3d_background.md):

- the coil's field as a grid background (its analytic field; the coil region kept, as a mu_r = 1 material) against the
  same coil as the design's winding, on one mesh: Z identical (the background does not touch the readout) and Psi
  equal up to the two paths' discretization, at kR = 0.02 and 0.8; at kR = 0.02 Psi(omega) is the 3D MQS Psi (with a
  background its reciprocal form); the background's exterior term against the axisymmetric engine's exterior term for
  the winding; the winding model carries none of the background's outputs (no-background invariance);
- a magnet longer than the domain sphere (its field crosses R1, as DMRadio-Core's would): Psi at kR = 0.8 does not
  depend on R1, which it does without the background's exterior term;
- a superconducting disk that screens the coil's field: the reaction field (Psi_lab != 0, with its own exterior term)
  next to the background, against the winding model and at two domain radii;
- a uniform background with the radiating boundary: its exterior term omitted, labelled and warned about;
- a closed domain (boundary "pec") with a grid background and source_check: the direct axion-source solve with the
  background in its right-hand side reproduces the reciprocal Psi;
- a grid that ends inside the domain sphere: labelled, warned about, no exterior term (null).

The default suite has the 3D MQS Psi with a background (tests/test_background_3d.py). Coarse ND2 meshes; set
AXF_SLOW=1 (AXF_MPI=1: 4 ranks).
"""
import os
import sys
from pathlib import Path

import pytest

import axionfem as af

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import axi_background_exterior as bx  # noqa: E402
import fullwave3d_background as fb  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1"),
    pytest.mark.skipif(not af.build_info()["engine_3d"],
                       reason="needs the 3D engine (an MPI build; the wheels have none)"),
]

MESH = {"order": 2, "size": 0.09, "surface_size": 0.03}
NEW_ROW_KEYS = ("Psi_background", "Psi_over_MQS_reciprocal")
NEW_FULLWAVE_KEYS = ("background", "exterior_term_background")


def _solve(path, doc):
    r = af.Design(doc).solve(path, verbose=0, mpi_ranks=4 if os.environ.get("AXF_MPI") else 1)
    for row in r.fullwave["sweep"]:
        assert row["solver"]["relative_residual"] < 1e-6
    return r


def _psi(row):
    return complex(row["Psi_re"], row["Psi_im"])


def _part(row, key):
    return complex(row["Psi_background"][key]["re"], row["Psi_background"][key]["im"])


def _z(row):
    return complex(row["Z_re_ohm"], row["Z_im_ohm"])


def test_analytic_coil_field():
    check = fb.field_self_check()
    assert check["on_axis_relative_error"] < 1e-12
    assert check["loop_superposition_relative_error"] < 1e-10
    assert check["divergence_over_mu0_J"] < 1e-6


def test_background_grid_matches_the_winding(tmp_path):
    kr = (0.02, 0.8)
    grid = fb.grid(2.0, 2.0, h_fine=0.004, h=0.02)
    wnd = _solve(tmp_path / "windings", fb.design(True, None, fb.R1, kr, mesh=MESH))
    bg = _solve(tmp_path / "grid", fb.design(False, grid, fb.R1, kr, mesh=MESH))
    # no background: none of the background's outputs, and the MQS Psi in its primary form
    for key in NEW_FULLWAVE_KEYS:
        assert key not in wnd.fullwave
    for row in wnd.fullwave["sweep"]:
        for key in NEW_ROW_KEYS:
            assert key not in row
    assert wnd.fullwave["exterior_source_multipoles"]["field"] == "static B_lab"
    assert wnd.data["checks"]["Psi_reported_form"].startswith("primary")
    assert "Psi_primary_form" not in wnd.data["checks"]
    fw = bg.fullwave
    assert fw["exterior_term_background"] == "included (grid, then its exterior multipoles)"
    ext = fw["background"]["exterior"]
    print("background:", fw["background"])
    assert ext["quadrature_relative_change_doubled_panels"] < 1e-4
    assert ext["continuation_radius_m"] == pytest.approx(2.0)
    assert ext["continuation_samples"] > 0 and ext["continuation_mismatch"] < 1e-4
    assert fw["background"]["physical_volume_outside_grid_fraction"] == 0
    for a, b in zip(wnd.fullwave["sweep"], bg.fullwave["sweep"]):
        pa, pb = _psi(a), _psi(b)
        print(f"kR {b['k_R']:.3f}: Psi winding {pa:.6e}, grid {pb:.6e} (volume {_part(b, 'volume'):.4e}, exterior "
              f"{_part(b, 'exterior'):.4e}), |dPsi|/|Psi| {abs(pb - pa) / abs(pa):.2e}; L {a['L_H']:.6e} {b['L_H']:.6e}")
        # the readout problem is the same on the same mesh: the background enters only Psi
        assert _z(b) == pytest.approx(_z(a), rel=1e-9)
        assert abs(pb - pa) < 5e-3 * abs(pa)
        assert abs(_psi(b) - _part(b, "volume") - _part(b, "exterior")) < 1e-12 * abs(pb)
    # with a background the 3D MQS Psi is its reciprocal form, the low-frequency limit of Psi(omega)
    checks = bg.data["checks"]
    assert checks["Psi_reported_form"].startswith("reciprocal")
    assert abs(bg.coupling["Psi_SI_H_T_m2"]) == pytest.approx(abs(checks["Psi_A_lab_dot_B_pick"]), rel=1e-14)
    assert "over_reported" in checks["Psi_primary_form"]
    assert bg.fullwave["sweep"][0]["Psi_over_MQS"] == pytest.approx(1.0, abs=2e-3)
    # the axisymmetric engine (uniform annular feed, as this 360-degree port) with the coil as its winding: its exterior
    # term is the same integral, int A_coil . B_I outside R1, from its own static multipoles (coarse mesh: 0.6 % and
    # 2.2 % measured, totals 4.5e-6 and 1.4 %)
    axi = af.Design(fb.design(True, None, fb.R1, kr, axisymmetric=True)).solve(tmp_path / "axi", verbose=0)
    assert bg.coupling["Psi_SI_H_T_m2"] == pytest.approx(axi.coupling["Psi_SI_H_T_m2"], rel=1e-4)
    for a, b in zip(axi.fullwave["sweep"], bg.fullwave["sweep"]):
        ea = complex(a["Psi_exterior"]["re"], a["Psi_exterior"]["im"])
        print(f"kR {b['k_R']:.3f}: exterior 3D {_part(b, 'exterior'):.5e}, axisymmetric {ea:.5e}; "
              f"Psi 3D {_psi(b):.6e}, axisymmetric {_psi(a):.6e}")
        assert abs(_part(b, "exterior") - ea) < 0.05 * abs(ea)
        assert abs(_psi(b) - _psi(a)) < 0.03 * abs(_psi(a))
    assert abs(_psi(bg.fullwave["sweep"][0]) - _psi(axi.fullwave["sweep"][0])) < 1e-4 * abs(_psi(axi.fullwave["sweep"][0]))


def test_long_magnet_needs_the_exterior_term(tmp_path):
    kr = (0.8,)
    grid = fb.grid(2.0, 3.0, z1=-fb.LONG_HALF, z2=fb.LONG_HALF, fine_z=1.1, h_fine=0.004, h=0.02)
    rows = {}
    for scale in (1.0, 1.5):
        r = _solve(tmp_path / f"R{scale}", fb.design(False, grid, scale * fb.R1, kr, mesh=MESH))
        rows[scale] = r.fullwave["sweep"][0]
    a, b = rows[1.0], rows[1.5]
    pa, pb = _psi(a), _psi(b)
    ea, eb = _part(a, "exterior"), _part(b, "exterior")
    print(f"R1 x1: Psi {pa:.6e} (exterior {ea:.4e}); R1 x1.5: Psi {pb:.6e} (exterior {eb:.4e})")
    print(f"|dPsi|/|Psi| {abs(pb - pa) / abs(pa):.2e}; without the exterior term {abs((pb - eb) - (pa - ea)) / abs(pa):.2e}")
    assert abs(ea) > 0.05 * abs(pa)   # the magnet crosses R1: a sizeable part of Psi lies outside
    assert abs(pb - pa) < 0.1 * abs((pb - eb) - (pa - ea))


def test_reaction_field_next_to_the_background(tmp_path):
    """A superconducting disk screens the coil's field: with the coil as a background, A_lab is the disk's reaction
    field (E . curl A_lab with its multipole exterior term) and A_bg the coil's (the reciprocal form with its own
    exterior term). Their sum must match the coil as a winding (one A_lab, one path) on the same mesh, and not depend on
    the domain radius."""
    kr = (0.02, 0.8)
    grid = fb.grid(2.0, 2.0, h_fine=0.004, h=0.02)
    wnd = _solve(tmp_path / "windings", fb.design(True, None, fb.R1, kr, mesh=MESH, plug=True))
    bg = _solve(tmp_path / "grid", fb.design(False, grid, fb.R1, kr, mesh=MESH, plug=True))
    big = _solve(tmp_path / "grid_R1x1.5", fb.design(False, grid, 1.5 * fb.R1, (0.8,), mesh=MESH, plug=True))
    for a, b in zip(wnd.fullwave["sweep"], bg.fullwave["sweep"]):
        pa, pb = _psi(a), _psi(b)
        lab = pb - _part(b, "volume") - _part(b, "exterior")   # the reaction field's part
        ext_lab = complex(b["Psi_exterior"]["re"], b["Psi_exterior"]["im"])
        print(f"kR {b['k_R']:.3f}: Psi winding {pa:.6e}, grid {pb:.6e}; reaction part {lab:.4e} (its exterior "
              f"{ext_lab:.3e}); |dPsi|/|Psi| {abs(pb - pa) / abs(pa):.2e}")
        assert _z(b) == pytest.approx(_z(a), rel=1e-9)
        assert abs(lab) > 0.01 * abs(pb)   # the reaction is not negligible
        assert abs(pb - pa) < 5e-3 * abs(pa)
    assert bg.fullwave["sweep"][0]["Psi_over_MQS"] == pytest.approx(1.0, abs=3e-3)
    a, b = bg.fullwave["sweep"][1], big.fullwave["sweep"][0]
    pa, pb = _psi(a), _psi(b)
    ea, eb = _part(a, "exterior"), _part(b, "exterior")
    print(f"R1 x1.5: |dPsi|/|Psi| {abs(pb - pa) / abs(pa):.2e}; without the background's exterior term "
          f"{abs((pb - eb) - (pa - ea)) / abs(pa):.2e}")
    assert abs(pb - pa) < 0.1 * abs((pb - eb) - (pa - ea))


def test_uniform_background_with_the_radiating_boundary(tmp_path):
    """A uniform background fills all space: its exterior term diverges and is omitted, labelled and warned about.
    At kR = 0.02 Psi(omega) is still the MQS Psi (the reciprocal form) up to the (kR)^2 dispersion."""
    doc = fb.design(False, {"type": "uniform", "Bz": 1.0}, fb.R1, (0.02,), mesh=MESH)
    r = _solve(tmp_path / "uniform", doc)
    fw = r.fullwave
    row = fw["sweep"][0]
    print("uniform:", fw["exterior_term_background"], row["Psi_background"], row["Psi_over_MQS"])
    assert fw["exterior_term_background"].startswith("omitted (uniform background")
    assert any("omitted for the uniform background" in w for w in r.data["warnings"])
    assert row["Psi_background"]["exterior"] is None   # no term computed: null, not zero
    assert "Psi_over_MQS_reciprocal" not in row
    assert row["Psi_over_MQS"] == pytest.approx(1.0, abs=3e-3)


def test_closed_domain_grid_background_source_check(tmp_path):
    """boundary "pec" with a grid background and source_check: the direct solve driven by the axion source (the right-
    hand side Bt, the background's int curl N . A_bg included) must give the voltage -i omega Psi of the reciprocal
    solve, which the solver checks componentwise (it raises otherwise)."""
    grid = fb.grid(2.0, 2.0, h_fine=0.004, h=0.02)
    doc = fb.design(False, grid, fb.R1, (0.4,), boundary="pec", mesh=MESH)
    doc["fullwave"]["source_check"] = True
    r = _solve(tmp_path / "pec", doc)
    fw = r.fullwave
    row = fw["sweep"][0]
    print("source reciprocity:", row["source_reciprocity_relative_error"], row["source_reciprocity_componentwise"])
    assert fw["exterior_term_background"].startswith("none (closed")
    assert row["source_reciprocity_componentwise"]["passed"]
    assert row["source_reciprocity_relative_error"] < 1e-5
    assert row["Psi_background"]["exterior"] is None
    assert abs(_part(row, "volume")) > 0.99 * abs(_psi(row))


def test_grid_that_ends_inside_the_domain_sphere(tmp_path):
    """The coil's grid cut to r <= 0.4 m, |z| <= 0.4 m (corners at 0.57 m) inside a domain sphere of R1 = 0.6 m has no
    exterior moments: the magnet's field beyond the grid is coupled nowhere. The label says so, the coverage warning and
    the grid-inside warning fire, and no exterior term is computed: Psi_background.exterior is null, in the row and per
    port, with no last-order fraction."""
    cut = bx.crop(fb.grid(2.0, 2.0, h_fine=0.004, h=0.02), 0.4, -0.4, 0.4)
    r = _solve(tmp_path / "inside", fb.design(False, cut, 0.6, (0.8,), mesh=MESH))
    fw = r.fullwave
    print("inside:", fw["exterior_term_background"], fw["background"]["physical_volume_outside_grid_fraction"],
          fw["background"]["grid_edge_A_over_max"], r.data["warnings"])
    assert fw["exterior_term_background"].startswith("none (the grid ends inside the domain sphere")
    assert fw["background"]["physical_volume_outside_grid_fraction"] > 0.1   # 0.566 measured
    assert fw["background"]["grid_edge_A_over_max"] > 1e-3                    # 0.342
    assert fw["background"]["exterior"]["radial_nodes"] == 0
    assert any("ends inside the domain sphere" in w for w in r.data["warnings"])
    assert any("lies outside the background grid" in w for w in r.data["warnings"])
    row = fw["sweep"][0]
    assert row["Psi_background"]["exterior"] is None
    assert "exterior_last_order_fraction" not in row["Psi_background"]
    for port in row.get("port_solutions", []):
        assert port["Psi_background"]["exterior"] is None
