"""The axisymmetric engine's exterior term for a grid background (docs/physics/fullwave_background.md, "The exterior
term"; validation docs/validation/axi_background_exterior.md, driver benchmarks/axi_background_exterior.py).

The model is the 3D validation's coax-in-a-solenoid (benchmarks/fullwave3d_background.py) on the axisymmetric engine:
the coil's analytic field as a grid background (the coil region kept, as a mu_r = 1 material, so the mesh is the
winding model's), or a 1.8 m magnet of the same radii and current density, whose field crosses the domain sphere.

Default suite: the coil as a coarse grid against the coil as the design's winding on the same mesh (the winding's
exterior term comes from its static multipoles), and the domain-radius independence that the term restores; the same
with a superconducting disk next to the coax (a reaction field, with its own exterior term, next to the grid's); a grid
that ends inside the domain sphere or is cut at its box (labels and warnings); an eigenmode-only run (no sweep labels).
The moments' cut-at-box path and their thread-count independence are tested in C++
(cpp/tests/test_background_exterior.cpp). AXF_SLOW: the long magnet against the 3D record
(benchmarks/records/fullwave3d_background.json) and against the magnet as a 2D winding inside a domain that encloses it,
its domain-radius independence, the MQS-limit rate and the convergence in the multipole order.
"""
import functools
import json
import math
import os
import sys
from pathlib import Path

import pytest

import axionfem as af

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import fullwave3d_background as fb  # noqa: E402

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="slow; set AXF_SLOW=1")
REC3D = Path(__file__).resolve().parents[1] / "benchmarks" / "records" / "fullwave3d_background.json"


def _psi(row):
    return complex(row["Psi_re"], row["Psi_im"])


def _cx(d):
    return complex(d["re"], d["im"])


def _solve(path, doc):
    return af.Design(doc).solve(path, verbose=0)


@functools.lru_cache(maxsize=None)
def _coarse_grid_json():
    """The coil's field on a coarse grid (the continuation from 2 m), cheap to sample (a JSON string: callers copy)."""
    return json.dumps(fb.grid(2.0, 2.0, h_fine=0.005, h=0.025))


def _coarse_grid():
    return json.loads(_coarse_grid_json())


def _crop(g, r_max, z_lo, z_hi):
    """The grid cut to r <= r_max, z_lo <= z <= z_hi (the field beyond is then taken as zero)."""
    nr = len(g["r"])
    ir = [i for i, r in enumerate(g["r"]) if r <= r_max + 1e-12]
    iz = [j for j, z in enumerate(g["z"]) if z_lo - 1e-12 <= z <= z_hi + 1e-12]
    return {"type": "grid", "r": [g["r"][i] for i in ir], "z": [g["z"][j] for j in iz],
            "Br": [g["Br"][j * nr + i] for j in iz for i in ir],
            "Bz_grid": [g["Bz_grid"][j * nr + i] for j in iz for i in ir]}


def test_grid_background_exterior_term_matches_the_winding(tmp_path):
    """Above the MQS limit the grid's exterior term is 4 % (kR = 0.4) and 15 % (kR = 0.8) of Psi. With it, the coil as a
    grid background gives the coil-as-winding Psi on the same mesh (which carries the same integral as its static
    multipoles' exterior term), and Psi no longer depends on the domain radius."""
    kr = (0.02, 0.4, 0.8)
    grid = _coarse_grid()
    wnd = _solve(tmp_path / "windings", fb.design(True, None, fb.R1, kr, axisymmetric=True))
    bg = _solve(tmp_path / "grid", fb.design(False, grid, fb.R1, kr, axisymmetric=True))
    big = _solve(tmp_path / "grid_R1x1.5", fb.design(False, grid, 1.5 * fb.R1, kr, axisymmetric=True))
    fw = bg.fullwave
    assert fw["exterior_term"] == "included"   # the reaction field's term (zero here: copper is static-transparent)
    assert fw["exterior_term_background"] == "included (grid, then its exterior multipoles)"
    ext = fw["background"]["exterior"]
    assert ext["continuation_radius_m"] == pytest.approx(2.0)
    assert ext["quadrature_relative_change_doubled_panels"] < 1e-4
    assert ext["n_max"] == 80
    assert fw["background"]["physical_volume_outside_grid_fraction"] == 0
    assert not any("exterior reciprocity term" in w or "grid" in w for w in bg.data["warnings"])
    # no background: none of the background's outputs
    assert "background" not in wnd.fullwave and "exterior_term_background" not in wnd.fullwave
    for a, b, c in zip(wnd.fullwave["sweep"], fw["sweep"], big.fullwave["sweep"]):
        pa, pb, pc = _psi(a), _psi(b), _psi(c)
        ea = _cx(a["Psi_exterior"])
        eb = _cx(b["Psi_background"]["exterior"])
        ec = _cx(c["Psi_background"]["exterior"])
        vb = _cx(b["Psi_background"]["volume"])
        print(f"kR {b['k_R']:.2f}: Psi winding {pa:.7e}, grid {pb:.7e} (|d| {abs(pb - pa) / abs(pa):.1e}); exterior "
              f"winding {ea:.5e}, grid {eb:.5e} (|d| {abs(eb - ea) / abs(ea):.1e}); R1 x1.5 {abs(pc - pb) / abs(pb):.1e}, "
              f"without the term {abs((pc - ec) - (pb - eb)) / abs(pb):.1e}")
        assert _cx(b["Psi_exterior"]) == 0   # no reaction field
        assert abs(vb + eb - pb) < 1e-12 * abs(pb)   # both parts are in Psi
        assert abs(eb - ea) < 1e-3 * abs(ea)   # the same integral, from the grid and from the static multipoles
        assert abs(pb - pa) < 2e-4 * abs(pa)
        assert abs(pc - pb) < 2e-4 * abs(pb)
        if b["k_R"] > 0.1:   # without the term: 2e-2 and 1e-1
            assert abs((pc - ec) - (pb - eb)) > 50 * abs(pc - pb)
            assert abs(eb) > 0.03 * abs(pb)
        # the coil is symmetric about the domain centre: only odd orders, so the diagnostic reads the last two orders
        assert 0 < b["Psi_background"]["exterior_last_order_fraction"] < 1e-12


def test_reaction_field_next_to_a_grid_background(tmp_path):
    """A superconducting disk above the coax screens the coil's field. With the coil as a grid, A_lab is the disk's
    reaction alone, with its own exterior term (Psi_exterior), and the grid has its term; the two add up to the single
    exterior term of the coil as the design's winding (one A_lab), and Psi matches it on the same mesh."""
    kr = (0.02, 0.4, 0.8)
    grid = _coarse_grid()
    wnd = _solve(tmp_path / "windings", fb.design(True, None, fb.R1, kr, axisymmetric=True, plug=True))
    bg = _solve(tmp_path / "grid", fb.design(False, grid, fb.R1, kr, axisymmetric=True, plug=True))
    for a, b in zip(wnd.fullwave["sweep"], bg.fullwave["sweep"]):
        pa, pb = _psi(a), _psi(b)
        ea = _cx(a["Psi_exterior"])
        e_lab = _cx(b["Psi_exterior"])
        e_bg = _cx(b["Psi_background"]["exterior"])
        print(f"kR {b['k_R']:.2f}: Psi winding {pa:.7e}, grid {pb:.7e} (|d| {abs(pb - pa) / abs(pa):.1e}); exterior "
              f"winding {ea:.4e} = reaction {e_lab:.4e} + grid {e_bg:.4e} (|d| {abs(e_lab + e_bg - ea) / abs(ea):.1e})")
        assert abs(pb - pa) < 2e-4 * abs(pa)
        assert abs(e_lab + e_bg - ea) < 1e-3 * abs(ea)
        if b["k_R"] > 0.1:
            assert abs(e_lab) > 2e-3 * abs(pb)   # the reaction's own term is not negligible (0.3 % and 1.3 %)


def test_grid_that_does_not_reach_past_the_domain_sphere(tmp_path):
    """The coil's grid cut to r <= 0.4 m, |z| <= 0.4 m (corners at 0.57 m): at R1 = 0.6 m it ends inside the domain
    sphere, so the magnet's field beyond it is coupled nowhere; at R1 = 0.45 m its inscribed sphere (0.4 m) lies inside
    the domain sphere and the term is cut at its box. Both are labelled and warned about, and Psi(omega) differs from
    the full grid's by the missing field (16 % at kR = 0.8 in the validation)."""
    kr = (0.8,)
    cropped = _crop(_coarse_grid(), 0.4, -0.4, 0.4)
    inside = _solve(tmp_path / "inside", fb.design(False, cropped, 0.6, kr, axisymmetric=True))
    full = _solve(tmp_path / "full", fb.design(False, _coarse_grid(), 0.6, kr, axisymmetric=True))
    cut = _solve(tmp_path / "cut", fb.design(False, cropped, fb.R1, kr, axisymmetric=True))
    fw = inside.fullwave
    print("inside:", fw["exterior_term_background"], fw["background"]["physical_volume_outside_grid_fraction"],
          fw["background"]["grid_edge_A_over_max"], inside.data["warnings"])
    assert fw["exterior_term_background"].startswith("none (the grid ends inside the domain sphere")
    assert fw["background"]["physical_volume_outside_grid_fraction"] > 1e-3
    assert fw["background"]["grid_edge_A_over_max"] > 1e-3
    assert any("ends inside the domain sphere" in w for w in inside.data["warnings"])
    assert any("lies outside the background grid" in w for w in inside.data["warnings"])
    assert fw["sweep"][0]["Psi_background"]["exterior"] is None
    assert "exterior_last_order_fraction" not in fw["sweep"][0]["Psi_background"]
    d = abs(_psi(fw["sweep"][0]) - _psi(full.fullwave["sweep"][0])) / abs(_psi(full.fullwave["sweep"][0]))
    print(f"Psi, cut grid against the full grid at kR 0.8: {d:.2e}")
    assert d > 0.05
    assert not any("grid" in w for w in full.data["warnings"])
    print("cut:", cut.fullwave["exterior_term_background"], cut.data["warnings"])
    assert cut.fullwave["exterior_term_background"] == "included (grid only: cut at its box)"
    assert any("inscribed sphere lies inside the domain sphere" in w for w in cut.data["warnings"])
    assert any("lies outside the background grid" in w for w in cut.data["warnings"])


def test_uniform_background_keeps_the_static_fields_exterior_term(tmp_path):
    """With a uniform background the background's own exterior term is omitted (it diverges), but A_lab's, the reaction
    of a superconducting disk above the coax, is included (as in 3D). The same field given as a 1 T
    grid has the same domain integral (psi = r^2/2 is exact on the grid) and the same reaction, so its Psi_exterior must
    equal the uniform run's, and its Psi minus its background term must equal the uniform run's Psi."""
    kr = (0.4, 0.8)
    uniform = fb.design(False, {"type": "uniform", "Bz": 1.0}, fb.R1, kr, axisymmetric=True, plug=True)
    n = 41
    as_grid = {"type": "grid", "r": [2.0 * i / (n - 1) for i in range(n)],
               "z": [-2.0 + 4.0 * j / (2 * n - 2) for j in range(2 * n - 1)],
               "Br": [0.0] * (n * (2 * n - 1)), "Bz_grid": [1.0] * (n * (2 * n - 1))}
    u = _solve(tmp_path / "uniform", uniform)
    g = _solve(tmp_path / "grid", fb.design(False, as_grid, fb.R1, kr, axisymmetric=True, plug=True))
    assert u.fullwave["exterior_term"] == "included"
    assert u.fullwave["exterior_term_background"].startswith("omitted (uniform background")
    for a, b in zip(u.fullwave["sweep"], g.fullwave["sweep"]):
        pa = _psi(a)
        ea = _cx(a["Psi_exterior"])
        eb = _cx(b["Psi_exterior"])
        rest = _psi(b) - _cx(b["Psi_background"]["exterior"])
        print(f"kR {a['k_R']:.2f}: reaction term uniform {ea:.6e}, as grid {eb:.6e}; Psi uniform {pa:.7e}, grid minus "
              f"its background term {rest:.7e}; term / Psi {abs(ea) / abs(pa):.2e}")
        assert a["Psi_background"]["exterior"] is None
        assert abs(ea - eb) < 1e-10 * abs(ea)
        assert abs(rest - pa) < 1e-12 * abs(pa)
        assert abs(ea) > 3e-3 * abs(pa)   # 0.66 % and 1.8 %


def test_eigenmode_only_run_reports_no_sweep_labels(tmp_path):
    """Without frequencies there is no Psi(omega): no exterior labels, background block or exterior warnings."""
    doc = fb.design(False, _coarse_grid(), fb.R1, (0.4,), axisymmetric=True)
    doc["fullwave"] = {"modes": 2, "boundary": "radiation"}
    fw = _solve(tmp_path / "eig", doc).fullwave
    for key in ("exterior_term", "exterior_term_background", "background", "sweep"):
        assert key not in fw


# ---------------------------------------------------------------- AXF_SLOW
def _long_grid():
    return fb.grid(2.0, 3.0, z1=-fb.LONG_HALF, z2=fb.LONG_HALF, fine_z=1.1, h_fine=0.004, h=0.02)


def _long_winding(domain_radius, kr):
    """The long magnet as the 2D design's winding (the domain must enclose it)."""
    doc = fb.design(True, None, domain_radius, kr, axisymmetric=True)
    for region in doc["regions"]:
        if region["name"] == "winding":
            region["shape"] = {"rect": {"r": [fb.R, fb.R + fb.TW], "z": [-fb.LONG_HALF, fb.LONG_HALF]}}
    for winding in doc["windings"]:
        winding["ampere_turns"] = fb.NI * 2 * fb.LONG_HALF / fb.H
    return doc


@slow
def test_long_magnet_against_3d_and_the_2d_winding(tmp_path):
    """The 1.8 m magnet crosses the domain sphere; its exterior term is 12 % (kR = 0.4) and 39 % (kR = 0.8) of Psi. With it
    the 2D Psi agrees with the 3D record (to the 3D mesh's level, which L shows) and with the magnet as a 2D winding in a
    domain that encloses it (R1 = 1.2 m), and does not depend on the domain radius."""
    kr = fb.KR
    grid = _long_grid()
    rows = {}
    for rb in (fb.R1, 1.5 * fb.R1, 2.0 * fb.R1):
        rows[rb] = _solve(tmp_path / f"grid_{rb}", fb.design(False, grid, rb, kr, axisymmetric=True)).fullwave["sweep"]
    ref = _solve(tmp_path / "winding", _long_winding(1.2, kr)).fullwave["sweep"]
    rec = json.loads(REC3D.read_text())["models"]
    d3 = {name: rec[name]["rows"] for name in ("B_long", "B_long_R1x1.5")}
    for i, k in enumerate(kr):
        a = rows[fb.R1][i]
        pa = _psi(a)
        ea = _cx(a["Psi_background"]["exterior"])
        p_ref = _psi(ref[i])
        assert ref[i]["f_Hz"] == pytest.approx(a["f_Hz"], rel=1e-12)
        for rb, name in ((fb.R1, "B_long"), (1.5 * fb.R1, "B_long_R1x1.5")):
            r3 = d3[name][i]
            p3 = complex(*r3["Psi"])
            p2 = _psi(rows[rb][i])
            e3 = complex(*r3["Psi_bg_exterior"])
            e2 = _cx(rows[rb][i]["Psi_background"]["exterior"])
            dl = r3["L_H"] / rows[rb][i]["L_H"] - 1
            print(f"kR {k}: R1 {rb:.3f}: Psi 2D {p2:.6e}, 3D {p3:.6e} (|d| {abs(p2 - p3) / abs(p3):.1e}, L {dl:.1e}); "
                  f"exterior 2D {e2:.4e}, 3D {e3:.4e} (|d| {abs(e2 - e3) / abs(e3):.1e}); without it "
                  f"{abs(p2 - e2 - p3) / abs(p3):.1e}")
            assert abs(p2 - p3) < max(1.5e-2, 2 * abs(dl)) * abs(p3)
            assert abs(e2 - e3) < 3e-2 * abs(e3)
        for rb in (1.5 * fb.R1, 2.0 * fb.R1):
            pb = _psi(rows[rb][i])
            eb = _cx(rows[rb][i]["Psi_background"]["exterior"])
            print(f"kR {k}: R1 {rb:.3f} against {fb.R1}: {abs(pb - pa) / abs(pa):.1e}, without the term "
                  f"{abs((pb - eb) - (pa - ea)) / abs(pa):.1e}")
            assert abs(pb - pa) < 3e-4 * abs(pa)   # the mesh: L changes by up to 2e-5 between these radii
            assert abs((pb - eb) - (pa - ea)) > 30 * abs(pb - pa)
        print(f"kR {k}: grid (R1 {fb.R1}) against the 2D winding (R1 1.2): {abs(pa - p_ref) / abs(p_ref):.1e}")
        assert abs(pa - p_ref) < 2e-4 * abs(p_ref)
        if k > 0.1:
            assert abs(ea) > 0.1 * abs(pa)


@slow
def test_exterior_term_vanishes_in_the_mqs_limit(tmp_path):
    """The exterior term is O((kR)^2) in its real part and O((kR)^3) in its imaginary part (H' = O(omega^2); the
    radiating part of the moments O(k))."""
    kr = (0.005, 0.01, 0.02, 0.04)
    grid = _long_grid()
    r = _solve(tmp_path / "mqs", fb.design(False, grid, fb.R1, kr, axisymmetric=True))
    psi0 = r.coupling["Psi_SI_H_T_m2"]
    ext = [_cx(row["Psi_background"]["exterior"]) for row in r.fullwave["sweep"]]
    for i in range(1, len(kr)):
        q = kr[i] / kr[i - 1]
        re_order = math.log(abs(ext[i].real / ext[i - 1].real)) / math.log(q)
        im_order = math.log(abs(ext[i].imag / ext[i - 1].imag)) / math.log(q)
        print(f"kR {kr[i]}: exterior / Psi_MQS {abs(ext[i]) / psi0:.3e}; orders Re {re_order:.4f}, Im {im_order:.4f}")
        assert re_order == pytest.approx(2.0, abs=0.01)
        assert im_order == pytest.approx(3.0, abs=0.01)
    assert abs(ext[0]) < 1e-4 * psi0


@slow
def test_exterior_term_converges_in_the_multipole_order(tmp_path):
    """fullwave.dtn_modes sets the exterior map's order and the exterior term's: Psi converges with it."""
    kr = (0.8,)
    grid = _long_grid()
    psi = {}
    last = {}
    for n in (8, 20, 40, 80):
        doc = fb.design(False, grid, fb.R1, kr, axisymmetric=True)
        doc["fullwave"]["dtn_modes"] = n
        row = _solve(tmp_path / f"n{n}", doc).fullwave["sweep"][0]
        psi[n] = _psi(row)
        last[n] = row["Psi_background"]["exterior_last_order_fraction"]
        print(f"dtn_modes {n}: Psi {psi[n]:.8e}, last two orders' fraction {last[n]:.1e}")
    assert abs(psi[8] - psi[80]) < 1e-6 * abs(psi[80])   # 9e-8 in the benchmark
    # the diagnostic sees the truncation although the magnet has no even orders (the last order alone read 1e-20)
    assert last[8] > 1e-9
    assert last[80] < last[20] < last[8]
    assert abs(psi[20] - psi[80]) < 1e-9 * abs(psi[80])
    assert abs(psi[40] - psi[80]) < 1e-9 * abs(psi[80])
