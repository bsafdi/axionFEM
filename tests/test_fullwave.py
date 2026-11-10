"""Full-wave (beyond MQS) axisymmetric solver against exact references.

- perfectly conducting sphere: TE0/TM0 resonances (zeros of j_n and of [x j_n(x)]')
- small superconducting loop: radiation resistance of a uniform-current loop
- sealed coaxial cavity with the readout port between two shorted stubs: transmission-line impedance
  and the TEM resonance
- toroid and solenoid templates: Psi(omega) and L(omega) reduce to the MQS values with (kR)^2 corrections
"""
import json
import math

import pytest

import axionfem as af

C = 299792458.0
ETA = 376.730313668  # free-space impedance (ohm)


def _sphere(tmp_path, h=0.06):
    doc = {"schema": "axionfem.design/1", "name": "sphere",
           "regions": [{"name": "box", "shape": {"rect": {"r": [0.0, 0.6], "z": [-0.6, 0.6]}}, "material": "vacuum",
                        "mesh_size": h},
                       {"name": "coil", "shape": {"rect": {"r": [0.62, 0.63], "z": [-0.01, 0.01]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1}],
           "fullwave": {"modes": 3, "families": ["TE", "TM"]},
           "mesh": {"order": 2, "size": h, "surface_size": h, "domain_radius": 1.0}}
    return af.Design(doc).solve(tmp_path / "sphere", verbose=0)


def test_sphere_cavity_modes(tmp_path):
    r = _sphere(tmp_path)
    eig = r.data["fullwave"]["eigenmodes"]
    exact = {"TE": [4.493409, 5.763459, 6.987932],   # j_n(x) = 0, n = 1, 2, 3
             "TM": [2.743707, 3.870239, 4.973420]}   # [x j_n(x)]' = 0, n = 1, 2, 3
    for fam, xs in exact.items():
        got = [m["f_Hz"] * 2 * math.pi / C for m in eig[fam]["modes"]]
        assert len(got) == 3
        for g, x in zip(got, xs):
            assert g == pytest.approx(x, rel=1e-5), fam


def test_small_loop_radiation_resistance(tmp_path):
    scipy = pytest.importorskip("scipy")
    from scipy.integrate import quad
    from scipy.special import jv
    a, b = 0.5, 0.0025
    kas = [0.05, 0.5, 1.5]
    doc = {"schema": "axionfem.design/1", "name": "loop",
           "regions": [{"name": "ring", "shape": {"disk": {"center": [a, 0.0], "radius": b}}, "material": "superconductor"},
                       {"name": "coil", "shape": {"rect": {"r": [0.1, 0.11], "z": [-0.01, 0.01]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1}],
           "superconductors": [{"body": "ring", "azimuthal_slit": True}],
           "pickup": {"type": "azimuthal", "conductor": "ring"},
           "fullwave": {"frequencies": [ka * C / (2 * math.pi * a) for ka in kas]},
           "mesh": {"order": 2, "size": 0.05, "surface_size": 0.001, "domain_radius": 1.0}}
    r = af.Design(doc).solve(tmp_path / "loop", verbose=0)
    L0 = r.coupling["L_p_H"]
    assert L0 == pytest.approx(4e-7 * math.pi * a * (math.log(8 * a / b) - 2), rel=3e-4)  # thin superconducting ring
    for ka, row in zip(kas, r.data["fullwave"]["sweep"]):
        # uniform-current loop: R = eta (pi/2) ka int_0^{2ka} J_2(x) dx; b/a = 0.005 leaves ~3e-4
        R_exact = ETA * math.pi / 2 * ka * quad(lambda x: jv(2, x), 0, 2 * ka)[0]
        assert row["R_rad_ohm"] == pytest.approx(R_exact, rel=1.5e-3)
        assert row["power_balance_rel"] < 1e-10
        assert row["solver"]["factorization"] == "LDLT"
    assert r.data["fullwave"]["sweep"][0]["L_over_MQS"] == pytest.approx(1.0, abs=1e-3)
    assert not r.data["warnings"]


def test_sealed_coax_transmission_line(tmp_path):
    a1, b1, t, g, l1, l2 = 0.05, 0.15, 0.02, 0.005, 0.6, 0.3
    freqs = [1e6, 50e6, 150e6]
    body = {"union": [{"rect": {"r": [0.0, a1], "z": [-l2 - t, g + l1 + t]}},
                      {"rect": {"r": [b1, b1 + t], "z": [-l2 - t, g + l1 + t]}},
                      {"rect": {"r": [a1, b1], "z": [g + l1, g + l1 + t]}},
                      {"rect": {"r": [a1, b1], "z": [-l2 - t, -l2]}}]}
    doc = {"schema": "axionfem.design/1", "name": "coax",
           "regions": [{"name": "coax", "shape": body, "material": "superconductor"},
                       {"name": "port", "shape": {"rect": {"r": [a1, b1], "z": [0.0, g]}}, "material": "vacuum"},
                       {"name": "coil", "shape": {"rect": {"r": [0.3, 0.32], "z": [-0.1, 0.1]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
           "pickup": {"type": "poloidal", "port": "port", "inside_point": [0.1, 0.3]},
           "fullwave": {"frequencies": freqs, "modes": 6, "families": ["TM"]},
           "mesh": {"order": 2, "size": 0.03, "surface_size": 0.0075}}
    r = af.Design(doc).solve(tmp_path / "coax", verbose=0)
    fw = r.data["fullwave"]
    assert fw["enclosed_cavities"] == 1
    Z0 = ETA / (2 * math.pi) * math.log(b1 / a1)
    L1, L2 = l1 + g / 2, l2 + g / 2
    L_tl = Z0 / C * L1 * L2 / (L1 + L2)
    L0 = r.coupling["L_p_H"]
    assert L0 == pytest.approx(L_tl, rel=1e-2)
    for f, row in zip(freqs, fw["sweep"]):
        k = 2 * math.pi * f / C
        t1, t2 = math.tan(k * L1), math.tan(k * L2)
        ratio_tl = (t1 * t2 / (t1 + t2)) / (k * L1 * L2 / (L1 + L2))  # Z / (-i omega L), shorted stubs in parallel
        assert row["L_over_MQS"] == pytest.approx(ratio_tl, rel=5e-3)
        assert abs(row["R_rad_ohm"]) < 1e-9  # sealed: no radiation
    f_tem = C / (2 * (l1 + l2 + g))
    assert min(abs(m["f_Hz"] / f_tem - 1) for m in fw["eigenmodes"]["TM"]["modes"]) < 1e-4


@pytest.mark.parametrize("name", ["solenoid", "toroid"])
def test_mqs_limit_of_templates(tmp_path, name):
    d = af.template(name)
    # surface-loss data on the readout conductors (checks the RF loss integral against the MQS one)
    mats = d.doc.setdefault("materials", {})
    rf = {"model": "power", "Rs_ohm": 1e-6, "f_ref": 1e6, "exponent": 2, "T_K": 0.01}
    for rg in d.doc["regions"]:
        if rg.get("material") in ("nb", "superconductor", "copper"):
            kind = "conductor" if rg["material"] == "copper" else "superconductor"
            mats[rg["material"] + "_rf"] = {"kind": kind, "rf": rf}
            rg["material"] += "_rf"
    r0 = d.solve(tmp_path / "mqs", verbose=0)
    R = r0.coupling["device_radius_m"]
    kRs = [1e-3, 1e-2]
    d.doc["fullwave"] = {"frequencies": [k * C / (2 * math.pi * R) for k in kRs]}
    r = d.solve(tmp_path / "fw", verbose=0)
    s = r.data["fullwave"]["sweep"]
    for row in s:
        assert row["power_balance_rel"] < 1e-10
    dpsi = [row["Psi_over_MQS"] - 1 for row in s]
    dl = [row["L_over_MQS"] - 1 for row in s]
    assert abs(dpsi[0]) < 1e-5 and abs(dl[0]) < 1e-5
    # corrections grow as (kR)^2
    assert dpsi[1] / dpsi[0] == pytest.approx(100, rel=0.1)
    assert dl[1] / dl[0] == pytest.approx(100, rel=0.1)
    # RF loss integrals Gamma = int |H|^2 dA / I^2 reduce to the MQS pickup values
    G_mqs = {k: v["Gamma"] for k, v in r.data["pickup"]["losses"]["surfaces"].items()}
    assert G_mqs
    for k, G in G_mqs.items():
        assert s[0]["surface_Gamma"][k] == pytest.approx(G, rel=2e-5)
    assert s[0]["Q_total"] == pytest.approx(1 / (1 / s[0]["Q_loss"] + 1 / s[0]["Q_rad"]), rel=1e-12)
    # RF field in the scene (first sweep frequency, kR = 1e-3): |B_rf| per ampere -> the MQS pickup field
    from axionfem.plots import _decode as _arr

    f = r.scene()["field"]
    assert f["rf_frequency_Hz"] == pytest.approx(s[0]["f_Hz"])
    rb, pb = _arr(f["quantities"]["rf_Bmag"]).ravel(), _arr(f["quantities"]["pickup_Bmag"]).ravel()
    big = pb > 1e-2 * pb.max()
    assert abs(rb[big] / pb[big] - 1).max() < 1e-3
    assert _arr(f["quantities"]["rf_Emag"]).max() > 0


# ---------------------------------------------------------------- losses in the full-wave fields
def _coax_doc(fill=None, freqs=(1e6, 50e6, 150e6)):
    a1, b1, t, g, l1, l2 = 0.05, 0.15, 0.02, 0.005, 0.6, 0.3
    body = {"union": [{"rect": {"r": [0.0, a1], "z": [-l2 - t, g + l1 + t]}},
                      {"rect": {"r": [b1, b1 + t], "z": [-l2 - t, g + l1 + t]}},
                      {"rect": {"r": [a1, b1], "z": [g + l1, g + l1 + t]}},
                      {"rect": {"r": [a1, b1], "z": [-l2 - t, -l2]}}]}
    regions = [{"name": "coax", "shape": body, "material": "superconductor"}]
    if fill:  # dielectric filling both stubs (the 5 mm port gap stays vacuum)
        regions += [{"name": "fill1", "shape": {"rect": {"r": [a1, b1], "z": [g, g + l1]}}, "material": "diel"},
                    {"name": "fill2", "shape": {"rect": {"r": [a1, b1], "z": [-l2, 0.0]}}, "material": "diel"}]
    regions += [{"name": "port", "shape": {"rect": {"r": [a1, b1], "z": [0.0, g]}}, "material": "vacuum"},
                {"name": "coil", "shape": {"rect": {"r": [0.3, 0.32], "z": [-0.1, 0.1]}}, "material": "winding"}]
    doc = {"schema": "axionfem.design/1", "name": "coax", "regions": regions,
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
           "pickup": {"type": "poloidal", "port": "port", "inside_point": [0.1, 0.3]},
           "fullwave": {"frequencies": list(freqs)},
           "mesh": {"order": 2, "size": 0.03, "surface_size": 0.0075}}
    if fill:
        doc["materials"] = {"diel": {"kind": "linear", **fill}}
    return doc, (a1, b1, g, l1, l2)


def test_lossy_dielectric_coax_matches_lossy_transmission_line(tmp_path):
    import cmath
    eps_r, tand = 2.2, 1e-3
    freqs = (1e6, 50e6, 100e6)
    doc, (a1, b1, g, l1, l2) = _coax_doc({"eps_r": eps_r, "tan_delta": tand}, freqs)
    r = af.Design(doc).solve(tmp_path / "lossy", verbose=0)
    from axionfem.circuit import fullwave_tables
    tab = fullwave_tables(r.data)
    for i, row in enumerate(r.data["fullwave"]["sweep"]):
        assert tab["R_loss_ohm"][i] == pytest.approx(row["R_bulk_ohm"] + row.get("R_loss_ohm", 0.))
        assert tab["R_rad_ohm"][i] == pytest.approx(row["R_rad_ohm"])
        assert tab["R_ohm"][i] == pytest.approx(row["Z_re_ohm"] + row.get("R_loss_ohm", 0.), rel=1e-7)
    ec = eps_r * (1 + 1j * tand)
    Z0 = ETA / (2 * math.pi) * math.log(b1 / a1) / cmath.sqrt(ec)
    Z0v = ETA / (2 * math.pi) * math.log(b1 / a1)
    for f, row in zip(freqs, r.data["fullwave"]["sweep"]):
        k, k0 = 2 * math.pi * f / C * cmath.sqrt(ec), 2 * math.pi * f / C

        def stub(l):  # dielectric shorted stub seen through half the vacuum port gap (e^{-i omega t} line formulas)
            ZL = -1j * Z0 * cmath.tan(k * l)
            t = cmath.tan(k0 * g / 2)
            return Z0v * (ZL - 1j * Z0v * t) / (Z0v - 1j * ZL * t)
        Z1, Z2 = stub(l1), stub(l2)
        Zt = Z1 * Z2 / (Z1 + Z2)
        Z = complex(row["Z_re_ohm"], row["Z_im_ohm"])
        assert abs(Z / Zt - 1) < 5e-3, (f, Z, Zt)
        assert Z.real > 0
        assert row["power_balance_rel"] < 1e-8           # dissipation in the dielectric = 1/2 Re Z
        assert "diel" in row["bulk_losses"]


def test_eddy_currents_conducting_sphere_in_loop(tmp_path):
    """A resistive sphere (radius s, sigma) at the centre of a superconducting loop: the induced impedance is
    -i omega mu0 alpha h0^2 with the exact magnetic polarizability of a conducting sphere,
    alpha = -2 pi s^3 [1 - 3/(k s)^2 + 3 cot(k s)/(k s)], k = (1 + i)/delta, and h0 the loop's field per ampere
    at the centre (higher multipoles enter at (s/a)^4)."""
    import cmath
    a, b, s, sigma = 0.5, 0.005, 0.05, 1e6
    freqs = [300.0, 3e3, 3e4]
    def doc(sphere_mat):
        return {"schema": "axionfem.design/1", "name": "eddy",
                "materials": {"steel": {"kind": "linear", "sigma_S_m": sigma}},
                "regions": [{"name": "ring", "shape": {"disk": {"center": [a, 0.0], "radius": b}}, "material": "superconductor"},
                            {"name": "ball", "shape": {"polygon": {"points": [[0.0, -s]] + [[s * math.sin(math.pi * i / 64), -s * math.cos(math.pi * i / 64)] for i in range(1, 64)] + [[0.0, s]]}},
                             "material": sphere_mat, "mesh_size": 0.0015},
                            {"name": "coil", "shape": {"rect": {"r": [0.2, 0.21], "z": [-0.3, -0.29]}}, "material": "winding"}],
                "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1}],
                "superconductors": [{"body": "ring", "azimuthal_slit": True}],
                "pickup": {"type": "azimuthal", "conductor": "ring"},
                "fullwave": {"frequencies": freqs},
                "mesh": {"order": 2, "size": 0.05, "surface_size": 0.002, "domain_radius": 1.0}}
    rv = af.Design(doc("vacuum")).solve(tmp_path / "vac", verbose=0)
    rl = af.Design(doc("steel")).solve(tmp_path / "steel", verbose=0)
    h0 = rv.sample([0.0], [0.0])["pickup_Bz"][0] / (4e-7 * math.pi)   # loop field per ampere at the centre
    for f, row_v, row_l in zip(freqs, rv.data["fullwave"]["sweep"], rl.data["fullwave"]["sweep"]):
        w = 2 * math.pi * f
        dZ = complex(row_l["Z_re_ohm"] - row_v["Z_re_ohm"], row_l["Z_im_ohm"] - row_v["Z_im_ohm"])
        delta = math.sqrt(2 / (w * 4e-7 * math.pi * sigma))
        x = (1 + 1j) / delta * s
        alpha = -2 * math.pi * s ** 3 * (1 - 3 / x ** 2 + 3 / (x * cmath.tan(x)))
        dZ_exact = -1j * w * 4e-7 * math.pi * alpha * h0 ** 2
        assert abs(dZ / dZ_exact - 1) < 3e-3, (f, dZ, dZ_exact)  # measured 6e-4 to 9e-4
        assert row_l["power_balance_rel"] < 1e-6


def test_magnetoresistance_weights_surface_resistance(tmp_path):
    # constant resistivity factor 4 on a skin-effect surface: R_s and the loss integral double
    d = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                    mesh={"order": 2, "size": 0.08, "surface_size": 0.03})
    base = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 2e-10, "T_K": 4.0}}
    mr = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 2e-10, "T_K": 4.0,
                                      "magnetoresistance": {"B_T": [0.0, 10.0], "factor": [4.0, 4.0]}}}
    out = []
    for m in (base, mr):
        dd = af.Design(json.loads(json.dumps(d.doc)))
        dd.doc.setdefault("materials", {})["cu"] = m
        for rg in dd.doc["regions"]:
            if rg["material"] == "copper":
                rg["material"] = "cu"
        out.append(dd.solve(tmp_path / f"mr{len(out)}", verbose=0).data["pickup"]["losses"]["surfaces"])
    for k in out[0]:
        assert out[1][k]["Gamma_magnetoresistance"] == pytest.approx(2 * out[0][k]["Gamma"], rel=1e-12)
        assert out[1][k]["R_eff_ohm"]["f_min"] == pytest.approx(2 * out[0][k]["R_eff_ohm"]["f_min"], rel=1e-12)


def test_tm_exterior_reciprocity_tail_removes_domain_dependence(tmp_path):
    """The TM reciprocity integral int A_lab . mu0 H' dV extends beyond the domain sphere: with the exterior
    multipole term, Psi(omega) no longer depends on the domain radius."""
    out = {}
    for Rb in (3.0, 4.5):
        d = af.template("solenoid", mesh={"domain_radius": Rb})
        R = 1.0
        d.doc["fullwave"] = {"frequencies": [k * C / (2 * math.pi * R) for k in (0.3, 0.6)]}
        out[Rb] = d.solve(tmp_path / f"rb{Rb}", verbose=0).data["fullwave"]["sweep"]
    for i in range(2):
        a, b = out[3.0][i], out[4.5][i]
        pa, pb = complex(a["Psi_re"], a["Psi_im"]), complex(b["Psi_re"], b["Psi_im"])
        ta = complex(a["Psi_exterior"]["re"], a["Psi_exterior"]["im"])
        tb = complex(b["Psi_exterior"]["re"], b["Psi_exterior"]["im"])
        with_tail = abs(pa - pb) / abs(pa)
        without = abs((pa - ta) - (pb - tb)) / abs(pa)
        print(f"kR row {i}: |tail|/|Psi| = {abs(ta) / abs(pa):.3e}, {abs(tb) / abs(pb):.3e}; "
              f"domain dependence with tail {with_tail:.3e}, without {without:.3e}")
        assert abs(ta) > abs(tb)  # a larger domain leaves less outside
        # a normalization bug leaves with_tail near without; what remains is the discretization difference of the
        # two meshes, 5e-5 to 2e-3 of without depending on the mesher
        assert with_tail < 1e-2 * without


def test_tm_exterior_term_omitted_with_background_is_reported(tmp_path):
    """A uniform background fills all space, so its own exterior integral diverges: that term is omitted, labelled and
    warned about (a grid background's is included: tests/test_axi_background_exterior.py). The design's own static
    field (here the template's winding) keeps its exterior term, as in 3D."""
    d = af.template("solenoid")
    d.doc["background"] = {"type": "uniform", "Bz": 1.0}
    d.doc["fullwave"] = {"frequencies": [0.1 * C / (2 * math.pi)]}
    r = d.solve(tmp_path / "bg", verbose=0)
    assert r.data["fullwave"]["exterior_term"] == "included"
    assert r.data["fullwave"]["exterior_term_background"].startswith("omitted (uniform background")
    assert any("exterior reciprocity term" in w and "give the magnet's field as a grid" in w
               for w in r.data["warnings"])
    row = r.data["fullwave"]["sweep"][0]
    assert row["Psi_background"]["exterior"] is None   # no background term computed
    psi = complex(row["Psi_re"], row["Psi_im"])
    ext = complex(row["Psi_exterior"]["re"], row["Psi_exterior"]["im"])
    assert abs(ext) > 2e-4 * abs(psi)   # the winding's term: 7.0e-4 of Psi at kR = 0.1 here, O((kR)^2)
