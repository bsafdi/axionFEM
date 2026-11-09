"""Axisymmetric linear elasticity against closed-form references.

- Lamé thick-walled cylinder under internal pressure, with the axial ends fixed (plane strain,
  u_z = 0) and free (generalized plane stress, sigma_zz = 0);
- long thin solenoid: hoop stress from the Lorentz body force against the magnetic pressure
  B^2/2 mu0 transmitted through a thin shell, and the axial self-compression -p pi R^2 / A_w;
- superconducting thin shell under the Maxwell surface pressure, cross-checked against the
  independent surface-force post-processing of model.cpp;
- uniform cooldown of a free ring: zero stress and u = -alpha x;
- axial equilibrium closure (applied loads + support reactions = 0).
"""
import math

import numpy as np
import pytest

import axionfem as af

MU0 = 1.25663706212e-6
E, NU = 200e9, 0.3
A, B, H = 0.1, 0.3, 4.0      # Lamé cylinder: inner/outer radius, length
P = 1.0e6                    # internal pressure [Pa]


def _lame(fix_z, size=0.02, order=2):
    doc = {"schema": "axionfem.design/1", "name": "lame",
           "materials": {"steel": {"kind": "linear", "mechanical": {"E_Pa": E, "nu": NU, "alpha": 0.0}}},
           "regions": [
               {"name": "bore", "shape": {"rect": {"r": [0.0, A], "z": [-H / 2, H / 2]}}, "material": "vacuum"},
               {"name": "cyl", "shape": {"rect": {"r": [A, B], "z": [-H / 2, H / 2]}}, "material": "steel",
                "mesh_size": size},
               # a 1 A coil: the solver needs a magnet, its Lorentz load is ~1e-4 Pa
               {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mechanics": {"pressures": [{"on": "cyl", "from": "bore", "p_Pa": P}]},
           "mesh": {"order": order, "size": 0.3, "domain_radius": 12.0}}
    if fix_z:
        doc["mechanics"]["supports"] = [{"region": "cyl", "fix": "z"}]
    return af.Design(doc)


def _lame_exact(r, plane_strain):
    k = P * A ** 2 / (B ** 2 - A ** 2)
    s_rr = k * (1 - B ** 2 / r ** 2)
    s_tt = k * (1 + B ** 2 / r ** 2)
    if plane_strain:
        return (1 + NU) / E * k * ((1 - 2 * NU) * r + B ** 2 / r), s_rr, s_tt, NU * (s_rr + s_tt)
    return k / E * ((1 - NU) * r + (1 + NU) * B ** 2 / r), s_rr, s_tt, np.zeros_like(r)


@pytest.mark.parametrize("fix_z", [True, False], ids=["ends_fixed", "ends_free"])
def test_lame_thick_walled_cylinder(tmp_path, fix_z):
    res = _lame(fix_z).solve(tmp_path / f"lame_{fix_z}", verbose=0)
    w = B - A
    r = np.linspace(A + 0.1 * w, B - 0.1 * w, 9)
    s = res.sample_mech(r, np.zeros_like(r))
    ur, s_rr, s_tt, s_zz = _lame_exact(r, fix_z)
    assert np.max(np.abs(s["u_r"] / ur - 1)) < 1e-3
    assert np.max(np.abs(s["sigma_hoop"] / s_tt - 1)) < 5e-3
    assert np.max(np.abs(s["sigma_rr"] - s_rr)) < 1e-2 * P
    assert np.max(np.abs(s["sigma_zz"] - s_zz)) < 2e-2 * P
    if fix_z:
        assert np.max(np.abs(s["u_z"])) == 0.0            # u_z is a Dirichlet zero everywhere
        assert s["sigma_zz"][4] == pytest.approx(2 * NU * P * A ** 2 / (B ** 2 - A ** 2), rel=1e-3)
        sup = res.mechanics["supports"]
        assert len(sup) == 1 and sup[0]["region"] == "cyl" and sup[0]["fix"] == "z"
        # the ends carry no axial traction, so the support supplies nothing
        assert abs(sup[0]["reaction_Fz_N"]) < 1e-6 * P * A ** 2
    ck = res.mechanics["checks"]
    assert ck["equilibrium_Fz_round_off"] < 1e-10 and ck["equilibrium_Fz_relative"] < 1e-10
    # the pressure acts on r = A over the full length: 2 pi int t_r r dl / 2 pi = P A H
    assert ck["applied_radial_N_per_rad"] == pytest.approx(P * A * H, rel=1e-6)
    assert res.mechanics["regions"]["cyl"]["E_Pa"] == E


def test_lame_mesh_convergence(tmp_path):
    """The hoop stress converges towards the exact Lamé value under refinement."""
    w = B - A
    r = np.linspace(A + 0.1 * w, B - 0.1 * w, 9)
    _, _, s_tt, _ = _lame_exact(r, True)
    errs = []
    for size in (0.08, 0.04, 0.02):
        res = _lame(True, size=size).solve(tmp_path / f"conv_{size}", verbose=0)
        s = res.sample_mech(r, np.zeros_like(r))
        errs.append(float(np.max(np.abs(s["sigma_hoop"] / s_tt - 1))))
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 5e-3


def test_solenoid_winding_hoop_stress(tmp_path):
    """Long thin solenoid: the Lorentz body force gives the thin-shell hoop stress p Rm/t with
    p = B^2/2 mu0, and the axial self-attraction gives sigma_zz = -p pi R^2 / (2 pi Rm t)."""
    R, t, Hs, NI = 0.5, 0.005, 10.0, 1.0e6
    doc = {"schema": "axionfem.design/1", "name": "sol",
           "materials": {"coil": {"kind": "vacuum", "mechanical": {"E_Pa": 100e9, "nu": 0.3}}},
           "regions": [{"name": "w", "shape": {"rect": {"r": [R, R + t], "z": [-Hs / 2, Hs / 2]}},
                        "material": "coil", "mesh_size": t}],
           "windings": [{"region": "w", "type": "azimuthal", "ampere_turns": NI}],
           "mechanics": {},
           "mesh": {"order": 2, "size": 0.5, "domain_radius": 30.0}}
    res = af.Design(doc).solve(tmp_path / "sol", verbose=0)
    f = res.sample([R - 1e-5, R + t + 1e-5], [0.0, 0.0])
    p = (f["Bz"][0] ** 2 - f["Bz"][1] ** 2) / (2 * MU0)
    assert abs(f["Bz"][0]) == pytest.approx(MU0 * NI / Hs, rel=1e-2)  # long solenoid
    rm = R + t / 2
    s = res.sample_mech(np.linspace(R + 0.1 * t, R + 0.9 * t, 5), np.zeros(5))
    assert float(np.mean(s["sigma_hoop"])) == pytest.approx(p * rm / t, rel=1e-2)
    # axial: the two halves attract with p pi R^2, carried by the winding cross-section 2 pi rm t
    assert float(np.mean(s["sigma_zz"])) == pytest.approx(-p * R ** 2 / (2 * rm * t), rel=2e-2)
    assert res.mechanics["checks"]["equilibrium_Fz_round_off"] < 1e-10
    assert res.mechanics["checks"]["equilibrium_Fz_relative"] < 1e-9
    assert res.mechanics["lorentz_windings"] == ["w"]


def test_superconductor_magnetic_pressure(tmp_path):
    """A zero-field-cooled superconducting tube inside a solenoid: the Maxwell surface pressure
    B^2/2 mu0 compresses it to the thin-shell hoop stress -p Rm/t, and the axial load agrees with
    the independent surface-force post-processing."""
    a, t, Ls, Hs, NI = 0.5, 0.005, 8.0, 12.0, 1.0e6
    doc = {"schema": "axionfem.design/1", "name": "shell",
           "materials": {"nbm": {"kind": "superconductor", "mechanical": {"E_Pa": 105e9, "nu": 0.4}}},
           "regions": [{"name": "sh", "shape": {"rect": {"r": [a, a + t], "z": [-Ls / 2, Ls / 2]}},
                        "material": "nbm", "mesh_size": t},
                       {"name": "w", "shape": {"rect": {"r": [0.8, 0.81], "z": [-Hs / 2, Hs / 2]}},
                        "material": "winding"}],
           "windings": [{"region": "w", "type": "azimuthal", "ampere_turns": NI}],
           "superconductors": [{"body": "sh", "azimuthal": {"flux": 0.0}}],
           "mechanics": {},
           "mesh": {"order": 2, "size": 0.5, "surface_size": t, "domain_radius": 40.0}}
    res = af.Design(doc).solve(tmp_path / "shell", verbose=0)
    f = res.sample([a - 1e-5, a + t + 1e-5], [0.0, 0.0])
    assert abs(f["Bz"][0]) < 1e-10                         # screened bore
    p = (f["Bz"][1] ** 2 - f["Bz"][0] ** 2) / (2 * MU0)    # external (compressive) pressure
    rm = a + t / 2
    s = res.sample_mech(np.linspace(a + 0.1 * t, a + 0.9 * t, 5), np.zeros(5))
    assert float(np.mean(s["sigma_hoop"])) == pytest.approx(-p * rm / t, rel=1e-2)
    assert res.mechanics["magnetic_pressure_bodies"] == ["sh"]
    # the same surface integral as results["forces"], from an independent quadrature
    fz_post = res.data["forces"]["superconductor:sh"]["Fz_N"]
    assert res.mechanics["checks"]["applied_Fz_N"] == pytest.approx(fz_post, rel=1e-3)
    assert res.mechanics["checks"]["equilibrium_Fz_round_off"] < 1e-10


def test_free_ring_cooldown(tmp_path):
    """A free ring contracting uniformly: u = -alpha x and zero stress."""
    alpha = 2.5e-3
    doc = {"schema": "axionfem.design/1", "name": "ring",
           "materials": {"m": {"kind": "linear", "mechanical": {"E_Pa": 120e9, "nu": 0.33, "alpha": alpha}}},
           "regions": [{"name": "ring", "shape": {"rect": {"r": [0.4, 0.6], "z": [-0.1, 0.1]}},
                        "material": "m", "mesh_size": 0.03},
                       {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mechanics": {"cooldown": True},
           "mesh": {"order": 2, "size": 0.3, "domain_radius": 12.0}}
    res = af.Design(doc).solve(tmp_path / "ring", verbose=0)
    r = np.array([0.42, 0.5, 0.58, 0.45, 0.55])
    z = np.array([0.02, 0.05, -0.08, 0.09, -0.02])
    s = res.sample_mech(r, z)
    assert np.allclose(s["u_r"], -alpha * r, rtol=1e-9)
    assert np.allclose(s["u_z"], -alpha * z, rtol=1e-9)   # the ring is symmetric about z = 0
    reg = res.mechanics["regions"]["ring"]
    assert reg["von_mises_max_Pa"] < 1e-9 * E * alpha      # stress free
    assert reg["u_max_m"] == pytest.approx(alpha * math.hypot(0.6, 0.1), rel=1e-9)
    assert reg["thermal_strain"] == pytest.approx(-alpha)
    assert res.mechanics["checks"]["equilibrium_Fz_round_off"] < 1e-10


def test_cooldown_of_a_clamped_ring_is_in_tension(tmp_path):
    """Clamping the inner face against the cooldown puts the ring in hoop tension; the support
    reaction then balances the applied axial load exactly."""
    alpha = 2.0e-3
    doc = {"schema": "axionfem.design/1", "name": "clamped",
           "materials": {"m": {"kind": "linear", "mechanical": {"E_Pa": 100e9, "nu": 0.3, "alpha": alpha}},
                         "rigid": {"kind": "linear", "mechanical": {"E_Pa": 100e9, "nu": 0.3, "alpha": 0.0}}},
           "regions": [{"name": "hub", "shape": {"rect": {"r": [0.38, 0.4], "z": [-0.1, 0.1]}},
                        "material": "rigid", "mesh_size": 0.02},
                       {"name": "ring", "shape": {"rect": {"r": [0.4, 0.6], "z": [-0.1, 0.1]}},
                        "material": "m", "mesh_size": 0.02},
                       {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mechanics": {"cooldown": True, "supports": [{"region": "hub", "fix": "all"}]},
           "mesh": {"order": 2, "size": 0.3, "domain_radius": 12.0}}
    res = af.Design(doc).solve(tmp_path / "clamped", verbose=0)
    reg = res.mechanics["regions"]["ring"]
    # restrained contraction: the hoop stress is of order E alpha / (1 - nu)
    assert reg["sigma_hoop_Pa"]["max"] > 0.3 * 100e9 * alpha
    ck = res.mechanics["checks"]
    assert ck["equilibrium_Fz_round_off"] < 1e-10
    sup = res.mechanics["supports"][0]
    assert sup["reaction_Fz_N"] == pytest.approx(-ck["applied_Fz_N"], abs=1e-6 * ck["assembly_scale_N"])
    assert "touching regions share mesh nodes" in res.mechanics["contact"]


@pytest.mark.parametrize("name,winding", [("toroid", "toroid"), ("solenoid", "solenoid")])
def test_template_lorentz_load_matches_the_force_postprocessing(tmp_path, name, winding):
    """On the toroid and solenoid templates the Lorentz body force integrated by the elasticity assembly must
    reproduce the winding force computed independently in results["forces"].  This covers the
    poloidal-winding path (toroid: J x B_phi) and the azimuthal one (solenoid: J_phi x B_pol)."""
    d = af.template(name)
    d.doc["mesh"]["size"] = "max(R, H)/14"
    d.doc["mechanics"] = {"cooldown": True}
    res = d.solve(tmp_path / name, verbose=0)
    ck, fw = res.mechanics["checks"], res.data["forces"][winding]
    assert ck["applied_radial_N_per_rad"] == pytest.approx(fw["radial_force_per_radian_N"], rel=1e-9)
    # the elasticity load is the full J x B, including the winding's own field (real internal stress, discrete
    # net residual); results["forces"] reports it as Fz_N (and Fz_excluding_self_N without the self-force)
    raw_fz = fw["Fz_N"]
    assert ck["applied_Fz_N"] == pytest.approx(raw_fz, abs=1e-6 * abs(fw["radial_force_per_radian_N"]))
    assert ck["equilibrium_Fz_round_off"] < 1e-10
    # the cooldown dominates the displacement; the winding carries the magnetic hoop load
    assert res.mechanics["regions"]["winding"]["von_mises_max_Pa"] > 1e5
    assert res.mechanics["lorentz_windings"] == [winding]
    s = res.sample_mech([0.0], [0.0])
    assert s["in_structure"][0] == 0.0           # the bore is not part of the structure


def test_builtin_material_defaults_and_errors(tmp_path):
    """Built-in materials carry documented structural defaults; bad input is rejected clearly."""
    doc = {"schema": "axionfem.design/1", "name": "defaults",
           "regions": [{"name": "shell", "shape": {"rect": {"r": [0.4, 0.42], "z": [-0.2, 0.2]}},
                        "material": "nb", "mesh_size": 0.02},
                       {"name": "coil", "shape": {"rect": {"r": [0.2, 0.22], "z": [-0.2, 0.2]}},
                        "material": "winding", "mesh_size": 0.02}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e5}],
           "mechanics": {"cooldown": True},
           "mesh": {"order": 2, "size": 0.2, "domain_radius": 4.0}}
    res = af.Design(doc).solve(tmp_path / "defaults", verbose=0)
    rg = res.mechanics["regions"]
    assert rg["shell"]["E_Pa"] == pytest.approx(105e9)     # niobium
    assert rg["shell"]["thermal_strain"] == pytest.approx(-1.43e-3)
    assert rg["coil"]["E_Pa"] == pytest.approx(80e9)       # smeared impregnated winding
    assert res.mechanics["checks"]["equilibrium_Fz_round_off"] < 1e-10

    bad = dict(doc)
    bad["materials"] = {"x": {"kind": "linear", "mechanical": {"E_Pa": 1e9, "nu": 0.5}}}
    with pytest.raises(Exception, match="Poisson"):
        af.Design(bad).validate()
    bad2 = dict(doc, mechanics={"supports": [{"region": "nope"}]})
    with pytest.raises(Exception, match="unknown region"):
        af.Design(bad2).validate()
    bad3 = dict(doc, mechanics={"pressures": [{"on": "shell", "from": "coil", "p_Pa": 1.0}]})
    with pytest.raises(Exception, match="touches no mesh face"):
        af.Design(bad3).solve(tmp_path / "bad3", verbose=0)


# ================================================================ superconducting sheets, supports, warnings

def _sc_sheet_on_tube(tmp_path, name, sheet_r, coil_mat="winding"):
    """A steel tube with a zero-thickness superconducting sheet at `sheet_r`, inside a longer solenoid."""
    a, t, L, NI = 0.5, 0.01, 8.0, 1.0e6
    doc = {"schema": "axionfem.design/1", "name": "sheet",
           "materials": {"st": {"kind": "linear", "mechanical": {"E_Pa": 200e9, "nu": 0.3}},
                         "cv": {"kind": "vacuum"}},
           "regions": [{"name": "tube", "shape": {"rect": {"r": [a, a + t], "z": [-L / 2, L / 2]}},
                        "material": "st", "mesh_size": t},
                       {"name": "w", "shape": {"rect": {"r": [0.8, 0.81], "z": [-6, 6]}},
                        "material": coil_mat}],
           "curves": [{"name": "skin", "path": {"segment": [[sheet_r, -L / 2], [sheet_r, L / 2]]},
                       "material": "nb"}],
           "windings": [{"region": "w", "type": "azimuthal", "ampere_turns": NI}],
           "superconductors": [{"body": "skin", "azimuthal": {"flux": 0.0}}],
           "mechanics": {},
           "mesh": {"order": 2, "size": 0.4, "surface_size": t, "domain_radius": 40.0}}
    return af.Design(doc).solve(tmp_path / name, verbose=0)


def test_superconducting_sheet_transmits_the_maxwell_pressure(tmp_path):
    """A zero-thickness SC `curve` carries the screening current, so its magnetic surface traction must reach
    the structure it lies on: a steel tube with an SC skin screening 0.1045 T has the thin-shell hoop stress
    -p b/t = -2.2e5 Pa (dropping the traction would give exactly 0)."""
    a, t = 0.5, 0.01
    b = a + t
    res = _sc_sheet_on_tube(tmp_path, "sheet_outer", b)   # sheet on the OUTER surface: field outside
    assert res.mechanics["magnetic_pressure_sheets"] == ["skin"]
    B_in = float(res.sample([a + 0.5 * t], [0.0])["Bz"][0])
    B_out = float(res.sample([b + 0.01], [0.0])["Bz"][0])
    assert abs(B_in) < 1e-9 and abs(B_out) > 0.1        # the sheet screens the wall
    p = (B_out ** 2 - B_in ** 2) / (2 * MU0)
    rr = np.linspace(a + 0.1 * t, b - 0.1 * t, 9)
    s = res.sample_mech(rr, np.zeros_like(rr))
    # exact thick wall, external pressure p on r = b, free ends:  k = -p b^2/(b^2 - a^2)
    k = -p * b ** 2 / (b ** 2 - a ** 2)
    assert np.max(np.abs(np.asarray(s["sigma_hoop"]) / (k * (1 + a ** 2 / rr ** 2)) - 1)) < 1e-3
    mean_hoop = float(np.trapezoid(np.asarray(s["sigma_hoop"]), rr) / (rr[-1] - rr[0]))
    assert mean_hoop == pytest.approx(-p * b / t, rel=1e-3)   # thin-shell membrane value
    # the dropped (B.n) B part of the Maxwell traction is negligible on a superconductor
    assert res.mechanics["checks"]["maxwell_traction_Bn_over_B_max"] < 1e-9
    assert any("IS applied to the structure" in w for w in res.data["warnings"])


def test_superconducting_sheet_with_no_structure_says_so(tmp_path):
    """A free-standing sheet has nothing to push on; that must be said, not implied."""
    res = _sc_sheet_on_tube(tmp_path, "sheet_free", 0.65, coil_mat="cv")
    assert res.mechanics["magnetic_pressure_sheets"] == []
    assert res.mechanics["regions"]["tube"]["von_mises_max_Pa"] == 0.0
    assert any("applied nowhere" in w for w in res.data["warnings"])


def _bar(supports, cooldown=False, pressures=None, nu=0.3, alpha=2e-3, size=0.05):
    a2, b2, L2 = 0.4, 0.5, 2.0
    mech = {"E_Pa": 200e9, "nu": nu}
    if alpha is not None:
        mech["alpha"] = alpha
    doc = {"schema": "axionfem.design/1", "name": "bar",
           "materials": {"st": {"kind": "linear", "mechanical": mech}},
           "regions": [{"name": "bore", "shape": {"rect": {"r": [0, a2], "z": [-L2 / 2, L2 / 2]}},
                        "material": "vacuum"},
                       {"name": "bar", "shape": {"rect": {"r": [a2, b2], "z": [-L2 / 2, L2 / 2]}},
                        "material": "st", "mesh_size": size},
                       {"name": "pad", "shape": {"rect": {"r": [a2, b2], "z": [L2 / 2, L2 / 2 + 0.1]}},
                        "material": "vacuum"},
                       {"name": "coil", "shape": {"rect": {"r": [3.0, 3.05], "z": [-0.05, 0.05]}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mechanics": {"cooldown": cooldown,
                         "pressures": pressures if pressures is not None
                         else [{"on": "bar", "from": "bore", "p_Pa": 1.0e7}],
                         "supports": supports},
           "mesh": {"order": 2, "size": 0.3, "domain_radius": 12.0}}
    return doc


def test_duplicate_support_on_one_region_is_rejected(tmp_path):
    """ElasticRole::support holds one group index, so a second entry on the same region would silently
    replace the first ([{bar, r}, {bar, z}] would leave u_r unconstrained): it is refused."""
    r = af.Design(_bar([{"region": "bar", "fix": "all"}], cooldown=True)).solve(tmp_path / "all", verbose=0)
    s = r.sample_mech([0.45], [0.3])
    assert abs(float(s["u_r"][0])) < 1e-15 and abs(float(s["u_z"][0])) < 1e-15
    for sup in ([{"region": "bar", "fix": "r"}, {"region": "bar", "fix": "z"}],
                [{"region": "bar", "fix": "z"}, {"region": "bar", "fix": "r"}],
                [{"region": "bar", "fix": "all"}, {"region": "bar", "fix": "all"}]):
        with pytest.raises(Exception, match="already has a support"):
            af.Design(_bar(sup, cooldown=True)).solve(tmp_path / "dup", verbose=0)


def test_near_incompressible_locking_warning(tmp_path):
    """The pure-displacement formulation locks as nu -> 1/2; a warning must say so."""
    for nu, want in ((0.40, False), (0.46, True), (0.49, True)):
        r = af.Design(_bar([{"region": "bar", "fix": "z"}], nu=nu)).solve(tmp_path / f"nu{nu}", verbose=0)
        assert any("locking-free" in w for w in r.data["warnings"]) is want


def test_free_assembly_axial_warning_uses_the_axial_load_scale(tmp_path):
    """The warning scales a net AXIAL load by the axial load scale: load_scale_N mixes |f_r| and |f_z|, so a
    real 28 kN axial imbalance would hide behind a large radial pressure, and round-off of dummy 1 A-turn
    coils would fire it."""
    a2, b2 = 0.4, 0.5
    press = [{"on": "bar", "from": "bore", "p_Pa": 1.0e7}, {"on": "bar", "from": "pad", "p_Pa": 1.0e5}]
    r = af.Design(_bar([], pressures=press, size=0.025)).solve(tmp_path / "loaded", verbose=0)
    ck = r.mechanics["checks"]
    exact = -1.0e5 * math.pi * (b2 ** 2 - a2 ** 2)
    assert ck["applied_Fz_N"] == pytest.approx(exact, rel=1e-6)
    assert ck["axial_load_scale_N"] == pytest.approx(abs(exact), rel=1e-6)
    assert abs(ck["applied_Fz_N"]) < 1e-3 * ck["load_scale_N"]      # invisible to the old threshold
    assert any("net axial load" in w for w in r.data["warnings"])
    # no axial load at all: no warning (round-off must not fire it)
    r0 = af.Design(_bar([], pressures=[{"on": "bar", "from": "bore", "p_Pa": 1.0e7}],
                        size=0.025)).solve(tmp_path / "radial", verbose=0)
    ck0 = r0.mechanics["checks"]
    assert abs(ck0["applied_Fz_N"]) < 1e-6
    assert not any("net axial load" in w for w in r0.data["warnings"])


def test_cooldown_without_alpha_warns(tmp_path):
    """A cooldown with a material that has no `alpha` (so alpha = 0) warns."""
    r = af.Design(_bar([{"region": "bar", "fix": "z"}], cooldown=True, alpha=None)).solve(
        tmp_path / "noalpha", verbose=0)
    assert any("no (or zero) \"alpha\"" in w for w in r.data["warnings"])
    assert r.mechanics["regions"]["bar"]["thermal_strain"] == 0.0
    # with alpha given, no such warning
    r2 = af.Design(_bar([{"region": "bar", "fix": "z"}], cooldown=True, alpha=2e-3)).solve(
        tmp_path / "alpha", verbose=0)
    assert not any("alpha" in w for w in r2.data["warnings"])


def test_sample_mech_is_nan_outside_the_domains(tmp_path):
    """sample_mech returns NaN, not 0 (a legitimate value), for points where the quantity is not defined
    (outside the structure or outside the thermal domain)."""
    doc = _bar([{"region": "bar", "fix": "z"}])
    doc["materials"]["st"]["thermal"] = {"k_W_mK": 10.0}
    doc["thermal"] = {"fixed": [{"region": "bar", "T_K": 4.0}]}
    doc["regions"][3]["material"] = "cv"
    doc["materials"]["cv"] = {"kind": "vacuum"}
    r = af.Design(doc).solve(tmp_path / "nan", verbose=0)
    s = r.sample_mech([0.2, 0.45, 50.0], [0.0, 0.0, 0.0])
    assert math.isnan(float(s["u_r"][0])) and math.isnan(float(s["T_K"][0]))   # the vacuum bore
    assert math.isnan(float(s["u_r"][2])) and math.isnan(float(s["T_K"][2]))   # outside the mesh
    assert not math.isnan(float(s["u_r"][1]))                                  # the structure
    assert float(s["T_K"][1]) == pytest.approx(4.0, abs=1e-9)
    assert list(np.asarray(s["in_structure"])) == [0.0, 1.0, 0.0]
