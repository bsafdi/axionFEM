"""Axisymmetric electrostatics (design key `electrostatics`) against closed forms.

- isolated sphere: C = 4 pi eps0 a (exact with the open exterior map)
- concentric spheres: the full Maxwell matrix
- sphere with a dielectric shell: the layered spherical capacitor
- sphere inside a grounded sphere
- two spheres on the axis: the bispherical series, matrix symmetry, energy and charge checks
- the readout gap of a coaxial pickup: the rigorous Dirichlet bound and the self-resonance estimate

Which checks are rigorous, and the boundary handling of the gap solve and of region names, are covered by
`test_exterior_flux_is_a_discretization_check`, `test_port_gap_dirichlet_bound`,
`test_gap_solve_honours_the_boundary`, `test_region_named_air_is_rejected` and
`test_corner_touching_conductors_are_one_body`.
"""
import math

import pytest

import axionfem as af

MU0 = 1.25663706212e-6
EPS0 = 1.0 / (MU0 * 299792458.0**2)
FPE = 4 * math.pi * EPS0


def disk(rho, zc=0.0):
    return {"disk": {"center": [0.0, zc], "radius": rho}}


def shell(r_in, r_out):
    return {"difference": [disk(r_out), disk(r_in)]}


def solve(tmp_path, name, regions, materials=None, mesh=None, boundary="open"):
    doc = {"schema": "axionfem.design/1", "name": name, "regions": regions,
           "electrostatics": {"boundary": boundary},
           "mesh": mesh or {"order": 2, "size": 0.1, "surface_size": 0.02, "domain_radius": 2.0}}
    if materials:
        doc["materials"] = materials
    r = af.Design(doc).solve(tmp_path / name, verbose=0)
    return r


def body(e, name):
    return [b["name"] for b in e["bodies"]].index(name)


def assert_consistent(e, sym=1e-9, energy=1e-10, gauss=2e-3, mono=1e-6, slack=-1e-6, flux=2e-2):
    c = e["checks"]
    # algebraic identities of the discrete system (they verify the assembly and the solves)
    assert c["symmetry_relative"] < sym
    assert c["energy_max_relative_difference"] < energy
    assert c["energy_superposition"]["relative_difference"] < energy
    assert c["exterior_monopole_max_relative_difference"] < mono
    # genuine discretization checks
    assert c["charge_gauss_max_relative_difference"] < gauss
    assert c["exterior_flux_max_relative_difference"] < flux
    assert c["off_diagonal_signs_ok"]
    assert c["min_slack_relative"] > slack


def test_isolated_sphere(tmp_path):
    a = 0.5
    r = solve(tmp_path, "sphere", [{"name": "ball", "shape": disk(a), "material": "superconductor"}])
    e = r.data["electrostatics"]
    assert e["boundary"] == "open"
    assert [b["name"] for b in e["bodies"]] == ["ball"]
    assert e["capacitance_F"][0][0] == pytest.approx(FPE * a, rel=1e-5)
    assert e["capacitance_to_ground_F"][0] == pytest.approx(FPE * a, rel=1e-5)
    assert_consistent(e)
    assert not r.data["warnings"]
    # a conductor-material body is a conductor here too
    r2 = solve(tmp_path, "sphere_cu", [{"name": "ball", "shape": disk(a), "material": "copper"}])
    assert r2.data["electrostatics"]["capacitance_F"][0][0] == pytest.approx(FPE * a, rel=1e-5)
    # a design with no magnet and no pickup still renders and summarises
    assert r.scene()["schema"].startswith("axionfem.scene")
    assert "C[ball]" in r.summary()
    assert r.C_matrix == r.electrostatics["capacitance_F"]
    assert r.port_capacitance == {}


def test_concentric_spheres(tmp_path):
    a, b, c = 0.3, 0.5, 0.6
    r = solve(tmp_path, "concentric",
              [{"name": "inner", "shape": disk(a), "material": "superconductor"},
               {"name": "outer", "shape": shell(b, c), "material": "superconductor"}])
    e = r.data["electrostatics"]
    i, o = body(e, "inner"), body(e, "outer")
    C = e["capacitance_F"]
    Cin = FPE / (1 / a - 1 / b)
    assert C[i][i] == pytest.approx(Cin, rel=1e-4)
    assert C[i][o] == pytest.approx(-Cin, rel=1e-4)
    assert C[o][o] == pytest.approx(Cin + FPE * c, rel=1e-4)
    # the inner conductor is fully screened: its row sums to zero
    assert abs(e["capacitance_to_ground_F"][i]) < 1e-6 * Cin
    # the mutual capacitance of the pair (two conductors: grounded and floating agree)
    p = e["pairs"][0]
    assert p["C_mutual_others_grounded_F"] == pytest.approx(Cin, rel=1e-4)
    assert p["C_mutual_others_floating_F"] == pytest.approx(Cin, rel=1e-4)
    assert_consistent(e)


def test_dielectric_shell(tmp_path):
    a, b, eps_r = 0.3, 0.5, 4.0
    r = solve(tmp_path, "shell",
              [{"name": "ball", "shape": disk(a), "material": "superconductor"},
               {"name": "diel", "shape": shell(a, b), "material": "dielectric"}],
              materials={"dielectric": {"kind": "linear", "eps_r": eps_r}})
    e = r.data["electrostatics"]
    exact = FPE / ((1 / a - 1 / b) / eps_r + 1 / b)
    assert e["capacitance_F"][0][0] == pytest.approx(exact, rel=1e-4)
    assert_consistent(e)


def test_grounded_outer_sphere(tmp_path):
    a, Rb = 0.3, 1.0
    r = solve(tmp_path, "grounded", [{"name": "ball", "shape": disk(a), "material": "superconductor"}],
              mesh={"order": 2, "size": 0.1, "surface_size": 0.02, "domain_radius": Rb}, boundary="grounded")
    e = r.data["electrostatics"]
    assert e["boundary"] == "grounded"
    assert e["solver"]["dtn_modes"] == 0
    assert e["capacitance_F"][0][0] == pytest.approx(FPE / (1 / a - 1 / Rb), rel=1e-4)
    assert_consistent(e, mono=0.0 + 1e-300)


def test_two_spheres_on_the_axis(tmp_path):
    a, D = 0.1, 0.5
    r = solve(tmp_path, "two",
              [{"name": "upper", "shape": disk(a, +D / 2), "material": "superconductor"},
               {"name": "lower", "shape": disk(a, -D / 2), "material": "superconductor"}],
              mesh={"order": 2, "size": 0.08, "surface_size": 0.015, "domain_radius": 2.0})
    e = r.data["electrostatics"]
    u, l = body(e, "upper"), body(e, "lower")
    C = e["capacitance_F"]
    eta = math.acosh(D / (2 * a))
    s = math.sinh(eta)
    c11 = FPE * a * s * sum(1 / math.sinh((2 * n + 1) * eta) for n in range(60))
    c12 = -FPE * a * s * sum(1 / math.sinh(2 * n * eta) for n in range(1, 60))
    assert C[u][u] == pytest.approx(c11, rel=1e-4)
    assert C[u][l] == pytest.approx(c12, rel=1e-4)
    assert C[u][u] == pytest.approx(C[l][l], rel=1e-6)   # equal spheres
    # the reaction charge and the surface flux integral differ at the discretization level (0.6 % on
    # this deliberately coarse mesh); the validation records show it falling with h
    assert_consistent(e, gauss=1e-2)
    # floating conductors: the solved potential must match the matrix prediction
    f = e["checks"]["floating_solve"]
    assert f["max_difference_V"] < 1e-9
    assert f["V_floating"][0] == pytest.approx(-C[u][l] / C[u][u], rel=1e-6)


def test_readout_gap_capacitance(tmp_path):
    """Coax shorted at the top with the readout gap at the bottom (the solenoid-pickup topology):
    the two gap estimates and the self-resonance."""
    a1, b1, t, g, L = 0.05, 0.15, 0.02, 0.01, 0.3
    coax = {"union": [{"rect": {"r": [a1 - t, a1], "z": [-L / 2, L / 2]}},
                      {"rect": {"r": [b1, b1 + t], "z": [-L / 2, L / 2]}},
                      {"rect": {"r": [a1 - t, b1 + t], "z": [L / 2 - t, L / 2]}}]}
    doc = {"schema": "axionfem.design/1", "name": "coax_gap",
           "regions": [{"name": "coax", "shape": coax, "material": "superconductor"},
                       {"name": "port", "shape": {"rect": {"r": [a1, b1], "z": [-L / 2, -L / 2 + g]}},
                        "material": "vacuum"},
                       {"name": "coil", "shape": {"rect": {"r": [0.3, 0.32], "z": [-0.1, 0.1]}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
           "pickup": {"type": "poloidal", "port": "port", "inside_point": [0.1, 0.0]},
           "mesh": {"order": 2, "size": 0.04, "surface_size": 0.005}}
    r = af.Design(doc).solve(tmp_path / "coax_gap", verbose=0)
    pc = r.data["pickup"]["port_capacitance"]
    assert pc["terminals_on_one_body"]
    # the port region alone is a coaxial capacitor of height g with insulating end faces:
    #   C = 2 pi eps0 g / ln(b/a)  (a lower bound: no fringing)
    assert pc["C_gap_port_region_F"] == pytest.approx(
        2 * math.pi * EPS0 * g / math.log(b1 / a1), rel=0.02)
    # fringing can only add capacitance (the domain of the variational problem is larger)
    assert pc["C_gap_F"] > pc["C_gap_port_region_F"] > 0
    # there is no `self_resonance_upper_bound_Hz`: that bound does not hold.  What is reported is the
    # region-only estimate under a neutral name.
    assert "self_resonance_upper_bound_Hz" not in pc
    assert pc["self_resonance_from_port_region_Hz"] > pc["self_resonance_Hz"]
    assert pc["self_resonance_Hz"] == pytest.approx(
        1 / (2 * math.pi * math.sqrt(r.L_p * pc["C_gap_F"])), rel=1e-12)
    assert pc["checks"]["gap_symmetry_relative"] < 1e-9
    assert pc["checks"]["region_symmetry_relative"] < 1e-9
    # the only rigorous relation between the two values
    db = pc["checks"]["dirichlet_bound"]
    assert db["holds"]
    assert db["C_gap_plus_quarter_terminal_to_ground_F"] == pytest.approx(
        db["C_gap_F"] + 0.25 * sum(db["C_terminal_to_ground_F"]), rel=1e-12)
    assert db["C_gap_plus_quarter_terminal_to_ground_F"] >= db["C_gap_port_region_F"]


def test_zero_thickness_sheet_conductor(tmp_path):
    """A conductor sheet (a `curves` entry) shaped as a spherical shell: C = 4 pi eps0 a."""
    a = 0.4
    doc = {"schema": "axionfem.design/1", "name": "sheet_sphere",
           "regions": [],
           "curves": [{"name": "shell",
                       "path": {"arc": {"center": [0.0, 0.0], "radius": a,
                                        "angles": [-math.pi / 2, math.pi / 2]}},
                       "material": "superconductor", "mesh_size": 0.01}],
           "electrostatics": {"boundary": "open"},
           "mesh": {"order": 3, "size": 0.05, "domain_radius": 2.0}}
    r = af.Design(doc).solve(tmp_path / "sheet", verbose=0)
    e = r.data["electrostatics"]
    assert [b["name"] for b in e["bodies"]] == ["shell"]
    assert e["capacitance_F"][0][0] == pytest.approx(FPE * a, rel=1e-4)
    assert_consistent(e, gauss=1e-3)


def test_electrostatics_needs_a_conductor(tmp_path):
    doc = {"schema": "axionfem.design/1", "name": "no_conductor",
           "regions": [{"name": "blob", "shape": disk(0.2), "material": "vacuum"}],
           "electrostatics": {},
           "mesh": {"order": 2, "size": 0.1, "domain_radius": 1.0}}
    with pytest.raises(Exception, match="conductor"):
        af.Design(doc).solve(tmp_path / "nc", verbose=0)


def test_unknown_boundary_is_rejected():
    doc = {"schema": "axionfem.design/1", "name": "bad",
           "regions": [{"name": "ball", "shape": disk(0.2), "material": "superconductor"}],
           "electrostatics": {"boundary": "floating"}}
    with pytest.raises(Exception, match="open"):
        af.Design(doc).validate()


# ---------------------------------------------------------------- which checks are rigorous

def test_exterior_flux_is_a_discretization_check(tmp_path):
    """`exterior_monopole_*` is an algebraic identity (1^T K phi = eps_out p_0), not a
    discretization check -- it saturates while the true error falls.  `exterior_flux_*` compares the
    weakly imposed DtN map with the strong flux -int eps grad(phi).n dA and does converge."""
    a = 0.5
    got = {}
    for order in (1, 2, 3):
        r = solve(tmp_path, f"sph_p{order}", [{"name": "ball", "shape": disk(a), "material": "superconductor"}],
                  mesh={"order": order, "size": 0.1, "surface_size": 0.03, "domain_radius": 2.0})
        c = r.data["electrostatics"]["checks"]
        got[order] = (abs(r.data["electrostatics"]["capacitance_F"][0][0] / (FPE * a) - 1),
                      c["exterior_monopole_max_relative_difference"],
                      c["exterior_flux_max_relative_difference"])
    # the true error of C falls by ~10^5 from order 1 to order 3
    assert got[1][0] > 1e-4 > got[2][0] > got[3][0]
    # the flux check tracks it: it falls by more than 10^3 and by >10x from order 2 to order 3
    assert got[1][2] > 1e-2
    assert got[3][2] < 1e-3 * got[1][2]
    assert got[3][2] < 0.1 * got[2][2]
    # the monopole "check" does not: it is the SAME number at order 2 and order 3 (it has already hit the
    # floor set by the solver residual and the geometry of the boundary arc) while the real error of C
    # falls by two more decades.  That is why it is not a discretization check.
    assert got[3][1] == pytest.approx(got[2][1], rel=1e-6)
    assert got[3][0] < 0.1 * got[2][0]
    assert got[1][2] > 10 * got[1][1]   # at order 1 the real error is >10x what the identity reports


def test_port_gap_dirichlet_bound(tmp_path):
    """C_gap >= C_gap_port_region is not proven. What is rigorous (and exact for the discrete
    problems) is C_gap_port_region <= C_gap + (C_Ag + C_Bg)/4, and it is loose."""
    a1, b1, t, L = 0.05, 0.15, 0.02, 0.3
    coax = {"union": [{"rect": {"r": [a1 - t, a1], "z": [-L / 2, L / 2]}},
                      {"rect": {"r": [b1, b1 + t], "z": [-L / 2, L / 2]}},
                      {"rect": {"r": [a1 - t, b1 + t], "z": [L / 2 - t, L / 2]}}]}
    ratios = {}
    for g in (0.005, 0.05):
        doc = {"schema": "axionfem.design/1", "name": "cx",
               "regions": [{"name": "coax", "shape": coax, "material": "superconductor"},
                           {"name": "port", "shape": {"rect": {"r": [a1, b1], "z": [-L / 2, -L / 2 + g]}},
                            "material": "vacuum"},
                           {"name": "coil", "shape": {"rect": {"r": [0.3, 0.32], "z": [-0.1, 0.1]}},
                            "material": "winding"}],
               "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
               "pickup": {"type": "poloidal", "port": "port", "inside_point": [0.1, 0.0]},
               "mesh": {"order": 2, "size": 0.04, "surface_size": 0.005}}
        r = af.Design(doc).solve(tmp_path / f"cx{g}", verbose=0)
        db = r.data["pickup"]["port_capacitance"]["checks"]["dirichlet_bound"]
        assert db["holds"]
        assert db["bound_over_C_region"] >= 1.0
        ratios[g] = (db["C_gap_over_C_region"], db["bound_over_C_region"])
    # The extra quarter-term is NOT small: at the narrow gap it is comparable to C_gap itself, and it
    # grows (relative to C_gap) as the gap narrows.  That is why the simple inequality C_gap >= C_region
    # cannot be read off from this bound.
    for g, (cg, bd) in ratios.items():
        assert (bd - cg) > 0.3 * cg
    assert (ratios[0.005][1] - ratios[0.005][0]) / ratios[0.005][0] > \
           (ratios[0.05][1] - ratios[0.05][0]) / ratios[0.05][0]
    assert ratios[0.005][1] > ratios[0.05][1]


# ---------------------------------------------------------------- boundaries, region names, touching conductors

def test_gap_solve_honours_the_boundary(tmp_path):
    """The readout-gap solve honours `electrostatics.boundary` (not always an open exterior)."""
    a1, b1, t, g, L = 0.05, 0.15, 0.02, 0.05, 0.3
    coax = {"union": [{"rect": {"r": [a1 - t, a1], "z": [-L / 2, L / 2]}},
                      {"rect": {"r": [b1, b1 + t], "z": [-L / 2, L / 2]}},
                      {"rect": {"r": [a1 - t, b1 + t], "z": [L / 2 - t, L / 2]}}]}
    out = {}
    for bnd in ("open", "grounded"):
        doc = {"schema": "axionfem.design/1", "name": "cx",
               "regions": [{"name": "coax", "shape": coax, "material": "superconductor"},
                           {"name": "port", "shape": {"rect": {"r": [a1, b1], "z": [-L / 2, -L / 2 + g]}},
                            "material": "vacuum"},
                           {"name": "coil", "shape": {"rect": {"r": [0.3, 0.32], "z": [-0.1, 0.1]}},
                            "material": "winding"}],
               "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
               "pickup": {"type": "poloidal", "port": "port", "inside_point": [0.1, 0.0]},
               "electrostatics": {"boundary": bnd},
               "mesh": {"order": 2, "size": 0.04, "surface_size": 0.005}}
        r = af.Design(doc).solve(tmp_path / f"bnd{bnd}", verbose=0)
        pc = r.data["pickup"]["port_capacitance"]
        out[bnd] = (pc["gap_solver"]["boundary"], pc["C_gap_F"], pc["C_terminal_to_ground_F"])
        assert r.data["electrostatics"]["boundary"] == bnd
    assert out["open"][0] == "open" and out["grounded"][0] == "grounded"
    # grounding the outer sphere adds capacitance to ground and changes the reduced gap value
    assert out["grounded"][1] != out["open"][1]
    assert sum(out["grounded"][2]) > sum(out["open"][2])


def test_region_named_air_is_rejected():
    """A region named `air` would shadow the implicit background and be forced to vacuum (a dielectric
    shell named `air` would give the bare-sphere capacitance, -30 %, with no warning), so it is refused."""
    doc = {"schema": "axionfem.design/1", "name": "shadow",
           "materials": {"oil": {"kind": "linear", "eps_r": 4.0}},
           "regions": [{"name": "ball", "shape": disk(0.3), "material": "superconductor"},
                       {"name": "air", "shape": shell(0.3, 0.5), "material": "oil"}],
           "electrostatics": {"boundary": "open"}}
    with pytest.raises(Exception, match="reserved region name"):
        af.Design(doc).validate()
    doc["regions"][1]["name"] = "infinity_shell"
    with pytest.raises(Exception, match="reserved region name"):
        af.Design(doc).validate()
    # a curve may not take the name either
    doc2 = {"schema": "axionfem.design/1", "name": "shadow2",
            "curves": [{"name": "air", "path": {"segment": [[0.1, 0.0], [0.2, 0.0]]},
                        "material": "superconductor"}]}
    with pytest.raises(Exception, match="reserved region name"):
        af.Design(doc2).validate()


def test_corner_touching_conductors_are_one_body(tmp_path):
    """Two conductor regions meeting at a corner must not raise a spurious "touch" error.  In
    axisymmetry a shared node is a whole circle of contact, and the H1 space forces one potential
    there, so they are one body."""
    mesh = {"order": 2, "size": 0.1, "surface_size": 0.02, "domain_radius": 2.0}
    corner = [{"name": "a", "shape": {"rect": {"r": [0.1, 0.3], "z": [0.0, 0.2]}},
               "material": "superconductor"},
              {"name": "b", "shape": {"rect": {"r": [0.3, 0.5], "z": [0.2, 0.4]}},
               "material": "superconductor"}]
    r = solve(tmp_path, "corner", corner, mesh=mesh)
    e = r.data["electrostatics"]
    assert len(e["bodies"]) == 1
    assert sorted(e["bodies"][0]["regions"]) == ["a", "b"]
    assert e["capacitance_F"][0][0] > 0
    assert_consistent(e, gauss=1e-1, flux=1e-1)   # re-entrant corners: a coarse, singular geometry
    # face-touching regions still merge, as before
    face = [{"name": "a", "shape": {"rect": {"r": [0.1, 0.3], "z": [0.0, 0.2]}},
             "material": "superconductor"},
            {"name": "b", "shape": {"rect": {"r": [0.1, 0.3], "z": [0.2, 0.4]}},
             "material": "superconductor"}]
    r2 = solve(tmp_path, "face", face, mesh=mesh)
    assert len(r2.data["electrostatics"]["bodies"]) == 1
