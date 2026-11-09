"""Cavity haloscopes in 3D (axionfem.cavity, beta): eigenmodes, shift-invert, form factor, Q0 and overlaps.

Closed PEC cavities (``mesh.domain_solid`` wall, ``fullwave.rf_only``, no pickup), ND order 2, coarse meshes.
References: the closed forms of an independent (blind) analytic derivation, through
``axionfem.cavity`` (cylinder_modes_3d, box_modes, coax_tm_mode, eta_tm010_te01p) and the axisymmetric solver.
Tight convergence tables: benchmarks/cavity_3d_validation.py -> docs/validation/cavity_3d.md.
The solves are 3D (seconds to a minute each on one core): set AXF_SLOW=1; the multi-rank test also needs a
SLURM allocation (srun).
"""
import math
import os

import pytest

import axionfem as af
from axionfem import cavity as cv

A, L = 0.1, 0.2
RHO = 1.7e-8
WALL = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": RHO, "T_K": 4.0}}
slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D cavity eigenmodes (slow); set AXF_SLOW=1")
# a target frequency selects the shift-invert eigensolver, which factorizes with STRUMPACK
needs_strumpack = pytest.mark.skipif(not af.build_info().get("strumpack"),
                                     reason="the 3D shift-invert eigensolver needs a STRUMPACK build")


def _modes(r):
    return r.data["fullwave"]["eigenmodes"]["modes"]


def _sorted_pairs(modes, ref):
    got = sorted(modes, key=lambda m: m["f_Hz"])
    assert len(ref) >= len(got)
    return list(zip(got, ref[:len(got)]))


# ------------------------------------------------------------------ spec (fast)
def test_cavity3d_spec_errors():
    d = cv.cylinder_design_3d(A, L)
    d.validate()
    d.doc["fullwave"]["rf_only"] = False
    d.doc["mesh"].pop("domain_solid")
    with pytest.raises(af.AxionFEMError, match="rf_only"):
        d.validate()
    d = cv.cylinder_design_3d(A, L)
    d.doc["cavity"]["interior_point"] = [0.01, 0.0]
    with pytest.raises(af.AxionFEMError, match=r"\[x, y, z\]"):
        d.validate()
    d = cv.cylinder_design_3d(A, L)
    d.doc["fullwave"]["eigensolver"] = {"method": "shift_invert"}
    with pytest.raises(af.AxionFEMError, match="target_f_Hz"):
        d.validate()
    d = cv.cylinder_design_3d(A, L, target_f_Hz=2e9)
    d.doc["fullwave"]["eigensolver"] = {"method": "ame"}
    with pytest.raises(af.AxionFEMError, match="cannot target"):
        d.validate()
    d = cv.cylinder_design_3d(A, L)
    d.doc["cavity"]["wall_material"] = "unobtainium"
    with pytest.raises(af.AxionFEMError, match="not a material"):
        d.validate()
    d = cv.cylinder_design_3d(A, L)
    d.doc["fullwave"]["frequencies"] = [1e9]
    with pytest.raises(af.AxionFEMError, match="frequency sweep|port"):
        d.validate()
    d = cv.cylinder_design_3d(A, L, target_f_Hz=2e9)
    d.doc["fullwave"]["eigensolver"] = {"completeness_steps": 2.5}
    with pytest.raises(af.AxionFEMError, match="completeness_steps"):
        d.validate()
    # a wall material without rf data: rejected unless lossless walls are explicitly allowed
    d = cv.cylinder_design_3d(A, L)
    assert d.doc["cavity"]["allow_lossless_walls"] is True  # the builder default says so explicitly
    d.doc["cavity"].pop("allow_lossless_walls")
    with pytest.raises(af.AxionFEMError, match="no rf data"):
        d.validate()
    d.doc["cavity"]["losses"] = False  # no losses: nothing to warn about
    d.validate()
    d = cv.cylinder_design_3d(A, L, wall_material=WALL)
    assert "allow_lossless_walls" not in d.doc["cavity"]
    d.validate()
    d2 = cv.pillbox_design(A, L)  # 2D: the 3D eigensolver block is rejected with a hint
    d2.doc["fullwave"]["eigensolver"] = {"method": "ame"}
    with pytest.raises(af.AxionFEMError, match="axisymmetric"):
        d2.validate()


def test_rod_gap_refinement_regions():
    rod = {"radius": 0.015, "gap": 0.02, "gap_size": 0.004}
    d = cv.cylinder_design_3d(A, L, rod=rod)
    regs = {g["name"]: g for g in d.doc["regions"]}
    assert [g["name"] for g in d.doc["regions"]] == ["cavity", "rod_gap_lo", "rod_gap_hi", "rod"]  # rod owns overlaps
    assert regs["rod_gap_lo"]["mesh_size"] == 0.004 and regs["rod"]["surface_size"] == 0.004
    lo = regs["rod_gap_lo"]["solid"]["cylinder"]
    assert lo["base"][2] == -L / 2 and lo["axis"][2] == pytest.approx(0.035) and lo["radius"] == pytest.approx(0.03)
    hi = regs["rod_gap_hi"]["solid"]["cylinder"]
    assert hi["base"][2] + hi["axis"][2] == pytest.approx(L / 2)
    d.validate()
    with pytest.raises(ValueError, match="refine_margin"):
        cv.cylinder_design_3d(A, L, rod={**rod, "x": 0.075})


def test_reference_closed_forms_3d():
    """The dielectric-rod reference (Bessel matching) in the limits it must reduce to."""
    x = cv.dielectric_rod_tm_mode(A, 0.02, 1.0 + 1e-12, L)  # eps -> 1: the empty TM010
    e = cv.cylinder_modes_3d(A, L, 1.2e9)[0]
    assert x["f_Hz"] == pytest.approx(e["f_Hz"], rel=1e-9) and x["C"] == pytest.approx(e["C"], rel=1e-9)
    assert x["G_ohm"] == pytest.approx(e["G_ohm"], rel=1e-9)
    y = cv.dielectric_rod_tm_mode(A, 0.02, 10.0, L)  # an independent value
    assert y["f_Hz"] == pytest.approx(695944201.2605058, rel=1e-12) and y["fe_rod"] == pytest.approx(0.7709362008, rel=1e-9)
    assert y["C"] == pytest.approx(0.18003188063, rel=1e-9) and y["G_ohm"] == pytest.approx(288.2515481, rel=1e-8)


def test_mirror_parity_of_the_analytic_subsets():
    """The symmetric subsets used by the quarter-model tests (independent of the solver)."""
    names = [m["name"] for m in cv.cylinder_modes_3d(A, L, 2.2e9, {"x": "pec", "y": "pmc"})]
    assert names[:4] == ["TE111(sin)", "TE112(sin)", "TM110(cos)", "TM111(cos)"]
    names = [m["name"] for m in cv.cylinder_modes_3d(A, L, 2.2e9, {"x": "pmc", "y": "pmc"})]
    assert names[:3] == ["TM010", "TM011", "TE211(sin)"]
    full = cv.cylinder_modes_3d(A, L, 2.5e9)
    # every mode belongs to exactly one of the four (x, y) parity classes
    n = sum(len(cv.cylinder_modes_3d(A, L, 2.5e9, {"x": a, "y": b})) for a in ("pec", "pmc") for b in ("pec", "pmc"))
    assert n == len(full)


# ------------------------------------------------------------------ AME, no pickup
@slow
def test_pillbox_lowest_modes_ame(tmp_path):
    r = cv.cylinder_design_3d(A, L, h=0.03, modes=10).solve(tmp_path / "p", verbose=0)
    eig = r.data["fullwave"]["eigenmodes"]
    assert eig["converged"] and eig["returned"] == 10 and eig["solver"]["method"] == "ame"
    ref = cv.cylinder_modes_3d(A, L, 2.5e9)
    assert [x["name"][:5] for x in ref[:10]] == ["TM010", "TE111", "TE111", "TM011", "TE211", "TE211", "TE112",
                                                 "TE112", "TM110", "TM110"]
    for m, x in _sorted_pairs(_modes(r), ref):
        assert m["f_Hz"] == pytest.approx(x["f_Hz"], rel=1e-3), x["name"]
        assert m["divergence_fraction"] < 1e-20
        assert list(m["energy_fraction"]) == ["cavity"]
    cav = r.data["fullwave"]["cavity"]
    assert cav["volume_m3"] == pytest.approx(math.pi * A * A * L, rel=1e-4)
    assert cav["boundaries"] == ["outer"] and cav["excluded_components"] == 0
    rows = r.cavity_modes()
    assert len(rows) == 10 and rows[0]["family"] == "3D"
    # the builder's uniform background field draws no W_mag/cost warning: an eigenmode design has neither
    assert not any("exclude the background magnet" in w for w in r.data["warnings"]), r.data["warnings"]


@slow
def test_quarter_model_symmetric_subset(tmp_path):
    sym = {"x": "pec", "y": "pmc"}
    r = cv.cylinder_design_3d(A, L, h=0.03, modes=5, symmetry=sym).solve(tmp_path / "q", verbose=0)
    assert r.data["fullwave"]["eigenmodes"]["converged"]
    assert set(r.data["fullwave"]["cavity"]["boundaries"]) == {"outer", "sym_x", "sym_y"}
    for m, x in _sorted_pairs(_modes(r), cv.cylinder_modes_3d(A, L, 3e9, sym)):
        assert m["f_Hz"] == pytest.approx(x["f_Hz"], rel=1e-3), x["name"]
        # structural (cannot fail by construction: the image sum cancels an odd mode's overlap identically); the
        # non-zero image sums are tested in test_tilted_field_mirror_images
        assert m["form_factor"]["C"] < 1e-6


# ------------------------------------------------------------------ shift-invert
@slow
@needs_strumpack
def test_shift_invert_window_and_exact_target(tmp_path):
    ft, nev = 2.1e9, 17  # 17 nearest modes: no degenerate pair cut at the window edge
    r = cv.cylinder_design_3d(A, L, h=0.02, modes=nev, target_f_Hz=ft).solve(tmp_path / "si", verbose=0)
    eig = r.data["fullwave"]["eigenmodes"]
    assert eig["converged"] and eig["returned"] == nev and eig["target_f_Hz"] == ft
    got = _modes(r)
    dmax = max(abs(m["f_Hz"] ** 2 - ft ** 2) for m in got)
    ref = [x for x in cv.cylinder_modes_3d(A, L, 4e9) if abs(x["f_Hz"] ** 2 - ft ** 2) <= dmax * 1.001]
    assert len(ref) == nev  # the returned window is complete
    for m, x in _sorted_pairs(got, ref):
        assert m["f_Hz"] == pytest.approx(x["f_Hz"], rel=1e-3), x["name"]
        assert m["residual_true"] < 1e-9 and m["divergence_fraction"] < 1e-20
        assert abs(m["rayleigh_ratio"] - 1) < 1e-10
    cc = eig["solver"]["completeness_check"]
    assert cc["missing_in_window"] == 0 and cc["near_degenerate_outside_window"] == 0 and cc["steps"] > 5


@slow
@needs_strumpack
def test_exact_target_restart(tmp_path):
    """Target = a NON-degenerate eigenvalue (TM010) of the same mesh, copied at full precision from a previous run:
    the Lanczos breakdown fires (restarts >= 1) and still returns every requested mode."""
    base = cv.cylinder_design_3d(A, L, h=0.02, modes=6, target_f_Hz=1.2e9).solve(tmp_path / "b", verbose=0)
    got = _modes(base)
    f010 = min(m["f_Hz"] for m in got)
    assert f010 == pytest.approx(cv.cylinder_modes_3d(A, L, 1.2e9)[0]["f_Hz"], rel=1e-4)
    r = cv.cylinder_design_3d(A, L, h=0.02, modes=6, target_f_Hz=f010).solve(tmp_path / "x", verbose=0)
    eig = r.data["fullwave"]["eigenmodes"]
    assert eig["solver"]["restarts_after_invariant_subspace"] >= 1
    assert eig["converged"] and eig["returned"] == eig["requested"] == 6
    assert min(abs(m["f_Hz"] / f010 - 1) for m in _modes(r)) < 1e-12
    ref = cv.cylinder_design_3d(A, L, h=0.02, modes=10, target_f_Hz=1.3e9).solve(tmp_path / "r", verbose=0)
    for m in _modes(r):  # the same modes as an independent window on the same mesh
        g = min(_modes(ref), key=lambda y: abs(y["f_Hz"] - m["f_Hz"]))
        assert m["f_Hz"] == pytest.approx(g["f_Hz"], rel=1e-10)


@slow
@needs_strumpack
def test_completeness_check_flags_cut_pair(tmp_path):
    """One mode at 1.2 GHz: the nearest is one member of the TE111 pair (split ~1e-5 by the mesh); the check finds
    the partner just outside the window and warns. Two modes: the pair is complete."""
    r1 = cv.cylinder_design_3d(A, L, h=0.02, modes=1, target_f_Hz=1.2e9).solve(tmp_path / "1", verbose=0)
    cc = r1.data["fullwave"]["eigenmodes"]["solver"]["completeness_check"]
    assert cc["near_degenerate_outside_window"] == 1 and cc["missing_in_window"] == 0
    assert any("near-degenerate pair is cut" in w for w in r1.data["warnings"])
    r2 = cv.cylinder_design_3d(A, L, h=0.02, modes=2, target_f_Hz=1.2e9).solve(tmp_path / "2", verbose=0)
    cc = r2.data["fullwave"]["eigenmodes"]["solver"]["completeness_check"]
    assert cc["near_degenerate_outside_window"] == 0 and cc["missing_in_window"] == 0
    f = sorted(m["f_Hz"] for m in _modes(r2))
    assert f[1] / f[0] - 1 < 1e-4  # both TE111 members


@slow
@needs_strumpack
def test_exactly_degenerate_pair_and_interior_sheet(tmp_path):
    """Imported, exactly mirror-symmetric mesh (tests/_cavity_meshes.py): TE101/TE011 are degenerate to round-off.
    One mode at their frequency: the partner is flagged; two: both returned. A PEC sheet inside the cavity is
    rejected when losses are on (its wall current would be the net of both sides)."""
    import _cavity_meshes as cm

    a, dz = 0.15, 0.1
    cm.build(a, dz, 6, 4, str(tmp_path / "s.msh"), str(tmp_path / "s.json"))
    ft = cv.box_modes(a, a, dz, 2e9)[1]["f_Hz"]

    def des(nev, msh="s"):
        d = cv.box_design_3d(a, a, dz, h=0.03, modes=nev, target_f_Hz=ft)
        d.doc["mesh"]["file"] = str(tmp_path / f"{msh}.msh")
        d.doc["mesh"]["info_file"] = str(tmp_path / f"{msh}.json")
        d.doc["mesh"].pop("domain_solid")  # mesher input only; an imported mesh rejects it (spec.cpp)
        return d
    r1 = des(1).solve(tmp_path / "n1", verbose=0)
    assert r1.data["fullwave"]["eigenmodes"]["solver"]["completeness_check"]["near_degenerate_outside_window"] == 1
    r2 = des(2).solve(tmp_path / "n2", verbose=0)
    f = sorted(m["f_Hz"] for m in _modes(r2))
    assert r2.data["fullwave"]["eigenmodes"]["converged"] and f[1] / f[0] - 1 < 1e-12
    cm.build(a, dz, 6, 4, str(tmp_path / "w.msh"), str(tmp_path / "w.json"), sheet=True)
    with pytest.raises(af.AxionFEMError, match="zero-thickness"):
        des(2, "w").solve(tmp_path / "w1", verbose=0)
    d = des(2, "w")
    d.doc["cavity"]["losses"] = False
    r = d.solve(tmp_path / "w2", verbose=0)
    assert r.data["fullwave"]["cavity"]["interior_sheets"] == {"sheet": 48}  # 3 x 4 cell faces, 4 tris each
    assert any("zero-thickness sheets" in w for w in r.data["warnings"])


@slow
@needs_strumpack
def test_floating_rod_no_static_contamination(tmp_path):
    rod = {"radius": 0.02, "gap": 0.02}
    r = cv.cylinder_design_3d(A, L, h=0.03, modes=8, rod=rod, target_f_Hz=0.6e9).solve(tmp_path / "si", verbose=0)
    eig = r.data["fullwave"]["eigenmodes"]
    assert eig["converged"] and eig["returned"] == 8
    assert r.data["fullwave"]["cavity"]["boundaries"] == ["outer", "pec_body_0"]
    for m in _modes(r):
        assert m["f_Hz"] > 3e8 and m["residual_true"] < 1e-9 and m["divergence_fraction"] < 1e-20
    # AME does not deflate the floating conductor's static field: its lowest "mode" is lambda ~ 0 and the gate fails
    ra = cv.cylinder_design_3d(A, L, h=0.03, modes=2, rod=rod).solve(tmp_path / "ame", verbose=0)
    ea = ra.data["fullwave"]["eigenmodes"]
    assert not ea["converged"] and min(m["f_Hz"] for m in _modes(ra)) < 1e6
    lowest = min(_modes(r), key=lambda m: m["f_Hz"])  # the first true mode agrees between the solvers
    assert max(_modes(ra), key=lambda m: m["f_Hz"])["f_Hz"] == pytest.approx(lowest["f_Hz"], rel=1e-6)


@slow
@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs >= 4 MPI slots; set AXF_MPI=1")
@needs_strumpack
def test_shift_invert_rank_independence(tmp_path):
    d = cv.cylinder_design_3d(A, L, h=0.03, modes=8, target_f_Hz=1.9e9)
    f1 = sorted(m["f_Hz"] for m in _modes(d.solve(tmp_path / "r1", verbose=0)))
    f4 = sorted(m["f_Hz"] for m in _modes(d.solve(tmp_path / "r4", verbose=0, mpi_ranks=4)))
    for a, b in zip(f1, f4):
        assert a == pytest.approx(b, rel=1e-10)


# ------------------------------------------------------------------ C, Q0, overlaps
@slow
@needs_strumpack
def test_pillbox_tm010_form_factor_and_q(tmp_path):
    x = cv.cylinder_modes_3d(A, L, 1.2e9)[0]
    assert x["C"] == pytest.approx(0.6916602761, rel=1e-9) and x["G_ohm"] == pytest.approx(301.9902287, rel=1e-8)
    r = cv.cylinder_design_3d(A, L, h=0.02, modes=1, target_f_Hz=1.1e9, wall_material=WALL).solve(tmp_path / "p", verbose=0)
    m = _modes(r)[0]
    assert m["f_Hz"] == pytest.approx(x["f_Hz"], rel=1e-4)
    assert m["form_factor"]["C"] == pytest.approx(x["C"], rel=2e-3)
    assert m["form_factor"]["C_rms"] == pytest.approx(m["form_factor"]["C"], rel=1e-6)  # uniform B0
    lo = m["losses"]
    assert lo["G_ohm"] == pytest.approx(x["G_ohm"], rel=5e-3)
    assert lo["Q0"] == pytest.approx(lo["G_ohm"] / cv.skin_Rs(m["f_Hz"], RHO), rel=1e-12)
    assert list(lo["walls"]) == ["outer"] and lo["walls"]["outer"]["P_frac"] == 1.0
    # the pointwise curl is the independent check: same G to its (lower) accuracy
    assert lo["walls"]["outer"]["G_ohm_pointwise"] == pytest.approx(x["G_ohm"], rel=3e-2)


@slow
@needs_strumpack
def test_box_tm110_form_factor_and_q(tmp_path):
    ab = (0.2, 0.15, 0.1)
    x = cv.box_modes(*ab, 1.3e9)[0]
    assert x["name"] == "TM110" and x["C"] == pytest.approx(64 / math.pi ** 4, rel=1e-12)
    r = cv.box_design_3d(*ab, h=0.02, modes=1, target_f_Hz=1.2e9, wall_material=WALL).solve(tmp_path / "b", verbose=0)
    m = _modes(r)[0]
    assert m["f_Hz"] == pytest.approx(x["f_Hz"], rel=1e-4)
    assert m["form_factor"]["C"] == pytest.approx(0.6570228643, rel=2e-3)
    assert m["losses"]["G_ohm"] == pytest.approx(x["G_ohm"], rel=5e-3)


@slow
@needs_strumpack
def test_coax_rod_vs_analytic_and_2d(tmp_path):
    a, b, Lc = 0.05, 0.2, 0.2
    x = cv.coax_tm_mode(a, b)
    assert x["kb"] == pytest.approx(4.0976855393, rel=1e-9) and x["C"] == pytest.approx(0.7949373125, rel=1e-9)
    r2 = cv.coax_design(a, b, Lc, h=0.005, modes=2, wall_material=WALL, domain_radius=1.0).solve(tmp_path / "2d", verbose=0)
    m2 = min(r2.data["fullwave"]["eigenmodes"]["TM"]["modes"], key=lambda m: abs(m["f_Hz"] - x["f_Hz"]))
    r = cv.cylinder_design_3d(b, Lc, h=0.02, modes=1, rod={"radius": a}, wall_material=WALL,
                              target_f_Hz=0.96e9).solve(tmp_path / "3d", verbose=0)
    m = _modes(r)[0]
    assert r.data["fullwave"]["cavity"]["boundaries"] == ["outer", "pec_body_0"]
    assert m["f_Hz"] == pytest.approx(x["f_Hz"], rel=1e-3)
    assert m["form_factor"]["C"] == pytest.approx(x["C"], rel=1e-3)
    assert m["losses"]["G_ohm"] == pytest.approx(m2["losses"]["G_ohm"], rel=1e-3)
    assert set(m["losses"]["walls"]) == {"outer", "rod"}


@slow
@needs_strumpack
def test_overlaps_and_symmetry_images(tmp_path):
    r = cv.cylinder_design_3d(A, L, h=0.02, modes=14, target_f_Hz=1.58e9, mode_overlaps=True).solve(tmp_path / "e", verbose=0)
    modes = _modes(r)
    eta = r.data["fullwave"]["eigenmodes"]["overlaps"]["eta"]
    f = [m["f_Hz"] for m in modes]
    ref = cv.cylinder_modes_3d(A, L, 2.2e9)
    f011 = next(x["f_Hz"] for x in ref if x["name"] == "TE011")
    i0 = min(range(len(f)), key=lambda i: abs(f[i] - ref[0]["f_Hz"]))
    tri = [j for j, y in enumerate(f) if abs(y / f011 - 1) < 2e-3]
    assert len(tri) == 3  # TE011 and the TM111 pair are exactly degenerate (chi'01 = chi11)
    eta_x = cv.eta_tm010_te01p(A, L, 1)
    assert math.sqrt(sum(eta[i0][j] ** 2 for j in tri)) == pytest.approx(eta_x, rel=5e-3)
    # TM010 with the m != 0 modes and TM011: zero in the continuum; small here (discretization only)
    for j in range(len(f)):
        if j != i0 and j not in tri:
            assert eta[i0][j] < 1e-3
    # quarter model with two PMC planes: TM010 survives; C from the mirror-image sum of B0, symmetry planes
    # carry no loss, full-cavity overlaps of one model vanish (pseudoscalar E.B)
    x = ref[0]
    q = cv.cylinder_design_3d(A, L, h=0.02, modes=2, symmetry={"x": "pmc", "y": "pmc"}, wall_material=WALL,
                              mode_overlaps=True, target_f_Hz=1.1e9).solve(tmp_path / "q", verbose=0)
    m = min(_modes(q), key=lambda y: abs(y["f_Hz"] - x["f_Hz"]))
    assert m["form_factor"]["C"] == pytest.approx(x["C"], rel=2e-3)
    assert m["losses"]["G_ohm"] == pytest.approx(x["G_ohm"], rel=5e-3)
    assert list(m["losses"]["walls"]) == ["outer"]
    ov = q.data["fullwave"]["eigenmodes"]["overlaps"]
    assert all(v == 0.0 for row in ov["eta"] for v in row) and "eta_meshed_part" in ov
    ew = r.data["fullwave"]["eigenmodes"]["overlaps"]["eta_energy"]  # vacuum: the energy norm is the L2 norm
    assert all(abs(eta[i][j] - ew[i][j]) <= 1e-8 * max(eta[i][j], 1e-6) for i in range(len(eta)) for j in range(len(eta)))


# ------------------------------------------------------------------ mirror images, lossless walls, floating rods
@slow
@needs_strumpack
def test_tilted_field_mirror_images(tmp_path):
    """Non-zero mirror-image form factor: B0 = (0.6, 0, 0.8) T. The quarter model with a PEC plane x = 0 and a PMC
    plane y = 0 holds one TE111 member, whose C from the image sum is 0.36 x 16/(pi^2 (x'11^2 - 1)); the
    (PMC, PMC) quarter holds TM010 with C = 0.64 x 4/chi01^2."""
    from scipy.special import jnp_zeros

    xp = jnp_zeros(1, 1)[0]
    c_te111 = 0.36 * 16 / (math.pi ** 2 * (xp * xp - 1))
    B = (0.6, 0.0, 0.8)
    q = cv.cylinder_design_3d(A, L, h=0.02, modes=1, target_f_Hz=1.16e9, wall_material=WALL, B_T=B,
                              symmetry={"x": "pec", "y": "pmc"}).solve(tmp_path / "q", verbose=0)
    m = _modes(q)[0]
    assert m["f_Hz"] == pytest.approx(next(x["f_Hz"] for x in cv.cylinder_modes_3d(A, L, 1.3e9)
                                           if x["name"].startswith("TE111")), rel=1e-3)
    assert m["form_factor"]["C"] == pytest.approx(c_te111, rel=1e-3)
    assert m["form_factor"]["C"] == pytest.approx(m["form_factor"]["C_rms"], rel=1e-9)  # B_ref = volume rms
    p = cv.cylinder_design_3d(A, L, h=0.02, modes=1, target_f_Hz=1.12e9, wall_material=WALL, B_T=B,
                              symmetry={"x": "pmc", "y": "pmc"}).solve(tmp_path / "p", verbose=0)
    assert _modes(p)[0]["form_factor"]["C"] == pytest.approx(0.64 * cv.cylinder_modes_3d(A, L, 1.2e9)[0]["C"], rel=2e-3)


@slow
@needs_strumpack
def test_dielectric_rod_losses_3d(tmp_path):
    """On-axis dielectric rod (eps_r 10, tan d 1e-4, r = 0.02, full length) in copper walls vs Bessel matching:
    f, eps-weighted C, wall G, the rod's electric-energy fraction and Q_d = 1/(tan d fraction), Q0."""
    x = cv.dielectric_rod_tm_mode(A, 0.02, 10.0, L)
    mat = {"kind": "linear", "eps_r": 10.0, "tan_delta": 1e-4}
    r = cv.cylinder_design_3d(A, L, h=0.02, modes=1, rod={"radius": 0.02, "material": mat, "hole": False},
                              target_f_Hz=0.95 * x["f_Hz"], wall_material=WALL).solve(tmp_path / "d", verbose=0)
    m = _modes(r)[0]
    lo = m["losses"]
    assert m["f_Hz"] == pytest.approx(x["f_Hz"], rel=1e-3)
    assert m["form_factor"]["C"] == pytest.approx(x["C"], rel=3e-3)
    assert lo["G_ohm"] == pytest.approx(x["G_ohm"], rel=2e-3)
    fe = lo["dielectric"]["rod"]["electric_energy_fraction"]
    assert fe == pytest.approx(x["fe_rod"], rel=1e-3)
    assert lo["Q_dielectric"] == pytest.approx(1 / (1e-4 * x["fe_rod"]), rel=1e-3)
    q_wall = x["G_ohm"] / cv.skin_Rs(m["f_Hz"], RHO)
    assert lo["Q0"] == pytest.approx(1 / (1 / q_wall + 1e-4 * x["fe_rod"]), rel=3e-3)
    assert not lo.get("lossless_walls") and not r.data["warnings"]


@slow
@needs_strumpack
def test_lossless_walls_flagged_3d(tmp_path):
    """Walls without rf data never yield a silently incomplete Q0."""
    sleeve = {"r1": 0.03, "r2": 0.05, "material": {"kind": "linear", "eps_r": 4.0, "tan_delta": 1e-4}}
    d = cv.cylinder_design_3d(A, L, h=0.03, modes=1, target_f_Hz=0.8e9, sleeve=sleeve)
    r = d.solve(tmp_path / "a", verbose=0)  # builder default: lossless walls explicitly allowed
    lo = _modes(r)[0]["losses"]
    assert lo["walls_lossless_by_request"] is True and lo["lossless_walls"] == ["outer"]
    assert lo["Q0"] == pytest.approx(lo["Q_dielectric"], rel=1e-12) and "Q_wall" not in lo
    assert not any("rf data" in w for w in r.data["warnings"])
    d.doc["cavity"].pop("allow_lossless_walls")
    d.doc["cavity"].pop("wall_material")  # the domain wall without a material: Q0 omitted, warned
    r = d.solve(tmp_path / "b", verbose=0)
    lo = _modes(r)[0]["losses"]
    assert "Q0" not in lo and lo["Q_dielectric"] > 0 and lo["lossless_walls"] == ["outer"]
    assert any("Q0 is not reported" in w for w in r.data["warnings"])
    assert any("outer have no rf data" in w for w in r.data["warnings"])
    # copper wall, rod without rf data: Q0 reported from the other losses, the rod flagged and warned
    d = cv.cylinder_design_3d(A, L, h=0.03, modes=1, target_f_Hz=0.8e9, sleeve=sleeve, wall_material=WALL,
                              rod={"radius": 0.01, "x": 0.07, "material": {"kind": "conductor"}})
    r = d.solve(tmp_path / "c", verbose=0)
    lo = _modes(r)[0]["losses"]
    assert lo["lossless_walls"] == ["rod"] and "Q0" in lo and "Q_wall" in lo
    assert any("rod have no rf data" in w for w in r.data["warnings"])


@slow
@needs_strumpack
def test_floating_rod_vs_2d(tmp_path):
    """On-axis floating rod (r = 0.015, gaps 0.02) vs the axisymmetric solver (rod surface
    refined), per wall body. Without local refinement the rod is under-resolved (C ~4 % low at h = 0.02) and the
    flux/pointwise indicator says so; with gap refinement f, C, G and per-body G agree to the levels of the
    refinement study (docs/validation/cavity_3d.md, which extrapolates to the 2D values as h -> 0)."""
    r2 = cv.floating_rod_design(A, L, 0.015, 0.02, h=0.0025, wall_material=WALL).solve(tmp_path / "2d", verbose=0)
    x = min(r2.data["fullwave"]["eigenmodes"]["TM"]["modes"], key=lambda m: m["f_Hz"])
    assert x["f_Hz"] == pytest.approx(602.06e6, rel=2e-4) and x["form_factor"]["C"] == pytest.approx(0.11834, rel=1e-3)
    w2 = x["losses"]["walls"]
    rod = {"radius": 0.015, "gap": 0.02}
    rc = cv.cylinder_design_3d(A, L, h=0.02, modes=1, rod=rod, target_f_Hz=0.6e9, wall_material=WALL).solve(
        tmp_path / "coarse", verbose=0)
    m = _modes(rc)[0]
    assert m["losses"]["walls"]["rod"]["resolution_warning"] is True
    assert m["losses"]["walls"]["rod"]["G_flux_pointwise_disagreement"] > 0.05
    assert any("disagree" in w and "rod" in w for w in rc.data["warnings"])
    assert m["form_factor"]["C"] / x["form_factor"]["C"] - 1 < -0.02  # the unresolved gap: C a few % low
    rf = cv.cylinder_design_3d(A, L, h=0.02, modes=1, rod={**rod, "gap_size": 0.005}, target_f_Hz=0.6e9,
                               wall_material=WALL).solve(tmp_path / "fine", verbose=0)
    m = _modes(rf)[0]
    w = m["losses"]["walls"]
    assert not any("disagree" in s for s in rf.data["warnings"])
    assert w["rod"]["G_flux_pointwise_disagreement"] < 0.05
    assert m["f_Hz"] == pytest.approx(x["f_Hz"], rel=4e-3)
    assert m["form_factor"]["C"] == pytest.approx(x["form_factor"]["C"], rel=8e-3)
    assert m["losses"]["G_ohm"] == pytest.approx(x["losses"]["G_ohm"], rel=5e-3)
    assert w["rod"]["G_ohm"] == pytest.approx(w2["rod"]["G_ohm"], rel=6e-3)
    assert w["outer"]["G_ohm"] == pytest.approx(w2["wall"]["G_ohm"], rel=3e-3)
    assert m["f_Hz"] < x["f_Hz"] and m["form_factor"]["C"] < x["form_factor"]["C"]  # converging from below


@slow
@needs_strumpack
def test_retained_conductor_consistent_flux(tmp_path):
    """An off-centre floating rod kept as a meshed conductor volume (no mesh.remove_pec) gets the same
    consistent-flux wall current as the PEC hole (a pointwise curl gives +8.7 % in G)."""
    rod = {"radius": 0.015, "gap": 0.02, "x": 0.045}
    res = {}
    for tag, rd in (("hole", rod), ("ret", {**rod, "hole": False})):
        r = cv.cylinder_design_3d(A, L, h=0.02, modes=1, rod=rd, target_f_Hz=0.6e9, wall_material=WALL).solve(
            tmp_path / tag, verbose=0)
        res[tag] = (r, _modes(r)[0])
    h, k = res["hole"][1], res["ret"][1]
    assert any("retained conductor" in w for w in res["ret"][0].data["warnings"])
    assert k["f_Hz"] == pytest.approx(h["f_Hz"], rel=1e-4)
    assert k["losses"]["G_ohm"] == pytest.approx(h["losses"]["G_ohm"], rel=1e-3)
    for b in ("outer", "rod"):
        assert k["losses"]["walls"][b]["G_ohm"] == pytest.approx(h["losses"]["walls"][b]["G_ohm"], rel=2e-3)


# ------------------------------------------------------------------ shared R_s models in 3D
@slow
@needs_strumpack
def test_builtin_copper_q0_matches_2d(tmp_path):
    """Builtin `copper` (anomalous skin) on a pillbox: the 3D TM010 Q0 equals the axisymmetric
    one (same R_s model and value through spec::RFLoss; G agrees to the 3D discretization, 5e-3 at h = 0.02)."""
    d2 = cv.pillbox_design(A, L, h=0.005, modes=1, families=("TM",), wall_material={"kind": "conductor"})
    d2.doc["regions"][1]["material"] = "copper"
    d2.doc["cavity"].pop("allow_lossless_walls", None)
    m2 = d2.solve(tmp_path / "2d", verbose=0).data["fullwave"]["eigenmodes"]["TM"]["modes"][0]
    d3 = cv.cylinder_design_3d(A, L, h=0.02, modes=1, target_f_Hz=1.1e9)
    d3.doc["cavity"]["wall_material"] = "copper"
    d3.doc["cavity"].pop("allow_lossless_walls")
    r3 = d3.solve(tmp_path / "3d", verbose=0)
    m3 = _modes(r3)[0]
    w2, w3 = m2["losses"]["walls"]["wall"], m3["losses"]["walls"]["outer"]
    assert w2["Rs_model"] == w3["Rs_model"] == "anomalous"
    assert m3["f_Hz"] == pytest.approx(m2["f_Hz"], rel=1e-4)
    assert w3["Rs_ohm"] == pytest.approx(w2["Rs_ohm"], rel=1e-4)  # R_s ~ f^(2/3): f to 1e-4
    assert m3["losses"]["G_ohm"] == pytest.approx(m2["losses"]["G_ohm"], rel=5e-3)
    assert m3["losses"]["Q0"] == pytest.approx(m2["losses"]["Q0"], rel=5e-3)
    assert m3["losses"]["Q0"] == pytest.approx(m3["losses"]["G_ohm"] / w3["Rs_ohm"], rel=1e-12)
    assert "lossless_walls" not in m3["losses"] and not any("rf data" in w for w in r3.data["warnings"])


@slow
@needs_strumpack
def test_flux_flow_orientation_3d_pillbox(tmp_path):
    """Shared R_s code in 3D (docs/physics/surface_resistance.md Sec. 6): the 3D wall loss weights the flux_flow term
    per wall point with s = sin^2(J_rf, B0). TM010 in an axial 1 T field: barrel current axial (s = 0), cap currents
    radial (s = 1); the |H_t|^2 weight of the caps is a / (a + L) = 1/3, so the `outer` body (barrel + caps) has s = 1/3
    and R_ff = R_ff,perp / 3 (to the 3D G accuracy)."""
    ff = {"Bc2_T": 10.0, "rho_n_ohm_m": 1.5e-7, "lambda_m": 1.1e-7, "f_depin_Hz": 1e9}
    nb = {"model": "bcs", "A_ohm_K": 2e-4, "f_ref": 1.5e9, "Delta_over_kTc": 17.67 / 9.25, "Tc_K": 9.25,
          "R_res_ohm": 5e-9, "T_K": 4.2}
    rf = dict(nb, flux_flow=dict(ff))
    r = cv.cylinder_design_3d(A, L, h=0.02, modes=1, target_f_Hz=1.1e9, B_T=(0.0, 0.0, 1.0),
                              wall_material={"kind": "conductor", "rf": rf}).solve(tmp_path / "ff", verbose=0)
    m = _modes(r)[0]
    f = m["f_Hz"]
    w = m["losses"]["walls"]["outer"]
    base = cv.surface_resistance(nb, f)
    _, perp = cv.surface_resistance(dict(nb, flux_flow=dict(ff, B_T=1.0)), f, flux_flow_part=True)
    assert w["Rs_model"] == "bcs"
    assert w["Rs_flux_flow_perpendicular_ohm"] == pytest.approx(perp, rel=1e-9)  # uniform |B0| = 1 T
    assert w["flux_flow_orientation_factor"] == pytest.approx(A / (A + L), rel=1e-2)
    assert w["Rs_flux_flow_ohm"] == pytest.approx(perp * A / (A + L), rel=1e-2)
    assert w["Rs_ohm"] == pytest.approx(base + w["Rs_flux_flow_ohm"], rel=1e-12)
    assert any("sin^2(J_rf, B0)" in x for x in r.data["warnings"])
