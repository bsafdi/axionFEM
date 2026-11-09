"""Sparse direct solver path of the axisymmetric engine.

Backends: STRUMPACK (when built in) and Eigen's LDLT must give the same results, and the backend must be
reported so that this cannot pass vacuously (AXF_STRUMPACK_MIN_DOFS routes small systems to STRUMPACK here).
Exterior map (DtN): the default low-rank (Woodbury) application and the reference path that assembles the
dense block into the matrix (AXF_DTN_DENSE=1) must agree to round-off, and the low-rank path must give less
fill -- the dense block is a clique of the sparsity graph.
Repeated factorizations (Newton) must reuse the symbolic analysis.
Fill-reducing ordering (AXF_ORDERING): METIS and AMD must give the same answer, and an unknown value falls back to the
automatic choice with a warning.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import axionfem as af
from axionfem._paths import axf_path

HAVE_STRUMPACK = bool(af.build_info().get("strumpack"))


def _run(tmp, design, env_extra):
    """Solve `design` through the CLI with extra environment (the solver backend is chosen per process)."""
    Path(tmp).mkdir(parents=True, exist_ok=True)
    p = Path(tmp) / "d.json"
    p.write_text(json.dumps(design.doc))
    env = dict(os.environ, **env_extra)
    out = subprocess.run([axf_path(), "solve", str(p), "-o", str(tmp), "-v", "0"], capture_output=True, text=True,
                         env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    return json.loads((Path(tmp) / "results.json").read_text())


def _solve(tmp, env_extra):
    return _run(tmp, af.template("solenoid", mesh={"order": 2, "size": 0.04, "surface_size": 0.01}), env_extra)


def _poloidal(res):
    """Diagnostics of the A_phi (poloidal-flux) solve, the one that carries the exterior map."""
    return res["static"]["poloidal"]["solver"]


@pytest.mark.skipif(axf_path() is None, reason="axf executable not built")
def test_strumpack_and_eigen_agree(tmp_path):
    a = _solve(tmp_path / "eigen", {"AXF_STRUMPACK_MIN_DOFS": "1000000000"})
    assert _poloidal(a)["backend"] == "eigen-ldlt"
    if not HAVE_STRUMPACK:
        pytest.skip("this build has no STRUMPACK (the sequential wheels are built without it)")
    b = _solve(tmp_path / "strumpack", {"AXF_STRUMPACK_MIN_DOFS": "1000"})
    assert _poloidal(b)["backend"] == "strumpack"  # not a vacuous comparison of two Eigen runs
    for k in ("Psi_SI_H_T_m2", "L_p_H", "W_mag_J"):
        assert b["coupling"][k] == pytest.approx(a["coupling"][k], rel=1e-9), k


@pytest.mark.skipif(axf_path() is None, reason="axf executable not built")
def test_strumpack_min_dofs_parses_scientific_notation(tmp_path):
    """`AXF_STRUMPACK_MIN_DOFS=1e9` means 'never'; an integer parse (atoi) would read 1, i.e. 'always'."""
    r = _solve(tmp_path / "e9", {"AXF_STRUMPACK_MIN_DOFS": "1e9"})
    assert _poloidal(r)["backend"] == "eigen-ldlt"
    if HAVE_STRUMPACK:
        r = _solve(tmp_path / "e3", {"AXF_STRUMPACK_MIN_DOFS": "1e3"})
        assert _poloidal(r)["backend"] == "strumpack"


@pytest.mark.skipif(axf_path() is None, reason="axf executable not built")
def test_strumpack_min_dofs_rejects_garbage(tmp_path):
    """A value that is not a number falls back to the default and says so."""
    p = Path(tmp_path) / "d.json"
    p.write_text(json.dumps(af.template("solenoid", mesh={"order": 2, "size": 0.06, "surface_size": 0.02}).doc))
    env = dict(os.environ, AXF_STRUMPACK_MIN_DOFS="lots")
    out = subprocess.run([axf_path(), "solve", str(p), "-o", str(tmp_path), "-v", "1"], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "AXF_STRUMPACK_MIN_DOFS" in out.stderr


@pytest.mark.skipif(axf_path() is None, reason="axf executable not built")
def test_dtn_low_rank_matches_the_assembled_block(tmp_path):
    """The exterior map as a Woodbury update (default) vs assembled into the matrix (AXF_DTN_DENSE=1)."""
    d = af.template("core", mesh={"order": 2})
    dense = _run(tmp_path / "dense", d, {"AXF_DTN_DENSE": "1", "AXF_STRUMPACK_MIN_DOFS": "1e18"})
    low = _run(tmp_path / "low", d, {"AXF_STRUMPACK_MIN_DOFS": "1e18"})
    sd, sl = _poloidal(dense), _poloidal(low)
    assert sd["dtn_assembly"] == "dense" and sl["dtn_assembly"] == "low-rank"
    assert sd["dtn_modes"] == sl["dtn_modes"] > 0
    for k in ("Psi_SI_H_T_m2", "L_p_H", "W_mag_J"):
        assert low["coupling"][k] == pytest.approx(dense["coupling"][k], rel=1e-10), k
    assert low["fields"]["B_max_T"] == pytest.approx(dense["fields"]["B_max_T"], rel=1e-10)
    # the point of the change: the dense block is a clique of the sparsity graph
    assert sl["nonzeros_L"] < 0.9 * sd["nonzeros_L"]
    assert sl["solve_residual"] < 1e-8 and sd["solve_residual"] < 1e-8  # measured 1e-13


@pytest.mark.skipif(axf_path() is None, reason="axf executable not built")
def test_dtn_low_rank_matches_the_assembled_block_with_strumpack(tmp_path):
    if not HAVE_STRUMPACK:
        pytest.skip("this build has no STRUMPACK")
    d = af.template("core", mesh={"order": 2})
    dense = _run(tmp_path / "dense", d, {"AXF_DTN_DENSE": "1", "AXF_ORDERING": "strumpack"})
    low = _run(tmp_path / "low", d, {"AXF_ORDERING": "strumpack"})
    assert _poloidal(low)["backend"] == _poloidal(dense)["backend"] == "strumpack"
    for k in ("Psi_SI_H_T_m2", "L_p_H", "W_mag_J"):
        assert low["coupling"][k] == pytest.approx(dense["coupling"][k], rel=1e-10), k
    assert _poloidal(low)["nonzeros_L"] < _poloidal(dense)["nonzeros_L"]


@pytest.mark.skipif(axf_path() is None, reason="axf executable not built")
def test_newton_reuses_the_symbolic_analysis(tmp_path):
    """Nonlinear iron: the ordering and symbolic factorization are computed once, not per Newton step."""
    if not HAVE_STRUMPACK:
        pytest.skip("this build has no STRUMPACK (Eigen's LDLT reuses analyzePattern by construction)")
    d = af.template("core", iron_t=0.1, mesh={"order": 2})
    r = _run(tmp_path / "nl", d, {"AXF_ORDERING": "strumpack"})
    s = _poloidal(r)
    assert s["nonlinear"] and s["newton_iterations"] >= 2
    assert s["symbolic_analyses"] == 1
    assert s["reused_factorizations"] == s["newton_iterations"]
    assert s["converged"] and s["solve_residual"] < 1e-8  # measured 8e-11 at 7.8e4 unknowns


@pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="fine 2D template (about 3 min); set AXF_SLOW=1")
@pytest.mark.skipif(axf_path() is None, reason="axf executable not built")
def test_fine_template_solves(tmp_path):
    """The 706k-DOF template that the Eigen path could not solve.  Eigen's own AMD
    ordering is excluded on purpose -- it takes 500-700 s on this matrix and has been seen to segfault
    inside Eigen, which is why the solver does not choose it at this size (docs/validation/solvers.md)."""
    d = af.template("toroid", mesh={"order": 2, "size": 0.02, "surface_size": 0.004})
    ref = None
    for name, env in (("metis", {"AXF_ORDERING": "metis", "AXF_STRUMPACK_MIN_DOFS": "1e18"}),
                      ("default", {})):
        r = _run(tmp_path / name, d, env)
        s = r["pickup"]["solver"]
        assert s["reduced_unknowns"] > 400000
        assert s["ordering"] in ("metis", "strumpack"), s["ordering"]  # never Eigen's AMD at this size
        assert s["dtn_assembly"] == "low-rank" and s["solve_residual"] < 1e-8
        if ref is None:
            ref = r
        else:
            for k in ("Psi_SI_H_T_m2", "L_p_H"):
                assert r["coupling"][k] == pytest.approx(ref["coupling"][k], rel=1e-9), (name, k)


# ---------------------------------------------------------------- fill-reducing ordering
def _ring_design(**mesh_extra):
    return {"schema": "axionfem.design/1", "name": "ring",
            "regions": [{"name": "coil", "material": "winding",
                         "shape": {"rect": {"r": [0.9, 1.1], "z": [-0.1, 0.1]}}, **mesh_extra}],
            "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
            "mesh": {"order": 1, "size": 0.25, "surface_size": 0.25}}


def test_metis_and_amd_orderings_agree(tmp_path):
    """Both fill-reducing orderings must give the same answer; only the cost differs."""
    out = {}
    for which in ("amd", "metis"):
        env = dict(os.environ, AXF_ORDERING=which)
        script = (
            "import json, axionfem as af\n"
            f"r = af.template('core').solve(workdir={str(tmp_path / which)!r}, order=2, verbose=0,"
            " self_force=False)\n"
            "s = r.data['static']['poloidal']['solver']\n"
            "print(json.dumps({'Psi': r.Psi, 'L_p': r.L_p, 'W': r.W, 'ordering': s['ordering'],\n"
            "                  'nonzeros_L': s['nonzeros_L'], 'n': s['reduced_unknowns']}))\n")
        p = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
        out[which] = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["amd"]["ordering"] == "amd"
    if out["metis"]["ordering"] != "metis":
        pytest.skip("this build has no METIS (the sequential wheels are built without it)")
    for k in ("Psi", "L_p", "W"):
        assert out["metis"][k] == pytest.approx(out["amd"][k], rel=1e-10), k
    # METIS nested dissection gives less fill than AMD on this mesh
    assert out["metis"]["nonzeros_L"] < out["amd"]["nonzeros_L"]


def test_bad_ordering_falls_back_to_auto(tmp_path):
    env = dict(os.environ, AXF_ORDERING="supernodal")
    script = ("import axionfem as af\n"
              f"r = af.Design({_ring_design()!r}).solve(workdir={str(tmp_path / 'x')!r}, verbose=1)\n"
              "print('ordering', r.data['static']['poloidal']['solver']['ordering'])\n")
    p = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "ordering amd" in p.stdout
    assert "AXF_ORDERING" in p.stderr  # the warning names the variable
