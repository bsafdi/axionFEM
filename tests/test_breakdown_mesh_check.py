"""Solver-failure messages name the worst mesh element.

A long 3D run can fail with AMS-PCG breakdown messages after hours, and the cause is often an inverted or flat
element. Every breakdown or non-convergence message therefore ends with a mesh
check: the worst element's number, region, centroid, Jacobian aspect ratio and det J ratio, the counts of bad
elements, and what to change in the design (mesh_quality.hpp; computed on the failure path only). The iteration caps
are forced down so that the solves stall deterministically:

- 3D full wave (GMRES) on the TEM line's imported mesh with one vertex pushed to within 1e-4 of the opposite face of
  a tetrahedron (a deliberate sliver): the message names that sliver, its region and centroid, and the remedy; on
  the unmodified mesh it says the mesh is probably not the cause;
- 3D MQS (AMS-PCG): the warning and results mesh.info.worst_elements agree; a successful solve has neither;
- the same over 2 MPI ranks (AXF_MPI): the worst elements reduced across the ranks are the serial run's;
- 2D (Newton on nonlinear iron): the axisymmetric warning, with (r, z) and the 2D remedies.

The element measures themselves (sliver, inverted curved tetrahedron and triangle, MPI pack/merge) are checked in
cpp/tests/test_breakdown_mesh_check.cpp. Seconds each.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import axionfem as af

needs_3d = pytest.mark.skipif(not af.build_info()["engine_3d"],
                              reason="needs the 3D engine (an MPI build of axionFEM)")
from axionfem import _core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

def _slab_doc():
    """A coil and a thin vacuum slab (2 mm under a 0.15 m mesh size)."""
    coil = {"name": "coil", "solid": {"revolve": {"profile": {"rect": {"r": [0.5, 0.6], "z": [-0.05, 0.05]}}}},
            "material": "winding"}
    slab = {"name": "slab", "solid": {"box": {"min": [-0.3, -0.3, 0.2], "max": [0.3, 0.3, 0.202]}},
            "material": "vacuum"}
    return {"schema": "axionfem.design/1", "name": "slab", "symmetry": "3d", "regions": [coil, slab],
            "windings": [{"name": "w", "region": "coil", "type": "azimuthal", "ampere_turns": 1e4}],
            "mesh": {"size": 0.15, "surface_size": 0.08, "order": 1}}


def _stalled_warnings(r):
    return [w for w in r["warnings"] if "3D linear solve" in w]


def _check_message(r, stalled):
    """The warnings' mesh check names the first of results mesh.info.worst_elements."""
    assert stalled, r["warnings"]
    worst = r["mesh"]["info"]["worst_elements"]
    top = worst["worst"][0]
    assert worst["elements"] == r["mesh"]["elements"]
    assert len(worst["worst"]) == min(5, worst["elements"])
    for w in stalled:
        assert "Mesh check (run because this solve failed)" in w
        assert f"element {top['element']} in region '{top['region']}' (attribute {top['attribute']})" in w
        assert f"centred at ({top['centroid_m'][0]:g}, " in w
        assert "Jacobian aspect ratio" in w and ("Fix in the design:" in w or "Fix: the mesh is imported" in w)
    return top


def _tets(msh):
    """Nodes and tetrahedra (0-based) of a gmsh MSH 2.2 file."""
    lines = msh.read_text().splitlines()
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    X = np.array([[float(t) for t in lines[i + 2 + k].split()[1:4]] for k in range(n)])
    j = lines.index("$Elements")
    T = [[int(t) - 1 for t in f.split()[5:9]] for f in lines[j + 2:j + 2 + int(lines[j + 1])]
         if f.split()[1] == "4"]
    return lines, i, X, np.array(T)


def _make_sliver(msh, centre):
    """Push the vertex nearest `centre` to 1e-4 of its height above the opposite face of one of its tetrahedra; the
    other tetrahedra around it keep their orientation. Returns the moved vertex and the tetrahedra (0-based, in file
    order, which is MFEM's element order)."""
    lines, i, X, T = _tets(msh)
    v = int(np.argmin(np.linalg.norm(X - centre, axis=1)))
    t = int(np.where((T == v).any(axis=1))[0][0])
    A, B, C = X[[u for u in T[t] if u != v]]
    nrm = np.cross(B - A, C - A)
    nrm /= np.linalg.norm(nrm)
    Y = X.copy()
    Y[v] = X[v] - np.dot(X[v] - A, nrm) * nrm * (1 - 1e-4)

    def vol(P):
        a = P[T]
        return np.einsum("ij,ij->i", np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), a[:, 3] - a[:, 0])

    assert (np.sign(vol(Y)) == np.sign(vol(X))).all()   # no tetrahedron inverted, one flattened
    lines[i + 2 + v] = f"{v + 1} " + " ".join(f"{c:.17g}" for c in Y[v])
    msh.write_text("\n".join(lines) + "\n")
    return v, T


def _fullwave_stalled(tmp_path, monkeypatch, sliver, frequencies=(1e6,), verbose=0):
    import fullwave_indicator as FWI

    monkeypatch.setenv("AXF_FW_SOLVER", "ams")
    monkeypatch.setenv("AXF_FW_MAX_ITER", "1")
    for key in ("AXF_FW_GAUGE", "AXF_FW_GAUSS_GUARD", "AXF_FW_TC_CHARGE_PROJECTION"):
        monkeypatch.delenv(key, raising=False)
    wd = tmp_path / ("sliver" if sliver else "clean")
    doc, g = FWI.tem_design(wd, "pec", 2, 0, list(frequencies))
    at = _make_sliver(Path(doc["mesh"]["file"]), np.array([g["l"] / 2, g["h"] / 2, g["W"] / 2])) if sliver else None
    r = af.Design(doc).solve(wd, verbose=verbose).data
    stalled = [w for w in r["warnings"] if "GMRES did not converge" in w]
    return r, stalled, at


@needs_3d
def test_3d_fullwave_stall_names_the_sliver(tmp_path, monkeypatch, capfd):
    r, stalled, at = _fullwave_stalled(tmp_path, monkeypatch, True, frequencies=(1e5, 1e6, 1e7), verbose=1)
    top = _check_message(r, stalled)
    # three stalled frequencies, one computation of the check (one "solver failure:" log line): the cache
    assert len(stalled) == 3
    assert capfd.readouterr().err.count("solver failure:") == 1
    worst = r["mesh"]["info"]["worst_elements"]
    assert top["region"] == "field" and not top["inverted"] and not top["curved"]
    assert 1e3 < top["jacobian_aspect_ratio"] < 1e6   # flat, and still below the engine's refusal at 1e6
    assert worst["flat"] >= 1 and worst["inverted"] == 0
    v, T = at
    assert v in T[top["element"]]   # the flattened tetrahedron or another one around the moved vertex
    # an imported mesh: the remedy is about the file, not about the mesher's settings
    assert "Fix: the mesh is imported (mesh.file '" in stalled[0] and "plates3d.msh" in stalled[0]
    assert "Repair or regenerate it in the tool that wrote it: a flat (sliver) element" in stalled[0]
    assert "optimize_netgen" not in stalled[0] and "surface_size" not in stalled[0]


@needs_3d
def test_3d_fullwave_stall_on_a_clean_mesh(tmp_path, monkeypatch):
    r, stalled, _ = _fullwave_stalled(tmp_path, monkeypatch, False)
    _check_message(r, stalled)
    assert "probably not the cause" in stalled[0]
    worst = r["mesh"]["info"]["worst_elements"]
    assert worst["inverted"] == 0 and worst["flat"] == 0


@needs_3d
def test_3d_mqs_stall_names_the_worst_element(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("AXF_SOLVER_MAXIT", "2")
    r = json.loads(_core.run_3d(json.dumps(_slab_doc()), str(tmp_path / "s"), 1, 1))
    stalled = _stalled_warnings(r)
    _check_message(r, stalled)
    assert "mesh.info.worst_elements in the results" in stalled[0]
    assert capfd.readouterr().err.count("solver failure:") == len(stalled)


@needs_3d
def test_3d_successful_solve_has_no_mesh_check(tmp_path):
    r = json.loads(_core.run_3d(json.dumps(_slab_doc()), str(tmp_path / "s"), 1, 0))
    assert not _stalled_warnings(r)
    assert "worst_elements" not in r["mesh"]["info"]
    assert not any("Mesh check" in w for w in r["warnings"])


# The serial run (in this process) and the 2-rank run evaluate the same elements' Jacobians, but from node coordinates
# held in another order (the serial mesh against each rank's ParMesh), so the measures agree to round-off amplified by
# the element's conditioning (Jacobian aspect ratios near 1e2 here): 8e-13 relative was measured. 1e-10 leaves room for
# that, while another element or a wrong reduction across the ranks changes a measure at O(1). Element numbers,
# attributes, regions, flags and counts must agree exactly.
MEASURE_REL = 1e-10


def _same_elements(got, ref, where="worst"):
    if isinstance(ref, dict):
        assert isinstance(got, dict) and got.keys() == ref.keys(), (where, got, ref)
        for k in ref:
            _same_elements(got[k], ref[k], f"{where}.{k}")
    elif isinstance(ref, list):
        assert isinstance(got, list) and len(got) == len(ref), (where, got, ref)
        for i, (g, r) in enumerate(zip(got, ref)):
            _same_elements(g, r, f"{where}[{i}]")
    elif isinstance(ref, float) and not isinstance(got, bool):
        # centroids are metres in a ~1 m model: an absolute floor for coordinates at 0
        assert got == pytest.approx(ref, rel=MEASURE_REL, abs=1e-12), (where, got, ref)
    else:
        assert got == ref, (where, got, ref)


@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs >= 2 MPI slots; set AXF_MPI=1")
@needs_3d
def test_3d_worst_element_is_reduced_over_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("AXF_SOLVER_MAXIT", "2")
    serial = json.loads(_core.run_3d(json.dumps(_slab_doc()), str(tmp_path / "np1"), 1, 0))
    res = af.Design(_slab_doc()).solve(tmp_path / "np2", verbose=0, mpi_ranks=2)
    assert res.data["mesh"]["mpi_ranks"] == 2
    # the real status file of axf (rank 0, --status-file, kept as axf_status.json): the summary, with the warnings
    status = json.loads((tmp_path / "np2" / "axf_status.json").read_text())
    assert status["ok"] is True and status["results"].endswith("results.json")
    assert status["warnings"] == res.data["warnings"]
    assert not list((tmp_path / "np2").glob("axf_status.*.json*"))   # no per-launch or temporary file left
    top = _check_message(res.data, _stalled_warnings(res.data))
    ref = serial["mesh"]["info"]["worst_elements"]
    _same_elements(res.data["mesh"]["info"]["worst_elements"]["worst"], ref["worst"])   # serial numbering on every rank
    assert top["element"] == ref["worst"][0]["element"]
    for key in ("elements", "inverted", "distorted_curved", "flat"):
        assert res.data["mesh"]["info"]["worst_elements"][key] == ref[key]


def test_2d_newton_stall_names_the_worst_element(tmp_path, monkeypatch):
    monkeypatch.setenv("AXF_AXI_MAX_NEWTON", "1")
    doc = {"schema": "axionfem.design/1", "name": "fe",
           "regions": [{"name": "coil", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.2, 0.2]}}, "material": "winding"},
                       {"name": "core", "shape": {"rect": {"r": [0.0, 0.4], "z": [-0.3, 0.3]}}, "material": "iron"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 3e6}],
           "mesh": {"order": 2, "size": 0.03, "surface_size": 0.01}}
    r = af.Design(doc).solve(tmp_path / "fe", verbose=0).data
    stalled = [w for w in r["warnings"] if "Newton did not converge" in w]
    assert stalled, r["warnings"]
    worst = r["mesh"]["info"]["worst_elements"]
    top = worst["worst"][0]
    assert worst["elements"] == r["mesh"]["elements"] and len(top["centroid_m"]) == 2
    assert f"element {top['element']} in region '{top['region']}'" in stalled[0]
    assert "centred at (r, z) = (" in stalled[0] and "Fix in the design:" in stalled[0]


def test_2d_newton_cap_is_validated_before_meshing(tmp_path, monkeypatch):
    monkeypatch.setenv("AXF_AXI_MAX_NEWTON", "zero")
    d = af.template("solenoid", mesh={"order": 1, "size": 0.2, "surface_size": 0.1})
    with pytest.raises(af.AxionFEMError, match="AXF_AXI_MAX_NEWTON"):
        d.solve(tmp_path / "s", verbose=0)
    assert not list((tmp_path / "s").glob("*.msh"))


def test_2d_factorization_failure_is_rethrown_with_the_check(tmp_path, monkeypatch):
    """SolverFailure from the poloidal engine (forced by the test hook) reaches Python as AxionFEMError with the
    engine's message and hint kept, the mesh check appended to the hint, and no pointer to results that a thrown
    error does not write."""
    monkeypatch.setenv("AXF_TEST_AXI_FACTORIZATION_FAILURE", "1")
    d = af.template("solenoid", mesh={"order": 1, "size": 0.2, "surface_size": 0.1})
    with pytest.raises(af.AxionFEMError) as info:
        d.solve(tmp_path / "s", verbose=0)
    text = str(info.value)
    assert text.startswith("the field system could not be factorized (singular or indefinite)")
    assert "Hint: Typical causes:" in text and "zero-volume region. Mesh check (run because this solve failed): " in text
    assert "centred at (r, z) = (" in text and "Fix in the design:" in text
    assert "worst_elements" not in text and "  Mesh" not in text
