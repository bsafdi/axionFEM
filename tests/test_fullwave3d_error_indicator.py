"""Error indicator of the 3D full-wave port fields (docs/physics/fullwave_error_indicator.md; FwErrorIndicator).

Per frequency and port the run reports the field's estimated energy-norm error, its magnetic and electric parts, where
it sits (regions, the largest element), the conductor faces' surface term and the estimate it implies for Z and L. At
order 1 it is a recovery estimate; at order 2 and above element residual problems give the magnetic part and a global
gradient projection the electric part, with a(e, e) estimated with sign.

The default suite runs the parallel-plate TEM line (exact Z, seconds per run): the effectivity against the exact error
at orders 1 and 2, indicator on and off giving identical results, the interior port face as a cut of the recovery and
of the residual problems, PML and Robin groups withholding the Z and L estimates, a failed solve withholding the
estimate, the signed estimate withheld where the two parts nearly cancel, the line half filled with a conducting
material (the complex-eps gradient projection), the projection skipped in the deep MQS limit, an interior pmc plane
that must change nothing and (with AXF_MPI) a projection that did not converge, withheld on every rank of 2. The slow
tests (AXF_SLOW) run the two-port frame: the per-port report, the partition, refinement, the gauged path and the off
switch; and the TEM line and the conducting line at order 2 on 1 and 4 ranks. Benchmark:
benchmarks/fullwave_indicator.py."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import axionfem as af
from axionfem.convergence import refine_and_compare

ROOT = Path(__file__).resolve().parents[1]
TWO_PORT = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
RANKS = 4 if os.environ.get("AXF_MPI") else 1
GMRES_PARTITION_REL = 1e-4   # the complex projection on 1 against 4 ranks: 3e-5 measured on the conducting line
sys.path.insert(0, str(ROOT / "benchmarks"))

import fullwave_indicator as FWI  # noqa: E402  (the TEM line's designs and exact Z)

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1")
pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"],
                                reason="needs the 3D engine (an MPI build; the wheels have none)")


def _two_port(f=1e7, order=1, export=None):
    doc = json.loads(TWO_PORT.read_text())
    for key in ("source_check", "validation_shunt_R_ohm", "validation_loads"):
        doc["fullwave"].pop(key, None)
    doc["fullwave"]["frequencies"] = [f]
    doc["mesh"]["order"] = order
    if export:
        doc["fullwave"]["field_export_directory"] = str(export)
    return doc


def _tem(wd, case, order, refinements, frequencies, mpi_ranks=1):
    """A TEM line run in this process; returns its results data and the rows against the exact Z."""
    if case == "interior_port":
        doc, geometry = FWI.tem_interior_port_design(wd, order, refinements, frequencies)
    elif case == "sheet":
        doc, geometry = FWI.tem_design(wd, "scattering", order, refinements, frequencies, sheet=FWI.TEM_SHEET)
    elif case == "pec_iso":   # cubic Kuhn cells
        doc, geometry = FWI.tem_design(wd, "pec", order, refinements, frequencies, nx=30, W=0.02)
    else:
        doc, geometry = FWI.tem_design(wd, case, order, refinements, frequencies)
    data = af.Design(doc).solve(wd, verbose=0, mpi_ranks=mpi_ranks).data
    exact = case in ("pec", "pec_iso", "interior_port")
    rows = [FWI.tem_row(row, FWI.tem_exact_z("interior_port" if case == "interior_port" else "pec", row["f_Hz"],
                                             geometry) if exact else None)
            for row in data["fullwave"]["sweep"]]
    return data, rows


# ------------------------------------------------------------------ default suite: the TEM line
def test_tem_line_effectivity_against_the_exact_error(tmp_path, monkeypatch):
    """Shorted line, 5:1 cells, 100 MHz (12 % of the norm electric). Order 1 (recovery): the L estimate is 1.4 times the
    exact error and the signed estimate 0.9-1.0 of it, on two meshes. Order 2 (element
    residual problems and the gradient projection): 0.99 and 0.90; the recovery was 0.38 and 0.31 there, because its
    space contains the discrete curl and so sees only its jumps (0.33 of the exact magnetic error)."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    for n in (0, 1):
        data, rows = _tem(tmp_path / f"p1_r{n}", "pec", 1, n, [1e8])
        row = rows[0]
        assert 1.2 < row["L_effectivity"] < 1.7
        assert 0.85 < row["L_signed_ratio"] < 1.1
        assert data["fullwave"]["error_indicator"]["cut_faces"] == 0   # the port is on the line's end face
        assert "recovery" in data["fullwave"]["error_indicator"]["method"]
        assert "local_order" not in data["fullwave"]["error_indicator"]
    data, rows = _tem(tmp_path / "p2_r0", "pec", 2, 0, [1e8, 2e8])
    describe = data["fullwave"]["error_indicator"]
    assert describe["local_order"] == 3 and "element residual" in describe["method"]
    assert 0.85 < rows[0]["L_effectivity"] < 1.15
    assert 0.8 < rows[0]["L_signed_ratio"] < 1.0
    # 200 MHz: 38 % of the norm electric; the unsigned estimate counts both parts. The parts are 4.7 apart, within the
    # factor 8 below which the signed estimate is withheld
    assert 1.2 < rows[1]["L_effectivity"] < 1.6
    assert rows[1]["indicator"]["L_signed_relative_error_estimate"] is None
    sweep = data["fullwave"]["sweep"]
    ind = sweep[0]["error_indicator"]
    # the signed Z estimate is a(e, e) / (i omega): its imaginary part is -omega L times the signed L estimate
    w = 2 * math.pi * sweep[0]["f_Hz"]
    dz = ind["Z_signed_error_estimate_ohm"]
    assert dz["im"] == pytest.approx(-w * sweep[0]["L_H"] * ind["L_signed_relative_error_estimate"], rel=1e-9)
    assert abs(dz["re"]) < 1e-6 * abs(dz["im"])   # lossless and closed
    assert ind["residual_seconds"] >= 0
    assert ind["gradient"]["converged"] and ind["gradient"]["iterations"] > 0 and "PCG" in ind["gradient"]["solver"]
    assert ind["gradient"]["relative_residual"] < 1e-4   # the projection's tolerance
    # the recovery's magnetic part, a diagnostic: about 0.33 of the exact magnetic error here, against the residual
    # problems' 0.89
    assert 0.2 < (ind["recovery_magnetic_relative_error"] / ind["magnetic_relative_error"]) ** 2 < 0.6


def test_signed_estimate_withheld_where_the_parts_nearly_cancel(tmp_path, monkeypatch):
    """Cubic cells at 600 MHz: the exact magnetic and electric errors are within 25 % of each other, so the change of
    L is a small difference of the two. At order 2 the estimated parts are within a factor 8 and the signed estimate
    is withheld (it had the wrong sign there). The unsigned estimate stays, and lies above the error of L here."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    data, rows = _tem(tmp_path / "iso", "pec_iso", 2, 0, [6e8])
    ind = data["fullwave"]["sweep"][0]["error_indicator"]
    assert ind["L_signed_relative_error_estimate"] is None and "factor 8" in ind["L_signed_withheld"]
    assert ind["Z_signed_error_estimate_ohm"] is None
    assert rows[0]["L_effectivity"] > 1


def test_indicator_on_and_off_give_identical_results(tmp_path, monkeypatch):
    """The indicator only reads the fields: every output outside its own keys and the timings is bit-identical."""
    def leaves(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from leaves(v, f"{path}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from leaves(v, f"{path}/{i}")
        else:
            yield path, x

    def kept(path):
        return not any(s in path for s in ("error_indicator", "seconds", "/timing/", "rss", "memory"))

    runs = {}
    for mode in ("on", "off"):
        monkeypatch.setenv("AXF_FW_ERROR_INDICATOR", mode)
        data, _ = _tem(tmp_path / "line", "pec", 2, 0, [1e8, 3e8])
        runs[mode] = {p: v for p, v in leaves(data) if kept(p)}
    assert runs["on"].keys() == runs["off"].keys()
    different = [p for p in runs["on"] if runs["on"][p] != runs["off"][p]]
    assert not different, different[:10]


def test_an_interior_port_face_cuts_the_recovery(tmp_path, monkeypatch):
    """The port on an interior plane between two shorted stubs carries the readout current, so tangential H jumps
    across it. Averaged across, the estimate would be 241-2039 times the exact error at order 1 and fall like h. As a
    cut, each side is recovered on its own; at order 2 the residual problems take each
    side's own traction there, and the gradient projection is zero on it."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    estimates = []
    for n in (0, 1):
        data, rows = _tem(tmp_path / f"r{n}", "interior_port", 1, n, [1e8])
        describe = data["fullwave"]["error_indicator"]
        assert describe["cut_faces"] == 8 * 4 ** n   # the port plane: 2 x 2 squares of 2 triangles, refined
        assert describe["cut_groups"] == ["lport1 (port)"]
        row = rows[0]
        assert 0.5 < row["L_effectivity"] < 3
        estimates.append(row["indicator"]["L_relative_error_estimate"])
    assert estimates[0] / estimates[1] > 3   # order 1: the error falls 4 times per level, not 2 as before
    data, rows = _tem(tmp_path / "p2", "interior_port", 2, 0, [1e8])
    assert 0.8 < rows[0]["L_effectivity"] < 1.2   # 0.97 (the recovery: 0.32)
    assert 0.8 < rows[0]["L_signed_ratio"] < 1.1   # 0.95


def test_a_lossy_fill_uses_the_complex_projection(tmp_path, monkeypatch):
    """The 5:1 line with its lower half filled by a conducting material (eps'' / eps' = 3): the loss phase of eps
    changes across the fill's surface, where the exact field keeps n . eps_c E continuous but not n . |eps| E. With
    |eps| weights the electric part was a mesh-independent artifact (0.48-0.49 on every mesh, a signed Z estimate of
    -5.4 + 10.7i ohm against changes of Z of 5e-5 ohm). The complex-eps projection
    converges with the field, and its signed estimate predicts the change of Z between the two finer meshes."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    doc = FWI.layered_line_design(tmp_path / "mesh", "sigma", [1e8])
    rows = []
    for n in (0, 1, 2):
        doc["mesh"]["refinements"] = n
        rows.append(af.Design(doc).solve(tmp_path / f"r{n}", verbose=0).data["fullwave"]["sweep"][0])
    electric = [row["error_indicator"]["electric_relative_error"] for row in rows]
    assert electric[0] / electric[1] > 1.3 and electric[1] / electric[2] > 2   # 1.5e-3, 9.8e-4, 4.1e-4
    for row in rows:
        gradient = row["error_indicator"]["gradient"]
        assert "complex eps" in gradient["solver"]
        # vacuum against arg eps = atan(sigma / omega eps0), about 72 degrees: the projection's energy can exceed the
        # electric error by up to 1 / cos^2(36 degrees)
        loss = FWI.LAYER_MATERIALS["sigma"]["sigma_S_m"] / (2 * math.pi * 1e8 * 8.8541878128e-12)
        assert gradient["loss_phase_spread_deg"] == pytest.approx(math.degrees(math.atan(loss)), rel=1e-6)
        assert 1.4 < gradient["electric_bound_factor"] < 1.6
    dz = [row["error_indicator"]["Z_signed_error_estimate_ohm"] for row in rows[1:]]
    assert abs(complex(dz[0]["re"], dz[0]["im"])) < 1e-3
    change = complex(rows[2]["Z_re_ohm"] - rows[1]["Z_re_ohm"], rows[2]["Z_im_ohm"] - rows[1]["Z_im_ohm"])
    predicted = complex(dz[1]["re"] - dz[0]["re"], dz[1]["im"] - dz[0]["im"])
    assert 0.5 < (predicted / change).real < 1.3 and abs((predicted / change).imag) < 0.2


@pytest.mark.parametrize("case", ["pec", "sigma"])
def test_the_projection_tolerance_override(tmp_path, monkeypatch, case):
    """AXF_FW_INDICATOR_TOL sets the gradient projection's relative tolerance (default 1e-4), for tests that compare
    two solvers at a tight tolerance (tests/test_gpu.py), on the PCG path (the lossless
    line) and the GMRES path (the conducting line): the results record it, the projection meets it, the estimates move
    by less than the default tolerance allows, and a value outside (0, 1e-4] is refused before the sweep."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    if case == "pec":
        doc, _ = FWI.tem_design(tmp_path / "mesh", "pec", 2, 0, [1e8])
    else:
        doc = FWI.layered_line_design(tmp_path / "mesh", "sigma", [1e8])
    rows = {}
    for tol in ("", "1e-10"):
        monkeypatch.setenv("AXF_FW_INDICATOR_TOL", tol)
        data = af.Design(doc).solve(tmp_path / f"run_{tol or 'default'}", verbose=0).data
        rows[tol] = data["fullwave"]["sweep"][0]["error_indicator"]
    assert rows[""]["gradient"]["tolerance"] == 1e-4
    tight = rows["1e-10"]["gradient"]
    assert tight["tolerance"] == 1e-10 and tight["converged"] is True
    assert 0 < tight["relative_residual"] <= 1e-10
    assert tight["iterations"] > rows[""]["gradient"]["iterations"]
    for key in ("estimated_relative_error", "electric_relative_error"):
        assert rows["1e-10"][key] == pytest.approx(rows[""][key], rel=1e-3), key
    for bad in ("abc", "0", "1", "-1e-6", "nan", "1e-3", "0.5"):   # (0, 1e-4] only: a looser one would lose energy
        monkeypatch.setenv("AXF_FW_INDICATOR_TOL", bad)
        with pytest.raises(Exception, match="AXF_FW_INDICATOR_TOL must be a number in"):
            af.Design(doc).solve(tmp_path / "bad", verbose=0)


@pytest.mark.parametrize("fill", ["sigma", None], ids=["complex_eps_presb", "abs_eps_pcg"])
def test_the_ridge_line_reproduces_the_stored_numbers(tmp_path, fill):
    """The CPU half of tests/test_gpu.py's ridge line (fullwave_indicator.ridge_line_design: a 270-degree conductor
    ridge, order 2, 0.3 and 1 GHz, lossless or half weakly conducting) against the stored CPU numbers
    (fullwave_indicator.RIDGE_LINE_CPU), to RIDGE_LINE_STORED_REL relative (the projection's tolerance). It runs
    `axf solve` in a subprocess, so that OpenBLAS's kernels can be pinned (OPENBLAS_CORETYPE=Haswell: MFEM builds its
    bases through LAPACK, and the kernel family moves the order-2 indicator; docs/validation/gpu.md, Open), with the
    projection at 1e-10 (AXF_FW_INDICATOR_TOL). Here, on CPU nodes, a change of the indicator or of the build trips
    it in the CPU suite, not first on a GPU node; refresh the stored numbers as their comment says."""
    from axionfem import _paths
    exe = _paths.axf_path()
    if not exe:
        pytest.skip("no axf executable found (set AXF_EXE, or build the package's)")
    wd = tmp_path / "run"
    wd.mkdir()
    doc = FWI.ridge_line_design(tmp_path / "mesh", fill)
    (wd / "design.json").write_text(json.dumps(doc))
    env = dict(os.environ)
    env.update({"AXF_DEVICE": "cpu", "OPENBLAS_CORETYPE": FWI.RIDGE_LINE_CORETYPE,
                "AXF_FW_INDICATOR_TOL": FWI.RIDGE_LINE_TOL})
    env.pop("AXF_FW_ERROR_INDICATOR", None)
    proc = subprocess.run([exe, "solve", str(wd / "design.json"), "-o", str(wd), "-v", "1"], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    sweep = json.loads((wd / "results.json").read_text())["fullwave"]["sweep"]
    assert [row["f_Hz"] for row in sweep] == FWI.RIDGE_LINE_FREQUENCIES
    keys = ("estimated_relative_error", "magnetic_relative_error", "electric_relative_error")
    for row in sweep:
        assert len(row["port_solutions"]) == 1
        ind = row["port_solutions"][0]["error_indicator"]
        gradient = ind["gradient"]
        assert gradient["tolerance"] == float(FWI.RIDGE_LINE_TOL) and gradient["converged"] is True, gradient
        assert gradient["iterations"] > 0 and 0 < gradient["relative_residual"] <= float(FWI.RIDGE_LINE_TOL), gradient
        assert ("complex eps" if fill else "PCG, |eps|") in gradient["solver"], gradient
        for key, stored in zip(keys, FWI.RIDGE_LINE_CPU[fill][row["f_Hz"]]):
            assert ind[key] == pytest.approx(stored, rel=FWI.RIDGE_LINE_STORED_REL, abs=0), (row["f_Hz"], key)


def test_the_projection_is_skipped_in_the_deep_mqs_limit(tmp_path, monkeypatch):
    """At 100 kHz the line's electric share of the norm is about 1e-7: below 1e-6 the gradient projection is not
    computed, and the estimate is the residual problems' alone. The electric error is then not estimated, so it is
    reported as null."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    data, rows = _tem(tmp_path / "mqs", "pec", 2, 0, [1e5])
    ind = data["fullwave"]["sweep"][0]["error_indicator"]
    assert ind["electric_norm_share"] < 1e-6
    assert "below 1e-6" in ind["gradient"]["skipped"]
    assert ind["electric_relative_error"] is None and "skipped" in ind["electric_relative_error_withheld"]
    assert ind["L_signed_relative_error_estimate"] < 0 and ind["estimated_relative_error"] > 0


def test_an_interior_pmc_plane_changes_nothing(tmp_path, monkeypatch):
    """The solver imposes nothing on an interior pmc face (a natural condition between two elements is no condition),
    so a pmc group on the TEM line's mid-plane leaves Z unchanged, and the order-2 estimate must be unchanged too. With
    zero traction there the residual problems counted the whole traction as error: 0.43 against 2.8e-4, a signed L
    estimate of -25 % against -1e-7."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    out = {}
    for plane in (True, False):
        doc = FWI.pmc_plane_design(tmp_path / f"mesh_{plane}", plane, 2, [1e8])
        out[plane] = af.Design(doc).solve(tmp_path / f"run_{plane}", verbose=0).data["fullwave"]["sweep"][0]
    assert out[True]["Z_im_ohm"] == pytest.approx(out[False]["Z_im_ohm"], rel=1e-12)
    for key in ("estimated_relative_error", "magnetic_relative_error", "electric_relative_error",
                "L_signed_relative_error_estimate"):
        assert out[True]["error_indicator"][key] == pytest.approx(out[False]["error_indicator"][key], rel=1e-9)
    assert out[True]["error_indicator"]["estimated_relative_error"] < 1e-3


@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs MPI ranks (AXF_MPI=1)")
@pytest.mark.parametrize("case", ["pec", "sigma"])
def test_a_projection_that_did_not_converge_withholds_on_every_rank(tmp_path, monkeypatch, case):
    """With the projection capped at one iteration (AXF_FW_INDICATOR_MAX_ITER) it does not converge, on the PCG path
    (the lossless line) and the GMRES path (the conducting line). Every port is withheld on every rank, and the run
    finishes: the decision is global (a rank-local norm could split the ranks across the collectives)."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    monkeypatch.setenv("AXF_FW_INDICATOR_MAX_ITER", "1")
    if case == "pec":
        doc, _ = FWI.tem_design(tmp_path / "mesh", "pec", 2, 0, [1e8, 2e8])
    else:
        doc = FWI.layered_line_design(tmp_path / "mesh", "sigma", [1e8, 2e8])
    data = af.Design(doc).solve(tmp_path / "run", verbose=0, mpi_ranks=2).data
    for row in data["fullwave"]["sweep"]:
        ind = row["error_indicator"]
        assert "did not converge" in ind["withheld"] and ind["gradient"]["converged"] is False
        assert ind["gradient"]["iterations"] == 1 and "estimated_relative_error" not in ind
    describe = data["fullwave"]["error_indicator"]
    assert describe["max_estimated_relative_error"] is None and describe["estimates_left_out_of_max"] == 2


@pytest.mark.parametrize("order", [1, 2])
def test_pml_and_robin_groups_withhold_the_z_and_l_estimates(tmp_path, monkeypatch, order):
    """Their terms are part of the operator but not of the norm (a PML line's Z estimate was 2e-5 of the true error),
    so the Z and L estimates are withheld; the field estimate stays. PML elements are left out of the field."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    data, rows = _tem(tmp_path / "pml", "pml", order, 0, [1e8])
    describe = data["fullwave"]["error_indicator"]
    assert describe["excluded_elements"]["pml"] > 0
    assert describe["field_elements"] + describe["excluded_elements"]["pml"] == data["mesh"]["elements"]
    assert "PML layers" in describe["terminations_left_out"]
    indicator = data["fullwave"]["sweep"][0]["error_indicator"]
    assert "PML layers" in indicator["Z_L_withheld"]
    assert "Z_error_estimate_ohm" not in indicator and "L_relative_error_estimate" not in indicator
    assert "Z_signed_error_estimate_ohm" not in indicator
    assert indicator["estimated_relative_error"] > 0
    data, rows = _tem(tmp_path / "sheet", "sheet", order, 0, [1e8])
    describe = data["fullwave"]["error_indicator"]
    assert "film (sheet)" in describe["terminations_left_out"] and "end (scattering)" in describe["terminations_left_out"]
    assert describe["cut_groups"] == ["film (sheet)"]   # the sheet is an interior face: a cut too
    indicator = data["fullwave"]["sweep"][0]["error_indicator"]
    assert "Z_L_withheld" in indicator and indicator["estimated_relative_error"] > 0


@pytest.mark.parametrize("order", [1, 2])
def test_a_failed_solve_withholds_its_estimate(tmp_path, monkeypatch, order):
    """A solve that did not pass its residual gate has no estimate (not a misleading 0),
    and the run's maximum leaves it out."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    monkeypatch.setenv("AXF_FW_MAX_ITER", "1")
    data, _ = _tem(tmp_path / "failed", "pec", order, 0, [1e8, 2e8])
    for row in data["fullwave"]["sweep"]:
        assert row["solver"]["accepted"] is False
        assert "residual gate" in row["error_indicator"]["withheld"]
        assert "estimated_relative_error" not in row["error_indicator"]
    describe = data["fullwave"]["error_indicator"]
    assert describe["max_estimated_relative_error"] is None and describe["max_at"] is None
    assert describe["estimates_left_out_of_max"] == 2


# ------------------------------------------------------------------ slow: the two-port frame (MPI)
@slow
def test_indicator_reports_per_frequency_and_port(tmp_path, monkeypatch):
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    r = af.Design(_two_port(export=tmp_path / "fields")).solve(tmp_path / "run", verbose=0, mpi_ranks=RANKS)
    fw = r.fullwave
    describe = fw["error_indicator"]
    assert describe["recovery_order"] == 1
    assert describe["field_elements"] > 0 and describe["conductor_faces"] > 0
    assert describe["cut_faces"] == 0 and describe["terminations_left_out"] == []
    assert describe["excluded_elements"]["perfect_conductor"] > 0 and describe["excluded_elements"]["shell"] > 0
    row = fw["sweep"][0]
    w = 2 * math.pi * row["f_Hz"]
    estimates = []
    for port in row["port_solutions"]:
        ind = port["error_indicator"]
        estimates.append(ind["estimated_relative_error"])
        # the total is the norm-weighted mix of the magnetic and electric parts
        s = ind["electric_norm_share"]
        mixed = (1 - s) * ind["magnetic_relative_error"] ** 2 + s * ind["electric_relative_error"] ** 2
        assert ind["estimated_relative_error"] ** 2 == pytest.approx(mixed, rel=1e-10)
        assert sum(v["share"] for v in ind["by_region"].values()) == pytest.approx(1, rel=1e-12)
        assert sum(v["norm_share"] for v in ind["by_region"].values()) == pytest.approx(1, rel=1e-12)
        z = complex(port["Z_re_ohm"], port["Z_im_ohm"])
        assert ind["Z_relative_error_estimate"] == pytest.approx(ind["Z_error_estimate_ohm"] / abs(z), rel=1e-12)
        assert ind["L_relative_error_estimate"] == pytest.approx(ind["Z_error_estimate_ohm"] / (w * port["L_H"]),
                                                                 rel=1e-12)
        # with sign: the magnetic part lowers L, the electric part raises it
        magnetic = (1 - s) * ind["magnetic_relative_error"] ** 2
        electric = s * ind["electric_relative_error"] ** 2
        signed = -(magnetic - electric) / (magnetic + electric) * ind["L_relative_error_estimate"]
        assert ind["L_signed_relative_error_estimate"] == pytest.approx(signed, rel=1e-9)
        assert ind["L_signed_relative_error_estimate"] < 0   # magnetic errors dominate on this frame at 10 MHz
        assert 0 < ind["largest_element"]["share"] <= 1
        assert 0 < ind["surface_relative_error"] < 1
        assert ind["curl_from"] == "curl representative"   # the Gauss-law guard's curl part
    assert row["error_indicator"] == row["port_solutions"][0]["error_indicator"]
    assert describe["max_estimated_relative_error"] == max(estimates)
    # its cost: the construction once, and the evaluations summed over the sweep's frequencies and ports
    assert describe["setup_seconds"] >= 0
    seconds = [port["error_indicator"]["seconds"] for port in row["port_solutions"]]
    assert describe["evaluation_seconds"] == pytest.approx(sum(seconds), rel=1e-12)
    # one L2 file per port and rank, beside the exported fields
    files = sorted((tmp_path / "fields").glob("f_0_port_*_error_indicator_rank_*.gf"))
    assert len(files) == 2 * RANKS
    assert "L2_3D_P0" in files[0].read_text()[:200]   # MFEM L2_FECollection(0, 3), Gauss-Legendre basis


@slow
@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs MPI ranks (AXF_MPI=1)")
def test_indicator_does_not_depend_on_the_partition(tmp_path):
    one = af.Design(_two_port()).solve(tmp_path / "one", verbose=0, mpi_ranks=1).fullwave["sweep"][0]
    four = af.Design(_two_port()).solve(tmp_path / "four", verbose=0, mpi_ranks=4).fullwave["sweep"][0]
    for a, b in zip(one["port_solutions"], four["port_solutions"]):
        for key in ("estimated_relative_error", "magnetic_relative_error", "electric_relative_error",
                    "surface_relative_error", "Z_error_estimate_ohm"):
            assert b["error_indicator"][key] == pytest.approx(a["error_indicator"][key], rel=1e-9)


@slow
@pytest.mark.skipif(not os.environ.get("AXF_MPI"), reason="needs MPI ranks (AXF_MPI=1)")
def test_order_2_estimate_does_not_depend_on_the_partition(tmp_path, monkeypatch):
    """The residual problems read the face neighbours of other ranks, and the gradient projection is a parallel solve.
    1 and 4 ranks agreed to 5e-9 on the shorted line and 3e-7 with the interior port. That difference is the field
    solve's algebraic error entering the magnetic part (it fell 500 times at AXF_FW_RTOL=1e-13), not the
    projection's tolerance; the recovery agreed to 3e-11."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)

    # the estimates the indicator may withhold (None), with the key of its reason: the signed L estimate at order 2
    # with the magnetic and electric parts within a factor 8 (at 0.2 GHz here), and the electric error when the
    # gradient projection is skipped. A withheld estimate must be withheld on both partitions, for the same reason;
    # every other estimate must be a number on both.
    withheld_by = {"L_signed_relative_error_estimate": "L_signed_withheld",
                   "electric_relative_error": "electric_relative_error_withheld"}

    def same(ia, ib, key, rel):
        x, y = ia[key], ib[key]
        if key in withheld_by and (x is None or y is None):
            assert x is None and y is None, (key, x, y)
            reason = withheld_by[key]
            assert ia.get(reason) and ia.get(reason) == ib.get(reason), (key, ia.get(reason), ib.get(reason))
        else:
            assert x is not None and y is not None, (key, x, y)
            assert y == pytest.approx(x, rel=rel), key

    for case in ("pec", "interior_port"):
        one = _tem(tmp_path / f"{case}_1", case, 2, 1, [1e8, 2e8], mpi_ranks=1)[0]["fullwave"]["sweep"]
        four = _tem(tmp_path / f"{case}_4", case, 2, 1, [1e8, 2e8], mpi_ranks=4)[0]["fullwave"]["sweep"]
        for a, b in zip(one, four):
            for key in ("estimated_relative_error", "magnetic_relative_error", "electric_relative_error",
                        "L_relative_error_estimate", "L_signed_relative_error_estimate"):
                same(a["error_indicator"], b["error_indicator"], key, 1e-5)
    # the GMRES path (the conducting line): its stopping test is on the diagonally scaled energy
    doc = FWI.layered_line_design(tmp_path / "sigma_mesh", "sigma", [1e8])
    doc["mesh"]["refinements"] = 1
    one = af.Design(doc).solve(tmp_path / "sigma_1", verbose=0, mpi_ranks=1).data["fullwave"]["sweep"][0]
    four = af.Design(doc).solve(tmp_path / "sigma_4", verbose=0, mpi_ranks=4).data["fullwave"]["sweep"][0]
    assert "GMRES" in four["error_indicator"]["gradient"]["solver"]
    for key in ("estimated_relative_error", "electric_relative_error", "L_signed_relative_error_estimate"):
        same(one["error_indicator"], four["error_indicator"], key, GMRES_PARTITION_REL)


@slow
@pytest.mark.skipif(not af.build_info().get("strumpack"),
                    reason="the gauged full-wave path (AXF_FW_SOLVER=strumpack) needs a STRUMPACK build")
def test_the_gauged_path_gives_the_same_estimate(tmp_path, monkeypatch):
    """The tree-cotree path's curl representative and fields agree with AMS and the Gauss-law guard (to about
    1.7e-11); so must the estimate."""
    monkeypatch.delenv("AXF_FW_ERROR_INDICATOR", raising=False)
    ams = af.Design(_two_port()).solve(tmp_path / "ams", verbose=0, mpi_ranks=RANKS).fullwave["sweep"][0]
    for key, value in (("AXF_FW_SOLVER", "strumpack"), ("AXF_FW_GAUGE", "tree-cotree")):
        monkeypatch.setenv(key, value)
    gauged = af.Design(_two_port()).solve(tmp_path / "gauged", verbose=0, mpi_ranks=RANKS).fullwave["sweep"][0]
    for a, b in zip(ams["port_solutions"], gauged["port_solutions"]):
        assert b["error_indicator"]["curl_from"] == "curl representative"
        for key in ("estimated_relative_error", "magnetic_relative_error", "electric_relative_error",
                    "surface_relative_error", "L_relative_error_estimate"):
            assert b["error_indicator"][key] == pytest.approx(a["error_indicator"][key], rel=1e-8)


@slow
def test_indicator_tracks_the_error_under_refinement(tmp_path):
    """The two-port frame at order 1 on its mesh and one uniform refinement (the same geometry, h halved). Its fields
    are singular at the frame's edges, and L converges like h^1.2 (benchmarks/fullwave_indicator.py, three levels:
    the estimate of L's error is about 0.85-1.18 times the error against the extrapolated L)."""
    out = refine_and_compare(_two_port(f=1e5), tmp_path, levels=2, mpi_ranks=RANKS)
    frequency = out["frequencies"][0]
    coarse, fine = frequency["rows"]
    assert fine["L_H"] > coarse["L_H"]   # a current-driven port's FEM L converges from below
    # against the refined mesh, which still carries about 0.45 of the coarse mesh's error: effectivity 1.6
    assert 0.8 < coarse["effectivity"] < 3
    # the estimate falls as the error does (2.24x per level here)
    assert 1.8 < coarse["L_relative_error_estimate"] / fine["L_relative_error_estimate"] < 3
    # the signed estimates predict the change of L (8.4e-2 for an observed 1.0e-1)
    change = frequency["changes"][0]
    assert 0.6 < change["predicted_change"] / change["L_change_relative"] < 1.5


@slow
def test_indicator_off_and_bad_values(tmp_path, monkeypatch):
    monkeypatch.setenv("AXF_FW_ERROR_INDICATOR", "off")
    r = af.Design(_two_port()).solve(tmp_path / "off", verbose=0, mpi_ranks=RANKS)
    assert "error_indicator" not in r.fullwave
    assert "error_indicator" not in r.fullwave["sweep"][0]
    monkeypatch.setenv("AXF_FW_ERROR_INDICATOR", "sometimes")
    with pytest.raises(Exception, match="AXF_FW_ERROR_INDICATOR must be on or off"):
        af.Design(_two_port()).solve(tmp_path / "bad", verbose=0, mpi_ranks=RANKS)
