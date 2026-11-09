"""A CUDA build's 3D solves with hypre on the GPU agree with the same build on the CPU (docs/validation/gpu.md).

Runs only where it can mean something, and is skipped everywhere else: the executable that the solves run ($AXF_EXE,
else the package's own `axf`) must be a CUDA build (it accepts AXF_DEVICE=cuda), and the CUDA driver must see a GPU
from this process (cuDeviceGetCount, which honours CUDA_VISIBLE_DEVICES). $AXF_DEVICE is read once per process
(cpp/src/device.cpp), so the two solves of each design are `axf solve` subprocesses of the same executable, one with
AXF_DEVICE=cpu and one with AXF_DEVICE=cuda, on the same mesh. Each test asserts where hypre ran
(results.json: solver.hypre_device, or the mechanics' AMG label).

What runs on the GPU is hypre: AMS in the MQS and full-wave H(curl) solves, AME in the cavity eigensolver, BoomerAMG
in the H1 solves (Gauss-law guard, port feeds, divergence cleaning, windings) and in the mechanics. It is not the same
algorithm on both devices: the H1 BoomerAMG keeps MFEM's GPU defaults on a GPU (PMIS coarsening, l1-Jacobi
smoothing), and the mechanics use unknown-based systems AMG there (no rigid-body modes). The iteration counts differ,
so the comparison is at the solver tolerance, not at round-off. The order-2 error indicator's gradient projection
(BoomerAMG on the GPU; PCG on the |eps| path, GMRES with PRESB on the complex-eps path) is checked on both paths: it
can fail silently on a GPU (docs/validation/gpu.md).
"""
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import axionfem as af
from axionfem import _paths

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import fullwave_indicator as FWI  # noqa: E402  (the singular lines of the error indicator's benchmark)


def _gpu_count() -> int:
    """GPUs the CUDA driver shows this process (0 without a driver, or with CUDA_VISIBLE_DEVICES empty)."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return 0
    if cuda.cuInit(0) != 0:
        return 0
    n = ctypes.c_int(0)
    if cuda.cuDeviceGetCount(ctypes.byref(n)) != 0:
        return 0
    return n.value


def _cuda_exe() -> tuple[str | None, str]:
    """The axf executable the solves run when it accepts AXF_DEVICE=cuda (a -DAXF_GPU=CUDA build), else None and why."""
    exe = _paths.axf_path()
    if not exe:
        return None, "no axf executable found (set AXF_EXE, or build the package's)"
    env = dict(os.environ)
    env["AXF_DEVICE"] = "cuda"
    try:
        proc = subprocess.run([exe, "templates"], capture_output=True, text=True, env=env, timeout=120)
    except (OSError, subprocess.SubprocessError) as err:
        return None, f"the axf executable {exe} does not run: {err}"
    if proc.returncode != 0:
        return None, (f"the axf executable {exe} is not a CUDA build (configure with -DAXF_GPU=CUDA; docs/install.md, "
                      "GPU builds)")
    return exe, ""


EXE, EXE_REASON = _cuda_exe()

pytestmark = [
    pytest.mark.skipif(EXE is None, reason=EXE_REASON),
    pytest.mark.skipif(_gpu_count() == 0, reason="no GPU visible to this process (CUDA driver, CUDA_VISIBLE_DEVICES)"),
]


# The OpenBLAS kernel family of every solve. MFEM builds the ND and H1 bases through LAPACK, and OpenBLAS picks its
# kernels at run time (SkylakeX on AVX-512 hosts, Zen on AMD EPYC hosts); the bases' last bits, and with them the
# order-2 indicator, depend on it.
# Haswell runs on every node and gives Zen's bits, so the stored CPU numbers below hold on any host.
OPENBLAS_CORETYPE = FWI.RIDGE_LINE_CORETYPE


def _solve(doc: dict, wd: Path, device: str, extra_env: dict | None = None) -> dict:
    wd.mkdir(parents=True, exist_ok=True)
    design = wd / "design.json"
    design.write_text(json.dumps(doc))
    env = dict(os.environ)
    env["AXF_DEVICE"] = device
    env["OPENBLAS_CORETYPE"] = OPENBLAS_CORETYPE
    env.update(extra_env or {})
    proc = subprocess.run([EXE, "solve", str(design), "-o", str(wd), "-v", "1"], capture_output=True, text=True,
                          env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads((wd / "results.json").read_text())


def _both(doc: dict, tmp_path: Path) -> tuple[dict, dict]:
    """The design solved with hypre on the CPU and on the GPU, on the same mesh."""
    host = _solve(doc, tmp_path / "cpu", "cpu")
    dev = _solve(doc, tmp_path / "cuda", "cuda")
    assert (tmp_path / "cpu" / "mesh3d.msh").read_bytes() == (tmp_path / "cuda" / "mesh3d.msh").read_bytes()
    return host, dev


def _where(host: dict, dev: dict) -> None:
    """hypre ran on the CPU in the host run and on the GPU in the device run."""
    assert host["solver"]["hypre_device"] == "cpu"
    assert dev["solver"]["hypre_device"].startswith("cuda:"), dev["solver"]["hypre_device"]


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _z(row: dict) -> np.ndarray:
    return np.array(row["Z_matrix_re_ohm"]) + 1j * np.array(row["Z_matrix_im_ohm"])


def _dz(row: dict) -> np.ndarray:
    return np.array(row["dZ_domega_H"]["re"]) + 1j * np.array(row["dZ_domega_H"]["im"])


def test_mqs_solenoid_on_the_gpu_matches_the_cpu(tmp_path):
    """3D MQS with an infinity shell (the coarse 3D solenoid of test_3d.py): L_p, Psi and W at the PCG tolerance."""
    from axionfem.convert import to_3d
    d2 = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.015})
    doc = to_3d(d2, mesh={"size": 0.4, "surface_size": 0.13, "order": 1}).doc
    host, dev = _both(doc, tmp_path)
    _where(host, dev)
    for key in ("L_p_H", "Psi_SI_H_T_m2", "W_mag_J"):
        assert dev["coupling"][key] == pytest.approx(host["coupling"][key], rel=1e-8), key
    assert dev["solver"]["max_relative_residual"] < 1e-9
    assert not [w for w in dev["warnings"] if "did not converge" in w or "iteration cap" in w]


def test_fullwave_two_port_frame_on_the_gpu_matches_the_cpu(tmp_path):
    """The two-port frame: the MQS pickup and an AMS-GMRES sweep with the Gauss-law guard on (its default), two
    ports, validation shunts and the source check: Z, dZ/domega and the Psi vector at the GMRES tolerance."""
    doc = json.loads((ROOT / "benchmarks" / "designs" / "stage1_twoport.json").read_text())
    doc["fullwave"]["frequencies"] = [1e5, 1.2e7]
    host, dev = _both(doc, tmp_path)
    _where(host, dev)
    assert dev["coupling"]["L_p_H"] == pytest.approx(host["coupling"]["L_p_H"], rel=1e-8)
    assert dev["fullwave"]["solver_path"] == "AMS-GMRES"
    frequencies = [[row["f_Hz"] for row in r["fullwave"]["sweep"]] for r in (host, dev)]
    assert frequencies[0] == frequencies[1] == doc["fullwave"]["frequencies"], frequencies
    for rh, rd in zip(host["fullwave"]["sweep"], dev["fullwave"]["sweep"]):
        assert rd["gauss_law_guard"]["applied"] and rh["gauss_law_guard"]["applied"]
        assert rd["validation_accepted"]
        assert _rel(_z(rd), _z(rh)) < 1e-7, rd["f_Hz"]
        assert _rel(_dz(rd), _dz(rh)) < 1e-7, rd["f_Hz"]
        psi_h = np.array(rh["Psi_vector"]["re"]) + 1j * np.array(rh["Psi_vector"]["im"])
        psi_d = np.array(rd["Psi_vector"]["re"]) + 1j * np.array(rd["Psi_vector"]["im"])
        assert _rel(psi_d, psi_h) < 1e-7, rd["f_Hz"]


def test_cavity_wall_losses_on_the_gpu_match_the_cpu(tmp_path):
    """The TM010 pillbox of test_cavity3d.py with the AME eigensolver (AMS-preconditioned LOBPCG, hypre's) and
    skin-effect walls: f, the form factor, and the wall losses (G, Q0, Q_wall), whose wall-current solve on the wall
    DOFs can lose them on a GPU without a word (rows that hypre leaves empty there).

    The tolerances follow the eigenvector: AME stops at a mode residual of about 1e-5 on either device, so a pointwise
    peak (first order in the eigenvector's error) agrees to about 3e-5, G and Q to about 5e-7, the form factor to 5e-10
    and f (second order) to 2e-11 (measured on A40 and H100). A lost or garbage G is off by its whole value."""
    from axionfem import cavity as cv
    wall = {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.7e-8, "T_K": 4.0}}
    doc = cv.cylinder_design_3d(0.1, 0.2, h=0.02, modes=1, wall_material=wall).doc
    host, dev = _both(doc, tmp_path)
    _where(host, dev)
    for r in (host, dev):
        assert not [w for w in r["warnings"] if "wall-current" in w], r["warnings"]
    mh = host["fullwave"]["eigenmodes"]["modes"][0]
    md = dev["fullwave"]["eigenmodes"]["modes"][0]
    assert md["f_Hz"] == pytest.approx(mh["f_Hz"], rel=1e-9)
    assert md["form_factor"]["C"] == pytest.approx(mh["form_factor"]["C"], rel=1e-7)
    lh = mh["losses"]
    ld = md["losses"]
    for key in ("G_ohm", "Q0", "Q_wall"):
        assert ld[key] == pytest.approx(lh[key], rel=1e-5), key
    assert list(ld["walls"]) == ["outer"]
    assert ld["walls"]["outer"]["G_ohm"] == pytest.approx(lh["walls"]["outer"]["G_ohm"], rel=1e-5)
    assert ld["wall_current_solve"]["relative_residual"] < 1e-10
    peak_h = mh["peak_wall_fields"]["B_t_max_flux"]
    assert md["peak_wall_fields"]["B_t_max_flux"] == pytest.approx(peak_h, rel=3e-4, abs=0.0)
    assert ld["G_ohm"] == pytest.approx(301.99, rel=5e-3)   # the analytic TM010 geometry factor (test_cavity3d.py)


def test_mechanics_cooldown_on_the_gpu_matches_the_cpu(tmp_path):
    """The free cooldown of test_mechanics3d.py (an unsupported L-shaped body, six rigid motions pinned): the same
    displacements, while the results name the AMG each device ran (systems AMG on a GPU, without the rigid-body modes
    that hypre's GPU BoomerAMG cannot take; several times the CPU's PCG iterations)."""
    boxes = [{"box": {"min": [0, 0, 0], "max": [0.4, 0.1, 0.1]}}, {"box": {"min": [0, 0, 0], "max": [0.1, 0.3, 0.1]}}]
    pts = [[0.35, 0.05, 0.05], [0.05, 0.25, 0.05], [0.2, 0.08, 0.02]]
    doc = {"schema": "axionfem.design/1", "name": "cool3d", "symmetry": "3d",
           "regions": [{"name": "body", "material": "m", "solid": {"union": boxes}, "mesh_size": 0.05}],
           "materials": {"m": {"kind": "linear", "mechanical": {"E_Pa": 1.2e11, "nu": 0.3, "alpha": 2.5e-3}}},
           "mesh": {"order": 2, "size": 0.1, "surface_size": 0.05, "far_size": 0.4, "domain_radius": 1.0},
           "mechanics": {"cooldown": True, "probes": pts}}
    host, dev = _both(doc, tmp_path)
    _where(host, dev)
    mh = host["mechanics"]
    md = dev["mechanics"]
    assert mh["solver"]["amg"] == "elasticity (rigid-body modes)"
    assert md["solver"]["amg"].startswith("systems"), md["solver"]["amg"]
    uh = np.array([p["u_m"] for p in mh["probes"]])
    ud = np.array([p["u_m"] for p in md["probes"]])
    assert _rel(ud, uh) < 1e-7
    assert not [w for w in dev["warnings"] if "did not converge" in w or "iteration cap" in w]


# The ridge line, its environment and its stored CPU numbers: fullwave_indicator.ridge_line_design, RIDGE_LINE_TOL
# (AXF_FW_INDICATOR_TOL; the default is 1e-4), RIDGE_LINE_CORETYPE (= OPENBLAS_CORETYPE here) and RIDGE_LINE_CPU, which
# tests/test_fullwave3d_error_indicator.py also checks on CPU nodes (so an indicator change trips it without a GPU).
TIGHT_TOL = FWI.RIDGE_LINE_TOL
# the GPU run against the CPU run, both at TIGHT_TOL: at most 1.7e-10 measured (the magnetic part, which follows the
# field's GMRES solve), against a factor 52-79 for the silent failure described in the test below
TOL_DEVICES = 1e-7
# the CPU run against the stored numbers: the same bits on an A40 node with a Xeon Gold 6230 host; the check allows
# the projection's own tolerance (fullwave_indicator.RIDGE_LINE_STORED_REL), as on CPU nodes
TOL_STORED = FWI.RIDGE_LINE_STORED_REL


@pytest.mark.parametrize("fill", ["sigma", None], ids=["complex_eps_presb", "abs_eps_pcg"])
def test_order2_indicator_on_the_gpu_matches_the_cpu(tmp_path, fill):
    """The order-2 error indicator's gradient projection on the shorted line with a perfect-conductor ridge, whose
    elements are outside the field, so that every projection operator has rows that are zero there: the rows hypre on
    a GPU drops (docs/validation/gpu.md). Two paths:
    - complex_eps_presb: the line's upper half weakly conducting (0.0167 S/m; loss phases 45 and 17 degrees at 0.3 and
      1 GHz), so the complex-eps projection (GMRES with PRESB: BoomerAMG of A_re + A_im on the GPU, the Jacobi scaling
      from that sum's diagonal; A_im is zero in the vacuum half, more empty rows);
    - abs_eps_pcg: the lossless line, the |eps| projection (hypre's PCG with BoomerAMG).

    If those rows are not kept, the |eps| projection on the two-port frame at ND2 on an A40 returns after 0
    iterations with an electric part 79 times too small, with no warning and every solve unchanged; on this line
    complex_eps_presb stops in BoomerAMG's setup (hypre error 12), and abs_eps_pcg reports 0 iterations, residual 0,
    "converged" and an electric part 52 times too small. So the projection must
    have iterated and converged on both devices, and the indicator's parts must agree with the CPU's and with the
    stored CPU numbers.

    Both devices solve the projection to TIGHT_TOL (AXF_FW_INDICATOR_TOL), so that the comparison is sharp on both
    paths. At the default 1e-4 the GPU's BoomerAMG (PMIS, l1-Jacobi) gives another
    iterate, and on the GMRES path, which is not a Galerkin method, the parts move to first order in the iterate's
    error: 0.7-2.3e-5 measured on A40s, against at most 8.3e-10 on the PCG path. The GPU is compared with the CPU
    first, then the CPU run with the stored numbers (relative 1e-12, no absolute slack; the same OpenBLAS kernels on
    every host), so that stale stored numbers cannot hide the device comparison."""
    frequencies = FWI.RIDGE_LINE_FREQUENCIES
    doc = FWI.ridge_line_design(tmp_path / "mesh", fill)
    tight = {"AXF_FW_INDICATOR_TOL": TIGHT_TOL}
    host = _solve(doc, tmp_path / "cpu", "cpu", tight)
    dev = _solve(doc, tmp_path / "cuda", "cuda", tight)
    _where(host, dev)
    assert host["mesh"]["dofs"] == dev["mesh"]["dofs"]
    for r in (host, dev):
        assert [row["f_Hz"] for row in r["fullwave"]["sweep"]] == frequencies
        assert all(len(row["port_solutions"]) == 1 for row in r["fullwave"]["sweep"])
    solver = "complex eps" if fill else "PCG, |eps|"
    keys = ("estimated_relative_error", "magnetic_relative_error", "electric_relative_error")
    for rh, rd in zip(host["fullwave"]["sweep"], dev["fullwave"]["sweep"]):
        f = rd["f_Hz"]
        assert _rel(_z(rd), _z(rh)) < 1e-7, f
        ih = rh["port_solutions"][0]["error_indicator"]
        idev = rd["port_solutions"][0]["error_indicator"]
        for ind in (ih, idev):
            gradient = ind["gradient"]
            assert solver in gradient["solver"], (f, gradient)
            assert gradient["tolerance"] == float(TIGHT_TOL), (f, gradient)
            assert gradient["converged"] is True, (f, gradient)
            assert gradient["iterations"] > 0, (f, gradient)
            assert 0 < gradient["relative_residual"] <= float(TIGHT_TOL), (f, gradient)
            assert "withheld" not in ind, (f, ind.get("withheld"))
        for key in keys:
            assert idev[key] == pytest.approx(ih[key], rel=TOL_DEVICES, abs=0), (f, key)
    for rh in host["fullwave"]["sweep"]:
        ih = rh["port_solutions"][0]["error_indicator"]
        for key, stored in zip(keys, FWI.RIDGE_LINE_CPU[fill][rh["f_Hz"]]):
            assert ih[key] == pytest.approx(stored, rel=TOL_STORED, abs=0), (rh["f_Hz"], key)
