"""S/Y/Z parameters and Touchstone v1 export/import of full-wave port results (axionfem.touchstone).

Closed forms (series RLC one-port, lossless and lossy TEM line two-ports), the time-convention conjugation, the
two-port frame's multiport record (reciprocity, lossless and lossy passivity, the reversed port), the passivity warning
without clipping, round trips for S, Z and Y with 1-5 ports, the v1 reader's units/formats/2-port order, the
adaptive dense curve, the command line and the MCP tool; scikit-rf cross-checks when it is importable.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from axionfem import circuit as ck
from axionfem import touchstone as ts

ROOT = Path(__file__).resolve().parents[1]
TWO_PORT_RECORD = ROOT / "benchmarks" / "records" / "multiport_stage1.json"
C0 = 299792458.0


def _rlc(f, R=3.0, L=2e-7, C=1e-11):
    """Series RLC in the engine's exp(-i omega t) convention: Z = R - i omega L + i/(omega C)."""
    w = 2 * np.pi * np.asarray(f)
    return R - 1j * w * L + 1j / (w * C)


def _results_from_rows(rows, ports=None, name="unit"):
    fw = {"sweep": rows}
    if ports is not None:
        fw["ports"] = [{"name": p} for p in ports]
    return {"design": {"name": name}, "fullwave": fw}


def _matrix_rows(f, Z):
    """Sweep rows (engine convention) carrying Z_matrix like the 3D engine."""
    rows = []
    for fi, M in zip(f, Z):
        M = np.atleast_2d(M)
        rows.append({"f_Hz": float(fi), "Z_matrix_re_ohm": M.real.tolist(), "Z_matrix_im_ohm": M.imag.tolist(),
                     "Z_re_ohm": float(M[0, 0].real), "Z_im_ohm": float(M[0, 0].imag)})
    return rows


def _line_z(f, l, Zc, alpha=0.0, v=C0):
    """Engineering (exp(+j omega t)) Z matrix of a uniform line: Z11 = Zc coth(gl), Z12 = Zc csch(gl)."""
    g = alpha + 1j * 2 * np.pi * np.asarray(f) / v
    gl = g * l
    z11 = Zc / np.tanh(gl)
    z12 = Zc / np.sinh(gl)
    return np.stack([np.stack([z11, z12], -1), np.stack([z12, z11], -1)], -2)


# ------------------------------------------------------------------ conventions and closed forms
def test_convention_inductor_is_plus_j_omega_l():
    """The engine's Z = -i omega L becomes +j omega L; an inductor's S11 turns clockwise (Im S11 > 0 at low f)."""
    w = 2 * np.pi * 1e6
    Ze = -1j * w * 1e-6
    assert ts.to_engineering(Ze, "axionfem") == pytest.approx(1j * w * 1e-6)
    S = ts.z_to_s(Ze, 50, convention="axionfem")[0, 0]
    z = 1j * w * 1e-6 / 50
    assert S == pytest.approx((z - 1) / (z + 1), rel=1e-14)
    assert S.imag > 0 and abs(S) == pytest.approx(1, abs=1e-14)
    with pytest.raises(ValueError, match="convention"):
        ts.z_to_s(Ze, 50, convention="physics")
    with pytest.raises(TypeError):
        ts.z_to_s(Ze, 50)   # the convention is required


def test_series_rlc_s_and_y_closed_form(tmp_path):
    f = np.linspace(1e7, 1e8, 41)
    Ze = _rlc(f)
    w = 2 * np.pi * f
    Zeng = 3.0 + 1j * w * 2e-7 - 1j / (w * 1e-11)
    S = ts.z_to_s(Ze, 50, convention="axionfem")[:, 0, 0]
    assert np.allclose(S, (Zeng - 50) / (Zeng + 50), rtol=1e-13, atol=1e-15)
    Y = ts.z_to_y(Ze, convention="axionfem")[:, 0, 0]
    assert np.allclose(Y, 1 / Zeng, rtol=1e-13)
    # at resonance S11 = (R - 50)/(R + 50), real
    f0 = 1 / (2 * np.pi * math.sqrt(2e-7 * 1e-11))
    s0 = ts.z_to_s(_rlc(f0), 50, convention="axionfem")[0, 0]
    assert s0 == pytest.approx((3 - 50) / 53, abs=1e-12)
    res = _results_from_rows([{"f_Hz": float(x), "Z_re_ohm": float(z.real), "Z_im_ohm": float(z.imag)}
                              for x, z in zip(f[::-1], Ze[::-1])], ports=["pickup"])
    rep = ts.write_touchstone(res, tmp_path / "rlc.s1p")
    back = ts.read_touchstone(tmp_path / "rlc.s1p")
    assert np.array_equal(back["f_Hz"], f)   # sorted on output
    assert np.allclose(back["S"][:, 0, 0], (Zeng - 50) / (Zeng + 50), rtol=1e-15, atol=1e-16)
    assert np.allclose(back["Z"][:, 0, 0], Zeng, rtol=1e-12)
    assert rep["passivity_violations_Hz"] == [] and rep["sigma_max"] < 1
    assert back["ports"] == ["pickup"]


@pytest.mark.parametrize("alpha", [0.0, 0.3])
def test_matched_tem_line_two_port(alpha, tmp_path):
    """A line of Zc = z0: S11 = S22 = 0, S21 = S12 = exp(-gamma l) (a phase delay in exp(+j omega t))."""
    f = np.linspace(5e7, 4e8, 30)
    l = 0.37
    Zeng = _line_z(f, l, 50.0, alpha)
    S = ts.z_to_s(np.conj(Zeng), 50, convention="axionfem")
    g = alpha + 1j * 2 * np.pi * f / C0
    assert np.max(np.abs(S[:, 0, 0])) < 1e-12 and np.max(np.abs(S[:, 1, 1])) < 1e-12
    assert np.allclose(S[:, 1, 0], np.exp(-g * l), atol=1e-12)
    assert np.allclose(S[:, 0, 1], S[:, 1, 0], atol=1e-14)
    res = _results_from_rows(_matrix_rows(f, np.conj(Zeng)), ports=["a", "b"])
    rep = ts.write_touchstone(res, tmp_path / "line.s2p")
    sig = ts.passivity(S)["sigma_max"]
    if alpha == 0:
        assert np.allclose(sig, 1, atol=1e-12)
    else:
        assert np.all(sig < 1) and np.allclose(sig, math.exp(-alpha * l), rtol=1e-10)
    assert rep["passivity_violations_Hz"] == [] and rep["reciprocity_error"] < 1e-14
    # the 2-port order on disk: f S11 S21 S12 S22
    first = [ln for ln in (tmp_path / "line.s2p").read_text().splitlines() if ln and ln[0].isdigit()][0].split()
    assert complex(float(first[3]), float(first[4])) == pytest.approx(S[0, 1, 0], abs=1e-15)
    back = ts.read_touchstone(tmp_path / "line.s2p")
    assert np.allclose(back["S"], S, rtol=1e-15, atol=1e-15)
    assert back["ports"] == ["a", "b"]


def test_mismatched_line_against_cascade_formula():
    """Zc != z0: S11 = Gamma (1 - P^2)/(1 - Gamma^2 P^2), S21 = P (1 - Gamma^2)/(1 - Gamma^2 P^2)."""
    f = np.linspace(1e7, 6e8, 25)
    l, Zc, z0 = 0.21, 75.0, 50.0
    S = ts.z_to_s(_line_z(f, l, Zc), z0, convention="engineering")
    G = (Zc - z0) / (Zc + z0)
    P = np.exp(-1j * 2 * np.pi * f / C0 * l)
    assert np.allclose(S[:, 0, 0], G * (1 - P ** 2) / (1 - G ** 2 * P ** 2), atol=1e-12)
    assert np.allclose(S[:, 1, 0], P * (1 - G ** 2) / (1 - G ** 2 * P ** 2), atol=1e-12)


def test_per_port_z0_and_inverse():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3))
    Z = A @ np.conj(np.swapaxes(A, -1, -2)) + (A + np.swapaxes(A, -1, -2)) * 1j   # passive-ish, complex
    z0 = [25.0, 50.0, 75.0]
    S = ts.z_to_s(Z, z0, convention="engineering")
    assert np.allclose(ts.s_to_z(S, z0), Z, rtol=1e-11, atol=1e-11)
    # one-port definition element by element: S_kk of a diagonal Z
    D = np.diag([10.0, 60.0, 200.0]).astype(complex)
    Sd = ts.z_to_s(D, z0, convention="engineering")
    assert np.allclose(np.diag(Sd), [(10 - 25) / 35, (60 - 50) / 110, (200 - 75) / 275])
    assert np.allclose(ts.y_to_z(ts.z_to_y(Z, convention="engineering")), Z, rtol=1e-11)
    for bad in ([50, 50], [50, -1, 50], [50, 50j, 50]):
        with pytest.raises(ValueError, match="z0"):
            ts.z_to_s(Z, bad, convention="engineering")


# ------------------------------------------------------------------ records: the two-port frame
def _two_port(case):
    return json.loads(TWO_PORT_RECORD.read_text())["cases"][case]


def test_two_port_record_reciprocity_and_passivity(tmp_path):
    """Lossless frame: S symmetric to 1e-11 and sigma_max = 1 to roundoff; lossy: sigma_max < 1; reversed port 2:
    S12 changes sign, S11 and S22 do not; no passivity violation."""
    out = {}
    for case in ("multiport", "lossy", "reversed"):
        c = _two_port(case)
        res = _results_from_rows(c["sweep"], ports=["pickup", "capacitor_1"], name=case)
        net = ts.port_parameters(res)
        S = ts.z_to_s(net["Z"], 50, convention="engineering")
        out[case] = (net, S)
        rep = ts.write_touchstone(res, tmp_path / f"{case}.s2p")
        assert rep["passivity_violations_Hz"] == []
        assert rep["reciprocity_error"] < 1e-11
        assert np.max(ts.reciprocity_error(S)) < 1e-11
        # Im Z of the engine is negative (inductive): positive in the file's convention
        assert np.all(net["Z"][:, 0, 0].imag > 0)
    sig = ts.passivity(out["multiport"][1])["sigma_max"]
    assert np.allclose(sig, 1, atol=1e-13)
    lossy = ts.passivity(out["lossy"][1])
    assert np.all(lossy["sigma_max"] < 1 - 1e-9)
    # dissipation: 1 - S^H S is positive definite; 1 - sigma = 3.3e-8 comes from the resistance matrix's smallest
    # eigenvalue (0.82 micro-ohm, resistance_min_eigenvalue_ohm: the two ports are almost perfectly coupled)
    Sl = out["lossy"][1][0]
    assert np.linalg.eigvalsh(np.eye(2) - Sl.conj().T @ Sl).min() > 1e-9
    a, b = out["multiport"][1], out["reversed"][1]
    assert np.allclose(a[:, 0, 0], b[:, 0, 0], atol=1e-12) and np.allclose(a[:, 1, 1], b[:, 1, 1], atol=1e-12)
    assert np.allclose(a[:, 0, 1], -b[:, 0, 1], atol=1e-12)


def test_two_port_record_oneport_without_matrix(tmp_path):
    """A one-port row with only Z_re_ohm/Z_im_ohm (the 2D engine and the old main path)."""
    c = _two_port("oneport_main")
    res = _results_from_rows(c["sweep"], ports=["pickup"])
    rep = ts.write_touchstone(res, tmp_path / "one.s1p", kind="Z")
    back = ts.read_touchstone(tmp_path / "one.s1p")
    want = [complex(r["Z_re_ohm"], -r["Z_im_ohm"]) for r in c["sweep"]]
    assert np.allclose(back["Z"][:, 0, 0], want, rtol=1e-15)
    L = back["Z"][:, 0, 0].imag / (2 * np.pi * back["f_Hz"])
    assert np.allclose(L, [r["L_H"] for r in c["sweep"]], rtol=1e-12)
    assert rep["ports"] == ["pickup"]


# ------------------------------------------------------------------ passivity reporting, never clipping
def test_passivity_violation_is_warned_and_written_unclipped(tmp_path):
    f = np.array([1e6, 2e6])
    Ze = np.array([-0.5 - 3j, 1.0 - 6j])   # negative resistance at 1 MHz
    res = _results_from_rows([{"f_Hz": float(x), "Z_re_ohm": z.real, "Z_im_ohm": z.imag} for x, z in zip(f, Ze)])
    with pytest.warns(RuntimeWarning, match="not passive at 1 of 2"):
        rep = ts.write_touchstone(res, tmp_path / "gain.s1p")
    assert rep["passivity_violations_Hz"] == [1e6]
    back = ts.read_touchstone(tmp_path / "gain.s1p")
    assert abs(back["S"][0, 0, 0]) > 1
    assert back["S"][0, 0, 0] == pytest.approx(ts.z_to_s(Ze[0], 50, convention="axionfem")[0, 0], rel=1e-15)


# ------------------------------------------------------------------ round trips and the file format
@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("kind", ["S", "Z", "Y"])
def test_round_trip(n, kind, tmp_path):
    rng = np.random.default_rng(10 * n + len(kind))
    f = np.sort(rng.uniform(1e5, 1e9, 7))
    A = rng.normal(size=(7, n, n)) + 1j * rng.normal(size=(7, n, n))
    R = A @ np.conj(np.swapaxes(A, -1, -2))
    X = rng.normal(size=(7, n, n))
    Zeng = R.real + 1j * (X + np.swapaxes(X, -1, -2)) * 40   # reciprocal, passive
    rep = ts.write_touchstone_data(tmp_path / f"x.s{n}p", f, Zeng, convention="engineering", kind=kind, z0=37.5,
                                   ports=[f"p{k}" for k in range(n)])
    assert rep["passivity_violations_Hz"] == []
    back = ts.read_touchstone(tmp_path / f"x.s{n}p")
    assert back["kind"] == kind and back["z0_ohm"] == 37.5
    assert np.array_equal(back["f_Hz"], f)
    assert np.allclose(back["Z"], Zeng, rtol=1e-10, atol=1e-10 * np.abs(Zeng).max())
    if kind == "S":
        assert np.array_equal(back["S"], ts.z_to_s(Zeng, 37.5, convention="engineering"))
    if kind == "Z":
        assert np.allclose(back["data"], Zeng, rtol=1e-15, atol=0)   # 17 digits: exact up to the normalisation
    text = (tmp_path / f"x.s{n}p").read_text().splitlines()
    assert f"# Hz {kind} RI R 37.5" in text
    assert all(len(ln.split()) <= 1 + 8 for ln in text if not ln.startswith(("!", "#")))
    assert back["ports"] == [f"p{k}" for k in range(n)]


def test_reader_units_formats_and_noise(tmp_path):
    """MA and DB in GHz/kHz, the v1 default option line, values over several lines, 2-port noise data."""
    p = tmp_path / "ma.s2p"
    p.write_text("! hand written\n# GHz S MA R 50\n"
                 "1.0 0.5 90 0.8 -45\n 0.8 -45 0.25 180\n"
                 "2.0 0.4 0 0.7 -90 0.7 -90 0.2 0\n"
                 "! noise parameters\n1.0 1.2 0.3 45 0.1\n")
    with pytest.warns(RuntimeWarning, match="ignored 1 noise-parameter rows"):
        r = ts.read_touchstone(p)
    assert np.array_equal(r["f_Hz"], [1e9, 2e9])
    assert r["S"][0, 0, 0] == pytest.approx(0.5j, abs=1e-15)
    assert r["S"][0, 1, 0] == pytest.approx(0.8 * np.exp(-1j * np.pi / 4))
    assert r["S"][0, 1, 1] == pytest.approx(-0.25)
    q = tmp_path / "db.s1p"
    q.write_text("1 -6.020599913279624 0\n2 0 180\n")   # no option line: GHz S MA... but DB needs the line
    with pytest.warns(RuntimeWarning, match="no option line"):
        r = ts.read_touchstone(q)
    assert r["f_Hz"][0] == 1e9 and r["S"][0, 0, 0] == pytest.approx(-6.020599913279624)  # read as MA by default
    q.write_text("# kHz S DB R 75\n1 -6.020599913279624 0\n2 0 180\n")
    r = ts.read_touchstone(q)
    assert r["f_Hz"][0] == 1e3 and r["z0_ohm"] == 75
    assert r["S"][0, 0, 0] == pytest.approx(0.5, rel=1e-12) and r["S"][1, 0, 0] == pytest.approx(-1)
    v2 = tmp_path / "v2.s1p"
    v2.write_text("[Version] 2.0\n# Hz S RI R 50\n")
    with pytest.raises(ValueError, match="Touchstone v2"):
        ts.read_touchstone(v2)


def test_writer_refusals(tmp_path):
    f = [1e6, 2e6]
    Z = np.array([[[1 + 1j]], [[2 + 1j]]])
    with pytest.raises(ValueError, match=r"must end in '\.s1p'"):
        ts.write_touchstone_data(tmp_path / "x.s2p", f, Z, convention="engineering")
    with pytest.raises(TypeError):
        ts.write_touchstone_data(tmp_path / "x.s1p", f, Z)   # the convention is required
    Z2 = np.array([np.eye(2) * 10, np.eye(2) * 20], dtype=complex)
    with pytest.raises(ValueError, match="only for S-parameters"):
        ts.write_touchstone_data(tmp_path / "x.s2p", f, Z2, convention="engineering", kind="Z", z0=[25.0, 50.0])
    with pytest.raises(ValueError, match="kind"):
        ts.write_touchstone_data(tmp_path / "x.s1p", f, Z, convention="engineering", kind="H")
    with pytest.raises(ValueError, match="strictly increasing"):
        ts.write_touchstone_data(tmp_path / "x.s1p", [2e6, 1e6], Z, convention="engineering")
    rows = [{"f_Hz": 1e6, "Z_re_ohm": 1.0, "Z_im_ohm": -1.0}, {"f_Hz": 1e6, "Z_re_ohm": 1.0, "Z_im_ohm": -1.0}]
    with pytest.raises(ValueError, match="repeat"):
        ts.write_touchstone(_results_from_rows(rows), tmp_path / "x.s1p")
    with pytest.raises(ValueError, match="no 'fullwave'"):
        ts.write_touchstone({}, tmp_path / "x.s1p")
    with pytest.raises(ValueError, match="adaptive"):
        ts.write_touchstone(_results_from_rows(rows[:1]), tmp_path / "x.s1p", dense=True)


def test_surface_losses_option(tmp_path):
    rows = [{"f_Hz": 1e6, "Z_re_ohm": 0.0, "Z_im_ohm": -10.0, "R_loss_ohm": 0.01},
            {"f_Hz": 2e6, "Z_re_ohm": 0.0, "Z_im_ohm": -20.0, "R_loss_ohm": 0.04}]
    net = ts.port_parameters(_results_from_rows(rows), surface_losses=True)
    assert np.allclose(net["Z"][:, 0, 0], [0.01 + 10j, 0.04 + 20j])
    assert "PLUS the resistive part" in net["notes"][0]
    # a multiport needs the in-phase R_loss_matrix_ohm
    Zm = np.array([[-10j, -3j], [-3j, -12j]])
    mrows = _matrix_rows([1e6], [Zm])
    with pytest.raises(ValueError, match="R_loss_matrix_ohm"):
        ts.port_parameters(_results_from_rows(mrows), surface_losses=True)
    mrows[0]["R_loss_matrix_ohm"] = {"re": [[0.01, 0.002], [0.002, 0.02]], "im": [[0, 0], [0, 0]]}
    net = ts.port_parameters(_results_from_rows(mrows), surface_losses=True)
    assert np.allclose(net["Z"][0], np.conj(Zm) + [[0.01, 0.002], [0.002, 0.02]])
    mrows[0]["Z_matrix_re_ohm"] = [[0.5, 0.0], [0.0, 0.5]]   # radiation: not in phase
    with pytest.raises(ValueError, match="not in phase"):
        ts.port_parameters(_results_from_rows(mrows), surface_losses=True)


# ------------------------------------------------------------------ the adaptive dense curve
def _adaptive_results(f, Z):
    """A results document with an adaptive block whose dense curve is Z (engine convention) and whose model is a
    single support point (only fullwave_dense's format matters here)."""
    n = Z.shape[-1]
    dense = {"f_Hz": list(map(float, f)), "Z_re_ohm": Z[:, 0, 0].real.tolist(), "Z_im_ohm": Z[:, 0, 0].imag.tolist(),
             "L_H": (-Z[:, 0, 0].imag / (2 * np.pi * f)).tolist(), "Psi_re": [0.0] * len(f), "Psi_im": [0.0] * len(f),
             "Z_matrix_re_ohm": Z.real.tolist(), "Z_matrix_im_ohm": Z.imag.tolist(),
             "Psi_vector": {"re": np.zeros((len(f), n)).tolist(), "im": np.zeros((len(f), n)).tolist()},
             "dZ_domega_H": {"re": np.zeros_like(Z.real).tolist(), "im": np.zeros_like(Z.real).tolist()},
             "estimated_relative_error": [1e-6] * len(f)}
    rows = _matrix_rows(f[::10], Z[::10])
    return {"design": {"name": "adaptive_unit"},
            "fullwave": {"sweep": rows, "ports": [{"name": f"p{k}"} for k in range(n)],
                         "adaptive": {"converged": True, "settings": {"tolerance": 1e-5}, "dense": dense,
                                      "model": {"support_Hz": [1.0]}}}}


def test_dense_curve_export(tmp_path):
    f = np.linspace(1e8, 4e8, 101)
    Zeng = _line_z(f, 0.3, 50.0, 0.05)
    res = _adaptive_results(f, np.conj(Zeng))
    rep = ts.write_touchstone(res, tmp_path / "dense.s2p", dense=True)
    assert rep["n_frequencies"] == 101 and "adaptive model" in rep["source"]
    back = ts.read_touchstone(tmp_path / "dense.s2p")
    assert np.allclose(back["Z"], Zeng, rtol=1e-12)
    rep_rows = ck.write_touchstone(res, tmp_path / "rows.s2p")   # the circuit-level function
    assert rep_rows["n_frequencies"] == 11 and "solved rows" in rep_rows["source"]
    res["fullwave"]["adaptive"]["converged"] = False
    with pytest.warns(RuntimeWarning, match="did not converge"):
        ts.write_touchstone(res, tmp_path / "dense2.s2p", dense=True)


# ------------------------------------------------------------------ command line and MCP
def test_command_line(tmp_path, capsys):
    from axionfem._cli import main

    res = _results_from_rows(_two_port("lossy")["sweep"], ports=["pickup", "capacitor_1"], name="frame")
    rp = tmp_path / "results.json"
    rp.write_text(json.dumps(res))
    assert main(["export-touchstone", str(rp), "--kind", "Y", "--z0", "25"]) == 0
    assert "frame_Y.s2p" in capsys.readouterr().out
    back = ts.read_touchstone(tmp_path / "frame_Y.s2p")
    assert back["kind"] == "Y" and back["z0_ohm"] == 25
    assert main(["export-touchstone", str(rp), "-o", str(tmp_path / "bad.s3p")]) == 2
    assert "must end in '.s2p'" in capsys.readouterr().err
    # every failure is 2 with a message on stderr, never a traceback with Python's 1
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2]")
    assert main(["export-touchstone", str(lst)]) == 2
    assert "not a results document" in capsys.readouterr().err
    # written but not passive: its own code, 3
    gain = _results_from_rows([{"f_Hz": 1e6, "Z_re_ohm": -0.5, "Z_im_ohm": -3.0}], name="gain")
    gp = tmp_path / "gain.json"
    gp.write_text(json.dumps(gain))
    with pytest.warns(RuntimeWarning, match="not passive"):
        assert main(["export-touchstone", str(gp)]) == 3
    assert (tmp_path / "gain_S.s1p").exists()
    # the default stem is sanitised: no path escape from the design name
    esc = _results_from_rows([{"f_Hz": 1e6, "Z_re_ohm": 1.0, "Z_im_ohm": -3.0}], name="../escaped/x")
    ep = tmp_path / "sub" / "esc.json"
    ep.parent.mkdir()
    ep.write_text(json.dumps(esc))
    assert main(["export-touchstone", str(ep)]) == 0
    assert not (tmp_path / "escaped").exists()
    assert (ep.parent / "_escaped_x_S.s1p").exists()


def test_mcp_tool(tmp_path):
    from axionfem import _mcp_modules as mm

    ws = mm.Workspace(tmp_path)
    (ws.runs / "frame").mkdir(parents=True)
    res = _results_from_rows(_two_port("multiport")["sweep"], ports=["pickup", "capacitor_1"], name="frame")
    (ws.runs / "frame" / "results.json").write_text(json.dumps(res))
    out = mm.tool_export_touchstone(ws, "frame")
    assert out["ok"] and out["path"].endswith("frame_S.s2p") and out["passivity_violations_Hz"] == []
    assert not out["overwrote"]
    out_y = mm.tool_export_touchstone(ws, "frame", kind="Y")
    assert out_y["path"].endswith("frame_Y.s2p") and Path(out["path"]).exists()   # S not replaced by Y
    assert mm.tool_export_touchstone(ws, "frame")["overwrote"]
    assert Path(out["path"]).parent == ws.runs / "frame" / "touchstone"
    with pytest.raises(FileNotFoundError, match="solve_design"):
        mm.tool_export_touchstone(ws, "missing")


# ------------------------------------------------------------------ scikit-rf cross-check (optional)
def test_against_scikit_rf(tmp_path):
    skrf = pytest.importorskip("skrf")
    f = np.linspace(1e7, 5e8, 23)
    Zeng = _line_z(f, 0.4, 75.0, 0.2)
    # Y files are left out: scikit-rf 2.2 de-normalises a v1 Y file by multiplying with R, as for Z, while the v1
    # normalised admittance is Y R (docs/physics/touchstone.md)
    for kind in ("S", "Z"):
        p = tmp_path / f"line_{kind}.s2p"
        ts.write_touchstone_data(p, f, Zeng, convention="engineering", kind=kind, z0=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = skrf.Network(str(p))
        assert np.allclose(net.f, f)
        assert np.allclose(net.s, ts.z_to_s(Zeng, 50, convention="engineering"), atol=1e-12)
        assert np.allclose(net.z, Zeng, rtol=1e-9)
        assert np.allclose(net.y, np.linalg.inv(Zeng), rtol=1e-9)
    p = tmp_path / "line_Y.s2p"
    ts.write_touchstone_data(p, f, Zeng, convention="engineering", kind="Y", z0=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = skrf.Network(str(p))
    assert np.allclose(net.y, np.linalg.inv(Zeng) * 50 ** 2, rtol=1e-9)   # its convention, documented
    # and the other way: scikit-rf's own conversion and writer
    net = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), z=Zeng, z0=50)
    assert np.allclose(net.s, ts.z_to_s(Zeng, 50, convention="engineering"), atol=1e-12)
    net.write_touchstone(str(tmp_path / "skrf_out"))
    back = ts.read_touchstone(tmp_path / "skrf_out.s2p")
    assert np.allclose(back["Z"], Zeng, rtol=1e-6)
    z0 = np.array([30.0, 80.0])
    net = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), z=Zeng, z0=z0, s_def="power")
    assert np.allclose(net.s, ts.z_to_s(Zeng, z0, convention="engineering"), atol=1e-12)


def test_results_method_default_path(tmp_path):
    """Results.write_touchstone names the file after the design and the port count in the work directory."""
    import types

    from axionfem.results import Results

    r = Results.__new__(Results)
    r.data = _results_from_rows(_two_port("multiport")["sweep"], ports=["pickup", "capacitor_1"])
    r.data.pop("design")
    r.design = types.SimpleNamespace(name="frame")
    r.workdir = tmp_path
    rep = r.write_touchstone(kind="Z")
    assert rep["path"] == str(tmp_path / "frame_Z.s2p")
    assert "! design frame" in (tmp_path / "frame_Z.s2p").read_text()


# ---------------------------------------------------------- file layout, normalisation, robustness
def _nonreciprocal_z(f):
    """A passive but non-reciprocal 2-port (engineering convention): the lossy line plus a gyrator (a real
    antisymmetric Z adds nothing to the Hermitian part, so the network stays passive)."""
    return _line_z(f, 0.3, 50.0, 0.1) + np.array([[0, 5], [-5, 0]])


def _data_lines(path):
    return [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip() and ln.lstrip()[0] not in "!#"]


def test_two_port_tokens_on_disk_distinguish_s21_from_s12(tmp_path):
    """With S21 != S12 the raw tokens are f, S11, S21, S12, S22 (the v1 2-port order), and they read back."""
    f = np.array([1e8, 2e8])
    Z = _nonreciprocal_z(f)
    S = ts.z_to_s(Z, 50, convention="engineering")
    assert np.min(np.abs(S[:, 1, 0] - S[:, 0, 1])) > 1e-2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts.write_touchstone_data(tmp_path / "nr.s2p", f, Z, convention="engineering")   # passive: no warning
    for line, M in zip(_data_lines(tmp_path / "nr.s2p"), S):
        v = [float(x) for x in line]
        got = [complex(v[k], v[k + 1]) for k in (1, 3, 5, 7)]
        assert got == pytest.approx([M[0, 0], M[1, 0], M[0, 1], M[1, 1]], abs=1e-15)
    back = ts.read_touchstone(tmp_path / "nr.s2p")
    assert np.array_equal(back["S"], S)
    assert np.allclose(back["Z"], Z, rtol=1e-12)
    # the same matrix as Z: tokens are Z11/R, Z21/R, Z12/R, Z22/R
    ts.write_touchstone_data(tmp_path / "nrz.s2p", f, Z, convention="engineering", kind="Z")
    v = [float(x) for x in _data_lines(tmp_path / "nrz.s2p")[0]]
    assert complex(v[3], v[4]) == pytest.approx(Z[0, 1, 0] / 50, abs=1e-15)
    assert complex(v[5], v[6]) == pytest.approx(Z[0, 0, 1] / 50, abs=1e-15)


def test_three_port_rows_on_disk(tmp_path):
    """N >= 3 is written row by row (row j of the matrix on line j of the block)."""
    M = (np.arange(9).reshape(3, 3) + 1) * (1 + 0.5j) + np.diag([100, 200, 300])
    ts.write_touchstone_data(tmp_path / "x.s3p", [1e6], M[None], convention="engineering", kind="Z", z0=1)
    lines = _data_lines(tmp_path / "x.s3p")
    assert len(lines) == 3 and len(lines[0]) == 7 and len(lines[1]) == 6
    rows = [lines[0][1:], lines[1], lines[2]]
    for j, tok in enumerate(rows):
        vals = [complex(float(tok[2 * k]), float(tok[2 * k + 1])) for k in range(3)]
        assert vals == pytest.approx(list(M[j]), abs=1e-12)
    assert np.allclose(ts.read_touchstone(tmp_path / "x.s3p")["Z"][0], M, rtol=1e-15)


def test_raw_normalised_values(tmp_path):
    """Z = 25 ohm at R = 50 is stored as 0.5 in a Z file and 2 in a Y file."""
    for kind, want in (("Z", "0.5 0"), ("Y", "2 0")):
        ts.write_touchstone_data(tmp_path / f"r{kind}.s1p", [1e6], [25.0], convention="engineering", kind=kind)
        line = _data_lines(tmp_path / f"r{kind}.s1p")[0]
        assert " ".join(line[1:]) == want
        assert ts.read_touchstone(tmp_path / f"r{kind}.s1p")["Z"][0, 0, 0] == pytest.approx(25.0, rel=1e-15)


def test_reader_on_hand_written_nonreciprocal_two_port(tmp_path):
    """A hand-written v1 2-port with S21 != S12 (an isolator-like block)."""
    p = tmp_path / "iso.s2p"
    p.write_text("# MHz S RI R 50\n100 0.1 0  0.9 0  0.01 0  0.2 0\n")
    S = ts.read_touchstone(p)["S"][0]
    assert S[1, 0] == pytest.approx(0.9) and S[0, 1] == pytest.approx(0.01)
    assert S[0, 0] == pytest.approx(0.1) and S[1, 1] == pytest.approx(0.2)


def test_reader_files_without_z(tmp_path):
    """An ideal thru, an open and a series element given as Y are valid files; S and the data are always
    returned, Z is NaN where it does not exist, with a warning naming the file and the frequencies."""
    thru = tmp_path / "thru.s2p"
    thru.write_text("# Hz S RI R 50\n1e6 0 0 1 0 1 0 0 0\n2e6 0.1 0 0.8 0 0.8 0 0.1 0\n")
    with pytest.warns(RuntimeWarning, match=r"thru\.s2p: Z does not exist at 1 frequencies \(1000000 Hz"):
        r = ts.read_touchstone(thru)
    assert np.array_equal(r["S"][0], [[0, 1], [1, 0]])
    assert np.all(np.isnan(r["Z"][0])) and np.all(np.isfinite(r["Z"][1]))
    assert r["Z_undefined_Hz"] == [1e6]
    op = tmp_path / "open.s1p"
    op.write_text("# Hz S RI R 50\n1e6 1 0\n")
    with pytest.warns(RuntimeWarning, match="Z does not exist"):
        r = ts.read_touchstone(op)
    assert r["S"][0, 0, 0] == 1
    ser = tmp_path / "series.s2p"   # a 50-ohm series element: Y = [[1, -1], [-1, 1]]/50, stored Y R
    ser.write_text("# Hz Y RI R 50\n1e6 1 0 -1 0 -1 0 1 0\n")
    with pytest.warns(RuntimeWarning, match="Z does not exist"):
        r = ts.read_touchstone(ser)
    # a series Z between 50-ohm ports: S11 = Z/(Z + 100) = 1/3, S21 = 100/(Z + 100) = 2/3
    assert np.allclose(r["S"][0], [[1 / 3, 2 / 3], [2 / 3, 1 / 3]], atol=1e-15)
    assert np.allclose(r["data"][0], np.array([[1, -1], [-1, 1]]) / 50)
    with pytest.raises(ValueError, match=r"Z \+ z0 is singular at matrix index \[0\]"):
        ts.z_to_s(-50.0, 50, convention="engineering")


def test_comment_injection_is_sanitised(tmp_path):
    """A newline in a design or port name cannot inject a data line."""
    rows = [{"f_Hz": 1e6, "Z_re_ohm": 1.0, "Z_im_ohm": -3.0}]
    res = _results_from_rows(rows, ports=["a\n2 0.5 0"], name="a\n1 2 3\r4 5 6")
    ts.write_touchstone(res, tmp_path / "inj.s1p")
    back = ts.read_touchstone(tmp_path / "inj.s1p")
    assert back["f_Hz"].tolist() == [1e6]
    assert back["ports"] == ["a 2 0.5 0"]


def test_surface_loss_beyond_perturbation_warns(tmp_path):
    """R_loss/|X| > 0.1 is warned about and listed; the note says the surface reactance is not added."""
    rows = [{"f_Hz": 1e6, "Z_re_ohm": 0.0, "Z_im_ohm": -1.0, "R_loss_ohm": 5.0, "R_loss_over_abs_X": 5.0},
            {"f_Hz": 2e6, "Z_re_ohm": 0.0, "Z_im_ohm": -20.0, "R_loss_ohm": 0.04}]
    with pytest.warns(RuntimeWarning, match="R_loss/\\|X\\| exceeds 0.1 at 1 rows"):
        rep = ts.write_touchstone(_results_from_rows(rows), tmp_path / "sl.s1p", surface_losses=True)
    assert rep["surface_loss_beyond_perturbation_Hz"] == [1e6]
    assert any("exceeds" in w for w in rep["warnings"])
    text = (tmp_path / "sl.s1p").read_text()
    assert "surface reactance (internal or kinetic inductance) is NOT included" in text
    assert "perturbation not accurate" in text


def test_y_conditioning(tmp_path):
    """A graded but well-determined matrix is not flagged; nearly dependent ports are; a
    one-port near a series resonance has Z itself poorly determined (reported for every kind, no Y warning), and the
    unsigned indicator is reported as an upper estimate."""
    graded = np.array([[1e-6j, -1e-9j], [-1e-9j, -6e9j]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Y = ts.z_to_y(graded, convention="engineering")
    assert np.allclose(Y @ graded, np.eye(2), atol=1e-12)
    dependent = np.array([[1j, 1j], [1j, 1j * (1 + 1e-14)]])
    with pytest.warns(RuntimeWarning, match="equilibrated"):
        ts.z_to_y(dependent, convention="engineering")
    ei = {"Z_error_estimate_ohm": 1e-3}
    rows = [{"f_Hz": 1e6, "Z_re_ohm": 1e-5, "Z_im_ohm": -1e-4, "error_indicator": ei},
            {"f_Hz": 2e6, "Z_re_ohm": 1.0, "Z_im_ohm": -100.0, "error_indicator": ei}]
    for kind in "SZY":
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rep = ts.write_touchstone(_results_from_rows(rows), tmp_path / f"y{kind}.s1p", kind=kind)
        assert rep["Z_poorly_determined_Hz"] == [1e6] and rep["Y_ill_determined_Hz"] == []
        assert "upper estimate" in (tmp_path / f"y{kind}.s1p").read_text()
    net = ts.port_parameters(_results_from_rows(rows))
    assert net["Z_relative_error"] == pytest.approx([1e-3 / abs(1e-5 - 1e-4j), 1e-3 / abs(1 - 100j)])
    # the signed estimate is preferred where given
    rows[1]["error_indicator"] = {"Z_error_estimate_ohm": 1e-3, "Z_signed_error_estimate_ohm": {"re": 0, "im": 1e-5}}
    net = ts.port_parameters(_results_from_rows(rows))
    assert net["Z_relative_error"][1] == pytest.approx(1e-5 / abs(1 - 100j))
    # no estimate at all (a direct 2D solve): reported, Y not checked
    bare = [{"f_Hz": 1e6, "Z_re_ohm": 1e-12, "Z_im_ohm": -1e-12}]
    rep = ts.write_touchstone(_results_from_rows(bare), tmp_path / "bare.s1p", kind="Y")
    assert rep["Y_check"].startswith("no error estimate: Y not checked")
    assert "no error estimate of Z" in (tmp_path / "bare.s1p").read_text()


def test_per_port_reference_v11(tmp_path):
    """S with per-port R is written and read as the Version 1.1 option line; Z/Y with it are refused."""
    f = np.array([1e8, 2e8])
    Z = _nonreciprocal_z(f)
    z0 = [25.0, 75.0]
    rep = ts.write_touchstone_data(tmp_path / "pp.s2p", f, Z, convention="engineering", z0=z0)
    assert rep["z0_ohm"] == z0
    text = (tmp_path / "pp.s2p").read_text()
    assert "# Hz S RI R 25 75" in text and "v1.1" in text
    back = ts.read_touchstone(tmp_path / "pp.s2p")
    assert back["z0_ohm"] == z0
    assert np.array_equal(back["S"], ts.z_to_s(Z, z0, convention="engineering"))
    assert np.allclose(back["Z"], Z, rtol=1e-12)
    zf = tmp_path / "ppz.s2p"
    zf.write_text("# Hz Z RI R 25 75\n1 1 0 0 0 0 0 1 0\n")
    with pytest.raises(ValueError, match="per-port reference resistances with Z-parameters are not supported"):
        ts.read_touchstone(zf)
    bad = tmp_path / "bad.s2p"
    bad.write_text("# Hz S RI R 25 50 75\n1 1 0 0 0 0 0 1 0\n")
    with pytest.raises(ValueError, match="3 reference resistances for 2 ports"):
        ts.read_touchstone(bad)


def test_reader_strictness(tmp_path):
    """Data before the option line is refused; the noise break is validated and reported."""
    p = tmp_path / "late.s1p"
    p.write_text("1 0.5 0\n# MHz S RI R 50\n2 0.5 0\n")
    with pytest.raises(ValueError, match="option line comes after data"):
        ts.read_touchstone(p)
    rep = tmp_path / "rep.s2p"   # a repeated frequency in the network data, followed by more network rows
    rep.write_text("# Hz S RI R 50\n" + "1 0 0 1 0 1 0 0 0\n" * 2 + "2 0 0 1 0 1 0 0 0\n")
    with pytest.raises(ValueError, match="not rows of 5"):
        ts.read_touchstone(rep)
    good = tmp_path / "noise.s2p"
    good.write_text("# Hz S RI R 50\n1 0.1 0 0.8 0 0.8 0 0.1 0\n2 0.1 0 0.8 0 0.8 0 0.1 0\n"
                    "1 1.2 0.3 45 0.1\n2 1.3 0.3 40 0.1\n")
    with pytest.warns(RuntimeWarning, match="ignored 2 noise-parameter rows"):
        r = ts.read_touchstone(good)
    assert r["noise_rows_ignored"] == 2 and r["f_Hz"].tolist() == [1.0, 2.0]


def test_violations_at_unresolved_rows_and_dense_flags(tmp_path):
    """A violation at a Re_Z_unresolved row is named as such; the dense curve carries the
    rows' flags."""
    rows = [{"f_Hz": 1e6, "Z_re_ohm": -1e-6, "Z_im_ohm": -40.0,
             "Re_Z_unresolved": {"true_relative_residual": 1e-6}},
            {"f_Hz": 2e6, "Z_re_ohm": 0.1, "Z_im_ohm": -80.0}]
    with pytest.warns(RuntimeWarning, match="1 of them are at rows whose Re Z the solve's residual does not resolve"):
        rep = ts.write_touchstone(_results_from_rows(rows), tmp_path / "u.s1p")
    assert rep["passivity_violations_at_unresolved_Re_Z_Hz"] == [1e6]
    f = np.linspace(1e8, 4e8, 101)
    res = _adaptive_results(f, np.conj(_line_z(f, 0.3, 50.0, 0.05)))
    res["fullwave"]["sweep"][3]["Re_Z_unresolved"] = {"true_relative_residual": 1e-7}
    net = ts.port_parameters(res, dense=True)
    assert net["Re_Z_unresolved_Hz"] == [res["fullwave"]["sweep"][3]["f_Hz"]]
    assert net["Re_Z_unresolved"].sum() == 20   # the dense points between the flagged solve and its neighbours
    assert any("adaptive rational model" in x for x in net["notes"])
    assert any("fitted to 1 solved rows whose Re Z" in x for x in net["notes"])


def test_port_polarity_in_the_comment(tmp_path):
    """Each port line carries its orientation (toward) and terminals; the reader keeps the name."""
    rows = _two_port("reversed")["sweep"]
    res = {"design": {"name": "rev"}, "fullwave": {"sweep": rows, "ports": [
        {"name": "pickup", "toward": [1, 0, 0], "region": "gap0", "terminal_patches": 2},
        {"name": "capacitor_1", "toward": [-1, 0, 0], "region": "gap1", "terminal_patches": 2}]}}
    ts.write_touchstone(res, tmp_path / "rev.s2p")
    text = (tmp_path / "rev.s2p").read_text()
    assert "!   port 2: capacitor_1 | toward [-1, 0, 0]; region gap1; terminal_patches 2" in text
    assert ts.read_touchstone(tmp_path / "rev.s2p")["ports"] == ["pickup", "capacitor_1"]


def test_committed_record_passes_its_gates():
    """The committed validation record holds every gate of benchmarks/touchstone_check.py."""
    import sys
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("touchstone_check", ROOT / "benchmarks" / "touchstone_check.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    record = json.loads((ROOT / "benchmarks" / "records" / "touchstone_check.json").read_text())
    assert tc.evaluate(record) == []
    assert record.get("gates_failed") == []
    assert record.get("gates_skipped") == []   # the committed record was made with scikit-rf
    assert record.get("git_head") and all(c.get("git_head") == record["git_head"] for c in record["cases"].values())


def test_nonreciprocal_s_against_pozar_table():
    """S of a non-reciprocal Z against the closed form (Pozar, Microwave Engineering, Table 4.2):
    Delta = (Z11 + z0)(Z22 + z0) - Z12 Z21, S11 = ((Z11 - z0)(Z22 + z0) - Z12 Z21)/Delta, S12 = 2 Z12 z0/Delta,
    S21 = 2 Z21 z0/Delta, S22 = ((Z11 + z0)(Z22 - z0) - Z12 Z21)/Delta.  Catches a transposition of S."""
    z0 = 50.0
    Z = np.array([[30 + 40j, 5 + 12j], [-7 + 3j, 80 - 20j]])
    d = (Z[0, 0] + z0) * (Z[1, 1] + z0) - Z[0, 1] * Z[1, 0]
    want = np.array([[((Z[0, 0] - z0) * (Z[1, 1] + z0) - Z[0, 1] * Z[1, 0]) / d, 2 * Z[0, 1] * z0 / d],
                     [2 * Z[1, 0] * z0 / d, ((Z[0, 0] + z0) * (Z[1, 1] - z0) - Z[0, 1] * Z[1, 0]) / d]])
    assert np.allclose(ts.z_to_s(Z, z0, convention="engineering"), want, rtol=1e-14, atol=1e-15)
    assert np.allclose(ts.s_to_z(want, z0), Z, rtol=1e-13)


# ---------------------------------------------------------- per-port references, conditioning, sanitising
def test_per_port_s_against_pozar_table():
    """S with unequal per-port R against the closed form (Pozar Table 4.2 with R1, R2):
    Delta = (Z11 + R1)(Z22 + R2) - Z12 Z21, S11 = ((Z11 - R1)(Z22 + R2) - Z12 Z21)/Delta,
    S12 = 2 Z12 sqrt(R1 R2)/Delta, S21 = 2 Z21 sqrt(R1 R2)/Delta, S22 = ((Z11 + R1)(Z22 - R2) - Z12 Z21)/Delta."""
    R1, R2 = 25.0, 75.0
    Z = np.array([[30 + 40j, 5 + 12j], [-7 + 3j, 80 - 20j]])
    d = (Z[0, 0] + R1) * (Z[1, 1] + R2) - Z[0, 1] * Z[1, 0]
    g = 2 * math.sqrt(R1 * R2)
    want = np.array([[((Z[0, 0] - R1) * (Z[1, 1] + R2) - Z[0, 1] * Z[1, 0]) / d, g * Z[0, 1] / d],
                     [g * Z[1, 0] / d, ((Z[0, 0] + R1) * (Z[1, 1] - R2) - Z[0, 1] * Z[1, 0]) / d]])
    assert np.allclose(ts.z_to_s(Z, [R1, R2], convention="engineering"), want, rtol=1e-14, atol=1e-15)
    assert np.allclose(ts.s_to_z(want, [R1, R2]), Z, rtol=1e-13)


def test_reciprocal_z_gives_symmetric_s_for_unequal_z0(tmp_path):
    """Power waves keep S symmetric for a reciprocal Z with unequal z0 (voltage waves would not: S_ij scales by
    sqrt(r_i/r_j)); also through a v1.1 file."""
    f = np.array([1e8, 3e8])
    Z = _line_z(f, 0.3, 50.0, 0.1)
    S = ts.z_to_s(Z, [25.0, 75.0], convention="engineering")
    assert np.max(ts.reciprocity_error(S)) < 1e-14
    assert np.min(np.abs(S[:, 0, 1])) > 0.1
    ts.write_touchstone_data(tmp_path / "pp.s2p", f, Z, convention="engineering", z0=[25.0, 75.0])
    back = ts.read_touchstone(tmp_path / "pp.s2p")["S"]
    assert np.max(ts.reciprocity_error(back)) < 1e-14


def test_two_port_y_amplification(tmp_path):
    """The multiport branch of the Y check.  Nearly coupled ports (equilibrated condition number ~2e5) with a
    small relative error of Z: Y is flagged at 1e-6 (x 2e5 > 0.1), not at 1e-9; a proportional (model) error is not
    amplified."""
    f = np.array([1e6, 2e6])
    Z = np.array([[[10j, 10j * (1 - 1e-5)], [10j * (1 - 1e-5), 10j]]] * 2)
    cond = ts.equilibrated_condition(Z)
    assert np.all((cond > 1e5) & (cond < 1e12))
    for rel, flagged in ((1e-6, [1e6]), (1e-9, [])):
        rels = np.array([rel, 1e-9])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = ts.write_touchstone_data(tmp_path / "y.s2p", f, Z, convention="engineering", kind="Y",
                                           Z_relative_error=rels)
        assert rep["Y_ill_determined_Hz"] == flagged
        assert bool(w) == bool(flagged)
    rep = ts.write_touchstone_data(tmp_path / "y.s2p", f, Z, convention="engineering", kind="Y",
                                   Z_relative_error=[1e-6, 1e-6], Z_relative_error_kind="proportional")
    assert rep["Y_ill_determined_Hz"] == []


def test_ascii_output_and_line_separators(tmp_path):
    """Non-ASCII characters and Unicode line separators in names become '?'; the file is ASCII and reads back;
    a third-party file with a Latin-1 byte in a comment reads."""
    rows = [{"f_Hz": 1e6, "Z_re_ohm": 1.0, "Z_im_ohm": -3.0}]
    res = _results_from_rows(rows, ports=["coil 10 \u00b5H \u2028 L\u2081"], name="a\u0085b\u2029c")
    ts.write_touchstone(res, tmp_path / "u.s1p")
    raw = (tmp_path / "u.s1p").read_bytes()
    assert all(b < 0x80 for b in raw)
    back = ts.read_touchstone(tmp_path / "u.s1p")
    assert back["ports"] == ["coil 10 ?H ? L?"] and back["f_Hz"].tolist() == [1e6]
    lat = tmp_path / "lat.s1p"
    lat.write_bytes(b"! 10 \xb5H coil\r\n# Hz S RI R 50\r\n1 0.5 0\r\n")
    assert ts.read_touchstone(lat)["S"][0, 0, 0] == 0.5


def test_reader_numbers_and_missing_option_line(tmp_path):
    """Only decimal/scientific finite numbers; a missing option line is warned about."""
    for bad, word in (("nan 0.5 0", "nan"), ("1_0 0.5 0", "1_0"), ("1 inf 0", "inf"), ("1 1e999 0", "infinity")):
        p = tmp_path / "b.s1p"
        p.write_text("# Hz S RI R 50\n" + bad + "\n")
        with pytest.raises(ValueError, match=word):
            ts.read_touchstone(p)
    p = tmp_path / "noopt.s1p"
    p.write_text("1 0.5 0\n")
    with pytest.warns(RuntimeWarning, match="no option line"):
        r = ts.read_touchstone(p)
    assert r["f_Hz"].tolist() == [1e9]
    p.write_text("# Hz S RI R 50\n1.5e-3 -.5 +2.\n")
    assert ts.read_touchstone(p)["S"][0, 0, 0] == complex(-0.5, 2.0)


def test_equal_z0_stems_and_malformed_port_metadata(tmp_path):
    """Equal per-port z0 collapse to v1.0 (Z accepted); no leading '-' in a stem; a malformed 'toward'
    and a '|' in a port name are not fatal and do not truncate the name."""
    f = [1e6]
    Z = np.array([[[10, 1], [1, 20]]], dtype=complex)
    rep = ts.write_touchstone_data(tmp_path / "e.s2p", f, Z, convention="engineering", kind="Z", z0=[50.0, 50.0])
    assert rep["z0_ohm"] == 50.0 and "# Hz Z RI R 50" in (tmp_path / "e.s2p").read_text()
    assert ts.safe_stem("-rf") == "rf" and ts.safe_stem("--") == "fullwave"
    rep = ts.write_touchstone_data(tmp_path / "p.s2p", f, Z, convention="engineering", ports=["a|b", "c"])
    assert ts.read_touchstone(tmp_path / "p.s2p")["ports"] == ["a/b", "c"]
    rows = [{"f_Hz": 1e6, "Z_re_ohm": 1.0, "Z_im_ohm": -3.0}]
    res = {"fullwave": {"sweep": rows, "ports": [{"name": "p", "toward": "+x", "direction": [1, None, 0]}]}}
    ts.write_touchstone(res, tmp_path / "t.s1p")
    assert "toward +x" in (tmp_path / "t.s1p").read_text()


def test_reader_vectorised_with_fallback(tmp_path):
    """The stack converts at once; one singular frequency among many is NaN alone."""
    n = 2000
    f = np.arange(1, n + 1, dtype=float)
    S = np.tile(np.array([[0.1, 0.8], [0.8, 0.1]]), (n, 1, 1)).astype(complex)
    S[700] = [[0, 1], [1, 0]]
    lines = ["# Hz S RI R 50"] + [f"{fi:g} " + " ".join(f"{v.real:g} 0" for v in (M[0, 0], M[1, 0], M[0, 1], M[1, 1]))
                                  for fi, M in zip(f, S)]
    (tmp_path / "big.s2p").write_text("\n".join(lines) + "\n")
    with pytest.warns(RuntimeWarning, match="Z does not exist at 1 frequencies"):
        r = ts.read_touchstone(tmp_path / "big.s2p")
    assert np.isnan(r["Z"][700]).all() and np.isfinite(r["Z"][np.arange(n) != 700]).all()
