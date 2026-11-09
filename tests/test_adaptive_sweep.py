"""Adaptive 3D full-wave sweep (fullwave.adaptive; docs/physics/adaptive_sweep.md).

The discrete Z(omega) of a closed structure is rational, with poles at the discrete eigenfrequencies of the same mesh
(port open). The default suite runs the shorted TEM line of benchmarks/fw3d_robin_validation.py (a quarter-wave pole
at c/4l, seconds per run): the sweep converges, its one in-band pole sits on the same mesh's eigenfrequency and near
c/4l, the model reproduces its own solves and the exact Z between them, and the rows equal a fixed list's at the same
frequencies bit for bit; plus the spec's refusals. The slow test (AXF_SLOW) runs the two-port frame in its closed
sphere with a lossy slab: the poles against the eigenmodes, their Q against 1/(tan delta p_e). The fit itself is
tested on synthetic responses by the C++ test cpp/tests/test_rational_fit.cpp; the benchmark is
benchmarks/adaptive_sweep.py."""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import axionfem as af
from axionfem import circuit as ck

ROOT = Path(__file__).resolve().parents[1]
TWO_PORT = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
sys.path.insert(0, str(ROOT / "benchmarks"))

import fullwave_indicator as FWI  # noqa: E402  (the TEM line's designs and exact Z)

pytestmark = pytest.mark.skipif(not af.build_info()["engine_3d"],
                                reason="needs the 3D engine (an MPI build; the wheels have none)")

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D full wave (slow); set AXF_SLOW=1")
C0 = 299792458.0
# "memory": the per-frequency memory records, as volatile as the timings
TIMING_WORDS = ("seconds", "rss", "timing", "date", "path", "directory", "file", "memory")


def _tem(wd: Path, fullwave: dict, order: int = 2, nx: int = 6, termination: str = "pec") -> tuple[dict, dict]:
    doc, geometry = FWI.tem_design(wd, termination, order, 0, [1e8], nx=nx)
    doc["fullwave"].pop("frequencies")
    doc["fullwave"].update(fullwave)
    return doc, geometry


def _leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if any(word in k.lower() for word in TIMING_WORDS) or k.endswith("_s"):
                continue
            yield from _leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{path}/{i}")
    else:
        yield path, x


@pytest.fixture(scope="module")
def tem_adaptive(tmp_path_factory):
    wd = tmp_path_factory.mktemp("tem_adaptive")
    doc, geometry = _tem(wd, {"adaptive": {"f_min_Hz": 1e8, "f_max_Hz": 4e8, "tolerance": 1e-5,
                                           "spacing": "linear"}, "modes": 2})
    return af.Design(doc).solve(wd, verbose=0).data, geometry, wd


def test_tem_line_pole_on_the_eigenmode(tem_adaptive):
    """One pole in 100-400 MHz: on the same mesh's eigenfrequency (to the fit), near c/4l = 249.8 MHz (the
    discretization), lossless (no Q)."""
    data, geometry, _ = tem_adaptive
    fw = data["fullwave"]
    ad = fw["adaptive"]
    assert ad["converged"], ad["stop_reason"]
    assert ad["estimated_error"] < 1e-5 and ad["last_prediction_error"] < 1e-5
    assert ad["solves"] <= 20
    assert not ad["refused"]
    res = fw["self_resonances"]
    assert len(res) == 1
    f_pole = res[0]["f_Hz"]
    eig = [m["f_Hz"] for m in fw["eigenmodes"]["modes"] if m["converged"]]
    f_eig = min(eig, key=lambda f: abs(f - f_pole))
    assert abs(f_pole / f_eig - 1) < 1e-7
    assert abs(f_pole / (C0 / (4 * geometry["l"])) - 1) < 1e-2
    assert res[0]["Q"] is None or res[0]["Q"] > 1e7
    assert res[0]["between_Hz"][0] < f_pole < res[0]["between_Hz"][1]
    assert res[0]["class"] == "resonance"
    # lossless: Im p within its leave-one-out uncertainty, so no Q but a lower bound
    assert res[0]["Q"] is None and res[0]["lossless_within_fit"] and res[0]["Q_min"] > 1e7
    assert 2 * res[0]["reproduced_by_refits"][0] >= res[0]["reproduced_by_refits"][1] > 0
    assert fw["self_resonances_determined"] is True
    assert ad["hermite"]["used_in_final_fit"] and ad["fit_consistent"]
    assert [p for p in ad["poles"] if p["class"] == "resonance"] == res
    # noise-bound exactly when the fit tolerance is below kNoiseBoundFactor x pole_noise, with a warning then
    st = ad["settings"]
    assert ad["noise_bound"] is (st["fit_tolerance"] < st["noise_bound_factor"] * st["pole_noise"])
    assert any("adaptive sweep: noise-bound." in w for w in data["warnings"]) is ad["noise_bound"]
    # within a further factor 10 the run warns that the sweep may be noise-bound next to its poles (near-pole errors
    # up to 24 x pole_noise have been seen there)
    near = not ad["noise_bound"] and st["fit_tolerance"] < 10 * st["noise_bound_factor"] * st["pole_noise"]
    assert any("may be noise-bound there" in w for w in data["warnings"]) is near
    assert ad["invisible_poles_removed"] >= 0 and ad["sample_fit_error_before_removal"] >= 0


def test_tem_line_model_matches_solves_and_exact_z(tem_adaptive):
    """The model returns its solves at the support points, the rows are sorted, the dense curve is the model, and
    between the solves the model follows the exact Z to the discretization error (as the solves do)."""
    data, geometry, _ = tem_adaptive
    fw = data["fullwave"]
    rows = fw["sweep"]
    assert [r["f_Hz"] for r in rows] == sorted(r["f_Hz"] for r in rows)
    model = ck.fullwave_model(data)
    for r in rows:
        z = model(r["f_Hz"])["Z_matrix"][0, 0]
        assert abs(z - complex(r["Z_re_ohm"], r["Z_im_ohm"])) <= 1e-5 * abs(z)
    dense = ck.fullwave_dense(data)
    assert len(dense["f_Hz"]) >= fw["adaptive"]["settings"]["dense_points"]
    assert np.allclose(model(dense["f_Hz"])["Z_matrix"][:, 0, 0], dense["Z"], rtol=1e-8)
    assert np.allclose(dense["L_H"], -dense["Z"].imag / (2 * math.pi * dense["f_Hz"]), rtol=1e-12)
    # the solves' own error against the exact Z bounds what the model can do between them
    solve_err = max(abs(complex(r["Z_re_ohm"], r["Z_im_ohm"]) / FWI.tem_exact_z("pec", r["f_Hz"], geometry) - 1)
                    for r in rows)
    pole = C0 / (4 * geometry["l"])
    mids = [0.5 * (a["f_Hz"] + b["f_Hz"]) for a, b in zip(rows, rows[1:])]
    mids = [f for f in mids if abs(f / pole - 1) > 0.05]
    model_err = max(abs(model(f)["Z_matrix"][0, 0] / FWI.tem_exact_z("pec", f, geometry) - 1) for f in mids)
    assert model_err < 3 * solve_err + 1e-4


def test_adaptive_rows_equal_a_fixed_list(tem_adaptive, tmp_path):
    """The per-frequency body is shared: a fixed list at the adaptive sweep's frequencies gives the same rows, bit for
    bit (timings and file names aside)."""
    data, _, _ = tem_adaptive
    freqs = sorted(data["fullwave"]["adaptive"]["solved_frequencies_Hz"])
    doc, _ = _tem(tmp_path, {"frequencies": freqs, "modes": 2})
    fixed = af.Design(doc).solve(tmp_path, verbose=0).data
    a = dict(_leaves(data["fullwave"]["sweep"]))
    b = dict(_leaves(fixed["fullwave"]["sweep"]))
    assert len(a) > 100
    assert a.keys() == b.keys()
    assert all(a[k] == b[k] for k in a), [k for k in a if a[k] != b[k]][:5]
    # and the fixed list keeps the sign-change rule
    assert all("between_Hz" in r and "f_Hz" not in r for r in fixed["fullwave"]["self_resonances"])


def test_adaptive_rows_carry_memory_records_and_the_thresholds_are_restored(tmp_path, monkeypatch):
    """The memory record (tests/test_fullwave3d_memory.py) is written by the per-frequency body that an adaptive sweep
    shares with a fixed list: every adaptive row carries one. The malloc thresholds that the analysis fixed at its
    first direct factorization are relaxed when the adaptive analysis ends, so that the next analysis of the process
    finds them relaxed and fixes them again for itself."""
    if not af.build_info().get("strumpack"):
        pytest.skip("needs the STRUMPACK direct path")
    for key in ("AXF_FW_MALLOC_TRIM", "AXF_FW_MALLOC_THRESHOLDS", "AXF_FW_FREQUENCY_PEAK", "MALLOC_TRIM_THRESHOLD_",
                "MALLOC_MMAP_THRESHOLD_", "GLIBC_TUNABLES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AXF_FW_SOLVER", "strumpack")
    doc, _ = _tem(tmp_path / "adaptive", {"adaptive": {"f_min_Hz": 1e8, "f_max_Hz": 4e8, "tolerance": 1e-4,
                                                       "spacing": "linear"}})
    fw = af.Design(doc).solve(tmp_path / "adaptive", verbose=0).data["fullwave"]
    if fw["memory"]["malloc_trim"] == "unavailable (not glibc)":
        pytest.skip("not glibc's malloc")
    assert fw["adaptive"]["converged"], fw["adaptive"]["stop_reason"]
    assert len(fw["sweep"]) >= 3
    for row in fw["sweep"]:
        memory = row["memory"]
        for where in ("frequency_start", "frequency_end"):
            assert memory[where]["rss_rank0_bytes"] > 0
            assert memory[where]["peak_rss_rank0_bytes"] >= memory[where]["rss_rank0_bytes"]
        assert memory["after_factorization_rank0"]["rss_bytes"] > 0
    assert fw["memory"]["after_sweep"]["rss_rank0_bytes"] > 0
    assert fw["memory"]["malloc_thresholds_mode"] == "fixed"
    assert fw["memory"]["malloc_thresholds_applied_ranks"] >= 1
    assert "after it" in fw["memory"]["malloc_thresholds"]
    # had the adaptive analysis left them fixed, the next one would find them fixed and not set them (applied 0)
    doc, _ = _tem(tmp_path / "next", {"frequencies": [1e8]})
    after = af.Design(doc).solve(tmp_path / "next", verbose=0).data["fullwave"]["memory"]
    assert after["malloc_thresholds_applied_ranks"] >= 1
    assert "left by an earlier full-wave analysis" in after["malloc_thresholds_before"]


def test_adaptive_spec_refusals():
    doc = json.loads(TWO_PORT.read_text())
    good = {"f_min_Hz": 1e6, "f_max_Hz": 1e7}
    cases = [({"frequencies": [1e6]}, "not both"),
             ({"f_min": 1e6, "f_max": 1e7}, "not both"),
             ({"tolerance": 1.0}, "tolerance must lie in"),
             ({"initial_points": 2}, "initial_points must be an integer between 3 and 50"),
             ({"max_solves": 3}, "max_solves must be an integer between initial_points and 500"),
             ({"spacing": "cubic"}, "spacing must be"),
             ({"f_max_Hz": 1e5}, "0 < f_min_Hz < f_max_Hz")]
    for extra, message in cases:
        d = json.loads(json.dumps(doc))
        fw = d["fullwave"]
        fw.pop("frequencies")
        adaptive = dict(good)
        for key in ("tolerance", "initial_points", "max_solves", "spacing", "f_max_Hz"):
            if key in extra:
                adaptive[key] = extra.pop(key)
        fw["adaptive"] = adaptive
        fw.update(extra)
        with pytest.raises(Exception, match=message):
            af.Design(d).validate()
    toroid = af.template("toroid", R=2, H=2)
    toroid.doc["fullwave"] = {"adaptive": good}
    with pytest.raises(Exception, match="3D full-wave sweeps only"):
        toroid.validate()


def _model_against_direct_solves(data: dict, doc: dict, wd: Path, freqs: list[float]) -> float:
    """The adaptive model against fixed-list solves at `freqs` on the same design, in the fit's measure."""
    doc["fullwave"].pop("adaptive", None)
    doc["fullwave"]["frequencies"] = freqs
    rows = af.Design(doc).solve(wd, verbose=0).data["fullwave"]["sweep"]
    model = ck.fullwave_model(data)
    zs = [abs(complex(r["Z_re_ohm"], r["Z_im_ohm"])) for r in data["fullwave"]["sweep"]]
    floor = 0.01 * float(np.median(zs))
    worst = 0.0
    for r in rows:
        z = complex(r["Z_re_ohm"], r["Z_im_ohm"])
        worst = max(worst, abs(model(r["f_Hz"])["Z_matrix"][0, 0] - z) / max(abs(z), floor))
    return worst


@pytest.mark.parametrize("termination, nx", [("scattering", 6), ("scattering", 12), ("pml", 6)])
def test_adaptive_with_robin_and_pml_terminations(termination, nx, tmp_path):
    """The matched TEM line (an absorbing Robin end) and the PML-terminated line: their dZ/dw is exact, so the sweep
    converges with derivative rows, matches direct solves it never saw, and reports no resonance (an inexact
    derivative produced five and 29 spurious ones). At nx 12 the matched line's Z is constant to 6e-7 over the band,
    so one support point fits it; the comparison model of a one-point fit must be defined, or that sweep runs to
    max_solves."""
    band = {"f_min_Hz": 1e8, "f_max_Hz": 3e8, "tolerance": 1e-3, "spacing": "linear", "max_solves": 25}
    doc, _ = _tem(tmp_path / "adaptive", {"adaptive": band}, termination=termination, nx=nx)
    data = af.Design(doc).solve(tmp_path / "adaptive", verbose=0).data
    ad = data["fullwave"]["adaptive"]
    assert ad["converged"], ad["stop_reason"]
    assert ad["hermite"]["used_in_final_fit"] and ad["fit_consistent"]
    assert data["fullwave"]["self_resonances"] == []
    assert not [p for p in ad["poles"] if p["class"] == "resonance"]
    check, _ = _tem(tmp_path / "check", {}, termination=termination, nx=nx)
    worst = _model_against_direct_solves(data, check, tmp_path / "check", [1.13e8, 1.57e8, 2.11e8, 2.68e8])
    assert worst < 1e-3


def test_adaptive_unconverged_reports_no_resonance(tmp_path):
    """Out of solves (tolerance 1e-6, max_solves 6): converged false, a warning, and the model's poles stay in
    fullwave.adaptive.poles instead of self_resonances."""
    band = {"f_min_Hz": 1e8, "f_max_Hz": 4e8, "tolerance": 1e-6, "spacing": "linear", "max_solves": 6}
    doc, _ = _tem(tmp_path, {"adaptive": band})
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    ad = data["fullwave"]["adaptive"]
    assert not ad["converged"] and ad["stop_reason"] == "max_solves" and ad["solves"] == 6
    assert data["fullwave"]["self_resonances"] == []
    assert data["fullwave"]["self_resonances_determined"] is False
    assert ad["poles"]
    assert any("adaptive sweep: stopped (max_solves)" in w for w in data["warnings"])


def test_adaptive_noise_bound_is_flagged(tmp_path, monkeypatch):
    """Solves loosened to a GMRES tolerance of 1e-9 (AXF_FW_RTOL; true residuals 1e-8 to 6e-8) at tolerance 1e-6
    (fit tolerance 1e-9): the sweep is noise-bound (fit tolerance below 10 x pole_noise), and says so in
    fullwave.adaptive.noise_bound and a warning."""
    monkeypatch.setenv("AXF_FW_RTOL", "1e-9")
    band = {"f_min_Hz": 1e8, "f_max_Hz": 4e8, "tolerance": 1e-6, "spacing": "linear", "max_solves": 6}
    doc, _ = _tem(tmp_path, {"adaptive": band})
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    ad = data["fullwave"]["adaptive"]
    assert ad["settings"]["pole_noise"] > 1e-10
    st = ad["settings"]
    assert ad["noise_bound"] is (st["fit_tolerance"] < st["noise_bound_factor"] * st["pole_noise"])
    assert ad["noise_bound"] is True
    assert any("adaptive sweep: noise-bound" in w and "AXF_FW_RTOL" in w for w in data["warnings"])


def test_adaptive_near_noise_bound_warns(tmp_path):
    """The TEM line at tolerance 1e-4 (fit tolerance 1e-7, about 26 x pole_noise): not noise-bound, but within
    10 x the flag's line, so the run warns that the sweep may be noise-bound next to its poles (the warning's
    positive branch)."""
    band = {"f_min_Hz": 1e8, "f_max_Hz": 4e8, "tolerance": 1e-4, "spacing": "linear"}
    doc, _ = _tem(tmp_path, {"adaptive": band})
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    ad = data["fullwave"]["adaptive"]
    st = ad["settings"]
    ratio = st["fit_tolerance"] / st["pole_noise"]
    assert st["noise_bound_factor"] <= ratio < 10 * st["noise_bound_factor"], ratio
    assert ad["noise_bound"] is False
    assert any("may be noise-bound there" in w for w in data["warnings"])
    assert not any("adaptive sweep: noise-bound." in w for w in data["warnings"])


def test_adaptive_with_every_solve_refused(tmp_path, monkeypatch):
    """GMRES starved (AXF_FW_MAX_ITER=2) with a validation shunt, whose gate refuses every frequency: the sweep stops
    after the initial points, unconverged, with no model, and writes its results instead of crashing. The refused
    rows' diagnostics are renamed, so no fullwave_rejected.json is left behind."""
    monkeypatch.setenv("AXF_FW_MAX_ITER", "2")
    band = {"f_min_Hz": 1e8, "f_max_Hz": 3e8, "tolerance": 1e-3, "max_solves": 9}
    doc, _ = _tem(tmp_path, {"adaptive": band, "validation_shunt_R_ohm": [50.0]})
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    ad = data["fullwave"]["adaptive"]
    assert not ad["converged"] and ad["stop_reason"] == "no usable solve among the initial points"
    assert ad["solves"] == 5 and len(ad["refused"]) == 5
    assert ad["model"] is None and ad["dense"] is None and data["fullwave"]["self_resonances"] == []
    assert data["fullwave"]["status"] == "provisional"
    assert any("no frequency gave a usable solve" in w for w in data["warnings"])
    assert (tmp_path / "results.json").exists()
    assert not (tmp_path / "fullwave_rejected.json").exists()
    assert all(Path(r["diagnostics_file"]).exists() for r in ad["refused"])
    # the gates' mesh check points at the renamed diagnostics and the results, which carry worst_elements
    refused = [w for w in data["warnings"] if "was refused" in w]
    assert len(refused) == 5
    for w, r in zip(refused, ad["refused"]):
        assert "Mesh check (run because this solve failed)" in w and "fullwave_rejected.json" not in w
        assert f"worst_elements in the results and in {r['diagnostics_file']}" in w
    assert "worst_elements" in data["mesh"]["info"]
    with pytest.raises(ValueError, match="no model"):
        ck.fullwave_dense(data)


def test_adaptive_two_ports_on_two_ranks(tmp_path):
    """The two-port frame (two gap ports, uniform source) in its PEC sphere over 115-150 MHz on two MPI ranks (one
    in-process run without AXF_MPI): converges, and its 2 x 2 Z and Psi match direct solves it never saw."""
    ranks = 2 if os.environ.get("AXF_MPI") else 1
    doc = json.loads(TWO_PORT.read_text())
    fw = doc["fullwave"]
    for key in ("frequencies", "source_check", "validation_shunt_R_ohm"):
        fw.pop(key, None)
    fw["adaptive"] = {"f_min_Hz": 1.15e8, "f_max_Hz": 1.5e8, "tolerance": 1e-3, "spacing": "linear"}
    data = af.Design(doc).solve(tmp_path / "adaptive", verbose=0, mpi_ranks=ranks).data
    ad = data["fullwave"]["adaptive"]
    assert ad["converged"], ad["stop_reason"]
    assert len(ad["model"]["outputs"]) == 6   # Z (2 x 2) and Psi (2)
    fw.pop("adaptive")
    freqs = [1.21e8, 1.33e8, 1.46e8]
    fw["frequencies"] = freqs + [1.5e8]   # the band's top keeps the mesh of the adaptive run
    rows = [r for r in af.Design(doc).solve(tmp_path / "check", verbose=0, mpi_ranks=ranks).data["fullwave"]["sweep"]
            if r["f_Hz"] in freqs]
    model = ck.fullwave_model(data)
    zs = [np.abs(np.array(r["Z_matrix_re_ohm"]) + 1j * np.array(r["Z_matrix_im_ohm"])) for r in data["fullwave"]["sweep"]]
    floor = 0.01 * float(np.median([z.max() for z in zs]))
    for r in rows:
        z = np.array(r["Z_matrix_re_ohm"]) + 1j * np.array(r["Z_matrix_im_ohm"])
        psi = np.array(r["Psi_vector"]["re"]) + 1j * np.array(r["Psi_vector"]["im"])
        m = model(r["f_Hz"])
        assert np.max(np.abs(m["Z_matrix"] - z) / np.maximum(np.abs(z), floor)) < 1e-3
        assert np.max(np.abs(m["Psi_vector"] - psi) / np.maximum(np.abs(psi), 0.01 * np.max(np.abs(psi)))) < 1e-3


# ------------------------------------------------------------------ slow: the two-port frame with a lossy slab
@slow
def test_two_port_frame_lossy_cavity_poles_and_q(tmp_path):
    """The frame in its closed sphere with a slab (eps_r 2, tan delta 1e-3): every in-band pole sits on an eigenmode of
    the same mesh (Re p against the lossless eigenfrequency, O(tan delta^2)), and its Q is 1 / (tan delta p_e), p_e the
    slab's share of the mode's electric energy, to O(tan delta)."""
    doc = json.loads(TWO_PORT.read_text())
    fw = doc["fullwave"]
    for key in ("frequencies", "source_check", "validation_shunt_R_ohm"):
        fw.pop(key, None)
    doc["materials"] = {"slab_dielectric": {"kind": "linear", "mu_r": 1.0, "eps_r": 2.0, "tan_delta": 1e-3}}
    doc["regions"].append({"name": "slab", "material": "slab_dielectric",
                           "solid": {"box": {"min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]}}})
    fw["adaptive"] = {"f_min_Hz": 1.15e8, "f_max_Hz": 1.55e8, "tolerance": 1e-4, "spacing": "linear"}
    fw["modes"] = 8
    data = af.Design(doc).solve(tmp_path, verbose=0).data
    out = data["fullwave"]
    assert out["adaptive"]["converged"], out["adaptive"]["stop_reason"]
    modes = [m for m in out["eigenmodes"]["modes"] if m["converged"]]
    in_band = [m for m in modes if 1.15e8 < m["f_Hz"] < 1.55e8]
    strong = [r for r in out["self_resonances"] if r["relative_strength"] > 1e-6]
    assert strong and len(strong) <= len(in_band)
    compared = 0
    for r in strong:
        m = min(modes, key=lambda m: abs(m["f_Hz"] - r["f_Hz"]))
        assert abs(r["f_Hz"] / m["f_Hz"] - 1) < 1e-5
        assert r["Q"] is not None and r["Q"] > 0
        if "slab" not in m["energy_fraction"]:   # the modes list their three largest regions only
            continue
        q_ref = 1.0 / (1e-3 * m["energy_fraction"]["slab"])
        assert abs(r["Q"] / q_ref - 1) < 1e-3
        compared += 1
    assert compared >= 1
