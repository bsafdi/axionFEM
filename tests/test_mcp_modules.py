"""MCP tools for the cavity haloscope, SRF heterodyne, dc SQUID and COMSOL modules (axionfem._mcp_modules).

Each tool must return plain JSON (no numpy, complex or non-finite numbers) and agree with the Python API
it wraps; the workspace tools are run against a temporary workspace, and an in-process call through the
server checks that they are registered and that errors come back as {"ok": false, ...}.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

import axionfem as af
from axionfem import _mcp_modules as M

ROOT = Path(__file__).resolve().parents[1]


def strict_json(x) -> dict:
    """Round-trip through the strict JSON encoder (NaN/inf are rejected)."""
    return json.loads(json.dumps(x, allow_nan=False))


def test_jsonable():
    import numpy as np

    out = strict_json(M.jsonable({"a": np.arange(3), "b": np.float64(2.5), "c": 1 + 2j, "d": (1, 2),
                                  "e": float("nan"), "f": np.int64(4), "g": Path("x/y"), 3: np.inf}))
    assert out == {"a": [0, 1, 2], "b": 2.5, "c": {"re": 1.0, "im": 2.0}, "d": [1, 2], "e": None, "f": 4,
                   "g": "x/y", "3": None}


# ---------------------------------------------------------------- cavity haloscopes
def test_cavity_presets_and_limit():
    from axionfem import haloscope as hs

    p = strict_json(M.tool_cavity_presets())
    names = [r["name"] for r in p["presets"]]
    assert names == sorted(hs.PRESETS) and "HAYSTAC_2017" in names
    lim = strict_json(M.tool_cavity_preset_limit("HAYSTAC_2017"))
    ref = hs.preset_limit("HAYSTAC_2017")
    assert lim["g_GeV"] == pytest.approx(ref["g_GeV"], rel=1e-12)
    assert lim["implied_scan"]["rate_Hz_s"] == pytest.approx(hs.implied_rate("HAYSTAC_2017")["rate_Hz_s"], rel=1e-12)
    with pytest.raises(ValueError, match="available"):
        M.tool_cavity_preset_limit("NOPE")
    # a run without a published scan rate still gets the rate its published limit implies
    admx = strict_json(M.tool_cavity_preset_limit("ADMX_1D"))
    assert "limit_error" in admx and admx["implied_scan"]["rate_Hz_s"] > 0
    assert p["presets"][names.index("HAYSTAC_2017")]["sources"]           # provenance kept


def test_cavity_scan_matches_api():
    from axionfem import haloscope as hs

    rx = {"T_cav_K": 0.47, "T_in_K": 0.127, "N_A": 1.35, "eta_snr": 0.76}
    out = strict_json(M.tool_cavity_scan(5.75e9, 9.0, 1.545e-3, 0.5, 3e4, rx, beta=2.0, model="KSVZ",
                                         rate_Hz_s=17.87, f_stop_Hz=5.8e9))
    cav = hs.Cavity(B_T=9.0, V_m3=1.545e-3, C=0.5, Q0=3e4, beta=2.0)
    r = hs.Receiver(**rx)
    g = float(hs.g_axion(hs.freq_to_mass_eV(5.75e9), "KSVZ"))
    assert out["scan_rate_Hz_s"] == pytest.approx(hs.scan_rate(5.75e9, cav, r, g_GeV=g), rel=1e-12)
    assert out["g_limit_at_rate_GeV"] == pytest.approx(hs.g_limit_scan(5.75e9, cav, r, rate_Hz_s=17.87), rel=1e-12)
    assert out["Q_L"] == pytest.approx(1e4)
    # the scan rate at g scales as g^4: the coupling reached at the given rate follows from it
    assert out["g_limit_over_g"] == pytest.approx((17.87 / out["scan_rate_Hz_s"]) ** 0.25, rel=1e-6)
    assert out["cover"]["live_time_s"] > 0
    half = strict_json(M.tool_cavity_scan(5.75e9, 9.0, 1.545e-3, 0.5, 3e4, rx, beta=2.0, model="KSVZ",
                                          f_stop_Hz=5.8e9, duty_cycle=0.5))
    assert half["cover"]["live_time_s"] == pytest.approx(out["cover"]["live_time_s"], rel=1e-12)
    assert half["cover"]["elapsed_days"] == pytest.approx(2 * out["cover"]["live_time_days"], rel=1e-12)
    for bad in (dict(beta=-0.5), dict(eta=1.5), dict(duty_cycle=0.0)):
        with pytest.raises(ValueError):
            M.tool_cavity_scan(5.75e9, 9.0, 1.545e-3, 0.5, 3e4, rx, **bad)
    free = strict_json(M.tool_cavity_scan(5.75e9, 9.0, 1.545e-3, 0.5, 3e4, rx, model="KSVZ"))
    assert free["beta"] == free["beta_optimal"]
    assert free["scan_rate_Hz_s"] >= out["scan_rate_Hz_s"] * (1 - 1e-9)      # the optimal beta never does worse
    with pytest.raises(ValueError, match="exactly one"):
        M.tool_cavity_scan(5.75e9, 9.0, 1.545e-3, 0.5, 3e4, {"T_cav_K": 0.1})   # no noise level given
    with pytest.raises(ValueError, match="exactly one"):
        M.tool_cavity_scan(5.75e9, 9.0, 1.545e-3, 0.5, 3e4, {"T_sys_K": 0.5, "N_A": 1.0})
    with pytest.raises(ValueError, match="positive"):
        M.tool_cavity_scan(5.75e9, 9.0, 1.545e-3, 0.5, 3e4, rx, g_GeV=0.0)
    # the QCD convention (C_gamma = 1.77) and another halo density pass through
    q = strict_json(M.tool_cavity_scan(5.75e9, 9.0, 1.545e-3, 0.5, 3e4, rx, beta=2.0, convention="qcd",
                                       rho_GeV_cm3=0.3))
    g_q = float(hs.g_axion(hs.freq_to_mass_eV(5.75e9), "DFSZ", convention="qcd"))
    assert q["g_GeV"] == pytest.approx(g_q, rel=1e-12)
    assert q["scan_rate_Hz_s"] == pytest.approx(hs.scan_rate(5.75e9, cav, r, g_GeV=g_q, rho_GeV_cm3=0.3), rel=1e-12)


def test_create_cavity_and_solve(tmp_path):
    from axionfem._mcp_worker import compact

    ws = M.Workspace(tmp_path)
    out = M.tool_create_cavity(ws, "pb", a=0.1, L=0.2, wall="copper", wall_T_K=4.0, modes=3)
    assert out["ok"] and (tmp_path / "designs" / "pb.json").exists()
    res = ws.load("pb").solve(workdir=ws.runs / "pb", verbose=0)
    c = strict_json(M.jsonable(compact(res)))
    assert "coupling" not in c and "benchmarks" not in c          # no lumped pickup in a cavity design
    tm010 = c["cavity_modes"][0]
    # TM010 of a 10 cm x 20 cm pillbox: f = 2.405 c / (2 pi a), C = 4/2.405^2 = 0.692, G = 453 ohm / (1 + a/L)
    assert tm010["family"] == "TM" and tm010["f_Hz"] == pytest.approx(2.404826 * 299792458 / (2 * math.pi * 0.1), rel=2e-3)
    assert tm010["C"] == pytest.approx(0.6916, rel=5e-3)
    assert tm010["G_ohm"] == pytest.approx(453.0 / 1.5, rel=5e-3)
    assert tm010["Q0"] > 1e5 and tm010["walls"]["wall"]["Rs_model"] == "anomalous"
    with pytest.raises(ValueError, match="b"):
        M.tool_create_cavity(ws, "cx", shape="coax", a=0.02)
    with pytest.raises(ValueError, match="wall"):
        M.tool_create_cavity(ws, "w", wall="gold")


# ---------------------------------------------------------------- SRF heterodyne
def test_srf_design_reproduces_model():
    from axionfem import heterodyne as het

    out = strict_json(M.tool_srf_design("optimistic"))
    ref = het.design_A(het.SCENARIOS_A["optimistic"])
    assert out["design"]["V_m3"] == pytest.approx(ref["V_m3"], rel=1e-12)
    assert out["design"]["V_m3"] == pytest.approx(259.926, rel=1e-4)
    assert out["design"]["cost_MCHF"]["total"] == pytest.approx(64.693, rel=2e-3)
    assert out["design"]["scenario"] == "optimistic"
    hot = strict_json(M.tool_srf_design("optimistic", T_K=4.5))
    assert hot["design"]["V_m3"] > out["design"]["V_m3"]           # more wall noise needs more volume
    assert out["inputs"]["geometry"] == "projectx" and out["geometry_figures"]["xi"] == pytest.approx(7.643, abs=1e-3)
    with pytest.raises(ValueError, match="geometry"):
        M.tool_srf_design("optimistic", geometry="unknown")
    with pytest.raises(ValueError, match="scenario"):
        M.tool_srf_design("best")
    with pytest.raises(ValueError, match="refrigerator_factor"):
        M.tool_srf_design("optimistic", T_K=3.0)
    assert M.tool_srf_design("optimistic", T_K=3.0, refrigerator_factor=2.0)["design"]["V_m3"] > 259.9
    with pytest.raises(ValueError, match="f_min_Hz < f_max_Hz"):
        M.tool_srf_design("optimistic", f_min_Hz=2e6)
    eq = M.tool_srf_design("optimistic", geometry="equator_diagnostic")
    assert any("equator" in w for w in eq["warnings"])
    assert eq["geometry_figures"]["Bpk_over_B0"] < out["geometry_figures"]["Bpk_over_B0"]


def test_srf_mode_pair():
    out = strict_json(M.tool_srf_mode_pair("TM020/TE011"))
    assert out["eta_10"] == pytest.approx(0.43697, rel=1e-4)
    assert out["pump"]["mode"] == "TM020" and out["signal"]["mode"] == "TE011"


# ---------------------------------------------------------------- dc SQUID
def test_squid_chain():
    from axionfem import squid

    out = strict_json(M.tool_squid_chain("C1+low+RT", [1e5, 1e6], Q=1e6, T_K=0.03))
    ref = squid.chain("C1+low+RT").summary([1e5, 1e6], Q=1e6, T_K=0.03)
    assert out["eta"] == pytest.approx(ref["eta"], rel=1e-12)
    assert out["first_stage"]["eta_TC"] == pytest.approx(3.70, abs=0.01)
    assert len(out["xi_opt"]) == 2 and out["warnings"] == []
    cold = strict_json(M.tool_squid_chain("C1+low+RT", 1e6, T2_K=0.1))     # scalar f; warnings from the chain itself
    assert cold["warnings"], cold["warnings"]
    with pytest.raises(ValueError, match="preset"):
        M.tool_squid_chain("C2")


def test_squid_reach(tmp_path):
    from axionfem import scan, squid

    ws = M.Workspace(tmp_path)
    with pytest.raises(FileNotFoundError, match="solve_design"):
        M.tool_squid_reach(ws, "tor")
    d = af.template("toroid", R=0.5, H=0.5)
    ws.save("tor", d)
    res = ws.load("tor").solve(workdir=ws.runs / "tor", verbose=0, self_force=False, force_samples=0)
    out = strict_json(M.tool_squid_reach(ws, "tor", "C1+low+RT", 1e6, 0.03, "as_designed", 4))
    ref = scan.reach(res.Psi, res.L_p, squid.benchmark("C1+low+RT", Q=1e6, T_K=0.03), n=4)
    assert out["reach"]["g95_GeV"] == pytest.approx(list(ref["g95_GeV"]), rel=1e-9)
    assert out["reach"]["eta_A"][0] == pytest.approx(9.7, rel=0.01)
    # cryogenic preamplifiers are specified from 5 MHz: the band can be moved there
    cryo = strict_json(M.tool_squid_reach(ws, "tor", "C1+low+cryo", 1e6, 0.03, "as_designed", 3, 5e6, 2e7))
    assert cryo["reach"]["f_Hz"][0] == pytest.approx(5e6)
    # Psi is signed; the reach uses |Psi|
    d.set(NI=-1e6)
    ws.save("neg", d)
    ws.load("neg").solve(workdir=ws.runs / "neg", verbose=0, self_force=False, force_samples=0)
    neg = strict_json(M.tool_squid_reach(ws, "neg", "C1+low+RT", 1e6, 0.03, "as_designed", 4))
    assert neg["reach"]["g95_GeV"] == pytest.approx(out["reach"]["g95_GeV"], rel=1e-6)
    # results older than the design are refused
    d.set(NI=2e6)
    ws.save("neg", d)
    with pytest.raises(ValueError, match="older than the design"):
        M.tool_squid_reach(ws, "neg")
    # names cannot leave the workspace
    with pytest.raises(ValueError, match="invalid design name"):
        M.tool_squid_reach(ws, "../tor")


# ---------------------------------------------------------------- COMSOL
def _synthetic_mph(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_comsol_import import synthetic_mph
    finally:
        sys.path.pop(0)
    return synthetic_mph(tmp_path / "synthetic.mph")


def test_inspect_and_import_comsol(tmp_path):
    mph = _synthetic_mph(tmp_path)
    ins = strict_json(M.tool_inspect_comsol(str(mph)))
    assert ins["parameters"]["f0"]["value_SI"] == pytest.approx(2e6)
    feats = {f["tag"]: f for p in ins["physics"] for f in p["features"]}
    assert feats["lport1"]["type"] == "LumpedPort" and feats["ecd1"].get("disabled") is True
    ws = M.Workspace(tmp_path / "ws")
    out = strict_json(M.tool_import_comsol(ws, str(mph), "syn"))
    assert out["ok"] and out["symmetry"] == "3d" and not out["report"]["blocking"]
    doc = ws.load("syn").doc
    assert doc["fullwave"]["frequencies"] == pytest.approx([2e6, 4e6])
    assert Path(doc["mesh"]["file"]).is_absolute() and Path(doc["mesh"]["file"]).exists()
    assert Path(out["report_markdown"]).exists()
    # only the design lands in designs/: the mesh and the report stay under imports/<name>/
    assert sorted(p.name for p in (tmp_path / "ws" / "designs").iterdir()) == ["syn.json"]
    with pytest.raises(FileNotFoundError):
        M.tool_import_comsol(ws, str(tmp_path / "none.mph"), "x")
    # a second import gets its own directory: the first design's mesh is never overwritten
    mesh1 = Path(doc["mesh"]["file"])
    before = mesh1.read_bytes()
    M.tool_import_comsol(ws, str(mph), "syn")
    mesh2 = Path(ws.load("syn").doc["mesh"]["file"])
    assert mesh2 != mesh1 and mesh1.read_bytes() == before


def test_export_comsol(tmp_path):
    ws = M.Workspace(tmp_path)
    ws.save("tor", af.template("toroid"))
    out = strict_json(M.tool_export_comsol(ws, "tor", "java"))
    assert Path(out["file"]).exists() and out["summary"]["dimension"] == 2
    for name, cls in (("my-toroid", "my_toroid"), ("2211", "M2211"), ("run.1", "run_1")):
        ws.save(name, af.template("toroid"))
        o = M.tool_export_comsol(ws, name, "java")
        assert Path(o["file"]).name == f"{cls}.java" and o["summary"]["class"] == cls
    from axionfem._paths import mesher_path

    if mesher_path():
        out = strict_json(M.tool_export_comsol(ws, "tor", "mphtxt"))
        assert Path(out["file"]).read_text().startswith("#") and "sheath" in out["summary"]["selections"]
    with pytest.raises(ValueError, match="format"):
        M.tool_export_comsol(ws, "tor", "mph")


# ---------------------------------------------------------------- server registration
def test_server_registers_module_tools_and_reports_errors(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    import asyncio

    from axionfem import mcp_server as ms

    names = {t.name for t in asyncio.run(ms.server.list_tools())}
    assert {"cavity_presets", "cavity_preset_limit", "cavity_scan", "create_cavity", "srf_design", "srf_mode_pair",
            "squid_chain", "squid_reach", "export_comsol", "inspect_comsol", "import_comsol",
            "export_touchstone"} <= names
    for word in ("cavity_scan", "srf_design", "squid_chain", "import_comsol"):
        assert word in ms.GUIDE
    # a generic MCP client shows its agent only the instructions and the tool descriptions: the beta modules'
    # tools must say that they are beta
    tools = {t.name: t for t in asyncio.run(ms.server.list_tools())}
    for name in ("cavity_presets", "cavity_preset_limit", "cavity_scan", "create_cavity", "srf_design",
                 "srf_mode_pair", "squid_chain", "squid_reach", "inspect_comsol", "import_comsol"):
        assert tools[name].description.startswith("Beta."), name
    assert "beta modules" in ms.GUIDE and "not a construction estimate" in ms.GUIDE
    monkeypatch.setattr(ms, "WORKSPACE", tmp_path)
    monkeypatch.setattr(ms, "DESIGNS", tmp_path / "designs")
    monkeypatch.setattr(ms, "RUNS", tmp_path / "runs")
    out = asyncio.run(ms.server.call_tool("squid_reach", {"name": "nothing"}))
    payload = json.loads(out.content[0].text)
    assert payload["ok"] is False and "solve_design" in payload["error"]
