"""What MCP chat clients need from the server.

List tools return an object (an empty list result reaches some clients as no content at all); tool output and the
guide carry no internal wording; a full-wave sweep says plainly up to what frequency the MQS numbers hold; viewer
pages are written readable; and the mcp dependency pin excludes incompatible releases.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
from pathlib import Path

import pytest

import axionfem as af

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = re.compile(r"\bpaper\b|\bdraft\b", re.I)


def test_guide_skill_and_templates_have_no_internal_wording():
    pytest.importorskip("mcp")
    from axionfem import mcp_server

    texts = {"GUIDE": mcp_server.GUIDE,
             "SKILL.md": (ROOT / ".claude/skills/axionfem-design/SKILL.md").read_text()}
    for t in af.templates():
        texts[f"template {t['name']}"] = t["description"] + json.dumps(t["parameters"])
    for name, text in texts.items():
        assert not INTERNAL.search(text), (name, INTERNAL.search(text).group(0))


@pytest.fixture(scope="module")
def small_toroid(tmp_path_factory):
    wd = tmp_path_factory.mktemp("toroid")
    return af.template("toroid", R=0.5, H=0.5).solve(workdir=wd, self_force=False, force_samples=0)


def test_solve_note_has_no_internal_wording(small_toroid):
    assert not INTERNAL.search(small_toroid.data["coupling"]["note"])


def test_design_names_cannot_leave_the_workspace(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from axionfem import mcp_server as ms

    monkeypatch.setattr(ms, "DESIGNS", tmp_path / "designs")
    for bad in ("..", ".hidden", "../x", "a/b", ""):
        out = ms.create_design("toroid", bad)
        assert out["ok"] is False and "invalid design name" in out["error"], bad


def test_list_tools_return_objects(tmp_path, monkeypatch):
    """An empty list result reaches some clients as no content at all: return an object."""
    pytest.importorskip("mcp")
    from axionfem import mcp_server as ms

    monkeypatch.setattr(ms, "WORKSPACE", tmp_path)
    monkeypatch.setattr(ms, "DESIGNS", tmp_path / "designs")
    monkeypatch.setattr(ms, "RUNS", tmp_path / "runs")
    out = ms.list_designs()
    assert out == {"ok": True, "workspace": str(tmp_path), "designs": []}
    assert ms.create_design("toroid", "t1")["ok"]
    out = ms.list_designs()
    assert [d["name"] for d in out["designs"]] == ["t1"] and out["designs"][0]["solved"] is False
    tpl = ms.list_templates()
    assert tpl["ok"] and {t["name"] for t in tpl["templates"]} >= {"toroid", "solenoid", "core"}


def test_mqs_validity_summary():
    from axionfem._mcp_worker import mqs_validity

    # Psi/Psi_MQS = 1 + (f/f0)^2 with f0 = 1 MHz, L constant: 1% at 100 kHz, 10% at 316 kHz
    f = [10 ** (3 + 0.25 * i) for i in range(13)]          # 1 kHz .. 1 MHz
    sweep = [{"f_Hz": x, "Psi_over_MQS": 1 + (x / 1e6) ** 2, "L_over_MQS": 1.0} for x in f]
    v = mqs_validity(sweep)
    assert v["Psi"]["1%"] == pytest.approx(1e5, rel=1e-9)
    assert v["Psi"]["10%"] == pytest.approx(10 ** 5.5, rel=1e-9)
    assert v["L_p"]["1%"] is None                            # never departs: null, with the sweep top in the note
    # F = Psi^4: (1 + u)^4 - 1 = 0.01 at u = 0.0024907
    u = 1.01 ** 0.25 - 1
    assert v["F"]["1%"] == pytest.approx(1e6 * math.sqrt(u), rel=1e-3)   # (1+x)^4 - 1 is not quite a power law
    assert "1e+06" in v["note"]
    assert mqs_validity(sweep[:1]) is None
    # already departed at the first point: flagged; a 3D-style sweep with L only still reports L_p
    v = mqs_validity([{"f_Hz": 1e3, "Psi_over_MQS": 1.2, "L_over_MQS": 1.0},
                      {"f_Hz": 1e4, "Psi_over_MQS": 1.3, "L_over_MQS": 1.0}])
    assert v["Psi"]["1%"] == 1e3 and "Psi 1%" in v["exceeded_at_start"]
    v = mqs_validity([{"f_Hz": 1e3, "L_over_MQS": 1.0}, {"f_Hz": 1e4, "L_over_MQS": 1.02}])
    assert set(v) == {"L_p", "note"} and 1e3 < v["L_p"]["1%"] < 1e4
    # from an exact zero the crossing is interpolated linearly, not pushed to the next point
    v = mqs_validity([{"f_Hz": 1e3, "L_over_MQS": 1.0}, {"f_Hz": 1e5, "L_over_MQS": 1.02}])
    assert v["L_p"]["1%"] == pytest.approx(1e4)


def test_export_html_is_readable_by_others(small_toroid, tmp_path):
    from axionfem.viewer import _umask

    out = Path(small_toroid.export_html(tmp_path / "s.html"))
    mode = stat.S_IMODE(os.stat(out).st_mode)
    assert mode == 0o666 & ~_umask(), (oct(mode), oct(_umask()))   # the normal mode, not mkstemp's private 0600


def test_mcp_extra_requires_mcp2():
    """The server imports mcp.server.mcpserver, which exists from mcp 2.0 on (1.x calls it FastMCP)."""
    text = (ROOT / "pyproject.toml").read_text()
    pins = re.findall(r'"mcp([<>=!~][^"]*)"', text)
    assert pins and all(p.startswith(">=2") for p in pins), pins


def test_solve_design_compacts_a_cavity_design(tmp_path):
    """A cavity design has no lumped pickup: the reply must not crash on it and must carry the eigenmodes."""
    from axionfem import cavity as cv
    from axionfem._mcp_worker import compact

    res = cv.pillbox_design(a=0.1, L=0.2, modes=2, families=("TM",)).solve(workdir=tmp_path / "pb", verbose=0)
    out = compact(res)
    assert "coupling" not in out and "benchmarks" not in out
    m = out["cavity_modes"][0]
    assert m["f_Hz"] == pytest.approx(2.404826 * 299792458 / (2 * math.pi * 0.1), rel=2e-3)
    json.dumps(out, allow_nan=False)


def test_compact_keeps_the_coupling_of_a_lumped_design_without_pickup(tmp_path):
    """Only eigenmode (cavity) designs drop the lumped coupling: a magnet without a pickup keeps W and sizes."""
    from axionfem._mcp_worker import compact

    d = af.template("toroid", R=0.5, H=0.5)
    d.doc.pop("pickup")
    out = compact(d.solve(workdir=tmp_path / "np", self_force=False, force_samples=0))
    assert out["coupling"]["W_mag_J"] > 0 and "cavity_modes" not in out


def test_set_parameters_save_as_makes_a_variant(tmp_path, monkeypatch):
    """Trying one parameter change must not require rebuilding the design: save_as keeps the original."""
    pytest.importorskip("mcp")
    from axionfem import mcp_server as ms

    monkeypatch.setattr(ms, "DESIGNS", tmp_path / "designs")
    assert ms.create_design("solenoid", "base")["ok"]
    out = ms.set_parameters("base", {"x": 0.3}, save_as="base_x03")
    assert out["ok"] and out["name"] == "base_x03" and out["parameters"]["x"] == 0.3
    assert ms.get_design("base")["design"]["parameters"]["x"] != 0.3
    assert ms.get_design("base_x03")["design"]["name"] == "base_x03"
