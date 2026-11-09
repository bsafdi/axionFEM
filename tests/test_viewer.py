"""Tests for the viewer: single-file HTML export and the live server.

python/axionfem/viewer.py is loaded directly from its file, so these tests do
not depend on the compiled core or on axionfem/__init__.py.

Visual checks (headless Firefox screenshots) are described in docs/viewer.md;
a smoke test runs when AXIONFEM_BROWSER_TESTS=1 and Firefox is available.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWER_PY = ROOT / "python" / "axionfem" / "viewer.py"
VIEWER_DIR = ROOT / "viewer"
SAMPLES = [ROOT / "runs" / n / "scene.json" for n in ("toroid", "solenoid", "core")]


def _load_viewer():
    spec = importlib.util.spec_from_file_location("axionfem_viewer_under_test", VIEWER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load_viewer()


# ----------------------------------------------------------------- helpers
def _arr(values, cols, dtype="float32"):
    code = "f" if dtype == "float32" else "I"
    raw = struct.pack("<%d%s" % (len(values), code), *values)
    return {"dtype": dtype, "shape": [len(values) // cols, cols], "data": base64.b64encode(raw).decode()}


def synthetic_scene(name="unit<&>test", with_field=True, curves=False):
    """A tiny scene: a rectangular Nb shell plus a field sampled on a grid."""
    shell = [0.4, -0.5, 0.5, -0.5, 0.5, 0.5, 0.4, 0.5]
    outline = shell + shell[:2]  # closed loop with the closing duplicate
    scene = {
        "schema": "axionfem.scene/1",
        "name": name,
        "symmetry": "axisymmetric",
        "units": "m",
        "extent": {"r_max": 0.5, "z_min": -0.5, "z_max": 0.5},
        "view": {"r_max": 1.0, "z_min": -1.0, "z_max": 1.0},
        "regions": [
            {
                "name": "shell",
                "role": "superconductor",
                "material": "nb",
                "color": "#7c8ea6",
                "triangles": {"vertices": _arr(shell, 2), "indices": _arr([0, 1, 2, 0, 2, 3], 3, "uint32")},
                "outline": [_arr(outline, 2)],
            }
        ],
        "curves": [],
        "summary": {
            "coupling": {"Psi_SI_H_T_m2": 1.5e-7, "L_p_H": 2.5e-7, "W_mag_J": 6.8e5, "g_factor": 0.12,
                         "F_natural_T4_m10": 0.011, "note": "test"},
            "fields": {"B_max_T": 0.65, "by_region_T": {"air": 0.6}, "superconductor_surface_T": {"shell": 0.3}},
        },
        "design": {"schema": "axionfem.design/1", "name": name, "description": "Synthetic scene. For tests.",
                   "parameters": {"R": 1.0}},
    }
    if curves:
        scene["curves"] = [{"name": "sheet", "role": "superconductor", "color": "#7c8ea6",
                            "segments": _arr([0.7, -0.3, 0.7, 0.3], 2)}]
    if with_field:
        n = 9
        verts, idx = [], []
        for j in range(n):
            for i in range(n):
                verts += [i / (n - 1), -1 + 2 * j / (n - 1)]
        for j in range(n - 1):
            for i in range(n - 1):
                a = j * n + i
                idx += [a, a + 1, a + n + 1, a, a + n + 1, a + n]
        r = verts[0::2]
        q = {
            "Bmag": [abs(x - 0.45) for x in r],
            "Br": [0.0] * len(r),
            "Bz": [0.45 - x for x in r],
            "Bphi": [0.0] * len(r),
            "psi": [x * x / 2 for x in r],
            "pickup_Bmag": [1e-7 * x for x in r],
            "pickup_psi": [0.0] * len(r),
        }
        scene["field"] = {
            "vertices": _arr(verts, 2),
            "indices": _arr(idx, 3, "uint32"),
            "quantities": {k: _arr(v, 1) for k, v in q.items()},
            "ranges": {k: [min(v), max(v)] for k, v in q.items()},
            "units": {"Bmag": "T", "Br": "T", "Bz": "T", "Bphi": "T", "psi": "Wb/rad",
                      "pickup_Bmag": "T/A", "pickup_psi": "Wb/(rad A)"},
        }
        scene["field_lines"] = {"static": _arr([0.2, -1, 0.2, 1, 0.3, -1, 0.3, 1], 2), "pickup": _arr([], 2)}
    return scene


def _script_block(page: str, sid: str) -> str:
    m = re.search(r'<script type="application/json" id="%s">(.*?)</script>' % re.escape(sid), page, re.S)
    assert m, f"missing <script id={sid}>"
    return m.group(1)


def _import_map(page: str) -> dict:
    m = re.search(r'<script type="importmap">(.*?)</script>', page, re.S)
    assert m, "missing import map"
    return json.loads(m.group(1))["imports"]


def _data_url_text(url: str) -> str:
    prefix = "data:text/javascript;base64,"
    assert url.startswith(prefix), url[:60]
    return base64.b64decode(url[len(prefix):]).decode("utf-8")


def _check_page(page: str, scene: dict) -> None:
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert "{{AXF_" not in page, "unreplaced placeholder"
    # exactly the expected script blocks: config, scene, import map, classic boot, module boot
    assert page.count("<script") == 5
    imports = _import_map(page)
    assert set(imports) == {
        "three",
        "axf-three-core",
        "three/addons/controls/OrbitControls.js",
        "three/addons/environments/RoomEnvironment.js",
        "axionfem-viewer",
    }
    vend = VIEWER_DIR / "vendor"
    assert _data_url_text(imports["axf-three-core"]) == (vend / "three.core.min.js").read_text("utf-8")
    module = (vend / "three.module.min.js").read_text("utf-8")
    assert _data_url_text(imports["three"]) == module.replace('"./three.core.min.js"', '"axf-three-core"')
    assert _data_url_text(imports["three/addons/controls/OrbitControls.js"]) == (
        vend / "addons/controls/OrbitControls.js"
    ).read_text("utf-8")
    assert _data_url_text(imports["axionfem-viewer"]) == (VIEWER_DIR / "src/viewer.js").read_text("utf-8")
    raw = _script_block(page, "axf-scene")
    assert "<" not in raw and "</script" not in raw.lower()
    assert json.loads(raw) == json.loads(json.dumps(scene))  # exact round trip
    cfg = json.loads(_script_block(page, "axf-config"))
    assert cfg["three"] == V.THREE_VERSION and "sceneUrl" not in cfg


# ----------------------------------------------------------------- assets
def test_vendor_hashes_match_readme():
    readme = (VIEWER_DIR / "vendor" / "README.md").read_text("utf-8")
    rows = re.findall(r"^\| `([^`]+)` \| `[^`]+` \| `([0-9a-f]{64})` \|$", readme, re.M)
    assert len(rows) == 5
    for rel, digest in rows:
        data = (VIEWER_DIR / "vendor" / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, rel
    assert V.THREE_VERSION in readme
    assert "MIT" in (VIEWER_DIR / "vendor" / "LICENSE-three.txt").read_text("utf-8")


def test_template_and_viewer_module():
    tpl = (VIEWER_DIR / "template.html").read_text("utf-8")
    names = set(re.findall(r"\{\{(AXF_[A-Z_]+)\}\}", tpl))
    assert names == set(V._PLACEHOLDERS)
    js = (VIEWER_DIR / "src" / "viewer.js").read_text("utf-8")
    assert "export function createViewer(container, scene, options = {})" in js
    for method in ("setQuantity(", "setScale(", "setMode(", "setCutaway(", "toggleLayer(", "screenshot", "dispose()"):
        assert method in js, method
    assert "{{AXF_" not in js
    # no network access from the viewer itself
    assert not re.search(r"https?://(?!threejs\.org)", js.split("*/", 1)[1])


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_viewer_js_syntax():
    subprocess.run(["node", "--check", str(VIEWER_DIR / "src" / "viewer.js")], check=True)


# ----------------------------------------------------------------- export
def test_export_synthetic(tmp_path):
    scene = synthetic_scene(curves=True)
    out = V.export_html(scene, tmp_path / "s.html", title="A <b>&</b> test")
    assert out == str(tmp_path / "s.html")
    page = Path(out).read_text("utf-8")
    _check_page(page, scene)
    assert "<title>A &lt;b&gt;&amp;&lt;/b&gt; test</title>" in page
    assert 0.8e6 < len(page.encode()) < 2.0e6  # three.js + viewer inlined


def test_export_without_field_and_options(tmp_path):
    scene = synthetic_scene(with_field=False)
    out = V.export_html(scene, tmp_path / "nofield.html", options={"mode": "section", "scale": "log"})
    page = Path(out).read_text("utf-8")
    _check_page(page, scene)
    assert json.loads(_script_block(page, "axf-config"))["options"] == {"mode": "section", "scale": "log"}


def test_export_to_directory_and_path_input(tmp_path):
    src = tmp_path / "run" / "scene.json"
    src.parent.mkdir()
    src.write_text(json.dumps(synthetic_scene(name="my design")))
    out = V.export_html(src.parent, tmp_path)  # a run directory in, a directory out
    assert Path(out).name == "my_design.html"
    assert "my design" in Path(out).read_text("utf-8")


def test_numpy_arrays_and_nonfinite_values(tmp_path):
    np = pytest.importorskip("numpy")
    scene = synthetic_scene()
    scene["regions"][0]["triangles"]["vertices"] = np.array([[0.4, -0.5], [0.5, -0.5], [0.5, 0.5], [0.4, 0.5]])
    scene["regions"][0]["triangles"]["indices"] = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    scene["summary"]["coupling"]["g_factor"] = float("nan")
    scene["summary"]["coupling"]["W_mag_J"] = np.float64(1.5)
    data = V.load_scene(scene)
    tri = data["regions"][0]["triangles"]
    assert tri["vertices"]["dtype"] == "float32" and tri["vertices"]["shape"] == [4, 2]
    assert tri["indices"]["dtype"] == "uint32" and tri["indices"]["shape"] == [2, 3]
    assert np.frombuffer(base64.b64decode(tri["indices"]["data"]), "<u4").tolist() == [0, 1, 2, 0, 2, 3]
    assert data["summary"]["coupling"]["g_factor"] is None
    assert data["summary"]["coupling"]["W_mag_J"] == 1.5
    page = Path(V.export_html(scene, tmp_path / "np.html")).read_text("utf-8")
    assert "NaN" not in _script_block(page, "axf-scene")


def test_load_scene_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        V.load_scene(tmp_path / "missing.json")
    bad = tmp_path / "partial.json"
    bad.write_text(json.dumps(synthetic_scene())[:200])
    with pytest.raises(ValueError, match="still being written"):
        V.load_scene(bad)
    with pytest.raises(ValueError, match="schema"):
        V.load_scene({"schema": "something/else", "regions": []})
    with pytest.raises(ValueError, match="regions"):
        V.load_scene({"schema": "axionfem.scene/1"})
    with pytest.raises(TypeError):
        V.load_scene(42)


@pytest.mark.parametrize("path", SAMPLES, ids=[p.parent.name for p in SAMPLES])
def test_export_sample_scenes(tmp_path, path):
    if not path.is_file():
        pytest.skip(f"{path} not present (runs/ is not tracked)")
    scene = json.loads(path.read_text("utf-8"))
    out = V.export_html(path, tmp_path / f"{path.parent.name}.html")
    page = Path(out).read_text("utf-8")
    _check_page(page, scene)
    size, scene_size = Path(out).stat().st_size, path.stat().st_size
    # the scene is embedded as compact JSON next to ~1.2 MB of inlined code
    assert scene_size * 0.9 < size < scene_size + 2.0e6
    assert f"<title>{scene['name']} · axionFEM</title>" in page


def test_cli_export(tmp_path, capsys):
    src = tmp_path / "scene.json"
    src.write_text(json.dumps(synthetic_scene(name="cli")))
    assert V.main(["export", str(src), str(tmp_path / "cli.html"), "--mode", "section"]) == 0
    assert (tmp_path / "cli.html").is_file()
    assert "cli.html" in capsys.readouterr().out


# ----------------------------------------------------------------- live server
def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def test_live_server(tmp_path):
    run = tmp_path / "runs" / "a"
    run.mkdir(parents=True)
    scene_file = run / "scene.json"
    scene_file.write_text(json.dumps(synthetic_scene(name="live")))
    srv = V.serve(tmp_path / "runs", port=0, block=False)
    try:
        base = srv.url
        st, ct, body = _get(base, "api/scenes")
        lst = json.loads(body)
        assert st == 200 and len(lst) == 1 and lst[0]["name"] == "a"
        sid = lst[0]["id"]
        st, ct, body = _get(base, "")
        assert st == 200 and b"axionFEM live viewer" in body
        st, ct, body = _get(base, f"view/{sid}")
        page = body.decode()
        assert st == 200 and "text/html" in ct and "{{AXF_" not in page
        imports = _import_map(page)
        assert imports["three"] == "../assets/vendor/three.module.min.js"
        cfg = json.loads(_script_block(page, "axf-config"))
        assert cfg["sceneUrl"] == f"../scene/{sid}.json" and cfg["version"]
        assert json.loads(_script_block(page, "axf-scene"))["name"] == "live"  # first frame embedded
        st, ct, body = _get(base, f"scene/{sid}.json")
        assert st == 200 and json.loads(body)["name"] == "live"
        v0 = json.loads(_get(base, f"api/version/{sid}")[2])["version"]
        for rel in V._ALLOWED_ASSETS:
            st, ct, body = _get(base, f"assets/{rel}")
            assert st == 200 and "javascript" in ct and body == (VIEWER_DIR / rel).read_bytes()
        for bad in ("assets/../template.html", "assets/vendor/README.md", "scene/0000000000.json", "nope"):
            assert _get(base, bad)[0] == 404, bad
        # a partially written file is reported as not ready; a complete one gets a new version
        full = json.dumps(synthetic_scene(name="live2"), separators=(",", ":"))
        scene_file.write_text(full[: len(full) // 2])
        assert _get(base, f"scene/{sid}.json")[0] == 503
        time.sleep(0.02)
        scene_file.write_text(full)
        v1 = json.loads(_get(base, f"api/version/{sid}")[2])["version"]
        assert v1 != v0
        assert json.loads(_get(base, f"scene/{sid}.json")[2])["name"] == "live2"
        # NaN tokens are not valid JSON for browsers: they are served as null
        txt = full.replace('"g_factor":0.12', '"g_factor":NaN')
        assert txt != full
        scene_file.write_text(txt)
        served = _get(base, f"scene/{sid}.json")[2].decode()
        assert "NaN" not in served and json.loads(served)["summary"]["coupling"]["g_factor"] is None
        # new runs appear without restarting the server
        (tmp_path / "runs" / "b").mkdir()
        (tmp_path / "runs" / "b" / "scene.json").write_text(json.dumps(synthetic_scene(name="b")))
        assert {s["name"] for s in json.loads(_get(base, "api/scenes")[2])} == {"a", "b"}
        st, ct, body = _get(base, f"export/{sid}.html")
        assert st == 200 and b"data:text/javascript;base64," in body
    finally:
        srv.stop()


def test_serve_single_file_redirects(tmp_path):
    f = tmp_path / "one.scene.json"
    f.write_text(json.dumps(synthetic_scene(name="one")))
    srv = V.serve(f, port=0, block=False)
    try:
        with urllib.request.urlopen(srv.url, timeout=10) as r:
            assert "/view/" in r.geturl()
    finally:
        srv.stop()


def test_serve_port_in_use(tmp_path):
    f = tmp_path / "scene.json"
    f.write_text(json.dumps(synthetic_scene()))
    srv = V.serve(f, port=0, block=False)
    try:
        with pytest.raises(OSError, match="another port"):
            V.serve(f, port=srv.server_address[1], block=False)
    finally:
        srv.stop()


# ----------------------------------------------------------------- browser smoke test
@pytest.mark.skipif(
    os.environ.get("AXIONFEM_BROWSER_TESTS") != "1" or not Path("/usr/bin/firefox").exists(),
    reason="set AXIONFEM_BROWSER_TESTS=1 (needs Firefox) to run the headless rendering smoke test",
)
def test_browser_canvas_fallback_renders(tmp_path):
    out = V.export_html(synthetic_scene(), tmp_path / "s.html")
    png = tmp_path / "s.png"
    profile = tmp_path / "profile"
    profile.mkdir()
    subprocess.run(
        ["/usr/bin/firefox", "--headless", "--no-remote", "--profile", str(profile), "--screenshot", str(png),
         "--window-size=900,600", f"file://{out}?renderer=canvas&mode=section&ui=0"],
        timeout=180, capture_output=True,
    )
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000
