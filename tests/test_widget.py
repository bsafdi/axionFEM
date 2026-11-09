"""Tests for the Jupyter widget (python/axionfem/widget.py).

Headless: no browser, no Jupyter kernel and no compiled core.  ``widget.py`` is
loaded straight from its file (like ``viewer.py`` in test_viewer.py), so these
tests run in a bare environment.  They check that

* the widget knows which quantities the scene offers, in display order;
* the HTML it emits embeds the complete viewer page (the scene round-trips out
  of the iframe's ``srcdoc``), is correctly escaped and makes no network
  requests;
* switching the quantity, the scale and the height changes the page the widget
  would display, and bad values raise errors that list the valid ones;
* the anywidget back end, when anywidget is installed, carries the same page
  and quantity list in its synchronised traits.

The browser side (the viewer actually colouring by the quantity) is covered by
the viewer's own browser smoke test, see docs/viewer.md.
"""
from __future__ import annotations

import html as html_mod
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WIDGET_PY = ROOT / "python" / "axionfem" / "widget.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_viewer import _script_block, synthetic_scene  # noqa: E402


def _load_widget():
    spec = importlib.util.spec_from_file_location("axionfem_widget_under_test", WIDGET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load_widget()


# ----------------------------------------------------------------- helpers
def scene_3d(name="unit3d"):
    """A minimal 3D scene (``"symmetry": "3d"``) with one slice."""
    s = synthetic_scene(name=name)
    s["symmetry"] = "3d"
    field = s.pop("field")
    s["slices"] = [
        {
            "plane": {"normal": [0, 1, 0], "offset": 0.0},
            "vertices": field["vertices"],
            "indices": field["indices"],
            "quantities": {k: field["quantities"][k] for k in ("Bmag", "Bz", "pickup_Bmag")},
            "ranges": {k: field["ranges"][k] for k in ("Bmag", "Bz", "pickup_Bmag")},
            "units": {"Bmag": "T", "Bz": "T", "pickup_Bmag": "T/A"},
        }
    ]
    return s


def srcdoc_of(html: str) -> str:
    """The iframe's srcdoc attribute, unescaped: the viewer page itself."""
    m = re.search(r'srcdoc="(.*?)"></iframe>', html, re.S)
    assert m, "no iframe with a srcdoc in the widget HTML"
    return html_mod.unescape(m.group(1))


def widget(scene=None, **kw):
    kw.setdefault("backend", "html")
    return W.SceneWidget(scene if scene is not None else synthetic_scene(), **kw)


# ----------------------------------------------------------------- quantities
def test_scene_quantities_order():
    q = W.scene_quantities(synthetic_scene())
    assert q == ["Bmag", "Br", "Bz", "Bphi", "psi", "pickup_Bmag", "pickup_psi"]


def test_scene_quantities_3d_from_slices():
    assert W.scene_quantities(scene_3d()) == ["Bmag", "Bz", "pickup_Bmag"]


def test_scene_quantities_unknown_names_sort_last():
    s = synthetic_scene()
    s["field"]["quantities"]["zzz_custom"] = s["field"]["quantities"]["Bmag"]
    s["field"]["quantities"]["aaa_custom"] = s["field"]["quantities"]["Bmag"]
    q = W.scene_quantities(s)
    assert q[:7] == ["Bmag", "Br", "Bz", "Bphi", "psi", "pickup_Bmag", "pickup_psi"]
    assert q[7:] == ["aaa_custom", "zzz_custom"]


def test_scene_without_field_has_no_quantities():
    w = widget(synthetic_scene(with_field=False))
    assert w.quantities == [] and w.quantity == ""
    assert "<iframe" in w.html()  # geometry alone still displays


# ----------------------------------------------------------------- the page in the output
def test_html_embeds_the_whole_viewer_page():
    scene = synthetic_scene()
    w = widget(scene)
    html = w.html()
    assert html.count("<iframe") == 1
    assert f'id="{w._id}"' in html
    page = srcdoc_of(html)
    assert page == w.page()
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert "{{AXF_" not in page, "unreplaced placeholder"
    # the scene itself is in the page, byte for byte
    assert json.loads(_script_block(page, "axf-scene")) == json.loads(json.dumps(scene))
    # three.js and viewer.js are inlined as data: URLs, so the page is offline
    assert page.count("data:text/javascript;base64,") == 5
    assert 0.8e6 < len(page.encode()) < 2.5e6


def test_html_escapes_the_page_so_the_attribute_cannot_break_out():
    # the synthetic scene's name contains < & >
    html = widget().html()
    body = html.split('srcdoc="', 1)[1].rsplit('"></iframe>', 1)[0]
    for bad in ('"', "<", ">"):
        assert bad not in body, f"unescaped {bad!r} inside the srcdoc attribute"


def test_html_makes_no_network_requests():
    html = widget().html()
    assert not re.search(r"https?://", html)
    assert "srcdoc=" in html and " src=" not in html


def test_repr_html_is_the_output():
    w = widget()
    assert w._repr_html_() == w.html()


def test_height_and_title():
    w = widget(height=321, title="My <design>")
    assert 'height:321px' in w.html()
    assert "My &lt;design&gt;" in w.html()                       # iframe title attribute
    assert "<title>My &lt;design&gt;</title>" in w.page()
    w.height = 400
    assert "height:400px" in w.html()


# ----------------------------------------------------------------- switching the quantity
def _options(page: str) -> dict:
    return json.loads(_script_block(page, "axf-config"))["options"]


def test_default_quantity_is_the_first_available():
    w = widget()
    assert w.quantity == "Bmag"
    assert _options(w.page())["quantity"] == "Bmag"


def test_set_quantity_changes_the_page():
    w = widget()
    w.quantity = "Bz"
    assert w.quantity == "Bz"
    assert _options(w.page())["quantity"] == "Bz"
    assert '<option value="Bz" selected>Bz</option>' in w.html()


def test_selector_lists_every_quantity():
    w = widget()
    html = w.html()
    for q in w.quantities:
        assert f'<option value="{q}"' in html
    assert html.count("<option") == len(w.quantities)
    assert "setQuantity(s.value)" in html  # the selector drives the live viewer


def test_quantity_must_exist():
    w = widget()
    with pytest.raises(ValueError) as e:
        w.quantity = "Bnope"
    assert "Bnope" in str(e.value) and "pickup_Bmag" in str(e.value)
    assert w.quantity == "Bmag"  # unchanged
    with pytest.raises(ValueError):
        widget(quantity="Bnope")


def test_scale_and_mode():
    w = widget(scale="log")
    assert _options(w.page())["scale"] == "log"
    w.scale = "linear"
    assert _options(w.page())["scale"] == "linear"
    with pytest.raises(ValueError):
        w.scale = "logarithmic"
    w.mode = "section"
    assert _options(w.page())["mode"] == "section"
    with pytest.raises(ValueError):
        w.mode = "both"


def test_3d_scene_has_no_section_mode():
    w = widget(scene_3d())
    assert w.is_3d and w.quantity == "Bmag"
    with pytest.raises(ValueError) as e:
        w.mode = "section"
    assert "3D scenes have no section mode" in str(e.value)
    # and the constructor refuses it too, not only the setter
    with pytest.raises(ValueError):
        widget(scene_3d(), mode="section")


def test_options_pass_through():
    w = widget(options={"colormap": "plasma", "theme": "dark"})
    o = _options(w.page())
    assert o["colormap"] == "plasma" and o["theme"] == "dark" and o["quantity"] == "Bmag"


def test_accepts_a_scene_file(tmp_path):
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(synthetic_scene(name="fromfile")))
    assert widget(p).name == "fromfile"
    assert widget(tmp_path).name == "fromfile"  # a directory holding scene.json


def test_export_html_from_the_widget(tmp_path):
    w = widget(quantity="psi")
    out = Path(w.export_html(tmp_path / "w.html"))
    assert _options(out.read_text("utf-8"))["quantity"] == "psi"


def test_show_without_ipython_returns_the_widget():
    # IPython is not a dependency; show() must still hand back a displayable object.
    w = widget()
    assert w.show() is w
    assert "<iframe" in w._repr_html_()


def test_repr_mentions_the_backend_and_quantity():
    r = repr(widget())
    assert "SceneWidget(" in r and "backend='html'" in r and "quantity='Bmag'" in r


def test_backend_choice_is_validated():
    with pytest.raises(ValueError):
        W.SceneWidget(synthetic_scene(), backend="qt")


def test_backend_env_override(monkeypatch):
    monkeypatch.setenv("AXF_WIDGET_BACKEND", "html")
    assert W.have_anywidget() is False
    monkeypatch.delenv("AXF_WIDGET_BACKEND")


def test_module_level_show_helper():
    w = W.show(synthetic_scene(), backend="html")
    assert isinstance(w, W.SceneWidget)


# ----------------------------------------------------------------- anywidget back end
def test_esm_and_css_are_sane():
    assert "export default { render }" in W._ESM
    assert "axfViewer" in W._ESM and "srcdoc" in W._ESM
    assert "setQuantity" in W._ESM and "setScale" in W._ESM
    assert ".axf-widget-frame" in W._CSS


@pytest.mark.skipif(not W.have_anywidget(), reason="anywidget is not installed")
def test_anywidget_backend_carries_the_page():
    scene = synthetic_scene()
    w = W.SceneWidget(scene, backend="anywidget")
    impl = w._anywidget()
    assert impl.quantities == w.quantities
    assert impl.quantity == "Bmag"
    assert json.loads(_script_block(impl.page, "axf-scene")) == json.loads(json.dumps(scene))
    w.quantity = "Bz"
    assert impl.quantity == "Bz" and w.quantity == "Bz"
    w.height = 700
    assert impl.height == 700
    w.scale = "log"
    assert impl.scale == "log"


@pytest.mark.skipif(not W.have_anywidget(), reason="anywidget is not installed")
def test_anywidget_class_is_cached():
    assert W._anywidget_class() is W._anywidget_class()


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="node not available")
def test_esm_syntax_with_node(tmp_path):
    import subprocess

    p = tmp_path / "esm.mjs"
    p.write_text(W._ESM)
    subprocess.run(["node", "--check", str(p)], check=True)


def test_javascript_parses():
    """Both snippets this module emits must be syntactically valid JavaScript.

    A pure-Python parser, so this runs on machines without Node.js.
    """
    esprima = pytest.importorskip("esprima", reason="esprima is not installed")
    esprima.parseModule(W._ESM)                      # the anywidget ESM
    inline = re.search(r"<script>(.*?)</script>", widget().html(), re.S)
    assert inline, "the fallback emits no selector script"
    esprima.parseScript(inline.group(1))             # the fallback's selector script


# ----------------------------------------------------------------- the example notebook
WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"


def test_example_notebook_is_valid_and_stores_no_widget_state():
    """The quick-start notebook is stored executed, but without widget views or widget state (they would add
    megabytes and cannot render without a kernel)."""
    nb_path = ROOT / "examples" / "notebook_quickstart.ipynb"
    nb = json.loads(nb_path.read_text("utf-8"))
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code, "the notebook has no code cells"
    ids = [c.get("id") for c in nb["cells"]]
    # nbformat 4.5 requires a unique id on every cell.
    assert all(ids) and len(set(ids)) == len(ids), f"missing or duplicate cell ids: {ids}"
    assert "widgets" not in nb.get("metadata", {}), "strip metadata.widgets before committing"
    for c in code:
        for out in c.get("outputs", []):
            assert WIDGET_VIEW_MIME not in out.get("data", {}), "strip widget views before committing"
    src = "\n".join("".join(c["source"]) for c in code)
    assert "res.show(" in src, "the notebook must demonstrate Results.show()"
    assert "w.quantity" in src, "the notebook must demonstrate switching the quantity"
