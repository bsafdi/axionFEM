"""Jupyter widget: the axionFEM three.js viewer inside a notebook.

``Results.show()`` and ``Results3D.show()`` return a :class:`SceneWidget`, which
displays the same self-contained page that :func:`axionfem.viewer.export_html`
writes, inside an ``<iframe srcdoc=...>``.  Nothing is fetched from the network
and the notebook needs no Jupyter server extension or ``nbextension install``
step: the whole viewer (three.js, ``viewer.js``, the scene) travels in the
output of the cell.

Two back ends, chosen automatically:

* **anywidget** (used when ``anywidget`` is installed).  The quantity selector
  and the Python attribute ``w.quantity`` are synchronised both ways through
  the widget comm, so ``w.quantity = "Bz"`` recolours the *live* viewer without
  re-sending the page, and picking a quantity in the widget updates the Python
  object.  ``w.height`` and ``w.scale`` work the same way.
* **Fallback** (no extra packages).  ``_repr_html_`` returns the iframe plus a
  plain ``<select>``; the selector and the viewer's own control panel work
  entirely in the browser.  Setting ``w.quantity`` from Python re-renders the
  cell output in place when IPython is available (``display_id``), and
  otherwise just changes what ``w.html()`` returns.

Quantity switching inside the page always works through the viewer's own
control panel, which is part of the exported page.

Limitations are listed in docs/viewer.md; the notebook-specific ones:

* each widget carries its own copy of three.js (about 1.2 MB) plus the scene,
  because the page is self-contained.  Clear the outputs before committing a
  notebook;
* the iframe needs import maps and ``data:`` module URLs, so JupyterLab 3.1+,
  Notebook 7, VS Code 1.80+ or Colab.  In an *untrusted* notebook JupyterLab
  strips ``<iframe>`` and ``<script>`` from HTML output; re-run the cell (or
  trust the notebook) to see the viewer;
* WebGL must be available in the front end.  Without it the page falls back to
  the Canvas2D section renderer, as documented in docs/viewer.md.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import html as _html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

__all__ = ["SceneWidget", "show", "scene_quantities", "have_anywidget"]

# Quantities are offered in this order when the scene has them; anything else
# in the scene follows, sorted.  (viewer.js applies the same preference.)
_QUANTITY_ORDER = (
    "Bmag", "Br", "Bz", "Bphi", "Bx", "By", "psi",
    "pickup_Bmag", "pickup_Br", "pickup_Bz", "pickup_Bx", "pickup_By", "pickup_psi",
)

DEFAULT_HEIGHT = 560

# Distinct DOM ids when several widgets share a page.
_widget_count = 0

_NO_SECTION_IN_3D = (
    "3D scenes have no section mode; use Results3D.scene() with the viewer's "
    "cut controls, or solve the axisymmetric design for a section view"
)


# --------------------------------------------------------------------------- viewer module
def _viewer_module():
    """``axionfem.viewer``, also when this file is loaded on its own.

    ``viewer.py`` deliberately imports no other ``axionfem`` module, so the pair
    works without the compiled core (tests, documentation builds).
    """
    try:
        from . import viewer as _v  # normal package import
    except ImportError:
        import importlib.util
        import sys

        path = Path(__file__).resolve().parent / "viewer.py"
        name = "axionfem_viewer_standalone"
        mod = sys.modules.get(name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                raise ImportError(f"cannot load the axionFEM viewer module from {path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        _v = mod
    return _v


def have_anywidget() -> bool:
    """True when the anywidget back end can be used.

    Set ``AXF_WIDGET_BACKEND=html`` to force the fallback (used by the tests so
    that both back ends are exercised on a machine that has anywidget).
    """
    if os.environ.get("AXF_WIDGET_BACKEND", "").lower() in ("html", "iframe", "fallback"):
        return False
    try:
        import anywidget  # noqa: F401
        import traitlets  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- scene helpers
def scene_quantities(scene: Mapping[str, Any]) -> list[str]:
    """Field quantities that can be plotted for ``scene``, in display order.

    Axisymmetric scenes keep them in ``field.quantities``; 3D scenes keep one
    set per slice in ``slices[].quantities`` (docs/scene_format.md).
    """
    names: set[str] = set()
    field = scene.get("field")
    if isinstance(field, Mapping):
        q = field.get("quantities")
        if isinstance(q, Mapping):
            names.update(str(k) for k in q)
    for sl in scene.get("slices") or []:
        if isinstance(sl, Mapping) and isinstance(sl.get("quantities"), Mapping):
            names.update(str(k) for k in sl["quantities"])
    ordered = [q for q in _QUANTITY_ORDER if q in names]
    return ordered + sorted(names - set(ordered))


def _default_quantity(quantities: Sequence[str]) -> str:
    return quantities[0] if quantities else ""


# --------------------------------------------------------------------------- the widget
class SceneWidget:
    """A notebook view of one scene.

    ``SceneWidget(scene)`` accepts anything :func:`axionfem.viewer.load_scene`
    accepts: a scene dict, a path to ``scene.json``, or a ``Results`` object.
    Display it by returning it from a cell, or call :meth:`show`.

    Attributes that can be read and written: ``quantity``, ``scale``
    (``"linear"``/``"log"``), ``mode`` (``"3d"``/``"section"``; axisymmetric
    scenes only) and ``height`` in pixels.  ``quantities`` lists the choices.
    """

    def __init__(
        self,
        scene: Any,
        title: Optional[str] = None,
        *,
        quantity: Optional[str] = None,
        mode: Optional[str] = None,
        scale: str = "linear",
        height: int = DEFAULT_HEIGHT,
        options: Optional[Mapping[str, Any]] = None,
        backend: Optional[str] = None,
    ):
        v = _viewer_module()
        self._scene = v.load_scene(scene)
        self.name = str(self._scene.get("name") or "design")
        self.title = title or self.name
        self.quantities = scene_quantities(self._scene)
        self.is_3d = str(self._scene.get("symmetry", "axisymmetric")) == "3d"
        self._options = dict(options or {})
        if quantity is None:
            quantity = str(self._options.get("quantity") or _default_quantity(self.quantities))
        elif self.quantities and quantity not in self.quantities:
            raise ValueError(
                f"unknown quantity {quantity!r} for scene {self.name!r}; "
                f"available: {', '.join(self.quantities) or '(none: the scene has no field)'}"
            )
        self._quantity = quantity
        self._scale = str(scale)
        if self._scale not in ("linear", "log"):
            raise ValueError(f"scale must be 'linear' or 'log', got {scale!r}")
        self._mode = str(mode or self._options.get("mode") or "3d")
        if self._mode not in ("3d", "section"):
            raise ValueError(f"mode must be '3d' or 'section', got {mode!r}")
        if self.is_3d and self._mode == "section":
            raise ValueError(_NO_SECTION_IN_3D)
        self._height = int(height)
        global _widget_count
        _widget_count += 1
        self._id = f"axf-w{_widget_count}"
        if backend is None:
            backend = "anywidget" if have_anywidget() else "html"
        if backend not in ("anywidget", "html"):
            raise ValueError(f"backend must be 'anywidget' or 'html', got {backend!r}")
        self.backend = backend
        self._impl = None  # the anywidget instance, created on demand
        self._handle = None  # IPython display handle for in-place updates

    # ------------------------------------------------------------ state
    def _viewer_options(self) -> dict:
        o = dict(self._options)
        o["mode"] = self._mode
        o["scale"] = self._scale
        if self._quantity:
            o["quantity"] = self._quantity
        # The viewer's own panels are left at their default (open, and closed by
        # viewer.js itself below 760 px): in the fallback back end they are the
        # controls that always work, even when a renderer strips our <script>.
        return o

    @property
    def scene(self) -> dict:
        """The scene dictionary being displayed."""
        return self._scene

    @property
    def quantity(self) -> str:
        if self._impl is not None:
            return str(self._impl.quantity)
        return self._quantity

    @quantity.setter
    def quantity(self, name: str) -> None:
        name = str(name)
        if self.quantities and name not in self.quantities:
            raise ValueError(
                f"unknown quantity {name!r}; available: {', '.join(self.quantities)}"
            )
        self._quantity = name
        if self._impl is not None:
            self._impl.quantity = name  # live update through the widget comm
        else:
            self._refresh()

    @property
    def scale(self) -> str:
        if self._impl is not None:
            return str(self._impl.scale)
        return self._scale

    @scale.setter
    def scale(self, value: str) -> None:
        if value not in ("linear", "log"):
            raise ValueError(f"scale must be 'linear' or 'log', got {value!r}")
        self._scale = value
        if self._impl is not None:
            self._impl.scale = value
        else:
            self._refresh()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("3d", "section"):
            raise ValueError(f"mode must be '3d' or 'section', got {value!r}")
        if self.is_3d and value == "section":
            raise ValueError(_NO_SECTION_IN_3D)
        self._mode = value
        self._rebuild()

    @property
    def height(self) -> int:
        return int(self._impl.height) if self._impl is not None else self._height

    @height.setter
    def height(self, px: int) -> None:
        self._height = int(px)
        if self._impl is not None:
            self._impl.height = int(px)
        else:
            self._refresh()

    # ------------------------------------------------------------ pages
    def page(self) -> str:
        """The self-contained viewer page (the same HTML ``export_html`` writes)."""
        v = _viewer_module()
        return v.render_html(self._scene, self.title, options=self._viewer_options())

    def html(self) -> str:
        """The notebook output: a quantity selector and the page in an iframe."""
        page = self.page()
        frame_id = self._id
        srcdoc = _html.escape(page, quote=True)
        sel = ""
        if len(self.quantities) > 1:
            opts = "".join(
                '<option value="%s"%s>%s</option>'
                % (_html.escape(q, quote=True), " selected" if q == self._quantity else "", _html.escape(q))
                for q in self.quantities
            )
            sel = (
                '<label style="font:13px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;'
                'display:inline-flex;gap:.4em;align-items:center">quantity'
                f'<select id="{frame_id}-q" style="font:inherit">{opts}</select></label>'
            )
        return (
            f'<div class="axf-widget" style="max-width:100%">{sel}'
            f'<iframe id="{frame_id}" title="{_html.escape(self.title, quote=True)}" '
            f'style="width:100%;height:{self._height}px;border:1px solid rgba(128,128,128,.35);'
            'border-radius:6px;display:block;margin-top:4px" '
            f'srcdoc="{srcdoc}"></iframe>'
            + (
                "<script>(function(){var s=document.getElementById(%s),f=document.getElementById(%s);"
                "if(!s||!f)return;s.addEventListener('change',function(){try{"
                "f.contentWindow.axfViewer.setQuantity(s.value);}catch(e){"
                "console.warn('axionFEM widget: '+e.message);}});})();</script>"
                % (json.dumps(frame_id + "-q"), json.dumps(frame_id))
                if sel
                else ""
            )
            + "</div>"
        )

    def _repr_html_(self) -> str:
        return self.html()

    # ------------------------------------------------------------ display
    def _anywidget(self):
        if self._impl is None:
            cls = _anywidget_class()
            self._impl = cls(
                page=self.page(),
                quantity=self._quantity,
                quantities=list(self.quantities),
                scale=self._scale,
                height=self._height,
                title=self.title,
            )
        return self._impl

    def _rebuild(self) -> None:
        """The page itself changed (mode): resend it."""
        if self._impl is not None:
            self._impl.quantity = self._quantity
            self._impl.scale = self._scale
            self._impl.page = self.page()
        else:
            self._refresh()

    def _refresh(self) -> None:
        """Fallback back end: redraw the cell output in place, if we own one."""
        if self._handle is not None:
            try:
                self._handle.update(self)
            except Exception:  # pragma: no cover - a stale handle must not raise
                pass

    def show(self):
        """Display the widget in the current notebook cell and return ``self``.

        Outside IPython nothing is displayed and the object is returned as is;
        its ``_repr_html_`` still carries the whole page, so a notebook that
        merely echoes it also works.
        """
        impl = self._anywidget() if self.backend == "anywidget" else self
        try:
            from IPython.display import display
        except Exception:
            return self
        if impl is self:
            # Keep the handle so that setting .quantity can redraw in place.
            self._handle = display(self, display_id=True)
        else:
            display(impl)
        return self

    def _ipython_display_(self):  # pragma: no cover - needs IPython
        if self.backend == "anywidget":
            from IPython.display import display

            display(self._anywidget())
        else:
            from IPython.display import display_html

            display_html(self.html(), raw=True)

    def export_html(self, path: str | os.PathLike) -> str:
        """Write the same page to a file (see ``axionfem.viewer.export_html``)."""
        v = _viewer_module()
        return v.export_html(self._scene, path, self.title, options=self._viewer_options())

    def __repr__(self) -> str:
        return (
            f"SceneWidget({self.name!r}, backend={self.backend!r}, quantity={self.quantity!r}, "
            f"quantities={self.quantities}, {'3d' if self.is_3d else 'axisymmetric'})"
        )


def show(scene: Any, **kw):
    """``SceneWidget(scene, **kw).show()`` — display a scene in a notebook."""
    return SceneWidget(scene, **kw).show()


# --------------------------------------------------------------------------- anywidget back end
# The viewer page is a complete document (it needs an import map, which only
# works in a document), so the widget hosts it in a same-origin srcdoc iframe
# and talks to `window.axfViewer`, the API object the page exports.
_ESM = r"""
function waitForViewer(frame, tries = 400) {
  return new Promise((resolve, reject) => {
    const tick = (n) => {
      let v = null;
      try { v = frame.contentWindow && frame.contentWindow.axfViewer; } catch (e) { /* not ready */ }
      if (v) return resolve(v);
      if (n <= 0) return reject(new Error('the viewer did not start in the iframe'));
      setTimeout(() => tick(n - 1), 50);
    };
    tick(tries);
  });
}

function render({ model, el }) {
  el.classList.add('axf-widget');
  const bar = document.createElement('div');
  bar.className = 'axf-widget-bar';
  const label = document.createElement('label');
  label.textContent = 'quantity ';
  const sel = document.createElement('select');
  label.appendChild(sel);
  const scaleLabel = document.createElement('label');
  scaleLabel.textContent = 'scale ';
  const scaleSel = document.createElement('select');
  for (const s of ['linear', 'log']) {
    const o = document.createElement('option');
    o.value = o.textContent = s;
    scaleSel.appendChild(o);
  }
  scaleLabel.appendChild(scaleSel);
  const note = document.createElement('span');
  note.className = 'axf-widget-note';
  bar.append(label, scaleLabel, note);

  const frame = document.createElement('iframe');
  frame.className = 'axf-widget-frame';
  frame.title = model.get('title') || 'axionFEM viewer';

  const quantities = () => model.get('quantities') || [];
  function fillQuantities() {
    sel.textContent = '';
    for (const q of quantities()) {
      const o = document.createElement('option');
      o.value = o.textContent = q;
      sel.appendChild(o);
    }
    sel.value = model.get('quantity') || '';
    label.style.display = quantities().length > 1 ? '' : 'none';
  }

  let api = null;
  function load() {
    api = null;
    note.textContent = '';
    frame.srcdoc = model.get('page') || '';
    waitForViewer(frame)
      .then((v) => {
        api = v;
        apply();
      })
      .catch((e) => { note.textContent = e.message; });
  }
  function apply() {
    if (!api) return;
    try {
      const q = model.get('quantity');
      if (q) api.setQuantity(q);
      api.setScale(model.get('scale') || 'linear');
    } catch (e) {
      note.textContent = e.message;
    }
  }
  function applyHeight() {
    frame.style.height = (model.get('height') || 560) + 'px';
  }

  sel.addEventListener('change', () => { model.set('quantity', sel.value); model.save_changes(); });
  scaleSel.addEventListener('change', () => { model.set('scale', scaleSel.value); model.save_changes(); });

  model.on('change:page', load);
  model.on('change:quantities', fillQuantities);
  model.on('change:quantity', () => { sel.value = model.get('quantity') || ''; apply(); });
  model.on('change:scale', () => { scaleSel.value = model.get('scale') || 'linear'; apply(); });
  model.on('change:height', applyHeight);

  el.append(bar, frame);
  fillQuantities();
  scaleSel.value = model.get('scale') || 'linear';
  applyHeight();
  load();

  return () => { frame.srcdoc = 'about:blank'; api = null; };
}

export default { render };
"""

_CSS = """
.axf-widget { max-width: 100%; }
.axf-widget-bar { display: flex; gap: 1em; align-items: center; flex-wrap: wrap;
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; margin-bottom: 4px; }
.axf-widget-bar label { display: inline-flex; gap: .4em; align-items: center; }
.axf-widget-bar select { font: inherit; }
.axf-widget-note { color: #cf222e; }
.axf-widget-frame { width: 100%; border: 1px solid rgba(128, 128, 128, .35);
  border-radius: 6px; display: block; background: transparent; }
"""

_anywidget_cls = None


def _anywidget_class():
    """The ``anywidget.AnyWidget`` subclass, built on first use."""
    global _anywidget_cls
    if _anywidget_cls is not None:
        return _anywidget_cls
    import anywidget
    import traitlets

    class AxionFEMViewer(anywidget.AnyWidget):
        """anywidget view of an axionFEM scene (see axionfem.widget.SceneWidget)."""

        _esm = _ESM
        _css = _CSS
        page = traitlets.Unicode("").tag(sync=True)
        title = traitlets.Unicode("axionFEM").tag(sync=True)
        quantity = traitlets.Unicode("").tag(sync=True)
        quantities = traitlets.List(traitlets.Unicode()).tag(sync=True)
        scale = traitlets.Unicode("linear").tag(sync=True)
        height = traitlets.Int(DEFAULT_HEIGHT).tag(sync=True)

    _anywidget_cls = AxionFEMViewer
    return _anywidget_cls
