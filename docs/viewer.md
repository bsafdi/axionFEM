# The axionFEM viewer

The viewer displays a solved design (a *scene*, schema `axionfem.scene/1`, see
[scene_format.md](scene_format.md)) in two modes:

- **3D cutaway.** Region outlines are revolved about the z axis over 270° by
  default, leaving a 90° wedge open. The two cut half-planes show the material
  cross-sections and the field map, coloured by the selected quantity, with
  the poloidal field lines drawn on top. Superconductors are metallic grey,
  windings copper, iron dark grey and readout ports green. There is a z-axis
  arrow with ticks and a radial ruler in metres. Rotate, pan and zoom with
  the mouse or touch.
- **Section.** The meridional (r, z) half-plane mirrored to r < 0, drawn
  orthographically with axes in metres. Drag to pan and scroll to zoom.

Scenes of full 3D designs (`"symmetry": "3d"`) open in a third mode
(`createViewer3D`, chosen automatically): the region surfaces with the quadrant
facing the camera cut away, the field on the two exposed planes, superconductor
cross-sections in grey, optional colouring of the conductor surfaces by the
surface |B|, a colour bar, a region legend with visibility toggles, and the
results summary in the title. API: `setQuantity`, `setScale`, `setColormap`,
`setCutOpen`, `setSurfaceField`, `toggleRegion`, `resetView`, `screenshot`,
`getState`, `dispose` (these setters ignore unknown names and do not chain).
3D scenes have no section mode.

The axisymmetric modes have a colour bar with units, a quantity selector, linear or log
scale, a colormap choice, layer toggles (field map, field lines, surfaces, cut
planes, axes, every region), and a probe. Hovering, or clicking/tapping to pin,
shows r, z (and φ in 3D), the region, and the field values interpolated on the
field triangulation. A collapsible results panel shows the scene `summary`:
Ψ (SI and natural units), L_p, the figure of merit F = Ψ⁴/L_p², W_mag, g, peak
|B| (overall, per winding, on superconductor surfaces), forces, and, for each
benchmark, F/F_req, the amplitude scale, the cost at the present size and the
cost-optimal size. Values are shown with units and 4 significant figures.
The layout follows the system light/dark theme and works at phone width.

Everything is in `viewer/`:

| file | purpose |
|---|---|
| `viewer/src/viewer.js` | ES module: `createViewer(container, scene, options)` and helpers |
| `viewer/template.html` | HTML shell for single-file export and the live server |
| `viewer/vendor/` | three.js r185 (MIT), pinned and checksummed; see its README |
| `python/axionfem/viewer.py` | `export_html`, `render_html`, `serve`, CLI (standard library only) |
| `python/axionfem/widget.py` | `SceneWidget`: the same page in a Jupyter cell (anywidget, or an iframe) |

## Single-file HTML

```python
from axionfem.viewer import export_html
export_html("runs/core/scene.json", "core.html")                # path or scene dict
export_html(scene_dict, "core.html", title="CORE, H/R = 1",
            options={"mode": "section", "quantity": "Bz"})      # default view options
res.export_html("core.html")                                    # from a solved Results object
```

```bash
python -m axionfem.viewer export runs/core/scene.json core.html [--mode section] [--quantity Bz] [--scale log]
python python/axionfem/viewer.py export runs/core/scene.json core.html   # without installing
```

The page is self-contained. three.js, the viewer and the scene are inlined
(about 1.2 MB plus the scene), it makes no network requests, and it works from
`file://`. `render_html(scene, title, options=...)` returns the HTML as a
string, which is useful for embedding or publishing.

## In a Jupyter notebook

```python
res = d.solve()
w = res.show()                       # the viewer in the cell
w.quantity = "pickup_Bmag"           # switch what is plotted
w.scale = "log"; w.height = 700
w.quantities                         # what this scene offers
res.show(mode="section", quantity="Bphi")   # a second widget
af.show("runs/core/scene.json")       # a saved scene
```

`Results.show()` and `Results3D.show()` return a
`axionfem.widget.SceneWidget`, which displays the *same* self-contained page
`export_html` writes, inside an `<iframe srcdoc=...>`. Nothing is fetched from
the network and no Jupyter server extension or `nbextension install` step is
needed: the viewer travels in the output of the cell.

| back end | when | what you get |
|---|---|---|
| **anywidget** | the `notebook` extra (anywidget) | `w.quantity`/`w.scale`/`w.height` are traitlets synced both ways, so setting one recolours the *live* viewer without re-sending the page, and picking a quantity in the widget updates the Python object |
| **fallback** | always available, no extra packages | `_repr_html_` returns the iframe plus a plain `<select>`; the selector and the viewer's own panel work in the browser. Setting `w.quantity` from Python redraws the cell in place when IPython is there |

`AXF_WIDGET_BACKEND=html` forces the fallback; `SceneWidget(..., backend=...)`
picks one explicitly. `w.page()` returns the HTML, `w.export_html(path)` writes
it. Both back ends talk to `window.axfViewer`, the API object the exported page
publishes, through a same-origin `srcdoc` iframe — the page needs an import map,
which only works in a document, so it cannot be loaded as a bare ES module.

Notebook-specific limitations (the rest are under
[Limitations](#limitations)):

- each widget carries its own copy of three.js (about 1.2 MB) plus the scene,
  because the page is self-contained. **Clear the widget outputs before committing a
  notebook**; `examples/quickstart.ipynb` is stored with static images in place of the widget;
- in an *untrusted* notebook JupyterLab strips `<iframe>` and `<script>` from
  HTML output. Re-run the cell, or trust the notebook;
- the front end needs import maps and `data:` module URLs: JupyterLab 3.1+,
  Notebook 7, VS Code 1.80+ or Colab;
- 3D scenes have no section mode, so `w.mode = "section"` raises there.

## Live viewer on a remote machine

```python
from axionfem.viewer import serve
serve("runs")                     # a directory (searched up to 4 levels for scene.json / *.scene.json)
serve(["runs/core/scene.json", "runs/toroid"], port=8765)
srv = serve("runs", block=False)  # background thread; srv.url, srv.stop()
```

```bash
python -m axionfem.viewer serve runs --port 8765
# from your laptop:
ssh -L 8765:localhost:8765 <user>@<login-node>     # then open http://localhost:8765/
```

The index page lists the watched scenes and updates as new runs appear. Each
viewer page polls the scene's modification time and size (default every
1 s) and replaces the scene in place when the file changes. The camera, the
quantity and the layer settings are kept, and the camera is refitted only if
the device size changes. A file that is still being written is answered with
HTTP 503 and retried at the next poll. NaN/Infinity tokens, which are not
JSON, are served as `null`. The first scene is embedded in the page, so the
first frame needs no extra request. The server listens on 127.0.0.1 by default,
serves only the scenes it found and a fixed list of viewer files, and also
offers `/export/<id>.html` (a single-file download of that scene). The MCP
tool `start_viewer` runs the same server on the workspace's `runs/`.

## Embedding in a web page

```html
<script type="importmap">
{"imports": {
  "three": "./vendor/three.module.min.js",
  "three/addons/controls/OrbitControls.js": "./vendor/addons/controls/OrbitControls.js",
  "three/addons/environments/RoomEnvironment.js": "./vendor/addons/environments/RoomEnvironment.js"
}}
</script>
<div id="app" style="height: 600px"></div>
<script type="module">
  import { createViewer } from './src/viewer.js';
  const scene = await (await fetch('scene.json')).json();
  const viewer = createViewer(document.getElementById('app'), scene, { mode: '3d', panels: false });
</script>
```

The container must have a size, because the viewer fills it. Several viewers
can share a page. A tool that needs a single file should use
`render_html`/`export_html`. The exported page can also be published as is on
any static host that allows inline scripts and `data:` module URLs (see
Limitations).

### JavaScript API

`createViewer(container, scene, options)` returns:

| method | effect |
|---|---|
| `setQuantity(name)` | colour by `Bmag`, `Br`, `Bz`, `Bphi`, `psi`, `pickup_Bmag`, `pickup_psi` (any quantity present in the scene) |
| `setScale('linear' \| 'log')` | colour scale |
| `setMode('3d' \| 'section')` | view mode |
| `setCutaway(deg)` | opening angle of the removed wedge (0 = full revolution; default 90, so 270° is revolved) |
| `toggleLayer(name, visible?)` | a region or curve name, or `field`, `lines:static`, `lines:pickup`, `surfaces`, `cuts`, `axes` (`region:<name>` also works); omit `visible` to toggle |
| `screenshot()` | PNG data URL of the current view with axes, colour bar and title |
| `dispose()` | remove the viewer and free GPU memory |
| `setColormap(name)` | `auto`, `viridis`, `inferno`, `magma`, `plasma`, `cividis`, `coolwarm` |
| `setRange(min, max)` | fixed colour range in the quantity's units; `setRange(null)` for automatic |
| `setScene(scene)` | swap in a new scene, keeping the view settings |
| `setTransparent(bool)`, `setFieldBeyondDevice(bool)`, `setTheme('auto'\|'dark'\|'light')` | display options |
| `probe(r, z)` | `{region, inField, values}` at a point, in metres (no DOM needed) |
| `getState()`, `layers()`, `resetView()`, `setStatus(text, kind)`, `render()` | state, layer list, camera reset, header badge, forced redraw |

Setters return the viewer, so calls chain. Invalid names raise errors that list
the valid ones. The module also exports `prepareScene`, `makeSampler` (a point
sampler that needs no DOM or WebGL), `decodeArray`, `optionsFromURL`,
`fmtNum`, `fmtSI` and `COLORMAPS`.

### Options

| option | default | meaning |
|---|---|---|
| `mode` | `'3d'` | `'3d'` or `'section'` |
| `quantity` | first non-zero of `Bmag`, `pickup_Bmag`, … | initial quantity |
| `scale` | `'linear'` | `'linear'` or `'log'` |
| `colormap` | `'auto'` | see `setColormap` |
| `cutaway` | `90` | removed wedge in degrees (`azimuth: 270` is the same, as the revolved range) |
| `azimuthStart` | `0` | azimuth (deg) where the revolved surfaces start |
| `transparent` | `false` | semi-transparent revolved surfaces |
| `fieldBeyondDevice` | `false` | 3D: show the whole field window on the cut planes, not only the device envelope |
| `theme` | `'auto'` | follows `prefers-color-scheme`; or `'dark'`, `'light'` |
| `ui` | `true` | `false`: canvas and colour bar only |
| `panels`, `results` | `true` | open the control and results panels at start (closed below 760 px width) |
| `hide`, `show` | `[]` | layer names to hide or show at start |
| `logDecades` | `4` | log-scale floor: max × 10^-decades (or the smallest positive value, if larger) |
| `range` | `null` | fixed colour range `[min, max]` |
| `elevation` | `24` | 3D camera elevation (deg) |
| `probeAt` | `null` | `[x, y]` pixel position of a pinned probe at start |
| `renderer` | `'auto'` | `'canvas'` forces the Canvas2D section renderer |
| `preserveDrawingBuffer` | `true` | keeps the WebGL frame for external capture (headless screenshots); `false` saves a copy per frame |
| `maxPixelRatio`, `segments`, `title` | `2`, `160`, design name | resolution cap, azimuthal segments per turn, header title |

### URL parameters

These work in exported pages and in live pages, and override the embedded
options: `mode=3d|section`, `q=<quantity>`, `scale=linear|log`,
`cmap=<colormap>`, `cut=<deg>`, `az=<revolved deg>`, `theme=dark|light`,
`ui=0`, `panels=0`, `results=0`, `hide=a,b`, `show=a,b`, `transparent=1`,
`beyond=1`, `decades=<n>`, `elev=<deg>`, `probe=<x>,<y>`,
`renderer=canvas`.

Example: `core.html?mode=section&q=pickup_Bmag&scale=log&theme=dark`.

## What the colours and lines mean

- `Bmag`, `Br`, `Bz`, `Bphi`: the static laboratory field B_lab in T.
- `psi`: ψ = r A_φ of B_lab in Wb/rad (2πψ is the flux through the circle of
  radius r).
- `pickup_Bmag`, `pickup_psi`: the field and flux function of a unit readout
  current, per ampere (T/A, Wb/(rad A)).
- Field lines are contours of ψ, which are the exact poloidal field lines. They
  are shown for the family of the selected quantity: static lines for static
  quantities, pickup lines for pickup quantities. Both can be switched on
  together. A purely toroidal static field (B_φ only, as in the toroid)
  has no poloidal lines, and the layer list says so.
- Superconductor interiors are not in the field mesh; the field there is zero,
  and the probe says so.
- Colormaps (all exact 256-entry tables): `viridis` for static |B|, `plasma`
  for pickup |B_I|, `cividis` for flux functions, and the diverging `coolwarm`
  for signed components, symmetric about zero. viridis, inferno, magma, plasma
  and cividis are the CC0 matplotlib maps; coolwarm is Moreland's map. Linear
  scales start at zero for magnitudes. Log scales show |value| between a floor
  and the maximum, and the floor is printed on the colour bar. Units get an SI
  prefix when that reads better (e.g. nT/A).
- Field lines switch between a light and a dark tone depending on the
  brightness of the colormap underneath, so they stay visible everywhere.

## Design notes

- **One module, no build step.** `viewer.js` is plain ES2020 that imports
  three.js through bare specifiers. The HTML shell resolves them with an
  import map. For single-file export, each module becomes a base64 `data:`
  URL in the import map. That is static, so the whole module graph loads and
  the first frame renders before the page's `load` event, which is what
  headless screenshot tools wait for. It also needs no HTML escaping of
  code. `three.module.min.js` imports `./three.core.min.js`, which cannot be
  resolved from a `data:` URL, so the exporter rewrites that one specifier to
  `axf-three-core`. The vendored files are never modified.
- **Geometry.** Outlines are cleaned (duplicates removed), simplified with
  Douglas–Peucker at 1.5×10⁻⁴ of the device size, and oriented so that the
  material lies to the left (holes are found by nesting parity). Revolved
  surfaces use 160 segments per turn with smooth normals along fillets and
  hard edges at corners sharper than 35°. Cut faces reuse the region
  triangulations. The field map, region faces, outlines and lines sit on each
  cut plane at small offsets toward the open wedge, with polygon offset on the
  field map, so they do not z-fight. In 3D the field map and lines are
  clipped exactly (GPU clip planes) to the device envelope plus 3% unless
  "field beyond the device" is on. In the section they are clipped to the
  scene's `view` window.
- **Exact colours.** Each field vertex carries the normalised value as a
  texture coordinate into a 256×1 sRGB colormap texture. The value is
  interpolated linearly across each triangle and then looked up, so every
  pixel has an exact colormap colour. Interpolating RGB vertex colours would
  produce colours that are not in the map.
- **Probe.** A uniform grid locates the triangle that contains (r, z). The
  probe interpolates every quantity with barycentric weights. Pinned probes
  and hover share the code; `makeSampler` exposes the same sampler without a
  DOM or WebGL. In 3D the pick is an
  analytic ray / half-plane intersection combined with a raycast against the
  revolved surfaces.
- **Rendering on demand.** Frames are drawn only when something changes. Axis
  labels and section axes are drawn on a 2D overlay canvas, so they are
  crisp, follow the theme, and are included in `screenshot()`.
- **Fallback.** Without WebGL the viewer switches to a Canvas2D renderer for
  the section, with a per-pixel software rasteriser for the field map, says
  so in a notice, and disables 3D. `?renderer=canvas` forces it.
- **Live server.** It uses `http.server` plus browser polling, which is simple
  and robust through SSH tunnels and proxies, and needs no file-watching
  dependency. Page URLs are relative, so the server also works behind a path
  prefix.

## Limitations

- The displayed field is the piecewise-linear interpolation of the vertex
  samples written by the exporter (`subdiv` sub-triangles per element), not the
  full high-order finite-element field. B is discontinuous across elements;
  a shared display vertex carries one element's value. Peak values in the
  results panel come from the solver and can exceed the colour-bar maximum.
- The probe near curved boundaries snaps to the nearest straight display
  triangle (within 10⁻³ in barycentric coordinates).
- Field lines are drawn only on the cut planes (and in the section), as
  1-pixel WebGL lines.
- 3D scenes show the field only on the two exposed slices, have no section
  mode, no probe and no field lines.
- Exported pages need import maps (Chrome/Edge 89+, Firefox 108+, Safari
  16.4+). A host whose content-security policy forbids inline scripts or
  `data:` script URLs will not run them; the page then shows an explanatory
  message.
- Headless Firefox has WebGL only with an X display. Use `xvfb-run` (below);
  without one the page falls back to the Canvas2D section.

## Verification

```bash
python -m pytest -q tests/test_viewer.py      # export, assets, checksums, live server
python -m pytest -q tests/test_widget.py      # the Jupyter widget, headless (both back ends)
AXIONFEM_BROWSER_TESTS=1 python -m pytest -q tests/test_viewer.py -k browser
python -m axionfem.viewer export runs/core/scene.json /tmp/core.html
xvfb-run -a -s "-screen 0 2000x1400x24" firefox --headless --no-remote --profile /tmp/ffprof \
    --screenshot /tmp/core.png --window-size=1400,900 "file:///tmp/core.html?theme=dark"
```

The browser tests (`AXIONFEM_BROWSER_TESTS=1`) need a headless Firefox and
render an exported page. A quick manual physics check: in the toroid template's
scene the probe gives B_φ = μ₀NI/(2πr) inside the winding, and the interior of
a closed superconducting shell (the `core` template) is field-free.

## Licensing

The viewer code is BSD-3-Clause, like axionFEM. three.js is MIT and vendored
with its license. The colormap tables are CC0 (matplotlib's viridis family)
or derived from Moreland's published coolwarm map. There are no other
dependencies.
