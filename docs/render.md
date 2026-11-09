# PNG snapshots (`axf::render`)

`axf::render` turns a solved scene (`axionfem.scene/1`, see
[scene_format.md](scene_format.md)) into a PNG image without a GPU, OpenGL,
fonts or Python. It serves two readers:

- **Checking a design at a glance.** A single image shows the geometry, the
  field map with field lines, and the key results (Ψ, L_p, 𝓕/𝓕_req, peak
  fields, cost), so a person or an automated tool can check it from the
  picture alone.
- **Reports.** The output is clean, antialiased and deterministic, so the
  same scene and options always give the same bytes.

Source: `cpp/include/axionfem/render.hpp`, `cpp/src/render.cpp`. PNG encoding
uses the vendored `cpp/third_party/stb_image_write.h` (public domain/MIT, see
`cpp/third_party/README.md`).

## API

```cpp
#include "axionfem/render.hpp"
namespace R = axf::render;

nlohmann::json scene = model.scene();            // or json::parse(scene.json)
R::RenderOptions o;                              // defaults: section view of |B|
o.view = "3d";
R::write_png("design.png", scene, o);            // render + write

std::vector<unsigned char> png = R::render_png(scene, o);  // bytes only
R::Image img = R::render_image(scene, o);        // raw RGB + plot geometry
R::write_file("x.png", R::encode_png(img));

R::RenderOptions p = R::options_from_json({{"view", "3d"}, {"quantity", "Br"}});
nlohmann::json j = R::options_to_json(p);        // round-trips
```

Errors throw `axf::Error`, whose text says what was wrong, where (a JSON path
such as `/field/quantities` or `/render/view`) and how to fix it. Examples:
an unknown quantity lists the available ones; an unknown option key lists the
valid keys; corrupt typed arrays name the array.

`Image` holds `width`, `height`, `rgb` (row-major, top row first) and the plot
frame `plot_x0, plot_y0, plot_x1, plot_y1` in output pixels. For the section
view it also holds the data window `x_min, x_max, z_min, z_max` in metres, so a
pixel maps back to a position:
`x = x_min + (px − plot_x0)/(plot_x1 − plot_x0)·(x_max − x_min)`, and z the same
way, counted downwards from `z_max`.

## Options

| option | default | meaning |
|---|---|---|
| `view` | `"section"` | axisymmetric scenes: `"section"`, a meridional cut mirrored about the axis, or `"3d"`, a cutaway body of revolution. 3D scenes are always drawn in perspective (see "3D scenes" below) |
| `quantity` | `"Bmag"` | a key of `scene.field.quantities` (`Bmag`, `Br`, `Bz`, `Bphi`, `psi`, `pickup_Bmag`, `pickup_psi`), for 3D scenes of `scene.slices[0].quantities` (`Bmag`, `Bx`, `By`, `Bz`, `pickup_Bmag`, ...), or `"none"` (materials only). A scene without a field falls back to `"none"` |
| `log_scale` | `false` | log10 colour scale. Signed quantities are shown as log10\|q\|. The automatic range covers at most 4 decades below the maximum |
| `width`, `height` | 1600, 1000 | output size in pixels (320–8192 by 200–8192). Layout and text scale with the size |
| `field_lines` | `true` | draw `scene.field_lines.static` (white on viridis, dark on the diverging map) and `.pickup` (pink, or green on the diverging map) |
| `annotate` | `true` | results panel at the right (coupling, peak fields, benchmarks, regions, field-line key) |
| `cutaway_deg` | 270 | 3d view of an axisymmetric scene: azimuthal extent of the drawn body. The open wedge (360 − cutaway) is centred on the camera. 360 draws the closed body, with no cut planes and no field |
| `elev_deg`, `azim_deg` | 25, −60 | 3d: camera elevation above the z = const plane, and azimuth of the camera in the x–y plane (matplotlib convention) |
| `theme` | `"light"` | `"light"` or `"dark"` |
| `vmin`, `vmax` | NaN (auto) | fixed colour range in the quantity's units, for comparing designs. With `log_scale` both must be > 0. Values outside the range are clamped |
| `supersample` | 2 | render at this factor and box-filter down (1–4) |
| `title` | `""` | replaces the automatic title |

## What is drawn

### Section view

- The (r, z) half-plane and its mirror image at r < 0. This is a full
  meridional cut through the axis, labelled `x (m)` with x = ±r. z points up.
  Metres per pixel are equal in x and z.
- The window is `scene.view` (from `extent` × 1.3 when missing).
- **Field map.** Each field triangle interpolates the quantity
  barycentrically, then maps the value through the colormap:
  - sequential quantities (|B|, pickup |B|) use **viridis**;
  - signed quantities (B_r, B_z, B_φ, ψ, ψ_p) use **coolwarm**, a diverging
    map centred at 0 with a symmetric range;
  - both are exact 256-entry tables from matplotlib, interpolated linearly.

  A NaN value leaves the pixel showing the material or the background. A
  quantity that is identically constant is drawn at the map centre, and the
  colour bar reads `≡ value`.

  The mirrored half shows the same cylindrical components (B_r, B_φ), not
  Cartesian ones.
- **Materials.** Region triangulations are filled with the region colour and a
  diagonal hatch, underneath the field. They therefore show where the scene
  has no field: superconductor interiors.
- **Outlines and curves.** Region outlines are crisp, antialiased lines in
  the region colour with a dark casing. Zero-thickness sheets (`curves`) are
  drawn in the superconductor colour.
- **Field lines.** Thin antialiased lines are drawn from the contour segments
  of ψ = rA_φ, for the static field and for the pickup mode.
- **Plot furniture.**
  - the symmetry axis as a dash-dot line;
  - a frame with inward ticks, and tick labels in metres on a shared nice
    step;
  - a colour bar with tick labels and units; small or large ranges are
    factored as `(10⁻⁷ T/A)`;
  - a title (design · quantity · view) and a subtitle with the device
    extent.

### 3d view

- **Revolved surfaces.** Region outlines are revolved over the cutaway range,
  with outlines simplified to about 0.3 px first. Shading is Gouraud: ambient
  plus two-sided Lambert plus a faint specular term.
- **Edges.** Crease circles are drawn as faint feature lines.
- **Cut half-planes.** The two cut planes show:
  - the field map, clipped to the device extent box
    `[0, r_max] × [z_min, z_max]`;
  - superconductor cross-sections in their material colour;
  - region outlines;
  - field lines.

  In `"none"` mode every region cross-section is filled and no field lines
  are drawn, since they would float over empty space.
- **Axis.** The z axis is drawn as a dash-dot line.
- **Rendering.** The camera is a perspective camera at 3.6× the bounding
  radius, fitted to the plot area. Triangles go through a z-buffer with
  perspective-correct interpolation. Lines are depth-tested against it.
- The colour bar and results panel are the same as in the section view.

### 3D scenes

A scene of the 3D engine (`"symmetry": "3d"`, see
[scene_format.md](scene_format.md#3d-scenes-symmetry-3d)) is drawn in
perspective from `elev_deg`/`azim_deg`: the region surfaces with the scene's
cutaway quadrant removed, and the two slices coloured by `quantity`.
Triangles inside superconductors, where the field is NaN, are drawn in the
material colour. The camera, z-buffer, colour bar and results panel work as in
the 3d view above. `Results3D.png()` calls this path.

### Results panel

The panel is built from `scene.summary` and skips any entry that is missing:

- **Coupling.** Ψ (natural T m³ and SI H T m²), L_p, 𝓕 = Ψ⁴/L_p² (natural
  T⁴ m¹⁰), W_mag, g, and the pickup kind.
- **Peak fields as solved.** |B|_max with its region, and the
  superconductor-surface peak for each superconductor.
- **One block per benchmark.** Each block shows 𝓕_req and 𝓕/𝓕_req. It then
  gives, scaled to 𝓕 = 𝓕_req: |B|_max, the superconductor-surface peak,
  W_mag, and the cost metric in MCHF together with its value at the
  cost-optimal size (a scaling metric for comparing designs, not a
  construction estimate).
- **Regions.** A colour swatch, the name and the role/material of each
  region.
- **Field-line key.**

Numbers are formatted to 3 significant figures (4 for L_p) with SI prefixes
or ×10ⁿ. The panel is for reading. Agents that need exact values should read
`summary` from the scene JSON.

## Rendering details

- Everything is drawn at `supersample`× resolution into a float RGB canvas,
  then box-filtered down. Lines are round-capped segments with analytic
  coverage. They are accumulated into a max-coverage mask and composited once
  per stroke group, so overlapping segments and joints are not
  double-blended.
- **Text.** The embedded font has 5×7 glyphs (with descenders), drawn at
  integer scales:
  - it covers printable ASCII, Greek Ψ ψ φ Φ μ Δ η κ, and ° ± × · ≤ ≥ ≈ ≡ − – — … → ∈ ∞;
  - markup `_{…}` and `^{…}` gives sub/superscripts, and Unicode super- and
    subscript digits work too;
  - at scale ≥ 3 the glyphs are smoothed on their diagonals.

  Text scale follows the geometric-mean image size: 2 at 1600×1000, 3 for the
  title. The layout unit follows the limiting dimension.
- **Determinism.** The renderer is single-threaded and has no randomness.
  Rendering twice gives identical PNG bytes; this is tested.
- **Encoding.** PNG encoding uses stb_image_write at zlib level 6.

## Limitations

- **Bitmap font.** The text is a pixel font, not a typeset one. It is legible
  at the default sizes, but for a publication figure set `annotate=false` and
  add labels in your own typesetting if a uniform font is required.
- **3d field extent.** The 3d view shows the field only on the cut planes and
  only inside the device extent box. The section view shows the full window,
  including the fringe field outside the device.
- **Thin parts.** Thin bands of revolution from zero-thickness `curves` are
  not drawn as surfaces in 3d, only their traces on the cut planes.
  Superconductor regions thinner than a pixel show up as outlines only.
- **Field lines.** Field lines are the scene's contour segments; their density
  follows ψ levels, not |B|. Marching-triangle segments are drawn as given,
  with no smoothing.
- **3D scenes.** A 3D scene shows the field only on its two slices, and the
  `view`, `cutaway_deg` and `field_lines` options do not apply to it (the
  cutaway is the scene's own).

## Tests

`cpp/tests/test_render.cpp` (target `test_render`) runs three groups of checks.

1. **Synthetic scene.** The scene has a known field, a NaN hole and a
   superconducting block. The checks cover:
   - orientation (z up, mirror symmetry, axis centred);
   - equal aspect ratio;
   - viridis and coolwarm colour values;
   - NaN handling, so the material shows through;
   - clamping with `vmin`/`vmax`;
   - determinism;
   - a camera/option sweep.
2. **Error messages.** It checks the message for an unknown quantity, an
   unknown option and corrupt indices.
3. **Sample scenes.** If `runs/{toroid,solenoid,core}/scene.json` exist in
   the source tree (for example after `axf solve ... -o runs/toroid --scene`), it
   renders section and 3d views (plus variants) to
   `render_<name>_<variant>.png` in the working directory. Each PNG is checked
   for its signature, IHDR size and IEND, and for non-blank pixels. Missing
   scenes are skipped.

```bash
cmake --build build --target test_render
(cd /tmp && /path/to/build/test_render)   # writes render_*.png in the working directory
```
