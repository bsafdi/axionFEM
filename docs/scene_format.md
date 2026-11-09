# Scene format `axionfem.scene/1`

A scene is the single, open description of a solved design that every display
consumes: the three.js viewer, single-file HTML exports, PNG snapshots and
notebooks. It is produced by `AxiModel::scene()` and `Model3D::scene()` (C++), `axf solve --scene`
(CLI) and `Results.scene()` / `Results3D.scene()` (Python).

All lengths are metres. Fields are SI (tesla). Large arrays are typed arrays
encoded as base64 of little-endian binary:

```json
{"dtype": "float32" | "uint32", "shape": [rows, cols], "data": "<base64>"}
```

## Top level

| key | meaning |
|---|---|
| `schema` | `"axionfem.scene/1"` |
| `name` | design name |
| `symmetry` | `"axisymmetric"`, or `"3d"` (see "3D scenes" below) |
| `units` | `"m"` |
| `extent` | device bounding box in the meridional half-plane: `r_max`, `z_min`, `z_max` |
| `view` | suggested field window: `r_max`, `z_min`, `z_max` |
| `regions` | list of material regions (below) |
| `curves` | zero-thickness superconducting sheets: `name`, `role`, `color`, `segments` (float32 [2n, 2], consecutive pairs are line segments) |
| `field` | sampled fields on a triangulation of the meridional plane (below), optional |
| `field_lines` | `static`, `pickup`: float32 [2n, 2] segment endpoints; contours of the flux function ψ = r A_φ, i.e. exact poloidal field lines of the static field and of the pickup mode; optional `static_label` (string): legend text of the `static` lines (cavity mode scenes, `Results.mode_scene`, put the mode's E or H lines there) |
| `summary` | key results: `coupling` (Ψ, L_p, 𝓕, W, g), `fields` (peaks), `benchmarks` (𝓕_req, scaling, cost per benchmark), `geometry`, `forces`, `pickup`; cavity mode scenes: `cavity_mode` (`label`, `f_Hz`, `C`, `G_ohm`, `Q0`, `V_m3`) |
| `design` | the design document that produced the scene |

## Regions

```json
{"name": "cavity", "role": "superconductor", "material": "nb", "color": "#7c8ea6",
 "triangles": {"vertices": float32 [n, 2] (r, z), "indices": uint32 [m, 3]},
 "outline": [float32 [k, 2] polyline, ...]}
```

`role` is one of `winding`, `superconductor`, `iron`, `conductor`, `port`,
`magnetic`, `vacuum`. Each outline is a closed boundary loop, made of
consecutive curved mesh edges sampled at `outline_samples` points each.
Adjacent edges share endpoints, so consecutive duplicate points can occur. The
first loop is the outer boundary.

## Field

```json
{"vertices": float32 [n, 2], "indices": uint32 [m, 3],
 "quantities": {"Bmag", "Br", "Bz", "Bphi", "psi", "pickup_Bmag", "pickup_psi": float32 [n, 1]},
 "ranges": {quantity: [min, max]}, "units": {quantity: unit}}
```

- `Bmag`, `Br`, `Bz` and `Bphi` describe the static laboratory field in T.
- `psi` is r A_φ of the static field in Wb/rad. Its contours are field lines.
- `pickup_Bmag` is the pickup-mode field per ampere of readout current in T/A.
- `pickup_psi` is the pickup-mode flux function per ampere.
- With a full-wave sweep (`fullwave`), `rf_Bmag` [T/A] and `rf_Emag` [V/(m A)] are the
  amplitudes of the time-harmonic readout field per ampere at `field.rf_frequency_Hz`
  (`fullwave.scene_frequency`, default the first sweep frequency). At low frequency
  `rf_Bmag` equals `pickup_Bmag`; `rf_Emag` shows where the readout's electric field
  (and thus dielectric loss and capacitive coupling) concentrates.
- Cavity eigenmode scenes (`Results.mode_scene`, `Results.plot_mode`) carry, on the cavity
  elements only, `Emag`, `Er`, `Ez`, `Ephi` [m^-3/2] and `Hmag`, `Hr`, `Hz`, `Hphi`
  [m^-3/2 / ohm] of one mode normalized to ∫ε_r|E|² dV = 1 (real amplitudes: complex fields E_c = iE, H_c = H),
  on a lattice of points per element (points on element edges repeat).
- Superconductor interiors are excluded, since the field there is zero.

## Rendering an axisymmetric scene in 3D

The meridional data describe a body of revolution about the z axis:

- **Surfaces.** Revolve each region outline (x = r cos φ, y = r sin φ, z) over
  an azimuthal range. three.js `LatheGeometry` takes points (r, z) directly.
- **Cutaway.** Revolving over φ ∈ [φ0, φ0 + 270°] leaves a 90° wedge open.
- **Cut faces.** Draw the region triangulations on the two cut half-planes at
  φ0 and φ0 + 270° (rotate the (r, z) triangulation into each plane).
- **Field maps.** Draw the field triangulation on the same planes, colored by
  a chosen quantity, with field lines drawn just in front of it.

Also useful is a flat 2D view: the (r, z) half-plane mirrored to r < 0.

## 3D scenes (`"symmetry": "3d"`)

Produced by `Model3D::scene()` / `Results3D.scene()` for designs solved by the
3D engine. Coordinates are (x, y, z) in metres.

| key | meaning |
|---|---|
| `extent3` | `{"min": [x, y, z], "max": [x, y, z]}`: bounding box of the region surfaces |
| `regions[]` | `name`, `role`, `material`, `surface`: `{"vertices": float32 [n, 3], "indices": uint32 [m, 3], "Bmag": float32 [n, 1]}`; triangles are the region's boundary faces, oriented outward; `Bmag` is the static \|B\| at each vertex evaluated on the outer side (the vacuum side of superconductors), averaged over adjacent faces |
| `cutaway` | `{"center": [xc, yc]}`: the quadrant x > xc, y < yc is removed for display |
| `slices[]` | exact intersections of the tetrahedral mesh with the two exposed half-planes (y = yc for x ≥ xc, and x = xc for y ≤ yc), clipped to the device box: `plane` (`normal`, `offset`), `vertices` float32 [n, 3], `indices` uint32 [m, 3], `region_attr` uint32 [m, 1], `quantities` (`Bmag`, `Bx`, `By`, `Bz`, `pickup_Bmag` per vertex; with a 3D full-wave sweep also `rf_Emag` [V/(m A)] and `rf_Bmag` [T/A] at `rf_frequency_Hz`, the sweep frequency nearest to `fullwave.scene_frequency`), `ranges`, `units`. Inside superconductors (B = 0) the quantities are NaN: draw those triangles in the material colour |
| `attr_names` | mesh attribute → region name |
| `summary`, `design` | as for axisymmetric scenes |

Rendering: draw the surfaces with the cutaway quadrant clipped (three.js: two
clipping planes with `clipIntersection = true`), and the slices coloured by a
quantity. The viewer (`createViewer` dispatches to `createViewer3D`) and the PNG
renderer (`axf render`, `Results3D.png()`) do exactly this.
