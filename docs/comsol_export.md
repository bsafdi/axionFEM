# Exporting axionFEM geometries to COMSOL Multiphysics

axionFEM can hand a design or a mesh to COMSOL Multiphysics so that results can be
cross-checked there. COMSOL is not needed on the machine that exports. There are three formats.
They are listed from the fewest COMSOL add-on modules needed to the most:

| format | what COMSOL gets | COMSOL modules needed | fidelity |
|---|---|---|---|
| `mphtxt` | the axionFEM **mesh** (straight or second-order simplices), with a full domain/boundary/edge/point partition and named selections | none (Mesh > Import is in the base package) | exactly the mesh that axionFEM solved on |
| `java` | a **Model.java** that rebuilds the geometry parametrically from COMSOL primitives, with named selections | none for the geometry (`comsolcompile` and `comsol batch` ship with COMSOL) | exact CAD for the primitive kinds. The approximations are listed below |
| `step` | the conformal OpenCASCADE solids built by `axf-mesher` (3D designs) | CAD Import Module (or Design Module / a LiveLink) | exact CAD, but region names come from a JSON side file |

By default the files contain geometry and named selections only: the region kinds (superconductor,
conductor, port) arrive as selections. With **`--physics fullwave`** (3D designs with a port pickup) the
Java model, or the mesh-import program of the `mphtxt` format, also contains the Electromagnetic
Waves physics, materials, lumped ports, the axion source, a Frequency study and the Z/V exports, so
that COMSOL reruns the axionFEM problem (section 4).

## Commands

```bash
# from a design (meshed first with `axf mesh` into <out>_mesh/), a mesh, or a run directory
axf export-comsol design.json        -o detector.mphtxt               # mesh + named selections
axf export-comsol runs/primary       -o detector.mphtxt --order 2     # mesh of an existing run (no re-meshing)
axf export-comsol design.json        -o DetectorModel.java --format java
axf export-comsol design3d.json      -o detector.step --format step
axf export-comsol design3d.json      -o DetectorModel.java --format java --physics fullwave   # + physics (section 4)
axf export-comsol runs/primary       -o detector.mphtxt --physics fullwave                    # + Import_detector.java with physics
axf export-comsol runs/primary/mesh3d.msh -o detector.mphtxt --feature-angle 40
```

`axf export-comsol` is handled by the Python console script (`axionfem._cli`), so it needs the
`axf` that `pip install` puts on `PATH`. The C++ binary on its own does not have this
command. The same functionality is available in Python:

```python
import axionfem as af
from axionfem import comsol

d = af.load("design.json")
comsol.export_comsol(d, "detector.mphtxt")                         # -> detector.mphtxt (+ .entities.json, Import_detector.java)
comsol.export_comsol(d, "DetectorModel.java", format="java")
comsol.export_comsol(d, "DetectorModel.java", format="java", physics="fullwave")
comsol.gmsh_to_mphtxt("runs/primary/mesh3d.msh", "detector.mphtxt", order=2)
```

New C++ entry points support this. They are also useful on their own:

- `axf mesh design.json -o DIR`, also available as `_core.mesh(doc, workdir)`, meshes without
  solving. It writes `DIR/mesh.msh` (axisymmetric) or `DIR/mesh3d.msh` (3D), plus the mesher
  input `*_in.json` and the report `*_info.json`, exactly as `axf solve` would.
- `axf resolve design.json`, also available as `_core.resolve(doc)`, prints the geometry with
  every expression evaluated: parameters, regions with their material kind and the resolved
  `shape`/`solid`, curves, and the mesh section.
- `axf-mesher` accepts `"step_output"`: it writes the fragmented solids as STEP and lists every
  solid's region, volume, centre of mass and bounding box under `info.step`. It also accepts
  `"geometry_only"`, which stops before meshing.

The export source can be one of the following:

- a `Design` or design dict,
- a design `.json`,
- an `axf-mesher` `.msh`,
- a run directory written by `axf solve -o DIR` or `axf mesh -o DIR`. It holds `mesh.msh` or
  `mesh3d.msh`, the `*_in.json`/`*_info.json` next to it, and for the Java format `design.json`.

## 1. COMSOL mesh file (`.mphtxt`)

### Format

The writer follows the COMSOL native text format, version 0.1, as specified in the
*COMSOL Multiphysics Programming Reference Manual*, chapter **"The COMSOL File Formats"**, in
these sections:

- "File Structure"
  (<https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.18.html>)
- "Mesh" (class version 4)
  (<https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.35.html>)
- "Selection" (version 0)
  (<https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.39.html>)
- "Mesh Element Numbering Conventions", in the meshing chapter
  (<https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_mesh.49.039.html>)

The file is laid out as follows:

```
0 1                         # file format: major 0, minor 1
1+S # number of tags        # tag strings: "mesh1", then "sel1".."selS"
<len> mesh1                 # every string is "<number of UTF-8 bytes> <string>"
...
1+S # number of types       # "3 obj" each (ignored by COMSOL)
0 0 1                       # object header
4 Mesh # class
4 # version
D # sdim                    # 2: axisymmetric (r, z) meridian plane; 3: 3D (metres)
N # number of mesh vertices
0 # lowest mesh vertex index
x y [z]                     # N lines, %.17g
T # number of element types # vtx, edg, tri, tet (2D: vtx, edg, tri); edg2/tri2/tet2 with --order 2
  <len> <type name>
  k # number of vertices per element
  n # number of elements
  n lines of k vertex indices
  n # number of geometric entity indices
  n lines: the entity of each element
0 0 1                       # then one Selection object per named selection
9 Selection # class
0 # version
<len> <label>
5 mesh1                     # the mesh this selection refers to
d # dimension               # 0 point, 1 edge, 2 face (3D boundary / 2D domain), 3 domain
m # number of entities
m lines: entity indices
```

Entity numbering follows the manual. Domains (`tet` in 3D, `tri` in 2D) are **1-based**.
Boundaries, edges and points are **0-based**. The COMSOL GUI shows every index plus one.

Element types and local node order follow "Mesh Element Numbering Conventions". Corner
vertices come first, then the mid-edge nodes:

- `edg2`: v0 v1 m01
- `tri2`: v0 v1 v2 m01 m02 m12
- `tet2`: v0 v1 v2 v3 m01 m02 m12 m03 m13 m23

Mid-edge nodes are mapped from gmsh's order. The writer checks that every gmsh mid node lies
near its edge midpoint and refuses the file otherwise. Triangles in 2D are written
counter-clockwise and tetrahedra with positive volume.

### Geometric entities (how the mesh becomes a COMSOL "geometry")

COMSOL builds the geometry of an imported mesh from the entity indices in the file. The writer
therefore derives a complete, consistent partition from the axionFEM mesh:

- **Domains.** Each domain is a face-connected set of cells that share one region attribute. A
  region that consists of several disconnected pieces becomes several domains, all in one
  selection.
- **Boundaries.** Facets become boundaries in three cases:
  - they lie on the outside of the mesh,
  - they lie between two different domains,
  - they carry a tag in the gmsh file (outer sphere or semicircle, axis, symmetry planes, PEC
    holes, zero-thickness sheets).

  The facets are grouped by (the two adjacent domains, the tag) and split into connected pieces.
  In 3D, a boundary that would have no bounding edge is cut in two along its longest extent.
  Examples are a sealed conductor wall surrounded by air and a closed interface. After the cut,
  every face has edges, as faces in COMSOL's own geometries do.
- **Edges and points.** In 3D, edges are the chains of mesh edges where the set of adjacent
  boundaries changes. Points are where chains end or branch. Closed loops get two points. In
  2D, boundaries are chains of mesh edges between points, built by the same rule.
- **`--feature-angle A`** (3D, optional) also splits boundaries where neighbouring facets meet
  more sharply than `A` degrees. This gives CAD-like faces: the walls of a box become six faces.

### Selections and the tag map

| selection label | dimension | what it holds | axionFEM origin |
|---|---|---|---|
| `<region>` (e.g. `pickup_sheath`, `module_0_gap`) | domain | every domain of that region | region attribute 2, 3, … in design order |
| `air` | domain | unclaimed space | attribute 1 |
| `infinity_shell` | domain | 3D open-boundary shell [R, 1.25 R] | attribute 898. In axionFEM it is a coordinate map to infinity. In COMSOL, use it as an **Infinite Element Domain** (spherical) |
| `axis` | boundary | r = 0 (axisymmetric) | 900 |
| `outer` | boundary | outer semicircle or sphere, or the custom `mesh.domain_solid` wall | 901 |
| `slit`, `midplane`, … | boundary | named symmetry planes (`mesh.symmetry`) | 910 + i; `type` pec/pmc is in the side file |
| `pec_body_<k>` and `walls_of_<regions>` | boundary | walls of conductor volumes removed by `mesh.remove_pec` | 1000 + k; the side file lists the regions of each body |
| `<curve>` | boundary (2D) | zero-thickness sheet | 1000 + i (2D curves) |
| `<region>_boundary` | boundary | every boundary adjacent to that region (interfaces included) | derived |
| `exterior_boundary` | boundary | every outside boundary | derived |

Tag 1000 + k is used both for 2D curves and for 3D PEC bodies, but never in the same mesh. Names
come from the gmsh `$PhysicalNames` and from the mesher input (`mesh*_in.json`).

Each export also writes the side file **`<file>.mphtxt.entities.json`**. It records every
domain with its index, attribute, region, volume or area, and a point inside it. It records
every boundary with its index, adjacent domains (0 = outside), tag, group name, the mesher
metadata (type pec/pmc, conductor regions) and a point on it. It also lists all selections. If a
selection did not come through, use the point to make a Ball Selection in COMSOL. The
**volumes per region** are an easy first comparison against COMSOL's Measure.

### Import in the COMSOL Desktop

1. Model Wizard: create a component, 3D for 3D meshes or **2D Axisymmetric** for axisymmetric
   ones (r, z in metres). Choose physics and add no geometry.
2. Mesh > Import. Source: *COMSOL Multiphysics file*, then the `.mphtxt` file, then **Import**.
   COMSOL builds the geometry from the mesh entities. The file's Selection objects are the
   format COMSOL itself writes with *Mesh > Export*. They should appear as named selections of
   the imported mesh; check the result in your version.
3. Assign materials and physics with the selections. Check the domain volumes against
   `entities.json`, and check the sign of Im Z, because conventions differ.

### Import in batch

Every mphtxt export also writes `Import_<name>.java`, a short program that creates the component
and imports the mesh with `mesh.create("imp1","Import")`, `set("filename", …)` and
`importData()`. It then saves an `.mph` file:

```bash
comsolcompile Import_detector.java
comsol batch -inputfile Import_detector.class     # -> detector.mph
```

The file name is written as an absolute path. Edit it if the mesh moves to another cluster.

## 2. COMSOL Java model (`.java`)

`write_java` emits plain `model.component("comp1").geom("geom1")…` calls, the same style as
COMSOL's own *File > Save as Java*.

**What it builds:**

- **Parameters.** The design `parameters` become COMSOL parameters with translated expressions:
  - `pow(a, b)` becomes `(a)^(b)`,
  - `e` becomes `exp(1)`,
  - `mu0` becomes `mu0_const`,
  - every unary minus, product, quotient, power and group is parenthesized; sums stay bare only at the
    top level, inside parentheses and as function arguments. COMSOL documents `^` above unary minus but
    not the associativity of `^`; with these parentheses neither matters, so the value is axionFEM's
    (standard math: `-2^2` is -4, `2^3^2` is 512). The tests evaluate the text under all four
    combinations (unary minus tighter or looser than `^`, `^` right- or left-associative);
  - the exporter evaluates as the core does (variadic `min`/`max`, IEEE intermediates, CODATA `mu0`); a
    parameter it cannot translate is written as the number the core resolved.

  Names that COMSOL reserves (`x`, `y`, `z`, `r`, `phi`, `t`, `e`, `pi`, `i`, `j`, `eps`, …)
  get the prefix `par_`, so that `y` becomes `par_y`. Geometry numbers that the design gives
  as expressions stay expressions, so changing a parameter in COMSOL rebuilds the geometry.
  Derived numbers stay expressions as well, for example the inner rectangle of a `frame`
  (`r0+t`) and the inner fillet `max(f-t,0)`. `R_domain` (and `z_domain` in 2D) sets the air
  domain.
- **2D axisymmetric (r, z).** The geometry is built from these features:
  - `Rectangle`, plus `Fillet` on its four corners,
  - `frame` as a difference of two filleted rectangles,
  - `Polygon` (table source),
  - `Circle`,
  - `Union` / `Difference`,
  - the domain as a half-disk (`Circle`, angle 180, rot −90).

  Polygons with fillets, and arcs among the curves, become a **BezierPolygon** with
  rational-quadratic arcs. Those arcs are exact circles, computed exactly as axf-mesher's
  `polygon_wire` does. Their coordinates are numbers, not expressions.
- **3D.** The primitives map to COMSOL features as follows:

  | axionFEM solid | COMSOL feature(s) |
  |---|---|
  | `box` | `Block` |
  | `cylinder` | `Cylinder`, cartesian axis, height = \|axis\| |
  | `cone` | `Cone` (`specifytop` radius) |
  | `sphere` | `Sphere` |
  | `torus` | `Torus`, cartesian axis |
  | `revolve` | a 2D profile on an `xz` work plane, then `Revolve` about the global +z axis (`axistype 3d`) from `angle1` to `angle2` in degrees, then `Rotate`/`Move` for a non-default axis or centre |
  | `extrude` | a `Polygon` on an `xy` work plane at `quickz = z0`, then `Extrude` by z1 − z0 |
  | `step` | `Import` (needs the CAD Import Module) |
  | `union` / `difference` / `intersection` | the matching Boolean features, with interior boundaries removed |
  | `rotate` / `translate` | `Rotate` / `Move` |

  The air domain is a sphere of radius `R_domain`, or `mesh.domain_solid`. When axionFEM would
  use the infinity shell, a second sphere of radius `R_shell = 1.25*R_domain` is added and
  `infinity_shell` is selected; use it for an Infinite Element Domain.
- **Selections ("later regions win").** Each region feature contributes to a cumulative
  selection, "<region> (as drawn)". After Form Union, a `DifferenceSelection` labelled with the
  region name removes everything drawn later (a chain of `UnionSelection`s, so the size stays
  linear in the number of regions). `air` is the domain minus all regions. For every region
  there is also an `AdjacentSelection` labelled `<region>_boundary`. `all_superconductors` and
  `all_conductors` group the regions by material kind.
- **Symmetry models (`mesh.symmetry`, e.g. a quarter model).**
  - Every object is intersected with a clipping block for the kept half-spaces.
  - Regions whose bounding box lies entirely on the discarded side are left out, and the
    header comment lists them, as axf-mesher drops them.
  - The symmetry planes become BoxSelections, `sym_<name>_<pec|pmc>`. The boundary conditions
    themselves are yours to set.
- Java methods are limited to 64 kB of bytecode. The calls are therefore split over
  `run1(model)`, `run2(model)`, … by estimated size, as COMSOL's own export does.

**Build and run:**

```bash
comsolcompile DetectorModel.java          # COMSOL's javac wrapper -> DetectorModel.class
comsol batch -inputfile DetectorModel.class   # runs main(): builds and saves <design name>.mph
```

You can also open the `.class` file in the COMSOL Desktop with *File > Open*.

**Approximations and things to check:**

- `mesh.remove_pec`: axionFEM removes conductor volumes and imposes PEC on their walls. The
  Java model keeps those volumes as ordinary domains. Exclude them from the physics and put
  PEC on `<region>_boundary`.
- When the design gives no `mesh.domain_radius`, the radius is estimated from bounding boxes:
  1.6 × device radius in 3D (axf-mesher's rule with the infinity shell) and 2.5 × in 2D. The
  header says so. Give `mesh.domain_radius`, or export from a run directory whose
  `mesh*_info.json` holds the radius that was used.
- **Revolve sense.** Angles are counted right-handed about +z from the +x half-plane, as in
  OpenCASCADE. Verify this once in the GUI for a non-symmetric angle range. A slit centred on
  0° is unaffected.
- Filleted polygons and arcs are numeric (BezierPolygon). COMSOL marks BezierPolygon as
  deprecated since 5.5 in favour of CompositeCurve, but it is still supported and gives exact
  rational arcs.
- The property names follow the COMSOL 6.3 Programming Reference Manual. The tests in this
  repository compile every generated file with `javac` against stand-ins of the API
  (`tests/comsol_stubs`), which checks the Java syntax and the call shapes. The files have not
  yet been run in a COMSOL installation: **the first `comsol batch` run is the real test.**

## 3. STEP (`.step`)

`axf export-comsol design3d.json -o x.step --format step`, or the same command on a 3D run
directory, runs `axf-mesher` up to and including the OpenCASCADE `fragment`. It writes the
conformal solids with gmsh's STEP writer, including the clipping by symmetry planes and the
infinity shell, before any PEC volume is removed. It does not mesh.

`x.step.solids.json` lists each solid with these fields:

- `volume_tag`,
- `region`,
- `attr`,
- `volume_m3`,
- `center_of_mass`,
- `bbox`.

STEP carries no region names, so use these fields to identify the solids after import. In
COMSOL use Geometry > Import > *3D CAD file* (CAD Import Module), then Form Union. Axisymmetric
designs have no STEP export. For those, use `java`, or convert the design with
`axionfem.convert.to_3d` first.

## 4. Physics (`--physics fullwave`)

`axf export-comsol design.json -o M.java --format java --physics fullwave` (Python:
`export_comsol(..., physics="fullwave")`, code in `python/axionfem/comsol_physics.py`) turns an axionFEM 3D
full-wave problem (`fullwave` section, `port` pickup) into a COMSOL Electromagnetic Waves, Frequency Domain
model. The default, `--physics none`, keeps the geometry-only output unchanged. Two routes:

- `--format java`: the parametric geometry of section 2, plus the physics attached to geometry selections.
- `--format mphtxt`: `Import_<name>.java` imports axionFEM's own mesh and attaches the same physics to
  Explicit selections, named like the mesh selections and filled from `<name>.mphtxt.entities.json`. Use this
  for complex 3D designs whose parametric Java geometry is approximate. A source with physics needs the design,
  so give a design, a design `.json` or a run directory that holds `design.json`. Unless `--feature-angle` is
  given, a physics export uses a feature angle of 40°, so that the walls of a box gap become separate faces.

The feature type names and property keys are copied from a 3D impedance study saved by COMSOL 6.4 (not
distributed). The calls that this
template does not contain are marked `UNVERIFIED` in the generated Java and are listed below. The RF Module is
needed (emw, Lumped Port).

### What is exported

| axionFEM | COMSOL |
|---|---|
| air, `vacuum`/`linear`/`iron` regions | emw domains, one `Common` material per axionFEM material: `relpermittivity` ε_r, `relpermeability` μ_r, `electricconductivity` σ + 2π·freq·ε0·ε_r·tanδ (axionFEM's ε'(1 + i tanδ) + iσ/ω) |
| `superconductor`, `conductor` (e.g. `copper`) regions, and the walls of `mesh.remove_pec` holes | left out of the emw interface; `PerfectElectricConductor` on their walls (`pec_walls`) |
| `fullwave.boundary: pec` | closed PEC sphere at `mesh.domain_radius` (COMSOL's default `pec1` on the exterior; the mphtxt route also leaves the infinity-shell domain out) |
| `fullwave.boundary: radiation` | `Scattering` boundary condition on the outer sphere: first order, **approximate** (axionFEM uses an exact exterior map). Use `pec` for like-for-like comparisons |
| `mesh.symmetry` planes | `PerfectElectricConductor` (type `pec`) or `PerfectMagneticConductor` (type `pmc`) |
| `pickup` port (gap region) | `LumpedPort` 1, driven in the "imp" run |
| `fullwave.ports[k]` | `LumpedPort` k+2, **inactive** (open circuit, as in axionFEM's Z matrix, where the other ports carry no current). Activate one to load it with its Z_ref |
| axion source | `ExternalCurrentDensity` `ecd1` on all emw domains, `Je = Je_scale * B0` (below) |
| `fullwave.frequencies` (or `f_min`/`f_max`/`points`) | `Frequency` study, `punit` MHz, `plist`; the solver sequence of the template (GMRES + multigrid) |
| port impedance, source voltage | EvaluationGroups `eg1` (`real/imag(comp1.emw.Zport_1)`) and `eg2` (`real/imag(comp1.emw.Vport_1)`), saved to `<Class>_impedance.txt` and `<Class>_voltage.txt` in the working directory. Rows are `freq_MHz re im` (a plain-text table); `axionfem.comsol_physics.read_eval_table` reads them |

`main()` builds the model and runs the two solves of the template one after the other on one solution:

1. **imp**: `ecd1` inactive, `lport1` `PortExcitation` on. This gives Z.
2. **volt**: `ecd1` active, `lport1` `PortExcitation` off. This gives V.

Then it saves `<design name>.mph`. `run()` alone builds the model without solving, for the GUI.

**Port face.** axionFEM feeds the port current through the gap *volume* (uniform-conductivity Laplace
profile). A COMSOL lumped port is a *boundary*. For an axis-aligned box gap with a `toward` direction, the
port is the face of the box that contains `toward`. It is normal to the gap's thinnest other axis and lies on
the side farther from the origin (the outer face of a detector). The face is found with a BoxSelection (Java
route) or from the boundary bounding boxes in `entities.json` (mphtxt route). Other gaps fall back to all of
their non-PEC walls (a warning; UNVERIFIED as a lumped port). The two feeds differ by a modelling error of
order (gap size / loop size). Expect it in L at the per-cent level for gaps that are not thin.

### Conventions: time dependence, source normalization, port loading

- **Time dependence.** axionFEM uses e^{−iωt} and COMSOL uses e^{+jωt}, so COMSOL's complex numbers are the
  complex conjugates of axionFEM's. axionFEM's Z = R − iωL becomes COMSOL's Zport = R + jωL. Im Z > 0 is
  inductive, as a COMSOL impedance table read with `read_eval_table` expects.
- **Source.** axionFEM's 3D source overlap is b = ∫ N·B0 dV over all non-PEC volumes outside the mapped shell.
  Here B0 is the laboratory field: `fullwave.source` when it is given, otherwise the static solve of the windings.
  The direct-source solve is A e = iω b, so the axion current density is J_a = κ B0. axionFEM reports
  the unit coefficient κ = 1 A m⁻² T⁻¹: the open-circuit voltage is V_a/κ = −iωΨ (`Psi_vector`). The export
  writes Je [A/m²] = `Je_scale` · B0 [T] with the parameter `Je_scale` = 1, so that COMSOL's open-circuit
  voltage is conj(−iωΨ), up to the port polarity. B0 is exported as follows:

  | B0 | Je |
  |---|---|
  | `fullwave.source` `uniform` | the constant vector `B_T` (exact) |
  | `fullwave.source` `toroidal_box` | B_φ = B0 (or B0 r0/ρ) inside the annular box via COMSOL `if()`, 0 outside (exact) |
  | `fullwave.source` `segments` (≤ 100 segments) | the closed-form finite-wire Biot–Savart sum, with axionFEM's own kernel `1e-7 I (a+b)/(ab(ab+u·v)) d×u` (exact) |
  | poloidal windings on `frame` regions (2D shape, or 3D `revolve` about the global z axis), no `fullwave.source` | the ideal toroidal field B_φ = μ0 N I f/(2πρ), with f = clamp(min of the four leg depths / thickness, 0, 1): 1 in the frame's hole, 0 outside, linear across the legs. This is exact in the hole of an axisymmetric toroid and approximate in the winding corners. It ignores superconductor screening, so compare the MQS Ψ first |
  | anything else (≥ 101 segments, `azimuthal_average_samples`, solenoids, `loop` windings, circuits) | **not exported**: `Je = 0`, a warning in the header and the summary, and the volt run is skipped |

  **Normalisation to |J| = 1 A/m².** To compare with COMSOL voltage runs normalised to |J| = 1 A/m², set
  `Je_scale = 1/B_ref` [A m⁻² T⁻¹], or divide COMSOL's V by B_ref. The export summary gives `B_ref_T`: |B_T| for a uniform source,
  and B_φ at the inner radius of the field box for the toroidal profile, which is the "J_φ = 1 A/m² at r_min"
  normalisation.
- **Loaded port.** COMSOL terminates a lumped port that is not driven with its reference impedance
  (Z_ref = 50 Ω by default). The volt run therefore reports the loaded voltage
  V_port = V_oc Z_ref/(Z_ref + Zport). axionFEM's V is the open-circuit (Thévenin) voltage. Convert with
  `axionfem.comsol_physics.open_circuit_voltage(V_port, Zport, 50)`, where Zport is taken from the imp run at
  the same frequency. At
  12 MHz in the example below the loading changes |V| by 18–22 %.
- **Polarity.** COMSOL chooses the direction of a uniform lumped port from the face. It is not tied to
  axionFEM's `toward`, so compare |V|. Z does not depend on the polarity.

### UNVERIFIED API items (not in the COMSOL 6.4 template)

| call | used for |
|---|---|
| `selection().named("geom1_<tag>")` / `named("sel<k>")` on physics features and materials | attaching the physics to geometry selections (Java route) and to component Explicit selections (mphtxt route) |
| `create("pmc1", "PerfectMagneticConductor", 2)` | PMC symmetry planes |
| `create("sctr1", "Scattering", 2)` | `boundary: radiation` |
| `feature("ecd1").set("Je", new String[]{...})` | an expression-valued Je (the template sets `new int[]{0, 0, 1}`) |
| `propertyGroup("def").addInput("frequency")` | materials with tanδ (the template calls `addInput` on a `LossTangentDF` group) |
| `selection("sel<k>").add(int[])` | mphtxt route, selections with more than 2000 entities |
| entity numbering after Mesh > Import: domains as in the file, boundaries = file index + 1 | mphtxt route; check the domain volumes against `entities.json` once |
| a lumped port on several non-coplanar faces | fallback for gaps that are not boxes |
| the default port numbering (lport1 → `Zport_1`), datasets created by `sol1.runAll()` | the evaluation groups use the default dataset |

The verified pieces come from the template:

- the `physics().create("emw", "ElectromagneticWaves", "geom1")` call;
- the `ExternalCurrentDensity`, `PerfectElectricConductor` and `LumpedPort` features, and `PortExcitation`
  on/off;
- `material().create(.., "Common")` with `relpermittivity`/`relpermeability`/`electricconductivity` as
  9-component arrays;
- the `Frequency` study with `punit`/`plist`, and the solver sequence;
- `EvaluationGroup` with `storetable onfile`/`filename` and `EvalGlobal` `setIndex("expr", ..)`;
- the `Explicit` component selection with `geom(dim)` and `set(ints)`;
- `evaluationGroup(..).run()` and `.save(file)`, which the `mph` Python package calls.

The geometry selections (`UnionSelection`, `AdjacentSelection`, `DifferenceSelection`, `BoxSelection`) follow
the Programming Reference Manual, as in section 2. **The generated physics Java has not been run in COMSOL**;
check it with `comsol batch` before relying on it.

### Not exported

- MQS (`mf`, Magnetic Fields): not implemented. The template has no `mf` interface to copy from, and the
  MQS numbers (L_eff, dV/dν) follow from the full-wave model at low frequency.
- 2D-axisymmetric designs and `azimuthal` (ideal-slit) pickups: there is no lumped-port counterpart. Convert
  with `axionfem.convert.to_3d(design, pickup="slit")`. Note that the slit port of a `to_3d` design is a
  revolve sector, so its port falls back to the multi-face selection.
- Sources without a closed form (see the table above), RF surface losses (`rf` loss data) and the
  infinity-shell coordinate map. Conductor walls are PEC in the export; when their material has `rf` data
  (e.g. the builtin `copper`, anomalous skin) the export prints a WARNING. COMSOL's
  ImpedanceBoundaryCondition with a conductivity is classical skin: it does not reproduce an `anomalous`
  R_s (copper RRR 100: low by 2.4 % at 1 MHz, 23 % at 100 MHz, 40 % at 1 GHz). For a like-for-like loss
  comparison use `copper_classical` in axionFEM, or give COMSOL a frequency-dependent surface impedance
  from `axionfem.cavity.surface_resistance`.
- Mesh controls. COMSOL meshes the Java route with its physics-controlled default, and the mphtxt route uses
  axionFEM's mesh (first order). Refine and check convergence in COMSOL before comparing beyond the
  per-cent level.

### Worked example: the two-port frame

`examples/comsol_physics/` contains the exported `Stage1TwoPort.java` of
`benchmarks/designs/stage1_twoport.json`, and `make_example.py`, which regenerates it, the mphtxt route
(`stage1_twoport.mphtxt` + `Import_stage1_twoport.java`) and axionFEM's reference numbers. The design is a
0.6 m copper frame read out across a box gap, inside a PEC sphere of radius 0.7 m. The source is the uniform
B0 = (1, 0.2, 0.3) T, and the frequencies are 1 and 12 MHz. The second gap is a port that stays open.

On the COMSOL machine:

```bash
comsolcompile Stage1TwoPort.java
comsol batch -inputfile Stage1TwoPort.class        # writes Stage1TwoPort_impedance.txt, Stage1TwoPort_voltage.txt
# mesh-import route (generate it with make_example.py OUTDIR, copy OUTDIR over and fix the absolute .mphtxt path):
comsolcompile Import_stage1_twoport.java && comsol batch -inputfile Import_stage1_twoport.class
python examples/comsol_physics/make_example.py OUT --compare DIR_WITH_THE_TXT_FILES
```

axionFEM's numbers are in `examples/comsol_physics/axionfem_reference.json`, generated by
`make_example.py --solve`. The mesh has 85 562 tetrahedra. Port 2 is open and the
COMSOL values are the conjugates of axionFEM's:

| f [MHz] | mesh order (DOFs) | expected COMSOL Zport_1 [Ω] | L [nH] | \|V_open\| [V] (Je_scale = 1) | expected \|Vport_1\| in the volt run, 50 Ω load [V] |
|---|---|---|---|---|---|
| 1 | 1 (108 964) | 0 + 2.8248i | 449.59 | 1.3266e-3 | 1.3245e-3 |
| 12 | 1 (108 964) | 0 + 35.202i | 466.89 | 0.10926 | 0.08934 |
| 1 | 2 (572 252) | 0 + 3.2097i | 510.84 | 1.2518e-3 | 1.2492e-3 |
| 12 | 2 (572 252) | 0 + 39.487i | 523.71 | 0.09560 | 0.07502 |

How to read the table:

- **axionFEM is not mesh-converged here.** Going from first to second order on the same mesh changes L by
  14 % and |V| by 6 %, so use the order-2 rows as the reference. This example checks the tooling, not
  accuracy.
- **Feeds differ.** On top of that, the face-versus-volume feed of the port differs. The gap measures
  8 × 4.5 × 10 cm³ in a 0.6 m frame. COMSOL should agree within roughly 10 % at a converged mesh, not
  better.
- **Compare Im Zport_1 and the open-circuit |V|.** Re Z = 0, because the model is lossless.
- **Use `--compare`.** `make_example.py OUT --compare DIR` prints the ratios and converts COMSOL's loaded
  voltage to open circuit.

## Limitations (all formats)

- Physics is exported only with `--physics fullwave`, for 3D designs with a `port` pickup (section 4 lists
  what is and is not exported). Without it, selections carry the region roles. There is no MQS
  (Magnetic Fields, `mf`) export and no 2D-axisymmetric physics export.
- The mesh export reads the gmsh MSH 2.2 files that axf-mesher writes (`mesh.msh`, `mesh3d.msh`,
  ASCII or binary). It does not read MFEM `.mesh` files such as `fields/mesh_rank_*.mesh`. Use
  the `.msh` of the run directory instead. `mesh.refinements` (uniform refinement inside MFEM)
  is not reflected, because the export uses the mesher output.
- The second-order export (`--order 2`) writes axionFEM's curved nodes. A mesh whose curved
  elements axionFEM had to straighten (warning `curved_elements_invalid`) is exported as gmsh
  wrote it.
- The tests do not load the exported files in a COMSOL installation.
  - The `.mphtxt` structure follows the manual's grammar exactly, and the tests parse it back
    and compare it with the source mesh.
  - The Java files are compiled against stand-ins.
  - Report any import error together with the COMSOL version.

## Tests

`tests/test_comsol_export.py` covers these checks:

- **Round trips on synthetic gmsh meshes, ASCII and binary, 2D and 3D.** The checks are:
  - vertex and element counts,
  - exact coordinates,
  - the volume or area of every region,
  - tagged boundary triangles,
  - contiguous entity numbering,
  - every edge is a chain between two points,
  - every boundary has edges,
  - positive orientation,
  - the closed-surface split,
  - the feature-angle split.
- Second-order node order in `edg2`/`tri2`/`tet2`.
- Expression translation, checked against axionFEM precedence.
- Exact-arc fillet weights.
- Generated Java (2D, 3D with symmetry, a 300-region model that must be split into methods, and
  the import program) compiled with `javac`.
- With the core and mesher built: `resolve` on a template, a Java export from a template, and an
  mphtxt export of a meshed 2D design.
- Physics (`--physics fullwave`):
  - the generated Java of a frame with a box gap, a second port, a lossy dielectric and a PMC plane compiles
    against the stubs, and every feature sits on the intended selection: emw and `ecd1` on all non-PEC
    domains, PEC on the conductor walls, PMC on the symmetry plane, `lport1` on the gap's outer face and
    `lport2` inactive;
  - the material conductivity carries tanδ;
  - the study, the solver and the exports are present;
  - the source normalization: Je evaluates to B_T for a uniform source. For segments it matches an
    independent Biot–Savart evaluation to 1e-12 and the long-wire limit μ0I/(2πρ). The toroidal profile
    gives μ0NI/(2πρ) φ̂ in the hole, 0 outside and half in the middle of a leg;
  - the open-circuit conversion of the loaded port voltage;
  - the mphtxt route on a synthetic mesh: the port is a single face, and the emw, PMC and PEC sets are right.
    The route compiles;
  - designs that cannot be expressed are rejected with a reason;
  - with the core and mesher built, both routes on `stage1_twoport.json` export and compile.
