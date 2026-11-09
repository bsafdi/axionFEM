# Importing COMSOL models (`.mph` → axionFEM design)

!!! warning "Beta"
    The COMSOL importer (`axionfem.comsol_import` and its helper modules) is beta. The layout of COMSOL's model
    tree (`dmodel.xml`) is not documented by COMSOL; it was established on real files and may change between
    COMSOL versions. Read the import report of every model.

`axf import-comsol` reads a COMSOL Multiphysics model file and writes an axionFEM design that runs with
`axf solve` (or `Design.solve`) without further editing. No COMSOL installation or licence is needed: the
archive is read directly. The reverse direction (axionFEM → COMSOL) is [comsol_export.md](comsol_export.md).

```bash
axf import-comsol Small_Shield.mph -o toroid/design.json            # mesh route (default)
axf import-comsol Small_Shield.mph -o toroid/design.json --frequencies 5e6,20e6,60e6
axf import-comsol Small_Shield.mph -o toroid_cad/design.json --replay-route   # editable geometry
mpirun -n 8 axf solve toroid/design.json -o runs/toroid               # AXF_FW_SOLVER=strumpack for lossless models
```

```python
import axionfem as af
d = af.import_comsol("Small_Shield.mph", "toroid/design.json")   # -> Design; d.import_report
print(d.import_report.to_markdown())
```

The command writes, next to the design:

- `import_report.md` / `import_report.json`: every model item (parameters, functions, materials, physics
  features, studies, mesh, geometry) with its status **mapped**, **approximated**, **unsupported** or
  **ignored**, and a summary (domain volumes, boundary types, frequencies, port geometry, source);
- mesh route: `<stem>.msh` (gmsh MSH 2.2, second-order tetrahedra) and `<stem>_mesh_info.json`;
- replay route: `<stem>_replay.step` (+ `.solids.json`), the geometry check.

If an item that changes the result cannot be mapped (a coaxial port, a second physics interface, a current
density that is not uniform or azimuthal, a thick transition layer, ...), the import **fails** and says what; with
`--allow-partial` (`allow_partial=True`) the design is written anyway and the report lists the blocking items.

## Two routes

| | mesh route (default) | replay route (`--replay-route`) |
|---|---|---|
| geometry | COMSOL's stored mesh (`mesh*.mphbin`), exactly the tetrahedra COMSOL solved on, with curved boundaries restored (below) | the geometry sequence rebuilt from axionFEM solids, meshed by axionFEM |
| editable | no (regions are `{"imported": ...}`) | yes: revolve/box/cylinder solids with the COMSOL dimensions |
| zero-thickness PEC sheets and face ports | exact (interior PEC boundary groups, surface lumped port) | thin regions (`sheet_thickness`, default 1e-4 × model size) and a volume (gap) port: an approximation |
| needs | a mesh saved in the file (Mesh > Build All before saving) | the supported geometry features (below) and a stored mesh (for the selection mapping) |
| checked against COMSOL | yes: reproduces COMSOL's tables to 1e-5 on the solenoid (below) | geometry (volumes, exact); sweeps approximate (below) |

Use the mesh route to reproduce a COMSOL result; use the replay route to start an axionFEM design from a COMSOL
model (change dimensions, refine, optimize).

## What is mapped

| COMSOL | axionFEM |
|---|---|
| global parameters (expressions with units) | evaluated (SI) wherever they are used; listed in the report |
| Interpolation (table, linear/constant/nearest extrapolation) and Analytic functions | evaluated (e.g. a frequency-dependent ε′) |
| geometry (3D) | mesh route: stored mesh; replay route: see below |
| materials: relative permittivity, loss tangent (`LossTangent*` groups), dielectric loss, refractive index, permeability, conductivity (isotropic, constant or functions of `freq`) | `materials`: `linear` with `eps_r`, `tan_delta`, `mu_r`, `sigma_S_m`; vacuum when all are trivial. A frequency-dependent property uses its median over the design frequencies (spread in the report) |
| Electromagnetic Waves, Frequency Domain (emw): interface selection | one region per emw domain; domains outside the interface are PEC holes (as COMSOL's default PEC on its exterior) |
| Wave Equation, Electric (displacement field model, user-defined ε/μ/σ) | material of each domain |
| Perfect Electric Conductor (default exterior and explicit, also on interior zero-thickness boundaries) | PEC boundary groups (mesh route), PEC thin regions (replay route) |
| Perfect Magnetic Conductor | PMC boundary groups (mesh route) |
| Lumped Port, type Uniform | the readout port: `pickup.port_boundary` (mesh route; uniform surface current between the two PEC edges, as COMSOL's Uniform port) or a gap region (replay route) |
| External Current Density J_e (uniform vector, or azimuthal about z with uniform or 1/ρ magnitude), also when disabled | `fullwave.source` (`uniform` / `toroidal_box`) with B0 [T] = J_e [A/m²] and `fullwave.source_regions` = its domains |
| Frequency Domain / Adaptive Frequency Sweep (`plist`, `punit`, `range()`) | `fullwave.frequencies` (subsampled to `--max-frequencies`, default 20; or `--frequencies`) |
| Eigenfrequency (number of modes) | `fullwave.modes` (closed PEC domain) |
| emw shape order (`order_electricfield`) | `mesh.order` |
| Impedance Boundary Condition (material from a boundary-level material or user-defined ε_r, μ_r, σ) | boundary group `impedance` (Leontovich: n × H = −E_t/Z_s, Z_s = sqrt(μ/ε_c), as COMSOL's IBC); exterior boundaries; also terminals of lumped ports |
| Transition Boundary Condition (thin layer, thickness d) | boundary group `sheet`: surface admittance Y_s = −(2i/η) tan(kd/2) with E_t continuous (**approximated**; refused when \|k d\| > 0.5, e.g. a metal layer thicker than the skin depth) |
| Scattering Boundary Condition (no incident field) | boundary group `scattering`: first-order absorbing, β = −iω/η; spherical/cylindrical waves with the curvature term of the E formulation (**approximated**: COMSOL's own curvature term is not verified); second order → first order (approximated) |
| Perfectly Matched Layer (Definitions, coordinate system `PML`: Cartesian, cylindrical, spherical; scaling factor, curvature parameter, typical wavelength) | `fullwave.pml` (stretched-coordinate layers, [2D full-wave formulation](physics/fw2d.md)); exact layout of COMSOL's stretch not reproduced (**approximated**: the solution outside the layer agrees to the layer's discretization error) |
| 2D and 2D-axisymmetric components (emw, m = 0 / k_z = 0) | `"symmetry": "planar"` / axisymmetric design on the imported 2D mesh, solved by the 2D engine (below) |

Not mapped (import fails unless `--allow-partial`): coaxial/user-defined/numeric ports, more than one lumped port,
surface/edge currents, dipoles, periodic conditions, scattering boundaries with an incident field, thick transition
layers, infinite-element domains, anisotropic materials (these stop even with `--allow-partial`), J_e expressions
other than the two above or in a non-global coordinate system, magnetic-loss and scattered-field formulations,
unparsed selections, physics other than emw (Magnetic Fields `mf`, Electrostatics `es`, ...: reported; they would
need the replay route and the MQS/electrostatic solvers), 2D azimuthal mode numbers m ≠ 0 and out-of-plane wave
numbers k_z ≠ 0, 2D lumped ports on the out-of-plane field, eigenfrequency studies with PML, scattering or
transition boundaries (complex modes). Ignored (reported): mesh-generation settings (the stored mesh is used),
solver settings, initial values, result nodes and tables.

## Conventions

- **Time dependence.** COMSOL uses e^{+jωt}, axionFEM e^{−iωt}: axionFEM's Z is the complex conjugate of COMSOL's
  `Zport`. An inductive port has Im Zport > 0 in COMSOL and `Z_im_ohm` < 0 in axionFEM.
- **Source.** J_e [A/m²] = κ B0 with κ = 1 A m⁻² T⁻¹, as in the export (comsol_export.md §4). axionFEM's
  open-circuit voltage is V_oc = −iωΨ; compare |V_oc| = ω|Ψ|.
- **Loaded port.** COMSOL's `Vport` of a non-driven lumped port is loaded by Zref:
  V_port = V_oc Zref/(Zref + Zport). Convert with `axionfem.comsol_physics.open_circuit_voltage`. axionFEM's Z is
  the open-port impedance (Zref is recorded in the report, not used).
- **Polarity.** The port direction (`pickup.toward`) is the direction between the two PEC edges of the port,
  oriented with its largest Cartesian component positive. COMSOL picks its own; compare |V|. In 2D the port is an
  edge chain: the current runs from the chain end with the smaller to the one with the larger projection on
  `toward`, whose first two components are the in-plane coordinates ((x, y) planar, (r, z) axisymmetric); without
  `toward`, along the chain's dominant in-plane coordinate, positive. Z does not depend on the polarity; Ψ and V_oc
  flip sign with it.
- **Numbering.** COMSOL domains are 1-based; boundaries in the GUI are 1-based (the file stores them 0-based).
  Mesh route attributes: region k (domain order) → element attribute 2 + k; COMSOL boundary b → boundary attribute
  10000 + b; lumped port k → 9000 + k. The `.msh` has physical names `domain_<n>`, `comsol_boundary_<b>`, `lport1`.

## Mesh route details

1. **Mesh.** `mesh*.mphbin` (the one named by the active mesh sequence) is decoded: vertices, tetrahedra with
   their domain, boundary triangles with their boundary number. Only tetrahedral meshes are supported.
2. **Curved boundaries.** The stored mesh has corner vertices only; COMSOL maps boundary elements onto the CAD
   surfaces. The importer fits, per COMSOL boundary, a plane or a unique quadric (cylinders, spheres, cones) to the
   boundary's vertices and moves the mid-edge nodes of its edges onto it (edges on two faces onto their
   intersection), writing tet10/tri6 elements. When the general quadric is not unique (e.g. a band meshed with
   vertices on its rims only), a cylinder about a coordinate axis is tried first, then the ruled surface (cylinder or
   cone) through the coaxial rings, then a sphere (>= 3 rings); anything else stays straight. The report gives the
   minimum det J of the curved elements (≥ 0.7 of the straight one on the test files). On the 2211.00008 models the domain volumes then match the exact CAD volumes to ≤ 6e-6 (straight
   elements: −0.24 % in the toroid's shield volume, −0.6 % in the solenoid's source bore). Surfaces that fit
   neither (tori, splines) stay straight; the report lists them. `--straight` turns this off.
3. **Boundary conditions.** Boundary groups in `<stem>_mesh_info.json` (`{"attr", "name", "type": "pec"|"pmc"|"port"}`):
   every exterior boundary of the emw domains is PEC unless a PMC feature or the port claims it; interior
   boundaries listed by a PEC feature are PEC on both sides (zero-thickness sheets); other interior boundaries are
   continuous.
4. **Port.** `pickup: {"type": "port", "port_boundary": "lport1", "toward": [...]}`. The solver finds the two
   chains of port-face edges that lie on PEC, takes the width w as their mean length, and drives the uniform
   surface current K = (I/w) ê across the faces; Z = V/I with V = −(1/I)∫K·E dS (results: `fullwave.ports[0]`
   reports width, length, area/(w·length) and a warning if the port is not rectangular).

New design-format keys (design_format.md): region solid `{"imported": {...}}`, `pickup.port_boundary`,
`fullwave.source_regions`; together with the existing `mesh.file`/`mesh.info_file`.

## Replay route details

The geometry sequence is replayed symbolically. Supported features: WorkPlane (quick planes xy, yz, zx, yx, zy,
xz with an offset), Square, Rectangle (with layers), Circle (full), Polygon (solid), LineSegment, Revolve (2D axis
in the work plane or a 3D axis; segments become revolved sheets), Extrude (one distance), Block, Cylinder, Sphere,
Cone, Torus, Rotate, Move, Copy (one displacement), Union/Difference/Intersection, Form Union. Work-plane objects
are partitioned like COMSOL's "unite objects" (cells of the overlapping shapes). Anything else stops the replay
with the feature named (and the importer falls back to the mesh route).

Selections are mapped geometrically: each COMSOL tetrahedron centroid is located in the replayed cells; every
COMSOL domain must map onto one combination of cells (≥ 99.5 % of its elements), and combinations with no COMSOL
elements are the domains COMSOL left out of the physics (PEC holes). Typed COMSOL boundaries are matched to the
replayed zero-thickness sheets (all three triangle vertices on the sheet). Each sheet becomes a thin region on the
side where COMSOL meshed the emw domain. The design uses `mesh.remove_pec` and `mesh.domain_solid` (the union of the
replayed solids) with a closed PEC exterior, and mesh sizes from the COMSOL mesh (90th percentile and median edge).
The report compares every region volume with the COMSOL mesh.

## Validation (arXiv:2211.00008 models)

Records: `benchmarks/comsol_import/axionfem_results/` (`import_settings_check.json`, `toroid_lossless_import.json`,
`solenoid_lossless_import.json`, `epoxy_import.json`, `solenoid_comsol64_import.json`); scripts `check_settings.py` and
`compare_import.py`. Runs: order 2, STRUMPACK (lossless) or AMS-GMRES (epoxy), 8 MPI ranks. The COMSOL files and
tables are not distributed with axionFEM; `tests/test_comsol_import.py` repeats the settings checks when
`AXF_COMSOL_2211_DIR` names their directory (and `AXF_COMSOL64_SOLENOID_MPH` the COMSOL 6.4 re-save of the
solenoid's impedance model) and skips otherwise.

**Settings.** For the three 2211.00008 files the importer recovers the settings checked in
`import_settings_check.json`: emw domains, PEC sheet boundaries, port boundaries and
size (toroid 0.01° × 1 mm, solenoid 0.5 × 1 mm), Zref, J_e (toroidal 1 A/m² on the interior; axial 1 A/m² in the
bore), TACHYON ε′ = 3.049 at 650 MHz and tan δ = 0.0021, study frequency lists, and the exact CAD volumes of every
emw domain (curved mesh route ≤ 5e-6; replay route ≤ 1e-9).

**Results against the COMSOL tables** (axionFEM X = −Im Z_axf and |V_oc| = ω|Ψ|; COMSOL Im Zport and the
open-circuit voltage from the loaded Vport):

| model | route, mesh | X / X_COMSOL | \|V_oc\| / COMSOL | first pole (between sweep points) |
|---|---|---|---|---|
| shielded solenoid (Model_File.mph, tables 5–160 MHz) | mesh route, curved (COMSOL's 187 594 tets, 1.24 M DOFs) | 1.00000 ± 1e-4 at all 17 frequencies, poles included | 1.0000 ± 2e-4 (6e-3 at the 69.4 MHz zero of V) | 22.117 vs 22.119 MHz |
| | mesh route, straight | 1.004 (5 MHz), 0.84–1.2 next to the poles | 0.997 | 22.06 |
| | hand-built reproduction (benchmarks/beyond_mqs_2211, 794k tets) | 1.003 (5 MHz) | 1.014 | 22.40 |
| shielded lossless toroid (Small_Shield.mph, tables 5–500 MHz) | mesh route, curved (132 524 tets, 0.88 M DOFs) | 0.9998 (5 MHz), ≤ 1e-3 away from poles, ≤ 7e-3 within 0.5 MHz of a pole | 0.9999 (5 MHz) | 26.719 vs 26.724 MHz |
| | mesh route, straight | 0.9990 (5 MHz) | 0.9991 | 26.73 |
| | hand-built reproduction (479k tets) | 1.0022 | 1.0044 | 26.75 |
| lossy (TACHYON) toroid (Toroid_With_Epoxy, tables 5–650 MHz) | mesh route, curved (251 106 tets) | 1.012 (5 MHz); R/R_COMSOL 1.04 | 1.017 | 26.34 vs 26.41 MHz |
| the same shielded solenoid re-saved by COMSOL 6.4 (its own tables) | mesh route, curved | 1.00002–1.00003 | 0.9989 | — |

Reading the table:

- On the solenoid the import and COMSOL solve the same discrete problem: the agreement (1e-5) is at the level of
  the solver tolerances, also at the poles. The hand-built model's +1.3 % pole offset is therefore COMSOL's own
  mesh error.
- On the lossless toroid L and |V_oc| agree to 2e-4 at 5–20 MHz and the first pole to 2e-4 (26.719 vs 26.724 MHz);
  the 0.5–0.7 % differences right next to the poles follow from that small pole shift. The remaining 2e-4 is not
  the curving (cylinder versus sphere fits of the wall bands change results by 1e-4); the tables may come from a
  slightly different model state than the one saved (the Je-driven toroid run is not the saved state; the file
  also holds an older mesh, `mesh1.mphbin`, of an earlier geometry, and the importer warns when `--mesh-file`
  selects it).
- The epoxy tables are 1.1–1.2 % below both the lossless COMSOL toroid and this import of the saved epoxy model at
  5 MHz, where the dielectric cannot change L: they were computed on a different mesh/state than the one saved
  (the tables postdate the saved model). Resistances agree to 4 % away from poles.
- COMSOL 6.4's own |Vport| for the re-saved solenoid is 0.1 % above COMSOL 5.6's 2022 table on the same
  mesh; the import matches the 5.6 table to 1e-5 and 6.4 to 1e-3.
- Straight-sided elements cost 0.1–0.4 % in L and |V| here (the source bore of the solenoid loses 0.6 % of its
  volume): the curving matters at this level.

The replay route (editable geometry, thin regions for the zero-thickness bridge and port) solves axionFEM's own,
finer mesh of the same geometry. On the solenoid it reproduces the hand-built reproduction (poles 22.424 / 70.241 /
132.26 MHz vs 22.404 / 70.213 / 132.22; L(5 MHz) 273.13 vs 273.12 nH), i.e. it differs from COMSOL by COMSOL's mesh
error (+1.4 % in the first pole). On the toroid it is less close: first pole 26.615 MHz vs 26.749 (hand-built)
and 26.724 (COMSOL), L(5 MHz) +0.17 % and |V_oc| -1.3 % vs the hand-built reproduction (cause not investigated;
the thin bridge/port slabs are the likely suspect). Replay meshes can be large (3.5 M DOFs here; the STRUMPACK
path needs a large-memory node).

## 2D and 2D-axisymmetric models

An emw interface on a 2D geometry goes through `axionfem.comsol_import2d`: the stored mesh (triangles; quadrilaterals
are split in two) is written as second-order triangles with boundary mid-nodes on fitted lines/circles, one region
`{"shape": {"imported": ...}}` per emw domain, and the design runs the 2D imported-mesh engine (`cpp/src/fw2d.cpp`,
[formulation](physics/fw2d.md)):

```json
{"symmetry": "axisymmetric", "regions": [{"name": "domain_1", "shape": {"imported": {"comsol_domains": [1]}}}],
 "pickup": {"type": "port", "port_boundary": "lport1"},
 "fullwave": {"rf_only": true, "boundary": "pec", "frequencies": [3e8], "pml": [...], "source": {...}},
 "mesh": {"order": 2, "file": "design.msh", "info_file": "design_mesh_info.json"}}
```

- **Axisymmetric or planar**: from the component's spatial coordinates (`r, phi, z` in COMSOL axisymmetric
  components); `--axisymmetric` / `--planar` (`axisymmetric=True/False`) override. Planar models take COMSOL's
  out-of-plane thickness `d` (`fullwave.thickness_m`); every result is the 3D quantity (full revolution / thickness D).
- **Families**: the in-plane field (axisymmetric TM = (E_r, E_z), planar TE = (E_x, E_y)) carries lumped ports and
  frequency sweeps; the out-of-plane field (axisymmetric TE = E_φ, planar TM = E_z) has eigenmodes only. COMSOL's
  "electric field components solved for" selects the eigenmode families.
- **Axis**: exterior boundaries on r = 0 (COMSOL's Axial Symmetry) become `axis` groups (E_φ = 0; nothing on the
  in-plane E).
- **Lumped port** (Uniform; Coaxial in 2D axisymmetry): the edge chain between two PEC/impedance boundaries; the
  current I flows across it uniformly around the circumference, K = I/(2πr) t (planar K = I/D t); V = −∫ t·E dl
  ([formulation](physics/fw2d.md)). This is the Coaxial port's definition; for a Uniform port on an edge that is not
  at constant r COMSOL's definition is not verified (reported as approximated). User-defined 2D ports (field out of
  the plane) are not mapped. Some 2D-axisymmetric COMSOL models record Im Zport with the opposite sign to the 3D
  models; axionFEM's Z is −iωL for an inductive port in both, so compare |Im Z| and check the sign on the file
  before trusting a comparison.
- **Source**: J_e = (J_r, J_φ, J_z) in axisymmetric components: uniform J_z (`uniform`, in the port family) or J_φ
  uniform / ∝ 1/r (`toroidal_box`, out-of-plane: it does not couple to a 2D port, Ψ = 0 with a note); planar as in 3D.
- **Eigenfrequency**: closed lossless models; impedance walls are PEC in the eigenproblem and give the perturbative
  `Q_wall` (½ ∮ Re Z_s |H_t|²); lumped ports are open circuits; COMSOL's shift becomes `target_f_Hz`. Interior PEC
  boundaries (walls, septa) act in both families. The 3D importer drops an eigenfrequency study (reported) when the
  model has scattering or transition boundaries or PML.
- **Curved boundaries**: boundary mid-nodes move onto fitted lines/circles; a move that would distort a thin element
  (boundary layers: det J below 0.2 of the straight element at a node or the centroid) is undone, that edge stays
  straight and the report says how many (`curving.reverted_mid_nodes`, `min_detJ_over_straight`). An inverted
  written element is an error. Quadrilaterals are checked to be in COMSOL's tensor vertex order before splitting.
- **Checks in the engine**: every exterior edge needs a boundary group; impedance/scattering/PMC groups must be
  exterior and sheets interior; one port group; a φ-dependent source in axisymmetry is averaged over φ (warning).

`Design.solve()` returns `axionfem.fw2d.Fw2DResults`: `sweep()` (f, Z, L, Ψ, V_oc), `modes()`, `sample(r, z,
frequency_index=k | mode=(family, i))`, and per frequency the power by region, Robin group and PML with a power
balance check. `axionfem.fw2d.mapped_mesh` / `write_mesh` build such meshes without COMSOL (used by the tests).

## Impedance, transition, scattering boundaries and PML (2D and 3D)

The same boundary groups (`impedance`, `sheet`, `scattering`, with `material`, `thickness_m`, `wave`, `center`,
`axis`, or a fixed `Zs_ohm` [re, im] in e^{−iωt}) and `fullwave.pml` layers work in the 3D full-wave solver
(model3d_fullwave.cpp): the Robin term enters the operator, so these sweeps use AMS-GMRES (the STRUMPACK path is
for real, lossless operators). A PML entry may list several domains on different sides (a frame, a box shell, side
+ cap + corner): each point is stretched only in the coordinates where it lies beyond the physical domain on that
side. Results add `boundary_power_W_per_A2` per group, `R_boundary_ohm`, and
`power_balance_rel` (or `pml_absorbed_W_per_A2` with PML). 3D eigenmodes treat impedance walls as PEC (loss not
evaluated) and reject scattering/sheet/PML.

### What is validated against what

The real-file checks of these features were made on one COMSOL 6.3 file that is not distributed with axionFEM: a
2D-axisymmetric emw model of a coaxial pickup with copper Impedance Boundary Condition walls, a Coaxial lumped port, a
disabled PML node and a disabled source defined by table functions (the same model with the source enabled is
refused, because its J_e comes from interpolation tables stored in result tables). Its import was compared with
axionFEM's axisymmetric engine on a hand-built design of the same geometry and with a lossy transmission-line model;
X, R and the tuned Q agreed to within 4e-3 at 80-130 MHz. Because the file is not distributed, no test in this
repository repeats that comparison. It checked the 2D-axisymmetric mesh decoding and detection, the `Impedance`
feature (COMSOL's op name) with a boundary-level material, the Coaxial lumped port, the axis, the PML node layout
(`Coordsys op="PML"`, properties `p:ScalingType`; its parameters are defaults, not stored) and the parameter layout
(`<ModelParam><expressions name=… expr=…>`). The 2211.00008 3D models (and the COMSOL 6.4 re-save of one) have none of these features. So:

| path | validated against |
|---|---|
| 2D binary mesh decoding (Mesh v7, sdim 2, `edg`/`tri`) | the documented format + the v7 deviations established on 3D files; synthetic archives only |
| 2D-axisymmetric detection (`spatialCoord r, phi, z`), emw `components`, 2D mesh v7, `Impedance` feature with boundary material, Coaxial port, PML node and `p:ScalingType` | a real file (not distributed; above) |
| planar 2D detection and thickness `d`, PML parameters other than the scaling type (`PMLfactor`, `PMLgamma`/`PMLp`, `typicalWavelength`: defaults are not stored), enabled PML, Transition and Scattering features (`d`, `WaveType`, `IncidentField`, `r0`) | **assumed** names (COMSOL Java API); synthetic archives only. Check the import report on a real file first |
| 2D engine, all boundary types, PML, ports, sources, eigenmodes | closed forms ([2D formulation](physics/fw2d.md), `benchmarks/records/fw2d_validation.json`): coax, biconical line, parallel plates, wedge, pillbox, axisymmetric radial line, septum cavity: 1e-9 to 6e-4 (PML: falls with layer resolution; a two-sided layer entry 7e-6); thin sheet 1e-9; source overlap Ψ 5e-10; IBC attenuation 2.7e-4 (= O(δ/a)); perturbative vs driven IBC Q 8e-6; TE011 wall Q 2e-5 |
| 3D Robin groups and PML | closed forms (`benchmarks/records/fw3d_robin_validation.json`): parallel-plate line, scattering 2e-8, PEC 1e-8, PML 4e-4, IBC attenuation 1.6e-4 (O(δ/h)), power balance 2e-11, thin sheet on interior faces, the IBC case on 2 MPI ranks |
| end-to-end import → solve | synthetic 2D-axisymmetric coax `.mph` (IBC walls, PML or scattering end, J_e): Z within 3e-3 of Z0, wall loss vs closed form 2 %; synthetic 3D box with IBC + scattering walls: power balance (tests/test_comsol_import2d.py) |
| the 3D importer on the real files | unchanged (tests/test_comsol_import.py; the files are not distributed: those tests run when `AXF_COMSOL_2211_DIR` / `AXF_COMSOL64_SOLENOID_MPH` name their directories and skip otherwise) |

## Files and formats

The `.mph` archive is a zip. Its binary resources follow COMSOL's documented native binary format
(COMSOL Programming Reference Manual, "The COMSOL File Formats":
[file structure](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_fileformats.53.19.html),
[binary format](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.22.html),
[terminology](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_fileformats.53.21.html),
[Mesh](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.35.html),
[Selection](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.39.html),
[Geom3](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_fileformats.54.32.html)).
The meshes stored in `.mph` files are Mesh class version 7, which extends the documented version 4 (extra flags,
length-prefixed rows, element parameters, up/down data); the reader's docstring
(`python/axionfem/comsol_mphbin.py`) lists the differences, measured on files saved by COMSOL 5.6, 6.0 and 6.4.
The model tree `dmodel.xml` is **not documented**; its layout was established on the same files and may change
between COMSOL versions (see "Measured file-format details" below). Geometry binaries are decoded when they are
`Geom3` objects; CAD-kernel (`Cad3Part`) geometries are not, which is why the replay route works from the geometry
sequence instead.

### Measured file-format details

These were established on files saved by COMSOL 5.6, 6.0, 6.3 and 6.4 and are not part of COMSOL's documentation:

- **Mesh class version 7** (as stored in `.mph`) differs from the documented version 4: two Boolean flags after the
  space dimension and an integer `1` before the vertex count; every matrix is an outer count followed by
  length-prefixed rows (coordinates `[d][np]`, elements per local vertex `[nep][ne]`); after the elements a pair
  `(npar, nrows)` and, if `nrows > 0`, element parameters (for triangles in a layout that the reader skips by
  searching for the entity marker); entities as `(ne, ne, values)`; then up/down data, `0` or `2, (ne, ...)×2`.
- **Numbering.** Domain indices are 1-based, boundaries, edges and points 0-based (as documented); a boundary that
  the GUI calls 123 is entity 122 in the file.
- **Mesh vertices on curved CAD faces are not exactly on the surface** (up to about 2e-5 m on a 1 m radius in the
  test files); the curving fit tolerances allow for that.
- **`dmodel.xml`.** Nodes are tagged; parameters read `param value="N|1,'a'|1,'b'"`; `propertyValue value=` holds
  the expression and `valueMatrix` the evaluated form (sometimes the only one, then in the step's unit); a
  geometry feature's `key` string is its built state (names and values concatenated, parsed by the first occurrence
  of `name<number>`); explicit selections read `entities="11,1,6"` for domains (header ndom + 1) and
  `"-125,-1,123"` for lower dimensions (header `-N, -1`). `MeshNative/<mesh resource=...>` names the active mesh
  binary when a file holds several.
- **Studies.** A study's `plist` property is its current setting; its `key` is what was last computed. They can
  differ; the report shows both.
- **Global parameters.** COMSOL 6.3 stores them as `<ModelParam><expressions name="..." expr="..."/>`; the
  `ModelParamGroup` layouts are assumed.
- **Materials.** A `RelativePermittivity` that refers to a material with only a `LossTangentDF` group is
  evaluated from that group, as COMSOL does.

axionFEM's own COMSOL export writes Java and `.mphtxt`, not `.mph` (an `.mph` needs COMSOL to be written), so it
cannot produce test files for this importer; the tests build small synthetic `.mph` archives instead
(`tests/test_comsol_import.py`) and use the 2211.00008 files (and the COMSOL 6.4 re-save) when present.

## Limits

- The `dmodel.xml` layout is reverse-engineered; the 2D, Robin and PML parts of it are assumed (above).
- Expressions referring to model variables (`comp1.emw.*`, `withsol`, ...) cannot be evaluated outside COMSOL.
- Parametric sweeps, cases and multiple components are not mapped; the first emw interface and the first stored
  mesh of its geometry are used.
- Global parameters: read from `<ModelParam><expressions>` (COMSOL 6.3, checked on the real file above); the other group
  layouts are assumed (the report lists every parameter it found and its value).
