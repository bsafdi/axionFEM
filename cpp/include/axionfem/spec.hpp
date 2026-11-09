// Design specification: the single, typed description of a detector that
// agents and people edit.  JSON in, validated C++ structures out.
//
// All lengths are metres, currents amperes, fields tesla, temperatures kelvin.
// Any numeric field may be a JSON number or a string expression over
// "parameters" (see expr.hpp).
#pragma once

#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/expr.hpp"

namespace axf::spec {

using json = nlohmann::json;

struct Vec2 {
  double r = 0, z = 0;
};

// ---------------------------------------------------------------- geometry
// A 2D shape in the meridional (r, z) half-plane, r >= 0.
struct Shape {
  enum class Kind { Rect, Frame, Polygon, Disk, Union, Difference, Imported, Contour };
  // Contour: a closed loop from `start` through straight segments and axis-aligned elliptical arcs
  // (point(t) = center + (ar cos t, az sin t), t from t0 to t1 in radians); the mesher builds the arcs exactly
  // (rational quadratic B-splines), so curved walls (e.g. an SRF cell) are not faceted
  struct ContourSeg {
    bool ellipse = false;
    Vec2 to;                       // line: end point (ellipse: derived from t1)
    Vec2 center;                   // ellipse
    double ar = 0, az = 0, t0 = 0, t1 = 0;
  };
  Kind kind = Kind::Rect;
  json imported;                          // Imported: provenance (elements from mesh.file, 2D imported meshes)
  double r0 = 0, r1 = 0, z0 = 0, z1 = 0;  // Rect / Frame extents
  double fillet = 0;                      // corner radius (Frame: outer corners f, inner corners max(f - thickness, 0))
  double thickness = 0;                   // Frame wall thickness
  std::vector<Vec2> points;               // Polygon vertices (counter-clockwise or not)
  std::vector<double> fillets;            // Polygon: per-vertex corner radius (empty: `fillet` at every vertex)
  Vec2 center;                            // Disk
  double radius = 0;                      // Disk
  std::vector<Shape> children;            // Union / Difference (first minus the rest)
  Vec2 start;                             // Contour
  std::vector<ContourSeg> segments;       // Contour

  json resolved() const;  // numbers only, consumed by the mesher
  void bbox(double& rmin, double& rmax, double& zmin, double& zmax) const;
};

// A curve in the (r, z) half-plane: zero-thickness sheets.
struct Path {
  enum class Kind { Segment, Polyline, Rect, Arc };
  Kind kind = Kind::Segment;
  std::vector<Vec2> points;  // Segment (2) / Polyline
  bool closed = false;       // Polyline
  double r0 = 0, r1 = 0, z0 = 0, z1 = 0, fillet = 0;  // Rect outline
  Vec2 center;
  double radius = 0, angle0 = 0, angle1 = 0;  // Arc, angles in radians from +r towards +z

  json resolved() const;
  void bbox(double& rmin, double& rmax, double& zmin, double& zmax) const;
};

// A 3D solid (OpenCASCADE). Coordinates (x, y, z) in metres.
struct Vec3 {
  double x = 0, y = 0, z = 0;
};

struct Solid {
  enum class Kind { Box, Cylinder, Sphere, Torus, Cone, Revolve, Extrude, Step, Union, Difference, Intersection, WedgeFrame,
                    Imported };  // Imported: the region's elements come from mesh.file (no CAD; e.g. a COMSOL import)
  Kind kind = Kind::Box;
  Vec3 p0, p1;                  // Box: min/max corners; Cylinder/Cone: base point and axis vector (p1)
  double r1 = 0, r2 = 0;        // Cylinder radius r1; Cone radii r1 (base), r2 (top); Torus major r1, minor r2; Sphere r1
  Vec3 center, axis{0, 0, 1};   // Sphere/Torus centre, Torus axis
  Shape profile;                // Revolve: (r, z) shape revolved about `axis` through `center`
  double angle0 = 0, angle1 = 360;  // Revolve angular range in degrees (from the +x direction)
  std::vector<Vec2> polygon;    // Extrude: (x, y) polygon, extruded from z0 to z1
  double z0 = 0, z1 = 0;
  std::string file;             // Step: path; scale converts file units to metres
  double scale = 1.0;
  std::vector<Solid> children;  // booleans
  json wedge;                   // WedgeFrame: evaluated parameters (numbers only) for axf-mesher
  json imported;                // Imported: provenance (free-form, e.g. {"comsol_domains": [6]}), echoed as given
  // optional rigid transform applied after construction: rotate (about axis through origin) then translate
  bool has_rotation = false;
  Vec3 rot_axis{0, 0, 1}, rot_origin;
  double rot_deg = 0;
  Vec3 translate;

  json resolved() const;
};

// ---------------------------------------------------------------- materials
struct BHCurve {
  // Either tabulated (B, H) pairs or the simple "knee" model:
  // H = B/(mu0 mu_r) for B <= B_sat, then slope 1/mu0.
  std::string model = "knee";  // "knee" | "table"
  double mu_r = 1000.0, B_sat = 2.0;
  std::vector<double> B, H;
};

enum class MaterialKind {
  Vacuum,          // mu0
  Linear,          // constant mu_r
  Iron,            // nonlinear B-H; warm by default
  Superconductor,  // ideal Meissner: excludes static and RF fields
  Conductor,       // normal metal: transparent to static field, perfect conductor at RF (ideal)
};

// RF surface loss of a conductor surface (models: docs/design_format.md, physics: docs/physics/surface_resistance.md;
// the formulas live in surface_resistance.hpp so that every engine evaluates the same R_s(f)):
//   "power"     R_s = Rs_ohm (f / f_ref)^exponent
//   "skin"      classical skin effect R_s = sqrt(pi f mu0 rho)
//   "table"     measured R_s(f), log-log interpolation; no extrapolation (Rs() throws outside the table)
//   "anomalous" Reuter-Sondheimer anomalous skin effect (rho, rho*l, specularity p, optional v_F)
//   "bcs"       A/T (f/f_ref)^2 exp(-Delta/kT) [optional Mattis-Bardeen log factor] or a tabulated R_BCS(f, T), plus R_res
// plus an optional additive "flux_flow" term (vortex losses, full local London + vortex impedance with the
// orientation factor sin^2(J, B0); order-of-magnitude model) for bcs/power/table.
struct RFLoss {
  std::string model;      // "" none, "power", "skin", "table", "anomalous", "bcs"
  double Rs_ohm = 0, f_ref = 1e6, exponent = 0;
  double rho_ohm_m = 0;   // skin and anomalous: dc resistivity at T_K (anomalous: given, or rho_300K / RRR)
  // "table" (and "bcs" with a tabulated R_BCS(f, T), reduced to f at T_K at parse time): log-log interpolation;
  // no extrapolation (Rs() throws outside [f_Hz.front, f_Hz.back])
  std::vector<double> tab_f_Hz, tab_Rs_ohm;
  // "anomalous"
  double rho_l = 0;       // rho * mean free path [ohm m^2]
  double specularity = 0; // p: 0 diffuse, 1 specular
  double v_F = 0;         // Fermi velocity [m/s]; 0: quasi-static (omega tau -> 0)
  // "bcs": R_BCS = bcs_A / T (f / f_ref)^2 exp(-bcs_gap Tc / T) (analytic), or the table; + R_res
  double bcs_A = 0, bcs_gap = 0, bcs_Tc = 0, R_res = 0;
  bool bcs_table = false;
  bool bcs_log = false;   // include ln(2.246 kT/hf) / ln(2.246 kT/hf_ref) (analytic bcs)
  // additive vortex term Re sqrt(i mu0 w (i mu0 w lambda^2 + s rho_v)), rho_v = rho_n |B|/B_c2 (eps + i f/f0)/(1 + i f/f0),
  // s = orientation factor sin^2(angle(J_rf, B0)) (axf::rs::flux_flow)
  bool flux_flow = false;
  double ff_B = 0;          // |B| inside the wall [T]; NaN: the local static field (cavity wall losses only)
  double ff_Bc2 = 0, ff_rho_n = 0, ff_lambda = 0, ff_f0 = 0, ff_eps = 0;
  double ff_orientation = -1;  // fixed factor s in [0, 1]; < 0: "field" (per wall point from B0 and the RF
                               // current where the caller knows them; 1 = perpendicular, upper bound, elsewhere)
  bool ff_B_local() const { return flux_flow && std::isnan(ff_B); }
  std::string owner;      // material name (error messages)
  double T_K = 0;         // temperature of this surface (T/Q budget; also the BCS / anomalous temperature)
  // Magnetoresistance: resistivity factor m(B) >= 1 in the local static field B ("kohler": 1 + (B/B0)^n,
  // or a table). It scales rho (skin model: R_s ~ sqrt(m)) or R_s directly (power, table models); not allowed
  // with anomalous / bcs (spec error).
  std::string mr_model;   // "" none, "kohler", "table"
  double mr_B0 = 1, mr_n = 2;
  std::vector<double> mr_B, mr_factor;
  double mr(double B) const;
  double Rs(double f) const;           // Rs_base(f) + Rs_flux_flow(f)
  double Rs(double f, double B) const;  // with magnetoresistance in the static field B [T]
  double Rs_base(double f) const;      // R_s without the flux_flow term
  // The flux_flow part of R_s (0 without it) at the field |B| (NaN: ff_B) and orientation factor s (NaN: the fixed
  // ff_orientation, or 1 = current perpendicular to the vortices when it is "field"). Throws when the field is
  // "local" and no B is given.
  double Rs_flux_flow(double f, double B = std::numeric_limits<double>::quiet_NaN(),
                      double s = std::numeric_limits<double>::quiet_NaN()) const;
  // Validity / model warnings for an R_s evaluation at f (shared by every engine that reports wall losses):
  // (key, text) pairs; keys "flux_flow", "skin_anomalous", "bcs_log", "bcs_hf_kT". Engines report each key once.
  std::vector<std::pair<std::string, std::string>> warnings(double f) const;
  double Rs_factor(double B) const;     // Rs(f, B) / Rs(f): the magnetoresistance factor (any model)
};

// ---- mechanics / thermal material properties (see docs/physics/mechanics_thermal.md) ----
// Isotropic linear-elastic properties.  `alpha` is the TOTAL thermal contraction strain between
// room temperature and the operating temperature (dimensionless, positive for a material that
// shrinks on cooling): the stress-free strain applied by `mechanics.cooldown` is -alpha.
struct MechanicalProps {
  bool given = false;
  double E_Pa = 0;
  double nu = 0;
  double alpha = 0;
  std::string notes;
};

// Thermal conductivity: a constant, or a table k(T) interpolated piecewise linearly in T and
// clamped to the end values outside the tabulated range.
struct ThermalProps {
  bool given = false;
  double k_const = 0;                  // W/(m K), used when the table is empty
  std::vector<double> kT_T, kT_k;      // k(T) table (T strictly increasing)
  std::string notes;
};
// ---- end mechanics / thermal material properties ----

struct Material {
  std::string name;
  MaterialKind kind = MaterialKind::Vacuum;
  RFLoss rf;
  double mu_r = 1.0;
  double eps_r = 1.0;        // relative permittivity (full-wave problems)
  double tan_delta = 0.0;    // dielectric loss tangent (full-wave problems)
  double sigma_S_m = 0.0;    // bulk conductivity of a non-PEC medium (full-wave eddy currents)
  BHCurve bh;
  double density = 0.0;      // kg/m^3
  bool warm = false;         // excluded from the cold envelope
  double cost_per_kg = 0.0;  // CHF/kg (cost model), 0 = not costed
  MechanicalProps mech;      // structural properties
  ThermalProps therm;        // thermal properties
};

// ---------------------------------------------------------------- conductors
// A technical superconductor: its critical surface Jc(B, T), the fraction of the winding (or tape
// stack) cross-section that carries it, and the E-J power law used by the critical-state solver.
// Parameter meanings and the published default sets are in docs/design_format.md and
// docs/validation/hts.md; the laws themselves are in conductor.hpp.
struct Conductor {
  std::string name;
  std::string law = "nbti_bottura";  // nbti_bottura | nb3sn_durham | rebco_kim | table
  double T_op = 4.2;                 // operating temperature [K]
  double fill_factor = 1.0;          // superconductor area / conductor (winding) area
  // NbTi (Bottura) and Nb3Sn (Durham/ITER) critical surfaces
  double C0 = 0;                     // A T / m^2
  double alpha = 0, beta = 0, gamma = 0;            // NbTi exponents
  double p = 0, q = 0;                              // Nb3Sn pinning-force exponents
  double Bc20 = 0, Tc0 = 0;                         // T, K (at zero strain for Nb3Sn)
  double strain = 0, Ca1 = 0, Ca2 = 0, eps_0a = 0;  // Nb3Sn deviatoric strain function (fixed strain)
  // REBCO anisotropic Kim: Jc0 / [1 + sqrt((k B_par)^2 + B_perp^2)/B0]^b_exp times the T factor
  double Jc0 = 0, B0 = 0, b_exp = 0, k_anis = 1;
  double T_ref = 0, Tc = 0, m_T = 0;  // factor ((Tc - T)/(Tc - T_ref))^m_T (1 at T_ref)
  // table law: Jc[iB * nT + iT] [A/m^2] at tab_B[iB] [T] and tab_T[iT] [K]
  std::vector<double> tab_B, tab_T, tab_Jc;
  double n_value = 25, E0 = 1e-4;  // E-J power law E = E0 (|J|/Jc)^n (critical-state solver)
  double normal_r = 0, normal_z = 1;  // unit normal of the tape plane in the (r, z) half-plane
  double B_min = 0;    // Jc is evaluated at max(|B|, B_min) (laws that diverge as B -> 0)
  bool anisotropic = false;  // the law distinguishes the field normal to the tape from the in-plane one
};

// ---------------------------------------------------------------- items
struct Region {
  std::string name;
  Shape shape;              // axisymmetric designs
  Solid solid;              // 3D designs
  std::string material;
  double mesh_size = 0;     // 0 = automatic (element size inside the region)
  double surface_size = 0;  // 0 = automatic (element size at and near this region's own boundary)
};

struct Curve {
  std::string name;
  Path path;
  std::string material;     // a superconductor or conductor (sheet), or a linear material (thin layer)
  // > 0: a thin magnetic layer of this thickness (m) and the material's mu_r
  // (docs/physics/thin_magnetic_layer.md); 0: a sheet
  double thickness = 0;
  bool thin_layer() const { return thickness > 0; }
  double mesh_size = 0;
  double surface_size = 0;  // 0 = automatic (element size at and near this curve)
};

enum class WindingType { Azimuthal, Poloidal, Loop };  // Loop: 3D, harmonic current around the region's loop

struct Winding {
  std::string name;
  std::string region;
  WindingType type = WindingType::Azimuthal;
  double ampere_turns = 0;  // N*I [A] for a standalone powered winding; sign sets the field direction
  std::string circuit;      // non-empty: the winding belongs to this circuit
  double turns = 0;         // signed number of turns when part of a circuit
  std::string conductor;    // non-empty: report the critical-current margins of this conductor
  // 3D: azimuthal ("loop") windings circulate about this axis (right-handed)
  Vec3 axis_origin, axis_direction{0, 0, 1};
};

// A circuit connecting one or more windings in series.
//  "current":    powered, carries the prescribed current;
//  "persistent": closed superconducting loop; its total flux linkage is conserved
//                (prescribed value, default 0 = cooled with no field), the current follows.
struct Circuit {
  std::string name;
  enum class Kind { Current, Persistent } kind = Kind::Persistent;
  double current = 0;        // A (Current)
  double flux_linkage = 0;   // Wb-turns (Persistent)
};

struct Constraint {
  enum class Kind { NetCurrent, Flux };
  Kind kind = Kind::NetCurrent;
  double value = 0;  // A for NetCurrent, Wb for Flux (total flux through the loop)
};

struct Superconductor {
  std::string body;          // region or curve name
  Constraint azimuthal;      // loops around the symmetry axis (static solve)
  Constraint poloidal;       // loops around the body's own cross-section (static)
  bool azimuthal_slit = false;  // azimuthal loops are cut (e.g. toroid sheath slit)
  // 3D: symmetry axis of the body (azimuthal/poloidal loops are taken about it; default: the z axis)
  bool has_axis = false;
  Vec3 axis_origin, axis_direction{0, 0, 1};
};

struct Pickup {
  // Azimuthal / Poloidal: axisymmetric designs.  Port: 3D readout gap region whose two faces
  // touch conductors; the readout current flows across it (terminal order set by `toward`).
  enum class Kind { Azimuthal, Poloidal, Port };
  Vec3 toward{0, 0, 0};   // Port: approximate direction of the readout current (0 = automatic)
  Kind kind = Kind::Azimuthal;
  std::string conductor;  // Azimuthal: superconducting body carrying the readout current
  std::string port;       // Poloidal: region spanning the readout gap
  // Port (3D, imported meshes only): name of a mesh-info boundary group of type "port" whose faces carry a
  // uniform surface current between the two edges where the face meets PEC (a COMSOL Uniform lumped port).
  // Exactly one of `port` and `port_boundary` is set for a Port pickup.
  std::string port_boundary;
  std::optional<Vec2> inside_point;  // Poloidal: a point inside the pickup cavity (optional)
};

struct Benchmark {
  std::string name;
  double Q = 2e7;       // intrinsic quality factor
  double T = 0.01;      // loss-weighted temperature [K]
  double eta_A = 0.1;   // amplifier noise relative to the SQL
};

struct ScanSpec {
  double f_min = 120.9e3;     // Hz
  double f_max = 1.0e6;       // Hz
  double live_time = 3.0;     // years
  std::string model = "DFSZ"; // "DFSZ" (E/N = 8/3) | "KSVZ" (E/N = 0) | "custom"
  double E_over_N = 8.0 / 3.0;
  double C_QCD = 1.77;        // model-independent contribution (lattice, Brandt et al., arXiv:2603.29153)
  double rho_DM = 0.3;        // GeV/cm^3
  double Q_axion = 6.8e5;     // SHM
  double Theta95 = 2.71;      // expected 95% limit test statistic
  double ma_fa = 5.70e-3;     // GeV^2
};

struct CostSpec {
  // Default coefficients of the cost metric (cost.hpp): illustrative values; no source is claimed for them.
  std::string model = "herve_balbekov";
  double a = 0.25;            // MCHF
  double kappa = 1.2;
  double p_energy = 0.7, p_area = 0.8;
  double iron_cost_per_kg = 5.0e-6;  // MCHF/kg (5 CHF/kg)
  // Cold envelope: "bounding_cylinder" of all non-warm, non-vacuum parts.
  std::string cold_envelope = "bounding_cylinder";
  double B_peak_cap = 0;      // T, 0 = none (winding peak field)
  double sc_wall_cap = 0;     // T, 0 = none (superconductor surface)
  double load_line_cap = 0;   // fraction of Jc on a winding's conductor, 0 = none
};

struct MeshSpec {
  json domain_solid;         // optional explicit 3D closed PEC computational domain
  json symmetry = json::array(); // named zero-coordinate plane clips, PEC or natural PMC
  bool remove_pec = false;   // retain conductor identities on holes, not volume elements
  std::string file, info_file; // checksum-recorded existing mesh and boundary metadata
  int order = 2;
  int geometry_order = 0;    // 3D: order of the element geometry from the mesher (0: min(order, 2)); 1 skips the
                             // curved-element untangling, exact for polyhedral geometry (e.g. wedge_frame coils)
  double size = 0;           // characteristic element size in the device (0 = auto)
  double surface_size = 0;   // size at superconductor/iron surfaces (0 = auto)
  double domain_radius = 0;  // outer boundary radius (0 = auto)
  double domain_center_z = 0;  // z of the centre of the outer semicircle / infinity sphere
  int refinements = 0;       // uniform refinements after meshing
  // adaptive refinement: re-mesh with a size field from error indicators of the previous solution until
  // Psi, L_p and W_mag change by less than `tol` (relative) or `cycles` re-meshes were done
  struct Adapt {
    int cycles = 0;              // 0: off
    double tol = 1e-3;
    double reduction = 0.5;      // targeted reduction of the estimated error per cycle
    long max_elements = 2000000;
    std::vector<std::string> watch = {"Psi", "L_p", "W_mag"};  // quantities whose change stops the loop (+ "Gamma")
  } adapt;
};

// Full-wave analysis beyond magnetoquasistatics (axisymmetric designs).
struct FullWaveSpec {
  struct Port {
    std::string name;
    std::string region;
    Vec3 toward;
  };
  bool enabled = false;
  bool rf_only = false;     // prepare ports without a static/MQS solve (components and PEC holes)
  json validation_loads;    // per-port series R + L + C; all-zero branch is open
  std::string field_export_directory;
  json field_currents;
  std::vector<Vec3> field_probes;
  // Surface quadrature on the conductor boundary faces: writes position, outward normal into the
  // field region, area weight and the per-port B to a binary file.  This is the exact boundary
  // integral of K = n x H, with no point location and no probe offset.
  std::string surface_quadrature_file;
  int surface_quadrature_order = 2;   // integration order on each boundary face
  std::vector<double> frequencies;  // Hz
  // Adaptive frequency sweep (3D, docs/physics/adaptive_sweep.md): solve at initial_points frequencies of the band,
  // fit a barycentric rational model to Z and Psi, and add the frequency where the model is least certain until its
  // estimated error is below `tolerance` and the two newest solves agreed with the models fitted before them, or
  // max_solves is reached. `frequencies` then holds the initial points (so mesh sizing and the exterior map see the
  // band).
  struct Adaptive {
    bool enabled = false;
    double f_min_Hz = 0;
    double f_max_Hz = 0;
    double tolerance = 1e-3;     // relative, pointwise (floor: 1 % of the output group's typical size)
    int max_solves = 30;         // every frequency solved, the initial points included
    int initial_points = 5;      // the band ends included
    std::string spacing = "log"; // "log" | "linear": initial points, candidate grid and dense evaluation
    int dense_points = 400;      // dense evaluation of the model over the band (results)
  } adaptive;
  std::string boundary = "radiation";  // "radiation" (exact outgoing DtN) | "pec" (closed conducting sphere)
  // Perfectly matched layers (beta): [{regions, type, center, axis, scaling_factor, curvature,
  // wavelength_m}], see axionfem/pml.hpp. Closed (pec) boundary, frequency sweeps only.
  json pml = json::array();
  double thickness_m = 1.0;  // planar 2D designs: out-of-plane thickness D (3D quantities = per-length x D)
  int modes = 0;                       // eigenmodes per family (closed sphere)
  std::vector<std::string> families;   // eigenmode families: "TE", "TM" (default: the pickup's, or both)
  int dtn_modes = 80;
  double scene_frequency = 0;          // Hz: RF fields kept for scenes (0: the first sweep frequency)
  // Additional independent 3D current ports; the existing pickup is always port 0.
  std::vector<Port> ports;
  json source;                        // prescribed B0, evaluated at physical quadrature points
  std::vector<std::string> source_regions;  // nonempty: B0 is zero outside these regions (e.g. a COMSOL Je domain)
  json source_controls = json::array(); // {name,source,quadrature_order}: overlap controls, no extra Maxwell RHS
  bool source_check = false;          // direct volume-source vs reciprocal open-port voltage (PEC)
  int source_quadrature_order = 0;    // 0: MFEM default; positive: explicit tetrahedral integration rule
  std::vector<double> validation_shunt_R_ohm;  // positive shunt resistors: direct loaded FEM closure check
  // Eigenmodes nearest this frequency (shift-invert about (2 pi f)^2, both sides); 0: the lowest modes
  double target_f_Hz = 0;
  // 3D eigensolver (beta): "ame" (lowest modes, hypre AME + AMS) or
  // "shift_invert" (modes nearest target_f_Hz: STRUMPACK factor of A - sigma M, M-inner-product Lanczos).
  // Empty: shift_invert when target_f_Hz is given, otherwise ame. max_iter/tol: AME; max_steps: Lanczos cap.
  struct Eigensolver {
    std::string method;
    int max_iter = 500;
    double tol = 1e-8;
    int max_steps = 0;  // 0: automatic
    // shift-invert completeness check: Lanczos steps from a fresh start vector M-orthogonal to the returned
    // modes, flagging Ritz values inside the returned window (0: off)
    int completeness_steps = 20;
  } eigensolver;
};

// Cavity haloscope analysis of the axisymmetric eigenmodes (docs/design_format.md, "Cavity haloscopes").
// Beta (docs/validation/cavity_2d.md).
struct CavitySpec {
  bool enabled = false;
  Vec2 interior_point;              // selects the connected non-conductor component that is the cavity
  std::optional<Vec2> B_ref_point;  // |B0| reference for C at this point (default: the cavity volume rms of B0)
  // Static field B0 of the form factor: "applied" (default): the field of the background and windings with
  // superconducting regions/sheets that bound the cavity treated as transparent (a real SC cavity in a haloscope
  // magnet is in the mixed state or a thin film: the magnet field penetrates it); "static": the full static
  // solution, in which closed superconducting walls expel the field (Meissner).
  std::string B0_field = "applied";
  bool losses = true;               // wall (rf models) and dielectric losses per mode: G, Q_wall, Q_d, Q0
  bool interior_given = false;      // interior_point was given (required for axisymmetric designs)
  // walls without rf data are lossless on purpose: no warning, losses.walls_lossless_by_request, Q0 = Q_dielectric
  // when no wall has rf data; without it such walls are warned and an all-lossless Q0 is omitted (as in 3D)
  bool allow_lossless_walls = false;
  // "symmetry": "xsec" (z-uniform cross-section) only:
  std::string wall_material;        // material of the domain boundary (outer wall); "" = lossless
  double length = 0;                // cavity length L [m] for the end-cap losses of TM_z modes (0 = per unit length)
  std::string end_cap_material;     // material of the two end caps (default: wall_material)
};

// Cavity haloscope analysis of 3D eigenmodes (same JSON section "cavity"; beta, docs/validation/cavity_3d.md).
// Needs fullwave.rf_only (closed PEC domain, no static solve); B0 for C is fullwave.source.
struct Cavity3DSpec {
  bool enabled = false;
  Vec3 interior_point;              // selects the connected non-conductor component that is the cavity
  std::optional<Vec3> B_ref_point;  // |B0| reference for the form factor (default: the cavity volume rms of B0)
  bool losses = true;               // wall and dielectric losses per mode: G, Q_wall, Q_d, Q0
  std::string wall_material;        // material (rf data) of the outer PEC boundary (domain_solid wall or sphere)
  // walls without rf data are lossless on purpose: Q0 (= Q_dielectric then) is reported without a warning,
  // flagged by losses.walls_lossless_by_request; without it such walls warn and an all-lossless Q0 is omitted
  bool allow_lossless_walls = false;
  bool mode_overlaps = false;       // eta_ij = |int E_i . B_j| / sqrt(int |E_i|^2 int |B_j|^2)
  // write every wall quadrature point (x, y, z, dA, |E|, |E_n| consistent flux, |B_t|, |B_t| consistent flux) of each
  // mode to <workdir>/wall_samples_<k>.f64 (rank 0; peak_wall_fields.samples_file), e.g. for SRF wall peaks
  bool wall_samples = false;
};

// Critical-state (screening-current) analysis of designated HTS regions (axisymmetric).
// The regions keep the ideal-superconductor (Meissner) model in the static and pickup solves; this
// analysis is a separate time-dependent solve in which they carry power-law currents.
struct HTSRegion {
  std::string region;
  std::string conductor;
  std::string initial = "zero_field_cooled";  // "zero_field_cooled" | "field_cooled"
};

struct HTSSpec {
  bool enabled = false;
  std::vector<HTSRegion> regions;
  // Drive waveform: every winding current and the background field are multiplied by scale(t),
  // piecewise linear through (times[i], scales[i]).
  std::vector<double> times, scales;
  int steps_per_segment = 20;
  std::vector<Vec2> field_of_view;  // points where the screening-current-induced field is reported
  double newton_tol = 1e-4;
  int max_newton = 60;
  double e_floor = 1e-6;    // Newton Jacobian uses |E| >= e_floor * E0 (the residual is exact)
  int profile_points = 41;  // final J(r) profile along each region's radial midline (0 = none)
};

// ---- mechanics / thermal analyses (axisymmetric and 3D; see docs/physics/mechanics_thermal.md) ----
struct Support {
  std::string region;
  std::string fix = "all";  // axisymmetric: "all" | "r" | "z"; 3D: "all" | "x" | "y" | "z"
};

// A prescribed normal pressure on the faces of region `on` shared with `from` (a region name, or
// "vacuum" for every element that is not part of the structure).  p > 0 compresses `on`.
struct PressureLoad {
  std::string on, from;
  double p_Pa = 0;
};

struct MechanicsSpec {
  bool enabled = false;
  std::vector<Support> supports;
  std::vector<PressureLoad> pressures;
  bool cooldown = false;
  // 3D only (parsed from the source document by cpp/src/model3d_mech.cpp): symmetry_planes, probes,
  // external_field
};

struct ThermalFix {
  std::string region, curve;  // exactly one of the two
  double T_K = 0;
};

struct ThermalLoad {
  std::string region;
  double W = 0, W_m3 = 0;
  bool total = false;  // true: W is the total power in the region; false: W_m3 is the density
};

struct ThermalSpec {
  bool enabled = false;
  std::vector<ThermalFix> fixed;
  std::vector<ThermalLoad> loads;
};
// ---- end mechanics / thermal analyses ----
// ---- electrostatics -----------------------------------------------------------
// Capacitance matrices, parasitic and neck capacitance, self-resonance (axisymmetric designs).
// The presence of the section enables the analysis.
struct ElectroSpec {
  bool enabled = false;
  std::string boundary = "open";  // "open" (exact Laplace DtN, ground at infinity) | "grounded" (outer sphere)
  std::string notes;
};
// -----------------------------------------------------------------------------

// Externally given static field (e.g. a magnet modeled elsewhere, a COMSOL export).
// The solver computes the reaction of superconductors and persistent circuits to it.
struct Background {
  std::string type;            // "" (none), "uniform", "grid"
  double Bz = 0;               // uniform axial field [T]
  std::vector<double> r, z;    // grid nodes (r[0] must be 0), metres
  std::vector<double> Br, Bz_grid;  // row-major [iz * nr + ir], tesla
  double scale = 1.0;
  std::string interpolation = "hermite";  // "hermite" (bicubic psi, default) or "bilinear" (legacy)
};

struct Design {
  std::string schema = "axionfem.design/1";
  std::string name;
  std::string description;
  std::string symmetry = "axisymmetric";
  ParamMap parameters;
  std::map<std::string, Material> materials;
  std::map<std::string, Conductor> conductors;
  std::vector<Region> regions;
  std::vector<std::string> disabled_regions;  // regions switched off by their `enabled` expression
  std::vector<Curve> curves;
  std::vector<Winding> windings;
  std::vector<Circuit> circuits;
  std::vector<Superconductor> superconductors;
  std::optional<Pickup> pickup;
  Background background;
  FullWaveSpec fullwave;
  CavitySpec cavity;
  Cavity3DSpec cavity3d;  // the "cavity" section of a 3D design
  HTSSpec hts;
  MechanicsSpec mechanics;
  ThermalSpec thermal;
  ElectroSpec electrostatics;
  std::vector<Benchmark> benchmarks;
  ScanSpec scan;
  CostSpec cost;
  MeshSpec mesh;
  json source;  // the input document, unchanged

  const Material& material_of_region(const std::string& region) const;
  const Region* find_region(const std::string& name) const;
  const Curve* find_curve(const std::string& name) const;
};

Design parse_design(const json& doc);

// Built-in materials available to every design ("vacuum", "air", "winding",
// "superconductor", "nb", "copper", "copper_classical", "iron", "iron_knee").
std::map<std::string, Material> builtin_materials();

// Built-in conductors available to every design ("nbti", "nb3sn", "rebco"), with the published
// parameter sets of docs/validation/hts.md.  A design may override them by name.
std::map<std::string, Conductor> builtin_conductors();

// Built-in readout benchmarks ("optimistic", "conservative", "conservative_solenoid"; docs/design_format.md).
Benchmark builtin_benchmark(const std::string& name);

// The geometry of a parsed design with every expression evaluated: parameters (numbers), regions in
// painting order with their material and material kind and the resolved `shape` (axisymmetric) or `solid`
// (3D) exactly as the mesher receives them, curves, disabled regions, the mesh section and the pickup.
// Consumed by the COMSOL exporter (python/axionfem/comsol.py, docs/comsol_export.md).
json resolved_geometry(const Design& d);

// Machine-readable JSON Schema of the design document (for agents/editors).
json design_schema();

}  // namespace axf::spec
