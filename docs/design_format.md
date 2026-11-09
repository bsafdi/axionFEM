# Design format `axionfem.design/1`

A design is one JSON document. The C++ core parses and validates it
(`axf validate`, `axionfem.Design.validate()`); `axf schema` prints the JSON
Schema. Units are SI: metres, amperes, tesla, kelvin, webers. Any numeric
field may be a number or a string expression over `parameters`
(`"0.6*R"`, `"sqrt(H*R)"`, `"exp(-0.5)"`; functions: sqrt exp log log10 sin
cos tan asin acos atan atan2 abs min max pow; constants pi, e, mu0).

**Expression grammar and precedence.** Expressions follow standard math
precedence, loosest first:

| Level | Operators | Associativity |
|---|---|---|
| sum | `a + b`, `a - b` | left |
| product | `a * b`, `a / b` | left |
| sign | unary `-a`, `+a` | right |
| power | `a ^ b` (the exponent may carry a sign) | right |
| primary | number, parameter, constant, `f(args)`, `( expr )` | |

This is Python's `**` (and Fortran's, Julia's, R's): `2^3^2` is 512, not
MATLAB's 64, and `-2^2` is -4, not Excel's 4. So `exp(-z^2/(2*s^2))` is a
Gaussian, `2^-2` is 0.25, `-x^-2` is -(x^-2) and `3*-2^2` is -12. Write `(-x)^2`
for the square of a negative number.

Numbers are strict: `1.2.5`, `1..5` and `.` are errors (not 1.2, 1 and 0), and
a literal must be 0 or have a magnitude between 2.2e-308 and 1.7e308 (`1e400`
and the subnormal `1e-310` are errors). `min` and `max` take one or more
arguments and reject a NaN argument. Nesting deeper than 1000 levels (signs,
exponents, parentheses) is an error. A result that is not finite is an error.

Designs are **axisymmetric** by default: everything lives in the meridional
half-plane (r ≥ 0, z) and is revolved about the z axis. `"symmetry": "3d"`
selects the full 3D solver (see the end of this page); `"symmetry": "xsec"` the
(x, y) cross-section of a z-uniform cavity (see "Cross-section cavities");
`"symmetry": "planar"` a 2D full-wave model on an imported mesh (as an axisymmetric design with `mesh.file`; see
"Imported meshes" at the end of this page).

Named `parameters` are also the variables of the shape sensitivities and of the
gradient-based optimizer: `Results.sensitivities()` returns dΨ/dp, dL_p/dp,
d𝓕/dp, dcost/dp … for every parameter, and `axionfem.optimize()` drives them.
See [optimization.md](optimization.md).

```json
{
  "schema": "axionfem.design/1",
  "name": "my_core",
  "description": "free text",
  "parameters": {"Rc": 1.0, "b": "2.5*Rc"},
  "materials": {"steel": {"kind": "iron", "bh": "table", "B": [...], "H": [...]}},
  "conductors": {"coil": {"law": "nbti_bottura", "T_op": 4.2, "fill_factor": 0.3}},
  "regions": [ ... ],
  "curves": [ ... ],
  "windings": [ ... ],
  "circuits": [ ... ],
  "superconductors": [ ... ],
  "pickup": { ... },
  "background": { ... },
  "hts": { ... },
  "benchmarks": ["optimistic", "conservative"],
  "scan": { ... },
  "cost": { ... },
  "mesh": { ... }
}
```

## Regions

```json
{"name": "cavity", "shape": SHAPE, "material": "nb", "mesh_size": 0.01, "surface_size": 0.002}
```

Regions are painted in order: where shapes overlap, **later regions win**.
Space not covered by any region is `air`.

Region and curve names must be unique, and `air` and `infinity_shell` are
**reserved**: the solvers use them for the parts of the domain that no region
describes (the background and the 3D open-boundary shell), so a region of that
name is rejected — it would be silently treated as vacuum by the electrostatic
and mechanical analyses.

`"enabled": EXPR` (number, expression or boolean) switches a region on or off;
the region exists only if the value is nonzero. Templates use it for optional
parts (e.g. CORE's iron caps, `"enabled": "iron_t"`), so `Design.set(iron_t=…)`
adds or removes them.

| shape | fields |
|---|---|
| `{"rect": {"r": [r0, r1], "z": [z0, z1], "fillet": f}}` | rectangle, optional corner radius |
| `{"frame": {"r": [...], "z": [...], "thickness": t, "fillet": f}}` | rectangular ring (wall of a box); inner corners get radius max(f − t, 0) |
| `{"polygon": {"points": [[r, z], ...], "fillet": f}}` | polygon, optional fillet at every vertex |
| `{"polygon": {"points": [[r, z], [r, z, f1], ...]}}` or `{"polygon": {"points": [...], "fillets": [f0, f1, ...]}}` | polygon with a **per-vertex** fillet radius; a third point coordinate or a parallel `fillets` list (one radius per point, 0 = sharp), not both. A radius is rejected if two neighbouring fillets would need more of the edge between them than it is long |
| `{"contour": {"start": [r, z], "segments": [{"line": [r, z]}, {"ellipse": {"center": [rc, zc], "axes": [ar, az], "t": [t0, t1]}}, ...]}}` | closed loop of straight segments and axis-aligned elliptical arcs, point(t) = (rc + ar cos t, zc + az sin t) with t in radians from t0 to t1 (either direction, \|t1 − t0\| < 2π). Each arc must start where the previous segment ends (to 1e-9 of its size); a closing line back to `start` is added unless the last segment ends there. The mesher builds every arc exactly (a rational quadratic B-spline), so the wall is smooth and second-order mesh nodes lie on the true curve (no facet corners; used by the SRF cells, `srf.elliptical_cell_contour(...)["shape"]`). Also valid as a `revolve` profile |
| `{"disk": {"center": [r, z], "radius": a}}` | disk (a torus of revolution) |
| `{"union": [SHAPE, ...]}`, `{"difference": [SHAPE, SHAPE, ...]}` | booleans (difference: first minus the rest) |

## Curves (zero-thickness sheets)

```json
{"name": "shield", "path": PATH, "material": "superconductor", "surface_size": 0.002}
```

PATH is one of `{"segment": [[r,z],[r,z]]}`, `{"polyline": {"points": [...], "closed": bool}}`,
`{"rect": {"r": [...], "z": [...], "fillet": f}}` (outline) or
`{"arc": {"center": [r,z], "radius": a, "angles": [θ0, θ1]}}`.
Sheets are supported in the static poloidal and azimuthal-pickup problems and in cavity eigenmodes.
Designs with a toroidal field (a poloidal winding or a port pickup) reject
sheets with an error; use thin regions there. Field peaks at sheet edges are
singular and are flagged (see Results).

### Thin magnetic layers

```json
"materials": {"mumetal": {"kind": "linear", "mu_r": 2.0e4}},
"curves": [{"name": "shield", "path": PATH, "material": "mumetal", "thickness": 0.002}]
```

A curve with a `thickness` (m) and a `linear` material with `mu_r` > 1 is a thin
high-permeability layer: a shield (mu-metal, soft iron) of that thickness, centred
on the curve, that is not meshed through its thickness. This is COMSOL's "Magnetic
Shielding" condition. The vector potential jumps across the curve by the flux the layer carries.
Physics, weak form and validity conditions: docs/physics/thin_magnetic_layer.md. Validation:
docs/validation/thin_magnetic_layer.md.

- **Ψ is zero for every design accepted with a layer.** An azimuthal pickup needs a toroidal static field, and
  the toroidal-field problem, port pickups and the full-wave analysis refuse layers. So 𝓕, the reach and the cost at
  the requirement are not available with layers yet. Shielding factors, poloidal static fields, L_p, W_mag and
  their sensitivities are.
- Valid when d ≪ the layer's radius of curvature (in the meridional plane and r) and d ≪ the length over
  which the field varies along it. μ_r d/R may be large: that is the shielding regime. For a sphere the
  error of S − 1 is −(13/12 + 1/(2μ_r))(d/R)².
- Put the curve on the layer's mid-surface.
- **Mesh next to the layer.** Keep it several layer thicknesses high, using the curve's `surface_size`.
  - A warning flags `thin_layers[*].min_h_perp_over_d` below max(4, ⌈p²/4⌉) at mesh order p: 4 up to order 4,
    then 7, 9, 13 and 16.
  - A much finer mesh makes the system indefinite, and the solve stops with an error. With layers the field
    system always uses the Eigen LDLT, whose pivots give that check. `AXF_STRUMPACK_MIN_DOFS` and
    `AXF_ORDERING=strumpack` are ignored; `AXF_ORDERING=amd|metis` are honoured.
- **Where it acts.** The layer acts in the static poloidal problem (windings, persistent circuits,
  superconductors, `background`) and in the azimuthal pickup problem. In the pickup problem it is a
  non-conducting magnetic layer, and a warning says so: a metal shield is a conductor at the readout frequency
  (see the physics note).
- **Junctions.**
  - A layer may run along a superconductor or a linear region, and may end while running along one, as a partial
    coating does.
  - A superconducting sheet may end on a layer, which leaves the layer's flux free, or cross it, which cuts it.
  - Superconductors (of different regions or curves) on the two sides of a layer are separate bodies. A
    superconducting region crossed by a layer stays one body and cuts the layer, as a crossing sheet does.
  - Layers may meet and cross each other.
  - **Refused:**
    - a layer that ends on a magnetic region (|μ_r − 1| > 1 %, or iron) that none of its end faces runs along
      (the rule is per region): its flux cannot pass into the region. Mesh the junction as a region, make both
      parts layers, or run the layer along the region;
    - a sheet drawn on a layer's own path: offset it by about an element;
    - two superconducting bodies (regions or sheets) meeting exactly at a layer's end: extend the layer past the
      contact;
    - a pickup conductor that forms several separate bodies, with or without layers.
- **Results.** `static.poloidal.thin_layers` and `pickup.thin_layers` list per layer:
  - the peak B in the layer (`B_layer_max_T`), with the saturation threshold `B_sat_T`: the material's
    `B_sat` (a `linear` material may give one) or 0.7 T. A warning flags the peak above it;
  - `H_layer_max_A_per_m`;
  - the peak excess flux;
  - the smallest excess permeance `P_min_H`;
  - the number of split DOFs;
  - `min_h_perp_over_d`, over the field sides.
- **Refused** (with an error naming the curve):
  - `iron` (nonlinear) layers, and layers next to nonlinear iron;
  - the junctions listed above;
  - toroidal-field problems (poloidal windings, port pickups);
  - the `fullwave`, `cavity`, `hts`, `electrostatics` and `mechanics` analyses, and `mesh.adapt`;
  - 3D, cross-section and planar designs.
- **Not refused.**
  - Forces on layers are not computed; the force closure check is then marked invalid, with a caveat.
  - `thermal` ignores layers, with a warning.
  - A layer's material `warm` flag keeps it out of the cold envelope.

## Materials

Built in:

- `vacuum`, `air`, and `winding` (vacuum-like stranded coil).
- `superconductor` and `nb` (ideal Meissner).
- `copper` (conductor: transparent to the static field, perfect conductor for
  the RF pickup mode; RF wall loss: cold copper, RRR ≈ 100, ρ = 1.7×10⁻¹⁰ Ω m,
  `T_K` 0.03, `rf` model `anomalous` with ρℓ = 6.6×10⁻¹⁶ Ω m², diffuse p = 0,
  quasi-static). The classical `skin` model with the same ρ underestimates R_s of
  cold copper above ~1 MHz (the anomalous R_s is higher by 2.4 % at 1 MHz, 10 % at
  10 MHz, 30 % at 100 MHz, 1.65× at 1 GHz;
  [surface_resistance.md](physics/surface_resistance.md) Sec. 1.1).
  It is the zero-field value of a clean surface: an upper bound on Q. Measured cold
  copper cavities reach ~80–86 % of it (Sec. 2.4 there), and in a multi-tesla static
  field magnetoresistance and cyclotron-orbit effects (not modelled, no warning) raise
  R_s further; use a measured `table` R_s(f) at the operating field when it matters.
- `copper_classical`: the same copper with the classical `skin` model (same ρ and
  `T_K`, otherwise identical). Use it for like-for-like comparisons with a
  classical impedance boundary condition (e.g. COMSOL's); its cavity losses warn in
  the anomalous regime.
- `iron` and `iron_knee` (warm soft iron; knee law μr = 1000, B_sat = 2 T).

Custom materials: `{"kind": "vacuum" | "linear" | "iron" | "superconductor" | "conductor", "mu_r", "eps_r", "bh": "knee" | "table", "B_sat", "B": [...], "H": [...], "density", "warm", "cost_per_kg", "mechanical": {...}, "thermal": {...}}`
(`mechanical` and `thermal` are used by the mechanics and thermal analyses, see below).
`eps_r` (relative permittivity, default 1), `tan_delta` (dielectric loss tangent) and `sigma_S_m` (bulk
conductivity of a non-PEC medium: `vacuum`, `linear` or `iron` kinds) only enter full-wave analyses, where
they give a complex permittivity ε'(1 + i tanδ) + iσ/ω. Bulk eddy currents are resolved by the mesh (a
warning flags elements larger than the skin depth); good conductors are better modelled as `conductor`
(perfect at RF) with an `rf` surface-resistance model.

`rf` surface-resistance models (formulas, regimes of validity and sources: `docs/physics/surface_resistance.md`;
all models are evaluated by `axf::rs` / `RFLoss::Rs`, the same code in every engine; `axionfem.cavity.surface_resistance(rf, f)`
evaluates an `rf` block from Python). Fields that belong to another model are rejected.

| model | fields | R_s(f) |
|---|---|---|
| `skin` | `rho_ohm_m`, `T_K` | √(π f μ0 ρ) (classical skin effect, mean free path ≪ skin depth). Not for pure metals at cryogenic temperature above ~MHz: a cavity loss evaluation warns when the anomalous parameter α (with the copper ρℓ) exceeds 1 (R_s then low by ≳ 4 %); use `anomalous` (the builtin `copper` does; `copper_classical` is `skin`) |
| `power` | `Rs_ohm`, `f_ref`, `exponent`, `T_K` | Rs_ohm (f/f_ref)^exponent |
| `table` | `f_Hz: [...]`, `Rs_ohm: [...]`, `T_K` | measured R_s(f): ≥ 2 points, f strictly increasing, log-log interpolation, **no extrapolation** (a frequency outside the table is an error; a cavity mode outside it reports no Q_wall/Q0, with a warning) |
| `anomalous` | `rho_ohm_m` **or** (`RRR`, `rho_300K_ohm_m`), `rho_l_ohm_m2`, `specularity` (default 0), `v_F_m_s` (default 0), `T_K` | Reuter–Sondheimer anomalous skin effect over the whole classical → anomalous crossover (normal metals at cryogenic T). ρ = rho_ohm_m or rho_300K_ohm_m/RRR is the resistivity at T_K; `rho_l_ohm_m2` = ρ × mean free path (copper 6.6e-16); `specularity` p ∈ [0, 1] (0 diffuse, the usual assumption; 1 specular). `RRR` is the residual-resistivity ratio ρ(room T)/ρ(4.2 K) and gives the residual resistivity, so it is refused for `T_K` > 20 K (give `rho_ohm_m` at T_K there); `v_F_m_s` > 0 includes the relaxation time τ = ℓ/v_F (copper 1.57e6; 0 = quasi-static ωτ → 0, error ≲ 1 % for ωτ ≲ 0.2). Limits: `skin` for ℓ ≪ δ; (√3 ρℓ μ0²ω²/16π)^{1/3} (diffuse), 8/9 of it (specular) for ℓ ≫ δ |
| `bcs` | `A_ohm_K`, `f_ref`, `Delta_over_kTc`, `Tc_K`, `R_res_ohm`, `log_factor` (default false), `T_K` | A/T (f/f_ref)² exp(−(Δ/kT_c) T_c/T) + R_res (low-T BCS approximation; T_K < T_c/2 is enforced). `log_factor: true` multiplies by ln(2.246 kT/hf)/ln(2.246 kT/hf_ref), the Mattis–Bardeen factor the plain f² law drops (needs h f_ref < kT_K); without it a cavity loss evaluation warns when this factor differs from 1 by > 10 % (f² scaling from 1.5 to 10 GHz overestimates R_BCS by 1.6–1.9×), and with h f > kT. Niobium: `A_ohm_K` 2e-4, `f_ref` 1.5e9, `Delta_over_kTc` 1.91 (17.67 K/9.25 K), `Tc_K` 9.25; R_res is preparation dependent (1–10 nΩ for good Nb). Away from f_ref prefer the tabulated form |
| `bcs` (tabulated) | `f_Hz: [...]`, `T_grid_K: [...]`, `Rs_bcs_ohm: [[...], ...]` (one row per T, one value per f), `R_res_ohm`, `T_K` | R_BCS(f, T) from a full BCS code (e.g. SRIMP) or measurement, + R_res: log-log in f, ln R linear in 1/T; no extrapolation in f or T |

`bcs`, `power` and `table` accept an additive vortex term
`"flux_flow": {"Bc2_T", "rho_n_ohm_m", "lambda_m", "f_depin_Hz", "creep_epsilon", "B_T", "orientation"}`:
ΔR_s = Re √(iμ0ω(iμ0ωλ² + s ρ_v)) with ρ_v = ρ_n (|B|/B_c2) (ε + i f/f0)/(1 + i f/f0) (Bardeen–Stephen flux-flow
resistivity, Gittleman–Rosenblum pinning, Coffey–Clem creep in the Pompeo–Silva form; `f_depin_Hz` = f0, 0 for free
flux flow; `creep_epsilon` = ε ∈ [0, 1], default 0). This is the full local London + vortex impedance: ≈ Re ρ_v/(2λ)
when |ρ_v| ≪ μ0ωλ², ≈ Re √(iμ0ωρ_v) when |ρ_v| ≫ μ0ωλ² (the multi-tesla haloscope case). `rho_n_ohm_m` is the
normal-state resistivity, `lambda_m` the London depth. `B_T`: |B| in the wall; omitted (or `"local"`): the local
static field B0 at each wall point (cavity wall losses only; other analyses need `B_T`). `orientation`: `"field"`
(default) or a fixed factor s ∈ [0, 1]. With `"field"` the cavity loss evaluation uses s = sin²(J_rf, B0) per wall point
(B0 = the form-factor static field, J_rf ∥ n × H of the mode), so current parallel to the vortices is lossless; other
engines (driven sweep, MQS losses) use s = 1, the upper bound. |B| < B_c2 is enforced (also for the
local field). This is an **order-of-magnitude model** (see [surface_resistance.md](physics/surface_resistance.md), Sec. 4); cavity results report per wall
`Rs_flux_flow_ohm`, `Rs_flux_flow_perpendicular_ohm` (s = 1) and `flux_flow_orientation_factor` (|H_t|²-weighted s),
and a warning. An `Rn_ohm` entry is rejected: give `rho_n_ohm_m` and `lambda_m`.

`rf` surface losses accept `"magnetoresistance": {"model": "kohler", "B0_T", "n"}` (resistivity factor
1 + (B/B0)ⁿ) or `{"B_T": [...], "factor": [...]}` (table), evaluated in the local static field. The factor
scales ρ for the `skin` model (R_s ∝ √ρ) and R_s itself for the `power` and `table` models; the loss integral is then
reported as `Gamma_magnetoresistance`. It is rejected with `anomalous` (at high field the cyclotron radius falls below the
mean free path, outside the RS model; use a measured `table` at the operating field — `anomalous` with a zero-field ρ
is then an uncontrolled estimate), with `bcs` (use `flux_flow`) and together with `flux_flow`.
`T_K` is the surface temperature used in the T/Q budget; for `bcs` it is also the temperature of R_BCS.

**`pickup.losses`.** Both the 2D and the 3D MQS engine report it.
- **Geometric factor:** for every conductor or superconductor surface region,
  Γ = ∮ |H_I|² dA / I² over its faces to the field region (the readout field per unit current, H on the
  field side). On an iron field side H = B/μ uses the secant permeability of the pickup solve (both engines;
  the initial `mu_r` would underestimate H next to saturated iron).
- **Per surface with an `rf` block:** `R_eff_ohm` = R_s(f) Γ and `Q` = 2πf L_p / R_eff at the scan band edges
  (`f_min`, `f_max`), plus `T_K` and `T_over_Q_K`. Only conductor and superconductor surfaces count; an `rf`
  block on another kind of material is ignored, with a warning.
- **Totals:** `T_over_Q_total_K`, and `vs_benchmarks` comparing it with each benchmark's T/Q.
- **3D only:** `Gamma_by_field_region` splits each surface's Γ by the region on the field side (e.g. the
  cavity air, the port and an open slit gap), and `losses.engine` names the engine.
- **Agreement, 3D against 2D:**
  - **Smooth surface field.** On the solenoid template (a 1/r surface field, no singular edge), order 2 agrees
    with 2D and the closed form to 1.5 % in Γ and Q on a coarse mesh and 0.5 % on a finer one.
    Order 1 is good only to about 20 %, since its surface field is piecewise constant
    (`benchmarks/rf_losses_3d.py`, record `benchmarks/records/rf_losses_3d.json`).
  - **Slits and partial ports.** The lips of a slit carry an edge-concentrated surface field, and a port across
    part of the slit adds the field that leaks through the open rest of it. Refine the slit surfaces and compare
    before quoting Γ (`benchmarks/fullwave_surface_losses.py`, the slotted-box cases).
  - **Sharp edges.** Next to a sharp convex edge |H|² is singular and Γ converges slowly. Refine the surfaces
    there and compare before quoting Γ.
- **Builtin materials:** `copper` has a skin-effect `rf` block; `nb` and `superconductor` have none, so they
  report Γ only.
- **Q ratios:** for the same surface material, frequency and temperature, and without magnetoresistance,
  Q ∝ L_p/Γ, so ratios between configurations need only Γ and L_p. With magnetoresistance use
  `Gamma_magnetoresistance`.

B–H tables must start at (0, 0) and be strictly increasing. They are
interpolated with monotone cubics and continued with slope 1/μ0.

Iron is nonlinear in every static field: Newton for poloidal fields; in a
static toroidal field H = G/r is exact in current-free regions and B = B(H) is
applied pointwise. The stored energy is ∫H·dB. The RF pickup problems use the
secant permeability B/H of the full static field, which is exact for RF fields
perpendicular to the static field (toroid, solenoid and CORE topologies) and an approximation
otherwise.

`copper` is transparent to the static field and a perfect conductor at RF. Ψ
then includes the axion current that flows inside the copper walls, an O(t/R)
modelling choice (about 0.5 % for t = 0.005 R).

## Conductors (critical surfaces Jc(B, T))

```json
"conductors": {
  "coil":  {"law": "nbti_bottura",  "T_op": 4.2, "fill_factor": 0.3},
  "cable": {"law": "nb3sn_durham",  "T_op": 4.5, "fill_factor": 0.25, "strain": -0.0025},
  "tape":  {"law": "rebco_kim",     "T_op": 20,  "fill_factor": 0.012, "tape_normal": [0, 1]},
  "meas":  {"law": "table", "B_T": [0, 1, 2], "T_K": [4.2, 20], "Jc": [4e9, 2e9, 2e9, 1e9, 0, 0]}
}
```

Three conductors are built in and may be used by name or overridden: `nbti`, `nb3sn`, `rebco`.
A conductor is attached to a winding (`"conductor": "coil"` in `windings`) to get its
critical-current margins, and to a critical-state region (`hts.regions[].conductor`).
Current densities are in A/m², fields in tesla, temperatures in kelvin.

| law | formula and parameters |
|---|---|
| `nbti_bottura` | Bottura's practical fit: Jc = (C0/B) b^α (1−b)^β (1−t^1.7)^γ, b = B/Bc2(T), Bc2(T) = Bc20(1−t^1.7), t = T/Tc0. Parameters `C0` [A·T/m²], `alpha`, `beta`, `gamma`, `Bc20`, `Tc0`. |
| `nb3sn_durham` | Durham/ITER unified scaling at fixed axial `strain`: Jc = (C0/B) s(ε)(1−t^1.52)(1−t²) b^p (1−b)^q, Bc2*(T,ε) = Bc20 s(ε)(1−t^1.52), Tc*(ε) = Tc0 s(ε)^{1/3}. Parameters `C0`, `p`, `q`, `Bc20`, `Tc0`, `Ca1`, `Ca2`, `eps_0a`, `strain`. |
| `rebco_kim` | anisotropic (elliptical) Kim: Jc = Jc0 g(T) / [1 + √((k B_∥)² + B_⊥²)/B0]^b_exp with B_⊥ normal to the tape and g(T) = ((Tc−T)/(Tc−T_ref))^m_T. Parameters `Jc0`, `B0`, `b_exp`, `k_anis`, `T_ref`, `Tc`, `m_T`. |
| `table` | `B_T` (increasing), optional `T_K` (increasing) and `Jc` with len(B_T)·len(T_K) values, row-major with B outer; bilinear, constant outside in B. With `k_anis` ≠ 1 the table is indexed by the anisotropic effective field √((k B_∥)² + B_⊥²). |

Common fields: `T_op` (operating temperature), `fill_factor` (superconductor cross-section divided by
the winding or tape-stack cross-section, so the engineering current density is `fill_factor` × Jc),
`tape_normal` ([n_r, n_z], the normal of the tape plane in the (r, z) half-plane; the default [0, 1]
is a pancake winding, whose harmful field component is the axial one), `B_min` (Jc is evaluated at
max(|B|, B_min): the NbTi and Nb3Sn fits diverge as B → 0, default 50 mT for them), and the E–J power
law of the critical-state solver, `n_value` and `E0` [V/m].

Defaults come from the published parameter set of the chosen law, so a design only gives what it
changes; the sets and the reference points they reproduce are in docs/validation/hts.md.
`axionfem.conductors()` lists the built-ins and `axionfem.jc(conductor, B_perp, B_par, T)` evaluates
any of them from Python.

## Windings and circuits

```json
{"name": "magnet", "region": "coil", "type": "azimuthal", "ampere_turns": 1e6}
{"name": "pack1", "region": "p1", "type": "azimuthal", "turns": 120, "circuit": "c1"}
```

- **`azimuthal`**: the current circulates around the z axis (solenoids, CORE
  magnet, loops).
- **`poloidal`**: the current loops around the cross-section of a
  ring-shaped region. This is a toroidal magnet: the region must have exactly
  one hole.
- **Uniform current density.** The current density is uniform over the region.
  For poloidal windings the current spreads like a uniform-resistivity
  conductor, which is uniform in vertical legs.
- **`conductor`.** `{"region": "coil", "type": "azimuthal", "ampere_turns": 1e6, "conductor": "coil"}`
  reports the winding's critical-current margins (see Results).
- **Circuits.** `{"name": "c1", "type": "persistent", "flux_linkage": 0}` is a
  closed superconducting loop that conserves its flux. Its current is solved
  for (passive cancellation coils). `flux_linkage` is the **total** linkage
  (Σ turns × flux, including any background field): 0 means cooled in zero
  field, so the loop screens later fields. In the RF pickup problem it conserves
  the linkage change. `{"name": "c2", "type": "current", "current": 50}` is a
  powered series circuit. Windings in a circuit give `turns` and `circuit`
  instead of `ampere_turns`. Results report each circuit's current, achieved
  and target linkage (and the background part); a mismatch is a warning.

## Superconductors

```json
{"body": "cavity", "azimuthal": {"net_current": 0}, "poloidal": {"net_current": 0}, "azimuthal_slit": false}
```

Each superconducting region or sheet gets a default state; list it here to
change it.

- **`azimuthal`** constrains loops around the z axis in the static problem.
  `net_current` (default 0) means no transport current, i.e. the body was
  field-cooled with the magnet on. `{"flux": Φ}` freezes the flux through the
  loop, e.g. 0 for a zero-field-cooled body.
- **`poloidal`** constrains loops around the body's cross-section. These exist
  only for closed shells that enclose a cavity, e.g. the toroid sheath.
- **`azimuthal_slit`** marks a meridional slit, as in a toroid sheath. The body
  then carries no net azimuthal current in either problem.

For the RF pickup mode every uncut superconducting loop conserves its fluxoid.
The solver applies this automatically.

## Pickup (readout)

- `{"type": "azimuthal", "conductor": "sheath"}`: the readout current circulates
  around the axis on a slit superconductor (toroid sheath, loops).
- `{"type": "poloidal", "port": "neck", "inside_point": [r, z]}`: the readout
  current flows across a gap between conductors (coax, CORE neck). The port is
  a small vacuum region spanning the gap. `inside_point` is optional and picks
  the cavity side when it is ambiguous.

## Background field

An external static field (e.g. a magnet mapped in COMSOL) that the design sits
in. Superconductors and persistent circuits respond to it; W_mag and the cost
exclude it (warning).

```json
{"type": "uniform", "Bz": 0.01}
{"type": "grid", "r": [0, ...], "z": [...], "Br": [...], "Bz_grid": [...],
 "scale": 1.0, "interpolation": "hermite", "source": "...", "notes": "..."}
```

- Grid values are row-major, `[iz * nr + ir]`, in tesla; `r[0]` must be 0.
- The flux function ψ = r A_φ is integrated from the axis and interpolated with
  bicubic Hermite polynomials in (u = r², z), using ψ_u = B_z/2 and
  ψ_z = −r B_r at the nodes. The interpolated field is continuous,
  divergence-free and O(h³) accurate, including near the axis.
  `"interpolation": "bilinear"` selects the legacy scheme (C0 kinks that show up
  as spurious superconductor surface fields under mesh refinement).
- Outside the grid the background is taken as zero (warning if superconductors,
  cavities or windings lie outside).
- `python/axionfem/background.py` builds grids from COMSOL exports
  (`from_comsol_table`, `grid_block`, harmonic fits and blends).
- Linear magnetic regions and thin magnetic layers respond to the background in the static (poloidal)
  solve (a μ_r = 10 sphere in a uniform 1 T background has 2.500 T inside). The full-wave TM sweep uses that
  static field and is consistent.
  Nonlinear iron with a background is refused. So is the critical-state (`hts`) analysis with a background and
  a linear region of μ_r ≠ 1, because its source does not include that drive yet.

## Full-wave analysis beyond MQS (`fullwave`, axisymmetric)

The magnetoquasistatic results (Ψ, L_p, 𝓕) assume the detector is small compared
with the axion Compton wavelength. `fullwave` solves the time-harmonic Maxwell
equations (e^{−iωt}) for the readout at given frequencies:

```json
"fullwave": {"f_min": 1e6, "f_max": 3e8, "points": 30,
             "boundary": "radiation", "modes": 4}
```

| field | meaning |
|---|---|
| `frequencies` | list of frequencies [Hz] (or `f_min`, `f_max`, `points`, log-spaced; default 20 points) |
| `boundary` | `radiation` (default): exact outgoing-wave condition on the outer sphere (radiation resistance is computed); `pec`: a closed perfectly conducting sphere at `mesh.domain_radius` |
| `modes` | number of eigenmodes per family, computed with the closed sphere (0 = none) |
| `families` | eigenmode families, `["TE", "TM"]`; default: the pickup's family, both without a pickup |
| `dtn_modes` | multipoles in the exterior map (default 80) |
| `scene_frequency` | frequency [Hz] whose RF field is kept for scenes (default: the first sweep frequency) |
| `target_f_Hz` | eigenmodes nearest this frequency instead of the lowest ones (shift-invert about (2π f)², modes on both sides, the `modes` nearest in ω²); 2D only (beta) |

The readout family follows the pickup: an azimuthal pickup (toroid sheaths,
loops) is TE0 (azimuthal E); a poloidal port (coax, CORE) is TM0 (azimuthal H).
For each frequency `results.fullwave.sweep[]` gives, for I = 1 A (`k_R` = 2πf R/c with
R = `coupling.device_radius_m`, the outer radius of all regions):

- `Psi_re`, `Psi_im`, `Psi_abs`: Ψ(ω) = (1/(iωI)) ∫ E_I·B_lab dV. The axion EMF
  across the readout is V = −iωκΨ(ω) for the effective current κB_lab, so
  Ψ(ω) → the MQS Ψ as ω → 0. `Psi_over_MQS` = |Ψ(ω)|/Ψ_MQS. For TM readouts
  the integral includes the region outside the domain sphere (`Psi_exterior`:
  the static field's multipoles times the outgoing readout wave, radial
  integrals in closed form), so Ψ(ω) does not depend on the domain radius. With a grid `background` the
  background has its own exterior term, from the grid's moments as in 3D (the grid up to the sphere
  inscribed in its box, its exterior multipoles beyond, to order `dtn_modes`; so the grid should reach past the
  domain sphere and its inscribed sphere enclose the magnet), and `Psi_exterior` is the static lab field's term
  (windings and superconductor reactions, without the background). `Psi_exterior` is 2D's reciprocal-form split and
  differs from 3D's `Psi_exterior` (the primary form's exterior part): compare only Ψ and `Psi_background` across the
  engines. Each row then has `Psi_background` {`volume`, `exterior`, `exterior_last_order_fraction`} (both parts
  included in Ψ; `exterior` null where no term is computed; the fraction is that of the last two orders, as in 3D);
  `fullwave.background` {`type`, `form`, `physical_volume_outside_grid_fraction`, `grid_edge_A_over_max`, `exterior`:
  quadrature, continuation and sliver diagnostics} and `fullwave.exterior_term_background` describe it, with the 3D
  engine's labels and warnings, including a grid that ends inside the domain sphere (docs/physics/fullwave_background.md,
  "The axisymmetric engine"). Only a sweep reports these, not an eigenmode-only run.
  With a uniform background the background's integral diverges (A_φ ∝ ρ), so its exterior term is omitted, with a
  warning and `exterior_term_background = "omitted (uniform background: ...)"`; the static lab field keeps its term
  (`exterior_term = "included"`, `Psi_exterior`: the design's windings and superconductor reactions), as in 3D.
  Ψ(ω) then depends on the domain radius above the
  MQS limit: between R1 and 1.5 R1 it moved by 0.7–44 % at kR = 0.4 and by up to 159 % at kR = 0.8 in
  docs/validation/axi_background_exterior.md. Vary `mesh.domain_radius` to see it, or give the magnet's field as a
  grid.
- `Z_re_ohm`, `Z_im_ohm`: readout impedance Z = R − iωL(ω); `L_H` = −Im Z/ω,
  `L_over_MQS`; `R_rad_ohm` the radiation resistance and `Q_rad` = ωL/R.
- `F_SI_T4_H2_m8` = |Ψ|⁴/L(ω)² (omitted above the first self-resonance, where
  L(ω) < 0, with a warning).
- `radiated_W` (from the exterior map) and `power_balance_rel` (|½ Re Z − P_rad|
  relative; round-off level).
- `bulk_losses` per lossy material (power, resistance, Q) with `R_bulk_ohm`, `Q_bulk`; `R_rad_ohm` and
  `Q_rad` are then the radiation part of Re Z alone.
- `surface_Gamma` per conductor body: Γ = ∫|H|² dA / I² with the full-wave
  field. With `rf` loss data on the materials: `R_loss_ohm` = Σ R_s(f) Γ,
  `Q_loss` = ωL/R_loss, `Q_total` = ωL/(R_loss + R_rad) and the loss-weighted
  `T_over_Q_K` (perturbative in R_s).

These columns are the inputs of a **beyond-MQS reach curve**: `axionfem.circuit.fullwave_tables`
returns them as arrays, `circuit.pickup_coupling` and `circuit.pickup_impedance(model="fullwave")`
turn them into Ψ(ω) and Z_p(ω) for the readout circuit, and
`axionfem.scan.reach_fullwave(results, benchmark)` runs the optimal scan with Ψ(ω), L(ω) and
the tuned-pole L_eff = L + ½ dL/dln(ω). Independent losses add as
R_total = R_rad + R_bulk + R_surface + ω L_p/Q_other, where the legacy Q_floor
argument (or benchmark Q) specifies intrinsic MQS Q_other. Radiation may have
its own T_radiation_K; omitting it assumes the material bath temperature.
The local noise-bandwidth and slope checks must pass; otherwise use the exact
finite_band circuit route. Both reactive and effective inductances are named in the output. Frequencies where
L(ω) ≤ 0 are above the first self-resonance and are reported as unreachable rather than tuned.
See docs/validation/finite_band.md and docs/physics/readout_circuit.md Sec. 5.3.

`results.fullwave.eigenmodes.{TE,TM}.modes[]` lists resonances (`f_Hz`, `k_R`)
with the energy fraction by region, which tells a pickup resonance from a mode of
the surrounding sphere. The closed sphere stands for a shielded room; with
`boundary: radiation` the open structure's resonances appear in the sweep as
poles of L(ω).

Validity: the axisymmetric solver drives the readout uniformly around the axis (an ideal slit for
azimuthal pickups, an annular gap for ports). That is exact while the readout loop's circumference is
small compared with the wavelength. A real slit or gap fed at one point also excites non-axisymmetric
(m ≠ 0) currents: standing waves around the circumference set the first self-resonance (about 25 MHz for
the 1 m toroid of Benabou et al. 2022, where circumference ≈ λ/4–λ/2) and add radiation. Above roughly a
quarter of that frequency, use the axisymmetric results as the uniform-feed limit and check with a 3D
full-wave solve. The m = 0 radiation resistance itself (dominant at low frequency) is exact.

Modelling: superconductors and `conductor` materials are perfect conductors
(with perturbative RF surface loss when specified in 2D), iron uses the RF secant permeability as in the MQS
pickup, and every cavity sealed by conductors keeps zero RF flux. When
frequencies are given, the mesh is refined to about 12 nodes per wavelength
(order 2); `mesh_kh_max` reports the largest k·h and a warning flags coarse
meshes. Validation: docs/validation/fullwave.md.

### 3D full wave (`"symmetry": "3d"`)

Current ports and multiport impedance matrices are validated; prescribed sources, perfectly matched layers, the error
indicator, the Gauss-law guard, the adaptive sweep, the tree-cotree solve and the 3D eigenmodes are beta. Every 3D
full-wave output block carries `"status": "provisional"`, a fixed label rather than a per-row verdict (acceptance is
`validation_accepted`, `refused` and the warnings); a row or diagnostic rejected by a solver or validation gate
carries `"status": "provisional_rejected"`.

The RF-only quarter/PEC-hole path (beta) and named electric/magnetic
boundaries are documented in
[boundaries and solvers](validation/fullwave_stage2a_boundaries_solver.md).
All-column energy, source controls and reactive validation loads are described
in [energy identity](validation/fullwave_stage2a_energy.md).

The same `fullwave` section works for 3D designs with a `port` pickup (e.g. `to_3d(design, pickup="slit")`):
the readout current is fed through the port region at its real location, so non-axisymmetric current
patterns and the self-resonances of a slit appear (the axisymmetric solver's uniform-feed limit does not
have them). Formulation: complex Nédélec E field, `curl(ν curl E) − ω² ε_c E = iω J_port`, perfect
conductors eliminated. The radiation exterior uses a static shell capped at F R1 (default F=3),
whose map endpoint is constrained to equal that cap. A low-rank vector-spherical-harmonic update
subtracts its static response and inserts the outgoing response: TE coefficient
(γ_n − st_n)/(μ0 R1), with st_n = [n + (n+1)q]/(1−q), q = F^{−(2n+1)};
TM coefficient −ω²ε0R1/γ_n. Up to `dtn_modes` (at most 40, normally about kR+10)
are retained. The continuum exterior identity still has finite-element and multipole-truncation error;
GMRES uses block AMS plus the exterior correction. `boundary: pec` closes the domain sphere.
Results per frequency include `Psi_*`, `Z_*`, `L_H`, `R_rad_ohm`, `Q_rad`, `F_*`, bulk losses and
true GMRES residuals. `fullwave.self_resonances` identifies sign changes of L(ω) between sweep points (an adaptive
sweep reports its model's poles instead, below);
`modes` gives closed-sphere eigenmodes with residual, mass-orthogonality and completeness diagnostics.
In 3D iron uses its linear `mu_r`, without static-bias correction.

**Surface losses (3D).** Every sweep row and every `port_solutions` entry carries the perturbative
RF surface losses of the conductor and superconductor surfaces at that frequency
([physics/fullwave_surface_losses.md](physics/fullwave_surface_losses.md)). They are evaluated over the faces of
`pickup.losses` (Materials, above) from the full-wave field H = curl E/(iωμ) on the field side, per unit port current:

| key | meaning |
|---|---|
| `surface_Gamma` | Γ = ∮\|H\|² dA / I² per surface, dimensionless (H per ampere is in 1/m; the axisymmetric engine's key) |
| `surface_losses.surfaces` | per surface: `Gamma`, `Gamma_by_field_region`, `Gamma_magnetoresistance` (with a magnetoresistance model), and with `rf` data `R_s_ohm`, `R_eff_ohm` = R_s(f) Γ, `Q` = ωL/R_eff, `Q_tuned` = ωL_eff/R_eff, `T_K`, `T_over_Q_K` |
| `surface_losses.H_normal_share` | ∮\|H_n\|²/∮\|H\|²: zero to rounding on perfect conductor faces (a check) |
| `R_loss_ohm` | Σ R_eff over the surfaces with `rf` data: 2P_loss/I², the power the conductor surfaces dissipate (to first order the series resistance they add to Re Z) |
| `Q_loss`, `Q_total`, `T_over_Q_K` | ωL/R_loss (the reactance's Q), ωL/(R_loss + max(Re Z, 0)), Σ T R_eff/(ωL) over the rf surfaces (radiation and bulk loss are not in T/Q), with L = −Im Z/ω (as 2D). `Q_total` is withheld (`surface_losses.Q_total_withheld`) where the solve's true relative residual ρ times \|X\| (about what the residual can move Re Z by) exceeds 1 % of R_loss + max(Re Z, 0) |
| `Re_Z_unresolved` (row) | in a dissipating model, where Re Z < 100 ρ\|X\|: the residual cannot show Re Z, `R_rad_ohm` and `Q_rad` to be resolved (warned about; `R_bulk_ohm` and `R_boundary_ohm` are field integrals and unaffected). A tighter GMRES tolerance (`AXF_FW_RTOL`) lowers ρ where double precision allows |
| `L_eff_H` | L + (ω/2) dL/dω from `dZ_domega_H` when its scope is exact (else omitted, with the tuned Q's): the tuned pole's inductance (4 W_m for a closed lossless structure). The losses are post-processing and add nothing to A(ω) or dZ/dω |
| `Q_loss_tuned`, `Q_total_tuned`, `T_over_Q_tuned_K` | **the headline Q:** the same with L_eff, the Q of the readout tuned to f by an ideal capacitor (exact to first order; near a self-resonance it tends to that resonance's own Q, where `Q_loss` tends to 0) |
| `R_loss_over_abs_X` | R_loss/\|Im Z\|: the perturbation parameter; the neglected order is about its square |
| `surface_Gamma_matrix`, `R_loss_matrix_ohm` | multiport (primary row): the Hermitian ∮H_j*·H_k dA per surface and Σ R_s ∮H_j*·H_k dA, the dissipated power ½I†RI of any port currents; it is the impedance perturbation only for in-phase port fields (closed, lossless), so do not add it to Z with radiation or bulk loss |

- **Summary.** `fullwave.surface_losses` lists:
  - the surfaces (material, rf model, field-side regions), and face and point counts;
  - the surfaces without `rf` data;
  - the largest perturbation parameter (rows above 0.1 are warned about);
  - the rows with `Q_total` withheld (warned about);
  - the PEC boundary groups that carry no loss here (warned about when they are all the conductors);
  - the port rows whose solve failed its residual gate (their losses are not reliable);
  - the cost.
- **Scope.** The Q values are omitted where L ≤ 0 (above the first self-resonance). The loss is perturbative: the
  field of the perfectly conducting structure, valid for Q ≫ 1 and away from a resonance's width.
- **Not counted:** conductors meshed as PEC holes (`mesh.remove_pec`; warned about when they have `rf` data),
  imported PEC groups and a closed outer sphere. Faces to PML layers are left out with a warning.
- **Leaking readouts** (a partial port, an open slit) see their environment: a closed `pec` sphere shifts L by the
  leakage dipole's image, and `radiation` lets it radiate as a magnetic dipole (Q_rad ∝ f⁻³). Neither is the
  experiment's shield, cryostat or iron; mesh those as conductors with `rf` blocks for their loss
  ([physics/fullwave_surface_losses.md](physics/fullwave_surface_losses.md), Validation).
- **Magnetoresistance** weights R_s by the static field as `pickup.losses` does, with the factor R_s(f, B)/R_s(f)
  (it never evaluates R_s, so measured tables work); with `fullwave.rf_only` there is no static field, and R_s is taken
  at zero field (warning).
- **Adaptive sweeps** evaluate the losses at the solved frequencies (the rows) only; the model and its dense curve
  carry Z and Ψ.
- **Python:**
  - `axionfem.circuit.fullwave_losses(results, port)` (`Results.fullwave_losses()`) returns the columns as arrays;
    2D rows have no exact dZ/dω, so their tuned columns are NaN.
  - `circuit.fullwave_tables`, `pickup_impedance(model="fullwave")` and `scan.reach_fullwave` add `R_loss_ohm` to
    Re Z, for 3D sweeps as for 2D, so the reach's Q_other (the benchmark Q) must exclude the computed conductor loss.
    `fullwave_tables` warns where an unresolved Re Z can move R_ohm by more than 1 %.
  - `reach_fullwave` uses the rows' exact `L_eff_H` when every row has it (3D designs with conductors meshed as
    volumes, with or without `rf` data). Between the rows it takes a cubic Hermite interpolant of L with the exact
    slopes, which must agree with the linear interpolation of the rows' L_eff within 5 %, or the reach refuses. The
    MQS comparison keeps L_p.
  - `Results.summary()` prints the tuned Q, and marks R_rad and Q_rad where Re Z is unresolved.
  - `circuit.loss_interpolation_check` warns where R_loss interpolated log-log between rows is unreliable: a slope of
    R_loss/\|X\| above f³, or an estimated error above 5 % (against ωL_eff/Q_tuned with the rows' L_eff, else from the
    curvature of the neighbouring rows).

**Conductor-surface quadrature export (`fullwave.surface_quadrature_file`, 3D only; beta).** Writes, per frequency row and MPI rank, the quadrature points of the conductor PEC boundary
faces with the per-port magnetic field there, so that the sheet current K = n × H (and the Maxwell stress) can be
integrated over the conductor surfaces exactly, without point location or a probe offset.

| key | meaning |
|---|---|
| `surface_quadrature_file` | file prefix; the files are `<prefix>.f<frequency_index>.rank<rank>.bin` (the frequency index is the solve order, as the field export's) |
| `surface_quadrature_order` | integration order on each boundary face (default 2) |

- **Requires `fullwave.rf_only`**: only then can conductors be PEC boundary
  faces; in other 3D designs they are volume elements and the files came out empty. Zero exported points is an
  error; when only the outer wall (901) is exported (conductors kept as volumes) there is a warning.
- **Faces:** the boundary groups of type `pec` (a group without `type` is PEC), minus symmetry planes (groups with
  `axis`): the surfaces of conductors removed from the mesh (`mesh.remove_pec`), imported PEC groups, and the outer
  boundary (attribute 901, group `outer`), which in an `rf_only` model is a real conductor: the closed PEC sphere or
  the `mesh.domain_solid` wall. Filter by `attr` to keep the device's conductors only (the mesher's holes are
  attributes 1000 + body id). Faces whose field side is an infinity-shell element (`skipped_faces_shell`; no current
  `rf_only` mesh has a shell, so this is a guard), a PML element (`skipped_faces_pml`) or a meshed PEC element
  (`skipped_faces_pec_side`) are skipped, and so are PEC boundary elements on interior, shared or nonconforming faces
  (`skipped_faces_interior`, warned about: a two-sided PEC sheet is not exported). Conductors kept as volume
  elements, impedance and sheet walls are not exported.
- **File layout** (float64 in the writing machine's native byte order, little-endian on x86): `npts`, `nports`; then `npts` records (x, y, z, n_x, n_y, n_z, dA, attr) in
  m, with n the unit normal out of the conductor into the field region and dA the quadrature weight in m²; then
  `nports × npts` records (Re B_x, Re B_y, Re B_z, Im B_x, Im B_y, Im B_z) in T per incoming terminal ampere of that
  port's basis column, B = curl E/(iω) evaluated on the field-side element (the curl representative when the
  tree-cotree solver or the Gauss-law guard is on, as for `field_probes`). The union of the rank files is the surface.
- **Summary:** `fullwave.surface_quadrature` (`file_pattern`, `order`, `attributes`, `points`,
  `points_per_attribute`, `area_m2`, the skip counts); each accepted sweep row names its rank's file in
  `surface_quadrature_file`. A row's file is written after its last validation gate: a refused row leaves no file.
  The file's directory is created and a probe file written and removed on every rank at setup; a write failure is
  agreed across ranks, names the failing ranks, and every rank removes its own file of that row.
- **Not done:** under `mesh.symmetry` only the meshed part is written and B is not mirrored (no image scaling); the
  values are those of the row's solve, before any circuit loading; an adaptive sweep writes the accepted rows only
  (a refused row leaves no file).
- **Validation:** `tests/test_fullwave3d_surface_quadrature.py`: the shorted TEM line (groups, area, normals,
  n · B = 0, |B| = μ₀|I(x)|/W to < 1 % at order 2, one file per row, PML faces skipped) and a generated mesh with a
  PEC hole (a slit ring in the closed sphere: groups, areas, normals, n · B = 0).

**Adaptive sweep (`fullwave.adaptive`, 3D only, beta).** Instead of a frequency list, give a band and
a tolerance. The engine solves at a few frequencies and fits a rational model to Z and Ψ, then adds solves where
the model is least certain, until it is accurate over the band
([physics/adaptive_sweep.md](physics/adaptive_sweep.md)):

```json
"fullwave": {"boundary": "radiation",
             "adaptive": {"f_min_Hz": 7e7, "f_max_Hz": 1.4e8, "tolerance": 1e-3}}
```

| field | meaning |
|---|---|
| `f_min_Hz`, `f_max_Hz` | the band [Hz] (required). It excludes `frequencies` and `f_min`/`f_max`/`points` |
| `tolerance` | relative error of every Z entry and Ψ entry over the band (default 1e-3, range 1e-6-0.1). It is pointwise, with a floor of 1 % of the entry's typical magnitude (`Z[j][k]`: of √(\|Z_jj\| \|Z_kk\|); Ψ: of the largest entry), so entries near zero are held to an absolute error. Below 1e-6 the fit would chase the solves' own error (true residual 1e-10, gated at 1e-6) |
| `max_solves` | frequencies solved at most, the initial ones and refused ones included (default 30). If they are spent first: `converged: false` and a warning |
| `initial_points` | first solves, the band ends included (default 5, at least 3) |
| `spacing` | `log` (default) or `linear`: initial points, candidate grid and dense output |
| `dense_points` | points of the dense evaluation in the results (default 400) |

The mesh, the exterior map and the scene are set up for the band, as for a list that reaches `f_max_Hz` (the scene
frequency is the initial point nearest `scene_frequency`). The sweep stops when the error estimate is below the
tolerance at every candidate frequency, each of the two newest solves agreed with the model fitted before it, and
the fit reproduces its own solves. A frequency that cannot be solved (a validation gate, an unaccepted solve) is
recorded in `refused` and skipped; if every initial point fails, the sweep stops unconverged with no model. A sweep
that ends unconverged (`max_solves` spent) warns and sets `converged: false`.

Every solve is a regular sweep row in `fullwave.sweep`, sorted by frequency. `fullwave.adaptive` holds the model
(support points, weights, values, poles), the estimate's history, refused frequencies and a dense evaluation of
Z, Ψ and dZ/dω over the band (`axionfem.circuit.fullwave_dense`, `fullwave_model`; `Results.write_touchstone(path,
dense=True)` writes it as a Touchstone file, docs/physics/touchstone.md). `fullwave.adaptive.poles` lists
every in-band pole of the model with its uncertainty (leave-one-out and noise refits) and a class: `unresolved`,
`non_passive` (significantly above the real axis), `negligible` (changes no output by the tolerance), `background`
(Q < 2) or `resonance`. Q is given only where the loss is resolved, with its uncertainty; otherwise the pole is
`lossless_within_fit` with a 3σ lower bound `Q_min`. `fullwave.self_resonances` then lists the resonances of a
converged sweep (`f_Hz`, `Q`, residue, the solves around them), not sign changes of L, and nothing for an unconverged
one; `fullwave.self_resonances_determined` tells the two apart. `fullwave.adaptive.noise_bound` (with a warning) says
that the fit tolerance (1e-3 × `tolerance`) is below 10 × the solves' relative residual, so that the stop is
unreliable; raise `tolerance` or tighten the solves.

The exterior term of Ψ(ω) follows the continuum identity for a decaying laboratory field:
B_lab = −∇Φ, ∇·E = 0, and ∫_{r>R1} E·B_lab dV = ∮_{R1} Φ E_r dA.
`Psi_exterior` evaluates it from the static multipoles and finite-element traces. It is the exterior part of the
primary form (1/(iωI)) ∫ E·B_lab dV, which does not vanish as ω → 0; the axisymmetric engine's `Psi_exterior` is the
exterior part of the reciprocal form instead, so the two are different parts of the same Ψ.

An imposed `background` (uniform or grid) is coupled too: A_lab then holds only the reaction
field, and the background enters as Ψ_bg(ω) = (1/(iωI)) ∫ curl E·A_bg dV with A_bg = (ψ/r) φ̂, over the domain
sphere and, with `boundary: "radiation"`, outside it (the TM m = 0 exterior multipoles of the readout field
against A_bg: the grid up to the sphere inscribed in its box, the grid's exterior multipoles beyond, so the grid
should extend past R1 and its inscribed sphere enclose the magnet; the domain centre must lie on the z axis). Each
sweep row reports `Psi_background` {`volume`, `exterior`, `exterior_last_order_fraction`} (both included in Ψ;
`exterior` null where no exterior term is computed; the fraction is the last two orders' share of the term, since a
magnet symmetric about the domain centre has no even orders); `fullwave.background` and
`fullwave.exterior_term_background` describe the coupling. `Psi_over_MQS` divides by the 3D MQS Ψ, which with a
background is its reciprocal form ∫(A_lab + A_bg)·B_pick (see "3D designs" below), the low-frequency limit of Ψ(ω). A
uniform background fills all space, so its exterior integral diverges and is omitted with a warning (Ψ(ω) then depends
on the domain radius). With `boundary: "pec"` nothing is omitted, but the model is then a cavity of radius R1, not
open space. With PML layers the background is coupled in the physical region only (label and warning). A grid that
ends inside the model with A_bg still large at its edge is warned about. A background cannot be combined with
`fullwave.source` or `fullwave.rf_only` (refused). Formulation:
[physics/fullwave_background.md](physics/fullwave_background.md); validation:
[validation/fullwave3d_background.md](validation/fullwave3d_background.md). Other validation:
docs/validation/fullwave3d.md.

## Cavity haloscopes (`cavity`, axisymmetric; beta)

Resonant-cavity post-processing of the `fullwave` eigenmodes (m = 0: TM0np with azimuthal H, TE0np with
azimuthal E). Beta; validation: [validation/cavity_2d.md](validation/cavity_2d.md).

```json
"materials": {"cu": {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.7e-8, "T_K": 4}},
              "sapphire": {"kind": "linear", "eps_r": 9.4, "tan_delta": 1e-7}},
"background": {"type": "uniform", "Bz": 8.0},
"fullwave": {"modes": 6, "families": ["TM", "TE"], "boundary": "pec", "target_f_Hz": 5e9},
"cavity": {"interior_point": [0.05, 0.0], "form_factor": {"B0": "applied"}, "losses": true}
```

| field | meaning |
|---|---|
| `interior_point` | [r, z] inside the cavity (required). The connected non-conductor component containing it (not crossing perfectly conducting sheets) is the cavity; every other non-conductor element is left out of the full-wave problems, so the room between the wall and the closed outer sphere adds no modes. The cavity must be enclosed by conductors (error otherwise). A point exactly on the wall surface is accepted (the adjacent cavity element is used, with a warning); it only selects the cavity and does not enter C |
| `form_factor.B0` | static field of C: `"applied"` (default): background and windings, with superconducting regions and sheets that bound the cavity treated as transparent (re-solved as `conductor` on the same mesh); `"static"`: the full static solution, in which closed superconducting walls expel B0 (Meissner) |
| `form_factor.B_ref_point` | [r, z] where \|B0\| normalizes C (optional; default: the volume rms of B0 over the cavity, so C = C_rms) |
| `losses` | per-mode wall and dielectric losses (default true) |
| `allow_lossless_walls` | walls without `rf` data are lossless on purpose (default false; same meaning in 2D, xsec and 3D): no warning, `losses.walls_lossless_by_request`, and Q0 = Q_dielectric when no wall has rf data. Without it every wall lacking rf data is listed (`losses.lossless_walls`) and warned, and Q0 is omitted (Q_dielectric kept) when no wall has rf data. The `axionfem.cavity` builders set it when `wall_material` is not given |

Requirements: `fullwave.modes > 0`; no `fullwave.frequencies` (a sweep in a cavity design is rejected for now);
a static field for C: a `background` and/or windings. Walls are `conductor` or `superconductor` regions (perfect
at RF), with an optional `rf` model for the losses; dielectrics are `linear` regions with `eps_r`, `tan_delta`,
`sigma_S_m`.

Which B0: in a haloscope the relevant field is the magnet field that penetrates the cavity. Normal (copper) walls
are transparent to static fields. A superconducting cavity in a multi-tesla magnet is far above H_c1 (mixed state,
or a thin film on copper), so it does not expel the field either; the default `B0 = "applied"` therefore treats
the superconducting regions/sheets that bound the cavity as transparent (`fullwave.cavity.B0_transparent_walls`
lists them). `"static"` keeps the Meissner solution of the static problem (closed superconducting walls then give
B0 ≈ 0 inside). When the cavity rms of B0 is below 1e-6 of the largest \|B0\| in the design, C and C_rms
would be round-off: they are omitted, `fullwave.cavity.B0_negligible` is true and a warning is issued.

Conducting sheets (`curves`) inside the cavity, with the cavity on both sides (irises, baffles), are supported:
TM: H_φ is split across the sheet (separate DOFs per side, natural n × E = 0 on each face; joined at a free edge);
TE: A_φ is tied to the sheet, and each face's current is recovered separately (see `losses`). The TE wall loss
of a zero-thickness free edge diverges logarithmically under refinement wherever the mode's field does not
vanish there, so a warning is issued (`fullwave.cavity.sheets_with_free_edge`); use a thin region with a finite
edge for converged Q of such modes.

Results: `fullwave.cavity` = {`regions`, `volume_m3` (V: the cavity volume, conductors excluded, dielectrics
included), `excluded_components`, `B0_field`, `B_ref` (how B_ref is defined), `B_ref_point` (if given),
`B_ref_T`, `B0_rms_T`, `B0_max_cavity_T`, `B0_max_design_T`, `B0_negligible`, `status`, and when relevant
`B0_transparent_walls`, `two_sided_sheets`, `TM_split_dofs`, `sheets_with_free_edge`}; each mode of
`fullwave.eigenmodes.{TM,TE}.modes[]` adds `residual_true` (‖Kx − λMx‖/‖Kx‖) and

- `form_factor`: `C` = |∫E·B0 dV|² / (B_ref² V ∫ε_r|E|² dV) (ε-weighted, Sikivie/Al Kenany form), `C_rms`
  with ∫|B0|² dV in place of B_ref² V (Jeong 2018 form; equal for a uniform field), `V_m3`,
  `overlap_EB_T_m32` = |∫E·B0|/(∫ε_r|E|²)^{1/2}, `B_ref_T`. TM modes couple to (B_r, B_z), TE modes (E_φ) to a
  toroidal B_φ only.
- `losses` (perturbative in R_s and tan δ, the lossless mode): `walls` per conductor body {`G_ohm` =
  ω∫μ|H|² dV / ∮|H_t|² dA, `Rs_ohm` at the mode frequency, `Rs_model`, `Rs_flux_flow_ohm`, `Rs_flux_flow_perpendicular_ohm`, `flux_flow_orientation_factor` (with a flux_flow term), `Q` = G/R_s, `P_frac`, `T_K`}, `G_ohm` (all walls,
  the geometry factor Q0·R_s for one R_s), `Q_wall`, `dielectric` per lossy region {`tan_delta_eff` = tan δ +
  σ/(ωε'), `electric_energy_fraction`, `Q`}, `Q_dielectric`, `Q0` (1/Q0 = 1/Q_wall + 1/Q_dielectric) and
  `T_over_Q_K` (Σ T_b/Q_b). Walls without `rf` data are lossless (G is still reported) and listed in
  `lossless_walls`; unless `allow_lossless_walls` is set a warning says that Q_wall, Q0 and P_frac omit them, and
  when no wall has rf data Q0 is not reported (it would be Q_dielectric alone). For TE
  modes the wall current is recovered from the weak reaction on the conductor DOFs (consistent flux, converges
  like the eigenvalue), per face of a two-sided sheet (the reaction of each side's elements); at the free edge of
  a sheet, where the sides join, the rows use the pointwise current of each face. `G_ohm_pointwise` is the direct
  derivative, an independent check that converges at second order in h.
- `peak_wall_fields` (with `losses`; beta): the 3D solver's definition on the meridional
  section — `E_max` (largest normal |E|) and `B_t_max` (largest tangential |B|) over the Gauss points of the cavity's
  wall faces, evaluated in the cavity element (never at mesh vertices), `B_rms` over the cavity volume (μ_r = 1),
  `E_max_face_rms` / `B_t_max_face_rms` (largest face rms), `E_t_max` (largest tangential |E|: zero on a PEC wall
  in the limit; for TM modes the wall condition is natural, so it is a resolution check), `E_max_at_rz`,
  `B_t_max_at_rz`, `n_points`, and the ratios `Bpk_over_Brms`, `cEpk_over_Brms` = E_max/(c B_rms),
  `Bpk_face_rms_over_Brms`, `cEpk_face_rms_over_Brms`, `cEt_max_over_Brms`. The fields are in eigenvector units:
  only the ratios are meaningful.

`Results.cavity_modes()` returns the modes as flat rows; `axionfem.cavity` has the pillbox and coaxial builders
and the closed forms used by the tests.

Mode fields (Python, from the solved model; no design keys): `Results.cavity_mode_field(i)` gives E and H of mode
`i` (a row index of `cavity_modes()`, or `family=` and the index within the family) on the cavity mesh,
`cavity_mode_sample(i, r, z)` / `cavity_mode_probe(i, r=(r0, r1, n), z=(z0, z1, n))` at points (NaN outside the
cavity), `save_cavity_modes(dir)` one `.npz` per mode, `mode_scene(i)` / `plot_mode(i)` / `png(mode=i)` a rendered
meridional section (quantities `Emag`, `Hmag`, `Er`, `Ez`, `Ephi`, `Hr`, `Hz`, `Hphi`). Normalization
int eps_r |E|^2 dV = 1 over the cavity (E in m^-3/2; H the matching physical field, int mu_r |H|^2 dV = 1/Z0^2);
the arrays are real amplitudes with complex fields E_c = i E, H_c = H (e^{-i omega t}), sign fixed by the largest
E component at max |E| (a convention; it may differ between meshes when the extrema are mirror images).
TM: (E_r, E_z, H_phi); TE: (E_phi, H_r, H_z). Accuracy: docs/validation/cavity_2d.md.

## Cross-section cavities (`"symmetry": "xsec"`; beta)

A z-uniform cavity (posts, vanes, partitions and dielectric rods spanning the full length, no end gaps) has
z-independent modes that solve a scalar problem on the (x, y) cross-section (formulation
`cpp/include/axionfem/xsec.hpp`, validation [validation/cavity_xsec.md](validation/cavity_xsec.md)):

- **TM_z** (E = E_z z^): −∇·(μ_r⁻¹∇E_z) = k₀² ε_r E_z, E_z = 0 on every conductor. ε_r enters the mass matrix
  (E_z and μ_r⁻¹∂_nE_z continuous at dielectric interfaces). With flat end caps these are exact 3D modes (TM_mn0)
  for any length: the modes that couple to an axial B0.
- **TE_z** (H = H_z z^): −∇·(ε_r⁻¹∇H_z) = k₀² μ_r H_z, natural condition on conductors, H_z split across a sheet
  with the cavity on both sides. Cutoff modes of the guide: C = 0, and they do not satisfy end caps (they need z
  dependence). Optional, for crossings within the 2D model.

Modes with z dependence (TM_mnp/TE_mnp with p ≥ 1, hybrid modes of dielectric-loaded guides) are **not** captured
(3D solver). For a vacuum cross-section `axionfem.cavity_xsec.z_dependent_spectrum` lists them (separable).

```json
{"schema": "axionfem.design/1", "name": "two_rods", "symmetry": "xsec",
 "materials": {"cu": {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.72e-8, "T_K": 300}}},
 "regions": [{"name": "cavity", "shape": {"disk": {"center": [0, 0], "radius": 0.05}}},
             {"name": "rod_a", "shape": {"disk": {"pivot": [0.02, 0], "arm": 0.01, "angle_deg": 30, "radius": 0.006}},
              "material": "cu"},
             {"name": "post", "shape": {"disk": {"center": [-0.02, 0.0], "radius": 0.004}}, "material": "cu"}],
 "fullwave": {"modes": 6, "families": ["TM"], "target_f_Hz": 4.6e9},
 "cavity": {"wall_material": "cu", "length": 0.25},
 "mesh": {"order": 2, "size": 0.002, "surface_size": 0.001}}
```

Geometry: the regions' shapes live in the (x, y) plane, same shape language as the axisymmetric designs (rect,
frame, polygon, disk, union, difference; no r ≥ 0 restriction). `rect`/`frame` take `"x"`, `"y"` (or `"r"`, `"z"`);
a `disk` is placed by `center` [x, y] or by `pivot` [x, y] (default [0, 0]), `arm` and `angle_deg` (centre = pivot
+ arm (cos, sin)(angle), angle from +x towards +y): a tuning rod on a rotating arm. The meshed domain is the union of
the regions; its boundary is the cavity wall (material `cavity.wall_material`). Conductor/superconductor regions
inside are PEC posts and vanes; `curves` are zero-thickness PEC partitions (E_z = 0 on them; TE_z split). Linear
regions are dielectrics (`eps_r`, `mu_r`, `tan_delta`, `sigma_S_m`). Iron is rejected.

| key | meaning |
|---|---|
| `fullwave.modes` | number of modes per family (required, > 0) |
| `fullwave.families` | `["TM"]` (default), `["TM", "TE"]` |
| `fullwave.target_f_Hz` | modes nearest this frequency (shift-invert), as in the axisymmetric solver |
| `cavity.interior_point` | [x, y] in the cavity; required only when conductors or sheets split the cross-section into several cells |
| `cavity.wall_material` | conductor material of the domain boundary (its `rf` gives R_s); absent: lossless wall (warned). With `losses`, a material without `rf` is rejected unless `allow_lossless_walls` |
| `cavity.length` | L [m]: adds the two flat end caps to the TM_z losses and V = A L |
| `cavity.end_cap_material` | material of the caps (default `wall_material`) |
| `cavity.losses` | default true |
| `cavity.allow_lossless_walls` | as in the axisymmetric section (default false; `cylinder_rods_design`/`sector_cell_design` set it without `wall_rf`) |
| `background` | optional `{"type": "uniform", "Bz": …}`: \|B0\| for magnetoresistance / flux-flow R_s (C does not depend on it) |
| `mesh` | `order` (FE order, default 2; geometry order ≤ 2), `size` (default extent/30), `surface_size` (walls, post and dielectric boundaries, sheets; default size/2), `refinements` |

Not allowed in a cross-section design: windings, circuits, superconductors, pickup, hts, electrostatics, mechanics,
thermal, benchmarks, scan, cost, mesh.adapt/domain_radius, fullwave sweeps, `cavity.form_factor`.

Results (`results.json`, `axionfem.cavity_xsec.XsecResults`): `fullwave.eigenmodes.{TM,TE}.modes[]` with `f_Hz`,
`residual_true`, `energy_balance` (μ0∫μ_r|H|²/(ε0∫ε_r|E|²) − 1, a check), `energy_fraction` per region and

- `form_factor`: `C` = |∫E_z dA|² / (A ∫ε_r|E_z|² dA) for a uniform axial B0 (the length cancels; A = cavity area,
  conductors excluded, dielectrics included), `C_full` with the area enclosed by the outer wall (free-standing posts included;
  the "full cylinder" convention; a post touching or overlapping the wall is part of the outer boundary and is
  excluded: reported in `fullwave.cavity.outer_boundary_bodies` with a warning), `A_m2`, `A_full_m2`, `overlap_Ez_m`, `V_m3` (= A L with a length). TE_z: C = 0.
- `losses`: `walls` per body (`wall` = the domain boundary, conductor region names, curve names) with `G_ohm` =
  ω∫μ|H|²dA / ∮|H_t|²dl (per unit length), `G_ohm_pointwise` (TM: direct derivative, a second-order check of the
  consistent-flux value), `Rs_ohm`, `Q`, `P_frac`; `end_caps` (TM_z with a length: G = ωL∫μ|H|²dA / (2∫|H|²dA),
  = ωμ0L/2 for μ_r = 1); `G_ohm` (all surfaces incl. caps, one R_s), `G_ohm_2d` (barrel only); `Q_wall`, `Q_wall_2d`;
  `dielectric`, `Q_dielectric`; `Q0` (with caps), `Q0_2d` (per unit length); `T_over_Q_K`.
- `fullwave.cavity`: `area_m2`, `area_full_m2`, `walls` (perimeter per body), `regions`, `components`,
  `two_sided_sheets`, `sheets_with_free_edge` (singular field at a free sheet edge: warned),
  `outer_boundary_bodies` (when posts touch the wall).

Python: `Design.solve()` returns `XsecResults` with `cavity_modes()`, `cavity_mode_field(i)` (x, y, E_x..H_z;
normalization ∫ε_r|E|²dA = 1, E in m⁻¹; E_c = iE, H_c = H), `cavity_mode_sample(i, x, y)`,
`cavity_mode_probe(i, x=(x0, x1, n), y=(...))`, `save_cavity_modes()`, `plot_mode(i, quantity="Emag")`,
`geometry()`. Builders and sweeps in `axionfem.cavity_xsec`: `cylinder_rods_design`, `sector_cell_design`,
`xsec_design`, `sweep`, `rod_angle_sweep`, `plot_mode_map`, closed forms (`disk_tm_modes`, `disk_te_modes`,
`coax_tm_G2d`, `sector_cell_mode`, `eccentric_rod_tm_mode`, `eccentric_dielectric_rod_tm_mode`). `sweep(...,
track="C" | "overlap" | "nearest")`: the largest-C mode per step (default), the mode with the largest E_z overlap
with the previous step (robust at crossings), or the nearest frequency (fragile at crossings). With flux-flow walls
in an axial B0 the barrel TM_z currents are parallel to the vortices (orientation factor 0) and the end-cap currents
perpendicular (1). Near-degenerate pairs (e.g. two cells nearly separated by a post) mix: their individual C are not
meaningful, their sum is.

## Cavity haloscopes in 3D (`cavity` in a `"symmetry": "3d"` design; beta)

The same `cavity` section post-processes the 3D eigenmodes (all azimuthal orders: TE111, TM110, hybrid and
rod-perturbed modes). Beta; validation: [validation/cavity_3d.md](validation/cavity_3d.md).

```json
"materials": {"cu": {"kind": "conductor", "rf": {"model": "skin", "rho_ohm_m": 1.7e-8, "T_K": 4}}},
"regions": [{"name": "cavity", "material": "vacuum", "solid": {"cylinder": {"base": [0, 0, -0.1], "axis": [0, 0, 0.2], "radius": 0.1}}},
            {"name": "rod", "material": "cu", "solid": {"cylinder": {"base": [0.04, 0, -0.1], "axis": [0, 0, 0.2], "radius": 0.01}}}],
"mesh": {"order": 2, "size": 0.01, "domain_solid": {"cylinder": {"base": [0, 0, -0.1], "axis": [0, 0, 0.2], "radius": 0.1}},
         "remove_pec": true, "symmetry": [{"name": "sym_y", "axis": "y", "type": "pmc"}]},
"fullwave": {"rf_only": true, "boundary": "pec", "modes": 8, "target_f_Hz": 2.0e9,
             "eigensolver": {"method": "shift_invert"}, "source": {"kind": "uniform", "B_T": [0, 0, 8]}},
"cavity": {"interior_point": [0.05, 0.03, 0.001], "wall_material": "cu", "losses": true, "mode_overlaps": true}
```

| field | meaning |
|---|---|
| `fullwave.rf_only` | required: a closed PEC model without a static solve. Without a pickup it is allowed for eigenmodes only (`modes > 0`, no `frequencies`). The wall is `mesh.domain_solid` (its boundary, group `outer`, is PEC) or the closed sphere; rods and vanes are conductor regions, best as PEC holes (`mesh.remove_pec`) |
| `fullwave.target_f_Hz` | modes nearest this frequency (shift-invert); omitted: the lowest modes (AME) |
| `fullwave.eigensolver` | `{"method": "ame" \| "shift_invert", "max_iter": 500, "tol": 1e-8, "max_steps": 0, "completeness_steps": 20}`: `ame` = hypre AME (LOBPCG + AMS; `max_iter`, `tol`), `shift_invert` = STRUMPACK factorization of A − σM, σ = (2π f_t/c)², and an M-inner-product Lanczos with full reorthogonalization (`max_steps`: cap on Lanczos steps, 0 = automatic; `completeness_steps`: length of the completeness check below, 0 = off). Default: `shift_invert` with `target_f_Hz`, else `ame`. A STRUMPACK build is needed for `shift_invert` |
| `fullwave.source` | the applied static field B0 for C (`uniform` or filament `segments`, ...): the magnet's field as if the cavity were absent (copper and superconducting walls alike; no Meissner screening) |
| `cavity.interior_point` | [x, y, z] inside the cavity (required). The connected non-conductor component containing it is the cavity; other non-conductor components are closed off as PEC |
| `cavity.form_factor.B_ref_point` | [x, y, z] where \|B0\| normalizes C (default: none, B_ref = the cavity volume rms of B0, so C = C_rms; with a point, `B_ref_over_B0_rms` is reported and a > 1 % difference warned) |
| `cavity.wall_material` | material (with `rf` data) of the `outer` boundary (the domain wall); conductor regions and holes use their own material. With `losses`, a material without `rf` is rejected unless `allow_lossless_walls` |
| `cavity.allow_lossless_walls` | walls without `rf` data are lossless on purpose (default false): no warning, `losses.walls_lossless_by_request`, and Q0 = Q_dielectric when no wall has rf data. Without it every wall body lacking rf data is warned and listed (`losses.lossless_walls`), and Q0 is omitted (Q_dielectric kept) when no wall has rf data |
| `cavity.losses` | per-mode wall and dielectric losses (default true) |
| `cavity.mode_overlaps` | η matrix between the returned modes (default false) |

AME returns the lowest modes; a conductor that touches no wall (a rod with gaps at both end caps) adds a static
λ ≈ 0 field per floating body that AME does not deflate (the gate then fails): use `shift_invert`, which
deflates discrete gradients and discards λ ≈ 0 Ritz values (`kernel_ritz_values_discarded`).

Completeness of a shift-invert window is checked, not certified (the STRUMPACK inertia is not usable here):
after convergence a short Lanczos run (`completeness_steps`) on the M-orthogonal complement of the returned modes,
from a fresh start vector, looks for Ritz values inside the returned window [σ − d, σ + d]
(`solver.completeness_check`): `missing_in_window` > 0 fails the gate and warns (a member of an exactly degenerate
cluster whose round-off seed had not grown); `near_degenerate_outside_window` > 0 warns that the window edge cuts a
near-degenerate pair (request more modes). Ritz values approach from outside the window, so a clean check is
evidence, not proof.

Floating tuning rods converge slowly: the rod tips (270° edges) and the gap capacitance need local refinement
(at h = a/5 with no local refinement C is a few % low; docs/validation/cavity_3d.md, "Floating tuning rod"). Give
the rod region a `surface_size` and small vacuum regions with `mesh_size` around the gaps
(`axionfem.cavity.cylinder_design_3d(rod={..., "gap_size": s})` builds them), and watch the indicator below.

Results: `fullwave.cavity` = {`interior_point`, `regions`, `excluded_components`, `volume_m3` (meshed volume),
`boundaries` (boundary groups touching the cavity), `symmetry`, `B_ref` ("volume_rms" or "point"), `B_ref_point`, `B_ref_T`, `B0_rms_T`, `B_ref_over_B0_rms`, `source`, `interior_sheets`};
`fullwave.eigenmodes` = {`modes`, `requested`, `returned`, `converged` (count, residual 1e-4, M-orthogonality
1e-4 and, for shift-invert, Lanczos convergence and no mode missing from the window by the completeness check), `solver` (method, steps, timings, Ritz estimates),
`target_f_Hz`, `overlaps`}. Each mode: `f_Hz`, `residual_true` (‖Ax − λMx‖/‖Ax‖), `rayleigh_ratio`
(xᵀAx/(λ xᵀMx)), `divergence_fraction` (M-norm² share of the discrete-gradient part), `energy_fraction`,
`E_azimuthal_fraction` (∫ε_r|E_φ|²/∫ε_r|E|² with φ about the z axis through the origin: 0 for TM0np-like and 1 for
TE0np-like modes of a cavity axisymmetric about z; identifies m = 0 modes, e.g. the SRF pump/signal pair, docs/srf),
`peak_wall_fields` (with `losses`; beta: `E_max` and `B_t_max` = largest |E| and |curl E|_t/ω over the wall
quadrature points, `B_t_max_flux` from the consistent-flux current, `B_rms` = (∫|curl E|²/(ω² V))^{1/2} with μ_r = 1,
and the ratios `Bpk_over_Brms`, `Bpk_flux_over_Brms`, `cEpk_over_Brms` = E_max/(c B_rms); pointwise peaks converge
slower than G, and are sensitive to faceted (polygonal) walls; `*_face_rms`: the largest face rms, robust to single
quadrature-point outliers; `E_max_at`, `B_t_max_at`: locations), `probes` (when `fullwave.field_probes` is given:
`E` and `B` = curl E/ω of the mode at those points, E normalized as above, the factor 1/i of B dropped;
`fullwave.eigenmodes.field_probes` lists the points and `valid` flags; points outside the cavity carry zeros. Sample
strictly inside: a point on a face shared with a removed/closed element may be flagged invalid), and

- `form_factor`: as in 2D (`C`, `C_rms`, `V_m3`, `overlap_EB_T_m32`, `B_ref_T`), E normalized to ∫ε_r|E|² = 1.
  Under `mesh.symmetry` the overlap is summed over the mirror images (E(Rx) = ±R E(x): + for PMC planes, − for
  PEC planes), so C refers to the full cavity; `V_m3` is the meshed (reduced) volume.
- `losses`: as in 2D, per wall body (`outer`, conductor regions/holes by region name), with the same R_s models
  (`spec::RFLoss`: skin, anomalous, bcs, power, table; the builtin `copper` is anomalous) and the same per-wall
  keys (`Rs_model`; with a flux_flow term `Rs_flux_flow_ohm`, `Rs_flux_flow_perpendicular_ohm`,
  `flux_flow_orientation_factor`, the orientation factor sin²(J_rf, B0) evaluated per wall point with J along the
  consistent-flux surface current and B0 = `fullwave.source`). The model warnings are emitted once per material. H_t comes from the
  consistent-flux surface current k = n × (curl E/μ_r) on the PEC boundary faces and on the faces between the
  cavity and a retained (not removed) conductor volume (weak reaction (A − λM)x solved with the tangential
  Nédélec trace mass matrix of those faces); `G_ohm_pointwise` is the pointwise curl (a check, second order away
  from edges). `G_flux_pointwise_disagreement` = |G/G_pointwise − 1| per body is an a-posteriori resolution
  indicator (the two converge from opposite sides near edges): above 0.05 the body gets `resolution_warning` and
  the run a warning. Symmetry planes carry no loss. PEC boundary faces with the cavity on both sides (a
  zero-thickness sheet, possible only with `mesh.file`) are rejected when `losses` is on (the reaction there is
  the net current of both sides) and warned otherwise (`cavity.interior_sheets`).
- `overlaps.eta[i][j]` = |∫E_i·B_j| / (∫|E_i|² ∫|B_j|²)^{1/2}, B_j = curl E_j/(iω_j), unweighted, rows E, columns
  B. Under `mesh.symmetry` all modes of one model share the mirror parities and the full-cavity η vanishes
  identically (E·B is a pseudoscalar); `eta_meshed_part` keeps the reduced-domain values. `overlaps.eta_energy`
  is the same overlap normalized by the stored energies, (∫ε_r|E_i|² ∫|B_j|²/μ_r)^{1/2}: equal to η in vacuum,
  the natural normalization with dielectrics.

`Results.cavity_modes()` returns the 3D modes with family "3D". Builders and references: `axionfem.cavity`
(`cylinder_design_3d`, `box_design_3d`, `cylinder_modes_3d` with mirror-parity subsets, `box_modes`,
`eta_tm010_te01p`, `dielectric_rod_tm_mode`, and the 2D floating-rod reference `floating_rod_design`). The 3D
builders' default wall is a perfect conductor without rf data with `allow_lossless_walls` set; pass
`wall_material` with an `rf` block for wall losses.

## Cavity tuning sweeps and mode tracking (`axionfem.cavity.sweep`; beta)

Tests: `tests/test_cavity_track.py`; validation: [validation/cavity_tracking.md](validation/cavity_tracking.md). No new design keys: a sweep solves one design per value of the tuning parameter and
compares the modes of neighbouring steps through **probes**, the mode's E sampled on a fixed point set:

| backend | probe source | components, weight |
|---|---|---|
| axisymmetric (`cavity`) | `Results.cavity_mode_sample` on an (r, z) grid after the solve | E_r, E_φ, E_z; ε_r r |
| cross-section (`"symmetry": "xsec"`) | `XsecResults.cavity_mode_sample` on an (x, y) grid | E_x, E_y, E_z; ε_r |
| 3D | `fullwave.field_probes` (set before the solve; each mode's `probes.E`) | E_x, E_y, E_z; 1 |

Probes outside the cavity at a step (in metal, in a removed rod, in the half removed by `mesh.symmetry`) are
invalid there; each overlap uses the probes valid at both steps.

```python
from axionfem import cavity as cv
T = cv.sweep(lambda th: build(th), thetas, workdir="runs/rod", probes=pts,          # build(value) -> Design
             solve_kw={"mpi_ranks": 16}, refine=3, min_step=0.25, track_kw={"degenerate_rtol": 1e-5})
print(T.summary())                     # branches, crossings, avoided crossings, ambiguous intervals
tm010 = T.branch("max_C", diabatic=True)  # character-following branch through the largest-C mode of step 0
rows = T.branch_table([tm010])         # value, f_Hz, C, Q0, overlap, degenerate, ambiguous per point
dia = T.diabatic()                     # avoided crossings resolved: character-following branches
T.plot("mode_map.png")
```

- Linking (`cv.track`): overlap O_ij = |⟨e_i, e'_j⟩_w| of probe vectors normalized on the common valid probes;
  degenerate clusters (relative splitting < `degenerate_rtol`, default 1e-6; use ~1e-5 in 3D, where the mesh splits
  degenerate pairs by 1e-5–1e-4) are orthonormalized on the probes and rotated inside their span to match the
  previous step (orthogonal Procrustes); Hungarian assignment on (1 − O) + `freq_weight` · min(1, |f − f_pred|/s)²,
  f_pred the linear extrapolation of the branch, s the median mode spacing. A link needs O ≥ `accept` (0.5); a best
  match with O ≤ `reject_floor` (0.2) is an end and a start (e.g. the shift-invert window moved). The interval is
  **ambiguous** when a link has O < `good` (0.9), a second overlap exceeds half the best (and 0.3), or a best match
  is rejected with `reject_floor` < O < `accept`; when the competitor is a near-degenerate partner (split <
  `doublet` (100) × `degenerate_rtol`) the interval is flagged `doublet_mixing` instead (refinement cannot help).
- Branches are **adiabatic** (continuous eigenvector). **Crossings** (`kind: "crossing"`): the frequency order of two
  branches changes with both overlaps near 1 (different symmetry, no interaction) — or an avoided crossing narrower
  than the interval. The order must be defined on both sides (nearest samples outside the doublet band and outside
  degenerate clusters); a crossing that falls on a degenerate sample offers both halves for refinement.
  **Avoided crossings** (`kind: "avoided"`): at each local gap minimum of a branch pair a window extends while the
  gap grows, up to `gap_factor` (3) × the minimum (`truncated` if the range ends first); overlaps across the window
  are *transported* (product of consecutive-step overlap matrices, re-orthonormalized at every step). Pairs where a
  branch loses its character to the other form candidates; candidates sharing a branch with overlapping windows and
  comparable gaps (within 10×) are clustered, together with the branches between their frequencies, and matched between the window ends by the best
  assignment of transported squared overlaps. A non-identity match with mean ≥ 0.5 (`exchange`, versus `stay` for
  the identity) is an avoided crossing: a swap of two branches or a `permutation` of several (multi-level).
  Spectators keep their character and map to themselves. The two-level fit gap² = s²(x − x0)² + (2κ)² on the three
  samples nearest the narrowest minimum gives `x0`, `gap_min_Hz` (= 2κ) and `kappa_Hz`. Doublets flagged as
  `doublet_mixing` produce no events. Gap minima at the sweep edge with a character change, and crossings wider than
  the range, are warned about.
- Limitation: an avoided crossing with 2κ below `doublet` × `degenerate_rtol` × f, sampled too coarsely for its
  links to stay above `good`, is reported as doublet mixing rather than refined; lower `doublet` or sample finer.
- `refine` rounds bisect ambiguous intervals, crossings (to tell a true crossing from a narrow avoided one; a
  crossing that survives down to `min_step` with overlaps ≈ 1 is true to that `resolution`) and avoided crossings with
  fewer than three fit points.
- Caching: each step lives in `<workdir>/step_<hash>/` (`track_step.npz`, `track_step.json`; the hash covers the
  design document (in 3D including the probes), in 2D the probe set, and `families`); a re-run reuses it. Sweep
  values may be Python or numpy numbers (sorted, refined); other values are kept in order and not refined. The default 2D probe grid (40 × 40) covers the
  union of the cavity boxes at the first and last values (`probes.json`, reused only for the same backend, `probe_n`
  and value range; the two box solves become cached steps); the default 3D set is
  `cv.probe_points_3d(design)` (a jittered lattice in the `mesh.domain_solid` cylinder or box).
- `cv.track(steps)` works on any list of `cv.ModeStep(value, modes, vectors, valid, weights)`, e.g. synthetic or
  imported mode sets; `cv.two_level_steps` builds the analytic two-level test case.

## Electrostatics (`electrostatics`, axisymmetric and 3D)

Capacitance matrices, parasitic and neck capacitance, and the readout
self-resonance. The section's presence enables the analysis; it solves
∇·(ε∇φ) = 0 — on the meridian half-plane for an axisymmetric design, on the 3D
mesh (H1 of `mesh.order`, MPI-parallel, hypre BoomerAMG + PCG) for a
`"symmetry": "3d"` design — with every conductor an equipotential:

```json
"electrostatics": {"boundary": "open", "notes": "parasitics of the pickup"}
```

| field | meaning |
|---|---|
| `boundary` | `open` (default): ground at infinity — in 2D the exact Dirichlet-to-Neumann map for the Laplace potential on the outer sphere, in 3D the infinity shell with the scalar radial map (see below); `grounded`: φ = 0 on the sphere at `mesh.domain_radius` |
| `notes` | free text, echoed in the results |

Conductors are the connected bodies of `superconductor` and `conductor`
materials, zero-thickness `curves` included (the same body detection as the
pickup problems); `materials[*].eps_r` sets the permittivity of every other
region. A design with an `electrostatics` section needs no magnet
(`examples/pickup_capacitance.json` is a pickup ring in a shield can).
`results.electrostatics` holds

- `bodies[]`: the conductors, in the order of the matrix rows;
- `capacitance_F`: the Maxwell matrix C_ij = Q_i for V_j = 1 V with every other
  conductor at 0 V (charges from the reactions), and `capacitance_to_ground_F`
  = Σ_j C_ij, the capacitance of each conductor to ground (infinity, or the
  outer sphere);
- `pairs[]`: for every pair, `C_mutual_others_grounded_F` = −C_ij (the lumped
  capacitor of the equivalent network) and `C_mutual_others_floating_F`, with
  the remaining conductors at zero net charge (an exact reduction of the
  matrix), plus their capacitances to ground;
- `checks`, which `checks.what_each_check_tests` also spells out in the
  results. **Algebraic identities** of the discrete system — they verify the
  assembly and the linear solves, not the discretization:
  `symmetry_relative` (the matrix is built from reactions, not from a
  symmetrized Schur complement), `energy_*` (½VᵀCV against ∫½ε|E|² by
  quadrature, for each column and for a mixed potential vector),
  `floating_solve` (a floating-conductor solve against the exact matrix
  reduction) and `exterior_monopole_*` — 2π·1ᵀKφ = Q_exterior holds
  identically, because the stiffness annihilates a constant and ∫P_n r dl =
  2R_b δ_{n0} on a sphere, so this only reports the solver residual and the
  geometric error of the boundary arc. **Discretization checks** — these are
  the ones to read, and they fall under h/p refinement: `charge_gauss_*`
  (reactions against the strong ∮εE·dA over the conductors) and
  `exterior_flux_*` (the net charge from the strong −∮ε∂φ/∂n dA over the outer
  sphere, which the weakly imposed DtN map reproduces only in the limit). On
  the isolated sphere at `surface_size` 0.03, the flux check falls
  3.3e-2 → 3.3e-4 → 3.1e-6 for order 1, 2, 3 while the capacitance error falls
  1.1e-3 → 1.1e-6 → 1.2e-8; the monopole identity is the same 2.4e-9 at orders
  2 and 3. Also reported: `diagonal_dominance` (C_ii ≥ Σ_{j≠i}|C_ij|, with
  equality for a fully screened conductor).

A region may not be named `air` or `infinity_shell`: those are the names the
mesher and the solvers give the parts of the domain that no region describes,
and a region of that name would be silently treated as vacuum.

Differences in 3D ([formulation](physics/electrostatics.md#3d),
[validation](validation/electrostatics3d.md)):

- the exterior is the **infinity shell** that the 3D mesher always builds,
  [R₁, R₂) mapped onto [R₁, ∞) by ρ′ = R₁(R₂−R₁)/(R₂−ρ); for the scalar
  potential this becomes the anisotropic permittivity
  ε_eff = ε₀[(f²/(f′ρ²)) rrᵀ + f′(I−rrᵀ)] with φ = 0 on the shell's outer
  sphere. This map contains the exact continuum monopole; Cartesian tetrahedra
  do not represent ρ affinely, so discrete monopoles and higher multipoles require h/p convergence;
- conductor bodies are the connected components of `superconductor`/`conductor`
  elements (3D designs have no `curves`); the Maxwell matrix costs one
  unit-potential Dirichlet solve per body and one BoomerAMG setup, and the
  charges come from the reactions Q_i = u_i·(Kφ) as a single global reduction, so
  the results do not depend on the rank count (verified to 1e-8 over 1, 4 and 8
  ranks);
- `checks.floating_solve` holds the exact matrix reduction with the floating
  charges recomputed by the surface-flux path (3D has no separate
  floating-conductor solver), and `checks.mixed_potential_solve` is an
  independent solve at a non-unit potential vector;
- `checks.exterior_monopole_*` compares the net charge with the flux through the
  sphere at `mesh.domain_radius`.

For a poloidal (port) pickup the readout gap is solved as well and reported in
`pickup.port_capacitance`, with no `electrostatics` section needed:

- `C_gap_port_region_F`: the capacitance of the port region alone, with natural
  (insulating) conditions on the faces open to the cavity — no fringing;
- `C_gap_F`: the two terminal faces driven at V = ±½ on the whole domain with
  the rest of their own conductor body insulating (n·D = 0) and every other
  conductor floating, which adds the fringing field at the mouths of the gap;
- `checks.dirichlet_bound`: the one rigorous relation between the two,
  C_region ≤ C_gap + ¼(C_A,ground + C_B,ground), from the Dirichlet principle
  (restricting the full-domain minimizer to the port subdomain is admissible
  for the region-only problem) and exact for the discrete problems. The
  stronger C_gap ≥ C_region is **not** proven: it holds on every case run so
  far, with C_gap/C_region falling towards 1 as the gap grows relative to the
  electrode spacing (1.02 at the widest gap tested, 5.6 at the narrowest), but
  the bound above is the only guaranteed one, and it is loose — the extra
  quarter-term is comparable to C_gap itself at a narrow gap;
- `self_resonance_Hz` = 1/(2π√(L_p C_gap)) and
  `self_resonance_from_port_region_Hz` from the region-only value.

Both capacitances are lumped-element estimates: the conductor body is
electrically continuous around the port, so a capacitance "across the readout"
exists only in a lumped model. `self_resonance_Hz` is an **estimate of** the
first self-resonance, not a bound in either direction: C_gap is only the gap's
share of the structure's capacitance, which pushes the estimate up, while a
distributed line resonates at (π/2)/(2π√(L C_total)), which pushes the true pole
up. The estimate exceeds the first pole only while C_gap/C_total < (2/π)² =
0.405; for the shipped solenoid and core templates C_gap/C_total is 0.09 and
0.16, which is why it comes out 2.5–3.4× high there
(docs/validation/electrostatics.md). The reference for a distributed structure
is the full-wave readout impedance (`fullwave`): the first self-resonance is the
pole of L(ω), where L changes sign. When a `fullwave` sweep is present,
`pickup.port_capacitance.vs_fullwave` reports that pole, the ratio to the lumped
estimate and the capacitance C = 1/((2πf)²L_p) that reproduces it. A 3D
`port` pickup reports the same numbers, computed the same way (the terminal *patches* of the port region
replace the terminal faces) when the design has an `electrostatics` section; `vs_fullwave` is axisymmetric
only.

## Mechanics and thermal (`mechanics`, `thermal`, axisymmetric and 3D)

Two optional auxiliary analyses run on the same mesh as the magnetostatic solve, each only when its
section is present. They need material data, which lives in two optional material blocks:

```json
"materials": {
  "g10": {"kind": "linear",
          "mechanical": {"E_Pa": 28e9, "nu": 0.21, "alpha": 2.4e-3, "notes": "warp direction"},
          "thermal": {"k_W_mK": [[4, 0.07], [20, 0.18], [77, 0.4]], "notes": "NIST fit"}}
}
```

| field | meaning |
|---|---|
| `mechanical.E_Pa`, `.nu` | isotropic Young's modulus [Pa] and Poisson's ratio (−1 < ν < 0.5) |
| `mechanical.alpha` | **total** thermal contraction strain from room temperature to the operating temperature (dimensionless, positive when the material shrinks). The stress-free strain applied by `mechanics.cooldown` is −α, so a free body contracts: u = −α x |
| `thermal.k_W_mK` | conductivity [W/(m·K)]: a number, or a table `[[T, k], ...]` interpolated piecewise linearly in T and clamped outside its range (a warning flags clamping) |

The **structure** is every region whose material has a `mechanical` block; the **thermal domain** is
every region whose material has a `thermal` block. Vacuum, air, gaps and ports have neither and are
excluded; a region of matter without the data is skipped with a warning. Touching regions share mesh
nodes, so the assembly is **bonded**: there is no contact, sliding or separation.

Built-in materials carry documented defaults — `nb`/`superconductor` (niobium, E = 105 GPa, ν = 0.40,
α = 1.43e−3, k = 5 W/(m·K) near 4 K), `copper` (130 GPa, 0.34, 3.26e−3, 600 W/(m·K) for OFHC
RRR ≈ 100 near 4 K), `iron` (211 GPa, 0.29, 1.98e−3, 80 W/(m·K) at room temperature) and `winding`
(a **smeared isotropic** impregnated coil, ~50 vol% metal + 50 vol% epoxy: 80 GPa, 0.30, 3.0e−3,
1 W/(m·K) transverse). Real windings are strongly orthotropic and cryogenic data vary with purity and
heat treatment, so override these before quoting engineering numbers. `cooldown` skips materials
declared `warm` (they stay at room temperature), which includes the built-in `iron`.

```json
"mechanics": {"supports": [{"region": "feet", "fix": "all"}],
              "pressures": [{"on": "vessel", "from": "vacuum", "p_Pa": 1e5}],
              "cooldown": true, "notes": "..."}
```

Small-strain axisymmetric elasticity in (u_r, u_z): ε_rr = ∂_r u_r, ε_zz = ∂_z u_z, ε_θθ = u_r/r,
γ_rz = ∂_z u_r + ∂_r u_z, with u_r = 0 on the axis. Loads:

- **Lorentz body force** in windings, f = J×B with the total static field. Azimuthal windings:
  f = J_φ(B_z r̂ − B_r ẑ). Poloidal windings: f = −J_z B_φ r̂ + J_r B_φ ẑ.
- **Maxwell surface pressure** on superconductors, t = −(B²/2μ0)n̂ with B on the vacuum side (the
  same integral as `forces`). A zero-thickness superconducting sheet (`curves`) reports no stress of
  its own, but its surface traction (B_in² − B_out²)n̂/(2μ0) **is** applied to whichever region it
  lies on (`mechanics.magnetic_pressure_sheets` lists the sheets that load the structure; a sheet
  with no structural region on either side gets a warning saying the load goes nowhere). Only the
  pressure part of the Maxwell traction is applied: the full traction has an extra (B·n̂)B/μ0, which
  vanishes identically on an ideal superconductor, and
  `checks.maxwell_traction_Bn_over_B_max` reports how large |B·n̂|/|B_poloidal| actually is on those
  faces (4e-14 on the validation case).
- **`pressures`**: a uniform normal pressure (positive = compressive) on the faces region `on` shares
  with region `from`, or with the non-structural surroundings (`"from": "vacuum"`, the default).
- **`cooldown`**: the uniform stress-free strain −α of each material.

`supports` fix `"all"`, `"r"` or `"z"` on every node of the named regions. **One support per region:**
a second entry on the same region is an error (it would silently replace the first), so use
`"fix": "all"` to constrain both components. Each connected assembly with no axial support keeps a
rigid axial translation: the solver pins one node and reports the displacement relative to that
assembly's volume-weighted mean u_z (a warning fires if the assembly also carries a net axial load,
which makes it unbalanced; the test is |ΣF_z| against `checks.axial_load_scale_N` = ∫|f_z|, not
against the mixed radial+axial `load_scale_N`). A material with ν > 0.45 gets a warning: the
pure-displacement formulation is not locking-free, and on a fixed mesh the stresses come out
over-stiff by roughly 1/(1 − 2ν) (7 % at ν = 0.49, 70 % at ν = 0.499 on the Lamé benchmark). A
`cooldown` with a material that has no `alpha` also warns, since that material then contracts by
nothing while the parts around it do.

`results.mechanics` gives, per region, `u_max_m` and `von_mises_max_Pa` with their locations and the
min/max of `sigma_hoop_Pa`, `sigma_rr_Pa`, `sigma_zz_Pa`, `sigma_rz_Pa`; per support, the reaction
`reaction_Fz_N` (total) and `reaction_radial_N_per_rad`; and in `checks` the axial-equilibrium
closure (`equilibrium_Fz_round_off` is the number to read — the identity Σ applied F_z + Σ reactions
= 0 is exact, so the residual only measures assembly round-off).

```json
"thermal": {"fixed": [{"region": "sink", "T_K": 4.0}, {"curve": "strap", "T_K": 0.9}],
            "loads": [{"region": "coil", "W": 1e-3}, {"region": "wall", "W_m3": 20.0}],
            "notes": "..."}
```

Steady conduction ∇·(k∇T) + q = 0. `fixed` prescribes T on a region (all its nodes) or on a
zero-thickness curve; every other boundary of the thermal domain is adiabatic. Two `fixed` entries
that meet on a mesh node with different temperatures are an error, not resolved by the order of the entries. `loads` give either
the total power `W` in a region or the density `W_m3`. A tabulated k(T) is solved by a Picard
iteration on k(T_prev) (iteration count and final change are in `solver`). Each connected part of the
thermal domain must touch a prescribed temperature, otherwise its level is undetermined and the
solver says which regions are the problem.

`results.thermal` gives per-region `T_min_K`, `T_max_K` and the conductivity used, and in `checks`
the heat balance `generated_W` vs `outflow_W` (also an exact identity of the discrete system, so
`balance_relative` only measures assembly round-off). `outflow_W` is the **net** reaction over all
prescribed-temperature degrees of freedom, so it cancels for a part carried between a cold and a warm
sink; `heat_removed_W` (what the absorbing sinks take, i.e. the cold-stage heat load) and
`heat_added_W` (what the warm boundaries supply) are reported separately, and `balance_relative` is
scaled by the largest of the three.

`Results.sample_mech(r, z)` samples `u_r`, `u_z`, `sigma_rr`, `sigma_zz`, `sigma_hoop`, `sigma_rz`,
`von_mises` and `T_K`, with `in_structure` / `in_thermal_domain` flags. Outside the structure, outside
the thermal domain or outside the mesh the value is **NaN**, not 0 (0 is a legitimate displacement and
0 K a legitimate temperature). Validation:
[validation/mechanics_thermal.md](validation/mechanics_thermal.md); formulation and limitations:
[physics/mechanics_thermal.md](physics/mechanics_thermal.md).

### Torsional dynamics (axisymmetric, `Results.torsion`; beta)

Not a design key: a post-processing call on a solved axisymmetric model.
It solves the **m = 0 torsional** problem only, the azimuthal displacement u_φ(r, z) of an axisymmetric structure
(rotation about the axis, uniform in φ), which decouples from (u_r, u_z) for m = 0. It is not a general modal
analysis: no axisymmetric (u_r, u_z) dynamics (breathing, axial, bending in the meridional plane), no m ≥ 1 modes
(beam bending, ovalling), no 3D, no supports (all boundaries are traction-free; u_φ = 0 on the axis only), no
pre-stress, no thermal or magnetic loads.

```python
res = design.solve(workdir)
d = res.torsion(dofs=True)                 # dof_r, dof_z of the H1 space (to build a load)
m = res.torsion(nmodes=6, shapes=False)    # f_Hz, residual, modal_mass_kg (= 1), diagnostics
h = res.torsion(load=g, freqs_Hz=[...], eta=1e-3)   # harmonic: q, Y = -i omega q per frequency
r = res.torsion(nmodes=6, regions={"tube": {"mu_Pa": 8e10, "density": 7900}})   # overrides
```

- **Structure:** the regions whose material has a `mechanical` block and a positive `density`; shear modulus
  μ = E/(2(1 + ν)). Regions with a mechanical block but no density (or μ ≤ 0) are skipped with a warning naming them.
  `regions: {<name>: {mu_Pa | E_Pa (+ nu), density, exclude}}` overrides per region without editing the design;
  unknown region names or keys, `nu` without `E_Pa`, both `mu_Pa` and `E_Pa`, `eta` < 0, `nmodes` < 0 and
  `lanczos_extra` < 0 are errors. Excluded regions are
  listed in `excluded_regions`. Request keys other than `nmodes`, `shapes`, `lanczos_extra`, `load`, `loads`,
  `freqs_Hz`, `band`, `eta`, `regions`, `dofs`, `verbose`, `solver`, `mesh_check` are errors; `freqs_Hz` or `band`
  needs `load` or `loads`; a request that computes nothing says so
  in `note`. 3D results raise `NotImplementedError`.
- **`modes`:** the lowest `nmodes` eigenfrequencies of K x = ω² M x (shift-invert Lanczos, full
  reorthogonalisation; `lanczos_extra` extra steps, default 24). An unsupported body has the rigid rotation as its
  first mode at f ≈ 0. Each mode has its relative residual (meaningless for the rigid mode, 0/0) and unit modal
  mass (2π xᵀMx = 1). With `load`, `participation` c_α = 2π Fᵀx_α; with `shapes`, the mass-normalised u_φ per DOF.
  `modes.diagnostics`: `unconverged_modes` (relative residual above 1e-6, warned about), `rigid_modes`, and
  `inertia`, a Sylvester count of the eigenvalues below the highest returned one (`complete: false`, warned about,
  means a mode or a copy of a repeated one was missed: raise `nmodes` or `lanczos_extra`; `complete: null`, also
  warned about, means the check could not factorize). A mode is rigid when its Ritz value is at most 1e-6 σ (the
  Lanczos shift), and rigid modes are left out of `unconverged_modes`.
- **`load`:** one azimuthal body-force value g per DOF of the space (N/m³ per unit load scale). The FEM integrates
  its **P_p interpolant** over the structure's elements, so a step between two nodes is smeared over one element:
  a step load of width W_z built from an indicator applies W_z ± h_e/3 at P2, depending on whether the edge nodes
  pass the test by rounding (this was the "+15 %" of the first validation). The response reports what was applied:
  `load.force_N` = ∫g dV, `load.torque_N_m` = ∫g r dV, `load.support_z_m`. Put a load edge on a region boundary,
  or use a smooth load, when the width matters.
- **`harmonic`:** with `freqs_Hz` and the structural loss factor `eta` (default 1e-4), time dependence
  **e^{−iωt}** as the full-wave engine: it solves (K(1 − iη) − ω²M) u = F and returns q = ∫ g u_φ dV over the full
  ring and the generalised mobility Y = −iωq (Re Y > 0; a mass gives Im Y > 0). Fluctuation-dissipation:
  S_q = 4 k_B T Re[Y]/ω² (one-sided). One sparse LU per frequency, one symbolic analysis (`solver` below).
- **Long-tube asymptote, validity range:** for a load uniform through the wall and f < c_T/(2t) (the first
  thickness-shear cutoff), the band mean of Re Y over many modes tends to the damped infinite T(0,1) waveguide
  Re Y_d (`benchmarks/torsion.py`, `damped_waveguide_re_y`), and as η → 0 to (∫g dV)² κ(k)² Y_inf,
  Y_inf = 1/(2b√(Sρ_s)), κ the load's normalised axial transform. Re Y_d differs from the lossless value at first
  order in η, with a sign set by the shape of |κ(k)|² (−1.7 % for the Gaussian test load at η = 0.03, +0.4 % for a
  0.4 m ring at 4-8 kHz, −3η²/8 for a flat transform). Validated: ±2 % against the applied load's lossless value,
  and against Re Y_d for a Gaussian load to 0.06 % at a participating overlap of about 1, 0.8 % at 0.5 (validation,
  §4b). A **surface traction above the cutoff** also
  drives the T(0, n ≥ 2) branches, and Re Y exceeds the T(0,1) value by 4-17× (§5): do not use the 1D asymptote
  there; use the multi-branch waveguide or resolve the wall and solve the harmonic problem (below).
- Validation: [validation/torsion.md](validation/torsion.md); formulation, limits and open issues:
  [physics/mechanics_thermal.md](physics/mechanics_thermal.md#torsional-dynamics).
- **Named loads (`loads`):** a list, summed with `load`; each entry is one
  of `{"region": R, "g_N_m3": v}` (body force on region R), `{"surface": {"on": R, "from": S | "vacuum"},
  "traction_Pa": v}` (azimuthal traction, along +φ, on the faces R shares with S; `vacuum` = every element outside
  the structure and the domain boundary) and `{"curve": C, "traction_Pa": v}` (a zero-thickness curve, applied once
  through the structure side), with an optional `name`. v is a number, an expression in `r` and `z` [m] or
  one value per DOF. Values are evaluated at quadrature points, so edges on region boundaries are exact.
  `loads.entries[i]` and `loads.total` report
  `force_N` (∫ dV or dA), `torque_N_m` and `support_z_m` as applied. Unknown keys, a region outside the structure,
  `traction_Pa` on a region, `g_N_m3` on a surface, an unknown `from` region or curve, a face set that is empty and an
  expression that does not evaluate are errors that name the entry (`/torsion/loads/i`).
- **Band means (`band`):** `band: {"f_Hz": [f0, f1], "n": N, "length_m": L}` (N odd, ≥ 5; `length_m` optional;
  instead of `freqs_Hz`) returns `harmonic.band`: `mean_Re_Y` (trapezoid), `quadrature_uncertainty` (|full grid −
  every second point|, an upper estimate of the grid's error), `ripple_rms`, `ripple_sem` and `subband_means` (n ≥ 17:
  eight disjoint sub-bands of Re Y minus its quadratic trend; **a ripple diagnostic, not a bound or an estimate of
  the finite-length scatter**), `edge_estimate` (with `length_m`: rms ripple × √2 (c_T/L)/(πB), **a conservative scale**
  of the band mean's finite-length deviation, not an estimate: on uniform tubes 58× the deviation at the median and
  below it in 6 of 108 cases, by up to 17 %; assumes B ≫ c_T/L), `df_Hz`; a warning when Δf > η f0/2 (resonances undersampled). A failed factorization in the
  band is an error naming the frequencies. A `freqs_Hz` request carries `harmonic.note`: single frequencies are not
  certified by the mesh guard. `Results.torsion_band(f0, f1, loads=…, eta=…, n=…, length_m=…)` (default solver:
  serial STRUMPACK, or serial Eigen SparseLU without STRUMPACK).
- **Solver (`solver`):** `{"backend": "auto" | "eigen" | "strumpack", "parallel_frequencies": p | "auto"}`. The complex
  symmetric operator K(1 − iη) − ω²M is factorized per frequency (Eigen SparseLU, or STRUMPACK complex LU above
  `AXF_STRUMPACK_MIN_DOFS`), analysed once per solver; p frequencies run concurrently with threads/p OpenMP threads
  each. `"auto"` (opt-in) reads the available memory first (the cgroup limit over the cgroup's ancestors less its
  usage without file cache; else the batch scheduler's memory allocation from `SLURM_MEM_PER_NODE` or
  `SLURM_MEM_PER_CPU` × `SLURM_CPUS_ON_NODE`; bounded by MemAvailable), solves the first frequency with STRUMPACK (its solver kept), probes Eigen SparseLU only where it
  can pay off and fits (≥ 8 frequencies left, STRUMPACK fill ≤ 90 per DOF, one Eigen solve predicted to fit), and
  runs the rest as concurrent Eigen solves or serial STRUMPACK, whichever is predicted faster;
  `harmonic.timing.parallel_choice` reports the memory sources, the probes and the reasons. Concurrent Eigen wins on
  thin walls, serial STRUMPACK on compact cross-sections (validation/torsion_highfreq.md).
- **Mesh-resolution guard (`mesh_check`, default `"error"`):** `harmonic.resolution` gives, per structure region,
  the elements per shear wavelength c_T/(f_max h) (h = longest element edge) at the highest frequency. When the 1st
  percentile of the elements of any loaded region, or of the structure, is below the order's refuse threshold the
  request fails; when any region's minimum is below the warn threshold it warns: warn below 24/3.5/2 (band-mean
  error about 1 %), refuse below 7/1.5/1.25 (about 10 % for P1; the coarsest resolution studied for P2/P3) for
  P1/P2/P3, from the convergence study ([physics/torsion_highfreq.md](physics/torsion_highfreq.md)); `"warn"` never fails, `"off"` skips. Only band
  means are certified: single-frequency values of a long structure converge much more slowly (resonance shifts).
- **Multi-branch waveguide (`axionfem.torsion_waveguide`):** the analytic mobility of an infinite uniform tube over
  all torsional branches T(0, n), damped, for volume and surface loads with Gaussian, top-hat, ring or sampled axial
  profiles (`Tube(a, t, rho, E=…, nu=…).mobility(f, TubeLoad(Gaussian(s), inner=1.0), eta)`), and the exact finite
  free-free tube (`finite_tube_mobility`). It replaces the T(0,1) asymptote above the thickness-shear cutoff.
  Derivation and validity: [physics/torsion_highfreq.md](physics/torsion_highfreq.md); validation:
  [validation/torsion_highfreq.md](validation/torsion_highfreq.md).

### Mechanics and thermal in 3D (`"symmetry": "3d"`)

The same keys, material blocks and result layout, solved with H1 elements of order `mesh.order` on the 3D mesh
(`cpp/src/model3d_mech.cpp`). Differences from the axisymmetric version:

- **Supports** fix `"all"`, `"x"`, `"y"` or `"z"` (Cartesian components) on every node of a region; one support
  per region, as in 2D.
- **`mechanics.symmetry_planes`**: `[{"name": ..., "point": [x, y, z], "normal": [nx, ny, nz]}]`. On every
  boundary face of the structure that lies in the plane, u·n = 0 (frictionless). Use it for plane-strain ends,
  and for the cut faces of a sector model (the wedge faces of one coil of a toroid). Any orientation: nodes
  whose constraint normals are not Cartesian are solved in a rotated basis (exact elimination, no penalty).
  A plane that meets no boundary face of the structure is reported with a warning. (`mesh.symmetry` cuts
  require `fullwave.rf_only`, which has no static solve, so they never carry mechanics.)
- **Rigid motions.** Every rigid motion of a connected assembly that the supports and planes leave free (up to
  six) is pinned in unconstrained directions and projected out afterwards (volume-weighted). The net load along
  each free motion is removed before the solve (balanced by nodal loads along that motion; a nodal balancing,
  not mass-weighted inertia relief),
  so the pins carry nothing and the result does not depend on where they are; `mechanics.rigid_modes` lists
  the assemblies, the free modes and the removed net load, and a net load above 1e-3 of the assembly's own
  loads warns (the assembly cannot be in equilibrium). Parts that touch only along an edge or at a point are a
  mechanism and are rejected.
- **Loads**: Lorentz f = J×B in windings with the static source current density J of the 3D solve (azimuthal,
  poloidal and `loop` windings, circuits); Maxwell pressure on superconductor faces (B from the field side);
  `pressures`; `cooldown`.
- **`mechanics.external_field`**: a prescribed static field in the format of `fullwave.source`
  (`{"file": ...}` or an inline `{"kind": "segments" | "uniform" | ...}`) added to B in the **Lorentz load of the
  windings only**: the field of sources that are not in the model (e.g. the other coils of a toroid when one
  coil is meshed). It is not part of the static solve, so superconductors do not screen it (warning if any
  are present).
- **`mechanics.probes`**, **`thermal.probes`**: `[[x, y, z], ...]`; results give per point `u_m`,
  `sigma_Pa` (xx, yy, zz, yz, xz, xy), `von_mises_Pa`, `B_T` (static field) and `B_external_T`, or `T_K`, NaN
  outside the structure / thermal domain. They work in MPI runs (the 3D results have no `sample_mech`).
- `thermal.fixed` takes regions only (3D designs have no curves), one entry per region. Entries that touch
  (they must then have the same temperature) form one sink group (`sink_group`); each entry reports its group's
  `heat_absorbed_W`, and `heat_removed_W` / `heat_added_W` sum the groups by sign (per group, not per node:
  nodal reactions of quadratic elements are not sign definite).
- `mesh.geometry_order` (3D, 1 or 2, at most `mesh.order`): straight-sided elements for polyhedral geometry,
  which skips the mesher's curved-element untangling.
- `mesh.extend_from_boundary` (3D, 0/1/2, default 0; gmsh `Mesh.MeshSizeExtendFromBoundary`) and
  `mesh.optimize_netgen` (3D, boolean, default false): grade the volume mesh from the boundary sizes and run
  gmsh's Netgen optimizer. Without grading, a curvature-refined curved wall (`mesh.curvature_points`) meets the coarse
  interior in one layer of flat elements, whose pointwise wall fields are unreliable; the SRF cell
  (`srf.elliptical_cell_design_3d`, smooth wall) sets both. The mesher reports element quality in
  `mesh.info.quality` (`gamma_min`, `gamma_below_0.1`, `minSJ_min`).
- `mechanics` and `thermal` are rejected with `fullwave.rf_only` (no static solve).

`results.mechanics` per region: `u_max_m`, `von_mises_max_Pa` with locations (`x_m`, `y_m`, `z_m`),
`sigma_hoop_Pa`, `sigma_rr_Pa` (cylindrical about the z axis), `sigma_zz_Pa`, `sigma_principal_Pa` (min of the
smallest, max of the largest principal stress). `supports[].reaction_F_N` and
`symmetry_planes[].reaction_F_N` are 3-vectors (at a node with several constraint sources the reaction is split
by the minimum-norm combination of all their normals); the planes also give the normal reaction split into
`reaction_normal_compressive_N` (pushes the structure inward) and `reaction_normal_tensile_N` (would have to
hold it: a joint that is not bonded would open). `lorentz_forces` gives F and torque per winding with the part
due to the external field. `checks` holds force and moment closure (`equilibrium_F_round_off`,
`equilibrium_M_residual_Nm`: identities of the discrete system), the integrated `applied_F_N` /
`applied_M_Nm`, and the pin reaction `rigid_pin_F_N` (zero when the free assemblies are balanced).
Validation: docs/validation/mech3d.md.

## Mesh size control

Element size is set by a few fields, from coarse to fine:

| field | effect |
|---|---|
| `mesh.size` | element size inside the device (default: device extent / 40) |
| `mesh.surface_size` | element size at and near the boundaries of superconductors, iron, normal conductors, windings and the readout port, and at every sheet (default: `mesh.size`/4) |
| `mesh.domain_radius` | radius of the outer semicircle (default 2.5 × device radius). The open boundary is exact, so this only costs elements |
| `mesh.domain_center_z` | z of the centre of that semicircle, default 0 |
| `<region>.mesh_size` | element size **inside** one region |
| `<region>.surface_size` | element size at and near **that region's own** boundary; overrides `mesh.surface_size` there (3D designs: passed to the mesher as the region's own `surface_size`, `cpp/src/model3d.cpp` `in["regions"]`) |
| `<curve>.mesh_size`, `<curve>.surface_size` | the same for one zero-thickness sheet |
| `mesh.refinements` | uniform refinements of the finished mesh (each one quadruples the element count) |

A region or curve that names its own `surface_size` is taken out of the group
that follows `mesh.surface_size`, so the two never fight; giving every refined
region the same `surface_size` reproduces the global setting exactly.

### Devices far from z = 0

`mesh.domain_center_z: z0` puts the outer semicircle (and therefore the exact
Dirichlet-to-Neumann boundary condition, whose multipole expansion is taken
about that point) at (r, z) = (0, z0). Without it a device centred at, say,
z = 10 m is enclosed by a sphere about the origin that must be at least 10 m in
radius, which wastes elements and spoils the DtN truncation. The automatic
domain radius, the "region outside the domain" check and the far-field size
grading are all measured from (0, z0). A device shifted in z together with
`domain_center_z` reproduces the centred result to mesh accuracy (validated in
`tests/test_mesh_options.py`).

## Distributed forces (`forces.distributions`)

For each body, the load that the static field applies to it, sampled along the
body. Summing the samples reproduces the body's net force exactly.

A **superconductor**, **iron** or **normal-conductor** body gets a *surface*
distribution (`"geometry": "surface"`): the Maxwell traction of the field just
outside,

    t = (1/mu0) [ (B.n) B - |B|^2 n / 2 ],   n = outward unit normal,

as `traction_r_Pa` / `traction_z_Pa`, together with the force per unit
meridional length with the azimuth integrated out,
`f_r_N_per_m` = 2 pi r t_r and `f_z_N_per_m` = 2 pi r t_z, so that
sum(`f_z_N_per_m` * `dl_m`) = `Fz_N`. Samples are ordered along the surface and
carry `r_m`, `z_m`, `arclength_m`, `dl_m`, `n_r`, `n_z`. On a superconductor
B.n = 0 and the traction is the pure magnetic pressure −|B|²/(2 mu0) n; the
check "superconductor surfaces: normal field B.n = 0" reports how well that
holds. This is exact only where the neighbouring material is magnetically
vacuum (mu = mu0); `exterior_nonvacuum_fraction` says how much of the boundary
is not, and a warning is emitted when it matters.

A **winding** gets a *volume* distribution (`"geometry": "volume"`) under its
own name: the Lorentz force density J × B at the element centroids
(`force_density_r_N_per_m3`, `force_density_z_N_per_m3`) with the per-cell
azimuthally integrated force (`f_r_N`, `f_z_N`) and volume (`dV_m3`), so that
sum(`f_z_N`) = `Fz_N`. The surface traction on a winding boundary is also
reported, as `<name>:boundary`, but its net force is a small difference of two
large opposing tractions (the field jumps by mu0 J t across the conductor) and
is much less accurate than the volume integral — `Fz_lorentz_N` in that entry is
the volume value to compare against.

Plot either kind with
`axionfem.plots.force_distribution(results, bodies=[...], quantity="f_z")`.
The MCP server returns only the scalar summary of each distribution; the sample
arrays are in `results.json`.

`force_samples` (`Design.solve`, `solve_design`, `axf solve --force-samples`,
default 200) is the number of samples per body. Consecutive pieces are merged down
to that many and the merge preserves the integral, so the net forces do not depend
on it — it only sets how finely the load is resolved and how big `results.json`
gets, at about 160 bytes per sample per body. On CORE: 10 kB with
`force_samples=0` (net forces and the closure check only, no arrays), 51 kB at 50,
149 kB at 200 and 884 kB at 2000, with every net force identical to 3e-10.

### Winding self-force and Newton's third law

The net axial force a rigid current loop exerts on itself is exactly zero, so
whatever `int J x B` returns for it is discretization error. For every powered
azimuthal winding the solver therefore solves the same mesh once more with that
winding alone in vacuum (all of them share one factorization: the free-space
operator does not depend on the source) and reports

* `Fz_self_N` — that (spurious) self-force: the error bar on `Fz_N`;
* `Fz_excluding_self_N` = `Fz_N` − `Fz_self_N` — the force from every other
  source, including the superconducting screening currents the winding itself
  induces, which are physical. Because the field superposes, the subtraction is
  exact for a linear design; with nonlinear iron only the error bar is reported.

The radial (hoop) load is **not** corrected: a loop's self-repulsion is real.
`radial_force_per_radian_self_N` is the free-space part of it, for reference.

Set `self_force=False` in `Design.solve` / `solve_design`, or `--no-self-force`
on `axf solve`, to skip the extra solve (one per powered azimuthal winding; above
twelve of them it is skipped automatically and `forces.closure.caveats` says so).

A **poloidal** winding needs no such estimate. Its axial force is
`F_z = -pi int (mu/r) [G^2] dr` with the bracket taken from below the winding to
above it, and `G` vanishes at both ends, so `F_z` is identically zero at every
radius whatever the cross-section looks like, as long as mu is uniform along the
legs. Only iron inside the legs, or a second poloidal winding, can break that; the
closure caveats name those two cases. For the same reason `B_phi = mu G / r` does
not depend on z in a current-free region, so the axial force on anything sitting in
a toroid's bore is exactly zero.

`forces.closure` sums the axial force on every body — self-corrected windings,
superconductors and iron. The device is isolated and its field decays, so the
sum must vanish; `relative` compares it with the largest force in the device
(the hoop load included, so an up/down symmetric design is not judged against
its own round-off). The same numbers appear in `checks` as "Newton's third law:
total axial force on all bodies". `caveats` lists what invalidates it: an
external `background` field (its source is outside the model), nonlinear iron
(the iron force is the Maxwell stress just outside, which converges slowly at
corners) and poloidal windings (no self-force estimate). On CORE the residual
is dominated by the slowly converging Maxwell pressure on the cavity.

## HTS critical state (`hts`, axisymmetric)

Screening currents in HTS bulks or homogenized tape stacks.  The static and pickup solves keep the
ideal (Meissner) superconductor model, so this analysis is **opt-in and additive**: the regions listed
here are solved again in a time-dependent problem where they carry power-law currents
E = E0 (|J|/Jc(B, T))^n.
With a `background`, a `linear` region of μ_r ≠ 1 is refused: the critical-state source does not
include the background's drive of magnetic regions yet, which the static solve does.

```json
"hts": {
  "regions": [{"region": "shield", "conductor": "tape", "initial": "zero_field_cooled"}],
  "drive": {"t_ramp": 600, "t_hold": 200, "ramp_down": true, "cycles": 1, "scale": 1.0},
  "steps_per_segment": 20,
  "field_of_view": [[0, 0], [0.1, 0]],
  "newton_tol": 1e-4, "max_newton": 60, "e_floor": 1e-6, "profile_points": 41
}
```

| field | meaning |
|---|---|
| `regions` | list of region names, or `{region, conductor, initial}`. The region's material must be a superconductor. `conductor` names a `conductors` entry (default `rebco`); `initial` is `zero_field_cooled` (cooled in zero field, the waveform must start at drive scale 0) or `field_cooled` (cooled with the drive already on, no screening currents at t₀) |
| `drive` | the waveform is built from `t_ramp` [s] (0 → `scale`), `t_hold`, `ramp_down` (back to 0), `t_hold_end` and `cycles` |
| `waveform` | instead of `drive`: `times` [s] and `scales`, piecewise linear (negative scales are allowed, e.g. symmetric AC cycles) |
| `steps_per_segment` | backward-Euler steps per waveform segment |
| `field_of_view` | (r, z) points where the screening-current-induced field (SCIF) is reported |
| `newton_tol`, `max_newton` | Newton tolerance on the update and the iteration cap per step |
| `e_floor` | the E–J law is continued linearly below \|E\| = `e_floor`·E0 (a minimum resistivity); results must not depend on it (docs/validation/hts.md) |
| `profile_points` | points of the final J(r) profile along each region's mid-height (0 = none) |

Current densities in the `hts` outputs are **engineering** (homogenized) values: Jc_eng = `fill_factor` × Jc
of the superconductor, and J is the current density averaged over the stack, so their ratio is the
fraction of the critical current, as in the conductor margins.

Every winding current **and** the background field are multiplied by scale(t), so the drive is either a
winding-current waveform or a background-field ramp (or both).

The analysis uses the design's own currents (scale(t) multiplies them); it is not repeated at the
benchmark amplitude or at the cost-optimal size, so to study a scaled magnet, scale the design.

Modelling and limits: the screening currents are azimuthal, so the analysis covers poloidal static
fields (azimuthal windings, axial background fields); a toroidal static field (poloidal winding) is an
error.  Nonlinear iron is linearized at its initial permeability (warning), persistent circuits are an
error, zero-thickness superconducting sheets are an error, and other ideal superconductors keep their
static constraint at every time step (warning).  Ψ is recomputed with the critical-state field for a
port (poloidal) pickup, the only case in which a poloidal static field couples to the readout.

The two models of the same body differ in two ways, and both show up in `psi.relative_change`:

- **History.** The static solve uses the body's `superconductors` state (by default `net_current: 0`,
  i.e. field cooled with the magnet on); the critical-state run uses `initial`.  Compare like with like
  by using `initial: "field_cooled"` and a waveform that holds the drive.
- **The wall.** An ideal superconductor excludes the field from its own volume; a critical-state region
  is magnetically transparent and carries transport currents, which is the realistic picture for a
  type-II HTS. The difference is O(wall thickness / radius) — 3.8 % in Ψ for the 12 mm wall of a 1.2 m
  CORE cavity (docs/validation/hts.md).

Validation: docs/validation/hts.md.

## Benchmarks, scan, cost, mesh

| section | fields |
|---|---|
| `benchmarks` | names (`optimistic`, `conservative`, `conservative_solenoid`) or `{name, Q, T, eta_A}` |
| `scan` | (defaults: DFSZ over 120.9 kHz–1 MHz in 3 years, an illustrative target) `f_min`, `f_max` [Hz], `live_time` [yr], `model` (DFSZ/KSVZ/custom), `E_over_N`, `C_QCD`, `rho_DM` [GeV/cm³], `Q_axion`, `Theta95`, `ma_fa` [GeV²] |
| `cost` | (illustrative defaults a = 0.25, kappa = 1.2, p_energy = 0.7, p_area = 0.8, 5 CHF/kg: [cost metric](physics/index.md)) `a` [MCHF], `kappa`, `p_energy`, `p_area`, `iron_cost_per_kg` [MCHF/kg], `cold_envelope` (only `bounding_cylinder`: A_cold = 2πR(R+H) of the cold parts), `B_peak_cap` (winding peak), `sc_wall_cap` (superconductor wall, regular peak) [T], `load_line_cap` (largest allowed fraction of a conductor's Jc) |
| `mesh` | `order` (FE order, default 2), `size`, `surface_size`, `domain_radius` (default 2.5 × device radius; exact open boundary, so the value only affects cost), `domain_center_z`, `refinements`, `adapt` (below) |

Adaptive refinement, `"mesh": {"adapt": {"cycles": 4, "tol": 1e-3, "reduction": 0.5, "max_elements": 2e6, "watch": [...]}}`
(2D and 3D): after each solve, Zienkiewicz–Zhu indicators of the static and the pickup field (recovered
per region, so material interfaces keep their jumps) set a target size per element that equidistributes
an error `reduction` times the current estimate; the geometry is re-meshed with that size field (a gmsh
background view combined by minimum with the regular size fields, so meshes only get finer) and solved
again, until the watched quantities change by less than `tol` between cycles or `cycles` re-meshes were done.
`results.adapt.history` lists elements, unknowns, the three quantities and the estimated relative
error per cycle (the energy-norm estimate; with `"Gamma"` watched in 3D it includes the surface term below);
the last change is an error bar on the reported numbers.
- **`watch`:** the quantities of the stopping test, from `"Psi"`, `"L_p"`, `"W_mag"` (the
  default is all three) and `"Gamma"`.
  - **What `"Gamma"` compares.** It compares the pickup's losses (`pickup.losses`): the R_eff at the band bottom
    f_min of the surfaces with `rf` data, which use `Gamma_magnetoresistance` where it applies. Without rf data it compares every
    surface's Γ. Surfaces below 1 % of the total are skipped, so a negligible one cannot hold the loop.
  - **It also watches L_p,** since Q = 2πf L_p/R_eff; `results.adapt.watch` shows the effective list.
  - **Refused** without a pickup and a conductor or superconductor surface.
  - **Never converged:** a missing or non-finite watched value, or a surface that appears, gives an infinite
    change. A non-finite Ψ, L_p or W_mag now does the same; before, it was silently skipped.
  - **Surface indicator (3D).** With `"Gamma"` watched, the indicator also includes the pickup field's recovery
    error on the conductor faces of `pickup.losses`. It is weighted by 1/μ_r² of the field side, as Γ weighs
    |H|², and taken relative to the faces' norm. It refines where that error is large (lips, edges and port
    terminals, where they exist). On the solenoid it gave about 20 % lower Γ errors per element than the
    volume indicator alone (`tests/test_adapt.py`). It is not R_s-weighted. In 2D the watch affects only the
    stopping test.
  - **History:** the rows carry `Gamma` per surface, and `R_eff_ohm` (at f_min) for the surfaces with rf data,
    whenever the results have `pickup.losses`.
- **`refinements`:** `mesh.refinements` cannot be combined with `adapt`. Every cycle meshes anew and would
  be refined uniformly again.
- **Floor:** the adapted mesh's size floor starts from cycle 0's floor, per-region and curve sizes included
  (2D and 3D). Before, a region size could be clipped when `max_elements` scaled the targets up.
- **Extrapolation:** `axionfem.convergence.adapt_convergence(results)` extrapolates the three finest cycles.
  This is a heuristic: the meshes are not similar, and h = unknowns^(−1/d), with d = 3 for 3D designs and
  2 otherwise.
- **h-studies:** `axionfem.convergence.h_study(design, factors, workdir)` scales each mesh's device-region sizes by
  the factor: `mesh.size`, `surface_size`, `far_size` if given, every region's and curve's sizes, and the 3D
  `curvature_points`.
  - **Not scaled:** the automatic far-field size and the 2D curvature sizing. `dof_ratios` and `warnings`
    report when the actual refinement departs from the factors by more than 20 %. `mesh.file` and
    `mesh.adapt` are refused.
  - **Output:** the three-mesh Richardson extrapolation (`richardson`: observed order, extrapolated value, GCI,
    `extrapolation_error`) of L_p, Ψ, W_mag and each surface's Γ. For monotone data the order is the root of
    the exact three-mesh equation.
  - **Reading the GCI:** on pre-asymptotic meshes it is an estimate, not a bound. On the solenoid's order-2
    surfaces it came out below the actual finest-mesh error (`benchmarks/rf_losses_3d.py hstudy`).
  - **Conservative GCI:** `formal_order=` gives Roache's variant (order min(observed, formal), safety factor 3
    when they differ by more than 10 %), which brackets those errors.
    - **Per quantity:** it can be one number or a dict per quantity, e.g. `{"L_p_H": 4, "Gamma": 2}` for order-2
      elements; energies converge at twice the field order.
    - **Far above the formal order:** an observed order more than twice the formal one is flagged in `behaviour`,
      but still gets this GCI.
  - **No extrapolation** when there is no usable order (an order below 0.5, or a difference at the noise floor of
    1e3 machine epsilons), an unreliable one (above 8 without `formal_order`; `order` is then null), or an oscillation the
    iteration cannot resolve. `behaviour` says which.

## Results (`results.json`)

| key | contents |
|---|---|
| `coupling` | Ψ (SI and natural), L_p, 𝓕, W_mag, g, device size |
| `static` | static solves: energies, superconductor fluxes and net currents, circuits, solver diagnostics |
| `pickup` | pickup inductance (two independent estimates for azimuthal pickups); `losses`: the RF surface-loss factor Γ per conductor surface and, with `rf` data, R_eff, Q and T/Q (2D and 3D; see `pickup.losses` under Materials) |
| `fields` | peak \|B\| overall (with location) and per region; superconductor surfaces: peak, its location, `superconductor_surface_regular_T` (excluding 3 element sizes around sharp corners), `superconductor_sharp_corners` (convex corners with interior angle < 150°, sheet edges) |
| `forces` | per winding `Fz_N` and `radial_force_per_radian_N` (hoop tension), and for a powered azimuthal winding `Fz_self_N` (the numerical self-force; exactly zero in theory, so the error bar on `Fz_N`) and `Fz_excluding_self_N` (the force from every other source — the number to quote for a linear design). Maxwell-pressure `Fz_N` on each superconductor (`superconductor:<body>`), iron body (`iron:<region>`) and magnetizable linear body (`magnetic:<region>`, a `linear` material with mu_r != 1). `closure` is the Newton's-third-law sum over all bodies, which must vanish. `distributions` holds the distributed load of each body (see below) |
| `windings` | flux per turn and ampere-turns of each azimuthal winding (mutual inductances) |
| `benchmarks[]` | 𝓕_req, 𝓕/𝓕_req, field-amplitude scale needed, W, peaks and superconductor fields at the requirement, cost, cost-optimal size (with its scaled fields); `caps_at_req` and `cost_optimal_size.caps` compare with `B_peak_cap`/`sc_wall_cap`/`load_line_cap` (`violation` = max(value/cap − 1, 0)); `conductor_margins` repeats the conductor block at the required amplitude and at the cost-optimal size, where lengths scale by `length_scale` so the current density scales as (field scale)/(length scale). Amplitude scaling is exact only for linear designs without fixed fluxes, circuits or backgrounds (a `note` says so) |
| `geometry` | cold-envelope area, iron mass |
| `conductors` | per winding with a `conductor`: the operating point (`J_engineering_A_m2`, `J_superconductor_A_m2` = J_e/fill_factor, `B_peak_T` and its location, `worst_point` with B_⊥, B_∥ and the field angle to the tape), `Jc_A_m2` at the worst point of the winding, `load_line_fraction` = J_sc/Jc, `short_sample` (the scale at which the load line meets the critical surface, and the current density and field there), `current_sharing_T_K` and `temperature_margin_K`. A field above the critical surface gives `load_line_fraction` 1e30 and a `status` line |
| `fullwave` | full-wave sweep and eigenmodes (when `fullwave` is given; see above). The port matrix exports as S/Z/Y Touchstone v1 in exp(+jωt) with `Results.write_touchstone` or `axf export-touchstone` (docs/physics/touchstone.md) |
| `fullwave.cavity` | cavity analysis (when `cavity` is given): per-mode `form_factor` and `losses` (see "Cavity haloscopes") |
| `hts` | critical-state analysis (when `hts` is given): `regions`, `drive`, `steps[]` (screening current, moment, max \|J\|/Jc, SCIF at the field-of-view points, Ψ, loss, energy balance, Newton diagnostics), `final` (field quality, `trapped` after a ramp-down, `current_profiles`), `total_loss_J`, `loss_per_cycle_J` / `ac_loss_per_cycle_J` (emitted when the drive returns to its starting scale), `psi` (Ψ with the critical-state field versus the ideal model), `newton`, `energy_balance_*` |
| `electrostatics` | capacitance matrix, pairwise capacitances and checks (when `electrostatics` is given; see above); `pickup.port_capacitance` carries the readout-gap capacitance and the self-resonance estimate of a port pickup |
| `mechanics`, `thermal` | stress/displacement and temperature results (when `mechanics` / `thermal` are given; see above) |
| `mesh`, `timing`, `warnings`, `checks` | provenance and diagnostics (`mesh.info.worst_elements`: the mesh check of a failed solve, below); `checks` includes the static Newton status and every circuit's linkage residual (with `linkage_scale_Wb`, the sum of the absolute contributions to the linkage integral, and `relative`: a zero-flux circuit reaches its target by cancelling large contributions, so the residual is judged against that scale, not against the target). Read `warnings` first: non-convergence, singular corners and approximations are reported there |

### Mesh check on solver failures

The breakdown and non-convergence messages listed below (2D and 3D, MQS and
full wave; a warning, or an error's hint) end with
` Mesh check (run because this solve failed): ...`:

- the worst element: number (the serial mesh's in MPI runs), region,
  centroid, whether it is inverted or degenerate, curved, min det J / max
  |det J| and the Jacobian aspect ratio. Straight simplices are measured at
  their centre (constant Jacobian); curved and non-simplex elements at the
  centre, the reference vertices and edge midpoints and the order-6
  quadrature points ("at the sampled points": an inversion between them can
  be missed);
- the counts of inverted, distorted (det ratio ≤ 1e-3) and flat (aspect
  ratio > 1e3) elements;
- the remedy: `Fix in the design: ...` (for example `mesh.geometry_order` 1
  for a folded curved 3D element, `mesh.optimize_netgen` or the region's
  `surface_size` for a sliver, "none needed" for a clean mesh), or, for an
  imported mesh (`mesh.file`; every 2D imported-mesh full-wave run),
  `Fix: the mesh is imported ...`: repair or regenerate it in the tool that
  wrote it (for a flat or folded element, or a moderate worst aspect ratio).

The five worst are in `mesh.info.worst_elements` (the JSON key
`distorted_curved` counts the distorted elements, curved or not simplices) of
`results.json` (warnings), or, for a thrown 3D full-wave validation error, of
the rejected-frequency diagnostics: `fullwave_rejected.json`, or
`f_<i>_rejected.json` in `fullwave.field_export_directory` when it is set. In an
adaptive sweep a refused frequency is a warning, and its message points at the
results and at its renamed diagnostics, `fullwave.adaptive.refused[].diagnostics_file`
(`fullwave_adaptive_refused_f_<i>.json`). Other thrown errors write no results
and say nothing about them. The check is computed only on a failure path (in
3D once per mesh state, and in MPI runs reduced over the ranks).

Covered: 3D AMS-PCG (not converged, iteration cap), the port and winding
conduction and divergence-cleaning PCG caps, the port conduction solve's
true-residual error, nonlinear-iron Newton; 3D full-wave GMRES, the multiport
and direct-validation true-residual gates, the cavity wall-current solve; 2D
poloidal Newton, the toroidal iron secant iteration, factorization and solve
failures of the poloidal, full-wave and toroidal engines; 2D imported-mesh
full wave (residual warning, LU failure); the 3D full-wave direct solvers
(rank-0 and distributed STRUMPACK setup, factorization and solve, the
tree-cotree factorization and solve). Not covered: electrostatics,
mechanics, thermal and HTS solves; the 2D eigenmode and adjoint
factorizations and the DtN update; the 2D imported-mesh eigenmode
factorization; the cross-section eigenproblem; the morphing extension
(sensitivities); the tree-cotree setup (it may refuse a design); the error
indicator's gradient projection (withheld, not a failure).

`coupling.g_factor` = Ψ²/(L_p W R²) with R the outer radius of all regions
(sheaths and windings included). Published definitions of R (e.g. the field-box radius) can
differ by a few percent; compare g only between designs normalised the same way.

## 3D designs (`"symmetry": "3d"`; requires an MPI build)

Regions carry a `solid` instead of a `shape`: `box`, `cylinder`, `sphere`,
`torus`, `cone`, `revolve` (an (r, z) SHAPE over an angle range), `extrude`,
`step` (CAD import), `wedge_frame` and booleans, each with an optional rigid transform.
`wedge_frame` (`{"polygon": [[x, z], ...], "hole": [[x, z], ...], "tan_a": t, "c_pos": a, "c_neg": b}`; `hole`
is optional) is a boolean-free polyhedral frame. In local coordinates it is the set of points with (x, z) inside
`polygon` and outside `hole`, and -(t x - b) <= y <= t x - a. Its two side faces are planes, and the width must
be positive at every vertex. The mesher builds it directly from planar faces, which is much faster than an
extrusion clipped by booleans. Place it with `rotate`/`translate`. Its values may be numbers or expressions of
named parameters; they are evaluated before meshing, and a non-positive width is reported with the vertex path.
`axionfem.convert.to_3d(design, pickup="ideal" | "slit")` revolves an
axisymmetric design. With `pickup="ideal"` (default) a slit-sheath pickup stays
closed and becomes a 3D `azimuthal` pickup, the exact counterpart of the 2D
ideal slit; `pickup="slit"` cuts real slits (`slit_deg`) and feeds the readout
through a port region.

- Pickups:
  - `{"type": "azimuthal", "conductor": BODY}`: a closed conductor (a ring
    about the z axis) carries the readout current I around the axis; its other
    loops keep their RF fluxoid. `pickup.L_p_from_flux_H` is the second,
    independent estimate of L_p.
  - `{"type": "port", "port": REGION, "toward": [x, y, z]}`: a region bridging
    two faces of one conductor; the readout current flows across it with a
    uniform-conductivity distribution, which is a modelling choice (for the
    toroid sheath it changes Ψ by about 1 % relative to the ideal slit).
    Without `toward`, the current direction is chosen so that Ψ ≥ 0 (as in 2D).
    A port between two different conductors is an error (no closed path).
  - **Distributed port.** The port region may consist of several
    disjoint gap pieces, e.g. `{"union": [SOLID, SOLID, ...]}` of readout
    modules spread along a slit, which are then fed in parallel.
    - **Terminals:** each piece must touch conductors on exactly two face
      patches. `toward` is required and orients every piece.
    - **Current split:** the conduction solve divides the total current between
      the pieces by their conductance, w = G/ΣG, so the port measures
      Z = wᵀZ_p w (Z_p: the pieces as separate ports).
    - **When it is a bus:** that equals an ideal common bus 1/(1ᵀZ_p⁻¹1) only
      when a symmetry of the structure permutes the pieces (identical and
      symmetrically placed). This was checked against N separate
      `fullwave.ports`, to 2e-5 to 4e-4 in Z
      (`benchmarks/core_readout_ports.py` u2 against n2).
    - **Otherwise:** in MQS the split gives L ≥ L_bus. For the bus itself use
      separate `fullwave.ports` (full wave) and the impedance matrix.
    - **Orientation limits:** every piece must be crossed within 60° of
      `toward`, and pieces between two bodies must connect them the same way
      round (`docs/physics/multiport_fullwave.md`). The gap-capacitance
      analysis and the COMSOL export do not handle distributed ports yet.
    - **Cost:** one solve, against one per port for separate ports. The same
      holds for `fullwave.ports` regions.
    - **Diagnostics:** the port reports `port_pieces`, `terminal_patches` and,
      for several pieces, `pieces`: each piece's centroid, crossing, cos θ and
      current fraction. They appear in full-wave `ports` and, for MQS runs,
      under `pickup.port_feed`. A warning appears when the fractions differ by
      more than 1 % or a piece carries under 1 %.
- Superconducting bodies keep their fluxoids through cohomology generators.
  Explicit `azimuthal`/`poloidal` constraints and `azimuthal_slit` need exact
  geometric cuts, i.e. bodies of revolution about the z axis; other bodies only
  take the default state (an error says so).
- Windings: azimuthal windings must be closed rings about their axis (a
  non-solenoidal current is reported as a warning); poloidal windings need a
  hollow (ring-shaped) cross-section and carry a harmonic current. `"type":
  "loop"` (3D only) is any closed coil — racetrack, saddle, square frame — with
  exactly one loop: the current is the harmonic (resistive) distribution
  around that loop, right-handed about `axis.direction` (the magnetic moment
  points along it).
- Nonlinear iron uses Newton with the differential reluctivity tensor; the
  pickup mode uses the secant permeability.
- `pickup.losses` (RF surface losses of the pickup) works as in 2D, plus the
  3D-only `Gamma_by_field_region` (see `pickup.losses` under Materials).
- Open boundary: an "infinity shell" (the outer 20 % of the domain sphere is
  mapped to infinity), so the domain radius only affects cost (default 1.6 ×
  the distance to the bounding-box corner). `Results3D.sample()` maps points
  beyond the shell's inner radius to the exterior field.
- Persistent circuits (azimuthal windings, linear problems) and powered
  circuits work as in 2D. Background fields (`background`, uniform or an
  (r, z) grid) are axisymmetric about the z axis through the origin; the
  solver computes the reaction field (superconductors, circuits, linear
  magnetic materials; not with nonlinear iron) and reports total fields;
  W_mag and cost exclude the background magnet. With a background, Ψ is
  the reciprocal form ∫(A_lab + A_bg)·B_pick: the
  primary form ∫A_pick·B_lab cuts the background at the domain sphere and
  then depends on the gauge, i.e. the mesh (0.70-0.88 of the correct value
  for a uniform field); it is kept as `checks.Psi_primary_form`, with a
  warning when it differs by more than 1e-3. Full-wave sweeps couple the
  background ("3D full wave" above). Zero-thickness curves are not
  available in 3D.
- `forces`: F and torque about the origin on every winding (∫J×B),
  superconducting body (Maxwell pressure) and magnetic material
  (`magnetic_materials`: iron and linear, the volume ("eggshell") form of the
  Maxwell stress over the vacuum elements touching the body, which avoids the
  singular field at the body's edges; `F_surface_N` is the surface integral for
  comparison, and `method` says which one is reported — the surface form, a
  few-percent estimate, is used when the body touches a winding, a
  superconductor or another magnetic body), and their sum as a closure check.
- `checks` compares the two reciprocal forms of Ψ (∫A_I·B_lab and ∫A_lab·B_I); `Psi_reported_form` says which one
  `coupling` reports (the reciprocal one with a background, the primary one otherwise).
- Superconductor surface fields: `superconductor_surface_T` is the raw
  maximum; `superconductor_surface_regular_T` excludes points within three
  face sizes of sharp convex edges (interior dihedral angle < 150°, listed in
  `superconductor_sharp_edges`), where the field is singular, as the 2D engine
  does at corners. A body with no surface point clear of its edges (a coarse
  surface mesh) falls back to its raw peak, with a warning, so the cap checks
  never pass vacuously. Benchmarks report the cost-optimal size and the field caps
  (`cost.sc_wall_cap`, `cost.B_peak_cap`) as in 2D.
- Results report the solver status (`solver.max_relative_residual`) and warn on
  any linear solve that did not converge or reached its iteration cap, and on
  curved elements that had to be straightened.
- Solver-failure messages end with a mesh check: see "Mesh check on solver
  failures" under Results.
- Order 2 (ND2) is the validated default; order 3 can stall next to thin
  features (warning). Meshes are deterministic (single-threaded meshing).
- Curved (order-2) elements: axf-mesher raises the order with nodes on the
  CAD, then runs gmsh's elastic pass and, where it leaves inverted elements
  (det J ≤ 1e-3 max |det J|), gmsh's Jacobian-targeted HighOrder optimizer
  (environment `AXF_MESHER_HO=auto`, the default; `full` always runs it,
  `elastic` never, `none` neither). If inverted elements remain after
  gmsh's pass, up to two more passes run with 2× and 4× the patch layers
  (`auto` and `full` alike, so `full` can also take more than one pass), each kept only
  if it lowers the count; each costs up to about as much as the first. The mesh info lists them (`high_order_passes`: layers, inverted
  elements after, kept, seconds) with the counts (`high_order_counts`).
  Both gmsh steps order their work by memory address, so axf-mesher meshes
  in a forked child and runs them in a process that has not meshed
  (`high_order_isolation`), with address randomization off on Linux
  (`address_randomization`).
  - On Linux that process is started with fixed inputs
    (`high_order_canonical`):
    - a copy of the mesher in memory (`high_order_executable`), for every
      build, a wheel's too; if the copy cannot be made or executed, the
      process runs from the file and warns;
    - fixed file names;
    - a fixed environment;
    - stdin `/dev/null`, and pipes for stdout and stderr;
    - a fixed CPU count (glibc only; up to 1024 possible CPUs).
  - It runs in a private work directory `<output>.axf-mesher.XXXXXX`,
    which it removes again, also when stopped by SIGTERM, SIGINT or
    SIGHUP.
  - `<output>.axf-mesher.lock` stops a second mesher on the same output.
    It must be a regular file of the user, not a link.
  - The curved mesh then depends neither on the output directory nor on
    the caller's environment, stdin, stdout or stderr, and it was the same
    on different nodes running the same operating-system image. It does
    depend on the input (including `verbose`), the build of axf-mesher and
    gmsh, and `step_output`.
  - The surface dump (`AXF_MESHER_SURFACE_DUMP`) is written by a child
    process and does not change the mesh.
  - If the process cannot be started, the mesher stops with the reason.
    `AXF_MESHER_HO_ISOLATE=0` meshes in one process instead (not
    reproducible). `AXF_MESHER_ASLR=on` keeps address randomization.
  - A curved mesh made without any of these steps comes with a warning. It
    is printed on stderr (in `axf`'s `mesh3d_mesher.log`) and listed in the
    results' `warnings`. On macOS there is no such process, so every
    curved mesh warns.
- Parallel runs (MPI): `srun -n N axf solve design.json -o DIR --save`
  (or `mpirun -np N`), or `Design.solve(mpi_ranks=N)` from Python, which runs
  that command and loads the saved solution (one solve per workdir at a time:
  `fields/` and `results.json` are shared; rank 0's summary goes to a status
  file of each launch, kept as `axf_status.json`). Every rank reads the mesh; the
  elements are partitioned with METIS; the topology (bodies, cuts, ports) is
  computed on the whole mesh on every rank and projected, so results do not
  depend on N beyond the solver tolerance (benchmarks/mpi_check.py: samples
  agree to 4e-8 or better on 1, 4 and 8 ranks). Scenes, sampling and the
  superconductor surface fields use a copy of the fields gathered on the whole
  mesh (memory per rank grows with the mesh, not with N). hypre/MFEM are built
  without OpenMP (threading gave no speedup there); mesh generation is
  serial. ND2 solves of 1–3 M unknowns can take tens of minutes on one rank.
- Imported meshes (`fullwave.rf_only` designs): `mesh.file` (gmsh MSH 2.2, linear or second-order tetrahedra) and
  `mesh.info_file` (JSON with `"ok": true` and `boundary_groups: [{"attr", "name", "type": "pec"|"pmc"|"port"}]`)
  replace the mesher. Element attribute 2 + k belongs to region k (1 is air); a region whose elements come only
  from the file has the solid `{"imported": {...}}` (free-form provenance; no CAD, no transform). A boundary group
  of type `port` can be the readout: `"pickup": {"type": "port", "port_boundary": NAME, "toward": [x, y, z]}` drives
  a uniform surface current between the two edge chains where the port faces meet PEC (a COMSOL Uniform lumped
  port). `fullwave.source_regions: [REGION, ...]` sets the prescribed source B0 to zero outside those regions.
  `axf import-comsol` writes such designs from COMSOL `.mph` files ([comsol_import.md](comsol_import.md)).
  Boundary groups may also be Robin conditions (beta; [formulation](physics/fw2d.md)):
  `{"type": "impedance", "material": NAME}` (Leontovich wall, Z_s = sqrt(μ/ε_c) of a linear material, or
  `"Zs_ohm": [re, im]` for e^{−iωt}), `{"type": "sheet", "material": NAME, "thickness_m": d}` (thin layer on an
  interior boundary, E_t continuous), `{"type": "scattering", "wave": "plane"|"spherical"|"cylindrical",
  "center": [...], "axis": [...]}` (first-order absorbing). Impedance groups also serve as lumped-port terminals.
  `fullwave.pml: [{"regions": [...], "type": "cartesian"|"cylindrical"|"spherical", "center", "axis",
  "scaling_factor": 1, "curvature": 1, "wavelength_m": 0}]` (beta) turns regions into perfectly matched layers (stretch
  q~ = q0 + sign (d ξ + (1+i) F λ ξ^p), λ = c/f unless given; closed `boundary: "pec"` only; sweeps only; one
  entry may cover several sides, each point stretched only where it lies beyond the physical domain). Robin groups
  must be exterior (impedance, scattering) or interior (sheet) and not next to a PEC element.
- 2D designs on an imported mesh (beta; [formulation](physics/fw2d.md)): `"symmetry": "axisymmetric"` (coordinates (r, z)) with `mesh.file`,
  or `"symmetry": "planar"` ((x, y), `fullwave.thickness_m` D, default 1 m): second-order triangles, regions
  `{"shape": {"imported": {...}}}` (attribute 2 + k), boundary groups `pec`, `pmc`, `axis`, `port`, `impedance`,
  `sheet`, `scattering`; `fullwave.rf_only: true`, `boundary: "pec"`; a sweep needs
  `pickup: {"type": "port", "port_boundary": NAME}` (in-plane field; the current runs along the port's edge chain
  towards `toward`, whose x/y components are the in-plane coordinates — (r, z) in axisymmetry; without `toward`
  along the chain's dominant coordinate, positive, not chosen for Ψ ≥ 0); `modes`/`families`/`target_f_Hz` for closed
  lossless models; `source` + `source_regions` as in 3D. Solved by `Fw2DModel` (`axionfem.fw2d.Fw2DResults`);
  families: axisymmetric TE = E_φ, TM = (E_r, E_z); planar TM = E_z, TE = (E_x, E_y).
- `axf mesh design.json -o DIR` meshes without solving; `axf export-comsol` hands a design, mesh or run
  directory to COMSOL (`.mphtxt` mesh, Java model or STEP), see [comsol_export.md](comsol_export.md).
- `--save` (CLI) or `Results3D.save()` writes the solution (MFEM mesh and grid
  functions plus `state.json`) to `DIR/fields`; `axionfem.load_solution(DIR/fields)`
  restores results, sampling and scenes in one process without solving again.
