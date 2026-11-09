# An imposed background in 3D full-wave sweeps

!!! warning "Beta"
    The imposed background in 3D full-wave sweeps is beta.

A design's `background` ([design_format.md](../design_format.md), "Background field") is a static field whose sources are not part
of the model: a magnet computed elsewhere, axisymmetric about the z axis, given as a uniform B_z or as an (r, z)
grid with its flux function ψ = r A_φ. The 3D MQS solve adds the reaction of superconductors, persistent circuits
and linear magnetic materials to it; A_lab holds that reaction field only. A 3D frequency sweep couples the
background as well. Code: `cpp/src/model3d_fullwave.cpp` (`BackgroundA`
next to the B_lab source) and, for the exterior term, `axi::BackgroundExterior` (`cpp/include/axionfem/fullwave.hpp`,
`cpp/src/axi_fullwave.cpp`; shared with the axisymmetric engine). Validation:
[fullwave3d_background.md](../validation/fullwave3d_background.md).

## The coupling

The axion source of a sweep is the total static field, B0 = B_bg + curl A_lab. Its reaction part keeps the existing
form, Ψ_lab(ω) = (1/(iωI)) ∫ E·curl A_lab dV over the domain sphere plus the multipole exterior term ∮ Φ_lab E_r dA
(`Psi_exterior`): the reaction field's sources lie inside R1, so it is a decaying potential field outside.

The background enters in the reciprocal form

    Ψ_bg(ω) = (1/(iωI)) ∫ curl E · A_bg dV,    A_bg = (ψ / r) φ̂,

over all space: the physical elements inside R1 from the finite-element field (perfect conductors, the shell and PML
layers excluded), and outside R1 from the exterior multipoles of the readout field (below). For B_bg = curl A_bg,

    ∫_V E·B_bg dV = ∫_V curl E·A_bg dV + ∮_∂V (A_bg × E)·n dA,

so over all space the two forms are the same number, and the right-hand side `Bt` of `source_check` is the same
functional on a closed domain (a perfectly conducting sphere at R1), where E_t = 0 on every boundary. With PML layers
the domain is open: the forms then differ by ∮(A_bg × E)·n on the layers' interface, and neither couples the
background in or beyond the layers, as for B_lab (`exterior_term_background` says so, with a warning).

The reciprocal form is used because it is blind to the gradient part of E (curl ∇φ = 0, also discretely). E·B_bg is
not: wherever B_bg is cut off with a normal component (at the edge of a grid that ends inside the field, or at R1 when
the exterior part is approximate), the electrostatic part of E multiplies the cut's magnetic charge. For an open
readout, whose current charges its ends (an electric dipole, a capacitive gap), that part is O(1/ω), and the cut adds
an O(1/ω²) error to Ψ. For a closed readout, such as the validation's shorted coax, it is O(ω) like the inductive part,
and the cancellation is a fixed fraction instead: the winding path's exterior identity carries 7.6–12 % of |Ψ| from
kR = 0.02 to 0.8 there, and its O(h) trace error shows at 1.3e-3. In the reciprocal form a cut-off leaves only the
field it removes, −iω ∫_cut B_I·A_bg dV. As ω → 0 it tends to (1/I) ∫ A_bg·B_pick dV, the background's part of the
reciprocal MQS form, which is the 3D MQS Ψ with a background (below).

The axisymmetric engine's TM sweep uses the same form inside its domain (∫ A_lab μ H′ dV, background included), and
the same exterior part for a grid background, from the same moments of the grid (next section,
"The axisymmetric engine"; validation [axi_background_exterior.md](../validation/axi_background_exterior.md)). For a
grid that part is finite, 4.3–39 % of Ψ at kR = 0.4 to 0.8 in the 3D validation, so above the MQS limit a grid
design's Ψ(ω) depends on it.

## The exterior term

Outside R1 the readout field is outgoing. A_bg is azimuthal and axisymmetric about the z axis, so with the domain
centre on that axis (required) only the TM (electric) multipoles of order m = 0 contribute. For the Debye potential
a h_n(kρ) Y_n0,

    E_t(R1) = −(a/R1) h_n(kR1) γ_n(kR1) ∇_Ω Y_n0,      curl E = −k² a h_n(kρ) ∂_θ Y_n0 φ̂,

with γ_n(x) = n − x h_{n−1}(x)/h_n(x) (the exterior map's coefficient). The projection of the trace on the map's
orthonormal TM harmonic W_n0 = ∇_Ω Y_n0/(√(n(n+1)) R1) is t_n = −a h_n(kR1) γ_n √(n(n+1)), so

    ∫_{ρ>R1} A_bg·curl E dV = Σ_n k² t_n G_n(k) / (γ_n(kR1) √(n(n+1))),
    G_n(k) = ∫_{ρ>R1} A_φ ∂_θ Y_n0 h_n(kρ)/h_n(kR1) dV,

for n up to the exterior map's order. t_n is read from the field with the map's projection matrix (no extra solve).
The Hankel ratio comes from the stable upward recurrence of h_n/h_{n−1} and tends to (R1/ρ)^{n+1} as k → 0.

- **Grid backgrounds** are zero outside their box [0, r_max] × [z_lo, z_hi], but the magnet they stand for is not, and
  the integral converges slowly: a magnet's A_φ falls off as 1/ρ² (its dipole), so G_1 cut at ρ_max misses a fraction
  of order R1/ρ_max, and at low frequency the imaginary part of G_n comes from ρ ~ 1/k, outside any practical grid.
  (Cutting at a 2.8 m box gave an exterior term 20 % low at kR = 0.02 in the validation.) So G_n is split at the
  sphere ρ_s inscribed in the box (about the domain centre):
  - R1 < ρ < ρ_s: the grid. On each sphere ρ, composite Gauss–Legendre in θ; in the radius, composite Gauss–Legendre
    graded towards R1 (128 panels of 8 points each way; `AXF_FW_BG_PANELS`);
  - ρ > ρ_s: the grid's exterior multipoles, A_φ = Σ α_n (ρ_s/ρ)^{n+1} P_n^1(cos θ), exact when the magnet's sources
    lie inside ρ_s. Order n's angular moment then scales as (ρ_s/ρ)^{n+1}, and its radial integral against the
    outgoing wave has the closed form the axisymmetric engine uses (`axi::exterior_radial_integrals`):
    G_n ⊃ M_n(ρ_s) h_n(kρ_s)/h_n(kR1) J_n(k, ρ_s).

  `continuation_mismatch` compares the continuation with the grid where the box reaches beyond ρ_s (relative rms of
  A_φ over a polar sample; warning above 1e-2: the sources may reach ρ_s, so extend the grid). A grid whose inscribed
  sphere lies inside R1 is used as given over the box minus the ball (at most two arcs per sphere), with a warning.
  The ω-independent moments are computed once; per frequency only the Hankel ratios change.
  `fullwave.background.exterior` also reports the change of the k → 0 moments with doubled panels (warning above
  1e-4) and the first-order correction for the slivers between the faceted R1 interface and the sphere, which neither
  the finite-element volume nor the quadrature covers.
- **Uniform backgrounds** fill all space (A_φ ∝ ρ); the exterior integral diverges, as in the axisymmetric engine. It is
  omitted with a warning (`exterior_term_background: "omitted (uniform background: ...)"`), and Ψ(ω) then depends on
  the domain radius above the MQS limit (the static lab field keeps its own term, `Psi_exterior`, in both engines). With `boundary: "pec"` the model is a cavity of radius R1 (E = 0 outside):
  nothing is omitted for that model, but it is not the open-space answer, and it too depends on R1 above the MQS limit.

A magnet longer than the domain (DMRadio-Core's 2.95 m solenoid around a 1.5 m pickup) crosses R1: its field is not a
decaying potential field outside, and only this exterior term makes Ψ(ω) independent of the domain radius.

**The axisymmetric engine** (`cpp/src/model.cpp`, TM sweep). Its readout unknown is H′ = H − H_MQS
(the MQS pickup field vanishes outside the device), and outside its domain sphere Rb (centred on the axis at
`mesh.domain_center_z`) H′ = Σ b_n h_n(kρ)/h_n(kRb) P_n^1(cos θ), with b_n the projections of H′ on the outer
semicircle that its winding term already uses. With ∂_θY_n0 = −c_n P_n^1, c_n = √((2n+1)/(4π)),

    (1/I) ∫_{ρ>Rb} A_bg μ0 H′ dV = −μ0 Σ_n b_n G_n(k) / c_n

with the same G_n as above, from the same code (`axi::BackgroundExterior`: the grid up to its inscribed sphere, its
exterior multipoles beyond, the k → 0 quadrature check, the continuation check), to the order of the exterior map
(`fullwave.dtn_modes`, default 80). The order-2 outer boundary lies off the sphere by its curvature error; the slivers
get the same first-order correction as the 3D facets (`sliver_correction_over_moments`). The static lab field's own
term (`Psi_exterior`: the windings and the reaction of superconductors) keeps its static-multipole form, now with
A_lab without the background, which has its own term. That is the field 3D uses for its `Psi_exterior` too, but not
the same integral: 2D splits Ψ in the reciprocal form, (1/I) ∫_{ρ>Rb} A_lab μ0 H′ dV outside, and 3D in the primary
form, (1/(iωI)) ∮ Φ_lab E_r dA, which does not vanish as ω → 0 (7.6–12 % of Ψ for the 3D validation's winding). Only Ψ
and `Psi_background` (the reciprocal form in both engines) compare across engines, not `Psi_exterior`.

For a uniform background both engines omit the background's own exterior term, which diverges, and keep the static
lab field's, which is finite: its sources (the design's windings, the reaction of superconductors and persistent
circuits) lie inside the domain sphere. It is 2.9 % and 9.7 % of Ψ at kR = 0.4 and 0.8
for the coax with the coil as its winding in 1 T, and 0.66 % and 1.8 % for the reaction of a superconducting disk
(validation note, section 10). The uniform field's own term still makes Ψ(ω) depend on the domain radius above the MQS
limit: between R1 and 1.5 R1 it moved by 0.7–44 % at kR = 0.4 and by up to 159 % at kR = 0.8 in the validation's
designs. The warning says so, and to vary the radius or give the field as a grid. For grids the two engines agree to
the mesh (validation note).

**A grid that ends inside the domain sphere** (its box inside the sphere) has no exterior moments: the magnet's field
beyond the grid is coupled neither inside the domain (where the grid's field is taken as zero) nor outside. Both
engines label it "none (the grid ends inside the domain sphere, where its field is cut off: grid_edge_A_over_max)" and
warn when A_bg on the grid's edge is still above 1e-3 of its maximum; the coverage diagnostics below warn too. Such a
grid can be complete for the MQS outputs (the TM readout's MQS field lives inside the device) and still leave Ψ(ω) 16 %
off at kR = 0.8 (validation note, part C).

**Threads.** The moments are computed in an OpenMP region, the spheres in parallel, with the same result on any thread
count. In 3D every MPI rank computes them; set `OMP_NUM_THREADS` for the ranks (`scripts/env.sh` sets 1), or a rank
started without binding briefly uses every core.

## Outputs

Per sweep row (and per port), `Psi_background` = {`volume`, `exterior`, `exterior_last_order_fraction`}; both parts
are included in `Psi_*` and `Psi_vector`. `exterior` is null where no exterior term is computed (uniform, closed, PML,
no exterior modes, a grid inside the domain sphere), and
`exterior_last_order_fraction` is then absent. The fraction is |term_N| + |term_{N−1}| over |Σ_n term_n|: the last two
orders, because a magnet symmetric about the domain centre has no even orders, and the last one alone read ~1e-20
whether the sum had converged or not. It indicates the series' convergence; it does not bound the error
of truncating at N, because `dtn_modes` also sets the exterior map, which changes the kept orders (it read 5.4e-7 to
1.2e-6 where the term moved by 2.6e-6 to 3.6e-6): compare a run at a larger `dtn_modes`.
`fullwave.background` reports the form, for a grid
`physical_volume_outside_grid_fraction` and `grid_edge_A_over_max` (A_bg on the grid's edge over its maximum; with
part of the model outside the grid and an edge value above 1e-3, a warning), and the exterior diagnostics above:
`sliver_correction_over_moments`
and `quadrature_relative_change_doubled_panels`, the check with twice the panels to order min(N, 12) (the full order
cost four fifths of the moments at N = 80; for a grid continued beyond its inscribed sphere the largest moments are the
low orders', for a grid cut at its box the check covers orders ≤ 12 only), with at least panels/64 angular panels on
any arc and panels/32 on any radial piece, so that doubling refines short ones too (the default results are
unchanged). For a grid cut at its box with the domain's equator inside it the moments converge as h^1.5, and the
check reads about half their error (2e-9 to 9e-7 at the default; validation note, section 6).
`fullwave.exterior_term` keeps describing the static lab field's term; `exterior_term_background` the background's.
`Psi_over_MQS` divides by the 3D MQS Ψ, which with a background is the reciprocal form (next section).

The axisymmetric TM sweep reports the same row block, `Psi_background` = {`volume` (the background's part of the
domain integral, (1/I) ∫ A_bg μ H dV over the non-conducting elements), `exterior`, `exterior_last_order_fraction`},
and `fullwave.background` = {`type`, `form`, and for a grid `physical_volume_outside_grid_fraction` (by volume, over
the elements of the TM problem), `grid_edge_A_over_max` and `exterior`}, with the same warnings. The labels are the 3D
engine's: `exterior_term_background` is "included (grid, then its exterior multipoles)", "included (grid only: cut at
its box)", "none (the grid ends inside the domain sphere, where its field is cut off: grid_edge_A_over_max)",
"omitted (uniform background: the exterior integral diverges)" or "none (closed: a perfectly conducting sphere at R1,
outside which E = 0)". Only a sweep reports them: an eigenmode-only run has no exterior labels, no `background` block
and no exterior warnings. Designs without a background have none of these keys. With a uniform background and no
windings or reaction fields the exterior term is zero; with them it is exactly A_lab's exterior term (validation
note, sections 7 and 10).

## The 3D MQS Ψ with a background

The 3D MQS solve has two forms of Ψ: the primary (1/I) ∫ A_pick·B_lab dV and the reciprocal
(1/I) ∫ (A_lab + A_bg)·B_pick dV (`checks.Psi_A_pick_dot_B_lab`, `checks.Psi_A_lab_dot_B_pick`). Without a background
they agree to round-off. With one, the primary form is ill-defined: the background is cut at R1 (the shell is
excluded), which leaves a jump of its normal component there, so ∫_{ρ<R1} A_pick·B_bg changes by ∮_{R1} λ B_bg·n dA
under A_pick → A_pick + ∇λ. The solve fixes A_pick's gradient part with the per-element gauge mass κ ν_e/h_e², so the
mesh sets the value: for the validation's uniform field it was 0.70–0.88 of the reciprocal value over three meshes and
three domain radii, 0.91–0.98 for its long magnet and 0.97–0.99 for its coil. Even the Coulomb gauge gives 2/3 for a uniform field: outside a closed readout A_pick = ∇χ with a dipole
part, and the surface term −(1/I) ∮ A_φ A_pick,θ dA keeps −Ψ/3 of it at any R1.

The reciprocal form needs nothing outside R1 for a readout inside it: A_bg is azimuthal, so it sees only the azimuthal
average of B_pick,φ, which vanishes outside R1 (Ampère's law on the circle through a point there, spanned by the
spherical cap, which no readout current crosses). Its A_lab part is integrated over the mapped shell, where the pairing
of A and B is invariant under the map. It agrees with the axisymmetric engine, which uses this form, to 3e-6, is stable
to 5.6e-6 over meshes and radii, and is the low-frequency limit of Ψ(ω). So **with a background the 3D MQS Ψ is the
reciprocal form**: `coupling.Psi_SI_H_T_m2` and everything derived from it (F, g, the benchmarks,
`Results.Psi`/`F`, reach, `Psi_over_MQS`), with the orientation chosen by its sign; `checks.Psi_reported_form` names
the form, `checks.Psi_primary_form` keeps the primary value and its ratio as a diagnostic, and the solve warns when
they differ by more than 1e-3.

## Scope and limits

- The background's sources (windings, iron, cryostat) are not part of the RF model: they are transparent to the RF
  field unless meshed as regions.
- A background is refused together with `fullwave.source` (which replaces the static field altogether) and with
  `fullwave.rf_only` (no static solve, so no reaction field).
- Inside the domain, a grid cut off where A_bg is still large truncates the coupling there (the diagnostics above say
  how much A_bg is left at the edge); extend the grid over the domain sphere, and far enough that its inscribed
  sphere encloses the magnet.
- The domain centre must lie on the background's axis for the exterior term. Every background sweep meets this:
  generated meshes are centred at the origin, and imported meshes, `mesh.symmetry` and `mesh.domain_solid` need
  `rf_only`, which is refused with a background. For the same reason a background sweep always runs on a full sphere,
  with no symmetry images.
- A localized port is covered by the 360° comparisons: A_bg is axisymmetric, so it couples only to the m = 0 part of
  the readout field, and on an axisymmetric structure a port of current I drives that part exactly as the uniform feed
  of I does (Core: the 30° port and the 360° control differ in Ψ by 2.2e-4 at 70 MHz and 1.3e-3 at 140 MHz, the mesh).
  On a non-axisymmetric structure (DMRadio-Core's own pickup, with plates across the bore) this argument fails.
