# 2D full wave on imported meshes, Robin boundaries and PML

!!! warning "Beta"
    The 2D imported-mesh engine (`Fw2DModel`, `axionfem.fw2d`) and the Robin boundary groups and PML layers (2D and
    3D) are beta.

Code: `cpp/src/fw2d.cpp` (`cpp/include/axionfem/fw2d.hpp`), `cpp/src/pml.cpp` (`pml.hpp`), the Robin/PML block of
`cpp/src/model3d_fullwave.cpp`, `python/axionfem/fw2d.py`, `python/axionfem/comsol_import2d.py`. User docs:
[comsol_import.md](../comsol_import.md), [design_format.md](../design_format.md). The axisymmetric receiver-coupling
module ([cavity/coupling.md](../cavity/coupling.md)) solves its probe model with this engine.

## Why a separate 2D engine

The axisymmetric `AxiModel` full wave (`axi_fullwave.cpp`) is built around the axion readout: shapes meshed by
axionFEM, the MQS solve first, pickups as conductor "masters", an exact DtN sphere. A COMSOL 2D model is a closed
meshed domain with boundary conditions on boundary edges, a lumped port on an edge and domain sources. `Fw2DModel`
takes exactly that: the imported mesh (gmsh MSH 2.2, second-order triangles), boundary groups in
`mesh.info_file`, regions `{"shape": {"imported": {...}}}` (element attribute 2 + k), and `fullwave` with
`rf_only: true`, `boundary: "pec"`. It uses MFEM for the element machinery (H1 and Nédélec spaces, curved
geometry) and assembles complex matrices directly (Eigen `SparseLU`): 2D problems are small.

## Formulation (time dependence e^{-iωt}, every quantity the 3D one)

Weight w = 2πr (axisymmetric, coordinates (r, z), components (r, z, φ)) or w = D (planar thickness, components
(x, y, z)). ν' = (μΛ)^{-1}, ε' = ε_c Λ with ε_c = ε0 ε_r (1 + i tan δ) + iσ/ω and Λ = 1 outside PML layers.

- **out-of-plane family** (axisymmetric "TE" E = E_φ; planar "TM" E = E_z), H1 unknown u:
  ∫ c(u)ᵀ ν'_in c(v) w dA − ω² ∫ ε'_nn u v w dA, with c(u) = (−∂_z u, ∂_r u + u/r) axisymmetric,
  (∂_y u, −∂_x u) planar (the in-plane components of curl(u e_n)).
- **in-plane family** (axisymmetric "TM" (E_r, E_z); planar "TE" (E_x, E_y)), Nédélec unknown E:
  ∫ ν'_nn (∂_1E_2 − ∂_2E_1)(…) w dA − ω² ∫ Eᵀ ε'_in v w dA.
- **Robin boundaries**: + β ∮ E_t·v_t w dl, the boundary term ∮(n × ν curl E)·v of the integration by parts with
  n × ν curl E = β E_t:
  - impedance (Leontovich): β = −iω/Z_s, Z_s = sqrt(μ_m/ε_c,m) (Re Z_s ≥ 0; ≈ (1 − i)/(σδ) for a good conductor),
    or a given `Zs_ohm` [re, im] in the e^{-iωt} convention. Absorbed power ½ Re(Z_s)|H_t|².
  - sheet (thin layer d, interior boundary, E_t continuous): β = −iω Y_s, Y_s = −(2i/η_m) tan(k_m d/2)
    (→ (σ − iωε) d for |k_m d| ≪ 1). The two-face layer relation lets E_t jump by ≈ −iωμ_m d H_t,av; the
    continuous-E sheet is accurate when that jump is small, which the importer enforces by |k_m d| ≤ 0.5.
  - scattering (first-order absorbing): β = −iω/η of the adjacent medium (exact for a normally outgoing plane wave).
    Curvature terms of the *E-field* formulation: spherical waves none (n × curl E = −(1/r)∂_r(rE_t) = −ikE_t
    exactly for e^{ikr}/r); cylindrical waves (e^{ikρ}/√ρ) (ν/2ρ)[(t·a)² − (t·φ̂)²] for E_t along the unit vector t
    (a: cylinder axis, φ̂ azimuthal), i.e. +ν/(2ρ) along the axis, −ν/(2ρ) azimuthal, 0 for E_t radial (an end cap
    of a cylindrical boundary); the 2D engine evaluates it from the edge tangent. (The scalar Sommerfeld forms ν(1/r − ik) apply to scalar fields, not to E_t.) In 3D the
    cylindrical correction is the tensor (ν/2ρ)(aaᵀ − φ̂φ̂ᵀ).
- **lumped port** (in-plane family): impressed surface current K = I t / w (axisymmetric: uniform current I around
  the circumference, K = I/(2πr) t; planar K = I/D t) along the port edge chain from one terminal to the other;
  right-hand side iωI ∫_Γ t·v dl (the weight cancels). V = −∫_Γ t·E dl, Z = V/I, so that ½VI* is the delivered
  complex power. Uniform per circle is what keeps the current conserved on a radial port edge.
- **source**: prescribed B0 (`fullwave.source`, evaluated at (r, 0, z) or (x, y, 0); axisymmetric: when B0 depends
  on φ (a sector, unaveraged segments; tested at 8 angles on the source elements) its azimuthal average over 64
  angles, the part that couples to m = 0 fields, with a warning), restricted to
  `source_regions`; Ψ = (1/(iωI)) ∫ E_I·B0 dV, V_oc = −iωΨ (reciprocity, as in 3D). Only the in-plane components
  couple to the port family; an out-of-plane B0 gives Ψ = 0 with a warning.
- **essential conditions**: PEC (E_t = 0) on PEC groups and PEC (conductor) elements; out-of-plane family also on
  the axis (E_φ = 0). The in-plane family needs nothing on the axis (E_r is normal there; the r weight).
- **power diagnostics**: per part P = −Im(u^H A_part u)/(2ω) (regions, PML, each Robin group); the parts add up to
  ½ Re Z |I|² exactly (`balance_rel`).
- **eigenmodes** (closed, lossless): real H1 problems with axionFEM's shift-invert Lanczos. Out-of-plane family in
  E (PEC, axis and impedance walls: u = 0); in-plane family through its dual H = H_n e_n (stiffness 1/ε', mass μ;
  PMC, axis and open exterior ports: H = 0; PEC natural). A field component with no Dirichlet DOF has a static
  mode (H_φ = G/r in a coaxial region, constant H_z or E_z in a closed planar region) that P_k elements represent
  only approximately (a spurious low frequency); the exact constraint ∫ β u dA = 0 (flux through the meridional
  cross-section, from Faraday/Ampère around its boundary) removes it, imposed by a Lagrange multiplier. Impedance walls are PEC in
  the eigenproblem and give the perturbative Q_wall = ωU/P, P = ½ ∮ Re Z_s |H_t|² w dl. Scattering, sheet and PML
  are rejected for eigenmodes (complex, frequency-dependent problems).
  - Interior PEC boundaries (zero-thickness walls with field on both sides): E family u = 0 there; in the dual H
    family H_t jumps by the wall current, so the H DOFs on the wall are split into one copy per fan of elements on
    each side (`sheet_split_dofs`; the copies join at a free wall end), and the static-mode search does not cross
    the wall.
  - Q_wall of the out-of-plane family: the wall current is recovered by consistent flux, g = M_b⁻¹ [(K − ω²M) u]
    restricted to the wall DOFs (|H_t| = |g|/ω, O(h^{2p})) instead of the one-sided curl (O(h^p)).
  - Every mode reports its true residual `residual_true` = |(K − ω²M)x| / |Kx| (constraint directions removed); a
    residual above 1e-6 or fewer converged modes than requested is a warning.
- **input checks**: every exterior edge needs a boundary element with a group (else it would silently be PMC);
  impedance, scattering and PMC groups must be exterior, sheets interior, no Robin group next to a conductor
  element; one port group (the pickup's; another would be an open circuit). Robin groups crossing a PML layer are
  not stretched there (warning). 3D applies the same placement checks on the serial mesh.

## PML

`axionfem/pml.hpp`: complex coordinate stretching per layer, q~ = q0 + sign (d ξ + (1 + i) F λ ξ^p),
ξ = sign (q − q0)/d ∈ [0, 1] (COMSOL's polynomial stretch (1 − j) F λ ξ^p for e^{+jωt}, plus the identity term
d ξ that keeps s ≥ 1 at the interface for any p). Equivalent medium ε' = ε Λ, μ' = μ Λ,
Λ = det J J^{-1} J^{-T} with J diagonal in the local frame: Cartesian (s_x, s_y, s_z); cylindrical
(s_ρ, ρ~/ρ, s_axial); spherical (s_r, r~/r, r~/r). Axisymmetric designs use the cylindrical form about the axis in
components (r, z, φ) (also for COMSOL's "Cartesian" PML there); a spherical layer in (r, z) is a full 2×2 block.
The layer geometry (start, thickness, direction) comes from the extents of the layer's elements beyond the
non-PML field domain, per stretched coordinate and per side: a point is stretched in coordinate i only where it
lies beyond the physical domain's face on that side, so one layer entry can be a frame, a box shell or
side + cap + corner domains (COMSOL's usual PML selection). λ defaults to c/f. The AMS preconditioner of the 3D solver sees
Σ|Λ_i| e_i e_iᵀ.

## Validation (closed forms; `benchmarks/fw2d_validation.py`, `benchmarks/fw3d_robin_validation.py`)

Records `benchmarks/records/fw2d_validation.json` and `benchmarks/records/fw3d_robin_validation.json`; the end-to-end
import tests are in `tests/test_comsol_import2d.py`.

| case | quantity | result |
|---|---|---|
| axisym coax, PEC short | Z = −iZ0 tan kl | 2.9e-6 |
| axisym coax, plane scattering BC | Z = Z0 | 9.3e-7 |
| axisym coax, cylindrical PML (20 cells, order 2 / 3) | Z = Z0 | 3.4e-5 / 6.9e-6 |
| axisym coax, copper impedance walls | α = R_s(1/a + 1/b)/(2η ln(b/a)) | 2.7e-4 (= O(δ/a) beyond the perturbative formula); power balance 1e-14 |
| axisym biconical line, spherical scattering BC | Z = (η/π) ln cot(θ0/2) | 6.0e-5 |
| axisym biconical line, spherical PML (order 2 / 3) | same | 5.6e-4 / 4e-6 |
| planar plates, scattering / PML | Z = η h/D | 1e-9 / 2.5e-5 |
| planar wedge, cylindrical scattering BC | 1D Hankel solution with the Robin condition | 1.0e-5 |
| planar wedge, cylindrical PML | outgoing Hankel wave | 1.4e-4 (6e-6 at order 3) |
| pillbox TM010 / TE011 | frequency | 1.6e-8 / 1.2e-6 |
| pillbox TM010, copper walls | perturbative Q_wall vs G/R_s, G = p01 η/(2(1 + a/d)) | 7e-8 |
| pillbox TM010, driven through a wall gap | series-resonance Q of Z(f) vs gap-corrected perturbative Q | 8e-6 |
| 3D parallel plates (imported tets, surface port): PEC short / plane SBC / Cartesian PML | Z | 1e-8 / 2e-8 / 3.9e-4 |
| 3D plates, copper impedance plates | α = R_s/(ηh) | 1.6e-4 (O(δ/h)); balance 2e-11 |
| pillbox TE011, copper walls | Q_wall vs Pozar's closed form (wall-current recovery) | 2.0e-5 (3e-4 on the coarser mesh) |
| planar plates, thin resistive sheet across the matched line | shunt Y_s line formula | 1.1e-9 |
| planar plates, uniform B0 source | closed-form Ψ (sign included) | 5e-10 |
| planar plates, one PML entry at both ends, port in the middle | Z = Z0/2 | 7.0e-6 |
| axisym radial line, cylindrical scattering BC (E_t axial) / radial cylindrical PML (order 3) | Hankel solution | 1.6e-6 / 6.3e-6 |
| planar cavity with an interior PEC septum, both families | half-cavity modes, each twice | 3.9e-5 (TE) / 3.6e-4 (TM) |
| 3D plates, thin resistive sheet on interior faces | shunt Y_s line formula | 1.3e-8; balance 3e-12 |
| 3D plates, copper impedance plates on 2 MPI ranks | vs 1 rank | 5e-11 |

The PML error is set by the resolution of the strongly stretched layer (|s| ≈ 1 + Fλ/d ~ 10 here) and falls
with cells across the layer and with order (rows `*_pml_convergence` of the record).

## Limits

- The COMSOL importer's XML layout for planar 2D components, PML parameters and the Robin features is partly assumed:
  one real 2D-axisymmetric file confirmed the axisymmetric detection, the impedance feature, the Coaxial port and the
  PML node ([comsol_import.md](../comsol_import.md), "What is validated against what").
- 2D ports drive only the in-plane family; out-of-plane lumped ports (planar E_z across the thickness) are not
  implemented. Azimuthal mode numbers m ≠ 0 and out-of-plane wave numbers k_z ≠ 0 are rejected.
- Exact transition boundaries (E_t jumping across a thick layer) would need split DOFs (2D) or a cracked mesh
  with paired faces (3D, parallel); only the thin-sheet regime is implemented.
- 3D eigenmodes treat impedance walls as PEC (also in the gradient projector) without evaluating their loss (the
  2D engine evaluates it).
- The 3D energy operators (stored energy, energy-derivative identity) are not exported with Robin groups or PML.
- Not validated: 3D cylindrical scattering condition (only its 2D counterparts), MPI > 2.
