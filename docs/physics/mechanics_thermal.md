# Mechanics and thermal: elasticity from Lorentz loads, steady conduction, torsional dynamics

Two auxiliary analyses run on the mesh of the magnetostatic solve when their design sections are present
(`mechanics`, `thermal`; [design_format.md](../design_format.md)): small-strain linear elasticity under the Lorentz
body force, the Maxwell pressure on superconductors, prescribed pressures and the cooldown contraction; and steady
heat conduction with k(T). A third, post-processing analysis (`Results.torsion`) computes the axisymmetric m = 0
torsional modes and the mobility to an azimuthal load. Code: `cpp/include/axionfem/mech.hpp`, `cpp/src/axi_mech.cpp`,
`cpp/src/model_mech.cpp` (axisymmetric), `cpp/src/model3d_mech.cpp` (3D), `cpp/include/axionfem/torsion.hpp`,
`cpp/src/axi_torsion.cpp`, `cpp/src/model_torsion.cpp` (torsion). Validation:
[mechanics_thermal.md](../validation/mechanics_thermal.md), [mech3d.md](../validation/mech3d.md),
[torsion.md](../validation/torsion.md), [torsion_highfreq.md](../validation/torsion_highfreq.md). Tests:
`tests/test_mechanics.py`, `tests/test_thermal.py`, `tests/test_torsion.py`, `tests/test_torsion_highfreq.py`.

## Axisymmetric elasticity

Small-strain, isotropic, torsion-free axisymmetric elasticity in u = (u_r, u_z) on the H1 space of the mesh order.
With ε = [ε_rr, ε_zz, ε_θθ, γ_rz] = [∂_r u_r, ∂_z u_z, u_r/r, ∂_z u_r + ∂_r u_z],

    σ = λ tr(ε) I + 2μ ε − (3λ + 2μ) ε_th I,    λ = Eν/((1+ν)(1−2ν)),  μ = E/(2(1+ν)),

and the weak form ∫ (B δu)ᵀ D (B u) r dr dz = ∫ f·δu r dr dz + ∮ t·δu r dl (per radian; a total force is 2π times
the per-radian integral).

1. **Displacement formulation**, no reduced integration, no stabilisation; quadrature of order 2p + 4. It is not
   locking-free: ν = 0.5 is rejected and ν > 0.45 warns, because the stresses come out over-stiff on a fixed mesh
   (on the Lamé benchmark at P2, h = 0.02, the σ_θθ error is 2.3e-3 at ν = 0.3, 7.2e-2 at ν = 0.49 and 0.72 at
   ν = 0.499, roughly 1/(1 − 2ν)). Refine or reduce ν.
2. **The axis.** u_r = 0 on every DOF at r = 0; in the hoop term u_r/r is replaced by its limit ∂_r u_r at quadrature
   points on the axis.
3. **The structure** is every region whose material has a `mechanical` block (so a `winding`, magnetically vacuum, is a
   solid, while ports, bores and gaps are not). A region of matter without the data is skipped with a warning.
4. **Bonded contact.** Touching regions share mesh nodes; no contact, friction, sliding or separation is modelled.
5. **Rigid axial modes.** An assembly with no axial support keeps one rigid translation. The solver pins one axial DOF
   (exact when the net axial load is zero, the condition for a solution to exist) and then shifts the assembly to
   zero volume-weighted mean u_z. A non-negligible net axial load (compared with ∫|f_z| dV, the axial load scale)
   warns that the axial displacement is not physical. There is no radial rigid mode (a uniform u_r strains the hoop).
6. **Loads.**
   - The Lorentz body force f = J × B at each quadrature point with the total static field (the local force density;
     azimuthal and poloidal windings, persistent circuits included). It reproduces the winding forces of
     `results["forces"]`.
   - The Maxwell pressure on superconductors, t = −(B²/2μ0) n̂, B on the field side. It is applied to the surface
     nodes, shared with whatever is bonded on the other side. Zero-thickness superconducting sheets (`curves`)
     transmit the two-sided pressure (B_other² − B_solid²)/(2μ0) to the solid they lie on, once; they report no stress
     of their own (`mechanics.magnetic_pressure_sheets` lists the ones that load something).
   - Only the pressure part of the Maxwell traction [(B·n̂)B − ½B² n̂]/μ0 is applied. That is exact on an ideal
     superconductor, where B·n̂ = 0; `checks.maxwell_traction_Bn_over_B_max` reports the largest |B·n̂|/|B_poloidal|
     the discrete field shows on those faces.
   - `mechanics.pressures`: a prescribed uniform normal pressure between two regions, or a region and the
     non-structural surroundings.
   - `mechanics.cooldown`: the stress-free strain ε_th = −α of each material, α the **integrated** contraction from
     room temperature to the operating temperature (α > 0 shrinks). `warm` materials are skipped; a material without
     `alpha` warns.
7. **Supports**: one entry per region (`"fix": "all" | "r" | "z"`); a duplicate is an error.
8. **Reactions and equilibrium.** Reactions are K u − F at the constrained DOFs, per support group. Axial equilibrium
   is an exact identity of the discrete system; `equilibrium_Fz_round_off` (the residual over the assembly scale) is
   the number to read.
9. **Solver.** Eigen `SimplicialLDLT` on the free-DOF system.

## Axisymmetric conduction

∇·(k∇T) + q = 0 on the thermal domain (every region whose material has a `thermal` block), weak form
∫ k ∇T·∇v r dr dz = ∫ q v r dr dz.

- **k(T)** by Picard iteration on k at the previous temperature at each quadrature point. The Kirchhoff transform is
  exact only when one k(T) law covers a whole connected part of the domain (T, not θ = ∫k dT, is continuous across a
  material interface), so Picard is the general method and the Kirchhoff solution an independent reference. k(T)
  tables are piecewise linear and clamped outside their range, with a warning.
- **Boundary conditions.** `thermal.fixed` prescribes T on a region or on a zero-thickness curve (a heat strap).
  Everything else is adiabatic; there is no convection or radiation boundary. Two prescribed temperatures that meet
  at a node with different values are an error. Every connected part of the thermal domain must touch a prescribed
  temperature.
- **Heat balance.** The reactions at the prescribed temperatures are split by sign into `heat_removed_W` (the sinks)
  and `heat_added_W` (the boundaries that supply heat); the net `outflow_W` equals the generated power exactly, and
  `balance_relative` divides the difference by max(|generated|, removed, added).

## Built-in material data

The built-in materials carry order-of-magnitude handbook values, each with a `notes` string. **They are documented
placeholders, not design data.**

| material | E [GPa] | ν | α (293 K → T_op) | assumption |
|---|---|---|---|---|
| `nb`, `superconductor` | 105 | 0.40 | 1.43e−3 | niobium near 4 K |
| `copper` | 130 | 0.34 | 3.26e−3 | OFHC near 4 K |
| `iron` | 211 | 0.29 | 1.98e−3 | room temperature (built-in iron is `warm`, so α is unused unless `warm: false`) |
| `winding` | 80 | 0.30 | 3.0e−3 | **smeared isotropic** epoxy-impregnated coil (~50 vol% metal) |

A real impregnated coil is strongly orthotropic, so the isotropic `winding` value is adequate for overall deflection
and the membrane hoop stress, not for conductor strain limits. The thermal conductivities are in the same `notes`.

`Results.sample_mech(r, z)` returns NaN (not 0) outside the structure, the thermal domain or the mesh.

## 3D

`cpp/src/model3d_mech.cpp` solves the same two problems on the MPI mesh of the 3D magnetostatic solve, with H1
elements of order `mesh.order` and hypre BoomerAMG-PCG (elasticity options with rigid-body modes, or systems options
when constraints are rotated), plus defect-correction passes so that the residual on the free rows meets 1e-12.
What differs from the axisymmetric engine:

- **Assembly on the whole mesh** with zero coefficients outside the structure or thermal domain, and elimination of
  every true DOF that no element of matter touches.
- **Loads assembled element by element** (Lorentz with the quadrature order of `results.forces`, so the two agree to
  round-off; pressures; thermal prestress), with their resultants (`applied_F_N`, `applied_M_Nm`,
  `lorentz_forces`) from the same pass. Surface loads use the serial mesh for adjacency and the gathered field.
- **Constraints by elimination in a rotated basis.** Supports and planes give each node constraint normals; an
  orthonormal basis whose first columns span them is eliminated exactly, for planes of any orientation (the wedge
  faces of a toroid sector, for example).
- **Rigid modes.** The motions the constraints leave free are pinned by pivoting over unconstrained nodes and
  removed after the solve by a volume-weighted projection. Before the solve the net load along each free mode is
  removed (a nodal balancing), so that a discretely unbalanced load does not end up on the pins; the removed load is
  reported and warns above 1e-3 of the assembly's load. Assemblies whose parts touch only along an edge or a point
  are rejected as mechanisms.
- **Heat per sink group**: the heat through each group of touching prescribed-temperature entries is the sum of its
  nodal reactions.
- **Probes** (`mechanics.probes`, `thermal.probes`) evaluate u, σ, von Mises, B and T at listed points during the
  solve, instead of `sample_mech`.
- **`mechanics.external_field`**: a prescribed field (`fullwave.source` format) added to B in the windings' Lorentz
  load only, for sources outside the model (the other coils of a toroid when one sector is meshed).
- `mesh.geometry_order: 1` gives straight-sided elements under FE order `mesh.order`.
- Extremes are lattice maxima on a 2p lattice per tetrahedron; `sigma_hoop` and `sigma_rr` are cylindrical
  components about the z axis.

3D limits: tetrahedra only; no zero-thickness curves; `mesh.symmetry` cuts need `fullwave.rf_only`, which has no
static solve and therefore no mechanics, so 3D models are full models; thin layers must be meshable at the
surrounding element size.

## Torsional dynamics

!!! warning "Beta"
    The torsional analysis (`Results.torsion`, `Results.torsion_band`) is beta.

For axisymmetric (m = 0) motion the azimuthal displacement u_φ(r, z) decouples from (u_r, u_z) and the problem is
scalar:

    ε_rφ = ½(∂_r u − u/r),   ε_zφ = ½ ∂_z u,
    K_ij = ∫ μ [(∂_r N_i − N_i/r)(∂_r N_j − N_j/r) + ∂_z N_i ∂_z N_j] r dr dz,   M_ij = ∫ ρ N_i N_j r dr dz

(per radian). u_φ = 0 on the axis, with the same treatment of u/r as the hoop term above. It matters for detectors
with a coaxial pickup: the flux coupling weight (n × B_p) × B_lab/(μ0 I_p) is purely azimuthal, so only u_φ couples
the static field into the pickup flux at first order.

- **`modes`**: M-orthogonal shift-invert Lanczos with full reorthogonalisation on (K + σM)⁻¹M (the shift keeps the
  matrix definite despite the rigid rotation u_φ = c r). Each mode reports its relative residual
  ‖Kx − λMx‖/(‖Kx‖ + |λ|‖Mx‖) and is normalised to unit modal mass; a Sylvester inertia count checks that no
  eigenvalue up to the highest returned one was missed (`inertia.complete`).
- **`harmonic`**: the complex response to an azimuthal load with time dependence e^{−iωt} and one structural loss
  factor η: (K(1 − iη) − ω²M) u = F. It returns the generalised coordinate q = ∫ g u_φ dV of the load and the
  mobility Y = −iωq, the quantity the fluctuation-dissipation theorem needs: S_q(ω) = 4 k_B T Re[Y]/ω² (one-sided).
- **Loads**: nodal values (integrated as their P_p interpolant), or named loads on regions, surfaces and curves,
  evaluated at quadrature points with exact edges ([torsion_highfreq.md](torsion_highfreq.md), part B).
- **The structure** is the set of elements with a `mechanical` block and a positive `density`; regions without a
  density are skipped with a warning (the built-in `winding` has none). Per-region overrides
  (`regions: {<name>: {mu_Pa | E_Pa (+nu), density, exclude}}`) let a study sweep materials without editing the
  design.
- **Long-tube asymptote.** For a thin tube and a load uniform through the wall, below the first thickness-shear cutoff
  f < c_T/(2t), the band mean of Re Y tends to the damped infinite T(0,1) waveguide value, and as η → 0 to
  (∫g dV)² κ(k)² Y_inf with Y_inf = 1/(2b√(Sρ_s)). Above the cutoff a surface traction also drives the T(0, n ≥ 2)
  branches and the T(0,1) value is several times too low; use the multi-branch waveguide or resolve the wall
  ([torsion_highfreq.md](torsion_highfreq.md)).

Limits: m = 0 torsion only (no (u_r, u_z) dynamics, no m ≥ 1 modes, no 3D); no supports other than u_φ = 0 on the
axis (every body is free-free); no pre-stress; one uniform η; serial Eigen solvers. A repeated eigenvalue is not
guaranteed to be found with its multiplicity by the single-start Lanczos; the inertia count and the residuals warn
(`inertia.complete: false`, `unconverged_modes`).

## Limitations

1. **Linear elasticity only**: small strain, no plasticity, no buckling (often the real limit of a thin shell under
   external pressure), no large rotations, no contact.
2. **Isotropic materials only**; the impregnated-winding default is a smeared composite.
3. **Bonded contact**: no shrink fits, slip planes, bolted joints or gaps (build them from separate parts and
   supports).
4. **Static loads only**: no quench transient, eddy-current or dB/dt loads, gravity or seismic loads.
5. **The magnetic load is not coupled back**: the fields are computed on the undeformed geometry.
6. **No forces on iron**: magnetized iron feels a Maxwell stress that is not computed, so designs with iron
   under-report their loads.
7. **Thermal**: steady state only; no convection, radiation or contact conductance; isotropic k; no coupling of the
   temperature field into the mechanics (`cooldown` is a uniform strain per material, not α(T(x))); heat loads are
   given by hand.
8. **The mesh is the magnetostatic mesh**: no independent mechanical refinement and no singularity treatment at
   re-entrant corners. Fillet the geometry for a meaningful peak stress. Reported extremes are lattice maxima.
9. Displacement and stress are not exported to scenes; use `Results.sample_mech` (axisymmetric) or probes (3D).
10. Every design needs a magnet (or an `electrostatics` section), so a purely thermal study carries a dummy winding.
