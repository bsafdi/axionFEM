# Thin high-permeability layers in the axisymmetric magnetostatic engine

A mu-metal or soft-iron shield a few mm thick around a metre-scale detector is expensive to mesh through its
thickness. A zero-thickness curve (`curves`, [design_format.md](../design_format.md)) can instead be a
**thin magnetic layer**: a linear material with relative permeability μ_r > 1 and a thickness `thickness` = d. This is
the analogue of COMSOL's "Magnetic Shielding" boundary condition. The layer is a surface term in the energy of the
axisymmetric A_φ problem (`axi::PoloidalProblem`, cpp/src/axi_poloidal.cpp). It acts in the static poloidal problem
(azimuthal windings, persistent circuits, superconductor reactions, a background field) and in the azimuthal (MQS)
pickup problem. Validation: [validation/thin_magnetic_layer.md](../validation/thin_magnetic_layer.md).

**What it can and cannot compute.** Every design accepted with a layer has Ψ = 0. An azimuthal pickup couples
to the axion only through a toroidal static field, which needs the toroidal-field (G) problem; the G problem, port
(poloidal) pickups and the full-wave analysis all refuse layers. So the layer gives shielding factors, static fields
(poloidal), L_p, W_mag and their sensitivities. It does not give Ψ, 𝓕, the reach or the cost at the requirement of a
shielded toroidal detector. Those need layers in the G problem, which is not implemented.

## The engine's formulation

The unknown is A = A_φ(r, z), with B[A] = (−∂_z A, ∂_r A + A/r) and ψ = rA the flux function (flux per radian through
the circle at (r, z)). The solution minimises, per radian,

  W[A] = ∫ ( w(B[A]) − J A ) r dr dz,   w = ν|B|²/2 for linear media,

with H1 elements of order p. All energies are per radian below; the engine reports 2π times them.

## What a thin high-permeability layer does

Take a layer of thickness d and permeability μ = μ_r μ0 along a curve Γ with unit tangent t and unit normal n in the
meridional plane. Let ℓ be the shortest length over which the field varies along the layer. This includes the
distance to the layer's ends and corners, and to other objects. Let R_c be the layer's radius of curvature, in the
meridional plane and also the azimuthal radius r.

When d ≪ ℓ, the field inside the layer is nearly uniform across the thickness. The layer's interior potential varies
along the layer as e^{ikx} with k ~ 1/ℓ, so across the thickness it changes by a relative O((kd)²). The jump
conditions at the two faces (H_t and B_n continuous) then give, to leading order in d/ℓ and d/R_c:

- H_t is the same on both faces and inside the layer;
- inside the layer, B_t = μ H_t;
- the layer carries a tangential flux per unit circumference μ d H_t;
- B_n changes across the layer only as much as flux leaves or joins this tangential flux.

μ_r d/ℓ need not be small. A shield with shielding factor ≫ 1 has μ_r d ≫ R. That is the regime the condition is for.

In the flux function, the flux carried by the layer between its two faces is 2π (ψ₊ − ψ₋). Here ± label the two faces
and r is continuous across the layer. So, with [·] = (·)₊ − (·)₋:

  [ψ] = r μ d H_t,   i.e.   [A] = μ d H_t.

A thin high-μ layer therefore makes A **discontinuous** across Γ. The jump is proportional to the flux the layer
carries. A continuous-A surface term such as ∫ d (ν − ν0)|B|² cannot represent this: it is a first-order
perturbation, valid only while μ_r d ≪ ℓ, so it fails for shields.

## The model: an overlay on the unbroken mesh

The mesh does not remove the layer's volume. The curve Γ lies on the layer's **mid-surface**. The field of each
side fills the space up to Γ, so each side's medium (vacuum, or whatever the adjacent elements are) occupies a
half strip of width d/2 where the layer really is. The surface term must therefore be the **difference** between
the layer and these two half strips. Let μ_s,± and ν_s,± be the media on the two sides. The model is

  W_layer = ∫_Γ [ [A]² / (2 P)  +  Σ_± (d/4)(ν − ν_s,±) B_n±² ] r dl,

with

- P = (μ − μ̄_s) d, the excess tangential permeance per unit length, where μ̄_s = (μ_s,+ + μ_s,−)/2;
- B_n± = (1/r) ∂_l(r A±), the normal flux density on each side. It is the tangential derivative of ψ along Γ, so
  it needs only the trace of A on that side.

The first term is the energy of the excess tangential flux F = [A]. The layer carries (μ − μ̄_s) d H_t more
tangential flux than the two half strips. The half strips' own share, μ̄_s d H_t, is already in the continuous
field of the mesh.

The second term replaces the normal-flux energy of each half strip, ν_s,± (d/2) B_n±²/2, by the layer's
ν (d/2) B_n±²/2. For μ_r ≫ 1 this is essentially −(d/4) ν_s Σ B_n±²: it removes the field energy of the two half
strips, which in reality contain the layer and almost no energy.

This term is not small in the shielding regime. On the outside of a shield, B_n is of order B0, while the field
inside is B0/S. The half strip therefore matters at relative order (d/R)·S ~ μ_r (d/R)² compared with the
shielded field. Two cheaper choices were tried and rejected (sphere, closed forms below; μ_r = 1000, d/R = 0.01):

- leaving the term out: error +1.4 %;
- weighting it by the mean of B_n² across a linear profile, (B_n+² + B_n+ B_n− + B_n−²)/3, instead of half strip by
  half strip: error −2.8 %; at μ_r = 10⁵ the error is −77 %.

With the half strips, the error is −0.94 × 10⁻⁴. The term is negative, so it is guarded (see "Validity
conditions", item 4).

### The weak form

Each DOF on Γ gets one copy per side of the layer, and the elements on each side use their side's copy. At a free
end of the layer the two sides meet, so the DOF is not split there, and [A] = 0: the layer's flux vanishes at its
end. The end face (area d) is neglected. The discrete problem minimises W + W_layer over this enlarged space. Its
bilinear form is

```text
a(A, v) = Σ_K ∫_K ν B[A]·B[v] r dA
        + ∫_Γ { [A][v]/P + Σ_± (d/2)(ν − ν_s,±) B_n±[A] B_n±[v] } r dl.
```

Varying v± on Γ gives the thin-shell conditions. To leading order:

- H_t is continuous;
- the excess flux is [A] = P H_t, so the total tangential flux in the layer is μ̄_s d H_t + P H_t = μ d H_t;
- the half-strip term adds a jump of H_t of order d ∂_l B_n.

Varying v in the interior of each side gives the field equations on that side. The code assembles this in
`PoloidalProblem::assemble_at` (cpp/src/axi_poloidal.cpp). The DOF copies come from `set_thin_layers`, and the
quadrature data (N, G = (1/r) d(rN)/dl, P, d(ν − ν_s)/4) from `reduce`.

### The spherical shell in closed form

Take a spherical shell of mid-radius R in a uniform field B0 in vacuum, with a uniform interior field B_i and an
exterior dipole x = m/R³. With c = (d/4)(ν − ν0)/R, the model's conditions are the 2 × 2 linear system (at B0 = 1)

  −(1 − x)/μ0 + (R/P)(1/2 + x − B_i/2) + 2c(1 + 2x) = 0,
   B_i/μ0     − (R/P)(1/2 + x − B_i/2) + 2c B_i      = 0,

which gives the model's shielding factor S_model = 1/B_i (`shield_model` in benchmarks/thin_magnetic_layer.py).
The exact thick shell, with inner radius a and outer radius b (Jackson, Classical Electrodynamics, Section 5.12),
has

  S_exact = [ (2μ_r + 1)(μ_r + 2) − 2 (a/b)³ (μ_r − 1)² ] / (9 μ_r).

With δ = d/R, a = R − d/2 and b = R + d/2,

  S_model − 1 = (2/3)δ(μ_r − 1)²/μ_r − δ²(μ_r − 1)²/μ_r + δ³(μ_r − 1)³/(3μ_r²) + …,

and the exact shell has the same first two terms. So the model agrees with the exact shell **through second order**
in δ, and

  (S_model − 1)/(S_exact − 1) − 1 = −(13/12 + 1/(2μ_r)) δ² + O(δ³),

that is −1.083 δ² for μ_r ≫ 1, −1.088 δ² at μ_r = 100 and −1.133 δ² at μ_r = 10. A fit over δ from 10⁻³ to 0.1 gives
−1.084 to −1.106. The ratio S_model/S_exact − 1 is −0.94e-4 at μ_r = 1000, δ = 0.01
(docs/validation/thin_magnetic_layer.md, section 2). The plain first-order formula S = 1 + (2/3)δ(μ_r − 1)²/μ_r is
off by −(3/2)δ in S − 1. Since S − 1 = (2/9)((μ_r − 1)²/μ_r)(1 − a³/b³) exactly, S − 1 ≈ (2/3) μ_r d/R for μ_r ≫ 1
and d ≪ R (no condition on μ_r d): the coefficient is 2/3, not 2/9.

## Background fields (scattered-field form)

With a `background` the engine solves for the reaction field u. The total field is A = u + A_bg, where A_bg is a
vacuum solution with its sources outside the domain. The existing formulation drops the cross term ∫ ν0 B_u·B_bg. For
continuous u that term reduces to boundary integrals, which the exterior map and the superconductor ties account for.
Two cases need more:

- **Magnetic material regions** need the cross term ∫ (ν − ν0) B_bg·B[v] r dA. Without it a linear material in a
  background field does not respond at all.

  With the term, a solid μ_r = 10 sphere in a uniform 1 T background has B = 2.500 T inside, the exact value.
  Nonlinear iron with a background is refused.

- **Thin layers.** u jumps across Γ, so the vacuum cross term leaves a surface part, ∫_Γ H_t,bg [v] r dl, and the
  half-strip terms must use the total B_n,± + B_n,bg. Both are added as a constant right-hand side. The surface
  part is computed as Σ_K ∫_K ν0 B_bg·B[v] r dA (ν0, not the element's ν: the (ν_K − ν0) part is the volume term
  above), taken once with each element's own copies of the layer DOFs and once with the unsplit DOFs, and
  differenced. This is exact at the discrete level, and it is zero away from layers.
  **Every** element that holds a layer copy takes part, superconducting elements included (with no volume term).
  Dropping the unsplit sum is legitimate only if the unsplit numbering is tied inside the superconductor. Skipping
  the superconducting elements would make the answer depend on which side of the layer keeps the original DOF
  number, that is on the order of `regions` (a coated superconducting ball would get B − 1 with the wrong sign in
  one order).

## Interaction with windings, iron, superconductors and other layers

- **Windings:** a layer may run through or along a winding region. The winding is vacuum-like (μ_s = μ0) and its
  current density is unchanged. A circuit's linkage uses each element's own copies.
- **Linear magnetic regions:** a layer may run **along** them, for example a coating on a linear core, matched
  against the exact three-region sphere to 9e-5. Each half strip is counted in its own side's medium. The layer must
  be more permeable than the mean of the two sides, or the solve is refused.
  - **A layer may not end on a magnetic region that it meets only at its end** (linear with |μ_r − 1| > 1 %, or
    iron). The rule is per region: a region in the end's star is refused unless one of the end's layer faces runs
    along that same region. At an end the two sides join, so the layer's DOF is not split and [A] = 0.
    - Another part along the end (a stainless collar, say) does not exempt the region in front (exempting it gives
      +23.5 % with a μ_r = 1.05 collar on the hybrid can of the validation).
    - A skirt of the caps' material that overlaps the wall is accepted when it is drawn as part of the cap's
      region (a union), and refused when it is a separate region.
    - **A region in front of the end.** The model passes none of the layer's flux into the region through the
      contact, where in reality the contact carries it all. The error is global, not local: a can with a thin wall
      ending on meshed caps is off by 24 % at the centre. Such a design is refused with an
      error naming the curve, the region and the point.
    - **A layer that runs along or inside the region up to its end.** Examples are a coating on part of a linear
      core, or a strip embedded in one. There the flux leaves through the layer's faces, which the model resolves,
      and the end has only the usual O(d/ℓ) error. These designs are accepted. A hemispherical coating on a
      μ_c = 50 ball matches the meshed coating to about 1e-3 (validation section 6).
  - To model the junction, mesh it as a region (the wall's end and the part it meets), or make both parts thin
    layers. Curves that meet pass flux between them: each face couples its own sides, and the fans enforce flux
    conservation at the junction.
- **Nonlinear iron** (a B-H law) adjacent to a layer is refused. The overlay correction would need the iron's state.
- **Superconductors:** a layer may touch a superconducting region or lie on its surface. The layer DOFs used by the
  superconductor's elements are tied to its flux (rA = ψ_m); the copies on the field side stay free. The flux the
  layer carries at the contact goes into the gap between the layer and the superconductor, where B_n = 0 on the
  superconductor. The superconducting side's B_n is zero and its half strip carries no field, so it is counted with
  μ_s = 0 and no normal correction. A coated superconducting ball matches its exact solution to 2.6e-4 in B − 1, in
  either region order.
- **Superconducting sheets** (curves) meeting a layer: each sheet face ties the copies that the two elements next to
  it use, as a superconducting region ties its own elements' copies.
  - A sheet that **ends on** one face of a layer (a T) ties only the copy on its own side. The far side stays free,
    and the layer's flux passes the contact.
  - A sheet that **crosses** the layer has faces on both sides and ties both copies, so it cuts the layer's flux, as
    a superconductor through the shield does.

  Tying every copy would make a sheet ending on a can wall block the wall's flux: +98 % at the can's centre.
- **Superconducting bodies on the two sides of a layer are different bodies.** Body detection (`find_bodies`,
  cpp/src/model.cpp) does not join elements of *different* regions across a layer face. It joins sheets and
  regions at a shared vertex only when they lie in the same fan of the elements around that vertex, which are the
  fans the DOF copies are made from.
  - Two sheets meeting the wall from inside and outside at one height, or a flange on a coated ball, are two
    bodies, and the layer carries their flux difference.
  - When both sides of a layer face are superconducting (two bodies pressed onto the layer), the face keeps its
    tangential energy [A]²/(2P) with P = μd. With no displaced field medium, there is no half-strip term.
  - Merging such parts into one body would cut the layer's flux there: 1.266 T against 0.336 T meshed at the
    two-sheet can's centre.
  - Two bodies that meet exactly at an **end** of a layer still share the end's unsplit DOF. This covers regions,
    sheets, or one of each. No vertex unions are made at a layer's end, so they are not merged either. The case is
    refused, located at the layer's `/curves/<i>/path`, with the hint to extend the layer past the contact.
- **A superconducting region crossed by a layer stays one body and cuts the layer**, as a crossing sheet does.
  - Elements of the *same* region are joined across a layer face, so the region's copies on the two sides are tied
    together, and the layer's flux is cut there.
  - An nb plate region across a can wall gives 0.993 T at the centre, like the crossing sheet's 0.983 T.
  - Splitting it into two bodies would give 0.328 T, and a crossed pickup ring would carry the readout current
    twice.
- **A pickup conductor that forms several separate bodies is refused**, with or without layers. Each piece would
  carry the readout current, and L_p from the energy and from the flux would disagree.
- **A sheet drawn on a layer's own path** (sharing its mesh faces) is refused, naming both curves. It would tie both copies and short the layer, whichever face of the layer the conductor really sits on.
  Offset the sheet by about an element, or model the superconductor as a region with the layer on its surface. Two
  layers on the same path stack: their permeances add.
- **Other layers:** curves may meet or cross. Each face couples the copies of its own two sides, so T-junctions and
  corners split the DOF into as many copies as there are sectors.
- **The axis:** DOFs on r = 0 have A = 0 on every side. The flux a layer carries at the axis is zero (its
  circumference is).
- **The outer (DtN) boundary** must not touch a layer. The mesher keeps curves inside the domain. A layer face with
  only one adjacent element is an error.

## Validity conditions

1. d ≪ R_c, the radius of curvature of the layer, in the meridional plane and in r. Put Γ on the mid-surface. The
   sphere's S − 1 is then off by −1.09 (d/R)² relative.
2. d ≪ ℓ, the length of the layer and the scale on which the field varies along it. Near free ends, corners and
   junctions between layers the model is wrong within a few d. The integrated quantities (shielding factor, L_p)
   are affected at O(d/ℓ). This does not cover a layer ending on a meshed magnetic region, which is refused
   (above).
3. A linear, isotropic, non-conducting layer: μ_r is a constant greater than 1. A metallic shield at the readout
   frequency is a conductor. Mu-metal at 1 MHz has a skin depth of a few μm. The azimuthal pickup problem treats the
   layer magnetically, as it treats linear volume regions, so for a metal shield the pickup result describes a
   ferrite-like, non-conducting layer, and the results carry a warning saying so. Saturation is not modelled.
   - The engine reports the peak flux density in each layer, B_t = μ H_t, in `thin_layers[*].B_layer_max_T`.
   - It warns when that peak exceeds the material's `B_sat`. A `linear` material may give one; without it the
     threshold is 0.7 T, mu-metal's saturation.
   - The validation's tube carries 915 T in its linear model at 1 T, a reminder that the drive must be scaled to the
     real field before shielding numbers are trusted.
4. The mesh next to the layer must not be much thinner than the layer. The half-strip term is negative, and it can
   make the discrete energy indefinite when the adjacent elements are too thin. h_⊥ is the height of an adjacent
   field-side element over the layer face. The onset grows with the element order p. Last refused and first
   accepted h_⊥/d on a sphere:

   | p | last refused | first accepted |
   |---:|---:|---:|
   | 2 | 1.0 | 1.5 |
   | 3 | 1.5 | 2 |
   | 4 | 3.0 | 3.5 |
   | 5 | 4.0 | 5 |
   | 6 | 6.0 | — |
   | 7 | 8.0 | 10 |

   A can's corners at order 2 fail up to about 1.5. That is a regime where the layer is not thin compared with the
   resolved field anyway.
   - The engine reports min h_⊥/d per layer and warns below max(4, ⌈p²/4⌉): 4 up to p = 4, then 7, 9, 13 and 16.
   - With layers present the field system always uses the Eigen LDLT, because STRUMPACK does not expose its pivots.
     `AXF_STRUMPACK_MIN_DOFS` and `AXF_ORDERING=strumpack` are ignored (the latter with a note).
     `AXF_ORDERING=amd|metis` are honoured as without layers.
   - A build without METIS refuses more than 400 k unknowns unless `AXF_ORDERING=amd` asks for AMD, as without
     layers.
   - Forcing Eigen costs little: in one measurement Eigen+METIS was about 2× faster than STRUMPACK from 1 M to
     4.9 M unknowns.
   - A non-positive pivot stops the solve with an error located at the curve's `surface_size`. The check runs before
     the exterior map's low-rank update, so its message is the one that appears.
   - The LDLT of the symmetrically permuted matrix has the matrix's inertia, and the exterior map is positive
     semidefinite, so the check is exact for the energy without the exterior map.
   - Order 6 fails without layers too: the exterior map's capacitance matrix has rank 0 there. That failure is
     unrelated to layers.
5. `mesh.adapt` is refused with layers. Its error indicator recovers vertex values per region and would count the
   layer's physical jump in B_n as error, refining at the layer until the guard of item 4 refuses.

## Not modelled

- the toroidal-field (G = r H_φ) problems: a poloidal winding or a port pickup;
- the full-wave, cavity, critical-state (HTS), electrostatics and mechanics analyses;
- 3D designs, cross-section designs and planar designs;
- nonlinear (saturating) layers;
- a layer ending on a meshed magnetic region that it meets only at its end;
- a superconducting sheet on a layer's path, and two superconducting bodies (regions or sheets) meeting at a layer's
  end;
- a pickup conductor that forms several bodies;
- `mesh.adapt`.

All of these are refused with an error that names the curve. Not refused, but documented:
- **Forces on a layer are not computed.** The force closure check then reports `"valid": false`, with a caveat.
- **Thermal** analyses ignore layers, which carry neither conduction nor heat capacity; a warning says so.
- **The cold envelope** respects the layer material's `warm` flag.

## The 3D engine: what a thin layer would take (assessment, not implemented)

The 3D magnetostatic engine (`Model3D`, cpp/src/model3d.cpp) solves curl ν curl A = J with Nédélec (H(curl))
elements, hypre AMS, and cohomology elimination of superconductors. It has no zero-thickness curves or surfaces in
the magnetostatic problems: `curves` are refused. The only interior-surface machinery is in the full-wave solver.
There, imported-mesh boundary groups of type `sheet` (cpp/src/model3d_fullwave.cpp) add a Robin transition term
for a thin conducting sheet. The tangential E stays single-valued across the sheet, and only a surface admittance
is added. That is the electric analogue. It does not carry over, because a magnetic layer needs the tangential A
to **jump**.

The 3D version of the model above would need four pieces:

1. **A cracked H(curl) space.** The tangential trace n × A must be two-valued on Γ, since [n × A] is the tangential
   flux the layer carries, the 3D analogue of [A_φ]. The simplest route is to split the mesh along Γ: duplicate the
   vertices, edges and faces of Γ except on its boundary curve, which is where the layer ends. The gmsh "Crack"
   plugin does this in the mesher (axf-mesher's mesh3d), keeping the two sides as two boundary groups. MFEM then
   sees an ordinary mesh with an internal slit. A ParFiniteElementSpace on it works unchanged.
2. **The surface terms.** The analogue of the 2D integrand is
   ∫_Γ [ |[n × A]|²/P + Σ_± (d/4)(ν − ν_s,±)(n · curl A±)² ] dS.
   - n · curl A± is the surface curl of the tangential trace on each side, which the face Nédélec shape functions
     provide.
   - The coupling term needs the traces of both sides at the same points, with matched face orientations. That is
     an interior-face integrator over the pairs of slit faces, which MFEM does not have: a custom
     `BilinearFormIntegrator` on the face pairs, with the pairing taken from the crack.
3. **The solver.** AMS needs the discrete gradient and the vertex coordinates of the cracked mesh, which follow from
   the split mesh. The stiff spring 1/P (P = μ d, small) and the negative half-strip term both affect the
   conditioning. The AMS robustness study (benchmarks/ams_robustness.py) would need repeating. The guard against
   indefiniteness cannot read pivots in an iterative solve. It would need a local check, for example the smallest
   eigenvalue of each face pair's local matrix together with its elements.
4. **Superconductors, ports and the readout.** The cohomology basis of a superconducting body that touches Γ must use
   the copies of its own side. A layer through a port region or a pickup terminal must be refused.

An alternative in the style of COMSOL's mfnc interface is a reduced scalar potential H = H_s − ∇φ outside the
current regions. There the layer is a continuous φ with a surface Laplacian ∫_Γ d(μ − μ̄_s) ∇_t φ · ∇_t w, and no
crack is needed. It does not fit the engine, which is an A formulation with windings, persistent circuits and
cohomology. It would need a total/reduced potential split and cuts for multiply connected current-free regions.

Estimate: the cracked-mesh A formulation is the consistent extension. It needs mesher work, one face-pair
integrator, the cohomology changes, and a validation that reuses this note's sphere and can in 3D. The 2D engine's
results and closed forms serve as the reference. Roughly one to two weeks of focused work.
