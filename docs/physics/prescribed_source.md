# Prescribed static source in 3D full wave

!!! warning "Beta"
    The prescribed full-wave source (`fullwave.source`) is beta.

`fullwave.source` overrides only the real B0 used by the 3D full-wave source
overlap. It works with `fullwave.boundary: "pec"` (closed shield) and with `"radiation"` (unshielded detector; see *Radiating exterior*
below). Static winding,
energy, force and MQS coupling outputs still refer to the ordinary static
problem and must not be interpreted as results for this supplied field.

Examples:

```json
{"source": {"kind": "uniform", "B_T": [0, 1, 0]}, "source_check": true}
```

```json
{"source": {"kind": "segments", "segments_m_A": [[0, 0, -1, 0, 0, 1, 3]]}}
```

```json
{"source": {"file": "/path/to/source.json", "azimuthal_average_samples": 32}}
```

```json
{"source": {"kind": "toroidal_box", "B0_T": 10, "r_m": [0.525, 0.975], "z_m": [-0.225, 0.225], "profile": "uniform"}}
```

`toroidal_box` is the idealized toroidal field of arXiv:2211.00008: B = B0_T φ̂ (`profile: uniform`) or
B0_T r_m[0]/ρ φ̂ (`profile: inverse_r`, the field of ideal toroidal windings) for r_m[0] ≤ ρ ≤ r_m[1] and
z_m[0] ≤ z ≤ z_m[1] about the global z axis, and zero elsewhere. The field is discontinuous on the box surface.
Put that surface on mesh faces (e.g. the inner walls of a PEC sheath) so the source quadrature stays exact;
where it cuts elements, the source integral converges only as O(h).

These are entries within `fullwave`, with a frequency sweep and existing port
pickup. A segment row is `[ax,ay,az,bx,by,bz,current_A]`, endpoints in metres.
Positive current runs from a to b. The standalone segment example tests a
kernel; a physical magnet's currents must form a current-conserving geometry.
The file contains an inline source object, with no recursive `file` reference.
Only the averaging count may accompany a file reference. Paths are resolved
relative to the solver process, so absolute paths are recommended. Optional
`metadata` is copied to the result's source summary.

The kernel evaluates exact finite-wire Biot–Savart fields at actual FEM
quadrature points. It uses no smoothing or azimuthal averaging by default.
On-filament evaluations fail explicitly. PEC and mapped-shell elements are
excluded before evaluation. `azimuthal_average_samples` optionally averages
rotated vector fields about the global z axis, including the rotation back to
the original basis. Zero means unaveraged; an enabled average requires at
least two samples and angular convergence.

`fullwave.source_quadrature_order` selects an explicit tetrahedral integration
rule for assembling the source vector. Zero keeps the MFEM default; positive
integer orders through 30 are accepted. An explicit rule currently requires
tetrahedral elements. Refine this order independently of the RF polynomial
order and the segment/pack resolution. Agreement of direct and reciprocal
drives alone cannot validate this spatial integration.

`axionfem.prescribed_source.racetrack_source(detector_spec,
quadrature_order=3, arc_segments_per_quarter=12)` generates a finite rectangular
pack from an explicit coil list (`{"coils": {"list": [...]}}`, each coil with
its angle, leg radii, height, corner radius, pack dimensions and ampere-turns).
It uses the geometry/current as supplied. Gaussian transverse integration and polygonal arcs both need separate
convergence; the resulting field is not analytically exact for a curved pack.
The pack model does not resolve individual insulated turns or leads and does
not solve static screening. For large sources the direct evaluation cost is
O(number of segments × number of FEM source quadrature points).

For exp(-i omega t), the port field satisfies D E_j=i omega p_j I_j. With
b_i=integral N_i.B0 dV, `Psi_vector[j] = b^T E_j/(i omega I_j)` and the
open-circuit source voltage is `V_a/kappa = -i omega Psi_vector`. The source
vector is assembled once and reused for every frequency and port.
`source_check: true` additionally solves D E_a=i omega b and compares its
terminal voltages with that reciprocal prediction. The result includes both
complex vectors, a relative difference, and the independent solve's true
residual. This is an algebraic insertion/sign check; both routes share the
same b and therefore share any source quadrature error.

## Radiating exterior (`boundary: "radiation"`)

The reciprocity overlap is an integral over all space,
`Psi_j = (1/(i omega I_j)) int E_j . B0 dV`. With a closed PEC sphere E_j
vanishes outside it and the volume term is complete. With the radiating
exterior the FE domain ends at the physical sphere R1 (the mapped shell beyond it
carries only the exterior map), so the part outside R1 is added exactly as for
winding sources: outside R1 the prescribed field is curl-free and decaying,
`B0 = -grad Phi_0`, and `div E_j = 0` in the vacuum exterior, so

    int_{rho>R1} E_j . B0 dV = oint_{R1} Phi_0 E_j,r dA,
    Phi_0 = sum_{n>=1,m} R1 beta_nm/(n+1) Y_nm (R1/rho)^(n+1),
    beta_nm = (1/R1^2) oint_{R1} B0_r Y_nm dA.

`B0_r` is the Biot–Savart field evaluated at the R1 surface quadrature points
(not the static B_lab of the windings), and `E_j,r` the physical-side FE field.
The result row reports this part as `Psi_exterior`, already included in
`Psi_vector`. The identity needs every source current inside R1 and closed
current loops (an open filament's field is not curl-free outside it), so the solve
refuses a source whose largest current distance from the domain centre
(including rotated copies of an azimuthal average) is `>= R1`, or whose segment
endpoints (merged within `1e-9 R1`) have a net current above `1e-6` of the
largest segment current; `uniform` sources are rejected at parse time.
`fullwave.source.exterior` records the distance, R1 and imbalance; the solve also
requires the source inside `0.95 x` the innermost point of the faceted R1 interface
(`R1_faceted_min_radius_m`). The source series is truncated at
`n_max = clamp(ceil(log 1e-6 / log(r_source/R1)), N_DtN, 40)` (only scalar `Y_nm`
are needed, so this can exceed the DtN order);
`fullwave.exterior_source_multipoles.unrepresented_fraction` is the signed fraction of
`oint B0_r^2 dA` outside `n<=n_max` (truncation plus surface quadrature; negative
means quadrature over-capture), with a warning above `1e-3`. Keep R1 comfortably
larger than the source and check this number.

*Discrete R1 interface.* The interface between the physical domain
and the mapped shell is the meshed sphere, which deviates from R1 (by `~h^2/8R1`
for straight-sided facets; by ~6e-7 R1 on the curved order-2 validation meshes),
and the volume overlap stops there. The exterior identity is therefore evaluated
on that surface: `oint Phi_0 E_j.n dA` with the outward face normal `n` and
`Phi_0` at the actual `rho` (`(R1/rho)^(n+1)` per order). The sphere coefficients
`beta_nm` are obtained by radial projection (solid-angle weight `dA n.rhat/rho^2`,
the order-n part of `B_r` rescaled by `(rho/R1)^(n+2)`); this is exact order by
order, with cross-order leakage `O((n-n') (1 - rho/R1))`. The Poynting flux also
uses the face normal. Using `rhat` and `Phi_0(R1)` instead would be an
`O(1 - rho/R1)` inconsistency that matters only on straight-sided meshes when
`Psi_exterior` is a large part of a cancelling `Psi` (TM ports). On the curved
validation meshes the two differ by 4e-6 relative in Psi, so this surface
treatment is not independently validated (an analytic surface-identity test is
not available). This applies to winding sources (`B_lab`) too.

*Filaments in the source domain.* `B0 ~ 1/d` near a filament, so
Gaussian quadrature of `int N_i.B0 dV` does not converge in an element a
filament passes through: raising `source_quadrature_order` moves the result
without settling. When the overlap is a cancellation (a TM port, where the
electrostatic part of `int E.B0` vanishes identically) `Psi` then changes by
O(1): an electric-dipole port moved 100% between orders 4, 8 and 12 with a
source loop in vacuum. Run filaments inside PEC conductors, which is the
physical situation for a superconducting or copper winding (PEC elements are
excluded from the overlap); `fullwave.source.filament_proximity` counts the
non-PEC, non-shell elements a filament meets (exact segment/tetrahedron test
on the straight-sided elements, including rotated copies of an azimuthal
average) and the solve warns when the count is nonzero. Elements *next to* a
filament are as bad when the distance is much smaller than the element: the
same block reports `min_centroid_distance_over_element_size` and the count with
centroid within half an element size (no warning; check them for thin
conductors with a coarse surrounding mesh). A PEC sleeve added only to silence
the guard changes `E_j`, i.e. the device: model the real conductor. Finite packs in a
vacuum-like winding region therefore warn: demonstrate quadrature convergence
there before quoting a TM-like coupling.

The port impedance and radiated power do not depend on the source: Z comes
from the port solves with the exact radiating exterior, so radiation is in
`Re Z` (`R_rad_ohm = Re Z - R_bulk`, port power balance). Every radiating row
now also reports `P_poynting_R1_W` / `R_rad_poynting_ohm`, the flux of
`Re(E x H*)/2` through R1 integrated from the FE field itself, an independent
check of that balance (it converges more slowly than Z because it uses curl E
on the boundary). Energy/reactance-slope outputs remain PEC-only as before.
`source_check`, `source_controls` and the direct validation loads remain
PEC-only: with a radiating exterior the source outside R1 has no FE right-hand
side, and the controls would need their own exterior term.

Validation: `tests/test_fullwave3d_radiating_source.py` (AXF_SLOW) and
`docs/validation/fullwave_radiating_source.md`.

Tests cover analytic finite wire and circular-loop limits, independent
numerical integration of Biot–Savart, source singularity/input errors, finite
pack closure/current normalization/area, a small four-coil Ampere integral,
and a small 3D PEC multiport source-versus-reciprocity check. They do not qualify
a production magnet. Continuum self-field quadrature and source-assembly acceleration remain
separate work.
