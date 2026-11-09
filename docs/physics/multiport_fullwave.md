# Independent 3D current ports and prescribed magnetic sources

!!! warning "Beta: prescribed source"
    The 3D full-wave current ports are validated ([fullwave3d.md](../validation/fullwave3d.md)); the prescribed source
    (`fullwave.source`) is beta.

The implementation is in `model3d.cpp`, `model3d_fullwave.cpp`,
`prescribed_source.cpp`, and `spec.cpp`.

The existing `pickup` remains port 0. Additional independently driven physical
gap volumes are listed in `fullwave.ports`:

```json
"fullwave": {
  "boundary": "pec",
  "frequencies": [1000000, 12000000],
  "ports": [{"name": "capacitor_1", "region": "gap_1", "toward": [1, 0, 0]}],
  "source": {"file": "segments.json"},
  "source_quadrature_order": 6,
  "source_check": true,
  "validation_shunt_R_ohm": [10, 17]
}
```

Each gap is a field volume touching conductors on exactly two edge-connected
terminal patches. Each patch belongs to one connected RF conductor body.

**Distributed ports.** A port region may instead consist of several
disjoint gap pieces, each with two terminal patches; they are fed in parallel.

- **Orientation.** `toward` orients every piece. Each piece's crossing must be
  within 60° of it (|cos θ| ≥ 0.5), which `toward` therefore must allow. Pieces
  that bridge two bodies must all connect them the same way round. A warning
  flags pieces crossed in different directions in the local (r, φ, z) frame.
  Gaps whose crossing direction turns around the device (radial or azimuthal
  gaps around a ring) need separate ports.
- **What the port computes.** Let G_k be the unit-conductivity conductance of
  piece k and w = G/ΣG. The conduction feed splits the current as w, so the port
  measures:
  - Z = wᵀ Z_p w;
  - Ψ = wᵀ Ψ_p;
  - dZ/dω = wᵀ Z_p′ w.

  Here Z_p is the pieces' impedance matrix as separate ports.
- **When this equals an ideal common bus.** The bus is
  1/(1ᵀ Z_p⁻¹ 1) only when a symmetry of the structure permutes the pieces:
  identical and symmetrically placed, not merely identical.
- **In MQS** (Z = −iωL), the split gives L ≥ L_bus. The excess is
  (w − w_bus)ᵀ L (w − w_bus), with w_bus = L⁻¹1/(1ᵀL⁻¹1).
- **Diagnostics.** The port's `pieces` list gives each piece's centroid, its
  oriented crossing, cos θ and current fraction. A warning appears when the
  fractions differ by more than 1 % (not a bus) or a piece carries under 1 %
  (likely a sliver). MQS results carry the same under `pickup.port_feed`.
- **Limits.** The gap-capacitance analysis and the COMSOL export do not handle
  distributed ports yet. The primary pickup still requires a DC return through the same
body; an additional port may bridge two separate bodies. Additional ports require
an explicit `toward` with a component along terminal separation. Separate ports
must reference separate region names; their assembled current vectors must be
linearly independent after essential-DOF elimination. A uniform-conductivity
Laplace problem in each gap sets the spatial feed profile; this is a lumped-port
modeling choice, not a resolved terminal contact impedance.

With time dependence exp(-i omega t), the discretized port solutions obey

\[
 A(\omega)e_k=i\omega b_k I_k,\qquad
 Z_{jk}=-b_j^T e_k/I_k,\qquad
 \Psi_k=b_0^T e_k/(i\omega I_k).
\]

Here `b_k` is the FE projection of a current density carrying 1 A, and `b_0`
is the projection of the prescribed static magnetic field. The transpose is
bilinear, without conjugation, as required by Lorentz reciprocity. Port voltage
is defined as minus the work of the electric field on the normalized current;
passive inductors have `Z=-i omega L`. The Hermitian part `(Z+Z†)/2` gives power
dissipation. The implementation assembles and preconditions the Maxwell operator
once at each frequency and solves every port RHS with that same operator and AMS
instance. No off-diagonal symmetrization is applied.
Small solver asymmetries can therefore give the raw Hermitian resistance a
tiny negative eigenvalue in an otherwise lossless case. The diagnostic is
retained; it is not repaired or interpreted as physical gain. A downstream
circuit interface must disclose any bounded reciprocity/passivity projection
and validate it against solver and frequency-convergence errors.

`fullwave.ports` records the port order and conduction diagnostics. The matrix exports as S-, Z- or Y-parameters
(Touchstone v1, exp(+j omega t) with Z conjugated) through `Results.write_touchstone` or `axf export-touchstone`
(`docs/physics/touchstone.md`). The export reports passivity and reciprocity and does not project them. Each sweep
row retains the existing scalar keys for port 0 and adds `Z_matrix_re_ohm`,
`Z_matrix_im_ohm`, `Psi_vector` (`re`, `im`), per-port solutions/residuals,
reciprocity error, resistance eigenvalues, and assembly/solve counts. The RF
scene remains the field driven by port 0. An additional port's static dipole
diagnostic is omitted because its separate MQS field has not been solved.

**Exact frequency derivative.** Every row with all port fields
also carries `dZ_domega_H` (`re`, `im`; H = Ω s), the exact derivative of the
full impedance matrix at that frequency, from the stored port fields with no
extra solve. With `A(ω) x_k = iω J_k`, `Z_jk = -J_jᵀ x_k / I²` and `Aᵀ = A`
(every solver path assembles a complex-symmetric operator),

    dZ_jk/dω = Z_kj/ω - (i/ω) e_jᵀ A'(ω) e_k,   e = x/I,
    A'(ω) = M[-2ω ε0 εr (1 + i tan δ) - iσ] + U diag(dc/dω) Uᵀ,

where the curl-curl term, the shell's static infinity map and its gauge mass do
not depend on ω, and the last term is the radiating exterior's low-rank map
(`axi::fw_exterior_coefficient_dw`; dγ_n/dx from the Hankel recurrence ratio,
`axi::hankel_log_derivative_dx`).

**Robin boundaries and PML.** Two more terms depend on ω, and A′ holds them:
- a Robin boundary group adds dβ/dω on its faces, from the β of `RobinCoef`.
  With y = √(ε_c/μ) and dε_c/dω = −iσ/ω²:
  - impedance with a fixed Z_s: −i/Z_s;
  - impedance with a material: −i(y + ω dy/dω);
  - scattering: the same, with the adjacent medium (the cylindrical correction
    does not depend on ω);
  - sheet: β = −2ωy tan(kd/2), differentiated through y and k = ω√(με_c);
- a PML layer adds ν dΛ⁻¹/dω to the curl-curl term, and dε Λ − ω²ε_c dΛ/dω to the
  mass term (`pml::Layer::tensors`). With the vacuum wavelength c/f every stretch
  J_i − 1 scales as 1/ω, so dJ_i/dω = −(J_i − 1)/ω. With a fixed `wavelength_m`
  the tensors do not depend on ω.

Without them dZ/dω would be wrong by 100 % on such models (on the matched TEM
line −9.5e-8 + 1.5e-7i H against a true ≈ 0). With these terms present,
`dZ_domega_scope` says so. Validation: `benchmarks/dzdw_check.py`'s `tem_*` cases and
`tests/test_fullwave3d_dzdw_terminations.py`. It holds with dielectric and conduction loss
and with `boundary: "radiation"`, where the energy identity
(`d_ImZ_domega_from_energy_H`, closed lossless only) does not, and it gives the
real part too. With the e^{-iωt} convention X = -Im Z, so dX/dω = -Im dZ/dω
(the Foster `L_eff = (ω dX/dω + X)/(2ω)` of `published_benchmarks._fullwave_rows`).
Related keys: `dZ_domega_scope`; `dZ_domega_seconds` (two mass-matrix
assemblies and products; 0.3 s per frequency, 1-5 % of the frequency time, on
the two-port validation frame); `dZ_domega_exterior_TE_H`,
`dZ_domega_exterior_TM_H` (the exterior term's share, already included in
`dZ_domega_H`; radiating boundary only); `dZ_domega_vs_energy_identity_relative`
(closed lossless only). Validation: `benchmarks/dzdw_check.py` (energy identity,
tan δ and σ slabs, radiating at kR1 ≈ 0.7 and 4.4, Richardson differences and a
Simpson consistency check; record `benchmarks/records/dzdw_check.json`) and the
C++ test `test_exterior_derivative`.

**Error indicator.** Every port solution also carries
`error_indicator` (the row's own is port 0's): an estimate of the port field's
energy-norm error, its magnetic and electric parts, where it sits (`by_region`,
`largest_element`), the conductor faces' surface term, and the estimate it
implies for the port's own Z and L (`Z_error_estimate_ohm`,
`L_relative_error_estimate`, and with sign `L_signed_relative_error_estimate`).
At order 1 it is a recovery (Zienkiewicz-Zhu) estimate; at order 2 and above
element residual problems give the magnetic part and a global
gradient projection the electric part. See
[fullwave_error_indicator.md](fullwave_error_indicator.md).

Prescribed sources are either `{"kind":"uniform","B_T":[Bx,By,Bz]}` or
`{"kind":"segments","segments_m_A":[[x0,y0,z0,x1,y1,z1,I],...]}`; `file`
loads the same JSON object. Coordinates are metres and currents amperes. Sources
are evaluated directly at physical volume quadrature points, excluding PEC
volumes and the mapped exterior shell. The finite straight-wire kernel is exact;
curved arcs and finite pack cross sections still need geometric/quadrature
convergence. The source does **not** solve superconducting screening or a
cooldown history. It represents a separately justified prescribed static
field. `azimuthal_average_samples` explicitly enables an averaging control.

The source field replaces only the full-wave overlap/source; the existing MQS
static field and its reported `coupling` remain unchanged. Consequently no
`Psi_over_MQS` is emitted for a prescribed source. MQS comparisons must
evaluate the same prescribed source with the static pickup field independently.
With `boundary: "pec"` the overlap is the volume integral alone. With
`boundary: "radiation"` the overlap outside the physical sphere R1
is added as `oint_R1 Phi_0 E_r dA`, where `B_0=-grad Phi_0` outside R1 is
represented by the static multipoles `n<=n_max` of the prescribed field's
radial component on R1; this requires closed source currents inside R1 (checked)
and excludes `uniform` sources (see `prescribed_source.md`). `source_quadrature_order=0` uses MFEM's normal
integration rule; an explicit order 1--30 selects the tetrahedral rule and is
recorded in the result. Neither setting alone establishes convergence.

Two optional small-model checks run independently driven FEM solves:

* `source_check`: solve `A e_a=i omega b_0`, then compare every measured open
  terminal voltage `-b_j^T e_a` with `-i omega Psi_j`. This is a unit source
  coefficient; physical axion normalization is supplied by the signal model.
* `validation_shunt_R_ohm`: add positive shunt conductances `Y` directly to the
  FE operator, `(A-i omega B Y B^T)e=i omega b_0port`, and compare measured
  voltages with `(1+Z Y)^(-1) Z[:,0]`. This exercises circuit closure against a
  separately solved loaded field. The direct check supports resistors,
  not capacitors or arbitrary complex loads.

These new multiport/source paths reject a nonfinite or >=1e-6 true relative
linear residual. The tolerance is a solver gate, not a field-accuracy estimate.
The two-terminal conduction solve has its own 1e-8 residual gate. Source/port
geometry, contact profiles, h/p convergence, capacitor ESL/Cmin extraction,
loaded eigenmodes, circuit loss temperatures and rational model passivity remain
separate checks for each design. Small-model evidence is in
[validation/multiport_stage1.md](../validation/multiport_stage1.md); it does not
qualify a detector-scale design.
