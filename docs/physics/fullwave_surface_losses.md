# Surface losses of the 3D full-wave readout

The 3D full-wave sweep reports, per frequency and per port, the RF surface losses of every conductor and
superconductor surface. Beyond the MQS regime the pickup's current distribution changes with frequency, and its
loss and Q change with it. The MQS engines compute the same quantity as `pickup.losses`; the axisymmetric full-wave
engine computes it per frequency as well. Code: `cpp/src/model3d_surface_losses.cpp` and
`cpp/src/surface_losses3d.hpp` (`FwSurfaceLosses`). Benchmark: `benchmarks/fullwave_surface_losses.py`, record
`benchmarks/records/fullwave_surface_losses.json`.

!!! note "Beta: the surface-resistance models"
    The loss integrals Γ below are checked against closed forms and the axisymmetric engine. The surface-resistance
    models R_s(f) that turn Γ into R_loss and Q (normal and anomalous skin, BCS, flux flow;
    [surface_resistance.md](surface_resistance.md)) are beta.

## What is computed

The solve treats every conductor and superconductor as a perfect conductor (E = 0 inside). The loss is evaluated
afterwards from that field (first-order perturbation theory, below). For port k at angular frequency ω, with the
port field E_k per incoming ampere:

- **Γ per surface** (`surface_Gamma`, dimensionless: H per ampere is in 1/m): Γ_b = ∮_b |H_k|² dA / |I|².
  - H_k = curl E_k / (iωμ) is taken on the field side of each face, with that side's μ. That is the full-wave
    solve's own μ: μ0 μ_r for linear and iron materials. The full-wave solve has no static bias of iron.
  - The faces are those of `pickup.losses`: a conductor or superconductor element on one side and a field element
    on the other. Faces to the infinity shell are left out. Faces to a PML layer are left out too, with a warning.
  - The quadrature is a triangle rule of degree 2p + 2 on the curved faces.
  - `Gamma_by_field_region` splits Γ by the field-side region (cavity, port, open slit gap).
- **Magnetoresistance.** A surface whose `rf` block has a magnetoresistance model also gets
  `Gamma_magnetoresistance`: Γ weighted by R_s(f, B)/R_s(f) = `Rs_factor(B)` at the local static |B| (B_lab and an
  imposed background), as in `pickup.losses` and the 2D engine. The weight never evaluates R_s itself, so a measured
  R_s table works with it, also one that does not reach 1 Hz. Without a static solve (`fullwave.rf_only`) the weight is not applied,
  with a warning.
- **Per surface with rf data** (`surface_losses.surfaces`):
  - `R_s_ohm` = R_s(f);
  - `R_eff_ohm` = R_s(f) Γ (or Γ_magnetoresistance);
  - `Q` = ωL/R_eff and `Q_tuned` = ωL_eff/R_eff;
  - `T_over_Q_K`.
- **Per row** (the keys of the axisymmetric engine's sweep rows, plus the tuned Q):
  - `R_loss_ohm` = Σ_b R_eff,b over the surfaces with rf data: 2P_loss/|I|², the power the conductor surfaces
    dissipate (to first order the series resistance they add to Re Z);
  - **`Q_loss_tuned`, the headline Q** (next section), with `L_eff_H`, `Q_total_tuned` and `T_over_Q_tuned_K`;
  - `Q_loss` = ωL/R_loss, the reactance's Q (the key shared with 2D and MQS);
  - `Q_total` = ωL/(R_loss + max(Re Z, 0)). A true relative residual ρ can move Re Z by up to about ρ|X| (an
    estimate, not a bound: the largest residual-driven error in the runs was 0.2 ρ|X|). So `Q_total` and
    `Q_total_tuned` are withheld, with the reason in `surface_losses.Q_total_withheld`, where ρ|X| exceeds 1 % of
    R_loss + max(Re Z, 0).
    - **Where the conductor loss dominates**, `Q_total` is resolved even when Re Z is not. The solenoid template keeps
      `Q_total` at 45 kHz and 1 MHz, where it equals `Q_loss` to better than 1e-4 (ρ|X|/R_loss = 1.6e-5 and 7.5e-5).
    - **Where radiation must carry it**, it is withheld: deep in the MQS regime a radiation resistance (∝ f⁴ for a
      magnetic dipole) falls below ρ|X| first.
    - **A rule on Re Z/|X| alone** (Re Z/|X| ≥ 100 ρ) would also withhold the solenoid's resolved rows. Their Re Z is
      physical: it follows the 2D engine's f⁶ law to −13 % and +1.6 %.
  - **Re Z itself** is flagged in the row (`Re_Z_unresolved`, with a warning) where a dissipating model's Re Z is below
    100 ρ|X|. `R_rad_ohm` and `Q_rad` come from Re Z and share the flag; `R_bulk_ohm` and `R_boundary_ohm` are
    integrals of the field and do not.
    - **Remedy:** a tighter GMRES tolerance (`AXF_FW_RTOL`) lowers ρ, as far as double precision allows. A residual of
      1.3e-12 resolved the order-1 solenoid at 1 MHz, but not at 45 kHz, which would need ρ < 6e-19. The direct solver
      is not a remedy: it refuses radiating designs (closed PEC spheres only);
  - `T_over_Q_K` = Σ_b T_b R_eff,b/(ωL), over the rf surfaces only: radiation and bulk loss enter `Q_total` but not
    T/Q, as the engine does not know their temperatures. Each surface is weighted by T_K, the classical
    limit; `axionfem.circuit` uses the occupation n(ω, T);
  - `R_loss_over_abs_X` = R_loss/|Im Z|, the perturbation parameter (section Validity).

  Here L = −Im Z/ω (Z = R − iωL under e^{−iωt}). The Q values are omitted where L ≤ 0, above the first
  self-resonance, as `F_SI_T4_H2_m8` is.
- **Multiport** (in the primary row): `surface_Gamma_matrix` per surface, the Hermitian ∮ H_j*·H_k dA, and
  `R_loss_matrix_ohm` = Σ_b R_s,b ∮ H_j*·H_k dA.
  - **The matrix is the dissipated power:** ½ I† R_loss I for any combination I of port currents, exact to first order.
  - **It is not the impedance perturbation in general.** For a reciprocal network that is the unconjugated, symmetric
    δZ_jk = Σ_b Z_s,b ∮ H_j·H_k dA. The two agree only when the port fields per ampere are in phase, as in a closed
    lossless structure.
  - **With radiation or bulk loss do not add the matrix to Z:** its imaginary off-diagonal parts would break
    reciprocity at first order in the fields' phase spread. A single port's `R_loss_ohm` is fine: there the two forms
    agree to second order.
- **Summary** `fullwave.surface_losses`: the surfaces with their materials and field-side regions, face and point
  counts, the surfaces without rf data, the largest perturbation parameter, `H_normal_share_max`, the rows with
  `Q_total` withheld, the PEC boundary groups, the scope and the cost. Port rows whose solve failed its residual gate
  keep their losses (from the unconverged field) but are counted in `port_rows_with_unaccepted_solve` and left out of
  the maxima. A model whose conductors are all PEC boundary groups gets the summary and a warning that no surface
  loss is computed.

**Python.**
- **`R_loss_ohm` in the rows.** `circuit.fullwave_tables`, `circuit.pickup_impedance(model="fullwave")` and
  `scan.reach_fullwave` read it. They were written for the axisymmetric engine and now take a 3D sweep's surface loss
  as well.
- **What that changes for 3D.** A 3D reach now adds R_loss to the benchmark's ωL_p/Q_other, so Q_other must exclude
  the computed conductor loss. A benchmark Q that stands for the total intrinsic Q counts it twice. The reach's `Q_radiation_limited` holds radiation, bulk and
  surface loss (`Q_radiation_limited_note`).
- **The exact L_eff in the reach.** `fullwave_tables` carries `L_eff_H` when every row has it: every 3D design whose
  conductors are meshed as volumes and whose dZ/dω is exact, with or without rf data. So the reach of all such designs
  changed in this round, not only of those with rf blocks.
  - **The full-wave scan** takes L_eff between the rows from a cubic Hermite interpolant of L with the rows' exact
    slopes dL/dln f = 2(L_eff − L).
  - **The guard.** It must agree with the linear interpolation of the rows' L_eff within 5 % at every scan frequency,
    or the reach refuses: the rows are too sparse for the pole slope.
    - This replaces the other path's guards (at least four rows, and spline against PCHIP within 5 %).
    - On the sealed coax's closed form it refuses the benchmark's rows, which reach to 1.8 % below the pole (linear
      L_eff there is up to +104 % off), and the two-row sweep.
    - It accepts rows far below the pole.
  - **The MQS comparison** keeps L_p (giving it the rows' L_eff would make `mqs` and `ratio_g95_fullwave_over_mqs`
    wrong for such designs).
- **`Results.summary()`** prints the tuned Q, and marks R_rad and Q_rad where Re Z is unresolved.
- **An unresolved Re Z in the tables.** `fullwave_tables` warns where a row's `Re_Z_unresolved` can move R_ohm by more
  than 1 %.
- **R_loss between rows.** `fullwave_tables`, `pickup_impedance` and the reach interpolate R_loss log-log.
  `circuit.loss_interpolation_check`, called by the first two, warns where that is unreliable. It flags an interval in
  either of two cases:
  - **The slope:** R_loss/|X| changes faster than f³ (near a self-resonance R_loss grows like 1/Δ², X like 1/Δ).
  - **The estimated error** of the log-log R_loss exceeds 5 % (the slope alone misses 100-140 MHz on the coax rows,
    where R_loss is 38 % off).
    - **With the rows' exact L_eff (3D):** near a pole R_loss follows ωL_eff, so 1/Q_tuned = R_loss/(ωL_eff) varies
      slowly. The estimate compares the log-log R_loss with ωL_eff/Q_tuned, where L_eff comes from the Hermite
      interpolant and 1/Q_tuned is interpolated log-log.
    - **Otherwise (2D):** the curvature of ln R_loss at the neighbouring rows times the interval squared over 8. At the
      ends of a run of rows, the neighbouring interval's line is also extrapolated to the midpoint.
  - **On the closed form:**
    - with L_eff it flags every interval above 5 %, plus one at 4 %;
    - without L_eff it misses only a lone two-row interval, which has nothing to estimate from.
  - Add solves where it warns.
- **`circuit.fullwave_losses(results, port)`** (`Results.fullwave_losses`) returns the columns as arrays.
  - The 2D engine has no exact dZ/dω, so its tuned columns are NaN.
  - A finite difference of L between the rows is not offered: on the coax benchmark's 2D rows it is −21 % to +106 % off
    the closed form, and on the solenoid's +76 % and −69 %. An exact 2D dZ/dω by reciprocity would be the way to add
    it.

## The quality factor

In the MQS engine Q = 2πf L_p/R_eff. Beyond MQS the port reactance X(ω) = −Im Z = ωL(ω) is no longer ωL_p, and two
continuations of the MQS Q are reported.

- **`Q_loss` = ωL(ω)/R_loss** replaces L_p by L(ω). It is the Q of the pickup as an inductor with the reactance it
  has at f, and it is the axisymmetric engine's `Q_loss`. Near the pickup's first self-resonance (a pole of Z) it
  is misleading: L(ω) diverges like 1/Δ in the distance Δ to the pole, R_loss like 1/Δ², and Q_loss → 0.
- **`Q_loss_tuned` = ωL_eff/R_loss, with L_eff = L + (ω/2) dL/dω = ½(X/ω + dX/dω).** This is the Q of the readout
  resonance when an ideal series capacitor tunes the port to f.
  - The resonance's Q is (ω/2R) d(X + X_C)/dω at X + X_C = 0. The capacitor adds dX_C/dω = 1/(ω²C) = X/ω.
  - For a closed lossless structure Foster's theorem gives dX/dω = 4(W_m + W_e) and X = 4ω(W_m − W_e) per ampere²
    (time-averaged energies). So L_eff = 4W_m and Q_loss_tuned = 2ωW_m/P_loss: the stored energy of the tuned
    circuit over the loss per radian. For the discrete closed lossless system this is an identity, so the check
    L_eff = 4W_m (coax, 3e-12) verifies the dZ/dω and energy-Gram code, not the accuracy of L_eff; the closed form
    does that.
  - L_eff is the tuned-pole inductance of `scan.reach_dispersive`. Here it comes from the exact per-frequency
    dZ/dω of the sweep (`dZ_domega_H`), not from a finite difference.
  - **Which derivative is used.** As in the adaptive sweep, it is used only when its `dZ_domega_scope` starts with
    "exact"; otherwise `L_eff_H` and the tuned Q's are omitted. A′(ω) includes the Robin groups' dβ/dω and the PML's
    d/dω, so the scope is exact for every termination.
  - **The losses add nothing to A(ω).** They are post-processing of the perfectly conducting solution, so they leave
    the system matrix, A′(ω), dZ/dω and its "exact" scope unchanged.
  - **The validation here** has no Robin groups or PML: a closed sphere or the radiating exterior.
  - Near a self-resonance it tends to the Q of that resonance itself: a finite, physical number.
- In the MQS limit both equal ωL_p/R_eff. They differ by the factor 1 + ½ d ln L/d ln ω, which departs from 1 at
  order (kR)².
- **`Q_loss_tuned` is the headline Q.**
  - **Exactness:** on an exact lossy transmission line it equals the capacitor-tuned resonance's Q to ≤ 3e-8 up to
    0.6 % below the pole, while `Q_loss` is 7 % low at 50 MHz, 32 % at 100 MHz and 99 % there.
  - **Where it leads:** `Results.summary()` prints it, and the reach uses the same exact L_eff.
  - **`Q_loss` stays** as the key shared with 2D and MQS: the reactance's Q.
  - **Its limits:**
    - the capacitor is ideal and lossless;
    - L > 0 (above the first self-resonance an inductor would tune the port; that Q is not reported);
    - a broadband, untuned readout needs R_loss itself.
- `Q_total` and `Q_total_tuned` add max(Re Z, 0) to R_loss: radiation, and the bulk, Robin and PML losses of the
  solve itself.

## Validity: a perturbative loss

- **The perturbation.** The field is that of the perfectly conducting structure, and the surface resistance enters
  only through the dissipated power ½R_s∮|H|². The first neglected order is the change of the field by the loss.
  - **Near a pole of Z** (Z ≈ iA/(Δ + iγ), A real, for a resonance of half-width γ at distance Δ): the exact R is
    Aγ/(Δ² + γ²) and the perturbative one is Aγ/Δ², so the relative error is (γ/Δ)² = (R_loss/|X|)².
  - **Far from a pole:** the same parameter bounds the relative change of the current distribution.
  - `R_loss_over_abs_X` reports it. Rows above 0.1 (a 1 % error) are counted in the summary and warned about.
  - **Where X crosses zero** (a series resonance of the port, which occurs only above its first self-resonance) the
    parameter grows without the field being resonant: the current is prescribed and the field stays finite. A flag
    there is conservative. Below the first self-resonance (L > 0) it is 1/Q_loss.
  - The perturbation is therefore valid for Q ≫ 1, outside a few half-widths of a resonance. On the coax below it is
    1.7e-4 at 1.8 % below the pole.
- **Surface impedance.** R_s(f) of the `rf` model assumes a skin depth (or penetration depth) small compared with
  the curvature radii and thicknesses. The reactive part of the surface impedance is not added.
  - **The internal inductance of a normal metal** (X_s = R_s) is a relative correction R_loss/(ωL) = 1/Q to L.
  - **The kinetic inductance of a superconductor** (X_s = ωμ0λ ≫ R_s) is δL/L = μ0λΓ/L = (X_s/R_s)/Q, larger than 1/Q
    by X_s/R_s (10²-10⁵ at MHz). It is still small: λ over the field-weighted length L/(μ0Γ).
- **Not counted:**
  - **PEC boundary groups:** conductors meshed as holes (`mesh.remove_pec`) and imported PEC groups. They are listed
    in the summary, with a warning when such a conductor has rf data or when they are all the conductors there are.
    Mesh the conductor as a volume for its loss.
  - **A closed outer sphere** (`boundary: pec`): it stands for a lossless, perfectly conducting room.
  - **Symmetry planes.**
- **Lossy volumes.** Re Z already contains the bulk loss (`R_bulk_ohm`), Robin boundary groups (impedance walls) and
  PML absorption. R_loss adds the conductor surfaces, which none of those contains, so nothing is counted twice.
  With bulk loss or a radiating exterior the field has a spatially varying phase. ∮|H|² is still the dissipated
  power at first order.
- **Symmetry cuts.** With `mesh.symmetry` the meshed part carries the port current as in Z (no image scaling), so
  R_loss and Z refer to the same model and Q is that of the full device.
- **The exterior.** `boundary: pec` is a perfectly conducting room; `radiation` is free space. For a readout whose
  field leaks (a partial port, an open slit) both are idealizations, and the real environment is neither.
  - **The closed room** shifts L by the image of the leakage dipole, −(μ0/2π)|m|²/a³ for a sphere of radius a, and
    hides the radiation. For the small slotted box's 30° port at kR = 0.06, L/L_MQS is 0.9988 in its closed sphere
    against 1.0037 with the radiating exterior.
  - **Free space** has the leak radiate as a magnetic dipole (Validation, "A partial port").
  - **To get the environment's loss,** model the shield or vessel as a meshed conductor with an rf block, or use a
    feed with no dipole moment (N ≥ 2 equally spaced modules).

## The field on the surface

- **H_n.** On a face of a perfect conductor element the tangential E vanishes, so n·curl E (the surface curl of E_t)
  is zero to rounding: |H|² = |H_t|². `surface_losses.H_normal_share` (∮|H_n|²/∮|H|²) checks it per row.
- **Curl representatives.** Where the solve has a curl representative of E (the Gauss-law guard's E − G_c S⁻¹G_cᵀME,
  or the tree-cotree path's S_c e), the curl is taken of it. Its curl is the same, without the rounding of a large
  gradient part.
- **The same faces as MQS.** The face selection, the rule, the field-side μ and the magnetoresistance weight are
  shared with `pickup.losses` (`surface_losses3d.hpp`); the record checks that the MQS results are bit-identical with
  the shared code (`extra.mqs_pickup_losses_bit_identical`), the magnetoresistance case included.
- **Errors on one rank.** Only the ranks holding a face's field side visit it. An error while the faces are set up is
  caught, agreed on and raised on every rank before any collective.

## MPI

- **The faces** are found on the serial mesh, identically on every rank.
- **Each rank** keeps the quadrature points whose field-side element it holds. Every local element has the
  reference map of its serial element, which the partition checks.
- **One reduction** (MPI_Allreduce) per frequency sums the per-surface, per-region and per-port parts, so every
  rank reports the same numbers.
- **No field is gathered,** unlike the MQS losses, which use the gathered serial view once.

## The adaptive sweep

The adaptive frequency sweep (`fullwave.adaptive`, [adaptive_sweep.md](adaptive_sweep.md)) fits Z and Ψ with a rational
model. The losses are not part of that model.

- **What an adaptive run reports.** The losses are evaluated at the solved frequencies only: the rows of
  `fullwave.sweep`. The model and its dense curve carry Z and Ψ, not R_loss. `circuit.fullwave_tables` and
  `fullwave_losses` read the solved rows, so a reach or circuit built from them interpolates R_loss between the
  solves.
- **Why the dense curve does not carry them.**
  - **Pole structure:** Γ(f) has a double pole wherever Z has a simple one, since H ∝ 1/(f − f_p). Fitting it would
    need its own model with the squared denominator, or the fields of the residues.
  - **Validity:** within a few half-widths of those poles the perturbative loss fails anyway (R_loss/|X| → ∞).
  - **Placement:** the solves sit where Z and Ψ need them, not where Γ does.
- **What to use near a resonance.** A solved row near a pole gives the resonance's Q as its `Q_loss_tuned`, which keeps
  its accuracy there (the coax below). Without such a row, compute the Q from the eigenmode (the cavity path's
  perturbative wall loss) or with lossy walls in the solve itself.
- **Future work: pole damping.** If dense losses are wanted, carry them as pole damping, not as a fitted R_loss(f).
  - For each model pole p_n, take Q_n from the solved row nearest to it; on the lossy line `Q_loss_tuned` is within 6 %
    of the resonance's Q at 14 % below the pole and 0.7 % at 1.8 % below.
  - Then p_n → p_n(1 − i/2Q_n), plus a background R_loss interpolated away from the poles, gives a dense lossy Z to
    first order in 1/Q, peaks included. Q_n would go into `self_resonances`.
  - Meanwhile `circuit.loss_interpolation_check` warns where rows are too sparse for R_loss.
- **Away from poles** R_loss is smooth and slowly varying: R_s ∝ f^{1/2} or f², and Γ changes at order (kR)².
  Interpolating the solved rows is accurate there.
- **Where the sweep is long.** Add solve frequencies where the losses matter, as `fullwave.adaptive` initial points.

## Validation

`benchmarks/fullwave_surface_losses.py`, record `benchmarks/records/fullwave_surface_losses.json`.
- **Runs.** The record names the build and run of each case. Its runs come from builds with the same loss
  evaluation: reruns of a subset on a later build reproduce every Γ, R_loss, L and Q of the earlier ones.
- **`Q_total`.** The solenoid's is kept at 45 kHz and 1 MHz, where it equals `Q_loss` to 1e-4, with its Re Z flagged
  (`Re_Z_unresolved`).
- **Designs** are those of the benchmark script; the slotted box is the `mini` box of `benchmarks/slotted_box.py`
  (subcommand `slotted`; its record keys are labelled `abra`).

### A closed form at every frequency: the sealed coax

**The design.** A copper coax (inner radius a = 0.08 m, outer b = 0.2 m), shorted at both ends. A radial port slab
(height g = 2 cm) splits it into two stubs, l1 = 0.6 m and l2 = 0.3 m. The design closes the domain with a PEC
sphere.

**Why the field is exact TEM.** The slab's feed J_r ∝ 1/r, uniform in z, is the TEM profile, and the end walls are flat.
So the field is TEM at every frequency: H = I(z)/(2πr), with I'' + k²I = −i_s'(z), i_s = I/g on the slab, and I' = 0 at
the shorts.
- Γ of the walls is ∫|I|²dz/(2πa) and ∫|I|²dz/(2πb); that of the end plates is |I(end)|² ln(b/a)/(2π).
- Z, L_eff and the tuned Q follow from the same solution.
- The pole of Z is the half wave of the whole length, 162.93 MHz.
- The sweep runs from 1 MHz to 1.8 % below the pole (160 MHz) and above it (175 MHz).

**The axisymmetric engine** (order 3, 61k DOFs) matches this closed form to ≤ 5.5e-7 in every Γ and
2.6e-7 in L, at every frequency including 160 MHz. That confirms the closed form, and the 2D engine's own full-wave
losses.

**3D, order 2.** Four levels; the surface size goes 0.04 m → 0.014 m by √2 per level, 0.8-2.7M DOFs.
The relative error against the closed form:

| f | surface | level 0 | level 1 | level 2 | level 3 | order-2 extrapolation (levels 2, 3) |
|---|---|---|---|---|---|---|
| 1 MHz | inner wall | −5.3e-2 | −2.7e-2 | −1.7e-2 | −8.8e-3 | −5.4e-4 |
| | outer wall | −1.5e-2 | −6.8e-3 | −3.3e-3 | −1.9e-3 | −5.1e-4 |
| | end plates | −2.2e-3, −3.0e-3 | −1.4e-3, −1.0e-3 | −4.2e-4, −2.8e-4 | −1.6e-4, −1.2e-4 | +1.1e-4, +3.4e-5 |
| 160 MHz | inner wall | −3.7e-2 | −2.1e-2 | −1.5e-2 | −8.0e-3 | −1.2e-3 |
| | outer wall | +2.1e-3 | −1.4e-3 | −1.4e-3 | −9.4e-4 | −4.6e-4 |
| | end plates | +1.7e-2, +1.6e-2 | +5.3e-3, +5.5e-3 | +2.3e-3, +2.3e-3 | +1.1e-3, +1.1e-3 | −2e-5, −7e-5 |
| | L | +8.0e-3 | +2.6e-3 | +9.6e-4 | +4.3e-4 | |
| | `Q_loss_tuned` | +3.6e-2 | +1.7e-2 | +1.1e-2 | +5.6e-3 | |

- **The order-2 extrapolation** from the two finest levels lands within 1.2e-3 of the closed form on every surface, at
  every frequency.
- **Observed rates between levels.** The outer wall goes at 1.6-2.3, and L at 2.5-3.2.
- **The inner wall** (12.6 to 36 faces around a 1/r field) goes at 1.3-1.95. Its three-level Richardson order is not
  usable (0.4-0.8), so the formal order 2 is the extrapolation to quote there.
- **Near the pole:**
  - L and the end plates carry the discrete pole's shift: δf_p/f_p ≈ −1.4e-4 at level 0. Γ then moves by twice that
    over the distance to the pole, 2δf_p/Δf.
  - `Q_loss` falls to 0.036 of `Q_loss_tuned` at 160 MHz.
  - **`Q_loss_tuned` keeps its off-pole accuracy** (+5.6e-3 at level 3 there, +6.0e-3 at 1 MHz), because the tuned
    Q of a resonance does not depend on where exactly the pole sits. It tends to the resonator's Q, 1.65e5 at
    160 MHz.
- **L_eff = 4W_m** (Foster), from dZ/dω against the energy Gram of the same fields, to ≤ 3.3e-12 at every level and
  frequency. For the discrete closed lossless system this is an identity, so it checks the code, not the accuracy;
  the closed form above does that (+0.56 % at level 3).
- **H_normal_share** ≤ 1.6e-27.

### The MQS limit

Γ(f) at low frequency against `pickup.losses` of the same run (the same mesh). The physical change is removed with the
closed form on the coax, and is (kR)² small elsewhere.

| design | elements | f (kR) | full wave / MQS − 1, after the physical change |
|---|---|---|---|
| coax, order 1 | straight | 1 MHz | 4e-7 to 1.5e-6 (default-suite test) |
| coax, order 2 | straight (geometry order 1) | 1 MHz | −6.7e-8 to +1.5e-7 (`coax_straight`) |
| coax, order 2 | curved, levels 0-3 | 1 MHz | −6.5e-4 → −1.3e-4 (inner), −2.0e-4 → −2.7e-5 (outer), ≤ 1.7e-4 (plates) |
| solenoid template, order 2 | curved | kR = 1e-3 | −2.0e-4, −9.8e-5, +4.0e-5 (inner, outer, short) |

- **On straight elements** the two engines' static limits are one discrete problem, and they agree to rounding of the
  solves.
- **On curved elements** they differ by 1e-4 to 6e-4. The MQS curl-curl uses a degree 2p+1 rule on meshes with the
  infinity shell; the full-wave operator uses MFEM's default. On curved elements that changes the discrete operator.
  The difference shrinks under refinement. It belongs to the engines' operators, not to the loss evaluation, which
  shares its faces, rule and μ with `pickup.losses`.

**A partial port: a magnetic dipole, a closed sphere's image, and neither is the experiment.** One 30° port on the
slotted box (n1) is a magnetic-dipole antenna.
- **The mechanism.** The current that crosses the slit must spread around the circumference before it closes as the
  uniform poloidal current of the cavity field. It spreads along the lips and the box's outer faces.
  - Those azimuthal currents carry a net moment, tangential at the port (the MQS `mqs_magnetic_moment_per_A_m2`).
  - The uniform feed (c360) has none, and its field stays in the cavity.
- **In a closed PEC sphere** of radius a the moment's image lowers L by (μ0/2π)|m|²/a³ per ampere².
- **The two exteriors on the small box** (n1, kR = 0.06): L/L_MQS 0.9988 in the sphere against 1.0037 open, and the
  open gap's Γ −1.7 % against −2e-4.
- **In free space the leak radiates as a magnetic dipole,** R_rad = μ0 c k⁴|m|²/(6π) for the MQS moment, so Q_rad
  falls like f⁻³; at low frequency the residual cannot show such a Re Z to be resolved (`Re_Z_unresolved`), and
  `Q_total` is withheld (above).
- **Neither idealization is the experiment's Q, nor a bound on it.** At MHz frequencies the wavelength is hundreds of
  metres, and the box sits inside a cryostat, next to its magnet and, in some designs, warm iron close to the slit. The
  same leakage field reaches all of them, and the loss it drives depends on their surface resistance and distance, not
  on free space.
  - **Far conductors see the dipole.** A normal-metal shield adds a loss set by its R_s and the dipole field there; a
    superconducting shield adds no loss but shifts L, as the PEC sphere does.
  - **Near conductors see the near field**, which is much stronger. Iron near the slit could dominate the loss by
    orders of magnitude. Neither this engine nor the model can say how much: iron is not a loss surface here, and the
    box model has no iron.
- **Consequences for a readout across part of the slit.**
  - **A single port's Q cannot be assumed** without modelling the shield, the cryostat and the iron.
  - **N ≥ 2 equally spaced modules cancel the dipole** by symmetry, leaving quadrupole radiation (∝ f⁶).
  - **This feature computes the environment's loss** once the shield or vessel is meshed as a conductor with an rf
    block (the outer PEC sphere carries none).

### Against the axisymmetric engine

- **The coax** (above): 3D order 2 converges to the 2D engine, which equals the closed form.
- **The solenoid template** (order 2, 0.9M DOFs, radiating exterior; 2D: order 2).
  - **A frequency-independent offset** of −1.5 %, −0.66 % and −0.07 % (inner, outer, short) up to 10 MHz. It is the
    3D mesh's resolution of the 1/r field, the same as for `pickup.losses` (benchmarks/rf_losses_3d.py).
  - **At 30 MHz**, 0.77 of the coax's quarter-wave resonance, the 3D L is 3.3 % above 2D and Γ 5-7 %. That is the
    discrete pole's shift, amplified as on the coax.
  - **Radiation matters there:** the open coax mouth radiates (R_rad = 9.2 Ω), and `Q_total` is 20 in both engines.

### The error indicator's surface term

On the coax levels the indicator's `surface_relative_error` η_s (the recovery error of curl E on the conductor faces)
tracks the Γ error at a near-constant ratio.
- **Off the pole.** At 1 MHz the total Γ is 3.7e-2, 1.9e-2, 1.2e-2 and 6.0e-3 low on levels 0-3, against
  η_s = 1.4e-2, 8.0e-3, 4.7e-3 and 2.7e-3: |ΔΓ/Γ| = (2.2-2.6) η_s.
- **Near the pole** the ratio is 1.3-2.0. There the pole's shift, which η_s does not see, partly cancels the profile
  error.
- **2η_s is not a bound.** The Cauchy-Schwarz form 2η_s + η_s² would be one if η_s were the true surface error. Here it
  underestimates the error by up to 30 %: the recovery underestimates a systematically low surface field.
- **As a size indicator** it orders the meshes correctly.
- **Independence.** The losses do not use the indicator; it is only compared here. At order 2 and above the volume
  estimate is the residual-problem and gradient-projection estimator, but `surface_relative_error` is the recovery at
  every order ([fullwave_error_indicator.md](fullwave_error_indicator.md), "Surface term"), so these numbers apply.

### Multiport, MPI, adaptive

- **Multiport.** On the two-port frame (default-suite test) the Γ matrix is Hermitian, its diagonal is each
  port's Γ, and the R_loss matrix's diagonal is each port's R_loss.
- **MPI.**
  - **Curved order 2:** the coax at level 0 on 4 ranks (the slow test's fixture) against 16 ranks
    (the benchmark) gives every Γ to ≤ 1.1e-11, R_loss to 9e-12 and L to 5e-12 at 100, 160 and 175 MHz.
  - **Order 1:** one process against three ranks, in the slow test `test_ranks_agree`.
  - **The MQS results** with the shared loss code are bit-identical to the MQS engine's own on 2 ranks, on all three
    designs, the magnetoresistance one included (`mqs_identity`; the record's `largest_relative_difference` is 0).
- **Adaptive.** An adaptive sweep's rows carry the losses and its dense curve does not (slow test).
- **Magnetoresistance.** The solenoid template with a Kohler model on the coax copper, with the classical skin
  model and with a measured R_s table (1 kHz to 1 GHz) (default-suite test, order 1, kR = 1e-3).
  - **The weights agree:** the full-wave weights Γ_magnetoresistance/Γ (1.71, 1.85, 1.48 for the skin model) equal those of `pickup.losses` to ≤ 2e-8. The test holds them to 1e-6, and R_eff takes the
    weighted Γ.
  - **The table case runs:** the weight never evaluates R_s at 1 Hz.
- **Solve paths.** On the order-1 coax at 0.1, 1 and 100 MHz (`coax_paths`)
  every Γ, R_loss and tuned Q is compared with the default path (AMS with the Gauss-law guard, the curl of the guard's
  representative):
  - the guard off (the curl of the solver's own E) agrees to 1.1e-15. The gradient part has no curl;
  - plain STRUMPACK agrees to 1.8e-11;
  - the tree-cotree gauge (the curl of S_c e) agrees to 1.8e-11.

## Cost

- **Per frequency:** one curl evaluation per port, real and imaginary part, at each surface quadrature point of the
  rank's faces, then one reduction.
- **Setup:** one pass over the serial mesh's faces (0.01-0.16 s in the record's runs).
- **Measured:** 0.06-0.17 s per frequency, at most 0.3 % of the frequency's time:
  - coax level 3: 0.47M points, 0.11 s;
  - solenoid: 75k points, 0.12 s;
  - the slotted box: 56k points, 0.08-0.17 s.
