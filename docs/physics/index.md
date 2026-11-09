# Physics and conventions

This section states what axionFEM solves, in which units and conventions, and with which approximations. The user
guides say how to run a calculation; the [validation pages](../validation/index.md) say what each capability has been
checked against.

## Units

- **SI internally.** Every solver works in SI units; inputs carry their units in their names (`f_Hz`, `B_T`,
  `rho_GeV_cm3`, `g_GeV`, ...).
- **Natural Heaviside–Lorentz values are reported alongside**, with fields quoted in tesla and lengths (inductances
  included) in metres: Ψ in T m³ (Ψ_SI/μ0), L_p in m (L/μ0), 𝓕 in T⁴ m¹⁰. The conversions are exact consequences of
  the SI definitions (CODATA 2018); the table is in [scan_and_signal.md](scan_and_signal.md#1-units-and-conversions).
- **Time convention e^{−iωt}** for every time-harmonic quantity, so a passive inductor has Z = R − iωL. Touchstone
  files and circuit simulators use e^{+jωt}; [touchstone.md](touchstone.md) conjugates on export.
- Power spectral densities follow the normalisation ⟨X²⟩ = (1/π)∫₀^∞ S_X dω, half the one-sided-in-ν PSD
  ([scan_and_signal.md](scan_and_signal.md#32-psd-normalisation)).

## The quantities a lumped-element design is judged by

- **Coupling.** Ψ = (1/I)∫ A_I·B_lab d³x, the overlap of the pickup mode (vector potential per unit readout current)
  with the static laboratory field; the axion flux amplitude is Φ_a = κ_a Ψ with κ_a = g_aγγ √(2ρ_DM). It is computed
  by reciprocity, with whichever of A_pick·B_lab or A_lab·B_pick the potentials allow.
- **Scan requirement.** 𝓕 = Ψ⁴/L_p² must reach 𝓕_req, the value a benchmark readout needs to scan the target band in
  the live time. The built-in benchmarks are illustrative presets (a design can give its own `{name, Q, T, eta_A}`):
  `optimistic` (Q = 2×10⁷, T = 10 mK, η_A = 0.1), `conservative`
  (Q = 10⁶, 30 mK, η_A = 10) and `conservative_solenoid` (Q = 10⁵); the default scan, also an illustrative choice that a
  design's `scan` section replaces, is 3 years over 120.9 kHz–1 MHz, DFSZ with C_γ = 1.77, ρ_DM = 0.3 GeV/cm³, Q_a = 6.8×10⁵, Θ₉₅ = 2.71 ([scan_and_signal.md](scan_and_signal.md)).
- **Geometry factor** g = Ψ²/(L_p W_mag R²), with R the outer radius of all regions.
- **Cost metric.** C = a[(W/MJ)^p_E + κ(A_cold/m²)^p_A] + c_Fe M_Fe in MCHF: the two-term form of A. Hervé's magnet-cost
  law in stored energy and cold surface (as quoted, for example, in the nuSTORM proposal, arXiv:1308.6822), plus warm
  iron, with the defaults a = 0.25 MCHF, κ = 1.2, p_E = 0.7, p_A = 0.8, adjustable in the design's `cost` section.
  These defaults are illustrative (no source is claimed for them); set them for your own study. It is a metric for
  comparing designs, not a construction estimate.

## Formulations and approximations

- **Axisymmetric magnetoquasistatics.** Two scalar problems on the meridional half-plane: poloidal fields from
  ψ = rA_φ and toroidal fields from G = rH_φ (constant in every current-free region). Superconductors are ideal
  Meissner bodies, eliminated as "master" degrees of freedom (one per body, with a fixed fluxoid or a prescribed net
  current); the outer boundary carries an exact Dirichlet-to-Neumann map; nonlinear iron is solved by Newton's method
  with a line search.
- **Iron in RF problems** uses the secant permeability B/H of the full static field: exact for RF fields
  perpendicular to the static field, an approximation otherwise.
- **3D magnetoquasistatics** (MPI builds): Nédélec elements with hypre AMS, an infinity-shell map of the exterior,
  divergence cleaning with body potentials and curved elements; every Krylov solve checks its true residual.
- **Full wave beyond MQS.** The axisymmetric engine solves both m = 0 families with an exact radiation map; the MQS
  outputs are its ω → 0 limit. The 3D engine adds current ports, a prescribed source, an error indicator, a
  Gauss-law guard and an adaptive sweep (pages below).
- **Sharp superconductor corners** carry field singularities: results report the peak, the corner value and a
  "regular" peak, and warn.
- **Surface losses** are first-order perturbations of the perfectly conducting solution (Q = G/R_s), valid while
  R_s ≪ ωμ × size.
- Every result carries diagnostics (mesh size and order, DOFs, solver residuals, consistency checks). Quote a number
  only after an h or p convergence check.

## Pages

**Signal, readout and scan**

- [Readout, signal PSD, statistics and optimal scan](scan_and_signal.md) and its [C++ API](scan_cpp_api.md)
- [Readout circuit network](readout_circuit.md)
- [dc SQUID amplifier chains](squid_chain.md) (beta)
- [Multi-band and multipole scans](multipole_scan.md) (beta)
- [Multiport circuit closure](multiport.md) and [component and energy helpers](multiport_components.md)

**Magnetostatics and materials**

- [Thin high-permeability layers](thin_magnetic_layer.md)
- [Electrostatics and capacitance](electrostatics.md)
- [Surface resistance of RF walls](surface_resistance.md) (beta)

**Full wave**

- [3D current ports and prescribed sources](multiport_fullwave.md) (the prescribed source is beta)
- [Prescribed static source](prescribed_source.md) (beta)
- [Imposed background field](fullwave_background.md) (beta)
- [Error indicator](fullwave_error_indicator.md) (beta)
- [Gauss-law guard](fullwave_gauss_guard.md) (beta)
- [Tree-cotree gauged solve](fullwave_tree_cotree.md) (beta)
- [Surface losses](fullwave_surface_losses.md)
- [Adaptive frequency sweep](adaptive_sweep.md) (beta)
- [S-, Y- and Z-parameters and Touchstone files](touchstone.md)
- [2D full wave on imported meshes, Robin boundaries and PML](fw2d.md) (beta)

**Cavities**

- [Cavity-haloscope sensitivity](haloscope_sensitivity.md) (beta)
- [Receiver coupling through an antenna port](../cavity/coupling.md) (beta)
- [SRF heterodyne cavities](../srf/README.md), [sensitivity model](../srf/sensitivity.md) and
  [wall vibrations](../srf/vibrations.md) (beta)

**Mechanics and thermal**

- [Elasticity, conduction and torsional dynamics](mechanics_thermal.md) (torsional dynamics: beta)
- [High-frequency torsional mobility](torsion_highfreq.md) (beta)
