# Scope and limitations

This page lists what axionFEM covers, how far each capability has been checked, the approximations it makes, and
what it does not do. "Validated" means that the test suite checks the capability against closed-form solutions,
manufactured solutions, an independent solver or published results; the [validation pages](../validation/index.md)
give the references and tolerances. "Beta" means that the capability works and is tested, but its interface or its
validation is not complete.

## Status by capability

| capability | status | build |
|---|---|---|
| Axisymmetric magnetoquasistatics: static field, pickup mode, Ψ, L_p, F, peak fields, forces | validated | any |
| Nonlinear iron, superconductors (fluxoid and current constraints), persistent circuits, background fields | validated | any |
| Axisymmetric full wave: Ψ(ω), L(ω), radiation, eigenmodes of the closed domain, bulk losses, surface-loss integrals | validated | any |
| Surface-resistance models R_s(f) (normal and anomalous skin, BCS, flux flow) that turn the surface-loss integrals into losses and Q | beta | any |
| 3D magnetoquasistatics; 3D full wave with current ports and multiport impedance matrices | validated | MPI |
| 3D full wave: prescribed static sources, the imposed background field, the error indicator and Gauss-law guard (on by default), perfectly matched layers, 3D eigenmodes; adaptive frequency sweep and tree-cotree gauged solve (opt-in) | beta | MPI |
| 3D full wave: closed RF-only models (`fullwave.rf_only`: symmetry surfaces, removed PEC volumes, lumped gaps, direct factorization) and the energy-matrix, field-export and reactive-load diagnostics | beta | MPI |
| Readout circuits, scan reach, finite-band scans, Touchstone export | validated | any |
| Multi-band and multipole scans (`axionfem.multipole`) | beta | any |
| dc SQUID amplifier chains | beta | any |
| Adjoint shape sensitivities, local and global optimization | validated | any |
| Conductor critical surfaces, HTS critical state, thin magnetic layers | validated | any |
| Electrostatics (capacitance matrices; axisymmetric and 3D) | validated | any (3D: MPI) |
| Elasticity with magnetic and cooldown loads, conduction with k(T) (axisymmetric and 3D) | validated | any (3D: MPI) |
| Torsional dynamics | beta | any |
| Cavity eigenmodes (axisymmetric, cross-section, 3D), wall models, tuning and mode tracking, haloscope sensitivity and presets, receiver coupling | beta | any (3D: MPI) |
| SRF cells, mode pairs, figures of merit, heterodyne sensitivity | beta | any (3D: MPI) |
| COMSOL export | available | any |
| COMSOL `.mph` import, 2D full wave on imported meshes | beta | any |
| Fluids: Stokes, steady and transient Navier-Stokes, GLS, inductionless MHD, scalar transport, Boussinesq | beta | MPI |
| Chemistry: kinetics, reactors and equilibrium; current distribution; porous electrodes | beta | any (3D: MPI) |

The sequential build (the configuration of the planned PyPI wheels) has no MPI: everything marked "MPI" needs the
[MPI source build](install.md#mpi-source-build).

## Approximations to keep in mind

- **Quasistatic limit.** The MQS results assume a detector small compared with the signal wavelength; the full-wave
  solver measures the departure ([Full wave](fullwave.md)).
- **Uniform feed.** The axisymmetric full-wave solver drives the readout uniformly around the axis. A real slit or
  gap fed at one point also excites non-axisymmetric currents, which set the first self-resonance of a real device;
  check with a 3D full-wave solve well before it.
- **Iron at RF** uses the secant permeability at the static operating point, which is exact only for RF fields
  perpendicular to the static field.
- **Superconductors** are ideal (Meissner) in the static and pickup solves; the HTS critical-state analysis is an
  additional, opt-in solve with azimuthal screening currents only.
- **Sharp corners** of superconductors carry field singularities: the peak surface field there grows under
  refinement. Results report the regular peak that excludes the corners, and warn.
- **Surface losses** are perturbative in the surface resistance.
- **Scaling to the requirement** (field amplitude and size) is exact only for linear designs without fixed fluxes,
  circuits or backgrounds; the results say so when it is not.
- **Cost metric.** A scaling metric for comparing designs, not a construction estimate.
- **Material data.** Built-in mechanical and thermal properties are documented defaults; give measured data before
  quoting engineering numbers.

## Limits of the methods

- **Shape sensitivities** by the adjoint method cover axisymmetric MQS designs with linear materials. Nonlinear iron
  uses finite differences with remeshing (`method="fd_remesh"`); 3D and full-wave results are not differentiated.
- **Conversion to 3D** (`axionfem.convert.to_3d`) cannot revolve zero-thickness curves (model sheets as thin
  regions); the torsional analysis is axisymmetric only.
- **Cavity coupling** covers an axisymmetric pillbox with an on-axis coaxial probe and TM0np modes; off-axis probes and
  loops are not modelled.
- **GPU builds** move only hypre's solvers in the 3D engine to the GPU, on one MPI rank
  ([build options](../install.md#gpu-builds)).
- **Fluids** take meshes given as arrays: there is no design-file, mesher, command-line, viewer or MCP integration.
  Some solvers run on one MPI rank only (inductionless MHD, scalar transport and Boussinesq convection, point
  evaluation of transient flows); transient Boussinesq convection and time-dependent MHD are not offered. Turbulence
  models, open-boundary backflow stabilization, conjugate heat transfer, porous media, non-Newtonian fluids and
  fluid-structure interaction are not implemented ([fluids guide](fluids.md)).
- **Chemistry** has no species transport, Nernst-Planck transport, transient double layer, battery models or flow
  coupling; the porous electrode takes linear kinetics only ([chemistry guide](chemistry.md)).
- **Platforms.** Linux and macOS; Windows is not supported. The source builds are tested on Linux; on
  macOS they have not been tested yet ([install](install.md#source-builds)).

## Out of scope

Quench simulation, procurement-grade cost estimates and Windows support are outside the scope of this release.
