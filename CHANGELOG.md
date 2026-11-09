# Changelog

All notable changes to axionFEM are recorded here. Versions follow [semantic versioning](https://semver.org/);
while the version is 0.x, minor versions may change the API. Modules marked **beta** are usable but less validated,
and their interfaces may change.

## Unreleased

## 0.1.0: initial public release

The first public release. It contains the following capabilities.

### Lumped-element detectors

- Axisymmetric magnetoquasistatic engine: the poloidal (ψ = rA_φ) and toroidal (G = rH_φ) problems on high-order
  H1 elements with an exact open boundary (Dirichlet-to-Neumann map); nonlinear iron (B–H curves); ideal
  superconductors with fluxoid or net-current constraints; stranded windings, persistent circuits and imposed
  background fields; thin magnetic layers (shields that are not meshed through their thickness).
- Parametric templates `toroid`, `solenoid` and `core`. Designs are JSON documents with parameter expressions,
  checked against a JSON Schema; errors name the JSON path and give a hint.
- Results: the signal coupling Ψ, the pickup inductance L_p, the scan figure of merit 𝓕 = Ψ⁴/L_p², the stored
  energy, peak fields and superconductor surface fields (with sharp-corner detection), forces (net and distributed,
  winding self-force, a Newton's-third-law closure check), and for each noise benchmark 𝓕/𝓕_req, the amplitude
  scale needed, a cost metric and the cost-optimal size. Every result carries diagnostics and consistency checks.
- Conductor critical surfaces (NbTi, Nb3Sn, REBCO) with critical-current and temperature margins; the HTS critical
  state (E–J power law) for azimuthal screening currents.
- Electrostatics: capacitance matrices and readout-gap capacitance, axisymmetric and 3D.
- 3D H(curl) magnetoquasistatic engine with an infinity-shell open boundary (MPI builds).
- Full wave beyond magnetoquasistatics, axisymmetric (both m = 0 families, exact outgoing boundary) and 3D (MPI
  builds): readout impedance Z(ω), Ψ(ω), radiation Q, eigenmodes and self-resonances; S/Z/Y parameters and
  Touchstone v1 export with passivity and reciprocity checks; 3D current ports and multiport impedance matrices.
  **Beta:** the surface-resistance models of RF walls behind the perturbative surface losses, and in 3D the
  prescribed static source, the imposed background field, perfectly matched layers, the error indicator, the Gauss-law guard on the ungauged paths,
  the adaptive rational-fit frequency sweep, the opt-in tree-cotree gauge, the eigenmodes, closed RF-only models
  (symmetry surfaces, removed PEC volumes, lumped gaps, direct factorization) and the energy-matrix, field-export and
  reactive-load diagnostics.
- Readout circuits (modified nodal analysis with the noise of every element), multiport networks, finite-band scans
  with the exact network noise, reach curves, and multi-band and multipole scans implementing the methods of
  arXiv:2211.00008 and arXiv:2302.14084 (`axionfem.multipole`, **beta**).
- Shape sensitivities by a discrete adjoint on a morphed mesh (axisymmetric designs with linear materials);
  gradient-based (L-BFGS-B, SLSQP), global (differential evolution) and derivative-free (bounded Powell)
  optimization.
- Mechanics (axisymmetric linear elasticity with bonded contact, cooldown loads) and steady heat conduction with a
  temperature-dependent conductivity; 3D versions in MPI builds.
- Torsional dynamics of axisymmetric structures and an analytic torsional-waveguide mobility (**beta**).

### Other detector types (beta)

- Cavity haloscopes: axisymmetric, cross-section and 3D eigenmodes, form and geometry factors, Q0 from
  surface-resistance models, tuning sweeps with mode tracking; haloscope sensitivity (scan rate, optimal coupling,
  exclusion) and the parameters of published haloscope runs.
- Coupling of a cavity mode to a coaxial probe on the axis of a pillbox (`axionfem.cavity_coupling`).
- SRF heterodyne cavities: elliptical-cell geometry, degenerate pump and signal mode pairs, their figures of merit,
  and the sizing of a scan.
- dc SQUID readout chains: added noise and the reach of a detector read out by them.

### Fluids and chemistry (beta)

- `axionfem.fluids` (MPI builds only): Taylor–Hood Stokes (planar, axisymmetric with swirl, distributed 3D), steady
  Navier–Stokes, stabilized P1/P1 (planar), inductionless MHD (2.5D, one MPI rank), transient Navier–Stokes
  (BDF1/BDF2), and scalar transport with Boussinesq convection (`axionfem.transport`, one MPI rank). Meshes are
  given as arrays; there is no design-document, viewer or MCP integration.
- `axionfem.chem`: stiff chemical kinetics and 0D reactors with axionFEM's own time integrators, chemical
  equilibrium as an index-1 DAE; primary and secondary current distributions (planar and axisymmetric; 3D in MPI
  builds); a 1D porous electrode with linearized kinetics.
- Importing `axionfem.fluids`, `axionfem.chem` or `axionfem.transport` emits a one-time `axionfem.BetaWarning`.
  User guides: docs/guides/fluids.md and docs/guides/chemistry.md.
- Verification of both modules as case definitions with acceptance criteria and lean reference numbers, run by
  `benchmarks/fluids/run.py` and `benchmarks/chem/run.py` (list, show, run, report); the optional cross-code
  reference generators (NGSolve, FEniCSx) are in `tools/fluids_mms`. The measured results, including what is not
  verified, are on docs/validation/fluids.md and docs/validation/chemistry.md.

### Interfaces

- Python API, the `axf` command line, a Jupyter widget, single-file HTML export, a live viewer server, VTU export
  for ParaView and a deterministic PNG renderer.
- An MCP server with 36 tools for AI assistants, an agent skill for detector design and a project MCP configuration.
- COMSOL export (mesh, Java model, STEP, physics settings) and import of `.mph` models (**beta**), both without a
  COMSOL installation.

### Validation

- Tests against closed-form solutions, manufactured solutions and independent codes, and comparisons with published
  detectors and experiments, each quantity with a verdict (agrees, disagrees or not decidable): ABRACADABRA-10cm,
  DMRadio-50L, DMRadio-m3, DMRadio-GUT, DMRadio-Core, HAYSTAC, RADES, CAPP, QUAX and the COMSOL models of
  arXiv:2211.00008 (shielded scenarios). The documentation's validation pages describe the checks and their
  tolerances.

### Builds and packaging

- Sequential builds (`scripts/build_deps_wheel.sh`, then `AXIONFEM_DEPS=$PWD/third_party/install-seq pip install .` in a clone; wheels configured for Linux x86_64 and macOS x86_64 and arm64,
  CPython 3.10–3.13, not yet uploaded to PyPI): everything except the 3D engine and the fluids module. Source builds with MPI add both; OpenMP, STRUMPACK and CUDA (hypre in the 3D
  engine) are optional.
- Pinned dependencies: MFEM 4.10, hypre 2.33, METIS 5.1, Eigen 3.4, nlohmann/json 3.12 and the gmsh 4.15 SDK. gmsh
  is used only as a separate program and is not redistributed in the wheels.
