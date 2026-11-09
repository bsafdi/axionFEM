# Design notes

This page records why axionFEM is built the way it is: the formulations, conventions, solver choices and
dependency rules that the code relies on. Each note states the choice and its reason. The physics pages give the
equations, and the validation pages the evidence.

## Architecture

- **A C++ core on MFEM, Python as the interface.** The physics layer (formulations, constraints, post-processing)
  is axionFEM's own, on top of MFEM's finite-element spaces and assembly. Python builds designs, runs solves and
  inspects results; the readout networks, haloscope sensitivity, SQUID chains and COMSOL bridges are pure Python,
  because they are small dense problems or file formats.
- **One viewer.** A single three.js viewer, shipped with the package, serves the Jupyter widget, single-file HTML
  export and the live viewer server; PNG snapshots come from a software rasterizer in C++, so images need no
  browser or display.
- **Long MCP jobs run in a worker subprocess**, not a thread: a job is stopped by signalling its process group (the
  solver also starts `axf-mesher`), and a crash in the solver cannot take the server down.
- **The documentation builds without the compiled core.** The API reference is generated from docstrings by static
  analysis, so the site builds without compiling axionFEM.

## Dependencies and licensing

- Only permissively licensed libraries are linked into `libaxionfem`, `_core` and `axf`: MFEM (BSD-3), hypre
  (Apache-2.0 or MIT), METIS (Apache-2.0), Eigen (MPL-2.0, unmodified headers), nlohmann/json (MIT), nanobind (BSD-3),
  the optional STRUMPACK (BSD) and `stb_image_write` (public domain or MIT). three.js (MIT) is vendored unmodified
  and checksummed.
- gmsh is GPL. It runs as a separate program, `axf-mesher`, that exchanges files with the library, so the library
  and the Python module stay BSD-licensed. The wheels do not contain gmsh: the `mesh` extra installs gmsh's own
  wheel, which the mesher loads at run time.
- CHOLMOD's supernodal solver (GPL) and PARDISO are not used. The fluids module links nothing beyond the libraries
  above (MFEM, hypre, Eigen and the optional STRUMPACK); PETSc is not linked. The chemistry module links no SUNDIALS,
  PETSc or Cantera: they appear only as optional, out-of-band references in cross-checks.

## Builds

- **Wheels are sequential.** A wheel on PyPI cannot require a system MPI, so the wheels use a serial MFEM without
  hypre. MPI and hypre are used only by the 3D engine, the fluids module and the 3D current distribution of the
  chemistry module; a serial build compiles a stub whose 3D entry points raise an error that names the configuration
  and the install page. `AXF_MPI=AUTO` follows MFEM's own configuration.
- **OpenMP is optional**, so macOS builds work without an OpenMP runtime; results with and without it agree to
  round-off.
- **GPU support is a build capability; `AXF_DEVICE` is the run-time switch.** A CUDA build defaults to the CPU, so
  importing the package on a machine without a GPU cannot fail. Only the 3D engine uses a GPU, and only hypre runs
  there: MFEM stays on the host (its full assembly has no GPU path), which keeps every host-side access to MFEM
  vectors valid by construction. Host code that reads the structure of a hypre matrix finds the diagonal by its
  column, and eliminations never rely on stored zero rows, because hypre's GPU assembly drops them.

## Conventions

- **SI internally.** Every result also reports the natural Heaviside–Lorentz values (T and m) used in the
  axion-detector literature, with explicit conversion factors and unit-carrying keys.
- **Time convention e^{-iωt}** throughout the engines, the circuits and the multiports (Z = R − iωL). The Touchstone
  export conjugates element-wise into e^{+jωt}, the convention of network analysers, SPICE and scikit-rf; the
  conversion functions take a required `convention` argument, so a silent default cannot conjugate twice or never.
- **Signal coupling by reciprocity.** Φ_a = κ_a Ψ with Ψ = (1/I)∫A_I·B_lab d³x. The engine evaluates whichever form
  the available potentials allow (A_pick·B_lab for azimuthal pickups, A_lab·B_I for port pickups); the two are equal
  on all of space.
- **Every result carries diagnostics and at least one independent consistency check** (L_p from flux and from
  energy, Newton's third law over all bodies, Gauss's law over conductors, symmetry of matrices assembled without
  symmetrization). Solver failures, refused approximations and singular corner fields are warnings in the result,
  never silent.
- **Errors say what, where and how.** A message names the problem, the JSON path in the design and a hint, because
  agents read and act on them. Invalid input is refused rather than projected silently.
- **Expressions use standard precedence.** In design expressions `^` binds tighter than unary minus and is
  right-associative, as Python's `**`, so `exp(-z^2/(2*s^2))` is a Gaussian. Number literals are strict (malformed
  and subnormal literals are errors), `min`/`max` reject NaN, and nesting is capped. The COMSOL exporter
  parenthesizes every operation, so COMSOL's own precedence rules do not matter.
- **Published benchmarks use public information only, with provenance.** Every dimension of a reproduced detector
  is recorded in `benchmarks/published/sources.json` with its source and marked "chosen" where the source does not
  fix it. Coupling conversions are derived identities, not fits. No reach curve is claimed where a needed input
  (a noise spectrum, a live time) is unpublished.

## Axisymmetric magnetoquasistatics

- **Two scalar problems.** Poloidal fields use ψ = rA_φ and toroidal fields G = rH_φ (the enclosed poloidal
  current over 2π). Both are scalar H1 problems on one engine: element weights ν/r, "master" degrees of freedom
  for floating bodies (superconductors for ψ, vacuum regions for G), and global constraints. The open boundary is an
  exact spherical-harmonic Dirichlet-to-Neumann map on a semicircle. The result is an exact open boundary, an exact
  treatment of superconductor topology and a symmetric positive definite system.
- **Sharp superconductor corners.** The surface field at a convex corner is singular. Results report the peak, its
  location, the detected corners and a "regular" peak that excludes three element sizes around them, and warn when
  the peak sits at a corner; cost caps use the regular peak.
- **Iron.** Static fields use the B–H law with Newton's method and an energy (Armijo) line search; a step in the
  round-off regime is accepted only if it reduces the residual. In a static toroidal field H = G/r is fixed by the
  linked currents, so B follows pointwise. RF problems use the secant permeability B/H of the static field, which is
  exact for RF fields perpendicular to the static field; parallel components would need the differential tensor.
- **Background fields on grids** are interpolated by bicubic Hermite polynomials in (r², z) with nodal derivatives
  from the field, so the interpolated field is continuous, divergence-free and third-order accurate.
- **Thin magnetic layers** (shields not meshed through their thickness) make A_φ two-valued across the layer; the
  jump is the excess flux the layer carries. A continuous-potential surface term would be a first-order perturbation
  that fails in the shielding regime μ_r d ≳ R, which is the regime the feature exists for. The normal-flux
  correction is taken per half strip, and meshes that would make the system indefinite are refused.
- **Forces.** The winding self-force is the force of the same mesh with that winding alone in vacuum (exact for linear
  problems by superposition); it is reported next to the raw integral and serves as its error bar. Distributed loads
  use the Maxwell traction on material bodies and the Lorentz density inside windings, where a surface integral
  would be a small difference of large tractions.
- **HTS critical state** uses a time-dependent A-formulation with a regularized E–J power law; screening currents
  are azimuthal, so the static field must be poloidal. The static and pickup solves keep the ideal Meissner model,
  so the analysis is opt-in and additive.
- **Shape sensitivities** use a discrete adjoint on a morphed mesh: exact for the discrete model and free of
  remeshing noise. Derivatives are total: parameters written as expressions of p move with p. Nonlinear iron and
  parameters that switch regions use finite differences with remeshing.
- **Linear algebra.** Eigen's simplicial LDLᵀ (AMD ordering, METIS for large systems) on the constraint-reduced
  system; the dense exterior map stays out of the factorization and enters by a Woodbury update. STRUMPACK, chosen
  for its BSD licence and its OpenMP multifrontal factorization without MPI, takes large systems when it is built
  in. Calls into STRUMPACK go through a guard, because the library keeps process-wide state that concurrent callers
  must coordinate; on versions the guard was not validated for, it serializes the calls and warns.

## 3D engine

- **MPI layout.** Every rank holds the serial mesh; METIS partitions it. Topology (bodies, cohomology and geometric
  cuts, ports, winding loops) is computed on the serial mesh identically on every rank and projected onto the
  partitioned spaces, so the code stays serial-simple and results do not depend on the rank count. Memory per rank
  grows with the mesh.
- **Open boundary by an infinity shell.** A spherical shell is mapped onto the exterior out to infinity, with the
  mapped vacuum reluctivity as a tensor. A truncated sphere biased Ψ and W of a solenoid by about −0.44 % independently
  of the mesh; the map removes that bias at the cost of more AMS iterations.
- **Gauge and divergence cleaning.** The curl-curl operator carries a gauge mass scaled per element
  (ε_e = κ ν_e/h_e²), so its relative size is κ everywhere, independent of the mesh, the device and the shell
  radius; κ is calibrated between an accuracy bound and a round-off floor. Divergence cleaning removes the
  right-hand side's components along the discrete harmonic gradients of eliminated conductors, besides the usual
  gradients.
- **Every Krylov solve checks its true residual**, and a solve that reaches its iteration cap is reported: a
  breakdown that returns x = 0 after one iteration must not pass.
- **Curved elements.** Second-order geometry can invert next to thin curved features: gmsh's elastic optimizer runs
  first, its HighOrder optimizer when elements stay inverted, and any element still invalid is straightened locally.
- **Pickups.** The 3D azimuthal pickup keeps the sheath closed and prescribes the readout current around the axis,
  the exact 3D counterpart of the axisymmetric ideal slit; real slits fed by a port are available for studying the
  slit itself.
- **Adaptivity by remeshing**, not by nonconforming refinement: the geometry stays exact, the topology code needs no
  hanging nodes, and 2D and 3D share one path.
- **3D scenes** show a quarter cutaway with exact mesh slices on the two cut half-planes, without resampling.

## Full wave

- **Axisymmetric first, both m = 0 families, exact exterior.** The readout is driven with I = 1 A; Ψ(ω) and
  Z = R − iωL(ω) both reduce to the magnetoquasistatic Ψ and L_p as ω → 0, so MQS results are a limit of the same
  outputs, not a separate model. The axisymmetric model prescribes a uniform azimuthal readout current: it is the
  uniform-feed limit, valid while the readout circumference is small compared with the wavelength. Slit and port
  self-resonances need the 3D solver.
- **TM0 uses H_φ split as H_MQS + H′.** Nédélec elements for (E_r, E_z) with the r weight produce spurious modes
  localized at the axis and were rejected. The scalar H_φ form is spectrally exact, and splitting off the MQS pickup
  field leaves a smooth O(ω²) correction, so accuracy is uniform down to the MQS limit. TE0 uses A_φ on the MQS
  space.
- **Sealed cavities keep zero RF flux**, imposed as one linear constraint per cavity, so the static cavity mode
  cannot resonate with a sweep.
- **The exterior is the exact outgoing Dirichlet-to-Neumann map** (spherical Hankel ratios). The reciprocity integral
  for Ψ(ω) extends to infinity; its exterior part is added exactly, since without it Ψ depends on the domain radius.
- **Complex symmetric LDLᵀ** without conjugation, with the symbolic analysis reused across a sweep, iterative
  refinement and a residual check that falls back to pivoting LU. Eigenmodes come from shift-invert Lanczos.
- **3D full wave** solves for a complex Nédélec field driven by the MQS port current distribution. The radiating
  exterior combines the infinity shell, capped by a perfect conductor, with an exact low-rank correction in vector
  spherical harmonics, which a Woodbury update carries into the block AMS preconditioner.
- **The error indicator is on by default**: every other output is identical with it on or off, and it typically
  costs a few percent of a frequency's solve. It is an indicator, not an estimator, and estimates that are not
  reliable are withheld.
- **The Gauss-law guard estimates and corrects by default** on the ungauged paths: the correction removes the gradient
  part of the error, which the low-frequency breakdown amplifies, and leaves the curl part. The tree-cotree gauge
  stays opt-in because it covers a narrow case (lowest-order elements, perfectly conducting boundaries, a real
  lossless operator).
- **Surface losses are perturbative, per port**, from the perfectly conducting structure's fields: one solve serves
  every R_s(f), material and temperature, and the MQS limit is exact. The tuned Q, ωL_eff/R_loss with
  L_eff = ½(X/ω + dX/dω), is the headline; Q_total is withheld where the solve's residual could move it. The
  surface-resistance models that supply R_s(f) are beta.
- **The adaptive sweep** fits a rational model (set-valued AAA) in a normalized frequency to Z and Ψ, uses dZ/dω as
  Hermite data only where the derivative is exact, and solves where the model is least certain. Self-resonances are
  classified poles of a converged sweep.
- **Passivity and reciprocity are reported, never repaired**: nothing is clipped, symmetrized or projected.

## Readout, scans and electrostatics

- **Readout circuits use full modified nodal analysis** (every element a current unknown), so mutual inductance,
  ideal transformers, sources and shorts share one stamp and the matrix is symmetric for every reciprocal network,
  a free run-time check. Noise is referred to a series EMF in the pickup branch, so every downstream statistic applies
  unchanged.
- **The circuit hands the scan (Ψ, L_eff, Q, T_eff, η_A).** L_eff is the reactance-slope inductance; passing it
  instead of L_p reproduces the (L_PU/L)² degradation of Chaudhuri et al. (arXiv:1803.01627) without changing the
  scan code. T_eff and η_A are defined so that the noise at resonance is reproduced exactly; `scan.finite_band` plans
  with the exact network noise where the single pole is not enough.
- **Circuit drawings are generated** from the circuit objects and checked by rebuilding the netlist from the drawing,
  so figures and API cannot drift apart.
- **Electrostatics** uses the H1 space of the MQS problems, conductors as master degrees of freedom with a prescribed
  potential or charge, and the exact Laplace exterior map including the monopole term, which lets net charge escape
  to infinity. The capacitance matrix is built from the reactions rather than symmetrized, so its asymmetry is a
  diagnostic. Lumped readout-gap estimates are reported as estimates; the self-resonance reference is the pole of
  L(ω) in a full-wave sweep.

## Mechanics and heat conduction

- **Displacement formulation** on the magnetostatic mesh's H1 space, without reduced integration, so ν = 0.5 is
  refused. Touching regions share nodes and are bonded; contact, sliding and gaps are out of scope.
- **Material data define the domains**: a region takes part when its material has a `mechanical` (`thermal`) block,
  which puts windings in the structure and keeps bores and gaps out; missing data produce a warning naming the regions.
- **Rigid axial modes are pinned, not penalized.** Each unsupported assembly keeps one axial translation, removed by
  pinning one degree of freedom and shifting to zero mean displacement. Force equilibrium is an exact identity of the
  discrete system and is reported as a round-off diagnostic.
- **k(T) by Picard iteration**, because the Kirchhoff transform is exact only within one material. A two-point table
  reproduces a linear law exactly, so the Kirchhoff solution is an independent reference for the nonlinear path.

## Meshing

- **gmsh runs in a separate program**, for licensing and for robustness: a crash in meshing does not take down the
  solver or the Python process.
- **Meshes are reproducible.** 3D meshing is single-threaded, because multi-threaded meshing was not reproducible
  and spoiled finite differences. The curved-mesh step runs in a process started with fixed inputs, because gmsh's
  second-order elements depended on allocator state (environment size, path lengths, address randomization). This is
  verified on Linux with glibc; reproducibility on other platforms is not guaranteed.

## Cavity haloscopes and SRF cavities (beta)

- **The cavity** is the connected non-conducting region containing a given interior point; an open cavity is an
  error. Losses are perturbative on the lossless mode; TE wall currents come from a consistent-flux recovery, which
  converges like the eigenvalue, with the pointwise value kept as an independent check.
- **Q0 is never silently incomplete**: lossless walls are warned unless explicitly allowed, with one rule for the
  axisymmetric, cross-section and 3D solvers, so the same design reports the same quantities in each.
- **Cross-section cavities** are a separate symmetry (`"xsec"`), not a cavity option, because they change the
  geometric space of every shape and the engine.
- **Surface resistance**: the anomalous skin effect is evaluated from the exact Reuter–Sondheimer integrals, not a
  fit; magnetoresistance is refused in combination with anomalous, BCS or flux-flow models rather than combined
  approximately.
- **SRF wall peaks** are the maximum along the meridian of the m = 0 wall field on the exact wall contour. In 3D the
  peak comes from a least-squares fit near the maximum, since single-point maxima measure the azimuthal scatter of
  the discrete solution.

## COMSOL bridges

- **Export** writes COMSOL's own `.mphtxt` mesh with named selections, so a cross-check needs no add-on module; the
  Java model is a convenience for editing geometry, and STEP is written by `axf-mesher` (gmsh stays isolated).
- **Import** (beta) solves on COMSOL's stored mesh by default (the only way to reproduce a COMSOL result to discretization
  accuracy without COMSOL's CAD kernel), fits curved boundaries to recover second-order geometry, and reports how
  every COMSOL feature was mapped. A replay route rebuilds the geometry for editing. Neither needs a COMSOL
  installation.

## Fluids (beta)

- **A separate library** on top of `libaxionfem`, built only with MPI, with Taylor–Hood pairs (P2/P1, Q2/Q1) for
  inf-sup stability.
- **A general sparse LU.** The symmetric solver of the EM engines is not valid for Oseen and Newton Jacobians, so the
  fluids module has its own LU interface over Eigen and STRUMPACK. Matching keeps saddle-point pivots nonzero but does
  not make the factorization accurate, so every solve runs residual-checked iterative refinement and throws, on every
  rank, when the target is missed.
- **Its own nonlinear driver** (a Stokes start, Picard steps, Newton with an Armijo line search on a discrete
  dual-norm merit, continuation and pseudo-transient continuation) instead of MFEM's Newton solver, which measures
  and stops on its own residual norm. The same driver serves Navier–Stokes and Boussinesq convection.
- **Transient Navier–Stokes** uses its own variable-step BDF1/BDF2 with error control: the velocity–pressure system
  is an index-2 DAE that general ODE/DAE libraries do not serve directly.

## Chemistry (beta)

- **Own stiff integrators** (variable-order BDF, an L-stable ESDIRK pair) with step control, dense output and event
  location. MFEM's time integrators have no BDF, error estimator or DAE interface, and one-step methods keep restarts
  after discontinuities simple.
- **Chemical equilibrium is an index-1 DAE** in reaction-invariant form, with consistent initialization by Newton.

## Looking ahead

Directions that the code does not cover yet: adjoint sensitivities for nonlinear iron, 3D designs and full-wave
quantities; the HTS critical state in 3D and with a background field driving magnetic regions; for the fluids module,
design-document, mesher, viewer and MCP integration, multi-rank scalar transport, Boussinesq convection and MHD,
transient Boussinesq and MHD, and preconditioned iterative solvers for Navier–Stokes; for chemistry, species
transport and coupling to flow; cavity coupling through off-axis probes and loops. Quench simulation,
procurement-grade cost estimates and native Windows support are out of scope.
