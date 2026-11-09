# Chemistry (beta): formulation and numerics

This page collects the details of the Python interface, the formulation, the numerical methods, the MPI behaviour and the verification of `axionfem.chem` (beta). The [chemistry guide](../guides/chemistry.md) shows how to use it.

## Using the API

- **Conventions**: SI units throughout (mechanism inputs may give other concentration, time and energy units, which
  the kernel converts once); anodic currents are positive; the overpotential is `eta = phi_s - phi - E_eq`;
  axisymmetric meshes live in the meridian `(r, z)` and every integral carries `r` ("per radian"), while galvanostatic
  currents are 3D totals.
- **Mechanisms** are dictionaries (`{"species": [...], "reactions": [...], "units": {...}}`, described in the
  docstring of `axionfem.chem.kinetics`); a reactor builder returns a problem that every integrator accepts, and a
  trajectory has `t`, `y`, `dense(t)`, `events` and `stats`. `kinetics.DECLARATIONS` states the order, error estimate,
  starter and dense output of each integrator.
- **Electrodes**: `add_electrode(name, attributes, reactions=[...], phi_s=...)` for a kinetic electrode
  (potentiostatic, or galvanostatic with `galvanostatic=True, current=I`), `primary=True, E_eq=...` for a primary one;
  boundary attributes without an electrode are insulating. Reactions: `Reaction("bv" | "linear" | "tafel_a" |
  "tafel_c", i0, E_eq, alpha_a, alpha_c, A)`.
- **Convergence**: `solve()` raises `chem.ConvergenceError` when Newton does not converge, with a message that says
  what stalled (`solve(check=False)` returns the report instead). Large cell voltages need continuation: start from
  `initial="primary"` (2D) or step the applied potential or current with `solve_continuation`.
- **Results**: the field and its derivatives through the solver object (`state()`, `residual(...)`, `jacobian(...)`,
  `boundary_quadrature(...)`, `domain_quadrature(...)`); currents are integrals of the reaction current densities
  over an electrode's boundary quadrature, as in the example of the
  [chemistry guide](../guides/chemistry.md).
- **Beta warning**: the first import of `axionfem.chem` (or of `axionfem.chem.kinetics`) in a process emits one
  `axionfem.BetaWarning`. Silence it with `warnings.simplefilter("ignore", axionfem.BetaWarning)`.

## MPI

`CurrentDistribution3D` needs the MPI build and runs on several ranks (`mpirun -n 4 python script.py`, or `srun` in a
SLURM job): every rank builds the same `meshes3d.TetMesh`, the solver partitions it with METIS, and every array it
returns is the rank's local part (local elements, local boundary elements, owned degrees of freedom). The kinetics,
the 2D current distribution and the porous electrode are serial and run in every build, the sequential one included.

## Verification

The chemistry verification cases (closed-form and manufactured solutions, conservation, stiff test problems, and
comparisons with Cantera and DOLFINx), with their acceptance criteria and the results of the last run, are listed on the
[chemistry validation page](../validation/chemistry.md). They run with `python benchmarks/chem/run.py` from a source
checkout (`list`, `run <case>`, `run --all --cost short`); see `benchmarks/chem/README.md`. The unit tests are
`tests/test_chem_*.py` and `cpp/src/chem/tests/`.

## Formulation and numerics

### Time integration (`axf::time`)

Problems `M y' = f(t, y) + s(t)` with a constant mass matrix `M` that may be singular (index-1 differential-algebraic
systems in reaction-invariant form) and an analytic Jacobian of `f`. Every method is implicit, with full Newton on the
analytic Jacobian and dense LU (the systems are 0D and small):

- **BDF-k, k = 1..5**, in variable-coefficient form on the actual past times, with the local error estimate from the
  difference between the corrector and the degree-k predictor;
- **one-step ESDIRK** methods given by a Butcher tableau with an embedded pair: ESDIRK3(2)4L[2]SA (Kennedy and
  Carpenter 2003; order 3, stage order 2, embedded order 2, L-stable, stiffly accurate) and the trapezoid rule (theta =
  1/2) as a non-L-stable control;
- drivers on prescribed grids, adaptive drivers with PI step control and stop-time outputs (every output time is an
  accepted step end), a **variable-order BDF1..5** driver with order selection, dense output (the BDF interpolant or a
  cubic Hermite polynomial) and event location by Brent's method on the dense output.

Consistent initial derivatives of a DAE are computed from the differentiated algebraic rows.

### Kinetics kernel

Mass-action rates of progress `q_j = k_f,j(T) prod c_i^{o_ij} - k_r,j(T) prod c_i^{nu''_ij}` with production
`omega_i = sum_j nu_ij q_j`; modified Arrhenius `k = A T^n exp(-Ea / (R T))`; reverse rates from an equilibrium constant
`K_c` or their own Arrhenius parameters; explicit (also non-integer) forward orders on irreversible reactions. Units of
the input (mol/m^3, mol/L, mol/cm^3; s, min; J, cal, kcal, kJ per mol) are converted to SI once. Cycles of reversible
reactions must satisfy the Wegscheider condition (refused otherwise unless explicitly allowed). The kernel provides the
analytic Jacobians `d omega / d c` and `d omega / d T`.

### Reactors and equilibrium

Each builder returns a time-integration problem with its Jacobian: batch, CSTR, semibatch (with the volume as an
unknown), isothermal constant-pressure ideal gas, adiabatic liquid (`rho c_p T' = sum (-dH_j) q_j`) and the isothermal
isobaric plug-flow reactor. **Equilibrium reactions** are imposed as algebraic constraints
`prod c^{nu''} - K prod c^{nu'} = 0` together with the reaction-invariant balances `W c' = W (S q + s)` (`W` spans the
left null space of the equilibrium stoichiometry), an index-1 DAE; consistent initialization keeps `W c` and solves the
constraints by Newton.

### Current distribution

The electrolyte potential `phi` solves `-div(kappa grad phi) = f` with, on kinetic electrodes,
`kappa grad phi . n = sum_j i_j(eta_j)` (overpotential `eta_j = phi_s - phi - E_eq,j`, anodic currents positive), on
primary electrodes `phi = phi_s - E_eq` and on insulators a prescribed (usually zero) flux.

- **Kinetics**: Butler-Volmer, linearized Butler-Volmer, anodic and cathodic Tafel, several reactions per electrode;
  exponent arguments are clipped (and the clipping is reported).
- **Control**: potentiostatic (`phi_s` given) or galvanostatic (`phi_s` an unknown with the bordered row
  `int sum_j i_j dA = I`); a galvanostatic primary electrode is a floating conductor.
- **Discretization**: Lagrange P1/P2 triangles (P2 optionally isoparametric), planar or axisymmetric (every integral
  weighted by `r`), optional half-space even-mode Dirichlet-to-Neumann map on an outer arc; 3D on tetrahedra (P1/P2)
  distributed with METIS over MPI ranks.
- **Solver**: Newton iterated to stagnation of the relative residual, with a limiter on the overpotential update, a
  halving line search, continuation in the applied potential or current (`solve_continuation`) and a warm start from a
  previous state. The Jacobian is symmetric positive definite for monotone kinetics on a grounded problem; the linear
  solver is a sparse LDL^T factorization in 2D and CG with BoomerAMG in 3D (bordered row by its Schur complement).
- **1D porous electrode** (`PorousElectrode`): solid and electrolyte potentials coupled by a volumetric exchange
  `j = a_v i(eta)`, galvanostatic or potentiostatic at the separator; **only linearized Butler-Volmer kinetics** is
  accepted (Butler-Volmer and Tafel are refused).

## Limitations

- Beta: the interface may change; there is no design-file (spec/CLI), viewer or MCP integration.
- The 3D current distribution (`CurrentDistribution3D`) needs the MPI build; the rest works in every build.
- Large applied potentials need continuation: Newton from `phi = 0` lowers the overpotential by about `RT/(alpha F)`
  per step on the exponential branch.
- The porous electrode accepts linearized Butler-Volmer exchange only (Butler-Volmer and Tafel exchange are refused).
- The time integrators are meant for small dense 0D systems (dense LU), not for spatially discretized PDEs.
- No mass transport: concentrations do not vary in space; the current distribution is secondary (kinetics) at most,
  not tertiary.
- The stiff test problems use chosen constants and are compared with tight-tolerance reference solutions, not with
  published values; the two 3D cases were measured only on their first two refinement levels (see the validation
  page).

## References

- J. A. V. Butler, Studies in heterogeneous equilibria. Part II. The kinetic interpretation of the Nernst theory of
  electromotive force, *Trans. Faraday Soc.* **19**, 729-733 (1924); T. Erdey-Grúz and M. Volmer, Zur Theorie der
  Wasserstoffüberspannung, *Z. Phys. Chem.* **150A**, 203-213 (1930). (Butler-Volmer kinetics.)
- D. G. Goodwin, H. K. Moffat, I. Schoegl, R. L. Speth, B. W. Weber et al., Cantera: An object-oriented software
  toolkit for chemical kinetics, thermodynamics, and transport processes, Version 3.2.0 (2025),
  doi:10.5281/zenodo.17620923, https://www.cantera.org.
- C. A. Kennedy and M. H. Carpenter, Additive Runge-Kutta schemes for convection-diffusion-reaction equations,
  *Appl. Numer. Math.* **44**, 139-181 (2003). (ESDIRK3(2)4L[2]SA.)
