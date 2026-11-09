# Fluids (beta): formulation and numerics

This page collects the details of the Python interface, the formulation, the numerical methods, the MPI behaviour and the verification of `axionfem.fluids` and `axionfem.transport` (beta). The [fluids guide](../guides/fluids.md) shows how to use them.

## Using the API

- **Meshes**: `fluids.Mesh.from_arrays(vertices, triangles or quadrilaterals, boundary segments, boundary attributes)`,
  optionally with isoparametric P2 geometry nodes; `Mesh.unit_square(n)` and `Mesh.annulus(...)` for tests;
  `fluids.Mesh3D.from_arrays` in 3D. Meshes are 2D in the (x, y) plane; axisymmetric problems use the meridional half
  plane `(r, z) = (x, y)`.
- **Boundary conditions**: a list of dicts, exactly one per boundary attribute (`{"attribute": 2, "kind":
  "do_nothing"}`); an attribute without a condition, or with two, is refused. The kinds each solver accepts are listed
  in its docstring.
- **Data**: viscosity, force and boundary values are numbers or Python callables `f(x, y)` (`f(x, y, t)` for the time
  stepper); with `vectorized=True` the callables receive arrays and are tabulated once.
- **Units**: `solve_stokes` and `solve_stokes_3d` are kinematic (everything divided by the density, `nu` the kinematic
  viscosity); Navier-Stokes, the time stepper, MHD and Boussinesq are dynamic (`mu`, `rho`, physical pressure).
- **Convergence**: no solver raises on non-convergence. Check `sol.diagnostics["converged"]` (and
  `["relative_residual"]`); the fields of a solve that did not converge are refused. An option that is not offered
  is refused (`fluids.FluidsError`, or `ValueError` for arguments checked in Python) with a message that says what to
  use instead.
- **Results**: point evaluation at reference points of given elements (`sol.velocity(elements, ref_points)`, `pressure`,
  `geometry`), Babuska-Miller forces and torques (`sol.reaction(attributes)`), boundary nodes, and for Navier-Stokes the
  free-DOF residual and Jacobian (`state`, `residual`, `jacobian_action`) for Taylor tests.
- **Build**: the module is compiled only in the MPI source build (see the [installation guide](../guides/install.md));
  the serial wheels and the sequential build do not contain it, and `import axionfem.fluids` then raises
  `ImportError`. `axionfem.build_info()["fluids"]` says whether a build has it. STRUMPACK is optional: without it
  the direct solves use Eigen's sparse LU.
- **Beta warning**: the first import of `axionfem.fluids` in a process emits one `axionfem.BetaWarning`; the first
  import of `axionfem.transport` emits two (its own and, because it imports `axionfem.fluids`, the fluids one).
  Silence them with `warnings.simplefilter("ignore", axionfem.BetaWarning)`.

## MPI

- 2D Stokes and Navier-Stokes solves partition the mesh over `MPI_COMM_WORLD` when the script runs on several ranks
  (`mpirun -n 4 python script.py`, or `srun` in a SLURM job), but point evaluation and `boundary_nodes()` (and
  `blocks()` of a Stokes solution) refuse more than one rank, because element numbers are then rank-local. In
  practice 2D problems run on one rank.
- 3D Stokes is the distributed path: `solve_stokes_3d` is collective over `MPI_COMM_WORLD`; every rank builds the same
  serial mesh, the solver partitions it with METIS, and each rank evaluates on its own elements, addressed by input
  element id (`sol.element_ids()`).
- Scalar transport, Boussinesq and MHD run on one rank and refuse a partitioned mesh. GLS works on vertex and
  triangle arrays and solves serially (on several ranks every rank would solve the same problem). The time stepper is
  tested on one rank.
- The verification cases that need several ranks start them through the launcher of `axionfem`
  (`$AXF_MPI_LAUNCHER`, else `srun` inside a SLURM job, else `mpirun`/`mpiexec`).

## Verification

The verification cases (manufactured solutions, closed-form solutions, discrete identities and comparisons with
NGSolve and DOLFINx on the same discrete problem), with their acceptance criteria and the results of the last run,
are listed on the [fluids validation page](../validation/fluids.md). They run with `python benchmarks/fluids/run.py`
from a source checkout (`list`, `show <case>`, `run <case>`, `run --cost short`); the case format is described in
`benchmarks/fluids/README.md`, and the optional generators of the reference data (SymPy, NGSolve, DOLFINx) are in
`tools/fluids_mms/`. The unit tests are `tests/test_fluids_*.py` and `cpp/tests/test_fluids_*.cpp`.

## Formulation and numerics

### Equations

Incompressible flow with a viscous form chosen per solve:

- **Stokes** (kinematic units, everything divided by the density):
  `-div sigma_F(u, p) = f`, `div u = 0`.
- **Steady Navier-Stokes** (dynamic units: dynamic viscosity `mu`, constant density `rho`, physical pressure):
  `rho (u . grad) u - div sigma_F(u, p) = f`, `div u = 0`, with the convective term in the standard form or in Temam's
  skew-symmetric form (`+ rho/2 (div u) u`).
- **Transient Navier-Stokes**: `rho (du/dt + (u . grad) u) - div sigma_F(u, p) = f`, `div u = 0`.

Two viscous forms: the **stress form** `sigma_S = 2 mu eps(u) - p I` (`a(u, v) = int 2 mu eps(u) : eps(v)`, the Cauchy
stress; required for a variable viscosity and for physical shear-stress conditions) and the **Laplacian form**
`sigma_L = mu grad u - p I` (`a(u, v) = int mu grad u : grad v`; its natural condition is the pseudo-traction
`mu du/dn - p n`, so on curved slip or traction walls the solved problem is not the physical one; use the stress form
there). An optional grad-div term `gamma (div u, div v)` (gamma in viscosity units) is available for Stokes.

**Geometries.** Planar 2D; axisymmetric meridional half-plane `(r, z) = (x, y)` with swirl (unknowns `(u_r, u_z,
u_theta)`, every integral weighted by `r`, continuity tested with `r q`, hoop terms `u_r / r`, `u_theta / r`,
centrifugal and Coriolis terms in the convection); 3D (Stokes only, on a partitioned `ParMesh`); planar 2.5D
(z-invariant, three velocity components; offered only with the MHD coupling).

### Discretization

- **Taylor-Hood** P2/P1 (Q2/Q1) elements for velocity and pressure, with isoparametric P2/Q2 geometry on curved
  boundaries. Meshes whose triangles (tetrahedra) violate the Taylor-Hood mesh condition (all vertices on the boundary;
  no interior vertex) are reported by `Mesh.taylor_hood_check()` / `Mesh3D.taylor_hood_check()`.
- **Quadrature**: fixed rules that integrate the polynomial parts exactly on affine elements (the trilinear convective
  form at degree `3p - 1` planar and `3p` axisymmetric; no quadrature point on `r = 0`).
- **Boundary conditions** (exactly one per boundary attribute; an attribute without a condition is refused):
  Dirichlet (P2 nodal interpolation of the data, eliminated symmetrically), do-nothing, traction (optionally depending
  on the outward normal of the curved boundary), slip and symmetry (`u . n = 0` strongly, by a local rotation of the
  nodal frame with the consistent, mass-conserving normal; corners are detected from the turning angle), periodic
  (with a prescribed pressure drop or flow rate), axis (`u_r = u_theta = 0` at every velocity node on `r = 0`), and a
  pressure point (`p(x0)` prescribed, raw pressure reported). At a node shared by several conditions Dirichlet wins.
- **Pressure nullspace**: detected algebraically after elimination (`||B^T 1|| / (||B||_F ||1||) <= 1e-10`) and
  required to agree with what the conditions imply (no open boundary); removed by a mean-value multiplier, a pinned
  pressure node, or (MINRES) an orthogonal projection. The continuity right-hand side is projected onto the constants'
  complement before every solve and the flux defect of the interpolated Dirichlet data is reported. A velocity
  nullspace (a rigid motion with zero viscous energy that the boundary conditions leave free) is refused.
- **Forces and torques**: Babuska-Miller variational reactions (the residual tested with a rigid motion supported on
  the body), always with the stress-form residual for the physical force; the Laplacian-form pseudo-traction is
  available as a diagnostic.

### Stabilized equal-order elements (GLS)

`gls_problem` solves planar Stokes, Oseen or Navier-Stokes with continuous P1/P1 and residual-based Galerkin/least
squares stabilization: SUPG + PSPG + LSIC with Shakib-type `tau` built from the metric tensor `G` of the equilateral
reference triangle (`C_I = 36`, `C_G = 1`; `tau_supg` includes `(2/dt)^2` for one implicit BDF1/BDF2 step). The strong
residual keeps the viscosity-gradient term; all element integrals use a fixed 6-point degree-4 rule. One templated
element routine gives the residual, the exact Jacobian (forward-mode automatic differentiation, `tau` differentiated
with the state) and the Jacobian action. Enclosed flows only (every boundary vertex Dirichlet); axisymmetric GLS is not
offered.

### Scalar transport and Boussinesq

`axionfem.transport.ScalarTransport` (and `fluids.solve_heat`) solves steady `a b . grad T - div(kappa grad T) = s`
with P1 or P2 elements, planar or axisymmetric, in the convective form or one of two conservative forms (total-flux or
diffusive-flux natural data), with SUPG (Shakib `tau` with the longest edge, evaluated at the quadrature points) and an
exact Jacobian in `T` and in the advecting velocity. Flux data use the inward convention `g = n . kappa grad T`;
boundary conditions: Dirichlet, flux, Robin, zero flux, total flux (conservative form), axis. `fluids.solve_boussinesq`
couples it (P2 temperature) to Taylor-Hood Navier-Stokes with the buoyancy `rho0 beta g (T - T_ref)` and solves the
monolithic system by Newton with the exact coupled Jacobian and optional buoyancy continuation. Both are steady and run
on one MPI rank.

### Inductionless MHD

With `geometry="planar25d"` and `mhd={...}`, `solve_navier_stokes` adds the Lorentz force of a prescribed field
`B(x, y)` and conductivity `sigma(x, y)`: the current is `J = sigma(-grad phi + E_z e_z + u x B)` with the electric
potential `phi` (P2) as an extra unknown, the force is `J x B` with the same `J`, and the potential equation is the
conservative weak form `(J, grad psi) = <j_n, psi> - (q, psi)`. Electric conditions: insulating, prescribed normal
current, prescribed potential; a potential nullspace is detected and removed like the pressure one. Laplacian viscous
form only, one MPI rank, no induced field.

### Nonlinear solvers

Steady Navier-Stokes, Boussinesq and MHD share one driver: a Stokes start from the Dirichlet lift, optional Picard
(Oseen) steps, then Newton with the exact Jacobian (one kernel gives the convective residual, the Oseen matrix and the
Newton matrix) and an Armijo line search on the discrete dual norm of the residual; Reynolds-number (viscosity)
continuation and pseudo-transient continuation are available. Every solve stops on the steady relative residual in the
dual norm (`newton_rtol`, default 1e-12) and never raises on non-convergence: `diagnostics["converged"]` is the flag,
and the fields are refused when it is false.

### Time integration

`fluids.TransientNavierStokes`: variable-step BDF1/BDF2 (BDF2 starts with one BDF1 step), solved per step either by
Newton or semi-implicitly (an extrapolated advecting velocity, one Oseen solve). The initial field is the discrete
L2 (Leray) projection of the nodal interpolant onto the discretely divergence-free fields with the nodal Dirichlet
data. Adaptive steps use a Milne-device error estimate in the (r-weighted) L2 norm and a step controller with a ratio
cap of 2 (below the BDF2 zero-stability bound). Time-dependent force and boundary data are callables of `(x, y, t)`.

### Linear solvers

Direct sparse LU (STRUMPACK when the build has it, with MC64 matching, else Eigen's SparseLU on one rank) with
iterative refinement checked against the system's own residual; every factorization of a system with a pressure
nullspace is probed so that a singular matrix is refused instead of "solved". MINRES with a block-diagonal
preconditioner is available for Stokes. In 3D a direct solve whose refinement stagnates is finished by restarted GMRES
preconditioned with the same factorization. In transient steps a solve whose refined residual exceeds 1e-6 is repeated
with STRUMPACK without matching and then with Eigen; such fallbacks are reported in the diagnostics.

## Limitations

- Beta: the interface may change; there is no design-file (spec/CLI), viewer or MCP integration, and the module needs
  the MPI source build.
- Ranks: 3D Stokes is the only distributed path. 2D Stokes and Navier-Stokes solves run on several ranks but their
  fields can be evaluated only on one; transport, Boussinesq and MHD refuse more than one rank, and GLS is serial.
- Time stepping: MHD is steady only (the time stepper has no MHD option, and the solver refuses MHD with a
  time-stepping strategy); transient Boussinesq is not offered; BDF orders above 2, generalized-alpha, slip, symmetry
  and periodic conditions and the skew convective form are refused in time steps.
- Not offered: open-boundary stabilization (backflow), turbulence models, compressible or variable-density flow,
  temperature-dependent properties, MHD in axisymmetry, induced magnetic fields, slip, symmetry and periodic conditions
  with convection, iterative (Krylov) solvers for Navier-Stokes, 3D Navier-Stokes.
- STRUMPACK has occasionally returned an inaccurate solution for large transient systems (about 7.5e4 unknowns); the
  solver then repeats the solve without matching and with Eigen (reported in the diagnostics). This chain is not
  exercised by the verification runs, which use a build without STRUMPACK.
- The verification covers manufactured, closed-form and cross-code problems; no published benchmark value (cavity,
  cylinder, natural convection) is compared yet, and some cases were measured only on their first two refinement
  levels (see the validation page).

## References

- I. Babuška and A. Miller, The post-processing approach in the finite element method, Part 1: Calculation of
  displacements, stresses and other higher derivatives of the displacements, *Int. J. Numer. Methods Eng.* **20**,
  1085-1109 (1984). (Variational forces and torques.)
- F. Shakib, T. J. R. Hughes and Z. Johan, A new finite element formulation for computational fluid dynamics: X. The
  compressible Euler and Navier-Stokes equations, *Comput. Methods Appl. Mech. Eng.* **89**, 141-219 (1991).
  (Shakib-type stabilization parameter.)
- R. Temam, *Navier-Stokes Equations: Theory and Numerical Analysis*, North-Holland, Amsterdam (1977). (Skew-symmetric
  convective form.)
