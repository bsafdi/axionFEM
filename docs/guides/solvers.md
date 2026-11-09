# Solvers and numerical controls

A design states what to compute; axionFEM chooses the solver from its `symmetry` and its sections. This page lists
the solvers, the settings that control their accuracy and cost, and where each solve reports what it did.

## Which solver runs

| design | solver | build |
|---|---|---|
| axisymmetric (default) | magnetoquasistatic (MQS) static field and pickup mode: flux function ψ = rA_φ for poloidal fields and G = rH_φ for toroidal fields, exact Dirichlet-to-Neumann open boundary, superconductors as floating bodies with fluxoid or current constraints, Newton iteration for nonlinear iron | any |
| axisymmetric with `fullwave` | full wave, m = 0 (both TE and TM families) with exact radiation or PEC boundary; MQS is its ω → 0 limit | any |
| axisymmetric with `cavity` | cavity eigenmodes, form factors and wall losses (beta) | any |
| `"symmetry": "xsec"` | eigenmodes of the cross-section of a z-uniform cavity (beta) | any |
| `"symmetry": "3d"` | 3D H(curl) engine: Nédélec elements, open boundary by an infinity-shell map, hypre AMS preconditioning; MQS, full wave and eigenmodes (beta) | MPI |
| with `electrostatics` | capacitance matrix (axisymmetric and 3D) | any (3D: MPI) |
| with `mechanics` / `thermal` | elasticity under magnetic and cooldown loads; steady conduction with k(T) (axisymmetric and 3D) | any (3D: MPI) |
| with `hts` | critical-state screening currents in HTS regions (axisymmetric) | any |

The [physics pages](../physics/index.md) give the formulations, and the [design format](../design_format.md)
every section's fields. A 3D design in a build without MPI stops with an error that says so.

## Accuracy: order and mesh

| setting | effect |
|---|---|
| `mesh.order` (or `solve(order=...)`) | element order, default 2. Raising it is the cheapest convergence check |
| `mesh.size`, `mesh.surface_size` | element size inside the device and at boundaries of superconductors, iron, conductors, windings and ports |
| `<region>.mesh_size`, `<region>.surface_size` | the same for one region |
| `mesh.refinements` | uniform refinements of the generated mesh (each multiplies the element count by four in 2D) |
| `mesh.adapt` | adaptive refinement until Ψ, L_p and W (or a chosen list) change by less than `tol` between cycles |
| `mesh.domain_radius` | radius of the outer boundary; the open boundary is exact, so it only changes the cost |

[Concepts](concepts.md#convergence) shows a convergence check, and the
[mesh size control](../design_format.md#mesh-size-control) section of the design format lists every field.

## Cost: threads, solvers and memory

- **Threads.** The C++ core uses OpenMP in its assembly loops and in STRUMPACK. `solve(threads=N)`,
  `af.set_threads(N)` or the environment variable `AXF_THREADS` set the thread count; otherwise `OMP_NUM_THREADS`
  applies. Results do not depend on the thread count beyond round-off.
- **Axisymmetric systems** are factorized by a sparse direct solver: Eigen's LDLᵀ with AMD ordering for small
  systems, METIS nested dissection above 1.5 × 10⁵ unknowns, and STRUMPACK, when the build has it, above
  `AXF_STRUMPACK_MIN_DOFS` (default 3 × 10⁵). A build with neither METIS nor STRUMPACK refuses systems above
  4 × 10⁵ unknowns rather than risk an unreliable ordering. All backends agree to round-off.
- **3D systems** use Krylov solvers preconditioned by hypre's AMS (and BoomerAMG for the scalar problems) on MPI
  ranks, with STRUMPACK as an optional direct path for full wave. Every Krylov solve checks its true residual.
- **Memory** of 3D solves grows quickly with the element order and the number of unknowns; ranks × threads should
  not exceed the cores of the machine. [3D designs, MPI and clusters](3d.md) covers runs on several ranks.

[Build options](../install.md#sparse-direct-solvers) describes how to build with METIS and STRUMPACK and the
run-time switches (`AXF_ORDERING`, `AXF_STRUMPACK_MIN_DOFS`).

## What each solve reports

Every solve records which backend ran and what it cost, so a result can be checked after the fact:

```python
import axionfem as af

d = af.template("toroid", R=2.0, H=2.0)
d.set(y=0.6)
res = d.solve("runs/toroid", verbose=0)
s = res.data["pickup"]["solver"]
print(s["backend"], s["dofs"], s["final_residual"])   # e.g. eigen-ldlt, the unknowns, the true residual
for c in res.data["checks"]:                          # independent consistency checks
    print(c["name"])
print(res.data["warnings"])
```

For this design the checks compare the pickup inductance from flux and from energy, and verify that the axial forces
on all bodies sum to zero (Newton's third law). A solve that fails to converge says so in `warnings`, or stops with an
error whose hint includes a check of the worst mesh elements.
