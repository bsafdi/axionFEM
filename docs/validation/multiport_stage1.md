# Small-model multiport and prescribed-source validation

!!! warning "Beta"
    The prescribed source is beta. This page validates the multiport and prescribed-source tooling on small models;
    it is not a convergence study of a detector.

Record: `benchmarks/records/multiport_stage1.json`. The two-port design is `benchmarks/designs/stage1_twoport.json`;
the same record feeds the Touchstone export check (`benchmarks/touchstone_check.py`,
[touchstone.md](../physics/touchstone.md)). Physics: [multiport_fullwave.md](../physics/multiport_fullwave.md) and
[multiport.md](../physics/multiport.md).

## Models

- **Two-port frame.** A copper (RF PEC) square frame with two separate gap volumes sharing a slit. Order-1 Nédélec
  mesh: 85,562 tetrahedra, 108,964 DOFs. Both ports are driven independently at 1 and 12 MHz (cases `multiport`,
  `reversed`), and the frame with one port solved by the multiport build and by a single-port build (cases
  `oneport_new` and `oneport_main`, the record's names), which must agree.
- **Floating electrode.** An isolated electrode adds a current port that bridges two distinct conductor bodies
  (109,483 DOFs, 12 MHz; case `floating_port`). The primary pickup keeps its DC return.
- **Lossy gaps.** A lossy dielectric in both feed gaps gives a genuinely complex Z and a volume-dissipation check
  (case `lossy`).
- **Non-uniform source.** A tilted, offset 64-segment current loop outside the cavity, evaluated exactly and as a
  32-sample azimuthal average on identical RF meshes with order-6 source quadrature (cases `exact_source`,
  `averaged_source`).

All solves use one MPI rank, a PEC sphere at 0.7 m, and one per-frequency Maxwell operator and AMS preconditioner for
all port right-hand sides.

## Results

| Check | Observed (record) | Gate |
|---|---:|---:|
| Unsymmetrized Z reciprocity, maximum | 6.241e-12 | 2e-6 |
| Direct axion-source vs reciprocal overlap, maximum | 1.808e-07 | 2e-6 |
| Direct shunt-loaded FEM vs circuit closure, maximum | 1.102e-10 | 2e-6 |
| One-port L and Im(Z): multiport build vs single-port build | 3.176e-14 | 2e-7 |
| Reversed-port matrix sign transformation | 4.042e-14 | 2e-6 |
| Lossy volume power vs Re(Z) | 8.620e-08 | 5e-5 |
| Lossy resistance eigenvalues [Ω] | 8.24564e-07, 0.00734851 | nonnegative |

The metrics are under `metrics` in the record; the per-case sweeps under `cases`.

- At 12 MHz the shared-loop mutual impedance of the two-port frame is Z12 = −34.512185i Ω (record sign convention,
  e^{−iωt}).
- The floating-electrode port has Z22 = 526.28479i Ω and Z12 = 0.067602406i Ω: its driving impedance is capacitive,
  with a nonzero mutual response.
- Lossless raw resistance matrices can have tiny negative eigenvalues from antisymmetric solver error; they are
  reported, not projected.
- For the non-uniform source, the norm of the Ψ-vector difference between the exact and the azimuthally averaged
  source, divided by the exact-source norm, is 0.97922114 (`source_comparison`). This deliberately asymmetric test
  shows that the source structure enters the overlap; it is not a prediction for a real winding or an accuracy
  estimate. Both sources pass the driven-source reciprocity check.

Direct-load tests insert resistors into the FE operator. Reactive loads are covered on the
[energy and reactive-load page](fullwave_stage2a_energy.md).

## Tests

`tests/test_multiport.py` and `tests/test_multiport_components.py` check the circuit closure;
`tests/test_multiport_fem_spec.py` the parser; `tests/test_fullwave3d.py` the 3D ports on small meshes.

## Not established here

h/p convergence, invariance under the number of MPI ranks, near-winding source quadrature for a real winding,
physical terminal-contact models, imported winding topology, capacitor and ESL extraction, loaded eigenmodes and
frequency-grid convergence of a sweep. No prescribed-source claim is made for radiation boundaries.
