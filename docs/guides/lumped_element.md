# Lumped-element detectors

A lumped-element detector puts a static magnetic field **B**_lab in a volume and reads out the tiny oscillating flux
that axion dark matter would induce through a superconducting pickup. axionFEM computes the quantities that define
such a search:

- the static field **B**_lab, with nonlinear iron, superconductors and the exact open boundary;
- the **pickup mode**, the field of a unit readout current, which gives the pickup inductance L_p;
- the **signal coupling** Ψ = (1/I)∫ **A**_I · **B**_lab dV, so that the axion flux is Φ_a = κ_a Ψ;
- the **scan figure of merit** F = Ψ⁴/L_p² against the requirement F_req of the optimal resonant scan, for a noise
  benchmark;
- stored energy, peak fields in windings and on superconducting surfaces, forces, and a cost metric.

These are magnetoquasistatic (MQS) results, valid while the detector is small compared with the signal wavelength;
[Full wave](fullwave.md) covers larger detectors and higher frequencies.

## The three templates

`af.template` builds the three common topologies with sensible defaults (see [Concepts](concepts.md#templates)):

```python
import axionfem as af

for name in ("toroid", "solenoid", "core"):
    res = af.template(name).solve(f"runs/{name}", verbose=0)
    b = res.benchmark("optimistic")
    print(f"{name:9s} Psi = {res.Psi:.3e} H T m^2   L_p = {res.L_p:.3e} H   F/F_req = {b['F_over_F_req']:.2e}")
```

`res.summary()` prints the full set: Ψ (SI and natural units), L_p, the stored energy, the geometry factor, the peak
field, the peak surface field on superconductors, and for each benchmark the ratio F/F_req, the field amplitude the
design would need to meet the requirement, the energy and peak field at that amplitude, the cost metric and the
cost-optimal size.

- **Benchmarks.** `optimistic`, `conservative` and `conservative_solenoid` fix the resonator quality factor,
  temperature and amplifier noise. They are illustrative presets; a design may also give its own (`"benchmarks": [{"name": ..., "Q": ...,
  "T": ..., "eta_A": ...}]`). [Signal, noise and optimal scan](../physics/scan_and_signal.md) defines F_req.
- **Scaling to the requirement** is exact only for linear designs without fixed fluxes, circuits or backgrounds;
  the result says so when it is not.
- **Cost metric.** A scaling metric built from the stored energy and the cold area, for comparing designs; it is not a
  construction estimate.

## Building a design

Templates are a starting point. A design is a JSON document with regions, materials, windings, superconductors and a
pickup ([design format](../design_format.md)):

- **Windings** are `azimuthal` (current around the z axis: solenoids, loops) or `poloidal` (current around the
  cross-section of a ring-shaped region: toroids), with `ampere_turns` or `turns` in a circuit. Circuits can be
  powered (`current`) or persistent (a closed superconducting loop that conserves its flux linkage).
- **Superconductors** get a default state that can be changed: no net transport current (field-cooled), or a fixed
  flux (`{"flux": 0}` for zero-field cooling); `azimuthal_slit` marks a meridional slit. In the pickup problem every
  uncut superconducting loop conserves its fluxoid automatically.
- **Pickup.** `{"type": "azimuthal", "conductor": ...}` for a slit superconductor (toroid sheath, loops) or
  `{"type": "poloidal", "port": ...}` for a gap between conductors (coax, CORE neck).
- **Iron** with a B-H table is solved by Newton iteration; in RF problems it uses the secant permeability at the
  static operating point, which is exact for RF fields perpendicular to the static field.
- **Background field.** A uniform field or a grid (for example a magnet field mapped in another code) in which the
  design sits; superconductors and persistent circuits respond to it.

Change parameters with `d.set(...)`, edit `d.doc` directly for anything else, and run `d.validate()` before solving.

## Superconductors, conductors and HTS

- **Surface fields.** Results report the peak field on every superconductor surface and a regular peak that
  excludes three element sizes around sharp corners, where the field is singular and grows under refinement. Round
  the corners (`fillet`) before relying on a peak surface field.
- **Conductor margins.** A `conductors` entry (built in: `nbti`, `nb3sn`, `rebco`, or a custom critical surface)
  attached to a winding gives the operating point, the load-line fraction J/Jc, the short-sample scale and the
  temperature margin. `af.conductors()` lists the built-ins and `af.jc(...)` evaluates them.
- **HTS critical state.** An `hts` section solves the screening currents of high-temperature superconductor regions
  under a field ramp with an E-J power law (axisymmetric, azimuthal currents): field quality, AC loss and the effect
  on Ψ ([validation](../validation/hts.md)).
- **Thin magnetic layers** model foils and shields as zero-thickness sheets
  ([thin magnetic layers](../physics/thin_magnetic_layer.md)).
- **Capacitance.** For a port pickup every solve reports the readout-gap capacitance (`pickup.port_capacitance`)
  with an estimate of the self-resonance it causes; an `electrostatics` section computes the capacitance matrix of
  the conductors (axisymmetric and 3D).

## Forces

Every solve reports the Lorentz force on each winding (with its numerical self-force as an error bar), the Maxwell
pressure on superconductors and iron, and a closure check that the forces on all bodies sum to zero.
`res.forces["distributions"]` holds the distributed load of each body; [Mechanics and thermal](mechanics_thermal.md)
turns it into stresses.

## Next steps

- [Walkthrough notebook](../../examples/lumped_element.ipynb): templates, fields, convergence, geometry comparison.
- [Readout and SQUID chains](readout_multiband.md): the circuit behind the pickup and its noise.
- [Reach, sensitivities and optimization](optimization.md): scan reach, required field and size, gradients.
- [3D designs](3d.md): real slits and ports.
- Validation: [published lumped-element designs](../validation/published_benchmarks.md) (ABRACADABRA-10 cm, DMRadio)
  and [axisymmetric solvers](../validation/solvers.md).
