# Full wave (beyond MQS)

The magnetoquasistatic (MQS) results Ψ, L_p and F assume that the detector is small compared with the wavelength of
the signal. As the size approaches the wavelength, displacement current matters: the coupling and the inductance
become frequency dependent, the pickup radiates, and self-resonances appear. The full-wave solver keeps displacement
current and computes, with time dependence e^{−iωt},

$$
\Psi(\omega)=\frac{1}{i\omega I}\int \mathbf E_I\cdot\mathbf B_{\rm lab}\,dV,
\qquad L(\omega)=-\frac{\operatorname{Im}Z(\omega)}{\omega},
$$

where **E**_I is the field driven by a pickup current I, **B**_lab the static field and Z the terminal impedance.
Both reduce to their MQS values as ω → 0; a negative L(ω) marks a frequency above the first self-resonance.

## A frequency sweep

A `fullwave` section with a list of frequencies (or `f_min`, `f_max`, `points`) adds a sweep to the solve:

```python
import axionfem as af

d = af.template("toroid", R=2.0, H=2.0)
d.set(y=0.6)
d.doc["fullwave"] = {"frequencies": [1e5, 1e6, 5e6], "boundary": "radiation"}
res = d.solve("runs/toroid_fw", verbose=0)
for row in res.data["fullwave"]["sweep"]:
    print(f"{row['f_Hz']:.0e} Hz: |Psi|/Psi_MQS = {row['Psi_over_MQS']:.5f}, "
          f"L/L_MQS = {row['L_over_MQS']:.5f}, R_rad = {row['R_rad_ohm']:.2e} ohm")
```

Each sweep row gives Ψ(ω) (`Psi_re`, `Psi_im`, `Psi_abs`), the impedance (`Z_re_ohm`, `Z_im_ohm`), L(ω), the
radiation resistance and Q, a power-balance check (`power_balance_rel`, at round-off level), and, with loss data on
the materials, bulk and surface losses. `axionfem.circuit.fullwave_tables(res.data)` returns the same columns as
sorted arrays.

- **Boundary.** `radiation` (default) is an exact outgoing-wave condition on the outer sphere, so the radiation
  resistance is computed; `pec` closes the domain with a perfectly conducting sphere (a shielded room).
- **Eigenmodes.** `"modes": N` adds the lowest resonances of the closed sphere per family, with the energy fraction by
  region, which tells a pickup resonance from a mode of the surrounding sphere.
- **Family.** An azimuthal pickup (toroid sheath, loops) is driven as TE0, a poloidal port (coax, CORE) as TM0.
- **Mesh.** With frequencies given, the mesh is refined to about 12 nodes per wavelength at order 2; `mesh_kh_max`
  reports the largest k h, and a warning flags coarse meshes.

**Validity of the axisymmetric solver.** It drives the readout uniformly around the axis: an ideal slit for azimuthal
pickups, an annular gap for ports. That is exact while the readout loop's circumference is small compared with the
wavelength. A real slit or gap fed at one point also excites non-axisymmetric currents, which set the first
self-resonance of a real device; well below it, the axisymmetric result is the uniform-feed limit, and a 3D
full-wave solve ([3D designs](3d.md)) checks it.

## Reach from a full-wave sweep

The sweep feeds the readout and the scan. `circuit.pickup_coupling(res.data)` and
`circuit.pickup_impedance(res.data, model="fullwave")` turn it into Ψ(ω) and Z_p(ω) for a readout circuit, and
`axionfem.scan.reach_fullwave(res.data, benchmark)` runs the optimal scan with Ψ(ω), L(ω) and the tuned-pole
effective inductance. Frequencies above the first self-resonance are reported as unreachable rather than tuned.
Where the response varies across the scan band, `axionfem.scan.finite_band` evaluates the exact circuit noise
([readout guide](readout_multiband.md)). The [full-wave walkthrough](../../examples/fullwave.ipynb) runs these steps,
and [validation/finite_band.md](../validation/finite_band.md) compares them with published results.

## Losses

- **Bulk losses** of lossy materials (`tan_delta`, conductivity) enter Z directly (`bulk_losses` per material).
- **Surface losses** of conductors are perturbative: `surface_Gamma` per conductor body is Γ = ∫|H|² dA / I², and with
  `rf` loss data on the material the solver reports R_loss = Σ R_s(f) Γ, Q_loss and Q_total
  ([surface losses](../physics/fullwave_surface_losses.md)). The R_s(f) models of the `rf` block are beta
  ([surface resistance](../physics/surface_resistance.md)).

## 3D full wave

A 3D design (`"symmetry": "3d"`, MPI build) with a `fullwave` section solves the same problem without the
axisymmetric restriction: real slits and gaps, several ports, non-axisymmetric sources.

- **Ports.** Each gap in `fullwave.ports` touches two conducting terminal patches, and `toward` sets the direction of
  its current. Each sweep row then holds the impedance matrix (`Z_matrix_re_ohm`, `Z_matrix_im_ohm`) and the coherent
  coupling vector `Psi_vector`; the pickup comes first.
- **Prescribed static source** (beta). `"boundary": "pec", "source": {"kind": "uniform", "B_T": [0, 1, 0]}` inside
  `fullwave` replaces the static field of the overlap integral by a given one (here 1 T along y, inside a closed
  conducting sphere). A uniform source needs `pec`: with `radiation` boundaries the source must be compact and
  current-conserving, such as closed current segments inside the domain sphere. Prescribed sources need a 3D design
  or a 2D design on an imported mesh. MQS energies and forces are unchanged, and no screening is computed
  ([prescribed source](../physics/prescribed_source.md)).
- **Error indicator and Gauss-law guard** (beta). Both are on by default: every row carries an estimate of its
  discretization error and a check of the divergence condition
  ([error indicator](../physics/fullwave_error_indicator.md), [Gauss-law guard](../physics/fullwave_gauss_guard.md)).
- **Adaptive sweep** (opt-in, beta, 3D): `fullwave.adaptive` with a band and a tolerance fits a rational model to the
  solves and places new solves where it is least certain ([adaptive sweep](../physics/adaptive_sweep.md)).
- **Tree-cotree solve** (opt-in, beta): at low frequency, `AXF_FW_SOLVER=strumpack` with `AXF_FW_GAUGE=tree-cotree`
  separates gradient fields and avoids cancellation. It needs STRUMPACK, first-order elements, real lossless materials
  and PEC on every boundary ([tree-cotree](../physics/fullwave_tree_cotree.md)).
- **Perfectly matched layers and 3D eigenmodes** (beta): `fullwave.pml` and the eigenmodes of a 3D design
  ([design format](../design_format.md)).
- **Closed RF-only models** (beta): `fullwave.rf_only` with `mesh.symmetry` and `mesh.remove_pec` solves closed
  lossless PEC problems with symmetry surfaces, removed conductor volumes and lumped gaps
  ([closed RF-only models](../validation/fullwave_stage2a_boundaries_solver.md)). The energy matrices, field export,
  probes and reactive validation loads of the closed lossless problem are beta as well
  ([energy identity and reactive loads](../validation/fullwave_stage2a_energy.md)).

Current ports and multiport impedance matrices are validated ([3D validation](../validation/fullwave3d.md)); the
features marked beta above are tested but less validated. The output's `"status": "provisional"` is a fixed label of
every 3D full-wave block (and of the 2D imported-mesh engine), not a verdict on a row: whether a row passed is
`validation_accepted`, the `refused` list and the warnings. A row or diagnostic that a solver or validation gate
rejected carries `"status": "provisional_rejected"` and is kept for inspection only.

## Multiport circuits

`axionfem.multiport.MultiportCircuit` connects the detector's impedance matrix to external loads and keeps the
relative phases of the axion-induced voltages; `axionfem.multiport_components.LosslessTransmissionLine` adds the
delay and stored energy of a distributed connection. Include only components that are not already in the FEM model,
and give reciprocal, passive impedances ([multiport circuits](../physics/multiport.md)).

## Touchstone export

A sweep's port impedances can be written as a Touchstone file for an external circuit tool (SPICE, ADS, Microwave
Office, scikit-rf). The e^{−iωt} impedances are conjugated into the e^{+jωt} convention these tools assume, so an
inductor reads Z = +jωL:

```python
report = res.write_touchstone("pickup.s1p", kind="S", z0=50)
print(report["sigma_max"], report["passivity_violations_Hz"], report["reciprocity_error"])
```

From the command line: `axf export-touchstone runs/NAME/results.json --kind S --z0 50`. The command exits with 0 when
the file was written and is passive, 3 when it was written but is not passive, and 2 on an error. The file is
Touchstone v1 with increasing frequencies and a comment giving the port order and polarity (port 1 is the pickup).
The report gives σ_max(S), which must not exceed 1 for a passive structure; a violation is reported, and the data are
written as computed, never clipped. `res.write_touchstone(..., dense=True)` writes the dense model curve of an
adaptive sweep, and `axionfem.touchstone.read_touchstone` reads files back
([S-, Y- and Z-parameters](../physics/touchstone.md)).
