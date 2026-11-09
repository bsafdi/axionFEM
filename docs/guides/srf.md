# SRF heterodyne cavities

!!! warning "Beta"
    `axionfem.srf` and `axionfem.heterodyne` are beta: the mode solutions are checked against closed forms, a 3D
    model and published cavities, but the interfaces and the sensitivity model may still change.

In a heterodyne search a superconducting radio-frequency (SRF) cavity is driven in a pump mode, and axion dark matter
converts pump photons into a nearly degenerate signal mode. The signal grows with the overlap of the pump's magnetic
field with the signal's electric field, η₁₀, and with the stored energy the walls allow. axionFEM builds the cavity
geometries, finds and labels the pump and signal modes, tunes them to degeneracy, and computes the figures of merit:
η₁₀, the geometry factors G, the ratio ξ of surface to volume fields, and the peak wall fields relative to the volume
rms field.

## A mode pair in a closed cylinder

`srf.cylinder_pair_design` builds a closed cylinder at the aspect ratio where a named pair of modes is degenerate,
and `srf.figures_of_merit` evaluates the pair. This example is the 60 L cylinder of R. Lasenby, Phys. Rev. D 102,
015008 (2020), arXiv:1912.11056, with the TE012 pump and the TM013 signal:

```python
from axionfem import srf

d = srf.cylinder_pair_design("TE012/TM013", 0.20, h=0.20 / 20, order=3, modes=8)   # radius 20 cm
res = d.solve("runs/srf_cylinder", verbose=0)
fom = srf.figures_of_merit(res, "TE012", "TM013")
print(f"f = {fom['f_pump_Hz'] / 1e9:.4f} GHz, V = {fom['V_m3'] * 1e3:.1f} L, eta_10 = {fom['eta_10']:.5f}")
print(f"closed form: eta_10 = {srf.cylinder_pair('TE012/TM013')['eta_10']:.5f}")
```

The pairs with closed forms are listed in `srf.CYLINDER_PAIRS`. `figures_of_merit` also returns both geometry factors
(`G_pump_ohm`, `G_signal_ohm`), `xi_pump` and `xi_signal`, the wall peaks `Bpk_over_B0` and `Epk_over_cB0`, the
frequency splitting and the wall area.

## Elliptical cells

`srf.elliptical_cell_design` builds an elliptical cell from its contour parameters; `srf.PROJECTX_650_B09` is the
Project X 650 MHz β = 0.9 cell. The wall is an exact elliptical contour (curved elements), and the wall peaks are taken
at the Gauss points of the wall, so they converge under refinement.

- **Labels.** `srf.classify_modes` labels the modes of a solve (TM020, TE011, ...) and `srf.find_mode` picks one.
- **Degeneracy.** `srf.tune_degeneracy("TM020", "TE011", ...)` searches the axial scale factor of the cell at which
  the two modes are degenerate.
- **3D.** `srf.elliptical_cell_design_3d` and `srf.figures_of_merit_3d` repeat the analysis in 3D (MPI build) as a
  check of the axisymmetric wall peaks.

## Sensitivity

`axionfem.heterodyne` turns the figures of merit into a heterodyne sensitivity: signal and noise models, the optimal
coupling, the scan rate and the required volume for a target coupling
([heterodyne sensitivity](../srf/sensitivity.md)). Wall vibrations mix the pump into the signal mode; their
requirement is in [vibrations and mode mixing](../srf/vibrations.md).

## Next steps

- [Walkthrough notebook](../../examples/srf_heterodyne.ipynb): closed cylinders, the elliptical cell, tuning, wall
  fields, the 3D comparison and the sensitivity.
- [Modes and figures of merit](../srf/README.md): definitions, wall representation and validation (TESLA cells,
  closed cylinders, measured niobium surface resistance, 2D against 3D).
