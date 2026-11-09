# Cavity haloscopes

!!! warning "Beta"
    The cavity modules (`axionfem.cavity`, `cavity_xsec`, `cavity_track`, `haloscope`, `haloscope_presets` and
    `cavity_coupling`) are beta: they are tested against closed forms, independent methods and published cavities,
    but their interfaces and validation are not complete. Check the warnings of every result.

A cavity haloscope converts axions into photons in a resonant mode of a cavity in a strong static field. The signal
power is proportional to B² V C Q, where C is the form factor of the mode (its overlap with the static field), V the
volume and Q the loaded quality factor. axionFEM computes the modes and their figures of merit, the wall losses that
set Q₀, tuning sweeps that follow a mode as the geometry changes, and the scan rate and reach of a haloscope built
from them.

## Modes, form factor and Q₀

`axionfem.cavity.pillbox_design` builds an axisymmetric cylindrical cavity with a uniform axial field; the solve
returns its eigenmodes with frequency, form factor and losses:

```python
from axionfem import cavity as cv

a, L = 0.10, 0.30                                         # radius and length [m]
cu = {"kind": "conductor",                                # perfect at RF in the field solve; R_s enters perturbatively
      "rf": {"model": "anomalous", "RRR": 100, "rho_300K_ohm_m": 1.725e-8, "rho_l_ohm_m2": 6.6e-16,
             "specularity": 0.0, "T_K": 0.15}}
d = cv.pillbox_design(a, L, h=0.01, modes=4, families=("TM",), wall_material=cu, Bz=1.0)
res = d.solve("runs/pillbox", verbose=0)
m = res.cavity_modes()[0]                                 # the TM010 mode
print(f"f = {m['f_Hz'] / 1e6:.3f} MHz, C = {m['C']:.4f}, Q0 = {m['Q0']:.3g}, G = {m['G_ohm']:.2f} ohm")
exact = [q for q in cv.pillbox_modes(a, L, 2e9) if (q["family"], q["n"], q["p"]) == ("TM", 1, 0)][0]
print(f"closed form: f = {exact['f_Hz'] / 1e6:.3f} MHz, C = {exact['C']:.4f}")
```

- **Designs.** Besides `pillbox_design`, `coax_design` (a centred rod), `floating_rod_design` and dielectric fills
  are axisymmetric; `axionfem.cavity_xsec.cylinder_rods_design` and `xsec_design` solve the cross-section of a
  z-uniform cavity with off-centre rods (`"symmetry": "xsec"`); `cylinder_design_3d` and `box_design_3d` are 3D
  designs (MPI build). Any design with a `cavity` section works the same way ([design format](../design_format.md)).
- **Mode properties.** Each mode carries `f_Hz`, the form factor `C`, the volume `V_m3`, the geometry factor `G_ohm`,
  and the quality factors from the wall model (`Q_wall`), dielectrics (`Q_dielectric`) and their combination `Q0`.
- **Wall models.** `axionfem.cavity.surface_resistance` evaluates the normal, anomalous-skin, BCS and flux-flow
  models of the `rf` block ([surface resistance](../physics/surface_resistance.md)).
- **Mode fields.** `res.plot_mode(...)`, `res.cavity_mode_probe(...)` and `res.cavity_mode_sample(...)` draw and
  sample a mode; `res.png(mode=i)` renders it.
- **Closed forms** for checks: `pillbox_modes`, `coax_tm_mode`, `sleeve_tm_mode`, and in `cavity_xsec`
  `eccentric_rod_tm_mode` (method of particular solutions).

## Tuning and mode tracking

`axionfem.cavity.sweep(build, values)` solves the design `build(v)` for each value of a tuning parameter (a rod
position, a gap, a length), compares the modes of consecutive steps through their fields sampled on fixed probe
points, and links them into branches (`axionfem.cavity.track`), so that crossings and avoided crossings between
modes are resolved rather than mislabelled; steps too coarse to identify a mode are flagged and can be refined.
Sweeps run with the axisymmetric, cross-section and 3D backends
([tuning and mode tracking](../validation/cavity_tracking.md)).

## Sensitivity and reach

`axionfem.haloscope` turns a mode into a haloscope: signal power, receiver noise (HEMT, quantum-limited, squeezed),
the optimal antenna coupling, the scan rate and the exclusion limit of a scan
([haloscope sensitivity](../physics/haloscope_sensitivity.md)):

```python
from axionfem import haloscope as hs

cav = hs.Cavity(B_T=8.0, V_m3=m["V_m3"], C=m["C"], Q0=m["Q0"], beta=2.0)
rx = hs.Receiver(T_sys_K=0.5)                             # a flat system noise temperature
print(hs.beta_opt(m["f_Hz"], cav, rx))                    # the coupling that maximizes the scan rate
print(hs.scan_rate(m["f_Hz"], cav, rx, snr=5.0))          # Hz/s for a DFSZ axion
```

Presets recompute the limits of published haloscope runs from their published cavity, magnet, receiver and scan
parameters (on the homepage figure the recomputed limits are 0.75 to 1.15 times the published ones):
`hs.preset_models(name)` returns the cavity, receiver and analysis, and `hs.preset_limit(name)` the recomputed limit
next to the published one. A few values are inferred or assumed where a paper does not state them; the presets mark
them. The homepage's reach figure compares the nine reproducible runs with their published limits.

## Receiver coupling

`axionfem.cavity_coupling` computes the external quality factor Q_ext, the coupling β = Q₀/Q_ext and the loaded Q_L
of a coaxial antenna, by a driven solve and by an eigenmode expansion, and hands the result to `axionfem.haloscope`
(`to_haloscope_cavity`). It covers an axisymmetric pillbox with an on-axis probe and TM0np modes only
([receiver coupling](../cavity/coupling.md)).

## Next steps

- [Walkthrough notebook](../../examples/cavity_haloscope.ipynb): empty, rod-loaded and dielectric cavities, wall
  materials, sensitivity and coupling.
- Validation: [published cavities](../validation/cavity_reproductions.md) (HAYSTAC, RADES, CAPP, QUAX and others),
  [axisymmetric modes](../validation/cavity_2d.md), [3D modes](../validation/cavity_3d.md),
  [cross-section solver](../validation/cavity_xsec.md).
