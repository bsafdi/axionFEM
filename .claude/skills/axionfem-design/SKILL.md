---
name: axionfem-design
description: Design, solve and compare axion dark-matter detectors with the axionFEM finite-element package - lumped-element detectors (toroid/ABRACADABRA, solenoid/DMRadio, CORE, custom geometries) and the beta modules for cavity haloscopes, SRF heterodyne cavities and dc SQUID readout chains. Use when the user asks to build, modify, optimize, visualize or evaluate a detector geometry, its pickup coupling, inductance, fields on superconductors, forces, scan reach, readout noise or cost.
---

# Designing axion detectors with axionFEM

## Tools
Prefer the `axionfem` MCP tools when they are available, and call `axionfem_guide` first. Otherwise use Python in
the environment where axionFEM is installed (`python -c "import axionfem"` succeeds there) with
`import axionfem as af`; setting `PYTHONDONTWRITEBYTECODE=1` keeps a source checkout free of `__pycache__`.
`af.build_info()` says whether the build has the 3D engine (`engine_3d`); the wheels do not.

## Workflow
1. Start from a template (`af.templates()`: toroid, solenoid, core) or write a design document
   (docs/design_format.md). Geometry is axisymmetric (r, z) unless `"symmetry": "3d"`.
2. Validate (`d.validate()`); errors name the JSON path and give a hint: fix exactly that.
3. Solve (`d.solve(workdir=...)`), read `res.summary()`; check `res.data["warnings"]` and `res.data["checks"]`.
4. Look at the result: `res.png(view="section")` (open the PNG and inspect it), `res.png(view="3d")`,
   `res.export_html()` for the interactive viewer.
5. Interpret the result: F/F_req per benchmark, amplitude scale needed, peak fields
   at the requirement (winding peak against the 9 T / 16 T rules of thumb for NbTi / Nb3Sn; superconductor surface
   against Nb limits, ~86 mT with margin, 172 mT bare), cost metric (not a construction estimate), reach curve
   (`res.reach("optimistic")`), signal/noise PSD (`axionfem.plots.psd(res, f0)`).
6. Explore: `d.set(param=value)` and re-solve; parameter sweeps, shape sensitivities and
   optimization are available (MCP `sweep`, `sensitivities`, `optimize_design` (gradient-based),
   `optimize_global` (differential evolution), `optimize` (derivative-free, bounded Powell)).

## Physics conventions that matter
- Psi = (1/I) int A_I . B_lab d^3x [H T m^2 SI]; natural Psi = Psi_SI/mu0 [T m^3]. Phi_a = kappa_a Psi.
- Benchmarks (illustrative presets; give a design its own {name, Q, T, eta_A} for a real readout):
  optimistic (Q=2e7, T=10 mK, eta_A=0.1), conservative (Q=1e6, 30 mK, eta_A=10),
  conservative_solenoid (Q=1e5). F_req for the default scan (an illustrative target; set the design's `scan` section
  for your own search): 3 yr, 0.5-4.1 neV, DFSZ with C_QCD = 1.77.
- Superconductors: ideal Meissner bodies; default state "no net transport current"
  (field-cooled). RF pickup modes conserve fluxoid automatically. A toroid sheath needs
  `azimuthal_slit: true` (axisymmetric) or a real slit + port region (3D).
- Iron is warm and nonlinear (B-H law); its cost enters via mass. RF problems use its secant permeability.
- Conductors: give a winding `"conductor": "nbti" | "nb3sn" | "rebco"` (or your own under
  `conductors`) to get real critical-current margins instead of the rules of thumb above:
  `res.conductors` and, per benchmark, `conductor_margins` report J_e and J_sc (fill factor), the field
  at the worst point of the winding (with the angle to the tape), Jc there, I/Ic, the short-sample
  scale and the temperature margin. `cost.load_line_cap` makes `optimize` respect it.
- HTS screening currents: an `hts` section (regions + drive waveform) solves the critical state
  (E-J power law) in designated superconducting regions while the static solve keeps the ideal model:
  `res.hts` gives the screening-current distribution, the screening-current-induced field at the
  field-of-view points, trapped flux after a ramp-down, AC loss per cycle and Psi recomputed with the
  critical-state field. Azimuthal screening currents only (poloidal static fields).
- Beyond magnetoquasistatics: a `fullwave` section (MCP `set_fullwave`) gives Z(omega), Psi(omega), the radiation Q
  and self-resonances. The axisymmetric model is the uniform-feed limit (readout circumference much smaller than the
  wavelength); slit and port resonances need a 3D design.

## Other detector types
These modules are **beta**: say so when you report their results.
- Cavity haloscopes: `cavity_presets` / `cavity_preset_limit` (published runs and the exclusion recomputed from their
  numbers), `cavity_scan` (scan rate, optimal coupling, live time to cover a band), `create_cavity` then
  `solve_design` (eigenmodes, form factor C, geometry factor G, Q0). Python: `axionfem.cavity`, `axionfem.haloscope`.
- SRF heterodyne cavities: `srf_design` (volume, frequency, RF power and cost metric for a scan), `srf_mode_pair`
  (figures of merit of a degenerate mode pair). Python: `axionfem.srf`, `axionfem.heterodyne`.
- dc SQUID readout: `squid_chain` (added noise of a two-stage chain), `squid_reach` (reach of a solved lumped-element
  design read out by it). Python: `axionfem.squid`.
- COMSOL and network tools: `export_comsol`, `inspect_comsol`, `import_comsol` (import is beta), and
  `export_touchstone` (S/Z/Y parameters of a full-wave sweep).
- Python only, no MCP tools: fluids (`axionfem.fluids`, MPI builds), scalar transport (`axionfem.transport`),
  chemistry (`axionfem.chem`) and cavity probe coupling (`axionfem.cavity_coupling`).

## Honesty rules
Report mesh convergence (solve with order=3 or a finer mesh) before quoting numbers; say when a
result depends on a sharp corner (field singularities) or on an idealization (perfect Meissner,
assigned Q). Never claim a design is feasible beyond what was computed.
