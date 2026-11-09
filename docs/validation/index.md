# Validation

Each capability below is checked, and the *Check type* column says how: against a closed form, a published result,
an independent implementation or the other engine, by an identity the discrete problem must satisfy, by convergence,
or (where noted) only as a regression or round trip. The pages below give the numbers; the table says where each check
lives, so that it can be rerun.

- **Benchmarks** (`benchmarks/*.py`) run the full checks and write a record (`benchmarks/records/*.json`) with the
  results and the provenance of the run (commit, executables, environment; the commit identifiers refer to the
  development history, which is not public). Pages marked *generated* are written by their benchmark from the record;
  the others are written by hand from the record.
- **Tests** (`tests/*.py`, `cpp/tests/*.cpp`) repeat the main checks on small meshes, or against stored record values.
  The default suite (`pytest -q tests`, [install.md](../install.md)) is fast; tests marked `AXF_SLOW` or needing MPI
  or a GPU run only when enabled, as their docstrings say.
- **Agreement with a publication** is stated only in the pages' own verdict columns. A benchmark's `pass` flag means
  internal consistency (identities, convergence, regression), not agreement with the literature.
- Modules labelled **beta** ([physics index](../physics/index.md)) are validated as described, but their interfaces
  and coverage may change.

Check types: **closed form** (an exact or asymptotic analytic solution), **published** (numbers from the literature),
**independent code** (another implementation of the same quantity), **cross-engine** (the 3D engine against the
axisymmetric one, or 2D against 3D), **identity** (a relation the discrete solution satisfies exactly, which tests
the solves), **convergence** (h or p refinement), **regression** (bit-identity or stored values).

## Magnetoquasistatics and the readout

| Capability | Check type | Reference | Page | Benchmark / record | Tests |
|---|---|---|---|---|---|
| 3D MQS engine | cross-engine, convergence | axisymmetric engine on solenoid, toroid and CORE templates | [3D vs axisymmetric](m2_3d_vs_2d.md) (generated) | `benchmarks/m2_cross.py`, `m2_cross_*.json` | `tests/test_3d.py` |
| AMS-preconditioned 3D solves | convergence, regression | per-element vs global gauge mass on slits, ND3 and iron | [ams_robustness.md](ams_robustness.md) | `benchmarks/ams_robustness.py` | `tests/test_3d.py` |
| Sparse direct solvers (axisymmetric) | regression | STRUMPACK against Eigen LDLT, same results | [solvers.md](solvers.md) (generated) | `benchmarks/solver_scaling.py` | `tests/test_solvers.py`, `cpp/tests/test_solvers.cpp` |
| Adjoint shape sensitivities | independent code | finite differences at fixed topology and with remeshing; analytic derivatives | [adjoint.md](adjoint.md) (generated) | `benchmarks/adjoint_validation.py` | `tests/test_adjoint.py`, `tests/test_global_opt.py` |
| Thin high-permeability layers | closed form, cross-model | exact thick shells and cans; wall meshed through its thickness | [thin_magnetic_layer.md](thin_magnetic_layer.md) | `benchmarks/thin_magnetic_layer.py`, `thin_magnetic_layer.json` | `tests/test_thin_magnetic_layer.py` |
| Critical state (HTS screening currents) | closed form, published | Bean critical state of a long cylinder; published J_c parametrizations | [hts.md](hts.md) (generated) | `benchmarks/hts_validation.py` | `tests/test_hts.py` |
| Signal, noise, statistics and optimal scan | closed form, published | check values of the scan formulas; published scan rates | [scan_and_signal.md](../physics/scan_and_signal.md#8-check-values-full-list-and-more-digits-in-the-json) | `tests/reference/scan_reference_values.json` | `cpp/tests/test_scan.cpp`, `tests/test_circuit.py` |
| Finite-band scan with the exact circuit noise | independent code, identity | single-pole reduction of the same settings; beyond-MQS reach | [finite_band.md](finite_band.md) (generated) | `benchmarks/finite_band.py` | `tests/test_finite_band.py` |
| Multi-band and multipole scans (beta) | independent code, closed form | band, slope and allocation rules on a synthetic Foster-form table; the C++ scan core; single-band reduction to `scan.reach_fullwave`; MQS-limit closed forms; a transcription of the computation behind arXiv:2211.00008 | [multipole_scan.md](../physics/multipole_scan.md) | — | `tests/test_multipole_scan.py` (tests on the arXiv:2211.00008 tables skip without them) |
| dc SQUID amplifier chains (beta) | closed form | closed forms of arXiv:2504.20398 | [squid_chain.md](../physics/squid_chain.md) | — | `tests/test_squid.py` |
| Published lumped-element detectors | published, closed form | ABRACADABRA-10 cm, DMRadio-50L, -m³, -GUT, -Core | [published_benchmarks.md](published_benchmarks.md) (generated, section 7 by hand) | `benchmarks/published_benchmarks.py`, `published_*.json` | `tests/test_published.py` |

## Full wave

| Capability | Check type | Reference | Page | Benchmark / record | Tests |
|---|---|---|---|---|---|
| Axisymmetric full wave | closed form, published | PEC-sphere eigenmodes, loop radiation resistance, sealed coax; Benabou et al. (2022); MQS limit of the templates | [fullwave.md](fullwave.md) (generated) | `benchmarks/fullwave_validation.py` | `tests/test_fullwave.py` |
| 3D full wave, current ports | closed form, cross-engine | thin-wire loop theory; axisymmetric solver; exact cavity modes | [fullwave3d.md](fullwave3d.md) (generated; its last section by hand, without a record) | `benchmarks/fullwave3d_validation.py` | `tests/test_fullwave3d.py` |
| Multiport closure; prescribed source (beta) | identity | reciprocity, source reciprocity, loaded closure, power balance | [multiport and prescribed source](multiport_stage1.md) | `multiport_stage1.json` | `tests/test_multiport.py`, `tests/test_multiport_fem_spec.py` |
| Closed RF-only models: symmetry surfaces, PEC holes, lumped gaps, direct solvers (beta) | closed form, independent code | exact TEM line; distributed against rank-zero STRUMPACK against AMS (symmetry surfaces: no automated test yet) | [closed RF-only models](fullwave_stage2a_boundaries_solver.md) | `benchmarks/strumpack_compression_check.py` | `tests/test_fullwave3d_strumpack_mpi.py`, `tests/test_fullwave3d_solver_paths.py` |
| Energy identity, field export, reactive loads (beta) | identity | dX/dω = −4G at fixed geometry; circuit closure with C, L, ESL loads | [energy identity and reactive loads](fullwave_stage2a_energy.md) | — | `tests/test_fullwave_energy_reference.py`, `tests/test_fullwave3d_dzdw_terminations.py` |
| Prescribed source with a radiating exterior (beta) | identity, closed form, convergence | power balance, dipole formula, Kirchhoff closure; mesh convergence of the exterior interface | [fullwave_radiating_source.md](fullwave_radiating_source.md) | — | `tests/test_fullwave3d_radiating_source.py` (AXF_SLOW) |
| Imposed background field, 3D (beta) | cross-model, closed form | the coil as a winding and as a background on one mesh; MQS limit; domain-radius independence | [fullwave3d_background.md](fullwave3d_background.md) | `benchmarks/fullwave3d_background.py`, `fullwave3d_background.json` | `tests/test_background_3d.py`, `tests/test_fullwave3d_background.py` |
| Background exterior term, axisymmetric | cross-engine, closed form | 3D engine; exact exterior of a solenoid | [axi_background_exterior.md](axi_background_exterior.md) | `benchmarks/axi_background_exterior.py`, `axi_background_exterior.json` | `tests/test_axi_background_exterior.py`, `cpp/tests/test_background_exterior.cpp` |
| Tree-cotree gauged solve (beta) | independent code, closed form | AMS and plain STRUMPACK paths; capacitor closed form; MQS limit | [tree_cotree.md](tree_cotree.md) (generated), [fullwave_tree_cotree.md](../physics/fullwave_tree_cotree.md) | `benchmarks/tree_cotree_*.py`, `tree_cotree_*.json` | `tests/test_fullwave3d_tree_cotree_options.py` |
| Error indicator (beta) | closed form, convergence | exact error on a TEM line; refined solutions of a crack, a 270° edge, lossy and high-contrast lines | [fullwave_error_indicator.md](../physics/fullwave_error_indicator.md) | `fullwave_indicator.json` | `tests/test_fullwave3d_error_indicator.py`, `cpp/tests/test_fw_indicator_local.cpp` |
| Gauss-law guard (beta) | independent code | the tree-cotree gauged solve of the same system | [fullwave_gauss_guard.md](../physics/fullwave_gauss_guard.md) | `gauss_law_guard.json` | `tests/test_fullwave3d_gauss_guard.py` |
| Surface losses (the R_s models that turn them into Q are beta) | closed form, cross-engine | sealed coax at every frequency; MQS limit; axisymmetric engine | [fullwave_surface_losses.md](../physics/fullwave_surface_losses.md) | `fullwave_surface_losses.json` | `tests/test_fullwave3d_surface_losses.py` |
| Adaptive frequency sweep (beta) | independent code | random rational responses (the driver alone); fixed-list solves of the same FEM models | [adaptive_sweep.md](../physics/adaptive_sweep.md) | `adaptive_sweep.json` | `tests/test_adaptive_sweep.py`, `cpp/tests/test_rational_fit.cpp` |
| S-, Y-, Z-parameters and Touchstone files | closed form | series RLC, TEM lines; reciprocity and passivity of solved models | [touchstone.md](../physics/touchstone.md) | `benchmarks/touchstone_check.py`, `touchstone_check.json` | `tests/test_touchstone.py` |
| 2D full wave on imported meshes, Robin boundaries, PML (beta) | closed form | closed-form fields with impedance, transition and scattering boundaries and PML | [fw2d.md](../physics/fw2d.md) | `fw2d_validation.json`, `fw3d_robin_validation.json` | `tests/test_comsol_import2d.py` |
| Memory over long sweeps | regression | bit-identity with and without the allocator settings | [fullwave_memory.md](fullwave_memory.md) | `benchmarks/fullwave_memory.py`, `fullwave_memory.json` | `tests/test_fullwave3d_memory.py` (AXF_SLOW) |
| GPU (CUDA) build | independent code, regression | the CPU build on the same mesh | [gpu.md](gpu.md) | `benchmarks/gpu_validation.py`, `gpu_validation.json` | `tests/test_gpu.py` (needs a GPU) |

## Cavities (beta)

| Capability | Check type | Reference | Page | Benchmark / record | Tests |
|---|---|---|---|---|---|
| Axisymmetric cavity modes, form factors, losses | closed form, convergence | pillbox and coaxial modes and their C, G, Q | [cavity_2d.md](cavity_2d.md) (generated) | `benchmarks/cavity_2d_validation.py`, `cavity_mc*.json` | `tests/test_cavity.py`, `tests/test_cavity_modefields.py` |
| 3D cavity modes | closed form, cross-engine | analytic TM/TE modes; axisymmetric solver | [cavity_3d.md](cavity_3d.md) (generated) | `benchmarks/cavity_3d_validation.py`, `cavity_mc{5,6,7}_*.json` | `tests/test_cavity3d.py` |
| Cross-section cavity solver | closed form, cross-engine | TM_mn0 modes; 3D solver | [cavity_xsec.md](cavity_xsec.md) | `benchmarks/cavity/xsec_*.py`, `cavity_xsec_*.json` | `tests/test_cavity_xsec.py` |
| Published cavity simulations | published, independent code | HAYSTAC, QUAX, RADES, CAPP and others; an independent solver; independently derived values | [cavity_reproductions.md](cavity_reproductions.md) | `benchmarks/cavity/repro/`, `cavity_repro_*.json` | `tests/test_cavity_repro_models.py` |
| Tuning sweeps and mode tracking | published | published tuning curves | [cavity_tracking.md](cavity_tracking.md) | `benchmarks/cavity/tracking/` | `tests/test_cavity_track.py` |
| Haloscope sensitivity | closed form | two independent derivations of the scan-rate formulas | [haloscope_sensitivity.md](../physics/haloscope_sensitivity.md) | — | `tests/test_haloscope.py` |
| Receiver coupling through an antenna port | published, convergence | probed-pillbox external Q | [coupling.md](../cavity/coupling.md) | `cavity_coupling.json` | `tests/test_cavity_coupling.py` |
| Surface resistance of RF walls | closed form, published | anomalous-skin and BCS limits; published values | [surface_resistance.md](../physics/surface_resistance.md) | — | `tests/test_surface_resistance.py`, `cpp/tests/test_surface_resistance.cpp` |
| SRF cavities and wall vibrations | published, closed form | TESLA and published SRF numbers; selection rule | [srf/README.md](../srf/README.md), [vibrations.md](../srf/vibrations.md) | `benchmarks/srf/results/*.json` | `tests/test_srf.py`, `tests/test_srf_external.py`, `tests/test_srf_vibration.py` |

## Electrostatics, mechanics and thermal

| Capability | Check type | Reference | Page | Benchmark / record | Tests |
|---|---|---|---|---|---|
| Axisymmetric electrostatics and capacitance | closed form, identity | isolated, concentric and dielectric-coated spheres, a conductor sheet, two spheres (bispherical series), thin rings; Gauss-law and energy identities | [electrostatics.md](electrostatics.md) | `benchmarks/electrostatics_validation.py`, `electrostatics_*.json` | `tests/test_electrostatics.py` |
| 3D electrostatics | closed form, cross-engine | the same references; the axisymmetric solver | [electrostatics3d.md](electrostatics3d.md) | `benchmarks/electrostatics3d_validation.py`, `electrostatics3d_*.json` | `tests/test_electrostatics3d.py` |
| Axisymmetric elasticity and conduction | closed form, convergence | Lamé cylinder, solenoid hoop stress, superconducting tube under Maxwell pressure, cooldown; conduction with generation and with k(T) | [mechanics_thermal.md](mechanics_thermal.md) (generated) | `benchmarks/mechanics_thermal.py`, `mechanics_*.json`, `thermal_*.json` | `tests/test_mechanics.py`, `tests/test_thermal.py` |
| 3D elasticity and conduction | closed form, cross-engine | Lamé cylinder (ends fixed, free, sector), conduction slabs | [mech3d.md](mech3d.md) (generated) | `benchmarks/mech3d_validation.py`, `mech3d_*.json` | `tests/test_mechanics3d.py` |
| Torsional dynamics (axisymmetric m = 0, beta) | closed form | free-free tube modes, rigid rotation | [torsion.md](torsion.md) (generated) | `benchmarks/torsion.py` | `tests/test_torsion.py` |
| High-frequency torsional mobility (beta) | closed form, independent code | finite free-free tube modal sum; multi-branch waveguide | [torsion_highfreq.md](torsion_highfreq.md) (generated) | `benchmarks/torsion_highfreq.py`, `torsion_highfreq_*.json` | `tests/test_torsion_highfreq.py` |

## Fluids and chemistry (beta)

| Capability | Check type | Reference | Page | Benchmark / record | Tests |
|---|---|---|---|---|---|
| Incompressible flow, scalar transport, Boussinesq convection, inductionless MHD | manufactured solutions, closed form, cross-code, conservation | manufactured solutions; exact Poiseuille, Couette, Hartmann and Womersley flows; NGSolve and FEniCSx reference solutions | [fluids.md](fluids.md) | `benchmarks/fluids/run.py` (case definitions in `benchmarks/fluids/`) | `tests/test_fluids_*.py` |
| Time integrators, kinetics, reactors, current distributions, porous electrodes | closed form, manufactured solutions, cross-code | manufactured problems; closed forms of reactors and electrode geometries; Cantera; DOLFINx and a series solution of the disk electrode | [chemistry.md](chemistry.md) | `benchmarks/chem/run.py` | `tests/test_chem_*.py` |

## Interoperability

| Capability | Check type | Reference | Page | Benchmark / record | Tests |
|---|---|---|---|---|---|
| COMSOL import (beta) | independent code | COMSOL's own solutions of the imported models (the COMSOL files are not distributed; the tests that need them skip) | [comsol_import.md](../comsol_import.md) | `benchmarks/comsol_import/compare_import.py`, `benchmarks/comsol_import/axionfem_results/*.json` | `tests/test_comsol_import.py`, `tests/test_comsol_import2d.py` |
| COMSOL export | regression | round trips of the written mesh and model files | [comsol_export.md](../comsol_export.md) | `examples/comsol_physics/axionfem_reference.json` | `tests/test_comsol_export.py` |
