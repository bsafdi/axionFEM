# About

Lumped-element detectors, cavity haloscopes and SRF heterodyne cavities use the axion–photon coupling to search for axion dark matter. Here we outline the goals of axionFEM and provide citation and contact information.

axionFEM is introduced in

> J. N. Benabou, J. W. Foster, B. R. Safdi, to appear (2026).

## Goal

The package solves the coupled axion–electromagnetic equations of motion for laboratory
experiments that use the axion–photon coupling, and turns the solutions into sensitivity
estimates and experimental requirements. Its main focus is lumped-element detectors based on
the ABRACADABRA concept (toroidal, solenoidal and CORE geometries); it also covers resonant
cavity haloscopes, SRF heterodyne cavities and SQUID readout chains, so that different
approaches can be compared on the same footing.

## Scope

Several modules are beta: they work and are tested, but their interfaces or their validation are
not complete. They are the cavity-haloscope, SRF and dc SQUID-chain modules, multi-band and
multipole scans, torsional dynamics, the COMSOL import, the fluids, scalar-transport and chemistry
modules, the surface-resistance models, and the 3D full-wave features other than current ports.
The 3D engine and the fluids module need the MPI source build. Adjoint shape sensitivities
cover axisymmetric magnetoquasistatic designs with linear materials. The cost metric compares
designs; it is not a construction estimate.

## Citing axionFEM

We ask that the paper above be cited in work that uses axionFEM. The software itself can be cited
with the metadata in [CITATION.cff](https://github.com/bsafdi/axionFEM/blob/main/CITATION.cff):

```bibtex
@software{axionFEM,
  author  = {Benabou, Joshua N. and Foster, Joshua W. and Safdi, Benjamin R.},
  title   = {axionFEM},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/bsafdi/axionFEM}
}
```

Please also cite the methods and benchmark papers your work relies on and [MFEM](https://mfem.org), on which the
solvers are built.

## How axionFEM was made

axionFEM was developed by its authors with Claude (Anthropic) as an AI coding agent in Claude Code, mostly Claude Opus
5.5; most commits in its development history carry a Claude Opus 5.5 co-author trailer. The correctness of the code
rests on its checks, not on how it was written: each validated capability is checked against closed-form solutions,
manufactured solutions, independent solvers or published results, most of those checks run in the test suite, and
modules with less validation evidence are labelled beta.
The [validation pages](validation/index.md) document the checks and their tolerances, so that users can judge the
results independently of how the code was produced.

## Contact

Questions, bug reports and feature requests can be submitted as issues on
[GitHub](https://github.com/bsafdi/axionFEM/issues).

## Contributors

- Joshua N. Benabou
- Joshua W. Foster
- Benjamin R. Safdi
