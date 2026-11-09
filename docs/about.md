# About axionFEM

axionFEM is a finite-element package for designing and evaluating axion dark-matter detectors that use the
axion-photon coupling: lumped-element detectors, cavity haloscopes and SRF heterodyne cavities, together with their
readout. It solves the electromagnetic problems that define each detector type, turns the solutions into the
figures of merit, noise, scan rates and reach of an experiment, and adds the mechanics and heat conduction of the
structure. Designs are JSON documents, so different detector concepts can be compared on the same footing, from a
Python session, the command line or an AI assistant through the MCP server.

## Authors

Joshua N. Benabou, Joshua W. Foster, Benjamin R. Safdi

## Citing axionFEM

If you use axionFEM, please cite

> J. N. Benabou, J. W. Foster, B. R. Safdi, to appear (2026).

and the software itself; the repository's
[CITATION.cff](https://github.com/bsafdi/axionFEM/blob/main/CITATION.cff) has its metadata, and GitHub offers it in
several formats ("Cite this repository"). A BibTeX entry for the software:

```bibtex
@software{axionFEM,
  author  = {Benabou, Joshua N. and Foster, Joshua W. and Safdi, Benjamin R.},
  title   = {axionFEM},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/bsafdi/axionFEM}
}
```

Please also cite the methods and benchmark papers your work relies on (each physics and validation page gives its own
sources; [References](references.md) collects the main ones) and [MFEM](https://mfem.org), on which the solvers are built.

## How axionFEM was made

axionFEM was developed by its authors with Claude (Anthropic) as an AI coding agent in Claude Code, mostly Claude Opus
5.5; most commits in its development history carry a Claude Opus 5.5 co-author trailer. The correctness of the code
rests on its checks, not on how it was written: each validated capability is checked against closed-form solutions,
manufactured solutions, independent solvers or published results, most of those checks run in the test suite, and
modules with less validation evidence are labelled beta.
The [validation pages](validation/index.md) document the checks and their tolerances, so that users can judge the
results independently of how the code was produced.

## Built on

axionFEM links only permissively licensed libraries: [MFEM](https://mfem.org) (finite elements),
[hypre](https://github.com/hypre-space/hypre) (algebraic multigrid, in the 3D engine),
[METIS](https://github.com/KarypisLab/METIS) (orderings), [Eigen](https://eigen.tuxfamily.org) (sparse direct
solvers), [nlohmann/json](https://github.com/nlohmann/json), [nanobind](https://github.com/wjakob/nanobind) (Python
bindings), [three.js](https://threejs.org) (the viewer) and, optionally, [STRUMPACK](https://github.com/pghysels/STRUMPACK)
(sparse direct solvers). [gmsh](https://gmsh.info) generates the meshes; it is GPL-licensed and runs as a separate
program, `axf-mesher`, and is not redistributed in the wheels.

## License

axionFEM is distributed under the BSD 3-Clause license
([LICENSE](https://github.com/bsafdi/axionFEM/blob/main/LICENSE)).
[Build options and packaging](install.md#licensing-of-each-wheel) lists the licence of every file in a wheel.

## Contact

Questions, bug reports and feature requests are welcome as issues on
[GitHub](https://github.com/bsafdi/axionFEM/issues). [Contributing](developer/contributing.md) describes how to
propose changes.
