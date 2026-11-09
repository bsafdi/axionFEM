# Third-party licences

axionFEM itself is BSD-3-Clause (`LICENSE` at the top of the repository). A wheel, and any build that bundles its
dependencies (`AXF_BUNDLE_LIBS=ON`, the default under scikit-build-core), also contains third-party code; this
directory holds the licence texts that its redistribution requires, and `pyproject.toml` (`wheel.license-files`) puts
them into every wheel. The texts are copied unmodified: from the pinned upstream releases, and for stb
(`stb-LICENSE.txt`) from the licence section at the end of the vendored header `cpp/third_party/stb_image_write.h`.

| component | where it is in a wheel | licence | file(s) |
|---|---|---|---|
| MFEM 4.10 | `axionfem/_libs/libmfem.*` | BSD-3-Clause | `MFEM-LICENSE.txt`, `MFEM-NOTICE.txt` |
| METIS 5.1 | `axionfem/_libs/libmetis.*` | Apache-2.0 | `METIS-LICENSE.txt`, `Apache-2.0.txt` |
| Eigen 3.4 (headers) | compiled into `_core`, `libaxionfem`, `axf` | MPL-2.0 (unmodified; source at https://eigen.tuxfamily.org) | `Eigen-MPL-2.0.txt` |
| nlohmann/json 3.12 (header) | compiled into `_core`, `libaxionfem`, `axf` | MIT | `nlohmann-json-LICENSE.txt` |
| nanobind | compiled into `_core` | BSD-3-Clause | `nanobind-LICENSE.txt` |
| stb_image_write (`cpp/third_party/`) | compiled into `libaxionfem` (PNG output) | MIT or public domain | `stb-LICENSE.txt` |
| three.js r185 (`viewer/vendor/`) | `axionfem/_viewer/vendor/` | MIT | `viewer/vendor/LICENSE-three.txt` |
| hypre 2.33 | `axionfem/_libs/libHYPRE.*`, only in a build against an MPI stack | Apache-2.0 or MIT | `hypre-COPYRIGHT.txt`, `hypre-LICENSE-MIT.txt`, `hypre-NOTICE.txt`, `Apache-2.0.txt` |

The wheels of the sequential configuration contain MFEM, METIS and the compiled-in headers, not hypre. A build that
bundles STRUMPACK (`libstrumpack.*`, only when the dependency prefix has it; the wheels do not) must add STRUMPACK's
own `LICENSE` file from its source release to what it redistributes. gmsh (GPL) is not in any wheel: the mesher loads
it from gmsh's own package (docs/install.md, "Licensing of each wheel").
