# Reference data and generators of the fluids verification cases

Everything here is **optional tooling**: the verification cases (`benchmarks/fluids/`) only read the data files in
`out/` and the reference values written into the case definitions. The scripts document how those numbers were
produced and regenerate them; they need packages that axionFEM itself does not use (SymPy, mpmath, NGSolve,
DOLFINx/FEniCSx) and are not run by the test suite. File names and docstrings use the case-group ids (F1a, F1b, ...,
F13a) defined in `benchmarks/fluids/README.md`.

## Manufactured and closed-form fields (`out/`, SymPy)

`out/*.json` are the exact solutions and forcing terms of the manufactured-solution (MMS) and closed-form cases, as
NumPy expressions with metadata. Each case names its files with their sha256, and the runners refuse a file whose
hash differs, so a regenerated file must be committed together with the updated hashes of the case definitions.

| generator | fields |
|---|---|
| `generate.py` (`stokes2d.py`, `f1a_fields.py`) | planar Stokes (`sq_*`, `couette_*`, `poiseuille_*`, `per_*`, `rot_slip`, `annulus_*`, `noflow`, ...) |
| `generate_f1b.py` (`axi_stokes.py`, `f1b_fields.py`) | axisymmetric Stokes with swirl (`out/f1b/`) |
| `generate_f1c.py` (`stokes3d.py`, `f1c_fields.py`) | 3D Stokes (`f1c_*`) |
| `generate_f2a.py` (`f2a_fields.py`) | steady Navier-Stokes (`ns_*`) |
| `generate_f3a.py` (`f3a_fields.py`, `f3a_spec.py`) | transient Navier-Stokes (`f3a_tr_*`) |
| `generate_f4a.py` (`f4a_fields.py`) | scalar transport, Graetz problems, Boussinesq, hydrostatics (`heat_*`, `graetz_*`, `flux_*`, `bous_*`, `hydro_*`, `cond_annulus`) |
| `f5_fields.py` | stabilized P1/P1 (`gls_*`) |
| `f13a_fields.py`, `f13a_annulus_fields.py`, `f13a_suction_fields.py` | inductionless MHD (`mhd_*`) |

Every generator runs its own self-checks (symbolic residuals, finite-difference residuals, boundary identities)
before it writes a file.

## Series and closed-form reference values (mpmath, `decimal`)

| script | output |
|---|---|
| `duct_series_reference.py` | `duct/duct_series_reference.json`: Poiseuille flow in a rectangular duct (3D Stokes) |
| `f13a_reference.py` | `f13a_reference/analytic_reference.json`: Hartmann, Shercliff and Hunt flows |
| (closed forms in the check modules) | `f13a_reference/mhd_annulus_reference.json`, `mhd_suction_reference.json` |

## Cross-code references (`xcode/`, NGSolve and DOLFINx/FEniCSx)

The `cross_code` cases compare axionFEM with another finite-element code solving the **same discrete problem**
(same meshes, elements, quadrature and boundary data), so agreement is expected to round-off or to the stated
tolerance. The reference values are in the case definitions; the scripts and their outputs are kept here:

| script (output `*.json` next to it) | case |
|---|---|
| `ngsolve_cavity_reference.py` | `xcode_stokes_cavity_reg_2d` (planar Stokes cavity) |
| `ngsolve_axi_cavity_reference.py` | `xcode_stokes_axi_cavity_swirl` (axisymmetric cavity with swirl) |
| `fenicsx_cavity3d_reference.py`, `ngsolve_cavity3d_reference.py` | `xcode_stokes_cavity_reg_3d` (two codes) |
| `ngsolve_ns_cavity_reference.py`, `dolfinx_ns_cavity_check.py` | `xcode_cavity_re1000` (Navier-Stokes cavity, Re = 1000) |
| `ngsolve_tr_cavity_reference.py`, `dolfinx_tr.py` | `xcode_ns_transient_cavity` (transient cavity) |
| `ngsolve_gls_cavity_reference.py` | `xcode_gls_cavity` (stabilized P1/P1) |
| `ngsolve_th_cavity_reference.py` | `xcode_gls_vs_taylor_hood` (Taylor-Hood reference for the GLS cavity) |
| `ngsolve_natconv_reference.py`, `dolfinx_natconv_check.py` | `xcode_natconv_ra1e5` (natural convection, two codes) |
| `ngsolve_mhd_duct_reference.py`, `fenicsx_mhd_duct_check.py`, `compare_mhd_duct.py` | `xcode_mhd_duct_2d` (MHD ducts, two codes) |

The NGSolve solvers `ngs_ns.py` (steady Navier-Stokes), `ngs_tr.py` (transient Navier-Stokes), `ngs_heat.py` (scalar
transport and Boussinesq), `ngs_gls.py` (stabilized P1/P1) and `ngs_mhd.py` (inductionless MHD) are shared by these
scripts. Each script prints its usage in its docstring and writes the JSON file named on its command line; the
NGSolve scripts were run with NGSolve 6.2.2608 and the DOLFINx/FEniCSx scripts with DOLFINx 0.11.0.
