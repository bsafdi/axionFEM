"""Tree-cotree gauged full-wave solve (AXF_FW_GAUGE=tree-cotree) against the default AMS path.

  python benchmarks/tree_cotree_validation.py OUTDIR [NRANKS]
  python benchmarks/tree_cotree_validation.py OUTDIR --report-only     # tables from the saved record

The two-port copper frame (benchmarks/designs/stage1_twoport.json: lowest-order Nedelec, closed PEC
sphere, two gap ports, uniform source, source check, shunt-loaded validation) is solved at seven frequencies
from 10 Hz to 12 MHz with seven solver configurations (the gauged ones with each charge projection and refinement rule):

  ams          AXF_FW_SOLVER=ams        (default: AMS-preconditioned GMRES, summed matrix K - w^2 M)
  strumpack    AXF_FW_SOLVER=strumpack  (root-owned LU of the summed matrix, iterative refinement)
  tree_cotree  AXF_FW_SOLVER=strumpack AXF_FW_GAUGE=tree-cotree  (E = S_c e + G_c phi, K G_c = 0 exact)
  tree_cotree                    with AXF_FW_TC_CHARGE_PROJECTION=0 (no projection)
  tree_cotree_charge_projection  ... =1 (the free-vertex rows' rounding charge removed)
  tree_cotree_charge_projection2 ... =2 (also the conductor rows'; the default)
  tree_cotree_componentwise[_projection2]  AXF_FW_TC_REFINE=componentwise, projection 0 or 2

At 100 Hz the frame is 3e-7 wavelengths across. There is no residual floor (the gauged field's residual
under the summed matrix is 4e-13); what grows at low frequency is the ungauged solves' forward error, of
order (k h_min)^-2 times the residual, which residual gates do not see. The gauged path never forms the
difference K - w^2 M on gradients. The tables are a cross-check of the linear solves, plus
the electric energy's f^2 scaling below 1 kHz. The record keeps every sweep row of every configuration
(Z matrix, Psi vector, residuals, source/load validation errors, the gauged path's own diagnostics), and the
rejected rows' diagnostics, so that the comparison can be redone.

Every frequency is a separate solve. Writes OUTDIR/tree_cotree_validation.json, benchmarks/records/tree_cotree_validation.json and
docs/validation/tree_cotree.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "python"))   # after PYTHONPATH, so that a package copy (runs/pkgN) can take precedence

import axionfem as af  # noqa: E402

DESIGN = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
FREQUENCIES = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1.2e7]
FIT_HZ = (1e4, 1e5)   # U_e = f^2 c (1 + beta f^2) through these (every path agrees there) ...
TEST_MAX_HZ = 1e3     # ... and the values up to here must match it
CRITERION = 1e-8
_DEFAULTS = {"AXF_FW_TC_CHARGE_PROJECTION": "0", "AXF_FW_TC_REFINE": "normwise",   # set in every config (os.environ
             "AXF_FW_TC_RESIDUAL": ""}
_GAUGED = {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree", **_DEFAULTS}  # carries over between them)
CONFIGS = {
    "ams": {"AXF_FW_SOLVER": "ams", "AXF_FW_GAUGE": "none", **_DEFAULTS},
    "strumpack": {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "none", "AXF_FW_STRUMPACK_DIST": "0", **_DEFAULTS},
    "tree_cotree": _GAUGED,
    # experimental, the candidate causes of the low-frequency error:
    # the right-hand side's free-vertex Gauss rows set to zero (1), and also its uncharged conductor rows (2); and the
    # refinement stopped on the componentwise backward error instead of the normwise residual
    "tree_cotree_charge_projection": {**_GAUGED, "AXF_FW_TC_CHARGE_PROJECTION": "1"},
    "tree_cotree_charge_projection2": {**_GAUGED, "AXF_FW_TC_CHARGE_PROJECTION": "2"},
    # componentwise stopping needs the double residual (with the extended one the loop stops on the correction)
    "tree_cotree_componentwise": {**_GAUGED, "AXF_FW_TC_REFINE": "componentwise", "AXF_FW_TC_RESIDUAL": "double"},
    "tree_cotree_componentwise_projection2": {**_GAUGED, "AXF_FW_TC_REFINE": "componentwise",
                                              "AXF_FW_TC_CHARGE_PROJECTION": "2", "AXF_FW_TC_RESIDUAL": "double"},
}
ROW_KEYS = ("reciprocity_relative_error", "source_reciprocity_relative_error", "loaded_closure_relative_error",
            "loaded_field_closure_relative_error", "validation_accepted", "frequency_total_seconds",
            "direct_factorization", "solver")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout
    except Exception:  # not a git checkout, or git unavailable
        return ""


def file_record(path_str: str) -> dict:
    path = Path(path_str) if path_str else None
    if not path or not path.exists():
        return {"path": Path(path_str).name if path_str else path_str}
    return {"path": path.name, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "sha256": sha256(path)}


def provenance(nranks: int) -> dict:
    """The code that solved: the commit, the uncommitted diff's hash, and the executables. axf
    differs between build directories only in its RPATH; the solver is libaxionfem.so next to it."""
    diff = git("diff", "HEAD")
    exe = os.environ.get("AXF_EXE", "")
    return {"git_head": git("rev-parse", "HEAD").strip(), "git_describe": git("describe", "--always", "--dirty").strip(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "AXF_EXE": file_record(exe), "libaxionfem": file_record(str(Path(exe).parent / "libaxionfem.so") if exe else ""),
            "AXF_MESHER": file_record(os.environ.get("AXF_MESHER", "")),
            "python_core": file_record(af._core.__file__), "python_core_build_info": {k: v for k, v in af.build_info().items() if k != "library"},
            "driver": file_record(__file__),
            "design": str(DESIGN.relative_to(ROOT)), "design_sha256": sha256(DESIGN), "frequencies_Hz": FREQUENCIES,
            "mpi_ranks": nranks, "date": time.strftime("%Y-%m-%d")}


def design_doc(frequency: float) -> dict:
    """The two-port design with a single frequency, so that one failing frequency cannot abort the others."""
    doc = json.loads(DESIGN.read_text())
    doc["fullwave"]["frequencies"] = [frequency]
    return doc


def complex_entry(row: dict, key_re: str, key_im: str, i: int, j: int | None = None) -> complex:
    re = row[key_re]
    im = row[key_im]
    if j is None:
        return complex(re[i], im[i])
    return complex(re[i][j], im[i][j])


def run_config(name: str, env: dict, wd: Path, nranks: int) -> dict:
    """Solve every frequency separately with one solver configuration; failures are recorded per frequency."""
    for key, value in env.items():
        os.environ[key] = value
    t0 = time.time()
    out = {"config": name, "env": dict(env), "workdir": str(wd), "sweep": [], "errors": {}, "rejected": {},
           "warnings": []}
    for f in FREQUENCIES:
        design = af.Design(design_doc(f))
        fdir = wd / f"f_{f:.0f}Hz"
        t1 = time.time()
        try:
            res = design.solve(fdir, verbose=0, mpi_ranks=nranks)
        except Exception as ex:  # keep going with the other frequencies and configurations
            out["errors"][str(f)] = f"{type(ex).__name__}: {ex}"
            rejected = fdir / "fullwave_rejected.json"   # the refused row's diagnostics
            if rejected.exists():
                row = json.loads(rejected.read_text())["primary_row"]
                row.pop("field_basis_files", None)
                row["wall_seconds"] = time.time() - t1
                row["mesh_sha256"] = sha256(fdir / "mesh3d.msh")
                out["rejected"][str(f)] = row
            print(f"   {f:.3g} Hz failed after {time.time() - t1:.0f} s: {ex}", flush=True)
            continue
        fw = res.data["fullwave"]
        out.setdefault("dofs", res.data["mesh"].get("dofs"))
        out.setdefault("mesh", {k: v for k, v in res.data["mesh"].items() if isinstance(v, (int, float, str))})
        out.setdefault("solver_path", fw.get("solver_path"))
        out.setdefault("solver_settings", fw.get("solver_settings"))
        out.setdefault("tree_cotree_topology", fw.get("tree_cotree"))
        for w in res.data.get("warnings", []):
            if w not in out["warnings"]:
                out["warnings"].append(w)
        row = dict(fw["sweep"][0])
        row["wall_seconds"] = time.time() - t1
        row["mesh_sha256"] = sha256(fdir / "mesh3d.msh")
        out["sweep"].append(row)
        print(f"   {f:.3g} Hz done in {time.time() - t1:.0f} s", flush=True)
    out["seconds"] = time.time() - t0
    if out["errors"] and not out["sweep"]:
        out["error"] = "every frequency failed: " + next(iter(out["errors"].values()))
    return out


def gauged_solves(row: dict) -> dict:
    """The gauged path's statistics of every solve in a sweep row: each port, then each validation solve."""
    out = {}
    for port in row.get("port_solutions", []):
        tc = (port.get("solver") or {}).get("tree_cotree")
        if tc:
            out[f"port_{port['port_index']}"] = tc
    for key, value in row.items():
        if key.endswith("_solver") and isinstance(value, dict) and value.get("tree_cotree"):
            out[key] = value["tree_cotree"]
    return out


def solver_diagnostics(row: dict) -> dict:
    """The solver's own residual and iteration counts, and the gauged path's diagnostics, from one sweep row. The
    charge diagnostics are the largest over every port and validation solve."""
    solver = row.get("solver") or {}
    tc = solver.get("tree_cotree") or {}
    fact = row.get("direct_factorization") or {}
    solves = gauged_solves(row)

    def largest(key):
        values = [s.get(key) for s in solves.values() if s.get(key) is not None]
        return max(values) if values else None

    return {"relative_residual": solver.get("relative_residual"),
            "gmres_iterations": solver.get("gmres_iterations"),
            "accepted": solver.get("accepted"),
            "legacy_summed_matrix_relative_residual_diagnostic": tc.get("legacy_summed_matrix_relative_residual_diagnostic"),
            "split_true_relative_residual": tc.get("split_true_relative_residual"),
            "curl_block_relative_residual": tc.get("curl_block_relative_residual"),
            "gauss_block_relative_residual": tc.get("gauss_block_relative_residual"),
            "rhs_free_vertex_charge_relative": largest("rhs_free_vertex_charge_relative"),
            "rhs_free_vertex_charge_relative_solved": largest("rhs_free_vertex_charge_relative_solved"),
            "rhs_uncharged_conductor_charge_relative": largest("rhs_uncharged_conductor_charge_relative"),
            "rhs_uncharged_conductor_charge_relative_solved": largest("rhs_uncharged_conductor_charge_relative_solved"),
            "refinement_steps_max": largest("refinement_steps"),
            "componentwise_backward_error": {
                block: max((s["componentwise_backward_error"][block] for s in solves.values()
                            if s.get("componentwise_backward_error")), default=None)
                for block in ("curl_rows", "free_vertex_gauss_rows", "conductor_gauss_rows")},
            # per port: the field the right-hand side's own charge and the final
            # residual's Gauss rows would add, against the port's field
            "port_rhs_charge_field_fraction": [solves[f"port_{j}"].get("rhs_charge_field_fraction")
                                               for j in range(2) if f"port_{j}" in solves],
            "port_gauss_residual_field_fraction": [solves[f"port_{j}"].get("gauss_residual_field_fraction")
                                                   for j in range(2) if f"port_{j}" in solves],
            "gauged_solves": len(solves),
            "gradient_block_check": fact.get("gradient_block_check"),
            "factor_nonzeros": (fact.get("factor") or {}).get("factor_nonzeros"),
            "factor_seconds": (fact.get("factor") or {}).get("seconds")}


def energies(row: dict, f: float) -> dict:
    """Each port's own electric energy over f^2 and magnetic energy, per unit current (diagonal of the matrices)."""
    out = {}
    ue = (row.get("energy_electric_J_per_A2") or {}).get("re")
    um = (row.get("energy_magnetic_J_per_A2") or {}).get("re")
    if ue:
        out["U_e_over_f2"] = [ue[i][i] / f ** 2 for i in range(len(ue))]
    if um:
        out["U_m"] = [um[i][i] for i in range(len(um))]
    return out


def energy_protocol(record: dict) -> dict:
    """The low-frequency energy test: per configuration and port,
    U_e = f^2 c (1 + beta f^2) through the values at FIT_HZ (where every path agrees), and every value up to TEST_MAX_HZ
    against it. The deviation d must be <= CRITERION, and so must that of U_e22/U_e11. A spurious low-frequency field
    is a discrete gradient, orthogonal to the charge-free port field, so it adds its own energy: the field error is
    sqrt(d)."""
    out = {}
    f1, f2 = FIT_HZ
    for name, cfg in record["configs"].items():
        rows = {r["f_Hz"]: r for r in cfg.get("sweep", [])}
        if f1 not in rows or f2 not in rows:
            out[name] = {"note": "a fit frequency was refused"}
            continue
        u = {f: energies(r, f)["U_e_over_f2"] for f, r in rows.items()}
        fits = []
        for i in range(2):
            beta = (u[f2][i] - u[f1][i]) / (u[f1][i] * f2 ** 2 - u[f2][i] * f1 ** 2)
            fits.append((u[f1][i] / (1 + beta * f1 ** 2), beta))
        tests = []
        for f in sorted(rows):
            if f > TEST_MAX_HZ:
                continue
            model = [c * (1 + beta * f ** 2) for c, beta in fits]
            dev = [u[f][i] / model[i] - 1 for i in range(2)]
            ratio = (u[f][1] / u[f][0]) / (model[1] / model[0]) - 1
            tests.append({"f_Hz": f, "U_e_deviation": dev, "field_error": [math.sqrt(abs(d)) for d in dev],
                          "U_e22_over_U_e11_deviation": ratio,
                          "passes": max(abs(d) for d in dev + [ratio]) <= CRITERION})
        out[name] = {"fit_Hz": list(FIT_HZ), "criterion": CRITERION,
                     "ports": [{"c_J_per_A2_Hz2": c, "beta_per_Hz2": beta} for c, beta in fits], "tests": tests}
    return out


def compare(record: dict) -> list[dict]:
    """Per frequency: each configuration's own gates and its deviation from the AMS reference."""
    ref_rows = {}
    ams = record["configs"].get("ams", {})
    for row in ams.get("sweep", []):
        ref_rows[row["f_Hz"]] = row
    table = []
    for f in FREQUENCIES:
        entry = {"f_Hz": f}
        ref = ref_rows.get(f)
        for name, cfg in record["configs"].items():
            rows = [r for r in cfg.get("sweep", []) if r["f_Hz"] == f]
            if not rows:
                entry[name] = {"missing": cfg.get("errors", {}).get(str(f), cfg.get("error", "no row"))}
                continue
            row = rows[0]
            item = {k: row.get(k) for k in ROW_KEYS if k in row}
            item.update(solver_diagnostics(row))
            item["wall_seconds"] = row.get("wall_seconds")
            item["Z11_ohm"] = [row["Z_matrix_re_ohm"][0][0], row["Z_matrix_im_ohm"][0][0]]
            item["Z12_ohm"] = [row["Z_matrix_re_ohm"][0][1], row["Z_matrix_im_ohm"][0][1]]
            item["Z22_ohm"] = [row["Z_matrix_re_ohm"][1][1], row["Z_matrix_im_ohm"][1][1]]
            item["L_H"] = row.get("L_H")
            item["Psi_abs"] = [abs(complex_entry(row["Psi_vector"], "re", "im", i)) for i in range(2)]
            item.update(energies(row, f))
            if ref is not None and name != "ams":
                item["vs_ams"] = {}
                for key, (i, j) in (("Z11", (0, 0)), ("Z12", (0, 1)), ("Z22", (1, 1))):
                    z = complex_entry(row, "Z_matrix_re_ohm", "Z_matrix_im_ohm", i, j)
                    z0 = complex_entry(ref, "Z_matrix_re_ohm", "Z_matrix_im_ohm", i, j)
                    item["vs_ams"][key] = abs(z - z0) / abs(z0) if z0 != 0 else None
                for i in range(2):
                    p = complex_entry(row["Psi_vector"], "re", "im", i)
                    p0 = complex_entry(ref["Psi_vector"], "re", "im", i)
                    item["vs_ams"][f"Psi{i + 1}"] = abs(p - p0) / abs(p0) if p0 != 0 else None
                if row.get("L_H") and ref.get("L_H"):
                    item["vs_ams"]["L"] = abs(row["L_H"] - ref["L_H"]) / abs(ref["L_H"])
                for key in ("U_e_over_f2", "U_m"):
                    if key in item and key in energies(ref, f):
                        item["vs_ams"][key] = [abs(a / b - 1) if b else None
                                               for a, b in zip(item[key], energies(ref, f)[key])]
            entry[name] = item
        table.append(entry)
    return table


def fmt(x, spec="{:.3e}") -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return spec.format(x)
    return str(x)


def fhz(f: float) -> str:
    if f >= 1e6:
        return f"{f / 1e6:g} MHz"
    if f >= 1e3:
        return f"{f / 1e3:g} kHz"
    return f"{f:g} Hz"


def report(record: dict) -> str:
    cfgs = record["configs"]
    prov = record["provenance"]
    L = ["# Tree-cotree gauged full-wave solve vs the AMS path", "",
         '!!! warning "Beta"', "    The tree-cotree gauged solve is beta.", "",
         "Generated by `benchmarks/tree_cotree_validation.py`; record `benchmarks/records/tree_cotree_validation.json`.",
         f"Design `{prov['design']}`; {prov['mpi_ranks']} MPI ranks. The record carries the commit, the design and "
         "executable hashes and the solver environment of every run.", "",
         "Agreement between the configurations tests the linear algebra of one discrete system, not accuracy. Read the",
         "tables as a cross-check of the linear solves of the same discrete system, plus the low-frequency scaling of the",
         "energies. The independent low-frequency references are a capacitor against its closed form",
         "(`benchmarks/tree_cotree_capacitor.py`, record `benchmarks/records/tree_cotree_capacitor.json`) and the MQS limit of",
         "L (`benchmarks/tree_cotree_mqs_limit.py`, record `benchmarks/records/tree_cotree_mqs_limit.json`); see",
         "[fullwave_tree_cotree.md](../physics/fullwave_tree_cotree.md). The ungauged configurations refuse the lowest",
         "frequencies here because validation loads are configured; their refused rows are kept (section 'Refused rows').",
         "",
         "The two-port copper frame (0.6 m square, lowest-order Nédélec, closed PEC sphere of 0.7 m, two",
         "gap ports, uniform source B0 = (1, 0.2, 0.3) T, source check and shunt-loaded validation) is solved at",
         f"{len(prov.get('frequencies_Hz', FREQUENCIES))} frequencies with {len(cfgs)} solver configurations. `ams` is",
         "the default path and the reference; `strumpack` factorizes the summed matrix K − ω²M; `tree_cotree`",
         "factorizes the gauged operator (E = S_c e + G_c φ, one potential per PEC-connected conductor, K G_c = 0",
         "used exactly) with no charge projection; the other gauged configurations zero the right-hand side's",
         "free-vertex Gauss rows (`AXF_FW_TC_CHARGE_PROJECTION=1`) or also its uncharged conductor rows (`=2`,",
         "the default), and stop the refinement on the componentwise backward error",
         "(`AXF_FW_TC_REFINE=componentwise`).",
         "The frame is 3e-7 wavelengths across at 100 Hz; there the ungauged solves' forward error grows like",
         "(k h_min)^-2 times their residual, which a residual gate does not detect.", "",
         "Every frequency is a separate solve (one mesh generation each), so that a frequency at which a path",
         "fails its gates does not abort the others; failures are listed per frequency.", "",
         "## Runs", "", "| configuration | environment | DOFs | wall, all frequencies [s] | solver path | failed frequencies |",
         "|---|---|---:|---:|---|---|"]
    for name, cfg in cfgs.items():
        env = " ".join(f"{k}={v}" for k, v in cfg["env"].items())
        failed = "; ".join(f"{fhz(float(f))}: {e}" for f, e in cfg.get("errors", {}).items()) or "none"
        L.append(f"| {name} | `{env}` | {fmt(cfg.get('dofs'))} | {fmt(cfg.get('seconds'), '{:.0f}')} | "
                 f"{fmt(cfg.get('solver_path'))} | {failed} |")
    topo = cfgs.get("tree_cotree", {}).get("tree_cotree_topology")
    if topo:
        L += ["", f"Tree-cotree topology: `{json.dumps(topo)}`."]
    L += ["", "## Deviation from the AMS reference", "",
          "Relative differences |x − x_ams| / |x_ams| of the impedance matrix, the source overlaps Ψ₁, Ψ₂ and the",
          "pickup inductance L. A missing AMS row leaves the column empty.", "",
          "| f | config | Z11 | Z12 | Z22 | Ψ1 | Ψ2 | L |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for entry in record["comparison"]:
        for name in ("strumpack", "tree_cotree"):
            item = entry.get(name, {})
            d = item.get("vs_ams", {})
            L.append(f"| {fhz(entry['f_Hz'])} | {name} | {fmt(d.get('Z11'))} | {fmt(d.get('Z12'))} | {fmt(d.get('Z22'))} "
                     f"| {fmt(d.get('Psi1'))} | {fmt(d.get('Psi2'))} | {fmt(d.get('L'))} |")
    L += ["", "## Deviation from the gauged path", "",
          "The same differences taken against `tree_cotree`, which passes its gates at every frequency, so that the",
          "frequencies where the AMS reference refuses are covered too.", "",
          "| f | config | Z11 | Z12 | Z22 | Ψ1 | Ψ2 | L |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    gauged = {r["f_Hz"]: r for r in cfgs.get("tree_cotree", {}).get("sweep", [])}
    for f in FREQUENCIES:
        g = gauged.get(f)
        for name in ("ams", "strumpack"):
            rows = [r for r in cfgs.get(name, {}).get("sweep", []) if r["f_Hz"] == f]
            if g is None or not rows:
                L.append(f"| {fhz(f)} | {name} | refused | | | | | |")
                continue
            row = rows[0]
            d = {}
            for key, (i, j) in (("Z11", (0, 0)), ("Z12", (0, 1)), ("Z22", (1, 1))):
                z = complex_entry(row, "Z_matrix_re_ohm", "Z_matrix_im_ohm", i, j)
                z0 = complex_entry(g, "Z_matrix_re_ohm", "Z_matrix_im_ohm", i, j)
                d[key] = abs(z - z0) / abs(z0)
            for i in range(2):
                p = complex_entry(row["Psi_vector"], "re", "im", i)
                p0 = complex_entry(g["Psi_vector"], "re", "im", i)
                d[f"Psi{i + 1}"] = abs(p - p0) / abs(p0)
            d["L"] = abs(row["L_H"] - g["L_H"]) / abs(g["L_H"])
            L.append(f"| {fhz(f)} | {name} | {fmt(d['Z11'])} | {fmt(d['Z12'])} | {fmt(d['Z22'])} | "
                     f"{fmt(d['Psi1'])} | {fmt(d['Psi2'])} | {fmt(d['L'])} |")
    L += ["", "## Low-frequency energies", "",
          "Each port's own electric energy over f² and magnetic energy per unit current (diagonal entries). In the",
          "MQS limit U_e/f² and U_m are constant (U_m changes by the physical f² term, about 5e-6 up to 100 kHz).", "",
          "| f | config | U_e11/f² | U_e22/f² | U_m11 | U_m22 | vs AMS: U_e/f² (1, 2) | vs AMS: U_m (1, 2) |",
          "|---|---|---:|---:|---:|---:|---|---|"]
    for entry in record["comparison"]:
        for name in cfgs:
            item = entry.get(name, {})
            if "missing" in item or "U_e_over_f2" not in item:
                continue
            d = item.get("vs_ams", {})
            ve = ", ".join(fmt(x) for x in d.get("U_e_over_f2", [])) or "–"
            vm = ", ".join(fmt(x) for x in d.get("U_m", [])) or "–"
            L.append(f"| {fhz(entry['f_Hz'])} | {name} | {fmt(item['U_e_over_f2'][0], '{:.10e}')} | "
                     f"{fmt(item['U_e_over_f2'][1], '{:.10e}')} | {fmt(item['U_m'][0], '{:.8e}')} | "
                     f"{fmt(item['U_m'][1], '{:.8e}')} | {ve} | {vm} |")
    proto = record.get("energy_protocol") or energy_protocol(record)
    L += ["", "### Low-frequency energy test", "",
          f"U_e = f² c (1 + β f²) is fixed per port by the values at {fhz(FIT_HZ[0])} and {fhz(FIT_HZ[1])}, where",
          f"every path agrees; the values up to {fhz(TEST_MAX_HZ)} must match it to ≤ {CRITERION:g}, and so must",
          "U_e22/U_e11. A spurious low-frequency field is a discrete gradient, orthogonal to the charge-free port",
          "field, so it adds its own energy: the field error is √d. Field outputs",
          "(probes, exports, surface fields) carry it, and no gate or residual sees it.", "",
          "| config | f | d, port 1 | d, port 2 | field error √d (1, 2) | U_e22/U_e11 | passes |",
          "|---|---|---:|---:|---|---:|---|"]
    for name, p in proto.items():
        if "tests" not in p:
            L.append(f"| {name} | – | {p.get('note')} | | | | |")
            continue
        for t in p["tests"]:
            L.append(f"| {name} | {fhz(t['f_Hz'])} | {fmt(t['U_e_deviation'][0])} | {fmt(t['U_e_deviation'][1])} | "
                     f"{fmt(t['field_error'][0], '{:.1e}')}, {fmt(t['field_error'][1], '{:.1e}')} | "
                     f"{fmt(t['U_e22_over_U_e11_deviation'])} | {'yes' if t['passes'] else '**no**'} |")
    L += ["", "**Findings.**"]
    for name, p in proto.items():
        if "tests" not in p:
            continue
        failed = [t for t in p["tests"] if not t["passes"]]
        if not p["tests"]:
            L.append(f"- `{name}`: refused every frequency up to {fhz(TEST_MAX_HZ)} (section 'Refused rows').")
            continue
        if not failed:
            L.append(f"- `{name}`: passes at every frequency down to {fhz(p['tests'][0]['f_Hz'])}.")
            continue
        worst = max(failed, key=lambda t: max(t["field_error"]))
        L.append(f"- `{name}`: fails at {', '.join(fhz(t['f_Hz']) for t in failed)}; the largest field error is "
                 f"{max(worst['field_error']):.1e} at {fhz(worst['f_Hz'])} (d = {fmt(max(worst['U_e_deviation']))}).")
    q = [solver_diagnostics(r) for r in cfgs.get("tree_cotree", {}).get("sweep", [])]
    qmax = max((x["rhs_free_vertex_charge_relative"] or 0 for x in q), default=None)
    L += ["", "The right-hand side's free-vertex charge (`rhs_free_vertex_charge_relative`: the largest |G_cᵀb| over",
          "free vertices divided by the largest Σ|G_c||b| over the same vertices, a ratio of maxima, over every port",
          f"and validation solve) is at most {fmt(qmax)} in `tree_cotree`: at rounding, yet the excess above follows",
          "the f⁻⁴ that rounding-level charge gives (δφ ∝ q/ω², so d ∝ q²/ω⁴). Projection 1 removes it at free",
          "vertices, projection 2 also in the rows of uncharged conductors (here the frame); the estimates below",
          "attribute the whole excess to that charge."]
    L += ["", "### The solver's own estimate", "",
          "Per port solve the gauged path estimates the field two Gauss-row sources would add, against the port's",
          "field in the energy norm (one more solve with the factor each): the right-hand side's own charge in the",
          "rows the projection treats (`rhs_charge_field_fraction`; with the projection on, the field it removed)",
          "and the Gauss rows of the final residual (`gauss_residual_field_fraction`). Beside them: the measured",
          "field error √d of the energy test, and the largest componentwise backward error of the free-vertex and",
          "conductor Gauss rows over every solve.", "",
          "| config | f | measured √d (1, 2) | charge estimate (1, 2) | residual estimate (1, 2) | backward error, free / conductor rows | refinement steps |",
          "|---|---|---|---|---|---|---:|"]
    for name, p in proto.items():
        if "tests" not in p or not name.startswith("tree_cotree"):
            continue
        rows = {r["f_Hz"]: r for r in cfgs[name].get("sweep", [])}
        for t in p["tests"]:
            diag = solver_diagnostics(rows[t["f_Hz"]])
            if not diag["port_rhs_charge_field_fraction"]:
                continue
            bw = diag["componentwise_backward_error"]
            L.append(f"| {name} | {fhz(t['f_Hz'])} | "
                     f"{', '.join(fmt(x, '{:.1e}') for x in t['field_error'])} | "
                     f"{', '.join(fmt(x, '{:.1e}') for x in diag['port_rhs_charge_field_fraction'])} | "
                     f"{', '.join(fmt(x, '{:.1e}') for x in diag['port_gauss_residual_field_fraction'])} | "
                     f"{fmt(bw['free_vertex_gauss_rows'], '{:.1e}')} / {fmt(bw['conductor_gauss_rows'], '{:.1e}')} | "
                     f"{fmt(diag['refinement_steps_max'])} |")
    L += ["", "## Own gates of every configuration", "",
          "`residual` is the true relative residual the solver reports for its primary solve (for the gauged path:",
          "the original-space residual in split form; it covers the port-0 solve). The solver gates the residual at",
          "1e-6 and the source and loaded-closure checks at 1e-5; Z reciprocity is reported but not gated (2e-6 is",
          "the two-port validation's own criterion). The last column is the gauged field's residual under the legacy summed",
          "matrix (about 3e-13 at every frequency: the two systems are equivalent, and there is no residual floor).", "",
          "| f | config | residual | GMRES it. | Z reciprocity | source reciprocity | loaded closure (circuit / field) | legacy-matrix residual (gauged) | wall [s] |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in record["comparison"]:
        for name in cfgs:
            item = entry.get(name, {})
            if "missing" in item:
                L.append(f"| {fhz(entry['f_Hz'])} | {name} | failed: {item['missing']} | | | | | | |")
                continue
            L.append(f"| {fhz(entry['f_Hz'])} | {name} | {fmt(item.get('relative_residual'))} | "
                     f"{fmt(item.get('gmres_iterations'))} | {fmt(item.get('reciprocity_relative_error'))} | "
                     f"{fmt(item.get('source_reciprocity_relative_error'))} | "
                     f"{fmt(item.get('loaded_closure_relative_error'))} / {fmt(item.get('loaded_field_closure_relative_error'))} | "
                     f"{fmt(item.get('legacy_summed_matrix_relative_residual_diagnostic'))} | "
                     f"{fmt(item.get('wall_seconds'), '{:.0f}')} |")
    L += ["", "## Gradient-block check of the gauged path", "",
          "Per frequency the gauged path applies the legacy summed matrix and the separately assembled K to a",
          "random discrete gradient of the reduced system (φ = 0 on every vertex of an essential, PEC, edge). In",
          "exact arithmetic both numbers are 0; what they show is the rounding of K on a gradient against ω²M on it,",
          "which grows exactly like f⁻²: the error the summed-matrix paths make on gradients and the gauged path",
          "avoids. The curl and Gauss block residuals are those of the transformed system, each against its own",
          "rows' size; for a charge-free source the Gauss rows cancel, so that ratio grows with frequency and is not",
          "an accuracy measure.", "",
          "| f | gradient-block check | curl block residual | Gauss block residual |", "|---|---|---:|---:|"]
    for entry in record["comparison"]:
        item = entry.get("tree_cotree", {})
        L.append(f"| {fhz(entry['f_Hz'])} | `{json.dumps(item.get('gradient_block_check'))}` | "
                 f"{fmt(item.get('curl_block_relative_residual'))} | {fmt(item.get('gauss_block_relative_residual'))} |")
    L += ["", "## Absolute values", "", "| f | config | Z11 [Ω] | Z12 [Ω] | Z22 [Ω] | L [H] | |Ψ1| | |Ψ2| |",
          "|---|---|---|---|---|---:|---:|---:|"]
    for entry in record["comparison"]:
        for name in cfgs:
            item = entry.get(name, {})
            if "missing" in item:
                continue
            z = ["{:.6g}{:+.6g}i".format(*item[k]) for k in ("Z11_ohm", "Z12_ohm", "Z22_ohm")]
            L.append(f"| {fhz(entry['f_Hz'])} | {name} | {z[0]} | {z[1]} | {z[2]} | {fmt(item.get('L_H'), '{:.6e}')} | "
                     f"{fmt(item['Psi_abs'][0], '{:.6e}')} | {fmt(item['Psi_abs'][1], '{:.6e}')} |")
    refused = [(name, f, row) for name, cfg in cfgs.items() for f, row in cfg.get("rejected", {}).items()]
    L += ["", "## Refused rows", "",
          "Frequencies a configuration refused, from the solver's retained diagnostics (`fullwave_rejected.json`):",
          "the gate that refused, and the refused row's Z and U_e/f² against the gauged path.",
          ""]
    if not refused:
        L.append("None retained.")
    else:
        L += ["| config | f | gate | reason | Z11 | Z22 | U_e/f² (1, 2) |", "|---|---|---|---|---:|---:|---|"]
        for name, f, row in refused:
            g = gauged.get(float(f))
            rej = row.get("rejection") or {}
            dz = ["–", "–"]
            de = "–"
            if g is not None and row.get("Z_matrix_re_ohm"):
                for k, i in enumerate((0, 1)):
                    z = complex_entry(row, "Z_matrix_re_ohm", "Z_matrix_im_ohm", i, i)
                    z0 = complex_entry(g, "Z_matrix_re_ohm", "Z_matrix_im_ohm", i, i)
                    dz[k] = fmt(abs(z - z0) / abs(z0))
                e, e0 = energies(row, float(f)).get("U_e_over_f2"), energies(g, float(f)).get("U_e_over_f2")
                if e and e0:
                    de = ", ".join(fmt(a / b - 1) for a, b in zip(e, e0))
            L.append(f"| {name} | {fhz(float(f))} | {rej.get('gate', '–')} | {rej.get('reason', '–')} | "
                     f"{dz[0]} | {dz[1]} | {de} |")
    warnings = {name: cfg.get("warnings", []) for name, cfg in cfgs.items() if cfg.get("warnings")}
    L += ["", "## Warnings", "", "```", json.dumps(warnings, indent=1), "```", "",
          "## Scope", "",
          "A single small lowest-order mesh, one geometry and one source. This records the agreement of the three",
          "solver paths and the frequency at which the summed-matrix paths lose accuracy; it is not an h/p study,",
          "and the gauged path's restrictions (lowest-order Nédélec, all-PEC boundary, STRUMPACK) are unchanged.", ""]
    return "\n".join(L)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(2)
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "tree_cotree_validation.json"
    if "--report-only" in sys.argv:
        record = json.loads(record_path.read_text())
    else:
        nranks = int(args[1]) if len(args) > 1 else 1
        record = {"provenance": provenance(nranks), "configs": {}}
        for name, env in CONFIGS.items():
            print(f"== {name}: {env}", flush=True)
            record["configs"][name] = run_config(name, env, out / name, nranks)
            cfg = record["configs"][name]
            print(f"   {cfg.get('seconds', 0):.0f} s, dofs {cfg.get('dofs')}, error {cfg.get('error')}", flush=True)
            record_path.write_text(json.dumps(record, indent=1, default=float) + "\n")
        record["comparison"] = compare(record)
        record["energy_protocol"] = energy_protocol(record)
        record_path.write_text(json.dumps(record, indent=1, default=float) + "\n")
        (ROOT / "benchmarks" / "records" / "tree_cotree_validation.json").write_text(json.dumps(record, indent=1, default=float) + "\n")
    if "comparison" not in record:
        record["comparison"] = compare(record)
    record["energy_protocol"] = energy_protocol(record)
    text = report(record)
    (ROOT / "docs" / "validation" / "tree_cotree.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
