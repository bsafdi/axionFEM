"""Exact per-frequency dZ/dw (`dZ_domega_H`) against the energy identity and centred finite differences.

    python benchmarks/dzdw_check.py BUILD_DIR [NRANKS]          (solve every case, then check)
    python benchmarks/dzdw_check.py --no-solve                  (re-check the solves in runs/dzdw_check)

On benchmarks/designs/stage1_twoport.json (two gap ports on a copper frame):
  closed          PEC sphere, lossless, at 1 and 12 MHz: Im dZ/dw must equal the energy identity Re(-4 G)
                  (`dZ_domega_vs_energy_identity_relative`) to the solve residual;
  lossy           a dielectric slab (eps_r 3, tan delta 0.05) inside the frame, 12 MHz: the tan delta term of A';
  conducting      a conducting slab (sigma 0.01 S/m, loss tangent ~15 at 12 MHz): the sigma term of A';
  radiating       boundary "radiation" (no uniform source), 50 MHz, x = k R1 ~ 0.7;
  radiating_high  the same at 300 MHz, x ~ 4.4 (the Core runs reach x ~ 4.6).
The finite-difference cases are solved at f0 (1 + s d), s = -2..2, d = 1e-3. With D_d = (Z(+d) - Z(-d)) / (2 d w0):
  richardson   (4 D_d - D_2d) / 3 (truncation O(d^4), an independent check of dZ/dw at f0), per Re and Im part;
  ratio        (D_2d - dZ) / (D_d - dZ), ~4 while the d^2 truncation is above the solve noise (~1e-10 / d);
  simpson      Z(+d) - Z(-d) against (2 d w0 / 6)(dZ(-d) + 4 dZ(0) + dZ(+d)) (O(d^5); the exact derivatives at
               three frequencies against the impedances), per part;
  exterior     the radiating cases record the share of the exterior term (TE, TM) in dZ/dw, gated to be exercised.
On the parallel-plate TEM line of benchmarks/fw3d_robin_validation.py (one surface port, nx 12, order 2, 170 MHz;
the Robin and PML terms of A' are included):
  tem_pec            shorted (control);
  tem_scattering     matched by a first-order absorbing (Robin) boundary: beta = -i w sqrt(eps/mu);
  tem_impedance      matched, with copper impedance-boundary plates (material Z_s ~ sqrt(w));
  tem_impedance_Zs   ended by an impedance wall with a fixed Z_s = 100 - 50i ohm: beta = -i w / Z_s;
  tem_sheet          matched, with a resistive sheet across the line (sheet group, beta = -2 w y tan(k d / 2));
  tem_pml            terminated by a Cartesian PML (vacuum wavelength c / f: the stretch scales with 1 / w);
  tem_pml_fixed      the same PML with a fixed wavelength (stretched tensors independent of w).
These are gated on errors relative to max(|difference quotient|, |Z| / w), since a matched line's dZ/dw is ~0:
richardson_scaled and simpson_scaled (the span against max(|span|, 2 d |Z|)), both below 1e-6: the PML solves stall at a
true relative residual of about 1e-7 (the other cases reach 1e-9 to 1e-10).
A case fails if its solve fails or a gate is missed; the record's top-level `pass` is the conjunction. Writes
benchmarks/records/dzdw_check.json (with the commit, the dirty flag and the source hashes of the solve).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
RUNS = ROOT / "runs" / "dzdw_check"
OUT = ROOT / "benchmarks" / "records" / "dzdw_check.json"
SOURCES = ["cpp/src/model3d_fullwave.cpp", "cpp/src/axi_fullwave.cpp", "cpp/src/pml.cpp"]
D = 1e-3
STEPS = [-2, -1, 0, 1, 2]
GATES = {"energy_identity": 1e-9, "richardson_im": 1e-6, "richardson_re": 1e-4, "simpson": 1e-7,
         "exterior_share_min": 1e-3, "richardson_scaled": 1e-6, "simpson_scaled": 1e-6}
TEM_F0 = 1.7e8
TEM_NX = 12
TEM_CASES = ("tem_pec", "tem_scattering", "tem_impedance", "tem_impedance_Zs", "tem_sheet", "tem_pml", "tem_pml_fixed")


def tem_case(name: str, wd: Path) -> dict:
    """The TEM line with the termination of `name`, solved at TEM_F0 (1 + s D); the mesh files are written to wd."""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import fw3d_robin_validation as robin
    import fullwave_indicator as fwi
    freqs = [TEM_F0 * (1 + s * D) for s in STEPS]
    if name == "tem_impedance":
        doc, _ = robin.build(wd, termination="scattering", plates="impedance", nx=TEM_NX, order=2)
    elif name == "tem_sheet":
        doc, _ = fwi.tem_design(wd, "scattering", 2, 0, freqs, sheet=fwi.TEM_SHEET, nx=TEM_NX)
    elif name in ("tem_pml", "tem_pml_fixed"):
        doc, _ = fwi.tem_design(wd, "pml", 2, 0, freqs, nx=TEM_NX)
        if name == "tem_pml_fixed":
            doc["fullwave"]["pml"][0]["wavelength_m"] = 1.5
    else:
        termination = {"tem_pec": "pec", "tem_scattering": "scattering", "tem_impedance_Zs": "pec"}[name]
        doc, _ = fwi.tem_design(wd, termination, 2, 0, freqs, nx=TEM_NX)
    if name == "tem_impedance_Zs":
        # the shorting end becomes an impedance wall of fixed Z_s
        info = Path(doc["mesh"]["info_file"])
        meta = json.loads(info.read_text())
        for group in meta["boundary_groups"]:
            if group["name"] == "end":
                group["type"] = "impedance"
                group["Zs_ohm"] = [100.0, -50.0]
        info.write_text(json.dumps(meta))
    doc["fullwave"]["frequencies"] = freqs
    doc["mesh"]["refinements"] = 0
    return doc


def slab(material: dict) -> tuple[list[dict], dict]:
    return [{"name": "slab", "material": "slab_material",
             "solid": {"box": {"min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]}}}], {"slab_material": material}


def cases() -> dict:
    base = json.loads(DESIGN.read_text())
    out = {}
    closed = copy.deepcopy(base)
    closed["fullwave"]["frequencies"] = [1e6, 12e6]
    out["closed"] = closed
    for name, material in [("lossy", {"kind": "linear", "mu_r": 1.0, "eps_r": 3.0, "tan_delta": 0.05}),
                           ("conducting", {"kind": "linear", "mu_r": 1.0, "eps_r": 1.0, "sigma_S_m": 0.01})]:
        doc = copy.deepcopy(base)
        regions, materials = slab(material)
        doc["materials"] = materials
        doc["regions"] += regions
        doc["fullwave"]["frequencies"] = [12e6 * (1 + s * D) for s in STEPS]
        out[name] = doc
    for name, f0 in [("radiating", 50e6), ("radiating_high", 300e6)]:
        doc = copy.deepcopy(base)
        for k in ("source", "source_check", "validation_shunt_R_ohm"):
            doc["fullwave"].pop(k, None)
        doc["fullwave"]["boundary"] = "radiation"
        doc["fullwave"]["frequencies"] = [f0 * (1 + s * D) for s in STEPS]
        out[name] = doc
    for name in TEM_CASES:
        out[name] = tem_case(name, RUNS / name)
    return out


def cmat(m) -> np.ndarray:
    return np.array(m["re"], dtype=float) + 1j * np.array(m["im"], dtype=float)


def zmat(s) -> np.ndarray:
    return np.array(s["Z_matrix_re_ohm"], dtype=float) + 1j * np.array(s["Z_matrix_im_ohm"], dtype=float)


def rel_parts(err: np.ndarray, ref: np.ndarray) -> dict:
    """max |err| / max |ref| for the real and the imaginary part separately."""
    out = {}
    for part, f in (("re", np.real), ("im", np.imag)):
        out[part] = float(np.max(np.abs(f(err))) / max(float(np.max(np.abs(f(ref)))), 1e-300))
    return out


def provenance() -> dict:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, env=env).stdout.strip()  # noqa: E731
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCES}}


def check_fd(sweep: list[dict]) -> dict:
    f = np.array([s["f_Hz"] for s in sweep])
    order = np.argsort(f)
    sweep = [sweep[i] for i in order]
    w = 2 * np.pi * np.array([s["f_Hz"] for s in sweep])
    Z = [zmat(s) for s in sweep]
    dZ = [cmat(s["dZ_domega_H"]) for s in sweep]
    ex = dZ[2]
    d1 = (Z[3] - Z[1]) / (w[3] - w[1])
    d2 = (Z[4] - Z[0]) / (w[4] - w[0])
    rich = (4 * d1 - d2) / 3
    ratio = np.abs(d2 - ex) / np.maximum(np.abs(d1 - ex), 1e-300)
    span = Z[3] - Z[1]
    simpson = (w[3] - w[1]) / 6 * (dZ[1] + 4 * ex + dZ[3])
    c = {"f0_Hz": float(sweep[2]["f_Hz"]),
         "exact_dZ_H": {"re": ex.real.tolist(), "im": ex.imag.tolist()},
         "fd_d_minus_exact": rel_parts(d1 - ex, ex),
         "richardson_minus_exact": rel_parts(rich - ex, ex),
         "truncation_ratio_median": float(np.median(ratio)),
         "simpson_residual": rel_parts(span - simpson, span)}
    # relative to max(|difference quotient|, |Z| / w): meaningful where dZ/dw is ~0 (a matched line)
    z_over_w = float(np.max(np.abs(Z[2]))) / w[2]
    c["richardson_scaled"] = float(np.max(np.abs(rich - ex)) / max(float(np.max(np.abs(rich))), z_over_w))
    c["simpson_scaled"] = float(np.max(np.abs(span - simpson)) /
                                max(float(np.max(np.abs(span))), 2 * D * float(np.max(np.abs(Z[2])))))
    c["Z_ohm"] = {"re": Z[2].real.tolist(), "im": Z[2].imag.tolist()}
    c["richardson_H"] = {"re": rich.real.tolist(), "im": rich.imag.tolist()}
    c["scope"] = sweep[2].get("dZ_domega_scope")
    if "dZ_domega_exterior_TE_H" in sweep[2]:
        scale = float(np.max(np.abs(ex)))
        c["exterior_share"] = {fam: float(np.max(np.abs(cmat(sweep[2][f"dZ_domega_exterior_{fam}_H"])))) / scale
                               for fam in ("TE", "TM")}
    return c


def gates(name: str, c: dict) -> list[str]:
    missed = []
    if "error" in c:
        return ["solve failed"]
    if not all(r["has_dZ"] for r in c["rows"]):
        missed.append("dZ_domega_H missing")
    if name == "closed":
        worst = max(r["vs_energy_identity"] for r in c["rows"])
        if not worst < GATES["energy_identity"]:
            missed.append(f"energy identity {worst:.2e}")
        return missed
    if "richardson_minus_exact" not in c:
        return missed + [f"finite-difference check not run ({len(c['rows'])} of {len(STEPS)} frequencies)"]
    if name.startswith("tem_"):
        for key in ("richardson_scaled", "simpson_scaled"):
            if not c[key] < GATES[key]:
                missed.append(f"{key} {c[key]:.2e}")
        return missed
    if not c["richardson_minus_exact"]["im"] < GATES["richardson_im"]:
        missed.append(f"richardson im {c['richardson_minus_exact']['im']:.2e}")
    if not c["richardson_minus_exact"]["re"] < GATES["richardson_re"]:
        missed.append(f"richardson re {c['richardson_minus_exact']['re']:.2e}")
    for part in ("re", "im"):
        if not c["simpson_residual"][part] < GATES["simpson"]:
            missed.append(f"simpson {part} {c['simpson_residual'][part]:.2e}")
    if name.startswith("radiating"):
        share = c.get("exterior_share")
        if not share or not max(share.values()) > GATES["exterior_share_min"]:
            missed.append(f"exterior term not exercised: {share}")
    return missed


def _mpi_launcher(nranks: int) -> list:
    """The launcher prefix for nranks > 1, by the rule of axionfem.Design.solve: $AXF_MPI_LAUNCHER, else srun inside
    a SLURM job when srun is on PATH, else mpirun, else mpiexec."""
    if nranks <= 1:
        return []
    import shlex
    import shutil
    cmd = os.environ.get("AXF_MPI_LAUNCHER")
    if not cmd:
        if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
            cmd = "srun"
        else:
            cmd = "mpirun" if shutil.which("mpirun") or not shutil.which("mpiexec") else "mpiexec"
    return shlex.split(cmd) + ["-n", str(nranks)]


def solve(name: str, doc: dict, build: Path, nranks: int) -> dict:
    wd = RUNS / name
    wd.mkdir(parents=True, exist_ok=True)
    res = wd / "results.json"
    res.unlink(missing_ok=True)
    (wd / "design.json").write_text(json.dumps(doc, indent=1))
    env = dict(os.environ, AXF_MESHER=str(build / "axf-mesher"))
    launcher = _mpi_launcher(nranks)
    p = subprocess.run(launcher + [str(build / "axf"), "solve", str(wd / "design.json"), "-o", str(wd), "-v", "1"],
                       capture_output=True, text=True, env=env)
    (wd / "solve.log").write_text(p.stdout + "\n" + p.stderr)
    return {"returncode": p.returncode, "tail": (p.stderr or p.stdout)[-1500:]}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_solve = "--no-solve" not in sys.argv
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    if do_solve:
        build = Path(args[0]).resolve()
        nranks = int(args[1]) if len(args) > 1 else 4
        rec = {"check": "dzdw_check", "date": time.strftime("%Y-%m-%d %H:%M"), "build": build.name,
               "provenance": provenance()}
    else:
        rec = {k: old.get(k) for k in ("check", "date", "build", "provenance")}
        rec["reanalysed"] = time.strftime("%Y-%m-%d %H:%M")
    rec.update({"d": D, "steps": STEPS, "gates": GATES, "cases": {}})
    for name, doc in cases().items():
        run = solve(name, doc, build, nranks) if do_solve else None
        res = RUNS / name / "results.json"
        if (run and run["returncode"] != 0) or not res.exists():
            c = {"error": run["tail"] if run else f"no {res}", "returncode": run and run["returncode"]}
        else:
            sweep = json.loads(res.read_text())["fullwave"]["sweep"]
            c = {"returncode": run and run["returncode"],
                 "rows": [{"f_Hz": s["f_Hz"], "has_dZ": "dZ_domega_H" in s,
                           "vs_energy_identity": s.get("dZ_domega_vs_energy_identity_relative"),
                           "dZ_seconds": s.get("dZ_domega_seconds")} for s in sweep]}
            if name != "closed" and len(sweep) == len(STEPS) and all("dZ_domega_H" in s for s in sweep):
                c.update(check_fd(sweep))
        c["gates_missed"] = gates(name, c)
        c["pass"] = not c["gates_missed"]
        rec["cases"][name] = c
        print(name, "PASS" if c["pass"] else "FAIL",
              json.dumps({k: v for k, v in c.items() if k not in ("exact_dZ_H", "rows", "tail")}), flush=True)
    rec["pass"] = all(c["pass"] for c in rec["cases"].values())
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print("overall", "PASS" if rec["pass"] else "FAIL")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
