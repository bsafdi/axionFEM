"""Tree-cotree validation, run 2: a capacitor at low frequency against its closed form.

    python benchmarks/tree_cotree_capacitor.py run [--ranks N]
    python benchmarks/tree_cotree_capacitor.py run-plain [--ranks N]   (only the plain-STRUMPACK comparison)
    python benchmarks/tree_cotree_capacitor.py report
    (AXF_TC_CAP_RUNS: the run directory, default runs/tree_cotree_capacitor)

The tree-cotree validation needs an
independent capacitive reference: a floating sphere inside a conductor shell, C = 4 pi eps0 a b / (b - a), to <= 0.5 %
over three mesh levels, and the gauged path against plain STRUMPACK to <= 1e-9 at >= 100 kHz. This replaces
a stub-fed sphere inside a 0.7 m wall and the isolated-electrode pilot on the two-port frame (a port
bridging two distinct conductor bodies, capacitive at 12 MHz).

Design: the two-port frame (benchmarks/designs/stage1_twoport.json, which supplies the primary readout port)
plus a sphere (a = 5 cm) inside a closed shell (inner radius b = 10 cm, 2 cm thick), both copper (perfect at RF),
bridged by a thin vacuum cylinder (radius 6 mm) along +z, the third full-wave port. The shell is closed, so it screens
the frame, and the port is vacuum like the gap, so the closed form is the continuum reference. The gap and the port
are their own regions with a volume size (a Restrict field), so every level refines the gap uniformly: size s in the
gap, 0.6 s in the port. The
surface-size thresholds reach 0.1 m (mesh.surface_layer), so the refinement stays near the capacitor: the frame's
vertices are the same at every level, its tetrahedra and the air mesh are not (the frame ports' U_m varies by 0.7 %).

C = 1 / (w Im Z_cap,cap): the engine's reactance is X = -Im Z, so a capacitor has Im Z = +1/(w C). Gauged path
(AXF_FW_GAUGE=tree-cotree, lowest-order Nedelec) at 10 Hz to 12 MHz on two geometries at five values of s: straight
tetrahedra (mesh order 1), and curved (mesh order 2) with the field still order 1 (solve order override). The three
finest levels are extrapolated (axionfem.convergence.richardson) in the measured gap size h = (V_gap / N_gap)^(1/3).
The gauged variants are the candidate causes of the low-frequency error: the charge projection
(AXF_FW_TC_CHARGE_PROJECTION 0, 1 or 2) and the refinement's stopping rule (AXF_FW_TC_REFINE). The committed record
has all five on the straight levels (but componentwise with projection 2 at 2.5 mm) and two (0 and 2) on the curved
ones (AXF_TC_CAP_GEOMETRY / AXF_TC_CAP_VARIANTS restrict a run). The coarsest and finest straight levels also run plain
STRUMPACK at 100 kHz to 12 MHz (the ungauged comparison), one frequency per design. Every run records each port's
electric and
magnetic energy and right-hand-side charge, and the frame ports' U_e/f^2 at 10 Hz and 100 Hz against 1 kHz with the
implied field error sqrt(dU_e/U_e) (the low-frequency check of run 1).
Writes benchmarks/records/tree_cotree_capacitor.json.
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

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "python"))   # after PYTHONPATH, so that a package copy (runs/pkgN) can take precedence

RUNS = Path(os.environ.get("AXF_TC_CAP_RUNS", ROOT / "runs" / "tree_cotree_capacitor")).resolve()
OUT = ROOT / "benchmarks" / "records" / "tree_cotree_capacitor.json"
STAGE1 = ROOT / "benchmarks" / "designs" / "stage1_twoport.json"
EPS0 = 1.0 / (1.25663706212e-6 * 299792458.0 ** 2)
A, B, T, RP = 0.05, 0.10, 0.02, 0.006
CENTER = [0.0, 0.4, 0.0]
FREQUENCIES = [1e1, 1e2, 1e3, 1e5, 1e6, 1.2e7]
HIGH = [1e5, 1e6, 1.2e7]                       # the gauged-vs-plain comparison
SIZES = [0.01, 0.007, 0.005, 0.0035, 0.0025]   # gap size s (the three finest are extrapolated)
SURFACE_LAYER = 0.1                            # reach of the surface-size thresholds [m]
GEOMETRY = {"straight": 1, "curved": 2}        # mesh.order (geometry); the field is ND1 in both
GAUGED = {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "tree-cotree", "AXF_FW_TC_CHARGE_PROJECTION": "0",
          "AXF_FW_TC_REFINE": "normwise", "AXF_FW_TC_RESIDUAL": ""}   # empty: the default (the extended default)
VARIANTS = {   # every level: the low-frequency error's candidate causes
    "": GAUGED,
    "_projection": {**GAUGED, "AXF_FW_TC_CHARGE_PROJECTION": "1"},
    "_projection2": {**GAUGED, "AXF_FW_TC_CHARGE_PROJECTION": "2"},
    # componentwise stopping needs the double residual (with the extended one the loop stops on the correction)
    "_componentwise": {**GAUGED, "AXF_FW_TC_REFINE": "componentwise", "AXF_FW_TC_RESIDUAL": "double"},
    "_componentwise_projection2": {**GAUGED, "AXF_FW_TC_REFINE": "componentwise", "AXF_FW_TC_CHARGE_PROJECTION": "2",
                                   "AXF_FW_TC_RESIDUAL": "double"},
}
PLAIN = {"AXF_FW_SOLVER": "strumpack", "AXF_FW_GAUGE": "none", "AXF_FW_STRUMPACK_DIST": "0",
         "AXF_FW_TC_CHARGE_PROJECTION": "0", "AXF_FW_TC_REFINE": "normwise", "AXF_FW_TC_RESIDUAL": ""}   # straight, at HIGH
PLAIN_SIZES = [SIZES[0], SIZES[-1]]   # the coarsest and finest gap sizes


def exact_C() -> float:
    return 4 * math.pi * EPS0 * A * B / (B - A)


def ball(r):
    return {"sphere": {"center": CENTER, "radius": r}}


def design(s: float, geometry_order: int, frequencies=FREQUENCIES) -> dict:
    doc = json.loads(STAGE1.read_text())
    doc["name"] = f"tree_cotree_capacitor_s{s:g}_g{geometry_order}"
    shell = {"difference": [ball(B + T), ball(B)]}
    cylinder = {"cylinder": {"base": CENTER, "axis": [0, 0, B + T / 2], "radius": RP}}
    doc["regions"] += [
        {"name": "cap_inner", "material": "copper", "solid": ball(A), "surface_size": s},
        {"name": "cap_shell", "material": "copper", "solid": shell, "surface_size": s},
        {"name": "cap_gap", "material": "vacuum", "solid": {"difference": [ball(B), ball(A), cylinder]},
         "mesh_size": s, "surface_size": s},
        {"name": "cap_port", "material": "vacuum", "solid": {"difference": [cylinder, ball(A), shell]},
         "mesh_size": 0.6 * s, "surface_size": 0.6 * s},
    ]
    fw = doc["fullwave"]
    fw["frequencies"] = list(frequencies)
    fw["ports"].append({"name": "cap", "region": "cap_port", "toward": [0, 0, 1]})
    for key in ("source_check", "validation_shunt_R_ohm"):
        fw.pop(key, None)
    doc["mesh"]["order"] = geometry_order
    doc["mesh"]["surface_layer"] = SURFACE_LAYER
    return doc


def solve(name: str, doc: dict, env: dict, ranks: int) -> None:
    import axionfem as af
    for k, v in env.items():
        os.environ[k] = v
    t0 = time.time()
    try:
        af.Design(doc).solve(RUNS / name, order=1, verbose=0, mpi_ranks=ranks)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001  (the report records what is missing)
        print(f"{name} failed: {type(exc).__name__}: {exc}", flush=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    produced = {"AXF_EXE": Path(os.environ.get("AXF_EXE", "")).name}
    (RUNS / name / "env.json").write_text(json.dumps({**env, "produced_by": produced}) + "\n")


def plain_name(s: float, f: float) -> str:
    return f"straight_s{s:g}_plain_strumpack_f{f:g}"


def run_plain(ranks: int) -> None:
    """Plain STRUMPACK, one design per frequency so that a refused frequency does not abort the others (it refused
    100 kHz on the finest mesh: the capacitive port's residual 1.1e-5 against the 1e-6 gate)."""
    for s in PLAIN_SIZES:
        for f in HIGH:
            solve(plain_name(s, f), design(s, 1, [f]), PLAIN, ranks)


def run(ranks: int) -> None:
    """Every geometry, level and variant; AXF_TC_CAP_GEOMETRY and AXF_TC_CAP_VARIANTS ("+"-separated, since a batch
    --export splits on commas; "base" is the unsuffixed variant) restrict it, e.g. to run the curved levels in a
    separate job."""
    geometries = os.environ.get("AXF_TC_CAP_GEOMETRY", "+".join(GEOMETRY)).split("+")
    variants = [v if v != "base" else "" for v in os.environ.get("AXF_TC_CAP_VARIANTS", "").split("+") if v] or VARIANTS
    if "AXF_TC_CAP_GEOMETRY" not in os.environ:
        run_plain(ranks)
    for geom in geometries:
        for s in SIZES:
            for suffix in variants:
                name = f"{geom}_s{s:g}{suffix}"
                if (RUNS / name / "results.json").exists():   # done (a rerun after an interrupted job)
                    print(f"{name}: kept", flush=True)
                    continue
                # Each variant meshes anew. Straight meshes are byte-identical. Curved ones were not
                # with gmsh's high-order optimizer, which moved nodes by up to 1 % of the element size run to run
                # (the record's meshes come from a mesher without this guarantee); axf-mesher gives one mesh per input, scope in
                # docs/physics/fullwave_tree_cotree.md. Importing one mesh (mesh.file) would need fullwave.rf_only,
                # which also drops the infinity shell, so the variants would no longer differ only in the solver
                # option.
                solve(name, design(s, GEOMETRY[geom]), VARIANTS[suffix], ranks)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        return ""


def file_record(path_str: str) -> dict:
    path = Path(path_str) if path_str else None
    if not path or not path.exists():
        return {"path": Path(path_str).name if path_str else path_str}
    return {"path": path.name, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "sha256": sha256(path)}


def provenance() -> dict:
    """The code: the commit and diff hash at report time, and the solver itself (axf differs between build
    directories only in its RPATH; the solver is libaxionfem.so next to it). Each run directory's env.json names the
    job and executable that produced it."""
    import axionfem as af
    diff = git("diff", "HEAD")
    exe = os.environ.get("AXF_EXE", "")
    return {"git_describe": git("describe", "--always", "--dirty").strip(), "git_head": git("rev-parse", "HEAD").strip(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "AXF_EXE": file_record(exe), "libaxionfem": file_record(str(Path(exe).parent / "libaxionfem.so") if exe else ""),
            "AXF_MESHER": file_record(os.environ.get("AXF_MESHER", "")),
            "python_core": file_record(af._core.__file__), "driver": file_record(__file__),
            "date": time.strftime("%Y-%m-%d")}


def gap_mesh(run_dir: Path) -> dict:
    """Size of the gap's tetrahedra (centroid between the spheres, outside the port): count, volume, mean edge
    length and h = (V / N)^(1/3). Geometric, so it also measures meshes whose gap was air."""
    from axionfem.comsol import read_gmsh
    mesh_in = json.loads((run_dir / "mesh3d_in.json").read_text())
    port = next(r["attr"] for r in mesh_in["regions"] if r["name"] == "cap_port")
    mesh = read_gmsh(run_dir / "mesh3d.msh")
    tets = np.concatenate([conn[phys != port, :4] for _, phys, conn in mesh.elements(3)])
    radius = np.linalg.norm(mesh.nodes[tets].mean(axis=1) - np.array(CENTER), axis=1)
    tets = tets[(radius > A) & (radius < B)]
    x = mesh.nodes[tets]
    vol = np.abs(np.einsum("ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))) / 6
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = np.mean([np.linalg.norm(x[:, i] - x[:, j], axis=1) for i, j in pairs], axis=0)
    return {"tetrahedra": int(len(tets)), "vertices": int(len(np.unique(tets))), "volume_m3": float(vol.sum()),
            "mean_edge_m": float(edges.mean()), "h_m": float((vol.sum() / len(tets)) ** (1 / 3)),
            "geometry_order": mesh_in.get("order"), "mesh_sha256": sha256(run_dir / "mesh3d.msh")}


def tc_stats(solver: dict | None) -> dict:
    tc = (solver or {}).get("tree_cotree") or {}
    keys = ("rhs_free_vertex_charge_relative", "rhs_free_vertex_charge_relative_solved",
            "rhs_uncharged_conductor_charge_relative", "rhs_uncharged_conductor_charge_relative_solved",
            "charged_conductor_rows", "charge_projection", "refinement", "refinement_steps",
            "rhs_charge_field_fraction", "gauss_residual_field_fraction", "componentwise_backward_error",
            "split_true_relative_residual", "gauss_block_relative_residual", "curl_representative_norm_fraction")
    return {k: tc.get(k) for k in keys}


def summarize(name: str, C0: float) -> dict | None:
    run_dir = RUNS / name
    res = run_dir / "results.json"
    if not res.exists():
        return None
    d = json.loads(res.read_text())
    fw = d["fullwave"]
    rows = []
    ue = {}
    for row in fw["sweep"]:
        f = row["f_Hz"]
        w = 2 * math.pi * f
        n = len(row["Z_matrix_im_ohm"])
        e = (row.get("energy_electric_J_per_A2") or {}).get("re")
        m = (row.get("energy_magnetic_J_per_A2") or {}).get("re")
        if e:
            ue[f] = [e[i][i] / f ** 2 for i in range(n)]
        rows.append({
            "f_Hz": f,
            "C_F": 1.0 / (w * row["Z_matrix_im_ohm"][2][2]),
            "C_over_exact": 1.0 / (w * row["Z_matrix_im_ohm"][2][2]) / C0,
            "Z_re_ohm": row["Z_matrix_re_ohm"], "Z_im_ohm": row["Z_matrix_im_ohm"],
            "U_e_diag_J_per_A2": [e[i][i] for i in range(n)] if e else None,
            "U_m_diag_J_per_A2": [m[i][i] for i in range(n)] if m else None,
            "accepted": [(p.get("solver") or {}).get("accepted") for p in row.get("port_solutions", [])],
            "port_tree_cotree": [tc_stats(p.get("solver")) for p in row.get("port_solutions", [])],
        })
    low = {}   # frame ports (0, 1): U_e/f^2 against 1 kHz, and the field error it implies (the excess is a gradient)
    if 1000.0 in ue:
        for f in (10.0, 100.0):
            if f in ue:
                dev = [ue[f][i] / ue[1000.0][i] - 1 for i in range(2)]
                low[f"{f:g}"] = {"U_e_over_f2_vs_1kHz": dev, "field_error": [math.sqrt(max(x, 0.0)) for x in dev]}
    env_file = run_dir / "env.json"
    return {"dofs": d["mesh"].get("dofs"), "elements": d["mesh"].get("elements"), "gap": gap_mesh(run_dir),
            "env": json.loads(env_file.read_text()) if env_file.exists() else None,
            "solver_path": fw.get("solver_path"), "status": fw.get("status"), "warnings": d.get("warnings", []),
            "rows": rows, "frame_ports_low_frequency": low}


def report() -> None:
    from axionfem.convergence import richardson
    C0 = exact_C()
    rec = {"check": "tree_cotree_capacitor", "provenance": provenance(), "C_exact_F": C0,
           "geometry": {"a": A, "b": B, "shell_thickness": T, "port_radius": RP, "center": CENTER,
                        "surface_layer": SURFACE_LAYER},
           "frequencies_Hz": FREQUENCIES, "gap_sizes": SIZES, "runs": {}, "extrapolation": {}, "comparison": {}}
    for geom in GEOMETRY:
        for suffix in VARIANTS:
            pts = []
            for s in SIZES:
                name = f"{geom}_s{s:g}{suffix}"
                r = summarize(name, C0)
                rec["runs"][name] = r or {"missing": True}
                if not r:
                    continue
                pts.append((r["gap"]["h_m"], r["rows"][0]["C_F"]))   # 10 Hz
                stats = [t for row in r["rows"] for t in row["port_tree_cotree"]]
                q = max(t["rhs_free_vertex_charge_relative"] or 0 for t in stats)
                print(name, r["dofs"], r["gap"]["tetrahedra"], f"h {r['gap']['h_m'] * 1e3:.2f} mm",
                      [round(x["C_over_exact"], 6) for x in r["rows"]], f"q {q:.1e}",
                      json.dumps(r["frame_ports_low_frequency"]), flush=True)
            finest = sorted(pts)[:3]
            if len(pts) < len(SIZES):   # the three finest levels only, never a coarser stand-in for a missing one
                missing = [s for s in SIZES if f"{geom}_s{s:g}{suffix}" not in rec["runs"]
                           or rec["runs"][f"{geom}_s{s:g}{suffix}"].get("missing")]
                if any(s in missing for s in sorted(SIZES)[:3]):
                    rec["extrapolation"][geom + suffix] = {"not_extrapolated": f"missing levels {missing}"}
                    continue
            if len(pts) >= 3:
                ex = richardson([h for h, _ in finest], [c for _, c in finest], formal_order=2)
                ex["extrapolated_over_exact"] = ex["extrapolated"] / C0 if ex["extrapolated"] else None
                rec["extrapolation"][geom + suffix] = ex
                print(geom + suffix, "extrapolated / exact:", ex["extrapolated_over_exact"], ex["behaviour"], flush=True)
    # plain STRUMPACK against the gauged path on the same mesh level, per frequency; a refused frequency is compared
    # from its retained row (fullwave_rejected.json) and marked with the gate that refused it
    for s in PLAIN_SIZES:
        base = rec["runs"].get(f"straight_s{s:g}")
        if not base or base.get("missing"):
            continue
        gauged = {r["f_Hz"]: r for r in base["rows"]}
        for f in HIGH:
            run_dir = RUNS / plain_name(s, f)
            entry = {"f_Hz": f, "gap_size": s}
            if (run_dir / "results.json").exists():
                row = json.loads((run_dir / "results.json").read_text())["fullwave"]["sweep"][0]
            elif (run_dir / "fullwave_rejected.json").exists():
                row = json.loads((run_dir / "fullwave_rejected.json").read_text())["primary_row"]
                entry["refused"] = row.get("rejection")
                entry["port_true_residuals"] = [(p.get("solver") or {}).get("relative_residual")
                                                for p in row.get("port_solutions", [])]
            else:
                entry["missing"] = True
                rec["comparison"][plain_name(s, f)] = entry
                continue
            g = gauged.get(f)
            if g is not None and row.get("Z_matrix_re_ohm"):
                za = np.array(row["Z_matrix_re_ohm"], dtype=float) + 1j * np.array(row["Z_matrix_im_ohm"], dtype=float)
                zb = np.array(g["Z_re_ohm"]) + 1j * np.array(g["Z_im_ohm"])
                entry["Z_max_relative_vs_gauged"] = float(np.abs(za - zb).max() / np.abs(zb).max())
                entry["Z_cap_relative_vs_gauged"] = float(abs(za[2, 2] - zb[2, 2]) / abs(zb[2, 2]))
            rec["comparison"][plain_name(s, f)] = entry
            print(plain_name(s, f), json.dumps(entry), flush=True)
    rec["runs_directory"] = str(RUNS)
    (RUNS / "record.json").write_text(json.dumps(rec, indent=1) + "\n")
    # the committed record only from the default run directory (a trial with AXF_TC_CAP_RUNS writes its own copy)
    if RUNS == (ROOT / "runs" / "tree_cotree_capacitor").resolve():
        OUT.write_text(json.dumps(rec, indent=1) + "\n")
        print("wrote", OUT, "and", RUNS / "record.json")
    else:
        print("wrote", RUNS / "record.json", "(not the committed record: AXF_TC_CAP_RUNS is set)")


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "run-plain", "report"):
        sys.exit(__doc__.split("\n\n")[1])
    ranks = int(a[a.index("--ranks") + 1]) if "--ranks" in a else int(os.environ.get("SLURM_NTASKS", "4"))
    if a[0] == "run":
        run(ranks)
    if a[0] == "run-plain":
        run_plain(ranks)
    report()


if __name__ == "__main__":
    main()
