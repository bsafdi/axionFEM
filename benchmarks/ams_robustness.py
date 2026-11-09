"""Robustness and cost of the AMS-preconditioned 3D static/pickup solves.

The cases are ones that stalled with the earlier solver settings: a toroid with a real slit at ND2/ND3, CORE with
iron caps at ND2 and a superconducting sphere in a background field.  Each case is solved with several
solver variants, selected through the AXF_* environment knobs of `solve()` in cpp/src/model3d.cpp, so one
build measures both the earlier ("m2") and the fixed settings.

    python benchmarks/ams_robustness.py write runs/ams [case ...]   # write the designs
    python benchmarks/ams_robustness.py cases                       # case names, one per line
    python benchmarks/ams_robustness.py ranks CASE                  # MPI ranks the case is run with
    python benchmarks/ams_robustness.py env VARIANT                 # "K=V ..." of a variant
    python benchmarks/ams_robustness.py collect runs/ams            # record + markdown table

The solves are run with `srun -n <ranks> build/axf solve ...` (or the site's MPI launcher), one job per case and variant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import axionfem as af  # noqa: E402
from axionfem.convert import to_3d  # noqa: E402

REC = ROOT / "benchmarks" / "records"
DOC = ROOT / "docs" / "validation" / "ams_robustness.md"

# ---------------------------------------------------------------------------- cases


def _toroid(slit: float | None, h: float, order: int) -> dict:
    d2 = af.template("toroid", R=1.0, H=1.0, tw=0.06, g=0.04, ts=0.05, f=0.06,
                     mesh={"order": 2, "size": 0.04, "surface_size": 0.015})
    return to_3d(d2, slit_deg=slit or 4.0, pickup="slit" if slit else "ideal",
                 mesh={"size": h, "surface_size": h / 2.5, "order": order}).doc


def _core(h: float, order: int, surface: float) -> dict:
    # iron_t = 0.1 are the iron caps of the regression tests. The CORE template's own surface size,
    # min(w, gn)/2 = 5 mm at Rc = 1 m, does not mesh in any reasonable time on a 2.5 m cavity, so the
    # surface mesh is h/3, which gives 1.6 M DOFs at h = 0.5 and is 17x coarser
    # than the 10 mm walls and neck gap of the template.
    return to_3d(af.template("core", iron_t=0.1),
                 mesh={"size": h, "surface_size": surface, "order": order}).doc


def _solenoid_ext(radius: float) -> dict:
    # the coordinator's full-wave diagnostic designs (coax pickup through a port, large infinity shell):
    # the static solve broke down after 1-10 iterations on 16 ranks, the worse the larger the shell
    d = to_3d(af.template("solenoid"),
              mesh={"size": 0.35, "surface_size": 0.08, "order": 2, "domain_radius": radius}).doc
    d.pop("fullwave", None)  # the magnetostatic/pickup solves are what fails
    return d


def _sphere(h: float, order: int = 2) -> dict:
    return {"schema": "axionfem.design/1", "name": "sc_sphere", "symmetry": "3d",
            "regions": [{"name": "ball", "solid": {"sphere": {"center": [0, 0, 0], "radius": 0.2}},
                         "material": "superconductor"}],
            "windings": [], "background": {"type": "uniform", "Bz": 0.1},
            "mesh": {"size": h, "surface_size": h / 4, "order": order}}


# name -> (design, MPI ranks). The comments record the outcome with the earlier settings.
CASES: dict[str, tuple] = {
    "toroid_slit1.5_nd1_h0.45": (lambda: _toroid(1.5, 0.45, 1), 1),
    "toroid_slit3_nd2_h0.45": (lambda: _toroid(3.0, 0.45, 2), 8),      # earlier settings: 880 s, converged
    "toroid_slit1.5_nd2_h0.45": (lambda: _toroid(1.5, 0.45, 2), 8),    # earlier settings: all solves at the cap
    "toroid_slit3_nd3_h0.45": (lambda: _toroid(3.0, 0.45, 3), 8),      # earlier settings: pickup residual 0.38
    "toroid_ideal_nd3_h0.45": (lambda: _toroid(None, 0.45, 3), 8),     # the same at ND3 without a slit
    # the CORE case (5 solves at the iteration cap, 3.1 h with the earlier settings). Its surface mesh is much coarser than
    # the thin walls it has to resolve, which is the hardest case for the gauge mass: the pre-fix solver
    # breaks down on it within a few iterations, the default converges but needs ~780 iterations, and
    # AXF_GAUGE=1e-11 needs ~150 (docs/validation/ams_robustness.md).
    "core_iron_nd2_h0.5_coarse_surface": (lambda: _core(0.5, 2, 0.5 / 3), 8),
    "core_iron_nd2_h0.5_coarse_surface_np1": (lambda: _core(0.5, 2, 0.5 / 3), 1),
    "solenoid_coax_nd2_R3": (lambda: _solenoid_ext(3.0), 8),
    "solenoid_coax_nd2_R4.5": (lambda: _solenoid_ext(4.5), 8),
    "sc_sphere_h0.12": (lambda: _sphere(0.12), 4),
    "sc_sphere_h0.08": (lambda: _sphere(0.08), 8),                     # earlier settings: cap at 1.1e-10
}

# ---------------------------------------------------------------------------- solver variants
# "m2" reproduces the solver as it was before the gauge-mass change (one global gauge mass, hypre's AMS defaults,
# no reuse); the empty dict is the current default. The others isolate one setting each.
VARIANTS: dict[str, dict[str, str]] = {
    "m2": {"AXF_GAUGE": "1e-8", "AXF_GAUGE_GLOBAL": "1", "AXF_AMS_THETA": "0.25", "AXF_AMS_AGG": "1",
           "AXF_AMS_ESS_PROJ": "0", "AXF_SOLVER_REUSE": "0"},
    "new": {},
    "m2_plus_reuse": {"AXF_GAUGE": "1e-8", "AXF_GAUGE_GLOBAL": "1", "AXF_AMS_THETA": "0.25",
                      "AXF_AMS_AGG": "1", "AXF_AMS_ESS_PROJ": "0"},
    "gauge_global": {"AXF_GAUGE": "1e-8", "AXF_GAUGE_GLOBAL": "1"},
    "gauge1e-11": {"AXF_GAUGE": "1e-11"},
    "gauge1e-12": {"AXF_GAUGE": "1e-12"},
    "gauge1e-13": {"AXF_GAUGE": "1e-13"},
    "gauge1e-14": {"AXF_GAUGE": "1e-14"},
    "gauge1e-15": {"AXF_GAUGE": "1e-15"},
    "gauge0": {"AXF_GAUGE": "0"},
    "amg_m2": {"AXF_AMS_THETA": "0.25", "AXF_AMS_AGG": "1"},
    "theta0.25": {"AXF_AMS_THETA": "0.25"},
    "agg1": {"AXF_AMS_AGG": "1"},
    "sweeps2": {"AXF_AMS_SWEEPS": "2"},
    "sweeps3": {"AXF_AMS_SWEEPS": "3"},
    "chebyshev": {"AXF_AMS_RLX": "16"},
    "maxit3000": {"AXF_SOLVER_MAXIT": "3000"},
    "singular": {"AXF_GAUGE": "0", "AXF_AMS_SINGULAR": "1"},
    "fgmres": {"AXF_SOLVER_FGMRES": "1"},
    "no_proj": {"AXF_AMS_ESS_PROJ": "0"},
}

KEYS = ("Psi_SI_H_T_m2", "L_p_H", "W_mag_J")

# Environment knob -> the key it sets in results.json's solver.options. `collect` checks that the variant
# really reached the ranks: a batch environment that exports nothing (SLURM_EXPORT_ENV=NONE) silently drops them from srun.
OPTION_OF = {"AXF_GAUGE": "gauge", "AXF_GAUGE_GLOBAL": "gauge_global", "AXF_AMS_THETA": "ams_theta",
             "AXF_AMS_AGG": "ams_agg_levels", "AXF_AMS_SWEEPS": "ams_sweeps", "AXF_AMS_CYCLE": "ams_cycle",
             "AXF_AMS_ESS_PROJ": "essential_projection", "AXF_SOLVER_REUSE": "reuse",
             "AXF_AMS_SINGULAR": "singular", "AXF_SOLVER_FGMRES": "fgmres", "AXF_SOLVER_MAXIT": "max_iterations"}


def _env_applied(variant: str, options: dict) -> list[str]:
    bad = []
    for k, v in VARIANTS[variant].items():
        key = OPTION_OF.get(k)
        if key is None or key not in options:
            continue
        got, want = options[key], float(v)
        if isinstance(got, bool):
            got = float(got)
        if abs(float(got) - want) > 1e-12 * max(1.0, abs(want)):
            bad.append(f"{k}={v} but solver.options.{key}={options[key]}")
    return bad

# ---------------------------------------------------------------------------- commands


def write(base: Path, names: list[str]) -> None:
    for n in names:
        d = base / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "design.json").write_text(json.dumps(CASES[n][0](), indent=1))
        print(n)


def collect(base: Path) -> None:
    rows = []
    for name in CASES:
        for var in VARIANTS:
            run = base / name / var
            f = run / "results.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            s, t, c = r.get("solver", {}), r.get("timing", {}), r.get("coupling", {})
            bad = _env_applied(var, s.get("options", {}))
            if bad:
                print(f"!! {name}/{var}: the variant did not reach the solver: {'; '.join(bad)}")
            rss, wall = 0.0, 0.0
            log = base / name / f"{var}.log"
            if log.exists():  # "rank <s> s <maxrss> kB" per rank and "wall <s> s" from /usr/bin/time
                for line in log.read_text().splitlines():
                    p = line.split()
                    if len(p) == 5 and p[0] == "rank" and p[4] == "kB":
                        rss = max(rss, float(p[3]) / 1024 ** 2)
                    elif len(p) == 3 and p[0] == "wall":
                        wall = float(p[1])
            rows.append({"case": name, "variant": var, "ranks": CASES[name][1],
                         "dofs": r.get("mesh", {}).get("dofs"), "iterations": s.get("pcg_iterations_total"),
                         "max_relative_residual": s.get("max_relative_residual"),
                         "solves": [[x["iterations"], x["relative_residual"]] for x in s.get("solves", [])],
                         "solver_seconds": sum(x["setup_s"] + x["solve_s"] for x in s.get("solves", [])),
                         "total_seconds": t.get("total_s"), "mesh_seconds": t.get("mesh_s"),
                         "max_rss_GB_per_rank": rss, "wall_seconds": wall,
                         "options": s.get("options", {}), "variant_applied": not bad,
                         "stalled": any("iteration cap" in w or "did not converge" in w for w in r.get("warnings", [])),
                         "warnings": r.get("warnings", []),
                         **{k: c.get(k) for k in KEYS}})
    REC.mkdir(parents=True, exist_ok=True)
    (REC / "ams_robustness.json").write_text(json.dumps(rows, indent=1))
    tab = table(rows)
    if DOC.exists():  # the tables in the validation note are generated, never hand-edited
        doc = DOC.read_text()
        a, b = doc.index("<!-- TABLE -->"), doc.index("<!-- /TABLE -->")
        DOC.write_text(doc[:a] + "<!-- TABLE -->\n" + tab + "\n" + doc[b:])
    print(f"{len(rows)} runs -> {REC / 'ams_robustness.json'} and {DOC}")
    print(tab)


def table(rows: list[dict]) -> str:
    out = []
    for name in CASES:
        rs = [r for r in rows if r["case"] == name]
        if not rs:
            continue
        ref = next((r for r in rs if r["variant"] == "new"), rs[0])
        out.append(f"\n### {name} ({ref['dofs']:,} DOFs, {ref['ranks']} MPI ranks)\n" if ref["dofs"] else f"\n### {name}\n")
        out.append("| variant | PCG iterations (per solve) | max rel. residual | solver [s] | total [s] |"
                   " peak RSS/rank [GB] | Psi rel. to `new` | L_p rel. | W rel. |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            its = [i for i, _ in r["solves"]]
            per = (" + ".join(str(i) for i in its) if len(its) <= 6 else
                   f"{its[0]} + {its[1]} + ... + {its[-1]}, {len(its)} solves") or "-"
            d = []
            for k in KEYS:
                a, b = r.get(k), ref.get(k)
                d.append("-" if not a or not b else f"{(a - b) / b:+.1e}")
            flag = " **stall**" if r["stalled"] else ""
            if not r["variant_applied"]:
                flag += " **(variant not applied!)**"
            out.append(f"| `{r['variant']}` | {r['iterations']} ({per}){flag} | {r['max_relative_residual']:.1e} |"
                       f" {r['solver_seconds']:.0f} | {r['total_seconds']:.0f} | {r['max_rss_GB_per_rank']:.1f} |"
                       f" {d[0]} | {d[1]} | {d[2]} |")
    return "\n".join(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cases":
        print("\n".join(CASES))
    elif cmd == "variants":
        print("\n".join(VARIANTS))
    elif cmd == "ranks":
        print(CASES[sys.argv[2]][1])
    elif cmd == "env":
        print(" ".join(f"{k}={v}" for k, v in VARIANTS[sys.argv[2]].items()))
    elif cmd == "write":
        write(Path(sys.argv[2]), sys.argv[3:] or list(CASES))
    elif cmd == "collect":
        collect(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
