"""3D full-wave validation (complex Nedelec E field, port feed, exact exterior: static shell + low-rank DtN).

  python benchmarks/fullwave3d_validation.py OUTDIR [NRANKS]

1. Small superconducting loop read out across a slit: L(omega)/L(0) and R(omega) vs the thin-wire theory of a
   gap-fed loop (Storer 1956, Wu 1962; benchmarks/loop_theory.py, all current modes n = 0..2b/a), and vs the
   axisymmetric solver (uniform feed: the n = 0 mode only, so it misses the n >= 1 contributions).
2. Closed perfectly conducting sphere: eigenmodes vs the zeros of [x j_n(x)]' and j_n(x).
3. The fiducial toroid of Benabou et al. 2022 with a real slit: L(omega), Psi(omega) and the first
   self-resonance, which the axisymmetric (uniform-feed) solver cannot see; they quote f_SRF ~ 25 MHz.
Writes OUTDIR/fullwave3d.json and docs/validation/fullwave3d.md.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import axionfem as af  # noqa: E402
from axionfem.convert import to_3d  # noqa: E402

sys.path.insert(0, str(ROOT / "benchmarks"))
from loop_theory import loop_L_R  # noqa: E402

C = 299792458.0
ETA = 376.730313668


def loop(wd: Path, nr: int):
    a, h = 0.3, 0.02
    kas = [0.02, 0.05, 0.1, 0.2, 0.3]
    doc = {"schema": "axionfem.design/1", "name": "loop",
           "regions": [{"name": "ring", "shape": {"rect": {"r": [a - h / 2, a + h / 2], "z": [-h / 2, h / 2]}},
                        "material": "superconductor"},
                       {"name": "coil", "shape": {"rect": {"r": [0.1, 0.12], "z": [-0.05, 0.05]}}, "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1e3}],
           "superconductors": [{"body": "ring", "azimuthal_slit": True}],
           "pickup": {"type": "azimuthal", "conductor": "ring"},
           "fullwave": {"frequencies": [ka * C / (2 * math.pi * a) for ka in kas]},
           "mesh": {"order": 2, "size": 0.01, "surface_size": 0.003}}
    d2 = af.Design(doc)
    r2 = d2.solve(wd / "loop2d", verbose=0)
    out = {"kas": kas, "a": a, "h": h, "2d": r2.data["fullwave"]["sweep"], "2d_mqs": r2.coupling}
    # thin-wire theory: square section h -> equivalent radius 0.59 h; drive spread over the 4 degree slit
    th = [loop_L_R(ka * C / (2 * math.pi * a), a, 0.59 * h, gap_rad=math.radians(4.0)) for ka in kas]
    L0 = loop_L_R(1e-4 * C / (2 * math.pi * a), a, 0.59 * h, gap_rad=math.radians(4.0))[0]
    out["theory"] = [{"L_H": L, "L_over_L0": L / L0, "R_ohm": R} for L, R in th]
    for name, mesh in (("coarse", {"size": 0.1, "surface_size": 0.012, "order": 2}),
                       ("fine", {"size": 0.07, "surface_size": 0.007, "order": 2})):
        d3 = to_3d(d2, pickup="slit", slit_deg=4.0, mesh=mesh)
        t = time.time()
        r3 = d3.solve(wd / f"loop3d_{name}", verbose=0, mpi_ranks=nr)
        out[name] = {"sweep": r3.data["fullwave"]["sweep"], "mqs": r3.coupling, "dofs": r3.data["mesh"]["dofs"],
                     "seconds": time.time() - t, "warnings": r3.data["warnings"]}
    return out


def sphere(wd: Path, nr: int):
    R = 1.0
    doc = {"schema": "axionfem.design/1", "name": "sph", "symmetry": "3d",
           "regions": [{"name": "coil", "solid": {"revolve": {"profile": {"rect": {"r": [0.3, 0.32], "z": [-0.01, 0.01]}}}},
                        "material": "winding"}],
           "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1}],
           "fullwave": {"modes": 12},
           "mesh": {"order": 2, "size": 0.08, "surface_size": 0.03, "domain_radius": R}}
    r = af.Design(doc).solve(wd / "sphere3d", verbose=0, mpi_ranks=nr)
    xs = sorted(m["f_Hz"] * 2 * math.pi * R / C for m in r.data["fullwave"]["eigenmodes"]["modes"])
    exact = [2.743707] * 3 + [3.870239] * 5 + [4.493409] * 3 + [4.973420]
    assert len(xs) == len(exact), "AME did not return every requested mode"
    assert r.data["fullwave"]["eigenmodes"]["converged"], "AME diagnostics failed"
    return {"diagnostics": r.data["fullwave"]["eigenmodes"], "computed": xs, "exact": exact, "dofs": r.data["mesh"]["dofs"]}


def benabou(wd: Path, nr: int):
    d2 = af.template("toroid", R=1.0, y=0.5, H=0.5, tw=0.02, g=0.01, ts=0.025, f=0.03, NI=2.5e7,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.01})
    freqs = [1e6, 5e6, 10e6, 15e6, 18e6, 21e6, 24e6, 27e6, 30e6, 35e6, 40e6]
    d2.doc["fullwave"] = {"frequencies": freqs}
    r2 = d2.solve(wd / "benabou2d", verbose=0)
    d3 = to_3d(d2, pickup="slit", slit_deg=2.0, mesh={"size": 0.25, "surface_size": 0.04, "order": 2})
    t = time.time()
    r3 = d3.solve(wd / "benabou3d", verbose=0, mpi_ranks=nr)
    return {"freqs": freqs, "2d": r2.data["fullwave"]["sweep"], "3d": r3.data["fullwave"]["sweep"],
            "3d_resonances": r3.data["fullwave"].get("self_resonances"), "mqs3d": r3.coupling, "mqs2d": r2.coupling,
            "dofs": r3.data["mesh"]["dofs"], "seconds": time.time() - t, "warnings": r3.data["warnings"]}


def fmt(x, f="{:.4e}"):
    return "–" if x is None else f.format(x)


def report(rec) -> str:
    L = ["# 3D full-wave validation", "",
         "Generated by `benchmarks/fullwave3d_validation.py` (ND2, MPI). Time dependence e^{−iωt}, readout current",
         "I = 1 A fed through a port region across a real slit. Formulation: `cpp/src/model3d_fullwave.cpp`.", ""]
    lp = rec["loop"]
    a, h = lp["a"], lp["h"]
    L += [f"## Small superconducting loop (a = {a} m, {h * 1000:.0f} mm square section, 4° slit)", "",
          "References: (i) thin-wire theory of a gap-fed loop (Storer 1956, Wu 1962; `benchmarks/loop_theory.py`,",
          "current modes n = 0..2b/a, equivalent wire radius 0.59 × side, drive spread over the slit), normalised",
          "to its own low-frequency limit, with finite-section/feed and mode-cutoff uncertainties. Agreement to 1-2 %",
          "is not an exact-reference claim; (ii) the axisymmetric solver, whose uniform feed excites only n = 0, so it",
          "misses the n ≥ 1 reactance (the approach to the first antiresonance near ka ≈ 0.5) and radiation.", "",
          "| ka | L/L(0) theory | L/L_MQS 3D coarse | L/L_MQS 3D fine | L/L_MQS 2D | R theory [Ω] | R 3D coarse | "
          "R 3D fine | R 2D (n = 0) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, ka in enumerate(lp["kas"]):
        s2, sc, sf, t = lp["2d"][i], lp["coarse"]["sweep"][i], lp["fine"]["sweep"][i], lp["theory"][i]
        L.append(f"| {ka} | {t['L_over_L0']:.4f} | {fmt(sc.get('L_over_MQS'), '{:.4f}')} | "
                 f"{fmt(sf.get('L_over_MQS'), '{:.4f}')} | {fmt(s2.get('L_over_MQS'), '{:.4f}')} | {fmt(t['R_ohm'])} | "
                 f"{fmt(sc.get('R_rad_ohm'))} | {fmt(sf.get('R_rad_ohm'))} | {fmt(s2['R_rad_ohm'])} |")
    L += ["", f"3D unknowns: coarse {lp['coarse']['dofs']}, fine {lp['fine']['dofs']}; wall time "
          f"{lp['coarse']['seconds']:.0f} s and {lp['fine']['seconds']:.0f} s for {len(lp['kas'])} frequencies.", ""]
    sp = rec["sphere"]
    L += ["## Closed perfectly conducting sphere (R = 1 m): eigenmodes (hypre AME; 3D eigenmodes are beta)", "",
          "| mode | kR exact | kR computed | rel. |", "|---|---|---|---|"]
    for i, (g, x) in enumerate(zip(sp["computed"], sp["exact"])):
        L.append(f"| {i + 1} | {x:.6f} | {g:.6f} | {(g - x) / x:+.1e} |")
    bb = rec["benabou"]
    L += ["", "## Benabou et al. 2022 toroid with a real slit (2°)", "",
          "Axisymmetric = uniform feed (no slit resonance); 3D = slit fed at one place. Benabou et al. quote a",
          "transmission-line scale near 25 MHz; their 5-degree gap and PEC bridge/feed differ from this 2-degree slit.",
          "This is a qualitative demonstration, not a geometry-matched validation.", "",
          "| f [MHz] | L 2D / L_MQS | L 3D / L_MQS(3D) | Ψ 3D / Ψ_MQS | R 3D [Ω] |", "|---|---|---|---|---|"]
    for f, s2, s3 in zip(bb["freqs"], bb["2d"], bb["3d"]):
        L.append(f"| {f / 1e6:g} | {fmt(s2.get('L_over_MQS'), '{:.4f}')} | {fmt(s3.get('L_over_MQS'), '{:.4f}')} | "
                 f"{fmt(s3.get('Psi_over_MQS'), '{:.4f}')} | {fmt(s3.get('Z_re_ohm'))} |")
    L += ["", f"3D self-resonances (sign change of L): {json.dumps(bb['3d_resonances'])}; 3D unknowns {bb['dofs']}, "
          f"wall time {bb['seconds']:.0f} s.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    wd = Path(sys.argv[1])
    nr = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    wd.mkdir(parents=True, exist_ok=True)
    rec = {}
    for name, fn in (("loop", loop), ("sphere", sphere), ("benabou", benabou)):
        t = time.time()
        rec[name] = fn(wd, nr)
        print(name, f"{time.time() - t:.0f} s", flush=True)
        (wd / "fullwave3d.json").write_text(json.dumps(rec, indent=1))
    (ROOT / "docs" / "validation" / "fullwave3d.md").write_text(report(rec))
    (ROOT / "benchmarks" / "records").mkdir(exist_ok=True)
    (ROOT / "benchmarks" / "records" / "fullwave3d.json").write_text(json.dumps(rec, indent=1))
