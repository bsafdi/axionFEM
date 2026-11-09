"""Stern et al. 1603.06990 (RSI 86, 123305) Table I multi-post designs A-G with the cross-section solver
("symmetry": "xsec"; beta, docs/validation/cavity_xsec.md).

R = 50 mm, d = 250 mm (end caps), copper sigma = 5.8e7 S/m (300 K, skin), posts spanning the full length.
N_stat posts at angles 360 k / N_stat on R_STAT, N_mov posts at phi + 360 k / N_mov on R_MOV; the rotation range and
phase are not published (refs gap), so phi is swept from the closest approach (0.5 mm clearance) to 180 / N_mov and
the comparison is made along Lambda = f R (GHz m), as the paper plots Q, C vs Lambda. TM010-like = the TM mode with
the largest C. C is reported both ways: C (vacuum area) and C_full (full cylinder pi R^2).

Reference: benchmarks/cavity/refs/stern_multipost_1603.json (digitized Fig. 6). Writes
benchmarks/records/cavity_xsec_stern.json and docs/images/cavity_xsec_stern.png.
Usage: python benchmarks/cavity/xsec_stern.py REFS_JSON [workdir]
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from axionfem import cavity_xsec as cx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REF = json.loads(Path(sys.argv[1]).read_text())
WD = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/xsec_stern")
R, D, SIGMA = 0.05, 0.25, 5.8e7
DESIGNS = {r[0]: r[1:] for r in REF["reference_values"]["tables"][0]["rows"]}


def posts(ns, nm, rmov, rstat, dia, phi):
    out = [{"radius": dia * R / 2, "center": [rstat * R * math.cos(2 * math.pi * k / ns),
                                               rstat * R * math.sin(2 * math.pi * k / ns)], "name": f"stat{k}"}
           for k in range(ns)]
    out += [{"radius": dia * R / 2, "center": [rmov * R * math.cos(phi + 2 * math.pi * k / nm),
                                                rmov * R * math.sin(phi + 2 * math.pi * k / nm)], "name": f"mov{k}"}
            for k in range(nm)]
    return out


def min_gap(ps):
    g = R - max(math.hypot(*p["center"]) + p["radius"] for p in ps)
    for i in range(len(ps)):
        for j in range(i):
            g = min(g, math.dist(ps[i]["center"], ps[j]["center"]) - ps[i]["radius"] - ps[j]["radius"])
    return g


def sweep_design(name):
    ns, nm, rmov, rstat, dia = DESIGNS[name]
    phis = np.linspace(0, math.pi / nm, 241)
    ok = [p for p in phis if min_gap(posts(ns, nm, rmov, rstat, dia, p)) > 5e-4]
    phis = np.linspace(min(ok), max(ok), 14)
    rows = []
    for k, phi in enumerate(phis):
        d = cx.cylinder_rods_design(R, posts(ns, nm, rmov, rstat, dia, phi), name=f"stern_{name}", modes=6,
                                    wall_rf=cx.skin_rf(SIGMA, 300.0), length=D, h=R / 30, surface_size=R / 150)
        res = d.solve(WD / name / f"{k:02d}", verbose=0)
        m = max((r for r in res.cavity_modes() if r["family"] == "TM"), key=lambda r: r["C"])
        rows.append({"phi_deg": math.degrees(phi), "Lambda": m["f_Hz"] * R / 1e9, "f_Hz": m["f_Hz"], "C": m["C"],
                     "C_full": m["C_full"], "Q0": m["Q0"], "Q0_2d": m["Q0_2d"], "gap_m": min_gap(posts(ns, nm, rmov, rstat, dia, phi))})
    return rows


def digitized(panel_src, label):
    for c in REF["reference_values"]["curves"]:
        if c["source"] == panel_src:
            for s in c["series"]:
                if s["label"] == label:
                    return [p for p in s["points"] if p[1] is not None]
    return []


def compare(rows, pts, key):
    L = np.array([r["Lambda"] for r in rows])
    o = np.argsort(L)
    out = []
    for x, y in pts:
        if L.min() <= x <= L.max():
            out.append({"Lambda": x, "paper": y, "ours": float(np.interp(x, L[o], np.array([r[key] for r in rows])[o]))})
    return out


def main():
    WD.mkdir(parents=True, exist_ok=True)
    rec = {"source": "1603.06990 Table I, Fig. 6 (digitized)", "designs": {}}
    panels = {"A": "Fig. 6a", "B": "Fig. 6a", "C": "Fig. 6a", "D": "Fig. 6b", "E": "Fig. 6b", "F": "Fig. 6b", "G": "Fig. 6b"}
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    for i, name in enumerate("ABCDEFG"):
        rows = sweep_design(name)
        pc = digitized(f"{panels[name]} (middle panel)", f"design {name}")
        pq = digitized(f"{panels[name]} (top panel)", f"design {name}")
        cmp_c = compare(rows, pc, "C")
        cmp_cf = compare(rows, pc, "C_full")
        cmp_q = compare(rows, pq, "Q0")
        rms = lambda c: float(np.sqrt(np.mean([(e["ours"] / e["paper"] - 1) ** 2 for e in c]))) if c else None  # noqa: E731
        rec["designs"][name] = {"table_I": DESIGNS[name], "sweep": rows, "C_vs_paper": cmp_c, "C_full_vs_paper": cmp_cf,
                                "Q0_vs_paper": cmp_q, "rms_rel_C": rms(cmp_c), "rms_rel_C_full": rms(cmp_cf),
                                "rms_rel_Q0": rms(cmp_q),
                                "Lambda_range_ours": [min(r["Lambda"] for r in rows), max(r["Lambda"] for r in rows)],
                                "Lambda_range_paper": [min(p[0] for p in pc), max(p[0] for p in pc)] if pc else None}
        print(name, {k: v for k, v in rec["designs"][name].items() if k.startswith("rms") or "range" in k}, flush=True)
        col = f"C{i}"
        ax = axs[0, 0] if name in "ABC" else axs[0, 1]
        L = [r["Lambda"] for r in rows]
        ax.plot(L, [r["C"] for r in rows], "-", color=col, label=f"{name} C (vac)")
        ax.plot(L, [r["C_full"] for r in rows], "--", color=col, label=f"{name} C (full)")
        if pc:
            ax.plot(*zip(*pc), "o", color=col, mfc="none")
        ax2 = axs[1, 0] if name in "ABC" else axs[1, 1]
        ax2.plot(L, [r["Q0"] for r in rows], "-", color=col, label=name)
        if pq:
            ax2.plot(*zip(*pq), "o", color=col, mfc="none")
    for ax in axs.ravel():
        ax.set_xlabel("Λ = f R [GHz m]")
        ax.legend(fontsize=7)
    axs[0, 0].set_ylabel("C"); axs[1, 0].set_ylabel("Q0")
    fig.suptitle("Stern et al. 1603.06990 multi-post designs: axionFEM xsec (lines) vs digitized COMSOL (circles); "
                 "beta", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "cavity_xsec_stern.png", dpi=140)
    (ROOT / "benchmarks" / "records" / "cavity_xsec_stern.json").write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
