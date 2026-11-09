"""Blind check of the cavity reproductions: axionFEM on four problems that were also solved independently (beta).

    python benchmarks/cavity/repro/blind_check.py

The independent values were obtained outside this repository, from the geometry and material specifications alone
(KLASH empty cylinder; QUAX 2004 hollow sapphire with flat end plates and the Table 1 one-parameter changes; QUAX 2410
single tube without gaps; HAYSTAC-like coax without gaps), by closed forms and independent numerical evaluation, without
sight of any axionFEM number. They are the constants in `INDEPENDENT` below (the boxed results of that derivation).
This script solves the same problems with the 2D axisymmetric solver (order 3) and compares number by number.
Writes benchmarks/records/cavity_repro_blind_check.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as cm  # noqa: E402

NAME = "blind_check"
MM = 1e-3
INDEPENDENT = {
    "P1": {"f_Hz": 57.3712639176e6, "C": 0.691660276123, "Q0_L4": 152819.864658, "Q0_L3.5": 145873.507174,
           "ratio": 1.04761904762},
    "P2": {"f_Hz": 10.8825727063e9, "C": 0.0409835344281, "p_e": 0.167773168717, "Q_wall": 1252432.53010,
           "end_share": 0.890744622863, "Q_diel": 2980214.32046, "Q0": 881840.014862,
           "Q0_tand_1e-5": 403848.206293, "Q0_tand_1e-7": 1226657.44281,
           "df_MHz": {"a": -143.981766716, "b": -13.3087193339, "c": -145.307173883, "d": -14.1232631538,
                      "e": -23.9332478850, "f": 170.407128754},
           "C_var": {"a": 0.0442616460243, "b": 0.0390603096033, "c": 0.0447967159331, "d": 0.0383266626548,
                     "e": 0.0413859366280, "f": 0.0384322113014}},
    "P3": {"f_Hz": 11.0432350630e9, "C": 0.336619555504, "f_empty_Hz": 4.02958833486e9, "C_empty": 0.691660276123,
           "Qwall_ratio": 4.30707975158},
    "P4": {"f_Hz": 5.86655844896e9, "C_vac": 0.806157079044, "C_full": 0.604617809283, "Q0": 13853.6675621},
}


def cyl(R, L):
    return [[0, -L / 2], [R, -L / 2], [R, L / 2], [0, L / 2]]


def tubes_design(tag, R, L, tubes, eps, *, wall, h, target, tand=None, order=3):
    diel = [{"name": f"tube{i}", "shape": {"rect": {"r": [r0, r1], "z": [-L / 2, L / 2]}}, "eps_r": eps,
             "tan_delta": tand, "mesh_size": h / 2} for i, (r0, r1) in enumerate(tubes)]
    return cm.design(tag, cyl(R, L), h=h, order=order, modes=6, families=("TM",), wall=wall, dielectrics=diel,
                     target_f_Hz=target, interior_point=[0.95 * R, 0.0])


def best(data, fmin, fmax, diel=()):
    ms = [m for m in cm.modes(data) if fmin < m["f_Hz"] < fmax]
    m = max(ms, key=lambda x: x["form_factor"]["C"])
    return cm.mode_row(data, m, diel_regions=diel), m


def main():
    jobs = []
    # P1
    for L in (4.0, 3.5):
        jobs.append((f"P1_L{L}", cm.design(f"P1_L{L}", cyl(2.0, L), h=2.0 / 8, order=3, modes=3, wall=cm.skin(5.8e7, 300))))
    # P2 baseline and variants (flat end plates)
    R2, L2 = 29 * MM, 180 * MM
    base = dict(r1=10.7, t1=1.9, r2=19.8, t2=1.6, eps=11.2)
    var = {"base": {}, "a": {"r1": 10.9}, "b": {"r2": 20.0}, "c": {"t1": 2.1}, "d": {"t2": 1.8}, "e": {"eps": 11.5},
           "f": {"eps": 9.39}}
    for k, ch in var.items():
        p = dict(base, **ch)
        tubes = [(p["r1"] * MM, (p["r1"] + p["t1"]) * MM), (p["r2"] * MM, (p["r2"] + p["t2"]) * MM)]
        jobs.append((f"P2_{k}", tubes_design(f"P2_{k}", R2, L2, tubes, p["eps"], wall=cm.const_Rs(5.5e-3),
                                             h=0.7 * MM, target=10.9e9, tand=2e-6)))
    # P3 tube mode and empty TM010 (same constant R_s)
    R3, L3 = 28.475 * MM, 420.39 * MM
    jobs.append(("P3_tube", tubes_design("P3_tube", R3, L3, [(10.866 * MM, (10.866 + 2.348) * MM)], 11.2,
                                         wall=cm.const_Rs(5.5e-3), h=0.7 * MM, target=11.0e9)))
    jobs.append(("P3_empty", cm.design("P3_empty", cyl(R3, L3), h=R3 / 8, order=3, modes=3, wall=cm.const_Rs(5.5e-3))))
    # P4 coax touching both end plates
    R4, a4, L4 = 50.8 * MM, 25.4 * MM, 254 * MM
    rod = [{"name": "rod", "shape": {"rect": {"r": [0.0, a4], "z": [-L4 / 2 - 0.01, L4 / 2 + 0.01]}}}]
    jobs.append(("P4", cm.design("P4", cyl(R4, L4), h=(R4 - a4) / 8, order=3, modes=3, wall=cm.skin(5.8e7, 300),
                                 conductors=rod, interior_point=[0.5 * (R4 + a4), 0.0], target_f_Hz=5.8e9)))
    out = {t: d for t, d, _ in cm.solve_many(jobs, NAME)}
    cmp = []

    def add(problem, q, independent, ours):
        cmp.append({"problem": problem, "quantity": q, "independent": independent, "axionFEM": ours,
                    "rel_diff": (ours / independent - 1) if independent else ours - independent})
    # P1
    for L in (4.0, 3.5):
        m = cm.modes(out[f"P1_L{L}"])[0]
        add("P1", f"f L={L}", INDEPENDENT["P1"]["f_Hz"], m["f_Hz"])
        add("P1", f"Q0 L={L}", INDEPENDENT["P1"][f"Q0_L{L:g}"], m["losses"]["Q0"])
        if L == 4.0:
            add("P1", "C", INDEPENDENT["P1"]["C"], m["form_factor"]["C"])
    q4, q35 = (cm.modes(out[f"P1_L{L}"])[0]["losses"]["Q0"] for L in (4.0, 3.5))
    add("P1", "Q0 ratio", INDEPENDENT["P1"]["ratio"], q4 / q35)
    # P2
    rows = {k: best(out[f"P2_{k}"], 10.5e9, 11.3e9, ("tube0", "tube1")) for k in var}
    b, mb = rows["base"]
    V2 = math.pi * R2 ** 2 * L2
    add("P2", "f", INDEPENDENT["P2"]["f_Hz"], b["f_Hz"])
    add("P2", "C", INDEPENDENT["P2"]["C"], b["C"] * b["V_m3"] / V2)
    add("P2", "p_e", INDEPENDENT["P2"]["p_e"], b["p_e"])
    add("P2", "Q_wall", INDEPENDENT["P2"]["Q_wall"], b["Q_wall"])
    add("P2", "Q_diel", INDEPENDENT["P2"]["Q_diel"], b["Q_dielectric"])
    add("P2", "Q0", INDEPENDENT["P2"]["Q0"], b["Q0"])
    for tand, tk in ((1e-5, "1e-5"), (1e-7, "1e-7")):
        add("P2", f"Q0 tand={tand:g}", INDEPENDENT["P2"][f"Q0_tand_{tk}"], 1 / (1 / b["Q_wall"] + tand * b["p_e"]))
    for k in "abcdef":
        r, _ = rows[k]
        add("P2", f"df({k}) MHz", INDEPENDENT["P2"]["df_MHz"][k], (r["f_Hz"] - b["f_Hz"]) / 1e6)
        add("P2", f"C({k})", INDEPENDENT["P2"]["C_var"][k], r["C"] * r["V_m3"] / V2)
    # P2 end-plate share: the barrel and the plates form one wall body; G from the closed form of a z-uniform mode:
    # P_ends / P_total = 2M / (L R u'(R)^2 + 2M) needs u'(R); not separable from the solver output (one body), so
    # the end share is checked indirectly: Q_wall(flat) with and without end loss is not available -> listed n/a.
    # P3
    t, _ = best(out["P3_tube"], 10.5e9, 11.5e9, ("tube0",))
    e = cm.modes(out["P3_empty"])[0]
    add("P3", "f tube mode", INDEPENDENT["P3"]["f_Hz"], t["f_Hz"])
    add("P3", "C tube mode", INDEPENDENT["P3"]["C"], t["C"])
    add("P3", "f empty TM010", INDEPENDENT["P3"]["f_empty_Hz"], e["f_Hz"])
    add("P3", "C empty TM010", INDEPENDENT["P3"]["C_empty"], e["form_factor"]["C"])
    add("P3", "Q_wall ratio tube/empty", INDEPENDENT["P3"]["Qwall_ratio"], t["Q_wall"] / e["losses"]["Q_wall"])
    # P4
    m = max(cm.modes(out["P4"]), key=lambda x: x["form_factor"]["C"])
    V4f = math.pi * R4 ** 2 * L4
    V4v = math.pi * (R4 ** 2 - a4 ** 2) * L4
    Vm = out["P4"]["fullwave"]["cavity"]["volume_m3"]
    add("P4", "f", INDEPENDENT["P4"]["f_Hz"], m["f_Hz"])
    add("P4", "C vacuum volume", INDEPENDENT["P4"]["C_vac"], m["form_factor"]["C"] * Vm / V4v)
    add("P4", "C full volume", INDEPENDENT["P4"]["C_full"], m["form_factor"]["C"] * Vm / V4f)
    add("P4", "Q0", INDEPENDENT["P4"]["Q0"], m["losses"]["Q0"])
    rec = {"benchmark": "blind independent check of the cavity reproductions: KLASH, QUAX 2004 flat ends + Table 1 changes, "
                        "QUAX 2410 no gap, coax", "comparison": cmp,
           "max_abs_rel_diff": max(abs(c["rel_diff"]) for c in cmp),
           "not_compared": ["P2 end-plate share of the wall loss (barrel and plates are one wall body in axionFEM)"]}
    cm.write_record(NAME, rec)
    for c in cmp:
        print(f"{c['problem']} {c['quantity']:28s} independent {c['independent']:.10g} axionFEM {c['axionFEM']:.10g} rel {c['rel_diff']:+.2e}")


if __name__ == "__main__":
    main()
