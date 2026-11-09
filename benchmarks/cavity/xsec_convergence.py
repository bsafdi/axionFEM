"""h- and p-convergence of the cross-section solver ("symmetry": "xsec"; beta) against closed forms and the
independent method-of-particular-solutions reference. Writes benchmarks/records/cavity_xsec_convergence.json.
Usage: python benchmarks/cavity/xsec_convergence.py [workdir]"""
import json
import math
import sys
from pathlib import Path

from axionfem import cavity_xsec as cx

ROOT = Path(__file__).resolve().parents[2]
WD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/xsec_conv")
J01 = 2.404825557695773


def tm010(res):
    return max((r for r in res.cavity_modes() if r["family"] == "TM"), key=lambda r: r["C"])


def main():
    out = {"disk": [], "offcentre": []}
    R, L = 0.05, 0.25
    f_ex, C_ex, G2_ex, G_ex = 299792458.0 * J01 / (2 * math.pi * R), 4 / J01 ** 2, cx.Z0 * J01 / 2, cx.Z0 * J01 * L / (2 * (R + L))
    for order in (2, 3):
        for n in (8, 16, 32):
            res = cx.cylinder_rods_design(R, [], modes=1, h=R / n, length=L, order=order).solve(WD / f"d{order}_{n}", verbose=0)
            m = tm010(res)
            w = m["walls"]["wall"]
            out["disk"].append({"order": order, "h_over_R": 1 / n, "dofs": res.data["mesh"]["dofs"],
                                "f_rel": m["f_Hz"] / f_ex - 1, "C_rel": m["C"] / C_ex - 1, "G2d_rel": m["G_ohm_2d"] / G2_ex - 1,
                                "G2d_pointwise_rel": w["G_ohm_pointwise"] / G2_ex - 1, "G_caps_rel": m["G_ohm"] / G_ex - 1})
            print(out["disk"][-1], flush=True)
    Rw, a, dx = 0.2, 0.05, 0.06
    ref = cx.eccentric_rod_tm_mode(Rw, a, dx)
    out["offcentre_reference"] = {k: float(v) for k, v in ref.items()}
    for n in (10, 20, 40):
        res = cx.cylinder_rods_design(Rw, [{"radius": a, "center": [dx, 0.0]}], modes=2, h=Rw / n).solve(WD / f"e{n}", verbose=0)
        m = tm010(res)
        out["offcentre"].append({"h_over_R": 1 / n, "f_rel": m["f_Hz"] / ref["f_Hz"] - 1, "C_rel": m["C"] / ref["C"] - 1,
                                 "G_rod_rel": m["walls"]["rod0"]["G_ohm"] / ref["G2d_rod_ohm"] - 1,
                                 "G_wall_rel": m["walls"]["wall"]["G_ohm"] / ref["G2d_wall_ohm"] - 1})
        print(out["offcentre"][-1], flush=True)
    (ROOT / "benchmarks" / "records" / "cavity_xsec_convergence.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
