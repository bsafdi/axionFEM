"""Cross-validation: full 3D solver vs the axisymmetric engine.

    python benchmarks/m2_cross.py solenoid [order] [-] [h1,h2,...]
    python benchmarks/m2_cross.py toroid [order] [slit_deg | ideal] [h1,h2,...]

Each case solves the same design (revolved; annular port for the solenoid's coax) with both engines and
records Psi, L_p, W, g and their relative differences over a sequence of 3D mesh sizes, so the 3D-vs-2D
differences can be read as discretization error.

Toroid: 'ideal' keeps the sheath closed and prescribes the readout current around the axis (3D pickup type
'azimuthal'), the exact 3D counterpart of the axisymmetric ideal slit. A slit angle instead cuts a real slit
and feeds the readout through a port region; that model differs from the ideal slit by about 1 % in Psi
(the port's current distribution).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import axionfem as af  # noqa: E402
from axionfem import _core  # noqa: E402
from axionfem.convert import to_3d  # noqa: E402

REC = ROOT / "benchmarks" / "records"
RUNS = ROOT / "runs" / "m2"


def compare(name: str, d2: af.Design, order3d: int = 2, slit: float | None = None, mesh3d: dict | None = None) -> dict:
    t = time.time()
    r2 = d2.solve(RUNS / f"{name}_2d", verbose=0)
    t2 = time.time() - t
    d3 = to_3d(d2, slit_deg=slit or 4.0, mesh=mesh3d, pickup="slit" if slit else "ideal")
    t = time.time()
    r3 = json.loads(_core.run_3d(json.dumps(d3.doc), str(RUNS / f"{name}_3d_p{order3d}"), order3d, 1))
    t3 = time.time() - t
    info = r3["mesh"].get("info", {})
    out = {"case": name, "slit_deg": slit, "order_3d": order3d, "seconds_2d": t2, "seconds_3d": t3,
           "dofs_3d": r3["mesh"]["dofs"], "elements_3d": r3["mesh"]["elements"],
           "open_boundary": "infinity shell" if info.get("shell_attr", -1) > 0 else "truncated (n x A = 0)",
           "domain_radius": info.get("domain_radius"), "timing_3d": r3.get("timing"),
           "solver_3d": r3.get("solver"), "warnings_3d": r3.get("warnings")}
    for key, k3 in (("Psi_SI_H_T_m2", "Psi_SI_H_T_m2"), ("L_p_H", "L_p_H"), ("W_mag_J", "W_mag_J"), ("g_factor", "g_factor")):
        a, b = r2.coupling[key], r3["coupling"][k3]
        out[key] = {"2d": a, "3d": b, "rel_diff": (b - a) / a if a else None}
    out["bodies_3d_static"] = r3["static"].get("bodies")
    out["bodies_3d_pickup"] = r3["pickup"].get("bodies")
    REC.mkdir(parents=True, exist_ok=True)
    (REC / f"m2_cross_{name}_p{order3d}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("bodies")}, indent=1), flush=True)
    return out


def solenoid(order: int = 2, sizes=(0.4, 0.28, 0.2)):
    d2 = af.template("solenoid", R=1.0, H=2.0, x=0.534, tw=0.05, g=0.03, tc=0.03, tp=0.05,
                     mesh={"order": 2, "size": 0.05, "surface_size": 0.015})
    return [compare(f"solenoid_h{h:g}", d2, order, mesh3d={"size": h, "surface_size": h / 3}) for h in sizes]


def toroid(order: int = 2, slit: float | None = None, sizes=(0.45, 0.32, 0.22)):
    d2 = af.template("toroid", R=1.0, H=1.0, tw=0.06, g=0.04, ts=0.05, f=0.06,
                     mesh={"order": 2, "size": 0.04, "surface_size": 0.015})
    tag = f"slit{slit:g}" if slit else "ideal"
    return [compare(f"toroid_{tag}_h{h:g}", d2, order, slit=slit, mesh3d={"size": h, "surface_size": h / 2.5})
            for h in sizes]


if __name__ == "__main__":
    case = sys.argv[1]
    order = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    sizes = tuple(float(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    if case == "solenoid":
        solenoid(order, *( [sizes] if sizes else []))
    elif case == "toroid":
        arg = sys.argv[3] if len(sys.argv) > 3 else "ideal"
        toroid(order, None if arg in ("ideal", "-") else float(arg), *([sizes] if sizes else []))
