"""Cross-code reference for xcode_mhd_duct_2d (out of band): NGSolve 6.2.2608, written independently of axionFEM.

Discrete problem: the full 2.5D inductionless system (u in P2^3, p in P1, phi in P2) through the NGSolve solver
tools/fluids_mms/ngs_mhd.py on the fixed mesh builder f13a_mhd_check.duct_mesh, Shercliff and Hunt square ducts at
Ha = 100, levels k = 1, 2 (mu = 1/2, sigma = 2, G = 1, half-width 1).  Writes the reference JSON and the mesh arrays
for the second code (fenicsx_mhd_duct_check.py), which must agree to ~1e-12 relative (checked by
compare_mhd_duct.py).

    python -B tools/fluids_mms/xcode/ngsolve_mhd_duct_reference.py OUT.json MESHES.json
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "benchmarks" / "fluids" / "cases"))
import ngsolve as ngs  # noqa: E402
import ngs_mhd as M  # noqa: E402
import f13a_mhd_check as C  # noqa: E402

HA, LEVELS = 100.0, (1, 2)
MU, SIGMA, G, A_ = 0.5, 2.0, 1.0, 1.0      # as in the analytic MHD cases (tools/fluids_mms/f13a_reference.py)


def B0_of(Ha):
    return Ha / (A_ * math.sqrt(SIGMA / MU))


def duct_run(kind, Ha, k):
    """Fully developed duct flow driven by G e_z, B = B0 e_y, no-slip walls; Hunt: perfectly conducting walls
    y = -+1 (phi = 0), insulating x = -+1; Shercliff: all walls insulating."""
    B0 = B0_of(Ha)
    V, T = C.duct_mesh(Ha, k)
    mesh = M.ng_mesh(V, T, M.rect_label(-1.0, 1.0, -1.0, 1.0))
    z = ngs.CF(0.0)
    phi_dir = {"y0": z, "y1": z} if kind == "hunt" else {}
    pb = M.Problem(mu=MU, sigma=ngs.CF(SIGMA), B=[z, ngs.CF(B0), z], Ez=0.0, Gz=G,
                   u_dir={e: [z, z, z] for e in ("x0", "x1", "y0", "y1")}, phi_dir=phi_dir)
    gf, diag = M.solve(mesh, pb)
    ev = M.Evaluator(mesh, V, T, gf)
    integ = C.integrals(ev, SIGMA, [0.0, B0, 0.0], 0.0, MU)
    q = {"Q_h": integ["flow"][2], "w_center_h": C.point_value(ev, (0.0, 0.0), "w")}
    if kind == "hunt":
        q["phi_side_h"] = C.point_value(ev, (1.0, 0.0), "phi")
        q["M_var_h"] = C.wall_current_moment_variational(ev, SIGMA, [0.0, B0, 0.0])
    else:
        q["dphi_side_h"] = C.point_value(ev, (1.0, 0.0), "phi") - C.point_value(ev, (-1.0, 0.0), "phi")
    return q, diag, V, T


def main():
    out, meshes = {}, {}
    diag = {}
    for kind in ("shercliff", "hunt"):
        for k in LEVELS:
            q, d, V, T = duct_run(kind, HA, k)
            key = f"{kind}_Ha100_k{k}"
            qs = ["Q_h", "w_center_h"] + (["phi_side_h", "M_var_h"] if kind == "hunt" else ["dphi_side_h"])
            for name in qs:
                out[f"{key}_{name}"] = float(q[name])
            diag[key] = {"relative_residual": d["relative_residual"], "ndof": d["ndof"]}
            meshes[key] = {"kind": kind, "Ha": HA, "k": k, "V": V.tolist(), "T": T.tolist()}
            print(key, {name: out[f"{key}_{name}"] for name in qs}, flush=True)
    res = {"format": "axionfem.fluids.xcode_output/1", "code": "NGSolve", "version": ngs.__version__,
           "case": "xcode_mhd_duct_2d", "quantities": out, "diagnostics": diag}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    Path(sys.argv[2]).write_text(json.dumps({"meshes": meshes}) + "\n")


if __name__ == "__main__":
    main()
