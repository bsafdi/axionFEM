"""Readout-port layouts on the reduced DMRadio-Core design: N ports around the slit, or one wider port, against the
published single 30-degree port; checks the port matrices, their bus combination and distributed ports.

    python benchmarks/core_readout_ports.py designs                    # write and validate the designs only
    python benchmarks/core_readout_ports.py run CONFIG [--ranks N] [--mesh fast|fine] [--points P]
    python benchmarks/core_readout_ports.py report

In the 3D Core comparison (docs/validation/published_benchmarks.md section 7.1) the published 30-degree readout port
sees the same axion EMF (Psi) as an azimuthally uniform feed and a 1.3-2.3x larger effective inductance. This script
compares other port layouts on the same design, with the exact dZ/dw, as a check of the multiport features.

Configurations (same reduced design and mesh as `published_benchmarks.py core3d_fast/core3d`):
  n1                the published single 30-degree port (reference)
  n2, n3, n4, n6    N equal 30-degree ports, equally spaced; the rest of the slit stays an open gap
  n12               tiles the slit; not run: with equal port currents it is, by symmetry, the 360-degree control
                    (runs/published/core3d_port360*), and it costs 12 solves per frequency
  w60, w90, w180    one wider port
  c360              the full-circumference port: the uniform-feed control (as published_benchmarks' 360-degree
                    variant), rerun so that its L_eff also uses the exact dZ/dw
The full-wave solve with the radiating exterior is AMS-GMRES (the STRUMPACK path needs a closed PEC boundary), one
solve per port and frequency, so the cost grows linearly with N. For N equally spaced identical ports Z is circulant
and 1 is an eigenvector, so equal currents give equal voltages: the bus combination below then also equals separate
equal capacitors at each port (no bus) for the axion-driven mode; readout loading at one port breaks the symmetry.
One 3D full-wave solve per configuration gives the N x N impedance matrix Z and the Psi vector at each frequency
(all ports oriented along +z, the current direction across the slit). The ports are combined as if wired in parallel
by ideal lossless leads, an upper bound on what real cabling around a 1.2 m ring achieves at 70-140 MHz:
    Y = Z^-1,   Z_eq = 1 / (1^T Y 1),   Psi_eq = (1^T Y Psi) / (1^T Y 1),   dZ_eq/dw = Z_eq^2 1^T Y Z' Y 1,
with Z' the per-frequency dZ/dw from the solver, L_eff = (w dX/dw + X) / (2 w) (X = -Im Z) and F_eff = |Psi_eq|^4 / L_eff^2,
as in published_benchmarks._fullwave_rows. n12 combined should come close to the 360-degree control.
Writes benchmarks/records/core_readout_ports.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HERE))

RUNS = ROOT / "runs" / "core_readout_ports"
OUT = ROOT / "benchmarks" / "records" / "core_readout_ports.json"
DESIGNS = ROOT / "benchmarks" / "published" / "core_readout_ports"
MESH = {"fast": (0.16, 0.025), "fine": (0.12, 0.025)}
CONFIGS = {"n1": (1, 30.0), "n2": (2, 30.0), "n3": (3, 30.0), "n4": (4, 30.0), "n6": (6, 30.0),
           "n12": (12, 30.0), "w60": (1, 60.0), "w90": (1, 90.0), "w180": (1, 180.0), "c360": (1, 360.0),
           "u2": (2, 30.0)}
# configurations whose N pieces form ONE distributed port (a port may be several disjoint gap
# pieces fed in parallel, the current split by conductance); u2 checks it against n2's bus combination
DISTRIBUTED = {"u2"}
TOWARD = [0.0, 0.0, 1.0]   # the readout current crosses the slit along z


def sectors(n: int, width: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Port and gap sectors (degrees, increasing) for n equal ports of `width`, the first centred on 0."""
    ports = []
    for k in range(n):
        c = 360.0 * k / n
        ports.append((c - width / 2, c + width / 2))
    gaps = []
    for k in range(n):
        a = ports[k][1]
        b = ports[(k + 1) % n][0] + (360.0 if k == n - 1 else 0.0)
        if b - a > 1e-9:
            gaps.append((a, b))
    return ports, gaps


def design(config: str, mesh: str, freqs: dict):
    import axionfem as af
    import published_benchmarks as P
    from axionfem.convert import to_3d
    n, width = CONFIGS[config]
    d2 = P.design_core_reduced()
    d3 = to_3d(d2)
    doc = d3.doc
    prof = next(r["shape"] for r in d2.doc["regions"] if r["name"] == "neck")
    regs = [r for r in doc["regions"] if r["name"] != "neck"]
    ports, gaps = sectors(n, width)
    if config in DISTRIBUTED:
        regs.append({"name": "neck_port0", "material": "vacuum",
                     "solid": {"union": [{"revolve": {"profile": prof, "angles": [a, b]}} for a, b in ports]},
                     "notes": f"one distributed readout port: {n} pieces of {width:g} degrees fed in parallel"})
    else:
        for k, (a, b) in enumerate(ports):
            angles = [0.0, 360.0] if b - a >= 360.0 else [a, b]   # the full port as published_benchmarks' 360 control
            regs.append({"name": f"neck_port{k}", "material": "vacuum",
                         "solid": {"revolve": {"profile": prof, "angles": angles}},
                         "notes": f"readout port {k}: {b - a:g} degrees of the slit"})
    for k, (a, b) in enumerate(gaps):
        regs.append({"name": f"neck_gap{k}", "material": "vacuum",
                     "solid": {"revolve": {"profile": prof, "angles": [a, b]}},
                     "notes": "open, undriven part of the slit"})
    doc["regions"] = regs
    doc["pickup"] = {"type": "port", "port": "neck_port0", "toward": TOWARD}
    h, hs = MESH[mesh]
    doc["mesh"] = {"order": 2, "size": h, "surface_size": hs}
    doc["fullwave"] = dict(freqs)
    if n > 1 and config not in DISTRIBUTED:
        doc["fullwave"]["ports"] = [{"name": f"port{k}", "region": f"neck_port{k}", "toward": TOWARD}
                                    for k in range(1, n)]
    doc["name"] = f"dmradio_core_reduced_120cm_3d_readout_{config}"
    doc["description"] = (d2.doc["description"].rstrip(".") + f"; revolved to 3D with {n} readout port(s) of "
                          f"{width:g} degrees, equally spaced, the rest of the slit open.")
    return af.Design(doc)


def band(points: int) -> dict:
    import published_benchmarks as P
    f1, f2 = P.SRC["dmradio_core"]["sensitivity_inputs"]["f_range_Hz"][0]
    return {"f_min": f1, "f_max": f2, "points": points, "boundary": "radiation"}


def write_designs(mesh: str = "fast", points: int = 4) -> None:
    DESIGNS.mkdir(parents=True, exist_ok=True)
    for c in CONFIGS:
        d = design(c, mesh, band(points))
        p = DESIGNS / f"{c}.json"
        p.write_text(json.dumps(d.doc, indent=1) + "\n")
        print("wrote", p, "ports:", 1 + len(d.doc["fullwave"].get("ports", [])), flush=True)


def run(config: str, ranks: int, mesh: str, points: int) -> None:
    d = design(config, mesh, band(points))
    wd = RUNS / f"{config}_{mesh}"
    t0 = time.time()
    try:
        d.solve(wd, verbose=1, mpi_ranks=ranks)
    except Exception as exc:  # noqa: BLE001  (results.json is written by the C++ solver)
        print("solve raised:", type(exc).__name__, exc, flush=True)
    print(f"{config}: {time.time() - t0:.0f} s", flush=True)


def combine(Z: np.ndarray, dZ: np.ndarray | None, Psi: np.ndarray) -> dict:
    """Ideal parallel combination of the ports (see the module docstring)."""
    Y = np.linalg.inv(Z)
    one = np.ones(len(Psi))
    ytot = one @ Y @ one
    Zeq = 1.0 / ytot
    Psi_eq = (one @ Y @ Psi) / ytot
    dZeq = (Zeq ** 2) * (one @ Y @ dZ @ Y @ one) if dZ is not None else None
    return {"Z": Zeq, "dZ": dZeq, "Psi": Psi_eq}


def _cmat(m) -> np.ndarray:
    return np.array(m["re"], dtype=float) + 1j * np.array(m["im"], dtype=float)


def rows(results: dict) -> list[dict]:
    out = []
    for s in results["fullwave"]["sweep"]:
        w = 2 * math.pi * s["f_Hz"]
        Z = _cmat({"re": s["Z_matrix_re_ohm"], "im": s["Z_matrix_im_ohm"]}) if "Z_matrix_re_ohm" in s else \
            np.array([[complex(s["Z_re_ohm"], s["Z_im_ohm"])]])
        psi = s.get("Psi_vector")
        Psi = (np.array(psi["re"]) + 1j * np.array(psi["im"])) if psi else np.array([complex(s["Psi_re"], s["Psi_im"])])
        dz = s.get("dZ_domega_H")
        dZ = _cmat(dz) if dz else None
        c = combine(Z, dZ, Psi)
        X = -c["Z"].imag
        dX = -c["dZ"].imag if c["dZ"] is not None else None
        Le = (w * dX + X) / (2 * w) if dX is not None else None
        out.append({"f_Hz": s["f_Hz"], "Z_eq_ohm": [c["Z"].real, c["Z"].imag], "L_H": X / w,
                    "L_eff_H": Le, "Psi_abs": abs(c["Psi"]),
                    "F_eff": (abs(c["Psi"]) ** 4 / Le ** 2) if (Le and Le > 0) else None,
                    "ports": len(Psi), "Psi_spread": float(np.ptp(np.abs(Psi)) / np.mean(np.abs(Psi)))})
    return out


def scan_time_yr(rr: list[dict]) -> float | None:
    """Band scan time of one configuration: the integrand of published_benchmarks.check_core3d (published Q, T
    and eta(omega), DFSZ), trapezoid in ln f over the solved points. The reduced design's absolute years are not
    the device's; ratios between configurations are the result."""
    import axionfem as af
    import published_benchmarks as P
    sic = P.SRC["dmradio_core"]["sensitivity_inputs"]
    base = dict(model="DFSZ", rho_DM=sic["rho_DM_GeV_cm3"][0], Q_axion=sic["Q_axion"][0],
                Theta95=sic["SNR"][0] ** 2)
    lf, inv = [], []
    for x in rr:
        if not x.get("F_eff"):
            return None
        b = {"name": "core", "Q": sic["Q"][0], "T": sic["T_K"][0], "eta_A": P.eta_A_at(x["f_Hz"])}
        req = af.requirement(b, live_time=1.0, f_min=x["f_Hz"] * 0.995, f_max=x["f_Hz"] * 1.005, **base)
        lf.append(math.log(x["f_Hz"]))
        inv.append(req["F_req_natural_T4_m10"] / (x["F_eff"] / P.MU0 ** 2) / math.log(1.005 / 0.995))
    return float(np.trapezoid(inv, lf))


def distributed_check() -> dict | None:
    """u2 (ONE port holding n2's two pieces; axionFEM splits the current by conductance) against n2 (two separate
    ports), per frequency: the ideal common bus Z_bus = 1/(1^T Z^-1 1), Psi_bus = Z_bus 1^T Z^-1 Psi, and the
    conductance split Z_split = w^T Z w, Psi_split = w^T Psi with w from n2's raw port conductances. For these two
    identical opposite pieces split and bus coincide; u2 - split is the mesh difference of the two region layouts."""
    pu, pn = RUNS / "u2_fast" / "results.json", RUNS / "n2_fast" / "results.json"
    if not (pu.exists() and pn.exists()):
        return None
    ru, rn = json.loads(pu.read_text()), json.loads(pn.read_text())
    G = np.array([p["raw_conductance_m"] for p in rn["fullwave"]["ports"]])
    w = G / G.sum()
    one = np.ones(len(w))
    out = {"w_from_n2_conductances": w.tolist(), "rows": []}
    for su, sn in zip(ru["fullwave"]["sweep"], rn["fullwave"]["sweep"]):
        Z = _cmat({"re": sn["Z_matrix_re_ohm"], "im": sn["Z_matrix_im_ohm"]})
        Psi = np.array(sn["Psi_vector"]["re"]) + 1j * np.array(sn["Psi_vector"]["im"])
        Y = np.linalg.inv(Z)
        Zbus = 1.0 / (one @ Y @ one)
        Psi_bus = Zbus * (one @ Y @ Psi)
        Zsplit, Psi_split = w @ Z @ w, w @ Psi
        Zu = complex(su["Z_re_ohm"], su["Z_im_ohm"])
        Psi_u = complex(su["Psi_re"], su["Psi_im"])
        out["rows"].append({"f_Hz": su["f_Hz"], "Z_u2_ohm": [Zu.real, Zu.imag], "Z_bus_ohm": [Zbus.real, Zbus.imag],
                            "Z_split_ohm": [Zsplit.real, Zsplit.imag],
                            "u2_vs_bus": abs(Zu - Zbus) / abs(Zbus), "u2_vs_split": abs(Zu - Zsplit) / abs(Zsplit),
                            "split_vs_bus": abs(Zsplit - Zbus) / abs(Zbus),
                            "Psi_u2_over_bus_minus_1": abs(Psi_u) / abs(Psi_bus) - 1,
                            "Psi_split_over_bus_minus_1": abs(Psi_split) / abs(Psi_bus) - 1})
    return out


def report() -> None:
    rec = {"check": "core_readout_ports", "date": time.strftime("%Y-%m-%d"),
           "reference": "c360_fast: the full-circumference port (uniform feed), also with the exact dZ/dw",
           "configs": {}, "scan_time_over_c360": {}}
    for wd in sorted(RUNS.glob("*_*")):
        res = wd / "results.json"
        if res.exists():
            rec["configs"][wd.name] = rows(json.loads(res.read_text()))
    ref = rec["configs"].get("c360_fast")
    t_ref = scan_time_yr(ref) if ref else None
    for name, rr in rec["configs"].items():
        line = []
        for r in rr:
            r0 = next((x for x in (ref or []) if abs(x["f_Hz"] - r["f_Hz"]) < 1), None)
            if r0 and r0.get("F_eff") and r.get("F_eff"):
                r["F_eff_over_c360"] = r["F_eff"] / r0["F_eff"]
                r["L_over_c360"] = r["L_H"] / r0["L_H"]
            line.append(f"{r['f_Hz'] / 1e6:.0f} MHz: L/c360 {r.get('L_over_c360', float('nan')):.3f}, "
                        f"F_eff/c360 {r.get('F_eff_over_c360', float('nan')):.3f}")
        t = scan_time_yr(rr)
        if t and t_ref:
            rec["scan_time_over_c360"][name] = t / t_ref
        print(name, " | ".join(line), f"| scan time x{rec['scan_time_over_c360'].get(name, float('nan')):.3f}")
    chk = distributed_check()
    if chk:
        rec["distributed_port_check"] = chk
        for r in chk["rows"]:
            print(f"u2 vs n2 at {r['f_Hz'] / 1e6:.0f} MHz: u2/bus {r['u2_vs_bus']:.1e}, split/bus {r['split_vs_bus']:.1e}, "
                  f"Psi u2/bus-1 {r['Psi_u2_over_bus_minus_1']:.1e}")
    OUT.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("wrote", OUT)


def main() -> None:
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    if a[0] == "designs":
        write_designs(opt("--mesh", "fast"), int(opt("--points", "4")))
    elif a[0] == "run":
        run(a[1], int(opt("--ranks", os.environ.get("SLURM_NTASKS", "16"))), opt("--mesh", "fast"),
            int(opt("--points", "4")))
    else:
        report()


if __name__ == "__main__":
    main()
