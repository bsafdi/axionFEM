"""Model A of axionfem.heterodyne with the computed SRF cell geometry: volumes, power and cost of the benchmarks.

For each model-A benchmark (``heterodyne.SCENARIOS_A``: optimistic, conservative) and each computed geometry preset
of :mod:`axionfem.heterodyne`

* ``computed_wall`` (PROJECTX_086, the default computed preset: the figures of merit of the cached cell record,
  B_pk = the largest wall |B|, 1.464 B0) and
* ``equator_diagnostic`` (the same record with B_pk := the equator |B|, 1.404 B0; a diagnostic, not a field limit:
  the iris-side wall would then run 4.3 % above B_pk),

this computes the required volume V and from it f0, U0, P_RF, A_cav, Q_i, Q0, the cost (``heterodyne.cost_live``) and
the tuning fraction, with

* the weak-signal statistic (closed-form optimal coupling beta*, ``heterodyne.design_A``), and
* the exact Asimov statistic with beta re-optimised at every mass (``heterodyne.required_volume_exact``, n_f = 32).

    python benchmarks/srf/model_a_geometry.py     # about 3 min on 8 cores; the exact statistic is the heavy part

Writes benchmarks/srf/results/model_a_geometry.json, with the generation stamp and fingerprint of the geometry record
it was computed from (benchmarks/srf/srf_vibration_mixing.py and tests/test_heterodyne.py check them), and prints the
results as a markdown table.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from axionfem import heterodyne as H  # noqa: E402

OUT = ROOT / "benchmarks/srf/results/model_a_geometry.json"
GEOS = ("computed_wall", "equator_diagnostic")
STATS = ("weak", "exact")


def scenario(key, geo):
    return replace(H.SCENARIOS_A[key], geometry=H.GEOMETRIES[geo])


def derived(sc, V):
    target = H.Target()                          # the default scan target of design_A / required_volume_exact
    f0 = float(sc.geometry.f0_Hz(V))
    U0 = sc.B0_T ** 2 * V / (2 * H.MU0)
    P = 2 * math.pi * f0 * U0 / (sc.geometry.G_pump_ohm / sc.Rs(f0))
    A = float(sc.geometry.area_m2(V))
    return dict(V_m3=V, f0_Hz=f0, B0_T=sc.B0_T, U0_J=U0, P_RF_W=P, A_cav_m2=A, Q_i=sc.geometry.G_signal_ohm / sc.Rs(f0),
                Q0=sc.geometry.G_pump_ohm / sc.Rs(f0), cost_MCHF=H.cost_live(A, P, sc.T_K),
                tuning_fraction=(target.f_max_Hz - target.f_min_Hz) / f0)


def job(case):
    key, geo, stat = case
    sc = scenario(key, geo)
    if stat == "weak":
        d = H.design_A(sc)
        out = derived(sc, d["V_m3"])
        out.update(beta_range=d["beta_range"], peak_bin_snr_ends=d["peak_bin_snr_ends"],
                   adiabatic_violated_above_Hz=d["adiabatic_violated_above_Hz"])
    else:
        r = H.required_volume_exact(sc, exact=True, n_f=32)
        out = derived(sc, r["V_m3"])
        out.update(beta_ends=r["beta_ends"], peak_bin_snr_ends=r["peak_bin_snr_ends"])
    return key, geo, stat, out


def main():
    cases = [(k, g, s) for k in H.SCENARIOS_A for g in GEOS for s in STATS]
    with Pool(min(8, len(cases))) as pool:
        res = pool.map(job, cases)
    out = {"geometries": {g: {f: getattr(H.GEOMETRIES[g], f) for f in H.SRFGeometry.__dataclass_fields__}
                          for g in GEOS},
           "record": {k: json.loads(H.PROJECTX_RECORD.read_text())[k] for k in ("generated", "fingerprint")},
           "results": {}}
    for key, geo, stat, d in res:
        out["results"].setdefault(key, {}).setdefault(geo, {})[stat] = d
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")

    hdr = ("| benchmark | geometry | statistic | V [m³] | f0 [MHz] | Q_i | P_RF [kW] | A_cav [m²] "
           "| cost [MCHF] (Nb + cryostat + plant) |")
    print(hdr + "\n|" + "---|" * 9)
    for key in H.SCENARIOS_A:
        for geo in GEOS:
            for stat in STATS:
                d = out["results"][key][geo][stat]
                c = d["cost_MCHF"]
                print(f"| {key} | {geo} | {stat} | {d['V_m3']:.1f} | {d['f0_Hz'] / 1e6:.2f} | {d['Q_i']:.2g} | "
                      f"{d['P_RF_W'] / 1e3:.3g} | {d['A_cav_m2']:.0f} | **{c['total']:.1f}** ({c['niobium']:.1f} + "
                      f"{c['cryostat']:.1f} + {c['refrigerator']:.1f}) |")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
