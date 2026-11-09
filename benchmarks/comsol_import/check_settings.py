#!/usr/bin/env python
"""Check what `axf import-comsol` recovers from the arXiv:2211.00008 .mph files against the settings recovered by
hand from the unpacked model files (MODEL_SETTINGS*.md in the directory given by --unpacked).  beta.

    python check_settings.py --unpacked DIR [--out axionfem_results/import_settings_check.json]

Runs both routes (mesh, replay) on the three files in a temporary directory and compares: emw domains, PEC
boundary sets, port boundaries and geometry (length along the current x width), Zref, the Je kind/value/domains,
material constants, study frequencies, and the region volumes against the exact CAD volumes.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PI = math.pi

# expectations transcribed from MODEL_SETTINGS.md / MODEL_SETTINGS_epoxy.md
TORUS = PI * (1.0 ** 2 - 0.5 ** 2) * 0.5 * 355 / 360
EXPECT = {
    "Shielded_Toroid/Small_Shield.mph": {
        "emw_domains": [1, 6], "pec": [121, 122, 124, 125], "port": [123], "Zref": 50.0,
        "port_length_m": 1.0 * math.radians(0.01), "port_width_m": 1e-3,
        "source": {"kind": "toroidal_box", "B0_T": 1.0, "domains": [6]},
        "materials": {}, "study_Hz": {"std2": (1e5, 1e7, 100), "std3": (3e8, 3e8, 1)},
        "volumes": {1: PI * 1.5 ** 2 * 1.0 - TORUS, 6: PI * (0.975 ** 2 - 0.525 ** 2) * 0.45 * 355 / 360}},
    "Toroid_With_Epoxy/Small_Shield.mph": {
        "emw_domains": [1, 3, 4], "pec": [61, 62, 64, 65], "port": [63], "Zref": 50.0,
        "port_length_m": 1.0 * math.radians(0.01), "port_width_m": 1e-3,
        "source": {"kind": "toroidal_box", "B0_T": 1.0, "domains": [3, 4]},
        "materials": {"eps_r_at_650MHz": 3.05333 - 6.667e-12 * 650e6, "tan_delta": 0.0021},
        "study_Hz": {"std5": (6.5e8, 6.5e8, 1)},
        "volumes": {1: PI * 1.5 ** 2 * 1.0 - TORUS,
                    3: PI * 2 * 0.75 * (0.45 ** 2 - 0.405 ** 2) * 355 / 360,
                    4: PI * 2 * 0.75 * 0.405 ** 2 * 355 / 360}},
    "Shielded_Solenoid/Model_File.mph": {
        "emw_domains": [1, 3, 4, 5], "pec": [74, 75, 76, 78, 81, 82, 83, 84], "port": [79, 80], "Zref": 50.0,
        "port_length_m": 5e-4, "port_width_m": 1e-3,
        "source": {"kind": "uniform", "B_T": [0, 0, 1], "domains": [5]},
        "materials": {}, "study_Hz": {"std2": (5e6, 25e6, 20001), "std3": (5e6, 5e6, 1)},
        "volumes": {5: PI * 0.5 ** 2 * 1.5, 4: PI * (0.755 ** 2 - 0.745 ** 2) * 0.025,
                    3: PI * (0.975 ** 2 - 0.525 ** 2) * 1.45,
                    1: PI * 1.5 ** 2 * 2.5 - PI * 1.0 ** 2 * 1.5}},
}


def check(unp: Path, rel: str, exp: dict, tmp: Path) -> dict:
    from axionfem import comsol_import as ci
    out = {"file": rel, "checks": []}

    def add(name, ok, got, want):
        out["checks"].append({"check": name, "ok": bool(ok), "got": got, "expected": want})

    d = ci.import_comsol(unp / rel, tmp / Path(rel).stem / "mesh" / "design.json", frequencies=[5e6])
    doc = d.doc if hasattr(d, "doc") else d
    rep = d.import_report.to_json()
    s = rep["summary"]
    add("emw domains", s["emw_domains"] == exp["emw_domains"], s["emw_domains"], exp["emw_domains"])
    bt = s["boundary_types"]
    pec_int = sorted(int(b) for b, t in bt.items() if t == "pec" and int(b) in exp["pec"])
    add("PEC sheet boundaries", pec_int == exp["pec"], pec_int, exp["pec"])
    port = s["port"]
    add("port boundaries", port["comsol_boundaries"] == exp["port"], port["comsol_boundaries"], exp["port"])
    add("port length (along the current)", abs(port["length_m"] / exp["port_length_m"] - 1) < 2e-3, port["length_m"],
        exp["port_length_m"])
    add("port width", abs(port["width_m"] / exp["port_width_m"] - 1) < 1e-6, port["width_m"], exp["port_width_m"])
    add("Zref", port["Zref_ohm"] == exp["Zref"], port["Zref_ohm"], exp["Zref"])
    src = doc["fullwave"]["source"]
    e = exp["source"]
    ok = src["kind"] == e["kind"] and src["metadata"]["domains"] == e["domains"]
    ok &= (abs(src.get("B0_T", 0) - e.get("B0_T", 0)) < 1e-12) if "B0_T" in e else src["B_T"] == e["B_T"]
    add("Je -> source", ok, {k: v for k, v in src.items() if k != "metadata"} | {"domains": src["metadata"]["domains"]}, e)
    if exp["materials"]:
        d650 = ci.import_comsol(unp / rel, tmp / Path(rel).stem / "m650" / "design.json", frequencies=[650e6])
        mats = (d650.doc if hasattr(d650, "doc") else d650)["materials"]
        m = next(iter(mats.values()))
        add("material eps' (650 MHz)", abs(m["eps_r"] - exp["materials"]["eps_r_at_650MHz"]) < 1e-4, m["eps_r"],
            exp["materials"]["eps_r_at_650MHz"])
        add("material tan delta", abs(m["tan_delta"] - exp["materials"]["tan_delta"]) < 1e-12, m["tan_delta"],
            exp["materials"]["tan_delta"])
    st = {x["study"]: (x["f_min_Hz"], x["f_max_Hz"], x["n"]) for x in s["studies"]}
    for k, v in exp["study_Hz"].items():
        got = st.get(k)
        add(f"study {k} frequencies", got is not None and abs(got[0] / v[0] - 1) < 1e-9 and abs(got[1] / v[1] - 1) < 1e-9
            and got[2] == v[2], got, v)
    vols = s["domain_volumes_m3"]
    for dom, v in exp["volumes"].items():
        got = vols[dom]["curved"]
        add(f"mesh route volume, domain {dom} (curved)", abs(got / v - 1) < 1e-4, got, v)
        out.setdefault("straight_volume_rel", {})[dom] = vols[dom]["straight"] / v - 1
    # replay route
    try:
        dr = ci.import_comsol(unp / rel, tmp / Path(rel).stem / "replay" / "design.json", route="replay",
                              frequencies=[5e6])
        rr = dr.import_report.to_json()
        add("replay route used", rr["route"] == "replay", rr["route"], "replay")
        rv = {r["domain"]: r["replay_volume_m3"] for r in rr["summary"].get("replay_volumes", [])}
        for dom, v in exp["volumes"].items():
            if dom in rv and rv[dom]:
                add(f"replay volume, domain {dom}", abs(rv[dom] / v - 1) < 1e-6, rv[dom], v)
        out["replay_regions"] = [r["name"] for r in (dr.doc if hasattr(dr, "doc") else dr)["regions"]]
    except Exception as ex:  # noqa: BLE001
        add("replay route", False, str(ex), "replay")
    out["ok"] = all(c["ok"] for c in out["checks"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unpacked", type=Path, required=True, help="directory of the unpacked COMSOL model files")
    ap.add_argument("--out", type=Path, default=HERE / "axionfem_results" / "import_settings_check.json")
    a = ap.parse_args()
    res = []
    with tempfile.TemporaryDirectory() as td:
        for rel, exp in EXPECT.items():
            r = check(a.unpacked, rel, exp, Path(td))
            res.append(r)
            print(f"== {rel}: {'OK' if r['ok'] else 'MISMATCH'}")
            for c in r["checks"]:
                print(f"   [{'ok' if c['ok'] else 'XX'}] {c['check']}: {c['got']} (expected {c['expected']})")
            if "straight_volume_rel" in r:
                print("   straight-sided volume error:", {k: f"{v:+.2e}" for k, v in r["straight_volume_rel"].items()})
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
