"""MCP tools for the cavity haloscope, SRF heterodyne, dc SQUID readout and COMSOL modules.

Thin wrappers over the Python APIs (``axionfem.haloscope``, ``axionfem.cavity``, ``axionfem.heterodyne``,
``axionfem.srf``, ``axionfem.squid``, ``axionfem.comsol``, ``axionfem.comsol_import``,
``axionfem.comsol_model``): JSON in, JSON out.  :func:`register` adds them to the server in
``axionfem.mcp_server``; the plain functions below (``tool_*``) take the workspace explicitly so that
they can be tested without an MCP client.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

GUIDE = """\
Other detector types (beta modules: say so when you report their results; no field solve unless stated)
- Cavity haloscopes (beta): cavity_presets() lists published experiments (ADMX, HAYSTAC, CAPP, TASEH, QUAX,
  ORGAN) with their cavity, receiver and analysis numbers; cavity_preset_limit(name) recomputes a
  published exclusion and the scan rate it implies.  cavity_scan(f_Hz, B_T, V_m3, C, Q0, receiver)
  gives the scan rate df/dt for a DFSZ/KSVZ axion (or a given g), the optimal coupling beta, the
  coupling excluded at a given scan rate and the live time to cover a band (haloscope coupling
  convention and rho = 0.45 GeV/cm^3 by default, as experiments quote them).  receiver: exactly one of
  {"T_sys_K"}, {"N_A"} (added noise in quanta; 0.5 = standard quantum limit) or {"T_A_K"}, plus
  "T_cav_K" (physical temperature).  create_cavity(name, shape, ...) writes a pillbox or coaxial
  cavity design; solve_design(name) then returns its eigenmodes under cavity_modes (f, form factor C,
  volume, geometry factor G, Q0 when the walls have a surface resistance).
- SRF heterodyne (beta): srf_design(scenario) sizes a superconducting pump/signal cavity for a scan target
  (default: DFSZ over 0.5-4.1 neV in 3 years, an illustrative target; f_min_Hz, f_max_Hz, live_time_yr set another): volume, pump frequency, quality factor, RF power, wall area and
  a cost metric (a scaling metric for comparing designs, not a construction estimate).  Its scenarios
  'optimistic' and 'conservative' are illustrative presets: override their parameters for a real
  design.  srf_mode_pair(pair) gives the closed-form figures of merit of a degenerate mode pair in a
  cylinder (overlap eta_10, xi, peak-field ratios, geometry factors).
- dc SQUID readout (beta): squid_chain(preset, f_Hz) gives the noise of a two-stage dc SQUID chain:
  eta = added noise in units of the quantum limit, its uncorrelated part, the referred noise
  breakdown and, with Q and T_K, the scan-optimal coupling.  squid_reach(name, preset, Q, T_K) uses
  that chain as the amplifier of a solved lumped-element design and returns its reach g_95(m_a)
  ('+cryo' chains are specified from 5 MHz up: give f_min_Hz).
- COMSOL: export_comsol(name, format) writes a design as a COMSOL mesh (.mphtxt, named selections)
  or a COMSOL Java model (.java; compile and run it with COMSOL).  inspect_comsol(path) summarises a
  .mph file (parameters, geometry, materials, physics, studies) and import_comsol(path, name) (beta) turns
  it into a design here (read 'report': every COMSOL feature and how it was mapped).  No COMSOL
  installation is needed for any of these.
- Touchstone: export_touchstone(name, kind, z0, dense, surface_losses) writes a solved design's full-wave port
  parameters (S, Z or Y; exp(+j omega t), the engine's Z conjugated) as
  runs/<name>/touchstone/<name>[_dense]_<kind>.sNp, with the passivity (sigma_max of S) and reciprocity checks.
"""


# ---------------------------------------------------------------- workspace
class Workspace:
    """The parts of the MCP server's workspace that the module tools use (designs/<name>.json, runs/<name>/)."""

    def __init__(self, root):
        self.workspace = Path(root)

    @property
    def designs(self) -> Path:
        return self.workspace / "designs"

    @property
    def runs(self) -> Path:
        return self.workspace / "runs"

    def check_name(self, name: str) -> Path:
        import re

        if not re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$", name):
            raise ValueError(f"invalid design name {name!r}: use letters, digits, '_', '-', '.'")
        return self.designs / f"{name}.json"

    def load(self, name: str):
        from . import load

        p = self.check_name(name)
        if not p.exists():
            have = sorted(x.stem for x in self.designs.glob("*.json"))
            raise FileNotFoundError(f"no design named {name!r}; existing designs: {have}")
        return load(p)

    def save(self, name: str, d) -> dict:
        p = self.check_name(name)
        d.doc["name"] = name
        info = d.validate()
        p.parent.mkdir(parents=True, exist_ok=True)
        d.save(p)
        return info


# ---------------------------------------------------------------- JSON conversion
def jsonable(x: Any) -> Any:
    """Plain JSON types: numpy -> lists/floats, complex -> {"re", "im"}, dataclasses -> dicts,
    tuples -> lists, paths -> strings, non-finite floats -> None."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return jsonable({f.name: getattr(x, f.name) for f in dataclasses.fields(x)
                         if not callable(getattr(x, f.name))})
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (complex, np.complexfloating)):
        return {"re": jsonable(float(x.real)), "im": jsonable(float(x.imag))}
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, str):
        return x
    if callable(x):
        return getattr(x, "__name__", "function")
    return str(x)


def _call(fn: Callable, *a, **kw) -> tuple[Any, list[str]]:
    """Run fn, returning its value and the text of any Python warnings it raised."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(x.message) for x in w]


# ---------------------------------------------------------------- cavity haloscopes
def tool_cavity_presets() -> dict:
    from . import haloscope as hs
    from .haloscope_presets import REPRODUCIBLE

    rows = []
    for name in sorted(hs.PRESETS):
        p = hs.PRESETS[name]
        pub = p["published"]
        rows.append({"name": name, "experiment": p.get("experiment"), "run": p.get("run"),
                     "f_range_Hz": list(p["f_range_Hz"]), "cavity": p["cavity"], "receiver": p["receiver"],
                     "analysis": {k: v for k, v in p["analysis"].items() if k != "sources"},
                     "published": dict(pub), "refs": p.get("refs"), "notes": p.get("notes"),
                     "sources": p.get("sources"), "reproducible": name in REPRODUCIBLE})
    return {"ok": True, "presets": jsonable(rows),
            "units": "cavity V_L in litres, B_T tesla, Q0 or Q_L unloaded/loaded quality factor; receiver "
                     "temperatures in K, N_A in quanta",
            "note": "the numbers are read from each publication; 'sources' says where each one comes from and which "
                    "are inferred; reproducible = false: the publication does not give enough to recompute the limit"}


def tool_cavity_preset_limit(name: str, f_Hz: float | None = None) -> dict:
    from . import haloscope as hs

    if name not in hs.PRESETS:
        raise ValueError(f"unknown preset {name!r}; available: {sorted(hs.PRESETS)}")
    out: dict = {"ok": True, "name": name}
    try:
        out.update(hs.preset_limit(name, f_Hz))
    except Exception as e:  # noqa: BLE001  (e.g. a run whose scan rate is not published)
        out["limit_error"] = str(e)
    try:
        out["implied_scan"] = hs.implied_rate(name, f_Hz)
    except Exception as e:  # noqa: BLE001
        out["implied_scan_error"] = str(e)
    out["note"] = ("g_GeV is the exclusion recomputed from the preset's cavity, receiver and analysis numbers; "
                   "ratio = g_GeV / g_published_GeV; implied_scan is the scan rate (and live time over the "
                   "preset's band) at which the published coupling is reached")
    return jsonable(out)


def tool_cavity_scan(f_Hz: float, B_T: float, V_m3: float, C: float, Q0: float, receiver: dict,
                     beta: float | None = None, eta: float = 1.0, model: str = "DFSZ",
                     g_GeV: float | None = None, snr: float = 5.0, rate_Hz_s: float | None = None,
                     f_stop_Hz: float | None = None, duty_cycle: float = 1.0, rho_GeV_cm3: float = 0.45,
                     convention: str = "haloscope") -> dict:
    from . import haloscope as hs

    if g_GeV is not None and not g_GeV > 0:
        raise ValueError("g_GeV must be a positive coupling in GeV^-1 (or leave it out to use `model`)")
    if model not in ("DFSZ", "KSVZ"):
        raise ValueError("model must be 'DFSZ' or 'KSVZ'")
    if convention not in ("haloscope", "qcd"):
        raise ValueError("convention must be 'haloscope' (g_gamma = -0.97 / 0.36, Lambda = 77.6 MeV) or 'qcd' "
                         "(E/N with C_gamma = 1.77)")
    for k, v in dict(f_Hz=f_Hz, B_T=B_T, V_m3=V_m3, C=C, Q0=Q0, snr=snr, rho_GeV_cm3=rho_GeV_cm3).items():
        if not v > 0:
            raise ValueError(f"{k} must be positive (got {v})")
    if beta is not None and not beta > 0:
        raise ValueError(f"beta (antenna coupling) must be positive (got {beta}); leave it out for the optimum")
    if not 0 < eta <= 1 or not 0 < duty_cycle <= 1:
        raise ValueError("eta and duty_cycle are fractions in (0, 1]")
    if rate_Hz_s is not None and not rate_Hz_s > 0:
        raise ValueError("rate_Hz_s must be positive")
    if len([k for k in ("T_sys_K", "N_A", "T_A_K") if receiver.get(k) is not None]) != 1:
        raise ValueError("receiver: give exactly one of T_sys_K, N_A (added noise in quanta) or T_A_K")
    rx = hs.Receiver(**receiver)
    m = hs.freq_to_mass_eV(f_Hz)

    def g_at(f):
        return float(g_GeV) if g_GeV is not None else float(hs.g_axion(hs.freq_to_mass_eV(f), model,
                                                                         convention=convention))

    g = g_at(f_Hz)
    probe = hs.Cavity(B_T=B_T, V_m3=V_m3, C=C, Q0=Q0, beta=1.0, eta=eta)
    b_opt = hs.beta_opt(f_Hz, probe, rx, snr=snr, g_GeV=g, rho_GeV_cm3=rho_GeV_cm3)
    b = b_opt if beta is None else float(beta)
    cav = hs.Cavity(B_T=B_T, V_m3=V_m3, C=C, Q0=Q0, beta=b, eta=eta)
    rate = hs.scan_rate(f_Hz, cav, rx, g_GeV=g, snr=snr, rho_GeV_cm3=rho_GeV_cm3)
    out: dict = {"ok": True, "f_Hz": f_Hz, "m_a_eV": m, "g_GeV": g,
                 "g_source": "given" if g_GeV is not None else f"{model} ({convention} convention)",
                 "rho_GeV_cm3": rho_GeV_cm3,
                 "beta": b, "beta_optimal": b_opt, "Q_L": Q0 / (1 + b),
                 "signal_power_W": hs.signal_power(g, f_Hz, B_T, V_m3, C, Q0=Q0, beta=b,
                                                   rho_GeV_cm3=rho_GeV_cm3) * eta * rx.eta_out,
                 "scan_rate_Hz_s": rate, "scan_rate_MHz_per_day": rate * 86400 / 1e6, "snr": snr}
    if rate_Hz_s is not None:
        out["g_limit_at_rate_GeV"] = hs.g_limit_scan(f_Hz, cav, rx, rate_Hz_s=rate_Hz_s, snr=snr,
                                                     rho_GeV_cm3=rho_GeV_cm3)
        out["g_limit_over_g"] = out["g_limit_at_rate_GeV"] / g
    if f_stop_Hz is not None:
        f1, f2 = min(f_Hz, f_stop_Hz), max(f_Hz, f_stop_Hz)
        fs = np.geomspace(f1, f2, 60)
        rates = np.array([hs.scan_rate(fi, cav, rx, g_GeV=g_at(fi), snr=snr, rho_GeV_cm3=rho_GeV_cm3)
                          for fi in fs])
        t_s = float(np.trapezoid(1.0 / rates, fs))
        out["cover"] = {"f1_Hz": f1, "f2_Hz": f2, "live_time_s": t_s, "live_time_days": t_s / 86400.0,
                        "live_time_yr": t_s / (365.25 * 86400.0), "duty_cycle": duty_cycle,
                        "elapsed_days": t_s / duty_cycle / 86400.0,
                        "note": "live time = integration time; elapsed = live / duty_cycle. Cavity parameters held "
                                "fixed across the band; beta fixed at the value above"}
    return jsonable(out)


def tool_create_cavity(ctx, name: str, shape: str = "pillbox", a: float = 0.1, L: float = 0.2,
                       b: float | None = None, Bz: float = 1.0, modes: int = 6, wall: str = "lossless",
                       wall_T_K: float = 0.15, wall_rf: dict | None = None, target_f_Hz: float | None = None,
                       order: int = 2) -> dict:
    from . import cavity as cv

    if wall_rf is not None:
        wm = {"kind": "conductor", "rf": dict(wall_rf)}
    elif wall == "lossless":
        wm = None
    elif wall == "copper":
        wm = {"kind": "conductor", "rf": {"model": "anomalous", "RRR": 100, "rho_300K_ohm_m": 1.725e-8,
                                          "rho_l_ohm_m2": 6.6e-16, "specularity": 0.0, "T_K": float(wall_T_K)}}
    else:
        raise ValueError("wall must be 'lossless' or 'copper', or give wall_rf (an rf block of the design format)")
    kw = dict(Bz=Bz, modes=modes, target_f_Hz=target_f_Hz, order=order, wall_material=wm)
    if shape == "pillbox":
        d = cv.pillbox_design(a=a, L=L, **kw)
    elif shape == "coax":
        if b is None:
            raise ValueError("a coaxial cavity needs b (outer radius) > a (rod radius)")
        d = cv.coax_design(a=a, b=b, L=L, **kw)
    else:
        raise ValueError("shape must be 'pillbox' or 'coax'")
    info = ctx.save(name, d)
    return {"ok": True, "name": name, "shape": shape, "resolved_parameters": info.get("parameters"),
            "next": "solve_design(name) -> cavity_modes"}


# ---------------------------------------------------------------- SRF heterodyne
_SRF_GEOMETRY_ALIASES = {"projectx": "computed_wall", "equator_diagnostic": "equator_diagnostic"}


def tool_srf_design(scenario: str = "optimistic", geometry: str = "projectx", Rs_ohm: float | None = None,
                    Bpk_T: float | None = None, A: float | None = None, T_K: float | None = None,
                    f_min_Hz: float | None = None, f_max_Hz: float | None = None,
                    live_time_yr: float | None = None, refrigerator_factor: float | None = None) -> dict:
    from . import heterodyne as het

    lo = f_min_Hz if f_min_Hz is not None else het.DEFAULT_TARGET.f_min_Hz
    hi = f_max_Hz if f_max_Hz is not None else het.DEFAULT_TARGET.f_max_Hz
    if not 0 < lo < hi:
        raise ValueError(f"the scan band needs 0 < f_min_Hz < f_max_Hz (got {lo}, {hi})")
    if T_K is not None and refrigerator_factor is None and T_K not in (2.0, 4.5):
        raise ValueError("the refrigerator cost is tabulated at 2 K and 4.5 K only: for another T_K also give "
                         "refrigerator_factor (power drawn at room temperature per watt at T_K, relative to 4.5 K)")

    if scenario not in het.SCENARIOS_A:
        raise ValueError(f"scenario must be one of {sorted(het.SCENARIOS_A)}")
    if geometry not in _SRF_GEOMETRY_ALIASES:
        raise ValueError(f"geometry must be one of {sorted(_SRF_GEOMETRY_ALIASES)}")
    geo_key = _SRF_GEOMETRY_ALIASES[geometry]
    sc = het.SCENARIOS_A[scenario]
    over = {k: v for k, v in dict(Rs_ohm=Rs_ohm, Bpk_T=Bpk_T, A=A, T_K=T_K,
                                  refrigerator_factor=refrigerator_factor).items() if v is not None}
    sc = dataclasses.replace(sc, geometry=het.GEOMETRIES[geo_key], name=scenario, **over)
    tgt = dataclasses.replace(het.DEFAULT_TARGET, **{k: v for k, v in dict(
        f_min_Hz=f_min_Hz, f_max_Hz=f_max_Hz, live_time_yr=live_time_yr).items() if v is not None})
    d, warn = _call(het.design_A, sc, tgt)
    d["scenario"] = scenario
    if d.get("adiabatic_violated_above_Hz") is not None:
        warn.append(f"above {d['adiabatic_violated_above_Hz']:.4g} Hz the optimal scan is faster than the adiabatic "
                    "condition allows (the sizing does not impose it): the volume there is optimistic")
    if max(d.get("peak_bin_snr_ends") or [0.0]) > 1:
        warn.append(f"peak per-bin signal/noise reaches {max(d['peak_bin_snr_ends']):.3g}: the weak-signal "
                    "statistics behind the sizing are approximate there")
    g = sc.geometry
    out = {"ok": True, "inputs": {"scenario": scenario, "geometry": geometry, "Rs_ohm": sc.Rs_ohm,
                                  "Bpk_T": sc.Bpk_T, "A_noise_quanta": sc.A, "T_K": sc.T_K,
                                  "target": tgt},
           "geometry_figures": {"xi": g.xi, "eta": g.eta, "G_pump_ohm": g.G_pump_ohm,
                                "G_signal_ohm": g.G_signal_ohm, "Bpk_over_B0": g.Bpk_over_B0},
           "design": d, "warnings": warn + ([
               "equator_diagnostic uses the pump field at the equator, not the peak wall field: it understates "
               "B_pk and lets the pump exceed the field limit"] if geo_key == "equator_diagnostic" else []),
           "geometries": ["projectx", "equator_diagnostic"]}
    return jsonable(out)


def tool_srf_mode_pair(pair: str = "TM020/TE011") -> dict:
    from . import srf

    if pair not in srf.CYLINDER_PAIRS:
        raise ValueError(f"pair must be one of {sorted(srf.CYLINDER_PAIRS)}")
    return jsonable({"ok": True, **srf.cylinder_pair(pair)})


# ---------------------------------------------------------------- dc SQUID chains
def tool_squid_chain(preset: str = "C1+low+RT", f_Hz: list[float] | float = 1e6, Q: float | None = None,
                     T_K: float | None = None, T2_K: float = 1.0) -> dict:
    from . import squid

    if preset not in squid.presets():
        raise ValueError(f"preset must be one of {squid.presets()}")
    s, warn = _call(lambda: squid.chain(preset, T2_K=T2_K).summary(f_Hz, Q=Q, T_K=T_K))
    return jsonable({"ok": True, "preset": preset, **s, "warnings": warn,
                     "note": "eta: added noise of the chain relative to the quantum limit (input-referred, "
                             "correlations included); eta_uncorrelated ignores the correlation"})


def tool_squid_reach(ctx, name: str, preset: str = "C1+low+RT", Q: float = 1e6, T_K: float = 0.03,
                     amplitude: str = "as_designed", n: int = 6, f_min_Hz: float = 120.9e3,
                     f_max_Hz: float = 1.0e6, T2_K: float = 1.0) -> dict:
    import json as _json

    from . import scan, squid

    ctx.check_name(name)
    res_path = ctx.runs / name / "results.json"
    if not res_path.exists():
        raise FileNotFoundError(f"design {name!r} has no results; call solve_design({name!r}) first")
    if amplitude != "as_designed":
        raise ValueError("amplitude must be 'as_designed' (the reach of the solved drive)")
    if preset not in squid.presets():
        raise ValueError(f"preset must be one of {squid.presets()}")
    data = _json.loads(res_path.read_text())
    solved = (data.get("design") or {}).get("parameters") or {}
    try:
        now = ctx.load(name).resolved_params()
    except FileNotFoundError:
        now = solved
    stale = [k for k in set(now) | set(solved)
             if not (isinstance(now.get(k), (int, float)) and isinstance(solved.get(k), (int, float))
                     and math.isclose(now[k], solved[k], rel_tol=1e-12, abs_tol=0.0)) and now.get(k) != solved.get(k)]
    if stale:
        raise ValueError(f"the results of {name!r} are older than the design (changed: {sorted(stale)}); "
                         f"call solve_design({name!r}) again")
    c = data.get("coupling") or {}
    Psi, L = abs(float(c.get("Psi_SI_H_T_m2") or 0.0)), float(c.get("L_p_H") or 0.0)
    if Psi == 0.0 or L <= 0.0:
        raise ValueError(f"design {name!r} has no lumped pickup signal (Psi = {Psi}, L_p = {L}); squid_reach needs "
                         "a lumped-element detector")
    r, warn = _call(lambda: scan.reach(Psi, L, squid.benchmark(squid.chain(preset, T2_K=T2_K), Q=Q, T_K=T_K),
                                       n=n, f_min=f_min_Hz, f_max=f_max_Hz))
    return jsonable({"ok": True, "name": name, "preset": preset, "Psi_SI_H_T_m2": Psi, "L_p_H": L,
                     "solved_parameters": solved, "reach": r, "warnings": warn})


# ---------------------------------------------------------------- COMSOL
def tool_export_comsol(ctx, name: str, format: str = "mphtxt", order: int = 1, physics: str = "none") -> dict:
    from .comsol import export_comsol

    d = ctx.load(name)   # checks the name
    ext = {"mphtxt": ".mphtxt", "java": ".java", "step": ".step"}
    if format not in ext:
        raise ValueError("format must be 'mphtxt', 'java' or 'step'")
    outdir = ctx.runs / name / "comsol"
    outdir.mkdir(parents=True, exist_ok=True)
    import re

    stem = name
    if format == "java":   # the file name is the Java class name: letters, digits, '_', not starting with a digit
        stem = re.sub(r"[^A-Za-z0-9_]", "_", name)
        stem = stem if stem[0].isalpha() or stem[0] == "_" else f"M{stem}"
    out = outdir / f"{stem}{ext[format]}"
    summary = export_comsol(d, out, format=format, order=order, workdir=outdir / "mesh", physics=physics)
    return jsonable({"ok": True, "name": name, "file": out, "summary": summary})


# ---------------------------------------------------------------- Touchstone
def tool_export_touchstone(ctx, name: str, kind: str = "S", z0: float = 50.0, dense: bool = False,
                           surface_losses: bool = False) -> dict:
    import json

    from .touchstone import port_parameters, write_touchstone

    ctx.check_name(name)
    rp = ctx.runs / name / "results.json"
    if not rp.exists():
        raise FileNotFoundError(f"no results for design {name!r} ({rp} is missing): run solve_design({name!r}) on a "
                                "design with a fullwave section first")
    results = json.loads(rp.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = port_parameters(results, dense=dense, surface_losses=surface_losses)["Z"].shape[-1]
    out = ctx.runs / name / "touchstone" / f"{name}{'_dense' if dense else ''}_{kind}.s{n}p"
    rep, warn = _call(write_touchstone, results, out, kind, z0, dense, surface_losses=surface_losses)
    return jsonable({"ok": True, "name": name, **rep, "warnings": sorted(set(rep["warnings"]) | set(warn))})


def tool_inspect_comsol(path: str) -> dict:
    from .comsol_model import load_model

    m = load_model(Path(path).expanduser())

    def feats(fs):
        return [{"tag": f.tag, "type": f.op, "label": f.name, **({"disabled": True} if f.disabled else {})}
                for f in fs]

    return jsonable({
        "ok": True, "file": str(m.path), "name": m.name, "comsol_version": m.version,
        "parameters": {k: {"expr": v.get("expr"), "value_SI": v.get("value"), "description": v.get("descr")}
                       for k, v in m.params.items()},
        "geometries": {t: {"dimension": g.get("sdim"), "features": feats(g.get("features", []))}
                       for t, g in m.geometries.items()},
        "materials": [{"tag": x.get("tag"), "label": x.get("name"),
                       **({"disabled": True} if x.get("disabled") else {})} for x in m.materials],
        "physics": [{"tag": p["tag"], "type": p["op"], "label": p.get("name"),
                     **({"disabled": True} if p.get("disabled") else {}),
                     "features": feats(p.get("features", []))} for p in m.physics],
        "studies": [{"tag": s["tag"], "label": s.get("name"), "steps": feats(s.get("steps", []))}
                    for s in m.studies],
        "meshes": list(m.meshes), "warnings": m.warnings})


def tool_import_comsol(ctx, path: str, name: str, route: str = "mesh", allow_partial: bool = False,
                       study: str | None = None, max_frequencies: int = 20) -> dict:
    from .comsol_import import import_comsol

    src = Path(path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"no file {str(src)!r}")
    ctx.check_name(name)
    # a fresh directory per import: a failed or repeated import never overwrites the mesh of a saved design
    base = ctx.workspace / "imports" / name
    base.mkdir(parents=True, exist_ok=True)
    k = 1
    while (base / str(k)).exists():
        k += 1
    outdir = base / str(k)
    outdir.mkdir()
    d = import_comsol(src, outdir / "design.json", route=route, allow_partial=allow_partial, study=study,
                      max_frequencies=int(max_frequencies))
    info = ctx.save(name, d)
    rep = d.import_report.to_json() if getattr(d, "import_report", None) is not None else {}
    fw = d.doc.get("fullwave") or {}
    return jsonable({"ok": True, "name": name, "symmetry": d.doc.get("symmetry", "axisymmetric"),
                     "frequencies_Hz": fw.get("frequencies"),
                     "resolved_parameters": info.get("parameters"), "report": rep,
                     "report_markdown": str(outdir / "import_report.md"), "files": str(outdir)})


# ---------------------------------------------------------------- registration
_LOCK = threading.Lock()   # the server runs sync tools on worker threads; warning capture is process-wide


def register(server, ctx, err: Callable[[Exception], dict]) -> None:
    """Add the module tools to ``server``.  ``ctx`` provides workspace, runs, load, save, check_name."""

    def wrap(fn):
        def run(*a, **kw):
            try:
                with _LOCK:
                    return fn(*a, **kw)
            except Exception as e:  # noqa: BLE001
                return err(e)
        return run

    @server.tool()
    def cavity_presets() -> dict:
        """Beta. Published cavity haloscope runs (ADMX, HAYSTAC, CAPP, TASEH, QUAX, ORGAN): frequency range,
        cavity (B_T, V_L [litres], form factor C, Q0 or Q_L, beta), receiver noise, analysis and the
        published exclusion."""
        return wrap(tool_cavity_presets)()

    @server.tool()
    def cavity_preset_limit(name: str, f_Hz: float | None = None) -> dict:
        """Beta. Recompute a preset's exclusion |g_agg| [GeV^-1] from its numbers and compare it with the published
        one (ratio); also the scan rate and live time the published limit implies."""
        return wrap(tool_cavity_preset_limit)(name, f_Hz)

    @server.tool()
    def cavity_scan(f_Hz: float, B_T: float, V_m3: float, C: float, Q0: float, receiver: dict[str, Any],
                    beta: float | None = None, eta: float = 1.0, model: str = "DFSZ", g_GeV: float | None = None,
                    snr: float = 5.0, rate_Hz_s: float | None = None, f_stop_Hz: float | None = None,
                    duty_cycle: float = 1.0, rho_GeV_cm3: float = 0.45, convention: str = "haloscope") -> dict:
        """Beta. Sensitivity of one cavity mode at frequency f_Hz: scan rate df/dt [Hz/s] reaching `snr` on a
        `model` ('DFSZ' or 'KSVZ') axion or on coupling g_GeV, the optimal antenna coupling beta (used
        when beta is not given), the loaded Q and signal power.  receiver: exactly one of T_sys_K,
        N_A (added noise in quanta, 0.5 = quantum limit) or T_A_K, plus T_cav_K (cavity temperature) and
        optionally T_in_K, eta_out, squeeze_dB, phase_sensitive.  eta: signal transmission (power).
        rate_Hz_s: also return the coupling excluded at that scan rate.  f_stop_Hz: also return the live
        time to scan from f_Hz to f_stop_Hz with these (fixed) cavity parameters.  Couplings follow the
        haloscope convention (g_gamma = -0.97 KSVZ / 0.36 DFSZ, Lambda = 77.6 MeV) and rho_GeV_cm3 = 0.45, as
        experiments quote them; convention='qcd' uses E/N with C_gamma = 1.77 like the other tools."""
        return wrap(tool_cavity_scan)(f_Hz, B_T, V_m3, C, Q0, receiver, beta, eta, model, g_GeV, snr,
                                      rate_Hz_s, f_stop_Hz, duty_cycle, rho_GeV_cm3, convention)

    @server.tool()
    def create_cavity(name: str, shape: str = "pillbox", a: float = 0.1, L: float = 0.2, b: float | None = None,
                      Bz: float = 1.0, modes: int = 6, wall: str = "lossless", wall_T_K: float = 0.15,
                      wall_rf: dict[str, Any] | None = None, target_f_Hz: float | None = None,
                      order: int = 2) -> dict:
        """Beta. Write a cavity design: shape 'pillbox' (radius a, length L) or 'coax' (rod radius a, outer radius
        b, length L) [m], in a uniform axial field Bz [T]. wall: 'lossless' or 'copper' (anomalous skin
        effect, RRR 100, at wall_T_K); wall_rf: any rf block of the design format instead (e.g.
        {"model": "bcs", ...}; see docs/design_format.md). modes: eigenmodes per family (TE and TM for a pillbox, TM for
        a coax); target_f_Hz: look for modes near this frequency. Then solve_design(name): the result's
        cavity_modes lists f_Hz, form factor C, V_m3, G_ohm, Q0."""
        return wrap(tool_create_cavity)(ctx, name, shape, a, L, b, Bz, modes, wall, wall_T_K, wall_rf,
                                        target_f_Hz, order)

    @server.tool()
    def srf_design(scenario: str = "optimistic", geometry: str = "projectx", Rs_ohm: float | None = None,
                   Bpk_T: float | None = None, A: float | None = None, T_K: float | None = None,
                   f_min_Hz: float | None = None, f_max_Hz: float | None = None,
                   live_time_yr: float | None = None, refrigerator_factor: float | None = None) -> dict:
        """Beta. Size an SRF heterodyne detector (pump and signal modes of one superconducting cavity) to scan
        DFSZ over f_min_Hz..f_max_Hz (default 120.9 kHz - 1 MHz, i.e. 0.5-4.1 neV) in live_time_yr (3); the default
        target is illustrative: set the band and live time of your study.
        scenario: 'optimistic' (R_s 1.5 nOhm, B_pk 150 mT, A = 1 quantum, 2 K) or 'conservative' (20 nOhm,
        90 mT, A = 10, 4.5 K), illustrative presets; override any of Rs_ohm, Bpk_T, A (flat noise in quanta), T_K (2 or 4.5 K, or
        any T_K together with refrigerator_factor = room-temperature power per watt at T_K relative to 4.5 K).
        geometry: 'projectx' (default: the computed Project X 650 MHz cell, shortened to make TM020/TE011
        degenerate) or 'equator_diagnostic' (a diagnostic only: it takes the pump field at the equator instead
        of the peak wall field).
        Returns the volume V_m3, pump frequency f0_Hz, Q_i, stored energy, RF power P_RF_W, wall area,
        scan rates at the band ends and the cost metric (MCHF: niobium, cryostat, refrigerator; a scaling metric
        for comparing designs, not a construction estimate).  The sizing uses
        weak-signal statistics and an unconstrained scan schedule; `warnings` says where either is stretched."""
        return wrap(tool_srf_design)(scenario, geometry, Rs_ohm, Bpk_T, A, T_K, f_min_Hz, f_max_Hz, live_time_yr,
                                     refrigerator_factor)

    @server.tool()
    def srf_mode_pair(pair: str = "TM020/TE011") -> dict:
        """Beta. Closed-form figures of merit of a degenerate pump/signal mode pair of a closed cylinder
        ('TM020/TE011', 'TM030/TE021', 'TE012/TM013'): L/R at degeneracy, xi = omega_0 V^(1/3)/c,
        overlap eta_10, pump peak-field ratios B_pk/B0 and E_pk/(c B0), A/V^(2/3) and geometry factors."""
        return wrap(tool_srf_mode_pair)(pair)

    @server.tool()
    def squid_chain(preset: str = "C1+low+RT", f_Hz: list[float] | float | None = None, Q: float | None = None,
                    T_K: float | None = None, T2_K: float = 1.0) -> dict:
        """Beta. Noise of a two-stage dc SQUID readout chain at the frequencies f_Hz (default [1e5, 1e6]).
        preset: 'C1' (first stage alone) or 'C1+<low|medium|high>+<RT|cryo>' (second-stage array 48x3,
        32x2 or 16x1 at T2_K, then a room-temperature or cryogenic preamplifier; the cryogenic one is
        specified for 5-500 MHz only, every chain for f <= 300 MHz).  Returns eta (added
        noise / quantum limit), eta_uncorrelated, eps_uc/hbar, the referred noise breakdown and first-stage
        checks; with Q and T_K also the scan-optimal global coupling kappa_g^2 = M^2/(L_sq L_tot) and xi."""
        return wrap(tool_squid_chain)(preset, f_Hz if f_Hz is not None else [1e5, 1e6], Q, T_K, T2_K)

    @server.tool()
    def squid_reach(name: str, preset: str = "C1+low+RT", Q: float = 1e6, T_K: float = 0.03, n: int = 6,
                    f_min_Hz: float = 120.9e3, f_max_Hz: float = 1.0e6, T2_K: float = 1.0) -> dict:
        """Beta. Reach g_95(m_a) of a solved lumped-element design (its Psi and L_p at the drive in the design)
        read out by a dc SQUID chain preset (see squid_chain), with circuit quality factor Q and
        temperature T_K, scanning f_min_Hz..f_max_Hz (default 0.5-4.1 neV, an illustrative band) for DFSZ in 3 years
        (rho = 0.3 GeV/cm^3). '+cryo' presets need f_min_Hz >= 5 MHz. n: number of mass points returned."""
        return wrap(tool_squid_reach)(ctx, name, preset, Q, T_K, "as_designed", n, f_min_Hz, f_max_Hz, T2_K)

    @server.tool()
    def export_comsol(name: str, format: str = "mphtxt", order: int = 1, physics: str = "none") -> dict:
        """Export a saved design for COMSOL into <workspace>/runs/<name>/comsol/: format 'mphtxt' (meshed,
        with named selections for every region and boundary; order 2 keeps curved second-order nodes),
        'java' (a COMSOL Java model that rebuilds the geometry; compile and run it with COMSOL) or 'step'
        (3D CAD). physics: 'none', or 'fullwave' to add the frequency-domain physics (3D designs with a
        port pickup; java or mphtxt)."""
        return wrap(tool_export_comsol)(ctx, name, format, order, physics)

    @server.tool()
    def export_touchstone(name: str, kind: str = "S", z0: float = 50.0, dense: bool = False,
                          surface_losses: bool = False) -> dict:
        """Write a solved design's full-wave port parameters (fullwave section; multiport 3D gives the full matrix)
        as a Touchstone v1 file <workspace>/runs/<name>/touchstone/<name>[_dense]_<kind>.s<N>p (overwritten; the
        result says so). kind: 'S' (default), 'Z' or 'Y'; z0: reference resistance in ohm (one for all ports);
        dense: the adaptive sweep's dense model curve instead of the solved rows; surface_losses: add the resistive
        part of the perturbative conductor loss (not the surface reactance). Values use exp(+j omega t) (the
        engine's exp(-i omega t) Z is conjugated). Returns sigma_max of S (passivity; violations are reported,
        never clipped), the reciprocity error and the port order."""
        return wrap(tool_export_touchstone)(ctx, name, kind, z0, dense, surface_losses)

    @server.tool()
    def inspect_comsol(path: str) -> dict:
        """Beta. Summary of a COMSOL .mph file without importing it: parameters (expression and SI value),
        geometry features, materials, physics interfaces and their features, studies, meshes."""
        return wrap(tool_inspect_comsol)(path)

    @server.tool()
    def import_comsol(path: str, name: str, route: str = "mesh", allow_partial: bool = False,
                      study: str | None = None, max_frequencies: int = 20) -> dict:
        """Beta. Import a COMSOL .mph file as design `name` (mesh, materials, ports, frequencies). route: 'mesh'
        (default: the stored mesh, as COMSOL solved it), 'replay' (rebuild the geometry from its sequence;
        zero-thickness sheets get a small thickness and face ports become thin gap volumes) or 'auto' (replay
        when the whole sequence is supported). Features that
        would change the results and cannot be mapped stop the import unless allow_partial=True; the
        'report' lists every feature and what became of it. A study's frequency list longer than
        max_frequencies is thinned to that many points (each one is a full solve); frequencies_Hz shows the
        result. Then solve_design(name)."""
        return wrap(tool_import_comsol)(ctx, path, name, route, allow_partial, study, max_frequencies)
