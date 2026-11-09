"""Import a COMSOL Multiphysics ``.mph`` model as an axionFEM design (``axionfem.design/1``).  Beta.

    axf import-comsol model.mph -o out/design.json [--mesh-route | --replay-route] [--allow-partial]
    d = axionfem.import_comsol("model.mph")            # -> Design (files next to the default output)

No COMSOL installation or licence is used: the archive is read directly (docs/comsol_import.md).

Two routes:

- **mesh route** (default, exact): the mesh stored in the ``.mph`` (``mesh*.mphbin``) is decoded and written as a
  gmsh MSH 2.2 file whose element attributes are axionFEM region attributes (one region per COMSOL domain of the
  physics) and whose boundary attributes are ``10000 + COMSOL boundary number``.  The design then names that mesh
  (``mesh.file``/``mesh.info_file``); boundary conditions are boundary groups (PEC/PMC/port) in the info file.
  The problem axionFEM solves is the one COMSOL solved, on the same tetrahedra (straight-sided: COMSOL curves the
  boundary elements onto the CAD, see the report).
- **replay route** (editable): the geometry sequence is rebuilt from axionFEM solids (:mod:`axionfem.comsol_replay`)
  and meshed by axionFEM; selections are mapped geometrically against the decoded COMSOL mesh.  Unsupported
  features fall back to the mesh route.

Every item of the model is listed in the import report as mapped / approximated / unsupported / ignored.
Unsupported items that change results make the import fail unless ``allow_partial`` is set.

Conventions (docs/comsol_export.md, sec. 4): COMSOL uses e^{+jωt}, axionFEM e^{-iωt}; axionFEM's Z is the complex
conjugate of COMSOL's Zport.  A COMSOL External Current Density Je [A/m²] becomes the prescribed field
B0 [T] = Je / (1 A m⁻² T⁻¹) of ``fullwave.source``, so axionFEM's open-circuit voltage −iωΨ corresponds to COMSOL's
V_oc (conjugated).  COMSOL's ``Vport`` of a non-driven lumped port is the loaded voltage V_oc·Zref/(Zref+Zport).
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path

import numpy as np

from . import comsol_mphbin as mb
from .comsol_model import ComsolModel, ExpressionError, Feature, evaluate, load_model, parse_list, unit_factor

__all__ = ["import_comsol", "ImportReport", "ImportError_", "main", "BOUNDARY_ATTR0", "PORT_ATTR0"]

BOUNDARY_ATTR0 = 10000   # boundary attribute = 10000 + COMSOL boundary number (1-based, as in the GUI)
PORT_ATTR0 = 9000        # lumped port k (1-based) -> attribute 9000 + k (one group per port feature)
REGION_ATTR0 = 2         # axionFEM: region k (0-based) owns element attribute 2 + k


class ImportError_(RuntimeError):
    """The model cannot be imported faithfully (raise with allow_partial=False)."""


# ------------------------------------------------------------------------------------------------ report
@dataclass
class ImportReport:
    source: str
    route: str = "mesh"
    items: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add(self, category: str, comsol: str, status: str, axionfem: str = "", detail: str = "",
            changes_results: bool = False):
        assert status in ("mapped", "approximated", "unsupported", "ignored", "info")
        self.items.append({"category": category, "comsol": comsol, "status": status, "axionfem": axionfem,
                           "detail": detail, "changes_results": bool(changes_results)})

    @property
    def blocking(self) -> list:
        return [i for i in self.items if i["status"] == "unsupported" and i["changes_results"]]

    def to_json(self) -> dict:
        counts = {}
        for i in self.items:
            counts[i["status"]] = counts.get(i["status"], 0) + 1
        return {"schema": "axionfem.comsol_import_report/1", "beta": True, "source": self.source,
                "route": self.route, "counts": counts, "blocking": self.blocking, "summary": self.summary,
                "warnings": self.warnings, "items": self.items}

    def to_markdown(self) -> str:
        j = self.to_json()
        out = ["# COMSOL import report (beta)", "", f"- source: `{self.source}`", f"- route: **{self.route}**",
               "- items: " + ", ".join(f"{k} {v}" for k, v in sorted(j["counts"].items()))]
        if self.blocking:
            out += ["", "**Blocking (unsupported, changes results):**", ""]
            out += [f"- {i['category']} `{i['comsol']}`: {i['detail']}" for i in self.blocking]
        if self.summary:
            out += ["", "## Summary", "", "```json", json.dumps(self.summary, indent=1, default=str)[:20000], "```"]
        for st in ("unsupported", "approximated", "mapped", "info", "ignored"):
            rows = [i for i in self.items if i["status"] == st]
            if not rows:
                continue
            out += ["", f"## {st.capitalize()} ({len(rows)})", "", "| category | COMSOL | axionFEM | detail |",
                    "|---|---|---|---|"]
            for i in rows:
                cell = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
                out.append(f"| {cell(i['category'])} | {cell(i['comsol'])} | {cell(i['axionfem'])} | {cell(i['detail'])} |")
        if self.warnings:
            out += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ helpers
def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s.encode("ascii", "ignore").decode()).strip("_").lower()
    return s or "x"


def _feat_desc(f: Feature) -> str:
    return f"{f.tag} ({f.op}{', ' + f.name if f.name else ''})"


def _num(model: ComsolModel, expr, variables=None, functions=None):
    fun = dict(model.functions)
    fun.update(functions or {})
    return evaluate(expr, model.param_values, variables, fun)


def _tensor(model, vals: list, variables=None, functions=None, what=""):
    """A 1-, 3- or 9-component COMSOL value -> (scalar, isotropic?) evaluated with SI units."""
    if vals is None:
        return None, True
    if isinstance(vals, str):
        vals = [vals]
    v = [complex(np.real_if_close(np.asarray(_num(model, x, variables, functions)).mean())) for x in vals]
    if len(v) == 1:
        return v[0], True
    if len(v) == 9:
        d = [v[0], v[4], v[8]]
        off = [v[k] for k in (1, 2, 3, 5, 6, 7)]
        iso = all(abs(x) <= 1e-12 * max(1.0, abs(d[0])) for x in off) and \
            all(abs(x - d[0]) <= 1e-9 * max(1.0, abs(d[0])) for x in d)
        return d[0], iso
    if len(v) == 3:
        iso = all(abs(x - v[0]) <= 1e-9 * max(1.0, abs(v[0])) for x in v)
        return v[0], iso
    raise ExpressionError(f"{what}: {len(v)} components")


# ------------------------------------------------------------------------------------------------ frequencies
def study_frequencies(model: ComsolModel, report: ImportReport, study: str | None = None) -> dict:
    """Frequencies (Hz) and eigenfrequency requests of the frequency-domain studies."""
    out = {"frequencies": [], "studies": [], "modes": 0, "mode_shift_Hz": None}
    for st in model.studies:
        if study and st["tag"] != study:
            continue
        for stp in st["steps"]:
            desc = f"{st['tag']}/{stp.tag} ({stp.op})"
            if stp.disabled or st["disabled"]:
                report.add("study", desc, "ignored", detail="disabled")
                continue
            if stp.op in ("Frequency", "FrequencyAdaptive", "FrequencyStationary"):
                pl = stp.props.get("plist")
                unit = stp.props.get("punit", "Hz") if stp.props.get("usefrequnit", "on") != "off" else "Hz"
                try:
                    if isinstance(pl, list):
                        fs = [float(r[0]) * unit_factor(unit) for r in pl if r]
                    else:
                        fs = parse_list(pl, model.param_values, model.functions, unit, model.unit_params)
                except ExpressionError as e:
                    report.add("study", desc, "unsupported", detail=f"frequency list '{pl}': {e}")
                    continue
                key = stp.key or ""
                m = re.search(r"plist(.*?)(?:show|aweon|aweoff|geometricNonlinearity|preusesol)", key)
                built = m.group(1) if m else None
                note = ""
                if built and isinstance(pl, str) and built.replace(" ", "") != pl.replace(" ", ""):
                    note = f"; the stored solution was computed with plist {built} (edited afterwards)"
                out["frequencies"] += fs
                out["studies"].append({"study": st["tag"], "step": stp.tag, "op": stp.op, "n": len(fs),
                                       "f_min_Hz": min(fs) if fs else None, "f_max_Hz": max(fs) if fs else None,
                                       "plist": pl if isinstance(pl, str) else None, "built_plist": built})
                report.add("study", desc, "mapped", "fullwave.frequencies",
                           f"{len(fs)} frequencies {min(fs):.6g}..{max(fs):.6g} Hz (plist {pl!s:.80}, punit {unit})"
                           + (" — adaptive (AWE) sweep: axionFEM solves each frequency" if stp.op == "FrequencyAdaptive" else "")
                           + note)
            elif stp.op in ("Eigenfrequency",):
                n = int(float(stp.props.get("neigs", "6")))
                out["modes"] = max(out["modes"], n)
                sh = stp.props.get("shift")
                try:
                    out["mode_shift_Hz"] = float(np.real(_num(model, sh))) if sh else None
                except ExpressionError:
                    pass
                report.add("study", desc, "approximated", "fullwave.modes",
                           f"{n} eigenmodes; axionFEM computes the lowest modes of the closed PEC domain "
                           f"(COMSOL shift {sh} not used)")
            elif stp.op in ("Stationary",):
                report.add("study", desc, "unsupported", detail="stationary studies are not mapped (mesh route: emw "
                           "frequency domain only)", changes_results=False)
            else:
                report.add("study", desc, "unsupported", detail=f"study step {stp.op} is not mapped")
    return out


def choose_frequencies(fs: list, max_n: int) -> list:
    fs = sorted(set(float(f) for f in fs if f > 0))
    if len(fs) <= max_n:
        return fs
    idx = np.unique(np.round(np.linspace(0, len(fs) - 1, max_n)).astype(int))
    return [fs[i] for i in idx]


# ------------------------------------------------------------------------------------------------ materials
def domain_materials(model: ComsolModel, geom: str, ndom: int, report: ImportReport, sdim: int = 3) -> dict:
    """COMSOL domain -> material dict (later material nodes override earlier ones, as in COMSOL)."""
    out = {}
    for mat in model.materials:
        sel = mat["selection"]
        desc = f"{mat['tag']} ({mat['name']})"
        if mat["disabled"]:
            report.add("material", desc, "ignored", detail="disabled")
            continue
        if sel is None or (sel.geom and sel.geom.split("/")[-1] != geom):
            continue
        if sel.unknown:
            raise ImportError_(f"material {desc}: its selection is not an explicit or named selection the importer "
                               "understands; refusing to guess which domains it covers")
        if sel.dim is not None and sel.dim != sdim:
            if sel.dim == sdim - 1:   # boundary material: read by boundary_materials (impedance / transition)
                report.add("material", desc, "info", detail=f"boundary-level material on boundaries "
                           f"{sel.entities if sel.entities is not None else 'all'}: used by impedance/transition "
                           "boundaries that take their material from the boundary")
                continue
            report.add("material", desc, "ignored", detail=f"material on dimension-{sel.dim} entities; only domain "
                       "and boundary materials enter the emw model")
            continue
        if mat.get("op") not in (None, "", "Common"):
            raise ImportError_(f"material {desc} is a {mat.get('op')} node (e.g. a link to a global material); "
                               "only local materials (Common) are read")
        doms = sel.entities if sel.entities is not None else list(range(1, ndom + 1))
        for d in doms:
            out[d] = mat
    return out


def boundary_materials(model: ComsolModel, geom: str, sdim: int, nbnd: int) -> dict:
    """COMSOL boundary (GUI number) -> material assigned on boundary level (dimension sdim - 1), later nodes win."""
    out = {}
    for mat in model.materials:
        sel = mat["selection"]
        if mat["disabled"] or sel is None or sel.unknown or sel.dim != sdim - 1:
            continue
        if sel.geom and sel.geom.split("/")[-1] != geom:
            continue
        for b in (sel.entities if sel.entities is not None else range(1, nbnd + 1)):
            out[int(b)] = mat
    return out


def _robin_material(model, f: Feature, bnds, bmat: dict, dmat_of_side, freqs, report, materials: dict, what: str):
    """Material of an impedance / transition boundary: the feature's user-defined values, else the boundary
    material of its boundaries (all the same), else (COMSOL "domain material" option) refused.  Adds a linear
    design material and returns (name, props)."""
    if f.op in ("ImpedanceBoundaryCondition", "Impedance"):
        # the impedance feature names its permeability / conductivity overrides murbnd(_mat) / sigmabnd(_mat)
        ren = {"murbnd_mat": "mur_mat", "murbnd": "mur", "sigmabnd_mat": "sigma_mat", "sigmabnd": "sigma"}
        params = dict(f.params)
        for a, b in ren.items():
            if a in f.params:
                params[b] = f.params[a]
        f = _dc_replace(f, params=params)
    userdef = any(f.p(k) == "userdef" for k in ("epsilonr_mat", "mur_mat", "sigma_mat"))
    mats = {bmat[b]["tag"]: bmat[b] for b in bnds if b in bmat}
    if len(mats) > 1:
        raise ImportError_(f"{_feat_desc(f)}: its boundaries carry different boundary materials {sorted(mats)}; split "
                           "the feature per material")
    mat = next(iter(mats.values())) if mats else None
    if mat is None and not userdef:
        src = (f.p("MaterialType") or f.p("minput_material") or "")
        raise ImportError_(f"{_feat_desc(f)}: no boundary material on boundaries {list(bnds)} and no user-defined "
                           f"material properties ({src or 'boundary material'}); assign a material on boundary level")
    props = material_properties(model, mat, f, freqs, report)
    nm = "comsol_" + what + "_" + _slug(f.tag)
    materials[nm] = {"kind": "linear", "eps_r": props["eps_r"], "mu_r": props["mu_r"], "tan_delta": props["tan_delta"],
                     "sigma_S_m": props["sigma_S_m"]}
    return nm, props


def _robin_freq_note(desc, pr, report):
    """Report a frequency-dependent Robin material (axionFEM uses constant properties: the median)."""
    if pr.get("frequency_dependence"):
        report.add("material", desc, "approximated", "constant boundary material",
                   "frequency-dependent properties " + json.dumps(pr["frequency_dependence"]) +
                   " — axionFEM materials are constant (median over the study frequencies used)", changes_results=True)


def robin_feature(model, f: Feature, ents, bset_ext: set, bset_int: set, bmat: dict, freqs, report, materials: dict):
    """Map an emw ImpedanceBoundaryCondition / TransitionBoundaryCondition / Scattering feature to boundary-group
    parameters ({"type": ..., ...}) for its boundaries; raises ImportError_ when it cannot be mapped."""
    desc = _feat_desc(f)
    op = f.op
    if op in ("ImpedanceBoundaryCondition", "Impedance"):   # COMSOL's op name is "Impedance" (6.3 file)
        bad = [b for b in ents if b not in bset_ext]
        if bad:
            raise ImportError_(f"{desc}: impedance condition on interior boundaries {bad} (only exterior walls)")
        for k in ("Es", "Es1", "Es2"):   # surface source field: not mapped
            v = f.p(k)
            if v is None:
                continue
            try:
                vals = [complex(np.asarray(_num(model, x, {"freq": fr})).item()) for x in (v if isinstance(v, list) else [v])
                        for fr in (freqs or [1e6])]
            except Exception as exc:  # noqa: BLE001
                raise ImportError_(f"{desc}: surface source field {k} = {v} cannot be evaluated ({exc}); a "
                                   "boundary source is not mapped") from exc
            if any(abs(x) > 0 for x in vals):
                raise ImportError_(f"{desc}: nonzero surface source field {k} = {v}: impedance boundaries with a "
                                   "source field are not mapped (only the homogeneous Leontovich condition)")
        zdef = str(f.p("surfaceImpedanceDefinition") or "fromMaterial")
        if zdef != "fromMaterial":   # an explicit surface impedance Zs (COMSOL e^{+jwt}: Zs = R + jX)
            zexpr = f.p("surfaceImpedance")
            try:
                zs = [complex(np.asarray(_num(model, zexpr, {"freq": fr})).item()) for fr in (freqs or [1e6])]
            except Exception as exc:  # noqa: BLE001
                raise ImportError_(f"{desc}: surface impedance '{zexpr}' ({zdef}) cannot be evaluated outside COMSOL "
                                   f"({exc})") from exc
            if max(abs(z - zs[0]) for z in zs) > 1e-12 * max(abs(zs[0]), 1e-300):
                raise ImportError_(f"{desc}: surface impedance '{zexpr}' depends on frequency; axionFEM's Zs_ohm is "
                                   "a constant")
            z = zs[0]
            if z.real < 0 or abs(z) == 0:
                raise ImportError_(f"{desc}: surface impedance {z} must have Re >= 0 and be nonzero")
            report.add("physics", desc, "mapped", "boundary_groups type impedance, Zs_ohm",
                       f"boundaries {list(ents)}: explicit surface impedance {zexpr} = {z} ohm (COMSOL e^(+jwt)); "
                       f"Zs_ohm = [{z.real:g}, {-z.imag:g}] in e^(-iwt)")
            return {"type": "impedance", "Zs_ohm": [z.real, -z.imag]}
        nm, pr = _robin_material(model, f, ents, bmat, None, freqs, report, materials, "ibc")
        _robin_freq_note(desc, pr, report)
        report.add("physics", desc, "mapped", "boundary_groups type impedance",
                   f"boundaries {list(ents)}: Leontovich wall Z_s = sqrt(mu/eps_c) of {nm} (sigma {pr['sigma_S_m']:.4g} S/m, "
                   f"eps_r {pr['eps_r']:.4g}, mu_r {pr['mu_r']:.4g}), as COMSOL's n x H + sqrt(eps_c/mu) n x (E x n) = 0")
        return {"type": "impedance", "material": nm}
    if op == "TransitionBoundaryCondition":
        bad = [b for b in ents if b not in bset_int]
        if bad:
            raise ImportError_(f"{desc}: transition condition on exterior boundaries {bad} (only interior layers)")
        d = f.p("d")
        if d is None:
            raise ImportError_(f"{desc}: no layer thickness d")
        dval = float(np.real(_num(model, d)))
        nm, pr = _robin_material(model, f, ents, bmat, None, freqs, report, materials, "tbc")
        _robin_freq_note(desc, pr, report)
        eps0 = 8.8541878128e-12
        mu0 = 1.25663706212e-6
        kd = 0.0
        fd = pr.get("frequency_dependence") or {}
        hi = {k: fd[k]["max"] if k in fd else pr[k] for k in ("eps_r", "tan_delta", "mu_r", "sigma_S_m")}
        for fr in (freqs or [1e6]):   # thinness with the largest property values over the study frequencies
            w = 2 * math.pi * fr
            ec = eps0 * hi["eps_r"] * (1 + 1j * hi["tan_delta"]) + 1j * hi["sigma_S_m"] / w
            kd = max(kd, abs(w * np.sqrt(mu0 * hi["mu_r"] * ec) * dval))
        if kd > 0.5:
            raise ImportError_(f"{desc}: layer of thickness {dval:g} m with |k d| up to {kd:.3g} > 0.5 at the study "
                               "frequencies: the tangential E differs on the two faces (e.g. a metal layer thicker than "
                               "the skin depth); axionFEM's sheet model (equal E_t) does not apply. Use impedance "
                               "boundaries on both faces of a meshed layer instead")
        report.add("physics", desc, "approximated", "boundary_groups type sheet",
                   f"boundaries {list(ents)}: thin layer d = {dval:g} m of {nm}; sheet admittance "
                   f"Y_s = -(2i/eta) tan(k d/2) with E_t continuous (|k d| <= {kd:.3g}; COMSOL's transition condition "
                   "also lets E_t jump by ~ i omega mu d H_t, relative error O(|k d|))")
        return {"type": "sheet", "material": nm, "thickness_m": dval}
    # scattering
    bad = [b for b in ents if b not in bset_ext]
    if bad:
        raise ImportError_(f"{desc}: scattering condition on interior boundaries {bad}")
    inc = str(f.p("IncidentField") or "NoIncidentField")
    if inc.lower() not in ("noincidentfield", "none", "off"):
        raise ImportError_(f"{desc}: incident field {inc} is not mapped (only an absorbing boundary without "
                           "incident wave)")
    wt = str(f.p("WaveType") or f.p("waveType") or "PlaneWave").lower()
    wave = "spherical" if "spher" in wt else "cylindrical" if "cyl" in wt else "plane"
    grp = {"type": "scattering", "wave": wave}
    detail = f"boundaries {list(ents)}: first-order absorbing condition, {wave} wave"
    status = "mapped"
    if wave != "plane":
        r0 = f.p("r0")
        if r0 is not None:
            try:
                grp["center"] = [float(np.real(_num(model, x))) for x in (r0 if isinstance(r0, list) else [r0])]
            except ExpressionError:
                pass
        ax = f.p("raxis") or f.p("rAxis")
        if ax is not None and wave == "cylindrical":
            try:
                grp["axis"] = [float(np.real(_num(model, x))) for x in (ax if isinstance(ax, list) else [ax])]
            except ExpressionError:
                pass
        status = "approximated"
        detail += ("; curvature term of the E-field formulation (spherical: none at first order; cylindrical: "
                   "+-nu/(2 rho) for E along / around the axis): COMSOL's own curvature term is not verified (no "
                   "file); both are transparent to O(1/(k R)^2)")
    order = str(f.p("Order") or f.p("order") or "").lower()
    if "second" in order or order == "2":
        status = "approximated"
        detail += "; COMSOL second-order condition mapped to first order (more reflection at oblique incidence)"
    report.add("physics", desc, status, "boundary_groups type scattering", detail)
    return grp


def pml_layers(model, geom: str, sdim: int, phys_doms: list, region_of: dict, report, axisym: bool = False):
    """COMSOL PML coordinate systems on the emw domains -> fullwave.pml entries (axionfem/pml.hpp)."""
    out = []
    for cs in model.coordsys:
        if cs.op not in ("PML", "PerfectlyMatchedLayer", "InfiniteElement", "InfiniteElementDomain"):
            continue
        desc = _feat_desc(cs)
        if cs.disabled:
            report.add("physics", desc, "ignored", detail="disabled")
            continue
        if cs.op.startswith("Infinite"):
            report.add("physics", desc, "unsupported", detail="infinite element domains are a static-field feature, not "
                       "mapped", changes_results=True)
            continue
        if cs.selection is None or cs.selection.unknown:
            raise ImportError_(f"{desc}: PML without an explicit domain selection")
        if cs.selection.geom and cs.selection.geom.split("/")[-1] != geom:
            report.add("physics", desc, "ignored", detail=f"PML of geometry {cs.selection.geom}, not of the emw "
                       f"geometry {geom}")
            continue
        doms = [d for d in (cs.selection.entities or []) if d in region_of]
        if not doms:
            report.add("physics", desc, "ignored", detail="no emw domain in its selection")
            continue
        # PML settings are property values (p:ScalingType ...) in COMSOL 6.3 files; defaults are not stored
        cp = lambda n: cs.p(n) if cs.p(n) is not None else cs.props.get(n)  # noqa: E731
        st = str(cp("ScalingType") or cp("scalingType") or "Cartesian").lower()
        typ = "spherical" if "spher" in st else "cylindrical" if "cyl" in st else "cartesian"
        status, notes = "mapped", []
        if axisym and typ == "cartesian":
            typ = "cylindrical"
            notes.append("Cartesian scaling in an axisymmetric component = cylindrical (r, z) stretching")
        if "user" in st or "general" in st or "conical" in st:
            raise ImportError_(f"{desc}: PML scaling type {st} is not supported (cartesian, cylindrical, spherical)")
        e = {"regions": [region_of[d] for d in doms], "type": typ}

        def num(*names, default=None):
            for n in names:
                v = cs.p(n) if cs.p(n) is not None else cs.props.get(n)
                if v is not None:
                    try:
                        return float(np.real(_num(model, v)))
                    except ExpressionError:
                        notes.append(f"{n} = {v} not evaluated")
            return default
        F = num("PMLfactor", "scalingFactor", default=1.0)
        pp = num("PMLgamma", "PMLp", "curvatureParameter", default=1.0)
        e["scaling_factor"], e["curvature"] = F, max(pp, 1.0)
        if pp < 1:
            notes.append(f"curvature parameter {pp} < 1 raised to 1")
        wsrc = str(cp("wavelengthSourceType") or "fromPhysics").lower()
        if "user" in wsrc:
            lam = num("typicalWavelength", "wavelength")
            if lam:
                e["wavelength_m"] = lam
        stype = str(cp("stretchingType") or "polynomial").lower()
        if "polyn" not in stype:
            status = "approximated"
            notes.append(f"stretching type {stype}: mapped to axionFEM's polynomial stretch")
        notes.append("stretch q~ = q0 + sign (d xi + (1+i) F lambda xi^p): COMSOL's polynomial stretch plus the identity "
                     "term d xi; the solution outside the layer agrees up to the layer's discretization error")
        if typ != "cartesian":
            c = cp("r0") or cp("center")
            if c is not None:
                try:
                    e["center"] = [float(np.real(_num(model, x))) for x in (c if isinstance(c, list) else [c])]
                except ExpressionError:
                    pass
            if "center" not in e:
                status = "approximated"
                notes.append("center not stored: the origin is used" + (" (on the axis)" if axisym else ""))
                e["center"] = [0.0, 0.0] if sdim == 2 else [0.0, 0.0, 0.0]
            if typ == "cylindrical" and sdim == 3:
                a = cp("raxis") or cp("axis")
                if a is not None:
                    try:
                        e["axis"] = [float(np.real(_num(model, x))) for x in (a if isinstance(a, list) else [a])]
                    except ExpressionError:
                        pass
                if "axis" not in e:
                    status = "approximated"
                    notes.append("axis not stored: z is used")
        report.add("physics", desc, status, "fullwave.pml",
                   f"{typ} PML on domains {doms}, scaling factor {F:g}, curvature {pp:g}; " + "; ".join(notes))
        out.append(e)
    return out


def _group_param(mat, names):
    for g in mat["groups"].values():
        for n in names:
            if n in g and g[n]:
                return g[n]
    return None


def model_temperature(model: ComsolModel, wee: Feature | None) -> float:
    """Temperature that material functions are evaluated at: the feature's user-defined model input, else the
    common default model input (Default Model Inputs), else COMSOL's default 293.15 K."""
    if wee is not None and wee.p("minput_temperature_src") == "userdef" and wee.p("minput_temperature"):
        return float(np.real(_num(model, wee.p("minput_temperature"))))
    for cf in model.archive.root.iter("CommonFeature"):
        for prm in cf.findall("param"):
            if prm.get("param") == "minput_temperature":
                from .comsol_model import parse_value
                v = parse_value(prm.get("value"))
                if v:
                    try:
                        return float(np.real(_num(model, v[0])))
                    except ExpressionError:
                        pass
    return 293.15


def material_properties(model: ComsolModel, mat: dict | None, wee: Feature | None, freqs: list,
                        report: ImportReport) -> dict:
    """eps_r, tan_delta, mu_r, sigma of one domain for emw (COMSOL e^{+jwt}: eps = eps'(1 - j tan_delta)).

    Every property the emw constitutive relations need must be found (feature user-defined value or material
    property group); anything missing, anisotropic or using an unsupported model option raises ImportError_."""
    fun = dict(mat["functions"]) if mat else {}
    fvals = freqs if freqs else [1e6]
    model_name = (wee.p("DisplacementFieldModel") if wee else None) or "RelativePermittivity"
    where = (mat["tag"] if mat else "no material") + (f" / {wee.tag}" if wee else "")
    notes = []
    T = model_temperature(model, wee)
    if wee is not None:
        h = wee.p("ConstitutiveRelationH") or "RelativePermeability"
        if h != "RelativePermeability":
            raise ImportError_(f"{where}: magnetic constitutive relation {h} is not supported (only relative "
                               "permeability; magnetic losses would be dropped)")
        if (wee.p("coordinateSystem") or "GlobalSystem") != "GlobalSystem":
            raise ImportError_(f"{where}: material coordinate system {wee.p('coordinateSystem')} is not supported")

    def from_feature_or_mat(prop_mats, prop_val: str, mat_names: list, required=True):
        prop_mats = [prop_mats] if isinstance(prop_mats, str) else prop_mats
        if wee is not None and any(wee.p(pm) == "userdef" for pm in prop_mats):
            return wee.params.get(prop_val), f"{wee.tag}.{prop_val} (user defined)"
        if mat is None:
            if required:
                raise ImportError_(f"{where}: no material on this emw domain and {prop_val} is not user defined "
                                   "(COMSOL would not solve this model)")
            return None, "no material"
        v = _group_param(mat, mat_names)
        if v is None and required:
            raise ImportError_(f"{where}: material {mat['tag']} has no {mat_names[0]} (needed by emw)")
        return v, f"{mat['tag']}.{mat_names[0]}"

    def ev(vals, src):
        res = []
        for f in fvals:
            v, iso = _tensor(model, vals, {"freq": f, "T": T}, fun, src)
            if not iso:
                raise ImportError_(f"{src} is anisotropic; axionFEM materials are isotropic")
            res.append(v)
        res = np.array(res)
        return res

    eps = None
    if model_name == "RelativePermittivity":
        vals, src = from_feature_or_mat("epsilonr_mat", "epsilonr", ["relpermittivity"], required=False)
        if vals is None and mat is not None:
            ep = _group_param(mat, ["epsilonPrim"])
            td = _group_param(mat, ["tanDelta"])
            if ep is None:
                raise ImportError_(f"{where}: no relative permittivity (relpermittivity or a loss-tangent group)")
            e1 = ev(ep, f"{mat['tag']}.epsilonPrim")
            t = ev(td, f"{mat['tag']}.tanDelta") if td is not None else np.zeros_like(e1)
            eps = e1 * (1 - 1j * t)
            notes.append(f"relpermittivity taken from the loss-tangent group (eps' {mat['tag']}.epsilonPrim, tan "
                         f"delta {mat['tag']}.tanDelta), which is what COMSOL evaluated")
        elif vals is None:
            raise ImportError_(f"{where}: no material and no user-defined relative permittivity")
        else:
            eps = ev(vals, src)
    elif model_name in ("LossTangent", "LossTangentDF", "LossTangentLossFactor", "DissipationFactor"):
        # COMSOL's switches: dissipation factor (epsilonPrimDF, tanDelta) or loss angle (epsilonPrimLA, delta)
        if wee is not None and (wee.p("delta_mat") == "userdef" or model_name == "LossTangent" and
                                wee.p("epsilonPrimLA_mat") == "userdef"):
            ep, s1 = from_feature_or_mat(["epsilonPrimLA_mat"], "epsilonPrimLA", ["epsilonPrim"])
            dl, s2 = from_feature_or_mat(["delta_mat"], "delta", ["delta"])
            eps = ev(ep, s1) * (1 - 1j * np.tan(ev(dl, s2)))
        else:
            ep, s1 = from_feature_or_mat(["epsilonPrimDF_mat", "epsilonPrim_mat"], "epsilonPrimDF"
                                         if wee is not None and wee.p("epsilonPrimDF_mat") == "userdef" else
                                         "epsilonPrim", ["epsilonPrim"])
            td, s2 = from_feature_or_mat(["tanDelta_mat"], "tanDelta", ["tanDelta"])
            eps = ev(ep, s1) * (1 - 1j * ev(td, s2))
    elif model_name == "DielectricLoss":
        ep, s1 = from_feature_or_mat("epsilonPrim_mat", "epsilonPrim", ["epsilonPrim"])
        eb, s2 = from_feature_or_mat("epsilonBis_mat", "epsilonBis", ["epsilonBis"])
        eps = ev(ep, s1) - 1j * ev(eb, s2)
    elif model_name == "RefractiveIndex":
        n_, s1 = from_feature_or_mat("n_mat", "n", ["n"])
        k_, s2 = from_feature_or_mat("ki_mat", "ki", ["ki"])
        eps = (ev(n_, s1) - 1j * ev(k_, s2)) ** 2
    else:
        raise ImportError_(f"{where}: displacement field model {model_name} is not supported")
    mvals, msrc = from_feature_or_mat("mur_mat", "mur", ["relpermeability"])
    mu = ev(mvals, msrc)
    svals, ssrc = from_feature_or_mat("sigma_mat", "sigma", ["electricconductivity"])
    sig = ev(svals, ssrc)
    if np.any(np.abs(mu.imag) > 0) or np.any(np.abs(sig.imag) > 0):
        raise ImportError_(f"{where}: complex permeability or conductivity is not supported")
    er = eps.real
    tand = -eps.imag / er
    if abs(T - 293.15) > 1e-9:
        notes.append(f"material functions evaluated at T = {T:g} K (model input)")
    out = {"eps_r": float(np.median(er)), "tan_delta": float(np.median(tand)), "mu_r": float(np.median(mu.real)),
           "sigma_S_m": float(np.median(sig.real)), "model": model_name, "notes": notes,
           "frequency_dependence": None, "T_K": T}
    span = {k: (float(v.min()), float(v.max())) for k, v in
            {"eps_r": er, "tan_delta": tand, "mu_r": mu.real, "sigma_S_m": sig.real}.items()}
    for k, (lo, hi) in span.items():
        if hi - lo > 1e-9 * max(1.0, abs(hi)):
            out["frequency_dependence"] = out.get("frequency_dependence") or {}
            out["frequency_dependence"][k] = {"min": lo, "max": hi, "f_min_Hz": min(fvals), "f_max_Hz": max(fvals),
                                              "used": "median over the study frequencies"}
    return out


# ------------------------------------------------------------------------------------------------ source
def classify_source(model: ComsolModel, je: list, pts: np.ndarray) -> dict:
    """Je (3 expressions in A/m^2) sampled at points -> an axionFEM prescribed-source kind, or raise."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    var = {"x": x, "y": y, "z": z, "X": x, "Y": y, "Z": z, "r": np.hypot(x, y)}
    J = []
    if je is None or len(je) != 3:
        raise ImportError_(f"Je must have three components (got {je})")
    for c in je:
        v = np.asarray(_num(model, c, var))
        if np.iscomplexobj(v) and np.any(np.abs(np.imag(v)) > 1e-12 * max(np.abs(v).max(), 1e-300)):
            raise ImportError_(f"complex Je component '{c}' (a phase between components) is not representable "
                               "by axionFEM's real prescribed source")
        J.append(np.broadcast_to(np.real(v), x.shape).astype(float))
    J = np.array(J).T                         # (n, 3)
    scale = max(np.abs(J).max(), 1e-300)
    tol = 1e-9 * scale
    if np.all(np.abs(J - J[0]) <= tol):
        return {"kind": "uniform", "B_T": [float(v) for v in J[0]], "exact": True}
    rho = np.hypot(x, y)
    if rho.min() <= 0:
        raise ImportError_("Je is not uniform and its domain touches the z axis: no prescribed-source kind fits")
    ph = np.stack([-y / rho, x / rho, np.zeros_like(x)], axis=1)
    Jphi = (J * ph).sum(axis=1)
    rest = J - Jphi[:, None] * ph
    if np.abs(rest).max() > 1e-7 * scale:
        raise ImportError_("Je is neither uniform nor azimuthal (phi-hat) about the z axis; only those map to "
                           "axionFEM sources (uniform, toroidal_box)")
    rmin, rmax = float(rho.min()), float(rho.max())
    zmin, zmax = float(z.min()), float(z.max())
    pad = 1e-6 * max(rmax, zmax - zmin)
    if np.all(np.abs(Jphi - Jphi[0]) <= 1e-7 * scale):
        return {"kind": "toroidal_box", "B0_T": float(Jphi[0]), "r_m": [max(rmin - pad, 1e-9), rmax + pad],
                "z_m": [zmin - pad, zmax + pad], "profile": "uniform", "exact": True}
    c = Jphi * rho
    if np.all(np.abs(c - c[0]) <= 1e-7 * np.abs(c).max()):
        r0 = max(rmin - pad, 1e-9)
        return {"kind": "toroidal_box", "B0_T": float(c[0] / r0), "r_m": [r0, rmax + pad],
                "z_m": [zmin - pad, zmax + pad], "profile": "inverse_r", "exact": True}
    raise ImportError_("azimuthal Je with a profile other than uniform or 1/rho is not representable")


# ------------------------------------------------------------------------------------------------ mesh route
def sel_entities(sel, all_entities, what: str):
    """Entities of a feature selection; "all" only for an explicit selection without entities."""
    if sel is None or sel.unknown:
        raise ImportError_(f"{what}: its selection is not an explicit or named selection the importer understands; "
                           "refusing to guess which entities it covers")
    return list(all_entities) if sel.entities is None else list(sel.entities)


def _faces_of_tets(tets: np.ndarray) -> np.ndarray:
    f = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]])
    return np.sort(f, axis=1)


def _face_keys(f: np.ndarray, nv: int) -> np.ndarray:
    if float(nv) ** 3 >= 2.0 ** 63:
        raise ImportError_(f"mesh with {nv} vertices: face keys would overflow int64 (limit ~2.09 M vertices)")
    f = np.sort(f, axis=1).astype(np.int64)
    return (f[:, 0] * nv + f[:, 1]) * nv + f[:, 2]


def _tet_volume(p, t):
    a, b, c, d = (p[t[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def write_msh22(path: Path, points: np.ndarray, tets: np.ndarray, tet_attr: np.ndarray, tris: np.ndarray,
                tri_attr: np.ndarray, names: dict):
    """gmsh MSH 2.2 ASCII (what MFEM reads): nodes 1-based, type 4 = tetrahedron, 2 = triangle."""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        if names:
            f.write(f"$PhysicalNames\n{len(names)}\n")
            for (dim, tag), name in sorted(names.items()):
                f.write(f'{dim} {tag} "{name}"\n')
            f.write("$EndPhysicalNames\n")
        f.write(f"$Nodes\n{len(points)}\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(points) + 1), points]), fmt="%d %.17g %.17g %.17g")
        f.write("$EndNodes\n")
        n = len(tris) + len(tets)
        f.write(f"$Elements\n{n}\n")
        k = np.arange(1, len(tris) + 1)
        if len(tris):
            np.savetxt(f, np.column_stack([k, np.full(len(tris), 2), np.full(len(tris), 2), tri_attr, tri_attr,
                                           tris + 1]), fmt="%d")
        k = np.arange(len(tris) + 1, n + 1)
        if len(tets):
            np.savetxt(f, np.column_stack([k, np.full(len(tets), 4), np.full(len(tets), 2), tet_attr, tet_attr,
                                           tets + 1]), fmt="%d")
        f.write("$EndElements\n")


_TET_EDGES = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]   # gmsh tet10 mid-node order
_TRI_EDGES = [(0, 1), (1, 2), (2, 0)]                           # gmsh tri6


def fit_surface(X: np.ndarray, rtol: float = 1e-9):
    """Implicit surface through the points X: ("plane", (n, d)), ("quadric", q) or (None, reason).

    The quadric q (10 coefficients of x², y², z², xy, xz, yz, x, y, z, 1 in the normalised frame stored with it)
    is accepted only when it is unique (one null vector of the design matrix) and fits to ``rtol``; this covers
    COMSOL's cylinders, spheres and cones, not tori or splines."""
    if len(X) < 4:
        return None, "too few vertices"
    c = X.mean(axis=0)
    sc = float(np.abs(X - c).max()) or 1.0
    Y = (X - c) / sc
    A = np.column_stack([Y, np.ones(len(Y))])
    _, sv, vt = np.linalg.svd(A, full_matrices=False)
    if sv[-1] <= rtol * sv[0] * math.sqrt(len(Y)):
        return "plane", None
    if len(Y) < 12:
        return None, "too few vertices for a quadric"
    x, y, z = Y.T
    D = np.column_stack([x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, np.ones_like(x)])
    _, sv, vt = np.linalg.svd(D, full_matrices=False)
    if sv[-1] > 1e-4 * sv[0] or sv[-2] < 1e-4 * sv[0] or sv[-2] < 100 * sv[-1]:
        # not unique: typically a band meshed with vertices on its rim circles only, which many quadrics contain.
        # Assume straight generators (cylinder or cone through coaxial rings); a sphere only with >= 3 rings.
        cf = _axis_cylinder_fit(Y, X, c, sc)
        if cf is not None:
            return "quadric", cf
        rf = _ring_fit(Y, X, c, sc)
        if rf is not None:
            return "quadric", rf
        return None, f"no unique quadric (singular values {sv[-1] / sv[0]:.1e}, {sv[-2] / sv[0]:.1e})"
    fit = {"q": vt[-1], "c": c, "s": sc}
    # geometric residual: |f|/|grad f| at the vertices
    res = max(abs(f) / max(np.linalg.norm(g), 1e-300) for f, g in (_quadric_fg(fit, x) for x in X[:: max(1, len(X) // 200)]))
    if res > 2e-4 * sc:      # COMSOL mesh vertices sit within ~1e-5 relative of the CAD surfaces
        return None, f"quadric fit residual {res:.2e} m"
    fit["residual_m"] = float(res)
    return "quadric", fit


def _axis_cylinder_fit(Y, X, c, sc, tol=2e-4):
    """A circular cylinder about a coordinate axis through the vertices (preferred over spheres and cones when the
    general quadric is not unique, e.g. short bands; a cone or sphere band does not fit a cylinder)."""
    best = None
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        u, v = Y[:, i], Y[:, j]
        A = np.column_stack([u, v, np.ones_like(u)])
        if np.linalg.matrix_rank(A, tol=1e-9) < 3:
            continue
        sol, *_ = np.linalg.lstsq(A, -(u * u + v * v), rcond=None)
        q = np.zeros(10)
        q[i] = q[j] = 1.0
        q[6 + i], q[6 + j], q[9] = sol
        if sol[0] ** 2 / 4 + sol[1] ** 2 / 4 - sol[2] <= 0:
            continue
        fit = {"q": q, "c": c, "s": sc, "family": "cylinder"}
        res = max(abs(f) / max(np.linalg.norm(g), 1e-300) for f, g in (_quadric_fg(fit, x) for x in X))
        if res <= tol * sc and (best is None or res < best[0]):
            fit["residual_m"] = float(res)
            best = (res, fit)
    return best[1] if best else None


def _ring_fit(Y, X, c, sc, tol=2e-4):
    """Vertices (normalised Y) on >= 2 coaxial rings about a coordinate axis -> the ruled surface through them
    (cylinder if the radii agree, else a cone), or a sphere when >= 3 rings are not on a straight generator."""
    best = None
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        w = Y[:, k]
        order = np.argsort(w)
        levels, cur = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if w[b] - w[a] <= 1e-5:
                cur.append(b)
            else:
                levels.append(cur)
                cur = [b]
        levels.append(cur)
        if len(levels) < 2 or any(len(lv) < 3 for lv in levels):
            continue
        cen, rad, wl, ok = [], [], [], True
        for lv in levels:
            u, v = Y[lv, i], Y[lv, j]
            A = np.column_stack([u, v, np.ones_like(u)])
            sol, *_ = np.linalg.lstsq(A, -(u * u + v * v), rcond=None)
            cu, cv = -sol[0] / 2, -sol[1] / 2
            R2 = cu * cu + cv * cv - sol[2]
            if R2 <= 0:
                ok = False
                break
            R = math.sqrt(R2)
            if np.abs(np.hypot(u - cu, v - cv) - R).max() > tol:
                ok = False
                break
            cen.append((cu, cv))
            rad.append(R)
            wl.append(float(w[lv].mean()))
        if not ok:
            continue
        cen = np.array(cen)
        if np.abs(cen - cen.mean(axis=0)).max() > tol:
            continue
        cu, cv = cen.mean(axis=0)
        wl, rad = np.array(wl), np.array(rad)
        b_, a_ = np.polyfit(wl, rad, 1)
        ruled = np.abs(a_ + b_ * wl - rad).max() <= tol
        q = np.zeros(10)
        if ruled:
            q[i] = q[j] = 1.0
            q[k] = -b_ * b_
            q[6 + i], q[6 + j], q[6 + k] = -2 * cu, -2 * cv, -2 * a_ * b_
            q[9] = cu * cu + cv * cv - a_ * a_
            fam = "cylinder" if abs(b_) * np.ptp(wl) <= tol else "cone"
        elif len(levels) >= 3:
            # sphere centred on the axis: (u-cu)^2 + (v-cv)^2 + (w-cw)^2 = R^2  <=>  R(w)^2 = R^2 - (w-cw)^2
            p2 = np.polyfit(wl, rad ** 2, 2)
            if abs(p2[0] + 1) > 1e-3 or np.abs(np.polyval(p2, wl) - rad ** 2).max() > tol:
                continue
            q[0] = q[1] = q[2] = 1.0
            q[6 + i], q[6 + j], q[6 + k] = -2 * cu, -2 * cv, -p2[1]
            q[9] = cu * cu + cv * cv - p2[2]
            fam = "sphere"
        else:
            continue
        fit = {"q": q, "c": c, "s": sc, "family": fam}
        res = max(abs(f) / max(np.linalg.norm(g), 1e-300) for f, g in (_quadric_fg(fit, x) for x in X))
        if res <= 2e-4 * sc and (best is None or res < best[0]):
            fit["residual_m"] = float(res)
            best = (res, fit)
    return best[1] if best else None


def _quadric_fg(fit, p):
    q, c, sc = fit["q"], fit["c"], fit["s"]
    x, y, z = (p - c) / sc
    f = q[0] * x * x + q[1] * y * y + q[2] * z * z + q[3] * x * y + q[4] * x * z + q[5] * y * z + q[6] * x + q[7] * y + q[8] * z + q[9]
    g = np.array([2 * q[0] * x + q[3] * y + q[4] * z + q[6], 2 * q[1] * y + q[3] * x + q[5] * z + q[7],
                  2 * q[2] * z + q[4] * x + q[5] * y + q[8]]) / sc
    return f, g


def curve_midpoints(P: np.ndarray, tris: np.ndarray, tbnd: np.ndarray, report: ImportReport | None = None,
                    bnd_names=None):
    """Mid-edge nodes of the boundary-triangle edges moved onto the fitted boundary surfaces.

    Returns {edge (a, b) with a < b: point}.  An edge on several curved boundaries is moved onto their
    intersection (Newton, minimum-norm steps); a move larger than 0.3 x the edge length is rejected."""
    fits = {}
    for b in np.unique(tbnd):
        V = P[np.unique(tris[tbnd == b])]
        kind, fit = fit_surface(V)
        fits[int(b)] = (kind, fit)
    edge_b: dict = {}
    for k in range(3):
        a, bb = tris[:, k], tris[:, (k + 1) % 3]
        for e0, e1, b in zip(np.minimum(a, bb), np.maximum(a, bb), tbnd):
            edge_b.setdefault((int(e0), int(e1)), set()).add(int(b))
    out, moved, rejected, maxd = {}, 0, 0, 0.0
    for (a, bb), bs in edge_b.items():
        quads = [fits[b][1] for b in bs if fits[b][0] == "quadric"]
        if not quads:
            continue
        m0 = 0.5 * (P[a] + P[bb])
        planes = []
        for b in bs:
            if fits[b][0] == "plane":
                V = P[np.unique(tris[tbnd == b])]
                cc = V.mean(axis=0)
                n = np.linalg.svd(V - cc)[2][-1]
                planes.append((n, cc))
        # constraints: keep one of several (nearly) tangent surfaces (e.g. the two fitted halves of one cylinder
        # meeting at a seam), otherwise their "intersection" is ill-defined
        cons = [("q", fq) for fq in quads] + [("p", pl) for pl in planes]
        kept, normals = [], []
        for kind, c_ in cons:
            g = _quadric_fg(c_, m0)[1] if kind == "q" else c_[0]
            g = g / (np.linalg.norm(g) or 1.0)
            if all(abs(g @ n) < math.cos(math.radians(10)) for n in normals):
                kept.append((kind, c_))
                normals.append(g)
        m = m0.copy()
        for _ in range(30):
            F, J = [], []
            for kind, c_ in kept:
                if kind == "q":
                    f, g = _quadric_fg(c_, m)
                else:
                    f, g = c_[0] @ (m - c_[1]), c_[0]
                F.append(f)
                J.append(g)
            F, J = np.array(F), np.array(J)
            try:
                step = J.T @ np.linalg.lstsq(J @ J.T, F, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            m = m - step
            if np.linalg.norm(step) < 1e-15 * (1 + np.linalg.norm(m)):
                break
        d = float(np.linalg.norm(m - m0))
        ev = P[bb] - P[a]
        le = float(np.linalg.norm(ev))
        if not np.isfinite(d) or d > 0.15 * le or abs((m - m0) @ ev) > 0.02 * le * le:   # sagitta: normal to the chord
            rejected += 1
            continue
        if d > 0:
            out[(a, bb)] = m
            moved += 1
            maxd = max(maxd, d)
    if report is not None:
        kinds = {}
        for b, (k, f) in fits.items():
            kinds[k or "unfitted"] = kinds.get(k or "unfitted", 0) + 1
        report.summary["curving"] = {"boundaries": kinds, "moved_mid_nodes": moved, "rejected_moves": rejected,
                                     "max_displacement_m": maxd,
                                     "unfitted": {b: f for b, (k, f) in fits.items() if k is None}}
    return out


def write_msh22_order2(path: Path, points: np.ndarray, tets: np.ndarray, tet_attr: np.ndarray, tris: np.ndarray,
                       tri_attr: np.ndarray, names: dict, curved: dict):
    """gmsh MSH 2.2 with second-order tetrahedra (type 11) and triangles (type 9): mid-edge nodes at the edge
    midpoints, except boundary edges listed in ``curved`` ({(a, b): point}, vertex indices into ``points``)."""
    nv = len(points)
    E = np.concatenate([np.sort(tets[:, list(e)], axis=1) for e in _TET_EDGES])
    ue, inv = np.unique(E, axis=0, return_inverse=True)
    inv = inv.reshape(len(_TET_EDGES), len(tets)).T
    mid = 0.5 * (points[ue[:, 0]] + points[ue[:, 1]])
    key = ue[:, 0].astype(np.int64) * nv + ue[:, 1]
    for (a, b), m in curved.items():
        i = np.searchsorted(key, a * nv + b)
        if i < len(key) and key[i] == a * nv + b:
            mid[i] = m
    allp = np.vstack([points, mid])
    t10 = np.column_stack([tets, nv + inv])
    TE = np.stack([np.sort(tris[:, list(e)], axis=1) for e in _TRI_EDGES], axis=1)
    tk = TE[:, :, 0].astype(np.int64) * nv + TE[:, :, 1]
    ti = np.searchsorted(key, tk)
    if np.any(key[np.clip(ti, 0, len(key) - 1)] != tk):
        raise ImportError_("a boundary triangle edge is not an edge of the tetrahedra")
    t6 = np.column_stack([tris, nv + ti])
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        if names:
            f.write(f"$PhysicalNames\n{len(names)}\n")
            for (dim, tag), name in sorted(names.items()):
                f.write(f'{dim} {tag} "{name}"\n')
            f.write("$EndPhysicalNames\n")
        f.write(f"$Nodes\n{len(allp)}\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(allp) + 1), allp]), fmt="%d %.17g %.17g %.17g")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(t6) + len(t10)}\n")
        if len(t6):
            np.savetxt(f, np.column_stack([np.arange(1, len(t6) + 1), np.full(len(t6), 9), np.full(len(t6), 2),
                                           tri_attr, tri_attr, t6 + 1]), fmt="%d")
        if len(t10):
            np.savetxt(f, np.column_stack([np.arange(len(t6) + 1, len(t6) + len(t10) + 1), np.full(len(t10), 11),
                                           np.full(len(t10), 2), tet_attr, tet_attr, t10 + 1]), fmt="%d")
        f.write("$EndElements\n")
    return allp, t10


def _quadratic_volumes(allp, t10, attr):
    """Volume per attribute of second-order tetrahedra (Keast degree-4 rule, exact for det J of degree 3), and the
    smallest det J at the quadrature points relative to the tetrahedron's linear det J (< 0: an inverted element)."""
    # Keast 11-point rule (degree 4) on the reference tetrahedron
    r = np.array([[0.25, 0.25, 0.25], [0.0714285714285714, 0.0714285714285714, 0.0714285714285714],
                  [0.785714285714286, 0.0714285714285714, 0.0714285714285714],
                  [0.0714285714285714, 0.785714285714286, 0.0714285714285714],
                  [0.0714285714285714, 0.0714285714285714, 0.785714285714286],
                  [0.399403576166799, 0.399403576166799, 0.100596423833201],
                  [0.399403576166799, 0.100596423833201, 0.399403576166799],
                  [0.100596423833201, 0.399403576166799, 0.399403576166799],
                  [0.399403576166799, 0.100596423833201, 0.100596423833201],
                  [0.100596423833201, 0.399403576166799, 0.100596423833201],
                  [0.100596423833201, 0.100596423833201, 0.399403576166799]])
    w = np.array([-0.0131555555555556] + [0.00762222222222222] * 4 + [0.0248888888888889] * 6)
    vols = {}
    X = allp[t10]                                   # (n, 10, 3)
    A = X[:, 1:4, :] - X[:, :1, :]
    det_lin = np.linalg.det(np.transpose(A, (0, 2, 1)))
    min_ratio = np.inf
    for (u, v, s_), wq in zip(r, w):
        l = np.array([1 - u - v - s_, u, v, s_])
        # gradients of the 10 quadratic shape functions wrt (u, v, s)
        dl = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        G = np.zeros((10, 3))
        for i in range(4):
            G[i] = (4 * l[i] - 1) * dl[i]
        for k, (i, j) in enumerate(_TET_EDGES):
            G[4 + k] = 4 * (l[i] * dl[j] + l[j] * dl[i])
        J = np.einsum("nkd,ke->nde", X, G)
        det = np.linalg.det(J)
        min_ratio = min(min_ratio, float((det / det_lin).min()))
        for a in np.unique(attr):
            vols[int(a)] = vols.get(int(a), 0.0) + float((det[attr == a] * wq).sum())
    return vols, min_ratio


@dataclass
class MeshRoute:
    design: dict
    info: dict
    mesh_path: Path
    info_path: Path


def build_mesh_route(model: ComsolModel, report: ImportReport, out_dir: Path, stem: str, *,
                     frequencies=None, max_frequencies: int = 20, study: str | None = None,
                     order: int | None = None, physics_tag: str | None = None, ports_toward=None,
                     curved: bool = True, mesh_resource: str | None = None,
                     axisymmetric: bool | None = None) -> MeshRoute:
    t0 = time.time()
    # ---- physics interface
    emws = [p for p in model.physics if p["op"] == "ElectromagneticWaves" and not p["disabled"]]
    for p in model.physics:
        if p["op"] != "ElectromagneticWaves":
            report.add("physics", f"{p['tag']} ({p['op']})", "unsupported",
                       detail="only Electromagnetic Waves, Frequency Domain (emw) is mapped by the importer",
                       changes_results=not p["disabled"])
    if physics_tag:
        emws = [p for p in emws if p["tag"] == physics_tag]
    if not emws:
        raise ImportError_("the model has no active Electromagnetic Waves, Frequency Domain (emw) interface; "
                           "nothing to run in axionFEM's full-wave solvers (2D or 3D)")
    if len(emws) > 1:
        report.add("physics", ", ".join(p["tag"] for p in emws[1:]), "unsupported",
                   detail="only the first emw interface is imported", changes_results=True)
    ph = emws[0]
    geom = ph["geom"]
    g = model.geometries.get(geom)
    if g is not None and g["sdim"] == 2:   # 2D / 2D-axisymmetric component
        from .comsol_import2d import build_mesh_route_2d
        return build_mesh_route_2d(model, report, out_dir, stem, ph, frequencies=frequencies,
                                   max_frequencies=max_frequencies, study=study, order=order, ports_toward=ports_toward,
                                   curved=curved, mesh_resource=mesh_resource, axisymmetric=axisymmetric)
    if g is None or g["sdim"] != 3:
        raise ImportError_(f"emw interface {ph['tag']} lives on geometry {geom!r} of dimension "
                           f"{g['sdim'] if g else '?'}; the mesh route supports 3D, 2D and 2D-axisymmetric components")
    if (g.get("lengthUnit") or "m") != "m":
        raise ImportError_(f"geometry {geom} uses the length unit {g['lengthUnit']}: how COMSOL stores mesh coordinates "
                           "for non-metre geometries is not verified yet; re-save the model with length unit m")
    solve_for = ph["props"].get("SolveFor", "fullField")
    if solve_for != "fullField":
        raise ImportError_(f"emw {ph['tag']} solves for the {solve_for} (background field): only the full-field "
                           "formulation is mapped")
    # ---- mesh
    meshes = [m for m in model.meshes.values() if m["geom"] == geom and m["resource"]]
    if not meshes:
        raise ImportError_(f"no stored mesh for geometry {geom}: save the .mph with its mesh (Mesh > Build All), "
                           "or use the replay route")
    msq = dict(meshes[0])
    if mesh_resource:
        if not model.archive.has(mesh_resource):
            raise ImportError_(f"{mesh_resource} is not in the archive ({[n for n in model.archive.names if n.startswith('mesh')]})")
        report.add("mesh", mesh_resource, "approximated", detail=f"mesh resource chosen by the user instead of the "
                   f"active {msq['resource']}: an inactive mesh may belong to an earlier geometry (other boundary "
                   "numbering); check the port geometry in the summary", changes_results=False)
        report.warnings.append(f"--mesh-file {mesh_resource}: not the active mesh; selections are those of the current model")
        msq["resource"] = mesh_resource
    data = model.archive.read(msq["resource"])
    mesh = mb.read_mesh(data, msq["resource"])
    report.add("mesh", f"{msq['tag']} -> {msq['resource']}", "mapped", "mesh.file",
               f"decoded Mesh v{mesh.version}: {len(mesh.points)} vertices, " +
               ", ".join(f"{len(v)} {k}" for k, v in mesh.elements.items()))
    for k in mesh.elements:
        if k not in ("vtx", "edg", "tri", "tet"):
            raise ImportError_(f"mesh element type {k} is not supported by the mesh route (tetrahedra only)")
    P = mesh.points
    tets = mesh.elements["tet"]
    tdom = mesh.entities["tet"]
    tris = mesh.elements["tri"]
    tbnd = mesh.entities["tri"] + 1          # 1-based COMSOL boundary numbers (GUI numbering)
    counts = dict(g["counts"])
    ndom = counts.get(3) or (msq["numEntities"][3] if msq["numEntities"] else int(tdom.max()))
    nbnd = counts.get(2) or (msq["numEntities"][2] if msq["numEntities"] else int(tbnd.max()))
    # ---- physics domains
    phys_doms = sorted(sel_entities(ph["selection"], range(1, ndom + 1), f"emw interface {ph['tag']}"))
    meshed = set(int(x) for x in np.unique(tdom))
    missing = [d for d in phys_doms if d not in meshed]
    if missing:
        raise ImportError_(f"emw domains {missing} have no elements in the stored mesh")
    excluded = sorted(set(range(1, ndom + 1)) - set(phys_doms))
    report.add("physics", f"{ph['tag']} selection", "mapped", "regions",
               f"emw on domains {phys_doms}; domains {excluded} are outside the interface: their surfaces are "
               "exterior boundaries (COMSOL default PEC), their volumes are not meshed in axionFEM")
    keep = np.isin(tdom, phys_doms)
    T = tets[keep]
    Td = tdom[keep]
    # orientation (positive volume)
    vol = _tet_volume(P, T)
    neg = vol < 0
    T[neg] = T[neg][:, [0, 2, 1, 3]]
    if np.any(np.abs(vol) < 1e-30):
        raise ImportError_("degenerate tetrahedra in the stored mesh")
    # ---- features
    feats = [f for f in ph["features"]]
    frq = study_frequencies(model, report, study)
    fl = frequencies if frequencies is not None else choose_frequencies(frq["frequencies"], max_frequencies)
    if frequencies is None and len(set(frq["frequencies"])) > len(fl):
        report.add("study", "frequency list", "approximated", "fullwave.frequencies",
                   f"{len(set(frq['frequencies']))} study frequencies subsampled to {len(fl)} "
                   f"(--max-frequencies; pass --frequencies to choose)")
    if not fl and not frq["modes"]:
        report.add("study", "(none)", "unsupported", detail="no frequency-domain or eigenfrequency study: give "
                   "frequencies explicitly (--frequencies)", changes_results=True)
    # domains -> material, wee
    dmat = domain_materials(model, geom, ndom, report)
    bmat = boundary_materials(model, geom, 3, nbnd)
    gparams: dict = {}     # feature tag -> extra boundary-group parameters (impedance, sheet, scattering)
    wee_of = {}
    for f in feats:
        if f.op == "WaveEquationElectric" and not f.disabled:
            for d in sel_entities(f.selection, phys_doms, _feat_desc(f)):
                wee_of[d] = f
    materials = {}
    region_of = {}
    regions = []
    mat_key = {}
    for d in phys_doms:
        mat = dmat.get(d)
        try:
            props = material_properties(model, mat, wee_of.get(d), fl, report)
        except (ImportError_, ExpressionError) as e:
            raise ImportError_(f"domain {d}: {e}") from e
        key = (round(props["eps_r"], 12), round(props["tan_delta"], 14), round(props["mu_r"], 12),
               round(props["sigma_S_m"], 9))
        if key == (1.0, 0.0, 1.0, 0.0):
            mname = "vacuum"
        else:
            if key not in mat_key:
                base = "comsol_" + _slug((mat["name"] if mat else "material") + "_" + (mat["tag"] if mat else ""))
                nm = base
                k = 2
                while nm in materials:
                    nm, k = f"{base}_{k}", k + 1
                mat_key[key] = nm
                materials[nm] = {"kind": "linear", "eps_r": props["eps_r"], "mu_r": props["mu_r"],
                                 "tan_delta": props["tan_delta"], "sigma_S_m": props["sigma_S_m"]}
            mname = mat_key[key]
        desc = f"domain {d} <- {(mat['tag'] + ' (' + mat['name'] + ')') if mat else 'no material'}"
        detail = (f"eps_r {props['eps_r']:.6g}, tan_delta {props['tan_delta']:.3g}, mu_r {props['mu_r']:.6g}, "
                  f"sigma {props['sigma_S_m']:.3g} S/m ({props['model']})")
        if props["notes"]:
            detail += "; " + "; ".join(props["notes"])
        if props["frequency_dependence"]:
            report.add("material", desc, "approximated", mname, detail + "; frequency dependent over the study band: "
                       + json.dumps(props["frequency_dependence"]) + " — axionFEM materials are constant (median used)",
                       changes_results=False)
        else:
            report.add("material", desc, "mapped", mname, detail)
        rname = f"domain_{d}"
        region_of[d] = rname
        regions.append({"name": rname, "material": mname,
                        "solid": {"imported": {"comsol_domains": [d], "comsol_material": mat["tag"] if mat else None}}})
    # ---- tet attributes
    attr_of_dom = {d: REGION_ATTR0 + i for i, d in enumerate(phys_doms)}
    tet_attr = np.vectorize(attr_of_dom.get)(Td)
    # ---- faces: adjacency to kept tets
    nv = len(P)
    fk = _face_keys(_faces_of_tets(T), nv)
    uk, cnt = np.unique(fk, return_counts=True)
    tri_k = _face_keys(tris, nv)
    pos = np.searchsorted(uk, tri_k)
    pos = np.clip(pos, 0, len(uk) - 1)
    tri_adj = np.where(uk[pos] == tri_k, cnt[pos], 0)
    ext_keys = uk[cnt == 1]
    covered = np.isin(ext_keys, tri_k)
    if not covered.all():
        raise ImportError_(f"{(~covered).sum()} exterior faces of the emw domains have no boundary triangle in the "
                           "stored mesh")
    btype: dict[int, tuple] = {}   # boundary -> (type, feature tag)
    bset_ext = set(int(b) for b in np.unique(tbnd[tri_adj == 1]))
    bset_int = set(int(b) for b in np.unique(tbnd[tri_adj == 2]))
    bset_all = bset_ext | bset_int
    mixed = bset_ext & bset_int
    if mixed:
        report.warnings.append(f"boundaries {sorted(mixed)} are partly exterior and partly interior to the emw domains")
    for b in bset_ext:
        btype[b] = ("pec", "pec1 (default: exterior)")
    lports, sources, notes = [], [], []
    pmc_feats = []
    for f in feats:
        desc = _feat_desc(f)
        if f.disabled and f.op != "ExternalCurrentDensity":
            report.add("physics", desc, "ignored", detail="disabled in the saved model")
            continue
        if f.selection is not None and f.selection.unknown:
            raise ImportError_(f"{desc}: its selection is not an explicit or named selection the importer understands")
        ents = f.selection.entities if f.selection else None
        for sub in f.children:
            if not sub.disabled and sub.op not in ("UniformElement",):
                report.add("physics", f"{desc} / {_feat_desc(sub)}", "unsupported",
                           detail="sub-feature not mapped", changes_results=True)
        if f.op in ("WaveEquationElectric",):
            report.add("physics", desc, "mapped", "materials", "constitutive relations per domain (see materials)")
        elif f.op in ("init", "InitialValues"):
            report.add("physics", desc, "ignored", detail="initial values do not affect a linear frequency-domain solve")
        elif f.op == "PerfectElectricConductor":
            if ents is None and f.default:
                report.add("physics", desc, "mapped", "boundary_groups type pec", "all exterior boundaries")
                continue
            if ents is None:
                ents = sorted(bset_all)
            for b in ents:
                btype[b] = ("pec", f.tag)
            report.add("physics", desc, "mapped", "boundary_groups type pec",
                       f"boundaries {ents}" + ("; interior (zero-thickness) sheets are PEC on both sides"
                                               if set(ents) & bset_int else ""))
        elif f.op == "PerfectMagneticConductor":
            pm = ents if ents is not None else sorted(bset_ext)
            if set(pm) & (bset_int - bset_ext):
                raise ImportError_(f"{desc}: PMC on interior boundaries {sorted(set(pm) & (bset_int - bset_ext))} is "
                                   "not supported (only exterior PMC walls)")
            for b in pm:
                btype[b] = ("pmc", f.tag)
            pmc_feats.append(f.tag)
            report.add("physics", desc, "mapped", "boundary_groups type pmc", f"boundaries {ents}")
        elif f.op == "LumpedPort":
            if ents is None:
                raise ImportError_(f"{desc}: lumped port without an explicit boundary list")
            ptype = f.p("PortType", "Uniform")
            if ptype != "Uniform":
                report.add("physics", desc, "unsupported", detail=f"PortType {ptype}: only Uniform lumped ports map "
                           "to axionFEM's surface port", changes_results=True)
                continue
            for b in ents:
                btype[b] = ("port", f.tag)
            lports.append(f)
        elif f.op == "ExternalCurrentDensity":
            sources.append(f)
        elif f.op in ("Continuity",):
            report.add("physics", desc, "ignored", detail="continuity on identity pairs: the imported mesh is "
                       "conforming (Form Union), so there are no pairs")
        elif f.op in ("ImpedanceBoundaryCondition", "Impedance", "TransitionBoundaryCondition", "Scattering",
                      "ScatteringBoundaryCondition"):
            if ents is None:
                ents = sorted(bset_ext) if f.op != "TransitionBoundaryCondition" else sorted(bset_int)
            try:
                gparams[f.tag] = robin_feature(model, f, ents, bset_ext, bset_int, bmat, fl, report, materials)
            except (ImportError_, ExpressionError) as e:
                report.add("physics", desc, "unsupported", detail=str(e), changes_results=True)
                continue
            for b in ents:
                btype[b] = (gparams[f.tag]["type"], f.tag)
        elif f.op in ("PerfectlyMatchedLayer", "Port", "CoaxialPort",
                      "SurfaceCurrentDensity", "SurfaceCurrent", "EdgeCurrent", "ElectricPointDipole",
                      "MagneticPointDipole", "PeriodicCondition", "Floquet", "DiffractionOrder", "FarFieldDomain",
                      "AnisotropicImpedanceBoundaryCondition", "LayeredImpedanceBoundaryCondition",
                      "ElectricField", "MagneticField", "MagneticCurrent", "SurfaceMagneticCurrentDensity"):
            report.add("physics", desc, "unsupported", detail=f"{f.op} has no axionFEM counterpart on the mesh route "
                       "(closed PEC domain, lumped ports, prescribed source)", changes_results=True)
        else:
            report.add("physics", desc, "unsupported", detail=f"unknown emw feature {f.op}", changes_results=True)
    # ---- source
    source = None
    source_regions = []
    active = [s for s in sources if not s.disabled]
    chosen = active if active else sources
    if len(chosen) > 1:
        report.add("physics", ", ".join(s.tag for s in chosen[1:]), "unsupported",
                   detail="several External Current Density features: only the first is mapped", changes_results=True)
    if chosen:
        s = chosen[0]
        je = s.params.get("Je")
        ents = [d for d in sel_entities(s.selection, phys_doms, _feat_desc(s)) if d in phys_doms]
        if not ents:
            raise ImportError_(f"{_feat_desc(s)}: no emw domain in its selection")
        if (s.p("coordinateSystem") or "GlobalSystem") != "GlobalSystem":
            raise ImportError_(f"{_feat_desc(s)}: Je given in coordinate system {s.p('coordinateSystem')}; only the "
                               "global Cartesian system is mapped")
        sel_t = np.isin(Td, ents)
        cen = P[T[sel_t]].mean(axis=1)
        try:
            src = classify_source(model, je, np.vstack([cen, P[np.unique(T[sel_t])]]))
            exact = src.pop("exact")
            source = {**src, "metadata": {"comsol_feature": s.tag, "Je_A_m2": je, "domains": ents,
                                          "normalization": "B0 [T] = Je [A/m^2] / (1 A m^-2 T^-1)"}}
            source_regions = [region_of[d] for d in ents]
            report.add("physics", _feat_desc(s), "mapped", f"fullwave.source ({src['kind']}) + source_regions",
                       f"Je = {je} on domains {ents}" + ("" if not s.disabled else
                       " — DISABLED in the saved model (an impedance run); mapped anyway: axionFEM computes Z and the "
                       "source overlap Psi in one solve"))
        except (ImportError_, ExpressionError) as e:
            # a disabled source (an impedance run) does not change Z: report it, do not block the import
            report.add("physics", _feat_desc(s), "unsupported", detail=str(e) + (" (disabled in the saved model: "
                       "Z is unaffected, no Psi)" if s.disabled else ""), changes_results=not s.disabled)
    else:
        report.add("physics", "(no External Current Density)", "info",
                   detail="no axion source in the model: only the port impedance is meaningful")
    # ---- boundary groups, port
    groups = []
    for b in sorted(bset_all):
        t = btype.get(b)
        if t is None or t[0] == "port":
            continue
        groups.append({"attr": BOUNDARY_ATTR0 + b, "name": f"comsol_boundary_{b}", "type": t[0], "comsol_boundary": b,
                       "feature": t[1], "exterior": b in bset_ext, **gparams.get(t[1], {})})
    attr_tri = BOUNDARY_ATTR0 + tbnd
    pickup = None
    if len(lports) > 1:
        report.add("physics", ", ".join(p.tag for p in lports[1:]), "unsupported",
                   detail="several lumped ports: axionFEM imports the first as the readout port; additional surface "
                          "ports are not implemented", changes_results=True)
    tri_keep = tri_adj > 0
    port_info = None
    if lports:
        lp = lports[0]
        pb = [b for b in (lp.selection.entities or []) if btype.get(b, ("",))[0] == "port"]
        pattr = PORT_ATTR0 + 1
        mask = np.isin(tbnd, pb)
        attr_tri = np.where(mask, pattr, attr_tri)
        groups.append({"attr": pattr, "name": lp.tag, "type": "port", "comsol_boundaries": pb, "feature": lp.tag})
        port_info = _port_geometry(P, tris[mask & tri_keep], tris, tbnd, btype, tri_keep)
        toward = port_info["toward"] if ports_toward is None else list(ports_toward)
        pickup = {"type": "port", "port_boundary": lp.tag, "toward": toward,
                  "notes": f"COMSOL {lp.tag}: Uniform lumped port on boundaries {pb}, Zref {lp.p('Zref')}, "
                           f"excitation {lp.p('PortExcitation')}"}
        try:
            zref = float(np.real(_num(model, lp.p("Zref", "50[ohm]"))))
        except ExpressionError:
            zref = 50.0
        port_info.update({"Zref_ohm": zref, "PortExcitation": lp.p("PortExcitation"), "comsol_boundaries": pb})
        report.add("physics", _feat_desc(lp), "mapped", "pickup.port_boundary (surface port)",
                   f"Uniform lumped port on boundaries {pb}: width {port_info['width_m']:.4g} m, length "
                   f"{port_info['length_m']:.4g} m; uniform surface current between the two PEC edges. Zref "
                   f"{zref:g} ohm is not part of axionFEM's Z (open-port impedance); a non-driven COMSOL port reads "
                   "V_port = V_oc Zref/(Zref + Zport)")
    else:
        report.add("physics", "(no lumped port)", "unsupported", detail="axionFEM's 3D full-wave sweep needs a port",
                   changes_results=True)
    # ---- unused boundaries / sanity
    ttype = {b: t[0] for b, t in btype.items()}
    ext_untyped = [b for b in bset_ext if b not in ttype]
    if ext_untyped:
        raise ImportError_(f"internal: exterior boundaries {ext_untyped} without a boundary condition")
    # ---- write the mesh
    used_tri = tri_keep
    TR = tris[used_tri]
    TA = attr_tri[used_tri]
    used_v = np.unique(np.concatenate([T.ravel(), TR.ravel()]))
    remap = -np.ones(len(P), dtype=np.int64)
    remap[used_v] = np.arange(len(used_v))
    pts = P[used_v]
    T2, TR2 = remap[T], remap[TR]
    names = {(3, attr_of_dom[d]): region_of[d] for d in phys_doms}
    for gr in groups:
        names[(2, gr["attr"])] = gr["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = (out_dir / f"{stem}.msh").resolve()
    info_path = (out_dir / f"{stem}_mesh_info.json").resolve()
    vol_lin = {}
    tv = np.abs(_tet_volume(pts, T2))
    for d in phys_doms:
        vol_lin[d] = float(tv[tet_attr == attr_of_dom[d]].sum())
    vol_quad = None
    if curved:
        cm = curve_midpoints(pts, TR2, tbnd[used_tri], report)
        allp, t10 = write_msh22_order2(mesh_path, pts, T2, tet_attr, TR2, TA, names, cm)
        vq, min_ratio = _quadratic_volumes(allp, t10, tet_attr)
        vol_quad = {d: vq[attr_of_dom[d]] for d in phys_doms}
        report.summary.setdefault("curving", {})["min_detJ_over_linear"] = min_ratio
        if min_ratio <= 0:
            report.warnings.append(f"curving produced inverted second-order elements (min det J ratio {min_ratio:.3g}); "
                                   "axionFEM straightens them at load time, use --straight to avoid")
        cv = report.summary.get("curving", {})
        report.add("mesh", "curved geometry", "approximated", "second-order (curved) tetrahedra",
                   f"COMSOL maps boundary elements onto the CAD (geometry shape order 2); the stored mesh has only "
                   f"vertices. The importer fits a plane or a unique quadric (cylinder, sphere, cone) to the vertices "
                   f"of every COMSOL boundary and moves the mid-edge nodes of the boundary edges onto it: "
                   f"{cv.get('moved_mid_nodes')} nodes moved (max {cv.get('max_displacement_m', 0):.3g} m), "
                   f"boundaries {cv.get('boundaries')}; other surfaces (tori, splines) stay straight-sided")
    else:
        write_msh22(mesh_path, pts, T2, tet_attr, TR2, TA, names)
        report.add("mesh", "curved geometry", "approximated", "straight-sided tetrahedra",
                   "COMSOL maps boundary elements onto the CAD surfaces (geometry shape order 2 by default); here "
                   "curved boundaries are represented by their chords (sagitta ~h^2/(8R)); use curved=True")
    report.summary["domain_volumes_m3"] = {d: {"straight": vol_lin[d], **({"curved": vol_quad[d]} if vol_quad else {})}
                                           for d in phys_doms}
    c = pts.mean(axis=0)
    rdev = float(np.sqrt(((pts - c) ** 2).sum(axis=1)).max())
    info = {"ok": True, "imported_from": str(model.path), "importer": "axionfem.comsol_import (mesh route)",
            "elements_3d": int(len(T2)), "boundary_elements": int(len(TR2)), "vertices": int(len(pts)),
            "seconds": time.time() - t0, "boundary_groups": groups, "shell_attr": -1, "domain_center": c.tolist(),
            "device_radius": rdev, "domain_radius": rdev,
            "groups": [{"dim": 3, "attr": attr_of_dom[d], "name": region_of[d], "comsol_domain": d} for d in phys_doms],
            "comsol": {"version": model.version, "mesh_sequence": msq["tag"], "resource": msq["resource"],
                       "domains": ndom, "boundaries": nbnd, "emw_domains": phys_doms, "excluded_domains": excluded,
                       "attributes": {"region k": "2 + k (domain order)", "boundary b": f"{BOUNDARY_ATTR0} + b",
                                      "lumped port k": f"{PORT_ATTR0} + k"}},
            "port": port_info}
    info_path.write_text(json.dumps(info, indent=1, default=float))
    # ---- design
    shape_order = ph["props"].get("order_electricfield")
    mo = order or (int(float(shape_order)) if shape_order else 2)
    fw = {"boundary": "pec", "rf_only": True}
    pml = pml_layers(model, geom, 3, phys_doms, region_of, report)
    if pml:
        fw["pml"] = pml
        if frq["modes"]:
            report.add("study", "eigenfrequency with PML", "unsupported", detail="complex eigenmodes with PML are not "
                       "computed by axionFEM", changes_results=True)
    if fl:
        fw["frequencies"] = fl
    if frq["modes"]:
        if any(g["type"] in ("scattering", "sheet") for g in groups) or pml:
            report.add("study", "eigenfrequency", "unsupported", detail="eigenmodes with PML, scattering or transition "
                       "boundaries are complex; axionFEM's 3D eigenmodes are for closed lossless models",
                       changes_results=True)
        else:
            fw["modes"] = frq["modes"]
    if source:
        fw["source"] = source
        fw["source_regions"] = source_regions
    design = {"schema": "axionfem.design/1", "name": _slug(Path(model.path).stem), "symmetry": "3d",
              "notes": (f"Imported from {model.path} (COMSOL {model.version}) by axionfem.comsol_import, mesh route "
                        f"(beta): COMSOL's stored mesh {msq['resource']} ({len(T2)} tetrahedra on emw domains "
                        f"{phys_doms}); one region per COMSOL domain; boundary conditions as boundary groups in "
                        f"{info_path.name}. " + ("Boundary mid-edge nodes moved onto fitted planes/quadrics (curved "
                        "elements). " if curved else "Straight-sided elements. ") +
                        "See import_report.md."),
              "regions": regions, "fullwave": fw, "mesh": {"order": mo, "file": str(mesh_path), "info_file": str(info_path)}}
    if materials:
        design["materials"] = materials
    if pickup:
        design["pickup"] = pickup
    report.add("mesh", "element order", "mapped", f"mesh.order {mo}",
               f"COMSOL emw shape order {shape_order} (curl elements)")
    for mf in msq["features"]:
        report.add("mesh", _feat_desc(mf), "ignored", detail="mesh-generation setting; the stored mesh is used as is")
    report.summary.update({"comsol_version": model.version, "domains": ndom, "boundaries": nbnd,
                           "emw_domains": phys_doms, "tetrahedra": int(len(T2)),
                           "boundary_types": {b: ttype[b] for b in sorted(ttype)}, "frequencies_Hz": fl,
                           "studies": frq["studies"], "port": port_info, "source": source})
    return MeshRoute(design, info, mesh_path, info_path)


def _port_geometry(P, ptris, tris, tbnd, btype, tri_keep) -> dict:
    """Terminal chains of a surface port (edges shared with PEC boundaries), width, length and direction."""
    if len(ptris) == 0:
        raise ImportError_("the lumped port has no triangles adjacent to the emw domains")
    pec_b = {b for b, t in btype.items() if t[0] in ("pec", "impedance")}
    pec_tris = tris[np.isin(tbnd, list(pec_b)) & tri_keep]

    def edges(tr):
        e = np.concatenate([tr[:, [0, 1]], tr[:, [1, 2]], tr[:, [2, 0]]])
        return np.sort(e, axis=1)

    pe = edges(ptris)
    ue, cnt = np.unique(pe, axis=0, return_counts=True)
    outline = ue[cnt == 1]
    pece = {tuple(e) for e in edges(pec_tris)}
    term = np.array([e for e in outline if tuple(e) in pece])
    if len(term) == 0:
        raise ImportError_("the lumped port does not touch any PEC boundary")
    # chains via shared vertices (union-find)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in term:
        parent[find(a)] = find(b)
    roots = sorted({find(a) for a, _ in term})
    if len(roots) != 2:
        raise ImportError_(f"the lumped port meets PEC along {len(roots)} separate edge chains (a Uniform port needs 2)")
    L, C = [0.0, 0.0], [np.zeros(3), np.zeros(3)]
    for a, b in term:
        k = roots.index(find(a))
        l = float(np.linalg.norm(P[a] - P[b]))
        L[k] += l
        C[k] += 0.5 * l * (P[a] + P[b])
    C = [C[k] / L[k] for k in range(2)]
    t = C[1] - C[0]
    if t[int(np.argmax(np.abs(t)))] < 0:     # deterministic orientation: largest component positive
        t, C, L = -t, [C[1], C[0]], [L[1], L[0]]
    length = float(np.linalg.norm(t))
    a = P[ptris]
    area = float(0.5 * np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1).sum())
    w = 0.5 * (L[0] + L[1])
    return {"toward": (t / length).tolist(), "width_m": w, "length_m": length, "area_m2": area,
            "terminal_lengths_m": L, "terminal_centroids_m": [C[0].tolist(), C[1].tolist()],
            "rectangularity": area / (w * length), "triangles": int(len(ptris))}


# ------------------------------------------------------------------------------------------------ entry points
def import_comsol(path, out: str | Path | None = None, *, route: str = "mesh", allow_partial: bool = False,
                  frequencies=None, max_frequencies: int = 20, study: str | None = None, order: int | None = None,
                  physics: str | None = None, toward=None, write: bool = True, curved: bool = True,
                  mesh_resource: str | None = None, axisymmetric: bool | None = None):
    """Import ``path`` (a COMSOL ``.mph``) and return an :class:`axionfem.Design`.

    ``out`` is the design JSON to write (default ``<model stem>_axf/design.json`` next to the model's name in the
    current directory); the mesh, its info file and ``import_report.md``/``.json`` go next to it.  ``route`` is
    ``"mesh"``, ``"replay"`` or ``"auto"`` (replay when the whole geometry sequence is supported, else mesh).
    Raises :class:`ImportError_` when something that changes the results cannot
    be mapped, unless ``allow_partial``.  The report is attached as ``design.import_report``."""
    path = Path(path)
    out = Path(out) if out else Path.cwd() / f"{_slug(path.stem)}_axf" / "design.json"
    out_dir = out.parent
    stem = out.stem
    model = load_model(path)
    report = ImportReport(str(path))
    report.warnings += model.warnings
    _report_model(model, report)
    chosen = route
    replay = None
    if route in ("replay", "auto"):
        from . import comsol_replay
        try:
            replay = comsol_replay.build_replay_route(model, report, out_dir, stem, frequencies=frequencies,
                                                      max_frequencies=max_frequencies, study=study, order=order,
                                                      physics_tag=physics)
            chosen = "replay"
        except Exception as e:  # noqa: BLE001  (ReplayUnsupported, or a replay bug: never lose the import)
            report.add("geometry", "replay route", "info", detail=f"not possible ({e}); falling back to the mesh route")
            report.warnings.append(f"replay route not possible: {e}; the design uses the mesh route")
            chosen = "mesh"
    if chosen == "mesh":
        res = build_mesh_route(model, report, out_dir, stem, frequencies=frequencies, max_frequencies=max_frequencies,
                               study=study, order=order, physics_tag=physics, ports_toward=toward, curved=curved,
                               mesh_resource=mesh_resource, axisymmetric=axisymmetric)
        doc = res.design
    else:
        doc = replay.design
    report.route = chosen
    if report.blocking and allow_partial:
        doc["notes"] = ("INCOMPLETE IMPORT (allow_partial): unsupported items that change the result were left out, "
                        "see import_report.md. " + doc.get("notes", ""))
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not report.blocking or allow_partial:
            out.write_text(json.dumps(doc, indent=1, default=float))
        elif out.exists():
            out.unlink()                       # never leave a runnable design for a failed import
        (out_dir / "import_report.json").write_text(json.dumps(report.to_json(), indent=1, default=float))
        (out_dir / "import_report.md").write_text(report.to_markdown())
    if report.blocking and not allow_partial:
        raise ImportError_("import incomplete (see import_report.md):\n" +
                           "\n".join(f"  - {i['category']} {i['comsol']}: {i['detail']}" for i in report.blocking) +
                           "\nPass allow_partial=True (--allow-partial) to write the design anyway.")
    try:
        from .design import Design
        d = Design(doc)
    except Exception:  # noqa: BLE001  (the core may be unavailable; return the document)
        d = doc
    try:
        d.import_report = report
    except AttributeError:
        pass
    return d


def _report_model(model: ComsolModel, report: ImportReport):
    report.summary["file"] = {"name": model.name, "version": model.version}
    if model.params:
        for k, v in model.params.items():
            report.add("parameter", k, "mapped" if v.get("value") is not None else "unsupported", "values",
                       f"{v['expr']} = {v.get('value')}", changes_results=v.get("value") is None)
    else:
        report.add("parameter", "(global parameters)", "info", detail="none defined")
    for k, v in model.function_info.items():
        report.add("function", k, "mapped" if "status" not in v else "unsupported", detail=json.dumps(v)[:300])
    for t in model.tables:
        report.add("results", f"{t['tag']} ({t['name']})", "ignored", detail="result tables are not imported")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="axf import-comsol", description="Import a COMSOL .mph model as an axionFEM "
                                 "design (beta; docs/comsol_import.md).")
    ap.add_argument("model", help="COMSOL .mph file")
    ap.add_argument("-o", "--output", help="design JSON to write (default <stem>_axf/design.json)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--mesh-route", action="store_true", help="use COMSOL's stored mesh (exact; default)")
    grp.add_argument("--replay-route", action="store_true", help="rebuild the geometry from axionFEM solids")
    ap.add_argument("--allow-partial", action="store_true", help="write the design even if some physics is unsupported")
    ap.add_argument("--frequencies", help="comma-separated frequencies in Hz (default: the model's studies)")
    ap.add_argument("--max-frequencies", type=int, default=20)
    ap.add_argument("--study", help="study tag to take frequencies from (default: all frequency studies)")
    ap.add_argument("--order", type=int, help="finite-element order (default: COMSOL's emw shape order)")
    ap.add_argument("--physics", help="emw interface tag (default: the first)")
    ap.add_argument("--straight", action="store_true", help="mesh route: keep straight-sided elements (no curving)")
    ap.add_argument("--mesh-file", help="mesh route: archive member to use instead of the active mesh (e.g. mesh1.mphbin)")
    sym = ap.add_mutually_exclusive_group()
    sym.add_argument("--axisymmetric", action="store_true", help="2D models: treat as 2D axisymmetric (r, z)")
    sym.add_argument("--planar", action="store_true", help="2D models: treat as planar 2D (x, y)")
    a = ap.parse_args(argv)
    route = "replay" if a.replay_route else "mesh"
    fs = [float(x) for x in a.frequencies.split(",")] if a.frequencies else None
    try:
        d = import_comsol(a.model, a.output, route=route, allow_partial=a.allow_partial, frequencies=fs,
                          max_frequencies=a.max_frequencies, study=a.study, order=a.order, physics=a.physics,
                          curved=not a.straight, mesh_resource=a.mesh_file,
                          axisymmetric=True if a.axisymmetric else False if a.planar else None)
    except (ImportError_, ExpressionError, mb.MphbinError, ValueError) as e:
        print(f"axf import-comsol: {e}")
        return 2
    rep = getattr(d, "import_report", None)
    out = Path(a.output) if a.output else Path.cwd() / f"{_slug(Path(a.model).stem)}_axf" / "design.json"
    if rep is not None:
        cnt = rep.to_json()["counts"]
        print(f"wrote {out} ({rep.route} route; " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) +
              f"); report: {out.parent / 'import_report.md'}")
    return 0
