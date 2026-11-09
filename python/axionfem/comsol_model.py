"""Parse the model tree of a COMSOL ``.mph`` archive (``dmodel.xml``) without COMSOL.  Beta.

An ``.mph`` file is a zip archive.  The binary resources (``mesh*.mphbin``, ``geometry*.mphbin``) follow the
documented native binary format and are read by :mod:`axionfem.comsol_mphbin`.  The model tree itself,
``dmodel.xml``, is **not documented** by COMSOL; the layout used here was established on files saved by
COMSOL 5.6 (build 401) and 6.4 (build 293) and may change between versions (see
docs/comsol_import.md).  What this module relies on:

- tagged nodes carry ``tag``/``op``/``name`` attributes (``<GeomFeature tag="cyl1" op="Cylinder" ...>``);
- feature settings: ``<param param="Je" value="3|1,'expr'|1,'expr'|1,'0'">`` (count, then ``k,'string'``
  entries), ``<propertyValue name="p:r" value="1.5">`` (geometry/study properties that differ from the
  default), the built-state string ``<key>`` of geometry and study features, and ``<entityFlags>`` (``DISABLED``);
- explicit selections ``<explicit dim= hDim= geom= entities=>``: for domains the first integer is a header
  (number of domains + 1), for lower dimensions two negative integers ``-N,-1`` precede the entities;
  no ``entities`` attribute means "all applicable entities" (default features);
- ``<MeshNative>`` names the stored mesh resource of a mesh sequence; ``<GeomNative>`` the geometry resource.

Expressions are evaluated with :func:`evaluate`, a small safe evaluator of COMSOL syntax (``^``, unit brackets
``[MHz]``, ``range()``, user functions of type Interpolation/Analytic), never with ``eval``.
"""
from __future__ import annotations

import ast
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------ units
_PREFIX = {"Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "h": 1e2,
           "da": 1e1, "d": 1e-1, "c": 1e-2, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
           "a": 1e-18}
# SI value of each base symbol (dimensions are not tracked: the evaluator returns SI numbers)
_UNIT = {"m": 1.0, "g": 1e-3, "s": 1.0, "A": 1.0, "K": 1.0, "mol": 1.0, "cd": 1.0, "Hz": 1.0, "N": 1.0,
         "Pa": 1.0, "J": 1.0, "W": 1.0, "C": 1.0, "V": 1.0, "F": 1.0, "ohm": 1.0, "Ω": 1.0, "S": 1.0,
         "Wb": 1.0, "T": 1.0, "H": 1.0, "rad": 1.0, "sr": 1.0, "deg": math.pi / 180, "°": math.pi / 180,
         "min": 60.0, "h": 3600.0, "d": 86400.0, "L": 1e-3, "l": 1e-3, "in": 0.0254, "ft": 0.3048,
         "mil": 2.54e-5, "G": 1e-4, "Gs": 1e-4, "atm": 101325.0, "bar": 1e5, "eV": 1.602176634e-19,
         "1": 1.0, "dB": 1.0, "rpm": 2 * math.pi / 60, "mi": 1609.344, "yd": 0.9144, "Oe": 1e3 / (4 * math.pi),
         "gauss": 1e-4, "kg": 1.0}
_NO_PREFIX = {"min", "h", "d", "in", "ft", "mil", "atm", "deg", "°", "1", "dB", "rpm", "mi", "yd", "kg", "G"}


class ExpressionError(ValueError):
    """A COMSOL expression or unit that this evaluator does not handle."""


def unit_factor(unit: str) -> float:
    """SI factor of a COMSOL unit expression such as ``MHz``, ``A/m^2``, ``ohm``, ``kg/(m*s^2)``, ``1/s``."""
    u = unit.strip()
    if not u:
        return 1.0
    tokens = re.findall(r"[A-Za-zµΩ°_]+|\d+(?:\.\d*)?|[*/^()\-]", u)
    pos = 0

    def atom():
        nonlocal pos
        t = tokens[pos]
        if t == "(":
            pos += 1
            v = expr()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ExpressionError(f"unbalanced parentheses in unit [{unit}]")
            pos += 1
            return v
        pos += 1
        if re.fullmatch(r"\d+(?:\.\d*)?", t):
            return float(t)
        return _symbol(t, unit)

    def power():
        nonlocal pos
        v = atom()
        if pos < len(tokens) and tokens[pos] == "^":
            pos += 1
            sign = 1
            if tokens[pos] == "-":
                sign = -1
                pos += 1
            if tokens[pos] == "(":
                pos += 1
                e = float(tokens[pos]) * sign
                pos += 1
                if tokens[pos] == "/":
                    e /= float(tokens[pos + 1])
                    pos += 2
                pos += 1  # ')'
            else:
                e = float(tokens[pos]) * sign
                pos += 1
            v = v ** e
        return v

    def expr():
        nonlocal pos
        v = power()
        while pos < len(tokens) and tokens[pos] in "*/":
            op = tokens[pos]
            pos += 1
            w = power()
            v = v * w if op == "*" else v / w
        return v

    try:
        v = expr()
    except (IndexError, ValueError) as e:
        raise ExpressionError(f"cannot parse unit [{unit}]") from e
    if pos != len(tokens):
        raise ExpressionError(f"cannot parse unit [{unit}]")
    return v


def _symbol(t: str, unit: str) -> float:
    if t in _UNIT:
        return _UNIT[t]
    for p in sorted(_PREFIX, key=len, reverse=True):
        if t.startswith(p) and t[len(p):] in _UNIT and t[len(p):] not in _NO_PREFIX:
            base = t[len(p):]
            return _PREFIX[p] * _UNIT[base]
    raise ExpressionError(f"unknown unit '{t}' in [{unit}]")


# ------------------------------------------------------------------------------------------------ expressions
_CONST = {"pi": math.pi, "i": 1j, "j": 1j, "eps": np.finfo(float).eps, "inf": math.inf, "Inf": math.inf,
          "e_const": 1.602176634e-19, "mu0_const": 4e-7 * math.pi * 1.00000000055,
          "epsilon0_const": 8.8541878128e-12, "c_const": 299792458.0, "h_const": 6.62607015e-34,
          "k_B_const": 1.380649e-23, "R_const": 8.314462618, "N_A_const": 6.02214076e23,
          "Z0_const": 376.730313668, "g_const": 9.80665, "true": 1.0, "false": 0.0}


def _comsol_if(c, a, b):
    return np.where(c, a, b) if isinstance(c, np.ndarray) else (a if c else b)


_FUNC = {"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10, "sin": np.sin, "cos": np.cos,
         "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
         "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "abs": np.abs, "min": np.minimum, "max": np.maximum,
         "real": np.real, "imag": np.imag, "conj": np.conj, "arg": np.angle, "sign": np.sign, "floor": np.floor,
         "ceil": np.ceil, "round": np.round, "if": _comsol_if, "__if__": _comsol_if, "mod": np.mod, "acosh": np.arccosh,
         "asinh": np.arcsinh, "atanh": np.arctanh, "log2": np.log2, "hypot": np.hypot}


def _translate(expr: str) -> str:
    """COMSOL expression -> Python expression: units become calls, ``^`` becomes ``**``."""
    s = expr.strip()
    units: list[str] = []

    def keep(m):
        units.append(m.group(1))
        return f"*__unit__(__U{len(units) - 1}__)"

    s = re.sub(r"\[([^\[\]]*)\]", keep, s)
    s = s.replace("^", "**")
    s = re.sub(r"__U(\d+)__", lambda m: repr(units[int(m.group(1))]), s)
    s = re.sub(r"(?<![=!<>])!(?!=)", " not ", s)
    s = s.replace("&&", " and ").replace("||", " or ")
    s = re.sub(r"\bif\s*\(", "__if__(", s)            # 'if' is a Python keyword
    # a leading unit (e.g. "[V]" alone) leaves "*__unit__": make it 1*
    s = re.sub(r"^\s*\*", "1*", s)
    s = re.sub(r"([(,]\s*)\*__unit__", r"\g<1>1*__unit__", s)
    return s


def evaluate(expr, params: dict | None = None, variables: dict | None = None, functions: dict | None = None):
    """Evaluate a COMSOL expression to an SI number (or a numpy array if variables are arrays).

    ``params`` and ``variables`` map names to values (``x``, ``y``, ``z``, ``freq`` ...); ``functions`` maps user
    function names to callables.  Unknown names raise :class:`ExpressionError` naming them."""
    if isinstance(expr, (int, float, complex, np.ndarray)):
        return expr
    src = _translate(str(expr))
    if not src:
        raise ExpressionError("empty expression")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"cannot parse expression '{expr}'") from e
    names = dict(_CONST)
    names.update(params or {})
    names.update(variables or {})
    funcs = dict(_FUNC)
    funcs.update(functions or {})

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float, complex)):
                return n.value
            raise ExpressionError(f"unexpected constant {n.value!r} in '{expr}'")
        if isinstance(n, ast.Name):
            if n.id in names:
                v = names[n.id]
                if isinstance(v, str):
                    raise ExpressionError(f"'{n.id}' is not numeric in '{expr}'")
                return v
            raise ExpressionError(f"unknown name '{n.id}' in '{expr}'")
        if isinstance(n, ast.Attribute):
            raise ExpressionError(f"model variable '{ast.unparse(n)}' in '{expr}' cannot be evaluated outside COMSOL")
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.UAdd):
                return v
            if isinstance(n.op, ast.Not):
                return np.logical_not(v) * 1.0
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = type(n.op)
            if op is ast.Add:
                return a + b
            if op is ast.Sub:
                return a - b
            if op is ast.Mult:
                return a * b
            if op is ast.Div:
                return a / b
            if op is ast.Pow:
                return a ** b
            if op is ast.Mod:
                return np.mod(a, b)
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            out = vals[0]
            for v in vals[1:]:
                out = np.logical_and(out, v) if isinstance(n.op, ast.And) else np.logical_or(out, v)
            return out * 1.0
        if isinstance(n, ast.Compare):
            a = ev(n.left)
            res = True
            for op, c in zip(n.ops, n.comparators):
                b = ev(c)
                r = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
                     ast.Eq: np.equal, ast.NotEq: np.not_equal}[type(op)](a, b)
                res = np.logical_and(res, r)
                a = b
            return res * 1.0
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id == "__unit__":
                return unit_factor(n.args[0].value)
            if n.func.id in funcs:
                return funcs[n.func.id](*[ev(a) for a in n.args])
            raise ExpressionError(f"unknown function '{n.func.id}' in '{expr}'")
        raise ExpressionError(f"unsupported syntax '{ast.unparse(n)}' in '{expr}'")

    v = ev(tree)
    if isinstance(v, complex) and v.imag == 0:
        v = v.real
    return v


def parse_list(expr: str, params=None, functions=None, unit: str = "", unit_params=frozenset()) -> list[float]:
    """A COMSOL list: space/comma separated values and ``range(start, step, stop)``.  Entries without a unit are in
    ``unit`` (e.g. a study's ``punit``); an entry has a unit when its expression contains ``[...]`` or uses a
    parameter defined with one (``unit_params``)."""
    s = str(expr).strip()
    out: list[float] = []
    uf = unit_factor(unit) if unit else 1.0
    for m in re.finditer(r"range\(([^()]*(?:\([^()]*\)[^()]*)*)\)|[^\s,]+(?:\[[^\]]*\])?", s):
        t = m.group(0)
        if t.startswith("range("):
            args = _split_top(m.group(1))
            if len(args) != 3:
                raise ExpressionError(f"range() needs 3 arguments: {t}")
            a, st, b = (_with_unit(x, params, functions, uf, unit_params) for x in args)
            if st == 0:
                raise ExpressionError(f"zero step in {t}")
            n = int(math.floor((b - a) / st * (1 + 1e-12) + 1e-9)) + 1
            out.extend(a + st * k for k in range(max(n, 0)))
        else:
            out.append(_with_unit(t, params, functions, uf, unit_params))
    return out


def _with_unit(t: str, params, functions, uf: float, unit_params=frozenset()) -> float:
    """A list entry in SI: the list unit applies unless the expression carries a unit itself."""
    v = float(np.real(evaluate(t, params, functions=functions)))
    return v if has_unit(t, unit_params) else v * uf


def has_unit(expr: str, unit_params=frozenset()) -> bool:
    """Whether a COMSOL expression carries a unit: a ``[...]`` or a parameter defined with one."""
    s = str(expr)
    return "[" in s or any(n in unit_params for n in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s))


def _split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out]


# ------------------------------------------------------------------------------------------------ xml helpers
def parse_value(value: str | None) -> list[str]:
    """``"3|1,'a'|1,'b'|1,'c'"`` -> ``['a', 'b', 'c']`` (dmodel.xml ``param`` values; quotes may hold ``|``)."""
    if value is None:
        return []
    items, cur, q = [], "", False
    for ch in value:
        if ch == "'":
            q = not q
        if ch == "|" and not q:
            items.append(cur)
            cur = ""
        else:
            cur += ch
    items.append(cur)
    out = []
    for it in items[1:]:
        m = re.match(r"^\d+,'(.*)'$", it, re.S)
        out.append(m.group(1).replace("\\,", ",") if m else it)
    return out


def parse_matrix(value: str | None) -> list[list[str]]:
    """``valueMatrix="3|2,'2e9','3.04'|2,'5e9','3.02'|..."`` -> rows of strings."""
    if value is None:
        return []
    rows = []
    for it in value.split("|")[1:]:
        rows.append(re.findall(r"'((?:[^'\\]|\\.)*)'", it))
    return rows


def leaf(e, name: str):
    c = e.find(name)
    return None if c is None or c.text is None else c.text.strip()


def is_disabled(e) -> bool:
    f = leaf(e, "entityFlags") or ""
    return "DISABLED" in f.split("|")


def params_of(e) -> dict[str, list[str]]:
    return {p.get("param"): parse_value(p.get("value")) for p in e.findall("param")}


def props_of(e) -> dict:
    out = {}
    for p in e.findall("propertyValue"):
        name = (p.get("name") or "").replace("p:", "")
        if p.get("value") is not None:
            out[name] = p.get("value")
            if p.get("valueMatrix") is not None:
                out[name + "#matrix"] = parse_matrix(p.get("valueMatrix"))
        elif p.get("valueMatrix") is not None:
            out[name] = parse_matrix(p.get("valueMatrix"))
        elif len(p):
            out[name] = p
    return out


def tagged_children(e, kind: str | None = None):
    """Direct tagged children, looking through untagged containers and ``*List`` wrappers."""
    for c in e:
        if c.get("tag") is not None and not c.tag.endswith("List"):
            if kind is None or c.tag == kind:
                yield c
        elif c.tag.endswith("List") or c.get("tag") is None:
            yield from tagged_children(c, kind)


@dataclass
class Selection:
    dim: int | None
    entities: list[int] | None      # None = all applicable entities
    geom: str = ""
    named: str | None = None
    raw: str | None = None
    kind: str = "explicit"           # explicit | all (explicit without entities) | named | unknown

    @property
    def unknown(self) -> bool:
        return self.kind == "unknown"

    def to_json(self):
        if self.unknown:
            return {"dim": self.dim, "entities": "UNKNOWN (selection form not understood)"}
        return {"dim": self.dim, "entities": self.entities if self.entities is not None else "all",
                **({"named": self.named} if self.named else {})}


def parse_explicit(ex, sdim: int, counts: dict | None = None, warnings: list | None = None) -> Selection:
    dim = int(ex.get("dim")) if ex.get("dim") is not None else None
    raw = ex.get("entities")
    if raw is None:
        return Selection(dim, None, ex.get("geom", ""), raw=None, kind="all")
    toks = [int(t) for t in raw.split(",") if t.strip()]
    if any(t < 0 for t in toks):
        k = 0
        while k < len(toks) and toks[k] < 0:
            k += 1
        hdr, ents = toks[:k], toks[k:]
        if counts is not None and dim is not None and hdr and counts.get(dim) is None:
            counts[dim] = -hdr[0]
    elif dim == sdim and toks:
        hdr, ents = toks[:1], toks[1:]
        if counts is not None and counts.get(dim) is None:
            counts[dim] = hdr[0] - 1
    else:
        hdr, ents = [], toks
        if warnings is not None:
            warnings.append(f"explicit selection '{raw}' (dim {dim}) has no recognised header; all values taken "
                            "as entities")
    return Selection(dim, ents, ex.get("geom", ""), raw=raw)


def selection_of(e, sdim: int, counts=None, warnings=None, named_sel: dict | None = None) -> Selection | None:
    """The selection of a feature node: its ``<selection>`` (explicit or named) child."""
    s = e.find("selection")
    if s is None:
        s = e.find("outputSelection")
    if s is None:
        return None
    ex = s.find("explicit")
    if ex is not None:
        return parse_explicit(ex, sdim, counts, warnings)
    for k, v in s.attrib.items():  # named selection reference (layout unverified)
        if named_sel and isinstance(v, str):
            for tag, sel in named_sel.items():
                if v.endswith("/" + tag) or v == tag:
                    return Selection(sel.dim, sel.entities, sel.geom, named=tag,
                                     kind="all" if sel.entities is None else "named")
    # neither an explicit list nor a named selection we parsed: never treat this as "all" (callers block)
    return Selection(None, None, kind="unknown")


# ------------------------------------------------------------------------------------------------ container
class MphArchive:
    """An ``.mph`` file: zip members, ``fileversion`` and the parsed ``dmodel.xml``."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            self.zip = zipfile.ZipFile(self.path)
        except zipfile.BadZipFile as e:
            raise ValueError(f"{self.path}: not an .mph archive (zip)") from e
        self.names = self.zip.namelist()
        if "dmodel.xml" not in self.names:
            raise ValueError(f"{self.path}: no dmodel.xml in the archive (members: {self.names[:8]}...). "
                             "Only full .mph models are supported (not application or mphbin files).")
        self.fileversion = self.read("fileversion").decode("latin1").strip() if "fileversion" in self.names else ""
        self.root = ET.fromstring(self.read("dmodel.xml"))

    def read(self, name: str) -> bytes:
        return self.zip.read(name)

    def has(self, name: str) -> bool:
        return name in self.names


# ------------------------------------------------------------------------------------------------ model
@dataclass
class Feature:
    tag: str
    op: str
    name: str
    path: str
    disabled: bool
    params: dict
    props: dict
    key: str | None
    selection: Selection | None
    node: object = field(repr=False, default=None)
    children: list = field(default_factory=list)
    default: bool = False

    def p(self, name, default=None):
        v = self.params.get(name)
        if v is None:
            return default
        return v[0] if len(v) == 1 else v


@dataclass
class ComsolModel:
    path: Path
    version: str
    name: str
    params: dict                 # name -> {"expr", "value", "descr"}
    functions: dict              # name -> callable
    function_info: dict          # name -> description (for the report)
    geometries: dict             # tag -> {"sdim", "features": [Feature], "counts": {dim: n}, "resource", ...}
    materials: list              # [Feature + groups]
    physics: list                # [{"tag","op","name","geom","selection","features":[Feature]}]
    studies: list                # [{"tag","name","steps":[Feature]}]
    meshes: dict                 # tag -> {"geom", "resource", "features": [Feature]}
    selections: dict             # tag -> Selection
    tables: list
    warnings: list
    archive: MphArchive = field(repr=False, default=None)
    coordsys: list = field(default_factory=list)   # Definitions > coordinate systems [Feature] (PML, ...)
    components: dict = field(default_factory=dict)  # model node tag -> {"spatial": [...], "geom": [...]}

    def axisymmetric(self, geom: str) -> bool | None:
        """True when the component of geometry ``geom`` uses (r, phi, z) spatial coordinates (a COMSOL 2D
        axisymmetric component), False for (x, y, z), None when unknown."""
        for c in self.components.values():
            if geom in c.get("geoms", []) or len(self.components) == 1:
                sp = [x.lower() for x in c.get("spatial", [])]
                if sp:
                    return sp[0] == "r"
        return None

    @property
    def unit_params(self) -> frozenset:
        return frozenset(k for k, v in self.params.items() if v.get("has_unit"))

    def evaluate(self, expr, variables=None):
        return evaluate(expr, self.param_values, variables, self.functions)

    @property
    def param_values(self) -> dict:
        return {k: v["value"] for k, v in self.params.items() if v.get("value") is not None}


def _with_units(f, argunits: list, fununit: str):
    """COMSOL functions take their arguments in ``argunit`` and return values in ``fununit``: convert from/to SI."""
    af = [unit_factor(u) if u else 1.0 for u in argunits]
    ff = unit_factor(fununit) if fununit else 1.0
    if all(a == 1.0 for a in af) and ff == 1.0:
        return f

    def g(*vals):
        conv = [v / (af[k] if k < len(af) else 1.0) for k, v in enumerate(vals)]
        return f(*conv) * ff
    g.__name__ = getattr(f, "__name__", "f")
    return g


def _interp_function(rows: list[list[str]], extrap: str, name: str, extrap_value: float = 0.0):
    pts = np.array([[float(a) for a in r[:2]] for r in rows if len(r) >= 2])
    order = np.argsort(pts[:, 0])
    x, y = pts[order, 0], pts[order, 1]

    def f(t):
        t = np.asarray(t, dtype=float)
        v = np.interp(t, x, y)
        if extrap == "linear" and len(x) > 1:
            lo, hi = t < x[0], t > x[-1]
            v = np.where(lo, y[0] + (t - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), v)
            v = np.where(hi, y[-1] + (t - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]), v)
        elif extrap in ("none", "nan"):
            v = np.where((t < x[0]) | (t > x[-1]), np.nan, v)
        elif extrap == "value":
            v = np.where((t < x[0]) | (t > x[-1]), extrap_value, v)
        return v if v.ndim else float(v)

    f.__name__ = name
    return f, {"type": "Interpolation", "points": len(x), "x": [float(x[0]), float(x[-1])], "extrapolation": extrap,
               "table": [[float(a), float(b)] for a, b in zip(x, y)]}


def _collect_functions(root, params, out: dict, info: dict, warnings: list, where: str):
    for ff in root.iter("FunctionFeature"):
        pr = props_of(ff)
        name = pr.get("funcname")
        op = ff.get("op")
        if not name or name in out:
            continue
        argu = [u.strip() for u in pr.get("argunit", "").split(",")] if isinstance(pr.get("argunit"), str) else []
        funu = pr.get("fununit") if isinstance(pr.get("fununit"), str) else ""
        if op == "Interpolation":
            table = pr.get("table")
            method = pr.get("interp", "linear")
            if not isinstance(method, str):
                method = "linear"
            if method not in ("linear",):
                info[name] = {"type": op, "where": where, "status": f"interpolation method {method}: unsupported"}
                continue
            if isinstance(table, list) and table:
                ext = pr.get("extrap", "const")
                try:
                    extv = float(pr.get("extrapvalue", "0")) if isinstance(pr.get("extrapvalue", "0"), str) else 0.0
                except ValueError:
                    extv = float("nan")
                f, i = _interp_function(table, ext if isinstance(ext, str) else "const", name, extv)
                out[name] = _with_units(f, argu, funu)
                info[name] = {**i, "where": where, "argunit": argu, "fununit": funu}
            else:
                info[name] = {"type": op, "where": where, "status": "table stored in a file or result table: unsupported"}
        elif op == "Analytic":
            expr = pr.get("expr")
            raw = pr.get("args")
            if isinstance(raw, list):
                args = [a.strip() for row in raw for a in row if a.strip()]
            elif isinstance(raw, str) and raw.strip():
                args = [a.strip() for a in raw.split(",")]
            else:
                args = ["x"]
            if isinstance(expr, str):
                def f(*vals, _e=expr, _a=args):
                    return evaluate(_e, params, dict(zip(_a, vals)), out)
                out[name] = _with_units(f, argu, funu)
                info[name] = {"type": op, "expr": expr, "args": args, "where": where, "argunit": argu, "fununit": funu}
        else:
            info[name] = {"type": op, "where": where, "status": "not evaluated"}


def load_model(path) -> ComsolModel:
    """Parse an ``.mph`` archive into a :class:`ComsolModel` (no geometry or mesh decoding yet)."""
    arc = MphArchive(path)
    root = arc.root
    warnings: list[str] = []
    name = root.get("name") or arc.path.name
    # ---- parameters (global; the layout of non-empty groups is unverified on real files)
    params: dict = {}
    # COMSOL 6.3 (a 2D-axisymmetric emw file): <ModelParam><expressions name="ri" expr="0.052"/> directly under
    # ModelParam
    for mp in root.iter("ModelParam"):
        for ex in mp.findall("expressions"):
            if ex.get("name"):
                params[ex.get("name")] = {"expr": ex.get("expr", ""), "descr": ex.get("descr", "")}
    for grp in root.iter("ModelParamGroup"):
        for p in grp.findall("param"):
            vals = parse_value(p.get("value"))
            params[p.get("param")] = {"expr": vals[0] if vals else p.get("value"), "descr": p.get("descr", "")}
        for p in grp.iter("propertyValue"):
            if p.get("name") == "p:param" and p.get("valueMatrix"):
                for row in parse_matrix(p.get("valueMatrix")):
                    if len(row) >= 2:
                        params[row[0]] = {"expr": row[1], "descr": row[2] if len(row) > 2 else ""}
    # evaluate in dependency order (repeat until no progress)
    pending = dict(params)
    values: dict = {}
    for _ in range(len(pending) + 1):
        progress = False
        for k, v in list(pending.items()):
            try:
                values[k] = evaluate(v["expr"], values)
                params[k]["value"] = values[k]
                del pending[k]
                progress = True
            except ExpressionError:
                pass
        if not progress:
            break
    for k, v in pending.items():
        params[k]["value"] = None
        warnings.append(f"parameter {k} = {v['expr']} could not be evaluated")
    unit_params: set = set()
    for _ in range(len(params) + 1):          # a parameter has a unit if its expression or a parameter in it has
        new = {k for k, v in params.items() if has_unit(str(v["expr"]), unit_params)} - unit_params
        if not new:
            break
        unit_params |= new
    for k in params:
        params[k]["has_unit"] = k in unit_params
    functions: dict = {}
    finfo: dict = {}
    fl = root.find("FunctionFeatureList")
    if fl is not None:
        _collect_functions(fl, values, functions, finfo, warnings, "global")
    # ---- geometry sequences
    geometries: dict = {}
    for gs in root.iter("GeomSequence"):
        if gs.get("tag") in (None, "sequence2D") or (leaf(gs, "sequenceType") or "") == "WORK_PLANE":
            continue
        tag = gs.get("tag")
        sdim = int(leaf(gs, "sdim") or 3)
        gnat = gs.find(".//GeomNative") if gs.find(".//GeomNative") is not None else None
        feats = _geom_features(gs, f"/geom/{tag}", sdim)
        geometries[tag] = {"tag": tag, "sdim": sdim, "features": feats, "counts": {}, "lengthUnit":
                           leaf(gs, "lengthUnit") or "m", "angularUnit": leaf(gs, "angularUnit") or "deg",
                           "geomrep": leaf(gs, "geomrep"), "node": gs}
        res = None
        for b in gs.iter():
            if b.get("class") == "GEOMETRY" and b.get("resource"):
                res = b.get("resource")
        geometries[tag]["resource"] = res
        if (geometries[tag]["lengthUnit"] or "m") != "m":
            warnings.append(f"geometry {tag} uses length unit {geometries[tag]['lengthUnit']}; values are scaled to m")
    sdim_of = {t: g["sdim"] for t, g in geometries.items()}

    def sdim_for(e):
        ex = e.find(".//explicit")
        g = (ex.get("geom") if ex is not None else "") or ""
        return sdim_of.get(g.split("/")[-1], 3)

    def counts_for(e):
        ex = e.find(".//explicit")
        g = (ex.get("geom") if ex is not None else "") or ""
        t = g.split("/")[-1]
        return geometries[t]["counts"] if t in geometries else None

    # ---- named selections
    selections: dict = {}
    for sf in root.iter("SelectionFeature"):
        sel = selection_of(sf, sdim_for(sf), counts_for(sf), warnings)
        if sel is not None:
            selections[sf.get("tag")] = sel
    # ---- materials
    materials = []
    for mat in root.iter("Material"):
        if mat.get("tag") is None:
            continue
        sel = selection_of(mat, sdim_for(mat), counts_for(mat), warnings, selections)
        groups = {}
        for mm in mat.iter("MaterialModel"):
            groups[mm.get("tag")] = {k: v for k, v in params_of(mm).items()}
        local_f, local_i = {}, {}
        _collect_functions(mat, values, local_f, local_i, warnings, f"material {mat.get('tag')}")
        materials.append({"tag": mat.get("tag"), "name": mat.get("name"), "op": mat.get("op"),
                          "disabled": is_disabled(mat), "selection": sel, "groups": groups,
                          "functions": local_f, "function_info": local_i, "node": mat})
    # ---- physics
    physics = []
    for ph in root.iter("Physics"):
        geom = (leaf(ph, "geom") or "").split("/")[-1]
        sd = sdim_of.get(geom, 3)
        cnt = geometries[geom]["counts"] if geom in geometries else None
        sel = selection_of(ph, sd, cnt, warnings)
        feats = []
        _physics_features(ph, f"/physics/{ph.get('tag')}", sd, cnt, warnings, selections, feats)
        props = {}
        for pp in ph.iter("PhysicsProp"):
            props.update({k: (v[0] if len(v) == 1 else v) for k, v in params_of(pp).items()})
        physics.append({"tag": ph.get("tag"), "op": ph.get("op"), "name": ph.get("name"), "geom": geom,
                        "selection": sel, "features": feats, "props": props, "disabled": is_disabled(ph)})
    # ---- studies
    studies = []
    for st in root.iter("Study"):
        steps = []
        for sf in tagged_children(st, "StudyFeature"):
            steps.append(Feature(sf.get("tag"), sf.get("op") or "", sf.get("name") or "",
                                 f"/study/{st.get('tag')}/{sf.get('tag')}", is_disabled(sf), params_of(sf),
                                 props_of(sf), leaf(sf, "key"), None, sf))
        studies.append({"tag": st.get("tag"), "name": st.get("name"), "steps": steps, "disabled": is_disabled(st)})
    # ---- meshes
    meshes = {}
    for ms in root.iter("MeshSequence"):
        res = None
        mn = ms.find(".//MeshNative")
        if mn is not None:
            for b in mn.iter():
                if b.get("class") == "MESH" and b.get("resource"):
                    res = b.get("resource")
        geom = None
        for b in ms.iter("geom"):
            if b.text:
                geom = b.text.strip().split("/")[-1]
                break
        feats = []
        for mf in ms.iter("MeshFeature"):
            sd = sdim_of.get(geom, 3)
            feats.append(Feature(mf.get("tag"), mf.get("op") or "", mf.get("name") or "", "", is_disabled(mf),
                                 params_of(mf), props_of(mf), leaf(mf, "key"),
                                 selection_of(mf, sd, geometries.get(geom, {}).get("counts"), warnings), mf))
        ne = leaf(mn, "numEntities") if mn is not None else None
        meshes[ms.get("tag")] = {"tag": ms.get("tag"), "geom": geom, "resource": res, "features": feats,
                                 "numEntities": [int(x) for x in ne.split(",")] if ne else None}
    # ---- tables (for the report only)
    tables = [{"tag": t.get("tag"), "name": t.get("name")} for t in root.iter("TableFeature")]
    # ---- coordinate systems (Definitions): PML / infinite elements are coordinate systems in COMSOL
    #      (layout as the default Boundary System of the 2211.00008 files: <Coordsys op="Boundary" tag="sys1">;
    #      PML nodes are assumed to follow it with op="PML", unverified on a real file)
    coordsys = []
    for cs in root.iter("Coordsys"):
        if cs.get("tag") is None:
            continue
        coordsys.append(Feature(cs.get("tag"), cs.get("op") or "", cs.get("name") or "", f"/coordsys/{cs.get('tag')}",
                                is_disabled(cs), params_of(cs), props_of(cs), leaf(cs, "key"),
                                selection_of(cs, sdim_for(cs), counts_for(cs), warnings, selections), cs))
    # ---- components: spatial coordinates (axisymmetric components use r, phi, z) and their geometries
    components = {}
    for mn in root.iter("ModelNode"):
        if mn.get("tag") is None:
            continue
        sp = re.findall(r"'([^']*)'", leaf(mn, "spatialCoord") or "")   # "3,'r','phi','z'" 
        geoms = [gs.get("tag") for gs in mn.iter("GeomSequence") if gs.get("tag")]
        components[mn.get("tag")] = {"spatial": sp, "geoms": geoms}
    for tag, g in geometries.items():  # geometries listed outside their model node: attach by storedParent
        par = (leaf(g["node"], "storedParent") or "").split("/")[-1]
        if par in components and tag not in components[par]["geoms"]:
            components[par]["geoms"].append(tag)
    return ComsolModel(arc.path, arc.fileversion, name, params, functions, finfo, geometries, materials, physics,
                       studies, meshes, selections, tables, warnings, arc, coordsys, components)


def _geom_features(seq, path: str, sdim: int) -> list[Feature]:
    feats = []
    fl = None
    for c in seq:
        if c.tag == "GeomFeatureList":
            fl = c
            break
    if fl is None:
        return feats
    for gf in fl:
        if gf.get("tag") is None:
            continue
        f = Feature(gf.get("tag"), gf.get("op") or "", gf.get("name") or "", f"{path}/{gf.get('tag')}",
                    is_disabled(gf), params_of(gf), props_of(gf), leaf(gf, "key"), None, gf)
        f.props["_status"] = leaf(gf, "status")
        if f.op == "WorkPlane":
            for sub in gf.iter("GeomSequence"):
                if sub.get("tag") == "sequence2D":
                    f.children = _geom_features(sub, f"{path}/{gf.get('tag')}", 2)
                    break
        feats.append(f)
    return feats


def _physics_features(e, path, sdim, counts, warnings, named, out):
    for c in e:
        if c.tag == "PhysicsFeature":
            f = Feature(c.get("tag"), c.get("op") or "", c.get("name") or "", f"{path}/{c.get('tag')}",
                        is_disabled(c), params_of(c), props_of(c), None,
                        selection_of(c, sdim, counts, warnings, named), c)
            f.default = (leaf(c, "default") or "") == "true"
            sub = []
            _physics_features(c, f.path, sdim, counts, warnings, named, sub)
            f.children = sub
            out.append(f)
        elif c.tag == "PhysicsFeatureList":
            _physics_features(c, path, sdim, counts, warnings, named, out)
