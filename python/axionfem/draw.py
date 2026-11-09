"""Effective-circuit drawings of axionFEM readout objects (optional dependency: ``schemdraw``).

The drawings are generated from the objects themselves, so they show the topology exactly as it
was built: every branch, ideal-transformer winding and multiport port appears once, under its own
name, and the nodes are connected as in the netlist.  Entry points (all return a
``schemdraw.Drawing``; save it with ``d.save("x.svg")``, or let Jupyter display it):

* :meth:`axionfem.circuit.Circuit.draw`, :meth:`axionfem.circuit.Readout.draw`;
* :meth:`axionfem.multiport.MultiportCircuit.draw`, :meth:`axionfem.multiport.MultiportReadout.draw`;
* :meth:`axionfem.squid.Chain.draw`.

How a network is laid out
-------------------------
Nodes of degree 2 are absorbed into *series chains*; the remaining nodes are *hubs*.  Chains that
start and end on the same hub are drawn as one closed loop each; chains between the same two hubs
are drawn as one ladder (first chain on the left, the others as shunt columns, the last one along
the top and down the right).  The ground node is drawn with a ground symbol wherever it occurs,
and any other hub shared between several drawn blocks is labelled with its node name, so the
blocks connect exactly as the netlist does.  Mutual inductances and ideal transformers are dashed
coupling lines; the SQUID is drawn coupled to its input-coil branch.  A multiport is a box with
one row per port (its external load drawn as a loop from the port's two terminals); with many
identical rows ``compact=True`` folds consecutive identical rows into one.

Every drawn circuit element carries ``_axf_name`` and the drawing keeps the terminal coordinates
of each element (``d.axf_terminals``), its wires and its labelled hubs, so :func:`drawn_netlist` can
recover the netlist from the *geometry* and :func:`check_drawing` compares it with the object
(used by the tests: a drawing that does not match its circuit is a test failure).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

EL = 3.0            # length of one element symbol [schemdraw units]
FS = 11             # font size [pt]
INK = "#1a1a1a"
PICK = "#b0306a"    # pickup / axion-source elements
SQC = "#1f5fa8"     # SQUID and amplifier chain
PORTC = "#2d6a3e"   # multiport blocks
NOTE = "#555555"
COMPACT_PORTS = 12  # MultiportCircuit.draw folds identical rows by default above this many ports


# ===================================================================== optional dependency
def _schemdraw():
    try:
        import schemdraw
        import schemdraw.elements as elm
    except ImportError as exc:      # pragma: no cover - exercised only without schemdraw
        raise ImportError(
            "axionFEM circuit drawings need the optional dependency 'schemdraw' (>= 0.19): "
            "pip install schemdraw matplotlib (the package's 'draw' extra, axionfem[draw], installs the same).  "
            "Everything else in axionfem works without it.") from exc
    try:
        import matplotlib  # noqa: F401
        schemdraw.use("matplotlib")
    except ImportError:             # pragma: no cover
        schemdraw.use("svg")
    return schemdraw, elm


def _drawing():
    schemdraw, elm = _schemdraw()
    d = schemdraw.Drawing(show=False, fontsize=FS, lw=1.4, color=INK, bgcolor="white", font="DejaVu Sans",
                                margin=0.8)
    d.axf_terminals = {}     # element name -> (point at p, point at m)
    d.axf_kinds = {}         # element name -> circuit kind
    d.axf_wires = []         # (p0, p1) galvanic wire segments
    d.axf_hubs = []          # (point, node name): ground symbols and labelled shared nodes
    d.axf_couplings = []     # (name_a, name_b, kind)
    d.axf_dots = {}          # coupled element -> its polarity dot (at the p end)
    d.axf_plus = {}          # voltage source -> its "+" mark (at the m end)
    d.axf_compacted = {}     # representative name -> list of folded names (compact multiport rows)
    d.axf_fold_groups = []   # each folded row group: [names of row 0 (drawn), names of row 1, ...]
    d.axf_squid = None       # the SQUID's input-coil branch, when a SQUID is drawn
    d.axf_annotate = None    # whether element values were drawn
    d.axf_omega = None       # the frequency [rad/s] the values were evaluated at
    return d, elm


# ===================================================================== value formatting
_PREFIXES = ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"),
             (1e-9, "n"), (1e-12, "p"), (1e-15, "f"), (1e-18, "a"))


def fmt_si(x: float | None, unit: str, digits: int = 3) -> str:
    """``1.2e-6, "H"`` -> ``"1.2 µH"``: ``digits`` significant figures and an SI prefix."""
    if x is None:
        return "?"
    x = float(x)
    if not math.isfinite(x):
        return f"{x} {unit}".strip()
    if x == 0:
        return f"0 {unit}".strip()
    x = float(f"{x:.{digits}g}")          # round first, so 999.96 n -> 1 µ, not 1e+03 n
    a = abs(x)
    scale, pre = _PREFIXES[-1]
    for s, p in _PREFIXES:
        if a >= s * (1 - 1e-12):
            scale, pre = s, p
            break
    v = x / scale
    return f"{v:.{digits}g} {pre}{unit}".strip()


def fmt_num(x: float, digits: int = 3) -> str:
    """A dimensionless number; mathtext powers of ten outside [1e-3, 1e4)."""
    x = float(x)
    if x == 0 or not math.isfinite(x) or 1e-3 <= abs(x) < 1e4:
        return f"{float(f'{x:.{digits}g}'):g}"
    e = int(math.floor(math.log10(abs(x))))
    m = float(f"{x / 10 ** e:.{digits}g}")
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    return rf"$10^{{{e}}}$" if m == 1 else rf"${m:g}\times10^{{{e}}}$"


def tex_name(name: str) -> str:
    """``"L_in"`` -> ``$L_{\\rm in}$``; single letters in math italics; other names verbatim."""
    m = re.fullmatch(r"([A-Za-z]{1,3})_([A-Za-z0-9]+)", name)
    if m:
        return rf"${m[1]}_{{\rm {m[2]}}}$"
    if re.fullmatch(r"[A-Za-z]", name):
        return f"${name}$"
    return name.replace("$", "")


def _text_width(s: str, fs: float = FS) -> float:
    """Rough width [units] of a (possibly multi-line, possibly mathtext) label."""
    def one(line):
        t = re.sub(r"\\(rm|mathrm|,|;|quad)\b", "", line)
        t = re.sub(r"\\[A-Za-z]+", "x", t)
        t = re.sub(r"[{}$_^]", "", t)
        return len(t)
    n = max((one(x) for x in s.split("\n")), default=0)
    return n * fs / 62.0


def _lines(s: str) -> int:
    return s.count("\n") + 1 if s else 0


def _join(*parts: str) -> str:
    return ", ".join(p for p in parts if p)


# ===================================================================== netlist model
@dataclass
class Item:
    """One circuit element as drawn: a branch, an ideal-transformer winding or a multiport port."""
    name: str            # unique drawn name (branch name; "<T>.1"/"<T>.2" for transformer windings)
    kind: str            # R L C Z V I short T1 T2 port
    p: str
    m: str
    ref: Any = None      # the Branch / _Transformer
    block: str | None = None     # the multiport block of a port
    label: str = ""
    symbols: list = field(default_factory=list)   # [(factory name, label, color)]
    names: list = field(default_factory=list)     # folded names (compact rows)

    @property
    def owner(self) -> str:
        """The object name the element belongs to (a transformer for both of its windings)."""
        return self.name.rsplit(".", 1)[0] if self.kind in ("T1", "T2") else self.name


def circuit_items(circuit) -> list[Item]:
    """Every two-terminal element of a :class:`~axionfem.circuit.Circuit` (multiport ports and both
    windings of every ideal transformer included), in construction order."""
    ports = {}
    for blk in getattr(circuit, "_multiports", []):
        for nm in blk.ports:
            ports[nm] = blk.name
    out = []
    for b in circuit._branches:
        out.append(Item(b.name, "port" if b.name in ports else b.kind, b.p, b.m, b, ports.get(b.name)))
    for t in circuit._transformers:
        out.append(Item(f"{t.name}.1", "T1", t.p1, t.m1, t))
        out.append(Item(f"{t.name}.2", "T2", t.p2, t.m2, t))
    return out


@dataclass
class Block:
    """A drawn block: a closed loop at one hub (``a == b``) or a ladder of chains between two hubs.
    Each chain is a list of ``(item, forward)`` from ``a`` to ``b``; ``forward`` means p -> m."""
    a: str
    b: str
    chains: list

    @property
    def loop(self) -> bool:
        return self.a == self.b

    def items(self) -> list[Item]:
        return [it for ch in self.chains for it, _ in ch]


def _ends(it: Item, fwd: bool) -> tuple[str, str]:
    return (it.p, it.m) if fwd else (it.m, it.p)


def decompose(items: Sequence[Item], ground: str = "gnd") -> list[Block]:
    """Series chains between hubs (nodes of degree != 2), grouped into loops and ladders."""
    adj: dict[str, list[int]] = {}
    for k, it in enumerate(items):
        adj.setdefault(it.p, []).append(k)
        adj.setdefault(it.m, []).append(k)
    hubs = {n for n, es in adj.items() if len(es) != 2}
    # a component without any hub is a single cycle: anchor it at ground, else at its first node
    seen: set[str] = set()
    for n0 in list(adj):
        if n0 in seen:
            continue
        comp, stack = [], [n0]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.append(n)
            for k in adj[n]:
                stack.append(items[k].m if items[k].p == n else items[k].p)
        if not hubs.intersection(comp):
            hubs.add(ground if ground in comp else comp[0])
    used = [False] * len(items)
    chains = []
    order = [n for n in adj if n in hubs]
    order.sort(key=lambda n: (n != ground,))
    for h in order:
        for k0 in adj[h]:
            if used[k0]:
                continue
            ch, cur, k = [], h, k0
            while True:
                used[k] = True
                it = items[k]
                fwd = it.p == cur
                nxt = it.m if fwd else it.p
                ch.append((it, fwd))
                if nxt in hubs:
                    break
                others = [j for j in adj[nxt] if j != k]
                if len(others) != 1 or used[others[0]]:
                    break
                cur, k = nxt, others[0]
            chains.append((h, nxt, ch))
    blocks: list[Block] = []
    ladders: dict[frozenset, Block] = {}
    for a, b, ch in chains:
        if a == b:
            blocks.append(Block(a, a, [ch]))
            continue
        if a == ground:                                # put ground at the bottom rail (b)
            a, b, ch = b, a, [(it, not f) for it, f in reversed(ch)]
        key = frozenset((a, b))
        if key in ladders:
            blk = ladders[key]
            if blk.a != a:
                ch = [(it, not f) for it, f in reversed(ch)]
            blk.chains.append(ch)
        else:
            ladders[key] = Block(a, b, [ch])
            blocks.append(ladders[key])
    return blocks


def _couplings(circuit) -> list[tuple[str, str, str, float]]:
    """``(item a, item b, kind, value)``: mutual inductances and ideal transformers."""
    out = [(a, b, "M", M) for a, b, M in circuit._mutual]
    out += [(f"{t.name}.1", f"{t.name}.2", "T", t.ratio) for t in circuit._transformers]
    return out


# ===================================================================== element symbols and labels
def _branch_T(b) -> float | None:
    return getattr(b, "T_K", None)


def _mqs_form(Zf, w: float):
    """``(R(w), L, R_constant)`` if ``Z(omega) = R(omega) - i omega L`` with constant ``L > 0`` and
    ``R >= 0`` (checked at w and 2.7 w); else None."""
    try:
        z1, z2 = complex(Zf(w)), complex(Zf(2.7 * w))
    except Exception:
        return None
    L1, L2 = -z1.imag / w, -z2.imag / (2.7 * w)
    if L1 > 0 and abs(L1 - L2) <= 1e-9 * abs(L1) and z1.real >= 0 and z2.real >= 0:
        return z1.real, L1, abs(z1.real - z2.real) <= 1e-9 * max(abs(z1.real), 1e-300)
    return None


def noise_label(b, annotate: bool) -> str:
    """The Johnson-noise annotation of a dissipative branch, as :meth:`Circuit.noisy_branches`
    treats it: its temperature; ``0 K`` (vacuum noise only) when ``T_K`` is None; ``noiseless``
    when the branch was built with ``noise=False``."""
    if not getattr(b, "noise", True):
        return "noiseless"
    T = _branch_T(b)
    if T is None:
        return "0 K" if annotate else "$T=0$"
    return fmt_si(T, "K") if annotate else "$T$"


def _symbols(it: Item, annotate: bool, w_ref: float | None, ctx: dict) -> list:
    """``[(symbol, label, color)]`` for one item; one element can need several symbols (a source
    branch with its series impedance)."""
    b = it.ref
    nm = tex_name(it.name)
    Tlab = noise_label(b, annotate) if b is not None and it.kind not in ("T1", "T2", "port") else ""
    if it.kind == "R":
        val = fmt_si(b.value, "Ω") if (annotate and b.value is not None) else ("R(ω)" if b.value is None else "")
        return [("Resistor", "\n".join(x for x in (nm, _join(val, Tlab if b.value != 0 else "")) if x), INK)]
    if it.kind == "L":
        val = fmt_si(b.value, "H") if (annotate and b.value is not None) else ("L(ω)" if b.value is None else "")
        return [("Inductor2", "\n".join(x for x in (nm, val) if x), INK)]
    if it.kind == "C":
        td = getattr(b, "tan_delta", 0.0)
        val = fmt_si(b.value, "F") if (annotate and b.value is not None) else ("C(ω)" if b.value is None else "")
        tdl = (r"$\tan\delta$=" + fmt_num(td) if annotate else r"$\tan\delta$") if td else ""
        return [("Capacitor", "\n".join(x for x in (nm, _join(val, tdl, Tlab if td else "")) if x), INK)]
    if it.kind == "I":
        return [("SourceI", nm, INK)]
    if it.kind == "short":
        return [("Short", nm + " (short)", INK)]
    if it.kind in ("T1", "T2"):
        t = b
        lab = tex_name(t.name) + (" primary" if it.kind == "T1" else " secondary")
        if annotate and it.kind == "T1":
            lab += f"\n1 : {t.ratio:g}"
        return [("Winding", lab, INK)]
    if it.kind == "port":
        return [("Port", f"{it.name}\n({it.block} port)", PORTC)]
    if it.kind == "Z":
        z = ""
        if annotate and w_ref:
            mq = _mqs_form(b.Z, w_ref)
            if mq is not None and mq[2]:
                z = _join(fmt_si(mq[0], "Ω") if mq[0] else "", fmt_si(mq[1], "H"))
            else:                              # not a constant R - i omega L: its value at f
                z = "Z = " + _cfmt(complex(b.Z(w_ref))) + f" Ω at {fmt_si(w_ref / (2 * math.pi), 'Hz')}"
        z = z or "Z(ω)"
        try:                                   # dissipative at the probe frequency: label its noise
            lossy = complex(b.Z(w_ref or ctx.get("w_probe", 2 * math.pi * 1e6))).real > 0
        except Exception:
            lossy = True
        return [("RBox", "\n".join(x for x in (nm, _join(z, Tlab if lossy else "")) if x), INK)]
    if it.kind == "V":
        is_pickup = it.name == ctx.get("pickup")
        col = PICK if is_pickup else INK
        if is_pickup:
            src = r"$V_a=-i\omega\,\kappa_a\Psi$"
            psi = ctx.get("Psi")
            if annotate and psi is not None:
                src += "\n" + r"$\Psi$ = " + fmt_si(psi, "T H m²")
        else:
            src = nm + ("\n" + "emf " + _cfmt(b.emf_at(w_ref)) if annotate and w_ref and not callable(b.emf) else "")
        out = [("SourceSin", src, col)]
        w = w_ref or ctx.get("w_probe", 2 * math.pi * 1e6)
        mq = _mqs_form(b.Z, w)
        try:
            zero = all(complex(b.Z(x)) == 0 for x in (w, 2.7 * w))
        except Exception:
            zero = False
        if zero:
            return out
        if mq is not None:
            Rv, Lv, Rconst = mq
            rname = (r"$R_{\rm loss}$" if is_pickup else nm + " R") + ("" if Rconst else r"$(\omega)$")
            lname = r"$L_p$" if is_pickup else nm + " L"
            if Rv > 0:
                rv = ""
                if annotate:
                    rv = fmt_si(Rv, "Ω") + ("" if Rconst or not w_ref else f" at {fmt_si(w_ref / (2 * math.pi), 'Hz')}")
                out.append(("Resistor", "\n".join(x for x in (rname, _join(rv, Tlab)) if x), col))
            out.append(("Inductor2", "\n".join(x for x in (lname, fmt_si(Lv, "H") if annotate else "") if x), col))
        else:
            zl = r"$Z_p(\omega)$" if is_pickup else nm + r" $Z(\omega)$"
            if annotate and w_ref:
                zl += "\n" + _cfmt(complex(b.Z(w_ref))) + " Ω"
            out.append(("RBox", "\n".join(x for x in (zl, Tlab) if x), col))
        return out
    return [("RBox", nm, INK)]


def _cfmt(z: complex) -> str:
    z = complex(z)
    return f"{z.real:.3g}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.3g}i"


def fingerprint(it: Item, omega: float | None = None):
    """What makes two elements interchangeable in a drawing: kind, constant value, noise
    temperature/flag, ``tan delta``, the impedance at two probe frequencies and the EMF (or the
    multiport block of a port, the ratio of a transformer).  ``None`` when it cannot be evaluated
    (such elements are never folded together)."""
    b = it.ref
    if it.kind == "port":
        return ("port", it.block)
    if it.kind in ("T1", "T2"):
        return (it.kind, float(b.ratio))
    w = omega or 2 * math.pi * 1e6
    try:
        zs = tuple(f"{complex(b.Z(x)):.12g}" if b.Z is not None else "" for x in (w, 2.7 * w))
        emf = f"{b.emf_at(w):.12g}"
    except Exception:
        return None
    return (it.kind, b.value, _branch_T(b), getattr(b, "tan_delta", 0.0), bool(getattr(b, "noise", True)), zs, emf)


def _make(elm, sym: str):
    if sym == "Resistor":
        return elm.Resistor()
    if sym == "Inductor2":
        return elm.Inductor2(loops=3)
    if sym == "Winding":
        return elm.Inductor2(loops=4)
    if sym == "Capacitor":
        return elm.Capacitor()
    if sym == "SourceSin":
        return elm.SourceSin()
    if sym == "SourceI":
        return elm.SourceI()
    if sym in ("RBox", "Port"):
        return elm.RBox()
    if sym == "Short":
        return elm.Line()
    raise ValueError(sym)


# ===================================================================== the renderer
class _Renderer:
    def __init__(self, annotate: bool):
        self.d, self.elm = _drawing()
        self.annotate = annotate
        self.centers: dict[str, tuple] = {}        # item name -> (center point, orientation)
        self.right_x = 0.0
        self.shared: dict[str, int] = {}           # hub -> number of blocks it is drawn in
        self.coil: str | None = None               # SQUID input coil: label kept inside the loop
        self.coupled: set[str] = set()             # mutually coupled items: labels inside too
        self.rows = False                          # multiport row layout
        self.omega: float | None = None            # frequency of the value labels [rad/s]

    # -------------------------------------------------- primitives
    def wire(self, p0, p1):
        p0, p1 = tuple(map(float, p0)), tuple(map(float, p1))
        if math.dist(p0, p1) < 1e-9:
            return
        e = self.d.add(self.elm.Line().endpoints(p0, p1))
        self.d.axf_wires.append(((float(e.start[0]), float(e.start[1])), (float(e.end[0]), float(e.end[1]))))

    def hub(self, pt, node, ground, below=True):
        pt = tuple(map(float, pt))
        if node == ground:
            if below:
                self.d.add(self.elm.Ground().at(pt))
            else:
                self.d.add(self.elm.Dot().at(pt).label(node, loc="top", fontsize=FS - 2, color=NOTE))
        else:
            self.d.add(self.elm.Dot().at(pt).label(node, loc="top" if not below else "bottom",
                                                   fontsize=FS - 2, color=NOTE))
        self.d.axf_hubs.append((pt, node))

    def item(self, it: Item, fwd: bool, p0, direction: str, length: float, side: str, inside: bool = False):
        """Draw ``it`` from ``p0`` in ``direction`` ('up','down','right'), its traversal sense
        given by ``fwd`` (p at the start when True).  Returns the end point."""
        elm = self.elm
        dx, dy = {"up": (0, 1), "down": (0, -1), "right": (1, 0)}[direction]
        n = max(len(it.symbols), 1)
        seg = length / n
        cur = tuple(map(float, p0))
        start = cur
        drawn = []
        if (it.name == self.coil or (it.name in self.coupled and not self.rows)) and direction != "right" and inside:
            side = "left" if side == "right" else "right"   # leave the outside free for the coupling
        for k, (sym, lab, col) in enumerate(it.symbols):
            nxt = (cur[0] + dx * seg, cur[1] + dy * seg)
            # Circuit.current_source drives its value INTO node p: the arrow must point at p
            rev = sym == "SourceI" and fwd
            e = _make(elm, sym).endpoints(*((nxt, cur) if rev else (cur, nxt))).color(col)
            e._axf_rev = rev
            if sym == "Short":
                e = e.linestyle("-").linewidth(2.4)
            if direction == "right":
                loc = "top"
            else:
                loc = "top" if side == "left" else "bottom"
            if lab:
                e = e.label(lab, loc=loc, fontsize=FS - 1, color=col if col != INK else INK,
                            halign=("right" if (direction != "right" and side == "left") else
                                    "left" if direction != "right" else "center"))
            e._axf_name = it.name
            e._axf_names = it.names or [it.name]
            self.d.add(e)
            drawn.append(e)
            cur = nxt
        # the terminals are read back from the placed schemdraw elements (not from the intended
        # coordinates), so check_drawing tests what is actually drawn
        ends = [((e.end, e.start) if e._axf_rev else (e.start, e.end)) for e in drawn]
        start = (float(ends[0][0][0]), float(ends[0][0][1]))
        end = (float(ends[-1][1][0]), float(ends[-1][1][1]))
        for (_, a), (b, _) in zip(ends[:-1], ends[1:]):
            if math.dist(tuple(a), tuple(b)) > 1e-6:
                raise RuntimeError(f"draw: the symbols of {it.name!r} do not join")
        self.d.axf_terminals[it.name] = (start, end) if fwd else (end, start)
        self.d.axf_kinds[it.name] = it.kind
        self.polarity(it, fwd, drawn, direction, side)
        for nm in it.names[1:]:
            self.d.axf_compacted.setdefault(it.name, []).append(nm)
        self.centers[it.name] = (((start[0] + end[0]) / 2, (start[1] + end[1]) / 2), direction, side)
        return end

    def polarity(self, it: Item, fwd: bool, drawn: list, direction: str, side: str):
        """Orientation marks: a dot at the ``p`` end of every mutually coupled element, of the
        SQUID input coil and of each transformer winding (dot convention: positive ``M`` adds flux
        for currents entering the dots, ``Phi_SQ = M I_in`` with ``I_in`` entering the dot;
        ``V_2 = n V_1`` with both voltages taken dot-to-undotted), and ``+`` at the terminal
        of a voltage source that its EMF raises (``m``: ``V(m) - V(p) = emf`` at zero current)."""
        elm = self.elm
        if direction == "right":
            perp = (0.0, -1.0)                 # labels above: marks below
        else:
            perp = (1.0, 0.0) if side == "left" else (-1.0, 0.0)
        if it.name in self.coupled or it.name == self.coil:
            a, b = self.d.axf_terminals[it.name]
            L = math.dist(a, b) or 1.0
            u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
            pt = (a[0] + 0.55 * u[0] + 0.32 * perp[0], a[1] + 0.55 * u[1] + 0.32 * perp[1])
            self.d.add(elm.Dot(radius=0.09).at(pt).color(INK))
            self.d.axf_dots[it.name] = pt
        src = [e for e in drawn if isinstance(e, elm.SourceSin)] if it.kind == "V" else []
        if src:
            e = src[0]
            s0, s1 = tuple(map(float, e.start)), tuple(map(float, e.end))
            m_end = s1 if fwd else s0          # the symbol's end that faces the m terminal
            c = ((s0[0] + s1[0]) / 2, (s0[1] + s1[1]) / 2)
            u = (m_end[0] - c[0], m_end[1] - c[1])
            L = math.hypot(*u) or 1.0
            pt = (c[0] + u[0] / L * 0.95 + 0.3 * perp[0], c[1] + u[1] / L * 0.95 + 0.3 * perp[1])
            col = it.symbols[0][2] if it.symbols else INK
            self.d.add(elm.Label().at(pt).label("+", fontsize=FS + 1, color=col))
            self.d.axf_plus[it.name] = pt

    # -------------------------------------------------- geometry of one block
    def col_len(self, items) -> float:
        return sum(max(len(it.symbols), 1) * EL for it in items)

    def top_len(self, it) -> float:
        w = max((_text_width(l) for _, l, _ in it.symbols), default=0.0)
        return max(len(it.symbols), 1) * max(EL, w + 0.8)

    def label_w(self, items) -> float:
        return max((_text_width(l) for it in items for _, l, _ in it.symbols), default=0.0)

    def label_h(self, items) -> float:
        return max((_lines(l) for it in items for _, l, _ in it.symbols), default=0) * 0.42

    def layout(self, blk: Block, prefer_first, prefer_last):
        """Arrange ``blk`` as (left column, shunt columns, top row, right column), each a list of
        ``(item, fwd)`` oriented along the drawing direction; plus where its hubs sit."""
        if blk.loop:
            seq = list(blk.chains[0])
            n = len(seq)
            i0 = max(range(n), key=lambda k: (prefer_first(seq[k][0]), -k))
            seq = seq[i0:] + seq[:i0]
            if n > 2:
                pos = max(range(n), key=lambda k: (prefer_last(seq[k][0]), k))
                if pos < n - 1:
                    rev = [(seq[0][0], not seq[0][1])] + [(it, not f) for it, f in reversed(seq[1:])]
                    pos_r = max(range(n), key=lambda k: (prefer_last(rev[k][0]), k))
                    if pos_r > pos:
                        seq = rev
            left, top, right = [seq[0]], seq[1:-1], [seq[-1]]
            # the node before seq[k] is junction k; junction 0 = bottom wire
            hub_j = next(k for k in range(n) if _ends(*seq[k])[0] == blk.a)
            return dict(left=left, mids=[], top=top, right=right, hub_junction=hub_j, seq=seq)
        chains = list(blk.chains)
        i0 = max(range(len(chains)), key=lambda k: (max(prefer_first(it) for it, _ in chains[k]), -k))
        first = chains.pop(i0)
        if not chains:
            return dict(left=[(it, not f) for it, f in reversed(first)], mids=[], top=[], right=[], single=True)
        il = max(range(len(chains)), key=lambda k: (max(prefer_last(it) for it, _ in chains[k]), k))
        last = chains.pop(il)
        return dict(left=[(it, not f) for it, f in reversed(first)],     # drawn upward from b to a
                    mids=chains, top=last[:-1], right=[last[-1]])

    def draw_block(self, blk: Block, lay: dict, x0: float, y0: float, H: float, ground: str,
                   box_anchor: bool = False) -> float:
        """Draw a laid-out block with its bottom-left corner at ``(x0, y0)``; returns its right x."""
        left, mids, top, right = lay["left"], lay["mids"], lay["top"], lay["right"]
        ytop = y0 + H
        # ---- left column, upward from (x0, y0)
        if box_anchor:
            it, fwd = left[0]
            self.d.axf_terminals[it.name] = ((x0, y0), (x0, ytop)) if fwd else ((x0, ytop), (x0, y0))
            self.d.axf_kinds[it.name] = it.kind
            for nm in it.names[1:]:
                self.d.axf_compacted.setdefault(it.name, []).append(nm)
            self.centers[it.name] = ((x0, y0 + H / 2), "box", "left")
        else:
            cur = (x0, y0)
            for it, fwd in left:
                cur = self.item(it, fwd, cur, "up", max(len(it.symbols), 1) * EL, "left", inside=True)
            self.wire(cur, (x0, ytop))
        if lay.get("single"):
            self.hub((x0, ytop), blk.a, ground, below=False)
            self.hub((x0, y0), blk.b, ground)
            return x0
        x = x0
        # ---- shunt columns, downward from the top rail (labels on their right)
        prev_w = None
        for ch in mids:
            xs = x + (2.2 if prev_w is None else prev_w + 1.0)
            self.wire((x, ytop), (xs, ytop))
            h = self.col_len([it for it, _ in ch])
            self.wire((xs, ytop), (xs, y0 + h))
            cur = (xs, y0 + h)
            for it, fwd in ch:
                cur = self.item(it, fwd, cur, "down", max(len(it.symbols), 1) * EL, "right")
            prev_w = self.label_w([it for it, _ in ch])
            x = xs
        if prev_w is not None:
            self.wire((x, ytop), (x + prev_w + 0.6, ytop))
            x += prev_w + 0.6
        # ---- top row, rightward
        if top:
            gap = 1.0
            self.wire((x, ytop), (x + gap, ytop))
            cur = (x + gap, ytop)
            for it, fwd in top:
                cur = self.item(it, fwd, cur, "right", self.top_len(it), "top")
            x = cur[0] + 0.8
            self.wire(cur, (x, ytop))
        else:
            inner = [it for it, _ in right if it.name == self.coil or (it.name in self.coupled and not self.rows)]
            xr = x + (max(3.0, self.label_w(inner) + 1.4) if x == x0 else 0.0)
            self.wire((x, ytop), (xr, ytop))
            x = xr
        # ---- right column, downward
        h = self.col_len([it for it, _ in right])
        self.wire((x, ytop), (x, y0 + h))
        cur = (x, y0 + h)
        for it, fwd in right:
            cur = self.item(it, fwd, cur, "down", max(len(it.symbols), 1) * EL, "right", inside=True)
        # ---- bottom wire (node b, or loop junction 0)
        self.wire((x, y0), (x0, y0))
        # ---- hubs
        if blk.loop:
            j = lay["hub_junction"]
            pts = self.junction_points(lay, x0, y0, ytop, x)
            pt, below = pts[j]
            self.hub(pt, blk.a, ground, below=below)
        else:
            self.hub(((x0 + x) / 2, y0), blk.b, ground)
            if self.shared.get(blk.a, 0) > 1 or blk.a == ground:
                self.hub(((x0 + x) / 2, ytop), blk.a, ground, below=False)
        return x

    def junction_points(self, lay, x0, y0, ytop, xr):
        """Point (and whether a ground symbol fits below it) of each loop junction k."""
        seq = lay["seq"]
        pts = {0: (((x0 + xr) / 2, y0), True)}
        tops = [it.name for it, _ in lay["top"]]
        for k in range(1, len(seq)):
            nm_prev = seq[k - 1][0].name
            if k == 1:
                pts[k] = ((x0, ytop), False)
            else:
                (cx, cy), *_ = self.centers[nm_prev]
                if nm_prev in tops:
                    t = self.d.axf_terminals[nm_prev]
                    pts[k] = (max(t, key=lambda p: p[0]), False)
                else:
                    pts[k] = ((xr, ytop), False)
        return pts


# ===================================================================== public drawing functions
def _order_blocks(blocks, items_by_name, couplings, start_pred):
    """Blocks in a sensible left-to-right order: start at the source, follow the couplings."""
    where = {}
    for i, b in enumerate(blocks):
        for it in b.items():
            where[it.name] = i
    adj = {i: [] for i in range(len(blocks))}
    for a, b, *_ in couplings:
        if a in where and b in where and where[a] != where[b]:
            adj[where[a]].append(where[b])
            adj[where[b]].append(where[a])
    start = next((i for i, b in enumerate(blocks) if any(start_pred(it) for it in b.items())), 0)
    order, queue = [], [start]
    while queue or len(order) < len(blocks):
        if not queue:
            queue.append(next(i for i in range(len(blocks)) if i not in order))
        i = queue.pop(0)
        if i in order:
            continue
        order.append(i)
        queue.extend(j for j in adj[i] if j not in order)
    return order, where


def draw_network(circuit, *, annotate: bool = True, squid=None, pickup: str | None = None,
                 Psi=None, omega: float | None = None, sources: Iterable[str] = (),
                 squid_notes: Sequence[str] = (), title: str | None = None,
                 compact: bool | None = None):
    """Draw ``circuit`` (a Circuit or MultiportCircuit), optionally with a SQUID coupled to its
    input coil.  The building block of every ``draw()`` method; see the module docstring."""
    items = circuit_items(circuit)
    if not items:
        raise ValueError("draw: the circuit has no elements")
    ground = circuit.ground
    couplings = _couplings(circuit)
    coupled = {}
    for a, b, *_ in couplings:
        coupled.setdefault(a, []).append(b)
        coupled.setdefault(b, []).append(a)
    coil = squid.input_coil if squid is not None else None
    sources = set(sources)
    ctx = {"pickup": pickup, "Psi": Psi}
    R = _Renderer(annotate)
    R.omega = omega
    R.d.axf_annotate = bool(annotate)
    R.d.axf_omega = omega
    for it in items:
        it.symbols = _symbols(it, annotate, omega, ctx)
        if it.kind == "port" and it.name in sources:
            it.label = "source"
    blocks = decompose(items, ground)
    multiport = bool(getattr(circuit, "_multiports", None))
    by_name = {it.name: it for it in items}

    def prefer_first(it):
        return (3 if (it.kind == "V" or it.kind == "port") else 0) + (2 if it.name == pickup else 0) \
            + (1 if it.name in coupled else 0)

    def prefer_last(it):
        return (3 if it.name == coil else 0) + (1 if it.name in coupled else 0)

    # hubs drawn in several blocks get a node label
    shared: dict[str, int] = {}
    for b in blocks:
        for n in {b.a, b.b}:
            shared[n] = shared.get(n, 0) + 1
    R.shared = shared
    R.coil = coil
    R.coupled = set(coupled)

    if not multiport:
        order, where = _order_blocks(blocks, by_name, couplings,
                                     lambda it: it.name == pickup or it.kind == "V")
        x = 0.0
        prev_right_w = 0.0
        lays = [(blocks[i], R.layout(blocks[i], prefer_first, prefer_last)) for i in order]
        H = max(max(R.col_len([it for it, _ in lay["left"]]),
                    R.col_len([it for it, _ in lay["right"]]),
                    *(R.col_len([it for it, _ in ch]) for ch in lay["mids"]), EL)
                for _, lay in lays)
        for k, (blk, lay) in enumerate(lays):
            lw = R.label_w([it for it, _ in lay["left"] if it.name not in coupled])
            if k:
                x += prev_right_w + lw + 3.6
            xr = R.draw_block(blk, lay, x, 0.0, H, ground)
            prev_right_w = R.label_w([it for it, _ in lay["right"] if it.name != coil and it.name not in coupled])
            x = xr
        R.right_x = x + prev_right_w
        top_y = H + R.label_h([it for b in blocks for it in b.items()]) + 0.6
    else:
        R.rows = True
        R.right_x, top_y = _draw_multiport_rows(R, circuit, blocks, items, ground, prefer_first,
                                                prefer_last, sources, compact, couplings, coil)
    _draw_couplings(R, couplings, circuit)
    if squid is not None:
        _draw_squid(R, squid, coil, squid_notes)
        R.d.axf_squid = coil
    if title:
        R.d.add(R.elm.Label().at((0.0, top_y + 0.4)).label(title, fontsize=FS + 1, halign="left", valign="bottom"))
    return R.d


def _draw_multiport_rows(R, circuit, blocks, items, ground, prefer_first, prefer_last, sources,
                         compact, couplings, coil):
    mps = circuit._multiports
    port_block = {nm: blk.name for blk in mps for nm in blk.ports}
    n_ports = sum(len(b.ports) for b in mps)
    if compact is None:
        compact = n_ports > COMPACT_PORTS
    coupled = {a for a, b, *_ in couplings} | {b for a, b, *_ in couplings}

    def anchor_port(blk, lay):
        if len(lay["left"]) == 1 and lay["left"][0][0].kind == "port":
            return lay["left"][0][0]
        return None

    rows = []
    for blk in blocks:
        lay = R.layout(blk, prefer_first, prefer_last)
        rows.append((blk, lay, anchor_port(blk, lay)))
    rank = {}
    k = 0
    for mp in mps:
        for nm in mp.ports:
            rank[nm] = k
            k += 1
    rows = sorted(enumerate(rows), key=lambda kr: (rank[kr[1][2].name] if kr[1][2] is not None else 10 ** 6, kr[0]))
    rows = [r for _, r in rows]

    def walk(lay, with_dir=False):
        """The row's items in drawing order (anchor first)."""
        return [(it, f) if with_dir else it
                for part in (lay["left"], *lay["mids"], lay["top"], lay["right"]) for it, f in part]

    # ---- fold consecutive identical rows: same box, same hubs, same shape and orientation, and
    # interchangeable elements (fingerprint: kind, value, noise, Z at two frequencies, EMF) --
    # not merely identical labels, which a Z(omega) box or an unannotated drawing would give to
    # different elements; no couplings and no SQUID coil in the row
    def signature(row):
        blk, lay, ap = row
        if ap is None or lay.get("single"):
            return None
        its = walk(lay, True)[1:]
        if ap.name in coupled or any(it.name in coupled or it.name == coil or it.kind == "port" for it, _ in its):
            return None
        fps = tuple((fingerprint(it, R.omega), f) for it, f in its)
        if any(fp is None for fp, _ in fps):
            return None
        return (port_block[ap.name], ap.name in sources, blk.loop, blk.a, blk.b, walk(lay, True)[0][1],
                tuple(len(ch) for ch in lay["mids"]), len(lay["top"]), fps)
    folded = []
    for row in rows:
        sig = signature(row) if compact else None
        if folded and sig is not None and folded[-1][3] == sig:
            folded[-1][4].append(row)
        else:
            folded.append([row[0], row[1], row[2], sig, [row]])
    # rename the representatives of folded groups: element k of the row stands for element k of
    # every folded row
    for blk, lay, ap, sig, group in folded:
        if len(group) < 2:
            continue
        rows_items = [walk(g[1]) for g in group]
        R.d.axf_fold_groups.append([[it.name for it in w] for w in rows_items])
        for pos, it in enumerate(rows_items[0]):
            names = [w[pos].name for w in rows_items]
            it.names = names
            if it.kind == "port":
                continue
            span = f"{names[0]} … {names[-1]} (×{len(names)})" if len(names) > 2 else ", ".join(names)
            it.symbols = [(sym, "\n".join([span] + l.split("\n")[1:]), c) for sym, l, c in it.symbols]

    def port_label(ap):
        names = ap.names or [ap.name]
        lab = (f"{names[0]} … {names[-1]}\n({len(names)} ports)" if len(names) > 2 else ", ".join(names))
        if any(n in sources for n in names):
            lab += "\n" + r"$V_{a}=-i\omega\kappa_a\Psi$"
        return lab

    # ---- geometry
    port_w = max([_text_width(port_label(ap), FS - 1) for _, _, ap, _, _ in folded if ap is not None] + [0.0])
    titles = []
    for mp in mps:
        t = f"{mp.name}\nZ(ω): {len(mp.ports)}×{len(mp.ports)}\nreciprocal"
        if R.annotate:
            if mp.baths:
                t += "\nbaths: " + ", ".join(f"{b.name} {fmt_si(b.T_K, 'K')}" for b in mp.baths)
            else:
                t += f"\nT = {fmt_si(mp.T_K, 'K')}"
        titles.append(t)
    box_w = max(5.0, port_w + max(_text_width(t) for t in titles) + 1.2)
    y = 0.0
    xmax = 0.0
    box_rows: dict[str, list] = {}
    for blk, lay, ap, sig, group in folded:
        parts = [lay["left"], *lay["mids"], lay["right"]]
        H = max(max(R.col_len([it for it, _ in p]) for p in parts if p), EL) if ap is None else \
            max(max([R.col_len([it for it, _ in p]) for p in [*lay["mids"], lay["right"]] if p] + [EL * 0.8]), EL * 0.8)
        top_lab = R.label_h([it for it, _ in lay["top"]] + [it for ch in lay["mids"] for it, _ in ch])
        y -= H + max(top_lab, 0.4) + 1.6
        xr = R.draw_block(blk, lay, 0.0, y, H, ground, box_anchor=ap is not None)
        rw = R.label_w([it for it, _ in lay["right"]])
        xmax = max(xmax, xr + (0.6 if lay["right"] and lay["right"][0][0].name == coil else rw))
        if ap is not None:
            box_rows.setdefault(port_block[ap.name], []).append((y, y + H, ap))
    # ---- the boxes
    elm = R.elm
    for mp, title in zip(mps, titles):
        rs = box_rows.get(mp.name)
        if not rs:
            continue
        ylo = min(r[0] for r in rs) - 0.5
        yhi = max(r[1] for r in rs) + 0.5
        for a, b in (((-box_w, ylo), (0.0, ylo)), ((0.0, ylo), (0.0, yhi)), ((0.0, yhi), (-box_w, yhi)),
                     ((-box_w, yhi), (-box_w, ylo))):
            R.d.add(elm.Line().at(a).to(b).color(PORTC).linewidth(1.8))
        R.d.add(elm.Label().at((-box_w + 0.3, (ylo + yhi) / 2)).label(title, color=PORTC, fontsize=FS, halign="left"))
        for y0, y1, ap in rs:
            lab = port_label(ap)
            R.d.add(elm.Label().at((-0.25, (y0 + y1) / 2)).label(lab, color=PORTC, fontsize=FS - 1, halign="right"))
            for yy in (y0, y1):
                R.d.add(elm.Dot(radius=0.08).at((0.0, yy)).color(PORTC))
    return xmax, 1.0


def _draw_couplings(R, couplings, circuit):
    elm = R.elm
    for a, b, kind, val in couplings:
        if a not in R.centers or b not in R.centers:
            continue
        (pa, *_), (pb, *_) = R.centers[a], R.centers[b]
        if kind == "M":
            ba, bb = circuit._get(a), circuit._get(b)
            lab = "M" if not R.annotate else "M = " + fmt_si(val, "H")
            if R.annotate and ba.value and bb.value:
                lab += f"\nk = {val / math.sqrt(ba.value * bb.value):.3g}"
        else:
            lab = "ideal transformer" + (f"\n1 : {val:g}" if R.annotate else "")
        R.d.add(elm.Line().at(pa).to(pb).linestyle("--").color(NOTE).linewidth(1.0))
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 0.2)
        R.d.add(elm.Label().at(mid).label(lab, fontsize=FS - 2, color=NOTE, valign="bottom"))
        R.d.axf_couplings.append((a, b, kind))


def squid_symbol(d, elm, cx, cy, w=1.4, h=1.6, color=SQC):
    """dc SQUID: a loop with two Josephson junctions.  Returns its top and bottom midpoints."""
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    d.add(elm.Line().at((x0, y1)).to((x1, y1)).color(color))
    d.add(elm.Line().at((x0, y0)).to((x1, y0)).color(color))
    d.add(elm.Josephson().at((x0, y1)).to((x0, y0)).color(color))
    d.add(elm.Josephson().at((x1, y1)).to((x1, y0)).color(color))
    return (cx, y1), (cx, y0)


def amp(d, elm, xy, size=1.4, color=SQC):
    """Amplifier triangle with its input at ``xy``; returns the output point."""
    x, y = xy
    h = 0.5 * size
    d.add(elm.Line().at((x, y - h)).to((x, y + h)).color(color))
    d.add(elm.Line().at((x, y + h)).to((x + size, y)).color(color))
    d.add(elm.Line().at((x + size, y)).to((x, y - h)).color(color))
    return (x + size, y)


def _draw_squid(R, squid, coil, notes):
    d, elm = R.d, R.elm
    if coil not in R.centers:
        raise ValueError(f"draw: the SQUID's input coil {coil!r} is not an element of this circuit")
    (cx, cy), direction, side = R.centers[coil]
    mlab = "M" if not R.annotate else "M = " + fmt_si(squid.M_H, "H")
    if direction == "right":            # coil on a top row: put the SQUID above it
        sx, sy = cx, cy + 4.2
        top, bot = squid_symbol(d, elm, sx, sy)
        d.add(elm.Arc2(arrow="<->", k=0.0).at((cx, cy + 0.7)).to((sx, bot[1] - 0.15)).linewidth(1.2))
        mpt = (cx + 0.3, cy + 1.6)
    else:
        sx, sy = max(R.right_x, cx) + max(3.4, _text_width(mlab, FS + 1) + 2.4), cy
        top, bot = squid_symbol(d, elm, sx, sy)
        d.add(elm.Arc2(arrow="<->", k=0.0).at((cx + 0.75, cy)).to((sx - 0.95, cy)).linewidth(1.2))
        mpt = ((cx + sx) / 2 - 0.1, cy + 0.15)
    d.add(elm.Label().at(mpt).label(mlab, fontsize=FS + 1, valign="bottom"))
    d.add(elm.Label().at((sx, sy)).label(r"$L_{\rm sq}$", color=SQC, fontsize=FS - 1))
    d.add(elm.Line().at(top).up(0.8).color(SQC))
    d.add(elm.Line().right(0.9).color(SQC))
    out = amp(d, elm, d.here, size=1.2)
    d.add(elm.Arrow().at(out).right(0.8).color(SQC))
    d.add(elm.Label().at((top[0] + 1.5, top[1] + 1.5)).label("later stages\n(squid.Chain)", color=SQC,
                                                             fontsize=FS - 2, halign="center", valign="bottom"))
    d.add(elm.Line().at(bot).down(0.5).color(SQC))
    d.add(elm.Ground().color(SQC))
    lines = ["dc SQUID (input coil " + tex_name(coil) + ")"]
    if R.annotate:
        lines += list(notes)
    else:
        lines += [r"imprecision $S_\Phi$", r"backaction $S_J$ (EMF $+i\omega MJ$)", r"correlation $S_{\Phi J}$"]
    note_lines(d, elm, (sx - 0.9, bot[1] - 1.3), lines, SQC)


def note_lines(d, elm, xy, lines, color, fs=FS - 2, step=0.5, halign="left"):
    """One label per line (multi-line mathtext labels overlap in matplotlib)."""
    x, y = xy
    for k, ln in enumerate(lines):
        d.add(elm.Label().at((x, y - k * step)).label(ln, color=color, fontsize=fs, halign=halign, valign="top"))


def squid_notes(squid, omega=None, R0=None, M_eff=None, T_K=None) -> list[str]:
    """Value annotations of a :class:`~axionfem.circuit.Squid` (Foster PSDs shown one-sided)."""
    from .circuit import PHI0, n_bose, xi_opt
    out = [r"$\eta_A$ = " + f"{squid.eta_A:.3g}"]
    if squid.S_Phi > 0:
        out.append(r"imprecision $\sqrt{2S_\Phi}$ = " + f"{math.sqrt(2 * squid.S_Phi) / PHI0 * 1e6:.3g}" + r" µ$\Phi_0/\sqrt{\rm Hz}$ (1-sided)")
    if squid.S_J > 0:
        out.append(r"backaction $\sqrt{2S_J}$ = " + fmt_si(math.sqrt(2 * squid.S_J), "A") + r"$/\sqrt{\rm Hz}$")
    if squid.S_PhiJ and squid.S_Phi > 0 and squid.S_J > 0:
        rho = squid.S_PhiJ / math.sqrt(squid.S_Phi * squid.S_J)
        out.append(r"correlation $S_{\Phi J}/\sqrt{S_\Phi S_J}$ = " + f"{rho:+.3g}")
    else:
        out.append(r"$S_{\Phi J}$ = 0 (uncorrelated)")
    if omega and R0 and R0 > 0 and squid.S_Phi > 0:
        xi = squid.xi_at(omega, R0, M_eff)
        s = r"$\xi$ = " + fmt_num(xi)
        if T_K is not None:
            s += r",  $\xi_{\rm opt}$ = " + fmt_num(xi_opt(n_bose(omega, T_K), squid.eta_A))
        out.append(s)
    return out


# ===================================================================== the readout and the multiport
def _resonance(r):
    """The readout's resonance (:meth:`Readout.resonance` from its own guess), or None."""
    try:
        return r.resonance()
    except Exception:
        return None


def draw_readout(r, *, annotate: bool = True, f_Hz: float | None = None, title: str | None = None):
    """:meth:`axionfem.circuit.Readout.draw`."""
    w = None
    if f_Hz is not None:
        w = 2 * math.pi * float(f_Hz)
    elif annotate:
        w = _resonance(r)
    notes = []
    if annotate:
        R0 = M_eff = T = None
        if w:
            try:
                R0 = r.Z_eff(w).real
                M_eff = abs(r.M_eff(w))
                T = getattr(r.circuit._get(r.pickup), "T_K", None)
            except Exception:
                pass
        notes = squid_notes(r.squid, w, R0, M_eff, T)
        if w:
            notes.insert(0, f"at f = {fmt_si(w / (2 * math.pi), 'Hz', 4)}" + (" (resonance)" if f_Hz is None else ""))
    psi = None
    if annotate:
        try:
            psi = r.Psi_at(w or 2 * math.pi * 1e6)
        except Exception:
            psi = None
        psi = psi or None
    return draw_network(r.circuit, annotate=annotate, squid=r.squid, pickup=r.pickup, Psi=psi,
                        omega=w, squid_notes=notes, title=title)


def draw_multiport_readout(r, *, annotate: bool = True, omega: float | None = None,
                           compact: bool | None = None, title: str | None = None):
    """:meth:`axionfem.multiport.MultiportReadout.draw`."""
    src = r.emf_per_kappa
    names: list[str] = []
    if callable(src):
        if omega is not None:
            names = [k for k, v in src(omega).items() if v != 0]
    else:
        names = [k for k, v in src.items() if v != 0]
    notes = squid_notes(r.squid) if annotate else []
    if callable(src) and omega is None:
        notes.append("sources: emf_per_kappa(ω) (pass omega= to mark them)")
    return draw_network(r.circuit, annotate=annotate, squid=r.squid, omega=omega, sources=names,
                        squid_notes=notes, title=title, compact=compact)


# ===================================================================== the SQUID amplifier chain
def draw_chain(ch, *, f_Hz: float | None = None, annotate: bool = True, title: str | None = None):
    """:meth:`axionfem.squid.Chain.draw`: first stage, second-stage array and preamplifier."""
    from .circuit import HBAR, PHI0
    from .squid import CryoRFAmp, OpAmpPreamp, SwitchedPreamp
    d, elm = _drawing()
    ycap = -2.7

    def caption(x, lines):
        note_lines(d, elm, (x, ycap), lines, SQC, step=0.46, halign="center")
    fs_, st, ampl = ch.first, ch.second, ch.amplifier
    parts = None
    if annotate and f_Hz is not None:
        p = ch.one_sided(float(f_Hz))
        parts = {"TC": float(np.sqrt(p["S1_Phi_TC_Wb2_Hz"])) / PHI0 * 1e6,
                 **{k: float(np.sqrt(v)) / PHI0 * 1e6 for k, v in p["S1_Phi_ref_parts_Wb2_Hz"].items()}}

    def uphi(x):
        return f"{x:.3g} µ" + r"$\Phi_0/\sqrt{\rm Hz}$"

    # ---- input coil and first stage
    d.add(elm.Line().at((0, 2.0)).down(0.8))
    coil = d.add(elm.Inductor2(loops=4).down(2.4).reverse().label(r"$L_{\rm in}$", loc="top", ofst=0.25))
    d.add(elm.Line().down(0.8))
    d.add(elm.Label().at((0, 2.1)).label("readout loop", color=NOTE, fontsize=FS - 2, valign="bottom"))
    sx = 3.0
    top, bot = squid_symbol(d, elm, sx, 0.0)
    d.add(elm.Label().at((sx, 0.0)).label(r"$L_{\rm sq}$", color=SQC, fontsize=FS - 1))
    d.add(elm.Arc2(arrow="<->", k=0.0).at((0.7, 0.0)).to((sx - 0.95, 0.0)).linewidth(1.2))
    d.add(elm.Label().at(((0.7 + sx - 0.95) / 2, 0.15)).label(r"$M$", fontsize=FS + 1, valign="bottom"))
    t1 = ["1st stage: Tesche–Clarke dc SQUID" + (f" ({fs_.label})" if annotate else "")]
    if annotate:
        t1 += [rf"$L_{{\rm sq}}$ = {fmt_si(fs_.L_sq_H, 'H')},  $I_0$ = {fmt_si(fs_.I0_A, 'A')}",
               rf"$R_j$ = {fmt_si(fs_.R_j_ohm, 'Ω')},  $T_j$ = {fmt_si(fs_.T_j_K, 'K')}",
               rf"$V_\Phi$ = {fmt_si(fs_.V_Phi_Hz, 'V/Wb')},  $R_{{\rm dyn}}$ = {fmt_si(fs_.R_dyn_ohm, 'Ω')}",
               rf"$\eta_{{TC}}$ = {fs_.eta_TC:.3g},  $\epsilon_{{uc}}/\hbar$ = {fs_.eps_uc_J_s / HBAR:.3g}",
               f"correlation sign s = {ch.sign:+g} (open)"]
        if parts:
            t1.append("TC imprecision " + uphi(parts["TC"]))
    else:
        t1 += [r"$I_0,\ R_j,\ T_j$;  $M=\kappa\sqrt{L_{\rm sq}L_{\rm in}}$",
               r"noise $S_\Phi,\ S_J,\ S_{\Phi J}$ (sign $s$)"]
    x_end = sx + 1.0
    if st is None:
        d.add(elm.Line().at(top).up(0.9).color(SQC))
        d.add(elm.Line().right(1.0).color(SQC))
        out = amp(d, elm, d.here, size=1.3)
        d.add(elm.Arrow().at(out).right(0.9).color(SQC).label("output", loc="right"))
        d.add(elm.Line().at(bot).down(0.5).color(SQC))
        d.add(elm.Ground().color(SQC))
        caption(sx, t1 + ["(no later stages: a lower bound)"])
    else:
        x2 = 6.2
        d.add(elm.Line().at(top).to((sx, 2.0)).color(SQC))
        d.add(elm.Line().to((x2, 2.0)).color(SQC))
        d.add(elm.Line().down(0.8).color(SQC))
        d.add(elm.Inductor2(loops=4).down(2.4).reverse().color(SQC)
              .label(r"$L_2$" + (f"\n{fmt_si(st.L2_H, 'H')}" if annotate else ""), loc="top", ofst=0.25))
        d.add(elm.Line().down(0.8).color(SQC))
        d.add(elm.Line().to((sx, -2.0)).color(SQC))
        d.add(elm.Line().to(bot).color(SQC))
        if annotate:
            t1.append(rf"drives $L_2$: $\tau=L_2/R_{{\rm dyn,1}}$ = {fmt_si(ch.tau_s, 's')}")
        else:
            t1.append(r"$R_{\rm dyn,1}$ drives $L_2$:  $\tau=L_2/R_{\rm dyn,1}$")
        caption(2.6, t1)
        ax = x2 + 3.4
        for off in (-0.95, 0.95):
            squid_symbol(d, elm, ax + off, 0.0, w=0.75, h=1.0)
        d.add(elm.Label().at((ax, 0.0)).label(r"$\cdots$", color=SQC, fontsize=FS))
        d.add(elm.Arc2(arrow="<->", k=0.0).at((x2 + 0.7, 0.0)).to((ax - 1.75, 0.0)).linewidth(1.2))
        d.add(elm.Label().at(((x2 + 0.7 + ax - 1.75) / 2, 0.15)).label(r"$M_{IN2}$", fontsize=FS - 1, valign="bottom"))
        for (x0, y0), (x1, y1) in ((((ax - 1.6), -0.9), ((ax + 1.6), -0.9)), (((ax + 1.6), -0.9), ((ax + 1.6), 0.9)),
                                   (((ax + 1.6), 0.9), ((ax - 1.6), 0.9)), (((ax - 1.6), 0.9), ((ax - 1.6), -0.9))):
            d.add(elm.Line().at((x0, y0)).to((x1, y1)).linestyle("--").color(SQC).linewidth(1.0))
        t2 = ["2nd stage: series SQUID array", rf"$N_{{\rm ser}}\times N_{{\rm par}}$ = {st.N_ser}×{st.N_par}" + (
            rf" at $T_2$ = {fmt_si(st.T2_K, 'K')}" if annotate else r" at $T_2$")]
        if annotate:
            t2 += [rf"$M_{{IN2}}$ = {fmt_si(st.M_IN2_H, 'H')},  $R_{{\rm dyn}}$ = {fmt_si(st.R_dyn_ohm, 'Ω')}",
                   rf"$dV/d\Phi$ = {fmt_si(st.dV_dPhi_Hz, 'V/Wb')},  $P$ = {fmt_si(st.P_W, 'W')}",
                   rf"flux gain $|d\Phi_1/dI|/M_{{IN2}}$ = {fs_.dPhi_dI_H / st.M_IN2_H:.3g}"]
            if parts:
                t2.append("referred: " + uphi(parts.get("second_stage_squid", 0.0)))
        caption(ax, t2)
        d.add(elm.Line().at((ax + 1.6, 0.0)).right(0.4).color(SQC))
        a = ampl
        if isinstance(a, SwitchedPreamp) and f_Hz is not None:
            a = a.at(float(f_Hz))
        rl = getattr(a, "R_lead_ohm", None)
        lead = r"$R_{\rm lead}(T_{\rm lead})$"
        if annotate and rl is not None:
            lead = r"$R_{\rm lead}$" + f" = {fmt_si(rl, 'Ω')}\n{fmt_si(a.T_lead_K, 'K')}"
        if isinstance(a, CryoRFAmp):
            d.add(elm.Line().right(2.0).color(SQC).label("SC coax", ofst=0.2))
        else:
            d.add(elm.Resistor().right(2.0).color(SQC).label(lead, ofst=0.2))
        d.add(elm.Line().right(0.3).color(SQC))
        out = amp(d, elm, d.here, size=1.5)
        d.add(elm.Arrow().at(out).right(0.9).color(SQC).label(r"$V_{\rm out}$", loc="right"))
        t3 = []
        if annotate:
            def one(a_):
                if isinstance(a_, OpAmpPreamp):
                    return [a_.label, rf"$V_n$ = {fmt_si(a_.V_n_V_rtHz, 'V')}" + r"$/\sqrt{\rm Hz}$, " +
                            rf"$I_n$ = {fmt_si(a_.I_n_A_rtHz, 'A')}" + r"$/\sqrt{\rm Hz}$"]
                if isinstance(a_, CryoRFAmp):
                    return [a_.label, rf"$T_n$ = {fmt_si(a_.T_n_K, 'K')}, {fmt_si(a_.f_min_Hz, 'Hz')}–{fmt_si(a_.f_max_Hz, 'Hz')}"]
                return [getattr(a_, "label", str(a_))]
            if isinstance(a, SwitchedPreamp):
                t3 = ["preamp (switched by band):"]
                for lo, hi, aa in a.bands:
                    t3 += [f"{fmt_si(lo, 'Hz') if lo else '0'} – {fmt_si(hi, 'Hz')}: " + one(aa)[0]]
                    t3 += ["   " + x for x in one(aa)[1:]]
            else:
                t3 = ["preamp: " + one(a)[0]] + one(a)[1:]
            if parts:
                pre = math.sqrt(sum(v ** 2 for k, v in parts.items() if k not in ("TC", "second_stage_squid")))
                t3.append("referred: " + uphi(pre))
        else:
            t3 = ["preamp: " + getattr(ampl, "label", "amplifier"),
                  r"noise $T_n$ (matched)" if isinstance(ampl, CryoRFAmp) else r"noise $V_n,\ I_n$"]
        caption(ax + 1.6 + 0.4 + 2.0 + 0.3 + 0.75 + 0.6, t3)
        x_end = ax + 6
    if annotate and f_Hz is not None:
        summ = (f"at f = {fmt_si(f_Hz, 'Hz')}:  " + rf"$\eta$ = {float(ch.eta(f_Hz)):.3g}" +
                rf",  $\epsilon_{{uc}}/\hbar$ = {float(ch.eps_uc_over_hbar(f_Hz)):.3g}" +
                rf"  (uncorrelated $\eta$ = {float(ch.eta_uncorrelated(f_Hz)):.3g})")
        d.add(elm.Label().at((0, 3.2)).label(summ, fontsize=FS, halign="left", valign="bottom"))
    if title:
        d.add(elm.Label().at((0, 4.2)).label(title, fontsize=FS + 1, halign="left", valign="bottom"))
    d.axf_chain = {"first": fs_.label, "second": None if st is None else (st.N_ser, st.N_par),
                   "amplifier": None if ampl is None else getattr(ampl, "label", None), "x_end": x_end}
    return d


# ===================================================================== checking a drawing against its object
def drawn_netlist(d, tol: float = 1e-6) -> dict[str, tuple[int, int]]:
    """Recover ``{element name: (node id at p, node id at m)}`` from the drawing's *geometry*:
    wires join the points they touch (ends and T-junctions), and hub symbols with the same node
    name (every ground symbol) join their wires."""
    return _drawn_graph(d, tol)[0]


def _drawn_graph(d, tol: float = 1e-6):
    """:func:`drawn_netlist` and ``point -> drawn node id`` for any terminal, wire end or hub."""
    pts: list[tuple[float, float]] = []
    index: dict[tuple, int] = {}

    def pid(p):
        key = (round(p[0] / tol), round(p[1] / tol))
        if key not in index:
            index[key] = len(pts)
            pts.append((float(p[0]), float(p[1])))
        return index[key]

    for a, b in d.axf_terminals.values():
        pid(a), pid(b)
    for a, b in d.axf_wires:
        pid(a), pid(b)
    for p, _ in d.axf_hubs:
        pid(p)
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    def on(p, a, b):
        (x, y), (x0, y0), (x1, y1) = p, a, b
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if abs(cross) > 1e-6 * max(1.0, math.dist(a, b)):
            return False
        return min(x0, x1) - 1e-6 <= x <= max(x0, x1) + 1e-6 and min(y0, y1) - 1e-6 <= y <= max(y0, y1) + 1e-6

    for a, b in d.axf_wires:
        ia, ib = pid(a), pid(b)
        union(ia, ib)
        for k, p in enumerate(pts):
            if on(p, a, b):
                union(k, ia)
    by_name: dict[str, int] = {}
    for p, nm in d.axf_hubs:
        k = pid(p)
        if nm in by_name:
            union(k, by_name[nm])
        else:
            by_name[nm] = k
    net = {nm: (find(index[(round(a[0] / tol), round(a[1] / tol))]), find(index[(round(b[0] / tol), round(b[1] / tol))]))
           for nm, (a, b) in d.axf_terminals.items()}

    def node_id(p):
        k = index.get((round(p[0] / tol), round(p[1] / tol)))
        return None if k is None else find(k)
    return net, node_id


_DRAWN_CLASSES = ("SourceSin", "SourceI", "RBox", "Resistor", "Capacitor", "Inductor2", "Line")
_KIND_SYMBOLS = {"R": ["Resistor"], "L": ["Inductor2"], "C": ["Capacitor"], "Z": ["RBox"], "I": ["SourceI"],
                 "short": ["Line"], "T1": ["Inductor2"], "T2": ["Inductor2"], "port": []}


def _drawn_symbols(d) -> dict[str, list[str]]:
    """element name -> the schemdraw classes actually placed for it (in drawing order)."""
    import schemdraw.elements as elm
    out: dict[str, list[str]] = {}
    for e in d.elements:
        nm = getattr(e, "_axf_name", None)
        if nm is not None:
            cls = [c for c in _DRAWN_CLASSES if isinstance(e, getattr(elm, c))]
            out.setdefault(nm, []).append(cls[0] if cls else type(e).__name__)
    return out


def _drawn_labels(d) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for e in d.elements:
        nm = getattr(e, "_axf_name", None)
        if nm is not None:
            out.setdefault(nm, []).extend(str(lab.label) for lab in getattr(e, "_userlabels", []))
    return {k: "\n".join(v) for k, v in out.items()}


def check_drawing(d, obj) -> list[str]:
    """Differences between a drawing and its object (empty when the drawing is exact).

    ``obj`` is a Circuit/MultiportCircuit, or a Readout/MultiportReadout (then the SQUID's input
    coil is checked too).  Checked: every element drawn once (or folded into an interchangeable
    representative), with the symbol its kind needs (read from the placed schemdraw elements);
    the connections recovered from the geometry (the drawn nodes map one-to-one onto the
    circuit's nodes, with each element's p and m terminals) and every node label on the node it
    names; for annotated drawings the value, unit and noise label of each constant R, L, C; the
    couplings, exactly; and for each folded row group, that the folded rows have interchangeable
    elements, connected in the same way, on the same shared nodes."""
    circuit = getattr(obj, "circuit", obj)
    errs = []
    items = {it.name: it for it in circuit_items(circuit)}
    folded = {n for v in d.axf_compacted.values() for n in v}
    drawn = set(d.axf_terminals)
    missing = set(items) - drawn - folded
    extra = drawn - set(items)
    if missing:
        errs.append(f"not drawn: {sorted(missing)}")
    if extra:
        errs.append(f"drawn but not in the circuit: {sorted(extra)}")
    # ---- symbols, from the placed elements (not from the renderer's bookkeeping)
    syms = _drawn_symbols(d)
    for nm in sorted(drawn & set(items)):
        kind, got = items[nm].kind, syms.get(nm, [])
        if d.axf_kinds.get(nm) != kind:
            errs.append(f"{nm}: drawn as {d.axf_kinds.get(nm)!r}, circuit kind {kind!r}")
        if kind == "V":
            ok = bool(got) and got[0] == "SourceSin" and all(g in ("Resistor", "Inductor2", "RBox") for g in got[1:])
        elif kind == "port":
            ok = got in ([], ["RBox"])
        else:
            ok = got == _KIND_SYMBOLS.get(kind)
        if not ok:
            errs.append(f"{nm}: drawn with symbols {got}, a {kind!r} element needs {_KIND_SYMBOLS.get(kind, 'SourceSin+')}")
    # ---- values (annotated drawings)
    if d.axf_annotate:
        labs = _drawn_labels(d)
        for nm in sorted(drawn & set(items)):
            it = items[nm]
            b = it.ref
            if it.kind not in ("R", "L", "C") or b.value is None:
                continue
            lab = labs.get(nm, "")
            want = [fmt_si(b.value, {"R": "Ω", "L": "H", "C": "F"}[it.kind])]
            if (it.kind == "R" and b.value) or (it.kind == "C" and getattr(b, "tan_delta", 0.0)):
                want.append(noise_label(b, True))
            for w in want:
                if w not in lab:
                    errs.append(f"{nm}: label {lab!r} does not show {w!r}")
    # ---- connections
    net, gid = _drawn_graph(d)
    fwd: dict[int, str] = {}
    back: dict[str, int] = {}
    for nm, (gp, gm) in net.items():
        if nm not in items or nm in d.axf_compacted:
            continue
        for g, node in ((gp, items[nm].p), (gm, items[nm].m)):
            if g in fwd and fwd[g] != node:
                errs.append(f"{nm}: drawn node joins circuit nodes {fwd[g]!r} and {node!r}")
            if node in back and back[node] != g:
                errs.append(f"{nm}: circuit node {node!r} is split into several drawn nodes")
            fwd.setdefault(g, node)
            back.setdefault(node, g)
    # node labels (and ground symbols) must sit on the node they name
    for pt, name in d.axf_hubs:
        node = fwd.get(gid(pt))
        if node is not None and node != name:
            errs.append(f"node label {name!r} is drawn on circuit node {node!r}")
    # ---- couplings
    want = {(a, b) for a, b, *_ in _couplings(circuit)}
    have = {(a, b) for a, b, _ in d.axf_couplings}
    if want - have:
        errs.append(f"couplings not drawn: {sorted(want - have)}")
    if have - want:
        errs.append(f"couplings drawn but not in the circuit: {sorted(have - want)}")
    # ---- the SQUID
    sq = getattr(obj, "squid", None)
    if sq is not None and d.axf_squid != sq.input_coil:
        errs.append(f"SQUID drawn on {d.axf_squid!r}, its input coil is {sq.input_coil!r}")
    # ---- folded rows
    errs += _check_folds(d, circuit, items)
    return errs


def _check_folds(d, circuit, items) -> list[str]:
    errs = []
    inc: dict[str, set] = {}
    for nm, it in items.items():
        inc.setdefault(it.p, set()).add(nm)
        inc.setdefault(it.m, set()).add(nm)
    for group in getattr(d, "axf_fold_groups", []):
        rep_row = group[0]
        for row in group[1:]:
            if len(row) != len(rep_row) or any(n not in items for n in row):
                errs.append(f"folded row {row} does not match {rep_row}")
                continue
            phi: dict[str, str] = {}
            for rn, qn in zip(rep_row, row):
                r_, q_ = items[rn], items[qn]
                if fingerprint(r_, d.axf_omega) != fingerprint(q_, d.axf_omega) or fingerprint(r_, d.axf_omega) is None:
                    errs.append(f"folded {qn!r} is not interchangeable with {rn!r}")
                for u, v in ((r_.p, q_.p), (r_.m, q_.m)):
                    if phi.setdefault(u, v) != v:
                        errs.append(f"folded row {row}: node {u!r} maps to both {phi[u]!r} and {v!r}")
            if len(set(phi.values())) != len(phi):
                errs.append(f"folded row {row}: two nodes of {rep_row} map to one node")
            rs, qs = set(rep_row), set(row)
            for u, v in phi.items():
                if u == v:
                    continue
                if not (inc[u] <= rs and inc[v] <= qs):
                    errs.append(f"folded row {row}: node {v!r} (drawn as {u!r}) is not private to its row")
    return errs


__all__ = ["Block", "Item", "check_drawing", "circuit_items", "decompose", "draw_chain", "draw_multiport_readout",
           "draw_network", "draw_readout", "drawn_netlist", "fingerprint", "fmt_num", "fmt_si", "noise_label",
           "squid_notes", "tex_name"]
