"""Effective-circuit drawings (axionfem.draw; docs/physics/readout_circuit.md Sec. 7.1).

The drawing tests are adversarial: :func:`axionfem.draw.check_drawing` recovers the netlist from
the drawing's geometry (wires, T-junctions, ground symbols) and compares it element by element and
node by node with the circuit, and it is itself shown to catch corrupted drawings.  Tests that
render need ``schemdraw`` and are skipped without it.
"""
from __future__ import annotations

import builtins
import math
import sys

import numpy as np
import pytest

from axionfem import circuit as ck, squid
from axionfem import draw as dr
from axionfem.multiport import LossBath, MultiportCircuit, MultiportReadout
from axionfem.multiport_components import LosslessTransmissionLine, fixed_squid

F0 = 1e6
KIND_CLASS = {"R": "Resistor", "L": "Inductor2", "C": "Capacitor", "Z": "RBox", "I": "SourceI",
              "short": "Line", "T1": "Inductor2", "T2": "Inductor2"}


# ===================================================================== no schemdraw needed
@pytest.mark.parametrize("x,unit,txt", [(1.2e-6, "H", "1.2 µH"), (2e-10, "H", "200 pH"), (0.03, "K", "30 mK"),
                                        (4.0, "K", "4 K"), (1e-4, "Ω", "100 µΩ"), (9.9996e-7, "H", "1 µH"),
                                        (1.7425e-8, "F", "17.4 nF"), (0.0, "Ω", "0 Ω"), (-3.3e3, "V", "-3.3 kV"),
                                        (1.63e12, "V/Wb", "1.63 TV/Wb"), (None, "H", "?")])
def test_fmt_si(x, unit, txt):
    assert dr.fmt_si(x, unit) == txt


def test_fmt_num_and_names():
    assert dr.fmt_num(0.905) == "0.905" and dr.fmt_num(1e-6) == r"$10^{-6}$"
    assert dr.fmt_num(2.5e-7) == r"$2.5\times10^{-7}$" and dr.fmt_num(1.2e5) == r"$1.2\times10^{5}$"
    assert dr.tex_name("L_in") == r"$L_{\rm in}$" and dr.tex_name("C") == "$C$"
    assert dr.tex_name("component3") == "component3" and dr.tex_name("R_lead") == r"$R_{\rm lead}$"


def _netlist(blocks):
    """Edges (name, frozenset{nodes}) re-assembled from the decomposition's chains."""
    out = set()
    for b in blocks:
        for ch in b.chains:
            node = b.a
            for it, fwd in ch:
                p, m = (it.p, it.m) if fwd else (it.m, it.p)
                assert p == node, f"chain broken at {it.name}"
                out.add((it.name, frozenset((it.p, it.m))))
                node = m
            assert node == b.b
    return out


def _readouts():
    common = dict(L_p_H=1.4e-6, Q=1e6, f_Hz=F0, T_K=0.03)
    return {
        "plain": ck.series_readout(**common, L_in_H=2e-9),
        "no_input_coil": ck.series_readout(**common),
        "full_series": ck.series_readout(**common, L_lead_H=5e-8, R_lead_ohm=1e-4, T_lead_K=4.0,
                                         tan_delta=1e-6, L_in_H=2e-9, R_in_ohm=1e-5, Psi_SI=3e-3),
        "C_par": ck.series_readout(**common, L_in_H=2e-9, C_par_F=1e-12, tan_delta_par=1e-5),
        "ideal_T": ck.series_readout(**common, transformer_ratio=3.0, L_in_H=1e-7, R_in_ohm=1e-5, R_2_ohm=1e-4),
        "real_T": ck.series_readout(L_p_H=1e-6, f_Hz=F0, Q=1e6, T_K=.03, L_1_H=1e-6, L_2_H=1e-6, L_in_H=1e-7,
                                    k_T=0.7, M_H=1e-9, R_2_ohm=1e-4, C_par_F=1e-12),
        "fullwave_Zp": ck.series_readout(Z_p=lambda w: complex(1e-6 * (w / 6e6) ** 2, -w * 1.4e-6 * (1 + 1e-3 * w / 6e6)),
                                         f_Hz=F0, T_K=0.03, L_in_H=2e-9),
    }


@pytest.mark.parametrize("name", list(_readouts()))
def test_decomposition_is_the_netlist(name):
    c = _readouts()[name].circuit
    items = dr.circuit_items(c)
    blocks = dr.decompose(items, c.ground)
    assert _netlist(blocks) == {(it.name, frozenset((it.p, it.m))) for it in items}
    assert sorted(it.name for b in blocks for it in b.items()) == sorted(it.name for it in items)


def test_decomposition_shapes():
    r = _readouts()
    assert [(b.loop, len(b.chains)) for b in dr.decompose(dr.circuit_items(r["full_series"].circuit))] == [(True, 1)]
    blocks = dr.decompose(dr.circuit_items(r["C_par"].circuit))
    assert [(b.loop, len(b.chains), b.b) for b in blocks] == [(False, 3, "gnd")]     # pickup || C_par || chain
    assert [b.loop for b in dr.decompose(dr.circuit_items(r["ideal_T"].circuit))] == [True, True]


def _wheatstone():
    c = ck.Circuit()
    c.source("V", "gnd", "a", emf=1.0, Z=1.0)
    c.resistor("R1", "a", "b", 1.0, T_K=1.0)
    c.resistor("R2", "a", "c", 2.0)
    c.resistor("R3", "b", "gnd", 3.0)
    c.capacitor("C4", "c", "gnd", 1e-9)
    c.inductor("L5", "b", "c", 1e-6)
    c.current_source("I6", "b", "gnd", 0.1)
    return c


# ===================================================================== drawings
@pytest.fixture
def sd():
    return pytest.importorskip("schemdraw")


def _drawn_classes(d):
    """name -> drawn schemdraw elements, as the KIND_CLASS name they are an instance of."""
    import schemdraw.elements as elm
    out = {}
    for e in d.elements:
        nm = getattr(e, "_axf_name", None)
        if nm is not None:
            cls = [c for c in ("SourceSin", "SourceI", "RBox", "Resistor", "Capacitor", "Inductor2", "Line")
                   if isinstance(e, getattr(elm, c))]
            out.setdefault(nm, []).append(cls[0] if cls else type(e).__name__)
    return out


def _labels(d, name=None):
    return [lab.label for e in d.elements if name is None or getattr(e, "_axf_name", None) == name
            for lab in e._userlabels]


@pytest.mark.parametrize("annotate", [True, False])
@pytest.mark.parametrize("name", list(_readouts()))
def test_readout_drawing_matches_the_circuit(sd, name, annotate):
    r = _readouts()[name]
    d = r.draw(annotate=annotate)
    assert dr.check_drawing(d, r.circuit) == []
    cls = _drawn_classes(d)
    items = {it.name: it for it in dr.circuit_items(r.circuit)}
    assert set(cls) == set(items)
    for nm, it in items.items():
        if it.kind == "V":
            assert cls[nm][0] == "SourceSin"
        else:
            assert cls[nm] == [KIND_CLASS[it.kind]], nm
    # the SQUID is coupled to the input coil
    assert any("M" in str(l) for l in _labels(d))


def test_readout_values_are_annotated(sd):
    r = _readouts()["full_series"]
    d = r.draw()
    c = r.circuit
    C = c._get("C").value
    assert dr.fmt_si(C, "F") in _labels(d, "C")[0] and r"$\tan\delta$=$10^{-6}$" in _labels(d, "C")[0]
    assert _labels(d, "R_lead")[0] == "$R_{\\rm lead}$\n100 µΩ, 4 K"
    assert _labels(d, "L_in")[0] == "$L_{\\rm in}$\n2 nH"
    pick = _labels(d, "pickup")
    assert pick[0].startswith(r"$V_a=-i\omega\,\kappa_a\Psi$") and "3 mT H m²" in pick[0]
    Rloss = 2 * math.pi * F0 * 1.4e-6 / 1e6
    assert pick[1] == "$R_{\\rm loss}$\n" + dr.fmt_si(Rloss, "Ω") + ", 30 mK" and pick[2] == "$L_p$\n1.4 µH"
    allt = " ".join(_labels(d))
    assert "M = " + dr.fmt_si(r.squid.M_H, "H") in allt
    w0 = r.resonance()
    xi = r.squid.xi_at(w0, r.Z_eff(w0).real, abs(r.M_eff(w0)))
    assert "$\\xi$ = " + dr.fmt_num(xi) in allt and r"$\xi_{\rm opt}$" in allt
    sym = " ".join(_labels(r.draw(annotate=False)))
    assert "nF" not in sym and "µΩ" not in sym and "(EMF $+i\\omega MJ$)" in sym


def test_values_at_the_resonance_with_an_input_coil(sd):
    """L_p >> L_in (Readout._resonance_guess must include L_p): the drawing finds the resonance and
    reports xi there."""
    r = ck.series_readout(L_p_H=7.43e-7, Psi_SI=1.168e-6, f_Hz=F0, Q=1e6, T_K=0.03, L_in_H=2e-9, M_H=1e-9)
    allt = " ".join(_labels(r.draw()))
    assert "at f = 1 MHz (resonance)" in allt and r"$\xi$ = " in allt


def test_frequency_dependent_pickup_is_not_drawn_as_constants(sd):
    d = _readouts()["fullwave_Zp"].draw()
    lab = _labels(d, "pickup")
    assert any(r"$Z_p(\omega)$" in l for l in lab)              # L varies: a Z_p(omega) box
    r = ck.series_readout(Z_p=lambda w: complex(1e-6 * (w / 6e6) ** 2, -w * 1.4e-6), f_Hz=F0, T_K=0.03)
    lab = _labels(r.draw(f_Hz=F0), "pickup")
    assert lab[1].startswith(r"$R_{\rm loss}$$(\omega)$") and "at 1 MHz" in lab[1]   # R(omega), value at f


def test_generic_circuit_and_correlated_squid(sd):
    c = _wheatstone()
    for ann in (True, False):
        d = c.draw(annotate=ann, omega=1e6)
        assert dr.check_drawing(d, c) == []
    # the current source's arrow (its symbol's end) points at node p, where it injects its value
    (e,) = [e for e in d.elements if getattr(e, "_axf_name", None) == "I6"]
    assert math.dist(tuple(e.end), d.axf_terminals["I6"][0]) < 1e-9
    ch = squid.chain("C1+low+RT", sign=-1)
    r = _readouts()["plain"]
    r.squid = ch.squid(F0, r.squid.M_H)
    allt = " ".join(_labels(r.draw()))
    rho = r.squid.S_PhiJ / math.sqrt(r.squid.S_Phi * r.squid.S_J)
    assert f"{rho:+.3g}" in allt and rho < 0


def test_check_drawing_catches_wrong_drawings(sd):
    r = _readouts()["full_series"]
    d = r.draw()
    t = d.axf_terminals
    t["R_lead"] = (t["R_lead"][1], t["R_lead"][0])                 # reversed element
    assert any("R_lead" in e for e in dr.check_drawing(d, r.circuit))
    d = r.draw()
    d.axf_wires.pop(0)                                             # a missing wire
    assert dr.check_drawing(d, r.circuit)
    d = r.draw()
    del d.axf_terminals["C"]
    assert any("not drawn" in e for e in dr.check_drawing(d, r.circuit))
    other = _readouts()["ideal_T"]
    d2 = other.draw()
    d2.axf_hubs[0] = (d2.axf_hubs[0][0], "elsewhere")              # the two loops' grounds split
    assert any("split" in e for e in dr.check_drawing(d2, other.circuit))
    d3 = other.draw()
    d3.axf_couplings.clear()
    assert any("couplings" in e for e in dr.check_drawing(d3, other.circuit))


def test_check_drawing_catches_wrong_values_symbols_and_labels(sd):
    import copy
    import schemdraw.elements as elm
    r = _readouts()["full_series"]
    d = r.draw()
    assert dr.check_drawing(d, r) == []                            # with the SQUID
    c = copy.deepcopy(r.circuit)
    c._get("R_lead").value = 5.0                                   # a value the drawing does not show
    assert any("R_lead" in e and "5 Ω" in e for e in dr.check_drawing(d, c))
    c = copy.deepcopy(r.circuit)
    c._get("R_lead").T_K = 0.1                                     # a noise temperature it does not show
    assert any("100 mK" in e for e in dr.check_drawing(d, c))
    d = r.draw()
    k = next(i for i, e in enumerate(d.elements) if getattr(e, "_axf_name", None) == "C")
    fake = elm.Resistor().endpoints(d.elements[k].start, d.elements[k].end)
    fake._axf_name = "C"
    d.elements[k] = fake                                           # a capacitor drawn as a resistor
    assert any(e.startswith("C: drawn with symbols ['Resistor']") for e in dr.check_drawing(d, r.circuit))
    d = r.draw()
    d.axf_couplings.append(("L_lead", "L_in", "M"))
    assert any("not in the circuit" in e for e in dr.check_drawing(d, r.circuit))
    d = r.draw()
    d.axf_squid = "L_lead"
    assert any("SQUID drawn on" in e for e in dr.check_drawing(d, r))
    c = _wheatstone()
    d = c.draw(omega=1e6)
    d.axf_hubs = [(p, {"a": "b", "b": "a"}.get(n, n)) for p, n in d.axf_hubs]     # node names swapped
    assert any("node label 'a' is drawn on circuit node 'b'" in e for e in dr.check_drawing(d, c))


def test_noise_labels_follow_noisy_branches(sd):
    """A dissipative branch with T_K = None still has vacuum noise (Circuit.noisy_branches takes
    T = 0): drawn as 0 K, not as a silent resistor; noise=False is drawn as noiseless."""
    c = _wheatstone()
    c.impedance("Zq", "c", "gnd", 2.0 + 0j, noise=False)
    d = c.draw(omega=1e6)
    assert dr.check_drawing(d, c) == []
    assert _labels(d, "R3")[0].endswith("3 Ω, 0 K") and _labels(d, "R1")[0].endswith("1 Ω, 1 K")
    assert "noiseless" in _labels(d, "Zq")[0]
    assert "$T=0$" in _labels(c.draw(annotate=False), "R3")[0]


def test_polarity_marks(sd):
    """Dots at the p end of coupled windings and of the SQUID input coil; + at the m end of a
    voltage source (V(m) - V(p) = emf at zero current)."""
    for name in ("real_T", "ideal_T", "plain"):
        r = _readouts()[name]
        d = r.draw()
        marked = {a for a, b, *_ in dr._couplings(r.circuit)} | {b for a, b, *_ in dr._couplings(r.circuit)}
        marked.add(r.squid.input_coil)
        assert set(d.axf_dots) == marked
        for nm, pt in d.axf_dots.items():
            p, m = d.axf_terminals[nm]
            assert math.dist(pt, p) < math.dist(pt, m), nm
        pt = d.axf_plus["pickup"]
        p, m = d.axf_terminals["pickup"]
        src = next(e for e in d.elements if getattr(e, "_axf_name", None) == "pickup")
        ends = sorted((tuple(src.start), tuple(src.end)), key=lambda q: math.dist(q, m))
        assert math.dist(pt, ends[0]) < math.dist(pt, ends[1])     # on the source's m side


def _rows(Cs, far=None):
    """Port k of a box loaded by a capacitor C_k, both on ground (or on node far[k]); plus the pickup port."""
    far = far or {}
    m = MultiportCircuit()
    n = len(Cs) + 1
    m.multiport("field", [(f"mod{k}", far.get(k, "gnd"), f"n{k}") for k in range(len(Cs))]
                + [("pick", "gnd", "np")], lambda w: -1j * w * 1e-7 * np.eye(n), T_K=0.03)
    for k, C in enumerate(Cs):
        m.capacitor(f"C{k}", f"n{k}", far.get(k, "gnd"), C)
    if far:
        m.inductor("Lx1", "x", "gnd", 1e-7)
        m.inductor("Lx2", "x", "gnd", 2e-7)
    m.inductor("L_in", "np", "gnd", 1e-7)
    return m, MultiportReadout(m, fixed_squid(M_H=1e-9), {"mod0": 1.0, "mod1": 1.0, "mod2": 1.0})


def test_compact_folds_only_interchangeable_rows(sd):
    """Folding needs interchangeable elements (not merely equal labels) on the same shared nodes."""
    m, r = _rows([1e-9, 1e-9, 1e-9])
    d = r.draw(compact=True)
    assert d.axf_compacted == {"mod0": ["mod1", "mod2"], "C0": ["C1", "C2"]} and dr.check_drawing(d, r) == []
    for ann in (True, False):                       # different values: never folded, labelled or not
        m2, r2 = _rows([1e-9, 1e-8, 1e-9])
        d2 = r2.draw(compact=True, annotate=ann)
        assert d2.axf_compacted == {} and dr.check_drawing(d2, r2) == []
    m3, r3 = _rows([1e-9, 1e-9, 1e-9], far={1: "x"})        # row 1 sits on node x, not gnd
    d3 = r3.draw(compact=True)
    assert "C1" not in {n for v in d3.axf_compacted.values() for n in v} and dr.check_drawing(d3, r3) == []
    # a folded drawing checked against circuits whose folded rows differ from the drawn one
    import copy
    c = copy.deepcopy(m)
    c._get("C2").value = 2e-9
    c._get("C2").Z = lambda w: 1j / (w * 2e-9)
    assert any("not interchangeable" in e for e in dr.check_drawing(d, c))
    c = copy.deepcopy(m)
    c._get("C2").m = "x9"
    c._node("x9")
    assert any("not private" in e or "maps to both" in e for e in dr.check_drawing(d, c))


def _module_closure(n_modules=8):
    """A multiport closure: 8 module ports + the pickup port, a component on each port,
    the pickup's component in series with the SQUID input coil."""
    n = n_modules + 1
    rng = np.random.default_rng(1)
    A = rng.normal(size=(n, n))
    Lm = 1e-7 * (A @ A.T + n * np.eye(n))
    c = MultiportCircuit()
    names = [f"module{k}" for k in range(n_modules)] + ["pickup"]
    c.multiport("field", [(nm, "gnd", f"n{k}") for k, nm in enumerate(names)],
                lambda w: -1j * w * Lm + 1e-6 * np.eye(n), T_K=0.03)
    for k in range(n):
        c.impedance(f"component{k}", f"n{k}", "s" if k == n - 1 else "gnd", (lambda w: 1j / (w * 1e-9)), noise=False)
    c.inductor("L_in", "s", "gnd", 1e-7)
    r = MultiportReadout(c, fixed_squid(M_H=2e-9), lambda w: dict(zip(names, -1j * w * 1e-3 * np.ones(n))))
    return c, r


@pytest.mark.parametrize("compact", [False, True, None])
def test_multiport_module_closure(sd, compact):
    c, r = _module_closure()
    d = r.draw(omega=2 * math.pi * 1.2e5, compact=compact)
    assert dr.check_drawing(d, c) == []
    if compact:
        assert d.axf_compacted["module0"] == [f"module{k}" for k in range(1, 8)]
        assert d.axf_compacted["component0"] == [f"component{k}" for k in range(1, 8)]
        assert "pickup" in d.axf_terminals and "component8" in d.axf_terminals    # not folded: SQUID row
        assert set(d.axf_terminals) == {"module0", "component0", "pickup", "component8", "L_in"}
    else:            # 9 ports <= COMPACT_PORTS: every row drawn
        assert d.axf_compacted == {} and len(d.axf_terminals) == 19
    allt = " ".join(_labels(d))
    assert "Z(ω): 9×9" in allt and "T = 30 mK" in allt
    assert r"$V_{a}=-i\omega\kappa_a\Psi$" in allt                # sources found by evaluating at omega


def test_multiport_large_n_folds_by_default(sd):
    c, r = _module_closure(n_modules=24)
    d = r.draw(omega=2 * math.pi * 1.2e5)
    assert dr.check_drawing(d, c) == []
    assert len(d.axf_compacted["module0"]) == 23 and "module0 … module23\n(24 ports)" in " ".join(_labels(d))


def test_multiport_with_transformer_line_and_baths(sd):
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 4))
    L4 = 1e-7 * (A @ A.T + 4 * np.eye(4))
    c = MultiportCircuit()
    nm4 = [f"module{k}" for k in range(4)]
    c.multiport("field", [(nm, "gnd", f"n{k}") for k, nm in enumerate(nm4)], lambda w: -1j * w * L4 + 1e-6 * np.eye(4),
                baths=[LossBath("cu", 0.5e-6 * np.eye(4), 4.0), LossBath("sc", 0.5e-6 * np.eye(4), 0.03)])
    for k in range(4):
        c.capacitor(f"C{k}", f"n{k}", "s" if k == 3 else "gnd", 1e-9 * (1 + k))
    c.resistor("R_readout", "s", "s2", 1e-5, T_K=0.03)
    c.transformer("T", "s2", "gnd", "sec", "gnd", 0.3)
    line = LosslessTransmissionLine(1.0, 2.5e-7, 1e-10)
    c.multiport("line", [("line_near", "sec", "gnd"), ("line_far", "line_end", "gnd")], line.impedance)
    c.inductor("L_in", "line_end", "gnd", 1e-7)
    r = MultiportReadout(c, fixed_squid(), {nm: 1.0 for nm in nm4})
    for compact in (False, True):
        d = r.draw(compact=compact)
        assert dr.check_drawing(d, c) == []
        assert d.axf_compacted == {}               # different C values: nothing folds
    allt = " ".join(_labels(d))
    assert "baths: cu 4 K, sc 30 mK" in allt and "Z(ω): 2×2" in allt and "1 : 0.3" in allt
    assert ("T.1", "T.2", "T") in d.axf_couplings
    assert c.draw(annotate=False) is not None       # MultiportCircuit.draw without a SQUID


@pytest.mark.parametrize("preset", ["C1", "C1+low+RT", "C1+medium+cryo", "C1+high+RT"])
def test_chain_drawing(sd, preset):
    ch = squid.chain(preset)
    f = 1e8 if preset.endswith("cryo") else 1e6
    d = ch.draw(f_Hz=f)
    info = d.axf_chain
    assert info["first"] == "C1"
    allt = " ".join(_labels(d))
    if ch.second is None:
        assert info["second"] is None and "no later stages" in allt
    else:
        assert info["second"] == (ch.second.N_ser, ch.second.N_par)
        assert f"{ch.second.N_ser}×{ch.second.N_par}" in allt
        assert dr.fmt_si(ch.second.L2_H, "H") in allt and dr.fmt_si(ch.tau_s, "s") in allt
        assert f"{ch.first.dPhi_dI_H / ch.second.M_IN2_H:.3g}" in allt          # flux gain
    assert f"$\\eta$ = {ch.eta(f):.3g}" in allt and f"s = {ch.sign:+g}" in allt
    assert dr.fmt_si(ch.first.L_sq_H, "H") in allt and f"{ch.first.eta_TC:.3g}" in allt
    sym = " ".join(_labels(ch.draw(annotate=False)))
    assert "pH" not in sym


def test_results_draw_readout(sd):
    from test_circuit import _stub      # the stub Results used by the readout tests
    from axionfem.results import Results
    s = _stub()
    s.draw_readout = Results.draw_readout.__get__(s)
    d = s.draw_readout(f_Hz=1.209e5, benchmark="conservative", L_in_H=1e-7)
    r = s.readout(f_Hz=1.209e5, benchmark="conservative", L_in_H=1e-7, analyze=False)["readout"]
    assert dr.check_drawing(d, r.circuit) == []


def test_missing_schemdraw_is_a_clear_error(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("schemdraw"):
            raise ImportError("no schemdraw")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    for m in [m for m in sys.modules if m.startswith("schemdraw")]:
        monkeypatch.delitem(sys.modules, m)
    with pytest.raises(ImportError, match=r"axionfem\[draw\]"):
        _readouts()["plain"].draw()
    with pytest.raises(ImportError, match="schemdraw"):
        squid.chain("C1").draw()
