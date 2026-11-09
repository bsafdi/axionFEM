"""Regenerate the circuit schematics used in docs/physics/.

    python docs/images/make_circuit_figure.py [--out DIR] [name ...]

Writes ``<name>.svg`` next to this script (or into ``--out``), on a white page so that the drawings
stay legible on a dark theme. The physics pages show the SVG files.

Drawn by the package's own ``draw()`` methods (``axionfem.draw``), from the objects themselves,
so the docs and the API cannot drift apart:

* ``readout_circuit``: ``Readout.draw(annotate=False)`` of the plain series readout of
  :func:`axionfem.circuit.series_readout` with every optional series element present
  (``docs/physics/readout_circuit.md``)::

      gnd -[ pickup: EMF, Z_p ]- a -[ L_lead ]- b -[ R_lead ]- c
          -[ C (tan_delta) ]- d -[ R_in ]- e -[ L_in ]- gnd     (M couples L_in -> SQUID)

* ``squid_chain``: ``Chain.draw(annotate=False)`` of the ``"C1+low+RT"`` dc SQUID chain of
  :mod:`axionfem.squid` (``docs/physics/squid_chain.md``, arXiv:2504.20398).
* ``squid_chain_1MHz``: the same chain, ``Chain.draw(f_Hz=1e6)``: every stage's parameters and
  its noise referred to the first-stage flux at 1 MHz.
* ``squid_readout``: ``Readout.draw(annotate=False)`` of the minimal series readout (pickup
  ``L_p`` with ``R_loss`` at ``T``, tuning ``C``, input coil ``L_in``) with the correlated
  :class:`~axionfem.circuit.Squid` of ``Chain.squid_for_resonator`` (``squid_chain.md`` Sec. 4).

Both readout drawings are checked against their circuit before saving
(:func:`axionfem.draw.check_drawing`: every element, symbol, connection, node label, coupling
and the SQUID coil).

Device-level schematics, drawn here with schemdraw primitives because the package models these
devices by their response and noise, not as netlists (``docs/physics/squid_chain.md``):

* ``squid_first_stage``: the Tesche-Clarke dc SQUID of :class:`axionfem.squid.FirstStage`.
* ``squid_second_stage``: the first stage as a Thevenin source driving ``L_2`` and the
  ``N_ser x N_par`` series array of :class:`axionfem.squid.SecondStage`, leads and preamplifier.
* ``squid_noise_model``: how :meth:`axionfem.squid.Chain.one_sided` combines the noise sources.

Their formulas are not free text: :func:`check_model` recomputes every coefficient and scaling law
they print from :mod:`axionfem.squid` (TC 16/11/12, the beta and Gamma definitions, V_Phi, R_dyn,
the dummy count in ``L_2``, the array scalings, the referral of the later stages, eta and eps_uc)
and the script stops if any of them disagrees with the code.

Needs ``schemdraw`` (>= 0.19) and matplotlib (``pip install schemdraw matplotlib``, the ``draw`` extra).
"""

import math
import re
import sys
from pathlib import Path

import matplotlib

HERE = Path(__file__).resolve().parent
try:
    import axionfem  # noqa: F401
except ImportError:                       # source checkout without an install
    sys.path.insert(0, str(HERE.parents[1] / "python"))
from axionfem import circuit, squid  # noqa: E402
from axionfem.circuit import HBAR, KB, PHI0  # noqa: E402
from axionfem.draw import FS, INK, NOTE, PICK, SQC, amp, check_drawing, squid_symbol  # noqa: E402

GREY = "#8a8a8a"


# ===================================================================== drawn by the package
def readout_circuit():
    # Values only fix the topology here (annotate=False draws symbols): every optional series
    # element of the plain series readout is switched on.
    r = circuit.series_readout(L_p_H=1.4e-6, Q=1e6, f_Hz=1e6, T_K=0.03, L_lead_H=50e-9,
                               R_lead_ohm=1e-4, T_lead_K=4.0, tan_delta=1e-6, L_in_H=2e-9,
                               R_in_ohm=1e-5)
    d = r.draw(annotate=False)
    errs = check_drawing(d, r)
    if errs:
        raise RuntimeError(f"readout drawing does not match its circuit: {errs}")
    return d


def squid_chain():
    return squid.chain("C1+low+RT").draw(annotate=False)


def squid_chain_1MHz():
    return squid.chain("C1+low+RT").draw(f_Hz=1e6)


def squid_readout():
    """The minimal series readout with the C1+low+RT chain at its Eq. 48 coupling (1 MHz)."""
    f0, Q, T, L_p, L_in = 1e6, 1e6, 0.03, 1.4e-6, 2e-9
    r = circuit.series_readout(L_p_H=L_p, Q=Q, f_Hz=f0, T_K=T, L_in_H=L_in, M_H=1e-9)
    r.squid = squid.chain("C1+low+RT").squid_for_resonator(f0, Q, L_p + L_in, T)
    assert r.squid.S_PhiJ != 0.0 and r.squid.input_coil == "L_in"
    d = r.draw(annotate=False)
    errs = check_drawing(d, r)
    if errs:
        raise RuntimeError(f"readout drawing does not match its circuit: {errs}")
    return d


# ===================================================================== the model the schematics print
def check_model():
    """Recompute, from :mod:`axionfem.squid`, every formula the device-level schematics print."""
    def close(a, b, what, rel=1e-12):
        if not math.isclose(a, b, rel_tol=rel, abs_tol=0.0):
            raise RuntimeError(f"schematic text disagrees with axionfem.squid: {what}: {a!r} != {b!r}")
    fs = squid.FirstStage(L_sq_H=150e-12, I0_A=7e-6, R_j_ohm=5.0, T_j_K=0.1, C_j_F=2.0e-12, label="probe")
    kT, R, L = KB * fs.T_j_K, fs.R_j_ohm, fs.L_sq_H
    p = fs.one_sided()
    close(p["S1_Phi_Wb2_Hz"], 16 * kT * L ** 2 / R, "S_Phi = 16 k T_j L_sq^2/R_j")
    close(p["S1_J_A2_Hz"], 11 * kT / R, "S_J = 11 k T_j/R_j")
    close(p["S1_PhiJ_WbA_Hz"], 12 * kT * L / R, "|S_PhiJ| = 12 k T_j L_sq/R_j")
    close(fs.V_Phi_Hz, R / L, "V_Phi = R_j/L_sq")
    close(fs.R_dyn_ohm, R, "R_dyn = R_j")
    close(fs.beta_L, 2 * L * fs.I0_A / PHI0, "beta_L = 2 L_sq I0/Phi0")
    close(fs.beta_C, 2 * math.pi * fs.I0_A * R ** 2 * fs.C_j_F / PHI0, "beta_C = 2 pi I0 R_j^2 C_j/Phi0")
    close(fs.Gamma, 2 * math.pi * kT / (fs.I0_A * PHI0), "Gamma = 2 pi k T_j/(I0 Phi0)")
    close(fs.eps_uc_J_s, 8 * kT * L / R, "eps_uc = S_Phi/2L_sq")
    for s in (1.0, -1.0):
        ch = squid.Chain(fs, sign=s)
        close(float(ch.one_sided(1e6)["S1_PhiJ_WbA_Hz"]), s * 12 * kT * L / R, "S_PhiJ carries the sign s")
    # second stage
    st = squid.SecondStage(7, 3, T2_K=1.5)
    if st.n_dummy != 2:
        raise RuntimeError("schematic draws 2 dummy SQUIDs per bank; SecondStage.n_dummy changed")
    close(st.L2_H, st.N_par * (st.N_ser + 2) * st.L_SQ2_H + st.L_int_H, "L_2 = N_par (N_ser+2) L_SQ2 + L_int")
    close(st.tau_s(fs.R_dyn_ohm), st.L2_H / fs.R_dyn_ohm, "tau = L_2/R_dyn,1")
    u = squid.TABLE_I_PLUS
    close(st.R_dyn_ohm / u["dV_dI_ohm"], (st.N_ser / u["N_ser"]) / (st.N_par / u["N_par"]), "R_dyn ~ N_ser/N_par")
    close(st.dV_dPhi_Hz / u["dV_dPhi_Hz"], st.N_ser / u["N_ser"], "dV/dPhi ~ N_ser")
    close(st.dPhi_dI_H / u["dPhi_dI_H"], u["N_par"] / st.N_par, "|dPhi/dI| ~ 1/N_par")
    close(st.S1_Phi_squid() / (u["Phi_n_SQ_Phi0_rtHz"] * PHI0) ** 2,
          (u["N_ser"] * u["N_par"]) / st.N_SQ * st.T2_K / u["T_K"], "Phi_n,SQ^2 ~ T_2/(N_ser N_par)")
    if squid.SECOND_STAGES != {"low": (48, 3), "medium": (32, 2), "high": (16, 1)}:
        raise RuntimeError("schematic lists the presets low 48x3, medium 32x2, high 16x1; SECOND_STAGES changed")
    # the referral of the later stages, eta and eps_uc
    amp_rt = squid.OpAmpPreamp(0.4e-9, 3e-12, R_lead_ohm=1.5, T_lead_K=150.0)
    ch = squid.Chain(fs, st, amp_rt, sign=-1.0)
    f = 3e6
    w = 2 * math.pi * f
    o = ch.one_sided(f)
    g2 = (fs.dPhi_dI_H / st.M_IN2_H) ** 2 * (1 + (w * ch.tau_s) ** 2)
    close(fs.dPhi_dI_H, fs.R_dyn_ohm / fs.V_Phi_Hz, "|dPhi/dI|_V,1 = R_dyn,1/V_Phi")
    amp_v = (amp_rt.V_n_V_rtHz ** 2 + 4 * KB * amp_rt.T_lead_K * amp_rt.R_lead_ohm) / st.dV_dPhi_Hz ** 2
    amp_i = (amp_rt.I_n_A_rtHz * st.dPhi_dI_H) ** 2
    close(amp_i, (amp_rt.I_n_A_rtHz * st.R_dyn_ohm / st.dV_dPhi_Hz) ** 2, "I_n |dPhi/dI| = I_n R_dyn/(dV/dPhi)",
          rel=0.05)    # equal through B1 up to Table I's rounding (64.94 vs 64.9 ohm)
    close(float(o["S1_Phi_ref_Wb2_Hz"]), g2 * (st.S1_Phi_squid() + amp_v + amp_i), "S_Phi,ref = g2 (Phi_n,SQ^2 + Phi_n2,AMP^2)")
    tot = float(o["S1_Phi_Wb2_Hz"])
    close(tot, p["S1_Phi_Wb2_Hz"] + g2 * (st.S1_Phi_squid() + amp_v + amp_i), "S_Phi,tot = S_Phi + S_Phi,ref")
    close(float(ch.eta(f)), math.sqrt(tot * p["S1_J_A2_Hz"] - p["S1_PhiJ_WbA_Hz"] ** 2) / HBAR, "eta")
    close(float(ch.eps_uc(f)), tot / (2 * L), "eps_uc = S_Phi,tot/2L_sq")
    st_rf = squid.SecondStage(48, 3)
    rf = squid.CryoRFAmp(2.0, 5e6, 500e6)
    close(rf.S1_Phi2(st_rf)["rf_amp"], 4 * KB * rf.T_n_K * st_rf.R_dyn_ohm / st_rf.dV_dPhi_Hz ** 2,
          "CITLF2: 4 k T_n R_dyn/(dV/dPhi)^2", rel=0.05)
    close(squid.CITLF2.R_in_ohm, 50.0, "CITLF2 R_in = 50 ohm")
    # backaction EMF +i omega M J and S_IV = -i omega S_PhiJ
    M = 3e-9
    r = ch.input_referred(f, M)
    close(r["S_VV_V2_Hz"], w ** 2 * M ** 2 * p["S1_J_A2_Hz"], "S_VV = omega^2 M^2 S_J")
    close(r["S_IV_W_Hz"].imag, -w * float(o["S1_PhiJ_WbA_Hz"]), "S_IV = -i omega S_PhiJ")
    sq = circuit.Squid(M_H=M, S_Phi=1.0, S_J=1.0, S_PhiJ=0.5)
    H = 0.3 - 0.7j
    close(sq.correlation_term(w, H), 2 * ((1j * w * M * H).conjugate() * 0.5).real, "cross term of the EMF +i omega M J")


# ===================================================================== device-level schematics
def _drawing():
    import schemdraw
    import schemdraw.elements as elm
    schemdraw.use("matplotlib")
    d = schemdraw.Drawing(show=False, fontsize=FS, lw=1.4, color=INK, bgcolor="white", font="DejaVu Sans", margin=0.6)
    return d, elm


def _text(d, elm, xy, s, color=INK, fs=FS, halign="left", valign="center"):
    d.add(elm.Label().at(xy).label(s, color=color, fontsize=fs, halign=halign, valign=valign))


def _lines(d, elm, xy, lines, color=INK, fs=FS - 1, step=0.55, halign="left"):
    x, y = xy
    for k, s in enumerate(lines):
        _text(d, elm, (x, y - k * step), s, color, fs, halign, "top")


def _box(d, elm, x0, y0, x1, y1, color, lw=1.0):
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        d.add(elm.Line().at(a).to(b).linestyle("--").color(color).linewidth(lw))


def first_stage():
    """The Tesche-Clarke dc SQUID of ``squid.FirstStage`` coupled to the input coil."""
    d, elm = _drawing()
    xl, xr, xc = 6.0, 11.0, 8.5
    ytop, yj1, yj0, yl1, yl0, ybot = 8.2, 7.2, 5.0, 4.4, 2.0, 1.0
    d.add(elm.Line().at((xl, ytop)).to((xr, ytop)).color(SQC))
    d.add(elm.Line().at((xl, ybot)).to((xr, ybot)).color(SQC))
    # bias current in at the top, output voltage across the SQUID
    d.add(elm.Line().at((xc, ytop + 1.4)).to((xc, ytop)).color(SQC))
    d.add(elm.Dot(open=True).at((xc, ytop + 1.4)).color(SQC))
    d.add(elm.Arrow().at((xc - 0.45, ytop + 1.25)).to((xc - 0.45, ytop + 0.35)).color(SQC).linewidth(1.2))
    _text(d, elm, (xc - 0.6, ytop + 0.8), r"$I_{\rm out}$ (bias)", SQC, FS - 1, "right")
    _text(d, elm, (xc + 0.3, ytop + 1.45), r"$V_{\rm out}$", SQC, FS, "left")
    d.add(elm.Line().at((xc, ybot)).down(0.5).color(SQC))
    d.add(elm.Ground().color(SQC))
    for x, inner in ((xl, +1), (xr, -1)):
        d.add(elm.Line().at((x, ytop)).to((x, yj1)).color(SQC))
        d.add(elm.Line().at((x - 1.0, yj1)).to((x + 1.0, yj1)).color(SQC))
        d.add(elm.Line().at((x - 1.0, yj0)).to((x + 1.0, yj0)).color(SQC))
        d.add(elm.Josephson().at((x, yj1)).to((x, yj0)).color(SQC))
        _text(d, elm, (x + 0.25 * inner, (yj1 + yj0) / 2 - 0.55), r"$I_0$", SQC, FS - 1, "left" if inner > 0 else "right")
        xr_, xc_ = x + inner, x - inner
        d.add(elm.Resistor().at((xr_, yj1)).to((xr_, yj0)).color(SQC))
        _text(d, elm, (xr_ + 0.45 * inner, (yj1 + yj0) / 2), r"$R_j$", SQC, FS - 1, "left" if inner > 0 else "right")
        d.add(elm.Capacitor().at((xc_, yj1)).to((xc_, yj0)).color(SQC))
        _text(d, elm, (xc_ - 0.45 * inner, (yj1 + yj0) / 2), r"$C_j$", SQC, FS - 1, "right" if inner > 0 else "left")
        d.add(elm.Line().at((x, yj0)).to((x, yl1)).color(SQC))
        ind = elm.Inductor2(loops=3).at((x, yl1)).to((x, yl0)).color(SQC)
        if x == xr:
            ind = ind.reverse()
        d.add(ind)
        _text(d, elm, (x + 0.6 * inner, (yl1 + yl0) / 2), r"$L_{\rm sq}/2$", SQC, FS - 1, "left" if inner > 0 else "right")
        d.add(elm.Line().at((x, yl0)).to((x, ybot)).color(SQC))
    # loop flux and circulating current
    _text(d, elm, (xc, 6.3), r"$\Phi$", SQC, FS + 2, "center")
    d.add(elm.LoopArrow(direction="cw", width=1.3, height=1.3).at((xc, 3.3)).color(SQC).linewidth(1.2))
    _text(d, elm, (xc, 3.3), r"$J$", SQC, FS, "center")
    # input coil
    xi = 2.6
    d.add(elm.Line().at((0.0, 5.6)).to((xi, 5.6)))
    d.add(elm.Line().at((xi, 5.6)).to((xi, yl1)))
    d.add(elm.Inductor2(loops=3).at((xi, yl1)).to((xi, yl0)).reverse())
    d.add(elm.Line().at((xi, yl0)).to((xi, ybot)))
    d.add(elm.Line().at((xi, ybot)).to((0.0, ybot)))
    d.add(elm.Dot(open=True).at((0.0, 5.6)))
    d.add(elm.Dot(open=True).at((0.0, ybot)))
    _text(d, elm, (xi - 0.55, (yl1 + yl0) / 2), r"$L_{\rm in}$", INK, FS, "right")
    _lines(d, elm, (-0.2, 4.9), ["to the readout", "loop (Sec. 4)"], NOTE, FS - 2, 0.45)
    d.add(elm.Arc2(arrow="<->", k=0.35).at((xi + 0.65, (yl1 + yl0) / 2 - 0.4)).to((xl - 0.75, (yl1 + yl0) / 2 - 0.4)).linewidth(1.2))
    _text(d, elm, ((xi + xl) / 2 - 0.1, 3.75), r"$M$", INK, FS + 1, "center")
    _lines(d, elm, (-0.2, 0.3), [r"$M=\kappa\sqrt{L_{\rm sq}L_{\rm in}}$",
                                 r"measured flux $\Phi=MI_{\rm in}+\Phi_n$",
                                 r"backaction EMF $+i\omega MJ$ in $L_{\rm in}$"], INK, FS - 1, 0.5)
    _box(d, elm, xl - 1.9, ybot - 1.5, xr + 1.9, ytop + 0.3, SQC)
    # model
    _lines(d, elm, (xr + 2.6, 9.4), [
        "Tesche–Clarke dc SQUID (squid.FirstStage)",
        r"per junction: $I_0$, shunt $R_j$ at $T_j$, $C_j$",
        r"$\beta_L=2L_{\rm sq}I_0/\Phi_0\approx1$,  $\beta_C=2\pi I_0R_j^2C_j/\Phi_0\approx1$",
        r"$\Gamma=2\pi k_BT_j/I_0\Phi_0\lesssim0.05$",
        r"$V_\Phi=(\partial V_{\rm out}/\partial\Phi)_{I_{\rm out}}=R_j/L_{\rm sq}$,  $R_{\rm dyn}=R_j$",
        "",
        "noise (one-sided, referred to the loop flux):",
        r"imprecision  $S^{(1)}_\Phi=16k_BT_jL_{\rm sq}^2/R_j$",
        r"circulating current  $S^{(1)}_J=11k_BT_j/R_j$",
        r"correlation  $S^{(1)}_{\Phi J}=s\,12k_BT_jL_{\rm sq}/R_j$,  $s=\pm1$ (open)",
        r"$\eta_{TC}=\sqrt{S^{(1)}_\Phi S^{(1)}_J-(S^{(1)}_{\Phi J})^2}/\hbar=4\sqrt{2}\,k_BT_jL_{\rm sq}/\hbar R_j$",
        r"$\epsilon_{\rm uc}=S^{(1)}_\Phi/2L_{\rm sq}=8k_BT_jL_{\rm sq}/R_j$",
    ], SQC, FS - 1, 0.62)
    return d


def _squid_h(d, elm, cx, cy, w=1.0, h=0.8, color=SQC):
    """dc SQUID with its two junctions on the top and bottom edges; terminals left and right."""
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    d.add(elm.Line().at((x0, y0)).to((x0, y1)).color(color))
    d.add(elm.Line().at((x1, y0)).to((x1, y1)).color(color))
    d.add(elm.Josephson().at((x0, y1)).to((x1, y1)).color(color))
    d.add(elm.Josephson().at((x0, y0)).to((x1, y0)).color(color))
    return (x0, cy), (x1, cy)


def second_stage():
    """First stage (Thevenin) -> L_2 -> N_ser x N_par series array -> leads -> preamplifier."""
    d, elm = _drawing()
    yin = 4.0
    # first stage as a Thevenin source
    d.add(elm.Ground().at((0.0, 1.4)).color(SQC))
    d.add(elm.SourceV().at((0.0, 1.4)).to((0.0, yin)).color(SQC))
    _text(d, elm, (-0.75, 2.7), r"$V_\Phi\Phi$", SQC, FS, "right")
    d.add(elm.Resistor().at((0.0, yin)).to((2.2, yin)).color(SQC))
    d.add(elm.Line().at((2.2, yin)).to((2.6, yin)))
    _text(d, elm, (1.1, yin + 0.6), r"$R_{\rm dyn,1}$", SQC, FS - 1, "center")
    _box(d, elm, -2.3, 0.5, 2.4, yin + 1.3, SQC)
    _lines(d, elm, (-2.2, 0.25), ["1st stage, voltage-biased:", r"Thévenin source ($V_\Phi\Phi$, $R_{\rm dyn,1}$)"], SQC, FS - 2, 0.45)
    d.add(elm.Inductor2(loops=3).at((2.6, yin)).to((4.6, yin)))
    _text(d, elm, (3.6, yin + 0.6), r"$L_{\rm int}$", INK, FS - 1, "center")
    d.add(elm.Arrow().at((3.2, yin - 0.45)).to((4.2, yin - 0.45)).linewidth(1.0))
    _text(d, elm, (3.7, yin - 0.65), r"$I_1$", INK, FS - 1, "center", "top")
    # two banks drawn: bank 1 (coils left->right, SQUIDs above) and bank N_par (coils right->left, SQUIDs below)
    cw = 1.5
    labels = ["dummy", "1", "2", None, r"$N_{\rm ser}$", "dummy"]
    x0 = 4.6
    xs = [x0 + k * cw for k in range(len(labels) + 1)]
    xrail = xs[-1] + 1.6
    rows = ((yin, yin + 1.45, "bank 1"), (yin - 3.0, yin - 4.45, r"bank $N_{\rm par}$"))
    for r, (yc, ys, bank) in enumerate(rows):
        prev = None
        for k, lab in enumerate(labels):
            a, b = (xs[k], yc), (xs[k + 1], yc)
            cx = (a[0] + b[0]) / 2
            if lab is None:
                d.add(elm.Line().at(a).to(b))
                _text(d, elm, (cx, yc + 0.25), r"$\cdots$", INK, FS, "center", "bottom")
                if prev is not None:
                    d.add(elm.Line().at(prev).to((cx - 0.3, ys)).color(SQC))
                    _text(d, elm, (cx, ys), r"$\cdots$", SQC, FS, "center")
                    prev = (cx + 0.3, ys)
                continue
            ind = elm.Inductor2(loops=2).at(a if r == 0 else b).to(b if r == 0 else a)
            d.add(ind if r == 0 else ind.reverse())
            if lab == "dummy":
                _text(d, elm, (cx, yc + (-0.5 if r == 0 else 0.5)), "dummy", GREY, FS - 3, "center",
                      "top" if r == 0 else "bottom")
                continue
            left, right = _squid_h(d, elm, cx, ys)
            _text(d, elm, (cx, ys + (0.8 if r == 0 else -0.8)), lab, SQC, FS - 2, "center", "bottom" if r == 0 else "top")
            d.add(elm.Line().at((cx, yc + (0.45 if r == 0 else -0.45))).to((cx, ys + (-0.55 if r == 0 else 0.55)))
                  .linestyle(":").color(NOTE).linewidth(1.0))
            if prev is None:
                d.add(elm.Line().at(left).to((left[0] - 0.5, ys)).color(SQC))
                d.add(elm.Ground().at((left[0] - 0.5, ys)).color(SQC))
            else:
                d.add(elm.Line().at(prev).to(left).color(SQC))
            prev = right
        d.add(elm.Line().at(prev).to((xrail, ys)).color(SQC))
        _text(d, elm, (xs[0] - 0.2, ys), bank, SQC, FS - 1, "right")
    # input coils in series: bank 1 -> bank N_par -> ground
    d.add(elm.Line().at((xs[-1], yin)).to((xs[-1] + 0.6, yin)))
    d.add(elm.Line().at((xs[-1] + 0.6, yin)).to((xs[-1] + 0.6, yin - 3.0)))
    d.add(elm.Line().at((xs[-1] + 0.6, yin - 3.0)).to((xs[-1], yin - 3.0)))
    _text(d, elm, (xs[-1] + 0.75, yin - 1.5), r"$\vdots$", INK, FS, "left")
    d.add(elm.Line().at((xs[0], yin - 3.0)).to((xs[0] - 0.6, yin - 3.0)))
    d.add(elm.Ground().at((xs[0] - 0.6, yin - 3.0)))
    _text(d, elm, (xs[2] + 0.5 * cw, yin - 0.45), r"$L_{SQ2}$", INK, FS - 2, "center", "top")
    _text(d, elm, (xs[2] + 0.72 * cw, (yin + yin + 1.45) / 2 + 0.05), r"$M_{IN2}$", NOTE, FS - 2, "left")
    _text(d, elm, (xs[0] + 1.5 * cw, yin - 1.5), r"$\vdots$  $N_{\rm par}$ banks, outputs in parallel", SQC, FS - 2, "left")
    # output: rail, leads, preamplifier
    ytop, ybot = rows[0][1], rows[1][1]
    d.add(elm.Line().at((xrail, ytop)).to((xrail, ybot)).color(SQC))
    ymid = yin - 1.5
    d.add(elm.Line().at((xrail, ymid)).to((xrail + 0.6, ymid)).color(SQC))
    d.add(elm.Resistor().at((xrail + 0.6, ymid)).to((xrail + 3.0, ymid)).color(SQC))
    _text(d, elm, (xrail + 1.8, ymid + 0.6), r"$R_{\rm lead}(T_{\rm lead})$", SQC, FS - 1, "center", "bottom")
    d.add(elm.Line().at((xrail + 3.0, ymid)).to((xrail + 3.4, ymid)).color(SQC))
    out = amp(d, elm, (xrail + 3.4, ymid), size=1.5)
    d.add(elm.Arrow().at(out).right(0.9).color(SQC))
    _text(d, elm, (out[0] + 1.0, ymid), r"$V_{\rm out}$", SQC, FS, "left")
    _lines(d, elm, (xrail + 0.4, ymid - 1.2), ["preamplifier:", r"RT op-amp mode: $V_n$, $I_n$",
                                               r"or cryogenic RF (CITLF2):", r"$T_n$, $R_{\rm in}=50\,\Omega$, $R_{\rm lead}=0$"],
           SQC, FS - 2, 0.45)
    # model
    _lines(d, elm, (-2.3, -2.0), [
        r"input circuit:  $L_2=N_{\rm par}(N_{\rm ser}+2)L_{SQ2}+L_{\rm int}$  (2 dummies per bank: input inductance only),"
        r"   $\tau=L_2/R_{\rm dyn,1}$,   $I_1=|\partial I/\partial\Phi|_{V,1}\,\Phi/(1-i\omega\tau)$",
        r"array (Table I 20×1 unit, Eqs. 30–34):  $R_{\rm dyn}\propto N_{\rm ser}/N_{\rm par}$,  $\partial V/\partial\Phi\propto N_{\rm ser}$,"
        r"  $|\partial\Phi/\partial I|\propto1/N_{\rm par}$,  $P\propto N_{\rm ser}N_{\rm par}$,  $\Phi^2_{n,\rm SQ}\propto T_2/N_{\rm ser}N_{\rm par}$",
        r"presets: low 48×3, medium 32×2, high 16×1, at $T_2$ = 1 K.   Flux and current bias at Table I's + feedback slope;"
        " no feedback loop is modelled.",
    ], INK, FS - 1, 0.6)
    return d


def noise_model():
    """Signal-flow view of ``Chain.one_sided``: where each noise enters and how it is referred."""
    import schemdraw.dsp as dsp
    d, elm = _drawing()
    y = 0.0
    _text(d, elm, (-0.2, y), r"$MI_{\rm in}$", INK, FS, "right")
    d.add(dsp.Arrow().at((0.0, y)).right(1.0))
    s1 = d.add(dsp.Sum().anchor("W").color(SQC))
    d.add(dsp.Arrow().at(s1.E).right(0.9).color(SQC))
    _text(d, elm, (s1.E[0] + 0.45, y + 0.3), r"$\Phi$", SQC, FS, "center", "bottom")
    b1 = d.add(dsp.Box(w=3.6, h=1.3).anchor("W").label(r"$\dfrac{|\partial I/\partial\Phi|_{V,1}}{1-i\omega\tau}$", fontsize=FS + 1).color(SQC))
    d.add(dsp.Arrow().at(b1.E).right(0.9).color(SQC))
    _text(d, elm, (b1.E[0] + 0.45, y + 0.3), r"$I_1$", SQC, FS, "center", "bottom")
    b2 = d.add(dsp.Box(w=1.6, h=1.3).anchor("W").label(r"$M_{IN2}$").color(SQC))
    d.add(dsp.Arrow().at(b2.E).right(0.9).color(SQC))
    s2 = d.add(dsp.Sum().anchor("W").color(SQC))
    d.add(dsp.Arrow().at(s2.E).right(0.9).color(SQC))
    b3 = d.add(dsp.Box(w=2.4, h=1.3).anchor("W").label(r"$\partial V/\partial\Phi$" + "\n(array)", fontsize=FS - 1).color(SQC))
    d.add(dsp.Arrow().at(b3.E).right(0.9).color(SQC))
    s3 = d.add(dsp.Sum().anchor("W").color(SQC))
    d.add(dsp.Arrow().at(s3.E).right(1.0).color(SQC))
    _text(d, elm, (s3.E[0] + 1.15, y), r"$V_{\rm out}$", SQC, FS, "left")
    # noise inputs from above
    for s, lines in ((s1, [r"$\Phi_n$: TC imprecision", r"$S^{(1)}_\Phi=16k_BT_jL_{\rm sq}^2/R_j$"]),
                     (s2, [r"$\Phi_{n,\rm SQ}$: array", r"$\Phi^2_{n,20\times1}\,(20/N_{\rm ser}N_{\rm par})\,(T_2/4\,{\rm K})$"]),
                     (s3, [r"preamp: $V_n$, $4k_BT_{\rm lead}R_{\rm lead}$, $I_nR_{\rm dyn}$",
                           r"or RF amp: $4k_BT_nR_{\rm dyn}$"])):
        d.add(dsp.Arrow().at((s.N[0], y + 2.0)).to(s.N).color(SQC))
        _lines(d, elm, (s.N[0], y + 3.05), lines, SQC, FS - 1, 0.6, "center")
    # backaction branch
    jx = s1.S[0]
    d.add(elm.Line().at((jx, y - 2.6)).to((jx, y - 2.6)).color(SQC))
    _lines(d, elm, (jx, y - 2.3), [r"$J$: circulating current", r"$S^{(1)}_J=11k_BT_j/R_j$"], SQC, FS - 1, 0.6, "center")
    d.add(dsp.Arrow().at((jx - 1.6, y - 2.55)).to((-1.0, y - 2.55)).color(SQC))
    d.add(elm.Line().at((jx - 1.6, y - 2.55)).to((jx - 1.2, y - 2.55)).color(SQC))
    _lines(d, elm, (-1.1, y - 1.95), [r"backaction EMF $+i\omega MJ$ in $L_{\rm in}$", r"$S_{VV}=\omega^2M^2S^{(1)}_J$", "(drives $I_{\\rm in}$ through the readout, Sec. 4)"], INK, FS - 1, 0.55, "right")
    d.add(elm.Arc2(arrow="<->", k=0.0).at((jx + 0.15, y - 0.55)).to((jx + 0.15, y - 2.1)).linestyle("--").color(NOTE).linewidth(1.0))
    _lines(d, elm, (jx + 0.4, y - 1.1), [r"correlated: $S^{(1)}_{\Phi J}=s\,12k_BT_jL_{\rm sq}/R_j$"], NOTE, FS - 1, 0.5)
    # referral and results
    _lines(d, elm, (-4.6, y - 4.2), [
        r"referred to the first-stage flux (Eq. 35, Q3.2):   $S^{(1)}_{\Phi,\rm tot}=S^{(1)}_\Phi+g_2(\omega)\,"
        r"\left[\Phi^2_{n,\rm SQ}+\Phi^2_{n2,\rm AMP}\right]$,    $g_2(\omega)=\left(\frac{|\partial\Phi/\partial I|_{V,1}}{M_{IN2}}\right)^2(1+\omega^2\tau^2)$",
        r"$\Phi^2_{n2,\rm AMP}=\frac{V_n^2+4k_BT_{\rm lead}R_{\rm lead}}{(\partial V/\partial\Phi)^2}+(I_n|\partial\Phi/\partial I|)^2$"
        r"  (RT),    $\frac{4k_BT_nR_{\rm dyn}}{(\partial V/\partial\Phi)^2}$  (cryogenic RF);   later stages: no backaction, uncorrelated with $J$",
        r"$\eta(f)=\sqrt{S^{(1)}_{\Phi,\rm tot}S^{(1)}_J-(S^{(1)}_{\Phi J})^2}\,/\,\hbar$,     "
        r"$\epsilon_{\rm uc}(f)=S^{(1)}_{\Phi,\rm tot}/2L_{\rm sq}$     (Eqs. 42–43; both independent of $M$)",
    ], INK, FS, 0.85)
    return d


FIGURES = {"readout_circuit": readout_circuit, "squid_chain": squid_chain, "squid_chain_1MHz": squid_chain_1MHz,
           "squid_first_stage": first_stage, "squid_second_stage": second_stage,
           "squid_noise_model": noise_model, "squid_readout": squid_readout}


if __name__ == "__main__":
    matplotlib.use("Agg")  # only as a script: importing this module (squid_chain_demo.ipynb) must keep the notebook backend
    args = sys.argv[1:]
    out_dir = HERE
    if args[:1] == ["--out"]:
        out_dir, args = Path(args[1]), args[2:]
    check_model()
    for name in args or FIGURES:
        d = FIGURES[name]()
        out = out_dir / f"{name}.svg"
        d.save(str(out), transparent=False, dpi=150)  # white page: legible on a dark theme
        # no creation date in the file: the drawing depends only on the code
        out.write_text(re.sub(r"^\s*<dc:date>[^<]*</dc:date>\n", "", out.read_text(), flags=re.M))
        print(out, out.stat().st_size // 1024, "KB")
