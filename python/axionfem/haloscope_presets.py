"""Published parameters of cavity-haloscope runs, for :mod:`axionfem.haloscope`.  Beta.

Every number carries its source (arXiv id + section/table/equation/page).  Tags in ``notes``:

* INFERRED: derived by us from stated numbers (the derivation is given);
* ASSUMED: a choice we made where the paper gives a range or nothing usable;
* UNVERIFIED: not found in the paper; do not trust for a reproduction;
* FIGURE-READ: read off a published plot (no table exists), +-10% at best.

Conventions (see docs/physics/haloscope_sensitivity.md Sec. 7):

* ``cavity``: ``B_T`` as used in the paper's power formula (central or rms, see notes),
  ``V_L`` litres, ``C``, and either ``Q0`` + ``beta`` or ``Q_L`` + ``beta``.
* ``receiver``: keyword arguments of :class:`axionfem.haloscope.Receiver`.
* ``analysis``: ``mode`` "scan" (``rate_Hz_s`` = band / live integration time, the dense-scan
  equivalent of steps df/tau) or "dwell" (``t_s`` at one tuning); ``snr_target`` = SNR a
  signal needs to be excluded (threshold + one-sided z of the CL, or the paper's stated target);
  ``Q_a`` (radiometer convention, f/Delta f_eff) None -> SHM line (Q_a = 6.78e5) with optimal
  weighting, else Delta f_eff = f/Q_a.  ``limit_type`` (doc H6.1):
  "threshold" (candidate threshold + CL, the target is the paper's own), "likelihood" (the
  paper's limit formula is a zero-excess / power-constrained one that we reproduce exactly:
  QUAX), or "likelihood_approx" (a likelihood limit on real data that the zero-excess median
  only approximates: ADMX; excluded from the ensemble-bias statistic, own wider window).
* ``published``: reference frequency and the published exclusion there, as ``g_GeV`` or as
  ``x_KSVZ`` (|g_gamma/g_gamma^KSVZ|, converted with the haloscope convention, Lambda = 77.6 MeV).
"""
from __future__ import annotations

import math

PRESETS: dict[str, dict] = {
    # ------------------------------------------------------------------------------------ ADMX
    "ADMX_1A": {
        "experiment": "ADMX", "run": "Run 1A (2017)",
        "refs": ["arXiv:1804.05750 (PRL 120, 151301)", "arXiv:2312.16668 (Run 1A analysis details)",
                 "arXiv:2010.00169 (design and operations)"],
        "f_range_Hz": (645e6, 680e6),
        "cavity": {"B_T": 6.8, "V_L": 136.0, "C": 0.4, "Q_L": 5.0e4, "beta": 1.0},
        "receiver": {"T_sys_K": 0.5},
        "analysis": {"mode": "scan", "rate_Hz_s": 35e6 / (78958 * 100.0), "snr_target": 1.2816,
                     "confidence": 0.90, "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "likelihood_approx"},
        "published": {"f_Hz": 660e6, "g_GeV": 3.25e-16,
                      "statement": "excludes DFSZ (isothermal halo, 0.45 GeV/cm^3) between 645 and 676 MHz, "
                                   "90% upper confidence limit (1804.05750 p.4)",
                      "source": "FIGURE-READ 1804.05750 Fig. 4 at ~660 MHz: 3.0-3.5e-16 GeV^-1 (MB halo)"},
        "sources": {
            "B_T": "1804.05750 p.4 'typical field of 6.8 T'; Eq. 1 normalisation (B/6.8 T)",
            "V_L": "1804.05750 p.1 '136-liter'",
            "C": "1804.05750 Eq. 1 (C/0.4); 2010.00169 'average value of 0.4'",
            "Q_L": "1804.05750 Eq. 1 normalisation (Q/50,000), loaded Q",
            "beta": "1804.05750 p.2 'critically coupled antenna'; 2312.16668 Sec. II Q_loaded = Q_free/2",
            "T_sys_K": "2312.16668 'system noise temperature typically of ~500 mK'",
            "rate_Hz_s": "1804.05750: 78,958 spectra of 100 s used, 645-680 MHz (INFERRED rate = band/live)",
            "snr_target": "1804.05750 p.3-4 90% upper confidence limit; ASSUMED zero-excess one-sided z = 1.2816",
            "rho": "1804.05750 p.4: 0.45 GeV/cm^3 (MB), 0.63 (N-body)",
        },
        "notes": ["Q_L = 5e4 is the Eq. 1 normalisation, not a measured average (Q 40k-80k, 2010.00169).",
                  "Limit is a likelihood upper limit on the grand spectrum, not a threshold+CL target: "
                  "snr_target = 1.28 is the zero-excess median; rescans and SG-filter attenuation "
                  "(value NOT FOUND) are not modelled.",
                  "ADMX line shape: boosted MB, v_lab = 230 km/s (vs package SHM 220/232)."],
    },
    "ADMX_1D": {
        "experiment": "ADMX", "run": "Run 1D (2023-24)",
        "refs": ["arXiv:2504.07279 (PRL 135, 191001)"],
        "f_range_Hz": (1.10e9, 1.31e9),
        "cavity": {"B_T": 7.6, "V_L": 106.0, "C": 0.4, "Q0": 9.0e4, "Q_L": 4.5e4, "beta": 1.0},
        "receiver": {"T_sys_K": 0.59, "eta_snr": 0.90},
        "analysis": {"mode": "scan", "rate_Hz_s": None, "snr_target": 1.2816, "confidence": 0.90,
                     "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "likelihood_approx"},
        "published": {"f_Hz": 1.25e9, "g_GeV": 1.15e-15,
                      "statement": "better-than-KSVZ sensitivity across 1.10-1.31 GHz, 90% CL (2504.07279 p.5)",
                      "source": "FIGURE-READ 2504.07279 Fig. 3 inset at ~1250 MHz: 1.0-1.3e-15 GeV^-1"},
        "sources": {
            "B_T": "2504.07279 Eq. 1 normalisation (B/7.6 T); 7.8 T central, 5.3 T at the ends",
            "V_L": "2504.07279 Eq. 1 (V/106 l), 'volume of empty space in the cavity' (139 l total)",
            "C": "2504.07279 Eq. 1 normalisation (C/0.4); typical value NOT stated",
            "Q0": "2504.07279 p.2 'unloaded quality factor was about 90,000'; Eq. 1 Q_L/45,000",
            "beta": "2504.07279 'kept in an overcoupled state'; beta = 1 from Q0/Q_L normalisations (INFERRED)",
            "T_sys_K": "2504.07279 p.4 'mean value of Tsys was 0.59 +/- 0.31 K'",
            "eta_snr": "2504.07279 p.5 'signal efficiency (90%) due to the Savitzky-Golay filter' (power; "
                       "applied here to the SNR, i.e. to signal power at fixed noise)",
            "rate_Hz_s": "NOT FOUND (step size and live time not published); None",
            "snr_target": "2504.07279 Eq. 5 truncated-normal 90% CL; ASSUMED zero-excess z = 1.2816",
        },
        "notes": ["Step size and live time NOT FOUND: no scan rate, so no reproduction; use "
                  "haloscope.implied_rate() to get the rate the published limit implies."],
    },
    # --------------------------------------------------------------------------------- HAYSTAC
    "HAYSTAC_2017": {
        "experiment": "HAYSTAC", "run": "Phase I, first results",
        "refs": ["arXiv:1610.02580 (PRL 118, 061302)", "arXiv:1706.08388 (analysis procedure)",
                 "arXiv:1611.07123 (design)"],
        "f_range_Hz": (5.694e9, 5.803e9),
        "cavity": {"B_T": 9.0, "V_L": 1.545, "C": 0.5, "Q0": 3.0e4, "beta": 2.0},
        "receiver": {"T_cav_K": 0.47, "T_in_K": 0.127, "N_A": 1.35, "eta_snr": 0.76},
        "analysis": {"mode": "scan", "rate_Hz_s": 108.8e6 / (6766 * 900.0), "snr_target": 5.1,
                     "confidence": 0.95, "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "threshold"},
        "published": {"f_Hz": 5.75e9, "x_KSVZ": 2.3, "Lambda_MeV": 78.0,
                      "statement": "On average we exclude |g_gamma| >~ 2.3 x |g_gamma^KSVZ| for 23.55 < m_a < "
                                   "24.0 ueV (1610.02580 p.5); 95% per 1706.08388, '>=90%' in the PRL",
                      "source": "1610.02580 p.5 (text)"},
        "sources": {
            "B_T": "1610.02580 p.2; 1706.08388 p.10 '9 T'",
            "V_L": "1706.08388 p.10 V = 1.545 L excluding the rod",
            "C": "1610.02580 p.2 'C010 ~ 0.5'",
            "Q0": "1610.02580 p.2 'Q0 ~ 3 x 10^4'", "beta": "1610.02580 p.2 'beta ~ 2'",
            "T_in_K": "1610.02580 p.2 'T_C = 127 mK' (fridge/attenuator temperature)",
            "N_A": "1610.02580 p.3 'N_A = 1.35 quanta in total' (0.63 JPA, 0.2 HEMT, 0.5 loss)",
            "T_cav_K": "INFERRED: 1610.02580 p.3 k_B T_S ~ 3 h nu on resonance falling to 2.2 h nu at 650 kHz; "
                       "a hot cavity (n_c = 1.25, T = 0.47 K) behind a 127 mK isolator reproduces both",
            "eta_snr": "1610.02580 p.4 'eta = 0.76' (SG fit, misalignment, loss before the switch)",
            "rate_Hz_s": "1706.08388: M = 6766 spectra of tau = 15 min (Mtau = 1692 h) over 23.55-24.0 ueV",
            "snr_target": "1610.02580 p.4 '5.1 sigma SNR target, candidate threshold of 3.455 sigma at 95%'",
        },
        "notes": ["eta = 0.76 covers loss before the microwave switch 'to which the noise calibration is not "
                  "sensitive' (1610.02580 p.4); the 0.5 quanta in N_A come from the ~2 dB after the switch "
                  "(p.3).  Distinct losses: no double counting.",
                  "Rescans (282 h) not counted; the rescan only clears candidates."],
    },
    "HAYSTAC_2021": {
        "experiment": "HAYSTAC", "run": "Phase IIa, squeezed (corrected)",
        "refs": ["arXiv:2008.01853 (Nature 590, 238)", "arXiv:2301.09721 (PRD 107, 072007) Sec. III, Table II"],
        "f_range_Hz": (4.100e9, 4.178e9),
        "cavity": {"B_T": 8.0, "V_L": 1.545, "C": 0.43, "Q_L": 6180.0, "beta": 6.9},
        "receiver": {"T_cav_K": 0.061, "n_cav": 2 * 0.41 - 0.5, "N_A": 0.08,
                     "squeeze_dB": 10.0 * math.log10(1.0 / 0.11),
                     "squeeze_eff": 0.63 / 0.794, "eta_out": 0.794,
                     "phase_sensitive": True, "eta_snr": 0.91},
        "analysis": {"mode": "scan", "rate_Hz_s": 73e6 / (828 * 3600.0), "snr_target": 5.1,
                     "confidence": 0.90, "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "threshold"},
        "published": {"f_Hz": 4.14e9, "x_KSVZ": 1.95,
                      "statement": "g_gamma >= 1.95 x g_gamma^KSVZ at the 90% level (corrected from the "
                                   "published 1.38 by sqrt(2); 2301.09721 Sec. III)",
                      "source": "2301.09721 Sec. III (text)"},
        "sources": {
            "B_T": "2008.01853 App. A 8 T", "V_L": "2301.09721 1.545 L (Nature: 1.5 L)",
            "C": "2301.09721 Table II C = 0.43 (Nature: ~0.5)",
            "Q_L": "2301.09721 Table II Q_L = 6180 (Q0 = Q_L(1+beta) = 48.8k; 2008.01853 App. A 'Q0 = 47000 +- 5000')",
            "beta": "2301.09721 Table II beta = 6.9",
            "T_cav_K": "2008.01853 App. A / 2301.09721: 61 mK mixing chamber (input load and loss baths, Eq. A4 S_mc)",
            "n_cav": "2301.09721 Table II S_C = 0.41 per quadrature, Eq. A3 S = (1/2)(1/2 + n) -> n_c = 2 S_C - 1/2 "
                     "= 0.32 (T ~ 0.14 K; excess cavity/rod noise, 2008.01853 App. A 0.41 vs 0.27 expected)",
            "N_A": "2301.09721 Table II N_a = 0.08 (single quadrature, referred to the AMP input, Eq. A4)",
            "squeeze_dB": "2301.09721 Table II G_S = 0.11 (SQ one-quadrature gain used in the analysis, Eqs. A4, A8; "
                          "~50% uncertainty): 9.59 dB.  The text's '13 dB SQ gain' (Sec. II A) would give 9% "
                          "less noise; NOT the output-measured 4.0 dB (that includes N_a and alpha: H4.3)",
            "squeeze_eff": "2301.09721 Table II: rho = (alpha rho)/alpha = 0.63/0.794 (squeezer -> cavity)",
            "eta_out": "2301.09721 Table II alpha = 0.794 (cavity -> AMP; = sqrt(0.63), App. A 2 even-split assumption)",
            "eta_snr": "ASSUMED = Phase IIb SG attenuation 0.91 (2301.09721 Sec. II E)",
            "rate_Hz_s": "2008.01853: 861 spectra (828 used) of 3600 s over 4.100-4.140 and 4.145-4.178 GHz (73 MHz)",
            "snr_target": "2301.09721 Sec. II E: 5.1 sigma target, 10% two-scan false negative (Bayesian aggregate)",
        },
        "notes": ["Bayesian aggregate exclusion (U = 10%), approximated by snr_target = 5.1.",
                  "Receiver = 2301.09721 Eq. A4 with the Table II averages (alpha, alpha rho, G_S, S_C, N_a); "
                  "lambda (loss before the squeezer) drops out for a load at T_mc."],
    },
    "HAYSTAC_2023": {
        "experiment": "HAYSTAC", "run": "Phase IIb, squeezed",
        "refs": ["arXiv:2301.09721 (PRD 107, 072007)"],
        "f_range_Hz": (4.459e9, 4.523e9),
        "cavity": {"B_T": 8.0, "V_L": 1.545, "C": 0.43, "Q_L": 5915.0, "beta": 6.5},
        "receiver": {"T_cav_K": 0.061, "n_cav": 2 * 0.40 - 0.5, "N_A": 0.08,
                     "squeeze_dB": 10.0 * math.log10(1.0 / 0.08),
                     "squeeze_eff": 0.60 / 0.775, "eta_out": 0.775,
                     "phase_sensitive": True, "eta_snr": 0.91},
        "analysis": {"mode": "scan", "rate_Hz_s": 64e6 / (801 * 3600.0), "snr_target": 5.1,
                     "confidence": 0.90, "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "threshold"},
        "published": {"f_Hz": 4.49e9, "x_KSVZ": 2.06,
                      "statement": "a 90% aggregate exclusion of g_gamma >= 2.06 x g_gamma^KSVZ over 18.44-18.71 ueV",
                      "source": "2301.09721 Sec. III and abstract (text)"},
        "sources": {
            "B_T": "2301.09721 8 T", "V_L": "2301.09721 Table II 1.545 L", "C": "2301.09721 Table II 0.43",
            "Q_L": "2301.09721 Table II Q_L = 5915 (Q0 = 44.4k; Sec. II B 'Q0 ~ 45 x 10^3')",
            "beta": "2301.09721 Table II average beta = 6.5",
            "T_cav_K": "2301.09721 61 mK mixing chamber (input load and loss baths, Eq. A4 S_mc)",
            "n_cav": "2301.09721 Table II S_C = 0.40 -> n_c = 2 S_C - 1/2 = 0.30 (Eq. A3)",
            "N_A": "2301.09721 Table II N_a = 0.08 (referred to the AMP input)",
            "squeeze_dB": "2301.09721 Table II G_S = 0.08 (analysis value, ~50% uncertainty): 10.97 dB; NOT the "
                          "output-measured 3.7 dB (H4.3)",
            "squeeze_eff": "2301.09721 Table II: rho = (alpha rho)/alpha = 0.60/0.775",
            "eta_out": "2301.09721 Table II alpha = 0.775 (cavity -> AMP)",
            "eta_snr": "2301.09721 Sec. II E SG filter signal attenuation 0.91",
            "rate_Hz_s": "2301.09721 Sec. III: 801 spectra ~80 kHz apart, tau = 1 h (Sec. II D)",
            "snr_target": "2301.09721 Sec. II E threshold 3.468 sigma for a 5.1 sigma target (10% false negative)",
        },
        "notes": ["Expected scan-rate enhancement from squeezing 1.9 (2301.09721 Sec. II C) refers to the delivered "
                  "3.7 dB with negligible added noise; reproduced with N_A = 0 (test), not by this preset.",
                  "Receiver = 2301.09721 Eq. A4 with the Table II averages."],
    },
    # ------------------------------------------------------------------------------------ CAPP
    "CAPP_12TB": {
        "experiment": "CAPP", "run": "CAPP-12TB DFSZ (2022)",
        "refs": ["arXiv:2210.10961 (PRL 130, 071002)"],
        "f_range_Hz": (1.0905e9, 1.1099e9),
        "cavity": {"B_T": 10.31, "V_L": 36.85, "C": 0.6, "Q0": 1.10e5, "beta": 2.0},
        "receiver": {"T_sys_K": 0.215, "eta_snr": 0.88},
        "analysis": {"mode": "scan", "rate_Hz_s": 10e3 / 500.0, "snr_target": 5.0, "confidence": 0.90,
                     "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "threshold"},
        "published": {"f_Hz": 1.10e9, "x_KSVZ": 0.92 * 0.36 / 0.97, "Lambda_MeV": 75.498,
                      "statement": "excluded g_agg down to about 6.2e-16 GeV^-1 over 4.51-4.59 ueV at 90% CL",
                      "source": "FIGURE-READ 2210.10961 Fig. 4 |g/g_DFSZ| ~ 0.87-1.05, typical 0.92 (best 6.2e-16 in text)"},
        "sources": {
            "B_T": "2210.10961 p.1 B_rms = 10.31 T over the cavity", "V_L": "2210.10961 p.1 36.85 L",
            "C": "2210.10961 p.2 ~0.6 (FEM)", "Q0": "2210.10961 p.2 Q0 107,000-114,000 (Q_L 35-38k)",
            "beta": "2210.10961 p.2 beta 1.8-2.2",
            "T_sys_K": "2210.10961 p.3 'extracted Tn at the cavity mode is about 215 mK' (App. B: ~140 mK)",
            "eta_snr": "2210.10961 p.5 eps_SNR = 88%",
            "rate_Hz_s": "2210.10961 p.3: 1996 steps of 10 kHz, ~500 s per step",
            "snr_target": "2210.10961 p.4: 3.718 sigma threshold, 90% one-sided limit at expected 5 sigma",
            "rho": "2210.10961 p.2 0.45 GeV/cm^3; SHM v_rms 270, Earth 230 km/s",
        },
        "notes": ["T_n quoted twice (215 mK p.3, 140 mK App. B): g ~ sqrt(T), a 24% spread in g.",
                  "Lambda_MeV = 75.498: CAPP's Eq. 1 normalisation (22.51 yW) implies g = alpha g_gamma/(pi f_a) "
                  "with m_a f_a = 5.70e-3 GeV^2 = Lambda^2 (INFERRED; 77.6 MeV gives 20.1 yW, -11%)."],
    },
    # ----------------------------------------------------------------------------------- TASEH
    "TASEH_2022": {
        "experiment": "TASEH", "run": "CD102 (2021)",
        "refs": ["arXiv:2205.05574 (PRL 129, 111802)", "arXiv:2205.01477 (design and operations)"],
        "f_range_Hz": (4.70750e9, 4.79815e9),
        "cavity": {"B_T": 8.0, "V_L": 0.234, "C": 0.605, "Q0": 6.07e4, "beta": 2.0},
        "receiver": {"T_sys_K": 2.2},
        "analysis": {"mode": "scan", "rate_Hz_s": 105e3 / (37 * 60.0), "snr_target": 5.0, "confidence": 0.95,
                     "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "threshold"},
        "published": {"f_Hz": 4.75e9, "g_GeV": 8.2e-14,
                      "statement": "|g_agg| >~ 8.2e-14 GeV^-1 at 95% C.L., a factor of eleven above KSVZ "
                                   "(per-bin 5.3-8.9e-14, average 8.2e-14)",
                      "source": "2205.05574 p.3-4 (text)"},
        "sources": {
            "B_T": "2205.05574 p.1 B0 = 8 T (central/nominal)", "V_L": "2205.05574 p.1 0.234 L",
            "C": "2205.05574 p.3 C = 0.60-0.61 (design paper: 0.62-0.63)",
            "Q0": "2205.05574 p.3 Q0 ~ 60,700", "beta": "2205.05574 beta ~ 2",
            "T_sys_K": "2205.01477 abstract 'Tsys ~ 2.2 K' (HEMT; 2.1-2.4 K)",
            "rate_Hz_s": "2205.05574 p.3: 837 steps of ~105 kHz, 32-42 min each (ASSUMED 37 min)",
            "snr_target": "2205.05574 p.3: candidates above 3.355 sigma, limit at SNR 5, 95% CL",
            "eta_snr": "SG-filter efficiency NOT FOUND (deferred to a companion paper); 1 used",
        },
        "notes": ["B = 8 T is the nominal central field; the design paper uses 7.8 T for its scan-rate study."],
    },
    # ------------------------------------------------------------------------------------ QUAX
    "QUAX_2019": {
        "experiment": "QUAX-a-gamma", "run": "NbTi superconducting cavity (2019)",
        "refs": ["arXiv:1903.06547 (PRD 99, 101101(R))"],
        "f_range_Hz": (9.067e9 - 22.5e3, 9.067e9 + 22.5e3),
        "cavity": {"B_T": 2.0, "V_L": 0.03643, "C": 0.589, "Q0": 4.02e5, "beta": 1.0},
        "receiver": {"T_sys_K": 15.3},
        "analysis": {"mode": "dwell", "t_s": 20 * 60.0, "snr_target": 1.64, "confidence": 0.95,
                     "rho_GeV_cm3": 0.45, "Q_a": 9.067e9 / 7812.5, "limit_type": "likelihood"},
        "published": {"f_Hz": 9.067e9, "g_GeV": 1.03e-12,
                      "statement": "g_agg < 1.03e-12 GeV^-1 in a 45 kHz band around m_a ~ 37.5 ueV (95% single-sided)",
                      "source": "1903.06547 Sec. III (text)"},
        "sources": {
            "B_T": "1903.06547 abstract, Sec. II B 2 T", "V_L": "1903.06547 Sec. II A 36.43 cm^3 (HFSS)",
            "C": "1903.06547 Sec. II A C010 = 0.589", "Q0": "1903.06547 Sec. III Q0 = 4.02e5 at 2 T, Q_L = 2.01e5",
            "beta": "critically coupled (INFERRED from Q_L = Q0/2)", "T_sys_K": "1903.06547 Sec. III T_S = 15.3 K",
            "t_s": "1903.06547 Sec. III Delta t = 20 min",
            "Q_a": "single 7812.5 Hz bin, no line-shape model (Sec. III): Delta f_eff = 7812.5 Hz",
            "snr_target": "1903.06547 Sec. III '95% single sided confidence limit (1.64 sigma)'",
        },
        "notes": ["Measured sigma_P = 6.19e-22 W vs Dicke 5.5e-22 W (Eq. 4): the paper's limit uses the "
                  "measured value, ~6% higher in g than a pure-Dicke reconstruction.",
                  "Lambda = 78 MeV in the paper (77.6 here): irrelevant for a g_GeV limit."],
    },
    "QUAX_2021": {
        "experiment": "QUAX-a-gamma", "run": "Cu cavity + JPA (2021)",
        "refs": ["arXiv:2012.09498 (PRD 103, 102004)"],
        "f_range_Hz": (10.40176e9 - 0.5e6, 10.40176e9 + 0.5e6),
        "cavity": {"B_T": 8.1, "V_L": 0.08056, "C": 0.69, "Q0": 7.6e4, "Q_L": 3.6e4, "beta": 76.0 / 36.0 - 1.0},
        "receiver": {"T_sys_K": 0.99 * 6.41 / 5.38},
        "analysis": {"mode": "dwell", "t_s": 4203.0, "snr_target": 1.2816, "confidence": 0.90,
                     "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "likelihood"},
        "published": {"f_Hz": 10.40176e9, "g_GeV": 0.766e-13,
                      "statement": "g_agg^CL < 0.766e-13 GeV^-1 at 90% C.L. (power-constrained)",
                      "source": "2012.09498 abstract, Sec. III (text)"},
        "sources": {
            "B_T": "2012.09498 Sec. II 8.1 T", "V_L": "2012.09498 Sec. II 80.56 cm^3",
            "C": "2012.09498 Sec. I 'about 0.69' (generic TM010 value)",
            "Q0": "2012.09498 Sec. II Q0 = 76,000, Q_L ~ 36,000 (critically coupled)",
            "beta": "INFERRED beta = Q0/Q_L - 1 = 1.11 ('critically coupled', Sec. II)",
            "T_sys_K": "INFERRED sigma_max-equivalent: T_n = 0.99 K (Sec. III) x sigma_max/sigma_Dicke = 6.41/5.38 "
                       "(the limit uses 'the most conservative noise power' sigma_max, 2012.09498 p.5) = 1.18 K",
            "t_s": "2012.09498 Sec. III Delta t = 4203 s",
            "snr_target": "90% CL Gaussian ML, power-constrained; ASSUMED zero-excess z = 1.2816",
        },
        "notes": ["The paper's limit uses sigma_max = 6.41e-24 W instead of the Dicke 5.38e-24 W; modelled through "
                  "T_sys_K (the Dicke T_n alone gives a 9% lower g).",
                  "Here Q_L ~ 0.05 Q_a: the SHM line is resolved by the cavity at the 5% level."],
    },
    # ----------------------------------------------------------------------------------- ORGAN
    "ORGAN_Q_2025": {
        "experiment": "ORGAN", "run": "ORGAN-Q (Phase 1, JPA)",
        "refs": ["arXiv:2407.18586"],
        "f_range_Hz": (6.15e9, 6.35e9),
        "cavity": {"B_T": 6.26, "V_L": 0.708, "C": 0.42, "Q_L": 1700.0, "beta": 1.45},
        "receiver": {"T_sys_K": 0.928, "eta_snr": 0.80 * 0.90},
        "analysis": {"mode": "scan", "rate_Hz_s": 200e6 / (489 * 45 * 60.0), "snr_target": 8.33,
                     "confidence": 0.95, "rho_GeV_cm3": 0.45, "Q_a": None, "limit_type": "threshold"},
        "published": {"f_Hz": 6.25e9, "g_GeV": 2.8e-13,
                      "statement": "excluding |g_agg| >~ 2.8e-13 at a 95% confidence level (25.45-26.27 ueV)",
                      "source": "2407.18586 abstract, Sec. III (text; units omitted, GeV^-1 implied)"},
        "sources": {
            "B_T": "2407.18586 Table I average B 6.26 T", "V_L": "2407.18586 Table I 0.708 L",
            "C": "2407.18586 Table I 0.42", "Q_L": "2407.18586 Table I Q_L ~ 1700 (sigma 400)",
            "beta": "2407.18586 Table I beta 1.45 (sigma 0.32)", "T_sys_K": "2407.18586 Table I T_S = 928 mK",
            "eta_snr": "2407.18586 Sec. II eta_SNR = eta_SG/xi_g = 0.80, times 0.90: the 9-bin Maxwellian grand-spectrum "
                       "filter 'retaining ~90% of the axion signal' (same paper; signal power at fixed noise, so on SNR)",
            "rate_Hz_s": "2407.18586: 489 frequencies, tau ~ 30-60 min (ASSUMED 45 min), ~17 days; 489 x 45 min = "
                         "15.3 d, so the mean tau is <= 50 min and 45 min already implies ~10% dead time",
            "snr_target": "2407.18586: max excess 6.68 sigma -> SNR target 8.33 sigma at 95%",
        },
        "notes": ["Q_L is low (hot rod); C, V, Q_L have 5-25% run-to-run spreads (Table I sigmas)."],
    },
}

REPRODUCIBLE = [k for k, v in PRESETS.items() if v["analysis"].get("rate_Hz_s", 1) is not None]
