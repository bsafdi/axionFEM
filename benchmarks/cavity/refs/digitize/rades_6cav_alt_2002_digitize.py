"""Digitize RADES 6-subcavity alternating filter figures (arXiv:2002.07639).

Usage: python3 -I rades_6cav_alt_2002_digitize.py /path/to/2002.07639.pdf
Writes REFS/rades_6cav_alt_2002_fig*.png (+ _overlay.png) and updates the
"reference_values.curves" / "measured_values.curves" entries of REFS/rades_6cav_alt_2002.json.
All figures are MATLAB vector graphics: points are the PDF path vertices.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402  (numpy before fitz: libstdc++ load order)
import fitz  # noqa: E402

import vector_digitize as vd  # noqa: E402

REFS = os.path.dirname(HERE)
NAME = "rades_6cav_alt_2002"


def red_square(page, frame):
    for dr in page.get_drawings():
        if dr.get("fill") and max(abs(a - b) for a, b in zip(dr["fill"], (1, 0, 0))) < 0.05:
            r = dr["rect"]
            if frame.contains(r) and r.width < 6:
                return ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
    return None


def do_fig(doc, pg, frame, tag, specs, xlabel, ylabel, desc, nmax=300, use_grid=False, marker=False, peaks=False, pad=(42, 12, 12, 26)):
    page = doc[pg]
    frame = fitz.Rect(*frame)
    cal = vd.axis_calibration(page, frame, use_grid=use_grid)
    series = []
    extra = {}
    for label, color, dashed in specs:
        drs = vd.curves_in(page, frame, color)
        drs = [d for d in drs if bool(d.get("dashes") and d["dashes"].strip().startswith("[ ") and d["dashes"].strip() != "[] 0") == dashed]
        assert len(drs) == 1, (tag, label, len(drs))
        xy = vd.to_data(cal, vd.polyline(drs[0]))
        xy = xy[np.argsort(xy[:, 0], kind="stable")]
        noisy = label.startswith("Measurements")
        series.append((label, vd.thin(xy, 800 if noisy else nmax, keep_extrema=not noisy)))
        if peaks and not noisy:
            y = xy[:, 1]
            x = xy[:, 0]
            pk = []
            for i in range(1, len(y) - 1):
                w = (x > x[i] - 0.005) & (x < x[i] + 0.005)
                if y[i] > -35 and y[i] >= y[w].max() and y[i] > y[w].min() + 3 and (not pk or x[i] - x[pk[-1]] > 1e-3):
                    pk.append(i)
            extra.setdefault("derived_peaks", []).append(
                {"series": label, "peak_frequencies": [round(float(xy[i, 0]), 4) for i in pk],
                 "peak_levels_dB": [round(float(y[i]), 2) for i in pk],
                 "note": "local maxima of the raw vector polyline: > -35 dB, max within +-5 MHz, >= 3 dB prominence"})
    if marker:
        m = red_square(page, frame)
        extra["chosen_point_marker"] = [float(v) for v in vd.to_data(cal, [m])[0]]
    clip = fitz.Rect(frame.x0 - pad[0], frame.y0 - pad[1], frame.x1 + pad[2], frame.y1 + pad[3])
    png = f"{NAME}_{tag}.png"
    ovl = f"{NAME}_{tag}_overlay.png"
    vd.render_crop(page, clip, os.path.join(REFS, png), dpi=250)
    vd.overlay(page, clip, cal, [(l, p[:: max(1, len(p) // 120)]) for l, p in series], os.path.join(REFS, ovl), dpi=250)
    ux = abs(cal["x"]["a"]) * 0.3 + cal["x"]["rms_residual"]
    uy = abs(cal["y"]["a"]) * 0.3 + cal["y"]["rms_residual"]
    return {
        "source": "Fig. " + tag[3:].replace("_left", " (left)").replace("_right", " (right)"),
        "description": desc,
        "x": xlabel,
        "y": ylabel,
        "series": [{"label": l, "points": [[round(float(a), 6), round(float(b), 6)] for a, b in p]} for l, p in series],
        **extra,
        "digitization": {
            "method": "vector paths via pymupdf get_drawings (PDF path vertices, thinned keeping local extrema)",
            "page": pg + 1,
            "clip_bbox_pt": [round(v, 2) for v in clip],
            "axis_calibration": {k: {"pt_value_pairs": cal[k]["pairs"], "rms_residual": cal[k]["rms_residual"], "log": False} for k in ("x", "y")},
            "uncertainty_x": float(f"{ux:.2g}"),
            "uncertainty_y": float(f"{uy:.2g}"),
            "uncertainty_basis": "0.3 pt (vertex rounding / line-join) converted to data units + calibration rms residual; the curve itself is exact PDF geometry",
            "figure_png": png,
            "overlay_png": ovl,
            "script": "digitize/" + os.path.basename(__file__),
        },
    }


def main(pdf):
    doc = fitz.open(pdf)
    blue, red = (0, 0, 1), (1, 0, 0)
    sim = []
    sim.append(do_fig(doc, 16, (111.197, 121.671, 277.433, 251.170), "fig5_left", [("k2 (inductive iris, t_ind = 2.0 mm), CST eigen PEC/PMC", blue, False)],
                      {"label": "w", "unit": "mm"}, {"label": "k2", "unit": "1"},
                      "Inductive-iris coupling k2 vs iris width w, t_ind = 2.0 mm (CST eigensolver, PEC/PMC symmetry plane, Eq. 3.2). Red square = chosen w.", marker=True))
    sim.append(do_fig(doc, 16, (327.921, 121.671, 494.157, 251.170), "fig5_right", [("k1 (capacitive iris, t_cap = 3.5 mm, no protrusions)", blue, False)],
                      {"label": "h", "unit": "mm"}, {"label": "k1", "unit": "1"},
                      "Capacitive-iris coupling k1 vs slot height h, t_cap = 3.5 mm, plain iris (no protrusions)."))
    sim.append(do_fig(doc, 16, (327.921, 520.991, 494.157, 650.490), "fig6_right", [("k1 (quasi-capacitive iris with g = 1.1 mm protrusions, t_cap = 3.5 mm)", blue, False)],
                      {"label": "h", "unit": "mm"}, {"label": "k1", "unit": "1"},
                      "Quasi-capacitive iris (with g = 1.1 mm protrusions) coupling k1 vs h. Red square = chosen h (text: 0.93 mm).", marker=True))
    sim.append(do_fig(doc, 20, (170.895, 103.177, 436.633, 313.196), "fig10", [("CST (2K)", blue, False), ("Ideal design (lossless circuit, Table 3)", red, True)],
                      {"label": "frequency", "unit": "GHz"}, {"label": "|S21|", "unit": "dB"},
                      "|S21| at 2 K: CST full-wave with probes, sigma_2K = 2.008e9 S/m (solid) vs ideal lossless circuit (dotted).", nmax=600, use_grid=True, peaks=True))
    meas = do_fig(doc, 22, (186.739, 100.802, 419.257, 284.566), "fig13", [("CST (as-built dims, RT)", blue, False), ("Measurements (VNA, RT)", red, True)],
                  {"label": "frequency", "unit": "GHz"}, {"label": "|S21|", "unit": "dB"},
                  "|S21| at room temperature: CST with as-built dimensions (Table 1 col. 3) and RT copper conductivity vs VNA measurement.", nmax=600, use_grid=True, peaks=True)
    jp = os.path.join(REFS, NAME + ".json")
    J = json.load(open(jp))
    J["reference_values"]["curves"] = sim
    J["measured_values"]["curves"] = [meas]
    with open(jp, "w") as f:
        json.dump(J, f, indent=1)
    for c in sim + [meas]:
        print(c["source"], [len(s["points"]) for s in c["series"]], c.get("chosen_point_marker"), c.get("derived_peaks"), c["digitization"]["uncertainty_x"], c["digitization"]["uncertainty_y"],
              c["digitization"]["axis_calibration"]["x"]["rms_residual"], c["digitization"]["axis_calibration"]["y"]["rms_residual"])


if __name__ == "__main__":
    main(sys.argv[1])
