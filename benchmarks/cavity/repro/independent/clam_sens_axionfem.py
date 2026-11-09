import sys, json
sys.path.insert(0, sys.argv[1])
import _common as cm
import quax_clamshell_2410 as q
P = q.PROTO
jobs = []
for dL in (-0.2, -0.1, 0.0, 0.1, 0.2):
    jobs.append((f"sensL{dL:+.1f}", q.build("sens", P["rc"], P["Lc"] + dL, P["r1"], P["t"], P["Lt"] + dL, 2, 0.5)))
for de in (-0.02, 0.02):
    q.EPS = 11.2 + de
    jobs.append((f"senseps{de:+.2f}", q.build("sens", P["rc"], P["Lc"], P["r1"], P["t"], P["Lt"], 2, 0.5)))
q.EPS = 11.2
out = cm.solve_many(jobs, "clam_sens")
for tag, data, dt in out:
    ms = sorted(cm.modes(data), key=lambda m: m["f_Hz"])
    cl = [(round(m["f_Hz"] / 1e9, 5), round(m["form_factor"]["C"], 4)) for m in ms if abs(m["f_Hz"] - 11.045e9) < 6e6]
    best = max(ms, key=lambda m: m["form_factor"]["C"])
    print(tag, "C_max=%.4f" % best["form_factor"]["C"], "Q=%.0f" % best["losses"]["Q0"], "cluster_sum=%.4f" % sum(c for _, c in cl), cl, flush=True)
