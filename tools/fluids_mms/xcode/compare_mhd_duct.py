"""Compare the NGSolve F13a cross-code reference with the FEniCSx re-computation (relative differences)."""
import json
import sys

a = json.load(open(sys.argv[1]))["quantities"]
b = json.load(open(sys.argv[2]))["quantities"]
worst = 0.0
for k in sorted(a):
    d = abs(a[k] - b[k]) / max(abs(a[k]), 1e-300)
    worst = max(worst, d)
    print(f"{k:40s} NGSolve {a[k]: .15e}  FEniCSx {b[k]: .15e}  rel {d:.1e}")
print(f"max relative difference {worst:.2e}")
