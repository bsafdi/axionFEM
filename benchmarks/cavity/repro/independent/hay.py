import sys, numpy as np, time
from axq1 import graded, solve
IN = 0.0254; R, H, A = 2*IN, 10*IN, 1*IN
def run(g, hmin, hmax, growth=0.12, kt=None):
    zb = [0, H/2] + ([H/2-g] if g > 0 else [])
    rg = graded([0, A, R], hmin, hmax, growth, fine=[A])
    zg = graded(zb, hmin, hmax, growth, fine=[H/2-g, H/2] if g > 0 else [H/2])
    rects = [(0, A, -1, H/2-g if g > 0 else 1.0, 'metal')]
    kt = kt or 2*np.pi*5.86e9/299792458
    t=time.time(); out, info = solve(R, 0, H/2, rects, rg, zg, kt, nev=6)
    return out, info, time.time()-t
if __name__ == "__main__":
    g = float(sys.argv[1])*1e-3
    for hmin, hmax in [(float(a)*1e-3, float(b)*1e-3) for a, b in (x.split(',') for x in sys.argv[2:])]:
        out, info, dt = run(g, hmin, hmax)
        print(f"g={g*1e3:.3f}mm hmin={hmin*1e3:.4f} hmax={hmax*1e3:.3f} ndof={info['ndof']} t={dt:.1f}s", flush=True)
        for o in out: print(f"   f={o['f_Hz']/1e9:.6f} GHz  C_vac={o['C']:.5f}")
