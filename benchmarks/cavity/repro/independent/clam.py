import sys, numpy as np, time
from axq1 import graded, solve
MM=1e-3
rc,Lc,r1,t,eps=28.475*MM,420.39*MM,10.866*MM,2.348*MM,11.2
def run(gap, hr_min, hr_max, hz_max, hz_min=None):
    Lt=Lc-gap; z0=0; zt=Lt; z1=Lc
    hz_min=hz_min or hr_min
    rg=graded([0,r1,r1+t,rc],hr_min,hr_max,0.15,fine=[r1,r1+t])
    zb=[0,Lc]+([zt] if gap>0 else [])
    zg=graded(zb,hz_min,hz_max,0.15,fine=zb)
    rects=[(r1,r1+t,-1,zt,eps)]
    T=time.time(); out,info=solve(rc,0,Lc,rects,rg,zg,2*np.pi*11.04e9/299792458,nev=int(__import__("os").environ.get("NEV","8")),sym=False,sigma=3.18e8)
    o=max(out,key=lambda o:o['C'])
    print(f"gap={gap/MM:.2f} hr={hr_min/MM:.3f}-{hr_max/MM:.3f} hz={hz_min/MM:.3f}-{hz_max/MM:.2f} ndof={info['ndof']} t={time.time()-T:.0f}s f={o['f_Hz']/1e9:.6f} C={o['C']:.5f} Q={o['Q']:.0f}  others={[ (round(x['f_Hz']/1e9,4),round(x['C'],4)) for x in out]}",flush=True)
for a in sys.argv[2:]:
    p=[float(x)*MM for x in a.split(',')]
    run(float(sys.argv[1])*MM,*p)
