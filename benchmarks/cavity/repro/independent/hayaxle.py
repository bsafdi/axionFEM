import numpy as np
from axq1 import graded, solve
IN=0.0254; R,H,A=2*IN,10*IN,1*IN; g=0.254e-3
ae=1+(np.pi*(0.125*IN)**2/(np.pi*((0.6*IN)**2-(0.35*IN)**2)))*8.8
print('eps_eff',ae)
for hmin,hmax in ((0.016e-3,0.5e-3),(0.008e-3,0.25e-3)):
    rg=graded([0,0.35*IN,A,0.6*IN,R],hmin,hmax,0.12,fine=[A]); zg=graded([0,H/2-g,H/2],hmin,hmax,0.12,fine=[H/2-g,H/2])
    for eps in (None,ae):
        rects=[(0,A,-1,H/2-g,'metal')]+([(0.35*IN,0.6*IN,H/2-g,H/2+1,eps)] if eps else [])
        out,info=solve(R,0,H/2,rects,rg,zg,2*np.pi*5.86e9/299792458,nev=3)
        o=max(out,key=lambda o:o['C']); print(hmin,eps,info['ndof'],o['f_Hz']/1e9,o['C'])
