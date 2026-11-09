import numpy as np
from axq1 import graded, solve
IN=0.0254; R,H,A=2*IN,10*IN,1*IN
for g,hmin,hmax in ((0.254e-3,0.016e-3,0.5e-3),(0.254e-3,0.008e-3,0.25e-3),(0,0.016e-3,0.5e-3)):
    zb=[0,H/2]+([H/2-g] if g>0 else [])
    rg=graded([0,A,R],hmin,hmax,0.12,fine=[A]); zg=graded(zb,hmin,hmax,0.12,fine=[H/2-g,H/2] if g>0 else [H/2])
    out,info=solve(R,0,H/2,[(0,A,-1,H/2-g if g>0 else 1,'metal')],rg,zg,2*np.pi*5.86e9/299792458,nev=2,sigma=5.8e7)
    o=max(out,key=lambda o:o['C']); print(g,hmin,info['ndof'],o['f_Hz']/1e9,o['C'],o['Q'])
out,info=solve(R,0,H/2,[],graded([0,R],1e-3,1e-3),graded([0,H/2],1e-3,1e-3),2*np.pi*2.2587e9/299792458,nev=1,sigma=5.8e7)
print('empty',out[0]['f_Hz'],out[0]['Q'])
