import numpy as np
from axq1 import graded, solve
IN=0.0254; R,H=2*IN,10*IN
for h in (2e-3,1e-3):
    out,info=solve(R,0,H/2,[],graded([0,R],h,h),graded([0,H/2],h,h),2*np.pi*2.2587e9/299792458,nev=2)
    print(info['ndof'],[(o['f_Hz']/1e9,o['C']) for o in out])
