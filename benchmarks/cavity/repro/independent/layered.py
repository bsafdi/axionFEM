import numpy as np
from scipy.special import j0,y0,j1,y1
from scipy.optimize import brentq
from scipy.integrate import quad
c=299792458.0
def ez_end(k0, layers, R):
    # layers: list of (r_outer, eps) from axis; start E=J0(k r) in first layer
    r0=0; a,b=1.0,0.0
    E=lambda r,k,a,b: a*j0(k*r)+b*y0(k*r)
    dE=lambda r,k,a,b: -k*(a*j1(k*r)+b*y1(k*r))
    k=k0*np.sqrt(layers[0][1])
    for (ro,eps),(ro2,eps2) in zip(layers[:-1],layers[1:]):
        e,de=E(ro,k,a,b),dE(ro,k,a,b)
        k2=k0*np.sqrt(eps2)
        M=np.array([[j0(k2*ro),y0(k2*ro)],[-k2*j1(k2*ro),-k2*y1(k2*ro)]])
        a,b=np.linalg.solve(M,[e,de]); k=k2
    return E(R,k,a,b)
def modes(layers,R,fmin,fmax,n=20000):
    fs=np.linspace(fmin,fmax,n); v=[ez_end(2*np.pi*f/c,layers,R) for f in fs]
    return [brentq(lambda f: ez_end(2*np.pi*f/c,layers,R),fs[i],fs[i+1],xtol=1) for i in range(n-1) if v[i]*v[i+1]<0]
MM=1e-3
cu,sa=3.26e-3,8e-4
R=28.55*(1-cu)*MM
R1,t1,R2,t2=[x*(1-sa)*MM for x in (10.875,2.35,20.90,1.94)]
for eps in (11.2,11.44):
    L=[(R1,1),(R1+t1,eps),(R2,1),(R2+t2,eps),(R,1)]
    print('2201 4.2K eps',eps,[round(f/1e9,6) for f in modes(L,R,9e9,11e9)])
# 293 K
L=[(10.875*MM,1),(13.225*MM,11.44),(20.9*MM,1),(22.84*MM,11.44),(28.55*MM,1)]
print('2201 293K',[round(f/1e9,6) for f in modes(L,28.55*MM,9e9,11e9)])
# 2004 flat
L=[(10.7*MM,1),(12.6*MM,11.2),(19.8*MM,1),(21.4*MM,11.2),(29*MM,1)]
print('2004',[round(f/1e9,6) for f in modes(L,29*MM,9e9,11.5e9)])
# clamshell no gap
L=[(10.866*MM,1),((10.866+2.348)*MM,11.2),(28.475*MM,1)]
print('clam',[round(f/1e9,6) for f in modes(L,28.475*MM,9e9,11.5e9)])
