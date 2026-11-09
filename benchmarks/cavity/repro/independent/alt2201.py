from layered import modes
MM=1e-3; cu,sa=3.26e-3,8e-4
def f(R1,t1,R2,t2,R=28.55,eps=11.2):
    s=lambda x:x*(1-sa)*MM
    L=[(s(R1),1),(s(R1+t1),eps),(s(R2),1),(s(R2+t2),eps),(R*(1-cu)*MM,1)]
    return [round(x/1e9,4) for x in modes(L,R*(1-cu)*MM,10.0e9,10.9e9,4000)]
print('nominal',f(10.875,2.35,20.90,1.94))
print('tube1 ID as OD',f(10.875-2.35,2.35,20.90,1.94))
print('tube2 ID as OD',f(10.875,2.35,20.90-1.94,1.94))
print('tube2 t as 1.94 dia diff (t=0.97)',f(10.875,2.35,20.90,0.97))
print('tube1 t 2.35 as diameter diff (t=1.175)',f(10.875,1.175,20.90,1.94))
print('eps 11.0',f(10.875,2.35,20.90,1.94,eps=11.0))
print('R cav 28.5',f(10.875,2.35,20.90,1.94,R=28.5))
print('tube1 OD 26.13 (t1 2.19)',f(10.875,2.19,20.90,1.94))
