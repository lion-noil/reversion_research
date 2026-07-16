# -*- coding: utf-8 -*-
"""US100 숏 진단 — 필터 해제, 패밀리별 최상위 기대값 관찰 (엣지 유무 판정용)."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000
m=_load_cache('US100','1')
C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64)
n=len(C)
def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
def zma(x,win):
    N=len(x); cs=np.concatenate([[0.],np.cumsum(x)]); cs2=np.concatenate([[0.],np.cumsum(x*x)])
    ma=np.full(N,np.nan); sd=np.full(N,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(x-ma)/sd,np.nan),ma,sd
def ema(x,s):
    a=2/(s+1); out=np.copy(x)
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out
def sim_short_tp(cand,tp_arr,mh_d,cd_ms):
    out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+mh_d*DAY,'right'))
        if hi<=ci+1: continue
        tp=tp_arr[ci]
        if not np.isfinite(tp) or tp>=price: continue
        TP=L[ci+1:hi]<=tp
        xp=tp if TP.any() else C[hi-1]
        out.append((int(ts[ci]),(price-xp)/price-FEE)); last=ts[ci]
    return out
def sim_short_time(cand,hold_min,cd_ms):
    out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        ex=int(np.searchsorted(ts,ts[ci]+hold_min*60000,'right'))-1
        if ex<=ci: continue
        out.append((int(ts[ci]),(C[ci]-C[ex])/C[ci]-FEE)); last=ts[ci]
    return out
def eval_raw(tr):
    if len(tr)<30: return None
    a=np.array([r for _,r in tr]); w=a>0
    pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
    ysum=defaultdict(float)
    for en,r in tr: ysum[yr(en)]+=r*100
    return dict(n=len(tr),exp=a.mean()*100,pf=pf,
                ys=' '.join(f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))
best={}
def keep(fam,name,tr):
    e=eval_raw(tr)
    if e and (fam not in best or e['exp']>best[fam][1]['exp']): best[fam]=(name,e)
for win in [720,1440,2880]:
    z,ma,sd=zma(C,win)
    for K in [3.5,4.0,4.5,5.0]:
        cand=np.flatnonzero(z>=K); cand=cand[cand>=win]
        if len(cand)<30: continue
        for B in [-1,0,1]:
            keep('z역숏',f'z역숏 w{win//60}h K{K} B{B} nosl',sim_short_tp(cand,ma+B*sd,14,3*H))
for M in [30,60,120,240]:
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    for r0 in [0.01,0.015,0.02,0.03]:
        cand=np.flatnonzero(ret>=r0)
        if len(cand)<30: continue
        for hold in [240,720,1440]:
            keep('급등페이드숏',f'급등페이드 {M}m≥+{r0*100:.1f}% T{hold//60}h',sim_short_time(cand,hold,30*60000))
for s in [720,1440,2880]:
    mu=ema(C,s); resid=C-mu; sig=ema(np.abs(resid),s)
    zz=resid/np.where(sig==0,1e-12,sig)
    for k in [2.5,3.0]:
        cand=np.flatnonzero(zz<=-k)
        if len(cand)<30: continue
        for hold in [360,720,1440]:
            keep('ewz추숏',f'ewz추숏 s{s//60}h k{k} T{hold//60}h',sim_short_time(cand,hold,6*3600000))
print(f'{"패밀리 최상위(필터해제)":32} {"n":>5} {"기대%":>7} {"PF":>5} | 연도별')
for fam,(name,e) in best.items():
    print(f'{name:36} {e["n"]:>5} {e["exp"]:>+7.3f} {e["pf"]:>5.2f} | {e["ys"]}')
