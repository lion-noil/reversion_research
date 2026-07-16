# -*- coding: utf-8 -*-
"""시간청산 패밀리 — 숏 미러 탐색. 하락 레짐 게이트(z5<=-K5f) + 1분 진입 + T시간 후 청산.
진입 2형: plunge(z1<=-K 순방향 숏) / bounce(z1>=+K 베어랠리 페이드 숏). 6.3년, fee 0.11%."""
import sys, os, time
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011
m1=_load_cache('BTCUSDT','1'); m5=_load_cache('BTCUSDT','5')
C1=np.array([x['close'] for x in m1]); ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
C5=np.array([x['close'] for x in m5]); ts5=np.array([x['start'] for x in m5],np.int64)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
def zma(C,win):
    n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan)
Z1={w:zma(C1,w) for w in [360,720,1320]}
idx=np.searchsorted(ts5+4*60000, ts1,'right')-1; idx=np.clip(idx,0,len(ts5)-1)
Z5={}
for wg in [576,2016,4032]:
    z5=zma(C5,wg); v=z5[idx].copy(); v[idx<0]=np.nan; Z5[wg]=v
def yof(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def sim(etype,w1,k1,cdh,Th,wg,k5f):
    z=Z1[w1]
    cand=np.flatnonzero(z<=-k1) if etype=='plunge' else np.flatnonzero(z>=k1)
    cand=cand[cand>=w1]
    if k5f is not None:
        cand=cand[Z5[wg][cand]<=-k5f]
    sel=[]; last=None; cd=int(cdh*H)
    for ci in cand:
        if last is not None and (ts1[ci]-last)<cd: continue
        sel.append(ci); last=ts1[ci]
    sel=np.array(sel,dtype=int)
    if len(sel)<100: return None
    ex=np.searchsorted(ts1, ts1[sel]+int(Th*H),'right')-1
    ok=(ex>sel)&(ex<n1); sel,ex=sel[ok],ex[ok]
    r=-(C1[ex]/C1[sel]-1)-FEE
    return sel,ex,r

def rep(tag,res):
    if res is None: return False
    sel,ex,r=res
    if r.mean()<=0.0005: return False   # 유의미한 것만 출력
    ev=[]
    for i in range(len(sel)): ev.append((ts1[sel[i]],1)); ev.append((ts1[ex[i]],-1))
    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
    for _,d in ev: cur+=d; mx=max(mx,cur)
    pf=r[r>0].sum()/(-r[r<0].sum()) if (r<0).any() else 99
    byy={}
    for i in range(len(sel)): byy.setdefault(yof(ts1[sel[i]]),[]).append(r[i])
    ys=' '.join(f'{y%100:02d}:{np.sum(byy[y])*100:+.0f}' for y in sorted(byy))
    neg=len([y for y in byy if np.sum(byy[y])<-10/100])
    print(f'{tag}: n{len(r)}({len(r)/months:.0f}/월) 기대{r.mean()*100:+.3f}% PF{pf:.2f} mc{mx} | {ys} | 큰마이너스연도 {neg}')
    return True

t0=time.time()
for etype,enm in [('plunge','추세숏형(급락순방향)'),('bounce','역추숏형(반등페이드)')]:
    print(f'===== {enm} =====',flush=True)
    cnt=0
    for w1 in [360,720,1320]:
        for k1 in [2.0,2.5,3.0]:
            for cdh in [0.5,1,3]:
                for Th in [24,48]:
                    for wg,wn in [(576,'2d'),(2016,'7d'),(4032,'14d')]:
                        for k5f in [1.0,2.0,2.5]:
                            if rep(f'  창{w1//60}h K{k1} cd{cdh}h T{Th}h 게이트{wn}<=-{k5f}',sim(etype,w1,k1,cdh,Th,wg,k5f)): cnt+=1
    if cnt==0: print('  (기대 +0.05% 초과 셀 없음)')
    print(f'({time.time()-t0:.0f}s)',flush=True)
