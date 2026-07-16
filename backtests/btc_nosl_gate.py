# -*- coding: utf-8 -*-
"""SL제거(TP+만기) 모드 — ①K·B·보유 재확정(K 6까지 확장) ②최적 셀에 5분 레짐 게이트 사다리.
창22h(1분), 쿨다운 3h, fee 0.11%. 게이트=같은방향(추세롱: z5>=K5f / 역추롱: z5<=-K5f)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011; W1=1320

m1=_load_cache('BTCUSDT','1'); m5=_load_cache('BTCUSDT','5')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
C5=np.array([x['close'] for x in m5]); ts5=np.array([x['start'] for x in m5],np.int64)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
def zma(C,win):
    n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd
z,ma,sd=zma(C1,W1)
Z5={}
for w in [288,864,2016]:
    z5,_,_=zma(C5,w)
    idx=np.searchsorted(ts5+4*60000, ts1,'right')-1; idx=np.clip(idx,0,len(z5)-1)
    v=z5[idx]; v[idx<0]=np.nan; Z5[w]=v

def sim(DIR,k1,b,mh_d,gate=None):
    pos = DIR=='trendlong'
    cand=np.flatnonzero(z>=k1) if pos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=W1]
    if gate is not None:
        z5v,k5f=gate
        g=(z5v[cand]>=k5f) if pos else (z5v[cand]<=-k5f)
        cand=cand[g]
    mh=int(mh_d*DAY); rets=[]; iv=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<3*H: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+mh,'right'))
        if hi<=ci+1: continue
        if pos:
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            tp=price*(1+(1-tpb/price))
        else:
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
        TP=H1[ci+1:hi]>=tp
        if TP.any(): ex=ci+1+int(np.argmax(TP)); xp=tp
        else: ex=hi-1; xp=C1[ex]
        rets.append(xp/price-1-FEE); iv.append((int(ts1[ci]),int(ts1[ex]))); last=ts1[ci]
    ev=[]
    for en,ex_ in iv: ev.append((en,1)); ev.append((ex_,-1))
    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
    for _,dd in ev: cur+=dd; mx=max(mx,cur)
    return np.array(rets),mx

def st(a,mx=None):
    n=len(a)
    if n<30: return None
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    return f"n{n:>4}({n/months:>4.1f}/월) 기대{a.mean()*100:>+7.3f}% PF{pf:>5.2f} 승{100*(a>0).mean():>3.0f}% 최악{a.min()*100:>+6.1f}%"+(f' mc{mx}' if mx is not None else '')+f' 월합{(n/months)*a.mean()*100:>+5.1f}%p'

for DIR,nm,KS in [('trendlong','추세롱',[4.5,4.75,5.0,5.25,5.5,6.0]),('long','역추롱',[4.25,4.5,4.75,5.0,5.5,6.0])]:
    print(f'===== {nm} — ① SL제거 최적 재확정 (K 확장·보유 7/14d) =====')
    best=None
    for k1 in KS:
        for b in [-3.0,-2.5,-2.0,-1.5,-1.0]:
            for mh in [7,14]:
                a,mx=sim(DIR,k1,b,mh)
                if len(a)<40: continue
                e=a.mean()
                if best is None or e>best[0]: best=(e,k1,b,mh,a,mx)
    e,k1,b,mh,a,mx=best
    print(f'  ★최적: K{k1} B{b} 보유{mh}d → {st(a,mx)}')
    # 주변 확인
    for k in KS:
        a2,mx2=sim(DIR,k,b,mh)
        s=st(a2,mx2)
        if s: print(f'    K{k:<5}: {s}')
    print(f'  ---- ② 5분 레짐 게이트 사다리 (고정: K{k1}/B{b}/보유{mh}d) ----')
    print(f'    게이트없음: {st(a,mx)}')
    for w5 in [288,864,2016]:
        for k5f in [1.0,1.5,2.0,2.5]:
            a3,mx3=sim(DIR,k1,b,mh,gate=(Z5[w5],k5f))
            s=st(a3,mx3)
            print(f'    w5={w5//288}d K5f={k5f}: '+(s if s else 'n<30'))
    print()
