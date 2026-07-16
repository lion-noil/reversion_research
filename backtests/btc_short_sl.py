# -*- coding: utf-8 -*-
"""숏 2방향 — SL을 독립 파라미터로 스윕 (SL거리 = TP거리 × m, m=0.5~3, none).
창22h, 쿨다운3h, fee 0.11%. 손절우선. 최적 셀 분기별까지."""
import sys, os
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011; W1=1320
m1=_load_cache('BTCUSDT','1')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
cs=np.concatenate([[0.],np.cumsum(C1)]); cs2=np.concatenate([[0.],np.cumsum(C1*C1)])
ma=np.full(n1,np.nan); sd=np.full(n1,np.nan)
s=cs[W1:]-cs[:-W1]; s2=cs2[W1:]-cs2[:-W1]; mm=s/W1
ma[W1-1:]=mm; sd[W1-1:]=np.sqrt(np.maximum(s2/W1-mm*mm,0))
z=np.where(sd>0,(C1-ma)/sd,np.nan)

def sim(DIR,k1,b,mh_d,slm):
    """DIR: trend(추세숏, z<=-K 진입) / short(역추숏, z>=+K 진입). slm=None→SL없음."""
    entry_pos = DIR=='short'
    cand=np.flatnonzero(z>=k1) if entry_pos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=W1]
    mh=int(mh_d*DAY); out=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<3*H: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+mh,'right'))
        if hi<=ci+1: continue
        if DIR=='trend':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct)
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price
        TP=L1[ci+1:hi]<=tp
        if slm is None: SLh=np.zeros(hi-ci-1,bool); sl=None
        else:
            sl=price*(1+pct*slm); SLh=H1[ci+1:hi]>=sl
        hit=SLh|TP
        if hit.any():
            j=int(np.argmax(hit)); ex=ci+1+j
            xp=sl if SLh[j] else tp
        else: ex=hi-1; xp=C1[ex]
        out.append((int(ts1[ci]),(price-xp)/price-FEE)); last=ts1[ci]
    return out

def agg(tr):
    a=np.array([r for _,r in tr])
    if len(a)<40: return None
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    return len(a),a.mean()*100,pf,100*(a>0).mean(),a.min()*100
def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'

for DIR,nm in [('short','역추숏'),('trend','추세숏')]:
    best=None
    for k1 in [4.0,4.5,5.0,5.5,6.0]:
        for b in [-3.0,-2.5,-2.0,-1.5,-1.0,0.0,1.0]:
            for mh in [7,14]:
                for slm in [0.5,1.0,1.5,2.0,3.0,None]:
                    tr=sim(DIR,k1,b,mh,slm)
                    g=agg(tr)
                    if g and (best is None or g[1]>best[0][1]): best=(g,k1,b,mh,slm,tr)
    g,k1,b,mh,slm,tr=best
    n,ex,pf,wr,mn=g
    print(f'===== {nm} 최적(SL 독립스윕): K{k1}/B{b}/보유{mh}d/SL={slm if slm else "없음"}×TP거리 =====')
    print(f'  n{n}({n/months:.1f}/월) 기대{ex:+.3f}% PF{pf:.2f} 승{wr:.0f}% 최악{mn:+.1f}%')
    # SL 반응 곡선 (최적 K/B/hold 고정)
    print(f'  SL배수 반응: ',end='')
    for slm2 in [0.5,1.0,1.5,2.0,3.0,None]:
        g2=agg(sim(DIR,k1,b,mh,slm2))
        if g2: print(f'[{slm2 if slm2 else "무"}] {g2[1]:+.2f}%/PF{g2[2]:.2f}/최악{g2[4]:+.0f}%',end='  ')
    print()
    byq={}
    for t,r in tr: byq.setdefault(qof(t),[]).append(r)
    print(f'  분기별:')
    for k in sorted(byq):
        a=np.array(byq[k])
        print(f'    {k}: n{len(a):>3} 합{a.sum()*100:>+7.1f}% 승{100*(a>0).mean():>3.0f}% 최악{a.min()*100:>+6.1f}%')
    print()
