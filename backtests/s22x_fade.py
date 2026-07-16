# -*- coding: utf-8 -*-
"""S22 확장판 — 급락페이드 13.5y 스크리닝 (15심볼, 4h봉). 시간청산/되돌림×1.5.
필터: n≥30·기대≥0.4%(FX 0.15%)·마이너스연도(<-3%p, FX<-1.5%p)≤2·최악>-30%p·연≥3건."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000
NONFX=['JP225','US100','GER40','UK100','HK50','XAGUSD','WTI','XAUUSD']
FX=['USDJPY','EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','NZDUSD']
def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
res=defaultdict(list)
for sym in NONFX+FX:
    m=_load_cache(sym,'240')
    isfx=sym in FX; fee=0.0002 if isfx else 0.0011
    emin=0.15 if isfx else 0.4; ybad=-1.5 if isfx else -3.0
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    ts=np.array([x['start'] for x in m],np.int64); n=len(C)
    yt=(int(ts[-1])-int(ts[0]))/(365.25*DAY)
    for M in [6,12,18,30]:
        ret=C/np.roll(C,M)-1; ret[:M]=np.nan
        thrs=[0.01,0.015,0.02,0.03] if isfx else [0.03,0.05,0.07,0.10]
        for thr in thrs:
            cand=np.flatnonzero(ret<=-thr)
            if len(cand)<30: continue
            for kind,cap in [('T',6),('T',30),('T',60),('R',30),('R',60)]:
                out=[]; last=-10**9
                for ci in cand:
                    if ci+1>=n: continue
                    if ci-last<6: continue
                    price=C[ci]; hi=min(ci+1+cap,n)
                    if kind=='R':
                        tp=price*(1+1.5*abs(ret[ci])); TP=Hh[ci+1:hi]>=tp
                        r=(tp/price-1-fee) if TP.any() else C[hi-1]/price-1-fee
                    else:
                        r=C[hi-1]/price-1-fee
                    out.append((int(ts[ci]),r)); last=ci
                if len(out)<30: continue
                a=np.array([r for _,r in out]); w=a>0
                exp=a.mean()*100
                if exp<emin: continue
                ysum=defaultdict(float)
                for en,r in out: ysum[yr(en)]+=r*100
                bad=[y for y,s in ysum.items() if s<ybad]; worst=min(ysum.values())
                if len(bad)>2 or worst<-30 or len(out)/yt<3: continue
                pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
                res[sym].append((f'fade {M*4}h≤-{thr*100:.1f}% {kind}{cap*4//24}d',
                    dict(n=len(out),exp=exp,pf=pf,wr=100*w.mean(),mo=len(out)/yt/12,score=exp*min(pf,5),
                         ys=' '.join(f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))))
    print(f'{sym} done ({len(res[sym])})',flush=True)
print()
for sym in NONFX+FX:
    rs=sorted(res.get(sym,[]),key=lambda x:-x[1]['score'])[:2]
    for name,e in rs:
        print(f'{sym:7} {name:24} n={e["n"]:>4} {e["exp"]:>+6.2f}% PF{e["pf"]:>5.2f} 승{e["wr"]:>3.0f}% 월{e["mo"]:.1f} | {e["ys"]}')
