# -*- coding: utf-8 -*-
"""1분봉 창 스윕 승자 셀 검증 — 보유상한 7일 vs 14일 비교. 단일 조합 재계산이라 빠름."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000

def prep(sym,win):
    m=_load_cache(sym,'1')
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64); n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    z=np.where(sd>0,(C-ma)/sd,np.nan)
    return dict(C=C,H=Hh,L=L,ts=ts,ma=ma,sd=sd,z=z,n=n,win=win)

def gen(d,DIR,k1,b,cd_ms,mh_ms):
    C,Hh,L,ts,ma,sd,z,n,win=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n'],d['win']
    cand=np.flatnonzero(z>=k1) if DIR in('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=win]; rets=[]; iv=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+mh_ms,'right'))
        if hi<=ci+1: continue
        if DIR=='trendlong':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tx/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        elif DIR=='trend':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; tx=price*(1-pct); sx=price*(1+pct)
            SL=Hh[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tx)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        elif DIR=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tp/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=Hh[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tp)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        rets.append(r); iv.append((int(ts[ci]),int(ts[ex]))); last=ts[ci]
    ev=[]
    for en,ex in iv: ev.append((en,1)); ev.append((ex,-1))
    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
    for _,dd in ev: cur+=dd; mx=max(mx,cur)
    return rets,mx

def cdms(s):
    s=str(s)
    return float(s[:-1])*60000 if s.endswith('m') else float(s[:-1])*H

CELLS=[  # (sym, dir, win, k1, b, cd)
 ('BTCUSDT','trendlong',20160,3.8,-2.0,'2h'),
 ('ETHUSDT','trendlong',20160,3.8,-1.8,'1.75h'),
 ('XRPUSDT','trendlong',30240,4.1,-2.0,'1.75h'),
 ('SOLUSDT','trendlong',10080,3.6,-2.0,'3h'),
 ('SOLUSDT','trend',10080,4.0,-2.0,'30m'),
 ('ETHUSDT','trend',4320,4.4,0.4,'2h'),
 ('BTCUSDT','trend',4320,3.9,-2.0,'2.75h'),
 ('BTCUSDT','long',20160,3.3,-0.8,'2.75h'),
 ('ETHUSDT','long',7200,4.5,-2.0,'30m'),
 ('ETHUSDT','long',30240,3.6,1.8,'2.5h'),
 ('SOLUSDT','long',30240,3.2,1.8,'2.5h'),
 ('XRPUSDT','long',30240,3.6,-1.6,'1.25h'),
 ('BTCUSDT','short',10080,4.2,-0.6,'1h'),
 ('ETHUSDT','short',5760,4.4,0.4,'30m'),
 ('XRPUSDT','short',8640,4.7,-1.8,'45m'),
]
def st(r):
    a=np.array(r)
    if len(a)==0: return '없음'
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    return a,pf

def main():
    cache={}
    print(f'{"심볼":9}{"방향":10}{"창":>7} | {"보유7d":^28} | {"보유14d":^28}')
    for sym,DIR,win,k1,b,cd in CELLS:
        key=(sym,win)
        if key not in cache: cache[key]=prep(sym,win)
        d=cache[key]
        cols=[]
        for mh in [7,14]:
            rets,mx=gen(d,DIR,k1,b,cdms(cd),mh*DAY)
            a=np.array(rets); pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
            cols.append(f'n{len(a):>4} {a.mean()*100:>+6.2f}%/PF{pf:>5.2f} mc{mx:>2}')
        print(f'{sym:9}{DIR:10}W{win:>6} | {cols[0]} | {cols[1]}',flush=True)

if __name__=='__main__':
    main()
