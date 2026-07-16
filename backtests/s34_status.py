# -*- coding: utf-8 -*-
"""S3/S4 채택 셀(HANDOFF_MASTER §3 볼드+🆕) 현재 실적 — 전체 vs 연도별(특히 2026 YTD).
모델: 일봉 v2 (봉마감 진입, TP/SL 대칭 손절우선, 보유15일, 진입쿨다운, fee 0.11%)."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; DAY=86400*1000; MAXHOLD=15*DAY

def prep(sym,itv,win):
    m=_load_cache(sym,itv)
    if not m or len(m)<=win: return None
    C=np.array([x['close'] for x in m]); H=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64); n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)]); w=win
    ma=np.full(n,np.nan); sd=np.full(n,np.nan); s=cs[w:]-cs[:-w]; s2=cs2[w:]-cs2[:-w]; mm=s/w
    ma[w-1:]=mm; sd[w-1:]=np.sqrt(np.maximum(s2/w-mm*mm,0)); z=np.where(sd>0,(C-ma)/sd,np.nan)
    return dict(C=C,H=H,L=L,ts=ts,ma=ma,sd=sd,z=z,n=n,win=win)

def gen(d,mode,k1,b,cdd):
    C,H,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cand=np.flatnonzero(z>=k1) if mode in('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=d['win']]; out=[]; last=None; cd_ms=cdd*DAY
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+MAXHOLD,'right'))
        if hi<=ci+1: continue
        if mode=='trendlong':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=H[ci+1:hi]>=tx; hit=SL|TP
            r=((sx/price-1-FEE) if SL[int(np.argmax(hit))] else (tx/price-1-FEE)) if hit.any() else C[hi-1]/price-1-FEE
        elif mode=='trend':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; tx=price*(1-pct); sx=price*(1+pct)
            SL=H[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tx; hit=SL|TP
            r=(((price-sx)/price-FEE) if SL[int(np.argmax(hit))] else ((price-tx)/price-FEE)) if hit.any() else (price-C[hi-1])/price-FEE
        elif mode=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=H[ci+1:hi]>=tp; hit=SL|TP
            r=((sx/price-1-FEE) if SL[int(np.argmax(hit))] else (tp/price-1-FEE)) if hit.any() else C[hi-1]/price-1-FEE
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=H[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            r=(((price-sx)/price-FEE) if SL[int(np.argmax(hit))] else ((price-tp)/price-FEE)) if hit.any() else (price-C[hi-1])/price-FEE
        out.append((int(ts[ci]),float(r))); last=ts[ci]
    return out

# (그룹, 셀명, 심볼, itv, mode, win, K, B, cd)
CELLS=[
 ('MT5','BTCUSD 추세롱','BTCUSD','D10','trendlong',90,2.5,-2.0,2),
 ('MT5','ETHUSD 추세롱','ETHUSD','D10','trendlong',60,2.8,-3.0,1),
 ('MT5','XAUUSD 추세롱','XAUUSD','D10','trendlong',90,2.2,-1.4,3),
 ('MT5','XAUUSD 역추숏','XAUUSD','D10','short',200,2.8,1.2,1),
 ('MT5','XAGUSD 추세롱','XAGUSD','D10','trendlong',200,2.5,0.8,1),
 ('MT5','WTI 역추롱','WTI','D10','long',200,1.4,0.8,5),
 ('MT5','US100 역추롱','US100','D10','long',150,2.0,-3.0,1),
 ('MT5','JP225 역추롱','JP225','D10','long',200,2.0,-3.0,1),
 ('MT5','GER40 역추롱','GER40','D10','long',200,2.2,-3.0,1),
 ('MT5','UK100 역추롱','UK100','D10','long',90,1.7,-0.8,2),
 ('MT5','UK100 역추숏','UK100','D10','short',200,2.3,1.0,1),
 ('MT5','HK50 역추롱','HK50','D10','long',200,1.8,-3.0,3),
 ('MT5','HK50 역추숏','HK50','D10','short',150,2.2,2.0,1),
 ('FX','EURUSD 역추롱','EURUSD','D10','long',200,2.2,-0.6,1),
 ('FX','GBPUSD 역추롱','GBPUSD','D10','long',150,2.4,1.6,1),
 ('FX','USDJPY 추세롱','USDJPY','D10','trendlong',120,2.6,-1.0,1),
 ('FX','AUDUSD 역추롱','AUDUSD','D10','long',90,2.3,-2.2,1),
 ('FX','USDCAD 역추롱','USDCAD','D10','long',150,1.9,1.2,1),
 ('FX','USDCHF 역추롱','USDCHF','D10','long',200,2.3,-0.6,1),
 ('FX','NZDUSD 역추롱','NZDUSD','D10','long',200,2.2,-3.0,1),
 ('Bybit','BTCUSDT 추세롱','BTCUSDT','D','trendlong',90,2.5,-2.0,2),
 ('Bybit','ETHUSDT 추세롱','ETHUSDT','D','trendlong',60,2.5,-3.0,1),
 ('Bybit','SOLUSDT 추세롱','SOLUSDT','D','trendlong',90,2.9,-3.0,1),
 ('Bybit','XRPUSDT 추세롱','XRPUSDT','D','trendlong',90,3.1,-3.0,1),
]

def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
grp_tot=defaultdict(lambda: defaultdict(float))
print(f'{"그룹":4} {"셀":16} | {"전체: n":>7} {"기대%":>6} {"PF":>5} {"승%":>4} | {"2026: n":>7} {"합계%":>7} | 최근거래')
for g,name,sym,itv,mode,win,k,b,cd in CELLS:
    d=prep(sym,itv,win)
    if d is None: print(f'{g:4} {name:16} | 데이터없음'); continue
    tr=gen(d,mode,k,b,cd)
    if not tr: print(f'{g:4} {name:16} | 거래없음'); continue
    a=np.array([r for _,r in tr]); w_=a>0
    pf=a[w_].sum()/(-a[~w_].sum()) if (~w_).any() else 99
    t26=[(en,r) for en,r in tr if yr(en)==2026]
    s26=sum(r for _,r in t26)*100
    lastd=datetime.fromtimestamp(tr[-1][0]/1000,tz=timezone.utc).strftime('%y-%m-%d')
    for en,r in tr: grp_tot[g][yr(en)]+=r
    print(f'{g:4} {name:16} | {len(a):>7} {a.mean()*100:>+6.2f} {pf:>5.2f} {100*w_.mean():>3.0f}% | {len(t26):>7} {s26:>+7.1f} | {lastd}')
print()
print('=== 그룹별 연도 합계 (수익률 단순합 %p — 사이징 2%면 ×0.02) ===')
yrs=sorted({y for v in grp_tot.values() for y in v})
hdr='그룹  '+' '.join(f'{y:>7}' for y in yrs)
print(hdr)
for g in ['MT5','FX','Bybit']:
    print(f'{g:5} '+' '.join(f'{grp_tot[g].get(y,0)*100:>+7.1f}' for y in yrs))
