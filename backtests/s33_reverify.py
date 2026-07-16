# -*- coding: utf-8 -*-
"""S33(일봉책) 재검증 — 채택 셀 20개(mt5d 13 + fxd 7)를 13.5년 일봉(D13, 시간봉 리샘플)으로.
현행 파라미터 성적 + 이웃격자(win·K·B ±1) 중앙/최소 → 유지/조정/기각 판정.
일봉 v2 모델: TP/SL 대칭·손절우선·보유15d·쿨다운(일). FEE 0.11%(FX도 일봉은 보수 0.11%)."""
import sys, os, itertools
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; DAY=86400*1000
FXs={'USDJPY','EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','NZDUSD'}

_c={}
def arrs(sym):
    if sym not in _c:
        m=_load_cache(sym,'D13')
        _c[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                 np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _c[sym]
def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
def zma(C,win):
    n=len(C); cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

def cell(sym,win,k,b,cdd,mode):
    C,Hh,L,ts=arrs(sym); n=len(C)
    if n<=win: return []
    z,ma,sd=zma(C,win)
    if mode=='long': cand=np.flatnonzero(z<=-k)
    elif mode=='short': cand=np.flatnonzero(z>=k)
    else: cand=np.flatnonzero(z>=k)
    cand=cand[cand>=win]; out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cdd*DAY: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+15*DAY,'right'))
        if hi<=ci+1: continue
        if mode=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); r=(sx/price-1-FEE) if SL[j] else (tp/price-1-FEE)
            else: r=C[hi-1]/price-1-FEE
        elif mode=='short':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=Hh[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); r=((price-sx)/price-FEE) if SL[j] else ((price-tp)/price-FEE)
            else: r=(price-C[hi-1])/price-FEE
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); r=(sx/price-1-FEE) if SL[j] else (tx/price-1-FEE)
            else: r=C[hi-1]/price-1-FEE
        out.append((int(ts[ci]),r)); last=ts[ci]
    return out

def ev(tr):
    if len(tr)<25: return None
    a=np.array([r for _,r in tr]); w=a>0
    exp=a.mean()*100
    pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
    ysum=defaultdict(float)
    for en,r in tr: ysum[yr(en)]+=r*100
    bad=[y for y,s in ysum.items() if s<-3]; worst=min(ysum.values())
    ok = exp>=0.3 and len(bad)<=2 and worst>-30
    return dict(exp=exp,pf=pf,ok=ok,n=len(tr),nbad=len(bad),worst=worst,
                ys=' '.join(f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))

# 채택 셀: (이름, 심볼, win, K, B, cd일, mode)
CELLS=[
 ('XAUUSD S3추롱','XAUUSD',90,2.2,-1.4,3,'tl'), ('XAUUSD S4역숏','XAUUSD',200,2.8,1.2,1,'short'),
 ('XAGUSD S3추롱','XAGUSD',200,2.5,0.8,1,'tl'), ('WTI S4역롱','WTI',200,1.4,0.8,5,'long'),
 ('US100 S4역롱','US100',150,2.0,-3.0,1,'long'), ('JP225 S4역롱','JP225',200,2.0,-3.0,1,'long'),
 ('GER40 S4역롱','GER40',200,2.2,-3.0,1,'long'),
 ('UK100 S4역롱','UK100',90,1.7,-0.8,2,'long'), ('UK100 S4역숏','UK100',200,2.3,1.0,1,'short'),
 ('HK50 S4역롱','HK50',200,1.8,-3.0,3,'long'), ('HK50 S4역숏','HK50',150,2.2,2.0,1,'short'),
 ('EURUSD S4역롱','EURUSD',200,2.2,-0.6,1,'long'), ('GBPUSD S4역롱','GBPUSD',150,2.4,1.6,1,'long'),
 ('USDJPY S3추롱','USDJPY',120,2.6,-1.0,1,'tl'), ('AUDUSD S4역롱','AUDUSD',90,2.3,-2.2,1,'long'),
 ('USDCAD S4역롱','USDCAD',150,1.9,1.2,1,'long'), ('USDCHF S4역롱','USDCHF',200,2.3,-0.6,1,'long'),
 ('NZDUSD S4역롱','NZDUSD',200,2.2,-3.0,1,'long'),
]
print(f'{"셀":16} {"n":>4} {"기대%":>6} {"PF":>5} | {"이웃중앙":>7} {"이웃min":>7} | 판정 | 연도별(13.5y)')
for name,sym,win,k,b,cdd,mode in CELLS:
    sel=ev(cell(sym,win,k,b,cdd,mode))
    if sel is None:
        print(f'{name:16} n<25 — 판정불가'); continue
    exps=[]
    for dw,dk,db in itertools.product([-30,0,30],[-0.2,0,0.2],[-0.4,0,0.4]):
        if dw==dk==db==0: continue
        e=ev(cell(sym,max(win+dw,30),round(k+dk,2),round(b+db,2),cdd,mode))
        if e is not None: exps.append(e['exp'])
    med=float(np.median(exps)) if exps else 0.0
    mn=float(min(exps)) if exps else 0.0
    verdict='✅유지' if (sel['ok'] and med>0.15) else ('⚠️조정' if med>0.15 else '❌기각후보')
    print(f'{name:16} {sel["n"]:>4} {sel["exp"]:>+6.2f} {sel["pf"]:>5.2f} | {med:>+7.2f} {mn:>+7.2f} | {verdict} | {sel["ys"]}')
