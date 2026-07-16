# -*- coding: utf-8 -*-
"""S22 Bybit 11셀 견고성 검증 — 선택 파라미터 vs 이웃 격자(각 축 ±1스텝).
스파이크(선택값이 이웃 중앙값의 2배↑ or 이웃 다수 탈락) 탐지 → 라이브 채택 판단.
출력: 셀별 [선택 기대%] vs [이웃 n/중앙값/최소/탈락수]."""
import sys, os, itertools
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000; HOLD=15*DAY

_c={}
def arrs(sym):
    if sym not in _c:
        m=_load_cache(sym,'240')
        _c[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                 np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _c[sym]
def ema(x,s):
    a=2/(s+1); out=np.copy(x)
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out
def zma(C,win):
    n=len(C); cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd
def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
def ev(tr,yt):
    if len(tr)<25: return None
    a=np.array([r for _,r in tr]); w=a>0
    exp=a.mean()*100
    pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
    ysum=defaultdict(float)
    for en,r in tr: ysum[yr(en)]+=r*100
    bad=[y for y,s in ysum.items() if s<-3]; worst=min(ysum.values())
    ok = exp>=0.4 and len(bad)<=2 and worst>-30
    return dict(exp=exp,pf=pf,ok=ok)

def sim_z(sym,win,k,b,mode,cdh):
    C,Hh,L,ts=arrs(sym); n=len(C)
    z,ma,sd=zma(C,win)
    cand=np.flatnonzero(z>=k) if mode=='tl' else np.flatnonzero(z<=-k)
    cand=cand[cand>=win]; out=[]; last=None; cd=cdh*H
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+HOLD,'right'))
        if hi<=ci+1: continue
        if mode=='tl':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            tp=price*(1+(1-tpb/price))
        else:
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
        TP=Hh[ci+1:hi]>=tp
        r=(tp/price-1-FEE) if TP.any() else C[hi-1]/price-1-FEE
        out.append((int(ts[ci]),r)); last=ts[ci]
    return out
def sim_ewz(sym,s,k,hb,side):
    C,Hh,L,ts=arrs(sym); n=len(C)
    mu=ema(C,s); resid=C-mu; sig=ema(np.abs(resid),s)
    z=resid/np.where(sig==0,1e-12,sig)
    cand=np.flatnonzero(z>=k) if side>0 else np.flatnonzero(z<=-k)
    out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        ex=min(ci+hb,n-1)
        out.append((int(ts[ci]),side*(C[ex]/C[ci]-1)-FEE)); last=ci
    return out
def sim_fade(sym,M,thr,kind,cap):
    C,Hh,L,ts=arrs(sym); n=len(C)
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    cand=np.flatnonzero(ret<=-thr); out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        price=C[ci]; hi=min(ci+1+cap,n)
        if kind=='R':
            tp=price*(1+1.5*abs(ret[ci])); TP=Hh[ci+1:hi]>=tp
            r=(tp/price-1-FEE) if TP.any() else C[hi-1]/price-1-FEE
        else:
            r=C[hi-1]/price-1-FEE
        out.append((int(ts[ci]),r)); last=ci
    return out

def yt_of(sym):
    _,_,_,ts=arrs(sym)
    return (int(ts[-1])-int(ts[0]))/(365.25*DAY)

CELLS=[
 ('BTC z추세롱','z','BTCUSDT',dict(win=240,k=3.25,b=-2.5,mode='tl',cdh=24),
   dict(win=[180,240,300],k=[3.0,3.25,3.5],b=[-3,-2.5,-2],mode=['tl'],cdh=[24])),
 ('BTC 페이드','f','BTCUSDT',dict(M=6,thr=0.10,kind='T',cap=6),
   dict(M=[3,6,12],thr=[0.07,0.10,0.12],kind=['T'],cap=[6,12])),
 ('BTC ewz추숏','e','BTCUSDT',dict(s=50,k=3.0,hb=18,side=-1),
   dict(s=[30,50,80],k=[2.5,3.0,3.5],hb=[12,18,30],side=[-1])),
 ('ETH ewz추롱','e','ETHUSDT',dict(s=200,k=3.0,hb=90,side=1),
   dict(s=[150,200,300],k=[2.5,3.0,3.5],hb=[60,90,120],side=[1])),
 ('ETH 페이드','f','ETHUSDT',dict(M=12,thr=0.15,kind='T',cap=6),
   dict(M=[6,12,18],thr=[0.12,0.15,0.18],kind=['T'],cap=[6,12])),
 ('ETH ewz추숏','e','ETHUSDT',dict(s=50,k=2.5,hb=18,side=-1),
   dict(s=[30,50,80],k=[2.0,2.5,3.0],hb=[12,18,30],side=[-1])),
 ('SOL ewz추롱','e','SOLUSDT',dict(s=20,k=3.0,hb=60,side=1),
   dict(s=[15,20,30],k=[2.5,3.0,3.5],hb=[30,60,90],side=[1])),
 ('SOL z역추롱','z','SOLUSDT',dict(win=240,k=2.75,b=-1,mode='rl',cdh=24),
   dict(win=[180,240,300],k=[2.5,2.75,3.0],b=[-1.5,-1,-0.5],mode=['rl'],cdh=[24])),
 ('XRP z추세롱','z','XRPUSDT',dict(win=240,k=4.0,b=-3,mode='tl',cdh=12),
   dict(win=[180,240,300],k=[3.75,4.0,4.25],b=[-3,-2.5],mode=['tl'],cdh=[12])),
 ('XRP z역추롱','z','XRPUSDT',dict(win=240,k=3.0,b=1.5,mode='rl',cdh=12),
   dict(win=[180,240,300],k=[2.75,3.0,3.25],b=[1.0,1.5],mode=['rl'],cdh=[12])),
 ('XRP 페이드','f','XRPUSDT',dict(M=12,thr=0.15,kind='R',cap=60),
   dict(M=[6,12,18],thr=[0.12,0.15,0.18],kind=['R'],cap=[30,60,90])),
]
SIM={'z':sim_z,'e':sim_ewz,'f':sim_fade}
print(f'{"셀":14} {"선택 기대%":>8} | {"이웃n":>4} {"중앙값":>7} {"최소":>7} {"탈락":>4} | 판정')
for name,typ,sym,chosen,grid in CELLS:
    yt=yt_of(sym)
    sel=ev([(t,r) for t,r in SIM[typ](sym,**chosen)],yt)
    keys=list(grid)
    exps=[]; fails=0
    for combo in itertools.product(*(grid[k] for k in keys)):
        p=dict(zip(keys,combo))
        if p==chosen: continue
        e=ev([(t,r) for t,r in SIM[typ](sym,**p)],yt)
        if e is None or not e['ok']: fails+=1
        if e is not None: exps.append(e['exp'])
    med=float(np.median(exps)) if exps else 0
    mn=float(min(exps)) if exps else 0
    spike = sel['exp']>2*max(med,0.01) or fails>len(exps)*0.5
    print(f'{name:14} {sel["exp"]:>+8.2f} | {len(exps):>4} {med:>+7.2f} {mn:>+7.2f} {fails:>4} | {"⚠️스파이크" if spike else "✅평지"}')
