# -*- coding: utf-8 -*-
"""S22 확장판(MT5·FX) 후보 24셀 — 이웃격자 견고성 검증 (13.5y, 4h봉).
각 축 ±1스텝 이웃의 기대값 분포로 스파이크 판정. 계획기대 = 이웃 중앙값(보수치).
판정: 선택기대 > 2×이웃중앙값 or 이웃 탈락(ok아님) > 50% → ⚠️스파이크."""
import sys, os, itertools
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; HOLD=15*DAY
FX={'USDJPY','EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','NZDUSD'}

_c={}
def arrs(sym):
    if sym not in _c:
        m=_load_cache(sym,'240')
        _c[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                 np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _c[sym]
def fee_of(sym): return 0.0002 if sym in FX else 0.0011
def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
def yt_of(sym):
    _,_,_,ts=arrs(sym); return (int(ts[-1])-int(ts[0]))/(365.25*DAY)
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

def ev(tr,sym):
    if len(tr)<30: return None
    isfx=sym in FX; emin=0.15 if isfx else 0.4; ybad=-1.5 if isfx else -3.0
    a=np.array([r for _,r in tr]); w=a>0
    exp=a.mean()*100
    pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
    ysum=defaultdict(float)
    for en,r in tr: ysum[yr(en)]+=r*100
    bad=[y for y,s in ysum.items() if s<ybad]; worst=min(ysum.values())
    ok = exp>=emin and len(bad)<=2 and worst>-30 and len(tr)/yt_of(sym)>=3
    return dict(exp=exp,pf=pf,ok=ok,n=len(tr))

def sim_z(sym,win,k,b,mode,cdh):
    C,Hh,L,ts=arrs(sym); n=len(C); fee=fee_of(sym)
    z,ma,sd=zma(C,win)
    epos=mode in('tl','rs')
    cand=np.flatnonzero(z>=k) if epos else np.flatnonzero(z<=-k)
    cand=cand[cand>=win]; out=[]; last=None; cd=cdh*H
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+HOLD,'right'))
        if hi<=ci+1: continue
        if mode=='tl':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            tp=price*(1+(1-tpb/price)); TP=Hh[ci+1:hi]>=tp
            r=(tp/price-1-fee) if TP.any() else C[hi-1]/price-1-fee
        elif mode=='rl':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            TP=Hh[ci+1:hi]>=tp
            r=(tp/price-1-fee) if TP.any() else C[hi-1]/price-1-fee
        elif mode=='ts':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            TPh=L[ci+1:hi]<=tp; SL=Hh[ci+1:hi]>=sl; hit=TPh|SL
            if hit.any(): j=int(np.argmax(hit)); xp=sl if SL[j] else tp
            else: xp=C[hi-1]
            r=(price-xp)/price-fee
        else:  # rs
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sl=price*(1+pct)
            TPh=L[ci+1:hi]<=tp; SL=Hh[ci+1:hi]>=sl; hit=TPh|SL
            if hit.any(): j=int(np.argmax(hit)); xp=sl if SL[j] else tp
            else: xp=C[hi-1]
            r=(price-xp)/price-fee
        out.append((int(ts[ci]),r)); last=ts[ci]
    return out
def sim_ewz(sym,s,k,hb,side,trig=None):
    # trig: 'hi'(z>=k) / 'lo'(z<=-k). 기본=추세( side>0→hi, side<0→lo ). 역추세는 명시 필요.
    C,Hh,L,ts=arrs(sym); n=len(C); fee=fee_of(sym)
    mu=ema(C,s); resid=C-mu; sig=ema(np.abs(resid),s)
    z=resid/np.where(sig==0,1e-12,sig)
    if trig is None: trig='hi' if side>0 else 'lo'
    cand=np.flatnonzero(z>=k) if trig=='hi' else np.flatnonzero(z<=-k)
    out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        ex=min(ci+hb,n-1)
        out.append((int(ts[ci]),side*(C[ex]/C[ci]-1)-fee)); last=ci
    return out
def sim_fade(sym,M,thr,kind,cap):
    C,Hh,L,ts=arrs(sym); n=len(C); fee=fee_of(sym)
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    cand=np.flatnonzero(ret<=-thr); out=[]; last=-10**9
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
    return out
def sim_sweep(sym,N,hb):
    C,Hh,L,ts=arrs(sym); n=len(C); fee=fee_of(sym)
    pl=np.full(n,np.nan)
    for i in range(N,n): pl[i]=L[i-N:i].min()
    cand=np.flatnonzero((L<pl)&(C>pl)); out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        ex=min(ci+hb,n-1)
        out.append((int(ts[ci]),C[ex]/C[ci]-1-fee)); last=ci
    return out
SIM={'z':sim_z,'e':sim_ewz,'f':sim_fade,'w':sim_sweep}

CELLS=[
 # 지수·원자재
 ('JP225 z추롱','z','JP225',dict(win=120,k=3.0,b=1,mode='tl',cdh=12),
   dict(win=[90,120,180],k=[2.5,3.0,3.5],b=[0.5,1,1.5],mode=['tl'],cdh=[12])),
 ('JP225 sweep롱','w','JP225',dict(N=120,hb=60),dict(N=[60,120,180],hb=[30,60,90])),
 ('JP225 페이드','f','JP225',dict(M=18,thr=0.05,kind='T',cap=60),
   dict(M=[12,18,30],thr=[0.04,0.05,0.06],kind=['T'],cap=[30,60,90])),
 ('US100 z역롱','z','US100',dict(win=180,k=3.0,b=1,mode='rl',cdh=12),
   dict(win=[120,180,240],k=[2.5,3.0,3.5],b=[0.5,1,1.5],mode=['rl'],cdh=[12])),
 ('US100 페이드','f','US100',dict(M=30,thr=0.07,kind='R',cap=30),
   dict(M=[18,30,42],thr=[0.06,0.07,0.08],kind=['R'],cap=[18,30,60])),
 ('GER40 z역롱','z','GER40',dict(win=120,k=3.5,b=0,mode='rl',cdh=12),
   dict(win=[90,120,180],k=[3.0,3.5,4.0],b=[-0.5,0,0.5],mode=['rl'],cdh=[12])),
 ('UK100 z역숏','z','UK100',dict(win=120,k=3.0,b=-2,mode='rs',cdh=12),
   dict(win=[90,120,180],k=[2.5,3.0,3.5],b=[-2.5,-2,-1.5],mode=['rs'],cdh=[12])),
 ('UK100 페이드','f','UK100',dict(M=12,thr=0.03,kind='T',cap=60),
   dict(M=[6,12,18],thr=[0.025,0.03,0.04],kind=['T'],cap=[30,60,90])),
 ('HK50 z추롱','z','HK50',dict(win=180,k=3.0,b=-2,mode='tl',cdh=12),
   dict(win=[120,180,240],k=[2.5,3.0,3.5],b=[-2.5,-2,-1.5],mode=['tl'],cdh=[12])),
 ('HK50 z역롱','z','HK50',dict(win=180,k=3.0,b=1,mode='rl',cdh=12),
   dict(win=[120,180,240],k=[2.5,3.0,3.5],b=[0.5,1,1.5],mode=['rl'],cdh=[12])),
 ('XAGUSD ewz역롱','e','XAGUSD',dict(s=200,k=2.5,hb=30,side=1,trig='lo'),
   dict(s=[150,200,300],k=[2.0,2.5,3.0],hb=[18,30,60],side=[1],trig=['lo'])),
 ('XAGUSD ewz추숏','e','XAGUSD',dict(s=50,k=3.0,hb=18,side=-1),
   dict(s=[30,50,80],k=[2.5,3.0,3.5],hb=[12,18,30],side=[-1])),
 ('XAUUSD z추롱','z','XAUUSD',dict(win=60,k=3.0,b=-1,mode='tl',cdh=48),
   dict(win=[42,60,90],k=[2.5,3.0,3.5],b=[-1.5,-1,-0.5],mode=['tl'],cdh=[48])),
 ('XAUUSD 페이드','f','XAUUSD',dict(M=6,thr=0.03,kind='T',cap=60),
   dict(M=[6,12],thr=[0.025,0.03,0.04],kind=['T'],cap=[30,60,90])),
 ('WTI ewz역롱','e','WTI',dict(s=200,k=2.5,hb=18,side=1,trig='lo'),
   dict(s=[150,200,300],k=[2.0,2.5,3.0],hb=[12,18,30],side=[1],trig=['lo'])),
 ('WTI 페이드','f','WTI',dict(M=6,thr=0.07,kind='T',cap=60),
   dict(M=[6,12],thr=[0.06,0.07,0.08],kind=['T'],cap=[30,60,90])),
 # FX
 ('USDJPY z역롱','z','USDJPY',dict(win=120,k=2.5,b=1,mode='rl',cdh=12),
   dict(win=[90,120,180],k=[2.0,2.5,3.0],b=[0.5,1,1.5],mode=['rl'],cdh=[12])),
 ('USDJPY 페이드','f','USDJPY',dict(M=18,thr=0.03,kind='T',cap=30),
   dict(M=[12,18,30],thr=[0.025,0.03,0.04],kind=['T'],cap=[18,30,60])),
 ('GBPUSD z역숏','z','GBPUSD',dict(win=120,k=3.0,b=-2,mode='rs',cdh=12),
   dict(win=[90,120,180],k=[2.5,3.0,3.5],b=[-2.5,-2,-1.5],mode=['rs'],cdh=[12])),
 ('AUDUSD z역숏','z','AUDUSD',dict(win=30,k=3.0,b=-2,mode='rs',cdh=12),
   dict(win=[24,30,42],k=[2.5,3.0,3.5],b=[-2.5,-2,-1.5],mode=['rs'],cdh=[12])),
 ('USDCHF z추숏','z','USDCHF',dict(win=90,k=3.0,b=-1,mode='ts',cdh=12),
   dict(win=[60,90,120],k=[2.5,3.0,3.5],b=[-1.5,-1,-0.5],mode=['ts'],cdh=[12])),
 ('NZDUSD z역숏','z','NZDUSD',dict(win=60,k=3.0,b=-1,mode='rs',cdh=24),
   dict(win=[42,60,90],k=[2.5,3.0,3.5],b=[-1.5,-1,-0.5],mode=['rs'],cdh=[24])),
 ('EURUSD ewz역숏','e','EURUSD',dict(s=100,k=2.5,hb=18,side=-1,trig='hi'),
   dict(s=[80,100,150],k=[2.0,2.5,3.0],hb=[12,18,30],side=[-1],trig=['hi'])),
 ('USDCAD sweep롱','w','USDCAD',dict(N=120,hb=60),dict(N=[60,120,180],hb=[30,60,90])),
]
print(f'{"셀":16} {"선택%":>7} {"n":>4} | {"이웃n":>4} {"중앙":>7} {"최소":>7} {"탈락":>4} | 판정')
for name,typ,sym,chosen,grid in CELLS:
    sel=ev(SIM[typ](sym,**chosen),sym)
    if sel is None:
        print(f'{name:16} 선택 자체가 n<30 — 제외'); continue
    keys=list(grid); exps=[]; fails=0
    for combo in itertools.product(*(grid[kk] for kk in keys)):
        p=dict(zip(keys,combo))
        if p==chosen: continue
        e=ev(SIM[typ](sym,**p),sym)
        if e is None or not e['ok']: fails+=1
        if e is not None: exps.append(e['exp'])
    med=float(np.median(exps)) if exps else 0.0
    mn=float(min(exps)) if exps else 0.0
    tot=len(exps)+ (0)
    spike = sel['exp']>2*max(med,0.01) or (fails> (len(exps) if exps else 1)*0.5)
    print(f'{name:16} {sel["exp"]:>+7.2f} {sel["n"]:>4} | {len(exps):>4} {med:>+7.2f} {mn:>+7.2f} {fails:>4} | {"⚠️스파이크" if spike else "✅평지"}')
