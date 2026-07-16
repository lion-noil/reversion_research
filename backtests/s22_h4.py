# -*- coding: utf-8 -*-
"""S22 후보 — 4시간봉 검증 (1m 리샘플). z 4방향 + 급락페이드, S11 방법론(연도균형).
z: 창 30~180봉(5~30일), 롱 SL無/숏 SL대칭, TP=밴드(추세=오버슛 미러, 역추=복귀), 보유캡 15일.
fade: M봉 낙폭 트리거, 시간청산/되돌림×1.5. FEE 0.11%. 필터: n≥30, 기대≥0.4%, 마이너스연도(<-3%p)≤2, 최악>-30%p."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000; BAR=4*H; HOLD=15*DAY

def load4h(sym):
    m=_load_cache(sym,'1')
    if not m: return None
    ts1=np.array([x['start'] for x in m],np.int64)
    C1=np.array([x['close'] for x in m]); H1=np.array([x['high'] for x in m]); L1=np.array([x['low'] for x in m]); O1=np.array([x['open'] for x in m])
    g=ts1//BAR
    ug,idx=np.unique(g,return_index=True)
    C=np.empty(len(ug)); Hh=np.maximum.reduceat(H1,idx); L=np.minimum.reduceat(L1,idx); O=O1[idx]
    ends=np.append(idx[1:],len(C1))-1
    C=C1[ends]; ts=ug*BAR
    return C,Hh,L,ts.astype(np.int64)

def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def zma(C,win):
    n=len(C); cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

def evaluate(tr,yrs_total,gmin=3):
    if len(tr)<30: return None
    a=np.array([r for _,r in tr]); w=a>0
    exp=a.mean()*100
    if exp<0.4: return None
    pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
    ysum=defaultdict(float)
    for en,r in tr: ysum[yr(en)]+=r*100
    bad=[y for y,s in ysum.items() if s<-3]; worst=min(ysum.values())
    if len(bad)>2 or worst<-30: return None
    gpy=len(tr)/yrs_total
    if gpy<gmin: return None
    return dict(n=len(tr),exp=exp,pf=pf,wr=100*w.mean(),mo=gpy/12,worst=worst,
                ys=' '.join(f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))

def zgen(C,Hh,L,ts,win,k,b,mode,cdh,ma,sd,z):
    n=len(C)
    epos = mode in('tl','rs')
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
            r=(tp/price-1-FEE) if TP.any() else C[hi-1]/price-1-FEE
        elif mode=='rl':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            TP=Hh[ci+1:hi]>=tp
            r=(tp/price-1-FEE) if TP.any() else C[hi-1]/price-1-FEE
        elif mode=='ts':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            TPh=L[ci+1:hi]<=tp; SL=Hh[ci+1:hi]>=sl; hit=TPh|SL
            if hit.any(): j=int(np.argmax(hit)); xp=sl if SL[j] else tp
            else: xp=C[hi-1]
            r=(price-xp)/price-FEE
        else:  # rs 역추숏
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sl=price*(1+pct)
            TPh=L[ci+1:hi]<=tp; SL=Hh[ci+1:hi]>=sl; hit=TPh|SL
            if hit.any(): j=int(np.argmax(hit)); xp=sl if SL[j] else tp
            else: xp=C[hi-1]
            r=(price-xp)/price-FEE
        out.append((int(ts[ci]),r)); last=ts[ci]
    return out

SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','JP225','US100','GER40','UK100','HK50','XAGUSD','WTI','XAUUSD']
MODENM={'tl':'추세롱','rl':'역추롱','ts':'추세숏','rs':'역추숏'}
res=defaultdict(list)
for sym in SYMS:
    d=load4h(sym)
    if d is None: continue
    C,Hh,L,ts=d
    yrs_total=(int(ts[-1])-int(ts[0]))/(365.25*DAY)
    # z 4방향
    for win in [30,60,90,120,180]:
        z,ma,sd=zma(C,win)
        for mode in ['tl','rl','ts','rs']:
            for k in [2.0,2.5,3.0,3.5,4.0]:
                for b in [-3,-2,-1,0,1]:
                    for cdh in [12,24,48]:
                        tr=zgen(C,Hh,L,ts,win,k,b,mode,cdh,ma,sd,z)
                        e=evaluate(tr,yrs_total)
                        if e: res[sym].append((f'z{MODENM[mode]} w{win*4//24}d K{k} B{b} cd{cdh}h',e))
    # fade롱
    for M in [3,6,12,18,30]:
        ret=C/np.roll(C,M)-1; ret[:M]=np.nan
        for r0 in [0.03,0.05,0.07,0.10]:
            cand=np.flatnonzero(ret<=-r0)
            if len(cand)<30: continue
            out=[]
            for exit_,cap in [('T',6),('T',30),('T',60),('R',60)]:  # 캡=봉수(4h): 1d/5d/10d
                trl=[]; last=None
                n=len(C)
                for ci in cand:
                    if ci+1>=n: break
                    if last is not None and (ts[ci]-last)<24*H: continue
                    price=C[ci]; hi=min(ci+1+cap,n)
                    if exit_=='R':
                        tp=price*(1+1.5*abs(ret[ci])); TP=Hh[ci+1:hi]>=tp
                        r=(tp/price-1-FEE) if TP.any() else C[hi-1]/price-1-FEE
                    else:
                        r=C[hi-1]/price-1-FEE
                    trl.append((int(ts[ci]),r)); last=ts[ci]
                e=evaluate(trl,yrs_total)
                if e: res[sym].append((f'fade롱 {M*4}h≤-{r0*100:.0f}% {exit_}{cap*4//24}d',e))
    print(f'{sym} done ({len(res[sym])})',flush=True)

print()
print(f'{"심볼":8} {"셀":30} {"n":>4} {"기대%":>6} {"PF":>5} {"승%":>4} {"월":>4} | 연도별')
for sym in SYMS:
    rs=sorted(res.get(sym,[]),key=lambda x:-x[1]['exp']*min(x[1]['pf'],5))
    seen=set(); shown=0
    for name,e in rs:
        fam=name.split()[0]
        if fam in seen: continue
        seen.add(fam); shown+=1
        print(f'{sym:8} {name:30} {e["n"]:>4} {e["exp"]:>+6.2f} {e["pf"]:>5.2f} {e["wr"]:>3.0f}% {e["mo"]:>4.1f} | {e["ys"]}')
        if shown>=4: break
