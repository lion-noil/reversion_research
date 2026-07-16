# -*- coding: utf-8 -*-
"""일봉 크립토 책(S3 추세롱 4셀, cryptod) vs S22(4h 8셀) 분기별 비교 + 한계기여.
각 북 독립 $10k·5% 실현복리 분기수익, 상관, 그리고 S22단독 vs S22+일봉 합산(같은 계좌) 비교.
일봉 v2 모델: TP/SL 대칭·손절우선·보유15일. S22: HANDOFF_S22 8셀."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000; HOLD4=15*DAY
FRAC=0.05

def zma(C,win):
    n=len(C); cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd
def ema(x,s):
    a=2/(s+1); out=np.copy(x)
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out

# ---- 일봉 S3 추세롱 (TP/SL 대칭, 보유15일) ----
def daily_tl(sym,win,k,b,cdd):
    m=_load_cache(sym,'D')
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64); n=len(C)
    z,ma,sd=zma(C,win)
    cand=np.flatnonzero(z>=k); cand=cand[cand>=win]
    out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cdd*DAY: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+15*DAY,'right'))
        if hi<=ci+1: continue
        tp=ma[ci]+b*sd[ci]
        if tp>=price: continue
        pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
        SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tx; hit=SL|TP
        if hit.any():
            j=int(np.argmax(hit)); ex=ci+1+j
            r=(sx/price-1-FEE) if SL[j] else (tx/price-1-FEE)
        else: ex=hi-1; r=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),r)); last=ts[ci]
    return out

# ---- S22 8셀 (4h) ----
_c={}
def a4(sym):
    if sym not in _c:
        m=_load_cache(sym,'240')
        _c[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                 np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _c[sym]
def z4(sym,win,k,b,mode,cdh):
    C,Hh,L,ts=a4(sym); n=len(C)
    z,ma,sd=zma(C,win)
    cand=np.flatnonzero(z>=k) if mode=='tl' else np.flatnonzero(z<=-k)
    cand=cand[cand>=win]; out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cdh*H: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+HOLD4,'right'))
        if hi<=ci+1: continue
        if mode=='tl':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            tp=price*(1+(1-tpb/price))
        else:
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
        TP=Hh[ci+1:hi]>=tp
        if TP.any(): ex=ci+1+int(np.argmax(TP)); r=tp/price-1-FEE
        else: ex=hi-1; r=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),r)); last=ts[ci]
    return out
def ewz4(sym,s,k,hb,side):
    C,Hh,L,ts=a4(sym); n=len(C)
    mu=ema(C,s); resid=C-mu; sig=ema(np.abs(resid),s)
    z=resid/np.where(sig==0,1e-12,sig)
    cand=np.flatnonzero(z>=k) if side>0 else np.flatnonzero(z<=-k)
    out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        ex=min(ci+hb,n-1)
        out.append((int(ts[ci]),int(ts[ex]),side*(C[ex]/C[ci]-1)-FEE)); last=ci
    return out
def fade4(sym,M,thr,cap):
    C,Hh,L,ts=a4(sym); n=len(C)
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    cand=np.flatnonzero(ret<=-thr); out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        price=C[ci]; hi=min(ci+1+cap,n)
        tp=price*(1+1.5*abs(ret[ci])); TP=Hh[ci+1:hi]>=tp
        if TP.any(): ex=ci+1+int(np.argmax(TP)); r=tp/price-1-FEE
        else: ex=hi-1; r=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),r)); last=ci
    return out

DAILY=[]
DAILY+=daily_tl('BTCUSDT',90,2.5,-2.0,2)
DAILY+=daily_tl('ETHUSDT',60,2.5,-3.0,1)
DAILY+=daily_tl('SOLUSDT',90,2.9,-3.0,1)
DAILY+=daily_tl('XRPUSDT',90,3.1,-3.0,1)
S22=[]
S22+=z4('BTCUSDT',240,3.25,-2.5,'tl',24)
S22+=ewz4('ETHUSDT',200,3.0,90,1)
S22+=ewz4('ETHUSDT',50,2.5,18,-1)
S22+=ewz4('SOLUSDT',20,3.0,60,1)
S22+=z4('SOLUSDT',240,2.75,-1,'rl',24)
S22+=z4('XRPUSDT',240,4.0,-3,'tl',12)
S22+=z4('XRPUSDT',240,3.0,1.5,'rl',12)
S22+=fade4('XRPUSDT',12,0.15,'R'==None and 0 or 60)
print(f'일봉 {len(DAILY)}건 / S22 {len(S22)}건',flush=True)

def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
def run_book(trades,frac):
    ev=[]
    for i,(en,ex,r) in enumerate(trades): ev.append((en,0,i)); ev.append((ex,1,i))
    ev.sort(key=lambda x:(x[0],x[1]))
    eq=10000.0; open_={}
    qend={}; peak=eq; mdd=0.0
    for t,typ,i in ev:
        if typ==0: open_[i]=frac*eq
        else:
            no=open_.pop(i,None)
            if no is not None:
                eq+=no*trades[i][2]
                peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak)
        qend[qof(t)]=eq
    return qend,eq,mdd
qd,eqd,mdd_d=run_book(sorted(DAILY),FRAC)
qs,eqs,mdd_s=run_book(sorted(S22),FRAC)
COMB=sorted(DAILY+S22)
qc,eqc,mdd_c=run_book(COMB,FRAC)

qs_all=sorted(set(qd)|set(qs))
print(f'\n{"분기":>7} | {"일봉S3(4셀)":>10} | {"S22(8셀)":>10} | 승자')
prevd=prevs=10000.0
rd=[]; rs=[]
for q in qs_all:
    ed=qd.get(q,prevd); es=qs.get(q,prevs)
    d_=(ed/prevd-1)*100; s_=(es/prevs-1)*100
    rd.append(d_); rs.append(s_)
    win='일봉' if d_>s_+0.001 else ('S22' if s_>d_+0.001 else '=')
    print(f'{q:>7} | {d_:>+9.1f}% | {s_:>+9.1f}% | {win}')
    prevd,prevs=ed,es
rd=np.array(rd); rs=np.array(rs)
corr=np.corrcoef(rd,rs)[0,1]
yrs=len(qs_all)/4
print(f'\n일봉S3 : 최종 ${eqd:,.0f} ({eqd/10000:.2f}배, CAGR {((eqd/10000)**(1/yrs)-1)*100:+.1f}%) 실현MDD {mdd_d*100:.1f}% | 승분기 {int((rd>rs+0.001).sum())}')
print(f'S22    : 최종 ${eqs:,.0f} ({eqs/10000:.2f}배, CAGR {((eqs/10000)**(1/yrs)-1)*100:+.1f}%) 실현MDD {mdd_s*100:.1f}% | 승분기 {int((rs>rd+0.001).sum())}')
print(f'분기수익 상관 {corr:.2f}')
print(f'\nS22 단독      : ${eqs:,.0f}')
print(f'S22+일봉 합산 : ${eqc:,.0f} (한계기여 {((eqc/eqs)-1)*100:+.1f}%, 합산 실현MDD {mdd_c*100:.1f}% vs S22단독 {mdd_s*100:.1f}%)')
