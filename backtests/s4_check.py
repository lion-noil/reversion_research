# -*- coding: utf-8 -*-
"""S4(일봉 역추세 크립토) 유지 여부 검토 — S4 단독 분기성적 vs S22, (S22+S3)에의 한계기여.
S4 셀(§3-C): BTC 역롱 W150 K1.1 B0.8 cd10 / ETH 역롱 W200 K1.9 B-0.4 cd2 / SOL 역롱 W200 K1.8 B0.4 cd1
/ XRP 역롱 W200 K1.6 B-1.4 cd2 / ETH 역숏 W200 K2.6 B2.2 cd1 / XRP 역숏 W200 K1.1 B0.2 cd7. v2모델(TP/SL대칭·보유15d)."""
import sys, os
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000
FRAC=0.05

def zma(C,win):
    n=len(C); cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

def daily_cell(sym,win,k,b,cdd,mode):
    m=_load_cache(sym,'D')
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64); n=len(C)
    z,ma,sd=zma(C,win)
    if mode in('long',): cand=np.flatnonzero(z<=-k)
    elif mode=='short': cand=np.flatnonzero(z>=k)
    else: cand=np.flatnonzero(z>=k)   # trendlong
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
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tp/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        elif mode=='short':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=Hh[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tp)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        else:  # trendlong
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tx/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),r)); last=ts[ci]
    return out

# ---- 북 구성 ----
S4=[]
S4+=daily_cell('BTCUSDT',150,1.1,0.8,10,'long')
S4+=daily_cell('ETHUSDT',200,1.9,-0.4,2,'long')
S4+=daily_cell('SOLUSDT',200,1.8,0.4,1,'long')
S4+=daily_cell('XRPUSDT',200,1.6,-1.4,2,'long')
S4+=daily_cell('ETHUSDT',200,2.6,2.2,1,'short')
S4+=daily_cell('XRPUSDT',200,1.1,0.2,7,'short')
S3=[]
S3+=daily_cell('BTCUSDT',90,2.5,-2.0,2,'tl')
S3+=daily_cell('ETHUSDT',60,2.5,-3.0,1,'tl')
S3+=daily_cell('SOLUSDT',90,2.9,-3.0,1,'tl')
S3+=daily_cell('XRPUSDT',90,3.1,-3.0,1,'tl')

# S22 8셀 트레이드 재사용 (s3s4_vs_s22와 동일 로직 재구현 대신 임포트 회피 — 간단 재생성)
_c={}
def a4(sym):
    if sym not in _c:
        m=_load_cache(sym,'240')
        _c[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                 np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _c[sym]
def ema(x,s):
    a=2/(s+1); out=np.copy(x)
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out
def z4(sym,win,k,b,mode,cdh):
    C,Hh,L,ts=a4(sym); n=len(C)
    z,ma,sd=zma(C,win)
    cand=np.flatnonzero(z>=k) if mode=='tl' else np.flatnonzero(z<=-k)
    cand=cand[cand>=win]; out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cdh*H: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+15*DAY,'right'))
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
S22=[]
S22+=z4('BTCUSDT',240,3.25,-2.5,'tl',24)
S22+=ewz4('ETHUSDT',200,3.0,90,1)
S22+=ewz4('ETHUSDT',50,2.5,18,-1)
S22+=ewz4('SOLUSDT',20,3.0,60,1)
S22+=z4('SOLUSDT',240,2.75,-1,'rl',24)
S22+=z4('XRPUSDT',240,4.0,-3,'tl',12)
S22+=z4('XRPUSDT',240,3.0,1.5,'rl',12)
S22+=fade4('XRPUSDT',12,0.15,60)
print(f'S4 {len(S4)}건 / S3 {len(S3)}건 / S22 {len(S22)}건',flush=True)

def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
def run_book(trades,frac):
    ev=[]
    for i,(en,ex,r) in enumerate(trades): ev.append((en,0,i)); ev.append((ex,1,i))
    ev.sort(key=lambda x:(x[0],x[1]))
    eq=10000.0; open_={}; qend={}; peak=eq; mdd=0.0
    for t,typ,i in ev:
        if typ==0: open_[i]=frac*eq
        else:
            no=open_.pop(i,None)
            if no is not None:
                eq+=no*trades[i][2]; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak)
        qend[qof(t)]=eq
    return qend,eq,mdd
q4,eq4,m4_=run_book(sorted(S4),FRAC)
qs,eqs,ms_=run_book(sorted(S22),FRAC)
qa,eqa,ma_=run_book(sorted(S22+S3),FRAC)
qb,eqb,mb_=run_book(sorted(S22+S3+S4),FRAC)
qs_all=sorted(set(q4)|set(qs))
print(f'\n{"분기":>7} | {"S4역추(6셀)":>10} | {"S22(8셀)":>10}')
prev4=prevs=10000.0
r4=[]; rs=[]
for q in qs_all:
    e4=q4.get(q,prev4); es=qs.get(q,prevs)
    d4=(e4/prev4-1)*100; ds=(es/prevs-1)*100
    r4.append(d4); rs.append(ds)
    print(f'{q:>7} | {d4:>+9.1f}% | {ds:>+9.1f}%')
    prev4,prevs=e4,es
r4=np.array(r4); rs=np.array(rs)
yrs=len(qs_all)/4
print(f'\nS4 단독: 최종 ${eq4:,.0f} ({eq4/10000:.2f}배, CAGR {((eq4/10000)**(1/yrs)-1)*100:+.1f}%) 실현MDD {m4_*100:.1f}% | 마이너스분기 {int((r4<-0.001).sum())}')
print(f'S22    : 최종 ${eqs:,.0f} | 상관 {np.corrcoef(r4,rs)[0,1]:.2f}')
print(f'\nS22+S3     : ${eqa:,.0f} (실현MDD {ma_*100:.1f}%)')
print(f'S22+S3+S4  : ${eqb:,.0f} (실현MDD {mb_*100:.1f}%) → S4 한계기여 {((eqb/eqa)-1)*100:+.1f}%')