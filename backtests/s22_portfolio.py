# -*- coding: utf-8 -*-
"""S22 Bybit 11셀 포트폴리오 — $10k 복리(실현) + 4h봉 MTM(장중최악) 검증.
분기별 자산·수익률 + 연도별 최대중첩·최대노출·MTM최저·낙폭 + 청산 판정.
진입 FRAC(기본 2%)/자산, fee 0.11% 포함(셀 수익률에 내장)."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000; HOLD=15*DAY
FRAC=float(os.getenv('FRAC','0.02'))

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
def zarr(C,win):
    n=len(C); cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

def zcell(sym,win,k,b,mode,cdh):
    C,Hh,L,ts=arrs(sym); n=len(C)
    z,ma,sd=zarr(C,win)
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
        if TP.any(): ex=ci+1+int(np.argmax(TP)); r=tp/price-1-FEE
        else: ex=hi-1; r=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),r,sym,1,float(price))); last=ts[ci]
    return out

def ewzcell(sym,s,k,hb,side):
    C,Hh,L,ts=arrs(sym); n=len(C)
    mu=ema(C,s); resid=C-mu; sig=ema(np.abs(resid),s)
    z=resid/np.where(sig==0,1e-12,sig)
    cand=np.flatnonzero(z>=k) if side>0 else np.flatnonzero(z<=-k)
    out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        ex=min(ci+hb,n-1)
        out.append((int(ts[ci]),int(ts[ex]),side*(C[ex]/C[ci]-1)-FEE,sym,side,float(C[ci]))); last=ci
    return out

def fadecell(sym,M,thr,kind,cap):
    C,Hh,L,ts=arrs(sym); n=len(C)
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    cand=np.flatnonzero(ret<=-thr); out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        price=C[ci]; hi=min(ci+1+cap,n)
        if kind=='R':
            tp=price*(1+1.5*abs(ret[ci])); TP=Hh[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); r=tp/price-1-FEE
            else: ex=hi-1; r=C[ex]/price-1-FEE
        else:
            ex=hi-1; r=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),r,sym,1,float(price))); last=ci
    return out

cells=[]
LIVE=os.getenv('LIVE')=='1'   # LIVE=1: 견고성 검증 통과 8셀만 (라이브 v1)
cells+=zcell('BTCUSDT',240,3.25,-2.5,'tl',24)      # BTC z추세롱 40d
if not LIVE: cells+=fadecell('BTCUSDT',6,0.10,'T',6)   # BTC 페이드 — 스파이크, 라이브 제외
if not LIVE: cells+=ewzcell('BTCUSDT',50,3.0,18,-1)    # BTC ewz추숏 — 취약, 라이브 제외
cells+=ewzcell('ETHUSDT',200,3.0,90,1)             # ETH ewz추롱
if not LIVE: cells+=fadecell('ETHUSDT',12,0.15,'T',6)  # ETH 페이드 — 스파이크, 라이브 제외
cells+=ewzcell('ETHUSDT',50,2.5,18,-1)             # ETH ewz추숏 (22년 헤지)
cells+=ewzcell('SOLUSDT',20,3.0,60,1)              # SOL ewz추롱
cells+=zcell('SOLUSDT',240,2.75,-1,'rl',24)        # SOL z역추롱 40d
cells+=zcell('XRPUSDT',240,4.0,-3,'tl',12)         # XRP z추세롱 40d
cells+=zcell('XRPUSDT',240,3.0,1.5,'rl',12)        # XRP z역추롱 40d
cells+=fadecell('XRPUSDT',12,0.15,'R',60)          # XRP 페이드 48h-15% R10d
print(f'총 트레이드 {len(cells)}',flush=True)

# 심볼별 4h 저가/고가 마크 테이블
MK={}
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT']:
    C,Hh,L,ts=arrs(sym)
    MK[sym]={int(t):(float(l),float(h)) for t,l,h in zip(ts,L,Hh)}
def mark(sym,slot,side,fallback):
    v=MK[sym].get(slot)
    if v is None: return fallback
    return v[0] if side>0 else v[1]

trades=sorted(cells)
ev=[]
for i,(en,ex,r,sym,side,ep) in enumerate(trades): ev.append((en,0,i)); ev.append((ex,1,i))
ev.sort(key=lambda x:(x[0],x[1]))
eq=10000.0; open_={}; expo=0.0; ptr=0
BAR=4*H
t0=min(e[0] for e in ev)//BAR; t1=max(e[0] for e in ev)//BAR
def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
Q=defaultdict(lambda: dict(maxc=0,maxexpo=0.0,minmtm=1e18,peak=-1e18,mdd=0.0,eq_end=0.0))
peak=-1e18; mdd=0.0; min_mg=9e9; liq=None
for slot in range(t0,t1+1):
    tms=(slot+1)*BAR
    while ptr<len(ev) and ev[ptr][0]<tms:
        t,typ,i=ev[ptr]; ptr+=1
        if typ==0:
            no=FRAC*eq; open_[i]=no; expo+=no
        else:
            no=open_.pop(i,None)
            if no is not None: eq+=no*trades[i][2]; expo-=no
    un=0.0
    for i,no in open_.items():
        en,ex,r,sym,side,ep=trades[i]
        m=mark(sym,slot*BAR,side,ep)
        un+=no*side*(m/ep-1.0)
    mtm=eq+un
    q=qof(slot*BAR); d=Q[q]
    d['maxc']=max(d['maxc'],len(open_))
    if eq>0: d['maxexpo']=max(d['maxexpo'],expo/eq)
    d['minmtm']=min(d['minmtm'],mtm); d['eq_end']=eq
    d['peak']=max(d['peak'],mtm); d['mdd']=max(d['mdd'],(d['peak']-mtm)/d['peak'] if d['peak']>0 else 0)
    peak=max(peak,mtm); mdd=max(mdd,(peak-mtm)/peak if peak>0 else 0)
    if expo>0:
        mg=mtm/expo
        if mg<min_mg: min_mg=mg
        if mg<0.005 and liq is None: liq=slot
print(f'\n{"분기":>7} | {"실현자산":>10} | {"분기수익":>7} | {"최대중첩":>4} | {"최대노출":>5} | {"MTM최저":>10} | {"분기낙폭":>5}')
prev=10000.0
for q in sorted(Q):
    d=Q[q]
    ret=(d['eq_end']/prev-1)*100 if prev>0 else 0
    print(f'{q:>7} | ${d["eq_end"]:>9,.0f} | {ret:>+6.1f}% | {d["maxc"]:>4} | {d["maxexpo"]*100:>4.0f}% | ${d["minmtm"]:>9,.0f} | {d["mdd"]*100:>4.1f}%')
    prev=d['eq_end']
yrs=(t1-t0)*BAR/ (365.25*DAY)
print(f'\n최종 ${eq:,.0f} ({eq/10000:.2f}배, {yrs:.1f}년, CAGR {((eq/10000)**(1/yrs)-1)*100:+.1f}%) | 진입 {FRAC*100:.0f}%/자산')
print(f'전체 MTM 최대낙폭 {mdd*100:.1f}% | 최저 증거금율(MTM자산/노출) {min_mg*100:.0f}%')
print('청산 판정: '+('🔴 발생 '+datetime.fromtimestamp(liq*BAR/1000,tz=timezone.utc).strftime('%Y-%m-%d') if liq else f'🟢 불가능 수준 (Bybit 유지증거금 0.5% 대비 여유 {min_mg/0.005:.0f}배)'))
