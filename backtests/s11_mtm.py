# -*- coding: utf-8 -*-
"""S11 Bybit 포트폴리오 — MTM 정밀 검증.
시간단위 마크(보유중 포지션을 그 시간의 장중최악 저가/고가로 평가) 기준:
연도별 최대중첩·최대노출·MTM최저자산·MTM최대낙폭·최저증거금율(자산/노출) + 청산 판정.
진입 2%/자산(당시), $10,000, fee 포함."""
import sys, os
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011

_c={}
def arrs(sym):
    if sym not in _c:
        m=_load_cache(sym,'1')
        _c[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                 np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _c[sym]
def zma(C,win):
    n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

def zcell(sym,win,k1,b,mode,mh_d,cdh,use_sl=False):
    C,Hh,L,ts=arrs(sym); n=len(C)
    z,ma,sd=zma(C,win)
    epos = mode=='tl'
    cand=np.flatnonzero(z>=k1) if epos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=win]
    out=[]; last=None; cd=int(cdh*H); mh=int(mh_d*DAY)
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+mh,'right'))
        if hi<=ci+1: continue
        if mode=='tl':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            tp=price*(1+(1-tpb/price)); TP=Hh[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); xp=tp
            else: ex=hi-1; xp=C[ex]
            r=xp/price-1-FEE; side=1
        elif mode=='rl':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            TP=Hh[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); xp=tp
            else: ex=hi-1; xp=C[ex]
            r=xp/price-1-FEE; side=1
        else:
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            TP=L[ci+1:hi]<=tp; SL=(Hh[ci+1:hi]>=sl) if use_sl else np.zeros(hi-ci-1,bool)
            hit=SL|TP
            if hit.any():
                j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
            else: ex=hi-1; xp=C[ex]
            r=(price-xp)/price-FEE; side=-1
        out.append((int(ts[ci]),int(ts[ex]),r,sym,side,float(price))); last=ts[ci]
    return out

def fadecell(sym,M,thr,cdh,exit_kind,cap_h=48,Th=24):
    C,Hh,L,ts=arrs(sym); n=len(C)
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    cand=np.flatnonzero(ret<=-thr)
    out=[]; last=None; cd=int(cdh*H)
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd: continue
        price=C[ci]
        if exit_kind=='retr':
            tp=price*(1+1.5*abs(ret[ci]))
            hi=int(np.searchsorted(ts,ts[ci]+cap_h*H,'right'))
            if hi<=ci+1: continue
            TP=Hh[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); xp=tp
            else: ex=hi-1; xp=C[ex]
        else:
            ex=int(np.searchsorted(ts,ts[ci]+Th*H,'right'))-1
            if ex<=ci: continue
            xp=C[ex]
        out.append((int(ts[ci]),int(ts[ex]),xp/price-1-FEE,sym,1,float(price))); last=ts[ci]
    return out

cells=[]
cells+=zcell('BTCUSDT',1440,6.0,0.0,'tl',14,3)
cells+=zcell('BTCUSDT',1320,5.0,-1.0,'rl',14,1)
cells+=fadecell('BTCUSDT',60,0.04,0.5,'retr')
cells+=zcell('ETHUSDT',720,6.0,-3.0,'tl',14,3)
cells+=fadecell('ETHUSDT',30,0.04,0.5,'time',Th=24)
cells+=zcell('SOLUSDT',1440,5.5,2.5,'tl',14,3)
cells+=fadecell('SOLUSDT',15,0.05,0.5,'time',Th=24)
cells+=zcell('XRPUSDT',720,5.0,-0.5,'ts',14,1,use_sl=True)
cells+=fadecell('XRPUSDT',30,0.05,0.5,'time',Th=24)
cells+=zcell('XAUTUSDT',1440,4.0,-2.5,'tl',14,1)
cells+=zcell('XAUTUSDT',1440,4.25,-3.0,'rl',14,3)
print(f'총 트레이드 {len(cells)}',flush=True)

# 심볼별 시간단위 저가/고가 (장중최악 마크용)
HL={}
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','XAUTUSDT']:
    C,Hh,L,ts=arrs(sym)
    hr=ts//H
    uh,idx=np.unique(hr,return_index=True)
    lo=np.minimum.reduceat(L,idx); hi_=np.maximum.reduceat(Hh,idx)
    HL[sym]=(uh,lo,hi_)
def mark(sym,hour,side):
    uh,lo,hi_=HL[sym]
    i=int(np.searchsorted(uh,hour))
    if i>=len(uh) or uh[i]!=hour:
        i=min(max(i-1,0),len(uh)-1)
    return lo[i] if side>0 else hi_[i]

trades=sorted(cells)
FRAC=float(os.getenv('FRAC','0.02'))
ev=[]
for i,(en,ex,r,sym,side,ep) in enumerate(trades): ev.append((en,0,i)); ev.append((ex,1,i))
ev.sort(key=lambda x:(x[0],x[1]))
eq=10000.0; open_={}; expo=0.0
t0=min(e[0] for e in ev)//H; t1=max(e[0] for e in ev)//H
ptr=0
def yof(h): return datetime.fromtimestamp(h*3600,tz=timezone.utc).year
Y={}
peak=-1e18; mdd=0.0; min_mg=9e9; liq=None
for hour in range(t0,t1+1):
    tms=(hour+1)*H
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
        m=mark(sym,hour,side)
        un+=no*side*(m/ep-1.0)
    mtm=eq+un
    y=yof(hour)
    if y not in Y: Y[y]=dict(maxc=0,maxexpo=0.0,minmtm=1e18,peak=-1e18,mdd=0.0)
    d=Y[y]
    d['maxc']=max(d['maxc'],len(open_))
    if eq>0: d['maxexpo']=max(d['maxexpo'],expo/eq)
    d['minmtm']=min(d['minmtm'],mtm)
    d['peak']=max(d['peak'],mtm); d['mdd']=max(d['mdd'],(d['peak']-mtm)/d['peak'] if d['peak']>0 else 0)
    peak=max(peak,mtm); mdd=max(mdd,(peak-mtm)/peak if peak>0 else 0)
    if expo>0:
        mg=mtm/expo
        if mg<min_mg: min_mg=mg
        if mg<0.005 and liq is None: liq=hour   # 유지증거금 0.5% (Bybit 무기한 최소구간)
print(f'\n{"연도":>5} | {"최대중첩":>6} | {"최대노출":>7} | {"MTM최저자산":>12} | {"MTM최대낙폭":>10}')
for y in sorted(Y):
    d=Y[y]
    print(f'{y:>5} | {d["maxc"]:>6} | {d["maxexpo"]*100:>6.0f}% | ${d["minmtm"]:>10,.0f} | {d["mdd"]*100:>9.1f}%')
print(f'\n최종 ${eq:,.0f} | 전체 MTM 최대낙폭 {mdd*100:.1f}% | 최저 증거금율(자산/노출) {min_mg*100:.0f}%')
print('청산 판정: '+('🔴 발생' if liq else '🟢 불가능 수준 (노출<자산, 무레버리지 구조 — Bybit 유지증거금 0.5% 대비 여유 '+f'{min_mg/0.005:.0f}배)'))
