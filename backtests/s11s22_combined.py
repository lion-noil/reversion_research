# -*- coding: utf-8 -*-
"""S11(1분봉책) + S22(4h책) 동시운용 합산 시뮬 — 한 계좌 $10k, 각 북 FRAC 5%.
시간단위(1h) 장중최악 MTM: 분기별 자산·수익·최대중첩(합/북별)·최대노출·MTM최저·낙폭 + 청산 판정."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000
F11=float(os.getenv('F11','0.05')); F22=float(os.getenv('F22','0.05'))

# ---------- 1m 데이터 (S11 + MTM 마크) ----------
_m1={}
def m1(sym):
    if sym not in _m1:
        m=_load_cache(sym,'1')
        _m1[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                  np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _m1[sym]
_m4={}
def m4(sym):
    if sym not in _m4:
        m=_load_cache(sym,'240')
        _m4[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                  np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _m4[sym]
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

# ---------- S11 셀 (1m, s11_mtm.py와 동일) ----------
def zcell1(sym,win,k1,b,mode,mh_d,cdh,use_sl=False):
    C,Hh,L,ts=m1(sym); n=len(C)
    z,ma,sd=zma(C,win)
    cand=np.flatnonzero(z>=k1) if mode=='tl' else np.flatnonzero(z<=-k1)
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
            out.append((int(ts[ci]),int(ts[ex]),xp/price-1-FEE,sym,1,float(price))); last=ts[ci]
        elif mode=='rl':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            TP=Hh[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); xp=tp
            else: ex=hi-1; xp=C[ex]
            out.append((int(ts[ci]),int(ts[ex]),xp/price-1-FEE,sym,1,float(price))); last=ts[ci]
        else:
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            TPl=L[ci+1:hi]<=tp; SL=(Hh[ci+1:hi]>=sl) if use_sl else np.zeros(hi-ci-1,bool)
            hit=SL|TPl
            if hit.any():
                j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
            else: ex=hi-1; xp=C[ex]
            out.append((int(ts[ci]),int(ts[ex]),(price-xp)/price-FEE,sym,-1,float(price))); last=ts[ci]
    return out
def fadecell1(sym,M,thr,cdh,exit_kind,cap_h=48,Th=24):
    C,Hh,L,ts=m1(sym); n=len(C)
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    cand=np.flatnonzero(ret<=-thr); out=[]; last=None; cd=int(cdh*H)
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd: continue
        price=C[ci]
        if exit_kind=='retr':
            tp=price*(1+1.5*abs(ret[ci])); hi=int(np.searchsorted(ts,ts[ci]+cap_h*H,'right'))
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

# ---------- S22 셀 (4h) ----------
HOLD=15*DAY
def zcell4(sym,win,k,b,mode,cdh):
    C,Hh,L,ts=m4(sym); n=len(C)
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
        if TP.any(): ex=ci+1+int(np.argmax(TP)); r=tp/price-1-FEE
        else: ex=hi-1; r=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),r,sym,1,float(price))); last=ts[ci]
    return out
def ewzcell4(sym,s,k,hb,side):
    C,Hh,L,ts=m4(sym); n=len(C)
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
def fadecell4(sym,M,thr,kind,cap):
    C,Hh,L,ts=m4(sym); n=len(C)
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

s11=[]
s11+=zcell1('BTCUSDT',1440,6.0,0.0,'tl',14,3)
s11+=zcell1('BTCUSDT',1320,5.0,-1.0,'rl',14,1)
s11+=fadecell1('BTCUSDT',60,0.04,0.5,'retr')
s11+=zcell1('ETHUSDT',720,6.0,-3.0,'tl',14,3)
s11+=fadecell1('ETHUSDT',30,0.04,0.5,'time',Th=24)
s11+=zcell1('SOLUSDT',1440,5.5,2.5,'tl',14,3)
s11+=fadecell1('SOLUSDT',15,0.05,0.5,'time',Th=24)
s11+=zcell1('XRPUSDT',720,5.0,-0.5,'ts',14,1,use_sl=True)
s11+=fadecell1('XRPUSDT',30,0.05,0.5,'time',Th=24)
s11+=zcell1('XAUTUSDT',1440,4.0,-2.5,'tl',14,1)
s11+=zcell1('XAUTUSDT',1440,4.25,-3.0,'rl',14,3)
s22=[]   # 라이브 v1 8셀 (견고성 통과분)
s22+=zcell4('BTCUSDT',240,3.25,-2.5,'tl',24)
s22+=ewzcell4('ETHUSDT',200,3.0,90,1)
s22+=ewzcell4('ETHUSDT',50,2.5,18,-1)
s22+=ewzcell4('SOLUSDT',20,3.0,60,1)
s22+=zcell4('SOLUSDT',240,2.75,-1,'rl',24)
s22+=zcell4('XRPUSDT',240,4.0,-3,'tl',12)
s22+=zcell4('XRPUSDT',240,3.0,1.5,'rl',12)
s22+=fadecell4('XRPUSDT',12,0.15,'R',60)
print(f'S11 {len(s11)}건 / S22 {len(s22)}건',flush=True)

trades=[(en,ex,r,sym,side,ep,0) for en,ex,r,sym,side,ep in s11]+[(en,ex,r,sym,side,ep,1) for en,ex,r,sym,side,ep in s22]
trades.sort()
FR=[F11,F22]

# 시간단위 저가/고가 마크 (1m 기반)
HL={}
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','XAUTUSDT']:
    C,Hh,L,ts=m1(sym)
    hr=ts//H
    uh,idx=np.unique(hr,return_index=True)
    lo=np.minimum.reduceat(L,idx); hi_=np.maximum.reduceat(Hh,idx)
    HL[sym]=(uh,lo,hi_)
def mark(sym,hour,side):
    uh,lo,hi_=HL[sym]
    i=int(np.searchsorted(uh,hour))
    if i>=len(uh) or uh[i]!=hour: i=min(max(i-1,0),len(uh)-1)
    return lo[i] if side>0 else hi_[i]

ev=[]
for i,t in enumerate(trades): ev.append((t[0],0,i)); ev.append((t[1],1,i))
ev.sort(key=lambda x:(x[0],x[1]))
eq=10000.0; open_={}; expo=0.0; ptr=0
t0=min(e[0] for e in ev)//H; t1=max(e[0] for e in ev)//H
def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
Q=defaultdict(lambda: dict(maxc=0,mc11=0,mc22=0,maxexpo=0.0,minmtm=1e18,peak=-1e18,mdd=0.0,eq_end=0.0))
peak=-1e18; mdd=0.0; min_mg=9e9; liq=None
for hour in range(t0,t1+1):
    tms=(hour+1)*H
    while ptr<len(ev) and ev[ptr][0]<tms:
        t,typ,i=ev[ptr]; ptr+=1
        if typ==0:
            no=FR[trades[i][6]]*eq; open_[i]=no; expo+=no
        else:
            no=open_.pop(i,None)
            if no is not None: eq+=no*trades[i][2]; expo-=no
    un=0.0; c11=0; c22=0
    for i,no in open_.items():
        en,ex,r,sym,side,ep,bk=trades[i]
        m=mark(sym,hour,side)
        un+=no*side*(m/ep-1.0)
        if bk==0: c11+=1
        else: c22+=1
    mtm=eq+un
    q=qof(hour*H); d=Q[q]
    if c11+c22>d['maxc']: d['maxc']=c11+c22; d['mc11']=c11; d['mc22']=c22
    if eq>0: d['maxexpo']=max(d['maxexpo'],expo/eq)
    d['minmtm']=min(d['minmtm'],mtm); d['eq_end']=eq
    d['peak']=max(d['peak'],mtm); d['mdd']=max(d['mdd'],(d['peak']-mtm)/d['peak'] if d['peak']>0 else 0)
    peak=max(peak,mtm); mdd=max(mdd,(peak-mtm)/peak if peak>0 else 0)
    if expo>0:
        mg=mtm/expo
        if mg<min_mg: min_mg=mg
        if mg<0.005 and liq is None: liq=hour
print(f'\n{"분기":>7} | {"실현자산":>10} | {"분기수익":>7} | {"중첩(11+22)":>11} | {"노출":>5} | {"MTM최저":>10} | {"낙폭":>5}')
prev=10000.0
for q in sorted(Q):
    d=Q[q]
    ret=(d['eq_end']/prev-1)*100 if prev>0 else 0
    print(f'{q:>7} | ${d["eq_end"]:>9,.0f} | {ret:>+6.1f}% | {d["maxc"]:>3} ({d["mc11"]}+{d["mc22"]}) | {d["maxexpo"]*100:>4.0f}% | ${d["minmtm"]:>9,.0f} | {d["mdd"]*100:>4.1f}%')
    prev=d['eq_end']
yrs=(t1-t0)*H/(365.25*DAY)
print(f'\n최종 ${eq:,.0f} ({eq/10000:.2f}배, {yrs:.1f}년, CAGR {((eq/10000)**(1/yrs)-1)*100:+.1f}%) | S11 {F11*100:.0f}% + S22 {F22*100:.0f}%')
print(f'전체 MTM 최대낙폭 {mdd*100:.1f}% | 최저 증거금율 {min_mg*100:.0f}%')
print('청산 판정: '+('🔴 발생 '+datetime.fromtimestamp(liq*H/1000,tz=timezone.utc).strftime('%Y-%m-%d') if liq else f'🟢 불가능 수준 (유지증거금 0.5% 대비 여유 {min_mg/0.005:.0f}배)'))
