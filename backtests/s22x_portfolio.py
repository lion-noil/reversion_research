# -*- coding: utf-8 -*-
"""S22 확장판(MT5·FX) 17셀 포트폴리오 — $10k 복리(실현) + 4h MTM(장중최악) + 셀간 중복도.
견고성 통과 셀만. fee: 비FX 0.11%/FX 0.02%. FRAC env(기본 1% — MT5 보수 사이징).
연도별 + 최종: 중첩·노출·MTM최저·낙폭·증거금·청산 판정."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; HOLD=15*DAY
FX={'USDJPY','EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','NZDUSD'}
FRAC=float(os.getenv('FRAC','0.01'))

_c={}
def arrs(sym):
    if sym not in _c:
        m=_load_cache(sym,'240')
        _c[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                 np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _c[sym]
def fee_of(sym): return 0.0002 if sym in FX else 0.0011
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

def zcell(tag,sym,win,k,b,mode,cdh):
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
            if TP.any(): ex=ci+1+int(np.argmax(TP)); r=tp/price-1-fee
            else: ex=hi-1; r=C[ex]/price-1-fee
            side=1
        elif mode=='rl':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            TP=Hh[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); r=tp/price-1-fee
            else: ex=hi-1; r=C[ex]/price-1-fee
            side=1
        elif mode=='ts':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            TPh=L[ci+1:hi]<=tp; SL=Hh[ci+1:hi]>=sl; hit=TPh|SL
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
            else: ex=hi-1; xp=C[ex]
            r=(price-xp)/price-fee; side=-1
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sl=price*(1+pct)
            TPh=L[ci+1:hi]<=tp; SL=Hh[ci+1:hi]>=sl; hit=TPh|SL
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
            else: ex=hi-1; xp=C[ex]
            r=(price-xp)/price-fee; side=-1
        out.append((int(ts[ci]),int(ts[ex]),r,sym,side,float(price),tag)); last=ts[ci]
    return out
def ewzcell(tag,sym,s,k,hb,side,trig):
    C,Hh,L,ts=arrs(sym); n=len(C); fee=fee_of(sym)
    mu=ema(C,s); resid=C-mu; sig=ema(np.abs(resid),s)
    z=resid/np.where(sig==0,1e-12,sig)
    cand=np.flatnonzero(z>=k) if trig=='hi' else np.flatnonzero(z<=-k)
    out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        ex=min(ci+hb,n-1)
        out.append((int(ts[ci]),int(ts[ex]),side*(C[ex]/C[ci]-1)-fee,sym,side,float(C[ci]),tag)); last=ci
    return out
def fadecell(tag,sym,M,thr,kind,cap):
    C,Hh,L,ts=arrs(sym); n=len(C); fee=fee_of(sym)
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    cand=np.flatnonzero(ret<=-thr); out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        price=C[ci]; hi=min(ci+1+cap,n)
        if kind=='R':
            tp=price*(1+1.5*abs(ret[ci])); TP=Hh[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); r=tp/price-1-fee
            else: ex=hi-1; r=C[ex]/price-1-fee
        else:
            ex=hi-1; r=C[ex]/price-1-fee
        out.append((int(ts[ci]),int(ts[ex]),r,sym,1,float(price),tag)); last=ci
    return out
def sweepcell(tag,sym,N,hb):
    C,Hh,L,ts=arrs(sym); n=len(C); fee=fee_of(sym)
    pl=np.full(n,np.nan)
    for i in range(N,n): pl[i]=L[i-N:i].min()
    cand=np.flatnonzero((L<pl)&(C>pl)); out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        ex=min(ci+hb,n-1)
        out.append((int(ts[ci]),int(ts[ex]),C[ex]/C[ci]-1-fee,sym,1,float(C[ci]),tag)); last=ci
    return out

CELLS=[]
CELLS+=zcell('JP225-z추롱','JP225',120,3.0,1,'tl',12)
CELLS+=sweepcell('JP225-스윕','JP225',120,60)
CELLS+=fadecell('JP225-페이드','JP225',18,0.05,'T',60)
CELLS+=zcell('US100-z역롱','US100',180,3.0,1,'rl',12)
CELLS+=fadecell('US100-페이드','US100',30,0.07,'R',30)
CELLS+=zcell('HK50-z추롱','HK50',180,3.0,-2,'tl',12)
CELLS+=zcell('HK50-z역롱','HK50',180,3.0,1,'rl',12)
CELLS+=fadecell('UK100-페이드','UK100',12,0.03,'T',60)
CELLS+=ewzcell('XAGUSD-ewz역롱','XAGUSD',200,2.5,30,1,'lo')
CELLS+=ewzcell('XAGUSD-ewz추숏','XAGUSD',50,3.0,18,-1,'lo')
CELLS+=fadecell('XAUUSD-페이드','XAUUSD',6,0.03,'T',60)
CELLS+=fadecell('WTI-페이드','WTI',6,0.07,'T',60)     # ⚠️이벤트 몰빵 — 사이징 절반 권고
CELLS+=zcell('USDJPY-z역롱','USDJPY',120,2.5,1,'rl',12)
CELLS+=fadecell('USDJPY-페이드','USDJPY',18,0.03,'T',30)
CELLS+=zcell('AUDUSD-z역숏','AUDUSD',30,3.0,-2,'rs',12)
CELLS+=zcell('NZDUSD-z역숏','NZDUSD',60,3.0,-1,'rs',24)
CELLS+=sweepcell('USDCAD-스윕','USDCAD',120,60)
print(f'총 트레이드 {len(CELLS)}',flush=True)

# ── 셀간 신호 중복도 (24h) — 태그별 진입시각 ──
bytag=defaultdict(list)
for en,ex,r,sym,side,ep,tag in CELLS: bytag[tag].append(en)
tags=sorted(bytag)
print('\n=== 중복도 50%↑ 쌍 (행 진입의 24h내 열 진입) ===')
W=24*H
for i,a in enumerate(tags):
    ea=np.array(sorted(bytag[a]))
    for j,b in enumerate(tags):
        if i==j: continue
        eb=np.array(sorted(bytag[b]))
        idx=np.searchsorted(eb,ea)
        near=0
        for x,ii in zip(ea,idx):
            cds=[]
            if ii<len(eb): cds.append(abs(int(eb[ii])-int(x)))
            if ii>0: cds.append(abs(int(eb[ii-1])-int(x)))
            if cds and min(cds)<=W: near+=1
        pct=100*near/len(ea)
        if pct>=50: print(f'  {a} → {b}: {pct:.0f}%')

# ── 포트폴리오 (실현 복리 + 4h MTM) ──
MK={}
for sym in set(x[3] for x in CELLS):
    C,Hh,L,ts=arrs(sym)
    MK[sym]={int(t):(float(l),float(h)) for t,l,h in zip(ts,L,Hh)}
def mark(sym,slot,side,fallback):
    v=MK[sym].get(slot)
    if v is None: return fallback
    return v[0] if side>0 else v[1]
trades=sorted(CELLS)
ev=[]
for i,t in enumerate(trades): ev.append((t[0],0,i)); ev.append((t[1],1,i))
ev.sort(key=lambda x:(x[0],x[1]))
eq=10000.0; open_={}; expo=0.0; ptr=0
BAR=4*H
t0=min(e[0] for e in ev)//BAR; t1=max(e[0] for e in ev)//BAR
def yof(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
Y=defaultdict(lambda: dict(maxc=0,maxexpo=0.0,minmtm=1e18,peak=-1e18,mdd=0.0,eq_end=0.0))
peak=-1e18; mdd=0.0; min_mg=9e9; liq=None
for slot in range(t0,t1+1):
    tms=(slot+1)*BAR
    while ptr<len(ev) and ev[ptr][0]<tms:
        t,typ,i=ev[ptr]; ptr+=1
        if typ==0:
            no=FRAC*eq
            if trades[i][6]=='WTI-페이드': no*=0.5   # 이벤트 몰빵 셀 절반
            open_[i]=no; expo+=no
        else:
            no=open_.pop(i,None)
            if no is not None: eq+=no*trades[i][2]; expo-=no
    un=0.0
    for i,no in open_.items():
        en,ex,r,sym,side,ep,tag=trades[i]
        m=mark(sym,slot*BAR,side,ep)
        un+=no*side*(m/ep-1.0)
    mtm=eq+un
    y=yof(slot*BAR); d=Y[y]
    d['maxc']=max(d['maxc'],len(open_))
    if eq>0: d['maxexpo']=max(d['maxexpo'],expo/eq)
    d['minmtm']=min(d['minmtm'],mtm); d['eq_end']=eq
    d['peak']=max(d['peak'],mtm); d['mdd']=max(d['mdd'],(d['peak']-mtm)/d['peak'] if d['peak']>0 else 0)
    peak=max(peak,mtm); mdd=max(mdd,(peak-mtm)/peak if peak>0 else 0)
    if expo>0:
        mg=mtm/expo
        if mg<min_mg: min_mg=mg
        if mg<0.01 and liq is None: liq=slot
print(f'\n{"연도":>5} | {"자산":>10} | {"최대중첩":>4} | {"최대노출":>5} | {"MTM최저":>10} | {"낙폭":>5}')
prev=10000.0
for y in sorted(Y):
    d=Y[y]
    print(f'{y:>5} | ${d["eq_end"]:>9,.0f} | {d["maxc"]:>4} | {d["maxexpo"]*100:>4.0f}% | ${d["minmtm"]:>9,.0f} | {d["mdd"]*100:>4.1f}%')
    prev=d['eq_end']
yrs=(t1-t0)*BAR/(365.25*DAY)
print(f'\n최종 ${eq:,.0f} ({eq/10000:.2f}배, {yrs:.1f}년, CAGR {((eq/10000)**(1/yrs)-1)*100:+.1f}%) | 진입 {FRAC*100:.1f}%/자산')
print(f'전체 MTM 최대낙폭 {mdd*100:.1f}% | 최저 증거금율 {min_mg*100:.0f}%')
print('청산 판정: '+('🔴' if liq else f'🟢 불가능 수준 (여유 {min_mg/0.01:.0f}배 vs 유지증거금 1%)'))
