# -*- coding: utf-8 -*-
"""S11 「1분봉책」 포트폴리오 시뮬 — $10,000 복리, 실현 기준.
Bybit북(11셀, 2%/진입, 2020-03~) / MT5북(11셀, 1%/진입, 2023-07~ 완전밀도 구간).
진입 노셔널 = 비율 × 당시 자산. 청산 시 실현. 연도별 자산·최대낙폭(실현)·최대노출 출력."""
import sys, os
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000

_cache={}
def arrs(sym):
    if sym not in _cache:
        m=_load_cache(sym,'1')
        _cache[sym]=(np.array([x['close'] for x in m]),np.array([x['high'] for x in m]),
                     np.array([x['low'] for x in m]),np.array([x['start'] for x in m],np.int64))
    return _cache[sym]

def zma(C,win):
    n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

def zcell(sym,win,k1,b,mode,mh_d,cdh,fee,use_sl=False):
    """mode: 'tl'(추세롱) 'rl'(역추롱) 'ts'(추세숏)"""
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
            r=xp/price-1-fee
        elif mode=='rl':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            TP=Hh[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); xp=tp
            else: ex=hi-1; xp=C[ex]
            r=xp/price-1-fee
        else:  # 추세숏 (SL유 대칭)
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            TP=L[ci+1:hi]<=tp; SL=(Hh[ci+1:hi]>=sl) if use_sl else np.zeros(hi-ci-1,bool)
            hit=SL|TP
            if hit.any():
                j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
            else: ex=hi-1; xp=C[ex]
            r=(price-xp)/price-fee
        out.append((int(ts[ci]),int(ts[ex]),r)); last=ts[ci]
    return out

def fadecell(sym,M,thr,cdh,exit_kind,fee,cap_h=48,Th=24):
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
        out.append((int(ts[ci]),int(ts[ex]),xp/price-1-fee)); last=ts[ci]
    return out

FEE=0.0011; FX=0.0002
BYBIT=[
 ('BTC-z추세롱', zcell('BTCUSDT',1440,6.0,0.0,'tl',14,3,FEE)),
 ('BTC-z역추롱', zcell('BTCUSDT',1320,5.0,-1.0,'rl',14,1,FEE)),
 ('BTC-페이드',  fadecell('BTCUSDT',60,0.04,0.5,'retr',FEE)),
 ('ETH-z추세롱', zcell('ETHUSDT',720,6.0,-3.0,'tl',14,3,FEE)),
 ('ETH-페이드',  fadecell('ETHUSDT',30,0.04,0.5,'time',FEE,Th=24)),
 ('SOL-z추세롱', zcell('SOLUSDT',1440,5.5,2.5,'tl',14,3,FEE)),
 ('SOL-페이드',  fadecell('SOLUSDT',15,0.05,0.5,'time',FEE,Th=24)),
 ('XRP-추세숏',  zcell('XRPUSDT',720,5.0,-0.5,'ts',14,1,FEE,use_sl=True)),
 ('XRP-페이드',  fadecell('XRPUSDT',30,0.05,0.5,'time',FEE,Th=24)),
 ('XAUT-z추세롱',zcell('XAUTUSDT',1440,4.0,-2.5,'tl',14,1,FEE)),
 ('XAUT-z역추롱',zcell('XAUTUSDT',1440,4.25,-3.0,'rl',14,3,FEE)),
]
def build_mt5():
    return [
     ('JP225-z추세롱', zcell('JP225',1440,4.0,-1.5,'tl',14,3,FEE)),
     ('US100-z추세롱', zcell('US100',720,5.25,-2.5,'tl',14,1,FEE)),
     ('GER40-z추세롱', zcell('GER40',1440,3.75,-3.0,'tl',14,3,FEE)),
     ('UK100-z추세롱', zcell('UK100',1440,3.75,-3.0,'tl',14,3,FEE)),
     ('HK50-z추세롱',  zcell('HK50',360,5.75,-3.0,'tl',14,1,FEE)),
     ('XAGUSD-z추세롱',zcell('XAGUSD',1320,4.75,-2.5,'tl',14,1,FEE)),
     ('WTI-z추세롱',   zcell('WTI',720,4.5,-2.0,'tl',14,3,FEE)),
     ('JP225-페이드',  fadecell('JP225',240,0.03,0.5,'time',FEE,Th=24)),
     ('HK50-페이드',   fadecell('HK50',120,0.02,0.5,'time',FEE,Th=72)),
     ('USDJPY-z추세롱',zcell('USDJPY',1440,4.5,-2.5,'tl',14,1,FX)),
     ('USDJPY-페이드', fadecell('USDJPY',120,0.01,0.5,'time',FX,Th=48)),
    ]
def yof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc)
    return f'{t.year}-{t.month:02d}' if os.getenv('MONTHLY') else t.year

def portfolio(name,cells,frac,start_ms=None):
    trades=[]
    for tag,tr in cells:
        for en,ex,r in tr:
            if start_ms and en<start_ms: continue
            trades.append((en,ex,r,tag))
    ev=[]
    for i,(en,ex,r,tag) in enumerate(trades): ev.append((en,0,i)); ev.append((ex,1,i))
    ev.sort(key=lambda x:(x[0],x[1]))
    INIT=float(os.getenv('INIT','10000'))
    eq=INIT; open_={}; tot_no=0.0; max_expo=0.0; peak=eq; mdd=0.0
    yeq={}
    for t,typ,i in ev:
        if typ==0:
            no=frac*eq; open_[i]=no; tot_no+=no
            max_expo=max(max_expo,tot_no/eq)
        else:
            no=open_.pop(i,None)
            if no is None: continue
            eq+=no*trades[i][2]; tot_no-=no
            peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak)
            yeq[yof(trades[i][0])]=eq
    n=len(trades)
    yrs=(max(e[0] for e in ev)-min(e[0] for e in ev))/(365.25*DAY)
    print(f'===== {name} (진입 {frac*100:.0f}%/자산, ${INIT:,.0f} 시작) =====')
    for y in sorted(yeq): print(f'  {y}: ${yeq[y]:>10,.0f}')
    print(f'  최종 ${eq:,.0f} ({eq/INIT:.2f}배, {yrs:.1f}년, CAGR {((eq/INIT)**(1/yrs)-1)*100:+.1f}%)')
    print(f'  거래 {n}건 | 실현기준 최대낙폭 {mdd*100:.1f}% | 최대 동시노출 {max_expo*100:.0f}%')

BSTART=os.getenv('BSTART')
bs=int(datetime.fromisoformat(BSTART).replace(tzinfo=timezone.utc).timestamp()*1000) if BSTART else None
portfolio('Bybit S11 (5심볼 11셀)',BYBIT,float(os.getenv('FRAC','0.02')),start_ms=bs)
print()
if os.getenv('SKIP_MT5')!='1':
    MT5_START=int(datetime(2023,7,1,tzinfo=timezone.utc).timestamp()*1000)
    portfolio('MT5 S11 확장 (8심볼 11셀, 2023-07~ 완전밀도 구간)',build_mt5(),0.01,start_ms=MT5_START)
