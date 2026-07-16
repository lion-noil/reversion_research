# -*- coding: utf-8 -*-
"""BTCUSDT 1분봉 게임: ①월 ~20회 진입되는 K1 탐색(게이트 없음) → ②동일 설정에 5분 레짐 게이트 사다리.
게임: z1 ±K1 진입, TP/SL ±pct(B, w1앵커) 손절우선, 쿨다운 1h, 보유 2일, fee 0.11%/왕복."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011; MH=2*DAY

def zma(C,win):
    n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

m1=_load_cache('BTCUSDT','1'); m5=_load_cache('BTCUSDT','5')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
C5=np.array([x['close'] for x in m5]); ts5=np.array([x['start'] for x in m5],np.int64)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
Z5={}
for w in [288,864,2016]:
    z5,_,_=zma(C5,w)
    idx=np.searchsorted(ts5+4*60000, ts1,'right')-1; idx=np.clip(idx,0,len(z5)-1)
    v=z5[idx]; v[idx<0]=np.nan; Z5[w]=v

def sim(DIR,w1,k1,b,cd_ms,z1,ma1,sd1,gate=None):
    """gate=(z5v,k5f,reg_pos) or None"""
    pos= DIR in('short','trendlong')
    cand=np.flatnonzero(z1>=k1) if pos else np.flatnonzero(z1<=-k1)
    cand=cand[cand>=1440]
    if gate is not None:
        z5v,k5f,reg_pos=gate
        g=(z5v[cand]>=k5f) if reg_pos else (z5v[cand]<=-k5f)
        cand=cand[g]
    rets=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<cd_ms: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+MH,'right'))
        if hi<=ci+1: continue
        if DIR=='trendlong':
            tp=ma1[ci]+b*sd1[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L1[ci+1:hi]<=sx; TP=H1[ci+1:hi]>=tx
        elif DIR=='trend':
            tp=ma1[ci]-b*sd1[ci]
            if tp<=price: continue
            pct=tp/price-1; tx=price*(1-pct); sx=price*(1+pct)
            SL=H1[ci+1:hi]>=sx; TP=L1[ci+1:hi]<=tx
        elif DIR=='long':
            tp=ma1[ci]-b*sd1[ci]
            if tp<=price: continue
            pct=tp/price-1; tx=tp; sx=price*(1-pct)
            SL=L1[ci+1:hi]<=sx; TP=H1[ci+1:hi]>=tx
        else:
            tp=ma1[ci]+b*sd1[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=tp; sx=price*(1+pct)
            SL=H1[ci+1:hi]>=sx; TP=L1[ci+1:hi]<=tx
        hit=SL|TP
        if hit.any():
            j=int(np.argmax(hit)); xp=sx if SL[j] else tx
        else: xp=C1[hi-1]
        r=(xp/price-1-FEE) if DIR in('trendlong','long') else ((price-xp)/price-FEE)
        rets.append(r); last=ts1[ci]
    return np.array(rets)

def st(a):
    if len(a)==0: return None
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    return len(a),len(a)/months,a.mean()*100,pf,100*(a>0).mean()

DIRS=[('trendlong','추세롱',True),('trend','추세숏',False),('long','역추롱',False),('short','역추숏',True)]
# reg_pos: trendlong/short → z5 + 쪽 게이트, trend/long → − 쪽
t0=time.time()
for DIR,nm,reg_pos in DIRS:
    # ① 월~20회 되는 (w1,K1) 찾기: cd=1h, B는 {-3,-2,-1,0,1} 중 최적
    best=None
    for w1 in [360,720,1440]:
        z1,ma1,sd1=zma(C1,w1)
        for k1 in [round(2.0+0.1*i,1) for i in range(21)]:
            # 빈도 추정: B=-3(거의 항상 유효)로 n 측정
            a=sim(DIR,w1,k1,-3.0,H,z1,ma1,sd1)
            pm=len(a)/months
            if not (17<=pm<=23): continue
            for b in [-3.0,-2.0,-1.0,0.0,1.0]:
                a=sim(DIR,w1,k1,b,H,z1,ma1,sd1)
                s=st(a)
                if s and 17<=s[1]<=23 and (best is None or s[2]>best[1][2]):
                    best=((w1,k1,b),s,(z1,ma1,sd1))
        if best: pass
    if not best:
        print(f'{nm}: 월17~23회 대역 조합 없음'); continue
    (w1,k1,b),s,(z1,ma1,sd1)=best
    print(f'\n===== {nm} — 베이스라인(월~20회): w1={w1//60}h K1={k1} B{b} cd1h =====')
    print(f'  게이트없음: n{s[0]}({s[1]:.1f}/월) 기대{s[2]:+.3f}% PF{s[3]:.2f} 승{s[4]:.0f}%')
    print(f'  {"레짐":12}{"n(월)":>12}{"기대":>9}{"PF":>7}{"승%":>5}{"월합":>8}')
    for w5 in [288,864,2016]:
        for k5f in [1.0,1.5,2.0,2.5]:
            a=sim(DIR,w1,k1,b,H,z1,ma1,sd1,gate=(Z5[w5],k5f,reg_pos))
            s2=st(a)
            if s2: print(f'  w5={w5//288}d K5f={k5f:<4}{s2[0]:>6}({s2[1]:>4.1f}) {s2[2]:>+8.3f}%{s2[3]:>7.2f}{s2[4]:>5.0f}{s2[1]*s2[2]:>+7.1f}%p')
            else: print(f'  w5={w5//288}d K5f={k5f:<4}  n부족')
    print(f'  ({time.time()-t0:.0f}s)',flush=True)
