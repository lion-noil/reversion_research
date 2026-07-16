# -*- coding: utf-8 -*-
"""청산 메커니즘 진단 — B 곡선을 따라 TP/SL/만기 비율과 유형별 평균수익 분해.
대상: 1분 게임(월~20회 K) + 최적 5분 레짐 게이트 셀 (추세롱·역추롱).
역추롱 TP=MA−B·σ → B=0: 평균도달 청산 / B<0: 오버슛 / B>0: 부분회귀 조기청산."""
import sys, os
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
def z5on1(w):
    z5,_,_=zma(C5,w)
    idx=np.searchsorted(ts5+4*60000, ts1,'right')-1; idx=np.clip(idx,0,len(z5)-1)
    v=z5[idx]; v[idx<0]=np.nan; return v

def sim_diag(DIR,w1,k1,b,cd_ms,z1,ma1,sd1,gate):
    pos= DIR in('short','trendlong')
    cand=np.flatnonzero(z1>=k1) if pos else np.flatnonzero(z1<=-k1)
    cand=cand[cand>=1440]
    z5v,k5f,reg_pos=gate
    g=(z5v[cand]>=k5f) if reg_pos else (z5v[cand]<=-k5f)
    cand=cand[g]
    out={'TP':[], 'SL':[], 'TIME':[]}; last=None
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
        elif DIR=='long':
            tp=ma1[ci]-b*sd1[ci]
            if tp<=price: continue
            pct=tp/price-1; tx=tp; sx=price*(1-pct)
            SL=L1[ci+1:hi]<=sx; TP=H1[ci+1:hi]>=tx
        else: raise ValueError
        hit=SL|TP
        if hit.any():
            j=int(np.argmax(hit))
            if SL[j]: out['SL'].append(sx/price-1-FEE)
            else: out['TP'].append(tx/price-1-FEE)
        else: out['TIME'].append(C1[hi-1]/price-1-FEE)
        last=ts1[ci]
    return out

def report(tag,DIR,w1,k1,cd_h,w5,k5f,reg_pos,BS):
    z1,ma1,sd1=zma(C1,w1); z5v=z5on1(w5)
    print(f'\n===== {tag}: w1={w1//60}h K1={k1} cd{cd_h}h | 게이트 w5={w5//288}d K5f={k5f} | 보유2d =====')
    print(f'{"B":>5} {"n":>5}{"(월)":>6} | {"TP%":>5}{"SL%":>5}{"만기%":>6} | {"TP평균":>8}{"SL평균":>8}{"만기평균":>9} | {"기대":>8}{"PF":>6}')
    for b in BS:
        o=sim_diag(DIR,w1,k1,b,int(cd_h*H),z1,ma1,sd1,(z5v,k5f,reg_pos))
        allr=np.array(o['TP']+o['SL']+o['TIME']); n=len(allr)
        if n<30: print(f'{b:>5}  n<30'); continue
        pf=allr[allr>0].sum()/(-allr[allr<0].sum()) if (allr<0).any() else 99
        def m(x): return f'{np.mean(x)*100:+.2f}%' if x else '   —  '
        print(f'{b:>5} {n:>5}({n/months:>4.1f}) | {100*len(o["TP"])/n:>4.0f}%{100*len(o["SL"])/n:>4.0f}%{100*len(o["TIME"])/n:>5.0f}% | {m(o["TP"]):>8}{m(o["SL"]):>8}{m(o["TIME"]):>9} | {allr.mean()*100:>+7.3f}%{pf:>6.2f}')

# 추세롱: 최적 게이트 셀 (7d/K5f2.0)
report('추세롱','trendlong',720,3.7,1,2016,2.0,True,[-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0])
# 역추롱: 최적 게이트 셀 (7d/K5f1.5) — B>0=부분회귀, 0=평균, <0=오버슛
report('역추롱','long',720,3.7,1,2016,1.5,False,[-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5])
