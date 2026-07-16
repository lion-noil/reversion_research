# -*- coding: utf-8 -*-
"""청산 변형 A/B — 진입은 z 그대로, 청산만 교체. BTCUSDT 1분봉 6.3년, fee 0.11%.
A 순수 시간청산: 진입 후 T시간 뒤 종가 청산 (TP/SL 없음)
B 브레이크이븐 2단계: 초기 SL=s×σ → +p×σ 도달 시 SL을 본전(+fee)으로 이동, TP 없음, 시간캡 48h"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011
m1=_load_cache('BTCUSDT','1')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
def zma(win):
    cs=np.concatenate([[0.],np.cumsum(C1)]); cs2=np.concatenate([[0.],np.cumsum(C1*C1)])
    ma=np.full(n1,np.nan); sd=np.full(n1,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C1-ma)/sd,np.nan),ma,sd

def cands(z,win,k1,epos,cd_ms):
    cand=np.flatnonzero(z>=k1) if epos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=win]; sel=[]; last=None
    for ci in cand:
        if last is not None and (ts1[ci]-last)<cd_ms: continue
        sel.append(ci); last=ts1[ci]
    return np.array(sel,dtype=int)

def simA(z,win,DIR,k1,cd_ms,T_ms):
    epos = DIR in('trendlong','short'); is_long = DIR in('trendlong','long')
    sel=cands(z,win,k1,epos,cd_ms)
    ex=np.searchsorted(ts1, ts1[sel]+T_ms,'right')-1
    ok=(ex>sel)&(ex<n1)
    sel,ex=sel[ok],ex[ok]
    r=C1[ex]/C1[sel]-1-FEE
    if not is_long: r=-(C1[ex]/C1[sel]-1)-FEE
    return r,sel

def simB(z,ma,sd,win,DIR,k1,cd_ms,s_sl,p_be,cap_ms):
    epos = DIR in('trendlong','short'); is_long = DIR in('trendlong','long')
    sel=cands(z,win,k1,epos,cd_ms)
    out=[]
    for ci in sel:
        if ci+1>=n1: break
        price=C1[ci]; sig=sd[ci]
        hi=int(np.searchsorted(ts1,ts1[ci]+cap_ms,'right'))
        if hi<=ci+1: continue
        if is_long:
            sl=price-s_sl*sig; be_trg=price+p_be*sig; be=False
            seg_end=min(hi,n1)
            r=None
            for k in range(ci+1,seg_end):
                if not be and H1[k]>=be_trg: be=True; sl=price*(1+FEE)
                if L1[k]<=sl: r=sl/price-1-FEE; break
            if r is None: r=C1[seg_end-1]/price-1-FEE
        else:
            sl=price+s_sl*sig; be_trg=price-p_be*sig; be=False
            seg_end=min(hi,n1); r=None
            for k in range(ci+1,seg_end):
                if not be and L1[k]<=be_trg: be=True; sl=price*(1-FEE)
                if H1[k]>=sl: r=(price-sl)/price-FEE; break
            if r is None: r=(price-C1[seg_end-1])/price-FEE
        out.append(r)
    return np.array(out)

def rep(tag,a,n_div=None):
    if len(a)<40: print(f'  {tag:38} n<40'); return
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    pm=len(a)/months; pd_=pm/30.44
    print(f'  {tag:38} n{len(a):>5}({pm:>5.1f}/월={pd_:.2f}/일) 기대{a.mean()*100:>+7.3f}% PF{pf:>5.2f} 승{100*(a>0).mean():>3.0f}% 최악{a.min()*100:>+6.1f}% 월합{pm*a.mean()*100:>+6.1f}%p')

t0=time.time()
print('===== A. 순수 시간청산 (진입 z / T시간 후 청가) =====')
for win in [360,720,1320]:
    z,ma,sd=zma(win)
    for DIR in ['trendlong','long']:
        for k1 in [2.0,2.5,3.0,3.5,4.0,5.0]:
            for cdh in [0.5,1,3]:
                for Th in [2,4,8,24,48]:
                    a,_=simA(z,win,DIR,k1,int(cdh*H),int(Th*H))
                    pm=len(a)/months
                    if pm<40 or pm>150: continue   # 하루 1.3~5회 대역만 출력
                    if a.mean()<=0: continue
                    rep(f'{DIR:9} 창{win//60}h K{k1} cd{cdh}h T{Th}h',a)
    print(f'  (win {win//60}h 완료 {time.time()-t0:.0f}s)',flush=True)
print()
print('===== B. 브레이크이븐 2단계 (초기SL s×σ → +p×σ시 본전이동, 캡48h) =====')
for win in [720,1320]:
    z,ma,sd=zma(win)
    for DIR in ['trendlong','long']:
        for k1 in [2.5,3.0,3.5]:
            for s_sl in [1.0,2.0,3.0]:
                for p_be in [0.5,1.0,2.0]:
                    a=simB(z,ma,sd,win,DIR,k1,int(1*H),s_sl,p_be,48*H)
                    if len(a)<40 or a.mean()<=0: continue
                    rep(f'{DIR:9} 창{win//60}h K{k1} SL{s_sl}σ BE@{p_be}σ',a)
    print(f'  (win {win//60}h 완료 {time.time()-t0:.0f}s)',flush=True)
