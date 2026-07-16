# -*- coding: utf-8 -*-
"""US100 1분봉 숏 전용 1차 스크리닝 — z역추숏 / 급등페이드숏 / rsi과매수숏 / 양봉스트릭숏 / 스윕숏 / ewz추숏.
fee 0.11%, 필터: n≥30·기대≥0.4%·마이너스연도(<-3%p)≤2·최악>-30%p·월≥0.3.
⚠️ 1m 밀도: 2023-07+ 완전 / 2020 ~80% / 2021 24% / 2022 33% (백필 진행중 — 참고등급)."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000

m=_load_cache('US100','1')
C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64)
n=len(C); yrs_total=(int(ts[-1])-int(ts[0]))/(365.25*DAY)
print(f'US100 1m {n/1e6:.2f}M봉 {yrs_total:.1f}년',flush=True)

def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
def evaluate(tr):
    if len(tr)<30: return None
    a=np.array([r for _,r in tr]); w=a>0
    exp=a.mean()*100
    if exp<0.4: return None
    pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
    ysum=defaultdict(float)
    for en,r in tr: ysum[yr(en)]+=r*100
    bad=[y for y,s in ysum.items() if s<-3]; worst=min(ysum.values())
    if len(bad)>2 or worst<-30: return None
    mo=len(tr)/yrs_total/12
    if mo<0.3: return None
    return dict(n=len(tr),exp=exp,pf=pf,wr=100*w.mean(),mo=mo,score=exp*min(pf,5),
                ys=' '.join(f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))

def zma(x,win):
    N=len(x); cs=np.concatenate([[0.],np.cumsum(x)]); cs2=np.concatenate([[0.],np.cumsum(x*x)])
    ma=np.full(N,np.nan); sd=np.full(N,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(x-ma)/sd,np.nan),ma,sd
def ema(x,s):
    a=2/(s+1); out=np.copy(x)
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out

def sim_short_tp(cand,tp_arr,sl_arr,mh_d,cd_ms):
    """숏: TP=저가터치, SL=고가터치(없으면 None), 만기 종가"""
    out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+mh_d*DAY,'right'))
        if hi<=ci+1: continue
        tp=tp_arr[ci]
        if tp is None or not np.isfinite(tp) or tp>=price: continue
        TP=L[ci+1:hi]<=tp
        if sl_arr is not None:
            sl=sl_arr[ci]
            SL=Hh[ci+1:hi]>=sl; hit=TP|SL
            if hit.any():
                j=int(np.argmax(hit)); xp=sl if SL[j] else tp
            else: xp=C[hi-1]
        else:
            if TP.any(): xp=tp
            else: xp=C[hi-1]
        out.append((int(ts[ci]),(price-xp)/price-FEE)); last=ts[ci]
    return out
def sim_short_time(cand,hold_min,cd_ms):
    out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        ex=int(np.searchsorted(ts,ts[ci]+hold_min*60000,'right'))-1
        if ex<=ci: continue
        out.append((int(ts[ci]),(C[ci]-C[ex])/C[ci]-FEE)); last=ts[ci]
    return out

res=[]
# 1) z역추세숏: z>=K 숏, TP=MA+Bσ 복귀, SL 유/무
for win in [360,720,1440,2880]:  # 6h~48h
    z,ma,sd=zma(C,win)
    for K in [3.0,3.5,4.0,4.5,5.0]:
        cand=np.flatnonzero(z>=K); cand=cand[cand>=win]
        if len(cand)<30: continue
        for B in [-1,0,1,2]:
            tp=ma+B*sd
            for cdh in [1,3,6]:
                for slmode in ['nosl','sl']:
                    sl=None
                    if slmode=='sl':
                        pct=1-tp/C; sl=C*(1+np.abs(pct))
                    tr=sim_short_tp(cand,tp,sl,14,cdh*H)
                    e=evaluate(tr)
                    if e: res.append((f'z역숏 w{win//60}h K{K} B{B} cd{cdh} {slmode}',e))
# 2) 급등 페이드숏: M분 수익률 >= +r%
for M in [15,30,60,120,240]:
    ret=C/np.roll(C,M)-1; ret[:M]=np.nan
    for r0 in [0.01,0.015,0.02,0.03]:
        cand=np.flatnonzero(ret>=r0)
        if len(cand)<30: continue
        for hold in [240,720,1440,2880]:  # 4h~48h
            tr=sim_short_time(cand,hold,30*60000)
            e=evaluate(tr)
            if e: res.append((f'급등페이드숏 {M}m≥+{r0*100:.1f}% T{hold//60}h',e))
# 3) rsi 과매수숏 (5분 리샘플 rsi로 노이즈 완화)
g5=ts//(5*60000); u5,i5=np.unique(g5,return_index=True)
e5=np.append(i5[1:],n)-1; C5=C[e5]
d5=np.diff(C5,prepend=C5[0]); up=np.maximum(d5,0); dn=np.maximum(-d5,0)
for p in [3,4,6]:
    au=np.copy(up); ad=np.copy(dn)
    for i in range(1,len(C5)): au[i]=(au[i-1]*(p-1)+up[i])/p; ad[i]=(ad[i-1]*(p-1)+dn[i])/p
    rsi5=100-100/(1+au/np.where(ad==0,1e-12,ad))
    for th in [90,95]:
        c5=np.flatnonzero(rsi5>th)
        cand=e5[c5]
        if len(cand)<30: continue
        for hold in [240,720,1440]:
            tr=sim_short_time(cand,hold,60*60000)
            e=evaluate(tr)
            if e: res.append((f'rsi{p}(5m)>{th}숏 T{hold//60}h',e))
# 4) 양봉 스트릭숏 (30분 봉 기준 N연속 상승)
g30=ts//(30*60000); u30,i30=np.unique(g30,return_index=True)
e30=np.append(i30[1:],n)-1; C30=C[e30]
up30=(C30>np.roll(C30,1)).astype(int); up30[0]=0
run_=np.zeros(len(C30),int)
for i in range(1,len(C30)): run_[i]=run_[i-1]+1 if up30[i] else 0
for N in [5,7,9]:
    c30=np.flatnonzero(run_==N); cand=e30[c30]
    if len(cand)<30: continue
    for hold in [240,720,1440]:
        tr=sim_short_time(cand,hold,60*60000)
        e=evaluate(tr)
        if e: res.append((f'양봉스트릭{N}(30m)숏 T{hold//60}h',e))
# 5) 유동성 스윕숏: 고가가 N분 전고 돌파 후 종가 복귀
for N in [720,1440,2880,7200]:  # 12h~5d
    ph=np.full(n,np.nan)
    step=60  # 1h 단위로 롤링 최고 근사 (속도)
    hh1=ts//H; uh,ih=np.unique(hh1,return_index=True)
    Hmax=np.maximum.reduceat(Hh,ih)
    hr_of=np.searchsorted(uh,hh1)
    K_=N//60
    roll=np.full(len(uh),np.nan)
    for i in range(K_,len(uh)): roll[i]=Hmax[i-K_:i].max()
    ph=roll[np.clip(hr_of,0,len(uh)-1)]
    cand=np.flatnonzero((Hh>ph)&(C<ph))
    if len(cand)<30: continue
    for hold in [360,720,1440]:
        tr=sim_short_time(cand,hold,6*3600000)
        e=evaluate(tr)
        if e: res.append((f'스윕숏 N{N//60}h T{hold//60}h',e))
# 6) ewz 추세숏
for s in [240,720,1440,2880]:
    mu=ema(C,s); resid=C-mu; sig=ema(np.abs(resid),s)
    zz=resid/np.where(sig==0,1e-12,sig)
    for k in [2.5,3.0,3.5]:
        cand=np.flatnonzero(zz<=-k)
        if len(cand)<30: continue
        for hold in [360,720,1440]:
            tr=sim_short_time(cand,hold,6*3600000)
            e=evaluate(tr)
            if e: res.append((f'ewz추숏 s{s//60}h k{k} T{hold//60}h',e))

res.sort(key=lambda x:-x[1]['score'])
print(f'\n통과 {len(res)}개 (상위 20)')
print(f'{"셀":32} {"n":>4} {"기대%":>6} {"PF":>5} {"승%":>4} {"월":>4} | 연도별')
seen=set()
for name,e in res[:60]:
    fam=name.split()[0]
    tag='' if fam not in seen else ' (동일계열)'
    if fam in seen and sum(1 for nm,_ in res[:res.index((name,e))+1] if nm.split()[0]==fam)>3: continue
    seen.add(fam)
    print(f'{name:32} {e["n"]:>4} {e["exp"]:>+6.2f} {e["pf"]:>5.2f} {e["wr"]:>3.0f}% {e["mo"]:>4.1f} | {e["ys"]}')
