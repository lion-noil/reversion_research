# -*- coding: utf-8 -*-
"""신규 진입 패밀리 탐색 — 변동성돌파(A)·돈치안(B)·스퀴즈(C)·모멘텀점화(D). BTCUSDT 1분 6.3년, fee 0.11%.
청산: 시간청산 T (단순·공정비교). 필터: 기대>+0.05%, 연도균형(큰마이너스연도<=2, 최악연도>-30%p)."""
import sys, os, time
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011
m1=_load_cache('BTCUSDT','1')
O1=np.array([x['open'] for x in m1]); C1=np.array([x['close'] for x in m1])
H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
def yof(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def roll_max(x,w):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    out=np.full(len(x),np.nan)
    out[w-1:]=sw(x,w).max(axis=1)
    return out
def roll_min(x,w):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    out=np.full(len(x),np.nan)
    out[w-1:]=sw(x,w).min(axis=1)
    return out

def evaluate(sig_idx, side, Th, cd_ms, tag, results):
    """sig_idx: 진입 후보 인덱스(신호 bar 종가 진입). side=+1롱/-1숏. 시간청산 Th."""
    sel=[]; last=None
    for ci in sig_idx:
        if last is not None and (ts1[ci]-last)<cd_ms: continue
        sel.append(ci); last=ts1[ci]
    sel=np.array(sel,dtype=int)
    if len(sel)<100: return
    ex=np.searchsorted(ts1, ts1[sel]+int(Th*H),'right')-1
    ok=(ex>sel)&(ex<n1); sel,ex=sel[ok],ex[ok]
    r=side*(C1[ex]/C1[sel]-1)-FEE
    if r.mean()<=0.0005: return
    byy={}
    for i in range(len(sel)): byy.setdefault(yof(ts1[sel[i]]),[]).append(r[i])
    ys={y:sum(v)*100 for y,v in byy.items()}
    neg=[y for y,s in ys.items() if s<-10]
    if len(neg)>2 or (ys and min(ys.values())<-30): return
    pf=r[r>0].sum()/(-r[r<0].sum()) if (r<0).any() else 99
    yline=' '.join(f'{y%100:02d}:{ys[y]:+.0f}' for y in sorted(ys))
    results.append((r.mean(),f'{tag}: n{len(r)}({len(r)/months:.0f}/월={len(r)/months/30.44:.1f}/일) 기대{r.mean()*100:+.3f}% PF{pf:.2f} 승{100*(r>0).mean():.0f}% | {yline}'))

t0=time.time(); RES=[]
# ---- A. 변동성 돌파 (UTC일 기준) ----
day=ts1//(24*H)
day_start=np.searchsorted(day,np.unique(day))
dopen=np.full(n1,np.nan); prange=np.full(n1,np.nan)
ud=np.unique(day)
d_hi={}; d_lo={}; d_op={}
for i,d in enumerate(ud):
    s=day_start[i]; e=day_start[i+1] if i+1<len(ud) else n1
    d_op[d]=O1[s]; d_hi[d]=H1[s:e].max(); d_lo[d]=L1[s:e].min()
for i,d in enumerate(ud):
    if d-1 in d_hi:
        s=day_start[i]; e=day_start[i+1] if i+1<len(ud) else n1
        dopen[s:e]=d_op[d]; prange[s:e]=d_hi[d-1]-d_lo[d-1]
for k in [0.3,0.5,0.7]:
    trg=dopen+k*prange
    cross=(C1>=trg)&(np.roll(C1,1)<trg)&np.isfinite(trg)
    cross[0]=False
    idx=np.flatnonzero(cross)
    for Th in [8,24]:
        evaluate(idx,1,Th,20*H,f'A돌파 k{k} T{Th}h 롱',RES)
    trgS=dopen-k*prange
    crossS=(C1<=trgS)&(np.roll(C1,1)>trgS)&np.isfinite(trgS); crossS[0]=False
    for Th in [8,24]:
        evaluate(np.flatnonzero(crossS),-1,Th,20*H,f'A돌파 k{k} T{Th}h 숏',RES)
print(f'A done {time.time()-t0:.0f}s',flush=True)
# ---- B. 돈치안 돌파 ----
for Nh in [4,8,24,48,168]:
    w=Nh*60
    hh=roll_max(H1,w); ll=roll_min(L1,w)
    ph=np.roll(hh,1); pl=np.roll(ll,1)
    up=(C1>ph)&np.isfinite(ph); dn=(C1<pl)&np.isfinite(pl)
    up[0]=dn[0]=False
    for Th in [4,24,48]:
        for cdh in [1,6]:
            evaluate(np.flatnonzero(up),1,Th,int(cdh*H),f'B돈치안 {Nh}h신고 T{Th}h cd{cdh} 롱',RES)
            evaluate(np.flatnonzero(dn),-1,Th,int(cdh*H),f'B돈치안 {Nh}h신저 T{Th}h cd{cdh} 숏',RES)
print(f'B done {time.time()-t0:.0f}s',flush=True)
# ---- C. 스퀴즈 후 돌파 (6h레인지가 30일 중앙값의 q배 미만 → 12h고점 돌파) ----
w6=360
rng=roll_max(H1,w6)-roll_min(L1,w6)
med=np.full(n1,np.nan)
step=1440; base=np.copy(rng)
from collections import deque
# 30일 rolling median 근사: 일단위 다운샘플 median
dayidx=np.arange(n1)//1440
uds=np.unique(dayidx)
dmed={}
for d in uds:
    s=d*1440; e=min((d+1)*1440,n1)
    dmed[d]=np.nanmedian(rng[s:e])
med30=np.full(n1,np.nan)
keys=sorted(dmed)
import bisect
vals=[dmed[k] for k in keys]
for i,d in enumerate(keys):
    if i>=30:
        m30=np.median(vals[i-30:i])
        s=d*1440; e=min((d+1)*1440,n1)
        med30[s:e]=m30
hh12=roll_max(H1,720); ph12=np.roll(hh12,1)
ll12=roll_min(L1,720); pl12=np.roll(ll12,1)
for q in [0.5,0.7]:
    squeeze=rng<q*med30
    up=squeeze&(C1>ph12)&np.isfinite(ph12); dn=squeeze&(C1<pl12)&np.isfinite(pl12)
    up[0]=dn[0]=False
    for Th in [8,24,48]:
        evaluate(np.flatnonzero(up),1,Th,6*H,f'C스퀴즈 q{q} T{Th}h 롱',RES)
        evaluate(np.flatnonzero(dn),-1,Th,6*H,f'C스퀴즈 q{q} T{Th}h 숏',RES)
print(f'C done {time.time()-t0:.0f}s',flush=True)
# ---- D. 모멘텀 점화 (M분 수익률 r% 초과 → 추종/페이드) ----
for M in [15,60,240]:
    ret=C1/np.roll(C1,M)-1; ret[:M]=np.nan
    for rthr in [0.01,0.02,0.03]:
        upi=np.flatnonzero(ret>=rthr); dni=np.flatnonzero(ret<=-rthr)
        for Th in [4,24,48]:
            evaluate(upi,1,Th,3*H,f'D점화 {M}m>+{rthr*100:.0f}% T{Th}h 추종롱',RES)
            evaluate(dni,1,Th,3*H,f'D점화 {M}m<-{rthr*100:.0f}% T{Th}h 페이드롱',RES)
            evaluate(upi,-1,Th,3*H,f'D점화 {M}m>+{rthr*100:.0f}% T{Th}h 페이드숏',RES)
            evaluate(dni,-1,Th,3*H,f'D점화 {M}m<-{rthr*100:.0f}% T{Th}h 추종숏',RES)
print(f'D done {time.time()-t0:.0f}s',flush=True)
RES.sort(reverse=True,key=lambda x:x[0])
print(f'\n=== 연도균형 통과 셀 ({len(RES)}개) — 기대순 ===')
for _,line in RES[:30]: print(line)
