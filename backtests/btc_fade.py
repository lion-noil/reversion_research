# -*- coding: utf-8 -*-
"""모멘텀 점화 페이드(급락 매수) 정밀 탐색 — BTCUSDT 1분 6.3년, fee 0.11%.
1단계: 진입 그리드(M×r×cd) × 시간청산 T → 연도균형 상위 확인
2단계: 상위 진입셀에 청산 변형 — 고정TP/SL(원시%), 되돌림목표(낙폭의 x% 회복), SL무+시간캡
출력: 연도균형(큰마이너스연도<=2, 최악>-30%p) 통과만."""
import sys, os, time
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=float(os.getenv('FEE','0.0011'))
m1=_load_cache(os.getenv('SYM','BTCUSDT'),'1')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
def yof(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year
RET={}
for M in [5,10,15,30,60,120,240]:
    r=C1/np.roll(C1,M)-1; r[:M]=np.nan; RET[M]=r

def entries(M,thr,cd_ms):
    cand=np.flatnonzero(RET[M]<=-thr)
    sel=[]; last=None
    for ci in cand:
        if last is not None and (ts1[ci]-last)<cd_ms: continue
        sel.append(ci); last=ts1[ci]
    return np.array(sel,dtype=int)

def year_ok(sel,r):
    byy={}
    for i in range(len(sel)): byy.setdefault(yof(ts1[sel[i]]),[]).append(r[i])
    ys={y:sum(v)*100 for y,v in byy.items()}
    neg=[y for y,s in ys.items() if s<-10]
    if len(neg)>2 or (ys and min(ys.values())<-30): return None
    return ' '.join(f'{y%100:02d}:{ys[y]:+.0f}' for y in sorted(ys))

def stat_line(tag,sel,r):
    if len(r)<80 or r.mean()<=0.0008: return None
    yl=year_ok(sel,r)
    if yl is None: return None
    pf=r[r>0].sum()/(-r[r<0].sum()) if (r<0).any() else 99
    return (r.mean(),f'{tag}: n{len(r)}({len(r)/months:.1f}/월) 기대{r.mean()*100:+.3f}% PF{pf:.2f} 승{100*(r>0).mean():.0f}% 최악{r.min()*100:+.1f}% | {yl}')

t0=time.time()
print('===== 1단계: 진입 그리드 × 시간청산 =====',flush=True)
S1=[]
for M in [5,10,15,30,60,120,240]:
    for thr in [float(x) for x in os.getenv('THRS','0.015,0.02,0.025,0.03,0.04,0.05').split(',')]:
        for cdh in [0.5,1,3,6]:
            sel=entries(M,thr,int(cdh*H))
            if len(sel)<80: continue
            for Th in [2,4,8,24,48,72]:
                ex=np.searchsorted(ts1, ts1[sel]+int(Th*H),'right')-1
                ok=(ex>sel)&(ex<n1); s2,e2=sel[ok],ex[ok]
                r=C1[e2]/C1[s2]-1-FEE
                res=stat_line(f'M{M}m r-{thr*100:.1f}% cd{cdh}h T{Th}h',s2,r)
                if res: S1.append(res)
    print(f'  M{M}m done ({time.time()-t0:.0f}s)',flush=True)
S1.sort(reverse=True,key=lambda x:x[0])
for _,l in S1[:15]: print(l)

print('\n===== 2단계: 상위 진입셀 × 청산 변형 =====',flush=True)
# 상위 진입셀 3개 추출(중복 M,r 제거)
seen=set(); tops=[]
for e,l in S1:
    key=l.split(' cd')[0]
    if key in seen: continue
    seen.add(key); tops.append(l)
    if len(tops)>=4: break
# 파싱 대신 대표 3셀 하드코딩 후보: 1단계 결과에서 재구성
import re
cells=[]
for l in tops:
    m=re.match(r'M(\d+)m r-([\d.]+)% cd([\d.]+)h T(\d+)h',l)
    if m: cells.append((int(m.group(1)),float(m.group(2))/100,float(m.group(3))))
for M,thr,cdh in cells:
    sel=entries(M,thr,int(cdh*H))
    print(f'--- 진입 M{M}m r-{thr*100:.1f}% cd{cdh}h (n{len(sel)}) ---')
    # (a) 고정 TP/SL (원시%) + 캡48h
    for tp_p in [0.01,0.02,0.03]:
        for sl_p in [0.02,0.03,0.05,None]:
            out=[]
            for ci in sel:
                if ci+1>=n1: break
                price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+48*H,'right'))
                if hi<=ci+1: continue
                tp=price*(1+tp_p); TP=H1[ci+1:hi]>=tp
                if sl_p is None: SL=np.zeros(hi-ci-1,bool); sl=None
                else: sl=price*(1-sl_p); SL=L1[ci+1:hi]<=sl
                hit=SL|TP
                if hit.any():
                    j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
                else: ex=hi-1; xp=C1[ex]
                out.append((ci,xp/price-1-FEE))
            if not out: continue
            s2=np.array([o[0] for o in out]); r=np.array([o[1] for o in out])
            res=stat_line(f'  (a)TP{tp_p*100:.0f}%/SL{("무" if sl_p is None else f"{sl_p*100:.0f}%")} 캡48h',s2,r)
            if res: print(res[1])
    # (b) 되돌림 목표: 진입가×(1+x×낙폭), SL무, 캡48h
    for x in [0.5,1.0,1.5]:
        out=[]
        for ci in sel:
            if ci+1>=n1: break
            price=C1[ci]; drop=abs(RET[M][ci])
            tp=price*(1+x*drop)
            hi=int(np.searchsorted(ts1,ts1[ci]+48*H,'right'))
            if hi<=ci+1: continue
            TP=H1[ci+1:hi]>=tp
            if TP.any(): ex=ci+1+int(np.argmax(TP)); xp=tp
            else: ex=hi-1; xp=C1[ex]
            out.append((ci,xp/price-1-FEE))
        s2=np.array([o[0] for o in out]); r=np.array([o[1] for o in out])
        res=stat_line(f'  (b)되돌림 x{x} SL무 캡48h',s2,r)
        if res: print(res[1])
    print(f'  ({time.time()-t0:.0f}s)',flush=True)
