# -*- coding: utf-8 -*-
"""급등 페이드 숏 정밀 탐색 — 미러: M분 수익률 >= +r% → 숏. 6.3년, fee 0.11%.
1단계: M×r×cd × 시간청산. 2단계: 상위셀에 (a)고정TP/SL (b)되돌림목표(급등의 x% 반납) ×SL{무,3%,5%}.
연도균형(큰마이너스연도<=2, 최악>-30%p) 통과만 출력."""
import sys, os, time, re
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
    cand=np.flatnonzero(RET[M]>=thr); sel=[]; last=None
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
t0=time.time(); S1=[]
print('===== 1단계: 급등 페이드 숏 × 시간청산 =====',flush=True)
for M in [5,10,15,30,60,120,240]:
    for thr in [float(x) for x in os.getenv('THRS','0.015,0.02,0.025,0.03,0.04,0.05').split(',')]:
        for cdh in [0.5,1,3,6]:
            sel=entries(M,thr,int(cdh*H))
            if len(sel)<80: continue
            for Th in [2,4,8,24,48,72]:
                ex=np.searchsorted(ts1, ts1[sel]+int(Th*H),'right')-1
                ok=(ex>sel)&(ex<n1); s2,e2=sel[ok],ex[ok]
                r=-(C1[e2]/C1[s2]-1)-FEE
                res=stat_line(f'M{M}m r+{thr*100:.1f}% cd{cdh}h T{Th}h',s2,r)
                if res: S1.append(res)
print(f'통과 {len(S1)}개 ({time.time()-t0:.0f}s)',flush=True)
S1.sort(reverse=True,key=lambda x:x[0])
for _,l in S1[:12]: print(l)
print('\n===== 2단계: 상위(또는 대표) 진입셀 × 청산변형 =====',flush=True)
cells=[]
seen=set()
for e,l in S1:
    m=re.match(r'M(\d+)m r\+([\d.]+)% cd([\d.]+)h',l)
    if m:
        key=(m.group(1),m.group(2))
        if key in seen: continue
        seen.add(key); cells.append((int(m.group(1)),float(m.group(2))/100,float(m.group(3))))
    if len(cells)>=3: break
if not cells: cells=[(60,0.04,0.5),(30,0.04,0.5),(15,0.03,0.5)]  # 1단계 전멸 시 대표셀로 검증
for M,thr,cdh in cells:
    sel=entries(M,thr,int(cdh*H))
    print(f'--- 진입 M{M}m r+{thr*100:.1f}% cd{cdh}h (n{len(sel)}) ---')
    for x in [0.5,1.0,1.5]:
        for sl_p in [None,0.03,0.05]:
            out=[]
            for ci in sel:
                if ci+1>=n1: break
                price=C1[ci]; pump=abs(RET[M][ci]); tp=price*(1-x*pump)
                hi=int(np.searchsorted(ts1,ts1[ci]+48*H,'right'))
                if hi<=ci+1: continue
                TP=L1[ci+1:hi]<=tp
                if sl_p is None: SL=np.zeros(hi-ci-1,bool); sl=None
                else: sl=price*(1+sl_p); SL=H1[ci+1:hi]>=sl
                hit=SL|TP
                if hit.any():
                    j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
                else: ex=hi-1; xp=C1[ex]
                out.append((ci,(price-xp)/price-FEE))
            s2=np.array([o[0] for o in out]); r=np.array([o[1] for o in out])
            res=stat_line(f'  되돌림x{x} SL{("무" if sl_p is None else f"{sl_p*100:.0f}%")}',s2,r)
            if res: print(res[1])
            else:
                if len(r)>=80: print(f'  되돌림x{x} SL{("무" if sl_p is None else f"{sl_p*100:.0f}%")}: 기대{r.mean()*100:+.3f}% (연도균형 탈락 or 음수)')
    print(f'  ({time.time()-t0:.0f}s)',flush=True)
