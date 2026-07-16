# -*- coding: utf-8 -*-
"""S22 후보 — 일봉 신규 진입 패밀리 스윕 (S11 방법론: 전기간·연도균형·베이스자립).
패밀리: fade(M일 낙폭≤-r% 롱 / 급등숏), streak(연속 N음봉 롱), donch(N일 신고가 돌파 롱 / 신저가 숏).
청산: 시간청산 h일 종가 (fade 롱은 되돌림×1.5 TP 캡h 변형도). SL 없음(S11 교훈), 진입쿨다운 1일=매봉.
필터: n≥25, 기대≥+0.3%, 마이너스연도(<-3%p)≤2, 최악연도>-30%p, mc≤15. FEE 0.11%."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; DAY=86400*1000

SYMS=[('US100','D10'),('JP225','D10'),('GER40','D10'),('UK100','D10'),('HK50','D10'),
      ('XAUUSD','D10'),('XAGUSD','D10'),('WTI','D10'),
      ('USDJPY','D10'),('EURUSD','D10'),('GBPUSD','D10'),('AUDUSD','D10'),
      ('BTCUSDT','D'),('ETHUSDT','D'),('SOLUSDT','D'),('XRPUSDT','D')]

def load(sym,itv):
    m=_load_cache(sym,itv)
    if not m or len(m)<300: return None
    C=np.array([x['close'] for x in m]); H=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64)
    return C,H,L,ts

def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def simulate(C,H,L,ts,cand,side,h,retr=None):
    """cand=진입 인덱스, side=1롱/-1숏, h=보유일, retr=(mult,drop배열) 롱 되돌림TP"""
    n=len(C); out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<DAY: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+h*DAY,'right'))
        if hi<=ci+1: continue
        if retr is not None:
            tp=price*(1+retr[0]*abs(retr[1][ci]))
            TP=H[ci+1:hi]>=tp
            if TP.any(): xp=tp
            else: xp=C[hi-1]
            r=xp/price-1-FEE
        else:
            xp=C[hi-1]; r=side*(xp/price-1)-FEE
        out.append((int(ts[ci]),r)); last=ts[ci]
    return out

def maxc(tr,h):
    ev=[]
    for en,_ in tr: ev.append((en,1)); ev.append((en+h*DAY,-1))
    ev.sort(); c=0; mx=0
    for _,d in ev: c+=d; mx=max(mx,c)
    return mx

def evaluate(tr,yrs_total,h):
    if len(tr)<25: return None
    a=np.array([r for _,r in tr]); w=a>0
    exp=a.mean()*100
    if exp<0.3: return None
    pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
    ysum=defaultdict(float)
    for en,r in tr: ysum[yr(en)]+=r*100
    bad=[y for y,s in ysum.items() if s<-3]
    worst=min(ysum.values())
    if len(bad)>2 or worst<-30: return None
    if maxc(tr,h)>15: return None
    gpy=len(tr)/yrs_total
    if gpy<5: return None
    return dict(n=len(tr),exp=exp,pf=pf,wr=100*w.mean(),gpy=gpy,worst=worst,nbad=len(bad),
                ys=' '.join(f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))

results=defaultdict(list)
for sym,itv in SYMS:
    d=load(sym,itv)
    if d is None: continue
    C,H,L,ts=d; n=len(C)
    yrs_total=(int(ts[-1])-int(ts[0]))/(365.25*DAY)
    # --- fade: M일 수익률 ---
    for M in [2,3,5,7,10]:
        ret=C/np.roll(C,M)-1; ret[:M]=np.nan
        for r0 in [0.03,0.05,0.07,0.10,0.15]:
            cand=np.flatnonzero(ret<=-r0)
            if len(cand)<25: continue
            for h in [3,5,10,15]:
                for tag,retr in [('T',None),('R',(1.5,ret))]:
                    tr=simulate(C,H,L,ts,cand,1,h,retr)
                    e=evaluate(tr,yrs_total,h)
                    if e: results[sym].append((f'fade롱 {M}d≤-{r0*100:.0f}% {tag}{h}',e))
            # 급등 페이드숏
            cs=np.flatnonzero(ret>=r0)
            if len(cs)>=25:
                for h in [3,5,10]:
                    tr=simulate(C,H,L,ts,cs,-1,h)
                    e=evaluate(tr,yrs_total,h)
                    if e: results[sym].append((f'fade숏 {M}d≥+{r0*100:.0f}% T{h}',e))
    # --- streak: 연속 N음봉 ---
    dn=(C<np.roll(C,1)).astype(int); dn[0]=0
    run_=np.zeros(n,int)
    for i in range(1,n): run_[i]=run_[i-1]+1 if dn[i] else 0
    for N in [3,4,5,6,7]:
        cand=np.flatnonzero(run_==N)
        if len(cand)<25: continue
        for h in [3,5,10,15]:
            tr=simulate(C,H,L,ts,cand,1,h)
            e=evaluate(tr,yrs_total,h)
            if e: results[sym].append((f'streak {N}음봉 T{h}',e))
    # --- donchian: N일 신고/신저 돌파 ---
    for N in [20,55,100]:
        hh=np.full(n,np.nan); ll=np.full(n,np.nan)
        for i in range(N,n): hh[i]=C[i-N:i].max(); ll[i]=C[i-N:i].min()
        cl=np.flatnonzero(C>hh); cs=np.flatnonzero(C<ll)
        for h in [5,10,15,20]:
            if len(cl)>=25:
                tr=simulate(C,H,L,ts,cl,1,h)
                e=evaluate(tr,yrs_total,h)
                if e: results[sym].append((f'donch롱 {N}d돌파 T{h}',e))
            if len(cs)>=25:
                tr=simulate(C,H,L,ts,cs,-1,h)
                e=evaluate(tr,yrs_total,h)
                if e: results[sym].append((f'donch숏 {N}d돌파 T{h}',e))
    print(f'{sym} done ({len(results[sym])} 통과)',flush=True)

print()
print(f'{"심볼":8} {"셀":26} {"n":>4} {"기대%":>6} {"PF":>5} {"승%":>4} {"월":>4} | 연도별')
for sym,_ in SYMS:
    rs=sorted(results.get(sym,[]),key=lambda x:-x[1]['exp']*min(x[1]['pf'],5))
    seen=set(); shown=0
    for name,e in rs:
        fam=name.split()[0]
        if fam in seen: continue
        seen.add(fam); shown+=1
        print(f'{sym:8} {name:26} {e["n"]:>4} {e["exp"]:>+6.2f} {e["pf"]:>5.2f} {e["wr"]:>3.0f}% {e["gpy"]/12:>4.1f} | {e["ys"]}')
        if shown>=3: break
