# -*- coding: utf-8 -*-
"""S22 후보 2차 — 4h봉 신규 진입/청산 6패밀리 (실전 3 + 수학모델 3).
sweep: L이 N봉 저점 하향 이탈 후 종가 복귀(스탑헌트 반전) 롱 / 미러 숏
rsi: Wilder RSI(2~4) 과매도 롱 | nlow: N봉 종가 신저가 롱 (Connors류)
emapull: EMA20>EMA50 추세 중 EMA20 하향 터치 롱
ou: log가 편차 AR(1) 반감기 모델 — 진입 z≤-k, 청산 = m×반감기 시간청산 (모델이 보유기간 결정)
ewz: EWMA 적응평균 잔차 z (칼만-lite) 롱/숏
청산: T=N봉 시간청산(종가). SL 없음. FEE 0.11%. 필터: n≥30, 기대≥0.4%, 마이너스연도(<-3%p)≤2, 최악>-30%p, 연≥3건."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000

def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def evaluate(tr,yrs_total):
    if len(tr)<30: return None
    a=np.array([r for _,r in tr]); w=a>0
    exp=a.mean()*100
    if exp<0.4: return None
    pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
    ysum=defaultdict(float)
    for en,r in tr: ysum[yr(en)]+=r*100
    bad=[y for y,s in ysum.items() if s<-3]; worst=min(ysum.values())
    if len(bad)>2 or worst<-30: return None
    gpy=len(tr)/yrs_total
    if gpy<3: return None
    return dict(n=len(tr),exp=exp,pf=pf,wr=100*w.mean(),mo=gpy/12,
                ys=' '.join(f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))

def sim_time(C,ts,cand,side,hb,cd_bars=6,holds=None):
    """시간청산: hb봉 후 종가. holds=개별 보유봉수 배열(OU용)"""
    n=len(C); out=[]; last=-10**9
    for j,ci in enumerate(cand):
        if ci+1>=n: continue
        if ci-last<cd_bars: continue
        h=int(holds[j]) if holds is not None else hb
        ex=min(ci+max(h,1),n-1)
        r=side*(C[ex]/C[ci]-1)-FEE
        out.append((int(ts[ci]),r)); last=ci
    return out

def rollsum(x,w):
    cs=np.concatenate([[0.],np.cumsum(x)])
    out=np.full(len(x),np.nan); out[w-1:]=cs[w:]-cs[:-w]
    return out

SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','JP225','US100','GER40','UK100','HK50','XAGUSD','WTI','XAUUSD']
res=defaultdict(list)
for sym in SYMS:
    m=_load_cache(sym,'240')
    if not m: continue
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64)
    n=len(C); yrs_total=(int(ts[-1])-int(ts[0]))/(365.25*DAY)

    # ---- 1) liquidity sweep ----
    for N in [12,30,60,120]:
        pl=np.full(n,np.nan); ph=np.full(n,np.nan)
        for i in range(N,n): pl[i]=L[i-N:i].min(); ph[i]=Hh[i-N:i].max()
        swL=np.flatnonzero((L<pl)&(C>pl)); swH=np.flatnonzero((Hh>ph)&(C<ph))
        for hb in [6,18,30,60]:
            e=evaluate(sim_time(C,ts,swL,1,hb),yrs_total)
            if e: res[sym].append((f'sweep롱 N{N} T{hb}',e))
            e=evaluate(sim_time(C,ts,swH,-1,hb),yrs_total)
            if e: res[sym].append((f'sweep숏 N{N} T{hb}',e))
    # ---- 2) RSI 과매도 ----
    d=np.diff(C,prepend=C[0]); up=np.maximum(d,0); dn=np.maximum(-d,0)
    for p in [2,3,4]:
        au=np.copy(up); ad=np.copy(dn)
        for i in range(1,n): au[i]=(au[i-1]*(p-1)+up[i])/p; ad[i]=(ad[i-1]*(p-1)+dn[i])/p
        rsi=100-100/(1+au/np.where(ad==0,1e-12,ad))
        for th in [10,15,20]:
            cand=np.flatnonzero(rsi<th)
            for hb in [6,18,30]:
                e=evaluate(sim_time(C,ts,cand,1,hb),yrs_total)
                if e: res[sym].append((f'rsi{p}<{th} T{hb}',e))
    # ---- 3) N봉 종가 신저가 ----
    for N in [24,42,90]:
        lo=np.full(n,np.nan)
        for i in range(N,n): lo[i]=C[i-N:i].min()
        cand=np.flatnonzero(C<lo)
        for hb in [6,18,30]:
            e=evaluate(sim_time(C,ts,cand,1,hb),yrs_total)
            if e: res[sym].append((f'nlow{N} T{hb}',e))
    # ---- 4) EMA 풀백 ----
    def ema(x,s):
        a=2/(s+1); out=np.copy(x)
        for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
        return out
    e20=ema(C,20); e50=ema(C,50)
    for cond,nm in [((e20>e50)&(L<=e20)&(C>e50),'20/50'),((e20>e50)&(C<e20)&(np.roll(C,1)>=np.roll(e20,1)),'x20')]:
        cand=np.flatnonzero(cond)
        for hb in [18,30,60]:
            e=evaluate(sim_time(C,ts,cand,1,hb),yrs_total)
            if e: res[sym].append((f'emapull{nm} T{hb}',e))
    # ---- 5) OU 반감기 모델 ----
    lg=np.log(C)
    for W in [180,360,720]:
        ma=rollsum(lg,W)/W
        e_=lg-ma
        s1=rollsum(e_*np.roll(e_,1),W); s2=rollsum(np.roll(e_,1)**2,W)
        phi=s1/np.where(s2==0,1e-12,s2)
        var=rollsum(e_*e_,W)/W-(rollsum(e_,W)/W)**2
        sde=np.sqrt(np.maximum(var,1e-12))
        z=e_/sde
        hl=np.where((phi>0)&(phi<1),-np.log(2)/np.log(np.maximum(phi,1e-9)),np.nan)
        for k in [1.5,2.0,2.5]:
            cand=np.flatnonzero((z<=-k)&(hl>1)&(hl<90))
            if len(cand)<30: continue
            for mm in [1.0,1.5,2.0]:
                holds=np.clip(hl[cand]*mm,1,90)
                e=evaluate(sim_time(C,ts,cand,1,0,holds=holds),yrs_total)
                if e: res[sym].append((f'OU W{W} k{k} m{mm}',e))
    # ---- 6) EWMA 잔차 z (칼만-lite) ----
    for s in [20,50,100,200]:
        mu=ema(C,s); resid=C-mu
        sig=ema(np.abs(resid),s)
        z=resid/np.where(sig==0,1e-12,sig)
        for k,side,nm in [(2.5,1,'역롱'),(3.0,1,'역롱'),(2.5,-1,'추숏'),(3.0,-1,'추숏')]:
            cand=np.flatnonzero(z<=-k) if side>0 else np.flatnonzero(z>=k)
            for hb in [18,30,60]:
                e=evaluate(sim_time(C,ts,cand,side if nm=='역롱' else -1,hb) if nm=='역롱' else sim_time(C,ts,cand,-1,hb),yrs_total)
                if e: res[sym].append((f'ewz{nm} s{s} k{k} T{hb}',e))
    print(f'{sym} done ({len(res[sym])})',flush=True)

print()
fam_pass=defaultdict(set)
print(f'{"심볼":8} {"셀":24} {"n":>4} {"기대%":>6} {"PF":>5} {"승%":>4} {"월":>4} | 연도별')
for sym in SYMS:
    rs=sorted(res.get(sym,[]),key=lambda x:-x[1]['exp']*min(x[1]['pf'],5))
    seen=set(); shown=0
    for name,e in rs:
        fam=name.split()[0].rstrip('0123456789<').split('<')[0]
        fam=''.join(c for c in name.split()[0] if not c.isdigit()).replace('<','')
        fam_pass[fam].add(sym)
        if fam in seen: continue
        seen.add(fam); shown+=1
        print(f'{sym:8} {name:24} {e["n"]:>4} {e["exp"]:>+6.2f} {e["pf"]:>5.2f} {e["wr"]:>3.0f}% {e["mo"]:>4.1f} | {e["ys"]}')
        if shown>=4: break
print()
print('=== 패밀리별 통과 심볼 수 ===')
for f,ss in sorted(fam_pass.items(),key=lambda x:-len(x[1])):
    print(f'{f:12} {len(ss):>2}종: {" ".join(sorted(ss))}')
