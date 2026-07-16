# -*- coding: utf-8 -*-
"""S22 후보 3차 — 전 패밀리 롱/숏 양방향 + FX 7종 (4h봉).
패밀리: sweep(저점스윕롱/고점스윕숏), rsi(과매도롱/과매수숏), next(신저가롱/신고가숏),
emapull(상승추세 눌림롱/하락추세 반등숏), OU(롱/숏), ewz(역롱/역숏/추롱/추숏), z4방향(FX만 — 비FX는 s22_h4 완료).
FEE: 비FX 0.11% / FX 0.02%. 필터: n≥30, 기대≥0.4%(FX 0.15%), 마이너스연도(<-3%p, FX<-1.5%p)≤2, 최악>-30%p, 연≥3건."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; HOLD=15*DAY

NONFX=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','JP225','US100','GER40','UK100','HK50','XAGUSD','WTI','XAUUSD']
FX=['USDJPY','EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','NZDUSD']
if os.getenv('ONLY'):
    _o=os.getenv('ONLY').split(',')
    NONFX=[s for s in NONFX if s in _o]; FX=[s for s in FX if s in _o]

def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def make_eval(fee,isfx):
    emin=0.15 if isfx else 0.4; ybad=-1.5 if isfx else -3.0
    def evaluate(tr,yrs_total):
        if len(tr)<30: return None
        a=np.array([r for _,r in tr]); w=a>0
        exp=a.mean()*100
        if exp<emin: return None
        pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99
        ysum=defaultdict(float)
        for en,r in tr: ysum[yr(en)]+=r*100
        bad=[y for y,s in ysum.items() if s<ybad]; worst=min(ysum.values())
        if len(bad)>2 or worst<-30: return None
        if len(tr)/yrs_total<3: return None
        return dict(n=len(tr),exp=exp,pf=pf,wr=100*w.mean(),mo=len(tr)/yrs_total/12,
                    ys=' '.join(f'{y%100:02d}:{ysum[y]:+.1f}' if isfx else f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))
    return evaluate

def sim_time(C,ts,cand,side,hb,fee,cd_bars=6,holds=None):
    n=len(C); out=[]; last=-10**9
    for j,ci in enumerate(cand):
        if ci+1>=n: continue
        if ci-last<cd_bars: continue
        h=int(holds[j]) if holds is not None else hb
        ex=min(ci+max(h,1),n-1)
        out.append((int(ts[ci]),side*(C[ex]/C[ci]-1)-fee)); last=ci
    return out

def ema(x,s):
    a=2/(s+1); out=np.copy(x)
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out
def rollsum(x,w):
    cs=np.concatenate([[0.],np.cumsum(x)])
    out=np.full(len(x),np.nan); out[w-1:]=cs[w:]-cs[:-w]
    return out

def zgen(C,Hh,L,ts,win,k,b,mode,cdh,ma,sd,z,fee):
    n=len(C); epos=mode in('tl','rs')
    cand=np.flatnonzero(z>=k) if epos else np.flatnonzero(z<=-k)
    cand=cand[cand>=win]; out=[]; last=None; cd=cdh*H
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+HOLD,'right'))
        if hi<=ci+1: continue
        if mode=='tl':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            tp=price*(1+(1-tpb/price)); TP=Hh[ci+1:hi]>=tp
            r=(tp/price-1-fee) if TP.any() else C[hi-1]/price-1-fee
        elif mode=='rl':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            TP=Hh[ci+1:hi]>=tp
            r=(tp/price-1-fee) if TP.any() else C[hi-1]/price-1-fee
        elif mode=='ts':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            TPh=L[ci+1:hi]<=tp; SL=Hh[ci+1:hi]>=sl; hit=TPh|SL
            if hit.any(): j=int(np.argmax(hit)); xp=sl if SL[j] else tp
            else: xp=C[hi-1]
            r=(price-xp)/price-fee
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sl=price*(1+pct)
            TPh=L[ci+1:hi]<=tp; SL=Hh[ci+1:hi]>=sl; hit=TPh|SL
            if hit.any(): j=int(np.argmax(hit)); xp=sl if SL[j] else tp
            else: xp=C[hi-1]
            r=(price-xp)/price-fee
        out.append((int(ts[ci]),r)); last=ts[ci]
    return out

MODENM={'tl':'z추롱','rl':'z역롱','ts':'z추숏','rs':'z역숏'}
res=defaultdict(list)
for sym in NONFX+FX:
    m=_load_cache(sym,'240')
    if not m: continue
    isfx=sym in FX; fee=0.0002 if isfx else 0.0011
    evaluate=make_eval(fee,isfx)
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64)
    n=len(C); yrs_total=(int(ts[-1])-int(ts[0]))/(365.25*DAY)
    # 1) sweep 양방향
    for N in [12,30,60,120]:
        pl=np.full(n,np.nan); ph=np.full(n,np.nan)
        for i in range(N,n): pl[i]=L[i-N:i].min(); ph[i]=Hh[i-N:i].max()
        swL=np.flatnonzero((L<pl)&(C>pl)); swH=np.flatnonzero((Hh>ph)&(C<ph))
        for hb in [6,18,30,60]:
            e=evaluate(sim_time(C,ts,swL,1,hb,fee),yrs_total)
            if e: res[sym].append((f'sweep롱 N{N} T{hb}',e))
            e=evaluate(sim_time(C,ts,swH,-1,hb,fee),yrs_total)
            if e: res[sym].append((f'sweep숏 N{N} T{hb}',e))
    # 2) rsi 양방향
    d=np.diff(C,prepend=C[0]); up=np.maximum(d,0); dn=np.maximum(-d,0)
    for p in [2,3,4]:
        au=np.copy(up); ad=np.copy(dn)
        for i in range(1,n): au[i]=(au[i-1]*(p-1)+up[i])/p; ad[i]=(ad[i-1]*(p-1)+dn[i])/p
        rsi=100-100/(1+au/np.where(ad==0,1e-12,ad))
        for th in [10,15,20]:
            for hb in [6,18,30]:
                e=evaluate(sim_time(C,ts,np.flatnonzero(rsi<th),1,hb,fee),yrs_total)
                if e: res[sym].append((f'rsi{p}<{th}롱 T{hb}',e))
                e=evaluate(sim_time(C,ts,np.flatnonzero(rsi>100-th),-1,hb,fee),yrs_total)
                if e: res[sym].append((f'rsi{p}>{100-th}숏 T{hb}',e))
    # 3) 신저가롱 / 신고가숏
    for N in [24,42,90]:
        lo=np.full(n,np.nan); hi_=np.full(n,np.nan)
        for i in range(N,n): lo[i]=C[i-N:i].min(); hi_[i]=C[i-N:i].max()
        for hb in [6,18,30]:
            e=evaluate(sim_time(C,ts,np.flatnonzero(C<lo),1,hb,fee),yrs_total)
            if e: res[sym].append((f'n저가{N}롱 T{hb}',e))
            e=evaluate(sim_time(C,ts,np.flatnonzero(C>hi_),-1,hb,fee),yrs_total)
            if e: res[sym].append((f'n고가{N}숏 T{hb}',e))
    # 4) emapull 양방향
    e20=ema(C,20); e50=ema(C,50)
    for hb in [18,30,60]:
        e=evaluate(sim_time(C,ts,np.flatnonzero((e20>e50)&(L<=e20)&(C>e50)),1,hb,fee),yrs_total)
        if e: res[sym].append((f'ema눌림롱 T{hb}',e))
        e=evaluate(sim_time(C,ts,np.flatnonzero((e20<e50)&(Hh>=e20)&(C<e50)),-1,hb,fee),yrs_total)
        if e: res[sym].append((f'ema반등숏 T{hb}',e))
    # 5) OU 양방향
    lg=np.log(C)
    for W in [180,360,720]:
        ma_=rollsum(lg,W)/W; e_=lg-ma_
        s1=rollsum(e_*np.roll(e_,1),W); s2=rollsum(np.roll(e_,1)**2,W)
        phi=s1/np.where(s2==0,1e-12,s2)
        var=rollsum(e_*e_,W)/W-(rollsum(e_,W)/W)**2
        sde=np.sqrt(np.maximum(var,1e-12)); z=e_/sde
        hl=np.where((phi>0)&(phi<1),-np.log(2)/np.log(np.maximum(phi,1e-9)),np.nan)
        for k in [1.5,2.0,2.5]:
            for side,nm,cond in [(1,'롱',(z<=-k)),(-1,'숏',(z>=k))]:
                cand=np.flatnonzero(cond&(hl>1)&(hl<90))
                if len(cand)<30: continue
                for mm in [1.0,2.0]:
                    holds=np.clip(hl[cand]*mm,1,90)
                    e=evaluate(sim_time(C,ts,cand,side,0,fee,holds=holds),yrs_total)
                    if e: res[sym].append((f'OU{nm} W{W} k{k} m{mm}',e))
    # 6) ewz 4방향
    for s in [20,50,100,200]:
        mu=ema(C,s); resid=C-mu
        sig=ema(np.abs(resid),s); z=resid/np.where(sig==0,1e-12,sig)
        for k in [2.5,3.0]:
            for hb in [18,30,60]:
                e=evaluate(sim_time(C,ts,np.flatnonzero(z<=-k),1,hb,fee),yrs_total)
                if e: res[sym].append((f'ewz역롱 s{s} k{k} T{hb}',e))
                e=evaluate(sim_time(C,ts,np.flatnonzero(z>=k),-1,hb,fee),yrs_total)
                if e: res[sym].append((f'ewz역숏 s{s} k{k} T{hb}',e))
                e=evaluate(sim_time(C,ts,np.flatnonzero(z>=k),1,hb,fee),yrs_total)
                if e: res[sym].append((f'ewz추롱 s{s} k{k} T{hb}',e))
                e=evaluate(sim_time(C,ts,np.flatnonzero(z<=-k),-1,hb,fee),yrs_total)
                if e: res[sym].append((f'ewz추숏 s{s} k{k} T{hb}',e))
    # 7) z 4방향 (밴드TP 모델) — 13y 재검증부터 전 심볼 (env ZALL=1)
    if isfx or os.getenv('ZALL')=='1':
        for win in [30,60,90,120,180]:
            cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
            ma_=np.full(n,np.nan); sd_=np.full(n,np.nan)
            s_=cs[win:]-cs[:-win]; s2_=cs2[win:]-cs2[:-win]; mm_=s_/win
            ma_[win-1:]=mm_; sd_[win-1:]=np.sqrt(np.maximum(s2_/win-mm_*mm_,0))
            z_=np.where(sd_>0,(C-ma_)/sd_,np.nan)
            for mode in ['tl','rl','ts','rs']:
                for k in [2.0,2.5,3.0,3.5,4.0]:
                    for b in [-2,-1,0,1]:
                        for cdh in [12,24,48]:
                            tr=zgen(C,Hh,L,ts,win,k,b,mode,cdh,ma_,sd_,z_,fee)
                            e=evaluate(tr,yrs_total)
                            if e: res[sym].append((f'{MODENM[mode]} w{win*4//24}d K{k} B{b} cd{cdh}',e))
    print(f'{sym} done ({len(res[sym])})',flush=True)

print()
famcnt=defaultdict(set)
print(f'{"심볼":8} {"셀":26} {"n":>4} {"기대%":>6} {"PF":>5} {"승%":>4} {"월":>4} | 연도별')
for sym in NONFX+FX:
    rs=sorted(res.get(sym,[]),key=lambda x:-x[1]['exp']*min(x[1]['pf'],5))
    seen=set(); shown=0
    for name,e in rs:
        fam=name.split()[0]
        famcnt[fam].add(sym)
        if fam in seen: continue
        seen.add(fam); shown+=1
        if shown<=4:
            print(f'{sym:8} {name:26} {e["n"]:>4} {e["exp"]:>+6.2f} {e["pf"]:>5.2f} {e["wr"]:>3.0f}% {e["mo"]:>4.1f} | {e["ys"]}')
    if shown==0: print(f'{sym:8} (통과 없음)')
print()
print('=== 패밀리(방향)별 통과 심볼 ===')
for f,ss in sorted(famcnt.items(),key=lambda x:-len(x[1])):
    print(f'{f:14} {len(ss):>2}종: {" ".join(sorted(ss))}')
