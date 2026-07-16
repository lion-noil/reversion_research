# -*- coding: utf-8 -*-
"""S22 Bybit 확정 스윕 — 4h봉, BTC/ETH/SOL/XRP. 정밀 격자 + 연도균형 + 심볼별 패밀리 최적 + 신호 중복도.
패밀리: z추세롱(밴드TP)/z역추롱(밴드TP)/ewz추롱/ewz추숏/ewz역롱/급락페이드/sweep롱.
공통: SL無(숏은 시간청산=사실상 SL 대체), fee 0.11%, 필터 n≥30·기대≥0.4%·마이너스연도(<-3%p)≤2·최악>-30%p·월≥0.25."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000; HOLD=15*DAY

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
    mo=len(tr)/yrs_total/12
    if mo<0.25: return None
    return dict(n=len(tr),exp=exp,pf=pf,wr=100*w.mean(),mo=mo,score=exp*min(pf,5),
                ys=' '.join(f'{y%100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))

def ema(x,s):
    a=2/(s+1); out=np.copy(x)
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out

def zarr(C,win):
    n=len(C); cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

def sim_z(C,Hh,ts,win,k,b,mode,cdh,ma,sd,z):
    n=len(C); cand=np.flatnonzero(z>=k) if mode=='tl' else np.flatnonzero(z<=-k)
    cand=cand[cand>=win]; out=[]; last=None; cd=cdh*H
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+HOLD,'right'))
        if hi<=ci+1: continue
        if mode=='tl':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            tp=price*(1+(1-tpb/price))
        else:
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
        TP=Hh[ci+1:hi]>=tp
        r=(tp/price-1-FEE) if TP.any() else C[hi-1]/price-1-FEE
        out.append((int(ts[ci]),r)); last=ts[ci]
    return out

def sim_time(C,ts,cand,side,hb,cd_bars=6):
    n=len(C); out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<cd_bars: continue
        ex=min(ci+hb,n-1)
        out.append((int(ts[ci]),side*(C[ex]/C[ci]-1)-FEE)); last=ci
    return out

def sim_fade(C,Hh,ts,cand,ret,kind,cap):
    n=len(C); out=[]; last=-10**9
    for ci in cand:
        if ci+1>=n: continue
        if ci-last<6: continue
        price=C[ci]; hi=min(ci+1+cap,n)
        if kind=='R':
            tp=price*(1+1.5*abs(ret[ci])); TP=Hh[ci+1:hi]>=tp
            r=(tp/price-1-FEE) if TP.any() else C[hi-1]/price-1-FEE
        else:
            r=C[hi-1]/price-1-FEE
        out.append((int(ts[ci]),r)); last=ci
    return out

SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT']
best=defaultdict(dict)   # sym -> fam -> (name, eval, trades)
for sym in SYMS:
    m=_load_cache(sym,'240')
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64)
    n=len(C); yt=(int(ts[-1])-int(ts[0]))/(365.25*DAY)
    def keep(fam,name,tr):
        e=evaluate(tr,yt)
        if e and (fam not in best[sym] or e['score']>best[sym][fam][1]['score']):
            best[sym][fam]=(name,e,tr)
    # z 밴드TP (추세롱/역추롱)
    for win in [60,90,120,180,240]:
        z,ma,sd=zarr(C,win)
        for k in [2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.5]:
            for b in [-3,-2.5,-2,-1,0,1]:
                for cdh in [12,24,48]:
                    keep('z추세롱',f'z추롱 w{win//6}d K{k} B{b} cd{cdh}',sim_z(C,Hh,ts,win,k,b,'tl',cdh,ma,sd,z))
            for b in [-1,-0.5,0,0.5,1,1.5]:
                for cdh in [12,24,48]:
                    keep('z역추롱',f'z역롱 w{win//6}d K{k} B{b} cd{cdh}',sim_z(C,Hh,ts,win,k,b,'rl',cdh,ma,sd,z))
    # ewz 3방향
    for s in [20,50,100,150,200,300]:
        mu=ema(C,s); resid=C-mu; sig=ema(np.abs(resid),s)
        z=resid/np.where(sig==0,1e-12,sig)
        for k in [2.0,2.5,3.0,3.5]:
            for hb in [18,30,60,90]:
                keep('ewz추롱',f'ewz추롱 s{s} k{k} T{hb}',sim_time(C,ts,np.flatnonzero(z>=k),1,hb))
                keep('ewz추숏',f'ewz추숏 s{s} k{k} T{hb}',sim_time(C,ts,np.flatnonzero(z<=-k),-1,hb))
                keep('ewz역롱',f'ewz역롱 s{s} k{k} T{hb}',sim_time(C,ts,np.flatnonzero(z<=-k),1,hb))
    # 급락페이드
    for M in [3,6,12,18,24,30]:
        ret=C/np.roll(C,M)-1; ret[:M]=np.nan
        for thr in [0.05,0.07,0.10,0.15]:
            cand=np.flatnonzero(ret<=-thr)
            if len(cand)<30: continue
            for kind,cap in [('T',6),('T',30),('T',60),('R',30),('R',60)]:
                keep('페이드',f'fade {M*4}h≤-{thr*100:.0f}% {kind}{cap*4//24}d',sim_fade(C,Hh,ts,cand,ret,kind,cap))
    # sweep롱
    for N in [30,60,120,180]:
        pl=np.full(n,np.nan)
        for i in range(N,n): pl[i]=L[i-N:i].min()
        sw=np.flatnonzero((L<pl)&(C>pl))
        for hb in [18,30,60]:
            keep('sweep롱',f'sweep N{N} T{hb}',sim_time(C,ts,sw,1,hb))
    print(f'{sym} done',flush=True)

print()
print(f'{"심볼":8} {"패밀리":8} {"셀":26} {"n":>4} {"기대%":>6} {"PF":>5} {"승%":>4} {"월":>4} | 연도별')
allcells=[]
for sym in SYMS:
    for fam,(name,e,tr) in sorted(best[sym].items(),key=lambda x:-x[1][1]['score']):
        print(f'{sym:8} {fam:8} {name:26} {e["n"]:>4} {e["exp"]:>+6.2f} {e["pf"]:>5.2f} {e["wr"]:>3.0f}% {e["mo"]:>4.1f} | {e["ys"]}')
        allcells.append((f'{sym[:3]}-{fam}',tr))
    print()
# 신호 중복도: 진입시각 24h 이내 겹침 비율
print('=== 신호 중복도 (행 셀 진입 중 열 셀 진입과 24h내 겹침 %) ===')
names=[nm for nm,_ in allcells]
ents=[np.array(sorted(en for en,_ in tr)) for _,tr in allcells]
hdr='                    '+' '.join(f'{nm[:12]:>13}' for nm in names)
print(hdr)
W=24*H
for i,nm in enumerate(names):
    row=[]
    for j in range(len(names)):
        if i==j: row.append('    -'); continue
        if len(ents[i])==0: row.append('    0'); continue
        idx=np.searchsorted(ents[j],ents[i])
        near=0
        for x,ii in zip(ents[i],idx):
            c=[]
            if ii<len(ents[j]): c.append(abs(int(ents[j][ii])-int(x)))
            if ii>0: c.append(abs(int(ents[j][ii-1])-int(x)))
            if c and min(c)<=W: near+=1
        row.append(f'{100*near/len(ents[i]):>4.0f}%')
    print(f'{nm:20} '+' '.join(f'{v:>13}' for v in row))
