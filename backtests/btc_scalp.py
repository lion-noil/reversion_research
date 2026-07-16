# -*- coding: utf-8 -*-
"""초단기 스캘핑 탐색 — 하루 2~3회(월 60~90) 대역, 메이커 수수료 가정(왕복 0.04%).
1분봉 6.3년. 창 2h/6h/12h × K 1.75~3.0 × B −2~2 × 쿨다운 15m~1h × 보유 0.5/1/2d. 4방향.
env: FEE(0.0004), PMLO(40), PMHI(120)"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000
FEE=float(os.getenv('FEE','0.0004'))
PMLO=float(os.getenv('PMLO','40')); PMHI=float(os.getenv('PMHI','120'))
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

def run(z,ma,sd,win,DIR,k1,b,cd_ms,mh_ms):
    epos = DIR in('trendlong','short')
    is_long = DIR in('trendlong','long')
    cand=np.flatnonzero(z>=k1) if epos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=win]; rets=[]; iv=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<cd_ms: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+mh_ms,'right'))
        if hi<=ci+1: continue
        if DIR=='trendlong':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            pct=1-tpb/price; tp=price*(1+pct); sl=price*(1-pct)
            SL=L1[ci+1:hi]<=sl; TP=H1[ci+1:hi]>=tp
        elif DIR=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sl=price*(1-pct)
            SL=L1[ci+1:hi]<=sl; TP=H1[ci+1:hi]>=tp
        elif DIR=='trend':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            SL=H1[ci+1:hi]>=sl; TP=L1[ci+1:hi]<=tp
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sl=price*(1+pct)
            SL=H1[ci+1:hi]>=sl; TP=L1[ci+1:hi]<=tp
        hit=SL|TP
        if hit.any():
            j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
        else: ex=hi-1; xp=C1[ex]
        r=(xp/price-1-FEE) if is_long else ((price-xp)/price-FEE)
        rets.append(r); iv.append((int(ts1[ci]),int(ts1[ex]))); last=ts1[ci]
    ev=[]
    for en,ex_ in iv: ev.append((en,1)); ev.append((ex_,-1))
    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
    for _,dd in ev: cur+=dd; mx=max(mx,cur)
    return np.array(rets),mx

t0=time.time(); allrows=[]
for win in [120,360,720]:
    z,ma,sd=zma(win)
    for DIR in ['trendlong','trend','long','short']:
        for k1 in [1.75,2.0,2.25,2.5,3.0]:
            for b in [-2.0,-1.0,0.0,1.0,2.0]:
                for cdh in [0.25,0.5,1.0]:
                    for mh in [0.5,1.0,2.0]:
                        a,mx=run(z,ma,sd,win,DIR,k1,b,int(cdh*H),int(mh*DAY))
                        pm=len(a)/months
                        if not (PMLO<=pm<=PMHI) or mx>15: continue
                        pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
                        allrows.append(dict(win=win,dir=DIR,k1=k1,b=b,cd=cdh,mh=mh,n=len(a),pm=round(pm,1),
                            mc=mx,wr=round(100*float((a>0).mean())),exp=round(float(a.mean())*100,3),pf=round(float(pf),2)))
    print(f'win={win//60}h done ({time.time()-t0:.0f}s)',flush=True)
json.dump(allrows,open(os.path.join(os.path.dirname(__file__),'btc_scalp.json'),'w'))
allrows.sort(key=lambda r:(r['exp'],r['pf']),reverse=True)
print(f'\n월{PMLO:.0f}~{PMHI:.0f}회 대역 통과 {len(allrows)}조합 (fee 왕복 {FEE*100:.2f}%)')
print('상위 12:')
for r in allrows[:12]:
    print(f"  {r['dir']:9} 창{r['win']//60}h K{r['k1']} B{r['b']}/{r['cd']}h/보유{r['mh']}d n{r['n']}({r['pm']}/월) mc{r['mc']} 기대{r['exp']:+.3f}% PF{r['pf']} 승{r['wr']}% 월합{r['pm']*r['exp']:+.1f}%p")
