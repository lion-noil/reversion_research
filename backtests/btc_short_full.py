# -*- coding: utf-8 -*-
"""숏 2방향 SL 포함 전차원 재최적화: 창(6h~24h) × K(2.5~6) × B × 보유(2/7/14d) × SL(0.5~3,무) × 쿨다운(1/3h).
env: DIR=short|trend"""
import sys, os, time
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011
DIR=os.getenv('DIR','short')
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

def sim(z,ma,sd,win,k1,b,mh_d,slm,cd_h):
    entry_pos = DIR=='short'
    cand=np.flatnonzero(z>=k1) if entry_pos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=win]
    mh=int(mh_d*DAY); cd=int(cd_h*H); out=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<cd: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+mh,'right'))
        if hi<=ci+1: continue
        if DIR=='trend':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct)
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price
        TP=L1[ci+1:hi]<=tp
        if slm is None: SLh=np.zeros(hi-ci-1,bool); sl=None
        else: sl=price*(1+pct*slm); SLh=H1[ci+1:hi]>=sl
        hit=SLh|TP
        if hit.any():
            j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SLh[j] else tp
        else: ex=hi-1; xp=C1[ex]
        out.append((int(ts1[ci]),(price-xp)/price-FEE)); last=ts1[ci]
    return out

t0=time.time(); best=None; rows=[]
for win in [360,720,1080,1320,1440]:
    z,ma,sd=zma(win)
    for k1 in [2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]:
        for b in [round(-3+0.5*i,1) for i in range(9)]:
            for mh in [2,7,14]:
                for slm in [0.5,1.0,1.5,2.0,3.0,None]:
                    for cdh in [1,3]:
                        tr=sim(z,ma,sd,win,k1,b,mh,slm,cdh)
                        if len(tr)<40: continue
                        a=np.array([r for _,r in tr]); e=a.mean()
                        pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
                        rows.append((e,pf,win,k1,b,mh,slm,cdh,len(a)))
                        if best is None or e>best[0]: best=(e,pf,win,k1,b,mh,slm,cdh,tr)
    print(f'win={win//60}h done ({time.time()-t0:.0f}s)',flush=True)
e,pf,win,k1,b,mh,slm,cdh,tr=best
a=np.array([r for _,r in tr])
print(f'\n★{DIR} 조인트 최적: 창{win//60}h K{k1} B{b} 보유{mh}d SL={slm if slm else "무"}×TP cd{cdh}h')
print(f'  n{len(a)}({len(a)/months:.1f}/월) 기대{e*100:+.3f}% PF{pf:.2f} 승{100*(a>0).mean():.0f}% 최악{a.min()*100:+.1f}% 월합{len(a)/months*e*100:+.1f}%p')
# 상위 8 (견고성 확인)
print('  상위 8:')
for r in sorted(rows,reverse=True)[:8]:
    print(f'    창{r[2]//60}h K{r[3]} B{r[4]} {r[5]}d SL{r[6] if r[6] else "무"} cd{r[7]}h n{r[8]} 기대{r[0]*100:+.3f}%/PF{r[1]:.2f}')
def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
byq={}
for t_,r in tr: byq.setdefault(qof(t_),[]).append(r)
print('  분기별:')
for k in sorted(byq):
    aa=np.array(byq[k])
    print(f'    {k}: n{len(aa):>3} 합{aa.sum()*100:>+7.1f}% 승{100*(aa>0).mean():>3.0f}% 최악{aa.min()*100:>+6.1f}%')
