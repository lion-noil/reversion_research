# -*- coding: utf-8 -*-
"""S3 추세롱 변형: TP/SL 밴드가 보유기간에 걸쳐 선형 수축(1→1/3), 만기 종가청산.
+ 옵션: SL 터치 시 그 가격에서 1회 재진입(같은 pct%, 새 수축 게임).
베이스라인(고정밴드) vs 수축만 vs 재진입만 vs 둘다. 손절우선, fee0.11%, 진입쿨다운·동시보유."""
import sys, os
os.environ.setdefault('MAXHOLD_D','15')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import daily_sweep as ds
from daily_sweep import DAY, WIN
from s5_dd import dd_curve

FEE=0.0011

def sim(d,k1,b,cd_d,mh_d=15,shrink=1.0,reenter=False):
    """shrink: 만기시점 밴드 배율(1.0=고정, 1/3=제안). reenter: SL시 1회 재진입."""
    C,H,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cd=cd_d*DAY; mh=mh_d*DAY
    pos=[]  # dict(ei0(기록용 진입ts), ep, pct, start(ts), leg)
    out=[]; last=None; sig=z>=k1
    for i in range(WIN,n):
        t=ts[i]; keep=[]
        for p in pos:
            if p['start']>=t: keep.append(p); continue
            el=(t-p['start'])/mh
            if el>=1.0:
                out.append((p['ei0'],int(t),C[i]/p['ep']-1-FEE,p['ep'],1.0)); continue
            f=1.0-(1.0-shrink)*el
            tp=p['ep']*(1+p['pct']*f); sl=p['ep']*(1-p['pct']*f)
            if L[i]<=sl:
                out.append((p['ei0'],int(t),sl/p['ep']-1-FEE,p['ep'],1.0))
                if reenter and p['leg']==0:
                    keep.append(dict(ei0=int(t),ep=sl,pct=p['pct'],start=t,leg=1))
            elif H[i]>=tp:
                out.append((p['ei0'],int(t),tp/p['ep']-1-FEE,p['ep'],1.0))
            else: keep.append(p)
        pos=keep
        if sig[i] and i+1<n and (last is None or t-last>=cd):
            price=C[i]; tpb=ma[i]+b*sd[i]
            if tpb<price:
                pos.append(dict(ei0=int(t),ep=price,pct=1-tpb/price,start=t,leg=0)); last=t
    for p in pos:
        out.append((p['ei0'],int(ts[n-1]),C[n-1]/p['ep']-1-FEE,p['ep'],1.0))
    return out

CFG=[('BTCUSD','D10',2.5,-2.0,2),('ETHUSD','D10',2.0,-2.4,5),
     ('SOLUSDT','D',2.9,-3.0,1),('XRPUSDT','D',3.1,-3.0,1)]

def row(tag,d,trs):
    a=np.array([t[2] for t in trs])
    if len(a)==0: print(f'  {tag:22} 거래없음'); return
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    tot,mdd,_=dd_curve(d,trs)
    print(f'  {tag:22} n{len(a):>4} 승{100*(a>0).mean():>3.0f}% 기대{a.mean()*100:>+7.2f}% PF{pf:>6.2f} | 총{tot:>+7.0f}%p DD{mdd:>6.0f}%p MAR{(tot/mdd if mdd>0 else 99):>5.1f}')

def main():
    for sym,itv,k1,b,cd in CFG:
        ds.ITV=itv; d=ds.prep(sym)
        if d is None: continue
        print(f'\n===== {sym} S3롱 K{k1}/B{b}/{cd}d/보유15d =====')
        row('베이스라인(고정밴드)',d,sim(d,k1,b,cd,shrink=1.0,reenter=False))
        row('수축만(1→1/3)',d,sim(d,k1,b,cd,shrink=1/3,reenter=False))
        row('재진입만(고정밴드)',d,sim(d,k1,b,cd,shrink=1.0,reenter=True))
        row('수축+재진입(제안)',d,sim(d,k1,b,cd,shrink=1/3,reenter=True))

if __name__=='__main__':
    main()
