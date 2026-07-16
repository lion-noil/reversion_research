# -*- coding: utf-8 -*-
"""S3 vs S5 — 드로우다운 포함 비교. 트레이드 노셔널 1 고정(무복리),
에쿼티 = 실현누적 + 보유중 MTM(일 종가), 낙폭은 %p(노셔널 대비)와 MAR(총수익/최대낙폭)."""
import sys, os
os.environ['MAXHOLD_D']='15'
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import daily_sweep as ds
from daily_portfolio import gen as gen4          # (en,ex,r,entry_price) 4모드
from s5_chandelier import atr14, run_s5, DAY

def dd_curve(d, trades):
    """trades=[(en,ex,r,ep,side)]. 일단위 MTM 에쿼티(%p) → (총수익, 최대낙폭, 평균동시보유)"""
    ts,C=d['ts'],d['C']; n=len(ts)
    trs=sorted(trades); m=len(trs)
    realized=0.0; openl=[]; ptr=0; peak=-1e18; mdd=0.0; ccnt=[]
    for i in range(n):
        t=ts[i]
        while ptr<m and trs[ptr][0]<=t: openl.append(trs[ptr]); ptr+=1
        keep=[]
        for tr in openl:
            if tr[1]<=t: realized+=tr[2]
            else: keep.append(tr)
        openl=keep
        un=sum(tr[4]*(C[i]/tr[3]-1.0) for tr in openl)
        eq=(realized+un)*100
        peak=max(peak,eq); mdd=max(mdd,peak-eq); ccnt.append(len(openl))
    return realized*100, mdd, float(np.mean(ccnt))

def s3_trades(d,k1,b,cd):
    return [(en,ex,r,ep,1.0) for en,ex,r,ep in gen4(d,'trendlong',k1,b,cd*DAY)]

def s5_trades(d,atr,k1,m,cd):
    rets,iv,_=run_s5(d,atr,1,k1,m,cd*DAY)
    ts=d['ts']; C=d['C']
    out=[]
    for i,(en,ex) in enumerate(iv):
        ci=int(np.searchsorted(ts,en)); out.append((en,ex,rets[i],float(C[ci]),1.0))
    return out

CFG=[  # sym, itv, S3(k,b,cd), S5빈도(k,m,cd), S5질(k,m,cd)
 ('BTCUSD','D10',(2.5,-2.0,2),(2.1,4.5,5),(2.5,4.5,5)),
 ('ETHUSD','D10',(2.0,-2.4,5),(2.3,4.5,3),(2.6,4.5,3)),
 ('SOLUSDT','D',(2.9,-3.0,1),(2.1,4.5,3),(2.4,4.5,3)),
 ('XRPUSDT','D',(3.1,-3.0,1),(3.3,2.5,1),(3.3,4.5,1)),
]
def row(tag,d,trs):
    a=np.array([t[2] for t in trs])
    if len(a)==0: print(f'  {tag:24} 거래없음'); return
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    tot,mdd,avgc=dd_curve(d,trs)
    mar=tot/mdd if mdd>0 else 99
    print(f'  {tag:24} n{len(a):>4} 기대{a.mean()*100:>+7.2f}% PF{pf:>6.2f} | 총{tot:>+8.0f}%p 최대DD{mdd:>7.0f}%p MAR{mar:>5.1f} 동시{avgc:.1f}')

def main():
    for sym,itv,p3,p5a,p5b in CFG:
        ds.ITV=itv; d=ds.prep(sym)
        if d is None: continue
        atr=atr14(d)
        print(f'\n===== {sym} ({itv}) — 노셔널1/트레이드, MTM 일단위 =====')
        row(f'S3 고정TP K{p3[0]}',d,s3_trades(d,*p3))
        row(f'S5 샹들리에 K{p5a[0]}/m{p5a[1]}',d,s5_trades(d,atr,*p5a))
        row(f'S5 샹들리에 K{p5b[0]}/m{p5b[1]}',d,s5_trades(d,atr,*p5b))

if __name__=='__main__':
    main()
