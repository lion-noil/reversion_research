# -*- coding: utf-8 -*-
"""S3 vs S5 vs 블렌드(반반) — 반기(6개월) 성적 + MAR. 진입시점 기준 버킷, 노셔널1/트레이드."""
import sys, os
os.environ['MAXHOLD_D']='15'
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import daily_sweep as ds
from daily_portfolio import gen as gen4
from s5_chandelier import atr14, run_s5, DAY
from s5_dd import dd_curve, s3_trades, s5_trades

CFG=[
 ('BTCUSD','D10',(2.5,-2.0,2),(2.1,4.5,5)),
 ('ETHUSD','D10',(2.0,-2.4,5),(2.3,4.5,3)),
 ('SOLUSDT','D',(2.9,-3.0,1),(2.1,4.5,3)),
 ('XRPUSDT','D',(3.1,-3.0,1),(3.3,2.5,1)),
]
def half(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}H{1 if t.month<=6 else 2}'

def main():
    for sym,itv,p3,p5 in CFG:
        ds.ITV=itv; d=ds.prep(sym)
        if d is None: continue
        atr=atr14(d)
        t3=s3_trades(d,*p3); t5=s5_trades(d,atr,*p5)
        b3={};b5={}
        for tr in t3: b3.setdefault(half(tr[0]),[]).append(tr[2])
        for tr in t5: b5.setdefault(half(tr[0]),[]).append(tr[2])
        halves=sorted(set(b3)|set(b5))
        print(f'\n===== {sym} — S3(K{p3[0]}) vs S5(K{p5[0]}/m{p5[1]}) vs 블렌드50:50 =====')
        print(f'{"반기":>8} | {"S3 n":>5}{"합계":>9} | {"S5 n":>5}{"합계":>9} | {"블렌드":>9}')
        for h in halves:
            x3=b3.get(h,[]); x5=b5.get(h,[])
            s3s=sum(x3)*100; s5s=sum(x5)*100; bl=(s3s+s5s)/2
            print(f'{h:>8} | {len(x3):>5}{s3s:>+8.1f}% | {len(x5):>5}{s5s:>+8.1f}% | {bl:>+8.1f}%')
        # MAR: 단독 vs 블렌드(양쪽 0.5 노셔널)
        tot3,dd3,_=dd_curve(d,t3); tot5,dd5,_=dd_curve(d,t5)
        tb=[(en,ex,r*0.5,ep,sd*0.5) for en,ex,r,ep,sd in t3]+[(en,ex,r*0.5,ep,sd*0.5) for en,ex,r,ep,sd in t5]
        totb,ddb,_=dd_curve(d,tb)
        print(f'{"MAR":>8} | S3 {tot3/dd3 if dd3>0 else 99:.1f} (DD{dd3:.0f}%p) | S5 {tot5/dd5 if dd5>0 else 99:.1f} (DD{dd5:.0f}%p) | 블렌드 {totb/ddb if ddb>0 else 99:.1f} (DD{ddb:.0f}%p, 총{totb:+.0f}%p)')

if __name__=='__main__':
    main()
