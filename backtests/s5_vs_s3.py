# -*- coding: utf-8 -*-
"""S5(샹들리에·90d) vs S3(고정TP·15d) 추세롱 — 연도별(상승/하락장) 비교.
MT5 BTCUSD/ETHUSD(10년) + Bybit SOLUSDT/XRPUSDT(~5년). 진입연도 기준 버킷."""
import sys, os
os.environ['MAXHOLD_D']='15'          # S3 보유 15일 (daily_sweep import 전에)
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import daily_sweep as ds
from daily_quarterly import gen as gen_s3
from s5_chandelier import atr14, run_s5, DAY

# (sym, itv, S3(k1,b,cd), S5(k1,m,cd))
CFG=[
 ('BTCUSD','D10',(2.5,-2.0,2),(2.1,4.5,5)),
 ('ETHUSD','D10',(2.0,-2.4,5),(2.3,5.0,3)),
 ('SOLUSDT','D',(2.9,-3.0,1),(2.1,3.5,3)),
 ('XRPUSDT','D',(3.1,-3.0,1),(3.3,2.5,1)),
]
def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def stat(rets):
    if not rets: return None
    a=np.array(rets)
    return len(a), a.mean()*100, a.sum()*100, 100*(a>0).mean()

def main():
    for sym,itv,(k3,b3,c3),(k5,m5,c5) in CFG:
        ds.ITV=itv
        d=ds.prep(sym)
        if d is None: print(sym,'no data'); continue
        atr=atr14(d)
        s3=gen_s3(d,'trendlong',k3,b3,c3*DAY)                 # [(ts,ret)]
        r5,iv5,_=run_s5(d,atr,1,k5,m5,c5*DAY)
        s5=[(iv5[i][0],r5[i]) for i in range(len(r5))]
        by3={}; by5={}
        for t,r in s3: by3.setdefault(yr(t),[]).append(r)
        for t,r in s5: by5.setdefault(yr(t),[]).append(r)
        years=sorted(set(by3)|set(by5))
        a3=np.array([r for _,r in s3]); a5=np.array([r for _,r in s5])
        print(f'\n===== {sym} — S3롱(K{k3}/B{b3}/{c3}d/15d보유) vs S5롱(K{k5}/m{m5}/{c5}d/90d상한) =====')
        print(f'{"연도":>6} | {"S3: n":>6}{"평균":>8}{"합계":>9} | {"S5: n":>6}{"평균":>8}{"합계":>9}')
        for y in years:
            x3=stat(by3.get(y)); x5=stat(by5.get(y))
            c1=f'{x3[0]:>6}{x3[1]:>+7.1f}%{x3[2]:>+8.1f}%' if x3 else f'{"—":>22}'
            c2=f'{x5[0]:>6}{x5[1]:>+7.1f}%{x5[2]:>+8.1f}%' if x5 else f'{"—":>22}'
            print(f'{y:>6} | {c1} | {c2}')
        pf3=a3[a3>0].sum()/(-a3[a3<0].sum()) if (a3<0).any() else 99
        pf5=a5[a5>0].sum()/(-a5[a5<0].sum()) if (a5<0).any() else 99
        print(f'{"전체":>6} | {len(a3):>6}{a3.mean()*100:>+7.1f}%{a3.sum()*100:>+8.1f}% | {len(a5):>6}{a5.mean()*100:>+7.1f}%{a5.sum()*100:>+8.1f}%   (PF {pf3:.2f} vs {pf5:.2f})')

if __name__=='__main__':
    main()
