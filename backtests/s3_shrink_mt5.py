# -*- coding: utf-8 -*-
"""MT5 비환율 10종 S3 추세롱 — 밴드수축(1→1/3)+재진입 변형 vs 베이스라인, 연도별 비교.
파라미터=daily_quarterly_mt5.json S1_long(일봉 추세롱, 보유15d·maxc15 최적)."""
import sys, os, json
os.environ.setdefault('MAXHOLD_D','15')
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import daily_sweep as ds
from s3_shrink import sim
from s5_dd import dd_curve

def yr(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def main():
    J=json.load(open(os.path.join(os.path.dirname(__file__),'daily_quarterly_mt5.json'),encoding='utf-8'))
    ds.ITV='D10'
    summary=[]
    for sym,row in J.items():
        p=row.get('S1_long')
        if not p: continue
        d=ds.prep(sym)
        if d is None: continue
        k1,b,cd=p['k1'],p['b'],p['cd']
        base=sim(d,k1,b,cd,shrink=1.0,reenter=False)
        var =sim(d,k1,b,cd,shrink=1/3,reenter=True)
        bb={};vv={}
        for t in base: bb.setdefault(yr(t[0]),[]).append(t[2])
        for t in var:  vv.setdefault(yr(t[0]),[]).append(t[2])
        years=sorted(set(bb)|set(vv))
        tb,db,_=dd_curve(d,base); tv,dv,_=dd_curve(d,var)
        mb=tb/db if db>0 else 99; mv=tv/dv if dv>0 else 99
        print(f'\n===== {sym} S3롱 K{k1}/B{b}/{cd}d =====')
        print(f'{"연도":>6} | {"기존 합계":>10} | {"수축+재진입":>10} | {"차이":>8}')
        for y in years:
            sb=sum(bb.get(y,[]))*100; sv=sum(vv.get(y,[]))*100
            print(f'{y:>6} | {sb:>+9.1f}% | {sv:>+9.1f}% | {sv-sb:>+7.1f}%')
        print(f'{"전체":>6} | {tb:>+9.0f}%p | {tv:>+9.0f}%p |  DD {db:.0f}→{dv:.0f}%p  MAR {mb:.1f}→{mv:.1f}')
        summary.append((sym,tb,tv,db,dv,mb,mv))
    print('\n===== 요약 (전체 총수익 / DD / MAR: 기존→변형) =====')
    better=0
    for sym,tb,tv,db,dv,mb,mv in summary:
        mark='✅' if mv>mb else ('≈' if abs(mv-mb)<0.15 else '❌')
        if mv>mb: better+=1
        print(f'  {sym:8} 총 {tb:>+6.0f}→{tv:>+6.0f}%p | DD {db:>4.0f}→{dv:>4.0f}%p | MAR {mb:>4.1f}→{mv:>4.1f} {mark}')
    print(f'\nMAR 개선: {better}/{len(summary)}심볼')

if __name__=='__main__':
    main()
