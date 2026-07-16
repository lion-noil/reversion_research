# -*- coding: utf-8 -*-
"""portfolio_sim.py와 동일 엔진·채택셋으로 돌리되, 분기별×심볼 실현손익 분해.
$10k, 랏=자산5%, 최대200랏, 14일보유, fee0.11%. 각 거래의 실현$를 진입분기·심볼로 버킷.
출력: 분기 총괄표 + 심볼별 3년 + 분기×심볼 CSV.
"""
import sys, os, json, heapq, csv
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import portfolio_sim as PS
from portfolio_sim import trades, cd_ms, prep, INIT, LOT_FRAC, MAX_LOTS, DAY, MT5, CRY, FILES, DIRS, RES

def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}-Q{(t.month-1)//3+1}'

def main():
    picks=[]
    for dname,dkey in DIRS:
        for mkt,syms in [('mt5',MT5),('crypto',CRY)]:
            D=json.load(open(os.path.join(RES,f'{mkt}_14d{FILES[dkey]}.json'),encoding='utf-8'))
            for s in syms:
                if s in D and D[s] and D[s][0]['exp']>0:
                    r=D[s][0]; picks.append(dict(dkey=dkey,sym=s,k1=r['k1'],b=r['b'],cd=r['cd']))
    pc={}; alltr=[]
    for p in picks:
        d=pc.get(p['sym']) or prep(p['sym']); pc[p['sym']]=d
        for en,ex,ret in trades(d,p['dkey'],p['k1'],p['b'],cd_ms(p['cd'])):
            alltr.append((en,ex,ret,p['sym']))
    alltr.sort(key=lambda x:x[0])
    eq=INIT; heap=[]; openc=0
    qpnl=defaultdict(float); qcnt=defaultdict(int)            # 분기 총
    sq=defaultdict(float); sqc=defaultdict(int)               # (분기,심볼)
    spnl=defaultdict(float); scnt=defaultdict(int)            # 심볼 총
    qeq={}                                                    # 분기말 자본
    span1=max(t[1] for t in alltr)
    def realize(t):
        nonlocal eq,openc
        while heap and heap[0][0]<=t:
            _,ret,notional,sym,enq=heapq.heappop(heap)
            pnl=notional*ret; eq+=pnl; openc-=1
            qpnl[enq]+=pnl; qcnt[enq]+=1; sq[(enq,sym)]+=pnl; sqc[(enq,sym)]+=1
            spnl[sym]+=pnl; scnt[sym]+=1
    for en,ex,ret,sym in alltr:
        realize(en)
        if openc<MAX_LOTS and eq>0:
            heapq.heappush(heap,(ex,ret,LOT_FRAC*eq,sym,qof(en))); openc+=1
        # 분기말 자본 스냅(진입 시각 기준 분기 갱신)
        qeq[qof(en)]=eq
    realize(span1+1)
    # 마지막 분기 자본 갱신
    qs=sorted(qpnl)
    # 분기 총괄(실현 누적으로 자본 재구성)
    print(f'$10k 포트폴리오 · 분기별 (랏5%/최대200/14일/fee0.11%)')
    print(f'{"분기":>9}{"거래":>6}{"실현손익$":>12}{"분기%":>9}{"기말자본$":>13}')
    run=INIT
    for q in qs:
        st=run; run+=qpnl[q]
        print(f'{q:>9}{qcnt[q]:>6}{qpnl[q]:>+12.0f}{100*qpnl[q]/st:>+8.1f}%{run:>13,.0f}')
    print(f'\n총: $10,000 → ${run:,.0f}  ({run/INIT:.1f}배)')
    # 심볼별 3년
    print(f'\n=== 심볼별 3년 실현손익 (기여순) ===')
    for sym,p in sorted(spnl.items(),key=lambda x:-x[1]):
        print(f'  {sym:9} {p:>+10,.0f}$  ({scnt[sym]}건, 전체대비 {100*p/(run-INIT):>4.1f}%)')
    # 분기×심볼 CSV
    allsyms=sorted({s for _,s in sq})
    with open(os.path.join(os.path.dirname(__file__),'portfolio_quarterly.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['quarter']+allsyms+['TOTAL'])
        for q in qs:
            w.writerow([q]+[round(sq.get((q,s),0)) for s in allsyms]+[round(qpnl[q])])
    print('\n→ 분기×심볼 상세: portfolio_quarterly.csv')

if __name__=='__main__':
    main()
