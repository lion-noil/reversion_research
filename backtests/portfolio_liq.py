# -*- coding: utf-8 -*-
"""증거금 20%(=5배 레버리지) 가정, 보유 중 미실현까지 마크투마켓 → 청산 발생 검토.
- 채택셋·랏사이징은 portfolio_sim과 동일($10k, 랏=자산5%), 단 총명목 ≤ LEV×자산(증거금캡).
- 1시간봉 저(롱)/고(숏)로 장중 최악 MTM 자본 평가 → 청산(자본≤유지증거금) 발생시각·횟수.
"""
import sys, os, json, heapq
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_s1_quarters_mt5 import prep, WIN
from portfolio_sim import cd_ms, MT5, CRY, FILES, DIRS, RES
from reversion_calibrator import _load_cache

FEE=0.0011; DAY=86400*1000; MH=14*DAY; INIT=10_000.0; LOT_FRAC=0.05
LEV=5.0                     # 증거금 20% = 5배
MAINT=0.0                   # 유지증거금: 자본 MTM ≤ MAINT 면 청산 (0=완전소진 기준, 보수적으론 더 높음)
LONG={'long','trendlong'}

def gen(d, dkey, k1, b, cd):
    C,H,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cand=np.flatnonzero(z>=k1) if dkey in('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=WIN]; out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+MH,'right'))
        if hi<=ci+1: continue
        if dkey=='short':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sl=price*(1+pct); SLh=H[ci+1:hi]>=sl; TPh=L[ci+1:hi]<=tp; hit=SLh|TPh
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; ret=(price-sl)/price-FEE if SLh[j] else (price-tp)/price-FEE
            else: ex=hi-1; ret=(price-C[ex])/price-FEE
        elif dkey=='trend':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; tpS=price*(1-pct); slS=price*(1+pct); SLh=H[ci+1:hi]>=slS; TPh=L[ci+1:hi]<=tpS; hit=SLh|TPh
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; ret=(price-slS)/price-FEE if SLh[j] else (price-tpS)/price-FEE
            else: ex=hi-1; ret=(price-C[ex])/price-FEE
        elif dkey=='trendlong':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tpL=price*(1+pct); slL=price*(1-pct); SLh=L[ci+1:hi]<=slL; TPh=H[ci+1:hi]>=tpL; hit=SLh|TPh
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; ret=slL/price-1-FEE if SLh[j] else tpL/price-1-FEE
            else: ex=hi-1; ret=C[ex]/price-1-FEE
        else:
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sl=price*(1-pct); SLh=L[ci+1:hi]<=sl; TPh=H[ci+1:hi]>=tp; hit=SLh|TPh
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; ret=sl/price-1-FEE if SLh[j] else tp/price-1-FEE
            else: ex=hi-1; ret=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),float(ret),float(price)))
        last=ts[ci]
    return out

def main():
    picks=[]
    for dname,dkey in DIRS:
        for mkt,syms in [('mt5',MT5),('crypto',CRY)]:
            D=json.load(open(os.path.join(RES,f'{mkt}_14d{FILES[dkey]}.json'),encoding='utf-8'))
            for s in syms:
                if s in D and D[s] and D[s][0]['exp']>0:
                    r=D[s][0]; picks.append((dkey,s,r['k1'],r['b'],r['cd']))
    pc={}; tr=[]   # (entry,exit,ret,entry_price,sym,sign)
    for dkey,s,k1,b,cd in picks:
        d=pc.get(s) or prep(s); pc[s]=d
        sign=1.0 if dkey in LONG else -1.0
        for en,ex,ret,ep in gen(d,dkey,k1,b,cd_ms(cd)):
            tr.append((en,ex,ret,ep,s,sign))
    print(f'채택 {len(picks)} · 거래 {len(tr):,}')
    # 1h OHLC 가격맵
    px={}
    for s in {t[4] for t in tr}:
        c=_load_cache(s,'60')
        px[s]=(np.array([x['start'] for x in c],np.int64),np.array([x['high'] for x in c]),
               np.array([x['low'] for x in c]),np.array([x['close'] for x in c]))
    def worst_price(s,t,sign):
        ts_,hi,lo,cl=px[s]; i=int(np.searchsorted(ts_,t,'right'))-1
        if i<0: return None
        return lo[i] if sign>0 else hi[i]   # 롱=저가, 숏=고가 (장중 최악)
    # 이벤트: 진입/청산 + 1h 그리드
    grid=sorted({int(g) for s in px for g in px[s][0]})
    ev=[]
    for i,(en,ex,ret,ep,s,sign) in enumerate(tr): ev.append((en,0,i)); ev.append((ex,2,i))
    for g in grid: ev.append((g,1,-1))
    ev.sort(key=lambda x:(x[0],x[1]))
    eqr=INIT; opn={}; tot_notional=0.0; skipped=0
    min_mtm=INIT; min_at=0; peak=INIT; mdd=0.0; liq=None; liq_n=0
    for t,typ,i in ev:
        if typ==0:
            notional=LOT_FRAC*eqr
            if tot_notional+notional>LEV*eqr: skipped+=1; opn[i]=None; continue
            en,ex,rr,ep,s,sign=tr[i]; opn[i]=(notional,ep,s,sign,rr); tot_notional+=notional
        elif typ==2:
            v=opn.get(i)
            if not v: continue
            notional,ep,s,sign,rr=v; eqr+=notional*rr; tot_notional-=notional; del opn[i]
        else:  # 그리드 MTM(장중 최악)
            if not opn:
                mtm=eqr
            else:
                unreal=0.0
                for v in opn.values():
                    if not v: continue
                    notional,ep,s,sign,rr=v; wp=worst_price(s,t,sign)
                    if wp is None: continue
                    unreal+=notional*sign*(wp/ep-1.0)
                mtm=eqr+unreal
            if mtm<min_mtm: min_mtm=mtm; min_at=t
            peak=max(peak,mtm); dd=(peak-mtm)/peak
            if dd>mdd: mdd=dd
            if mtm<=MAINT and liq is None: liq=t
            if mtm<=MAINT: liq_n+=1
    def D(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    print(f'\n=== 증거금 20%(5x) 마크투마켓(1h 장중최악) ===')
    print(f'최종 실현자본: ${eqr:,.0f}  (스킵 {skipped:,} = 증거금캡 초과 진입)')
    print(f'MTM 최저자본: ${min_mtm:,.0f} @ {D(min_at)}  (초기 $10k 대비 {100*min_mtm/INIT-100:+.0f}%)')
    print(f'MTM 최대낙폭: {100*mdd:.1f}%')
    if liq is not None:
        print(f'🔴 청산 발생: {D(liq)} (자본 MTM ≤ ${MAINT:,.0f})  · 청산수준 그리드 {liq_n}회')
    else:
        print(f'🟢 청산 없음 (MTM 자본이 0 아래로 안 감)')

if __name__=='__main__':
    main()
