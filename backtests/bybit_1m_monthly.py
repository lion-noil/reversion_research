# -*- coding: utf-8 -*-
"""1분봉 Bybit 크립토 4종 · 최적조합 포트폴리오 · 진입50%/최대10배 · 월별 수익률.
목표 검증: '한 달 2배' 가능한지. 청산=노출대비 자산<20%. 3년(2023-06~)."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_s1_quarters_mt5 import prep, WIN
FEE=0.0011; H=3600*1000; DAY=86400*1000
MH=int(os.getenv('MH_D','14'))*DAY
INIT=10000.0
LOT=float(os.getenv('LOT','0.5')); MAXLEV=float(os.getenv('MAXLEV','10')); MAXLOTS=int(os.getenv('MAXLOTS','999'))
LIQ_MG=float(os.getenv('LIQ_MG','0.20'))
SIGN={'long':1.0,'short':-1.0,'trend':-1.0,'trendlong':1.0}

# (sym, DIR, k1, b, cd_h) — 기존 스윕 최적(양의 엣지)만
COMBOS=[
 ('BTCUSDT','trendlong',3.2,-2.0,3.0),('ETHUSDT','trendlong',2.35,1.2,2.5),
 ('SOLUSDT','trendlong',3.4,-2.0,3.0),('XRPUSDT','trendlong',2.55,-0.4,3.0),
 ('ETHUSDT','trend',3.45,-1.8,3.0),('SOLUSDT','trend',3.4,-2.0,1.5),
 ('BTCUSDT','long',3.3,-2.0,3.0),('ETHUSDT','long',3.15,-1.2,2.0),
 ('SOLUSDT','long',3.3,1.8,3.0),('XRPUSDT','long',3.5,-0.4,2.25),
 ('ETHUSDT','short',3.3,-1.2,3.0),('XRPUSDT','short',3.45,-2.0,1.75),
]

def gen(d,DIR,k1,b,cd_ms):
    C,Hh,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cand=np.flatnonzero(z>=k1) if DIR in('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=WIN]; out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+MH,'right'))
        if hi<=ci+1: continue
        if DIR=='short':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sl=price*(1+pct)
            SLh=Hh[ci+1:hi]>=sl; TPh=L[ci+1:hi]<=tp; hit=SLh|TPh
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sl)/price-FEE) if SLh[j] else ((price-tp)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        elif DIR=='trend':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; tpS=price*(1-pct); slS=price*(1+pct)
            SLh=Hh[ci+1:hi]>=slS; TPh=L[ci+1:hi]<=tpS; hit=SLh|TPh
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-slS)/price-FEE) if SLh[j] else ((price-tpS)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        elif DIR=='trendlong':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SLh=L[ci+1:hi]<=sx; TPh=Hh[ci+1:hi]>=tx; hit=SLh|TPh
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SLh[j] else (tx/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        else:  # long (역추세 롱)
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sx=price*(1-pct)
            SLh=L[ci+1:hi]<=sx; TPh=Hh[ci+1:hi]>=tp; hit=SLh|TPh
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SLh[j] else (tp/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        out.append((int(ts[ci]),int(ts[ex]),float(r),float(price))); last=ts[ci]
    return out

def mon(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}-{t.month:02d}'

def main():
    syms=sorted(set(c[0] for c in COMBOS))
    D={}; px={}
    for s in syms:
        d=prep(s); D[s]=d; px[s]=(d['ts'],d['C'])
        print(f'{s} prep {d["n"]:,}봉',flush=True)
    trades=[]
    for sym,DIR,k1,b,cdh in COMBOS:
        tr=gen(D[sym],DIR,k1,b,int(cdh*H)); sg=SIGN[DIR]
        for en,ex,r,ep in tr: trades.append((en,ex,r,ep,sg,sym))
        print(f'  {sym} {DIR}: {len(tr)}건',flush=True)
    print(f'총 {len(trades):,}건',flush=True)
    ev=[]
    for i,t in enumerate(trades): ev.append((t[0],0,i)); ev.append((t[1],2,i))
    ev.sort(key=lambda x:(x[0],x[1]))
    def price_at(s,t):
        ts_,C=px[s]; i=int(np.searchsorted(ts_,t,'right'))-1
        return C[i] if i>=0 else None
    eq=INIT; openpos={}; tot=0.0; lots=0; liq=None; peak=INIT; mdd=0.0; minmg=9.9; maxlev=0.0
    meq={}; skipped=0
    for t,typ,i in ev:
        if liq is not None: break
        if typ==0:
            notional=LOT*eq
            if lots>=MAXLOTS or tot+notional>MAXLEV*eq: skipped+=1; openpos[i]=None; continue
            en,ex,r,ep,sg,s=trades[i]; openpos[i]=(notional,ep,sg,s); tot+=notional; lots+=1
        else:
            v=openpos.get(i,'x')
            if v=='x': continue
            if v is None: del openpos[i]; continue
            notional,ep,sg,s=v; eq+=notional*r; tot-=notional; lots-=1; del openpos[i]
            meq[mon(trades[i][0])]=eq
        # MTM 평가(이벤트 시점)
        un=0.0
        for v in openpos.values():
            if not v: continue
            notional,ep,sg,s=v; p=price_at(s,t)
            if p is not None: un+=notional*sg*(p/ep-1.0)
        mtm=eq+un; peak=max(peak,mtm); dd=(peak-mtm)/peak; mdd=max(mdd,dd)
        if tot>0:
            maxlev=max(maxlev,tot/max(mtm,1e-9)); mg=max(mtm,0.0)/tot; minmg=min(minmg,mg)
            if mg<LIQ_MG and liq is None: liq=(t,mtm)
    # 월별 정리
    months=sorted(meq)
    prev=INIT; rows=[]
    for m in months:
        e=meq[m]; ret=e/prev-1; rows.append((m,e,ret*100)); prev=e
    print(f'\n=== 월별 (진입{LOT*100:.0f}% · 최대{MAXLEV:.0f}배 · 청산 노출대비{LIQ_MG*100:.0f}%) ===')
    print(f'{"월":>8}{"기말자산$":>14}{"월수익%":>10}')
    for m,e,rt in rows: print(f'{m:>8}{e:>14,.0f}{rt:>+9.1f}%')
    rr=np.array([x[2] for x in rows])
    print(f'\n최종 ${eq:,.0f} ({eq/INIT:.1f}배) | 스킵 {skipped}')
    print(f'월수익 평균 {rr.mean():+.1f}% · 중앙 {np.median(rr):+.1f}% · 최고 {rr.max():+.1f}% · 최저 {rr.min():+.1f}%')
    print(f'2배(+100%)이상 달: {int((rr>=100).sum())}/{len(rr)}개월')
    print(f'MTM최대낙폭 {mdd*100:.1f}% · 최대레버 {maxlev:.1f}x · 최저증거금율 {minmg*100:.1f}%')
    print('🔴 청산 '+datetime.fromtimestamp(liq[0]/1000,tz=timezone.utc).strftime('%Y-%m-%d')+f' (MTM ${liq[1]:,.0f})' if liq else '🟢 청산 없음')

if __name__=='__main__':
    main()
