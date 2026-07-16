# -*- coding: utf-8 -*-
"""일봉 포트폴리오 시뮬 — $10k, 랏=자산5%, 최대10배(200랏), MTM고점대비-20%면 강제청산.
파라미터=daily_quarterly.json(maxc≤10 최적). (A)심볼별 4전략 전부 vs (B)심볼별 최적1전략 비교.
MTM=일봉 고/저 장중최악. 청산시 계좌 종료. 분기별 실현자산 + 청산여부."""
import sys, os, json
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from daily_sweep import prep, WIN, DAY, MAXHOLD, ITV
from reversion_calibrator import _load_cache

FEE=0.0011; INIT=10_000.0
LOT=float(os.getenv('LOT','0.05')); MAXLEV=float(os.getenv('MAXLEV','10')); MAXLOTS=int(os.getenv('MAXLOTS','200'))
LIQ_DD=float(os.getenv('LIQ_DD','0.20'))
SIGN={'S1_long':1.0,'S1_short':-1.0,'S2_long':1.0,'S2_short':-1.0}
MODE={'S1_long':'trendlong','S1_short':'trend','S2_long':'long','S2_short':'short'}
FXSET=set('EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD'.split())
FXLOT=float(os.getenv('FXLOT',str(LOT)))   # FX 진입비율(미지정시 LOT)
def lot_of(s): return FXLOT if s in FXSET else LOT

def gen(d,mode,k1,b,cd_ms):
    C,H,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cand=np.flatnonzero(z>=k1) if mode in('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=WIN]; out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+MAXHOLD,'right'))
        if hi<=ci+1: continue
        if mode=='trendlong':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=H[ci+1:hi]>=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tx/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        elif mode=='trend':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; tx=price*(1-pct); sx=price*(1+pct)
            SL=H[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tx)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        elif mode=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=H[ci+1:hi]>=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tp/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=H[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tp)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        out.append((int(ts[ci]),int(ts[ex]),float(r),float(price))); last=ts[ci]
    return out

def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
def mof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}-{t.month:02d}'
def D(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).strftime('%Y-%m-%d')

def build_trades(J, pick_best):
    pc={}; trades=[]
    for s,row in J.items():
        dirs=[(k,v) for k,v in row.items() if v]
        if not dirs: continue
        if pick_best:
            k,v=max(dirs,key=lambda kv: kv[1]['exp']); dirs=[(k,v)]
        d=pc.get(s) or prep(s); pc[s]=d
        for k,v in dirs:
            for en,ex,r,ep in gen(d,MODE[k],v['k1'],v['b'],v['cd']*DAY):
                trades.append((en,ex,r,ep,SIGN[k],s))
    return trades

def simulate(trades, px):
    eq=INIT; openpos={}; tot_notional=0.0; skipped=0
    grid=sorted({int(g) for s in px for g in px[s][0]})
    ev=[]
    for i,(en,ex,r,ep,sg,s) in enumerate(trades): ev.append((en,0,i)); ev.append((ex,2,i))
    for g in grid: ev.append((g,1,-1))
    ev.sort(key=lambda x:(x[0],x[1]))
    peak=INIT; mdd=0.0; liq_dd=None; liq_mg=None; qeq={}; meq={}; lots=0
    maxlev=0.0; minmg=9.9
    def worst(s,t,sg):
        ts_,hi,lo,cl=px[s]; i=int(np.searchsorted(ts_,t,'right'))-1
        if i<0: return None
        return lo[i] if sg>0 else hi[i]
    for t,typ,i in ev:
        if liq_dd is not None and liq_mg is not None: pass
        if typ==0:
            notional=lot_of(trades[i][5])*eq
            if lots>=MAXLOTS or tot_notional+notional>MAXLEV*eq: skipped+=1; openpos[i]=None; continue
            en,ex,r,ep,sg,s=trades[i]; openpos[i]=(notional,ep,sg,s); tot_notional+=notional; lots+=1
        elif typ==2:
            v=openpos.get(i)
            if not v: continue
            notional,ep,sg,s=v; eq+=notional*r; tot_notional-=notional; lots-=1; del openpos[i]
            qeq[qof(trades[i][0])]=eq; meq[mof(trades[i][0])]=eq
        else:
            if not openpos: mtm=eq
            else:
                un=0.0
                for v in openpos.values():
                    if not v: continue
                    notional,ep,sg,s=v; wp=worst(s,t,sg)
                    if wp is not None: un+=notional*sg*(wp/ep-1.0)
                mtm=eq+un
            peak=max(peak,mtm); dd=(peak-mtm)/peak
            if dd>mdd: mdd=dd
            if dd>=LIQ_DD and liq_dd is None: liq_dd=(t,mtm)
            if tot_notional>0:
                lev=tot_notional/max(mtm,1e-9); maxlev=max(maxlev,lev)
                mg=max(mtm,0.0)/tot_notional; minmg=min(minmg,mg)
                if mg<0.20 and liq_mg is None: liq_mg=(t,mtm)
    return dict(final=eq, mdd=mdd, liq_dd=liq_dd, liq_mg=liq_mg, skipped=skipped, qeq=qeq, meq=meq, ntr=len(trades), maxlev=maxlev, minmg=minmg)

def main():
    J=json.load(open(os.path.join(os.path.dirname(__file__),os.getenv('IN','daily_quarterly.json')),encoding='utf-8'))
    flt=[s.strip().upper() for s in os.getenv('PORT_SYMS','').split(',') if s.strip()]
    if flt: J={s:J[s] for s in flt if s in J}; print(f'[필터] {len(J)}심볼: {",".join(J.keys())}')
    syms=set(J.keys())
    px={s:( (lambda c:(np.array([x['start'] for x in c],np.int64),np.array([x['high'] for x in c]),np.array([x['low'] for x in c]),np.array([x['close'] for x in c])))(_load_cache(s,ITV)) ) for s in syms}
    for name,best in [('A) 4전략 전부',False),('B) 최적 1전략만',True)]:
        tr=build_trades(J,best); R=simulate(tr,px)
        print(f'\n===== {name} =====  (거래 {R["ntr"]:,}, 스킵 {R["skipped"]:,})')
        if os.getenv('MONTHLY'):
            ms=sorted(R['meq']); prev=INIT; rr=[]
            for m in ms: rr.append(R['meq'][m]/prev-1); prev=R['meq'][m]
            rr=np.array(rr)*100 if rr else np.array([0.0])
            print(f'월수익 평균{rr.mean():+.1f}% 중앙{np.median(rr):+.1f}% 최고{rr.max():+.1f}% 최저{rr.min():+.1f}% | 2배+달 {int((rr>=100).sum())}/{len(rr)}')
        qs=sorted(R['qeq'])
        print(f'{"분기":>7}{"기말자산$":>14}')
        for q in qs: print(f'{q:>7}{R["qeq"][q]:>14,.0f}')
        print(f'최종 ${R["final"]:,.0f} ({R["final"]/INIT:.1f}배) | MTM최대낙폭 {100*R["mdd"]:.1f}% | 최대레버 {R["maxlev"]:.1f}x | 최저증거금율 {100*R["minmg"]:.1f}%')
        ld,lm=R['liq_dd'],R['liq_mg']
        print('  [해석1·고점대비 -20%]  '+(f'🔴 청산 {D(ld[0])} (MTM ${ld[1]:,.0f})' if ld else f'🟢 청산없음 (최대낙폭 {100*R["mdd"]:.1f}%)'))
        print('  [해석2·유지증거금 20%] '+(f'🔴 청산 {D(lm[0])} (MTM ${lm[1]:,.0f})' if lm else f'🟢 청산없음 (최저증거금율 {100*R["minmg"]:.1f}% > 20%)'))

if __name__=='__main__':
    main()
