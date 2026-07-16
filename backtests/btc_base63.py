# -*- coding: utf-8 -*-
"""6.3년 전체 · 무게이트 베이스 스윕 (원칙: 베이스 자립 → 레짐은 승률 보조).
1단계: 창{2,6,12,22,24h}×보유{2,7,14d}×K{3~6}×B{-3~3}×cd{1,3h}×4방향, SL={대칭밴드, 없음} 둘 다.
2단계: 방향별 상위 25셀에 연도분해 → '연도균형'(7년 중 마이너스 연도 ≤2 & 최악연도 > -30%p) 필터 → 생존자 출력."""
import sys, os, time, json
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=float(os.getenv('FEE','0.0011'))
m1=_load_cache(os.getenv('SYM','BTCUSDT'),'1')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
def zma(win):
    cs=np.concatenate([[0.],np.cumsum(C1)]); cs2=np.concatenate([[0.],np.cumsum(C1*C1)])
    ma=np.full(n1,np.nan); sd=np.full(n1,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C1-ma)/sd,np.nan),ma,sd
def yof(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).year

def sim(z,ma,sd,win,DIR,k1,b,mh_d,cdh,use_sl):
    epos = DIR in('trendlong','short'); is_long = DIR in('trendlong','long')
    cand=np.flatnonzero(z>=k1) if epos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=win]
    mh=int(mh_d*DAY); cd=int(cdh*H); out=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<cd: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+mh,'right'))
        if hi<=ci+1: continue
        if DIR=='trendlong':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            pct=1-tpb/price; tp=price*(1+pct); sl=price*(1-pct)
            TP=H1[ci+1:hi]>=tp; SL=(L1[ci+1:hi]<=sl) if use_sl else np.zeros(hi-ci-1,bool)
        elif DIR=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sl=price*(1-pct)
            TP=H1[ci+1:hi]>=tp; SL=(L1[ci+1:hi]<=sl) if use_sl else np.zeros(hi-ci-1,bool)
        elif DIR=='trend':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            pct=tpb/price-1; tp=price*(1-pct); sl=price*(1+pct)
            TP=L1[ci+1:hi]<=tp; SL=(H1[ci+1:hi]>=sl) if use_sl else np.zeros(hi-ci-1,bool)
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sl=price*(1+pct)
            TP=L1[ci+1:hi]<=tp; SL=(H1[ci+1:hi]>=sl) if use_sl else np.zeros(hi-ci-1,bool)
        hit=SL|TP
        if hit.any():
            j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SL[j] else tp
        else: ex=hi-1; xp=C1[ex]
        r=(xp/price-1-FEE) if is_long else ((price-xp)/price-FEE)
        out.append((int(ts1[ci]),r)); last=ts1[ci]
    return out

t0=time.time(); rows=[]
WINS=[120,360,720,1320,1440]
for win in WINS:
    z,ma,sd=zma(win)
    for DIR in ['trendlong','trend','long','short']:
        for k1 in [round(3.0+0.25*i,2) for i in range(13)]:
            for b in [round(-3+0.5*i,1) for i in range(13)]:
                for mh in [2,7,14]:
                    for cdh in [1,3]:
                        for use_sl in [True,False]:
                            tr=sim(z,ma,sd,win,DIR,k1,b,mh,cdh,use_sl)
                            if len(tr)<60: continue
                            a=np.array([r for _,r in tr])
                            e=float(a.mean())
                            if e<=0.002: continue   # 베이스 자립 후보만 (기대 +0.2%↑)
                            pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
                            rows.append((e,pf,win,DIR,k1,b,mh,cdh,use_sl,tr))
    print(f'win={win//60}h done ({time.time()-t0:.0f}s) 후보 {len(rows)}',flush=True)
print(f'\n=== 2단계: 방향별 상위 후보 연도균형 필터 ===')
for DIR in ['trendlong','trend','long','short']:
    cand=[r for r in rows if r[3]==DIR]
    cand.sort(reverse=True,key=lambda r:(r[0],r[1]))
    surv=0
    for e,pf,win,_,k1,b,mh,cdh,use_sl,tr in cand[:40]:
        byy={}
        for t,r in tr: byy.setdefault(yof(t),[]).append(r)
        ys={y:sum(v)*100 for y,v in byy.items()}
        neg=[y for y,s in ys.items() if s<-10]
        if len(neg)>2 or (ys and min(ys.values())<-30): continue
        a=np.array([r for _,r in tr])
        yline=' '.join(f'{y%100:02d}:{ys[y]:+.0f}' for y in sorted(ys))
        print(f'{DIR:9} 창{win//60}h K{k1} B{b} h{mh}d cd{cdh}h SL{"유" if use_sl else "무"}: n{len(a)}({len(a)/months:.1f}/월) 기대{e*100:+.3f}% PF{pf:.2f} 승{100*(a>0).mean():.0f}% | {yline}')
        surv+=1
        if surv>=6: break
    if surv==0: print(f'{DIR:9}: 연도균형 통과 셀 없음')
