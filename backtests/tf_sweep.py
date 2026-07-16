# -*- coding: utf-8 -*-
"""범용 타임프레임 스윕 — 5분/1시간봉 등 임의 봉에서 4방향(추세롱/숏·역추롱/숏) × 창 다중 스윕.
모델 v2: 봉마감 z=±K1 진입, TP/SL ±pct 대칭(손절우선), 진입쿨다운, 동시보유(독립·추매없음), 만기 종가청산.
env: TF(분,기본60) SYMS WINS(바 수, 콤마) K_MIN/K_MAX/K_STEP B_MIN/B_MAX/B_STEP CDH(시간, 콤마)
     MAXHOLD_D(14) NMIN(100) MAXC(15) FEE(0.0011) OUT"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000
TF=os.getenv('TF','60')
FEE=float(os.getenv('FEE','0.0011')); MH=int(float(os.getenv('MAXHOLD_D','14'))*DAY)
HOLD_MULT=float(os.getenv('HOLD_MULT','0'))   # >0이면 보유상한 = 창(분)×배수 (창비례)
NMIN=int(os.getenv('NMIN','100')); MAXC=int(os.getenv('MAXC','15'))
SYMS=[s.strip().upper() for s in os.getenv('SYMS','BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT').split(',') if s.strip()]
WINS=[int(x) for x in os.getenv('WINS','168,336,504').split(',')]
def _fr(a,b,s): return [round(a+s*i,2) for i in range(int(round((b-a)/s))+1)]
K1S=_fr(float(os.getenv('K_MIN','2.0')),float(os.getenv('K_MAX','5.0')),float(os.getenv('K_STEP','0.1')))
BS=_fr(float(os.getenv('B_MIN','-3')),float(os.getenv('B_MAX','3')),float(os.getenv('B_STEP','0.2')))
CDH=[float(x) for x in os.getenv('CDH','1,2,3,6,12,24').split(',')]
DIRS=['trendlong','trend','long','short']

def prep(sym,win):
    m=_load_cache(sym,TF)
    if not m or len(m)<=win: return None
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64); n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    z=np.where(sd>0,(C-ma)/sd,np.nan)
    return dict(C=C,H=Hh,L=L,ts=ts,ma=ma,sd=sd,z=z,n=n,win=win)

def run(d,DIR,k1,b,cd_ms,mh_ms=None):
    mh_ms=mh_ms or MH
    C,Hh,L,ts,ma,sd,z,n,win=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n'],d['win']
    cand=np.flatnonzero(z>=k1) if DIR in('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=win]; rets=[]; iv=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+mh_ms,'right'))
        if hi<=ci+1: continue
        if DIR=='trendlong':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tx/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        elif DIR=='trend':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; tx=price*(1-pct); sx=price*(1+pct)
            SL=Hh[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tx)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        elif DIR=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tp/price-1-FEE)
            else: ex=hi-1; r=C[ex]/price-1-FEE
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=Hh[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tp)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        rets.append(r); iv.append((int(ts[ci]),int(ts[ex]))); last=ts[ci]
    ev=[]
    for en,ex in iv: ev.append((en,1)); ev.append((ex,-1))
    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
    for _,dd in ev: cur+=dd; mx=max(mx,cur)
    return rets,mx

def main():
    t0=time.time()
    print(f'[cfg] TF={TF}m WINS={WINS} K{K1S[0]}~{K1S[-1]}/{len(K1S)} B{BS[0]}~{BS[-1]}/{len(BS)} CD{CDH}h hold{MH//DAY}d n>={NMIN} mc<={MAXC} SYMS={",".join(SYMS)}',flush=True)
    out={}
    for s in SYMS:
        out[s]={}
        for w in WINS:
            d=prep(s,w)
            if d is None: continue
            mh=int(w*int(TF)*60000*HOLD_MULT) if HOLD_MULT>0 else MH
            for DIR in DIRS:
                top=None
                for k1 in K1S:
                    for b in BS:
                        for cdh in CDH:
                            rets,mx=run(d,DIR,k1,b,int(cdh*H),mh)
                            if len(rets)<NMIN or mx>MAXC: continue
                            a=np.array(rets); e=float(a.mean())
                            pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99.0
                            key=(round(e,5),pf)
                            if top is None or key>top[0]:
                                top=(key,dict(k1=k1,b=b,cd=cdh,n=len(a),mc=mx,wr=round(100*float((a>0).mean())),
                                              exp=round(e*100,3),pf=round(float(pf),2)))
                out[s].setdefault(DIR,{})[w]=top[1] if top else None
            bs=[(v['exp'],DIR,w) for DIR in out[s] for w,v in out[s][DIR].items() if v and w==w]
            print(f'{s} W{w} done ({time.time()-t0:.0f}s)',flush=True)
    fn=os.getenv('OUT',f'tf{TF}_sweep.json')
    json.dump(out,open(os.path.join(os.path.dirname(__file__),fn),'w'),ensure_ascii=False)
    print(f'→ {fn} ({time.time()-t0:.0f}s)',flush=True)

if __name__=='__main__':
    main()
