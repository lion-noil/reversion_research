# -*- coding: utf-8 -*-
"""BTCUSDT 멀티TF AND 진입 — 5분봉 베이스 게임 + 1분봉 z 합류 필터.
진입: 5분봉 마감 시 z5가 ±K5 돌파 AND 같은 시각 1분봉 z1이 ±K1f 돌파(같은 방향).
K1f=0 → 필터 없음(베이스라인). TP/SL·쿨다운·보유는 5분봉 게임(v2 모델, 추매없음)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011
SYM='BTCUSDT'; MH=14*DAY; MAXC=15; NMIN=int(os.getenv('NMIN','50'))

def zseries(m,win):
    C=np.array([x['close'] for x in m]); n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan), ma, sd

def main():
    t0=time.time()
    m1=_load_cache(SYM,'1'); m5=_load_cache(SYM,'5')
    ts1=np.array([x['start'] for x in m1],np.int64)
    ts5=np.array([x['start'] for x in m5],np.int64)
    C5=np.array([x['close'] for x in m5]); H5=np.array([x['high'] for x in m5]); L5=np.array([x['low'] for x in m5])
    n5=len(C5)
    months=(ts5[-1]-ts5[0])/(30.44*DAY)
    # 1m z (필터용) — 5분봉 마감시각(=시작+4분 1분바)에 샘플
    idx=np.searchsorted(ts1, ts5+4*60000)
    idx=np.clip(idx,0,len(ts1)-1)
    Z1={}
    for w1 in [120,360,1440]:
        z1,_,_=zseries(m1,w1)
        Z1[w1]=z1[idx]   # 5분 grid에 정렬
    # 5m 베이스
    Z5={}
    for w5 in [2016,4032]:
        Z5[w5]=zseries(m5,w5)
    print(f'prep {time.time()-t0:.0f}s | {months:.1f}개월',flush=True)

    DIRS=['trendlong','trend','long','short']
    K5S=[2.5,3.0,3.5,4.0]; BS=[round(-3+0.5*i,1) for i in range(13)]; CDS=[1,2,3,6]
    K1FS=[0,1.0,1.5,2.0,2.5]; W1S=[120,360,1440]
    res=[]
    for w5 in Z5:
        z5,ma5,sd5=Z5[w5]
        for DIR in DIRS:
            pos= DIR in('short','trendlong')
            base_cand=np.flatnonzero(z5>=0) # dummy
            for k5 in K5S:
                cand0=np.flatnonzero(z5>=k5) if pos else np.flatnonzero(z5<=-k5)
                cand0=cand0[cand0>=w5]
                for w1 in W1S:
                    z1s=Z1[w1]
                    for k1f in K1FS:
                      for fs in ([1] if k1f==0 else [1,-1]):   # fs=1 같은부호(합류), fs=-1 반대부호(풀백)
                        if k1f==0: cand=cand0
                        else:
                            same=(z1s[cand0]>=k1f) if pos else (z1s[cand0]<=-k1f)
                            oppo=(z1s[cand0]<=-k1f) if pos else (z1s[cand0]>=k1f)
                            cand=cand0[same if fs==1 else oppo]
                        if len(cand)<NMIN: continue
                        for b in BS:
                            for cdh in CDS:
                                cd=int(cdh*H); rets=[]; iv=[]; last=None
                                for ci in cand:
                                    if ci+1>=n5: break
                                    if last is not None and (ts5[ci]-last)<cd: continue
                                    price=C5[ci]; hi=int(np.searchsorted(ts5,ts5[ci]+MH,'right'))
                                    if hi<=ci+1: continue
                                    if DIR=='trendlong':
                                        tp=ma5[ci]+b*sd5[ci]
                                        if tp>=price: continue
                                        pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
                                        SL=L5[ci+1:hi]<=sx; TP=H5[ci+1:hi]>=tx; hit=SL|TP
                                        if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tx/price-1-FEE)
                                        else: ex=hi-1; r=C5[ex]/price-1-FEE
                                    elif DIR=='trend':
                                        tp=ma5[ci]-b*sd5[ci]
                                        if tp<=price: continue
                                        pct=tp/price-1; tx=price*(1-pct); sx=price*(1+pct)
                                        SL=H5[ci+1:hi]>=sx; TP=L5[ci+1:hi]<=tx; hit=SL|TP
                                        if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tx)/price-FEE)
                                        else: ex=hi-1; r=(price-C5[ex])/price-FEE
                                    elif DIR=='long':
                                        tp=ma5[ci]-b*sd5[ci]
                                        if tp<=price: continue
                                        pct=tp/price-1; sx=price*(1-pct)
                                        SL=L5[ci+1:hi]<=sx; TP=H5[ci+1:hi]>=tp; hit=SL|TP
                                        if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-FEE) if SL[j] else (tp/price-1-FEE)
                                        else: ex=hi-1; r=C5[ex]/price-1-FEE
                                    else:
                                        tp=ma5[ci]+b*sd5[ci]
                                        if tp>=price: continue
                                        pct=1-tp/price; sx=price*(1+pct)
                                        SL=H5[ci+1:hi]>=sx; TP=L5[ci+1:hi]<=tp; hit=SL|TP
                                        if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tp)/price-FEE)
                                        else: ex=hi-1; r=(price-C5[ex])/price-FEE
                                    rets.append(r); iv.append((int(ts5[ci]),int(ts5[ex]))); last=ts5[ci]
                                if len(rets)<NMIN: continue
                                ev=[]
                                for en,ex_ in iv: ev.append((en,1)); ev.append((ex_,-1))
                                ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
                                for _,dd in ev: cur+=dd; mx=max(mx,cur)
                                if mx>MAXC: continue
                                a=np.array(rets)
                                pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99.0
                                res.append(dict(dir=DIR,w5=w5,k5=k5,w1=w1,k1f=k1f,fs=fs,b=b,cd=cdh,
                                    n=len(a),pm=round(len(a)/months,1),mc=mx,
                                    wr=round(100*float((a>0).mean())),exp=round(float(a.mean())*100,3),
                                    pf=round(float(pf),2)))
            print(f'{DIR} w5={w5} done ({time.time()-t0:.0f}s)',flush=True)
    json.dump(res,open(os.path.join(os.path.dirname(__file__),'btc_mtf_and.json'),'w'))
    # 방향별: 베이스라인 최적 vs 필터 최적
    print(f'\n=== 방향별 최적: 필터없음(K1f=0) vs AND필터 ===')
    for DIR in DIRS:
        rows=[r for r in res if r['dir']==DIR]
        b0=max([r for r in rows if r['k1f']==0],key=lambda r:(r['exp'],r['pf']),default=None)
        b1=max([r for r in rows if r['k1f']>0 and r['fs']==1],key=lambda r:(r['exp'],r['pf']),default=None)
        b2=max([r for r in rows if r['k1f']>0 and r['fs']==-1],key=lambda r:(r['exp'],r['pf']),default=None)
        for tag,r in [('베이스   ',b0),('AND합류 ',b1),('AND풀백 ',b2)]:
            if r:
                print(f"{DIR:10}{tag}: w5={r['w5']//288}d K5={r['k5']} | w1={r['w1']//60}h K1f={r['k1f']} | B{r['b']}/{r['cd']}h n{r['n']}({r['pm']}/월) mc{r['mc']} 기대{r['exp']:+.2f}% PF{r['pf']} 승{r['wr']}%")

if __name__=='__main__':
    main()
