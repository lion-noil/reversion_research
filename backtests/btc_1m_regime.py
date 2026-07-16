# -*- coding: utf-8 -*-
"""BTCUSDT — 1분봉 거래 게임 + 5분봉 레짐(게이트+조기청산).
게임(1분): z1=(C-MA_w1)/σ1, w1≤1일. 진입 z1이 ±K1 돌파, TP/SL=±pct(B, w1앵커) 손절우선, 쿨다운, 보유≤2일.
레짐(5분): z5(w5≤7일).
  게이트: 추세류=z5 같은방향 ≥K5f / 역추세류=z5 같은방향(괴리 확인). K5f=0이면 게이트 없음.
  레짐청산 X: 추세롱=z5≤X면 청산 / 추세숏=z5≥-X / 역추롱=z5≥X2(괴리해소) / 역추숏=z5≤-X2. None=끔.
청산 우선순위: SL > TP > 레짐청산(종가) > 만기(종가)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011
MH=2*DAY; MAXC=15; NMIN=100

def zma(closes,win):
    C=closes; n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    z=np.where(sd>0,(C-ma)/sd,np.nan)
    return z,ma,sd

def main():
    t0=time.time()
    m1=_load_cache('BTCUSDT','1'); m5=_load_cache('BTCUSDT','5')
    C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
    ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
    C5=np.array([x['close'] for x in m5]); ts5=np.array([x['start'] for x in m5],np.int64)
    months=(ts1[-1]-ts1[0])/(30.44*DAY)
    W1S=[120,360,720,1440]; W5S=[288,864,2016]   # 1분: 2h/6h/12h/24h · 5분: 1d/3d/7d
    Z1={w:zma(C1,w) for w in W1S}
    Z5={}
    for w in W5S:
        z5,_,_=zma(C5,w)
        idx=np.searchsorted(ts5+4*60000, ts1, 'right')-1
        idx=np.clip(idx,0,len(z5)-1)
        v=z5[idx]; v[idx<0]=np.nan
        Z5[w]=v   # 1분 grid에 ffill 정렬
    print(f'prep {time.time()-t0:.0f}s | {months:.1f}개월',flush=True)

    K1S=[2.5,3.0,3.5,4.0,4.5]; BS=[-3.0,-2.5,-2.0,-1.5,-1.0,0.0,1.0]; CDS=[1,3]
    K5FS=[0,1.5,2.0,2.5]; XS=[None,0.0]
    DIRS=['trendlong','trend','long','short']
    res=[]
    for DIR in DIRS:
        pos= DIR in('short','trendlong')          # z1 진입 부호
        for w1 in W1S:
            z1,ma1,sd1=Z1[w1]
            for k1 in K1S:
                cand_base=np.flatnonzero(z1>=k1) if pos else np.flatnonzero(z1<=-k1)
                cand_base=cand_base[cand_base>=max(W1S)]
                for w5 in W5S:
                    z5v=Z5[w5]
                    # 레짐 부호: 추세류=진입부호와 동일 / 역추세류=z1과 동일(괴리 같은방향)
                    reg_pos = DIR in('trendlong','short')   # z5 + 쪽이면 게이트 통과인 방향들: trendlong(상승레짐), short(과열괴리)
                    for k5f in K5FS:
                        if k5f==0: cand=cand_base
                        else:
                            g=(z5v[cand_base]>=k5f) if reg_pos else (z5v[cand_base]<=-k5f)
                            cand=cand_base[g]
                        if len(cand)<NMIN: continue
                        for X in XS:
                            # 레짐청산 조건 배열 (1분 grid bool)
                            if X is None: regexit=None
                            else:
                                if DIR=='trendlong': regexit=z5v<=X
                                elif DIR=='trend':   regexit=z5v>=-X
                                elif DIR=='long':    regexit=z5v>=X
                                else:                regexit=z5v<=-X
                            for b in BS:
                                for cdh in CDS:
                                    cd=int(cdh*H); rets=[]; iv=[]; last=None
                                    for ci in cand:
                                        if ci+1>=n1: break
                                        if last is not None and (ts1[ci]-last)<cd: continue
                                        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+MH,'right'))
                                        if hi<=ci+1: continue
                                        if DIR=='trendlong':
                                            tp=ma1[ci]+b*sd1[ci]
                                            if tp>=price: continue
                                            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
                                            SL=L1[ci+1:hi]<=sx; TP=H1[ci+1:hi]>=tx
                                        elif DIR=='trend':
                                            tp=ma1[ci]-b*sd1[ci]
                                            if tp<=price: continue
                                            pct=tp/price-1; tx=price*(1-pct); sx=price*(1+pct)
                                            SL=H1[ci+1:hi]>=sx; TP=L1[ci+1:hi]<=tx
                                        elif DIR=='long':
                                            tp=ma1[ci]-b*sd1[ci]
                                            if tp<=price: continue
                                            pct=tp/price-1; tx=tp; sx=price*(1-pct)
                                            SL=L1[ci+1:hi]<=sx; TP=H1[ci+1:hi]>=tx
                                        else:
                                            tp=ma1[ci]+b*sd1[ci]
                                            if tp>=price: continue
                                            pct=1-tp/price; tx=tp; sx=price*(1+pct)
                                            SL=H1[ci+1:hi]>=sx; TP=L1[ci+1:hi]<=tx
                                        REG=regexit[ci+1:hi] if regexit is not None else None
                                        hit=SL|TP if REG is None else (SL|TP|REG)
                                        if hit.any():
                                            j=int(np.argmax(hit)); ex=ci+1+j
                                            if SL[j]: xp=sx; sgn=-1
                                            elif TP[j]: xp=tx; sgn=1
                                            else: xp=C1[ex]; sgn=0
                                            r=(xp/price-1-FEE) if DIR in('trendlong','long') else ((price-xp)/price-FEE)
                                        else:
                                            ex=hi-1; xp=C1[ex]
                                            r=(xp/price-1-FEE) if DIR in('trendlong','long') else ((price-xp)/price-FEE)
                                        rets.append(r); iv.append((int(ts1[ci]),int(ts1[ex]))); last=ts1[ci]
                                    if len(rets)<NMIN: continue
                                    ev=[]
                                    for en,ex_ in iv: ev.append((en,1)); ev.append((ex_,-1))
                                    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
                                    for _,dd in ev: cur+=dd; mx=max(mx,cur)
                                    if mx>MAXC: continue
                                    a=np.array(rets)
                                    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99.0
                                    res.append(dict(dir=DIR,w1=w1,k1=k1,w5=w5,k5f=k5f,x=('N' if X is None else X),
                                        b=b,cd=cdh,n=len(a),pm=round(len(a)/months,1),mc=mx,
                                        wr=round(100*float((a>0).mean())),exp=round(float(a.mean())*100,3),
                                        pf=round(float(pf),2)))
            print(f'{DIR} w1={w1} done ({time.time()-t0:.0f}s)',flush=True)
    json.dump(res,open(os.path.join(os.path.dirname(__file__),'btc_1m_regime.json'),'w'))
    print(f'\n총 {len(res)}조합 | {time.time()-t0:.0f}s')
    print('=== 방향별: 게이트없음 vs 게이트만 vs 게이트+레짐청산 ===')
    for DIR in DIRS:
        rows=[r for r in res if r['dir']==DIR]
        c0=max([r for r in rows if r['k5f']==0 and r['x']=='N'],key=lambda r:(r['exp'],r['pf']),default=None)
        c1=max([r for r in rows if r['k5f']>0 and r['x']=='N'],key=lambda r:(r['exp'],r['pf']),default=None)
        c2=max([r for r in rows if r['k5f']>0 and r['x']!='N'],key=lambda r:(r['exp'],r['pf']),default=None)
        for tag,r in [('1분단독   ',c0),('+게이트   ',c1),('+게이트+청산',c2)]:
            if r:
                print(f"{DIR:10}{tag}: w1={r['w1']//60}h K1={r['k1']} | w5={r['w5']//288}d K5f={r['k5f']} X={r['x']} | B{r['b']}/{r['cd']}h n{r['n']}({r['pm']}/월) mc{r['mc']} 기대{r['exp']:+.3f}% PF{r['pf']} 승{r['wr']}%")

if __name__=='__main__':
    main()
