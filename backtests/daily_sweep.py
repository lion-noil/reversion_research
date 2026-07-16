# -*- coding: utf-8 -*-
"""일봉 기준 4방향 스윕 — MA/σ 창=90일 고정, K1·B·쿨다운(일) 촘촘.
S1추세(롱=z>=+K1, 숏=z<=-K1) / S2역추세(롱=z<=-K1, 숏=z>=+K1). TP=±pct 대칭, 손절우선, fee0.11%.
최대보유 30일, 진입기준 쿨다운(일), 동시보유 허용. 일봉이라 n 적음 → n>=NMIN만.
출력: 콘솔 4방향표 + results/daily90_{4dir}.json
"""
import sys, os, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache

FEE=float(os.getenv('FEE','0.0011')); DAY=86400*1000
WIN=int(os.getenv('WIN_D','90'))            # MA/σ 창(일)
MAXHOLD=int(os.getenv('MAXHOLD_D','30'))*DAY
NMIN=int(os.getenv('NMIN','8'))
ITV=os.getenv('DAILY_ITV','D')              # 'D'(3년) | 'D10'(10년)
GPY_MIN=float(os.getenv('GPY_MIN','0'))     # 연거래수 하한
GPY_MAX=float(os.getenv('GPY_MAX','1e9'))   # 연거래수 상한
_k0=float(os.getenv('K1_MIN','2.0')); _k1=float(os.getenv('K1_MAX','3.5')); _ks=float(os.getenv('K1_STEP','0.1'))
K1S=[round(_k0+_ks*i,2) for i in range(int(round((_k1-_k0)/_ks))+1)]   # 기본 2.0~3.5
_b0=float(os.getenv('B_MIN','-2')); _b1=float(os.getenv('B_MAX','2')); _bs=float(os.getenv('B_STEP','0.2'))
BS=[round(_b0+_bs*i,2) for i in range(int(round((_b1-_b0)/_bs))+1)]     # 기본 -2~2
CDD=[1,2,3,5,7,10]                            # 쿨다운(일)
_DEF=['US100','JP225','HK50','GER40','UK100','XAUUSD','XAGUSD','WTI','BTCUSD','ETHUSD',
      'BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD']
SYMS=[s.strip().upper() for s in os.getenv('DAILY_SYMS','').split(',') if s.strip()] or _DEF
MODES=[('S1','long','trendlong'),('S1','short','trend'),('S2','long','long'),('S2','short','short')]

def prep(sym):
    m=_load_cache(sym,ITV)
    if not m or len(m)<=WIN: return None
    C=np.array([x['close'] for x in m]); H=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64); n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)]); w=WIN
    ma=np.full(n,np.nan); sd=np.full(n,np.nan); s=cs[w:]-cs[:-w]; s2=cs2[w:]-cs2[:-w]; mm=s/w
    ma[w-1:]=mm; sd[w-1:]=np.sqrt(np.maximum(s2/w-mm*mm,0)); z=np.where(sd>0,(C-ma)/sd,np.nan)
    return dict(C=C,H=H,L=L,ts=ts,ma=ma,sd=sd,z=z,n=n)

def run(d,mode,k1,b,cd_ms):
    C,H,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cand=np.flatnonzero(z>=k1) if mode in('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=WIN]; rets=[]; iv=[]; last=None
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
        else:  # short
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=H[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-FEE) if SL[j] else ((price-tp)/price-FEE)
            else: ex=hi-1; r=(price-C[ex])/price-FEE
        rets.append(r); iv.append((int(ts[ci]),int(ts[ex]))); last=ts[ci]
    # maxc
    ev=[]
    for en,ex in iv: ev.append((en,1)); ev.append((ex,-1))
    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
    for _,dd in ev: cur+=dd; mx=max(mx,cur)
    return rets,mx

def stat(r):
    a=np.array(r); n=len(a)
    if n==0: return (0,0,0,0)
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99.0
    return (n,round(100*(a>0).mean()),round(float(a.mean())*100,3),round(float(pf),2))

def main():
    print(f'일봉 스윕[{ITV}] | MA창={WIN}일, 최대보유={MAXHOLD//DAY}일, fee0.11% | K1{K1S[0]}~{K1S[-1]} B-2~2 CD{CDD}일 | n>={NMIN} | {len(SYMS)}심볼')
    best={}  # sym -> {dir: row}
    for s in SYMS:
        d=prep(s)
        if d is None: print(f'{s}: 일봉 부족'); continue
        yrs=(int(d['ts'][-1])-int(d['ts'][WIN]))/(365.25*DAY)
        best[s]={}
        for strat,dn,mode in MODES:
            top=None
            for k1 in K1S:
                for b in BS:
                    for cdd in CDD:
                        rets,mx=run(d,mode,k1,b,cdd*DAY)
                        n,wr,exp,pf=stat(rets)
                        if n<NMIN: continue
                        gpy=n/yrs
                        if not (GPY_MIN<=gpy<=GPY_MAX): continue
                        key=(exp,pf)
                        if top is None or key>top[0]: top=(key,dict(k1=k1,b=b,cd=cdd,n=n,maxc=mx,wr=wr,exp=exp,pf=pf,gpy=round(gpy,1)))
            best[s][f'{strat}_{dn}']=top[1] if top else None
    # 출력: 심볼별 4방향 best + 최적
    print(f'\n{"심볼":9}{"S1추세롱":>20}{"S1추세숏":>20}{"S2역롱":>20}{"S2역숏":>20}  최적')
    def f(r): return (f"{r['k1']}/{r['b']}/{r['cd']}d/{r['exp']:+.2f}/PF{r['pf']}/{r.get('gpy','?')}yr" if r else '-')
    rank=[]
    for s in SYMS:
        if s not in best: continue
        bs=best[s]; cells=[bs.get('S1_long'),bs.get('S1_short'),bs.get('S2_long'),bs.get('S2_short')]
        names=['S1추세롱','S1추세숏','S2역롱','S2역숏']
        bestdir=max(range(4),key=lambda i: cells[i]['exp'] if cells[i] else -9)
        bd=cells[bestdir]
        mark=f"{names[bestdir]}({bd['exp']:+.2f}/PF{bd['pf']})" if bd else '-'
        print(f'{s:9}{f(cells[0]):>20}{f(cells[1]):>20}{f(cells[2]):>20}{f(cells[3]):>20}  {mark}')
        if bd: rank.append((bd['exp'],s,names[bestdir],bd))
    print(f'\n=== 일봉 최적방향 기대값순 ===')
    for exp,s,nm,bd in sorted(rank,reverse=True):
        print('  %-9s %-8s K%s/B%s/%sd  기대%+.2f%% PF%s n%d(%s/yr) 중복%d'%(s,nm,bd['k1'],bd['b'],bd['cd'],bd['exp'],bd['pf'],bd['n'],bd.get('gpy','?'),bd['maxc']))
    fn=f'results/daily{WIN}_{ITV}_best.json'
    json.dump(best,open(os.path.join(os.path.dirname(__file__),fn),'w'),ensure_ascii=False)
    print(f'\n→ {fn}')

if __name__=='__main__':
    main()
