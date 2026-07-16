# -*- coding: utf-8 -*-
"""S5 = 추세 진입(S3와 동일 z≥+K1/z≤-K1) + 샹들리에 트레일링 청산.
TP 없음. 초기스톱=진입가∓m×ATR, 이후 롱: max(스톱, HH−m×ATR) / 숏: min(스톱, LL+m×ATR).
최대보유 90일(3개월, 종가청산). K1·m·쿨다운 스윕. S3(고정TP) 대비 비교 출력.
env: DAILY_ITV(D10|D), S5_SYMS, GPY_MIN(10), MAXC_CAP(15), FEE, OUT"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from daily_sweep import prep, DAY, WIN

FEE=float(os.getenv('FEE','0.0011'))
MAXHOLD=int(os.getenv('S5_MAXHOLD_D','90'))*DAY          # 3개월
_k0=float(os.getenv('K_MIN','1.5')); _k1=float(os.getenv('K_MAX','3.5')); _ks=float(os.getenv('K_STEP','0.1'))
K1S=[round(_k0+_ks*i,2) for i in range(int(round((_k1-_k0)/_ks))+1)]
MS=[float(x) for x in os.getenv('MS','1.5,2.0,2.5,3.0,3.5,4.0,5.0').split(',')]   # 샹들리에 배수 m
CDD=[1,2,3,5,7,10]                                       # 쿨다운(일)
NMIN=int(os.getenv('NMIN','30')); GPY_MIN=float(os.getenv('GPY_MIN','10'))
MAXC_CAP=int(os.getenv('MAXC_CAP','15'))
SYMS=[s.strip().upper() for s in os.getenv('S5_SYMS','').split(',') if s.strip()]
ATR_N=14

def atr14(d):
    C,H,L,n=d['C'],d['H'],d['L'],d['n']
    tr=np.empty(n); tr[0]=H[0]-L[0]
    tr[1:]=np.maximum(H[1:]-L[1:],np.maximum(np.abs(H[1:]-C[:-1]),np.abs(L[1:]-C[:-1])))
    a=np.full(n,np.nan); c=np.cumsum(tr)
    a[ATR_N-1:]=(c[ATR_N-1:]-np.concatenate([[0.],c[:-ATR_N]]))/ATR_N
    return a

def run_s5(d,atr,side,k1,m,cd_ms):
    """side=+1 추세롱(z>=+K1) / -1 추세숏(z<=-K1). returns (rets, iv, holds)"""
    C,H,L,ts,z,n=d['C'],d['H'],d['L'],d['ts'],d['z'],d['n']
    cand=np.flatnonzero(z>=k1) if side>0 else np.flatnonzero(z<=-k1)
    cand=cand[cand>=max(WIN,ATR_N)]
    rets=[]; iv=[]; holds=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; a0=atr[ci]
        if not np.isfinite(a0) or a0<=0: continue
        hi=int(np.searchsorted(ts,ts[ci]+MAXHOLD,'right'))
        if hi<=ci+1: continue
        if side>0:
            stop=price-m*a0; hh=H[ci]; ex=None; xp=None
            for k in range(ci+1,hi):
                if L[k]<=stop: ex=k; xp=stop; break
                if H[k]>hh: hh=H[k]
                s2=hh-m*atr[k]
                if s2>stop: stop=s2
            if ex is None: ex=hi-1; xp=C[ex]
            r=xp/price-1-FEE
        else:
            stop=price+m*a0; ll=L[ci]; ex=None; xp=None
            for k in range(ci+1,hi):
                if H[k]>=stop: ex=k; xp=stop; break
                if L[k]<ll: ll=L[k]
                s2=ll+m*atr[k]
                if s2<stop: stop=s2
            if ex is None: ex=hi-1; xp=C[ex]
            r=(price-xp)/price-FEE
        rets.append(r); iv.append((int(ts[ci]),int(ts[ex]))); holds.append((ts[ex]-ts[ci])/DAY)
        last=ts[ci]
    return rets,iv,holds

def maxc_of(iv):
    ev=[]
    for en,ex in iv: ev.append((en,1)); ev.append((ex,-1))
    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
    for _,dd in ev: cur+=dd; mx=max(mx,cur)
    return mx

def main():
    out={}
    for s in SYMS:
        d=prep(s)
        if d is None: print(f'{s}: 데이터없음',flush=True); continue
        atr=atr14(d)
        yrs=(int(d['ts'][-1])-int(d['ts'][WIN]))/(365.25*DAY)
        out[s]={}
        for side,nm in [(1,'S5_long'),(-1,'S5_short')]:
            top=None
            for k1 in K1S:
                for m in MS:
                    for cdd in CDD:
                        rets,iv,holds=run_s5(d,atr,side,k1,m,cdd*DAY)
                        if len(rets)<NMIN: continue
                        mx=maxc_of(iv)
                        if mx>MAXC_CAP: continue
                        gpy=len(rets)/yrs
                        if gpy<GPY_MIN: continue
                        a=np.array(rets); e=float(a.mean())
                        pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99.0
                        key=(round(e,5),pf)
                        if top is None or key>top[0]:
                            top=(key,dict(k1=k1,m=m,cd=cdd,n=len(a),gpy=round(gpy,1),
                                wr=round(100*float((a>0).mean())),exp=round(e*100,2),
                                pf=round(float(pf),2),maxc=mx,hold=round(float(np.mean(holds)),1)))
            out[s][nm]=top[1] if top else None
            r=out[s][nm]
            print(f"{s} {nm}: "+(f"K{r['k1']}/m{r['m']}/{r['cd']}d n{r['n']}({r['gpy']}/yr) 승{r['wr']}% 기대{r['exp']:+.2f}% PF{r['pf']} mc{r['maxc']} 평균보유{r['hold']}d" if r else '조합없음'),flush=True)
    fn=os.getenv('OUT','s5_result.json')
    json.dump(out,open(os.path.join(os.path.dirname(__file__),fn),'w'),ensure_ascii=False)
    print(f'→ {fn}',flush=True)

if __name__=='__main__':
    main()
