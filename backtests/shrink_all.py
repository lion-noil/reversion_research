# -*- coding: utf-8 -*-
"""수축(1→1/3)+SL재진입 변형 — 추세전략 전체 적용: S3(일봉)·S1(1분봉) × 롱/숏 × MT5·Bybit.
베이스라인(고정밴드) vs 변형, MAR(총수익/최대DD) 비교. 손절우선, 만기 종가청산."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import daily_sweep as ds
from bt_s1_quarters_mt5 import prep as prep1m
FEE=0.0011; DAY=86400*1000; H=3600*1000

def sim2(d,win,side,k1,b,cd_ms,mh_ms,shrink,reenter):
    C,Hh,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    sig=(z>=k1) if side>0 else (z<=-k1)
    pos=[]; out=[]; last=None
    for i in range(win,n):
        if not pos and not sig[i]: continue
        t=ts[i]
        if pos:
            keep=[]
            for p in pos:
                if p['start']>=t: keep.append(p); continue
                el=(t-p['start'])/mh_ms
                if el>=1.0: out.append((p['e0'],int(t),(C[i]/p['ep']-1-FEE)*side if side>0 else (p['ep']-C[i])/p['ep']-FEE,p['ep'],side)); continue
                f=1.0-(1.0-shrink)*el
                if side>0:
                    tp=p['ep']*(1+p['pct']*f); sl=p['ep']*(1-p['pct']*f)
                    if L[i]<=sl:
                        out.append((p['e0'],int(t),sl/p['ep']-1-FEE,p['ep'],side))
                        if reenter and p['leg']==0: keep.append(dict(e0=int(t),ep=sl,pct=p['pct'],start=t,leg=1))
                    elif Hh[i]>=tp: out.append((p['e0'],int(t),tp/p['ep']-1-FEE,p['ep'],side))
                    else: keep.append(p)
                else:
                    tp=p['ep']*(1-p['pct']*f); sl=p['ep']*(1+p['pct']*f)
                    if Hh[i]>=sl:
                        out.append((p['e0'],int(t),(p['ep']-sl)/p['ep']-FEE,p['ep'],side))
                        if reenter and p['leg']==0: keep.append(dict(e0=int(t),ep=sl,pct=p['pct'],start=t,leg=1))
                    elif L[i]<=tp: out.append((p['e0'],int(t),(p['ep']-tp)/p['ep']-FEE,p['ep'],side))
                    else: keep.append(p)
            pos=keep
        if sig[i] and i+1<n and (last is None or ts[i]-last>=cd_ms):
            price=C[i]
            if side>0:
                tpb=ma[i]+b*sd[i]
                if tpb<price: pos.append(dict(e0=int(ts[i]),ep=price,pct=1-tpb/price,start=ts[i],leg=0)); last=ts[i]
            else:
                tpb=ma[i]-b*sd[i]
                if tpb>price: pos.append(dict(e0=int(ts[i]),ep=price,pct=tpb/price-1,start=ts[i],leg=0)); last=ts[i]
    for p in pos:
        r=(C[n-1]/p['ep']-1-FEE) if side>0 else ((p['ep']-C[n-1])/p['ep']-FEE)
        out.append((p['e0'],int(ts[n-1]),r,p['ep'],side))
    return out

def dd_fast(ts,C,trades):
    trs=sorted(trades); m=len(trs); n=len(ts)
    if m==0: return 0.0,0.0
    realized=0.0; openl=[]; ptr=0; peak=-1e18; mdd=0.0
    i=int(np.searchsorted(ts,trs[0][0]))
    while i<n:
        t=ts[i]
        while ptr<m and trs[ptr][0]<=t: openl.append(trs[ptr]); ptr+=1
        keep=[]
        for tr in openl:
            if tr[1]<=t: realized+=tr[2]
            else: keep.append(tr)
        openl=keep
        un=sum(tr[4]*(C[i]/tr[3]-1.0) for tr in openl)
        eq=(realized+un)*100
        if eq>peak: peak=eq
        elif peak-eq>mdd: mdd=peak-eq
        if not openl:
            if ptr>=m: break
            i=int(np.searchsorted(ts,trs[ptr][0]))
        else: i+=1
    return realized*100+0.0, mdd

def report(tag,d,base,var):
    lines=[]
    for nm,trs in [('기존',base),('변형',var)]:
        a=np.array([t[2] for t in trs])
        if len(a)==0: lines.append((nm,0,0,0,0,0,0)); continue
        pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
        tot,mdd=dd_fast(d['ts'],d['C'],trs)
        lines.append((nm,len(a),a.mean()*100,pf,tot,mdd,(tot/mdd if mdd>0 else 99)))
    b,v=lines
    mark='✅' if v[6]>b[6]+0.1 else ('≈' if abs(v[6]-b[6])<=0.1 else '❌')
    print(f'{tag:28} 기존 n{b[1]:>4} {b[2]:>+6.2f}%/PF{b[3]:>5.2f} 총{b[4]:>+7.0f} DD{b[5]:>5.0f} MAR{b[6]:>5.1f} | 변형 {v[2]:>+6.2f}%/PF{v[3]:>5.2f} 총{v[4]:>+7.0f} DD{v[5]:>5.0f} MAR{v[6]:>5.1f} {mark}',flush=True)
    return b[6],v[6],mark

def cdms(cd):
    s=str(cd)
    if s.endswith('m'): return float(s[:-1])*60*1000
    if s.endswith('h'): return float(s[:-1])*H
    return float(s)*DAY

def main():
    res=[]
    bd=os.path.dirname(os.path.abspath(__file__))
    # ---- 일봉 S3 ----
    for grp,jf,itv in [('일봉MT5','daily_quarterly_mt5.json','D10'),('일봉Bybit','daily_quarterly_bybit.json','D')]:
        J=json.load(open(os.path.join(bd,jf),encoding='utf-8')); ds.ITV=itv
        print(f'\n===== {grp} (S3 추세 롱/숏) =====',flush=True)
        for sym,row in J.items():
            d=ds.prep(sym)
            if d is None: continue
            for key,side,dn in [('S1_long',1,'롱'),('S1_short',-1,'숏')]:
                p=row.get(key)
                if not p or p['exp']<=0: continue
                base=sim2(d,90,side,p['k1'],p['b'],p['cd']*DAY,15*DAY,1.0,False)
                var =sim2(d,90,side,p['k1'],p['b'],p['cd']*DAY,15*DAY,1/3,True)
                res.append(report(f'{sym} {dn} K{p["k1"]}/B{p["b"]}',d,base,var))
    # ---- 1분봉 S1 ----
    for grp,jl,js in [('1분MT5','results/mt5_14d_maxc15_trendlong.json','results/mt5_14d_maxc15_trend.json'),
                      ('1분Bybit','results/crypto_14d_maxc15_trendlong.json','results/crypto_14d_maxc15_trend.json')]:
        print(f'\n===== {grp} (S1 추세 롱/숏) =====',flush=True)
        JL=json.load(open(os.path.join(bd,jl))); JS=json.load(open(os.path.join(bd,js)))
        syms=[s for s in JL if s!='XAUTUSDT']
        for sym in syms:
            d=prep1m(sym)
            if d is None: continue
            for J2,side,dn in [(JL,1,'롱'),(JS,-1,'숏')]:
                rows=J2.get(sym) or []
                if not rows or rows[0]['exp']<=0: continue
                p=rows[0]
                base=sim2(d,10080,side,p['k1'],p['b'],cdms(p['cd']),14*DAY,1.0,False)
                var =sim2(d,10080,side,p['k1'],p['b'],cdms(p['cd']),14*DAY,1/3,True)
                res.append(report(f'{sym} {dn} K{p["k1"]}/B{p["b"]}',d,base,var))
    ok=sum(1 for _,_,m in res if m=='✅'); eq=sum(1 for _,_,m in res if m=='≈')
    print(f'\n===== 총괄: MAR 개선 {ok} / 동률 {eq} / 악화 {len(res)-ok-eq}  (전체 {len(res)}조합) =====')

if __name__=='__main__':
    main()
