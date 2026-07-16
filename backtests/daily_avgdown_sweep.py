# -*- coding: utf-8 -*-
"""일봉(10년·MA90) + 1추매(avg-down) 스윕, 연~30거래 빈도 타겟. S1·S2 4방향.
bt_avgdown.sim_avg 로직 그대로(단일포지션, 재신호1회 추매, 평단·TP/SL 재앵커), win=90/일봉캐시.
연거래수(games/yr)가 [GLO,GHI]에 드는 조합 중 기대값 최대 선택."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache

FEE=0.0011; DAY=86400*1000
WIN=int(os.getenv('WIN_D','90')); MAXHOLD=int(os.getenv('MAXHOLD_D','30'))*DAY
ITV=os.getenv('DAILY_ITV','D10'); MAXADDS=int(os.getenv('MAXADDS','1'))
GLO=float(os.getenv('GLO','25')); GHI=float(os.getenv('GHI','35'))   # 연거래 타겟창
K1S=[round(1.0+0.1*i,2) for i in range(26)]   # 1.0~3.5
BS=[round(-2+0.2*i,2) for i in range(21)]
CDD=[1,2,3,5,7,10]
SYMS=['EURUSD','XAUUSD','XAGUSD','WTI','US100','JP225','GER40','UK100','HK50','BTCUSD','ETHUSD',
      'GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD']
SIDE={'rev_long':1,'rev_short':-1,'trend_long':1,'trend_short':-1}
ZS={'rev_long':-1,'rev_short':1,'trend_long':1,'trend_short':-1}
S1KINDS=['trend_long','trend_short']; S2KINDS=['rev_long','rev_short']

def prep(sym):
    m=_load_cache(sym,ITV)
    if not m or len(m)<=WIN: return None
    C=np.array([x['close'] for x in m]);H=np.array([x['high'] for x in m]);L=np.array([x['low'] for x in m])
    ts=np.array([x['start'] for x in m],np.int64);n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]);cs2=np.concatenate([[0.],np.cumsum(C*C)]);w=WIN
    ma=np.full(n,np.nan);sd=np.full(n,np.nan);s=cs[w:]-cs[:-w];s2=cs2[w:]-cs2[:-w];mm=s/w
    ma[w-1:]=mm;sd[w-1:]=np.sqrt(np.maximum(s2/w-mm*mm,0));z=np.where(sd>0,(C-ma)/sd,np.nan)
    return dict(C=C,H=H,L=L,ts=ts,ma=ma,sd=sd,z=z,n=n)

def tgt(kind,ma_k,sd_k,avg,ref,b):
    if kind=='rev_long':
        tp=ma_k-b*sd_k
        if tp<=avg: return 0,0,False
        pct=tp/avg-1; return tp,avg*(1-pct),True
    if kind=='rev_short':
        tp=ma_k+b*sd_k
        if tp>=avg: return 0,0,False
        pct=1-tp/avg; return tp,avg*(1+pct),True
    if kind=='trend_long':
        band=ma_k+b*sd_k; pct=1-band/ref
        if pct<=0: return 0,0,False
        return avg*(1+pct),avg*(1-pct),True
    band=ma_k-b*sd_k; pct=band/ref-1
    if pct<=0: return 0,0,False
    return avg*(1-pct),avg*(1+pct),True

def sim(d,kind,k1,b,cd_ms,max_adds=1):
    C,H,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    side=SIDE[kind];zs=ZS[kind]
    cand=(np.flatnonzero(z>=k1) if zs>0 else np.flatnonzero(z<=-k1)); cand=cand[cand>=WIN].tolist()
    sig=(z>=k1) if zs>0 else (z<=-k1)
    out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        p0=C[ci]; tp,sl,ok=tgt(kind,ma[ci],sd[ci],p0,p0,b)
        if not ok: continue
        en=ts[ci]; legs=[p0]; avg=p0; adds=0; end=en+MAXHOLD
        exi=expx=None; k=ci+1
        while k<n and ts[k]<=end:
            lo,hi=L[k],H[k]
            if side>0:
                if lo<=sl: exi,expx=k,sl; break
                if hi>=tp: exi,expx=k,tp; break
            else:
                if hi>=sl: exi,expx=k,sl; break
                if lo<=tp: exi,expx=k,tp; break
            if adds<max_adds and (ts[k]-en)>=cd_ms and sig[k]:
                p1=C[k]; na=(sum(legs)+p1)/(len(legs)+1); nt,ns,nok=tgt(kind,ma[k],sd[k],na,p1,b)
                if nok: legs.append(p1); avg=na; tp,sl=nt,ns; adds+=1
            k+=1
        if exi is None: kk=min(k,n-1); exi,expx=kk,C[kk]
        if side>0: lg=[expx/p-1 for p in legs]
        else: lg=[(p-expx)/p for p in legs]
        out.append((float(np.mean(lg)-FEE),adds>0)); last=ts[exi]
    return out

def st(o):
    if not o: return None
    r=np.array([x[0] for x in o]); w=r>0
    pf=r[w].sum()/(-r[~w].sum()) if (~w).any() else 99.0
    return dict(n=len(r),add=int(sum(x[1] for x in o)),wr=round(100*w.mean()),exp=round(float(r.mean())*100,3),pf=round(float(pf),2))

def main():
    print(f'일봉추매 스윕[{ITV}] MA{WIN}/보유{MAXHOLD//DAY}d/추매{MAXADDS} | 연거래 {GLO:.0f}~{GHI:.0f} 타겟 | K1 1.0~3.5')
    res={}
    for s in SYMS:
        d=prep(s)
        if d is None: print(s,'없음'); continue
        yrs=(int(d['ts'][-1])-int(d['ts'][WIN]))/(365.25*DAY)
        res[s]={}
        for grp,kinds in [('S1',S1KINDS),('S2',S2KINDS)]:
            top=None
            for kind in kinds:
                for k1 in K1S:
                    for b in BS:
                        for cdd in CDD:
                            o=sim(d,kind,k1,b,cdd*DAY,MAXADDS); ss=st(o)
                            if not ss: continue
                            gpy=ss['n']/yrs
                            if not (GLO<=gpy<=GHI): continue
                            key=(ss['exp'],ss['pf'])
                            if top is None or key>top[0]: top=(key,kind,k1,b,cdd,gpy,ss)
            res[s][grp]=top
    # 출력
    print(f'\n{"심볼":8} | S1(추세, 추매{MAXADDS}, ~30/yr) | S2(역추세, 추매{MAXADDS}, ~30/yr)')
    def f(t):
        if not t: return '연30 타겟 조합 없음'
        _,kind,k1,b,cdd,gpy,ss=t
        return f"{kind:11} K{k1}/B{b}/{cdd}d 기대{ss['exp']:+.2f}%/PF{ss['pf']}/n{ss['n']}({gpy:.0f}/yr)/추매{ss['add']}"
    for s in SYMS:
        if s not in res: continue
        print(f'{s:8} | {f(res[s].get("S1"))}\n         | {f(res[s].get("S2"))}')
    json.dump({s:{g:(t[1:] if t else None) for g,t in res[s].items()} for s in res},
              open(os.path.join(os.path.dirname(__file__),'results',f'daily{WIN}_avgdown_best.json'),'w'),ensure_ascii=False)
    print('\n→ results/daily90_avgdown_best.json')

if __name__=='__main__':
    main()
