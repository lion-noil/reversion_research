# -*- coding: utf-8 -*-
"""레짐 게이트 전면 스윕 (6.3년) — 4개 베이스 게임 × 게이트창(1~14d) × 부호(순방향/역방향) × 강도.
순방향(same): 레짐이 게임과 같은 쪽 / 역방향(oppo): 레짐이 반대쪽일 때 진입(불장눌림목·베어랠리페이드 등).
베이스(6.3년 재최적): 추세롱 22h/K5/B-2/h14/SL무/cd3h · 역추롱 동일게임 · 추세숏 24h/K6/B1/h14/SL2/cd3h · 역추숏 24h/K5.5/B-1/h2/SL0.5/cd1h"""
import sys, os, time, json
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011
m1=_load_cache('BTCUSDT','1'); m5=_load_cache('BTCUSDT','5')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
ts5=np.array([x['start'] for x in m5],np.int64); C5=np.array([x['close'] for x in m5])
months=(ts1[-1]-ts1[0])/(30.44*DAY)
def zma(C,win):
    n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd
Z1={}
for w in [1320,1440]: Z1[w]=zma(C1,w)
idx=np.searchsorted(ts5+4*60000, ts1,'right')-1; idx=np.clip(idx,0,len(ts5)-1)
Z5={}
for wg in [288,576,864,1440,2016,2880,4032]:
    z5,_,_=zma(C5,wg)
    v=z5[idx]; v=v.copy(); v[idx<0]=np.nan; Z5[wg]=v

BASE={
 '추세롱':dict(w=1320,k1=5.0,b=-2.0,mh=14,slm=None,cdh=3,epos=True, is_long=True),
 '역추롱':dict(w=1320,k1=5.0,b=-2.0,mh=14,slm=None,cdh=3,epos=False,is_long=True),
 '추세숏':dict(w=1440,k1=6.0,b=1.0, mh=14,slm=2.0, cdh=3,epos=False,is_long=False),
 '역추숏':dict(w=1440,k1=5.5,b=-1.0,mh=2, slm=0.5, cdh=1,epos=True, is_long=False),
}
def sim(cfg,gate=None):
    z,ma,sd=Z1[cfg['w']]; epos=cfg['epos']; is_long=cfg['is_long']
    k1,b,slm=cfg['k1'],cfg['b'],cfg['slm']
    cand=np.flatnonzero(z>=k1) if epos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=cfg['w']]
    if gate is not None:
        z5v,k5f,sign=gate
        zz=z5v[cand]
        pos_side = epos if sign=='same' else (not epos)
        g=(zz>=k5f) if pos_side else (zz<=-k5f)
        cand=cand[g]
    mh=int(cfg['mh']*DAY); cd=int(cfg['cdh']*H); out=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<cd: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+mh,'right'))
        if hi<=ci+1: continue
        if is_long:
            if epos:
                tpb=ma[ci]+b*sd[ci]
                if tpb>=price: continue
                tp=price*(1+(1-tpb/price))
            else:
                tp=ma[ci]-b*sd[ci]
                if tp<=price: continue
            TPh=H1[ci+1:hi]>=tp; SLh=np.zeros(hi-ci-1,bool); sl=None
        else:
            if not epos:
                tpb=ma[ci]-b*sd[ci]
                if tpb<=price: continue
                pct=tpb/price-1; tp=price*(1-pct)
            else:
                tp=ma[ci]+b*sd[ci]
                if tp>=price: continue
                pct=1-tp/price
            TPh=L1[ci+1:hi]<=tp
            sl=price*(1+pct*slm); SLh=H1[ci+1:hi]>=sl
        hit=SLh|TPh
        if hit.any():
            j=int(np.argmax(hit)); ex=ci+1+j; xp=sl if SLh[j] else tp
        else: ex=hi-1; xp=C1[ex]
        r=(xp/price-1-FEE) if is_long else ((price-xp)/price-FEE)
        out.append((int(ts1[ci]),r)); last=ts1[ci]
    return out
def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
def st(tr):
    a=np.array([r for _,r in tr])
    if len(a)<40: return None
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    return a,pf
t0=time.time()
for nm,cfg in BASE.items():
    tr0=sim(cfg); s0=st(tr0)
    a0,pf0=s0
    print(f'===== {nm} 베이스(게이트없음): n{len(a0)}({len(a0)/months:.1f}/월) 기대{a0.mean()*100:+.3f}% PF{pf0:.2f} 승{100*(a0>0).mean():.0f}% 최악{a0.min()*100:+.1f}% =====',flush=True)
    for sign,snm in [('same','순방향(레짐동조)'),('oppo','역방향(레짐역행)')]:
        rows=[]
        for wg in [288,576,864,1440,2016,2880,4032]:
            for k5f in [0.5,1.0,1.5,2.0,2.5]:
                tr=sim(cfg,gate=(Z5[wg],k5f,sign))
                s=st(tr)
                if s is None: continue
                a,pf=s
                rows.append((a.mean(),pf,wg,k5f,len(a),a,tr))
        if not rows: print(f'  [{snm}] 표본충족 조합 없음'); continue
        rows.sort(reverse=True,key=lambda r:(r[0],r[1]))
        print(f'  [{snm}] 상위 3:')
        for e,pf,wg,k5f,n,a,tr in rows[:3]:
            print(f'    게이트창{wg//288}d K5f{k5f}: n{n}({n/months:.1f}/월) 기대{e*100:+.3f}% PF{pf:.2f} 승{100*(a>0).mean():.0f}% 최악{a.min()*100:+.1f}% 월합{n/months*e*100:+.1f}%p')
        # 최고 셀 분기별(연도 요약: 2020~22 vs 2023~26)
        e,pf,wg,k5f,n,a,tr=rows[0]
        pre=[r for t,r in tr if t< 1672531200000]  # ~2023-01-01
        post=[r for t,r in tr if t>=1672531200000]
        if len(pre)>5 and len(post)>5:
            p1,p2=np.array(pre),np.array(post)
            print(f'    └ 최고셀 기간분해: 2020~22(하락장포함) n{len(p1)} {p1.mean()*100:+.2f}%/합{p1.sum()*100:+.0f}%p | 2023~26 n{len(p2)} {p2.mean()*100:+.2f}%/합{p2.sum()*100:+.0f}%p',flush=True)
    print(f'({time.time()-t0:.0f}s)')
