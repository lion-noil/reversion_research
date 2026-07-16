# -*- coding: utf-8 -*-
"""①추세숏·역추숏 SL제거 최적탐색(창22h, K·B·보유 스윕) + 분기별.
②역추롱(K5.0/B-2.0/14d) 게이트 없음 vs 5분 7d레짐 게이트(z5<=-2.0) 분기별 비교."""
import sys, os
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011; W1=1320
m1=_load_cache('BTCUSDT','1'); m5=_load_cache('BTCUSDT','5')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
C5=np.array([x['close'] for x in m5]); ts5=np.array([x['start'] for x in m5],np.int64)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
def zma(C,win):
    n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd
z,ma,sd=zma(C1,W1)
z5,_,_=zma(C5,2016)
idx=np.searchsorted(ts5+4*60000, ts1,'right')-1; idx=np.clip(idx,0,len(z5)-1)
z5v=z5[idx]; z5v[idx<0]=np.nan

def sim(DIR,k1,b,mh_d,gate_k5f=None):
    """DIR: trendlong/long(롱), trend/short(숏). SL 없음(TP+만기)."""
    entry_pos = DIR in('trendlong','short')   # z 진입 부호 +
    is_long   = DIR in('trendlong','long')
    cand=np.flatnonzero(z>=k1) if entry_pos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=W1]
    if gate_k5f is not None:
        g=(z5v[cand]>=gate_k5f) if entry_pos else (z5v[cand]<=-gate_k5f)
        cand=cand[g]
    mh=int(mh_d*DAY); out=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<3*H: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+mh,'right'))
        if hi<=ci+1: continue
        if DIR=='trendlong':
            tpb=ma[ci]+b*sd[ci]
            if tpb>=price: continue
            tp=price*(1+(1-tpb/price)); TPh=H1[ci+1:hi]>=tp
        elif DIR=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            TPh=H1[ci+1:hi]>=tp
        elif DIR=='trend':
            tpb=ma[ci]-b*sd[ci]
            if tpb<=price: continue
            tp=price*(1-(tpb/price-1)); TPh=L1[ci+1:hi]<=tp
        else:  # 역추숏
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            TPh=L1[ci+1:hi]<=tp
        if TPh.any(): ex=ci+1+int(np.argmax(TPh)); xp=tp
        else: ex=hi-1; xp=C1[ex]
        r=(xp/price-1-FEE) if is_long else ((price-xp)/price-FEE)
        out.append((int(ts1[ci]),r)); last=ts1[ci]
    return out

def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
def agg(tr):
    a=np.array([r for _,r in tr])
    if len(a)==0: return None
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    return len(a),a.mean()*100,pf,100*(a>0).mean(),a.min()*100
def quarterly(tr,label):
    byq={}
    for t,r in tr: byq.setdefault(qof(t),[]).append(r)
    print(f'  --- {label} 분기별 ---')
    for k in sorted(byq):
        a=np.array(byq[k])
        print(f'    {k}: n{len(a):>3} 합{a.sum()*100:>+7.1f}% 승{100*(a>0).mean():>3.0f}% 최악{a.min()*100:>+6.1f}%')

# ① 숏 2방향 최적 탐색
for DIR,nm in [('trend','추세숏'),('short','역추숏')]:
    best=None
    for k1 in [4.0,4.5,5.0,5.5,6.0]:
        for b in [-3.0,-2.5,-2.0,-1.5,-1.0,0.0,1.0]:
            for mh in [7,14]:
                tr=sim(DIR,k1,b,mh)
                if len(tr)<40: continue
                a=np.array([r for _,r in tr]); e=a.mean()
                if best is None or e>best[0]: best=(e,k1,b,mh,tr)
    if best is None: print(f'===== {nm}: n>=40 조합 없음 ====='); continue
    e,k1,b,mh,tr=best
    n,ex,pf,wr,mn=agg(tr)
    print(f'===== {nm} SL제거 최적: K{k1}/B{b}/보유{mh}d → n{n}({n/months:.1f}/월) 기대{ex:+.3f}% PF{pf:.2f} 승{wr:.0f}% 최악{mn:+.1f}% =====')
    quarterly(tr,nm)
    print()
# ② 역추롱 게이트 유/무 분기별
tr0=sim('long',5.0,-2.0,14)
tr1=sim('long',5.0,-2.0,14,gate_k5f=2.0)
n,ex,pf,wr,mn=agg(tr0); print(f'===== 역추롱 게이트없음: n{n}({n/months:.1f}/월) 기대{ex:+.3f}% PF{pf:.2f} 승{wr:.0f}% 최악{mn:+.1f}% =====')
n,ex,pf,wr,mn=agg(tr1); print(f'===== 역추롱 +7d레짐게이트(z5<=-2): n{n}({n/months:.1f}/월) 기대{ex:+.3f}% PF{pf:.2f} 승{wr:.0f}% 최악{mn:+.1f}% =====')
quarterly(tr0,'게이트 없음')
quarterly(tr1,'게이트 z5(7d)<=-2.0')
