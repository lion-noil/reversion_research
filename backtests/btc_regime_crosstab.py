# -*- coding: utf-8 -*-
"""4셀 성적을 BTC 분기 수익률 레짐(상승/횡보/하락)별로 교차분석.
상승: 분기 +10%↑ / 하락: −10%↓ / 횡보: 그 사이."""
import sys, os
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011
m1=_load_cache('BTCUSDT','1'); m5=_load_cache('BTCUSDT','5')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
C5=np.array([x['close'] for x in m5]); ts5=np.array([x['start'] for x in m5],np.int64)
def zma(C,win):
    n=len(C)
    cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd
Z={}
for w in [1320,1440]: Z[w]=zma(C1,w)
z5,_,_=zma(C5,2016)
idx=np.searchsorted(ts5+4*60000, ts1,'right')-1; idx=np.clip(idx,0,len(z5)-1)
z5v=z5[idx]; z5v[idx<0]=np.nan

def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
# 분기별 BTC 수익률 → 레짐 분류
qidx={}
for i,t in enumerate(ts1): qidx.setdefault(qof(t),[]).append(i)
qreg={}; qret={}
for q,ii in qidx.items():
    r=C1[ii[-1]]/C1[ii[0]]-1; qret[q]=r
    qreg[q]='상승' if r>0.10 else ('하락' if r<-0.10 else '횡보')

def sim(kind):
    if kind=='추세롱':   w,k1,b,mh,slm,cdh,pos,is_long,gate=(1320,5.0,-2.0,14,None,3,True,True,False)
    elif kind=='역추롱G': w,k1,b,mh,slm,cdh,pos,is_long,gate=(1320,5.0,-2.0,14,None,3,False,True,True)
    elif kind=='추세숏':  w,k1,b,mh,slm,cdh,pos,is_long,gate=(1440,6.0,-1.5,14,2.0,1,False,False,False)
    else:               w,k1,b,mh,slm,cdh,pos,is_long,gate=(1440,5.5,-0.5,7,1.5,1,True,False,False)
    z,ma,sd=Z[w]
    cand=np.flatnonzero(z>=k1) if pos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=w]
    if gate:
        g=z5v[cand]<=-2.0; cand=cand[g]
    mh_ms=int(mh*DAY); cd=int(cdh*H); out=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<cd: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+mh_ms,'right'))
        if hi<=ci+1: continue
        if is_long:
            if pos:  # 추세롱
                tpb=ma[ci]+b*sd[ci]
                if tpb>=price: continue
                tp=price*(1+(1-tpb/price))
            else:    # 역추롱
                tp=ma[ci]-b*sd[ci]
                if tp<=price: continue
            TPh=H1[ci+1:hi]>=tp; SLh=np.zeros(hi-ci-1,bool); sl=None
        else:
            if not pos:  # 추세숏 (z<=-K)
                tpb=ma[ci]-b*sd[ci]
                if tpb<=price: continue
                pct=tpb/price-1; tp=price*(1-pct)
            else:        # 역추숏 (z>=+K)
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

print('분기 레짐 분류 (BTC 분기수익):')
for q in sorted(qret): print(f'  {q}: {qret[q]*100:+.0f}% → {qreg[q]}')
print()
print(f'{"셀":10} | {"상승장":^26} | {"횡보장":^26} | {"하락장":^26}')
TRS={}
for kind in ['추세롱','역추롱G','추세숏','역추숏']:
    tr=sim(kind); TRS[kind]=tr
    cols=[]
    for reg in ['상승','횡보','하락']:
        rs=[r for t,r in tr if qreg[qof(t)]==reg]
        if len(rs)<5: cols.append(f'n{len(rs)} 표본부족'.center(26)); continue
        a=np.array(rs); pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
        cols.append(f'n{len(a):>3} {a.mean()*100:+.2f}% PF{pf:.2f} 승{100*(a>0).mean():.0f}%'.center(26))
    print(f'{kind:10} | '+' | '.join(cols))
if os.getenv('QDETAIL'):
    for kind,tr in TRS.items():
        a=np.array([r for _,r in tr])
        pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
        print(f'\n=== {kind} 전체: n{len(a)} 기대{a.mean()*100:+.3f}% PF{pf:.2f} 승{100*(a>0).mean():.0f}% 최악{a.min()*100:+.1f}% ===')
        byq={}
        for t,r in tr: byq.setdefault(qof(t),[]).append(r)
        for k in sorted(byq):
            aa=np.array(byq[k])
            print(f'  {k}({qreg[k]}): n{len(aa):>3} 합{aa.sum()*100:>+7.1f}% 승{100*(aa>0).mean():>3.0f}% 최악{aa.min()*100:>+6.1f}%')
