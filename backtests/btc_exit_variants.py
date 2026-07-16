# -*- coding: utf-8 -*-
"""1분봉 승자 셀(창22h/보유7d)에 청산기준 변형 테스트.
A 현행(고정 TP/SL 대칭) / B SL제거(TP+만기) / C z-복귀 청산(Zx 스윕) / D z-복귀+안전SL.
역추롱: K4.75 진입(z<=-4.75), 추세롱: K5.0 진입(z>=+5.0). 쿨다운 3h, fee 0.11%."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
H=3600*1000; DAY=86400*1000; FEE=0.0011; MH=7*DAY; W1=1320  # 22h

m1=_load_cache('BTCUSDT','1')
C1=np.array([x['close'] for x in m1]); H1=np.array([x['high'] for x in m1]); L1=np.array([x['low'] for x in m1])
ts1=np.array([x['start'] for x in m1],np.int64); n1=len(C1)
months=(ts1[-1]-ts1[0])/(30.44*DAY)
cs=np.concatenate([[0.],np.cumsum(C1)]); cs2=np.concatenate([[0.],np.cumsum(C1*C1)])
ma=np.full(n1,np.nan); sd=np.full(n1,np.nan)
s=cs[W1:]-cs[:-W1]; s2=cs2[W1:]-cs2[:-W1]; mm=s/W1
ma[W1-1:]=mm; sd[W1-1:]=np.sqrt(np.maximum(s2/W1-mm*mm,0))
z=np.where(sd>0,(C1-ma)/sd,np.nan)

def sim(DIR,k1,cd_ms,mode,b=None,zx=None,slmult=None):
    """mode: 'band'(TP/SL대칭) 'tponly'(SL없음) 'zexit'(z복귀) 'zexit_sl'."""
    pos = DIR=='trendlong'
    cand=np.flatnonzero(z>=k1) if pos else np.flatnonzero(z<=-k1)
    cand=cand[cand>=W1]
    out={'TP':[], 'SL':[], 'TIME':[], 'ZX':[]}; hold=[]; iv=[]; last=None
    for ci in cand:
        if ci+1>=n1: break
        if last is not None and (ts1[ci]-last)<cd_ms: continue
        price=C1[ci]; hi=int(np.searchsorted(ts1,ts1[ci]+MH,'right'))
        if hi<=ci+1: continue
        sl_lvl=None; tp_lvl=None
        if mode in('band','tponly'):
            if pos:
                tpb=ma[ci]+b*sd[ci]
                if tpb>=price: continue
                pct=1-tpb/price; tp_lvl=price*(1+pct)
                if mode=='band': sl_lvl=price*(1-pct)
            else:
                tpb=ma[ci]-b*sd[ci]
                if tpb<=price: continue
                pct=tpb/price-1; tp_lvl=tpb
                if mode=='band': sl_lvl=price*(1-pct)
        elif mode=='zexit_sl':
            sl_lvl=price*(1-slmult*sd[ci]/price) if not pos else price*(1-slmult*sd[ci]/price)
        sl_seg=None
        seg=slice(ci+1,hi)
        # 청산 탐색
        SL=np.zeros(hi-ci-1,bool); TP=np.zeros(hi-ci-1,bool); ZX=np.zeros(hi-ci-1,bool)
        if sl_lvl is not None: SL=L1[seg]<=sl_lvl
        if tp_lvl is not None: TP=H1[seg]>=tp_lvl
        if mode in('zexit','zexit_sl'):
            ZX=(z[seg]<=zx) if pos else (z[seg]>=zx)
        hit=SL|TP|ZX
        if hit.any():
            j=int(np.argmax(hit)); ex=ci+1+j
            if SL[j]: xp=sl_lvl; key='SL'
            elif TP[j]: xp=tp_lvl; key='TP'
            else: xp=C1[ex]; key='ZX'
        else:
            ex=hi-1; xp=C1[ex]; key='TIME'
        r=(xp/price-1-FEE)   # 롱 계열만
        out[key].append(r); hold.append((ts1[ex]-ts1[ci])/DAY); iv.append((int(ts1[ci]),int(ts1[ex]))); last=ts1[ci]
    ev=[]
    for en,ex_ in iv: ev.append((en,1)); ev.append((ex_,-1))
    ev.sort(key=lambda x:(x[0],x[1])); cur=mx=0
    for _,dd in ev: cur+=dd; mx=max(mx,cur)
    return out,hold,mx

def rep(tag,o,hold,mx):
    a=np.array(o['TP']+o['SL']+o['TIME']+o['ZX']); n=len(a)
    if n<30: print(f'  {tag:32} n<30'); return
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    sh=' '.join(f'{k}{100*len(v)/n:.0f}%' for k,v in o.items() if v)
    print(f'  {tag:32} n{n:>4}({n/months:>4.1f}/월) 기대{a.mean()*100:>+7.3f}% PF{pf:>5.2f} 승{100*(a>0).mean():>3.0f}% 최악{a.min()*100:>+6.1f}% mc{mx} 보유{np.mean(hold):.1f}d | {sh}')

for DIR,k1,nm in [('long',4.75,'역추롱'),('trendlong',5.0,'추세롱')]:
    print(f'===== {nm} (창22h K{k1} cd3h 보유≤7d) =====')
    # A 현행 밴드 (B 스윕)
    for b in ([-3.0,-2.5,-2.0] if DIR=='long' else [-2.5,-2.0,-1.5]):
        o,hd,mx=sim(DIR,k1,3*H,'band',b=b)
        rep(f'A 밴드 B{b}',o,hd,mx)
    # B SL 제거
    for b in ([-3.0,-2.5,-2.0] if DIR=='long' else [-2.5,-2.0,-1.5]):
        o,hd,mx=sim(DIR,k1,3*H,'tponly',b=b)
        rep(f'B SL제거 B{b}',o,hd,mx)
    # C z-복귀 청산
    ZXS=[-2.0,-1.0,0.0,1.0] if DIR=='long' else [3.0,2.0,1.0,0.0]
    for zx in ZXS:
        o,hd,mx=sim(DIR,k1,3*H,'zexit',zx=zx)
        rep(f'C z복귀 Zx={zx}',o,hd,mx)
    # D z-복귀 + 안전SL(진입가-slmult×σ)
    for zx,sm in ([(0.0,3.0),(0.0,5.0),(-1.0,3.0)] if DIR=='long' else [(1.0,3.0),(1.0,5.0),(2.0,3.0)]):
        o,hd,mx=sim(DIR,k1,3*H,'zexit_sl',zx=zx,slmult=sm)
        rep(f'D z복귀{zx}+SL{sm}σ',o,hd,mx)
    print()
