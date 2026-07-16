# -*- coding: utf-8 -*-
"""MT5·FX 일봉책(mt5d·fxd 볼드셀) vs S22 확장판(16셀) — 겹침·상관·한계기여 검증.
크립토에서 했던 s3s4_vs_s22 분석의 U2·U3판. 연도별 수익(각 $10k·2%), 상관, 결합 기여."""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache
FEE=0.0011; H=3600*1000; DAY=86400*1000
FXs={'USDJPY','EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','NZDUSD'}

def zma(C,win):
    n=len(C); cs=np.concatenate([[0.],np.cumsum(C)]); cs2=np.concatenate([[0.],np.cumsum(C*C)])
    ma=np.full(n,np.nan); sd=np.full(n,np.nan)
    s=cs[win:]-cs[:-win]; s2=cs2[win:]-cs2[:-win]; mm=s/win
    ma[win-1:]=mm; sd[win-1:]=np.sqrt(np.maximum(s2/win-mm*mm,0))
    return np.where(sd>0,(C-ma)/sd,np.nan),ma,sd

# ── 일봉 v2 (TP/SL 대칭·보유15d) — s4_check와 동일 ──
def daily_cell(sym,win,k,b,cdd,mode):
    m=_load_cache(sym,'D10')
    if not m or len(m)<=win: return []
    fee=FEE
    C=np.array([x['close'] for x in m]); Hh=np.array([x['high'] for x in m])
    L=np.array([x['low'] for x in m]); ts=np.array([x['start'] for x in m],np.int64); n=len(C)
    z,ma,sd=zma(C,win)
    if mode=='long': cand=np.flatnonzero(z<=-k)
    elif mode=='short': cand=np.flatnonzero(z>=k)
    else: cand=np.flatnonzero(z>=k)
    cand=cand[cand>=win]; out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cdd*DAY: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+15*DAY,'right'))
        if hi<=ci+1: continue
        if mode=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-fee) if SL[j] else (tp/price-1-fee)
            else: ex=hi-1; r=C[ex]/price-1-fee
        elif mode=='short':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=Hh[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=((price-sx)/price-fee) if SL[j] else ((price-tp)/price-fee)
            else: ex=hi-1; r=(price-C[ex])/price-fee
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=Hh[ci+1:hi]>=tx; hit=SL|TP
            if hit.any(): j=int(np.argmax(hit)); ex=ci+1+j; r=(sx/price-1-fee) if SL[j] else (tx/price-1-fee)
            else: ex=hi-1; r=C[ex]/price-1-fee
        out.append((int(ts[ci]),int(ts[ex]),r)); last=ts[ci]
    return out

# mt5d 볼드셀 (§3-A) + fxd 볼드셀 (§3-B)
MT5D=[]
MT5D+=daily_cell('BTCUSD',90,2.5,-2.0,2,'tl'); MT5D+=daily_cell('ETHUSD',60,2.8,-3.0,1,'tl')
MT5D+=daily_cell('XAUUSD',90,2.2,-1.4,3,'tl'); MT5D+=daily_cell('XAUUSD',200,2.8,1.2,1,'short')
MT5D+=daily_cell('XAGUSD',200,2.5,0.8,1,'tl'); MT5D+=daily_cell('WTI',200,1.4,0.8,5,'long')
MT5D+=daily_cell('US100',150,2.0,-3.0,1,'long'); MT5D+=daily_cell('JP225',200,2.0,-3.0,1,'long')
MT5D+=daily_cell('GER40',200,2.2,-3.0,1,'long')
MT5D+=daily_cell('UK100',90,1.7,-0.8,2,'long'); MT5D+=daily_cell('UK100',200,2.3,1.0,1,'short')
MT5D+=daily_cell('HK50',200,1.8,-3.0,3,'long'); MT5D+=daily_cell('HK50',150,2.2,2.0,1,'short')
FXD=[]
FXD+=daily_cell('EURUSD',200,2.2,-0.6,1,'long'); FXD+=daily_cell('GBPUSD',150,2.4,1.6,1,'long')
FXD+=daily_cell('USDJPY',120,2.6,-1.0,1,'tl'); FXD+=daily_cell('AUDUSD',90,2.3,-2.2,1,'long')
FXD+=daily_cell('USDCAD',150,1.9,1.2,1,'long'); FXD+=daily_cell('USDCHF',200,2.3,-0.6,1,'long')
FXD+=daily_cell('NZDUSD',200,2.2,-3.0,1,'long')

# ── S22 확장판 16셀 (s22x_portfolio와 동일 — 트레이드만 재생성) ──
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'s22x_portfolio.py'),encoding='utf-8').read().split('CELLS=[]')[0].split('# ── 셀간')[0].replace('FRAC=float','_FRAC=float'))
S22X_NONFX=[]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in zcell('JP225-z추롱','JP225',120,3.0,1,'tl',12)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in sweepcell('JP225-스윕','JP225',120,60)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in fadecell('JP225-페이드','JP225',18,0.05,'T',60)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in zcell('US100-z역롱','US100',180,3.0,1,'rl',12)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in fadecell('US100-페이드','US100',30,0.07,'R',30)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in zcell('HK50-z추롱','HK50',180,3.0,-2,'tl',12)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in zcell('HK50-z역롱','HK50',180,3.0,1,'rl',12)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in fadecell('UK100-페이드','UK100',12,0.03,'T',60)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in ewzcell('XAGUSD-ewz역롱','XAGUSD',200,2.5,30,1,'lo')]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in ewzcell('XAGUSD-ewz추숏','XAGUSD',50,3.0,18,-1,'lo')]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in fadecell('XAUUSD-페이드','XAUUSD',6,0.03,'T',60)]
S22X_NONFX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in fadecell('WTI-페이드','WTI',6,0.07,'T',60)]
S22X_FX=[]
S22X_FX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in zcell('USDJPY-z역롱','USDJPY',120,2.5,1,'rl',12)]
S22X_FX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in zcell('AUDUSD-z역숏','AUDUSD',30,3.0,-2,'rs',12)]
S22X_FX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in zcell('NZDUSD-z역숏','NZDUSD',60,3.0,-1,'rs',24)]
S22X_FX+=[(a,b,c) for a,b,c,_s,_sd,_e,_t in sweepcell('USDCAD-스윕','USDCAD',120,60)]

def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'
def run_book(trades,frac=0.02):
    ev=[]
    for i,(en,ex,r) in enumerate(trades): ev.append((en,0,i)); ev.append((ex,1,i))
    ev.sort(key=lambda x:(x[0],x[1]))
    eq=10000.0; open_={}; qend={}; peak=eq; mdd=0.0
    for t,typ,i in ev:
        if typ==0: open_[i]=frac*eq
        else:
            no=open_.pop(i,None)
            if no is not None:
                eq+=no*trades[i][2]; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak)
        qend[qof(t)]=eq
    return qend,eq,mdd
def qrets(qend):
    out={}; prev=10000.0
    for q in sorted(qend):
        out[q]=(qend[q]/prev-1)*100; prev=qend[q]
    return out

for label,daily,s22x in [('U2: mt5d(일봉13셀) vs S22x비FX(12셀)',MT5D,S22X_NONFX),
                          ('U3: fxd(일봉7셀) vs S22x FX(4셀)',FXD,S22X_FX)]:
    qd,eqd,mdd_d=run_book(sorted(daily)); qs,eqs,mdd_s=run_book(sorted(s22x))
    qc,eqc,mdd_c=run_book(sorted(daily+s22x))
    common=sorted(set(qd)&set(qs))
    rd=np.array([qrets(qd).get(q,0) for q in common]); rs=np.array([qrets(qs).get(q,0) for q in common])
    corr=np.corrcoef(rd,rs)[0,1] if len(common)>4 else float('nan')
    print(f'=== {label} ===')
    print(f'  일봉: ${eqd:,.0f} (MDD {mdd_d*100:.1f}%) | S22x: ${eqs:,.0f} (MDD {mdd_s*100:.1f}%)')
    print(f'  분기수익 상관: {corr:.2f} (공통 {len(common)}분기)')
    print(f'  결합: ${eqc:,.0f} — S22x 단독 대비 일봉의 한계기여 {((eqc/eqs)-1)*100:+.1f}%, 결합MDD {mdd_c*100:.1f}%')
    # 진입 겹침(5일)
    de=np.array(sorted(en for en,_,_ in daily)); se=np.array(sorted(en for en,_,_ in s22x))
    W=5*DAY
    near=sum(1 for x in se if len(de) and min(abs(de-x))<=W)
    print(f'  S22x 진입 중 일봉과 5일내 겹침: {100*near/max(len(se),1):.0f}%')
    print()
