# -*- coding: utf-8 -*-
"""S3 추세롱 + 바스켓 본전청산 변형.
규칙: 오픈 포지션 ≥ CMIN개 && 당일 저가가 바스켓 평단×(1+BUF) 이하 터치 → 전량 그 가격에 청산.
그 외엔 기존 S3 그대로(개별 TP/SL 손절우선, 최대보유 15일 종가청산, 진입쿨다운).
CMIN=inf → 순수 S3(베이스라인). MTM 낙폭·MAR 포함 비교."""
import sys, os
os.environ.setdefault('MAXHOLD_D','15')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import daily_sweep as ds
from daily_sweep import DAY, WIN
from s5_dd import dd_curve

FEE=0.0011

def sim(d,k1,b,cd_d,cmin,buf,maxhold_d=15):
    C,H,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cd=cd_d*DAY; mh=maxhold_d*DAY
    pos=[]  # dict(ei, ep, tp, sl, exp_ts)
    out=[]  # (en,ex,r,ep,1.0)
    last=None
    sig=z>=k1
    for i in range(WIN,n):
        t=ts[i]
        # 1) 바스켓 본전청산 (쌓임+평단 터치)
        if len(pos)>=cmin:
            avg=sum(p['ep'] for p in pos)/len(pos); trig=avg*(1+buf)
            if L[i]<=trig and ts[pos[-1]['ei']]<t:   # 당일 신규진입 제외 위해 진입일 다음날부터
                for p in pos:
                    out.append((int(ts[p['ei']]),int(t),trig/p['ep']-1-FEE,p['ep'],1.0))
                pos=[]
        # 2) 개별 TP/SL/만기 (손절우선)
        keep=[]
        for p in pos:
            if ts[p['ei']]>=t: keep.append(p); continue
            if L[i]<=p['sl']: out.append((int(ts[p['ei']]),int(t),p['sl']/p['ep']-1-FEE,p['ep'],1.0))
            elif H[i]>=p['tp']: out.append((int(ts[p['ei']]),int(t),p['tp']/p['ep']-1-FEE,p['ep'],1.0))
            elif t-ts[p['ei']]>=mh: out.append((int(ts[p['ei']]),int(t),C[i]/p['ep']-1-FEE,p['ep'],1.0))
            else: keep.append(p)
        pos=keep
        # 3) 진입 (종가, 쿨다운)
        if sig[i] and i+1<n and (last is None or t-last>=cd):
            price=C[i]; tpb=ma[i]+b*sd[i]
            if tpb<price:
                pct=1-tpb/price
                pos.append(dict(ei=i,ep=price,tp=price*(1+pct),sl=price*(1-pct)))
                last=t
    for p in pos:  # 잔여 종가청산
        out.append((int(ts[p['ei']]),int(ts[n-1]),C[n-1]/p['ep']-1-FEE,p['ep'],1.0))
    return out

CFG=[('BTCUSD','D10',2.5,-2.0,2),('ETHUSD','D10',2.0,-2.4,5),
     ('SOLUSDT','D',2.9,-3.0,1),('XRPUSDT','D',3.1,-3.0,1)]

def row(tag,d,trs):
    a=np.array([t[2] for t in trs])
    if len(a)==0: print(f'  {tag:26} 거래없음'); return
    pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
    tot,mdd,avgc=dd_curve(d,trs)
    print(f'  {tag:26} n{len(a):>4} 승{100*(a>0).mean():>3.0f}% 기대{a.mean()*100:>+7.2f}% PF{pf:>6.2f} | 총{tot:>+7.0f}%p DD{mdd:>6.0f}%p MAR{(tot/mdd if mdd>0 else 99):>5.1f}')

def main():
    for sym,itv,k1,b,cd in CFG:
        ds.ITV=itv; d=ds.prep(sym)
        if d is None: continue
        print(f'\n===== {sym} S3롱 K{k1}/B{b}/{cd}d =====')
        row('베이스라인(바스켓 없음)',d,sim(d,k1,b,cd,10**9,0))
        for cmin in [2,3,4,6]:
            for buf in [0.003,0.01]:
                row(f'바스켓 C{cmin}/버퍼{buf*100:.1f}%',d,sim(d,k1,b,cd,cmin,buf))

if __name__=='__main__':
    main()
