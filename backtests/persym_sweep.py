"""촘촘 격자 스윕(v2 모델) — 심볼별 최적 조합 추출.
K1 2.5~3.5(0.25) × B -2~1(0.25) × CD 0.5~3h(0.5h). n 100~900 필터, 심볼별 상위 TOPN.
규칙: win=10080, TP=MA-Bσ, SL대칭, fee0.11%, 진입쿨다운·동시보유·최대보유7일.
출력: persym_sweep.json (심볼별 top), persym_sweep_all.csv (필터통과 전부).
"""
from __future__ import annotations
import sys, os, json, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from bt_s1_quarters_mt5 import prep, SYMS, WIN

if os.getenv('S1_SYMS'):                       # 심볼셋 오버라이드(예: 크립토)
    SYMS=[x.strip().upper() for x in os.getenv('S1_SYMS').split(',') if x.strip()]
TAG=os.getenv('S1_TAG','')                     # 출력 파일 접미사
DIR=os.getenv('S1_DIR','long').lower()         # long(z<=-K1) | short(z>=+K1)

FEE=float(os.getenv('FEE','0.0011')); H=3600*1000; DAY=86400*1000
MAXHOLD_D=float(os.getenv('S1_MAXHOLD_D','7')); MH=int(MAXHOLD_D*DAY)  # 최대보유(일), env로 변경
def _frange(lo,hi,st):                                 # lo..hi 포함, st 간격
    n=int(round((hi-lo)/st)); return [round(lo+st*i,2) for i in range(n+1)]
K1_MIN=float(os.getenv('S1_K1_MIN','2.0'))             # K1 격자 하한(env)
K1_MAX=float(os.getenv('S1_K1_MAX','3.5'))             # K1 격자 상한(env). 숏 고K는 5.0까지.
K1_STEP=float(os.getenv('S1_K1_STEP','0.05'))          # K1 간격(env)
K1S=_frange(K1_MIN,K1_MAX,K1_STEP)                     # 기본 2.0 .. 3.5 (0.05, 촘촘)
BS=[round(-2+0.2*i,2) for i in range(21)]             # -2 .. 2 (0.2)
CDS=[round(0.5+0.25*i,2) for i in range(11)]          # 0.5h .. 3h (0.25)
NMIN=int(os.getenv('S1_NMIN','100'))                   # 최소 거래수(env). 2.97yr·30/yr≈89.
NMAX=int(os.getenv('S1_NMAX','900'))
MAXC_MAX=15; TOPN=12                                   # 최대중복 15 미만만

def run(d,k1,b,cd_ms):
    C,Hh,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cand=np.flatnonzero(z>=k1) if DIR in ('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=WIN]
    tp_,sl_,top_,tom_=[],[],[],[]; last=None; iv=[]
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]
        hi=int(np.searchsorted(ts,ts[ci]+MH,'right'))
        if hi<=ci+1: continue
        if DIR=='short':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue              # 숏: TP는 진입가 아래여야
            pct=1-tp/price; sl=price*(1+pct)
            SLh=Hh[ci+1:hi]>=sl; TPh=L[ci+1:hi]<=tp; hit=SLh|TPh
            if hit.any():
                j=int(np.argmax(hit)); ex=ci+1+j
                if SLh[j]: sl_.append((price-sl)/price-FEE)   # 손절(고가 닿음) 우선
                else: tp_.append((price-tp)/price-FEE)
            else:
                ex=hi-1; r=(price-C[ex])/price-FEE; (top_ if r>0 else tom_).append(r)
        elif DIR=='trend':
            # 추세 숏: 진입 z<=-K1(롱과 동일 신호)에서 '숏'(하락지속 베팅).
            # pct는 롱의 목표거리(설정 보존)로 산출, TP/SL만 아래/위로 미러.
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; tpS=price*(1-pct); slS=price*(1+pct)
            SLh=Hh[ci+1:hi]>=slS; TPh=L[ci+1:hi]<=tpS; hit=SLh|TPh
            if hit.any():
                j=int(np.argmax(hit)); ex=ci+1+j
                if SLh[j]: sl_.append((price-slS)/price-FEE)   # 상승=손절(우선)
                else: tp_.append((price-tpS)/price-FEE)        # 하락지속=익절
            else:
                ex=hi-1; r=(price-C[ex])/price-FEE; (top_ if r>0 else tom_).append(r)
        elif DIR=='trendlong':
            # 추세 롱: 진입 z>=+K1(과열/상승추세)에서 '롱'(상승지속 베팅).
            # pct는 역추세숏 산식(진입가-(MA+Bσ))으로, TP/SL만 위/아래로 미러.
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tpL=price*(1+pct); slL=price*(1-pct)
            SLh=L[ci+1:hi]<=slL; TPh=Hh[ci+1:hi]>=tpL; hit=SLh|TPh
            if hit.any():
                j=int(np.argmax(hit)); ex=ci+1+j
                if SLh[j]: sl_.append(slL/price-1-FEE)   # 하락=손절(우선)
                else: tp_.append(tpL/price-1-FEE)        # 상승지속=익절
            else:
                ex=hi-1; r=C[ex]/price-1-FEE; (top_ if r>0 else tom_).append(r)
        else:
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue              # 롱: TP는 진입가 위여야
            pct=tp/price-1; sl=price*(1-pct)
            SLh=L[ci+1:hi]<=sl; TPh=Hh[ci+1:hi]>=tp; hit=SLh|TPh
            if hit.any():
                j=int(np.argmax(hit)); ex=ci+1+j
                if SLh[j]: sl_.append(sl/price-1-FEE)
                else: tp_.append(tp/price-1-FEE)
            else:
                ex=hi-1; r=C[ex]/price-1-FEE; (top_ if r>0 else tom_).append(r)
        iv.append((int(ts[ci]),int(ts[ex])))   # (진입,청산) → 동시보유 계산용
        last=ts[ci]
    # 최대 동시보유(중복) 개수: 동시각 진입 전에 청산 처리(-1 우선)
    ev=[]
    for en,ex in iv: ev.append((en,1)); ev.append((ex,-1))
    ev.sort(key=lambda x:(x[0],x[1]))
    cur=mx=0
    for _,dd in ev: cur+=dd; mx=cur if cur>mx else mx
    return tp_,sl_,top_,tom_,mx

def main():
    t0=time.time(); per={}; allrows=[]
    print(f"[cfg] DIR={DIR} TAG='{TAG}' K1={K1S[0]}..{K1S[-1]}/{K1_STEP}({len(K1S)}) "
          f"B{BS[0]}..{BS[-1]} CD{CDS[0]}..{CDS[-1]}h n[{NMIN},{NMAX}] maxhold{MAXHOLD_D:g}d "
          f"SYMS={','.join(SYMS)}",flush=True)
    for s in SYMS:
        d=prep(s)
        if d is None: print(f'{s} 스킵',flush=True); continue
        weeks=(int(d['ts'][-1])-int(d['ts'][WIN]))/(7*86400e3)
        res=[]
        for k1 in K1S:
            for b in BS:
                for c in CDS:
                    tp_,sl_,top_,tom_,maxc=run(d,k1,b,int(c*H))
                    n=len(tp_)+len(sl_)+len(top_)+len(tom_)
                    if not (NMIN<=n<=NMAX): continue
                    if maxc>=MAXC_MAX: continue        # 최대중복 10 미만만
                    r=np.array(tp_+sl_+top_+tom_)
                    wins=int((r>0).sum()); pf=r[r>0].sum()/(-r[r<0].sum()) if (r<0).any() else 99.0
                    def m(x): return round(float(np.mean(x))*100,2) if x else None
                    row=dict(sym=s,k1=k1,b=b,cd=(f'{int(c*60)}m' if c<1 else f'{c:g}h'),
                        n=n,wk=round(n/weeks,2),maxc=maxc,wr=round(100*wins/n),exp=round(float(r.mean())*100,3),
                        pf=round(float(pf),2),tp=len(tp_),tpa=m(tp_),sl=len(sl_),sla=m(sl_),
                        top=len(top_),topa=m(top_),tom=len(tom_),toma=m(tom_))
                    res.append(row); allrows.append(row)
        res.sort(key=lambda x:(-x['exp'],-x['pf']))
        per[s]=res[:TOPN]
        best=res[0] if res else None
        print(f"{s}: {len(res)}조합 통과"+(f" | best K1={best['k1']} B={best['b']} {best['cd']} "
              f"n{best['n']} exp{best['exp']:+.3f}% PF{best['pf']}" if best else ""),flush=True)
    json.dump(per,open(f'persym_sweep{TAG}.json','w'),ensure_ascii=False)
    if allrows:
        ks=list(allrows[0].keys())
        with open(f'persym_sweep{TAG}_all.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=ks); w.writeheader(); w.writerows(allrows)
    print(f"\n완료 {time.time()-t0:.0f}s · {DIR} · 최대보유 {MAXHOLD_D:g}일 · maxc<{MAXC_MAX} · 통과 {len(allrows)}행 · persym_sweep{TAG}.json",flush=True)

if __name__=='__main__':
    main()
