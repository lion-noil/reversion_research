# -*- coding: utf-8 -*-
"""일봉 17심볼 × 4전략(S1추세롱/숏, S2역추세롱/숏) 최적값 + 분기별 성적 → JSON.
동시보유 모델(daily_sweep), MA90·10년·연≥GPY_MIN거래 중 기대값 최대.
각 최적조합의 분기별 n·승률·PF·기대 산출. 뷰어=daily_quarterly.html."""
import sys, os, json
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from daily_sweep import prep, run, WIN, DAY, MAXHOLD, ITV, K1S, BS, CDD, SYMS, MODES, GPY_MIN, GPY_MAX, NMIN
FEE=float(os.getenv('FEE','0.0011'))
MAXC_CAP=int(os.getenv('MAXC_CAP','10'))   # 최대중복(동시보유) 상한

def gen(d,mode,k1,b,cd_ms):
    C,H,L,ts,ma,sd,z,n=d['C'],d['H'],d['L'],d['ts'],d['ma'],d['sd'],d['z'],d['n']
    cand=np.flatnonzero(z>=k1) if mode in('short','trendlong') else np.flatnonzero(z<=-k1)
    cand=cand[cand>=WIN]; out=[]; last=None
    for ci in cand:
        if ci+1>=n: break
        if last is not None and (ts[ci]-last)<cd_ms: continue
        price=C[ci]; hi=int(np.searchsorted(ts,ts[ci]+MAXHOLD,'right'))
        if hi<=ci+1: continue
        if mode=='trendlong':
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; tx=price*(1+pct); sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=H[ci+1:hi]>=tx; hit=SL|TP
            r=((sx/price-1-FEE) if (hit.any() and SL[int(np.argmax(hit))]) else (tx/price-1-FEE)) if hit.any() else C[hi-1]/price-1-FEE
        elif mode=='trend':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; tx=price*(1-pct); sx=price*(1+pct)
            SL=H[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tx; hit=SL|TP
            r=(((price-sx)/price-FEE) if (hit.any() and SL[int(np.argmax(hit))]) else ((price-tx)/price-FEE)) if hit.any() else (price-C[hi-1])/price-FEE
        elif mode=='long':
            tp=ma[ci]-b*sd[ci]
            if tp<=price: continue
            pct=tp/price-1; sx=price*(1-pct)
            SL=L[ci+1:hi]<=sx; TP=H[ci+1:hi]>=tp; hit=SL|TP
            r=((sx/price-1-FEE) if (hit.any() and SL[int(np.argmax(hit))]) else (tp/price-1-FEE)) if hit.any() else C[hi-1]/price-1-FEE
        else:
            tp=ma[ci]+b*sd[ci]
            if tp>=price: continue
            pct=1-tp/price; sx=price*(1+pct)
            SL=H[ci+1:hi]>=sx; TP=L[ci+1:hi]<=tp; hit=SL|TP
            r=(((price-sx)/price-FEE) if (hit.any() and SL[int(np.argmax(hit))]) else ((price-tp)/price-FEE)) if hit.any() else (price-C[hi-1])/price-FEE
        out.append((int(ts[ci]),float(r))); last=ts[ci]
    return out

def qof(ms):
    t=datetime.fromtimestamp(ms/1000,tz=timezone.utc); return f'{t.year}Q{(t.month-1)//3+1}'

def st(rets):
    a=np.array(rets,float); m=len(a)
    if m==0: return None
    w=a>0; pf=a[w].sum()/(-a[~w].sum()) if (~w).any() else 99.0
    return dict(n=m,wr=round(100*float(w.mean())),exp=round(float(a.mean())*100,2),
                pf=round(float(pf),2) if np.isfinite(pf) else 99.0,cum=round((np.prod(1+a)-1)*100,1))

def main():
    out={}
    for s in SYMS:
        d=prep(s)
        if d is None: continue
        yrs=(int(d['ts'][-1])-int(d['ts'][WIN]))/(365.25*DAY)
        out[s]={}
        for strat,dn,mode in MODES:
            top=None
            for k1 in K1S:
                for b in BS:
                    for cdd in CDD:
                        rets,mx=run(d,mode,k1,b,cdd*DAY)
                        if len(rets)<NMIN: continue
                        if mx>MAXC_CAP: continue
                        gpy=len(rets)/yrs
                        if not (GPY_MIN<=gpy<=GPY_MAX): continue
                        e=float(np.mean(rets))
                        a=np.array(rets); pf=a[a>0].sum()/(-a[a<0].sum()) if (a<0).any() else 99
                        key=(round(e,5),pf)
                        if top is None or key>top[0]: top=(key,k1,b,cdd,mx,gpy)
            label=f'{strat}_{dn}'
            if top is None: out[s][label]=None; continue
            _,k1,b,cdd,mx,gpy=top
            tr=gen(d,mode,k1,b,cdd*DAY)
            byq=defaultdict(list)
            for en,r in tr: byq[qof(en)].append(r)
            quarters=[dict(q=q,**st(byq[q])) for q in sorted(byq)]
            ov=st([r for _,r in tr]); ov['maxc']=mx; ov['gpy']=round(gpy,1)
            out[s][label]=dict(k1=k1,b=b,cd=cdd,**ov,quarters=quarters)
        print(f'{s} done',flush=True)
    fn=os.getenv('OUT','daily_quarterly.json')
    json.dump(out,open(os.path.join(os.path.dirname(__file__),fn),'w'),ensure_ascii=False)
    print(f'→ {fn}')

if __name__=='__main__':
    main()
