# -*- coding: utf-8 -*-
"""_240m 재생성 (fetch_duka_fill 키혼합 버그 복구용) — _60m(듀카 1h) 리샘플 + 2023-07+ 1m 오버레이.
usage: rebuild_4h.py SYM1,SYM2,...  (혹은 인자 없이 = _60m 있는 전 심볼)"""
import sys, os, glob, pickle
from datetime import datetime, timezone
from collections import Counter
ROOT = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(ROOT, 'data')
H = 3600*1000; BAR = 4*H; CUT = int(datetime(2023, 7, 1, tzinfo=timezone.utc).timestamp()*1000)
CRYPTO = {'BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','XAUTUSDT'}  # 크립토는 Bybit 1m이 원본 — 건드리지 않음

syms = (sys.argv[1].split(',') if len(sys.argv) > 1 else
        [os.path.basename(p).replace('_60m.pkl','') for p in glob.glob(os.path.join(DATA,'*_60m.pkl'))])
for S in syms:
    if S in CRYPTO: continue
    p60 = os.path.join(DATA, f'{S}_60m.pkl')
    if not os.path.exists(p60): print(f'{S}: _60m 없음 skip'); continue
    bars = pickle.load(open(p60,'rb'))
    agg = {}
    for b in bars:
        kk = int((b['start']//BAR)*BAR)
        if kk not in agg: agg[kk] = dict(start=kk, open=b['open'], high=b['high'], low=b['low'], close=b['close'], _src='h1')
        else:
            a = agg[kk]
            if a.get('_src') == 'h1':
                a['high']=max(a['high'],b['high']); a['low']=min(a['low'],b['low']); a['close']=b['close']
    p1 = os.path.join(DATA, f'{S}_1m.pkl')
    if os.path.exists(p1):
        for b in (x for x in pickle.load(open(p1,'rb')) if x['start'] >= CUT):
            kk = int((b['start']//BAR)*BAR)
            a = agg.get(kk)
            if a is None or a.get('_src') != 'm1':
                agg[kk] = dict(start=kk, open=b['open'], high=b['high'], low=b['low'], close=b['close'], _src='m1')
            else:
                a['high']=max(a['high'],b['high']); a['low']=min(a['low'],b['low']); a['close']=b['close']
    h4 = [{k:v for k,v in agg[kk].items() if k != '_src'} for kk in sorted(agg)]
    pickle.dump(h4, open(os.path.join(DATA, f'{S}_240m.pkl'),'wb'))
    c = Counter(x['start'] for x in h4)
    dup = sum(1 for v in c.values() if v > 1)
    print(f'{S:7} 4h {len(h4):,}봉 중복 {dup} 단조 {all(h4[i]["start"]<h4[i+1]["start"] for i in range(len(h4)-1))}')
