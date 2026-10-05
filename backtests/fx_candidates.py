# -*- coding: utf-8 -*-
"""환율 확장 후보 압축 (2026-10-05) — '2021-07 이전 데이터로 골라도 통과'하고 '전체 기간으로 골라도 통과'하는 (심볼·봉·방향)만 남긴다.
그 교집합 셀을 2021-07 이전 파라미터 그대로 최근 5y(표본 밖)에 얹어 포트폴리오 수치를 낸다.
usage: py -3 backtests/fx_candidates.py  → results/FX_CANDIDATES.md
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx_portfolio_sim as P  # noqa: E402
from fx_cross_screen import MAJORS, MODENM  # noqa: E402

H = 3600 * 1000
CUT = P.T5Y
full = json.load(open(os.path.join(HERE, 'results', 'fx_cross_cells.json'), encoding='utf-8'))
oos = json.load(open(os.path.join(HERE, 'results', 'fx_cross_cells_oos.json'), encoding='utf-8'))
fk = {(c['sym'], c['tf'], c['mode']): c for c in full}
live, live_keys = P.live_trades()
rows = []
for c in oos:
    k = (c['sym'], c['tf'], c['mode'])
    if k not in fk or k in live_keys:
        continue
    o = [t[2] * 100 for t in c['trades'] if t[0] >= CUT]
    i = [t[2] * 100 for t in c['trades'] if t[0] < CUT]
    f = fk[k]
    rows.append(dict(c=c, key=k, n_is=len(i), e_is=float(np.mean(i)), n_oos=len(o), e_oos=float(np.mean(o)) if o else 0.0,
                     sum_oos=float(np.sum(o)), same=(c['win'], c['k'], c['b']) == (f['win'], f['k'], f['b']), f=f))
rows.sort(key=lambda r: -r['sum_oos'])
L = [f"# 환율 확장 후보 — 두 선정 구간 모두 통과한 신규 셀 {len(rows)}개 (생성 {datetime.now():%Y-%m-%d %H:%M})\n",
     "파라미터는 2021-07 이전 데이터로 고른 값. '이후 5y'는 그 파라미터를 손대지 않고 돌린 표본 밖 성적.\n",
     "| 심볼 | 봉 | 방향 | 창 | K | B | 쿨다운 | ~2021 n·건당 | 이후 5y n·건당 | 5y 합계 | 전체기간 선정값과 같은가 |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    c = r['c']
    L.append(f"| {c['sym']} | {c['tf']} | {MODENM[c['mode']]} | {c['win']} | {c['k']} | {c['b']} | {c['cd_ms']//H}h | {r['n_is']}·{r['e_is']:+.2f}% | "
             f"{r['n_oos']}·{r['e_oos']:+.2f}% | {r['sum_oos']:+.1f}%p | {'같음' if r['same'] else f'다름(전체: 창{r[chr(102)][chr(119)+chr(105)+chr(110)]} K{r[chr(102)][chr(107)]} B{r[chr(102)][chr(98)]})'} |")
keys = {r['key'] for r in rows}
cross_d = lambda c: (c['sym'], c['tf'], c['mode']) in keys and c['sym'] not in MAJORS and c['tf'] == '일봉'   # noqa: E731
anyk = lambda c: (c['sym'], c['tf'], c['mode']) in keys   # noqa: E731
scen = {"A. 지금(라이브 환율 셀)": live,
        "G. + 교집합 크로스 일봉 셀": live + P.cand_trades(oos, cross_d),
        "H. + 교집합 전부(메이저·4h 포함)": live + P.cand_trades(oos, anyk)}
L += ["\n## 포트폴리오 (최근 5y, 신규 셀은 표본 밖)\n",
      "| 구성 | 월 진입 | 월평균 수익 | 월 중앙 | 2%↑ 달 | 음수 달 | 실현 최대낙폭 | 평가낙폭 월95%/최악 | 일최대 99%/최악 | 최대 동시 노출 |",
      "|---|---|---|---|---|---|---|---|---|---|"]
base = None
for name, tr in scen.items():
    s, m = P.row(name, tr)
    L.append(s)
    if base is None:
        base = m
    else:
        k = base['mtm95'] / m['mtm95']
        L.append(P.row(f"　└ 같은 평소 낙폭으로 맞춤(사이징 ×{k:.2f})", tr, k)[0])
md = "\n".join(L)
open(os.path.join(HERE, 'results', 'FX_CANDIDATES.md'), 'w', encoding='utf-8').write(md)
print(md)
