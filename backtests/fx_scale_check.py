# -*- coding: utf-8 -*-
"""환율 유니버스(U3) 진입비 배수별 낙폭·수익 (2026-10-05) — 비환율 수준 낙폭까지 올리면 어떻게 되나.
엔진 = perf_mtm_benchmark 와 동일(5y 시간단위 MTM, 하루 07:00 KST). 배수 1 = 라이브(1분·4h 2.3%, 일봉 11.5%).
usage: py -3 backtests/fx_scale_check.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prop_sim  # noqa: E402

base, mark, t5y = prop_sim.load_trades("U3", 2.3)
print("배수 | 1분·4h/일봉 진입비 | 월평균 수익 | 월 중앙 | 실현 최대낙폭 | 평가낙폭 월95%/최악 | 일최대 99%/최악 | 최대 동시 노출(계좌 대비)")
for mult in (1, 2, 2.5, 3, 3.5, 4, 5):
    tr = [t[:7] + (t[7] * mult,) + tuple(t[8:]) for t in base]
    seq = prop_sim.simulate_equity(tr, mark, t5y)
    hours = np.array([h for h, _, _ in seq]); real = np.array([r for _, r, _ in seq]); mtm = np.array([m for _, _, m in seq])
    # 최대 동시 노출
    ev = sorted([(t[0], t[7]) for t in tr] + [(t[1], -t[7]) for t in tr])
    cur = mx = 0.0
    for _, f in ev:
        cur += f; mx = max(mx, cur)
    # 월별
    ym = np.array([(datetime.fromtimestamp(int(h) * 3600, timezone.utc) + timedelta(hours=2)).strftime("%Y-%m") for h in hours])
    days = (hours + 2) // 24
    mret, mdd_m = [], []
    for k in sorted(set(ym))[1:-1]:
        i = np.where(ym == k)[0]
        mret.append((real[i[-1]] / real[i[0]] - 1) * 100)
        pk = np.maximum.accumulate(mtm[i]); mdd_m.append(((pk - mtm[i]) / pk).max() * 100)
    ddd = []
    for d in np.unique(days):
        i = np.where(days == d)[0]
        ddd.append(max(0.0, (mtm[i[0]] - mtm[i].min()) / mtm[i[0]] * 100))
    pk = np.maximum.accumulate(real); rdd = ((pk - real) / pk).max() * 100
    print(f"×{mult} | {2.3*mult:.1f}% / {11.5*mult:.1f}% | {np.mean(mret):+.2f}% | {np.median(mret):+.2f}% | {rdd:.2f}% | "
          f"{np.percentile(mdd_m, 95):.2f}% / {max(mdd_m):.2f}% | {np.percentile(ddd, 99):.2f}% / {max(ddd):.2f}% | {mx*100:.0f}%")
