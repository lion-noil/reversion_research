# -*- coding: utf-8 -*-
"""환율이 월 2%에 못 미치는 이유 분해 (2026-10-05) — 유니버스별·환율 심볼/책별: 셀 수, 월 진입, 건당 수익률, 승률, 건당 변동, 월 기여.
usage: py -3 backtests/fx_edge_breakdown.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_vs_backtest import DAY, FX, LIVE, T5Y, run_cell, sizing_pct  # noqa: E402

rows = []   # (universe, sym, book, per_month, mean%, std%, win, sizing%)
last = T5Y
cells = []
for book, subs in LIVE.items():
    for sub in subs:
        strat, iv, hold_ms = sub['strategy'], sub['candle_interval'], int(sub['max_hold_sec']) * 1000
        for sym, sides in (sub['params'] or {}).items():
            for side, p in sides.items():
                if float(p.get('k1', 0) or 0) >= 99:
                    continue
                tr = [t for t in run_cell(strat, sym, side, p, iv, hold_ms) if t[0] >= T5Y]
                if not tr:
                    continue
                last = max(last, tr[-1][1])
                uni = "크립토" if book in ("s11", "s22", "s33") else ("환율" if sym in FX else "비환율")
                bk = {"1": "1분", "240": "4h", "D": "일봉"}.get(str(iv), str(iv))
                cells.append((uni, sym, bk, np.array([t[2] for t in tr]) * 100, sizing_pct(book, strat, sym)))
years = (last - T5Y) / (365.25 * DAY)


def line(label, group):
    n = sum(len(a) for *_, a, _ in group)
    allr = np.concatenate([a for *_, a, _ in group])
    contrib = sum(a.sum() * sz / 100 for *_, a, sz in group) / years / 12
    hold = ""
    print(f"{label} | 셀 {len(group)} | 월 {n/years/12:.1f}건 | 건당 {allr.mean():+.2f}% (σ {allr.std():.2f}) | 승률 {100*(allr>0).mean():.0f}% | "
          f"건당÷σ {allr.mean()/allr.std():.2f} | 월 기여 {contrib:+.2f}%")


print("== 유니버스 ==")
for u in ("크립토", "비환율", "환율"):
    line(u, [c for c in cells if c[0] == u])
print("\n== 환율: 책별 ==")
for b in ("1분", "4h", "일봉"):
    g = [c for c in cells if c[0] == "환율" and c[2] == b]
    if g:
        line(b, g)
print("\n== 환율: 심볼별 ==")
for s in sorted({c[1] for c in cells if c[0] == "환율"}):
    line(s, [c for c in cells if c[0] == "환율" and c[1] == s])
print("\n== 비환율: 책별(비교) ==")
for b in ("1분", "4h", "일봉"):
    g = [c for c in cells if c[0] == "비환율" and c[2] == b]
    if g:
        line(b, g)
