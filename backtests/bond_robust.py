# -*- coding: utf-8 -*-
"""UKGILT 통과 셀 이웃격자 견고성 (s22_robust 방식: 각 축 ±1스텝 → 이웃 n/중앙값/탈락수).
대상: ① 4h z역추숏 w60봉(10d) K3.0 B-3 cd12h  ② D z역추숏 w150d K2.0 B-3 cd1d"""
import sys, os
import numpy as np
from itertools import product
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bond_screen import zma, zgen, arrs, evaluate  # 동일 모델 재사용

H = 3600 * 1000
DAY = 86400 * 1000
HOLD = 15 * DAY


def neighbors(base, grid):
    """base 파라미터의 각 축 ±1스텝 조합(자기 자신 제외)."""
    axes = []
    for kk, v in base.items():
        g = grid[kk]
        i = g.index(v)
        axes.append([g[j] for j in (i - 1, i, i + 1) if 0 <= j < len(g)])
    return [dict(zip(base.keys(), c)) for c in product(*axes) if dict(zip(base.keys(), c)) != base]


def run_cell(sym, itv, mode, win, k, b, cd_ms):
    C, Hh, L, ts = arrs(sym, itv)
    yrs = (int(ts[-1]) - int(ts[0])) / (365.25 * DAY)
    z, ma, sd = zma(C, win)
    tr = zgen(C, Hh, L, ts, win, k, b, mode, cd_ms, ma, sd, z, HOLD)
    if not tr:
        return None, 0
    a = np.array([r for _, r in tr])
    return dict(n=len(tr), exp=a.mean() * 100,
                passed=evaluate(tr, yrs) is not None), len(tr)


CASES = [
    ("UKGILT", '240', 'rs', "4h z역추숏",
     dict(win=60, k=3.0, b=-3, cd=12),
     dict(win=[30, 60, 90, 120, 180], k=[2.0, 2.5, 3.0, 3.5, 4.0], b=[-3, -2, -1, 0, 1], cd=[12, 24, 48]),
     H),
    ("UKGILT", 'D10', 'rs', "D z역추숏",
     dict(win=150, k=2.0, b=-3, cd=1),
     dict(win=[60, 90, 150, 200], k=[1.6, 1.8, 2.0, 2.3, 2.6, 3.0], b=[-3, -2, -1, -0.4, 0, 0.4, 1, 2], cd=[1, 2, 3, 5, 7]),
     DAY),
]

for sym, itv, mode, label, base, grid, cd_unit in CASES:
    sel, _ = run_cell(sym, itv, mode, base['win'], base['k'], base['b'], base['cd'] * cd_unit)
    nb = []
    for p in neighbors(base, grid):
        r, _ = run_cell(sym, itv, mode, p['win'], p['k'], p['b'], p['cd'] * cd_unit)
        nb.append(r)
    exps = [r['exp'] for r in nb if r and r['n'] >= 10]
    passed = sum(1 for r in nb if r and r.get('passed'))
    med = float(np.median(exps)) if exps else float('nan')
    print(f"{sym} {label} {base}")
    print(f"  선택셀: exp{sel['exp']:+.2f}% n{sel['n']}")
    print(f"  이웃 {len(nb)}개: 유효 {len(exps)} · 필터통과 {passed} · 기대 중앙 {med:+.2f}% · "
          f"최소 {min(exps):+.2f}% · 최대 {max(exps):+.2f}%" if exps else "  이웃 표본 없음")
    spike = sel['exp'] > 2 * med if exps and med > 0 else True
    print(f"  판정: {'⚠️ 스파이크 의심(이웃 대비 돌출)' if spike else '✅ 이웃과 연속적'}\n")
