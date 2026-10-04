# -*- coding: utf-8 -*-
"""성적표 기준값(시뮬) 생성 (2026-10-04) — universe_monthly 와 같은 엔진·라이브 셀·사이징으로 5y 를 돌려
성적표 구분(BYBIT / MT5-비환율 / MT5-환율 / MT5 합계)별 '이 정도면 정상' 기준을 JSON 으로 낸다.

모든 %는 성적표와 같은 정의 = 실현 손익 ÷ 계좌 에쿼티 단순합(복리·미실현 미반영):
  건당 기여% = 건당 수익률 × 사이징%,  일별/월별 = 청산일 기준 합산,  낙폭 = 일별 누적 곡선의 고점 대비 하락(%p).
usage: py -3 backtests/perf_benchmark.py   (위치: reversion_research)
  → results/PERF_BENCHMARK.json + News_scrap/app/perf_benchmark.json (perf_report.py 가 읽음)
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_vs_backtest import DAY, FX, LIVE, T5Y, run_cell, sizing_pct  # noqa: E402

OUT_APP = os.path.join(HERE, '..', '..', 'News_scrap', 'app', 'perf_benchmark.json')


def month_key(ms):
    d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return d.year * 12 + d.month - 1


def drawdown(daily):
    """일별 기여% 배열 → (최대낙폭, 일별 낙폭 깊이 95백분위, 최장 고점 미회복 일수)"""
    cum = np.cumsum(daily)
    peak = np.maximum.accumulate(np.maximum(cum, 0.0))
    dd = peak - cum
    longest = run = 0
    for x in dd:
        run = run + 1 if x > 1e-9 else 0
        longest = max(longest, run)
    return float(dd.max()), float(np.percentile(dd, 95)), int(longest)


def main():
    trades = defaultdict(list)      # 구분 → [(exit_ms, 기여%, 건당 수익률%)]
    cells = defaultdict(int)
    for book, subs in LIVE.items():
        bybit = book in ("s11", "s22", "s33")
        for sub in subs:
            strat, iv, hold_ms = sub['strategy'], sub['candle_interval'], int(sub['max_hold_sec']) * 1000
            for sym, sides in (sub['params'] or {}).items():
                for side, p in sides.items():
                    if float(p.get('k1', 0) or 0) >= 99:
                        continue
                    tr = [t for t in run_cell(strat, sym, side, p, iv, hold_ms) if t[0] >= T5Y]
                    if not tr:
                        continue
                    sz = sizing_pct(book, strat, sym) / 100
                    segs = ["BYBIT"] if bybit else ["MT5-환율" if sym in FX else "MT5-비환율", "MT5"]
                    for g in segs:
                        cells[g] += 1
                        trades[g].extend((ex, r * 100 * sz, r * 100) for _, ex, r in tr)

    last_exit = max(t[0] for v in trades.values() for t in v)
    m0, m1 = month_key(T5Y) + 1, month_key(last_exit) - 1          # 첫·마지막 부분월 제외
    d0, d1 = T5Y // DAY, last_exit // DAY
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "window": f"5y 2021-07-16~{datetime.fromtimestamp(last_exit / 1000, tz=timezone.utc):%Y-%m-%d}",
           "basis": "실현÷에쿼티 단순합(복리·미실현 미반영), 라이브 셀·FINAL_PARAMS 사이징, 봉마감 체결", "segments": {}}
    for g, tr in trades.items():
        contrib = np.array([t[1] for t in tr])
        raw = np.array([t[2] for t in tr])
        monthly = defaultdict(float)
        daily = np.zeros(d1 - d0 + 1)
        for ex, c, _ in tr:
            monthly[month_key(ex)] += c
            daily[ex // DAY - d0] += c
        ms = np.array([monthly.get(m, 0.0) for m in range(m0, m1 + 1)])
        mdd, dd95, uw = drawdown(daily)
        years = (last_exit - T5Y) / (365.25 * DAY)
        out["segments"][g] = {
            "cells": cells[g], "trades": len(tr),
            "trades_per_month": round(len(tr) / years / 12, 1),
            "trade_avg_eq_pct": round(float(contrib.mean()), 4),      # 건당 계좌 기여 %
            "trade_avg_pct": round(float(raw.mean()), 2),             # 건당 수익률 %(사이징 전)
            "win_rate": round(100 * float((raw > 0).mean()), 1),
            "month_mean": round(float(ms.mean()), 2), "month_median": round(float(np.median(ms)), 2),
            "month_neg_ratio": round(100 * float((ms < 0).mean()), 0), "month_worst": round(float(ms.min()), 2),
            "mdd": round(mdd, 2), "dd_p95": round(dd95, 2), "underwater_max_days": uw,
        }
    js = json.dumps(out, ensure_ascii=False, indent=1)
    for path in (os.path.join(HERE, 'results', 'PERF_BENCHMARK.json'), OUT_APP):
        open(path, 'w', encoding='utf-8').write(js)
        print(f"[saved] {os.path.abspath(path)}")
    print(js)


if __name__ == "__main__":
    main()
