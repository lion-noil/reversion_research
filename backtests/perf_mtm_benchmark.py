# -*- coding: utf-8 -*-
"""성적표 평가손 기준(시뮬) (2026-10-05) — 유니버스별 5y 시간단위 MTM 평가액 곡선에서
'평가낙폭(월중 평가액 고점 대비)'과 '일최대 낙폭(하루 시작 평가액 대비 최저)'의 분포를 낸다.

엔진 = prop_sim.load_trades/simulate_equity (universe_report 셀·MTM 마크: 보유 포지션을 그 시간 최악가로 평가 — 보수적).
사이징 = 라이브: U1 ×1(5%/2%) · U2 ×3.5(3.5%/7%) · U3 ×5.75(5.75%/28.75%, 10-05 ×2.5).  하루 경계 = 07:00 KST(성적표 equity_hist 와 동일).
usage: py -3 backtests/perf_mtm_benchmark.py   → PERF_BENCHMARK.json(results·News_scrap/app) 의 segments[*].mtm 갱신
       (perf_benchmark.py 가 끝에 자동 호출)
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UNIVERSES = {"BYBIT": ("U1", 1.0), "MT5-비환율": ("U2", 3.5), "MT5-환율": ("U3", 5.75)}   # U3: 2026-10-05 ×2.5 (구 2.3)
PATHS = [os.path.join(HERE, "results", "PERF_BENCHMARK.json"),
         os.path.join(HERE, "..", "..", "News_scrap", "app", "perf_benchmark.json")]
DAY_SHIFT_H = 2      # 07:00 KST = 22:00 UTC → (hour + 2) // 24 가 하루 번호
CRYPTO = {"BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}


def mtm_stats(uni: str, scale: float) -> dict:
    _argv = sys.argv
    import prop_sim
    trades, mark0, t5y = prop_sim.load_trades(uni, scale)
    sys.argv = _argv

    def mark(sym, hour, side, ep):      # 마크 이상치 차단(prop_sim_ftmo 와 동일 — 폴백 바 튐으로 생기는 가짜 낙폭)
        m = mark0(sym, hour, side, ep)
        return ep if abs(m / ep - 1.0) > (0.6 if sym in CRYPTO else 0.25) else m

    seq = prop_sim.simulate_equity(trades, mark, t5y)
    hours = np.array([h for h, _, _ in seq])
    mtm = np.array([m for _, _, m in seq])
    days = (hours + DAY_SHIFT_H) // 24
    day_dd, month_dd = [], {}
    peak_all, mdd_all = mtm[0], 0.0
    cur_day, day_open, day_low = None, None, None
    m_key, m_peak = None, None
    for h, d, v in zip(hours, days, mtm):
        peak_all = max(peak_all, v)
        mdd_all = max(mdd_all, (peak_all - v) / peak_all * 100)
        if d != cur_day:
            if cur_day is not None:
                day_dd.append(max(0.0, (day_open - day_low) / day_open * 100))
            cur_day, day_open, day_low = d, v, v
        day_low = min(day_low, v)
        ym = (datetime.fromtimestamp(int(h) * 3600, timezone.utc) + timedelta(hours=9 - 7)).strftime("%Y-%m")
        if ym != m_key:
            m_key, m_peak = ym, v
            month_dd[ym] = 0.0
        m_peak = max(m_peak, v)
        month_dd[ym] = max(month_dd[ym], (m_peak - v) / m_peak * 100)
    dd = np.array(day_dd)
    md = np.array(list(month_dd.values())[1:-1])      # 첫·마지막 부분월 제외
    r = lambda x: round(float(x), 2)   # noqa: E731
    return {"basis": f"{uni} 라이브 사이징(×{scale}) 5y 시간단위 MTM(보유 포지션을 그 시간 최악가로 평가), 하루 07:00 KST",
            "mdd": r(mdd_all),
            "month_dd": {"mean": r(md.mean()), "p50": r(np.median(md)), "p95": r(np.percentile(md, 95)), "max": r(md.max())},
            "day_dd": {"mean": r(dd.mean()), "p95": r(np.percentile(dd, 95)), "p99": r(np.percentile(dd, 99)), "max": r(dd.max())}}


def main():
    out = {seg: mtm_stats(uni, sc) for seg, (uni, sc) in UNIVERSES.items()}
    for p in PATHS:
        j = json.load(open(p, encoding="utf-8"))
        for seg, st in out.items():
            j["segments"].setdefault(seg, {})["mtm"] = st
        open(p, "w", encoding="utf-8").write(json.dumps(j, ensure_ascii=False, indent=1))
        print("[saved]", os.path.abspath(p))
    print(json.dumps(out, ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main()
