# -*- coding: utf-8 -*-
"""tradingBot trade_config → results/live_cells.json 재생성 (universe_report.py 입력).
북 구조: s11/s22/s33(U1) + s11m/s22m(U2·U3 공유) + s33m(U2 일봉) + s33f(U3 일봉FX)."""
import sys, os, json

sys.path.insert(0, r"C:\Users\Hyeongeon\PycharmProjects\tradingBot")
from bots import trade_config as tc

BOOKS = {
    "s11":  [tc.make_s11_trend_config, tc.make_s11_rev_config, tc.make_s11_fade_config],
    "s22":  [tc.make_s22_trend_config, tc.make_s22_rev_config, tc.make_s22_ewz_config, tc.make_s22_fade_config],
    "s33":  [tc.make_crypto_daily_trend_config, tc.make_crypto_daily_rev_config],
    "s11m": [tc.make_s11_mt5_trend_config, tc.make_s11_mt5_rev_config, tc.make_s11_mt5_fade_config],
    "s22m": [tc.make_s22m_trend_config, tc.make_s22m_rev_config, tc.make_s22m_ewz_config,
             tc.make_s22m_fade_config, tc.make_s22m_sweep_config],
    # s33m = 비환율+FX 통합 (2026-07-21 fxd 흡수). s33f는 드레인(오픈 소진만)이라 제외 —
    #   포함하면 FX 셀이 s33m과 이중 카운트됨.
    "s33m": [tc.make_mt5_daily_trend_config, tc.make_mt5_daily_rev_config],
}

out = {}
ncell = 0
for book, factories in BOOKS.items():
    subs = []
    for f in factories:
        cfg = f()
        params = cfg.s1_params_by_symbol or {}
        ncell += sum(len(v) for v in params.values())
        subs.append({
            "strategy": cfg.strategy,
            "candle_interval": cfg.candle_interval,
            "max_hold_sec": int(cfg.s1_max_hold_sec),
            "params": params,
        })
    out[book] = subs

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "live_cells.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"OK {dst} — 셀 {ncell}개 (드레인 k1=99 포함)")
