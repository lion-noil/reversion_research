# -*- coding: utf-8 -*-
"""환율 크로스 장기 히스토리 — HFM MT5 터미널(C:\\MT5_FX, 읽기 전용)에서 D1·H4 를 받아 캐시(_D10m.pkl / _240m.pkl).
bond_screen 과 같은 관례(MT5 서버시각 그대로, {start,open,high,low,close}). 1분봉은 브로커 보관이 수개월뿐이라 받지 않는다.
usage: <executor-a2 python> fetch_mt5_hist_fx.py [SYM,SYM...]   (기본 = 크로스 21종)
"""
import sys
from datetime import datetime, timezone

import MetaTrader5 as mt5

from reversion_calibrator import _save_cache

CROSSES = ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
           "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURNZD",
           "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
           "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "CADCHF"]
syms = sys.argv[1].split(",") if len(sys.argv) > 1 else CROSSES
assert mt5.initialize(path=r"C:\MT5_FX\terminal64.exe", portable=True), mt5.last_error()
t0 = datetime(2010, 1, 1, tzinfo=timezone.utc)
t1 = datetime.now(timezone.utc)
for s in syms:
    info = mt5.symbol_info(s)
    if info is None:
        print(f"{s:8} 심볼 없음"); continue
    if not info.visible:
        mt5.symbol_select(s, True)
    for tf, itv in ((mt5.TIMEFRAME_D1, "D10"), (mt5.TIMEFRAME_H4, "240")):
        r = None
        for _ in range(4):          # 첫 요청은 서버에서 내려받는 중이라 비어 올 수 있다
            r = mt5.copy_rates_range(s, tf, t0, t1)
            if r is not None and len(r) > 100:
                break
            import time; time.sleep(3)
        if r is None or len(r) == 0:
            print(f"{s:8} {itv:4} 실패 {mt5.last_error()}"); continue
        cs = [{"start": int(x["time"]) * 1000, "open": float(x["open"]), "high": float(x["high"]),
               "low": float(x["low"]), "close": float(x["close"])} for x in r if x["high"] > x["low"]]
        _save_cache(s, itv, cs)
        d = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
        print(f"{s:8} {itv:4} {len(cs):>6}봉 {d(cs[0]['start'])}~{d(cs[-1]['start'])}  spread {info.spread}pt ({info.spread*info.point/cs[-1]['close']*100:.4f}%)")
mt5.shutdown()
