# -*- coding: utf-8 -*-
"""KODEX200(069500) 5분봉 → QuantConnect 커스텀 데이터 CSV(연도별).

원천 ① reversion_research/data/069500_5m.pkl (크레온, 2021-07~2026-08-20, OHLCV, start=UTC ms)
     ② kis_trader/state/align3_bars.json (ALIGN3 실행기가 WS 체결로 만든 5분봉 종가, 2026-08-06~) — ①의 마지막 날짜 이후만 사용, O=H=L=C=종가, 거래량 0
출력  kospi/data/069500_5m_<YYYY>.csv  헤더 time,open,high,low,close,volume · time = 봉 시작시각 KST "yyyyMMdd HH:mm"
      kospi/data/manifest.json (연도별 행 수·기간)
usage: .venv-quant\Scripts\python platforms/quantconnect/kospi/export_kospi_qc.py"""
import os, sys, json, pickle
from datetime import datetime, timezone, timedelta
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "..", "..", "..", "data", "069500_5m.pkl")
BARS = os.path.join(HERE, "..", "..", "..", "..", "kis_trader", "state", "align3_bars.json")
OUT = os.path.join(HERE, "data")
KST = timezone(timedelta(hours=9))


def main():
    rows = OrderedDict()   # (date, hhmm) -> (o,h,l,c,v)
    for b in pickle.load(open(PKL, "rb")):
        t = datetime.fromtimestamp(b["start"] / 1000, tz=KST)
        rows[(t.strftime("%Y%m%d"), t.strftime("%H%M"))] = (b["open"], b["high"], b["low"], b["close"], int(b.get("volume") or 0))
    last_pkl_date = max(d for d, _ in rows)
    added = 0
    if os.path.exists(BARS):
        for d, hhmm, c in json.load(open(BARS, encoding="utf-8")):
            if d > last_pkl_date:
                rows[(d, hhmm)] = (float(c), float(c), float(c), float(c), 0)
                added += 1
    keys = sorted(rows)
    os.makedirs(OUT, exist_ok=True)
    by_year = {}
    for d, hhmm in keys:
        by_year.setdefault(d[:4], []).append((d, hhmm))
    manifest = {"symbol": "069500", "pkl_last_date": last_pkl_date, "appended_from_align3": added, "years": {}}
    for y, ks in sorted(by_year.items()):
        fn = os.path.join(OUT, f"069500_5m_{y}.csv")
        with open(fn, "w", encoding="utf-8", newline="\n") as f:
            f.write("time,open,high,low,close,volume\n")
            for d, hhmm in ks:
                o, h, l, c, v = rows[(d, hhmm)]
                f.write(f"{d} {hhmm[:2]}:{hhmm[2:]},{o:g},{h:g},{l:g},{c:g},{v}\n")
        manifest["years"][y] = {"rows": len(ks), "first": ks[0][0], "last": ks[-1][0], "bytes": os.path.getsize(fn)}
        print(y, len(ks), ks[0][0], ks[-1][0], f"{os.path.getsize(fn)//1024} KB")
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print("총", len(keys), "봉 · pkl 마지막", last_pkl_date, "· align3 추가", added)


if __name__ == "__main__":
    main()
