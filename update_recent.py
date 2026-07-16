"""1분봉 캐시 증분 갱신 — 기존 3년 이력 보존하고 '최근 구간만' 받아 ts로 병합.
크립토(Bybit 공개 API) + MT5(Dukascopy, 기존 dukascopy 헬퍼 재사용).
사용: python update_recent.py            # 전 심볼 1m 캐시를 now까지 연장"""
from __future__ import annotations
import os, sys, time, tempfile
from datetime import datetime, timezone, timedelta
import requests
sys.path.insert(0, os.path.dirname(__file__))
from reversion_calibrator import _load_cache, _save_cache
from fetch_dukascopy_data import _run_dukascopy, _read_csv_into, SYMBOL_MAP, TF

CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "XAUTUSDT"]
MT5 = ["US100", "JP225", "HK50", "GER40", "UK100", "XAUUSD", "XAGUSD", "WTI", "BTCUSD", "ETHUSD"]


def _last(cache): return cache[-1]["start"] if cache else 0
def _dt(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _merge_save(sym, cache, new):
    m = {c["start"]: c for c in cache}
    for c in new:
        m[c["start"]] = c
    out = [m[k] for k in sorted(m)]
    _save_cache(sym, "1", out)
    return out


def update_crypto(sym):
    cache = _load_cache(sym, "1"); last = _last(cache)
    url = "https://api.bybit.com/v5/market/kline"; new = []; end = None
    while True:
        p = {"category": "linear", "symbol": sym, "interval": "1", "limit": 1000}
        if end:
            p["end"] = end
        r = requests.get(url, params=p, timeout=20).json()
        lst = (r.get("result") or {}).get("list") or []
        if not lst:
            break
        chunk = [{"start": int(c[0]), "open": float(c[1]), "high": float(c[2]),
                  "low": float(c[3]), "close": float(c[4])} for c in lst if int(c[0]) > last]
        new += chunk
        oldest = int(lst[-1][0])
        if oldest <= last or len(lst) < 1000:
            break
        end = oldest - 1; time.sleep(0.2)
    out = _merge_save(sym, cache, new)
    print(f"  [{sym:>9}] +{len(new):>4}봉  → 끝 {_dt(_last(out))}")
    return len(new)


def update_mt5(sym):
    cache = _load_cache(sym, "1"); last = _last(cache)
    frm = (datetime.fromtimestamp(last / 1000, tz=timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    # ⚠️ date_to는 "now" 금지(주말/미래 경계서 dukascopy 'Unknown error'). 오늘 명시날짜 사용.
    to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    acc = {}
    try:
        with tempfile.TemporaryDirectory(prefix="dk_upd_") as tmp:
            path = _run_dukascopy(SYMBOL_MAP[sym], TF["1"], frm, to, "bid", tmp, "upd")
            _read_csv_into(path, acc)
    except Exception as e:
        print(f"  [{sym:>9}] 실패: {str(e)[:80]}")
        return 0
    new = [acc[k] for k in sorted(acc) if k > last]
    out = _merge_save(sym, cache, new)
    print(f"  [{sym:>9}] +{len(new):>4}봉  → 끝 {_dt(_last(out))}")
    return len(new)


def main():
    print(f"증분 갱신 시작 (now={_dt(int(datetime.now(timezone.utc).timestamp()*1000))} UTC)")
    print("─ 크립토 (Bybit) ─")
    for s in CRYPTO:
        try:
            update_crypto(s)
        except Exception as e:
            print(f"  [{s:>9}] 실패: {str(e)[:80]}")
    print("─ MT5 (Dukascopy) ─")
    for s in MT5:
        update_mt5(s)
    print("완료.")


if __name__ == "__main__":
    main()
