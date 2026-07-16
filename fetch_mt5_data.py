"""
MT5 심볼 캔들 적재 — mt5_server(api.hyeongeonnoil.com)에서 받아 data/ 로컬 캐시(pickle).

Bybit fetch_data.py와 **동일 캐시 포맷**: {start,open,high,low,close} dict 리스트, 파일 {SYM}_{1m|Dm}.pkl.
→ 기존 백테스트/캘리브레이터가 그대로 로드 가능.

⚠️ 데이터 가용성 제약 (HFM 브로커):
  - 일봉(D): ~3년 (지수는 브로커가 가진 만큼, 예 US100 ~2023-09부터; 크립토CFD는 full 3년)
  - 1분봉(1): **~3~4개월만 보관** → 3년치 1m은 소스에 없음. 가용 전체를 받는다.

소스: GET {BASE}/v5/market/candles/with-gaps  (canonical 심볼 → 서버가 broker로 매핑)
  - range 모드(start_ms+end_ms): limit 제한 없이 구간 반환 → 청크 다운로드
  - 1분봉 응답은 휴장 갭을 None OHLC로 패딩 → 여기서 None 봉 제거(실봉만 저장)

사용:
  python fetch_mt5_data.py --all                         # 전 심볼 일봉3년 + 1분봉(가용전체)
  python fetch_mt5_data.py --daily --days 1095           # 전 심볼 일봉만
  python fetch_mt5_data.py --m1                           # 전 심볼 1분봉만
  python fetch_mt5_data.py --symbols US100,BTCUSD --all
  python fetch_mt5_data.py --list                         # 캐시 현황
"""
from __future__ import annotations
import argparse
import os
import time
import requests
from datetime import datetime, timezone

from reversion_calibrator import _save_cache, _load_cache, CACHE_DIR

BASE = os.getenv("MT5_PRICE_REST_URL", "https://api.hyeongeonnoil.com").rstrip("/")


def _load_api_key() -> str:
    # 우선순위: 환경변수 → 리포 루트 .env(gitignore) → 이웃 tradingBot/.env (동일 키 보유)
    key = os.getenv("MT5_API_KEY", "")
    here = os.path.dirname(os.path.abspath(__file__))
    for env_path in (os.path.join(here, ".env"),
                     os.path.join(here, "..", "tradingBot", ".env")):
        if key:
            break
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MT5_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
    if not key:
        raise SystemExit("MT5_API_KEY 필요: 환경변수 또는 .env(MT5_API_KEY=...) 로 지정")
    return key


API_KEY = _load_api_key()
HDR = {"X-API-Key": API_KEY}
EP = "/v5/market/candles/with-gaps"
DAY = 86_400_000

# 현재 라이브 MT5 심볼 (tradingBot .env MT5_SYMBOLS와 동일)
MT5_SYMBOLS = ["US100", "JP225", "XAUUSD", "WTI", "XAGUSD",
               "BTCUSD", "ETHUSD", "HK50", "GER40", "UK100"]


def _get(params, retries=5, log=print):
    backoff = 1.0
    for a in range(retries):
        try:
            r = requests.get(BASE + EP, params=params, headers=HDR, timeout=90)
            r.raise_for_status()
            j = r.json()
            rc = j.get("retCode")
            if rc not in (0, None):
                raise RuntimeError(f"retCode={rc} {j.get('retMsg')}")
            return (j.get("result") or {}).get("list") or []
        except Exception as e:
            if a == retries - 1:
                raise
            log(f"    [retry {a+1}/{retries}] {e} - wait {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    return []


def _norm(rows, lo, hi):
    """[[ms,o,h,l,c,vol]] → [{start,o,h,l,c}] (None·범위밖 제거, lo<=ms<hi)."""
    out = []
    for c in rows:
        if not c or len(c) < 5:
            continue
        ms = int(c[0])
        if ms < lo or ms >= hi:
            continue
        if c[1] is None or c[4] is None:
            continue
        out.append({"start": ms, "open": float(c[1]), "high": float(c[2]),
                    "low": float(c[3]), "close": float(c[4])})
    return out


def _dedup_sort(cs):
    seen, out = set(), []
    for c in sorted(cs, key=lambda x: x["start"]):
        if c["start"] in seen:
            continue
        seen.add(c["start"])
        out.append(c)
    return out


def _report(sym, interval, cs):
    tag = {"1": "1m", "D": "일봉"}[interval]
    if cs:
        a = datetime.fromtimestamp(cs[0]["start"] / 1000, tz=timezone.utc)
        b = datetime.fromtimestamp(cs[-1]["start"] / 1000, tz=timezone.utc)
        print(f"  [{sym} {tag}] {len(cs):,}봉  {a:%Y-%m-%d}~{b:%Y-%m-%d}  → {sym}_{interval}m.pkl")
    else:
        print(f"  [{sym} {tag}] 데이터 없음")


def fetch_daily(sym, days, sleep_sec):
    now = int(time.time() * 1000)
    start = now - days * DAY
    rows = _get({"symbol": sym, "interval": "D", "start_ms": start, "end_ms": now})
    cs = _dedup_sort(_norm(rows, start, now + DAY))
    _save_cache(sym, "D", cs)
    _report(sym, "D", cs)
    time.sleep(sleep_sec)
    return len(cs)


def fetch_m1(sym, max_days, chunk_days, sleep_sec, empty_stop=3):
    """now에서 과거로 chunk_days씩 거슬러 받다가, 연속 empty_stop회 빈 청크면 중단(브로커 1m 컷오프)."""
    now = int(time.time() * 1000)
    floor = now - max_days * DAY
    chunk = chunk_days * DAY
    end = now
    empties = 0
    acc = {}
    while end > floor and empties < empty_stop:
        start = max(floor, end - chunk)
        try:
            rows = _get({"symbol": sym, "interval": "1", "start_ms": start, "end_ms": end})
        except Exception as e:
            print(f"    [{sym} 1m] 청크 실패 {e}")
            rows = []
        cs = _norm(rows, start, end)
        if cs:
            for c in cs:
                acc[c["start"]] = c
            empties = 0
        else:
            empties += 1
        end = start
        time.sleep(sleep_sec)
    out = _dedup_sort(list(acc.values()))
    _save_cache(sym, "1", out)
    _report(sym, "1", out)
    return len(out)


def list_cache():
    if not os.path.isdir(CACHE_DIR):
        print("(캐시 없음)"); return
    print(f"캐시: {CACHE_DIR}")
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.endswith(".pkl"):
            mb = os.path.getsize(os.path.join(CACHE_DIR, f)) / 1e6
            print(f"  {f:<22} {mb:>7.1f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="콤마구분(기본: 라이브 MT5 10심볼)")
    ap.add_argument("--daily", action="store_true", help="일봉 적재")
    ap.add_argument("--m1", action="store_true", help="1분봉 적재(가용 전체)")
    ap.add_argument("--all", action="store_true", help="일봉+1분봉")
    ap.add_argument("--days", type=int, default=1095, help="일봉 기간(기본 3년)")
    ap.add_argument("--m1-max-days", type=int, default=240, help="1분봉 스캔 상한(자동 중단됨)")
    ap.add_argument("--m1-chunk-days", type=int, default=7, help="1분봉 청크 크기(일)")
    ap.add_argument("--sleep", type=float, default=0.4, help="요청 간 대기초(라이브 터미널 보호)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_cache(); return

    syms = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            if args.symbols else MT5_SYMBOLS)
    do_daily = args.daily or args.all
    do_m1 = args.m1 or args.all
    if not (do_daily or do_m1):
        ap.print_help(); return

    print(f"소스={BASE}  심볼={syms}")
    if do_daily:
        print(f"\n=== 일봉 {args.days}일 적재 ===")
        for i, s in enumerate(syms):
            print(f" [{i+1}/{len(syms)}] {s} 일봉...")
            try:
                fetch_daily(s, args.days, args.sleep)
            except Exception as e:
                print(f"  [{s} 일봉] 실패 {e}")
    if do_m1:
        print(f"\n=== 1분봉 적재 (가용 전체, 상한 {args.m1_max_days}일, {args.m1_chunk_days}일 청크) ===")
        for i, s in enumerate(syms):
            print(f" [{i+1}/{len(syms)}] {s} 1분봉...")
            try:
                fetch_m1(s, args.m1_max_days, args.m1_chunk_days, args.sleep)
            except Exception as e:
                print(f"  [{s} 1m] 실패 {e}")
    print("\n완료.")
    list_cache()


if __name__ == "__main__":
    main()
