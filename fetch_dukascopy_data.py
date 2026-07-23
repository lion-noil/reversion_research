"""
Dukascopy 심볼 캔들 적재 — dukascopy-node(CLI)로 받아 data/ 로컬 캐시(pickle).

왜 Dukascopy인가:
  - mt5_server(HFM 브로커)는 **1분봉을 ~3~4개월만 보관** → 3년치 1m이 소스에 없음(fetch_mt5_data.py 참고).
  - Dukascopy는 2003~ 틱 아카이브 무료 제공 → 지수CFD·메탈·에너지·크립토 전부 3년 1m/일봉 커버.

⚠️ 소스 일관성 주의:
  - Dukascopy 호가는 HFM과 절대레벨/스프레드가 다름.
  - **한 심볼 시계열에서 mt5_server 캐시와 섞지 말 것**(이음매 가격점프 → σ-복귀 가짜신호).
    백테스트 히스토리는 전 구간 Dukascopy 단일 소스로 받는 걸 권장.

포맷:
  Bybit/MT5 fetch와 **동일 캐시 포맷**: {start,open,high,low,close} dict 리스트,
  파일 {SYM}_{1m|Dm}.pkl  → 기존 백테스트/캘리브레이터가 그대로 로드.

전제:
  - node + npx 설치돼 있어야 함(dukascopy-node를 npx로 on-demand 실행).
  - 첫 실행 시 npx가 dukascopy-node를 받아오므로 약간 지연될 수 있음.
    자주 쓸 거면 `npm i -g dukascopy-node` 후 환경변수 DUKASCOPY_CMD=dukascopy-node 권장.

사용:
  python fetch_dukascopy_data.py --all                       # 전 심볼 일봉3년 + 1분봉3년
  python fetch_dukascopy_data.py --daily --years 3           # 전 심볼 일봉만
  python fetch_dukascopy_data.py --m1                        # 전 심볼 1분봉만(3년)
  python fetch_dukascopy_data.py --symbols US100,BTCUSD --all
  python fetch_dukascopy_data.py --price ask --all           # bid(기본)/ask 선택
  python fetch_dukascopy_data.py --list                      # 캐시 현황
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone

from reversion_calibrator import _save_cache, _load_cache, CACHE_DIR

DAY_MS = 86_400_000

# ── 봇 정규(canonical) 심볼 → Dukascopy 인스트루먼트 ID ────────────────────────
# ID는 dukascopy-node 패키지 번들에서 직접 추출해 검증함.
# 라이브 MT5 심볼(fetch_mt5_data.py MT5_SYMBOLS)과 1:1 매핑.
SYMBOL_MAP: dict[str, str] = {
    "US100":  "usatechidxusd",   # 나스닥100 (USA Tech 100 index)
    "JP225":  "jpnidxjpy",       # 닛케이225 (Japan 225 index)
    "HK50":   "hkgidxhkd",       # 항셍 (Hong Kong index)
    "GER40":  "deuidxeur",       # DAX40 (Germany index)
    "UK100":  "gbridxgbp",       # FTSE100 (UK 100 index)
    "XAUUSD": "xauusd",          # 금
    "XAGUSD": "xagusd",          # 은
    "WTI":    "lightcmdusd",     # WTI 경질원유 (Light Crude)
    "BTCUSD": "btcusd",          # 비트코인
    "ETHUSD": "ethusd",          # 이더리움
    # ── 환율 메이저 7 (HFM 거래가능 · Dukascopy ID=소문자 페어명) ──
    "EURUSD": "eurusd",          # 유로/달러
    "GBPUSD": "gbpusd",          # 파운드/달러
    "USDJPY": "usdjpy",          # 달러/엔
    "AUDUSD": "audusd",          # 호주달러/달러
    "USDCAD": "usdcad",          # 달러/캐나다달러 (WTI 상관 주의)
    "USDCHF": "usdchf",          # 달러/스위스프랑 (EURUSD 역상관 주의)
    "NZDUSD": "nzdusd",          # 뉴질랜드달러/달러 (AUDUSD 고상관 주의)
    # ── 채권 선물 CFD (HFM Bonds 그룹, 2026-07-24 추가) ──
    "EUBUND": "bundtreur",       # 유로 분트(독일 10년) — HFM EUBUND.F
    "UKGILT": "ukgilttrgbp",     # 영국 길트 — HFM UKGILT.F
    "US10YR": "ustbondtrusd",    # ⚠️ Dukascopy는 US T-Bond(30년) — HFM US10YR.F(10년)의 프록시
}

# 기본 대상 = 라이브 MT5 10심볼 (fetch_mt5_data.py와 동일 집합)
DEFAULT_SYMBOLS = list(SYMBOL_MAP.keys())

# npx로 on-demand 실행(전역 설치 시 DUKASCOPY_CMD=dukascopy-node 로 덮어쓰기)
DUKASCOPY_CMD = os.getenv(
    "DUKASCOPY_CMD",
    "npx -y -p dukascopy-node@latest dukascopy-node",
)

# 타임프레임: 봇 interval("1"/"D") → dukascopy-node -t 값
TF = {"1": "m1", "D": "d1"}


def _run_dukascopy(instrument: str, tf: str, date_from: str, date_to: str,
                   price: str, out_dir: str, fname: str) -> str:
    """dukascopy-node 1회 실행 → 저장된 csv 파일 경로 반환.

    ⚠️ `-v`(volumes) 필수: flats(틱 없는 휴장 분)는 '0 거래량'으로 판별되는데,
       volumes를 안 받으면 flat 판별이 안 돼 --flats(기본 false) 필터가 무력화되고
       주말/야간 휴장이 평봉(O=H=L=C)으로 통째 패딩됨. -v를 줘야 실세션 봉만 남음.
    """
    cmd = (
        f'{DUKASCOPY_CMD} -i {instrument} -from {date_from} -to {date_to} '
        f'-t {tf} -p {price} -v -f csv -dir "{out_dir}" -fn {fname} '
        f'-bs 10 -bp 500 -r 5 -rp 800 -re'
    )
    # Windows/posix 모두 동작하도록 shell 실행. dukascopy-node는 진행률(√ 등 비ASCII)을
    # stdout에 찍으므로, cp949 로캘에서 깨지지 않게 utf-8/replace로 강제 디코드.
    proc = subprocess.run(cmd, shell=True, capture_output=True,
                          encoding="utf-8", errors="replace")
    path = os.path.join(out_dir, f"{fname}.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(f"출력 없음/빈 파일 (exit={proc.returncode})\n{tail}")
    return path


def _read_csv_into(path: str, acc: dict[int, dict]) -> int:
    """dukascopy csv(header: timestamp,open,high,low,close,volume) → acc[ms]={start,o,h,l,c}.
    여러 청크를 같은 acc에 누적하면 자동으로 중복(ms) 제거됨. 추가된 행 수 반환.

    ⚠️ 헤더만 있는 빈 CSV(>0바이트)는 파일크기 체크를 통과하므로, 여기서 추가행 0을
       반환→호출부가 '빈 청크'로 간주해 재시도하게 함(조용한 결손 방지)."""
    added = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)  # 헤더 스킵
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 5:
                continue
            try:
                ms = int(p[0])
                o, h, l, c = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            except ValueError:
                continue
            acc[ms] = {"start": ms, "open": o, "high": h, "low": l, "close": c}
            added += 1
    return added


def _chunks(date_from: str, date_to: str, chunk_days: int):
    """[date_from, date_to)를 chunk_days 윈도로 분할. 마지막 윈도는 date_to('now' 가능) 그대로."""
    start = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = (datetime.now(timezone.utc) if date_to == "now"
           else datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    cur = start
    while cur < end:
        nxt = cur + timedelta(days=chunk_days)
        if nxt >= end:
            yield cur.strftime("%Y-%m-%d"), date_to
            return
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt


def _report(sym: str, interval: str, cs: list[dict]) -> None:
    tag = {"1": "1m", "D": "일봉"}[interval]
    if cs:
        a = datetime.fromtimestamp(cs[0]["start"] / 1000, tz=timezone.utc)
        b = datetime.fromtimestamp(cs[-1]["start"] / 1000, tz=timezone.utc)
        print(f"  [{sym} {tag}] {len(cs):,}봉  {a:%Y-%m-%d}~{b:%Y-%m-%d}  → {sym}_{interval}m.pkl")
    else:
        print(f"  [{sym} {tag}] 데이터 없음")


def fetch_one(sym: str, interval: str, date_from: str, date_to: str, price: str) -> int:
    inst = SYMBOL_MAP.get(sym.upper())
    if not inst:
        print(f"  [{sym}] 매핑 없음 — SYMBOL_MAP에 추가 필요. 스킵.")
        return 0
    acc: dict[int, dict] = {}
    with tempfile.TemporaryDirectory(prefix="dk_") as tmp:
        if interval == "1":
            # 3년 1m을 한 방에 받으면 거대 파일/중단으로 실패 → 180일 청크로 누적.
            # 청크 단위 재시도(일시적 rate-limit/빈응답 대응). 끝까지 실패한 윈도는
            # 결손으로 명시(조용히 누락시키지 않음).
            failed: list[tuple[str, str]] = []
            for j, (a, b) in enumerate(_chunks(date_from, date_to, 180)):
                last_err = None
                for attempt in range(4):
                    try:
                        path = _run_dukascopy(inst, TF[interval], a, b, price, tmp, f"out{j}_{attempt}")
                        n = _read_csv_into(path, acc)
                        if n == 0:
                            raise RuntimeError("0행(헤더만) 응답 — 빈 청크")
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(4 * (attempt + 1))
                if last_err is not None:
                    failed.append((a, b))
                    print(f"    [{sym} 1m] 청크 {a}~{b} 3회 실패 — 건너뜀: {last_err}")
            if failed:
                print(f"    [{sym} 1m] ⚠️ 결손 윈도 {len(failed)}개: {failed}")
        else:
            path = _run_dukascopy(inst, TF[interval], date_from, date_to, price, tmp, "out")
            _read_csv_into(path, acc)
    cs = [acc[k] for k in sorted(acc)]
    _save_cache(sym.upper(), interval, cs)
    _report(sym.upper(), interval, cs)
    return len(cs)


def _resample_daily(cs1m: list[dict]) -> list[dict]:
    """1분봉(start 오름차순) → 일봉(UTC 자정 경계).
    O=첫 봉 open, H=max, L=min, C=마지막 봉 close, start=그날 UTC 자정 ms."""
    buckets: dict[int, dict] = {}
    for c in cs1m:
        day = (c["start"] // DAY_MS) * DAY_MS
        b = buckets.get(day)
        if b is None:
            buckets[day] = {"start": day, "open": c["open"], "high": c["high"],
                            "low": c["low"], "close": c["close"]}
        else:
            if c["high"] > b["high"]:
                b["high"] = c["high"]
            if c["low"] < b["low"]:
                b["low"] = c["low"]
            b["close"] = c["close"]
    return [buckets[k] for k in sorted(buckets)]


def build_daily(sym: str) -> int:
    """일봉을 d1 엔드포인트(일부 지수에서 'Unknown error'로 불안정) 대신
    이미 받은 1분봉 캐시에서 리샘플 생성 → 1m과 소스/세션 100% 일치."""
    cs1 = _load_cache(sym.upper(), "1")
    if not cs1:
        print(f"  [{sym} 일봉] 1m 캐시 없음 — 먼저 --m1/--all 필요. 스킵.")
        return 0
    d = _resample_daily(cs1)
    _save_cache(sym.upper(), "D", d)
    _report(sym.upper(), "D", d)
    return len(d)


def list_cache() -> None:
    if not os.path.isdir(CACHE_DIR):
        print("(캐시 없음)"); return
    print(f"캐시: {CACHE_DIR}")
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.endswith(".pkl"):
            mb = os.path.getsize(os.path.join(CACHE_DIR, f)) / 1e6
            print(f"  {f:<22} {mb:>7.1f}MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="콤마구분(기본: Dukascopy 매핑 10심볼)")
    ap.add_argument("--daily", action="store_true", help="일봉 적재")
    ap.add_argument("--m1", action="store_true", help="1분봉 적재")
    ap.add_argument("--all", action="store_true", help="일봉+1분봉")
    ap.add_argument("--years", type=float, default=3.0, help="적재 기간(년, 기본 3)")
    ap.add_argument("--price", choices=["bid", "ask"], default="bid", help="호가 종류(기본 bid)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_cache(); return

    syms = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            if args.symbols else DEFAULT_SYMBOLS)
    do_daily = args.daily or args.all
    do_m1 = args.m1 or args.all
    if not (do_daily or do_m1):
        ap.print_help(); return

    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(days=int(args.years * 365))).strftime("%Y-%m-%d")
    date_to = "now"
    print(f"소스=Dukascopy  심볼={syms}  기간={date_from}~now  price={args.price}")

    # 1m 먼저(일봉은 1m에서 리샘플하므로 순서 중요).
    if do_m1:
        print("\n=== 1분봉 적재 (3년은 다운로드가 큼: 심볼당 수분 소요 가능) ===")
        for i, s in enumerate(syms):
            print(f" [{i+1}/{len(syms)}] {s} 1분봉...")
            try:
                fetch_one(s, "1", date_from, date_to, args.price)
            except Exception as e:
                print(f"  [{s} 1m] 실패 {e}")

    if do_daily:
        print("\n=== 일봉 생성 (1분봉에서 리샘플) ===")
        for i, s in enumerate(syms):
            print(f" [{i+1}/{len(syms)}] {s} 일봉...")
            try:
                build_daily(s)
            except Exception as e:
                print(f"  [{s} 일봉] 실패 {e}")

    print("\n완료.")
    list_cache()


if __name__ == "__main__":
    main()
