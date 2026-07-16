"""현재 라이브 배포(tradingBot trade_config, 2026-07-04) Bybit 구성 그대로 3년 통합 시뮬.

구성 = 실배포와 1:1:
- 1분 S1(추세, 병렬·추매無)  : TREND_BYBIT 5심볼(방향별), win=10080, 보유14d
- 1분 S2(역추세, ontop 추매1회/게임): REV_BYBIT, win=10080, 보유14d
- 일봉 S3(추세)·S4(역추세, 추매無): CRYPTOD_*, win=90, 보유15d, maxc캡, 쿨다운 일단위
- 사이징 = 라이브 동일: 랏 notional = 진입시점 자산 2% (bal×0.04%×레버50)
- 총 동시 랏캡 500 (= max_effective_leverage 10배 ÷ 2%)
- fee 0.11% 왕복, 시작 $5,000, 기간 = 1분 데이터 3년 스팬(일봉도 동일 창으로 클립)

출력: 콘솔 요약 + live_bybit_sim_result.json
"""
import sys, os, json, heapq
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bt_s1_quarters_mt5 import prep  # 1분 prep (win=10080)
from bt_avgdown_overlap import sim_overlap
from bt_avgdown import _targets, SIDE, ZSIGN
from reversion_calibrator import _load_cache

FEE = 0.0011
DAY = 86400 * 1000
INIT = 5_000.0
LOT_FRAC = 0.02          # 라이브: bal × 0.04/100 × 50 = 2%
MAX_LOTS = 500           # max_effective_leverage 10 ÷ 2%
H = 1.0                  # 시간단위(시)

# ── 배포 파라미터 (bots/trade_config.py 그대로) ──────────────────────────────
# 1분 S1 추세 (kind, k1, b, cd_h)
S1_1M = [
    ("BTCUSDT", "trend_long", 3.2, -2.0, 3.0),
    ("ETHUSDT", "trend_long", 2.35, 1.2, 2.5),
    ("ETHUSDT", "trend_short", 3.45, -1.8, 3.0),
    ("SOLUSDT", "trend_long", 3.4, -2.0, 3.0),
    ("SOLUSDT", "trend_short", 3.4, -2.0, 1.5),
    ("XRPUSDT", "trend_long", 2.55, -0.4, 3.0),
    ("XAUTUSDT", "trend_long", 3.25, -2.0, 2.0),
    ("XAUTUSDT", "trend_short", 3.5, 0.8, 0.75),
]
# 1분 S2 역추세 (ontop 추매 1회)
S2_1M = [
    ("BTCUSDT", "rev_long", 3.3, -2.0, 3.0),
    ("BTCUSDT", "rev_short", 4.6, -0.4, 0.5),
    ("ETHUSDT", "rev_long", 3.15, -1.2, 2.0),
    ("ETHUSDT", "rev_short", 3.3, -1.2, 3.0),
    ("SOLUSDT", "rev_long", 3.3, 1.8, 3.0),
    ("XRPUSDT", "rev_long", 3.5, -0.4, 2.25),
    ("XRPUSDT", "rev_short", 5.0, -2.0, 0.5),
    ("XAUTUSDT", "rev_long", 2.75, -1.8, 3.0),
]
# 일봉 S3 추세 (kind, k1, b, cd_d, maxc)
S3_D = [
    ("BTCUSDT", "trend_long", 2.5, -2.0, 2, 8),
    ("BTCUSDT", "trend_short", 2.4, 0.2, 1, 10),
    ("ETHUSDT", "trend_long", 2.6, -1.4, 1, 15),
    ("ETHUSDT", "trend_short", 1.6, -0.8, 5, 3),
    ("SOLUSDT", "trend_long", 2.9, -3.0, 1, 13),
    ("SOLUSDT", "trend_short", 1.7, 0.2, 2, 8),
    ("XRPUSDT", "trend_long", 3.1, -3.0, 1, 13),
    ("XRPUSDT", "trend_short", 1.6, -1.4, 5, 3),
]
# 일봉 S4 역추세 (추매無)
S4_D = [
    ("BTCUSDT", "rev_long", 1.3, 1.2, 7, 3),
    ("SOLUSDT", "rev_long", 2.3, 1.6, 1, 14),
    ("XRPUSDT", "rev_long", 2.4, 1.4, 1, 8),
    ("XRPUSDT", "rev_short", 1.0, -0.4, 10, 2),
]

DWIN = 90
DHOLD_MS = 15 * DAY


def prep_daily(sym):
    m = _load_cache(sym, "D")
    if not m:
        return None
    C = np.array([x["close"] for x in m], float)
    Hh = np.array([x["high"] for x in m], float)
    L = np.array([x["low"] for x in m], float)
    ts = np.array([x["start"] for x in m], np.int64)
    n = len(C)
    if n <= DWIN:
        return None
    cs = np.concatenate([[0.0], np.cumsum(C)])
    cs2 = np.concatenate([[0.0], np.cumsum(C * C)])
    w = DWIN
    ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    s = cs[w:] - cs[:-w]; s2 = cs2[w:] - cs2[:-w]
    mm = s / w; var = np.maximum(s2 / w - mm * mm, 0.0)
    ma[w - 1:] = mm; sd[w - 1:] = np.sqrt(var)
    z = np.where(sd > 0, (C - ma) / sd, np.nan)
    return dict(C=C, H=Hh, L=L, ts=ts, ma=ma, sd=sd, z=z, n=n)


def daily_trades(d, kind, k1, b, cd_days, maxc):
    """일봉 병렬(추매無) + maxc 동시보유캡. (entry_ts, exit_ts, net_ret) 리스트."""
    C, Hh, L, ts, ma, sd, z, n = d["C"], d["H"], d["L"], d["ts"], d["ma"], d["sd"], d["z"], d["n"]
    side = SIDE[kind]; zs = ZSIGN[kind]
    cand = np.flatnonzero(z >= k1) if zs > 0 else np.flatnonzero(z <= -k1)
    cand = cand[cand >= DWIN]
    cd = cd_days * DAY
    out = []; last = None
    open_exits = []  # heap of exit_ts
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd:
            continue
        while open_exits and open_exits[0] <= ts[ci]:
            heapq.heappop(open_exits)
        if len(open_exits) >= maxc:      # 라이브: maxc 초과 스킵(쿨다운 미소비)
            continue
        price = C[ci]
        tp, sl, ok = _targets(kind, ma[ci], sd[ci], price, price, b)
        if not ok:
            continue
        hi = int(np.searchsorted(ts, ts[ci] + DHOLD_MS, "right"))
        if hi <= ci + 1:
            continue
        if side > 0:
            SLh = L[ci + 1:hi] <= sl; TPh = Hh[ci + 1:hi] >= tp
        else:
            SLh = Hh[ci + 1:hi] >= sl; TPh = L[ci + 1:hi] <= tp
        hit = SLh | TPh
        if hit.any():
            j = int(np.argmax(hit)); ex = ci + 1 + j
            px = sl if SLh[j] else tp
        else:
            ex = hi - 1; px = C[ex]
        ret = (px / price - 1.0 if side > 0 else (price - px) / price) - FEE
        out.append((int(ts[ci]), int(ts[ex]), float(ret)))
        heapq.heappush(open_exits, int(ts[ex]))
        last = ts[ci]
    return out


def main():
    cache1m, cacheD = {}, {}
    events = []  # (entry_ts, exit_ts, ret, book, sym)

    # ── 1분 S1 (병렬) ──
    for sym, kind, k1, b, cdh in S1_1M:
        d = cache1m.get(sym) or prep(sym); cache1m[sym] = d
        for p in sim_overlap(d, kind, k1, b, int(cdh * 3600 * 1000), mode="parallel"):
            for lt, lr in p["legs"]:
                events.append((lt, p["exit_ts"], lr, "s1", sym))

    # ── 1분 S2 (ontop 추매 1회) ──
    for sym, kind, k1, b, cdh in S2_1M:
        d = cache1m.get(sym) or prep(sym); cache1m[sym] = d
        for p in sim_overlap(d, kind, k1, b, int(cdh * 3600 * 1000), mode="ontop", max_adds=1):
            for lt, lr in p["legs"]:
                events.append((lt, p["exit_ts"], lr, "s2", sym))

    span0 = min(e[0] for e in events); span1 = max(e[1] for e in events)

    # ── 일봉 S3/S4 (1분 스팬으로 클립) ──
    for book, table in (("s3", S3_D), ("s4", S4_D)):
        for sym, kind, k1, b, cdd, mc in table:
            d = cacheD.get(sym) or prep_daily(sym); cacheD[sym] = d
            for en, ex, ret in daily_trades(d, kind, k1, b, cdd, mc):
                if en >= span0:
                    events.append((en, ex, ret, book, sym))

    print(f"총 레그(랏) 이벤트 {len(events):,}건")

    # ── 포트폴리오 ──
    events.sort(key=lambda x: x[0])
    equity = INIT; heap = []; open_cnt = 0; peak_lots = 0; skipped = taken = 0
    curve = []; by_book = {}; by_sym = {}
    ruin = False

    def realize(t):
        nonlocal equity, open_cnt, ruin
        while heap and heap[0][0] <= t:
            ex_ts, ret, notional, bk, sy = heapq.heappop(heap)
            pnl = notional * ret
            equity += pnl
            by_book[bk] = by_book.get(bk, 0.0) + pnl
            by_sym[sy] = by_sym.get(sy, 0.0) + pnl
            open_cnt -= 1
            curve.append((ex_ts, equity))
            if equity <= 0:
                ruin = True

    for en, ex, ret, bk, sy in events:
        realize(en)
        if open_cnt < MAX_LOTS and equity > 0:
            heapq.heappush(heap, (ex, ret, LOT_FRAC * equity, bk, sy))
            open_cnt += 1; taken += 1; peak_lots = max(peak_lots, open_cnt)
        else:
            skipped += 1
    realize(span1 + 1)

    eq = np.array([e for _, e in curve], float)
    ts_c = np.array([t for t, _ in curve], np.int64)
    pk = np.maximum.accumulate(eq); dd = (eq - pk) / pk
    mdd = float(dd.min()); mdd_at = int(ts_c[int(np.argmin(dd))])
    yrs = (span1 - span0) / (365.25 * DAY)
    cagr = (equity / INIT) ** (1 / yrs) - 1 if equity > 0 else -1.0

    def dts(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    # 분기별 스냅샷
    qsnap = []; seen = set()
    for t, e in curve:
        dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
        key = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        if key not in seen:
            seen.add(key); qsnap.append((key, round(e, 0)))
    qsnap.append(("최종", round(equity, 0)))

    print("=" * 64)
    print(f"기간 {dts(span0)} ~ {dts(span1)} ({yrs:.2f}년) | 시작 ${INIT:,.0f} | 랏=자산2% | 총랏캡{MAX_LOTS}")
    print(f"최종 ${equity:,.0f}  ({equity/INIT:.2f}배, CAGR {cagr*100:+.1f}%)")
    print(f"실현 MDD {mdd*100:.1f}% @ {dts(mdd_at)} | 체결 {taken:,}랏 / 스킵 {skipped:,} / 최대동시 {peak_lots}랏 | 파산 {'예' if ruin else '아니오'}")
    print("\n북별 기여:")
    for bk in sorted(by_book):
        print(f"  {bk}: ${by_book[bk]:+,.0f}")
    print("심볼별 기여:")
    for sy in sorted(by_sym, key=lambda s: -by_sym[s]):
        print(f"  {sy:9}: ${by_sym[sy]:+,.0f}")
    print("\n분기별 자산:")
    for k, v in qsnap:
        print(f"  {k}: ${v:,.0f}")

    json.dump(dict(init=INIT, final=round(equity, 2), mult=round(equity / INIT, 3),
                   cagr=round(cagr, 4), mdd=round(mdd, 4), mdd_at=dts(mdd_at), yrs=round(yrs, 2),
                   taken=taken, skipped=skipped, peak_lots=peak_lots, ruin=ruin,
                   span=[dts(span0), dts(span1)], lot_frac=LOT_FRAC, max_lots=MAX_LOTS,
                   by_book={k: round(v, 2) for k, v in by_book.items()},
                   by_sym={k: round(v, 2) for k, v in by_sym.items()},
                   quarterly=qsnap),
              open(os.path.join(os.path.dirname(__file__), "live_bybit_sim_result.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n→ live_bybit_sim_result.json")


if __name__ == "__main__":
    main()
