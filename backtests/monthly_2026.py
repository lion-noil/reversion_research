"""올해(2026) 월별 성적 — 크립토 계좌 $10k / MT5 계좌 $10k 따로.
확정 S1(추세 롱·숏)+S2(역추세 롱·숏) 4방향, 추매 없음(=확정 베이스), 중첩(라이브) 엔진.
사이징: 한 랏=자산 5%, 최대 200랏, 청산 시 실현(portfolio_sim 관례). fee 0.11%, 14일보유.
사용: python backtests/monthly_2026.py"""
from __future__ import annotations
import sys, os, heapq
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt_s1_quarters_mt5 import prep
from bt_avgdown import H_MS, S1_TREND_LONG, S2_REV_LONG, S2_REV_SHORT
from bt_avgdown_overlap import sim_overlap

S1_TREND_SHORT = {"SOLUSDT": (3.4, -2.0, 1.5), "XAUUSD": (3.2, 0.2, 1.0),
                  "XAGUSD": (2.65, 1.2, 2.0), "XAUTUSDT": (3.5, 0.8, 0.75), "WTI": (3.2, 1.8, 1.0)}
GROUPS = [("trend_long", S1_TREND_LONG), ("trend_short", S1_TREND_SHORT),
          ("rev_long", S2_REV_LONG), ("rev_short", S2_REV_SHORT)]
CRYPTO = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "XAUTUSDT"}
INIT, LOT_FRAC, MAX_LOTS = 10_000.0, 0.05, 200
MONTHS = [(2026, m) for m in range(1, 7)]


def m_end(y, m):
    return int(datetime(y + (m // 12), (m % 12) + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
def y0(): return int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def run_account(events):
    """events: (entry_ts, exit_ts, ret). 2026 진입만. → 월말자산곡선·월별집계."""
    ev = sorted([e for e in events if e[0] >= y0()], key=lambda x: x[0])
    equity = INIT; heap = []; open_cnt = 0; peak = 0
    curve = []; taken = skipped = 0
    by_month_n = {}
    span1 = max((e[1] for e in ev), default=y0())

    def realize(t):
        nonlocal equity, open_cnt
        while heap and heap[0][0] <= t:
            ex_ts, ret, notion = heapq.heappop(heap)
            equity += notion * ret; open_cnt -= 1
            curve.append((ex_ts, equity))

    for en, ex, ret in ev:
        realize(en)
        key = datetime.fromtimestamp(en / 1000, tz=timezone.utc).strftime("%Y-%m")
        by_month_n[key] = by_month_n.get(key, 0) + 1
        if open_cnt < MAX_LOTS and equity > 0:
            heapq.heappush(heap, (ex, ret, LOT_FRAC * equity)); open_cnt += 1; taken += 1
            peak = max(peak, open_cnt)
        else:
            skipped += 1
    realize(span1 + 1)

    # 월말 자산
    def eq_at(t):
        e = INIT
        for ts_, q in curve:
            if ts_ <= t: e = q
            else: break
        return e
    rows = []; prev = INIT
    for (y, m) in MONTHS:
        me = m_end(y, m)
        eq = eq_at(me)
        n = by_month_n.get(f"{y}-{m:02d}", 0)
        rows.append((f"{y}-{m:02d}", n, prev, eq, eq / prev - 1 if prev > 0 else 0))
        prev = eq
    return rows, taken, skipped, peak, equity


def main():
    cache = {}
    events = {"crypto": [], "mt5": []}
    for kind, params in GROUPS:
        for s, (k1, b, cdh) in params.items():
            d = cache.get(s) or prep(s); cache[s] = d
            if d is None: continue
            for t in sim_overlap(d, kind, k1, b, int(cdh * H_MS), mode="parallel"):
                venue = "crypto" if s in CRYPTO else "mt5"
                events[venue].append((t["entry_ts"], t["exit_ts"], t["ret"]))

    for venue, lbl in [("crypto", "크립토 계좌 (Bybit: BTC·ETH·SOL·XRP·XAUT USDT)"),
                       ("mt5", "MT5 계좌 (지수·금속·WTI·BTCUSD·ETHUSD)")]:
        rows, taken, skipped, peak, final = run_account(events[venue])
        print("\n" + "=" * 64)
        print(f"  {lbl}   시작 $10,000")
        print("=" * 64)
        print(f"{'월':>9}{'거래':>6}{'월초$':>11}{'월말$':>11}{'수익%':>9}")
        for mk, n, beg, end, r in rows:
            print(f"{mk:>9}{n:>6}{beg:>11,.0f}{end:>11,.0f}{r*100:>+8.1f}%")
        print("-" * 64)
        print(f"  올해누적: ${INIT:,.0f} → ${final:,.0f}  ({final/INIT:.2f}배, {final/INIT*100-100:+.1f}%)  "
              f"| 체결 {taken}랏 스킵 {skipped} 최대동시 {peak}랏")
    print("\n  ※ 추매 없음(확정 베이스)·중첩·랏=자산5%·최대200랏·fee0.11%. 데이터~6/20(크립토)·~6/25(MT5).")
    print("  ※ 6월은 미완(데이터 컷오프)+미청산분 마지막봉 평가 포함. 단순 실현곡선이라 장중위험 과소평가.")


if __name__ == "__main__":
    main()
