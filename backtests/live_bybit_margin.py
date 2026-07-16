"""live_bybit_sim 구성의 MTM 증거금 분석 (일봉 고/저 장중최악 — daily_portfolio 방식).

각 랏의 (진입가, 방향, notional)을 추적, 매 일봉마다 오픈 랏들의 장중 최악 평가손 합산:
  MTM자산 = 실현자산 + Σ worst_unrealized
  증거금율 = MTM자산 / 총오픈노셔널  (핸드오프 청산기준: <20% = 강제청산)
출력: 최저 증거금율 시점/값, MTM 최대낙폭, 상위 위험일.
"""
import sys, os, heapq
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bt_s1_quarters_mt5 import prep
from bt_avgdown_overlap import sim_overlap
from bt_avgdown import SIDE
from live_bybit_sim import (S1_1M, S2_1M, S3_D, S4_D, prep_daily, daily_trades,
                            INIT, LOT_FRAC, MAX_LOTS, DAY)


def px_at(d, ts_ms):
    i = int(np.searchsorted(d["ts"], ts_ms))
    i = min(max(i, 0), d["n"] - 1)
    return float(d["C"][i])


def build_events():
    """(entry_ts, exit_ts, ret, side, sym, entry_px) 전 레그."""
    cache1m, cacheD = {}, {}
    ev = []
    for sym, kind, k1, b, cdh in S1_1M:
        d = cache1m.get(sym) or prep(sym); cache1m[sym] = d
        for p in sim_overlap(d, kind, k1, b, int(cdh * 3600 * 1000), mode="parallel"):
            for lt, lr in p["legs"]:
                ev.append((lt, p["exit_ts"], lr, SIDE[kind], sym, px_at(d, lt)))
    for sym, kind, k1, b, cdh in S2_1M:
        d = cache1m.get(sym) or prep(sym); cache1m[sym] = d
        for p in sim_overlap(d, kind, k1, b, int(cdh * 3600 * 1000), mode="ontop", max_adds=1):
            for lt, lr in p["legs"]:
                ev.append((lt, p["exit_ts"], lr, SIDE[kind], sym, px_at(d, lt)))
    span0 = min(e[0] for e in ev)
    for table in (S3_D, S4_D):
        for sym, kind, k1, b, cdd, mc in table:
            d = cacheD.get(sym) or prep_daily(sym); cacheD[sym] = d
            for en, ex, ret in daily_trades(d, kind, k1, b, cdd, mc):
                if en >= span0:
                    ev.append((en, ex, ret, SIDE[kind], sym, px_at(d, en)))
    return ev, cacheD


def main():
    events, cacheD = build_events()
    events.sort(key=lambda x: x[0])
    span0 = events[0][0]; span1 = max(e[1] for e in events)

    # 심볼별 일봉 H/L 맵 (day_ts -> (H, L))
    dHL = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "XAUTUSDT"):
        d = cacheD.get(sym) or prep_daily(sym)
        dHL[sym] = dict(zip(d["ts"].tolist(), zip(d["H"].tolist(), d["L"].tolist())))
    days = sorted(t for t in dHL["BTCUSDT"] if span0 <= t <= span1 + DAY)

    equity = INIT
    heap = []          # (exit_ts, ret, notional)
    open_lots = []     # dict(exit_ts, side, sym, px, notional)
    ei = 0
    worst = dict(ratio=9e9, ts=0, eq=0.0, expo=0.0, n=0)
    mtm_min_curve = []  # (day, mtm_equity)
    risky = []          # (ratio, day, mtm_eq, expo, nlots)

    def realize(t):
        nonlocal equity
        while heap and heap[0][0] <= t:
            _, ret, notional = heapq.heappop(heap)
            equity += notional * ret

    for dts_ in days:
        # 이 날 시작 전까지의 진입/청산 반영
        while ei < len(events) and events[ei][0] < dts_:
            en, ex, ret, side, sym, px = events[ei]; ei += 1
            realize(en)
            if len(open_lots) < MAX_LOTS and equity > 0:
                notional = LOT_FRAC * equity
                heapq.heappush(heap, (ex, ret, notional))
                open_lots.append(dict(exit_ts=ex, side=side, sym=sym, px=px, notional=notional))
        realize(dts_)
        open_lots = [l for l in open_lots if l["exit_ts"] > dts_]
        if not open_lots:
            continue
        hl_missing = False
        worst_unrl = 0.0; expo = 0.0
        for l in open_lots:
            hl = dHL[l["sym"]].get(dts_)
            if hl is None:
                hl_missing = True; break
            hi, lo = hl
            if l["side"] > 0:
                u = (lo / l["px"] - 1.0) * l["notional"]
            else:
                u = (l["px"] - hi) / l["px"] * l["notional"]
            worst_unrl += min(u, 0.0)   # 최악만(이익은 0으로 — 보수적)
            expo += l["notional"]
        if hl_missing or expo <= 0:
            continue
        mtm_eq = equity + worst_unrl
        ratio = mtm_eq / expo
        mtm_min_curve.append((dts_, mtm_eq))
        risky.append((ratio, dts_, mtm_eq, expo, len(open_lots)))
        if ratio < worst["ratio"]:
            worst = dict(ratio=ratio, ts=dts_, eq=mtm_eq, expo=expo, n=len(open_lots))

    def dstr(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    # MTM 기준 최대낙폭 (최악평가 포함 자산 vs 실현피크)
    eq_arr = np.array([e for _, e in mtm_min_curve])
    ts_arr = np.array([t for t, _ in mtm_min_curve], np.int64)
    pk = np.maximum.accumulate(eq_arr)
    ddw = (eq_arr - pk) / pk
    i = int(np.argmin(ddw))

    print("=" * 66)
    print(f"최저 증거금율(MTM자산/총노출): {worst['ratio']*100:.1f}%  @ {dstr(worst['ts'])}")
    print(f"  당일 MTM자산 ${worst['eq']:,.0f} / 노출 ${worst['expo']:,.0f} / 오픈 {worst['n']}랏 "
          f"(레버 {worst['expo']/worst['eq']:.2f}x)")
    print(f"  청산기준 20% 대비: {'🔴 청산' if worst['ratio'] < 0.20 else '🟢 안전'}")
    print(f"\nMTM(장중최악) 최대낙폭: {ddw[i]*100:.1f}% @ {dstr(int(ts_arr[i]))}")
    print("\n증거금율 하위 8일:")
    for r, t, eq, ex, n in sorted(risky)[:8]:
        print(f"  {dstr(t)}: {r*100:6.1f}%  (MTM ${eq:,.0f} / 노출 ${ex:,.0f} / {n}랏)")


if __name__ == "__main__":
    main()
