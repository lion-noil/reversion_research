"""확정 S1(추세)·S2(역추세) — 이번달(2026-06) 신호 발생 목록·시각·성적.
중첩(라이브) 엔진 기준(추매 없음=확정 베이스). 데이터 끝까지 청산 안 된 건 '진행중' 표시.
시각은 UTC·KST(UTC+9) 병기. 사용: python backtests/this_month_signals.py"""
from __future__ import annotations
import sys, os
import numpy as np
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt_s1_quarters_mt5 import prep
from bt_avgdown import FEE, H_MS, S1_TREND_LONG, S2_REV_LONG, S2_REV_SHORT, SIDE
from bt_avgdown_overlap import sim_overlap

DAY = 86400 * 1000
MH = 14 * DAY
MONTH_LO = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
MONTH_HI = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
KST = timezone(timedelta(hours=9))

# S1 추세 숏 채택(🟢 PF≥1.3·exp>0) — HANDOFF_S1_trend.md '추세 숏' 컬럼
S1_TREND_SHORT = {"SOLUSDT": (3.4, -2.0, 1.5), "XAUUSD": (3.2, 0.2, 1.0),
                  "XAGUSD": (2.65, 1.2, 2.0), "XAUTUSDT": (3.5, 0.8, 0.75), "WTI": (3.2, 1.8, 1.0)}

GROUPS = [("S1 추세 롱", "trend_long", S1_TREND_LONG),
          ("S1 추세 숏", "trend_short", S1_TREND_SHORT),
          ("S2 역추세 롱", "rev_long", S2_REV_LONG),
          ("S2 역추세 숏", "rev_short", S2_REV_SHORT)]


def u(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m/%d %H:%M")
def k(ms): return datetime.fromtimestamp(ms / 1000, tz=KST).strftime("%m/%d %H:%M")


def main():
    cache = {}
    rows = []   # (entry_ts, grp, sym, side, status, reason, ret, open_flag)
    for gname, kind, params in GROUPS:
        side = "롱" if SIDE[kind] > 0 else "숏"
        for s, (k1, b, cdh) in params.items():
            d = cache.get(s) or prep(s)
            cache[s] = d
            if d is None:
                continue
            end_ts = int(d["ts"][-1]); last_close = float(d["C"][-1])
            trs = sim_overlap(d, kind, k1, b, int(cdh * H_MS), mode="parallel")
            for t in trs:
                if not (MONTH_LO <= t["entry_ts"] < MONTH_HI):
                    continue
                # 진행중 판정: 14일창이 데이터끝을 넘고 아직 TP/SL 못 맞춘 경우
                open_flag = (t["exit_ts"] >= end_ts - 60000) and (t["entry_ts"] + MH > end_ts)
                rows.append(dict(entry=t["entry_ts"], grp=gname, sym=s, side=side,
                                 ret=t["ret"] * 100, open=open_flag))
    rows.sort(key=lambda r: r["entry"])

    # ── 전략별 출력 ──
    for gname, kind, params in GROUPS:
        sub = [r for r in rows if r["grp"] == gname]
        print("\n" + "=" * 78)
        print(f"  {gname}  —  이번달 신호 {len(sub)}건")
        print("=" * 78)
        if not sub:
            print("   (없음)"); continue
        print(f"{'진입(UTC)':>12}{'(KST)':>13} {'심볼':>9}{'방향':>4}  {'상태':>6}{'수익%':>8}")
        for r in sub:
            st = "진행중" if r["open"] else "종료"
            rv = f"{r['ret']:+.2f}" + ("*" if r["open"] else "")
            print(f"{u(r['entry']):>12}{k(r['entry']):>13} {r['sym']:>9}{r['side']:>4}  {st:>6}{rv:>8}")
        closed = [r for r in sub if not r["open"]]
        opn = [r for r in sub if r["open"]]
        if closed:
            cr = np.array([r["ret"] for r in closed])
            print(f"   ▶ 종료 {len(closed)}건: 승 {(cr>0).sum()}/패 {(cr<=0).sum()}, "
                  f"평균 {cr.mean():+.2f}%, 합 {cr.sum():+.2f}%")
        if opn:
            orr = np.array([r["ret"] for r in opn])
            print(f"   ▶ 진행중 {len(opn)}건(*=미실현 평가손익): 평균 {orr.mean():+.2f}%")

    # ── 총괄 ──
    print("\n" + "=" * 78)
    print("  총괄 (이번달 2026-06, 데이터 끝까지)")
    print("=" * 78)
    allc = [r for r in rows if not r["open"]]
    allo = [r for r in rows if r["open"]]
    print(f"  전체 신호 {len(rows)}건  =  종료 {len(allc)} + 진행중 {len(allo)}")
    if allc:
        a = np.array([r["ret"] for r in allc])
        print(f"  종료분 성적: 승률 {100*(a>0).mean():.0f}%  평균 {a.mean():+.2f}%/건  합 {a.sum():+.2f}%")
    print("  ※ 시각=신호봉 종가시각. 데이터 컷오프: 크립토~6/20, 지수·금속~6/24~25, HK50 6/18.")


if __name__ == "__main__":
    main()
