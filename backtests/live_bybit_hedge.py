"""롱·숏 헷지(넷팅) 효과 정량화 — live_bybit_sim 구성.

비교:
  (A) 동시최악(기존): 롱=당일저가, 숏=당일고가 — 헷지 무시(보수)
  (B) 일관가격(넷팅): 전 랏을 같은 시나리오로 마킹 — min(전부 저가, 전부 고가)
      같은 심볼 롱+숏 쌍은 (B)에서 상쇄됨 = 실제 헷지 반영.
추가: 총노출(gross) vs 순노출(net) 비율 — 책이 얼마나 내부 헷지돼 있는지.
"""
import sys, os, heapq
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from live_bybit_margin import build_events
from live_bybit_sim import prep_daily, INIT, LOT_FRAC, MAX_LOTS, DAY


def main():
    events, cacheD = build_events()
    events.sort(key=lambda x: x[0])
    span0 = events[0][0]; span1 = max(e[1] for e in events)

    dHL = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "XAUTUSDT"):
        d = cacheD.get(sym) or prep_daily(sym)
        dHL[sym] = dict(zip(d["ts"].tolist(), zip(d["H"].tolist(), d["L"].tolist())))
    days = sorted(t for t in dHL["BTCUSDT"] if span0 <= t <= span1 + DAY)

    equity = INIT
    heap = []; open_lots = []; ei = 0
    wA = dict(ratio=9e9, ts=0); wB = dict(ratio=9e9, ts=0)
    ddA_curve = []; ddB_curve = []
    gross_net = []  # (day, gross, net_abs)

    def realize(t):
        nonlocal equity
        while heap and heap[0][0] <= t:
            _, ret, notional = heapq.heappop(heap)
            equity += notional * ret

    for dts_ in days:
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
        ok = True
        uA = 0.0
        uLow = 0.0   # 전부 당일저가 시나리오
        uHigh = 0.0  # 전부 당일고가 시나리오
        gross = 0.0
        net = {}     # sym -> signed notional
        for l in open_lots:
            hl = dHL[l["sym"]].get(dts_)
            if hl is None:
                ok = False; break
            hi, lo = hl
            if l["side"] > 0:
                uA += min((lo / l["px"] - 1.0) * l["notional"], 0.0)
                uLow += (lo / l["px"] - 1.0) * l["notional"]
                uHigh += (hi / l["px"] - 1.0) * l["notional"]
            else:
                uA += min((l["px"] - hi) / l["px"] * l["notional"], 0.0)
                uLow += (l["px"] - lo) / l["px"] * l["notional"]
                uHigh += (l["px"] - hi) / l["px"] * l["notional"]
            gross += l["notional"]
            net[l["sym"]] = net.get(l["sym"], 0.0) + l["side"] * l["notional"]
        if not ok or gross <= 0:
            continue
        uB = min(uLow, uHigh, 0.0)          # 일관가격 최악
        eqA = equity + uA; eqB = equity + uB
        rA = eqA / gross; rB = eqB / gross
        ddA_curve.append((dts_, eqA)); ddB_curve.append((dts_, eqB))
        net_abs = sum(abs(v) for v in net.values())
        gross_net.append((dts_, gross, net_abs))
        if rA < wA["ratio"]: wA = dict(ratio=rA, ts=dts_, eq=eqA, expo=gross, n=len(open_lots))
        if rB < wB["ratio"]: wB = dict(ratio=rB, ts=dts_, eq=eqB, expo=gross, n=len(open_lots))

    def dstr(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    def mdd(curve):
        eq = np.array([e for _, e in curve]); ts = np.array([t for t, _ in curve], np.int64)
        pk = np.maximum.accumulate(eq); dd = (eq - pk) / pk
        i = int(np.argmin(dd)); return dd[i], int(ts[i])

    dA, tA = mdd(ddA_curve); dB, tB = mdd(ddB_curve)
    g = np.array([x[1] for x in gross_net]); na = np.array([x[2] for x in gross_net])
    hedge_frac = 1.0 - na / g   # 내부 상쇄된 비율

    print("=" * 70)
    print("(A) 동시최악(헷지 무시·기존): 최저 증거금율 "
          f"{wA['ratio']*100:.1f}% @ {dstr(wA['ts'])} | MTM MDD {dA*100:.1f}% @ {dstr(tA)}")
    print("(B) 일관가격(넷팅=실제 헷지): 최저 증거금율 "
          f"{wB['ratio']*100:.1f}% @ {dstr(wB['ts'])} | MTM MDD {dB*100:.1f}% @ {dstr(tB)}")
    print(f"\n노출 내부헷지 비율(1 - |순노출|/총노출): 평균 {hedge_frac.mean()*100:.0f}% / "
          f"중앙값 {np.median(hedge_frac)*100:.0f}% / 최대 {hedge_frac.max()*100:.0f}%")
    print(f"총노출 평균 ${g.mean():,.0f} vs 순노출 평균 ${na.mean():,.0f}")


if __name__ == "__main__":
    main()
