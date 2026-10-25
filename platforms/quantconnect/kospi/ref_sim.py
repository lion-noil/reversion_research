# -*- coding: utf-8 -*-
"""ALIGN3 참조 시뮬 — main_kospi.py(LEAN 이식)와 같은 규칙·사이징을 같은 CSV(kospi/data)로 로컬 재현.
QC 결과의 거래 건수·건당 수익·연도별 합계와 대조하는 '엔진' 열 원천. 체결 = 봉 종가, 수수료 편도 0.025%, 초기자본 1,000만.
출력: kospi/results/ref_sim.json {trades, yearly, monthly, stats}
usage: .venv-quant/Scripts/python platforms/quantconnect/kospi/ref_sim.py [--end 20260727] [--out ref_sim.json]
  --end : QuantConnect 무료 계층은 최근 90일을 표본 밖으로 잠가 백테스트 종료일이 당겨진다(2026-10-25 실행 → 2026-07-27) → 같은 구간으로 맞출 때 사용"""
import os, sys, glob, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
END = sys.argv[sys.argv.index("--end") + 1] if "--end" in sys.argv else "99999999"
OUT_NAME = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "ref_sim.json"
W_1D, W_4H, W_5M = 20, 20, 390
CD_BARS, HOLD_TDAYS, CAP_MAX = 12, 7, 20
ENTRY_LAST, EXIT_CLAMP = "1515", "1510"
FEE = 0.00025
CASH0 = 10_000_000


def zscore(vals, w):
    if len(vals) < w:
        return None
    seg = vals[-w:]
    m = sum(seg) / w
    var = sum((x - m) ** 2 for x in seg) / w
    return (seg[-1] - m) / math.sqrt(var) if var > 0 else None


def main():
    bars = []
    for fn in sorted(glob.glob(os.path.join(HERE, "data", "069500_5m_*.csv"))):
        for line in open(fn, encoding="utf-8"):
            if not line[0].isdigit():
                continue
            p = line.rstrip("\n").split(",")
            if p[0][:8] > END:
                continue
            bars.append((p[0][:8], p[0][9:11] + p[0][12:14], float(p[4])))
    closes, daily_dates, daily_closes, sess = [], [], [], []
    cur_date = cur_last = am_last = pm_last = None
    am_done = False
    lots, trades = [], []
    last_entry = -10 ** 9
    cash = CASH0
    eq_by_day = {}
    for idx, (d, hhmm, px) in enumerate(bars):
        if cur_date is not None and d != cur_date:
            daily_dates.append(cur_date); daily_closes.append(cur_last)
            if pm_last is not None:
                sess.append(pm_last)
            am_last = pm_last = None; am_done = False
        cur_date, cur_last = d, px
        if hhmm < "1300":
            am_last = px
        else:
            if not am_done and am_last is not None:
                sess.append(am_last); am_done = True
            pm_last = px
        closes.append(px)
        for lot in list(lots):
            held = sum(1 for dd in daily_dates if dd > lot["d"]) + (1 if d > lot["d"] else 0)
            if held >= HOLD_TDAYS and hhmm >= min(lot["t"], EXIT_CLAMP):
                cash += lot["shares"] * px * (1 - FEE)
                r = px / lot["px"] - 1 - 2 * FEE
                trades.append({"entry": f"{lot['d']} {lot['t']}", "exit": f"{d} {hhmm}", "entry_px": lot["px"], "exit_px": px,
                               "shares": lot["shares"], "ret_pct": round(r * 100, 3), "pnl": round(lot["shares"] * (px * (1 - FEE) - lot["px"] * (1 + FEE)))})
                lots.remove(lot)
        equity = cash + sum(l["shares"] * px for l in lots)
        eq_by_day[d] = equity
        if hhmm > ENTRY_LAST:
            continue
        z1, z4, z5 = zscore(daily_closes, W_1D), zscore(sess, W_4H), zscore(closes, W_5M)
        if None in (z1, z4, z5) or not (z1 >= 1 and z4 >= 1 and z5 >= 1):
            continue
        if idx - last_entry < CD_BARS:
            continue
        cap = min(CAP_MAX, max(1, int(equity // px)))
        if len(lots) >= cap:
            continue
        slot = equity / cap
        shares = int(min(slot, cash) // px)
        if shares <= 0:
            continue
        cost = shares * px * (1 + FEE)
        if cost > cash:
            continue
        cash -= cost
        lots.append({"d": d, "t": hhmm, "px": px, "shares": shares})
        last_entry = idx
    days = sorted(eq_by_day)
    monthly = {}
    for dd in days:
        monthly.setdefault(dd[:6], [None, None])
        if monthly[dd[:6]][0] is None:
            monthly[dd[:6]][0] = eq_by_day[dd]
        monthly[dd[:6]][1] = eq_by_day[dd]
    prev = CASH0
    mrows = []
    for ym in sorted(monthly):
        end = monthly[ym][1]
        mrows.append({"ym": f"{ym[:4]}-{ym[4:]}", "ret": round((end / prev - 1) * 100, 2), "end": round(end)})
        prev = end
    yearly = {}
    for t in trades:
        yearly.setdefault(t["exit"][:4], []).append(t["ret_pct"])
    peak, mdd = -1, 0
    for dd in days:
        e = eq_by_day[dd]
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak)
    final = eq_by_day[days[-1]]
    yrs = (len(days)) / 246
    rs = [t["ret_pct"] for t in trades]
    stats = {"final": round(final), "net_pct": round((final / CASH0 - 1) * 100, 2), "cagr_pct": round(((final / CASH0) ** (1 / yrs) - 1) * 100, 2),
             "mdd_pct": round(mdd * 100, 2), "trades": len(trades), "win_rate": round(sum(1 for r in rs if r > 0) / len(rs) * 100, 1) if rs else 0,
             "mean_ret_pct": round(sum(rs) / len(rs), 3) if rs else 0, "open_lots_end": len(lots), "first_day": days[0], "last_day": days[-1]}
    out = {"stats": stats, "yearly": {y: {"n": len(v), "sum": round(sum(v), 2), "mean": round(sum(v) / len(v), 3)} for y, v in sorted(yearly.items())},
           "monthly": mrows, "trades": trades}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", OUT_NAME), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))
    print("연도별:", json.dumps(out["yearly"], ensure_ascii=False))


if __name__ == "__main__":
    main()
