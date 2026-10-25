# -*- coding: utf-8 -*-
"""QuantConnect 결과 원본(results/read_<id>.json, qc_fetch_result.py) → 사이트 자체 검증 페이지용 압축 JSON + 보고서 발행본.
공개 임베드(quantconnect.cloud)가 죽어도(2026-10-25 실제) 사이트가 자본곡선·월별·셀별 표를 직접 그리기 위한 데이터.

출력:
  hyeongeonnoil/src/data/qc_backtest.json          — /verified 페이지와 보고서(verify:<label>)가 같이 쓰는 데이터
  results/verify_<label>.md / verify_<label>.data.json — 보고서 해시(trading:reports) 발행용 (publish_verify_report.py)
usage: py -3 platforms/quantconnect/build_verified_data.py [results/read_<id>.json] [--label 2026-10-18]
"""
import os, sys, re, json, glob, datetime as dt
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔 cp949 에서 '—' 출력 오류 방지
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SITE_DATA = os.path.join(HERE, "..", "..", "..", "hyeongeonnoil", "src", "data")
REPORT_MD = os.path.join(HERE, "results", "REPORT_2026-10-18.md")


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def fnum(s, default=None):
    try:
        return float(str(s).replace("%", "").replace(",", "").replace("$", "").strip())
    except Exception:
        return default


def day(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


def parse_cells_from_report(path):
    """REPORT md 의 '셀별 결과' 표 → 행 리스트. 열: #, 셀, QC건수, 엔진건수, QC건당%, 엔진건당%, QC합계%, TP/SL/시간"""
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            rows.append({"idx": int(m[1]), "cell": m[2], "qc_n": int(m[3]), "eng_n": int(m[4]),
                         "qc_avg": float(m[5]), "eng_avg": float(m[6]), "qc_sum": float(m[7]), "exits": m[8]})
    return rows


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    fn = files[0] if files else sorted(glob.glob(os.path.join(HERE, "results", "read_*.json")))[-1]
    label = arg("--label", "2026-10-18")
    r = json.load(open(fn, encoding="utf-8"))
    bt = r["backtest"]
    st = bt.get("statistics") or {}
    rt = bt.get("runtimeStatistics") or {}

    # ── 자본곡선(일 1점) + 낙폭 ──
    eq_vals = ((r.get("charts") or {}).get("Strategy Equity") or {}).get("series", {}).get("Equity", {}).get("values") or []
    dd_vals = ((r.get("charts") or {}).get("Drawdown") or {}).get("series", {}).get("Equity Drawdown", {}).get("values") or []
    dd_by_day = {}
    for v in dd_vals:
        dd_by_day[day(v[0])] = round(float(v[1]), 2)
    eq_by_day = {}
    for v in eq_vals:
        close = v[4] if len(v) >= 5 else v[1]
        eq_by_day[day(v[0])] = round(float(close), 2)   # 같은 날 여러 점이면 마지막(종가)
    equity = []
    last_dd = 0.0
    for d in sorted(eq_by_day):
        last_dd = dd_by_day.get(d, last_dd)
        equity.append([d, eq_by_day[d], last_dd])

    # ── 월별 수익률 (rollingWindow M1_*: 해당 월 시작→끝 자본) ──
    rw = bt.get("rollingWindow") or {}
    monthly = []
    for k in sorted(k for k in rw if k.startswith("M1_")):
        ps = rw[k].get("portfolioStatistics") or {}
        s, e = fnum(ps.get("startEquity")), fnum(ps.get("endEquity"))
        if not s or not e:
            continue
        ymd = k.split("_")[1]
        ym = f"{ymd[:4]}-{ymd[4:6]}"
        ts = rw[k].get("tradeStatistics") or {}
        monthly.append({"ym": ym, "ret": round((e / s - 1) * 100, 2), "start": round(s), "end": round(e),
                        "trades": int(ts.get("totalNumberOfTrades") or 0), "dd": round(fnum(ps.get("drawdown"), 0) * 100, 2)})
    # 월이 겹쳐 들어오는 경우(마지막 부분월) 같은 ym 은 뒤의 것으로
    mm = {}
    for m in monthly:
        mm[m["ym"]] = m
    monthly = [mm[k] for k in sorted(mm)]
    yearly = []
    by_year = defaultdict(list)
    for m in monthly:
        by_year[m["ym"][:4]].append(m["ret"])
    for y in sorted(by_year):
        g = 1.0
        for x in by_year[y]:
            g *= 1 + x / 100
        yearly.append({"year": y, "ret": round((g - 1) * 100, 1), "months": len(by_year[y])})
    rets = [m["ret"] for m in monthly]
    pos = sum(1 for x in rets if x > 0)
    srt = sorted(rets)
    median = srt[len(srt) // 2] if srt else 0
    top5 = sorted(rets, reverse=True)[:5]
    total_sum = sum(rets)

    # ── 청산 거래(closedTrades) → 심볼별·방향별 ──
    ct = (bt.get("totalPerformance") or {}).get("closedTrades") or []
    sym = defaultdict(lambda: {"n": 0, "win": 0, "pnl": 0.0, "long": 0, "short": 0})
    for t in ct:
        s = (t.get("symbols") or [{}])[0].get("value") or "?"
        a = sym[s]
        a["n"] += 1
        a["win"] += 1 if t.get("isWin") else 0
        a["pnl"] += float(t.get("profitLoss") or 0)
        if int(t.get("direction") or 0) == 0:
            a["long"] += 1
        else:
            a["short"] += 1
    symbols = [{"sym": s, "trades": a["n"], "win_rate": round(a["win"] / a["n"] * 100) if a["n"] else 0,
                "pnl": round(a["pnl"]), "avg_pnl": round(a["pnl"] / a["n"], 1) if a["n"] else 0, "long": a["long"], "short": a["short"]}
               for s, a in sym.items()]
    symbols.sort(key=lambda x: -x["pnl"])
    tsx = (bt.get("totalPerformance") or {}).get("tradeStatistics") or {}

    start_eq, end_eq = fnum(st.get("Start Equity")), fnum(st.get("End Equity"))
    data = {
        "kind": "qc_backtest",
        "label": label,
        "meta": {
            "project": r.get("projectId"), "backtest_id": r.get("backtestId"), "backtest_name": r.get("name"),
            "start": str(bt.get("backtestStart") or "")[:10], "end": str(bt.get("backtestEnd") or "")[:10],
            "tradeable_days": bt.get("tradeableDates"), "node": bt.get("nodeName"), "fetched_at": r.get("fetched_at"),
            "engine": "LEAN (QuantConnect 클라우드) · Oanda 분봉", "created": bt.get("created"),
            "public_url": f"https://www.quantconnect.cloud/backtest/{r.get('backtestId')}/?theme=darkly",
            "project_url": f"https://www.quantconnect.com/project/{r.get('projectId')}",
        },
        "stats": {
            "net_profit_pct": fnum(st.get("Net Profit")), "cagr_pct": fnum(st.get("Compounding Annual Return")),
            "sharpe": fnum(st.get("Sharpe Ratio")), "sortino": fnum(st.get("Sortino Ratio")), "psr_pct": fnum(st.get("Probabilistic Sharpe Ratio")),
            "max_dd_pct": fnum(st.get("Drawdown")), "dd_recovery_days": fnum(st.get("Drawdown Recovery")),
            "orders": int(fnum(st.get("Total Orders"), 0)), "win_rate_pct": fnum(st.get("Win Rate")), "loss_rate_pct": fnum(st.get("Loss Rate")),
            "avg_win_pct": fnum(st.get("Average Win")), "avg_loss_pct": fnum(st.get("Average Loss")), "pl_ratio": fnum(st.get("Profit-Loss Ratio")),
            "expectancy": fnum(st.get("Expectancy")), "alpha": fnum(st.get("Alpha")), "beta": fnum(st.get("Beta")),
            "ann_std": fnum(st.get("Annual Standard Deviation")), "fees": fnum(st.get("Total Fees")), "capacity": st.get("Estimated Strategy Capacity"),
            "capacity_asset": st.get("Lowest Capacity Asset"), "turnover_pct": fnum(st.get("Portfolio Turnover")),
            "start_equity": start_eq, "end_equity": end_eq, "volume": rt.get("Volume"),
            "closed_trades": int(tsx.get("totalNumberOfTrades") or 0), "winning_trades": int(tsx.get("numberOfWinningTrades") or 0),
            "largest_profit": fnum(tsx.get("largestProfit")), "largest_loss": fnum(tsx.get("largestLoss")),
            "avg_trade_pnl": fnum(tsx.get("averageProfitLoss")), "avg_duration": tsx.get("averageTradeDuration"),
        },
        "monthly": monthly,
        "monthly_summary": {"n": len(rets), "positive": pos, "median": round(median, 2), "mean": round(total_sum / len(rets), 2) if rets else 0,
                            "best": max(rets) if rets else 0, "worst": min(rets) if rets else 0,
                            "top5_share_pct": round(sum(top5) / total_sum * 100) if total_sum else 0},
        "yearly": yearly,
        "equity": equity,
        "symbols": symbols,
        "cell_rows": parse_cells_from_report(REPORT_MD),
    }
    os.makedirs(SITE_DATA, exist_ok=True)
    out1 = os.path.join(SITE_DATA, "qc_backtest.json")
    json.dump(data, open(out1, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    out2 = os.path.join(HERE, "results", f"verify_{label}.data.json")
    json.dump(data, open(out2, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"사이트 데이터: {out1} ({os.path.getsize(out1)//1024} KB) — 자본곡선 {len(equity)}일, 월 {len(monthly)}, 심볼 {len(symbols)}, 셀 {len(data['cell_rows'])}")
    print("월별 요약:", data["monthly_summary"])
    print("연도별:", yearly)

    # ── 보고서 마크다운(발행본): 기존 REPORT md 본문 + 자체 호스팅 안내 ──
    body = open(REPORT_MD, encoding="utf-8").read() if os.path.exists(REPORT_MD) else ""
    body = re.sub(r"^# .*\n", "", body, count=1)
    ytxt = " / ".join(f"{y['year']} {y['ret']:+.1f}%" for y in yearly)
    md = (
        f"# 환율 메이저 유니버스(17셀) 외부 재현 검증 — QuantConnect 독립 백테스트\n\n"
        f"**검증일 {label}** (QuantConnect 실행일). 유니버스 단위 기록이라 재검증하면 이 글이 갱신됩니다.\n\n"
        f"라이브 환율 봇의 규칙 17개 셀을 QuantConnect LEAN 엔진용으로 다시 구현해 **다른 데이터(Oanda 분봉)·다른 엔진·다른 체결 모델**로 돌린 결과입니다. "
        f"기간 {data['meta']['start']} ~ {data['meta']['end']}, 거래일 {data['meta']['tradeable_days']}일. "
        f"결과 원본(자본곡선·월별 통계·청산 거래 {data['stats']['closed_trades']}건)은 QuantConnect API에서 내려받아 이 사이트에 보관합니다 — "
        f"공개 임베드 페이지(quantconnect.cloud)가 접속되지 않아도 아래 표와 차트는 그대로 봅니다.\n\n"
        f"**핵심 숫자**: 순이익 {data['stats']['net_profit_pct']:+.1f}% (10만 → {end_eq:,.0f} 달러), CAGR {data['stats']['cagr_pct']:.1f}%, "
        f"샤프 {data['stats']['sharpe']}, 최대 낙폭 {data['stats']['max_dd_pct']}%, 주문 {data['stats']['orders']}건(승률 {data['stats']['win_rate_pct']:.0f}%). "
        f"연도별(월 복리): {ytxt}. 월 60개 중 플러스 {pos}개, 월 중앙값 {median:+.2f}%, 상위 5개월이 합계의 {data['monthly_summary']['top5_share_pct']}%.\n\n"
        f"> 자체 검증 페이지: https://hyeongeonnoil.com/verified · 원본(접속 불안정): {data['meta']['public_url']}\n\n"
        + body
    )
    out3 = os.path.join(HERE, "results", f"verify_{label}.md")
    open(out3, "w", encoding="utf-8").write(md)
    print("보고서 md:", out3, len(md), "자")


if __name__ == "__main__":
    main()
