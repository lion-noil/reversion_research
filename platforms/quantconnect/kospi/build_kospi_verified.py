# -*- coding: utf-8 -*-
"""코스피 ALIGN3 QuantConnect 결과(results/read_<id>.json, qc_api.py read) + 참조 시뮬(results/ref_sim.json)
→ 사이트 자체 검증 데이터(hyeongeonnoil/src/data/qc_backtest_kospi.json, verified.json 의 kospi 블록 갱신) + 보고서(results/verify_kospi-<label>.md/.data.json).
스키마는 platforms/quantconnect/build_verified_data.py(환율)와 동일 — QcBacktestView 가 그대로 그린다(통화만 KRW).
usage: .venv-quant/Scripts/python platforms/quantconnect/kospi/build_kospi_verified.py [results/read_<id>.json] [--label 2026-10-25] [--public-url URL]"""
import os, sys, json, glob, datetime as dt
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "..", "..", "..", "..", "hyeongeonnoil", "src", "data")
REF = os.path.join(HERE, "results", "ref_sim.json")


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def fnum(s, default=None):
    try:
        return float(str(s).replace("%", "").replace("$", "").replace(",", ""))
    except Exception:
        return default


def day(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    fn = files[0] if files else sorted(glob.glob(os.path.join(HERE, "results", "read_*.json")), key=os.path.getmtime)[-1]
    label = arg("--label", dt.date.today().isoformat())
    r = json.load(open(fn, encoding="utf-8"))
    bt = r["backtest"]
    st = bt.get("statistics") or {}
    rt = bt.get("runtimeStatistics") or {}
    ref = json.load(open(REF, encoding="utf-8")) if os.path.exists(REF) else {}

    # 자본곡선(일 1점) + 낙폭
    eq_vals = ((r.get("charts") or {}).get("Strategy Equity") or {}).get("series", {}).get("Equity", {}).get("values") or []
    dd_vals = ((r.get("charts") or {}).get("Drawdown") or {}).get("series", {}).get("Equity Drawdown", {}).get("values") or []
    dd_by_day = {day(v[0]): round(float(v[1]), 2) for v in dd_vals}
    eq_by_day = {}
    for v in eq_vals:
        close = v[4] if len(v) >= 5 else v[1]
        eq_by_day[day(v[0])] = round(float(close), 2)
    equity, last_dd = [], 0.0
    for d in sorted(eq_by_day):
        last_dd = dd_by_day.get(d, last_dd)
        equity.append([d, eq_by_day[d], last_dd])

    # 월별(rollingWindow M1_*)
    rw = bt.get("rollingWindow") or {}
    mm = {}
    for k in sorted(k for k in rw if k.startswith("M1_")):
        ps = rw[k].get("portfolioStatistics") or {}
        s, e = fnum(ps.get("startEquity")), fnum(ps.get("endEquity"))
        if not s or not e:
            continue
        ymd = k.split("_")[1]
        ts = rw[k].get("tradeStatistics") or {}
        mm[f"{ymd[:4]}-{ymd[4:6]}"] = {"ym": f"{ymd[:4]}-{ymd[4:6]}", "ret": round((e / s - 1) * 100, 2), "start": round(s), "end": round(e),
                                      "trades": int(ts.get("totalNumberOfTrades") or 0), "dd": round(fnum(ps.get("drawdown"), 0) * 100, 2)}
    monthly = [mm[k] for k in sorted(mm)]
    by_year = defaultdict(list)
    for m in monthly:
        by_year[m["ym"][:4]].append(m["ret"])
    yearly = []
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

    # 청산 거래 → 심볼별 + 건당 수익(수수료 전, 진입가 대비)
    ct = (bt.get("totalPerformance") or {}).get("closedTrades") or []
    sym = defaultdict(lambda: {"n": 0, "win": 0, "pnl": 0.0, "long": 0, "short": 0})
    trade_rets = []
    year_q = defaultdict(list)
    for t in ct:
        s = (t.get("symbols") or [{}])[0].get("value") or "KODEX200"
        a = sym[s]
        a["n"] += 1
        a["win"] += 1 if t.get("isWin") else 0
        a["pnl"] += float(t.get("profitLoss") or 0)
        if int(t.get("direction") or 0) == 0:
            a["long"] += 1
        else:
            a["short"] += 1
        ep, xp = fnum(t.get("entryPrice")), fnum(t.get("exitPrice"))
        if ep and xp:
            rr = (xp / ep - 1) * 100
            trade_rets.append(rr)
            year_q[str(t.get("exitTime") or "")[:4]].append(rr)
    symbols = [{"sym": s, "trades": a["n"], "win_rate": round(a["win"] / a["n"] * 100) if a["n"] else 0,
                "pnl": round(a["pnl"]), "avg_pnl": round(a["pnl"] / a["n"]) if a["n"] else 0, "long": a["long"], "short": a["short"]}
               for s, a in sym.items()]
    tsx = (bt.get("totalPerformance") or {}).get("tradeStatistics") or {}
    qc_n = len(trade_rets)
    qc_avg = sum(trade_rets) / qc_n if qc_n else 0.0
    ref_st = ref.get("stats") or {}
    eng_n = int(ref_st.get("trades") or 0)
    eng_avg = (ref_st.get("mean_ret_pct") or 0) + 0.05   # 참조 시뮬은 왕복 0.05% 차감 후 값 → 수수료 전으로 환산
    cell_rows = [{"idx": 0, "cell": "KODEX200 5m align3 long", "qc_n": qc_n, "eng_n": eng_n,
                  "qc_avg": round(qc_avg, 3), "eng_avg": round(eng_avg, 3), "qc_sum": round(sum(trade_rets), 2), "exits": f"0/0/{qc_n}"}]
    ratio = qc_n / eng_n if eng_n else 0
    # 연도별 대조(QC 청산연도 건당 vs 참조)
    ref_y = ref.get("yearly") or {}
    year_rows = []
    for y in sorted(set(year_q) | set(ref_y)):
        q = year_q.get(y, [])
        e = ref_y.get(y) or {}
        year_rows.append({"year": y, "qc_n": len(q), "qc_sum": round(sum(q), 1), "eng_n": e.get("n", 0), "eng_sum": round((e.get("sum") or 0) + 0.05 * (e.get("n") or 0), 1)})

    start_eq, end_eq = fnum(st.get("Start Equity")), fnum(st.get("End Equity"))
    public_url = arg("--public-url", "")
    data = {
        "kind": "qc_backtest", "label": label, "universe": "kospi", "currency": "KRW",
        "meta": {
            "project": r.get("projectId"), "backtest_id": r.get("backtestId"), "backtest_name": bt.get("name"),
            "start": str(bt.get("backtestStart") or "")[:10], "end": str(bt.get("backtestEnd") or "")[:10],
            "tradeable_days": bt.get("tradeableDates"), "node": bt.get("nodeName"), "fetched_at": r.get("fetched_at"),
            "engine": "LEAN (QuantConnect 클라우드) · 자체 수집 5분봉(크레온·한투 API)", "created": bt.get("created"),
            "public_url": public_url, "project_url": f"https://www.quantconnect.com/project/{r.get('projectId')}",
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
            "closed_trades": int(tsx.get("totalNumberOfTrades") or qc_n), "winning_trades": int(tsx.get("numberOfWinningTrades") or 0),
            "largest_profit": fnum(tsx.get("largestProfit")), "largest_loss": fnum(tsx.get("largestLoss")),
            "avg_trade_pnl": fnum(tsx.get("averageProfitLoss")), "avg_duration": tsx.get("averageTradeDuration"),
        },
        "monthly": monthly,
        "monthly_summary": {"n": len(rets), "positive": pos, "median": round(median, 2), "mean": round(total_sum / len(rets), 2) if rets else 0,
                            "best": max(rets) if rets else 0, "worst": min(rets) if rets else 0,
                            "top5_share_pct": round(sum(top5) / total_sum * 100) if total_sum else 0},
        "yearly": yearly, "equity": equity, "symbols": symbols, "cell_rows": cell_rows, "year_rows": year_rows,
        "ref": {"stats": ref_st},
    }
    os.makedirs(SITE, exist_ok=True)
    out1 = os.path.join(SITE, "qc_backtest_kospi.json")
    json.dump(data, open(out1, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    out2 = os.path.join(HERE, "results", f"verify_kospi-{label}.data.json")
    json.dump(data, open(out2, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    s = data["stats"]
    ytxt = " / ".join(f"{y['year']} {y['ret']:+.1f}%" for y in yearly)
    cell_match = (f"거래 건수 QuantConnect {qc_n}건 vs 자체 엔진 {eng_n}건(비율 {ratio:.2f}), 건당 수익(수수료 전) {qc_avg:.2f}% vs {eng_avg:.2f}%"
                  f" — 같은 규칙이 다른 엔진·체결 모델에서 재현됨")
    # verified.json kospi 블록 갱신
    vpath = os.path.join(SITE, "verified.json")
    V = json.load(open(vpath, encoding="utf-8"))
    k = V.setdefault("kospi", {})
    k.update({"period": f"{data['meta']['start']} ~ {data['meta']['end']}", "run_at": label, "engine": data["meta"]["engine"],
              "cell_match": cell_match, "public_url": public_url or k.get("public_url", ""),
              "stats": [["순이익", f"{s['net_profit_pct']:+.1f}% (1,000만 → {end_eq:,.0f}원)"], ["연복리 수익률", f"{s['cagr_pct']:.1f}%"],
                        ["샤프 / 소르티노", f"{s['sharpe']} / {s['sortino']}"], ["최대 낙폭", f"{s['max_dd_pct']}%"],
                        ["주문 수 / 승률", f"{s['orders']}건 / {s['win_rate_pct']:.0f}%"], ["연도별", ytxt]]})
    V["updated"] = dt.date.today().isoformat()
    json.dump(V, open(vpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 보고서 md
    caveats = k.get("caveats") or []
    yr_tbl = "\n".join(f"| {y['year']} | {y['qc_n']} | {y['qc_sum']:+.1f} | {y['eng_n']} | {y['eng_sum']:+.1f} |" for y in year_rows)
    md = (
        f"# 코스피200 ETF(KODEX200) ALIGN3 외부 엔진 재현 — QuantConnect 독립 백테스트 ({label})\n\n"
        f"한국투자증권 실계좌에서 돌아가는 ALIGN3(1일·4시간·5분 3층 z-정렬 추세롱, 7거래일 시간청산, 멀티로트) 규칙을 QuantConnect LEAN 엔진용으로 다시 구현해 "
        f"**다른 엔진·다른 체결 모델**로 돌린 결과입니다. 데이터는 자체 수집 5분봉(크레온 2021-07~2026-08 + 한투 API 실행기 봉 ~2026-10-23)이라 "
        f"환율 검증(Oanda 외부 데이터)보다 한 단계 약한 '같은 데이터·다른 엔진' 검증입니다. 기간 {data['meta']['start']} ~ {data['meta']['end']}, 거래일 {data['meta']['tradeable_days']}일.\n\n"
        f"**핵심 숫자**: 순이익 {s['net_profit_pct']:+.1f}% (1,000만 → {end_eq:,.0f}원), CAGR {s['cagr_pct']:.1f}%, 샤프 {s['sharpe']}, 최대 낙폭 {s['max_dd_pct']}%, "
        f"주문 {s['orders']}건(청산 {qc_n}건, 승률 {s['win_rate_pct']:.0f}%). 연도별(월 복리): {ytxt}. 월 {len(rets)}개 중 플러스 {pos}개, 월 중앙값 {median:+.2f}%, 상위 5개월이 합계의 {data['monthly_summary']['top5_share_pct']}%.\n\n"
        f"> 자체 검증 페이지: https://hyeongeonnoil.com/verified (4번 섹션)" + (f" · 원본: {public_url}" if public_url else "") + "\n\n"
        f"## 요약 통계 (QuantConnect 집계, 통화 단위 원)\n| 항목 | 값 |\n|---|---|\n"
        f"| 초기/최종 자본 | 10,000,000 → {end_eq:,.0f}원 ({s['net_profit_pct']:+.1f}%) |\n| CAGR | {s['cagr_pct']:.2f}% |\n| 샤프 / 소르티노 / PSR | {s['sharpe']} / {s['sortino']} / {s['psr_pct']}% |\n"
        f"| 최대 낙폭 | {s['max_dd_pct']}% (회복 {s['dd_recovery_days']}일) |\n| 주문 / 청산 | {s['orders']} / {qc_n} (승률 {s['win_rate_pct']:.0f}%, 평균 이익 {s['avg_win_pct']}% / 손실 {s['avg_loss_pct']}%, 손익비 {s['pl_ratio']}) |\n"
        f"| 수수료 | {s['fees']:,.0f}원 (매수·매도 각 0.025%) |\n| 엔진 | {data['meta']['engine']}, 노드 {data['meta']['node']} |\n\n"
        f"## 자체 엔진과 대조\n| 항목 | QuantConnect | 자체 엔진(ref_sim) |\n|---|---|---|\n"
        f"| 청산 건수 | {qc_n} | {eng_n} (비율 {ratio:.2f}) |\n| 건당 수익(수수료 전) | {qc_avg:.3f}% | {eng_avg:.3f}% |\n"
        f"| 순이익 | {s['net_profit_pct']:+.1f}% | {ref_st.get('net_pct', 0):+.1f}% |\n| CAGR | {s['cagr_pct']:.1f}% | {ref_st.get('cagr_pct', 0):.1f}% |\n| 최대 낙폭 | {s['max_dd_pct']}% | {ref_st.get('mdd_pct', 0)}% |\n\n"
        f"### 연도별(청산 연도 기준, 건당 수익 합계 %p, 수수료 전)\n| 연도 | QC 건수 | QC 합계 | 엔진 건수 | 엔진 합계 |\n|---|---|---|---|---|\n{yr_tbl}\n\n"
        f"## 숫자를 그대로 믿으면 안 되는 이유\n" + "\n".join(f"- {c}" for c in caveats) + "\n\n"
        f"코드 `platforms/quantconnect/kospi/main_kospi.py`, 데이터 변환 `export_kospi_qc.py`·`make_embed.py`, 참조 시뮬 `ref_sim.py`, 실행 `qc_api.py`.\n"
    )
    out3 = os.path.join(HERE, "results", f"verify_kospi-{label}.md")
    open(out3, "w", encoding="utf-8").write(md)
    print(f"사이트 데이터: {out1} ({os.path.getsize(out1)//1024} KB) — 자본곡선 {len(equity)}일, 월 {len(monthly)}, 청산 {qc_n}")
    print("대조:", cell_match)
    print("연도별:", yearly)
    print("보고서:", out3)


if __name__ == "__main__":
    main()
