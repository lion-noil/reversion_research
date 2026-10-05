# -*- coding: utf-8 -*-
"""환율 유니버스 포트폴리오 시뮬 (2026-10-05) — 지금 라이브 환율 셀 vs 스크리닝 통과 셀을 얹은 구성.

입력: 라이브 환율 셀(live_cells.json, live_vs_backtest 엔진) + results/fx_cross_cells.json(fx_cross_screen.py 통과 셀·거래 포함).
사이징 = 라이브(1분·4h 5.75%, 일봉 28.75%). 수익·낙폭은 계좌 대비 단순합(성적표와 같은 정의).
평가손 포함 낙폭 = 4시간봉 격자에서 보유 포지션을 그 봉 최악가(롱=저가, 숏=고가)로 평가 — 성적표 MTM 기준과 같은 보수적 방식.
  ⚠️ 메이저 4h(UTC)와 크로스 4h(MT5 서버시각)는 2~3시간 어긋난 채로 합친다 → 일 단위 수치는 근사.
창 = 최근 5년(T5Y~, 라이브 기준과 동일). 시나리오마다 '평소 평가낙폭(월 95%)이 지금과 같아지도록' 사이징을 맞춘 수익도 같이 낸다.
usage: py -3 backtests/fx_portfolio_sim.py  → results/FX_PORTFOLIO_SIM.md
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_vs_backtest import DAY, FX, LIVE, T5Y, arrs as lv_arrs, run_cell, sizing_pct  # noqa: E402
from fx_cross_screen import MAJORS, MODENM, arrs  # noqa: E402

H4 = 4 * 3600 * 1000
SZ_INTRA, SZ_DAILY = 5.75, 28.75
_h4 = {}


def h4(sym):
    if sym not in _h4:
        _h4[sym] = arrs(sym, '240')
    return _h4[sym]


def live_trades():
    """[(en, ex, r, side, price, sym, 사이징%, 라벨)] — 라이브 환율 셀"""
    out, live_keys = [], set()
    for book, subs in LIVE.items():
        for sub in subs:
            strat, iv, hold_ms = sub['strategy'], sub['candle_interval'], int(sub['max_hold_sec']) * 1000
            for sym, sides in (sub['params'] or {}).items():
                if sym not in FX:
                    continue
                for side, p in sides.items():
                    if float(p.get('k1', 0) or 0) >= 99:
                        continue
                    d = lv_arrs(sym, iv)
                    tf = {'1': '1분', '240': '4h', 'D': '일봉'}.get(str(iv), str(iv))
                    trend = strat in ('s1', 's3', 's11')
                    mode = ('t' if trend else 'r') + ('l' if side == 'long' else 's')
                    live_keys.add((sym, tf, mode))
                    for en, ex, r in run_cell(strat, sym, side, p, iv, hold_ms):
                        if en < T5Y:
                            continue
                        i = int(np.searchsorted(d[3], en, 'right')) - 1
                        out.append((en, ex, r, 1 if side == 'long' else -1, float(d[0][max(i, 0)]), sym, sizing_pct(book, strat, sym), f"live {sym} {tf} {MODENM[mode]}", (sym, tf, mode)))
    return out, live_keys


def cand_trades(cells, pick):
    out = []
    for c in cells:
        if not pick(c):
            continue
        sz = SZ_DAILY if c['tf'] == '일봉' else SZ_INTRA
        for en, ex, r, side, price in c['trades']:
            if en >= T5Y:
                out.append((en, ex, r, side, price, c['sym'], sz, f"new {c['sym']} {c['tf']} {MODENM[c['mode']]}", (c['sym'], c['tf'], c['mode'])))
    return out


def metrics(trades, scale=1.0):
    g0 = T5Y // H4
    g1 = max(t[1] for t in trades) // H4 + 1
    n = int(g1 - g0 + 1)
    realized = np.zeros(n)
    unreal = np.zeros(n)
    for en, ex, r, side, price, sym, sz, *_ in trades:
        f = sz / 100 * scale
        realized[int(ex // H4 - g0):] += f * r * 100
        d = h4(sym)
        if d is None:
            continue
        C, Hh, L, ts = d
        i0, i1 = int(np.searchsorted(ts, en, 'right')), int(np.searchsorted(ts, ex, 'left'))
        if i1 <= i0:
            continue
        worst = (L[i0:i1] if side > 0 else Hh[i0:i1])
        u = f * side * (worst / price - 1) * 100
        idx = (ts[i0:i1] // H4 - g0).astype(int)
        ok = (idx >= 0) & (idx < n)
        np.add.at(unreal, idx[ok], u[ok])
    eq = realized + unreal                      # 계좌 대비 %p (단순합)
    t = (np.arange(n) + g0) * H4
    ym = np.array([(datetime.fromtimestamp(int(x) / 1000, timezone.utc) + timedelta(hours=2)).strftime("%Y-%m") for x in t])
    day = (t + 2 * 3600 * 1000) // DAY
    months = sorted(set(ym))[1:-1]
    mret, mdd = [], []
    for k in months:
        i = np.where(ym == k)[0]
        mret.append(realized[i[-1]] - realized[i[0]])
        pk = np.maximum.accumulate(eq[i])
        mdd.append((pk - eq[i]).max())
    ddd = []
    for dd_ in np.unique(day):
        i = np.where(day == dd_)[0]
        ddd.append(max(0.0, eq[i[0]] - eq[i].min()))
    pk = np.maximum.accumulate(realized)
    yrs = (g1 - g0) * H4 / (365.25 * DAY)
    # 최대 동시 노출
    ev = sorted([(x[0], x[6] * scale) for x in trades] + [(x[1], -x[6] * scale) for x in trades])
    cur = mx = 0.0
    for _, v in ev:
        cur += v
        mx = max(mx, cur)
    return dict(per_month=len(trades) / yrs / 12, month_mean=float(np.mean(mret)), month_median=float(np.median(mret)),
                month_neg=100 * float(np.mean(np.array(mret) < 0)), hit2=100 * float(np.mean(np.array(mret) >= 2)),
                rdd=float((pk - realized).max()), mtm95=float(np.percentile(mdd, 95)), mtm_max=float(max(mdd)),
                day99=float(np.percentile(ddd, 99)), day_max=float(max(ddd)), expo=mx)


def row(name, trades, scale=1.0):
    m = metrics(trades, scale)
    return (f"| {name} | {m['per_month']:.1f}건 | {m['month_mean']:+.2f}% | {m['month_median']:+.2f}% | {m['hit2']:.0f}% | {m['month_neg']:.0f}% | "
            f"{m['rdd']:.2f}% | {m['mtm95']:.2f}% / {m['mtm_max']:.2f}% | {m['day99']:.2f}% / {m['day_max']:.2f}% | {m['expo']:.0f}% |"), m


def main():
    sfx = '_oos' if os.getenv('OOS') == '1' else ''     # OOS=1: 2021-07 이전 데이터로만 고른 셀을 그 이후 5y 에서 평가(표본 밖)
    cells = json.load(open(os.path.join(HERE, 'results', f'fx_cross_cells{sfx}.json'), encoding='utf-8'))
    live, live_keys = live_trades()
    is_new = lambda c: (c['sym'], c['tf'], c['mode']) not in live_keys   # noqa: E731
    scen = {
        "A. 지금(라이브 환율 셀)": live,
        "B. + 메이저 미편입 통과 셀": live + cand_trades(cells, lambda c: c['sym'] in MAJORS and is_new(c)),
        "C. + 크로스 일봉 셀": live + cand_trades(cells, lambda c: c['sym'] not in MAJORS and c['tf'] == '일봉'),
        "D. + 크로스 4h 셀": live + cand_trades(cells, lambda c: c['sym'] not in MAJORS and c['tf'] == '4h'),
        "E. 전부(B+C+D)": live + cand_trades(cells, is_new),
    }
    # F: 라이브 4h·일봉 셀 중 스크리닝에 같은 (심볼·봉·방향) 통과 셀이 있으면 그 파라미터로 교체 + 신규 전부
    scr_keys = {(c['sym'], c['tf'], c['mode']) for c in cells}
    scen["F. E + 기존 셀 파라미터 교체"] = [t for t in live if t[8] not in scr_keys] + cand_trades(cells, lambda c: True)
    L = [f"# 환율 유니버스 포트폴리오 시뮬 — 5y, 사이징 1분·4h {SZ_INTRA}%·일봉 {SZ_DAILY}% (생성 {datetime.now():%Y-%m-%d %H:%M})\n",
         "| 구성 | 월 진입 | 월평균 수익 | 월 중앙 | 2%↑ 달 | 음수 달 | 실현 최대낙폭 | 평가낙폭 월95%/최악 | 일최대 99%/최악 | 최대 동시 노출 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    base = None
    for name, tr in scen.items():
        s, m = row(name, tr)
        L.append(s)
        if base is None:
            base = m
        else:                                   # 평소 평가낙폭을 A 와 같게 맞춘 사이징
            k = base['mtm95'] / m['mtm95'] if m['mtm95'] > 0 else 1.0
            s2, _ = row(f"　└ 같은 평소 낙폭으로 맞춤(사이징 ×{k:.2f})", tr, k)
            L.append(s2)
    # 신규 셀별 기여(E 기준)
    contrib = defaultdict(lambda: [0, 0.0])
    yrs = (max(t[1] for t in scen["E. 전부(B+C+D)"]) - T5Y) / (365.25 * DAY)
    for en, ex, r, side, price, sym, sz, label, _k in scen["E. 전부(B+C+D)"]:
        contrib[label][0] += 1
        contrib[label][1] += r * sz
    L += ["\n## 셀별 5y 기여 (E 구성, 월 기여 큰 순)\n", "| 셀 | 월 진입 | 월 기여 |", "|---|---|---|"]
    for label, (n, s) in sorted(contrib.items(), key=lambda x: -x[1][1]):
        L.append(f"| {label} | {n/yrs/12:.2f}건 | {s/yrs/12:+.3f}% |")
    md = "\n".join(L)
    open(os.path.join(HERE, 'results', f'FX_PORTFOLIO_SIM{sfx}.md'), 'w', encoding='utf-8').write(md)
    print(md)


if __name__ == "__main__":
    main()
