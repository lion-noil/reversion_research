# -*- coding: utf-8 -*-
"""라이브 vs 백테스트 수렴 점검 (2026-09-26) — 셀별 실체결이 5년 기대치 분포 안에 있는가.

백테스트: universe_symbolcap 동일 엔진(시그마/ewz/페이드/스윕)으로 라이브 파라미터(live_cells.json) 5y(2021-07-16~) 실행
  → 건당 기대%·표준편차·승률·연간 건수. 라이브 표본 n개와 같은 크기의 연속 윈도 평균 분포에서 라이브 평균의 백분위를 구함
  (거래 자기상관을 반영한 '이 정도 표본이면 이 정도 흔들림이 정상' 기준).
라이브: Supabase trade_records EXIT(전략태그 박제 2026-08-20~), 건당% = 진입/청산가 기반 − 0.11.
계좌 기대 월수익 = Σ 셀 (연간건수/12 × 기대% × 사이징%) — FINAL_PARAMS 사이징(U1 5/2%, U2 3.5/7%, U3 2.3/11.5%, WTI 4h페이드 1.75%).
usage: py -3 backtests/live_vs_backtest.py   (위치: reversion_research) → results/LIVE_VS_BACKTEST.md
"""
import json
import os
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from reversion_calibrator import _load_cache  # noqa: E402

KST = timezone(timedelta(hours=9))
DAY = 86400 * 1000
T5Y = int(datetime(2021, 7, 16, tzinfo=timezone.utc).timestamp() * 1000)
LIVE_FROM = datetime(2026, 8, 20, tzinfo=KST)          # 전략태그 박제 시작
FX = {'USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'}
LIVE = json.load(open(os.path.join(HERE, 'results', 'live_cells.json'), encoding='utf-8'))
FEE_LIVE = 0.11


def env_val(path, key):
    for line in open(path, encoding="utf-8-sig"):
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return ""


SB_URL = env_val(r"C:\Users\Hyeongeon\PycharmProjects\News_scrap\app\.env", "SUPABASE_URL")
SB_KEY = (env_val(r"C:\Users\Hyeongeon\PycharmProjects\News_scrap\app\.env", "SUPABASE_SECRET_KEY")
          or env_val(r"C:\Users\Hyeongeon\PycharmProjects\News_scrap\app\.env", "SUPABASE_KEY"))

# ── 엔진 (universe_symbolcap 동일) ──
_c = {}


def arrs(sym, iv):
    key = (sym, iv)
    if key not in _c:
        m = _load_cache(sym, iv)
        _c[key] = None if not m else (
            np.array([x['close'] for x in m]), np.array([x['high'] for x in m]),
            np.array([x['low'] for x in m]), np.array([x['start'] for x in m], np.int64))
    return _c[key]


def fee_of(sym):
    return 0.0002 if sym in FX else 0.0011


def zma(C, win):
    n = len(C)
    cs = np.concatenate([[0.], np.cumsum(C)])
    cs2 = np.concatenate([[0.], np.cumsum(C * C)])
    ma = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    s = cs[win:] - cs[:-win]
    s2 = cs2[win:] - cs2[:-win]
    mm = s / win
    ma[win - 1:] = mm
    sd[win - 1:] = np.sqrt(np.maximum(s2 / win - mm * mm, 0))
    return np.where(sd > 0, (C - ma) / sd, np.nan), ma, sd


def ema(x, s):
    a = 2 / (s + 1)
    out = np.copy(x)
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def sigma_cell(tag, sym, side, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None:
        return []
    C, Hh, L, ts = d
    n = len(C)
    fee = fee_of(sym)
    win, k1, b = int(p['win']), float(p['k1']), float(p['b'])
    cd = int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    maxc = int(p.get('max_concurrent') or 200)
    no_sl = bool(p.get('no_sl'))
    trend = tag in ('s1', 's3', 's11')
    is_long = side == 'long'
    entry_high = is_long if trend else (not is_long)
    z, ma, sd = zma(C, win)
    cand = np.flatnonzero(z >= k1) if entry_high else np.flatnonzero(z <= -k1)
    cand = cand[cand >= win]
    out, last, open_ex = [], None, []
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd:
            continue
        open_ex = [x for x in open_ex if x > ts[ci]]
        if len(open_ex) >= maxc:
            continue
        price = C[ci]
        hi = int(np.searchsorted(ts, ts[ci] + hold, 'right'))
        if hi <= ci + 1:
            continue
        if is_long:
            if entry_high:
                tpb = ma[ci] + b * sd[ci]
                if tpb >= price:
                    continue
                tp = price * (1 + (1 - tpb / price))
            else:
                tp = ma[ci] - b * sd[ci]
                if tp <= price:
                    continue
            sl = None if no_sl else price * (1 - (tp / price - 1))
            TP = Hh[ci + 1:hi] >= tp
            SL = (L[ci + 1:hi] <= sl) if sl is not None else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            if hit.any():
                j = int(np.argmax(hit))
                ex = ci + 1 + j
                xp = sl if SL[j] else tp
            else:
                ex = hi - 1
                xp = C[ex]
            r = xp / price - 1 - fee
        else:
            if entry_high:
                tp = ma[ci] + b * sd[ci]
                if tp >= price:
                    continue
            else:
                tpb = ma[ci] - b * sd[ci]
                if tpb <= price:
                    continue
                tp = price * (1 - (tpb / price - 1))
            sl = None if no_sl else price * (1 + (1 - tp / price))
            TP = L[ci + 1:hi] <= tp
            SL = (Hh[ci + 1:hi] >= sl) if sl is not None else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            if hit.any():
                j = int(np.argmax(hit))
                ex = ci + 1 + j
                xp = sl if SL[j] else tp
            else:
                ex = hi - 1
                xp = C[ex]
            r = (price - xp) / price - fee
        out.append((int(ts[ci]), int(ts[ex]), r))
        last = ts[ci]
        open_ex.append(ts[ex])
    return out


def fade_cell(sym, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None:
        return []
    C, Hh, L, ts = d
    n = len(C)
    fee = fee_of(sym)
    M = max(1, int(p['m_min']))
    thr, cd = float(p['drop_pct']), int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    maxc = int(p.get('max_concurrent') or 200)
    retr = float(p.get('retr_mult') or 0)
    ret = C / np.roll(C, M) - 1
    ret[:M] = np.nan
    cand = np.flatnonzero(ret <= -thr)
    out, last, open_ex = [], None, []
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd:
            continue
        open_ex = [x for x in open_ex if x > ts[ci]]
        if len(open_ex) >= maxc:
            continue
        price = C[ci]
        hi = int(np.searchsorted(ts, ts[ci] + hold, 'right'))
        if hi <= ci + 1:
            continue
        if retr > 0:
            tp = price * (1 + retr * abs(ret[ci]))
            TP = Hh[ci + 1:hi] >= tp
            if TP.any():
                ex = ci + 1 + int(np.argmax(TP))
                xp = tp
            else:
                ex = hi - 1
                xp = C[ex]
        else:
            ex = hi - 1
            xp = C[ex]
        out.append((int(ts[ci]), int(ts[ex]), xp / price - 1 - fee))
        last = ts[ci]
        open_ex.append(ts[ex])
    return out


def ewz_cell(sym, side, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None:
        return []
    C, Hh, L, ts = d
    n = len(C)
    fee = fee_of(sym)
    s, k, cd = int(p['ewz_s']), float(p['k1']), int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    rev = bool(p.get('ewz_rev'))
    sd_ = 1 if side == 'long' else -1
    mu = ema(C, s)
    resid = C - mu
    sig = ema(np.abs(resid), s)
    z = resid / np.where(sig == 0, 1e-12, sig)
    ent_neg = (sd_ < 0) or rev
    cand = np.flatnonzero(z <= -k) if ent_neg else np.flatnonzero(z >= k)
    out, last = [], None
    for ci in cand:
        if ci + 1 >= n:
            continue
        if last is not None and (ts[ci] - last) < cd:
            continue
        ex = int(np.searchsorted(ts, ts[ci] + hold, 'right')) - 1
        if ex <= ci:
            continue
        out.append((int(ts[ci]), int(ts[ex]), sd_ * (C[ex] / C[ci] - 1) - fee))
        last = ts[ci]
    return out


def sweep_cell(sym, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None:
        return []
    C, Hh, L, ts = d
    n = len(C)
    fee = fee_of(sym)
    N = int(p['sweep_n'])
    cd = int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    pl = np.full(n, np.nan)
    for i in range(N, n):
        pl[i] = L[i - N:i].min()
    cand = np.flatnonzero((L < pl) & (C > pl))
    out, last = [], None
    for ci in cand:
        if ci + 1 >= n:
            continue
        if last is not None and (ts[ci] - last) < cd:
            continue
        ex = int(np.searchsorted(ts, ts[ci] + hold, 'right')) - 1
        if ex <= ci:
            continue
        out.append((int(ts[ci]), int(ts[ex]), C[ex] / C[ci] - 1 - fee))
        last = ts[ci]
    return out


def run_cell(strat, sym, side, p, iv, hold_ms):
    if strat == 's13':
        return fade_cell(sym, p, iv, hold_ms)
    if strat == 's14':
        return ewz_cell(sym, side, p, iv, hold_ms)
    if strat == 's15':
        return sweep_cell(sym, p, iv, hold_ms)
    return sigma_cell(strat, sym, side, p, iv, hold_ms)


# ── 라이브 ──
def fetch_live():
    rows, off = [], 0
    base = (f"{SB_URL}/rest/v1/trade_records?select=id,day,symbol,side,kind,signal,pnl,raw_json"
            f"&day=gte.{LIVE_FROM:%Y-%m-%d}&order=id")
    hdr = {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"}
    while True:
        d = json.load(urllib.request.urlopen(urllib.request.Request(base + f"&limit=1000&offset={off}", headers=hdr), timeout=40))
        rows.extend(d)
        if len(d) < 1000:
            break
        off += 1000
    return rows


def book_of(account, ns):
    n = (ns or "").lower()
    if account == "MT5":
        if n.startswith("s11"):
            return "s11m"
        if n.startswith("s22"):
            return "s22m"
        return "s33m" if n in ("mt5", "mt5d") else ("s33f" if n == "fxd" else "?")
    if n.startswith("s11"):
        return "s11"
    if n.startswith("s22"):
        return "s22"
    return "s33" if n in ("bybit", "cryptod") else "?"


def sizing_pct(book, strat, sym):
    """FINAL_PARAMS 라이브 사이징(명목 % of equity)"""
    daily = book in ("s33", "s33m", "s33f")
    if book in ("s11", "s22", "s33"):          # U1 Bybit
        return 2.0 if daily else 5.0
    if sym in FX:                               # U3
        return 28.75 if daily else 5.75        # 2026-10-05 환율 전용 계좌 분리 + ×2.5 (구 11.5/2.3)
    if strat == 's13' and sym == 'WTI' and book == 's22m':
        return 1.75
    return 7.0 if daily else 3.5                # U2


def main():
    live_rows = fetch_live()
    live = defaultdict(list)
    for r in live_rows:
        raw = r.get("raw_json") or {}
        if not isinstance(raw, dict) or str(r.get("kind") or "").upper() != "EXIT":
            continue
        tag = str(raw.get("strategy_tag") or "").upper()
        ns = raw.get("signal_ns") or ""
        if not tag.startswith("S") or not ns:
            continue
        account = str(raw.get("account") or "BYBIT")
        book = book_of(account, ns)
        strat = tag.lower()
        side = str(r.get("side") or "").lower()
        try:
            ep, xp = float(raw.get("entry_price") or 0), float(raw.get("exit_price") or 0)
        except (TypeError, ValueError):
            continue
        if ep <= 0 or xp <= 0:
            continue
        pct = ((xp / ep - 1) if side == "long" else (1 - xp / ep)) * 100 - FEE_LIVE
        ts = int(raw.get("ts_ms") or 0)
        live[(book, strat, r["symbol"], side)].append((ts, pct))

    now_ms = int(datetime.now(KST).timestamp() * 1000)
    live_days = (now_ms - int(LIVE_FROM.timestamp() * 1000)) / DAY
    out = [f"# 라이브 vs 백테스트 수렴 점검 — 라이브 {LIVE_FROM:%m-%d}~{datetime.now(KST):%m-%d} ({live_days:.0f}일) · 백테 5y(2021-07-16~) 동일 엔진 (생성 {datetime.now(KST):%Y-%m-%d %H:%M})\n",
           "판정: 백분위 = 백테 거래열에서 라이브와 같은 n개 연속 윈도 평균 분포 중 라이브 평균의 위치. 25~75 수렴 · 5~95 정상 · 그 밖 이탈. n<3 표본부족. "
           "z = (라이브평균−백테평균)/(백테σ/√n). 기대 n = 백테 연간건수 × 라이브 일수/365. 계좌 기대월% = Σ 연간건수/12 × 기대% × 사이징%.\n",
           "| 셀 | 라이브 n | 건당% | 승률 | 백테 건당% | 승률 | 연간n | 5y n | 기대 n | z | 백분위 | 판정 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    acct_exp = defaultdict(float)
    acct_live = defaultdict(lambda: [0, 0.0])
    summary = defaultdict(int)
    rows_out = []
    for book, subs in LIVE.items():
        for sub in subs:
            strat, iv, hold_ms = sub['strategy'], sub['candle_interval'], int(sub['max_hold_sec']) * 1000
            for sym, sides in (sub['params'] or {}).items():
                for side, p in sides.items():
                    if float(p.get('k1', 0) or 0) >= 99:
                        continue   # 드레인 셀
                    tr = [t for t in run_cell(strat, sym, side, p, iv, hold_ms) if t[0] >= T5Y]
                    if not tr:
                        continue
                    a = np.array([t[2] for t in tr]) * 100
                    yrs = max(0.5, (tr[-1][1] - T5Y) / (365.25 * DAY))
                    per_yr = len(tr) / yrs
                    acct = "BYBIT" if book in ("s11", "s22", "s33") else "MT5"
                    acct_exp[acct] += per_yr / 12 * a.mean() * sizing_pct(book, strat, sym) / 100
                    lv = live.get((book, strat, sym, side), [])
                    n_live = len(lv)
                    exp_n = per_yr * live_days / 365
                    if n_live == 0:
                        rows_out.append((0, f"| {book} {strat} {sym} {side} | 0 | | | {a.mean():+.2f} | {100*(a>0).mean():.0f}% | {per_yr:.0f} | {len(tr)} | {exp_n:.1f} | | | {'무거래(기대 ' + f'{exp_n:.1f}' + ')' if exp_n >= 2 else '무거래·정상'} |"))
                        summary['무거래'] += 1
                        continue
                    lm = float(np.mean([x[1] for x in lv]))
                    lw = 100 * float(np.mean([x[1] > 0 for x in lv]))
                    acct_live[acct][0] += n_live
                    acct_live[acct][1] += sum(x[1] for x in lv)
                    z = (lm - a.mean()) / (a.std(ddof=1) / np.sqrt(n_live)) if len(a) > 2 and a.std(ddof=1) > 0 else float('nan')
                    if n_live >= 3 and len(a) >= 2 * n_live:
                        w = np.convolve(a, np.ones(n_live) / n_live, mode='valid')
                        pctl = 100 * float((w < lm).mean())
                        verdict = "수렴" if 25 <= pctl <= 75 else ("정상" if 5 <= pctl <= 95 else ("이탈↑" if pctl > 95 else "이탈↓"))
                    else:
                        pctl, verdict = None, "표본부족"
                    summary[verdict] += 1
                    rows_out.append((n_live, f"| {book} {strat} {sym} {side} | {n_live} | {lm:+.2f} | {lw:.0f}% | {a.mean():+.2f} | {100*(a>0).mean():.0f}% | {per_yr:.0f} | {len(tr)} | {exp_n:.1f} | "
                                     f"{'' if np.isnan(z) else f'{z:+.1f}'} | {'' if pctl is None else f'{pctl:.0f}'} | {verdict} |"))
    for _, s in sorted(rows_out, key=lambda x: -x[0]):
        out.append(s)
    out.append("\n## 요약\n")
    out.append("- 판정 분포: " + " · ".join(f"{k} {v}" for k, v in sorted(summary.items(), key=lambda x: -x[1])))
    for acct in ("BYBIT", "MT5"):
        n, s = acct_live[acct]
        out.append(f"- **{acct}** 백테 기대 월수익(사이징 반영, 단순합) ≈ **{acct_exp[acct]:+.2f}%/월** · 라이브 {live_days:.0f}일 태그셀 청산 {n}건 건당평균 {(s/n if n else 0):+.2f}%")
    md = "\n".join(out)
    path = os.path.join(HERE, 'results', 'LIVE_VS_BACKTEST.md')
    open(path, 'w', encoding='utf-8').write(md)
    print(md)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
