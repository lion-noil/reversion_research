# -*- coding: utf-8 -*-
"""유니버스×책×심볼×분기 상세 성적 — universe_report.py와 동일 트레이드 생성(라이브 97셀),
성적 = 분기별 실현 기여 %p(트레이드 수익률 × 라이브 사이징, 비복리 — 셀 간 비교용), 청산 분기 귀속.
실행: python universe_detail.py → results/UNIVERSE_DETAIL.md"""
import sys, os, json
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache

H = 3600 * 1000; DAY = 86400 * 1000
T5Y = int(datetime(2021, 7, 16, tzinfo=timezone.utc).timestamp() * 1000)
FX = {'USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'}
BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = json.load(open(os.path.join(BASE, 'results', 'live_cells.json'), encoding='utf-8'))

_c = {}
def arrs(sym, iv):
    key = (sym, iv)
    if key not in _c:
        m = _load_cache(sym, iv)
        _c[key] = None if not m else (
            np.array([x['close'] for x in m]), np.array([x['high'] for x in m]),
            np.array([x['low'] for x in m]), np.array([x['start'] for x in m], np.int64))
    return _c[key]

def fee_of(sym): return 0.0002 if sym in FX else 0.0011

def zma(C, win):
    n = len(C); cs = np.concatenate([[0.], np.cumsum(C)]); cs2 = np.concatenate([[0.], np.cumsum(C * C)])
    ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    s = cs[win:] - cs[:-win]; s2 = cs2[win:] - cs2[:-win]; mm = s / win
    ma[win - 1:] = mm; sd[win - 1:] = np.sqrt(np.maximum(s2 / win - mm * mm, 0))
    return np.where(sd > 0, (C - ma) / sd, np.nan), ma, sd

def ema(x, s):
    a = 2 / (s + 1); out = np.copy(x)
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def sigma_cell(book, tag, sym, side, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None: return []
    C, Hh, L, ts = d; n = len(C); fee = fee_of(sym)
    win = int(p['win']); k1 = float(p['k1']); b = float(p['b'])
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
    out = []; last = None; open_ex = []
    for ci in cand:
        if ci + 1 >= n: break
        if last is not None and (ts[ci] - last) < cd: continue
        open_ex = [x for x in open_ex if x > ts[ci]]
        if len(open_ex) >= maxc: continue
        price = C[ci]
        hi = int(np.searchsorted(ts, ts[ci] + hold, 'right'))
        if hi <= ci + 1: continue
        if is_long:
            if entry_high:
                tpb = ma[ci] + b * sd[ci]
                if tpb >= price: continue
                tp = price * (1 + (1 - tpb / price))
            else:
                tp = ma[ci] - b * sd[ci]
                if tp <= price: continue
            sl = None if no_sl else price * (1 - (tp / price - 1))
            TP = Hh[ci + 1:hi] >= tp
            SL = (L[ci + 1:hi] <= sl) if sl is not None else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            if hit.any():
                j = int(np.argmax(hit)); ex = ci + 1 + j; xp = sl if SL[j] else tp
            else:
                ex = hi - 1; xp = C[ex]
            r = xp / price - 1 - fee
        else:
            if entry_high:
                tp = ma[ci] + b * sd[ci]
                if tp >= price: continue
            else:
                tpb = ma[ci] - b * sd[ci]
                if tpb <= price: continue
                tp = price * (1 - (tpb / price - 1))
            sl = None if no_sl else price * (1 + (1 - tp / price))
            TP = L[ci + 1:hi] <= tp
            SL = (Hh[ci + 1:hi] >= sl) if sl is not None else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            if hit.any():
                j = int(np.argmax(hit)); ex = ci + 1 + j; xp = sl if SL[j] else tp
            else:
                ex = hi - 1; xp = C[ex]
            r = (price - xp) / price - fee
        out.append((int(ts[ci]), int(ts[ex]), r))
        last = ts[ci]; open_ex.append(ts[ex])
    return out

def fade_cell(sym, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None: return []
    C, Hh, L, ts = d; n = len(C); fee = fee_of(sym)
    M = int(p['m_min']) if iv == '1' else max(1, int(p['m_min']) // 240)
    thr = float(p['drop_pct']); cd = int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    maxc = int(p.get('max_concurrent') or 200)
    retr = float(p.get('retr_mult') or 0)
    ret = C / np.roll(C, M) - 1; ret[:M] = np.nan
    cand = np.flatnonzero(ret <= -thr)
    out = []; last = None; open_ex = []
    for ci in cand:
        if ci + 1 >= n: break
        if last is not None and (ts[ci] - last) < cd: continue
        open_ex = [x for x in open_ex if x > ts[ci]]
        if len(open_ex) >= maxc: continue
        price = C[ci]
        hi = int(np.searchsorted(ts, ts[ci] + hold, 'right'))
        if hi <= ci + 1: continue
        if retr > 0:
            tp = price * (1 + retr * abs(ret[ci]))
            TP = Hh[ci + 1:hi] >= tp
            if TP.any(): ex = ci + 1 + int(np.argmax(TP)); xp = tp
            else: ex = hi - 1; xp = C[ex]
        else:
            ex = hi - 1; xp = C[ex]
        out.append((int(ts[ci]), int(ts[ex]), xp / price - 1 - fee))
        last = ts[ci]; open_ex.append(ts[ex])
    return out

def ewz_cell(sym, side, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None: return []
    C, Hh, L, ts = d; n = len(C); fee = fee_of(sym)
    s = int(p['ewz_s']); k = float(p['k1']); cd = int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    sd_ = 1 if side == 'long' else -1
    mu = ema(C, s); resid = C - mu; sig = ema(np.abs(resid), s)
    z = resid / np.where(sig == 0, 1e-12, sig)
    cand = np.flatnonzero(z >= k) if sd_ > 0 else np.flatnonzero(z <= -k)
    out = []; last = None
    for ci in cand:
        if ci + 1 >= n: continue
        if last is not None and (ts[ci] - last) < cd: continue
        ex = int(np.searchsorted(ts, ts[ci] + hold, 'right')) - 1
        if ex <= ci: continue
        out.append((int(ts[ci]), int(ts[ex]), sd_ * (C[ex] / C[ci] - 1) - fee))
        last = ts[ci]
    return out

def sweep_cell(sym, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None: return []
    C, Hh, L, ts = d; n = len(C); fee = fee_of(sym)
    N = int(p['sweep_n']); cd = int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    pl = np.full(n, np.nan)
    for i in range(N, n): pl[i] = L[i - N:i].min()
    cand = np.flatnonzero((L < pl) & (C > pl))
    out = []; last = None
    for ci in cand:
        if ci + 1 >= n: continue
        if last is not None and (ts[ci] - last) < cd: continue
        ex = int(np.searchsorted(ts, ts[ci] + hold, 'right')) - 1
        if ex <= ci: continue
        out.append((int(ts[ci]), int(ts[ex]), C[ex] / C[ci] - 1 - fee))
        last = ts[ci]
    return out

def want(uni, sym):
    if uni == 'U1': return sym.endswith('USDT')
    if uni == 'U2': return (not sym.endswith('USDT')) and (sym not in FX)
    return sym in FX

UNI_BOOKS = {
    'U1': {'s11': ('S11', 0.05), 's22': ('S22', 0.05), 's33': ('S33', 0.02)},
    'U2': {'s11m': ('S11', 0.01), 's22m': ('S22', 0.01), 's33m': ('S33', 0.02)},
    'U3': {'s11m': ('S11', 0.01), 's22m': ('S22', 0.01), 's33f': ('S33', 0.05)},
}

def qof(ms):
    t = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return f'{t.year}Q{(t.month - 1) // 3 + 1}'

QUARTERS = [f'{y}Q{q}' for y in range(2021, 2027) for q in range(1, 5)
            if not (y == 2021 and q < 3) and not (y == 2026 and q > 3)]

out_md = ["# 유니버스 × 책 × 심볼 × 분기 상세 성적 (2026-07-16, 라이브 97셀)",
          "",
          "값 = 분기 **실현 기여 %p** (트레이드 수익률 합 × 라이브 사이징, 비복리 — 통합 리포트의 복리 자산곡선과 별개).",
          "청산 분기 귀속 · n=트레이드 수 · 승=승률 · Σ=5년 누적 기여 %p.",
          "사이징: U1 S11/S22 5%·S33 2% | U2 S11/S22 1%(WTI 4h페이드 0.5%)·S33 2% | U3 S11/S22 1%·S33 5%.",
          ""]

for uni, books in UNI_BOOKS.items():
    out_md.append(f"\n## {uni} — {'Bybit 크립토' if uni=='U1' else ('MT5 비환율' if uni=='U2' else '환율 7종')}\n")
    for book, (label, frac) in books.items():
        rows = {}
        for sub in LIVE[book]:
            tag = sub['strategy']; iv = sub['candle_interval']; hold_ms = int(sub['max_hold_sec']) * 1000
            for sym, sides in (sub['params'] or {}).items():
                if not want(uni, sym): continue
                for side, p in sides.items():
                    if tag in ('s1', 's2', 's3', 's4', 's11', 's12'):
                        tl = sigma_cell(book, tag, sym, side, p, iv, hold_ms)
                    elif tag == 's13':
                        tl = fade_cell(sym, p, iv, hold_ms)
                    elif tag == 's14':
                        tl = ewz_cell(sym, side, p, iv, hold_ms)
                    elif tag == 's15':
                        tl = sweep_cell(sym, p, iv, hold_ms)
                    else:
                        tl = []
                    fr = frac
                    if book == 's22m' and tag == 's13' and sym == 'WTI': fr = frac * 0.5
                    d = rows.setdefault(sym, {'q': defaultdict(float), 'n': 0, 'w': 0, 'sum': 0.0})
                    for en, ex, r in tl:
                        if en < T5Y: continue
                        d['q'][qof(ex)] += r * fr * 100
                        d['n'] += 1; d['w'] += (1 if r > 0 else 0); d['sum'] += r * fr * 100
        if not rows: continue
        out_md.append(f"### {uni}·{label} ({book})\n")
        hdr = "| 심볼 | n | 승 | Σ%p | " + " | ".join(q[2:] for q in QUARTERS) + " |"
        sep = "|---|---:|---:|---:|" + "---:|" * len(QUARTERS)
        out_md.append(hdr); out_md.append(sep)
        tot = defaultdict(float); tn = 0; tw = 0; tsum = 0.0
        for sym in sorted(rows, key=lambda s: -rows[s]['sum']):
            d = rows[sym]
            cells = []
            for q in QUARTERS:
                v = d['q'].get(q, 0.0)
                cells.append(f"{v:+.2f}" if abs(v) >= 0.005 else "·")
                tot[q] += v
            wr = f"{100*d['w']/d['n']:.0f}%" if d['n'] else "-"
            out_md.append(f"| {sym} | {d['n']} | {wr} | **{d['sum']:+.2f}** | " + " | ".join(cells) + " |")
            tn += d['n']; tw += d['w']; tsum += d['sum']
        cells = [f"{tot[q]:+.2f}" if abs(tot[q]) >= 0.005 else "·" for q in QUARTERS]
        wr = f"{100*tw/tn:.0f}%" if tn else "-"
        out_md.append(f"| **합계** | {tn} | {wr} | **{tsum:+.2f}** | " + " | ".join(cells) + " |")
        out_md.append("")

path = os.path.join(BASE, 'results', 'UNIVERSE_DETAIL.md')
open(path, 'w', encoding='utf-8').write("\n".join(out_md))
print("저장:", path)
