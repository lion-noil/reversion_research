# -*- coding: utf-8 -*-
"""U2/U3 사이징 스케일 스윕 — 전 책 사이징 ×k 시 MTM 최대낙폭·최저증거금율·최종자산 변화.
U1(5%/5%/2%) 낙폭 56.1%·증거금 25%에 도달하는 k 탐색. 트레이드 1회 생성 후 k만 스윕.
실행: python universe_scale.py U2|U3"""
import sys, os, json
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("udetail", os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_detail.py"))

# universe_detail은 실행형이라 import 대신 함수만 복사 사용 — universe_report와 동일 생성기 재사용을 위해 여기서 직접 재정의
from reversion_calibrator import _load_cache
H = 3600 * 1000; DAY = 86400 * 1000
UNI = (sys.argv[1] if len(sys.argv) > 1 else 'U2').upper()
T5Y = int(datetime(2021, 7, 16, tzinfo=timezone.utc).timestamp() * 1000)
FX = {'USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'}
LIVE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'live_cells.json'), encoding='utf-8'))

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

def sigma_cell(tag, sym, side, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None: return []
    C, Hh, L, ts = d; n = len(C); fee = fee_of(sym)
    win = int(p['win']); k1 = float(p['k1']); b = float(p['b'])
    cd = int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    maxc = int(p.get('max_concurrent') or 200)
    no_sl = bool(p.get('no_sl'))
    trend = tag in ('s1', 's3', 's11'); is_long = side == 'long'
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
            else: ex = hi - 1; xp = C[ex]
            r = xp / price - 1 - fee; sd_ = 1
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
            else: ex = hi - 1; xp = C[ex]
            r = (price - xp) / price - fee; sd_ = -1
        out.append((int(ts[ci]), int(ts[ex]), r, sym, sd_, float(price)))
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
        out.append((int(ts[ci]), int(ts[ex]), xp / price - 1 - fee, sym, 1, float(price)))
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
        out.append((int(ts[ci]), int(ts[ex]), sd_ * (C[ex] / C[ci] - 1) - fee, sym, sd_, float(C[ci])))
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
        out.append((int(ts[ci]), int(ts[ex]), C[ex] / C[ci] - 1 - fee, sym, 1, float(C[ci])))
        last = ts[ci]
    return out

def want(uni, sym):
    if uni == 'U1': return sym.endswith('USDT')
    if uni == 'U2': return (not sym.endswith('USDT')) and (sym not in FX)
    return sym in FX

BOOK_FRAC = {
    'U1': {'s11': 0.05, 's22': 0.05, 's33': 0.02},
    'U2': {'s11m': 0.01, 's22m': 0.01, 's33m': 0.02},
    'U3': {'s11m': 0.01, 's22m': 0.01, 's33f': 0.05},
}[UNI]

trades = []
for book, frac in BOOK_FRAC.items():
    for sub in LIVE[book]:
        tag = sub['strategy']; iv = sub['candle_interval']; hold_ms = int(sub['max_hold_sec']) * 1000
        for sym, sides in (sub['params'] or {}).items():
            if not want(UNI, sym): continue
            for side, p in sides.items():
                if tag in ('s1', 's2', 's3', 's4', 's11', 's12'):
                    tl = sigma_cell(tag, sym, side, p, iv, hold_ms)
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
                trades += [(en, ex, r, sym, sd_, ep, fr) for en, ex, r, sym, sd_, ep in tl]
trades = sorted(t for t in trades if t[0] >= T5Y)
syms = sorted(set(t[3] for t in trades))
print(f'[{UNI}] 트레이드 {len(trades)}건', flush=True)

HL = {}
for sym in syms:
    d = arrs(sym, '1')
    if d is not None:
        C, Hh, L, ts = d
        hr = ts // H
        uh, idx = np.unique(hr, return_index=True)
        HL[sym] = (uh, np.minimum.reduceat(L, idx), np.maximum.reduceat(Hh, idx))
    else:
        HL[sym] = None
H4 = {}
for sym in syms:
    d = arrs(sym, '240')
    if d is not None:
        C, Hh, L, ts = d
        H4[sym] = (ts // H, L, Hh)
def mark(sym, hour, side, fallback):
    v = HL.get(sym)
    if v is not None:
        uh, lo, hi_ = v
        i = int(np.searchsorted(uh, hour, 'right')) - 1
        if 0 <= i < len(uh) and hour - uh[i] <= 4:
            return lo[i] if side > 0 else hi_[i]
    v = H4.get(sym)
    if v is not None:
        uh, lo, hi_ = v
        i = int(np.searchsorted(uh, hour, 'right')) - 1
        if 0 <= i < len(uh):
            return lo[i] if side > 0 else hi_[i]
    return fallback

ev = []
for i, t in enumerate(trades): ev.append((t[0], 0, i)); ev.append((t[1], 1, i))
ev.sort(key=lambda x: (x[0], x[1]))
t0 = T5Y // H; t1 = max(e[0] for e in ev) // H

def sim(scale):
    eq = 10000.0; open_ = {}; expo = 0.0; ptr = 0
    peak = -1e18; mdd = 0.0; min_mg = 9e9; liq = None
    for hour in range(t0, t1 + 1):
        tms = (hour + 1) * H
        while ptr < len(ev) and ev[ptr][0] < tms:
            t, typ, i = ev[ptr]; ptr += 1
            if typ == 0:
                no = min(trades[i][6] * scale, 1.0) * eq
                open_[i] = no; expo += no
            else:
                no = open_.pop(i, None)
                if no is not None: eq += no * trades[i][2]; expo -= no
        un = 0.0
        for i, no in open_.items():
            en, ex, r, sym, side, ep, fr = trades[i]
            m = mark(sym, hour, side, ep)
            un += no * side * (m / ep - 1.0)
        mtm = eq + un
        peak = max(peak, mtm); mdd = max(mdd, (peak - mtm) / peak if peak > 0 else 0)
        if expo > 0:
            mg = mtm / expo
            if mg < min_mg: min_mg = mg
            if mg < 0.005 and liq is None: liq = hour
        if eq <= 0: return eq, mdd, min_mg, liq
    return eq, mdd, min_mg, liq

base = {'U2': '1/1/2%', 'U3': '1/1/5%'}[UNI]
print(f'\nk배 | S11·S22 / S33 진입% | 최종자산 | MTM낙폭 | 최저증거금 | 청산')
for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20):
    eq, mdd, mg, liq = sim(k)
    s11p = 1 * k; s33p = (2 if UNI == 'U2' else 5) * k
    liq_s = datetime.fromtimestamp(liq * H / 1000, tz=timezone.utc).strftime('%Y-%m') if liq else '-'
    print(f'{k:>3} | {s11p:>4}% / {s33p:>4}% | ${eq:>11,.0f} | {mdd*100:>5.1f}% | {mg*100:>6.1f}% | {liq_s}', flush=True)
