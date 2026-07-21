# -*- coding: utf-8 -*-
"""s22m(4h MT5·FX 확장책) 셀별 역대 최대 중첩 실측 — 현행 trade_config 파라미터."""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\Hyeongeon\PycharmProjects\reversion_research")
sys.path.insert(0, r"C:\Users\Hyeongeon\PycharmProjects\reversion_research\backtests")
from reversion_calibrator import _load_cache

H = 3600 * 1000; DAY = 86400 * 1000
HOLD15 = 15 * DAY

_c = {}
def arrs(sym):
    if sym not in _c:
        m = _load_cache(sym, '240')
        _c[sym] = (np.array([x['close'] for x in m]), np.array([x['high'] for x in m]),
                   np.array([x['low'] for x in m]), np.array([x['start'] for x in m], np.int64))
    return _c[sym]

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

def zcell(sym, win, k, b, mode, cdh, use_sl=False):
    """mode: tl(추세롱)/rl(역추롱)/rs(역추숏: z>=+K, TP=MA+Bσ 미러SL)"""
    C, Hh, L, ts = arrs(sym); n = len(C)
    z, ma, sd = zma(C, win)
    cand = np.flatnonzero(z >= k) if mode in ('tl', 'rs') else np.flatnonzero(z <= -k)
    cand = cand[cand >= win]; out = []; last = None; cd = cdh * H
    for ci in cand:
        if ci + 1 >= n: break
        if last is not None and (ts[ci] - last) < cd: continue
        price = C[ci]; hi = int(np.searchsorted(ts, ts[ci] + HOLD15, 'right'))
        if hi <= ci + 1: continue
        if mode == 'tl':
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price: continue
            tp = price * (1 + (1 - tpb / price)); TP = Hh[ci + 1:hi] >= tp
            ex = ci + 1 + int(np.argmax(TP)) if TP.any() else hi - 1
        elif mode == 'rl':
            tp = ma[ci] - b * sd[ci]
            if tp <= price: continue
            TP = Hh[ci + 1:hi] >= tp
            ex = ci + 1 + int(np.argmax(TP)) if TP.any() else hi - 1
        else:  # rs
            tp = ma[ci] + b * sd[ci]
            if tp >= price: continue
            pct = 1 - tp / price; sl = price * (1 + pct)
            TP = L[ci + 1:hi] <= tp
            SL = (Hh[ci + 1:hi] >= sl) if use_sl else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            ex = ci + 1 + int(np.argmax(hit)) if hit.any() else hi - 1
        out.append((int(ts[ci]), int(ts[ex]))); last = ts[ci]
    return out

def ewzcell(sym, s, k, hb, side, rev=False):
    """side=+1 롱/-1 숏. rev=True: 역추(진입부호 반전 — 롱인데 ez<=-K)."""
    C, Hh, L, ts = arrs(sym); n = len(C)
    mu = ema(C, s); resid = C - mu; sig = ema(np.abs(resid), s)
    z = resid / np.where(sig == 0, 1e-12, sig)
    ent_neg = (side < 0) or rev          # 숏(추세) 또는 역추롱 → ez<=-K
    cand = np.flatnonzero(z <= -k) if ent_neg else np.flatnonzero(z >= k)
    out = []; last = -10 ** 9
    for ci in cand:
        if ci + 1 >= n: continue
        if ci - last < 6: continue
        ex = min(ci + hb, n - 1)
        out.append((int(ts[ci]), int(ts[ex]))); last = ci
    return out

def fadecell(sym, M, thr, hb, retr=False):
    C, Hh, L, ts = arrs(sym); n = len(C)
    ret = C / np.roll(C, M) - 1; ret[:M] = np.nan
    cand = np.flatnonzero(ret <= -thr); out = []; last = -10 ** 9
    for ci in cand:
        if ci + 1 >= n: continue
        if ci - last < 6: continue
        price = C[ci]; hi = min(ci + 1 + hb, n)
        if retr:
            tp = price * (1 + 1.5 * abs(ret[ci])); TP = Hh[ci + 1:hi] >= tp
            ex = ci + 1 + int(np.argmax(TP)) if TP.any() else hi - 1
        else:
            ex = hi - 1
        out.append((int(ts[ci]), int(ts[ex]))); last = ci
    return out

def sweepcell(sym, N, hb):
    C, Hh, L, ts = arrs(sym); n = len(C)
    out = []; last = -10 ** 9
    for ci in range(N + 1, n - 1):
        if ci - last < 6: continue
        prior = L[ci - N:ci].min()
        if L[ci] < prior and C[ci] > prior:
            ex = min(ci + hb, n - 1)
            out.append((int(ts[ci]), int(ts[ex]))); last = ci
    return out

def mo(tr):
    ev = sorted([(en, 1) for en, ex in tr] + [(ex, -1) for en, ex in tr], key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, d in ev: cur += d; mx = max(mx, cur)
    return mx

CELLS = [
    ('z추 JP225 (cap?)',    zcell('JP225', 120, 3.0, 1.0, 'tl', 12)),
    ('z추 HK50',            zcell('HK50', 180, 3.0, -2.0, 'tl', 12)),
    ('z역 US100',           zcell('US100', 180, 3.0, 1.0, 'rl', 12)),
    ('z역 HK50',            zcell('HK50', 180, 3.0, 1.0, 'rl', 12)),
    ('z역 USDJPY',          zcell('USDJPY', 120, 2.5, 1.0, 'rl', 12)),
    ('z역숏 AUDUSD(SL유)',  zcell('AUDUSD', 30, 3.0, -2.0, 'rs', 12, use_sl=True)),
    ('z역숏 NZDUSD(SL유)',  zcell('NZDUSD', 60, 3.0, -1.0, 'rs', 24, use_sl=True)),
    ('ewz역롱 XAGUSD',      ewzcell('XAGUSD', 200, 2.5, 30, 1, rev=True)),
    ('ewz숏 XAGUSD',        ewzcell('XAGUSD', 50, 3.0, 18, -1)),
    ('페 JP225',            fadecell('JP225', 18, 0.05, 60)),
    ('페 US100(retr)',      fadecell('US100', 30, 0.07, 30, retr=True)),
    ('페 UK100',            fadecell('UK100', 12, 0.03, 60)),
    ('페 XAUUSD',           fadecell('XAUUSD', 6, 0.03, 60)),
    ('페 WTI',              fadecell('WTI', 6, 0.07, 60)),
    ('스윕 JP225',          sweepcell('JP225', 120, 60)),
    ('스윕 USDCAD',         sweepcell('USDCAD', 120, 60)),
]
print(f"{'셀':22s} {'건수':>5s} {'역대최대중첩':>6s}")
for nm, tr in CELLS:
    print(f"{nm:22s} {len(tr):5d} {mo(tr):6d}")
