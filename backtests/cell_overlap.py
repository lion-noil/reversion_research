# -*- coding: utf-8 -*-
"""셀별 역대 최대 중첩(동시 열린 게임 수) 실측 — 현행 라이브 파라미터(trade_config 2026-07-19) 기준.
목적: z셀 max_concurrent 캡을 '백테스트 최악에서도 안 걸리는' 값으로 정하기 위한 근거."""
import sys, os
import numpy as np

sys.path.insert(0, r"C:\Users\Hyeongeon\PycharmProjects\reversion_research")
sys.path.insert(0, r"C:\Users\Hyeongeon\PycharmProjects\reversion_research\backtests")
from reversion_calibrator import _load_cache

H = 3600 * 1000
DAY = 86400 * 1000

_cache = {}
def arrs(sym):
    if sym not in _cache:
        m = _load_cache(sym, '1')
        _cache[sym] = (np.array([x['close'] for x in m]), np.array([x['high'] for x in m]),
                       np.array([x['low'] for x in m]), np.array([x['start'] for x in m], np.int64))
    return _cache[sym]

def zma(C, win):
    n = len(C)
    cs = np.concatenate([[0.], np.cumsum(C)]); cs2 = np.concatenate([[0.], np.cumsum(C * C)])
    ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    s = cs[win:] - cs[:-win]; s2 = cs2[win:] - cs2[:-win]; mm = s / win
    ma[win - 1:] = mm; sd[win - 1:] = np.sqrt(np.maximum(s2 / win - mm * mm, 0))
    return np.where(sd > 0, (C - ma) / sd, np.nan), ma, sd

def zcell(sym, win, k1, b, mode, mh_d, cdh, use_sl=False):
    C, Hh, L, ts = arrs(sym); n = len(C)
    z, ma, sd = zma(C, win)
    epos = mode == 'tl'
    cand = np.flatnonzero(z >= k1) if epos else np.flatnonzero(z <= -k1)
    cand = cand[cand >= win]
    out = []; last = None; cd = int(cdh * H); mh = int(mh_d * DAY)
    for ci in cand:
        if ci + 1 >= n: break
        if last is not None and (ts[ci] - last) < cd: continue
        price = C[ci]; hi = int(np.searchsorted(ts, ts[ci] + mh, 'right'))
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
        else:  # 추세숏
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price: continue
            pct = tpb / price - 1; tp = price * (1 - pct); sl = price * (1 + pct)
            TP = L[ci + 1:hi] <= tp
            SL = (Hh[ci + 1:hi] >= sl) if use_sl else np.zeros(hi - ci - 1, bool)
            hit = SL | TP
            ex = ci + 1 + int(np.argmax(hit)) if hit.any() else hi - 1
        out.append((int(ts[ci]), int(ts[ex]))); last = ts[ci]
    return out

def fadecell(sym, M, thr, cdh, exit_kind, cap_h=48, Th=24):
    C, Hh, L, ts = arrs(sym); n = len(C)
    ret = C / np.roll(C, M) - 1; ret[:M] = np.nan
    cand = np.flatnonzero(ret <= -thr)
    out = []; last = None; cd = int(cdh * H)
    for ci in cand:
        if ci + 1 >= n: break
        if last is not None and (ts[ci] - last) < cd: continue
        price = C[ci]
        if exit_kind == 'retr':
            tp = price * (1 + 1.5 * abs(ret[ci]))
            hi = int(np.searchsorted(ts, ts[ci] + cap_h * H, 'right'))
            if hi <= ci + 1: continue
            TP = Hh[ci + 1:hi] >= tp
            ex = ci + 1 + int(np.argmax(TP)) if TP.any() else hi - 1
        else:
            ex = int(np.searchsorted(ts, ts[ci] + Th * H, 'right')) - 1
            if ex <= ci: continue
        out.append((int(ts[ci]), int(ts[ex]))); last = ts[ci]
    return out

def max_overlap(intervals):
    ev = []
    for en, ex in intervals:
        ev.append((en, 1)); ev.append((ex, -1))
    ev.sort(key=lambda x: (x[0], x[1]))  # 같은 시각이면 청산(-1) 먼저
    cur = mx = 0
    for _, d in ev:
        cur += d; mx = max(mx, cur)
    return mx

# ── 현행 라이브 파라미터 (trade_config.py 2026-07-19 HEAD) ──
CELLS = [
    # Bybit s11 z추세
    ('BY s11 BTCUSDT 추세롱',  lambda: zcell('BTCUSDT', 1440, 6.0, 0.0, 'tl', 14, 3)),
    ('BY s11 ETHUSDT 추세롱',  lambda: zcell('ETHUSDT', 720, 6.0, -3.0, 'tl', 14, 3)),
    ('BY s11 SOLUSDT 추세롱',  lambda: zcell('SOLUSDT', 1440, 5.5, 2.5, 'tl', 14, 3)),
    ('BY s11 XAUTUSDT 추세롱', lambda: zcell('XAUTUSDT', 1440, 4.0, -2.5, 'tl', 14, 1)),
    ('BY s11 XRPUSDT 추세숏',  lambda: zcell('XRPUSDT', 720, 5.0, -0.5, 'ts', 14, 1, use_sl=True)),
    # Bybit s12 z역추세
    ('BY s12 BTCUSDT 역추롱',  lambda: zcell('BTCUSDT', 1320, 5.0, -1.0, 'rl', 14, 1)),
    ('BY s12 XAUTUSDT 역추롱', lambda: zcell('XAUTUSDT', 1440, 4.25, -3.0, 'rl', 14, 3)),
    # Bybit s13 페이드 (현행 캡 12 — 참고 측정)
    ('BY s13 BTCUSDT 페이드',  lambda: fadecell('BTCUSDT', 60, 0.04, 0.5, 'retr', cap_h=48)),
    ('BY s13 ETHUSDT 페이드',  lambda: fadecell('ETHUSDT', 30, 0.04, 0.5, 'time', Th=24)),
    ('BY s13 SOLUSDT 페이드',  lambda: fadecell('SOLUSDT', 15, 0.05, 0.5, 'time', Th=24)),
    ('BY s13 XRPUSDT 페이드',  lambda: fadecell('XRPUSDT', 30, 0.05, 0.5, 'time', Th=24)),
    # MT5 s11 z추세 (현행: US100 신규 w1440 K5.5 B-3 cd3h)
    ('MT s11 JP225 추세롱',   lambda: zcell('JP225', 1440, 4.0, -1.5, 'tl', 14, 3)),
    ('MT s11 US100 추세롱',   lambda: zcell('US100', 1440, 5.5, -3.0, 'tl', 14, 3)),
    ('MT s11 GER40 추세롱',   lambda: zcell('GER40', 1440, 3.75, -3.0, 'tl', 14, 3)),
    ('MT s11 UK100 추세롱',   lambda: zcell('UK100', 1440, 3.75, -3.0, 'tl', 14, 3)),
    ('MT s11 XAGUSD 추세롱',  lambda: zcell('XAGUSD', 1320, 4.75, -2.5, 'tl', 14, 1)),
    ('MT s11 WTI 추세롱',     lambda: zcell('WTI', 720, 4.5, -2.0, 'tl', 14, 3)),
    ('MT s11 USDJPY 추세롱',  lambda: zcell('USDJPY', 1440, 4.5, -2.5, 'tl', 14, 1)),
    ('MT s11 BTCUSD 추세롱',  lambda: zcell('BTCUSD', 1440, 6.0, 0.0, 'tl', 14, 3)),
    ('MT s11 ETHUSD 추세롱',  lambda: zcell('ETHUSD', 720, 6.0, -3.0, 'tl', 14, 3)),
    # MT5 s12 z역추세
    ('MT s12 BTCUSD 역추롱',  lambda: zcell('BTCUSD', 1320, 5.0, -1.0, 'rl', 14, 1)),
    ('MT s12 HK50 역추롱',    lambda: zcell('HK50', 2880, 3.5, 2.0, 'rl', 14, 1)),
    # MT5 s13 페이드 (현행 캡 12 — 참고 측정)
    ('MT s13 JP225 페이드',   lambda: fadecell('JP225', 240, 0.03, 0.5, 'time', Th=48)),
    ('MT s13 HK50 페이드',    lambda: fadecell('HK50', 120, 0.02, 0.5, 'time', Th=72)),
    ('MT s13 USDJPY 페이드',  lambda: fadecell('USDJPY', 120, 0.01, 0.5, 'time', Th=48)),
    ('MT s13 BTCUSD 페이드',  lambda: fadecell('BTCUSD', 60, 0.04, 0.5, 'retr', cap_h=48)),
    ('MT s13 ETHUSD 페이드',  lambda: fadecell('ETHUSD', 30, 0.04, 0.5, 'time', Th=24)),
]

print(f"{'셀':34s} {'거래수':>6s} {'역대최대중첩':>8s}")
worst_z = 0
for name, fn in CELLS:
    tr = fn()
    mo = max_overlap(tr)
    if ('s11' in name or 's12' in name):
        worst_z = max(worst_z, mo)
    print(f"{name:34s} {len(tr):6d} {mo:8d}")
print(f"\nz셀(s11/s12) 전셀 역대 최대중첩: {worst_z}")
