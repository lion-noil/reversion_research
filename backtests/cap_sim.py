# -*- coding: utf-8 -*-
"""셀 캡 시나리오 시뮬 — 캡 적용 상태로 재시뮬해서 (1)셀별 최대 (2)유니버스 최대중첩·노출 (3)캡으로 놓친 수익 비교.
시나리오: A=캡없음(백테스트 원본) / B=현행(z12·페이드12) / C=제안(z12·페이드 실측+여유).
근사: 캡 초과 진입은 드롭(라이브는 슬롯 나면 재진입 가능 → 실제 손실은 이보다 작음).
유니버스: crypto=Bybit s11(11셀,5%)+s22(8셀,5%) 한 계좌 / mt5=s11m 16셀(1%)."""
import sys, os
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, r"C:\Users\Hyeongeon\PycharmProjects\reversion_research")
sys.path.insert(0, r"C:\Users\Hyeongeon\PycharmProjects\reversion_research\backtests")
from reversion_calibrator import _load_cache

H = 3600 * 1000; DAY = 86400 * 1000
FEE = 0.0011; FX = 0.0002

_c = {}
def arrs(sym, iv='1'):
    key = (sym, iv)
    if key not in _c:
        m = _load_cache(sym, iv)
        _c[key] = (np.array([x['close'] for x in m]), np.array([x['high'] for x in m]),
                   np.array([x['low'] for x in m]), np.array([x['start'] for x in m], np.int64))
    return _c[key]

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

def zcell(sym, win, k1, b, mode, mh_d, cdh, fee=FEE, use_sl=False):
    C, Hh, L, ts = arrs(sym); n = len(C)
    z, ma, sd = zma(C, win)
    cand = np.flatnonzero(z >= k1) if mode == 'tl' else np.flatnonzero(z <= -k1)
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
            if TP.any(): ex = ci + 1 + int(np.argmax(TP)); xp = tp
            else: ex = hi - 1; xp = C[ex]
            r = xp / price - 1 - fee
        elif mode == 'rl':
            tp = ma[ci] - b * sd[ci]
            if tp <= price: continue
            TP = Hh[ci + 1:hi] >= tp
            if TP.any(): ex = ci + 1 + int(np.argmax(TP)); xp = tp
            else: ex = hi - 1; xp = C[ex]
            r = xp / price - 1 - fee
        else:
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price: continue
            pct = tpb / price - 1; tp = price * (1 - pct); sl = price * (1 + pct)
            TP = L[ci + 1:hi] <= tp; SL = (Hh[ci + 1:hi] >= sl) if use_sl else np.zeros(hi - ci - 1, bool)
            hit = SL | TP
            if hit.any():
                j = int(np.argmax(hit)); ex = ci + 1 + j; xp = sl if SL[j] else tp
            else: ex = hi - 1; xp = C[ex]
            r = (price - xp) / price - fee
        out.append((int(ts[ci]), int(ts[ex]), r)); last = ts[ci]
    return out

def fadecell(sym, M, thr, cdh, exit_kind, fee=FEE, cap_h=48, Th=24):
    C, Hh, L, ts = arrs(sym); n = len(C)
    ret = C / np.roll(C, M) - 1; ret[:M] = np.nan
    cand = np.flatnonzero(ret <= -thr); out = []; last = None; cd = int(cdh * H)
    for ci in cand:
        if ci + 1 >= n: break
        if last is not None and (ts[ci] - last) < cd: continue
        price = C[ci]
        if exit_kind == 'retr':
            tp = price * (1 + 1.5 * abs(ret[ci])); hi = int(np.searchsorted(ts, ts[ci] + cap_h * H, 'right'))
            if hi <= ci + 1: continue
            TP = Hh[ci + 1:hi] >= tp
            if TP.any(): ex = ci + 1 + int(np.argmax(TP)); xp = tp
            else: ex = hi - 1; xp = C[ex]
        else:
            ex = int(np.searchsorted(ts, ts[ci] + Th * H, 'right')) - 1
            if ex <= ci: continue
            xp = C[ex]
        out.append((int(ts[ci]), int(ts[ex]), xp / price - 1 - fee)); last = ts[ci]
    return out

# S22 4h 셀 (s11s22_combined.py 라이브 v1 8셀)
S22_HOLD = 15 * DAY
def zcell4(sym, win, k, b, mode, cdh):
    C, Hh, L, ts = arrs(sym, '240'); n = len(C)
    z, ma, sd = zma(C, win)
    cand = np.flatnonzero(z >= k) if mode == 'tl' else np.flatnonzero(z <= -k)
    cand = cand[cand >= win]; out = []; last = None; cd = cdh * H
    for ci in cand:
        if ci + 1 >= n: break
        if last is not None and (ts[ci] - last) < cd: continue
        price = C[ci]; hi = int(np.searchsorted(ts, ts[ci] + S22_HOLD, 'right'))
        if hi <= ci + 1: continue
        if mode == 'tl':
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price: continue
            tp = price * (1 + (1 - tpb / price))
        else:
            tp = ma[ci] - b * sd[ci]
            if tp <= price: continue
        TP = Hh[ci + 1:hi] >= tp
        if TP.any(): ex = ci + 1 + int(np.argmax(TP)); r = tp / price - 1 - FEE
        else: ex = hi - 1; r = C[ex] / price - 1 - FEE
        out.append((int(ts[ci]), int(ts[ex]), r)); last = ts[ci]
    return out
def ewzcell4(sym, s, k, hb, side):
    C, Hh, L, ts = arrs(sym, '240'); n = len(C)
    mu = ema(C, s); resid = C - mu; sig = ema(np.abs(resid), s)
    z = resid / np.where(sig == 0, 1e-12, sig)
    cand = np.flatnonzero(z >= k) if side > 0 else np.flatnonzero(z <= -k)
    out = []; last = -10 ** 9
    for ci in cand:
        if ci + 1 >= n: continue
        if ci - last < 6: continue
        ex = min(ci + hb, n - 1)
        out.append((int(ts[ci]), int(ts[ex]), side * (C[ex] / C[ci] - 1) - FEE)); last = ci
    return out
def fadecell4(sym, M, thr, kind, cap):
    C, Hh, L, ts = arrs(sym, '240'); n = len(C)
    ret = C / np.roll(C, M) - 1; ret[:M] = np.nan
    cand = np.flatnonzero(ret <= -thr); out = []; last = -10 ** 9
    for ci in cand:
        if ci + 1 >= n: continue
        if ci - last < 6: continue
        price = C[ci]; hi = min(ci + 1 + cap, n)
        if kind == 'R':
            tp = price * (1 + 1.5 * abs(ret[ci])); TP = Hh[ci + 1:hi] >= tp
            if TP.any(): ex = ci + 1 + int(np.argmax(TP)); r = tp / price - 1 - FEE
            else: ex = hi - 1; r = C[ex] / price - 1 - FEE
        else:
            ex = hi - 1; r = C[ex] / price - 1 - FEE
        out.append((int(ts[ci]), int(ts[ex]), r)); last = ci
    return out

print('셀 생성 중...', flush=True)
# (셀이름, 종류('z'|'fade'), 트레이드리스트) — 파라미터=현행 라이브 trade_config
BY_S11 = [
    ('BTC-z추',  'z', zcell('BTCUSDT', 1440, 6.0, 0.0, 'tl', 14, 3)),
    ('BTC-z역',  'z', zcell('BTCUSDT', 1320, 5.0, -1.0, 'rl', 14, 1)),
    ('BTC-페',   'fade', fadecell('BTCUSDT', 60, 0.04, 0.5, 'retr', cap_h=48)),
    ('ETH-z추',  'z', zcell('ETHUSDT', 720, 6.0, -3.0, 'tl', 14, 3)),
    ('ETH-페',   'fade', fadecell('ETHUSDT', 30, 0.04, 0.5, 'time', Th=24)),
    ('SOL-z추',  'z', zcell('SOLUSDT', 1440, 5.5, 2.5, 'tl', 14, 3)),
    ('SOL-페',   'fade', fadecell('SOLUSDT', 15, 0.05, 0.5, 'time', Th=24)),
    ('XRP-z숏',  'z', zcell('XRPUSDT', 720, 5.0, -0.5, 'ts', 14, 1, use_sl=True)),
    ('XRP-페',   'fade', fadecell('XRPUSDT', 30, 0.05, 0.5, 'time', Th=24)),
    ('XAUT-z추', 'z', zcell('XAUTUSDT', 1440, 4.0, -2.5, 'tl', 14, 1)),
    ('XAUT-z역', 'z', zcell('XAUTUSDT', 1440, 4.25, -3.0, 'rl', 14, 3)),
]
BY_S22 = [
    ('22BTC-z추',  'z', zcell4('BTCUSDT', 240, 3.25, -2.5, 'tl', 24)),
    ('22ETH-ewzL', 'z', ewzcell4('ETHUSDT', 200, 3.0, 90, 1)),
    ('22ETH-ewzS', 'z', ewzcell4('ETHUSDT', 50, 2.5, 18, -1)),
    ('22SOL-ewzL', 'z', ewzcell4('SOLUSDT', 20, 3.0, 60, 1)),
    ('22SOL-z역',  'z', zcell4('SOLUSDT', 240, 2.75, -1, 'rl', 24)),
    ('22XRP-z추',  'z', zcell4('XRPUSDT', 240, 4.0, -3, 'tl', 12)),
    ('22XRP-z역',  'z', zcell4('XRPUSDT', 240, 3.0, 1.5, 'rl', 12)),
    ('22XRP-페',   'fade', fadecell4('XRPUSDT', 12, 0.15, 'R', 60)),
]
MT5_START = int(datetime(2023, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
MT = [
    ('JP-z추',   'z', zcell('JP225', 1440, 4.0, -1.5, 'tl', 14, 3)),
    ('US1-z추',  'z', zcell('US100', 1440, 5.5, -3.0, 'tl', 14, 3)),
    ('GER-z추',  'z', zcell('GER40', 1440, 3.75, -3.0, 'tl', 14, 3)),
    ('UK-z추',   'z', zcell('UK100', 1440, 3.75, -3.0, 'tl', 14, 3)),
    ('XAG-z추',  'z', zcell('XAGUSD', 1320, 4.75, -2.5, 'tl', 14, 1)),
    ('WTI-z추',  'z', zcell('WTI', 720, 4.5, -2.0, 'tl', 14, 3)),
    ('UJ-z추',   'z', zcell('USDJPY', 1440, 4.5, -2.5, 'tl', 14, 1, fee=FX)),
    ('mBTC-z추', 'z', zcell('BTCUSD', 1440, 6.0, 0.0, 'tl', 14, 3)),
    ('mETH-z추', 'z', zcell('ETHUSD', 720, 6.0, -3.0, 'tl', 14, 3)),
    ('mBTC-z역', 'z', zcell('BTCUSD', 1320, 5.0, -1.0, 'rl', 14, 1)),
    ('HK-z역',   'z', zcell('HK50', 2880, 3.5, 2.0, 'rl', 14, 1)),
    ('JP-페',    'fade', fadecell('JP225', 240, 0.03, 0.5, 'time', Th=48)),
    ('HK-페',    'fade', fadecell('HK50', 120, 0.02, 0.5, 'time', Th=72)),
    ('UJ-페',    'fade', fadecell('USDJPY', 120, 0.01, 0.5, 'time', fee=FX, Th=48)),
    ('mBTC-페',  'fade', fadecell('BTCUSD', 60, 0.04, 0.5, 'retr', cap_h=48)),
    ('mETH-페',  'fade', fadecell('ETHUSD', 30, 0.04, 0.5, 'time', Th=24)),
]
MT = [(nm, kd, [t for t in tr if t[0] >= MT5_START]) for nm, kd, tr in MT]

# 제안 페이드 캡(실측+~30% 여유)
PROP_FADE = {'BTC-페': 16, 'ETH-페': 16, 'SOL-페': 32, 'XRP-페': 16,
             'JP-페': 24, 'HK-페': 36, 'UJ-페': 16, 'mBTC-페': 16, 'mETH-페': 16, '22XRP-페': 16}

def cap_of(scen, name, kind):
    if scen == 'A': return 10 ** 9
    if scen == 'B': return 12
    return 12 if kind == 'z' else PROP_FADE.get(name, 16)   # C

def run(uni_name, books, scen):
    """books: [(frac, cells)] — 한 계좌 합산. 반환: 지표 dict."""
    ev = []   # (t, typ0진입/1청산, book, cellname, r)
    for bi, (frac, cells) in enumerate(books):
        for nm, kd, tr in cells:
            cap = cap_of(scen, nm, kd)
            for en, ex, r in tr:
                ev.append((en, 0, bi, nm, kd, r, ex, cap))
    ev.sort(key=lambda x: x[0])
    # 진입 이벤트를 시간순 처리하며 셀 캡 적용 → 채택된 것만 청산 이벤트 생성
    opens = {}; per_cell = {}; taken = []
    import heapq
    exq = []   # (ex_ts, cell) 청산 힙
    dropped = []; drop_pnl = 0.0
    for en, typ, bi, nm, kd, r, ex, cap in ev:
        while exq and exq[0][0] <= en:
            _, cnm = heapq.heappop(exq); opens[cnm] -= 1
        cur = opens.get(nm, 0)
        if cur >= cap:
            dropped.append((nm, r)); drop_pnl += r
            continue
        opens[nm] = cur + 1
        per_cell[nm] = max(per_cell.get(nm, 0), opens[nm])
        taken.append((en, ex, r, bi, nm))
        heapq.heappush(exq, (ex, nm))
    # 채택분으로 자산·유니버스 중첩·노출 시뮬
    ev2 = []
    for i, (en, ex, r, bi, nm) in enumerate(taken):
        ev2.append((en, 0, i)); ev2.append((ex, 1, i))
    ev2.sort(key=lambda x: (x[0], x[1]))
    eq = 10000.0; open_ = {}; tot = 0.0; ngames = 0
    mx_games = 0; mx_expo = 0.0; peak = eq; mdd = 0.0
    for t, typ, i in ev2:
        if typ == 0:
            frac = books[taken[i][3]][0]
            no = frac * eq; open_[i] = no; tot += no; ngames += 1
            mx_games = max(mx_games, ngames); mx_expo = max(mx_expo, tot / eq)
        else:
            no = open_.pop(i, None)
            if no is None: continue
            eq += no * taken[i][2]; tot -= no; ngames -= 1
            peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    return dict(eq=eq, mdd=mdd, mx_games=mx_games, mx_expo=mx_expo,
                n_taken=len(taken), n_drop=len(dropped), drop_pnl=drop_pnl,
                per_cell=per_cell, dropped=dropped)

def report(uni, books):
    print(f'\n===== {uni} =====')
    base = None
    for scen, label in [('A', 'A 캡없음(백테스트)'), ('B', 'B 현행(z12·페12)'), ('C', 'C 제안(z12·페 실측+여유)')]:
        m = run(uni, books, scen)
        if scen == 'A': base = m
        d_eq = (m['eq'] / base['eq'] - 1) * 100
        print(f"{label:26s} 최종 ${m['eq']:>9,.0f} ({d_eq:+.1f}% vs A) | 낙폭 {m['mdd']*100:4.1f}% | "
              f"유니버스 최대중첩 {m['mx_games']:3d} | 최대노출 {m['mx_expo']*100:4.0f}% | "
              f"진입 {m['n_taken']}건, 드롭 {m['n_drop']}건(놓친 r합 {m['drop_pnl']*100:+.1f}%p)")
        if scen != 'A' and m['n_drop']:
            from collections import Counter
            cnt = Counter(nm for nm, _ in m['dropped'])
            top = ', '.join(f'{k}×{v}' for k, v in cnt.most_common(6))
            print(f'{"":26s} 드롭 상세: {top}')
    return base

b = report('crypto (Bybit 한 계좌: s11 5% + s22 5%)', [(0.05, BY_S11), (0.05, BY_S22)])
print('\n  [참고] s22 셀별 역대최대중첩(캡없음):')
for nm, kd, tr in BY_S22:
    ev = sorted([(en, 1) for en, ex, r in tr] + [(ex, -1) for en, ex, r in tr], key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, d in ev: cur += d; mx = max(mx, cur)
    print(f'    {nm:12s} {len(tr):4d}건 최대 {mx}')
report('mt5 (MT5 비환율+FX 통합 계좌: s11m 1%, 2023-07~)', [(0.01, MT)])
