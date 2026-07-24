# -*- coding: utf-8 -*-
"""레짐 틸트 MTM 정밀 시뮬 — 일봉책 라이브 셀, BASE/V1/V4 × 계좌(MT5·Bybit).
복리: 진입노셔널 = 진입시점 실현자산 × w% × mult. MTM = 일봉 고/저 장중최악 기준.
리포트: 연도별 실현수익(%), MTM 최대낙폭(%), 최대노출배수, 낙폭 -20/-30/-50% 도달(청산가능성 프록시).
레짐 = 심볼 자체 일봉 MA200(전일 기준). 사이징 = 라이브(크립토 2 / 비FX 7 / FX 11.5%).
"""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache

DAY = 86400 * 1000
HOLD = 15 * DAY
FXSET = set('EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD'.split())
BYBIT = set('BTCUSDT ETHUSDT SOLUSDT XRPUSDT'.split())
INIT = 10_000.0

CELLS = [
    ("BTCUSDT","tl",90,2.5,-2.0,2),("BTCUSDT","ts",90,2.4,0.2,1),("BTCUSDT","rl",150,1.1,0.8,10),
    ("ETHUSDT","tl",60,2.5,-3.0,1),("ETHUSDT","ts",90,1.6,-0.8,5),("ETHUSDT","rl",200,1.9,-0.4,2),("ETHUSDT","rs",200,2.6,2.2,1),
    ("SOLUSDT","tl",90,2.9,-3.0,1),("SOLUSDT","ts",90,1.7,0.2,2),("SOLUSDT","rl",200,1.8,0.4,1),
    ("XRPUSDT","tl",90,3.1,-3.0,1),("XRPUSDT","ts",90,1.6,-1.4,5),("XRPUSDT","rl",200,1.6,-1.4,2),("XRPUSDT","rs",200,1.1,0.2,7),
    ("USDJPY","tl",120,2.9,-1.4,5),
    ("BTCUSD","tl",90,2.5,-2.0,2),("BTCUSD","ts",90,2.4,-0.2,1),
    ("ETHUSD","tl",60,2.8,-3.0,1),("ETHUSD","ts",60,2.0,0.0,2),
    ("XAGUSD","tl",250,2.9,1.0,5),("XAUUSD","tl",150,2.6,0.2,3),
    ("WTI","ts",90,2.3,2.0,1),("JP225","tl",200,2.7,1.4,1),
    ("EURUSD","rl",120,2.9,-0.6,1),
    ("GBPUSD","rl",150,2.4,1.6,1),("GBPUSD","rs",150,2.1,0.6,1),
    ("USDJPY","rl",150,2.1,-3.0,1),
    ("AUDUSD","rl",200,2.6,-3.0,3),("AUDUSD","rs",90,1.8,0.2,3),
    ("USDCAD","rl",200,2.3,1.8,1),("USDCHF","rl",200,2.3,-0.6,1),
    ("NZDUSD","rl",250,2.9,-3.0,1),("NZDUSD","rs",150,1.4,-1.2,2),
    ("BTCUSD","rl",200,1.0,-3.0,5),("ETHUSD","rl",200,1.9,-0.4,1),
    ("XAUUSD","rs",200,2.8,1.2,1),
    ("US100","rl",150,2.0,-3.0,1),("JP225","rl",200,2.0,-3.0,1),
    ("GER40","rl",90,2.6,-3.0,1),
    ("UK100","rl",90,1.7,-0.8,2),("UK100","rs",250,2.3,1.0,5),
    ("HK50","rl",200,1.8,-3.0,3),("HK50","rs",150,2.2,2.0,1),
]

def w_of(sym):
    return 2.0 if sym in BYBIT else (11.5 if sym in FXSET else 7.0)
def fee_of(sym):
    return 0.0002 if sym in FXSET else 0.0011
def acct_of(sym):
    return 'Bybit' if sym in BYBIT else 'MT5'

_d = {}
def arrs(sym):
    if sym not in _d:
        m = _load_cache(sym, 'D') if sym in BYBIT else (_load_cache(sym, 'D13') or _load_cache(sym, 'D10'))
        _d[sym] = dict(C=np.array([x['close'] for x in m]), H=np.array([x['high'] for x in m]),
                       L=np.array([x['low'] for x in m]), ts=np.array([x['start'] for x in m], np.int64))
    return _d[sym]

def roll_ma_sd(C, win):
    n = len(C)
    cs = np.concatenate([[0.], np.cumsum(C)]); cs2 = np.concatenate([[0.], np.cumsum(C * C)])
    ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    s = cs[win:] - cs[:-win]; s2 = cs2[win:] - cs2[:-win]; mm = s / win
    ma[win - 1:] = mm; sd[win - 1:] = np.sqrt(np.maximum(s2 / win - mm * mm, 0))
    return ma, sd

_rg = {}
def regime(sym):
    if sym not in _rg:
        a = arrs(sym)
        ma, sd = roll_ma_sd(a['C'], 200)
        _rg[sym] = (a['C'] > ma, np.where(sd > 0, (a['C'] - ma) / sd, np.nan))
    return _rg[sym]

def mult_of(variant, sym, ci, dirn):
    if variant == 'BASE':
        return 1.0
    ab, z2 = regime(sym)
    i = ci - 1   # 전일 기준
    if i < 0 or not np.isfinite(z2[i]):
        return 1.0
    if variant == 'V1':
        match = (dirn > 0 and ab[i]) or (dirn < 0 and not ab[i])
        return 1.4 if match else 0.6
    return float(np.clip(1 + 0.4 * dirn * z2[i], 0.5, 1.6))

def gen(sym, mode, win, k1, b, cdd):
    """트레이드 (en_ci, ex_ci, dir, entry_px, exit_px, fee) — regime_universe와 동일 모델."""
    a = arrs(sym); C, Hh, L, ts = a['C'], a['H'], a['L'], a['ts']
    n = len(C); fee = fee_of(sym)
    ma, sd = roll_ma_sd(C, win)
    zz = np.where(sd > 0, (C - ma) / sd, np.nan)
    cand = np.flatnonzero(zz >= k1) if mode in ('tl', 'rs') else np.flatnonzero(zz <= -k1)
    cand = cand[cand >= win]
    out = []; last = None; cd = cdd * DAY
    for ci in cand:
        if ci + 1 >= n: break
        if last is not None and (ts[ci] - last) < cd: continue
        price = C[ci]; hi = int(np.searchsorted(ts, ts[ci] + HOLD, 'right'))
        if hi <= ci + 1: continue
        use_sl = mode in ('ts', 'rs')
        if mode == 'tl':
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price: continue
            tx = price * (2 - tpb / price); d = 1
            TP = Hh[ci + 1:hi] >= tx; SL = np.zeros(hi - ci - 1, bool); sx = 0.0
        elif mode == 'ts':
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price: continue
            tx = price * (2 - tpb / price); sx = price * (tpb / price); d = -1
            TP = L[ci + 1:hi] <= tx; SL = Hh[ci + 1:hi] >= sx
        elif mode == 'rl':
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price: continue
            tx = tpb; d = 1
            TP = Hh[ci + 1:hi] >= tx; SL = np.zeros(hi - ci - 1, bool); sx = 0.0
        else:
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price: continue
            tx = tpb; sx = price * (2 - tpb / price); d = -1
            TP = L[ci + 1:hi] <= tx; SL = Hh[ci + 1:hi] >= sx
        hit = TP | SL
        if hit.any():
            j = int(np.argmax(hit)); ex = ci + 1 + j
            xp = sx if (use_sl and SL[j]) else tx
        else:
            ex = hi - 1; xp = C[ex]
        out.append((ci, ex, d, float(price), float(xp), fee))
        last = ts[ci]
    return out
    # ⚠️ ts(추세숏) tx: 미러 대칭 price*(1-(tpb/price-1)) = price*(2-tpb/price)

# 전 심볼 트레이드 생성 + 계좌별 시뮬
all_tr = defaultdict(list)   # acct -> list of (sym, en_ci, ex_ci, d, epx, xpx, fee, w)
for sym, mode, win, k1, b, cdd in CELLS:
    for en, ex, d, epx, xpx, fee in gen(sym, mode, win, k1, b, cdd):
        all_tr[acct_of(sym)].append((sym, en, ex, d, epx, xpx, fee, w_of(sym)))

def yr(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year

for acct in ('MT5', 'Bybit'):
    print(f"\n════ {acct} 계좌 (일봉책 단독, 초기 ${INIT:,.0f}) ════")
    trs = all_tr[acct]
    syms = sorted({t[0] for t in trs})
    # 공통 캘린더
    alldays = sorted(set(int(x) for s in syms for x in arrs(s)['ts'] // DAY))
    day_ix = {dstr: i for i, dstr in enumerate(alldays)}
    sym_day = {s: {int(t // DAY): i for i, t in enumerate(arrs(s)['ts'])} for s in syms}
    print(f"{'변형':5} {'최종자산':>12} {'CAGR':>6} {'MTM최대낙폭':>10} {'최대노출':>8} {'20%↓':>5} {'30%↓':>5} {'50%↓':>5} | 연도별 실현수익%")
    for variant in ('BASE', 'V1', 'V4'):
        opens = defaultdict(list)   # en_day -> trades
        for t in trs:
            sym = t[0]
            en_day = int(arrs(sym)['ts'][t[1]] // DAY)
            opens[en_day].append(t)
        eq = INIT; peak = INIT; mdd = 0.0; maxexp = 0.0
        hit20 = hit30 = hit50 = False
        live = []   # (sym, ex_ci, d, epx, xpx, fee, notional)
        ycap = {}; prev_y = None
        for dkey in alldays:
            # 1) 청산 (오늘 만기/터치 트레이드 실현 — ex_ci의 날짜 == 오늘)
            still = []
            for sym, ex_ci, d, epx, xpx, fee, notional in live:
                ex_day = int(arrs(sym)['ts'][ex_ci] // DAY)
                if ex_day <= dkey:
                    r = (xpx / epx - 1) * d - fee
                    eq += notional * r
                else:
                    still.append((sym, ex_ci, d, epx, xpx, fee, notional))
            live = still
            # 2) 진입 (오늘 종가)
            for t in opens.get(dkey, []):
                sym, en_ci, ex_ci, d, epx, xpx, fee, w = t
                m = mult_of(variant, sym, en_ci, d)
                notional = eq * (w * m) / 100.0
                live.append((sym, ex_ci, d, epx, xpx, fee, notional))
            # 3) MTM 장중최악 + 노출
            unreal = 0.0; expo = 0.0
            for sym, ex_ci, d, epx, xpx, fee, notional in live:
                di = sym_day[sym].get(dkey)
                if di is None:
                    continue
                a = arrs(sym)
                worst = a['L'][di] if d > 0 else a['H'][di]
                unreal += notional * ((worst / epx - 1) * d)
                expo += notional
            mtm = eq + unreal
            peak = max(peak, mtm)
            dd = (peak - mtm) / peak
            mdd = max(mdd, dd)
            if dd >= 0.20: hit20 = True
            if dd >= 0.30: hit30 = True
            if dd >= 0.50: hit50 = True
            if eq > 0:
                maxexp = max(maxexp, expo / eq)
            # 연도 경계 실현자산 기록
            y = datetime.fromtimestamp(dkey * DAY / 1000, tz=timezone.utc).year
            ycap[y] = eq
        years = sorted(ycap)
        yret = []
        prev = INIT
        for y in years:
            yret.append((y, (ycap[y] / prev - 1) * 100))
            prev = ycap[y]
        yrs_span = max(1e-9, (alldays[-1] - alldays[0]) * DAY / (365.25 * DAY * 1000) * 1000)
        cagr = (eq / INIT) ** (1 / yrs_span) - 1 if eq > 0 else -1.0
        ystr = ' '.join(f"{y % 100:02d}:{r:+.0f}" for y, r in yret)
        print(f"{variant:5} {eq:>12,.0f} {cagr * 100:>+5.1f}% {mdd * 100:>9.1f}% {maxexp:>7.2f}x "
              f"{'⚠️' if hit20 else '—':>5} {'🔴' if hit30 else '—':>5} {'💀' if hit50 else '—':>5} | {ystr}")
