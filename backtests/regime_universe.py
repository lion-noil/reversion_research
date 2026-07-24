# -*- coding: utf-8 -*-
"""레짐 사이징 유니버스 교차검증 — 방향정합 틸트가 책(1m/4h/일봉)×유니버스(크립토/MT5/FX)
전 조합에서 공통 패턴인지. 라이브 셀 전체(ewz(s14) 2심볼만 제외), 레짐=심볼 자체 일봉 MA200(전일 기준).
변형: BASE / V1(정합 1.4×·불일치 0.6×) / V4(연속 clip(1+0.4·dir·z200, 0.5~1.6)).
평가: 단리 %p 누적(진입순) → 총, MDD, MAR. 데이터: 1m=듀카 3.5~6.3y, 4h=240캐시(MT5 13.5y),
일봉=D13/D10(13.5y)·크립토 D(전기간). fee 0.11%(FX 0.02%).
"""
import sys, os, gc
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache

H = 3600 * 1000
DAY = 86400 * 1000
FXSET = set('EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD'.split())
BYBIT = set('BTCUSDT ETHUSDT SOLUSDT XRPUSDT XAUTUSDT'.split())

def uni_of(sym):
    if sym in BYBIT:
        return '크립토'
    return '환율' if sym in FXSET else 'MT5'

def w_of(book, sym):
    if book == 'D':
        return 2.0 if sym in BYBIT else (11.5 if sym in FXSET else 7.0)
    return 5.0 if sym in BYBIT else (2.3 if sym in FXSET else 3.5)

def fee_of(sym):
    return 0.0002 if sym in FXSET else 0.0011

# ── 셀 정의: (book, itv, sym, kind, spec) ──
#  z: (mode, win, k1, b, cd_ms, use_sl)  mode: tl/ts/rl   fade: (m,drop,retr,hold_ms,cd_ms)
Z, F = 'z', 'fade'
CELLS = []
def z(book, itv, sym, mode, win, k1, b, cd_ms, sl=False):
    CELLS.append((book, itv, sym, Z, (mode, win, k1, b, cd_ms, sl)))
def fd(book, itv, sym, m, drop, retr, hold_ms, cd_ms):
    CELLS.append((book, itv, sym, F, (m, drop, retr, hold_ms, cd_ms)))

# 1분책 (hold 14d)
z('M1','1','BTCUSDT','tl',1440,6.0,0.0,3*H);   z('M1','1','ETHUSDT','tl',720,6.0,-3.0,3*H)
z('M1','1','SOLUSDT','tl',1440,5.5,2.5,3*H);   z('M1','1','XAUTUSDT','tl',1440,4.0,-2.5,1*H,sl=True)
z('M1','1','XRPUSDT','ts',720,5.0,-0.5,1*H,sl=True)
z('M1','1','BTCUSDT','rl',1320,5.0,-1.0,1*H);  z('M1','1','XAUTUSDT','rl',1440,4.25,-3.0,3*H)
fd('M1','1','BTCUSDT',60,0.04,1.5,48*H,1800e3); fd('M1','1','ETHUSDT',30,0.04,0,24*H,1800e3)
fd('M1','1','SOLUSDT',15,0.05,0,24*H,1800e3);   fd('M1','1','XRPUSDT',30,0.05,0,24*H,1800e3)
z('M1','1','JP225','tl',1440,4.0,-1.5,3*H);    z('M1','1','US100','tl',1440,5.5,-3.0,3*H)
z('M1','1','GER40','tl',1440,3.75,-3.0,3*H);   z('M1','1','UK100','tl',1440,3.75,-3.0,3*H)
z('M1','1','XAGUSD','tl',1320,4.75,-2.5,1*H);  z('M1','1','WTI','tl',720,4.5,-2.0,3*H)
z('M1','1','USDJPY','tl',1440,4.5,-2.5,1*H)
z('M1','1','BTCUSD','tl',1440,6.0,0.0,3*H);    z('M1','1','ETHUSD','tl',720,6.0,-3.0,3*H)
z('M1','1','BTCUSD','rl',1320,5.0,-1.0,1*H);   z('M1','1','HK50','rl',2880,3.5,2.0,1*H)
fd('M1','1','JP225',240,0.03,0,48*H,1800e3);   fd('M1','1','HK50',120,0.02,0,72*H,1800e3)
fd('M1','1','USDJPY',120,0.01,0,48*H,1800e3)
fd('M1','1','BTCUSD',60,0.04,1.5,48*H,1800e3); fd('M1','1','ETHUSD',30,0.04,0,24*H,1800e3)
# 4h책 (hold 15d) — ewz(ETH/SOL/XAG s14) 제외
z('H4','240','BTCUSDT','tl',240,3.25,-2.5,24*H); z('H4','240','XRPUSDT','tl',240,4.0,-3.0,12*H)
z('H4','240','SOLUSDT','rl',240,2.75,-1.0,24*H); z('H4','240','XRPUSDT','rl',240,3.0,1.5,12*H)
fd('H4','240','XRPUSDT',12,0.15,1.5,60*4*H,24*H)
z('H4','240','JP225','tl',120,3.0,1.0,12*H);   z('H4','240','HK50','tl',180,3.0,-2.0,12*H)
z('H4','240','US100','rl',180,3.0,1.0,12*H);   z('H4','240','HK50','rl',180,3.0,1.0,12*H)
z('H4','240','USDJPY','rl',120,2.5,1.0,12*H)
z('H4','240','AUDUSD','rs',30,3.0,-2.0,12*H,sl=True); z('H4','240','NZDUSD','rs',60,3.0,-1.0,24*H,sl=True)
fd('H4','240','JP225',18,0.05,0,60*4*H,24*H);  fd('H4','240','US100',30,0.07,1.5,30*4*H,24*H)
fd('H4','240','UK100',12,0.03,0,60*4*H,24*H);  fd('H4','240','XAUUSD',6,0.03,0,60*4*H,24*H)
fd('H4','240','WTI',6,0.07,0,60*4*H,24*H)
# 일봉책 (hold 15d) — regime_sizing.py CELLS 미러
_DC = [
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
for s_, m_, w_, k_, b_, cdd in _DC:
    z('D', 'DD', s_, m_, w_, k_, b_, cdd * DAY, sl=(m_ in ('ts', 'rs')))

HOLD = {'M1': 14 * DAY, 'H4': 15 * DAY, 'D': 15 * DAY}


def load_arrays(sym, itv):
    if itv == 'DD':
        m = _load_cache(sym, 'D') if sym in BYBIT else (_load_cache(sym, 'D13') or _load_cache(sym, 'D10'))
    else:
        m = _load_cache(sym, itv)
    if not m:
        return None
    a = dict(C=np.array([x['close'] for x in m]), H=np.array([x['high'] for x in m]),
             L=np.array([x['low'] for x in m]), ts=np.array([x['start'] for x in m], np.int64))
    del m
    return a


def roll_ma_sd(C, win):
    n = len(C)
    cs = np.concatenate([[0.], np.cumsum(C)])
    cs2 = np.concatenate([[0.], np.cumsum(C * C)])
    ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    s = cs[win:] - cs[:-win]; s2 = cs2[win:] - cs2[:-win]; mm = s / win
    ma[win - 1:] = mm; sd[win - 1:] = np.sqrt(np.maximum(s2 / win - mm * mm, 0))
    return ma, sd


def daily_regime(a):
    """일 단위 MA200 레짐 (전일 종가 기준 no-lookahead). returns (udays, above, z200)"""
    days = a['ts'] // DAY
    ud, idx = np.unique(days, return_index=True)
    ends = np.append(idx[1:], len(days)) - 1
    dc = a['C'][ends]
    ma, sd = roll_ma_sd(dc, 200)
    z = np.where(sd > 0, (dc - ma) / sd, np.nan)
    return ud, dc > ma, z


def regime_at(reg, ts, dirn, variant):
    ud, above, z200 = reg
    i = int(np.searchsorted(ud, ts // DAY)) - 1
    if i < 0 or not np.isfinite(z200[i]):
        return 1.0
    if variant == 'V1':
        match = (dirn > 0 and above[i]) or (dirn < 0 and not above[i])
        return 1.4 if match else 0.6
    if variant == 'V4':
        return float(np.clip(1 + 0.4 * dirn * z200[i], 0.5, 1.6))
    return 1.0


def gen_z(a, mode, win, k1, b, cd_ms, use_sl, hold_ms, fee):
    C, Hh, L, ts = a['C'], a['H'], a['L'], a['ts']
    n = len(C)
    ma, sd = roll_ma_sd(C, win)
    zz = np.where(sd > 0, (C - ma) / sd, np.nan)
    cand = np.flatnonzero(zz >= k1) if mode in ('ts', 'tl', 'rs') and mode != 'rl' else None
    if mode == 'tl' or mode == 'rs':
        cand = np.flatnonzero(zz >= k1)
    elif mode == 'ts':
        cand = np.flatnonzero(zz <= -k1)   # 추세숏 = 하단이탈 숏
    else:  # rl
        cand = np.flatnonzero(zz <= -k1)
    # ⚠️ 방향 규약: tl=상단돌파 롱, ts=하단돌파 숏(추세), rl=하단이탈 롱(역추), rs=상단이탈 숏(역추)
    cand = cand[cand >= win]
    out = []; last = None
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd_ms:
            continue
        price = C[ci]; hi = int(np.searchsorted(ts, ts[ci] + hold_ms, 'right'))
        if hi <= ci + 1:
            continue
        if mode == 'tl':
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price: continue
            pct = 1 - tpb / price; tx = price * (1 + pct); sx = price * (1 - pct)
            TP = Hh[ci + 1:hi] >= tx
            SL = (L[ci + 1:hi] <= sx) if use_sl else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            if hit.any():
                j = int(np.argmax(hit)); r = (sx / price - 1 - fee) if SL[j] else (tx / price - 1 - fee)
            else:
                r = C[hi - 1] / price - 1 - fee
            d = 1
        elif mode == 'ts':
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price: continue
            pct = tpb / price - 1; tx = price * (1 - pct); sx = price * (1 + pct)
            TP = L[ci + 1:hi] <= tx
            SL = (Hh[ci + 1:hi] >= sx) if use_sl else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            if hit.any():
                j = int(np.argmax(hit)); r = ((price - sx) / price - fee) if SL[j] else ((price - tx) / price - fee)
            else:
                r = (price - C[hi - 1]) / price - fee
            d = -1
        elif mode == 'rl':
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price: continue
            sx = price * (1 - (tpb / price - 1))
            TP = Hh[ci + 1:hi] >= tpb
            SL = (L[ci + 1:hi] <= sx) if use_sl else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            if hit.any():
                j = int(np.argmax(hit)); r = (sx / price - 1 - fee) if SL[j] else (tpb / price - 1 - fee)
            else:
                r = C[hi - 1] / price - 1 - fee
            d = 1
        else:  # rs
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price: continue
            sx = price * (1 + (1 - tpb / price))
            TP = L[ci + 1:hi] <= tpb
            SL = (Hh[ci + 1:hi] >= sx) if use_sl else np.zeros(hi - ci - 1, bool)
            hit = TP | SL
            if hit.any():
                j = int(np.argmax(hit)); r = ((price - sx) / price - fee) if SL[j] else ((price - tpb) / price - fee)
            else:
                r = (price - C[hi - 1]) / price - fee
            d = -1
        out.append((int(ts[ci]), float(r), d))
        last = ts[ci]
    return out


def gen_fade(a, m, drop, retr, hold_ms, cd_ms, fee):
    C, Hh, ts = a['C'], a['H'], a['ts']
    n = len(C)
    ret = C / np.roll(C, m) - 1
    ret[:m] = np.nan
    cand = np.flatnonzero(ret <= -drop)
    out = []; last = None
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd_ms:
            continue
        price = C[ci]; hi = int(np.searchsorted(ts, ts[ci] + hold_ms, 'right'))
        if hi <= ci + 1:
            continue
        if retr:
            tp = price * (1 + retr * abs(ret[ci]))
            TP = Hh[ci + 1:hi] >= tp
            r = (tp / price - 1 - fee) if TP.any() else C[hi - 1] / price - 1 - fee
        else:
            r = C[hi - 1] / price - 1 - fee
        out.append((int(ts[ci]), float(r), 1))
        last = ts[ci]
    return out


# ── 실행: (심볼,itv) 단위로 로드→셀 처리→해제 ──
by_symitv = defaultdict(list)
for book, itv, sym, kind, spec in CELLS:
    by_symitv[(sym, itv)].append((book, kind, spec))

trades = []   # (book, uni, en_ts, r, w, dirn, sym)
regs = {}
for (sym, itv), cells in sorted(by_symitv.items()):
    a = load_arrays(sym, itv)
    if a is None:
        print(f"  ⚠️ {sym} {itv}: 데이터 없음 — {len(cells)}셀 스킵")
        continue
    if sym not in regs:
        regs[sym] = daily_regime(a) if itv != '1' else daily_regime(a)
    reg = daily_regime(a)   # 해당 시계열 기준(가장 긴 시계열이면 그대로)
    for book, kind, spec in cells:
        fee = fee_of(sym)
        if kind == Z:
            mode, win, k1, b, cd_ms, sl = spec
            trs = gen_z(a, mode, win, k1, b, cd_ms, sl, HOLD[book], fee)
        else:
            m, drop, retr, hold_ms, cd_ms = spec
            trs = gen_fade(a, m, drop, retr, hold_ms, cd_ms, fee)
        w = w_of(book, sym)
        for en, r, d in trs:
            trades.append((book, uni_of(sym), en, r, w, d, sym, reg))
    del a
    gc.collect()

print(f"총 트레이드 {len(trades)}건\n")
print(f"{'책':4} {'유니버스':6} {'n':>5} | {'BASE MAR':>9} {'V1 MAR':>8} {'V4 MAR':>8} | {'BASE총%p':>9} {'V1총':>7} {'V4총':>7} | 판정")
def evalset(rows, variant):
    if not rows:
        return None
    rows = sorted(rows, key=lambda t: t[2])
    pnl = []
    for book, uni, en, r, w, d, sym, reg in rows:
        mult = regime_at(reg, en, d, variant)
        pnl.append(r * w * mult)
    p = np.array(pnl)
    cum = np.cumsum(p)
    peak = np.maximum.accumulate(cum)
    mdd = float(np.max(peak - cum)) if len(cum) else 0.0
    total = float(cum[-1])
    yrs = (rows[-1][2] - rows[0][2]) / (365.25 * DAY)
    mar = (total / yrs) / mdd if mdd > 0 else float('inf')
    return total, mdd, mar

for book in ('M1', 'H4', 'D'):
    for uni in ('크립토', 'MT5', '환율'):
        rows = [t for t in trades if t[0] == book and t[1] == uni]
        if not rows:
            continue
        rb = evalset(rows, 'BASE'); r1 = evalset(rows, 'V1'); r4 = evalset(rows, 'V4')
        better = sum(1 for x in (r1, r4) if x[2] > rb[2])
        verdict = '✅✅' if better == 2 else ('✅' if better == 1 else '❌')
        print(f"{book:4} {uni:6} {len(rows):>5} | {rb[2]:>9.2f} {r1[2]:>8.2f} {r4[2]:>8.2f} | "
              f"{rb[0]:>+9.1f} {r1[0]:>+7.1f} {r4[0]:>+7.1f} | {verdict}")
    rows = [t for t in trades if t[0] == book]
    rb = evalset(rows, 'BASE'); r1 = evalset(rows, 'V1'); r4 = evalset(rows, 'V4')
    print(f"{book:4} {'전체':6} {len(rows):>5} | {rb[2]:>9.2f} {r1[2]:>8.2f} {r4[2]:>8.2f} | "
          f"{rb[0]:>+9.1f} {r1[0]:>+7.1f} {r4[0]:>+7.1f} |")
    print()
