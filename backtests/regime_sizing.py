# -*- coding: utf-8 -*-
"""레짐 조건부 사이징 실험 — S33 라이브 셀 전체, 10년(가용 전기간).
질문: 진입비율을 추세(레짐)에 따라 매번 다르게 하면 고정 사이징보다 나은가?
베이스 = 라이브 사이징(크립토 2%·MT5 비FX 7%·FX 11.5%) × mult(레짐).
변형:
  BASE 고정 1.0
  V1 방향정합: 진입방향과 MA200 레짐 일치 시 1.4× / 불일치 0.6×
  V2 전략별: s3(추세)=V1 규칙, s4(역추세)=반대(하방에서 역추롱 1.4×: "더 싸게 살 때 크게")
  V3 vol타깃: mult = clip(중앙값σ20/σ20, 0.5, 2.0) — 저변동 크게·고변동 작게
  V4 연속 z200: mult = clip(1 + 0.4·dir·z200, 0.5, 1.6)
공정성: 각 변형을 '평균 mult=1'로 정규화한 버전도 병기(총 노출 동일 조건 비교).
평가: 단리 누적(Σ r×w×mult) 일별 시리즈 → 총%p, MDD, MAR(연수익/MDD), 연도별 합.
데이터: MT5=D13(13.5y, 없으면 D10) / 크립토=D(전기간). fee 0.11%(FX 0.02%). 보유≤15d.
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
CRYPTO_BYBIT = set('BTCUSDT ETHUSDT SOLUSDT XRPUSDT'.split())

# (sym, strat, side, win, k1, b, cd_days)  — trade_config 라이브 셀 미러(드레인 k1=99 제외)
CELLS = [
    # 크립토 일봉 (bybit ns)
    ("BTCUSDT", "s3", "L", 90, 2.5, -2.0, 2), ("BTCUSDT", "s3", "S", 90, 2.4, 0.2, 1),
    ("BTCUSDT", "s4", "L", 150, 1.1, 0.8, 10),
    ("ETHUSDT", "s3", "L", 60, 2.5, -3.0, 1), ("ETHUSDT", "s3", "S", 90, 1.6, -0.8, 5),
    ("ETHUSDT", "s4", "L", 200, 1.9, -0.4, 2), ("ETHUSDT", "s4", "S", 200, 2.6, 2.2, 1),
    ("SOLUSDT", "s3", "L", 90, 2.9, -3.0, 1), ("SOLUSDT", "s3", "S", 90, 1.7, 0.2, 2),
    ("SOLUSDT", "s4", "L", 200, 1.8, 0.4, 1),
    ("XRPUSDT", "s3", "L", 90, 3.1, -3.0, 1), ("XRPUSDT", "s3", "S", 90, 1.6, -1.4, 5),
    ("XRPUSDT", "s4", "L", 200, 1.6, -1.4, 2), ("XRPUSDT", "s4", "S", 200, 1.1, 0.2, 7),
    # MT5 일봉 S3
    ("USDJPY", "s3", "L", 120, 2.9, -1.4, 5),
    ("BTCUSD", "s3", "L", 90, 2.5, -2.0, 2), ("BTCUSD", "s3", "S", 90, 2.4, -0.2, 1),
    ("ETHUSD", "s3", "L", 60, 2.8, -3.0, 1), ("ETHUSD", "s3", "S", 60, 2.0, 0.0, 2),
    ("XAGUSD", "s3", "L", 250, 2.9, 1.0, 5), ("XAUUSD", "s3", "L", 150, 2.6, 0.2, 3),
    ("WTI", "s3", "S", 90, 2.3, 2.0, 1), ("JP225", "s3", "L", 200, 2.7, 1.4, 1),
    # MT5 일봉 S4
    ("EURUSD", "s4", "L", 120, 2.9, -0.6, 1),
    ("GBPUSD", "s4", "L", 150, 2.4, 1.6, 1), ("GBPUSD", "s4", "S", 150, 2.1, 0.6, 1),
    ("USDJPY", "s4", "L", 150, 2.1, -3.0, 1),
    ("AUDUSD", "s4", "L", 200, 2.6, -3.0, 3), ("AUDUSD", "s4", "S", 90, 1.8, 0.2, 3),
    ("USDCAD", "s4", "L", 200, 2.3, 1.8, 1), ("USDCHF", "s4", "L", 200, 2.3, -0.6, 1),
    ("NZDUSD", "s4", "L", 250, 2.9, -3.0, 1), ("NZDUSD", "s4", "S", 150, 1.4, -1.2, 2),
    ("BTCUSD", "s4", "L", 200, 1.0, -3.0, 5), ("ETHUSD", "s4", "L", 200, 1.9, -0.4, 1),
    ("XAUUSD", "s4", "S", 200, 2.8, 1.2, 1),
    ("US100", "s4", "L", 150, 2.0, -3.0, 1), ("JP225", "s4", "L", 200, 2.0, -3.0, 1),
    ("GER40", "s4", "L", 90, 2.6, -3.0, 1),
    ("UK100", "s4", "L", 90, 1.7, -0.8, 2), ("UK100", "s4", "S", 250, 2.3, 1.0, 5),
    ("HK50", "s4", "L", 200, 1.8, -3.0, 3), ("HK50", "s4", "S", 150, 2.2, 2.0, 1),
]

MODE = {("s3", "L"): "trendlong", ("s3", "S"): "trend", ("s4", "L"): "long", ("s4", "S"): "short"}
DIRSGN = {"L": 1.0, "S": -1.0}


def base_w(sym):
    if sym in CRYPTO_BYBIT:
        return 2.0
    return 11.5 if sym in FXSET else 7.0


def fee_of(sym):
    return 0.0002 if sym in FXSET else 0.0011


def load(sym):
    if sym in CRYPTO_BYBIT:
        return _load_cache(sym, 'D')
    m = _load_cache(sym, 'D13')
    return m if m else _load_cache(sym, 'D10')


_data = {}
def arrs(sym):
    if sym not in _data:
        m = load(sym)
        if not m:
            _data[sym] = None
        else:
            _data[sym] = dict(
                C=np.array([x['close'] for x in m]), H=np.array([x['high'] for x in m]),
                L=np.array([x['low'] for x in m]), ts=np.array([x['start'] for x in m], np.int64))
    return _data[sym]


def roll_ma_sd(C, win):
    n = len(C)
    cs = np.concatenate([[0.], np.cumsum(C)])
    cs2 = np.concatenate([[0.], np.cumsum(C * C)])
    ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    s = cs[win:] - cs[:-win]; s2 = cs2[win:] - cs2[:-win]; mm = s / win
    ma[win - 1:] = mm; sd[win - 1:] = np.sqrt(np.maximum(s2 / win - mm * mm, 0))
    return ma, sd


def gen(sym, mode, win, k1, b, cd_days, fee):
    d = arrs(sym)
    if d is None:
        return []
    C, Hh, L, ts = d['C'], d['H'], d['L'], d['ts']
    n = len(C)
    ma, sd = roll_ma_sd(C, win)
    z = np.where(sd > 0, (C - ma) / sd, np.nan)
    cand = np.flatnonzero(z >= k1) if mode in ('short', 'trendlong') else np.flatnonzero(z <= -k1)
    cand = cand[cand >= win]
    out = []; last = None; cd = cd_days * DAY
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd:
            continue
        price = C[ci]; hi = int(np.searchsorted(ts, ts[ci] + HOLD, 'right'))
        if hi <= ci + 1:
            continue
        if mode == 'trendlong':
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price: continue
            pct = 1 - tpb / price; tx = price * (1 + pct); sx = price * (1 - pct)
            SL = L[ci + 1:hi] <= sx; TP = Hh[ci + 1:hi] >= tx; hit = SL | TP
            if hit.any(): j = int(np.argmax(hit)); ex = ci + 1 + j; r = (sx / price - 1 - fee) if SL[j] else (tx / price - 1 - fee)
            else: ex = hi - 1; r = C[ex] / price - 1 - fee
        elif mode == 'trend':
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price: continue
            pct = tpb / price - 1; tx = price * (1 - pct); sx = price * (1 + pct)
            SL = Hh[ci + 1:hi] >= sx; TP = L[ci + 1:hi] <= tx; hit = SL | TP
            if hit.any(): j = int(np.argmax(hit)); ex = ci + 1 + j; r = ((price - sx) / price - fee) if SL[j] else ((price - tx) / price - fee)
            else: ex = hi - 1; r = (price - C[ex]) / price - fee
        elif mode == 'long':
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price: continue
            pct = tpb / price - 1; sx = price * (1 - pct)
            SL = L[ci + 1:hi] <= sx; TP = Hh[ci + 1:hi] >= tpb; hit = SL | TP
            if hit.any(): j = int(np.argmax(hit)); ex = ci + 1 + j; r = (sx / price - 1 - fee) if SL[j] else (tpb / price - 1 - fee)
            else: ex = hi - 1; r = C[ex] / price - 1 - fee
        else:
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price: continue
            pct = 1 - tpb / price; sx = price * (1 + pct)
            SL = Hh[ci + 1:hi] >= sx; TP = L[ci + 1:hi] <= tpb; hit = SL | TP
            if hit.any(): j = int(np.argmax(hit)); ex = ci + 1 + j; r = ((price - sx) / price - fee) if SL[j] else ((price - tpb) / price - fee)
            else: ex = hi - 1; r = (price - C[ex]) / price - fee
        out.append((ci, int(ts[ci]), int(ts[ex]), float(r)))
        last = ts[ci]
    return out


# ── 레짐 지표(심볼별, 진입 시점) ──
_regime = {}
def regime_of(sym):
    if sym not in _regime:
        d = arrs(sym)
        C = d['C']
        ma200, sd200 = roll_ma_sd(C, 200)
        z200 = np.where(sd200 > 0, (C - ma200) / sd200, np.nan)
        ret = np.diff(C) / C[:-1]
        s20 = np.full(len(C), np.nan)
        for i in range(20, len(C)):
            s20[i] = np.std(ret[i - 20:i])
        med = np.nanmedian(s20)
        _regime[sym] = dict(z200=z200, above=C > ma200, s20=s20, s20med=med)
    return _regime[sym]


def mult_of(variant, sym, strat, side, ci):
    rg = regime_of(sym)
    dirn = DIRSGN[side]
    z2 = rg['z200'][ci]; ab = rg['above'][ci]
    if variant == 'BASE':
        return 1.0
    if np.isnan(z2):
        return 1.0
    if variant.startswith('V1'):   # 방향정합 (강도 스윕: V1=1.4/0.6, V1a=1.2/0.8, V1b=1.6/0.4, V1c=2.0/0.0 게이트)
        hi_lo = {'V1': (1.4, 0.6), 'V1a': (1.2, 0.8), 'V1b': (1.6, 0.4), 'V1c': (2.0, 0.0)}[variant]
        match = (dirn > 0 and ab) or (dirn < 0 and not ab)
        return hi_lo[0] if match else hi_lo[1]
    if variant == 'V2':   # 전략별: 추세=정합, 역추세=역정합
        match = (dirn > 0 and ab) or (dirn < 0 and not ab)
        if strat == 's3':
            return 1.4 if match else 0.6
        return 0.6 if match else 1.4
    if variant == 'V3':   # vol타깃
        s = rg['s20'][ci]
        if not np.isfinite(s) or s <= 0:
            return 1.0
        return float(np.clip(rg['s20med'] / s, 0.5, 2.0))
    if variant.startswith('V4'):   # 연속 z200 (기울기 스윕: V4=0.4, V4a=0.2, V4b=0.6)
        slope = {'V4': 0.4, 'V4a': 0.2, 'V4b': 0.6}[variant]
        return float(np.clip(1 + slope * dirn * z2, 0.5, 1.6))
    return 1.0


def yr(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year


# ── 트레이드 생성(1회) ──
all_trades = []   # (sym, strat, side, ci, ts_en, ts_ex, r, w_base)
for sym, strat, side, win, k1, b, cdd in CELLS:
    mode = MODE[(strat, side)]
    trs = gen(sym, mode, win, k1, b, cdd, fee_of(sym))
    for ci, en, ex, r in trs:
        all_trades.append((sym, strat, side, ci, en, ex, r, base_w(sym)))
cov = {}
for s in sorted({t[0] for t in all_trades}):
    t0 = arrs(s)['ts'][0]
    cov[s] = datetime.fromtimestamp(t0 / 1000, tz=timezone.utc).strftime('%Y')
print(f"셀 {len(CELLS)}개, 트레이드 {len(all_trades)}건")
print("심볼 데이터 시작연도:", cov)

VARIANTS = ['BASE', 'V1a', 'V1', 'V1b', 'V1c', 'V2', 'V3', 'V4a', 'V4', 'V4b']
print(f"\n{'변형':6} {'평균mult':>8} {'총기여%p':>9} {'연평균':>7} {'MDD%p':>7} {'MAR':>6} {'정규화MAR':>9} | 연도별(정규화)")
for v in VARIANTS:
    mults = np.array([mult_of(v, s, st, sd_, ci) for s, st, sd_, ci, *_ in all_trades])
    mmean = mults.mean()
    rs = np.array([t[6] for t in all_trades]); ws = np.array([t[7] for t in all_trades])
    ens = np.array([t[4] for t in all_trades], np.int64)
    for norm in (False, True):
        m = mults / mmean if norm else mults
        pnl = rs * (ws * m) / 100.0 * 100.0   # %p 기여
        order = np.argsort(ens)
        p = pnl[order]; e = ens[order]
        cum = np.cumsum(p)
        peak = np.maximum.accumulate(cum)
        mdd = float(np.max(peak - cum)) if len(cum) else 0.0
        total = float(cum[-1]) if len(cum) else 0.0
        yrs_span = (e[-1] - e[0]) / (365.25 * DAY)
        mar = (total / yrs_span) / mdd if mdd > 0 else float('inf')
        ysum = defaultdict(float)
        for en, pp in zip(e, p):
            ysum[yr(en)] += pp
        if not norm:
            raw = (mmean, total, total / yrs_span, mdd, mar)
        else:
            ystr = ' '.join(f"{y % 100:02d}:{ysum[y]:+.0f}" for y in sorted(ysum))
            print(f"{v:6} {raw[0]:>8.2f} {raw[1]:>+9.1f} {raw[2]:>+7.1f} {raw[3]:>7.1f} {raw[4]:>6.2f} "
                  f"{mar:>9.2f} | {ystr}")
