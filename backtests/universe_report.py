# -*- coding: utf-8 -*-
"""3유니버스 통합 리포트 — 라이브 config(live_cells.json, 2026-07-16 확정 97셀) 그대로 재현.
유니버스별: $10k 5년 복리(실현) + 1h MTM(장중최악) → 분기 성적·최대중첩·최대노출·MTM낙폭·증거금·청산 판정.
사이징(라이브): U1 s11/s22 5%·s33 2% | U2 s11m/s22m 1%(WTI 4h페이드 0.5%)·s33m 2% | U3 책 1%·s33f 5%.
실행: python universe_report.py U1|U2|U3   (5y 창: 2021-07-16 ~)"""
import sys, os, json
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache

H = 3600 * 1000; DAY = 86400 * 1000
UNI = (sys.argv[1] if len(sys.argv) > 1 else os.getenv('U', 'U1')).upper()
T5Y = int(datetime(2021, 7, 16, tzinfo=timezone.utc).timestamp() * 1000)
FX = {'USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'}
CELLS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'live_cells.json')
LIVE = json.load(open(CELLS_JSON, encoding='utf-8'))

# ---------- 데이터 ----------
_c = {}
def arrs(sym, iv):
    key = (sym, iv)
    if key not in _c:
        m = _load_cache(sym, iv)
        if not m:
            _c[key] = None
        else:
            _c[key] = (np.array([x['close'] for x in m]), np.array([x['high'] for x in m]),
                       np.array([x['low'] for x in m]), np.array([x['start'] for x in m], np.int64))
    return _c[key]

def fee_of(sym):
    return 0.0002 if sym in FX else 0.0011

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

# ---------- 엔진 (라이브 signal_processor 규칙 재현) ----------
def sigma_cell(book, tag, sym, side, p, iv, hold_ms):
    """z-시그마 (s1/s2/s3/s4/s11/s12): 추세=s1/s3/s11(entry_high=long), 역추=s2/s4/s12.
    TP=밴드 미러(추세) 또는 밴드(역추), SL=미러(no_sl 아니면), TIME=hold. maxc 캡 적용."""
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
            if entry_high:  # 추세롱: z 상단 돌파, TP=상단밴드 미러 위
                tpb = ma[ci] + b * sd[ci]
                if tpb >= price: continue
                tp = price * (1 + (1 - tpb / price))
            else:           # 역추롱: z 하단, TP=밴드 복귀
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
            r = xp / price - 1 - fee; sd_ = 1
        else:
            if entry_high:  # 역추숏: z 상단, TP=밴드 복귀 아래
                tp = ma[ci] + b * sd[ci]
                if tp >= price: continue
            else:           # 추세숏: z 하단 돌파, TP=하단밴드 미러 아래
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
            r = (price - xp) / price - fee; sd_ = -1
        out.append((int(ts[ci]), int(ts[ex]), r, sym, sd_, float(price), book))
        last = ts[ci]; open_ex.append(ts[ex])
    return out

def fade_cell(book, sym, p, iv, hold_ms):
    """s13 급락페이드: M분(1m) 또는 M봉(4h) 수익률 ≤ -drop → 롱. retr_mult TP or 시간청산."""
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
        out.append((int(ts[ci]), int(ts[ex]), xp / price - 1 - fee, sym, 1, float(price), book))
        last = ts[ci]; open_ex.append(ts[ex])
    return out

def ewz_cell(book, sym, side, p, iv, hold_ms):
    """s14 ewz 추세: EMA 잔차 z 트리거, 시간청산 전용."""
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
        out.append((int(ts[ci]), int(ts[ex]), sd_ * (C[ex] / C[ci] - 1) - fee, sym, sd_, float(C[ci]), book))
        last = ts[ci]
    return out

def sweep_cell(book, sym, p, iv, hold_ms):
    """s15 유동성스윕: 저가<N봉저점 & 종가 복귀 → 롱, 시간청산."""
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
        out.append((int(ts[ci]), int(ts[ex]), C[ex] / C[ci] - 1 - fee, sym, 1, float(C[ci]), book))
        last = ts[ci]
    return out

# ---------- 유니버스 정의 (심볼 기준 분류 = bots universe_of와 동일) ----------
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
                    tl = sigma_cell(book, tag, sym, side, p, iv, hold_ms)
                elif tag == 's13':
                    tl = fade_cell(book, sym, p, iv, hold_ms)
                elif tag == 's14':
                    tl = ewz_cell(book, sym, side, p, iv, hold_ms)
                elif tag == 's15':
                    tl = sweep_cell(book, sym, p, iv, hold_ms)
                else:
                    tl = []
                fr = frac
                if book == 's22m' and tag == 's13' and sym == 'WTI': fr = frac * 0.5
                trades += [(en, ex, r, sym, sd_, ep, book, fr) for en, ex, r, sym, sd_, ep, bk in tl]

trades = [t for t in trades if t[0] >= T5Y]
trades.sort()
syms = sorted(set(t[3] for t in trades))
print(f'[{UNI}] 트레이드 {len(trades)}건 / 심볼 {syms}', flush=True)
# 1m 데이터 커버리지(진입 누락 가늠용)
for s in syms:
    d = arrs(s, '1')
    if d is not None:
        t0s = datetime.fromtimestamp(int(d[3][0]) / 1000, tz=timezone.utc).strftime('%Y-%m')
        print(f'  {s}: 1m {t0s}~ ({len(d[0]):,}봉)')

# ---------- 시간단위 MTM 마크 (1m 우선, 없으면 4h 폴백) ----------
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

# ---------- 포트폴리오 시뮬 ----------
ev = []
for i, t in enumerate(trades): ev.append((t[0], 0, i)); ev.append((t[1], 1, i))
ev.sort(key=lambda x: (x[0], x[1]))
eq = 10000.0; open_ = {}; expo = 0.0; ptr = 0
t0 = T5Y // H; t1 = max(e[0] for e in ev) // H

def qof(ms):
    t = datetime.fromtimestamp(ms / 1000, tz=timezone.utc); return f'{t.year}Q{(t.month - 1) // 3 + 1}'

Q = defaultdict(lambda: dict(maxc=0, bk={}, maxexpo=0.0, minmtm=1e18, peak=-1e18, mdd=0.0, eq_end=0.0))
peak = -1e18; mdd = 0.0; min_mg = 9e9; liq = None; maxc_all = 0
for hour in range(t0, t1 + 1):
    tms = (hour + 1) * H
    while ptr < len(ev) and ev[ptr][0] < tms:
        t, typ, i = ev[ptr]; ptr += 1
        if typ == 0:
            no = trades[i][7] * eq; open_[i] = no; expo += no
        else:
            no = open_.pop(i, None)
            if no is not None: eq += no * trades[i][2]; expo -= no
    un = 0.0; cbk = defaultdict(int)
    for i, no in open_.items():
        en, ex, r, sym, side, ep, bk, fr = trades[i]
        m = mark(sym, hour, side, ep)
        un += no * side * (m / ep - 1.0)
        cbk[bk] += 1
    mtm = eq + un
    c = sum(cbk.values())
    q = qof(hour * H); d = Q[q]
    if c > d['maxc']: d['maxc'] = c; d['bk'] = dict(cbk)
    maxc_all = max(maxc_all, c)
    if eq > 0: d['maxexpo'] = max(d['maxexpo'], expo / eq)
    d['minmtm'] = min(d['minmtm'], mtm); d['eq_end'] = eq
    d['peak'] = max(d['peak'], mtm); d['mdd'] = max(d['mdd'], (d['peak'] - mtm) / d['peak'] if d['peak'] > 0 else 0)
    peak = max(peak, mtm); mdd = max(mdd, (peak - mtm) / peak if peak > 0 else 0)
    if expo > 0:
        mg = mtm / expo
        if mg < min_mg: min_mg = mg
        if mg < 0.005 and liq is None: liq = hour

print(f'\n{"분기":>7} | {"실현자산":>10} | {"분기수익":>7} | {"최대중첩":>8} | {"노출":>5} | {"MTM최저":>10} | {"낙폭":>5}')
prev = 10000.0
for q in sorted(Q):
    d = Q[q]
    ret = (d['eq_end'] / prev - 1) * 100 if prev > 0 else 0
    bks = '+'.join(f'{v}' for k, v in sorted(d['bk'].items()))
    print(f'{q:>7} | ${d["eq_end"]:>9,.0f} | {ret:>+6.1f}% | {d["maxc"]:>3} ({bks}) | {d["maxexpo"]*100:>4.0f}% | ${d["minmtm"]:>9,.0f} | {d["mdd"]*100:>4.1f}%')
    prev = d['eq_end']
yrs = (t1 - t0) * H / (365.25 * DAY)
print(f'\n[{UNI}] 최종 ${eq:,.0f} ({eq/10000:.2f}배, {yrs:.1f}년, CAGR {((eq/10000)**(1/yrs)-1)*100:+.1f}%)')
print(f'최대중첩 {maxc_all} | 전체 MTM 최대낙폭 {mdd*100:.1f}% | 최저 증거금율 {min_mg*100:.1f}%')
print('청산 판정: ' + ('🔴 발생 ' + datetime.fromtimestamp(liq * H / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
      if liq else f'🟢 불가능 수준 (유지증거금 0.5% 대비 여유 {min_mg/0.005:.0f}배)'))
