# -*- coding: utf-8 -*-
"""유니버스(전 책 합산) 시뮬 + 심볼별 중첩·노출 실측 — 심볼캡 설계 근거 (2026-07-20).
universe_report.py 기반. 라이브 97셀(live_cells.json, 셀캡=현행 config) · 심볼캡 없음(SYMCAP env로 실험 가능).
사이징 = 현행 라이브(2026-07-16 868dbe4 상향 반영): U1 5/5/2% | U2 3.5/3.5/7% | U3 2.3/2.3/11.5%.
실행: python universe_symbolcap.py U1|U2|U3   (5y 창: 2021-07-16~)"""
import sys, os, json
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache

H = 3600 * 1000; DAY = 86400 * 1000
UNI = (sys.argv[1] if len(sys.argv) > 1 else os.getenv('U', 'U1')).upper()
SYMCAP = int(os.getenv('SYMCAP', '0'))   # 0=심볼캡 없음. >0이면 심볼 합산 게임수 캡(진입 드롭) 실험.
T5Y = int(datetime(2021, 7, 16, tzinfo=timezone.utc).timestamp() * 1000)
FX = {'USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'}
CELLS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'live_cells.json')
LIVE = json.load(open(CELLS_JSON, encoding='utf-8'))

# NOCAP33=1: 일봉(s33/s33m/s33f) 셀캡 제거 실험 — "캡 없으면 터지나?" 검증용
if os.getenv('NOCAP33'):
    for _bk in ('s33', 's33m', 's33f'):
        for _sub in LIVE.get(_bk, []):
            for _sym, _sides in (_sub['params'] or {}).items():
                for _side, _p in _sides.items():
                    _p['max_concurrent'] = 200
    print('⚠️ NOCAP33: 일봉 셀캡 전부 200(비구속)으로 override')

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
            else:
                ex = hi - 1; xp = C[ex]
            r = (price - xp) / price - fee; sd_ = -1
        out.append((int(ts[ci]), int(ts[ex]), r, sym, sd_, float(price), book))
        last = ts[ci]; open_ex.append(ts[ex])
    return out

def fade_cell(book, sym, p, iv, hold_ms):
    d = arrs(sym, iv)
    if d is None: return []
    C, Hh, L, ts = d; n = len(C); fee = fee_of(sym)
    M = int(p['m_min']) if iv == '1' else max(1, int(p['m_min']))
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
    d = arrs(sym, iv)
    if d is None: return []
    C, Hh, L, ts = d; n = len(C); fee = fee_of(sym)
    s = int(p['ewz_s']); k = float(p['k1']); cd = int(p['cooldown_sec']) * 1000
    hold = int(p.get('hold_sec') or 0) * 1000 or hold_ms
    rev = bool(p.get('ewz_rev'))
    sd_ = 1 if side == 'long' else -1
    mu = ema(C, s); resid = C - mu; sig = ema(np.abs(resid), s)
    z = resid / np.where(sig == 0, 1e-12, sig)
    ent_neg = (sd_ < 0) or rev
    cand = np.flatnonzero(z <= -k) if ent_neg else np.flatnonzero(z >= k)
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

def want(uni, sym):
    if uni == 'U1': return sym.endswith('USDT')
    if uni == 'U2': return (not sym.endswith('USDT')) and (sym not in FX)
    return sym in FX

# ✅ 현행 라이브 사이징 (2026-07-16 868dbe4: U2 ×3.5, U3 ×2.3)
BOOK_FRAC = {
    'U1': {'s11': 0.05, 's22': 0.05, 's33': 0.02},
    'U2': {'s11m': 0.035, 's22m': 0.035, 's33m': 0.07},
    # U3 일봉 = s33m 책(2026-07-21 fxd 통합) — want()가 FX 심볼만 걸러 11.5% 적용
    'U3': {'s11m': 0.023, 's22m': 0.023, 's33m': 0.115},
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
ALL_TRADES = trades

def apply_symcap(trades, symcap):
    """심볼 합산 게임수 캡 — 시간순으로 초과 진입 드롭. 반환 (채택, 심볼별 드롭통계)."""
    dropped = defaultdict(lambda: [0, 0.0])
    if symcap <= 0:
        return trades, dropped
    import heapq
    keep = []; opens = defaultdict(int); exq = []
    for t in trades:
        en, ex, r, sym = t[0], t[1], t[2], t[3]
        while exq and exq[0][0] <= en:
            _, s2_ = heapq.heappop(exq); opens[s2_] -= 1
        if opens[sym] >= symcap:
            d = dropped[sym]; d[0] += 1; d[1] += r
            continue
        opens[sym] += 1; keep.append(t); heapq.heappush(exq, (ex, sym))
    return keep, dropped

trades, dropped_by_sym = apply_symcap(ALL_TRADES, SYMCAP)
syms = sorted(set(t[3] for t in trades))
print(f'[{UNI}] SYMCAP={SYMCAP or "없음"} | 트레이드 {len(trades)}건 | 심볼 {len(syms)}', flush=True)

# MTM 마크 (1m 우선, 4h 폴백)
HL = {}
for sym in syms:
    d = arrs(sym, '1')
    if d is not None:
        C, Hh, L, ts = d; hr = ts // H
        uh, idx = np.unique(hr, return_index=True)
        HL[sym] = (uh, np.minimum.reduceat(L, idx), np.maximum.reduceat(Hh, idx))
    else:
        HL[sym] = None
H4 = {}
for sym in syms:
    d = arrs(sym, '240')
    if d is not None:
        C, Hh, L, ts = d; H4[sym] = (ts // H, L, Hh)

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

def run_portfolio(trades):
    """시간단위 MTM 포트폴리오 시뮬 → 지표 dict."""
    ev = []
    for i, t in enumerate(trades): ev.append((t[0], 0, i)); ev.append((t[1], 1, i))
    ev.sort(key=lambda x: (x[0], x[1]))
    eq = 10000.0; open_ = {}; expo = 0.0; ptr = 0
    t0 = T5Y // H; t1 = max(e[0] for e in ev) // H
    peak = -1e18; mdd = 0.0; min_mg = 9e9; liq = None; maxc_all = 0; max_expo = 0.0
    worst = None
    symstat = defaultdict(lambda: dict(mx_games=0, mx_no=0.0))
    for hour in range(t0, t1 + 1):
        tms = (hour + 1) * H
        while ptr < len(ev) and ev[ptr][0] < tms:
            t, typ, i = ev[ptr]; ptr += 1
            if typ == 0:
                no = trades[i][7] * eq; open_[i] = no; expo += no
            else:
                no = open_.pop(i, None)
                if no is not None: eq += no * trades[i][2]; expo -= no
        un = 0.0; csym = defaultdict(int); nosym = defaultdict(float)
        for i, no in open_.items():
            en, ex, r, sym, side, ep, bk, fr = trades[i]
            m = mark(sym, hour, side, ep)
            un += no * side * (m / ep - 1.0)
            csym[sym] += 1; nosym[sym] += no
        mtm = eq + un
        maxc_all = max(maxc_all, sum(csym.values()))
        if eq > 0: max_expo = max(max_expo, expo / eq)
        for s, g in csym.items():
            st = symstat[s]
            st['mx_games'] = max(st['mx_games'], g)
            if eq > 0: st['mx_no'] = max(st['mx_no'], nosym[s] / eq)
        peak = max(peak, mtm)
        dd = (peak - mtm) / peak if peak > 0 else 0
        if dd > mdd:
            mdd = dd
            worst = (hour, dict(csym), {s: nosym[s] / eq for s in nosym}, mtm)
        if expo > 0:
            mg = mtm / expo
            if mg < min_mg: min_mg = mg
            if mg < 0.005 and liq is None: liq = hour
    yrs = (t1 - t0) * H / (365.25 * DAY)
    return dict(eq=eq, yrs=yrs, mdd=mdd, min_mg=min_mg, liq=liq,
                maxc=maxc_all, max_expo=max_expo, symstat=symstat, worst=worst)

if os.getenv('SWEEP'):
    caps = [int(x) for x in os.getenv('SWEEP').split(',')]
    print(f'\n{"심볼캡":>5} | {"최종":>10} | {"배수":>6} | {"CAGR":>6} | {"MTM낙폭":>6} | {"최저증거금":>6} | {"최대중첩":>5} | {"노출":>5} | {"드롭":>5} | 드롭 r합')
    for cap in caps:
        tr, dr = apply_symcap(ALL_TRADES, cap)
        m = run_portfolio(tr)
        nd = sum(v[0] for v in dr.values()); rd = sum(v[1] for v in dr.values())
        cagr = ((m['eq'] / 10000) ** (1 / m['yrs']) - 1) * 100
        print(f'{cap or "없음":>5} | ${m["eq"]:>9,.0f} | {m["eq"]/10000:>5.1f}배 | {cagr:>+5.0f}% | {m["mdd"]*100:>5.1f}% | '
              f'{m["min_mg"]*100:>5.1f}% | {m["maxc"]:>5} | {m["max_expo"]*100:>4.0f}% | {nd:>5} | {rd*100:+.0f}%p'
              + ('  🔴청산' if m['liq'] else ''), flush=True)
else:
    m = run_portfolio(trades)
    eq, yrs, mdd, min_mg, liq = m['eq'], m['yrs'], m['mdd'], m['min_mg'], m['liq']
    print(f'\n[{UNI}] 최종 ${eq:,.0f} ({eq/10000:.2f}배, {yrs:.1f}년, CAGR {((eq/10000)**(1/yrs)-1)*100:+.1f}%)')
    print(f'유니버스 최대중첩 {m["maxc"]}게임 | 최대노출 {m["max_expo"]*100:.0f}% | MTM 최대낙폭 {mdd*100:.1f}% | 최저 증거금율 {min_mg*100:.1f}%')
    print('청산 판정: ' + ('🔴 발생 ' + datetime.fromtimestamp(liq * H / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
          if liq else f'🟢 불가능 수준 (유지증거금 0.5% 대비 여유 {min_mg/0.005:.0f}배)'))
    if SYMCAP > 0 and dropped_by_sym:
        print('심볼캡 드롭: ' + ', '.join(f'{s} {v[0]}건(r합 {v[1]*100:+.0f}%p)' for s, v in sorted(dropped_by_sym.items())))
    print(f'\n{"심볼":10s} {"역대최대 게임":>8s} {"최대노출/자산":>10s}')
    symstat = m['symstat']
    for s in sorted(symstat, key=lambda x: -symstat[x]['mx_games']):
        st = symstat[s]
        print(f'{s:10s} {st["mx_games"]:8d} {st["mx_no"]*100:9.0f}%')
    if m['worst']:
        hw, cs, ns, mtmw = m['worst']
        when = datetime.fromtimestamp(hw * H / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        top = ', '.join(f'{s} {g}겜/{ns.get(s,0)*100:.0f}%' for s, g in sorted(cs.items(), key=lambda x: -x[1])[:6])
        print(f'\nMTM 최악시점 {when} (MTM ${mtmw:,.0f}): {top}')
