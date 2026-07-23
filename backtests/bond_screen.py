# -*- coding: utf-8 -*-
"""채권 3종(EUBUND/UKGILT/US10YR) 신규 스크리닝 — S11 방법론(베이스자립+연도균형).
데이터: MT5 히스토리(D1 2017~ → _D10m.pkl / H4 2017~ → _240m.pkl). 1m은 100k상한(5~9개월)이라 1분책 보류.
⚠️ .F 만기물: 분기 롤 갭이 시계열에 포함 — 통과 셀은 롤 주간 성과를 별도 확인할 것.
4h: z 4방향(창 30~180봉) + 급락페이드 — s22_h4.py와 동일 모델/필터.
일봉: z 4방향(창 60~200일), 보유 15일, TP=밴드 대칭, 손절우선 — daily_sweep 모델 + 연도균형 필터.
필터: n≥25, 기대≥0.4%, 마이너스연도(<-3%p)≤2, 최악>-30%p, 연 3게임↑(일봉 2↑).
"""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache

FEE = 0.0011
H = 3600 * 1000
DAY = 86400 * 1000
HOLD = 15 * DAY
SYMS = ['EUBUND', 'UKGILT', 'US10YR']
MODENM = {'tl': '추세롱', 'rl': '역추롱', 'ts': '추세숏', 'rs': '역추숏'}


def yr(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year


def zma(C, win):
    n = len(C)
    cs = np.concatenate([[0.], np.cumsum(C)])
    cs2 = np.concatenate([[0.], np.cumsum(C * C)])
    ma = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    s = cs[win:] - cs[:-win]
    s2 = cs2[win:] - cs2[:-win]
    mm = s / win
    ma[win - 1:] = mm
    sd[win - 1:] = np.sqrt(np.maximum(s2 / win - mm * mm, 0))
    return np.where(sd > 0, (C - ma) / sd, np.nan), ma, sd


def evaluate(tr, yrs_total, nmin=25, gmin=3.0):
    if len(tr) < nmin:
        return None
    a = np.array([r for _, r in tr])
    w = a > 0
    exp = a.mean() * 100
    if exp < 0.4:
        return None
    pf = a[w].sum() / (-a[~w].sum()) if (~w).any() else 99
    ysum = defaultdict(float)
    for en, r in tr:
        ysum[yr(en)] += r * 100
    bad = [y for y, s in ysum.items() if s < -3]
    worst = min(ysum.values())
    if len(bad) > 2 or worst < -30:
        return None
    gpy = len(tr) / yrs_total
    if gpy < gmin:
        return None
    return dict(n=len(tr), exp=exp, pf=pf, wr=100 * w.mean(), gpy=gpy, worst=worst,
                ys=' '.join(f'{y % 100:02d}:{ysum[y]:+.0f}' for y in sorted(ysum)))


def zgen(C, Hh, L, ts, win, k, b, mode, cd_ms, ma, sd, z, hold_ms):
    """s22_h4.py zgen과 동일 모델(롱 SL無/숏 SL대칭, TP=밴드) — 봉 간격 무관."""
    n = len(C)
    epos = mode in ('tl', 'rs')
    cand = np.flatnonzero(z >= k) if epos else np.flatnonzero(z <= -k)
    cand = cand[cand >= win]
    out = []
    last = None
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd_ms:
            continue
        price = C[ci]
        hi = int(np.searchsorted(ts, ts[ci] + hold_ms, 'right'))
        if hi <= ci + 1:
            continue
        if mode == 'tl':
            tpb = ma[ci] + b * sd[ci]
            if tpb >= price:
                continue
            tp = price * (1 + (1 - tpb / price))
            TP = Hh[ci + 1:hi] >= tp
            r = (tp / price - 1 - FEE) if TP.any() else C[hi - 1] / price - 1 - FEE
        elif mode == 'rl':
            tp = ma[ci] - b * sd[ci]
            if tp <= price:
                continue
            TP = Hh[ci + 1:hi] >= tp
            r = (tp / price - 1 - FEE) if TP.any() else C[hi - 1] / price - 1 - FEE
        elif mode == 'ts':
            tpb = ma[ci] - b * sd[ci]
            if tpb <= price:
                continue
            pct = tpb / price - 1
            tp = price * (1 - pct)
            sl = price * (1 + pct)
            TPh = L[ci + 1:hi] <= tp
            SL = Hh[ci + 1:hi] >= sl
            hit = TPh | SL
            if hit.any():
                j = int(np.argmax(hit))
                xp = sl if SL[j] else tp
            else:
                xp = C[hi - 1]
            r = (price - xp) / price - FEE
        else:  # rs
            tp = ma[ci] + b * sd[ci]
            if tp >= price:
                continue
            pct = 1 - tp / price
            sl = price * (1 + pct)
            TPh = L[ci + 1:hi] <= tp
            SL = Hh[ci + 1:hi] >= sl
            hit = TPh | SL
            if hit.any():
                j = int(np.argmax(hit))
                xp = sl if SL[j] else tp
            else:
                xp = C[hi - 1]
            r = (price - xp) / price - FEE
        out.append((int(ts[ci]), r))
        last = ts[ci]
    return out


def arrs(sym, itv):
    m = _load_cache(sym, itv)
    if not m:
        return None
    return (np.array([x['close'] for x in m]), np.array([x['high'] for x in m]),
            np.array([x['low'] for x in m]), np.array([x['start'] for x in m], np.int64))


res = defaultdict(list)
for sym in SYMS:
    # ── 4h 스윕 ──
    d = arrs(sym, '240')
    if d:
        C, Hh, L, ts = d
        yrs_total = (int(ts[-1]) - int(ts[0])) / (365.25 * DAY)
        for win in [30, 60, 90, 120, 180]:
            z, ma, sd = zma(C, win)
            for mode in ['tl', 'rl', 'ts', 'rs']:
                for k in [2.0, 2.5, 3.0, 3.5, 4.0]:
                    for b in [-3, -2, -1, 0, 1]:
                        for cdh in [12, 24, 48]:
                            tr = zgen(C, Hh, L, ts, win, k, b, mode, cdh * H, ma, sd, z, HOLD)
                            e = evaluate(tr, yrs_total)
                            if e:
                                res[sym].append((f'4h z{MODENM[mode]} w{win * 4 // 24}d K{k} B{b} cd{cdh}h', e))
        # 급락페이드롱 (4h)
        for M in [3, 6, 12, 18, 30]:
            ret = C / np.roll(C, M) - 1
            ret[:M] = np.nan
            for r0 in [0.01, 0.02, 0.03, 0.05]:   # 채권 변동성 작음 → 낮은 트리거 포함
                cand = np.flatnonzero(ret <= -r0)
                if len(cand) < 25:
                    continue
                for exit_, cap in [('T', 6), ('T', 30), ('T', 60), ('R', 60)]:
                    trl = []
                    last = None
                    n = len(C)
                    for ci in cand:
                        if ci + 1 >= n:
                            break
                        if last is not None and (ts[ci] - last) < 24 * H:
                            continue
                        price = C[ci]
                        hi = min(ci + 1 + cap, n)
                        if exit_ == 'R':
                            tp = price * (1 + 1.5 * abs(ret[ci]))
                            TP = Hh[ci + 1:hi] >= tp
                            r = (tp / price - 1 - FEE) if TP.any() else C[hi - 1] / price - 1 - FEE
                        else:
                            r = C[hi - 1] / price - 1 - FEE
                        trl.append((int(ts[ci]), r))
                        last = ts[ci]
                    e = evaluate(trl, yrs_total)
                    if e:
                        res[sym].append((f'4h fade롱 {M * 4}h≤-{r0 * 100:.0f}% {exit_}{cap * 4 // 24}d', e))
    # ── 일봉 스윕 ──
    d = arrs(sym, 'D10')
    if d:
        C, Hh, L, ts = d
        yrs_total = (int(ts[-1]) - int(ts[0])) / (365.25 * DAY)
        for win in [60, 90, 150, 200]:
            z, ma, sd = zma(C, win)
            for mode in ['tl', 'rl', 'ts', 'rs']:
                for k in [1.6, 1.8, 2.0, 2.3, 2.6, 3.0]:
                    for b in [-3, -2, -1, -0.4, 0, 0.4, 1, 2]:
                        for cdd in [1, 2, 3, 5, 7]:
                            tr = zgen(C, Hh, L, ts, win, k, b, mode, cdd * DAY, ma, sd, z, HOLD)
                            e = evaluate(tr, yrs_total, nmin=25, gmin=2.0)
                            if e:
                                res[sym].append((f'D z{MODENM[mode]} w{win}d K{k} B{b} cd{cdd}d', e))
    print(f'{sym} done ({len(res[sym])})', flush=True)

print()
print(f'{"심볼":8} {"셀":34} {"n":>4} {"기대%":>6} {"PF":>5} {"승%":>4} {"연":>4} | 연도별')
for sym in SYMS:
    rs = sorted(res.get(sym, []), key=lambda x: -x[1]['exp'] * min(x[1]['pf'], 5))
    seen = set()
    for name, e in rs:
        fam = ' '.join(name.split()[:2])   # 'D z추세롱' / '4h fade롱' 단위로 대표 1개
        if fam in seen:
            continue
        seen.add(fam)
        print(f'{sym:8} {name:34} {e["n"]:>4} {e["exp"]:>+6.2f} {e["pf"]:>5.2f} {e["wr"]:>3.0f}% {e["gpy"]:>4.1f} | {e["ys"]}')
print('\n(통과 0줄이면 해당 심볼은 전 패밀리 기각)')
