# -*- coding: utf-8 -*-
"""환율 유니버스 확장 스크리닝 (2026-10-05) — 메이저 7종 재점검 + 크로스 21종 신규, 4h·일봉 z 4방향.

방법 = 기존 환율 스크리닝(s22_dirs_fx / bond_screen)과 같은 모델·같은 통과 기준:
  모델: z=(종가−MA)/σ. 추세롱(tl)·역추롱(rl) = 롱은 SL 없음, 추세숏(ts)·역추숏(rs) = SL 대칭, TP=밴드(추세는 밴드 미러), 보유 15일.
  통과: n≥30(일봉 25), 건당 기대≥0.15%, 마이너스 연도(<−1.5%p)≤2, 최악 연도>−30%p, 연 3건↑(일봉 2건↑)  ← 환율용 기준
  견고성: 통과 셀의 이웃 격자(각 축 ±1스텝)에서 기대 중앙값>0 이고 절반 이상이 통과 기준 충족
  전·후반: 기간을 반으로 나눠 양쪽 모두 건당 기대>0
수수료 = 심볼별 왕복 max(0.02%, 스프레드×1.5) (HFM 실측 스프레드, infra/tmp/hfm_fx_spreads.json). 스왑 미반영.
데이터: 메이저 일봉 D13(13.5y)·4h, 크로스 일봉 D10·4h(fetch_mt5_hist_fx.py, MT5 히스토리 2010~).
usage: py -3 backtests/fx_cross_screen.py [SYM,SYM...]  → results/FX_CROSS_SCREEN.md + fx_cross_cells.json
"""
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from reversion_calibrator import _load_cache  # noqa: E402

H = 3600 * 1000
DAY = 86400 * 1000
HOLD = 15 * DAY
MAJORS = ['USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
CROSSES = ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY", "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURNZD",
           "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD", "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "CADCHF"]
# 표본 밖 검증: OOS=1 이면 셀 선정(통과·견고성·대표 선택)을 CUT 이전 데이터로만 하고, 이후 구간은 손대지 않은 채 거래만 남긴다.
CUT = int(datetime(2021, 7, 16, tzinfo=timezone.utc).timestamp() * 1000) if os.getenv('OOS') == '1' else None
SFX = '_oos' if CUT else ''
MODENM = {'tl': '추세롱', 'rl': '역추롱', 'ts': '추세숏', 'rs': '역추숏'}
# 4h 는 환율 변동성이 작아 기존 격자(창 30~180봉·K 2.0~)에 낮은 문턱(K 1.5)·긴 창(360봉=60일)·먼 목표(B 2)를 더했다
GRID_4H = dict(win=[30, 60, 90, 120, 180, 360], k=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0], b=[-3, -2, -1, 0, 1, 2], cd=[12 * H, 24 * H, 48 * H])
GRID_D = dict(win=[60, 90, 150, 200], k=[1.6, 1.8, 2.0, 2.3, 2.6, 3.0], b=[-3, -2, -1, -0.4, 0, 0.4, 1, 2],
              cd=[1 * DAY, 2 * DAY, 3 * DAY, 5 * DAY, 7 * DAY])


def fees():
    """심볼별 현재 스프레드(%) — infra/tmp/fx_spreads.py 가 HFM 터미널에서 30초 중앙값으로 측정한 값"""
    p = os.path.join(HERE, '..', '..', 'infra', 'tmp', 'hfm_fx_spreads.json')
    return json.load(open(p)) if os.path.exists(p) else {}


SPREADS = fees()


def fee_of(sym, last_price=None):
    """왕복 비용 = max(0.02%, 스프레드×1.5). 스왑(보유 이자)은 넣지 않았다."""
    sp = SPREADS.get(sym)
    return 0.0004 if sp is None else max(0.0002, 1.5 * sp / 100)


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


def zgen(C, Hh, L, ts, win, k, b, mode, cd_ms, ma, sd, z, fee):
    """bond_screen.zgen 과 같은 모델. 반환 [(진입ms, 청산ms, 수익률, 방향, 진입가)]"""
    n = len(C)
    epos = mode in ('tl', 'rs')
    cand = np.flatnonzero(z >= k) if epos else np.flatnonzero(z <= -k)
    cand = cand[cand >= win]
    out, last = [], None
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd_ms:
            continue
        price = C[ci]
        hi = int(np.searchsorted(ts, ts[ci] + HOLD, 'right'))
        if hi <= ci + 1:
            continue
        xi = hi - 1
        if mode in ('tl', 'rl'):
            if mode == 'tl':
                tpb = ma[ci] + b * sd[ci]
                if tpb >= price:
                    continue
                tp = price * (1 + (1 - tpb / price))
            else:
                tp = ma[ci] - b * sd[ci]
                if tp <= price:
                    continue
            TP = Hh[ci + 1:hi] >= tp
            if TP.any():
                xi = ci + 1 + int(np.argmax(TP))
                r = tp / price - 1 - fee
            else:
                r = C[hi - 1] / price - 1 - fee
            side = 1
        else:
            if mode == 'ts':
                tpb = ma[ci] - b * sd[ci]
                if tpb <= price:
                    continue
                pct = tpb / price - 1
            else:
                tpv = ma[ci] + b * sd[ci]
                if tpv >= price:
                    continue
                pct = 1 - tpv / price
            tp, sl = price * (1 - pct), price * (1 + pct)
            hit = (L[ci + 1:hi] <= tp) | (Hh[ci + 1:hi] >= sl)
            if hit.any():
                j = int(np.argmax(hit))
                xi = ci + 1 + j
                xp = sl if Hh[ci + 1 + j] >= sl else tp
            else:
                xp = C[hi - 1]
            r = (price - xp) / price - fee
            side = -1
        out.append((int(ts[ci]), int(ts[xi]), r, side, float(price)))
        last = ts[ci]
    return out


def evaluate(tr, yrs_total, nmin, gmin):
    if len(tr) < nmin:
        return None
    a = np.array([t[2] for t in tr])
    exp = a.mean() * 100
    if exp < 0.15:
        return None
    w = a > 0
    ysum = defaultdict(float)
    for t in tr:
        ysum[yr(t[0])] += t[2] * 100
    if sum(1 for s in ysum.values() if s < -1.5) > 2 or min(ysum.values()) < -30:
        return None
    gpy = len(tr) / yrs_total
    if gpy < gmin:
        return None
    half = (tr[0][0] + tr[-1][0]) / 2
    h1 = [t[2] for t in tr if t[0] < half]
    h2 = [t[2] for t in tr if t[0] >= half]
    return dict(n=len(tr), exp=round(exp, 3), pf=round(float(a[w].sum() / (-a[~w].sum())) if (~w).any() else 99, 2),
                wr=round(100 * float(w.mean()), 0), gpy=round(gpy, 1), sd=round(float(a.std()) * 100, 2),
                h1=round(float(np.mean(h1)) * 100, 2) if h1 else None, h2=round(float(np.mean(h2)) * 100, 2) if h2 else None,
                ys=' '.join(f'{y % 100:02d}:{ysum[y]:+.1f}' for y in sorted(ysum)))


def arrs(sym, itv):
    m = _load_cache(sym, itv)
    if not m:
        return None
    return (np.array([x['close'] for x in m]), np.array([x['high'] for x in m]),
            np.array([x['low'] for x in m]), np.array([x['start'] for x in m], np.int64))


def neighbors(base, grid):
    axes = []
    for kk, v in base.items():
        g = grid[kk]
        i = g.index(v)
        axes.append([g[j] for j in (i - 1, i, i + 1) if 0 <= j < len(g)])
    return [dict(zip(base.keys(), c)) for c in product(*axes) if dict(zip(base.keys(), c)) != base]


def screen(sym, itv, grid, nmin, gmin, label):
    d = arrs(sym, itv)
    if d is None:
        return []
    C, Hh, L, ts = d
    fee = fee_of(sym, C[-1])
    t_end = min(int(ts[-1]), CUT) if CUT else int(ts[-1])
    yrs_total = (t_end - int(ts[0])) / (365.25 * DAY)
    zc = {w: zma(C, w) for w in grid['win']}
    cache = {}

    def run(p, mode):
        key = (mode, p['win'], p['k'], p['b'], p['cd'])
        if key not in cache:
            z, ma, sd = zc[p['win']]
            tr = zgen(C, Hh, L, ts, p['win'], p['k'], p['b'], mode, p['cd'], ma, sd, z, fee)
            sel = [t for t in tr if t[0] < CUT] if CUT else tr
            cache[key] = (tr, evaluate(sel, yrs_total, nmin, gmin))
        return cache[key]

    out = []
    for mode in MODENM:
        best = None
        for win, k, b, cd in product(grid['win'], grid['k'], grid['b'], grid['cd']):
            p = dict(win=win, k=k, b=b, cd=cd)
            tr, e = run(p, mode)
            if not e or e['h1'] is None or e['h2'] is None or e['h1'] <= 0 or e['h2'] <= 0:
                continue
            nb = [run(q, mode) for q in neighbors(p, grid)]
            nb_exp = [float(np.mean([t[2] for t in t_ if not CUT or t[0] < CUT] or [0.0])) * 100 for t_, _ in nb]
            nb_pass = sum(1 for _, e_ in nb if e_)
            if np.median(nb_exp) <= 0 or nb_pass < len(nb) / 2:
                continue
            score = e['exp'] / max(e['sd'], 1e-9) * np.sqrt(e['gpy'])     # 건당 품질 × 빈도
            if best is None or score > best[0]:
                best = (score, p, e, tr, round(float(np.median(nb_exp)), 2), f"{nb_pass}/{len(nb)}")
        if best:
            _, p, e, tr, nbm, nbp = best
            out.append(dict(sym=sym, tf=label, itv=itv, mode=mode, win=p['win'], k=p['k'], b=p['b'], cd_ms=p['cd'], fee=round(fee * 100, 3),
                            years=round(yrs_total, 1), nb_median=nbm, nb_pass=nbp, **e, trades=tr))
    return out


def main():
    syms = sys.argv[1].split(',') if len(sys.argv) > 1 else MAJORS + CROSSES
    cells = []
    for sym in syms:
        c4 = screen(sym, '240', GRID_4H, 30, 3.0, '4h')
        cd_ = screen(sym, 'D13' if sym in MAJORS else 'D10', GRID_D, 25, 2.0, '일봉')
        cells += c4 + cd_
        print(f"{sym} done 4h {len(c4)} · 일봉 {len(cd_)}", flush=True)
    L = [f"# 환율 확장 스크리닝 — 메이저 7 + 크로스 21, 4h·일봉 z 4방향 (생성 {datetime.now():%Y-%m-%d %H:%M})\n",
         "통과 = 기본 기준 + 이웃 격자 견고성 + 전·후반 모두 양수. 방향별 대표 1셀(건당품질×빈도 최고).\n",
         "| 심볼 | 봉 | 방향 | 창 | K | B | 쿨다운 | n | 연 건수 | 건당% | σ | 승률 | PF | 전반/후반 | 이웃 중앙·통과 | 수수료% | 연도별 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        L.append(f"| {c['sym']} | {c['tf']} | {MODENM[c['mode']]} | {c['win']} | {c['k']} | {c['b']} | {c['cd_ms']//H}h | {c['n']} | {c['gpy']} | "
                 f"{c['exp']:+.2f} | {c['sd']} | {c['wr']:.0f}% | {c['pf']} | {c['h1']:+.2f}/{c['h2']:+.2f} | {c['nb_median']:+.2f}·{c['nb_pass']} | {c['fee']} | {c['ys']} |")
    open(os.path.join(HERE, 'results', f'FX_CROSS_SCREEN{SFX}.md'), 'w', encoding='utf-8').write("\n".join(L))
    json.dump(cells, open(os.path.join(HERE, 'results', f'fx_cross_cells{SFX}.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"\n통과 셀 {len(cells)}개 → results/FX_CROSS_SCREEN.md")


if __name__ == "__main__":
    main()
