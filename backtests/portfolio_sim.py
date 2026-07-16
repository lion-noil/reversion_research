"""4방향 전략(역추세 롱/숏 + 추세 롱/숏) 통합 포트폴리오 3년 백테스트.
- 시작 $10,000, 한 랏=진입시점 실현자산의 5%, 최대 동시 200랏(초과 진입 스킵).
- 각 (심볼×방향) top-1 양수기대 조합만 채택(results/ JSON). 자산중복: BTC/ETH/SOL/XRP=USDT피드, 나머지=MT5.
- 피라미딩(진입기준 쿨다운, 플랫게이트無) = 백테스트 원본과 동일. 14일 최대보유, fee0.11%.
- 자산=실현잔고(청산 시 갱신). ⚠️10배 노출이라 실현곡선은 장중 청산위험 과소평가.
출력: portfolio_sim_result.json (요약+자산곡선+방향별기여), 콘솔 요약.
"""
import sys, os, json, heapq
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_s1_quarters_mt5 import prep, WIN

FEE = 0.0011
DAY = 86400 * 1000
MH = 14 * DAY                       # 최대보유 14일
INIT = 10_000.0
LOT_FRAC = 0.05                     # 한 랏 = 자산 5%
MAX_LOTS = 200                      # 최대 동시 랏
RES = os.path.join(os.path.dirname(__file__), 'results')

MT5 = ['US100', 'JP225', 'HK50', 'GER40', 'UK100', 'XAUUSD', 'XAGUSD', 'WTI']
CRY = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
FILES = {'long': '_maxc15', 'short': '_short_hik', 'trend': '_maxc15_trend', 'trendlong': '_maxc15_trendlong'}
DIRS = [('rev-long', 'long'), ('rev-short', 'short'), ('trend-short', 'trend'), ('trend-long', 'trendlong')]


def cd_ms(cd):
    return int((float(cd[:-1]) / 60 if cd.endswith('m') else float(cd[:-1])) * 3600 * 1000)


def trades(d, dkey, k1, b, cd):
    """(entry_ts, exit_ts, ret) 리스트. persym_sweep.run()과 동일 규칙(방향별)."""
    C, H, L, ts, ma, sd, z, n = d['C'], d['H'], d['L'], d['ts'], d['ma'], d['sd'], d['z'], d['n']
    cand = np.flatnonzero(z >= k1) if dkey in ('short', 'trendlong') else np.flatnonzero(z <= -k1)
    cand = cand[cand >= WIN]
    out = []; last = None
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd:
            continue
        price = C[ci]
        hi = int(np.searchsorted(ts, ts[ci] + MH, 'right'))
        if hi <= ci + 1:
            continue
        if dkey == 'short':
            tp = ma[ci] + b * sd[ci]
            if tp >= price: continue
            pct = 1 - tp / price; sl = price * (1 + pct)
            SLh = H[ci + 1:hi] >= sl; TPh = L[ci + 1:hi] <= tp; hit = SLh | TPh
            if hit.any():
                j = int(np.argmax(hit)); ex = ci + 1 + j
                ret = (price - sl) / price - FEE if SLh[j] else (price - tp) / price - FEE
            else:
                ex = hi - 1; ret = (price - C[ex]) / price - FEE
        elif dkey == 'trend':
            tp = ma[ci] - b * sd[ci]
            if tp <= price: continue
            pct = tp / price - 1; tpS = price * (1 - pct); slS = price * (1 + pct)
            SLh = H[ci + 1:hi] >= slS; TPh = L[ci + 1:hi] <= tpS; hit = SLh | TPh
            if hit.any():
                j = int(np.argmax(hit)); ex = ci + 1 + j
                ret = (price - slS) / price - FEE if SLh[j] else (price - tpS) / price - FEE
            else:
                ex = hi - 1; ret = (price - C[ex]) / price - FEE
        elif dkey == 'trendlong':
            tp = ma[ci] + b * sd[ci]
            if tp >= price: continue
            pct = 1 - tp / price; tpL = price * (1 + pct); slL = price * (1 - pct)
            SLh = L[ci + 1:hi] <= slL; TPh = H[ci + 1:hi] >= tpL; hit = SLh | TPh
            if hit.any():
                j = int(np.argmax(hit)); ex = ci + 1 + j
                ret = slL / price - 1 - FEE if SLh[j] else tpL / price - 1 - FEE
            else:
                ex = hi - 1; ret = C[ex] / price - 1 - FEE
        else:  # long
            tp = ma[ci] - b * sd[ci]
            if tp <= price: continue
            pct = tp / price - 1; sl = price * (1 - pct)
            SLh = L[ci + 1:hi] <= sl; TPh = H[ci + 1:hi] >= tp; hit = SLh | TPh
            if hit.any():
                j = int(np.argmax(hit)); ex = ci + 1 + j
                ret = sl / price - 1 - FEE if SLh[j] else tp / price - 1 - FEE
            else:
                ex = hi - 1; ret = C[ex] / price - 1 - FEE
        out.append((int(ts[ci]), int(ts[ex]), float(ret)))
        last = ts[ci]
    return out


def main():
    # 1) 채택 서브전략 수집
    picks = []
    for dname, dkey in DIRS:
        for mkt, syms in [('mt5', MT5), ('crypto', CRY)]:
            D = json.load(open(os.path.join(RES, f'{mkt}_14d{FILES[dkey]}.json'), encoding='utf-8'))
            for s in syms:
                if s in D and D[s] and D[s][0]['exp'] > 0:
                    r = D[s][0]
                    picks.append(dict(dname=dname, dkey=dkey, sym=s, k1=r['k1'], b=r['b'], cd=r['cd'],
                                      exp=r['exp'], pf=r['pf'], n=r['n']))
    print(f'채택 서브전략 {len(picks)}개')

    # 2) 거래 생성
    prep_cache = {}
    alltr = []  # (entry, exit, ret, dname, sym)
    for p in picks:
        d = prep_cache.get(p['sym']) or prep(p['sym'])
        prep_cache[p['sym']] = d
        tr = trades(d, p['dkey'], p['k1'], p['b'], cd_ms(p['cd']))
        for en, ex, ret in tr:
            alltr.append((en, ex, ret, p['dname'], p['sym']))
        p['ntr'] = len(tr)
    print(f'총 거래 {len(alltr):,}건')

    # 3) 포트폴리오 시뮬 (이벤트=진입순, 청산은 힙으로 지연실현)
    alltr.sort(key=lambda x: x[0])
    equity = INIT
    heap = []           # (exit_ts, ret, notional, dname)
    open_cnt = 0; peak_lots = 0; skipped = 0
    curve = []          # (ts, equity)  청산이벤트마다
    by_dir = {d[0]: 0.0 for d in DIRS}
    taken = 0
    span0 = min(t[0] for t in alltr); span1 = max(t[1] for t in alltr)

    def realize_until(t):
        nonlocal equity, open_cnt
        while heap and heap[0][0] <= t:
            ex_ts, ret, notional, dn = heapq.heappop(heap)
            pnl = notional * ret
            equity += pnl; by_dir[dn] += pnl; open_cnt -= 1
            curve.append((ex_ts, equity))

    for en, ex, ret, dname, sym in alltr:
        realize_until(en)
        if open_cnt < MAX_LOTS and equity > 0:
            notional = LOT_FRAC * equity
            heapq.heappush(heap, (ex, ret, notional, dname))
            open_cnt += 1; taken += 1
            peak_lots = max(peak_lots, open_cnt)
        else:
            skipped += 1
    realize_until(span1 + 1)   # 잔여 청산

    # 4) 지표
    eq = np.array([e for _, e in curve], float)
    ts_c = np.array([t for t, _ in curve], np.int64)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = float(dd.min()) if len(dd) else 0.0
    mdd_at = int(ts_c[int(np.argmin(dd))]) if len(dd) else 0
    yrs = (span1 - span0) / (365.25 * DAY)
    final = float(equity)
    cagr = (final / INIT) ** (1 / yrs) - 1 if final > 0 else -1.0
    ruin = bool((eq <= 0).any())
    ruin_at = int(ts_c[int(np.argmax(eq <= 0))]) if ruin else 0

    def dts(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')

    # 월별 스냅샷(곡선 경량화)
    monthly = []
    seen = set()
    for t, e in curve:
        key = datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime('%Y-%m')
        if key not in seen:
            seen.add(key); monthly.append((dts(t), round(e, 2)))
    monthly.append((dts(span1), round(final, 2)))

    print(f"\n{'='*60}")
    print(f"기간 {dts(span0)} ~ {dts(span1)} ({yrs:.2f}년)")
    print(f"시작 ${INIT:,.0f} → 최종 ${final:,.0f}  ({final/INIT:.2f}배, CAGR {cagr*100:+.1f}%)")
    print(f"최대낙폭(실현) {mdd*100:.1f}% @ {dts(mdd_at)}")
    print(f"체결 랏 {taken:,} / 스킵(200캡) {skipped:,} / 최대동시 {peak_lots}랏")
    print(f"파산(자산≤0): {'예 @ '+dts(ruin_at) if ruin else '아니오'}")
    print(f"\n방향별 실현손익:")
    for dn in by_dir:
        print(f"  {dn:>12}: ${by_dir[dn]:>+13,.0f}")

    json.dump(dict(
        init=INIT, final=round(final, 2), mult=round(final / INIT, 3), cagr=round(cagr, 4),
        mdd=round(mdd, 4), mdd_at=dts(mdd_at), yrs=round(yrs, 2),
        taken=taken, skipped=skipped, peak_lots=peak_lots, ruin=ruin,
        span=[dts(span0), dts(span1)], lot_frac=LOT_FRAC, max_lots=MAX_LOTS,
        by_dir={k: round(v, 2) for k, v in by_dir.items()},
        monthly=monthly,
        picks=[{k: p[k] for k in ('dname', 'sym', 'k1', 'b', 'cd', 'exp', 'pf', 'n', 'ntr')} for p in picks],
    ), open(os.path.join(os.path.dirname(__file__), 'portfolio_sim_result.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n→ portfolio_sim_result.json')


if __name__ == '__main__':
    main()
